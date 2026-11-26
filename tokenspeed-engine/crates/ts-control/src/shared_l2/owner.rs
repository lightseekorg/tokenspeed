// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! The pool's owner: one process serving the cache index to every client.
//!
//! It runs the scheduler's `SharedL2Pool` over every provider's memory and
//! holds no bytes. The pool is built when every expected provider has joined
//! and the first client reports its cache geometry; later clients must match
//! that geometry and the deployment the pool caches for. A provider joins
//! with its bytes; its share of the pool, in LCM blocks, is settled when the
//! pool is built, from the first client's LCM block. Every announcement is
//! one provider: one that is started again counts as one more, or is refused
//! once every expected provider has joined. A client silent for the client
//! timeout loses its pins and reservations, and its next request is answered
//! with an error.
//!
//! [`Owner::serve`] waits for a request, gives its payload to
//! `Owner::handle` and sends the reply behind the frames that came before
//! the payload; between requests it drops the clients silent for the client
//! timeout. `handle` reads the request's `op`: `add_provider` and `hello`
//! change the provider table, build the pool at the first hello after every
//! provider joined, and add the client; `stats` and `clear` need only the
//! pool; every other op needs a client that joined, refreshes its last-heard
//! time and makes one call into the pool. Any failure (a request that does
//! not decode, an error of the pool, a refused hello) is that request's
//! reply, `{"ok": false, "error": ...}`. What the request changed before it
//! failed stays: its client's last-heard time, the pins of the lookups
//! before a failed one (the client is never told their tickets; they go
//! when it is dropped), the keys a reserve refreshed, and the pool that a
//! first hello built before it failed to read its `client`.
//!
//! Requests ([`wire`]), by `op`; every reply also carries `ok`:
//!
//! * `add_provider`: `session` (the provider's `host:port`), `base_address`,
//!   `bytes`.
//! * `hello`: `client`, `geometry`, `deployment_id` -> `ready`, and once
//!   ready `providers`, `lcm_block_bytes`.
//! * `ping`: `client`.
//! * `lookup`: `client`, `lookups` `[[prefix_hashes, tokens, max_tokens]]`
//!   (`tokens` are the prompt's token ids, not a count) -> `hits`
//!   `[[ticket, num_tokens, kv_tokens, tail, [[key, bucket, page]]]]`, one
//!   per lookup. Each hit is pinned under its ticket; ticket 0 pins nothing.
//! * `release`: `client`, `ticket`.
//! * `reserve`: `client`, `keys`, `buckets` -> `ticket`, `pages` (one per
//!   key; negative: no page for the key now, -1 the pool holds it, -2 no
//!   page is free, -3 another reservation writes it).
//! * `commit`: `client`, `ticket`, `checkpoints` `[[tokens, prefix_hashes,
//!   tail, access_epoch]]` (`tokens` are a checkpoint's token ids, not a
//!   count).
//! * `touch`: `client`, `keys`.
//! * `clear` -> `cleared`; `stats` -> `stats`, `providers`, `num_clients`.
//!
//! The geometry is the first client's, kept as the msgpack value it arrived
//! as: a later client's must be that value, map order included (every rank
//! builds it with the same code, `cache/shared_l2/layout.py`), and the
//! pool's builder is handed its bytes.

use std::{
    collections::HashMap,
    error::Error,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use bytes::Bytes;
use tokio::runtime::Runtime;
use ts_scheduler::SharedL2Pool;
use ts_wire::{encode_frames, Value, Wire};
use zeromq::{RouterSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

use crate::{
    kv_events::bind_address,
    shared_l2::{geometry::Provider, wire},
};

/// How long [`Owner::serve`] waits for a request before it looks at its stop
/// flag and the client sweep again.
const SOCKET_POLL: Duration = Duration::from_millis(50);

/// The owner's ROUTER socket and the runtime that drives it; dropping it
/// closes the socket.
pub struct OwnerSocket {
    /// Drives the socket while the owner waits for a request or sends a
    /// reply, and at no other time.
    runtime: Runtime,
    socket: RouterSocket,
    address: String,
}

impl OwnerSocket {
    /// Bind at `address`, a ZMQ address as libzmq reads it (`tcp://*:PORT`
    /// is every interface).
    pub fn bind(address: &str) -> Result<Self, ZmqError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let mut socket = RouterSocket::new();
        let address = runtime
            .block_on(socket.bind(&bind_address(address)))?
            .to_string();
        Ok(Self {
            runtime,
            socket,
            address,
        })
    }

    /// Where the socket is bound, with the port it got.
    pub fn address(&self) -> &str {
        &self.address
    }
}

/// Builds the pool from `(geometry, num_lcm_blocks)`: the first client's
/// geometry, as its msgpack bytes, and the LCM blocks of every provider
/// together.
type MakePool = dyn FnMut(&[u8], i32) -> Result<SharedL2Pool, String> + Send;

/// A provider as it announced itself. Its bytes become LCM blocks when the
/// pool is built.
struct AnnouncedProvider {
    session: String,
    base_address: u64,
    bytes: u64,
}

/// The pool, with what its first client settled and with its clients.
struct BuiltPool {
    pool: SharedL2Pool,
    /// Each provider's share of the pool: its LCM blocks and where they
    /// start.
    providers: Vec<Provider>,
    geometry: Value,
    lcm_block_bytes: u64,
    deployment_id: String,
    /// Every client that joined the pool, and when it was last heard.
    clients: HashMap<u64, Instant>,
}

/// The pool's owner: the providers that joined, the pool once it is built,
/// and the clients that joined it.
pub struct Owner {
    /// Providers to wait for before the pool is built.
    num_providers: usize,
    /// Silence after which a client is dropped.
    client_timeout: Duration,
    /// Called at a hello once every provider has joined, until it has built
    /// the pool.
    make_pool: Box<MakePool>,
    announced_providers: Vec<AnnouncedProvider>,
    built_pool: Option<BuiltPool>,
}

impl Owner {
    /// An owner that waits for `num_providers` providers, drops a client
    /// silent for `client_timeout`, and builds its pool with `make_pool`
    /// from the first client's geometry (its msgpack bytes) and the LCM
    /// blocks of every provider together.
    pub fn new(
        num_providers: usize,
        client_timeout: Duration,
        make_pool: impl FnMut(&[u8], i32) -> Result<SharedL2Pool, String> + Send + 'static,
    ) -> Self {
        Self {
            num_providers,
            client_timeout,
            make_pool: Box::new(make_pool),
            announced_providers: Vec::new(),
            built_pool: None,
        }
    }

    /// Serve the requests that reach `socket` until `stop` is set, and drop
    /// the clients that fell silent between them.
    pub fn serve(&mut self, socket: &mut OwnerSocket, stop: &AtomicBool) {
        let OwnerSocket {
            runtime, socket, ..
        } = socket;
        let mut next_sweep = Instant::now() + self.client_timeout / 4;
        while !stop.load(Ordering::Relaxed) {
            match runtime.block_on(async { tokio::time::timeout(SOCKET_POLL, socket.recv()).await })
            {
                Ok(Ok(request)) => {
                    // The envelope (the identity, then a REQ's request id and
                    // empty delimiter or a rank's correlation frame) goes
                    // back unchanged.
                    let mut frames = request.into_vec();
                    let payload = frames.pop().expect("a message has frames");
                    frames.push(Bytes::from(self.handle(payload, Instant::now())));
                    let reply = ZmqMessage::try_from(frames).expect("a message has frames");
                    // A client that left gets no reply; the others are
                    // served on.
                    if let Err(error) = runtime.block_on(socket.send(reply)) {
                        tracing::warn!(%error, "a shared L2 reply was not sent");
                    }
                }
                Ok(Err(error)) => tracing::warn!(%error, "the shared L2 owner's socket failed"),
                // No request within the wait.
                Err(_) => {}
            }
            let now = Instant::now();
            if now >= next_sweep {
                next_sweep = now + self.client_timeout / 4;
                self.disconnect_timed_out_clients(now);
            }
        }
    }

    /// Drop every client last heard more than the client timeout before
    /// `now`, with what it holds in the pool.
    fn disconnect_timed_out_clients(&mut self, now: Instant) {
        let Some(built) = &mut self.built_pool else {
            return;
        };
        built.clients.retain(|client, heard| {
            if now.duration_since(*heard) <= self.client_timeout {
                return true;
            }
            tracing::warn!(client, "shared L2 client timed out");
            built.pool.disconnect(*client);
            false
        });
    }

    /// The reply to the request `payload`, heard at `now`.
    fn handle(&mut self, payload: Bytes, now: Instant) -> Vec<u8> {
        self.try_handle(&[payload], now).unwrap_or_else(|error| {
            tracing::error!(%error, "shared L2 owner request failed");
            wire::refused_reply(&error.to_string())
        })
    }

    fn try_handle(&mut self, request: &[Bytes], now: Instant) -> Result<Vec<u8>, Box<dyn Error>> {
        let op = wire::read_field(request, "op", String::decode)?;
        match op.as_str() {
            "add_provider" => return self.add_provider(request),
            "hello" => return self.hello(request, now),
            "clear" | "stats" => {
                let Some(built) = &mut self.built_pool else {
                    return Err("the shared L2 pool is not ready".into());
                };
                if op == "clear" {
                    let cleared = built.pool.clear_cache();
                    return Ok(wire::ok_reply(1, |encoder| {
                        encoder.str("cleared");
                        encoder.bool(cleared);
                    }));
                }
                let stats = built.pool.stats();
                return Ok(wire::ok_reply(3, |encoder| {
                    encoder.str("stats");
                    encoder.map_len(8);
                    encoder.str("cached_blocks");
                    stats.cached_blocks.encode(encoder);
                    encoder.str("checkpoints");
                    (stats.checkpoints as u64).encode(encoder);
                    encoder.str("empty_lcm_blocks");
                    stats.empty_lcm_blocks.encode(encoder);
                    encoder.str("evicted_blocks");
                    stats.evicted_blocks.encode(encoder);
                    encoder.str("num_lcm_blocks");
                    stats.num_lcm_blocks.encode(encoder);
                    encoder.str("open_lookups");
                    (stats.open_lookups as u64).encode(encoder);
                    encoder.str("pinned_blocks");
                    stats.pinned_blocks.encode(encoder);
                    encoder.str("reserved_blocks");
                    (stats.reserved_blocks as u64).encode(encoder);
                    encoder.str("providers");
                    wire::encode_providers(encoder, &built.providers);
                    encoder.str("num_clients");
                    (built.clients.len() as u64).encode(encoder);
                }));
            }
            _ => {}
        }
        let client = wire::read_field(request, "client", u64::decode)?;
        let Some((heard, pool)) = self
            .built_pool
            .as_mut()
            .and_then(|built| Some((built.clients.get_mut(&client)?, &mut built.pool)))
        else {
            // It timed out here, or this owner started after its hello: what
            // it held is gone, and it must not go on as if it were not. The
            // refusal goes through `Ok`: it is expected, and so not logged
            // as a failed request.
            return Ok(wire::refused_reply(&format!(
                "client {client} has not joined the pool"
            )));
        };
        *heard = now;
        match op.as_str() {
            "ping" => {}
            "lookup" => {
                let hits = wire::read_field(request, "lookups", wire::decode_lookups)?
                    .iter()
                    .map(|lookup| pool.lookup(client, lookup))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(wire::ok_reply(1, |encoder| {
                    encoder.str("hits");
                    encoder.array_len(hits.len());
                    for hit in &hits {
                        wire::encode_answer(encoder, hit);
                    }
                }));
            }
            "release" => pool.release(client, wire::read_field(request, "ticket", u64::decode)?)?,
            "reserve" => {
                let reservation = pool.reserve(
                    client,
                    &wire::read_field(request, "keys", wire::decode_keys)?,
                    &wire::read_field(request, "buckets", Vec::<i32>::decode)?,
                )?;
                return Ok(wire::ok_reply(2, |encoder| {
                    encoder.str("ticket");
                    reservation.ticket.encode(encoder);
                    encoder.str("pages");
                    reservation.pages.encode(encoder);
                }));
            }
            "commit" => pool.commit(
                client,
                wire::read_field(request, "ticket", u64::decode)?,
                &wire::read_field(request, "checkpoints", wire::decode_checkpoints)?,
            )?,
            "touch" => pool.touch(&wire::read_field(request, "keys", wire::decode_keys)?)?,
            _ => return Err(format!("unknown shared L2 request '{op}'").into()),
        }
        Ok(wire::ok_reply(0, |_| ()))
    }

    fn add_provider(&mut self, request: &[Bytes]) -> Result<Vec<u8>, Box<dyn Error>> {
        if self.announced_providers.len() == self.num_providers {
            return Err("every expected provider has already joined".into());
        }
        let provider = AnnouncedProvider {
            session: wire::read_field(request, "session", String::decode)?,
            base_address: wire::read_field(request, "base_address", u64::decode)?,
            bytes: wire::read_field(request, "bytes", u64::decode)?,
        };
        tracing::info!(
            session = %provider.session,
            bytes = provider.bytes,
            "shared L2 provider joined"
        );
        self.announced_providers.push(provider);
        Ok(wire::ok_reply(0, |_| ()))
    }

    fn hello(&mut self, request: &[Bytes], now: Instant) -> Result<Vec<u8>, Box<dyn Error>> {
        if self.announced_providers.len() < self.num_providers {
            return Ok(wire::ok_reply(1, |encoder| {
                encoder.str("ready");
                encoder.bool(false);
            }));
        }
        let geometry = wire::read_field(request, "geometry", Value::decode)?;
        let deployment_id = wire::read_field(request, "deployment_id", String::decode)?;
        let built = match &mut self.built_pool {
            Some(built) if built.geometry != geometry => {
                return Err("the client's cache geometry differs from the pool's".into())
            }
            // Same layout, other model, weights or runtime: its KV is not
            // this pool's.
            Some(built) if built.deployment_id != deployment_id => {
                return Err("the client's deployment differs from the pool's".into())
            }
            Some(built) => built,
            None => {
                let geometry_frames = encode_frames(&geometry);
                let lcm_block_bytes =
                    wire::read_field(&geometry_frames, "lcm_block_bytes", u64::decode)?;
                let mut providers = Vec::with_capacity(self.announced_providers.len());
                let mut num_lcm_blocks = 0;
                for provider in &self.announced_providers {
                    let provider_lcm_blocks =
                        provider.bytes.checked_div(lcm_block_bytes).unwrap_or(0);
                    if provider_lcm_blocks == 0 {
                        return Err(format!(
                            "provider {} holds {} bytes, less than one LCM block of \
                             {lcm_block_bytes} bytes",
                            provider.session, provider.bytes
                        )
                        .into());
                    }
                    providers.push(Provider {
                        session: provider.session.clone(),
                        base_address: provider.base_address,
                        num_lcm_blocks: provider_lcm_blocks,
                        first_lcm_block: num_lcm_blocks,
                    });
                    num_lcm_blocks += provider_lcm_blocks;
                }
                // The pool counts its LCM blocks in int32.
                let num_lcm_blocks = i32::try_from(num_lcm_blocks).map_err(|_| {
                    format!(
                        "the providers hold {num_lcm_blocks} LCM blocks of {lcm_block_bytes} \
                         bytes, more than a pool's {}",
                        i32::MAX
                    )
                })?;
                let pool = (self.make_pool)(&geometry_frames[0], num_lcm_blocks)?;
                tracing::info!(
                    %deployment_id,
                    stats = ?pool.stats(),
                    "shared L2 pool ready"
                );
                self.built_pool.insert(BuiltPool {
                    pool,
                    providers,
                    geometry,
                    lcm_block_bytes,
                    deployment_id,
                    clients: HashMap::new(),
                })
            }
        };
        built
            .clients
            .insert(wire::read_field(request, "client", u64::decode)?, now);
        Ok(wire::ok_reply(3, |encoder| {
            encoder.str("ready");
            encoder.bool(true);
            encoder.str("providers");
            wire::encode_providers(encoder, &built.providers);
            encoder.str("lcm_block_bytes");
            built.lcm_block_bytes.encode(encoder);
        }))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/shared_l2_owner.rs"]
mod tests;
