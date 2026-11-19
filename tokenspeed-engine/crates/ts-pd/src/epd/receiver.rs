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

//! A prefill rank's receiver of the encode nodes' embeddings.
//!
//! Threads: one connects items to their encode ranks (the bootstrap
//! server's HTTP lookups, one registration per encode rank), one applies
//! the encode ranks' statuses. The scheduling loop stages requests, leases
//! each item a slot of the receive pool, sends its pre-allocation once both
//! are ready, and polls. A landed item's rows stay in its slot until the
//! data plane copied them onto the item; a failed item's slot sits out a
//! quarantine first.
//!
//! The connect thread's cached resolutions (a node's parallel sizes, an
//! encode rank's control address) carry a TTL and are dropped when a room
//! through them fails: an encode node restart binds new control ports, and
//! the next room must resolve it again from the bootstrap server.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex, MutexGuard,
    },
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;

use super::{parse_status, pool::SlotPool, registration_frames, shard_rows, TransferInfo};
use crate::{
    control::{EmbeddingItem, EmbeddingReceiver, LandedItem, LandedRequest},
    http, net,
    status::TransferPoll,
    PeerConfig, StartError, Transport,
};

/// How long a bootstrap-server lookup may take.
const LOOKUP_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a failed item's slot waits before another receive: longer than
/// any write the encode node may still have in flight.
const QUARANTINE: Duration = Duration::from_secs(10);

/// How long a cached bootstrap resolution (a node's parallel sizes, an
/// encode rank's control address) stays trusted: an encode node restart
/// binds new control ports, so a resolution older than this is fetched
/// again from the bootstrap server even without a failure.
const RESOLUTION_TTL: Duration = Duration::from_secs(60);

/// How often an idle socket thread checks for shutdown.
const SOCKET_POLL: Duration = Duration::from_millis(100);

/// One receive pool: a device buffer of equal slots, registered for the
/// engine's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolRegion {
    /// The buffer's base.
    pub base: u64,
    /// A slot's bytes.
    pub slot_bytes: u64,
    /// Its slots.
    pub slots: usize,
}

/// What a receiver needs of the rank and its model.
#[derive(Debug, Clone)]
pub struct ReceiverConfig {
    /// The prefill node's attention TP size.
    pub tp_size: i64,
    /// This rank's attention TP rank.
    pub tp_rank: i64,
    /// Elements per embedding row.
    pub hidden: i64,
    /// Deepstack levels per row; 0 for a model without deepstack.
    pub num_deepstack: i64,
    /// The rows' dtype (`str(torch.dtype)`).
    pub dtype: String,
    /// Bytes per element.
    pub element_size: i64,
    /// Each rank receives only its share of an image's rows.
    pub shard: bool,
    /// The receive pool; `None` when disabled, which fails every item.
    pub pool: Option<PoolRegion>,
    /// The deepstack rows' pool, with the main pool's slot count; for a
    /// model with deepstack.
    pub deepstack_pool: Option<PoolRegion>,
    /// How long a staged request may wait for its embeddings.
    pub timeout: Duration,
}

/// An encode node's parallel sizes, from its bootstrap server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Parallel {
    tp_size: i64,
    dp_size: i64,
}

/// A connections-cache key: the bootstrap address, the DP group and the
/// encode rank.
type EncodeRankKey = (String, i64, i64);

/// An encode rank's cached control address, with when it was resolved.
struct Resolution {
    target: (String, u16),
    resolved_at: Instant,
}

/// One item's room as the threads see it.
#[derive(Debug, Clone)]
struct Room {
    status: TransferPoll,
    /// The encode rank's control address, once connected.
    target: Option<(String, u16)>,
    /// The connections-cache key this room resolved through, for the
    /// failure eviction.
    key: Option<EncodeRankKey>,
    /// The prefill ranks its encode rank serves.
    fanout: i64,
    failure: Option<String>,
}

#[derive(Default)]
struct State {
    rooms: HashMap<u64, Room>,
    /// Bootstrap address -> its node's parallel sizes, with the time they
    /// were fetched; fetched again past RESOLUTION_TTL.
    parallel: HashMap<String, (Parallel, Instant)>,
    /// Encode rank -> its control address; re-resolved past
    /// RESOLUTION_TTL, and dropped when a room through it fails.
    connections: HashMap<EncodeRankKey, Resolution>,
}

impl State {
    fn fail(&mut self, room: u64, reason: impl Into<String>) {
        if let Some(entry) = self.rooms.get_mut(&room) {
            entry.status = TransferPoll::Failed;
            entry.failure.get_or_insert_with(|| reason.into());
        }
    }

    /// The cached parallel sizes when still fresh, None otherwise.
    fn cached_parallel(&self, addr: &str) -> Option<Parallel> {
        let (parallel, at) = self.parallel.get(addr)?;
        (at.elapsed() < RESOLUTION_TTL).then_some(*parallel)
    }

    fn cache_parallel(&mut self, addr: &str, parallel: Parallel) {
        self.parallel
            .insert(addr.to_owned(), (parallel, Instant::now()));
    }

    /// The cached control address of an encode rank when still fresh, None
    /// otherwise.
    fn cached_connection(&self, key: &EncodeRankKey) -> Option<(String, u16)> {
        let resolution = self.connections.get(key)?;
        (resolution.resolved_at.elapsed() < RESOLUTION_TTL).then_some(resolution.target.clone())
    }

    fn cache_connection(&mut self, key: EncodeRankKey, target: (String, u16)) {
        self.connections.insert(
            key,
            Resolution {
                target,
                resolved_at: Instant::now(),
            },
        );
    }

    /// The connections-cache key a room resolves through, so a failure can
    /// evict it. A no-op once the room is gone.
    fn record_key(&mut self, room: u64, key: EncodeRankKey) {
        if let Some(entry) = self.rooms.get_mut(&room) {
            entry.key = Some(key);
        }
    }

    /// Drop the resolution a failed room went through: the encode node may
    /// have restarted onto new ports, and the next room must resolve it
    /// again from the bootstrap server.
    fn evict(&mut self, key: &EncodeRankKey) {
        self.connections.remove(key);
        self.parallel.remove(&key.0);
    }
}

/// What the receiver asks its connect thread to do.
enum Connect {
    /// Connect a room to its encode rank through a bootstrap address.
    Room(u64, String),
    /// Drop the cached resolution a failed room went through.
    Evict(EncodeRankKey),
}

struct Shared {
    transport: Arc<Transport>,
    sockets: net::Sockets,
    tp_size: i64,
    tp_rank: i64,
    state: Mutex<State>,
}

/// `host:port`, with an IPv6 host in brackets.
fn join_host_port(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

fn integer(value: &serde_json::Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|text| text.trim().parse().ok()))
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("the EPD state lock is poisoned")
    }

    /// The encode node's parallel sizes, fetched once a complete answer
    /// comes (`/route` may answer before the encode ranks registered).
    fn parallel(&self, addr: &str) -> Result<Parallel, String> {
        if let Some(parallel) = self.state().cached_parallel(addr) {
            return Ok(parallel);
        }
        let incomplete = || "no (complete) parallel info from bootstrap".to_owned();
        let body = http::get_text(
            &format!("http://{addr}/route?engine_rank=-1&target_dp_group=-1"),
            LOOKUP_TIMEOUT,
        )
        .map_err(|error| {
            tracing::error!(addr, error, "EPD bootstrap /route fetch failed");
            incomplete()
        })?;
        let info: serde_json::Value = serde_json::from_str(&body).map_err(|_| incomplete())?;
        let parallel = Parallel {
            tp_size: integer(&info["prefill_tp_size"]).ok_or_else(incomplete)?,
            dp_size: integer(&info["prefill_dp_size"]).ok_or_else(incomplete)?,
        };
        if parallel.tp_size <= 0 || parallel.dp_size <= 0 {
            return Err(incomplete());
        }
        self.state().cache_parallel(addr, parallel);
        Ok(parallel)
    }

    /// Connect `room` to the encode rank this rank pulls from: contiguous
    /// blocks of `fanout` prefill ranks share one encode rank. The first
    /// connection to an encode rank registers this rank with it.
    fn connect(&self, room: u64, addr: &str) -> Result<((String, u16), i64), String> {
        let parallel = self.parallel(addr)?;
        if self.tp_size % parallel.tp_size != 0 {
            return Err(format!(
                "EPD requires prefill_tp to be a multiple of encode_tp (encode_tp={}, \
                 prefill_tp={})",
                parallel.tp_size, self.tp_size
            ));
        }
        let fanout = self.tp_size / parallel.tp_size;
        let encode_rank = self.tp_rank / fanout;
        let dp_group = (room % parallel.dp_size as u64) as i64;
        let key = (addr.to_owned(), dp_group, encode_rank);
        // The failure eviction drops this key when the room fails, so record
        // it whether the cache answers or the bootstrap does.
        self.state().record_key(room, key.clone());
        if let Some(target) = self.state().cached_connection(&key) {
            return Ok((target, fanout));
        }
        let missing = || "no encode rank info from bootstrap".to_owned();
        let body = http::get_text(
            &format!("http://{addr}/route?engine_rank={encode_rank}&target_dp_group={dp_group}"),
            LOOKUP_TIMEOUT,
        )
        .map_err(|error| {
            tracing::error!(addr, error, "EPD bootstrap /route fetch failed");
            missing()
        })?;
        let info: serde_json::Value = serde_json::from_str(&body).map_err(|_| missing())?;
        let host = info["rank_ip"].as_str().ok_or_else(missing)?.to_owned();
        let port = integer(&info["rank_port"])
            .and_then(|port| u16::try_from(port).ok())
            .ok_or_else(missing)?;
        // The encode rank pushes this rank's statuses to the registered
        // socket; the pointers are informational.
        self.sockets
            .send(
                &host,
                port,
                registration_frames(
                    &self.transport.local_ip,
                    self.sockets.port(),
                    &self.transport.session_id(),
                    0,
                    0,
                ),
            )
            .map_err(|error| error.to_string())?;
        self.state().cache_connection(key, (host.clone(), port));
        Ok(((host, port), fanout))
    }

    /// Apply one encode rank's status of a room this rank tracks: a rank
    /// pulls from exactly one encode rank, so its success lands the room.
    fn apply(&self, frames: &[Bytes]) {
        let (room, status, _rank) = match parse_status(frames) {
            Ok(status) => status,
            Err(error) => {
                tracing::error!(%error, "rejecting a malformed EPD status");
                return;
            }
        };
        let mut state = self.state();
        match status {
            TransferPoll::Success => {
                if let Some(entry) = state.rooms.get_mut(&room) {
                    if entry.status != TransferPoll::Failed {
                        entry.status = TransferPoll::Success;
                    }
                }
            }
            TransferPoll::Failed => state.fail(room, "encode failed to send embedding"),
            _ => {}
        }
    }
}

/// One item of a staged request.
#[derive(Debug)]
struct StagedItem {
    item: EmbeddingItem,
    row_start: i64,
    row_count: i64,
    sharded: bool,
    slot: Option<usize>,
    preallocated: bool,
    failure: Option<String>,
}

#[derive(Debug)]
struct Staged {
    rid: String,
    since: Instant,
    items: Vec<StagedItem>,
}

/// A prefill rank's receiver of the encode nodes' embeddings.
pub struct Receiver {
    shared: Arc<Shared>,
    connect: mpsc::Sender<Connect>,
    stop: Arc<AtomicBool>,
    config: ReceiverConfig,
    pool: Option<SlotPool>,
    staged: Vec<Staged>,
    /// Landed slots whose rows the data plane has yet to copy out.
    awaiting: HashSet<usize>,
    timeout_logged: bool,
}

impl Receiver {
    /// Receive encoder embeddings without a P/D KV-transfer peer.
    pub fn start_standalone(config: ReceiverConfig, peer: PeerConfig) -> Result<Self, StartError> {
        let (incoming, _unused) = mpsc::channel();
        let transport = Transport::start(&peer, incoming)?;
        Self::start(config, transport)
    }

    /// Register the pools with `transport`'s engine, open the status socket
    /// and start the threads.
    pub(crate) fn start(
        config: ReceiverConfig,
        transport: Arc<Transport>,
    ) -> Result<Self, StartError> {
        for region in config.pool.iter().chain(&config.deepstack_pool) {
            let engine = transport.engine();
            engine.register(
                region.base as usize,
                region.slot_bytes as usize * region.slots,
            )?;
        }
        let (incoming, statuses) = mpsc::channel::<Vec<Bytes>>();
        let sockets = net::Sockets::start(&transport.local_ip, incoming)?;
        let shared = Arc::new(Shared {
            transport,
            sockets,
            tp_size: config.tp_size,
            tp_rank: config.tp_rank,
            state: Mutex::new(State::default()),
        });
        let stop = Arc::new(AtomicBool::new(false));
        {
            let shared = shared.clone();
            let stop = stop.clone();
            thread::Builder::new()
                .name("ts-epd-status".to_owned())
                .spawn(move || loop {
                    match statuses.recv_timeout(SOCKET_POLL) {
                        Ok(frames) => shared.apply(&frames),
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if stop.load(Ordering::Relaxed) {
                                return;
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => return,
                    }
                })?;
        }
        let (connect, messages) = mpsc::channel::<Connect>();
        {
            let shared = shared.clone();
            thread::Builder::new()
                .name("ts-epd-connect".to_owned())
                .spawn(move || {
                    while let Ok(message) = messages.recv() {
                        match message {
                            Connect::Room(room, addr) => match shared.connect(room, &addr) {
                                Ok((target, fanout)) => {
                                    let mut state = shared.state();
                                    if let Some(entry) = state.rooms.get_mut(&room) {
                                        if entry.status == TransferPoll::Bootstrapping {
                                            entry.status = TransferPoll::Bootstrapped;
                                        }
                                        entry.target = Some(target);
                                        entry.fanout = fanout;
                                    }
                                }
                                Err(reason) => shared.state().fail(room, reason),
                            },
                            Connect::Evict(key) => shared.state().evict(&key),
                        }
                    }
                })?;
        }
        let pool = config
            .pool
            .map(|region| SlotPool::new(region.slots, region.slot_bytes));
        tracing::info!(
            pool = ?config.pool,
            deepstack_pool = ?config.deepstack_pool,
            port = shared.sockets.port(),
            "EPD embedding receiver up"
        );
        Ok(Self {
            shared,
            connect,
            stop,
            config,
            pool,
            staged: Vec::new(),
            awaiting: HashSet::new(),
            timeout_logged: false,
        })
    }

    /// The status socket's port.
    pub fn port(&self) -> u16 {
        self.shared.sockets.port()
    }

    /// Bytes of `rows` rows of the main and the deepstack pool.
    fn bytes(&self, rows: i64) -> (u64, u64) {
        let main = rows * self.config.hidden * self.config.element_size;
        (main as u64, (main * self.config.num_deepstack) as u64)
    }

    /// Advance one item: lease its slot, then send its pre-allocation once
    /// its encode rank is connected. Returns its room's state.
    fn advance(&mut self, request: usize, index: usize, now: Instant) -> TransferPoll {
        let (main_bytes, deep_bytes) = self.bytes(self.staged[request].items[index].row_count);
        let item = &mut self.staged[request].items[index];
        if item.failure.is_some() {
            return TransferPoll::Failed;
        }
        let room = self.shared.state().rooms.get(&item.item.room).cloned();
        let Some(room) = room else {
            return TransferPoll::Failed;
        };
        if room.status == TransferPoll::Failed {
            item.failure = room
                .failure
                .or_else(|| Some("the encode node failed".to_owned()));
            return TransferPoll::Failed;
        }
        if item.slot.is_none() && item.row_count > 0 {
            let Some(pool) = &mut self.pool else {
                item.failure = Some(
                    "EPD receive pool is disabled (TOKENSPEED_EPD_RECV_POOL_SLOTS=0)".to_owned(),
                );
                return TransferPoll::Failed;
            };
            let deep_slot = self
                .config
                .deepstack_pool
                .map_or(0, |region| region.slot_bytes);
            if deep_bytes > deep_slot && self.config.num_deepstack > 0 {
                item.failure = Some(format!(
                    "EPD item's {deep_bytes} deepstack bytes exceed the receive pool's \
                     {deep_slot}-byte slots; raise TOKENSPEED_EPD_RECV_POOL_SLOT_MB"
                ));
                return TransferPoll::Failed;
            }
            match pool.lease(main_bytes, now) {
                Ok(Some(slot)) => item.slot = Some(slot),
                // Every slot is taken: wait, within the request's deadline.
                Ok(None) => return room.status,
                Err(oversized) => {
                    item.failure = Some(format!(
                        "EPD item's {} bytes exceed the receive pool's {}-byte slots; raise \
                         TOKENSPEED_EPD_RECV_POOL_SLOT_MB",
                        oversized.bytes, oversized.slot_bytes
                    ));
                    return TransferPoll::Failed;
                }
            }
        }
        if !item.preallocated && room.status >= TransferPoll::Bootstrapped {
            if let Some((host, port)) = &room.target {
                let pointer = |region: Option<PoolRegion>| {
                    region.map_or(0, |region| {
                        region.base + item.slot.unwrap_or(0) as u64 * region.slot_bytes
                    })
                };
                let info = TransferInfo {
                    room: item.item.room,
                    endpoint: self.shared.transport.local_ip.clone(),
                    port: self.shared.sockets.port(),
                    session_id: self.shared.transport.session_id(),
                    embedding_ptr: pointer(self.config.pool),
                    deepstack_ptr: if self.config.num_deepstack > 0 {
                        pointer(self.config.deepstack_pool)
                    } else {
                        0
                    },
                    rows: item.row_count,
                    hidden: self.config.hidden,
                    dtype: self.config.dtype.clone(),
                    has_deepstack: self.config.num_deepstack > 0,
                    fanout: room.fanout,
                    row_start: item.row_start,
                    span: item.item.rows,
                };
                // An empty shard still asks: the encode rank counts its
                // block's requests, not their bytes.
                if let Err(error) = self.shared.sockets.send(host, *port, info.to_frames()) {
                    tracing::warn!(%error, "could not publish an EPD destination");
                    item.failure = Some(error.to_string());
                    self.shared.state().fail(item.item.room, error.to_string());
                    return TransferPoll::Failed;
                }
                item.preallocated = true;
            }
        }
        room.status
    }

    /// Return a settled request's slots: a landed one's await their copy, a
    /// failed one's sit out the quarantine. A failed request also evicts the
    /// resolution each of its rooms went through: the encode node may have
    /// restarted onto new ports, and the next room must resolve it again
    /// from the bootstrap server.
    fn forget(&mut self, staged: &Staged, landed: bool, now: Instant) {
        let mut state = self.shared.state();
        let mut failed_keys = Vec::new();
        for item in &staged.items {
            let entry = state.rooms.get(&item.item.room);
            if !landed {
                if let Some(key) = entry.and_then(|entry| entry.key.clone()) {
                    failed_keys.push(key);
                }
            }
            state.rooms.remove(&item.item.room);
        }
        drop(state);
        for key in failed_keys {
            let _ = self.connect.send(Connect::Evict(key));
        }
        for slot in staged.items.iter().filter_map(|item| item.slot) {
            if landed {
                self.awaiting.insert(slot);
            } else if let Some(pool) = &mut self.pool {
                pool.quarantine(slot, now + QUARANTINE);
            }
        }
    }
}

impl Drop for Receiver {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl EmbeddingReceiver for Receiver {
    fn stage(&mut self, rid: &str, items: &[EmbeddingItem]) {
        let sharded = self.config.shard && self.config.tp_size > 1;
        let mut staged_items = Vec::with_capacity(items.len());
        for item in items {
            let (row_start, row_count) = if sharded {
                shard_rows(item.rows, self.config.tp_rank, self.config.tp_size)
            } else {
                (0, item.rows)
            };
            self.shared.state().rooms.insert(
                item.room,
                Room {
                    status: TransferPoll::Bootstrapping,
                    target: None,
                    key: None,
                    fanout: 0,
                    failure: None,
                },
            );
            if self
                .connect
                .send(Connect::Room(
                    item.room,
                    join_host_port(&item.host, item.port),
                ))
                .is_err()
            {
                self.shared
                    .state()
                    .fail(item.room, "the EPD connect thread stopped");
            }
            staged_items.push(StagedItem {
                item: item.clone(),
                row_start,
                row_count,
                sharded,
                slot: None,
                preallocated: false,
                failure: None,
            });
        }
        self.staged.push(Staged {
            rid: rid.to_owned(),
            since: Instant::now(),
            items: staged_items,
        });
    }

    fn poll(&mut self) -> Vec<i64> {
        let now = Instant::now();
        let mut codes = Vec::with_capacity(self.staged.len());
        for request in 0..self.staged.len() {
            let mut landed = true;
            let mut failed = false;
            for index in 0..self.staged[request].items.len() {
                match self.advance(request, index, now) {
                    TransferPoll::Success => {}
                    TransferPoll::Failed => failed = true,
                    _ => landed = false,
                }
            }
            let staged = &self.staged[request];
            let code = if failed {
                0
            } else if landed {
                2
            } else if now.duration_since(staged.since) > self.config.timeout {
                if !self.timeout_logged {
                    self.timeout_logged = true;
                    tracing::warn!(
                        rid = staged.rid,
                        timeout = ?self.config.timeout,
                        "an EPD embedding receive timed out; raise \
                         TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT if a longer TTFT is acceptable"
                    );
                }
                0
            } else {
                1
            };
            codes.push(code);
        }
        codes
    }

    fn settle(&mut self, codes: &[i64]) -> Vec<LandedRequest> {
        let now = Instant::now();
        let mut landed = Vec::new();
        let mut leftover = Vec::new();
        for (staged, &code) in std::mem::take(&mut self.staged).into_iter().zip(codes) {
            match code {
                2 => {
                    self.forget(&staged, true, now);
                    landed.push(LandedRequest {
                        rid: staged.rid,
                        items: staged
                            .items
                            .into_iter()
                            .map(|item| LandedItem {
                                index: item.item.index,
                                slot: item.slot,
                                row_start: item.row_start,
                                row_count: item.row_count,
                                rows: item.item.rows,
                                sharded: item.sharded,
                            })
                            .collect(),
                    });
                }
                0 => {
                    let reasons: Vec<&str> = staged
                        .items
                        .iter()
                        .filter_map(|item| item.failure.as_deref())
                        .collect();
                    tracing::warn!(rid = staged.rid, ?reasons, "EPD embedding receive failed");
                    self.forget(&staged, false, now);
                }
                _ => leftover.push(staged),
            }
        }
        self.staged = leftover;
        landed
    }

    fn release(&mut self, slots: &[usize]) {
        for &slot in slots {
            if self.awaiting.remove(&slot) {
                if let Some(pool) = &mut self.pool {
                    pool.release(slot);
                }
            }
        }
    }

    fn awaiting_release(&self) -> bool {
        !self.awaiting.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parallel(tp_size: i64, dp_size: i64) -> Parallel {
        Parallel { tp_size, dp_size }
    }

    fn key(addr: &str, dp_group: i64, encode_rank: i64) -> EncodeRankKey {
        (addr.to_owned(), dp_group, encode_rank)
    }

    /// An instant older than the resolution TTL, as the tests' stale entries
    /// carry.
    fn stale() -> Instant {
        Instant::now()
            .checked_sub(RESOLUTION_TTL + Duration::from_secs(1))
            .unwrap()
    }

    #[test]
    fn cached_resolutions_expire() {
        let mut state = State::default();
        let addr = "10.0.0.9:8998";
        state.cache_parallel(addr, parallel(2, 1));
        assert_eq!(state.cached_parallel(addr), Some(parallel(2, 1)));
        // A stale entry reads as a miss, so the next room re-resolves.
        state
            .parallel
            .insert(addr.to_owned(), (parallel(2, 1), stale()));
        assert_eq!(state.cached_parallel(addr), None);

        let rank = key(addr, 0, 0);
        state.cache_connection(rank.clone(), ("10.0.0.9".to_owned(), 20001));
        assert_eq!(
            state.cached_connection(&rank),
            Some(("10.0.0.9".to_owned(), 20001))
        );
        state.connections.insert(
            rank.clone(),
            Resolution {
                target: ("10.0.0.9".to_owned(), 20001),
                resolved_at: stale(),
            },
        );
        assert_eq!(state.cached_connection(&rank), None);
    }

    #[test]
    fn an_eviction_drops_only_its_own_resolution() {
        let mut state = State::default();
        let node_a = key("10.0.0.9:8998", 0, 0);
        let node_b = key("10.0.0.8:8998", 1, 1);
        state.cache_parallel(&node_a.0, parallel(2, 1));
        state.cache_parallel(&node_b.0, parallel(4, 2));
        state.cache_connection(node_a.clone(), ("10.0.0.9".to_owned(), 20001));
        state.cache_connection(node_b.clone(), ("10.0.0.8".to_owned(), 20002));

        state.evict(&node_a);
        assert_eq!(state.cached_connection(&node_a), None);
        assert_eq!(state.cached_parallel(&node_a.0), None);
        // Another node's resolution survives.
        assert_eq!(
            state.cached_connection(&node_b),
            Some(("10.0.0.8".to_owned(), 20002))
        );
        assert_eq!(state.cached_parallel(&node_b.0), Some(parallel(4, 2)));
    }

    #[test]
    fn rooms_remember_the_key_their_failure_evicts() {
        let mut state = State::default();
        state.rooms.insert(
            7,
            Room {
                status: TransferPoll::Bootstrapping,
                target: None,
                key: None,
                fanout: 0,
                failure: None,
            },
        );
        let room = key("10.0.0.9:8998", 1, 2);
        state.record_key(7, room.clone());
        assert_eq!(state.rooms[&7].key, Some(room.clone()));
        // Evicting the key clears the caches but leaves the room alone; the
        // room's failure handler removed it already.
        state.cache_connection(room.clone(), ("10.0.0.9".to_owned(), 20001));
        state.cache_parallel(&room.0, parallel(2, 1));
        state.evict(&room);
        assert_eq!(state.cached_connection(&room), None);
        assert_eq!(state.cached_parallel(&room.0), None);
        assert_eq!(state.rooms[&7].key, Some(room));
        // Recording for a room that settled before its connect completed is
        // a no-op.
        state.record_key(99, key("10.0.0.9:8998", 0, 0));
        assert!(!state.rooms.contains_key(&99));
    }
}
