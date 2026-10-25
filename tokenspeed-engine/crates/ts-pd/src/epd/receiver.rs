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
    StartError, Transport,
};

/// How long a bootstrap-server lookup may take.
const LOOKUP_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a failed item's slot waits before another receive: longer than
/// any write the encode node may still have in flight.
const QUARANTINE: Duration = Duration::from_secs(10);

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
#[derive(Debug, Clone, Copy)]
struct Parallel {
    tp_size: i64,
    dp_size: i64,
}

/// One item's room as the threads see it.
#[derive(Debug, Clone)]
struct Room {
    status: TransferPoll,
    /// The encode rank's control address, once connected.
    target: Option<(String, u16)>,
    /// The prefill ranks its encode rank serves.
    fanout: i64,
    failure: Option<String>,
}

#[derive(Default)]
struct State {
    rooms: HashMap<u64, Room>,
    /// Bootstrap address -> its node's parallel sizes.
    parallel: HashMap<String, Parallel>,
    /// (bootstrap address, DP group, encode rank) -> the encode rank's
    /// control address; registered with once.
    connections: HashMap<(String, i64, i64), (String, u16)>,
}

impl State {
    fn fail(&mut self, room: u64, reason: impl Into<String>) {
        if let Some(entry) = self.rooms.get_mut(&room) {
            entry.status = TransferPoll::Failed;
            entry.failure.get_or_insert_with(|| reason.into());
        }
    }
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
        if let Some(parallel) = self.state().parallel.get(addr).copied() {
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
        self.state().parallel.insert(addr.to_owned(), parallel);
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
        if let Some(target) = self.state().connections.get(&key).cloned() {
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
        self.sockets.send(
            &host,
            port,
            registration_frames(
                &self.transport.local_ip,
                self.sockets.port(),
                &self.transport.session_id(),
                0,
                0,
            ),
        );
        self.state().connections.insert(key, (host.clone(), port));
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
    connect: mpsc::Sender<(u64, String)>,
    stop: Arc<AtomicBool>,
    config: ReceiverConfig,
    pool: Option<SlotPool>,
    staged: Vec<Staged>,
    /// Landed slots whose rows the data plane has yet to copy out.
    awaiting: HashSet<usize>,
    timeout_logged: bool,
}

impl Receiver {
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
        let (connect, rooms) = mpsc::channel::<(u64, String)>();
        {
            let shared = shared.clone();
            thread::Builder::new()
                .name("ts-epd-connect".to_owned())
                .spawn(move || {
                    while let Ok((room, addr)) = rooms.recv() {
                        match shared.connect(room, &addr) {
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
                self.shared.sockets.send(host, *port, info.to_frames());
                item.preallocated = true;
            }
        }
        room.status
    }

    /// Return a settled request's slots: a landed one's await their copy, a
    /// failed one's sit out the quarantine.
    fn forget(&mut self, staged: &Staged, landed: bool, now: Instant) {
        let mut state = self.shared.state();
        for item in &staged.items {
            state.rooms.remove(&item.item.room);
        }
        drop(state);
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
                    fanout: 0,
                    failure: None,
                },
            );
            if self
                .connect
                .send((item.room, join_host_port(&item.host, item.port)))
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
