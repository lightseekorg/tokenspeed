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

//! A Decode rank's peer: it finds a request's Prefill ranks through the
//! bootstrap server, registers with them once per route, sends each
//! admitted request's page manifest, and collects their completions.
//!
//! Threads: one connects requests to their routes (the bootstrap server's
//! HTTP lookups), one handles the status socket, one watches the Prefill
//! nodes' health. The scheduling loop only registers requests and polls;
//! the forward thread hands over each plan's remote prefill
//! ([`DecodeOutbox`]). Routes and manifests are planned here, over the
//! rank's cache transfer contract ([`crate::layout`]).

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex, MutexGuard,
    },
    thread,
    time::{Duration, Instant},
};

use ts_collective::{Group, Mesh, Reduce};

use crate::{
    batch::{BatchRows, Staged},
    control::{BootstrapInfo, PdError, PdEvent, RemotePrefill, TransferPeer},
    http,
    layout::{
        contract::Contract,
        manifest::{build_manifest, validate_manifest},
        registration::{plan_route, PrefillRoute},
    },
    sleep_or_stop,
    status::{RoomTable, TransferPoll},
    wire::{self, StatusMessage},
    PeerConfig, StartError, Transport,
};

/// How often an idle socket thread checks for shutdown.
const SOCKET_POLL: Duration = Duration::from_millis(100);

/// How long a bootstrap-server lookup may take.
const LOOKUP_TIMEOUT: Duration = Duration::from_secs(5);

/// A Decode rank's settings.
#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// What both roles share.
    pub peer: PeerConfig,
    /// What this rank registers with a Prefill rank: its cache arena base,
    /// its cache transfer contract's wire bytes (with its landing
    /// addresses), and its attention TP size and rank.
    pub arena_base: u64,
    /// See `arena_base`.
    pub layout: Vec<u8>,
    /// See `arena_base`.
    pub tp_size: i64,
    /// See `arena_base`.
    pub tp_rank: i64,
    /// A request whose manifest went out this long ago without its KV
    /// landing fails.
    pub waiting_timeout: Duration,
    /// How often the Prefill nodes' health is checked.
    pub heartbeat_interval: Duration,
    /// Failed checks in a row after which a Prefill node counts as lost.
    pub heartbeat_max_failures: u32,
}

/// A Prefill rank a route reaches.
#[derive(Debug, Clone)]
struct Target {
    host: String,
    port: u16,
    is_dummy: bool,
}

/// The first valid value each Prefill rank reported for a room.
#[derive(Debug, Clone, Default)]
struct Landed {
    bootstrap_token: Option<i32>,
    spec_candidate_ids: Option<Vec<i32>>,
    bootstrap_logprob: Option<f64>,
}

/// What a route's registered targets are kept by: the bootstrap address,
/// the room's Prefill DP group and the target ranks.
type ConnectionKey = (String, u64, Vec<i64>);

#[derive(Default)]
struct State {
    rooms: RoomTable,
    /// The rooms of the requests the scheduling loop tracks. Only these
    /// keep state: a late report, connection or admission of a room the
    /// loop forgot must not bring it back.
    tracked: HashSet<u64>,
    /// The tracked requests' rooms, by request.
    rooms_by_rid: HashMap<String, u64>,
    expected: HashMap<u64, BTreeSet<i64>>,
    responses: HashMap<u64, BTreeSet<i64>>,
    pending: HashMap<u64, Landed>,
    landed: HashMap<u64, Landed>,
    cached_tokens: HashMap<u64, i64>,
    targets: HashMap<u64, Arc<Vec<Target>>>,
    admissions: HashMap<u64, i32>,
    waiting_since: HashMap<u64, Instant>,
    /// Bootstrap address -> the rooms reading from it.
    addr_rooms: HashMap<String, HashSet<u64>>,
    /// Bootstrap address -> its route; the addresses the heartbeat watches.
    routes: HashMap<String, Arc<PrefillRoute>>,
    connections: HashMap<ConnectionKey, Arc<Vec<Target>>>,
}

impl State {
    /// Fail a tracked room; an untracked one has nothing to fail.
    fn fail(&mut self, room: u64, reason: impl Into<String>) {
        if !self.tracked.contains(&room) {
            return;
        }
        self.rooms.record_failure(room, reason);
        self.rooms.update(room, TransferPoll::Failed);
    }

    /// Apply one Prefill rank's report.
    fn apply(&mut self, message: StatusMessage) {
        let room = message.room;
        if !self.tracked.contains(&room) {
            return;
        }
        match message.status {
            TransferPoll::Success => {
                let Some(expected) = self.expected.get(&room).cloned() else {
                    self.fail(
                        room,
                        "Received KV completion before its transfer route was registered",
                    );
                    return;
                };
                if !expected.contains(&message.prefill_rank) {
                    self.fail(
                        room,
                        "Received KV completion from an unexpected Prefill TP rank",
                    );
                    return;
                }
                let responses = self.responses.entry(room).or_default();
                responses.insert(message.prefill_rank);
                let arrived = responses.clone();
                // TP ranks describe overlapping prefixes, not disjoint ranges.
                let cached = self.cached_tokens.entry(room).or_insert(0);
                *cached = (*cached).max(message.cached_tokens);
                // The rank carrying the first token need not complete last;
                // -1 is a rank without one.
                let token = match message.bootstrap_token {
                    -1 => None,
                    token => {
                        match i32::try_from(token).ok().filter(|token| *token >= 0) {
                            Some(token) => Some(token),
                            None => {
                                self.fail(
                                room,
                                format!("Received KV completion with an invalid first token {token}"),
                            );
                                return;
                            }
                        }
                    }
                };
                let pending = self.pending.entry(room).or_default();
                if let Some(token) = token {
                    pending.bootstrap_token.get_or_insert(token);
                }
                if let Some(ids) = message.spec_candidate_ids {
                    pending.spec_candidate_ids.get_or_insert(ids);
                }
                if let Some(logprob) = message.bootstrap_logprob {
                    pending.bootstrap_logprob.get_or_insert(logprob);
                }
                if arrived.len() < expected.len() {
                    return;
                }
                if arrived != expected {
                    self.fail(
                        room,
                        "Prefill TP completion set disagrees with the transfer route",
                    );
                    return;
                }
                let landed = self.pending.remove(&room).unwrap_or_default();
                if landed.bootstrap_token.is_none() {
                    // The first decode feeds on that token: without it the
                    // request cannot run here.
                    self.fail(
                        room,
                        "Every Prefill rank reported the KV landed without the request's first token",
                    );
                    return;
                }
                self.landed.insert(room, landed);
                self.rooms.update(room, TransferPoll::Success);
            }
            TransferPoll::Failed => {
                self.fail(
                    room,
                    "Failed to get kvcache from prefill instance, it might be dead",
                );
            }
            status => self.rooms.update(room, status),
        }
    }

    /// A room is connected to `targets`, the ranks `route` names at `addr`:
    /// it is bootstrapped. A room whose request ended while the lookups ran
    /// stays forgotten. Returns whether the room was still tracked.
    fn attach(
        &mut self,
        room: u64,
        addr: &str,
        route: &PrefillRoute,
        targets: Arc<Vec<Target>>,
    ) -> bool {
        if !self.tracked.contains(&room) {
            return false;
        }
        self.expected
            .insert(room, route.expected_prefill_ranks.iter().copied().collect());
        self.targets.insert(room, targets);
        self.addr_rooms
            .entry(addr.to_owned())
            .or_default()
            .insert(room);
        self.rooms.update(room, TransferPoll::Bootstrapped);
        true
    }

    /// The Prefill node at bootstrap address `addr` is lost: forget its
    /// route and its connections, and fail its unfinished rooms. Returns how
    /// many failed.
    fn lose_node(&mut self, addr: &str) -> usize {
        self.connections
            .retain(|(connected, _, _), _| connected != addr);
        self.routes.remove(addr);
        let rooms = self.addr_rooms.remove(addr).unwrap_or_default();
        let mut affected = 0;
        for room in rooms {
            if matches!(
                self.rooms.get(room),
                None | Some(TransferPoll::Success | TransferPoll::Failed)
            ) || self.rooms.has_failure(room)
            {
                continue;
            }
            self.fail(
                room,
                format!("Losing connection with prefill instance (bootstrap_addr: {addr})"),
            );
            affected += 1;
        }
        affected
    }

    fn clear(&mut self, room: u64, addr: &str) {
        self.rooms.remove(room);
        self.tracked.remove(&room);
        self.expected.remove(&room);
        self.responses.remove(&room);
        self.pending.remove(&room);
        self.landed.remove(&room);
        self.cached_tokens.remove(&room);
        self.targets.remove(&room);
        self.admissions.remove(&room);
        self.waiting_since.remove(&room);
        if let Some(rooms) = self.addr_rooms.get_mut(addr) {
            rooms.remove(&room);
        }
    }
}

struct Shared {
    transport: Arc<Transport>,
    state: Mutex<State>,
    local_ip: String,
    engine_rank: i64,
    arena_base: u64,
    /// The contract's wire bytes, as registered.
    layout_wire: Vec<u8>,
    layout: Contract,
    tp_size: i64,
    tp_rank: i64,
    metrics: Option<Arc<ts_metrics::KvTransferMetrics>>,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("the PD state lock is poisoned")
    }

    /// The route for `addr`, fetched and planned once.
    fn route(&self, addr: &str) -> Result<Arc<PrefillRoute>, String> {
        if let Some(route) = self.state().routes.get(addr).cloned() {
            return Ok(route);
        }
        let info = http::get_text(
            &format!("http://{addr}/route?engine_rank=-1&target_dp_group=-1"),
            LOOKUP_TIMEOUT,
        )
        .map_err(|error| {
            tracing::error!(addr, error, "cannot fetch the prefill parallel info");
            format!("Could not fetch prefill parallel info from bootstrap_addr: {addr}")
        })?;
        let route = Arc::new(
            plan_route(&info, &self.layout, self.tp_size, self.tp_rank).map_err(|error| error.0)?,
        );
        self.state().routes.insert(addr.to_owned(), route.clone());
        Ok(route)
    }

    /// Connect a room to its Prefill ranks:
    /// look up the route, register with its ranks the first time the route
    /// is used, and mark the room bootstrapped.
    fn connect(&self, room: u64, addr: &str) -> Result<(), String> {
        let route = self.route(addr)?;
        let dp_group = room % route.prefill_dp_size.max(1) as u64;
        let key: ConnectionKey = (addr.to_owned(), dp_group, route.target_tp_ranks.clone());
        let known = self.state().connections.get(&key).cloned();
        let targets = match known {
            Some(targets) => targets,
            None => {
                let mut targets = Vec::with_capacity(route.target_tp_ranks.len());
                for &rank in &route.target_tp_ranks {
                    let lookup_failed = || {
                        format!(
                            "Could not fetch bootstrap info for engine rank: {} and \
                             target_dp_group: {dp_group}",
                            self.engine_rank
                        )
                    };
                    let body = http::get_text(
                        &format!(
                            "http://{addr}/route?engine_rank={rank}&target_dp_group={dp_group}"
                        ),
                        LOOKUP_TIMEOUT,
                    )
                    .map_err(|error| {
                        tracing::error!(addr, rank, error, "cannot fetch a prefill rank");
                        lookup_failed()
                    })?;
                    let info: serde_json::Value =
                        serde_json::from_str(&body).map_err(|_| lookup_failed())?;
                    let host = info["rank_ip"].as_str().ok_or_else(lookup_failed)?;
                    let port = info["rank_port"]
                        .as_u64()
                        .and_then(|port| u16::try_from(port).ok())
                        .ok_or_else(lookup_failed)?;
                    targets.push(Target {
                        host: host.to_owned(),
                        port,
                        is_dummy: route.dummy_tp_ranks.contains(&rank),
                    });
                }
                let targets = Arc::new(targets);
                self.state().connections.insert(key, targets.clone());
                // Registered once per route; a room's pre-allocation follows
                // on the same socket, so it arrives after the registration.
                let session_id = self.transport.session_id();
                for target in targets.iter() {
                    self.transport.sockets.send(
                        &target.host,
                        target.port,
                        wire::registration_frames(
                            &self.local_ip,
                            self.transport.sockets.port(),
                            &session_id,
                            self.arena_base,
                            &self.layout_wire,
                            self.tp_size,
                            self.tp_rank,
                        ),
                    );
                }
                targets
            }
        };
        if self.state().attach(room, addr, &route, targets) {
            tracing::debug!(room, "connected");
        }
        Ok(())
    }

    /// Fail every unfinished room reading from a lost Prefill node.
    fn lose_node(&self, addr: &str) {
        let affected = self.state().lose_node(addr);
        if let Some(metrics) = &self.metrics {
            for _ in 0..affected {
                metrics.record_failure();
            }
        }
        tracing::error!(
            addr,
            affected,
            "lost the connection with a prefill instance"
        );
    }

    fn watch(&self, interval: Duration, max_failures: u32, stop: &mpsc::Receiver<()>) {
        let mut failures: HashMap<String, u32> = HashMap::new();
        while !sleep_or_stop(stop, interval) {
            let addrs: Vec<String> = self.state().routes.keys().cloned().collect();
            for addr in addrs {
                let count = failures.entry(addr.clone()).or_insert(0);
                if http::healthy(
                    &format!("http://{addr}/health"),
                    Duration::from_secs(2),
                    Duration::from_secs(3),
                ) {
                    *count = 0;
                    // Drop the rooms that ended.
                    let mut state = self.state();
                    let State {
                        rooms, addr_rooms, ..
                    } = &mut *state;
                    if let Some(tracked) = addr_rooms.get_mut(&addr) {
                        tracked.retain(|&room| rooms.get(room).is_some());
                    }
                } else {
                    tracing::info!(addr, "attempting to reconnect to a prefill instance");
                    *count += 1;
                }
                if *count >= max_failures {
                    self.lose_node(&addr);
                }
            }
        }
    }
}

/// Where the forward thread hands a Decode rank's peer its work; never
/// waits for a transfer.
///
/// A plan's remote prefill is staged when the plan goes to the data plane
/// ([`DecodeOutbox::stage_remote_prefill`]) and handed over by the data plane
/// once the plan's page zeroing finished
/// ([`DecodeOutbox::hand_over_remote_prefill`]): the Prefill ranks' writes
/// are not ordered with the device's, so the pages they land in must be
/// sanitized first.
#[derive(Clone)]
pub struct DecodeOutbox {
    shared: Arc<Shared>,
    staged: Staged,
}

impl DecodeOutbox {
    /// A sleep releases the cache: deregister it from the transfer engine,
    /// refused while a tracked request's KV may still land.
    pub fn release_registration(&self) -> Result<(), String> {
        let mut receiving: Vec<u64> = self.shared.state().tracked.iter().copied().collect();
        if !receiving.is_empty() {
            receiving.sort_unstable();
            return Err(format!(
                "cannot release the cache while KV receives are in flight: rooms {receiving:?}"
            ));
        }
        self.shared.transport.release_regions()
    }

    /// A wake re-mapped the cache: register it under a fresh transfer
    /// session, as the Prefill ranks write into this memory with the keys of
    /// its registration, and forget the registrations made under the old
    /// one, so the next request registers the new session again.
    pub fn restore_registration(&self) -> Result<(), String> {
        self.shared.transport.restore_regions(true)?;
        self.shared.state().connections.clear();
        Ok(())
    }

    /// Stage a plan's remote prefill: the prompts whose pages it admitted.
    pub fn stage_remote_prefill(&self, rows: BatchRows) -> Result<(), String> {
        self.staged.stage(rows)
    }

    /// Whether a staged remote prefill still waits to be handed over.
    pub fn remote_prefill_pending(&self) -> bool {
        self.staged.is_pending()
    }

    /// The staged remote prefill's pages are sanitized: send each admitted
    /// prompt's manifest to its Prefill ranks (`HostDecodePeer.execute`).
    /// Every row is planned before any goes out.
    pub fn hand_over_remote_prefill(&self) -> Result<(), String> {
        let rows = self.staged.take()?;
        if rows.num_extends != rows.request_ids.len() {
            return Err("CachePD Decode admission does not support mixed batches".to_owned());
        }
        let layout = &self.shared.layout;
        let mut pending = Vec::new();
        for (index, rid) in rows.request_ids.iter().enumerate() {
            let Some(room) = self.shared.state().rooms_by_rid.get(rid).copied() else {
                continue;
            };
            let manifest = build_manifest(
                &rows.tables,
                layout,
                index,
                i64::from(rows.extend_prefix_lens[index]),
                i64::from(rows.prefill_lengths[index]),
                rows.state_prefix_cached[index],
            )
            .and_then(|manifest| {
                validate_manifest(&manifest, layout, "destination")?;
                manifest.to_wire()
            })
            .map_err(|error| error.0)?;
            pending.push((rid, room, rows.request_pool_indices[index], manifest));
        }
        for (rid, room, request_pool_index, manifest) in pending {
            self.receive(rid, room, request_pool_index, &manifest);
        }
        Ok(())
    }

    /// A request's pages are admitted: send its manifest to its Prefill
    /// ranks; its KV lands at `request_pool_index`.
    fn receive(&self, rid: &str, room: u64, request_pool_index: i32, manifest: &[u8]) {
        let targets = {
            let mut state = self.shared.state();
            if !state.tracked.contains(&room) {
                // The loop ended the request (its transfer failed) after it
                // planned this admission: its pages go back, so no Prefill
                // rank may write them.
                tracing::debug!(rid, room, "not pre-allocating an ended request");
                return;
            }
            state.admissions.insert(room, request_pool_index);
            state.waiting_since.insert(room, Instant::now());
            state.rooms.update(room, TransferPoll::WaitingForInput);
            state.targets.get(&room).cloned()
        };
        let Some(targets) = targets else {
            tracing::warn!(
                rid,
                room,
                "a request without a route reached its remote prefill"
            );
            return;
        };
        let session_id = self.shared.transport.session_id();
        for target in targets.iter() {
            self.shared.transport.sockets.send(
                &target.host,
                target.port,
                wire::preallocation_frames(
                    room,
                    &session_id,
                    (!target.is_dummy).then_some(manifest),
                ),
            );
        }
        tracing::debug!(rid, room, "pre-allocation sent");
    }
}

/// One request's transfer as the scheduling loop tracks it.
struct Receiver {
    room: u64,
    addr: String,
    local: TransferPoll,
    concluded: Option<TransferPoll>,
}

/// A Decode rank's PD transfer peer.
pub struct DecodePeer {
    shared: Arc<Shared>,
    connect: mpsc::Sender<(u64, String)>,
    stop: Arc<AtomicBool>,
    _heartbeat_stop: mpsc::Sender<()>,
    waiting_timeout: Duration,
    /// Tracked requests, in registration order (every replica member
    /// registers the same requests in the same order).
    order: Vec<String>,
    receivers: HashMap<String, Receiver>,
    timeout_logged: bool,
}

impl DecodePeer {
    /// Start the transfer engine, register the cache, open the status
    /// socket and start the threads.
    pub fn start(config: DecodeConfig) -> Result<(Self, DecodeOutbox), StartError> {
        let layout = Contract::from_wire(&config.layout)?;
        let (incoming, messages) = mpsc::channel::<Vec<bytes::Bytes>>();
        let transport = Transport::start(&config.peer, incoming)?;
        let shared = Arc::new(Shared {
            transport,
            state: Mutex::new(State::default()),
            local_ip: config.peer.local_ip.clone(),
            engine_rank: config.peer.engine_rank,
            arena_base: config.arena_base,
            layout_wire: config.layout,
            layout,
            tp_size: config.tp_size,
            tp_rank: config.tp_rank,
            metrics: config.peer.metrics.clone(),
        });
        let stop = Arc::new(AtomicBool::new(false));
        {
            let shared = shared.clone();
            let stop = stop.clone();
            thread::Builder::new()
                .name("ts-pd-status".to_owned())
                .spawn(move || loop {
                    match messages.recv_timeout(SOCKET_POLL) {
                        Ok(frames) => match StatusMessage::from_frames(&frames) {
                            Ok(message) => {
                                tracing::debug!(room = message.room, status = ?message.status, "status");
                                shared.state().apply(message);
                            }
                            Err(error) => {
                                tracing::error!(%error, "rejecting a malformed Prefill status message");
                            }
                        },
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
                .name("ts-pd-connect".to_owned())
                .spawn(move || {
                    while let Ok((room, addr)) = rooms.recv() {
                        if let Err(reason) = shared.connect(room, &addr) {
                            shared.state().fail(room, reason);
                        }
                    }
                })?;
        }
        let (heartbeat_stop, stopped) = mpsc::channel();
        {
            let shared = shared.clone();
            let interval = config.heartbeat_interval;
            let max_failures = config.heartbeat_max_failures.max(1);
            thread::Builder::new()
                .name("ts-pd-heartbeat".to_owned())
                .spawn(move || shared.watch(interval, max_failures, &stopped))?;
        }
        let outbox = DecodeOutbox {
            shared: shared.clone(),
            staged: Staged::default(),
        };
        Ok((
            Self {
                shared,
                connect,
                stop,
                _heartbeat_stop: heartbeat_stop,
                waiting_timeout: config.waiting_timeout,
                order: Vec::new(),
                receivers: HashMap::new(),
                timeout_logged: false,
            },
            outbox,
        ))
    }

    /// The status socket's port.
    pub fn port(&self) -> u16 {
        self.shared.transport.sockets.port()
    }

    /// The transfer engine's session id.
    pub fn session_id(&self) -> String {
        self.shared.transport.session_id()
    }

    fn forget(&mut self, rid: &str) {
        let Some(receiver) = self.receivers.remove(rid) else {
            return;
        };
        self.order.retain(|tracked| tracked != rid);
        let mut state = self.shared.state();
        state.clear(receiver.room, &receiver.addr);
        state.rooms_by_rid.remove(rid);
    }
}

impl Drop for DecodePeer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// `host:port`, with an IPv6 host in brackets (`join_host_port`).
fn join_host_port(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

impl TransferPeer for DecodePeer {
    fn register(&mut self, rid: &str, bootstrap: &BootstrapInfo) -> Result<(), PdError> {
        let room = bootstrap.room;
        let addr = join_host_port(&bootstrap.host, bootstrap.port);
        {
            let mut state = self.shared.state();
            state.tracked.insert(room);
            state.rooms_by_rid.insert(rid.to_owned(), room);
            state.rooms.update(room, TransferPoll::Bootstrapping);
        }
        if self
            .receivers
            .insert(
                rid.to_owned(),
                Receiver {
                    room,
                    addr: addr.clone(),
                    local: TransferPoll::Bootstrapping,
                    concluded: None,
                },
            )
            .is_none()
        {
            self.order.push(rid.to_owned());
        }
        self.connect
            .send((room, addr))
            .map_err(|_| PdError::Peer("the PD connect thread stopped".to_owned()))
    }

    fn poll(&mut self, mesh: &mut Mesh, replica: &Group) -> Result<Vec<PdEvent>, PdError> {
        if self.order.is_empty() {
            return Ok(Vec::new());
        }
        let now = Instant::now();
        let mut polls = Vec::with_capacity(self.order.len());
        {
            let mut state = self.shared.state();
            for rid in &self.order {
                let receiver = self.receivers.get_mut(rid).expect("tracked");
                let poll = match receiver.concluded {
                    Some(poll) => poll,
                    None => match state
                        .rooms
                        .get(receiver.room)
                        .unwrap_or(TransferPoll::Failed)
                    {
                        status @ (TransferPoll::Success | TransferPoll::Failed) => {
                            receiver.concluded = Some(status);
                            status
                        }
                        TransferPoll::WaitingForInput => {
                            let waited = state
                                .waiting_since
                                .get(&receiver.room)
                                .map(|since| now - *since);
                            match waited {
                                Some(waited) if waited >= self.waiting_timeout => {
                                    if !self.timeout_logged {
                                        self.timeout_logged = true;
                                        tracing::warn!(
                                            "Some requests never received their KV after \
                                             bootstrapping. Raise \
                                             TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT if a \
                                             longer TTFT is acceptable."
                                        );
                                    }
                                    state.rooms.record_failure(
                                        receiver.room,
                                        format!(
                                            "Request {} timed out after {:.1}s in \
                                             TransferPoll.WaitingForInput",
                                            receiver.room,
                                            waited.as_secs_f64()
                                        ),
                                    );
                                    if let Some(metrics) = &self.shared.metrics {
                                        metrics.record_timeout("waiting_for_input");
                                    }
                                    receiver.concluded = Some(TransferPoll::Failed);
                                    TransferPoll::Failed
                                }
                                _ => TransferPoll::WaitingForInput,
                            }
                        }
                        status => status,
                    },
                };
                polls.push(poll as i64);
            }
        }
        let agreed = if replica.size() > 1 {
            mesh.all_reduce(replica, &polls, Reduce::Min)?
        } else {
            polls
        };

        let mut events = Vec::new();
        let mut ended = Vec::new();
        for (rid, &poll) in self.order.iter().zip(&agreed) {
            let receiver = self.receivers.get_mut(rid).expect("tracked");
            let poll = TransferPoll::from_wire(poll).unwrap_or(TransferPoll::Failed);
            if receiver.local == TransferPoll::Bootstrapping && poll == TransferPoll::Bootstrapped {
                receiver.local = TransferPoll::Bootstrapped;
                events.push(PdEvent::Bootstrapped(rid.clone()));
            } else if poll == TransferPoll::Failed {
                let reason = self
                    .shared
                    .state()
                    .rooms
                    .failure(receiver.room)
                    .map(str::to_owned);
                tracing::warn!(rid, room = receiver.room, reason, "KV transfer failed");
                events.push(PdEvent::Failed(rid.clone()));
                ended.push(rid.clone());
            } else if receiver.local == TransferPoll::Bootstrapped && poll == TransferPoll::Success
            {
                receiver.local = TransferPoll::Success;
                let mut state = self.shared.state();
                let landed = state.landed.remove(&receiver.room).unwrap_or_default();
                let cached_tokens = state.cached_tokens.remove(&receiver.room).unwrap_or(0);
                let request_pool_index =
                    *state.admissions.get(&receiver.room).ok_or_else(|| {
                        PdError::Peer(format!("{rid}'s KV landed before its pages were admitted"))
                    })?;
                // A room lands only with its first token (`State::apply`).
                let bootstrap_token = landed.bootstrap_token.ok_or_else(|| {
                    PdError::Peer(format!("{rid}'s KV landed without its first token"))
                })?;
                events.push(PdEvent::RemotePrefillDone(RemotePrefill {
                    rid: rid.clone(),
                    bootstrap_token,
                    cached_tokens,
                    bootstrap_logprob: landed.bootstrap_logprob,
                    spec_candidate_ids: landed.spec_candidate_ids,
                    request_pool_index,
                }));
                ended.push(rid.clone());
            }
        }
        for rid in ended {
            self.forget(&rid);
        }
        Ok(events)
    }

    fn record_usage(&mut self, _rid: &str, _cached_tokens: i64, _bootstrap_logprob: Option<f64>) {}

    fn has_in_flight(&self) -> bool {
        !self.order.is_empty()
    }

    /// The decode node has no encoder to wait on.
    fn abort_room(&mut self, _: &str, _: u64) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn success(room: u64, prefill_rank: i64, bootstrap_token: i64) -> StatusMessage {
        StatusMessage {
            room,
            status: TransferPoll::Success,
            prefill_rank,
            bootstrap_token,
            spec_candidate_ids: None,
            cached_tokens: 0,
            bootstrap_logprob: None,
        }
    }

    fn route() -> PrefillRoute {
        PrefillRoute {
            target_tp_ranks: vec![0, 1],
            dummy_tp_ranks: Vec::new(),
            expected_prefill_ranks: vec![0, 1],
            prefill_dp_size: 1,
        }
    }

    /// Track `room` and connect it to Prefill ranks 0 and 1 at `addr`.
    fn connected(state: &mut State, room: u64, addr: &str) {
        state.tracked.insert(room);
        state.rooms.update(room, TransferPoll::Bootstrapping);
        assert!(state.attach(room, addr, &route(), Arc::new(Vec::new())));
    }

    #[test]
    fn a_room_lands_with_the_first_token_any_rank_carried() {
        let mut state = State::default();
        connected(&mut state, 1, "prefill:80");
        state.apply(success(1, 1, -1));
        assert_eq!(state.rooms.get(1), Some(TransferPoll::Bootstrapped));
        state.apply(success(1, 0, 7));
        assert_eq!(state.rooms.get(1), Some(TransferPoll::Success));
        assert_eq!(state.landed[&1].bootstrap_token, Some(7));
    }

    #[test]
    fn a_landing_without_a_valid_first_token_fails_the_room() {
        let mut state = State::default();
        connected(&mut state, 1, "prefill:80");
        state.apply(success(1, 0, -1));
        state.apply(success(1, 1, -1));
        assert_eq!(state.rooms.get(1), Some(TransferPoll::Failed));
        assert!(state.rooms.failure(1).unwrap().contains("first token"));
        connected(&mut state, 2, "prefill:80");
        state.apply(success(2, 0, i64::from(i32::MAX) + 1));
        assert_eq!(state.rooms.get(2), Some(TransferPoll::Failed));
    }

    #[test]
    fn a_forgotten_room_is_not_brought_back() {
        let mut state = State::default();
        connected(&mut state, 1, "prefill:80");
        state.clear(1, "prefill:80");
        // A late report, failure or connection of the room changes nothing.
        state.apply(success(1, 0, 7));
        state.fail(1, "late");
        assert!(!state.attach(1, "prefill:80", &route(), Arc::new(Vec::new())));
        assert_eq!(state.rooms.get(1), None);
        assert!(state.expected.is_empty());
        assert!(state.targets.is_empty());
        assert!(state.pending.is_empty() && state.landed.is_empty());
    }

    #[test]
    fn losing_a_node_keeps_another_whose_address_extends_its_own() {
        let mut state = State::default();
        for (room, addr) in [(1, "host:80"), (2, "host:8000")] {
            connected(&mut state, room, addr);
            state
                .connections
                .insert((addr.to_owned(), 0, vec![0, 1]), Arc::new(Vec::new()));
        }
        assert_eq!(state.lose_node("host:80"), 1);
        assert_eq!(state.rooms.get(1), Some(TransferPoll::Failed));
        assert_eq!(state.rooms.get(2), Some(TransferPoll::Bootstrapped));
        let kept: Vec<&str> = state
            .connections
            .keys()
            .map(|(addr, _, _)| addr.as_str())
            .collect();
        assert_eq!(kept, ["host:8000"]);
    }
}
