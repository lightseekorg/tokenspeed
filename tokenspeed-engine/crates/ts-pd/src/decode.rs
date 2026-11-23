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
//! bootstrap server, registers with them once per route (and waits until
//! each acknowledged the registration), sends each admitted request's page
//! manifest, and collects their completions.
//!
//! Threads: one per bootstrap address connects its requests to their routes
//! (HTTP lookups), one handles the status socket, one watches the Prefill
//! nodes' health. The scheduling loop only registers requests and polls;
//! the forward thread hands over each plan's remote prefill
//! ([`DecodeOutbox`]). Routes and manifests are planned here, over the
//! rank's cache transfer contract ([`crate::layout`]).

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc, Condvar, Mutex, MutexGuard,
    },
    thread,
    time::{Duration, Instant},
};

use ts_collective::{Group, Mesh, Reduce};

use crate::{
    batch::{BatchRows, Staged},
    control::{
        stalled_drains, BootstrapInfo, Draining, PdError, PdEvent, RemotePrefill, TransferPeer,
    },
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

/// The most announcements a heartbeat repeats to one Prefill endpoint: a
/// beat must not fill the endpoint's queue, which registrations and
/// manifests share.
const ANNOUNCE_BURST: usize = 256;
/// How long a route's registration waits for every target's acknowledgment.
/// A control connection that fails discards its queue after five seconds
/// (`net`); this leaves room for a resend on a fresh connection.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(10);

/// How often a registration not yet acknowledged by every target is resent.
const REGISTRATION_RESEND: Duration = Duration::from_secs(1);

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
    /// Maximum wait in each phase: registration through manifest publication,
    /// then publication through KV arrival. Failure still waits for write
    /// retirement; a failed receive still unretired after another such wait
    /// revokes the transfer engine (`Shared::revoke`).
    pub waiting_timeout: Duration,
    /// How often the Prefill nodes' health is checked.
    pub heartbeat_interval: Duration,
    /// Failed checks in a row after which a Prefill node counts as lost.
    pub heartbeat_max_failures: u32,
}

/// A route registration waiting for its targets' replies.
#[derive(Default)]
struct Registering {
    acknowledged: BTreeSet<i64>,
    /// Why each rank that rejected the registration did; only the route's
    /// targets count.
    rejections: BTreeMap<i64, String>,
}

/// A Prefill rank a route reaches.
#[derive(Debug, Clone)]
struct Target {
    rank: i64,
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
struct Cancellation {
    /// The cache the room's route registered, which its manifest names
    /// (else the current one); the cancellation and its acknowledgments name
    /// it too.
    cache_id: String,
    acknowledged: BTreeSet<i64>,
    sent: Option<Instant>,
}

#[derive(Default)]
struct State {
    /// Changes whenever the cache is restored, even if its address is reused.
    cache_id: String,
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
    /// The cache identity each attached room's route registered: its
    /// manifest names that cache, and so the transfer engine of that time.
    attached_caches: HashMap<u64, String>,
    waiting_since: HashMap<u64, Instant>,
    cancellations: HashMap<u64, Cancellation>,
    /// Bootstrap address -> the rooms reading from it.
    addr_rooms: HashMap<String, HashSet<u64>>,
    /// Bootstrap address -> its route; the addresses the heartbeat watches.
    routes: HashMap<String, (Option<String>, Arc<PrefillRoute>)>,
    /// The routes every target acknowledged this cache's registration on.
    connections: HashMap<ConnectionKey, Arc<Vec<Target>>>,
    /// The connected rooms whose Prefill node defers the decode allocation: the
    /// node was told of each, and its notice admits the room here.
    announced: HashSet<u64>,
    /// The Prefill ranks whose notice arrived: the room's request holds
    /// every page of its prompt there.
    notices: HashMap<u64, BTreeSet<i64>>,
    /// Route registrations in progress, by id.
    registering: HashMap<u64, Registering>,
    /// A revocation is replacing the transfer engine ([`Shared::revoke`]):
    /// no room connects, and the cache is not released, until it finished.
    revoking: bool,
    /// A revocation could not deregister the cache from the engine it
    /// replaced, whose keys may still reach the cache: it is never released.
    revocation_unfenced: bool,
    /// When a failed revocation may be tried again.
    revocation_not_before: Option<Instant>,
    /// Rooms whose manifests named a transfer engine that was replaced and
    /// deregistered: no remote write reaches their pages any more.
    revoked: HashSet<u64>,
    /// The engines revocations replaced, kept for the peer's lifetime. Each
    /// keeps its session (host and RPC port) bound to an engine without the
    /// cache registered: a revoked writer that resolves its old session
    /// again cannot reach a later engine that reused the port.
    revoked_engines: Vec<Arc<ts_mooncake::Engine>>,
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

    fn awaiting_connection(&self, room: u64) -> bool {
        self.tracked.contains(&room)
            && self.rooms.get(room) == Some(TransferPoll::Bootstrapping)
            && !self.cancellations.contains_key(&room)
    }

    /// An announced room whose Prefill ranks have not all sent their notice.
    fn awaiting_notice(&self, room: u64) -> bool {
        self.announced.contains(&room)
            && !self.expected.get(&room).is_some_and(|expected| {
                self.notices
                    .get(&room)
                    .is_some_and(|notices| expected.is_subset(notices))
            })
    }

    /// Apply one Prefill rank's report.
    fn apply(&mut self, message: StatusMessage) {
        let room = message.room;
        if !self.tracked.contains(&room) || self.cancellations.contains_key(&room) {
            tracing::debug!(room, prefill_rank = message.prefill_rank, status = ?message.status, "ignoring late transfer status");
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
            // A Prefill rank's notice. It is not a state of this room: the
            // room is bootstrapped once this rank connected it.
            TransferPoll::Bootstrapped => {
                self.notices
                    .entry(room)
                    .or_default()
                    .insert(message.prefill_rank);
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
        if !self.awaiting_connection(room) {
            return false;
        }
        self.expected
            .insert(room, route.expected_prefill_ranks.iter().copied().collect());
        self.targets.insert(room, targets);
        self.attached_caches.insert(room, self.cache_id.clone());
        self.addr_rooms
            .entry(addr.to_owned())
            .or_default()
            .insert(room);
        if route.deferred {
            self.announced.insert(room);
        }
        self.rooms.update(room, TransferPoll::Bootstrapped);
        true
    }

    fn acknowledge_cancellation(&mut self, room: u64, cache_id: &str, rank: i64) {
        if !self
            .targets
            .get(&room)
            .is_some_and(|targets| targets.iter().any(|target| target.rank == rank))
        {
            return;
        }
        if let Some(cancellation) = self
            .cancellations
            .get_mut(&room)
            .filter(|cancellation| cancellation.cache_id == cache_id)
        {
            cancellation.acknowledged.insert(rank);
        }
    }

    fn cancellation_retired(&self, room: u64) -> bool {
        let Some(cancellation) = self.cancellations.get(&room) else {
            return false;
        };
        // No manifest ever named these pages. The state lock excludes a
        // concurrent admission, which also checks the cancellation marker.
        !self.admissions.contains_key(&room)
            || self.revoked.contains(&room)
            || self.targets.get(&room).is_some_and(|targets| {
                targets
                    .iter()
                    .all(|target| cancellation.acknowledged.contains(&target.rank))
            })
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
        self.attached_caches.remove(&room);
        self.admissions.remove(&room);
        self.waiting_since.remove(&room);
        self.cancellations.remove(&room);
        self.announced.remove(&room);
        self.notices.remove(&room);
        self.revoked.remove(&room);
        if let Some(rooms) = self.addr_rooms.get_mut(addr) {
            rooms.remove(&room);
        }
    }
}

/// A running revocation ([`Shared::revoke`]). [`Shared::revoke`] ends it in
/// the critical section that marks the rooms it fenced, so no room can retire
/// in between; dropping one it did not end (no fresh engine, or a panic,
/// also while unwinding) ends it having fenced nothing.
struct Revocation {
    shared: Arc<Shared>,
    /// When another revocation may start unless this one succeeds.
    retry_at: Instant,
    /// Whether [`Revocation::end`] ran.
    ended: bool,
}

impl Revocation {
    /// Clear [`State::revoking`] in `state`, the lock its caller holds;
    /// `retry_at` is when another revocation may start.
    fn end(&mut self, state: &mut State, retry_at: Option<Instant>) {
        state.revoking = false;
        state.revocation_not_before = retry_at;
        self.shared.revocation_finished.notify_all();
        self.ended = true;
    }
}

impl Drop for Revocation {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        // Must not panic while unwinding: take the lock even if poisoned.
        let shared = self.shared.clone();
        let mut state = shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let retry_at = self.retry_at;
        self.end(&mut state, Some(retry_at));
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
    /// Signaled when a Prefill rank replies to a registration.
    registration_replied: Condvar,
    /// The id of the next route registration.
    next_registration_id: AtomicU64,
    /// Signaled when a revocation finished.
    revocation_finished: Condvar,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("the PD state lock is poisoned")
    }

    /// Prefill rank `rank` installed registration `id` of cache `cache_id`,
    /// or rejected it for `rejection`.
    fn reply_to_registration(&self, id: u64, cache_id: &str, rank: i64, rejection: Option<String>) {
        let mut state = self.state();
        if cache_id != state.cache_id {
            return;
        }
        let Some(registering) = state.registering.get_mut(&id) else {
            return;
        };
        match rejection {
            None => {
                registering.acknowledged.insert(rank);
            }
            Some(reason) => {
                registering.rejections.entry(rank).or_insert(reason);
            }
        }
        self.registration_replied.notify_all();
    }

    /// Register the current cache with a route's `targets`, resending until
    /// every target acknowledged it: a registration queued on a control
    /// socket is lost with that socket's backlog when its connection fails
    /// (`net::Sockets`). Returns the state lock once every target
    /// acknowledged, `None` if the room stopped waiting for its connection
    /// or the cache is being or was replaced meanwhile, and an error when a target
    /// rejected the registration or the targets did not all acknowledge it
    /// within [`REGISTRATION_TIMEOUT`].
    fn register<'a>(
        &'a self,
        mut state: MutexGuard<'a, State>,
        room: u64,
        targets: &[Target],
    ) -> Result<Option<MutexGuard<'a, State>>, String> {
        let id = self.next_registration_id.fetch_add(1, Ordering::Relaxed);
        let cache_id = state.cache_id.clone();
        // Built under the incarnation lock: the frames name the current
        // cache and session, both replaced together on a wake.
        let frames = wire::registration_frames(
            &self.local_ip,
            self.transport.sockets.port(),
            &self.transport.session_id(),
            &cache_id,
            self.arena_base,
            &self.layout_wire,
            self.tp_size,
            self.tp_rank,
            id,
        );
        state.registering.insert(id, Registering::default());
        let deadline = Instant::now() + REGISTRATION_TIMEOUT;
        let mut resend_at = Instant::now();
        let mut refused = None;
        let outcome = loop {
            if !state.awaiting_connection(room) || state.revoking || state.cache_id != cache_id {
                break Ok(false);
            }
            let registering = &state.registering[&id];
            if let Some((rank, reason)) = targets
                .iter()
                .find_map(|target| registering.rejections.get_key_value(&target.rank))
            {
                break Err(format!(
                    "Prefill rank {rank} rejected the Decode cache registration: {reason}"
                ));
            }
            let acknowledged = &registering.acknowledged;
            let unacknowledged: Vec<&Target> = targets
                .iter()
                .filter(|target| !acknowledged.contains(&target.rank))
                .collect();
            if unacknowledged.is_empty() {
                break Ok(true);
            }
            let now = Instant::now();
            if now >= deadline {
                let ranks: Vec<i64> = unacknowledged.iter().map(|target| target.rank).collect();
                let refused = refused
                    .map(|error| format!("; the last resend was refused: {error}"))
                    .unwrap_or_default();
                break Err(format!(
                    "Prefill ranks {ranks:?} did not acknowledge the Decode cache registration \
                     within {REGISTRATION_TIMEOUT:?} (Prefill must run a version that \
                     acknowledges registrations){refused}"
                ));
            }
            if now >= resend_at {
                // A room's pre-allocation follows on these same sockets. A
                // refused enqueue (a full or retiring queue) is retried on the
                // next resend, within the deadline.
                refused = None;
                for target in &unacknowledged {
                    if let Err(error) =
                        self.transport
                            .sockets
                            .send(&target.host, target.port, frames.clone())
                    {
                        refused = Some(error.to_string());
                    }
                }
                resend_at = now + REGISTRATION_RESEND;
            }
            state = self
                .registration_replied
                .wait_timeout(state, resend_at.min(deadline) - now)
                .expect("the PD state lock is poisoned")
                .0;
        };
        state.registering.remove(&id);
        match outcome {
            Ok(true) => Ok(Some(state)),
            Ok(false) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Queue cancellation on the same sockets as registration and manifests.
    /// Retrying an unacknowledged request is safe and does not block the loop.
    fn cancel(&self, room: u64) {
        let mut state = self.state();
        if !state.tracked.contains(&room) {
            return;
        }
        if !state.rooms.has_failure(room) {
            state.fail(
                room,
                "KV receive cancelled; waiting for remote writes to retire",
            );
        } else {
            state.rooms.update(room, TransferPoll::Failed);
        }
        let cache_id = state
            .attached_caches
            .get(&room)
            .unwrap_or(&state.cache_id)
            .clone();
        let cancellation = state
            .cancellations
            .entry(room)
            .or_insert_with(|| Cancellation {
                cache_id,
                ..Cancellation::default()
            });
        if cancellation
            .sent
            .is_some_and(|at| at.elapsed() < Duration::from_secs(1))
        {
            return;
        }
        cancellation.sent = Some(Instant::now());
        let acknowledged = cancellation.acknowledged.len();
        let frames = wire::cancellation_frames(room, &cancellation.cache_id);
        if let Some(targets) = state.targets.get(&room) {
            for target in targets.iter() {
                // A full queue is retried on the next cancellation poll.
                let _ = self.transport.sockets.send_coalesced(
                    &target.host,
                    target.port,
                    frames.clone(),
                );
            }
            let target_count = targets.len();
            drop(state);
            tracing::debug!(
                room,
                acknowledged,
                targets = target_count,
                "attempted KV cancellation"
            );
        }
    }

    /// Revalidate before attaching a new room: a healthy address may be a new P.
    fn route(&self, addr: &str) -> Result<Arc<PrefillRoute>, String> {
        let info = http::get_text(
            &format!("http://{addr}/route?engine_rank=-1&target_dp_group=-1"),
            LOOKUP_TIMEOUT,
        )
        .map_err(|error| {
            tracing::error!(addr, error, "cannot fetch the prefill parallel info");
            format!("Could not fetch prefill parallel info from bootstrap_addr: {addr}")
        })?;
        let description: serde_json::Value = serde_json::from_str(&info)
            .map_err(|error| format!("invalid prefill route: {error}"))?;
        let version = description["route_version"].as_str().map(str::to_owned);
        {
            let state = self.state();
            if let Some((known_version, route)) = state.routes.get(addr) {
                if version.is_some() && *known_version == version {
                    return Ok(route.clone());
                }
            }
        }
        let route = Arc::new(
            plan_route(&info, &self.layout, self.tp_size, self.tp_rank).map_err(|error| error.0)?,
        );
        let mut state = self.state();
        // Existing rooms retain their targets and retirement fences. Only new
        // rooms resolve the replacement ranks and register with them again.
        state
            .connections
            .retain(|(connected, _, _), _| connected != addr);
        state
            .routes
            .insert(addr.to_owned(), (version, route.clone()));
        Ok(route)
    }

    /// Connect a room to its Prefill ranks:
    /// look up the route, register with its ranks the first time the route
    /// is used (recording the route once every rank acknowledged), and mark
    /// the room bootstrapped.
    fn connect(&self, room: u64, addr: &str) -> Result<(), String> {
        let started = tracing::enabled!(tracing::Level::DEBUG).then(Instant::now);
        let cache_id = {
            let state = self
                .revocation_finished
                .wait_while(self.state(), |state| state.revoking)
                .expect("the PD state lock is poisoned");
            if !state.awaiting_connection(room) {
                return Ok(());
            }
            state.cache_id.clone()
        };
        let route = self.route(addr)?;
        if let Some(started) = started {
            tracing::debug!(
                room,
                bootstrap_addr = addr,
                elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
                "event=ts_decode_route_resolved"
            );
        }
        let dp_group = room % route.prefill_dp_size.max(1) as u64;
        let key: ConnectionKey = (addr.to_owned(), dp_group, route.target_tp_ranks.clone());
        let known = self.state().connections.get(&key).cloned();
        let targets = match known {
            Some(targets) => targets,
            None => {
                let mut targets = Vec::with_capacity(route.target_tp_ranks.len());
                for &rank in &route.target_tp_ranks {
                    // A retired worker only finishes the HTTP call already
                    // in progress; do not start another lookup for an ended room.
                    if !self.state().awaiting_connection(room) {
                        return Ok(());
                    }
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
                        rank,
                        host: host.to_owned(),
                        port,
                        is_dummy: route.dummy_tp_ranks.contains(&rank),
                    });
                }
                Arc::new(targets)
            }
        };
        let mut state = self.state();
        // The HTTP lookup may outlive a cancelled request or a cache release.
        // Do not repopulate the new cache's connections with that old work.
        if !state.awaiting_connection(room) {
            return Ok(());
        }
        if state.revoking || state.cache_id != cache_id {
            // The transfer engine is being or was replaced: connect again,
            // registering the new cache.
            drop(state);
            return self.connect(room, addr);
        }
        if !state.connections.contains_key(&key) {
            state = match self.register(state, room, &targets)? {
                Some(state) => state,
                // The transfer engine is being or was replaced: a room still
                // waiting connects again, registering the new cache.
                None if self.state().awaiting_connection(room) => return self.connect(room, addr),
                None => return Ok(()),
            };
            state.connections.insert(key, targets.clone());
        }
        if state.attach(room, addr, &route, targets.clone()) {
            tracing::debug!(
                room,
                bootstrap_addr = addr,
                elapsed_ms = started.map(|at| at.elapsed().as_secs_f64() * 1000.0),
                "event=ts_decode_connected"
            );
            if route.deferred {
                // The heartbeat repeats it until every rank answered.
                for target in targets.iter() {
                    let _ = self.transport.sockets.send_coalesced(
                        &target.host,
                        target.port,
                        wire::announce_frames(room, &state.cache_id),
                    );
                }
            }
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

    /// An announcement or its answer may be lost: repeat the announcements
    /// of the rooms still waiting for a notice, in room order from `resume`,
    /// at most [`ANNOUNCE_BURST`] per endpoint and none to an endpoint whose
    /// queue is full. Returns the first room left out, where the next beat
    /// starts, or 0 when every room was covered.
    fn announce_waiting(&self, resume: u64) -> u64 {
        let state = self.state();
        let mut waiting: Vec<u64> = state
            .announced
            .iter()
            .copied()
            .filter(|&room| state.awaiting_notice(room) && !state.cancellations.contains_key(&room))
            .collect();
        waiting.sort_unstable();
        let start = waiting.partition_point(|&room| room < resume);
        waiting.rotate_left(start);
        let mut sent: HashMap<(&str, u16), usize> = HashMap::new();
        let mut left_out = None;
        for &room in &waiting {
            let Some(targets) = state.targets.get(&room) else {
                continue;
            };
            if targets.iter().any(|target| {
                sent.get(&(target.host.as_str(), target.port))
                    .is_some_and(|&count| count >= ANNOUNCE_BURST)
            }) {
                left_out.get_or_insert(room);
                continue;
            }
            // The cache the room's route registered: after a revocation the
            // current one is unknown to Prefill until a room registers it.
            let cache_id = state.attached_caches.get(&room).unwrap_or(&state.cache_id);
            for target in targets.iter() {
                let count = sent.entry((target.host.as_str(), target.port)).or_default();
                match self.transport.sockets.send_coalesced(
                    &target.host,
                    target.port,
                    wire::announce_frames(room, cache_id),
                ) {
                    Ok(()) => *count += 1,
                    Err(_) => {
                        *count = ANNOUNCE_BURST;
                        left_out.get_or_insert(room);
                    }
                }
            }
        }
        left_out.unwrap_or(0)
    }

    /// The recovery entry for a failed receive whose writers did not all
    /// acknowledge its cancellation within the receive budget: the Prefill
    /// rank died or restarted, or a write of its may still be in flight.
    /// Starts [`Shared::revoke`] off the scheduling loop unless one runs, a
    /// failed one is cooling down for `retry_after`, or the transport cannot
    /// fence remote writers (the pages then stay pinned, as logged).
    fn start_revocation(self: &Arc<Self>, now: Instant, retry_after: Duration) {
        if !self.transport.deregistration_fences_writes() {
            return;
        }
        {
            let mut state = self.state();
            if state.revoking || state.revocation_not_before.is_some_and(|at| now < at) {
                return;
            }
            state.revoking = true;
        }
        let mut revocation = Revocation {
            shared: self.clone(),
            retry_at: now + retry_after,
            ended: false,
        };
        // A failed spawn drops the revocation, which ends it.
        let spawned = thread::Builder::new()
            .name("ts-pd-revoke".to_owned())
            .spawn(move || revocation.shared.clone().revoke(&mut revocation));
        if let Err(error) = spawned {
            tracing::error!(%error, "cannot start revoking unacknowledged KV receives");
        }
    }

    /// Fence every remote write into the pages published so far: register
    /// the cache under a fresh transfer engine and identity, then deregister
    /// it from the replaced engine, whose keys every published manifest
    /// named. Every receive attached to the replaced transport fails. Once
    /// the deregistration succeeded, the rooms whose manifests named the
    /// replaced engine retire without acknowledgments; if it failed, they
    /// keep waiting for them. Either way it ends `revocation` as it marks
    /// them.
    fn revoke(&self, revocation: &mut Revocation) {
        let fresh = self.transport.fresh_engine().and_then(|engine| {
            crate::new_instance_id()
                .map(|cache_id| (engine, cache_id))
                .map_err(|error| error.to_string())
        });
        let (engine, cache_id) = match fresh {
            Ok(fresh) => fresh,
            Err(error) => {
                tracing::error!(
                    rank = self.engine_rank,
                    error,
                    detail = "no fresh transfer engine: the unacknowledged KV receives stay \
                              pinned until a later revocation",
                    "event=ts_decode_revocation_not_started"
                );
                return;
            }
        };
        let (replaced, replaced_cache_id, failed) = {
            let mut state = self.state();
            let attached: Vec<u64> = state
                .targets
                .keys()
                .copied()
                .filter(|&room| {
                    state.tracked.contains(&room)
                        && !matches!(
                            state.rooms.get(room),
                            Some(TransferPoll::Success | TransferPoll::Failed)
                        )
                })
                .collect();
            for &room in &attached {
                state.fail(
                    room,
                    "KV receive revoked: the transfer engine was replaced to fence a \
                     Prefill writer that did not acknowledge a cancellation",
                );
            }
            // New registrations name the fresh engine and identity; no
            // manifest names the replaced engine from here on.
            let replaced = self.transport.install_engine(engine);
            let replaced_cache_id = std::mem::replace(&mut state.cache_id, cache_id);
            state.connections.clear();
            (replaced, replaced_cache_id, attached.len())
        };
        if let Some(metrics) = &self.metrics {
            for _ in 0..failed {
                metrics.record_failure();
            }
        }
        let deregistered = self.transport.deregister(&replaced);
        let replaced_sessions: Vec<String> = replaced
            .iter()
            .map(|engine| engine.session_id().to_owned())
            .collect();
        let fresh_session = self.transport.session_id();
        let mut state = self.state();
        state.revoked_engines.extend(replaced);
        // The rooms whose manifests named the replaced engine.
        let named: Vec<u64> = state
            .admissions
            .keys()
            .copied()
            .filter(|room| state.attached_caches.get(room) == Some(&replaced_cache_id))
            .collect();
        match deregistered {
            Ok(()) => {
                tracing::warn!(
                    rank = self.engine_rank,
                    retired_rooms = ?named,
                    failed_attached = failed,
                    ?replaced_sessions,
                    fresh_session,
                    "event=ts_decode_revocation_fenced"
                );
                // Marked under the lock that computed them: a room that
                // retires after this is cleared from `revoked` too.
                state.revoked.extend(named);
                revocation.end(&mut state, None);
            }
            Err(error) => {
                state.revocation_unfenced = true;
                tracing::error!(
                    rank = self.engine_rank,
                    error,
                    pinned_rooms = ?named,
                    failed_attached = failed,
                    ?replaced_sessions,
                    fresh_session,
                    detail = "the replaced engine may still accept writes with its old keys: \
                              these rooms keep waiting for cancellation acknowledgments and \
                              this rank never releases its memory",
                    "event=ts_decode_revocation_unfenced"
                );
                let retry_at = revocation.retry_at;
                revocation.end(&mut state, Some(retry_at));
            }
        }
    }

    fn watch(&self, interval: Duration, max_failures: u32, stop: &mpsc::Receiver<()>) {
        let mut failures: HashMap<String, u32> = HashMap::new();
        let mut resume = 0;
        while !sleep_or_stop(stop, interval) {
            let addrs: Vec<String> = {
                let state = self.state();
                failures.retain(|addr, _| state.routes.contains_key(addr));
                state.routes.keys().cloned().collect()
            };
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
            resume = self.announce_waiting(resume);
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
        // Held through the release: a revocation installs its engine under it.
        let state = self.shared.state();
        if state.revoking {
            return Err(
                "cannot release the cache while a revocation replaces its transfer engine"
                    .to_owned(),
            );
        }
        if state.revocation_unfenced {
            tracing::error!(
                rank = self.shared.engine_rank,
                detail = "a revocation could not deregister the cache from the engine it \
                          replaced (see event=ts_decode_revocation_unfenced)",
                "event=ts_decode_release_refused_unfenced"
            );
            return Err(
                "cannot release the cache: a replaced transfer engine kept it registered"
                    .to_owned(),
            );
        }
        let mut receiving: Vec<u64> = state.tracked.iter().copied().collect();
        if !receiving.is_empty() {
            receiving.sort_unstable();
            return Err(format!(
                "cannot release the cache while KV receives are in flight: rooms {receiving:?}"
            ));
        }
        self.shared.transport.release_regions()
    }

    /// A wake re-mapped the cache: replace both its transfer engine and its
    /// identity. The next request registers this incarnation even when the
    /// engine reuses an earlier transport address.
    pub fn restore_registration(&self) -> Result<(), String> {
        let cache_id = crate::new_instance_id()
            .map_err(StartError::CacheIdentity)
            .map_err(|error| error.to_string())?;
        let mut state = self.shared.state();
        self.shared.transport.restore_regions(true)?;
        state.cache_id = cache_id;
        state.connections.clear();
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
            tracing::debug!(
                rid,
                room,
                prefix_tokens = rows.extend_prefix_lens[index],
                prompt_tokens = rows.prefill_lengths[index],
                request_pool_index = rows.request_pool_indices[index],
                "decode landing manifest prepared"
            );
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
        let mut state = self.shared.state();
        if !state.tracked.contains(&room)
            || state.cancellations.contains_key(&room)
            || state.rooms.get(room) == Some(TransferPoll::Failed)
        {
            // The loop ended the request (its transfer failed) after it
            // planned this admission: its pages go back, so no Prefill
            // rank may write them.
            tracing::debug!(rid, room, "not pre-allocating an ended request");
            return;
        }
        state.admissions.insert(room, request_pool_index);
        state.waiting_since.insert(room, Instant::now());
        state.rooms.update(room, TransferPoll::WaitingForInput);
        let targets = state.targets.get(&room).cloned();
        let Some(targets) = targets else {
            tracing::warn!(
                rid,
                room,
                "a request without a route reached its remote prefill"
            );
            return;
        };
        for target in targets.iter() {
            if let Err(error) = self.shared.transport.sockets.send(
                &target.host,
                target.port,
                wire::preallocation_frames(
                    room,
                    &state.cache_id,
                    (!target.is_dummy).then_some(manifest),
                ),
            ) {
                // Some targets may already have their manifest: keep the
                // admission and let the usual cancellation fence retire it.
                state.fail(room, error.to_string());
                return;
            }
        }
        drop(state);
        tracing::debug!(
            rid,
            room,
            request_pool_index,
            targets = targets.len(),
            manifest_bytes = manifest.len(),
            "event=ts_decode_manifest_queued"
        );
    }
}

/// One request's transfer as the scheduling loop tracks it.
struct Receiver {
    room: u64,
    addr: String,
    started: Instant,
    /// When every Prefill rank's notice had arrived for an announced room:
    /// the admission timeout restarts there.
    noticed: Option<Instant>,
    local: TransferPoll,
    concluded: Option<TransferPoll>,
    draining: Option<Draining>,
}

/// A Decode rank's PD transfer peer.
pub struct DecodePeer {
    shared: Arc<Shared>,
    connect: HashMap<String, mpsc::Sender<u64>>,
    stop: Arc<AtomicBool>,
    _heartbeat_stop: mpsc::Sender<()>,
    waiting_timeout: Duration,
    /// Tracked requests, in registration order (every replica member
    /// registers the same requests in the same order).
    order: Vec<String>,
    receivers: HashMap<String, Receiver>,
    timeout_logged: bool,
    drain_logged_at: Option<Instant>,
}

impl DecodePeer {
    /// Start the transfer engine, register the cache, open the status
    /// socket and start the threads.
    pub fn start(config: DecodeConfig) -> Result<(Self, DecodeOutbox), StartError> {
        let layout = Contract::from_wire(&config.layout)?;
        let cache_id = crate::new_instance_id().map_err(StartError::CacheIdentity)?;
        let (incoming, messages) = mpsc::channel::<Vec<bytes::Bytes>>();
        let transport = Transport::start(&config.peer, incoming)?;
        let shared = Arc::new(Shared {
            transport,
            state: Mutex::new(State {
                cache_id,
                ..State::default()
            }),
            local_ip: config.peer.local_ip.clone(),
            engine_rank: config.peer.engine_rank,
            arena_base: config.arena_base,
            layout_wire: config.layout,
            layout,
            tp_size: config.tp_size,
            tp_rank: config.tp_rank,
            metrics: config.peer.metrics.clone(),
            registration_replied: Condvar::new(),
            next_registration_id: AtomicU64::new(0),
            revocation_finished: Condvar::new(),
        });
        let stop = Arc::new(AtomicBool::new(false));
        {
            let shared = shared.clone();
            let stop = stop.clone();
            thread::Builder::new()
                .name("ts-pd-status".to_owned())
                .spawn(move || loop {
                    match messages.recv_timeout(SOCKET_POLL) {
                        Ok(frames) if wire::is_registration_ack(&frames) => {
                            match wire::registration_ack(&frames) {
                                Ok((id, cache_id, rank)) => shared.reply_to_registration(id, &cache_id, rank, None),
                                Err(error) => tracing::error!(%error, "rejecting a malformed registration acknowledgment"),
                            }
                        }
                        Ok(frames) if wire::is_registration_reject(&frames) => {
                            match wire::registration_reject(&frames) {
                                Ok((id, cache_id, rank, reason)) => shared.reply_to_registration(id, &cache_id, rank, Some(reason)),
                                Err(error) => tracing::error!(%error, "rejecting a malformed registration rejection"),
                            }
                        }
                        Ok(frames) if wire::is_cancellation_ack(&frames) => {
                            match wire::cancellation_ack(&frames) {
                                Ok((room, cache_id, rank)) => shared.state().acknowledge_cancellation(room, &cache_id, rank),
                                Err(error) => tracing::error!(%error, "rejecting a malformed cancellation acknowledgment"),
                            }
                        }
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
                connect: HashMap::new(),
                stop,
                _heartbeat_stop: heartbeat_stop,
                waiting_timeout: config.waiting_timeout,
                order: Vec::new(),
                receivers: HashMap::new(),
                timeout_logged: false,
                drain_logged_at: None,
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

    /// The current cache incarnation, independent of its transfer address.
    pub fn cache_id(&self) -> String {
        self.shared.state().cache_id.clone()
    }

    /// Dropping a lost address's sender lets its worker drain terminal rooms
    /// and exit. A first lookup has no route yet, so retain its worker until
    /// the pending request settles; otherwise each poll could fork its FIFO.
    fn reap_connect_workers(&mut self) {
        let state = self.shared.state();
        let pending: HashSet<&str> = self
            .receivers
            .values()
            .filter(|receiver| {
                receiver.concluded.is_none() && state.awaiting_connection(receiver.room)
            })
            .map(|receiver| receiver.addr.as_str())
            .collect();
        self.connect
            .retain(|addr, _| state.routes.contains_key(addr) || pending.contains(addr.as_str()));
    }

    /// Whether a failed receive has waited the whole receive budget for its
    /// writers to acknowledge the cancellation while its manifest named the
    /// current transfer engine: only a revocation can retire it now.
    fn needs_revocation(&self, now: Instant) -> bool {
        let state = self.shared.state();
        self.receivers.values().any(|receiver| {
            receiver
                .draining
                .as_ref()
                .is_some_and(|drain| now - drain.since >= self.waiting_timeout)
                && !state.cancellation_retired(receiver.room)
                && state.admissions.contains_key(&receiver.room)
                && state.attached_caches.get(&receiver.room) == Some(&state.cache_id)
        })
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
        self.reap_connect_workers();
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
                    started: Instant::now(),
                    noticed: None,
                    local: TransferPoll::Bootstrapping,
                    concluded: None,
                    draining: None,
                },
            )
            .is_none()
        {
            self.order.push(rid.to_owned());
        }
        if !self.connect.contains_key(&addr) {
            let (sender, rooms) = mpsc::channel();
            let shared = self.shared.clone();
            let endpoint = addr.clone();
            thread::Builder::new()
                .name("ts-pd-connect".to_owned())
                .spawn(move || {
                    while let Ok(room) = rooms.recv() {
                        if let Err(reason) = shared.connect(room, &endpoint) {
                            shared.state().fail(room, reason);
                        }
                    }
                })
                .map_err(|error| PdError::Peer(error.to_string()))?;
            self.connect.insert(addr.clone(), sender);
        }
        self.connect[&addr]
            .send(room)
            .map_err(|_| PdError::Peer("the PD connect thread stopped".to_owned()))
    }

    fn poll(&mut self, mesh: &mut Mesh, replica: &Group) -> Result<Vec<PdEvent>, PdError> {
        if self.order.is_empty() {
            self.reap_connect_workers();
            return Ok(Vec::new());
        }
        let now = Instant::now();
        let mut polls = Vec::with_capacity(self.order.len());
        {
            let mut state = self.shared.state();
            for rid in &self.order {
                let receiver = self.receivers.get_mut(rid).expect("tracked");
                // An announced room is reported connecting until its
                // Prefill ranks' notices arrived; they restart its wait.
                let awaiting_notice = state.awaiting_notice(receiver.room);
                if !awaiting_notice && state.announced.contains(&receiver.room) {
                    receiver.noticed.get_or_insert(now);
                }
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
                        TransferPoll::Bootstrapping | TransferPoll::Bootstrapped
                            if now - receiver.noticed.unwrap_or(receiver.started)
                                >= self.waiting_timeout =>
                        {
                            state.rooms.record_failure(
                                receiver.room,
                                "Decode timed out before publishing its cache manifest".to_owned(),
                            );
                            if let Some(metrics) = &self.shared.metrics {
                                metrics.record_timeout("waiting_for_admission");
                            }
                            receiver.concluded = Some(TransferPoll::Failed);
                            TransferPoll::Failed
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
                        TransferPoll::Bootstrapped if awaiting_notice => {
                            TransferPoll::Bootstrapping
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

        // A failure report, timeout or client abort is not a remote-write
        // fence. Every replica member must cancel every target before the
        // scheduler may recycle any rank's destination pages.
        let mut retired = Vec::with_capacity(self.order.len());
        let mut has_failure = false;
        for (rid, &poll) in self.order.iter().zip(&agreed) {
            let receiver = self.receivers.get_mut(rid).expect("tracked");
            let failed = poll == TransferPoll::Failed as i64;
            if failed {
                has_failure = true;
                receiver.concluded = Some(TransferPoll::Failed);
                self.shared.cancel(receiver.room);
            }
            retired.push(i64::from(
                !failed || self.shared.state().cancellation_retired(receiver.room),
            ));
        }
        let retired = if replica.size() > 1 && has_failure {
            mesh.all_reduce(replica, &retired, Reduce::Min)?
        } else {
            retired
        };

        let mut events = Vec::new();
        let mut ended = Vec::new();
        for ((rid, &poll), &retired) in self.order.iter().zip(&agreed).zip(&retired) {
            let receiver = self.receivers.get_mut(rid).expect("tracked");
            let poll = TransferPoll::from_wire(poll).unwrap_or(TransferPoll::Failed);
            if receiver.local == TransferPoll::Bootstrapping && poll == TransferPoll::Bootstrapped {
                receiver.local = TransferPoll::Bootstrapped;
                events.push(PdEvent::Bootstrapped(rid.clone()));
            } else if poll == TransferPoll::Failed {
                if receiver.local != TransferPoll::Failed {
                    receiver.local = TransferPoll::Failed;
                    receiver.draining = Some(Draining {
                        rid: rid.clone(),
                        room: receiver.room,
                        since: now,
                    });
                    let reason = self
                        .shared
                        .state()
                        .rooms
                        .failure(receiver.room)
                        .map(str::to_owned);
                    tracing::warn!(
                        rid,
                        room = receiver.room,
                        bootstrap_room = receiver.room,
                        rank = self.shared.engine_rank,
                        reason,
                        "event=ts_decode_kv_transfer_failed"
                    );
                    events.push(PdEvent::Failed(rid.clone()));
                }
                if retired != 0 {
                    events.push(PdEvent::Stopped(rid.clone()));
                    ended.push(rid.clone());
                }
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
                drop(state);
                tracing::info!(
                    rid,
                    bootstrap_room = receiver.room,
                    rank = self.shared.engine_rank,
                    cached_tokens,
                    output_tokens = 1,
                    "event=ts_decode_kv_transfer_success"
                );
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
        if self.needs_revocation(now) {
            Shared::start_revocation(&self.shared, now, self.waiting_timeout);
        }
        self.reap_connect_workers();
        let stalled = stalled_drains(
            self.order
                .iter()
                .filter_map(|rid| self.receivers[rid].draining.as_ref()),
            &mut self.drain_logged_at,
            now,
        );
        if !stalled.is_empty() {
            tracing::warn!(
                ?stalled,
                "PD destination pages are waiting for cancellation acknowledgments"
            );
        }
        Ok(events)
    }

    fn record_usage(&mut self, _rid: &str, _cached_tokens: i64, _bootstrap_logprob: Option<f64>) {}

    fn has_in_flight(&self) -> bool {
        !self.order.is_empty()
    }

    fn cancel(&mut self, rid: &str) {
        if let Some(receiver) = self.receivers.get_mut(rid) {
            receiver.concluded = Some(TransferPoll::Failed);
            self.shared.cancel(receiver.room);
        }
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
            deferred: false,
        }
    }

    /// Track `room` and connect it to Prefill ranks 0 and 1 at `addr`.
    fn connected(state: &mut State, room: u64, addr: &str) {
        state.tracked.insert(room);
        state.rooms.update(room, TransferPoll::Bootstrapping);
        assert!(state.attach(room, addr, &route(), Arc::new(Vec::new())));
    }

    #[test]
    fn a_notice_is_kept_before_the_connection_and_needs_every_expected_rank() {
        let mut state = State::default();
        let deferred = PrefillRoute {
            deferred: true,
            ..route()
        };
        // Both ranks' notices arrive while the room still connects: they do
        // not connect it, and it needs none once connected.
        state.tracked.insert(1);
        state.rooms.update(1, TransferPoll::Bootstrapping);
        for rank in [0, 1] {
            state.apply(StatusMessage::bare(1, TransferPoll::Bootstrapped, rank));
        }
        assert_eq!(state.rooms.get(1), Some(TransferPoll::Bootstrapping));
        assert!(state.attach(1, "p:1", &deferred, Arc::new(Vec::new())));
        assert_eq!(state.rooms.get(1), Some(TransferPoll::Bootstrapped));
        assert!(!state.awaiting_notice(1));
        // One rank's notice is not enough.
        state.tracked.insert(2);
        state.rooms.update(2, TransferPoll::Bootstrapping);
        assert!(state.attach(2, "p:1", &deferred, Arc::new(Vec::new())));
        state.apply(StatusMessage::bare(2, TransferPoll::Bootstrapped, 0));
        assert!(state.awaiting_notice(2));
        state.apply(StatusMessage::bare(2, TransferPoll::Bootstrapped, 1));
        assert!(!state.awaiting_notice(2));
        // A room of a node that allocates early waits for none.
        connected(&mut state, 3, "p:1");
        assert!(!state.awaiting_notice(3));
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
    fn a_late_connection_does_not_revive_a_failed_room() {
        let mut state = State::default();
        state.tracked.insert(1);
        state.rooms.update(1, TransferPoll::Bootstrapping);
        state.fail(1, "bootstrap failed");
        assert!(!state.attach(1, "prefill:80", &route(), Arc::new(Vec::new())));
        assert_eq!(state.rooms.get(1), Some(TransferPoll::Failed));
        assert!(state.targets.is_empty());
    }

    #[test]
    fn failure_and_lost_routes_do_not_replace_cancellation_acknowledgments() {
        let mut state = State {
            cache_id: "cache".to_owned(),
            ..State::default()
        };
        connected(&mut state, 1, "prefill:80");
        state.targets.insert(
            1,
            Arc::new(
                (0..2)
                    .map(|rank| Target {
                        host: "prefill".to_owned(),
                        port: 80,
                        rank,
                        is_dummy: false,
                    })
                    .collect(),
            ),
        );
        state.admissions.insert(1, 3);
        state.apply(StatusMessage::bare(1, TransferPoll::Failed, 0));
        state.cancellations.insert(
            1,
            Cancellation {
                cache_id: "cache".to_owned(),
                ..Cancellation::default()
            },
        );
        assert!(!state.cancellation_retired(1));
        state.acknowledge_cancellation(1, "cache", 0);
        assert!(!state.cancellation_retired(1));
        state.lose_node("prefill:80");
        assert!(!state.cancellation_retired(1));
        state.apply(success(1, 1, 7));
        assert_eq!(state.rooms.get(1), Some(TransferPoll::Failed));
        assert!(!state.cancellation_retired(1));
        state.acknowledge_cancellation(1, "cache", 1);
        assert!(state.cancellation_retired(1));
    }

    #[test]
    fn cancellation_needs_every_target_including_dummy_and_the_current_cache() {
        let mut state = State {
            cache_id: "current".to_owned(),
            ..State::default()
        };
        connected(&mut state, 1, "prefill:80");
        state.targets.insert(
            1,
            Arc::new(vec![
                Target {
                    rank: 0,
                    host: "p0".to_owned(),
                    port: 1,
                    is_dummy: false,
                },
                Target {
                    rank: 1,
                    host: "p1".to_owned(),
                    port: 2,
                    is_dummy: true,
                },
            ]),
        );
        state.admissions.insert(1, 7);
        state.cancellations.insert(
            1,
            Cancellation {
                cache_id: "current".to_owned(),
                ..Cancellation::default()
            },
        );
        state.acknowledge_cancellation(1, "old", 0);
        state.acknowledge_cancellation(1, "current", 9);
        assert!(!state.cancellation_retired(1));
        state.acknowledge_cancellation(1, "current", 0);
        state.acknowledge_cancellation(1, "current", 0);
        state.apply(success(1, 1, 7));
        assert!(!state.cancellation_retired(1));
        assert_ne!(state.rooms.get(1), Some(TransferPoll::Success));
        state.acknowledge_cancellation(1, "current", 1);
        assert!(state.cancellation_retired(1));
        state.clear(1, "prefill:80");
        state.acknowledge_cancellation(1, "current", 0);
        assert!(state.cancellations.is_empty());
    }

    #[test]
    fn a_cancellation_is_acknowledged_for_the_cache_its_route_registered() {
        let mut state = State {
            cache_id: "fresh".to_owned(),
            ..State::default()
        };
        connected(&mut state, 1, "prefill:80");
        state.targets.insert(
            1,
            Arc::new(vec![Target {
                rank: 0,
                host: "p0".to_owned(),
                port: 1,
                is_dummy: false,
            }]),
        );
        state.admissions.insert(1, 3);
        state.cancellations.insert(
            1,
            Cancellation {
                cache_id: "replaced".to_owned(),
                ..Cancellation::default()
            },
        );
        state.acknowledge_cancellation(1, "fresh", 0);
        assert!(!state.cancellation_retired(1));
        state.acknowledge_cancellation(1, "replaced", 0);
        assert!(state.cancellation_retired(1));
    }

    #[test]
    fn a_revoked_room_retires_without_acknowledgments() {
        let mut state = State::default();
        connected(&mut state, 1, "prefill:80");
        state.targets.insert(
            1,
            Arc::new(vec![Target {
                rank: 0,
                host: "p0".to_owned(),
                port: 1,
                is_dummy: false,
            }]),
        );
        state.admissions.insert(1, 3);
        state.cancellations.insert(1, Cancellation::default());
        assert!(!state.cancellation_retired(1));
        state.revoked.insert(1);
        assert!(state.cancellation_retired(1));
        state.clear(1, "prefill:80");
        assert!(state.revoked.is_empty());
    }

    #[test]
    fn cancelling_before_admission_blocks_a_late_connection_without_waiting_for_ack() {
        let mut state = State::default();
        state.tracked.insert(1);
        state.cancellations.insert(1, Cancellation::default());
        assert!(state.cancellation_retired(1));
        assert!(!state.attach(1, "prefill:80", &route(), Arc::new(Vec::new())));
        assert!(state.targets.is_empty());
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

#[cfg(test)]
#[path = "../tests/unit/decode_lifecycle.rs"]
mod lifecycle_tests;
