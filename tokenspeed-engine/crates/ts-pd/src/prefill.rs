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

//! A Prefill rank's peer: Decode ranks register with it and pre-allocate
//! each request's pages; once the prompt's last chunk commits, a transfer
//! worker writes its KV and reports to every destination.
//!
//! Threads: one handles the control socket (registrations and
//! pre-allocations), a few run transfers, sharded by destination session so
//! one session's transfers stay ordered. The scheduling loop only registers
//! requests, records usage and polls.

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Condvar, Mutex, MutexGuard,
    },
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use ts_collective::{Group, Mesh, Reduce};
use ts_control::pd::{BootstrapInfo, PdError, PdEvent, TransferPeer};

use crate::{
    http,
    status::{RoomTable, TransferPoll},
    wire::{self, StatusMessage},
    Copy, PeerConfig, StartError, Transport,
};

/// A request still bootstrapping this long after registration is reported.
const BOOTSTRAP_STALL: Duration = Duration::from_secs(10);

/// How often an idle control-socket thread checks for shutdown.
const SOCKET_POLL: Duration = Duration::from_millis(100);

/// A Decode rank's planned registration with this rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Registration {
    /// Where the Decode rank's control socket listens.
    pub endpoint: String,
    /// Its port.
    pub port: u16,
    /// Its transfer session.
    pub session_id: String,
    /// The Decode node's attention TP size.
    pub decode_tp_size: i64,
    /// The Decode rank's attention TP rank.
    pub decode_tp_rank: i64,
    /// This rank sends it no data and only joins the status agreement.
    pub is_dummy: bool,
    /// The Decode ranks one room's pre-allocations come from, sorted.
    pub expected_decode_ranks: Vec<i64>,
}

/// The layout-dependent half of a Prefill rank's PD work, from the data
/// plane (`HostPrefillPeer`). Called from the peer's threads, never from the
/// scheduling loop.
pub trait PrefillPlanner: Send + Sync {
    /// Plan a Decode rank's registration (its frames as received) and keep
    /// its route. Fails when the registration is malformed, misrouted, or
    /// conflicts with the session's registration on record.
    fn prepare_registration(&self, frames: &[Bytes]) -> Result<Registration, String>;

    /// Forget a rejected session's registration.
    fn reject_registration(&self, session_id: &str);

    /// Validate a pre-allocation against its session's registration and keep
    /// its manifest. Returns whether it is a dummy (no manifest).
    fn accept_preallocation(&self, frames: &[Bytes]) -> Result<bool, String>;

    /// The copies that move `room`'s blocks to `session_id`.
    fn plan_send(&self, room: u64, session_id: &str) -> Result<Vec<Copy>, String>;

    /// Layerwise streaming: an armed chunk's copies to each of `sessions`,
    /// per transfer interval; the planner forgets the chunk.
    fn plan_layerwise(
        &self,
        chunk_id: u64,
        sessions: &[String],
    ) -> Result<Vec<Vec<Vec<Copy>>>, String>;

    /// Forget a room's manifests.
    fn drop_room(&self, room: u64);
}

/// A Prefill rank's settings.
#[derive(Debug, Clone)]
pub struct PrefillConfig {
    /// What both roles share.
    pub peer: PeerConfig,
    /// This rank in Decode's completion-tracking rank space.
    pub status_rank: i64,
    /// The bootstrap server's registration route.
    pub bootstrap_url: String,
    /// What this rank registers there, but its control port.
    pub bootstrap_payload: serde_json::Value,
    /// A request whose Decode ranks have not all pre-allocated this long
    /// after registration fails.
    pub bootstrap_timeout: Duration,
    /// After a transfer to a session fails, its later transfers fail fast
    /// for this long; zero disables.
    pub failed_session_ttl: Duration,
    /// Transfer workers.
    pub transfer_workers: usize,
    /// Layerwise streaming: the pinned host word the data plane updates with
    /// the last producer step whose KV landed (`StepCounter`); `None`
    /// without it.
    pub layerwise_step_address: Option<usize>,
}

/// One destination of a room.
#[derive(Debug, Clone)]
struct Destination {
    session_id: String,
    is_dummy: bool,
}

/// Work for a transfer worker.
enum Chunk {
    /// A room's whole KV is written: send it.
    Full {
        room: u64,
        bootstrap_token: i64,
        spec_candidate_ids: Option<Vec<i32>>,
    },
    /// Layerwise: one prefill chunk's KV, sent an interval at a time as the
    /// forward writes it; the last chunk then reports the room done, once
    /// its bootstrap payload arrived.
    Layerwise {
        room: u64,
        chunk_id: u64,
        /// The producer step each interval's KV is complete at.
        target_steps: Vec<u64>,
        is_last: bool,
    },
}

impl Chunk {
    fn room(&self) -> u64 {
        match self {
            Chunk::Full { room, .. } | Chunk::Layerwise { room, .. } => *room,
        }
    }
}

/// Producer steps count modulo this (`StepCounter.COUNT_NUM_MAX`).
const STEP_COUNT_MAX: u64 = 1 << 62;

/// The counter at `current` is strictly past `target`, across the counter's
/// wrap: a step more than half the range ahead counts as behind.
fn step_ready(current: u64, target: u64) -> bool {
    target != current && (target + STEP_COUNT_MAX - current) % STEP_COUNT_MAX > STEP_COUNT_MAX / 2
}

/// The last landed producer step: a plain load of the data plane's pinned
/// host word, which the device updates asynchronously.
#[allow(unsafe_code)]
fn ready_step(address: usize) -> u64 {
    // SAFETY: `address` is the pinned host tensor the data plane allocated
    // for the engine's lifetime; an aligned 8-byte load of it is atomic.
    unsafe { std::ptr::read_volatile(address as *const i64) as u64 }
}

#[derive(Default)]
struct State {
    rooms: RoomTable,
    registrations: HashMap<String, Registration>,
    /// Sessions whose registration was rejected: their reply route.
    rejected: HashMap<String, (String, u16)>,
    destinations: HashMap<u64, Vec<Destination>>,
    failed_sessions: HashMap<String, Instant>,
    cached_tokens: HashMap<u64, i64>,
    bootstrap_logprobs: HashMap<u64, f64>,
    /// Rooms whose KV streams layerwise.
    layerwise_rooms: HashSet<u64>,
    /// Layerwise rooms' bootstrap payloads, from their remote decode.
    prefill_metadata: HashMap<u64, (i64, Option<Vec<i32>>)>,
}

impl State {
    fn route_of(&self, session_id: &str) -> Option<(String, u16)> {
        self.registrations
            .get(session_id)
            .map(|registration| (registration.endpoint.clone(), registration.port))
            .or_else(|| self.rejected.get(session_id).cloned())
    }

    fn ended(&self, room: u64) -> bool {
        matches!(self.rooms.get(room), None | Some(TransferPoll::Failed))
    }

    /// One complete, unique Decode-rank set.
    fn validate_fanout(&self, destinations: &[Destination]) -> Result<(), String> {
        let registrations = destinations
            .iter()
            .map(|destination| {
                self.registrations
                    .get(&destination.session_id)
                    .ok_or_else(|| {
                        "CachePD request references an unregistered Decode session".to_owned()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sizes: BTreeSet<_> = registrations.iter().map(|r| r.decode_tp_size).collect();
        let expected: BTreeSet<_> = registrations
            .iter()
            .map(|r| r.expected_decode_ranks.clone())
            .collect();
        if sizes.len() != 1 || expected.len() != 1 {
            return Err("Cache-transfer room has inconsistent TP metadata".to_owned());
        }
        let expected: BTreeSet<i64> = expected
            .into_iter()
            .next()
            .expect("one set")
            .into_iter()
            .collect();
        let actual: BTreeSet<i64> = registrations.iter().map(|r| r.decode_tp_rank).collect();
        if destinations.len() != expected.len() || actual != expected {
            return Err(
                "Cache-transfer room destination ranks disagree with the typed route plan"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

struct Shared {
    transport: Arc<Transport>,
    planner: Arc<dyn PrefillPlanner>,
    state: Mutex<State>,
    status_rank: i64,
    failed_session_ttl: Duration,
    metrics: Option<Arc<ts_metrics::KvTransferMetrics>>,
    layerwise_step_address: Option<usize>,
    /// Signaled when a layerwise room's bootstrap payload arrives.
    metadata_ready: Condvar,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("the PD state lock is poisoned")
    }

    fn notify(&self, (endpoint, port): (String, u16), message: &StatusMessage) {
        self.transport
            .sockets
            .send(&endpoint, port, message.to_frames());
    }

    fn notify_failed(&self, route: (String, u16), room: u64) {
        self.notify(
            route,
            &StatusMessage::bare(room, TransferPoll::Failed, self.status_rank),
        );
    }

    /// Fail a room and tell every destination that pre-allocated for it
    /// (`abort_room`).
    fn abort_room(&self, room: u64, reason: &str) {
        let routes: Vec<_> = {
            let mut state = self.state();
            // Once per room: a room's later chunks fail here again.
            let counted = state.rooms.get(room) == Some(TransferPoll::Failed)
                || state.rooms.has_failure(room);
            if let (false, Some(metrics)) = (counted, &self.metrics) {
                metrics.record_failure();
            }
            state.rooms.record_failure(room, reason);
            state.rooms.update(room, TransferPoll::Failed);
            let destinations = state.destinations.remove(&room).unwrap_or_default();
            destinations
                .iter()
                .filter(|destination| !destination.is_dummy)
                .filter_map(|destination| {
                    let route = state.route_of(&destination.session_id);
                    if route.is_none() {
                        tracing::error!(
                            room,
                            session = destination.session_id,
                            "cannot notify Decode of a failed room: unregistered session"
                        );
                    }
                    route
                })
                .collect()
        };
        for route in routes {
            self.notify_failed(route, room);
        }
        self.planner.drop_room(room);
    }

    fn handle(&self, frames: &[Bytes]) {
        if wire::is_registration(frames) {
            self.handle_registration(frames);
        } else {
            self.handle_preallocation(frames);
        }
    }

    fn handle_registration(&self, frames: &[Bytes]) {
        match self.planner.prepare_registration(frames) {
            Ok(registration) => {
                let mut state = self.state();
                let session_id = registration.session_id.clone();
                state.rejected.remove(&session_id);
                state.failed_sessions.remove(&session_id);
                state.registrations.insert(session_id, registration);
            }
            Err(reason) => {
                tracing::error!(reason, "rejecting a CachePD Decode registration");
                if let Ok((endpoint, port, session_id)) = wire::registration_route(frames) {
                    self.reject_session(endpoint, port, &session_id, &reason);
                }
            }
        }
    }

    /// Invalidate a session and fail the rooms that reference it.
    fn reject_session(&self, endpoint: String, port: u16, session_id: &str, reason: &str) {
        let affected: Vec<u64> = self
            .state()
            .destinations
            .iter()
            .filter(|(_, destinations)| {
                destinations
                    .iter()
                    .any(|destination| destination.session_id == session_id)
            })
            .map(|(&room, _)| room)
            .collect();
        for room in affected {
            self.abort_room(
                room,
                &format!("Decode session {session_id} registration rejected: {reason}"),
            );
        }
        {
            let mut state = self.state();
            state.registrations.remove(session_id);
            state
                .rejected
                .insert(session_id.to_owned(), (endpoint, port));
        }
        self.planner.reject_registration(session_id);
    }

    fn handle_preallocation(&self, frames: &[Bytes]) {
        let (room, session_id) = match wire::preallocation_route(frames) {
            Ok(route) => route,
            Err(error) => {
                tracing::error!(%error, "rejecting a malformed CachePD pre-allocation");
                return;
            }
        };
        if let Err(reason) = self.accept_preallocation(room, &session_id, frames) {
            tracing::error!(room, reason, "rejecting CachePD pre-allocation metadata");
            self.abort_room(room, &format!("invalid pre-allocation metadata: {reason}"));
            match self.state().route_of(&session_id) {
                Some(route) => self.notify_failed(route, room),
                None => tracing::error!(room, "cannot notify a malformed Decode peer"),
            }
        }
    }

    fn accept_preallocation(
        &self,
        room: u64,
        session_id: &str,
        frames: &[Bytes],
    ) -> Result<(), String> {
        let registration = self
            .state()
            .registrations
            .get(session_id)
            .cloned()
            .ok_or("CachePD request references an unregistered Decode session")?;
        let route = (registration.endpoint.clone(), registration.port);
        if self.state().rooms.get(room) == Some(TransferPoll::Failed) {
            self.notify_failed(route, room);
            return Ok(());
        }
        let is_dummy = self.planner.accept_preallocation(frames)?;
        let mut state = self.state();
        let mut destinations = state.destinations.get(&room).cloned().unwrap_or_default();
        if destinations
            .iter()
            .any(|destination| destination.session_id == session_id)
        {
            return Err("duplicate CachePD pre-allocation for Decode session".to_owned());
        }
        destinations.push(Destination {
            session_id: session_id.to_owned(),
            is_dummy,
        });
        let expected = registration.expected_decode_ranks.len();
        if destinations.len() > expected {
            return Err("pre-allocation exceeds destination fanout".to_owned());
        }
        let complete = destinations.len() == expected;
        if complete {
            state.validate_fanout(&destinations)?;
        }
        // A transfer worker may have failed the room since the check above.
        if state.rooms.get(room) == Some(TransferPoll::Failed) {
            drop(state);
            self.notify_failed(route, room);
            return Ok(());
        }
        state.destinations.insert(room, destinations);
        if complete {
            state.rooms.update(room, TransferPoll::Bootstrapped);
        }
        Ok(())
    }

    fn session_failed(&self, session_id: &str) -> bool {
        if self.failed_session_ttl.is_zero() {
            return false;
        }
        let mut state = self.state();
        match state.failed_sessions.get(session_id) {
            None => false,
            Some(at) if at.elapsed() < self.failed_session_ttl => true,
            Some(_) => {
                state.failed_sessions.remove(session_id);
                false
            }
        }
    }

    fn mark_session_failed(&self, session_id: &str) {
        if self.failed_session_ttl.is_zero() {
            return;
        }
        tracing::warn!(
            session = session_id,
            ttl = ?self.failed_session_ttl,
            "Decode session marked failed"
        );
        self.state()
            .failed_sessions
            .insert(session_id.to_owned(), Instant::now());
    }

    /// Run one chunk of transfer work.
    fn transfer(&self, chunk: &Chunk) -> Result<(), String> {
        match chunk {
            Chunk::Full {
                room,
                bootstrap_token,
                spec_candidate_ids,
            } => self.transfer_full(*room, *bootstrap_token, spec_candidate_ids),
            Chunk::Layerwise {
                room,
                chunk_id,
                target_steps,
                is_last,
            } => self.transfer_layerwise(*room, *chunk_id, target_steps, *is_last),
        }
    }

    /// The live destinations' registrations, failing if one's session failed.
    fn live_registrations(
        &self,
        destinations: &[Destination],
    ) -> Result<Vec<Registration>, String> {
        destinations
            .iter()
            .map(|destination| {
                if self.session_failed(&destination.session_id) {
                    return Err(format!(
                        "Decode session {} is not alive",
                        destination.session_id
                    ));
                }
                self.state()
                    .registrations
                    .get(&destination.session_id)
                    .cloned()
                    .ok_or_else(|| {
                        "CachePD request references an unregistered Decode session".to_owned()
                    })
            })
            .collect()
    }

    fn write_copies(&self, registration: &Registration, copies: &[Copy]) -> Result<(), String> {
        let started = Instant::now();
        if let Err(error) = self.transport.write(&registration.session_id, copies) {
            self.mark_session_failed(&registration.session_id);
            return Err(format!(
                "Mooncake transfer failed for {}:{}: {error}",
                registration.endpoint, registration.port
            ));
        }
        if let Some(metrics) = &self.metrics {
            metrics.observe_latency(started.elapsed().as_secs_f64());
        }
        Ok(())
    }

    /// Tell every destination the room's KV arrived, with its bootstrap
    /// payload, and mark it done.
    fn report_success(
        &self,
        room: u64,
        registrations: &[Registration],
        bootstrap_token: i64,
        spec_candidate_ids: &Option<Vec<i32>>,
    ) {
        let (cached_tokens, bootstrap_logprob) = {
            let state = self.state();
            if state.ended(room) {
                return;
            }
            (
                state.cached_tokens.get(&room).copied().unwrap_or(0),
                state.bootstrap_logprobs.get(&room).copied(),
            )
        };
        let message = StatusMessage {
            room,
            status: TransferPoll::Success,
            prefill_rank: self.status_rank,
            bootstrap_token,
            spec_candidate_ids: spec_candidate_ids.clone(),
            cached_tokens,
            bootstrap_logprob,
        };
        for registration in registrations {
            self.notify((registration.endpoint.clone(), registration.port), &message);
        }
        let mut state = self.state();
        state.rooms.update(room, TransferPoll::Success);
        state.destinations.remove(&room);
    }

    /// Wait until the producer step `target` landed, failing if the room ends or a
    /// destination's session fails first.
    fn wait_for_step(
        &self,
        address: usize,
        target: u64,
        room: u64,
        sessions: &[String],
    ) -> Result<(), String> {
        let mut next_session_check = Instant::now();
        loop {
            if self.state().ended(room) {
                return Err(format!(
                    "Cache-transfer layerwise wait aborted for room {room}"
                ));
            }
            let now = Instant::now();
            if !sessions.is_empty() && now >= next_session_check {
                if sessions.iter().any(|session| self.session_failed(session)) {
                    return Err(format!(
                        "Cache-transfer layerwise peer failed for room {room}"
                    ));
                }
                next_session_check = now + Duration::from_millis(100);
            }
            if step_ready(ready_step(address), target) {
                return Ok(());
            }
            thread::sleep(Duration::from_micros(100));
        }
    }

    /// Send one layerwise chunk an interval at a time; the last one reports
    /// the room done once its bootstrap payload arrived.
    fn transfer_layerwise(
        &self,
        room: u64,
        chunk_id: u64,
        target_steps: &[u64],
        is_last: bool,
    ) -> Result<(), String> {
        let address = self
            .layerwise_step_address
            .ok_or("Cache-transfer layerwise transfer has no producer step counter")?;
        let destinations: Vec<Destination> = self
            .state()
            .destinations
            .get(&room)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|destination| !destination.is_dummy)
            .collect();
        if destinations.is_empty() {
            return Err("CachePD transfer room has no destinations".to_owned());
        }
        let registrations = self.live_registrations(&destinations)?;
        let sessions: Vec<String> = destinations.iter().map(|d| d.session_id.clone()).collect();
        let plans = self.planner.plan_layerwise(chunk_id, &sessions)?;
        for (interval, &target) in target_steps.iter().enumerate() {
            self.wait_for_step(address, target, room, &sessions)?;
            // Every peer's copies of this interval are planned before any DMA.
            for (registration, plan) in registrations.iter().zip(&plans) {
                let copies = plan
                    .get(interval)
                    .ok_or("a layerwise plan has fewer intervals than the chunk")?;
                if !copies.is_empty() {
                    self.write_copies(registration, copies)?;
                }
            }
        }
        tracing::debug!(
            room,
            chunk_id,
            intervals = target_steps.len(),
            is_last,
            "layerwise chunk sent"
        );
        if !is_last {
            return Ok(());
        }
        // The final chunk's bootstrap payload comes with the remote decode.
        let (bootstrap_token, spec_candidate_ids) = loop {
            let mut state = self.state();
            if state.ended(room) {
                return Ok(());
            }
            if let Some(metadata) = state.prefill_metadata.remove(&room) {
                break metadata;
            }
            let (_state, _) = self
                .metadata_ready
                .wait_timeout(state, Duration::from_millis(100))
                .expect("the PD state lock is poisoned");
        };
        self.report_success(room, &registrations, bootstrap_token, &spec_candidate_ids);
        Ok(())
    }

    /// Write one room's KV to every destination and report.
    fn transfer_full(
        &self,
        room: u64,
        bootstrap_token: i64,
        spec_candidate_ids: &Option<Vec<i32>>,
    ) -> Result<(), String> {
        let destinations = self
            .state()
            .destinations
            .get(&room)
            .cloned()
            .filter(|destinations| !destinations.is_empty())
            .ok_or("CachePD transfer room has no destinations")?;
        if destinations.iter().any(|destination| destination.is_dummy) {
            // This rank only joins the status agreement for the room.
            let mut state = self.state();
            if state.rooms.get(room).is_some() {
                state.rooms.update(room, TransferPoll::Success);
                state.destinations.remove(&room);
            }
            return Ok(());
        }
        let registrations = self.live_registrations(&destinations)?;
        let mut prepared = Vec::with_capacity(destinations.len());
        for (destination, registration) in destinations.iter().zip(registrations) {
            let copies = self.planner.plan_send(room, &destination.session_id)?;
            prepared.push((registration, copies));
        }
        for (registration, copies) in &prepared {
            self.write_copies(registration, copies)?;
        }
        let registrations: Vec<Registration> = prepared
            .into_iter()
            .map(|(registration, _)| registration)
            .collect();
        self.report_success(room, &registrations, bootstrap_token, spec_candidate_ids);
        Ok(())
    }

    fn run_worker(&self, queue: &mpsc::Receiver<Chunk>) {
        while let Ok(chunk) = queue.recv() {
            if let Err(reason) = self.transfer(&chunk) {
                if self.state().ended(chunk.room()) {
                    // Another thread failed or dropped the room after it was
                    // queued: failing it again would replace its reason.
                    tracing::debug!(
                        room = chunk.room(),
                        reason,
                        "dropping an ended room's transfer"
                    );
                    continue;
                }
                tracing::error!(room = chunk.room(), reason, "CachePD transfer failed");
                self.abort_room(chunk.room(), &format!("CachePD transfer failed: {reason}"));
            }
        }
    }
}

/// Where the data plane hands a Prefill rank's peer its work, from the
/// forward thread; never waits.
#[derive(Clone)]
pub struct PrefillOutbox {
    shared: Arc<Shared>,
    queues: Arc<Vec<mpsc::Sender<Chunk>>>,
}

impl PrefillOutbox {
    /// A sleep releases the cache: deregister it from the transfer engine,
    /// refused while a request's KV may still go out.
    pub fn release_registration(&self) -> Result<(), String> {
        let rooms = self.shared.state().rooms.rooms();
        if !rooms.is_empty() {
            return Err(format!(
                "cannot release the cache while KV sends are in flight: rooms {rooms:?}"
            ));
        }
        self.shared.transport.release_regions()
    }

    /// A wake re-mapped the cache: register it again, on the same session
    /// (every write starts from this rank's own registration, so no peer
    /// holds its keys).
    pub fn restore_registration(&self) -> Result<(), String> {
        self.shared.transport.restore_regions(false)
    }

    /// A request's KV is written and its manifest kept: send it to its
    /// destinations (`add_transfer_request`).
    pub fn send(
        &self,
        rid: &str,
        room: u64,
        bootstrap_token: i64,
        spec_candidate_ids: Option<Vec<i32>>,
    ) {
        {
            let mut state = self.shared.state();
            if state.layerwise_rooms.contains(&room) {
                // Its KV already streams: only its final report waits on this.
                if !state.ended(room) {
                    state
                        .prefill_metadata
                        .insert(room, (bootstrap_token, spec_candidate_ids));
                    self.shared.metadata_ready.notify_all();
                }
                return;
            }
        }
        self.enqueue(
            rid,
            Chunk::Full {
                room,
                bootstrap_token,
                spec_candidate_ids,
            },
        );
    }

    /// Layerwise streaming: a prefill chunk of `room`'s request was armed
    /// (`send_layerwise`); its intervals are complete at `target_steps`.
    pub fn send_layerwise(
        &self,
        rid: &str,
        room: u64,
        chunk_id: u64,
        target_steps: Vec<u64>,
        is_last: bool,
    ) {
        self.shared.state().layerwise_rooms.insert(room);
        self.enqueue(
            rid,
            Chunk::Layerwise {
                room,
                chunk_id,
                target_steps,
                is_last,
            },
        );
    }

    /// Queue `chunk` on the worker its destinations shard to: one room's
    /// chunks always reach the same worker, in order.
    fn enqueue(&self, rid: &str, chunk: Chunk) {
        let room = chunk.room();
        let shard = {
            let state = self.shared.state();
            if state.ended(room) {
                tracing::debug!(rid, room, "the room already failed or ended");
                return;
            }
            let Some(destinations) = state.destinations.get(&room) else {
                // A dummy rank already succeeded, or the room was dropped.
                return;
            };
            let ports: u64 = destinations
                .iter()
                .filter_map(|destination| wire::session_port(&destination.session_id))
                .sum();
            (ports % self.queues.len() as u64) as usize
        };
        if self.queues[shard].send(chunk).is_err() {
            tracing::error!(rid, room, "the PD transfer workers stopped");
        }
    }

    /// A request reached its remote decode with no destination on record:
    /// fine for a room that already failed or ended, a broken invariant
    /// otherwise.
    pub fn send_unplanned(&self, rid: &str, room: u64) -> Result<(), String> {
        if self.shared.state().ended(room) {
            return Ok(());
        }
        Err(format!(
            "Cache-transfer destination metadata is unavailable for {rid} (room {room})"
        ))
    }
}

/// One request's transfer as the scheduling loop tracks it.
struct Sender {
    room: u64,
    started: Instant,
    local: TransferPoll,
    concluded: Option<TransferPoll>,
}

/// A Prefill rank's PD transfer peer.
pub struct PrefillPeer {
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    bootstrap_timeout: Duration,
    /// Tracked requests, in registration order (every replica member
    /// registers the same requests in the same order).
    order: Vec<String>,
    senders: HashMap<String, Sender>,
    stall_logged_at: Option<Instant>,
    timeout_logged: bool,
}

impl PrefillPeer {
    /// Start the transfer engine, register the cache, open the control
    /// socket, start the threads and register with the bootstrap server.
    pub fn start(
        config: PrefillConfig,
        planner: Arc<dyn PrefillPlanner>,
    ) -> Result<(Self, PrefillOutbox), StartError> {
        let (incoming, messages) = mpsc::channel();
        let transport = Transport::start(&config.peer, incoming)?;
        let port = transport.sockets.port();
        let shared = Arc::new(Shared {
            transport,
            planner,
            state: Mutex::new(State::default()),
            status_rank: config.status_rank,
            failed_session_ttl: config.failed_session_ttl,
            metrics: config.peer.metrics.clone(),
            layerwise_step_address: config.layerwise_step_address,
            metadata_ready: Condvar::new(),
        });
        let stop = Arc::new(AtomicBool::new(false));
        {
            let shared = shared.clone();
            let stop = stop.clone();
            thread::Builder::new()
                .name("ts-pd-bootstrap".to_owned())
                .spawn(move || loop {
                    match messages.recv_timeout(SOCKET_POLL) {
                        Ok(frames) => shared.handle(&frames),
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if stop.load(Ordering::Relaxed) {
                                return;
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => return,
                    }
                })?;
        }
        let mut queues = Vec::with_capacity(config.transfer_workers.max(1));
        for index in 0..config.transfer_workers.max(1) {
            let (queue, chunks) = mpsc::channel();
            let shared = shared.clone();
            thread::Builder::new()
                .name(format!("ts-pd-transfer-{index}"))
                .spawn(move || shared.run_worker(&chunks))?;
            queues.push(queue);
        }

        let mut payload = config.bootstrap_payload.clone();
        payload["rank_port"] = port.into();
        match http::put_json(&config.bootstrap_url, &payload, Duration::from_secs(5)) {
            Ok(()) => tracing::debug!("registered with the bootstrap server"),
            Err(error) => tracing::error!(
                url = config.bootstrap_url,
                error,
                "the Prefill rank failed to register with the bootstrap server"
            ),
        }
        let outbox = PrefillOutbox {
            shared: shared.clone(),
            queues: Arc::new(queues),
        };
        Ok((
            Self {
                shared,
                stop,
                bootstrap_timeout: config.bootstrap_timeout,
                order: Vec::new(),
                senders: HashMap::new(),
                stall_logged_at: None,
                timeout_logged: false,
            },
            outbox,
        ))
    }

    /// The control socket's port.
    pub fn port(&self) -> u16 {
        self.shared.transport.sockets.port()
    }

    /// The transfer engine's session id.
    pub fn session_id(&self) -> String {
        self.shared.transport.session_id()
    }

    fn log_stalls(&mut self, polls: &[i64]) {
        let now = Instant::now();
        if self
            .stall_logged_at
            .is_some_and(|at| now - at < BOOTSTRAP_STALL)
        {
            return;
        }
        let stalled: Vec<String> = self
            .order
            .iter()
            .zip(polls)
            .filter(|(_, &poll)| poll == TransferPoll::Bootstrapping as i64)
            .filter_map(|(rid, _)| {
                let sender = &self.senders[rid];
                let age = now - sender.started;
                (age >= BOOTSTRAP_STALL)
                    .then(|| format!("{rid}(room={}, age={:.0}s)", sender.room, age.as_secs_f64()))
            })
            .take(8)
            .collect();
        if !stalled.is_empty() {
            self.stall_logged_at = Some(now);
            tracing::warn!(?stalled, "requests still bootstrapping");
        }
    }

    fn forget(&mut self, rid: &str) {
        let Some(sender) = self.senders.remove(rid) else {
            return;
        };
        self.order.retain(|tracked| tracked != rid);
        let mut state = self.shared.state();
        state.rooms.remove(sender.room);
        state.destinations.remove(&sender.room);
        state.cached_tokens.remove(&sender.room);
        state.bootstrap_logprobs.remove(&sender.room);
        state.layerwise_rooms.remove(&sender.room);
        state.prefill_metadata.remove(&sender.room);
    }
}

impl Drop for PrefillPeer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl TransferPeer for PrefillPeer {
    fn register(&mut self, rid: &str, bootstrap: &BootstrapInfo) -> Result<(), PdError> {
        let room = bootstrap.room;
        {
            let mut state = self.shared.state();
            state.cached_tokens.remove(&room);
            state.bootstrap_logprobs.remove(&room);
            state.rooms.update(room, TransferPoll::Bootstrapping);
        }
        if self
            .senders
            .insert(
                rid.to_owned(),
                Sender {
                    room,
                    started: Instant::now(),
                    local: TransferPoll::Bootstrapping,
                    concluded: None,
                },
            )
            .is_none()
        {
            self.order.push(rid.to_owned());
        }
        Ok(())
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
                let sender = self.senders.get_mut(rid).expect("tracked");
                let poll = match sender.concluded {
                    Some(poll) => poll,
                    None => match state.rooms.get(sender.room).unwrap_or(TransferPoll::Failed) {
                        status @ (TransferPoll::Success | TransferPoll::Failed) => {
                            sender.concluded = Some(status);
                            status
                        }
                        TransferPoll::Bootstrapping
                            if now - sender.started >= self.bootstrap_timeout =>
                        {
                            let elapsed = (now - sender.started).as_secs_f64();
                            if !self.timeout_logged {
                                self.timeout_logged = true;
                                tracing::warn!(
                                    "Some requests timed out when bootstrapping: the decode \
                                     instance never sent their cache manifest. Raise \
                                     TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT if a longer \
                                     TTFT is acceptable."
                                );
                            }
                            state.rooms.record_failure(
                                sender.room,
                                format!(
                                    "Request {} timed out after {elapsed:.1}s in \
                                     TransferPoll.Bootstrapping",
                                    sender.room
                                ),
                            );
                            if let Some(metrics) = &self.shared.metrics {
                                metrics.record_timeout("bootstrapping");
                            }
                            sender.concluded = Some(TransferPoll::Failed);
                            TransferPoll::Failed
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
        self.log_stalls(&agreed);

        let mut events = Vec::new();
        let mut ended = Vec::new();
        for (rid, &poll) in self.order.iter().zip(&agreed) {
            let sender = self.senders.get_mut(rid).expect("tracked");
            let poll = TransferPoll::from_wire(poll).unwrap_or(TransferPoll::Failed);
            if sender.local == TransferPoll::Bootstrapping && poll == TransferPoll::Bootstrapped {
                sender.local = TransferPoll::Bootstrapped;
                events.push(PdEvent::Bootstrapped(rid.clone()));
            } else if poll == TransferPoll::Failed {
                let reason = self
                    .shared
                    .state()
                    .rooms
                    .failure(sender.room)
                    .map(str::to_owned);
                tracing::warn!(rid, room = sender.room, reason, "KV transfer failed");
                events.push(PdEvent::Failed(rid.clone()));
                ended.push(rid.clone());
            } else if sender.local == TransferPoll::Bootstrapped && poll == TransferPoll::Success {
                sender.local = TransferPoll::Success;
                events.push(PdEvent::Succeeded(rid.clone()));
                ended.push(rid.clone());
            }
        }
        for rid in ended {
            self.forget(&rid);
        }
        Ok(events)
    }

    fn has_in_flight(&self) -> bool {
        !self.order.is_empty()
    }

    fn abort_room(&mut self, rid: &str, room: u64) {
        self.shared.abort_room(
            room,
            &format!("EPD: prefill aborted request {rid} (embedding receive timed out)"),
        );
    }

    fn record_usage(&mut self, rid: &str, cached_tokens: i64, bootstrap_logprob: Option<f64>) {
        let Some(sender) = self.senders.get(rid) else {
            return;
        };
        let mut state = self.shared.state();
        if state.ended(sender.room) {
            return;
        }
        state.cached_tokens.insert(sender.room, cached_tokens);
        if let Some(logprob) = bootstrap_logprob {
            state.bootstrap_logprobs.insert(sender.room, logprob);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_readiness_is_strict_and_survives_the_wrap() {
        assert!(!step_ready(0, 0));
        assert!(!step_ready(STEP_COUNT_MAX - 1, 0));
        assert!(step_ready(1, 0));
        assert!(step_ready(0, STEP_COUNT_MAX - 1));
    }
}
