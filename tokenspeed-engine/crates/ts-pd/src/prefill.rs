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
//! requests, records usage and polls; the forward thread hands over each
//! plan's remote decode and arms layerwise chunks ([`PrefillOutbox`]).
//!
//! Every route, manifest and copy is planned here, over the rank's cache
//! transfer contract ([`crate::layout`]).

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

use crate::{
    batch::{BatchRows, Staged},
    control::{BootstrapInfo, PdError, PdEvent, TransferPeer},
    http,
    layout::{
        contract::Contract,
        copies::{Source, Transfer},
        manifest::{build_manifest, build_selection, validate_manifest, Manifest, Selection},
        registration::{
            plan_registration, PlannedRegistration, Preallocation, PrefillLayout, RegistrationInfo,
        },
    },
    status::{RoomTable, TransferPoll},
    wire::{self, StatusMessage},
    Copy, PeerConfig, StartError, Transport,
};

/// A request still bootstrapping this long after registration is reported.
const BOOTSTRAP_STALL: Duration = Duration::from_secs(10);

/// How often an idle control-socket thread checks for shutdown.
const SOCKET_POLL: Duration = Duration::from_millis(100);

/// A Decode rank's planned registration with this rank.
#[derive(Debug, Clone, PartialEq)]
struct Registration {
    /// Where the Decode rank's control socket listens.
    endpoint: String,
    /// Its port.
    port: u16,
    /// Its transfer session.
    session_id: String,
    /// The Decode node's attention TP size.
    decode_tp_size: i64,
    /// The Decode rank's attention TP rank.
    decode_tp_rank: i64,
    /// Its arena base.
    arena_base: u64,
    /// Its contract, with its landing addresses.
    layout: Arc<Contract>,
    /// The route this rank planned to it.
    planned: Arc<PlannedRegistration>,
}

impl Registration {
    fn is_dummy(&self) -> bool {
        self.planned.is_dummy
    }
}

/// Layerwise streaming: each prefill chunk's KV goes out an interval of
/// producer steps at a time, as the forward writes it.
#[derive(Debug, Clone)]
pub struct Layerwise {
    /// The pinned host word the device updates with the last producer step
    /// whose KV landed (`StepCounter`).
    pub step_address: usize,
    /// The fields each producer step makes ready, in step order
    /// (`CacheProducerSchedule`).
    pub fields_by_step: Vec<Vec<String>>,
    /// Producer steps per transfer interval.
    pub interval: usize,
}

impl Layerwise {
    /// Each transfer interval's fields.
    fn interval_fields(&self) -> Vec<Option<BTreeSet<String>>> {
        let interval = self.interval.max(1);
        self.fields_by_step
            .chunks(interval)
            .map(|steps| Some(steps.iter().flatten().cloned().collect()))
            .collect()
    }
}

/// A Prefill rank's settings.
#[derive(Debug, Clone)]
pub struct PrefillConfig {
    /// What both roles share.
    pub peer: PeerConfig,
    /// What this rank plans its routes and copies with.
    pub layout: PrefillLayout,
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
    /// Layerwise streaming, when on.
    pub layerwise: Option<Layerwise>,
}

/// One destination of a room.
#[derive(Debug, Clone)]
struct Destination {
    session_id: String,
    /// Its manifest of the request's blocks; `None` from a rank this one
    /// only joins the status agreement with.
    manifest: Option<Arc<Manifest>>,
}

impl Destination {
    fn is_dummy(&self) -> bool {
        self.manifest.is_none()
    }
}

/// Work for a transfer worker.
enum Chunk {
    /// A room's whole KV is written: send it.
    Full {
        room: u64,
        /// This rank's manifest of the request's blocks; `None` when every
        /// destination is a dummy.
        source: Option<Arc<Manifest>>,
        bootstrap_token: i64,
        spec_candidate_ids: Option<Vec<i32>>,
    },
    /// Layerwise: one prefill chunk's KV, sent an interval at a time as the
    /// forward writes it; the last chunk then reports the room done, once
    /// its bootstrap payload arrived.
    Layerwise {
        room: u64,
        /// The blocks the chunk makes ready.
        selection: Arc<Selection>,
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
    /// The rooms of the requests the scheduling loop tracks: a sleep may
    /// not release the cache while one may still send.
    tracked: HashSet<u64>,
    /// The tracked requests' rooms, by request.
    rooms_by_rid: HashMap<String, u64>,
    /// Rooms no request here has claimed, since when: a Decode rank
    /// pre-allocated for one before (or without) its request registering
    /// here, or this rank failed one first (an EPD embedding timeout). One
    /// still unclaimed after the bootstrap timeout is dropped.
    unclaimed: HashMap<u64, Instant>,
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

    /// `room` gained state: start its unclaimed clock unless a request here
    /// tracks it.
    fn touch(&mut self, room: u64) {
        if !self.tracked.contains(&room) {
            self.unclaimed.entry(room).or_insert_with(Instant::now);
        }
    }

    /// Forget the rooms unclaimed for `ttl` or longer. Returns them, and the
    /// failure notices their pre-allocating Decode ranks are owed: a room
    /// this rank failed already told them.
    #[allow(clippy::type_complexity)]
    fn take_unclaimed(&mut self, ttl: Duration) -> (Vec<u64>, Vec<((String, u16), u64)>) {
        let mut expired: Vec<u64> = self
            .unclaimed
            .iter()
            .filter(|(_, since)| since.elapsed() >= ttl)
            .map(|(&room, _)| room)
            .collect();
        expired.sort_unstable();
        let mut notices = Vec::new();
        for &room in &expired {
            if self.rooms.get(room) != Some(TransferPoll::Failed) {
                let destinations = self.destinations.get(&room).map_or(&[][..], Vec::as_slice);
                notices.extend(
                    destinations
                        .iter()
                        .filter(|destination| !destination.is_dummy())
                        .filter_map(|destination| self.route_of(&destination.session_id))
                        .map(|route| (route, room)),
                );
            }
            self.clear(room);
        }
        (expired, notices)
    }

    /// Forget everything kept for `room`.
    fn clear(&mut self, room: u64) {
        self.rooms.remove(room);
        self.tracked.remove(&room);
        self.unclaimed.remove(&room);
        self.destinations.remove(&room);
        self.cached_tokens.remove(&room);
        self.bootstrap_logprobs.remove(&room);
        self.layerwise_rooms.remove(&room);
        self.prefill_metadata.remove(&room);
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
            .map(|r| r.planned.expected_decode_ranks.clone())
            .collect();
        if sizes.len() != 1 || expected.len() != 1 {
            return Err("Cache-transfer room has inconsistent TP metadata".to_owned());
        }
        let expected: BTreeSet<i64> = expected.into_iter().next().expect("one set");
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
    layout: PrefillLayout,
    state: Mutex<State>,
    status_rank: i64,
    failed_session_ttl: Duration,
    metrics: Option<Arc<ts_metrics::KvTransferMetrics>>,
    layerwise: Option<Layerwise>,
    /// Layerwise: each transfer interval's fields.
    interval_fields: Vec<Option<BTreeSet<String>>>,
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
            state.touch(room);
            let destinations = state.destinations.remove(&room).unwrap_or_default();
            destinations
                .iter()
                .filter(|destination| !destination.is_dummy())
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
    }

    fn handle(&self, frames: &[Bytes]) {
        if wire::is_registration(frames) {
            self.handle_registration(frames);
        } else {
            self.handle_preallocation(frames);
        }
    }

    /// Plan the route a Decode rank's registration asks of this rank.
    fn plan_registration(&self, frames: &[Bytes]) -> Result<Registration, String> {
        let info = RegistrationInfo::from_frames(frames).map_err(|error| error.0)?;
        let planned = plan_registration(&self.layout, &info).map_err(|error| error.0)?;
        Ok(Registration {
            endpoint: info.endpoint,
            port: info.port,
            session_id: info.session_id,
            decode_tp_size: info.tp_size,
            decode_tp_rank: info.tp_rank,
            arena_base: info.arena_base,
            layout: Arc::new(info.layout),
            planned: Arc::new(planned),
        })
    }

    fn handle_registration(&self, frames: &[Bytes]) {
        let registered = self.plan_registration(frames).and_then(|registration| {
            let mut state = self.state();
            let session_id = registration.session_id.clone();
            if state
                .registrations
                .get(&session_id)
                .is_some_and(|previous| *previous != registration)
            {
                return Err("conflicting registration for an active CachePD session".to_owned());
            }
            state.rejected.remove(&session_id);
            state.failed_sessions.remove(&session_id);
            state.registrations.insert(session_id, registration);
            Ok(())
        });
        match registered {
            Ok(()) => {}
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
        let mut state = self.state();
        state.registrations.remove(session_id);
        state
            .rejected
            .insert(session_id.to_owned(), (endpoint, port));
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
        let preallocation = Preallocation::from_frames(frames).map_err(|error| error.0)?;
        if preallocation.manifest.is_none() != registration.is_dummy() {
            return Err("CachePD request kind disagrees with its registration".to_owned());
        }
        if let Some(manifest) = &preallocation.manifest {
            validate_manifest(manifest, &registration.layout, "destination")
                .map_err(|error| error.0)?;
        }
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
            manifest: preallocation.manifest.map(Arc::new),
        });
        let expected = registration.planned.expected_decode_ranks.len();
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
        state.touch(room);
        if complete {
            state.rooms.update(room, TransferPoll::Bootstrapped);
        }
        Ok(())
    }

    /// Drop the rooms no request here claimed within `ttl`, telling each
    /// Decode rank that pre-allocated for one that it failed.
    fn drop_unclaimed(&self, ttl: Duration) {
        let (expired, notices) = self.state().take_unclaimed(ttl);
        if !expired.is_empty() {
            tracing::warn!(
                rooms = ?expired,
                ?ttl,
                "dropping rooms no request on this rank claimed"
            );
        }
        for (route, room) in notices {
            self.notify_failed(route, room);
        }
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
                source,
                bootstrap_token,
                spec_candidate_ids,
            } => self.transfer_full(
                *room,
                source.as_deref(),
                *bootstrap_token,
                spec_candidate_ids,
            ),
            Chunk::Layerwise {
                room,
                selection,
                target_steps,
                is_last,
            } => self.transfer_layerwise(*room, selection, target_steps, *is_last),
        }
    }

    /// The copies that move `source` to one Decode rank, per entry of
    /// `steps` (the fields each moves; `None`: every field).
    fn plan_copies(
        &self,
        registration: &Registration,
        source: Source<'_>,
        destination: &Destination,
        steps: &[Option<BTreeSet<String>>],
    ) -> Result<Vec<Vec<Copy>>, String> {
        let manifest = destination
            .manifest
            .as_deref()
            .ok_or("a dummy destination has no manifest to send to")?;
        Transfer {
            layout: &self.layout.local,
            arena_base: self.layout.arena_base,
            destination_base: registration.arena_base,
            destination_tp_rank: registration.decode_tp_rank,
            destination_layout: &registration.layout,
            source,
            destination: manifest,
            fragments: &registration.planned.fragments,
            owner_filters: &registration.planned.owner_filters,
        }
        .copies(steps)
        .map_err(|error| error.0)
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
        selection: &Selection,
        target_steps: &[u64],
        is_last: bool,
    ) -> Result<(), String> {
        let address = self
            .layerwise
            .as_ref()
            .map(|layerwise| layerwise.step_address)
            .ok_or("Cache-transfer layerwise transfer has no producer step counter")?;
        let destinations: Vec<Destination> = self
            .state()
            .destinations
            .get(&room)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|destination| !destination.is_dummy())
            .collect();
        if destinations.is_empty() {
            return Err("CachePD transfer room has no destinations".to_owned());
        }
        let registrations = self.live_registrations(&destinations)?;
        let sessions: Vec<String> = destinations.iter().map(|d| d.session_id.clone()).collect();
        // Every peer's copies of every interval, planned once, before any DMA.
        let plans = registrations
            .iter()
            .zip(&destinations)
            .map(|(registration, destination)| {
                self.plan_copies(
                    registration,
                    Source::Selection(selection),
                    destination,
                    &self.interval_fields,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (interval, &target) in target_steps.iter().enumerate() {
            self.wait_for_step(address, target, room, &sessions)?;
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
        source: Option<&Manifest>,
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
        if destinations.iter().any(Destination::is_dummy) {
            // This rank only joins the status agreement for the room.
            let mut state = self.state();
            if state.rooms.get(room).is_some() {
                state.rooms.update(room, TransferPoll::Success);
                state.destinations.remove(&room);
            }
            return Ok(());
        }
        let registrations = self.live_registrations(&destinations)?;
        let source = source.ok_or("Cache-transfer source manifest is unavailable")?;
        let mut prepared = Vec::with_capacity(destinations.len());
        for (destination, registration) in destinations.iter().zip(registrations) {
            let copies = self
                .plan_copies(
                    &registration,
                    Source::Manifest(source),
                    destination,
                    &[None],
                )?
                .pop()
                .unwrap_or_default();
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

/// Where the forward thread hands a Prefill rank's peer its work; never
/// waits for a transfer.
///
/// A plan's remote decode is staged when the plan goes to the data plane
/// ([`PrefillOutbox::stage_remote_decode`]) and handed over by the data plane
/// once the drafter's pending round finished
/// ([`PrefillOutbox::hand_over_remote_decode`]): the KV it sends includes
/// what that round writes.
#[derive(Clone)]
pub struct PrefillOutbox {
    shared: Arc<Shared>,
    queues: Arc<Vec<mpsc::Sender<Chunk>>>,
    staged: Staged,
    /// Layerwise: the producer steps armed so far, modulo `STEP_COUNT_MAX`
    /// (the host half of `StepCounter`).
    armed_steps: Arc<Mutex<u64>>,
}

impl PrefillOutbox {
    /// A sleep releases the cache: deregister it from the transfer engine,
    /// refused while a tracked request's KV may still go out. A room no
    /// request here claimed sends nothing, so it does not hold the release.
    pub fn release_registration(&self) -> Result<(), String> {
        let mut sending: Vec<u64> = self.shared.state().tracked.iter().copied().collect();
        if !sending.is_empty() {
            sending.sort_unstable();
            return Err(format!(
                "cannot release the cache while KV sends are in flight: rooms {sending:?}"
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

    /// Whether prefill chunks stream their KV layerwise.
    pub fn streams_layerwise(&self) -> bool {
        self.shared.layerwise.is_some()
    }

    /// Stage a plan's remote decode: the prompts whose KV goes out now.
    pub fn stage_remote_decode(&self, rows: BatchRows) -> Result<(), String> {
        self.staged.stage(rows)
    }

    /// Whether a staged remote decode still waits to be handed over.
    pub fn remote_decode_pending(&self) -> bool {
        self.staged.is_pending()
    }

    /// The staged remote decode's device prerequisites are met: send each
    /// prompt's KV to its destinations (`HostPrefillPeer.execute`). A room
    /// that streams layerwise already sends; it only gets its bootstrap
    /// payload.
    pub fn hand_over_remote_decode(&self) -> Result<(), String> {
        let rows = self.staged.take()?;
        for (index, rid) in rows.request_ids.iter().enumerate() {
            let Some(room) = self.shared.state().rooms_by_rid.get(rid).copied() else {
                continue;
            };
            let token = *rows
                .decode_input_ids
                .get(index)
                .ok_or("a remote decode row has no first token")?;
            if token < 0 {
                return Err("Cache-transfer bootstrap token is unavailable".to_owned());
            }
            let spec_candidate_ids = rows
                .spec_candidate_ids
                .get(index)
                .filter(|ids| !ids.is_empty())
                .cloned();
            let source = {
                let state = self.shared.state();
                let destinations = state.destinations.get(&room).map_or(&[][..], Vec::as_slice);
                if destinations.is_empty() {
                    drop(state);
                    self.send_unplanned(rid, room)?;
                    continue;
                }
                let windows: Vec<&Manifest> = destinations
                    .iter()
                    .filter_map(|destination| destination.manifest.as_deref())
                    .collect();
                match windows.first() {
                    Some(window) if !state.layerwise_rooms.contains(&room) => {
                        same_prompt_window(&windows)?;
                        Some(Arc::new(
                            build_manifest(
                                &rows.tables,
                                &self.shared.layout.local,
                                index,
                                window.prefix_len,
                                window.prompt_len,
                                window.state_prefix_cached,
                            )
                            .map_err(|error| error.0)?,
                        ))
                    }
                    _ => None,
                }
            };
            self.send(rid, room, i64::from(token), spec_candidate_ids, source);
        }
        Ok(())
    }

    /// Layerwise streaming: arm one chunk per extend row of the forward the
    /// plan launches whose request has a destination, ahead of the forward
    /// (`HostPrefillPeer.prepare_prefill`). Every row is planned before any
    /// is armed, and the forward's producer steps are counted once.
    pub fn arm_layerwise(&self, forward: &BatchRows) -> Result<(), String> {
        let Some(layerwise) = &self.shared.layerwise else {
            return Ok(());
        };
        if forward.num_extends == 0 {
            return Ok(());
        }
        let chunks = layerwise_chunks(&self.shared.state(), &self.shared.layout.local, forward)?;
        // The attention backends record one producer range per forward, even
        // when its chunk completes no block.
        let step_count = layerwise.fields_by_step.len() as u64;
        let begin = {
            let mut armed = self.armed_steps.lock().expect("the step lock is poisoned");
            let begin = *armed;
            *armed = (begin + step_count) % STEP_COUNT_MAX;
            begin
        };
        let targets = interval_targets(begin, step_count, layerwise.interval.max(1) as u64);
        for (rid, room, is_last, selection) in chunks {
            self.send_layerwise(&rid, room, Arc::new(selection), targets.clone(), is_last);
        }
        Ok(())
    }

    /// A request's KV is written: send it to its destinations
    /// (`add_transfer_request`), or, for a room streaming layerwise, give its
    /// final chunk the bootstrap payload.
    fn send(
        &self,
        rid: &str,
        room: u64,
        bootstrap_token: i64,
        spec_candidate_ids: Option<Vec<i32>>,
        source: Option<Arc<Manifest>>,
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
                source,
                bootstrap_token,
                spec_candidate_ids,
            },
        );
    }

    /// Layerwise streaming: queue one armed chunk of `room`'s request; its
    /// intervals are complete at `target_steps`.
    fn send_layerwise(
        &self,
        rid: &str,
        room: u64,
        selection: Arc<Selection>,
        target_steps: Vec<u64>,
        is_last: bool,
    ) {
        {
            let mut state = self.shared.state();
            if state.ended(room) {
                return;
            }
            state.layerwise_rooms.insert(room);
        }
        self.enqueue(
            rid,
            Chunk::Layerwise {
                room,
                selection,
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
    fn send_unplanned(&self, rid: &str, room: u64) -> Result<(), String> {
        if self.shared.state().ended(room) {
            return Ok(());
        }
        Err(format!(
            "Cache-transfer destination metadata is unavailable for {rid} (room {room})"
        ))
    }
}

/// One forward's layerwise chunks: for each extend row whose request has a
/// destination, the blocks its chunk makes ready and whether it ends the
/// prompt. A room's first chunk also carries the blocks this rank found in
/// its cache that the Decode rank still needs; later ones start at their
/// own boundary, so nothing goes twice. A chunk that completes no block is
/// skipped, but for the prompt's last.
fn layerwise_chunks(
    state: &State,
    layout: &Contract,
    forward: &BatchRows,
) -> Result<Vec<(String, u64, bool, Selection)>, String> {
    let mut chunks = Vec::new();
    for index in 0..forward.num_extends {
        let rid = &forward.request_ids[index];
        let Some(&room) = state.rooms_by_rid.get(rid) else {
            continue;
        };
        let windows: Vec<&Manifest> = state
            .destinations
            .get(&room)
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .filter_map(|destination| destination.manifest.as_deref())
            .collect();
        // A dummy-only room completes after the forward like any other.
        let Some(window) = windows.first() else {
            continue;
        };
        same_prompt_window(&windows)?;
        let chunk_begin = i64::from(forward.extend_prefix_lens[index]);
        let chunk_end = chunk_begin + i64::from(forward.input_lengths[index]);
        if chunk_end > window.prompt_len {
            return Err(
                "Cache-transfer Prefill chunk extends past Decode's prompt manifest".to_owned(),
            );
        }
        let is_last = chunk_end == window.prompt_len;
        let mut selection_start = chunk_begin;
        if !state.layerwise_rooms.contains(&room) {
            selection_start = selection_start.min(window.prefix_len);
        }
        let selection = build_selection(
            &forward.tables,
            layout,
            index,
            window.prefix_len,
            window.prompt_len,
            selection_start,
            chunk_end,
            window.state_prefix_cached,
        )
        .map_err(|error| error.0)?;
        if !is_last
            && selection
                .groups
                .iter()
                .all(|group| group.source_block_ids.is_empty())
        {
            continue;
        }
        chunks.push((rid.clone(), room, is_last, selection));
    }
    Ok(chunks)
}

/// The producer step each transfer interval of a forward is complete at,
/// for a forward of `step_count` steps after the first `begin`: its last.
fn interval_targets(begin: u64, step_count: u64, interval: u64) -> Vec<u64> {
    (0..step_count)
        .step_by(interval as usize)
        .map(|first| (begin + (first + interval).min(step_count) - 1) % STEP_COUNT_MAX)
        .collect()
}

/// Every destination of a room describes the same prompt window.
fn same_prompt_window(windows: &[&Manifest]) -> Result<(), String> {
    let Some(first) = windows.first() else {
        return Ok(());
    };
    if windows.iter().any(|other| {
        other.prefix_len != first.prefix_len
            || other.prompt_len != first.prompt_len
            || other.state_prefix_cached != first.state_prefix_cached
    }) {
        return Err("Cache-transfer destinations disagree on the prompt window".to_owned());
    }
    Ok(())
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
    pub fn start(config: PrefillConfig) -> Result<(Self, PrefillOutbox), StartError> {
        let (incoming, messages) = mpsc::channel();
        let transport = Transport::start(&config.peer, incoming)?;
        let port = transport.sockets.port();
        let interval_fields = config
            .layerwise
            .as_ref()
            .map(Layerwise::interval_fields)
            .unwrap_or_default();
        let shared = Arc::new(Shared {
            transport,
            layout: config.layout,
            state: Mutex::new(State::default()),
            status_rank: config.status_rank,
            failed_session_ttl: config.failed_session_ttl,
            metrics: config.peer.metrics.clone(),
            layerwise: config.layerwise,
            interval_fields,
            metadata_ready: Condvar::new(),
        });
        let stop = Arc::new(AtomicBool::new(false));
        {
            let shared = shared.clone();
            let stop = stop.clone();
            let unclaimed_ttl = config.bootstrap_timeout;
            thread::Builder::new()
                .name("ts-pd-bootstrap".to_owned())
                .spawn(move || {
                    let mut swept = Instant::now();
                    loop {
                        match messages.recv_timeout(SOCKET_POLL) {
                            Ok(frames) => shared.handle(&frames),
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                if stop.load(Ordering::Relaxed) {
                                    return;
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => return,
                        }
                        if swept.elapsed() >= SOCKET_POLL {
                            swept = Instant::now();
                            shared.drop_unclaimed(unclaimed_ttl);
                        }
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
            staged: Staged::default(),
            armed_steps: Arc::new(Mutex::new(0)),
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

    /// EPD, the first stage of a multimodal Prefill node: a receiver of the
    /// encode nodes' embeddings, over this peer's transfer engine.
    pub fn embedding_receiver(
        &self,
        config: crate::epd::receiver::ReceiverConfig,
    ) -> Result<crate::epd::receiver::Receiver, StartError> {
        crate::epd::receiver::Receiver::start(config, self.shared.transport.clone())
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
        state.clear(sender.room);
        state.rooms_by_rid.remove(rid);
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
            // Claimed: its pre-allocations, or its failure, are this
            // request's now.
            state.unclaimed.remove(&room);
            state.tracked.insert(room);
            state.rooms_by_rid.insert(rid.to_owned(), room);
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

    #[test]
    fn unclaimed_rooms_expire_and_their_decode_ranks_are_told() {
        let mut state = State::default();
        let layout =
            Contract::from_wire(crate::layout::contract::tests::wire().as_bytes()).unwrap();
        state.registrations.insert(
            "s0".to_owned(),
            Registration {
                endpoint: "decode".to_owned(),
                port: 7,
                session_id: "s0".to_owned(),
                decode_tp_size: 1,
                decode_tp_rank: 0,
                arena_base: 0x1000,
                layout: Arc::new(layout),
                planned: Arc::new(PlannedRegistration {
                    fragments: Vec::new(),
                    owner_filters: Default::default(),
                    is_dummy: false,
                    expected_decode_ranks: BTreeSet::from([0]),
                }),
            },
        );
        let destination = || {
            vec![Destination {
                session_id: "s0".to_owned(),
                manifest: Some(Arc::new(Manifest {
                    groups: Vec::new(),
                    prefix_len: 0,
                    prompt_len: 1,
                    state_prefix_cached: false,
                })),
            }]
        };
        // Pre-allocated by a Decode rank; its request never registers here.
        state.destinations.insert(1, destination());
        state.touch(1);
        // Failed here before any request claimed it (an EPD timeout): its
        // destinations were told then.
        state.rooms.update(2, TransferPoll::Failed);
        state.touch(2);
        // A request's room never expires.
        state.tracked.insert(3);
        state.destinations.insert(3, destination());
        state.touch(3);

        assert_eq!(
            state.take_unclaimed(Duration::from_secs(60)),
            (Vec::new(), Vec::new())
        );
        let (expired, notices) = state.take_unclaimed(Duration::ZERO);
        assert_eq!(expired, [1, 2]);
        assert_eq!(notices, [(("decode".to_owned(), 7), 1)]);
        assert!(state.unclaimed.is_empty());
        assert!(!state.destinations.contains_key(&1));
        assert_eq!(state.rooms.get(2), None);
        assert!(state.destinations.contains_key(&3));
    }

    /// A history group of 2-token blocks and two state groups.
    fn grouped_layout() -> Contract {
        let group = |id: &str| serde_json::json!({"group_id": id, "cache_blocks_per_lcm_block": 1, "page_count": 32});
        let field = |group: &str, field: &str, offset: i64| {
            serde_json::json!({"group_id": group, "field_id": field, "plane_id": "p0",
                               "shape": [4], "dtype": "bfloat16", "field_offset_bytes": offset,
                               "page_stride_bytes": 32})
        };
        let state = |id: &str| {
            serde_json::json!({"group_id": id, "attn_kind": "state", "replayable": false,
                               "transfer_policy": "latest_snapshot", "shard_count": 1})
        };
        let wire = serde_json::json!({
            "plan": {
                "prefix_granularity": 2,
                "lcm_block_bytes": 32,
                "num_lcm_blocks": 31,
                "groups": [group("history"), group("state-a"), group("state-b")],
                "planes": [{"plane_id": "p0", "bytes_per_lcm_block": 32, "arena_offset_bytes": 0}],
                "fields": [field("history", "layer.0.kv", 0), field("state-a", "layer.2.state", 8),
                           field("state-b", "layer.4.state", 16)],
                "resident_block_bytes": null
            },
            "group_specs": [
                {"group_id": "history", "attn_kind": "full_history", "rows_per_page": 2,
                 "entry_stride_tokens": 1, "replayable": false, "transfer_policy": "full_suffix",
                 "shard_count": 1},
                state("state-a"),
                state("state-b")
            ],
            "transfer_schema": {"fields": []},
            "field_addresses": null
        });
        Contract::from_wire(wire.to_string().as_bytes()).unwrap()
    }

    /// `rid`'s room `room`, pre-allocated by one Decode rank whose manifest
    /// spans `prefix..prompt`.
    fn destined(state: &mut State, rid: &str, room: u64, prefix: i64, prompt: i64) {
        state.rooms_by_rid.insert(rid.to_owned(), room);
        state.destinations.insert(
            room,
            vec![Destination {
                session_id: "decode:7000".to_owned(),
                manifest: Some(Arc::new(Manifest {
                    groups: Vec::new(),
                    prefix_len: prefix,
                    prompt_len: prompt,
                    state_prefix_cached: false,
                })),
            }],
        );
    }

    fn chunk_forward(rows: &[(&str, i32, i32)], tables: &[(&str, Vec<Vec<i64>>)]) -> BatchRows {
        BatchRows {
            request_ids: rows.iter().map(|row| row.0.to_owned()).collect(),
            extend_prefix_lens: rows.iter().map(|row| row.1).collect(),
            input_lengths: rows.iter().map(|row| row.2).collect(),
            num_extends: rows.len(),
            tables: tables
                .iter()
                .map(|(group, rows)| {
                    let columns = rows[0].len();
                    (
                        (*group).to_owned(),
                        crate::layout::manifest::BlockTable::from_rows(rows, columns),
                    )
                })
                .collect(),
            ..BatchRows::default()
        }
    }

    #[test]
    fn a_forward_arms_a_chunk_per_destined_row_only_when_every_row_plans() {
        let layout = grouped_layout();
        let mut state = State::default();
        destined(&mut state, "request-0", 9, 0, 4);
        destined(&mut state, "request-1", 10, 0, 4);
        let mut tables = vec![
            ("history", vec![vec![1, 2], vec![3, 4]]),
            ("state-a", vec![vec![5, 6], vec![7, 8]]),
            ("state-b", vec![vec![9, 10], vec![11, 12]]),
        ];
        let rows = [
            ("request-0", 0, 4),
            ("request-1", 0, 4),
            ("untracked", 0, 4),
        ];
        let chunks = layerwise_chunks(&state, &layout, &chunk_forward(&rows, &tables)).unwrap();
        let armed: Vec<(&str, u64, bool, Vec<i64>)> = chunks
            .iter()
            .map(|(rid, room, is_last, selection)| {
                (
                    rid.as_str(),
                    *room,
                    *is_last,
                    selection.groups[1].source_block_ids.clone(),
                )
            })
            .collect();
        // The whole prompt in one chunk: the state groups' live slots go too.
        assert_eq!(
            armed,
            [
                ("request-0", 9, true, vec![6]),
                ("request-1", 10, true, vec![8])
            ]
        );
        // A later bad row refuses the whole forward.
        tables[0].1[1][0] = 0;
        let error = layerwise_chunks(&state, &layout, &chunk_forward(&rows, &tables)).unwrap_err();
        assert!(error.contains("invalid block ID"), "{error}");
    }

    #[test]
    fn a_rooms_first_chunk_carries_a_deeper_prefill_cache_hit() {
        let layout = grouped_layout();
        let mut state = State::default();
        // Decode holds 2 tokens of the 9-token prompt; this rank's cache 4.
        destined(&mut state, "request-0", 9, 2, 9);
        let tables = [
            ("history", vec![vec![1, 2, 3, 4, 5]]),
            ("state-a", vec![vec![6, 7]]),
            ("state-b", vec![vec![11, 12]]),
        ];
        let history = |state: &State, begin, length| {
            let forward = chunk_forward(&[("request-0", begin, length)], &tables);
            let (_, _, is_last, selection) = layerwise_chunks(state, &layout, &forward)
                .unwrap()
                .pop()
                .unwrap();
            let group = selection.groups[0].clone();
            (is_last, group.source_block_ids, group.destination_positions)
        };
        // The first chunk, [4, 7), also sends the history blocks of [2, 4)
        // Decode lacks.
        assert_eq!(history(&state, 4, 3), (false, vec![2, 3], vec![0, 1]));
        state.layerwise_rooms.insert(9);
        // The last starts at its own boundary.
        assert_eq!(history(&state, 7, 2), (true, vec![4, 5], vec![2, 3]));
    }

    #[test]
    fn each_interval_completes_at_its_last_producer_step() {
        assert_eq!(interval_targets(0, 5, 2), [1, 3, 4]);
        // Every forward counts its whole range: the next starts after it.
        assert_eq!(interval_targets(0, 3, 1), [0, 1, 2]);
        assert_eq!(interval_targets(3, 3, 1), [3, 4, 5]);
        assert_eq!(
            interval_targets(STEP_COUNT_MAX - 1, 2, 1),
            [STEP_COUNT_MAX - 1, 0]
        );
    }
}
