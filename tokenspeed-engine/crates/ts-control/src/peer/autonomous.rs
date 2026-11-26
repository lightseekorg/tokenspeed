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

//! The single-member peer driver owns protocol progress while the control thread
//! waits for device work. Only completed attempts cross back into request admission.
use super::*;

enum Command {
    Lookup(Vec<PeerLookupView>, i64),
    DevicePlan(Vec<PeerFetchView>, Instant),
    Metrics(
        Option<Arc<ts_metrics::PeerMetrics>>,
        Option<Arc<ts_metrics::PeerMetrics>>,
    ),
    Version(String),
    End(String, String),
}

struct Worker {
    commands: mpsc::Sender<Command>,
    rounds: mpsc::Receiver<Result<PeerRound, PeerError>>,
    thread: thread::JoinHandle<()>,
    pending: usize,
    failed: bool,
    device: bool,
}
impl Worker {
    fn send(&self, command: Command) {
        // agree() reports worker errors and channel disconnection to the control loop.
        if self.commands.send(command).is_ok() {
            self.thread.thread().unpark();
        }
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        // Disconnect before waking an idle worker. Do not join while the
        // caller may own the scheduler mutex that a reservation is acquiring.
        drop(std::mem::replace(&mut self.commands, mpsc::channel().0));
        self.thread.thread().unpark();
    }
}

enum Driver {
    Rounds(Box<Protocol>),
    Background(Worker),
}

/// The protocol owner and its control-loop handoff.
pub struct Peer {
    driver: Driver,
    versioned: Option<super::coordinated::Control>,
    device_pin_gate: Arc<servicer::DevicePinGate>,
    port: Option<u16>,
    fence: PeerFence,
    origin: Instant,
    member: usize,
    metrics: Option<Arc<ts_metrics::PeerMetrics>>,
    next_sample_us: i64,
    previous_totals: [u64; 3],
}

impl Peer {
    /// Start the configured protocol owner.
    pub fn start(
        settings: PeerSettings,
        progress: PeerProgress,
        member: usize,
        replica: Vec<RankGeometry>,
        listener: Option<PeerListener>,
        scheduler: Arc<Mutex<Scheduler>>,
        transfers: PeerTransfers,
    ) -> Result<Self, PeerError> {
        let device = progress == PeerProgress::Device;
        let background = progress != PeerProgress::Coordinated;
        if background && replica.len() != 1 {
            return Err(PeerError::Topology);
        }
        let mut protocol = Protocol::start(
            settings,
            progress,
            member,
            replica,
            listener,
            scheduler.clone(),
            transfers,
        )?;
        let (port, fence, origin) = (protocol.port(), protocol.fence(), protocol.origin);
        let device_pin_gate = protocol
            .endpoint
            .as_ref()
            .map(|e| e.holder.lock().device_pin_gate.clone())
            .unwrap_or_default();
        let driver = if background {
            let wake = protocol.endpoint.as_ref().unwrap().driver.clone();
            let (commands, rx) = mpsc::channel();
            let (tx, rounds) = mpsc::channel();
            let thread = thread::Builder::new()
                .name("ts-peer-host".into())
                .spawn(move || {
                    wake.run(|| {
                        if let Err(error) = drive(&mut protocol, scheduler, rx, &tx) {
                            tracing::error!(%error, "peer worker stopped");
                            let _ = tx.send(Err(error));
                        }
                    })
                })?;
            Driver::Background(Worker {
                commands,
                rounds,
                thread,
                pending: 0,
                failed: false,
                device,
            })
        } else {
            Driver::Rounds(Box::new(protocol))
        };
        Ok(Self {
            versioned: None,
            driver,
            device_pin_gate,
            port,
            fence,
            origin,
            member,
            metrics: None,
            next_sample_us: 0,
            previous_totals: [0; 3],
        })
    }

    pub(crate) fn autonomous(&self) -> bool {
        matches!(&self.driver, Driver::Background(_))
    }

    /// Drain queued holder changes on the endpoint owner (empty on other
    /// members and autonomous peers, which apply exports directly).
    pub fn take_holder_changes(&self) -> Vec<PeerEventView> {
        match &self.driver {
            Driver::Rounds(p) => p.take_holder_changes(),
            Driver::Background(_) => Vec::new(),
        }
    }

    /// Encode holder changes for intra-replica broadcast.
    pub fn encode_changes(changes: &[PeerEventView]) -> Vec<u8> {
        Protocol::encode_changes(changes)
    }

    /// Decode holder changes from intra-replica broadcast.
    pub fn decode_changes(bytes: &[u8]) -> Result<Vec<PeerEventView>, PeerError> {
        Protocol::decode_changes(bytes)
    }

    /// Report to the normal replica collective (empty for the independent owner).
    pub fn report(&mut self) -> Vec<u8> {
        let legacy = match &mut self.driver {
            Driver::Rounds(p) => p.report(),
            Driver::Background(_) => Vec::new(),
        };
        match &mut self.versioned {
            Some(v) => v.report(legacy),
            None => legacy,
        }
    }

    /// Install the independently registered v1 control path.
    pub fn enable_versioned(
        &mut self,
        layout: Arc<super::registered::Replica>,
        name: String,
        incarnation: u64,
        channel: Option<super::VersionedChannel>,
        retirement_bound_ms: u32,
    ) {
        let (control, fence) = super::coordinated::Control::new(
            layout,
            self.member,
            name,
            incarnation,
            channel,
            self.device_pin_gate.clone(),
            retirement_bound_ms,
        );
        if control.reader() {
            self.fence = fence;
        }
        self.versioned = Some(control);
    }

    /// Apply source votes under the same scheduler lock that protects cache writes.
    pub fn agree_with_scheduler(
        &mut self,
        reports: &[Vec<u8>],
        scheduler: &mut Scheduler,
    ) -> Result<PeerRound, PeerError> {
        if self.versioned.is_none() {
            return self.agree(reports);
        }
        let (legacy, versioned) = super::coordinated::Control::unpack(reports)?;
        let mut round = self.agree(&legacy)?;
        let reads = match &self.driver {
            Driver::Rounds(p) => Some(&p.reads),
            Driver::Background(_) => None,
        };
        self.versioned
            .as_mut()
            .unwrap()
            .advance(versioned, scheduler, reads, &mut round)?;
        Ok(round)
    }
    /// Consume results already completed by the protocol owner.
    pub fn agree(&mut self, reports: &[Vec<u8>]) -> Result<PeerRound, PeerError> {
        match &mut self.driver {
            Driver::Rounds(p) => p.agree(reports),
            Driver::Background(w) => {
                if w.failed {
                    return Err(PeerError::WorkerExited);
                }
                let mut out = PeerRound::default();
                loop {
                    let round = match w.rounds.try_recv() {
                        Ok(Ok(round)) => round,
                        Ok(Err(error)) => {
                            w.failed = true;
                            return Err(error);
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            return Err(PeerError::WorkerExited)
                        }
                    };
                    // A positive lookup offers local admission while reservation is pending.
                    // Only a miss/revocation or a retired transfer finishes the attempt.
                    w.pending -= round
                        .events
                        .iter()
                        .filter(|e| {
                            e.kind == PeerEventKind::FetchDone
                                || (e.kind == PeerEventKind::LookupAnswered
                                    && (w.device || e.num_tokens == 0))
                        })
                        .count();
                    if let Some(metrics) = &self.metrics {
                        for at in round.lookup_received {
                            metrics.stage(
                                "lookup_reply_to_scheduler",
                                "ok",
                                at.elapsed().as_secs_f64(),
                            );
                        }
                        let now = self.origin.elapsed().as_micros() as i64;
                        for e in &round.events {
                            if e.kind == PeerEventKind::FetchDone
                                && e.dispatch_us > 0
                                && e.dispatch_us <= e.writable_us
                                && e.writable_us <= e.terminal_us
                                && e.terminal_us <= now
                            {
                                let outcome = if e.num_tokens > 0 { "ok" } else { "fallback" };
                                for (stage, us) in [
                                    ("grant_to_writable", e.writable_us - e.dispatch_us),
                                    ("writable_to_terminal", e.terminal_us - e.writable_us),
                                    ("terminal_to_scheduler", now - e.terminal_us),
                                ] {
                                    metrics.stage(stage, outcome, us as f64 / 1e6);
                                }
                            }
                        }
                    }
                    out.events.extend(round.events);
                    out.decisions.extend(round.decisions);
                    out.landed.extend(round.landed);
                }
                out.now_us = self.origin.elapsed().as_micros() as i64;
                Ok(out)
            }
        }
    }
    /// Hand off outgoing lookup attempts once.
    pub fn dispatch(
        &mut self,
        fetches: Vec<PeerFetchView>,
        lookups: Vec<PeerLookupView>,
        now_us: i64,
    ) {
        if let Some(reader) = self.versioned.as_mut().filter(|v| v.reader()) {
            reader.dispatch(fetches, lookups, now_us);
            return;
        }
        match &mut self.driver {
            Driver::Rounds(p) => p.dispatch(fetches, lookups, now_us),
            Driver::Background(w) => {
                if !fetches.is_empty() {
                    assert!(w.device, "Host reservations do not come from device plans");
                    w.pending += fetches.len();
                    w.send(Command::DevicePlan(fetches, Instant::now()));
                }
                if !lookups.is_empty() {
                    w.pending += lookups.len();
                    w.send(Command::Lookup(lookups, now_us));
                }
            }
        }
    }
    /// The port of this replica member.
    pub fn port(&self) -> Option<u16> {
        self.port
    }
    /// Gate new Device pins immediately, including when the protocol runs in the background.
    pub fn block_device_pins(&self, blocked: bool) {
        self.device_pin_gate.block(blocked);
    }
    /// The data-plane fence endpoint.
    pub fn fence(&self) -> PeerFence {
        self.fence.clone()
    }
    /// Attempts which still own progress or await consumption.
    pub fn num_landings(&self) -> usize {
        self.versioned.as_ref().map_or(0, |v| v.len())
            + match &self.driver {
                Driver::Rounds(p) => p.num_landings(),
                Driver::Background(w) => w.pending,
            }
    }
    /// Update logical and per-rank observations.
    pub fn set_metrics(
        &mut self,
        logical: Option<Arc<ts_metrics::PeerMetrics>>,
        physical: Option<Arc<ts_metrics::PeerMetrics>>,
    ) {
        self.metrics = logical.clone();
        if let Some(v) = &mut self.versioned {
            v.meter = physical.clone();
        }
        match &mut self.driver {
            Driver::Rounds(p) => p.set_metrics(logical, physical),
            Driver::Background(w) => w.send(Command::Metrics(logical, physical)),
        }
    }
    /// End measurements for a finished request.
    pub fn end_lookup(&mut self, rid: &str, outcome: &str) {
        match &mut self.driver {
            Driver::Rounds(p) => p.end_lookup(rid, outcome),
            Driver::Background(w) => w.send(Command::End(rid.into(), outcome.into())),
        }
    }
    /// Switch the namespace after outstanding work drains.
    pub fn set_weight_version(&mut self, version: &str) {
        if let Some(v) = &mut self.versioned {
            v.invalidate();
        }
        match &mut self.driver {
            Driver::Rounds(p) => p.set_weight_version(version),
            Driver::Background(w) => w.send(Command::Version(version.into())),
        }
    }
    /// The native counters survive gaps between scrapes; gauges are sampled once per second.
    pub fn sample_scheduler(&mut self, scheduler: &mut Scheduler, now_us: i64) {
        let Some(metrics) = self.metrics.as_ref().filter(|_| self.member == 0) else {
            return;
        };
        for sample in scheduler.take_peer_pin_samples() {
            metrics.pin(sample.held_us, sample.remaining_us, sample.expired);
        }
        if now_us < self.next_sample_us {
            return;
        }
        self.next_sample_us = now_us.saturating_add(1_000_000);
        let stats = scheduler.peer_stats();
        for (name, n) in [
            ("leases", stats.leases),
            ("pinned_blocks", stats.pinned_blocks),
            ("looking_requests", stats.looking_requests),
            ("landing_requests", stats.landing_requests),
        ] {
            metrics.snapshot(name, n);
        }
        let totals = [
            stats.landed_tokens,
            stats.failed_landings,
            stats.unused_host_landings,
        ];
        for (i, name) in ["landed_tokens", "failed_landings", "unused_host_landings"]
            .iter()
            .enumerate()
        {
            metrics.increment(name, totals[i].saturating_sub(self.previous_totals[i]));
        }
        self.previous_totals = totals;
    }
}

struct Pending {
    lease: ts_scheduler::PeerLeaseView,
    holder: String,
    rid: String,
    deadline: i64,
}
fn fallback(lease: ts_scheduler::PeerLeaseView) -> PeerEventView {
    PeerEventView {
        kind: PeerEventKind::LookupAnswered,
        lease,
        ..empty_event()
    }
}

// Control messages and hard-pin tickets belong to the same protocol worker.
#[derive(Default)]
struct Exports {
    pins: HashMap<ts_scheduler::PeerLeaseView, (ts_scheduler::PeerReleaseHandle, Instant)>,
    finished: HashMap<ts_scheduler::PeerLeaseView, (Instant, bool)>,
    cleanup: VecDeque<(PeerEventView, i64)>,
}
impl Exports {
    fn advance(
        &mut self,
        p: &Protocol,
        scheduler: &Mutex<Scheduler>,
        draining: bool,
    ) -> Result<bool, PeerError> {
        let endpoint = p.endpoint.as_ref().unwrap();
        let now = Instant::now();
        let micros = now.duration_since(p.origin).as_micros() as i64;
        self.finished.retain(|_, (until, _)| now < *until);
        let send = |(to, reply): (String, Reply)| {
            if let Err(error) = endpoint.net.reply(&to, &reply) {
                tracing::warn!(to, %error, "peer reply not sent");
            }
        };
        for _ in 0..32 {
            let Some((arrival, request)) = endpoint.control.lock().pop_front() else {
                break;
            };
            let (lease, demote) = match request {
                Request::Layout { reply_to } => {
                    send((reply_to, endpoint.holder.lock().layout()));
                    continue;
                }
                Request::Reading { lease } => {
                    if let Some((_, deadline)) = self.pins.get_mut(&lease) {
                        *deadline = arrival + p.settings.record_lifetime;
                    }
                    continue;
                }
                Request::Release { lease, demote } => (lease, demote),
                _ => unreachable!("cache-dependent holder request in control queue"),
            };
            let mut holder = endpoint.holder.lock();
            let replay = self.finished.contains_key(&lease);
            let held = if let Some((_, held)) = self.finished.get(&lease) {
                *held
            } else {
                let held = self
                    .pins
                    .remove(&lease)
                    .is_some_and(|(pin, _)| pin.release());
                holder.finish_record(&lease);
                self.finished
                    .insert(lease.clone(), (now + p.settings.record_lifetime, held));
                self.cleanup.push_back((
                    PeerEventView {
                        kind: PeerEventKind::Release,
                        lease: lease.clone(),
                        demote: demote && held,
                        ..empty_event()
                    },
                    micros,
                ));
                held
            };
            let mut answer = holder.answer(&lease, PeerAnswerKind::Release);
            answer.held = held;
            let reply = holder.reply(lease, answer);
            drop(holder);
            if !replay {
                if let Some(m) = &p.metrics {
                    reply.1.observe_holder(m, demote);
                }
            }
            send(reply);
        }

        // Expiration cannot establish that remote DMA has retired. Keep the native
        // strong refs and fail the engine explicitly so its supervisor can replace it.
        // Consume timely renewals/releases already in ingress before declaring a loss.
        let processed_until = endpoint
            .control
            .lock()
            .front()
            .map_or(now, |(at, _)| (*at).min(now));
        if let Some((lease, _)) = self
            .pins
            .iter()
            .find(|(_, (_, until))| *until < processed_until)
        {
            return Err(PeerError::UnreleasedPin(lease.clone()));
        }
        if draining {
            endpoint.requests.lock().clear();
        }

        // Register a waiter for the scheduler's round-boundary handoff on
        // contention. Bound the wait once per pass so RELEASE can progress.
        let mut handoff_wait = Some(Duration::from_micros(100));
        for _ in 0..16 {
            if self.cleanup.is_empty() && (draining || endpoint.requests.lock().is_empty()) {
                break;
            }
            let Some(mut cache) = scheduler
                .try_lock()
                .or_else(|| scheduler.try_lock_for(handoff_wait.take()?))
            else {
                return Ok(true);
            };
            if let Some((event, at)) = self.cleanup.pop_front() {
                cache
                    .apply_peer_export(&event, at)
                    .map_err(|e| PeerError::Worker(e.to_string()))?;
                continue;
            }
            let Some((at, request)) = endpoint.requests.lock().pop_front() else {
                break;
            };
            if now.saturating_duration_since(at) >= p.settings.record_lifetime {
                continue;
            }
            let mut holder = endpoint.holder.lock();
            if let Request::CheckpointPin { stamp, checkpoint } = request {
                let lease = &checkpoint.lease;
                if self.finished.contains_key(lease) || self.pins.contains_key(lease) {
                    continue;
                }
                let mut answer = holder.answer(lease, PeerAnswerKind::Pin);
                if p.progress == PeerProgress::Device
                    && holder.matches_stamp(&stamp)
                    && holder.device_pin_gate.allows(&checkpoint.pages)
                    && cache
                        .apply_peer_export(&checkpoint, micros)
                        .map_err(|e| PeerError::Worker(e.to_string()))?
                {
                    answer.held = true;
                    answer.num_tokens = checkpoint.num_tokens;
                    answer.pages = checkpoint.pages;
                    self.pins.insert(
                        lease.clone(),
                        (
                            cache.peer_release_handle(lease),
                            now + p.settings.record_lifetime,
                        ),
                    );
                }
                let reply = holder.reply(lease.clone(), answer);
                drop(holder);
                drop(cache);
                if let Some(m) = &p.metrics {
                    reply.1.observe_holder(m, false);
                }
                send(reply);
                continue;
            }
            let lease = match &request {
                Request::Pin { lease } | Request::Lookup { lease, .. } => Some(lease),
                _ => None,
            };
            if lease.is_some_and(|lease| self.finished.contains_key(lease)) {
                continue;
            }
            let (to, mut reply) = holder.handle(&*cache, request, now);
            for change in holder.take_changes() {
                let applied = cache
                    .apply_peer_export(&change, micros)
                    .map_err(|e| PeerError::Worker(e.to_string()))?;
                if let Reply::Answer { answer, .. } = &mut reply {
                    match answer.kind {
                        PeerAnswerKind::Lookup if !applied => {
                            answer.num_tokens = 0;
                            answer.pages.clear();
                        }
                        PeerAnswerKind::Pin => {
                            answer.held &= applied;
                            if answer.held {
                                self.pins.entry(answer.lease.clone()).or_insert_with(|| {
                                    (
                                        cache.peer_release_handle(&answer.lease),
                                        now + p.settings.record_lifetime,
                                    )
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
            drop(holder);
            drop(cache);
            if let Some(m) = &p.metrics {
                reply.observe_holder(m, false);
            }
            send((to, reply));
        }
        Ok(!self.cleanup.is_empty()
            || !endpoint.requests.lock().is_empty()
            || !endpoint.control.lock().is_empty())
    }
}

fn drive(
    p: &mut Protocol,
    scheduler: Arc<Mutex<Scheduler>>,
    commands: mpsc::Receiver<Command>,
    results: &mpsc::Sender<Result<PeerRound, PeerError>>,
) -> Result<(), PeerError> {
    let mut pending = HashMap::new();
    let mut exports = Exports::default();
    let mut reservations = VecDeque::new();
    let mut device_plans = VecDeque::new();
    let mut stopping = false;
    loop {
        loop {
            match commands.try_recv() {
                Ok(Command::Lookup(lookups, scheduler_tick)) => {
                    for mut l in lookups {
                        l.holder = normalize_peer_holder(&l.holder).to_owned();
                        if l.holder == p.settings.name {
                            p.self_lookup_answers.push(fallback(l.lease));
                            continue;
                        }
                        let budget = l.deadline_us.saturating_sub(scheduler_tick).max(0);
                        let sent = p.origin.elapsed().as_micros() as i64;
                        l.deadline_us = sent.saturating_add(budget);
                        pending.insert(
                            l.lease.id,
                            Pending {
                                lease: l.lease.clone(),
                                holder: l.holder.clone(),
                                rid: l.request_id.clone(),
                                deadline: l.deadline_us,
                            },
                        );
                        p.dispatch(vec![], vec![l], sent);
                    }
                }
                Ok(Command::DevicePlan(fetches, dispatch)) => {
                    device_plans.push_back((fetches, dispatch))
                }
                Ok(Command::Metrics(logical, physical)) => p.set_metrics(logical, physical),
                Ok(Command::Version(version)) => p.set_weight_version(&version),
                Ok(Command::End(rid, outcome)) => p.end_lookup(&rid, &outcome),
                Err(mpsc::TryRecvError::Disconnected) => {
                    stopping = true;
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
            }
        }
        if stopping {
            pending.clear();
            reservations.clear();
            device_plans.clear();
            let now = p.origin.elapsed().as_micros() as i64;
            let mut releases = Vec::new();
            for (rid, landing) in &mut p.landings {
                if !p.reading.contains(rid) {
                    if let Some(action) = landing.cancel_before_read(now) {
                        releases.push((rid.clone(), action));
                    }
                }
            }
            p.send(releases, now);
        }
        // Fence notifications can arrive before the plan command. Consume one
        // per plan, in FIFO order. Local device waits have no protocol deadline.
        while !device_plans.is_empty() {
            let Ok(fence) = p.fenced_plans.try_recv() else {
                break;
            };
            let (fetches, dispatch) = device_plans.pop_front().unwrap();
            // The data plane can signal before the driver receives the plan.
            // The peer can start only after both sides of the handoff are ready.
            let times = (p.progress == PeerProgress::Device).then(|| {
                (
                    dispatch.duration_since(p.origin).as_micros() as i64,
                    fence.max(dispatch).duration_since(p.origin).as_micros() as i64,
                )
            });
            p.start_landings(fetches, p.origin.elapsed().as_micros() as i64, false, times);
        }
        let holder_work = exports.advance(p, &scheduler, stopping)?;
        // Take the clock and queue under the same lock used to stamp arrivals.
        let (now, inbox) = {
            let mut queue = p.endpoint.as_ref().unwrap().replies.lock();
            (
                p.origin.elapsed().as_micros() as i64,
                std::mem::take(&mut *queue),
            )
        };
        let mut out = PeerRound::default();
        let mut report = Report {
            now_us: now,
            outcomes: std::mem::take(&mut p.unplaced),
            ..Report::default()
        };
        report.outcomes.extend(p.reads.poll());
        for (at, reply) in inbox {
            if let Reply::Answer { stamp, answer } = &reply {
                if answer.kind == PeerAnswerKind::Lookup {
                    let Some(attempt) = pending.get(&answer.lease.id) else {
                        continue;
                    };
                    if attempt.lease != answer.lease
                        || normalize_peer_holder(&stamp.name) != attempt.holder
                    {
                        continue;
                    }
                    let attempt = pending.remove(&answer.lease.id).unwrap();
                    if at.duration_since(p.origin).as_micros() as i64 >= attempt.deadline {
                        out.events.push(fallback(attempt.lease));
                        continue;
                    }
                    let mut events = Vec::new();
                    p.take_reply(&reply, now, &mut events, &mut Vec::new());
                    let hit = events.pop().expect("lookup answer");
                    if hit.num_tokens > 0 {
                        if p.metrics.is_some() {
                            out.lookup_received.push(at);
                        }
                        if p.progress == PeerProgress::Host {
                            reservations.push_back((attempt, hit.num_tokens, at));
                        }
                        out.events.push(hit);
                    } else {
                        out.events.push(fallback(attempt.lease));
                    }
                    continue;
                }
            }
            let arrival = at.duration_since(p.origin).as_micros() as i64;
            let mut actions = Vec::new();
            // A timely reply stays valid even when a different event is slow.
            // A late PIN still needs an explicit release at the holder.
            if let Reply::Answer { answer, .. } = &reply {
                if let Some((rid, landing)) = p
                    .landings
                    .iter_mut()
                    .find(|(_, l)| *l.lease() == answer.lease)
                {
                    if landing.unanswered_pin(arrival) {
                        actions.push((rid.clone(), Action::SendRelease { demote: false }));
                    }
                    landing.tick(arrival);
                }
            }
            p.take_reply(&reply, arrival, &mut out.events, &mut actions);
            if let Reply::Layout { stamp, .. } = &reply {
                let holder_name = normalize_peer_holder(&stamp.name);
                if p.layouts.contains_key(holder_name) {
                    for (rid, landing) in &mut p.landings {
                        if landing.fetch().holder == holder_name {
                            landing.tick(arrival);
                            if let Some(action) = landing.layout_known(arrival) {
                                actions.push((rid.clone(), action));
                            }
                        }
                    }
                }
            }
            p.send(actions, arrival);
        }
        // Retry each queued lease at most once per turn. The request's grant
        // dies on local admission/abort; a stale holder lease must still pass PIN.
        // Give each queue one bounded handoff so a busy holder queue cannot
        // consume the reader's only opportunity to register a scheduler waiter.
        let mut handoff_wait = Some(Duration::from_micros(100));
        for _ in 0..reservations.len().min(16) {
            let Some(mut cache) = scheduler
                .try_lock()
                .or_else(|| scheduler.try_lock_for(handoff_wait.take()?))
            else {
                break;
            };
            let (attempt, tokens, at) = reservations.pop_front().unwrap();
            let mut retry = false;
            let fetches = cache
                .reserve_peer_host(&attempt.lease, tokens, &mut retry)
                .map_err(|e| PeerError::Worker(e.to_string()))?;
            drop(cache);
            if retry {
                reservations.push_back((attempt, tokens, at));
            } else if fetches.is_empty() {
                out.events.push(fallback(attempt.lease));
            } else {
                if let Some(metrics) = &p.metrics {
                    metrics.stage("admission", "ok", at.elapsed().as_secs_f64());
                }
                p.decisions.push((attempt.rid, "landing"));
                p.start_landings(fetches, p.origin.elapsed().as_micros() as i64, false, None);
            }
        }
        pending.retain(|_, l| {
            if now < l.deadline {
                true
            } else {
                out.events.push(fallback(l.lease.clone()));
                false
            }
        });
        let lifetime_us = p.settings.record_lifetime.as_micros().min(i64::MAX as u128) as i64;
        for (rid, landing) in &mut p.landings {
            if p.reading.contains(rid) && landing.renew_read(now, lifetime_us) {
                let holder = &landing.fetch().holder;
                let message = Request::Reading {
                    lease: landing.lease().clone(),
                };
                if let Err(error) = p.endpoint.as_ref().unwrap().net.request(holder, &message) {
                    tracing::warn!(holder, %error, "could not renew active peer read");
                }
            }
        }
        let round = p.advance_reports(std::slice::from_ref(&report))?;
        out.events.extend(round.events);
        out.decisions.extend(round.decisions);
        out.landed.extend(round.landed);
        if !(out.events.is_empty() && out.decisions.is_empty() && out.landed.is_empty()) {
            let _ = results.send(Ok(out));
        }
        // The final RELEASE can finish the last landing during this pass.
        // Exit before parking: a disconnected controller will never wake us.
        if stopping
            && p.landings.is_empty()
            && exports.pins.is_empty()
            && exports.cleanup.is_empty()
        {
            return Ok(());
        }
        // Socket replies and new attempts wake us immediately. The read pool
        // returns through a channel; poll only while work or a timeout exists.
        if !holder_work
            && device_plans.is_empty()
            && reservations.is_empty()
            && pending.is_empty()
            && p.landings.is_empty()
        {
            if let Some(deadline) = exports.pins.values().map(|(_, deadline)| *deadline).min() {
                thread::park_timeout(deadline.saturating_duration_since(Instant::now()));
            } else {
                thread::park();
            }
        } else {
            thread::park_timeout(Duration::from_millis(1));
        }
    }
}
