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

//! The single-member Host driver owns protocol progress while the control thread
//! waits for device work. Only completed attempts cross back into request admission.
use super::*;

enum Command {
    Lookup(Vec<PeerLookupView>, i64),
    DevicePlan(Vec<PeerFetchView>, Vec<PeerLookupView>, Instant, i64),
    CancelGrant(String),
    Metrics(
        Option<Arc<ts_metrics::PeerMetrics>>,
        Option<Arc<ts_metrics::PeerMetrics>>,
    ),
    Version(String),
    End(String, String),
}

struct Worker {
    commands: mpsc::Sender<Command>,
    rounds: mpsc::Receiver<PeerRound>,
    thread: thread::JoinHandle<()>,
    pending: usize,
    device: bool,
    grant_first: bool,
}
impl Worker {
    fn send(&self, command: Command) {
        self.commands.send(command).expect("peer worker exited");
        self.thread.thread().unpark();
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
    port: Option<u16>,
    fence: PeerFence,
    origin: Instant,
    member: usize,
    metrics: Option<Arc<ts_metrics::PeerMetrics>>,
    next_sample_us: i64,
    previous_totals: [u64; 2],
}

impl Peer {
    /// Start the configured protocol owner.
    pub fn start(
        settings: PeerSettings,
        member: usize,
        replica: Vec<RankGeometry>,
        listener: Option<PeerListener>,
        scheduler: Arc<Mutex<Scheduler>>,
        reader: Arc<dyn PageReader>,
        pusher: Arc<dyn PagePusher>,
    ) -> Result<Self, PeerError> {
        let device = settings.device_admission.is_some();
        let grant_first = settings.grant_first;
        if let Some(adaptive) = settings.device_admission {
            if settings.autonomous_host
                || replica.len() != 1
                || !(if grant_first {
                    scheduler.lock().enable_grant_first_peer()
                } else {
                    scheduler.lock().enable_async_device_peer(adaptive)
                })
            {
                return Err(PeerError::DeviceTopology);
            }
        }
        let background = settings.autonomous_host || device;
        if settings.autonomous_host
            && (replica.len() != 1 || !scheduler.lock().enable_async_host_peer())
        {
            return Err(PeerError::HostTopology);
        }
        let protocol = Protocol::start(
            settings,
            member,
            replica,
            listener,
            scheduler.clone(),
            reader,
            pusher,
        )?;
        let (port, fence, origin) = (protocol.port(), protocol.fence(), protocol.origin);
        let driver = if background {
            let wake = protocol.endpoint.as_ref().unwrap().driver.clone();
            let (commands, rx) = mpsc::channel();
            let (tx, rounds) = mpsc::channel();
            let thread = thread::Builder::new()
                .name("ts-peer-host".into())
                .spawn(move || drive(protocol, scheduler, rx, tx))?;
            wake.set(thread.thread().clone())
                .expect("one protocol worker");
            Driver::Background(Worker {
                commands,
                rounds,
                thread,
                pending: 0,
                device,
                grant_first,
            })
        } else {
            Driver::Rounds(Box::new(protocol))
        };
        Ok(Self {
            driver,
            port,
            fence,
            origin,
            member,
            metrics: None,
            next_sample_us: 0,
            previous_totals: [0; 2],
        })
    }

    pub(crate) fn autonomous(&self) -> bool {
        matches!(&self.driver, Driver::Background(_))
    }

    /// Report to the normal replica collective (empty for the independent owner).
    pub fn report(&mut self) -> Vec<u8> {
        match &mut self.driver {
            Driver::Rounds(p) => p.report(),
            Driver::Background(_) => Vec::new(),
        }
    }
    /// Consume results already completed by the protocol owner.
    pub fn agree(&mut self, reports: &[Vec<u8>]) -> Result<PeerRound, PeerError> {
        match &mut self.driver {
            Driver::Rounds(p) => p.agree(reports),
            Driver::Background(w) => {
                let mut out = PeerRound::default();
                for round in w.rounds.try_iter() {
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
        match &mut self.driver {
            Driver::Rounds(p) => p.dispatch(fetches, lookups, now_us),
            Driver::Background(w) => {
                if w.grant_first {
                    assert_eq!(
                        fetches.len(),
                        lookups.len(),
                        "one lookup per granted request"
                    );
                    if !fetches.is_empty() {
                        w.pending += fetches.len();
                        w.send(Command::DevicePlan(
                            fetches,
                            lookups,
                            Instant::now(),
                            now_us,
                        ));
                    }
                    return;
                }
                if !fetches.is_empty() {
                    assert!(w.device, "Host reservations do not come from device plans");
                    w.pending += fetches.len();
                    w.send(Command::DevicePlan(fetches, vec![], Instant::now(), now_us));
                }
                if !lookups.is_empty() {
                    w.pending += lookups.len();
                    w.send(Command::Lookup(lookups, now_us));
                }
            }
        }
    }
    /// Withdraw an unstarted grant. Transfers already using the pages retire normally.
    pub fn cancel_grant(&mut self, rid: &str) {
        if let Driver::Background(w) = &self.driver {
            if w.grant_first {
                w.send(Command::CancelGrant(rid.to_owned()));
            }
        }
    }

    /// The port of this replica member.
    pub fn port(&self) -> Option<u16> {
        self.port
    }
    /// The data-plane fence endpoint.
    pub fn fence(&self) -> PeerFence {
        self.fence.clone()
    }
    /// Attempts which still own progress or await consumption.
    pub fn num_landings(&self) -> usize {
        match &self.driver {
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
        let totals = [stats.landed_tokens, stats.failed_landings];
        for (i, name) in ["landed_tokens", "failed_landings"].iter().enumerate() {
            metrics.increment(name, totals[i].saturating_sub(self.previous_totals[i]));
        }
        self.previous_totals = totals;
    }
}

struct DeviceGrant {
    fetch: PeerFetchView,
    allowed: u32,
    deadline: i64,
    times: (i64, i64),
}

fn unused_grant(fetch: PeerFetchView) -> PeerEventView {
    PeerEventView {
        kind: PeerEventKind::FetchDone,
        request_id: fetch.request_id,
        num_tokens: fetch.begin_tokens,
        ..empty_event()
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
    pins: HashMap<ts_scheduler::PeerLeaseView, ts_scheduler::PeerReleaseHandle>,
    finished: HashMap<ts_scheduler::PeerLeaseView, (Instant, bool)>,
    cleanup: VecDeque<(PeerEventView, i64)>,
}
impl Exports {
    fn advance(&mut self, p: &Protocol, scheduler: &Mutex<Scheduler>) -> bool {
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
            let Some(request) = endpoint.control.lock().pop_front() else {
                break;
            };
            let (lease, demote) = match request {
                Request::Layout { reply_to } => {
                    send((reply_to, endpoint.holder.lock().layout()));
                    continue;
                }
                Request::Release { lease, demote } => (lease, demote),
                _ => unreachable!("cache-dependent holder request in control queue"),
            };
            let mut holder = endpoint.holder.lock();
            let held = if let Some((_, held)) = self.finished.get(&lease) {
                *held
            } else {
                let held = self.pins.remove(&lease).is_some_and(|pin| pin.release());
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
            send(holder.reply(lease, answer));
        }

        for _ in 0..16 {
            if self.cleanup.is_empty() && endpoint.requests.lock().is_empty() {
                break;
            }
            let Some(mut cache) = scheduler.try_lock() else {
                return true;
            };
            if let Some((event, at)) = self.cleanup.pop_front() {
                cache
                    .apply_peer_export(&event, at)
                    .expect("retire holder export");
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
                if p.settings.checkpoint_pin
                    && holder.matches_stamp(&stamp)
                    && cache
                        .apply_peer_export(&checkpoint, micros)
                        .expect("validate and pin checkpoint")
                {
                    answer.held = true;
                    answer.num_tokens = checkpoint.num_tokens;
                    answer.pages = checkpoint.pages;
                    self.pins
                        .insert(lease.clone(), cache.peer_release_handle(lease));
                }
                let reply = holder.reply(lease.clone(), answer);
                drop(holder);
                drop(cache);
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
                    .expect("valid holder export");
                if let Reply::Answer { answer, .. } = &mut reply {
                    match answer.kind {
                        PeerAnswerKind::Lookup if !applied => {
                            answer.num_tokens = 0;
                            answer.pages.clear();
                        }
                        PeerAnswerKind::Pin => {
                            answer.held &= applied;
                            if answer.held {
                                self.pins
                                    .entry(answer.lease.clone())
                                    .or_insert_with(|| cache.peer_release_handle(&answer.lease));
                            }
                        }
                        _ => {}
                    }
                }
            }
            drop(holder);
            drop(cache);
            send((to, reply));
        }
        !self.cleanup.is_empty()
            || !endpoint.requests.lock().is_empty()
            || !endpoint.control.lock().is_empty()
    }
}

fn drive(
    mut p: Protocol,
    scheduler: Arc<Mutex<Scheduler>>,
    commands: mpsc::Receiver<Command>,
    results: mpsc::Sender<PeerRound>,
) {
    let mut pending = HashMap::new();
    let mut exports = Exports::default();
    let mut reservations = VecDeque::new();
    let mut device_plans: VecDeque<(Vec<PeerFetchView>, Vec<PeerLookupView>, Instant, i64)> =
        VecDeque::new();
    let mut grants: HashMap<u64, DeviceGrant> = HashMap::new();
    let mut cancelled = HashSet::new();
    let mut completions = Vec::new();
    let mut stopping = false;
    loop {
        loop {
            match commands.try_recv() {
                Ok(Command::Lookup(lookups, scheduler_tick)) => {
                    for mut l in lookups {
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
                Ok(Command::DevicePlan(fetches, lookups, dispatch, tick)) => {
                    device_plans.push_back((fetches, lookups, dispatch, tick))
                }
                Ok(Command::CancelGrant(rid)) => {
                    if device_plans
                        .iter()
                        .any(|(fs, _, _, _)| fs.iter().any(|f| f.request_id == rid))
                    {
                        cancelled.insert(rid);
                    } else if let Some(id) = grants
                        .iter()
                        .find(|(_, g)| g.fetch.request_id == rid)
                        .map(|(id, _)| *id)
                    {
                        let g = grants.remove(&id).unwrap();
                        p.end_lookup(&rid, "cancelled");
                        completions.push(unused_grant(g.fetch));
                    }
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
            if p.landings.is_empty() {
                return;
            }
        }
        // Fence notifications can arrive before the plan command. Consume one
        // per plan, in FIFO order. Local device waits have no protocol deadline.
        while !device_plans.is_empty() {
            let Ok(fence) = p.fenced_plans.try_recv() else {
                break;
            };
            let (fetches, lookups, dispatch, tick) = device_plans.pop_front().unwrap();
            // The data plane can signal before the driver receives the plan.
            // The peer can start only after both sides of the handoff are ready.
            let times = (p.settings.device_admission == Some(true)).then(|| {
                (
                    dispatch.duration_since(p.origin).as_micros() as i64,
                    fence.max(dispatch).duration_since(p.origin).as_micros() as i64,
                )
            });
            if lookups.is_empty() {
                p.start_landings(fetches, p.origin.elapsed().as_micros() as i64, false, times);
                continue;
            }
            for (fetch, mut lookup) in fetches.into_iter().zip(lookups) {
                assert_eq!(fetch.lease, lookup.lease);
                if cancelled.remove(&fetch.request_id) {
                    completions.push(unused_grant(fetch));
                    continue;
                }
                let sent = p.origin.elapsed().as_micros() as i64;
                lookup.deadline_us =
                    sent.saturating_add(lookup.deadline_us.saturating_sub(tick).max(0));
                grants.insert(
                    lookup.lease.id,
                    DeviceGrant {
                        fetch,
                        allowed: lookup.allowed_sources,
                        deadline: lookup.deadline_us,
                        times: times.expect("adaptive device fence"),
                    },
                );
                p.dispatch(vec![], vec![lookup], sent);
            }
        }
        let holder_work = exports.advance(&p, &scheduler);
        // Take the clock and queue under the same lock used to stamp arrivals.
        let (now, inbox) = {
            let mut queue = p.endpoint.as_ref().unwrap().replies.lock();
            (
                p.origin.elapsed().as_micros() as i64,
                std::mem::take(&mut *queue),
            )
        };
        let mut out = PeerRound::default();
        out.events.append(&mut completions);
        let mut report = Report {
            now_us: now,
            outcomes: std::mem::take(&mut p.unplaced),
            ..Report::default()
        };
        report.outcomes.extend(p.reads.poll());
        for (at, reply) in inbox {
            if let Reply::Answer { stamp, answer } = &reply {
                if answer.kind == PeerAnswerKind::Lookup {
                    if let Some(grant) = grants.get(&answer.lease.id) {
                        if stamp.name != grant.fetch.holder || answer.lease != grant.fetch.lease {
                            continue;
                        }
                        let grant = grants.remove(&answer.lease.id).unwrap();
                        let arrival = at.duration_since(p.origin).as_micros() as i64;
                        let rid = grant.fetch.request_id.clone();
                        let tiers = answer.pages.iter().fold(0u32, |mask, page| {
                            mask | if page.tier == PeerTier::Device { 1 } else { 2 }
                        });
                        let outcome = if arrival >= grant.deadline {
                            "timeout"
                        } else if stamp.identity != p.identity || answer.num_tokens == 0 {
                            "miss"
                        } else if answer.num_tokens != grant.fetch.end_tokens {
                            "boundary"
                        } else if grant.allowed & (1 << tiers) == 0 {
                            "cost"
                        } else {
                            "ok"
                        };
                        if outcome == "timeout" {
                            p.end_lookup(&rid, "timeout");
                        } else {
                            // The protocol owns the answer. No request FSM transition or scheduler lock here.
                            p.take_reply(&reply, arrival, &mut Vec::new(), &mut Vec::new());
                        }
                        if let Some(metrics) = &p.metrics {
                            metrics.stage(
                                "grant_lookup",
                                outcome,
                                (arrival - grant.times.1).max(0) as f64 / 1e6,
                            );
                        }
                        if outcome == "ok" {
                            p.decisions.push((rid.clone(), "landing"));
                            p.start_landings(vec![grant.fetch], now, false, Some(grant.times));
                            p.landings.get_mut(&rid).unwrap().grant_source = tiers;
                        } else {
                            p.decisions.push((rid, outcome));
                            out.events.push(unused_grant(grant.fetch));
                        }
                        continue;
                    }
                    let Some(attempt) = pending.get(&answer.lease.id) else {
                        continue;
                    };
                    if attempt.lease != answer.lease || stamp.name != attempt.holder {
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
                        if p.settings.autonomous_host {
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
                if p.layouts.contains_key(&stamp.name) {
                    for (rid, landing) in &mut p.landings {
                        if landing.fetch().holder == stamp.name {
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
        for _ in 0..reservations.len().min(16) {
            let Some(mut cache) = scheduler.try_lock() else {
                break;
            };
            let (attempt, tokens, at) = reservations.pop_front().unwrap();
            let mut retry = false;
            let fetches = cache
                .reserve_peer_host(&attempt.lease, tokens, &mut retry)
                .expect("reserve Host prefix");
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
        let expired: Vec<_> = grants
            .iter()
            .filter(|(_, g)| now >= g.deadline)
            .map(|(id, _)| *id)
            .collect();
        for id in expired {
            let grant = grants.remove(&id).unwrap();
            p.end_lookup(&grant.fetch.request_id, "timeout");
            p.decisions
                .push((grant.fetch.request_id.clone(), "lookup_timeout"));
            if let Some(metrics) = &p.metrics {
                metrics.stage(
                    "grant_lookup",
                    "timeout",
                    (now - grant.times.1) as f64 / 1e6,
                );
            }
            out.events.push(unused_grant(grant.fetch));
        }
        pending.retain(|_, l| {
            if now < l.deadline {
                true
            } else {
                out.events.push(fallback(l.lease.clone()));
                false
            }
        });
        let round = p
            .advance_reports(std::slice::from_ref(&report))
            .expect("local peer report");
        out.events.extend(round.events);
        out.decisions.extend(round.decisions);
        out.landed.extend(round.landed);
        if !(out.events.is_empty() && out.decisions.is_empty() && out.landed.is_empty()) {
            let _ = results.send(out);
        }
        // Socket replies and new attempts wake us immediately. The read pool
        // returns through a channel; poll only while work or a timeout exists.
        if !holder_work
            && device_plans.is_empty()
            && reservations.is_empty()
            && grants.is_empty()
            && pending.is_empty()
            && p.landings.is_empty()
        {
            thread::park();
        } else {
            thread::park_timeout(Duration::from_millis(1));
        }
    }
}
