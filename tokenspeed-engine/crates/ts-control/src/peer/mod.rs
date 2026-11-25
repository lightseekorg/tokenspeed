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

//! Sharing cached prefixes between engine replicas, as the control loop runs
//! it (`docs/rfcs/0004-peer-prefix-sharing.md`).
//!
//! The replica's first member owns the endpoint. As a holder it answers
//! other replicas' requests on a servicer thread, with reads of the scheduler
//! between the control thread's rounds ([`servicer::Holder`]); as a reader it
//! sends its scheduler's lookups and drives each landing's pin and release.
//! Every round the members share what they saw -- the first member's clock,
//! the holder's queued changes and the replies it received, each member's
//! read outcomes -- and step every landing alike from that agreement
//! ([`landing::Landing`]). Each member reads its own shard of the holder,
//! from the holder's member of the same index, once its plan's device work
//! has run.

pub mod geometry;
pub mod landing;
pub mod net;
pub mod prepared;
pub mod push;
pub mod read;
pub mod servicer;
pub mod storage;
pub mod storage_servicer;
pub mod wire;

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    num::NonZeroUsize,
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use parking_lot::Mutex;
use ts_collective::{Group, Mesh};
use ts_scheduler::{
    PeerAnswerKind, PeerEventKind, PeerEventView, PeerFetchView, PeerLeaseView, PeerLookupView,
    PeerTier, Scheduler,
};

use crate::{
    codec::{Reader, Writer},
    peer::{
        geometry::{push_items, read_spans, RankGeometry},
        landing::{Action, Landing, Timeouts},
        net::{Inbound, PeerNet},
        push::PagePusher,
        read::{PageReader, PushJob, ReadJob, ReadOutcome, Reads},
        servicer::{empty_event, Holder},
        wire::{read_change, write_change, Prompt, Reply, Request, Stamp},
    },
};

const READ_CONCURRENCY: NonZeroUsize = NonZeroUsize::new(4).unwrap();

/// What a replica's peer sharing runs with.
#[derive(Debug, Clone)]
pub struct PeerSettings {
    /// The replica's name: its endpoint, `host:port`.
    pub name: String,
    /// The process's incarnation.
    pub incarnation: u64,
    /// The cache namespace's deployment id: everything that shapes a
    /// cached page's bytes but the weights.
    pub deployment_id: String,
    /// The weight version the namespace starts at.
    pub weight_version: String,
    /// The waits a landing allows itself.
    pub timeouts: Timeouts,
    /// How long a holder keeps a lease's record past its last request.
    pub record_lifetime: Duration,
}

impl PeerSettings {
    /// The namespace: only a holder of the same identity is read.
    fn identity(&self) -> String {
        namespace(&self.deployment_id, &self.weight_version)
    }
}

fn namespace(deployment_id: &str, weight_version: &str) -> String {
    format!("{deployment_id}/{weight_version}")
}

/// A round's agreement could not be reached.
#[derive(Debug, thiserror::Error)]
pub enum PeerError {
    /// A member's report did not decode.
    #[error("a malformed peer report")]
    Report,
    /// The endpoint did not start.
    #[error("the peer endpoint: {0}")]
    Net(#[from] ts_pd::NetError),
    /// Only the first member owns the replica's listener.
    #[error("the first peer member must own a listener, and other members must not")]
    Listener,
    /// The local replica leaders could not exchange endpoints.
    #[error("peer node listener setup: {0}")]
    Mesh(#[from] ts_collective::Error),
    /// Every member stops before freeing destinations that may still be written.
    #[error("native peer read has not retired for {0}; worker restart required")]
    Unretired(String),
    /// A thread did not start.
    #[error("starting a peer thread: {0}")]
    Thread(#[from] std::io::Error),
}

/// What one member shares each round.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Report {
    now_us: i64,
    /// The first member's: the holder's changes since the last round.
    changes: Vec<PeerEventView>,
    /// The first member's: the replies it received.
    replies: Vec<Reply>,
    /// This member's reads that ended, by landing request.
    outcomes: Vec<(String, ReadOutcome)>,
}

impl Report {
    fn encode(&self) -> Vec<u8> {
        let mut out = Writer::new();
        out.i64(self.now_us);
        out.count(self.changes.len());
        for change in &self.changes {
            write_change(&mut out, change);
        }
        out.count(self.replies.len());
        for reply in &self.replies {
            out.bytes(&reply.encode());
        }
        out.count(self.outcomes.len());
        for (request, outcome) in &self.outcomes {
            out.bytes(request.as_bytes());
            out.u8(match outcome {
                ReadOutcome::Complete => 0,
                ReadOutcome::FailedRetired => 1,
                ReadOutcome::Unretired => 2,
            });
        }
        out.finish()
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut reader = Reader::new(bytes);
        let now_us = reader.i64()?;
        let count = reader.count()?;
        let mut changes = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            changes.push(read_change(&mut reader)?);
        }
        let count = reader.count()?;
        let mut replies = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            replies.push(Reply::decode(reader.bytes()?).ok()?);
        }
        let count = reader.count()?;
        let mut outcomes = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            let request = String::from_utf8(reader.bytes()?.to_vec()).ok()?;
            let outcome = match reader.u8()? {
                0 => ReadOutcome::Complete,
                1 => ReadOutcome::FailedRetired,
                2 => ReadOutcome::Unretired,
                _ => return None,
            };
            outcomes.push((request, outcome));
        }
        reader.finish(Self {
            now_us,
            changes,
            replies,
            outcomes,
        })
    }
}

/// What a round's agreement feeds the scheduler: the agreed clock, then the
/// events in order (`Scheduler::advance_peer`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PeerRound {
    /// The agreed peer clock.
    pub now_us: i64,
    /// The holder's changes first, then the reader's answers and landings.
    pub events: Vec<PeerEventView>,
    /// Confirmed remote intervals for per-request usage attribution.
    pub landed: Vec<(String, i32, i32)>,
    /// Attribution for lookup decisions and failed landings on the reporting owner.
    pub decisions: Vec<(String, &'static str)>,
}

/// A holder's layout, as a reader knows it.
#[derive(Debug, Clone)]
struct Layout {
    incarnation: u64,
    ranks: Vec<RankGeometry>,
    storage: bool,
}

/// A bound peer endpoint, owned continuously from binding through startup.
pub struct PeerListener {
    net: Arc<PeerNet>,
    received: mpsc::Receiver<Vec<Bytes>>,
}

/// The public listener dispatches both holder requests and reader replies.
struct NodeRoutes {
    own_rank: u32,
    local_ports: HashMap<u32, u16>,
}

impl NodeRoutes {
    fn dispatch(&self, net: &PeerNet, mut frames: Vec<Bytes>) -> Option<Vec<Bytes>> {
        let rank = match frames.as_slice() {
            // The original two-frame address names DP rank zero.
            [_, _] => 0,
            [destination, _, _] => {
                let Ok(bytes) = destination.as_ref().try_into() else {
                    tracing::warn!(
                        bytes = destination.len(),
                        "invalid peer destination rank frame"
                    );
                    return None;
                };
                let rank = u32::from_be_bytes(bytes);
                frames.remove(0);
                rank
            }
            _ => {
                tracing::warn!(frames = frames.len(), "unexpected peer message frame count");
                return None;
            }
        };
        if rank == self.own_rank {
            return Some(frames);
        }
        if let Some(port) = self.local_ports.get(&rank) {
            if let Err(error) = net.forward(*port, frames) {
                tracing::warn!(rank, %error, "peer message could not be forwarded");
            }
        } else {
            tracing::warn!(
                rank,
                own_rank = self.own_rank,
                "peer destination rank is not on this node"
            );
        }
        None
    }

    /// Route public traffic independently of the local holder's scheduler lock.
    fn start(
        self,
        net: &Arc<PeerNet>,
        incoming: mpsc::Receiver<Vec<Bytes>>,
    ) -> std::io::Result<mpsc::Receiver<Vec<Bytes>>> {
        let net = Arc::downgrade(net);
        let (local, received) = mpsc::channel();
        thread::Builder::new()
            .name("ts-peer-router".into())
            .spawn(move || {
                for frames in incoming {
                    let Some(net) = net.upgrade() else { break };
                    if let Some(frames) = self.dispatch(&net, frames) {
                        if local.send(frames).is_err() {
                            break;
                        }
                    }
                }
            })?;
        Ok(received)
    }
}

impl PeerListener {
    /// Bind on `host` at `port` (0 lets the OS choose) and return the listener.
    /// Its port stays reserved while the caller prepares the replica's name
    /// and scheduler, then moves the listener into `Peer::start`.
    pub fn bind(host: &str, port: u16) -> Result<Self, PeerError> {
        let (incoming, received) = mpsc::channel();
        let net = Arc::new(PeerNet::start(host, port, incoming)?);
        Ok(Self { net, received })
    }

    /// Bind one public control endpoint for the node. Its replica leaders
    /// exchange private loopback ports once, before they advertise readiness.
    pub fn bind_node(
        mesh: &mut Mesh,
        group_id: u32,
        leaders: &[(usize, u32)],
        host: &str,
        port: u16,
    ) -> Result<Self, PeerError> {
        let ranks = leaders.iter().map(|&(rank, _)| rank).collect();
        let group = Group::new(group_id, ranks, mesh.rank())?;
        // Reserve the public port before the OS assigns any loopback ports.
        let public = if group.index() == 0 {
            Some(Self::bind(host, port)?)
        } else {
            None
        };
        mesh.broadcast(&group, 0, Vec::new())?;
        let mut listener = match public {
            Some(listener) => listener,
            None => Self::bind("127.0.0.1", 0)?,
        };
        let ports = mesh.all_gather_i64(&group, &[i64::from(listener.port())])?;
        if group.index() == 0 {
            let routes = NodeRoutes {
                own_rank: leaders[0].1,
                local_ports: leaders
                    .iter()
                    .zip(ports)
                    .skip(1)
                    .map(|(&(_, rank), row)| (rank, row[0] as u16))
                    .collect(),
            };
            listener.received = routes.start(&listener.net, listener.received)?;
        }
        Ok(listener)
    }

    /// Return the bound port, including the OS-selected port when binding to 0.
    pub fn port(&self) -> u16 {
        self.net.port()
    }
}

/// The first member's endpoint and the threads around it.
struct Endpoint {
    net: Arc<PeerNet>,
    holder: Arc<Mutex<Holder>>,
    replies: mpsc::Receiver<Reply>,
    _servicer: thread::JoinHandle<()>,
}

/// Where a member's data plane reports, in plan order, that a plan with
/// landings ran its device prerequisites: the stores reading blocks the plan
/// granted, its copies and zeroing, the forwards in flight. Only then may the
/// holder's pages be read into the plan's blocks.
#[derive(Debug, Clone)]
pub struct PeerFence(mpsc::Sender<()>);

impl PeerFence {
    /// The oldest such plan not yet reported ran its device work.
    pub fn ready(&self) {
        // A stopped loop has nothing left to read.
        let _ = self.0.send(());
    }
}

/// One member's peer sharing.
pub struct Peer {
    storage_source: Option<storage_servicer::Service>,
    settings: PeerSettings,
    /// The namespace this member reads and serves under; a weight update
    /// moves it.
    identity: String,
    member: usize,
    members: usize,
    geometry: RankGeometry,
    endpoint: Option<Endpoint>,
    origin: Instant,
    layouts: HashMap<String, Layout>,
    /// Outstanding layout queries and their deadlines on the agreed clock.
    asked: HashMap<String, i64>,
    landings: BTreeMap<String, Landing>,
    /// One ordinary admission may read disjoint pages from several sources.
    /// Keys are source lease ids; the scheduler receives one completion only
    /// after every source is retired and its release acknowledged.
    landing_groups: BTreeMap<String, Vec<String>>,
    prepared_sources: HashMap<PeerLeaseView, Vec<Vec<prepared::PreparedObject>>>,
    /// This member's plans with landings, oldest first, whose device work
    /// has not reported done.
    fences: VecDeque<Vec<String>>,
    fence: PeerFence,
    fenced_plans: mpsc::Receiver<()>,
    /// Landings whose plan's device work has run on this member.
    fenced: HashSet<String>,
    /// Landings this member started reading.
    reading: HashSet<String>,
    reads: Reads,
    metrics: Option<Arc<ts_metrics::PeerMetrics>>,
    read_metrics: Option<Arc<ts_metrics::PeerMetrics>>,
    // Only the endpoint owner measures logical attempts.
    lookup_times: HashMap<u64, (String, Instant, i64)>,
    decisions: Vec<(String, &'static str)>,
    next_sample_us: i64,
    previous_totals: [u64; 2],
    /// Reads that could not be placed, to report as failed.
    unplaced: Vec<(String, ReadOutcome)>,
}

impl Peer {
    /// Start one member with every member's geometry in `replica`.
    /// The first (`member` 0) must supply the bound `listener` and serves
    /// `scheduler` as a holder; other members must supply `None`. The
    /// member reads through `reader` and has holders' Disk pages pushed
    /// through `pusher` (its rank's push endpoint).
    pub fn start(
        settings: PeerSettings,
        member: usize,
        replica: Vec<RankGeometry>,
        listener: Option<PeerListener>,
        scheduler: Arc<Mutex<Scheduler>>,
        reader: Arc<dyn PageReader>,
        pusher: Arc<dyn PagePusher>,
    ) -> Result<Self, PeerError> {
        let members = replica.len();
        let geometry = replica.get(member).cloned().ok_or(PeerError::Report)?;
        if listener.is_some() != (member == 0) {
            return Err(PeerError::Listener);
        }
        let (fence, fenced_plans) = mpsc::channel();
        let endpoint = listener
            .map(|listener| start_endpoint(&settings, replica, listener, scheduler))
            .transpose()?;
        Ok(Self {
            storage_source: None,
            identity: settings.identity(),
            settings,
            member,
            members,
            geometry,
            endpoint,
            origin: Instant::now(),
            layouts: HashMap::new(),
            asked: HashMap::new(),
            landings: BTreeMap::new(),
            landing_groups: BTreeMap::new(),
            prepared_sources: HashMap::new(),
            fences: VecDeque::new(),
            fence: PeerFence(fence),
            fenced_plans,
            fenced: HashSet::new(),
            reading: HashSet::new(),
            metrics: None,
            read_metrics: None,
            lookup_times: HashMap::new(),
            decisions: Vec::new(),
            next_sample_us: 0,
            previous_totals: [0; 2],
            reads: Reads::start(reader, pusher, READ_CONCURRENCY)?,
            unplaced: Vec::new(),
        })
    }

    /// Own the startup-selected storage source broker on member zero. It uses
    /// ordinary peer admission and owns no scheduler destination memory.
    pub fn attach_storage_source(&mut self, source: storage_servicer::Service) {
        assert!(self.member == 0 && self.storage_source.is_none());
        self.storage_source = Some(source);
    }

    /// Enable logical measurements on member zero and physical measurements on every member.
    pub fn set_metrics(
        &mut self,
        logical: Option<Arc<ts_metrics::PeerMetrics>>,
        physical: Option<Arc<ts_metrics::PeerMetrics>>,
    ) {
        self.metrics = logical;
        self.read_metrics = physical;
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

    /// Drop an unanswered lookup when its owning request terminates.
    pub fn end_lookup(&mut self, rid: &str, outcome: &str) {
        if let Some(metrics) = &self.metrics {
            self.lookup_times.retain(|_, (owner, start, _)| {
                if owner != rid {
                    return true;
                }
                metrics.stage("lookup", outcome, start.elapsed().as_secs_f64());
                false
            });
        }
    }

    /// The bound port of the first member's endpoint.
    pub fn port(&self) -> Option<u16> {
        self.endpoint.as_ref().map(|endpoint| endpoint.net.port())
    }

    /// Gate new Device pins on the replica's holder while its host drains a
    /// replay fence.
    pub fn block_device_pins(&self, blocked: bool) {
        if let Some(endpoint) = &self.endpoint {
            endpoint.holder.lock().block_device_pins(blocked);
        }
    }

    /// This member's report for the round.
    pub fn report(&mut self) -> Vec<u8> {
        let mut outcomes = std::mem::take(&mut self.unplaced);
        outcomes.extend(self.reads.poll());
        let mut report = Report {
            now_us: i64::try_from(self.origin.elapsed().as_micros()).unwrap_or(i64::MAX),
            outcomes,
            ..Report::default()
        };
        if let Some(endpoint) = &self.endpoint {
            report.changes = endpoint.holder.lock().take_changes();
            report.replies = endpoint.replies.try_iter().collect();
        }
        if let Some(storage) = &self.storage_source {
            report.replies.extend(storage.poll());
        }
        report.encode()
    }

    /// The round's agreement over every member's report, in member order;
    /// steps every landing, sends what the steps ask for and starts this
    /// member's reads.
    pub fn agree(&mut self, reports: &[Vec<u8>]) -> Result<PeerRound, PeerError> {
        let reports = reports
            .iter()
            .map(|bytes| Report::decode(bytes).ok_or(PeerError::Report))
            .collect::<Result<Vec<_>, _>>()?;
        let first = reports.first().ok_or(PeerError::Report)?;
        let now_us = first.now_us;
        self.asked.retain(|_, deadline_us| now_us < *deadline_us);
        let mut events = first.changes.clone();
        let mut actions = Vec::new();
        for reply in &first.replies {
            self.take_reply(reply, now_us, &mut events, &mut actions);
        }
        if let Some(metrics) = &self.metrics {
            self.lookup_times.retain(|_, (rid, start, deadline)| {
                if now_us < *deadline {
                    return true;
                }
                self.decisions.push((rid.clone(), "lookup_timeout"));
                metrics.stage("lookup", "timeout", start.elapsed().as_secs_f64());
                false
            });
        }
        for (member, report) in reports.iter().enumerate() {
            for (request, ok) in &report.outcomes {
                if let Some(landing) = self.landings.get_mut(request) {
                    if let Some(action) = landing.read(member, *ok, now_us) {
                        actions.push((request.clone(), action));
                    }
                }
            }
        }
        for (request, landing) in &mut self.landings {
            landing.tick(now_us);
            if self.layouts.contains_key(&landing.fetch().holder) {
                if let Some(action) = landing.layout_known(now_us) {
                    actions.push((request.clone(), action));
                }
            }
        }
        if let Some((_, landing)) = self
            .landings
            .iter()
            .find(|(_, landing)| landing.unretired())
        {
            // Do not send release or FetchDone. C++ still owns the landing's
            // pages, including after request cancellation. All mirrored ranks
            // observe the same terminal failure before another plan can run.
            return Err(PeerError::Unretired(landing.fetch().request_id.clone()));
        }
        self.send(actions, now_us);
        let done: Vec<String> = self
            .landing_groups
            .iter()
            .filter(|(_, keys)| keys.iter().all(|key| self.landings[key].done().is_some()))
            .map(|(request, _)| request.clone())
            .collect();
        let mut landed = Vec::new();
        for request in done {
            let keys = self
                .landing_groups
                .remove(&request)
                .expect("finished landing group");
            let begin = keys
                .iter()
                .map(|key| self.landings[key].fetch().begin_tokens)
                .min()
                .expect("nonempty group");
            let end = keys
                .iter()
                .map(|key| self.landings[key].fetch().end_tokens)
                .max()
                .expect("nonempty group");
            let complete = keys.iter().all(|key| {
                let landing = &self.landings[key];
                landing.done() == Some(landing.fetch().end_tokens)
            });
            let num_tokens = if complete { end } else { begin };
            tracing::debug!(request, num_tokens, "peer landing done");
            if self.metrics.is_some() && self.member == 0 && complete {
                landed.push((request.clone(), begin, end));
            }
            for key in keys {
                let landing = self.landings.remove(&key).expect("finished source landing");
                if let Some(stage) = landing.failure() {
                    self.decisions.push((request.clone(), stage));
                }
                self.prepared_sources.remove(landing.lease());
                self.fenced.remove(&key);
                self.reading.remove(&key);
            }
            events.push(PeerEventView {
                kind: PeerEventKind::FetchDone,
                request_id: request,
                num_tokens,
                ..empty_event()
            });
        }
        while self.fenced_plans.try_recv().is_ok() {
            if let Some(requests) = self.fences.pop_front() {
                self.fenced.extend(requests);
            }
        }
        self.start_reads(now_us);
        Ok(PeerRound {
            now_us,
            events,
            landed,
            decisions: std::mem::take(&mut self.decisions),
        })
    }

    /// A plan's landings and lookups: the landings start, and the first
    /// member sends the lookups. Every member calls this with the same plan.
    pub fn dispatch(
        &mut self,
        fetches: Vec<PeerFetchView>,
        lookups: Vec<PeerLookupView>,
        now_us: i64,
    ) {
        if !fetches.is_empty() {
            self.fences.push_back(
                fetches
                    .iter()
                    .map(|fetch| fetch.lease.id.to_string())
                    .collect(),
            );
        }
        let mut actions = Vec::new();
        for fetch in fetches {
            let known = self.layouts.contains_key(&fetch.holder);
            let request = fetch.request_id.clone();
            let key = fetch.lease.id.to_string();
            self.landing_groups
                .entry(request.clone())
                .or_default()
                .push(key.clone());
            tracing::debug!(
                request,
                holder = fetch.holder,
                begin = fetch.begin_tokens,
                end = fetch.end_tokens,
                layout_known = known,
                "peer landing started"
            );
            let (mut landing, action) =
                Landing::new(fetch, self.members, known, now_us, self.settings.timeouts);
            if self.member == 0 {
                landing.observe_with(self.metrics.clone());
            }
            assert!(
                self.landings.insert(key.clone(), landing).is_none(),
                "duplicate source lease"
            );
            actions.push((key, action));
        }
        self.send(actions, now_us);
        if let Some(endpoint) = &self.endpoint {
            for lookup in lookups {
                if self.metrics.is_some() {
                    self.lookup_times.insert(
                        lookup.lease.id,
                        (lookup.request_id, Instant::now(), lookup.deadline_us),
                    );
                }
                let prompt = Prompt {
                    prefix_hashes: lookup.prefix_hashes,
                    tokens: lookup.tokens,
                    max_tokens: lookup.max_tokens,
                };
                let result = if let Some(storage) = self
                    .storage_source
                    .as_ref()
                    .filter(|s| s.name() == lookup.holder)
                {
                    storage
                        .lookup(lookup.lease, prompt, lookup.reused_pages)
                        .map_err(|e| e.to_string())
                } else {
                    endpoint
                        .net
                        .request(
                            &lookup.holder,
                            &Request::Lookup {
                                lease: lookup.lease,
                                prompt,
                            },
                        )
                        .map_err(|e| e.to_string())
                };
                if let Err(error) = result {
                    // The lookup's wait ends at its deadline.
                    tracing::warn!(holder = lookup.holder, %error, "peer lookup not sent");
                }
            }
        }
    }

    /// Where this member's data plane reports its plans' device work; the
    /// reads it allows start with the next round's agreement.
    pub fn fence(&self) -> PeerFence {
        self.fence.clone()
    }

    /// The replica committed new weights under `weight_version`: it reads
    /// and serves only that namespace from now on. The flush that goes with
    /// the update left no landing or pin behind; the holders' layouts are
    /// fetched again, so none of the old namespace is read.
    pub fn set_weight_version(&mut self, weight_version: &str) {
        self.settings.weight_version = weight_version.to_owned();
        self.identity = self.settings.identity();
        if let Some(storage) = &self.storage_source {
            storage.set_namespace(self.identity.clone(), weight_version.to_owned());
        }
        self.layouts.clear();
        self.asked.clear();
        if let Some(endpoint) = &self.endpoint {
            endpoint.holder.lock().set_identity(self.identity.clone());
        }
    }

    /// Landings in flight.
    pub fn num_landings(&self) -> usize {
        self.landing_groups.len()
    }

    fn take_reply(
        &mut self,
        reply: &Reply,
        now_us: i64,
        events: &mut Vec<PeerEventView>,
        actions: &mut Vec<(String, Action)>,
    ) {
        match reply {
            Reply::StorageLayout { stamp, members } => {
                self.asked.remove(&stamp.name);
                if stamp.identity == self.identity && *members as usize == self.members {
                    self.layouts.insert(
                        stamp.name.clone(),
                        Layout {
                            incarnation: stamp.incarnation,
                            ranks: Vec::new(),
                            storage: true,
                        },
                    );
                }
            }
            Reply::Prepared {
                stamp,
                lease,
                num_tokens,
                ranks,
            } => {
                let trusted =
                    self.trust(stamp) && self.layouts.get(&stamp.name).is_some_and(|l| l.storage);
                if let Some((request, landing)) = self.landings.iter_mut().find(|(_, landing)| {
                    landing.lease() == lease && landing.fetch().holder == stamp.name
                }) {
                    if let Some(action) =
                        landing.pinned_prepared(*num_tokens, ranks, trusted, now_us)
                    {
                        actions.push((request.clone(), action));
                    } else if trusted
                        && landing.ready_to_read()
                        && !self.prepared_sources.contains_key(lease)
                    {
                        self.prepared_sources.insert(lease.clone(), ranks.clone());
                    }
                }
            }
            Reply::Layout { stamp, ranks } => {
                self.asked.remove(&stamp.name);
                if stamp.identity == self.identity && ranks.len() == self.members {
                    self.layouts.insert(
                        stamp.name.clone(),
                        Layout {
                            incarnation: stamp.incarnation,
                            ranks: ranks.clone(),
                            storage: false,
                        },
                    );
                } else {
                    tracing::warn!(
                        holder = stamp.name,
                        "peer holder of another namespace or shape"
                    );
                }
            }
            Reply::Answer { stamp, answer } => {
                let trusted = self.trust(stamp);
                match answer.kind {
                    PeerAnswerKind::Lookup => {
                        if let (Some(metrics), Some((rid, start, _))) =
                            (&self.metrics, self.lookup_times.remove(&answer.lease.id))
                        {
                            let outcome = if stamp.identity != self.identity {
                                "incompatible"
                            } else if answer.num_tokens == 0 {
                                "miss"
                            } else {
                                "ok"
                            };
                            let decision = match outcome {
                                "ok" => "admission",
                                "miss" => "lookup_miss",
                                _ => "incompatible",
                            };
                            self.decisions.push((rid, decision));
                            metrics.stage("lookup", outcome, start.elapsed().as_secs_f64());
                        }
                        events.push(PeerEventView {
                            kind: PeerEventKind::LookupAnswered,
                            lease: answer.lease.clone(),
                            // A holder of another namespace holds nothing of ours.
                            num_tokens: if stamp.identity == self.identity {
                                answer.num_tokens
                            } else {
                                0
                            },
                            ..empty_event()
                        });
                    }
                    PeerAnswerKind::Pin => {
                        if let Some((request, landing)) = self
                            .landings
                            .iter_mut()
                            .find(|(_, landing)| *landing.lease() == answer.lease)
                        {
                            tracing::debug!(
                                request,
                                held = answer.held,
                                trusted,
                                num_tokens = answer.num_tokens,
                                "peer pin answered"
                            );
                            if let Some(action) = landing.pinned(answer.clone(), trusted, now_us) {
                                actions.push((request.clone(), action));
                            }
                        }
                    }
                    _ => {
                        if let Some(landing) = self
                            .landings
                            .values_mut()
                            .find(|landing| *landing.lease() == answer.lease)
                        {
                            landing.released(answer, trusted, now_us);
                        }
                    }
                }
            }
        }
    }

    /// Whether an answer comes from the holder whose layout is known, in the
    /// same incarnation and namespace; a stale layout is forgotten.
    fn trust(&mut self, stamp: &Stamp) -> bool {
        if stamp.identity != self.identity {
            return false;
        }
        match self.layouts.get(&stamp.name) {
            Some(layout) if layout.incarnation == stamp.incarnation => true,
            Some(_) => {
                self.layouts.remove(&stamp.name);
                false
            }
            None => false,
        }
    }

    /// The first member sends what the steps asked for.
    fn send(&mut self, actions: Vec<(String, Action)>, now_us: i64) {
        let Some(endpoint) = &self.endpoint else {
            return;
        };
        for (request, action) in actions {
            let Some(landing) = self.landings.get(&request) else {
                continue;
            };
            let holder = &landing.fetch().holder;
            let lease = landing.lease().clone();
            let message = match action {
                Action::SendLayout => {
                    if self.asked.contains_key(holder) {
                        continue;
                    }
                    self.asked.insert(
                        holder.clone(),
                        now_us.saturating_add(self.settings.timeouts.answer_us),
                    );
                    Request::Layout {
                        reply_to: self.settings.name.clone(),
                    }
                }
                Action::SendPin => Request::Pin { lease },
                Action::SendRelease { demote } => Request::Release { lease, demote },
            };
            let layout = matches!(message, Request::Layout { .. });
            let result = if let Some(storage) =
                self.storage_source.as_ref().filter(|s| s.name() == holder)
            {
                match message {
                    Request::Pin { lease } => storage.prepare(lease, &landing.fetch().destinations),
                    other => storage.request(other),
                }
                .map_err(|e| e.to_string())
            } else {
                endpoint
                    .net
                    .request(holder, &message)
                    .map_err(|e| e.to_string())
            };
            if let Err(error) = result {
                if layout {
                    self.asked.remove(holder);
                }
                // The landing's wait ends at its deadline.
                tracing::warn!(holder, %error, "peer request not sent");
            }
        }
    }

    /// Start this member's reads of the landings that are pinned and whose
    /// plan's device work has run here.
    fn start_reads(&mut self, now_us: i64) {
        for (request, landing) in &self.landings {
            if self.reading.contains(request) || !self.fenced.contains(request) {
                continue;
            }
            if !landing.ready_to_read() {
                continue;
            }
            let fetch = landing.fetch();
            if let Some(metrics) = &self.read_metrics {
                metrics.fence(
                    &fetch.holder,
                    self.member,
                    (now_us - landing.phase_started_us()) as f64 / 1e6,
                );
            }
            let job = if let Some(ranks) = self.prepared_sources.get(landing.lease()) {
                ranks
                    .get(self.member)
                    .ok_or_else(|| "missing prepared owner".to_owned())
                    .and_then(|objects| {
                        prepared::read_spans(objects, &self.geometry, &fetch.destinations)
                            .map(|spans| (spans.into_iter().collect::<Vec<_>>(), None))
                            .map_err(|error| error.to_string())
                    })
            } else {
                self.layouts
                    .get(&fetch.holder)
                    .and_then(|layout| layout.ranks.get(self.member))
                    .zip(landing.pages_to_read())
                    .ok_or_else(|| "no cache layout or pinned pages of the holder".to_owned())
                    .and_then(|(holder, pinned)| {
                        let pages: Vec<_> = fetch
                            .destinations
                            .iter()
                            .filter_map(|destination| {
                                pinned
                                    .pages
                                    .iter()
                                    .find(|page| {
                                        page.group_id == destination.group_id
                                            && page.slot == destination.slot
                                    })
                                    .map(|page| (*page, *destination))
                            })
                            .collect();
                        let (disk_pages, read_pages): (Vec<_>, Vec<_>) = pages
                            .into_iter()
                            .partition(|(source, _)| source.tier == PeerTier::Disk);
                        let spans = read_spans(holder, &self.geometry, &read_pages)
                            .map_err(|error| error.to_string())?;
                        let push = if disk_pages.is_empty() {
                            None
                        } else {
                            Some(PushJob {
                                endpoint: holder.push.clone().ok_or_else(|| {
                                    "the holder named Disk pages without a push endpoint".to_owned()
                                })?,
                                session: self.geometry.session.clone(),
                                budget: Duration::from_micros(
                                    u64::try_from(self.settings.timeouts.push_us).unwrap_or(0),
                                ),
                                items: push_items(holder, &self.geometry, &disk_pages)
                                    .map_err(|error| error.to_string())?,
                            })
                        };
                        Ok((vec![(holder.session.clone(), spans)], push))
                    })
            };
            self.reading.insert(request.clone());
            match job {
                Ok((sources, push)) => {
                    let observation = self.read_metrics.as_ref().map(|metrics| {
                        metrics.queue_read(
                            fetch.holder.clone(),
                            self.member,
                            sources
                                .iter()
                                .map(|(_, spans)| spans.total_bytes())
                                .sum::<u64>()
                                + push.as_ref().map_or(0, PushJob::bytes),
                        )
                    });
                    self.reads.submit_observed(
                        ReadJob {
                            request_id: request.clone(),
                            sources,
                            push,
                        },
                        observation,
                    );
                }
                Err(error) => {
                    // Nothing was posted: a failed read with no late writes,
                    // reported with the next round.
                    tracing::warn!(request, %error, "peer read not placed");
                    self.unplaced
                        .push((request.clone(), ReadOutcome::FailedRetired));
                }
            }
        }
    }
}

fn start_endpoint(
    settings: &PeerSettings,
    replica: Vec<RankGeometry>,
    listener: PeerListener,
    scheduler: Arc<Mutex<Scheduler>>,
) -> Result<Endpoint, PeerError> {
    let PeerListener { net, received } = listener;
    let holder = Arc::new(Mutex::new(Holder::new(
        Stamp {
            name: settings.name.clone(),
            incarnation: settings.incarnation,
            identity: settings.identity(),
        },
        replica,
        settings.record_lifetime,
    )));
    let (replied, replies) = mpsc::channel();
    let servicer = {
        let (net, holder) = (Arc::downgrade(&net), holder.clone());
        thread::Builder::new()
            .name("ts-peer-servicer".to_owned())
            .spawn(move || {
                for frames in received {
                    match Inbound::decode(&frames) {
                        Ok(Inbound::Request(request)) => {
                            // Waits while the control thread runs a round's
                            // scheduler work; answers between rounds.
                            let (to, reply) = {
                                let scheduler = scheduler.lock();
                                holder.lock().handle(&*scheduler, request, Instant::now())
                            };
                            let Some(net) = net.upgrade() else { return };
                            if let Err(error) = net.reply(&to, &reply) {
                                tracing::warn!(to, %error, "peer reply not sent");
                            }
                        }
                        Ok(Inbound::Reply(reply)) => {
                            if replied.send(reply).is_err() {
                                return;
                            }
                        }
                        Err(_) => tracing::warn!("a malformed peer message dropped"),
                    }
                }
            })?
    };
    Ok(Endpoint {
        net,
        holder,
        replies,
        _servicer: servicer,
    })
}
