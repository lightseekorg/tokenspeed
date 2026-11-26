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

mod attempt;
mod autonomous;
mod coordinated;
mod offers;
pub use autonomous::Peer;
pub mod geometry;
pub mod landing;
pub mod layout;
pub mod net;
pub mod push;
pub mod read;
pub mod registered;
pub mod servicer;
pub mod v1;
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
    PeerAnswerKind, PeerEventKind, PeerEventView, PeerFetchView, PeerLookupView, PeerTier,
    Scheduler,
};

use crate::{
    codec::{Reader, Writer},
    peer::{
        geometry::{push_items, read_spans, RankGeometry},
        landing::{Action, Landing, Timeouts},
        net::{Inbound, PeerNet},
        push::PagePusher,
        read::{PageReader, PushJob, ReadJob, Reads},
        servicer::{empty_event, Holder},
        wire::{read_change, write_change, Prompt, Reply, Request, Stamp},
    },
};

const READ_CONCURRENCY: NonZeroUsize = NonZeroUsize::new(4).unwrap();

/// Data-transfer endpoints owned by this member's peer read workers.
pub struct PeerTransfers {
    /// Read remote Device and Host pages into local destinations.
    pub reader: Arc<dyn PageReader>,
    /// Request a holder-side push for Disk pages.
    pub pusher: Arc<dyn PagePusher>,
}

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

/// The owner of protocol progress for this replica's destination tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerProgress {
    /// All shards advance the same landing in their shared scheduling round.
    Coordinated,
    /// A single member reserves Host destinations on its protocol worker.
    Host,
    /// A single member receives writable Device grants from its scheduler.
    Device,
}

impl PeerProgress {
    fn identity_prefix(self) -> &'static str {
        match self {
            Self::Host => "async-host-v2:",
            Self::Device => "async-device-v3:",
            Self::Coordinated => "",
        }
    }

    /// Configure the scheduler and select the matching protocol owner before startup.
    pub fn for_replica(members: usize, scheduler: &mut Scheduler) -> Result<Self, PeerError> {
        match members {
            0 => Err(PeerError::Topology),
            1 if scheduler.enable_async_host_peer() => Ok(Self::Host),
            1 if scheduler.enable_async_device_peer(true) => Ok(Self::Device),
            1 => Err(PeerError::Topology),
            _ => Ok(Self::Coordinated),
        }
    }
}

impl PeerSettings {
    fn mismatch_reason(&self, progress: PeerProgress, other: &str) -> &'static str {
        let (mode, namespace) = other
            .split_once(':')
            .filter(|(m, _)| m.starts_with("async-host-v") || m.starts_with("async-device-v"))
            .unwrap_or(("", other));
        if mode != progress.identity_prefix().trim_end_matches(':') {
            "protocol_mismatch"
        } else if namespace
            .strip_prefix(&self.deployment_id)
            .is_some_and(|s| s.starts_with('/'))
        {
            "weight_mismatch"
        } else if namespace
            .strip_suffix(&self.weight_version)
            .is_some_and(|s| s.ends_with('/'))
        {
            "deployment_mismatch"
        } else {
            "namespace_mismatch"
        }
    }

    fn identity(&self, progress: PeerProgress) -> String {
        let ns = namespace(&self.deployment_id, &self.weight_version);
        format!("{}{ns}", progress.identity_prefix())
    }
}

fn namespace(deployment_id: &str, weight_version: &str) -> String {
    format!("{deployment_id}/{weight_version}")
}

/// Canonicalize DP0's non-canonical `"<host>:<port>/0"`, `"/00"`, `"/+0"`, etc.
/// aliases to `"<host>:<port>"`, matching `PeerNet::send` `u32::parse` routing.
pub fn normalize_peer_holder(endpoint: &str) -> &str {
    match endpoint.split_once('/') {
        Some((base, rank)) if !base.is_empty() && rank.parse::<u32>() == Ok(0) => base,
        _ => endpoint,
    }
}

fn endpoint_host(endpoint: &str) -> Option<&str> {
    let addr = endpoint.split_once('/').map_or(endpoint, |(base, _)| base);
    crate::peer::net::split_endpoint(addr)
        .ok()
        .map(|(host, _)| host)
}

fn same_peer_host(a: &str, b: &str) -> bool {
    matches!((endpoint_host(a), endpoint_host(b)), (Some(ha), Some(hb)) if ha == hb)
}

/// A round's agreement could not be reached.
#[derive(Debug, thiserror::Error)]
pub enum PeerError {
    /// An autonomous worker exited without reporting a result.
    #[error("peer worker disconnected")]
    WorkerExited,
    /// The native scheduler rejected an operation on the worker.
    #[error("peer worker: {0}")]
    Worker(String),
    /// Remote DMA retirement could not be established. Do not recycle these pages.
    #[error(
        "peer hard pin {0:?} was not released; replace the holder engine before reusing its cache"
    )]
    UnreleasedPin(ts_scheduler::PeerLeaseView),
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
    /// The configured peer driver cannot serve this replica.
    #[error("incompatible peer landing tier or replica size")]
    Topology,
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
    outcomes: Vec<(String, bool)>,
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
        for (request, ok) in &self.outcomes {
            out.bytes(request.as_bytes());
            out.flag(*ok);
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
            outcomes.push((request, reader.flag()?));
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
    /// Local receipt times of positive lookups on the independent owner.
    lookup_received: Vec<Instant>,
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
}

/// A bound peer endpoint, owned continuously from binding through startup.
pub struct PeerListener {
    net: Arc<PeerNet>,
    received: mpsc::Receiver<Vec<Bytes>>,
    registered: Option<Arc<registered::Replica>>,
    inbox: Arc<Mutex<VecDeque<v1::Message>>>,
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
        Ok(Self {
            net,
            received,
            registered: None,
            inbox: Arc::default(),
        })
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

    /// Attach the v1 layout only after every replica member has registered its MR.
    pub fn set_registered_layout(&mut self, layout: Arc<registered::Replica>) {
        self.registered = Some(layout);
    }

    /// Leader's versioned traffic, already routed by DP.
    pub fn versioned_channel(&self) -> VersionedChannel {
        VersionedChannel {
            net: self.net.clone(),
            inbox: self.inbox.clone(),
        }
    }
}

/// Communication endpoints owned by a registered replica leader.
pub struct VersionedChannel {
    net: Arc<PeerNet>,
    inbox: Arc<Mutex<VecDeque<v1::Message>>>,
}

#[derive(Default)]
struct DriverWake(std::sync::OnceLock<thread::Thread>);

impl DriverWake {
    fn run(&self, drive: impl FnOnce()) {
        // Earlier arrivals are in the queue; arrivals after the first drain must wake us.
        self.0.set(thread::current()).expect("one protocol worker");
        drive();
    }

    fn notify(&self) {
        if let Some(worker) = self.0.get() {
            worker.unpark();
        }
    }
}

/// The first member's endpoint and the threads around it.
struct Endpoint {
    net: Arc<PeerNet>,
    discovery: Arc<Mutex<Option<Arc<registered::Replica>>>>,
    holder: Arc<Mutex<Holder>>,
    replies: Arc<Mutex<VecDeque<(Instant, Reply)>>>,
    driver: Arc<DriverWake>,
    requests: Arc<Mutex<VecDeque<(Instant, Request)>>>,
    control: Arc<Mutex<VecDeque<(Instant, Request)>>>,
    _demux: thread::JoinHandle<()>,
    _servicer: Option<thread::JoinHandle<()>>,
}

/// Where a member's data plane reports, in plan order, that a plan with
/// landings ran its device prerequisites: the stores reading blocks the plan
/// granted, its copies and zeroing, the forwards in flight. Only then may the
/// holder's pages be read into the plan's blocks.
#[derive(Debug, Clone)]
pub struct PeerFence(mpsc::Sender<Instant>);

impl PeerFence {
    /// The oldest such plan not yet reported ran its device work.
    pub fn ready(&self) {
        // A stopped loop has nothing left to read.
        let _ = self.0.send(Instant::now());
    }
}

/// One member's peer sharing.
struct Protocol {
    progress: PeerProgress,
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
    /// This member's plans with landings, oldest first, whose device work
    /// has not reported done.
    fences: VecDeque<Vec<String>>,
    fence: PeerFence,
    fenced_plans: mpsc::Receiver<Instant>,
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
    /// Reads that could not be placed, to report as failed.
    unplaced: Vec<(String, bool)>,
    /// Synthetic miss answers for self-lookups short-circuited locally.
    self_lookup_answers: Vec<PeerEventView>,
}

impl Protocol {
    /// Start one member with every member's geometry in `replica`.
    /// The first (`member` 0) must supply the bound `listener` and serves
    /// `scheduler` as a holder; other members must supply `None`. The
    /// member uses its own transfer endpoints for direct reads and Disk pushes.
    pub fn start(
        mut settings: PeerSettings,
        progress: PeerProgress,
        member: usize,
        replica: Vec<RankGeometry>,
        listener: Option<PeerListener>,
        scheduler: Arc<Mutex<Scheduler>>,
        transfers: PeerTransfers,
    ) -> Result<Self, PeerError> {
        settings.name = normalize_peer_holder(&settings.name).to_owned();
        let members = replica.len();
        let geometry = replica.get(member).cloned().ok_or(PeerError::Report)?;
        if listener.is_some() != (member == 0) {
            return Err(PeerError::Listener);
        }
        let (fence, fenced_plans) = mpsc::channel();
        let origin = Instant::now();
        let endpoint = listener
            .map(|listener| start_endpoint(&settings, progress, replica, listener, scheduler))
            .transpose()?;
        Ok(Self {
            identity: settings.identity(progress),
            settings,
            progress,
            member,
            members,
            geometry,
            endpoint,
            origin,
            layouts: HashMap::new(),
            asked: HashMap::new(),
            landings: BTreeMap::new(),
            fences: VecDeque::new(),
            fence: PeerFence(fence),
            fenced_plans,
            fenced: HashSet::new(),
            reading: HashSet::new(),
            metrics: None,
            read_metrics: None,
            lookup_times: HashMap::new(),
            decisions: Vec::new(),
            reads: Reads::start(transfers.reader, transfers.pusher, READ_CONCURRENCY)?,
            unplaced: Vec::new(),
            self_lookup_answers: Vec::new(),
        })
    }

    /// Drain queued holder changes on the endpoint owner (empty on other members).
    pub fn take_holder_changes(&self) -> Vec<PeerEventView> {
        self.endpoint
            .as_ref()
            .map_or_else(Vec::new, |endpoint| endpoint.holder.lock().take_changes())
    }

    /// Encode holder changes for intra-replica broadcast.
    pub fn encode_changes(changes: &[PeerEventView]) -> Vec<u8> {
        if changes.is_empty() {
            return Vec::new();
        }
        let mut out = Writer::new();
        out.count(changes.len());
        for change in changes {
            write_change(&mut out, change);
        }
        out.finish()
    }

    /// Decode holder changes from intra-replica broadcast.
    pub fn decode_changes(bytes: &[u8]) -> Result<Vec<PeerEventView>, PeerError> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        let mut reader = Reader::new(bytes);
        let count = reader.count().ok_or(PeerError::Report)?;
        let mut changes = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            changes.push(read_change(&mut reader).ok_or(PeerError::Report)?);
        }
        reader.finish(changes).ok_or(PeerError::Report)
    }

    /// Enable logical measurements on member zero and physical measurements on every member.
    pub fn set_metrics(
        &mut self,
        logical: Option<Arc<ts_metrics::PeerMetrics>>,
        physical: Option<Arc<ts_metrics::PeerMetrics>>,
    ) {
        if let Some(endpoint) = &self.endpoint {
            endpoint.holder.lock().metrics.clone_from(&logical);
        }
        self.metrics = logical;
        self.read_metrics = physical;
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
            report.replies = endpoint.replies.lock().drain(..).map(|(_, r)| r).collect();
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
        self.advance_reports(&reports)
    }

    fn advance_reports(&mut self, reports: &[Report]) -> Result<PeerRound, PeerError> {
        let first = reports.first().ok_or(PeerError::Report)?;
        let now_us = first.now_us;
        self.asked.retain(|_, deadline_us| now_us < *deadline_us);
        let mut events = first.changes.clone();
        events.append(&mut self.self_lookup_answers);
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
            if self.layouts.contains_key(&landing.fetch().holder) {
                if let Some(action) = landing.layout_known(now_us) {
                    actions.push((request.clone(), action));
                }
            }
            if self.progress != PeerProgress::Coordinated && landing.unanswered_pin(now_us) {
                actions.push((request.clone(), Action::SendRelease { demote: false }));
            }
            if self.progress != PeerProgress::Coordinated {
                if let Some(retry) = landing.retry_release(now_us) {
                    actions.push((request.clone(), retry));
                }
            }
            landing.tick(now_us);
        }
        self.send(actions, now_us);
        let done: Vec<(String, i32)> = self
            .landings
            .iter()
            .filter_map(|(request, landing)| landing.done().map(|tokens| (request.clone(), tokens)))
            .collect();
        let mut landed = Vec::new();
        for (request, num_tokens) in done {
            tracing::debug!(request, num_tokens, "peer landing done");
            let landing = self.landings.remove(&request).expect("finished landing");
            if self.metrics.is_some()
                && self.member == 0
                && num_tokens == landing.fetch().end_tokens
            {
                landed.push((request.clone(), landing.fetch().begin_tokens, num_tokens));
            } else if let Some(stage) = landing.failure() {
                self.decisions.push((request.clone(), stage));
            }
            self.fenced.remove(&request);
            self.reading.remove(&request);
            events.push(PeerEventView {
                kind: PeerEventKind::FetchDone,
                dispatch_us: landing.device_times.unwrap_or_default().0,
                writable_us: landing.device_times.unwrap_or_default().1,
                terminal_us: landing
                    .device_times
                    .map_or(0, |_| landing.phase_started_us()),
                request_id: request,
                num_tokens,
                ..empty_event()
            });
        }
        while !self.fences.is_empty() && self.fenced_plans.try_recv().is_ok() {
            if let Some(requests) = self.fences.pop_front() {
                self.fenced.extend(requests);
            }
        }
        self.start_reads(now_us);
        Ok(PeerRound {
            lookup_received: Vec::new(),
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
        self.start_landings(fetches, now_us, true, None);
        for mut lookup in lookups {
            lookup.holder = normalize_peer_holder(&lookup.holder).to_owned();
            if lookup.holder == self.settings.name {
                self.self_lookup_answers.push(PeerEventView {
                    kind: PeerEventKind::LookupAnswered,
                    lease: lookup.lease,
                    num_tokens: 0,
                    ..empty_event()
                });
                continue;
            }
            if let Some(endpoint) = &self.endpoint {
                if self.metrics.is_some() {
                    self.lookup_times.insert(
                        lookup.lease.id,
                        (lookup.request_id, Instant::now(), lookup.deadline_us),
                    );
                }
                let request = Request::Lookup {
                    lease: lookup.lease,
                    prompt: Prompt {
                        prefix_hashes: lookup.prefix_hashes,
                        tokens: lookup.tokens,
                        max_tokens: lookup.max_tokens,
                    },
                };
                if let Err(error) = endpoint.net.request(&lookup.holder, &request) {
                    // The lookup's wait ends at its deadline.
                    tracing::warn!(holder = lookup.holder, %error, "peer lookup not sent");
                }
            }
        }
    }

    // Only destinations granted directly by ReservePeerHost can be ready
    // without a device plan. Device landings still wait on their plan fence.
    fn start_landings(
        &mut self,
        fetches: Vec<PeerFetchView>,
        now_us: i64,
        wait_for_plan: bool,
        device_times: Option<(i64, i64)>,
    ) {
        if wait_for_plan && !fetches.is_empty() {
            self.fences.push_back(
                fetches
                    .iter()
                    .map(|fetch| fetch.request_id.clone())
                    .collect(),
            );
        }
        let mut actions = Vec::new();
        for mut fetch in fetches {
            fetch.holder = normalize_peer_holder(&fetch.holder).to_owned();
            if !wait_for_plan {
                // The independent Device driver calls here only after the plan fence.
                debug_assert!(
                    self.progress == PeerProgress::Device
                        || fetch.destinations.iter().all(|d| d.tier == PeerTier::Host)
                );
                self.fenced.insert(fetch.request_id.clone());
            }
            let known = self.layouts.contains_key(&fetch.holder);
            let request = fetch.request_id.clone();
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
            landing.device_times = device_times;
            if self.member == 0 {
                landing.observe_with(self.metrics.clone());
            }
            self.landings.insert(request.clone(), landing);
            actions.push((request, action));
        }
        self.send(actions, now_us);
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
        self.identity = self.settings.identity(self.progress);
        self.layouts.clear();
        self.asked.clear();
        self.self_lookup_answers.clear();
        if let Some(endpoint) = &self.endpoint {
            endpoint.holder.lock().set_identity(self.identity.clone());
            *endpoint.discovery.lock() = None;
        }
    }

    /// Landings in flight.
    pub fn num_landings(&self) -> usize {
        self.landings.len()
    }

    fn is_self_stamp(&self, stamp: &Stamp) -> bool {
        normalize_peer_holder(&stamp.name) == self.settings.name
            || (stamp.incarnation == self.settings.incarnation
                && same_peer_host(&stamp.name, &self.settings.name))
    }

    fn take_reply(
        &mut self,
        reply: &Reply,
        now_us: i64,
        events: &mut Vec<PeerEventView>,
        actions: &mut Vec<(String, Action)>,
    ) {
        match reply {
            Reply::Layout { stamp, ranks } => {
                let holder_name = normalize_peer_holder(&stamp.name).to_owned();
                self.asked.remove(&holder_name);
                self.asked.remove(&stamp.name);
                if self.is_self_stamp(stamp) {
                    tracing::warn!(
                        holder = stamp.name,
                        incarnation = stamp.incarnation,
                        "ignoring peer layout reply from self"
                    );
                } else if stamp.identity == self.identity && ranks.len() == self.members {
                    self.layouts.insert(
                        holder_name,
                        Layout {
                            incarnation: stamp.incarnation,
                            ranks: ranks.clone(),
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
                let self_reply = self.is_self_stamp(stamp);
                let trusted = self.trust(stamp);
                match answer.kind {
                    PeerAnswerKind::Lookup => {
                        let valid_peer = stamp.identity == self.identity && !self_reply;
                        if self_reply {
                            tracing::warn!(
                                holder = stamp.name,
                                incarnation = stamp.incarnation,
                                "ignoring peer lookup reply from self"
                            );
                        }
                        if let (Some(metrics), Some((rid, start, _))) =
                            (&self.metrics, self.lookup_times.remove(&answer.lease.id))
                        {
                            let outcome = if !valid_peer {
                                if stamp.identity != self.identity {
                                    metrics.decision(
                                        self.settings
                                            .mismatch_reason(self.progress, &stamp.identity),
                                    );
                                }
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
                            checkpoint: if self.progress == PeerProgress::Device
                                && valid_peer
                                && answer.num_tokens > 0
                            {
                                reply.encode()
                            } else {
                                Vec::new()
                            },
                            pages: if self.progress == PeerProgress::Device && valid_peer {
                                answer.pages.clone()
                            } else {
                                Vec::new()
                            },
                            lease: answer.lease.clone(),
                            // A holder of another namespace or self holds nothing of ours.
                            num_tokens: if valid_peer { answer.num_tokens } else { 0 },
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
                            let same_checkpoint = self.progress != PeerProgress::Device
                                || matches!(
                                Reply::decode(&landing.fetch().checkpoint),
                                Ok(Reply::Answer { stamp: original, answer: expected })
                                    if original == *stamp && expected.lease == answer.lease
                                    && expected.num_tokens == answer.num_tokens && expected.pages == answer.pages);
                            if let Some(action) =
                                landing.pinned(answer.clone(), trusted && same_checkpoint, now_us)
                            {
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
    /// same incarnation and namespace; a stale layout or self-reply is rejected.
    fn trust(&mut self, stamp: &Stamp) -> bool {
        if stamp.identity != self.identity || self.is_self_stamp(stamp) {
            return false;
        }
        let holder_name = normalize_peer_holder(&stamp.name);
        match self.layouts.get(holder_name) {
            Some(layout) if layout.incarnation == stamp.incarnation => true,
            Some(_) => {
                self.layouts.remove(holder_name);
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
                Action::SendPin if self.progress == PeerProgress::Device => {
                    let fetch = landing.fetch();
                    let Ok(Reply::Answer { stamp, answer }) = Reply::decode(&fetch.checkpoint)
                    else {
                        continue;
                    };
                    if answer.kind != PeerAnswerKind::Lookup
                        || answer.lease != lease
                        || answer.num_tokens != fetch.end_tokens
                        || stamp.identity != self.identity
                    {
                        continue;
                    }
                    Request::CheckpointPin {
                        stamp,
                        checkpoint: PeerEventView {
                            kind: PeerEventKind::Pin,
                            lease,
                            num_tokens: fetch.end_tokens,
                            max_tokens: fetch.end_tokens,
                            prefix_hashes: fetch.prefix_hashes.clone(),
                            tokens: fetch.tokens.clone(),
                            pages: answer.pages,
                            ..empty_event()
                        },
                    }
                }
                Action::SendPin => Request::Pin { lease },
                Action::SendRelease { demote } => Request::Release { lease, demote },
            };
            if let Err(error) = endpoint.net.request(holder, &message) {
                if matches!(message, Request::Layout { .. }) {
                    self.asked.remove(holder);
                }
                // The landing's wait ends at its deadline.
                tracing::warn!(holder, %error, "peer request not sent");
            } else if self.progress != PeerProgress::Coordinated {
                self.landings
                    .get_mut(&request)
                    .unwrap()
                    .sent(self.origin.elapsed().as_micros() as i64);
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
            let Some(pinned) = landing.pages_to_read() else {
                continue;
            };
            let fetch = landing.fetch();
            if let Some(metrics) = &self.read_metrics {
                metrics.fence(
                    &fetch.holder,
                    self.member,
                    (now_us - landing.phase_started_us()) as f64 / 1e6,
                );
            }
            let holder = self
                .layouts
                .get(&fetch.holder)
                .and_then(|layout| layout.ranks.get(self.member));
            let pages: Vec<_> = fetch
                .destinations
                .iter()
                .filter_map(|destination| {
                    pinned
                        .pages
                        .iter()
                        .find(|page| {
                            page.group_id == destination.group_id && page.slot == destination.slot
                        })
                        .map(|page| (*page, *destination))
                })
                .collect();
            // The holder's Disk pages are pushed, the rest read.
            let (disk_pages, read_pages): (Vec<_>, Vec<_>) = pages
                .into_iter()
                .partition(|(source, _)| source.tier == PeerTier::Disk);
            let budget =
                Duration::from_micros(u64::try_from(self.settings.timeouts.push_us).unwrap_or(0));
            let job = holder
                .ok_or_else(|| "no layout of the holder".to_owned())
                .and_then(|holder| {
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
                            budget,
                            items: push_items(holder, &self.geometry, &disk_pages)
                                .map_err(|error| error.to_string())?,
                        })
                    };
                    Ok((holder.session.clone(), spans, push))
                });
            self.reading.insert(request.clone());
            match job {
                Ok((session, spans, push)) => {
                    let observation = self.read_metrics.as_ref().map(|metrics| {
                        metrics.queue_read(
                            fetch.holder.clone(),
                            self.member,
                            spans.total_bytes() + push.as_ref().map_or(0, PushJob::bytes),
                        )
                    });
                    self.reads.submit_observed(
                        ReadJob {
                            request_id: request.clone(),
                            session,
                            spans,
                            push,
                        },
                        observation,
                    );
                }
                Err(error) => {
                    // Nothing was posted: a failed read with no late writes,
                    // reported with the next round.
                    tracing::warn!(request, %error, "peer read not placed");
                    self.unplaced.push((request.clone(), false));
                }
            }
        }
    }
}

fn start_endpoint(
    settings: &PeerSettings,
    progress: PeerProgress,
    replica: Vec<RankGeometry>,
    listener: PeerListener,
    scheduler: Arc<Mutex<Scheduler>>,
) -> Result<Endpoint, PeerError> {
    let PeerListener {
        net,
        received,
        registered,
        inbox,
    } = listener;
    let sharded = registered
        .as_ref()
        .is_some_and(|layout| layout.members.len() == 8);
    let discovery = Arc::new(Mutex::new(registered));
    let generation = v1::Generation {
        name: settings.name.clone(),
        incarnation: settings.incarnation,
        epoch: 1,
    };
    let holder = Arc::new(Mutex::new(Holder::new(
        Stamp {
            name: settings.name.clone(),
            incarnation: settings.incarnation,
            identity: settings.identity(progress),
        },
        replica,
        settings.record_lifetime,
    )));
    let replies = Arc::new(Mutex::new(VecDeque::new()));
    let driver = Arc::new(DriverWake::default());
    let autonomous = progress != PeerProgress::Coordinated;
    let requests = Arc::new(Mutex::new(VecDeque::new()));
    let control = Arc::new(Mutex::new(VecDeque::new()));
    let (holder_tx, holder_rx) = if autonomous {
        (None, None)
    } else {
        let (tx, rx) = mpsc::channel::<Request>();
        (Some(tx), Some(rx))
    };
    // Fast non-blocking demux thread: delivers `Inbound::Reply`, routes
    // autonomous holder queues, and in coordinated mode answers
    // `Request::Layout` immediately without ever blocking on `scheduler.lock()`.
    let demux = {
        let discovery = discovery.clone();
        let (requests, control) = (requests.clone(), control.clone());
        let (incoming, wake) = (replies.clone(), driver.clone());
        let (net, holder) = (Arc::downgrade(&net), holder.clone());
        thread::Builder::new()
            .name("ts-peer-demux".to_owned())
            .spawn(move || {
                for frames in received {
                    if let [kind, payload] = frames.as_slice() {
                        if kind.as_ref() == b"q" && payload.first() == Some(&0x10) {
                            if let (Some(v1::Message::LayoutReq(id, to)), Some(layout)) = (
                                v1::Message::decode(payload, false),
                                discovery.lock().clone(),
                            ) {
                                let reply = v1::Message::LayoutReply(
                                    id,
                                    generation.clone(),
                                    layout.semantic,
                                    layout.digest,
                                    layout.descriptor.clone(),
                                );
                                let Some(net) = net.upgrade() else { return };
                                if let Err(error) = net.versioned(&to, &reply) {
                                    tracing::warn!(to, %error, "peer layout reply not sent");
                                }
                            }
                            continue;
                        }
                        if payload
                            .first()
                            .is_some_and(|tag| (0x11..=0x18).contains(tag))
                        {
                            if matches!(kind.as_ref(), b"q" | b"r") {
                                if let Some(message) =
                                    v1::Message::decode(payload, kind.as_ref() == b"r")
                                {
                                    let mut queue = inbox.lock();
                                    if queue.len() < 128 {
                                        queue.push_back(message);
                                    }
                                }
                            }
                            continue;
                        }
                        if sharded {
                            continue;
                        }
                    }
                    match Inbound::decode(&frames) {
                        Ok(Inbound::Request(request)) if autonomous => {
                            if matches!(
                                request,
                                Request::Reading { .. }
                                    | Request::Release { .. }
                                    | Request::Layout { .. }
                            ) {
                                control.lock().push_back((Instant::now(), request));
                            } else {
                                requests.lock().push_back((Instant::now(), request));
                            }
                            wake.notify();
                        }
                        Ok(Inbound::Request(Request::Reading { .. })) => continue,
                        Ok(Inbound::Request(Request::Layout { reply_to })) => {
                            let reply = holder.lock().layout();
                            let Some(net) = net.upgrade() else { return };
                            if let Err(error) = net.reply(&reply_to, &reply) {
                                tracing::warn!(to = reply_to, %error, "peer layout reply not sent");
                            }
                        }
                        Ok(Inbound::Request(request)) => {
                            if holder_tx
                                .as_ref()
                                .is_some_and(|tx| tx.send(request).is_err())
                            {
                                return;
                            }
                        }
                        Ok(Inbound::Reply(reply)) => {
                            incoming.lock().push_back((Instant::now(), reply));
                            wake.notify();
                        }
                        Err(_) => tracing::warn!("a malformed peer message dropped"),
                    }
                }
            })?
    };
    // Coordinated holder RPC worker: answers `Lookup`/`Pin`/`Release` RPCs
    // between control-loop rounds, releasing `scheduler.lock()` after each RPC
    // so a burst of probes does not stall the holder's next plan or commit.
    let servicer = holder_rx
        .map(|holder_rx| {
            let (net, holder) = (Arc::downgrade(&net), holder.clone());
            thread::Builder::new()
                .name("ts-peer-servicer".to_owned())
                .spawn(move || {
                    while let Ok(first) = holder_rx.recv() {
                        let mut next = Some(first);
                        while let Some(request) = next {
                            let read = matches!(&request, Request::Release { demote: true, .. });
                            let ((to, reply), metrics) = {
                                let scheduler = scheduler.lock();
                                let mut holder = holder.lock();
                                let now = Instant::now();
                                (
                                    holder.handle(&*scheduler, request, now),
                                    holder.metrics.clone(),
                                )
                            };
                            if let Some(m) = metrics {
                                reply.observe_holder(&m, read);
                            }
                            let Some(net) = net.upgrade() else { return };
                            if let Err(error) = net.reply(&to, &reply) {
                                tracing::warn!(to, %error, "peer reply not sent");
                            }
                            next = holder_rx.try_recv().ok();
                        }
                    }
                })
        })
        .transpose()?;
    Ok(Endpoint {
        discovery,
        net,
        holder,
        replies,
        driver,
        requests,
        control,
        _demux: demux,
        _servicer: servicer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_arrival_after_an_empty_drain_wakes_the_new_worker() {
        let wake = DriverWake::default();
        let (send, queue) = mpsc::channel();
        let (empty, drained) = mpsc::channel();
        let (handled, result) = mpsc::channel();
        thread::scope(|scope| {
            let wake_ref = &wake;
            let worker = scope.spawn(move || {
                wake_ref.run(|| {
                    assert!(queue.try_recv().is_err());
                    empty.send(()).unwrap();
                    loop {
                        thread::park();
                        if let Ok(message) = queue.try_recv() {
                            handled.send(message).unwrap();
                            break;
                        }
                    }
                })
            });
            drained.recv().unwrap();
            send.send(17).unwrap();
            wake.notify();
            let received = result.recv_timeout(Duration::from_secs(3));
            // Let a broken worker exit as well, so a regression fails instead of hanging CI.
            worker.thread().unpark();
            assert_eq!(received.unwrap(), 17);
        });
    }

    #[test]
    fn distinguishes_identity_components_visible_in_lookup_replies() {
        let settings = PeerSettings {
            name: "reader".into(),
            incarnation: 1,
            deployment_id: "model".into(),
            weight_version: "one".into(),
            record_lifetime: Duration::from_secs(2),
            timeouts: Timeouts {
                answer_us: 10,
                quarantine_us: 20,
                push_us: 30,
            },
        };
        for (other, reason) in [
            ("model/one", "protocol_mismatch"),
            ("async-host-v2:model/one", "protocol_mismatch"),
            ("async-device-v3:model/two", "weight_mismatch"),
            ("async-device-v3:other/one", "deployment_mismatch"),
            ("async-device-v3:other/two", "namespace_mismatch"),
        ] {
            assert_eq!(
                settings.mismatch_reason(PeerProgress::Device, other),
                reason
            );
        }
    }
}
