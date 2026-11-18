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
pub mod read;
pub mod servicer;
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
use ts_scheduler::{
    PeerAnswerKind, PeerEventKind, PeerEventView, PeerFetchView, PeerLookupView, Scheduler,
};

use crate::{
    codec::{Reader, Writer},
    peer::{
        geometry::{read_spans, RankGeometry},
        landing::{Action, Landing, Timeouts},
        net::{Inbound, PeerNet},
        read::{PageReader, ReadJob, Reads},
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
    /// The host the endpoint binds.
    pub bind_host: String,
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
    /// The agreed peer clock.
    pub now_us: i64,
    /// The holder's changes first, then the reader's answers and landings.
    pub events: Vec<PeerEventView>,
}

/// A holder's layout, as a reader knows it.
#[derive(Debug, Clone)]
struct Layout {
    incarnation: u64,
    ranks: Vec<RankGeometry>,
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
    fenced_plans: mpsc::Receiver<()>,
    /// Landings whose plan's device work has run on this member.
    fenced: HashSet<String>,
    /// Landings this member started reading.
    reading: HashSet<String>,
    reads: Reads,
    /// Reads that could not be placed, to report as failed.
    unplaced: Vec<(String, bool)>,
}

impl Peer {
    /// One member of a replica of `members`; the first (`member` 0) binds
    /// the endpoint at `port` and serves `scheduler` as a holder, with every
    /// member's geometry in `replica`.
    pub fn start(
        settings: PeerSettings,
        member: usize,
        replica: Vec<RankGeometry>,
        port: u16,
        scheduler: Arc<Mutex<Scheduler>>,
        reader: Arc<dyn PageReader>,
    ) -> Result<Self, PeerError> {
        let members = replica.len();
        let geometry = replica.get(member).cloned().ok_or(PeerError::Report)?;
        let (fence, fenced_plans) = mpsc::channel();
        let endpoint = if member == 0 {
            Some(start_endpoint(&settings, replica, port, scheduler)?)
        } else {
            None
        };
        Ok(Self {
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
            fences: VecDeque::new(),
            fence: PeerFence(fence),
            fenced_plans,
            fenced: HashSet::new(),
            reading: HashSet::new(),
            reads: Reads::start(reader, READ_CONCURRENCY)?,
            unplaced: Vec::new(),
        })
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
            report.replies = endpoint.replies.try_iter().collect();
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
        self.send(actions, now_us);
        let done: Vec<(String, i32)> = self
            .landings
            .iter()
            .filter_map(|(request, landing)| landing.done().map(|tokens| (request.clone(), tokens)))
            .collect();
        for (request, num_tokens) in done {
            self.landings.remove(&request);
            self.fenced.remove(&request);
            self.reading.remove(&request);
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
        self.start_reads();
        Ok(PeerRound { now_us, events })
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
                    .map(|fetch| fetch.request_id.clone())
                    .collect(),
            );
        }
        let mut actions = Vec::new();
        for fetch in fetches {
            let known = self.layouts.contains_key(&fetch.holder);
            let request = fetch.request_id.clone();
            let (landing, action) =
                Landing::new(fetch, self.members, known, now_us, self.settings.timeouts);
            self.landings.insert(request.clone(), landing);
            actions.push((request, action));
        }
        self.send(actions, now_us);
        if let Some(endpoint) = &self.endpoint {
            for lookup in lookups {
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
        self.layouts.clear();
        self.asked.clear();
        if let Some(endpoint) = &self.endpoint {
            endpoint.holder.lock().set_identity(self.identity.clone());
        }
    }

    /// Landings in flight.
    pub fn num_landings(&self) -> usize {
        self.landings.len()
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
                self.asked.remove(&stamp.name);
                if stamp.identity == self.identity && ranks.len() == self.members {
                    self.layouts.insert(
                        stamp.name.clone(),
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
                let trusted = self.trust(stamp);
                match answer.kind {
                    PeerAnswerKind::Lookup => events.push(PeerEventView {
                        kind: PeerEventKind::LookupAnswered,
                        lease: answer.lease.clone(),
                        // A holder of another namespace holds nothing of ours.
                        num_tokens: if stamp.identity == self.identity {
                            answer.num_tokens
                        } else {
                            0
                        },
                        ..empty_event()
                    }),
                    PeerAnswerKind::Pin => {
                        if let Some((request, landing)) = self
                            .landings
                            .iter_mut()
                            .find(|(_, landing)| *landing.lease() == answer.lease)
                        {
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
            if let Err(error) = endpoint.net.request(holder, &message) {
                if matches!(message, Request::Layout { .. }) {
                    self.asked.remove(holder);
                }
                // The landing's wait ends at its deadline.
                tracing::warn!(holder, %error, "peer request not sent");
            }
        }
    }

    /// Start this member's reads of the landings that are pinned and whose
    /// plan's device work has run here.
    fn start_reads(&mut self) {
        for (request, landing) in &self.landings {
            if self.reading.contains(request) || !self.fenced.contains(request) {
                continue;
            }
            let Some(pinned) = landing.pages_to_read() else {
                continue;
            };
            let fetch = landing.fetch();
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
            let job = holder
                .ok_or_else(|| "no layout of the holder".to_owned())
                .and_then(|holder| {
                    read_spans(holder, &self.geometry, &pages)
                        .map(|spans| (holder.session.clone(), spans))
                        .map_err(|error| error.to_string())
                });
            self.reading.insert(request.clone());
            match job {
                Ok((session, spans)) => self.reads.submit(ReadJob {
                    request_id: request.clone(),
                    session,
                    spans,
                }),
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
    replica: Vec<RankGeometry>,
    port: u16,
    scheduler: Arc<Mutex<Scheduler>>,
) -> Result<Endpoint, PeerError> {
    let (incoming, received) = mpsc::channel::<Vec<Bytes>>();
    let net = Arc::new(PeerNet::start(&settings.bind_host, port, incoming)?);
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
        let (net, holder) = (net.clone(), holder.clone());
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
