//! V1 control traffic and member votes ride the normal replica collective.
use super::{
    attempt::Attempt,
    offers::{Offers, Vote, CLOSED, DONE, PINNED, READY},
    read::Reads,
    registered::Replica,
    servicer::empty_event,
    v1::{Closure, Disposition, Generation, Key, Message},
    PeerError, PeerFence, PeerRound, VersionedChannel,
};
use crate::codec::{Reader, Writer};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{mpsc, Arc},
    time::Instant,
};
use ts_scheduler::{PeerEventKind, PeerEventView, PeerFetchView, PeerLookupView, Scheduler};

pub(super) struct Report {
    legacy: Vec<u8>,
    proposals: Vec<(i64, Message, Message)>,
    messages: Vec<Message>,
    votes: Vec<Vote>,
}
fn write_msg(w: &mut Writer, m: &Message) {
    w.flag(m.reply());
    w.bytes(&m.encode());
}
fn msg(r: &mut Reader<'_>) -> Option<Message> {
    let reply = r.flag()?;
    Message::decode(r.bytes()?, reply)
}
impl Report {
    fn encode(self) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(&self.legacy);
        w.count(self.proposals.len());
        for (issued_at, q, a) in self.proposals {
            w.i64(issued_at);
            write_msg(&mut w, &q);
            write_msg(&mut w, &a);
        }
        w.count(self.messages.len());
        for m in self.messages {
            write_msg(&mut w, &m);
        }
        w.count(self.votes.len());
        for v in self.votes {
            v.key.put(&mut w);
            w.u64(v.offer);
            w.u8(v.stage);
            w.u8(v.value);
        }
        w.finish()
    }
    fn decode(b: &[u8]) -> Option<Self> {
        let mut r = Reader::new(b);
        let legacy = r.bytes()?.to_vec();
        let proposals = (0..r.count()?)
            .map(|_| Some((r.i64()?, msg(&mut r)?, msg(&mut r)?)))
            .collect::<Option<Vec<_>>>()?;
        let messages = (0..r.count()?)
            .map(|_| msg(&mut r))
            .collect::<Option<Vec<_>>>()?;
        let votes = (0..r.count()?)
            .map(|_| {
                Some(Vote {
                    key: Key::get(&mut r)?,
                    offer: r.u64()?,
                    stage: r.u8()?,
                    value: r.u8()?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        r.finish(Self {
            legacy,
            proposals,
            messages,
            votes,
        })
    }
}

pub(super) struct Control {
    pub layout: Arc<Replica>,
    rank: usize,
    gen: Generation,
    ch: Option<VersionedChannel>,
    known: HashMap<String, (Generation, Arc<Replica>)>,
    attempts: BTreeMap<u64, Attempt>,
    offers: Offers,
    votes: Vec<Vote>,
    proposals: Vec<(i64, Message, Message)>,
    fences: VecDeque<Vec<u64>>,
    signaled: mpsc::Receiver<Instant>,
    missing: Vec<PeerEventView>,
    admission: bool,
    pub meter: Option<Arc<ts_metrics::PeerMetrics>>,
}
impl Control {
    pub fn new(
        layout: Arc<Replica>,
        rank: usize,
        name: String,
        incarnation: u64,
        ch: Option<VersionedChannel>,
    ) -> (Self, PeerFence) {
        let (tx, signaled) = mpsc::channel();
        (
            Self {
                layout,
                rank,
                gen: Generation {
                    name,
                    incarnation,
                    epoch: 1,
                },
                ch,
                known: HashMap::new(),
                attempts: BTreeMap::new(),
                offers: Offers::new(),
                votes: Vec::new(),
                proposals: Vec::new(),
                fences: VecDeque::new(),
                signaled,
                missing: Vec::new(),
                admission: true,
                meter: None,
            },
            PeerFence(tx),
        )
    }
    pub fn reader(&self) -> bool {
        self.layout.members.len() == 8
    }
    pub fn len(&self) -> usize {
        self.attempts.len() + usize::from(self.offers.active())
    }
    pub fn invalidate(&mut self) {
        self.admission = false;
        self.known.clear();
        for a in self.attempts.values_mut() {
            a.invalidate();
        }
    }
    fn send(&self, m: Message) {
        let Some(ch) = &self.ch else {
            return;
        };
        let to = match &m {
            Message::LayoutReq(..) | Message::LayoutReply(..) => return,
            Message::Lookup(k, ..)
            | Message::Pin(k, ..)
            | Message::Reading(k, ..)
            | Message::Release(k, ..) => &k.holder.name,
            Message::Offered(k, ..) | Message::Grant(k, ..) | Message::Ack(k, ..) => &k.reader.name,
        };
        if let Err(e) = ch.net.versioned(to, &m) {
            tracing::warn!(%e,to,"v1 peer message not queued");
        }
    }
    pub fn dispatch(
        &mut self,
        fetches: Vec<PeerFetchView>,
        lookups: Vec<PeerLookupView>,
        now: i64,
    ) {
        if !fetches.is_empty() {
            self.fences
                .push_back(fetches.iter().map(|f| f.lease.id).collect());
        }
        for f in fetches {
            let failed = PeerEventView {
                kind: PeerEventKind::FetchDone,
                request_id: f.request_id.clone(),
                lease: f.lease.clone(),
                ..empty_event()
            };
            if let Some(v) = self
                .attempts
                .get_mut(&f.lease.id)
                .and_then(|a| a.reserve(f, &self.layout, self.rank, now))
            {
                self.votes.push(v);
            } else {
                self.missing.push(failed);
            }
        }
        for q in lookups {
            let id = q.lease.id;
            if !self.admission || self.attempts.len() >= 4096 {
                continue;
            }
            let target = q.holder.clone();
            let mut a = Attempt::new(q);
            if let Some((g, source)) = self.known.get(&target) {
                if let Some(m) = a.query(&self.gen, &self.layout, g, source.clone()) {
                    self.send(m);
                }
            } else if let Some(ch) = &self.ch {
                let _ = ch
                    .net
                    .versioned(&target, &Message::LayoutReq(id, self.gen.name.clone()));
            }
            self.attempts.insert(id, a);
        }
    }
    pub fn report(&mut self, legacy: Vec<u8>) -> Vec<u8> {
        let messages = self
            .ch
            .as_ref()
            .map(|c| c.inbox.lock().drain(..).collect())
            .unwrap_or_default();
        Report {
            legacy,
            messages,
            votes: std::mem::take(&mut self.votes),
            proposals: std::mem::take(&mut self.proposals),
        }
        .encode()
    }
    pub fn unpack(reports: &[Vec<u8>]) -> Result<(Vec<Vec<u8>>, Vec<Report>), PeerError> {
        let mut rs = reports
            .iter()
            .map(|b| Report::decode(b).ok_or(PeerError::Report))
            .collect::<Result<Vec<_>, _>>()?;
        let legacy = rs
            .iter_mut()
            .map(|r| std::mem::take(&mut r.legacy))
            .collect();
        Ok((legacy, rs))
    }
    fn receive(&mut self, m: Message, s: &mut Scheduler, round: &mut PeerRound) {
        let now = round.now_us;
        match &m {
            Message::LayoutReply(id, g, sid, d, wire) => {
                if !self.admission
                    || *sid != self.layout.semantic
                    || !self
                        .attempts
                        .get(id)
                        .is_some_and(|a| a.lookup.holder == g.name)
                {
                    return;
                }
                let Some(source) = Replica::decode(*sid, wire.clone()) else {
                    return;
                };
                if *d != source.digest {
                    return;
                }
                let source = Arc::new(source);
                self.known
                    .insert(g.name.clone(), (g.clone(), source.clone()));
                if let Some(q) = self
                    .attempts
                    .get_mut(id)
                    .and_then(|a| a.query(&self.gen, &self.layout, g, source))
                {
                    self.send(q);
                }
            }
            Message::Lookup(k, ..) if self.rank == 0 => {
                let existing = self.offers.has(k);
                let reply = if self.admission {
                    self.offers.lookup(k, &self.gen, &self.layout, &m, s, now)
                } else {
                    Message::Offered(k.clone(), None)
                };
                if !existing && matches!(reply, Message::Offered(_, Some(_))) {
                    self.proposals.push((now, m.clone(), reply.clone()));
                }
                self.send(reply);
            }
            Message::Offered(k, offer) => {
                // A restarted holder can only reject a lookup carrying its old
                // generation. Rediscover after a miss, without evicting a newer layout.
                if offer.is_none()
                    && self
                        .known
                        .get(&k.holder.name)
                        .is_some_and(|(g, _)| g == &k.holder)
                {
                    self.known.remove(&k.holder.name);
                }
                if let Some(a) = self.attempts.get_mut(&k.lease) {
                    a.answer(&m, now, round);
                }
            }
            Message::Pin(k, o) if k.holder == self.gen => {
                if !self.admission {
                    self.send(Message::Grant(k.clone(), o.clone(), false));
                    return;
                }
                let (reply, v) = self.offers.pin(k, o, s, now);
                if let Some(v) = v {
                    self.votes.push(v);
                }
                if let Some(r) = reply {
                    self.send(r);
                }
            }
            Message::Grant(k, o, ok) => {
                if !*ok {
                    self.known.remove(&k.holder.name);
                }
                if let Some(r) = self
                    .attempts
                    .get_mut(&k.lease)
                    .and_then(|a| a.grant(k, o, *ok, now))
                {
                    self.send(r);
                }
            }
            Message::Release(k, id, d) => {
                if k.holder != self.gen {
                    self.send(Message::Ack(k.clone(), *id, *d, Closure::Stale));
                    return;
                }
                let (reply, v) = self.offers.close(k, *id, *d, s, now);
                if let Some(v) = v {
                    self.votes.push(v);
                }
                if let Some(r) = reply {
                    self.send(r);
                }
            }
            Message::Ack(k, id, d, c) => {
                if let Some(a) = self.attempts.get_mut(&k.lease) {
                    a.ack(k, *id, *d, *c, round);
                }
            }
            _ => (),
        }
    }
    pub fn advance(
        &mut self,
        mut reports: Vec<Report>,
        s: &mut Scheduler,
        reads: Option<&Reads>,
        round: &mut PeerRound,
    ) -> Result<(), PeerError> {
        if reports.len() != self.layout.members.len() {
            return Err(PeerError::Report);
        }
        let first = &mut reports[0];
        let now = round.now_us;
        for (issued_at, q, a) in first.proposals.drain(..) {
            if self.rank == 0 {
                continue;
            }
            self.offers.install(
                &q,
                &a,
                self.layout.contract.plan.prefix_granularity,
                self.layout.members.len(),
                issued_at,
            );
        }
        for m in first.messages.drain(..) {
            self.receive(m, s, round);
        }
        for (i, r) in reports.iter().enumerate() {
            for v in &r.votes {
                let reply = match v.stage {
                    PINNED | CLOSED => self.offers.vote(i, v, now),
                    READY | DONE => self
                        .attempts
                        .get_mut(&v.key.lease)
                        .and_then(|a| a.vote(i, v, now, round)),
                    _ => return Err(PeerError::Report),
                };
                if let Some(m) = reply {
                    if let Message::Grant(key, offer, false) = &m {
                        // Consensus refused this offer. No reader could have
                        // obtained GO, so each holder can roll back its own ticket
                        // even if the reader disappeared before sending RELEASE.
                        let (_, vote) =
                            self.offers.close(key, offer.id, Disposition::Abort, s, now);
                        self.votes.extend(vote);
                    }
                    self.send(m);
                }
            }
        }
        while !self.fences.is_empty() && self.signaled.try_recv().is_ok() {
            for id in self.fences.pop_front().unwrap() {
                if let Some(a) = self.attempts.get_mut(&id) {
                    a.fence();
                }
            }
        }
        let mut outgoing = Vec::new();
        for a in self.attempts.values_mut() {
            let (v, m) = a.tick(
                reads,
                self.meter.as_ref().map(|m| (m, self.rank)),
                now,
                round,
            );
            if let Some(v) = v {
                self.votes.push(v);
            }
            if let Some(m) = m {
                outgoing.push(m);
            }
        }
        for m in outgoing {
            self.send(m);
        }
        self.attempts.retain(|_, a| !a.finished());
        self.offers.expire(now);
        round.events.append(&mut self.missing);
        Ok(())
    }
}
