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
    retirement_bound_ms: u32,
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
        w.u32(self.retirement_bound_ms);
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
        let retirement_bound_ms = r.u32()?;
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
            retirement_bound_ms,
            proposals,
            messages,
            votes,
        })
    }
}

/// Timed-out closes retain only the message, never a lookup, layout or landing.
#[derive(Default)]
struct PendingReleases {
    holders: HashMap<String, HolderReleases>,
}
struct HolderReleases {
    messages: BTreeMap<u64, Message>,
    retry_us: i64,
    last_sent: u64,
}

impl PendingReleases {
    fn insert(&mut self, release: Message, now: i64) {
        let Message::Release(key, _, _) = &release else {
            unreachable!()
        };
        let holder = self.holders.entry(key.holder.name.clone()).or_insert_with(|| {
            tracing::warn!(holder = %key.holder.name, "peer close unconfirmed; pausing new lookups to holder");
            HolderReleases { messages: BTreeMap::new(), last_sent: 0, retry_us: now.saturating_add(1_000_000) }
        });
        holder.messages.insert(key.lease, release);
    }
    fn retry(&mut self, now: i64, outgoing: &mut Vec<Message>) {
        for holder in self.holders.values_mut() {
            if now >= holder.retry_us {
                // Rotate so an unreachable lease cannot monopolize this endpoint.
                use std::ops::Bound::{Excluded, Unbounded};
                let next = holder
                    .messages
                    .range((Excluded(holder.last_sent), Unbounded))
                    .next()
                    .or_else(|| holder.messages.first_key_value());
                if let Some((&id, release)) = next {
                    outgoing.push(release.clone());
                    holder.last_sent = id;
                }
                holder.retry_us = now.saturating_add(if holder.messages.len() == 1 {
                    5_000_000
                } else {
                    100_000
                });
            }
        }
    }
    fn ack(&mut self, key: &Key, offer: u64, disposition: Disposition, closure: Closure) -> bool {
        if closure == Closure::Unknown {
            return false;
        }
        let Some(holder) = self.holders.get_mut(&key.holder.name) else {
            return false;
        };
        if !matches!(holder.messages.get(&key.lease), Some(Message::Release(k, id, d)) if k == key && *id == offer && *d == disposition)
        {
            return false;
        }
        holder.messages.remove(&key.lease);
        if holder.messages.is_empty() {
            self.holders.remove(&key.holder.name);
        }
        true
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
    pending_releases: PendingReleases,
    cap_warn_after_us: i64,
    votes: Vec<Vote>,
    proposals: Vec<(i64, Message, Message)>,
    fences: VecDeque<Vec<u64>>,
    signaled: mpsc::Receiver<Instant>,
    missing: Vec<PeerEventView>,
    admission: bool,
    retirement_bound_ms: u32,
    pub meter: Option<Arc<ts_metrics::PeerMetrics>>,
}
impl Control {
    pub fn new(
        layout: Arc<Replica>,
        rank: usize,
        name: String,
        incarnation: u64,
        ch: Option<VersionedChannel>,
        device_pin_gate: Arc<super::servicer::DevicePinGate>,
        retirement_bound_ms: u32,
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
                offers: Offers::new(device_pin_gate),
                pending_releases: PendingReleases::default(),
                cap_warn_after_us: 0,
                votes: Vec::new(),
                proposals: Vec::new(),
                fences: VecDeque::new(),
                signaled,
                missing: Vec::new(),
                admission: true,
                retirement_bound_ms,
                meter: None,
            },
            PeerFence(tx),
        )
    }
    pub fn reader(&self) -> bool {
        self.layout.members.len() == 8
    }
    pub fn len(&self) -> usize {
        self.attempts.len() + self.offers.active_count()
    }
    pub fn invalidate(&mut self) {
        if self.admission {
            tracing::warn!(
                "weight generation changed; restart to resume cross-layout peer sharing"
            );
        }
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
            Message::Lookup(k, ..) | Message::Pin(k, ..) | Message::Release(k, ..) => {
                &k.holder.name
            }
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
            if !self.admission || self.pending_releases.holders.contains_key(&q.holder) {
                continue;
            }
            if self.attempts.len() >= 4096 {
                if self.rank == 0 && now >= self.cap_warn_after_us {
                    tracing::warn!("peer attempt limit reached; falling back to local prefill");
                    self.cap_warn_after_us = now.saturating_add(60_000_000);
                }
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
            retirement_bound_ms: self.retirement_bound_ms,
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
                    || self
                        .attempts
                        .get(id)
                        .is_none_or(|a| a.lookup.holder != g.name)
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
                } else if self.pending_releases.ack(k, *id, *d, *c)
                    && *c == Closure::Stale
                    && self
                        .known
                        .get(&k.holder.name)
                        .is_some_and(|(g, _)| g == &k.holder)
                {
                    self.known.remove(&k.holder.name);
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
        let retirement_bound_ms = first.retirement_bound_ms;
        // Coordinated rounds share rank zero's reported clock. Use it for both
        // proposal timestamps and expiry, even if member process origins differ.
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
        self.attempts.retain(|_, attempt| {
            if let Some(release) = attempt.take_pending_release(now) {
                self.pending_releases.insert(release, now);
            }
            !attempt.finished()
        });
        self.pending_releases.retry(now, &mut outgoing);
        for m in outgoing {
            self.send(m);
        }
        self.votes
            .extend(self.offers.expire(now, retirement_bound_ms, s));
        round.events.append(&mut self.missing);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_releases_retry_fairly_and_only_matching_terminal_acks_unblock_a_holder() {
        let generation = |name: &str| Generation {
            name: name.into(),
            incarnation: 1,
            epoch: 1,
        };
        let key = Key {
            reader: generation("reader"),
            holder: generation("offline"),
            lease: 1,
        };
        let second = Key {
            lease: 2,
            ..key.clone()
        };
        let disposition = Disposition::Abort;
        let first_release = Message::Release(key.clone(), 11, disposition);
        let second_release = Message::Release(second.clone(), 12, disposition);
        let mut pending = PendingReleases::default();
        pending.insert(first_release.clone(), 0);
        pending.insert(second_release.clone(), 0);
        assert!(!pending.holders.contains_key("healthy"));
        let mut out = Vec::new();
        for now in [0, 1_000_000, 1_000_001, 1_100_000, 1_200_000] {
            pending.retry(now, &mut out);
        }
        assert_eq!(
            out,
            vec![first_release.clone(), second_release, first_release]
        );
        assert!(!pending.ack(&key, 12, disposition, Closure::Closed));
        assert!(!pending.ack(&key, 11, disposition, Closure::Unknown));
        assert!(!pending.ack(&key, 11, Disposition::Complete, Closure::Closed));
        let mut stale_key = key.clone();
        stale_key.holder.epoch += 1;
        assert!(!pending.ack(&stale_key, 11, disposition, Closure::Closed));
        assert!(pending.ack(&second, 12, disposition, Closure::Stale));
        assert!(pending.holders.contains_key("offline"));
        assert!(pending.ack(&key, 11, disposition, Closure::Closed));
        assert!(pending.holders.is_empty());
    }
}
