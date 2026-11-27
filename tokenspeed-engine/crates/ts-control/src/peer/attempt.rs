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

//! Reader reservations, pin/GO agreement and asynchronous retirement.
use super::{
    coordinated::ReadPath,
    geometry::Spans,
    layout::PeerReadPlan,
    offers::{manifest, prefix, Vote, DONE, FREEZE, READY},
    push::PushTarget,
    read::{MemberRead, ReadRetirement, Reads},
    registered::Replica,
    servicer::empty_event,
    v1::{Closure, Disposition, Generation, Key, Message, Offer},
    wire::Prompt,
    PeerRound,
};
use std::{
    collections::HashMap,
    sync::{mpsc, Arc},
    time::Duration,
};
use ts_scheduler::{
    PeerEventKind, PeerEventView, PeerFetchView, PeerLookupView, PeerPageView, PeerTier,
};

struct MemberPlan {
    reads: Vec<(String, Spans)>,
    pushes: Vec<PushTarget>,
}

impl MemberPlan {
    fn is_empty(&self) -> bool {
        self.reads.is_empty() && self.pushes.is_empty()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Discovery,
    Query,
    Offered,
    Planning,
    Pin,
    Renewing,
    Reading,
    Quarantine,
    Closing(Disposition),
    Finished,
}
pub(super) struct Attempt {
    pub lookup: PeerLookupView,
    key: Option<Key>,
    source: Option<Arc<Replica>>,
    offer: Option<Offer>,
    pages: Vec<PeerPageView>,
    phase: Phase,
    fetch: Option<PeerFetchView>,
    plan: Option<MemberPlan>,
    job: Option<MemberRead>,
    votes: [Option<u8>; 8],
    fenced: bool,
    voted: bool,
    freeze_voted: bool,
    published_hit: Option<bool>,
    read_path: Option<ReadPath>,
    deadline: i64,
    retry: i64,
    submit_until: std::time::Instant,
    renewed: bool,
    pin_started_us: Option<i64>,
    retired_us: Option<i64>,
    retire_wait: bool,
    invalid: bool,
    frozen: bool,
    granted: bool,
}

fn event(kind: PeerEventKind, f: &PeerFetchView, end: i32) -> PeerEventView {
    PeerEventView {
        kind,
        request_id: f.request_id.clone(),
        lease: f.lease.clone(),
        num_tokens: end,
        ..empty_event()
    }
}
impl Attempt {
    pub fn new(lookup: PeerLookupView) -> Self {
        let deadline = lookup.deadline_us;
        Self {
            lookup,
            key: None,
            source: None,
            offer: None,
            pages: Vec::new(),
            phase: Phase::Discovery,
            fetch: None,
            plan: None,
            job: None,
            votes: [None; 8],
            fenced: false,
            voted: false,
            freeze_voted: false,
            published_hit: None,
            read_path: None,
            deadline,
            retry: 0,
            submit_until: std::time::Instant::now() + super::v1::SUBMIT_WINDOW,
            renewed: false,
            pin_started_us: None,
            retired_us: None,
            retire_wait: false,
            invalid: false,
            frozen: false,
            granted: false,
        }
    }
    /// Once all DMA is retired and the business deadline has passed, only the
    /// exact RELEASE needs to survive. Never detach a Reading/frozen attempt.
    pub fn take_pending_release(&mut self, now: i64) -> Option<Message> {
        let Phase::Closing(disposition) = self.phase else {
            return None;
        };
        if self.published_hit.is_none() || now < self.deadline {
            return None;
        }
        let release = Message::Release(self.key.take()?, self.offer.take()?.id, disposition);
        self.phase = Phase::Finished;
        Some(release)
    }
    pub fn set_lookup_deadline(&mut self, deadline: i64) {
        if matches!(self.phase, Phase::Discovery | Phase::Query) {
            self.deadline = deadline;
        }
    }
    pub fn finished(&self) -> bool {
        self.phase == Phase::Finished
    }
    pub fn query(
        &mut self,
        local: &Generation,
        layout: &Replica,
        holder: &Generation,
        source: Arc<Replica>,
    ) -> Option<Message> {
        if self.phase != Phase::Discovery
            || self.lookup.holder != holder.name
            || source.semantic != layout.semantic
        {
            return None;
        }
        let key = Key {
            reader: local.clone(),
            holder: holder.clone(),
            lease: self.lookup.lease.id,
        };
        let n = self.lookup.max_tokens as usize;
        if n > self.lookup.tokens.len() {
            return None;
        }
        let prompt = Prompt {
            max_tokens: self.lookup.max_tokens,
            prefix_hashes: self.lookup.prefix_hashes.clone(),
            tokens: self.lookup.tokens[..n].to_vec(),
        };
        let msg = Message::Lookup(
            key.clone(),
            layout.semantic,
            layout.digest,
            source.digest,
            prompt,
        );
        self.key = Some(key);
        self.source = Some(source);
        self.phase = Phase::Query;
        Some(msg)
    }
    pub fn awaiting_answer(&self, k: &Key, received: i64) -> bool {
        self.phase == Phase::Query && self.key.as_ref() == Some(k) && received < self.deadline
    }
    pub fn answer(&mut self, m: &Message, now: i64, economical: bool, round: &mut PeerRound) {
        let Message::Offered(k, offered) = m else {
            return;
        };
        if self.phase != Phase::Query || self.key.as_ref() != Some(k) || now >= self.deadline {
            return;
        }
        let mut e = PeerEventView {
            kind: PeerEventKind::LookupAnswered,
            lease: self.lookup.lease.clone(),
            ..empty_event()
        };
        self.phase = Phase::Finished;
        if let Some((offer, id, pages)) = offered {
            let end = offer.end as usize;
            if end <= self.lookup.max_tokens as usize
                && end <= self.lookup.tokens.len()
                && prefix(&self.lookup.tokens[..end]) == *id
                && manifest(pages) == offer.manifest
                && self.source.as_ref().is_some_and(|s| {
                    super::layout::covers_interval(&s.contract, 0, offer.end, pages)
                        && (s.can_push_disk() || pages.iter().all(|p| p.tier != PeerTier::Disk))
                })
            {
                if !economical {
                    round
                        .decisions
                        .push((self.lookup.request_id.clone(), "adaptive_recompute"));
                    round.events.push(e);
                    return;
                }
                e.num_tokens = offer.end;
                e.pages = pages.clone();
                e.checkpoint = m.encode();
                self.offer = Some(offer.clone());
                self.pages = pages.clone();
                self.phase = Phase::Offered;
                self.deadline = round.now_us.saturating_add(2_000_000);
            }
        }
        round.events.push(e);
    }
    /// The scheduler keeps the exact positive answer with the waiting request.
    /// Rehydrate it after the short-lived protocol attempt has been collected.
    pub fn restore(
        f: &PeerFetchView,
        local: &Generation,
        known: &(Generation, Arc<Replica>),
    ) -> Option<Self> {
        let Message::Offered(key, Some((offer, digest, pages))) =
            Message::decode(&f.checkpoint, true)?
        else {
            return None;
        };
        if &key.reader != local
            || key.holder != known.0
            || key.lease != f.lease.id
            || f.lease.reader != local.name
            || f.holder != key.holder.name
            || f.end_tokens != offer.end
            || f.tokens.len() != offer.end as usize
            || prefix(&f.tokens) != digest
            || manifest(&pages) != offer.manifest
            || (!known.1.can_push_disk() && pages.iter().any(|p| p.tier == PeerTier::Disk))
            || !super::layout::covers_interval(&known.1.contract, 0, offer.end, &pages)
        {
            return None;
        }
        let mut a = Self::new(PeerLookupView {
            request_id: f.request_id.clone(),
            deadline_us: 0,
            lease: f.lease.clone(),
            holder: f.holder.clone(),
            max_tokens: offer.end,
            local_tokens: f.begin_tokens,
            tokens: Vec::new(),
            prefix_hashes: Vec::new(),
        });
        a.key = Some(key);
        a.offer = Some(offer);
        a.source = Some(known.1.clone());
        a.pages = pages;
        a.phase = Phase::Offered;
        Some(a)
    }
    pub fn expired(&mut self, k: &Key, o: &Offer, local: &Replica, now: i64) -> Option<Message> {
        if self.phase != Phase::Pin
            || self.key.as_ref() != Some(k)
            || self.offer.as_ref() != Some(o)
        {
            return None;
        }
        if self.renewed || self.invalid {
            self.seal();
            return None;
        }
        let f = self.fetch.as_ref()?;
        let source = self.source.as_ref()?;
        self.renewed = true;
        self.phase = Phase::Renewing;
        self.deadline = now.saturating_add(2_000_000);
        Some(Message::Lookup(
            k.clone(),
            local.semantic,
            local.digest,
            source.digest,
            Prompt {
                max_tokens: f.end_tokens,
                prefix_hashes: f.prefix_hashes.clone(),
                tokens: f.tokens.clone(),
            },
        ))
    }
    pub fn renewed_offer(&mut self, m: &Message, now: i64) -> Option<Message> {
        let Message::Offered(k, answer) = m else {
            return None;
        };
        if self.phase != Phase::Renewing || self.key.as_ref() != Some(k) {
            return None;
        }
        let old = self.offer.as_ref()?;
        if let Some((o, digest, pages)) = answer {
            if o.id <= old.id {
                return None;
            } // delayed original answer
            if !self.invalid
                && now < self.deadline
                && o.end == old.end
                && o.manifest == old.manifest
                && manifest(pages) == old.manifest
                && prefix(&self.fetch.as_ref()?.tokens) == *digest
            {
                self.offer = Some(o.clone());
                self.phase = Phase::Pin;
                self.deadline = now.saturating_add(2_000_000);
                return Some(Message::Pin(k.clone(), o.clone()));
            }
        }
        self.seal();
        None
    }
    pub fn estimated_read_path(
        &self,
        key: &Key,
        local: &Replica,
        end: i32,
        pages: &[PeerPageView],
    ) -> Option<ReadPath> {
        use ts_pd::layout::manifest::logical_slot_ranges;
        let begin = self.lookup.local_tokens;
        if begin < 0 || begin >= end {
            return None;
        }
        let slots = local
            .contract
            .group_specs
            .iter()
            .map(|group| logical_slot_ranges(group, begin.into(), end.into(), false))
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let tiers = pages.iter().filter_map(|page| {
            let ranges = slots.get(page.group_id as usize)?;
            ranges
                .iter()
                .any(|r| r.contains(&i64::from(page.slot)))
                .then_some(page.tier)
        });
        ReadPath::from_tiers(key, self.source.as_ref()?.members.len(), tiers)
    }

    // A frozen landing has already fallen back even if DMA retirement or
    // a release ACK arrives later. Consume the clock only once per attempt.
    pub fn take_read_sample(&mut self, now: i64) -> Option<(ReadPath, i32, i64, bool)> {
        let hit = self.published_hit?;
        let path = self.read_path?;
        let f = self.fetch.as_ref()?;
        let elapsed = self
            .retired_us
            .unwrap_or(now)
            .checked_sub(self.pin_started_us.take()?)?;
        Some((path, f.end_tokens - f.begin_tokens, elapsed, hit))
    }
    pub fn reserve(
        &mut self,
        f: PeerFetchView,
        local: &Replica,
        rank: usize,
        now: i64,
    ) -> Option<Vote> {
        let k = self.key.as_ref()?;
        let o = self.offer.as_ref()?;
        if self.phase != Phase::Offered {
            return None;
        }
        let src = self.source.as_ref()?;
        let mut plan = None;
        if f.end_tokens == o.end
            && f.begin_tokens < o.end
            && f.holder == k.holder.name
            && f.lease == self.lookup.lease
            && super::layout::covers_interval(
                &local.contract,
                f.begin_tokens,
                f.end_tokens,
                &f.destinations,
            )
        {
            let by_slot: HashMap<_, _> = self
                .pages
                .iter()
                .map(|p| ((p.group_id, p.slot), *p))
                .collect();
            let pairs = f
                .destinations
                .iter()
                .map(|p| Some((*by_slot.get(&(p.group_id, p.slot))?, *p)))
                .collect::<Option<Vec<_>>>();
            if let Some(pairs) = pairs {
                let endpoints: Vec<_> = src.members.iter().map(|m| m.endpoint()).collect();
                let local_member = &local.members[rank];
                match PeerReadPlan::new(&src.contract, &local.contract, endpoints.len(), 8, rank)
                    .and_then(|planner| planner.plan(&endpoints, &local_member.endpoint(), &pairs))
                {
                    Ok(planned) => {
                        let reads = planned
                            .reads
                            .into_iter()
                            .map(|(i, s)| (src.members[i].geometry.session.clone(), s))
                            .collect::<Vec<_>>();
                        let pushes = planned
                            .pushes
                            .into_iter()
                            .map(|(i, items)| {
                                Some(PushTarget {
                                    endpoint: src.members[i].geometry.push.clone()?,
                                    session: local_member.geometry.session.clone(),
                                    holder_incarnation: k.holder.incarnation,
                                    offer_id: o.id,
                                    items,
                                })
                            })
                            .collect::<Option<Vec<_>>>();
                        if let Some(pushes) = pushes {
                            // All members pair the same logical slots, including slots
                            // whose physical destination belongs to a different rank.
                            self.read_path = ReadPath::from_tiers(
                                k,
                                endpoints.len(),
                                pairs.iter().map(|(p, _)| p.tier),
                            );
                            plan = Some(MemberPlan { reads, pushes });
                        } else {
                            tracing::warn!("peer reservation push target missing push endpoint");
                        }
                    }
                    Err(e) => tracing::warn!(%e, "peer reservation cannot be planned"),
                }
            }
        }
        let value = match &plan {
            None => 0,
            Some(p) if p.is_empty() => 1,
            Some(_) => 2,
        };
        self.submit_until = std::time::Instant::now() + super::v1::SUBMIT_WINDOW;
        self.phase = Phase::Planning;
        self.fetch = Some(f);
        self.plan = plan;
        self.deadline = now.saturating_add(2_000_000);
        Some(Vote {
            key: k.clone(),
            offer: o.id,
            stage: READY,
            value,
        })
    }
    pub fn fence(&mut self) {
        self.fenced = true;
    }
    pub fn invalidate(&mut self) {
        self.invalid = true;
    }
    fn seal(&mut self) {
        if self.phase != Phase::Reading {
            self.phase = Phase::Reading;
            self.votes = [None; 8];
        }
        self.retire_wait = true;
        if let Some(job) = &self.job {
            job.seal();
        }
    }
    fn freeze(&mut self, round: &mut PeerRound) {
        if self.published_hit.is_some() {
            return;
        }
        if let Some(f) = &self.fetch {
            round.events.push(event(PeerEventKind::FetchFrozen, f, 0));
            self.published_hit = Some(false);
            self.frozen = true;
        }
    }
    fn finish(&mut self, ok: bool, round: &mut PeerRound) {
        if self.published_hit.is_some() {
            return;
        }
        self.published_hit = Some(ok);
        if let Some(f) = &self.fetch {
            if ok {
                round
                    .landed
                    .push((f.request_id.clone(), f.begin_tokens, f.end_tokens));
            }
            round.events.push(event(
                PeerEventKind::FetchDone,
                f,
                if ok { f.end_tokens } else { 0 },
            ));
        }
    }
    fn close(&mut self, d: Disposition, now: i64) -> Message {
        self.retired_us = Some(now);
        self.phase = Phase::Closing(d);
        self.retry = now.saturating_add(50_000);
        self.deadline = now.saturating_add(2_000_000);
        Message::Release(
            self.key.clone().unwrap(),
            self.offer.as_ref().unwrap().id,
            d,
        )
    }
    pub fn vote(
        &mut self,
        member: usize,
        v: &Vote,
        quarantine_us: i64,
        now: i64,
        round: &mut PeerRound,
    ) -> Option<Message> {
        if self.key.as_ref() != Some(&v.key)
            || self.offer.as_ref()?.id != v.offer
            || !matches!(
                (self.phase, v.stage),
                (Phase::Planning, READY) | (Phase::Reading, DONE | FREEZE)
            )
        {
            return None;
        }
        if v.stage == FREEZE {
            // Early business fallback while Disk push cancels retry in the
            // background: every member seals and freezes in the same round,
            // while still awaiting final DONE votes for TargetRetired/RELEASE.
            self.seal();
            self.freeze(round);
            return None;
        }
        let entry = self.votes.get_mut(member)?;
        if entry.is_some() {
            return None;
        }
        *entry = Some(v.value);
        if v.stage == DONE && v.value == 2 {
            self.seal();
            self.freeze(round);
        }
        if self.votes.iter().any(Option::is_none) {
            return None;
        }
        if self.phase == Phase::Planning {
            if self.votes.contains(&Some(0)) || !self.votes.contains(&Some(2)) {
                self.seal();
                return None;
            }
            self.phase = Phase::Pin;
            self.pin_started_us = Some(now);
            self.votes = [None; 8];
            self.deadline = now.saturating_add(2_000_000);
            return Some(Message::Pin(v.key.clone(), self.offer.clone()?));
        }
        if self.votes.contains(&Some(2)) {
            // All members reported retirement and at least one is Unknown.
            // Keep target pages frozen (FetchFrozen) through the agreed
            // quarantine interval before emitting TargetRetired and
            // RELEASE(Abort), so write faults or vanished holders cannot
            // permanently leak device KV blocks or holder pins.
            let wait_us = quarantine_us.max(0);
            tracing::warn!(
                holder = %v.key.holder.name,
                lease = v.key.lease,
                offer = v.offer,
                quarantine_us = wait_us,
                "peer transfer retired with unknown write state; quarantining targets before TargetRetired and Release(Abort)"
            );
            self.pages.clear();
            self.source = None;
            self.plan = None;
            self.phase = Phase::Quarantine;
            self.deadline = now.saturating_add(wait_us);
            return None;
        }
        if self.frozen {
            if let Some(f) = &self.fetch {
                round.events.push(event(PeerEventKind::TargetRetired, f, 0));
            }
            self.frozen = false;
        }
        Some(self.close(
            if !self.retire_wait && self.votes.iter().all(|v| *v == Some(1)) {
                Disposition::Complete
            } else {
                Disposition::Abort
            },
            now,
        ))
    }
    pub fn grant(&mut self, k: &Key, o: &Offer, ok: bool, now: i64) -> Option<Message> {
        if self.phase != Phase::Pin
            || self.key.as_ref() != Some(k)
            || self.offer.as_ref() != Some(o)
        {
            return None;
        }
        if !ok {
            self.seal();
            return None;
        }
        self.granted = true;
        self.phase = Phase::Reading;
        self.deadline = now.saturating_add(30_000_000);
        None
    }
    pub fn ack(&mut self, k: &Key, id: u64, d: Disposition, c: Closure, round: &mut PeerRound) {
        if self.key.as_ref() != Some(k)
            || self.offer.as_ref().map(|o| o.id) != Some(id)
            || self.phase != Phase::Closing(d)
        {
            return;
        }
        match c {
            Closure::Unknown => return,
            Closure::Stale | Closure::Conflict => {
                // All local submissions have retired before Closing. The peer
                // cannot confirm this close. Finish locally without a hit.
                tracing::warn!(lease = k.lease, offer = id, ?c, "peer rejected release");
            }
            Closure::Closed | Closure::Gone => {}
        }
        self.finish(
            !self.invalid && c == Closure::Closed && d == Disposition::Complete,
            round,
        );
        self.phase = Phase::Finished;
    }
    pub fn tick(
        &mut self,
        reads: Option<&Reads>,
        meter: Option<(&Arc<ts_metrics::PeerMetrics>, usize)>,
        push_timeout: Duration,
        now: i64,
        round: &mut PeerRound,
    ) -> (Option<Vote>, Option<Message>) {
        if self.invalid
            && matches!(
                self.phase,
                Phase::Planning | Phase::Pin | Phase::Renewing | Phase::Reading
            )
        {
            self.seal();
        }
        if self.phase == Phase::Reading {
            if self.granted && (now >= self.deadline || self.invalid) {
                self.seal();
                self.freeze(round);
            }
            let mut vote = None;
            if !self.voted {
                if self.fenced && self.job.is_none() && !self.retire_wait {
                    if let Some(p) = self.plan.take() {
                        let metrics = meter.map(|(m, rank)| (m, self.lookup.holder.as_str(), rank));
                        self.job = reads.and_then(|w| {
                            w.submit_member_observed(
                                p.reads,
                                p.pushes,
                                push_timeout,
                                self.submit_until,
                                metrics,
                            )
                        });
                    }
                }
                let result = if let Some(j) = &self.job {
                    match j.try_retirement() {
                        Ok(ReadRetirement::Complete) => Some(1),
                        Ok(ReadRetirement::Aborted) => Some(0),
                        Ok(ReadRetirement::Unknown(e)) => {
                            tracing::error!(%e, "peer DMA retirement unknown; keeping landing and source pinned");
                            Some(2)
                        }
                        Err(mpsc::TryRecvError::Disconnected) => Some(2),
                        Err(mpsc::TryRecvError::Empty) => {
                            if !self.freeze_voted && j.is_cancelling() {
                                self.freeze_voted = true;
                                vote = Some(Vote {
                                    key: self.key.clone().unwrap(),
                                    offer: self.offer.as_ref().unwrap().id,
                                    stage: FREEZE,
                                    value: 1,
                                });
                            }
                            None
                        }
                    }
                } else if self.fenced && (self.retire_wait || self.plan.is_none()) {
                    Some(0)
                } else {
                    None
                };
                if let Some(value) = result {
                    self.voted = true;
                    self.job = None;
                    vote = Some(Vote {
                        key: self.key.clone().unwrap(),
                        offer: self.offer.as_ref().unwrap().id,
                        stage: DONE,
                        value,
                    });
                }
            }
            return (vote, None);
        }
        if self.phase == Phase::Quarantine {
            if now >= self.deadline {
                if self.frozen {
                    if let Some(f) = &self.fetch {
                        round.events.push(event(PeerEventKind::TargetRetired, f, 0));
                    }
                    self.frozen = false;
                }
                return (None, Some(self.close(Disposition::Abort, now)));
            }
            return (None, None);
        }
        if let Phase::Closing(d) = self.phase {
            if now >= self.deadline {
                self.finish(false, round);
            }
            if now >= self.retry {
                self.retry = now.saturating_add(100_000);
                return (
                    None,
                    Some(Message::Release(
                        self.key.clone().unwrap(),
                        self.offer.as_ref().unwrap().id,
                        d,
                    )),
                );
            }
        } else if now >= self.deadline {
            match self.phase {
                Phase::Pin | Phase::Planning | Phase::Renewing => self.seal(),
                Phase::Discovery | Phase::Query => {
                    round
                        .events
                        .push(super::autonomous::fallback(self.lookup.lease.clone()));
                    self.phase = Phase::Finished;
                }
                Phase::Offered => {
                    self.finish(false, round);
                    self.phase = Phase::Finished;
                }
                _ => (),
            }
        }
        (None, None)
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use ts_scheduler::PeerLeaseView;

    fn granted_candidate() -> (Attempt, Key, Offer) {
        let gen = |name: &str| Generation {
            name: name.into(),
            incarnation: 1,
            epoch: 1,
        };
        let key = Key {
            reader: gen("reader"),
            holder: gen("holder"),
            lease: 1,
        };
        let lease = PeerLeaseView {
            reader: key.reader.name.clone(),
            id: key.lease,
        };
        let lookup = PeerLookupView {
            request_id: "request".into(),
            deadline_us: 50000,
            lease: lease.clone(),
            holder: key.holder.name.clone(),
            prefix_hashes: vec![],
            tokens: vec![],
            max_tokens: 128,
            local_tokens: 0,
        };
        let offer = Offer {
            id: 7,
            end: 128,
            manifest: [0; 32],
        };
        let mut a = Attempt::new(lookup);
        a.key = Some(key.clone());
        a.offer = Some(offer.clone());
        a.fetch = Some(PeerFetchView {
            request_id: "request".into(),
            lease,
            holder: key.holder.name.clone(),
            begin_tokens: 0,
            end_tokens: 128,
            destinations: vec![],
            checkpoint: vec![],
            prefix_hashes: vec![],
            tokens: vec![],
        });
        a.phase = Phase::Pin;
        a.grant(&key, &offer, true, 1);
        assert!(a.phase == Phase::Reading);
        (a, key, offer)
    }

    // Enumerate the joint retirement observations on all eight members. Exercise
    // both directions of delivery and replay each report. The oracle uses only
    // the observations; all transitions and events come from the product FSM.
    #[test]
    fn eight_member_retirement_and_close_match_the_observed_outcomes() {
        for mut encoding in 0..3_usize.pow(8) {
            let mut values = [0; 8];
            for value in &mut values {
                *value = (encoding % 3) as u8;
                encoding /= 3;
            }
            for reverse in [false, true] {
                let (mut a, key, offer) = granted_candidate();
                let mut round = PeerRound::default();
                let mut release = None;
                let quarantine_us = 500_000;
                for i in 0..8 {
                    let rank = if reverse { 7 - i } else { i };
                    let vote = Vote {
                        key: key.clone(),
                        offer: offer.id,
                        stage: DONE,
                        value: values[rank],
                    };
                    let action = a.vote(rank, &vote, quarantine_us, 2, &mut round);
                    if i < 7 {
                        assert!(action.is_none());
                    }
                    if action.is_some() {
                        release = action;
                    }
                    assert!(a.vote(rank, &vote, quarantine_us, 3, &mut round).is_none());
                }
                assert!(round.landed.is_empty());
                if values.contains(&2) {
                    assert!(release.is_none());
                    assert!(!a.finished());
                    assert!(a.frozen);
                    assert!(a.take_pending_release(i64::MAX).is_none());
                    assert!(round
                        .events
                        .iter()
                        .any(|e| e.kind == PeerEventKind::FetchFrozen));
                    // Before quarantine deadline, stay frozen and emit nothing.
                    round.events.clear();
                    let (v_before, m_before) = a.tick(
                        None,
                        None,
                        Duration::from_secs(1),
                        2 + quarantine_us - 1,
                        &mut round,
                    );
                    assert!(v_before.is_none() && m_before.is_none());
                    assert!(round.events.is_empty());
                    assert!(a.frozen && !a.finished());
                    // Once quarantine elapses, emit TargetRetired and Release(Abort).
                    let (v_after, m_after) = a.tick(
                        None,
                        None,
                        Duration::from_secs(1),
                        2 + quarantine_us,
                        &mut round,
                    );
                    assert!(v_after.is_none());
                    assert!(!a.frozen);
                    assert_eq!(round.events.len(), 1);
                    assert_eq!(round.events[0].kind, PeerEventKind::TargetRetired);
                    assert_eq!(
                        m_after,
                        Some(Message::Release(key.clone(), offer.id, Disposition::Abort))
                    );
                    // Ack closes cleanly without emitting a duplicate FetchDone.
                    round.events.clear();
                    a.ack(
                        &key,
                        offer.id,
                        Disposition::Abort,
                        Closure::Closed,
                        &mut round,
                    );
                    assert!(a.finished());
                    assert!(round.events.is_empty());
                    continue;
                }
                let disposition = if values.contains(&0) {
                    Disposition::Abort
                } else {
                    Disposition::Complete
                };
                assert_eq!(
                    release,
                    Some(Message::Release(key.clone(), offer.id, disposition))
                );
                a.ack(&key, offer.id + 1, disposition, Closure::Closed, &mut round);
                a.ack(&key, offer.id, disposition, Closure::Unknown, &mut round);
                assert!(round.events.is_empty());
                a.ack(&key, offer.id, disposition, Closure::Closed, &mut round);
                a.ack(&key, offer.id, disposition, Closure::Closed, &mut round);
                assert!(a.finished());
                assert_eq!(round.events.len(), 1);
                assert_eq!(
                    round.events[0].num_tokens,
                    if disposition == Disposition::Complete {
                        128
                    } else {
                        0
                    }
                );
            }
        }
    }

    #[test]
    fn rejected_release_ends_only_local_cleanup() {
        for closure in [Closure::Stale, Closure::Conflict] {
            let (mut a, key, offer) = granted_candidate();
            let _ = a.close(Disposition::Complete, 10);
            let mut round = PeerRound::default();
            a.ack(&key, offer.id, Disposition::Complete, closure, &mut round);
            assert!(a.finished());
            assert!(round.landed.is_empty());
            assert_eq!(round.events.len(), 1);
            assert_eq!(round.events[0].num_tokens, 0);
        }
    }

    #[test]
    fn only_a_retired_business_fallback_can_detach_an_exact_release() {
        let (mut a, _, _) = granted_candidate();
        let mut round = PeerRound::default();
        a.freeze(&mut round);
        assert!(a.take_pending_release(i64::MAX).is_none());

        let (mut a, _, _) = granted_candidate();
        let release = a.close(Disposition::Abort, 10);
        assert!(a.take_pending_release(i64::MAX).is_none());
        a.tick(None, None, Duration::from_secs(1), 2_000_010, &mut round);
        assert!(a.take_pending_release(2_000_009).is_none());
        assert_eq!(a.take_pending_release(2_000_010), Some(release));
        assert!(a.finished());
        assert!(a.take_pending_release(i64::MAX).is_none());
    }

    #[test]
    fn lost_close_reply_retries_the_same_disposition_after_business_timeout() {
        for closure in [Closure::Gone, Closure::Closed] {
            let (mut a, key, offer) = granted_candidate();
            let release = a.close(Disposition::Complete, 1);
            let mut round = PeerRound::default();
            let (_, retry) = a.tick(None, None, Duration::from_secs(1), 3_000_001, &mut round);
            assert_eq!(retry, Some(release));
            assert_eq!(round.events.len(), 1);
            assert_eq!(round.events[0].num_tokens, 0);
            a.ack(&key, offer.id, Disposition::Complete, closure, &mut round);
            assert!(a.finished());
            assert!(round.landed.is_empty());
        }
    }

    #[test]
    fn early_freeze_vote_unfreezes_and_aborts_once_background_cancel_retires() {
        let (mut a, key, offer) = granted_candidate();
        let mut round = PeerRound::default();
        let freeze_vote = Vote {
            key: key.clone(),
            offer: offer.id,
            stage: FREEZE,
            value: 1,
        };
        assert!(a.vote(0, &freeze_vote, 500_000, 2, &mut round).is_none());
        assert_eq!(round.events.len(), 1);
        assert_eq!(round.events[0].kind, PeerEventKind::FetchFrozen);
        assert!(a.frozen);

        round.events.clear();
        let mut release = None;
        for rank in 0..8 {
            let done_vote = Vote {
                key: key.clone(),
                offer: offer.id,
                stage: DONE,
                value: 0,
            };
            if let Some(m) = a.vote(rank, &done_vote, 500_000, 3, &mut round) {
                release = Some(m);
            }
        }
        assert!(!a.frozen);
        assert_eq!(round.events.len(), 1);
        assert_eq!(round.events[0].kind, PeerEventKind::TargetRetired);
        assert_eq!(
            release,
            Some(Message::Release(key, offer.id, Disposition::Abort))
        );
    }

    #[test]
    fn quarantine_survives_invalidate_and_detaches_release_if_holder_is_dead() {
        let (mut a, key, offer) = granted_candidate();
        let mut round = PeerRound::default();
        let quarantine_us = 5_000_000;
        for rank in 0..8 {
            let vote = Vote {
                key: key.clone(),
                offer: offer.id,
                stage: DONE,
                value: if rank == 0 { 2 } else { 0 },
            };
            assert!(a
                .vote(rank, &vote, quarantine_us, 100, &mut round)
                .is_none());
        }
        assert!(a.frozen && !a.finished());
        // Weight invalidation must never shorten destination quarantine.
        a.invalidate();
        round.events.clear();
        let (_, m) = a.tick(None, None, Duration::from_secs(1), 4_999_999, &mut round);
        assert!(m.is_none() && round.events.is_empty() && a.frozen);
        assert!(a.take_pending_release(i64::MAX).is_none());

        // At quarantine expiry, emit TargetRetired and enter Closing(Abort).
        let (_, release) = a.tick(None, None, Duration::from_secs(1), 5_000_100, &mut round);
        assert!(!a.frozen);
        assert_eq!(round.events.len(), 1);
        assert_eq!(round.events[0].kind, PeerEventKind::TargetRetired);
        let expected = Message::Release(key, offer.id, Disposition::Abort);
        assert_eq!(release, Some(expected.clone()));

        // If holder is dead and never ACKs, Closing detaches Release after 2s without
        // emitting a duplicate FetchDone, reclaiming the Attempt slot cleanly.
        round.events.clear();
        a.tick(None, None, Duration::from_secs(1), 7_000_100, &mut round);
        assert!(round.events.is_empty());
        assert_eq!(a.take_pending_release(7_000_100), Some(expected));
        assert!(a.finished());
    }

    #[test]
    fn planning_or_pin_timeout_and_invalidate_do_not_freeze_before_grant() {
        for pre_grant_phase in [Phase::Planning, Phase::Pin] {
            let (mut a, _, _) = granted_candidate();
            a.phase = pre_grant_phase;
            a.granted = false;
            a.deadline = 100;
            let mut round = PeerRound::default();
            // Simulate timeout before Grant ever arrived -> must abort cleanly, not freeze.
            a.tick(None, None, Duration::from_secs(1), 200, &mut round);
            assert!(!a.frozen, "pre-grant timeout must not mark attempt frozen");
            assert!(
                !round
                    .events
                    .iter()
                    .any(|e| e.kind == PeerEventKind::FetchFrozen),
                "pre-grant timeout must not emit FetchFrozen"
            );

            // Simulate invalidate() before Grant -> must also not freeze.
            let (mut b, _, _) = granted_candidate();
            b.phase = pre_grant_phase;
            b.granted = false;
            b.deadline = 10_000;
            b.invalidate();
            let mut round2 = PeerRound::default();
            b.tick(None, None, Duration::from_secs(1), 200, &mut round2);
            assert!(!b.frozen, "pre-grant invalidate must not freeze");
            assert!(
                !round2
                    .events
                    .iter()
                    .any(|e| e.kind == PeerEventKind::FetchFrozen),
                "pre-grant invalidate must not emit FetchFrozen"
            );
        }
    }
}
