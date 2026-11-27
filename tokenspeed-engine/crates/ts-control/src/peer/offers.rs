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

//! Per-member exact pin and close votes for versioned source offers.
use super::{
    push::PagePusher,
    registered::Replica,
    v1::{put_pages, Closure, Digest, Disposition, Generation, Key, Message, Offer},
};
use crate::codec::Writer;
use sha2::{Digest as _, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use ts_scheduler::{PeerEventView, PeerLeaseView, PeerPageView, PeerTier, Scheduler};

/// One vote from a member; the reporting rank comes from the collective.
#[derive(Clone)]
pub(super) struct Vote {
    pub key: Key,
    pub offer: u64,
    pub stage: u8,
    pub value: u8,
}
pub(super) const PINNED: u8 = 1;
pub(super) const CLOSED: u8 = 2;
pub(super) const READY: u8 = 3;
pub(super) const DONE: u8 = 4;
pub(super) const FREEZE: u8 = 5;

pub(super) fn manifest(pages: &[PeerPageView]) -> Digest {
    let mut sorted = pages.to_vec();
    sorted.sort_by_key(|p| (p.group_id, p.slot));
    let mut w = Writer::new();
    w.u8(1);
    put_pages(&mut w, &sorted);
    Sha256::digest(w.finish()).into()
}
pub(super) fn prefix(tokens: &[i32]) -> Digest {
    let mut w = Writer::new();
    w.u8(1);
    w.i32s(tokens);
    Sha256::digest(w.finish()).into()
}

#[derive(Copy, Clone, PartialEq)]
enum Phase {
    Soft,
    Preparing,
    Granted,
    Closing(Disposition),
    Closed(Disposition),
}

impl Phase {
    fn cleanup(
        self,
        now: i64,
        until: i64,
        retirement_bound_ms: u32,
        votes: &[Option<u8>],
    ) -> Option<Disposition> {
        match self {
            Self::Preparing if now >= until => Some(Disposition::Abort),
            Self::Granted
                if retirement_bound_ms > 0
                    && now >= until.saturating_add(i64::from(retirement_bound_ms) * 1_000) =>
            {
                Some(Disposition::Abort)
            }
            Self::Closing(d)
                if now >= until
                    && votes.iter().all(Option::is_some)
                    && votes.contains(&Some(0)) =>
            {
                Some(d)
            }
            _ => None,
        }
    }
}

struct Record {
    offer: Offer,
    query: Digest,
    checkpoint: PeerEventView,
    prefix: Digest,
    phase: Phase,
    held: bool,
    unpin_retry_us: i64,
    votes: Vec<Option<u8>>,
    until: i64,
}
pub(super) struct Offers {
    device_pin_gate: Arc<super::servicer::DevicePinGate>,
    next: u64,
    records: HashMap<Key, Record>,
}

fn try_release_pin(
    r: &mut Record,
    now: i64,
    mut close_pin: impl FnMut(&PeerLeaseView, i64) -> Result<bool, String>,
) -> bool {
    if !r.held {
        return true;
    }
    match close_pin(&r.checkpoint.lease, now) {
        Ok(true) => {
            r.held = false;
            true
        }
        Ok(false) => {
            tracing::error!(
                lease = r.checkpoint.lease.id,
                reader = %r.checkpoint.lease.reader,
                offer = r.offer.id,
                "close_peer_offer returned false after unanimous drain; retaining pin for retry"
            );
            false
        }
        Err(error) => {
            tracing::error!(
                %error,
                lease = r.checkpoint.lease.id,
                reader = %r.checkpoint.lease.reader,
                offer = r.offer.id,
                "close_peer_offer failed after unanimous drain; retaining pin for retry"
            );
            false
        }
    }
}

impl Offers {
    pub fn new(device_pin_gate: Arc<super::servicer::DevicePinGate>) -> Self {
        Self {
            device_pin_gate,
            next: 1,
            records: HashMap::new(),
        }
    }
    pub fn active_count(&self) -> usize {
        self.records
            .values()
            .filter(|r| !matches!(r.phase, Phase::Soft | Phase::Closed(_)) || r.held)
            .count()
    }
    pub fn offer_id(&self, key: &Key) -> Option<u64> {
        self.records.get(key).map(|r| r.offer.id)
    }
    pub fn expire(
        &mut self,
        now: i64,
        retirement_bound_ms: u32,
        s: &mut Scheduler,
        pusher: &dyn PagePusher,
    ) -> Vec<Vote> {
        self.expire_with(now, retirement_bound_ms, pusher, |lease, t| {
            s.close_peer_offer(lease, t).map_err(|e| e.to_string())
        })
    }

    fn expire_with(
        &mut self,
        now: i64,
        retirement_bound_ms: u32,
        pusher: &dyn PagePusher,
        mut close_pin: impl FnMut(&PeerLeaseView, i64) -> Result<bool, String>,
    ) -> Vec<Vote> {
        for r in self.records.values_mut() {
            if matches!(r.phase, Phase::Closed(_))
                && r.held
                && now >= r.unpin_retry_us
                && !try_release_pin(r, now, &mut close_pin)
            {
                r.unpin_retry_us = now.saturating_add(1_000_000);
            }
        }
        let due: Vec<_> = self
            .records
            .iter()
            .filter_map(|(k, r)| {
                let disposition = r
                    .phase
                    .cleanup(now, r.until, retirement_bound_ms, &r.votes)?;
                Some((k.clone(), r.offer.id, disposition))
            })
            .collect();
        let mut votes = Vec::new();
        for (key, id, d) in due {
            let (_, vote) = self.close(&key, id, d, pusher, now);
            votes.extend(vote);
        }
        self.records.retain(|_, r| {
            !matches!(r.phase, Phase::Soft | Phase::Closed(_)) || r.held || r.until > now
        });
        votes
    }

    #[allow(clippy::too_many_arguments)]
    pub fn lookup(
        &mut self,
        key: &Key,
        local: &Generation,
        layout: &Replica,
        replica_can_serve_disk: bool,
        msg: &Message,
        scheduler: &mut Scheduler,
        pusher: &dyn PagePusher,
        now: i64,
    ) -> Message {
        let Message::Lookup(_, s, _, l, p) = msg else {
            unreachable!()
        };
        let miss = || Message::Offered(key.clone(), None);
        if key.holder != *local || *s != layout.semantic || *l != layout.digest {
            return miss();
        }
        let can_push_disk =
            layout.can_push_disk() && replica_can_serve_disk && pusher.can_serve_disk();
        let hash: Digest = Sha256::digest(msg.encode()).into();
        // Expiration may run after this round's messages. Replace only unpinned
        // metadata; a new lookup always gets a fresh holder-issued capability.
        if self
            .records
            .get(key)
            .is_some_and(|r| r.phase == Phase::Soft && r.until <= now)
        {
            self.records.remove(key);
        }
        if let Some(r) = self.records.get(key) {
            if r.query != hash
                || r.phase != Phase::Soft
                || r.until <= now
                || (!can_push_disk && r.checkpoint.pages.iter().any(|p| p.tier == PeerTier::Disk))
            {
                return miss();
            }
            return Message::Offered(
                key.clone(),
                Some((r.offer.clone(), r.prefix, r.checkpoint.pages.clone())),
            );
        }
        if self.records.len() >= 4096 {
            return miss();
        }
        let lease = PeerLeaseView {
            reader: key.reader.name.clone(),
            id: key.lease,
        };
        let mut event = p.lookup_event(&lease);
        // Soft offers reserve no source pages. PIN rechecks capacity and the exact manifest.
        // If this holder cannot serve Disk pushes (no staging ring / disk bounds /
        // quarantined push endpoint on any replica member), truncate cleanly to
        // its Device/Host prefix instead of returning a full miss.
        let answer = if can_push_disk {
            scheduler.answer_peer_lookup_bounded(&event, 0)
        } else {
            scheduler.answer_peer_lookup_without_disk(&event, 0)
        };
        let end = answer.num_tokens;
        if end <= 0
            || end > p.max_tokens
            || end as usize > p.tokens.len()
            || answer.pages.is_empty()
            || answer.pages.iter().any(|page| match page.tier {
                PeerTier::Device | PeerTier::Host => false,
                PeerTier::Disk => {
                    !can_push_disk
                        || layout
                            .members
                            .first()
                            .and_then(|m| m.disk_bounds.get(page.group_id as usize))
                            .is_none_or(|&bound| !(1..bound).contains(&i64::from(page.block_id)))
                }
                _ => true,
            })
        {
            return miss();
        }
        let hashes = end as usize / layout.contract.plan.prefix_granularity as usize;
        if hashes > event.prefix_hashes.len() {
            return miss();
        }
        event.tokens.truncate(end as usize);
        event.prefix_hashes.truncate(hashes);
        event.num_tokens = end;
        event.max_tokens = end;
        event.pages = answer.pages;
        // Only the leader issues, with no skipped sequence numbers. The local
        // collective distributes this same record before any PIN can acquire it.
        let id = self.next;
        let Some(next) = id.checked_add(1) else {
            return miss();
        };
        self.next = next;
        let pages = event.pages.clone();
        let offer = Offer {
            id,
            end,
            manifest: manifest(&pages),
        };
        let identity = prefix(&event.tokens);
        self.records.insert(
            key.clone(),
            Record {
                offer: offer.clone(),
                query: hash,
                checkpoint: event,
                prefix: identity,
                phase: Phase::Soft,
                held: false,
                unpin_retry_us: 0,
                votes: vec![None; layout.members.len()],
                until: now.saturating_add(2_000_000),
            },
        );
        Message::Offered(key.clone(), Some((offer, identity, pages)))
    }

    // Only the leader queries. Install the same proposed end and full manifest
    // on all members before applying inbound PIN or RELEASE from that round.
    pub fn install(
        &mut self,
        lookup: &Message,
        answer: &Message,
        grain: i64,
        width: usize,
        now: i64,
    ) {
        let (Message::Lookup(k, _, _, _, p), Message::Offered(other, Some((o, digest, pages)))) =
            (lookup, answer)
        else {
            return;
        };
        if k != other
            || self
                .records
                .get(k)
                .is_some_and(|r| r.phase != Phase::Soft || r.offer.id >= o.id)
        {
            return;
        }
        let lease = PeerLeaseView {
            reader: k.reader.name.clone(),
            id: k.lease,
        };
        let mut checkpoint = p.lookup_event(&lease);
        checkpoint.tokens.truncate(o.end as usize);
        checkpoint
            .prefix_hashes
            .truncate(o.end as usize / grain as usize);
        checkpoint.max_tokens = o.end;
        checkpoint.num_tokens = o.end;
        checkpoint.pages = pages.clone();
        self.records.insert(
            k.clone(),
            Record {
                offer: o.clone(),
                query: Sha256::digest(lookup.encode()).into(),
                checkpoint,
                prefix: *digest,
                phase: Phase::Soft,
                held: false,
                unpin_retry_us: 0,
                votes: vec![None; width],
                until: now.saturating_add(2_000_000),
            },
        );
    }

    pub fn pin(
        &mut self,
        k: &Key,
        o: &Offer,
        s: &mut Scheduler,
        pusher: &dyn PagePusher,
        now: i64,
    ) -> (Option<Message>, Option<Vote>) {
        let no = || {
            (
                None,
                Some(Vote {
                    key: k.clone(),
                    offer: o.id,
                    stage: PINNED,
                    value: 0,
                }),
            )
        };
        let expired = || (Some(Message::Expired(k.clone(), o.clone())), None);
        let Some(r) = self.records.get_mut(k) else {
            // A forgotten offer cannot grant a PIN. Let the reader look it up
            // once more and validate the exact pages under a fresh offer.
            return expired();
        };
        if *o != r.offer {
            return no();
        }
        match r.phase {
            Phase::Soft if r.until <= now => expired(),
            Phase::Soft => {
                r.phase = Phase::Preparing;
                r.until = now.saturating_add(2_000_000);
                let disk_ok = !r.checkpoint.pages.iter().any(|p| p.tier == PeerTier::Disk)
                    || pusher.can_serve_disk();
                r.held = disk_ok
                    && self.device_pin_gate.allows(&r.checkpoint.pages)
                    && s.pin_peer_offer(&r.checkpoint, now).unwrap_or(false);
                if r.held {
                    // Authorize this offer on the holder's push endpoint before
                    // voting PINNED so every member is ready before Grant leaves
                    // rank 0. Refused or closed offers revoke it before unpinning.
                    pusher.activate_offer(k.holder.incarnation, r.offer.id);
                }
                (
                    None,
                    Some(Vote {
                        key: k.clone(),
                        offer: o.id,
                        stage: PINNED,
                        value: r.held as u8,
                    }),
                )
            }
            Phase::Granted => (Some(Message::Grant(k.clone(), o.clone(), true)), None),
            Phase::Preparing => (None, None),
            _ => no(),
        }
    }
    pub fn close(
        &mut self,
        k: &Key,
        id: u64,
        d: Disposition,
        pusher: &dyn PagePusher,
        now: i64,
    ) -> (Option<Message>, Option<Vote>) {
        let ack = |c| (Some(Message::Ack(k.clone(), id, d, c)), None);
        let Some(r) = self.records.get_mut(k) else {
            // Issued ids never recur within this generation. A record with a
            // pending vote or a ticket cannot be evicted. Only the leader
            // replies, and GONE can end cleanup but cannot authorize a hit.
            return ack(
                if id > 0 && id < self.next && self.records.values().all(|r| r.offer.id != id) {
                    Closure::Gone
                } else {
                    Closure::Unknown
                },
            );
        };
        if id != r.offer.id {
            return ack(Closure::Conflict);
        }
        match r.phase {
            Phase::Closing(prev) | Phase::Closed(prev) if prev != d => {
                return ack(Closure::Conflict)
            }
            Phase::Closing(_) if r.phase.cleanup(now, r.until, 0, &r.votes).is_none() => {
                return (None, None)
            }
            Phase::Closing(_) => (),
            Phase::Closed(_) => return ack(Closure::Closed),
            Phase::Granted => (),
            _ if d == Disposition::Complete => return ack(Closure::Conflict),
            _ => (),
        }
        r.phase = Phase::Closing(d);
        r.votes.fill(None);
        // Revoke push authorization and cancel/drain any in-flight Disk pushes
        // BEFORE releasing scheduler pins. Do NOT call `close_peer_offer` here:
        // on mirrored TP/KVP schedulers, all members must unpin in lockstep on
        // the exact same round once every member votes `CLOSED = 1` (drained).
        let drained = pusher.cancel_offer(k.holder.incarnation, id);
        r.until = now.saturating_add(1_000_000);
        (
            None,
            Some(Vote {
                key: k.clone(),
                offer: id,
                stage: CLOSED,
                value: drained as u8,
            }),
        )
    }
    pub fn vote(&mut self, rank: usize, v: &Vote, s: &mut Scheduler, now: i64) -> Option<Message> {
        self.vote_with(rank, v, now, |lease, t| {
            s.close_peer_offer(lease, t).map_err(|e| e.to_string())
        })
    }

    fn vote_with(
        &mut self,
        rank: usize,
        v: &Vote,
        now: i64,
        close_pin: impl FnMut(&PeerLeaseView, i64) -> Result<bool, String>,
    ) -> Option<Message> {
        let r = self.records.get_mut(&v.key)?;
        if r.offer.id != v.offer
            || !matches!(
                (r.phase, v.stage),
                (Phase::Preparing, PINNED) | (Phase::Closing(_), CLOSED)
            )
        {
            return None;
        }
        let slot = r.votes.get_mut(rank)?;
        if slot.is_some() {
            return None;
        }
        *slot = Some(v.value);
        if !r.votes.iter().all(Option::is_some) {
            return None;
        }
        let ok = r.votes.iter().all(|v| *v == Some(1));
        match r.phase {
            Phase::Preparing => {
                if ok {
                    r.phase = Phase::Granted;
                    // Every reader starts its submit window before its READY
                    // vote, hence before this grant even if delivery is delayed.
                    r.until = now.saturating_add(super::v1::SUBMIT_WINDOW.as_micros() as i64);
                }
                Some(Message::Grant(v.key.clone(), r.offer.clone(), ok))
            }
            Phase::Closing(d) if ok => {
                // Unanimous drain consensus: every member has zero in-flight
                // Disk pushes for this offer. Release scheduler pins in lockstep
                // on the exact same round across all mirrored members. If
                // `close_peer_offer` unexpectedly fails, retain `held = true`
                // so the record stays unpruned and `expire()` retries release.
                if r.held && !try_release_pin(r, now, close_pin) {
                    r.unpin_retry_us = now.saturating_add(1_000_000);
                }
                r.phase = Phase::Closed(d);
                r.until = now.saturating_add(60_000_000);
                Some(Message::Ack(v.key.clone(), v.offer, d, Closure::Closed))
            }
            Phase::Closing(_) => {
                // At least one member is still draining Disk pushes; keep pins
                // held uniformly on all members and pull the collective retry
                // deadline in using shared round time `now`.
                r.until = r.until.min(now.saturating_add(50_000));
                None
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use crate::peer::push::PushItem;
    use crate::peer::servicer::{empty_event, DevicePinGate};
    use std::time::Duration;

    struct DummyPusher;
    impl PagePusher for DummyPusher {
        fn push(&self, _: &str, _: &str, _: Duration, _: Vec<PushItem>) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn orphan_close_respects_the_backend_bound_and_preserves_failed_close_votes() {
        let cutoff = 30_000_000;
        let votes = [Some(1), None];
        assert_eq!(Phase::Granted.cleanup(i64::MAX, cutoff, 0, &votes), None);
        assert_eq!(
            Phase::Granted.cleanup(cutoff + 9_999, cutoff, 10, &votes),
            None
        );
        assert_eq!(
            Phase::Granted.cleanup(cutoff + 10_000, cutoff, 10, &votes),
            Some(Disposition::Abort)
        );
        assert_eq!(
            Phase::Preparing.cleanup(cutoff, cutoff, 0, &votes),
            Some(Disposition::Abort)
        );
        let phase = Phase::Closing(Disposition::Complete);
        assert_eq!(phase.cleanup(cutoff, cutoff, 0, &votes), None);
        assert_eq!(
            phase.cleanup(cutoff, cutoff, 0, &[Some(1), Some(0)]),
            Some(Disposition::Complete)
        );
    }

    #[test]
    fn closed_offer_with_failed_unpin_stays_active_and_retries_until_released() {
        let mut offers = Offers::new(Arc::new(DevicePinGate::default()));
        let gen = |name: &str| Generation {
            name: name.into(),
            incarnation: 1,
            epoch: 1,
        };
        let key = Key {
            reader: gen("reader"),
            holder: gen("holder"),
            lease: 42,
        };
        offers.records.insert(
            key.clone(),
            Record {
                offer: Offer {
                    id: 7,
                    end: 4,
                    manifest: [0; 32],
                },
                query: [0; 32],
                checkpoint: empty_event(),
                prefix: [0; 32],
                phase: Phase::Closing(Disposition::Complete),
                held: true,
                unpin_retry_us: 0,
                votes: vec![None],
                until: 1_000_000,
            },
        );
        let vote = Vote {
            key: key.clone(),
            offer: 7,
            stage: CLOSED,
            value: 1,
        };
        // Simulated failure on initial close_peer_offer: transition to Closed,
        // emit Ack, but retain `held = true` and schedule retry at now + 1s.
        let ack = offers.vote_with(0, &vote, 1_000_000, |_, _| Err("transient fault".into()));
        assert!(matches!(
            ack,
            Some(Message::Ack(_, 7, Disposition::Complete, Closure::Closed))
        ));
        assert_eq!(offers.active_count(), 1);
        assert!(offers.records[&key].held);

        let pusher = DummyPusher;
        let mut calls = 0usize;
        // Before retry deadline (`1_500_000 < 2_000_000`), do not retry yet.
        let votes = offers.expire_with(1_500_000, 0, &pusher, |_, _| {
            calls += 1;
            Ok(true)
        });
        assert!(votes.is_empty());
        assert_eq!(calls, 0);
        assert_eq!(offers.active_count(), 1);

        // Even past `until` (`61_000_000`), if retry returns `Ok(false)` the
        // record stays retained and active instead of silently leaking pins.
        let votes = offers.expire_with(65_000_000, 0, &pusher, |_, _| {
            calls += 1;
            Ok(false)
        });
        assert!(votes.is_empty());
        assert_eq!(calls, 1);
        assert!(offers.offer_id(&key).is_some());
        assert_eq!(offers.active_count(), 1);

        // Next retry after 1s succeeds -> clears `held` and prunes expired record.
        let votes = offers.expire_with(66_000_000, 0, &pusher, |_, _| {
            calls += 1;
            Ok(true)
        });
        assert!(votes.is_empty());
        assert_eq!(calls, 2);
        assert!(offers.offer_id(&key).is_none());
        assert_eq!(offers.active_count(), 0);
    }
}
