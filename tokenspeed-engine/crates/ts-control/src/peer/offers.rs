//! Per-member exact pin and close votes for versioned source offers.
use super::{
    registered::Replica,
    v1::{put_pages, Closure, Digest, Disposition, Generation, Key, Message, Offer},
};
use crate::codec::Writer;
use sha2::{Digest as _, Sha256};
use std::collections::HashMap;
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

struct Record {
    offer: Offer,
    query: Digest,
    checkpoint: PeerEventView,
    prefix: Digest,
    phase: Phase,
    held: bool,
    votes: Vec<Option<u8>>,
    until: i64,
}
pub(super) struct Offers {
    next: u64,
    records: HashMap<Key, Record>,
}

impl Offers {
    pub fn new() -> Self {
        Self {
            next: 1,
            records: HashMap::new(),
        }
    }
    pub fn active(&self) -> bool {
        self.records
            .values()
            .any(|r| !matches!(r.phase, Phase::Soft | Phase::Closed(_)))
    }
    pub fn has(&self, key: &Key) -> bool {
        self.records.contains_key(key)
    }
    pub fn expire(&mut self, now: i64) {
        self.records
            .retain(|_, r| !matches!(r.phase, Phase::Soft | Phase::Closed(_)) || r.until > now);
    }

    pub fn lookup(
        &mut self,
        key: &Key,
        local: &Generation,
        layout: &Replica,
        msg: &Message,
        scheduler: &mut Scheduler,
        now: i64,
    ) -> Message {
        let Message::Lookup(_, s, _, l, p) = msg else {
            unreachable!()
        };
        let miss = || Message::Offered(key.clone(), None);
        if key.holder != *local || *s != layout.semantic || *l != layout.digest {
            return miss();
        }
        let hash: Digest = Sha256::digest(msg.encode()).into();
        if let Some(r) = self.records.get(key) {
            if r.query != hash || r.phase != Phase::Soft || r.until <= now {
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
        let answer = scheduler.answer_peer_lookup(&event);
        let end = answer.num_tokens;
        if end <= 0
            || end > p.max_tokens
            || end as usize > p.tokens.len()
            || answer.pages.is_empty()
            || answer
                .pages
                .iter()
                .any(|p| !matches!(p.tier, PeerTier::Device | PeerTier::Host))
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
        if k != other || self.records.contains_key(k) {
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
        let Some(r) = self.records.get_mut(k) else {
            return no();
        };
        if *o != r.offer {
            return no();
        }
        match r.phase {
            Phase::Soft if r.until > now => {
                r.phase = Phase::Preparing;
                r.held = s.pin_peer_offer(&r.checkpoint, now).unwrap_or(false);
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
        s: &mut Scheduler,
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
            Phase::Closing(_) => return (None, None),
            Phase::Closed(_) => return ack(Closure::Closed),
            Phase::Granted => (),
            _ if d == Disposition::Complete => return ack(Closure::Conflict),
            _ => (),
        }
        r.phase = Phase::Closing(d);
        r.votes.fill(None);
        let ok = !r.held
            || s.close_peer_offer(&r.checkpoint.lease, now)
                .unwrap_or(false);
        if ok {
            r.held = false;
        }
        (
            None,
            Some(Vote {
                key: k.clone(),
                offer: id,
                stage: CLOSED,
                value: ok as u8,
            }),
        )
    }
    pub fn vote(&mut self, rank: usize, v: &Vote, now: i64) -> Option<Message> {
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
                }
                Some(Message::Grant(v.key.clone(), r.offer.clone(), ok))
            }
            Phase::Closing(d) if ok => {
                r.phase = Phase::Closed(d);
                r.until = now.saturating_add(60_000_000);
                Some(Message::Ack(v.key.clone(), v.offer, d, Closure::Closed))
            }
            _ => None,
        }
    }
}
