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

//! A holder's answers (`docs/rfcs/0004-peer-prefix-sharing.md`): each request
//! is answered at once with reads of the cache as it stands, between the
//! control thread's rounds, and what it changes is queued for the next head,
//! where every member of the replica applies it before any other event.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use ts_scheduler::{
    PeerAnswerKind, PeerAnswerView, PeerEventKind, PeerEventView, PeerLeaseView, PeerPageView,
    PeerTier, Scheduler,
};

use crate::peer::{
    geometry::RankGeometry,
    wire::{Reply, Request, Stamp},
};

impl Reply {
    pub(super) fn observe_holder(&self, metrics: &ts_metrics::PeerMetrics, read: bool) {
        if let Self::Answer { answer, .. } = self {
            if let Some(outcome) = holder_outcome(answer, read) {
                metrics.holder_event(outcome);
            }
        }
    }
}

fn holder_outcome(answer: &PeerAnswerView, read: bool) -> Option<&'static str> {
    Some(match answer.kind {
        PeerAnswerKind::Lookup if answer.num_tokens > 0 => "lookup_hit",
        PeerAnswerKind::Lookup => "lookup_miss",
        PeerAnswerKind::Pin if answer.held => "pin_ok",
        PeerAnswerKind::Pin => "pin_rejected",
        PeerAnswerKind::Release if !answer.held => "release_rejected",
        PeerAnswerKind::Release if read => "read_released",
        PeerAnswerKind::Release => "cleanup_released",
        _ => return None,
    })
}

/// The reads a holder answers with (`Scheduler`'s peer reads).
pub trait HolderCache {
    /// A lookup's answer.
    fn answer_lookup(&self, lookup: &PeerEventView) -> PeerAnswerView;
    /// Whether the lookup's prompt, up to `num_tokens`, still resolves to
    /// exactly `pages`, and every page is ready for an unordered remote read.
    fn pages_match(&self, lookup: &PeerEventView, num_tokens: i32, pages: &[PeerPageView]) -> bool;
    /// Whether the lease's pin holds its pages now.
    fn lease_held(&self, lease: &PeerLeaseView) -> bool;
    /// Plans built so far.
    fn plan_calls(&self) -> i64;
}

impl HolderCache for Scheduler {
    fn answer_lookup(&self, lookup: &PeerEventView) -> PeerAnswerView {
        self.answer_peer_lookup(lookup)
    }

    fn pages_match(&self, lookup: &PeerEventView, num_tokens: i32, pages: &[PeerPageView]) -> bool {
        self.peer_pages_match(lookup, num_tokens, pages)
    }

    fn lease_held(&self, lease: &PeerLeaseView) -> bool {
        self.peer_lease_held(lease)
    }

    fn plan_calls(&self) -> i64 {
        self.plan_calls()
    }
}

/// What a lease's lookup asked and was answered, and when its pin was.
struct Record {
    lookup: PeerEventView,
    answer: PeerAnswerView,
    /// The holder's plan count when the pin was answered.
    pinned_at: Option<i64>,
    expires: Instant,
}

/// Shared with the control thread, which closes this gate while holding the
/// scheduler lock, before its new Device publications become visible.
#[derive(Default)]
pub(super) struct DevicePinGate(AtomicBool);

impl DevicePinGate {
    pub(super) fn block(&self, blocked: bool) {
        self.0.store(blocked, Ordering::Release);
    }

    pub(super) fn allows(&self, pages: &[PeerPageView]) -> bool {
        !self.0.load(Ordering::Acquire) || pages.iter().all(|page| page.tier != PeerTier::Device)
    }
}

/// The holder's half: answers, the leases' records and the queued changes.
pub struct Holder {
    pub(super) metrics: Option<std::sync::Arc<ts_metrics::PeerMetrics>>,
    pub(super) device_pin_gate: Arc<DevicePinGate>,
    stamp: Stamp,
    ranks: Vec<RankGeometry>,
    records: HashMap<PeerLeaseView, Record>,
    /// How long a record outlives its last request: longer than any pin the
    /// scheduler holds, so a release always finds its pin's record.
    record_lifetime: Duration,
    changes: Vec<PeerEventView>,
}

impl Holder {
    /// A holder answering as `stamp`, with its replica's geometries.
    pub fn new(stamp: Stamp, ranks: Vec<RankGeometry>, record_lifetime: Duration) -> Self {
        Self {
            metrics: None,
            device_pin_gate: Arc::default(),
            stamp,
            ranks,
            records: HashMap::new(),
            record_lifetime,
            changes: Vec::new(),
        }
    }

    /// Gate new Device pins without affecting existing leases or other tiers.
    pub fn block_device_pins(&mut self, blocked: bool) {
        self.device_pin_gate.block(blocked);
    }

    pub(super) fn matches_stamp(&self, stamp: &Stamp) -> bool {
        self.stamp == *stamp
    }

    pub(super) fn layout(&self) -> Reply {
        Reply::Layout {
            stamp: self.stamp.clone(),
            ranks: self.ranks.clone(),
        }
    }

    /// Answer as the namespace `identity` from now on.
    pub fn set_identity(&mut self, identity: String) {
        self.stamp.identity = identity;
    }

    /// The answer to `request`, and the endpoint it goes to.
    pub fn handle(
        &mut self,
        cache: &impl HolderCache,
        request: Request,
        now: Instant,
    ) -> (String, Reply) {
        self.records.retain(|_, record| now < record.expires);
        match request {
            Request::Reading { .. } => {
                unreachable!("read liveness belongs to the autonomous driver")
            }
            Request::Layout { reply_to } => (reply_to, self.layout()),
            Request::Lookup { lease, prompt } => {
                let lookup = prompt.lookup_event(&lease);
                let answer = cache.answer_lookup(&lookup);
                self.changes.push(lookup.clone());
                self.records.insert(
                    lease.clone(),
                    Record {
                        lookup,
                        answer: answer.clone(),
                        pinned_at: None,
                        expires: now + self.record_lifetime,
                    },
                );
                self.reply(lease, answer)
            }
            Request::CheckpointPin { checkpoint, .. } => {
                let answer = self.answer(&checkpoint.lease, PeerAnswerKind::Pin);
                self.reply(checkpoint.lease, answer)
            }
            Request::Pin { lease } => {
                let mut answer = self.answer(&lease, PeerAnswerKind::Pin);
                if let Some(record) = self.records.get_mut(&lease) {
                    let looked_up = &record.answer;
                    answer.num_tokens = looked_up.num_tokens;
                    answer.pages = looked_up.pages.clone();
                    answer.held = looked_up.num_tokens > 0
                        && self.device_pin_gate.allows(&looked_up.pages)
                        && cache.pages_match(
                            &record.lookup,
                            looked_up.num_tokens,
                            &looked_up.pages,
                        );
                    if answer.held {
                        // The head after this applies the pin; until then no
                        // plan runs, and none may evict the pages.
                        record.pinned_at = Some(cache.plan_calls());
                        record.expires = now + self.record_lifetime;
                        self.changes.push(PeerEventView {
                            kind: PeerEventKind::Pin,
                            lease: lease.clone(),
                            num_tokens: looked_up.num_tokens,
                            pages: looked_up.pages.clone(),
                            ..empty_event()
                        });
                    }
                }
                self.reply(lease, answer)
            }
            Request::Release { lease, demote } => {
                let mut answer = self.answer(&lease, PeerAnswerKind::Release);
                // Untouched from the pin: no plan ran since its answer, or its
                // pin holds the pages now.
                answer.held = self
                    .records
                    .remove(&lease)
                    .and_then(|record| record.pinned_at)
                    .is_some_and(|pinned_at| {
                        cache.plan_calls() == pinned_at || cache.lease_held(&lease)
                    });
                self.changes.push(PeerEventView {
                    kind: PeerEventKind::Release,
                    lease: lease.clone(),
                    demote: demote && answer.held,
                    ..empty_event()
                });
                self.reply(lease, answer)
            }
        }
    }

    pub(super) fn finish_record(&mut self, lease: &PeerLeaseView) {
        self.records.remove(lease);
    }

    /// The changes answered since the last take, in order, for the next head.
    pub fn take_changes(&mut self) -> Vec<PeerEventView> {
        std::mem::take(&mut self.changes)
    }

    /// Lease records held.
    pub fn num_records(&self) -> usize {
        self.records.len()
    }

    pub(super) fn answer(&self, lease: &PeerLeaseView, kind: PeerAnswerKind) -> PeerAnswerView {
        PeerAnswerView {
            lease: lease.clone(),
            kind,
            num_tokens: 0,
            pages: Vec::new(),
            held: false,
        }
    }

    pub(super) fn reply(&self, lease: PeerLeaseView, answer: PeerAnswerView) -> (String, Reply) {
        (
            lease.reader,
            Reply::Answer {
                stamp: self.stamp.clone(),
                answer,
            },
        )
    }
}

/// An event with nothing set, to fill in.
pub fn empty_event() -> PeerEventView {
    PeerEventView {
        kind: PeerEventKind::Lookup,
        lease: PeerLeaseView {
            reader: String::new(),
            id: 0,
        },
        prefix_hashes: Vec::new(),
        tokens: Vec::new(),
        max_tokens: 0,
        pages: Vec::new(),
        demote: false,
        num_tokens: 0,
        dispatch_us: 0,
        writable_us: 0,
        terminal_us: 0,
        checkpoint: Vec::new(),
        request_id: String::new(),
    }
}

#[cfg(test)]
mod holder_metrics_tests {
    use super::*;

    #[test]
    fn counts_the_reply_sent_and_separates_read_from_cleanup() {
        for (kind, tokens, held, read, expected) in [
            (PeerAnswerKind::Lookup, 64, false, false, "lookup_hit"),
            (PeerAnswerKind::Lookup, 0, false, false, "lookup_miss"),
            (PeerAnswerKind::Pin, 0, false, false, "pin_rejected"),
            (PeerAnswerKind::Pin, 64, true, false, "pin_ok"),
            (PeerAnswerKind::Release, 0, true, true, "read_released"),
            (PeerAnswerKind::Release, 0, true, false, "cleanup_released"),
            (PeerAnswerKind::Release, 0, false, true, "release_rejected"),
        ] {
            let answer = PeerAnswerView {
                kind,
                num_tokens: tokens,
                held,
                lease: empty_event().lease,
                pages: vec![],
            };
            assert_eq!(holder_outcome(&answer, read), Some(expected));
        }
    }
}
