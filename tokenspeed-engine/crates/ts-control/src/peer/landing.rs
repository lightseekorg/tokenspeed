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

//! One peer landing on the reader, as every member of the replica tracks it
//! from the facts the replica agreed on (`docs/rfcs/0004-peer-prefix-sharing.md`):
//! the holder's layout, the pin's answer, each member's read outcome and the
//! release's answer. The members step it alike; the first member sends what
//! a step asks for, and each member reads its own shard.

use std::sync::Arc;

use ts_scheduler::{
    profile::{self, Lane, Lanes},
    PeerAnswerView, PeerFetchView, PeerLeaseView,
};

use crate::peer::{prepared::PreparedObject, read::ReadOutcome};

/// How long each wait may last, in the agreed peer clock's microseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    /// For the holder's layout or the pin's answer.
    pub answer_us: i64,
    /// What a failed read's destinations are held for before they return: a
    /// read the transfer engine gave up on may still have writes queued at
    /// the NIC.
    pub quarantine_us: i64,
    /// How long a holder may take to push its Disk pages
    /// (`docs/rfcs/0006-disk-tier.md`); it posts no write past it.
    pub push_us: i64,
}

/// Where a landing stands.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Phase {
    /// Waiting for the holder's layout.
    Layout,
    /// Waiting for the pin's answer.
    Pin,
    /// Pinned: each member reads once its plan's device work has run.
    Read,
    /// Every member reported; waiting for the release's answer.
    Release { read: bool },
    /// A read failed: the destinations are held until `until_us`.
    Quarantine { until_us: i64 },
    /// At least one member cannot prove retirement. No timer releases pages.
    Unretired,
    /// Finished with this boundary.
    Done(i32),
}

/// What a step asks the replica to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// The first member asks the holder for its layout.
    SendLayout,
    /// The first member pins the lease.
    SendPin,
    /// The first member releases the lease.
    SendRelease {
        /// Ask the holder to demote its copy.
        demote: bool,
    },
}

/// A landing's row on a profile timeline: each phase, then the whole.
#[derive(Clone)]
struct Timeline {
    lane: Arc<Lane>,
    started_ns: i64,
    phase_ns: i64,
}

impl std::fmt::Debug for Timeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timeline")
            .field("started_ns", &self.started_ns)
            .finish_non_exhaustive()
    }
}

/// One landing.
#[derive(Debug, Clone)]
pub struct Landing {
    fetch: PeerFetchView,
    metrics: Option<std::sync::Arc<ts_metrics::PeerMetrics>>,
    /// None unless a profile records.
    timeline: Option<Timeline>,
    pub(super) device_times: Option<(i64, i64)>,
    started_us: i64,
    failure: Option<&'static str>,
    phase_us: i64,
    phase: Phase,
    deadline_us: i64,
    last_renewal_us: i64,
    retried_release: bool,
    timeouts: Timeouts,
    /// Each member's read outcome, once reported.
    outcomes: Vec<Option<ReadOutcome>>,
    /// The pin's pages, once it held.
    pinned: Option<PeerAnswerView>,
    /// Prepared storage sources have no cache block ids.
    prepared: bool,
}

impl Landing {
    /// A landing of `fetch` across `members` members, and its first action.
    pub fn new(
        fetch: PeerFetchView,
        members: usize,
        layout_known: bool,
        now_us: i64,
        timeouts: Timeouts,
    ) -> (Self, Action) {
        static LANDINGS: Lanes = Lanes::new("Peer landing");
        let (phase, action) = if layout_known {
            (Phase::Pin, Action::SendPin)
        } else {
            (Phase::Layout, Action::SendLayout)
        };
        (
            Self {
                fetch,
                metrics: None,
                timeline: LANDINGS.hold().map(|lane| {
                    let now_ns = profile::now_ns();
                    Timeline {
                        lane: Arc::new(lane),
                        started_ns: now_ns,
                        phase_ns: now_ns,
                    }
                }),
                device_times: None,
                started_us: now_us,
                failure: None,
                phase_us: now_us,
                phase,
                last_renewal_us: now_us,
                retried_release: false,
                deadline_us: deadline(now_us, timeouts.answer_us),
                timeouts,
                outcomes: vec![None; members],
                pinned: None,
                prepared: false,
            },
            action,
        )
    }

    pub(super) fn observe_with(
        &mut self,
        metrics: Option<std::sync::Arc<ts_metrics::PeerMetrics>>,
    ) {
        self.metrics = metrics;
    }

    fn transition(&mut self, next: Phase, outcome: &str, now_us: i64) {
        let stage = match self.phase {
            Phase::Layout => "layout",
            Phase::Pin => "pin",
            Phase::Read => "read_agreement",
            Phase::Release { .. } => "release",
            Phase::Quarantine { .. } => "quarantine",
            Phase::Unretired => "unretired",
            Phase::Done(_) => return,
        };
        if let Some(timeline) = &mut self.timeline {
            let now_ns = profile::now_ns();
            let lane = &timeline.lane;
            lane.record(profile::HOST, stage.into(), timeline.phase_ns, now_ns);
            if let Phase::Done(end) = next {
                let fetch = &self.fetch;
                lane.record(
                    profile::HOST,
                    format!(
                        "Peer landing {}: tokens {}..{end} of {} from {}",
                        fetch.request_id, fetch.begin_tokens, fetch.end_tokens, fetch.holder
                    )
                    .into(),
                    timeline.started_ns,
                    now_ns,
                );
            }
            timeline.phase_ns = now_ns;
        }
        if let Some(metrics) = &self.metrics {
            if outcome != "ok" && self.failure.is_none() {
                self.failure = Some(stage);
            }
            metrics.stage(stage, outcome, (now_us - self.phase_us) as f64 / 1e6);
            if let Phase::Done(end) = next {
                metrics.stage(
                    "landing",
                    if end == self.fetch.end_tokens {
                        "ok"
                    } else {
                        "fallback"
                    },
                    (now_us - self.started_us) as f64 / 1e6,
                );
            }
        }
        self.phase_us = now_us;
        self.phase = next;
    }

    pub(super) fn failure(&self) -> Option<&'static str> {
        self.failure
    }

    pub(super) fn phase_started_us(&self) -> i64 {
        self.phase_us
    }

    /// The fetch.
    pub fn fetch(&self) -> &PeerFetchView {
        &self.fetch
    }

    /// The lease.
    pub fn lease(&self) -> &PeerLeaseView {
        &self.fetch.lease
    }

    /// The holder's layout arrived.
    pub fn layout_known(&mut self, now_us: i64) -> Option<Action> {
        if self.phase != Phase::Layout {
            return None;
        }
        self.transition(Phase::Pin, "ok", now_us);
        self.deadline_us = deadline(now_us, self.timeouts.answer_us);
        Some(Action::SendPin)
    }

    /// The pin's answer, `trusted` when it came from the holder whose
    /// layout this landing reads with.
    pub fn pinned(&mut self, answer: PeerAnswerView, trusted: bool, now_us: i64) -> Option<Action> {
        if self.phase != Phase::Pin {
            return None;
        }
        let covers = self.fetch.destinations.iter().all(|destination| {
            answer
                .pages
                .iter()
                .any(|page| page.group_id == destination.group_id && page.slot == destination.slot)
        });
        if trusted && answer.held && answer.num_tokens == self.fetch.end_tokens && covers {
            self.transition(Phase::Read, "ok", now_us);
            self.last_renewal_us = now_us;
            self.pinned = Some(answer);
            None
        } else {
            // A pin that holds anything must go; nothing was read.
            self.release(false, now_us)
        }
    }

    /// Validate a prepared prefix using logical dependencies only. No source
    /// address is interpreted as a cache-pool block id.
    pub fn pinned_prepared(
        &mut self,
        num_tokens: i32,
        ranks: &[Vec<PreparedObject>],
        trusted: bool,
        now_us: i64,
    ) -> Option<Action> {
        if self.phase != Phase::Pin {
            return None;
        }
        let covers = ranks.len() == self.outcomes.len()
            && ranks.iter().all(|objects| {
                self.fetch.destinations.iter().all(|destination| {
                    objects.iter().any(|object| {
                        object.group_id == destination.group_id && object.slot == destination.slot
                    })
                })
            });
        if trusted && num_tokens == self.fetch.end_tokens && covers {
            self.prepared = true;
            self.transition(Phase::Read, "ok", now_us);
            None
        } else {
            self.release(false, now_us)
        }
    }

    /// Pinned cache pages or prepared source objects may now be read.
    pub fn ready_to_read(&self) -> bool {
        self.phase == Phase::Read
    }

    pub(super) fn renew_read(&mut self, now_us: i64, lifetime_us: i64) -> bool {
        if self.phase == Phase::Read
            && now_us.saturating_sub(self.last_renewal_us) >= (lifetime_us / 3).max(1)
        {
            self.last_renewal_us = now_us;
            return true;
        }
        false
    }

    // A lost request or acknowledgement is safe to replay on autonomous holders.
    pub(super) fn retry_release(&mut self, now_us: i64) -> Option<Action> {
        if let Phase::Release { read } = self.phase {
            if now_us >= self.deadline_us && !self.retried_release {
                self.retried_release = true;
                self.deadline_us = deadline(now_us, self.timeouts.answer_us);
                return Some(Action::SendRelease { demote: read });
            }
        }
        None
    }

    // The caller has checked that this worker never submitted the read.
    pub(super) fn cancel_before_read(&mut self, now_us: i64) -> Option<Action> {
        match self.phase {
            Phase::Layout => {
                self.transition(Phase::Done(self.fetch.begin_tokens), "cancelled", now_us);
                None
            }
            Phase::Pin | Phase::Read => {
                self.pinned = None;
                self.release(false, now_us)
            }
            _ => None,
        }
    }

    /// The pages to read, once the pin held and while the members read.
    pub fn pages_to_read(&self) -> Option<&PeerAnswerView> {
        (self.phase == Phase::Read)
            .then_some(self.pinned.as_ref())
            .flatten()
    }

    /// `member`'s read ended.
    pub fn read(&mut self, member: usize, outcome: ReadOutcome, now_us: i64) -> Option<Action> {
        if self.phase != Phase::Read {
            return None;
        }
        let slot = self.outcomes.get_mut(member)?;
        slot.get_or_insert(outcome);
        if self.outcomes.iter().any(Option::is_none) {
            return None;
        }
        if self
            .outcomes
            .iter()
            .any(|outcome| *outcome == Some(ReadOutcome::Unretired))
        {
            self.transition(Phase::Unretired, "unretired", now_us);
            return None;
        }
        let read = self
            .outcomes
            .iter()
            .all(|outcome| *outcome == Some(ReadOutcome::Complete));
        self.release(read, now_us)
    }

    /// The release's answer.
    pub fn released(&mut self, answer: &PeerAnswerView, trusted: bool, now_us: i64) {
        if let Phase::Release { read } = self.phase {
            self.settle(
                read,
                trusted && answer.held,
                if trusted && answer.held {
                    "ok"
                } else {
                    "rejected"
                },
                now_us,
            );
        }
    }

    pub(super) fn sent(&mut self, at: i64) {
        if matches!(
            self.phase,
            Phase::Layout | Phase::Pin | Phase::Release { .. }
        ) {
            self.deadline_us = deadline(at, self.timeouts.answer_us);
        }
    }

    /// A pin that never acknowledged cannot have started a read. Its
    /// holder may nevertheless have applied it and needs an explicit release.
    pub(super) fn unanswered_pin(&self, now_us: i64) -> bool {
        self.phase == Phase::Pin && now_us >= self.deadline_us
    }

    /// The agreed clock moved to `now_us`: a wait past its deadline ends.
    pub fn tick(&mut self, now_us: i64) {
        match self.phase {
            // No read started: nothing can still write the destinations.
            Phase::Layout | Phase::Pin if now_us >= self.deadline_us => {
                self.transition(Phase::Done(self.fetch.begin_tokens), "timeout", now_us);
            }
            // An unanswered release vouches for nothing.
            Phase::Release { read } if now_us >= self.deadline_us => {
                self.settle(read, false, "timeout", now_us)
            }
            Phase::Quarantine { until_us } if now_us >= until_us => {
                self.transition(Phase::Done(self.fetch.begin_tokens), "ok", now_us);
            }
            _ => {}
        }
    }

    /// A member reported a native failure without retirement proof. The owner
    /// must fail closed while keeping the scheduler's destinations allocated.
    pub fn unretired(&self) -> bool {
        self.phase == Phase::Unretired
    }

    /// The boundary the landing ended at, once it ended.
    pub fn done(&self) -> Option<i32> {
        match self.phase {
            Phase::Done(tokens) => Some(tokens),
            _ => None,
        }
    }

    fn release(&mut self, read: bool, now_us: i64) -> Option<Action> {
        self.transition(
            Phase::Release { read },
            if read { "ok" } else { "failed" },
            now_us,
        );
        self.deadline_us = deadline(now_us, self.timeouts.answer_us);
        Some(Action::SendRelease { demote: read })
    }

    fn settle(&mut self, read: bool, vouched: bool, outcome: &str, now_us: i64) {
        let failed_read = (self.pinned.is_some() || self.prepared) && !read;
        let next = if failed_read {
            Phase::Quarantine {
                until_us: deadline(now_us, self.timeouts.quarantine_us),
            }
        } else if read && vouched {
            Phase::Done(self.fetch.end_tokens)
        } else {
            Phase::Done(self.fetch.begin_tokens)
        };
        self.transition(next, outcome, now_us);
    }
}

fn deadline(now_us: i64, lifetime_us: i64) -> i64 {
    now_us.saturating_add(lifetime_us)
}

#[cfg(test)]
mod tests {
    use ts_scheduler::{PeerAnswerKind, PeerPageView, PeerTier};

    use super::*;

    const TIMEOUTS: Timeouts = Timeouts {
        answer_us: 100,
        quarantine_us: 1000,
        push_us: 500,
    };

    fn page(slot: i32) -> PeerPageView {
        PeerPageView {
            group_id: 0,
            slot,
            tier: PeerTier::Device,
            block_id: slot + 10,
        }
    }

    fn fetch() -> PeerFetchView {
        PeerFetchView {
            request_id: "r".into(),
            lease: PeerLeaseView {
                reader: "reader:1".into(),
                id: 1,
            },
            holder: "holder:1".into(),
            begin_tokens: 0,
            end_tokens: 8,
            destinations: vec![page(0), page(1)],
            checkpoint: vec![],
            prefix_hashes: vec![],
            tokens: vec![],
        }
    }

    fn answer(
        kind: PeerAnswerKind,
        held: bool,
        num_tokens: i32,
        pages: Vec<PeerPageView>,
    ) -> PeerAnswerView {
        PeerAnswerView {
            lease: fetch().lease,
            kind,
            num_tokens,
            pages,
            held,
        }
    }

    fn pinned_landing() -> Landing {
        let (mut landing, action) = Landing::new(fetch(), 2, true, 0, TIMEOUTS);
        assert_eq!(action, Action::SendPin);
        assert_eq!(
            landing.pinned(
                answer(PeerAnswerKind::Pin, true, 8, vec![page(0), page(1)]),
                true,
                0
            ),
            None
        );
        assert!(landing.pages_to_read().is_some());
        landing
    }

    #[test]
    fn an_unretired_read_never_releases_sources_or_destinations_on_a_timer() {
        let mut landing = pinned_landing();
        assert_eq!(landing.read(0, ReadOutcome::Unretired, 1), None);
        assert_eq!(landing.read(1, ReadOutcome::Complete, 2), None);
        assert!(landing.unretired());
        landing.tick(i64::MAX);
        assert_eq!(landing.done(), None);
        // A late response cannot turn an unproven transfer into a completed one.
        landing.released(&answer(PeerAnswerKind::Release, true, 0, vec![]), true, 3);
        assert_eq!(landing.done(), None);
    }

    #[test]
    fn a_landing_every_member_read_and_the_holder_vouched_for_keeps_its_end() {
        let mut landing = pinned_landing();
        assert_eq!(
            landing.read(1, ReadOutcome::Complete, 5),
            None,
            "one member still reads"
        );
        assert_eq!(
            landing.read(0, ReadOutcome::Complete, 6),
            Some(Action::SendRelease { demote: true })
        );
        landing.released(&answer(PeerAnswerKind::Release, true, 0, vec![]), true, 7);
        assert_eq!(landing.done(), Some(8));
        landing.transition(super::Phase::Layout, "ok", 8);
        assert_eq!(landing.done(), Some(8), "a finished landing is terminal");
    }

    #[test]
    fn renew_only_an_active_read_and_replay_release_once() {
        let mut landing = pinned_landing();
        assert!(!landing.renew_read(19, 60));
        assert!(landing.renew_read(20, 60));
        assert!(!landing.renew_read(39, 60));
        assert!(landing.renew_read(40, 60));
        assert_eq!(landing.read(0, ReadOutcome::Complete, 41), None);
        let release = Some(Action::SendRelease { demote: true });
        assert_eq!(landing.read(1, ReadOutcome::Complete, 42), release);
        assert!(!landing.renew_read(142, 60));
        assert_eq!(landing.retry_release(141), None);
        assert_eq!(landing.retry_release(142), release);
        assert_eq!(landing.retry_release(242), None);
        landing.tick(242);
        assert_eq!(landing.done(), Some(0));
    }

    #[test]
    fn draining_before_submission_cannot_authorize_a_late_read() {
        for known in [false, true] {
            let (mut landing, _) = Landing::new(fetch(), 1, known, 0, TIMEOUTS);
            let action = landing.cancel_before_read(1);
            assert_eq!(
                action,
                known.then_some(Action::SendRelease { demote: false })
            );
            assert_eq!(landing.layout_known(2), None);
            assert_eq!(
                landing.pinned(
                    answer(PeerAnswerKind::Pin, true, 8, vec![page(0), page(1)]),
                    true,
                    3
                ),
                None
            );
            assert!(landing.pages_to_read().is_none());
        }
        let mut pinned = pinned_landing();
        assert_eq!(
            pinned.cancel_before_read(5),
            Some(Action::SendRelease { demote: false })
        );
        assert!(pinned.pages_to_read().is_none());
    }

    #[test]
    fn a_landing_starts_with_the_layout_when_it_is_unknown() {
        let (mut landing, action) = Landing::new(fetch(), 1, false, 0, TIMEOUTS);
        assert_eq!(action, Action::SendLayout);
        assert_eq!(landing.layout_known(10), Some(Action::SendPin));
        assert_eq!(landing.layout_known(11), None);
        landing.tick(109);
        assert_eq!(landing.done(), None);
        landing.tick(110);
        assert_eq!(landing.done(), Some(0), "the pin's answer timed out");
    }

    #[test]
    fn an_unusable_pin_is_released_and_reads_nothing() {
        for (held, tokens, pages, trusted) in [
            (false, 8, vec![page(0), page(1)], true),
            (true, 6, vec![page(0), page(1)], true),
            (true, 8, vec![page(0)], true),
            (true, 8, vec![page(0), page(1)], false),
        ] {
            let (mut landing, _) = Landing::new(fetch(), 1, true, 0, TIMEOUTS);
            assert_eq!(
                landing.pinned(answer(PeerAnswerKind::Pin, held, tokens, pages), trusted, 1),
                Some(Action::SendRelease { demote: false })
            );
            assert!(landing.pages_to_read().is_none());
            landing.released(&answer(PeerAnswerKind::Release, true, 0, vec![]), true, 2);
            assert_eq!(landing.done(), Some(0), "nothing was read: no quarantine");
        }
    }

    #[test]
    fn a_failed_read_holds_its_destinations_through_the_quarantine() {
        let mut landing = pinned_landing();
        landing.read(0, ReadOutcome::Complete, 1);
        assert_eq!(
            landing.read(1, ReadOutcome::FailedRetired, 2),
            Some(Action::SendRelease { demote: false })
        );
        landing.released(&answer(PeerAnswerKind::Release, true, 0, vec![]), true, 3);
        assert_eq!(landing.done(), None);
        landing.tick(1002);
        assert_eq!(landing.done(), None);
        landing.tick(1003);
        assert_eq!(landing.done(), Some(0));
    }

    #[test]
    fn an_unvouched_or_unanswered_release_gives_a_complete_read_up() {
        let mut landing = pinned_landing();
        landing.read(0, ReadOutcome::Complete, 1);
        landing.read(1, ReadOutcome::Complete, 1);
        landing.released(&answer(PeerAnswerKind::Release, false, 0, vec![]), true, 2);
        assert_eq!(
            landing.done(),
            Some(0),
            "the pages may have changed under the read"
        );

        let mut landing = pinned_landing();
        landing.read(0, ReadOutcome::Complete, 1);
        landing.read(1, ReadOutcome::Complete, 1);
        landing.tick(101);
        assert_eq!(landing.done(), Some(0), "no answer vouches for nothing");
    }

    #[test]
    fn a_member_reports_once_and_reports_outside_a_read_change_nothing() {
        let mut landing = pinned_landing();
        landing.read(0, ReadOutcome::Complete, 1);
        landing.read(0, ReadOutcome::FailedRetired, 1);
        assert_eq!(
            landing.read(1, ReadOutcome::Complete, 1),
            Some(Action::SendRelease { demote: true })
        );
        assert_eq!(landing.read(1, ReadOutcome::Complete, 1), None);
        assert_eq!(landing.read(5, ReadOutcome::Complete, 1), None);
    }
}
