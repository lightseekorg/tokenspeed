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

use ts_scheduler::{PeerAnswerView, PeerFetchView, PeerLeaseView};

/// How long each wait may last, in the agreed peer clock's microseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    /// For the holder's layout or the pin's answer.
    pub answer_us: i64,
    /// What a failed read's destinations are held for before they return: a
    /// read the transfer engine gave up on may still have writes queued at
    /// the NIC.
    pub quarantine_us: i64,
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

/// One landing.
#[derive(Debug, Clone)]
pub struct Landing {
    fetch: PeerFetchView,
    phase: Phase,
    deadline_us: i64,
    timeouts: Timeouts,
    /// Each member's read outcome, once reported.
    outcomes: Vec<Option<bool>>,
    /// The pin's pages, once it held.
    pinned: Option<PeerAnswerView>,
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
        let (phase, action) = if layout_known {
            (Phase::Pin, Action::SendPin)
        } else {
            (Phase::Layout, Action::SendLayout)
        };
        (
            Self {
                fetch,
                phase,
                deadline_us: deadline(now_us, timeouts.answer_us),
                timeouts,
                outcomes: vec![None; members],
                pinned: None,
            },
            action,
        )
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
        self.phase = Phase::Pin;
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
            self.phase = Phase::Read;
            self.pinned = Some(answer);
            None
        } else {
            // A pin that holds anything must go; nothing was read.
            self.release(false, now_us)
        }
    }

    /// The pages to read, once the pin held and while the members read.
    pub fn pages_to_read(&self) -> Option<&PeerAnswerView> {
        (self.phase == Phase::Read)
            .then_some(self.pinned.as_ref())
            .flatten()
    }

    /// `member`'s read ended.
    pub fn read(&mut self, member: usize, ok: bool, now_us: i64) -> Option<Action> {
        if self.phase != Phase::Read {
            return None;
        }
        let slot = self.outcomes.get_mut(member)?;
        slot.get_or_insert(ok);
        if self.outcomes.iter().any(Option::is_none) {
            return None;
        }
        let read = self.outcomes.iter().all(|outcome| *outcome == Some(true));
        self.release(read, now_us)
    }

    /// The release's answer.
    pub fn released(&mut self, answer: &PeerAnswerView, trusted: bool, now_us: i64) {
        if let Phase::Release { read } = self.phase {
            self.settle(read, trusted && answer.held, now_us);
        }
    }

    /// The agreed clock moved to `now_us`: a wait past its deadline ends.
    pub fn tick(&mut self, now_us: i64) {
        match self.phase {
            // No read started: nothing can still write the destinations.
            Phase::Layout | Phase::Pin if now_us >= self.deadline_us => {
                self.phase = Phase::Done(self.fetch.begin_tokens);
            }
            // An unanswered release vouches for nothing.
            Phase::Release { read } if now_us >= self.deadline_us => {
                self.settle(read, false, now_us)
            }
            Phase::Quarantine { until_us } if now_us >= until_us => {
                self.phase = Phase::Done(self.fetch.begin_tokens);
            }
            _ => {}
        }
    }

    /// The boundary the landing ended at, once it ended.
    pub fn done(&self) -> Option<i32> {
        match self.phase {
            Phase::Done(tokens) => Some(tokens),
            _ => None,
        }
    }

    fn release(&mut self, read: bool, now_us: i64) -> Option<Action> {
        self.phase = Phase::Release { read };
        self.deadline_us = deadline(now_us, self.timeouts.answer_us);
        Some(Action::SendRelease { demote: read })
    }

    fn settle(&mut self, read: bool, vouched: bool, now_us: i64) {
        let failed_read = self.pinned.is_some() && !read;
        self.phase = if failed_read {
            Phase::Quarantine {
                until_us: deadline(now_us, self.timeouts.quarantine_us),
            }
        } else if read && vouched {
            Phase::Done(self.fetch.end_tokens)
        } else {
            Phase::Done(self.fetch.begin_tokens)
        };
    }
}

fn deadline(now_us: i64, lifetime_us: i64) -> i64 {
    now_us.saturating_add(lifetime_us)
}

#[cfg(test)]
mod tests {
    use ts_scheduler::{PeerAnswerKind, PeerPageView};

    use super::*;

    const TIMEOUTS: Timeouts = Timeouts {
        answer_us: 100,
        quarantine_us: 1000,
    };

    fn page(slot: i32) -> PeerPageView {
        PeerPageView {
            group_id: 0,
            slot,
            host: false,
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
    fn a_landing_every_member_read_and_the_holder_vouched_for_keeps_its_end() {
        let mut landing = pinned_landing();
        assert_eq!(landing.read(1, true, 5), None, "one member still reads");
        assert_eq!(
            landing.read(0, true, 6),
            Some(Action::SendRelease { demote: true })
        );
        landing.released(&answer(PeerAnswerKind::Release, true, 0, vec![]), true, 7);
        assert_eq!(landing.done(), Some(8));
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
        landing.read(0, true, 1);
        assert_eq!(
            landing.read(1, false, 2),
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
        landing.read(0, true, 1);
        landing.read(1, true, 1);
        landing.released(&answer(PeerAnswerKind::Release, false, 0, vec![]), true, 2);
        assert_eq!(
            landing.done(),
            Some(0),
            "the pages may have changed under the read"
        );

        let mut landing = pinned_landing();
        landing.read(0, true, 1);
        landing.read(1, true, 1);
        landing.tick(101);
        assert_eq!(landing.done(), Some(0), "no answer vouches for nothing");
    }

    #[test]
    fn a_member_reports_once_and_reports_outside_a_read_change_nothing() {
        let mut landing = pinned_landing();
        landing.read(0, true, 1);
        landing.read(0, false, 1);
        assert_eq!(
            landing.read(1, true, 1),
            Some(Action::SendRelease { demote: true })
        );
        assert_eq!(landing.read(1, true, 1), None);
        assert_eq!(landing.read(5, true, 1), None);
    }
}
