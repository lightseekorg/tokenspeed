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

//! A shared L2 pool as the loop sees it: the rank's client of the pool's
//! owner, and what the loop keeps about the work it handed the client
//! (`docs/design/event-loop.md`).
//!
//! With a pool, the scheduler's Host tier is a proxy of one Host L2 that
//! every prefill rank shares: an owner process keeps its index and provider
//! processes its bytes. The C++ scheduler decides what to ask the pool and
//! what to write to it; the client ([`PoolClient`]) talks to the owner. The
//! loop only relays. Before a plan, the waiting prompts not looked up yet
//! ask the owner in one request, which the round waits for
//! ([`Pool::look_up`]); a plan's pool commands go to the client when the
//! plan is dispatched, and the client carries them out on threads of its
//! own. At the head of every round the loop asks the client for the writes
//! it finished and for a failure it stored ([`Pool::finished_writes`]).
//!
//! A failure of the client or of the owner ends serving, as a failed
//! load-back does; a miss is an answer of the owner that names no page. The
//! client answers one rank, so the pool needs a replica of one member
//! (attention TP 1).

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use ts_scheduler::{
    ExecutionPlan, SharedL2AnswerView, SharedL2LookupView, SharedL2OpKind, SharedL2OpView,
    SharedL2PageView,
};

/// The pool's client or its owner failed; serving cannot continue.
#[derive(Debug, thiserror::Error)]
#[error("shared L2 pool client: {0}")]
pub struct PoolError(pub String);

/// The rank's client of the pool's owner. Only `lookup` waits for the owner.
pub trait PoolClient {
    /// Ask the owner about `lookups` in one request and wait for its reply:
    /// one answer per lookup, in order. A miss is an answer without pages;
    /// a request the owner did not answer, or answered with an error, fails.
    fn lookup(
        &mut self,
        lookups: &[SharedL2LookupView],
    ) -> Result<Vec<SharedL2AnswerView>, PoolError>;

    /// The ids of the writes finished since the last poll; never waits.
    /// Fails when a write or a notice to the owner (a release, a touch, a
    /// ping) failed.
    fn poll(&mut self) -> Result<Vec<u64>, PoolError>;

    /// The proxy Host blocks an answer created, each with the pool page a
    /// load-back from it reads; before the plan that may plan that load.
    fn map_pages(&mut self, pages: &[SharedL2PageView]) -> Result<(), PoolError>;

    /// Pool commands: writes to start, and releases and touches to tell the
    /// owner. Over all calls, the commands of one kind come in plan order.
    /// Between kinds no order is kept: a plan's writes may come in a later
    /// call than its other commands ([`Pool::dispatch`]).
    fn submit(&mut self, commands: &[SharedL2OpView]) -> Result<(), PoolError>;
}

/// The pool on this rank: its client, and the writes in flight there.
pub struct Pool {
    client: Box<dyn PoolClient + Send>,
    /// Writes the client has not finished.
    writes: Vec<u64>,
    /// Writes not handed to the client yet, each batch with the forwards
    /// that must still be committed first, in plan order.
    held_writes: VecDeque<(usize, Vec<SharedL2OpView>)>,
    metrics_sampled: Option<Instant>,
}

impl Pool {
    /// The pool behind `client`, with nothing in flight.
    pub fn new(client: Box<dyn PoolClient + Send>) -> Self {
        Self {
            client,
            writes: Vec::new(),
            held_writes: VecDeque::new(),
            metrics_sampled: None,
        }
    }

    /// Ask the owner about the lookups the scheduler handed out, in one
    /// request, and wait for the answers: one per lookup, in order.
    pub fn look_up(
        &mut self,
        lookups: &[SharedL2LookupView],
    ) -> Result<Vec<SharedL2AnswerView>, PoolError> {
        let answers = self.client.lookup(lookups)?;
        if answers.len() != lookups.len()
            || answers
                .iter()
                .zip(lookups)
                .any(|(answer, lookup)| answer.lookup_id != lookup.lookup_id)
        {
            return Err(PoolError(format!(
                "{} lookups were answered by {} answers that do not match them in order",
                lookups.len(),
                answers.len()
            )));
        }
        Ok(answers)
    }

    /// The scheduler took an answer: the proxy blocks it created go to the
    /// client.
    pub fn applied(&mut self, pages: &[SharedL2PageView]) -> Result<(), PoolError> {
        if pages.is_empty() {
            return Ok(());
        }
        self.client.map_pages(pages)
    }

    /// Writes not finished: the loop is not idle while there are any.
    pub fn open(&self) -> usize {
        self.writes.len()
    }

    /// The ids of the writes the client finished since the last call, for
    /// the head of every round. The client is asked whether or not a write
    /// is in flight: its poll also raises the failure a write or a notice
    /// stored, a failed ping of an idle rank among them.
    pub fn finished_writes(&mut self) -> Result<Vec<u64>, PoolError> {
        let finished = self.client.poll()?;
        self.writes.retain(|id| !finished.contains(id));
        Ok(finished)
    }

    /// Hand `plan`'s pool commands to the client; none waits for the owner.
    /// Returns whether the plan had any.
    ///
    /// A write is held back while a forward dispatched before the plan is
    /// not committed (`forwards_in_flight` of them). The scheduler starts a
    /// publication's write in the plan after its chunk's, whether or not the
    /// chunk's forward landed, and the client reads the Device blocks over
    /// RDMA, which nothing orders behind the device's streams: the forward
    /// may still be filling them. [`Pool::forward_committed`] lets the
    /// writes go.
    pub fn dispatch(
        &mut self,
        plan: &ExecutionPlan,
        forwards_in_flight: usize,
    ) -> Result<bool, PoolError> {
        let mut commands = plan.shared_l2_ops();
        if commands.is_empty() {
            return Ok(false);
        }
        self.writes.extend(
            commands
                .iter()
                .filter(|command| command.kind == SharedL2OpKind::Write)
                .map(|command| command.id),
        );
        if forwards_in_flight > 0 {
            let (writes, rest): (Vec<_>, Vec<_>) = commands
                .into_iter()
                .partition(|command| command.kind == SharedL2OpKind::Write);
            if !writes.is_empty() {
                self.held_writes.push_back((forwards_in_flight, writes));
            }
            commands = rest;
        }
        if !commands.is_empty() {
            self.client.submit(&commands)?;
        }
        Ok(true)
    }

    /// A dispatched forward was committed: the writes that waited for it,
    /// and for no later one, go to the client.
    pub fn forward_committed(&mut self) -> Result<(), PoolError> {
        for (forwards, _) in &mut self.held_writes {
            *forwards -= 1;
        }
        // A plan's forwards in flight are also in flight for the plans
        // before it: the batches are released in plan order.
        while self
            .held_writes
            .front()
            .is_some_and(|(forwards, _)| *forwards == 0)
        {
            let (_, writes) = self.held_writes.pop_front().expect("checked above");
            self.client.submit(&writes)?;
        }
        Ok(())
    }

    /// Whether to sample the metrics now: at most once a second.
    pub fn metrics_due(&mut self) -> bool {
        let due = self
            .metrics_sampled
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(1));
        if due {
            self.metrics_sampled = Some(Instant::now());
        }
        due
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    /// A client that answers each lookup as a miss under the ids it is told
    /// to, and records the pages it is given.
    #[derive(Default)]
    struct Recorder {
        answers: Vec<u64>,
        mapped: Arc<Mutex<Vec<usize>>>,
    }

    impl PoolClient for Recorder {
        fn lookup(
            &mut self,
            _: &[SharedL2LookupView],
        ) -> Result<Vec<SharedL2AnswerView>, PoolError> {
            Ok(self
                .answers
                .iter()
                .map(|&lookup_id| SharedL2AnswerView {
                    lookup_id,
                    ticket: 0,
                    num_tokens: 0,
                    kv_tokens: 0,
                    tail_tokens: 0,
                    tail_hash: String::new(),
                    keys: Vec::new(),
                    buckets: Vec::new(),
                    pages: Vec::new(),
                })
                .collect())
        }

        fn poll(&mut self) -> Result<Vec<u64>, PoolError> {
            Ok(Vec::new())
        }

        fn map_pages(&mut self, pages: &[SharedL2PageView]) -> Result<(), PoolError> {
            self.mapped.lock().unwrap().push(pages.len());
            Ok(())
        }

        fn submit(&mut self, _: &[SharedL2OpView]) -> Result<(), PoolError> {
            Ok(())
        }
    }

    fn lookup(lookup_id: u64) -> SharedL2LookupView {
        SharedL2LookupView {
            lookup_id,
            request_id: String::new(),
            prefix_hashes: Vec::new(),
            tokens: Vec::new(),
            max_tokens: 0,
        }
    }

    #[test]
    fn answers_that_do_not_match_their_lookups_fail() {
        let answered = |answers: &[u64]| {
            Pool::new(Box::new(Recorder {
                answers: answers.to_vec(),
                ..Recorder::default()
            }))
            .look_up(&[lookup(1), lookup(2)])
            .map(|answers| answers.len())
            .map_err(|error| error.0)
        };
        assert_eq!(answered(&[1, 2]), Ok(2));
        // One answer short, one too many, and out of order: a prompt would
        // wait for good, or take another prompt's hit.
        for answers in [&[1][..], &[1, 2, 3], &[2, 1]] {
            let error = answered(answers).unwrap_err();
            assert!(error.contains("do not match"), "{error}");
        }
    }

    #[test]
    fn a_hits_pages_go_to_the_client() {
        let client = Recorder::default();
        let mapped = client.mapped.clone();
        let mut pool = Pool::new(Box::new(client));
        let page = SharedL2PageView {
            group_id: 0,
            block: 3,
            bucket: 0,
            page: 9,
        };
        pool.applied(&[page, page]).unwrap();
        // A miss maps nothing.
        pool.applied(&[]).unwrap();
        assert_eq!(*mapped.lock().unwrap(), [2]);
    }
}
