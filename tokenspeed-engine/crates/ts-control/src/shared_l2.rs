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
//! what to write to it; the client ([`PoolClient`]) talks to the owner on
//! threads of its own. The loop only relays. A waiting prompt's lookup
//! leaves once the scheduler holds the prompt, its answer reaches the
//! scheduler at the head of a round, and a plan's pool commands go to the
//! client when the plan is dispatched. Nothing here waits for the owner but
//! [`Pool::poll`], when the round says it may.
//!
//! A slow, wrong or absent owner is never an error: the client answers a
//! miss and drops the write. The client answers one rank, so the pool needs
//! a replica of one member (attention TP 1).

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use ts_metrics::SharedL2Stats;
use ts_scheduler::{
    ExecutionPlan, SharedL2AnswerView, SharedL2LookupView, SharedL2OpKind, SharedL2OpView,
    SharedL2PageView, SharedL2StatsView, SharedL2WriteDoneView,
};

/// The longest a round waits for the answers of the lookups sent since the
/// last plan, where the round may wait at all (`EventLoop::poll_pool`).
pub const LOOKUP_WAIT: Duration = Duration::from_millis(2);

/// The least time between two polls that wait for nothing. The client's
/// threads need the interpreter lock to deliver an answer, and a loop
/// spinning through empty rounds would keep taking it from them.
const POLL_INTERVAL: Duration = Duration::from_micros(200);

/// The pool client itself failed; serving cannot continue.
#[derive(Debug, thiserror::Error)]
#[error("shared L2 pool client: {0}")]
pub struct PoolError(pub String);

/// What the client has for a round.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Polled {
    /// The lookups answered since the last poll. One past its deadline, or
    /// asked while the rank serves without the pool, is answered as a miss.
    pub answers: Vec<SharedL2AnswerView>,
    /// The writes that ended since the last poll.
    pub writes: Vec<SharedL2WriteDoneView>,
    /// The owner may have dropped pins the scheduler still counts on: a
    /// ticket the poll named, or one of these answers', no longer allows a
    /// load to start.
    pub lease_lost: bool,
}

/// The client's own counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClientStats {
    /// Lookups answered as a miss at their deadline, cumulative.
    pub lookup_timeouts: u64,
    /// Blocks of the writes it gave up or dropped, cumulative.
    pub dropped_write_blocks: u64,
    /// It serves without the pool.
    pub no_pool: bool,
}

/// The rank's client of the pool's owner. No call waits for the owner;
/// `poll` waits only as long as it is told to.
pub trait PoolClient {
    /// Send lookups to the owner.
    fn submit_lookups(&mut self, lookups: &[SharedL2LookupView]) -> Result<(), PoolError>;

    /// The answers and finished writes since the last poll, and whether
    /// every ticket of `tickets` and of those answers still allows a load to
    /// start. Waits up to `wait` for the answers of the `awaited` lookups.
    fn poll(
        &mut self,
        awaited: &[u64],
        wait: Duration,
        tickets: &[u64],
    ) -> Result<Polled, PoolError>;

    /// The proxy Host blocks the answer with `ticket` created, each with the
    /// pool page a load-back from it reads; before the plan that may plan
    /// that load.
    fn map_pages(&mut self, ticket: u64, pages: &[SharedL2PageView]) -> Result<(), PoolError>;

    /// Pool commands: writes to start, and releases, touches and abandons
    /// to tell the owner. Over all calls, the commands of one kind come in
    /// plan order. Between kinds no order is kept: a plan's writes may come
    /// in a later call than its other commands ([`Pool::dispatch`]).
    fn submit(&mut self, commands: &[SharedL2OpView]) -> Result<(), PoolError>;

    /// Its counters.
    fn stats(&mut self) -> Result<ClientStats, PoolError>;
}

/// The pool on this rank: its client, and the work in flight there.
pub struct Pool {
    client: Box<dyn PoolClient + Send>,
    /// Lookups sent and not answered, with their requests.
    pending: Vec<(u64, String)>,
    /// Of those, the ones sent since the last poll: the round may wait for
    /// them, once.
    fresh: Vec<u64>,
    /// Tickets of the hits the scheduler took and no plan released yet.
    tickets: Vec<u64>,
    /// Writes the client has not finished.
    writes: Vec<u64>,
    /// Writes not handed to the client yet, each batch with the forwards
    /// that must still be committed first, in plan order.
    held_writes: VecDeque<(usize, Vec<SharedL2OpView>)>,
    polled_at: Option<Instant>,
    metrics_sampled: Option<Instant>,
}

impl Pool {
    /// The pool behind `client`, with nothing in flight.
    pub fn new(client: Box<dyn PoolClient + Send>) -> Self {
        Self {
            client,
            pending: Vec::new(),
            fresh: Vec::new(),
            tickets: Vec::new(),
            writes: Vec::new(),
            held_writes: VecDeque::new(),
            polled_at: None,
            metrics_sampled: None,
        }
    }

    /// Send the lookups the scheduler handed out; none waits for the owner.
    pub fn look_up(&mut self, lookups: &[SharedL2LookupView]) -> Result<(), PoolError> {
        if lookups.is_empty() {
            return Ok(());
        }
        for lookup in lookups {
            self.pending
                .push((lookup.lookup_id, lookup.request_id.clone()));
            self.fresh.push(lookup.lookup_id);
        }
        self.client.submit_lookups(lookups)
    }

    /// Lookups not answered and writes not finished: the loop keeps polling
    /// while there are any.
    pub fn open(&self) -> usize {
        self.pending.len() + self.writes.len()
    }

    /// Whether lookups were sent since the last poll.
    pub fn has_fresh_lookups(&self) -> bool {
        !self.fresh.is_empty()
    }

    /// The requests whose lookups are not answered.
    pub fn waiting_requests(&self) -> impl Iterator<Item = &str> {
        self.pending.iter().map(|(_, request)| request.as_str())
    }

    /// The client's answers and finished writes, for the head of a round;
    /// with `wait`, after waiting up to [`LOOKUP_WAIT`] for the answers of
    /// the lookups sent since the last poll. `None` when the client was not
    /// asked, which says nothing about the lease either: nothing is in
    /// flight, or the last poll is less than [`POLL_INTERVAL`] old and this
    /// one would wait for nothing.
    pub fn poll(&mut self, wait: bool) -> Result<Option<Polled>, PoolError> {
        let fresh = std::mem::take(&mut self.fresh);
        if self.pending.is_empty() && self.writes.is_empty() && self.tickets.is_empty() {
            return Ok(None);
        }
        let awaited: Vec<u64> = fresh
            .into_iter()
            .filter(|fresh| wait && self.pending.iter().any(|(id, _)| id == fresh))
            .collect();
        if awaited.is_empty()
            && self
                .polled_at
                .is_some_and(|at| at.elapsed() < POLL_INTERVAL)
        {
            return Ok(None);
        }
        let wait = if awaited.is_empty() {
            Duration::ZERO
        } else {
            LOOKUP_WAIT
        };
        let polled = self.client.poll(&awaited, wait, &self.tickets)?;
        self.polled_at = Some(Instant::now());
        self.pending
            .retain(|(id, _)| !polled.answers.iter().any(|answer| answer.lookup_id == *id));
        self.writes
            .retain(|id| !polled.writes.iter().any(|write| write.write_id == *id));
        Ok(Some(polled))
    }

    /// The scheduler took an answer: its proxy blocks go to the client, and
    /// a hit's ticket counts as held until a plan releases it.
    pub fn applied(&mut self, ticket: u64, pages: &[SharedL2PageView]) -> Result<(), PoolError> {
        if ticket != 0 {
            self.tickets.push(ticket);
        }
        if pages.is_empty() {
            return Ok(());
        }
        self.client.map_pages(ticket, pages)
    }

    /// The scheduler refused an answer and holds nothing under its ticket,
    /// so no plan will release it: the client is told now.
    pub fn refused(&mut self, ticket: u64) -> Result<(), PoolError> {
        if ticket == 0 {
            return Ok(());
        }
        self.client.submit(&[SharedL2OpView {
            kind: SharedL2OpKind::Release,
            id: ticket,
            keys: Vec::new(),
            buckets: Vec::new(),
            blocks: Vec::new(),
            publications: Vec::new(),
        }])
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
        for command in &commands {
            match command.kind {
                SharedL2OpKind::Write => self.writes.push(command.id),
                SharedL2OpKind::Release => self.tickets.retain(|ticket| *ticket != command.id),
                // Its request left before the answer: nothing waits for it.
                SharedL2OpKind::Abandon => self.pending.retain(|(id, _)| *id != command.id),
                _ => {}
            }
        }
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

    /// The rank's pool observations, as the metrics take them: the
    /// scheduler's and the client's.
    pub fn stats(&mut self, scheduler: SharedL2StatsView) -> Result<SharedL2Stats, PoolError> {
        let client = self.client.stats()?;
        Ok(SharedL2Stats {
            lookups: scheduler.lookups as f64,
            hits: scheduler.hits as f64,
            lookup_timeouts: client.lookup_timeouts as f64,
            abandoned_lookups: scheduler.abandoned_lookups as f64,
            proxy_overflows: scheduler.proxy_overflows as f64,
            lease_expired_hits: scheduler.lease_expired_hits as f64,
            failed_writes: scheduler.failed_writes as f64,
            failed_write_blocks: scheduler.failed_write_blocks as f64,
            dropped_write_blocks: client.dropped_write_blocks as f64,
            pending_lookups: scheduler.pending_lookups as f64,
            tickets: scheduler.tickets as f64,
            writes_in_flight: scheduler.writes_in_flight as f64,
            no_pool: client.no_pool,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    /// One call a client got.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Call {
        Lookups(Vec<u64>),
        Poll {
            awaited: Vec<u64>,
            wait: Duration,
            tickets: Vec<u64>,
        },
        Pages(u64, usize),
    }

    /// A client that records its calls and answers the lookups it is told to
    /// at the next poll.
    #[derive(Default)]
    struct Recorder {
        calls: Arc<Mutex<Vec<Call>>>,
        answers: Arc<Mutex<Vec<u64>>>,
        lease_lost: bool,
    }

    impl PoolClient for Recorder {
        fn submit_lookups(&mut self, lookups: &[SharedL2LookupView]) -> Result<(), PoolError> {
            let ids = lookups.iter().map(|lookup| lookup.lookup_id).collect();
            self.calls.lock().unwrap().push(Call::Lookups(ids));
            Ok(())
        }

        fn poll(
            &mut self,
            awaited: &[u64],
            wait: Duration,
            tickets: &[u64],
        ) -> Result<Polled, PoolError> {
            self.calls.lock().unwrap().push(Call::Poll {
                awaited: awaited.to_vec(),
                wait,
                tickets: tickets.to_vec(),
            });
            Ok(Polled {
                answers: std::mem::take(&mut *self.answers.lock().unwrap())
                    .into_iter()
                    .map(|lookup_id| SharedL2AnswerView {
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
                    .collect(),
                writes: Vec::new(),
                lease_lost: self.lease_lost,
            })
        }

        fn map_pages(&mut self, ticket: u64, pages: &[SharedL2PageView]) -> Result<(), PoolError> {
            self.calls
                .lock()
                .unwrap()
                .push(Call::Pages(ticket, pages.len()));
            Ok(())
        }

        fn submit(&mut self, _: &[SharedL2OpView]) -> Result<(), PoolError> {
            Ok(())
        }

        fn stats(&mut self) -> Result<ClientStats, PoolError> {
            Ok(ClientStats::default())
        }
    }

    type Calls = Arc<Mutex<Vec<Call>>>;

    fn pool() -> (Pool, Calls, Arc<Mutex<Vec<u64>>>) {
        let client = Recorder::default();
        let (calls, answers) = (client.calls.clone(), client.answers.clone());
        (Pool::new(Box::new(client)), calls, answers)
    }

    fn lookup(lookup_id: u64, request: &str) -> SharedL2LookupView {
        SharedL2LookupView {
            lookup_id,
            request_id: request.to_owned(),
            prefix_hashes: Vec::new(),
            tokens: Vec::new(),
            max_tokens: 0,
        }
    }

    #[test]
    fn nothing_in_flight_polls_nothing() {
        let (mut pool, calls, _) = pool();
        assert_eq!(pool.poll(true).unwrap(), None);
        assert_eq!(pool.poll(false).unwrap(), None);
        assert_eq!(pool.open(), 0);
        assert!(calls.lock().unwrap().is_empty());
    }

    #[test]
    fn a_round_waits_for_a_lookup_once_and_only_while_it_is_open() {
        let (mut pool, calls, answers) = pool();
        pool.look_up(&[lookup(1, "a"), lookup(2, "b")]).unwrap();
        assert!(pool.has_fresh_lookups());
        assert_eq!(pool.waiting_requests().collect::<Vec<_>>(), ["a", "b"]);
        answers.lock().unwrap().push(1);
        assert_eq!(pool.poll(true).unwrap().unwrap().answers.len(), 1);
        // The lookup still open is no longer fresh: the next round does not
        // wait for it, however it polls.
        assert!(!pool.has_fresh_lookups());
        assert_eq!(pool.waiting_requests().collect::<Vec<_>>(), ["b"]);
        std::thread::sleep(POLL_INTERVAL);
        pool.poll(true).unwrap();
        // A lookup answered before the round's poll is not waited for.
        pool.look_up(&[lookup(3, "c")]).unwrap();
        answers.lock().unwrap().extend([2, 3]);
        std::thread::sleep(POLL_INTERVAL);
        pool.poll(false).unwrap();
        assert_eq!(pool.open(), 0);
        let poll = |awaited: &[u64], wait| Call::Poll {
            awaited: awaited.to_vec(),
            wait,
            tickets: Vec::new(),
        };
        assert_eq!(
            *calls.lock().unwrap(),
            [
                Call::Lookups(vec![1, 2]),
                poll(&[1, 2], LOOKUP_WAIT),
                poll(&[], Duration::ZERO),
                Call::Lookups(vec![3]),
                poll(&[], Duration::ZERO),
            ]
        );
    }

    #[test]
    fn polls_that_wait_for_nothing_keep_their_distance() {
        let (mut pool, calls, _) = pool();
        pool.look_up(&[lookup(1, "a")]).unwrap();
        let started = Instant::now();
        for _ in 0..10_000 {
            pool.poll(false).unwrap();
        }
        let polls = calls
            .lock()
            .unwrap()
            .iter()
            .filter(|call| matches!(call, Call::Poll { .. }))
            .count();
        // A spinning loop leaves the client's threads the interpreter.
        let most = started.elapsed().as_micros() / POLL_INTERVAL.as_micros() + 2;
        assert!(1 <= polls && polls as u128 <= most, "{polls} polls");
    }

    #[test]
    fn a_poll_that_does_not_ask_the_client_says_nothing_of_the_lease() {
        let mut pool = Pool::new(Box::new(Recorder {
            lease_lost: true,
            ..Recorder::default()
        }));
        pool.look_up(&[lookup(1, "a")]).unwrap();
        assert!(pool.poll(false).unwrap().unwrap().lease_lost);
        // The polls kept apart ask nothing: none of them reports the lease
        // as held, which only the client can say.
        let polls: Vec<_> = (0..10_000).map(|_| pool.poll(false).unwrap()).collect();
        assert!(polls.iter().any(Option::is_none));
        assert!(polls.iter().flatten().all(|polled| polled.lease_lost));
    }

    #[test]
    fn a_hit_is_held_and_its_pages_go_to_the_client() {
        let (mut pool, calls, _) = pool();
        let page = SharedL2PageView {
            group_id: 0,
            block: 3,
            bucket: 0,
            page: 9,
        };
        pool.applied(7, &[page, page]).unwrap();
        // A miss holds nothing and maps nothing.
        pool.applied(0, &[]).unwrap();
        // The held ticket alone keeps the poll going: its lease is checked.
        pool.poll(false).unwrap();
        assert_eq!(
            *calls.lock().unwrap(),
            [
                Call::Pages(7, 2),
                Call::Poll {
                    awaited: Vec::new(),
                    wait: Duration::ZERO,
                    tickets: vec![7],
                },
            ]
        );
    }
}
