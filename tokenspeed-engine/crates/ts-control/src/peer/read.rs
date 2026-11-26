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

//! A member's reads of its holder's shard (`docs/rfcs/0004-peer-prefix-sharing.md`):
//! one-sided reads into the member's own pages, and the push of the holder's
//! Disk pages into them (`docs/rfcs/0006-disk-tier.md`), run off the control
//! thread, each job reporting once all of it retired.

use std::{
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use crate::peer::{
    geometry::Spans,
    push::{PagePusher, PushItem},
};

/// What moves a read's bytes.
pub trait PageReader: Send + Sync {
    /// Copy `spans` from the memory registered under `session`; returns once
    /// every copy retired, or the failure. May be called concurrently for
    /// separate jobs, as the transfer engine's entry points allow.
    fn read(&self, session: &str, spans: &Spans) -> Result<(), String>;
}

/// The transfer engine's one-sided READs.
pub struct EngineReader(pub Arc<ts_mooncake::Engine>);

impl PageReader for EngineReader {
    fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
        self.0
            .read(session, &spans.sources, &spans.destinations, &spans.lengths)
            .map_err(|error| error.to_string())
    }
}

/// One read to run: the landing it serves, the holder's session and the
/// spans, and the push of the holder's Disk pages, if any.
pub struct ReadJob {
    /// The landing request.
    pub request_id: String,
    /// The holder rank's transfer session.
    pub session: String,
    /// What to copy.
    pub spans: Spans,
    /// The Disk pages the holder rank pushes into this member's pages.
    pub push: Option<PushJob>,
}

/// A push the holder rank runs for a read (`push`).
pub struct PushJob {
    /// The holder rank's push endpoint.
    pub endpoint: String,
    /// This member's transfer session, which the holder writes into.
    pub session: String,
    /// How long the holder may take.
    pub budget: Duration,
    /// The blocks.
    pub items: Vec<PushItem>,
}

impl PushJob {
    /// The bytes it lands.
    pub fn bytes(&self) -> u64 {
        self.items
            .iter()
            .flat_map(|item| &item.spans)
            .map(|span| span.length)
            .sum()
    }
}

/// One member's evidence for closing a versioned landing.
#[derive(Debug, PartialEq, Eq)]
pub enum ReadRetirement {
    /// All submitted batches retired successfully (includes an empty plan).
    Complete,
    /// Submission was sealed and all previously submitted batches retired.
    Aborted,
    /// A transport failure did not establish retirement. Keep both endpoints pinned.
    Unknown(String),
}

/// Owned by the lease controller even if its business request goes away.
/// Sealing requests cancellation; only the retirement result can release pages.
pub struct MemberRead {
    sealed: Arc<AtomicBool>,
    result: mpsc::Receiver<ReadRetirement>,
}

impl MemberRead {
    /// Stop claiming new batches. A batch claimed just before seal remains in flight.
    pub fn seal(&self) {
        self.sealed.store(true, Ordering::SeqCst);
    }

    /// Take the single retirement result. A disconnected worker cannot certify retirement.
    pub fn try_retirement(&self) -> Result<ReadRetirement, mpsc::TryRecvError> {
        self.result.try_recv()
    }
}

struct MemberJob {
    submit_until: Instant,
    batches: Vec<(String, Spans, Option<ts_metrics::QueuedPeerRead>)>,
    sealed: Arc<AtomicBool>,
    done: mpsc::Sender<ReadRetirement>,
}

impl MemberJob {
    fn run(self, reader: &dyn PageReader) {
        for (session, spans, observation) in self.batches {
            if self.sealed.load(Ordering::SeqCst) || Instant::now() >= self.submit_until {
                let _ = self.done.send(ReadRetirement::Aborted);
                return;
            }
            if !spans.is_empty() {
                let start = std::time::Instant::now();
                let result = reader.read(&session, &spans);
                if let Some(observation) = observation {
                    observation.finish(start, start.elapsed(), spans.host_bytes, 0, result.is_ok());
                }
                if let Err(error) = result {
                    let _ = self.done.send(ReadRetirement::Unknown(error));
                    return;
                }
            }
        }
        let result = if self.sealed.load(Ordering::SeqCst) {
            ReadRetirement::Aborted
        } else {
            ReadRetirement::Complete
        };
        let _ = self.done.send(result);
    }
}

enum Work {
    Legacy(Box<ReadJob>, Option<ts_metrics::QueuedPeerRead>),
    Member(MemberJob),
}

/// The member's bounded read workers: jobs in, outcomes out.
pub struct Reads {
    jobs: Option<mpsc::Sender<Work>>,
    outcomes: mpsc::Receiver<(String, bool)>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl Reads {
    /// Run at most `concurrency` reads at once over `reader`, and their
    /// pushes through `pusher`. Queued reads keep their landing's
    /// destinations held until their outcome arrives.
    pub fn start(
        reader: Arc<dyn PageReader>,
        pusher: Arc<dyn PagePusher>,
        concurrency: NonZeroUsize,
    ) -> std::io::Result<Self> {
        let (jobs, incoming) = mpsc::channel::<Work>();
        let (done, outcomes) = mpsc::channel();
        let incoming = Arc::new(Mutex::new(incoming));
        // Own each worker as soon as it starts, so a later spawn failure
        // closes the queue and joins the workers that already started.
        let mut reads = Self {
            jobs: Some(jobs),
            outcomes,
            workers: Vec::with_capacity(concurrency.get()),
        };
        for index in 0..concurrency.get() {
            let (reader, pusher, incoming, done) = (
                reader.clone(),
                pusher.clone(),
                incoming.clone(),
                done.clone(),
            );
            let worker = thread::Builder::new()
                .name(format!("ts-peer-read-{index}"))
                .spawn(move || loop {
                    // Release the queue lock before the blocking transfer.
                    let job = incoming.lock().recv();
                    let (job, observation) = match job {
                        Ok(Work::Member(member)) => {
                            member.run(reader.as_ref());
                            continue;
                        }
                        Ok(Work::Legacy(job, observation)) => (job, observation),
                        Err(_) => return,
                    };
                    let started = std::time::Instant::now();
                    let ReadJob {
                        request_id,
                        session,
                        spans,
                        push,
                    } = *job;
                    let total_spans = spans.lengths.len();
                    let host_bytes = spans.host_bytes;
                    tracing::debug!(
                        request = request_id,
                        spans = total_spans,
                        "peer read started"
                    );
                    // Overlap `PushJob` concurrently with one-sided `READ`,
                    // joining both before reporting the landing outcome so no
                    // transfer trails behind settlement.
                    let (outcome, disk_bytes) = match (push, !spans.is_empty()) {
                        (Some(push), true) => {
                            let bytes = push.bytes();
                            let PushJob {
                                endpoint,
                                session: push_session,
                                budget,
                                items,
                            } = push;
                            let (read_res, push_res) = std::thread::scope(|scope| {
                                let push_handle = scope
                                    .spawn(|| pusher.push(&endpoint, &push_session, budget, items));
                                let read_res = reader.read(&session, &spans);
                                let push_res = push_handle.join().unwrap_or_else(|_| {
                                    Err("peer push worker panicked".to_owned())
                                });
                                (read_res, push_res)
                            });
                            (read_res.and(push_res), bytes)
                        }
                        (Some(push), false) => {
                            let bytes = push.bytes();
                            let res =
                                pusher.push(&push.endpoint, &push.session, push.budget, push.items);
                            (res, bytes)
                        }
                        (None, true) => (reader.read(&session, &spans), 0),
                        (None, false) => (Ok(()), 0),
                    };
                    let ok = outcome.is_ok();
                    if let Some(observation) = observation {
                        observation.finish(started, started.elapsed(), host_bytes, disk_bytes, ok);
                    }
                    tracing::debug!(
                        request = request_id,
                        ok,
                        elapsed_us = started.elapsed().as_micros() as u64,
                        "peer read ended"
                    );
                    if let Err(error) = outcome {
                        tracing::warn!(
                            request = request_id,
                            %error,
                            spans = total_spans,
                            first_source = spans.sources.first().map(|a| format!("{a:#x}")),
                            first_destination =
                                spans.destinations.first().map(|a| format!("{a:#x}")),
                            last_destination =
                                spans.destinations.last().map(|a| format!("{a:#x}")),
                            "peer read failed"
                        );
                    }
                    if done.send((request_id, ok)).is_err() {
                        return;
                    }
                })?;
            reads.workers.push(worker);
        }
        Ok(reads)
    }

    /// Queue a read.
    pub fn submit(&self, job: ReadJob) {
        self.submit_observed(job, None);
    }

    /// Queue this member's entire negotiated plan after its destination fence.
    /// Every queue entry owns exactly one completion sender regardless of the
    /// number of holder sessions. None means nothing was queued.
    pub fn submit_member(&self, batches: Vec<(String, Spans)>) -> Option<MemberRead> {
        self.submit_member_observed(batches, Instant::now() + super::v1::SUBMIT_WINDOW, None)
    }

    pub(super) fn submit_member_observed(
        &self,
        batches: Vec<(String, Spans)>,
        submit_until: Instant,
        metrics: Option<(&Arc<ts_metrics::PeerMetrics>, &str, usize)>,
    ) -> Option<MemberRead> {
        let (done, result) = mpsc::channel();
        let sealed = Arc::new(AtomicBool::new(false));
        let batches = batches
            .into_iter()
            .map(|(session, spans)| {
                let observation = metrics.map(|(m, holder, rank)| {
                    m.queue_read(holder.into(), rank, spans.total_bytes())
                });
                (session, spans, observation)
            })
            .collect();
        self.jobs
            .as_ref()?
            .send(Work::Member(MemberJob {
                submit_until,
                batches,
                sealed: sealed.clone(),
                done,
            }))
            .ok()?;
        Some(MemberRead { sealed, result })
    }

    pub(super) fn submit_observed(
        &self,
        job: ReadJob,
        observation: Option<ts_metrics::QueuedPeerRead>,
    ) {
        if let Some(jobs) = &self.jobs {
            let _ = jobs.send(Work::Legacy(Box::new(job), observation));
        }
    }

    /// The reads that ended since the last poll, by landing request.
    pub fn poll(&self) -> Vec<(String, bool)> {
        self.outcomes.try_iter().collect()
    }
}

impl Drop for Reads {
    fn drop(&mut self) {
        // Drain queued reads and retire every transfer before releasing the
        // reader and the registered memory it owns.
        self.jobs.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Memory that test replicas share: an address is an offset into it, and
/// a read copies within it. Nothing outside tests uses it.
#[derive(Clone, Default)]
pub struct Arena(Arc<std::sync::Mutex<Vec<u8>>>);

impl Arena {
    /// An arena of `bytes` zero bytes.
    pub fn new(bytes: usize) -> Self {
        Self(Arc::new(std::sync::Mutex::new(vec![0; bytes])))
    }

    /// Write `bytes` at `address`.
    pub fn write(&self, address: u64, bytes: &[u8]) {
        let mut memory = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let start = address as usize;
        memory[start..start + bytes.len()].copy_from_slice(bytes);
    }

    /// The `len` bytes at `address`.
    pub fn read_bytes(&self, address: u64, len: usize) -> Vec<u8> {
        let memory = self.0.lock().unwrap_or_else(|e| e.into_inner());
        memory[address as usize..address as usize + len].to_vec()
    }
}

impl PageReader for Arena {
    fn read(&self, _session: &str, spans: &Spans) -> Result<(), String> {
        let mut memory = self.0.lock().unwrap_or_else(|e| e.into_inner());
        for ((&source, &destination), &length) in spans
            .sources
            .iter()
            .zip(&spans.destinations)
            .zip(&spans.lengths)
        {
            let (source, destination, length) =
                (source as usize, destination as usize, length as usize);
            if source + length > memory.len() || destination + length > memory.len() {
                return Err("a span outside the arena".into());
            }
            memory.copy_within(source..source + length, destination);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use parking_lot::Condvar;

    use super::*;

    struct Failing;

    /// A member without pushes: a push fails.
    struct NoPushes;

    impl PagePusher for NoPushes {
        fn push(&self, _: &str, _: &str, _: Duration, _: Vec<PushItem>) -> Result<(), String> {
            Err("no pushes".into())
        }
    }

    /// Records the pushes it is asked for and lands them.
    struct Pushes(std::sync::Mutex<Vec<(String, usize)>>);

    impl PagePusher for Pushes {
        fn push(
            &self,
            endpoint: &str,
            _: &str,
            _: Duration,
            items: Vec<PushItem>,
        ) -> Result<(), String> {
            self.0
                .lock()
                .unwrap()
                .push((endpoint.to_owned(), items.len()));
            Ok(())
        }
    }

    impl PageReader for Failing {
        fn read(&self, _: &str, _: &Spans) -> Result<(), String> {
            Err("down".into())
        }
    }

    fn wait_for(reads: &Reads) -> Vec<(String, bool)> {
        let start = Instant::now();
        loop {
            let polled = reads.poll();
            if !polled.is_empty() || start.elapsed() > Duration::from_secs(5) {
                return polled;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn a_read_copies_its_spans_and_reports_once_done() {
        let arena = Arena::new(64);
        arena.write(4, &[7; 8]);
        let reads = Reads::start(
            Arc::new(arena.clone()),
            Arc::new(NoPushes),
            NonZeroUsize::new(2).unwrap(),
        )
        .unwrap();
        reads.submit(ReadJob {
            request_id: "r".into(),
            session: "s".into(),
            spans: Spans {
                sources: vec![4],
                destinations: vec![32],
                lengths: vec![8],
                host_bytes: 0,
            },
            push: None,
        });
        assert_eq!(wait_for(&reads), vec![("r".to_owned(), true)]);
        assert_eq!(arena.read_bytes(32, 8), vec![7; 8]);
        assert_eq!(arena.read_bytes(40, 8), vec![0; 8]);
    }

    #[test]
    fn a_job_reports_once_its_reads_and_its_push_ended() {
        let arena = Arena::new(64);
        arena.write(4, &[7; 8]);
        let push = || PushJob {
            endpoint: "holder:7200".into(),
            session: "reader".into(),
            budget: Duration::from_secs(1),
            items: vec![PushItem {
                group: 0,
                block: 3,
                spans: Vec::new(),
            }],
        };
        let pushes = Arc::new(Pushes(Default::default()));
        let reads = Reads::start(
            Arc::new(arena.clone()),
            pushes.clone(),
            NonZeroUsize::new(1).unwrap(),
        )
        .unwrap();
        reads.submit(ReadJob {
            request_id: "r".into(),
            session: "s".into(),
            spans: Spans {
                sources: vec![4],
                destinations: vec![32],
                lengths: vec![8],
                host_bytes: 0,
            },
            push: Some(push()),
        });
        assert_eq!(wait_for(&reads), vec![("r".to_owned(), true)]);
        assert_eq!(
            *pushes.0.lock().unwrap(),
            vec![("holder:7200".to_owned(), 1)]
        );
        // A failed push fails the read it rides with.
        let reads = Reads::start(
            Arc::new(arena),
            Arc::new(NoPushes),
            NonZeroUsize::new(1).unwrap(),
        )
        .unwrap();
        reads.submit(ReadJob {
            request_id: "r".into(),
            session: "s".into(),
            spans: Spans::default(),
            push: Some(push()),
        });
        assert_eq!(wait_for(&reads), vec![("r".to_owned(), false)]);
    }

    #[test]
    fn push_and_read_overlap_concurrently() {
        use std::sync::Barrier;
        struct BReader {
            arena: Arena,
            barrier: Arc<Barrier>,
            calls: std::sync::Mutex<Vec<usize>>,
        }
        impl PageReader for BReader {
            fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
                self.calls.lock().unwrap().push(spans.lengths.len());
                self.barrier.wait();
                self.arena.read(session, spans)
            }
        }
        struct BPusher {
            barrier: Arc<Barrier>,
            fail: bool,
        }
        impl PagePusher for BPusher {
            fn push(&self, _: &str, _: &str, _: Duration, _: Vec<PushItem>) -> Result<(), String> {
                self.barrier.wait();
                if self.fail {
                    Err("push failed".into())
                } else {
                    Ok(())
                }
            }
        }
        let arena = Arena::new(64);
        arena.write(0, &[9; 8]);
        arena.write(8, &[4; 8]);
        let push = || PushJob {
            endpoint: "holder:7200".into(),
            session: "r".into(),
            budget: Duration::from_secs(1),
            items: vec![PushItem {
                group: 0,
                block: 1,
                spans: Vec::new(),
            }],
        };
        for (fail, want) in [(false, true), (true, false)] {
            let barrier = Arc::new(Barrier::new(2));
            let reader = Arc::new(BReader {
                arena: arena.clone(),
                barrier: barrier.clone(),
                calls: Default::default(),
            });
            let reads = Reads::start(
                reader.clone(),
                Arc::new(BPusher { barrier, fail }),
                NonZeroUsize::new(1).unwrap(),
            )
            .unwrap();
            reads.submit(ReadJob {
                request_id: "j".into(),
                session: "s".into(),
                spans: Spans {
                    sources: vec![0, 8],
                    destinations: vec![24, 32],
                    lengths: vec![8, 8],
                    host_bytes: 0,
                },
                push: Some(push()),
            });
            assert_eq!(wait_for(&reads), vec![("j".into(), want)]);
            assert_eq!(*reader.calls.lock().unwrap(), vec![2]);
        }
        assert_eq!(arena.read_bytes(24, 8), vec![9; 8]);
        assert_eq!(arena.read_bytes(32, 8), vec![4; 8]);
    }

    #[test]
    fn a_failed_read_reports_its_failure() {
        for reader in [
            Arc::new(Failing) as Arc<dyn PageReader>,
            Arc::new(Arena::new(4)),
        ] {
            let reads =
                Reads::start(reader, Arc::new(NoPushes), NonZeroUsize::new(2).unwrap()).unwrap();
            reads.submit(ReadJob {
                request_id: "r".into(),
                session: "s".into(),
                spans: Spans {
                    sources: vec![1],
                    destinations: vec![2],
                    lengths: vec![8],
                    host_bytes: 0,
                },
                push: None,
            });
            assert_eq!(wait_for(&reads), vec![("r".to_owned(), false)]);
        }
    }

    struct PausedReader {
        arena: Arena,
        started: mpsc::Sender<String>,
        gate: Arc<(Mutex<bool>, Condvar)>,
    }

    impl PageReader for PausedReader {
        fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
            self.started.send(session.to_owned()).unwrap();
            if session == "paused" {
                let (released, changed) = &*self.gate;
                let mut released = released.lock();
                while !*released {
                    changed.wait(&mut released);
                }
            }
            self.arena.read(session, spans)
        }
    }

    // Declared after Reads, so failed assertions unblock the transfers before
    // the worker pool joins them during unwinding.
    struct ReleaseReads(Arc<(Mutex<bool>, Condvar)>);

    impl Drop for ReleaseReads {
        fn drop(&mut self) {
            *self.0 .0.lock() = true;
            self.0 .1.notify_all();
        }
    }

    fn read_job(request: &str, session: &str, destination: u64) -> ReadJob {
        ReadJob {
            request_id: request.into(),
            session: session.into(),
            spans: Spans {
                sources: vec![0],
                destinations: vec![destination],
                lengths: vec![8],
                host_bytes: 0,
            },
            push: None,
        }
    }

    #[test]
    fn member_seal_waits_for_the_claimed_read_and_skips_later_sessions() {
        let arena = Arena::new(64);
        let (started, incoming) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let reader = Arc::new(PausedReader {
            arena,
            started,
            gate: gate.clone(),
        });
        let reads = Reads::start(reader, Arc::new(NoPushes), NonZeroUsize::MIN).unwrap();
        let release = ReleaseReads(gate);
        let spans = read_job("r", "unused", 16).spans;
        let job = reads
            .submit_member(vec![
                ("paused".into(), spans.clone()),
                ("skipped".into(), spans),
            ])
            .unwrap();
        assert_eq!(
            incoming.recv_timeout(Duration::from_secs(5)).unwrap(),
            "paused"
        );
        job.seal();
        assert_eq!(job.try_retirement(), Err(mpsc::TryRecvError::Empty));
        drop(release);
        assert_eq!(
            job.result.recv_timeout(Duration::from_secs(5)).unwrap(),
            ReadRetirement::Aborted
        );
        assert!(incoming.try_recv().is_err());
    }

    #[test]
    fn expired_queue_entry_skips_native_submission() {
        let reads = Reads::start(Arc::new(Failing), Arc::new(NoPushes), NonZeroUsize::MIN).unwrap();
        let job = reads
            .submit_member_observed(
                vec![("unreachable".into(), read_job("r", "unused", 16).spans)],
                Instant::now(),
                None,
            )
            .unwrap();
        assert_eq!(
            job.result.recv_timeout(Duration::from_secs(5)).unwrap(),
            ReadRetirement::Aborted
        );
    }

    #[test]
    fn member_empty_read_votes_and_native_failure_has_unknown_retirement() {
        let reads = Reads::start(Arc::new(Failing), Arc::new(NoPushes), NonZeroUsize::MIN).unwrap();
        let empty = reads.submit_member(Vec::new()).unwrap();
        assert_eq!(
            empty.result.recv_timeout(Duration::from_secs(5)).unwrap(),
            ReadRetirement::Complete
        );
        let job = reads
            .submit_member(vec![("failed".into(), read_job("r", "unused", 16).spans)])
            .unwrap();
        assert!(matches!(
            job.result.recv_timeout(Duration::from_secs(5)).unwrap(),
            ReadRetirement::Unknown(_)
        ));
    }

    #[test]
    fn a_slow_holder_does_not_block_a_read_from_another_holder() {
        let arena = Arena::new(64);
        arena.write(0, &[7; 8]);
        let (started, incoming) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let reader = Arc::new(PausedReader {
            arena: arena.clone(),
            started,
            gate: gate.clone(),
        });
        let reads =
            Reads::start(reader, Arc::new(NoPushes), NonZeroUsize::new(2).unwrap()).unwrap();
        let release = ReleaseReads(gate);
        reads.submit(read_job("slow", "paused", 16));
        assert_eq!(
            incoming.recv_timeout(Duration::from_secs(5)).unwrap(),
            "paused"
        );
        reads.submit(read_job("fast", "ready", 32));
        assert_eq!(wait_for(&reads), vec![("fast".to_owned(), true)]);
        assert_eq!(arena.read_bytes(32, 8), vec![7; 8]);
        assert_eq!(arena.read_bytes(16, 8), vec![0; 8]);
        drop(release);
        assert_eq!(wait_for(&reads), vec![("slow".to_owned(), true)]);
        assert_eq!(arena.read_bytes(16, 8), vec![7; 8]);
    }

    #[test]
    fn read_concurrency_is_bounded_and_drop_drains_queued_jobs() {
        let arena = Arena::new(64);
        arena.write(0, &[7; 8]);
        let (started, incoming) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let reader = Arc::new(PausedReader {
            arena: arena.clone(),
            started,
            gate: gate.clone(),
        });
        let reads =
            Reads::start(reader, Arc::new(NoPushes), NonZeroUsize::new(2).unwrap()).unwrap();
        let release = ReleaseReads(gate);
        for index in 1..=4 {
            reads.submit(read_job(&format!("r{index}"), "paused", index * 8));
        }
        for _ in 0..2 {
            incoming.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        assert!(matches!(
            incoming.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        drop(release);
        drop(reads);
        assert_eq!(arena.read_bytes(8, 32), vec![7; 32]);
    }
}
