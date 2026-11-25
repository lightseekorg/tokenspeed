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
    sync::{mpsc, Arc},
    thread,
    time::Duration,
};

use parking_lot::Mutex;

use crate::peer::{
    geometry::Spans,
    push::{PagePusher, PushItem},
};

/// Completion and native retirement are separate facts. An error after submission
/// need not mean that the NIC stopped writing the destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadOutcome {
    /// All copies completed and retired.
    Complete,
    /// No successful prefix, but no native access can remain.
    FailedRetired,
    /// Native access may still be in flight; destinations must stay owned.
    Unretired,
}

impl ReadOutcome {
    /// Whether every requested byte was copied successfully.
    pub fn succeeded(self) -> bool {
        self == Self::Complete
    }
    /// Whether native accesses are known to have stopped.
    pub fn retired(self) -> bool {
        self != Self::Unretired
    }
}

/// What moves a read's bytes.
pub trait PageReader: Send + Sync {
    /// Copy `spans` from memory registered under `session`. Success proves all
    /// copies retired; an error alone carries no retirement guarantee. May be called concurrently for
    /// separate jobs, as the transfer engine's entry points allow.
    fn read(&self, session: &str, spans: &Spans) -> Result<(), String>;

    /// A backend may report a retired failure only with an actual retirement
    /// guarantee. The native bridge currently proves retirement on success only.
    fn read_outcome(&self, session: &str, spans: &Spans) -> ReadOutcome {
        match self.read(session, spans) {
            Ok(()) => ReadOutcome::Complete,
            Err(error) => {
                tracing::warn!(%error, "native read retirement is unproven");
                ReadOutcome::Unretired
            }
        }
    }
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
    /// Real source sessions and their spans. Storage objects can live at more
    /// than one provider; the member reports once for the whole landing.
    pub sources: Vec<(String, Spans)>,
    /// Optional Disk pages pushed by the same holder.
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

/// The member's bounded read workers: jobs in, outcomes out.
pub struct Reads {
    jobs: Option<mpsc::Sender<(ReadJob, Option<ts_metrics::QueuedPeerRead>)>>,
    outcomes: mpsc::Receiver<(String, ReadOutcome)>,
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
        let (jobs, incoming) = mpsc::channel::<(ReadJob, Option<ts_metrics::QueuedPeerRead>)>();
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
                .spawn(move || {
                    loop {
                        // Release the queue lock before the blocking transfer.
                        let job = incoming.lock().recv();
                        let Ok((job, observation)) = job else {
                            return;
                        };
                        let started = std::time::Instant::now();
                        let mut outcome = ReadOutcome::Complete;
                        let host_bytes =
                            job.sources.iter().map(|(_, spans)| spans.host_bytes).sum();
                        for (session, spans) in &job.sources {
                            if spans.is_empty() {
                                continue;
                            }
                            outcome = reader.read_outcome(session, spans);
                            if !outcome.succeeded() {
                                break;
                            }
                        }
                        let disk_bytes = job.push.as_ref().map_or(0, PushJob::bytes);
                        // A push starts only after the memory reads retire. No
                        // new native operation follows an unretired failure.
                        if outcome.retired() {
                            if let Some(push) = job.push {
                                if let Err(error) = pusher.push(
                                    &push.endpoint,
                                    &push.session,
                                    push.budget,
                                    push.items,
                                ) {
                                    tracing::warn!(%error, "Disk push retirement is unproven");
                                    outcome = ReadOutcome::Unretired;
                                }
                            }
                        }
                        if let Some(observation) = observation {
                            observation.finish(
                                started,
                                started.elapsed(),
                                host_bytes,
                                disk_bytes,
                                outcome.succeeded(),
                            );
                        }
                        tracing::debug!(
                            request = job.request_id,
                            ?outcome,
                            elapsed_us = started.elapsed().as_micros() as u64,
                            "peer read ended"
                        );
                        if done.send((job.request_id, outcome)).is_err() {
                            return;
                        }
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

    pub(super) fn submit_observed(
        &self,
        job: ReadJob,
        observation: Option<ts_metrics::QueuedPeerRead>,
    ) {
        if let Some(jobs) = &self.jobs {
            let _ = jobs.send((job, observation));
        }
    }

    /// The reads that ended since the last poll, by landing request.
    pub fn poll(&self) -> Vec<(String, ReadOutcome)> {
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
    fn read_outcome(&self, session: &str, spans: &Spans) -> ReadOutcome {
        // Synchronous CPU copies leave no native work on either result.
        if self.read(session, spans).is_ok() {
            ReadOutcome::Complete
        } else {
            ReadOutcome::FailedRetired
        }
    }

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

    fn wait_for(reads: &Reads) -> Vec<(String, ReadOutcome)> {
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
            sources: vec![(
                "s".into(),
                Spans {
                    sources: vec![4],
                    destinations: vec![32],
                    lengths: vec![8],
                    host_bytes: 0,
                },
            )],
            push: None,
        });
        assert_eq!(
            wait_for(&reads),
            vec![("r".to_owned(), ReadOutcome::Complete)]
        );
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
            sources: vec![(
                "s".into(),
                Spans {
                    sources: vec![4],
                    destinations: vec![32],
                    lengths: vec![8],
                    host_bytes: 0,
                },
            )],
            push: Some(push()),
        });
        assert_eq!(
            wait_for(&reads),
            vec![("r".to_owned(), ReadOutcome::Complete)]
        );
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
            sources: vec![("s".into(), Spans::default())],
            push: Some(push()),
        });
        assert_eq!(
            wait_for(&reads),
            vec![("r".to_owned(), ReadOutcome::Unretired)]
        );
    }

    #[test]
    fn a_failed_read_reports_its_failure() {
        for (reader, outcome) in [
            (
                Arc::new(Failing) as Arc<dyn PageReader>,
                ReadOutcome::Unretired,
            ),
            (Arc::new(Arena::new(4)), ReadOutcome::FailedRetired),
        ] {
            let reads =
                Reads::start(reader, Arc::new(NoPushes), NonZeroUsize::new(2).unwrap()).unwrap();
            reads.submit(ReadJob {
                request_id: "r".into(),
                sources: vec![(
                    "s".into(),
                    Spans {
                        sources: vec![1],
                        destinations: vec![2],
                        lengths: vec![8],
                        host_bytes: 0,
                    },
                )],
                push: None,
            });
            assert_eq!(wait_for(&reads), vec![("r".to_owned(), outcome)]);
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
            sources: vec![(
                session.into(),
                Spans {
                    sources: vec![0],
                    destinations: vec![destination],
                    lengths: vec![8],
                    host_bytes: 0,
                },
            )],
            push: None,
        }
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
        assert_eq!(
            wait_for(&reads),
            vec![("fast".to_owned(), ReadOutcome::Complete)]
        );
        assert_eq!(arena.read_bytes(32, 8), vec![7; 8]);
        assert_eq!(arena.read_bytes(16, 8), vec![0; 8]);
        drop(release);
        assert_eq!(
            wait_for(&reads),
            vec![("slow".to_owned(), ReadOutcome::Complete)]
        );
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
