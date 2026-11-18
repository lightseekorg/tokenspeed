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
//! one-sided reads into the member's own Device pages, run off the control
//! thread, each reporting once it retired.

use std::{
    num::NonZeroUsize,
    sync::{mpsc, Arc},
    thread,
};

use parking_lot::Mutex;

use crate::peer::geometry::Spans;

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
/// spans.
pub struct ReadJob {
    /// The landing request.
    pub request_id: String,
    /// The holder rank's transfer session.
    pub session: String,
    /// What to copy.
    pub spans: Spans,
}

/// The member's bounded read workers: jobs in, outcomes out.
pub struct Reads {
    jobs: Option<mpsc::Sender<ReadJob>>,
    outcomes: mpsc::Receiver<(String, bool)>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl Reads {
    /// Run at most `concurrency` reads at once over `reader`. Queued reads
    /// keep their landing's destinations held until their outcome arrives.
    pub fn start(reader: Arc<dyn PageReader>, concurrency: NonZeroUsize) -> std::io::Result<Self> {
        let (jobs, incoming) = mpsc::channel::<ReadJob>();
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
            let (reader, incoming, done) = (reader.clone(), incoming.clone(), done.clone());
            let worker = thread::Builder::new()
                .name(format!("ts-peer-read-{index}"))
                .spawn(move || loop {
                    // Release the queue lock before the blocking transfer.
                    let job = incoming.lock().recv();
                    let Ok(job) = job else {
                        return;
                    };
                    let outcome = if job.spans.is_empty() {
                        Ok(())
                    } else {
                        reader.read(&job.session, &job.spans)
                    };
                    if let Err(error) = &outcome {
                        let spans = &job.spans;
                        tracing::warn!(
                            request = job.request_id,
                            %error,
                            spans = spans.lengths.len(),
                            first_source = spans.sources.first().map(|a| format!("{a:#x}")),
                            first_destination = spans.destinations.first().map(|a| format!("{a:#x}")),
                            last_destination = spans.destinations.last().map(|a| format!("{a:#x}")),
                            "peer read failed"
                        );
                    }
                    if done.send((job.request_id, outcome.is_ok())).is_err() {
                        return;
                    }
                })?;
            reads.workers.push(worker);
        }
        Ok(reads)
    }

    /// Queue a read.
    pub fn submit(&self, job: ReadJob) {
        if let Some(jobs) = &self.jobs {
            // The workers outlive the sender; a send fails only if they ended,
            // which the next poll does not hide: nothing reports then.
            let _ = jobs.send(job);
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
        let reads = Reads::start(Arc::new(arena.clone()), NonZeroUsize::new(2).unwrap()).unwrap();
        reads.submit(ReadJob {
            request_id: "r".into(),
            session: "s".into(),
            spans: Spans {
                sources: vec![4],
                destinations: vec![32],
                lengths: vec![8],
            },
        });
        assert_eq!(wait_for(&reads), vec![("r".to_owned(), true)]);
        assert_eq!(arena.read_bytes(32, 8), vec![7; 8]);
        assert_eq!(arena.read_bytes(40, 8), vec![0; 8]);
    }

    #[test]
    fn a_failed_read_reports_its_failure() {
        for reader in [
            Arc::new(Failing) as Arc<dyn PageReader>,
            Arc::new(Arena::new(4)),
        ] {
            let reads = Reads::start(reader, NonZeroUsize::new(2).unwrap()).unwrap();
            reads.submit(ReadJob {
                request_id: "r".into(),
                session: "s".into(),
                spans: Spans {
                    sources: vec![1],
                    destinations: vec![2],
                    lengths: vec![8],
                },
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
            },
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
        let reads = Reads::start(reader, NonZeroUsize::new(2).unwrap()).unwrap();
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
        let reads = Reads::start(reader, NonZeroUsize::new(2).unwrap()).unwrap();
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
