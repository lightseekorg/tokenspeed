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

//! What the backends share: the jobs in flight, their chunks waiting to be
//! issued, and the pacing of writes.
//!
//! One lock covers it all. A chunk is one request to the kernel; a job
//! reports once its last chunk retired. Reads are issued ahead of writes;
//! writes first wait in a backlog until the byte budget lets them through,
//! then take at most `write_depth` of the `depth` requests in flight.

use std::{
    collections::{BTreeMap, VecDeque},
    os::fd::{AsRawFd, OwnedFd},
    time::{Duration, Instant},
};

use parking_lot::{Condvar, Mutex};

use crate::executor::{Direction, IoError, JobReport, Pressure, Stats};

/// What a job runs once it ended: its report.
pub(crate) type Done = Box<dyn FnOnce(JobReport) + Send>;

/// Jobs that ended under the lock, reported after it is released.
pub(crate) type Finished = Vec<(Done, JobReport)>;

pub(crate) fn report(finished: Finished) {
    for (done, report) in finished {
        done(report);
    }
}

/// One request to the kernel: `len` bytes between `file_offset` and
/// `address`, all inside the fixed buffer `buffer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Chunk {
    pub job: u64,
    pub direction: Direction,
    pub file_offset: u64,
    pub address: usize,
    pub len: usize,
    pub buffer: u16,
}

struct JobEntry {
    direction: Direction,
    bytes: u64,
    submitted_at: Instant,
    /// When its first chunk went to the kernel.
    issued_at: Option<Instant>,
    queued: usize,
    inflight: usize,
    error: Option<IoError>,
    cancelled: bool,
    done: Option<Done>,
}

impl JobEntry {
    /// The job's outcome once nothing of it is queued or in flight.
    fn outcome(&mut self) -> Result<(), IoError> {
        match self.error.take() {
            Some(error) => Err(error),
            None if self.cancelled => Err(IoError::Cancelled),
            None => Ok(()),
        }
    }
}

/// The requests in flight at once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    pub depth: usize,
    pub write_depth: usize,
}

#[derive(Default)]
pub(crate) struct Queues {
    jobs: BTreeMap<u64, JobEntry>,
    reads: VecDeque<Chunk>,
    writes: VecDeque<Chunk>,
    backlog: VecDeque<Chunk>,
    inflight_reads: usize,
    inflight_writes: usize,
    inflight_bytes: u64,
    next_job: u64,
    closing: bool,
    broken: bool,
    stats: Stats,
}

impl Queues {
    /// Queue a job's chunks; returns its id.
    pub(crate) fn admit(
        &mut self,
        direction: Direction,
        chunks: impl IntoIterator<Item = Chunk>,
        done: Done,
        now: Instant,
    ) -> u64 {
        self.next_job += 1;
        let job = self.next_job;
        let mut queued = 0;
        let mut bytes = 0;
        for chunk in chunks {
            let chunk = Chunk { job, ..chunk };
            queued += 1;
            bytes += chunk.len as u64;
            match direction {
                Direction::Read => self.reads.push_back(chunk),
                Direction::Write => self.backlog.push_back(chunk),
            }
        }
        self.jobs.insert(
            job,
            JobEntry {
                direction,
                bytes,
                submitted_at: now,
                issued_at: None,
                queued,
                inflight: 0,
                error: None,
                cancelled: false,
                done: Some(done),
            },
        );
        job
    }

    /// The next chunk to issue within `limits`, now counted in flight.
    pub(crate) fn next(&mut self, limits: Limits) -> Option<Chunk> {
        if self.broken || self.inflight_reads + self.inflight_writes >= limits.depth {
            return None;
        }
        let chunk = match self.reads.pop_front() {
            Some(chunk) => chunk,
            None if self.inflight_writes < limits.write_depth => self.writes.pop_front()?,
            None => return None,
        };
        match chunk.direction {
            Direction::Read => self.inflight_reads += 1,
            Direction::Write => self.inflight_writes += 1,
        }
        self.inflight_bytes += chunk.len as u64;
        let entry = self.jobs.get_mut(&chunk.job).expect("a queued chunk's job");
        entry.queued -= 1;
        entry.inflight += 1;
        entry.issued_at.get_or_insert_with(Instant::now);
        Some(chunk)
    }

    /// A chunk retired with `result`; returns its job if that ended.
    pub(crate) fn finish(
        &mut self,
        chunk: Chunk,
        result: Result<(), IoError>,
    ) -> Option<(Done, JobReport)> {
        match chunk.direction {
            Direction::Read => self.inflight_reads -= 1,
            Direction::Write => self.inflight_writes -= 1,
        }
        self.inflight_bytes -= chunk.len as u64;
        let entry = self
            .jobs
            .get_mut(&chunk.job)
            .expect("an issued chunk's job");
        entry.inflight -= 1;
        if let Err(error) = result {
            entry.error.get_or_insert(error);
            // Its other chunks are not issued: the job failed already.
            if entry.queued > 0 {
                entry.queued = 0;
                self.drop_queued(chunk.job);
            }
        }
        self.end_if_retired(chunk.job)
    }

    /// Stop issuing `job`'s chunks; returns it if nothing of it is in flight.
    pub(crate) fn cancel(&mut self, job: u64) -> Option<(Done, JobReport)> {
        let entry = self.jobs.get_mut(&job)?;
        entry.cancelled = true;
        if entry.queued > 0 {
            entry.queued = 0;
            self.drop_queued(job);
        }
        self.end_if_retired(job)
    }

    /// Refuse new jobs and cancel every queued chunk; what is in flight
    /// still retires.
    pub(crate) fn close(&mut self) -> Finished {
        self.closing = true;
        let jobs: Vec<u64> = self.jobs.keys().copied().collect();
        jobs.into_iter()
            .filter_map(|job| self.cancel(job))
            .collect()
    }

    /// The backend can no longer reap what it issued: issue nothing more.
    /// The jobs in flight never report, so whatever they name stays held.
    pub(crate) fn break_down(&mut self) -> Finished {
        self.broken = true;
        self.close()
    }

    pub(crate) fn closing(&self) -> bool {
        self.closing
    }

    pub(crate) fn broken(&self) -> bool {
        self.broken
    }

    /// No chunk waits to be issued.
    pub(crate) fn drained(&self) -> bool {
        self.reads.is_empty() && self.writes.is_empty() && self.backlog.is_empty()
    }

    /// The size of the next write waiting for the byte budget.
    pub(crate) fn backlog_front(&self) -> Option<usize> {
        self.backlog.front().map(|chunk| chunk.len)
    }

    /// Let the next write through the budget.
    pub(crate) fn release_write(&mut self) {
        if let Some(chunk) = self.backlog.pop_front() {
            self.writes.push_back(chunk);
        }
    }

    pub(crate) fn pressure(&self, now: Instant) -> Pressure {
        let bytes = |queue: &VecDeque<Chunk>| queue.iter().map(|chunk| chunk.len as u64).sum();
        Pressure {
            queued_read_bytes: bytes(&self.reads),
            queued_write_bytes: bytes(&self.writes) + bytes(&self.backlog),
            inflight_bytes: self.inflight_bytes,
            jobs: self.jobs.len(),
            oldest_job: self
                .jobs
                .values()
                .next()
                .map_or(Duration::ZERO, |entry| now - entry.submitted_at),
        }
    }

    pub(crate) fn stats(&self) -> Stats {
        self.stats
    }

    fn drop_queued(&mut self, job: u64) {
        for queue in [&mut self.reads, &mut self.writes, &mut self.backlog] {
            queue.retain(|chunk| chunk.job != job);
        }
    }

    fn end_if_retired(&mut self, job: u64) -> Option<(Done, JobReport)> {
        let entry = self.jobs.get(&job)?;
        if entry.queued > 0 || entry.inflight > 0 {
            return None;
        }
        let mut entry = self.jobs.remove(&job).expect("the job just found");
        let outcome = entry.outcome();
        match &outcome {
            Ok(()) => match entry.direction {
                Direction::Read => {
                    self.stats.read_jobs += 1;
                    self.stats.read_bytes += entry.bytes;
                }
                Direction::Write => {
                    self.stats.write_jobs += 1;
                    self.stats.write_bytes += entry.bytes;
                }
            },
            Err(IoError::Cancelled) => self.stats.cancelled_jobs += 1,
            Err(_) => self.stats.failed_jobs += 1,
        }
        let report = JobReport {
            outcome,
            queued: entry
                .issued_at
                .map(|issued| issued.duration_since(entry.submitted_at)),
            elapsed: entry.submitted_at.elapsed(),
        };
        Some((entry.done.take().expect("a job reports once"), report))
    }
}

/// What the executor, the pacer and the backend threads share.
pub(crate) struct Shared {
    pub(crate) queues: Mutex<Queues>,
    /// Signalled whenever work arrives or a request retires.
    pub(crate) work: Condvar,
    pub(crate) limits: Limits,
    /// The io_uring thread's wake-up: it waits in the kernel, not on `work`.
    pub(crate) eventfd: Option<OwnedFd>,
}

impl Shared {
    pub(crate) fn new(limits: Limits, eventfd: Option<OwnedFd>) -> Self {
        Self {
            queues: Mutex::new(Queues::default()),
            work: Condvar::new(),
            limits,
            eventfd,
        }
    }

    /// Tell every thread the queues changed.
    pub(crate) fn wake(&self) {
        self.work.notify_all();
        if let Some(eventfd) = &self.eventfd {
            let one = 1u64;
            // SAFETY: writes eight bytes from a local into an eventfd this
            // value owns. A failure only means the counter is saturated, so
            // a wake-up is pending anyway.
            unsafe {
                libc::write(
                    eventfd.as_raw_fd(),
                    (&one as *const u64).cast(),
                    std::mem::size_of::<u64>(),
                );
            }
        }
    }
}

/// Bytes per second, with a burst of at least one chunk.
struct TokenBucket {
    rate: f64,
    burst: f64,
    tokens: f64,
    last: Instant,
}

impl TokenBucket {
    fn new(rate: u64, max_chunk: usize, now: Instant) -> Self {
        // A twentieth of a second of budget smooths the stream without
        // letting a backlog burst past the rate.
        let burst = (rate as f64 / 20.0).max(max_chunk as f64);
        Self {
            rate: rate as f64,
            burst,
            tokens: burst,
            last: now,
        }
    }

    /// Spend `bytes`, or say how long until they are available.
    fn take(&mut self, bytes: usize, now: Instant) -> Result<(), Duration> {
        let elapsed = now.saturating_duration_since(self.last).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.rate).min(self.burst);
        self.last = now;
        let bytes = bytes as f64;
        if self.tokens >= bytes {
            self.tokens -= bytes;
            return Ok(());
        }
        Err(Duration::from_secs_f64((bytes - self.tokens) / self.rate))
    }
}

/// The pacer thread: moves writes from the backlog to the issue queue at
/// `rate` bytes per second until the queues close.
pub(crate) fn pace(shared: &Shared, rate: u64, max_chunk: usize) {
    let mut bucket = TokenBucket::new(rate, max_chunk, Instant::now());
    let mut queues = shared.queues.lock();
    loop {
        if queues.closing() {
            return;
        }
        let Some(bytes) = queues.backlog_front() else {
            shared.work.wait(&mut queues);
            continue;
        };
        match bucket.take(bytes, Instant::now()) {
            Ok(()) => {
                queues.release_write();
                drop(queues);
                shared.wake();
                queues = shared.queues.lock();
            }
            Err(wait) => {
                shared.work.wait_for(&mut queues, wait);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(direction: Direction, len: usize) -> Chunk {
        Chunk {
            job: 0,
            direction,
            file_offset: 0,
            address: 0,
            len,
            buffer: 0,
        }
    }

    fn admit_chunks(
        queues: &mut Queues,
        direction: Direction,
        chunks: impl IntoIterator<Item = Chunk>,
        done: Done,
        now: Instant,
    ) -> u64 {
        queues.admit(direction, chunks, done, now)
    }

    type Outcomes = std::sync::Arc<Mutex<Vec<Result<(), IoError>>>>;

    fn recorder() -> (Done, Outcomes) {
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        (
            Box::new(move |report: JobReport| sink.lock().push(report.outcome)),
            seen,
        )
    }

    const LIMITS: Limits = Limits {
        depth: 3,
        write_depth: 1,
    };

    #[test]
    fn reads_go_first_and_writes_keep_to_their_share() {
        let mut queues = Queues::default();
        let now = Instant::now();
        let (done, _) = recorder();
        admit_chunks(
            &mut queues,
            Direction::Write,
            [chunk(Direction::Write, 1), chunk(Direction::Write, 1)],
            done,
            now,
        );
        queues.release_write();
        queues.release_write();
        let (done, _) = recorder();
        admit_chunks(
            &mut queues,
            Direction::Read,
            [chunk(Direction::Read, 1)],
            done,
            now,
        );
        assert_eq!(queues.next(LIMITS).unwrap().direction, Direction::Read);
        assert_eq!(queues.next(LIMITS).unwrap().direction, Direction::Write);
        // One write in flight is the share, though a slot is free.
        assert_eq!(queues.next(LIMITS), None);
    }

    #[test]
    fn a_job_reports_once_after_its_last_chunk_and_keeps_the_first_error() {
        let mut queues = Queues::default();
        let (done, seen) = recorder();
        admit_chunks(
            &mut queues,
            Direction::Read,
            [
                chunk(Direction::Read, 1),
                chunk(Direction::Read, 1),
                chunk(Direction::Read, 1),
            ],
            done,
            Instant::now(),
        );
        let first = queues.next(LIMITS).unwrap();
        let second = queues.next(LIMITS).unwrap();
        assert!(queues.finish(first, Err(IoError::Os(libc::EIO))).is_none());
        // The third chunk is no longer issued: the job failed.
        assert_eq!(queues.next(LIMITS), None);
        let (done, report) = queues
            .finish(second, Err(IoError::Os(libc::ENOSPC)))
            .unwrap();
        done(report);
        assert_eq!(*seen.lock(), vec![Err(IoError::Os(libc::EIO))]);
        assert_eq!(queues.stats().failed_jobs, 1);
    }

    #[test]
    fn a_cancelled_job_waits_for_its_chunks_in_flight() {
        let mut queues = Queues::default();
        let (done, _) = recorder();
        let job = admit_chunks(
            &mut queues,
            Direction::Read,
            [chunk(Direction::Read, 1), chunk(Direction::Read, 1)],
            done,
            Instant::now(),
        );
        let issued = queues.next(LIMITS).unwrap();
        assert!(queues.cancel(job).is_none());
        assert!(queues.drained());
        let (_, report) = queues.finish(issued, Ok(())).unwrap();
        assert_eq!(report.outcome, Err(IoError::Cancelled));
        assert!(report.queued.is_some_and(|queued| queued <= report.elapsed));
    }

    #[test]
    fn closing_cancels_what_is_queued() {
        let mut queues = Queues::default();
        let (done, _) = recorder();
        admit_chunks(
            &mut queues,
            Direction::Write,
            [chunk(Direction::Write, 1)],
            done,
            Instant::now(),
        );
        let finished = queues.close();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].1.outcome, Err(IoError::Cancelled));
        // Nothing of it reached the kernel.
        assert_eq!(finished[0].1.queued, None);
        assert!(queues.closing() && queues.drained());
    }

    #[test]
    fn the_bucket_paces_bytes_to_its_rate() {
        let start = Instant::now();
        let mut bucket = TokenBucket::new(1000, 100, start);
        // The burst is one chunk: the first passes, the next waits 0.1 s.
        assert_eq!(bucket.take(100, start), Ok(()));
        let wait = bucket.take(100, start).unwrap_err();
        assert!((wait.as_secs_f64() - 0.1).abs() < 1e-9);
        assert_eq!(bucket.take(100, start + Duration::from_millis(100)), Ok(()));
    }
}
