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

//! The executor: validates jobs, splits them into requests and runs them on
//! the chosen backend.

use std::{
    io,
    os::fd::{FromRawFd, OwnedFd, RawFd},
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use crate::{
    dispatch::{self, Chunk, Limits, Shared},
    file::{DiskFile, ALIGNMENT},
    psync, uring,
};

/// The largest fixed buffer io_uring registers; memory is registered in
/// pieces of at most this size, and no request crosses a piece.
pub(crate) const PIECE_BYTES: usize = 1 << 30;

/// How jobs reach the device. Chosen explicitly: io_uring needs a kernel and
/// a seccomp policy that allow it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// io_uring with the file and the memory registered.
    IoUring,
    /// A pool of threads calling `pread`/`pwrite`.
    Psync,
}

/// Which way a job moves bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// File to memory.
    Read,
    /// Memory to file.
    Write,
}

/// One range of a job: `len` bytes between `file_offset` in the file and
/// `address` in registered memory, each a multiple of [`ALIGNMENT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Io {
    /// Where in the file.
    pub file_offset: u64,
    /// Where in memory.
    pub address: usize,
    /// How many bytes.
    pub len: usize,
}

/// A job: ranges moved one way, reported together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    /// The way bytes move.
    pub direction: Direction,
    /// The ranges.
    pub ios: Vec<Io>,
}

/// How a job ended, and how long it took.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobReport {
    /// Whether every range moved.
    pub outcome: Result<(), IoError>,
    /// From the submission until its first request went to the kernel:
    /// the time it waited behind other jobs, reads before writes, and the
    /// write budget. None if no request of it was issued.
    pub queued: Option<Duration>,
    /// From the submission until the report.
    pub elapsed: Duration,
}

/// A submitted job, to cancel it by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub(crate) u64);

/// Why a job did not complete.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IoError {
    /// A request failed with this errno.
    #[error("disk I/O failed: {}", io::Error::from_raw_os_error(*.0))]
    Os(i32),
    /// A request moved fewer bytes than asked.
    #[error("a short disk I/O: {done} of {expected} bytes")]
    Short {
        /// Bytes asked.
        expected: usize,
        /// Bytes moved.
        done: usize,
    },
    /// The job was cancelled, or the executor closed, before all of it was
    /// issued. What was issued retired.
    #[error("the disk job was cancelled")]
    Cancelled,
}

/// The queues' state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pressure {
    /// Read bytes waiting to be issued.
    pub queued_read_bytes: u64,
    /// Write bytes waiting for the budget or to be issued.
    pub queued_write_bytes: u64,
    /// Bytes the kernel has.
    pub inflight_bytes: u64,
    /// Jobs not yet reported.
    pub jobs: usize,
    /// The age of the oldest of them.
    pub oldest_job: Duration,
}

/// What the executor did since it started.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Reads completed.
    pub read_jobs: u64,
    /// Their bytes.
    pub read_bytes: u64,
    /// Writes completed.
    pub write_jobs: u64,
    /// Their bytes.
    pub write_bytes: u64,
    /// Jobs that failed.
    pub failed_jobs: u64,
    /// Jobs cancelled.
    pub cancelled_jobs: u64,
}

/// How the executor runs. Every field is explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutorConfig {
    /// The backend.
    pub backend: Backend,
    /// Requests in flight at once; with `Psync`, the number of threads.
    pub queue_depth: usize,
    /// Of those, at most this many writes.
    pub write_queue_depth: usize,
    /// Ranges longer than this are issued as several requests.
    pub max_io_bytes: usize,
    /// The bytes per second writes may reach the file at.
    pub write_bytes_per_second: u64,
}

/// A configuration, file or memory the executor cannot run with.
#[derive(Debug, thiserror::Error)]
pub enum StartError {
    /// A setting out of range.
    #[error("invalid disk executor setting: {0}")]
    Config(&'static str),
    /// Memory that is not aligned, empty or overlapping.
    #[error("invalid registered memory: {0}")]
    Memory(&'static str),
    /// The ring did not set up.
    #[error("setting up io_uring: {0}")]
    Ring(io::Error),
    /// A thread did not start.
    #[error("starting a disk thread: {0}")]
    Thread(io::Error),
}

/// A job the executor refuses.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SubmitError {
    /// No ranges.
    #[error("a disk job names no ranges")]
    Empty,
    /// A range not aligned to [`ALIGNMENT`], or empty.
    #[error("disk range {0:?} is empty or not aligned to {ALIGNMENT}")]
    Misaligned(Io),
    /// A range past the file.
    #[error("disk range {0:?} is past the file")]
    PastFile(Io),
    /// A range outside the registered memory.
    #[error("disk range {0:?} is outside the registered memory")]
    Unregistered(Io),
    /// The executor is closing or broke down.
    #[error("the disk executor is closed")]
    Closed,
}

/// The executor did not stop in time: some request never retired, so the
/// memory it names may still be written.
#[derive(Debug, thiserror::Error)]
#[error("disk I/O did not retire before the deadline; its memory must stay mapped")]
pub struct CloseError;

/// The registered memory, cut into fixed-buffer pieces, by address.
#[derive(Debug, Clone)]
pub(crate) struct Pieces {
    /// (start, end) of each piece, sorted and disjoint.
    pieces: Vec<(usize, usize)>,
}

impl Pieces {
    fn new(memory: &[(usize, usize)]) -> Result<Self, StartError> {
        let mut regions = memory.to_vec();
        regions.sort_unstable();
        let mut pieces = Vec::new();
        let mut previous_end = 0;
        for (address, size) in regions {
            if size == 0 || address % ALIGNMENT != 0 || size % ALIGNMENT != 0 {
                return Err(StartError::Memory(
                    "every region must be non-empty and aligned",
                ));
            }
            let end = address
                .checked_add(size)
                .ok_or(StartError::Memory("a region wraps the address space"))?;
            if address < previous_end {
                return Err(StartError::Memory("regions overlap"));
            }
            previous_end = end;
            let mut start = address;
            while start < end {
                let piece_end = start + (end - start).min(PIECE_BYTES);
                pieces.push((start, piece_end));
                start = piece_end;
            }
        }
        if pieces.len() > usize::from(u16::MAX) {
            return Err(StartError::Memory("too many fixed-buffer pieces"));
        }
        Ok(Self { pieces })
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.pieces.iter().copied()
    }

    #[cfg(test)]
    pub(crate) fn from_raw_for_test(pieces: Vec<(usize, usize)>) -> Self {
        Self { pieces }
    }

    /// The piece holding `address`: its index and end.
    fn find(&self, address: usize) -> Option<(usize, usize)> {
        let index = self
            .pieces
            .partition_point(|&(start, _)| start <= address)
            .checked_sub(1)?;
        let (_, end) = self.pieces[index];
        (address < end).then_some((index, end))
    }

    /// Cut `io` into requests of at most `max_io_bytes` that each stay in
    /// one piece; None when part of it is not registered.
    fn split(&self, io: Io, direction: Direction, max_io_bytes: usize) -> Option<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let (mut file_offset, mut address, mut remaining) = (io.file_offset, io.address, io.len);
        while remaining > 0 {
            let (index, end) = self.find(address)?;
            let len = remaining.min(end - address).min(max_io_bytes);
            chunks.push(Chunk {
                job: 0,
                direction,
                file_offset,
                address,
                len,
                buffer: u16::try_from(index).expect("pieces fit u16"),
            });
            file_offset += len as u64;
            address += len;
            remaining -= len;
        }
        Some(chunks)
    }
}

/// Runs disk jobs on one file and the memory it was started with.
pub struct Executor {
    shared: Arc<Shared>,
    pieces: Pieces,
    file_size: u64,
    file_fd: RawFd,
    max_io_bytes: usize,
    /// The threads and their exit notices; `close` takes them once. The
    /// executor is shared (a Disk tier and its peer pushes submit to it), so
    /// closing takes `&self`.
    lifecycle: Mutex<Lifecycle>,
}

struct Lifecycle {
    threads: Vec<thread::JoinHandle<()>>,
    exited: mpsc::Receiver<()>,
    closed: bool,
}

impl Executor {
    /// Start running jobs between `file` and `memory`, a list of `(address,
    /// size)` regions. The regions must stay mapped until [`Executor::close`]
    /// returns `Ok`, or for the process if it fails; jobs may write any of
    /// them.
    pub fn start(
        file: DiskFile,
        memory: &[(usize, usize)],
        config: ExecutorConfig,
    ) -> Result<Self, StartError> {
        Self::start_inner(file, memory, config, false)
    }

    /// Like [`Executor::start`], but forces staging every `O_DIRECT` transfer
    /// through the anonymous mmap bounce pool (identical to the Ascend
    /// `VM_PFNMAP`/`VM_IO` `EFAULT` fallback path).
    pub fn start_bounced(
        file: DiskFile,
        memory: &[(usize, usize)],
        config: ExecutorConfig,
    ) -> Result<Self, StartError> {
        Self::start_inner(file, memory, config, true)
    }

    fn start_inner(
        file: DiskFile,
        memory: &[(usize, usize)],
        config: ExecutorConfig,
        force_bounce: bool,
    ) -> Result<Self, StartError> {
        if config.queue_depth == 0 || config.queue_depth > 4096 {
            return Err(StartError::Config("queue_depth must be in [1, 4096]"));
        }
        if config.write_queue_depth == 0 || config.write_queue_depth > config.queue_depth {
            return Err(StartError::Config(
                "write_queue_depth must be in [1, queue_depth]",
            ));
        }
        if config.max_io_bytes == 0
            || config.max_io_bytes % ALIGNMENT != 0
            || config.max_io_bytes > PIECE_BYTES
        {
            return Err(StartError::Config(
                "max_io_bytes must be a positive multiple of 4096 up to 1 GiB",
            ));
        }
        if config.write_bytes_per_second == 0 {
            return Err(StartError::Config(
                "write_bytes_per_second must be positive",
            ));
        }
        let pieces = Pieces::new(memory)?;
        let file_fd = file.raw_fd();
        let file = Arc::new(file);
        let limits = Limits {
            depth: config.queue_depth,
            write_depth: config.write_queue_depth,
        };
        let eventfd = match config.backend {
            Backend::IoUring => Some(new_eventfd().map_err(StartError::Ring)?),
            Backend::Psync => None,
        };
        let shared = Arc::new(Shared::new(limits, eventfd));
        let (exit_tx, exited) = mpsc::channel();
        let executor = Self {
            shared: shared.clone(),
            pieces: pieces.clone(),
            file_size: file.size(),
            file_fd,
            max_io_bytes: config.max_io_bytes,
            lifecycle: Mutex::new(Lifecycle {
                threads: Vec::new(),
                exited,
                closed: false,
            }),
        };
        // From here on, a failed start closes what already runs.
        let spawn = |name: String, body: Box<dyn FnOnce() + Send>| {
            let notice = ExitNotice(exit_tx.clone());
            thread::Builder::new()
                .name(name)
                .spawn(move || {
                    let _notice = notice;
                    body();
                })
                .map_err(StartError::Thread)
        };
        match config.backend {
            Backend::IoUring => {
                let ring = uring::Ring::new_with_mode(
                    &file,
                    &pieces,
                    config.queue_depth,
                    config.max_io_bytes,
                    force_bounce,
                )
                .map_err(StartError::Ring)?;
                if ring.is_bounce() {
                    shared.set_bounce_active();
                }
                let (shared, file) = (shared.clone(), file.clone());
                executor.lifecycle.lock().threads.push(spawn(
                    "ts-disk-uring".to_owned(),
                    Box::new(move || {
                        ring.run(&shared);
                        // The file stays open while the ring has it.
                        drop(file);
                    }),
                )?);
            }
            Backend::Psync => {
                if force_bounce {
                    shared.set_bounce_active();
                }
                let max_io_bytes = config.max_io_bytes;
                for index in 0..config.queue_depth {
                    let (shared, file) = (shared.clone(), file.clone());
                    executor.lifecycle.lock().threads.push(spawn(
                        format!("ts-disk-io-{index}"),
                        Box::new(move || psync::run(&shared, &file, max_io_bytes, force_bounce)),
                    )?);
                }
            }
        }
        let rate = config.write_bytes_per_second;
        let max_io_bytes = config.max_io_bytes;
        executor.lifecycle.lock().threads.push(spawn(
            "ts-disk-pacer".to_owned(),
            Box::new(move || dispatch::pace(&shared, rate, max_io_bytes)),
        )?);
        Ok(executor)
    }

    /// Queue `job`; `done` runs once with its report, after every request
    /// it issued retired, on an executor thread, so it must be quick.
    pub fn submit(
        &self,
        job: Job,
        done: impl FnOnce(JobReport) + Send + 'static,
    ) -> Result<JobId, SubmitError> {
        if job.ios.is_empty() {
            return Err(SubmitError::Empty);
        }
        let mut chunks = Vec::with_capacity(job.ios.len());
        for &io in &job.ios {
            if io.len == 0
                || io.file_offset % ALIGNMENT as u64 != 0
                || io.address % ALIGNMENT != 0
                || io.len % ALIGNMENT != 0
            {
                return Err(SubmitError::Misaligned(io));
            }
            if io
                .file_offset
                .checked_add(io.len as u64)
                .is_none_or(|end| end > self.file_size)
            {
                return Err(SubmitError::PastFile(io));
            }
            let split = self
                .pieces
                .split(io, job.direction, self.max_io_bytes)
                .ok_or(SubmitError::Unregistered(io))?;
            chunks.extend(split);
        }
        let mut queues = self.shared.queues.lock();
        if queues.closing() {
            return Err(SubmitError::Closed);
        }
        let id = queues.admit(job.direction, chunks, Box::new(done), Instant::now());
        drop(queues);
        self.shared.wake();
        Ok(JobId(id))
    }

    /// Issue no more of `job`. It still reports, with
    /// [`IoError::Cancelled`] unless a request failed, once what it issued
    /// retired. A job that already reported is ignored.
    pub fn cancel(&self, job: JobId) {
        let finished = self.shared.queues.lock().cancel(job.0);
        self.shared.wake();
        dispatch::report(finished.into_iter().collect());
    }

    /// The queues now.
    pub fn pressure(&self) -> Pressure {
        self.shared.queues.lock().pressure(Instant::now())
    }

    /// What the executor did so far.
    pub fn stats(&self) -> Stats {
        self.shared.queues.lock().stats()
    }

    /// Whether the backend broke down: it can no longer reap what it issued,
    /// so those jobs never report and their memory stays held.
    pub fn broken(&self) -> bool {
        self.shared.queues.lock().broken()
    }

    /// Whether the backend is currently staging `O_DIRECT` I/O through an
    /// anonymous `mmap` bounce pool.
    pub fn uses_bounce_pool(&self) -> bool {
        self.shared.is_bounce_active()
    }

    /// Backing file descriptor owned by this executor.
    pub fn raw_fd(&self) -> RawFd {
        self.file_fd
    }

    /// Cancel what is queued, wait up to `timeout` for what is in flight to
    /// retire, and stop the threads. On failure the threads are left
    /// running and the memory must stay mapped.
    pub fn close(&self, timeout: Duration) -> Result<(), CloseError> {
        let mut lifecycle = self.lifecycle.lock();
        if lifecycle.closed {
            return Ok(());
        }
        let finished = self.shared.queues.lock().close();
        self.shared.wake();
        dispatch::report(finished);
        let deadline = Instant::now() + timeout;
        lifecycle.closed = true;
        for _ in 0..lifecycle.threads.len() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if lifecycle.exited.recv_timeout(remaining).is_err() {
                // Leave the threads: they hold the file and finish when the
                // kernel retires their requests.
                lifecycle.threads.clear();
                return Err(CloseError);
            }
        }
        for thread in lifecycle.threads.drain(..) {
            let _ = thread.join();
        }
        Ok(())
    }
}

/// Tells `close` a thread ended, however it ended.
struct ExitNotice(mpsc::Sender<()>);

impl Drop for ExitNotice {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        if self.close(Duration::from_secs(60)).is_err() {
            tracing::error!("disk I/O did not retire; leaving its threads and memory");
        }
    }
}

fn new_eventfd() -> io::Result<OwnedFd> {
    // SAFETY: eventfd returns a fresh descriptor or -1.
    let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `fd` is a descriptor this function just created.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_cut_regions_and_requests_stay_inside_one() {
        let base = 1 << 40;
        let pieces = Pieces::new(&[(base, PIECE_BYTES + 2 * ALIGNMENT)]).unwrap();
        assert_eq!(pieces.iter().count(), 2);
        let io = Io {
            file_offset: 0,
            address: base + PIECE_BYTES - ALIGNMENT,
            len: 3 * ALIGNMENT,
        };
        let chunks = pieces
            .split(io, Direction::Read, 16 * ALIGNMENT)
            .expect("registered");
        let cut: Vec<_> = chunks
            .iter()
            .map(|chunk| (chunk.buffer, chunk.file_offset, chunk.len))
            .collect();
        assert_eq!(
            cut,
            vec![(0, 0, ALIGNMENT), (1, ALIGNMENT as u64, 2 * ALIGNMENT)]
        );
        // Past the region.
        let outside = Io {
            file_offset: 0,
            address: base + PIECE_BYTES + 2 * ALIGNMENT,
            len: ALIGNMENT,
        };
        assert!(pieces.split(outside, Direction::Read, ALIGNMENT).is_none());
    }

    #[test]
    fn long_requests_are_cut_to_the_largest_io() {
        let pieces = Pieces::new(&[(1 << 30, 8 * ALIGNMENT)]).unwrap();
        let io = Io {
            file_offset: 4 * ALIGNMENT as u64,
            address: 1 << 30,
            len: 5 * ALIGNMENT,
        };
        let lens: Vec<_> = pieces
            .split(io, Direction::Write, 2 * ALIGNMENT)
            .unwrap()
            .iter()
            .map(|chunk| chunk.len / ALIGNMENT)
            .collect();
        assert_eq!(lens, vec![2, 2, 1]);
    }

    #[test]
    fn overlapping_or_unaligned_memory_is_refused() {
        assert!(Pieces::new(&[(4096, 8192), (8192, 4096)]).is_err());
        assert!(Pieces::new(&[(4097, 4096)]).is_err());
        assert!(Pieces::new(&[(4096, 0)]).is_err());
    }
}
