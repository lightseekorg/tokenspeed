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

//! The io_uring backend: one thread owns a ring with the memory registered
//! as fixed buffers, keeps up to `depth` requests in flight, and waits in
//! the kernel for the next completion. A read of the executor's eventfd
//! stays armed in the ring, so new work wakes the thread like a completion
//! does. The file is not registered and there is no SQPOLL: the kernels and
//! container policies in use refuse fixed files (EBADF) and unprivileged
//! SQPOLL.

use std::{
    collections::VecDeque,
    io,
    os::fd::{AsRawFd, RawFd},
};

use io_uring::{opcode, squeue, types, IoUring};

use crate::{
    dispatch::{self, Chunk, Shared},
    executor::{Direction, IoError, Pieces},
    file::{DiskFile, ALIGNMENT},
    memory::Memory,
};

/// Maximum single-iovec length registered with `io_uring_register_buffers`.
const MAX_FIXED_BOUNCE_BYTES: usize = 1 << 30;

/// The user data of the eventfd read; requests use their slot index.
const WAKE_UP: u64 = u64::MAX;

/// Upper bound on transient `EINTR`/`EAGAIN` retries per chunk so a broken
/// device cannot livelock `Executor::close`.
const MAX_TRANSIENT_RETRIES: u8 = 16;

/// Linux `ioprio` encoding (`class << 13 | level`): foreground reads run at
/// best-effort level 0, background cleaning writes at best-effort level 7
/// (lowest best-effort priority, avoiding idle-class starvation under BFQ /
/// mq-deadline).
const IOPRIO_READ: u16 = 2 << 13;
const IOPRIO_WRITE: u16 = (2 << 13) | 7;

/// One in-flight request in a ring slot, tracking progress across short I/Os.
#[derive(Clone, Copy)]
struct InFlight {
    chunk: Chunk,
    done: usize,
    retries: u8,
}

enum BufferMode {
    /// Caller's registered memory pieces are pinned directly in io_uring fixed buffers.
    FixedCaller,
    /// Caller's memory is VM_PFNMAP/VM_IO (e.g. Ascend `torch_npu` `/dev/davinci_manager`)
    /// which Linux `O_DIRECT` rejects with `EFAULT`. Stage each slot through a page-aligned
    /// anonymous `mmap` bounce pool (registered as a single fixed buffer when possible).
    Bounce {
        pool: Memory,
        slot_size: usize,
        fixed: bool,
    },
}

impl BufferMode {
    fn bounce_slot_addr(&self, slot: usize) -> Option<usize> {
        match self {
            Self::Bounce {
                pool, slot_size, ..
            } => Some(pool.address() + slot * slot_size),
            Self::FixedCaller => None,
        }
    }

    #[cfg(test)]
    fn is_bounce(&self) -> bool {
        matches!(self, Self::Bounce { .. })
    }
}

pub(crate) struct Ring {
    ring: IoUring,
    fd: RawFd,
    depth: usize,
    max_io_bytes: usize,
    mode: BufferMode,
    r_disabled: bool,
}

fn allocate_bounce_mode(
    ring: &IoUring,
    depth: usize,
    max_io_bytes: usize,
) -> io::Result<BufferMode> {
    let total = depth
        .checked_mul(max_io_bytes)
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    let pool = Memory::map(total)?;
    let fixed = if total <= MAX_FIXED_BOUNCE_BYTES {
        let iov = libc::iovec {
            iov_base: pool.address() as *mut libc::c_void,
            iov_len: pool.size(),
        };
        // SAFETY: `pool` stays mapped until after `ring` is dropped (or both are forgotten on breakdown).
        match unsafe { ring.submitter().register_buffers(&[iov]) } {
            Ok(()) => true,
            Err(error)
                if matches!(
                    error.raw_os_error(),
                    Some(libc::ENOMEM | libc::EINVAL | libc::E2BIG | libc::EPERM)
                ) =>
            {
                tracing::warn!(
                    %error,
                    bounce_bytes = total,
                    "io_uring register_buffers failed on bounce pool; using non-fixed O_DIRECT on bounce pool"
                );
                false
            }
            Err(error) => return Err(error),
        }
    } else {
        false
    };
    Ok(BufferMode::Bounce {
        pool,
        slot_size: max_io_bytes,
        fixed,
    })
}

fn build_ring(entries: u32) -> io::Result<(IoUring, bool)> {
    // Linux >= 6.1: bind single-issuer ownership when `Ring::run` activates on
    // its dedicated worker thread, avoiding cross-thread kernel ring locks.
    match IoUring::builder()
        .setup_single_issuer()
        .setup_defer_taskrun()
        .setup_r_disabled()
        .build(entries)
    {
        Ok(ring) => return Ok((ring, true)),
        Err(error) if error.raw_os_error() == Some(libc::EINVAL) => {}
        Err(error) => return Err(error),
    }
    // Linux 5.19..6.0 fallback: cooperative task_work without IPI interrupts.
    match IoUring::builder().setup_coop_taskrun().build(entries) {
        Ok(ring) => return Ok((ring, false)),
        Err(error) if error.raw_os_error() == Some(libc::EINVAL) => {}
        Err(error) => return Err(error),
    }
    Ok((IoUring::new(entries)?, false))
}

impl Ring {
    pub(crate) fn new(
        file: &DiskFile,
        pieces: &Pieces,
        depth: usize,
        max_io_bytes: usize,
    ) -> io::Result<Self> {
        // One entry beyond the requests for the armed wake-up read.
        let entries = u32::try_from((depth + 1).next_power_of_two())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let (ring, r_disabled) = build_ring(entries)?;
        let buffers: Vec<libc::iovec> = pieces
            .iter()
            .map(|(start, end)| libc::iovec {
                iov_base: start as *mut libc::c_void,
                iov_len: end - start,
            })
            .collect();
        // SAFETY: the executor's caller keeps the memory mapped until close
        // succeeds (`Executor::start`), and a failed close leaks this ring.
        // `EFAULT` signals VM_PFNMAP/VM_IO device-pinned memory (e.g. Ascend
        // `torch_npu` `/dev/davinci_manager`), which Linux `O_DIRECT` also
        // rejects on non-fixed Read/Write -> stage through an anonymous mmap
        // bounce pool (RFC 0006). Other registration errors on the caller's
        // buffers (such as ENOMEM from a low RLIMIT_MEMLOCK) propagate
        // explicitly; on the anonymous bounce pool, memlock/limit errors
        // degrade to non-fixed O_DIRECT while unexpected errors like EFAULT
        // propagate.
        let mode = match unsafe { ring.submitter().register_buffers(&buffers) } {
            Ok(()) => BufferMode::FixedCaller,
            Err(error) if error.raw_os_error() == Some(libc::EFAULT) => {
                let bounce = allocate_bounce_mode(&ring, depth, max_io_bytes)?;
                let bounce_fixed = matches!(bounce, BufferMode::Bounce { fixed: true, .. });
                tracing::warn!(
                    %error,
                    pieces = buffers.len(),
                    bounce_bytes = depth * max_io_bytes,
                    bounce_fixed,
                    "io_uring register_buffers failed with EFAULT (device-pinned VM_PFNMAP/VM_IO memory); \
                     staging O_DIRECT through anonymous mmap bounce pool"
                );
                bounce
            }
            Err(error) => return Err(error),
        };
        Ok(Self {
            ring,
            fd: file.raw_fd(),
            depth,
            max_io_bytes,
            mode,
            r_disabled,
        })
    }

    #[cfg(test)]
    fn new_bounce(file: &DiskFile, depth: usize, max_io_bytes: usize) -> io::Result<Self> {
        let entries = u32::try_from((depth + 1).next_power_of_two())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let (ring, r_disabled) = build_ring(entries)?;
        let mode = allocate_bounce_mode(&ring, depth, max_io_bytes)?;
        Ok(Self {
            ring,
            fd: file.raw_fd(),
            depth,
            max_io_bytes,
            mode,
            r_disabled,
        })
    }

    pub(crate) fn run(mut self, shared: &Shared) {
        if self.r_disabled {
            if let Err(error) = self.ring.submitter().register_enable_rings() {
                tracing::error!(%error, "io_uring register_enable_rings failed");
                let finished = shared.queues.lock().break_down();
                shared.wake();
                dispatch::report(finished);
                return;
            }
        }
        let eventfd = shared
            .eventfd
            .as_ref()
            .expect("the io_uring backend wakes through an eventfd")
            .as_raw_fd();
        // The wake-up read's target. Dropped after the ring, whose teardown
        // cancels the read.
        let mut counter = Box::new(0u64);
        let mut slots: Vec<Option<InFlight>> = vec![None; self.depth];
        let mut free: Vec<usize> = (0..self.depth).rev().collect();
        let mut retries: VecDeque<InFlight> = VecDeque::new();
        let mut armed = false;
        let mut to_issue: Vec<InFlight> = Vec::with_capacity(self.depth);
        let mut outcomes: Vec<(Chunk, Result<(), IoError>)> = Vec::with_capacity(self.depth);
        loop {
            let sq_space = {
                let submission = self.ring.submission();
                let avail = submission.capacity().saturating_sub(submission.len());
                if armed {
                    avail
                } else {
                    avail.saturating_sub(1)
                }
            };
            let max_issue = free.len().min(sq_space);
            to_issue.clear();
            {
                let mut queues = shared.queues.lock();
                while to_issue.len() < max_issue {
                    let inflight = match retries.pop_front() {
                        Some(inflight) => inflight,
                        None => match queues.next(shared.limits) {
                            Some(chunk) => InFlight {
                                chunk,
                                done: 0,
                                retries: 0,
                            },
                            None => break,
                        },
                    };
                    to_issue.push(inflight);
                }
                if queues.closing()
                    && queues.drained()
                    && retries.is_empty()
                    && to_issue.is_empty()
                    && free.len() == self.depth
                {
                    break;
                }
            }
            {
                let mut submission = self.ring.submission();
                if !armed && !submission.is_full() {
                    let read = opcode::Read::new(
                        types::Fd(eventfd),
                        (&mut *counter as *mut u64).cast(),
                        8,
                    )
                    .build()
                    .user_data(WAKE_UP);
                    // SAFETY: `counter` outlives the ring.
                    unsafe { submission.push(&read) }.expect("the ring has room for the wake-up");
                    armed = true;
                }
                for inflight in to_issue.drain(..) {
                    let slot = free.pop().expect("free slot reserved for to_issue");
                    if let Some(bounce_addr) = self.mode.bounce_slot_addr(slot) {
                        let remaining = inflight.chunk.len - inflight.done;
                        debug_assert!(remaining <= self.max_io_bytes);
                        if inflight.chunk.direction == Direction::Write {
                            // SAFETY: source is caller's registered memory (stays mapped
                            // until close) and destination is this slot's owned bounce buffer.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    (inflight.chunk.address + inflight.done) as *const u8,
                                    bounce_addr as *mut u8,
                                    remaining,
                                );
                            }
                        }
                    }
                    let req = request(self.fd, inflight, slot, &self.mode).user_data(slot as u64);
                    // SAFETY: the chunk (or bounce slot) stays mapped; its slot
                    // keeps it until it completes.
                    unsafe { submission.push(&req) }.expect("the ring has an entry for every slot");
                    slots[slot] = Some(inflight);
                }
            }
            if let Err(error) = self.ring.submit_and_wait(1) {
                // Interrupted, or the completion queue is full: reap first.
                if !matches!(error.raw_os_error(), Some(libc::EINTR | libc::EBUSY)) {
                    tracing::error!(%error, "io_uring broke down; requests it holds never report");
                    let finished = shared.queues.lock().break_down();
                    shared.wake();
                    dispatch::report(finished);
                    // The kernel may still write into the memory, bounce pool,
                    // and counter: keep them, and the ring, for the process.
                    std::mem::forget(self.ring);
                    std::mem::forget(self.mode);
                    std::mem::forget(counter);
                    return;
                }
            }
            let completions: Vec<(u64, i32)> = self
                .ring
                .completion()
                .map(|completion| (completion.user_data(), completion.result()))
                .collect();
            outcomes.clear();
            for (user_data, result) in completions {
                if user_data == WAKE_UP {
                    armed = false;
                    continue;
                }
                let slot = usize::try_from(user_data).expect("a slot index");
                let mut inflight = slots[slot]
                    .take()
                    .expect("a completion of an issued request");
                let bounce_addr = self.mode.bounce_slot_addr(slot);
                if result == -libc::EINTR || result == -libc::EAGAIN {
                    free.push(slot);
                    if inflight.retries < MAX_TRANSIENT_RETRIES {
                        inflight.retries += 1;
                        retries.push_back(inflight);
                    } else {
                        outcomes.push((inflight.chunk, Err(IoError::Os(-result))));
                    }
                    continue;
                }
                let remaining = inflight.chunk.len - inflight.done;
                let outcome = match usize::try_from(result) {
                    Err(_) => Err(IoError::Os(-result)),
                    Ok(0) => Err(IoError::Short {
                        expected: inflight.chunk.len,
                        done: inflight.done,
                    }),
                    Ok(n) if n < remaining => {
                        if let Some(b_addr) = bounce_addr {
                            if inflight.chunk.direction == Direction::Read {
                                // SAFETY: kernel wrote `n` bytes into slot's bounce buffer;
                                // copy into caller's registered memory before freeing slot.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        b_addr as *const u8,
                                        (inflight.chunk.address + inflight.done) as *mut u8,
                                        n,
                                    );
                                }
                            }
                        }
                        let next_done = inflight.done + n;
                        if n % ALIGNMENT != 0 {
                            // O_DIRECT cannot resume from an unaligned offset.
                            Err(IoError::Short {
                                expected: inflight.chunk.len,
                                done: next_done,
                            })
                        } else {
                            free.push(slot);
                            inflight.done = next_done;
                            inflight.retries = 0;
                            retries.push_back(inflight);
                            continue;
                        }
                    }
                    Ok(n) if n == remaining => {
                        if inflight.chunk.direction == Direction::Read {
                            if let Some(b_addr) = bounce_addr {
                                // SAFETY: kernel wrote `n` bytes into slot's bounce buffer;
                                // copy into caller's registered memory before freeing slot.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        b_addr as *const u8,
                                        (inflight.chunk.address + inflight.done) as *mut u8,
                                        n,
                                    );
                                }
                            }
                        }
                        Ok(())
                    }
                    Ok(_) => Err(IoError::Os(libc::EIO)),
                };
                free.push(slot);
                outcomes.push((inflight.chunk, outcome));
            }
            let mut finished = Vec::new();
            if !outcomes.is_empty() {
                let mut queues = shared.queues.lock();
                for (chunk, outcome) in outcomes.drain(..) {
                    finished.extend(queues.finish(chunk, outcome));
                }
            }
            shared.work.notify_all();
            dispatch::report(finished);
        }
        drop(self.ring);
        drop(self.mode);
        drop(counter);
    }
}

fn request(fd: RawFd, inflight: InFlight, slot: usize, mode: &BufferMode) -> squeue::Entry {
    let chunk = inflight.chunk;
    let remaining = chunk.len - inflight.done;
    let len = u32::try_from(remaining).expect("requests are at most 1 GiB");
    let file_offset = chunk.file_offset + inflight.done as u64;
    match mode {
        BufferMode::FixedCaller => {
            let address = chunk.address + inflight.done;
            match chunk.direction {
                Direction::Read => {
                    opcode::ReadFixed::new(types::Fd(fd), address as *mut u8, len, chunk.buffer)
                        .offset(file_offset)
                        .ioprio(IOPRIO_READ)
                        .build()
                }
                Direction::Write => {
                    opcode::WriteFixed::new(types::Fd(fd), address as *const u8, len, chunk.buffer)
                        .offset(file_offset)
                        .ioprio(IOPRIO_WRITE)
                        .build()
                }
            }
        }
        BufferMode::Bounce {
            pool,
            slot_size,
            fixed,
        } => {
            let address = pool.address() + slot * slot_size;
            match (chunk.direction, *fixed) {
                (Direction::Read, true) => {
                    opcode::ReadFixed::new(types::Fd(fd), address as *mut u8, len, 0)
                        .offset(file_offset)
                        .ioprio(IOPRIO_READ)
                        .build()
                }
                (Direction::Write, true) => {
                    opcode::WriteFixed::new(types::Fd(fd), address as *const u8, len, 0)
                        .offset(file_offset)
                        .ioprio(IOPRIO_WRITE)
                        .build()
                }
                (Direction::Read, false) => {
                    opcode::Read::new(types::Fd(fd), address as *mut u8, len)
                        .offset(file_offset)
                        .ioprio(IOPRIO_READ)
                        .build()
                }
                (Direction::Write, false) => {
                    opcode::Write::new(types::Fd(fd), address as *const u8, len)
                        .offset(file_offset)
                        .ioprio(IOPRIO_WRITE)
                        .build()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        os::fd::{FromRawFd, OwnedFd},
        path::PathBuf,
        sync::{mpsc, Arc},
        thread,
        time::{Duration, Instant},
    };

    use super::*;
    use crate::{
        dispatch::{Chunk, Limits, Shared},
        executor::JobReport,
        memory::Memory,
    };

    struct ScratchFile(PathBuf);
    impl ScratchFile {
        fn new(name: &str) -> Self {
            let dir = std::env::var_os("TS_DISK_TEST_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir);
            Self(dir.join(format!("ts-disk-uring-{}-{name}", std::process::id())))
        }
    }
    impl Drop for ScratchFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn make_chunk(direction: Direction, file_offset: u64, address: usize, len: usize) -> Chunk {
        Chunk {
            job: 0,
            direction,
            file_offset,
            address,
            len,
            buffer: 0,
        }
    }

    #[test]
    fn register_buffers_efault_falls_back_to_bounce_pool_and_round_trips() {
        let scratch = ScratchFile::new("efault-and-rt");
        let file = DiskFile::create(&scratch.0, 16 * ALIGNMENT as u64).unwrap();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                ALIGNMENT,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        let pieces = Pieces::from_raw_for_test(vec![(ptr as usize, ptr as usize + ALIGNMENT)]);
        let ring = Ring::new(&file, &pieces, 4, 2 * ALIGNMENT).expect("EFAULT fallback to bounce");
        assert!(ring.mode.is_bounce());
        drop(ring);
        unsafe {
            libc::munmap(ptr, ALIGNMENT);
        }

        // Non-EFAULT register_buffers failures (e.g. empty buffer list -> EINVAL)
        // must propagate explicitly rather than silently falling back.
        let empty_pieces = Pieces::from_raw_for_test(Vec::new());
        let err = match Ring::new(&file, &empty_pieces, 4, 2 * ALIGNMENT) {
            Ok(_) => panic!("expected register_buffers error for empty pieces"),
            Err(err) => err,
        };
        assert_eq!(err.raw_os_error(), Some(libc::EINVAL));

        let ring = Ring::new_bounce(&file, 4, 2 * ALIGNMENT).unwrap();
        let mut mem = Memory::map(4 * ALIGNMENT).unwrap();
        let payload: Vec<u8> = (0..2 * ALIGNMENT)
            .map(|i| (i as u8).wrapping_mul(17).wrapping_add(5))
            .collect();
        mem.write_at(0, &payload);
        let efd = unsafe { OwnedFd::from_raw_fd(libc::eventfd(0, libc::EFD_CLOEXEC)) };
        let shared = Arc::new(Shared::new(
            Limits {
                depth: 4,
                write_depth: 2,
            },
            Some(efd),
        ));
        let ws = shared.clone();
        let worker = thread::spawn(move || ring.run(&ws));

        let (tx, rx) = mpsc::channel();
        {
            let mut q = shared.queues.lock();
            q.admit(
                Direction::Write,
                [
                    make_chunk(Direction::Write, 0, mem.address(), ALIGNMENT),
                    make_chunk(
                        Direction::Write,
                        ALIGNMENT as u64,
                        mem.address() + ALIGNMENT,
                        ALIGNMENT,
                    ),
                ],
                Box::new(move |r: JobReport| tx.send(r.outcome).unwrap()),
                Instant::now(),
            );
            q.release_write();
            q.release_write();
        }
        shared.wake();
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(10)).unwrap(),
            Ok(()),
            "bounce write"
        );

        let (tx, rx) = mpsc::channel();
        shared.queues.lock().admit(
            Direction::Read,
            [make_chunk(
                Direction::Read,
                0,
                mem.address() + 2 * ALIGNMENT,
                2 * ALIGNMENT,
            )],
            Box::new(move |r: JobReport| tx.send(r.outcome).unwrap()),
            Instant::now(),
        );
        shared.wake();
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(10)).unwrap(),
            Ok(()),
            "bounce read"
        );
        let mut actual = vec![0u8; 2 * ALIGNMENT];
        mem.read_at(2 * ALIGNMENT, &mut actual);
        assert_eq!(actual, payload, "bounce payload");
        let _ = shared.queues.lock().close();
        shared.wake();
        worker.join().unwrap();
    }
}
