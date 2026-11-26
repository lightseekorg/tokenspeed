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
    io,
    os::fd::{AsRawFd, RawFd},
};

use io_uring::{opcode, squeue, types, IoUring};

use crate::{
    dispatch::{self, Chunk, Shared},
    executor::{Direction, IoError, Pieces},
    file::DiskFile,
};

/// The user data of the eventfd read; requests use their slot index.
const WAKE_UP: u64 = u64::MAX;

pub(crate) struct Ring {
    ring: IoUring,
    fd: RawFd,
    depth: usize,
}

impl Ring {
    pub(crate) fn new(file: &DiskFile, pieces: &Pieces, depth: usize) -> io::Result<Self> {
        // One entry beyond the requests for the armed wake-up read.
        let entries = u32::try_from((depth + 1).next_power_of_two())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let ring = IoUring::new(entries)?;
        let buffers: Vec<libc::iovec> = pieces
            .iter()
            .map(|(start, end)| libc::iovec {
                iov_base: start as *mut libc::c_void,
                iov_len: end - start,
            })
            .collect();
        // SAFETY: the executor's caller keeps the memory mapped until close
        // succeeds (`Executor::start`), and a failed close leaks this ring.
        unsafe { ring.submitter().register_buffers(&buffers)? };
        Ok(Self {
            ring,
            fd: file.raw_fd(),
            depth,
        })
    }

    pub(crate) fn run(mut self, shared: &Shared) {
        let eventfd = shared
            .eventfd
            .as_ref()
            .expect("the io_uring backend wakes through an eventfd")
            .as_raw_fd();
        // The wake-up read's target. Dropped after the ring, whose teardown
        // cancels the read.
        let mut counter = Box::new(0u64);
        let mut slots: Vec<Option<Chunk>> = vec![None; self.depth];
        let mut free: Vec<usize> = (0..self.depth).rev().collect();
        let mut armed = false;
        loop {
            {
                let mut queues = shared.queues.lock();
                let mut submission = self.ring.submission();
                if !armed {
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
                while let Some(&slot) = free.last() {
                    let Some(chunk) = queues.next(shared.limits) else {
                        break;
                    };
                    free.pop();
                    let request = request(self.fd, chunk).user_data(slot as u64);
                    // SAFETY: the chunk lies in one registered piece, which
                    // stays mapped; its slot keeps it until it completes.
                    unsafe { submission.push(&request) }
                        .expect("the ring has an entry for every slot");
                    slots[slot] = Some(chunk);
                }
                drop(submission);
                if queues.closing() && queues.drained() && free.len() == self.depth {
                    break;
                }
            }
            if let Err(error) = self.ring.submit_and_wait(1) {
                // Interrupted, or the completion queue is full: reap first.
                if !matches!(error.raw_os_error(), Some(libc::EINTR | libc::EBUSY)) {
                    tracing::error!(%error, "io_uring broke down; requests it holds never report");
                    let finished = shared.queues.lock().break_down();
                    shared.wake();
                    dispatch::report(finished);
                    // The kernel may still write into the memory and the
                    // counter: keep them, and the ring, for the process.
                    std::mem::forget(self.ring);
                    std::mem::forget(counter);
                    return;
                }
            }
            let completions: Vec<(u64, i32)> = self
                .ring
                .completion()
                .map(|completion| (completion.user_data(), completion.result()))
                .collect();
            let mut finished = Vec::new();
            {
                let mut queues = shared.queues.lock();
                for (user_data, result) in completions {
                    if user_data == WAKE_UP {
                        armed = false;
                        continue;
                    }
                    let slot = usize::try_from(user_data).expect("a slot index");
                    let chunk = slots[slot]
                        .take()
                        .expect("a completion of an issued request");
                    free.push(slot);
                    finished.extend(queues.finish(chunk, outcome(result, chunk.len)));
                }
            }
            shared.work.notify_all();
            dispatch::report(finished);
        }
        drop(self.ring);
        drop(counter);
    }
}

fn request(fd: RawFd, chunk: Chunk) -> squeue::Entry {
    let len = u32::try_from(chunk.len).expect("requests are at most 1 GiB");
    match chunk.direction {
        Direction::Read => {
            opcode::ReadFixed::new(types::Fd(fd), chunk.address as *mut u8, len, chunk.buffer)
                .offset(chunk.file_offset)
                .build()
        }
        Direction::Write => {
            opcode::WriteFixed::new(types::Fd(fd), chunk.address as *const u8, len, chunk.buffer)
                .offset(chunk.file_offset)
                .build()
        }
    }
}

fn outcome(result: i32, expected: usize) -> Result<(), IoError> {
    match usize::try_from(result) {
        Err(_) => Err(IoError::Os(-result)),
        Ok(done) if done == expected => Ok(()),
        Ok(done) => Err(IoError::Short { expected, done }),
    }
}
