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

//! The thread-pool backend, for hosts whose policy refuses io_uring: each
//! of `queue_depth` threads issues one blocking `pread`/`pwrite` at a time.

use std::{
    io,
    os::fd::RawFd,
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

use crate::{
    dispatch::{Chunk, Shared},
    executor::{Direction, IoError},
    file::{DiskFile, ALIGNMENT},
    memory::Memory,
};

/// Upper bound on transient `EINTR`/`EAGAIN` retries per chunk so a broken
/// device cannot livelock `Executor::close`.
const MAX_TRANSIENT_RETRIES: u8 = 16;

static LOGGED_BOUNCE_FALLBACK: AtomicBool = AtomicBool::new(false);

pub(crate) fn run(shared: &Shared, file: &DiskFile, max_io_bytes: usize) {
    let mut bounce: Option<Memory> = None;
    loop {
        let chunk = {
            let mut queues = shared.queues.lock();
            loop {
                if let Some(chunk) = queues.next(shared.limits) {
                    break chunk;
                }
                if queues.closing() && queues.drained() {
                    return;
                }
                shared.work.wait(&mut queues);
            }
        };
        let result = transfer(file.raw_fd(), chunk, &mut bounce, max_io_bytes);
        let finished = shared.queues.lock().finish(chunk, result);
        shared.work.notify_all();
        if let Some((done, report)) = finished {
            done(report);
        }
    }
}

fn transfer(
    fd: RawFd,
    chunk: Chunk,
    bounce: &mut Option<Memory>,
    max_io_bytes: usize,
) -> Result<(), IoError> {
    match bounce.as_ref() {
        Some(buf) => transfer_bounced(fd, chunk, buf),
        None => match transfer_raw(fd, chunk) {
            Err(IoError::Os(libc::EFAULT)) => {
                let buf_size = max_io_bytes.max(chunk.len);
                match Memory::map(buf_size) {
                    Ok(buf) => {
                        if !LOGGED_BOUNCE_FALLBACK.swap(true, Ordering::Relaxed) {
                            tracing::warn!(
                                bounce_bytes = buf_size,
                                "psync O_DIRECT hit EFAULT (device-pinned VM_PFNMAP/VM_IO memory); \
                                 staging through per-worker anonymous mmap bounce buffer"
                            );
                        }
                        let res = transfer_bounced(fd, chunk, &buf);
                        *bounce = Some(buf);
                        res
                    }
                    Err(err) => Err(IoError::Os(err.raw_os_error().unwrap_or(libc::ENOMEM))),
                }
            }
            other => other,
        },
    }
}

fn transfer_bounced(fd: RawFd, chunk: Chunk, bounce: &Memory) -> Result<(), IoError> {
    debug_assert!(chunk.len <= bounce.size());
    if chunk.direction == Direction::Write {
        // SAFETY: source is caller's mapped memory; dest is worker-owned bounce buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                chunk.address as *const u8,
                bounce.address() as *mut u8,
                chunk.len,
            );
        }
    }
    let bounce_chunk = Chunk {
        address: bounce.address(),
        ..chunk
    };
    transfer_raw(fd, bounce_chunk)?;
    if chunk.direction == Direction::Read {
        // SAFETY: bounce holds the bytes read from disk; dest is caller's mapped memory.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bounce.address() as *const u8,
                chunk.address as *mut u8,
                chunk.len,
            );
        }
    }
    Ok(())
}

fn transfer_raw(fd: RawFd, chunk: Chunk) -> Result<(), IoError> {
    let base_offset = libc::off_t::try_from(chunk.file_offset).expect("files are below 2^63 bytes");
    let mut moved = 0usize;
    let mut retries = 0u8;
    while moved < chunk.len {
        let remaining = chunk.len - moved;
        let offset = base_offset + moved as libc::off_t;
        // SAFETY: the chunk lies in memory the executor's caller keeps mapped
        // (`Executor::start`); the file stays open while this thread runs.
        let done = unsafe {
            match chunk.direction {
                Direction::Read => libc::pread(
                    fd,
                    (chunk.address + moved) as *mut libc::c_void,
                    remaining,
                    offset,
                ),
                Direction::Write => libc::pwrite(
                    fd,
                    (chunk.address + moved) as *const libc::c_void,
                    remaining,
                    offset,
                ),
            }
        };
        let Ok(done) = usize::try_from(done) else {
            let errno = io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO);
            if (errno == libc::EINTR || errno == libc::EAGAIN) && retries < MAX_TRANSIENT_RETRIES {
                retries += 1;
                if errno == libc::EAGAIN {
                    thread::yield_now();
                }
                continue;
            }
            return Err(IoError::Os(errno));
        };
        if done == 0 {
            return Err(IoError::Short {
                expected: chunk.len,
                done: moved,
            });
        }
        if done > remaining {
            return Err(IoError::Os(libc::EIO));
        }
        moved += done;
        retries = 0;
        // O_DIRECT cannot resume from an unaligned offset; report the real
        // short I/O rather than letting the next syscall fail with EINVAL.
        if moved < chunk.len && done % ALIGNMENT != 0 {
            return Err(IoError::Short {
                expected: chunk.len,
                done: moved,
            });
        }
    }
    Ok(())
}
