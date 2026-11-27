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

/// Per-piece O_DIRECT capability discovered on the first I/O against each
/// registered piece (`Chunk::buffer`):
/// - `Unknown`: not yet probed on this worker.
/// - `Direct`: raw `pread`/`pwrite` succeeded; any later `EFAULT` propagates as
///   `IoError::Os(EFAULT)` without touching the caller address.
/// - `Bounce`: the piece's first raw `O_DIRECT` I/O returned `EFAULT` (e.g.
///   device-pinned `VM_PFNMAP`/`VM_IO` host memory registered via
///   `Executor::start`); stage subsequent chunks for this piece through the
///   worker's anonymous mmap bounce buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PieceIoState {
    Unknown,
    Direct,
    Bounce,
}

pub(crate) fn run(shared: &Shared, file: &DiskFile, max_io_bytes: usize) {
    let mut bounce: Option<Memory> = None;
    let mut piece_states: Vec<PieceIoState> = Vec::new();
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
        let result = transfer(
            file.raw_fd(),
            chunk,
            &mut bounce,
            &mut piece_states,
            max_io_bytes,
        );
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
    piece_states: &mut Vec<PieceIoState>,
    max_io_bytes: usize,
) -> Result<(), IoError> {
    let piece_idx = usize::from(chunk.buffer);
    if piece_idx >= piece_states.len() {
        piece_states.resize(piece_idx + 1, PieceIoState::Unknown);
    }
    match piece_states[piece_idx] {
        PieceIoState::Direct => transfer_raw(fd, chunk),
        PieceIoState::Bounce => {
            let buf = ensure_bounce(bounce, max_io_bytes.max(chunk.len))?;
            transfer_bounced(fd, chunk, buf)
        }
        PieceIoState::Unknown => match transfer_raw(fd, chunk) {
            Ok(()) => {
                piece_states[piece_idx] = PieceIoState::Direct;
                Ok(())
            }
            Err(IoError::Os(libc::EFAULT)) => {
                // Only pieces that fail their very first raw O_DIRECT I/O with
                // EFAULT switch to bounce staging; callers of `Executor::start`
                // guarantee every registered piece is CPU-mapped until `close`.
                let buf_size = max_io_bytes.max(chunk.len);
                let buf = ensure_bounce(bounce, buf_size)?;
                if !LOGGED_BOUNCE_FALLBACK.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        bounce_bytes = buf_size,
                        piece = piece_idx,
                        "psync O_DIRECT hit EFAULT on first access (device-pinned VM_PFNMAP/VM_IO memory); \
                         staging piece through per-worker anonymous mmap bounce buffer"
                    );
                }
                piece_states[piece_idx] = PieceIoState::Bounce;
                transfer_bounced(fd, chunk, buf)
            }
            other => other,
        },
    }
}

fn ensure_bounce(bounce: &mut Option<Memory>, min_bytes: usize) -> Result<&Memory, IoError> {
    if bounce.as_ref().is_none_or(|b| b.size() < min_bytes) {
        let mapped = Memory::map(min_bytes)
            .map_err(|err| IoError::Os(err.raw_os_error().unwrap_or(libc::ENOMEM)))?;
        *bounce = Some(mapped);
    }
    Ok(bounce.as_ref().expect("just initialized"))
}

fn transfer_bounced(fd: RawFd, chunk: Chunk, bounce: &Memory) -> Result<(), IoError> {
    debug_assert!(chunk.len <= bounce.size());
    if chunk.direction == Direction::Write {
        // SAFETY: `Executor::start` requires every registered piece to stay
        // CPU-readable until `Executor::close` returns; `bounce` is worker-owned.
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
        // SAFETY: `bounce` holds the bytes read from disk; `Executor::start`
        // requires every registered piece to stay CPU-writable until `close`.
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    struct ScratchFile(PathBuf);
    impl ScratchFile {
        fn new(name: &str) -> Self {
            let dir = std::env::var_os("TS_DISK_TEST_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir);
            Self(dir.join(format!("ts-disk-psync-{}-{name}", std::process::id())))
        }
    }
    impl Drop for ScratchFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn transfer_bounced_roundtrips_aligned_write_and_read() {
        let scratch = ScratchFile::new("bounce-rt");
        let len = ALIGNMENT * 4;
        let file = DiskFile::create(&scratch.0, len as u64).unwrap();
        let mut src = Memory::map(len).unwrap();
        let dst = Memory::map(len).unwrap();
        let pattern: Vec<u8> = (0..len)
            .map(|i| (i.wrapping_mul(37) ^ 0x5a) as u8)
            .collect();
        src.write_at(0, &pattern);

        let mut bounce = None;
        let mut states = vec![PieceIoState::Bounce];
        let write_chunk = Chunk {
            job: 1,
            direction: Direction::Write,
            file_offset: 0,
            address: src.address(),
            len,
            buffer: 0,
        };
        transfer(file.raw_fd(), write_chunk, &mut bounce, &mut states, len).unwrap();
        assert!(bounce.is_some());

        let read_chunk = Chunk {
            job: 2,
            direction: Direction::Read,
            file_offset: 0,
            address: dst.address(),
            len,
            buffer: 0,
        };
        transfer(file.raw_fd(), read_chunk, &mut bounce, &mut states, len).unwrap();
        let mut out = vec![0u8; len];
        dst.read_at(0, &mut out);
        assert_eq!(out, pattern);
    }

    #[test]
    fn direct_piece_propagates_efault_without_bouncing() {
        let scratch = ScratchFile::new("direct-efault");
        let len = ALIGNMENT;
        let file = DiskFile::create(&scratch.0, len as u64).unwrap();
        let mut valid = Memory::map(len).unwrap();
        valid.write_at(0, &vec![0xabu8; len]);

        let mut bounce = None;
        let mut states = Vec::new();
        // First I/O on piece 0 succeeds and marks piece 0 Direct.
        let ok_chunk = Chunk {
            job: 1,
            direction: Direction::Write,
            file_offset: 0,
            address: valid.address(),
            len,
            buffer: 0,
        };
        transfer(file.raw_fd(), ok_chunk, &mut bounce, &mut states, len).unwrap();
        assert_eq!(states, vec![PieceIoState::Direct]);
        assert!(bounce.is_none());

        // A later EFAULT on a Direct piece (e.g. invalid address) must return
        // Os(EFAULT) cleanly without trying to copy into a bounce buffer.
        let bad_chunk = Chunk {
            job: 2,
            direction: Direction::Write,
            file_offset: 0,
            address: ALIGNMENT, // unmapped low address
            len,
            buffer: 0,
        };
        let err = transfer(file.raw_fd(), bad_chunk, &mut bounce, &mut states, len).unwrap_err();
        assert_eq!(err, IoError::Os(libc::EFAULT));
        assert!(bounce.is_none());
    }
}
