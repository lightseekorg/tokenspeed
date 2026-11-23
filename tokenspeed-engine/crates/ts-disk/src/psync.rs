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

use std::{io, os::fd::RawFd};

use crate::{
    dispatch::{self, Chunk, Shared},
    executor::{Direction, IoError},
    file::DiskFile,
};

pub(crate) fn run(shared: &Shared, file: &DiskFile) {
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
        let result = transfer(file.raw_fd(), chunk);
        let finished = shared.queues.lock().finish(chunk, result);
        shared.work.notify_all();
        dispatch::report(finished.into_iter().collect());
    }
}

fn transfer(fd: RawFd, chunk: Chunk) -> Result<(), IoError> {
    let offset = libc::off_t::try_from(chunk.file_offset).expect("files are below 2^63 bytes");
    loop {
        // SAFETY: the chunk lies in memory the executor's caller keeps mapped
        // (`Executor::start`); the file stays open while this thread runs.
        let done = unsafe {
            match chunk.direction {
                Direction::Read => {
                    libc::pread(fd, chunk.address as *mut libc::c_void, chunk.len, offset)
                }
                Direction::Write => {
                    libc::pwrite(fd, chunk.address as *const libc::c_void, chunk.len, offset)
                }
            }
        };
        let Ok(done) = usize::try_from(done) else {
            let errno = io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO);
            if errno == libc::EINTR {
                continue;
            }
            return Err(IoError::Os(errno));
        };
        return if done == chunk.len {
            Ok(())
        } else {
            Err(IoError::Short {
                expected: chunk.len,
                done,
            })
        };
    }
}
