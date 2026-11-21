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

//! Block I/O between registered memory and one local file, for the Disk
//! cache tier (`docs/rfcs/0006-disk-tier.md`).
//!
//! A job moves aligned ranges between the file and the memory the executor
//! was started with: the Host cache pool, which reads land in and writes
//! come from, and a peer push ring. Two backends run the same jobs and are
//! chosen explicitly: io_uring with the memory registered as fixed buffers,
//! or a pool of threads calling `pread` and `pwrite`. Both open the file
//! with `O_DIRECT`, so every offset, address and length is a multiple of
//! [`ALIGNMENT`].
//!
//! Reads go ahead of writes. Writes clean the Host tier in the background:
//! they are paced to a byte budget, which protects the SSD's endurance, and
//! take at most a bounded share of the queue. A job reports once, after
//! every I/O it issued retired; memory an unfinished job names must not be
//! reused before then. Neither backend can stop an I/O the kernel already
//! has.

// The crate is the boundary to the kernel's I/O interfaces: it reads and
// writes memory by address.
#![allow(unsafe_code)]

mod dispatch;
pub mod executor;
pub mod file;
pub mod memory;
mod psync;
mod uring;

pub use executor::{
    Backend, Direction, Executor, ExecutorConfig, Io, IoError, Job, JobId, JobReport, Pressure,
    StartError, Stats, SubmitError,
};
pub use file::{DiskFile, FileError, ALIGNMENT};
pub use memory::Memory;
