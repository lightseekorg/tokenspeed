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

//! The Disk tier's file: one per cache-owning rank, opened for direct I/O
//! and allocated up front. Its contents never outlive the process that wrote
//! them: a restart reuses the file and ignores what is in it.

use std::{
    fs::{File, OpenOptions},
    io,
    os::unix::{
        fs::OpenOptionsExt,
        io::{AsRawFd, RawFd},
    },
    path::{Path, PathBuf},
};

/// The alignment of every direct-I/O offset, address and length: the
/// logical block size of 4Kn devices and a multiple of 512e devices'.
pub const ALIGNMENT: usize = 4096;

/// A file the Disk tier owns, open for direct I/O.
#[derive(Debug)]
pub struct DiskFile {
    file: File,
    path: PathBuf,
    size: u64,
}

/// Why the file could not be prepared.
#[derive(Debug, thiserror::Error)]
pub enum FileError {
    /// The size is zero, unaligned or past the file offsets the OS takes.
    #[error("a Disk tier file of {0} bytes is not a positive multiple of {ALIGNMENT} below 2^63")]
    Size(u64),
    /// The file did not open for direct I/O.
    #[error("opening {path} for direct I/O: {source}")]
    Open {
        /// The file.
        path: PathBuf,
        /// Why.
        source: io::Error,
    },
    /// The file system did not allocate the file.
    #[error("allocating {size} bytes of {path}: {source}")]
    Allocate {
        /// The file.
        path: PathBuf,
        /// The bytes asked for.
        size: u64,
        /// Why.
        source: io::Error,
    },
}

impl DiskFile {
    /// Open or create `path` for direct I/O and allocate `size` bytes of it.
    /// What an existing file holds is ignored.
    pub fn create(path: &Path, size: u64) -> Result<Self, FileError> {
        let Ok(length) = libc::off_t::try_from(size) else {
            return Err(FileError::Size(size));
        };
        if size == 0 || size % ALIGNMENT as u64 != 0 {
            return Err(FileError::Size(size));
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .map_err(|source| FileError::Open {
                path: path.to_owned(),
                source,
            })?;
        // Allocate the extents now, so a write never fails for space and
        // never pays for allocation.
        // SAFETY: fallocate on a descriptor this function owns; no memory is
        // involved.
        if unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, length) } != 0 {
            return Err(FileError::Allocate {
                path: path.to_owned(),
                size,
                source: io::Error::last_os_error(),
            });
        }
        Ok(Self {
            file,
            path: path.to_owned(),
            size,
        })
    }

    /// The bytes the tier may use.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Where the file is.
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}
