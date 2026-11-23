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

//! Page-aligned anonymous memory, mapped for the process: the staging a
//! peer push reads disk blocks into before it sends them.

use std::{io, ptr::NonNull};

/// An anonymous private mapping, unmapped on drop.
#[derive(Debug)]
pub struct Memory {
    base: NonNull<u8>,
    size: usize,
}

// SAFETY: the mapping is plain memory owned by this value; the copies below
// take `&self`/`&mut self` like any buffer.
unsafe impl Send for Memory {}
// SAFETY: as above; shared references only read the address and size.
unsafe impl Sync for Memory {}

impl Memory {
    /// Map `size` bytes of zeroed memory.
    pub fn map(size: usize) -> io::Result<Self> {
        if size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "an empty mapping",
            ));
        }
        // SAFETY: a fresh anonymous mapping, owned by the returned value.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            base: NonNull::new(base.cast()).expect("mmap returns a non-null mapping"),
            size,
        })
    }

    /// The mapping's first byte.
    pub fn address(&self) -> usize {
        self.base.as_ptr() as usize
    }

    /// Its length.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Copy `bytes` in at `offset`. No job may be using this range.
    pub fn write_at(&mut self, offset: usize, bytes: &[u8]) {
        assert!(
            offset
                .checked_add(bytes.len())
                .is_some_and(|end| end <= self.size),
            "a copy past the mapping"
        );
        // SAFETY: the range lies inside the mapping, checked above.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.base.as_ptr().add(offset),
                bytes.len(),
            );
        }
    }

    /// Copy out `bytes.len()` bytes at `offset`. No job may be writing this
    /// range.
    pub fn read_at(&self, offset: usize, bytes: &mut [u8]) {
        assert!(
            offset
                .checked_add(bytes.len())
                .is_some_and(|end| end <= self.size),
            "a copy past the mapping"
        );
        // SAFETY: the range lies inside the mapping, checked above.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.base.as_ptr().add(offset),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        // SAFETY: unmaps exactly the mapping `map` created.
        unsafe {
            libc::munmap(self.base.as_ptr().cast(), self.size);
        }
    }
}
