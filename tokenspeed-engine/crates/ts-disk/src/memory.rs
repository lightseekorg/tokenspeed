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
        #[cfg(target_os = "linux")]
        // SAFETY: `base..base+size` is the private mapping just created above
        // (without MAP_POPULATE, so no pages are faulted in yet). Applied to
        // every `ts_disk::Memory` allocation (io_uring/psync bounce pools and
        // peer-push staging rings): disable THP so `io_uring_register_buffers`
        // (`FOLL_LONGTERM`) does not fail on transparent huge pages, and mark
        // `MADV_DONTFORK` so a child process cannot trigger COW on DMA-pinned
        // pages.
        unsafe {
            let _ = libc::madvise(base, size, libc::MADV_NOHUGEPAGE);
            if libc::madvise(base, size, libc::MADV_DONTFORK) != 0 {
                tracing::warn!(
                    error = %io::Error::last_os_error(),
                    "madvise(MADV_DONTFORK) failed on disk staging memory"
                );
            }
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

    /// Move `len` bytes between two (possibly overlapping) ranges inside this
    /// mapping (`memmove` semantics). Caller must ensure exclusive ownership
    /// of both source and destination sub-ranges (no concurrent I/O or DMA).
    pub fn move_within(&self, src_offset: usize, dst_offset: usize, len: usize) {
        if len == 0 || src_offset == dst_offset {
            return;
        }
        let src_end = src_offset
            .checked_add(len)
            .expect("source range overflow in Memory::move_within");
        let dst_end = dst_offset
            .checked_add(len)
            .expect("destination range overflow in Memory::move_within");
        assert!(
            src_end <= self.size && dst_end <= self.size,
            "move_within past the mapping"
        );
        // SAFETY: both ranges lie inside `self.base..self.base+self.size`,
        // verified above; `ptr::copy` handles arbitrary overlap.
        unsafe {
            std::ptr::copy(
                self.base.as_ptr().add(src_offset),
                self.base.as_ptr().add(dst_offset),
                len,
            );
        }
    }

    /// Copy `len` bytes between two non-overlapping ranges inside this mapping.
    /// Caller must ensure exclusive ownership of both source and destination
    /// sub-ranges (no concurrent I/O or DMA touching either range).
    pub fn copy_within(&self, src_offset: usize, dst_offset: usize, len: usize) {
        let src_end = src_offset
            .checked_add(len)
            .expect("source range overflow in Memory::copy_within");
        let dst_end = dst_offset
            .checked_add(len)
            .expect("destination range overflow in Memory::copy_within");
        assert!(
            src_end <= self.size && dst_end <= self.size,
            "copy_within past the mapping"
        );
        assert!(
            src_end <= dst_offset || dst_end <= src_offset,
            "copy_within ranges overlap"
        );
        // SAFETY: both ranges lie inside `self.base..self.base+self.size` and
        // are non-overlapping, verified by the assertions above.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.base.as_ptr().add(src_offset),
                self.base.as_ptr().add(dst_offset),
                len,
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

#[cfg(test)]
mod tests {
    use super::Memory;

    #[test]
    fn copy_within_and_move_within_handle_disjoint_and_overlapping_ranges() {
        let mut mem = Memory::map(4096).unwrap();
        let pattern: Vec<u8> = (0u8..=255).collect();
        mem.write_at(128, &pattern);

        // Disjoint copy_within.
        mem.copy_within(128, 1024, pattern.len());
        let mut out = vec![0u8; pattern.len()];
        mem.read_at(1024, &mut out);
        assert_eq!(out, pattern);

        // Overlapping backward move_within (mirrors in-place span compaction).
        mem.move_within(128, 64, pattern.len());
        mem.read_at(64, &mut out);
        assert_eq!(out, pattern);

        // Overlapping forward move_within.
        mem.move_within(64, 100, pattern.len());
        mem.read_at(100, &mut out);
        assert_eq!(out, pattern);

        // Zero-length and self-move are no-ops.
        mem.move_within(100, 100, pattern.len());
        mem.move_within(100, 200, 0);
        mem.read_at(100, &mut out);
        assert_eq!(out, pattern);
    }
}
