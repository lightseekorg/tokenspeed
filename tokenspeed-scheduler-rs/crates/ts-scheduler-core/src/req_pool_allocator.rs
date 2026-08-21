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

//! Request-pool slot allocator.
//!
//! Ported from `tokenspeed-scheduler/csrc/resource/allocator/req_pool_allocator.{h,cpp}`.
//! Slot 0 is conventionally reserved (matches Python which starts from index 1);
//! real slots are `1..=size`.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

/// Allocates request-pool slots; slot 0 is reserved and never handed out.
pub struct ReqPoolAllocator {
    size: i32,
    free_slots: VecDeque<i32>,
}

/// Shared handle used by [`ReqPoolIndex`] so a slot is returned on drop even
/// when the allocator lives inside a `Scheduler` (C++ used a raw pointer with
/// an outlives invariant; `Rc<RefCell<..>>` makes it safe and keeps the
/// scheduler single-threaded).
pub type ReqPoolAllocatorHandle = Rc<RefCell<ReqPoolAllocator>>;

impl ReqPoolAllocator {
    /// Build an allocator with `size` usable slots (slots `1..=size`).
    pub fn new(size: i32) -> Self {
        assert!(size >= 0, "ReqPoolAllocator size must be >= 0");
        let mut free_slots = VecDeque::with_capacity(size as usize);
        for i in 1..=size {
            free_slots.push_back(i);
        }
        Self { size, free_slots }
    }

    /// Total number of usable slots.
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Number of currently free slots.
    pub fn available_slots(&self) -> i32 {
        self.free_slots.len() as i32
    }

    /// Return a slot to the free list (called by [`ReqPoolIndex::drop`]).
    pub(crate) fn deallocate(&mut self, slot: i32) {
        self.free_slots.push_back(slot);
    }
}

/// RAII handle that returns its slot to the allocator on drop.
///
/// Mirrors the C++ `ReqPoolIndex`: it is move-only and must not be cloned,
/// otherwise the same slot would be returned twice.
pub struct ReqPoolIndex {
    slot: i32,
    allocator: Option<ReqPoolAllocatorHandle>,
}

impl ReqPoolIndex {
    /// The allocated slot (1-based), or `-1` once moved-from.
    pub fn slot(&self) -> i32 {
        self.slot
    }

    /// Whether this handle still owns a slot.
    pub fn valid(&self) -> bool {
        self.allocator.is_some()
    }
}

impl Drop for ReqPoolIndex {
    fn drop(&mut self) {
        if let Some(allocator) = self.allocator.take() {
            allocator.borrow_mut().deallocate(self.slot);
        }
    }
}

impl std::fmt::Debug for ReqPoolIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReqPoolIndex")
            .field("slot", &self.slot)
            .field("valid", &self.valid())
            .finish()
    }
}

/// Allocate one slot. Panics when the pool is exhausted, with the same
/// diagnostic as the C++ `std::runtime_error`.
pub fn allocate(allocator: &ReqPoolAllocatorHandle) -> ReqPoolIndex {
    let slot = {
        let mut inner = allocator.borrow_mut();
        inner.free_slots.pop_front().unwrap_or_else(|| {
            panic!(
                "ReqPoolAllocator::Allocate: no request pool slots available; capacity={}",
                inner.size()
            )
        })
    };
    ReqPoolIndex {
        slot,
        allocator: Some(Rc::clone(allocator)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle(size: i32) -> ReqPoolAllocatorHandle {
        Rc::new(RefCell::new(ReqPoolAllocator::new(size)))
    }

    #[test]
    fn allocates_1_based_slots_in_order() {
        let pool = handle(3);
        let a = allocate(&pool);
        let b = allocate(&pool);
        let c = allocate(&pool);
        assert_eq!(a.slot(), 1);
        assert_eq!(b.slot(), 2);
        assert_eq!(c.slot(), 3);
        assert_eq!(pool.borrow().available_slots(), 0);
    }

    #[test]
    fn drop_returns_slot_to_free_list() {
        let pool = handle(2);
        let a = allocate(&pool); // takes slot 1; free list is now [2]
        assert_eq!(pool.borrow().available_slots(), 1);
        drop(a); // free list becomes [2, 1] (release appends)
        assert_eq!(pool.borrow().available_slots(), 2);
        // Allocation consumes the front of the deque -> slot 2.
        let reused = allocate(&pool);
        assert_eq!(reused.slot(), 2);
    }

    #[test]
    fn drop_returns_slot_exactly_once_after_move() {
        let pool = handle(1);
        let a = allocate(&pool); // takes slot 1
        assert!(a.valid());
        let b = a; // move; slot ownership transfers to `b`
        assert_eq!(pool.borrow().available_slots(), 0); // slot still held by b
        drop(b); // returns slot 1 exactly once
        assert_eq!(pool.borrow().available_slots(), 1);
        let c = allocate(&pool);
        assert_eq!(c.slot(), 1); // the free list holds exactly one copy
    }

    #[test]
    fn zero_size_pool_has_no_slots() {
        let pool = handle(0);
        assert_eq!(pool.borrow().available_slots(), 0);
    }

    #[test]
    #[should_panic(expected = "no request pool slots available")]
    fn allocate_exhausted_panics() {
        let pool = handle(1);
        let _held = allocate(&pool); // holds the only slot
        let _ = allocate(&pool); // pool exhausted -> panic
    }
}
