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

//! Physical LCM placement pool.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/core/block_pool.h`. The pool
//! deliberately has no cache key, LRU node, or ownership count; it only owns
//! the placement of child slots inside LCM-sized parent blocks. Parent blocks
//! are bound to one group; children are handed out as [`CacheBlockRef`]s whose
//! last owner returns the slot via [`CacheBlock`]'s `Drop`.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::cache_block_ref::{CacheBlock, CacheBlockLocation, CacheBlockRef};

/// Shared handle to a [`BlockPool`]. The scheduler and every [`CacheBlock`]
/// share one instance; `Rc` guarantees the pool outlives all live blocks
/// (C++ required the same invariant via raw pointers).
pub type BlockPoolHandle = Rc<RefCell<BlockPool>>;

/// One LCM-sized physical parent block.
struct LcmBlock {
    /// Group that owns this parent; unset while free.
    bound_group: Option<u32>,
    /// Child-slot occupancy; empty while free, sized `slots_per_parent` while
    /// bound. Port uses `Vec<bool>` (the C++ `std::vector<bool>` bit-packing is
    /// an implementation detail, not a contract).
    occupancy: Vec<bool>,
    occupied_count: u32,
}

/// Physical LCM placement pool for one scheduler.
pub struct BlockPool {
    lcm_blocks: Vec<LcmBlock>,
    /// Free parents are interchangeable: release appends and allocation
    /// consumes the front. Bound parents are selected separately by
    /// `plan_locations`.
    free_parent_ids: VecDeque<i32>,
}

impl BlockPool {
    /// Build a pool with `num_lcm_blocks` physical parents (ids `1..=n`).
    pub fn new(num_lcm_blocks: i32) -> Self {
        assert!(num_lcm_blocks >= 0, "num_lcm_blocks must be >= 0");
        let mut free_parent_ids = VecDeque::with_capacity(num_lcm_blocks as usize);
        for id in 1..=num_lcm_blocks {
            free_parent_ids.push_back(id);
        }
        Self {
            lcm_blocks: std::iter::repeat_with(|| LcmBlock {
                bound_group: None,
                occupancy: Vec::new(),
                occupied_count: 0,
            })
            .take(num_lcm_blocks as usize)
            .collect(),
            free_parent_ids,
        }
    }

    /// Number of physical LCM blocks (kernel page 0 is reserved separately).
    pub fn num_lcm_blocks(&self) -> i32 {
        self.lcm_blocks.len() as i32
    }

    /// Number of completely free LCM parents.
    pub fn num_empty_lcm_blocks(&self) -> i32 {
        self.free_parent_ids.len() as i32
    }

    /// Acquire a single block for `group_id`, or `None` when out of space.
    pub fn acquire_block(
        &mut self,
        handle: &BlockPoolHandle,
        group_id: u32,
        cache_blocks_per_lcm_block: i32,
    ) -> Option<CacheBlockRef> {
        self.acquire_blocks(handle, group_id, cache_blocks_per_lcm_block, 1)
            .into_iter()
            .next()
    }

    /// Acquire `num` blocks for `group_id`, or an empty vector when the pool
    /// cannot satisfy the full request.
    pub fn acquire_blocks(
        &mut self,
        handle: &BlockPoolHandle,
        group_id: u32,
        cache_blocks_per_lcm_block: i32,
        num: usize,
    ) -> Vec<CacheBlockRef> {
        assert!(
            cache_blocks_per_lcm_block > 0,
            "cache_blocks_per_lcm_block must be > 0"
        );
        if num == 0 {
            return Vec::new();
        }
        let locations = self.plan_locations(group_id, cache_blocks_per_lcm_block, num);
        if locations.len() != num {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(num);
        for location in locations {
            // Create the owner before committing the slot so its `Drop` (which
            // releases the location) is registered even on a later panic.
            let control = Rc::new(CacheBlock::new(Rc::clone(handle), location));
            self.occupy(group_id, cache_blocks_per_lcm_block, location);
            out.push(CacheBlockRef::new(control));
        }
        out
    }

    /// Group bound to a parent, if any.
    pub fn bound_group(&self, lcm_block_id: i32) -> Option<u32> {
        self.lcm_block(lcm_block_id).bound_group
    }

    /// Number of occupied child slots in a parent.
    pub fn occupied_count(&self, lcm_block_id: i32) -> i32 {
        self.lcm_block(lcm_block_id).occupied_count as i32
    }

    /// Whether a specific child slot is occupied.
    pub fn is_occupied(&self, location: CacheBlockLocation) -> bool {
        let parent = self.lcm_block(location.lcm_block_id);
        location.slot_index >= 0
            && (location.slot_index as usize) < parent.occupancy.len()
            && parent.occupancy[location.slot_index as usize]
    }

    /// Total number of occupied child slots across all parents.
    pub fn num_occupied_slots(&self) -> i32 {
        self.lcm_blocks
            .iter()
            .map(|b| b.occupied_count as i32)
            .sum()
    }

    /// Occupied child locations of one parent.
    pub fn occupied_locations(&self, lcm_block_id: i32) -> Vec<CacheBlockLocation> {
        let parent = self.lcm_block(lcm_block_id);
        parent
            .occupancy
            .iter()
            .enumerate()
            .filter(|(_, occupied)| **occupied)
            .map(|(slot, _)| CacheBlockLocation {
                lcm_block_id,
                slot_index: slot as i32,
            })
            .collect()
    }

    /// Return a child slot to its parent (called by [`CacheBlock::drop`]).
    /// The parent becomes free again once its last slot is released.
    pub fn release(&mut self, location: CacheBlockLocation) {
        assert!(
            location.lcm_block_id > 0 && (location.lcm_block_id as usize) <= self.lcm_blocks.len(),
            "CacheBlock location has invalid LCM block id"
        );
        let parent = &mut self.lcm_blocks[location.lcm_block_id as usize - 1];
        assert!(
            location.slot_index >= 0 && (location.slot_index as usize) < parent.occupancy.len(),
            "CacheBlock location has invalid slot"
        );
        let slot = location.slot_index as usize;
        assert!(
            parent.occupancy[slot] && parent.occupied_count > 0,
            "CacheBlock location is not occupied"
        );
        parent.occupancy[slot] = false;
        parent.occupied_count -= 1;
        if parent.occupied_count == 0 {
            parent.bound_group = None;
            parent.occupancy.clear();
            assert!(
                self.free_parent_ids.len() < self.lcm_blocks.len(),
                "free LCM block queue cannot exceed the pool size"
            );
            self.free_parent_ids.push_back(location.lcm_block_id);
        }
    }

    /// Immutable access to a parent (1-based id), panicking on invalid ids.
    fn lcm_block(&self, lcm_block_id: i32) -> &LcmBlock {
        assert!(
            lcm_block_id > 0 && (lcm_block_id as usize) <= self.lcm_blocks.len(),
            "LCM block id out of range"
        );
        &self.lcm_blocks[lcm_block_id as usize - 1]
    }

    /// Commit a planned location: binds the parent on first use and marks the
    /// slot occupied.
    fn occupy(&mut self, group_id: u32, slots_per_parent: i32, location: CacheBlockLocation) {
        let parent = &mut self.lcm_blocks[location.lcm_block_id as usize - 1];
        if parent.occupied_count == 0 {
            assert!(
                !self.free_parent_ids.is_empty()
                    && self.free_parent_ids.front() == Some(&location.lcm_block_id),
                "empty LCM placement must consume the next free parent"
            );
            assert!(
                parent.occupancy.is_empty(),
                "empty LCM parent must not retain child slots"
            );
            parent.occupancy = vec![false; slots_per_parent as usize];
            self.free_parent_ids.pop_front();
            parent.bound_group = Some(group_id);
        }
        assert!(
            parent.bound_group == Some(group_id)
                && parent.occupancy.len() == slots_per_parent as usize,
            "LCM parent binding changed while occupied"
        );
        let slot = location.slot_index as usize;
        assert!(
            slot < parent.occupancy.len(),
            "LCM child slot is out of range"
        );
        assert!(!parent.occupancy[slot], "LCM child slot already occupied");
        parent.occupancy[slot] = true;
        parent.occupied_count += 1;
    }

    /// Plan `count` child locations: fill partially occupied parents of the
    /// group first (most occupied first, then lowest id), then consume free
    /// parents from the front of the queue. Returns an empty vector when the
    /// pool cannot satisfy the full request.
    fn plan_locations(
        &self,
        group_id: u32,
        slots_per_parent: i32,
        count: usize,
    ) -> Vec<CacheBlockLocation> {
        let mut partially_filled_parent_ids: Vec<i32> = Vec::new();
        for (index, parent) in self.lcm_blocks.iter().enumerate() {
            if parent.bound_group != Some(group_id) {
                continue;
            }
            assert!(
                parent.occupancy.len() == slots_per_parent as usize,
                "group packing changed while LCM block is occupied"
            );
            if (parent.occupied_count as usize) < parent.occupancy.len() {
                partially_filled_parent_ids.push(index as i32 + 1);
            }
        }
        partially_filled_parent_ids.sort_by(|lhs, rhs| {
            let lhs_occupied = self.lcm_block(*lhs).occupied_count;
            let rhs_occupied = self.lcm_block(*rhs).occupied_count;
            rhs_occupied.cmp(&lhs_occupied).then_with(|| lhs.cmp(rhs))
        });

        let mut locations = Vec::with_capacity(count);
        for lcm_block_id in partially_filled_parent_ids {
            let parent = self.lcm_block(lcm_block_id);
            for (slot, occupied) in parent.occupancy.iter().enumerate() {
                if locations.len() == count {
                    break;
                }
                if !*occupied {
                    locations.push(CacheBlockLocation {
                        lcm_block_id,
                        slot_index: slot as i32,
                    });
                }
            }
            if locations.len() == count {
                return locations;
            }
        }

        for lcm_block_id in self.free_parent_ids.iter().copied() {
            for slot in 0..slots_per_parent {
                if locations.len() == count {
                    break;
                }
                locations.push(CacheBlockLocation {
                    lcm_block_id,
                    slot_index: slot,
                });
            }
            if locations.len() == count {
                return locations;
            }
        }
        // Match the C++ contract: an unsatisfiable request yields no locations
        // (the caller checks `len == num` and fails the whole acquisition).
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(n: i32) -> Rc<RefCell<BlockPool>> {
        Rc::new(RefCell::new(BlockPool::new(n)))
    }

    #[test]
    fn new_pool_has_all_parents_free() {
        let p = pool(4);
        assert_eq!(p.borrow().num_lcm_blocks(), 4);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 4);
        assert_eq!(p.borrow().num_occupied_slots(), 0);
    }

    #[test]
    fn acquire_single_block_consumes_first_free_parent() {
        let p = pool(3);
        let b = p.borrow_mut().acquire_block(&p, 7, 1).expect("block");
        assert_eq!(
            b.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 0
            })
        );
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 2);
        assert_eq!(p.borrow().bound_group(1), Some(7));
        assert_eq!(p.borrow().num_occupied_slots(), 1);
        assert!(p.borrow().is_occupied(CacheBlockLocation {
            lcm_block_id: 1,
            slot_index: 0
        }));
    }

    #[test]
    fn packing_packs_children_into_same_parent_then_next() {
        let p = pool(2);
        let a = p.borrow_mut().acquire_block(&p, 1, 2).expect("a");
        let b = p.borrow_mut().acquire_block(&p, 1, 2).expect("b");
        let c = p.borrow_mut().acquire_block(&p, 1, 2).expect("c");
        assert_eq!(
            a.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 0
            })
        );
        assert_eq!(
            b.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 1
            })
        );
        // Parent 1 is full; parent 2 is consumed next.
        assert_eq!(
            c.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 2,
                slot_index: 0
            })
        );
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 0);
        assert_eq!(p.borrow().num_occupied_slots(), 3);
    }

    #[test]
    fn releasing_last_child_frees_parent_and_reuses_it() {
        let p = pool(2);
        let a = p.borrow_mut().acquire_block(&p, 1, 2).expect("a");
        let b = p.borrow_mut().acquire_block(&p, 1, 2).expect("b");
        drop(a);
        assert_eq!(p.borrow().num_occupied_slots(), 1);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 1); // parent 2 free, parent 1 still bound
        drop(b);
        assert_eq!(p.borrow().num_occupied_slots(), 0);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 2);
        // Release appends, allocation consumes the front: after both parents
        // are free the queue is [2, 1], so parent 2 is consumed next.
        let c = p.borrow_mut().acquire_block(&p, 1, 2).expect("c");
        assert_eq!(
            c.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 2,
                slot_index: 0
            })
        );
    }

    #[test]
    fn plan_fills_most_occupied_parent_first() {
        let p = pool(3);
        // Fill parent 1 completely (2 slots) and parent 2 partially (1 slot).
        let a1 = p.borrow_mut().acquire_block(&p, 1, 2).expect("a1");
        let a2 = p.borrow_mut().acquire_block(&p, 1, 2).expect("a2");
        let b1 = p.borrow_mut().acquire_block(&p, 1, 2).expect("b1");
        drop(a1); // parent 1 now 1/2, parent 2 now 1/2
                  // Next acquisition prefers the most-occupied parent; both are 1/2 so
                  // the lowest id wins (parent 1).
        let c = p.borrow_mut().acquire_block(&p, 1, 2).expect("c");
        assert_eq!(
            c.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 0
            })
        );
        let _ = (a2, b1);
    }

    #[test]
    fn acquire_returns_empty_when_exhausted() {
        let p = pool(1);
        let a = p.borrow_mut().acquire_block(&p, 1, 1).expect("a");
        // Bind the result so the block drops before the next RefMut borrow.
        let none = p.borrow_mut().acquire_block(&p, 1, 1);
        assert!(none.is_none());
        drop(a);
        let some = p.borrow_mut().acquire_block(&p, 1, 1);
        assert!(some.is_some());
    }

    #[test]
    fn acquire_many_respects_partial_fill_then_free() {
        let p = pool(2);
        // Parent 1: 1 occupied, parent 2: free.
        let a = p.borrow_mut().acquire_block(&p, 5, 2).expect("a");
        let got = p.borrow_mut().acquire_blocks(&p, 5, 2, 3);
        assert_eq!(got.len(), 3);
        let locs: Vec<_> = got.iter().map(|r| r.location().unwrap()).collect();
        // parent 1 slot 1 (fill), then parent 2 slots 0,1.
        assert_eq!(
            locs,
            vec![
                CacheBlockLocation {
                    lcm_block_id: 1,
                    slot_index: 1
                },
                CacheBlockLocation {
                    lcm_block_id: 2,
                    slot_index: 0
                },
                CacheBlockLocation {
                    lcm_block_id: 2,
                    slot_index: 1
                },
            ]
        );
        let _ = a;
    }

    #[test]
    fn acquire_blocks_partial_failure_returns_empty() {
        let p = pool(1);
        let a = p.borrow_mut().acquire_block(&p, 1, 2).expect("a");
        // Only one slot left; asking for 2 must return empty (all-or-nothing).
        assert!(p.borrow_mut().acquire_blocks(&p, 1, 2, 2).is_empty());
        let _ = a;
    }

    #[test]
    fn pool_is_reusable_after_full_cycle() {
        let p = pool(3);
        let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 9, 1, 3);
        assert_eq!(p.borrow().num_occupied_slots(), 3);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 0);
        drop(blocks);
        assert_eq!(p.borrow().num_occupied_slots(), 0);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 3);
    }
}
