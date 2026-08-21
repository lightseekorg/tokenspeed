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

//! Per-request logical-page -> physical-page mapping.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/core/block_table.h`.

use crate::cache_block_ref::CacheBlockRef;

/// Per-request logical-page to physical-block mapping.
#[derive(Debug, Default)]
pub struct BlockTable {
    /// `pub(crate)` mirrors the C++ `friend` classes (`GroupAllocator`,
    /// `PrefixCacheIndex`): same-crate placement logic mutates the table
    /// directly; the public API stays read-only.
    pub(crate) blocks: Vec<CacheBlockRef>,
    /// Unconsumed capacity at the logical tail. May span multiple blocks when
    /// admission preallocates a later decode/MTP step.
    pub(crate) available_tokens: i32,
    /// Slots below this monotonic frontier have already released their request
    /// ownership. Sparse state tables may contain holes between live islands,
    /// so reclaim cannot infer this frontier from the first null slot.
    pub(crate) reclaimed_prefix_blocks: i32,
}

impl BlockTable {
    /// Build a table from blocks and the remaining (unconsumed) token budget.
    pub fn from_blocks(blocks: Vec<CacheBlockRef>, available_tokens: i32) -> Self {
        assert!(
            available_tokens >= 0,
            "BlockTable available_tokens must be non-negative"
        );
        Self {
            blocks,
            available_tokens,
            reclaimed_prefix_blocks: 0,
        }
    }

    /// Borrowed view of the block list (absolute logical-page indexing; null
    /// holes are empty refs and rows are not compacted).
    pub fn blocks(&self) -> &[CacheBlockRef] {
        &self.blocks
    }

    /// Number of blocks in the table.
    pub fn num_blocks(&self) -> i32 {
        self.blocks.len() as i32
    }

    /// Unconsumed token capacity at the logical tail.
    pub fn available_tokens(&self) -> i32 {
        self.available_tokens
    }

    /// Monotonic reclaim frontier (see field doc).
    pub fn reclaimed_prefix_blocks(&self) -> i32 {
        self.reclaimed_prefix_blocks
    }

    /// Replace the block at `index` with the null hole and return the evicted
    /// block (which releases its slot when the returned ref is dropped).
    pub fn evict_to_null(&mut self, index: usize) -> CacheBlockRef {
        assert!(index < self.blocks.len(), "EvictToNull index out of range");
        std::mem::take(&mut self.blocks[index])
    }
}

/// LCM ownership ids for scheduler accounting/debugging. Kernel-facing page
/// tables must instead go through the group allocator's page-id export.
pub fn block_table_lcm_block_ids(table: &BlockTable) -> Vec<i32> {
    table
        .blocks()
        .iter()
        .map(|block_ref| block_ref.location().map_or(0, |loc| loc.lcm_block_id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_pool::BlockPool;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn pool(n: i32) -> Rc<RefCell<BlockPool>> {
        Rc::new(RefCell::new(BlockPool::new(n)))
    }

    #[test]
    fn from_blocks_keeps_slots_in_order() {
        let p = pool(2);
        let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 1, 1, 2);
        let table = BlockTable::from_blocks(blocks, 0);
        assert_eq!(table.num_blocks(), 2);
        assert_eq!(table.available_tokens(), 0);
        assert_eq!(table.reclaimed_prefix_blocks(), 0);
        assert_eq!(block_table_lcm_block_ids(&table), vec![1, 2]);
    }

    #[test]
    fn evict_to_null_returns_block_and_leaves_hole() {
        let p = pool(2);
        let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 1, 1, 2);
        let mut table = BlockTable::from_blocks(blocks, 0);
        let evicted = table.evict_to_null(0);
        assert_eq!(
            evicted.location(),
            Some(crate::cache_block_ref::CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 0,
            })
        );
        assert!(table.blocks()[0].is_null());
        assert_eq!(table.blocks()[1].location().unwrap().lcm_block_id, 2);
        drop(evicted);
        assert_eq!(p.borrow().num_occupied_slots(), 1);
    }

    #[test]
    #[should_panic(expected = "EvictToNull index out of range")]
    fn evict_to_null_panics_out_of_range() {
        let mut table = BlockTable::default();
        table.evict_to_null(0);
    }

    #[test]
    #[should_panic(expected = "BlockTable available_tokens must be non-negative")]
    fn from_blocks_rejects_negative_budget() {
        let _ = BlockTable::from_blocks(Vec::new(), -1);
    }
}
