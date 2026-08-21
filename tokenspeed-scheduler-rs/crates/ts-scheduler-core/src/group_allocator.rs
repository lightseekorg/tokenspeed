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

//! One cache group's physical placement.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/allocator/group_allocator.h`.
//! It moves `CacheBlock`s between the `BlockPool` and `BlockTable`s and
//! resolves kernel page ids. It is deliberately token-free; every token
//! quantity is converted to block counts by `GroupGeometry` before it reaches
//! this class. Prefix reuse lives in the group's `PrefixCacheIndex`, match
//! policy in its `PrefixMatcher`.

use crate::acquire_plan::AcquirePlan;
use crate::block_pool::BlockPoolHandle;
use crate::block_table::BlockTable;
use crate::cache_block_ref::{CacheBlockLocation, CacheBlockRef};
use crate::cache_types::{BlockTransfer, PrefixMatch};
use crate::prefix_index::PrefixCacheIndex;

/// One cache group's physical placement (stateless: holds only packing
/// configuration; all mutable state lives in the pool and tables).
#[derive(Debug, Clone, Copy)]
pub struct GroupAllocator {
    cache_blocks_per_lcm_block: i32,
    group_id: u32,
}

impl GroupAllocator {
    pub fn new(cache_blocks_per_lcm_block: i32, group_id: u32) -> Self {
        assert!(
            cache_blocks_per_lcm_block > 0,
            "cache_blocks_per_lcm_block must be > 0"
        );
        Self {
            cache_blocks_per_lcm_block,
            group_id,
        }
    }

    pub fn cache_blocks_per_lcm_block(&self) -> i32 {
        self.cache_blocks_per_lcm_block
    }

    pub fn id(&self) -> u32 {
        self.group_id
    }

    /// Kernel-facing page id for a placement (1-based, contiguous per group).
    pub fn resolve_cache_block_id(&self, location: CacheBlockLocation) -> i32 {
        assert!(location.lcm_block_id > 0, "LCM block id must be > 0");
        assert!(
            0 <= location.slot_index && location.slot_index < self.cache_blocks_per_lcm_block,
            "cache block slot is out of range"
        );
        let page_id = 1
            + (location.lcm_block_id as i64 - 1) * self.cache_blocks_per_lcm_block as i64
            + location.slot_index as i64;
        assert!(
            page_id <= i32::MAX as i64,
            "kernel page id exceeds int32 range"
        );
        page_id as i32
    }

    /// Kernel page ids for every slot of `table` (null holes become 0).
    pub fn block_table_page_ids(&self, table: &BlockTable) -> Vec<i32> {
        table
            .blocks()
            .iter()
            .map(|block_ref| match block_ref.location() {
                Some(location) => self.resolve_cache_block_id(location),
                None => 0,
            })
            .collect()
    }

    /// Claims the hit blocks into a fresh table and advances the reclaim
    /// frontier past leading null holes.
    pub fn claim_hit_blocks(&self, table: &mut BlockTable, hit: PrefixMatch) {
        assert!(
            table.blocks.is_empty(),
            "ClaimHitBlocks requires a fresh (empty) table"
        );
        table.blocks = hit.blocks;
        while (table.reclaimed_prefix_blocks as usize) < table.blocks.len()
            && table.blocks[table.reclaimed_prefix_blocks as usize].is_null()
        {
            table.reclaimed_prefix_blocks += 1;
        }
    }

    /// Executes a `GroupGeometry` plan: acquires `plan.num_blocks` fresh
    /// blocks, places them (dense append or sparse suffix), and stores the
    /// planned bookkeeping. Returns false without mutation when the pool is
    /// short.
    pub fn acquire(
        &self,
        pool: &BlockPoolHandle,
        table: &mut BlockTable,
        plan: &AcquirePlan,
    ) -> bool {
        let old_num_blocks = table.num_blocks();
        let mut block_refs = Vec::new();
        if plan.num_blocks > 0 {
            block_refs = pool.borrow_mut().acquire_blocks(
                pool,
                self.group_id,
                self.cache_blocks_per_lcm_block,
                plan.num_blocks as usize,
            );
            if block_refs.len() < plan.num_blocks as usize {
                return false;
            }
        }
        if plan.suffix_start < 0 {
            table.blocks.extend(block_refs);
        } else {
            assert!(
                table.num_blocks() <= plan.suffix_start,
                "sparse suffix overlaps the existing block table"
            );
            assert!(
                plan.suffix_start + plan.num_blocks <= plan.table_blocks_after,
                "sparse suffix exceeds the planned table size"
            );
            table
                .blocks
                .resize(plan.table_blocks_after as usize, CacheBlockRef::default());
            if old_num_blocks == 0 {
                table.reclaimed_prefix_blocks = plan.suffix_start;
            }
            for (i, block_ref) in block_refs.into_iter().enumerate() {
                table.blocks[plan.suffix_start as usize + i] = block_ref;
            }
        }
        table.available_tokens = plan.available_tokens_after;
        true
    }

    /// Appends host-tier blocks to the table, pairing each source with a fresh
    /// device destination in `load_pairs`.
    pub fn append_host_extension(
        &self,
        pool: &BlockPoolHandle,
        table: &mut BlockTable,
        host_block_refs: Vec<CacheBlockRef>,
        load_pairs: &mut Vec<BlockTransfer>,
    ) {
        assert!(
            table.available_tokens == 0,
            "host extension must append on a full-page boundary"
        );
        let num_pages = host_block_refs
            .iter()
            .filter(|block_ref| !block_ref.is_null())
            .count();
        table
            .blocks
            .reserve(table.blocks.len() + host_block_refs.len());
        let destination_refs = pool.borrow_mut().acquire_blocks(
            pool,
            self.group_id,
            self.cache_blocks_per_lcm_block,
            num_pages,
        );
        assert!(
            destination_refs.len() == num_pages,
            "admission plan no longer fits the block pool"
        );
        let mut destination_it = destination_refs.into_iter();
        for host_block_ref in host_block_refs {
            if host_block_ref.is_null() {
                table.blocks.push(CacheBlockRef::default());
                continue;
            }
            let destination = destination_it
                .next()
                .expect("missing host extension destination");
            table.blocks.push(destination.clone());
            load_pairs.push(BlockTransfer {
                group_id: self.group_id,
                source: host_block_ref,
                destination,
            });
        }
        assert!(
            destination_it.next().is_none(),
            "unused host extension destination"
        );
    }

    /// Retention execution: the first `num_expired_blocks` table slots become
    /// null holes, so the table never shrinks and slot alignment stays stable.
    pub fn reclaim_expired(&self, table: &mut BlockTable, num_expired_blocks: i32) {
        let expired = num_expired_blocks.min(table.num_blocks());
        for i in table.reclaimed_prefix_blocks..expired {
            table.evict_to_null(i as usize).reset();
        }
        table.reclaimed_prefix_blocks = table.reclaimed_prefix_blocks.max(expired);
    }

    /// Only blocks uniquely owned by this table reach the free list, so shared
    /// ones don't count.
    pub fn blocks_reclaimable_at(
        &self,
        index: &PrefixCacheIndex,
        table: &BlockTable,
        num_expired_blocks: i32,
        count_uncached: bool,
    ) -> i32 {
        let expired = num_expired_blocks.min(table.num_blocks());
        let mut freed = 0;
        for i in table.reclaimed_prefix_blocks..expired {
            let block = &table.blocks[i as usize];
            if block.is_null() {
                continue;
            }
            let cached = index.contains_ref(block);
            let only_table_and_cache_owners = cached && block.use_count() == 2;
            if only_table_and_cache_owners || (count_uncached && !cached && block.unique()) {
                freed += 1;
            }
        }
        freed
    }

    /// Locations reclaimable once `released_locations` lose their request
    /// references.
    pub fn reclaimable_block_locations_at(
        &self,
        index: &PrefixCacheIndex,
        table: &BlockTable,
        num_expired_blocks: i32,
        released_locations: &[CacheBlockLocation],
    ) -> Vec<CacheBlockLocation> {
        let expired = num_expired_blocks.min(table.num_blocks());
        let mut locations = Vec::new();
        for i in table.reclaimed_prefix_blocks..expired {
            let block = &table.blocks[i as usize];
            if block.is_null() {
                continue;
            }
            let cached = index.contains_ref(block);
            let location = block.location().expect("non-null block has a location");
            let released_owners = released_locations
                .iter()
                .filter(|&&l| l == location)
                .count() as u32;
            if (cached && block.use_count() == 2 + released_owners) || (!cached && block.unique()) {
                locations.push(location);
            }
        }
        locations
    }

    pub fn consume_reserved_tokens(&self, table: &mut BlockTable, num_tokens: i32) {
        assert!(
            num_tokens >= 0 && num_tokens <= table.available_tokens,
            "token demand exceeds the available capacity"
        );
        table.available_tokens -= num_tokens;
    }

    pub fn free(&self, table: &mut BlockTable) {
        // Release the logical suffix first so newly emptied LCM parents enter
        // the FIFO free queue in deterministic table order.
        for block_ref in table.blocks.iter_mut().rev() {
            block_ref.reset();
        }
        table.blocks.clear();
        table.available_tokens = 0;
        table.reclaimed_prefix_blocks = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_pool::BlockPool;
    use crate::cache_types::{CacheBoundaryKind, CacheKey};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn pool(n: i32) -> Rc<RefCell<BlockPool>> {
        Rc::new(RefCell::new(BlockPool::new(n)))
    }

    fn key(group: u32, hash: &str, offset: i32) -> CacheKey {
        CacheKey {
            namespace_id: 0,
            group_id: group,
            content_hash: hash.to_string(),
            page_offset: offset,
        }
    }

    #[test]
    fn resolve_cache_block_id_maps_placement_to_page() {
        let a = GroupAllocator::new(1, 0);
        assert_eq!(
            a.resolve_cache_block_id(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 0
            }),
            1
        );
        assert_eq!(
            a.resolve_cache_block_id(CacheBlockLocation {
                lcm_block_id: 3,
                slot_index: 0
            }),
            3
        );
        let b = GroupAllocator::new(2, 1);
        assert_eq!(
            b.resolve_cache_block_id(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 0
            }),
            1
        );
        assert_eq!(
            b.resolve_cache_block_id(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 1
            }),
            2
        );
        assert_eq!(
            b.resolve_cache_block_id(CacheBlockLocation {
                lcm_block_id: 2,
                slot_index: 0
            }),
            3
        );
    }

    #[test]
    fn acquire_dense_appends_and_tracks_tokens() {
        let p = pool(4);
        let alloc = GroupAllocator::new(1, 1);
        let mut table = BlockTable::default();
        let plan = crate::group_geometry::GroupGeometry::new(4).plan_acquire(&table, 5, 0);
        assert!(alloc.acquire(&p, &mut table, &plan));
        assert_eq!(table.num_blocks(), 2);
        assert_eq!(table.available_tokens(), 3);
        assert_eq!(alloc.block_table_page_ids(&table), vec![1, 2]);
        assert_eq!(p.borrow().num_occupied_slots(), 2);
    }

    #[test]
    fn acquire_sparse_suffix_keeps_null_holes() {
        let p = pool(4);
        let alloc = GroupAllocator::new(1, 1);
        let mut table = BlockTable::default();
        let demand = crate::cache_types::GroupDemand {
            table: None,
            num_tokens: 5,
            prefix_hashes: &[],
            new_prefix_hash_begin: 0,
            completed_boundary_kind: None,
            num_computed_tokens: -1,
            reserve_tokens: 0,
            materialized_suffix_start: 2,
        };
        let plan =
            crate::group_geometry::GroupGeometry::new(1).plan_acquire_demand(&table, &demand);
        assert!(alloc.acquire(&p, &mut table, &plan));
        assert_eq!(table.num_blocks(), 5); // suffix starts at 2, extent 5
        assert!(table.blocks()[0].is_null());
        assert!(table.blocks()[1].is_null());
        assert!(!table.blocks()[2].is_null());
        assert_eq!(alloc.block_table_page_ids(&table), vec![0, 0, 1, 2, 3]);
        assert_eq!(table.reclaimed_prefix_blocks(), 2);
    }

    #[test]
    fn claim_hit_blocks_advances_reclaim_frontier_past_holes() {
        let p = pool(4);
        let alloc = GroupAllocator::new(1, 1);
        let mut hit = PrefixMatch { blocks: Vec::new() };
        let b = p.borrow_mut().acquire_block(&p, 1, 1).expect("block");
        hit.blocks.push(CacheBlockRef::default()); // hole
        hit.blocks.push(b);
        let mut table = BlockTable::default();
        alloc.claim_hit_blocks(&mut table, hit);
        assert_eq!(table.num_blocks(), 2);
        assert_eq!(table.reclaimed_prefix_blocks(), 1);
    }

    #[test]
    fn reclaim_expired_evicts_to_null_and_advances_frontier() {
        let p = pool(4);
        let alloc = GroupAllocator::new(1, 1);
        let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 1, 1, 3);
        let mut table = BlockTable::from_blocks(blocks, 0);
        alloc.reclaim_expired(&mut table, 2);
        assert!(table.blocks()[0].is_null());
        assert!(table.blocks()[1].is_null());
        assert!(!table.blocks()[2].is_null());
        assert_eq!(table.reclaimed_prefix_blocks(), 2);
        assert_eq!(p.borrow().num_occupied_slots(), 1);
    }

    #[test]
    fn free_releases_all_blocks_suffix_first() {
        let p = pool(4);
        let alloc = GroupAllocator::new(1, 1);
        let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 1, 1, 2);
        let mut table = BlockTable::from_blocks(blocks, 0);
        alloc.free(&mut table);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(p.borrow().num_occupied_slots(), 0);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 4);
    }

    #[test]
    fn blocks_reclaimable_counts_only_releasable_owners() {
        let p = pool(4);
        let alloc = GroupAllocator::new(1, 1);
        let mut index = PrefixCacheIndex::new(1);
        let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 1, 1, 2);
        // Register block 0 in the index (so it is cached: index + table owners).
        let table = BlockTable::from_blocks(blocks, 0);
        let mut b0 = table.blocks()[0].clone();
        index.register(
            &p,
            &mut b0,
            key(1, "h0", 0),
            1,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
        drop(b0); // remove the extra local owner; table + index remain
                  // Block 0 is cached (2 owners), block 1 is uncached (1 owner).
        assert_eq!(alloc.blocks_reclaimable_at(&index, &table, 2, true), 2);
        assert_eq!(alloc.blocks_reclaimable_at(&index, &table, 2, false), 1);
    }

    #[test]
    fn consume_reserved_tokens_checks_capacity() {
        let alloc = GroupAllocator::new(1, 1);
        let mut table = BlockTable::from_blocks(Vec::new(), 5);
        alloc.consume_reserved_tokens(&mut table, 3);
        assert_eq!(table.available_tokens(), 2);
        alloc.consume_reserved_tokens(&mut table, 2);
        assert_eq!(table.available_tokens(), 0);
    }
}
