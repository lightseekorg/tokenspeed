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

//! One cache group's token -> page arithmetic.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/coordinator/group_geometry.h`.
//! This is the logical side of the placement seam: it turns token demands into
//! the token-free `AcquirePlan` / block counts that `GroupAllocator` executes,
//! so the allocator never perceives `block_granularity` or any other token
//! quantity.

use crate::acquire_plan::AcquirePlan;
use crate::block_table::BlockTable;
use crate::cache_types::{AttnKind, CacheGroupSpec, GroupDemand};

/// Logical (token-unit) geometry of one cache group.
#[derive(Debug, Clone, Copy)]
pub struct GroupGeometry {
    block_granularity: i32,
}

impl GroupGeometry {
    /// Mamba keeps exactly the live state page plus its snapshot.
    pub const MAMBA_STATE_WINDOW: i32 = 2;

    pub fn new(block_granularity: i32) -> Self {
        assert!(block_granularity > 0, "block_granularity must be > 0");
        Self { block_granularity }
    }

    pub fn block_granularity(&self) -> i32 {
        self.block_granularity
    }

    /// Fresh blocks needed to cover `num_tokens` beyond the table's available
    /// tail capacity.
    pub fn blocks_needed_for(&self, table: &BlockTable, num_tokens: i32) -> i32 {
        if num_tokens <= table.available_tokens() {
            return 0;
        }
        let over = num_tokens - table.available_tokens();
        (over + self.block_granularity - 1) / self.block_granularity
    }

    /// Fresh blocks needed for a full demand (dense or sparse suffix).
    pub fn blocks_needed_for_demand(&self, table: &BlockTable, demand: &GroupDemand<'_>) -> i32 {
        if demand.materialized_suffix_start < 0 {
            self.blocks_needed_for(table, demand.num_tokens + demand.reserve_tokens)
        } else {
            self.sparse_suffix_blocks(table, demand)
        }
    }

    /// Plan placement for a demand (dense or sparse suffix).
    pub fn plan_acquire_demand(&self, table: &BlockTable, demand: &GroupDemand<'_>) -> AcquirePlan {
        if demand.materialized_suffix_start < 0 {
            return self.plan_acquire(table, demand.num_tokens, demand.reserve_tokens);
        }
        let num_blocks = self.sparse_suffix_blocks(table, demand);
        let extent = demand.num_tokens as i64 + demand.reserve_tokens as i64;
        let logical_blocks =
            ((extent + self.block_granularity as i64 - 1) / self.block_granularity as i64) as i32;
        AcquirePlan {
            num_blocks,
            suffix_start: demand.materialized_suffix_start,
            table_blocks_after: logical_blocks,
            available_tokens_after: logical_blocks * self.block_granularity - demand.num_tokens,
        }
    }

    /// Plan dense placement for `num_tokens` plus `reserve_tokens`.
    pub fn plan_acquire(
        &self,
        table: &BlockTable,
        num_tokens: i32,
        reserve_tokens: i32,
    ) -> AcquirePlan {
        assert!(
            num_tokens >= 0 && reserve_tokens >= 0,
            "token demand and reserve must be non-negative"
        );
        let num_blocks = self.blocks_needed_for(table, num_tokens + reserve_tokens);
        AcquirePlan {
            num_blocks,
            suffix_start: -1,
            table_blocks_after: table.num_blocks() + num_blocks,
            available_tokens_after: table.available_tokens() + num_blocks * self.block_granularity
                - num_tokens,
        }
    }

    /// Pages `[0, result)` of the table have fully expired under the group's
    /// retention policy at this progress; `kFull` never expires. This is where
    /// the sliding-window/state token policy meets page arithmetic.
    pub fn expired_blocks_at(&self, spec: &CacheGroupSpec, num_computed_tokens: i32) -> i32 {
        let window = match spec.kind {
            AttnKind::Full => return 0,
            AttnKind::SlidingWindow => spec.sliding_window,
            AttnKind::MambaState => Self::MAMBA_STATE_WINDOW,
        };
        assert!(window > 0, "retention window must be > 0");
        let skipped = num_computed_tokens - window + 1;
        // Only fully-slid-out pages expire.
        if skipped <= 0 {
            0
        } else {
            skipped / self.block_granularity
        }
    }

    /// Blocks to materialize for a sparse suffix demand (decode-side PD /
    /// snapshot-state restore).
    fn sparse_suffix_blocks(&self, table: &BlockTable, demand: &GroupDemand<'_>) -> i32 {
        // Decode-side prefix acquisition may have already installed aligned
        // null holes for state. They carry no ownership and remain safe to
        // extend sparsely up to the remote endpoint snapshot.
        assert!(
            table.available_tokens() == 0,
            "sparse suffix materialization requires a page boundary"
        );
        assert!(
            table.num_blocks() <= demand.materialized_suffix_start,
            "sparse suffix overlaps the existing block table"
        );
        assert!(
            demand.num_tokens > 0 && demand.reserve_tokens >= 0,
            "sparse suffix materialization requires a positive extent"
        );
        let extent = demand.num_tokens as i64 + demand.reserve_tokens as i64;
        assert!(
            extent <= i32::MAX as i64,
            "sparse suffix extent exceeds int32 range"
        );
        let last_block = ((extent - 1) / self.block_granularity as i64) as i32;
        assert!(
            demand.materialized_suffix_start <= last_block,
            "materialized suffix starts beyond the requested extent"
        );
        last_block - demand.materialized_suffix_start + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_table::BlockTable;

    #[test]
    fn blocks_needed_for_uses_available_tail() {
        let g = GroupGeometry::new(4);
        let table = BlockTable::from_blocks(Vec::new(), 2);
        assert_eq!(g.blocks_needed_for(&table, 1), 0); // within available tail
        assert_eq!(g.blocks_needed_for(&table, 2), 0);
        assert_eq!(g.blocks_needed_for(&table, 3), 1); // 1 over -> ceil(1/4)=1
        assert_eq!(g.blocks_needed_for(&table, 6), 1); // 4 over -> 1 block
        assert_eq!(g.blocks_needed_for(&table, 7), 2); // 5 over -> 2 blocks
    }

    #[test]
    fn plan_acquire_dense_tracks_available_tokens() {
        let g = GroupGeometry::new(4);
        let table = BlockTable::from_blocks(Vec::new(), 0);
        let plan = g.plan_acquire(&table, 5, 0);
        assert_eq!(plan.num_blocks, 2);
        assert_eq!(plan.suffix_start, -1);
        assert_eq!(plan.table_blocks_after, 2);
        assert_eq!(plan.available_tokens_after, 8 - 5);
    }

    #[test]
    fn plan_acquire_with_reserve() {
        let g = GroupGeometry::new(4);
        let table = BlockTable::from_blocks(Vec::new(), 0);
        let plan = g.plan_acquire(&table, 3, 2); // 5 tokens total
        assert_eq!(plan.num_blocks, 2);
        assert_eq!(plan.available_tokens_after, 8 - 3); // reserve not consumed
    }

    #[test]
    fn sparse_suffix_plan_materializes_suffix_only() {
        let g = GroupGeometry::new(4);
        let table = BlockTable::from_blocks(Vec::new(), 0);
        let demand = GroupDemand {
            table: None,
            num_tokens: 9,
            prefix_hashes: &[],
            new_prefix_hash_begin: 0,
            completed_boundary_kind: None,
            num_computed_tokens: -1,
            reserve_tokens: 0,
            materialized_suffix_start: 1,
        };
        // extent 9 -> last_block = 8/4 = 2 -> blocks = 2 - 1 + 1 = 2
        let plan = g.plan_acquire_demand(&table, &demand);
        assert_eq!(plan.num_blocks, 2);
        assert_eq!(plan.suffix_start, 1);
        assert_eq!(plan.table_blocks_after, 3);
        assert_eq!(plan.available_tokens_after, 12 - 9);
    }

    #[test]
    fn expired_blocks_at_full_never_expires() {
        let g = GroupGeometry::new(4);
        let spec = CacheGroupSpec {
            kind: AttnKind::Full,
            ..CacheGroupSpec::default()
        };
        assert_eq!(g.expired_blocks_at(&spec, 1000), 0);
    }

    #[test]
    fn expired_blocks_at_sliding_window() {
        let g = GroupGeometry::new(4);
        let spec = CacheGroupSpec {
            kind: AttnKind::SlidingWindow,
            sliding_window: 8,
            ..CacheGroupSpec::default()
        };
        // skipped = computed - 8 + 1; pages fully slid out = skipped / 4.
        assert_eq!(g.expired_blocks_at(&spec, 0), 0);
        assert_eq!(g.expired_blocks_at(&spec, 7), 0);
        assert_eq!(g.expired_blocks_at(&spec, 8), 0); // skipped=1 -> 0
        assert_eq!(g.expired_blocks_at(&spec, 11), 1); // skipped=4 -> 1
        assert_eq!(g.expired_blocks_at(&spec, 15), 2); // skipped=8 -> 2
    }

    #[test]
    fn expired_blocks_at_mamba_state_keeps_two() {
        let g = GroupGeometry::new(4);
        let spec = CacheGroupSpec {
            kind: AttnKind::MambaState,
            ..CacheGroupSpec::default()
        };
        assert_eq!(g.expired_blocks_at(&spec, 0), 0);
        assert_eq!(g.expired_blocks_at(&spec, 5), 1); // skipped=4 -> 1
    }
}
