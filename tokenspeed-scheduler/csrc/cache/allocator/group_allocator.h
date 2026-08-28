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

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "cache/core/acquire_plan.h"
#include "cache/core/block_pool.h"
#include "cache/core/block_table.h"
#include "cache/core/cache_block_ref.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/prefix_index.h"
#include "utils.h"

namespace tokenspeed {

// One cache group's physical placement: it moves CacheBlocks between the
// BlockPool and BlockTables and resolves kernel page ids. It is deliberately
// token-free -- it perceives no prefix_granularity, block_granularity, or window;
// every token quantity is converted to block counts by GroupGeometry in the
// coordinator layer before it reaches this class. Prefix reuse lives in the
// group's PrefixCacheIndex, match policy in its PrefixMatcher.
class GroupAllocator {
public:
    explicit GroupAllocator(std::int32_t cache_blocks_per_lcm_block = 1, std::uint32_t group_id = 0)
        : cache_blocks_per_lcm_block_{cache_blocks_per_lcm_block}, group_id_{group_id} {
        _assert(cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
    }

    GroupAllocator(const GroupAllocator&) = delete;
    GroupAllocator& operator=(const GroupAllocator&) = delete;

    std::int32_t CacheBlocksPerLcmBlock() const noexcept { return cache_blocks_per_lcm_block_; }
    std::uint32_t Id() const noexcept { return group_id_; }

    std::int32_t ResolveCacheBlockId(CacheBlockLocation location) const {
        _assert(location.lcm_block_id > 0, "LCM block id must be > 0");
        _assert(0 <= location.slot_index && location.slot_index < cache_blocks_per_lcm_block_,
                "cache block slot is out of range");
        const std::int64_t page_id =
            1 + (static_cast<std::int64_t>(location.lcm_block_id) - 1) * cache_blocks_per_lcm_block_ +
            location.slot_index;
        _assert(page_id <= std::numeric_limits<std::int32_t>::max(), "kernel page id exceeds int32 range");
        return static_cast<std::int32_t>(page_id);
    }
    std::vector<std::int32_t> BlockTablePageIds(const BlockTable& table) const {
        std::vector<std::int32_t> ids;
        ids.reserve(static_cast<std::size_t>(table.NumBlocks()));
        for (const CacheBlockRef& block_ref : table.Blocks()) {
            ids.push_back(block_ref ? ResolveCacheBlockId(block_ref->Location()) : 0);
        }
        return ids;
    }

    void ClaimHitBlocks(BlockTable& table, PrefixMatch&& hit) {
        _assert(table.blocks_.empty(), "ClaimHitBlocks requires a fresh (empty) table");
        table.blocks_ = std::move(hit.blocks);
        while (table.reclaimed_prefix_blocks_ < table.NumBlocks() &&
               !table.blocks_[static_cast<std::size_t>(table.reclaimed_prefix_blocks_)]) {
            ++table.reclaimed_prefix_blocks_;
        }
    }

    // Executes a GroupGeometry plan: acquires plan.num_blocks fresh blocks,
    // places them (dense append or sparse suffix), and stores the planned
    // bookkeeping. Returns false without mutation when the pool is short.
    bool Acquire(BlockPool& pool, BlockTable& table, const AcquirePlan& plan) {
        const std::int32_t old_num_blocks = table.NumBlocks();
        std::vector<CacheBlockRef> block_refs;
        if (plan.num_blocks > 0) {
            block_refs = pool.AcquireBlocks(group_id_, cache_blocks_per_lcm_block_, plan.num_blocks);
            if (static_cast<std::int32_t>(block_refs.size()) < plan.num_blocks) {
                return false;
            }
        }
        if (plan.suffix_start < 0) {
            table.blocks_.reserve(table.blocks_.size() + block_refs.size());
            for (CacheBlockRef& block_ref : block_refs) {
                table.blocks_.push_back(std::move(block_ref));
            }
        } else {
            _assert(table.NumBlocks() <= plan.suffix_start, "sparse suffix overlaps the existing block table");
            _assert(plan.suffix_start + plan.num_blocks <= plan.table_blocks_after,
                    "sparse suffix exceeds the planned table size");
            table.blocks_.resize(static_cast<std::size_t>(plan.table_blocks_after));
            if (old_num_blocks == 0) {
                table.reclaimed_prefix_blocks_ = plan.suffix_start;
            }
            for (std::size_t i = 0; i < block_refs.size(); ++i) {
                table.blocks_[static_cast<std::size_t>(plan.suffix_start) + i] = std::move(block_refs[i]);
            }
        }
        table.available_tokens_ = plan.available_tokens_after;
        return true;
    }

    void AppendHostExtension(BlockPool& pool, BlockTable& table, std::vector<CacheBlockRef>&& host_block_refs,
                             std::vector<BlockTransfer>& load_pairs) {
        _assert(table.available_tokens_ == 0, "host extension must append on a full-page boundary");
        const std::int32_t num_pages = static_cast<std::int32_t>(std::ranges::count_if(
            host_block_refs, [](const CacheBlockRef& block_ref) { return static_cast<bool>(block_ref); }));
        table.blocks_.reserve(table.blocks_.size() + host_block_refs.size());
        std::vector<CacheBlockRef> destination_refs =
            pool.AcquireBlocks(group_id_, cache_blocks_per_lcm_block_, num_pages);
        FatalCheck(static_cast<std::int32_t>(destination_refs.size()) == num_pages,
                   "admission plan no longer fits the block pool");
        auto destination_it = destination_refs.begin();
        for (CacheBlockRef& host_block_ref : host_block_refs) {
            if (!host_block_ref) {
                table.blocks_.emplace_back();
                continue;
            }
            _assert(destination_it != destination_refs.end(), "missing host extension destination");
            table.blocks_.push_back(std::move(*destination_it));
            ++destination_it;
            load_pairs.push_back(BlockTransfer{
                .group_id = group_id_,
                .source = std::move(host_block_ref),
                .destination = table.blocks_.back(),
            });
        }
        _assert(destination_it == destination_refs.end(), "unused host extension destination");
    }

    // Retention execution: the first num_expired_blocks table slots become
    // null holes, so the table never shrinks and slot alignment stays stable.
    // How many blocks expired is retention policy (GroupGeometry).
    void ReclaimExpired(BlockPool& /*pool*/, BlockTable& table, std::int32_t num_expired_blocks) {
        const std::int32_t expired = std::min(num_expired_blocks, table.NumBlocks());
        for (std::int32_t i = table.reclaimed_prefix_blocks_; i < expired; ++i) {
            table.EvictToNull(i).reset();
        }
        table.reclaimed_prefix_blocks_ = std::max(table.reclaimed_prefix_blocks_, expired);
    }

    // Only blocks uniquely owned by this table reach the free list, so shared ones don't count.
    std::int32_t BlocksReclaimableAt(const PrefixCacheIndex& index, const BlockTable& table,
                                     std::int32_t num_expired_blocks, bool count_uncached) const {
        const std::int32_t expired = std::min(num_expired_blocks, table.NumBlocks());
        std::int32_t freed = 0;
        for (std::int32_t i = table.ReclaimedPrefixBlocks(); i < expired; ++i) {
            const CacheBlockRef& block = table.Blocks()[static_cast<std::size_t>(i)];
            if (!block) {
                continue;
            }
            const bool cached = index.Contains(block);
            const bool only_table_and_cache_owners = cached && block.use_count() == 2;
            if (only_table_and_cache_owners || (count_uncached && !cached && block.unique())) {
                ++freed;
            }
        }
        return freed;
    }

    std::vector<CacheBlockLocation> ReclaimableBlockLocationsAt(const PrefixCacheIndex& index, const BlockTable& table,
                                                                std::int32_t num_expired_blocks) const {
        const std::int32_t expired = std::min(num_expired_blocks, table.NumBlocks());
        std::vector<CacheBlockLocation> locations;
        for (std::int32_t i = table.ReclaimedPrefixBlocks(); i < expired; ++i) {
            const CacheBlockRef& block = table.Blocks()[static_cast<std::size_t>(i)];
            if (!block) {
                continue;
            }
            const bool cached = index.Contains(block);
            if ((cached && block.use_count() == 2) || (!cached && block.unique())) {
                locations.push_back(block->Location());
            }
        }
        return locations;
    }

    void ConsumeReservedTokens(BlockTable& table, std::int32_t num_tokens) {
        _assert(num_tokens >= 0 && num_tokens <= table.available_tokens_,
                "token demand exceeds the available capacity");
        table.available_tokens_ -= num_tokens;
    }

    void Free(BlockTable& table) {
        // Release the logical suffix first so newly emptied LCM parents enter
        // the FIFO free queue in deterministic table order.
        for (auto it = table.blocks_.rbegin(); it != table.blocks_.rend(); ++it) {
            it->reset();
        }
        table.blocks_.clear();
        table.available_tokens_ = 0;
        table.reclaimed_prefix_blocks_ = 0;
    }

private:
    std::int32_t cache_blocks_per_lcm_block_;
    std::uint32_t group_id_;
};

}  // namespace tokenspeed
