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
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
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
    GroupAllocator(std::int32_t cache_blocks_per_lcm_block, std::uint32_t group_id, std::int32_t shard_count)
        : cache_blocks_per_lcm_block_{cache_blocks_per_lcm_block}, group_id_{group_id}, shard_count_{shard_count} {
        _assert(cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        _assert(shard_count > 0 && cache_blocks_per_lcm_block % shard_count == 0,
                "shard_count must be positive and divide parent packing");
    }

    GroupAllocator(const GroupAllocator&) = delete;
    GroupAllocator& operator=(const GroupAllocator&) = delete;

    std::int32_t CacheBlocksPerLcmBlock() const noexcept { return cache_blocks_per_lcm_block_; }
    std::uint32_t Id() const noexcept { return group_id_; }
    // Cyclic owners of this group's virtual blocks; 1 balances nothing.
    std::int32_t ShardCount() const noexcept { return shard_count_; }

    // Bare locations occur in recorded Device eviction segments. A live
    // reference carries its pool and must use the pool-aware overload below.
    std::int32_t ResolveDeviceCacheBlockId(CacheBlockLocation location) const {
        _assert(location.lcm_block_id > 0, "LCM block id must be > 0");
        _assert(0 <= location.slot_index && location.slot_index < cache_blocks_per_lcm_block_,
                "cache block slot is out of range");
        const std::int64_t page_id =
            1 + (static_cast<std::int64_t>(location.lcm_block_id) - 1) * cache_blocks_per_lcm_block_ +
            location.slot_index;
        _assert(page_id <= std::numeric_limits<std::int32_t>::max(), "kernel page id exceeds int32 range");
        return static_cast<std::int32_t>(page_id);
    }
    std::int32_t ResolveCacheBlockId(const CacheBlock& block) const {
        _assert(block.OwnerPool().BoundGroup(block.Location().lcm_block_id) == group_id_,
                "block reference belongs to another cache group");
        return block.OwnerPool().ResolveCacheBlockId(group_id_, block.Location());
    }
    std::vector<std::int32_t> BlockTablePageIds(const BlockTable& table) const {
        std::vector<std::int32_t> ids;
        ids.reserve(static_cast<std::size_t>(table.NumBlocks()));
        for (const CacheBlockRef& block_ref : table.Blocks()) {
            ids.push_back(block_ref ? ResolveCacheBlockId(*block_ref) : 0);
        }
        return ids;
    }

    // The table window handed to PrefixCacheIndex::RegisterFullBlocks. The
    // index owns no table: publishing a completed block may dedupe it against
    // the key's existing canonical block, and the table must then hold that
    // canonical reference, so the mutable window is issued here, by the one
    // component allowed to change a table.
    std::span<CacheBlockRef> BlocksToPublish(BlockTable& table, std::int32_t first_slot, std::size_t count) const {
        _assert(first_slot >= 0, "first_slot must be >= 0");
        _assert(static_cast<std::int64_t>(first_slot) + static_cast<std::int64_t>(count) <= table.NumBlocks(),
                "publish window exceeds table size");
        return std::span<CacheBlockRef>{table.blocks_}.subspan(static_cast<std::size_t>(first_slot), count);
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
            block_refs = pool.AcquireBlocks(group_id_, plan.num_blocks, BucketLoads(table));
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

    // The bucket a block's bytes belong to under page-cyclic sharding: the
    // residue of its slot inside the parent, which equals (virtual id - 1) %
    // shard_count because packing is a multiple of the shard count.
    std::int32_t BucketOf(CacheBlockLocation location) const noexcept { return location.slot_index % shard_count_; }

    // Rebuilds a table from a recorded shape: identical block count, null
    // holes, tail capacity and reclaimed prefix. `claimed` is parallel to
    // shape.slots: a non-empty entry is a Device block the coordinator claims
    // for that slot (a still-cached canonical block of the slot's key), every
    // empty entry gets one fresh block in the slot's bucket (read off its
    // Host block, which shares the Device block's residue). Returns false
    // without mutation when the pool cannot place the bucket demand. One
    // bucket degenerates to ordinary placement, so the shard count stays a
    // parameter of one path.
    bool AcquireShape(BlockPool& pool, BlockTable& table, const ImageTable& shape, std::vector<CacheBlockRef> claimed) {
        _assert(table.NumBlocks() == 0, "AcquireShape requires a fresh (empty) table");
        _assert(claimed.size() == shape.slots.size(), "claimed blocks must parallel the recorded slots");
        std::vector<std::int32_t> buckets;
        buckets.reserve(shape.slots.size());
        for (std::size_t i = 0; i < shape.slots.size(); ++i) {
            const ImageSlot& slot = shape.slots[i];
            _assert(slot.block && 0 <= slot.slot_index && slot.slot_index < shape.num_blocks,
                    "image slot must hold a block inside the recorded table");
            if (!claimed[i]) {
                buckets.push_back(BucketOf(slot.block->Location()));
            }
        }
        std::vector<CacheBlockRef> fresh = pool.AcquireBlocksInBuckets(group_id_, buckets);
        if (fresh.size() != buckets.size()) {
            return false;
        }
        table.blocks_.resize(static_cast<std::size_t>(shape.num_blocks));
        std::size_t next_fresh = 0;
        for (std::size_t i = 0; i < shape.slots.size(); ++i) {
            CacheBlockRef& destination = table.blocks_[static_cast<std::size_t>(shape.slots[i].slot_index)];
            destination = claimed[i] ? std::move(claimed[i]) : std::move(fresh[next_fresh++]);
        }
        table.available_tokens_ = shape.available_tokens;
        table.reclaimed_prefix_blocks_ = shape.reclaimed_prefix_blocks;
        return true;
    }

    // Swaps a fresh block into `slot`, in the bucket of the block it replaces
    // (any bucket when the slot was a hole), and sets the table's unconsumed
    // tail capacity. Returns the replaced block: a hitter's shared partial
    // block or an owner's just-published one, which the caller copies into
    // the fresh block; empty for a hole, which the caller zeroes instead.
    CacheBlockRef SwapInFreshBlock(BlockPool& pool, BlockTable& table, std::int32_t slot,
                                   std::int32_t available_tokens) {
        _assert(0 <= slot && slot < table.NumBlocks(), "swapped slot lies inside the table");
        _assert(available_tokens >= 0, "tail capacity must be non-negative");
        CacheBlockRef& current = table.blocks_[static_cast<std::size_t>(slot)];
        std::vector<CacheBlockRef> fresh;
        if (current) {
            const std::array<std::int32_t, 1> bucket{BucketOf(current->Location())};
            fresh = pool.AcquireBlocksInBuckets(group_id_, bucket);
        } else {
            fresh = pool.AcquireBlocks(group_id_, 1, BucketLoads(table));
        }
        FatalCheck(fresh.size() == 1, "admission plan no longer fits the block pool");
        CacheBlockRef replaced = std::exchange(current, std::move(fresh.front()));
        table.available_tokens_ = available_tokens;
        return replaced;
    }

    // Appends one Device destination per Host hit, each in its Host block's
    // bucket so the copy's two ends share an owner under page-cyclic sharding.
    // tail_available_tokens is the capacity left in the last appended block
    // when the Host hit ends inside it (0 on a block boundary).
    void AppendHostExtension(BlockPool& pool, BlockTable& table, std::vector<CacheBlockRef>&& host_block_refs,
                             std::vector<BlockTransfer>& load_pairs, std::int32_t tail_available_tokens) {
        _assert(table.available_tokens_ == 0, "host extension must append on a full-page boundary");
        _assert(tail_available_tokens >= 0, "tail capacity must be non-negative");
        std::vector<std::int32_t> buckets;
        for (const CacheBlockRef& block_ref : host_block_refs) {
            if (block_ref) {
                buckets.push_back(BucketOf(block_ref->Location()));
            }
        }
        table.blocks_.reserve(table.blocks_.size() + host_block_refs.size());
        std::vector<CacheBlockRef> destination_refs = pool.AcquireBlocksInBuckets(group_id_, buckets);
        FatalCheck(destination_refs.size() == buckets.size(), "admission plan no longer fits the block pool");
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
        table.available_tokens_ = tail_available_tokens;
    }

    // --- two-slot State tables -------------------------------------------
    // Slot 0 is the checkpoint the next forward reads, slot 1 the block it
    // writes. The table has exactly two slots once shaped.

    // Shapes a State table for its first admission: slot 0 the claimed hit
    // block (or null until the admission installs one), slot 1 empty until
    // AcquireStateLive.
    void ShapeStateTable(BlockTable& table, CacheBlockRef input) {
        _assert(table.blocks_.empty(), "ShapeStateTable requires a fresh (empty) table");
        table.blocks_.push_back(std::move(input));
        table.blocks_.emplace_back();
        table.available_tokens_ = 0;
        table.reclaimed_prefix_blocks_ = 0;
    }
    static void assertStateTable(const BlockTable& table) {
        _assert(table.NumBlocks() == 2, "a State table has exactly two slots");
    }
    // Drops the table's reference to slot 0; the index keeps the block when
    // it is published, otherwise it returns to the pool.
    void ReleaseStateInput(BlockTable& table) {
        assertStateTable(table);
        table.blocks_[0].reset();
    }
    // Slot 0 receives a block the admission acquired: a Host checkpoint's
    // fresh destination, or the zeroed block a prompt starts from.
    void InstallStateInput(BlockTable& table, CacheBlockRef input) {
        assertStateTable(table);
        _assert(!table.blocks_[0], "the input slot is installed only when empty");
        table.blocks_[0] = std::move(input);
    }
    // Slot 1 becomes slot 0 (the block the last forward wrote is the next
    // one's input); slot 1 is left empty for AcquireStateLive.
    void RotateState(BlockTable& table) {
        assertStateTable(table);
        table.blocks_[0] = std::exchange(table.blocks_[1], CacheBlockRef{});
    }
    bool AcquireStateLive(BlockPool& pool, BlockTable& table) {
        assertStateTable(table);
        _assert(!table.blocks_[1], "the live slot is acquired only once per step");
        return acquireStateSlot(pool, table, 1);
    }
    // A fresh block for an empty slot 0: the zero state a prompt starts from.
    bool AcquireStateInput(BlockPool& pool, BlockTable& table) {
        assertStateTable(table);
        _assert(!table.blocks_[0], "the input slot is acquired only when empty");
        return acquireStateSlot(pool, table, 0);
    }
    // A fresh block for an in-forward State checkpoint (StateStep::checkpoint),
    // placed like the table's own slots but held outside them; null when the
    // pool has none.
    CacheBlockRef AcquireStateCheckpoint(BlockPool& pool, const BlockTable& table) {
        assertStateTable(table);
        std::vector<CacheBlockRef> fresh = pool.AcquireBlocks(group_id_, 1, BucketLoads(table));
        return fresh.size() == 1 ? std::move(fresh.front()) : CacheBlockRef{};
    }
    // Slot 0's block when this step releases it and the table is its only
    // owner besides the index (reclaimable by the admission that releases it).
    std::optional<CacheBlockLocation> ReleasableStateInput(const PrefixCacheIndex& index, const BlockTable& table,
                                                           const StateStep& step) const {
        if (step.kind == StateStep::Kind::kKeep || table.NumBlocks() == 0 || !table.Blocks()[0]) {
            return std::nullopt;
        }
        const CacheBlockRef& block = table.Blocks()[0];
        const bool cached = index.Contains(block);
        if ((cached && block.use_count() == 2) || (!cached && block.unique())) {
            return block->Location();
        }
        return std::nullopt;
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
    bool acquireStateSlot(BlockPool& pool, BlockTable& table, std::size_t slot) {
        std::vector<CacheBlockRef> fresh = pool.AcquireBlocks(group_id_, 1, BucketLoads(table));
        if (fresh.size() != 1) {
            return false;
        }
        table.blocks_[slot] = std::move(fresh.front());
        return true;
    }
    std::vector<std::int32_t> BucketLoads(const BlockTable& table) const {
        if (shard_count_ == 1) {
            return {};
        }
        std::vector<std::int32_t> loads(static_cast<std::size_t>(shard_count_), 0);
        // Shared prefix refs and allocated headroom both count. Null slots in
        // sparse tables do not own a placement and contribute nothing.
        for (const CacheBlockRef& ref : table.Blocks()) {
            if (ref) {
                ++loads[static_cast<std::size_t>(ref->Location().slot_index % shard_count_)];
            }
        }
        return loads;
    }

    std::int32_t cache_blocks_per_lcm_block_;
    std::uint32_t group_id_;
    std::int32_t shard_count_;
};

}  // namespace tokenspeed
