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
#include <iterator>
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
    // kept_head_blocks: the leading table slots retention never releases (a
    // sliding group's sink blocks); 0 for every other group.
    GroupAllocator(std::int32_t cache_blocks_per_lcm_block, std::uint32_t group_id, std::int32_t shard_count,
                   std::int32_t kept_head_blocks)
        : cache_blocks_per_lcm_block_{cache_blocks_per_lcm_block},
          group_id_{group_id},
          shard_count_{shard_count},
          kept_head_blocks_{kept_head_blocks} {
        _assert(cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        _assert(shard_count > 0 && cache_blocks_per_lcm_block % shard_count == 0,
                "shard_count must be positive and divide parent packing");
        _assert(kept_head_blocks >= 0, "kept_head_blocks must be non-negative");
    }

    GroupAllocator(const GroupAllocator&) = delete;
    GroupAllocator& operator=(const GroupAllocator&) = delete;

    std::int32_t CacheBlocksPerLcmBlock() const noexcept { return cache_blocks_per_lcm_block_; }
    std::uint32_t Id() const noexcept { return group_id_; }
    // Cyclic owners of this group's virtual blocks; 1 balances nothing.
    std::int32_t ShardCount() const noexcept { return shard_count_; }
    std::int32_t KeptHeadBlocks() const noexcept { return kept_head_blocks_; }

    // The table slots retention has released, [first, past_end): every slot
    // below the table's reclaim frontier except the kept head. All are null.
    struct SlotRange {
        std::int32_t first{0};
        std::int32_t past_end{0};
    };
    SlotRange ReleasedSlots(const BlockTable& table) const noexcept {
        const std::int32_t frontier = table.ReclaimedPrefixBlocks();
        return SlotRange{.first = std::min(kept_head_blocks_, frontier), .past_end = frontier};
    }
    // The released slots a table starts with: an export skips them and they
    // hold no parent. A table that keeps a head block starts with none.
    std::int32_t LeadingReleasedBlocks(const BlockTable& table) const noexcept {
        const SlotRange released = ReleasedSlots(table);
        return released.first == 0 ? released.past_end : 0;
    }

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
        AppendBlockTablePageIds(table, ids, 0);
        return ids;
    }

    // Append the row from first_slot on, preserving null holes in that range.
    void AppendBlockTablePageIds(const BlockTable& table, std::vector<std::int32_t>& ids,
                                 std::size_t first_slot) const {
        assertPageIdsCurrent(table);
        const SlotRange slots = ReleasedSlots(table);
        const auto released_first = static_cast<std::size_t>(slots.first);
        const auto released_past_end = static_cast<std::size_t>(slots.past_end);
        const std::span<const std::int32_t> page_ids = table.PageIds();
        _assert(first_slot <= page_ids.size(), "block-table range starts beyond its row");
        // The kept head, the released slots (null, so not read), the rest.
        if (first_slot < released_first) {
            ids.insert(ids.end(), page_ids.begin() + static_cast<std::ptrdiff_t>(first_slot),
                       page_ids.begin() + static_cast<std::ptrdiff_t>(released_first));
        }
        const std::size_t zeros_from = std::max(first_slot, released_first);
        if (zeros_from < released_past_end) ids.insert(ids.end(), released_past_end - zeros_from, 0);
        ids.insert(ids.end(), page_ids.begin() + static_cast<std::ptrdiff_t>(std::max(first_slot, released_past_end)),
                   page_ids.end());
    }

    // The table window handed to PrefixCacheIndex::RegisterFullBlocks. The
    // index owns no table: publishing a completed block may dedupe it against
    // the key's existing canonical block, and the table must then hold that
    // canonical reference, so the mutable window is issued here, by the one
    // component allowed to change a table. The window's page ids follow the
    // references the index leaves when it closes.
    BlockTable::PublishWindow BlocksToPublish(BlockTable& table, std::int32_t first_slot, std::size_t count) const {
        _assert(first_slot >= 0, "first_slot must be >= 0");
        _assert(static_cast<std::int64_t>(first_slot) + static_cast<std::int64_t>(count) <= table.NumBlocks(),
                "publish window exceeds table size");
        return BlockTable::PublishWindow{group_id_, table, static_cast<std::size_t>(first_slot), count};
    }

    void ClaimHitBlocks(BlockTable& table, PrefixMatch&& hit) {
        _assert(table.blocks_.empty(), "ClaimHitBlocks requires a fresh (empty) table");
        table.assignAll(group_id_, std::move(hit.blocks));
        while (table.reclaimed_prefix_blocks_ < table.NumBlocks() &&
               !table.blocks_[static_cast<std::size_t>(table.reclaimed_prefix_blocks_)]) {
            ++table.reclaimed_prefix_blocks_;
        }
    }

    // Executes a GroupGeometry plan: acquires plan.num_blocks fresh blocks,
    // places them (dense append or sparse suffix), and stores the planned
    // bookkeeping. A dense append may be handed `placed`: blocks the caller
    // acquired in the buckets their slots must take, for table slots
    // [first_placed_slot, first_placed_slot + placed.size()) of the appended
    // run, so that every bucket-exact block is acquired before the plan's
    // other blocks take any hole (the order the admission planner's fit
    // assumes). Returns false when the pool is short of the other blocks,
    // without mutating the table; the placed blocks go back then.
    bool Acquire(BlockPool& pool, BlockTable& table, const AcquirePlan& plan, std::int32_t first_placed_slot,
                 std::vector<CacheBlockRef> placed) {
        const std::int32_t old_num_blocks = table.NumBlocks();
        _assert(placed.empty() || (plan.suffix_start < 0 && first_placed_slot >= old_num_blocks &&
                                   first_placed_slot + static_cast<std::int64_t>(placed.size()) <=
                                       static_cast<std::int64_t>(old_num_blocks) + plan.num_blocks),
                "placed blocks fill slots of a dense append");
        const std::int32_t num_others = plan.num_blocks - static_cast<std::int32_t>(placed.size());
        std::vector<CacheBlockRef> block_refs;
        if (num_others > 0) {
            // The placed blocks count toward the request's bucket loads too.
            std::vector<std::int32_t> loads = BucketLoads(table);
            for (const CacheBlockRef& block : placed) {
                if (!loads.empty()) {
                    ++loads[static_cast<std::size_t>(block->Location().slot_index % shard_count_)];
                }
            }
            block_refs = pool.AcquireBlocks(group_id_, num_others, loads);
            if (static_cast<std::int32_t>(block_refs.size()) < num_others) {
                return false;
            }
        }
        if (!placed.empty()) {
            // Slot order: the placed run sits among the other appended slots.
            const auto before = static_cast<std::ptrdiff_t>(first_placed_slot - old_num_blocks);
            block_refs.insert(block_refs.begin() + before, std::make_move_iterator(placed.begin()),
                              std::make_move_iterator(placed.end()));
        }
        if (plan.suffix_start < 0) {
            // Insert the batch together so allocation failure cannot leave a partial append.
            table.appendAll(group_id_, std::move(block_refs));
        } else {
            _assert(table.NumBlocks() <= plan.suffix_start, "sparse suffix overlaps the existing block table");
            _assert(plan.suffix_start + plan.num_blocks - plan.head_blocks <= plan.table_blocks_after,
                    "sparse suffix exceeds the planned table size");
            table.resize(static_cast<std::size_t>(plan.table_blocks_after));
            if (old_num_blocks == 0) {
                table.reclaimed_prefix_blocks_ = plan.suffix_start;
            }
            for (std::size_t i = 0; i < block_refs.size(); ++i) {
                const auto slot = i < static_cast<std::size_t>(plan.head_blocks)
                                      ? static_cast<std::size_t>(old_num_blocks) + i
                                      : static_cast<std::size_t>(plan.suffix_start) + i - plan.head_blocks;
                table.assign(group_id_, slot, std::move(block_refs[i]));
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
        table.resize(static_cast<std::size_t>(shape.num_blocks));
        std::size_t next_fresh = 0;
        for (std::size_t i = 0; i < shape.slots.size(); ++i) {
            table.assign(group_id_, static_cast<std::size_t>(shape.slots[i].slot_index),
                         claimed[i] ? std::move(claimed[i]) : std::move(fresh[next_fresh++]));
        }
        table.available_tokens_ = shape.available_tokens;
        table.reclaimed_prefix_blocks_ = shape.reclaimed_prefix_blocks;
        return true;
    }

    // Swaps a fresh block into `slot`, in the bucket of the block it replaces
    // (any bucket when the slot was a hole), and sets the table's unconsumed
    // tail capacity. Returns the replaced block: a hitter's shared partial
    // block, which the caller copies into the fresh block; empty for a hole,
    // which the caller zeroes instead. Installing into a hole ends the released
    // prefix at this slot, so subsequent exports include the new writable block.
    // With a destination the fresh block is that free slot of a parent bound to
    // this group (a lone request's compaction moving its own block; the caller
    // keeps a copied block's bucket, and may change an unwritten block's);
    // without one the pool places it.
    CacheBlockRef SwapInFreshBlock(BlockPool& pool, BlockTable& table, std::int32_t slot, std::int32_t available_tokens,
                                   std::optional<CacheBlockLocation> destination) {
        _assert(0 <= slot && slot < table.NumBlocks(), "swapped slot lies inside the table");
        _assert(available_tokens >= 0, "tail capacity must be non-negative");
        const CacheBlockRef& current = table.blocks_[static_cast<std::size_t>(slot)];
        std::vector<CacheBlockRef> fresh;
        if (destination) {
            fresh.push_back(pool.AcquireBlockAt(group_id_, *destination));
        } else if (current) {
            const std::array<std::int32_t, 1> bucket{BucketOf(current->Location())};
            fresh = pool.AcquireBlocksInBuckets(group_id_, bucket);
        } else {
            fresh = pool.AcquireBlocks(group_id_, 1, BucketLoads(table));
        }
        FatalCheck(fresh.size() == 1, "admission plan no longer fits the block pool");
        CacheBlockRef replaced = table.exchange(group_id_, static_cast<std::size_t>(slot), std::move(fresh.front()));
        table.available_tokens_ = available_tokens;
        table.reclaimed_prefix_blocks_ = std::min(table.reclaimed_prefix_blocks_, slot);
        return replaced;
    }

    // Swaps `block` -- an owner's copy-in block, held since the admission that
    // reserved it in the bucket of the block in `slot` -- into `slot`. Returns
    // the replaced block, the owner's just-published one, which the caller
    // copies into `block`. The table's tail capacity is unchanged.
    CacheBlockRef SwapInBlock(BlockTable& table, std::int32_t slot, CacheBlockRef block) {
        _assert(0 <= slot && slot < table.NumBlocks(), "swapped slot lies inside the table");
        const CacheBlockRef& current = table.blocks_[static_cast<std::size_t>(slot)];
        _assert(current && block && BucketOf(current->Location()) == BucketOf(block->Location()),
                "a copy-in block shares the bucket of the block it replaces");
        return table.exchange(group_id_, static_cast<std::size_t>(slot), std::move(block));
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
        table.reserve(table.blocks_.size() + host_block_refs.size());
        std::vector<CacheBlockRef> destination_refs = pool.AcquireBlocksInBuckets(group_id_, buckets);
        FatalCheck(destination_refs.size() == buckets.size(), "admission plan no longer fits the block pool");
        auto destination_it = destination_refs.begin();
        for (CacheBlockRef& host_block_ref : host_block_refs) {
            if (!host_block_ref) {
                table.append(group_id_, {});
                continue;
            }
            _assert(destination_it != destination_refs.end(), "missing host extension destination");
            table.append(group_id_, std::move(*destination_it));
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
        table.append(group_id_, std::move(input));
        table.append(group_id_, {});
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
        table.take(0).reset();
    }
    // Slot 0 receives a block the admission acquired: a Host checkpoint's
    // fresh destination, or the zeroed block a prompt starts from.
    void InstallStateInput(BlockTable& table, CacheBlockRef input) {
        assertStateTable(table);
        _assert(!table.blocks_[0], "the input slot is installed only when empty");
        table.assign(group_id_, 0, std::move(input));
    }
    // Slot 1 becomes slot 0 (the block the last forward wrote is the next
    // one's input); slot 1 is left empty for AcquireStateLive.
    void RotateState(BlockTable& table) {
        assertStateTable(table);
        table.assign(group_id_, 0, table.take(1));
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

    // Retention execution: the first num_expired_blocks table slots, except
    // the kept head, become null holes, so the table never shrinks and slot
    // alignment stays stable. How many blocks expired is retention policy
    // (GroupGeometry).
    void ReclaimExpired(BlockPool& /*pool*/, BlockTable& table, std::int32_t num_expired_blocks) {
        const std::int32_t expired = std::min(num_expired_blocks, table.NumBlocks());
        for (std::int32_t i = std::max(table.reclaimed_prefix_blocks_, kept_head_blocks_); i < expired; ++i) {
            table.EvictToNull(i).reset();
        }
        table.reclaimed_prefix_blocks_ = std::max(table.reclaimed_prefix_blocks_, expired);
    }

    // Unreclaimed slots below the retention frontier, including null holes
    // and shared blocks; the kept head is never among them. The size is an
    // optimistic reclaim bound; the actual candidates must come from this same
    // range and pass the ownership check.
    std::span<const CacheBlockRef> ExpiredBlockSlotsAt(const BlockTable& table, std::int32_t num_expired_blocks) const {
        const std::int32_t first = std::max(table.ReclaimedPrefixBlocks(), kept_head_blocks_);
        const std::int32_t count = std::max(std::min(num_expired_blocks, table.NumBlocks()) - first, 0);
        return count == 0 ? std::span<const CacheBlockRef>{} : table.Blocks().subspan(first, count);
    }

    // Only blocks uniquely owned by this table reach the free list, so shared ones don't count.
    std::int32_t BlocksReclaimableAt(const PrefixCacheIndex& index, const BlockTable& table,
                                     std::int32_t num_expired_blocks, bool count_uncached) const {
        std::int32_t freed = 0;
        for (const CacheBlockRef& block : ExpiredBlockSlotsAt(table, num_expired_blocks)) {
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
        std::vector<CacheBlockLocation> locations;
        for (const CacheBlockRef& block : ExpiredBlockSlotsAt(table, num_expired_blocks)) {
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
        table.releaseAll();
        table.available_tokens_ = 0;
        table.reclaimed_prefix_blocks_ = 0;
    }

private:
    // Debug builds re-resolve every live slot the way the export did before
    // tables kept their page ids, and fail on any slot a write left stale.
    // They also check that the released prefix holds no block: the export
    // skips it, and the slots' ParentRefs count every non-null slot.
    void assertPageIdsCurrent([[maybe_unused]] const BlockTable& table) const {
#ifndef NDEBUG
        const std::span<const CacheBlockRef> blocks = table.Blocks();
        const std::span<const std::int32_t> page_ids = table.PageIds();
        _assert(blocks.size() == page_ids.size(), "a table keeps one page id per slot");
        const SlotRange released = ReleasedSlots(table);
        for (std::size_t slot = 0; slot < blocks.size(); ++slot) {
            if (static_cast<std::size_t>(released.first) <= slot &&
                slot < static_cast<std::size_t>(released.past_end)) {
                _assert(!blocks[slot], "a released slot still holds a block");
                continue;
            }
            const std::int32_t expected = blocks[slot] ? ResolveCacheBlockId(*blocks[slot]) : 0;
            _assert(page_ids[slot] == expected, "a table's page id went stale");
        }
#endif
    }

    bool acquireStateSlot(BlockPool& pool, BlockTable& table, std::size_t slot) {
        std::vector<CacheBlockRef> fresh = pool.AcquireBlocks(group_id_, 1, BucketLoads(table));
        if (fresh.size() != 1) {
            return false;
        }
        table.assign(group_id_, slot, std::move(fresh.front()));
        return true;
    }
    std::vector<std::int32_t> BucketLoads(const BlockTable& table) const {
        if (shard_count_ == 1) {
            return {};
        }
        std::vector<std::int32_t> loads(static_cast<std::size_t>(shard_count_), 0);
        // Shared prefix refs and allocated headroom both count. Null slots in
        // sparse tables do not own a placement and contribute nothing.
        for (const CacheBlockRef& ref : table.Blocks().subspan(table.ReclaimedPrefixBlocks())) {
            if (ref) {
                ++loads[static_cast<std::size_t>(ref->Location().slot_index % shard_count_)];
            }
        }
        return loads;
    }

    std::int32_t cache_blocks_per_lcm_block_;
    std::uint32_t group_id_;
    std::int32_t shard_count_;
    std::int32_t kept_head_blocks_;
};

}  // namespace tokenspeed
