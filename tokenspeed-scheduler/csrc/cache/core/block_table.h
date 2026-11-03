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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <utility>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

class GroupAllocator;

// The kernel page id of a block in `group_id`'s tables, 0 for a null slot.
// The group's packing places the block in its parent and cannot change while
// the reference keeps the parent occupied, so a table resolves it once.
inline std::int32_t ResolvedPageId(std::uint32_t group_id, const CacheBlockRef& block) {
    if (!block) {
        return 0;
    }
    const BlockPool& pool = block->OwnerPool();
    const CacheBlockLocation location = block->Location();
    _assert(pool.BoundGroup(location.lcm_block_id) == group_id, "block reference belongs to another cache group");
    return pool.ResolveCacheBlockId(group_id, location);
}

// Per-request logical-page -> physical-page mapping. Beside each slot's block
// the table keeps the block's kernel page id, resolved when the block enters
// the slot: a forward batch exports every scheduled row each round, and
// resolving every block of a long context again each round was most of the
// round's work.
class BlockTable {
public:
    static BlockTable FromBlocks(std::uint32_t group_id, std::vector<CacheBlockRef> blocks,
                                 std::int32_t available_tokens) {
        _assert(available_tokens >= 0, "BlockTable available_tokens must be non-negative");
        BlockTable table;
        table.assignAll(group_id, std::move(blocks));
        table.available_tokens_ = available_tokens;
        return table;
    }

    std::span<const CacheBlockRef> Blocks() const noexcept { return blocks_; }
    // Each slot's kernel page id, parallel to Blocks(): 0 for a null slot.
    std::span<const std::int32_t> PageIds() const noexcept { return page_ids_; }
    std::int32_t NumBlocks() const { return static_cast<std::int32_t>(blocks_.size()); }
    std::int32_t AvailableTokens() const { return available_tokens_; }
    std::int32_t ReclaimedPrefixBlocks() const { return reclaimed_prefix_blocks_; }

    CacheBlockRef EvictToNull(std::int32_t index) {
        _assert(0 <= index && index < static_cast<std::int32_t>(blocks_.size()), "EvictToNull index out of range");
        return take(static_cast<std::size_t>(index));
    }

private:
    // The allocator is the only component that changes a table; publishing
    // its blocks to a prefix index reads a window from it. Every write goes
    // through the members below, which keep page_ids_ in step with blocks_.
    friend class GroupAllocator;

    void assign(std::uint32_t group_id, std::size_t slot, CacheBlockRef block) {
        page_ids_[slot] = ResolvedPageId(group_id, block);
        blocks_[slot] = std::move(block);
    }
    CacheBlockRef exchange(std::uint32_t group_id, std::size_t slot, CacheBlockRef block) {
        page_ids_[slot] = ResolvedPageId(group_id, block);
        return std::exchange(blocks_[slot], std::move(block));
    }
    // Empties the slot, returning its block.
    CacheBlockRef take(std::size_t slot) noexcept {
        page_ids_[slot] = 0;
        return std::exchange(blocks_[slot], {});
    }
    void append(std::uint32_t group_id, CacheBlockRef block) {
        page_ids_.push_back(ResolvedPageId(group_id, block));
        blocks_.push_back(std::move(block));
    }
    void appendAll(std::uint32_t group_id, std::vector<CacheBlockRef>&& blocks) {
        page_ids_.reserve(page_ids_.size() + blocks.size());
        for (const CacheBlockRef& block : blocks) {
            page_ids_.push_back(ResolvedPageId(group_id, block));
        }
        blocks_.insert(blocks_.end(), std::make_move_iterator(blocks.begin()), std::make_move_iterator(blocks.end()));
    }
    void assignAll(std::uint32_t group_id, std::vector<CacheBlockRef>&& blocks) {
        blocks_ = std::move(blocks);
        page_ids_.clear();
        page_ids_.reserve(blocks_.size());
        for (const CacheBlockRef& block : blocks_) {
            page_ids_.push_back(ResolvedPageId(group_id, block));
        }
    }
    // New slots are null.
    void resize(std::size_t slots) {
        blocks_.resize(slots);
        page_ids_.resize(slots, 0);
    }
    void reserve(std::size_t slots) {
        blocks_.reserve(slots);
        page_ids_.reserve(slots);
    }
    // Releases the logical suffix first, so newly emptied LCM parents enter
    // the FIFO free queue in deterministic table order.
    void releaseAll() noexcept {
        for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
            it->reset();
        }
        blocks_.clear();
        page_ids_.clear();
    }
    std::vector<CacheBlockRef> blocks_{};
    // page_ids_[i] is blocks_[i]'s kernel page id, 0 for a null slot.
    std::vector<std::int32_t> page_ids_{};
    // Unconsumed capacity at the logical tail. This may span multiple blocks
    // when admission preallocates a later decode/MTP step.
    std::int32_t available_tokens_{0};
    // Slots below this monotonic frontier have already released their request
    // ownership. A sliding group's table keeps a null hole at every slid-out
    // slot, so reclaim cannot infer this frontier from the first null slot.
    std::int32_t reclaimed_prefix_blocks_{0};
};

// LCM ownership ids for scheduler accounting/debugging. Kernel-facing page
// tables must instead go through GroupAllocator::BlockTablePageIds().
inline std::vector<std::int32_t> BlockTableLcmBlockIds(const BlockTable& table) {
    std::vector<std::int32_t> ids;
    ids.reserve(static_cast<std::size_t>(table.NumBlocks()));
    for (const CacheBlockRef& block_ref : table.Blocks()) {
        ids.push_back(block_ref ? block_ref->Location().lcm_block_id : 0);
    }
    return ids;
}

}  // namespace tokenspeed
