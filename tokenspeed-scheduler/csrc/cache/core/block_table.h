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

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "cache/core/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

// Per-request logical-page -> physical-page mapping.
class BlockTable {
public:
    static BlockTable FromBlocks(std::vector<CacheBlockRef> blocks, std::int32_t available_tokens) {
        _assert(available_tokens >= 0, "BlockTable available_tokens must be non-negative");
        BlockTable table;
        table.blocks_ = std::move(blocks);
        table.available_tokens_ = available_tokens;
        return table;
    }

    std::span<const CacheBlockRef> Blocks() const noexcept { return blocks_; }
    std::int32_t NumBlocks() const { return static_cast<std::int32_t>(blocks_.size()); }
    std::int32_t AvailableTokens() const { return available_tokens_; }
    std::int32_t ReclaimedPrefixBlocks() const { return reclaimed_prefix_blocks_; }

    CacheBlockRef EvictToNull(std::int32_t index) {
        _assert(0 <= index && index < static_cast<std::int32_t>(blocks_.size()), "EvictToNull index out of range");
        return std::exchange(blocks_[static_cast<std::size_t>(index)], {});
    }

private:
    friend class GroupAllocator;
    // Registering a block may replace it with the key's canonical block.
    friend class PrefixCacheIndex;

    std::vector<CacheBlockRef> blocks_{};
    // Unconsumed capacity at the logical tail. This may span multiple blocks
    // when admission preallocates a later decode/MTP step.
    std::int32_t available_tokens_{0};
    // Slots below this monotonic frontier have already released their request
    // ownership. Sparse state tables may contain holes between live islands,
    // so reclaim cannot infer this frontier from the first null slot.
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
