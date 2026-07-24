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
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

enum class AttnKind { kFull, kSlidingWindow, kMambaState };

using CacheNamespaceId = std::uint32_t;
using ContentHash = std::string;

inline constexpr CacheNamespaceId kDefaultCacheNamespaceId = 0;

struct CacheKey {
    CacheNamespaceId namespace_id{kDefaultCacheNamespaceId};
    GroupId group_id{0};
    ContentHash content_hash{};

    bool operator==(const CacheKey&) const noexcept = default;
};

struct KvCacheSpec {
    AttnKind kind;
    std::int32_t sliding_window;  // 0 for full attention
    // Number of this group's logical cache blocks packed into one physical
    // LCM block. It affects placement only, never prefix-match granularity.
    std::int32_t cache_blocks_per_lcm_block{1};
};

// Per-request logical-page -> physical-page mapping.
class BlockTable {
public:
    std::span<const CacheBlockRef> Blocks() const noexcept { return blocks_; }
    std::int32_t NumBlocks() const { return static_cast<std::int32_t>(blocks_.size()); }
    std::int32_t AvailableTokens() const { return available_tokens_; }

    CacheBlockRef EvictToNull(std::int32_t index) {
        _assert(0 <= index && index < static_cast<std::int32_t>(blocks_.size()), "EvictToNull index out of range");
        return std::exchange(blocks_[static_cast<std::size_t>(index)], {});
    }

private:
    friend class KvCacheManager;

    std::vector<CacheBlockRef> blocks_{};
    // Unconsumed capacity at the logical tail. This may span multiple blocks
    // when admission preallocates a later decode/MTP step.
    std::int32_t available_tokens_{0};
};

// Per-group token demand for one admission. The table remains owned by the
// request; ProbeAdmission only reads it and Acquire appends to it.
struct GroupDemand {
    BlockTable* table{nullptr};
    std::int32_t num_tokens{0};
    std::span<const std::string> content_hashes{};
    std::int32_t first_page_slot{0};
    std::int32_t num_computed_tokens{-1};
    std::int32_t reserve_tokens{0};
};

// LCM ownership ids for scheduler accounting/debugging. Kernel-facing page
// tables must instead go through KvCacheManager::BlockTablePageIds().
inline std::vector<std::int32_t> BlockTableLcmBlockIds(const BlockTable& table) {
    std::vector<std::int32_t> ids;
    ids.reserve(static_cast<std::size_t>(table.NumBlocks()));
    for (const CacheBlockRef& block : table.Blocks()) {
        ids.push_back(block ? block->Location().lcm_block_id : 0);
    }
    return ids;
}

struct PrefixMatch {
    std::vector<CacheBlockRef> blocks{};
    std::int32_t num_hit_blocks{0};
};

// Non-owning match shape. A nonzero slot is acquired only after the coordinator
// converges every group to the final common boundary.
struct PrefixProbe {
    std::vector<std::uint8_t> hits{};
};

// Pinned source/destination pages for one asynchronous cache transfer.
struct BlockTransfer {
    GroupId group_id{0};
    CacheBlockRef source;
    CacheBlockRef destination;
};

}  // namespace tokenspeed
