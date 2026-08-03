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
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

enum class AttnKind { kFull, kSlidingWindow, kMambaState };

// Why a resumable cache boundary was retained. The declaration order is its
// monotonic promotion order.
enum class CacheBoundaryKind { kChunk, kEndpoint, kPromoted };

using CacheNamespaceId = std::uint32_t;
using ContentHash = std::string;

inline constexpr CacheNamespaceId kDefaultCacheNamespaceId = 0;

struct CacheKey {
    CacheNamespaceId namespace_id{kDefaultCacheNamespaceId};
    GroupId group_id{0};
    ContentHash content_hash{};
    // CacheBlock position within the scheduler's enclosing P-token hash page.
    std::int32_t cache_block_offset{0};

    bool operator==(const CacheKey&) const noexcept = default;
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const noexcept {
        std::size_t hash = std::hash<CacheNamespaceId>{}(key.namespace_id);
        const std::size_t group_hash = std::hash<GroupId>{}(key.group_id);
        hash ^= group_hash + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        const std::size_t content_hash = std::hash<ContentHash>{}(key.content_hash);
        hash ^= content_hash + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        const std::size_t offset_hash = std::hash<std::int32_t>{}(key.cache_block_offset);
        return hash ^ (offset_hash + 0x9e3779b9U + (hash << 6U) + (hash >> 2U));
    }
};

struct KvCacheSpec {
    AttnKind kind;
    // Only kSlidingWindow uses this value. Mamba's one-checkpoint lookback is
    // an internal Manager policy rather than a model window.
    std::int32_t sliding_window;
    // Number of this group's CacheBlocks packed into one physical LCM block.
    // It affects placement only, not the scheduler-wide prefix boundary P.
    std::int32_t cache_blocks_per_lcm_block;
    // Tokens represented by one CacheBlock in this group. Zero keeps the
    // legacy meaning: use the coordinator-wide logical granularity P.
    std::int32_t cache_block_tokens{0};
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

// Per-group input for one admission. page_hashes is the request's cumulative
// completed-page history; new_page_hash_begin is the start of the hashes
// appended since the previous admission. completed_boundary_kind is present
// exactly when that suffix is non-empty. Non-closed groups select the trailing
// pages required to resume num_computed_tokens. The request owns table and the
// storage behind page_hashes.
struct GroupDemand {
    BlockTable* table{nullptr};
    std::int32_t num_tokens{0};
    std::span<const std::string> page_hashes{};
    std::int32_t new_page_hash_begin{0};
    std::optional<CacheBoundaryKind> completed_boundary_kind;
    std::int32_t num_computed_tokens{-1};
    std::int32_t reserve_tokens{0};
    // -1 materializes the ordinary dense suffix. A non-negative value keeps
    // earlier logical slots as null holes and materializes only this suffix.
    // Decode-side PD uses this for latest-snapshot state groups.
    std::int32_t materialized_suffix_start{-1};
};

// LCM ownership ids for scheduler accounting/debugging. Kernel-facing page
// tables must instead go through KvCacheManager::BlockTablePageIds().
inline std::vector<std::int32_t> BlockTableLcmBlockIds(const BlockTable& table) {
    std::vector<std::int32_t> ids;
    ids.reserve(static_cast<std::size_t>(table.NumBlocks()));
    for (const CacheBlockRef& block_ref : table.Blocks()) {
        ids.push_back(block_ref ? block_ref->Location().lcm_block_id : 0);
    }
    return ids;
}

struct PrefixMatch {
    std::vector<CacheBlockRef> blocks{};

    std::int32_t NumHitBlocks() const {
        std::int32_t count = 0;
        for (const CacheBlockRef& block_ref : blocks) {
            count += block_ref ? 1 : 0;
        }
        return count;
    }
};

// Non-owning match shape. A nonzero slot is acquired only after the coordinator
// converges every group to the final common boundary.
struct GroupPrefixProbe {
    std::vector<std::uint8_t> hits{};
};

// Pinned source/destination pages for one asynchronous cache transfer.
struct BlockTransfer {
    GroupId group_id{0};
    CacheBlockRef source;
    CacheBlockRef destination;
};

}  // namespace tokenspeed
