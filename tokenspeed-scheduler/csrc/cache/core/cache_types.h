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

#include "cache/core/block_table.h"
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
    std::uint32_t group_id{0};
    ContentHash content_hash{};
    // Ordinal of this group's page within the enclosing prefix page (the
    // prefix_granularity-token span covered by one content hash), NOT a
    // token/byte offset inside a page.
    std::int32_t page_offset{0};

    bool operator==(const CacheKey&) const noexcept = default;
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const noexcept {
        std::size_t hash = std::hash<CacheNamespaceId>{}(key.namespace_id);
        const std::size_t group_hash = std::hash<std::uint32_t>{}(key.group_id);
        hash ^= group_hash + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        const std::size_t content_hash = std::hash<ContentHash>{}(key.content_hash);
        hash ^= content_hash + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        const std::size_t offset_hash = std::hash<std::int32_t>{}(key.page_offset);
        return hash ^ (offset_hash + 0x9e3779b9U + (hash << 6U) + (hash >> 2U));
    }
};

struct CacheGroupSpec {
    AttnKind kind{AttnKind::kFull};
    // Only kSlidingWindow uses this value. Mamba's one-checkpoint lookback is
    // an internal retention policy rather than a model window.
    std::int32_t sliding_window{0};
    // Number of this group's CacheBlocks packed into one physical LCM block.
    // It affects placement only, not the scheduler-wide prefix granularity.
    std::int32_t cache_blocks_per_lcm_block{1};
    // Tokens represented by one CacheBlock in this group. Required: must be
    // a positive divisor of the coordinator-wide prefix granularity.
    std::int32_t block_granularity{0};
};

// Per-group input for one admission. prefix_hashes is the request's cumulative
// completed prefix-page history; new_prefix_hash_begin is the start of the
// hashes appended since the previous admission. completed_boundary_kind is
// present exactly when that suffix is non-empty. Non-closed groups select the
// trailing pages required to resume num_computed_tokens. The request owns
// table and the storage behind prefix_hashes.
struct GroupDemand {
    BlockTable* table{nullptr};
    std::int32_t num_tokens{0};
    std::span<const std::string> prefix_hashes{};
    std::int32_t new_prefix_hash_begin{0};
    std::optional<CacheBoundaryKind> completed_boundary_kind;
    std::int32_t num_computed_tokens{-1};
    std::int32_t reserve_tokens{0};
    // -1 materializes the ordinary dense suffix. A non-negative value keeps
    // earlier logical slots as null holes and materializes only this suffix.
    // Snapshot-state local prefill uses an absolute endpoint here; Decode-side
    // PD also uses it for latest snapshots and retained sliding tails.
    std::int32_t materialized_suffix_start{-1};
};

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

// Pinned source/destination blocks for one asynchronous cache transfer.
struct BlockTransfer {
    std::uint32_t group_id{0};
    CacheBlockRef source;
    CacheBlockRef destination;
    CacheKey key{};
    // True when source is a freshly allocated Host block that L3 must fill
    // before the Host→Device copy.
    bool prefetch_from_storage{false};
};

}  // namespace tokenspeed
