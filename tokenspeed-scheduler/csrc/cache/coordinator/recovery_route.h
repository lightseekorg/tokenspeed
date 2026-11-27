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
#include <optional>
#include <span>
#include <vector>

#include "cache/coordinator/cache_store.h"

namespace tokenspeed {
struct RecoveryDependency {
    CacheKey key;
    CacheTier tier;
};
struct RecoveryRoute {
    std::int32_t split_tokens;
    bool device_tail;
    bool device_only;
    std::vector<RecoveryDependency> dependencies;
};
struct RecoveryRange {
    std::size_t group;
    std::int32_t first;
    std::int32_t past_end;
    CacheTier tier;
};
struct RecoveryRanges {
    std::int32_t split_tokens;
    bool device_tail;
    bool device_only;
    std::vector<RecoveryRange> dependencies;
};

// The matcher and persistent index use the same route rules. FirstMissing
// returns past_end when the entire block interval is present. State is [0,1).
// A range summary can answer without materializing any individual keys.
template <typename FirstMissing>
std::optional<RecoveryRanges> ResolveRecoveryRanges(const CacheStore& store, std::int32_t tokens, bool allow_host,
                                                    bool include_state, const FirstMissing& first_missing) {
    const auto page = store.PrefixGranularity();
    const auto tail = tokens % page;
    const auto full = tokens - tail;
    auto split = full;
    for (auto i : store.RoutingHistoryGroups()) {
        const auto grain = store.Geometry()[i].BlockGranularity();
        const auto missing = first_missing(i, 0, split / grain, CacheTier::kDevice);
        const auto at = std::int64_t{missing} * grain;
        split = static_cast<std::int32_t>(at - at % page);
    }
    for (auto i : store.MatchOrder()) {
        if (store.GroupIsState(i) || store.Groups()[i].Matcher().IsPrefixClosed()) continue;
        const auto window = store.SlidingWindowBlocks(static_cast<std::int32_t>(i), tokens);
        const auto grain = store.Geometry()[i].BlockGranularity();
        for (auto range : {store.SinkBlocksBelow(static_cast<std::int32_t>(i), window), window}) {
            const auto end = std::min(range.past_end, split / grain);
            if (range.first >= end) continue;
            const auto missing = first_missing(i, range.first, end, CacheTier::kDevice);
            if (missing < end) {
                const auto at = std::int64_t{missing} * grain;
                split = static_cast<std::int32_t>(at - at % page);
            }
        }
    }
    bool device_tail = split == full;
    if (tail > 0 && device_tail) {
        for (auto i : store.MatchOrder()) {
            if (store.GroupIsState(i)) continue;
            const auto base = full / store.Geometry()[i].BlockGranularity();
            const auto first = base + store.FirstTailBlockNeeded(static_cast<std::int32_t>(i), tail);
            const auto end = base + store.TailBlocksFor(static_cast<std::int32_t>(i), tail);
            device_tail = device_tail && first_missing(i, first, end, CacheTier::kDevice) == end;
        }
    }
    RecoveryRanges result{split, device_tail, true, {}};
    const auto add = [&](std::size_t group, std::int32_t first, std::int32_t end, CacheTier tier) {
        if (first >= end) return true;
        if ((tier == CacheTier::kHost && !allow_host) || first_missing(group, first, end, tier) != end) return false;
        result.dependencies.push_back({group, first, end, tier});
        result.device_only = result.device_only && tier == CacheTier::kDevice;
        return true;
    };
    for (auto i : store.MatchOrder()) {
        if (store.GroupIsState(i)) {
            if (include_state &&
                !add(i, 0, 1, first_missing(i, 0, 1, CacheTier::kDevice) == 1 ? CacheTier::kDevice : CacheTier::kHost))
                return std::nullopt;
            continue;
        }
        const auto grain = store.Geometry()[i].BlockGranularity();
        const auto window = store.Groups()[i].Matcher().IsPrefixClosed()
                                ? CacheStore::BlockRange{0, tokens / grain + (tokens % grain != 0)}
                                : store.SlidingWindowBlocks(static_cast<std::int32_t>(i), tokens);
        for (auto range : {store.SinkBlocksBelow(static_cast<std::int32_t>(i), window), window}) {
            if (!add(i, range.first, std::min(range.past_end, split / grain), CacheTier::kDevice) ||
                !add(i, std::max(range.first, split / grain), std::min(range.past_end, full / grain),
                     CacheTier::kHost) ||
                !add(i, std::max(range.first, full / grain), range.past_end,
                     device_tail ? CacheTier::kDevice : CacheTier::kHost))
                return std::nullopt;
        }
    }
    return result;
}

template <typename Contains>
std::optional<RecoveryRoute> ResolveRecoveryRoute(const CacheStore& store, std::span<const ContentHash> prefix_hashes,
                                                  const ContentHash& hash, std::int32_t tokens, bool allow_host,
                                                  bool include_state, const Contains& contains) {
    const auto full_pages = tokens / store.PrefixGranularity();
    const auto key_at = [&](std::size_t group, std::int32_t block) {
        if (store.GroupIsState(group))
            return CacheKey{.group_id = static_cast<std::uint32_t>(group), .content_hash = hash};
        const auto per_page = store.BlocksPerBoundary(static_cast<std::int32_t>(group));
        const auto page_index = block / per_page;
        return CacheKey{
            .group_id = static_cast<std::uint32_t>(group),
            .content_hash = page_index < full_pages ? prefix_hashes[static_cast<std::size_t>(page_index)] : hash,
            .page_offset = block % per_page};
    };
    const auto ranges =
        ResolveRecoveryRanges(store, tokens, allow_host, include_state,
                              [&](std::size_t group, std::int32_t first, std::int32_t end, CacheTier tier) {
                                  while (first < end && contains(key_at(group, first), tier)) ++first;
                                  return first;
                              });
    if (!ranges) return std::nullopt;
    RecoveryRoute result{ranges->split_tokens, ranges->device_tail, ranges->device_only, {}};
    for (const auto& range : ranges->dependencies)
        for (auto b = range.first; b < range.past_end; ++b)
            result.dependencies.push_back({key_at(range.group, b), range.tier});
    return result;
}
}  // namespace tokenspeed
