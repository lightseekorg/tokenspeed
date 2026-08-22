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
#include <span>
#include <unordered_set>

#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/prefix_index.h"
#include "utils.h"

namespace tokenspeed {

inline bool PrefixKeyIsHit(const PrefixCacheIndex& index, const BlockPool& pool, const CacheKey& key,
                           const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits) {
    return index.Contains(pool, key) || (extra_hits != nullptr && extra_hits->contains(key));
}

// Per-attention-kind prefix-match policy. A matcher only reads the group's
// PrefixCacheIndex; it never touches allocation or physical placement.
class PrefixMatcher {
public:
    virtual ~PrefixMatcher() = default;

    // True when every match is a hole-free run from the prefix start, so a
    // shorter boundary always remains valid.
    virtual bool IsPrefixClosed() const = 0;
    // Cached pages a resumable boundary needs behind it, in group pages.
    virtual std::int32_t BoundaryLookbackPages() const = 0;
    // Probes keys[begin_blocks, begin_blocks + max_blocks) against the index.
    // probe.hits[i] marks keys[begin_blocks + i]; holes are 0.
    // extra_hits is an optional L3 storage index: keys known to exist below Host.
    virtual GroupPrefixProbe Probe(const PrefixCacheIndex& index, const BlockPool& pool, std::span<const CacheKey> keys,
                                   std::int32_t begin_blocks, std::int32_t max_blocks,
                                   const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits = nullptr) const = 0;
};

// Full attention: a hit is a contiguous run with no holes, so both the device
// and the host lookup walk left-to-right until the first miss.
class FullAttnMatcher : public PrefixMatcher {
public:
    bool IsPrefixClosed() const override { return true; }
    std::int32_t BoundaryLookbackPages() const override { return 0; }

    GroupPrefixProbe Probe(const PrefixCacheIndex& index, const BlockPool& pool, std::span<const CacheKey> keys,
                           std::int32_t begin_blocks, std::int32_t max_blocks,
                           const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits = nullptr) const override {
        const std::int32_t end_blocks =
            static_cast<std::int32_t>(std::min(keys.size(), static_cast<std::size_t>(std::max(max_blocks, 0))));
        GroupPrefixProbe probe;
        for (std::int32_t j = begin_blocks; j < end_blocks; ++j) {
            if (!PrefixKeyIsHit(index, pool, keys[static_cast<std::size_t>(j)], extra_hits)) {
                break;
            }
            probe.hits.push_back(1);
        }
        return probe;
    }
};

// Sliding window (and, with window == 2, mamba state snapshots): non-closed --
// shortening a match can cut its trailing run below the window, so match
// bound-first.
class SwaMatcher : public PrefixMatcher {
public:
    SwaMatcher(std::int32_t block_granularity, std::int32_t sliding_window)
        : block_granularity_{block_granularity}, sliding_window_{sliding_window} {
        _assert(block_granularity > 0, "block_granularity must be > 0");
        _assert(sliding_window > 0, "sliding_window must be > 0");
    }

    bool IsPrefixClosed() const override { return false; }
    std::int32_t BoundaryLookbackPages() const override { return pagesNeededToResume(); }

    // Right->left scan for a run backing a resumable boundary; slots left of it stay holes.
    GroupPrefixProbe Probe(const PrefixCacheIndex& index, const BlockPool& pool, std::span<const CacheKey> keys,
                           std::int32_t begin_blocks, std::int32_t max_blocks,
                           const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits = nullptr) const override {
        const std::int32_t end_blocks =
            static_cast<std::int32_t>(std::min(keys.size(), static_cast<std::size_t>(std::max(max_blocks, 0))));
        GroupPrefixProbe probe;
        if (begin_blocks >= end_blocks) {
            return probe;
        }
        // W == 1: no lookback, so every boundary is resumable with no cached page at all.
        if (pagesNeededToResume() == 0) {
            probe.hits.resize(static_cast<std::size_t>(end_blocks - begin_blocks));
            return probe;
        }
        const auto [boundary, hits_begin] = findResumableBoundary(
            [&](std::int32_t i) { return PrefixKeyIsHit(index, pool, keys[static_cast<std::size_t>(i)], extra_hits); },
            begin_blocks, end_blocks);
        if (boundary == begin_blocks) {
            return probe;
        }
        probe.hits.resize(static_cast<std::size_t>(boundary - begin_blocks));
        for (std::int32_t i = hits_begin; i < boundary; ++i) {
            probe.hits[static_cast<std::size_t>(i - begin_blocks)] = 1;
        }
        return probe;
    }

private:
    // Cached pages a boundary needs behind it: they cover the window's last (window - 1) tokens.
    std::int32_t pagesNeededToResume() const {
        return (sliding_window_ - 1 + block_granularity_ - 1) / block_granularity_;
    }

    struct ResumableBoundary {
        std::int32_t boundary;    // == begin_blocks when no boundary qualifies
        std::int32_t hits_begin;  // probe hits cover [hits_begin, boundary)
    };

    // Core scan shared by device and host lookup: the highest boundary backed by enough
    // consecutive probe hits -- pagesNeededToResume(), or fewer bottoming out at begin_blocks.
    template <typename Probe>
    ResumableBoundary findResumableBoundary(const Probe& probe, std::int32_t begin_blocks,
                                            std::int32_t end_blocks) const {
        const std::int32_t pages_needed = pagesNeededToResume();
        for (std::int32_t boundary = end_blocks; boundary > begin_blocks;) {
            std::int32_t hits_begin = boundary;
            while (hits_begin > begin_blocks && probe(hits_begin - 1)) {
                --hits_begin;
                if (boundary - hits_begin >= pages_needed) {
                    return {boundary, hits_begin};  // enough pages behind the boundary
                }
            }
            if (hits_begin == begin_blocks && hits_begin < boundary) {
                return {boundary, hits_begin};  // fewer, but nothing below begin_blocks is needed
            }
            // The miss at hits_begin-1 cuts every boundary in (hits_begin-1, boundary] short -- retry below it.
            boundary = hits_begin - 1;
        }
        return {begin_blocks, begin_blocks};
    }

    std::int32_t block_granularity_;
    std::int32_t sliding_window_;
};

}  // namespace tokenspeed
