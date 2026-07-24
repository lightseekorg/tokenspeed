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
#include <exception>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "cache/kv_cache_manager.h"
#include "utils.h"

namespace tokenspeed {

class SwaManager : public KvCacheManager {
public:
    SwaManager(std::int32_t block_size, std::int32_t sliding_window,
               KvTableLayout table_layout = KvTableLayout::kAbsolute)
        : KvCacheManager(block_size), sliding_window_{sliding_window}, table_layout_{table_layout} {
        _assert(sliding_window > 0, "sliding_window must be > 0");
    }

    bool MatchIsPrefixClosed() const override { return false; }

    // Search right-to-left for the highest resumable endpoint. Absolute tables
    // retain zero-valued probe holes; bounded tables return only the trailing
    // cached run and record its absolute base.
    PrefixProbe Probe(const BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                      std::int32_t max_blocks) const override {
        _assert(begin_blocks >= 0, "SWA probe begin must be >= 0");
        const std::int32_t end_blocks =
            static_cast<std::int32_t>(std::min(keys.size(), static_cast<std::size_t>(std::max(max_blocks, 0))));
        PrefixProbe probe;
        probe.base_logical_page = begin_blocks;
        if (begin_blocks >= end_blocks) {
            return probe;
        }

        // W=1 needs no predecessor page. The empty bounded range still ends at
        // the selected endpoint, which is why its base advances to end_blocks.
        if (pagesNeededToResume() == 0) {
            if (table_layout_ == KvTableLayout::kBoundedWindow) {
                probe.base_logical_page = end_blocks;
            } else {
                probe.hits.resize(static_cast<std::size_t>(end_blocks - begin_blocks), CachedBlockState::kMiss);
            }
            return probe;
        }

        probe.hits.resize(static_cast<std::size_t>(end_blocks - begin_blocks), CachedBlockState::kMiss);
        const auto [boundary, hits_begin] = findResumableBoundary(
            [&](std::int32_t logical_page) {
                const CachedBlockState state = pool.ProbeCachedBlock(keys[static_cast<std::size_t>(logical_page)]);
                probe.hits[static_cast<std::size_t>(logical_page - begin_blocks)] = state;
                return IsCachedBlockHit(state);
            },
            begin_blocks, end_blocks);
        if (boundary == begin_blocks) {
            probe.hits.clear();
            return probe;
        }

        if (table_layout_ == KvTableLayout::kBoundedWindow) {
            probe.base_logical_page = hits_begin;
            probe.hits.resize(static_cast<std::size_t>(boundary - begin_blocks));
            probe.hits.erase(probe.hits.begin(), probe.hits.begin() + (hits_begin - begin_blocks));
        } else {
            probe.hits.resize(static_cast<std::size_t>(boundary - begin_blocks));
        }
        return probe;
    }

    std::optional<PrefixProbe> ProbeExactBoundary(const BlockPool& pool, std::span<const std::string> keys,
                                                  std::int32_t begin_blocks,
                                                  std::int32_t boundary_blocks) const override {
        if (begin_blocks < 0 || boundary_blocks < begin_blocks ||
            boundary_blocks > static_cast<std::int32_t>(keys.size())) {
            return std::nullopt;
        }
        const std::int32_t retained_begin = std::max(begin_blocks, boundary_blocks - pagesNeededToResume());
        PrefixProbe probe;
        probe.base_logical_page = retained_begin;
        probe.hits.reserve(static_cast<std::size_t>(boundary_blocks - retained_begin));
        for (std::int32_t logical_page = retained_begin; logical_page < boundary_blocks; ++logical_page) {
            const CachedBlockState state = pool.ProbeCachedBlock(keys[static_cast<std::size_t>(logical_page)]);
            if (!IsCachedBlockHit(state)) {
                return std::nullopt;
            }
            probe.hits.push_back(state);
        }
        return probe;
    }

    void ReclaimExpired(BlockPool& /*pool*/, BlockTable& table, std::int32_t num_computed_tokens) noexcept override {
        const std::int32_t skipped_blocks = fullySlidOutBlocks(table, num_computed_tokens);
        if (table_layout_ == KvTableLayout::kBoundedWindow) {
            if (skipped_blocks > 0) {
                table.DropBeforeNoexcept(table.BaseLogicalPage() + skipped_blocks);
            }
            return;
        }

        std::int32_t first_live = skipped_blocks;
        for (std::int32_t i = skipped_blocks - 1; i >= 0; --i) {
            if (!table.Blocks()[static_cast<std::size_t>(i)]) {
                break;
            }
            first_live = i;
        }
        // Release low->high, matching the old reverse-batch LRU order without
        // allocating a temporary owner vector on the completion path.
        for (std::int32_t i = first_live; i < skipped_blocks; ++i) {
            BlockRef old = table.EvictToNullNoexcept(i);
            if (!old) {
                std::terminate();
            }
            old.reset();
        }
    }

    std::int32_t BlocksReclaimableAt(const BlockTable& table, std::int32_t num_computed_tokens,
                                     bool count_uncached) const override {
        const std::int32_t skipped_blocks = fullySlidOutBlocks(table, num_computed_tokens);
        std::int32_t freed = 0;
        for (std::int32_t i = skipped_blocks - 1; i >= 0; --i) {
            const BlockRef& block = table.Blocks()[static_cast<std::size_t>(i)];
            if (!block) {
                break;
            }
            if (block.unique() && (count_uncached || block->IsCached())) {
                ++freed;
            }
        }
        return freed;
    }

private:
    std::int32_t pagesNeededToResume() const { return (sliding_window_ - 1 + block_size_ - 1) / block_size_; }

    struct ResumableBoundary {
        std::int32_t boundary;
        std::int32_t hits_begin;
    };

    template <typename ProbeFn>
    ResumableBoundary findResumableBoundary(const ProbeFn& probe, std::int32_t begin_blocks,
                                            std::int32_t end_blocks) const {
        const std::int32_t pages_needed = pagesNeededToResume();
        for (std::int32_t boundary = end_blocks; boundary > begin_blocks;) {
            std::int32_t hits_begin = boundary;
            while (hits_begin > begin_blocks && probe(hits_begin - 1)) {
                --hits_begin;
                if (boundary - hits_begin >= pages_needed) {
                    return {boundary, hits_begin};
                }
            }
            if (hits_begin == begin_blocks && hits_begin < boundary) {
                return {boundary, hits_begin};
            }
            boundary = hits_begin - 1;
        }
        return {begin_blocks, begin_blocks};
    }

    // Local live-page count that lies wholly before the next query's window.
    std::int32_t fullySlidOutBlocks(const BlockTable& table, std::int32_t num_computed_tokens) const {
        const std::int32_t skipped_tokens = num_computed_tokens - sliding_window_ + 1;
        if (skipped_tokens <= 0) {
            return 0;
        }
        const std::int32_t drop_before = skipped_tokens / block_size_;
        if (drop_before <= table.BaseLogicalPage()) {
            return 0;
        }
        return std::min(drop_before - table.BaseLogicalPage(), table.LiveSize());
    }

    std::int32_t sliding_window_;
    KvTableLayout table_layout_{KvTableLayout::kAbsolute};
};

}  // namespace tokenspeed
