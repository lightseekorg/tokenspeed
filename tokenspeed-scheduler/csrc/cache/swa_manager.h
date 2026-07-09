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
#include <string>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "cache/kv_cache_manager.h"
#include "utils.h"

namespace tokenspeed {

class SwaManager : public KvCacheManager {
public:
    SwaManager(BlockPool& pool, std::int32_t page_size, std::int32_t sliding_window)
        : KvCacheManager(pool, page_size), sliding_window_{sliding_window} {
        _assert(sliding_window > 0, "sliding_window must be > 0");
    }

    // Non-closed: shortening a match can cut its trailing run below the window, so match bound-first.
    bool MatchIsPrefixClosed() const override { return false; }

    // Right->left scan for a contiguous cached run covering the window; holes left of it stay null_block padding.
    PrefixMatch MatchPrefix(std::span<const std::string> block_hashes, std::int32_t max_blocks) const override {
        const std::int32_t n = static_cast<std::int32_t>(
            std::min(block_hashes.size(), static_cast<std::size_t>(std::max(max_blocks, 0))));
        std::vector<CacheBlock*> hits(static_cast<std::size_t>(n), pool_.NullBlock());
        const auto [boundary, run_start] = findTrailingRun(
            [&](std::int32_t i) {
                CacheBlock* block = pool_.GetCachedBlock(block_hashes[static_cast<std::size_t>(i)]);
                if (block != nullptr) {
                    hits[static_cast<std::size_t>(i)] = block;
                }
                return block != nullptr;
            },
            0, n);
        PrefixMatch match;
        if (boundary == 0) {
            return match;
        }
        match.blocks.assign(hits.begin(), hits.begin() + boundary);
        match.num_hit_blocks = boundary - run_start;
        return match;
    }

    // Host-pool lookup: the same trailing-run scan; nullptr pads [begin, run_start).
    std::vector<CacheBlock*> MatchHostPages(BlockPool& host_pool, std::span<const std::string> keys,
                                            std::int32_t begin_blocks, std::int32_t max_blocks) const override {
        // W == 1: contiguousNeeded() is 0, so any boundary is resumable with no host page at all.
        if (contiguousNeeded() == 0) {
            return std::vector<CacheBlock*>(static_cast<std::size_t>(max_blocks - begin_blocks), nullptr);
        }
        std::vector<CacheBlock*> found(static_cast<std::size_t>(max_blocks), nullptr);
        const auto [boundary, run_start] = findTrailingRun(
            [&](std::int32_t i) {
                CacheBlock* block = host_pool.GetCachedBlock(keys[static_cast<std::size_t>(i)]);
                found[static_cast<std::size_t>(i)] = block;
                return block != nullptr;
            },
            begin_blocks, max_blocks);
        std::vector<CacheBlock*> pages(static_cast<std::size_t>(boundary - begin_blocks), nullptr);
        for (std::int32_t j = run_start; j < boundary; ++j) {
            pages[static_cast<std::size_t>(j - begin_blocks)] = found[static_cast<std::size_t>(j)];
        }
        return pages;
    }

    // Punches null holes so the table never shrinks (keeps slot alignment); reverse-collect evicts FIFO.
    void ReclaimExpired(BlockTable& table, std::int32_t num_computed_tokens) override {
        std::int32_t skipped_blocks = fullySlidOutBlocks(table, num_computed_tokens);
        std::vector<CacheBlock*> freed;
        for (std::int32_t i = skipped_blocks - 1; i >= 0; --i) {
            CacheBlock* old = table.EvictToNull(i, pool_.NullBlock());
            if (old == nullptr) {
                break;  // already null -> earlier slots are null too
            }
            freed.push_back(old);
        }
        pool_.FreeBlocks(freed);
    }

    // Only blocks whose last reference is this table (RefCount()==1) reach the free list, so shared ones don't count.
    std::int32_t BlocksReclaimableAt(const BlockTable& table, std::int32_t num_computed_tokens,
                                     bool count_uncached) const override {
        std::int32_t skipped_blocks = fullySlidOutBlocks(table, num_computed_tokens);
        std::int32_t freed = 0;
        for (std::int32_t i = skipped_blocks - 1; i >= 0; --i) {
            CacheBlock* block = table.Blocks()[i];
            if (block->IsNull()) {
                break;  // already null -> earlier slots are null too
            }
            if (block->RefCount() == 1 && (count_uncached || block->IsCached())) {
                ++freed;
            }
        }
        return freed;
    }

private:
    // Pages a valid match's trailing run must cover: ceil((window - 1) / page_size).
    std::int32_t contiguousNeeded() const { return (sliding_window_ - 1 + page_size_ - 1) / page_size_; }

    // Core scan shared by device and host lookup: right->left over [begin, end) for the
    // highest run of contiguousNeeded() hits -- or a shorter run bottoming at `begin`.
    // Returns {boundary, run_start}: hits cover [run_start, boundary); boundary == begin
    // means no acceptable run. Early-exits, so slots below run_start stay unprobed.
    template <typename Hit>
    std::pair<std::int32_t, std::int32_t> findTrailingRun(const Hit& hit, std::int32_t begin_blocks,
                                                          std::int32_t end_blocks) const {
        const std::int32_t needed = contiguousNeeded();
        std::int32_t run = 0;
        std::int32_t run_end = -1;
        for (std::int32_t i = end_blocks - 1; i >= begin_blocks; --i) {
            if (!hit(i)) {
                run = 0;
                run_end = -1;
                continue;
            }
            if (run == 0) {
                run_end = i;
            }
            ++run;
            if (run >= needed) {
                break;
            }
        }
        if (run == 0) {
            return {begin_blocks, begin_blocks};
        }
        return {run_end + 1, run_end + 1 - run};
    }

    // Pages [0, result) fully slid out: the next query reads keys [num_computed - window + 1, num_computed].
    std::int32_t fullySlidOutBlocks(const BlockTable& table, std::int32_t num_computed_tokens) const {
        std::int32_t skipped = num_computed_tokens - sliding_window_ + 1;
        if (skipped <= 0) {
            return 0;  // all tokens still inside the window
        }
        std::int32_t skipped_blocks = skipped / page_size_;  // only fully-slid-out pages
        // Safety cap: FSM-consistent input never engages it.
        return std::min(skipped_blocks, table.NumBlocks());
    }

    std::int32_t sliding_window_;
};

}  // namespace tokenspeed
