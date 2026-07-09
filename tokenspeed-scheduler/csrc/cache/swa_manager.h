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
    SwaManager(std::int32_t block_size, std::int32_t sliding_window)
        : KvCacheManager(block_size), sliding_window_{sliding_window} {
        _assert(sliding_window > 0, "sliding_window must be > 0");
    }

    // Non-closed: shortening a match can cut its trailing run below the window, so match bound-first.
    bool MatchIsPrefixClosed() const override { return false; }

    // Right->left scan for a run backing a resumable boundary; slots left of it stay holes.
    PrefixMatch Match(const BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                      std::int32_t max_blocks) const override {
        const std::int32_t end_blocks = static_cast<std::int32_t>(
            std::min(keys.size(), static_cast<std::size_t>(std::max(max_blocks, 0))));
        PrefixMatch match;
        if (begin_blocks >= end_blocks) {
            return match;
        }
        // W == 1: no lookback, so every boundary is resumable with no cached page at all.
        if (pagesNeededToResume() == 0) {
            match.blocks.assign(static_cast<std::size_t>(end_blocks - begin_blocks), pool.NullBlock());
            return match;
        }
        std::vector<CacheBlock*> probed(static_cast<std::size_t>(end_blocks), pool.NullBlock());
        const auto [boundary, hits_begin] = findResumableBoundary(
            [&](std::int32_t i) {
                CacheBlock* block = pool.GetCachedBlock(keys[static_cast<std::size_t>(i)]);
                if (block != nullptr) {
                    probed[static_cast<std::size_t>(i)] = block;
                }
                return block != nullptr;
            },
            begin_blocks, end_blocks);
        if (boundary == begin_blocks) {
            return match;
        }
        match.blocks.assign(probed.begin() + begin_blocks, probed.begin() + boundary);
        match.num_hit_blocks = boundary - hits_begin;
        return match;
    }

    // Punches null holes so the table never shrinks (keeps slot alignment); reverse-collect evicts FIFO.
    void ReclaimExpired(BlockPool& pool, BlockTable& table, std::int32_t num_computed_tokens) override {
        std::int32_t skipped_blocks = fullySlidOutBlocks(table, num_computed_tokens);
        std::vector<CacheBlock*> freed;
        for (std::int32_t i = skipped_blocks - 1; i >= 0; --i) {
            CacheBlock* old = table.EvictToNull(i, pool.NullBlock());
            if (old == nullptr) {
                break;  // already null -> earlier slots are null too
            }
            freed.push_back(old);
        }
        pool.FreeBlocks(freed);
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
    // Cached pages a boundary needs behind it: they cover the window's last (window - 1) tokens.
    std::int32_t pagesNeededToResume() const { return (sliding_window_ - 1 + block_size_ - 1) / block_size_; }

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

    // Pages [0, result) fully slid out: the next query reads keys [num_computed - window + 1, num_computed].
    std::int32_t fullySlidOutBlocks(const BlockTable& table, std::int32_t num_computed_tokens) const {
        std::int32_t skipped = num_computed_tokens - sliding_window_ + 1;
        if (skipped <= 0) {
            return 0;  // all tokens still inside the window
        }
        std::int32_t skipped_blocks = skipped / block_size_;  // only fully-slid-out pages
        // Safety cap: FSM-consistent input never engages it.
        return std::min(skipped_blocks, table.NumBlocks());
    }

    std::int32_t sliding_window_;
};

}  // namespace tokenspeed
