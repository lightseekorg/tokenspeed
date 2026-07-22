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
    // live_alloc_alignment > 0 = live-tail allocation: real blocks only for the live tail + resume pages behind each
    // alignment boundary; 0 = full allocation.
    SwaManager(std::int32_t block_size, std::int32_t sliding_window, std::int32_t live_alloc_alignment = 0)
        : KvCacheManager(block_size), sliding_window_{sliding_window}, live_alloc_alignment_{live_alloc_alignment} {
        _assert(sliding_window > 0, "sliding_window must be > 0");
        _assert(live_alloc_alignment >= 0, "live_alloc_alignment must be >= 0");
        _assert(live_alloc_alignment == 0 || live_alloc_alignment % block_size == 0,
                "live_alloc_alignment must be a multiple of block_size");
    }

    // Non-closed: shortening a match can cut its trailing run below the window, so match bound-first.
    bool MatchIsPrefixClosed() const override { return false; }

    // Right->left scan for a run backing a resumable boundary; slots left of it stay holes.
    PrefixMatch Match(const BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                      std::int32_t max_blocks) const override {
        const std::int32_t end_blocks =
            static_cast<std::int32_t>(std::min(keys.size(), static_cast<std::size_t>(std::max(max_blocks, 0))));
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

    // All-or-nothing like the base, but dead new slots become null holes.
    bool Acquire(BlockPool& pool, BlockTable& table, std::int32_t num_tokens) override {
        if (live_alloc_alignment_ == 0) {
            return KvCacheManager::Acquire(pool, table, num_tokens);
        }
        if (num_tokens <= 0) {
            return true;
        }
        const std::int32_t tail_avail = TableTailAvail(table);
        if (num_tokens <= tail_avail) {
            SetTableTailAvail(table, tail_avail - num_tokens);
            return true;
        }
        const std::int32_t end = TableExtentTokens(table, block_size_) + num_tokens;
        const std::int32_t over = num_tokens - tail_avail;
        const std::int32_t num_new = (over + block_size_ - 1) / block_size_;
        const std::int32_t first_new = table.NumBlocks();
        std::int32_t real = 0;
        for (std::int32_t s = first_new; s < first_new + num_new; ++s) {
            real += slotIsLive(s, end) ? 1 : 0;
        }
        std::vector<CacheBlock*> new_blocks = pool.AllocateBlocks(real);
        if (real > 0 && static_cast<std::int32_t>(new_blocks.size()) < real) {
            return false;  // AllocateBlocks is all-or-nothing; table untouched
        }
        std::size_t next = 0;
        for (std::int32_t s = first_new; s < first_new + num_new; ++s) {
            if (slotIsLive(s, end)) {
                AppendRealBlock(pool, table, new_blocks[next++]);
            } else {
                AppendHole(pool, table);
            }
        }
        const std::int32_t used_in_tail = over % block_size_;
        SetTableTailAvail(table, used_in_tail == 0 ? 0 : block_size_ - used_in_tail);
        return true;
    }

    // Mirror of Acquire's page math (real blocks only).
    std::int32_t BlocksNeededFor(const BlockTable& table, std::int32_t num_tokens) const override {
        if (live_alloc_alignment_ == 0) {
            return KvCacheManager::BlocksNeededFor(table, num_tokens);
        }
        const std::int32_t tail_avail = TableTailAvail(table);
        if (num_tokens <= tail_avail) {
            return 0;
        }
        const std::int32_t end = TableExtentTokens(table, block_size_) + num_tokens;
        const std::int32_t over = num_tokens - tail_avail;
        const std::int32_t num_new = (over + block_size_ - 1) / block_size_;
        const std::int32_t first_new = table.NumBlocks();
        std::int32_t real = 0;
        for (std::int32_t s = first_new; s < first_new + num_new; ++s) {
            real += slotIsLive(s, end) ? 1 : 0;
        }
        return real;
    }

    // Sequential Acquire(first) then Acquire(extra) can peak above the combined query: the later
    // frontier holes early slots, so BlocksNeededFor(first + extra) may undercount what the first
    // acquire really pins. Charge A (first batch at its own frontier) + B (extra batch, final frontier).
    std::int32_t BlocksNeededForSequential(const BlockTable& table, std::int32_t first_tokens,
                                           std::int32_t extra_tokens) const override {
        if (live_alloc_alignment_ == 0) {
            return KvCacheManager::BlocksNeededForSequential(table, first_tokens, extra_tokens);
        }
        if (first_tokens <= 0 || extra_tokens <= 0) {
            return BlocksNeededFor(table, std::max(first_tokens, 0) + std::max(extra_tokens, 0));
        }
        const std::int32_t needed_first = BlocksNeededFor(table, first_tokens);
        // Post-first geometry (extent grows by exactly first_tokens).
        const std::int32_t tail_avail = TableTailAvail(table);
        std::int32_t num_blocks_after = table.NumBlocks();
        std::int32_t tail_after = tail_avail - first_tokens;
        if (first_tokens > tail_avail) {
            const std::int32_t over = first_tokens - tail_avail;
            const std::int32_t num_new = (over + block_size_ - 1) / block_size_;
            num_blocks_after += num_new;
            tail_after = num_new * block_size_ - over;
        }
        if (extra_tokens <= tail_after) {
            return needed_first;
        }
        const std::int32_t over_extra = extra_tokens - tail_after;
        const std::int32_t num_new_extra = (over_extra + block_size_ - 1) / block_size_;
        const std::int32_t end = TableExtentTokens(table, block_size_) + first_tokens + extra_tokens;
        std::int32_t needed_extra = 0;
        for (std::int32_t s = num_blocks_after; s < num_blocks_after + num_new_extra; ++s) {
            needed_extra += slotIsLive(s, end) ? 1 : 0;
        }
        return needed_first + needed_extra;
    }

    // Punches null holes so the table never shrinks (keeps slot alignment); reverse-collect evicts FIFO.
    // The reclaim floor bounds the scan to the newly-expired band; punched history is never re-walked.
    void ReclaimExpired(BlockPool& pool, BlockTable& table, std::int32_t num_computed_tokens) override {
        const std::int32_t skipped_blocks = fullySlidOutBlocks(table, num_computed_tokens);
        const std::int32_t floor = TableReclaimFloor(table);
        std::vector<CacheBlock*> freed;
        for (std::int32_t i = skipped_blocks - 1; i >= floor; --i) {
            CacheBlock* old = table.EvictToNull(i, pool.NullBlock());
            if (old == nullptr) {
                // Live-tail interleaves holes with real checkpoint pages, so don't early-break.
                if (live_alloc_alignment_ == 0) {
                    break;  // already null -> earlier slots are null too
                }
                continue;
            }
            freed.push_back(old);
        }
        if (skipped_blocks > floor) {
            SetTableReclaimFloor(table, skipped_blocks);
        }
        pool.FreeBlocks(freed);
    }

    // Only blocks whose last reference is this table (RefCount()==1) reach the free list, so shared ones don't count.
    std::int32_t BlocksReclaimableAt(const BlockTable& table, std::int32_t num_computed_tokens,
                                     bool count_uncached) const override {
        const std::int32_t skipped_blocks = fullySlidOutBlocks(table, num_computed_tokens);
        const std::int32_t floor = TableReclaimFloor(table);
        std::int32_t freed = 0;
        for (std::int32_t i = skipped_blocks - 1; i >= floor; --i) {
            CacheBlock* block = table.Blocks()[i];
            if (block->IsNull()) {
                if (live_alloc_alignment_ == 0) {
                    break;  // already null -> earlier slots are null too
                }
                continue;  // interleaved hole (live-tail allocation)
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

    // Live-tail predicate: NEW slot s is real iff not fully slid out at frontier_tokens, or a resume page behind an
    // alignment boundary.
    bool slotIsLive(std::int32_t s, std::int32_t frontier_tokens) const {
        const std::int32_t skipped = frontier_tokens - sliding_window_ + 1;
        if (skipped <= 0 || s >= skipped / block_size_) {
            return true;
        }
        const std::int32_t blocks_per_aligned = live_alloc_alignment_ / block_size_;
        return s % blocks_per_aligned >= blocks_per_aligned - pagesNeededToResume();
    }

    std::int32_t sliding_window_;
    std::int32_t live_alloc_alignment_;
};

}  // namespace tokenspeed
