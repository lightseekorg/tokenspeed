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
#include <algorithm>
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
    SwaManager(BlockPool& pool, std::int32_t page_size, std::int32_t sliding_window)
        : KvCacheManager(pool, page_size), sliding_window_{sliding_window} {
        _assert(sliding_window > 0, "sliding_window must be > 0");
    }

    using KvCacheManager::MatchPrefix;  // keep the bounded overload visible

    // Read-only sliding-window prefix match. Scans the page-hash keys right->left,
    // looking for a run of `contiguous_needed = ceil((window-1)/page_size)`
    // cached pages that covers the window. Stops at the first run that reaches
    // that length; if none does, keeps the run still accumulating at the left
    // end. Holes left of the kept run stay as null_block padding. Does NOT change
    // ref counts -- callers claim hits via ClaimHitBlocks.
    PrefixMatch MatchPrefix(std::span<const std::string> block_hashes) const override {
        std::int32_t n = static_cast<std::int32_t>(block_hashes.size());
        std::int32_t contiguous_needed = (sliding_window_ - 1 + page_size_ - 1) / page_size_;

        PrefixMatch match;
        std::vector<CacheBlock*> hits(n, pool_.NullBlock());
        std::int32_t run = 0;
        std::int32_t run_end = -1;
        for (std::int32_t i = n - 1; i >= 0; --i) {
            CacheBlock* block = pool_.GetCachedBlock(block_hashes[i]);
            if (block != nullptr) {
                hits[i] = block;
                if (run == 0) {
                    run_end = i;
                }
                ++run;
                if (run >= contiguous_needed) {
                    break;
                }
            } else {
                run = 0;
                run_end = -1;
            }
        }

        if (run == 0) {
            return match;  // nothing cached -> empty match
        }
        std::int32_t keep = run_end + 1;
        match.blocks.assign(hits.begin(), hits.begin() + keep);
        match.num_hit_blocks = run;
        return match;
    }

    // Advance the window to num_computed_tokens: free every page that has fully
    // slid out of the sliding window, replacing its slot with a null hole. The
    // tail page (still in-window) and tail_avail_ are untouched; the table never
    // shrinks (holes keep logical-page -> slot alignment). Scans the skipped
    // range right->left and stops at the first already-null slot (earlier slots
    // were punched by prior calls). Reverse-collect + direct FreeBlocks evicts
    // the first-slid-out page first (FIFO).
    void AdvanceWindow(BlockTable& table, std::int32_t num_computed_tokens) {
        std::int32_t skipped = num_computed_tokens - sliding_window_ + 1;
        if (skipped <= 0) {
            return;  // all tokens still inside the window
        }
        std::int32_t skipped_blocks = skipped / page_size_;  // only fully-slid-out pages
        // Safety net for inconsistent input: with FSM-consistent num_computed_tokens
        // the cap never engages the tail page (a full tail leaves >=1 in-window
        // page), but an arbitrary oversized value could -- cap keeps us in bounds.
        skipped_blocks = std::min(skipped_blocks, table.NumBlocks());
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

private:
    std::int32_t sliding_window_;
};

}  // namespace tokenspeed
