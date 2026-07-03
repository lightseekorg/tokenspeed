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
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// Abstract base for the per-attention-type KV managers (FullAttnManager, and
// later SwaManager / MambaManager). MatchPrefix is the only type-specific
// operation -- the four others are shared, stateless over (pool_, page_size_)
// plus the BlockTable handed in. Holds no per-request state.
class KvCacheManager {
public:
    KvCacheManager(BlockPool& pool, std::int32_t page_size) : pool_{pool}, page_size_{page_size} {
        _assert(page_size > 0, "page_size must be > 0");
    }
    virtual ~KvCacheManager() = default;

    KvCacheManager(const KvCacheManager&) = delete;
    KvCacheManager& operator=(const KvCacheManager&) = delete;

    // Per-attention-type prefix match. Read-only: must NOT change ref counts.
    virtual PrefixMatch MatchPrefix(std::span<const std::string> block_hashes) const = 0;

    // Bounded prefix match: the manager's best valid match whose coverage is at
    // most max_blocks. Implemented by matching only the first max_blocks page
    // hashes, so each manager's own validity invariant (e.g. SwaManager's
    // trailing contiguous run) is enforced against the BOUNDED end -- never
    // computed unbounded and then chopped, which could break it. May return a
    // match shorter than max_blocks (or empty) if no valid match fits the bound.
    PrefixMatch MatchPrefix(std::span<const std::string> block_hashes, std::int32_t max_blocks) const {
        std::size_t bound = std::min(block_hashes.size(), static_cast<std::size_t>(std::max(max_blocks, 0)));
        return MatchPrefix(block_hashes.first(bound));
    }

    // Claim each real hit block (TouchBlock pulls it from the free list and bumps
    // ref) and append it to the table. null_block holes are appended as-is (they
    // keep logical-page alignment) but never touched -- the null block is not ref
    // counted. Must be called on a fresh table (prefill start) before any Acquire.
    bool ClaimHitBlocks(BlockTable& table, const PrefixMatch& hit) {
        _assert(table.blocks_.empty(), "ClaimHitBlocks requires a fresh (empty) table");
        for (CacheBlock* block : hit.blocks) {
            if (!block->IsNull()) {
                pool_.TouchBlock(block);
            }
            table.blocks_.push_back(block);
        }
        return true;
    }

    // Token-driven incremental allocation, mirroring LocalKVAllocator::Acquire.
    // Fills the tail page's remaining room first; if more is needed, allocates
    // ceil(overflow / page_size) fresh pages and appends them. All-or-nothing:
    // on a capacity shortfall the table is left unchanged and false is returned.
    bool Acquire(BlockTable& table, std::int32_t num_tokens) {
        if (num_tokens <= 0) {
            return true;
        }
        if (num_tokens <= table.tail_avail_) {
            table.tail_avail_ -= num_tokens;
            return true;
        }
        std::int32_t over = num_tokens - table.tail_avail_;
        std::int32_t num_pages = (over + page_size_ - 1) / page_size_;
        std::vector<CacheBlock*> new_blocks = pool_.AllocateBlocks(num_pages);
        if (static_cast<std::int32_t>(new_blocks.size()) < num_pages) {
            return false;  // AllocateBlocks is all-or-nothing; nothing claimed.
        }
        for (CacheBlock* block : new_blocks) {
            table.blocks_.push_back(block);
        }
        std::int32_t used_in_tail = over % page_size_;
        table.tail_avail_ = (used_in_tail == 0) ? 0 : page_size_ - used_in_tail;
        return true;
    }

    // Pure query: how many fresh pages this table would need to absorb
    // num_tokens, WITHOUT allocating or mutating anything. Mirrors Acquire's page
    // math exactly (tail room first, then ceil(overflow / page_size)). Used by
    // the coordinator to check capacity across all groups before committing.
    std::int32_t BlocksNeededFor(const BlockTable& table, std::int32_t num_tokens) const {
        if (num_tokens <= table.tail_avail_) {
            return 0;
        }
        std::int32_t over = num_tokens - table.tail_avail_;
        return (over + page_size_ - 1) / page_size_;
    }

    // Register the table's full pages [first-uncached, num_full_blocks) into the
    // pool under their page-hash keys so later requests can prefix-hit them.
    // Leading pages already carrying a hash (from a prefix hit or an earlier
    // call) are skipped. block_hashes is indexed in lockstep with table pages;
    // the caller passes num_full_blocks excluding any partial tail page.
    void CacheFullBlocks(BlockTable& table, std::span<const std::string> block_hashes,
                         std::int32_t num_full_blocks) {
        _assert(num_full_blocks <= table.NumBlocks(), "num_full_blocks exceeds table size");
        _assert(num_full_blocks <= static_cast<std::int32_t>(block_hashes.size()),
                "block_hashes shorter than num_full_blocks");
        for (std::int32_t i = 0; i < num_full_blocks; ++i) {
            CacheBlock* block = table.blocks_[i];
            if (block->IsCached()) {
                continue;  // already registered (prefix hit or earlier call)
            }
            pool_.CacheFullBlocks(block, block_hashes[i]);
        }
    }

    // Release every page the table holds (reverse-order via the pool) and reset
    // the table. Cached pages keep their hash on free, so they remain
    // prefix-reusable until evicted.
    void Free(BlockTable& table) {
        pool_.FreeBlocks(table.blocks_);
        table.blocks_.clear();
        table.tail_avail_ = 0;
    }

protected:
    BlockPool& pool_;
    std::int32_t page_size_;
};

}  // namespace tokenspeed
