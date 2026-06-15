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
#include <span>
#include <string>
#include <vector>

#include "block_pool/block_pool.h"
#include "cache/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// Stateless full-attention KV manager over a BlockPool. Holds no per-request
// state -- every operation acts on the BlockTable handed in. Consumes
// pre-computed BlockHashWithGroupId keys (does not depend on page_hasher).
class FullAttnManager {
public:
    FullAttnManager(BlockPool& pool, std::int32_t page_size) : pool_{pool}, page_size_{page_size} {
        _assert(page_size > 0, "page_size must be > 0");
    }

    FullAttnManager(const FullAttnManager&) = delete;
    FullAttnManager& operator=(const FullAttnManager&) = delete;

    // Read-only prefix match: walk the pre-computed page-hash keys until the
    // first cache miss. Returns the hit blocks in order. Does NOT change ref
    // counts -- callers claim hits via ClaimHitBlocks.
    PrefixMatch MatchPrefix(std::span<const std::string> block_hashes) const {
        PrefixMatch match;
        for (const std::string& hash : block_hashes) {
            CacheBlock* block = pool_.GetCachedBlock(hash);
            if (block == nullptr) {
                break;
            }
            match.blocks.push_back(block);
        }
        match.num_hit_blocks = static_cast<std::int32_t>(match.blocks.size());
        return match;
    }

    // Claim each hit block (TouchBlock pulls it from the free list and bumps
    // ref) and append it to the table. Must be called on a fresh table (prefill
    // start) before any Acquire: the table's tail_avail_ is therefore already 0
    // and the claimed full pages are never ordered after a partial tail page.
    // Always succeeds (claiming a cached-free block never allocates); the bool
    // return mirrors Acquire for a uniform call shape. Returns true.
    bool ClaimHitBlocks(BlockTable& table, const PrefixMatch& hit) {
        _assert(table.blocks_.empty(), "ClaimHitBlocks requires a fresh (empty) table");
        for (CacheBlock* block : hit.blocks) {
            pool_.TouchBlock(block);
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

private:
    BlockPool& pool_;
    std::int32_t page_size_;
};

}  // namespace tokenspeed
