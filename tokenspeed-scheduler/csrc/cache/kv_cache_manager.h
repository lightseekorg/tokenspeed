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
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/block_ref.h"
#include "cache/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// Pure per-attention-type policy over page_size (+ window): holds no pool, no per-request
// state -- every operation acts on the pool it is handed, identically for any tier.
class KvCacheManager {
public:
    explicit KvCacheManager(std::int32_t page_size) : page_size_{page_size} {
        _assert(page_size > 0, "page_size must be > 0");
    }
    virtual ~KvCacheManager() = default;

    KvCacheManager(const KvCacheManager&) = delete;
    KvCacheManager& operator=(const KvCacheManager&) = delete;

    // Downward-closed matches: any prefix of a valid match is itself valid, so the
    // coordinator may match once and trim late; non-closed managers re-match bound-first.
    virtual bool MatchIsPrefixClosed() const = 0;

    // One matcher for every tier: scan `pool` over slots [begin_blocks, max_blocks) of the FULL
    // key sequence (index = block index; slots below the floor are assumed valid in a lower
    // tier). Read-only; blocks are relative to begin_blocks, holes = pool.NullBlock().
    virtual PrefixMatch Match(const BlockPool& pool, std::span<const std::string> keys,
                              std::int32_t begin_blocks, std::int32_t max_blocks) const = 0;

    // null_block holes are appended as-is (never ref counted) to keep logical-page alignment.
    void ClaimHitBlocks(BlockPool& pool, BlockTable& table, const PrefixMatch& hit) {
        _assert(table.blocks_.empty(), "ClaimHitBlocks requires a fresh (empty) table");
        for (CacheBlock* block : hit.blocks) {
            _assert(block->IsNull() || block->IsCached(), "matched block lost its hash before the claim");
            table.blocks_.push_back(BlockRef::Share(pool, block));
        }
    }

    // All-or-nothing (tail-page room first, then fresh pages): on shortfall the table is unchanged, returns false.
    bool Acquire(BlockPool& pool, BlockTable& table, std::int32_t num_tokens) {
        if (num_tokens <= 0) {
            return true;
        }
        if (num_tokens <= table.tail_avail_) {
            table.tail_avail_ -= num_tokens;
            return true;
        }
        std::int32_t over = num_tokens - table.tail_avail_;
        std::int32_t num_pages = (over + page_size_ - 1) / page_size_;
        std::vector<CacheBlock*> new_blocks = pool.AllocateBlocks(num_pages);
        if (static_cast<std::int32_t>(new_blocks.size()) < num_pages) {
            return false;
        }
        for (CacheBlock* block : new_blocks) {
            table.blocks_.push_back(BlockRef::Adopt(pool, block));
        }
        std::int32_t used_in_tail = over % page_size_;
        table.tail_avail_ = (used_in_tail == 0) ? 0 : page_size_ - used_in_tail;
        return true;
    }

    // Contract on the forward_cache_ops facade; admission pre-charged the real slots via ext_real_pages.
    void AppendHostExtension(BlockPool& pool, BlockTable& table, std::span<CacheBlock* const> host_blocks,
                             std::vector<std::pair<CacheBlock*, CacheBlock*>>& load_pairs) {
        _assert(table.tail_avail_ == 0, "host extension must append on a full-page boundary");
        for (CacheBlock* host_block : host_blocks) {
            if (host_block->IsNull()) {
                table.blocks_.push_back(BlockRef::Share(pool, pool.NullBlock()));
                continue;
            }
            const bool acquired = Acquire(pool, table, page_size_);
            _assert(acquired, "pre-checked Acquire must succeed");
            load_pairs.emplace_back(host_block, table.blocks_.back().Get());
        }
    }

    // Pure query mirroring Acquire's page math exactly.
    std::int32_t BlocksNeededFor(const BlockTable& table, std::int32_t num_tokens) const {
        if (num_tokens <= table.tail_avail_) {
            return 0;
        }
        std::int32_t over = num_tokens - table.tail_avail_;
        return (over + page_size_ - 1) / page_size_;
    }

    // Pages already carrying a hash are skipped; the partial tail is excluded by the caller.
    void CacheFullBlocks(BlockPool& pool, BlockTable& table, std::span<const std::string> block_hashes,
                         std::int32_t first_slot = 0,
                         std::vector<std::pair<std::string, CacheBlock*>>* newly_cached = nullptr) {
        _assert(first_slot >= 0, "first_slot must be >= 0");
        _assert(static_cast<std::int64_t>(first_slot) + static_cast<std::int64_t>(block_hashes.size()) <=
                    table.NumBlocks(),
                "hash range exceeds table size");
        for (std::size_t j = 0; j < block_hashes.size(); ++j) {
            CacheBlock* block = table.blocks_[static_cast<std::size_t>(first_slot) + j].Get();
            if (block->IsNull()) {
                continue;
            }
            if (block->IsCached()) {
                continue;
            }
            pool.CacheFullBlock(block, block_hashes[j]);
            if (newly_cached != nullptr) {
                newly_cached->emplace_back(block_hashes[j], block);
            }
        }
    }

    // Reclaim pages the retention policy no longer needs at this computed position (full history: none).
    virtual void ReclaimExpired(BlockPool& /*pool*/, BlockTable& /*table*/, std::int32_t /*num_computed_tokens*/) {}

    // Pure twin of ReclaimExpired (pages a pending reclaim would free), overridden in lockstep with it.
    virtual std::int32_t BlocksReclaimableAt(const BlockTable& /*table*/,
                                             std::int32_t /*num_computed_tokens*/,
                                             bool /*count_uncached*/) const {
        return 0;
    }

    // Cached pages keep their hash on free, so they stay prefix-reusable until evicted.
    void Free(BlockPool& pool, BlockTable& table) {
        // Free-list order is semantics -- oldest-first recycling, pinned by tests -- so keep one
        // front->back batch here; per-ref destructors only cover stray/unwind paths.
        std::vector<CacheBlock*> batch;
        batch.reserve(table.blocks_.size());
        for (BlockRef& ref : table.blocks_) {
            batch.push_back(ref.Release());
        }
        pool.FreeBlocks(batch);
        table.blocks_.clear();
        table.tail_avail_ = 0;
    }

protected:
    std::int32_t page_size_;
};

}  // namespace tokenspeed
