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
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/block_ref.h"
#include "cache/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// Physical layout a decode-side PD destination needs for one cache group.
// DenseHistory keeps every prompt page; FinalStateSnapshot keeps logical
// alignment with null holes and materializes only the prompt-final state page
// plus the page needed by the first local decode token.
enum class DecodeDestinationLayout {
    kDenseHistory,
    kFinalStateSnapshot,
};

// Pure per-attention-type policy over block_size (+ window): holds no pool and no per-request
// state -- every operation acts on the pool it is handed, identically for any tier.
class KvCacheManager {
public:
    explicit KvCacheManager(std::int32_t block_size) : block_size_{block_size} {
        _assert(block_size > 0, "block_size must be > 0");
    }
    virtual ~KvCacheManager() = default;

    KvCacheManager(const KvCacheManager&) = delete;
    KvCacheManager& operator=(const KvCacheManager&) = delete;

    // Downward-closed matches: any prefix of a valid match is itself valid, so the
    // coordinator may match once and trim late; non-closed managers re-match bound-first.
    virtual bool MatchIsPrefixClosed() const = 0;

    // Probe slots [begin_blocks, max_blocks) without changing pool ownership.
    // Hit slots are relative to begin_blocks.
    virtual PrefixProbe Probe(const BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                              std::int32_t max_blocks) const = 0;

    PrefixMatch AcquireMatchedBlocks(BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                                     const PrefixProbe& probe) const {
        _assert(begin_blocks >= 0 && static_cast<std::size_t>(begin_blocks) + probe.hits.size() <= keys.size(),
                "matched block range is out of bounds");
        PrefixMatch match;
        match.blocks.resize(probe.hits.size());
        for (std::size_t i = 0; i < probe.hits.size(); ++i) {
            if (probe.hits[i] == 0) {
                continue;
            }
            BlockRef block = pool.AcquireCachedBlock(keys[static_cast<std::size_t>(begin_blocks) + i]);
            _assert(static_cast<bool>(block), "cached block disappeared between match probe and acquisition");
            match.blocks[i] = std::move(block);
            ++match.num_hit_blocks;
        }
        return match;
    }

    PrefixMatch Match(BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                      std::int32_t max_blocks) const {
        return AcquireMatchedBlocks(pool, keys, begin_blocks, Probe(pool, keys, begin_blocks, max_blocks));
    }

    // Move the already-pinned match into the request table.
    void ClaimHitBlocks(BlockTable& table, PrefixMatch&& hit) {
        _assert(table.blocks_.empty(), "ClaimHitBlocks requires a fresh (empty) table");
        for (const BlockRef& block : hit.blocks) {
            _assert(!block || block->IsCached(), "matched block lost its hash before the claim");
        }
        table.blocks_ = std::move(hit.blocks);
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
        std::int32_t num_pages = (over + block_size_ - 1) / block_size_;
        std::vector<BlockRef> new_blocks = pool.AcquireBlocks(num_pages);
        if (static_cast<std::int32_t>(new_blocks.size()) < num_pages) {
            return false;
        }
        for (BlockRef& block : new_blocks) {
            table.blocks_.push_back(std::move(block));
        }
        std::int32_t used_in_tail = over % block_size_;
        table.tail_avail_ = (used_in_tail == 0) ? 0 : block_size_ - used_in_tail;
        return true;
    }

    // Contract on the forward_cache_ops facade; admission pre-charged the real slots via ext_real_pages.
    void AppendHostExtension(BlockPool& pool, BlockTable& table, std::vector<BlockRef>&& host_blocks,
                             std::vector<BlockTransfer>& load_pairs) {
        _assert(table.tail_avail_ == 0, "host extension must append on a full-page boundary");
        for (BlockRef& host_block : host_blocks) {
            if (!host_block) {
                table.blocks_.emplace_back();
                continue;
            }
            const bool acquired = Acquire(pool, table, block_size_);
            _assert(acquired, "pre-checked Acquire must succeed");
            load_pairs.push_back(BlockTransfer{std::move(host_block), table.blocks_.back()});
        }
    }

    // Pure query mirroring Acquire's page math exactly.
    std::int32_t BlocksNeededFor(const BlockTable& table, std::int32_t num_tokens) const {
        if (num_tokens <= table.tail_avail_) {
            return 0;
        }
        std::int32_t over = num_tokens - table.tail_avail_;
        return (over + block_size_ - 1) / block_size_;
    }

    // Models opt into the final-snapshot layout by overriding this policy. The
    // coordinator and PD lifecycle stay group-generic.
    virtual DecodeDestinationLayout DecodeDestinationLayoutPolicy() const {
        return DecodeDestinationLayout::kDenseHistory;
    }

    // Exact, side-effect-free page count for a decode-side PD destination.
    // Dense groups append the prompt suffix and the first local decode input.
    // Final-state groups preserve absolute logical slots with null holes and
    // materialize only prompt-final..reserve-final inclusive.
    std::int32_t DecodeDestinationBlocksNeeded(const BlockTable& table, std::int32_t prompt_tokens,
                                               std::int32_t remaining_prompt_tokens,
                                               std::int32_t reserve_tokens) const {
        _assert(prompt_tokens > 0, "decode destination requires a non-empty prompt");
        _assert(remaining_prompt_tokens >= 0 && remaining_prompt_tokens <= prompt_tokens,
                "invalid decode destination prompt suffix");
        _assert(reserve_tokens >= 0, "decode destination reserve must be non-negative");

        if (DecodeDestinationLayoutPolicy() == DecodeDestinationLayout::kDenseHistory) {
            const std::int64_t extent = static_cast<std::int64_t>(remaining_prompt_tokens) + reserve_tokens;
            _assert(extent <= std::numeric_limits<std::int32_t>::max(), "dense decode destination extent overflow");
            return BlocksNeededFor(table, static_cast<std::int32_t>(extent));
        }

        // The Flat PD baseline intentionally supports one local decode input.
        // Wider speculative/MTP reservations need a model-defined contract for
        // which intermediate state snapshots are live, so fail closed here.
        _assert(reserve_tokens <= 1, "final-state decode destination supports at most one reserved token");
        const std::int64_t extent64 = static_cast<std::int64_t>(prompt_tokens) + reserve_tokens;
        _assert(extent64 <= std::numeric_limits<std::int32_t>::max(), "final-state decode destination extent overflow");
        const std::int32_t extent_tokens = static_cast<std::int32_t>(extent64);
        const std::int32_t first_real_slot = (prompt_tokens - 1) / block_size_;
        const std::int32_t last_real_slot = (extent_tokens - 1) / block_size_;
        _assert(table.NumBlocks() <= last_real_slot + 1, "state prefix exceeds decode destination extent");
        for (const BlockRef& block : table.Blocks()) {
            _assert(!block, "decode destination state prefix must contain null holes only");
        }
        return last_real_slot - first_real_slot + 1;
    }

    // All-or-nothing for one group. The coordinator prechecks the sum across
    // every group before calling this, so a multi-group destination is atomic.
    bool AcquireDecodeDestination(BlockPool& pool, BlockTable& table, std::int32_t prompt_tokens,
                                  std::int32_t remaining_prompt_tokens, std::int32_t reserve_tokens) {
        if (DecodeDestinationLayoutPolicy() == DecodeDestinationLayout::kDenseHistory) {
            const std::int64_t extent = static_cast<std::int64_t>(remaining_prompt_tokens) + reserve_tokens;
            _assert(extent <= std::numeric_limits<std::int32_t>::max(), "dense decode destination extent overflow");
            return Acquire(pool, table, static_cast<std::int32_t>(extent));
        }

        const std::int32_t needed =
            DecodeDestinationBlocksNeeded(table, prompt_tokens, remaining_prompt_tokens, reserve_tokens);
        std::vector<BlockRef> new_blocks = pool.AcquireBlocks(needed);
        if (static_cast<std::int32_t>(new_blocks.size()) != needed) {
            return false;
        }

        const std::int32_t extent_tokens = static_cast<std::int32_t>(static_cast<std::int64_t>(prompt_tokens) +
                                                                     static_cast<std::int64_t>(reserve_tokens));
        const std::int32_t first_real_slot = (prompt_tokens - 1) / block_size_;
        const std::int32_t last_real_slot = (extent_tokens - 1) / block_size_;
        while (table.NumBlocks() <= last_real_slot) {
            table.blocks_.emplace_back();
        }
        std::size_t next = 0;
        for (std::int32_t slot = first_real_slot; slot <= last_real_slot; ++slot) {
            BlockRef& ref = table.blocks_[static_cast<std::size_t>(slot)];
            _assert(!ref, "decode destination state slot aliases cached state");
            ref = std::move(new_blocks[next++]);
        }
        _assert(next == new_blocks.size(), "decode destination state allocation count drifted");
        const std::int32_t used_in_tail = extent_tokens % block_size_;
        table.tail_avail_ = used_in_tail == 0 ? 0 : block_size_ - used_in_tail;
        return true;
    }

    // State snapshots are only boundary-correct where a forward call ended page-aligned:
    // such groups register just the final full page of an aligned range.
    virtual bool RegistersAlignedFinalPageOnly() const { return false; }

    // Pages already carrying a hash are skipped; the partial tail is excluded by the caller.
    void CacheFullBlocks(BlockPool& pool, BlockTable& table, std::span<const std::string> block_hashes,
                         std::int32_t first_slot = 0,
                         std::vector<std::pair<std::string, BlockRef>>* newly_cached = nullptr) {
        _assert(first_slot >= 0, "first_slot must be >= 0");
        _assert(
            static_cast<std::int64_t>(first_slot) + static_cast<std::int64_t>(block_hashes.size()) <= table.NumBlocks(),
            "hash range exceeds table size");
        for (std::size_t j = 0; j < block_hashes.size(); ++j) {
            const BlockRef& block_ref = table.blocks_[static_cast<std::size_t>(first_slot) + j];
            if (!block_ref) {
                continue;
            }
            if (block_ref->IsCached()) {
                continue;
            }
            pool.CacheFullBlock(block_ref, block_hashes[j]);
            if (newly_cached != nullptr) {
                newly_cached->emplace_back(block_hashes[j], table.blocks_[static_cast<std::size_t>(first_slot) + j]);
            }
        }
    }

    // Reclaim pages the retention policy no longer needs at this computed position (full history: none).
    virtual void ReclaimExpired(BlockPool& /*pool*/, BlockTable& /*table*/, std::int32_t /*num_computed_tokens*/) {}

    // Pure twin of ReclaimExpired (pages a pending reclaim would free), overridden in lockstep with it.
    virtual std::int32_t BlocksReclaimableAt(const BlockTable& /*table*/, std::int32_t /*num_computed_tokens*/,
                                             bool /*count_uncached*/) const {
        return 0;
    }

    // Capacity upper bound used only to distinguish permanent OOM from
    // contention: count pages this request's pending retention slide could
    // release after other references disappear.
    virtual std::int32_t BlocksReclaimableIgnoringRefsAt(const BlockTable& /*table*/,
                                                         std::int32_t /*num_computed_tokens*/) const {
        return 0;
    }

    // Cached pages keep their hash on free, so they stay prefix-reusable until evicted.
    void Free(BlockTable& table) {
        // Release in reverse explicitly; vector destruction order is not the
        // free-list policy and must not choose the eviction order for us.
        for (auto it = table.blocks_.rbegin(); it != table.blocks_.rend(); ++it) {
            it->reset();
        }
        table.blocks_.clear();
        table.tail_avail_ = 0;
    }

protected:
    std::int32_t block_size_;
};

}  // namespace tokenspeed
