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
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/block_ref.h"
#include "cache/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// Pure per-attention-type policy over block size and retention. Managers own no
// pool and no request state; every page lifetime remains in BlockRef.
class KvCacheManager {
public:
    explicit KvCacheManager(std::int32_t block_size) : block_size_{block_size} {
        _assert(block_size > 0, "block_size must be > 0");
    }
    virtual ~KvCacheManager() = default;

    KvCacheManager(const KvCacheManager&) = delete;
    KvCacheManager& operator=(const KvCacheManager&) = delete;

    virtual bool MatchIsPrefixClosed() const = 0;

    // Probe [begin_blocks, max_blocks) without changing refcounts or LRU state.
    virtual PrefixProbe Probe(const BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                              std::int32_t max_blocks) const = 0;

    // Probe one already-selected endpoint. A miss is distinct from a valid
    // empty retained range, as used by a W=1 continuation-state group.
    virtual std::optional<PrefixProbe> ProbeExactBoundary(const BlockPool& pool, std::span<const std::string> keys,
                                                          std::int32_t begin_blocks,
                                                          std::int32_t boundary_blocks) const {
        if (begin_blocks < 0 || boundary_blocks < begin_blocks ||
            boundary_blocks > static_cast<std::int32_t>(keys.size())) {
            return std::nullopt;
        }
        PrefixProbe probe = Probe(pool, keys, begin_blocks, boundary_blocks);
        if (probe.base_logical_page != begin_blocks || probe.LogicalEnd() != boundary_blocks ||
            std::ranges::any_of(probe.hits, [](CachedBlockState state) { return !IsCachedBlockHit(state); })) {
            return std::nullopt;
        }
        return probe;
    }

    PrefixMatch AcquireMatchedBlocks(BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                                     const PrefixProbe& probe) const {
        _assert(begin_blocks >= 0 && probe.base_logical_page >= begin_blocks,
                "matched block range begins before its probe floor");
        _assert(probe.LogicalEnd() <= static_cast<std::int32_t>(keys.size()), "matched block range is out of bounds");
        PrefixMatch match;
        match.base_logical_page = probe.base_logical_page;
        match.blocks.resize(probe.hits.size());
        for (std::size_t i = 0; i < probe.hits.size(); ++i) {
            if (!IsCachedBlockHit(probe.hits[i])) {
                continue;
            }
            const std::size_t logical = static_cast<std::size_t>(probe.base_logical_page) + i;
            BlockRef block = pool.AcquireCachedBlock(keys[logical]);
            _assert(static_cast<bool>(block), "cached block disappeared between probe and acquisition");
            match.blocks[i] = std::move(block);
            ++match.num_hit_blocks;
        }
        return match;
    }

    PrefixMatch Match(BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                      std::int32_t max_blocks) const {
        return AcquireMatchedBlocks(pool, keys, begin_blocks, Probe(pool, keys, begin_blocks, max_blocks));
    }

    void ClaimHitBlocks(BlockTable& table, PrefixMatch&& hit) {
        _assert(table.blocks_.empty() && table.BaseLogicalPage() == 0, "ClaimHitBlocks requires a fresh (empty) table");
        for (const BlockRef& block : hit.blocks) {
            _assert(!block || block->IsCached(), "matched block lost its hash before the claim");
        }
        table.InitRange(hit.base_logical_page, std::move(hit.blocks));
    }

    // Convenience all-or-nothing acquisition. Coordinator transactions use
    // ReserveLiveSize + AcquireBlocksInto + CommitAcquire instead.
    bool Acquire(BlockPool& pool, BlockTable& table, std::int32_t num_tokens) {
        const std::int32_t num_pages = BlocksNeededFor(table, num_tokens);
        if (num_pages == 0) {
            if (num_tokens > 0) {
                table.tail_avail_ -= num_tokens;
            }
            return true;
        }
        table.ReserveLiveSize(table.LiveSize() + num_pages);
        std::vector<BlockRef> fresh_blocks = pool.AcquireBlocks(num_pages);
        if (static_cast<std::int32_t>(fresh_blocks.size()) != num_pages) {
            return false;
        }
        CommitAcquire(pool, table, num_tokens, fresh_blocks);
        return true;
    }

    // No-allocation commit half. Every incoming handle is moved exactly once
    // into the pre-reserved table, keeping BlockRef as the sole lifecycle API.
    void CommitAcquire(BlockPool& pool, BlockTable& table, std::int32_t num_tokens,
                       std::span<BlockRef> fresh_blocks) noexcept {
        const std::int32_t expected = BlocksNeededFor(table, num_tokens);
        if (expected != static_cast<std::int32_t>(fresh_blocks.size()) ||
            table.blocks_.capacity() < table.blocks_.size() + fresh_blocks.size()) {
            std::terminate();
        }
        for (const BlockRef& block : fresh_blocks) {
            if (!block || !block.IsOwnedBy(pool)) {
                std::terminate();
            }
        }
        if (num_tokens <= 0) {
            return;
        }
        if (num_tokens <= table.tail_avail_) {
            table.tail_avail_ -= num_tokens;
            return;
        }
        const std::int32_t over = num_tokens - table.tail_avail_;
        for (BlockRef& block : fresh_blocks) {
            table.blocks_.push_back(std::move(block));
        }
        const std::int32_t used_in_tail = over % block_size_;
        table.tail_avail_ = used_in_tail == 0 ? 0 : block_size_ - used_in_tail;
    }

    // Admission pre-charged destination slots. Empty source handles preserve
    // absolute-layout holes without manufacturing a sentinel page object.
    void AppendHostExtension(BlockPool& pool, BlockTable& table, std::vector<BlockRef>&& host_blocks,
                             std::vector<BlockTransfer>& load_pairs) {
        _assert(table.tail_avail_ == 0, "host extension must append on a full-page boundary");
        table.ReserveLiveSize(table.LiveSize() + static_cast<std::int32_t>(host_blocks.size()));
        load_pairs.reserve(load_pairs.size() + host_blocks.size());
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

    std::int32_t BlocksNeededFor(const BlockTable& table, std::int32_t num_tokens) const noexcept {
        if (num_tokens <= 0 || num_tokens <= table.tail_avail_) {
            return 0;
        }
        const std::int32_t over = num_tokens - table.tail_avail_;
        return (over + block_size_ - 1) / block_size_;
    }

    virtual bool RegistersAlignedFinalPageOnly() const { return false; }

    void CacheFullBlocks(BlockPool& pool, BlockTable& table, std::span<const std::string> block_hashes,
                         std::int32_t first_slot = 0,
                         std::vector<std::pair<std::string, BlockRef>>* newly_cached = nullptr) {
        _assert(first_slot >= table.BaseLogicalPage(), "hash range begins before the live logical range");
        _assert(static_cast<std::int64_t>(first_slot) + static_cast<std::int64_t>(block_hashes.size()) <=
                    table.LogicalEnd(),
                "hash range exceeds table logical range");
        for (std::size_t j = 0; j < block_hashes.size(); ++j) {
            const std::int64_t logical = static_cast<std::int64_t>(first_slot) + static_cast<std::int64_t>(j);
            _assert(logical <= std::numeric_limits<std::int32_t>::max(), "hash logical page overflows int32");
            BlockRef& block = table.AtLogical(static_cast<std::int32_t>(logical));
            if (!block || block->IsCached()) {
                continue;
            }
            pool.CacheFullBlock(block, block_hashes[j]);
            if (newly_cached != nullptr) {
                newly_cached->emplace_back(block_hashes[j], block);
            }
        }
    }

    virtual void ReclaimExpired(BlockPool& /*pool*/, BlockTable& /*table*/,
                                std::int32_t /*num_computed_tokens*/) noexcept {}

    virtual std::int32_t BlocksReclaimableAt(const BlockTable& /*table*/, std::int32_t /*num_computed_tokens*/,
                                             bool /*count_uncached*/) const {
        return 0;
    }

    // Quiescent-point tail rewind. Pages through retain_raw_end stay pinned;
    // later whole pages are released tail-first. Retained rows after the
    // accepted cursor become aggregate tail capacity.
    void RewindTail(BlockPool& /*pool*/, BlockTable& table, std::int32_t accepted_raw_end,
                    std::int32_t retain_raw_end) noexcept {
        if (accepted_raw_end < 0 || retain_raw_end < accepted_raw_end) {
            std::terminate();
        }
        const std::int64_t keep_end64 = (static_cast<std::int64_t>(retain_raw_end) + block_size_ - 1) / block_size_;
        if (keep_end64 > std::numeric_limits<std::int32_t>::max()) {
            std::terminate();
        }
        const std::int32_t keep_logical_end =
            std::clamp(static_cast<std::int32_t>(keep_end64), table.BaseLogicalPage(), table.LogicalEnd());
        const std::int64_t kept_raw_end = static_cast<std::int64_t>(keep_logical_end) * block_size_;
        if (accepted_raw_end > kept_raw_end ||
            kept_raw_end - accepted_raw_end > std::numeric_limits<std::int32_t>::max()) {
            std::terminate();
        }

        const std::int32_t keep_live = keep_logical_end - table.BaseLogicalPage();
        for (std::int32_t local = table.LiveSize(); local > keep_live; --local) {
            table.blocks_[static_cast<std::size_t>(local - 1)].reset();
        }
        table.blocks_.erase(table.blocks_.begin() + keep_live, table.blocks_.end());
        table.tail_avail_ = static_cast<std::int32_t>(kept_raw_end - accepted_raw_end);
    }

    void Free(BlockTable& table) noexcept { table.Reset(); }

    void Free(BlockPool& /*pool*/, BlockTable& table) noexcept { table.Reset(); }

protected:
    std::int32_t block_size_;
};

}  // namespace tokenspeed
