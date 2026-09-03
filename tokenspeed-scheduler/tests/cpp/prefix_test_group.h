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
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "cache/coordinator/group_geometry.h"
#include "cache/core/block_pool.h"
#include "cache/core/block_table.h"
#include "cache/core/cache_types.h"
#include "cache/allocator/group_allocator.h"
#include "cache/prefix/prefix_index.h"
#include "cache/prefix/prefix_matcher.h"

namespace tokenspeed::test {

// Test-only composite of one cache group's production pieces (token-free
// allocator + GroupGeometry + matcher + PrefixCacheIndex) exposing the
// pre-split token-based convenience API, so single-group tests keep reading
// naturally. Production composes the same pieces in CacheGroup plus the
// coordinator's geometry.
class PrefixTestGroup {
public:
    PrefixTestGroup(std::int32_t block_granularity, std::int32_t cache_blocks_per_lcm_block, std::uint32_t group_id,
                    std::int32_t sliding_window)
        : geometry_{block_granularity},
          allocator_{cache_blocks_per_lcm_block, group_id},
          index_{group_id},
          sliding_window_{sliding_window} {}

    // --- geometry / identity ---
    std::int32_t BlockGranularity() const { return geometry_.BlockGranularity(); }
    std::int32_t CacheBlocksPerLcmBlock() const { return allocator_.CacheBlocksPerLcmBlock(); }
    std::uint32_t Id() const { return allocator_.Id(); }
    PrefixCacheIndex& Index() { return index_; }
    const PrefixCacheIndex& Index() const { return index_; }

    // --- allocation (token API, converted through GroupGeometry) ---
    bool Acquire(BlockPool& pool, BlockTable& table, std::int32_t num_tokens, std::int32_t reserve_tokens = 0) {
        return allocator_.Acquire(pool, table, geometry_.PlanAcquire(table, num_tokens, reserve_tokens));
    }
    bool Acquire(BlockPool& pool, BlockTable& table, const GroupDemand& demand) {
        return allocator_.Acquire(pool, table, geometry_.PlanAcquire(table, demand));
    }
    std::int32_t BlocksNeededFor(const BlockTable& table, std::int32_t num_tokens) const {
        return geometry_.BlocksNeededFor(table, num_tokens);
    }
    void ClaimHitBlocks(BlockTable& table, PrefixMatch&& hit) { allocator_.ClaimHitBlocks(table, std::move(hit)); }
    void ConsumeReservedTokens(BlockTable& table, std::int32_t num_tokens) {
        allocator_.ConsumeReservedTokens(table, num_tokens);
    }
    void Free(BlockTable& table) { allocator_.Free(table); }
    std::int32_t ResolveCacheBlockId(CacheBlockLocation location) const {
        return allocator_.ResolveCacheBlockId(location);
    }
    std::vector<std::int32_t> BlockTablePageIds(const BlockTable& table) const {
        return allocator_.BlockTablePageIds(table);
    }
    void AppendHostExtension(BlockPool& pool, BlockTable& table, std::vector<CacheBlockRef>&& host_block_refs,
                             std::vector<BlockTransfer>& load_pairs) {
        allocator_.AppendHostExtension(pool, table, std::move(host_block_refs), load_pairs);
    }

    // --- retention (token API, policy converted through GroupGeometry) ---
    void ReclaimExpired(BlockPool& pool, BlockTable& table, std::int32_t num_computed_tokens) {
        allocator_.ReclaimExpired(pool, table, expiredBlocksAt(num_computed_tokens));
    }
    std::int32_t BlocksReclaimableAt(const BlockTable& table, std::int32_t num_computed_tokens,
                                     bool count_uncached) const {
        return allocator_.BlocksReclaimableAt(index_, table, expiredBlocksAt(num_computed_tokens), count_uncached);
    }
    std::vector<CacheBlockLocation> ReclaimableBlockLocationsAt(const BlockTable& table,
                                                                std::int32_t num_computed_tokens) const {
        return allocator_.ReclaimableBlockLocationsAt(index_, table, expiredBlocksAt(num_computed_tokens));
    }

    // --- prefix matching / index (pre-split convenience API) ---
    bool MatchIsPrefixClosed() const { return sliding_window_ == 0; }

    GroupPrefixProbe Probe(const BlockPool& pool, std::span<const CacheKey> keys, std::int32_t begin_blocks,
                           std::int32_t max_blocks) const {
        if (sliding_window_ > 0) {
            return SwaMatcher(geometry_.BlockGranularity(), sliding_window_)
                .Probe(index_, pool, keys, begin_blocks, max_blocks);
        }
        return FullAttnMatcher{}.Probe(index_, pool, keys, begin_blocks, max_blocks);
    }

    PrefixMatch Match(BlockPool& pool, std::span<const CacheKey> keys, std::int32_t begin_blocks,
                      std::int32_t max_blocks) {
        return index_.AcquireMatched(pool, keys, begin_blocks, Probe(pool, keys, begin_blocks, max_blocks),
                                     ++next_access_epoch_);
    }
    PrefixMatch AcquireMatchedBlocks(BlockPool& pool, std::span<const CacheKey> keys, std::int32_t begin_blocks,
                                     const GroupPrefixProbe& probe, std::uint64_t access_epoch) {
        return index_.AcquireMatched(pool, keys, begin_blocks, probe, access_epoch);
    }

    void RegisterCachedBlock(BlockPool& pool, CacheBlockRef& block, const CacheKey& key) {
        index_.Register(pool, block, key, ++next_access_epoch_);
    }
    void RegisterCachedBlock(BlockPool& pool, CacheBlockRef& block, const CacheKey& key, std::uint64_t access_epoch,
                             std::int32_t logical_block_index = -1,
                             CacheBoundaryKind boundary_kind = CacheBoundaryKind::kChunk) {
        index_.Register(pool, block, key, access_epoch, logical_block_index, boundary_kind);
    }
    void CacheFullBlocks(BlockPool& pool, BlockTable& table, std::span<const CacheKey> keys,
                         std::int32_t first_slot = 0) {
        index_.RegisterFullBlocks(pool, table, keys, ++next_access_epoch_, first_slot);
    }

    bool ContainsCachedBlock(const BlockPool& pool, const CacheKey& key) const { return index_.Contains(pool, key); }
    bool ContainsCachedBlock(const BlockPool& pool, CacheBlockLocation location) const {
        return index_.Contains(pool, location);
    }
    std::optional<CacheKey> EvictCachedBlock(const BlockPool& pool, CacheBlockLocation location) {
        return index_.Evict(pool, location);
    }
    std::int32_t NumCachedBlocks(const BlockPool& pool) const { return index_.NumEntries(pool); }
    std::vector<CacheBlockLocation> EvictableBlockLocations(const BlockPool& pool) const {
        return index_.EvictableLocations(pool);
    }
    std::optional<PrefixCacheIndex::CachedBlockMetadata> CachedBlockMetadataFor(const BlockPool& pool,
                                                                                CacheBlockLocation location) const {
        return index_.MetadataFor(pool, location);
    }
    bool ParentIsFullyEvictable(const BlockPool& pool, std::int32_t lcm_block_id) const {
        return index_.ParentIsFullyEvictable(pool, lcm_block_id, allocator_.CacheBlocksPerLcmBlock());
    }

private:
    std::int32_t expiredBlocksAt(std::int32_t num_computed_tokens) const {
        if (sliding_window_ == 0) {
            return 0;
        }
        const CacheGroupSpec spec{.kind = AttnKind::kSlidingWindow,
                                  .sliding_window = sliding_window_,
                                  .block_granularity = geometry_.BlockGranularity()};
        return geometry_.ExpiredBlocksAt(spec, num_computed_tokens);
    }

    GroupGeometry geometry_;
    GroupAllocator allocator_;
    PrefixCacheIndex index_;
    std::int32_t sliding_window_;
    std::uint64_t next_access_epoch_{0};
};

// The pre-split class names, so test bodies keep reading naturally.
class FullAttnManager : public PrefixTestGroup {
public:
    explicit FullAttnManager(std::int32_t block_granularity, std::int32_t cache_blocks_per_lcm_block = 1,
                             std::uint32_t group_id = 0)
        : PrefixTestGroup(block_granularity, cache_blocks_per_lcm_block, group_id, /*sliding_window=*/0) {}
};

class SwaManager : public PrefixTestGroup {
public:
    SwaManager(std::int32_t block_granularity, std::int32_t sliding_window)
        : PrefixTestGroup(block_granularity, /*cache_blocks_per_lcm_block=*/1, /*group_id=*/0, sliding_window) {}
    SwaManager(std::int32_t block_granularity, std::int32_t cache_blocks_per_lcm_block, std::int32_t sliding_window,
               std::uint32_t group_id = 0)
        : PrefixTestGroup(block_granularity, cache_blocks_per_lcm_block, group_id, sliding_window) {}
};

}  // namespace tokenspeed::test
