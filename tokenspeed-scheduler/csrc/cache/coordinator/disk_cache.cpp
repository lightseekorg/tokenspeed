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

#include "cache/coordinator/disk_cache.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "cache/coordinator/tier_eviction_planner.h"
#include "utils.h"

namespace tokenspeed {

DiskCache::DiskAllocationBatch DiskCache::AcquireDiskBlocks(std::span<const std::uint32_t> group_ids,
                                                            std::span<const std::int32_t> buckets) {
    _assert(store_.DiskPool() != nullptr, "AcquireDiskBlocks requires a Disk pool");
    _assert(group_ids.size() == buckets.size(), "Disk allocation needs one bucket per requested block");
    BlockPool& pool = *store_.DiskPool();
    DiskAllocationBatch batch;
    batch.blocks.resize(group_ids.size());
    const auto acquire_in_bucket = [&](std::uint32_t group_id, std::int32_t bucket) {
        const std::array<std::int32_t, 1> one{bucket};
        std::vector<CacheBlockRef> refs = pool.AcquireBlocksInBuckets(group_id, one);
        return refs.empty() ? CacheBlockRef{} : std::move(refs.front());
    };
    std::vector<std::size_t> unresolved;
    for (std::size_t row = 0; row < group_ids.size(); ++row) {
        _assert(group_ids[row] < store_.Groups().size(), "Disk block group id out of range");
        _assert(buckets[row] >= 0 && buckets[row] < store_.Groups()[group_ids[row]].Allocator().ShardCount(),
                "Disk block bucket is outside the group's shard count");
        batch.blocks[row] = acquire_in_bucket(group_ids[row], buckets[row]);
        if (!batch.blocks[row]) unresolved.push_back(row);
    }
    if (!unresolved.empty()) {
        std::vector<TierEvictionPlanner::VictimSet> victims;
        std::vector<std::size_t> accepted;
        {
            TierEvictionPlanner planner{store_, history_, pool, CacheTier::kDisk};
            std::vector<BlockPlacementDemand> block_demands = EmptyBlockDemands(store_);
            for (const std::size_t row : unresolved) {
                auto& bucket_blocks = block_demands[group_ids[row]].blocks_by_bucket[buckets[row]];
                ++bucket_blocks;
                if (planner.Fit(block_demands)) {
                    accepted.push_back(row);
                } else {
                    --bucket_blocks;
                }
            }
            planner.Minimize(block_demands);
            victims = planner.TakeVictims();
        }
        for (const auto& set : victims) {
            for (const auto& [group_id, location] : set.members) {
                if (store_.Groups()[group_id].Index().Contains(pool, location)) {
                    FatalCheck(evictor_.EvictDiskCachedBlock(group_id, location, &CacheEvictor::DiskEvictionStats::lru),
                               "a planned Disk victim became pinned");
                }
            }
        }
        for (const std::size_t row : accepted) {
            batch.blocks[row] = acquire_in_bucket(group_ids[row], buckets[row]);
            FatalCheck(static_cast<bool>(batch.blocks[row]), "planned Disk placement failed");
        }
    }
    batch.allocated = static_cast<std::size_t>(
        std::ranges::count_if(batch.blocks, [](const CacheBlockRef& block) { return static_cast<bool>(block); }));
    return batch;
}

void DiskCache::CacheDiskBlock(CacheBlockRef& block_ref, const CacheKey& key,
                               const PrefixCacheIndex::CachedBlockMetadata& metadata) {
    _assert(store_.DiskPool() != nullptr, "CacheDiskBlock requires a Disk pool");
    _assert(key.group_id < store_.Groups().size(), "CacheDiskBlock group id out of range");
    PrefixCacheIndex& index = store_.Groups()[key.group_id].Index();
    index.Register(*store_.DiskPool(), block_ref, key, metadata.eviction.last_access_epoch,
                   metadata.eviction.logical_block_index, metadata.eviction.boundary_kind, /*newly_cached=*/nullptr,
                   metadata.session);
    if (store_.GroupIsState(key.group_id)) {
        BoundaryTree::Node* node = store_.Boundaries().Find(key.content_hash);
        _assert(node != nullptr, "an indexed checkpoint has a tree node");
        node->latest_state_access_epoch =
            std::max({node->latest_state_access_epoch, metadata.eviction.last_access_epoch, metadata.max_access_epoch});
    }
}

bool DiskCache::ContainsDiskCachedBlock(const CacheKey& key) const {
    if (store_.DiskPool() == nullptr) {
        return false;
    }
    _assert(key.group_id < store_.Groups().size(), "Disk cache key group id out of range");
    return store_.Groups()[key.group_id].Index().Contains(*store_.DiskPool(), key);
}

CacheBlockRef DiskCache::AcquireDiskCachedBlock(const CacheKey& key) const {
    if (store_.DiskPool() == nullptr) {
        return {};
    }
    _assert(key.group_id < store_.Groups().size(), "Disk cache key group id out of range");
    return store_.Groups()[key.group_id].Index().Find(*store_.DiskPool(), key);
}

bool DiskCache::DropUnreadableEntry(const CacheKey& key) {
    if (store_.DiskPool() == nullptr) {
        return false;
    }
    return evictor_.InvalidateDiskEntry(key);
}

bool DiskCache::DropUnreadableEntry(const CacheKey& key, const CacheBlockRef& source) {
    // Another read may already have invalidated this source, and cleaning
    // may have published a new copy under the same immutable content key.
    if (!source || AcquireDiskCachedBlock(key) != source) return false;
    return DropUnreadableEntry(key);
}

std::int32_t DiskCache::NumDiskCachedBlocks() const {
    if (store_.DiskPool() == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : store_.Groups()) {
        count += group.Index().NumEntries(*store_.DiskPool());
    }
    return count;
}

std::int32_t DiskCache::NumPinnedDiskCachedBlocks() const {
    if (store_.DiskPool() == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : store_.Groups()) {
        count += group.Index().NumPinnedEntries(*store_.DiskPool());
    }
    return count;
}

std::vector<CacheKey> DiskCache::PrefixHits(const PromptPrefix& prompt) const {
    std::vector<CacheKey> hits;
    if (store_.DiskPool() == nullptr || NumDiskCachedBlocks() == 0 || prompt.max_tokens <= 0) {
        return hits;
    }
    const BlockPool& pool = *store_.DiskPool();
    const PreparedPrefixKeys keys = prober_.ResolvePreparedPrefixKeys(prompt);
    const auto num_pages = static_cast<std::size_t>(prompt.max_tokens / store_.PrefixGranularity());
    for (const std::size_t i : store_.MatchOrder()) {
        const PrefixCacheIndex& index = store_.Groups()[i].Index();
        const auto end =
            std::min((*keys)[i].size(),
                     num_pages * static_cast<std::size_t>(store_.BlocksPerBoundary(static_cast<std::int32_t>(i))));
        for (std::size_t j = 0; j < end; ++j) {
            if (index.Contains(pool, (*keys)[i][j])) {
                hits.push_back((*keys)[i][j]);
            }
        }
    }
    // Tails inside any page the prompt reaches, as a probe enumerates them.
    prober_.ForEachCandidateBoundary(prompt, /*floor_tokens=*/0, prompt.max_tokens,
                                     [&](std::int32_t, std::int32_t tail_tokens, const ContentHash& hash) {
                                         if (tail_tokens == 0) {
                                             return true;
                                         }
                                         for (const std::size_t i : store_.MatchOrder()) {
                                             for (CacheKey& key : store_.TailKeysForGroup(i, hash, tail_tokens)) {
                                                 if (store_.Groups()[i].Index().Contains(pool, key)) {
                                                     hits.push_back(std::move(key));
                                                 }
                                             }
                                         }
                                         return true;
                                     });
    return hits;
}

}  // namespace tokenspeed
