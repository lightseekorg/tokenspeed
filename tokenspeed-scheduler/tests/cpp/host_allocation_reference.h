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
#include <array>
#include <limits>
#include <tuple>

#include "cache/coordinator/cache_coordinator.h"

namespace tokenspeed::test {

// Collect/sort reference retained to check exact placement and victim order.
inline CacheCoordinator::HostAllocationBatch AllocateHostReference(CacheCoordinator& coordinator, BlockPool* host_pool,
                                                                   std::span<const std::uint32_t> group_ids,
                                                                   std::span<const std::int32_t> buckets) {
    const auto evict = [&](std::uint32_t group, CacheBlockLocation location) {
        return coordinator.GroupPrefixIndex(group).Evict(*host_pool, location).has_value();
    };
    CacheCoordinator::HostAllocationBatch batch;
    batch.blocks.resize(group_ids.size());
    batch.stats.requested = group_ids.size();
    _assert(group_ids.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "Host allocation batch request exceeds int32 range");

    // A Host block lands in the bucket of the Device block it mirrors, so under
    // page-cyclic sharding both ends of a copy have the same owner.
    const auto acquire_in_bucket = [&](std::uint32_t group_id, std::int32_t bucket) {
        const std::array<std::int32_t, 1> one{bucket};
        std::vector<CacheBlockRef> refs = host_pool->AcquireBlocksInBuckets(group_id, one);
        return refs.empty() ? CacheBlockRef{} : std::move(refs.front());
    };

    std::vector<std::uint32_t> group_order;
    group_order.reserve(static_cast<std::size_t>(coordinator.NumGroups()));
    std::vector<bool> seen_group(static_cast<std::size_t>(coordinator.NumGroups()), false);
    std::vector<std::vector<std::size_t>> unresolved_by_group(static_cast<std::size_t>(coordinator.NumGroups()));
    for (std::size_t i = 0; i < group_ids.size(); ++i) {
        _assert(group_ids[i] < static_cast<std::size_t>(coordinator.NumGroups()), "Host block group id out of range");
        _assert(buckets[i] >= 0 && buckets[i] < coordinator.Allocator(group_ids[i]).ShardCount(),
                "Host block bucket is outside the group's shard count");
        if (!seen_group[group_ids[i]]) {
            seen_group[group_ids[i]] = true;
            group_order.push_back(group_ids[i]);
        }
        // Free capacity first, in candidate order.
        batch.blocks[i] = acquire_in_bucket(group_ids[i], buckets[i]);
        if (!batch.blocks[i]) {
            unresolved_by_group[group_ids[i]].push_back(i);
        }
    }

    // Retention value of one Host cache entry: keep what a request has already
    // proven useful, then the most recently accessed, then a stable tie-break.
    const auto value = [](std::uint32_t candidate_group, CacheBlockLocation location,
                          const PrefixCacheIndex::CachedBlockMetadata& metadata) {
        return std::tuple{metadata.was_acquired, metadata.last_access_epoch, candidate_group, location.lcm_block_id,
                          location.slot_index};
    };
    const auto lookup_value = [&](std::uint32_t candidate_group, CacheBlockLocation location) {
        const auto metadata = coordinator.GroupPrefixIndex(candidate_group).MetadataFor(*host_pool, location);
        _assert(metadata.has_value(), "evictable Host block has no cache metadata");
        return value(candidate_group, location, *metadata);
    };
    using HostCacheValue = decltype(lookup_value(std::uint32_t{}, CacheBlockLocation{}));

    for (std::uint32_t group_id : group_order) {
        std::vector<std::size_t>& unresolved = unresolved_by_group[group_id];
        if (unresolved.empty()) {
            continue;
        }
        ++batch.stats.same_group_scans;
        // Host retention ranks was_acquired before epoch, unlike the eviction
        // index, so finding the best victims needs a full group scan; each
        // unresolved block then evicts the least valuable entry in its own
        // bucket, which is exactly the hole it needs.
        std::vector<PrefixCacheIndex::EvictionCandidate> local_victims =
            coordinator.GroupPrefixIndex(group_id).EvictableCandidates(*host_pool);
        std::ranges::sort(local_victims, {}, [&](const PrefixCacheIndex::EvictionCandidate& candidate) {
            return value(group_id, candidate.location, candidate.metadata);
        });
        const GroupAllocator& allocator = coordinator.Allocator(group_id);
        std::vector<bool> consumed(local_victims.size(), false);
        std::erase_if(unresolved, [&](std::size_t index) {
            const std::int32_t bucket = buckets[index];
            for (std::size_t v = 0; v < local_victims.size(); ++v) {
                if (consumed[v] || allocator.BucketOf(local_victims[v].location) != bucket) {
                    continue;
                }
                consumed[v] = true;
                _assert(evict(group_id, local_victims[v].location), "selected Host child is not evictable");
                batch.blocks[index] = acquire_in_bucket(group_id, bucket);
                _assert(static_cast<bool>(batch.blocks[index]), "evicting a Host child did not free its bucket");
                return true;
            }
            return false;
        });
    }
    const bool has_unresolved = std::ranges::any_of(
        unresolved_by_group, [](const std::vector<std::size_t>& unresolved) { return !unresolved.empty(); });
    if (has_unresolved) {
        ++batch.stats.cross_group_scans;
        std::vector<std::pair<HostCacheValue, std::int32_t>> victim_parents;
        victim_parents.reserve(static_cast<std::size_t>(host_pool->NumLcmBlocks()));
        for (std::int32_t parent_id = 1; parent_id <= host_pool->NumLcmBlocks(); ++parent_id) {
            const std::optional<std::uint32_t> bound_group = host_pool->BoundGroup(parent_id);
            if (!bound_group ||
                !coordinator.GroupPrefixIndex(*bound_group)
                     .ParentIsFullyEvictable(*host_pool, parent_id,
                                             coordinator.Allocator(*bound_group).CacheBlocksPerLcmBlock())) {
                continue;
            }
            std::optional<HostCacheValue> parent_value;
            for (std::int32_t slot = 0; slot < coordinator.Allocator(*bound_group).CacheBlocksPerLcmBlock(); ++slot) {
                const CacheBlockLocation location{.lcm_block_id = parent_id, .slot_index = slot};
                if (!host_pool->IsOccupied(location)) {
                    continue;
                }
                const auto child_value = lookup_value(*bound_group, location);
                parent_value = parent_value ? std::max(*parent_value, child_value) : child_value;
            }
            _assert(parent_value.has_value(), "evictable Host parent has no children");
            victim_parents.emplace_back(*parent_value, parent_id);
        }
        std::ranges::sort(victim_parents);

        std::size_t victim_index = 0;
        for (std::size_t result_index = 0; result_index < group_ids.size() && victim_index < victim_parents.size();
             ++result_index) {
            if (batch.blocks[result_index]) {
                continue;
            }
            const std::uint32_t target_group = group_ids[result_index];
            std::vector<std::size_t>& unresolved = unresolved_by_group[target_group];
            while (!unresolved.empty() && victim_index < victim_parents.size()) {
                const std::int32_t victim_parent = victim_parents[victim_index++].second;
                const std::optional<std::uint32_t> bound_group = host_pool->BoundGroup(victim_parent);
                if (!bound_group ||
                    !coordinator.GroupPrefixIndex(*bound_group)
                         .ParentIsFullyEvictable(*host_pool, victim_parent,
                                                 coordinator.Allocator(*bound_group).CacheBlocksPerLcmBlock())) {
                    continue;
                }
                const std::int32_t victim_packing = coordinator.Allocator(*bound_group).CacheBlocksPerLcmBlock();
                for (std::int32_t slot = 0; slot < victim_packing; ++slot) {
                    const CacheBlockLocation location{.lcm_block_id = victim_parent, .slot_index = slot};
                    if (host_pool->IsOccupied(location)) {
                        _assert(evict(*bound_group, location), "selected Host parent changed before eviction");
                    }
                }
                // The freed parent is the pool's only empty one, so each
                // bucket may take its share of it; the rest stay unresolved.
                const std::size_t before = unresolved.size();
                std::erase_if(unresolved, [&](std::size_t index) {
                    batch.blocks[index] = acquire_in_bucket(target_group, buckets[index]);
                    return static_cast<bool>(batch.blocks[index]);
                });
                _assert(unresolved.size() < before, "evicting a Host parent did not free a placement");
                break;
            }
        }
    }
    batch.stats.allocated = static_cast<std::size_t>(
        std::ranges::count_if(batch.blocks, [](const CacheBlockRef& block) { return static_cast<bool>(block); }));
    batch.stats.unallocated = batch.stats.requested - batch.stats.allocated;
    return batch;
}

}  // namespace tokenspeed::test
