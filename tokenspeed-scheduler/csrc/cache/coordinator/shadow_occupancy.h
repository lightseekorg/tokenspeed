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

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "cache/cache_group.h"
#include "cache/core/block_pool.h"
#include "cache/core/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

// The capacity one pool would have once some of its occupants left, kept
// without touching the pool: per group the holes per bucket in bound parents,
// the empty parents, and the occupants still left in each parent an occupant
// left. A planner removes and restores tentative victims and asks whether a
// demand fits, so it can try a victim set before evicting anything. `groups`
// gives each group's cyclic owner count; the pool supplies its own packing.
class ShadowOccupancy {
public:
    ShadowOccupancy(const BlockPool& pool, std::span<const CacheGroup> groups)
        : pool_{pool}, groups_{groups}, holes_by_bucket_(groups.size()), empty_parent_count_{pool.NumEmptyLcmBlocks()} {
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::span<const std::int64_t> holes = pool_.FreeSlotsByBucket(static_cast<std::uint32_t>(i));
            holes_by_bucket_[i].assign(holes.begin(), holes.end());
            if (holes_by_bucket_[i].empty()) {
                // Never allocated: no bound parent.
                holes_by_bucket_[i].assign(static_cast<std::size_t>(groups_[i].Allocator().ShardCount()), 0);
            }
        }
    }

    // Shadow holes follow the pool's own bookkeeping: a parent whose last
    // occupant leaves takes its holes with it into the empty FIFO, and every
    // bucket owned packing / shard_count of them.
    void Remove(std::uint32_t group_id, CacheBlockLocation location) {
        _assert(pool_.BoundGroup(location.lcm_block_id) == group_id, "a released location belongs to another group");
        auto it =
            remaining_occupied_.try_emplace(location.lcm_block_id, pool_.OccupiedCount(location.lcm_block_id)).first;
        std::int32_t& occupied = it->second;
        _assert(occupied > 0, "the shadow released the same location twice");
        const GroupAllocator& allocator = groups_[group_id].Allocator();
        std::vector<std::int64_t>& holes = holes_by_bucket_[group_id];
        const auto bucket = static_cast<std::size_t>(allocator.BucketOf(location));
        if (occupied == 1) {
            const std::int64_t per_bucket = pool_.Packing(group_id) / allocator.ShardCount();
            for (std::int64_t& bucket_holes : holes) {
                bucket_holes -= per_bucket;
            }
            ++holes[bucket];
            occupied = 0;
            ++empty_parent_count_;
        } else {
            --occupied;
            ++holes[bucket];
        }
    }

    void Restore(std::uint32_t group_id, CacheBlockLocation location) {
        auto it = remaining_occupied_.find(location.lcm_block_id);
        _assert(it != remaining_occupied_.end(), "a restored location has no shadow occupancy");
        std::int32_t& occupied = it->second;
        const GroupAllocator& allocator = groups_[group_id].Allocator();
        const std::int32_t slots = pool_.Packing(group_id);
        std::vector<std::int64_t>& holes = holes_by_bucket_[group_id];
        const auto bucket = static_cast<std::size_t>(allocator.BucketOf(location));
        if (occupied == 0) {
            _assert(empty_parent_count_ > 0, "restoring a location underflowed the empty parents");
            --empty_parent_count_;
            occupied = 1;
            const std::int64_t per_bucket = slots / allocator.ShardCount();
            for (std::int64_t& bucket_holes : holes) {
                bucket_holes += per_bucket;
            }
            --holes[bucket];
        } else {
            _assert(occupied < slots, "restoring a location overflowed its parent");
            ++occupied;
            --holes[bucket];
        }
    }

    // One fit formula for every demand (ParentsNeededForBuckets): per group the
    // bucket-constrained blocks need_by_bucket[i] share the parents they open,
    // the dense_need[i] blocks take any hole, and a demand with no bucket
    // constraint reduces to holes plus parents times packing.
    bool Fits(std::span<const std::vector<std::int64_t>> need_by_bucket,
              std::span<const std::int64_t> dense_need) const {
        _assert(need_by_bucket.size() == groups_.size() && dense_need.size() == groups_.size(),
                "a shadow fit takes one demand per group");
        std::int64_t parents_needed = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            parents_needed += ParentsNeededForBuckets(need_by_bucket[i], holes_by_bucket_[i], dense_need[i],
                                                      pool_.Packing(static_cast<std::uint32_t>(i)));
        }
        return parents_needed <= empty_parent_count_;
    }

private:
    const BlockPool& pool_;
    std::span<const CacheGroup> groups_;
    // Occupants left per parent a Remove touched; untouched parents keep the
    // pool's own count.
    std::unordered_map<std::int32_t, std::int32_t> remaining_occupied_;
    std::vector<std::vector<std::int64_t>> holes_by_bucket_;
    std::int64_t empty_parent_count_{0};
};

}  // namespace tokenspeed
