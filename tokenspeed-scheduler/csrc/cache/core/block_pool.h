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
#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <memory_resource>
#include <new>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "cache/core/block_placement_demand.h"
#include "cache/core/cache_block_ref.h"
#include "cache/core/fixed_slot_resource.h"
#include "cache/core/prefetch.h"
#include "utils.h"

namespace tokenspeed {

// Empty parents a group must open to place blocks_by_bucket[b] blocks in each
// bucket b plus any_bucket_blocks blocks in any bucket, given holes_by_bucket[b] free
// slots already in its bound parents. An opened parent gives every bucket
// packing / B slots at once, so the bucket demands share the opened parents
// rather than summing; the blocks without a fixed bucket take whatever holes the bucket
// demands leave. The pool's exact acquire, the coordinator's snapshot probe
// and the admission planner's shadow fit all answer with this one formula.
inline std::int64_t ParentsNeededForBuckets(std::span<const std::int64_t> blocks_by_bucket,
                                            std::span<const std::int64_t> holes_by_bucket,
                                            std::int64_t any_bucket_blocks, std::int64_t packing) {
    _assert(!blocks_by_bucket.empty() && blocks_by_bucket.size() == holes_by_bucket.size(),
            "bucket demand and holes must cover the same shard count");
    const auto buckets = static_cast<std::int64_t>(blocks_by_bucket.size());
    _assert(packing > 0 && (buckets == 1 || packing % buckets == 0) && any_bucket_blocks >= 0,
            "bucket count must divide the parent packing and dense demand must be non-negative");
    // Every eviction candidate asks again; one slot per parent or bucket, the
    // common geometry, skips the integer divisions.
    const auto parents_for = [](std::int64_t blocks, std::int64_t slots_per_parent) {
        return slots_per_parent == 1 ? blocks : (blocks + slots_per_parent - 1) / slots_per_parent;
    };
    const std::int64_t per_parent = buckets == 1 ? packing : packing / buckets;
    std::int64_t parents = 0;
    std::int64_t total_need = any_bucket_blocks;
    std::int64_t total_holes = 0;
    for (std::size_t b = 0; b < blocks_by_bucket.size(); ++b) {
        _assert(blocks_by_bucket[b] >= 0 && holes_by_bucket[b] >= 0, "bucket demand and holes must be non-negative");
        const std::int64_t short_by = std::max<std::int64_t>(blocks_by_bucket[b] - holes_by_bucket[b], 0);
        parents = std::max(parents, parents_for(short_by, per_parent));
        total_need += blocks_by_bucket[b];
        total_holes += holes_by_bucket[b];
    }
    const std::int64_t total_short = std::max<std::int64_t>(total_need - total_holes, 0);
    return std::max(parents, parents_for(total_short, packing));
}

// Where pairs of blocks go when a pair's two blocks must share a bucket but
// which bucket is free to choose (a tail block an admission allocates beside
// its copy-in block): one pair at a time, into the bucket with the most holes
// left over after blocks_by_bucket -- the blocks wanted per bucket so far, the pairs
// before it included -- the lowest such bucket on ties. Under
// ParentsNeededForBuckets no other choice needs fewer parents: the pairs add
// the same blocks to the total whatever buckets they take, and filling the
// least short bucket first keeps the largest bucket shortfall as small as it
// can be.
inline std::size_t NextPairBucket(std::span<const std::int64_t> blocks_by_bucket,
                                  std::span<const std::int64_t> holes_by_bucket) {
    _assert(!blocks_by_bucket.empty() && blocks_by_bucket.size() == holes_by_bucket.size(),
            "bucket demand and holes must cover the same shard count");
    std::size_t best = 0;
    for (std::size_t b = 1; b < blocks_by_bucket.size(); ++b) {
        if (holes_by_bucket[b] - blocks_by_bucket[b] > holes_by_bucket[best] - blocks_by_bucket[best]) {
            best = b;
        }
    }
    return best;
}

// The bucket each pair in block_demand takes (NextPairBucket).
inline std::vector<std::int32_t> AssignPairBuckets(const BlockPlacementDemand& block_demand,
                                                   std::span<const std::int64_t> holes_by_bucket) {
    _assert(block_demand.same_bucket_pairs >= 0, "pair demand must be non-negative");
    std::vector<std::int64_t> blocks_with_pairs = block_demand.blocks_by_bucket;
    std::vector<std::int32_t> buckets;
    buckets.reserve(static_cast<std::size_t>(block_demand.same_bucket_pairs));
    for (std::int64_t pair = 0; pair < block_demand.same_bucket_pairs; ++pair) {
        const std::size_t bucket = NextPairBucket(blocks_with_pairs, holes_by_bucket);
        blocks_with_pairs[bucket] += 2;
        buckets.push_back(static_cast<std::int32_t>(bucket));
    }
    return buckets;
}

// ParentsNeededForBuckets once the demand's pairs join its fixed-bucket blocks
// in the buckets NextPairBucket gives them. Scratch holds one count per bucket,
// so checking each tentative eviction allocates nothing.
inline std::int64_t ParentsNeededForBucketsAndPairs(const BlockPlacementDemand& block_demand,
                                                    std::span<const std::int64_t> holes_by_bucket, std::int64_t packing,
                                                    std::span<std::int64_t> scratch) {
    if (block_demand.same_bucket_pairs == 0) {
        return ParentsNeededForBuckets(block_demand.blocks_by_bucket, holes_by_bucket, block_demand.any_bucket_blocks,
                                       packing);
    }
    _assert(block_demand.same_bucket_pairs > 0 && scratch.size() == block_demand.blocks_by_bucket.size(),
            "pairs need one scratch entry per bucket");
    std::ranges::copy(block_demand.blocks_by_bucket, scratch.begin());
    for (std::int64_t pair = 0; pair < block_demand.same_bucket_pairs; ++pair) {
        scratch[NextPairBucket(scratch, holes_by_bucket)] += 2;
    }
    return ParentsNeededForBuckets(scratch, holes_by_bucket, block_demand.any_bucket_blocks, packing);
}

// Request-aware placement preference for a single BlockPool acquire call.
// When supplied, the pool prefers free slots in `preferred_parent`, then opens
// an empty parent ahead of existing holes while strictly more than
// `reserved_empty_parents` empty parents remain (preserving whole parents for
// later groups in the same admission), and falls back to existing holes once
// the empty-parent count reaches that reserve.
struct ParentAffinity {
    std::optional<std::int32_t> preferred_parent;
    std::int64_t reserved_empty_parents{0};
};

// Physical LCM placement only. It deliberately has no cache key, LRU node,
// CacheBlock pointer, or ownership count.
class BlockPool {
public:
    explicit BlockPool(std::int32_t num_lcm_blocks, std::vector<std::int32_t> slots_per_group)
        : lcm_blocks_(checkedLcmBlockCount(num_lcm_blocks)), group_availability_(slots_per_group.size()) {
        for (std::size_t group_id = 0; group_id < slots_per_group.size(); ++group_id) {
            const auto packing = slots_per_group[group_id];
            _assert(packing > 0, "slots_per_parent must be > 0");
            _assert(static_cast<std::int64_t>(num_lcm_blocks) * packing <= std::numeric_limits<std::int32_t>::max(),
                    "group cache page count exceeds int32 range");
            group_availability_[group_id].packing = packing;
            if (packing > kInlineOccupancySlots) {
                // Rounded up without overflowing near the int32 maximum.
                const auto words = packing / kInlineOccupancySlots + (packing % kInlineOccupancySlots != 0 ? 1 : 0);
                wide_occupancy_words_ = std::max(wide_occupancy_words_, static_cast<std::size_t>(words));
            }
        }
        // Ids 1..num_lcm_blocks, counted so that INT32_MAX ends the loop.
        for (std::int32_t i = 0; i < num_lcm_blocks; ++i) {
            free_parent_ids_.push_back(i + 1);
        }
    }

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;
    ~BlockPool() noexcept { FatalCheck(NumOccupiedSlots() == 0, "BlockPool destroyed with live block references"); }

    // Number of physical LCM blocks. Kernel page 0 is reserved separately.
    std::int32_t NumLcmBlocks() const noexcept { return static_cast<std::int32_t>(lcm_blocks_.size()); }
    std::int32_t NumEmptyLcmBlocks() const noexcept { return static_cast<std::int32_t>(free_parent_ids_.size()); }

    // Unoccupied child slots in parents already bound to this group.
    std::int32_t NumFreeSlots(std::uint32_t group_id) const noexcept {
        return static_cast<std::int32_t>(placement(group_id).FreeSlots());
    }
    // The same holes, per bucket (slot % shard_count). Buckets exist once
    // the group is registered; an unregistered group has none.
    std::span<const std::int64_t> FreeSlotsByBucket(std::uint32_t group_id) const noexcept {
        return placement(group_id).free_slots_by_bucket;
    }

    std::int32_t Packing(std::uint32_t group_id) const noexcept { return placement(group_id).packing; }

    // Minimum occupied slots in a bound parent, or the packing when this
    // group has no bound parent. Full parents need not appear in the hole
    // indexes; every partial parent appears in at least one bucket's order.
    std::int32_t MinimumParentOccupancy(std::uint32_t group_id) const noexcept {
        const GroupAvailability& group = placement(group_id);
        std::int32_t minimum = group.packing;
        for (const OccupancyBuckets& parents : group.parents_by_bucket) {
            const std::int32_t occ = parents.MinOccupied();
            if (occ > 0) {
                minimum = std::min(minimum, occ);
            }
        }
        return minimum;
    }

    std::int32_t ResolveCacheBlockId(std::uint32_t group_id, CacheBlockLocation location) const {
        const std::int32_t packing = Packing(group_id);
        _assert(location.lcm_block_id > 0 && location.lcm_block_id <= NumLcmBlocks(),
                "LCM block id is outside the owning pool");
        _assert(location.slot_index >= 0 && location.slot_index < packing, "cache block slot is out of range");
        const std::int64_t id =
            1 + (static_cast<std::int64_t>(location.lcm_block_id) - 1) * packing + location.slot_index;
        _assert(id <= std::numeric_limits<std::int32_t>::max(), "cache block id exceeds int32 range");
        return static_cast<std::int32_t>(id);
    }

    // Fix placement geometry before this group allocates any blocks. Repeated
    // registration is allowed only with the same geometry, including after free.
    void RegisterGroup(std::uint32_t group_id, std::int32_t packing, std::int32_t shard_count) {
        _assert(packing > 0 && shard_count > 0 && packing % shard_count == 0,
                "shard_count must be positive and divide parent packing");
        GroupAvailability& group = placement(group_id);
        _assert(group.packing == packing, "cache group packing changed after construction");
        const auto buckets = static_cast<std::size_t>(shard_count);
        if (group.parents_by_bucket.empty()) {
            if (packing > 1) {
                ensureLinkStride(buckets);
            }
            group.parents_by_bucket.resize(buckets);
            for (OccupancyBuckets& bucket_index : group.parents_by_bucket) {
                bucket_index.Init(packing);
            }
            group.free_slots_by_bucket.assign(buckets, 0);
        } else {
            _assert(group.packing == packing && group.parents_by_bucket.size() == buckets,
                    "cache group geometry changed after registration");
        }
    }

    CacheBlockRef AcquireBlock(std::uint32_t group_id) {
        std::vector<CacheBlockRef> blocks = AcquireBlocks(group_id, 1);
        if (blocks.empty()) {
            return {};
        }
        return std::move(blocks.front());
    }

    std::vector<CacheBlockRef> AcquireBlocks(std::uint32_t group_id, std::int32_t num) {
        return AcquireBlocks(group_id, num, {});
    }

    std::vector<CacheBlockRef> AcquireBlocks(std::uint32_t group_id, std::int32_t num,
                                             std::span<const std::int32_t> bucket_loads) {
        return acquireBlocksChecked(group_id, num, bucket_loads, std::nullopt);
    }

    // Request-aware block acquisition for packed groups (packing > 1): tries
    // free slots in `affinity.preferred_parent` (honoring `min_load` bucket
    // balance when `bucket_loads` has multiple entries), then opens an empty
    // FIFO parent and switches `preferred_parent` to it while strictly more
    // than `affinity.reserved_empty_parents` empty parents remain, and falls
    // back to the densest partial parent once empty parents reach the reserve.
    // All or nothing on insufficient capacity. `packing == 1` bypasses affinity.
    std::vector<CacheBlockRef> AcquireBlocksWithAffinity(std::uint32_t group_id, std::int32_t num,
                                                         std::span<const std::int32_t> bucket_loads,
                                                         ParentAffinity affinity) {
        _assert(affinity.reserved_empty_parents >= 0, "reserved empty parents must be non-negative");
        if (placement(group_id).packing == 1) {
            return acquireBlocksChecked(group_id, num, bucket_loads, std::nullopt);
        }
        return acquireBlocksChecked(group_id, num, bucket_loads, affinity);
    }

    // One block per entry of `buckets`, each in the named bucket (slot %
    // shard_count), returned in the same order. Rebuilds a recorded table
    // shape (a retraction snapshot or its restore) whose blocks must keep
    // their owner under page-cyclic sharding. All or nothing: the exact
    // capacity check (ParentsNeededForBuckets) precedes every mutation, so a
    // failed call leaves the indices, occupancy and FIFO untouched. Each block
    // takes the head of its bucket's partial-parent index -- the most recently
    // touched parent at the highest occupancy level, at its lowest free slot in
    // that bucket, as ordinary placement does -- or, when that bucket has no
    // hole, the FIFO's next empty parent at the bucket's own slot; the opened
    // parent's other slots then serve the remaining buckets. Deterministic on
    // every mirrored rank. With one bucket this is ordinary placement with the
    // shard count as a parameter.
    std::vector<CacheBlockRef> AcquireBlocksInBuckets(std::uint32_t group_id, std::span<const std::int32_t> buckets) {
        return acquireBlocksInBuckets(group_id, buckets, std::nullopt);
    }

    // Bucket-exact acquisition with request-aware `affinity`: for each bucket
    // entry, prefers `affinity.preferred_parent`, else opens an empty FIFO
    // parent (updating `preferred_parent` for the rest of the call) while
    // strictly more than `affinity.reserved_empty_parents` empty parents
    // remain, else falls back to the bucket's densest partial parent. All or
    // nothing on insufficient capacity; `packing == 1` bypasses affinity.
    std::vector<CacheBlockRef> AcquireBlocksInBucketsWithAffinity(std::uint32_t group_id,
                                                                  std::span<const std::int32_t> buckets,
                                                                  ParentAffinity affinity) {
        _assert(affinity.reserved_empty_parents >= 0, "reserved empty parents must be non-negative");
        if (placement(group_id).packing == 1) {
            return acquireBlocksInBuckets(group_id, buckets, std::nullopt);
        }
        return acquireBlocksInBuckets(group_id, buckets, affinity);
    }

    // One block at a given free child slot of a parent already bound to this
    // group. The one placement that names its slot: a lone request's
    // compaction (LoneRequestAdmission::Compact) moves a block into a
    // hole of one of the request's own denser parents, where the hole-first
    // order of nextLocation (most occupied parent first, then the FIFO) could
    // pick another parent or the block's own.
    CacheBlockRef AcquireBlockAt(std::uint32_t group_id, CacheBlockLocation location) {
        const LcmBlock& parent = lcmBlock(location.lcm_block_id);
        _assert(parent.bound_group == group_id && location.slot_index >= 0 && location.slot_index < parent.slots &&
                    !parent.SlotOccupied(static_cast<std::size_t>(location.slot_index)),
                "a placed block takes a free slot of a parent bound to its group");
        return createBlockRef(group_id, placement(group_id).packing, location);
    }

    std::vector<CacheBlockRef> AcquireUpToBlocks(std::uint32_t group_id, std::int32_t max_num) {
        const auto cache_blocks_per_lcm_block = placement(group_id).packing;
        if (max_num <= 0) {
            return {};
        }
        const auto take = static_cast<std::int32_t>(
            std::min(static_cast<std::size_t>(max_num), availableBlocks(group_id, cache_blocks_per_lcm_block)));
        return take > 0 ? acquireAvailableBlocks(group_id, cache_blocks_per_lcm_block, take, {}, std::nullopt)
                        : std::vector<CacheBlockRef>{};
    }

    std::optional<std::uint32_t> BoundGroup(std::int32_t lcm_block_id) const {
        return lcmBlock(lcm_block_id).bound_group;
    }
    std::int32_t OccupiedCount(std::int32_t lcm_block_id) const {
        return static_cast<std::int32_t>(lcmBlock(lcm_block_id).occupied_count);
    }
    bool IsOccupied(CacheBlockLocation location) const {
        const LcmBlock& lcm_block = lcmBlock(location.lcm_block_id);
        return location.slot_index >= 0 && location.slot_index < lcm_block.slots &&
               lcm_block.SlotOccupied(static_cast<std::size_t>(location.slot_index));
    }
    std::int32_t NumOccupiedSlots() const noexcept {
        std::int32_t count = 0;
        for (const LcmBlock& block : lcm_blocks_) {
            count += static_cast<std::int32_t>(block.occupied_count);
        }
        return count;
    }
    std::vector<CacheBlockLocation> OccupiedLocations(std::int32_t lcm_block_id) const {
        const LcmBlock& lcm_block = lcmBlock(lcm_block_id);
        std::vector<CacheBlockLocation> locations;
        locations.reserve(lcm_block.occupied_count);
        for (std::int32_t slot = 0; slot < lcm_block.slots; ++slot) {
            if (lcm_block.SlotOccupied(static_cast<std::size_t>(slot))) {
                locations.push_back(CacheBlockLocation{.lcm_block_id = lcm_block_id, .slot_index = slot});
            }
        }
        return locations;
    }

    // A hint for a later Release(location): prefetches its LCM parent's
    // record (and its intrusive link slot when the pool carries packed groups),
    // found by its ID alone. Parent records live as long as the pool.
    void PrefetchRelease(CacheBlockLocation location) const noexcept {
        if (location.lcm_block_id <= 0 || static_cast<std::size_t>(location.lcm_block_id) > lcm_blocks_.size()) return;
        const std::size_t idx = static_cast<std::size_t>(location.lcm_block_id - 1);
        PrefetchObject(lcm_blocks_[idx]);
        if (link_stride_ > 0) {
            PrefetchObject(bucket_links_[idx * link_stride_]);
        }
    }

    void Release(CacheBlockLocation location) noexcept {
        FatalCheck(location.lcm_block_id > 0 && static_cast<std::size_t>(location.lcm_block_id) <= lcm_blocks_.size(),
                   "CacheBlock location has invalid LCM block id");
        LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        FatalCheck(location.slot_index >= 0 && location.slot_index < parent.slots,
                   "CacheBlock location has invalid slot");
        const std::size_t slot = static_cast<std::size_t>(location.slot_index);
        FatalCheck(parent.SlotOccupied(slot) && parent.occupied_count > 0, "CacheBlock location is not occupied");
        FatalCheck(parent.bound_group.has_value(), "occupied LCM parent has no bound group");
        GroupAvailability& group = placement(*parent.bound_group);
        removeAvailability(group, parent, location.lcm_block_id);
        parent.SetSlotOccupied(slot, false);
        --parent.occupied_count;
        const std::size_t buckets = group.parents_by_bucket.size();
        ++group.free_slots_by_bucket[slot % buckets];
        if (parent.occupied_count == 0) {
            // An unbound parent's slots leave the group: they belong to the
            // pool-wide empty FIFO again.
            const auto per_bucket = static_cast<std::int64_t>(static_cast<std::size_t>(parent.slots) / buckets);
            for (std::int64_t& holes : group.free_slots_by_bucket) {
                FatalCheck(holes >= per_bucket, "group free-slot count underflow on unbind");
                holes -= per_bucket;
            }
            parent.bound_group.reset();
            parent.slots = 0;  // its occupancy bits are all clear again
            parent.first_free_slots.clear();
            FatalCheck(free_parent_ids_.size() < lcm_blocks_.size(), "LCM free queue overflow");
            free_parent_ids_.push_back(location.lcm_block_id);
        } else {
            auto& first = parent.first_free_slots[slot % buckets];
            first = std::min(first, location.slot_index);
            addAvailability(group, parent, location.lcm_block_id);
        }
    }

private:
    static constexpr std::int32_t kInlineOccupancySlots = 64;
    static constexpr std::size_t kInlineLoadBuckets = 8;

    // Intrusive doubly-linked list node stored in the pool-level flat table
    // `bucket_links_` indexed by `(lcm_block_id - 1) * link_stride_ + bucket`,
    // using 1-based parent IDs (0 = null) so all index updates are O(1) with
    // zero per-parent heap allocation and zero memory on packing == 1 pools.
    struct BucketLink {
        std::int32_t prev{0};
        std::int32_t next{0};
    };

    // Per-bucket partial-parent index bucketed by occupied_count in [1, packing - 1].
    // Bit (occ - 1) in the occupancy mask is set iff heads[occ - 1] != 0, so
    // finding the densest or sparsest partial parent is O(1) via bit scan.
    struct OccupancyBuckets {
        static constexpr std::size_t kWordBits = 64;
        static_assert(kInlineOccupancySlots == static_cast<std::int32_t>(kWordBits));

        std::vector<std::int32_t> heads;
        std::uint64_t inline_mask{0};
        std::vector<std::uint64_t> wide_mask;

        void Init(std::int32_t packing) {
            if (packing <= 1) {
                return;
            }
            const auto num_occ = static_cast<std::size_t>(packing - 1);
            heads.assign(num_occ, 0);
            inline_mask = 0;
            if (num_occ > kWordBits) {
                const std::size_t words = num_occ / kWordBits + (num_occ % kWordBits != 0 ? 1 : 0);
                wide_mask.assign(words, 0);
            } else {
                wide_mask.clear();
            }
        }

        std::int32_t Head(std::int32_t occupied_count) const noexcept {
            _assert(occupied_count >= 1 && static_cast<std::size_t>(occupied_count) <= heads.size(),
                    "occupancy bucket out of range");
            return heads[static_cast<std::size_t>(occupied_count - 1)];
        }

        void SetNonEmpty(std::size_t idx) noexcept {
            if (wide_mask.empty()) {
                inline_mask |= (std::uint64_t{1} << idx);
            } else {
                wide_mask[idx / kWordBits] |= (std::uint64_t{1} << (idx % kWordBits));
            }
        }

        void ClearNonEmpty(std::size_t idx) noexcept {
            if (wide_mask.empty()) {
                inline_mask &= ~(std::uint64_t{1} << idx);
            } else {
                wide_mask[idx / kWordBits] &= ~(std::uint64_t{1} << (idx % kWordBits));
            }
        }

        std::int32_t MaxOccupied() const noexcept {
            if (wide_mask.empty()) {
                if (inline_mask == 0) {
                    return 0;
                }
                return static_cast<std::int32_t>(kWordBits - static_cast<std::size_t>(std::countl_zero(inline_mask)));
            }
            for (std::size_t word = wide_mask.size(); word-- > 0;) {
                if (wide_mask[word] != 0) {
                    return static_cast<std::int32_t>(
                        word * kWordBits + (kWordBits - static_cast<std::size_t>(std::countl_zero(wide_mask[word]))));
                }
            }
            return 0;
        }

        std::int32_t MinOccupied() const noexcept {
            if (wide_mask.empty()) {
                if (inline_mask == 0) {
                    return 0;
                }
                return static_cast<std::int32_t>(std::countr_zero(inline_mask) + 1);
            }
            for (std::size_t word = 0; word < wide_mask.size(); ++word) {
                if (wide_mask[word] != 0) {
                    return static_cast<std::int32_t>(word * kWordBits +
                                                     static_cast<std::size_t>(std::countr_zero(wide_mask[word])) + 1);
                }
            }
            return 0;
        }
    };

    struct GroupAvailability {
        std::int32_t packing{0};
        // Holes in bound parents, per bucket (slot % shard_count); empty
        // parents remain pool-wide. The group total is derived from this, so
        // the two cannot disagree.
        std::vector<std::int64_t> free_slots_by_bucket;
        std::vector<OccupancyBuckets> parents_by_bucket;

        std::size_t FreeSlots() const noexcept {
            std::int64_t total = 0;
            for (const std::int64_t holes : free_slots_by_bucket) {
                total += holes;
            }
            return static_cast<std::size_t>(total);
        }
    };
    struct LcmBlock {
        std::optional<std::uint32_t> bound_group;
        std::uint32_t occupied_count{0};
        // The bound group's packing; 0 while the parent is empty.
        std::int32_t slots{0};
        // Bit s is set while child slot s is occupied, so an empty parent's
        // bits are all clear. Up to 64 slots they live in the parent itself:
        // the occupancy bits need no other memory. A wider parent uses
        // wide_occupancy instead, allocated at its first wide binding with a
        // word per 64 slots of the pool's widest packing and kept until the
        // pool is destroyed.
        std::uint64_t occupancy{0};
        std::unique_ptr<std::uint64_t[]> wide_occupancy;
        // Only the lowest free slot per bucket is needed, not a list of all holes.
        // The packing is the sentinel for a bucket with no free slot.
        std::vector<std::int32_t> first_free_slots;

        bool SlotOccupied(std::size_t slot) const noexcept {
            const std::uint64_t word =
                slots <= kInlineOccupancySlots ? occupancy : wide_occupancy[slot / kInlineOccupancySlots];
            return ((word >> (slot % kInlineOccupancySlots)) & 1U) != 0;
        }
        void SetSlotOccupied(std::size_t slot, bool occupied) noexcept {
            std::uint64_t& word =
                slots <= kInlineOccupancySlots ? occupancy : wide_occupancy[slot / kInlineOccupancySlots];
            const std::uint64_t bit = std::uint64_t{1} << (slot % kInlineOccupancySlots);
            word = occupied ? (word | bit) : (word & ~bit);
        }
    };

    BucketLink& bucketLink(std::int32_t id, std::size_t bucket) noexcept {
        _assert(id > 0 && static_cast<std::size_t>(id) <= lcm_blocks_.size(), "LCM block id out of range");
        _assert(bucket < link_stride_, "bucket index exceeds link stride");
        return bucket_links_[static_cast<std::size_t>(id - 1) * link_stride_ + bucket];
    }

    // Lazily allocates or widens the flat per-parent link table when a packed
    // group (packing > 1) registers with more buckets than `link_stride_`.
    // Preserves links already live in buckets [0, link_stride_).
    void ensureLinkStride(std::size_t buckets) {
        if (buckets <= link_stride_) {
            return;
        }
        const std::size_t num_parents = lcm_blocks_.size();
        if (num_parents > 0) {
            FatalCheck(buckets <= std::numeric_limits<std::size_t>::max() / num_parents,
                       "intrusive link table size exceeds address space");
            std::vector<BucketLink> grown(num_parents * buckets);
            if (link_stride_ > 0) {
                for (std::size_t p = 0; p < num_parents; ++p) {
                    std::copy_n(bucket_links_.data() + p * link_stride_, link_stride_, grown.data() + p * buckets);
                }
            }
            bucket_links_ = std::move(grown);
        }
        link_stride_ = buckets;
    }

    const GroupAvailability& placement(std::uint32_t group_id) const noexcept {
        FatalCheck(group_id < group_availability_.size(), "group id has no placement in this pool");
        return group_availability_[group_id];
    }
    GroupAvailability& placement(std::uint32_t group_id) noexcept {
        FatalCheck(group_id < group_availability_.size(), "group id has no placement in this pool");
        return group_availability_[group_id];
    }

    std::size_t availableBlocks(std::uint32_t group_id, std::int32_t packing) const {
        const auto& group = placement(group_id);
        _assert(group.packing == packing, "cache group packing changed after construction");
        return group.FreeSlots() + free_parent_ids_.size() * static_cast<std::size_t>(packing);
    }

    void removeAvailability(GroupAvailability& group, const LcmBlock& parent, std::int32_t id) noexcept {
        if (parent.occupied_count == static_cast<std::uint32_t>(parent.slots)) {
            return;
        }
        const std::size_t buckets = group.parents_by_bucket.size();
        const std::size_t idx = static_cast<std::size_t>(parent.occupied_count - 1);
        for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
            if (parent.first_free_slots[bucket] < group.packing) {
                OccupancyBuckets& index = group.parents_by_bucket[bucket];
                BucketLink& link = bucketLink(id, bucket);
                const std::int32_t prev = link.prev;
                const std::int32_t next = link.next;
                if (prev != 0) {
                    bucketLink(prev, bucket).next = next;
                } else {
                    FatalCheck(index.heads[idx] == id, "partial LCM parent missing from its bucket index");
                    index.heads[idx] = next;
                    if (next == 0) {
                        index.ClearNonEmpty(idx);
                    }
                }
                if (next != 0) {
                    bucketLink(next, bucket).prev = prev;
                }
                link = {};
            }
        }
    }

    void addAvailability(GroupAvailability& group, const LcmBlock& parent, std::int32_t id) noexcept {
        if (parent.occupied_count == static_cast<std::uint32_t>(parent.slots)) {
            return;
        }
        const std::size_t buckets = group.parents_by_bucket.size();
        const std::size_t idx = static_cast<std::size_t>(parent.occupied_count - 1);
        for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
            if (parent.first_free_slots[bucket] < group.packing) {
                OccupancyBuckets& index = group.parents_by_bucket[bucket];
                BucketLink& link = bucketLink(id, bucket);
                const std::int32_t old_head = index.heads[idx];
                FatalCheck(link.prev == 0 && link.next == 0 && old_head != id,
                           "partial LCM parent already present in its bucket index");
                link.prev = 0;
                link.next = old_head;
                if (old_head != 0) {
                    bucketLink(old_head, bucket).prev = id;
                } else {
                    index.SetNonEmpty(idx);
                }
                index.heads[idx] = id;
            }
        }
    }

    GroupAvailability& prepareAvailability(std::uint32_t group_id, std::int32_t packing) {
        (void)availableBlocks(group_id, packing);
        if (placement(group_id).parents_by_bucket.empty()) {
            RegisterGroup(group_id, packing, 1);
        }
        return placement(group_id);
    }

    CacheBlockLocation nextLocation(const GroupAvailability& group, std::span<const std::int64_t> loads) const {
        // Ordinary allocation orders by parent occupancy (most occupied first),
        // bucket list head, then slot. Balanced allocation puts request load
        // and bucket ID before that order.
        using Priority = std::tuple<std::int64_t, std::size_t, std::int32_t, std::int32_t, std::int32_t>;
        std::optional<Priority> best;
        CacheBlockLocation location;
        for (std::size_t bucket = 0; bucket < group.parents_by_bucket.size(); ++bucket) {
            const std::int32_t occ = group.parents_by_bucket[bucket].MaxOccupied();
            if (occ == 0) {
                continue;
            }
            const std::int32_t id = group.parents_by_bucket[bucket].Head(occ);
            const auto slot = lcmBlock(id).first_free_slots[bucket];
            const Priority priority{loads.empty() ? 0 : loads[bucket], loads.empty() ? 0 : bucket, -occ, id, slot};
            if (!best || priority < *best) {
                best = priority;
                location = CacheBlockLocation{id, slot};
            }
        }
        if (best) {
            return location;
        }
        // Every hole in bound parents is exhausted before opening a FIFO parent.
        FatalCheck(!free_parent_ids_.empty(), "prechecked block capacity was exhausted");
        const auto slot =
            loads.empty() ? 0 : static_cast<std::int32_t>(std::ranges::min_element(loads) - loads.begin());
        return CacheBlockLocation{free_parent_ids_.front(), slot};
    }

    // Placement confined to one bucket: the bucket's own partial-parent
    // index head, else the FIFO's next empty parent at the bucket's slot.
    CacheBlockLocation nextLocationInBucket(const GroupAvailability& group, std::size_t bucket) const {
        const OccupancyBuckets& parents = group.parents_by_bucket[bucket];
        const std::int32_t occ = parents.MaxOccupied();
        if (occ > 0) {
            const std::int32_t id = parents.Head(occ);
            return CacheBlockLocation{id, lcmBlock(id).first_free_slots[bucket]};
        }
        FatalCheck(!free_parent_ids_.empty(), "prechecked bucket capacity was exhausted");
        return CacheBlockLocation{free_parent_ids_.front(), static_cast<std::int32_t>(bucket)};
    }

    std::vector<CacheBlockRef> acquireBlocksChecked(std::uint32_t group_id, std::int32_t num,
                                                    std::span<const std::int32_t> bucket_loads,
                                                    std::optional<ParentAffinity> affinity) {
        const auto cache_blocks_per_lcm_block = placement(group_id).packing;
        if (num <= 0) {
            return {};
        }

        if (bucket_loads.size() > 1) {
            _assert(placement(group_id).parents_by_bucket.size() == bucket_loads.size(),
                    "allocation loads must match the registered shard_count");
            for (std::int32_t load : bucket_loads) {
                _assert(load >= 0, "allocation bucket loads must be non-negative");
            }
        }
        if (availableBlocks(group_id, cache_blocks_per_lcm_block) < static_cast<std::size_t>(num)) {
            return {};
        }
        return acquireAvailableBlocks(group_id, cache_blocks_per_lcm_block, num, bucket_loads, affinity);
    }

    std::vector<CacheBlockRef> acquireBlocksInBuckets(std::uint32_t group_id, std::span<const std::int32_t> buckets,
                                                      std::optional<ParentAffinity> affinity) {
        const auto packing = placement(group_id).packing;
        if (buckets.empty()) {
            return {};
        }
        GroupAvailability& group = prepareAvailability(group_id, packing);
        const std::size_t num_buckets = group.parents_by_bucket.size();
        std::array<std::int64_t, kInlineLoadBuckets> inline_counts;
        std::vector<std::int64_t> wide_counts;
        std::span<std::int64_t> blocks_by_bucket;
        if (num_buckets <= kInlineLoadBuckets) {
            std::fill_n(inline_counts.data(), num_buckets, 0);
            blocks_by_bucket = std::span<std::int64_t>{inline_counts.data(), num_buckets};
        } else {
            wide_counts.assign(num_buckets, 0);
            blocks_by_bucket = wide_counts;
        }
        for (const std::int32_t bucket : buckets) {
            _assert(bucket >= 0 && static_cast<std::size_t>(bucket) < blocks_by_bucket.size(),
                    "requested bucket is outside the registered shard count");
            ++blocks_by_bucket[static_cast<std::size_t>(bucket)];
        }
        if (ParentsNeededForBuckets(blocks_by_bucket, group.free_slots_by_bucket, /*any_bucket_blocks=*/0, packing) >
            static_cast<std::int64_t>(free_parent_ids_.size())) {
            return {};
        }
        std::optional<std::int32_t> preferred_parent = affinity ? affinity->preferred_parent : std::nullopt;
        const auto pick_in_bucket = [&](std::size_t bucket) -> CacheBlockLocation {
            if (affinity) {
                if (preferred_parent) {
                    _assert(*preferred_parent > 0 && *preferred_parent <= NumLcmBlocks(),
                            "preferred parent id out of range");
                    const LcmBlock& parent = lcmBlock(*preferred_parent);
                    if (parent.bound_group == group_id && parent.first_free_slots[bucket] < packing) {
                        return CacheBlockLocation{*preferred_parent, parent.first_free_slots[bucket]};
                    }
                }
                if (static_cast<std::int64_t>(free_parent_ids_.size()) > affinity->reserved_empty_parents) {
                    preferred_parent = free_parent_ids_.front();
                    return CacheBlockLocation{*preferred_parent, static_cast<std::int32_t>(bucket)};
                }
            }
            return nextLocationInBucket(group, bucket);
        };
        std::vector<CacheBlockRef> out;
        out.reserve(buckets.size());
        for (const std::int32_t bucket : buckets) {
            out.push_back(createBlockRef(group_id, packing, pick_in_bucket(static_cast<std::size_t>(bucket))));
        }
        return out;
    }

    std::vector<CacheBlockRef> acquireAvailableBlocks(std::uint32_t group_id, std::int32_t packing, std::int32_t count,
                                                      std::span<const std::int32_t> bucket_loads,
                                                      std::optional<ParentAffinity> affinity) {
        GroupAvailability& group = prepareAvailability(group_id, packing);
        // Exact capacity checking precedes all mutations, so a second,
        // pool-sized shadow allocator is unnecessary.
        std::vector<CacheBlockRef> out;
        out.reserve(static_cast<std::size_t>(count));
        if (packing == 1) {
            // Single-slot parents have no partial parents, one bucket, and no
            // affinity; pull directly from the empty-parent FIFO.
            for (std::int32_t i = 0; i < count; ++i) {
                FatalCheck(!free_parent_ids_.empty(), "prechecked block capacity was exhausted");
                out.push_back(createBlockRef(group_id, 1, CacheBlockLocation{free_parent_ids_.front(), 0}));
            }
            return out;
        }
        const bool balanced = bucket_loads.size() > 1;
        std::array<std::int64_t, kInlineLoadBuckets> inline_loads;
        std::vector<std::int64_t> wide_loads;
        std::span<std::int64_t> loads;
        if (balanced) {
            if (bucket_loads.size() <= kInlineLoadBuckets) {
                for (std::size_t b = 0; b < bucket_loads.size(); ++b) {
                    inline_loads[b] = bucket_loads[b];
                }
                loads = std::span<std::int64_t>{inline_loads.data(), bucket_loads.size()};
            } else {
                wide_loads.assign(bucket_loads.begin(), bucket_loads.end());
                loads = wide_loads;
            }
        }
        std::optional<std::int32_t> preferred_parent = affinity ? affinity->preferred_parent : std::nullopt;
        const auto pick_location = [&]() -> CacheBlockLocation {
            if (affinity) {
                const auto min_it = balanced ? std::ranges::min_element(loads) : loads.end();
                if (preferred_parent) {
                    _assert(*preferred_parent > 0 && *preferred_parent <= NumLcmBlocks(),
                            "preferred parent id out of range");
                    const LcmBlock& parent = lcmBlock(*preferred_parent);
                    if (parent.bound_group == group_id) {
                        if (!balanced) {
                            const std::int32_t slot = *std::ranges::min_element(parent.first_free_slots);
                            if (slot < packing) {
                                return CacheBlockLocation{*preferred_parent, slot};
                            }
                        } else {
                            const std::int64_t min_load = *min_it;
                            for (std::size_t bucket = 0; bucket < loads.size(); ++bucket) {
                                if (loads[bucket] == min_load && parent.first_free_slots[bucket] < packing) {
                                    return CacheBlockLocation{*preferred_parent, parent.first_free_slots[bucket]};
                                }
                            }
                        }
                    }
                }
                if (static_cast<std::int64_t>(free_parent_ids_.size()) > affinity->reserved_empty_parents) {
                    preferred_parent = free_parent_ids_.front();
                    const auto slot = balanced ? static_cast<std::int32_t>(min_it - loads.begin()) : 0;
                    return CacheBlockLocation{*preferred_parent, slot};
                }
            }
            return nextLocation(group, loads);
        };
        for (std::int32_t i = 0; i < count; ++i) {
            const CacheBlockLocation location = pick_location();
            out.push_back(createBlockRef(group_id, packing, location));
            if (balanced) {
                ++loads[static_cast<std::size_t>(location.slot_index) % loads.size()];
            }
        }
        return out;
    }

    static std::size_t checkedLcmBlockCount(std::int32_t num_lcm_blocks) {
        _assert(num_lcm_blocks >= 0, "num_lcm_blocks must be >= 0");
        return static_cast<std::size_t>(num_lcm_blocks);
    }

    const LcmBlock& lcmBlock(std::int32_t lcm_block_id) const {
        _assert(lcm_block_id > 0 && static_cast<std::size_t>(lcm_block_id) <= lcm_blocks_.size(),
                "LCM block id out of range");
        return lcm_blocks_[static_cast<std::size_t>(lcm_block_id - 1)];
    }

    CacheBlockRef createBlockRef(std::uint32_t group_id, std::int32_t slots_per_parent, CacheBlockLocation location) {
        using Control = internal_cache_block_ref::CacheBlockControl;
        if (slots_per_parent > kInlineOccupancySlots) [[unlikely]] {
            LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(location.lcm_block_id - 1)];
            if (parent.wide_occupancy == nullptr) {
                allocateWideOccupancy(parent, wide_occupancy_words_);
            }
        }
        void* storage = control_memory_.allocate(sizeof(Control), alignof(Control));
        auto* control = new (storage) Control(*this, location);
        // Allocate a wide parent's occupancy words and the control before
        // mutating the pool, then commit the location before publishing its
        // RAII owner: CacheBlock destruction releases this location and
        // therefore requires it to be occupied.
        occupy(group_id, slots_per_parent, location);
        return CacheBlockRef{*control};
    }

    // Out of line so the allocation stays out of the inlined acquire loops.
    [[gnu::cold, gnu::noinline]] static void allocateWideOccupancy(LcmBlock& parent, std::size_t words) {
        parent.wide_occupancy = std::make_unique<std::uint64_t[]>(words);
    }

    void occupy(std::uint32_t group_id, std::int32_t slots_per_parent, CacheBlockLocation location) noexcept {
        LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        if (slots_per_parent == 1) {
            FatalCheck(parent.occupied_count == 0, "LCM child slot already occupied");
            FatalCheck(!free_parent_ids_.empty() && free_parent_ids_.front() == location.lcm_block_id,
                       "empty LCM placement must consume the next free parent");
            FatalCheck(parent.slots == 0 && parent.occupancy == 0, "empty LCM parent must not retain child slots");
            FatalCheck(location.slot_index == 0, "LCM child slot is out of range");
            free_parent_ids_.pop_front();
            parent.bound_group = group_id;
            parent.slots = 1;
            parent.occupancy = 1;
            parent.occupied_count = 1;
            return;
        }
        GroupAvailability& group = placement(group_id);
        const std::size_t buckets = group.parents_by_bucket.size();
        if (parent.occupied_count == 0) {
            FatalCheck(!free_parent_ids_.empty() && free_parent_ids_.front() == location.lcm_block_id,
                       "empty LCM placement must consume the next free parent");
            FatalCheck(parent.slots == 0 && parent.occupancy == 0, "empty LCM parent must not retain child slots");
            FatalCheck(slots_per_parent <= kInlineOccupancySlots || parent.wide_occupancy != nullptr,
                       "a wide LCM parent binds without its occupancy words");
            FatalCheck(slots_per_parent == 1 ||
                           (buckets <= link_stride_ && bucket_links_.size() == lcm_blocks_.size() * link_stride_),
                       "a packed LCM parent binds without its link storage");
            parent.slots = slots_per_parent;
            if (slots_per_parent > 1) {
                parent.first_free_slots.resize(buckets);
                for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
                    parent.first_free_slots[bucket] = static_cast<std::int32_t>(bucket);
                }
            }
            free_parent_ids_.pop_front();
            parent.bound_group = group_id;
            // A newly bound parent gives every bucket its share of holes.
            const auto per_bucket = static_cast<std::int64_t>(static_cast<std::size_t>(slots_per_parent) / buckets);
            for (std::int64_t& holes : group.free_slots_by_bucket) {
                holes += per_bucket;
            }
        } else {
            removeAvailability(group, parent, location.lcm_block_id);
        }
        FatalCheck(parent.bound_group == group_id && parent.slots == slots_per_parent,
                   "LCM parent binding changed while occupied");
        const std::size_t slot = static_cast<std::size_t>(location.slot_index);
        FatalCheck(slot < static_cast<std::size_t>(parent.slots), "LCM child slot is out of range");
        FatalCheck(!parent.SlotOccupied(slot), "LCM child slot already occupied");
        parent.SetSlotOccupied(slot, true);
        ++parent.occupied_count;
        std::int64_t& bucket_holes = group.free_slots_by_bucket[slot % buckets];
        FatalCheck(bucket_holes > 0, "group free-slot count underflow on occupy");
        --bucket_holes;
        if (slots_per_parent > 1) {
            auto& first = parent.first_free_slots[slot % buckets];
            if (first == location.slot_index) {
                first = slots_per_parent;
                for (std::size_t next = slot + buckets; next < static_cast<std::size_t>(parent.slots);
                     next += buckets) {
                    if (!parent.SlotOccupied(next)) {
                        first = static_cast<std::int32_t>(next);
                        break;
                    }
                }
            }
        }
        addAvailability(group, parent, location.lcm_block_id);
    }

    friend class internal_cache_block_ref::CacheBlockControl;
    void destroyBlockControl(internal_cache_block_ref::CacheBlockControl* control) const noexcept {
        using Control = internal_cache_block_ref::CacheBlockControl;
        control->~Control();  // Releases its physical slot exactly once.
        control_memory_.deallocate(control, sizeof(Control), alignof(Control));
    }

    // Stable slots allocated on demand, independent of physical page IDs.
    // Last-owner release recycles the control only after cancelling observers
    // and releasing its CacheBlock; no stale observer survives slot reuse.
    mutable FixedSlotResource control_memory_{SlotShape::Of<internal_cache_block_ref::CacheBlockControl>(), 256,
                                              std::pmr::new_delete_resource()};
    std::vector<LcmBlock> lcm_blocks_;
    // Free parents are interchangeable: release appends and allocation consumes
    // the front. Bound parents have per-group, per-bucket availability indices.
    std::deque<std::int32_t> free_parent_ids_;
    // Dense group IDs and immutable packing are configured at construction.
    // Bucket geometry is fixed by registration before the first allocation.
    std::vector<GroupAvailability> group_availability_;
    // Words of a wide parent's occupancy bits; 0 when every packing fits inline.
    std::size_t wide_occupancy_words_{0};
    // Flat intrusive links indexed by (lcm_block_id - 1) * link_stride_ + bucket.
    // Remains empty (link_stride_ == 0) on pools with only packing == 1 groups.
    std::vector<BucketLink> bucket_links_;
    std::size_t link_stride_{0};
};

}  // namespace tokenspeed
