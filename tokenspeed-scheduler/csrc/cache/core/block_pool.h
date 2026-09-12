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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "cache/core/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

// Physical LCM placement only. It deliberately has no cache key, LRU node,
// or CacheBlock pointer.
class BlockPool {
public:
    explicit BlockPool(std::int32_t num_lcm_blocks, std::vector<std::int32_t> slots_per_group)
        : lcm_blocks_(checkedLcmBlockCount(num_lcm_blocks)), num_available_lcm_blocks_{num_lcm_blocks} {
        group_placements_.reserve(slots_per_group.size());
        for (std::int32_t slots_per_parent : slots_per_group) {
            _assert(slots_per_parent > 0, "slots_per_parent must be > 0");
            _assert(static_cast<std::int64_t>(num_lcm_blocks) * slots_per_parent <=
                        std::numeric_limits<std::int32_t>::max(),
                    "group cache page count exceeds int32 range");
            group_placements_.emplace_back(slots_per_parent);
        }
        for (std::int32_t id = 1; id <= num_lcm_blocks; ++id) {
            free_parent_ids_.push_back(id);
        }
    }

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;
    ~BlockPool() noexcept { FatalCheck(NumOccupiedSlots() == 0, "BlockPool destroyed with live block references"); }

    // Number of physical LCM blocks. Kernel page 0 is reserved separately.
    std::int32_t NumLcmBlocks() const noexcept { return static_cast<std::int32_t>(lcm_blocks_.size()); }
    std::int32_t NumEmptyLcmBlocks() const noexcept { return static_cast<std::int32_t>(free_parent_ids_.size()); }
    std::int32_t NumAvailableLcmBlocks() const noexcept { return num_available_lcm_blocks_; }
    std::int32_t NumFreeSlots(std::uint32_t group_id) const { return group_placements_[group_id].FreeSlots(); }

    CacheBlockRef AcquireBlock(std::uint32_t group_id) {
        std::vector<CacheBlockRef> blocks = AcquireBlocks(group_id, 1);
        if (blocks.empty()) {
            return {};
        }
        return std::move(blocks.front());
    }

    std::vector<CacheBlockRef> AcquireBlocks(std::uint32_t group_id, std::int32_t num) {
        if (num <= 0) {
            return {};
        }

        std::vector<CacheBlockLocation> locations = planLocations(group_id, static_cast<std::size_t>(num));
        if (locations.size() != static_cast<std::size_t>(num)) {
            return {};
        }

        std::vector<CacheBlockRef> out;
        out.reserve(locations.size());
        for (CacheBlockLocation location : locations) {
            out.push_back(createBlockRef(group_id, location));
        }
        return out;
    }

    std::vector<CacheBlockRef> AcquireUpToBlocks(std::uint32_t group_id, std::int32_t max_num) {
        if (max_num <= 0) {
            return {};
        }
        std::vector<CacheBlockLocation> locations = planLocations(group_id, static_cast<std::size_t>(max_num));
        std::vector<CacheBlockRef> out;
        out.reserve(locations.size());
        for (CacheBlockLocation location : locations) {
            out.push_back(createBlockRef(group_id, location));
        }
        return out;
    }

    std::vector<CacheBlockRef> AcquireAvailableBlocksInOrder(std::span<const std::uint32_t> group_ids) {
        std::vector<std::deque<CacheBlockLocation>> available_locations(group_placements_.size());
        std::vector<bool> initialized_groups(group_placements_.size(), false);
        for (std::uint32_t group_id : group_ids) {
            const GroupPlacement& placement = group_placements_[group_id];
            if (initialized_groups[group_id]) {
                continue;
            }
            initialized_groups[group_id] = true;
            for (const auto& [negative_occupancy, parent_id] : placement.PartialParents()) {
                _assert(negative_occupancy == -OccupiedCount(parent_id), "partial-parent index has stale occupancy");
                const LcmBlock& parent = lcmBlock(parent_id);
                for (std::size_t slot = 0; slot < parent.occupancy.size(); ++slot) {
                    if (!parent.occupancy[slot]) {
                        available_locations[group_id].push_back(CacheBlockLocation{
                            .lcm_block_id = parent_id,
                            .slot_index = static_cast<std::int32_t>(slot),
                        });
                    }
                }
            }
        }

        std::vector<CacheBlockRef> out(group_ids.size());
        for (std::size_t i = 0; i < group_ids.size(); ++i) {
            const std::uint32_t group_id = group_ids[i];
            const std::int32_t slots_per_parent = group_placements_[group_id].SlotsPerParent();
            std::deque<CacheBlockLocation>& available = available_locations[group_id];
            if (available.empty()) {
                if (free_parent_ids_.empty()) {
                    continue;
                }
                const std::int32_t parent_id = free_parent_ids_.front();
                out[i] = createBlockRef(group_id, CacheBlockLocation{.lcm_block_id = parent_id, .slot_index = 0});
                for (std::int32_t slot = 1; slot < slots_per_parent; ++slot) {
                    available.push_back(CacheBlockLocation{.lcm_block_id = parent_id, .slot_index = slot});
                }
                continue;
            }
            const CacheBlockLocation location = available.front();
            available.pop_front();
            out[i] = createBlockRef(group_id, location);
        }
        return out;
    }

    std::vector<CacheBlockRef> AcquireUpToBlocksFromEmptyParent(std::uint32_t group_id, std::int32_t lcm_block_id,
                                                                std::int32_t max_num) {
        const std::int32_t slots_per_parent = group_placements_[group_id].SlotsPerParent();
        if (max_num <= 0) {
            return {};
        }
        const LcmBlock& parent = lcmBlock(lcm_block_id);
        _assert(parent.occupied_count == 0 && !parent.bound_group, "directed Host parent must be empty");
        _assert(!free_parent_ids_.empty() && free_parent_ids_.front() == lcm_block_id,
                "directed Host parent must be the next free parent");

        const std::int32_t take = std::min(max_num, slots_per_parent);
        std::vector<CacheBlockRef> out;
        out.reserve(static_cast<std::size_t>(take));
        for (std::int32_t slot = 0; slot < take; ++slot) {
            out.push_back(
                createBlockRef(group_id, CacheBlockLocation{.lcm_block_id = lcm_block_id, .slot_index = slot}));
        }
        return out;
    }

    std::optional<std::uint32_t> BoundGroup(std::int32_t lcm_block_id) const {
        return lcmBlock(lcm_block_id).bound_group;
    }
    std::int32_t OccupiedCount(std::int32_t lcm_block_id) const {
        return static_cast<std::int32_t>(lcmBlock(lcm_block_id).occupied_count);
    }
    bool IsOccupied(CacheBlockLocation location) const {
        const LcmBlock& lcm_block = lcmBlock(location.lcm_block_id);
        return location.slot_index >= 0 && static_cast<std::size_t>(location.slot_index) < lcm_block.occupancy.size() &&
               lcm_block.occupancy[static_cast<std::size_t>(location.slot_index)];
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
        for (std::size_t slot = 0; slot < lcm_block.occupancy.size(); ++slot) {
            if (lcm_block.occupancy[slot]) {
                locations.push_back(
                    CacheBlockLocation{.lcm_block_id = lcm_block_id, .slot_index = static_cast<std::int32_t>(slot)});
            }
        }
        return locations;
    }

    void Release(CacheBlockLocation location) noexcept {
        FatalCheck(location.lcm_block_id > 0 && static_cast<std::size_t>(location.lcm_block_id) <= lcm_blocks_.size(),
                   "CacheBlock location has invalid LCM block id");
        LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        FatalCheck(location.slot_index >= 0 && static_cast<std::size_t>(location.slot_index) < parent.occupancy.size(),
                   "CacheBlock location has invalid slot");
        const std::size_t slot = static_cast<std::size_t>(location.slot_index);
        FatalCheck(parent.occupancy[slot] && parent.occupied_count > 0, "CacheBlock location is not occupied");
        FatalCheck(!parent.reclaimable[slot], "cache-owned block released without dropping cache ownership");
        FatalCheck(parent.bound_group.has_value(), "occupied LCM parent has no group binding");
        GroupPlacement& placement = group_placements_[*parent.bound_group];
        const std::uint32_t old_occupied_count = parent.occupied_count;
        parent.occupancy[slot] = false;
        FatalCheck(parent.non_reclaimable_count > 0, "released block is missing its non-reclaimable count");
        --parent.non_reclaimable_count;
        if (parent.non_reclaimable_count == 0) {
            ++num_available_lcm_blocks_;
        }
        --parent.occupied_count;
        placement.ReleaseSlot(location.lcm_block_id, old_occupied_count);
        if (parent.occupied_count == 0) {
            parent.bound_group.reset();
            parent.occupancy.clear();
            parent.reclaimable.clear();
            FatalCheck(free_parent_ids_.size() < lcm_blocks_.size(),
                       "free LCM block queue cannot exceed the pool size");
            free_parent_ids_.push_back(location.lcm_block_id);
        }
    }

private:
    struct LcmBlock {
        std::optional<std::uint32_t> bound_group;
        std::vector<bool> occupancy;
        std::vector<bool> reclaimable;
        std::uint32_t occupied_count{0};
        std::uint32_t non_reclaimable_count{0};
    };

    class GroupPlacement {
    public:
        using ParentIndex = std::set<std::pair<std::int32_t, std::int32_t>>;

        explicit GroupPlacement(std::int32_t slots_per_parent) : slots_per_parent_{slots_per_parent} {}

        void OccupySlot(std::int32_t parent_id, std::uint32_t old_occupied_count) noexcept {
            FatalCheck(old_occupied_count < static_cast<std::uint32_t>(slots_per_parent_),
                       "cannot occupy a slot in a full LCM parent");
            transitionParent(parent_id, old_occupied_count, old_occupied_count + 1);
        }

        void ReleaseSlot(std::int32_t parent_id, std::uint32_t old_occupied_count) noexcept {
            FatalCheck(old_occupied_count > 0 && old_occupied_count <= static_cast<std::uint32_t>(slots_per_parent_),
                       "cannot release a slot from an empty LCM parent");
            transitionParent(parent_id, old_occupied_count, old_occupied_count - 1);
        }

        std::int32_t SlotsPerParent() const noexcept { return slots_per_parent_; }
        std::int32_t FreeSlots() const noexcept { return free_slots_; }
        const ParentIndex& PartialParents() const noexcept { return partial_parents_; }

    private:
        std::int32_t freeSlotsFor(std::uint32_t occupied_count) const noexcept {
            return occupied_count == 0 ? 0 : slots_per_parent_ - static_cast<std::int32_t>(occupied_count);
        }

        void transitionParent(std::int32_t parent_id, std::uint32_t old_occupied_count,
                              std::uint32_t new_occupied_count) noexcept {
            if (old_occupied_count > 0 && old_occupied_count < static_cast<std::uint32_t>(slots_per_parent_)) {
                const std::size_t erased =
                    partial_parents_.erase({-static_cast<std::int32_t>(old_occupied_count), parent_id});
                FatalCheck(erased == 1, "partial LCM parent is missing from its placement index");
            }

            if (old_occupied_count == 0) {
                ++bound_parents_;
            } else if (new_occupied_count == 0) {
                FatalCheck(bound_parents_ > 0, "group placement has no bound LCM parent to release");
                --bound_parents_;
            }

            free_slots_ += freeSlotsFor(new_occupied_count) - freeSlotsFor(old_occupied_count);
            FatalCheck(free_slots_ >= 0, "group free-slot count underflow");

            if (new_occupied_count > 0 && new_occupied_count < static_cast<std::uint32_t>(slots_per_parent_)) {
                const bool inserted =
                    partial_parents_.emplace(-static_cast<std::int32_t>(new_occupied_count), parent_id).second;
                FatalCheck(inserted, "partial LCM parent was indexed twice");
            }

            if (bound_parents_ == 0) {
                FatalCheck(free_slots_ == 0 && partial_parents_.empty(),
                           "empty group placement retained physical capacity");
            }
        }

        const std::int32_t slots_per_parent_;
        std::int32_t free_slots_{0};
        std::int32_t bound_parents_{0};
        ParentIndex partial_parents_;
    };

    static std::size_t checkedLcmBlockCount(std::int32_t num_lcm_blocks) {
        _assert(num_lcm_blocks >= 0, "num_lcm_blocks must be >= 0");
        return static_cast<std::size_t>(num_lcm_blocks);
    }

    const LcmBlock& lcmBlock(std::int32_t lcm_block_id) const {
        _assert(lcm_block_id > 0 && static_cast<std::size_t>(lcm_block_id) <= lcm_blocks_.size(),
                "LCM block id out of range");
        return lcm_blocks_[static_cast<std::size_t>(lcm_block_id - 1)];
    }

    CacheBlockRef createBlockRef(std::uint32_t group_id, CacheBlockLocation location) {
        auto* control = new internal_cache_block_ref::CacheBlockControl(*this, location);
        // Allocate the control before mutating the pool, then commit the
        // location before publishing its RAII owner: CacheBlock destruction
        // releases this location and therefore requires it to be occupied.
        occupy(group_id, location);
        return CacheBlockRef{*control};
    }

    void occupy(std::uint32_t group_id, CacheBlockLocation location) noexcept {
        LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        GroupPlacement& placement = group_placements_[group_id];
        const std::int32_t slots_per_parent = placement.SlotsPerParent();
        const std::uint32_t old_occupied_count = parent.occupied_count;
        if (parent.occupied_count == 0) {
            FatalCheck(!free_parent_ids_.empty() && free_parent_ids_.front() == location.lcm_block_id,
                       "empty LCM placement must consume the next free parent");
            FatalCheck(parent.occupancy.empty(), "empty LCM parent must not retain child slots");
            parent.occupancy.assign(static_cast<std::size_t>(slots_per_parent), false);
            parent.reclaimable.assign(static_cast<std::size_t>(slots_per_parent), false);
            free_parent_ids_.pop_front();
            parent.bound_group = group_id;
        }
        FatalCheck(
            parent.bound_group == group_id && parent.occupancy.size() == static_cast<std::size_t>(slots_per_parent),
            "LCM parent binding changed while occupied");
        const std::size_t slot = static_cast<std::size_t>(location.slot_index);
        FatalCheck(slot < parent.occupancy.size(), "LCM child slot is out of range");
        FatalCheck(!parent.occupancy[slot], "LCM child slot already occupied");
        parent.occupancy[slot] = true;
        parent.reclaimable[slot] = false;
        if (parent.non_reclaimable_count == 0) {
            FatalCheck(num_available_lcm_blocks_ > 0, "available LCM block count underflow");
            --num_available_lcm_blocks_;
        }
        ++parent.non_reclaimable_count;
        ++parent.occupied_count;
        placement.OccupySlot(location.lcm_block_id, old_occupied_count);
    }

    std::vector<CacheBlockLocation> planLocations(std::uint32_t group_id, std::size_t count) const {
        const GroupPlacement& placement = group_placements_[group_id];
        const std::int32_t slots_per_parent = placement.SlotsPerParent();
        if (slots_per_parent == 1) {
            std::vector<CacheBlockLocation> locations;
            const std::size_t take = std::min(count, free_parent_ids_.size());
            locations.reserve(take);
            for (std::size_t i = 0; i < take; ++i) {
                locations.push_back(CacheBlockLocation{.lcm_block_id = free_parent_ids_[i], .slot_index = 0});
            }
            return locations;
        }

        std::vector<CacheBlockLocation> locations;
        locations.reserve(count);
        for (const auto& [negative_occupancy, lcm_block_id] : placement.PartialParents()) {
            _assert(negative_occupancy == -OccupiedCount(lcm_block_id), "partial-parent index has stale occupancy");
            const LcmBlock& parent = lcmBlock(lcm_block_id);
            for (std::size_t slot = 0; slot < parent.occupancy.size() && locations.size() < count; ++slot) {
                if (!parent.occupancy[slot]) {
                    locations.push_back(CacheBlockLocation{
                        .lcm_block_id = lcm_block_id,
                        .slot_index = static_cast<std::int32_t>(slot),
                    });
                }
            }
            if (locations.size() == count) {
                return locations;
            }
        }

        // Allocate free lcmBlock if partially-filled block is not enough.
        for (std::int32_t lcm_block_id : free_parent_ids_) {
            for (std::int32_t slot = 0; slot < slots_per_parent && locations.size() < count; ++slot) {
                locations.push_back(CacheBlockLocation{
                    .lcm_block_id = lcm_block_id,
                    .slot_index = slot,
                });
            }
            if (locations.size() == count) {
                return locations;
            }
        }
        return locations;
    }

    void SetReclaimable(CacheBlockLocation location, bool reclaimable) noexcept {
        FatalCheck(location.lcm_block_id > 0 && static_cast<std::size_t>(location.lcm_block_id) <= lcm_blocks_.size(),
                   "CacheBlock location has invalid LCM block id");
        LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        FatalCheck(location.slot_index >= 0 && static_cast<std::size_t>(location.slot_index) < parent.occupancy.size(),
                   "CacheBlock location has invalid slot");
        const std::size_t slot = static_cast<std::size_t>(location.slot_index);
        FatalCheck(parent.occupancy[slot], "CacheBlock location is not occupied");
        if (parent.reclaimable[slot] == reclaimable) {
            return;
        }
        parent.reclaimable[slot] = reclaimable;
        if (reclaimable) {
            FatalCheck(parent.non_reclaimable_count > 0, "reclaimable child count underflow");
            --parent.non_reclaimable_count;
            if (parent.non_reclaimable_count == 0) {
                ++num_available_lcm_blocks_;
            }
        } else {
            if (parent.non_reclaimable_count == 0) {
                FatalCheck(num_available_lcm_blocks_ > 0, "available LCM block count underflow");
                --num_available_lcm_blocks_;
            }
            ++parent.non_reclaimable_count;
        }
    }

    std::vector<LcmBlock> lcm_blocks_;
    // Free parents are interchangeable: release appends and allocation consumes
    // the front. Bound parents are selected separately by planLocations().
    std::deque<std::int32_t> free_parent_ids_;
    // C++ group ids are dense scheduler indices. Their immutable packing is
    // configured with the pool before any physical parent is allocated.
    std::vector<GroupPlacement> group_placements_;
    std::int32_t num_available_lcm_blocks_{0};

    friend class CacheBlock;
};

}  // namespace tokenspeed
