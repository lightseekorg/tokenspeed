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
#include <optional>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "cache/core/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

// Physical LCM placement only. It deliberately has no cache key, LRU node,
// CacheBlock pointer, or ownership count.
//
// Placement never scans the pool: every parent that can still take a child is
// indexed by its group and its current occupancy, and both indices are updated
// as part of the occupy/release transitions that create them.
class BlockPool {
public:
    explicit BlockPool(std::int32_t num_lcm_blocks) : lcm_blocks_(checkedLcmBlockCount(num_lcm_blocks)) {
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

    CacheBlockRef AcquireBlock(std::uint32_t group_id, std::int32_t cache_blocks_per_lcm_block) {
        std::vector<CacheBlockRef> blocks = AcquireBlocks(group_id, cache_blocks_per_lcm_block, 1);
        if (blocks.empty()) {
            return {};
        }
        return std::move(blocks.front());
    }

    std::vector<CacheBlockRef> AcquireBlocks(std::uint32_t group_id, std::int32_t cache_blocks_per_lcm_block,
                                             std::int32_t num) {
        _assert(cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        if (num <= 0) {
            return {};
        }

        std::vector<CacheBlockLocation> locations =
            planLocations(group_id, cache_blocks_per_lcm_block, static_cast<std::size_t>(num));
        if (locations.size() != static_cast<std::size_t>(num)) {
            return {};
        }

        std::vector<CacheBlockRef> out;
        out.reserve(locations.size());
        for (CacheBlockLocation location : locations) {
            out.push_back(createBlockRef(group_id, cache_blocks_per_lcm_block, location));
        }
        return out;
    }

    std::vector<CacheBlockRef> AcquireUpToBlocks(std::uint32_t group_id, std::int32_t cache_blocks_per_lcm_block,
                                                 std::int32_t max_num) {
        _assert(cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        if (max_num <= 0) {
            return {};
        }
        std::vector<CacheBlockLocation> locations =
            planLocations(group_id, cache_blocks_per_lcm_block, static_cast<std::size_t>(max_num));
        std::vector<CacheBlockRef> out;
        out.reserve(locations.size());
        for (CacheBlockLocation location : locations) {
            out.push_back(createBlockRef(group_id, cache_blocks_per_lcm_block, location));
        }
        return out;
    }

    // Places one block per entry of group_ids, in order. A group first spends
    // the holes it already owns; only when those run out does it break into an
    // empty parent, whose remaining slots stay available to its later entries.
    // Entries that find no placement are left null.
    std::vector<CacheBlockRef> AcquireAvailableBlocksInOrder(std::span<const std::uint32_t> group_ids,
                                                             std::span<const std::int32_t> cache_blocks_per_group) {
        std::vector<std::size_t> demand_by_group(cache_blocks_per_group.size(), 0);
        for (std::uint32_t group_id : group_ids) {
            _assert(group_id < cache_blocks_per_group.size(), "group id has no packing");
            _assert(cache_blocks_per_group[group_id] > 0, "cache_blocks_per_lcm_block must be > 0");
            ++demand_by_group[group_id];
        }

        // Plan each requested group's existing holes once, capped by what that
        // group actually asked for.
        std::vector<std::vector<CacheBlockLocation>> available_by_group(cache_blocks_per_group.size());
        std::vector<std::size_t> next_available(cache_blocks_per_group.size(), 0);
        for (std::size_t group_id = 0; group_id < demand_by_group.size(); ++group_id) {
            if (demand_by_group[group_id] > 0) {
                appendPartialLocations(static_cast<std::uint32_t>(group_id), cache_blocks_per_group[group_id],
                                       demand_by_group[group_id], available_by_group[group_id]);
            }
        }

        std::vector<CacheBlockRef> out(group_ids.size());
        for (std::size_t i = 0; i < group_ids.size(); ++i) {
            const std::uint32_t group_id = group_ids[i];
            const std::int32_t packing = cache_blocks_per_group[group_id];
            std::vector<CacheBlockLocation>& available = available_by_group[group_id];
            std::size_t& next = next_available[group_id];
            if (next < available.size()) {
                out[i] = createBlockRef(group_id, packing, available[next++]);
                continue;
            }
            if (free_parent_ids_.empty()) {
                continue;
            }
            const std::int32_t parent_id = free_parent_ids_.front();
            out[i] = createBlockRef(group_id, packing, CacheBlockLocation{.lcm_block_id = parent_id, .slot_index = 0});
            for (std::int32_t slot = 1; slot < packing; ++slot) {
                available.push_back(CacheBlockLocation{.lcm_block_id = parent_id, .slot_index = slot});
            }
        }
        return out;
    }

    std::vector<CacheBlockRef> AcquireUpToBlocksFromEmptyParent(std::uint32_t group_id,
                                                                std::int32_t cache_blocks_per_lcm_block,
                                                                std::int32_t lcm_block_id, std::int32_t max_num) {
        _assert(cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        if (max_num <= 0) {
            return {};
        }
        const LcmBlock& parent = lcmBlock(lcm_block_id);
        _assert(parent.occupied_count == 0 && !parent.bound_group, "directed Host parent must be empty");
        _assert(!free_parent_ids_.empty() && free_parent_ids_.front() == lcm_block_id,
                "directed Host parent must be the next free parent");

        const std::int32_t take = std::min(max_num, cache_blocks_per_lcm_block);
        std::vector<CacheBlockRef> out;
        out.reserve(static_cast<std::size_t>(take));
        for (std::int32_t slot = 0; slot < take; ++slot) {
            out.push_back(createBlockRef(group_id, cache_blocks_per_lcm_block,
                                         CacheBlockLocation{.lcm_block_id = lcm_block_id, .slot_index = slot}));
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
    std::int32_t NumOccupiedSlots() const noexcept { return occupied_slots_; }
    // Free child slots inside parents already bound to group_id. Slots that an
    // empty parent would offer are not counted here; NumEmptyLcmBlocks()
    // reports those, because taking one costs the whole parent.
    std::int32_t NumFreeSlotsInGroup(std::uint32_t group_id) const noexcept {
        return group_id < partials_by_group_.size() ? partials_by_group_[group_id].free_slots : 0;
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
        FatalCheck(parent.bound_group.has_value() && *parent.bound_group < partials_by_group_.size(),
                   "occupied LCM parent lost its group binding");
        GroupPartials& partials = partials_by_group_[*parent.bound_group];
        unindexParent(partials, location.lcm_block_id, parent.occupied_count);
        parent.occupancy[slot] = false;
        --parent.occupied_count;
        ++partials.free_slots;
        --occupied_slots_;
        if (parent.occupied_count > 0) {
            indexParent(partials, location.lcm_block_id, parent.occupied_count);
            return;
        }
        // The parent no longer belongs to the group: its slots stop being that
        // group's local capacity and become an interchangeable empty parent.
        partials.free_slots -= static_cast<std::int32_t>(parent.occupancy.size());
        --partials.bound_parents;
        parent.bound_group.reset();
        parent.occupancy.clear();
        FatalCheck(free_parent_ids_.size() < lcm_blocks_.size(), "free LCM block queue cannot exceed the pool size");
        free_parent_ids_.push_back(location.lcm_block_id);
    }

private:
    struct LcmBlock {
        std::optional<std::uint32_t> bound_group;
        std::vector<bool> occupancy;
        std::uint32_t occupied_count{0};
    };

    // One group's parents, bucketed by how many child slots they hold. Bucket k
    // holds every parent bound to the group with exactly k occupied slots;
    // bucket 0 stays empty because an emptied parent returns to
    // free_parent_ids_, and a full parent is in no bucket at all. Ordered ids
    // give placement a stable tie-break inside one bucket.
    struct GroupPartials {
        std::vector<std::set<std::int32_t>> by_occupancy;
        std::int32_t bound_parents{0};
        std::int32_t free_slots{0};
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

    CacheBlockRef createBlockRef(std::uint32_t group_id, std::int32_t slots_per_parent, CacheBlockLocation location) {
        auto* control = new internal_cache_block_ref::CacheBlockControl(*this, location);
        // Allocate the control before mutating the pool, then commit the
        // location before publishing its RAII owner: CacheBlock destruction
        // releases this location and therefore requires it to be occupied.
        occupy(group_id, slots_per_parent, location);
        return CacheBlockRef{*control};
    }

    // Returns the group's occupancy buckets, sized to its packing on first use.
    GroupPartials& groupPartials(std::uint32_t group_id, std::int32_t slots_per_parent) {
        if (group_id >= partials_by_group_.size()) {
            partials_by_group_.resize(static_cast<std::size_t>(group_id) + 1);
        }
        GroupPartials& partials = partials_by_group_[group_id];
        if (partials.by_occupancy.size() != static_cast<std::size_t>(slots_per_parent)) {
            FatalCheck(partials.bound_parents == 0, "group packing changed while the pool holds its parents");
            partials.by_occupancy.assign(static_cast<std::size_t>(slots_per_parent), std::set<std::int32_t>{});
        }
        return partials;
    }

    // Bucket membership for the two occupancy counts that own no bucket -- an
    // empty parent (free_parent_ids_) and a full one -- is a no-op.
    static void indexParent(GroupPartials& partials, std::int32_t parent_id, std::uint32_t occupied) noexcept {
        if (occupied > 0 && occupied < partials.by_occupancy.size()) {
            partials.by_occupancy[occupied].insert(parent_id);
        }
    }
    static void unindexParent(GroupPartials& partials, std::int32_t parent_id, std::uint32_t occupied) noexcept {
        if (occupied > 0 && occupied < partials.by_occupancy.size()) {
            partials.by_occupancy[occupied].erase(parent_id);
        }
    }

    void occupy(std::uint32_t group_id, std::int32_t slots_per_parent, CacheBlockLocation location) noexcept {
        LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        GroupPartials& partials = groupPartials(group_id, slots_per_parent);
        if (parent.occupied_count == 0) {
            FatalCheck(!free_parent_ids_.empty() && free_parent_ids_.front() == location.lcm_block_id,
                       "empty LCM placement must consume the next free parent");
            FatalCheck(parent.occupancy.empty(), "empty LCM parent must not retain child slots");
            parent.occupancy.assign(static_cast<std::size_t>(slots_per_parent), false);
            free_parent_ids_.pop_front();
            parent.bound_group = group_id;
            ++partials.bound_parents;
            partials.free_slots += slots_per_parent;
        }
        FatalCheck(
            parent.bound_group == group_id && parent.occupancy.size() == static_cast<std::size_t>(slots_per_parent),
            "LCM parent binding changed while occupied");
        const std::size_t slot = static_cast<std::size_t>(location.slot_index);
        FatalCheck(slot < parent.occupancy.size(), "LCM child slot is out of range");
        FatalCheck(!parent.occupancy[slot], "LCM child slot already occupied");
        unindexParent(partials, location.lcm_block_id, parent.occupied_count);
        parent.occupancy[slot] = true;
        ++parent.occupied_count;
        --partials.free_slots;
        ++occupied_slots_;
        indexParent(partials, location.lcm_block_id, parent.occupied_count);
    }

    // Appends up to count free child slots from parents already bound to
    // group_id, densest parent first: a placement should close an open parent
    // before an untouched one is broken into.
    void appendPartialLocations(std::uint32_t group_id, std::int32_t slots_per_parent, std::size_t count,
                                std::vector<CacheBlockLocation>& locations) const {
        if (group_id >= partials_by_group_.size()) {
            return;
        }
        const std::vector<std::set<std::int32_t>>& by_occupancy = partials_by_group_[group_id].by_occupancy;
        for (std::size_t occupied = by_occupancy.size(); occupied > 1;) {
            --occupied;
            for (std::int32_t parent_id : by_occupancy[occupied]) {
                const LcmBlock& parent = lcmBlock(parent_id);
                _assert(parent.occupancy.size() == static_cast<std::size_t>(slots_per_parent),
                        "group packing changed while LCM block is occupied");
                for (std::size_t slot = 0; slot < parent.occupancy.size() && locations.size() < count; ++slot) {
                    if (!parent.occupancy[slot]) {
                        locations.push_back(CacheBlockLocation{
                            .lcm_block_id = parent_id,
                            .slot_index = static_cast<std::int32_t>(slot),
                        });
                    }
                }
                if (locations.size() == count) {
                    return;
                }
            }
        }
    }

    // Appends up to count child slots carved out of the free-parent FIFO.
    void appendEmptyParentLocations(std::int32_t slots_per_parent, std::size_t count,
                                    std::vector<CacheBlockLocation>& locations) const {
        for (std::int32_t parent_id : free_parent_ids_) {
            for (std::int32_t slot = 0; slot < slots_per_parent && locations.size() < count; ++slot) {
                locations.push_back(CacheBlockLocation{
                    .lcm_block_id = parent_id,
                    .slot_index = slot,
                });
            }
            if (locations.size() == count) {
                return;
            }
        }
    }

    std::vector<CacheBlockLocation> planLocations(std::uint32_t group_id, std::int32_t slots_per_parent,
                                                  std::size_t count) const {
        std::vector<CacheBlockLocation> locations;
        locations.reserve(count);
        appendPartialLocations(group_id, slots_per_parent, count, locations);
        appendEmptyParentLocations(slots_per_parent, count, locations);
        return locations;
    }

    std::vector<LcmBlock> lcm_blocks_;
    // Free parents are interchangeable: release appends and allocation consumes
    // the front. Bound parents are selected separately by planLocations().
    std::deque<std::int32_t> free_parent_ids_;
    // Indexed by group id, so the pool can answer "where can this group place a
    // block" and "how much room does this group already own" without walking
    // lcm_blocks_.
    std::vector<GroupPartials> partials_by_group_;
    std::int32_t occupied_slots_{0};
};

}  // namespace tokenspeed
