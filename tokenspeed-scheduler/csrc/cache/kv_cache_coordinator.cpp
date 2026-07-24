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

#include "cache/kv_cache_coordinator.h"

#include <algorithm>
#include <memory>
#include <numeric>

#include "cache/full_attn_manager.h"
#include "cache/mamba_state_manager.h"
#include "cache/swa_manager.h"
#include "scheduler/page_hasher.h"
#include "utils.h"

namespace tokenspeed {

KvCacheCoordinator::KvCacheCoordinator(std::vector<CacheGroup> groups, BlockPool& pool, BlockPool* host_pool,
                                       std::int32_t base_block_size, std::int32_t lcm_block_size)
    : groups_{std::move(groups)},
      pool_{pool},
      host_pool_{host_pool},
      base_block_size_{base_block_size},
      lcm_block_size_{lcm_block_size} {
    _assert(base_block_size_ > 0 && lcm_block_size_ > 0, "coordinator needs positive base/lcm block sizes");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (groups_[i].Manager().MatchIsPrefixClosed()) {
            match_order_.push_back(i);
        }
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!groups_[i].Manager().MatchIsPrefixClosed()) {
            match_order_.push_back(i);
        }
    }
}

std::vector<std::string> KvCacheCoordinator::keysForGroup(std::span<const std::string> content_hashes,
                                                          std::uint32_t group_id, std::int32_t group_block_size,
                                                          std::int32_t first_base) const {
    const std::int32_t m = group_block_size / base_block_size_;
    return MakeFoldedGroupKeys(content_hashes, group_id, m, first_base);
}

namespace {

// Shared match skeleton: one ordered sweep (closed groups first), then re-match any window
// group left above the settled bound -- with 2+ window groups a later group can shrink the
// bound UNDER an earlier one's boundary-dependent match. A re-matched group lands at or
// under the current bound and only a further bound drop can lift it back above, so
// re-matches are finite; the result is the greatest boundary every group supports.
//
// Bounds align DOWN to align_tokens (the groups' lcm): a servable boundary must land on a
// whole block of EVERY group, else a closed group's floor-truncation drops straddled coverage.
template <typename MatchGroup, typename ExtentTokens>
std::int32_t SweepThenConverge(std::span<const std::size_t> order, const std::vector<CacheGroup>& groups,
                               std::int32_t bound_tokens, std::int32_t align_tokens, const MatchGroup& match,
                               const ExtentTokens& extent) {
    const auto align_down = [align_tokens](std::int32_t tokens) { return tokens - tokens % align_tokens; };
    bound_tokens = align_down(bound_tokens);
    for (std::size_t i : order) {
        match(i, bound_tokens);
        bound_tokens = std::min(bound_tokens, align_down(extent(i)));
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i : order) {
            if (groups[i].Manager().MatchIsPrefixClosed() || extent(i) <= bound_tokens) {
                continue;
            }
            match(i, bound_tokens);
            bound_tokens = std::min(bound_tokens, align_down(extent(i)));
            changed = true;
        }
    }
    return bound_tokens;
}

}  // namespace

std::vector<std::vector<std::string>> KvCacheCoordinator::buildGroupKeys(
    std::span<const std::string> content_hashes) const {
    std::vector<std::vector<std::string>> group_keys(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        group_keys[i] = keysForGroup(content_hashes, groups_[i].GroupId(), groups_[i].Spec().block_size,
                                     /*first_base=*/0);
    }
    return group_keys;
}

// The one tier matcher: slots below floor_tokens are assumed valid in a lower tier; per_group
// blocks are relative to the floor, num_common_tokens is the absolute converged boundary (in
// TOKENS). num_base_pages = content_hashes.size(); group_keys[i] is folded to group i's blocks.
KvCacheCoordinator::CoordinatorProbe KvCacheCoordinator::probeTierWithKeys(
    const BlockPool& pool, std::span<const std::vector<std::string>> group_keys, std::int32_t num_base_pages,
    std::int32_t floor_tokens) const {
    CoordinatorProbe out;
    out.per_group.resize(groups_.size());
    if (groups_.empty()) {
        return out;
    }
    const std::int32_t boundary_tokens = SweepThenConverge(
        match_order_, groups_, num_base_pages * base_block_size_, lcm_block_size_,
        [&](std::size_t i, std::int32_t bound_tokens) {
            const std::int32_t group_block_size = groups_[i].Spec().block_size;
            out.per_group[i] = groups_[i].Manager().Probe(pool, group_keys[i], floor_tokens / group_block_size,
                                                          bound_tokens / group_block_size);
        },
        [&](std::size_t i) {
            const std::int32_t group_block_size = groups_[i].Spec().block_size;
            return (floor_tokens / group_block_size + static_cast<std::int32_t>(out.per_group[i].hits.size())) *
                   group_block_size;
        });

    // Truncate closed probes to the converged boundary.
    // Non-closed groups were re-probed against the settled bound and are already at or below it.
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t group_block_size = groups_[i].Spec().block_size;
        PrefixProbe& probe = out.per_group[i];
        const std::int32_t floor_blocks = floor_tokens / group_block_size;
        if ((floor_blocks + static_cast<std::int32_t>(probe.hits.size())) * group_block_size > boundary_tokens) {
            _assert(groups_[i].Manager().MatchIsPrefixClosed(), "window group left above the converged boundary");
            probe.hits.resize(static_cast<std::size_t>(boundary_tokens / group_block_size - floor_blocks));
        }
        for (std::size_t j = 0; j < probe.hits.size(); ++j) {
            if (probe.hits[j] != 0 &&
                pool.IsCachedBlockFree(group_keys[i][static_cast<std::size_t>(floor_blocks) + j])) {
                ++out.num_free_hit_blocks;
            }
        }
    }
    out.num_common_tokens = boundary_tokens;
    return out;
}

CoordinatorMatch KvCacheCoordinator::acquireTierWithKeys(BlockPool& pool,
                                                         std::span<const std::vector<std::string>> group_keys,
                                                         std::int32_t floor_tokens, CoordinatorProbe&& probe) const {
    CoordinatorMatch out;
    out.num_common_tokens = probe.num_common_tokens;
    out.per_group.resize(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t floor_blocks = floor_tokens / groups_[i].Spec().block_size;
        out.per_group[i] =
            groups_[i].Manager().AcquireMatchedBlocks(pool, group_keys[i], floor_blocks, probe.per_group[i]);
    }
    return out;
}

KvCacheCoordinator::AdmissionProbe KvCacheCoordinator::ProbePrefix(std::span<const std::string> content_hashes) const {
    const std::vector<std::vector<std::string>> group_keys = buildGroupKeys(content_hashes);
    const std::int32_t num_base_pages = static_cast<std::int32_t>(content_hashes.size());
    AdmissionProbe out;
    out.device = probeTierWithKeys(pool_, group_keys, num_base_pages, /*floor_tokens=*/0);
    if (host_pool_ != nullptr) {
        out.host = probeTierWithKeys(*host_pool_, group_keys, num_base_pages,
                                     /*floor_tokens=*/out.device.num_common_tokens);
    }
    return out;
}

KvCacheCoordinator::AdmissionMatch KvCacheCoordinator::AcquirePrefix(std::span<const std::string> content_hashes,
                                                                     AdmissionProbe&& probe) {
    const std::vector<std::vector<std::string>> group_keys = buildGroupKeys(content_hashes);
    AdmissionMatch out;
    out.device = acquireTierWithKeys(pool_, group_keys, /*floor_tokens=*/0, std::move(probe.device));
    if (host_pool_ != nullptr) {
        out.host = acquireTierWithKeys(*host_pool_, group_keys, out.device.num_common_tokens, std::move(probe.host));
    }
    return out;
}

KvCacheCoordinator::AdmissionMatch KvCacheCoordinator::MatchPrefix(std::span<const std::string> content_hashes) {
    return AcquirePrefix(content_hashes, ProbePrefix(content_hashes));
}

void KvCacheCoordinator::ClaimCommonPrefix(std::span<BlockTable> tables, CoordinatorMatch&& hit) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    if (hit.per_group.empty()) {
        _assert(hit.num_common_tokens == 0, "empty per_group with nonzero num_common_tokens");
        return;
    }
    _assert(hit.per_group.size() == groups_.size(), "hit/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ClaimHitBlocks(tables[i], std::move(hit.per_group[i]));
    }
}

std::vector<BlockTransfer> KvCacheCoordinator::LoadHostExtension(std::span<BlockTable> tables,
                                                                 CoordinatorMatch&& host) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    std::vector<BlockTransfer> pairs;
    if (host.per_group.empty()) {
        return pairs;
    }
    _assert(host.per_group.size() == groups_.size(), "host match/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().AppendHostExtension(pool_, tables[i], std::move(host.per_group[i].blocks), pairs);
    }
    return pairs;
}

std::int32_t KvCacheCoordinator::BlocksNeededFor(std::span<const BlockTable> tables, std::int32_t num_tokens) const {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    std::int32_t total_needed = 0;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        total_needed += groups_[i].Manager().BlocksNeededFor(tables[i], num_tokens);
    }
    return total_needed;
}

std::int32_t KvCacheCoordinator::BlocksNeededFor(std::int32_t num_tokens) const {
    const BlockTable fresh;
    std::int32_t total_needed = 0;
    for (const CacheGroup& group : groups_) {
        total_needed += group.Manager().BlocksNeededFor(fresh, num_tokens);
    }
    return total_needed;
}

bool KvCacheCoordinator::Acquire(std::span<BlockTable> tables, std::int32_t num_tokens) {
    // Check-then-act: no group is ever left in a partial/unaligned state.
    if (BlocksNeededFor(tables, num_tokens) > pool_.NumFreeBlocks()) {
        return false;
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const bool acquired = groups_[i].Manager().Acquire(pool_, tables[i], num_tokens);
        _assert(acquired, "pre-checked Acquire must succeed");
    }
    return true;
}

void KvCacheCoordinator::CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                                         std::int32_t first_slot, std::int32_t end_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    if (content_hashes.empty()) {
        return;  // hot decode rounds usually fill no page
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t group_block_size = groups_[i].Spec().block_size;
        const std::int32_t m = group_block_size / base_block_size_;
        // The first coarse block sits at table slot ceil(first_slot / m): the first grid-aligned
        // block fully inside the range (fold drops any leading remainder).
        std::vector<std::string> keys =
            keysForGroup(content_hashes, groups_[i].GroupId(), group_block_size, /*first_base=*/first_slot);
        std::int32_t group_first_slot = (first_slot + m - 1) / m;
        std::span<const std::string> group_keys = keys;
        if (groups_[i].Manager().RegistersAlignedFinalPageOnly()) {
            // Interior boundaries never received a state write; only an aligned chunk end holds a
            // real snapshot, in the final full coarse block.
            if (end_tokens < 0 || end_tokens % group_block_size != 0 || keys.empty()) {
                continue;
            }
            const std::int32_t past_end_slot = group_first_slot + static_cast<std::int32_t>(keys.size());
            group_first_slot = past_end_slot - 1;
            group_keys = group_keys.last(1);
            const bool aligned_range = past_end_slot == end_tokens / group_block_size;
            _assert(aligned_range, "state registration range must end at the aligned boundary");
        }
        std::vector<std::pair<std::string, BlockRef>> newly_cached;
        groups_[i].Manager().CacheFullBlocks(pool_, tables[i], group_keys, group_first_slot,
                                             host_pool_ != nullptr ? &newly_cached : nullptr);
        for (auto& [key, block] : newly_cached) {
            pending_stores_.push_back(StoreCandidate{std::move(key), std::move(block)});
        }
    }
}

void KvCacheCoordinator::ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ReclaimExpired(pool_, tables[i], num_computed_tokens);
    }
}

void KvCacheCoordinator::Free(std::span<BlockTable> tables) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().Free(tables[i]);
    }
}

KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, BlockPool& pool, BlockPool* host_pool) {
    _assert(!specs.empty(), "MakeCoordinator requires at least one spec");
    std::int32_t base = specs[0].block_size;
    std::int32_t lcm = specs[0].block_size;
    for (const KvCacheSpec& spec : specs) {
        base = std::gcd(base, spec.block_size);
        lcm = std::lcm(lcm, spec.block_size);
    }
    std::vector<CacheGroup> groups;
    groups.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const KvCacheSpec& spec = specs[i];
        _assert(spec.block_size % base == 0, "group block_size must be a multiple of base");
        std::unique_ptr<KvCacheManager> manager;
        if (spec.kind == AttnKind::kFull) {
            manager = std::make_unique<FullAttnManager>(spec.block_size);
        } else if (spec.kind == AttnKind::kMambaState) {
            manager = std::make_unique<MambaStateManager>(spec.block_size);
        } else {
            manager = std::make_unique<SwaManager>(spec.block_size, spec.sliding_window);
        }
        groups.emplace_back(spec, static_cast<std::uint32_t>(i), std::move(manager));
    }
    return KvCacheCoordinator{std::move(groups), pool, host_pool, base, lcm};
}

}  // namespace tokenspeed
