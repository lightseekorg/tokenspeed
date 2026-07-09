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

#include "cache/full_attn_manager.h"
#include "cache/swa_manager.h"
#include "scheduler/page_hasher.h"
#include "utils.h"

namespace tokenspeed {

KvCacheCoordinator::KvCacheCoordinator(std::vector<CacheGroup> groups, BlockPool& pool)
    : groups_{std::move(groups)}, pool_{pool} {
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
                                                          std::uint32_t group_id) const {
    std::vector<std::string> keys;
    keys.reserve(content_hashes.size());
    for (const std::string& h : content_hashes) {
        keys.push_back(MakeKeyWithGroupId(h, group_id));
    }
    return keys;
}

namespace {

// Shared match skeleton: one ordered sweep (closed groups first), then re-match any window
// group left above the settled bound -- with 2+ window groups a later group can shrink the
// bound UNDER an earlier one's boundary-dependent match. A re-matched group lands at or
// under the current bound and only a further bound drop can lift it back above, so
// re-matches are finite; single-window models never re-enter, and the result is the
// greatest boundary every group supports. `match(i, bound_tokens)` stores group i's match;
// `extent(i)` reads it back as a token extent.
template <typename MatchGroup, typename ExtentTokens>
std::int32_t SweepThenConverge(std::span<const std::size_t> order, const std::vector<CacheGroup>& groups,
                               std::int32_t bound_tokens, const MatchGroup& match, const ExtentTokens& extent) {
    for (std::size_t i : order) {
        match(i, bound_tokens);
        bound_tokens = std::min(bound_tokens, extent(i));
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i : order) {
            if (groups[i].Manager().MatchIsPrefixClosed() || extent(i) <= bound_tokens) {
                continue;
            }
            match(i, bound_tokens);
            bound_tokens = std::min(bound_tokens, extent(i));
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
        group_keys[i] = keysForGroup(content_hashes, groups_[i].GroupId());
    }
    return group_keys;
}

CoordinatorMatch KvCacheCoordinator::MatchPrefix(std::span<const std::string> content_hashes) const {
    return matchPrefixWithKeys(buildGroupKeys(content_hashes),
                               static_cast<std::int32_t>(content_hashes.size()));
}

HostMatch KvCacheCoordinator::MatchHostExtension(std::span<const std::string> content_hashes,
                                                 std::int32_t device_common_blocks,
                                                 BlockPool& host_pool) const {
    return matchHostExtensionWithKeys(buildGroupKeys(content_hashes),
                                      static_cast<std::int32_t>(content_hashes.size()),
                                      device_common_blocks, host_pool);
}

KvCacheCoordinator::AdmissionMatch KvCacheCoordinator::MatchPrefixWithHostExtension(
    std::span<const std::string> content_hashes, BlockPool& host_pool) const {
    const std::vector<std::vector<std::string>> group_keys = buildGroupKeys(content_hashes);
    const std::int32_t total_blocks = static_cast<std::int32_t>(content_hashes.size());
    AdmissionMatch out;
    out.device = matchPrefixWithKeys(group_keys, total_blocks);
    const std::int32_t device_common_blocks =
        groups_.empty() ? 0 : out.device.num_common_tokens / groups_[0].Spec().page_size;
    out.host = matchHostExtensionWithKeys(group_keys, total_blocks, device_common_blocks, host_pool);
    return out;
}

CoordinatorMatch KvCacheCoordinator::matchPrefixWithKeys(std::span<const std::vector<std::string>> group_keys,
                                                         std::int32_t total_blocks) const {
    CoordinatorMatch out;
    out.per_group.resize(groups_.size());
    if (groups_.empty()) {
        return out;
    }
    // Hashes are one per page at the scheduler granularity; hetero-P converts per group below.
    const std::int32_t common_tokens = SweepThenConverge(
        match_order_, groups_, total_blocks * groups_[0].Spec().page_size,
        [&](std::size_t i, std::int32_t bound_tokens) {
            out.per_group[i] =
                groups_[i].Manager().MatchPrefix(group_keys[i], bound_tokens / groups_[i].Spec().page_size);
        },
        [&](std::size_t i) {
            return static_cast<std::int32_t>(out.per_group[i].blocks.size()) * groups_[i].Spec().page_size;
        });

    // Linear cleanup: closed groups truncate to the converged bound (any prefix stays valid);
    // non-closed groups are at or under it by construction above.
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t page_size = groups_[i].Spec().page_size;
        PrefixMatch& m = out.per_group[i];
        if (static_cast<std::int32_t>(m.blocks.size()) * page_size <= common_tokens) {
            continue;
        }
        _assert(groups_[i].Manager().MatchIsPrefixClosed(), "window group left above the converged bound");
        m.blocks.resize(static_cast<std::size_t>(common_tokens / page_size));
        std::int32_t real = 0;
        for (CacheBlock* b : m.blocks) {
            if (!b->IsNull()) {
                ++real;
            }
        }
        m.num_hit_blocks = real;
    }
    out.num_common_tokens = common_tokens;
    return out;
}

// TODO(flat-l2): probe the device pool before accepting a host page to skip re-loading device-resident blocks.
HostMatch KvCacheCoordinator::matchHostExtensionWithKeys(std::span<const std::vector<std::string>> group_keys,
                                                         std::int32_t total_blocks,
                                                         std::int32_t device_common_blocks,
                                                         BlockPool& host_pool) const {
    HostMatch out;
    out.per_group.resize(groups_.size());
    if (groups_.empty()) {
        return out;
    }
    const std::int32_t page_size = groups_[0].Spec().page_size;
    _assert(device_common_blocks >= 0 && device_common_blocks <= total_blocks,
            "device_common_blocks out of range");
    const std::int32_t dev = device_common_blocks;
    const std::int32_t dev_tokens = dev * page_size;

    // Boundary tracked in TOKENS as future-proofing for heterogeneous page sizes; the slot
    // math still assumes the uniform page size MakeCoordinator asserts.
    const std::int32_t boundary_tokens = SweepThenConverge(
        match_order_, groups_, total_blocks * page_size,
        [&](std::size_t i, std::int32_t bound_tokens) {
            const std::int32_t group_page = groups_[i].Spec().page_size;
            out.per_group[i] = groups_[i].Manager().MatchHostPages(
                host_pool, group_keys[i], dev_tokens / group_page, bound_tokens / group_page);
        },
        [&](std::size_t i) {
            const std::int32_t group_page = groups_[i].Spec().page_size;
            return (dev_tokens / group_page + static_cast<std::int32_t>(out.per_group[i].size())) * group_page;
        });

    const std::int32_t end_blocks = boundary_tokens / page_size;
    out.num_extension_blocks = end_blocks - dev;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        std::vector<CacheBlock*>& pages = out.per_group[i];
        if (static_cast<std::int32_t>(pages.size()) <= out.num_extension_blocks) {
            continue;
        }
        _assert(groups_[i].Manager().MatchIsPrefixClosed(), "window group left above the converged boundary");
        pages.resize(static_cast<std::size_t>(out.num_extension_blocks));
    }
    // Pin after convergence: a later host-pool allocation (this round's store drain) must not
    // evict a matched page before the load emission takes the refs.
    for (const std::vector<CacheBlock*>& pages : out.per_group) {
        for (CacheBlock* block : pages) {
            if (block != nullptr) {
                out.pinned.push_back(BlockRef::Share(host_pool, block));
            }
        }
    }
    return out;
}

void KvCacheCoordinator::ClaimCommonPrefix(std::span<BlockTable> tables, const CoordinatorMatch& hit) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    if (hit.per_group.empty()) {
        _assert(hit.num_common_tokens == 0, "empty per_group with nonzero num_common_tokens");
        return;
    }
    _assert(hit.per_group.size() == groups_.size(), "hit/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ClaimHitBlocks(tables[i], hit.per_group[i]);
    }
}

std::vector<std::pair<std::int32_t, CacheBlock*>> KvCacheCoordinator::LoadHostExtension(std::span<BlockTable> tables,
                                                                                        const HostMatch& host) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    std::vector<std::pair<std::int32_t, CacheBlock*>> pairs;
    if (host.num_extension_blocks == 0) {
        return pairs;
    }
    _assert(host.per_group.size() == groups_.size(), "host match/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().AppendHostExtension(tables[i], host.per_group[i], pairs);
    }
    return pairs;
}

std::int32_t KvCacheCoordinator::BlocksConsumedByClaim(const CoordinatorMatch& hit) const {
    std::int32_t consumed = 0;
    for (const PrefixMatch& match : hit.per_group) {
        for (const CacheBlock* block : match.blocks) {
            if (!block->IsNull() && block->RefCount() == 0) {
                ++consumed;
            }
        }
    }
    return consumed;
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
        _assert(groups_[i].Manager().Acquire(tables[i], num_tokens), "pre-checked Acquire must succeed");
    }
    return true;
}

void KvCacheCoordinator::CacheFullBlocks(std::span<BlockTable> tables,
                                         std::span<const std::string> content_hashes,
                                         std::int32_t first_slot) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    if (content_hashes.empty()) {
        return;  // hot decode rounds usually fill no page
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        std::vector<std::string> keys = keysForGroup(content_hashes, groups_[i].GroupId());
        std::vector<std::pair<std::string, CacheBlock*>> newly_cached;
        groups_[i].Manager().CacheFullBlocks(tables[i], keys, first_slot,
                                             collect_store_candidates_ ? &newly_cached : nullptr);
        for (auto& [key, block] : newly_cached) {
            pending_stores_.push_back(StoreCandidate{std::move(key), BlockRef::Share(pool_, block)});
        }
    }
}

void KvCacheCoordinator::Free(std::span<BlockTable> tables) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().Free(tables[i]);
    }
}

KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, BlockPool& pool) {
    _assert(!specs.empty(), "MakeCoordinator requires at least one spec");
    std::int32_t page_size = specs[0].page_size;
    std::vector<CacheGroup> groups;
    groups.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const KvCacheSpec& spec = specs[i];
        _assert(spec.page_size == page_size, "all groups must share the same page_size");
        std::unique_ptr<KvCacheManager> manager;
        if (spec.kind == AttnKind::kFull) {
            manager = std::make_unique<FullAttnManager>(pool, spec.page_size);
        } else {
            manager = std::make_unique<SwaManager>(pool, spec.page_size, spec.sliding_window);
        }
        groups.emplace_back(spec, static_cast<std::uint32_t>(i), std::move(manager));
    }
    return KvCacheCoordinator{std::move(groups), pool};
}

}  // namespace tokenspeed
