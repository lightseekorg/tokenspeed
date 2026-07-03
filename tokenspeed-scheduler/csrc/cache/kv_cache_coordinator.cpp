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

std::vector<std::string> KvCacheCoordinator::keysForGroup(std::span<const std::string> content_hashes,
                                                          std::uint32_t group_id) const {
    std::vector<std::string> keys;
    keys.reserve(content_hashes.size());
    for (const std::string& h : content_hashes) {
        keys.push_back(MakeKeyWithGroupId(h, group_id));
    }
    return keys;
}

CoordinatorMatch KvCacheCoordinator::MatchPrefix(std::span<const std::string> content_hashes) const {
    CoordinatorMatch out;
    out.per_group.resize(groups_.size());
    if (groups_.empty()) {
        return out;  // num_common_blocks stays 0
    }
    std::int32_t common = static_cast<std::int32_t>(content_hashes.size());

    // Pass 1: full-attention groups. Any prefix of a full-attention match is
    // itself a valid shorter match, so min-now-truncate-later is always safe.
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (groups_[i].Spec().kind != AttnKind::kFull) {
            continue;
        }
        std::vector<std::string> keys = keysForGroup(content_hashes, groups_[i].GroupId());
        out.per_group[i] = groups_[i].Manager().MatchPrefix(keys);
        common = std::min(common, static_cast<std::int32_t>(out.per_group[i].blocks.size()));
    }

    // Pass 2: sliding-window groups, each matched BOUNDED to `common` so the
    // trailing contiguous-run invariant is enforced against the bounded end.
    // Truncating an unbounded SWA match instead could cut its run below
    // contiguous_needed and leave a null hole inside the last window of the
    // claimed prefix (attention would read garbage KV). A bounded match may
    // come back shorter than the bound; that lowers `common`, and any SWA
    // group already matched longer must RE-match at the new bound (a shorter
    // bound can move where the run must sit, possibly shortening it again).
    // Iterate to fixpoint: terminates because `common` is a non-negative
    // integer that strictly decreases on every extra pass.
    std::vector<std::int32_t> swa_len(groups_.size(), -1);  // -1: not matched yet
    bool stable = false;
    while (!stable) {
        stable = true;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (groups_[i].Spec().kind == AttnKind::kFull || (swa_len[i] >= 0 && swa_len[i] <= common)) {
                continue;  // full groups truncate later; SWA already valid at <= common
            }
            std::vector<std::string> keys = keysForGroup(content_hashes, groups_[i].GroupId());
            out.per_group[i] = groups_[i].Manager().MatchPrefix(keys, common);
            swa_len[i] = static_cast<std::int32_t>(out.per_group[i].blocks.size());
            if (swa_len[i] < common) {
                common = swa_len[i];
                stable = false;
            }
        }
    }

    // Truncate full-attention groups (matched before the bound settled) to the
    // fixpoint. SWA groups already sit exactly at it: a bounded match is never
    // longer than the bound, and any shorter one became the bound itself.
    for (PrefixMatch& m : out.per_group) {
        if (static_cast<std::int32_t>(m.blocks.size()) > common) {
            m.blocks.resize(static_cast<std::size_t>(common));
            std::int32_t real = 0;
            for (CacheBlock* b : m.blocks) {
                if (!b->IsNull()) {
                    ++real;
                }
            }
            m.num_hit_blocks = real;
        }
    }
    out.num_common_blocks = common;
    return out;
}

void KvCacheCoordinator::ClaimCommonPrefix(std::span<BlockTable> tables, const CoordinatorMatch& hit) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    _assert(hit.per_group.size() == groups_.size(), "hit/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ClaimHitBlocks(tables[i], hit.per_group[i]);
    }
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
    const BlockTable fresh;  // not-yet-allocated request: empty table, no tail credit
    std::int32_t total_needed = 0;
    for (const CacheGroup& group : groups_) {
        total_needed += group.Manager().BlocksNeededFor(fresh, num_tokens);
    }
    return total_needed;
}

bool KvCacheCoordinator::Acquire(std::span<BlockTable> tables, std::int32_t num_tokens) {
    // Check-then-act: sum the blocks every group needs first. If the shared pool
    // cannot supply them all, allocate nothing and fail -- no group is ever left
    // in a partial/unaligned state, so there is no rollback.
    if (BlocksNeededFor(tables, num_tokens) > pool_.NumFreeBlocks()) {
        return false;
    }
    // Capacity guaranteed -> every group's Acquire now succeeds and the tables
    // stay page-aligned.
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        _assert(groups_[i].Manager().Acquire(tables[i], num_tokens), "pre-checked Acquire must succeed");
    }
    return true;
}

void KvCacheCoordinator::CacheFullBlocks(std::span<BlockTable> tables,
                                         std::span<const std::string> content_hashes,
                                         std::int32_t num_full_blocks) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        std::vector<std::string> keys = keysForGroup(content_hashes, groups_[i].GroupId());
        groups_[i].Manager().CacheFullBlocks(tables[i], keys, num_full_blocks);
    }
}

void KvCacheCoordinator::Free(std::span<BlockTable> tables) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().Free(tables[i]);
    }
}

void KvCacheCoordinator::AdvanceWindow(std::span<BlockTable> tables, std::int32_t num_computed_tokens) {
    _assert(static_cast<std::int32_t>(tables.size()) == NumGroups(),
            "AdvanceWindow: tables size must match group count");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (groups_[i].Spec().kind == AttnKind::kSlidingWindow) {
            auto& swa = static_cast<SwaManager&>(groups_[i].Manager());
            swa.AdvanceWindow(tables[i], num_computed_tokens);
        }
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
