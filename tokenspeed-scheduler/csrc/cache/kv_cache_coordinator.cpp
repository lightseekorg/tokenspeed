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
#include <limits>
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
    out.per_group.reserve(groups_.size());
    std::int32_t common = std::numeric_limits<std::int32_t>::max();
    for (const CacheGroup& g : groups_) {
        std::vector<std::string> keys = keysForGroup(content_hashes, g.GroupId());
        PrefixMatch m = g.Manager().MatchPrefix(keys);
        common = std::min(common, static_cast<std::int32_t>(m.blocks.size()));
        out.per_group.push_back(std::move(m));
    }
    if (groups_.empty()) {
        common = 0;
    }
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

bool KvCacheCoordinator::Acquire(std::span<BlockTable> tables, std::int32_t num_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    // Check-then-act: sum the blocks every group needs first. If the shared pool
    // cannot supply them all, allocate nothing and fail -- no group is ever left
    // in a partial/unaligned state, so there is no rollback.
    std::int32_t total_needed = 0;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        total_needed += groups_[i].Manager().BlocksNeededFor(tables[i], num_tokens);
    }
    if (total_needed > pool_.NumFreeBlocks()) {
        return false;
    }
    // Capacity guaranteed -> every group's Acquire now succeeds and the tables
    // stay page-aligned.
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().Acquire(tables[i], num_tokens);
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
