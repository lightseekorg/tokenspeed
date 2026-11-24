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
#include <cstdint>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "cache/coordinator/cache_store.h"

namespace tokenspeed {
struct CleanHostCopy {
    std::uint64_t epoch;
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> members;
};

// Every removed key, including an implicit State cascade, already has a
// published Disk copy. No new write or capacity promise makes a copy clean.
inline std::vector<CleanHostCopy> DiskCleanHostCopies(const CacheStore& store) {
    std::vector<CleanHostCopy> result;
    if (!store.DiskPool() || !store.HostPool()) return result;
    std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> seen;
    for (const auto& group : store.Groups()) {
        group.Index().ForEachEntry(*store.HostPool(), [&](const CacheKey& key, const auto& entry) {
            if (entry.pinned || seen.contains(entry.location) || !group.Index().Contains(*store.DiskPool(), key))
                return;
            CleanHostCopy copy{entry.metadata.eviction.last_access_epoch, {{group.Id(), entry.location}}};
            bool clean = true;
            if (store.GroupIsState(group.Id())) {
                store.ForEachBoundaryKey(key, 0, [&](std::size_t g, const CacheKey& sibling) {
                    if (g == group.Id() || !store.GroupIsState(g)) return;
                    const auto host = store.Groups()[g].Index().Inspect(*store.HostPool(), sibling);
                    if (!host || host->pinned) return;
                    if (!store.Groups()[g].Index().Contains(*store.DiskPool(), sibling)) clean = false;
                    copy.members.emplace_back(static_cast<std::uint32_t>(g), host->location);
                    copy.epoch = std::max(copy.epoch, host->metadata.eviction.last_access_epoch);
                });
            }
            if (!clean) return;
            for (const auto& [g, location] : copy.members) seen.insert(location);
            result.push_back(std::move(copy));
        });
    }
    std::ranges::sort(result, [](const auto& a, const auto& b) {
        return std::tuple{a.epoch, a.members.front().first, a.members.front().second.lcm_block_id,
                          a.members.front().second.slot_index} < std::tuple{b.epoch, b.members.front().first,
                                                                            b.members.front().second.lcm_block_id,
                                                                            b.members.front().second.slot_index};
    });
    return result;
}
}  // namespace tokenspeed
