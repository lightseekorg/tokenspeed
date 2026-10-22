// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
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
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "utils.h"

namespace tokenspeed {

struct CacheCoordinatorTestAccess {
    static auto MatchPrefix(CacheCoordinator& coordinator, std::span<const std::string> content_hashes) {
        return coordinator.acquirePrefix(coordinator.ProbePrefix(content_hashes), ++coordinator.next_access_epoch_);
    }
    // A probe over the prompt's tokens too (tails included), acquired as an
    // admission would.
    static auto MatchPromptPrefix(CacheCoordinator& coordinator, const CacheCoordinator::PromptPrefix& prompt) {
        return coordinator.acquirePrefix(coordinator.ProbePromptPrefix(prompt), ++coordinator.next_access_epoch_);
    }

    static std::uint64_t NextAccessEpoch(CacheCoordinator& coordinator) { return ++coordinator.next_access_epoch_; }

    // The planner's Device eviction of one entry, with its tail cascade.
    static bool EvictCachedBlock(CacheCoordinator& coordinator, std::uint32_t group_id, CacheBlockLocation location) {
        return coordinator.evictCachedBlock(group_id, location);
    }

    // A Device-cached block, located rather than referenced: a held reference
    // would pin it against eviction.
    static std::optional<CacheBlockLocation> DeviceLocationOf(const CacheCoordinator& coordinator,
                                                              std::size_t group_index, const CacheKey& key) {
        return coordinator.deviceLocationOf(group_index, key);
    }
};

inline auto MatchPrefixForTest(CacheCoordinator& coordinator, std::span<const std::string> content_hashes) {
    return CacheCoordinatorTestAccess::MatchPrefix(coordinator, content_hashes);
}

inline void CacheFullBlocksForTest(CacheCoordinator& coordinator, std::span<BlockTable> tables,
                                   std::span<const std::string> content_hashes, std::int32_t first_slot = 0) {
    coordinator.CacheFullBlocks(tables, content_hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                first_slot, CacheBoundaryKind::kChunk);
}

// Publishes the newly hashed pages [first_new_prefix_page, prefix_hashes.size())
// at a page-aligned point; publish_state marks it a natural checkpoint of the
// State groups (their slot 1 is registered under the last page's hash).
inline void CacheCompletedBlocksForTest(CacheCoordinator& coordinator, std::span<BlockTable> tables,
                                        std::span<const std::string> prefix_hashes, std::uint64_t access_epoch,
                                        std::int32_t first_new_prefix_page, std::int32_t num_computed_tokens,
                                        CacheBoundaryKind boundary_kind, bool stream_completed_to_host,
                                        bool publish_state) {
    _assert(tables.size() == static_cast<std::size_t>(coordinator.NumGroups()), "tables/groups size mismatch");
    _assert(first_new_prefix_page >= 0 && static_cast<std::size_t>(first_new_prefix_page) < prefix_hashes.size(),
            "completed page range must be non-empty");
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = prefix_hashes,
                .first_new_prefix_page = first_new_prefix_page,
                .publish_state = publish_state,
                .boundary_kind = boundary_kind,
                .stream_completed_to_host = stream_completed_to_host,
            },
        .num_computed_tokens = num_computed_tokens,
    };
    coordinator.CacheCompletedBlocks(tables, progress, access_epoch);
}

// Admits every History group with the same demand prototype as a new request;
// a State group takes `state_step` (kRotate: a prefill step, the shape of a
// first admission).
inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(
    CacheCoordinator& coordinator, std::vector<BlockTable>& tables, CacheCoordinator::PrefixProbe&& prefix,
    GroupDemand prototype, const RequestProgress& progress,
    StateStep state_step = StateStep{StateStep::Kind::kRotate}) {
    std::vector<GroupDemand> demands;
    demands.reserve(tables.size());
    for (std::size_t i = 0; i < tables.size(); ++i) {
        GroupDemand demand = prototype;
        demand.table = &tables[i];
        if (coordinator.GroupIsState(i)) {
            demand.extent = state_step;
            demand.reserve_tokens = 0;
        }
        demands.push_back(demand);
    }
    return coordinator.Admit(std::move(prefix), demands, progress, std::nullopt);
}

// The overloads without progress admit a request that has computed nothing
// yet: nothing to publish, nothing for retention to reclaim.
inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(CacheCoordinator& coordinator,
                                                                     std::vector<BlockTable>& tables,
                                                                     CacheCoordinator::PrefixProbe&& prefix,
                                                                     GroupDemand prototype) {
    return AdmitForTest(coordinator, tables, std::move(prefix), prototype, RequestProgress{});
}

inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(CacheCoordinator& coordinator,
                                                                     std::vector<BlockTable>& tables,
                                                                     GroupDemand prototype,
                                                                     const RequestProgress& progress) {
    return AdmitForTest(coordinator, tables, coordinator.ProbePrefix({}), prototype, progress);
}

inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(CacheCoordinator& coordinator,
                                                                     std::vector<BlockTable>& tables,
                                                                     GroupDemand prototype) {
    return AdmitForTest(coordinator, tables, prototype, RequestProgress{});
}

inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(CacheCoordinator& coordinator,
                                                                     std::vector<BlockTable>& tables,
                                                                     std::int32_t num_tokens) {
    return AdmitForTest(coordinator, tables, GroupDemand{.extent = DenseGrowth{num_tokens}});
}

}  // namespace tokenspeed
