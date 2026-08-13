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

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"

namespace tokenspeed {

struct CacheCoordinatorTestAccess {
    static auto MatchPrefix(CacheCoordinator& coordinator, std::span<const std::string> content_hashes) {
        return coordinator.acquirePrefix(coordinator.ProbePrefix(content_hashes), ++coordinator.next_access_epoch_);
    }

    static std::uint64_t NextAccessEpoch(CacheCoordinator& coordinator) { return ++coordinator.next_access_epoch_; }
};

inline auto MatchPrefixForTest(CacheCoordinator& coordinator, std::span<const std::string> content_hashes) {
    return CacheCoordinatorTestAccess::MatchPrefix(coordinator, content_hashes);
}

inline void CacheFullBlocksForTest(CacheCoordinator& coordinator, std::span<BlockTable> tables,
                                   std::span<const std::string> content_hashes, std::int32_t first_slot = 0) {
    coordinator.CacheFullBlocks(tables, content_hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                first_slot);
}

inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(CacheCoordinator& coordinator,
                                                                       std::vector<BlockTable>& tables,
                                                                       CacheCoordinator::PrefixProbe&& prefix,
                                                                       GroupDemand prototype) {
    std::vector<GroupDemand> demands;
    demands.reserve(tables.size());
    for (BlockTable& table : tables) {
        prototype.table = &table;
        demands.push_back(prototype);
    }
    return coordinator.Admit(std::move(prefix), demands);
}

inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(CacheCoordinator& coordinator,
                                                                       std::vector<BlockTable>& tables,
                                                                       GroupDemand prototype) {
    return AdmitForTest(coordinator, tables, coordinator.ProbePrefix({}), prototype);
}

inline std::optional<CacheCoordinator::AdmissionResult> AdmitForTest(CacheCoordinator& coordinator,
                                                                       std::vector<BlockTable>& tables,
                                                                       std::int32_t num_tokens) {
    return AdmitForTest(coordinator, tables, GroupDemand{.num_tokens = num_tokens});
}

}  // namespace tokenspeed
