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

#include "scheduler/operations/group_demands.h"

#include <algorithm>

#include "utils.h"

namespace tokenspeed {

std::vector<GroupDemand> MakeGroupDemands(std::vector<BlockTable>& tables, GroupDemand prototype) {
    std::vector<GroupDemand> demands;
    demands.reserve(tables.size());
    for (BlockTable& table : tables) {
        prototype.table = &table;
        demands.push_back(prototype);
    }
    return demands;
}

std::int64_t SnapshotStateReserveTokens(std::int64_t block_granularity, std::int64_t decode_tokens) {
    return std::max(block_granularity, decode_tokens);
}

namespace {

std::int32_t groupReserveTokens(const CacheGroupConfig& group, const PrefillReserve& reserve) {
    if (group.IsSnapshotStateGroup()) {
        if (!reserve.reserve_snapshot_state_growth) {
            return 0;
        }
        return static_cast<std::int32_t>(
            SnapshotStateReserveTokens(group.block_granularity, reserve.decode_input_tokens));
    }
    if (group.retention == CacheGroupConfig::Retention::SlidingWindow) {
        return reserve.DecodeTokens();
    }
    return std::max(reserve.DecodeTokens(), reserve.prompt_headroom_tokens);
}

}  // namespace

void ReservePrefillDemands(std::span<GroupDemand> demands, std::span<const CacheGroupConfig> cache_groups,
                           const PrefillReserve& reserve) {
    _assert(demands.size() == cache_groups.size(), "demands/cache groups size mismatch");
    _assert(reserve.decode_input_tokens >= 0 && reserve.prompt_headroom_tokens >= 0,
            "prefill reserve inputs must be non-negative");
    for (std::size_t i = 0; i < demands.size(); ++i) {
        _assert(demands[i].reserve_tokens == 0, "a prefill demand's reserve is decided here and nowhere else");
        demands[i].reserve_tokens = groupReserveTokens(cache_groups[i], reserve);
    }
}

std::int32_t StateCheckpointMaterializationStart(std::int32_t before_tokens, std::int32_t after_tokens,
                                                 std::int32_t prefix_granularity) {
    _assert(before_tokens >= 0 && after_tokens > before_tokens, "state checkpoint extent must advance");
    _assert(prefix_granularity > 0, "prefix_granularity must be > 0");
    const std::int32_t completed_boundary = after_tokens - after_tokens % prefix_granularity;
    return completed_boundary > before_tokens ? completed_boundary : after_tokens;
}

void MakeSnapshotStatePrefillSparse(std::span<GroupDemand> demands, std::span<const CacheGroupConfig> cache_groups,
                                    const CacheCoordinator& coordinator, std::int32_t before_tokens,
                                    std::int32_t after_tokens) {
    _assert(demands.size() == cache_groups.size(), "demands/cache groups size mismatch");
    _assert(before_tokens >= 0 && after_tokens > before_tokens,
            "snapshot-state prefill requires a positive advancing extent");
    for (std::size_t i = 0; i < demands.size(); ++i) {
        if (!cache_groups[i].IsSnapshotStateGroup()) {
            continue;
        }
        const std::int32_t block_granularity = coordinator.GroupBlockGranularity(static_cast<std::int32_t>(i));
        demands[i].num_tokens = after_tokens;
        // A completing prefill may end off a prefix boundary. Materialize the
        // last completed checkpoint as well as the final continuation state:
        // the runtime writes both from this one model forward. Earlier slots
        // remain holes, preserving absolute block-table positions.
        const std::int32_t first_materialized_token =
            StateCheckpointMaterializationStart(before_tokens, after_tokens, coordinator.PrefixGranularity());
        demands[i].materialized_suffix_start = (first_materialized_token - 1) / block_granularity;
    }
}

}  // namespace tokenspeed
