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

std::vector<GroupDemand> MakeGroupDemands(std::vector<BlockTable>& tables,
                                          std::span<const CacheGroupConfig> cache_groups, GroupDemand prototype,
                                          StateStep state_step) {
    _assert(tables.size() == cache_groups.size(), "tables/cache groups size mismatch");
    std::vector<GroupDemand> demands;
    demands.reserve(tables.size());
    for (std::size_t i = 0; i < tables.size(); ++i) {
        GroupDemand demand = prototype;
        demand.table = &tables[i];
        if (cache_groups[i].attn_kind == AttnKind::kState) {
            demand.extent = state_step;
            demand.reserve_tokens = 0;
        }
        demands.push_back(demand);
    }
    return demands;
}

namespace {

std::int32_t groupReserveTokens(const CacheGroupConfig& group, const PrefillReserve& reserve) {
    if (group.attn_kind == AttnKind::kState) {
        return 0;
    }
    if (group.attn_kind == AttnKind::kSlidingWindow) {
        // Prepay the growth recycling cannot fund, at most the lookback a token
        // reads (window - 1). A replayable group holds no shared hit page, and
        // once the prompt's own tokens after the hit reach the lookback plus
        // one block, every hit page slides out before the first decode.
        const std::int32_t lookback = std::max(group.sliding_window_tokens.value_or(0) - 1, 0);
        const bool hit_counts =
            !group.replayable && reserve.swa_prompt_after_hit_tokens < lookback + group.block_granularity;
        const std::int32_t unrecycled =
            std::min(std::max(hit_counts ? reserve.swa_hit_tokens : 0, reserve.swa_later_chunk_tokens), lookback);
        return std::max(reserve.DecodeTokens(), std::min(reserve.prompt_headroom_tokens, unrecycled));
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

}  // namespace tokenspeed
