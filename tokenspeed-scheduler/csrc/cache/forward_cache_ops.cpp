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

#include "cache/forward_cache_ops.h"

#include "resource/allocator/paged_cache_group.h"
#include "scheduler/types.h"

namespace tokenspeed {

std::int32_t AlignFlatPrefillChunk(std::int32_t first_pos, std::int32_t unscheduled, std::int32_t token_budget,
                                   std::int32_t page_size, std::int32_t promotion_boundary_tokens) {
    _assert(first_pos >= 0 && unscheduled >= 0 && token_budget >= 0, "prefill positions must be non-negative");
    _assert(page_size > 0, "page_size must be > 0");
    std::int32_t chunk_size = std::min(unscheduled, token_budget);
    if (promotion_boundary_tokens > first_pos) {
        chunk_size = std::min(chunk_size, promotion_boundary_tokens - first_pos);
    }
    if (chunk_size == unscheduled) {
        return chunk_size;
    }

    const std::int32_t page_offset = first_pos % page_size;
    if (page_offset != 0) {
        const std::int32_t tokens_to_boundary = page_size - page_offset;
        return token_budget >= tokens_to_boundary ? tokens_to_boundary : 0;
    }
    return chunk_size - chunk_size % page_size;
}

std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config) {
    _assert(config.block_size > 0, "cache_block_tokens must be > 0");
    std::vector<KvCacheSpec> specs;
    specs.reserve(config.paged_cache_groups.size());
    for (const PagedCacheGroupConfig& group : config.paged_cache_groups) {
        _assert(group.cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        _assert(group.block_size == 0 || group.block_size == config.block_size,
                "per-group block_size cannot override the shared cache_block_tokens");
        // family=State marks trailing-window prefix reuse and covers both SWA and
        // linear-attention groups; only a State group WITHOUT SlidingWindow
        // retention is a mamba-style state group.
        if (group.family == PagedCacheGroupFamily::State &&
            group.retention != PagedCacheGroupConfig::Retention::SlidingWindow) {
            specs.push_back(KvCacheSpec{
                .kind = AttnKind::kMambaState,
                .sliding_window = 0,
                .cache_blocks_per_lcm_block = group.cache_blocks_per_lcm_block,
            });
            continue;
        }
        const bool is_swa = group.retention == PagedCacheGroupConfig::Retention::SlidingWindow;
        specs.push_back(KvCacheSpec{
            .kind = is_swa ? AttnKind::kSlidingWindow : AttnKind::kFull,
            .sliding_window = is_swa ? group.sliding_window_tokens.value_or(0) : 0,
            .cache_blocks_per_lcm_block = group.cache_blocks_per_lcm_block,
        });
    }
    return specs;
}

void FreeRequest(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables) {
    if (tables.empty()) {
        return;  // request never got tables, or a failure path already released them
    }
    coordinator.Free(tables);
}

std::map<std::string, std::vector<std::int32_t>> BuildFlatBlockTables(const KvCacheCoordinator& coordinator,
                                                                      const std::vector<BlockTable>& tables,
                                                                      std::span<const std::string> group_ids) {
    _assert(tables.size() == group_ids.size(), "BuildFlatBlockTables: tables/group_ids size mismatch");
    _assert(tables.size() == static_cast<std::size_t>(coordinator.NumGroups()),
            "BuildFlatBlockTables: tables/coordinator size mismatch");
    std::map<std::string, std::vector<std::int32_t>> out;
    for (std::size_t i = 0; i < tables.size(); ++i) {
        out.emplace(group_ids[i], coordinator.GroupManager(static_cast<std::int32_t>(i)).BlockTablePageIds(tables[i]));
    }
    return out;
}

}  // namespace tokenspeed
