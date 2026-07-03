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

bool PrefillFirstChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                       std::span<const std::string> content_hashes, std::int32_t num_tokens) {
    CoordinatorMatch hit = coordinator.MatchPrefix(content_hashes);
    coordinator.ClaimCommonPrefix(tables, hit);      // pure claim, never fails
    return coordinator.Acquire(tables, num_tokens);  // check-then-act
}

bool PrefillChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                  std::span<const std::string> content_hashes, std::int32_t num_tokens,
                  std::int32_t num_full_blocks) {
    if (!coordinator.Acquire(tables, num_tokens)) {
        return false;
    }
    coordinator.CacheFullBlocks(tables, content_hashes, num_full_blocks);
    return true;
}

bool DecodeStep(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                std::int32_t num_tokens, std::int32_t num_computed_tokens) {
    if (!coordinator.Acquire(tables, num_tokens)) {
        return false;
    }
    coordinator.AdvanceWindow(tables, num_computed_tokens);
    return true;
}

std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config) {
    std::vector<KvCacheSpec> specs;
    specs.reserve(config.paged_cache_groups.size());
    for (const PagedCacheGroupConfig& group : config.paged_cache_groups) {
        const bool is_swa = group.retention == PagedCacheGroupConfig::Retention::SlidingWindow;
        specs.push_back(KvCacheSpec{
            .kind = is_swa ? AttnKind::kSlidingWindow : AttnKind::kFull,
            .page_size = config.page_size,
            .sliding_window = is_swa ? group.sliding_window_tokens.value_or(0) : 0,
        });
    }
    return specs;
}

void FreeRequest(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables) {
    if (tables.empty()) {
        return;  // nothing was ever allocated, or a failure path already released the pages
    }
    coordinator.Free(tables);
}

std::map<std::string, std::vector<std::int32_t>> BuildFlatBlockTables(
    const std::vector<BlockTable>& tables, std::span<const std::string> group_ids) {
    _assert(tables.size() == group_ids.size(),
            "BuildFlatBlockTables: tables/group_ids size mismatch");
    std::map<std::string, std::vector<std::int32_t>> out;
    for (std::size_t i = 0; i < tables.size(); ++i) {
        out.emplace(group_ids[i], BlockTablePageIds(tables[i]));
    }
    return out;
}

}  // namespace tokenspeed
