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

#include <map>
#include <span>
#include <string>
#include <vector>

#include "cache/core/cache_types.h"
#include "cache/coordinator/cache_coordinator.h"

namespace tokenspeed {

struct SchedulerConfig;

// One CacheGroupSpec per config cache_group (group_id = index); all groups share config.prefix_granularity.
// Pure translation: the caller must have accepted `config` through
// SchedulerConfig::Validate() first, which is what makes every field read here
// (packing, block granularity, a sliding group's window) well-formed.
std::vector<CacheGroupSpec> MakeSpecsFromConfig(const SchedulerConfig& config);

// Token count for the next chunk starting at first_pos, bounded by unscheduled
// tokens, token_budget and a pending promotion boundary (0 means none).
// Incomplete chunks end on a prefix boundary; a final extent stays whole even
// when it ends off-boundary. Returns 0 when no legal chunk fits the budget.
std::int32_t AlignPrefillChunk(std::int32_t first_pos, std::int32_t unscheduled, std::int32_t token_budget,
                               std::int32_t prefix_granularity, std::int32_t promotion_boundary_tokens);

// Absolute token boundary of the first output state in (before_tokens,
// after_tokens]: the latest prefix boundary crossed, or after_tokens if none.
// This is not a block-table slot; callers convert it with (boundary - 1) / the
// group's block_granularity. An internal checkpoint and the endpoint are both
// materialized in the same model forward; an aligned endpoint needs one output.
std::int32_t StateCheckpointMaterializationStart(std::int32_t before_tokens, std::int32_t after_tokens,
                                                 std::int32_t prefix_granularity);

// Reserve in tokens beyond a snapshot-state endpoint: at least one group block
// or the full decode width, whichever is larger. It is additional to the
// materialized suffix. Admission and its startup bound share this rule;
// callers decide whether the role and round need any reserve at all.
std::int64_t SnapshotStateReserveTokens(std::int64_t block_granularity, std::int64_t decode_tokens);

void FreeRequest(CacheCoordinator& coordinator, std::vector<BlockTable>& tables);

// One row per config group_id. Each group allocator resolves the LCM placement
// to the kernel-visible page id.
std::map<std::string, std::vector<std::int32_t>> BuildBlockTables(const CacheCoordinator& coordinator,
                                                                  const std::vector<BlockTable>& tables,
                                                                  std::span<const std::string> group_ids);

}  // namespace tokenspeed
