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

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "cache/cache_types.h"
#include "cache/kv_cache_coordinator.h"

namespace tokenspeed {

struct SchedulerConfig;  // defined in scheduler/types.h; only used by-ref below

// Prefill first chunk: match the common cached prefix across groups, claim the
// hit blocks into each table, then allocate pages for this chunk's tokens.
// Returns false if the shared pool cannot supply the chunk. The chunk itself is
// check-then-act (Acquire allocates nothing on failure), but the prefix blocks
// claimed by ClaimCommonPrefix REMAIN in the tables -- on failure the caller
// must FreeRequest the tables to release those refs. tables must be sized to
// coordinator.NumGroups().
bool PrefillFirstChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                       std::span<const std::string> content_hashes, std::int32_t num_tokens);

// Subsequent prefill chunk: allocate this chunk's pages, then register the now-
// complete pages' content hashes so later requests can prefix-hit them.
// num_full_blocks = number of fully-filled logical pages after this chunk.
bool PrefillChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                  std::span<const std::string> content_hashes, std::int32_t num_tokens,
                  std::int32_t num_full_blocks);

// One decode step: allocate the new token's page need, then slide the SWA window
// forward (frees pages fully out of window in sliding-window groups).
bool DecodeStep(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                std::int32_t num_tokens, std::int32_t num_computed_tokens);

// Translate the Python-provided per-group cache config into KvCacheSpecs (one
// per paged_cache_group, group_id = index). All groups share config.page_size.
std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config);

// Finish / abort: return every page in every table to the pool. No-op on empty
// tables (nothing allocated yet, or already released by a failure path).
void FreeRequest(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables);

// flat per-group block table extraction. One row per group: the BlockId()
// sequence of table.Blocks(), with null-block holes written as 0, in absolute
// logical-page order (no compaction). key = group_id string, taken from
// config_.paged_cache_groups[i].group_id (e.g. "full"/"swa"), so the keys match
// the downstream / Python assertions rather than a bare index.
std::map<std::string, std::vector<std::int32_t>> BuildFlatBlockTables(
    const std::vector<BlockTable>& tables, std::span<const std::string> group_ids);

}  // namespace tokenspeed
