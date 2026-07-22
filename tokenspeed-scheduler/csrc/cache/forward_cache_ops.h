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
#include <utility>
#include <vector>

#include "cache/cache_types.h"
#include "cache/kv_cache_coordinator.h"

namespace tokenspeed {

struct SchedulerConfig;

// Stream-ordering safety: all forwards share one execution stream, so reuse writes of freed/slid-out
// pages enqueue after in-flight KV kernels, and claimed pages stay ref>1 -- never rewritten from outside.
// The one out-of-stream writer (load-back H2D) fences before joining: per-layer load events gate the
// attention reads and loadback is eager-only.

// On false (pool short) nothing is acquired but the claimed prefix blocks REMAIN -- caller must FreeRequest.
// SWA peak = ceil((chunk+W-1)/P) pages (chunk fully resident during its forward; pinned by
// FlatPrefillPlateauSuite -- shrinking it needs a kernel-level ring buffer).
bool PrefillFirstChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables, CoordinatorMatch&& hit,
                       std::int32_t num_new_tokens);

// Appends the host extension to each group's table (null-block slots -> device null, real slots
// Acquire one page); returns already-pinned pairs group-major.
std::vector<BlockTransfer> LoadHostExtension(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                                             CoordinatorMatch&& host);

// Register prior chunks' pages, slide to num_computed_tokens, then acquire; false = pool
// short (registration and slide already ran, nothing allocated) -- same for the two ops below.
bool PrefillChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                  std::span<const std::string> content_hashes, std::int32_t num_tokens,
                  std::int32_t num_computed_tokens);

bool DecodeStep(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                std::span<const std::string> content_hashes, std::int32_t first_page_slot, std::int32_t num_tokens,
                std::int32_t num_computed_tokens);

bool FinalizePrefillAndReserveDecode(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                                     std::span<const std::string> content_hashes, std::int32_t reserve_tokens,
                                     std::int32_t num_computed_tokens);

// One KvCacheSpec per config paged_cache_group (group_id = index); all groups share config.block_size.
std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config);

void FreeRequest(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables);

// One row per config group_id (page encoding: BlockTablePageIds).
std::map<std::string, std::vector<std::int32_t>> BuildFlatBlockTables(const std::vector<BlockTable>& tables,
                                                                      std::span<const std::string> group_ids);

}  // namespace tokenspeed
