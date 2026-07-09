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

struct SchedulerConfig;  // defined in scheduler/types.h; only used by-ref below

// Stream-ordering safety: all forwards share one execution stream, so reuse writes of freed/slid-out
// pages enqueue after in-flight KV kernels, and claimed pages stay ref>1 -- never rewritten from outside.
// TODO(flat-l2): out-of-stream writers (load-back H2D) must fence before joining the flat path.

// On false (pool short) nothing is acquired but the claimed prefix blocks REMAIN -- caller must FreeRequest.
// SWA transient footprint contract: each chunk is fully resident during its forward (intra-chunk reads),
// so per-request SWA peak = ceil((chunk+W-1)/P) pages, bounded by the scheduler round budget -- same
// plateau as vLLM. Pinned by FlatPrefillPlateauSuite; shrinking it further needs a kernel-level ring buffer.
bool PrefillFirstChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                       const CoordinatorMatch& hit, std::int32_t num_new_tokens);

// Appends the host-extension segment to each group's table: -1 slots push the
// null block (slot alignment), real slots Acquire one page. Returns
// (host_page, device_block) pairs, group-major: the LoadBack op wires BlockId(),
// the emission ledger pins the block itself for the in-flight H2D copy.
std::vector<std::pair<std::int32_t, CacheBlock*>> LoadHostExtension(KvCacheCoordinator& coordinator,
                                                                    std::vector<BlockTable>& tables,
                                                                    const HostMatch& host);

// Register prior chunks' pages, slide to num_computed_tokens, then acquire; false = pool
// short (registration and slide already ran, nothing allocated) -- same for the two ops below.
bool PrefillChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                  std::span<const std::string> content_hashes, std::int32_t num_tokens,
                  std::int32_t num_computed_tokens);

bool DecodeStep(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                std::span<const std::string> content_hashes, std::int32_t first_page_slot,
                std::int32_t num_tokens, std::int32_t num_computed_tokens);

bool FinalizePrefillAndReserveDecode(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                                     std::span<const std::string> content_hashes, std::int32_t reserve_tokens,
                                     std::int32_t num_computed_tokens);

// One KvCacheSpec per config paged_cache_group (group_id = index); all groups share config.page_size.
std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config);

void FreeRequest(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables);

// One row per config group_id: BlockId() per logical page, null-block holes written as 0 (no compaction).
std::map<std::string, std::vector<std::int32_t>> BuildFlatBlockTables(
    const std::vector<BlockTable>& tables, std::span<const std::string> group_ids);

}  // namespace tokenspeed
