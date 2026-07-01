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
#include <span>
#include <string>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_group.h"
#include "cache/cache_types.h"

namespace tokenspeed {

// Common-prefix match across all groups. num_common_blocks is the page-aligned
// min coverage; per_group[i] is group i's PrefixMatch truncated to that length.
struct CoordinatorMatch {
    std::int32_t num_common_blocks{0};
    std::vector<PrefixMatch> per_group;
};

// Stateless multi-group fan-out over the per-attention managers. Holds the
// groups (config + managers) and a reference to the shared BlockPool (for the
// cross-group capacity check). Per-request BlockTables are passed in by the
// caller, aligned by group index. Consumes one content-hash stream (64-hex, no
// group_id) and wraps it per group internally.
//
// Per-request flow (driven by FSM transition events):
//   prefill first chunk:  MatchPrefix -> ClaimCommonPrefix -> Acquire(remainder)
//   later prefill/decode: Acquire(new tokens)
class KvCacheCoordinator {
public:
    KvCacheCoordinator(std::vector<CacheGroup> groups, BlockPool& pool)
        : groups_{std::move(groups)}, pool_{pool} {}

    std::int32_t NumGroups() const { return static_cast<std::int32_t>(groups_.size()); }

    CoordinatorMatch MatchPrefix(std::span<const std::string> content_hashes) const;

    // Claim the common-prefix hit blocks into each group's table. Pure claim, no
    // allocation, never fails (ClaimHitBlocks skips null holes). Call on fresh
    // tables before Acquire, at prefill start.
    void ClaimCommonPrefix(std::span<BlockTable> tables, const CoordinatorMatch& hit);

    // Token-driven incremental allocation across all groups. Check-then-act: sums
    // the pages every group needs, and if the shared pool cannot supply them all,
    // allocates NOTHING and returns false (no partial/unaligned state ever exists,
    // so no rollback is needed). Otherwise allocates in every group and returns
    // true. Used for both the prefill remainder and each decode step.
    bool Acquire(std::span<BlockTable> tables, std::int32_t num_tokens);

    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::int32_t num_full_blocks);
    void Free(std::span<BlockTable> tables);

    // Fan out window-eviction to sliding-window groups only. Full-attention
    // groups never evict mid-sequence, so they're skipped via the spec.kind
    // gate. tables are aligned by group index (size must equal NumGroups()).
    // Mirrors Acquire's per-group fan-out shape.
    void AdvanceWindow(std::span<BlockTable> tables, std::int32_t num_computed_tokens);

private:
    std::vector<std::string> keysForGroup(std::span<const std::string> content_hashes,
                                          std::uint32_t group_id) const;
    std::vector<CacheGroup> groups_;
    BlockPool& pool_;
};

// Factory: build one CacheGroup per spec (group_id = index), all sharing `pool`.
// Asserts every spec has the same page_size (single shared page geometry).
KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, BlockPool& pool);

}  // namespace tokenspeed
