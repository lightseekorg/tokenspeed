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
// fixpoint coverage every group can validly claim: full-attention matches are
// truncated to it (any prefix of a full match is valid), and sliding-window
// matches are computed BOUNDED to it, so each SWA group's trailing
// contiguous-run invariant holds at exactly that length (no null hole inside
// the last window). per_group[i] is group i's PrefixMatch at that length.
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

    // Pure pre-check: fresh pages the shared pool must supply for every group's
    // table to absorb num_tokens (sum of the per-group BlocksNeededFor math;
    // tail-page room is credited per group). Acquire's check-then-act gate and
    // the scheduler's flat admission check both call this, so the page math
    // lives in exactly one place.
    std::int32_t BlocksNeededFor(std::span<const BlockTable> tables, std::int32_t num_tokens) const;
    // Overload for a not-yet-allocated request (prefill first chunk): every
    // group starts from a fresh, empty table (no tail credit).
    std::int32_t BlocksNeededFor(std::int32_t num_tokens) const;

    // Would Acquire(tables, num_tokens) succeed right now? Same math as
    // Acquire's gate, no allocation, no mutation.
    bool CanAcquire(std::span<const BlockTable> tables, std::int32_t num_tokens) const {
        return BlocksNeededFor(tables, num_tokens) <= pool_.NumFreeBlocks();
    }
    // Fresh-request variant (see BlocksNeededFor overload above).
    bool CanAcquire(std::int32_t num_tokens) const { return BlocksNeededFor(num_tokens) <= pool_.NumFreeBlocks(); }

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
