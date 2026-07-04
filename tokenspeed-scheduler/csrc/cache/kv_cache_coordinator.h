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
    // tables before Acquire, at prefill start. A default-constructed
    // CoordinatorMatch (empty per_group, num_common_blocks == 0) is the
    // canonical zero hit and claims nothing; any non-empty per_group must be
    // sized to the group count.
    void ClaimCommonPrefix(std::span<BlockTable> tables, const CoordinatorMatch& hit);

    // Pure query, gate-side twin of ClaimCommonPrefix (same pattern as
    // AdvanceWindow / BlocksFreedByAdvance): free-list blocks the claim will
    // consume. ClaimCommonPrefix TouchBlock()s every real hit block, and
    // touching a ref_cnt==0 cached block REMOVES it from the free list
    // (block_pool.h), so a claim shrinks NumFreeBlocks() by exactly the number
    // of ref-0 hit blocks -- on top of anything Acquire takes. Admission gates
    // must charge this or the free count they check overstates what the
    // transition's Acquire will find. Hit blocks with ref > 0 (still held by a
    // live request) are not in the free list and cost nothing; null holes are
    // TouchBlock no-ops. No block is ever counted twice across groups: a block
    // carries at most one hash (CacheBlock::SetHash asserts) and hash keys are
    // group-scoped (MakeKeyWithGroupId), so a physical block can appear in at
    // most one group's match.
    std::int32_t BlocksConsumedByClaim(const CoordinatorMatch& hit) const;

    // Token-driven incremental allocation across all groups. Check-then-act: sums
    // the pages every group needs, and if the shared pool cannot supply them all,
    // allocates NOTHING and returns false (no partial/unaligned state ever exists,
    // so no rollback is needed). Otherwise allocates in every group and returns
    // true. Used for both the prefill remainder and each decode step.
    bool Acquire(std::span<BlockTable> tables, std::int32_t num_tokens);

    // Pure pre-check: fresh pages the shared pool must supply for every group's
    // table to absorb num_tokens (sum of the per-group BlocksNeededFor math;
    // tail-page room is credited per group). Acquire's check-then-act gate and
    // the scheduler's flat admission gates both build on this, so the page math
    // lives in exactly one place. (No raw would-Acquire-succeed helper is
    // exposed: the scheduler's gates must also account outstanding decode
    // reservations and the pending SWA slide, so they compose this with
    // BlocksFreedByAdvance and their reservation ledger instead.)
    std::int32_t BlocksNeededFor(std::span<const BlockTable> tables, std::int32_t num_tokens) const;
    // Overload for a not-yet-allocated request (prefill first chunk): every
    // group starts from a fresh, empty table (no tail credit).
    std::int32_t BlocksNeededFor(std::int32_t num_tokens) const;

    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::int32_t num_full_blocks);
    void Free(std::span<BlockTable> tables);

    // Fan out window-eviction to every group, mirroring Acquire's per-group
    // fan-out shape. Managers without a retention window (FullAttnManager)
    // inherit KvCacheManager's no-op default, so only window-evicting managers
    // do work. tables are aligned by group index (size must equal NumGroups()).
    void AdvanceWindow(std::span<BlockTable> tables, std::int32_t num_computed_tokens);

    // Pure query, fan-out twin of AdvanceWindow: pages a pending
    // AdvanceWindow(tables, num_computed_tokens) would return to the shared
    // pool, summed over all groups (0 for full-history groups). Lets admission
    // gates credit the slide DecodeStep performs before its Acquire without a
    // second copy of the window math (each manager mirrors its own AdvanceWindow).
    std::int32_t BlocksFreedByAdvance(std::span<const BlockTable> tables, std::int32_t num_computed_tokens) const;

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
