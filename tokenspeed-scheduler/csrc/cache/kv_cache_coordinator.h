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
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/block_ref.h"
#include "cache/cache_group.h"
#include "cache/cache_types.h"

namespace tokenspeed {

// num_common_tokens is the coverage every group can validly claim, measured in TOKENS -- the
// cross-group unit is tokens because per-group page sizes may differ; per_group[i] is group i's
// PrefixMatch at exactly that length, in group i's own blocks. Default-constructed = the
// canonical zero hit (empty per_group, zero tokens); HostMatch mirrors this convention.
struct CoordinatorMatch {
    std::int32_t num_common_tokens{0};
    std::vector<PrefixMatch> per_group;
};

// Host-tier extension of a device prefix match, produced by MatchHostExtension.
// Move-only: `pinned` holds one BlockRef per real matched host page, so pin
// lifetime rides the match result (drop the match, drop the pins).
struct HostMatch {
    // per_group[g][j]: host block backing extension block j for group g, or nullptr
    // (SWA hole -- slots the window has already passed at the final boundary).
    std::vector<std::vector<CacheBlock*>> per_group;
    std::vector<BlockRef> pinned;
    std::int32_t num_extension_blocks{0};
};

// Multi-group fan-out over the per-attention managers, all sharing one BlockPool; the caller owns the
// per-request BlockTables (index-aligned to groups). Holds no per-request state; its only cross-round
// mutable state is the streaming-sink mailbox (pending_stores_), which the scheduler must drain every round.
class KvCacheCoordinator {
public:
    KvCacheCoordinator(std::vector<CacheGroup> groups, BlockPool& pool);

    std::int32_t NumGroups() const { return static_cast<std::int32_t>(groups_.size()); }

    // Per-group manager access for lifecycle calls the coordinator no longer wraps (e.g. ReclaimExpired).
    KvCacheManager& GroupManager(std::int32_t i) { return groups_[static_cast<std::size_t>(i)].Manager(); }
    const KvCacheManager& GroupManager(std::int32_t i) const {
        return groups_[static_cast<std::size_t>(i)].Manager();
    }

    CoordinatorMatch MatchPrefix(std::span<const std::string> content_hashes) const;

    // Host-tier twin of MatchPrefix over the extension [device_common_blocks, hashes.size()):
    // same sweep-then-converge semantics; every matched page is pinned into HostMatch::pinned.
    HostMatch MatchHostExtension(std::span<const std::string> content_hashes,
                                 std::int32_t device_common_blocks, BlockPool& host_pool) const;

    // Admission entry: both matches over ONE key-wrapping pass (the two calls above each
    // rebuild the per-group keys; back-to-back admission use goes through here).
    struct AdmissionMatch {
        CoordinatorMatch device;
        HostMatch host;
    };
    AdmissionMatch MatchPrefixWithHostExtension(std::span<const std::string> content_hashes,
                                                BlockPool& host_pool) const;

    // Pure claim into fresh tables, never fails; a non-empty per_group must be sized to the group count.
    void ClaimCommonPrefix(std::span<BlockTable> tables, const CoordinatorMatch& hit);

    // Per-group AppendHostExtension fan-out; the contract lives on the forward_cache_ops facade.
    std::vector<std::pair<std::int32_t, CacheBlock*>> LoadHostExtension(std::span<BlockTable> tables,
                                                                        const HostMatch& host);

    // Free-list blocks the claim will consume (TouchBlock pulls ref-0 cached hits); gates charge these too.
    std::int32_t BlocksConsumedByClaim(const CoordinatorMatch& hit) const;

    // All-or-nothing across all groups: on shortfall allocates NOTHING and returns false (no rollback needed).
    bool Acquire(std::span<BlockTable> tables, std::int32_t num_tokens);

    // Single home of the gate-side page math; Acquire's check and the flat admission gates both build on it.
    std::int32_t BlocksNeededFor(std::span<const BlockTable> tables, std::int32_t num_tokens) const;
    // Fresh-table overload for a not-yet-allocated request (no tail credit).
    std::int32_t BlocksNeededFor(std::int32_t num_tokens) const;

    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::int32_t first_slot = 0);
    void Free(std::span<BlockTable> tables);

    struct StoreCandidate {
        std::string key;  // group-wrapped (MakeKeyWithGroupId), the host-tier index key
        BlockRef block;   // pinned (Share) until WriteBackDone or a drain-time drop releases the ref
    };
    void EnableStoreCandidateCollection() { collect_store_candidates_ = true; }
    std::vector<StoreCandidate> TakePendingStores() { return std::exchange(pending_stores_, {}); }
    // Gate-side slide credit flips count_uncached on this flag; exposed for the callers' per-group loops.
    bool CollectingStoreCandidates() const { return collect_store_candidates_; }

private:
    std::vector<std::string> keysForGroup(std::span<const std::string> content_hashes,
                                          std::uint32_t group_id) const;
    std::vector<std::vector<std::string>> buildGroupKeys(std::span<const std::string> content_hashes) const;
    CoordinatorMatch matchPrefixWithKeys(std::span<const std::vector<std::string>> group_keys,
                                         std::int32_t total_blocks) const;
    HostMatch matchHostExtensionWithKeys(std::span<const std::vector<std::string>> group_keys,
                                         std::int32_t total_blocks, std::int32_t device_common_blocks,
                                         BlockPool& host_pool) const;
    std::vector<CacheGroup> groups_;
    // Closed groups first: they truncate safely, so non-closed groups match against a settled
    // bound (vLLM's full-attention-first).
    std::vector<std::size_t> match_order_;
    BlockPool& pool_;
    bool collect_store_candidates_ = false;
    std::vector<StoreCandidate> pending_stores_;
};

// One CacheGroup per spec (group_id = index); asserts every spec shares the same page_size.
KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, BlockPool& pool);

}  // namespace tokenspeed
