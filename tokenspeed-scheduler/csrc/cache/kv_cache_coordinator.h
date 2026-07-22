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

// num_common_tokens is in TOKENS -- the cross-group unit, since per-group block sizes may
// differ; per_group[i] is group i's PrefixMatch at exactly that length. Default = the zero hit.
struct CoordinatorMatch {
    std::int32_t num_common_tokens{0};
    std::vector<PrefixMatch> per_group;
};

// Multi-group fan-out over the per-attention managers, one shared BlockPool. Holds no per-request
// state; the only cross-round mutable state is the streaming-sink mailbox, drained every round.
class KvCacheCoordinator {
public:
    // The host tier is fixed at construction: bound, CacheFullBlocks feeds the sink mailbox.
    KvCacheCoordinator(std::vector<CacheGroup> groups, BlockPool& pool, BlockPool* host_pool = nullptr,
                       std::int32_t base_block_size = 0, std::int32_t lcm_block_size = 0);

    std::int32_t NumGroups() const { return static_cast<std::int32_t>(groups_.size()); }

    // GCD/LCM of every group's block_size: the granularity keys fold at, and the one they align to.
    std::int32_t BaseBlockSize() const { return base_block_size_; }
    std::int32_t LcmBlockSize() const { return lcm_block_size_; }

    KvCacheManager& GroupManager(std::int32_t i) { return groups_[static_cast<std::size_t>(i)].Manager(); }
    const KvCacheManager& GroupManager(std::int32_t i) const { return groups_[static_cast<std::size_t>(i)].Manager(); }

    // The admission entry: the device match, plus the host-tier match above the device boundary when a
    // host tier is bound. Every real hit is already pinned in the returned match.
    struct AdmissionMatch {
        CoordinatorMatch device;
        CoordinatorMatch host;
    };
    struct CoordinatorProbe {
        std::int32_t num_common_tokens{0};
        std::int32_t num_free_hit_blocks{0};
        std::vector<PrefixProbe> per_group;
    };
    struct AdmissionProbe {
        CoordinatorProbe device;
        CoordinatorProbe host;
    };
    // Split probing from ownership acquisition so admission checks can defer
    // without refreshing free cached blocks in the eviction order.
    AdmissionProbe ProbePrefix(std::span<const std::string> content_hashes) const;
    AdmissionMatch AcquirePrefix(std::span<const std::string> content_hashes, AdmissionProbe&& probe);
    AdmissionMatch MatchPrefix(std::span<const std::string> content_hashes);

    // Pure claim into fresh tables, never fails; a non-empty per_group must be sized to the group count.
    void ClaimCommonPrefix(std::span<BlockTable> tables, CoordinatorMatch&& hit);

    // Contract on the forward_cache_ops facade.
    std::vector<BlockTransfer> LoadHostExtension(std::span<BlockTable> tables, CoordinatorMatch&& host);

    // All-or-nothing across all groups: on shortfall allocates NOTHING and returns false (no rollback needed).
    bool Acquire(std::span<BlockTable> tables, std::int32_t num_tokens);

    // Single home of the gate-side page math; Acquire's check and the flat admission gates both build on it.
    std::int32_t BlocksNeededFor(std::span<const BlockTable> tables, std::int32_t num_tokens) const;
    // Fresh-table overload for a not-yet-allocated request (no tail credit).
    std::int32_t BlocksNeededFor(std::int32_t num_tokens) const;

    // end_tokens = the chunk's end position (-1 = unknown/legacy): aligned-final-page-only
    // groups register nothing without it, since only an aligned chunk end holds a real snapshot.
    // first_slot may reach back to the fold grid (decode re-covers so coarse groups can fold);
    // registered slots are skipped per slot, so the overlap is idempotent.
    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::int32_t first_slot = 0, std::int32_t end_tokens = -1);
    void ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens);
    void Free(std::span<BlockTable> tables);

    struct StoreCandidate {
        std::string key;  // group-wrapped (MakeKeyWithGroupId), the host-tier index key
        BlockRef block;   // pinned until WriteBackDone or a drain-time drop releases the ref
    };
    std::vector<StoreCandidate> TakePendingStores() { return std::exchange(pending_stores_, {}); }
    // Collection/pinning follows host-tier presence, so the slide credit flips count_uncached on this.
    bool HasHostTier() const { return host_pool_ != nullptr; }

private:
    // Base-granularity content_hashes -> the group's coarse-block lookup keys: fold m = the
    // group's block_size / base base pages into one coarse block, then wrap with group_id.
    // first_base is content_hashes[0]'s global base-page index, so a chunk starting mid-grid
    // drops its leading remainder (see MakeFoldedGroupKeys).
    std::vector<std::string> keysForGroup(std::span<const std::string> content_hashes, std::uint32_t group_id,
                                          std::int32_t group_block_size, std::int32_t first_base = 0) const;
    std::vector<std::vector<std::string>> buildGroupKeys(std::span<const std::string> content_hashes) const;
    CoordinatorProbe probeTierWithKeys(const BlockPool& pool, std::span<const std::vector<std::string>> group_keys,
                                       std::int32_t num_base_pages, std::int32_t floor_tokens) const;
    CoordinatorMatch acquireTierWithKeys(BlockPool& pool, std::span<const std::vector<std::string>> group_keys,
                                         std::int32_t floor_tokens, CoordinatorProbe&& probe) const;
    std::vector<CacheGroup> groups_;
    // Closed groups first, so non-closed groups match against a settled bound.
    std::vector<std::size_t> match_order_;
    BlockPool& pool_;
    BlockPool* host_pool_{nullptr};
    std::int32_t base_block_size_{0};
    std::int32_t lcm_block_size_{0};
    std::vector<StoreCandidate> pending_stores_;
};

// One CacheGroup per spec (group_id = index); carries GCD/LCM of the per-group block_sizes.
KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, BlockPool& pool, BlockPool* host_pool = nullptr);

}  // namespace tokenspeed
