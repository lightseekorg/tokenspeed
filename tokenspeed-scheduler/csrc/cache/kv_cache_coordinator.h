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
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_block_ref.h"
#include "cache/cache_group.h"
#include "cache/cache_types.h"

namespace tokenspeed {

struct KvCacheCoordinatorTestAccess;

// num_common_tokens is in tokens at the one shared CacheBlock granularity P.
// per_group[i] is group i's PrefixMatch at exactly that length.
struct CoordinatorMatch {
    std::int32_t num_common_tokens{0};
    std::vector<PrefixMatch> per_group;
};

// Multi-group fan-out over the per-attention managers, one shared BlockPool. Holds no per-request
// state; the only cross-round mutable state is the streaming-sink mailbox, drained every round.
class KvCacheCoordinator {
public:
    // The host tier is fixed at construction: bound, CacheFullBlocks feeds the sink mailbox.
    KvCacheCoordinator(std::vector<CacheGroup> groups, std::int32_t cache_block_tokens, BlockPool& pool,
                       BlockPool* host_pool = nullptr);

    std::int32_t NumGroups() const { return static_cast<std::int32_t>(groups_.size()); }

    std::int32_t CacheBlockTokens() const noexcept { return cache_block_tokens_; }

    KvCacheManager& GroupManager(std::int32_t i) { return groups_[static_cast<std::size_t>(i)].Manager(); }
    const KvCacheManager& GroupManager(std::int32_t i) const { return groups_[static_cast<std::size_t>(i)].Manager(); }

    struct PrefixProbe {
        struct Tier {
            std::int32_t num_common_tokens{0};
            std::vector<tokenspeed::PrefixProbe> per_group;
        };

        std::vector<std::vector<CacheKey>> group_keys;
        Tier device;
        Tier host;
    };
    struct AdmissionPlan {
        struct Group {
            // Borrows GroupDemand::table and content_hashes from the caller.
            // The plan must be acquired before either backing object changes.
            GroupDemand demand;
            std::vector<CacheBlockLocation> placements;
            std::int32_t host_placement_count{0};
        };

        PrefixProbe prefix;
        std::vector<Group> per_group;
        std::vector<std::pair<GroupId, CacheBlockLocation>> victims;
    };
    struct AdmissionResult {
        std::int32_t device_prefix_tokens{0};
        std::int32_t host_prefix_tokens{0};
        std::vector<BlockTransfer> load_pairs;
    };

    // ProbePrefix and ProbeAdmission are read-only. Cache ownership, LRU and
    // placement change only in Acquire after every independent gate succeeds.
    PrefixProbe ProbePrefix(std::span<const std::string> content_hashes) const;
    std::optional<AdmissionPlan> ProbeAdmission(PrefixProbe&& prefix, std::span<const GroupDemand> demands) const;
    AdmissionResult Acquire(AdmissionPlan&& plan);

    // Single home of the gate-side page math.
    std::int32_t BlocksNeededFor(std::span<const BlockTable> tables, std::int32_t num_tokens) const;
    // Fresh-table overload for a not-yet-allocated request (no tail credit).
    std::int32_t BlocksNeededFor(std::int32_t num_tokens) const;
    std::int32_t NumAvailableLcmBlocks() const;

    // end_tokens = the chunk's end position (-1 = unknown/legacy): aligned-final-page-only
    // groups register nothing without it, since only an aligned chunk end holds a real snapshot.
    // registered slots are skipped per slot, so repeated coverage is idempotent.
    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::int32_t first_slot = 0, std::int32_t end_tokens = -1);
    void ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens);
    void ConsumeAvailable(std::span<BlockTable> tables, std::int32_t num_tokens);
    void Free(std::span<BlockTable> tables);

    struct StoreCandidate {
        CacheKey key;
        CacheBlockRef block;  // pinned until WriteBackDone or a drain-time drop releases the ref
    };
    std::vector<StoreCandidate> TakePendingStores() { return std::exchange(pending_stores_, {}); }
    // Collection/pinning follows host-tier presence, so the slide credit flips count_uncached on this.
    bool HasHostTier() const { return host_pool_ != nullptr; }
    bool ContainsHostCachedBlock(const CacheKey& key) const;
    bool IsHostCachedBlock(CacheBlockLocation location) const;
    std::int32_t NumHostCachedBlocks() const;
    std::int32_t NumPinnedHostCachedBlocks() const;
    void CacheHostBlock(CacheBlockRef& block, const CacheKey& key);

private:
    friend struct KvCacheCoordinatorTestAccess;

    struct AcquiredPrefix {
        CoordinatorMatch device;
        CoordinatorMatch host;
    };

    std::vector<CacheKey> keysForGroup(std::span<const std::string> content_hashes, GroupId group_id) const;
    std::vector<std::vector<CacheKey>> buildGroupKeys(std::span<const std::string> content_hashes) const;
    PrefixProbe::Tier probeTierWithKeys(const BlockPool& pool, std::span<const std::vector<CacheKey>> group_keys,
                                        std::int32_t num_cache_blocks, std::int32_t floor_tokens) const;
    CoordinatorMatch acquireTierWithKeys(BlockPool& pool, std::span<const std::vector<CacheKey>> group_keys,
                                         std::int32_t floor_tokens, PrefixProbe::Tier&& probe);
    AcquiredPrefix acquirePrefix(PrefixProbe&& probe);
    void cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table,
                                 std::span<const std::string> content_hashes, std::int32_t first_slot,
                                 std::int32_t end_tokens);
    std::vector<CacheGroup> groups_;
    // Closed groups first, so non-closed groups match against a settled bound.
    std::vector<std::size_t> match_order_;
    BlockPool& pool_;
    BlockPool* host_pool_{nullptr};
    std::int32_t cache_block_tokens_{0};
    std::uint64_t next_recency_{0};
    std::vector<StoreCandidate> pending_stores_;
};

// One CacheGroup per spec (group_id = index), all sharing cache_block_tokens.
KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, std::int32_t cache_block_tokens, BlockPool& pool,
                                   BlockPool* host_pool = nullptr);

}  // namespace tokenspeed
