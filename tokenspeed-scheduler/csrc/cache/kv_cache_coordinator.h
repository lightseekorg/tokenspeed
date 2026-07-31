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
// state; the request access clock is global, while each request carries its issued epoch.
class KvCacheCoordinator {
public:
    // The host tier is fixed at construction: bound, CacheFullBlocks feeds the sink mailbox.
    KvCacheCoordinator(std::vector<CacheGroup> groups, std::int32_t cache_block_tokens, BlockPool& pool,
                       BlockPool* host_pool = nullptr);

    std::int32_t NumGroups() const { return static_cast<std::int32_t>(groups_.size()); }

    std::int32_t CacheBlockTokens() const noexcept { return cache_block_tokens_; }
    bool HasMambaStateGroup() const;

    KvCacheManager& GroupManager(std::int32_t i) { return groups_[static_cast<std::size_t>(i)].Manager(); }
    const KvCacheManager& GroupManager(std::int32_t i) const { return groups_[static_cast<std::size_t>(i)].Manager(); }

    struct PrefixProbe {
        struct Tier {
            std::int32_t num_common_tokens{0};
            // Common coverage after all prefix-closed groups, before a
            // window/state group shortens the resumable boundary.
            std::int32_t prefix_closed_tokens{0};
            std::vector<GroupPrefixProbe> per_group;
        };

        std::vector<std::vector<CacheKey>> group_keys;
        Tier device;
        Tier host;
    };
    struct AdmissionResult {
        std::int32_t device_prefix_tokens{0};
        std::int32_t host_prefix_tokens{0};
        // Device-only boundary worth materializing for non-closed groups.
        std::int32_t promotion_boundary_tokens{0};
        std::uint64_t access_epoch{0};
        std::vector<BlockTransfer> load_pairs;
        // Fresh device child pages appended by ordinary Acquire, aligned by
        // GroupId. Cache hits and host-loaded destinations are excluded.
        std::vector<std::vector<std::int32_t>> new_page_ids;
    };

    // ProbePrefix is read-only. Flat cache state must not change before its
    // result is passed to Admit. Admit consumes the probe even when admission
    // fails. It returns nullopt before committing when capacity is unavailable.
    // A missing epoch starts a new request; a supplied epoch continues that
    // request. Once commit starts, an internal plan/pool mismatch is fatal
    // because partial commit is not rolled back.
    PrefixProbe ProbePrefix(std::span<const std::string> content_hashes) const;
    // Decode-side PD reuses local history pages, while final-state groups are
    // restored from the remote endpoint snapshot. Their aligned null holes do
    // not count as cache hits.
    PrefixProbe ProbeDecodeDestinationPrefix(std::span<const std::string> content_hashes) const;
    std::optional<AdmissionResult> Admit(PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                         std::optional<std::uint64_t> request_access_epoch = std::nullopt);

    std::int32_t NumAvailableLcmBlocks() const;

    // Registers an exact range, used for transferred prefix blocks and tests.
    // Runtime publication during Admit follows each manager's boundary contract.
    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::uint64_t access_epoch, std::int32_t first_slot = 0,
                         CacheBoundaryKind boundary_kind = CacheBoundaryKind::kChunk);
    void ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens);
    void ConsumeAvailable(std::span<BlockTable> tables, std::int32_t num_tokens);
    void Free(std::span<BlockTable> tables);

    struct StoreCandidate {
        CacheKey key;
        CacheBlockRef block_ref;  // pinned until WriteBackDone or a drain-time drop releases the ref
    };
    std::vector<StoreCandidate> TakePendingStores() { return std::exchange(pending_stores_, {}); }
    // Collection/pinning follows host-tier presence, so the slide credit flips count_uncached on this.
    bool HasHostTier() const { return host_pool_ != nullptr; }
    bool ContainsHostCachedBlock(const CacheKey& key) const;
    bool IsHostCachedBlock(CacheBlockLocation location) const;
    std::int32_t NumHostCachedBlocks() const;
    std::int32_t NumPinnedHostCachedBlocks() const;
    void CacheHostBlock(CacheBlockRef& block_ref, const CacheKey& key);

private:
    friend struct KvCacheCoordinatorTestAccess;

    struct AcquiredPrefix {
        CoordinatorMatch device;
        CoordinatorMatch host;
    };

    std::vector<CacheKey> keysForGroup(std::span<const std::string> content_hashes, GroupId group_id) const;
    std::vector<std::vector<CacheKey>> buildGroupKeys(std::span<const std::string> content_hashes) const;
    PrefixProbe::Tier probeTierWithKeys(const BlockPool& pool, std::span<const std::vector<CacheKey>> group_keys,
                                        std::span<const std::size_t> match_order, std::int32_t num_cache_blocks,
                                        std::int32_t floor_tokens) const;
    CoordinatorMatch acquireTierWithKeys(BlockPool& pool, std::span<const std::vector<CacheKey>> group_keys,
                                         std::int32_t floor_tokens, PrefixProbe::Tier&& probe,
                                         std::uint64_t access_epoch);
    AcquiredPrefix acquirePrefix(PrefixProbe&& probe, std::uint64_t access_epoch);
    void cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table,
                                 std::span<const std::string> content_hashes, std::int32_t first_slot,
                                 std::uint64_t access_epoch, CacheBoundaryKind boundary_kind);
    void cacheCompletedBlocksForGroup(std::size_t group_index, const GroupDemand& demand, std::uint64_t access_epoch);
    std::vector<CacheGroup> groups_;
    // Closed groups first, so non-closed groups match against a settled bound.
    std::vector<std::size_t> match_order_;
    BlockPool& pool_;
    BlockPool* host_pool_{nullptr};
    std::int32_t cache_block_tokens_{0};
    std::uint64_t next_access_epoch_{0};
    std::vector<StoreCandidate> pending_stores_;
};

// One CacheGroup per spec (group_id = index), all sharing cache_block_tokens.
KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, std::int32_t cache_block_tokens, BlockPool& pool,
                                   BlockPool* host_pool = nullptr);

}  // namespace tokenspeed
