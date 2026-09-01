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
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_block_ref.h"
#include "cache/coordinator/group_geometry.h"
#include "cache/cache_group.h"
#include "cache/core/cache_types.h"

namespace tokenspeed {

struct CacheCoordinatorTestAccess;

enum class CacheTier { kDevice, kHost };

// num_common_tokens is in tokens, aligned to the shared prefix granularity.
// per_group[i] is group i's PrefixMatch at exactly that length.
struct CoordinatorMatch {
    std::int32_t num_common_tokens{0};
    std::vector<PrefixMatch> per_group;
};

// Multi-group fan-out over the per-attention groups, one shared BlockPool. Holds no per-request
// state; the request access clock is global, while each request carries its issued epoch.
class CacheCoordinator {
public:
    enum class CacheMutation { kStored, kRemoved };
    using CacheMutationSink = std::function<void(const CacheKey&, CacheMutation)>;

    // The Host pool is available to explicit tier operations. Streaming controls
    // whether ordinary Device prefix publication also feeds the Host tier.
    CacheCoordinator(std::vector<CacheGroup> groups, std::int32_t prefix_granularity, BlockPool& pool,
                     BlockPool* host_pool = nullptr, bool stream_device_cache_to_host = true);

    std::int32_t NumGroups() const { return static_cast<std::int32_t>(groups_.size()); }

    std::int32_t PrefixGranularity() const noexcept { return prefix_granularity_; }
    bool HasMambaStateGroup() const;

    GroupAllocator& Allocator(std::int32_t i) { return groups_[static_cast<std::size_t>(i)].Allocator(); }
    const GroupAllocator& Allocator(std::int32_t i) const { return groups_[static_cast<std::size_t>(i)].Allocator(); }
    AttnKind GroupKind(std::int32_t i) const { return groups_[static_cast<std::size_t>(i)].Spec().kind; }
    PrefixCacheIndex& GroupPrefixIndex(std::int32_t i) { return groups_[static_cast<std::size_t>(i)].Index(); }
    const PrefixCacheIndex& GroupPrefixIndex(std::int32_t i) const {
        return groups_[static_cast<std::size_t>(i)].Index();
    }
    const PrefixMatcher& GroupMatcher(std::int32_t i) const { return groups_[static_cast<std::size_t>(i)].Matcher(); }
    // Match-policy and geometry views for scheduling code, in logical page
    // units. The managers are token-free; every token -> page conversion goes
    // through the per-group GroupGeometry held here.
    bool GroupIsPrefixClosed(std::int32_t i) const {
        return groups_[static_cast<std::size_t>(i)].Matcher().IsPrefixClosed();
    }
    std::int32_t GroupBoundaryLookbackPages(std::int32_t i) const {
        return groups_[static_cast<std::size_t>(i)].Matcher().BoundaryLookbackPages();
    }
    std::int32_t GroupBlockGranularity(std::int32_t i) const {
        return geometry_[static_cast<std::size_t>(i)].BlockGranularity();
    }
    std::int32_t GroupBlocksNeededFor(std::int32_t i, const BlockTable& table, std::int32_t num_tokens) const {
        return geometry_[static_cast<std::size_t>(i)].BlocksNeededFor(table, num_tokens);
    }
    bool GroupHasReclaimableBlocksAt(std::int32_t i, const BlockTable& table, std::int32_t num_computed_tokens) const {
        const CacheGroup& group = groups_[static_cast<std::size_t>(i)];
        return !group.Allocator()
                    .ReclaimableBlockLocationsAt(group.Index(), table, groupExpiredBlocksAt(i, num_computed_tokens))
                    .empty();
    }
    std::int32_t GroupBlocksReclaimableAt(std::int32_t i, const BlockTable& table, std::int32_t num_computed_tokens,
                                          bool count_uncached) const {
        const CacheGroup& group = groups_[static_cast<std::size_t>(i)];
        return group.Allocator().BlocksReclaimableAt(group.Index(), table, groupExpiredBlocksAt(i, num_computed_tokens),
                                                     count_uncached);
    }

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
        // Longer prefix-closed coverage worth materializing for non-closed groups.
        std::int32_t promotion_boundary_tokens{0};
        std::uint64_t access_epoch{0};
        std::vector<BlockTransfer> load_pairs;
        // Fresh device child pages appended by ordinary Acquire, aligned by
        // group_id. Cache hits and host-loaded destinations are excluded.
        std::vector<std::vector<std::int32_t>> new_page_ids;
    };

    // ProbePrefix is read-only. Cache state must not change before its
    // result is passed to Admit. Admit leaves the probe intact when capacity is
    // unavailable so the caller may perform a hypothetical-release check.
    // A missing epoch starts a new request; a supplied epoch continues that
    // request. Once commit starts, an internal plan/pool mismatch is fatal
    // because partial commit is not rolled back.
    PrefixProbe ProbePrefix(std::span<const std::string> content_hashes) const;
    // Decode-side PD reuses local history pages, while final-state groups are
    // restored from the remote endpoint snapshot. Their aligned null holes do
    // not count as cache hits.
    PrefixProbe ProbeDecodeDevicePrefix(std::span<const std::string> content_hashes) const;
    std::int32_t PromotionBoundaryTokens(const PrefixProbe& prefix) const;
    std::optional<AdmissionResult> Admit(PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                         std::optional<std::uint64_t> request_access_epoch = std::nullopt);
    // Capacity views for scheduling code, counted in LCM parent blocks. The
    // counts are opaque capacity units to the scheduler: all packing/geometry
    // arithmetic stays behind these methods.
    //
    // Number of LCM blocks that become reclaimable after dropping the exact
    // request-owned refs in tables. Used only to rank Retraction victims.
    std::int32_t NumNewlyReleasableLcmBlocks(std::span<const BlockTable> tables) const;
    std::int32_t NumAvailableLcmBlocks() const;
    std::int32_t TotalLcmBlocks() const { return pool_.NumLcmBlocks(); }
    std::int32_t NumFreeHostLcmBlocks() const { return host_pool_ == nullptr ? 0 : host_pool_->NumEmptyLcmBlocks(); }
    // LCM blocks required to place group_pages[g] pages for every group g.
    std::int64_t LcmBlocksNeededFor(std::span<const std::int64_t> group_pages) const;
    // Distinct LCM blocks referenced by the given per-request table sets.
    std::size_t NumActiveLcmBlocks(std::span<const std::span<const BlockTable>> request_tables) const;
    // Free pages (group page units) this group could still place, counting its
    // partially filled parents and every empty parent.
    std::int32_t GroupAvailablePages(std::int32_t group_index) const;

    // Registers an exact range, used for transferred prefix blocks and tests.
    // Runtime publication during Admit follows each group's boundary contract.
    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::uint64_t access_epoch, std::int32_t first_slot = 0,
                         CacheBoundaryKind boundary_kind = CacheBoundaryKind::kChunk);
    void CacheCompletedBlocks(std::span<BlockTable> tables, std::span<const std::string> prefix_hashes,
                              std::uint64_t access_epoch, std::int32_t first_new_prefix_page,
                              std::int32_t num_computed_tokens, CacheBoundaryKind boundary_kind,
                              bool stream_completed_to_host = false);
    void ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens);
    void ConsumeReservedTokens(std::span<BlockTable> tables, std::int32_t num_tokens);
    void Free(std::span<BlockTable> tables);
    // Clears only the Device prefix index. Returns false without mutation when
    // any cached block still has an owner outside its prefix index.
    bool ClearDeviceCache();
    // Clears both Device and Host prefix indexes. Returns false without
    // mutation when either tier still has a pinned cached block.
    bool ClearCache();

    struct StoreCandidate {
        CacheKey key;
    };
    struct HostAllocationStats {
        std::size_t requested{0};
        std::size_t allocated{0};
        std::size_t unallocated{0};
        std::size_t same_group_scans{0};
        std::size_t cross_group_scans{0};
    };
    struct HostAllocationBatch {
        std::vector<CacheBlockRef> blocks;
        HostAllocationStats stats;
    };
    // Queue every already-published non-state Device cache entry for D2H Store.
    // Missing keys and an absent Host tier are silently skipped.
    void QueueCachedBlocksForStore(std::span<const std::string> prefix_hashes);
    // Queue the newest Device-resident checkpoint from each snapshot-state
    // group. State checkpoints are intentionally deferred from continuous
    // Host streaming and persisted at request lifecycle boundaries instead.
    void QueueLatestSnapshotBlocksForStore(std::span<const std::string> prefix_hashes);
    std::vector<StoreCandidate> TakePendingStores() { return std::exchange(pending_stores_, {}); }
    CacheBlockRef AcquireDeviceCachedBlock(const CacheKey& key) const;
    HostAllocationBatch AcquireHostBlocks(std::span<const std::uint32_t> group_ids);
    CacheBlockRef AcquireHostBlock(std::uint32_t group_id);
    // Collection/pinning follows host-tier presence, so the slide credit flips count_uncached on this.
    bool StreamsDeviceCacheToHost() const { return stream_device_cache_to_host_; }
    bool ContainsHostCachedBlock(const CacheKey& key) const;
    bool IsHostCachedBlock(CacheBlockLocation location) const;
    std::int32_t NumHostCachedBlocks() const;
    std::int32_t NumPinnedHostCachedBlocks() const;
    void CacheHostBlock(CacheBlockRef& block_ref, const CacheKey& key);

    // Reports real device-cache entry insertions and removals. The scheduler
    // folds the per-group mutations into one externally visible prefix event.
    void SetCacheMutationSink(CacheMutationSink sink) { cache_mutation_sink_ = std::move(sink); }

private:
    friend struct CacheCoordinatorTestAccess;

    struct AcquiredPrefix {
        CoordinatorMatch device;
        CoordinatorMatch host;
    };

    std::vector<CacheKey> keysForGroup(std::span<const std::string> content_hashes, std::uint32_t group_id) const;
    std::vector<std::vector<CacheKey>> buildGroupKeys(std::span<const std::string> content_hashes) const;
    template <CacheTier Tier>
    BlockPool& tierPool();
    template <CacheTier Tier>
    const BlockPool& tierPool() const;
    template <CacheTier Tier>
    PrefixProbe::Tier probeTierWithKeys(std::span<const std::vector<CacheKey>> group_keys,
                                        std::span<const std::size_t> match_order, std::int32_t num_prefix_pages,
                                        std::int32_t floor_tokens) const;
    template <CacheTier Tier>
    CoordinatorMatch acquireTierWithKeys(std::span<const std::vector<CacheKey>> group_keys, std::int32_t floor_tokens,
                                         PrefixProbe::Tier&& probe, std::uint64_t access_epoch);
    AcquiredPrefix acquirePrefix(PrefixProbe&& probe, std::uint64_t access_epoch);
    template <CacheTier Tier>
    void cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table, std::span<const CacheKey> keys,
                                 std::int32_t first_cache_block, std::uint64_t access_epoch,
                                 CacheBoundaryKind boundary_kind, bool stream_completed_to_host = false);
    template <CacheTier Tier>
    void cacheCompletedBlocksForGroup(std::size_t group_index, const GroupDemand& demand, std::uint64_t access_epoch);
    void cacheDeviceCompletedBlocksForGroup(std::size_t group_index, const GroupDemand& demand,
                                            std::uint64_t access_epoch);
    bool evictCachedBlock(std::uint32_t group_id, CacheBlockLocation location);
    std::int32_t groupExpiredBlocksAt(std::int32_t i, std::int32_t num_computed_tokens) const {
        return geometry_[static_cast<std::size_t>(i)].ExpiredBlocksAt(groups_[static_cast<std::size_t>(i)].Spec(),
                                                                      num_computed_tokens);
    }
    std::vector<CacheGroup> groups_;
    // Per-group token -> page arithmetic, aligned with groups_.
    std::vector<GroupGeometry> geometry_;
    // Closed groups first, so non-closed groups match against a settled bound.
    std::vector<std::size_t> match_order_;
    BlockPool& pool_;
    BlockPool* host_pool_{nullptr};
    bool stream_device_cache_to_host_{false};
    std::int32_t prefix_granularity_{0};
    std::uint64_t next_access_epoch_{0};
    std::vector<StoreCandidate> pending_stores_;
    CacheMutationSink cache_mutation_sink_;
};

// One CacheGroup per spec (group_id = index), sharing one scheduler prefix
// domain P while each group may use a smaller cache-page token count.
CacheCoordinator MakeCoordinator(std::span<const CacheGroupSpec> specs, std::int32_t prefix_granularity,
                                 BlockPool& pool, BlockPool* host_pool = nullptr,
                                 bool stream_device_cache_to_host = true);

}  // namespace tokenspeed
