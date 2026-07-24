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
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/block_pool_set.h"
#include "cache/cache_group.h"
#include "cache/cache_types.h"

namespace tokenspeed {

namespace detail {
struct KvCacheGroupCountsTag {};
using KvCacheGroupCounts = CompactCounts<KvCacheGroupCountsTag>;
}  // namespace detail

// num_common_tokens is the cross-group token boundary. per_group[i] owns the
// exact set of pinned pages needed by group i at that boundary.
struct CoordinatorMatch {
    std::int32_t num_common_tokens{0};
    std::vector<PrefixMatch> per_group;
};

class KvCacheCoordinator {
public:
    struct StoreCandidate {
        std::string key;
        BlockRef block;
    };

    KvCacheCoordinator(std::vector<KvCacheGroupSchema> schema, std::vector<CacheGroup> groups, BlockPool& pool,
                       BlockPool* host_pool = nullptr, std::int32_t base_block_size = 0,
                       std::int32_t history_alignment_tokens = 0);
    KvCacheCoordinator(std::vector<KvCacheGroupSchema> schema, std::vector<CacheGroup> groups, BlockPoolSet& pools,
                       BlockPool* host_pool = nullptr, std::int32_t base_block_size = 0,
                       std::int32_t history_alignment_tokens = 0);

    std::int32_t NumGroups() const { return static_cast<std::int32_t>(groups_.size()); }
    std::int32_t BaseBlockSize() const { return base_block_size_; }
    std::int32_t HistoryAlignmentTokens() const { return history_alignment_tokens_; }
    // Compatibility name for the legacy all-history-anchor geometry.
    std::int32_t LcmBlockSize() const { return history_alignment_tokens_; }

    KvCacheManager& GroupManager(std::int32_t i) { return groups_[static_cast<std::size_t>(i)].Manager(); }
    const KvCacheManager& GroupManager(std::int32_t i) const { return groups_[static_cast<std::size_t>(i)].Manager(); }
    std::span<const KvCacheGroupSchema> Schema() const noexcept { return schema_; }
    const KvCacheGroupSchema& GroupSchema(std::size_t schema_index) const { return schema_.at(schema_index); }
    std::optional<std::size_t> FindSchemaIndex(std::string_view group_id) const noexcept;

    struct AdmissionMatch {
        CoordinatorMatch device;
        CoordinatorMatch host;
    };

    struct CoordinatorProbe {
        std::int32_t num_common_tokens{0};
        PoolDemand free_hit_blocks_by_pool;
        std::vector<PrefixProbe> per_group;
    };

    struct AdmissionProbe {
        AdmissionProbe() = default;
        AdmissionProbe(AdmissionProbe&& other) noexcept
            : group_keys{std::move(other.group_keys)},
              device{std::move(other.device)},
              host{std::move(other.host)},
              consumable_{std::exchange(other.consumable_, false)} {}
        AdmissionProbe& operator=(AdmissionProbe&& other) noexcept {
            if (this != &other) {
                group_keys = std::move(other.group_keys);
                device = std::move(other.device);
                host = std::move(other.host);
                consumable_ = std::exchange(other.consumable_, false);
            }
            return *this;
        }
        AdmissionProbe(const AdmissionProbe&) = delete;
        AdmissionProbe& operator=(const AdmissionProbe&) = delete;

        std::vector<std::vector<std::string>> group_keys;
        CoordinatorProbe device;
        CoordinatorProbe host;

    private:
        friend class KvCacheCoordinator;

        bool Consume() noexcept { return std::exchange(consumable_, false); }
        bool Consumable() const noexcept { return consumable_; }

        bool consumable_{false};
    };

    // Probe owns the folded keys for one admission attempt, but no BlockRefs;
    // it does not refresh LRU state. Acquire consumes it after every gate has
    // succeeded.
    AdmissionProbe ProbePrefix(std::span<const std::string> content_hashes) const;
    AdmissionMatch AcquirePrefix(AdmissionProbe probe);

    class PreparedPrefix {
    public:
        PreparedPrefix(PreparedPrefix&&) noexcept = default;
        PreparedPrefix& operator=(PreparedPrefix&&) noexcept = default;
        PreparedPrefix(const PreparedPrefix&) = delete;
        PreparedPrefix& operator=(const PreparedPrefix&) = delete;

        std::int32_t HitTokens() const noexcept { return probe_.device.num_common_tokens; }
        const PoolDemand& ClaimDemand() const noexcept { return probe_.device.free_hit_blocks_by_pool; }

    private:
        friend class KvCacheCoordinator;
        explicit PreparedPrefix(AdmissionProbe probe) : probe_{std::move(probe)} {}

        AdmissionProbe probe_;
    };

    struct CommittedPrefix {
        std::int32_t hit_tokens{0};
        std::vector<BlockTable> tables;
    };

    // Inline pool demand for first-chunk admission. Legacy Flat consumes only
    // this gate/reservation view and does not allocate per-group scratch.
    class FreshDemandPlan {
    public:
        FreshDemandPlan(FreshDemandPlan&&) noexcept = default;
        FreshDemandPlan& operator=(FreshDemandPlan&&) noexcept = default;
        FreshDemandPlan(const FreshDemandPlan&) = delete;
        FreshDemandPlan& operator=(const FreshDemandPlan&) = delete;

        const PoolDemand& ChunkDemand() const noexcept { return chunk_demand_; }
        const PoolDemand& ProtectedDemand() const noexcept { return protected_demand_; }
        const PoolDemand& ReservationDemand() const noexcept { return reservation_demand_; }

    private:
        friend class KvCacheCoordinator;

        FreshDemandPlan(PoolDemand chunk_demand, PoolDemand protected_demand, PoolDemand reservation_demand)
            : chunk_demand_{std::move(chunk_demand)},
              protected_demand_{std::move(protected_demand)},
              reservation_demand_{std::move(reservation_demand)} {}

        PoolDemand chunk_demand_;
        PoolDemand protected_demand_;
        PoolDemand reservation_demand_;
    };

    // Explicit Flat commit additionally consumes the per-group counts produced
    // by the same geometry sweep. It remains a short-lived value, not a
    // publication transaction.
    class FreshAllocationPlan {
    public:
        FreshAllocationPlan(FreshAllocationPlan&&) noexcept = default;
        FreshAllocationPlan& operator=(FreshAllocationPlan&&) noexcept = default;
        FreshAllocationPlan(const FreshAllocationPlan&) = delete;
        FreshAllocationPlan& operator=(const FreshAllocationPlan&) = delete;

        const FreshDemandPlan& Demand() const noexcept { return demand_; }

    private:
        friend class KvCacheCoordinator;

        FreshAllocationPlan(std::int32_t chunk_tokens, detail::KvCacheGroupCounts fresh_counts, FreshDemandPlan demand)
            : chunk_tokens_{chunk_tokens}, fresh_counts_{std::move(fresh_counts)}, demand_{std::move(demand)} {}

        std::int32_t chunk_tokens_{0};
        detail::KvCacheGroupCounts fresh_counts_;
        FreshDemandPlan demand_;
    };

    PreparedPrefix PreparePrefix(std::span<const std::string> content_hashes) const;
    FreshDemandPlan PlanFreshDemand(std::int32_t chunk_tokens, std::int32_t protected_tokens) const;
    FreshAllocationPlan PlanFreshAllocation(std::int32_t chunk_tokens, std::int32_t protected_tokens) const;
    std::optional<CommittedPrefix> CommitPreparedPrefix(PreparedPrefix prepared, FreshAllocationPlan allocation);

    void ClaimCommonPrefix(std::span<BlockTable> tables, CoordinatorMatch&& hit);
    std::vector<BlockTransfer> LoadHostExtension(std::span<BlockTable> tables, CoordinatorMatch&& host);

    bool Acquire(std::span<BlockTable> tables, std::int32_t num_tokens);
    PoolDemand BlocksNeededByPool(std::span<const BlockTable> tables, std::int32_t num_tokens) const;
    PoolDemand BlocksReclaimableByPool(std::span<const BlockTable> tables, std::int32_t num_computed_tokens,
                                       bool count_uncached) const;
    PoolDemand BlocksReleasedByFreeByPool(std::span<const BlockTable> tables) const;

    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::int32_t first_slot = 0, std::int32_t end_tokens = -1);
    bool AdvanceRequest(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                        std::int32_t first_page_slot, std::int32_t num_new_tokens, std::int32_t num_computed_tokens);

    void PublishReadyBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                            std::span<const std::int32_t> previous_raw_ends,
                            std::span<const std::int32_t> ready_raw_ends);
    void RewindTail(std::span<BlockTable> tables, std::int32_t accepted_raw_end, std::int32_t retain_raw_end) noexcept;
    void ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens) noexcept;
    void Free(std::span<BlockTable> tables) noexcept;

    std::vector<StoreCandidate> TakePendingStores() { return std::exchange(pending_stores_, {}); }
    bool HasHostTier() const { return host_pool_ != nullptr; }
    void SetCompletionFencedPublication(bool enabled) { completion_fenced_publication_ = enabled; }
    bool CompletionFencedPublication() const { return completion_fenced_publication_; }

private:
    std::vector<std::string> keysForGroup(std::span<const std::string> content_hashes, std::uint32_t schema_index,
                                          std::int32_t group_block_size, std::int32_t first_base = 0) const;
    std::vector<std::vector<std::string>> buildGroupKeys(std::span<const std::string> content_hashes) const;
    CoordinatorProbe probeTierWithKeys(const BlockPool* shared_pool,
                                       std::span<const std::vector<std::string>> group_keys,
                                       std::int32_t num_base_pages, std::int32_t floor_tokens) const;
    CoordinatorMatch acquireTierWithKeys(BlockPool* shared_pool, std::span<const std::vector<std::string>> group_keys,
                                         std::int32_t floor_tokens, CoordinatorProbe&& probe);
    FreshDemandPlan planFreshDemand(std::int32_t chunk_tokens, std::int32_t protected_tokens,
                                    detail::KvCacheGroupCounts* fresh_counts) const;
    void Initialize();

    // The coordinator is the sole value owner. CacheGroup and completion
    // machinery borrow entries by stable schema index for this lifetime.
    std::vector<KvCacheGroupSchema> schema_;
    std::unordered_map<std::string_view, std::size_t> schema_index_by_group_id_;
    std::vector<CacheGroup> groups_;
    std::vector<std::size_t> match_order_;
    bool has_continuation_state_{false};
    std::vector<BlockPool*> pools_;
    BlockPool* host_pool_{nullptr};
    std::int32_t base_block_size_{0};
    std::int32_t history_alignment_tokens_{0};
    bool completion_fenced_publication_{false};
    std::vector<StoreCandidate> pending_stores_;
};

KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, BlockPool& pool, BlockPool* host_pool = nullptr);
KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, BlockPoolSet& pools,
                                   BlockPool* host_pool = nullptr);

// Legacy FSM facade declaration. The implementation remains in
// forward_cache_ops.cpp; declaring it here keeps forward_events independent of
// the configuration-helper header.
std::vector<BlockTransfer> LoadHostExtension(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                                             CoordinatorMatch&& host);

}  // namespace tokenspeed
