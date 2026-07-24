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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache/cache_types.h"
#include "resource/types.h"
#include "scheduler/types.h"
#include "scheduler/request.h"
#include "scheduler/execution_plan.h"
#include "scheduler/execution_event.h"
#include "scheduler/kv_cache_events.h"

#include "resource/allocator/page_allocator.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/mamba_host_allocator.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"

#include "fsm/forward_events.h"
#include "fsm/cache_events.h"
#include "fsm/pd_events.h"

#if TOKENSPEED_FLAT_KVCACHE
#include <unordered_set>

#include "cache/block_pool.h"
#include "cache/block_pool_set.h"
#include "cache/flat_reservation_tracker.h"
#include "cache/kv_cache_coordinator.h"
#include "scheduler/flat_kv_completion_ledger.h"
#endif
namespace tokenspeed {

// Allocation-free cross-language summary for the scheduler's heterogeneous
// device pools. Pressure is retained as an exact ratio so Python can derive
// equivalent legacy page counts without floating-point rounding drift.
struct FlatPoolAggregate {
    std::int64_t active_bytes{0};
    std::int64_t capacity_bytes{0};
    std::int32_t pressure_numerator{0};
    std::int32_t pressure_denominator{1};
};

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config);

    void SubmitRequests(const std::vector<RequestSpec>& request_specs);
    std::vector<std::string> CalcRollingHash(const std::vector<std::int32_t>& input_tokens, bool apply_match = false);

    ExecutionPlan NextExecutionPlan();

    void Advance(const ExecutionEvent& event);
    std::vector<KvCacheEvent> DrainKvEvents();

    std::size_t WaitingSize() const;
    std::size_t DecodingSize() const;
    std::size_t RetractedSize() const;
    std::size_t AvailableKvPages() const;
    std::size_t ActiveKvPages() const;
    std::size_t PrefillSize() const;
    std::int32_t GetRequestTokenSize(const std::string& id) const;
    std::vector<std::string> PagedCacheGroupIds() const;
    std::int32_t PagedCacheGroupTotalPages(const std::string& group_id) const;
    std::int32_t PagedCacheGroupAvailablePages(const std::string& group_id) const;
    std::int64_t PagedCacheGroupFailedAllocCount(const std::string& group_id) const;
    std::vector<std::int32_t> GetRequestPagedCachePageIds(const std::string& request_id,
                                                          const std::string& group_id) const;
    // Compact-view base logical-page offset; 0 for full-history / unseen.
    std::int32_t GetRequestPagedCacheBaseLogicalPage(const std::string& request_id, const std::string& group_id) const;
#if TOKENSPEED_FLAT_KVCACHE
    // Legacy pool-0 scalar accessor; unchanged for single-pool configs.
    std::int32_t FlatPoolFreeBlocks() const { return block_pool_.NumFreeBlocks(); }
    PoolDemand FlatReservedBlocksByPool() const;
    FlatPoolAggregate FlatPoolAggregateStats() const;
    std::vector<BlockPoolSnapshot> FlatPoolSnapshots() const;
    std::uint64_t FlatKVGeneration() const;
    bool FlatKVQuiescent() const;
    std::uint64_t ResetFlatKVCache();
    std::int32_t FlatHostPoolCachedBlocks() const { return flat_host_pool_.NumCachedBlocks(); }
    std::int32_t FlatHostPoolFreeBlocks() const { return flat_host_pool_.NumFreeBlocks(); }
    std::int32_t FlatHostPoolPinnedBlocks() const { return flat_host_pool_.NumPinnedCachedBlocks(); }
#endif

private:
    // Second element is LoadBackOperation list (normal path) or WriteBackOperation list (retract triggered).
    std::tuple<std::vector<ForwardOperation>,
               std::variant<std::vector<LoadBackOperation>, std::vector<WriteBackOperation>>>
    newForwardOperation(std::vector<Request*> candidates);
    std::vector<WriteBackOperation> newWriteBackOperation(
        std::unordered_map<std::string, std::unique_ptr<Request>>& requests);
#if !TOKENSPEED_FLAT_KVCACHE
    std::optional<WriteBackOperation> newRetractOperation(Request* retract_request);
#endif

    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                             std::vector<LoadBackOperation>& loadback_ops);
#if TOKENSPEED_FLAT_KVCACHE
    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleFlatPrefillFirstChunkEvent event);
#endif
    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event);
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event,
                                            std::size_t outstanding_dispatches);
#if !TOKENSPEED_FLAT_KVCACHE
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent event);
    std::optional<WriteBackOperation> applyEventAndGenerateOp(Request* request, fsm::ScheduleRetractEvent event);
#endif
    PrefetchOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefetchEvent event);

#if !TOKENSPEED_FLAT_KVCACHE
    void finalizeRadixPageTableEmission(Request* request, ForwardOperationBase& op, bool force_full);
#endif

    std::optional<fsm::SchedulePrefetchEvent> schedulePrefetch(Request* request, const MatchResult& match);

    std::optional<fsm::SchedulePrefillFirstChunkEvent> schedulePrefillFirstChunk(
        Request* request, std::int32_t remaining, std::int32_t decode_input_tokens, bool disable_l2_cache,
        std::map<std::string, std::int32_t>& simulated_free, PoolDemand* flat_shortfall);
#if TOKENSPEED_FLAT_KVCACHE
    std::optional<fsm::ScheduleFlatPrefillFirstChunkEvent> scheduleExplicitFlatPrefillFirstChunk(
        Request* request, std::int32_t remaining, std::int32_t decode_input_tokens, PoolDemand* flat_shortfall);
#endif
    std::optional<fsm::SchedulePrefillEvent> schedulePrefill(Request* request, std::int32_t remaining,
                                                             std::int32_t reserve_num_tokens_in_next_schedule_event,
                                                             std::map<std::string, std::int32_t>& simulated_free,
                                                             PoolDemand* flat_shortfall);
    std::optional<fsm::ScheduleDecodeEvent> scheduleDecode(Request* request,
                                                           std::map<std::string, std::int32_t>& simulated_free,
                                                           PoolDemand* flat_shortfall,
                                                           std::size_t outstanding_dispatches);
#if !TOKENSPEED_FLAT_KVCACHE
    std::optional<fsm::ScheduleDecodeFromRetractedEvent> scheduleDecodeFromRetracted(
        Request* request, std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleRetractEvent> scheduleRetract(Request* request);
#endif

#if TOKENSPEED_FLAT_KVCACHE
    // One hash pass at admission: the probe owns the folded group keys used by
    // acquisition after every admission check succeeds.
    struct FlatAdmissionMatch {
        KvCacheCoordinator::AdmissionProbe probe;
        std::vector<std::string> ext_hashes;
    };
    std::vector<std::string> flatPrefixHashesAtAdmission(Request* request) const;
    FlatAdmissionMatch matchFlatPrefixAtAdmission(Request* request);
    std::optional<PoolDemand> flatAdmitFirstChunk(Request* request, const PoolDemand& claim_demand,
                                                  const KvCacheCoordinator::FreshDemandPlan& demand,
                                                  std::int32_t ext_real_pages, PoolDemand* shortfall);
    std::optional<PoolDemand> flatAdmitPrefillChunk(Request* request, std::int32_t chunk_tokens,
                                                    std::int32_t decode_reserve_tokens,
                                                    std::int32_t num_computed_tokens, PoolDemand* shortfall);
    bool flatAdmitDecode(Request* request, std::size_t outstanding_dispatches, PoolDemand* shortfall);
    static std::int32_t decodeReservationTokensForSlots(std::int32_t verify_width, std::size_t slots);
    void resolveFlatStarvation(const std::vector<Request*>& candidates, Request* target, const PoolDemand& shortfall,
                               std::span<Request* const> scheduled_requests) noexcept;
    void attachFlatKVCompletionInput(Request* request, FlatForwardOperation& op, std::size_t row,
                                     bool apply_fsm_result);
    void ApplyFlatCompletion(const FlatKVReadyCompletion& ready, Request* request);
    void FinalizeFlatQuiescentState(Request* request);
    bool deferFlatTerminal(const std::string& request_id, FlatPendingTerminal terminal);
    bool applyPendingFlatTerminal(Request* request);
    void applyFlatFinish(const std::string& request_id);
    void applyFlatAbort(const std::string& request_id);
    void consumeFlatCompletionDebt(const std::string& request_id, std::size_t count);
#endif

    void check_device_mem();

private:
    void handleEvent(const cache::PrefetchDone& event);
    void handleEvent(const cache::WriteBackDone& event);
    void handleEvent(const cache::LoadBackDone& event);
    void handleEvent(const pd::BootstrappedEvent& event);
    void handleEvent(const pd::FailedEvent& event);
    void handleEvent(const pd::SucceededEvent& event);
    void handleEvent(const pd::RemotePrefillDoneEvent& event);
    void handleEvent(const forward::ExtendResult& event);
    void handleEvent(const forward::Abort& event);
    void handleEvent(const forward::Finish& event);
    void handleEvent(const forward::UpdateReserveNumTokens& event);

private:
    Request* find_request(const std::string& rid) {
        auto it = requests_.find(rid);
        return it != requests_.end() ? it->second.get() : nullptr;
    }

    // Coordinator-owned schema for flat KV-cache ops; empty on the radix
    // build so op construction stays model-agnostic and #if-free.
    std::span<const KvCacheGroupSchema> FlatCacheSchema() const {
#if TOKENSPEED_FLAT_KVCACHE
        return coordinator_.Schema();
#else
        return {};
#endif
    }

private:
    KVPrefixCache& radixPrefixCache() {
        _assert(kv_prefix_cache_.has_value(), "radix prefix cache is unavailable for explicit flat KV");
        return *kv_prefix_cache_;
    }

    SchedulerConfig config_;

private:
    PageAllocator device_allocator_;
    PageAllocator host_allocator_;
    std::optional<MambaChunkAllocator> mamba_allocator_{};
    std::optional<MambaHostAllocator> mamba_host_allocator_{};
    // Explicit-pool Flat owns no radix object, including the root
    // TreeNode. Compatibility paths construct this in-place before Hybrid.
    std::optional<KVPrefixCache> kv_prefix_cache_;
    ReqPoolAllocator req_pool_allocator_;
    std::optional<HybridPrefixCache> hybrid_prefix_cache_{};

#if !TOKENSPEED_FLAT_KVCACHE
    struct RadixPageTableEmission {
        std::int32_t prefix_pages{-1};
        std::vector<std::int32_t> local_pages;
    };
    // Baseline of the page table last emitted for each Python req_to_page row.
    // Slot 0 is reserved, matching ReqPoolAllocator and the Python request pool.
    std::vector<RadixPageTableEmission> radix_page_table_emissions_;
#endif

#if TOKENSPEED_FLAT_KVCACHE
    BlockPoolSet block_pools_;
    // Compatibility alias for the legacy host-tier/ticket path, which is
    // explicitly restricted to a one-pool BlockPoolSet.
    BlockPool& block_pool_;
    // Host tier = a second BlockPool, isomorphic to the device pool (block 0 is the null
    // placeholder there too); the two differ only in which memory the ids index.
    BlockPool flat_host_pool_;
    KvCacheCoordinator coordinator_;
    FlatKVCompletionLedger flat_completion_ledger_;
    // Legacy-only completion debt. Explicit-pool mode has exactly one authority:
    // the fixed FIFO stored directly in Request.
    std::unordered_map<std::string, std::int32_t> pending_forward_results_;
    // Requests own fixed-shape reservation accounts; the tracker stores only
    // their component-wise aggregate. This member precedes requests_, so every
    // account is destroyed before its tracker.
    FlatReservationTracker flat_reserved_pages_;
    // Flat retract requires TWO consecutive starved rounds (an in-flight Finish fakes one)
    // before releasing a victim; see resolveFlatStarvation.
    std::int32_t flat_starved_rounds_{0};
    std::string flat_starved_request_id_;
    // Requests terminalized as flat OOM (pool wedged by unretractable mid-prefill holders, no
    // retract victim); drained into the plan being built for the client layer to fail them.
    std::vector<std::string> flat_oom_request_ids_;
    // Reused terminal-id buffer. SubmitRequests grows it before admitting a
    // longer id; an OOM move consumes it and the next plan restores capacity
    // before any FSM mutation.
    std::string flat_oom_terminal_scratch_;
    std::size_t flat_max_request_id_size_{0};

    struct FlatStoreTicket {
        std::string key;
        BlockRef device_block;  // source page, pinned under the D2H copy
        BlockRef host_block;    // destination page, unhashed until WriteBackDone publishes it
    };
    // In-flight D2H stores. The host pool is transaction-blind like the device pool, so the
    // key-dedupe index lives here, paired with the op ledger: Add/Retire are the only mutation
    // points, so keys_ always equals the union of in-flight ticket keys.
    class FlatStoreLedger {
    public:
        void Add(cache_op_id id, std::vector<FlatStoreTicket> tickets) {
            for (const FlatStoreTicket& t : tickets) {
                keys_.insert(t.key);
            }
            const bool inserted = ops_.emplace(id, std::move(tickets)).second;
            _assert(inserted, "duplicate flat store op id");
        }
        // Empty result: unknown op (the radix WriteBackDone path owns it).
        std::vector<FlatStoreTicket> Retire(cache_op_id id) {
            auto it = ops_.find(id);
            if (it == ops_.end()) {
                return {};
            }
            for (const FlatStoreTicket& t : it->second) {
                keys_.erase(t.key);
            }
            std::vector<FlatStoreTicket> tickets = std::move(it->second);
            ops_.erase(it);
            return tickets;
        }
        bool InFlight(const std::string& key) const { return keys_.contains(key); }
        bool Empty() const { return ops_.empty(); }

    private:
        std::unordered_map<cache_op_id, std::vector<FlatStoreTicket>> ops_;
        std::unordered_set<std::string> keys_;
    };
    FlatStoreLedger flat_store_ops_;

    struct FlatLoadTicket {
        std::vector<BlockRef> host_pins;
        std::vector<BlockRef> device_blocks;
    };
    // In-flight H2D loads: op_id -> the pinned source host pages plus the pinned destination
    // device pages (a freed destination must not be recycled under the copy); LoadBackDone drops both.
    std::unordered_map<cache_op_id, FlatLoadTicket> flat_load_ops_;

    // Component-wise admission uses need + other reservations <= free + reclaim
    // credit, avoiding signed/negative pool budgets.
    bool flatCanAdmit(const Request& request, const PoolDemand& need, const PoolDemand* credit,
                      PoolDemand* shortfall) noexcept;
#endif

private:
    std::unordered_map<std::string, std::unique_ptr<Request>> requests_;
    std::unordered_map<cache_op_id, CacheOpSpec> cache_op_tracker_;
    std::vector<KvCacheEvent> kv_events_;
    SchedulerStats stats_;
};

}  // namespace tokenspeed
