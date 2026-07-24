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
#include "cache/kv_cache_coordinator.h"
#include "cache/forward_cache_ops.h"
#endif
namespace tokenspeed {

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
    std::size_t AvailableHostKvPages() const;
    std::size_t ActiveKvPages() const;
    std::size_t PrefillSize() const;
    std::int32_t GetRequestTokenSize(const std::string& id) const;
    std::vector<std::string> PagedCacheGroupIds() const;
    std::int32_t PagedCacheGroupTotalPages(const std::string& group_id) const;
    std::int32_t PagedCacheGroupAvailablePages(const std::string& group_id) const;
    std::int64_t PagedCacheGroupFailedAllocCount(const std::string& group_id) const;
    std::int32_t PagedCacheHostGroupTotalPages(const std::string& group_id) const;
    std::int32_t PagedCacheHostGroupAvailablePages(const std::string& group_id) const;
    std::int64_t PagedCacheHostGroupFailedAllocCount(const std::string& group_id) const;
    std::vector<std::int32_t> GetRequestPagedCachePageIds(const std::string& request_id,
                                                          const std::string& group_id) const;
    // Compact-view base logical-page offset; 0 for full-history / unseen.
    std::int32_t GetRequestPagedCacheBaseLogicalPage(const std::string& request_id, const std::string& group_id) const;
#if TOKENSPEED_FLAT_KVCACHE
    // Free pages in the flat shared BlockPool; int32 twin of AvailableKvPages() for C++ tests.
    std::int32_t FlatPoolFreeBlocks() const { return block_pool_.NumFreeBlocks(); }
    std::int32_t FlatHostPoolCachedBlocks() const { return flat_host_pool_.NumCachedBlocks(); }
    std::int32_t FlatHostPoolFreeBlocks() const { return flat_host_pool_.NumFreeBlocks(); }
    std::int32_t FlatHostPoolPinnedBlocks() const { return flat_host_pool_.NumPinnedCachedBlocks(); }
#endif

private:
    enum class ScheduleFailure {
        kNone,
        kGenericResource,
        kPagedCache,
    };

    template <typename Event>
    struct ScheduleAttempt {
        std::optional<Event> event;
        ScheduleFailure failure{ScheduleFailure::kGenericResource};
    };

    // Second element is LoadBackOperation list (normal path) or WriteBackOperation list (retract triggered).
    std::tuple<std::vector<ForwardOperation>,
               std::variant<std::vector<LoadBackOperation>, std::vector<WriteBackOperation>>>
    newForwardOperation(std::vector<Request*> candidates);
    std::vector<WriteBackOperation> newWriteBackOperation(
        std::unordered_map<std::string, std::unique_ptr<Request>>& requests);
    std::optional<WriteBackOperation> newRetractOperation(Request* retract_request);

    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                             std::vector<LoadBackOperation>& loadback_ops);
    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event);
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event);
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent& event);
    std::optional<WriteBackOperation> applyEventAndGenerateOp(Request* request, fsm::ScheduleRetractEvent event);
    PrefetchOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefetchEvent event);

#if !TOKENSPEED_FLAT_KVCACHE
    void finalizeRadixPageTableEmission(Request* request, ForwardOperationBase& op, bool force_full);
#endif

    std::optional<fsm::SchedulePrefetchEvent> schedulePrefetch(Request* request, const MatchResult& match);

    std::optional<fsm::SchedulePrefillFirstChunkEvent> schedulePrefillFirstChunk(
        Request* request, std::int32_t remaining, std::int32_t decode_input_tokens, bool disable_l2_cache,
        std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::SchedulePrefillEvent> schedulePrefill(Request* request, std::int32_t remaining,
                                                             std::int32_t reserve_num_tokens_in_next_schedule_event,
                                                             std::map<std::string, std::int32_t>& simulated_free);
    ScheduleAttempt<fsm::ScheduleDecodeEvent> scheduleDecode(Request* request,
                                                             std::map<std::string, std::int32_t>& simulated_free);
    ScheduleAttempt<fsm::ScheduleDecodeFromRetractedEvent> scheduleDecodeFromRetracted(
        Request* request, std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleRetractEvent> scheduleRetract(Request* request);
    LoadBackOperation newLoadBackOperation(const std::string& request_id, const std::vector<TreeNode*>& diff,
                                           const std::vector<TreeNode*>& mamba_nodes,
                                           std::vector<PagedCacheTransferPair> paged_cache_transfers,
                                           TreeNode* paged_cache_host_node);

#if TOKENSPEED_FLAT_KVCACHE
    // One hash pass at admission: non-owning device/host probes plus the hashes
    // retained for acquisition after every admission check succeeds.
    struct FlatAdmissionMatch {
        KvCacheCoordinator::AdmissionProbe probe;
        std::vector<std::string> hashes;
        std::vector<std::string> ext_hashes;
    };
    FlatAdmissionMatch matchFlatPrefixAtAdmission(Request* request);
    std::optional<std::int32_t> flatAdmitFirstChunk(Request* request, std::int32_t device_free_hit_blocks,
                                                    std::int32_t ext_real_pages, std::int32_t chunk_tokens,
                                                    std::int32_t decode_reserve_tokens) const;
    std::optional<std::int32_t> flatAdmitPrefillChunk(Request* request, std::int32_t chunk_tokens,
                                                      std::int32_t decode_reserve_tokens,
                                                      std::int32_t num_computed_tokens) const;
    bool flatAdmitDecode(Request* request) const;
    bool flatPoolWedged(const std::vector<Request*>& candidates) const;
    void resolveFlatStarvation(const std::vector<Request*>& candidates, bool made_progress);
#endif

    void abortRequest(Request* request, std::string message);
    void check_device_mem();

private:
    struct DeferredAbort {
        bool discard_writeback{false};
        std::string scheduler_message;
    };

    bool hasInFlightCacheOp(const std::string& request_id) const;
    void deferAbort(const std::string& request_id, bool discard_writeback, std::string scheduler_message = {});
    void tryFinalizeDeferredAbort(const std::string& request_id);

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
    Request* find_request(std::string rid) {
        auto it = requests_.find(rid);
        return it != requests_.end() ? it->second.get() : nullptr;
    }

    // Group-id list for flat KV-cache ops; empty span on the radix path so call sites stay #if-free.
    std::span<const std::string> FlatGroupIds() const {
#if TOKENSPEED_FLAT_KVCACHE
        return flat_group_ids_;
#else
        return {};
#endif
    }

private:
    SchedulerConfig config_;

private:
    PageAllocator device_allocator_;
    PageAllocator host_allocator_;
    std::optional<MambaChunkAllocator> mamba_allocator_{};
    std::optional<MambaHostAllocator> mamba_host_allocator_{};
    KVPrefixCache kv_prefix_cache_;
    ReqPoolAllocator req_pool_allocator_;
    std::optional<HybridPrefixCache> hybrid_prefix_cache_{};
    // Prefill-completing and decode operations still owed by the executor.
    // Starvation recovery must not release their request-owned cache pages.
    std::unordered_map<std::string, std::int32_t> pending_forward_results_;

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
    // Lifetime anchor: these pools are declared before every member that may
    // hold BlockRef, so reverse member destruction releases all handles first.
    BlockPool block_pool_;
    // Host tier = a second BlockPool, isomorphic to the device pool (block 0 is the null
    // placeholder there too); the two differ only in which memory the ids index.
    BlockPool flat_host_pool_;
    KvCacheCoordinator coordinator_;
    std::vector<std::string> flat_group_ids_;  // group_id per cache group, index-aligned to coordinator groups
    // Reserve ledger: decode pages promised at admission but Acquired only at PrefillDone->Decoding; until
    // then they sit in the free count, so every flat gate subtracts OTHER requests' entries.
    std::unordered_map<std::string, std::int32_t> flat_reserved_pages_;
    // Flat retract requires TWO consecutive starved rounds (an in-flight Finish fakes one)
    // before releasing a victim; see resolveFlatStarvation.
    std::int32_t flat_starved_rounds_{0};
    // Requests terminalized as flat OOM (pool wedged by unretractable mid-prefill holders, no
    // retract victim); drained into the plan being built for the client layer to fail them.
    std::vector<std::string> flat_oom_request_ids_;

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

    // Sum excluding request_id: a request consuming its own reservation must not be gated by it.
    std::int32_t flatReservedPagesExcept(const std::string& request_id) const {
        std::int32_t total = 0;
        for (const auto& [id, pages] : flat_reserved_pages_) {
            if (id != request_id) {
                total += pages;
            }
        }
        return total;
    }

    // Pool budget the flat gates charge against: free blocks minus other requests' decode reservations.
    std::int32_t flatFreeBudget(const std::string& request_id) const {
        return block_pool_.NumFreeBlocks() - flatReservedPagesExcept(request_id);
    }
#endif

private:
    std::unordered_map<std::string, std::unique_ptr<Request>> requests_;
    std::unordered_map<cache_op_id, CacheOpSpec> cache_op_tracker_;
    std::unordered_map<std::string, DeferredAbort> deferred_aborts_;
    std::vector<KvCacheEvent> kv_events_;
    std::vector<SchedulerAbort> scheduler_aborts_;
    SchedulerStats stats_;
};

}  // namespace tokenspeed
