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
#include <unordered_set>
#include <vector>

#include "cache/block_pool.h"
#include "core/types.h"
#include "cache/forward_cache_ops.h"
#include "cache/kv_cache_coordinator.h"
#include "fsm/forward_events.h"
#include "fsm/pd_events.h"
#include "resource/allocator/req_pool_allocator.h"
#include "scheduler/execution_event.h"
#include "scheduler/execution_plan.h"
#include "scheduler/kv_cache_events.h"
#include "scheduler/request.h"
#include "scheduler/types.h"

namespace tokenspeed {

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config);

    void SubmitRequests(const std::vector<RequestSpec>& request_specs);

    ExecutionPlan NextExecutionPlan();
    void Advance(const ExecutionEvent& event);
    std::vector<KvCacheEvent> DrainKvEvents();

    std::size_t WaitingSize() const;
    std::size_t DecodingSize() const;
    std::size_t RetractedSize() const { return static_cast<std::size_t>(retract_count_); }
    std::size_t AvailableKvPages() const;
    std::size_t ActiveKvPages() const;
    std::size_t PrefillSize() const;
    std::int32_t RequestTokenSize(const std::string& id) const;
    // Maximum logical request extent that one request can reserve in an
    // otherwise reclaimable device pool. The runtime must enforce this limit
    // before submitting requests.
    std::int32_t MaxSingleRequestTokens() const { return max_single_request_tokens_; }

    std::int32_t PagedCacheGroupTotalPages(const std::string& group_id) const;
    std::int32_t PagedCacheGroupAvailablePages(const std::string& group_id) const;

    bool PdTransferPinned(const std::string& request_id) const { return pd_transfer_pins_.contains(request_id); }
    std::int32_t PoolFreeBlocks() const { return coordinator_.NumAvailableLcmBlocks(); }
    std::int32_t HostPoolCachedBlocks() const { return coordinator_.NumHostCachedBlocks(); }
    std::int32_t HostPoolFreeBlocks() const { return host_pool_.NumEmptyLcmBlocks(); }
    std::int32_t HostPoolPinnedBlocks() const { return coordinator_.NumPinnedHostCachedBlocks(); }

private:
    struct AdmissionMatch {
        KvCacheCoordinator::PrefixProbe probe;
        std::vector<std::string> candidate_page_hashes;
        std::vector<std::string> extension_hashes;
        std::vector<std::string> page_hashes;
    };

    struct KvEventHashProgress {
        std::vector<std::uint64_t> block_hashes;
    };

    std::pair<std::vector<ForwardOperation>, std::vector<LoadBackOperation>> buildForwardOperations(
        std::vector<Request*> candidates);
    std::optional<fsm::SchedulePrefillFirstChunkEvent> schedulePrefillFirstChunk(Request* request,
                                                                                 std::int32_t remaining,
                                                                                 std::int32_t decode_input_tokens);
    std::optional<fsm::SchedulePrefillEvent> schedulePrefill(Request* request, std::int32_t remaining,
                                                             std::int32_t reserve_num_tokens_in_next_schedule_event);
    std::optional<fsm::ScheduleDecodeEvent> scheduleDecode(Request* request);

    PrefillOperation applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                                 std::vector<LoadBackOperation>& load_back_operations);
    PrefillOperation applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillEvent event);
    DecodeOperation applyEventAndBuildOperation(Request* request, fsm::ScheduleDecodeEvent event);

    AdmissionMatch matchPrefixAtAdmission(Request* request);
    std::optional<KvCacheCoordinator::AdmissionResult> admit(
        KvCacheCoordinator::PrefixProbe&& prefix, std::span<const GroupDemand> demands,
        std::optional<std::uint64_t> request_access_epoch = std::nullopt);
    std::optional<KvCacheCoordinator::AdmissionResult> admit(std::span<const GroupDemand> demands,
                                                             std::uint64_t request_access_epoch);
    std::vector<CacheKey> registerKvEventPages(const Request& request, std::span<const std::string> page_hashes,
                                               std::int32_t first_page);
    void discardUncachedKvEventPages(std::span<const CacheKey> keys);
    void handleCacheMutation(const CacheKey& key, KvCacheCoordinator::CacheMutation mutation);
    void publishCompletedPages(Request& request);
    void retractForCapacity(const std::vector<Request*>& candidates);

    void emitPendingStores(std::vector<WriteBackOperation>& write_back_operations);
    cache_op_id allocateCacheOpId() { return next_cache_op_id_++; }
    std::size_t groupIndex(const std::string& group_id) const;
    Request* findRequest(const std::string& request_id);

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

    std::int32_t calculateMaxSingleRequestTokens() const;
    std::int64_t singleRequestParentsRequired(std::int32_t token_limit) const;

    SchedulerConfig config_;
    ReqPoolAllocator req_pool_allocator_;

    // Pools outlive every CacheBlockRef stored below.
    BlockPool block_pool_;
    BlockPool host_pool_;
    KvCacheCoordinator coordinator_;
    std::vector<std::string> cache_group_ids_;
    std::int32_t max_single_request_tokens_{0};

    std::map<std::string, std::vector<std::int32_t>> new_page_ids_;
    std::unordered_map<std::string, std::int32_t> pending_forward_results_;
    std::unordered_set<std::string> pd_transfer_pins_;
    bool lcm_admission_failed_{false};

    struct StoreTicket {
        CacheKey key;
        CacheBlockRef device_block_ref;
        CacheBlockRef host_block_ref;
    };

    class StoreLedger {
    public:
        void Add(cache_op_id id, std::vector<StoreTicket> tickets);
        std::vector<StoreTicket> Retire(cache_op_id id);
        bool InFlight(const CacheKey& key) const { return keys_.contains(key); }
        bool Empty() const { return ops_.empty(); }

    private:
        std::unordered_map<cache_op_id, std::vector<StoreTicket>> ops_;
        std::unordered_set<CacheKey, CacheKeyHash> keys_;
    };

    struct LoadTicket {
        std::vector<CacheBlockRef> host_pins;
        std::vector<CacheBlockRef> device_blocks;
    };

    StoreLedger store_ops_;
    std::unordered_map<cache_op_id, LoadTicket> load_ops_;
    cache_op_id next_cache_op_id_{0};

    std::unordered_map<std::string, std::unique_ptr<Request>> requests_;
    std::vector<KvCacheEvent> kv_events_;
    std::unordered_map<std::string, KvEventHashProgress> kv_event_hash_progress_;
    std::unordered_map<CacheKey, KvBlockStoredEvent, CacheKeyHash> kv_event_pages_;
    std::unordered_map<CacheKey, std::int32_t, CacheKeyHash> cached_event_group_counts_;
    std::int64_t retract_count_{0};
};

}  // namespace tokenspeed
