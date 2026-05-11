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
#include <string>
#include <unordered_map>
#include <vector>

#include "resource/types.h"
#include "scheduler/types.h"
#include "scheduler/request.h"
#include "scheduler/execution_plan.h"
#include "scheduler/execution_event.h"

#include "resource/allocator/page_allocator.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"

#include "fsm/forward_events.h"
#include "fsm/cache_events.h"
#include "fsm/pd_events.h"
namespace tokenspeed {

struct PagedCacheGroupAdmission {
    bool ok{true};
    std::map<std::string, std::int32_t> releasable_pages{};
    std::map<std::string, std::int32_t> new_pages_needed{};
};

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config);

    void SubmitRequests(const std::vector<RequestSpec>& request_specs);
    std::vector<std::string> CalcRollingHash(const std::vector<std::int32_t>& input_tokens, bool apply_match = false);

    ExecutionPlan NextExecutionPlan();

    void Advance(const ExecutionEvent& event);

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
    // Compact-view base logical-page offset (column 0 of PageIds() = absolute
    // logical page returned here). 0 for full-history groups and unseen
    // request/group pairs. Tests use this to address compact tables.
    std::int32_t GetRequestPagedCacheBaseLogicalPage(const std::string& request_id, const std::string& group_id) const;

    // Evict all KV pages cached under the given LoRA adapter's namespace and
    // remove its virtual root from the prefix tree. Call on adapter unload.
    void EvictLoraNamespace(std::int32_t lora_id);

private:
    // Second element is LoadBackOperation list (normal path) or WriteBackOperation list (retract triggered).
    std::tuple<std::vector<ForwardOperation>,
               std::variant<std::vector<LoadBackOperation>, std::vector<WriteBackOperation>>>
    newForwardOperation(std::vector<Request*> candidates);
    std::vector<WriteBackOperation> newWriteBackOperation(
        std::unordered_map<std::string, std::unique_ptr<Request>>& requests);
    std::optional<WriteBackOperation> newRetractOperation(Request* retract_request);

    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event);
    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event);
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event);
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent event);
    std::optional<WriteBackOperation> applyEventAndGenerateOp(Request* request, fsm::ScheduleRetractEvent event);
    PrefetchOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefetchEvent event);

    std::optional<fsm::SchedulePrefetchEvent> schedulePrefetch(Request* request, const MatchResult& match);

    std::optional<fsm::SchedulePrefillFirstChunkEvent> schedulePrefillFirstChunk(
        Request* request, std::int32_t remaining, std::int32_t reserve_num_tokens_in_next_schedule_event,
        bool disable_l2_cache, std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::SchedulePrefillEvent> schedulePrefill(Request* request, std::int32_t remaining,
                                                             std::int32_t reserve_num_tokens_in_next_schedule_event,
                                                             std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleDecodeEvent> scheduleDecode(Request* request,
                                                           std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleDecodeFromRetractedEvent> scheduleDecodeFromRetracted(
        Request* request, std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleRetractEvent> scheduleRetract(Request* request);

    void check_device_mem();

private:
    void handleEvent(const cache::PrefetchDone& event);
    void handleEvent(const cache::WriteBackDone& event);
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

private:
    SchedulerConfig config_;

private:
    void acquirePagedCachePagesForRequest(const std::string& request_id, std::int32_t first_raw_position_of_op,
                                          std::int32_t target_raw_tokens_exclusive);
    PagedCacheGroupAdmission checkPagedCacheGroupAdmission(
        const std::string& request_id, std::int32_t first_raw_position_of_op, std::int32_t target_raw_tokens_exclusive,
        const std::map<std::string, std::int32_t>& simulated_free) const;
    std::map<std::string, std::int32_t> initialPagedCacheGroupSimulatedFree() const;
    static void applyPagedCacheGroupAdmissionDebit(std::map<std::string, std::int32_t>& simulated_free,
                                                   const PagedCacheGroupAdmission& admission);
    void releasePagedCachePagesForRequest(const std::string& request_id);
    void populatePagedCachePagesForOp(ForwardOperationBase& op_base) const;

private:
    PageAllocator device_allocator_;
    PageAllocator host_allocator_;
    KVPrefixCache kv_prefix_cache_;
    ReqPoolAllocator req_pool_allocator_;
    std::optional<MambaChunkAllocator> mamba_allocator_{};
    std::optional<HybridPrefixCache> hybrid_prefix_cache_{};
    std::map<std::string, std::unique_ptr<PagedCacheGroupAllocator>> paged_cache_allocators_;
    std::unordered_map<std::string, std::map<std::string, PagedCacheGroupTable>> request_paged_cache_tables_;

private:
    std::unordered_map<std::string, std::unique_ptr<Request>> requests_;
    std::unordered_map<cache_op_id, CacheOpSpec> cache_op_tracker_;
    // Stats
    SchedulerStats stats_;
};

}  // namespace tokenspeed
