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
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/core/block_pool.h"
#include "cache/tier/transfer_manager.h"
#include "fsm/forward_events.h"
#include "fsm/pd_events.h"
#include "resource/allocator/req_pool_allocator.h"
#include "scheduler/execution_event.h"
#include "scheduler/execution_plan.h"
#include "scheduler/kv_cache_events.h"
#include "scheduler/operations/cache.h"
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
    // Testing/control-plane operation. A successful return means the complete
    // Device L1 prefix cache was removed; Host L2 is never touched.
    bool ClearL1Cache();
    // Public flush operation. A successful return means both Device L1 and
    // Host L2 prefix indexes were removed.
    bool ClearCache();

    std::size_t WaitingSize() const;
    std::size_t DecodingSize() const;
    std::size_t AvailableKvPages() const;
    std::size_t ActiveKvPages() const;
    std::size_t PrefillSize() const;
    std::int32_t RequestTokenSize(const std::string& id) const;
    // Maximum logical request extent that one request can reserve in an
    // otherwise reclaimable device pool. The runtime must enforce this limit
    // before submitting requests.
    std::int32_t MaxSingleRequestTokens() const { return max_single_request_tokens_; }

    std::int32_t CacheGroupTotalPages(const std::string& group_id) const;
    std::int32_t CacheGroupAvailablePages(const std::string& group_id) const;

    bool PdTransferPinned(const std::string& request_id) const { return pd_transfer_pins_.contains(request_id); }
    std::int32_t PoolFreeBlocks() const { return coordinator_.NumAvailableLcmBlocks(); }
    std::int32_t HostPoolCachedBlocks() const { return coordinator_.NumHostCachedBlocks(); }
    std::int32_t HostPoolFreeBlocks() const { return coordinator_.NumFreeHostLcmBlocks(); }
    std::int32_t HostPoolPinnedBlocks() const { return coordinator_.NumPinnedHostCachedBlocks(); }

    // L3 storage (Mooncake Store, etc.) sits below Host. Python queries the
    // backend for existing objects, then registers the matching CacheKeys so
    // ProbePrefix can treat them as Host hits that require prefetch.
    std::vector<std::string> PrefixHashesForTokens(const std::vector<std::int32_t>& tokens) const;
    std::vector<CacheKey> ExpandPrefixKeys(std::span<const std::string> content_hashes) const {
        return coordinator_.ExpandPrefixKeys(content_hashes);
    }
    void RegisterStorageKeys(std::span<const CacheKey> keys) { coordinator_.RegisterStorageKeys(keys); }

private:
    bool clearCache(bool include_host);
    struct AdmissionMatch {
        CacheCoordinator::PrefixProbe probe;
        std::vector<std::string> candidate_prefix_hashes;
        std::vector<std::string> extension_hashes;
        std::vector<std::string> prefix_hashes;
    };

    struct KvEventHashProgress {
        std::vector<std::uint64_t> block_hashes;
    };

    // What the per-request admission layer reports back to the role
    // grammars. Together with the output ExecutionPlan (where admit()
    // records fresh pages to zero) this is ALL that layer sees of a
    // plan-building pass: batch composition lives in PlanBuild, which the
    // grammars alone hold, so nothing below them can bypass pushOperation.
    struct AdmissionFeedback {
        // Set by admit() when the last admission failed for capacity (as
        // opposed to a chunk that aligned to zero tokens). The phases clear
        // it before each attempt.
        bool admission_failed{false};
        // The first candidate whose admission failed for capacity, in phase
        // order -- the request a retraction would serve. A readmission is
        // never recorded here: when it needs a victim, the two simply do not
        // fit together, and swapping them is pure thrash.
        Request* capacity_blocker{nullptr};
    };

    std::pair<std::vector<ForwardOperation>, std::vector<LoadBackOperation>> buildForwardOperations(
        ExecutionPlan& plan, std::vector<Request*> candidates, std::vector<WriteBackOperation>& write_back_operations);
    std::optional<fsm::SchedulePrefillFirstChunkEvent> schedulePrefillFirstChunk(ExecutionPlan& plan,
                                                                                 AdmissionFeedback& feedback,
                                                                                 Request* request,
                                                                                 std::int32_t remaining,
                                                                                 std::int32_t decode_input_tokens);
    std::optional<fsm::SchedulePrefillEvent> schedulePrefill(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                             Request* request, std::int32_t remaining,
                                                             std::int32_t reserve_num_tokens_in_next_schedule_event);
    std::optional<fsm::ScheduleDecodeEvent> scheduleDecode(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                           Request* request);

    PrefillOperation applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                                 std::vector<LoadBackOperation>& load_back_operations);
    PrefillOperation applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillEvent event);
    DecodeOperation applyEventAndBuildOperation(Request* request, fsm::ScheduleDecodeEvent event);

    AdmissionMatch matchPrefixAtAdmission(Request* request);
    std::optional<CacheCoordinator::AdmissionResult> admit(
        ExecutionPlan& plan, AdmissionFeedback& feedback, CacheCoordinator::PrefixProbe&& prefix,
        std::span<const GroupDemand> demands, std::optional<std::uint64_t> request_access_epoch = std::nullopt);
    std::optional<CacheCoordinator::AdmissionResult> admit(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                           std::span<const GroupDemand> demands,
                                                           std::uint64_t request_access_epoch);
    bool admitWithKvEventTracking(ExecutionPlan& plan, AdmissionFeedback& feedback, Request& request,
                                  const fsm::CacheProgress& cache_progress, std::int32_t new_prefix_hash_begin,
                                  std::span<const GroupDemand> demands);
    std::vector<CacheKey> registerKvEventPrefixPages(const Request& request, std::span<const std::string> prefix_hashes,
                                                     std::int32_t first_page);
    void discardUncachedKvEventPages(std::span<const CacheKey> keys);
    void handleCacheMutation(const CacheKey& key, CacheCoordinator::CacheMutation mutation);
    std::optional<WriteBackOperation> publishCompletedPages(Request& request);

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

    // Mutable state of one plan-building pass: the output plan, the model
    // batch under construction, the transfer peer's two streams, and the
    // composition flags the role grammars consult. buildForwardOperations
    // owns one and hands it to the role's builder. The admission layer never
    // sees this struct -- it receives build.plan and an AdmissionFeedback --
    // so batch and budget accounting stay behind pushOperation.
    struct PlanBuild {
        explicit PlanBuild(ExecutionPlan& output_plan) : plan{output_plan} {}

        ExecutionPlan& plan;
        std::vector<ForwardOperation> operations;
        std::vector<ForwardOperation> remote_decode;
        std::vector<ForwardOperation> remote_prefill;
        std::vector<LoadBackOperation> load_backs;
        // A round schedules each request at most once, whatever states it
        // moves through while the phases run (a prompt completed by the
        // prefill phase is PrefillDone by the time the decode phase walks
        // the same candidates -- its first decode is next round's work).
        std::unordered_set<const Request*> scheduled;
        std::int32_t token_budget{0};
        // Budget the decode batch must leave untouched: one state-checkpoint
        // page for a pending local mamba prefill, which cannot advance in
        // sub-page chunks (fused mixed mode only).
        std::int32_t state_prefill_reserve{0};
        bool pushed_prefill{false};
        bool pushed_decode{false};
        bool Full(std::int32_t max_batch_size) const {
            return token_budget <= 0 || operations.size() == static_cast<std::size_t>(max_batch_size);
        }
        bool Scheduled(const Request& request) const { return scheduled.contains(&request); }
        // No prefill made progress this round: nothing new was admitted and
        // no resident prefill advanced a chunk. Decode steps do not count --
        // they release no capacity, so a round of pure decode leaves a
        // stalled prefill exactly as stuck as an empty one.
        bool NoPrefillProgress() const { return !pushed_prefill && remote_prefill.empty(); }
    };

    // Budget/flag accounting for one operation entering the model batch.
    // Work handed to the transfer peer takes neither budget nor a batch
    // slot, so it bypasses this (but still marks the request scheduled).
    template <typename Operation>
    void pushOperation(PlanBuild& build, const Request& request, Operation operation) {
        build.scheduled.insert(&request);
        // The D role budgets prompt tokens only: its prefills are the peer's
        // work (they ride plan.remote_prefill and never reach here), so its
        // decode steps run against no chunk budget.
        if constexpr (std::is_same_v<Operation, PrefillOperation>) {
            build.token_budget -= operation.input_length;
            build.pushed_prefill = true;
        } else if (config_.role != Role::kD) {
            build.token_budget -= operation.input_length;
        }
        if constexpr (std::is_same_v<Operation, DecodeOperation>) {
            build.pushed_decode = true;
        }
        build.operations.push_back(std::move(operation));
    }

    // Admission for one prefill-work candidate: a resumed chunk for a
    // Prefilling request, the first chunk otherwise. Returns the built
    // operation, or nullopt when admission fails (feedback.admission_failed
    // says whether capacity was the reason).
    std::optional<PrefillOperation> schedulePrefillCandidate(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                             Request* request, std::int32_t token_budget,
                                                             std::int32_t decode_reserve,
                                                             std::vector<LoadBackOperation>& load_backs);

    // The readmission this round may schedule, or nullptr: among the
    // retracted requests holding a recoverable snapshot, decode-origin
    // victims first, then oldest retraction epoch. Derived from the states
    // themselves, so a request that finishes or aborts while retracted
    // simply stops qualifying. A snapshot-less retraction is not in this
    // ordering at all -- it re-prefills through the ordinary admission path
    // (admitsLikeNewPrompt).
    static Request* nextReadmission(std::span<Request* const> candidates);

    // The capacity-retraction entry shared by the D and fused grammars:
    // fires only when no prefill progressed and admission failed. Retracts
    // victims and RETRIES the blocked admission in the same plan build, so
    // the freed capacity reaches the request it was freed for -- never a
    // free page waiting for whoever asks first next round.
    void maybeRetractForCapacity(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates,
                                 std::vector<WriteBackOperation>& write_back_operations);
    Request* chooseVictim(std::span<Request* const> candidates) const;
    void retractVictim(Request& victim, std::vector<WriteBackOperation>& write_back_operations);

    // One plan-building grammar per engine role: the roles share the
    // scheduling mechanism (schedulePrefill / scheduleDecode / admission)
    // and the two phase helpers below, but compose genuinely different
    // batches, so each role states its own story instead of one loop full of
    // role conditionals. Every phase walks the candidates in submission
    // order (requests_ is the FIFO) -- what used to be a priority ladder is
    // now the phase sequence itself, and within a phase older requests win.
    void buildPrefillWorkerPlan(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates);
    void buildDecodeWorkerPlan(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates,
                               std::vector<WriteBackOperation>& write_back_operations);
    void buildFusedPlan(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates,
                        std::vector<WriteBackOperation>& write_back_operations);
    void scheduleLocalPrefillWork(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates,
                                  Request* readmission, std::int32_t decode_reserve);
    void scheduleDecodeBatch(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates);

    std::int32_t calculateMaxSingleRequestTokens(std::int64_t usable_lcm_blocks) const;
    std::int64_t singleRequestLcmBlocksRequired(std::int32_t token_limit) const;

    SchedulerConfig config_;
    ReqPoolAllocator req_pool_allocator_;

    // Pools outlive every CacheBlockRef stored below.
    BlockPool block_pool_;
    BlockPool host_pool_;
    CacheCoordinator coordinator_;
    TierTransferManager tier_transfers_;
    std::vector<WriteBackOperation> pending_write_back_operations_;
    std::vector<std::string> cache_group_ids_;
    std::int32_t max_single_request_tokens_{0};

    std::unordered_set<std::string> pd_transfer_pins_;
    // Stamped onto each retraction; the readmission order lives on the
    // Retracted states themselves (nextReadmission).
    std::int64_t next_retraction_epoch_{1};

    // Submission order -- the FIFO every scheduling phase walks, identical
    // on every rank because the mirrored schedulers receive identical
    // batches. The unique_ptr keeps each Request's address stable across
    // vector reshuffles, so the id index below never dangles.
    std::vector<std::unique_ptr<Request>> requests_;
    std::unordered_map<std::string, Request*> requests_by_id_;
    std::vector<KvCacheEvent> kv_events_;
    std::unordered_map<std::string, KvEventHashProgress> kv_event_hash_progress_;
    std::unordered_map<CacheKey, KvBlockStoredEvent, CacheKeyHash> kv_event_pages_;
    // Number of resident child cache entries behind each scheduler-level
    // boundary. A group may contribute more than one child entry.
    std::unordered_map<CacheKey, std::int32_t, CacheKeyHash> cached_event_child_counts_;
    std::int32_t cache_entries_per_event_boundary_{0};
};

}  // namespace tokenspeed
