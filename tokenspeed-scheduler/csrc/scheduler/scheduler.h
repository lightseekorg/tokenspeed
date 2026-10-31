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
#include "cache/tier/storage_manager.h"
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

// The first admission's prefix hit by tier, per admitted request.
// device_hit + host_hit is the hit: host_hit is the KV the Host loads (the
// hit less the Device's prefix -- its full pages, and its KV tail when the
// Device holds the tail), host_gain is what the Host adds over the Device-only
// hit (PrefixProbe::device_only_hit_tokens). unresumable_kv is the KV cached
// beyond the hit (the junction).
struct PrefixHitStats {
    std::uint64_t prompt_tokens{0};
    std::uint64_t device_hit_tokens{0};
    std::uint64_t host_hit_tokens{0};
    std::uint64_t host_gain_tokens{0};
    std::uint64_t unresumable_kv_tokens{0};
};

// Host L2 counters, and the Device evictions they are read against:
// cumulative since start and never reset. Accounting only, for tests and
// inspection: no scheduling decision reads them and the runtime exports none.
struct HostCacheStats {
    PrefixHitStats prefix_hits;
    TierTransferManager::WriteBackStats publication_stores;
    TierTransferManager::WriteBackStats finish_stores;
    TierTransferManager::WriteBackStats retraction_stores;
    TierTransferManager::WriteBackStats eviction_stores;
    TierTransferManager::LoadBackStats prefix_loads;
    TierTransferManager::LoadBackStats restore_loads;
    CacheCoordinator::HostEvictionStats host_evictions;
    std::uint64_t store_candidates_dropped_by_flush{0};
    // What admissions and restores evicted under write-back and dropped at
    // once, no probe being able to resume from it on the Host (dead entries,
    // never written back).
    std::uint64_t dead_entries_not_written_back{0};
    CacheCoordinator::DeviceEvictionStats device_evictions;
};

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
    // Same in-flight, pin and suspended-request checks as ClearCache, with no
    // mutation. Weight updates MIN-reduce this across the replica before any
    // rank clears. Every flush refuses while RetractedSize() > 0: a
    // suspended request's image would otherwise resume old KV under a flushed
    // (re-weighted) engine, and its snapshot-pool leg is visible through no
    // pin.
    bool CanClearCache() const;
    bool StoragePending() const { return storage_transfers_ && storage_transfers_->HasPending(); }
    std::size_t StorageHeldBlocks() const { return storage_transfers_ ? storage_transfers_->NumHeldBlocks() : 0; }
    std::size_t StorageQuarantined() const { return storage_transfers_ ? storage_transfers_->NumQuarantined() : 0; }
    std::size_t StorageNegativeKeys() const { return storage_transfers_ ? storage_transfers_->NumNegativeKeys() : 0; }
    StorageStats GetStorageStats() const;

    // Lifecycle counters read the current FSM state; they do not schedule work.
    std::size_t BootstrappingSize() const;
    // Requests waiting for an admission: Submitted or Prefetching once PD
    // bootstrap is done, plus the suspended ones --
    // Retracted (waiting to be restored) and Restoring (their copy back is in
    // flight) -- which hold no schedulable work until they resume.
    std::size_t WaitingSize() const;
    std::size_t DecodingSize() const;
    std::size_t PrefillSize() const;
    // Device pool parents that are empty or hold only unpinned cache entries.
    // Scans the pool and every prefix index: a leak check for tests and
    // diagnostics, not a per-step gauge.
    std::int32_t AvailableLcmBlocks() const { return coordinator_.NumAvailableLcmBlocks(); }
    // Per-step gauges: empty parents are O(1), active parents walk the live
    // requests' block tables. Parents resident only as cache are
    // TotalLcmBlocks - EmptyLcmBlocks - ActiveLcmBlocks.
    std::int32_t EmptyLcmBlocks() const { return coordinator_.NumEmptyLcmBlocks(); }
    std::int32_t ActiveLcmBlocks() const;
    // RemotePrefilling only: a subset of PrefillSize(), never an extra total.
    std::size_t RemotePrefillSize() const;
    // Requests whose pages PD still pins. This resource count overlaps FSM states.
    std::size_t PdTransferSize() const;
    std::int32_t RequestTokenSize(const std::string& id) const;
    // Only fused, page-less requests can finish without the normal forward/PD
    // completion fence. This is a read-only check for the head event channel.
    bool CanAbortBeforeAdmission(const std::string& id) const;
    // Maximum logical request extent that one request can reserve in an
    // otherwise reclaimable device pool. The runtime must enforce this limit
    // before submitting requests.
    std::int32_t MaxSingleRequestTokens() const { return max_single_request_tokens_; }

    std::int32_t CacheGroupTotalPages(const std::string& group_id) const;
    std::int32_t CacheGroupAvailablePages(const std::string& group_id) const;

    // Test/diagnostic view of pdTransferInFlight by request id.
    bool PdTransferPinned(const std::string& request_id) const;
    std::int32_t HostPoolCachedBlocks() const { return coordinator_.NumHostCachedBlocks(); }
    std::int32_t HostPoolFreeBlocks() const { return coordinator_.NumFreeHostLcmBlocks(); }
    // Host pool parents, empty or not (0 without a Host pool).
    std::int32_t HostPoolTotalBlocks() const { return host_pool_.NumLcmBlocks(); }
    std::int32_t HostPoolPinnedBlocks() const { return coordinator_.NumPinnedHostCachedBlocks(); }
    HostCacheStats GetHostCacheStats() const;
    // Published tail boundaries the probe can enumerate, and how many the
    // per-prior cap has dropped since start.
    std::size_t TailEntries() const { return coordinator_.NumTailEntries(); }
    std::int64_t TailCapEvictions() const { return coordinator_.TailCapEvictions(); }
    // Empty parents of the request-private snapshot pool: every block of it
    // belongs to some retracted request's image, so a request that finished
    // or aborted while retracted must have returned its share (leak check).
    std::int32_t SnapshotPoolFreeBlocks() const { return coordinator_.NumFreeSnapshotLcmBlocks(); }
    // Requests suspended with an image: Retracted (waiting to be restored)
    // and Restoring (their copy back is in flight). A weight update must not
    // restore an image of old-weight KV under new weights, so the runtime
    // refuses to flush while this is non-zero.
    std::size_t RetractedSize() const;

    // Key geometry helpers; runtime workers receive immutable keys from
    // storage operations, never a mutable shadow of remote existence.
    std::vector<std::string> PrefixHashesForTokens(const std::vector<std::int32_t>& tokens) const;
    std::vector<CacheKey> ExpandPrefixKeys(std::span<const std::string> content_hashes) const {
        return coordinator_.ExpandPrefixKeys(content_hashes);
    }

private:
    bool clearCache(bool include_host);
    bool cacheIsClearable(bool include_host) const;
    struct AdmissionMatch {
        CacheCoordinator::PrefixProbe probe;
        std::string prefix_diagnostic;
        // Views of the request's immutable prompt hashes. Only an admitted
        // request copies its reused prefix into mutable execution progress.
        std::span<const std::string> candidate_prefix_hashes;
        std::span<const std::string> extension_hashes;
        std::span<const std::string> prefix_hashes;
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
        // Every candidate whose admission failed for capacity this round, in
        // phase order (the blocker first). When the victim turns out to be
        // the blocker itself, the next of these, if any, is who its pages
        // serve first (maybeRetractForCapacity lists the rest).
        std::vector<Request*> capacity_blocked;

        void NoteCapacityBlocked(Request* request) {
            if (capacity_blocker == nullptr) {
                capacity_blocker = request;
            }
            capacity_blocked.push_back(request);
        }
    };

    // What one plan-building pass emits beside the ExecutionPlan's own fields.
    struct BuiltOperations {
        std::vector<ForwardOperation> forward;
        std::vector<LoadBackOperation> load_backs;
        std::vector<SnapshotStoreOperation> snapshot_stores;
        std::vector<SnapshotRestoreOperation> snapshot_restores;
    };
    BuiltOperations buildForwardOperations(ExecutionPlan& plan, std::vector<Request*> candidates,
                                           std::vector<WriteBackOperation>& write_back_operations);
    // What considering a Submitted request for admission produced: the
    // first-chunk event (admitted), or no event (feedback distinguishes
    // capacity failure from waiting for a request slot).
    struct FirstChunkOutcome {
        // The D role: every State group's prompt-end checkpoint was claimed,
        // so the peer omits the state transfer (PrefillOperation::state_prefix_cached).
        bool state_prefix_cached{false};
        std::optional<fsm::SchedulePrefillFirstChunkEvent> event;
    };
    FirstChunkOutcome schedulePrefillFirstChunk(ExecutionPlan& plan, AdmissionFeedback& feedback, Request* request,
                                                std::int32_t remaining, std::int32_t decode_input_tokens,
                                                std::int64_t kv_token_limit);
    std::optional<fsm::SchedulePrefillEvent> schedulePrefill(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                             Request* request, std::int32_t remaining,
                                                             std::int32_t reserve_num_tokens_in_next_schedule_event,
                                                             std::int64_t kv_token_limit);
    std::optional<fsm::ScheduleDecodeEvent> scheduleDecode(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                           Request* request);

    PrefillOperation applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                                 std::vector<LoadBackOperation>& load_back_operations);
    PrefillOperation applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillEvent event);
    DecodeOperation applyEventAndBuildOperation(Request* request, fsm::ScheduleDecodeEvent event);

    AdmissionMatch matchPrefixAtAdmission(Request* request);
    // Records an admission's fresh pages (to zero) and copies on the plan.
    void recordAdmittedPages(ExecutionPlan& plan, const CacheCoordinator::AdmissionResult& result);
    std::optional<CacheCoordinator::AdmissionResult> admit(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                           CacheCoordinator::PrefixProbe&& prefix,
                                                           std::span<const GroupDemand> demands,
                                                           const RequestProgress& progress,
                                                           std::optional<std::uint64_t> request_access_epoch,
                                                           bool evict_blocks_past_segments);
    std::optional<CacheCoordinator::AdmissionResult> admitWithKvEventTracking(
        ExecutionPlan& plan, AdmissionFeedback& feedback, Request& request, const fsm::CacheProgress& cache_progress,
        std::span<const GroupDemand> demands, const RequestProgress& progress, bool evict_blocks_past_segments);
    void registerKvEventPrefixPages(const Request& request, std::span<const std::string> prefix_hashes,
                                    std::int32_t first_page);
    // The descriptors a publication mutates: its newly hashed full pages and,
    // at a tail, the partial page under its tail hash (token_ids = the rows
    // below T, parent = the last full page).
    void registerKvEventPublication(const Request& request, std::span<const std::string> prefix_hashes,
                                    const CompletedPages& completed);
    void markKvEventBoundaryForReconcile(const CacheKey& boundary);
    // Finished requests whose forwards have all landed: publish the frontier
    // (state and KV tail) and release the pages. Runs after every batch of
    // events, so a request that finished quiescent settles with its Finish
    // and one whose finish overtook an overlapped decode settles with that
    // decode's result -- paused or not, plan or no plan.
    void settleFinishedRequests();
    std::optional<WriteBackOperation> publishCompletedPages(Request& request);

    std::size_t groupIndex(const std::string& group_id) const;
    Request* findRequest(const std::string& request_id);
    // A PD transfer is out against this request's pages, so nothing here may
    // retract, finish or flush them. Derived from the FSM, never recorded: on
    // the D role it is the peer's prefill writing the destination pages
    // (RemotePrefilling); on the P role it is the peer's decode reading them,
    // which begins with the first scheduled chunk and ends only with the PD
    // ACK -- every page-holding state before Finishing. A fused engine never
    // transfers.
    bool pdTransferInFlight(const Request& request) const;

    void handleEvent(const cache::StorageTick& event);
    void handleEvent(const cache::StorageDone& event);
    void handleEvent(const cache::StorageDetach& event);
    void prepareStorage(Request& request);
    void releaseStorage(Request& request);
    void expireStorage(Request& request);
    std::int32_t cacheablePrefixTokens(const Request& request) const;
    void handleEvent(const cache::WriteBackDone& event);
    void handleEvent(const cache::LoadBackDone& event);
    void handleEvent(const cache::SnapshotDone& event);
    void handleEvent(const cache::RestoreDone& event);
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
        // A retraction's tail-leg store (its L2 leg joins the write-backs) and
        // a readmission's restore: cache ops riding beside the batch.
        std::vector<SnapshotStoreOperation> snapshot_stores;
        std::vector<SnapshotRestoreOperation> snapshot_restores;
        // A round schedules each request at most once, whatever states it
        // moves through while the phases run (a prompt completed by the
        // prefill phase is PrefillDone by the time the decode phase walks
        // the same candidates -- its first decode is next round's work).
        std::unordered_set<const Request*> scheduled;
        std::int32_t token_budget{0};
        std::int32_t kv_page_budget{0};
        // Budget the decode batch must leave untouched for a pending local
        // prefill that cannot advance in smaller chunks (MinPrefillChunkTokens;
        // fused mixed mode only).
        std::int32_t pending_prefill_reserve{0};
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
            if (config_.max_kv_page_num > 0) {
                const std::int64_t end =
                    static_cast<std::int64_t>(operation.extend_prefix_len) + operation.input_length;
                build.kv_page_budget -=
                    static_cast<std::int32_t>((end + config_.prefix_granularity - 1) / config_.prefix_granularity);
            }
            build.pushed_prefill = true;
        } else if (config_.role != Role::kD) {
            build.token_budget -= operation.input_length;
        }
        if constexpr (std::is_same_v<Operation, DecodeOperation>) {
            build.pushed_decode = true;
        }
        build.operations.push_back(std::move(operation));
    }

    // A resumed chunk or a new prompt's first chunk. Waiting for storage
    // returns nullopt without reporting a capacity failure.
    std::optional<PrefillOperation> schedulePrefillCandidate(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                             Request* request, std::int32_t token_budget,
                                                             std::int32_t decode_reserve,
                                                             std::vector<LoadBackOperation>& load_backs,
                                                             std::int32_t kv_page_budget);

    // The readmissions this round may restore, in rank order: among the
    // retracted requests whose image has landed, victims with generated
    // output first (they resume a generation a client is reading), then
    // oldest retraction epoch. Derived from the states themselves, so a
    // request that finishes or aborts while retracted simply stops qualifying.
    static std::vector<Request*> rankedReadmissions(std::span<Request* const> candidates);
    // Allocates fresh Device pages for the whole image plus the reserve the
    // resumed state needs, issues the restore op beside the batch and moves
    // the request to Restoring. False when it does not fit (feedback says
    // whether capacity was the reason): the readmission waits and is never
    // recorded as the capacity blocker.
    bool scheduleRestore(AdmissionFeedback& feedback, PlanBuild& build, Request* request);
    // The restore phase of the D and fused grammars: tries the ranked
    // readmissions in order, a bounded number of them, until one restores
    // (one per round). Returns whether new-prompt admission is sealed this
    // round: true once any landed image waited for Device pages, so a
    // newcomer cannot take the pages it is waiting for.
    bool scheduleReadmission(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates);

    // The capacity-retraction entry shared by the D and fused grammars:
    // fires only when no prefill progressed and admission failed. Retracts
    // victims and RETRIES the blocked admission in the same plan build, so
    // the freed capacity reaches the request it was freed for -- never a
    // free page waiting for whoever asks first next round. The empty-round
    // retraction offers it to the round's failed decodes first.
    void maybeRetractForCapacity(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates,
                                 std::vector<WriteBackOperation>& write_back_operations);
    // Who gives way, and whether it can be imaged. `victim` is nullptr when
    // nobody can (every resident is exempt or there is no resident); with a
    // shortfall it is the newest retractable resident, to be aborted because
    // no candidate's image fits the host budgets, and the shortfall is its.
    struct VictimChoice {
        Request* victim{nullptr};
        std::optional<ImageShortfall> shortfall;
    };
    // `ignore_exemption`: the empty-round retraction (docs/design/scheduler.md §2).
    VictimChoice chooseVictim(std::span<Request* const> candidates, bool ignore_exemption) const;
    // The first half of a retraction: publishes the pages the victim completed
    // since its last admission (see forward.cpp).
    void publishCompletedPrefix(Request& victim);
    // Why the host budgets cannot hold the request's image right now
    // (projected, nothing mutated), unset when they can (see forward.cpp).
    std::optional<ImageShortfall> imageShortfall(const Request& request) const;
    // Images the victim and suspends it, or returns the shortfall before
    // anything is acquired or evicted for the image; past the blob-slot check
    // its completed pages stay published (see forward.cpp).
    std::optional<ImageShortfall> retractVictim(Request& victim, PlanBuild& build,
                                                std::vector<WriteBackOperation>& write_back_operations);
    // The one site of the last-resort policy: a capacity retraction whose
    // victim cannot be imaged aborts that victim instead -- its pages and
    // slot return in this very round and the blocked grant proceeds as after
    // a retraction -- and records it on the plan for the runtime to fail the
    // request toward its client. A forced victim refused after its
    // publication is aborted here too.
    void onImageDoesNotFit(Request& victim, ImageShortfall shortfall, PlanBuild& build);
    // The debug_force_retraction_interval knob: arms the oldest Decoding
    // (N > 0) or Prefilling (N < 0) request every |N| plans, keeps it out of
    // the batch until it is quiescent, then retracts it. A victim whose image
    // does not fit (imageShortfall) runs on with nothing published; one
    // refused only after its publication is aborted.
    void maybeForceRetraction(PlanBuild& build, std::span<Request* const> candidates,
                              std::vector<WriteBackOperation>& write_back_operations);

    // The stall diagnostic (docs/design/scheduler.md §2), called once per
    // NextExecutionPlan. It never changes a plan.
    void trackStalledRounds(std::size_t num_live_requests, bool plan_is_empty);

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
                                  bool new_prompts_sealed, std::int32_t decode_reserve);
    void scheduleDecodeBatch(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates);

    SchedulerConfig config_;
    ReqPoolAllocator req_pool_allocator_;
    // One slot per retracted request: the index of its slot-state blob in
    // the runtime's arena, released by the FSM transition that drops the image.
    SnapshotSlotAllocator snapshot_slots_;

    // Pools outlive every CacheBlockRef stored below.
    BlockPool block_pool_;
    BlockPool host_pool_;
    BlockPool snapshot_pool_;
    CacheCoordinator coordinator_;
    TierTransferManager tier_transfers_;
    std::unique_ptr<StorageTransferManager> storage_transfers_;
    std::int64_t storage_now_us_{0};
    std::uint64_t storage_wait_timeouts_{0};
    std::uint64_t storage_ready_expirations_{0};
    bool storage_allowed_{false};
    std::uint64_t storage_pressure_bypasses_{0};
    bool forward_allowed_{true};
    std::vector<WriteBackOperation> pending_write_back_operations_;
    std::vector<std::string> cache_group_ids_;
    std::int32_t max_single_request_tokens_{0};
    // Accounting only (schedulePrefillFirstChunk); no decision reads it.
    PrefixHitStats prefix_hit_stats_;

    // Stamped onto each retraction; the readmission order lives on the
    // Retracted states themselves (rankedReadmissions).
    std::int64_t next_retraction_epoch_{1};
    // The forced-retraction knob's clock and the request it has armed (empty
    // when none): kept out of every batch until it is quiescent, then
    // retracted.
    std::int64_t plan_calls_{0};
    std::string forced_victim_id_;

    // Consecutive stalled rounds (trackStalledRounds); 0 after any round
    // that schedules work or leaves work in flight.
    std::int64_t stalled_rounds_{0};

    // Submission order -- the FIFO every scheduling phase walks, identical
    // on every rank because the mirrored schedulers receive identical
    // batches. The unique_ptr keeps each Request's address stable across
    // vector reshuffles, so the id index below never dangles.
    std::vector<std::unique_ptr<Request>> requests_;
    std::unordered_map<std::string, Request*> requests_by_id_;
    std::unordered_map<std::string, KvEventHashProgress> kv_event_hash_progress_;
    // What the prefix index cannot tell us about a boundary: the token
    // descriptor the external event carries, and whether that event is
    // currently out (a BlockStored not yet followed by its BlockRemoved).
    // Residency itself is the coordinator's answer, never mirrored here.
    // Cache mutations and registrations only mark a boundary for reconcile;
    // DrainKvEvents reconciles each marked boundary against its residency, so
    // the order of evictions and publications inside one admission does not
    // matter. After a drain, a descriptor exists exactly for the boundaries
    // with a cached child.
    struct KvEventBoundary {
        KvBlockStoredEvent stored;
        // Position in the prefix chain (page index). Consumers resolve a
        // Stored event's parent_block_hash on receipt, so DrainKvEvents
        // orders each batch by depth rather than by mutation order.
        std::int32_t depth{0};
        bool published{false};
        bool needs_reconcile{false};
    };
    std::unordered_map<CacheKey, KvEventBoundary, CacheKeyHash> kv_event_boundaries_;
    // The boundaries marked for reconcile. Mutations mark them in whatever
    // order the coordinator touches pages (evictions run suffix-first), so
    // this order carries no meaning; DrainKvEvents sorts the batch.
    std::vector<CacheKey> kv_event_boundaries_to_reconcile_;
};

}  // namespace tokenspeed
