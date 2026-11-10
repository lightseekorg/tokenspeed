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
#include "scheduler/checkpoint_lineage.h"
#include "scheduler/execution_event.h"
#include "scheduler/execution_plan.h"
#include "scheduler/kv_cache_events.h"
#include "scheduler/operations/cache.h"
#include "scheduler/request.h"
#include "scheduler/types.h"

namespace tokenspeed {

class StagedCacheProgress;

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
// cumulative since start and never reset. Accounting only: no scheduling
// decision reads them. The runtime exports the prefix hits and L3 query
// tokens (`PrefixHitView`); the rest stay for tests and inspection.
struct HostCacheStats {
    PrefixHitStats prefix_hits;
    // Cumulative token spans of successful storage keys, counted per cache group.
    std::uint64_t storage_query_hit_tokens{0};
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
    // What they evicted under write-back and dropped as demoted entries
    // (SchedulerConfig::demote_superseded_state_checkpoints), never written
    // back; counted under the eviction's cause.
    std::uint64_t demoted_entries_not_written_back{0};
    CacheCoordinator::DeviceEvictionStats device_evictions;
    // CacheCoordinator::LoneCompactions and LoneCompactionMoves: lone-request
    // compactions (docs/design/scheduler.md §1.4) and the blocks they moved,
    // copied or re-placed.
    std::uint64_t lone_compactions{0};
    std::uint64_t lone_compaction_moves{0};
    // CacheCoordinator::CheckpointResumeGaps, the gaps an admission resumed
    // a State checkpoint after, on either tier.
    std::vector<std::uint64_t> checkpoint_resume_gaps;
};

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config);

    // The stall diagnostic (docs/design/scheduler.md §2): consecutive plans
    // built empty with nothing in flight before the first warning, which
    // repeats at every doubling of the streak (so a stall that never ends
    // still logs only a few dozen lines), and the plan at which a lone
    // request behind such a stall is aborted on the fused role (§1.4): two
    // doublings after the first warning, so the warning precedes the abort
    // in every log.
    static constexpr std::int64_t kStalledRoundsBeforeWarning{1024};
    static constexpr std::int64_t kNoProgressAbortPlans{4096};

    void SubmitRequests(const std::vector<RequestSpec>& request_specs);

    // What one plan call may do, decided by the control plane's prefill-delay
    // vote (docs/design/prefill-delay.md).
    struct PlanOptions {
        // The fused role may schedule prefill work this round: resident
        // chunks and fresh prompts. False holds all of it for a later round;
        // decode work is unaffected. The P role (prompt-only) and the D role
        // (whose prompts are the peer's work) ignore it.
        bool allow_prefill;
        // Delay enabled: check the watermark before each new admission,
        // including after a resident chunk. Resident chunks ignore pressure.
        // False preserves ordinary admission when delay is disabled.
        bool apply_prefill_watermark;
    };
    ExecutionPlan NextExecutionPlan(PlanOptions options);

    // What the control plane's prefill-delay vote reads from every
    // attention-DP rank before deciding a plan. Read-only.
    struct PrefillGateProbe {
        // A resident chunk regardless of pressure, or a new prompt at or
        // below the watermark. Actual admission still checks cache demands.
        bool wants_prefill;
        // Decode work this rank's round would otherwise run: a first decode
        // or a decoding request. A rank with none has nothing to protect
        // from a prefill round, so the vote does not count it as a blocker.
        bool has_decode_work;
    };
    PrefillGateProbe ProbePrefillGate() const;

    // Above this pressure hold new prompts back when delay is enabled.
    // Resident chunks must be able to consume their own reservations.
    static constexpr float kPrefillWatermark = 0.9F;

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
    bool SharedPrefixStorageEnabled() const;
    struct SharedPrefixPublication {
        std::string status;
        std::int32_t confirmed_prefix_tokens{0};
        std::uint64_t cache_epoch{0};
    };
    SharedPrefixPublication GetSharedPrefixPublication(std::span<const std::int32_t> tokens,
                                                       std::int32_t max_prefix_tokens) const;

    // Lifecycle counters read the current FSM state; they do not schedule work.
    std::size_t BootstrappingSize() const;
    // Requests waiting for an admission: Submitted or Prefetching once PD
    // bootstrap is done, plus the suspended ones --
    // Retracted (waiting to be restored) and Restoring (their copy back is in
    // flight) -- which hold no schedulable work until they resume.
    std::size_t WaitingSize() const;
    std::size_t DecodingSize() const;
    std::size_t PrefillSize() const;
    // Requests in the running batch: local prefill forwards, prompts whose
    // prefill is done (about to decode or to transfer, or waiting for their
    // Decode destination), and decodes. RemotePrefilling (waiting for the
    // peer's KV), waiting and retracted requests are not running.
    std::size_t RunningSize() const;
    // P role, independent admission: computed prompts waiting for their
    // Decode destination (PrefillAwaitingDestination, a subset of
    // PrefillSize() and RunningSize()), and the Device parents in their block
    // tables, counted as ActiveLcmBlocks() counts them; blocks held outside the
    // tables (e.g. a pending in-forward State checkpoint) are not included.
    std::size_t AwaitingDestinationSize() const;
    std::int32_t AwaitingDestinationLcmBlocks() const;
    // Device pool parents that are empty or hold only unpinned cache entries.
    // Scans the pool and every prefix index: a leak check for tests and
    // diagnostics, not a per-step gauge.
    std::int32_t AvailableLcmBlocks() const { return coordinator_.NumAvailableLcmBlocks(); }
    std::int32_t FreeRequestSlots() const { return req_pool_allocator_.AvailableSlots(); }
    std::int32_t TotalRequestSlots() const { return req_pool_allocator_.Size(); }
    std::int32_t TotalLcmBlocks() const { return block_pool_.NumLcmBlocks(); }
    std::vector<std::int64_t> AdmissionCapacityCurve(std::int32_t token_step) const;
    // Per-step gauges, both O(1): the pool keeps its empty parents, the
    // requests' block tables count the parents they hold as they change.
    // Parents resident only as cache are
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
    std::size_t DemoteSession(const std::string& id, bool ended) { return coordinator_.DemoteSession(id, ended); }
    std::size_t DemoteCacheGroups(const std::vector<std::string>& ids, bool ended) {
        return coordinator_.DemoteCacheGroups(ids, ended);
    }
    std::size_t DemoteSessions(const std::vector<std::string>& ids, bool ended) {
        return coordinator_.DemoteSessions(ids, ended);
    }

    // Published tail boundaries the probe can enumerate, and how many the
    // per-prior cap has dropped since start.
    std::size_t TailEntries() const { return coordinator_.NumTailEntries(); }
    std::int64_t TailCapEvictions() const { return coordinator_.TailCapEvictions(); }
    // The entries superseded checkpoint demotion demoted since start
    // (demoteSupersededCheckpoint), on either tier or on an eviction
    // write-back's way to the Host; always 0 without
    // SchedulerConfig::demote_superseded_state_checkpoints. Accounting only.
    std::uint64_t DemotedCheckpointEntries() const { return demoted_checkpoint_entries_; }
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
    friend struct SchedulerTestAccess;
    bool clearCache(bool include_host);
    bool cacheIsClearable(bool include_host) const;
    struct AdmissionMatch {
        CacheCoordinator::PrefixProbe probe;
        bool waits_for_eviction_store{false};
        std::string prefix_diagnostic;
        // Views of the request's immutable prompt hashes. Only an admitted
        // request copies its reused prefix into mutable execution progress.
        std::span<const std::string> candidate_prefix_hashes;
        std::span<const std::string> extension_hashes;
        std::span<const std::string> prefix_hashes;
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
        // The request this build left blocked for capacity -- the first
        // whose admission failed for it, else the landed image the restore
        // phase tried first (rankedReadmissions) -- while no other request
        // holds pages, so no release can fund it: the one the no-progress
        // abort takes (NextExecutionPlan).
        Request* blocked_alone{nullptr};
    };
    BuiltOperations buildForwardOperations(ExecutionPlan& plan, std::vector<Request*> candidates,
                                           std::vector<WriteBackOperation>& write_back_operations, PlanOptions options);
    // What considering a Submitted request for admission produced: the
    // first-chunk event (admitted), or no event (feedback distinguishes
    // capacity failure from waiting for a request slot).
    struct FirstChunkOutcome {
        // The D role: every State group's prompt-end checkpoint was claimed,
        // so the peer omits the state transfer (PrefillOperation::state_prefix_cached).
        bool state_prefix_cached{false};
        std::optional<fsm::SchedulePrefillFirstChunkEvent> event;
    };
    // `yield_tokens`: what a chunk that cannot finish gives up for whole
    // prompts behind it (YieldingChunkBudget). `must_complete`: admit only a
    // first chunk that completes the prompt -- a whole prompt carried in such
    // a chunk's forward; one that would not is skipped (no capacity failure),
    // and one whose in-forward checkpoint block does not fit fails for
    // capacity rather than ending at the checkpoint.
    FirstChunkOutcome schedulePrefillFirstChunk(ExecutionPlan& plan, AdmissionFeedback& feedback, Request* request,
                                                std::int32_t remaining, std::int32_t decode_input_tokens,
                                                std::int64_t kv_token_limit, std::int32_t yield_tokens,
                                                bool must_complete);
    std::optional<fsm::SchedulePrefillEvent> schedulePrefill(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                             Request* request, std::int32_t remaining,
                                                             std::int32_t reserve_num_tokens_in_next_schedule_event,
                                                             std::int64_t kv_token_limit, std::int32_t yield_tokens);
    std::optional<fsm::ScheduleDecodeEvent> scheduleDecode(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                           Request* request);

    PrefillOperation applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                                 std::vector<LoadBackOperation>& load_back_operations);
    PrefillOperation applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillEvent event);
    DecodeOperation applyEventAndBuildOperation(Request* request, fsm::ScheduleDecodeEvent event);

    AdmissionMatch matchPrefixAtAdmission(Request* request);
    // Records an admission's fresh pages (to zero) and copies on the plan, a
    // copy from a block an earlier copy of the plan fills reading that copy's
    // source. Asserts that each copy destination is written once and never
    // zeroed in the plan.
    void recordAdmittedPages(ExecutionPlan& plan, const CacheCoordinator::AdmissionResult& result);
    // copy_in_blocks and copy_in_point as for CacheCoordinator::Admit.
    std::optional<CacheCoordinator::AdmissionResult> admit(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                           CacheCoordinator::PrefixProbe&& prefix,
                                                           std::span<const GroupDemand> demands,
                                                           const RequestProgress& progress,
                                                           std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                                           std::optional<CacheCoordinator::CopyInPoint> copy_in_point,
                                                           std::optional<std::uint64_t> request_access_epoch,
                                                           bool evict_blocks_past_segments);
    // Admits a scheduled request's next step against its live cache progress,
    // which the caller has advanced for the step (`staged`, committed by the
    // caller on success, or by publishCompletedPagesFirst when a lone
    // request's completed pages are published ahead of a step that still
    // does not fit). The publication of a prompt end spends the copy-in
    // blocks the request holds.
    std::optional<CacheCoordinator::AdmissionResult> admitWithKvEventTracking(
        ExecutionPlan& plan, AdmissionFeedback& feedback, Request& request, StagedCacheProgress& staged,
        std::span<const GroupDemand> demands, const RequestProgress& progress,
        std::optional<CacheCoordinator::CopyInPoint> copy_in_point, bool evict_blocks_past_segments);
    // Registers the descriptors of full pages [first_page, past_end_page).
    void registerKvEventPrefixPages(Request& request, std::span<const std::string> prefix_hashes,
                                    std::int32_t first_page, std::int32_t past_end_page);
    // The descriptors a publication mutates: its full pages from the first it
    // registers an entry under (CacheCoordinator::FirstPublishedPage) and, at
    // a tail, the partial page under its tail hash (token_ids = the rows below
    // T, parent = the last full page).
    void registerKvEventPublication(Request& request, std::span<const std::string> prefix_hashes,
                                    const CompletedPages& completed);
    // The boundaries (in tokens) of the tails an image's published slots sit
    // at, read off their descriptors at the retraction (none with the feed
    // off): the restore republishes such a slot after a drain may have
    // dropped its descriptor.
    std::vector<std::int32_t> kvEventTailsOf(const RetractionImage& image) const;
    // The descriptors a restore's ACK mutates when it republishes the image's
    // L2 slots: the whole prefix chain and the image's tails (kvEventTailsOf).
    // A drain dropped them when the victim's pages left the Device.
    void registerKvEventRestore(Request& request, std::span<const std::string> prefix_hashes,
                                std::span<const std::int32_t> tails);
    // The descriptor of `tail` past the full pages `prefix_hashes`.
    void registerKvEventTail(Request& request, std::span<const std::string> prefix_hashes, const TailPublication& tail);
    void markKvEventBoundaryForReconcile(const CacheKey& boundary);
    // Finished requests whose forwards have all landed: publish the frontier
    // (state and KV tail) and release the pages. Runs after every batch of
    // events, so a request that finished quiescent settles with its Finish
    // and one whose finish overtook an overlapped decode settles with that
    // decode's result -- paused or not, plan or no plan.
    void settleFinishedRequests();
    std::optional<WriteBackOperation> publishCompletedPages(Request& request);
    // The dead checkpoints SchedulerConfig::state_checkpoint_in_forward skips
    // (outside_event_handler.cpp): the pre-tail State checkpoint that makes a
    // request's prompt end dead, from the prompt and where the request
    // resumed (zero for none); whether that checkpoint exists, so a
    // publication in place at the prompt end publishes no tail and no state
    // there; and whether a finishing request publishes its frontier like a
    // chunk end.
    std::int32_t promptEndCheckpoint(const Request& request, std::int32_t resume_tokens) const;
    bool promptEndIsCoveredByCheckpoint(const Request& request, const fsm::CacheProgress& progress) const;
    bool finishCheckpointIsDead(const Request& request, const fsm::CacheProgress& progress) const;
    // SchedulerConfig::drop_short_prompt_end_checkpoints at a finish: whether
    // the request's computed frontier is the end of a prompt too short for an
    // in-forward checkpoint, inside a page, which the finish publishes like a
    // chunk end (finishCheckpointIsDead) and which supersedes no input
    // (MarkSupersededStateInput).
    bool shortPromptEndIsDropped(const Request& request) const;
    // Whether the first decode of a request that resumed at resume_tokens
    // publishes its prompt end and keeps writing past it, and the copy-in
    // point every admission of its prompt reserves for that publication
    // (operations/forward.cpp). Both read promptEndCheckpoint, so the
    // admissions reserve copy-in blocks exactly for a prompt end the first
    // decode publishes.
    bool firstDecodePublishesPromptEnd(const Request& request, std::int32_t resume_tokens) const;
    std::optional<CacheCoordinator::CopyInPoint> promptEndCopyInPoint(const Request& request,
                                                                      std::int32_t resume_tokens) const;
    // Superseded checkpoint demotion, at a finish that published an in-forward
    // State checkpoint C past the page boundary R it resumed from, before C
    // is recorded: when R is itself a recorded in-forward checkpoint, demotes
    // the one R's publisher resumed from -- two turns back on this
    // conversation -- with the window blocks behind it that neither R nor C
    // resumes with. R stays, so a next prompt that diverges inside the last
    // turn resumes there. A shared checkpoint stays too: one another first
    // admission resumed at, or one an access dated after R's publisher
    // acquired it. A boundary no in-forward checkpoint was recorded at (a
    // junction or prompt end alone) is never demoted, though it can need a
    // window block it shares with the demoted one. The scheduler sees no
    // conversations, so another conversation's boundary only this one
    // resumed from so far is demoted like its own.
    void demoteSupersededCheckpoint(const fsm::CacheProgress& progress);
    // Records, at a finish, the in-forward State checkpoint the request
    // published past the point it resumed from.
    void recordCheckpointPublication(const fsm::CacheProgress& progress);

    std::size_t groupIndex(const std::string& group_id) const;
    Request* findRequest(const std::string& request_id);
    // A PD transfer is out against this request's pages, so nothing here may
    // retract, finish, flush or free them. Derived from the FSM, never
    // recorded: on the D role it is the peer's prefill writing the
    // destination pages (RemotePrefilling); on the P role it is the peer's
    // decode reading them, which begins with the first scheduled chunk and
    // ends only with the PD ACK -- every page-holding state before Finishing.
    // A request that ends while pinned (an abort, a failed transfer) parks
    // in PdTransferring until the transport reports the transfer stopped. A
    // fused engine never transfers.
    bool pdTransferInFlight(const Request& request) const;
    // Ends a request on an abort or a failed transfer, through PdTransferring
    // while a transfer may still touch its pages.
    void abortRequest(Request& request);

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
    void handleEvent(const pd::DestinationReadyEvent& event);
    void handleEvent(const pd::FailedEvent& event);
    void handleEvent(const pd::TransferStoppedEvent& event);
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
        PlanBuild(ExecutionPlan& output_plan, PlanOptions plan_options) : plan{output_plan}, options{plan_options} {}

        ExecutionPlan& plan;
        PlanOptions options;
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
    // returns nullopt without reporting a capacity failure. `yield_tokens`
    // and `must_complete` as for schedulePrefillFirstChunk (the yield applies
    // to a resumed chunk too).
    std::optional<PrefillOperation> schedulePrefillCandidate(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                             Request* request, std::int32_t token_budget,
                                                             std::int32_t decode_reserve,
                                                             std::vector<LoadBackOperation>& load_backs,
                                                             std::int32_t kv_page_budget, std::int32_t yield_tokens,
                                                             bool must_complete);

    // SchedulerConfig::prefill_chunk_yield_tokens (docs/design/scheduler.md
    // §1.1): a Submitted prompt a chunk that cannot finish may carry in its
    // forward -- its position in the round's candidates, and the tokens
    // (replay included) and KV pages the whole prompt takes.
    struct WholePrompt {
        std::size_t index{0};
        Request* request{nullptr};
        std::int32_t tokens{0};
        std::int32_t pages{0};
    };
    // The tokens a Submitted request's first chunk takes to complete its
    // prompt, replay included; nullopt when that chunk could not complete it
    // whatever the budget -- its prefix waits for an eviction store, a
    // junction cuts it, or it spans its in-forward checkpoint from inside a
    // page -- and, on a model with a State group (`has_state_group`), when it
    // shares the first page its KV lacks with the pages `resident_holder`'s
    // admission is about to publish. A probe only: no KV event is registered
    // and nothing is admitted.
    std::optional<std::int32_t> wholePromptTokens(Request& request, Request* resident_holder, bool has_state_group);
    // The round's whole prompts, probed once: the Submitted candidates from
    // position `from` on that no phase scheduled yet and whose whole prompt
    // takes at most prefill_chunk_yield_tokens, in candidate order, after at
    // most kMaxYieldProbesPerRound probes; `resident_holder` is the resident
    // chunk they are probed for, null for a new prompt's.
    std::vector<WholePrompt> wholePromptsAfter(std::span<Request* const> candidates, std::size_t from,
                                               const PlanBuild& build, Request* resident_holder);
    // The whole prompts a holder carries: from the round's list, those from
    // position `from` on still Submitted and unscheduled, in order, while
    // their tokens fit prefill_chunk_yield_tokens and they and the holder fit
    // the batch, the free request slots (one more for a new holder,
    // `holder_is_new`) and, with max_kv_page_num, the KV page budget beside
    // the holder's most pages (`holder_max_pages`).
    std::vector<WholePrompt> carriedPrompts(std::span<const WholePrompt> whole_prompts, std::size_t from,
                                            const PlanBuild& build, bool holder_is_new,
                                            std::int32_t holder_max_pages) const;
    // Admits the carried prompts behind a holder that kept the head of line
    // (its operation the last pushed), each only if its first chunk completes
    // it on the budget left; stops at the first capacity failure, and admits
    // none once the plan is a lone-request compaction's (the holder's own;
    // last_lone_compaction_plan_). Nothing it admits holds the head of line,
    // and the holder's operation then moves behind theirs: the completing
    // prefills run first, the incomplete one last. candidates supplies the
    // round's FIFO order. With enable_short_prefill_first, a successful
    // carried first admission that overtakes the oldest waiting prompt
    // increments that prompt's bypass count.
    void carryWholePrompts(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates,
                           std::span<const WholePrompt> carried, std::int32_t decode_reserve);

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
    // Lone-request compaction for the admission `retry` repeats, which just
    // failed in its last variant: gathers the request's blocks, records the
    // moves on the plan, stamps the plan as the compaction's and retries
    // once (see forward.cpp).
    template <typename Retry>
    std::optional<CacheCoordinator::AdmissionResult> compactLoneRequestAndRetry(
        ExecutionPlan& plan, const Request& request, CacheCoordinator::PrefixProbe&& prefix,
        std::span<const GroupDemand> demands, const RequestProgress& progress,
        std::span<std::vector<CacheBlockRef>> copy_in_blocks,
        std::optional<CacheCoordinator::CopyInPoint> copy_in_point, const Retry& retry);
    // Publishes a lone request's completed pages ahead of a step that does
    // not fit, so the next plan's compaction can move them, and commits
    // `staged` (see forward.cpp).
    void publishCompletedPagesFirst(ExecutionPlan& plan, Request& request, StagedCacheProgress& staged,
                                    const RequestProgress& progress);
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
    // True when a local-prefill candidate has been held back by the
    // prefill/decode interval for prefill_decode_interval rounds in total
    // (docs/design/scheduler.md §3.3).
    bool prefillDeferralExhausted(std::span<Request* const> candidates) const;
    bool belowPrefillWatermark() const;
    bool prefillAllowed(const Request& request, const PlanBuild& build) const;
    void scheduleLocalPrefillWork(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates,
                                  bool new_prompts_sealed, bool blocked_prompt_seals, std::int32_t decode_reserve);
    void scheduleDecodeBatch(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates);

    SchedulerConfig config_;
    ReqPoolAllocator req_pool_allocator_;
    // One slot per retracted request: the index of its slot-state blob in
    // the runtime's arena, released by the FSM transition that drops the image.
    SnapshotSlotAllocator snapshot_slots_;

    // Pools outlive every CacheBlockRef stored below, and the coordinator
    // every BlockTable it binds (the requests' tables, in requests_ below).
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
    // The plan the last lone-request compaction ran in, or published a
    // request's completed pages first for the next plan's: at most one per
    // plan, whichever admission of the build asks (plan_calls_ stamps it),
    // and never a plan the no-progress abort takes.
    std::int64_t last_lone_compaction_plan_{-1};

    // Stamped onto each retraction; the readmission order lives on the
    // Retracted states themselves (rankedReadmissions).
    std::int64_t next_retraction_epoch_{1};
    // The forced-retraction knob's clock and the request it has armed (empty
    // when none): kept out of every batch until it is quiescent, then
    // retracted.
    std::int64_t plan_calls_{0};
    std::string forced_victim_id_;

    // prefill_decode_interval: reset by a prefill round, counted down by a
    // decode round. Derived from the plans alone, so mirrored ranks agree.
    std::int32_t prefill_deferral_rounds_left_{0};

    // Consecutive stalled rounds (trackStalledRounds); 0 after any round
    // that schedules work, leaves work in flight or aborts a request (the
    // no-progress abort resets it itself, its plan judged stalled before).
    std::int64_t stalled_rounds_{0};

    // Submission order -- the FIFO every scheduling phase walks, identical
    // on every rank because the mirrored schedulers receive identical
    // batches. The unique_ptr keeps each Request's address stable across
    // vector reshuffles, so the id index below never dangles.
    std::vector<std::unique_ptr<Request>> requests_;
    std::unordered_map<std::string, Request*> requests_by_id_;
    // What the prefix index cannot tell us about a boundary: the token
    // descriptor the external event carries, and whether that event is
    // currently out (a BlockStored not yet followed by its BlockRemoved).
    // Residency itself is the coordinator's answer, never mirrored here.
    // Cache mutations and registrations only mark a boundary for reconcile;
    // DrainKvEvents reconciles each marked boundary against its residency, so
    // the order of evictions and publications inside one admission does not
    // matter. After a drain, a descriptor exists exactly for the boundaries
    // with any Device cached child, including State entries excluded from
    // routing residency: their later mutations still need the descriptor.
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
    KvEventPrefixView kv_event_prefix_view_;

    // SchedulerConfig::demote_superseded_state_checkpoints where it takes
    // effect (a model with State groups, outside the D role): what finishes
    // and first admissions record for demoteSupersededCheckpoint. Empty
    // otherwise.
    std::optional<CheckpointLineage> checkpoint_lineage_;
    // DemotedCheckpointEntries.
    std::uint64_t demoted_checkpoint_entries_{0};
    // Decode steps admitted through the coordinator instead of consuming
    // their reserve in place (scheduleDecode). Read by tests only.
    std::uint64_t decode_admissions_{0};
};

}  // namespace tokenspeed
