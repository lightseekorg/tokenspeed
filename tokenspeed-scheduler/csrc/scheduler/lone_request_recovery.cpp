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

#include "scheduler/lone_request_recovery.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "fsm/forward_states.h"
#include "scheduler/operations/admission.h"
#include "scheduler/operations/group_demands.h"
#include "scheduler/operations/progress.h"

namespace tokenspeed {

bool LoneRequestRecovery::alone(const Request& request) const {
    return !OthersHoldPages(requests_, request);
}

std::optional<AdmissionResult> LoneRequestRecovery::record(ExecutionPlan& plan,
                                                           std::optional<AdmissionResult> admission) {
    if (admission) {
        RecordAdmittedPages(plan, cache_group_ids_, *admission);
    }
    return admission;
}

// Lone-request compaction, the last means for a request whose admission
// `retry` repeats just failed in its last variant: the request's blocks of
// packed groups are gathered into fewer parents
// (LoneRequestAdmission::Compact), the moves go on the plan with the
// retry's own pages, the plan is marked as the compaction's (ActedInPlan:
// never twice in one plan, never the no-progress abort's), and the admission
// is retried once. One line logs the moves, what
// clearing their destinations evicted and how the retry went. Nullopt when
// nothing could move, or the retry still fails.
template <typename Retry>
std::optional<AdmissionResult> LoneRequestRecovery::compactAndRetry(
    ExecutionPlan& plan, const Request& request, PrefixProbe&& prefix, std::span<const GroupDemand> demands,
    const RequestProgress& progress, std::span<std::vector<CacheBlockRef>> copy_in_blocks,
    std::optional<CopyInPoint> copy_in_point, const Retry& retry) {
    const std::uint64_t moves_before = admission_.CompactionMoves();
    const std::uint64_t evicted_before = coordinator_.DeviceEvictions().compaction.blocks;
    const std::optional<AdmissionResult> compacted =
        admission_.Compact(std::move(prefix), demands, progress, copy_in_blocks, copy_in_point);
    if (!compacted) {
        return std::nullopt;
    }
    acted_in_plan_ = true;
    RecordAdmittedPages(plan, cache_group_ids_, *compacted);
    std::optional<AdmissionResult> admission = retry();
    spdlog::info(
        "[Scheduler] lone-request compaction: request {} ({} tokens) moved {} blocks and evicted {} cache blocks "
        "beside them; the retry {}",
        request.Id(), request.TokenSize(), admission_.CompactionMoves() - moves_before,
        coordinator_.DeviceEvictions().compaction.blocks - evicted_before, admission ? "admits it" : "still fails");
    return admission;
}

std::optional<AdmissionResult> LoneRequestRecovery::AdmitFirstChunk(ExecutionPlan& plan, const Request& request,
                                                                    PrefixProbe prefix,
                                                                    std::span<const GroupDemand> demands,
                                                                    std::optional<CopyInPoint> copy_in_point) {
    if (!alone(request)) {
        return std::nullopt;
    }
    const auto admit_alone = [&](PrefixProbe&& probe) {
        return admission_.Admit(std::move(probe), demands, RequestProgress{}, /*copy_in_blocks=*/{}, copy_in_point,
                                /*request_access_epoch=*/std::nullopt, request.SessionId(), request.CacheGroupId());
    };
    // The Admit that just failed planned exactly what this plans before the
    // block-by-block pass, and changed nothing.
    std::optional<AdmissionResult> admission = admit_alone(PrefixProbe{prefix});
    // A Submitted request holds no pages, so nothing of its own is in flight;
    // what may scatter is the hit pages, which move with their entries.
    if (!admission && !tier_transfers_.HasLoadBacksInFlight() && !tier_transfers_.HasPinnedStoresInFlight() &&
        !acted_in_plan_) {
        admission =
            compactAndRetry(plan, request, PrefixProbe{prefix}, demands, RequestProgress{},
                            /*copy_in_blocks=*/{}, copy_in_point, [&] { return admit_alone(std::move(prefix)); });
    }
    return admission;
}

std::optional<AdmissionResult> LoneRequestRecovery::AdmitStep(ExecutionPlan& plan, Request& request,
                                                              StagedCacheProgress& staged,
                                                              std::span<const GroupDemand> demands,
                                                              const RequestProgress& progress,
                                                              std::optional<CopyInPoint> copy_in_point) {
    if (!alone(request)) {
        return std::nullopt;
    }
    const StepCopyIns copy_ins = StepCopyInsOf(request, progress, copy_in_point);
    const std::uint64_t access_epoch = request.CacheProgressRef().access_epoch;
    const auto attempt = [&] {
        return record(plan, admission_.Admit(coordinator_.Prober().ProbePrefix(std::span<const std::string>{}), demands,
                                             progress, copy_ins.spent, copy_in_point, access_epoch, /*session_id=*/{},
                                             /*cache_group_id=*/{}));
    };
    // The Admit that just failed planned exactly what this plans before the
    // block-by-block pass, and changed nothing.
    std::optional<AdmissionResult> admission = attempt();
    // Quiescent -- no result, PD transfer, load-back or pinned store out
    // against Device pages -- with blocks leaving one by one still not
    // enough, the request's own blocks of packed groups are gathered into
    // fewer parents and the step is retried once in this plan build. Never
    // twice in one plan.
    if (!admission && request.ResultsInFlight() == 0 && !tier_transfers_.HasLoadBacksInFlight() &&
        !tier_transfers_.HasPinnedStoresInFlight() && !PdTransferInFlight(config_.role, request) && !acted_in_plan_) {
        admission = compactAndRetry(plan, request, coordinator_.Prober().ProbePrefix(std::span<const std::string>{}),
                                    demands, progress, copy_ins.held, copy_in_point, attempt);
        // A page this admission publishes cannot move: the runtime stores it
        // before the plan's copies land, and a published page's store pins
        // its block until the ACK (the gate above), so the move waits for
        // the next plan, with the pages published now.
        if (!admission && progress.completed_pages &&
            progress.completed_pages->first_new_prefix_page <
                static_cast<std::int32_t>(progress.completed_pages->prefix_hashes.size())) {
            publishCompletedPagesFirst(plan, request, staged, progress);
        }
    }
    return admission;
}

std::optional<AdmissionResult> LoneRequestRecovery::Restore(const Request& request, const RetractionImage& image,
                                                            std::span<const GroupDemand> demands,
                                                            std::optional<CopyInPoint> copy_in_point,
                                                            std::uint64_t request_access_epoch) {
    if (!alone(request)) {
        return std::nullopt;
    }
    // A retracted request holds no block to move: what can scatter is the
    // cached pages its restore claims, and every slot copied back fits
    // whenever the image and its reserve fit the pool.
    return admission_.Restore(image, demands, copy_in_point, request_access_epoch);
}

// The corner lone-request compaction cannot reach (AdmitStep): the
// request's scattered written blocks are the very pages its failed step
// publishes, which stay put, and the unwritten reserve beyond them frees no
// parent. Where a retraction publishes the victim's whole computed prefix,
// tail included, and suspends it (publishCompletedPrefix), this publishes the
// completed full pages alone, by a zero-growth admission that needs no block,
// and the request stays as it is: its step waits for the next plan, where
// they are published blocks that move. The tail and the state of a
// junction-cut chunk end go unpublished, as when the copy-in is dropped. The
// plan is the compaction's own (ActedInPlan): the no-progress
// abort (NextExecutionPlan) must not take the plan whose publication is what
// the next plan's compaction moves.
void LoneRequestRecovery::publishCompletedPagesFirst(ExecutionPlan& plan, Request& request, StagedCacheProgress& staged,
                                                     const RequestProgress& progress) {
    RequestProgress pages_only = progress;
    pages_only.completed_pages->tail.reset();
    pages_only.completed_pages->publish_state = false;
    const std::vector<GroupDemand> publish_only =
        MakeGroupDemands(request.BlockTablesRef(), config_.cache_groups, GroupDemand{.extent = DenseGrowth{0}},
                         StateStep{.kind = StateStep::Kind::kKeep});
    // No tail: the copy-in blocks stay held; no growth: none is reserved.
    if (!record(plan,
                coordinator_.Admit(coordinator_.Prober().ProbePrefix(std::span<const std::string>{}), publish_only,
                                   pages_only, /*copy_in_blocks=*/{}, /*copy_in_point=*/std::nullopt,
                                   request.CacheProgressRef().access_epoch, /*session_id=*/{},
                                   /*cache_group_id=*/{}))) {
        return;
    }
    staged.Commit();
    acted_in_plan_ = true;
    spdlog::info(
        "[Scheduler] lone-request compaction: request {} ({} tokens) publishes {} completed pages first; its step "
        "waits for the next plan",
        request.Id(), request.TokenSize(),
        pages_only.completed_pages->prefix_hashes.size() -
            static_cast<std::size_t>(pages_only.completed_pages->first_new_prefix_page));
}

}  // namespace tokenspeed
