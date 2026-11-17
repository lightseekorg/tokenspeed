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

// LoneRequestRecovery: lone requests (docs/design/scheduler.md §1.4). A request no other one holds
// pages beside has no release to wait for, so when its admission fails the
// scheduler escalates instead of waiting: what whole segments passed over
// leaves block by block (LoneRequestAdmission::Admit), then its own
// blocks of packed groups are gathered into fewer parents
// (LoneRequestAdmission::Compact) and the admission is retried once in
// the same plan build; a restore drops the claims that leave no plan
// (LoneRequestAdmission::Restore). The ordinary path hands over to the
// entries here right where it failed, never later in the build: an admission
// in between would leave the request no longer alone. Everything a failed
// ordinary attempt dropped stays dropped: an optional part is given up before
// blocks leave one by one.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/coordinator/lone_request_admission.h"
#include "cache/tier/transfer_manager.h"
#include "scheduler/execution_plan.h"
#include "scheduler/request.h"
#include "scheduler/types.h"

namespace tokenspeed {

class StagedCacheProgress;

// What the scheduler does for a request no other one holds pages beside
// once its ordinary admission or restore failed. The scheduler hands over
// through the three entries, each right where its last attempt failed,
// starts each plan (StartPlan), reads ActedInPlan and the compaction counters;
// it knows nothing else of lone requests. Each entry returns nullopt at once
// when another request holds pages. AdmitStep records on the plan the pages
// of what it admits; for AdmitFirstChunk and Restore the caller records them
// as it does for an ordinary admission. A component of the Scheduler: it
// reads the scheduler's coordinator, transfers, requests and configuration
// through references, so it lives as long as the Scheduler that owns it.
class LoneRequestRecovery {
public:
    LoneRequestRecovery(CacheCoordinator& coordinator, const TierTransferManager& tier_transfers,
                        const std::vector<std::unique_ptr<Request>>& requests, const SchedulerConfig& config,
                        const std::vector<std::string>& cache_group_ids)
        : coordinator_{coordinator},
          tier_transfers_{tier_transfers},
          requests_{requests},
          config_{config},
          cache_group_ids_{cache_group_ids},
          admission_{coordinator} {}
    LoneRequestRecovery(const LoneRequestRecovery&) = delete;
    LoneRequestRecovery& operator=(const LoneRequestRecovery&) = delete;

    // A plan build starts: nothing has acted in it yet.
    void StartPlan() { acted_in_plan_ = false; }
    // Whether a compaction ran, or a request's completed pages were published
    // first, in this plan build: the plan then admits nothing more, retracts
    // no request whose blocks its copies fill, and is never the no-progress
    // abort's. At most one compaction runs per plan.
    bool ActedInPlan() const { return acted_in_plan_; }
    // Compactions that moved a block, and the blocks they moved (accounting
    // only; LoneRequestAdmission).
    std::uint64_t Compactions() const { return admission_.Compactions(); }
    std::uint64_t CompactionMoves() const { return admission_.CompactionMoves(); }

    // A first chunk whose last Admit just failed with `prefix` and `demands`:
    // LoneRequestAdmission::Admit, then the compaction and its retry.
    std::optional<AdmissionResult> AdmitFirstChunk(ExecutionPlan& plan, const Request& request, PrefixProbe prefix,
                                                   std::span<const GroupDemand> demands,
                                                   std::optional<CopyInPoint> copy_in_point);
    // A later chunk or a decode whose ordinary admission just failed:
    // LoneRequestAdmission::Admit, then the compaction and its retry, and when nothing
    // admits, the completed pages published first (committing `staged`).
    std::optional<AdmissionResult> AdmitStep(ExecutionPlan& plan, Request& request, StagedCacheProgress& staged,
                                             std::span<const GroupDemand> demands, const RequestProgress& progress,
                                             std::optional<CopyInPoint> copy_in_point);
    // A restore whose CacheCoordinator::Restore just failed:
    // LoneRequestAdmission::Restore.
    std::optional<AdmissionResult> Restore(const Request& request, const RetractionImage& image,
                                           std::span<const GroupDemand> demands,
                                           std::optional<CopyInPoint> copy_in_point,
                                           std::uint64_t request_access_epoch);

private:
    bool alone(const Request& request) const;
    // Records an admission's pages on the plan; returns it.
    std::optional<AdmissionResult> record(ExecutionPlan& plan, std::optional<AdmissionResult> admission);
    // The compaction for the admission `retry` repeats: gathers the request's
    // blocks, records the moves on the plan, marks the plan and retries once.
    template <typename Retry>
    std::optional<AdmissionResult> compactAndRetry(ExecutionPlan& plan, const Request& request, PrefixProbe&& prefix,
                                                   std::span<const GroupDemand> demands,
                                                   const RequestProgress& progress,
                                                   std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                                   std::optional<CopyInPoint> copy_in_point, const Retry& retry);
    // Publishes the request's completed pages ahead of a step that does not
    // fit, so the next plan's compaction can move them, and commits `staged`.
    void publishCompletedPagesFirst(ExecutionPlan& plan, Request& request, StagedCacheProgress& staged,
                                    const RequestProgress& progress);

    CacheCoordinator& coordinator_;
    const TierTransferManager& tier_transfers_;
    const std::vector<std::unique_ptr<Request>>& requests_;
    const SchedulerConfig& config_;
    const std::vector<std::string>& cache_group_ids_;
    // The coordinator's side of it.
    LoneRequestAdmission admission_;
    bool acted_in_plan_{false};
};

}  // namespace tokenspeed
