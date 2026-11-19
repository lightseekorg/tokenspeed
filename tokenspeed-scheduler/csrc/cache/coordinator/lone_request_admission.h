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

// LoneRequestAdmission: the cache side of the lone-request fallback -- for a
// request no other one holds pages beside, which no release can fund, once
// its ordinary CacheCoordinator::Admit or Restore failed
// (docs/design/scheduler.md §1.4). It works through the coordinator's public
// interface -- AdmitWith with its own planner, RestoreWithoutClaimsIfUnfit,
// and the components it exposes (Store, Evictor, History) -- and
// CacheCoordinator knows nothing of it.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"

namespace tokenspeed {

class LoneRequestAdmission {
public:
    explicit LoneRequestAdmission(CacheCoordinator& coordinator) : coordinator_{coordinator} {}
    LoneRequestAdmission(const LoneRequestAdmission&) = delete;
    LoneRequestAdmission& operator=(const LoneRequestAdmission&) = delete;

    // CacheCoordinator::Admit for a request no other one holds pages beside, which waiting
    // would never fund (docs/design/scheduler.md §1.4), with the block
    // eviction fallback: under write_back_evictions, once whole segments
    // cannot fund the admission, what they passed over leaves block by
    // block, written back like any set. It plans as Admit does up to that
    // point, so after a failed Admit it is the same as having called it
    // instead. Only the last attempt of a step calls it: what the step may
    // drop instead (an in-forward State checkpoint's block) never pays for
    // such an eviction.
    std::optional<AdmissionResult> Admit(PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                         const RequestProgress& progress,
                                         std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                         std::optional<CopyInPoint> copy_in_point,
                                         std::optional<std::uint64_t> request_access_epoch,
                                         const std::string& session_id, const std::string& cache_group_id);
    // CacheCoordinator::Restore for a request no other one holds pages beside: when the claims
    // leave no plan, it plans once more claiming nothing under either
    // policy, since no release will come (docs/design/scheduler.md §1.4).
    // After a failed Restore it is the same as having called it instead.
    std::optional<AdmissionResult> Restore(const RetractionImage& snapshot, std::span<const GroupDemand> demands,
                                           std::optional<CopyInPoint> copy_in_point,
                                           std::uint64_t request_access_epoch);

    // Lone-request compaction, the last means for a request no other one
    // holds pages beside, whose Admit just failed although the
    // bound admits it (docs/design/scheduler.md §1.4):
    // its blocks of every packed, token-spanned group -- and, at a first
    // admission, the Device pages and tail blocks the probe hits, at most one
    // tail block per call -- are gathered into fewer parents, so the parents
    // they leave return to the empty FIFO for the caller's retry in the same
    // plan build (whose copy of a moved tail block reads where it was:
    // RecordAdmittedPages). Nothing runs unless a dense
    // layout of those blocks plus the demand (its copy-in pairs counted)
    // fits the pool. Per group the target is ParentsNeededForBuckets over
    // the blocks alone; every parent holding a block that cannot move (a page
    // this admission publishes, one another holder pins; an expired slot the
    // retry reclaims is not counted) is kept first and counts toward it, then
    // the densest (of equally dense ones, the one with more bucket-bound
    // blocks), and the rest are vacated in the reverse order, a parent whole
    // or not at all, and only when the retry can take
    // its other occupants (a slot it reclaims, cache its block-by-block
    // pass may evict). Every Device block the probe hits, in every group
    // (on the D role each State group's prompt-end checkpoint too), is held
    // through the call, so no eviction the clearing takes along reaches one
    // before the retry acquires it. An unwritten reserve block
    // re-places anywhere in a kept parent and is zeroed (new_page_ids); a
    // written one moves by a Device copy into a slot of its own bucket
    // (pages_to_copy, the contract of the owner's copy-in) -- with none left
    // in the kept parents, one a free block of the request gives up, that
    // block re-placing in its own parent first; the keep order leaves
    // another of the request's blocks in that parent, so it never empties
    // (and unbinds) between the two moves -- and a published
    // one's entry, like a probe block's, moves onto the copy
    // (PrefixCacheIndex::Relocate), its boundary-tree hold with it.
    // copy_in_blocks are the copy-in
    // blocks the request holds (Request::CopyInBlocksRef, every admission's,
    // empty when it holds none), aligned to the slots of its copy-in point:
    // copy_in_point, the one this step's Admit reserves for, else the
    // computed frontier the step publishes. Each is held in its slot's
    // bucket, so the two re-place within that bucket (the copy-in block
    // swapped in copy_in_blocks, not zeroed: the copy-in fills it whole, in
    // this plan when the retry spends it). A kept parent's slot held by a
    // cache entry not in use is cleared first, block by block as the
    // planner's last pass would take it (under write-back recorded as a
    // segment for this plan's drain; DeviceEvictionStats::compaction);
    // other cache occupants stay for the retry's own eviction. Vacating
    // stops once the admission plans. Returns what moved for the plan, or
    // nullopt when nothing could (the dense layout does not fit, or no
    // parent can be vacated). A retracted request holds no block to move:
    // alone, Restore drops the claims that leave no plan instead.
    std::optional<AdmissionResult> Compact(PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                           const RequestProgress& progress,
                                           std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                           std::optional<CopyInPoint> copy_in_point);

    // Lone-request compactions that moved a block (Compact) and
    // the blocks they moved, copied or re-placed. Cumulative, never reset
    // (accounting only).
    std::uint64_t Compactions() const { return compactions_; }
    std::uint64_t CompactionMoves() const { return compaction_moves_; }

private:
    CacheCoordinator& coordinator_;
    std::uint64_t compactions_{0};
    std::uint64_t compaction_moves_{0};
};

}  // namespace tokenspeed
