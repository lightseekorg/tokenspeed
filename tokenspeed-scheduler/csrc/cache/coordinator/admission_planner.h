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

// The admission planner (CacheCoordinator::Admit, Restore and the lone-request
// entries): which cached blocks an admission must evict to fit, planned on
// shadow occupancy without touching the pool. Internal to the coordinator's
// translation units.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "cache/coordinator/admission_candidates.h"
#include "cache/coordinator/cache_coordinator.h"
#include "cache/coordinator/eviction_candidates.h"
#include "cache/coordinator/eviction_plan.h"
#include "cache/coordinator/flat_hash_map.h"
#include "cache/coordinator/group_geometry.h"
#include "cache/coordinator/shadow_occupancy.h"
#include "cache/coordinator/victim_set.h"
#include "cache/core/block_placement_demand.h"
#include "core/profile_span.h"
#include "utils.h"

namespace tokenspeed {

// A copy-in block an admission reserves: the table slot it serves and the
// bucket it takes, the bucket of that slot's block -- for a slot the
// admission's dense growth fills, the bucket the fit chose for the block and
// its copy-in block together (-1 until it has).
struct CopyInReservation {
    std::int32_t slot{0};
    std::int32_t bucket{-1};
};

struct AdmissionPlan {
    PrefixProbe prefix;
    // Own all member storage through commit, after the planner has died.
    // Declared before victims so their views die before the arena does.
    std::unique_ptr<std::pmr::monotonic_buffer_resource> member_resource;
    std::vector<VictimSet> victims;
    // Per group, the copy-in blocks the admission reserves, in slot order
    // (AdmissionPlanner::copyInReservations). Empty without a copy-in point.
    std::vector<std::vector<CopyInReservation>> copy_ins;
};

// The table shape a restore rebuilds before its dense demand is placed: how
// many blocks each bucket must receive, the block count and the tail capacity
// the rebuilt table then has, and the rebuilt slots among the copy-in slots
// with the buckets they are rebuilt in. An admission plans against the live
// table instead and passes none.
struct PlannedShape {
    std::vector<std::int64_t> blocks_by_bucket;
    std::int32_t num_blocks{0};
    std::int32_t available_tokens{0};
    std::vector<CopyInReservation> rebuilt_copy_ins;
};

// The table slots a tail ending at `tokens` occupies in a token-spanned
// group: the blocks of the page below the boundary, [first, past_end), and
// the capacity the last of them has left.
struct TailSlots {
    std::int32_t first{0};
    std::int32_t past_end{0};
    std::int32_t available_tokens{0};
};

inline TailSlots tailSlots(const GroupGeometry& geometry, std::int32_t prefix_granularity, std::int32_t tokens) {
    const std::int32_t block_granularity = geometry.BlockGranularity();
    const std::int32_t page_tokens = tokens - tokens % prefix_granularity;
    const std::int32_t past_end = CeilDiv(tokens, block_granularity);
    return TailSlots{
        .first = page_tokens / block_granularity,
        .past_end = past_end,
        .available_tokens = static_cast<std::int32_t>(std::int64_t{past_end} * block_granularity - tokens)};
}

// A plan's mutable working set. It never outlives the pool or result resource.
// Planner methods receive it explicitly; request inputs remain read-only.
struct AdmissionSelectionState {
    AdmissionSelectionState(const CacheStore& store, std::pmr::memory_resource* resource)
        : member_resource{resource}, eviction_plan{store.DevicePool(), store.Groups()} {}

    std::pmr::memory_resource* const member_resource;
    std::vector<VictimSet> victims;
    std::size_t next_passed_over{0};
    // The plan's lookup tables, open-addressed in this arena (FlatHashMap),
    // which frees nothing until this working set dies; returned victim vectors
    // own their storage independently. No table is iterated, so no plan
    // depends on their slot order. The arena starts in this object's raw
    // buffer (left uninitialized: only the arena writes it), so a small
    // admission calls no allocator for its tables.
    static constexpr std::size_t kInlineScratchBytes = 16 * 1024;
    alignas(std::max_align_t) std::array<std::byte, kInlineScratchBytes> scratch_buffer;
    std::pmr::monotonic_buffer_resource scratch{scratch_buffer.data(), scratch_buffer.size()};
    // The pool as the tentative victims would leave it.
    CacheEvictionPlan eviction_plan;
    // Every block a selected set holds: the set it joined (or last moved
    // into) and its position there. A later set may since have taken that
    // set's lists over in place, positions unchanged: taken_by leads there.
    FlatHashMap<CacheBlockLocation, std::pair<std::size_t, std::size_t>> set_member_by_location{&scratch};
    // Per set: the later set that took its lists over in place, else itself.
    std::vector<std::size_t> taken_by;
    // The selected segments by the tree node each starts at: every Device block
    // of that node's subtree, in post-order.
    FlatHashMap<const BoundaryTree::Node*, std::size_t> segment_at{&scratch};
    // Blocks that expand to a set found unable to leave under the current
    // reclaim selection: skipped until a retry invalidates that decision.
    FlatHashSet<CacheBlockLocation> unevictable_locations{&scratch};
    // Pages two or more Device checkpoint boundaries depend on, passed over
    // while candidates last, oldest first: tried again once they run out.
    std::vector<CacheEvictionCandidates::Candidate> deferred_shared_pages;
    // Candidates no segment rule let leave (a set member in use or kept, or a
    // shared page still shared once the candidates ran out), in the order
    // they were passed over: retried as segments after reclaim selection
    // changes, then block by block if the admission permits it.
    std::vector<CacheEvictionCandidates::Candidate> passed_over;
    // Per tree node: how many checkpoint boundaries at or under it have every
    // Device checkpoint in a selected set, counted once the candidates run out.
    FlatHashMap<const BoundaryTree::Node*, std::int32_t> leaving_checkpoint_boundary_counts{&scratch};
};

struct AdmissionCapacity {
    std::vector<BlockPlacementDemand> block_demands;
    std::vector<std::vector<CopyInReservation>> copy_ins;
};

struct AdmissionDecisions {
    std::vector<VictimSet> victims;
    std::vector<std::vector<CopyInReservation>> copy_ins;
};

// Immutable request view. Each Plan call owns its working set and returns decisions.
class AdmissionPlanner {
public:
    // device_aging: the Device checkpoint aging candidates rank by
    // (AdmissionHistory::CheckpointAgingFor).
    AdmissionPlanner(const CacheStore& store, CheckpointAging device_aging, std::span<const GroupDemand> demands,
                     std::optional<std::int32_t> num_computed_tokens, std::optional<std::int32_t> copy_in_point,
                     const PrefixProbe& prefix, std::span<const PlannedShape> shapes,
                     std::span<const std::pair<std::uint32_t, CacheBlockLocation>> planned_claim_locations,
                     bool evicts_segments)
        : store_{store},
          device_aging_{device_aging},
          groups_{store.Groups()},
          geometry_{store.Geometry()},
          prefix_granularity_{store.PrefixGranularity()},
          pool_{store.DevicePool()},
          host_pool_{store.HostPool()},
          demands_{demands},
          num_computed_tokens_{num_computed_tokens},
          copy_in_point_{copy_in_point},
          prefix_{prefix},
          shapes_{shapes},
          planned_claim_locations_{planned_claim_locations},
          boundary_tree_{store.Boundaries()},
          evicts_segments_{evicts_segments} {}

    // The returned victim members use result_resource, which outlives the result.
    std::optional<AdmissionDecisions> Plan(std::pmr::memory_resource* result_resource) const {
        AdmissionCapacity capacity = buildCapacity();
        AdmissionSelectionState selection{store_, result_resource};
        if (!fitWithEviction(capacity.block_demands, selection)) {
            return std::nullopt;
        }
        // Assign the pairs against the space left by the final victim set.
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (capacity.block_demands[i].same_bucket_pairs == 0) continue;
            const auto pair_buckets =
                AssignPairBuckets(capacity.block_demands[i], selection.eviction_plan.HolesByBucket(i));
            auto next_pair = pair_buckets.begin();
            for (CopyInReservation& copy_in : capacity.copy_ins[i]) {
                if (copy_in.bucket < 0) copy_in.bucket = *next_pair++;
            }
            _assert(next_pair == pair_buckets.end(), "every pair of the fit has a copy-in slot");
        }
        return AdmissionDecisions{.victims = std::move(selection.victims), .copy_ins = std::move(capacity.copy_ins)};
    }

    virtual ~AdmissionPlanner() = default;

protected:
    using VictimCandidate = CacheEvictionCandidates::Candidate;
    using NodeEntry = CacheStore::NodeEntry;

    bool fitWithEviction(std::span<const BlockPlacementDemand> block_demands,
                         AdmissionSelectionState& selection) const {
        ProfileSpan profile_span{"AdmissionPlanner::fitWithEviction"};
        // Existing local holes plus empty parents are the zero-eviction
        // capacity. Do not discard cache merely to make placement denser: a
        // request's own scattered blocks are gathered by its compaction
        // (LoneRequestAdmission::Compact) once it is alone and nothing
        // else funds it, never here.
        if (selection.eviction_plan.Fits(block_demands)) {
            return true;
        }
        if (!couldFitAfterEviction(block_demands)) {
            return false;
        }

        const AdmissionCandidateConstraints constraints = BuildAdmissionCandidateConstraints(
            store_, demands_, num_computed_tokens_, prefix_, planned_claim_locations_, &selection.scratch);
        if (const std::optional<std::vector<VictimCandidate>> through_fit =
                candidatesThroughParentFit(block_demands, constraints)) {
            selectThroughParentFit(*through_fit, block_demands, constraints, selection);
            selection.victims = selection.eviction_plan.Minimize(std::move(selection.victims), block_demands);
            return true;
        }
        CacheEvictionCandidates candidates{groups_, pool_, device_aging_};
        if (!couldFitAfterReleasingCandidates(candidates, constraints, block_demands)) {
            return false;
        }
        // Room for the blocks the free slots lack, about what the selected
        // sets hold: a hint only, so the table rarely regrows.
        std::int64_t shortfall = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::int64_t free_slots = pool_.NumFreeSlots(static_cast<std::uint32_t>(i));
            shortfall += std::max<std::int64_t>(block_demands[i].TotalBlocks() - free_slots, 0);
        }
        selection.set_member_by_location.reserve(static_cast<std::size_t>(shortfall));
        bool candidates_ran_out = false;
        std::size_t next_deferred_shared_page = 0;
        // Once every source of the segment rules ran out: a subclass's own
        // candidates (nextCandidatePastSegments).
        bool past_segments = false;
        bool reclaim_selection_changed = false;
        std::vector<VictimCandidate> retry_segments;
        std::size_t next_retry_segment = 0;
        const bool fits = selection.eviction_plan.SelectUntilFits(block_demands, [&] {
            while (true) {
                std::optional<VictimCandidate> candidate = candidates.Next(
                    constraints.request_reclaim_candidates, constraints.protected_locations,
                    constraints.request_reclaim_locations,
                    [&](CacheBlockLocation location) { return selection.set_member_by_location.contains(location); });
                if (!candidate && next_deferred_shared_page < selection.deferred_shared_pages.size()) {
                    // The candidates ran out before the demand fits, and pages two
                    // or more checkpoint boundaries share are still in place. Count
                    // the checkpoint boundaries the selected sets take whole: a
                    // page only they depend on may leave now, with everything under
                    // it.
                    if (!candidates_ran_out) {
                        candidates_ran_out = true;
                        FlatHashSet<const BoundaryTree::Node*> counted{&selection.scratch};
                        for (const VictimSet& set : selection.victims) {
                            for (const VictimSet::Member& member : set.members) {
                                if (!store_.GroupIsState(member.group_id)) {
                                    continue;
                                }
                                const std::optional<CacheKey> key =
                                    groups_[member.group_id].Index().KeyOf(pool_, member.location);
                                const BoundaryTree::Node* checkpoint_node =
                                    key ? boundary_tree_.Find(key->content_hash) : nullptr;
                                if (checkpoint_node == nullptr || !counted.insert(checkpoint_node) ||
                                    !std::ranges::all_of(
                                        store_.NodeEntries(pool_, *checkpoint_node), [&](const NodeEntry& entry) {
                                            return !store_.GroupIsState(entry.group_id) ||
                                                   selection.set_member_by_location.contains(entry.location);
                                        })) {
                                    continue;
                                }
                                for (const BoundaryTree::Node* ancestor = checkpoint_node; ancestor != nullptr;
                                     ancestor = ancestor->parent) {
                                    ++selection.leaving_checkpoint_boundary_counts[ancestor];
                                }
                            }
                        }
                    }
                    candidate = selection.deferred_shared_pages[next_deferred_shared_page++];
                }
                if (!candidate && !past_segments && next_retry_segment == retry_segments.size() &&
                    reclaim_selection_changed) {
                    // A reclaim selected after a rejected segment can make that
                    // segment evictable. Retry a fixed batch before block fallback;
                    // another pass requires a newly selected reclaim block.
                    reclaim_selection_changed = false;
                    selection.unevictable_locations.clear();
                    retry_segments = std::exchange(selection.passed_over, {});
                    next_retry_segment = 0;
                }
                if (!candidate && next_retry_segment < retry_segments.size()) {
                    candidate = retry_segments[next_retry_segment++];
                }
                if (!candidate) {
                    candidate = nextCandidatePastSegments(selection);
                    past_segments = past_segments || candidate.has_value();
                }
                if (!candidate) {
                    return false;
                }
                std::optional<ExpandedVictimSet> expanded =
                    past_segments ? expandCandidatePastSegments(*candidate, constraints, selection)
                                  : expandVictimSet(*candidate, candidates_ran_out, constraints, selection);
                if (!expanded) {
                    continue;
                }
                reclaim_selection_changed =
                    takeVictimSet(std::move(*expanded), candidate->location, selection) || reclaim_selection_changed;
                return true;
            }
        });

        if (!fits) return false;
        selection.victims = selection.eviction_plan.Minimize(std::move(selection.victims), block_demands);
        return true;
    }

    // What an expansion returns beside the set: how many of its first members
    // it took over in place from an earlier segment (their
    // selection.set_member_by_location entries as they are) and that set's index, and
    // the tree node a segment set starts at (nullptr for any other set).
    struct ExpandedVictimSet {
        VictimSet set;
        std::size_t num_taken_over_members{0};
        std::optional<std::size_t> taken_over_set_index;
        const BoundaryTree::Node* segment_top{nullptr};
    };

    // Takes the set chosen for the candidate at `candidate` into the
    // selection. An earlier set it contains moves into it whole, each block
    // keeping why it left; the rest leave the shadow now. The lists taken
    // over in place are there already. True when a request's own reclaim
    // left with it.
    bool takeVictimSet(ExpandedVictimSet&& expanded, CacheBlockLocation candidate,
                       AdmissionSelectionState& selection) const {
        auto& [set, num_taken_over_members, taken_over_set_index, segment_top] = expanded;
        _assert(std::ranges::find(set.members.begin() + static_cast<std::ptrdiff_t>(num_taken_over_members),
                                  set.members.end(), candidate, &VictimSet::Member::location) != set.members.end(),
                "a victim set holds the candidate it was chosen for");
        bool takes_request_reclaim = false;
        const std::size_t set_index = selection.victims.size();
        std::vector<std::size_t> absorbed;
        std::size_t absorbed_members = 0;
        for (std::size_t j = num_taken_over_members; j < set.members.size(); ++j) {
            VictimSet::Member& member = set.members[j];
            const auto [joined, inserted] =
                selection.set_member_by_location.TryEmplace(member.location, std::pair{set_index, j});
            if (inserted) {
                selection.eviction_plan.Remove(member.group_id, member.location);
                takes_request_reclaim = takes_request_reclaim || member.is_request_reclaim;
                continue;
            }
            // The set holding the block now, through the sets that took
            // lists over.
            std::size_t earlier = joined->first;
            while (selection.taken_by[earlier] != earlier) {
                earlier = selection.taken_by[earlier] = selection.taken_by[selection.taken_by[earlier]];
            }
            const std::size_t position = joined->second;
            member = selection.victims[earlier].members[position];
            if (std::ranges::find(absorbed, earlier) == absorbed.end()) {
                absorbed.push_back(earlier);
            }
            ++absorbed_members;
            *joined = std::pair{set_index, j};
        }
        for (const std::size_t earlier : absorbed) {
            absorbed_members -= selection.victims[earlier].members.size();
            selection.victims[earlier] = VictimSet{selection.member_resource};
        }
        _assert(absorbed_members == 0, "an earlier victim set lies whole inside the set that takes it");
        if (taken_over_set_index) {
            selection.taken_by[*taken_over_set_index] = set_index;
        }
        selection.victims.push_back(std::move(set));
        selection.taken_by.push_back(set_index);
        if (segment_top != nullptr) {
            selection.segment_at[segment_top] = set_index;
        }
        return takes_request_reclaim;
    }

    // Once the segment rules and their retries have nothing left to offer, a
    // subclass may go on with candidates of its own, expanded its own way
    // (LoneRequestPlanner, lone_request_admission.cpp). This planner stops there.
    virtual std::optional<VictimCandidate> nextCandidatePastSegments(AdmissionSelectionState& selection) const {
        return std::nullopt;
    }
    virtual std::optional<ExpandedVictimSet> expandCandidatePastSegments(
        const VictimCandidate& candidate, const AdmissionCandidateConstraints& constraints,
        AdmissionSelectionState& selection) const {
        _assert(false, "a planner that offers no candidates past the segment rules expands none");
        return std::nullopt;
    }

    // Whether a block may join a set: one a selected set holds already, or
    // one nothing uses, keeps or reclaims. A selected request reclaim still
    // has its table reference until Admit executes. A later segment may
    // absorb that set; the merge in fitWithEviction keeps its capacity
    // counted once and Minimize must restore them together.
    bool isEvictable(const NodeEntry& entry, const AdmissionCandidateConstraints& constraints,
                     const AdmissionSelectionState& selection) const {
        return selection.set_member_by_location.contains(entry.location) ||
               (!entry.in_use && !constraints.protected_locations.contains(entry.location) &&
                !constraints.request_reclaim_locations.contains(entry.location));
    }

    // The Device checkpoint boundaries at or under a node that no selected
    // set takes whole: every one while candidates last, then those staying.
    std::int32_t numStayingCheckpointBoundaries(const BoundaryTree::Node& node,
                                                const AdmissionSelectionState& selection) const {
        const std::int32_t* leaving = selection.leaving_checkpoint_boundary_counts.Find(&node);
        return node.checkpoint_boundaries_in_subtree[BoundaryTree::kDevice] - (leaving == nullptr ? 0 : *leaving);
    }

    // The tree node a candidate's set follows when evicting by segments;
    // nullptr otherwise, or for a block with no tree node (never published,
    // or registered straight into an index).
    const BoundaryTree::Node* candidateNode(const VictimCandidate& candidate) const {
        if (!evicts_segments_) {
            return nullptr;
        }
        const std::optional<CacheKey> key = groups_[candidate.group_id].Index().KeyOf(pool_, candidate.location);
        return key ? boundary_tree_.Find(key->content_hash) : nullptr;
    }

    // Return the sets every rule expands alike: a block that leaves
    // alone (no tree node to follow, or a window block, which only shortens
    // the boundaries its prefix resumes at, unless reclaimed) and the
    // request's own reclaim (its released slots at this hash, and at a tail
    // the tail's other blocks, which their eviction takes along). Nullopt for
    // any other candidate.
    std::optional<VictimSet> expandAloneOrReclaim(const VictimCandidate& candidate,
                                                  const BoundaryTree::Node* candidate_node,
                                                  const AdmissionCandidateConstraints& constraints,
                                                  const AdmissionSelectionState& selection) const {
        VictimSet set{selection.member_resource};
        const bool is_request_reclaim = constraints.request_reclaim_locations.contains(candidate.location);
        const bool sliding =
            !groups_[candidate.group_id].Matcher().IsPrefixClosed() && !store_.GroupIsState(candidate.group_id);
        if (candidate_node == nullptr || (sliding && !is_request_reclaim)) {
            set.members.push_back(VictimSet::Member{.group_id = candidate.group_id,
                                                    .location = candidate.location,
                                                    .in_stranded_suffix = false,
                                                    .is_request_reclaim = is_request_reclaim});
            return set;
        }
        if (!is_request_reclaim) {
            return std::nullopt;
        }
        for (const NodeEntry& entry : store_.NodeEntries(pool_, *candidate_node)) {
            if (constraints.request_reclaim_locations.contains(entry.location) ||
                (candidate_node->tail_tokens > 0 && isEvictable(entry, constraints, selection))) {
                set.members.push_back(VictimSet::Member{.group_id = entry.group_id,
                                                        .location = entry.location,
                                                        .in_stranded_suffix = false,
                                                        .is_request_reclaim = true});
            }
        }
        return set;
    }

    // Return every State group's checkpoint at `node`, leaving alone: at
    // a tail its eviction takes the tail's other unpinned blocks along
    // (CacheEvictor::EvictCachedBlock), uncounted here. Nullopt, each remembered as unable
    // to leave, when one is in use or kept.
    std::optional<VictimSet> expandCheckpointsAlone(const BoundaryTree::Node& node,
                                                    const AdmissionCandidateConstraints& constraints,
                                                    AdmissionSelectionState& selection) const {
        VictimSet set{selection.member_resource};
        bool evictable = true;
        for (const NodeEntry& entry : store_.NodeEntries(pool_, node)) {
            if (store_.GroupIsState(entry.group_id)) {
                evictable = evictable && isEvictable(entry, constraints, selection);
                set.members.push_back(VictimSet::Member{.group_id = entry.group_id,
                                                        .location = entry.location,
                                                        .in_stranded_suffix = false,
                                                        .is_request_reclaim = false});
            }
        }
        if (!evictable) {
            for (const VictimSet::Member& member : set.members) {
                selection.unevictable_locations.insert(member.location);
            }
            return std::nullopt;
        }
        return set;
    }

    // Return the Device subtree under `top` in post-order: a node after
    // the nodes under it, its own entries in NodeEntries order --
    // `candidate_entries` for the candidate's node, `climbed` (in climb
    // order) for the nodes a segment climbed to above it. A child the Device
    // does not hold has nothing on the Device under it when the model has a
    // prefix-closed group (eviction keeps the Device's coverage closed from
    // the root), so the walk skips the Host's subtrees; without one, what
    // such a child keeps under it on the Device needs none of these pages
    // either. A page a restore republished below such a child stays and
    // leaves later as a segment of its own. A block under the candidate's
    // node is in its stranded suffix. Nullopt at the first entry that cannot
    // leave, unless it is at `top` and `top_keeps_unavailable`: it stays.
    std::optional<ExpandedVictimSet> walkSubtree(
        const BoundaryTree::Node& top, const BoundaryTree::Node& candidate_node,
        std::span<const NodeEntry> candidate_entries,
        std::span<const std::pair<const BoundaryTree::Node*, std::vector<NodeEntry>>> climbed,
        bool top_keeps_unavailable, const AdmissionCandidateConstraints& constraints,
        AdmissionSelectionState& selection) const {
        ExpandedVictimSet out{.set = VictimSet{selection.member_resource}};
        struct Frame {
            const BoundaryTree::Node* node;
            const BoundaryTree::Node* next_child;
            bool in_stranded_suffix;
        };
        VictimSet& set = out.set;
        std::vector<Frame> stack{Frame{.node = &top, .next_child = top.first_child, .in_stranded_suffix = false}};
        std::size_t next_climbed = 0;
        // A child an earlier segment starts at: that segment holds the
        // child's whole Device subtree in this order already, so it goes
        // where the walk reaches it, after the blocks walked so far, and is
        // not walked again (a chain reclaimed page by page from its suffix
        // would otherwise be walked once per page).
        std::vector<std::pair<std::size_t, const BoundaryTree::Node*>> spliced;
        while (!stack.empty()) {
            Frame& frame = stack.back();
            if (const BoundaryTree::Node* child = frame.next_child) {
                frame.next_child = child->next_sibling;
                if (selection.segment_at.contains(child)) {
                    spliced.emplace_back(set.members.size(), child);
                } else if (child->OnDevice()) {
                    const bool in_stranded_suffix = frame.in_stranded_suffix || frame.node == &candidate_node;
                    stack.push_back(Frame{
                        .node = child, .next_child = child->first_child, .in_stranded_suffix = in_stranded_suffix});
                }
                continue;
            }
            const bool climbed_node = next_climbed < climbed.size() && climbed[next_climbed].first == frame.node;
            std::vector<NodeEntry> subtree_entries;
            std::span<const NodeEntry> entries;
            if (climbed_node) {
                entries = climbed[next_climbed++].second;
            } else if (frame.node == &candidate_node) {
                entries = candidate_entries;
            } else {
                subtree_entries = store_.NodeEntries(pool_, *frame.node);
                entries = subtree_entries;
            }
            for (const NodeEntry& entry : entries) {
                if (!isEvictable(entry, constraints, selection)) {
                    if (top_keeps_unavailable && frame.node == &top) {
                        continue;
                    }
                    return std::nullopt;
                }
                set.members.push_back(VictimSet::Member{.group_id = entry.group_id,
                                                        .location = entry.location,
                                                        .in_stranded_suffix = frame.in_stranded_suffix,
                                                        .is_request_reclaim = false});
            }
            stack.pop_back();
        }
        if (!spliced.empty()) {
            VictimSet walked = std::exchange(set, VictimSet{selection.member_resource});
            std::size_t next_walked = 0;
            for (const auto& [walked_before, child] : spliced) {
                set.members.insert(set.members.end(), walked.members.begin() + next_walked,
                                   walked.members.begin() + walked_before);
                next_walked = walked_before;
                const std::size_t earlier_index = *selection.segment_at.Find(child);
                VictimSet& earlier = selection.victims[earlier_index];
                _assert(!earlier.members.empty(), "a segment set still starts at its top");
                if (set.members.empty()) {
                    // First in the order: its lists become this set's.
                    set = std::exchange(earlier, VictimSet{selection.member_resource});
                    out.taken_over_set_index = earlier_index;
                    out.num_taken_over_members = set.members.size();
                } else {
                    // Moved in below like any other earlier set it holds.
                    set.members.insert(set.members.end(), earlier.members.begin(), earlier.members.end());
                }
                selection.segment_at.Erase(child);
            }
            set.members.insert(set.members.end(), walked.members.begin() + next_walked, walked.members.end());
        }
        return out;
    }

    // One candidate expanded to the set it leaves with under the segment
    // rules -- as a segment, alone, or as the request's own reclaim -- or
    // nullopt when it is skipped this pass (a block in the set is in use or
    // kept; the shared page is deferred). Pure apart from the pass's own memo
    // lists (selection.passed_over, selection.deferred_shared_pages, selection.unevictable_locations)
    // and, for a segment, the earlier sets it takes over in place.
    std::optional<ExpandedVictimSet> expandVictimSet(const VictimCandidate& candidate, bool candidates_ran_out,
                                                     const AdmissionCandidateConstraints& constraints,
                                                     AdmissionSelectionState& selection) const {
        // Already leaving with an earlier set, or found unable to leave
        // (passed over).
        if (selection.set_member_by_location.contains(candidate.location)) {
            return std::nullopt;
        }
        if (selection.unevictable_locations.contains(candidate.location)) {
            selection.passed_over.push_back(candidate);
            return std::nullopt;
        }
        const BoundaryTree::Node* candidate_node = candidateNode(candidate);
        if (auto set = expandAloneOrReclaim(candidate, candidate_node, constraints, selection)) {
            return ExpandedVictimSet{.set = std::move(*set)};
        }
        if (numStayingCheckpointBoundaries(*candidate_node, selection) >= 2) {
            // Deeper checkpoint boundaries still read this page. A nested
            // checkpoint leaves alone, every State group's at once; the page
            // leaves with the last checkpoint boundary under it, or is tried
            // again once the candidates run out.
            if (!store_.GroupIsState(candidate.group_id)) {
                (candidates_ran_out ? selection.passed_over : selection.deferred_shared_pages).push_back(candidate);
                return std::nullopt;
            }
            auto set = expandCheckpointsAlone(*candidate_node, constraints, selection);
            if (!set) return std::nullopt;
            return ExpandedVictimSet{.set = std::move(*set)};
        }
        // A segment: the one checkpoint boundary at or under this boundary
        // with the pages only it depends on -- the climb stops below a page
        // another checkpoint boundary shares, one in use or a hit this
        // admission keeps, or one the Device does not hold -- and everything
        // the Device holds under them, so no Device page outlives its parent.
        // The climb keeps the entries of each node it passes, from the
        // candidate's parent up.
        const BoundaryTree::Node* top = candidate_node;
        const std::vector<NodeEntry> candidate_entries = store_.NodeEntries(pool_, *candidate_node);
        std::vector<std::pair<const BoundaryTree::Node*, std::vector<NodeEntry>>> climbed;
        if (numStayingCheckpointBoundaries(*candidate_node, selection) == 1) {
            while (top->parent != nullptr && top->parent->OnDevice() &&
                   numStayingCheckpointBoundaries(*top->parent, selection) == 1) {
                std::vector<NodeEntry> entries = store_.NodeEntries(pool_, *top->parent);
                if (!std::ranges::all_of(entries, [this, &constraints, &selection](const NodeEntry& entry) {
                        return isEvictable(entry, constraints, selection);
                    })) {
                    break;
                }
                top = top->parent;
                climbed.emplace_back(top, std::move(entries));
            }
        }
        auto expanded = walkSubtree(*top, *candidate_node, candidate_entries, climbed,
                                    /*top_keeps_unavailable=*/false, constraints, selection);
        if (!expanded) {
            selection.passed_over.push_back(candidate);
            // One unavailable member rejects this whole set; there is no
            // need to inspect the rest of its subtree. Remember only
            // candidates that would expand to the same set. A descendant or
            // sibling can still leave independently.
            const auto remember_same_set = [&](std::span<const NodeEntry> entries) {
                for (const NodeEntry& entry : entries) {
                    if (!constraints.request_reclaim_locations.contains(entry.location) &&
                        (groups_[entry.group_id].Matcher().IsPrefixClosed() || store_.GroupIsState(entry.group_id))) {
                        selection.unevictable_locations.insert(entry.location);
                    }
                }
            };
            remember_same_set(candidate_entries);
            for (const auto& [node, entries] : climbed) {
                remember_same_set(entries);
            }
            return std::nullopt;
        }
        expanded->segment_top = top;
        return expanded;
    }

    struct PrefixAllocation {
        // Tail capacity can reduce the blocks charged by the ordinary growth.
        std::int64_t any_bucket_adjustment{0};
        std::vector<std::int64_t> blocks_by_bucket;
        std::vector<CopyInReservation> tail_slots;
    };

    AdmissionCapacity buildCapacity() const {
        _assert(demands_.size() == groups_.size(), "demands/groups size mismatch");
        _assert(shapes_.empty() || shapes_.size() == groups_.size(), "shapes/groups size mismatch");
        AdmissionCapacity capacity{
            .block_demands = std::vector<BlockPlacementDemand>(groups_.size()),
            .copy_ins = std::vector<std::vector<CopyInReservation>>(copy_in_point_ ? groups_.size() : 0)};
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const GroupDemand& demand = demands_[i];
            _assert(demand.table != nullptr, "group demand requires a block table");
            BlockPlacementDemand& block_demand = capacity.block_demands[i];
            std::vector<CopyInReservation> prefix_tail_slots;
            const auto buckets = static_cast<std::size_t>(groups_[i].Allocator().ShardCount());
            if (shapes_.empty()) {
                std::int32_t growth_blocks = geometry_[i].BlocksNeededFor(*demand.table, demand);
                if (const auto* sparse = std::get_if<SparseSuffix>(&demand.extent); sparse && sparse->head_blocks > 0) {
                    // A matched sink is installed before growth. Host loads
                    // and Device tail copies are charged in their buckets below.
                    const auto prefix_tokens =
                        std::max(prefix_.device.num_common_tokens, prefix_.host.num_common_tokens);
                    const auto grain = geometry_[i].BlockGranularity();
                    const auto prefix_blocks = prefix_tokens / grain + (prefix_tokens % grain != 0);
                    growth_blocks = geometry_[i].BlocksNeededFor(std::max(demand.table->NumBlocks(), prefix_blocks),
                                                                 *sparse, demand.reserve_tokens);
                }
                PrefixAllocation tail = tailAllocation(i);
                block_demand.blocks_by_bucket = std::move(tail.blocks_by_bucket);
                block_demand.any_bucket_blocks = growth_blocks + tail.any_bucket_adjustment + zeroStateBlocksNeeded(i);
                prefix_tail_slots = std::move(tail.tail_slots);
                if (!prefix_.host.per_group.empty()) {
                    const PrefixAllocation host = hostHitAllocation(i);
                    for (std::size_t b = 0; b < buckets; ++b)
                        block_demand.blocks_by_bucket[b] += host.blocks_by_bucket[b];
                    prefix_tail_slots.insert(prefix_tail_slots.end(), host.tail_slots.begin(), host.tail_slots.end());
                }
            } else {
                // Rebuild the recorded buckets before growing from their tail.
                _assert(demand.table->NumBlocks() == 0, "a restore rebuilds into an empty table");
                _assert(shapes_[i].blocks_by_bucket.size() == buckets,
                        "restore shape buckets must match the shard count");
                if (const auto* step = std::get_if<StateStep>(&demand.extent)) {
                    _assert(step->kind == StateStep::Kind::kKeep, "a restore rebuilds a State table as imaged");
                } else {
                    const auto* dense = std::get_if<DenseGrowth>(&demand.extent);
                    _assert(dense != nullptr, "a restore demand grows the rebuilt table densely");
                    block_demand.any_bucket_blocks = geometry_[i].BlocksNeededBeyond(
                        shapes_[i].available_tokens, dense->num_tokens + demand.reserve_tokens);
                }
                block_demand.blocks_by_bucket = shapes_[i].blocks_by_bucket;
            }
            if (copy_in_point_) {
                capacity.copy_ins[i] = copyInReservations(i, prefix_tail_slots);
                for (const CopyInReservation& copy_in : capacity.copy_ins[i]) {
                    if (copy_in.bucket < 0) {
                        // The growing block now belongs to a pair, counted once.
                        ++block_demand.same_bucket_pairs;
                        --block_demand.any_bucket_blocks;
                    } else {
                        ++block_demand.blocks_by_bucket[static_cast<std::size_t>(copy_in.bucket)];
                    }
                }
            }
        }
        return capacity;
    }

    // Blocks a tail costs this group beyond the growth demand. A table never
    // keeps a tail entry's block: the block holds the rows of a page that is
    // later published under the page's own hash, and one block carries one
    // key. So on a first admission every block of a Device tail becomes the
    // hitter's own (a copy of the shared block, or a fresh block where the
    // probe left a hole). A copy takes its fresh block in its source's bucket
    // (SwapInFreshBlock), so it is counted by bucket like a Host hit and
    // recorded for its slot (PrefixAllocation::tail_slots); a hole's fresh block takes
    // any bucket and is returned with the growth. A dense growth starts from
    // the capacity the last tail block leaves, which the live table cannot
    // show yet; a sparse suffix (the D role's landing of a sliding window)
    // starts at an absolute block past the tail and is costed whole by
    // BlocksNeededFor.
    PrefixAllocation tailAllocation(std::size_t i) const {
        PrefixAllocation allocation{.blocks_by_bucket =
                                        std::vector<std::int64_t>(groups_[i].Allocator().ShardCount(), 0)};
        if (!geometry_[i].HasTokenSpan()) {
            return allocation;  // a State group's step is costed by BlocksNeededFor
        }
        const std::int32_t block_granularity = geometry_[i].BlockGranularity();
        const GroupAllocator& allocator = groups_[i].Allocator();
        if (const auto& tail = prefix_.device.tail; tail && !tail->group_keys[i].empty()) {
            const std::vector<CacheBlockLocation> copied =
                groups_[i].Index().MatchedLocations(pool_, tail->group_keys[i], /*begin_block=*/0, tail->per_group[i]);
            // Key j of the tail is the j-th block of its page.
            const std::int32_t first_slot =
                tailSlots(geometry_[i], prefix_granularity_, prefix_.device.num_common_tokens).first;
            auto source = copied.begin();
            for (std::size_t j = 0; j < tail->per_group[i].hits.size(); ++j) {
                if (tail->per_group[i].hits[j] == 0) {
                    continue;
                }
                const std::int32_t bucket = allocator.BucketOf(*source++);
                ++allocation.blocks_by_bucket[static_cast<std::size_t>(bucket)];
                allocation.tail_slots.push_back(
                    CopyInReservation{.slot = first_slot + static_cast<std::int32_t>(j), .bucket = bucket});
            }
            allocation.any_bucket_adjustment +=
                static_cast<std::int64_t>(tail->group_keys[i].size()) - static_cast<std::int64_t>(copied.size()) -
                FirstTailBlockNeeded(groups_[i].Spec(), block_granularity, tail->tail_tokens);
            if (const auto* dense = std::get_if<DenseGrowth>(&demands_[i].extent)) {
                const std::int32_t partial_rows = tail->tail_tokens % block_granularity;
                const std::int32_t available = partial_rows == 0 ? 0 : block_granularity - partial_rows;
                const std::int32_t growth = dense->num_tokens + demands_[i].reserve_tokens;
                allocation.any_bucket_adjustment += geometry_[i].BlocksNeededBeyond(available, growth) -
                                                    geometry_[i].BlocksNeededFor(*demands_[i].table, growth);
            } else {
                const auto* sparse = std::get_if<SparseSuffix>(&demands_[i].extent);
                _assert(sparse != nullptr && sparse->first_block * block_granularity >= tail->tail_tokens,
                        "a sparse suffix lands past the hit's tail blocks");
            }
        }
        return allocation;
    }

    // One reservation per newly allocated copy-in slot. Existing or restored
    // blocks fix its bucket; -1 pairs a new block with its copy-in in a bucket
    // chosen during placement. Earlier admissions already reserved their slots.
    std::vector<CopyInReservation> copyInReservations(std::size_t i,
                                                      std::span<const CopyInReservation> prefix_tail_slots) const {
        if (!copy_in_point_) {
            return {};
        }
        const CacheStore::BlockRange slots = store_.CopyInSlots(i, *copy_in_point_);
        if (slots.first == slots.past_end) {
            return {};
        }
        const GroupDemand& demand = demands_[i];
        const auto* dense = std::get_if<DenseGrowth>(&demand.extent);
        _assert(dense != nullptr, "the slots of a copy-in point grow densely");
        const BlockTable& table = *demand.table;
        // The slots before the dense growth -- the rebuilt shape, the live
        // table, or a first admission's prefix -- and the capacity the last
        // of them leaves.
        std::int32_t base = 0;
        std::int32_t available = 0;
        if (!shapes_.empty()) {
            base = shapes_[i].num_blocks;
            available = shapes_[i].available_tokens;
        } else if (table.NumBlocks() > 0) {
            base = table.NumBlocks();
            available = table.AvailableTokens();
        } else {
            const TailSlots prefix =
                tailSlots(geometry_[i], prefix_granularity_,
                          std::max(prefix_.device.num_common_tokens, prefix_.host.num_common_tokens));
            base = prefix.past_end;
            available = prefix.available_tokens;
        }
        const std::int32_t growth_end =
            base + geometry_[i].BlocksNeededBeyond(available, dense->num_tokens + demand.reserve_tokens);
        const auto known_bucket = [](std::span<const CopyInReservation> known, std::int32_t slot) {
            const auto it = std::ranges::find(known, slot, &CopyInReservation::slot);
            _assert(it != known.end(), "a copy-in slot below the growth holds a block in a known bucket");
            return it->bucket;
        };
        std::vector<CopyInReservation> reservations;
        for (std::int32_t slot = slots.first; slot < std::min(slots.past_end, growth_end); ++slot) {
            std::int32_t bucket = -1;
            if (slot < base) {
                if (!shapes_.empty()) {
                    bucket = known_bucket(shapes_[i].rebuilt_copy_ins, slot);
                } else if (table.NumBlocks() > 0) {
                    continue;  // an earlier admission already reserved this slot
                } else {
                    bucket = known_bucket(prefix_tail_slots, slot);
                }
            }
            reservations.push_back(CopyInReservation{.slot = slot, .bucket = bucket});
        }
        return reservations;
    }

    // Whether the probe matched a checkpoint for State group i on either
    // tier, at a page boundary, a tail or (the D role) the prompt end.
    bool stateCheckpointMatched(std::size_t i) const {
        const auto any_hit = [](const GroupPrefixProbe& probe) {
            return std::ranges::any_of(probe.hits, [](std::uint8_t hit) { return hit != 0; });
        };
        const auto tier_hit = [&](const PrefixProbe::Tier& tier) {
            return (!tier.per_group.empty() && any_hit(tier.per_group[i])) ||
                   (tier.tail && !tier.tail->per_group.empty() && any_hit(tier.tail->per_group[i]));
        };
        return tier_hit(prefix_.device) || tier_hit(prefix_.host) ||
               (prefix_.state_at_prompt_end && any_hit(prefix_.state_at_prompt_end->per_group[i]));
    }

    // A State group's first rotation without a checkpoint to read starts from
    // the zero state, held as a fresh zeroed block in slot 0: the table holds
    // its two blocks from the first admission on, and every later rotation
    // swaps the released input for the fresh live slot.
    std::int64_t zeroStateBlocksNeeded(std::size_t i) const {
        if (geometry_[i].HasTokenSpan() || demands_[i].table->NumBlocks() != 0) {
            return 0;
        }
        const auto* step = std::get_if<StateStep>(&demands_[i].extent);
        return step != nullptr && step->kind == StateStep::Kind::kRotate && !stateCheckpointMatched(i) ? 1 : 0;
    }

    // A Host hit's Device destination must share the Host block's bucket, so
    // the copy's two ends have one owner under page-cyclic sharding. A hit
    // with no Host block yet (an L3 prefetch, replicated groups only) takes
    // bucket 0, where its block will be allocated. A Host tail's blocks land
    // the same way, into the hitter's private pages, each recorded for its
    // slot (PrefixAllocation::tail_slots).
    PrefixAllocation hostHitAllocation(std::size_t group_index) const {
        PrefixAllocation allocation{.blocks_by_bucket =
                                        std::vector<std::int64_t>(groups_[group_index].Allocator().ShardCount(), 0)};
        _assert(host_pool_ != nullptr, "Host prefix hits require a Host pool");
        const GroupPrefixProbe& probe = prefix_.host.per_group[group_index];
        const std::span<const CacheKey> keys = (*prefix_.group_keys)[group_index];
        // A State group's hits count from page 0 (PrefixProbe::Tier).
        const std::int32_t begin_block =
            geometry_[group_index].HasTokenSpan()
                ? prefix_.device.full_page_tokens / geometry_[group_index].BlockGranularity()
                : 0;
        const GroupAllocator& allocator = groups_[group_index].Allocator();
        const auto count_hit = [&](const CacheKey& key) {
            // A replicated group has only bucket 0, independent of whether
            // this source is already on Host or is an incoming L3 row.
            if (allocator.ShardCount() == 1) {
                ++allocation.blocks_by_bucket[0];
                return 0;
            }
            const CacheBlockRef host_block = groups_[group_index].Index().Find(*host_pool_, key);
            const std::int32_t bucket = host_block ? allocator.BucketOf(host_block->Location()) : 0;
            ++allocation.blocks_by_bucket[static_cast<std::size_t>(bucket)];
            return bucket;
        };
        for (std::size_t hit_index = 0; hit_index < probe.hits.size(); ++hit_index) {
            if (probe.hits[hit_index] == 0) {
                continue;
            }
            const std::size_t key_index = static_cast<std::size_t>(begin_block) + hit_index;
            _assert(key_index < keys.size(), "host prefix hit is outside the probed key range");
            count_hit(keys[key_index]);
        }
        if (const auto& tail = prefix_.host.tail; tail && !tail->group_keys[group_index].empty()) {
            const GroupPrefixProbe& tail_probe = tail->per_group[group_index];
            // Key j of the tail is the j-th block of its page; a State
            // group's one block has no slot a copy-in could ask about.
            const std::int32_t first_slot =
                geometry_[group_index].HasTokenSpan()
                    ? tailSlots(geometry_[group_index], prefix_granularity_, prefix_.host.num_common_tokens).first
                    : 0;
            for (std::size_t j = 0; j < tail_probe.hits.size(); ++j) {
                if (tail_probe.hits[j] == 0) {
                    continue;
                }
                const std::int32_t bucket = count_hit(tail->group_keys[group_index][j]);
                if (geometry_[group_index].HasTokenSpan()) {
                    allocation.tail_slots.push_back(
                        CopyInReservation{.slot = first_slot + static_cast<std::int32_t>(j), .bucket = bucket});
                }
            }
        }
        return allocation;
    }

    bool couldFitAfterEviction(std::span<const BlockPlacementDemand> block_demands) const {
        std::int64_t parents_needed = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::int64_t need = block_demands[i].TotalBlocks();
            // Only this step's released input or newly expired window slots
            // can fund it. Live history and the State output stay owned by the
            // request. Counting them would hide an otherwise impossible fit.
            std::int64_t request_reclaim = 0;
            const BlockTable& table = *demands_[i].table;
            if (num_computed_tokens_) {
                if (const auto* step = std::get_if<StateStep>(&demands_[i].extent)) {
                    request_reclaim =
                        groups_[i].Allocator().ReleasableStateInput(groups_[i].Index(), table, *step) ? 1 : 0;
                } else {
                    const std::int32_t expired_block_end =
                        geometry_[i].ExpiredBlockEndAt(groups_[i].Spec(), *num_computed_tokens_);
                    request_reclaim = groups_[i].Allocator().ExpiredBlockSlotsAt(table, expired_block_end).size();
                }
            }
            // Still relax cache pins, prefix protection, bucket placement,
            // segment constraints, and ownership/null holes in expired slots.
            const std::int64_t removable =
                static_cast<std::int64_t>(groups_[i].Index().PotentiallyEvictableEntries(pool_)) + request_reclaim;
            const std::int64_t excess = need - pool_.NumFreeSlots(static_cast<std::uint32_t>(i)) - removable;
            const std::int64_t packing = pool_.Packing(static_cast<std::uint32_t>(i));
            // Existing holes cannot move to another group by themselves.
            // Emptying any parent requires releasing at least the smallest
            // current parent occupancy, even if all eligible slots line up.
            const std::int64_t parents_reclaimable =
                removable / pool_.MinimumParentOccupancy(static_cast<std::uint32_t>(i));
            parents_needed +=
                excess > 0 ? (excess + packing - 1) / packing : -std::min((-excess) / packing, parents_reclaimable);
        }
        return parents_needed <= pool_.NumEmptyLcmBlocks();
    }

    // Ignore segment restrictions, but preserve actual parent occupancy and
    // bucket placement. If releasing every available block still cannot fit,
    // no victim ordering can help. Request releases often fund a State rotation
    // by themselves, so test those before visiting the cache's available index.
    bool couldFitAfterReleasingCandidates(CacheEvictionCandidates& candidates,
                                          const AdmissionCandidateConstraints& constraints,
                                          std::span<const BlockPlacementDemand> block_demands) const {
        ShadowOccupancy maximum{pool_, groups_};
        for (const VictimCandidate& candidate : constraints.request_reclaim_candidates) {
            maximum.Remove(candidate.group_id, candidate.location);
        }
        // The fit is asked again only once enough blocks have left to close
        // the shortfall (ShadowOccupancy::ParentShortfall): at most once
        // before an epoch's end and once at it, never in an epoch too short
        // to complete the fit. The walk thus usually stops releasing at the
        // block that completes the fit; that block's epoch is still read
        // whole.
        std::int64_t removals_until_fit = maximum.ParentShortfall(block_demands);
        if (removals_until_fit == 0) {
            return true;
        }
        return candidates.PreScanUntil(
            [&](std::uint32_t group_id, std::span<const PrefixCacheIndex::EvictionCandidate> entries) {
                bool asked = false;
                for (const auto& entry : entries) {
                    if (constraints.protected_locations.contains(entry.location) ||
                        constraints.request_reclaim_locations.contains(entry.location)) {
                        continue;
                    }
                    maximum.Remove(group_id, entry.location);
                    if (removals_until_fit > 0) {
                        --removals_until_fit;
                    }
                    if (removals_until_fit == 0 && !asked) {
                        asked = true;
                        removals_until_fit = maximum.ParentShortfall(block_demands);
                        if (removals_until_fit == 0) {
                            return true;
                        }
                    }
                }
                if (removals_until_fit == 0) {
                    removals_until_fit = maximum.ParentShortfall(block_demands);
                }
                return removals_until_fit == 0;
            });
    }

    // The candidates the walk in fitWithEviction would select up to the
    // demand's first fit, less those that cannot change what it keeps; or
    // nullopt where that is not shown here, and the walk runs instead.
    //
    // A demand some uncached group (a replayable window, say) cannot place in
    // its holes needs empty parents, and a parent of a cached group empties
    // only once the walk reaches its newest entry: with the epochs mixed in
    // every parent, that is most of the cache. Each tracked parent's newest
    // epoch (PrefixCacheIndex::ParentEntries) gives that point directly. The
    // walk's fit holds first at the event F where the demand fits with every
    // cached group's own holes taken as enough -- a parent emptying, or a
    // request reclaim -- once each cached group's oldest candidates outside
    // the parents emptied by F (Q) give it the holes it lacks, all before F.
    // Then the walk selects exactly what this returns plus cached entries
    // outside those parents and newer than Q, which only add holes no group
    // still needs: no fit and no Minimize decision changes with them, and
    // Minimize restores every one. Without segments every candidate leaves
    // alone; State groups rank by checkpoint aging, which a parent's epoch
    // does not give, so either keeps the walk.
    //
    // Virtual so that a test can compare the selection with the walk.
    virtual std::optional<std::vector<VictimCandidate>> candidatesThroughParentFit(
        std::span<const BlockPlacementDemand> block_demands, const AdmissionCandidateConstraints& constraints) const {
        ProfileSpan profile_span{"AdmissionPlanner::candidatesThroughParentFit"};
        if (evicts_segments_ || store_.NumStateGroups() > 0) {
            return std::nullopt;
        }
        // The demand with the cached groups' share dropped: what must fit
        // when their holes are taken as enough.
        std::vector<bool> cached(groups_.size(), false);
        std::vector<BlockPlacementDemand> uncached_demands(block_demands.begin(), block_demands.end());
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const PrefixCacheIndex& index = groups_[i].Index();
            if (index.NumEntries(pool_) == 0) {
                continue;
            }
            if (index.EntriesByParent(pool_).empty()) {
                return std::nullopt;
            }
            cached[i] = true;
            uncached_demands[i] = BlockPlacementDemand{
                .blocks_by_bucket = std::vector<std::int64_t>(block_demands[i].blocks_by_bucket.size(), 0)};
        }
        // A reclaim in a cached group would take holes from it as Minimize
        // restores it.
        if (std::ranges::any_of(constraints.request_reclaim_candidates,
                                [&](const VictimCandidate& candidate) { return cached[candidate.group_id]; })) {
            return std::nullopt;
        }
        ShadowOccupancy shadow{pool_, groups_};
        if (shadow.Fits(uncached_demands)) {
            // Only cached holes are short: the walk ends within the oldest entries.
            return std::nullopt;
        }

        // Every cached parent whose occupants are all entries, earliest
        // emptied first: by its newest epoch, then, once its children are
        // read, by its newest child as the walk ranks it.
        struct Parent {
            std::uint64_t newest_access_epoch;
            std::int32_t lcm_block_id;
            std::uint32_t group_id;
            // Its children in walk order, once read (children_of).
            std::size_t children{0};
            bool read{false};
        };
        std::vector<std::vector<VictimCandidate>> children_of;
        const auto later = [&](const Parent& lhs, const Parent& rhs) {
            if (lhs.newest_access_epoch != rhs.newest_access_epoch) {
                return lhs.newest_access_epoch > rhs.newest_access_epoch;
            }
            if (lhs.read != rhs.read) {
                return lhs.read;
            }
            return lhs.read &&
                   CacheEvictionCandidates::Before(children_of[rhs.children].back(), children_of[lhs.children].back());
        };
        std::vector<Parent> parents;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (!cached[i]) {
                continue;
            }
            const std::span<const PrefixCacheIndex::ParentEntries> entries = groups_[i].Index().EntriesByParent(pool_);
            for (std::size_t id = 1; id < entries.size(); ++id) {
                const auto lcm_block_id = static_cast<std::int32_t>(id);
                if (entries[id].count > 0 && entries[id].count == pool_.OccupiedCount(lcm_block_id)) {
                    parents.push_back(Parent{.newest_access_epoch = entries[id].newest_access_epoch,
                                             .lcm_block_id = lcm_block_id,
                                             .group_id = static_cast<std::uint32_t>(i)});
                }
            }
        }
        std::ranges::make_heap(parents, later);
        // Reads the earliest parent's children until the earliest is read,
        // dropping a parent with one the walk never offers (in use or kept).
        const auto read_earliest = [&] {
            while (!parents.empty() && !parents.front().read) {
                std::ranges::pop_heap(parents, later);
                Parent parent = parents.back();
                parents.pop_back();
                std::vector<VictimCandidate> children;
                bool offered = true;
                for (std::int32_t slot = 0; offered && slot < pool_.Packing(parent.group_id); ++slot) {
                    const CacheBlockLocation location{.lcm_block_id = parent.lcm_block_id, .slot_index = slot};
                    if (!pool_.IsOccupied(location)) {
                        continue;
                    }
                    const auto entry = groups_[parent.group_id].Index().UnpinnedCandidateAt(pool_, location);
                    offered = entry && !constraints.protected_locations.contains(location);
                    if (offered) {
                        children.push_back(CacheEvictionCandidates::MakeCandidate(groups_[parent.group_id], *entry));
                    }
                }
                if (!offered) {
                    continue;
                }
                CacheEvictionCandidates::Sort(children);
                _assert(children.back().rank_epoch == parent.newest_access_epoch,
                        "a parent's newest child ranks at the parent's newest epoch");
                parent.read = true;
                parent.children = children_of.size();
                children_of.push_back(std::move(children));
                parents.push_back(parent);
                std::ranges::push_heap(parents, later);
            }
        };

        // Events in walk order until the demand fits on the uncached groups.
        std::vector<VictimCandidate> through_fit;
        // By LCM block id: the parents the events emptied.
        std::vector<bool> emptied(static_cast<std::size_t>(pool_.NumLcmBlocks()) + 1, false);
        std::size_t next_reclaim = 0;
        std::optional<VictimCandidate> fit;
        while (!fit) {
            read_earliest();
            const VictimCandidate* reclaim = next_reclaim < constraints.request_reclaim_candidates.size()
                                                 ? &constraints.request_reclaim_candidates[next_reclaim]
                                                 : nullptr;
            if (!parents.empty() &&
                (reclaim == nullptr ||
                 CacheEvictionCandidates::Before(children_of[parents.front().children].back(), *reclaim))) {
                std::ranges::pop_heap(parents, later);
                const Parent parent = parents.back();
                parents.pop_back();
                for (const VictimCandidate& child : children_of[parent.children]) {
                    shadow.Remove(child.group_id, child.location);
                    through_fit.push_back(child);
                }
                emptied[static_cast<std::size_t>(parent.lcm_block_id)] = true;
                if (shadow.Fits(uncached_demands)) {
                    fit = children_of[parent.children].back();
                }
            } else if (reclaim != nullptr) {
                shadow.Remove(reclaim->group_id, reclaim->location);
                through_fit.push_back(*reclaim);
                ++next_reclaim;
                if (shadow.Fits(uncached_demands)) {
                    fit = *reclaim;
                }
            } else {
                return std::nullopt;
            }
        }

        // Q: each cached group's oldest candidates outside the emptied
        // parents, until its holes are enough; all must precede the fit.
        std::vector<std::int64_t> pair_scratch;
        const auto short_of_holes = [&](std::uint32_t i) {
            const std::span<const std::int64_t> holes = shadow.HolesByBucket(i);
            pair_scratch.assign(holes.size(), 0);
            return ParentsNeededForBucketsAndPairs(block_demands[i], holes, pool_.Packing(i), pair_scratch) > 0;
        };
        std::size_t num_short = 0;
        for (std::uint32_t i = 0; i < groups_.size(); ++i) {
            num_short += cached[i] && short_of_holes(i) ? 1 : 0;
        }
        CacheEvictionCandidates oldest{groups_, pool_, device_aging_};
        // Selection only matters to checkpoint aging, which needs a State group.
        const auto not_selected = [](CacheBlockLocation) { return false; };
        while (num_short > 0) {
            const std::optional<VictimCandidate> candidate =
                oldest.Next(std::span<const VictimCandidate>{}, constraints.protected_locations,
                            constraints.request_reclaim_locations, not_selected);
            if (!candidate || !CacheEvictionCandidates::Before(*candidate, *fit)) {
                return std::nullopt;
            }
            if (emptied[static_cast<std::size_t>(candidate->location.lcm_block_id)] ||
                !short_of_holes(candidate->group_id)) {
                continue;
            }
            shadow.Remove(candidate->group_id, candidate->location);
            through_fit.push_back(*candidate);
            num_short -= short_of_holes(candidate->group_id) ? 0 : 1;
        }
        CacheEvictionCandidates::Sort(through_fit);
        return through_fit;
    }

    // Selects `candidates`, each alone as the walk takes it; the demand first
    // fits after the last (candidatesThroughParentFit).
    void selectThroughParentFit(std::span<const VictimCandidate> candidates,
                                std::span<const BlockPlacementDemand> block_demands,
                                const AdmissionCandidateConstraints& constraints,
                                AdmissionSelectionState& selection) const {
        selection.set_member_by_location.reserve(candidates.size());
        std::size_t next = 0;
        const bool fits = selection.eviction_plan.SelectUntilFits(block_demands, [&] {
            if (next == candidates.size()) {
                return false;
            }
            const VictimCandidate& candidate = candidates[next++];
            std::optional<ExpandedVictimSet> expanded =
                expandVictimSet(candidate, /*candidates_ran_out=*/false, constraints, selection);
            _assert(expanded.has_value(), "a candidate without segments leaves alone");
            takeVictimSet(std::move(*expanded), candidate.location, selection);
            return true;
        });
        _assert(fits && next == candidates.size(), "the demand first fits after the last candidate through the fit");
    }

    // The store planned on, read through its public views (NodeEntries,
    // GroupIsState, CopyInSlots); groups_, geometry_, the pools and the
    // boundary tree are its own.
    const CacheStore& store_;
    const CheckpointAging device_aging_;
    const std::vector<CacheGroup>& groups_;
    const std::span<const GroupGeometry> geometry_;
    const std::int32_t prefix_granularity_;
    const BlockPool& pool_;
    const BlockPool* const host_pool_;
    const std::span<const GroupDemand> demands_;
    const std::optional<std::int32_t> num_computed_tokens_;
    // The point whose copy-in blocks this admission reserves (Admit).
    const std::optional<std::int32_t> copy_in_point_;
    const PrefixProbe& prefix_;
    const std::span<const PlannedShape> shapes_;
    // Device-cached blocks a restore claims instead of copying: protected
    // from eviction like the probe's prefix hits.
    const std::span<const std::pair<std::uint32_t, CacheBlockLocation>> planned_claim_locations_;
    const BoundaryTree& boundary_tree_;
    // Whether a victim takes its segment along
    // (CacheCoordinator::WritesBackEvictions).
    const bool evicts_segments_;
};

// Plans with `Planner` (AdmissionPlanner, or a subclass that goes on past
// the segment rules) and packages the plan for the commit.
template <typename Planner>
std::optional<AdmissionPlan> planAdmissionWith(
    const CacheStore& store, CheckpointAging device_aging, PrefixProbe&& prefix, std::span<const GroupDemand> demands,
    std::optional<std::int32_t> num_computed_tokens, std::optional<std::int32_t> copy_in_point,
    std::span<const PlannedShape> shapes,
    std::span<const std::pair<std::uint32_t, CacheBlockLocation>> planned_claim_locations, bool evicts_segments) {
    ProfileSpan profile_span{"planAdmission"};
    _assert(demands.size() == store.Groups().size(), "demands/groups size mismatch");

    auto member_resource = std::make_unique<std::pmr::monotonic_buffer_resource>();
    const Planner planner{store,          device_aging, demands, num_computed_tokens,
                          copy_in_point,  prefix,       shapes,  planned_claim_locations,
                          evicts_segments};
    std::optional<AdmissionDecisions> decisions = planner.Plan(member_resource.get());
    if (!decisions) return std::nullopt;
    return AdmissionPlan{.prefix = std::move(prefix),
                         .member_resource = std::move(member_resource),
                         .victims = std::move(decisions->victims),
                         .copy_ins = std::move(decisions->copy_ins)};
}

inline std::optional<AdmissionPlan> planAdmission(
    const CacheStore& store, CheckpointAging device_aging, PrefixProbe&& prefix, std::span<const GroupDemand> demands,
    std::optional<std::int32_t> num_computed_tokens, std::optional<std::int32_t> copy_in_point,
    std::span<const PlannedShape> shapes,
    std::span<const std::pair<std::uint32_t, CacheBlockLocation>> planned_claim_locations, bool evicts_segments) {
    return planAdmissionWith<AdmissionPlanner>(store, device_aging, std::move(prefix), demands, num_computed_tokens,
                                               copy_in_point, shapes, planned_claim_locations, evicts_segments);
}

template <typename Planner>
std::optional<AdmissionResult> CacheCoordinator::AdmitWith(PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                                           const RequestProgress& progress,
                                                           std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                                           std::optional<CopyInPoint> copy_in_point,
                                                           std::optional<std::uint64_t> request_access_epoch,
                                                           const std::string& session_id,
                                                           const std::string& cache_group_id) {
    const std::vector<GroupDemand> replayed =
        prepareAdmission(prefix, demands, progress, copy_in_blocks, copy_in_point);
    if (!replayed.empty()) {
        demands = replayed;
    }
    // The publication this admission carries needs no block beyond the plan's:
    // an owner that keeps writing past a tail copies into the blocks it holds
    // for it (copy_in_blocks).
    std::optional<AdmissionPlan> plan = planAdmissionWith<Planner>(
        store_, history_.CheckpointAgingFor(CacheTier::kDevice), std::move(prefix), demands,
        progress.num_computed_tokens, copy_in_point ? std::optional<std::int32_t>{copy_in_point->tokens} : std::nullopt,
        /*shapes=*/{},
        /*planned_claim_locations=*/{}, write_back_evictions_);
    if (!plan) {
        return std::nullopt;
    }
    return commitAdmission(std::move(*plan), demands, progress, copy_in_blocks, copy_in_point, request_access_epoch,
                           session_id, cache_group_id);
}

}  // namespace tokenspeed
