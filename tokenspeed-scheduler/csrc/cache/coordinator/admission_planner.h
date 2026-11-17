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
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/coordinator/eviction_plan.h"
#include "cache/coordinator/flat_hash_map.h"
#include "cache/coordinator/group_geometry.h"
#include "cache/coordinator/shadow_occupancy.h"
#include "cache/coordinator/victim_set.h"
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
    // (AdmissionPlanner::reserveCopyIns). Empty without a copy-in point.
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

class AdmissionPlanner {
public:
    // device_aging: the Device checkpoint aging candidates rank by
    // (AdmissionHistory::CheckpointAgingFor).
    AdmissionPlanner(const CacheStore& store, CheckpointAging device_aging, std::span<const GroupDemand> demands,
                     std::optional<std::int32_t> num_computed_tokens, std::optional<std::int32_t> copy_in_point,
                     const PrefixProbe& prefix, std::span<const PlannedShape> shapes,
                     std::span<const std::pair<std::uint32_t, CacheBlockLocation>> claimed, bool evicts_segments,
                     std::vector<VictimSet>& victims, std::pmr::memory_resource* member_resource,
                     std::vector<std::vector<CopyInReservation>>& copy_ins)
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
          claimed_{claimed},
          boundary_tree_{store.Boundaries()},
          evicts_segments_{evicts_segments},
          victims_{victims},
          member_resource_{member_resource},
          copy_ins_{copy_ins},
          eviction_plan_{store.DevicePool(), store.Groups()},
          shape_need_(store.Groups().size()),
          blocks_needed_(store.Groups().size()),
          pair_need_(store.Groups().size(), 0),
          prefix_tail_buckets_(store.Groups().size()) {
        copy_ins_.assign(copy_in_point ? store.Groups().size() : 0, {});
    }

    bool Plan() {
        victims_.clear();
        initializeCapacity();
        if (!fitWithEviction()) {
            return false;
        }
        // The fit chose a bucket for every pair against the shadow its
        // victims leave; the copy-in blocks of the slots the growth fills
        // take those buckets.
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (pair_need_[i] == 0) {
                continue;
            }
            const std::vector<std::int32_t> pair_buckets =
                AssignPairBuckets(shape_need_[i], eviction_plan_.HolesByBucket(i), pair_need_[i]);
            auto next_pair = pair_buckets.begin();
            for (CopyInReservation& copy_in : copy_ins_[i]) {
                if (copy_in.bucket < 0) {
                    copy_in.bucket = *next_pair++;
                }
            }
            _assert(next_pair == pair_buckets.end(), "every pair of the fit has a copy-in slot");
        }
        return true;
    }

    virtual ~AdmissionPlanner() = default;

protected:
    using VictimCandidate = CacheEvictionCandidates::Candidate;
    using NodeEntry = CacheStore::NodeEntry;

    bool fitWithEviction() {
        ProfileSpan profile_span{"AdmissionPlanner::fitWithEviction"};
        // Existing local holes plus empty parents are the zero-eviction
        // capacity. Do not discard cache merely to make placement denser: a
        // request's own scattered blocks are gathered by its compaction
        // (LoneRequestAdmission::Compact) once it is alone and nothing
        // else funds it, never here.
        if (eviction_plan_.Fits(shape_need_, blocks_needed_, pair_need_)) {
            return true;
        }
        if (!couldFitAfterEviction()) {
            return false;
        }

        collectCandidates();
        if (!couldFitAfterReleasingCandidates()) {
            return false;
        }
        // Room for the blocks the free slots lack, about what the selected
        // sets hold: a hint only, so the table rarely regrows.
        std::int64_t shortfall = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::int64_t free_slots = pool_.NumFreeSlots(static_cast<std::uint32_t>(i));
            shortfall += std::max<std::int64_t>(blocksDemanded(i) - free_slots, 0);
        }
        set_member_by_location_.reserve(static_cast<std::size_t>(shortfall));
        bool candidates_ran_out = false;
        std::size_t next_deferred_shared_page = 0;
        // Once every source of the segment rules ran out: a subclass's own
        // candidates (nextCandidatePastSegments).
        bool past_segments = false;
        bool reclaim_selection_changed = false;
        std::vector<VictimCandidate> retry_segments;
        std::size_t next_retry_segment = 0;
        const bool fits = eviction_plan_.SelectUntilFits(shape_need_, blocks_needed_, pair_need_, [&] {
            while (true) {
                std::optional<VictimCandidate> candidate = eviction_plan_.Next(request_reclaim_candidates_);
                if (!candidate && next_deferred_shared_page < deferred_shared_pages_.size()) {
                    // The candidates ran out before the demand fits, and pages two
                    // or more checkpoint boundaries share are still in place. Count
                    // the checkpoint boundaries the selected sets take whole: a
                    // page only they depend on may leave now, with everything under
                    // it.
                    if (!candidates_ran_out) {
                        candidates_ran_out = true;
                        FlatHashSet<const BoundaryTree::Node*> counted{&scratch_};
                        for (const VictimSet& set : victims_) {
                            for (const VictimSet::Member& member : set.members) {
                                if (!store_.GroupIsState(member.group_id)) {
                                    continue;
                                }
                                const std::optional<CacheKey> key =
                                    groups_[member.group_id].Index().KeyOf(pool_, member.location);
                                const BoundaryTree::Node* checkpoint_node =
                                    key ? boundary_tree_.Find(key->content_hash) : nullptr;
                                if (checkpoint_node == nullptr || !counted.insert(checkpoint_node) ||
                                    !std::ranges::all_of(store_.NodeEntries(pool_, *checkpoint_node),
                                                         [&](const NodeEntry& entry) {
                                                             return !store_.GroupIsState(entry.group_id) ||
                                                                    set_member_by_location_.contains(entry.location);
                                                         })) {
                                    continue;
                                }
                                for (const BoundaryTree::Node* ancestor = checkpoint_node; ancestor != nullptr;
                                     ancestor = ancestor->parent) {
                                    ++leaving_checkpoint_boundary_counts_[ancestor];
                                }
                            }
                        }
                    }
                    candidate = deferred_shared_pages_[next_deferred_shared_page++];
                }
                if (!candidate && !past_segments && next_retry_segment == retry_segments.size() &&
                    reclaim_selection_changed) {
                    // A reclaim selected after a rejected segment can make that
                    // segment evictable. Retry a fixed batch before block fallback;
                    // another pass requires a newly selected reclaim block.
                    reclaim_selection_changed = false;
                    unevictable_locations_.clear();
                    retry_segments = std::exchange(passed_over_, {});
                    next_retry_segment = 0;
                }
                if (!candidate && next_retry_segment < retry_segments.size()) {
                    candidate = retry_segments[next_retry_segment++];
                }
                if (!candidate) {
                    candidate = nextCandidatePastSegments();
                    past_segments = past_segments || candidate.has_value();
                }
                if (!candidate) {
                    return false;
                }
                std::optional<ExpandedVictimSet> expanded = past_segments
                                                                ? expandCandidatePastSegments(*candidate)
                                                                : expandVictimSet(*candidate, candidates_ran_out);
                if (!expanded) {
                    continue;
                }
                auto& [set, num_taken_over_members, taken_over_set_index, segment_top] = *expanded;
                const CacheBlockLocation location = candidate->location;
                _assert(
                    std::ranges::find(set.members.begin() + static_cast<std::ptrdiff_t>(num_taken_over_members),
                                      set.members.end(), location, &VictimSet::Member::location) != set.members.end(),
                    "a victim set holds the candidate it was chosen for");
                // An earlier set this one contains moves into it whole, each block
                // keeping why it left; the rest leave the shadow now. The lists
                // taken over in place are there already.
                const std::size_t set_index = victims_.size();
                std::vector<std::size_t> absorbed;
                std::size_t absorbed_members = 0;
                for (std::size_t j = num_taken_over_members; j < set.members.size(); ++j) {
                    VictimSet::Member& member = set.members[j];
                    const auto [joined, inserted] =
                        set_member_by_location_.TryEmplace(member.location, std::pair{set_index, j});
                    if (inserted) {
                        eviction_plan_.Remove(member.group_id, member.location);
                        reclaim_selection_changed = reclaim_selection_changed || member.is_request_reclaim;
                        continue;
                    }
                    // The set holding the block now, through the sets that took
                    // lists over.
                    std::size_t earlier = joined->first;
                    while (taken_by_[earlier] != earlier) {
                        earlier = taken_by_[earlier] = taken_by_[taken_by_[earlier]];
                    }
                    const std::size_t position = joined->second;
                    member = victims_[earlier].members[position];
                    if (std::ranges::find(absorbed, earlier) == absorbed.end()) {
                        absorbed.push_back(earlier);
                    }
                    ++absorbed_members;
                    *joined = std::pair{set_index, j};
                }
                for (const std::size_t earlier : absorbed) {
                    absorbed_members -= victims_[earlier].members.size();
                    victims_[earlier] = VictimSet{member_resource_};
                }
                _assert(absorbed_members == 0, "an earlier victim set lies whole inside the set that takes it");
                if (taken_over_set_index) {
                    taken_by_[*taken_over_set_index] = set_index;
                }
                victims_.push_back(std::move(set));
                taken_by_.push_back(set_index);
                if (segment_top != nullptr) {
                    segment_at_[segment_top] = set_index;
                }
                return true;
            }
        });

        if (!fits) return false;
        eviction_plan_.Minimize(victims_, shape_need_, blocks_needed_, pair_need_);
        return true;
    }

    // What an expansion returns beside the set: how many of its first members
    // it took over in place from an earlier segment (their
    // set_member_by_location_ entries as they are) and that set's index, and
    // the tree node a segment set starts at (nullptr for any other set).
    struct ExpandedVictimSet {
        VictimSet set;
        std::size_t num_taken_over_members{0};
        std::optional<std::size_t> taken_over_set_index;
        const BoundaryTree::Node* segment_top{nullptr};
    };

    // Once the segment rules and their retries have nothing left to offer, a
    // subclass may go on with candidates of its own, expanded its own way
    // (LoneRequestPlanner, lone_request_admission.cpp). This planner stops there.
    virtual std::optional<VictimCandidate> nextCandidatePastSegments() { return std::nullopt; }
    virtual std::optional<ExpandedVictimSet> expandCandidatePastSegments(const VictimCandidate& candidate) {
        _assert(false, "a planner that offers no candidates past the segment rules expands none");
        return std::nullopt;
    }

    // Whether a block may join a set: one a selected set holds already, or
    // one nothing uses, keeps or reclaims. A selected request reclaim still
    // has its table reference until Admit executes. A later segment may
    // absorb that set; the merge in fitWithEviction keeps its capacity
    // counted once and Minimize must restore them together.
    bool isEvictable(const NodeEntry& entry) {
        return set_member_by_location_.contains(entry.location) ||
               (!entry.in_use && !protected_locations_.contains(entry.location) &&
                !request_reclaim_locations_.contains(entry.location));
    }

    // The Device checkpoint boundaries at or under a node that no selected
    // set takes whole: every one while candidates last, then those staying.
    std::int32_t numStayingCheckpointBoundaries(const BoundaryTree::Node& node) {
        const std::int32_t* leaving = leaving_checkpoint_boundary_counts_.Find(&node);
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

    // The sets every rule expands alike, into `set`: a block that leaves
    // alone (no tree node to follow, or a window block, which only shortens
    // the boundaries its prefix resumes at, unless reclaimed) and the
    // request's own reclaim (its released slots at this hash, and at a tail
    // the tail's other blocks, which their eviction takes along). False for
    // any other candidate.
    bool expandAloneOrReclaim(const VictimCandidate& candidate, const BoundaryTree::Node* candidate_node,
                              VictimSet& set) {
        const bool is_request_reclaim = request_reclaim_locations_.contains(candidate.location);
        const bool sliding =
            !groups_[candidate.group_id].Matcher().IsPrefixClosed() && !store_.GroupIsState(candidate.group_id);
        if (candidate_node == nullptr || (sliding && !is_request_reclaim)) {
            set.members.push_back(VictimSet::Member{.group_id = candidate.group_id,
                                                    .location = candidate.location,
                                                    .in_stranded_suffix = false,
                                                    .is_request_reclaim = is_request_reclaim});
            return true;
        }
        if (!is_request_reclaim) {
            return false;
        }
        for (const NodeEntry& entry : store_.NodeEntries(pool_, *candidate_node)) {
            if (request_reclaim_locations_.contains(entry.location) ||
                (candidate_node->tail_tokens > 0 && isEvictable(entry))) {
                set.members.push_back(VictimSet::Member{.group_id = entry.group_id,
                                                        .location = entry.location,
                                                        .in_stranded_suffix = false,
                                                        .is_request_reclaim = true});
            }
        }
        return true;
    }

    // Every State group's checkpoint at `node`, into `set`, leaving alone: at
    // a tail its eviction takes the tail's other unpinned blocks along
    // (CacheEvictor::EvictCachedBlock), uncounted here. False, each remembered as unable
    // to leave, when one is in use or kept.
    bool expandCheckpointsAlone(const BoundaryTree::Node& node, VictimSet& set) {
        bool evictable = true;
        for (const NodeEntry& entry : store_.NodeEntries(pool_, node)) {
            if (store_.GroupIsState(entry.group_id)) {
                evictable = evictable && isEvictable(entry);
                set.members.push_back(VictimSet::Member{.group_id = entry.group_id,
                                                        .location = entry.location,
                                                        .in_stranded_suffix = false,
                                                        .is_request_reclaim = false});
            }
        }
        if (!evictable) {
            for (const VictimSet::Member& member : set.members) {
                unevictable_locations_.insert(member.location);
            }
        }
        return evictable;
    }

    // The Device subtree under `top`, into `out`, in post-order: a node after
    // the nodes under it, its own entries in NodeEntries order --
    // `candidate_entries` for the candidate's node, `climbed` (in climb
    // order) for the nodes a segment climbed to above it. A child the Device
    // does not hold has nothing on the Device under it when the model has a
    // prefix-closed group (eviction keeps the Device's coverage closed from
    // the root), so the walk skips the Host's subtrees; without one, what
    // such a child keeps under it on the Device needs none of these pages
    // either. A page a restore republished below such a child stays and
    // leaves later as a segment of its own. A block under the candidate's
    // node is in its stranded suffix. False at the first entry that cannot
    // leave, unless it is at `top` and `top_keeps_unavailable`: it stays.
    bool walkSubtree(const BoundaryTree::Node& top, const BoundaryTree::Node& candidate_node,
                     std::span<const NodeEntry> candidate_entries,
                     std::span<const std::pair<const BoundaryTree::Node*, std::vector<NodeEntry>>> climbed,
                     bool top_keeps_unavailable, ExpandedVictimSet& out) {
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
                if (segment_at_.contains(child)) {
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
                if (!isEvictable(entry)) {
                    if (top_keeps_unavailable && frame.node == &top) {
                        continue;
                    }
                    return false;
                }
                set.members.push_back(VictimSet::Member{.group_id = entry.group_id,
                                                        .location = entry.location,
                                                        .in_stranded_suffix = frame.in_stranded_suffix,
                                                        .is_request_reclaim = false});
            }
            stack.pop_back();
        }
        if (!spliced.empty()) {
            VictimSet walked = std::exchange(set, VictimSet{member_resource_});
            std::size_t next_walked = 0;
            for (const auto& [walked_before, child] : spliced) {
                set.members.insert(set.members.end(), walked.members.begin() + next_walked,
                                   walked.members.begin() + walked_before);
                next_walked = walked_before;
                const std::size_t earlier_index = *segment_at_.Find(child);
                VictimSet& earlier = victims_[earlier_index];
                _assert(!earlier.members.empty(), "a segment set still starts at its top");
                if (set.members.empty()) {
                    // First in the order: its lists become this set's.
                    set = std::exchange(earlier, VictimSet{member_resource_});
                    out.taken_over_set_index = earlier_index;
                    out.num_taken_over_members = set.members.size();
                } else {
                    // Moved in below like any other earlier set it holds.
                    set.members.insert(set.members.end(), earlier.members.begin(), earlier.members.end());
                }
                segment_at_.Erase(child);
            }
            set.members.insert(set.members.end(), walked.members.begin() + next_walked, walked.members.end());
        }
        return true;
    }

    // One candidate expanded to the set it leaves with under the segment
    // rules -- as a segment, alone, or as the request's own reclaim -- or
    // nullopt when it is skipped this pass (a block in the set is in use or
    // kept; the shared page is deferred). Pure apart from the pass's own memo
    // lists (passed_over_, deferred_shared_pages_, unevictable_locations_)
    // and, for a segment, the earlier sets it takes over in place.
    std::optional<ExpandedVictimSet> expandVictimSet(const VictimCandidate& candidate, bool candidates_ran_out) {
        // Already leaving with an earlier set, or found unable to leave
        // (passed over).
        if (set_member_by_location_.contains(candidate.location)) {
            return std::nullopt;
        }
        if (unevictable_locations_.contains(candidate.location)) {
            passed_over_.push_back(candidate);
            return std::nullopt;
        }
        ExpandedVictimSet out{.set = VictimSet{member_resource_}};
        const BoundaryTree::Node* candidate_node = candidateNode(candidate);
        if (expandAloneOrReclaim(candidate, candidate_node, out.set)) {
            return out;
        }
        if (numStayingCheckpointBoundaries(*candidate_node) >= 2) {
            // Deeper checkpoint boundaries still read this page. A nested
            // checkpoint leaves alone, every State group's at once; the page
            // leaves with the last checkpoint boundary under it, or is tried
            // again once the candidates run out.
            if (!store_.GroupIsState(candidate.group_id)) {
                (candidates_ran_out ? passed_over_ : deferred_shared_pages_).push_back(candidate);
                return std::nullopt;
            }
            if (!expandCheckpointsAlone(*candidate_node, out.set)) {
                return std::nullopt;
            }
            return out;
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
        if (numStayingCheckpointBoundaries(*candidate_node) == 1) {
            while (top->parent != nullptr && top->parent->OnDevice() &&
                   numStayingCheckpointBoundaries(*top->parent) == 1) {
                std::vector<NodeEntry> entries = store_.NodeEntries(pool_, *top->parent);
                if (!std::ranges::all_of(entries, [this](const NodeEntry& entry) { return isEvictable(entry); })) {
                    break;
                }
                top = top->parent;
                climbed.emplace_back(top, std::move(entries));
            }
        }
        if (!walkSubtree(*top, *candidate_node, candidate_entries, climbed, /*top_keeps_unavailable=*/false, out)) {
            passed_over_.push_back(candidate);
            // One unavailable member rejects this whole set; there is no
            // need to inspect the rest of its subtree. Remember only
            // candidates that would expand to the same set. A descendant or
            // sibling can still leave independently.
            const auto remember_same_set = [&](std::span<const NodeEntry> entries) {
                for (const NodeEntry& entry : entries) {
                    if (!request_reclaim_locations_.contains(entry.location) &&
                        (groups_[entry.group_id].Matcher().IsPrefixClosed() || store_.GroupIsState(entry.group_id))) {
                        unevictable_locations_.insert(entry.location);
                    }
                }
            };
            remember_same_set(candidate_entries);
            for (const auto& [node, entries] : climbed) {
                remember_same_set(entries);
            }
            return std::nullopt;
        }
        out.segment_top = top;
        return out;
    }

    void initializeCapacity() {
        _assert(demands_.size() == groups_.size(), "demands/groups size mismatch");
        _assert(shapes_.empty() || shapes_.size() == groups_.size(), "shapes/groups size mismatch");
        pair_need_.assign(groups_.size(), 0);
        for (std::vector<CopyInReservation>& known : prefix_tail_buckets_) {
            known.clear();
        }
        for (std::vector<CopyInReservation>& copy_ins : copy_ins_) {
            copy_ins.clear();
        }
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const GroupDemand& demand = demands_[i];
            _assert(demand.table != nullptr, "group demand requires a block table");
            const auto buckets = static_cast<std::size_t>(groups_[i].Allocator().ShardCount());
            if (shapes_.empty()) {
                // An admission grows the live table densely; each Host hit
                // and each Device tail copy lands on a fresh Device block in
                // its source block's own bucket, exactly like a restored slot.
                shape_need_[i].assign(buckets, 0);
                std::int32_t growth_blocks = geometry_[i].BlocksNeededFor(*demand.table, demand);
                if (const auto* sparse = std::get_if<SparseSuffix>(&demand.extent); sparse && sparse->head_blocks > 0) {
                    // A matched sink is installed before growth: Device hits
                    // are shared, Host hits and tail copies are charged below.
                    const auto prefix_tokens =
                        std::max(prefix_.device.num_common_tokens, prefix_.host.num_common_tokens);
                    const auto grain = geometry_[i].BlockGranularity();
                    const auto prefix_blocks = prefix_tokens / grain + (prefix_tokens % grain != 0);
                    growth_blocks = geometry_[i].BlocksNeededFor(std::max(demand.table->NumBlocks(), prefix_blocks),
                                                                 *sparse, demand.reserve_tokens);
                }
                blocks_needed_[i] = growth_blocks + tailBlocksNeeded(i, shape_need_[i]) + zeroStateBlocksNeeded(i);
                if (!prefix_.host.per_group.empty()) {
                    collectHostHitBuckets(i, shape_need_[i]);
                }
                // After the prefix's blocks: a copy-in slot may hold one.
                blocks_needed_[i] -= reserveCopyIns(i, shape_need_[i]);
            } else {
                // A restore rebuilds the recorded shape, bucket for bucket,
                // and grows the rebuilt table from its recorded tail.
                _assert(demand.table->NumBlocks() == 0, "a restore rebuilds into an empty table");
                _assert(shapes_[i].blocks_by_bucket.size() == buckets,
                        "restore shape buckets must match the shard count");
                if (const auto* step = std::get_if<StateStep>(&demand.extent)) {
                    _assert(step->kind == StateStep::Kind::kKeep, "a restore rebuilds a State table as imaged");
                    blocks_needed_[i] = 0;
                } else {
                    const auto* dense = std::get_if<DenseGrowth>(&demand.extent);
                    _assert(dense != nullptr, "a restore demand grows the rebuilt table densely");
                    blocks_needed_[i] = geometry_[i].BlocksNeededBeyond(shapes_[i].available_tokens,
                                                                        dense->num_tokens + demand.reserve_tokens);
                }
                shape_need_[i] = shapes_[i].blocks_by_bucket;
                blocks_needed_[i] -= reserveCopyIns(i, shape_need_[i]);
            }
        }
    }

    // Blocks a tail costs this group beyond the growth demand. A table never
    // keeps a tail entry's block: the block holds the rows of a page that is
    // later published under the page's own hash, and one block carries one
    // key. So on a first admission every block of a Device tail becomes the
    // hitter's own (a copy of the shared block, or a fresh block where the
    // probe left a hole). A copy takes its fresh block in its source's bucket
    // (SwapInFreshBlock), so it is added to `need` like a Host hit and
    // recorded for its slot (prefix_tail_buckets_); a hole's fresh block takes
    // any bucket and is returned with the growth. A dense growth starts from
    // the capacity the last tail block leaves, which the live table cannot
    // show yet; a sparse suffix (the D role's landing of a sliding window)
    // starts at an absolute block past the tail and is costed whole by
    // BlocksNeededFor.
    std::int64_t tailBlocksNeeded(std::size_t i, std::vector<std::int64_t>& need) {
        if (!geometry_[i].HasTokenSpan()) {
            return 0;  // a State group's step is costed by BlocksNeededFor
        }
        const std::int32_t block_granularity = geometry_[i].BlockGranularity();
        const GroupAllocator& allocator = groups_[i].Allocator();
        std::int64_t blocks = 0;
        if (const auto& tail = prefix_.device.tail; tail && !tail->group_keys[i].empty()) {
            const std::vector<CacheBlockLocation> copied =
                groups_[i].Index().MatchedLocations(pool_, tail->group_keys[i], /*begin_blocks=*/0, tail->per_group[i]);
            // Key j of the tail is the j-th block of its page.
            const std::int32_t first_slot =
                tailSlots(geometry_[i], prefix_granularity_, prefix_.device.num_common_tokens).first;
            auto source = copied.begin();
            for (std::size_t j = 0; j < tail->per_group[i].hits.size(); ++j) {
                if (tail->per_group[i].hits[j] == 0) {
                    continue;
                }
                const std::int32_t bucket = allocator.BucketOf(*source++);
                ++need[static_cast<std::size_t>(bucket)];
                prefix_tail_buckets_[i].push_back(
                    CopyInReservation{.slot = first_slot + static_cast<std::int32_t>(j), .bucket = bucket});
            }
            blocks += static_cast<std::int64_t>(tail->group_keys[i].size()) - static_cast<std::int64_t>(copied.size()) -
                      FirstTailBlockNeeded(groups_[i].Spec(), block_granularity, tail->tail_tokens);
            if (const auto* dense = std::get_if<DenseGrowth>(&demands_[i].extent)) {
                const std::int32_t partial_rows = tail->tail_tokens % block_granularity;
                const std::int32_t available = partial_rows == 0 ? 0 : block_granularity - partial_rows;
                const std::int32_t growth = dense->num_tokens + demands_[i].reserve_tokens;
                blocks += geometry_[i].BlocksNeededBeyond(available, growth) -
                          geometry_[i].BlocksNeededFor(*demands_[i].table, growth);
            } else {
                const auto* sparse = std::get_if<SparseSuffix>(&demands_[i].extent);
                _assert(sparse != nullptr && sparse->first_block * block_granularity >= tail->tail_tokens,
                        "a sparse suffix lands past the hit's tail blocks");
            }
        }
        return blocks;
    }

    // Charges the copy-in blocks this admission reserves for the request's
    // copy-in point (copy_in_point_): one for each slot of CopyInSlots whose
    // block this admission allocates, fills from the prefix or rebuilds. A
    // copy-in block is reserved with the block it serves, so a request whose
    // first admission prepaid its prompt holds them from that admission on.
    // Each shares its slot's bucket, so its copy's two ends have one owner
    // under page-cyclic sharding: a slot the prefix fills (a hitter's copy, a
    // Host load: prefix_tail_buckets_) or a restore rebuilds has its block's
    // bucket, counted in `need`; a slot the dense growth fills takes its
    // block and its copy-in block in one bucket the fit chooses, as a pair
    // (pair_need_), so the growth counts that block no more. A slot an
    // earlier admission allocated holds its copy-in block already; one past
    // the growth gets it with its block. Records the reservations
    // (copy_ins_) and returns the number of pairs.
    std::int64_t reserveCopyIns(std::size_t i, std::vector<std::int64_t>& need) {
        if (!copy_in_point_) {
            return 0;
        }
        const CacheStore::BlockRange slots = store_.CopyInSlots(i, *copy_in_point_);
        if (slots.first == slots.past_end) {
            return 0;
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
        for (std::int32_t slot = slots.first; slot < std::min(slots.past_end, growth_end); ++slot) {
            std::int32_t bucket = -1;
            if (slot >= base) {
                ++pair_need_[i];
            } else if (!shapes_.empty()) {
                bucket = known_bucket(shapes_[i].rebuilt_copy_ins, slot);
            } else if (table.NumBlocks() > 0) {
                continue;  // allocated, with its copy-in block, by an earlier admission
            } else {
                bucket = known_bucket(prefix_tail_buckets_[i], slot);
            }
            if (bucket >= 0) {
                ++need[static_cast<std::size_t>(bucket)];
            }
            copy_ins_[i].push_back(CopyInReservation{.slot = slot, .bucket = bucket});
        }
        return pair_need_[i];
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
    // slot (prefix_tail_buckets_).
    void collectHostHitBuckets(std::size_t group_index, std::vector<std::int64_t>& need) {
        _assert(host_pool_ != nullptr, "Host prefix hits require a Host pool");
        const GroupPrefixProbe& probe = prefix_.host.per_group[group_index];
        const std::span<const CacheKey> keys = (*prefix_.group_keys)[group_index];
        // A State group's hits count from page 0 (PrefixProbe::Tier).
        const std::int32_t floor_pages =
            geometry_[group_index].HasTokenSpan()
                ? prefix_.device.full_page_tokens / geometry_[group_index].BlockGranularity()
                : 0;
        const GroupAllocator& allocator = groups_[group_index].Allocator();
        const auto count_hit = [&](const CacheKey& key) {
            // A replicated group has only bucket 0, independent of whether
            // this source is already on Host or is an incoming L3 row.
            if (allocator.ShardCount() == 1) {
                ++need[0];
                return 0;
            }
            const CacheBlockRef host_block = groups_[group_index].Index().Find(*host_pool_, key);
            const std::int32_t bucket = host_block ? allocator.BucketOf(host_block->Location()) : 0;
            ++need[static_cast<std::size_t>(bucket)];
            return bucket;
        };
        for (std::size_t hit_index = 0; hit_index < probe.hits.size(); ++hit_index) {
            if (probe.hits[hit_index] == 0) {
                continue;
            }
            const std::size_t key_index = static_cast<std::size_t>(floor_pages) + hit_index;
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
                    prefix_tail_buckets_[group_index].push_back(
                        CopyInReservation{.slot = first_slot + static_cast<std::int32_t>(j), .bucket = bucket});
                }
            }
        }
    }

    // Every block group i's demand takes: in named buckets, in any bucket,
    // and both blocks of each pair.
    std::int64_t blocksDemanded(std::size_t i) const {
        std::int64_t need = blocks_needed_[i] + 2 * pair_need_[i];
        for (const std::int64_t bucket_need : shape_need_[i]) {
            need += bucket_need;
        }
        return need;
    }

    bool couldFitAfterEviction() const {
        std::int64_t parents_needed = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::int64_t need = blocksDemanded(i);
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
                    const std::int32_t expired = geometry_[i].ExpiredBlocksAt(groups_[i].Spec(), *num_computed_tokens_);
                    request_reclaim = groups_[i].Allocator().ExpiredBlockSlotsAt(table, expired).size();
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

    void collectCandidates() {
        eviction_plan_.StartCandidates(&protected_locations_, &request_reclaim_locations_, device_aging_);
        const auto protect = [&](std::span<const CacheBlockLocation> locations) {
            protected_locations_.reserve(protected_locations_.size() + locations.size());
            for (const CacheBlockLocation location : locations) {
                protected_locations_.insert(location);
            }
        };
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            protect(groups_[i].Index().MatchedLocations(pool_, (*prefix_.group_keys)[i], /*begin_blocks=*/0,
                                                        prefix_.device.per_group[i]));
            if (const auto& tail = prefix_.device.tail; tail && !tail->group_keys[i].empty()) {
                protect(groups_[i].Index().MatchedLocations(pool_, tail->group_keys[i], /*begin_blocks=*/0,
                                                            tail->per_group[i]));
            }
            if (const auto& state = prefix_.state_at_prompt_end; state && !state->group_keys[i].empty()) {
                protect(groups_[i].Index().MatchedLocations(pool_, state->group_keys[i], /*begin_blocks=*/0,
                                                            state->per_group[i]));
            }
        }
        protected_locations_.reserve(protected_locations_.size() + claimed_.size());
        for (const auto& [group_id, location] : claimed_) {
            (void)group_id;
            protected_locations_.insert(location);
        }

        if (num_computed_tokens_) {
            collectRequestReclaimCandidates(*num_computed_tokens_);
        }
    }

    // Ignore segment restrictions, but preserve actual parent occupancy and
    // bucket placement. If releasing every available block still cannot fit,
    // no victim ordering can help. Request releases often fund a State rotation
    // by themselves, so test those before visiting the cache's available index.
    bool couldFitAfterReleasingCandidates() {
        ShadowOccupancy maximum{pool_, groups_};
        for (const VictimCandidate& candidate : request_reclaim_candidates_) {
            maximum.Remove(candidate.group_id, candidate.location);
        }
        // The fit is asked again only once enough blocks have left to close
        // the shortfall (ShadowOccupancy::ParentShortfall): at most once
        // before an epoch's end and once at it, never in an epoch too short
        // to complete the fit. The walk thus usually stops releasing at the
        // block that completes the fit; that block's epoch is still read
        // whole.
        std::int64_t removals_until_fit = maximum.ParentShortfall(shape_need_, blocks_needed_, pair_need_);
        if (removals_until_fit == 0) {
            return true;
        }
        return eviction_plan_.Candidates().PreScanUntil(
            [&](std::uint32_t group_id, std::span<const PrefixCacheIndex::EvictionCandidate> entries) {
                bool asked = false;
                for (const auto& entry : entries) {
                    if (protected_locations_.contains(entry.location) ||
                        request_reclaim_locations_.contains(entry.location)) {
                        continue;
                    }
                    maximum.Remove(group_id, entry.location);
                    if (removals_until_fit > 0) {
                        --removals_until_fit;
                    }
                    if (removals_until_fit == 0 && !asked) {
                        asked = true;
                        removals_until_fit = maximum.ParentShortfall(shape_need_, blocks_needed_, pair_need_);
                        if (removals_until_fit == 0) {
                            return true;
                        }
                    }
                }
                if (removals_until_fit == 0) {
                    removals_until_fit = maximum.ParentShortfall(shape_need_, blocks_needed_, pair_need_);
                }
                return removals_until_fit == 0;
            });
    }

    // The request's own slots that its retention has expired at this
    // progress, and a State table's slot 0 when this step releases it. They
    // are still pinned by the table here; Admit publishes and reclaims them
    // before it evicts, so they can fund this same admission.
    void collectRequestReclaimCandidates(std::int32_t num_computed_tokens) {
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::uint32_t group_id = static_cast<std::uint32_t>(i);
            std::vector<CacheBlockLocation> reclaimable;
            if (const auto* step = std::get_if<StateStep>(&demands_[i].extent)) {
                if (const auto location =
                        groups_[i].Allocator().ReleasableStateInput(groups_[i].Index(), *demands_[i].table, *step)) {
                    reclaimable.push_back(*location);
                }
            } else {
                const std::int32_t expired_blocks =
                    geometry_[i].ExpiredBlocksAt(groups_[i].Spec(), num_computed_tokens);
                reclaimable = groups_[i].Allocator().ReclaimableBlockLocationsAt(groups_[i].Index(), *demands_[i].table,
                                                                                 expired_blocks);
            }
            for (CacheBlockLocation location : reclaimable) {
                if (protected_locations_.contains(location)) {
                    continue;
                }
                const bool first_occurrence = request_reclaim_locations_.insert(location);
                if (!first_occurrence) {
                    continue;
                }
                request_reclaim_candidates_.push_back(eviction_plan_.Candidates().MakeCandidate(
                    group_id, location, groups_[i].Index().MetadataFor(pool_, location)));
            }
        }
        eviction_plan_.Candidates().Sort(request_reclaim_candidates_);
    }

    // The store planned on, read through its public views (NodeEntries,
    // GroupIsState, CopyInSlots); groups_, geometry_, the pools and the
    // boundary tree are its own.
    const CacheStore& store_;
    CheckpointAging device_aging_;
    const std::vector<CacheGroup>& groups_;
    std::span<const GroupGeometry> geometry_;
    std::int32_t prefix_granularity_;
    const BlockPool& pool_;
    const BlockPool* host_pool_;
    std::span<const GroupDemand> demands_;
    std::optional<std::int32_t> num_computed_tokens_;
    // The point whose copy-in blocks this admission reserves (Admit).
    std::optional<std::int32_t> copy_in_point_;
    const PrefixProbe& prefix_;
    std::span<const PlannedShape> shapes_;
    // Device-cached blocks a restore claims instead of copying: protected
    // from eviction like the probe's prefix hits.
    std::span<const std::pair<std::uint32_t, CacheBlockLocation>> claimed_;
    const BoundaryTree& boundary_tree_;
    // Whether a victim takes its segment along
    // (CacheCoordinator::WritesBackEvictions).
    bool evicts_segments_;
    std::vector<VictimSet>& victims_;
    std::pmr::memory_resource* member_resource_;
    // The plan's copy-in reservations (AdmissionPlan::copy_ins).
    std::vector<std::vector<CopyInReservation>>& copy_ins_;
    // The plan's lookup tables, open-addressed in this arena (FlatHashMap),
    // which frees nothing until the planner dies; returned victim vectors
    // own their storage independently. No table is iterated, so no plan
    // depends on their slot order. The arena starts in the planner's own raw
    // buffer (left uninitialized: only the arena writes it), so a small
    // admission calls no allocator for its tables.
    static constexpr std::size_t kInlineScratchBytes = 16 * 1024;
    alignas(std::max_align_t) std::array<std::byte, kInlineScratchBytes> scratch_buffer_;
    std::pmr::monotonic_buffer_resource scratch_{scratch_buffer_.data(), scratch_buffer_.size()};
    // The pool as the tentative victims would leave it.
    CacheEvictionPlan eviction_plan_;
    // Per group: the bucket-constrained blocks (a restore's rebuilt slots; an
    // admission's Host-hit destinations and tail copies; the copy-in blocks
    // of slots whose blocks are in a known bucket), and the dense blocks
    // wanted in any bucket.
    std::vector<std::vector<std::int64_t>> shape_need_;
    std::vector<std::int64_t> blocks_needed_;
    // Per group, the block pairs whose shared bucket the fit chooses: a tail
    // block the dense growth fills and its copy-in block (reserveCopyIns).
    std::vector<std::int64_t> pair_need_;
    // Per group, the slots a first admission's prefix tail fills with a block
    // in a known bucket -- a Device tail's copies, a Host tail's loads --
    // with that bucket, for a copy-in slot among them (reserveCopyIns).
    std::vector<std::vector<CopyInReservation>> prefix_tail_buckets_;
    std::vector<VictimCandidate> request_reclaim_candidates_;
    CacheEvictionCandidates::LocationSet protected_locations_{&scratch_};
    CacheEvictionCandidates::LocationSet request_reclaim_locations_{&scratch_};
    // Every block a selected set holds: the set it joined (or last moved
    // into) and its position there. A later set may since have taken that
    // set's lists over in place, positions unchanged: taken_by_ leads there.
    FlatHashMap<CacheBlockLocation, std::pair<std::size_t, std::size_t>> set_member_by_location_{&scratch_};
    // Per set: the later set that took its lists over in place, else itself.
    std::vector<std::size_t> taken_by_;
    // The selected segments by the tree node each starts at: every Device block
    // of that node's subtree, in post-order.
    FlatHashMap<const BoundaryTree::Node*, std::size_t> segment_at_{&scratch_};
    // Blocks that expand to a set found unable to leave under the current
    // reclaim selection: skipped until a retry invalidates that decision.
    FlatHashSet<CacheBlockLocation> unevictable_locations_{&scratch_};
    // Pages two or more Device checkpoint boundaries depend on, passed over
    // while candidates last, oldest first: tried again once they run out.
    std::vector<VictimCandidate> deferred_shared_pages_;
    // Candidates no segment rule let leave (a set member in use or kept, or a
    // shared page still shared once the candidates ran out), in the order
    // they were passed over: retried as segments after reclaim selection
    // changes, then block by block if the admission permits it.
    std::vector<VictimCandidate> passed_over_;
    // Per tree node: how many checkpoint boundaries at or under it have every
    // Device checkpoint in a selected set, counted once the candidates run out.
    FlatHashMap<const BoundaryTree::Node*, std::int32_t> leaving_checkpoint_boundary_counts_{&scratch_};
};

// Plans with `Planner` (AdmissionPlanner, or a subclass that goes on past
// the segment rules) and packages the plan for the commit.
template <typename Planner>
std::optional<AdmissionPlan> planAdmissionWith(const CacheStore& store, CheckpointAging device_aging,
                                               PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                               std::optional<std::int32_t> num_computed_tokens,
                                               std::optional<std::int32_t> copy_in_point,
                                               std::span<const PlannedShape> shapes,
                                               std::span<const std::pair<std::uint32_t, CacheBlockLocation>> claimed,
                                               bool evicts_segments) {
    ProfileSpan profile_span{"planAdmission"};
    _assert(demands.size() == store.Groups().size(), "demands/groups size mismatch");

    auto member_resource = std::make_unique<std::pmr::monotonic_buffer_resource>();
    std::vector<VictimSet> victims;
    std::vector<std::vector<CopyInReservation>> copy_ins;
    Planner planner{store,   device_aging,    demands, num_computed_tokens,   copy_in_point, prefix, shapes,
                    claimed, evicts_segments, victims, member_resource.get(), copy_ins};
    if (!planner.Plan()) {
        return std::nullopt;
    }
    return AdmissionPlan{.prefix = std::move(prefix),
                         .member_resource = std::move(member_resource),
                         .victims = std::move(victims),
                         .copy_ins = std::move(copy_ins)};
}

inline std::optional<AdmissionPlan> planAdmission(const CacheStore& store, CheckpointAging device_aging,
                                                  PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                                  std::optional<std::int32_t> num_computed_tokens,
                                                  std::optional<std::int32_t> copy_in_point,
                                                  std::span<const PlannedShape> shapes,
                                                  std::span<const std::pair<std::uint32_t, CacheBlockLocation>> claimed,
                                                  bool evicts_segments) {
    return planAdmissionWith<AdmissionPlanner>(store, device_aging, std::move(prefix), demands, num_computed_tokens,
                                               copy_in_point, shapes, claimed, evicts_segments);
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
        /*claimed=*/{}, write_back_evictions_);
    if (!plan) {
        return std::nullopt;
    }
    return commitAdmission(std::move(*plan), demands, progress, copy_in_blocks, copy_in_point, request_access_epoch,
                           session_id, cache_group_id);
}

}  // namespace tokenspeed
