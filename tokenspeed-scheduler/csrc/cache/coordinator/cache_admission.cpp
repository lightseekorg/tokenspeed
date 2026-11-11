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

#include "cache/coordinator/cache_coordinator.h"

#include "core/profile_span.h"
#include "cache/coordinator/eviction_plan.h"
#include "cache/coordinator/group_geometry.h"
#include "cache/coordinator/shadow_occupancy.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory_resource>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils.h"

namespace tokenspeed {
namespace {

// Blocks a plan evicts together, in the order they leave: one block, or,
// when the coordinator evicts by segments, a whole segment -- deepest node
// first, a node's State checkpoints before its other blocks.
struct VictimSet {
    struct Member {
        std::uint32_t group_id;
        CacheBlockLocation location;
        // Whether it lies strictly under the boundary of the candidate the
        // set was chosen for, taken along so no Device page outlives its
        // parent (counted as a stranded suffix).
        bool in_stranded_suffix;
        // Kept per block when a segment absorbs the request's released slots.
        bool is_request_reclaim;
    };
    std::vector<Member> members;
};

// A copy-in block an admission reserves: the table slot it serves and the
// bucket it takes, the bucket of that slot's block -- for a slot the
// admission's dense growth fills, the bucket the fit chose for the block and
// its copy-in block together (-1 until it has).
struct CopyInReservation {
    std::int32_t slot{0};
    std::int32_t bucket{-1};
};

struct AdmissionPlan {
    CacheCoordinator::PrefixProbe prefix;
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

TailSlots tailSlots(const GroupGeometry& geometry, std::int32_t prefix_granularity, std::int32_t tokens) {
    const std::int32_t block_granularity = geometry.BlockGranularity();
    const std::int32_t page_tokens = tokens - tokens % prefix_granularity;
    const std::int32_t past_end = (tokens + block_granularity - 1) / block_granularity;
    return TailSlots{.first = page_tokens / block_granularity,
                     .past_end = past_end,
                     .available_tokens = past_end * block_granularity - tokens};
}

class AdmissionPlanner {
public:
    AdmissionPlanner(const CacheCoordinator& coordinator, const std::vector<CacheGroup>& groups,
                     std::span<const GroupGeometry> geometry, std::int32_t prefix_granularity, const BlockPool& pool,
                     const BlockPool* host_pool, std::span<const GroupDemand> demands,
                     std::optional<std::int32_t> num_computed_tokens, std::optional<std::int32_t> copy_in_point,
                     const CacheCoordinator::PrefixProbe& prefix, std::span<const PlannedShape> shapes,
                     std::span<const std::pair<std::uint32_t, CacheBlockLocation>> claimed,
                     const BoundaryTree& boundary_tree, bool evicts_segments, bool evicts_blocks_past_segments,
                     std::vector<VictimSet>& victims, std::vector<std::vector<CopyInReservation>>& copy_ins)
        : coordinator_{coordinator},
          groups_{groups},
          geometry_{geometry},
          prefix_granularity_{prefix_granularity},
          pool_{pool},
          host_pool_{host_pool},
          demands_{demands},
          num_computed_tokens_{num_computed_tokens},
          copy_in_point_{copy_in_point},
          prefix_{prefix},
          shapes_{shapes},
          claimed_{claimed},
          boundary_tree_{boundary_tree},
          evicts_segments_{evicts_segments},
          evicts_blocks_past_segments_{evicts_blocks_past_segments},
          victims_{victims},
          copy_ins_{copy_ins},
          eviction_plan_{pool, groups},
          shape_need_(groups.size()),
          blocks_needed_(groups.size()),
          pair_need_(groups.size(), 0),
          prefix_tail_buckets_(groups.size()) {
        copy_ins_.assign(copy_in_point ? groups.size() : 0, {});
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

    // The parents a dense layout of the request's blocks and this demand
    // would take, every parent empty: per group ParentsNeededForBucketsAndPairs
    // over the blocks that keep their bucket (bucket_pinned, by bucket) plus
    // the bucket-exact demand, the blocks that may take any bucket
    // (free_movers, a request's unwritten reserve) plus the dense demand, and
    // the demand's same-bucket pairs (reserveCopyIns). The gate of
    // CacheCoordinator::CompactLoneRequest, and what prepares the planner's
    // capacity and candidate views for PlanBlockAlone.
    std::int64_t DenseLayoutParents(std::span<const std::vector<std::int64_t>> bucket_pinned,
                                    std::span<const std::int64_t> free_movers) {
        _assert(bucket_pinned.size() == groups_.size() && free_movers.size() == groups_.size(),
                "one bucket layout and one free-mover count per group");
        if (!std::exchange(views_prepared_, true)) {
            initializeCapacity();
            collectCandidates();
        }
        std::int64_t parents = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            _assert(bucket_pinned[i].size() == shape_need_[i].size(), "bucket layout must match the shard count");
            std::vector<std::int64_t> need(shape_need_[i].size());
            for (std::size_t b = 0; b < need.size(); ++b) {
                need[b] = bucket_pinned[i][b] + shape_need_[i][b];
            }
            const std::vector<std::int64_t> holes(need.size(), 0);
            std::vector<std::int64_t> scratch(need.size(), 0);
            parents += ParentsNeededForBucketsAndPairs(need, holes, free_movers[i] + blocks_needed_[i], pair_need_[i],
                                                       pool_.Packing(groups_[i].Id()), scratch);
        }
        return parents;
    }

    // The cache entry at `location`, expanded as the block-by-block pass
    // takes it (expandVictimSet with blocks_alone): the entry alone, or under
    // the write-back policy with everything the Device holds under its node;
    // nullopt when no entry the index alone holds is there (a free slot, a
    // block a table or a ticket holds), when a block under it is in use or
    // kept, or when the entry itself is one this admission keeps -- a page,
    // tail block or State checkpoint its probe hit, a restore's claim -- or
    // a slot the retry reclaims as the request's own: the candidate pass
    // never offers those (StartCandidates), so a compaction never clears
    // them either.
    std::optional<VictimSet> PlanBlockAlone(std::uint32_t group_id, CacheBlockLocation location) {
        _assert(views_prepared_, "DenseLayoutParents prepares the planner's views before a slot is planned alone");
        if (!groups_[group_id].Index().UnpinnedKeyAt(pool_, location) || protected_locations_.contains(location) ||
            request_reclaim_locations_.contains(location)) {
            return std::nullopt;
        }
        std::optional<ExpandedVictimSet> expanded =
            expandVictimSet(eviction_plan_.Candidates().MakeCandidate(
                                group_id, location, groups_[group_id].Index().MetadataFor(pool_, location)),
                            /*blocks_alone=*/true, /*candidates_ran_out=*/true);
        if (!expanded) {
            return std::nullopt;
        }
        _assert(std::ranges::find(expanded->set.members, location, &VictimSet::Member::location) !=
                    expanded->set.members.end(),
                "a victim set holds the candidate it was chosen for");
        return std::move(expanded->set);
    }

    // Whether `location` is one of the request's own slots the admission
    // reclaims before it evicts (an expired window block, a released State
    // input): the retry frees it, though PlanBlockAlone never takes it.
    bool IsRequestReclaim(CacheBlockLocation location) const {
        _assert(views_prepared_, "DenseLayoutParents prepares the planner's views before a slot is judged");
        return request_reclaim_locations_.contains(location);
    }

private:
    using VictimCandidate = CacheEvictionCandidates::Candidate;

    bool fitWithEviction() {
        ProfileSpan profile_span{"AdmissionPlanner::fitWithEviction"};
        // Existing local holes plus empty parents are the zero-eviction
        // capacity. Do not discard cache merely to make placement denser: a
        // request's own scattered blocks are gathered by its compaction
        // (CacheCoordinator::CompactLoneRequest) once it is alone and nothing
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
        using NodeEntry = CacheCoordinator::NodeEntry;
        bool candidates_ran_out = false;
        std::size_t next_deferred_shared_page = 0;
        // Once the shared pages ran out as well: the candidates the segment
        // rules passed over, tried again block by block.
        bool blocks_alone = false;
        std::size_t next_passed_over = 0;
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
                        std::pmr::unordered_set<const BoundaryTree::Node*> counted{&scratch_};
                        for (const VictimSet& set : victims_) {
                            for (const VictimSet::Member& member : set.members) {
                                if (!coordinator_.GroupIsState(member.group_id)) {
                                    continue;
                                }
                                const std::optional<CacheKey> key =
                                    groups_[member.group_id].Index().KeyOf(pool_, member.location);
                                const BoundaryTree::Node* checkpoint_node =
                                    key ? boundary_tree_.Find(key->content_hash) : nullptr;
                                if (checkpoint_node == nullptr || !counted.insert(checkpoint_node).second ||
                                    !std::ranges::all_of(coordinator_.NodeEntries(pool_, *checkpoint_node),
                                                         [&](const NodeEntry& entry) {
                                                             return !coordinator_.GroupIsState(entry.group_id) ||
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
                if (!candidate && !blocks_alone && next_retry_segment == retry_segments.size() &&
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
                if (!candidate && evicts_blocks_past_segments_ && next_passed_over < passed_over_.size()) {
                    blocks_alone = true;
                    candidate = passed_over_[next_passed_over++];
                }
                if (!candidate) {
                    return false;
                }
                std::optional<ExpandedVictimSet> expanded =
                    expandVictimSet(*candidate, blocks_alone, candidates_ran_out);
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
                    const auto it = set_member_by_location_.find(member.location);
                    if (it == set_member_by_location_.end()) {
                        eviction_plan_.Remove(member.group_id, member.location);
                        set_member_by_location_.emplace(member.location, std::pair{set_index, j});
                        reclaim_selection_changed = reclaim_selection_changed || member.is_request_reclaim;
                        continue;
                    }
                    // The set holding the block now, through the sets that took
                    // lists over.
                    std::size_t earlier = it->second.first;
                    while (taken_by_[earlier] != earlier) {
                        earlier = taken_by_[earlier] = taken_by_[taken_by_[earlier]];
                    }
                    const std::size_t position = it->second.second;
                    member = victims_[earlier].members[position];
                    if (std::ranges::find(absorbed, earlier) == absorbed.end()) {
                        absorbed.push_back(earlier);
                    }
                    ++absorbed_members;
                    it->second = std::pair{set_index, j};
                }
                for (const std::size_t earlier : absorbed) {
                    absorbed_members -= victims_[earlier].members.size();
                    victims_[earlier] = VictimSet{};
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

    // What expandVictimSet returns beside the set: how many of its first
    // members a segment took over in place from an earlier set (their
    // set_member_by_location_ entries as they are) and that set's index, and
    // the tree node a segment set starts at (nullptr block by block).
    struct ExpandedVictimSet {
        VictimSet set;
        std::size_t num_taken_over_members{0};
        std::optional<std::size_t> taken_over_set_index;
        const BoundaryTree::Node* segment_top{nullptr};
    };
    // One candidate expanded to the set it leaves with -- as a segment, block
    // by block, or as the request's own reclaim -- or nullopt when it is
    // skipped this pass (a block in the set is in use or kept; the shared
    // page is deferred). Pure apart from the pass's own memo lists
    // (passed_over_, deferred_shared_pages_, unevictable_locations_) and,
    // for a segment, the earlier sets it takes over in place.
    std::optional<ExpandedVictimSet> expandVictimSet(const VictimCandidate& candidate, bool blocks_alone,
                                                     bool candidates_ran_out) {
        using NodeEntry = CacheCoordinator::NodeEntry;
        // A selected request reclaim still has its table reference until Admit
        // executes. A later segment may absorb that set; the merge in
        // fitWithEviction keeps its capacity counted once and Minimize must
        // restore them together.
        const auto is_evictable = [&](const NodeEntry& entry) {
            return set_member_by_location_.contains(entry.location) ||
                   (!entry.in_use && !protected_locations_.contains(entry.location) &&
                    !request_reclaim_locations_.contains(entry.location));
        };
        // The Device checkpoint boundaries at or under a node that no selected
        // set takes whole: every one while candidates last, then those staying.
        const auto num_staying_checkpoint_boundaries = [&](const BoundaryTree::Node& node) {
            const auto it = leaving_checkpoint_boundary_counts_.find(&node);
            return node.checkpoint_boundaries_in_subtree[BoundaryTree::kDevice] -
                   (it == leaving_checkpoint_boundary_counts_.end() ? 0 : it->second);
        };
        const std::uint32_t group_id = candidate.group_id;
        const CacheBlockLocation location = candidate.location;
        // Already leaving with an earlier set, or found unable to leave
        // (passed over: tried again block by block).
        if (set_member_by_location_.contains(location)) {
            return std::nullopt;
        }
        if (!blocks_alone && unevictable_locations_.contains(location)) {
            passed_over_.push_back(candidate);
            return std::nullopt;
        }
        const bool is_request_reclaim = request_reclaim_locations_.contains(location);
        VictimSet set;
        // A segment that takes an earlier segment's lists over in place
        // keeps them first (the first num_taken_over_members members, their
        // set_member_by_location_ entries as they are).
        std::size_t num_taken_over_members = 0;
        std::optional<std::size_t> taken_over_set_index;
        // The tree node a segment set starts at.
        const BoundaryTree::Node* segment_top = nullptr;
        const BoundaryTree::Node* candidate_node = nullptr;
        if (evicts_segments_) {
            if (const std::optional<CacheKey> key = groups_[group_id].Index().KeyOf(pool_, location)) {
                candidate_node = boundary_tree_.Find(key->content_hash);
            }
        }
        const bool sliding = !groups_[group_id].Matcher().IsPrefixClosed() && !coordinator_.GroupIsState(group_id);
        if (candidate_node == nullptr || (sliding && !is_request_reclaim)) {
            // Block by block; or a block with no tree node (never
            // published, or registered straight into an index); or a window
            // block, which only shortens the boundaries its prefix resumes
            // at.
            set.members.push_back(VictimSet::Member{.group_id = group_id,
                                                    .location = location,
                                                    .in_stranded_suffix = false,
                                                    .is_request_reclaim = is_request_reclaim});
        } else if (is_request_reclaim) {
            // The request's released slots at this hash, and at a tail
            // the tail's other blocks, which their eviction takes along.
            for (const NodeEntry& entry : coordinator_.NodeEntries(pool_, *candidate_node)) {
                if (request_reclaim_locations_.contains(entry.location) ||
                    (candidate_node->tail_tokens > 0 && is_evictable(entry))) {
                    set.members.push_back(VictimSet::Member{.group_id = entry.group_id,
                                                            .location = entry.location,
                                                            .in_stranded_suffix = false,
                                                            .is_request_reclaim = true});
                }
            }
        } else if (blocks_alone ? coordinator_.GroupIsState(group_id)
                                : num_staying_checkpoint_boundaries(*candidate_node) >= 2) {
            // Deeper checkpoint boundaries still read this page. A nested
            // checkpoint leaves alone, every State group's at once; the
            // page leaves with the last checkpoint boundary under it, or is
            // tried again once the candidates run out. Block by block, every
            // checkpoint leaves so; at a tail its eviction takes the tail's
            // other unpinned blocks along (evictCachedBlock), uncounted here.
            if (!coordinator_.GroupIsState(group_id)) {
                (candidates_ran_out ? passed_over_ : deferred_shared_pages_).push_back(candidate);
                return std::nullopt;
            }
            bool evictable = true;
            for (const NodeEntry& entry : coordinator_.NodeEntries(pool_, *candidate_node)) {
                if (coordinator_.GroupIsState(entry.group_id)) {
                    evictable = evictable && is_evictable(entry);
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
                return std::nullopt;
            }
        } else {
            // A segment: the one checkpoint boundary at or under this
            // boundary with the pages only it depends on -- the climb stops
            // below a page another checkpoint boundary shares, one in use
            // or a hit this admission keeps, or one the Device does not
            // hold -- and everything the Device holds under them, so no
            // Device page outlives its parent. The climb keeps the entries
            // of each node it passes, from the candidate's parent up. Block
            // by block, a page leaves with its boundary's blocks not in use
            // and everything under it, climbing nowhere.
            const BoundaryTree::Node* top = candidate_node;
            const std::vector<NodeEntry> candidate_entries = coordinator_.NodeEntries(pool_, *candidate_node);
            std::vector<std::pair<const BoundaryTree::Node*, std::vector<NodeEntry>>> climbed;
            if (!blocks_alone && num_staying_checkpoint_boundaries(*candidate_node) == 1) {
                while (top->parent != nullptr && top->parent->OnDevice() &&
                       num_staying_checkpoint_boundaries(*top->parent) == 1) {
                    std::vector<NodeEntry> entries = coordinator_.NodeEntries(pool_, *top->parent);
                    if (!std::ranges::all_of(entries, is_evictable)) {
                        break;
                    }
                    top = top->parent;
                    climbed.emplace_back(top, std::move(entries));
                }
            }
            // Post-order: a node after the nodes under it, its own entries
            // in NodeEntries order. It completes the climbed nodes in climb
            // order. A child the Device does not hold has nothing on the
            // Device under it when the model has a prefix-closed group
            // (eviction keeps the Device's coverage closed from the root),
            // so the walk skips the Host's subtrees; without one, what such
            // a child keeps under it on the Device needs none of these pages
            // either. A page a restore republished below such a child stays
            // and leaves later as a segment of its own.
            struct Frame {
                const BoundaryTree::Node* node;
                const BoundaryTree::Node* next_child;
                bool in_stranded_suffix;
            };
            std::vector<Frame> stack{Frame{.node = top, .next_child = top->first_child, .in_stranded_suffix = false}};
            std::size_t next_climbed = 0;
            bool evictable = true;
            // A child an earlier segment starts at: that segment holds the
            // child's whole Device subtree in this order already, so it
            // goes where the walk reaches it, after the blocks walked so
            // far, and is not walked again (a chain reclaimed page by page
            // from its suffix would otherwise be walked once per page).
            std::vector<std::pair<std::size_t, const BoundaryTree::Node*>> spliced;
            while (!stack.empty()) {
                Frame& frame = stack.back();
                if (const BoundaryTree::Node* child = frame.next_child) {
                    frame.next_child = child->next_sibling;
                    if (segment_at_.contains(child)) {
                        spliced.emplace_back(set.members.size(), child);
                    } else if (child->OnDevice()) {
                        const bool in_stranded_suffix = frame.in_stranded_suffix || frame.node == candidate_node;
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
                } else if (frame.node == candidate_node) {
                    entries = candidate_entries;
                } else {
                    subtree_entries = coordinator_.NodeEntries(pool_, *frame.node);
                    entries = subtree_entries;
                }
                for (const NodeEntry& entry : entries) {
                    if (blocks_alone && frame.node == candidate_node && !is_evictable(entry)) {
                        continue;
                    }
                    if (!is_evictable(entry)) {
                        evictable = false;
                        break;
                    }
                    set.members.push_back(VictimSet::Member{.group_id = entry.group_id,
                                                            .location = entry.location,
                                                            .in_stranded_suffix = frame.in_stranded_suffix,
                                                            .is_request_reclaim = false});
                }
                if (!evictable) {
                    break;
                }
                stack.pop_back();
            }
            if (!evictable) {
                if (!blocks_alone) {
                    passed_over_.push_back(candidate);
                }
                // One unavailable member rejects this whole set; there is
                // no need to inspect the rest of its subtree. Remember
                // only candidates that would expand to the same set. A
                // descendant or sibling can still leave independently.
                const auto remember_same_set = [&](const BoundaryTree::Node& node, std::span<const NodeEntry> entries) {
                    for (const NodeEntry& entry : entries) {
                        if (blocks_alone && &node == candidate_node && !is_evictable(entry)) {
                            continue;
                        }
                        if (!request_reclaim_locations_.contains(entry.location) &&
                            (groups_[entry.group_id].Matcher().IsPrefixClosed() ||
                             coordinator_.GroupIsState(entry.group_id))) {
                            unevictable_locations_.insert(entry.location);
                        }
                    }
                };
                remember_same_set(*candidate_node, candidate_entries);
                for (const auto& [node, entries] : climbed) {
                    remember_same_set(*node, entries);
                }
                return std::nullopt;
            }
            if (!spliced.empty()) {
                VictimSet walked = std::exchange(set, VictimSet{});
                std::size_t next_walked = 0;
                for (const auto& [walked_before, child] : spliced) {
                    set.members.insert(set.members.end(), walked.members.begin() + next_walked,
                                       walked.members.begin() + walked_before);
                    next_walked = walked_before;
                    const auto it = segment_at_.find(child);
                    VictimSet& earlier = victims_[it->second];
                    _assert(!earlier.members.empty(), "a segment set still starts at its top");
                    if (set.members.empty()) {
                        // First in the order: its lists become this set's.
                        set = std::exchange(earlier, VictimSet{});
                        taken_over_set_index = it->second;
                        num_taken_over_members = set.members.size();
                    } else {
                        // Moved in below like any other earlier set it holds.
                        set.members.insert(set.members.end(), earlier.members.begin(), earlier.members.end());
                    }
                    segment_at_.erase(it);
                }
                set.members.insert(set.members.end(), walked.members.begin() + next_walked, walked.members.end());
            }
            // A page leaving block by block may leave blocks in use at its
            // boundary: no later walk takes it for that node's subtree.
            segment_top = blocks_alone ? nullptr : top;
        }
        return ExpandedVictimSet{.set = std::move(set),
                                 .num_taken_over_members = num_taken_over_members,
                                 .taken_over_set_index = taken_over_set_index,
                                 .segment_top = segment_top};
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
                blocks_needed_[i] = geometry_[i].BlocksNeededFor(*demand.table, demand) +
                                    tailBlocksNeeded(i, shape_need_[i]) + zeroStateBlocksNeeded(i);
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
        const CacheCoordinator::BlockRange slots = coordinator_.CopyInSlots(i, *copy_in_point_);
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
        const auto tier_hit = [&](const CacheCoordinator::PrefixProbe::Tier& tier) {
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

    bool couldFitAfterEviction() const {
        std::int64_t parents_needed = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            std::int64_t need = blocks_needed_[i] + 2 * pair_need_[i];
            for (const std::int64_t bucket_need : shape_need_[i]) {
                need += bucket_need;
            }
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
        eviction_plan_.StartCandidates(&protected_locations_, &request_reclaim_locations_,
                                       coordinator_.CheckpointAgingFor(CacheTier::kDevice));
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::vector<CacheBlockLocation> hits = groups_[i].Index().MatchedLocations(
                pool_, (*prefix_.group_keys)[i], /*begin_blocks=*/0, prefix_.device.per_group[i]);
            protected_locations_.insert(hits.begin(), hits.end());
            if (const auto& tail = prefix_.device.tail; tail && !tail->group_keys[i].empty()) {
                const std::vector<CacheBlockLocation> tail_hits = groups_[i].Index().MatchedLocations(
                    pool_, tail->group_keys[i], /*begin_blocks=*/0, tail->per_group[i]);
                protected_locations_.insert(tail_hits.begin(), tail_hits.end());
            }
            if (const auto& state = prefix_.state_at_prompt_end; state && !state->group_keys[i].empty()) {
                const std::vector<CacheBlockLocation> state_hits = groups_[i].Index().MatchedLocations(
                    pool_, state->group_keys[i], /*begin_blocks=*/0, state->per_group[i]);
                protected_locations_.insert(state_hits.begin(), state_hits.end());
            }
        }
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
    bool couldFitAfterReleasingCandidates() const {
        ShadowOccupancy maximum{pool_, groups_};
        for (const VictimCandidate& candidate : request_reclaim_candidates_) {
            maximum.Remove(candidate.group_id, candidate.location);
        }
        if (maximum.Fits(shape_need_, blocks_needed_, pair_need_)) {
            return true;
        }
        std::vector<PrefixCacheIndex::EvictionCandidate> entries;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            PrefixCacheIndex::EvictionCursor cursor;
            const auto& index = groups_[i].Index();
            while (const auto epoch = index.PeekEvictionEpoch(pool_, cursor)) {
                entries.clear();
                index.NextEvictionEpoch(pool_, cursor, *epoch, entries);
                for (const auto& entry : entries) {
                    if (!protected_locations_.contains(entry.location) &&
                        !request_reclaim_locations_.contains(entry.location)) {
                        maximum.Remove(static_cast<std::uint32_t>(i), entry.location);
                    }
                }
                if (maximum.Fits(shape_need_, blocks_needed_, pair_need_)) {
                    return true;
                }
            }
        }
        return false;
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
                const bool first_occurrence = request_reclaim_locations_.insert(location).second;
                if (!first_occurrence) {
                    continue;
                }
                request_reclaim_candidates_.push_back(eviction_plan_.Candidates().MakeCandidate(
                    group_id, location, groups_[i].Index().MetadataFor(pool_, location)));
            }
        }
        eviction_plan_.Candidates().Sort(request_reclaim_candidates_);
    }

    std::optional<VictimCandidate> nextVictimCandidate() { return eviction_plan_.Next(request_reclaim_candidates_); }

    // The coordinator planned for, read through its public views
    // (ForEachBoundaryKey, NodeEntries); groups_, geometry_ and the pools are
    // its own.
    const CacheCoordinator& coordinator_;
    const std::vector<CacheGroup>& groups_;
    std::span<const GroupGeometry> geometry_;
    std::int32_t prefix_granularity_;
    const BlockPool& pool_;
    const BlockPool* host_pool_;
    std::span<const GroupDemand> demands_;
    std::optional<std::int32_t> num_computed_tokens_;
    // The point whose copy-in blocks this admission reserves (Admit).
    std::optional<std::int32_t> copy_in_point_;
    const CacheCoordinator::PrefixProbe& prefix_;
    std::span<const PlannedShape> shapes_;
    // Device-cached blocks a restore claims instead of copying: protected
    // from eviction like the probe's prefix hits.
    std::span<const std::pair<std::uint32_t, CacheBlockLocation>> claimed_;
    const BoundaryTree& boundary_tree_;
    // Whether a victim takes its segment along
    // (CacheCoordinator::write_back_evictions_).
    bool evicts_segments_;
    // Whether, once the segments cannot fund the demand, what they passed over
    // leaves block by block (Admit's evict_blocks_past_segments).
    bool evicts_blocks_past_segments_;
    std::vector<VictimSet>& victims_;
    // The plan's copy-in reservations (AdmissionPlan::copy_ins).
    std::vector<std::vector<CopyInReservation>>& copy_ins_;
    // Lookup nodes live for this plan; returned victim vectors own their
    // storage independently. One arena avoids allocating each hash node from
    // the general heap while trying and rejecting candidate sets.
    std::pmr::monotonic_buffer_resource scratch_;
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
    // Whether the capacity and candidate views Plan() collects before it
    // selects are prepared for a compaction's calls: DenseLayoutParents
    // prepares them once, PlanBlockAlone requires them.
    bool views_prepared_{false};
    std::vector<VictimCandidate> request_reclaim_candidates_;
    std::pmr::unordered_set<CacheBlockLocation, CacheBlockLocationHash> protected_locations_{&scratch_};
    std::pmr::unordered_set<CacheBlockLocation, CacheBlockLocationHash> request_reclaim_locations_{&scratch_};
    // Every block a selected set holds: the set it joined (or last moved
    // into) and its position there. A later set may since have taken that
    // set's lists over in place, positions unchanged: taken_by_ leads there.
    std::pmr::unordered_map<CacheBlockLocation, std::pair<std::size_t, std::size_t>, CacheBlockLocationHash>
        set_member_by_location_{&scratch_};
    // Per set: the later set that took its lists over in place, else itself.
    std::vector<std::size_t> taken_by_;
    // The selected segments by the tree node each starts at: every Device block
    // of that node's subtree, in post-order.
    std::pmr::unordered_map<const BoundaryTree::Node*, std::size_t> segment_at_{&scratch_};
    // Blocks that expand to a set found unable to leave under the current
    // reclaim selection: skipped until a retry invalidates that decision.
    std::pmr::unordered_set<CacheBlockLocation, CacheBlockLocationHash> unevictable_locations_{&scratch_};
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
    std::pmr::unordered_map<const BoundaryTree::Node*, std::int32_t> leaving_checkpoint_boundary_counts_{&scratch_};
};

std::optional<AdmissionPlan> planAdmission(
    const CacheCoordinator& coordinator, const std::vector<CacheGroup>& groups, std::span<const GroupGeometry> geometry,
    std::int32_t prefix_granularity, const BlockPool& pool, const BlockPool* host_pool,
    CacheCoordinator::PrefixProbe&& prefix, std::span<const GroupDemand> demands,
    std::optional<std::int32_t> num_computed_tokens, std::optional<std::int32_t> copy_in_point,
    std::span<const PlannedShape> shapes, std::span<const std::pair<std::uint32_t, CacheBlockLocation>> claimed,
    const BoundaryTree& boundary_tree, bool evicts_segments, bool evicts_blocks_past_segments) {
    ProfileSpan profile_span{"planAdmission"};
    _assert(demands.size() == groups.size(), "demands/groups size mismatch");

    std::vector<VictimSet> victims;
    std::vector<std::vector<CopyInReservation>> copy_ins;
    AdmissionPlanner planner{coordinator,
                             groups,
                             geometry,
                             prefix_granularity,
                             pool,
                             host_pool,
                             demands,
                             num_computed_tokens,
                             copy_in_point,
                             prefix,
                             shapes,
                             claimed,
                             boundary_tree,
                             evicts_segments,
                             evicts_blocks_past_segments,
                             victims,
                             copy_ins};
    if (!planner.Plan()) {
        return std::nullopt;
    }
    return AdmissionPlan{.prefix = std::move(prefix), .victims = std::move(victims), .copy_ins = std::move(copy_ins)};
}

// Acquires the copy-in blocks an admission reserves in group `group_id`
// (copy_ins, slot order) into `reserved`, aligned to the copy-in point's slots
// [first_slot, ...): each in its slot's bucket. A slot `table` holds already
// (a prefix's tail block, a rebuilt slot) has its block's bucket; a slot the
// dense growth fills takes its own block in its copy-in block's bucket,
// acquired in the same call ahead of the copy-in blocks, and returned for
// GroupAllocator::Acquire to place. So every bucket-exact block of the group
// is acquired before the growth's other blocks, the order the planner's fit
// assumes.
struct PairedBlocks {
    std::int32_t first_slot{0};
    std::vector<CacheBlockRef> blocks;
};

PairedBlocks acquireCopyIns(BlockPool& pool, std::uint32_t group_id, const GroupAllocator& allocator,
                            const BlockTable& table, std::span<const CopyInReservation> copy_ins,
                            std::int32_t first_slot, std::vector<CacheBlockRef>& reserved) {
    PairedBlocks paired{.first_slot = table.NumBlocks()};
    std::vector<std::int32_t> buckets;
    for (const CopyInReservation& copy_in : copy_ins) {
        _assert(copy_in.bucket >= 0, "the plan buckets every copy-in block");
        if (copy_in.slot < table.NumBlocks()) {
            const CacheBlockRef& block = table.Blocks()[static_cast<std::size_t>(copy_in.slot)];
            _assert(block && allocator.BucketOf(block->Location()) == copy_in.bucket,
                    "a copy-in block shares its slot's bucket");
            continue;
        }
        if (buckets.empty()) {
            paired.first_slot = copy_in.slot;
        }
        buckets.push_back(copy_in.bucket);
    }
    const std::size_t num_paired = buckets.size();
    for (const CopyInReservation& copy_in : copy_ins) {
        buckets.push_back(copy_in.bucket);
    }
    std::vector<CacheBlockRef> acquired = pool.AcquireBlocksInBuckets(group_id, buckets);
    FatalCheck(acquired.size() == buckets.size(), "admission plan no longer fits the block pool");
    auto next = std::make_move_iterator(acquired.begin());
    paired.blocks.assign(next, next + static_cast<std::ptrdiff_t>(num_paired));
    next += static_cast<std::ptrdiff_t>(num_paired);
    for (const CopyInReservation& copy_in : copy_ins) {
        reserved[static_cast<std::size_t>(copy_in.slot - first_slot)] = *next++;
    }
    return paired;
}

}  // namespace

std::vector<GroupDemand> CacheCoordinator::replayedDemands(const PrefixProbe& prefix,
                                                           std::span<const GroupDemand> demands) const {
    // A replayable group claims no hit pages, so before a hit its table is
    // empty: materialize it as a sparse private suffix from the replay
    // window's first token -- the slots below stay null holes, as
    // absolute-slot tables require -- and the model regenerates the rows.
    // Closed groups keep their dense demand beyond the hit. A demand that
    // already names a sparse suffix is a remote landing (the peer's retained
    // tail) and is left alone: nothing is regenerated here.
    std::vector<GroupDemand> replayed;
    const std::int32_t hit_tokens = std::max(prefix.device.num_common_tokens, prefix.host.num_common_tokens);
    if (replay_window_tokens_ > 0 && hit_tokens > 0) {
        const std::int32_t replay_begin = hit_tokens - ReplayTokens(hit_tokens);
        replayed.assign(demands.begin(), demands.end());
        for (std::size_t i = 0; i < replayed.size(); ++i) {
            const auto* dense = std::get_if<DenseGrowth>(&replayed[i].extent);
            if (!GroupIsReplayable(static_cast<std::int32_t>(i)) || dense == nullptr) {
                continue;
            }
            _assert(replayed[i].table->NumBlocks() == 0, "a replayable group holds no hit pages at admission");
            // The dense growth was relative to an empty table, so the hit plus
            // the growth is the absolute extent.
            replayed[i].extent = SparseSuffix{
                .extent_tokens = hit_tokens + dense->num_tokens,
                .first_block = replay_begin / geometry_[i].BlockGranularity(),
            };
        }
    }
    return replayed;
}

bool CacheCoordinator::writableImageSlot(std::size_t group_index, const ImageTable& shape,
                                         std::int32_t slot_index) const {
    if (GroupIsState(group_index)) {
        return slot_index == 1;  // the live slot; slot 0 is only ever read
    }
    // The tail's blocks: the slots of the page the computed frontier ends
    // inside. The request writes the last of them and later publishes all of
    // them under the page's hash, so none may stay the tail entry's block.
    const std::int32_t block_granularity = geometry_[group_index].BlockGranularity();
    const std::int32_t num_tokens = shape.num_blocks * block_granularity - shape.available_tokens;
    const std::int32_t tail_start = (num_tokens - num_tokens % prefix_granularity_) / block_granularity;
    return slot_index >= tail_start;
}

void CacheCoordinator::ReleaseStateInputs(std::span<BlockTable> tables) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (GroupIsState(i) && tables[i].NumBlocks() == 2) {
            groups_[i].Allocator().ReleaseStateInput(tables[i]);
        }
    }
}

std::optional<CacheCoordinator::AdmissionResult> CacheCoordinator::Admit(
    PrefixProbe&& prefix, std::span<const GroupDemand> demands, const RequestProgress& progress,
    std::span<std::vector<CacheBlockRef>> copy_in_blocks, std::optional<CopyInPoint> copy_in_point,
    std::optional<std::uint64_t> request_access_epoch, bool evict_blocks_past_segments, const std::string& session_id) {
    ProfileSpan profile_span{"CacheCoordinator::Admit"};
    _assert(demands.size() == groups_.size(), "demands/groups size mismatch");
    for (std::size_t i = 0; i < demands.size(); ++i) {
        _assert(demands[i].table != nullptr, "group demand requires a block table");
        _assert(GroupIsState(i) == std::holds_alternative<StateStep>(demands[i].extent),
                "a State group takes a StateStep; a token-spanned group a dense or sparse extent");
    }
    validateProgress(progress);
    validateCopyInBlocks(progress, copy_in_blocks);
    _assert(!copy_in_point || copy_in_point->tokens > progress.num_computed_tokens.value_or(0),
            "a copy-in point lies past the computed frontier");

    const std::vector<GroupDemand> replayed = replayedDemands(prefix, demands);
    if (!replayed.empty()) {
        demands = replayed;
    }

    // The publication this admission carries needs no block beyond the plan's:
    // an owner that keeps writing past a tail copies into the blocks it holds
    // for it (copy_in_blocks).
    std::optional<AdmissionPlan> candidate =
        planAdmission(*this, groups_, geometry_, prefix_granularity_, pool_, host_pool_, std::move(prefix), demands,
                      progress.num_computed_tokens,
                      copy_in_point ? std::optional<std::int32_t>{copy_in_point->tokens} : std::nullopt, /*shapes=*/{},
                      /*claimed=*/{}, boundary_tree_, write_back_evictions_, evict_blocks_past_segments);
    if (!candidate) {
        return std::nullopt;
    }
    AdmissionPlan plan = std::move(*candidate);

    if (request_access_epoch.has_value()) {
        _assert(*request_access_epoch > 0 && *request_access_epoch <= next_access_epoch_,
                "request access epoch was not issued by this coordinator");
    }
    const std::uint64_t access_epoch = request_access_epoch.has_value() ? *request_access_epoch : ++next_access_epoch_;
    // Registered before anything is acquired or published under the epoch.
    SessionAccess session_access;
    if (!request_access_epoch && !session_id.empty()) {
        session_access = SessionAccess{*this, access_epoch, sessionTag(session_id)};
    }
    // The boundaries inside a block, per tier: the hitter's table ends in a
    // private copy of the Device tail's partial block, or in a fresh page the
    // Host tail is loaded into.
    const std::int32_t device_tail_tokens = plan.prefix.device.tail ? plan.prefix.device.tail->tail_tokens : 0;
    const std::int32_t host_tail_tokens = plan.prefix.host.tail ? plan.prefix.host.tail->tail_tokens : 0;
    const std::int32_t admitted_tokens =
        std::max(plan.prefix.device.num_common_tokens, plan.prefix.host.num_common_tokens);
    const std::int32_t kv_tokens = std::max(plan.prefix.device.kv_tokens, plan.prefix.host.kv_tokens);
    const bool state_prefix_cached = plan.prefix.state_at_prompt_end.has_value();
    AcquiredPrefix acquired_prefix = acquirePrefix(std::move(plan.prefix), access_epoch);
    if (!request_access_epoch.has_value()) {
        recordAdmission(access_epoch);
    }
    AdmissionResult result{
        .device_prefix_tokens = acquired_prefix.device.num_common_tokens,
        .host_prefix_tokens = acquired_prefix.host.num_common_tokens,
        .junction_tokens = kv_tokens > admitted_tokens ? kv_tokens : 0,
        .state_prefix_cached = state_prefix_cached,
        .access_epoch = access_epoch,
        .new_page_ids = std::vector<std::vector<std::int32_t>>(groups_.size()),
        .pages_to_copy = std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>>(groups_.size()),
        .state_checkpoint_blocks = std::vector<CacheBlockRef>(groups_.size()),
        .copy_in_blocks = std::vector<std::vector<CacheBlockRef>>(copy_in_point ? groups_.size() : 0),
        .session_access = std::move(session_access),
    };
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (GroupIsState(i)) {
            continue;  // a State hit is installed below, as slot 0
        }
        if (acquired_prefix.device.num_common_tokens > 0) {
            groups_[i].Allocator().ClaimHitBlocks(*demands[i].table, std::move(acquired_prefix.device.per_group[i]));
        }
    }
    // A State group's first admission shapes its two slots: slot 0 the one
    // checkpoint the probe matched on the Device (a Host checkpoint lands by
    // load below; the D's prompt-end checkpoint is claimed here), else null
    // until the step below installs the zero state; slot 1 is acquired with
    // the step below. The shaping is the first rotation: a shaped-now table
    // rotates nothing further.
    std::vector<bool> shaped_now(groups_.size(), false);
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!GroupIsState(i) || demands[i].table->NumBlocks() != 0) {
            continue;
        }
        const StateStep::Kind kind = std::get<StateStep>(demands[i].extent).kind;
        if (kind != StateStep::Kind::kRotate && kind != StateStep::Kind::kLand) {
            continue;
        }
        shaped_now[i] = true;
        CacheBlockRef input;
        for (CacheBlockRef& block : acquired_prefix.device.per_group[i].blocks) {
            if (block) {
                _assert(!input, "a State group matches one checkpoint");
                input = std::move(block);
            }
        }
        if (!acquired_prefix.state_at_prompt_end.empty() && acquired_prefix.state_at_prompt_end[i]) {
            _assert(!input, "a State group matches one checkpoint");
            input = std::move(acquired_prefix.state_at_prompt_end[i]);
        }
        if (const std::optional<CacheKey> key = groups_[i].Index().KeyOf(pool_, input);
            key && boundary_tree_.Tails().Contains(key->content_hash)) {
            result.resumed_tail_hash = key->content_hash;
        }
        groups_[i].Allocator().ShapeStateTable(*demands[i].table, std::move(input));
    }
    // A remote landing fills whole every block that ends at or below the
    // prompt end (slot b holds tokens [b*g, (b+1)*g)): a copy destination,
    // not a page to zero.
    const auto landed_whole = [&](std::size_t i, std::int32_t slot) {
        const std::int32_t landed_tokens = demands[i].landed_tokens;
        return landed_tokens > 0 &&
               static_cast<std::int64_t>(slot + 1) * geometry_[i].BlockGranularity() <= landed_tokens;
    };
    // Gives a hitter's table a private block in `slot`: a copy of the shared
    // block there (copy-on-write), or a fresh one where the slot was a hole,
    // zeroed unless the peer's landing fills it.
    const auto privatize_slot = [&](std::size_t i, BlockTable& table, std::int32_t slot,
                                    std::int32_t available_tokens) {
        GroupAllocator& allocator = groups_[i].Allocator();
        const CacheBlockRef replaced =
            allocator.SwapInFreshBlock(pool_, table, slot, available_tokens, /*destination=*/std::nullopt);
        const std::int32_t destination = allocator.ResolveCacheBlockId(*table.Blocks()[static_cast<std::size_t>(slot)]);
        if (replaced) {
            result.pages_to_copy[i].emplace_back(allocator.ResolveCacheBlockId(*replaced), destination);
        } else if (!landed_whole(i, slot)) {
            result.new_page_ids[i].push_back(destination);
        }
    };
    std::vector<
        std::tuple<std::uint32_t, CacheBlockLocation, DeviceEvictionCounts DeviceEvictionStats::*, EvictedSegment*>>
        prospective_victims;
    // Record cached victims for write-back even when this request releases
    // them: a published input can still resume another request. Uncached
    // slots and explicitly superseded inputs contribute no store records.
    // A reclaimable table block may still be pinned by both the request and
    // cache here. Evict what is already free, then slide the request tables and
    // retry the blocks whose request reference has just been released.
    {
        ProfileSpan evicting{"CacheCoordinator::Admit (evict)"};
        for (const VictimSet& set : plan.victims) {
            EvictedSegment* segment = write_back_evictions_ ? &evicted_segments_.emplace_back() : nullptr;
            for (const auto& [group_id, location, in_stranded_suffix, is_request_reclaim] : set.members) {
                DeviceEvictionCounts DeviceEvictionStats::*reason =
                    is_request_reclaim   ? &DeviceEvictionStats::request_reclaim
                    : in_stranded_suffix ? &DeviceEvictionStats::stranded_suffix
                                         : &DeviceEvictionStats::admission;
                if (!evictCachedBlock(group_id, location, reason, segment)) {
                    prospective_victims.emplace_back(group_id, location, reason, segment);
                }
            }
        }
    }
    // Held while the groups register under the publication's nodes.
    BoundaryTree::Hold publication;
    if (progress.completed_pages) {
        publication = boundary_tree_.LinkPublication(*progress.completed_pages);
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        BlockTable& table = *demands[i].table;
        if (progress.completed_pages) {
            cacheDeviceCompletedBlocksForGroup(i, table, *progress.completed_pages, access_epoch);
        }
        if (GroupIsState(i)) {
            // The step's release: slot 0 leaves the table (the index keeps it
            // when published, else it returns to the pool and may fund this
            // very admission); a rotation moves the block the last forward
            // wrote -- published just above when the point was a checkpoint
            // -- into slot 0 and leaves slot 1 for the acquisition below.
            if (table.NumBlocks() == 0) {
                continue;
            }
            switch (std::get<StateStep>(demands[i].extent).kind) {
                case StateStep::Kind::kRotate:
                    if (!shaped_now[i]) {
                        groups_[i].Allocator().RotateState(table);
                    }
                    break;
                case StateStep::Kind::kRelease:
                    groups_[i].Allocator().ReleaseStateInput(table);
                    break;
                case StateStep::Kind::kKeep:
                case StateStep::Kind::kLand:
                    break;
            }
            continue;
        }
        if (progress.num_computed_tokens) {
            groups_[i].Allocator().ReclaimExpired(
                pool_, table, geometry_[i].ExpiredBlocksAt(groups_[i].Spec(), *progress.num_computed_tokens));
        }
    }
    if (progress.completed_pages) {
        const CompletedPages& completed = *progress.completed_pages;
        // A model without a prefix-closed group may register nothing under
        // the deepest boundary linked (a chunk end of State groups alone): it
        // leaves now.
        publication.reset();
        recordCheckpoints(completed, access_epoch);
        recordPublishedTail(completed);
    }
    for (const auto& [group_id, location, reason, segment] : prospective_victims) {
        if (!evictCachedBlock(group_id, location, reason, segment)) {
            FatalCheck(!pool_.IsOccupied(location), "admission victim changed before acquisition");
        }
    }
    // A set whose every block was dropped records nothing: this admission's
    // empty records go, and an earlier call left none.
    std::erase_if(evicted_segments_, [](const EvictedSegment& segment) { return segment.members.empty(); });

    for (std::size_t i = 0; i < groups_.size(); ++i) {
        // A group's bucket-exact blocks (a hitter's tail copies, the Host
        // extension, the copy-in blocks reserved below) are acquired before
        // its dense growth, which is what the planner's fit assumes.
        const GroupDemand& demand = demands[i];
        BlockTable& table = *demand.table;
        // An owner that keeps writing past its published prompt end moves off
        // every block it just registered into the copy-in block it holds for
        // the slot, in that block's bucket: a published block is never
        // written again and never carries a second key. A slot whose key
        // another request published first was not registered and stays the
        // owner's to write; its copy-in block goes back unused. (A State
        // group's published slot 1 moves to slot 0 by rotation instead.)
        if (!copy_in_blocks.empty()) {
            const BlockRange slots = CopyInSlots(i, *progress.num_computed_tokens);
            _assert(slots.past_end <= table.NumBlocks(), "the table holds every copy-in slot of its frontier");
            std::vector<CacheBlockRef>& held = copy_in_blocks[i];
            GroupAllocator& allocator = groups_[i].Allocator();
            for (std::int32_t slot = slots.first; slot < slots.past_end; ++slot) {
                if (!groups_[i].Index().Contains(table.Blocks()[static_cast<std::size_t>(slot)])) {
                    continue;
                }
                const CacheBlockRef published =
                    allocator.SwapInBlock(table, slot, std::move(held[static_cast<std::size_t>(slot - slots.first)]));
                result.pages_to_copy[i].emplace_back(
                    allocator.ResolveCacheBlockId(*published),
                    allocator.ResolveCacheBlockId(*table.Blocks()[static_cast<std::size_t>(slot)]));
            }
            held.clear();  // the unused ones go back
        }
        // A first admission's table ends in the tail's blocks: shared full
        // pages, then this request's own copies.
        if (!GroupIsState(i) && device_tail_tokens > 0 && table.NumBlocks() > 0) {
            const TailSlots slots =
                tailSlots(geometry_[i], prefix_granularity_, acquired_prefix.device.num_common_tokens);
            _assert(slots.past_end == table.NumBlocks(), "the Device tail's blocks end the claimed table");
            // A sliding group's tail blocks below its window stay holes.
            const std::int32_t first =
                slots.first + FirstTailBlockNeeded(static_cast<std::int32_t>(i), device_tail_tokens);
            for (std::int32_t slot = first; slot < slots.past_end; ++slot) {
                privatize_slot(i, table, slot, slots.available_tokens);
            }
        }
        if (GroupIsState(i)) {
            const StateStep& step = std::get<StateStep>(demand.extent);
            GroupAllocator& allocator = groups_[i].Allocator();
            if (table.NumBlocks() == 0) {
                continue;  // never shaped: a step other than a rotation leaves it so
            }
            // A Host checkpoint is loaded into a fresh slot 0.
            if (!acquired_prefix.host.per_group.empty()) {
                for (CacheBlockRef& host_block : acquired_prefix.host.per_group[i].blocks) {
                    if (!host_block) {
                        continue;
                    }
                    _assert(!table.Blocks()[0], "a State group matches one checkpoint");
                    std::vector<CacheBlockRef> destination = pool_.AcquireBlocksInBuckets(
                        groups_[i].Id(), std::array<std::int32_t, 1>{allocator.BucketOf(host_block->Location())});
                    FatalCheck(destination.size() == 1, "admission plan no longer fits the block pool");
                    allocator.InstallStateInput(table, std::move(destination.front()));
                    result.load_pairs.push_back(BlockTransfer{.group_id = groups_[i].Id(),
                                                              .source = std::move(host_block),
                                                              .destination = table.Blocks()[0]});
                }
            }
            // A prompt with no checkpoint to read starts from the zero state:
            // a fresh zeroed block in slot 0, so the request holds its two
            // blocks from its first admission on.
            if (step.kind == StateStep::Kind::kRotate && shaped_now[i] && !table.Blocks()[0]) {
                FatalCheck(allocator.AcquireStateInput(pool_, table), "admission plan no longer fits the block pool");
                result.new_page_ids[i].push_back(allocator.ResolveCacheBlockId(*table.Blocks()[0]));
            }
            if (step.kind == StateStep::Kind::kRotate || step.kind == StateStep::Kind::kLand) {
                FatalCheck(allocator.AcquireStateLive(pool_, table), "admission plan no longer fits the block pool");
                const std::int32_t live = allocator.ResolveCacheBlockId(*table.Blocks()[1]);
                if (step.kind == StateStep::Kind::kLand && state_prefix_cached) {
                    // The peer omits this state transfer. A bootstrap token
                    // can finish the request before any decode writes live.
                    _assert(static_cast<bool>(table.Blocks()[0]), "an exact state hit must have an input checkpoint");
                    result.pages_to_copy[i].emplace_back(allocator.ResolveCacheBlockId(*table.Blocks()[0]), live);
                } else if (step.kind != StateStep::Kind::kLand) {
                    // A landing's live slot is filled whole by the peer's
                    // transfer, like a copy destination: nothing to zero.
                    result.new_page_ids[i].push_back(live);
                }
            }
            if (step.checkpoint) {
                _assert(step.kind == StateStep::Kind::kRotate, "an in-forward checkpoint rides a prefill chunk");
                CacheBlockRef checkpoint = allocator.AcquireStateCheckpoint(pool_, table);
                FatalCheck(static_cast<bool>(checkpoint), "admission plan no longer fits the block pool");
                result.new_page_ids[i].push_back(allocator.ResolveCacheBlockId(*checkpoint));
                result.state_checkpoint_blocks[i] = std::move(checkpoint);
            }
            continue;
        }
        if (!acquired_prefix.host.per_group.empty() && !acquired_prefix.host.per_group[i].blocks.empty()) {
            const std::int32_t available = host_tail_tokens > 0 ? tailSlots(geometry_[i], prefix_granularity_,
                                                                            acquired_prefix.host.num_common_tokens)
                                                                      .available_tokens
                                                                : 0;
            // A Host tail's blocks land on fresh destinations (one load row
            // each); a sliding group's tail blocks below its window stay holes.
            groups_[i].Allocator().AppendHostExtension(
                pool_, table, std::move(acquired_prefix.host.per_group[i].blocks), result.load_pairs, available);
        }
        // The copy-in blocks this admission reserves, each with the block it
        // serves (acquireCopyIns), the paired ones placed by the growth.
        PairedBlocks paired{.first_slot = demand.table->NumBlocks()};
        if (copy_in_point) {
            const BlockRange slots = CopyInSlots(i, copy_in_point->tokens);
            result.copy_in_blocks[i].resize(static_cast<std::size_t>(slots.past_end - slots.first));
            paired = acquireCopyIns(pool_, groups_[i].Id(), groups_[i].Allocator(), *demand.table, plan.copy_ins[i],
                                    slots.first, result.copy_in_blocks[i]);
        }
        const std::int32_t first_new_block = demand.table->NumBlocks();
        const bool acquired =
            groups_[i].Allocator().Acquire(pool_, *demand.table, geometry_[i].PlanAcquire(*demand.table, demand),
                                           paired.first_slot, std::move(paired.blocks));
        FatalCheck(acquired, "admission plan no longer fits the block pool");
        // Only a dense table's slots are block-aligned to the prompt the way
        // landed_whole assumes.
        _assert(demand.landed_tokens == 0 || std::holds_alternative<DenseGrowth>(demand.extent),
                "a remote landing's extent is a dense growth");
        const std::span<const CacheBlockRef> blocks = demand.table->Blocks();
        for (std::int32_t block = first_new_block; block < demand.table->NumBlocks(); ++block) {
            if (!blocks[static_cast<std::size_t>(block)] || landed_whole(i, block)) {
                continue;
            }
            result.new_page_ids[i].push_back(
                groups_[i].Allocator().ResolveCacheBlockId(*blocks[static_cast<std::size_t>(block)]));
        }
    }
    return result;
}

std::optional<CacheCoordinator::AdmissionResult> CacheCoordinator::Restore(const RetractionImage& image,
                                                                           std::span<const GroupDemand> demands,
                                                                           std::optional<CopyInPoint> copy_in_point,
                                                                           std::uint64_t request_access_epoch,
                                                                           bool drop_claims_if_unfit) {
    _assert(demands.size() == groups_.size(), "demands/groups size mismatch");
    _assert(image.tables.size() == groups_.size(), "image/groups size mismatch");
    _assert(request_access_epoch > 0 && request_access_epoch <= next_access_epoch_,
            "request access epoch was not issued by this coordinator");
    // The rebuilt slots among the copy-in slots, each in its imaged block's
    // bucket: the partial page at the image's frontier, which the restore
    // always copies (never claims).
    const auto rebuilt_copy_ins = [&](std::size_t i) {
        std::vector<CopyInReservation> rebuilt;
        if (!copy_in_point) {
            return rebuilt;
        }
        const BlockRange range = CopyInSlots(i, copy_in_point->tokens);
        const ImageTable& shape = image.tables[i];
        for (std::int32_t slot = range.first; slot < std::min(range.past_end, shape.num_blocks); ++slot) {
            const auto it = std::ranges::find(shape.slots, slot, &ImageSlot::slot_index);
            _assert(it != shape.slots.end() && writableImageSlot(i, shape, slot),
                    "a rebuilt copy-in slot is a slot of the partial page at the image's frontier");
            rebuilt.push_back(
                CopyInReservation{.slot = slot, .bucket = groups_[i].Allocator().BucketOf(it->block->Location())});
        }
        return rebuilt;
    };
    // A Host L2 slot whose key still has a Device-cached canonical block is
    // claimed instead of copied: the same bytes by construction (the victim's
    // own block, cache-only since it was freed, the canonical block its table
    // already pointed at, or any block published under the key since, wherever
    // it sits). Those blocks are protected from the planner's
    // eviction like prefix hits; every other slot is a bucket-constrained
    // fresh block.
    // A published slot the resumed request will never write again is claimed
    // from the Device index when its key is still cached there; the slot it
    // keeps writing -- the partial page at its computed frontier -- always
    // gets a fresh copy, so the published block stays immutable.
    const auto claimable = [&](std::size_t i, const ImageTable& shape, const ImageSlot& slot) {
        return slot.InHostCache() && !writableImageSlot(i, shape, slot.slot_index);
    };
    // When a plan that claims fails, plan once more claiming nothing; every
    // slot is then a fresh copy, an L2 slot republished at its ACK
    // (docs/design/cache-concepts.md, AdmissionPlanner). Under segment
    // eviction the claims can pin the only evictable cache; with
    // drop_claims_if_unfit (no other request holds pages), claimed blocks one
    // per parent beside cached pages can bind more parents than a dense
    // layout of the image needs, and no release will come.
    bool claim_cached_slots = true;
    std::optional<AdmissionPlan> candidate;
    for (const bool claim : {true, false}) {
        claim_cached_slots = claim;
        std::vector<PlannedShape> shapes(groups_.size());
        std::vector<std::pair<std::uint32_t, CacheBlockLocation>> protected_locations;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            _assert(demands[i].table != nullptr && demands[i].table->NumBlocks() == 0,
                    "a restore rebuilds into an empty table");
            const GroupAllocator& allocator = groups_[i].Allocator();
            const ImageTable& shape = image.tables[i];
            shapes[i].blocks_by_bucket.assign(static_cast<std::size_t>(allocator.ShardCount()), 0);
            shapes[i].num_blocks = shape.num_blocks;
            shapes[i].available_tokens = shape.available_tokens;
            shapes[i].rebuilt_copy_ins = rebuilt_copy_ins(i);
            for (const ImageSlot& slot : shape.slots) {
                _assert(static_cast<bool>(slot.block), "image slot lost its block");
                const CacheBlockRef canonical =
                    claim && claimable(i, shape, slot) ? groups_[i].Index().Find(pool_, slot.key) : CacheBlockRef{};
                if (canonical) {
                    protected_locations.emplace_back(groups_[i].Id(), canonical->Location());
                    continue;
                }
                ++shapes[i].blocks_by_bucket[static_cast<std::size_t>(allocator.BucketOf(slot.block->Location()))];
            }
        }
        candidate = planAdmission(
            *this, groups_, geometry_, prefix_granularity_, pool_, host_pool_, ProbePrefix({}), demands,
            /*num_computed_tokens=*/std::nullopt,
            copy_in_point ? std::optional<std::int32_t>{copy_in_point->tokens} : std::nullopt, shapes,
            protected_locations, boundary_tree_, write_back_evictions_, /*evicts_blocks_past_segments=*/false);
        if (candidate || protected_locations.empty() || !(write_back_evictions_ || drop_claims_if_unfit)) {
            break;
        }
    }
    if (!candidate) {
        return std::nullopt;
    }
    // Claim before evicting, as Admit does: a claimed entry is pinned, so a
    // victim's eviction cascade (a tail's companions, a page's state
    // checkpoints) cannot take a block the image is about to resume on.
    std::vector<std::vector<CacheBlockRef>> claimed(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const ImageTable& shape = image.tables[i];
        claimed[i].resize(shape.slots.size());
        for (std::size_t j = 0; j < shape.slots.size(); ++j) {
            const ImageSlot& slot = shape.slots[j];
            if (!claim_cached_slots || !claimable(i, shape, slot) || !groups_[i].Index().Find(pool_, slot.key)) {
                continue;
            }
            // The planner protected this entry; claim it like a prefix hit.
            const std::array<CacheKey, 1> keys{slot.key};
            PrefixMatch hit =
                groups_[i].Index().AcquireMatched(pool_, keys, /*begin_blocks=*/0, GroupPrefixProbe{.hits = {1}},
                                                  request_access_epoch, SessionForAccess(request_access_epoch));
            claimed[i][j] = std::move(hit.blocks.front());
        }
    }
    // Nothing is published here and the tables start empty, so no set is the
    // request's own reclaim: every planned victim is a cache-only block the
    // planner saw as evictable (or a tail companion or same-hash State
    // checkpoint an earlier victim's cascade already took). Under the
    // write-back policy each set is recorded whole as one segment for this
    // plan's drain, as Admit does.
    for (const VictimSet& set : candidate->victims) {
        EvictedSegment* segment = write_back_evictions_ ? &evicted_segments_.emplace_back() : nullptr;
        for (const VictimSet::Member& member : set.members) {
            _assert(!member.is_request_reclaim, "a restore rebuilds empty tables and reclaims nothing of its own");
            if (!evictCachedBlock(member.group_id, member.location, &DeviceEvictionStats::restore, segment)) {
                FatalCheck(!pool_.IsOccupied(member.location), "restore victim changed before acquisition");
            }
        }
    }
    // A set whose every block was dropped records nothing: this restore's
    // empty records go, and an earlier call left none.
    std::erase_if(evicted_segments_, [](const EvictedSegment& segment) { return segment.members.empty(); });

    AdmissionResult result{
        .access_epoch = request_access_epoch,
        .new_page_ids = std::vector<std::vector<std::int32_t>>(groups_.size()),
        .pages_to_copy = std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>>(groups_.size()),
        .copy_in_blocks = std::vector<std::vector<CacheBlockRef>>(copy_in_point ? groups_.size() : 0),
    };
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        GroupAllocator& allocator = groups_[i].Allocator();
        BlockTable& table = *demands[i].table;
        const ImageTable& shape = image.tables[i];
        FatalCheck(allocator.AcquireShape(pool_, table, shape, std::move(claimed[i])),
                   "restore plan no longer fits the block pool");
        for (const ImageSlot& slot : shape.slots) {
            const CacheBlockRef& destination = table.Blocks()[static_cast<std::size_t>(slot.slot_index)];
            if (slot.InHostCache() && groups_[i].Index().Contains(destination)) {
                continue;  // claimed: already the published Device block, nothing to copy
            }
            // The writable slot's row is unkeyed: its destination is the
            // request's private copy and must not be republished at the ACK.
            const bool republish = claimable(i, shape, slot);
            BlockTransfer transfer{
                .group_id = groups_[i].Id(),
                .source = slot.block,
                .destination = destination,
                .key = republish ? slot.key : CacheKey{},
                .logical_block_index = slot.logical_block_index,
                .boundary_kind = slot.boundary_kind,
            };
            // A published slot comes back from Host L2 (and is re-published at
            // the ACK); a private slot from the snapshot pool. Neither is
            // listed in new_page_ids: the copy fills the whole block, as a
            // Host hit's load-back does.
            if (slot.InHostCache()) {
                result.load_pairs.push_back(std::move(transfer));
            } else {
                result.snapshot_pairs.push_back(std::move(transfer));
            }
        }
        if (GroupIsState(i)) {
            continue;  // the imaged two slots are the whole table
        }
        // The copy-in blocks, each with the rebuilt or grown block it serves
        // (acquireCopyIns), before the reserve's other blocks.
        PairedBlocks paired{.first_slot = table.NumBlocks()};
        if (copy_in_point) {
            const BlockRange slots = CopyInSlots(i, copy_in_point->tokens);
            result.copy_in_blocks[i].resize(static_cast<std::size_t>(slots.past_end - slots.first));
            paired = acquireCopyIns(pool_, groups_[i].Id(), allocator, table, candidate->copy_ins[i], slots.first,
                                    result.copy_in_blocks[i]);
        }
        // Only the reserve appended beyond the imaged shape is fresh, unfilled
        // Device memory for the plan to zero.
        const std::int32_t first_new_block = table.NumBlocks();
        FatalCheck(allocator.Acquire(pool_, table, geometry_[i].PlanAcquire(table, demands[i]), paired.first_slot,
                                     std::move(paired.blocks)),
                   "restore plan no longer fits the block pool");
        const std::span<const CacheBlockRef> blocks = table.Blocks();
        for (std::int32_t block = first_new_block; block < table.NumBlocks(); ++block) {
            result.new_page_ids[i].push_back(allocator.ResolveCacheBlockId(*blocks[static_cast<std::size_t>(block)]));
        }
    }
    return result;
}

std::optional<CacheCoordinator::AdmissionResult> CacheCoordinator::CompactLoneRequest(
    PrefixProbe&& prefix, std::span<const GroupDemand> demands, const RequestProgress& progress,
    std::span<std::vector<CacheBlockRef>> copy_in_blocks, std::optional<CopyInPoint> copy_in_point) {
    _assert(demands.size() == groups_.size(), "demands/groups size mismatch");
    validateProgress(progress);
    _assert(copy_in_blocks.empty() || copy_in_blocks.size() == groups_.size(),
            "copy-in blocks are held per cache group");
    _assert(copy_in_blocks.empty() || copy_in_point || progress.num_computed_tokens,
            "held copy-in blocks serve the point an admission reserves for or the step publishes");
    const std::vector<GroupDemand> replayed = replayedDemands(prefix, demands);
    if (!replayed.empty()) {
        demands = replayed;
    }
    const std::optional<std::int32_t> reserve_point =
        copy_in_point ? std::optional<std::int32_t>{copy_in_point->tokens} : std::nullopt;
    // Whether the retry admits now: Admit's plan.
    const auto admission_fits = [&] {
        return planAdmission(*this, groups_, geometry_, prefix_granularity_, pool_, host_pool_, PrefixProbe{prefix},
                             demands, progress.num_computed_tokens, reserve_point, /*shapes=*/{}, /*claimed=*/{},
                             boundary_tree_, write_back_evictions_, /*evicts_blocks_past_segments=*/true)
            .has_value();
    };

    // The blocks of the request in each packed, token-spanned group -- the
    // table's, and the copy-in blocks it holds beside them -- and at a first
    // admission the Device pages and tail blocks its probe hits, by what may
    // happen to them. An unwritten block moves anywhere and is zeroed, unless
    // a copy-in block is held for its slot: the two share a bucket (the
    // copy-in's contract), so each re-places within it. A written block moves
    // by a same-bucket copy, a published one with its entry; a hit page or
    // tail block moves like a published block. Fixed: a page this admission
    // publishes (the runtime stores it before the plan's copies land), a
    // block another holder pins (a store ticket, a retained checkpoint,
    // another table), a hit another request pins. Slots the retry reclaims
    // first (a released State input, expired window blocks) are not the
    // request's footprint and are left out.
    struct Member {
        enum class Kind : std::uint8_t { kUnwritten, kCopyIn, kWritten, kPublished, kHit, kTail, kFixed };
        Kind kind;
        // Whether the block keeps its bucket: every one but a free unwritten
        // block.
        bool in_bucket;
        CacheBlockLocation location;
        std::int32_t slot;  // the table slot (a copy-in block: the slot it serves), -1 for a probe block
        std::size_t hit;    // the probe block's pin in hit_pins (kHit, kTail; 0 otherwise)
    };
    // Every Device block the probe hits is held from here to the return --
    // the pages and tail blocks of every group, compacted or not, and on the
    // D role each State group's checkpoint at the prompt end: clearing a
    // destination slot evicts what its entry takes along (a tail's
    // companions, a page's checkpoints), and an unheld hit taken so would be
    // gone when the retry acquires it. The slot itself may be a block the
    // probe leaves a hole (the D role's partial tail block, the prompt's last
    // page past its bound).
    std::vector<CacheBlockRef> hit_pins;
    std::vector<std::vector<Member>> members(groups_.size());
    std::vector<std::vector<std::int64_t>> bucket_pinned(groups_.size());
    std::vector<std::int64_t> free_movers(groups_.size(), 0);
    // Per group, the slots the held copy-in blocks serve.
    std::vector<BlockRange> copy_in_slots(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const GroupAllocator& allocator = groups_[i].Allocator();
        const PrefixCacheIndex& index = groups_[i].Index();
        const BlockTable& table = *demands[i].table;
        bucket_pinned[i].assign(static_cast<std::size_t>(allocator.ShardCount()), 0);
        const bool compacts = geometry_[i].HasTokenSpan() && pool_.Packing(groups_[i].Id()) > 1;
        // The copy-in blocks the request holds, each with the block it serves
        // (Admit reserves them so): that block keeps its bucket as well.
        std::vector<bool> copy_in_beside(static_cast<std::size_t>(table.NumBlocks()), false);
        if (!copy_in_blocks.empty()) {
            copy_in_slots[i] = CopyInSlots(i, copy_in_point ? copy_in_point->tokens : *progress.num_computed_tokens);
            _assert(copy_in_blocks[i].size() ==
                        static_cast<std::size_t>(copy_in_slots[i].past_end - copy_in_slots[i].first),
                    "the request holds one copy-in block per copy-in slot");
            for (std::size_t j = 0; j < copy_in_blocks[i].size(); ++j) {
                const CacheBlockRef& block = copy_in_blocks[i][j];
                if (!block) {
                    continue;
                }
                const std::int32_t slot = copy_in_slots[i].first + static_cast<std::int32_t>(j);
                _assert(slot < table.NumBlocks(), "a copy-in block is held with the block it serves");
                copy_in_beside[static_cast<std::size_t>(slot)] = true;
                ++bucket_pinned[i][static_cast<std::size_t>(allocator.BucketOf(block->Location()))];
                if (compacts) {
                    members[i].push_back(Member{.kind = Member::Kind::kCopyIn,
                                                .in_bucket = true,
                                                .location = block->Location(),
                                                .slot = slot,
                                                .hit = 0});
                }
            }
        }
        std::int32_t first_slot = table.ReclaimedPrefixBlocks();
        if (GroupIsState(i)) {
            first_slot = std::get<StateStep>(demands[i].extent).kind == StateStep::Kind::kKeep ? 0 : 1;
        } else if (progress.num_computed_tokens) {
            first_slot =
                std::max(first_slot, geometry_[i].ExpiredBlocksAt(groups_[i].Spec(), *progress.num_computed_tokens));
        }
        std::vector<bool> publishing(static_cast<std::size_t>(table.NumBlocks()), false);
        if (compacts && progress.completed_pages) {
            for (const PublishedRange& range : completedBlockRanges(i, *progress.completed_pages)) {
                for (std::size_t j = 0; j < range.keys.size(); ++j) {
                    const std::int32_t slot = range.first_cache_block + static_cast<std::int32_t>(j);
                    if (slot < table.NumBlocks()) {
                        publishing[static_cast<std::size_t>(slot)] = true;
                    }
                }
            }
        }
        for (std::int32_t slot = first_slot; slot < table.NumBlocks(); ++slot) {
            const CacheBlockRef& block = table.Blocks()[static_cast<std::size_t>(slot)];
            if (!block) {
                continue;
            }
            const bool cached = index.Contains(block);
            Member::Kind kind = Member::Kind::kFixed;
            if (compacts && !publishing[static_cast<std::size_t>(slot)] &&
                (cached ? block.use_count() == 2 : block.unique())) {
                // Rows past the computed frontier are unwritten (a draft row a
                // verify step rewrites counts as unwritten too).
                kind = cached ? Member::Kind::kPublished
                       : progress.num_computed_tokens &&
                               static_cast<std::int64_t>(slot) * geometry_[i].BlockGranularity() >=
                                   *progress.num_computed_tokens
                           ? Member::Kind::kUnwritten
                           : Member::Kind::kWritten;
            }
            const bool in_bucket = kind != Member::Kind::kUnwritten || copy_in_beside[static_cast<std::size_t>(slot)];
            if (in_bucket) {
                ++bucket_pinned[i][static_cast<std::size_t>(allocator.BucketOf(block->Location()))];
            } else {
                ++free_movers[i];
            }
            if (compacts) {
                members[i].push_back(Member{
                    .kind = kind, .in_bucket = in_bucket, .location = block->Location(), .slot = slot, .hit = 0});
            }
        }
        if (prefix.group_keys == nullptr || i >= prefix.device.per_group.size()) {
            continue;
        }
        // The probe's hit pages, then its Device tail's blocks: the retry
        // claims the pages, and copies a tail block into the hitter's own
        // fresh block from where it was before this plan moved it
        // (Scheduler::recordAdmittedPages).
        std::vector<CacheBlockLocation> probe_blocks =
            index.MatchedLocations(pool_, (*prefix.group_keys)[i], /*begin_blocks=*/0, prefix.device.per_group[i]);
        const std::size_t num_pages = probe_blocks.size();
        if (const auto& tail = prefix.device.tail; tail && !tail->group_keys[i].empty()) {
            const std::vector<CacheBlockLocation> tail_blocks =
                index.MatchedLocations(pool_, tail->group_keys[i], /*begin_blocks=*/0, tail->per_group[i]);
            probe_blocks.insert(probe_blocks.end(), tail_blocks.begin(), tail_blocks.end());
        }
        if (!compacts) {
            // A group the compaction does not gather (a State group among
            // them, with the prompt-end checkpoint the D role claims): its
            // hits are held where they are.
            if (const auto& state = prefix.state_at_prompt_end; state && !state->group_keys[i].empty()) {
                const std::vector<CacheBlockLocation> state_blocks =
                    index.MatchedLocations(pool_, state->group_keys[i], /*begin_blocks=*/0, state->per_group[i]);
                probe_blocks.insert(probe_blocks.end(), state_blocks.begin(), state_blocks.end());
            }
            for (const CacheBlockLocation location : probe_blocks) {
                hit_pins.push_back(index.Find(pool_, *index.KeyOf(pool_, location)));
            }
            continue;
        }
        for (std::size_t b = 0; b < probe_blocks.size(); ++b) {
            const CacheBlockLocation location = probe_blocks[b];
            const CacheKey key = *index.KeyOf(pool_, location);
            Member::Kind kind = Member::Kind::kFixed;
            std::size_t hit = 0;
            if (!index.Inspect(pool_, key)->pinned) {
                hit = hit_pins.size();
                hit_pins.push_back(index.Find(pool_, key));
                kind = b < num_pages ? Member::Kind::kHit : Member::Kind::kTail;
            }
            ++bucket_pinned[i][static_cast<std::size_t>(allocator.BucketOf(location))];
            members[i].push_back(Member{.kind = kind, .in_bucket = true, .location = location, .slot = -1, .hit = hit});
        }
    }

    std::vector<VictimSet> victims;
    std::vector<std::vector<CopyInReservation>> copy_ins;
    AdmissionPlanner planner{*this,
                             groups_,
                             geometry_,
                             prefix_granularity_,
                             pool_,
                             host_pool_,
                             demands,
                             progress.num_computed_tokens,
                             reserve_point,
                             prefix,
                             /*shapes=*/{},
                             /*claimed=*/{},
                             boundary_tree_,
                             write_back_evictions_,
                             /*evicts_blocks_past_segments=*/true,
                             victims,
                             copy_ins};
    // Not fragmentation: the bound's own caveats, which compacting would
    // only churn at. The request waits as before. The count covers the
    // groups compaction gathers and the demand, not the parents the probe
    // holds in the others (a hit page or tail block of a group of one block
    // per parent, a State checkpoint), so passing it does not promise that
    // the retry fits (docs/design/scheduler.md §1.4).
    if (planner.DenseLayoutParents(bucket_pinned, free_movers) > pool_.NumLcmBlocks()) {
        return std::nullopt;
    }

    AdmissionResult result{
        .new_page_ids = std::vector<std::vector<std::int32_t>>(groups_.size()),
        .pages_to_copy = std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>>(groups_.size()),
    };
    const std::uint64_t moves_before = lone_compaction_moves_;
    bool fits = false;
    // At most one of the probe's tail blocks moves per compaction, in any
    // group: the retry copies each from where it was (recordAdmittedPages),
    // and two such copies landing on each other's old places would swap their
    // bytes, which no order of the plan's copies can do.
    std::int64_t tails_moved = 0;
    for (std::size_t i = 0; i < groups_.size() && !fits; ++i) {
        if (members[i].empty()) {
            continue;
        }
        const std::uint32_t group_id = groups_[i].Id();
        GroupAllocator& allocator = groups_[i].Allocator();
        PrefixCacheIndex& index = groups_[i].Index();
        BlockTable& table = *demands[i].table;
        const std::int32_t packing = pool_.Packing(group_id);
        const std::int32_t buckets = allocator.ShardCount();
        // The parents holding the request's blocks: one with a block that
        // cannot move is kept whatever it holds, then the densest, and of
        // equally dense ones the one with more bucket-bound blocks. So a
        // parent of free blocks alone is vacated before an equally dense one
        // holding a bound block: its blocks re-place in any bucket, with no
        // copy and no displacement. The order also keeps every kept parent
        // bound through the moves. A free block gives its slot up (below)
        // only to a bound block of a vacated parent, which ranks after every
        // kept one. So a kept parent holding one block of the request, a
        // free one (never a fixed parent), loses none: every parent ranked
        // after it holds one free block alone. Any other kept parent keeps a
        // block of the request (held, so no clearing evicts it) until the
        // bound block takes the slot a free one gave up.
        struct Parent {
            std::int32_t id;
            std::int32_t count{0};
            std::int32_t bound{0};
            bool fixed{false};
            std::vector<std::size_t> members;
        };
        std::vector<Parent> parents;
        std::unordered_map<std::int32_t, std::size_t> parent_index;
        for (std::size_t m = 0; m < members[i].size(); ++m) {
            const Member& member = members[i][m];
            const auto [it, inserted] = parent_index.try_emplace(member.location.lcm_block_id, parents.size());
            if (inserted) {
                parents.push_back(Parent{.id = member.location.lcm_block_id});
            }
            Parent& parent = parents[it->second];
            ++parent.count;
            parent.bound += member.in_bucket ? 1 : 0;
            parent.fixed = parent.fixed || member.kind == Member::Kind::kFixed;
            parent.members.push_back(m);
        }
        std::ranges::sort(parents, [](const Parent& a, const Parent& b) {
            return std::tuple(!a.fixed, -a.count, -a.bound, a.id) < std::tuple(!b.fixed, -b.count, -b.bound, b.id);
        });
        const std::vector<std::int64_t> holes(static_cast<std::size_t>(buckets), 0);
        const std::size_t kept = static_cast<std::size_t>(
            std::max<std::int64_t>(ParentsNeededForBuckets(bucket_pinned[i], holes, free_movers[i], packing),
                                   std::ranges::count_if(parents, &Parent::fixed)));
        // Vacated in the reverse order -- the sparsest first, of those the
        // one with the fewest bucket-bound blocks -- a parent whole or not
        // at all: a block moved out of a parent that keeps another of the
        // request's blocks frees nothing.
        for (std::size_t p = parents.size(); p-- > kept && !fits;) {
            // Nor does one moved out of a parent the retry cannot empty:
            // beside the request's blocks there, every occupant must be a
            // slot the retry reclaims as the request's own or a cache entry
            // its block-by-block pass takes. A held page keeps the parent
            // bound, so it is skipped like a parent with a block that finds
            // no slot. So is one whose moves would make a second tail block
            // of the probe move: it waits for a later compaction when
            // another moved first (tails_moved), and is never vacated when it
            // holds two of them.
            const auto tails_here = std::ranges::count_if(
                parents[p].members, [&](std::size_t m) { return members[i][m].kind == Member::Kind::kTail; });
            bool vacatable = tails_moved + tails_here <= 1;
            for (std::int32_t child = 0; child < packing && vacatable; ++child) {
                const CacheBlockLocation slot{.lcm_block_id = parents[p].id, .slot_index = child};
                vacatable = !pool_.IsOccupied(slot) ||
                            std::ranges::any_of(parents[p].members,
                                                [&](std::size_t m) { return members[i][m].location == slot; }) ||
                            planner.IsRequestReclaim(slot) || planner.PlanBlockAlone(group_id, slot).has_value();
            }
            if (!vacatable) {
                continue;
            }
            struct Move {
                std::size_t member{0};
                CacheBlockLocation destination;
                // The cache entry at the destination and what it takes along.
                std::optional<VictimSet> clearing;
            };
            std::vector<Move> moves;
            std::vector<CacheBlockLocation> reserved;
            // A slot of a kept parent, in `bucket` or in any, `first_parent`'s
            // before the others when given: free, or held by a cache entry
            // nothing holds that the block-by-block pass may take. A block of
            // the request's own, or one in use, never.
            const auto find_destination = [&](std::optional<std::int32_t> bucket,
                                              std::optional<std::int32_t> first_parent) -> std::optional<Move> {
                std::vector<std::int32_t> searched;
                searched.reserve(kept + 1);
                if (first_parent) {
                    searched.push_back(*first_parent);
                }
                for (std::size_t k = 0; k < kept; ++k) {
                    if (parents[k].id != first_parent) {
                        searched.push_back(parents[k].id);
                    }
                }
                for (const std::int32_t parent : searched) {
                    for (std::int32_t child = bucket.value_or(0); child < packing; child += bucket ? buckets : 1) {
                        const CacheBlockLocation candidate{.lcm_block_id = parent, .slot_index = child};
                        if (std::ranges::find(reserved, candidate) != reserved.end()) {
                            continue;
                        }
                        if (!pool_.IsOccupied(candidate)) {
                            return Move{.destination = candidate};
                        }
                        if (std::optional<VictimSet> clearing = planner.PlanBlockAlone(group_id, candidate)) {
                            return Move{.destination = candidate, .clearing = std::move(clearing)};
                        }
                    }
                }
                return std::nullopt;
            };
            // Bucket-bound blocks first; a free block takes what is left, in
            // any bucket.
            std::vector<std::size_t> order = parents[p].members;
            std::ranges::stable_partition(order, [&](std::size_t m) { return members[i][m].in_bucket; });
            bool whole = true;
            for (const std::size_t m : order) {
                const Member& member = members[i][m];
                const std::int32_t bucket = allocator.BucketOf(member.location);
                std::optional<Move> move = find_destination(
                    member.in_bucket ? std::optional<std::int32_t>{bucket} : std::nullopt, std::nullopt);
                // No slot of its bucket left in the kept parents: one of the
                // request's free blocks there gives its slot up and re-places
                // anywhere (page-cyclic sharding keeps a written block's bytes
                // with their owner) -- in its own parent first. It leaves
                // before the member arrives, and a parent it left empty would
                // be released (unbound from the group) in between. The parent
                // order above leaves another of the request's blocks there in
                // any case; re-placing in its own parent first keeps the block
                // itself there whenever a free or clearable slot is left in it
                // (in another bucket: none of this bucket is, in any kept
                // parent), whatever the order.
                for (std::size_t k = 0; k < kept && !move && member.in_bucket; ++k) {
                    for (const std::size_t u : parents[k].members) {
                        const Member& other = members[i][u];
                        if (other.in_bucket || allocator.BucketOf(other.location) != bucket ||
                            std::ranges::find(reserved, other.location) != reserved.end()) {
                            continue;
                        }
                        std::optional<Move> displaced = find_destination(std::nullopt, other.location.lcm_block_id);
                        if (!displaced) {
                            break;
                        }
                        displaced->member = u;
                        reserved.push_back(displaced->destination);
                        moves.push_back(std::move(*displaced));
                        move = Move{.destination = other.location};
                        break;
                    }
                }
                if (!move) {
                    whole = false;
                    break;
                }
                move->member = m;
                reserved.push_back(move->destination);
                moves.push_back(std::move(*move));
            }
            if (!whole) {
                continue;
            }
            for (Move& move : moves) {
                Member& member = members[i][move.member];
                if (move.clearing && pool_.IsOccupied(move.destination)) {
                    EvictedSegment* segment = write_back_evictions_ ? &evicted_segments_.emplace_back() : nullptr;
                    for (const VictimSet::Member& victim : move.clearing->members) {
                        evictCachedBlock(victim.group_id, victim.location, &DeviceEvictionStats::compaction, segment);
                    }
                }
                FatalCheck(!pool_.IsOccupied(move.destination), "compaction destination changed before placement");
                // A kept parent never empties before the moves into it
                // (above): one released would read unoccupied and unbound.
                _assert(pool_.BoundGroup(move.destination.lcm_block_id) == group_id,
                        "a compaction destination lies in a parent bound to its group");
                if (member.kind == Member::Kind::kHit || member.kind == Member::Kind::kTail) {
                    // Pinned through the move: the entry follows its copy,
                    // and the pin with it.
                    CacheBlockRef& pin = hit_pins[member.hit];
                    CacheBlockRef fresh = pool_.AcquireBlockAt(group_id, move.destination);
                    _assert(allocator.BucketOf(fresh->Location()) == allocator.BucketOf(member.location),
                            "a copied block keeps its bucket");
                    result.pages_to_copy[i].emplace_back(allocator.ResolveCacheBlockId(*pin),
                                                         allocator.ResolveCacheBlockId(*fresh));
                    index.Relocate(pool_, member.location, fresh);
                    pin = std::move(fresh);
                } else if (member.kind == Member::Kind::kCopyIn) {
                    // Held outside the table for the slot it serves: re-placed
                    // in its bucket, never zeroed. The copy-in fills it whole,
                    // in this very plan when the retry spends it, and the
                    // runtime copies before it zeroes.
                    CacheBlockRef fresh = pool_.AcquireBlockAt(group_id, move.destination);
                    _assert(allocator.BucketOf(fresh->Location()) == allocator.BucketOf(member.location),
                            "a copy-in block keeps its bucket");
                    copy_in_blocks[i][static_cast<std::size_t>(member.slot - copy_in_slots[i].first)] =
                        std::move(fresh);
                } else {
                    // The block the slot gives up is the entry's until the
                    // entry has followed the copy; it leaves with `replaced`.
                    const CacheBlockRef replaced = allocator.SwapInFreshBlock(
                        pool_, table, member.slot, table.AvailableTokens(), move.destination);
                    const CacheBlockRef& moved = table.Blocks()[static_cast<std::size_t>(member.slot)];
                    _assert(!member.in_bucket ||
                                allocator.BucketOf(moved->Location()) == allocator.BucketOf(member.location),
                            "a block bound to its bucket keeps it");
                    if (member.kind == Member::Kind::kUnwritten) {
                        // A free block an earlier vacate round displaced
                        // can be displaced again, and the page it leaves
                        // can take a copy: it is no longer zeroed.
                        std::erase(result.new_page_ids[i], allocator.ResolveCacheBlockId(*replaced));
                        result.new_page_ids[i].push_back(allocator.ResolveCacheBlockId(*moved));
                    } else {
                        result.pages_to_copy[i].emplace_back(allocator.ResolveCacheBlockId(*replaced),
                                                             allocator.ResolveCacheBlockId(*moved));
                    }
                    if (member.kind == Member::Kind::kPublished) {
                        index.Relocate(pool_, member.location, moved);
                    }
                }
                member.location = move.destination;
                ++lone_compaction_moves_;
            }
            tails_moved += tails_here;
            fits = admission_fits();
        }
    }
    // A set whose every block was dropped records nothing.
    std::erase_if(evicted_segments_, [](const EvictedSegment& segment) { return segment.members.empty(); });
    if (lone_compaction_moves_ == moves_before) {
        return std::nullopt;
    }
    ++lone_compactions_;
    return result;
}

}  // namespace tokenspeed
