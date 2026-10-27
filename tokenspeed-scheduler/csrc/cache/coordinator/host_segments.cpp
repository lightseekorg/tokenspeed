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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <queue>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/coordinator/shadow_occupancy.h"
#include "utils.h"

namespace tokenspeed {

CacheCoordinator::HostPlacementPlan CacheCoordinator::planHostSegments(
    std::span<const std::uint32_t> group_ids, std::span<const std::int32_t> buckets,
    std::span<const HostSegmentRows> segments) const {
    _assert(host_pool_ != nullptr, "Host placement requires a host pool");
    _assert(group_ids.size() == buckets.size(), "Host allocation needs one bucket per requested block");

    std::size_t next_row = 0;
    for (const HostSegmentRows& segment : segments) {
        _assert(segment.first_row == next_row && segment.past_end_row >= segment.first_row,
                "Host segments cover the rows in order");
        next_row = segment.past_end_row;
    }
    _assert(next_row == group_ids.size(), "Host segments cover every row");
    // The placed segments' rows by group and bucket, the demand the shadow
    // fits: every segment's to begin with.
    std::vector<std::vector<std::int64_t>> need_by_bucket(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        need_by_bucket[i].assign(static_cast<std::size_t>(groups_[i].Allocator().ShardCount()), 0);
    }
    for (std::size_t row = 0; row < group_ids.size(); ++row) {
        _assert(group_ids[row] < groups_.size(), "Host block group id out of range");
        _assert(buckets[row] >= 0 && buckets[row] < groups_[group_ids[row]].Allocator().ShardCount(),
                "Host block bucket is outside the group's shard count");
        ++need_by_bucket[group_ids[row]][static_cast<std::size_t>(buckets[row])];
    }
    const std::vector<std::int64_t> no_dense_need(groups_.size(), 0);
    ShadowOccupancy shadow{*host_pool_, groups_};
    // Nothing is evicted while free capacity fits every segment.
    if (shadow.Fits(need_by_bucket, no_dense_need)) {
        return HostPlacementPlan{.num_placed = segments.size()};
    }
    std::size_t num_placed_segments = segments.size();

    // What one victim takes from the Host: its entries in eviction order
    // (the deepest node first, a node's checkpoints before its other
    // entries), the nodes whose Host page (their KV entries) it takes, the
    // nearest node at or above it whose Host page it leaves
    // (nearest_untaken_page: a page whose checkpoints alone it takes is that
    // page itself), the counters of its class, and whether the placed segments
    // still need it (required: false once the restore pass returns it).
    struct Victim {
        std::vector<std::pair<std::uint32_t, CacheBlockLocation>> members{};
        std::vector<const BoundaryTree::Node*> nodes_with_page_taken{};
        const BoundaryTree::Node* nearest_untaken_page{nullptr};
        HostEvictionCounts HostEvictionStats::*reason{nullptr};
        bool required{true};
    };
    std::vector<Victim> victims;
    std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> taken;
    // Per node, how many Host checkpoint boundaries at or under it a victim
    // takes.
    std::unordered_map<const BoundaryTree::Node*, std::int32_t> taken_checkpoint_boundary_counts;
    // Per node, the victim that took its Host page (its KV entries); a
    // victim that takes only a node's checkpoints leaves the page and is
    // not recorded.
    std::unordered_map<const BoundaryTree::Node*, std::size_t> victim_index_taking_page;
    const auto num_remaining_checkpoint_boundaries = [&](const BoundaryTree::Node& node) {
        const auto it = taken_checkpoint_boundary_counts.find(&node);
        return node.checkpoint_boundaries_in_subtree[BoundaryTree::kHost] -
               (it == taken_checkpoint_boundary_counts.end() ? 0 : it->second);
    };
    // The Host entries of one node no victim takes yet, its checkpoints
    // first.
    const auto untaken_entries_of = [&](const BoundaryTree::Node& node) {
        std::vector<NodeEntry> entries = NodeEntries(*host_pool_, node);
        std::erase_if(entries, [&](const NodeEntry& entry) { return taken.contains(entry.location); });
        return entries;
    };
    // The newest access epoch of a node's entries on one tier (of its KV
    // entries alone when kv_only), nullopt when the tier holds none.
    const auto newest_epoch = [&](const BlockPool& pool, const BoundaryTree::Node& node, bool kv_only) {
        std::optional<std::uint64_t> newest;
        ForEachBoundaryKey(
            CacheKey{.content_hash = *node.hash}, node.tail_tokens, [&](std::size_t i, const CacheKey& key) {
                if (kv_only && GroupIsState(i)) {
                    return;
                }
                if (const std::optional<CacheBlockLocation> location = locationOf(pool, i, key)) {
                    newest = std::max(newest.value_or(0),
                                      groups_[i].Index().MetadataFor(pool, *location)->last_access_epoch);
                }
            });
        return newest;
    };

    // A candidate victim's node and its rank, the smallest first: by class
    // (VictimClass: a dead entry, a duplicate or a superseded checkpoint
    // boundary, Host-only cache), then by epoch -- a dead entry's and
    // Host-only cache's newest Host entry, oldest first; a duplicate's or
    // superseded checkpoint boundary's Device copy, newest first, since the
    // Device keeps that one longest -- then the shallower boundary, keeping
    // the longer frontier as Device eviction does, then the candidate
    // node's first Host entry.
    enum class VictimClass : std::uint8_t { kDead, kCoveredByDevice, kHostOnly };
    struct Candidate {
        const BoundaryTree::Node* candidate_node{nullptr};
        VictimClass victim_class{VictimClass::kDead};
        // The epoch above, complemented for kCoveredByDevice so its newest
        // goes first.
        std::uint64_t epoch_sort_key{0};
        std::int32_t boundary_tokens{0};
        CacheBlockLocation first_entry_location;
        HostEvictionCounts HostEvictionStats::*reason{nullptr};
    };
    const auto ranks_after = [](const Candidate& lhs, const Candidate& rhs) {
        return std::tuple{lhs.victim_class, lhs.epoch_sort_key, lhs.boundary_tokens,
                          lhs.first_entry_location.lcm_block_id, lhs.first_entry_location.slot_index} >
               std::tuple{rhs.victim_class, rhs.epoch_sort_key, rhs.boundary_tokens,
                          rhs.first_entry_location.lcm_block_id, rhs.first_entry_location.slot_index};
    };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(ranks_after)> candidates{ranks_after};
    const auto add_candidate = [&](const BoundaryTree::Node& node) {
        const std::vector<NodeEntry> entries = untaken_entries_of(node);
        if (entries.empty()) {
            return;
        }
        Candidate candidate{.candidate_node = &node,
                            .boundary_tokens = node.boundary_tokens,
                            .first_entry_location = entries.front().location};
        const auto num_checkpoint_entries =
            std::ranges::count_if(entries, [&](const NodeEntry& entry) { return GroupIsState(entry.group_id); });
        // Dead on the Host: a tail no probe enumerates any more; a
        // checkpoint boundary missing some State group's checkpoint; with
        // State groups, a node with no remaining Host checkpoint boundary
        // at or under it.
        const bool dead = (node.tail_tokens > 0 && !tails_.Contains(*node.hash)) ||
                          (num_checkpoint_entries > 0 && num_checkpoint_entries < num_state_groups_) ||
                          (num_state_groups_ > 0 && num_remaining_checkpoint_boundaries(node) == 0);
        // Superseded: a Host checkpoint boundary whose page (its parent for
        // a tail, else the node itself) has a Device checkpoint boundary
        // under it other than the page's own and this node's; it ranks by
        // that page's newest Device KV access.
        const BoundaryTree::Node* page = node.tail_tokens > 0 ? node.parent : &node;
        const std::int32_t num_deeper_device_checkpoint_boundaries =
            page == nullptr ? 0
                            : page->checkpoint_boundaries_in_subtree[BoundaryTree::kDevice] -
                                  (page->has_checkpoint[BoundaryTree::kDevice] ? 1 : 0) -
                                  (page != &node && node.has_checkpoint[BoundaryTree::kDevice] ? 1 : 0);
        if (dead) {
            candidate.victim_class = VictimClass::kDead;
            candidate.epoch_sort_key = *newest_epoch(*host_pool_, node, /*kv_only=*/false);
            candidate.reason = &HostEvictionStats::dead_entry;
        } else if (DeviceBoundaryResidency(CacheKey{.content_hash = *node.hash}, node.tail_tokens) ==
                   BoundaryResidency::kComplete) {
            candidate.victim_class = VictimClass::kCoveredByDevice;
            candidate.epoch_sort_key = ~*newest_epoch(pool_, node, /*kv_only=*/false);
            candidate.reason = &HostEvictionStats::duplicate;
        } else if (const std::optional<std::uint64_t> superseding_epoch =
                       num_checkpoint_entries > 0 && num_deeper_device_checkpoint_boundaries >= 1
                           ? newest_epoch(pool_, *page, /*kv_only=*/true)
                           : std::nullopt) {
            candidate.victim_class = VictimClass::kCoveredByDevice;
            candidate.epoch_sort_key = ~*superseding_epoch;
            candidate.reason = &HostEvictionStats::superseded;
        } else {
            candidate.victim_class = VictimClass::kHostOnly;
            candidate.epoch_sort_key = *newest_epoch(*host_pool_, node, /*kv_only=*/false);
            candidate.reason = &HostEvictionStats::lru;
        }
        candidates.push(candidate);
    };

    // The candidate nodes: every Host checkpoint boundary, found through
    // the State groups' unpinned Host entries, and every Host leaf no Host
    // checkpoint boundary depends on (every Host leaf without State
    // groups). The Host's KV entries are never scanned.
    std::unordered_set<const BoundaryTree::Node*> listed;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!GroupIsState(i)) {
            continue;
        }
        const std::vector<PrefixCacheIndex::EvictionCandidate> entries =
            groups_[i].Index().EvictableCandidates(*host_pool_);
        for (const PrefixCacheIndex::EvictionCandidate& entry : entries) {
            const std::optional<CacheKey> key = groups_[i].Index().KeyOf(*host_pool_, entry.location);
            // An entry registered straight into an index has no node.
            const BoundaryTree::Node* node = boundary_tree_.Find(key->content_hash);
            if (node != nullptr && listed.insert(node).second) {
                add_candidate(*node);
            }
        }
    }
    for (const BoundaryTree::Node* leaf = boundary_tree_.FirstHostLeaf(); leaf != nullptr;
         leaf = leaf->next_host_leaf) {
        if (leaf->checkpoint_boundaries_in_subtree[BoundaryTree::kHost] == 0) {
            add_candidate(*leaf);
        }
    }

    // One joint fit over every segment: victims leave the shadow until
    // every row fits or no candidate is left.
    std::unordered_set<const BoundaryTree::Node*> parents_added_as_candidates;
    // Per page above a taken victim, the first of its children the check
    // has not yet found empty of Host entries. The children before it stay
    // empty for the rest of the plan (walked_by_taken_victims and taken
    // only grow), so each check resumes there instead of walking a wide
    // fork's children again for every branch that leaves.
    std::unordered_map<const BoundaryTree::Node*, const BoundaryTree::Node*> next_child_to_check;
    // The nodes a taken victim's walk covered: every Host page under them
    // is taken, so no later walk descends into them again (a chain taken
    // page by page would otherwise be walked once per page).
    std::unordered_set<const BoundaryTree::Node*> walked_by_taken_victims;
    while (!shadow.Fits(need_by_bucket, no_dense_need) && !candidates.empty()) {
        const Candidate candidate = candidates.top();
        candidates.pop();
        const BoundaryTree::Node& candidate_node = *candidate.candidate_node;
        std::vector<NodeEntry> candidate_entries = untaken_entries_of(candidate_node);
        if (candidate_entries.empty()) {
            continue;  // taken with an earlier victim
        }
        Victim victim{.reason = candidate.reason};
        // The nodes the victim takes entries of, with those entries and
        // whether they include the node's Host page.
        struct TakenNode {
            const BoundaryTree::Node* node;
            std::vector<NodeEntry> entries;
            bool takes_page;
        };
        std::vector<TakenNode> taken_nodes;
        std::vector<const BoundaryTree::Node*> walked;
        const auto num_candidate_checkpoint_entries = static_cast<std::size_t>(std::ranges::count_if(
            candidate_entries, [&](const NodeEntry& entry) { return GroupIsState(entry.group_id); }));
        if (num_candidate_checkpoint_entries > 0 && num_remaining_checkpoint_boundaries(candidate_node) >= 2) {
            // Deeper Host checkpoint boundaries still read this page: its
            // checkpoints leave alone.
            candidate_entries.resize(num_candidate_checkpoint_entries);
            taken_nodes.push_back(
                TakenNode{.node = &candidate_node, .entries = std::move(candidate_entries), .takes_page = false});
            victim.nearest_untaken_page = &candidate_node;
        } else {
            // The candidate node with the Host pages above it that serve no
            // other Host checkpoint boundary and are not in use -- none
            // without State groups, where the chain's Host pages leave leaf
            // by leaf -- and the Host pages under them. The climb keeps the
            // entries it looked up, the candidate node's first.
            const BoundaryTree::Node* top = &candidate_node;
            std::vector<std::pair<const BoundaryTree::Node*, std::vector<NodeEntry>>> climbed;
            climbed.emplace_back(&candidate_node, std::move(candidate_entries));
            if (num_state_groups_ > 0) {
                while (top->parent != nullptr && top->parent->on_host && top->parent->pending_host_stores == 0 &&
                       num_remaining_checkpoint_boundaries(*top->parent) ==
                           num_remaining_checkpoint_boundaries(candidate_node)) {
                    std::vector<NodeEntry> entries = untaken_entries_of(*top->parent);
                    if (std::ranges::any_of(entries, &NodeEntry::in_use)) {
                        break;
                    }
                    top = top->parent;
                    climbed.emplace_back(top, std::move(entries));
                }
            }
            // A page an eviction write-back is bringing here is judged when
            // it lands by the checkpoint boundaries under it: taking these
            // would leave it to arrive dead, above the victim or in it (a
            // KV page on its way under its Host checkpoints).
            if (top->parent != nullptr && top->parent->pending_host_stores > 0) {
                continue;
            }
            victim.nearest_untaken_page = top->parent;
            // Post-order over the Host children: a node after the nodes
            // under it, which completes the climbed nodes in climb order.
            // What hangs under a page off the Host continues the Device's
            // prefix, not this one.
            struct Frame {
                const BoundaryTree::Node* node;
                const BoundaryTree::Node* next_child;
            };
            std::vector<Frame> stack{Frame{.node = top, .next_child = top->first_child}};
            std::size_t next_climbed = 0;
            while (!stack.empty()) {
                Frame& frame = stack.back();
                if (const BoundaryTree::Node* child = frame.next_child) {
                    frame.next_child = child->next_sibling;
                    if (child->on_host && !walked_by_taken_victims.contains(child)) {
                        stack.push_back(Frame{.node = child, .next_child = child->first_child});
                    }
                    continue;
                }
                walked.push_back(frame.node);
                if (std::vector<NodeEntry> entries =
                        next_climbed < climbed.size() && climbed[next_climbed].first == frame.node
                            ? std::move(climbed[next_climbed++].second)
                            : untaken_entries_of(*frame.node);
                    !entries.empty()) {
                    const bool takes_page = std::ranges::any_of(
                        entries, [&](const NodeEntry& entry) { return !GroupIsState(entry.group_id); });
                    taken_nodes.push_back(
                        TakenNode{.node = frame.node, .entries = std::move(entries), .takes_page = takes_page});
                }
                stack.pop_back();
            }
            if (std::ranges::any_of(walked,
                                    [](const BoundaryTree::Node* node) { return node->pending_host_stores > 0; })) {
                continue;
            }
        }
        // One member in use passes the whole victim over.
        if (std::ranges::any_of(taken_nodes, [](const TakenNode& taken_node) {
                return std::ranges::any_of(taken_node.entries, &NodeEntry::in_use);
            })) {
            continue;
        }
        walked_by_taken_victims.insert(walked.begin(), walked.end());
        for (const TakenNode& taken_node : taken_nodes) {
            bool takes_checkpoints = false;
            for (const NodeEntry& entry : taken_node.entries) {
                shadow.Remove(entry.group_id, entry.location);
                taken.insert(entry.location);
                victim.members.emplace_back(entry.group_id, entry.location);
                takes_checkpoints = takes_checkpoints || GroupIsState(entry.group_id);
            }
            if (takes_checkpoints && taken_node.node->has_checkpoint[BoundaryTree::kHost]) {
                for (const BoundaryTree::Node* ancestor = taken_node.node; ancestor != nullptr;
                     ancestor = ancestor->parent) {
                    ++taken_checkpoint_boundary_counts[ancestor];
                }
            }
            if (taken_node.takes_page) {
                victim_index_taking_page.emplace(taken_node.node, victims.size());
                victim.nodes_with_page_taken.push_back(taken_node.node);
            }
        }
        // The page above is a candidate too once no Host checkpoint
        // boundary depends on it and every Host page under it is gone.
        const BoundaryTree::Node* nearest_untaken_page = victim.nearest_untaken_page;
        victims.push_back(std::move(victim));
        if (nearest_untaken_page != nullptr && nearest_untaken_page->on_host &&
            num_remaining_checkpoint_boundaries(*nearest_untaken_page) == 0 &&
            !parents_added_as_candidates.contains(nearest_untaken_page)) {
            const BoundaryTree::Node*& child =
                next_child_to_check.try_emplace(nearest_untaken_page, nearest_untaken_page->first_child).first->second;
            for (; child != nullptr; child = child->next_sibling) {
                if (child->on_host && !walked_by_taken_victims.contains(child) && !untaken_entries_of(*child).empty()) {
                    break;
                }
            }
            if (child == nullptr) {
                parents_added_as_candidates.insert(nearest_untaken_page);
                add_candidate(*nearest_untaken_page);
            }
        }
    }

    // Whole segments go, the last first, while the rest cannot fit.
    while (!shadow.Fits(need_by_bucket, no_dense_need)) {
        _assert(num_placed_segments > 0, "a batch without rows always fits");
        --num_placed_segments;
        const HostSegmentRows& dropped = segments[num_placed_segments];
        for (std::size_t row = dropped.first_row; row < dropped.past_end_row; ++row) {
            --need_by_bucket[group_ids[row]][static_cast<std::size_t>(buckets[row])];
        }
    }

    // Return each victim the placed segments can do without, newest first
    // and whole; one whose nearest untaken page a later victim took stays
    // gone, or the Host would keep a page without its parent.
    for (std::size_t v = victims.size(); v > 0; --v) {
        Victim& victim = victims[v - 1];
        if (victim.nearest_untaken_page != nullptr && victim_index_taking_page.contains(victim.nearest_untaken_page)) {
            continue;
        }
        for (const auto& [group_id, location] : victim.members) {
            shadow.Restore(group_id, location);
        }
        if (!shadow.Fits(need_by_bucket, no_dense_need)) {
            for (const auto& [group_id, location] : victim.members) {
                shadow.Remove(group_id, location);
            }
            continue;
        }
        victim.required = false;
        for (const BoundaryTree::Node* node : victim.nodes_with_page_taken) {
            victim_index_taking_page.erase(node);
        }
    }

    HostPlacementPlan plan{.num_placed = num_placed_segments};
    for (const Victim& victim : victims) {
        if (!victim.required) {
            continue;
        }
        for (const auto& [group_id, location] : victim.members) {
            plan.victims.push_back(
                HostPlacementPlan::Victim{.group_id = group_id, .location = location, .reason = victim.reason});
        }
    }
    return plan;
}

CacheCoordinator::HostAllocationBatch CacheCoordinator::AcquireHostSegments(std::span<const std::uint32_t> group_ids,
                                                                            std::span<const std::int32_t> buckets,
                                                                            std::span<const HostSegmentRows> segments) {
    return acquireHostPlacement(planHostSegments(group_ids, buckets, segments), group_ids, buckets, segments);
}

CacheCoordinator::HostAllocationBatch CacheCoordinator::acquireHostPlacement(
    const HostPlacementPlan& plan, std::span<const std::uint32_t> group_ids, std::span<const std::int32_t> buckets,
    std::span<const HostSegmentRows> segments) {
    // Evict, then allocate: the fit holds once the victims are gone.
    for (const auto& [group_id, location, reason] : plan.victims) {
        // A checkpoint takes the other State groups' at its hash along.
        if (groups_[group_id].Index().Contains(*host_pool_, location)) {
            FatalCheck(evictHostCachedBlock(group_id, location, reason), "a planned Host victim became pinned");
        }
        FatalCheck(!host_pool_->IsOccupied(location), "a planned Host victim's block is still occupied");
    }
    HostAllocationBatch batch;
    batch.blocks.resize(group_ids.size());
    batch.stats.requested = group_ids.size();
    const std::size_t num_placed_rows = plan.num_placed == 0 ? 0 : segments[plan.num_placed - 1].past_end_row;
    for (std::uint32_t group_id = 0; group_id < groups_.size(); ++group_id) {
        std::vector<std::size_t> rows;
        std::vector<std::int32_t> group_buckets;
        for (std::size_t row = 0; row < num_placed_rows; ++row) {
            if (group_ids[row] == group_id) {
                rows.push_back(row);
                group_buckets.push_back(buckets[row]);
            }
        }
        std::vector<CacheBlockRef> blocks = host_pool_->AcquireBlocksInBuckets(group_id, group_buckets);
        FatalCheck(blocks.size() == rows.size(), "a planned Host placement did not fit");
        for (std::size_t j = 0; j < rows.size(); ++j) {
            batch.blocks[rows[j]] = std::move(blocks[j]);
        }
    }
    batch.stats.allocated = num_placed_rows;
    batch.stats.unallocated = batch.stats.requested - num_placed_rows;
    return batch;
}

}  // namespace tokenspeed
