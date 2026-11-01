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
#include <iterator>
#include <memory_resource>
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

template <typename Visitor>
void CacheCoordinator::visitHostKvDuplicateCandidates(Visitor&& visit) const {
    if (!boundary_tree_.HasHostKvDuplicateCandidate()) {
        return;
    }
    std::pmr::monotonic_buffer_resource scratch;
    std::int64_t lookback_tokens = 0;
    std::vector<std::size_t> closed_groups;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (groups_[i].Matcher().IsPrefixClosed()) {
            closed_groups.push_back(i);
        }
        if (!GroupIsReplayable(i) && !GroupIsState(i) && !groups_[i].Matcher().IsPrefixClosed()) {
            lookback_tokens =
                std::max(lookback_tokens, static_cast<std::int64_t>(groups_[i].Matcher().BoundaryLookbackPages()) *
                                              geometry_[i].BlockGranularity());
        }
    }
    const std::uint64_t scan_generation = ++host_duplicate_scan_generation_;
    _assert(scan_generation != 0, "Host duplicate scan generation overflow");
    std::vector<const BoundaryTree::Node*> ancestors;
    std::vector<const BoundaryTree::Node*> pending;
    std::pmr::unordered_map<const BoundaryTree::Node*, std::int32_t> device_window_splits{&scratch};
    const auto device_window_split = [&](const BoundaryTree::Node& boundary) {
        const auto found = device_window_splits.find(&boundary);
        if (found != device_window_splits.end()) {
            return found->second;
        }
        // Match the Host-suffix split: only a missing Device window block
        // below the full-page frontier forces closed KV onto the Host.
        // Several duplicate pages can depend on the same checkpoint window.
        std::int32_t split = boundary.boundary_tokens - boundary.tail_tokens;
        const BoundaryTree::Node* page = boundary.tail_tokens > 0 ? boundary.parent : &boundary;
        const std::int64_t first_window_token = std::max<std::int64_t>(0, boundary.boundary_tokens - lookback_tokens);
        while (page != nullptr && page->boundary_tokens > first_window_token) {
            const std::int32_t page_start = page->boundary_tokens - prefix_granularity_;
            ForEachBoundaryKey(
                CacheKey{.content_hash = *page->hash}, /*tail_tokens=*/0, [&](std::size_t i, const CacheKey& key) {
                    if (GroupIsState(i) || groups_[i].Matcher().IsPrefixClosed()) {
                        return;
                    }
                    const std::int32_t block = page_start / geometry_[i].BlockGranularity() + key.page_offset;
                    const BlockRange window =
                        SlidingWindowBlocks(static_cast<std::int32_t>(i), boundary.boundary_tokens);
                    if (block >= window.first && block < window.past_end && !groups_[i].Index().Contains(pool_, key)) {
                        split = std::min(split, page_start);
                    }
                });
            page = page->parent;
        }
        device_window_splits.emplace(&boundary, split);
        return split;
    };
    const auto store_pending_at_or_above = [&](const BoundaryTree::Node& node) {
        ancestors.clear();
        const BoundaryTree::Node* ancestor = &node;
        bool store_pending = false;
        while (ancestor != nullptr) {
            if (ancestor->host_store_scan_generation == scan_generation) {
                store_pending = ancestor->host_store_ancestor_pending;
                break;
            }
            ancestors.push_back(ancestor);
            if (ancestor->pending_host_stores > 0) {
                store_pending = true;
                break;
            }
            ancestor = ancestor->parent;
        }
        // Candidate chains share ancestors. Cache each answer only within
        // this scan, while Host stores and tree links cannot change.
        for (const BoundaryTree::Node* visited : ancestors) {
            visited->host_store_scan_generation = scan_generation;
            visited->host_store_ancestor_pending = store_pending;
        }
        return store_pending;
    };
    std::pmr::unordered_map<const BoundaryTree::Node*, bool> eligible_nodes{&scratch};
    const auto eligible = [&](const BoundaryTree::Node& node) {
        if (!node.has_host_kv_duplicate || !node.device_kv_prefix_complete) {
            return false;
        }
        const auto [entry, inserted] = eligible_nodes.try_emplace(&node, false);
        if (!inserted) {
            return entry->second;
        }
        if (store_pending_at_or_above(node)) {
            return false;
        }
        // A checkpoint can need this page only within its lookback distance.
        // Inspect that bounded subtree, including pinned and pending states,
        // rather than scanning every checkpoint in the cache for every plan.
        if (lookback_tokens > 0) {
            const std::int64_t past_end = node.boundary_tokens + lookback_tokens;
            pending.clear();
            pending.push_back(&node);
            while (!pending.empty()) {
                const BoundaryTree::Node& boundary = *pending.back();
                pending.pop_back();
                if (boundary.pending_host_stores > 0 ||
                    ((num_state_groups_ == 0 || boundary.has_checkpoint[BoundaryTree::kDevice] ||
                      boundary.has_checkpoint[BoundaryTree::kHost]) &&
                     device_window_split(boundary) < node.boundary_tokens)) {
                    return false;
                }
                for (const BoundaryTree::Node* child = boundary.first_child; child != nullptr;
                     child = child->next_sibling) {
                    if (child->boundary_tokens < past_end) {
                        pending.push_back(child);
                    }
                }
            }
        }
        entry->second = true;
        return true;
    };
    // Held Device copies cannot be demoted. Among the rest, newer Device
    // copies are likely to stay longer and avoid another Host write-back.
    // The indexes already order Device accesses. Merge one epoch at a time
    // across closed groups, sorting only its boundary/location ties, so a
    // placement that fits early never inspects every older duplicate.
    std::vector<PrefixCacheIndex::CommonCursor> cursors(closed_groups.size());
    std::vector<std::vector<PrefixCacheIndex::CommonEntry>> unpinned(closed_groups.size());
    std::vector<std::size_t> next_unpinned(closed_groups.size(), 0);
    const auto next_entry = [&](std::size_t i, bool pinned) -> std::optional<PrefixCacheIndex::CommonEntry> {
        if (pinned) {
            while (const auto entry = groups_[closed_groups[i]].Index().NextCommonRecentEntry(pool_, cursors[i])) {
                if (entry->source.pinned) {
                    return entry;
                }
                unpinned[i].push_back(*entry);
            }
            return std::nullopt;
        }
        // The first pass exhausted the index before reaching this class.
        // Its borrowed snapshots retain the same order and ownership facts.
        if (next_unpinned[i] < unpinned[i].size()) {
            return unpinned[i][next_unpinned[i]++];
        }
        return std::nullopt;
    };
    for (const bool pinned : {true, false}) {
        std::vector<std::optional<PrefixCacheIndex::CommonEntry>> next(closed_groups.size());
        for (std::size_t i = 0; i < closed_groups.size(); ++i) {
            next[i] = next_entry(i, pinned);
        }
        std::vector<HostKvDuplicateCandidate> candidates;
        while (true) {
            std::optional<std::uint64_t> epoch;
            for (const auto& entry : next) {
                if (entry) {
                    epoch = std::max(epoch.value_or(0), entry->source.metadata.last_access_epoch);
                }
            }
            if (!epoch) {
                break;
            }
            candidates.clear();
            for (std::size_t i = 0; i < closed_groups.size(); ++i) {
                while (next[i] && next[i]->source.metadata.last_access_epoch == *epoch) {
                    const PrefixCacheIndex::CommonEntry& device = *next[i];
                    const bool boundary_key =
                        device.key->namespace_id == kDefaultCacheNamespaceId && device.key->page_offset >= 0 &&
                        device.key->page_offset < BlocksPerBoundary(static_cast<std::int32_t>(closed_groups[i]));
                    const BoundaryTree::Node* node =
                        boundary_key ? boundary_tree_.Find(device.key->content_hash) : nullptr;
                    if (node != nullptr && !device.peer.pinned && eligible(*node)) {
                        candidates.push_back(HostKvDuplicateCandidate{
                            .group_id = groups_[closed_groups[i]].Id(),
                            .location = device.peer.location,
                            .device_epoch = *epoch,
                            .boundary_tokens = node->boundary_tokens,
                            .device_in_use = pinned,
                        });
                    }
                    next[i] = next_entry(i, pinned);
                }
            }
            std::ranges::sort(candidates, [](const HostKvDuplicateCandidate& lhs, const HostKvDuplicateCandidate& rhs) {
                return std::tuple{lhs.boundary_tokens, lhs.group_id, lhs.location.lcm_block_id,
                                  lhs.location.slot_index} < std::tuple{rhs.boundary_tokens, rhs.group_id,
                                                                        rhs.location.lcm_block_id,
                                                                        rhs.location.slot_index};
            });
            for (const HostKvDuplicateCandidate& candidate : candidates) {
                if (!visit(candidate)) {
                    return;
                }
            }
        }
    }
}

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
    // These snapshots and indexes live for one plan only. Allocate their
    // nodes together and release them after producing the owning result.
    std::pmr::monotonic_buffer_resource scratch;
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
    std::pmr::unordered_set<CacheBlockLocation, CacheBlockLocationHash> taken{&scratch};
    // Per node, how many Host checkpoint boundaries at or under it a victim
    // takes.
    std::pmr::unordered_map<const BoundaryTree::Node*, std::int32_t> taken_checkpoint_boundary_counts{&scratch};
    // Per node, the victim that took its Host page (its KV entries); a
    // victim that takes only a node's checkpoints leaves the page and is
    // not recorded.
    std::pmr::unordered_map<const BoundaryTree::Node*, std::size_t> victim_index_taking_page{&scratch};
    const auto num_remaining_host_checkpoint_boundaries = [&](const BoundaryTree::Node& node) {
        const auto it = taken_checkpoint_boundary_counts.find(&node);
        return node.checkpoint_boundaries_in_subtree[BoundaryTree::kHost] -
               (it == taken_checkpoint_boundary_counts.end() ? 0 : it->second);
    };
    // A plan changes only shadow occupancy. Reuse non-owning Host snapshots
    // while ranking and expanding its victims; the next plan reads them anew.
    struct HostEntry {
        std::uint32_t group_id;
        CacheBlockLocation location;
        bool in_use;
        std::uint64_t last_access_epoch;
    };
    struct HostEntries {
        explicit HostEntries(std::pmr::memory_resource* resource) : states{resource}, all{resource} {}
        std::pmr::vector<HostEntry> states;
        std::pmr::vector<HostEntry> all;
        bool all_loaded{false};
    };
    std::pmr::unordered_map<const BoundaryTree::Node*, HostEntries> host_entries{&scratch};
    const auto entries_for = [&](const BoundaryTree::Node& node) -> HostEntries& {
        return host_entries.try_emplace(&node, &scratch).first->second;
    };
    std::vector<const BoundaryTree::Node*> state_candidates;
    std::pmr::unordered_set<const BoundaryTree::Node*> listed{&scratch};
    // Read every State group before ranking a boundary: an unpinned group
    // can share its checkpoint with a pinned group. Borrow the index's key
    // and metadata together instead of copying keys and looking them up again.
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!GroupIsState(i)) {
            continue;
        }
        groups_[i].Index().ForEachEntry(
            *host_pool_, [&](const CacheKey& key, const PrefixCacheIndex::CachedBlockSnapshot& entry) {
                const BoundaryTree::Node* node = boundary_tree_.Find(key.content_hash);
                if (node == nullptr) {
                    return;
                }
                if (!entry.pinned && listed.insert(node).second) {
                    state_candidates.push_back(node);
                }
                if (key.namespace_id == kDefaultCacheNamespaceId && key.group_id == groups_[i].Id() &&
                    key.page_offset == 0) {
                    std::pmr::vector<HostEntry>& states = entries_for(*node).states;
                    if (states.empty()) {
                        states.reserve(num_state_groups_);
                    }
                    states.push_back(HostEntry{.group_id = key.group_id,
                                               .location = entry.location,
                                               .in_use = entry.pinned,
                                               .last_access_epoch = entry.metadata.last_access_epoch});
                }
            });
    }
    const auto host_entries_of = [&](const BoundaryTree::Node& node) -> const std::pmr::vector<HostEntry>& {
        HostEntries& entries = entries_for(node);
        if (!entries.all_loaded) {
            entries.all = entries.states;
            ForEachBoundaryKey(
                CacheKey{.content_hash = *node.hash}, node.tail_tokens, [&](std::size_t i, const CacheKey& key) {
                    if (GroupIsState(i)) {
                        return;
                    }
                    if (const auto entry = groups_[i].Index().Inspect(*host_pool_, key)) {
                        entries.all.push_back(HostEntry{.group_id = groups_[i].Id(),
                                                        .location = entry->location,
                                                        .in_use = entry->pinned,
                                                        .last_access_epoch = entry->metadata.last_access_epoch});
                    }
                });
            entries.all_loaded = true;
        }
        return entries.all;
    };
    // The Host entries of one node no victim takes yet, its checkpoints first.
    const auto untaken_entries_of = [&](const BoundaryTree::Node& node) {
        const auto& snapshot = host_entries_of(node);
        std::vector<HostEntry> entries(snapshot.begin(), snapshot.end());
        std::erase_if(entries, [&](const HostEntry& entry) { return taken.contains(entry.location); });
        return entries;
    };
    // The newest access epoch of a node's entries on one tier (of its KV
    // entries alone when kv_only), nullopt when the tier holds none.
    const auto newest_epoch = [&](const BlockPool& pool, const BoundaryTree::Node& node, bool kv_only) {
        std::optional<std::uint64_t> newest;
        if (&pool == host_pool_) {
            for (const HostEntry& entry : host_entries_of(node)) {
                if (!kv_only || !GroupIsState(entry.group_id)) {
                    newest = std::max(newest.value_or(0), entry.last_access_epoch);
                }
            }
            return newest;
        }
        ForEachBoundaryKey(CacheKey{.content_hash = *node.hash}, node.tail_tokens,
                           [&](std::size_t i, const CacheKey& key) {
                               if (kv_only && GroupIsState(i)) {
                                   return;
                               }
                               if (const auto entry = groups_[i].Index().Inspect(pool, key)) {
                                   newest = std::max(newest.value_or(0), entry->metadata.last_access_epoch);
                               }
                           });
        return newest;
    };

    // Keep aligned checkpoints at their State LRU rank: a later prompt can
    // diverge before the deeper frontier even within the same request cohort.
    // A continued tail retains its existing priority so old input tails do
    // not crowd out other sessions after the request has moved on.
    std::pmr::unordered_map<std::uint64_t, std::pmr::vector<const BoundaryTree::Node*>> device_checkpoints_by_epoch{
        &scratch};
    bool device_checkpoints_indexed = false;
    const auto device_continues_checkpoint = [&](const BoundaryTree::Node& checkpoint) {
        if (checkpoint.tail_tokens == 0 || checkpoint.latest_state_access_epoch == 0) {
            return false;
        }
        const BoundaryTree::Node* root = checkpoint.parent;
        if (root == nullptr) {
            return false;
        }
        if (!device_checkpoints_indexed) {
            // Most old tails have no Device checkpoint from the same request.
            // Index the Device State entries once, including pinned entries,
            // instead of walking all deeper token pages for every Host tail.
            std::pmr::unordered_set<const BoundaryTree::Node*> indexed{&scratch};
            for (std::size_t i = 0; i < groups_.size(); ++i) {
                if (!GroupIsState(i)) {
                    continue;
                }
                groups_[i].Index().ForEachEntry(pool_, [&](const CacheKey& key,
                                                           const PrefixCacheIndex::CachedBlockSnapshot&) {
                    const BoundaryTree::Node* node = boundary_tree_.Find(key.content_hash);
                    if (node != nullptr && node->has_checkpoint[BoundaryTree::kDevice] && indexed.insert(node).second) {
                        device_checkpoints_by_epoch[node->latest_state_access_epoch].push_back(node);
                    }
                });
            }
            device_checkpoints_indexed = true;
        }
        const auto found = device_checkpoints_by_epoch.find(checkpoint.latest_state_access_epoch);
        if (found == device_checkpoints_by_epoch.end()) {
            return false;
        }
        for (const BoundaryTree::Node* node : found->second) {
            if (node->boundary_tokens <= checkpoint.boundary_tokens) {
                continue;
            }
            for (const BoundaryTree::Node* ancestor = node; ancestor != nullptr; ancestor = ancestor->parent) {
                if (ancestor == root) {
                    return true;
                }
            }
        }
        return false;
    };

    // A candidate victim's node and its rank, the smallest first: by class
    // (VictimClass: a dead entry, a duplicate or a superseded checkpoint
    // boundary, Host-only cache), then by epoch -- a dead entry's and
    // Host-only checkpoint's newest State entry (a KV-only node's newest
    // Host entry), oldest first; a duplicate's or
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
    std::priority_queue<Candidate, std::pmr::vector<Candidate>, decltype(ranks_after)> candidates{
        ranks_after, std::pmr::vector<Candidate>{&scratch}};
    const auto add_candidate = [&](const BoundaryTree::Node& node) {
        // A Host-only checkpoint's rank and first location come from State.
        // Read its history only when its class needs it or it becomes a victim.
        std::size_t num_checkpoint_entries = 0;
        std::uint64_t state_epoch = 0;
        std::optional<CacheBlockLocation> first_entry_location;
        for (const HostEntry& entry : entries_for(node).states) {
            if (!taken.contains(entry.location)) {
                ++num_checkpoint_entries;
                state_epoch = std::max(state_epoch, entry.last_access_epoch);
                if (!first_entry_location) {
                    first_entry_location = entry.location;
                }
            }
        }
        if (!first_entry_location) {
            for (const HostEntry& entry : host_entries_of(node)) {
                if (!taken.contains(entry.location)) {
                    first_entry_location = entry.location;
                    break;
                }
            }
        }
        if (!first_entry_location) {
            return;
        }
        Candidate candidate{.candidate_node = &node,
                            .boundary_tokens = node.boundary_tokens,
                            .first_entry_location = *first_entry_location};
        // A Device checkpoint can still need this Host history, especially
        // a window absent on the Device, or other State groups split across
        // tiers. Incomplete Host State alone does not make these entries
        // dead; they keep their ordinary eviction ranking.
        const bool no_device_checkpoint = node.checkpoint_boundaries_in_subtree[BoundaryTree::kDevice] == 0;
        const bool dead =
            (node.tail_tokens > 0 && !tails_.Contains(*node.hash)) ||
            (no_device_checkpoint && ((num_checkpoint_entries > 0 && num_checkpoint_entries < num_state_groups_) ||
                                      (num_state_groups_ > 0 && num_remaining_host_checkpoint_boundaries(node) == 0)));
        const BoundaryTree::Node* page = node.tail_tokens > 0 ? node.parent : &node;
        const std::int32_t num_deeper_device_checkpoint_boundaries =
            page == nullptr ? 0
                            : page->checkpoint_boundaries_in_subtree[BoundaryTree::kDevice] -
                                  (page->has_checkpoint[BoundaryTree::kDevice] ? 1 : 0);
        if (dead) {
            candidate.victim_class = VictimClass::kDead;
            candidate.epoch_sort_key = *newest_epoch(*host_pool_, node, /*kv_only=*/false);
            candidate.reason = &HostEvictionStats::dead_entry;
        } else if ((num_state_groups_ == 0 || node.has_checkpoint[BoundaryTree::kDevice]) &&
                   DeviceBoundaryResidency(CacheKey{.content_hash = *node.hash}, node.tail_tokens) ==
                       BoundaryResidency::kComplete) {
            candidate.victim_class = VictimClass::kCoveredByDevice;
            candidate.epoch_sort_key = ~*newest_epoch(pool_, node, /*kv_only=*/false);
            candidate.reason = &HostEvictionStats::duplicate;
        } else if (const std::optional<std::uint64_t> superseding_epoch =
                       num_checkpoint_entries > 0 && num_deeper_device_checkpoint_boundaries >= 1 &&
                               device_continues_checkpoint(node)
                           ? newest_epoch(pool_, *page, /*kv_only=*/true)
                           : std::nullopt) {
            candidate.victim_class = VictimClass::kCoveredByDevice;
            candidate.epoch_sort_key = ~*superseding_epoch;
            candidate.reason = &HostEvictionStats::superseded;
        } else {
            candidate.victim_class = VictimClass::kHostOnly;
            if (num_checkpoint_entries > 0) {
                // A longer resume touches this page's KV without using its
                // checkpoint. Rank the State by its own accesses, so that KV
                // traversal does not keep an unused checkpoint young.
                candidate.epoch_sort_key = state_epoch;
            } else {
                candidate.epoch_sort_key = *newest_epoch(*host_pool_, node, /*kv_only=*/false);
            }
            candidate.reason = &HostEvictionStats::lru;
        }
        candidates.push(candidate);
    };

    // The candidate nodes: every Host checkpoint boundary, found through
    // the State groups' unpinned Host entries, and every Host leaf no Host
    // checkpoint boundary depends on (every Host leaf without State
    // groups). The Host's KV entries are never scanned.
    for (const BoundaryTree::Node* node : state_candidates) {
        add_candidate(*node);
    }
    for (const BoundaryTree::Node* leaf = boundary_tree_.FirstHostLeaf(); leaf != nullptr;
         leaf = leaf->next_host_leaf) {
        if (leaf->checkpoint_boundaries_in_subtree[BoundaryTree::kHost] == 0) {
            add_candidate(*leaf);
        }
    }

    // One joint fit over every segment: victims leave the shadow until
    // every row fits or no candidate is left.
    std::pmr::unordered_set<const BoundaryTree::Node*> parents_added_as_candidates{&scratch};
    // Per page above a taken victim, the first of its children the check
    // has not yet found empty of Host entries. The children before it stay
    // empty for the rest of the plan (walked_by_taken_victims and taken
    // only grow), so each check resumes there instead of walking a wide
    // fork's children again for every branch that leaves.
    std::pmr::unordered_map<const BoundaryTree::Node*, const BoundaryTree::Node*> next_child_to_check{&scratch};
    // The nodes a taken victim's walk covered: every Host page under them
    // is taken, so no later walk descends into them again (a chain taken
    // page by page would otherwise be walked once per page).
    std::pmr::unordered_set<const BoundaryTree::Node*> walked_by_taken_victims{&scratch};
    bool added_kv_duplicates = false;
    while (!shadow.Fits(need_by_bucket, no_dense_need)) {
        // Existing dead and Device-covered segments retain their priority.
        // Reclaim safe duplicates before the only copy of a checkpoint,
        // including when the Device KV is no longer held by a request.
        if (!added_kv_duplicates && (candidates.empty() || candidates.top().victim_class == VictimClass::kHostOnly)) {
            visitHostKvDuplicateCandidates([&](const HostKvDuplicateCandidate& duplicate) {
                if (taken.insert(duplicate.location).second) {
                    shadow.Remove(duplicate.group_id, duplicate.location);
                    Victim victim;
                    victim.members.emplace_back(duplicate.group_id, duplicate.location);
                    victim.reason = &HostEvictionStats::duplicate_kv;
                    victims.push_back(std::move(victim));
                }
                return !shadow.Fits(need_by_bucket, no_dense_need);
            });
            added_kv_duplicates = true;
            continue;
        }
        if (candidates.empty()) {
            break;
        }
        const Candidate candidate = candidates.top();
        candidates.pop();
        const BoundaryTree::Node& candidate_node = *candidate.candidate_node;
        std::vector<HostEntry> candidate_entries = untaken_entries_of(candidate_node);
        if (candidate_entries.empty()) {
            continue;  // taken with an earlier victim
        }
        Victim victim{.reason = candidate.reason};
        // The nodes the victim takes entries of, with those entries and
        // whether they include the node's Host page.
        struct TakenNode {
            const BoundaryTree::Node* node;
            std::vector<HostEntry> entries;
            bool takes_page;
        };
        std::vector<TakenNode> taken_nodes;
        std::vector<const BoundaryTree::Node*> walked;
        const auto num_candidate_checkpoint_entries = static_cast<std::size_t>(std::ranges::count_if(
            candidate_entries, [&](const HostEntry& entry) { return GroupIsState(entry.group_id); }));
        if (num_candidate_checkpoint_entries > 0 &&
            (num_remaining_host_checkpoint_boundaries(candidate_node) >= 2 ||
             candidate_node.checkpoint_boundaries_in_subtree[BoundaryTree::kDevice] > 0)) {
            // Other checkpoints on either tier can still read this history.
            // A Device copy at this same boundary also survives removal of
            // the Host State. Reclaim only the candidate's Host State.
            candidate_entries.resize(num_candidate_checkpoint_entries);
            taken_nodes.push_back(
                TakenNode{.node = &candidate_node, .entries = std::move(candidate_entries), .takes_page = false});
            victim.nearest_untaken_page = &candidate_node;
        } else {
            // The candidate node with the Host pages above it that serve no
            // other checkpoint boundary on either tier and are not in use -- none
            // without State groups, where the chain's Host pages leave leaf
            // by leaf -- and the Host pages under them. The climb keeps the
            // entries it looked up, the candidate node's first.
            const BoundaryTree::Node* top = &candidate_node;
            std::vector<std::pair<const BoundaryTree::Node*, std::vector<HostEntry>>> climbed;
            climbed.emplace_back(&candidate_node, std::move(candidate_entries));
            if (num_state_groups_ > 0) {
                while (top->parent != nullptr && top->parent->on_host && top->parent->pending_host_stores == 0 &&
                       num_remaining_host_checkpoint_boundaries(*top->parent) ==
                           num_remaining_host_checkpoint_boundaries(candidate_node) &&
                       top->parent->checkpoint_boundaries_in_subtree[BoundaryTree::kDevice] ==
                           candidate_node.checkpoint_boundaries_in_subtree[BoundaryTree::kDevice]) {
                    std::vector<HostEntry> entries = untaken_entries_of(*top->parent);
                    if (std::ranges::any_of(entries, &HostEntry::in_use)) {
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
                if (std::vector<HostEntry> entries =
                        next_climbed < climbed.size() && climbed[next_climbed].first == frame.node
                            ? std::move(climbed[next_climbed++].second)
                            : untaken_entries_of(*frame.node);
                    !entries.empty()) {
                    const bool takes_page = std::ranges::any_of(
                        entries, [&](const HostEntry& entry) { return !GroupIsState(entry.group_id); });
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
        // A KV load does not hold this boundary's State unless it actually
        // resumes there. If it blocks the segment, reclaim only the unused
        // checkpoint and leave every history block (and its pins) intact.
        if (std::ranges::any_of(taken_nodes, [](const TakenNode& taken_node) {
                return std::ranges::any_of(taken_node.entries, &HostEntry::in_use);
            })) {
            std::vector<HostEntry> checkpoint_entries = untaken_entries_of(candidate_node);
            checkpoint_entries.resize(num_candidate_checkpoint_entries);
            if (checkpoint_entries.empty() || std::ranges::any_of(checkpoint_entries, &HostEntry::in_use)) {
                continue;
            }
            taken_nodes.clear();
            taken_nodes.push_back(
                TakenNode{.node = &candidate_node, .entries = std::move(checkpoint_entries), .takes_page = false});
            walked.clear();
            victim.nearest_untaken_page = &candidate_node;
        }
        walked_by_taken_victims.insert(walked.begin(), walked.end());
        for (const TakenNode& taken_node : taken_nodes) {
            bool takes_checkpoints = false;
            for (const HostEntry& entry : taken_node.entries) {
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
        // Device dependencies keep its ordinary ranking; history can still
        // be selected separately when State alone cannot fund placement.
        const BoundaryTree::Node* nearest_untaken_page = victim.nearest_untaken_page;
        victims.push_back(std::move(victim));
        if (nearest_untaken_page != nullptr && nearest_untaken_page->on_host &&
            num_remaining_host_checkpoint_boundaries(*nearest_untaken_page) == 0 &&
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
