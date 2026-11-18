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

// CacheStore: the cache's structure. It owns the cache groups (each one's
// allocator, prefix index and matcher), their token geometry and match order,
// the boundary tree and checkpoint tree every group's index entries hold
// nodes of, the active-parent gauge, and the facts the group specs fix. It
// references the Device, Host and snapshot pools. The coordinator and its
// components work on it through this interface. Built once; never moves,
// since index entries point at its trees.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/cache_group.h"
#include "cache/coordinator/admission_types.h"
#include "cache/coordinator/group_geometry.h"
#include "cache/core/active_parents.h"
#include "cache/core/block_pool.h"
#include "cache/prefix/boundary_tree.h"
#include "cache/prefix/checkpoint_tree.h"
#include "utils.h"

namespace tokenspeed {

// What a cache mutation reports to its observer (KV cache events): a key
// stored in or removed from the Device index (kStored, kRemoved), or a change
// of a boundary's Host entries (kHostResidency, keyed by its content hash
// alone). The boundary of a Host change may be one no Device publication ever
// described: a Host block an L3 read brought in.
enum class CacheMutation { kStored, kRemoved, kHostResidency };
using CacheMutationSink = std::function<void(const CacheKey&, CacheMutation)>;

// The first block (page offset) of a tail a group needs: 0 for a group that
// keeps every row; for a sliding group the block holding the first row of the
// window [T - W + 1, T) when that lies inside the tail's page, below which
// the tail's blocks are holes.
inline std::int32_t FirstTailBlockNeeded(const CacheGroupSpec& spec, std::int32_t block_granularity,
                                         std::int32_t tail_tokens) {
    if (spec.attn_kind != AttnKind::kSlidingWindow) {
        return 0;
    }
    return std::max(0, tail_tokens - spec.sliding_window + 1) / block_granularity;
}

class CacheStore {
public:
    // Issued access epochs start at 1. A superseded input is re-registered
    // at this one (CacheCoordinator::MarkSupersededStateInput), below every other cache entry
    // and level with the blocks no entry holds, until an acquisition or a
    // publication touches it again.
    static constexpr std::uint64_t kSupersededAccessEpoch{0};
    // Superseded checkpoint demotion (SchedulerConfig::
    // demote_superseded_state_checkpoints, docs/design/scheduler.md §1.2).
    // A demoted entry carries this access epoch, which no access issues, so
    // the Device and a write-through Host evict it ahead of every dated entry.
    // A session's demotion (CacheCoordinator::DemoteSessions) and a superseded input
    // (CacheCoordinator::MarkSupersededStateInput) date entries to it as well; only a demotion
    // marks an entry demoted (PrefixCacheIndex::CachedBlockMetadata::demoted).
    static constexpr std::uint64_t kDemotedAccessEpoch{0};

    // Registers every group with each pool. groups[i] must have id i.
    // write_back_evictions configures the boundary tree to queue dead-node
    // candidates for the write-back policy.
    CacheStore(std::vector<CacheGroup> groups, std::int32_t prefix_granularity, BlockPool& pool, BlockPool* host_pool,
               BlockPool* snapshot_pool, bool write_back_evictions, std::int32_t checkpoint_retention_leaf_distance);
    CacheStore(const CacheStore&) = delete;
    CacheStore& operator=(const CacheStore&) = delete;
    CacheStore(CacheStore&&) = delete;
    CacheStore& operator=(CacheStore&&) = delete;

    std::vector<CacheGroup>& Groups() { return groups_; }
    const std::vector<CacheGroup>& Groups() const { return groups_; }
    // Per-group token -> page arithmetic, aligned with Groups().
    const std::vector<GroupGeometry>& Geometry() const { return metadata_.geometry; }
    // Closed groups first, so non-closed groups match against a settled
    // bound. Replayable groups are absent: they neither constrain nor claim
    // a hit.
    const std::vector<std::size_t>& MatchOrder() const { return metadata_.match_order; }
    BlockPool& DevicePool() { return pool_; }
    const BlockPool& DevicePool() const { return pool_; }
    // Null without a Host tier.
    BlockPool* HostPool() { return host_pool_; }
    const BlockPool* HostPool() const { return host_pool_; }
    // Request-private retraction images: never prefix-indexed, never
    // evicted. Null without one.
    BlockPool* SnapshotPool() { return snapshot_pool_; }
    const BlockPool* SnapshotPool() const { return snapshot_pool_; }
    // The Device parents request tables hold. Admit and Restore bind every
    // table they fill to it; the store never moves, so the pointer holds as
    // long as every bound table is destroyed before the store.
    ActiveParents& ActiveParentGauge() { return active_parents_; }
    const ActiveParents& ActiveParentGauge() const { return active_parents_; }
    // State checkpoint ancestry for retention ranking; the boundary tree
    // hands it every change of a checkpoint's residency.
    CheckpointTree& Checkpoints() { return checkpoints_; }
    const CheckpointTree& Checkpoints() const { return checkpoints_; }
    // The tree of the boundaries either tier indexes, with the published
    // tail hints and the dead-node candidates. The indexes' entries and the
    // recorded write-backs hold its nodes. No match reads it; evicting by
    // segments selects along it.
    BoundaryTree& Boundaries() { return boundary_tree_; }
    const BoundaryTree& Boundaries() const { return boundary_tree_; }

    std::int32_t PrefixGranularity() const noexcept { return prefix_granularity_; }
    std::int32_t CheckpointGranularity() const noexcept { return metadata_.checkpoint_granularity; }
    // The largest sliding window over the replayable groups; 0 when none is.
    std::int32_t ReplayWindowTokens() const noexcept { return metadata_.replay_window_tokens; }
    // Tokens re-fed ahead of a prefix hit so every replayable group resumes
    // there: the window before the hit, or the whole hit when shorter.
    std::int32_t ReplayTokens(std::int32_t hit_tokens) const {
        _assert(hit_tokens >= 0, "hit tokens must be non-negative");
        return std::min(metadata_.replay_window_tokens, hit_tokens);
    }
    // How many groups are State groups. With any, a boundary serves a resume
    // only with a checkpoint boundary at or under it.
    std::int32_t NumStateGroups() const noexcept { return metadata_.num_state_groups; }
    // The prefix-closed groups, in group order.
    const std::vector<std::size_t>& RoutingHistoryGroups() const { return metadata_.routing_history_groups; }
    // Some group is prefix-closed, and needs every page below a boundary: a
    // State or sliding group resumes past a page that holds nothing.
    bool HasPrefixClosedGroups() const noexcept { return !metadata_.routing_history_groups.empty(); }
    bool GroupIsState(std::size_t i) const { return groups_[i].Spec().attn_kind == AttnKind::kState; }
    // A replayable group (CacheGroupSpec::replayable) is regenerated by the
    // model from re-fed prompt tokens instead of being prefix-cached: it takes
    // no part in matching or publication on either tier, and Admit
    // materializes it from the replay window before a hit. Retention still
    // slides its request-private pages out.
    bool GroupIsReplayable(std::int32_t i) const { return groups_[static_cast<std::size_t>(i)].Spec().replayable; }

    // Group blocks one boundary's content hash expands to: the blocks of a
    // prefix page for a History group, one state for a State group.
    std::int32_t BlocksPerBoundary(std::int32_t i) const;
    // Group blocks a tail entry holds in group i: every block of the page
    // below T, the last one partial unless T is a multiple of the block span.
    std::int32_t TailBlocksFor(std::int32_t i, std::int32_t tail_tokens) const;
    std::int32_t FirstTailBlockNeeded(std::int32_t i, std::int32_t tail_tokens) const {
        const auto index = static_cast<std::size_t>(i);
        if (GroupIsState(index)) {
            return 0;
        }
        return tokenspeed::FirstTailBlockNeeded(groups_[index].Spec(), metadata_.geometry[index].BlockGranularity(),
                                                tail_tokens);
    }
    // A run of table slots, [first, past_end).
    struct BlockRange {
        std::int32_t first{0};
        std::int32_t past_end{0};
    };
    // The blocks of sliding group i a boundary at `boundary_tokens` needs
    // behind it, [first, past_end): those holding rows [T - W + 1, T), the
    // window the first query after T attends. An unaligned T can need one
    // block more than a page-aligned one, since the window's first row falls
    // inside a block instead of on its edge.
    BlockRange SlidingWindowBlocks(std::int32_t i, std::int32_t boundary_tokens) const;
    // The sink blocks of sliding group i a boundary needs beside `window`:
    // [0, min(sink blocks, window.first)), the ones the window does not hold
    // already. Empty for a group without a sink.
    BlockRange SinkBlocksBelow(std::int32_t i, const BlockRange& window) const;
    // The table slots of group i an owner copies in at a publication point
    // ending at `tokens` that it keeps writing past: exactly the slots that
    // point's publication registers under the tail's hash
    // (CompletedBlockRanges) -- the page's blocks below `tokens`, from the
    // window's first block on in a sliding group. None at a page boundary,
    // for a State group (which moves off its published slot by rotation) or
    // for a replayable one (which publishes nothing).
    BlockRange CopyInSlots(std::size_t i, std::int32_t tokens) const;
    // Visits every group's keys of a boundary: (group index, key), the key
    // reusing one storage whose group id and page offset change per visit.
    template <typename Visit>
    void ForEachBoundaryKey(CacheKey boundary, std::int32_t tail_tokens, Visit&& visit) const {
        ForEachBoundaryKeyIn(boundary, tail_tokens, std::views::iota(std::size_t{0}, groups_.size()),
                             std::forward<Visit>(visit));
    }

    // One entry a BoundaryTree node holds on the tier of `pool`: in use when
    // anything beside the index holds its block.
    struct NodeEntry {
        const BlockPool* pool{nullptr};
        std::uint32_t group_id{0};
        CacheBlockLocation location;
        bool in_use{false};
    };
    // The entries `pool`'s tier holds for a node's keys (ForEachBoundaryKey),
    // State checkpoints first.
    std::vector<NodeEntry> NodeEntries(const BlockPool& pool, const BoundaryTree::Node& node) const;
    // Where `pool`'s tier of group_index's index holds key, without keeping a
    // reference to it.
    std::optional<CacheBlockLocation> LocationOf(const BlockPool& pool, std::size_t group_index,
                                                 const CacheKey& key) const;
    // The keys of `content_hashes` in one group: every block each boundary's
    // hash expands to (BlocksPerBoundary), in order.
    std::vector<CacheKey> KeysForGroup(std::span<const std::string> content_hashes, std::uint32_t group_id) const;
    // Device residency of one scheduler-level prefix boundary: the child
    // entries every group holds for that content hash -- one per group block
    // within the prefix granularity for a chain hash (tail_tokens 0), one per
    // group block below T for a tail hash (tail_tokens = T mod P, which the
    // caller knows from its own publication record), one per state group. The
    // boundary key's group and offset are ignored; only its namespace and
    // content hash identify the boundary. This includes every participating
    // group, including State; replayable groups publish no entries.
    enum class BoundaryResidency { kNone, kPartial, kComplete };
    BoundaryResidency DeviceBoundaryResidency(const CacheKey& boundary, std::int32_t tail_tokens) const;
    // The pool of a tier.
    template <CacheTier Tier>
    const BlockPool& TierPool() const {
        if constexpr (Tier == CacheTier::kDevice) {
            return pool_;
        }
        FatalCheck(host_pool_ != nullptr, "Host cache tier is not configured");
        return *host_pool_;
    }
    // The keys of a tail ending tail_tokens into the page `hash` names, in
    // group i (TailBlocksFor of them).
    std::vector<CacheKey> TailKeysForGroup(std::size_t i, const ContentHash& hash, std::int32_t tail_tokens) const;
    std::vector<CacheKey> ExpandPrefixKeys(std::span<const std::string> content_hashes) const;
    // The Host tier's entry under `key`, held; empty without a Host tier.
    CacheBlockRef AcquireHostCachedBlock(const CacheKey& key) const;
    // One run of table slots a publication registers in a group, the key of
    // each (keys[j] is slot first_cache_block + j), and the boundary kind it
    // registers them under.
    struct PublishedRange {
        std::int32_t first_cache_block{0};
        std::vector<CacheKey> keys;
        CacheBoundaryKind boundary_kind{CacheBoundaryKind::kChunk};
        // A sliding group's sink blocks below a boundary's window: published
        // with an earlier boundary already, and registered with this one again
        // so they stay as recent as every boundary that needs them.
        bool sink{false};
    };
    // What publishing `completed` covers in one group -- every newly hashed
    // page plus the tail's blocks of a prefix-closed group, the blocks holding
    // the window behind the newest boundary of a sliding group, slot 1 of a
    // State group at a non-chunk point, nothing for a replayable one.
    std::vector<PublishedRange> CompletedBlockRanges(std::size_t group_index, const CompletedPages& completed) const;
    // Asserts a request's progress is one the store can publish.
    void ValidateProgress(const RequestProgress& progress) const;
    // The demands an admission with `prefix` plans: each replayable group's
    // dense growth made the sparse suffix it regenerates from the replay
    // window; empty when the probe hit nothing or no group replays.
    std::vector<GroupDemand> ReplayedDemands(const PrefixProbe& prefix, std::span<const GroupDemand> demands) const;

    // The observer of Device index stores and removals, and of the boundary
    // tree's Host entry changes; empty for none.
    const CacheMutationSink& MutationSink() const { return mutation_sink_; }
    void SetMutationSink(CacheMutationSink sink);

    // Drops every group, and with them the index entries holding the trees'
    // nodes, while whoever their entries point at still exists.
    void ClearGroups() { groups_.clear(); }

private:
    // Derived once from the group specs; publication never changes this geometry.
    struct GroupMetadata {
        std::vector<GroupGeometry> geometry;
        std::vector<std::size_t> match_order;
        std::vector<std::size_t> routing_history_groups;
        std::int32_t checkpoint_granularity{1};
        std::int32_t replay_window_tokens{0};
        std::int32_t num_state_groups{0};
    };
    static GroupMetadata buildGroupMetadata(std::span<const CacheGroup> groups, std::int32_t prefix_granularity);

    // Reuse the caller's key storage and expand only the selected groups.
    template <typename GroupRange, typename Visit>
    void ForEachBoundaryKeyIn(CacheKey& key, std::int32_t tail_tokens, const GroupRange& group_indices,
                              Visit&& visit) const {
        _assert(tail_tokens >= 0 && tail_tokens < prefix_granularity_, "a tail lies strictly inside a prefix page");
        for (const std::size_t group_index : group_indices) {
            const auto i = static_cast<std::int32_t>(group_index);
            // A replayable group never holds an entry.
            if (GroupIsReplayable(i)) {
                continue;
            }
            const std::int32_t past_end = tail_tokens > 0 ? TailBlocksFor(i, tail_tokens) : BlocksPerBoundary(i);
            key.group_id = groups_[group_index].Id();
            for (key.page_offset = tail_tokens > 0 ? FirstTailBlockNeeded(i, tail_tokens) : 0;
                 key.page_offset < past_end; ++key.page_offset) {
                visit(group_index, std::as_const(key));
            }
        }
    }
    // The keys one boundary holds across the groups (ForEachBoundaryKey).
    std::int32_t NumBoundaryKeys(std::int32_t tail_tokens) const;
    // The key of table slot `block` of a History group at a publication point:
    // its page's chain hash below the full pages, the tail hash from there on.
    CacheKey historyBlockKey(std::size_t group_index, const CompletedPages& completed, std::int32_t block) const;

    BlockPool& pool_;
    BlockPool* const host_pool_;
    BlockPool* const snapshot_pool_;
    const std::int32_t prefix_granularity_;
    // Built before the groups and dropped after them: their index entries
    // hold the trees' nodes.
    CheckpointTree checkpoints_;
    BoundaryTree boundary_tree_;
    std::vector<CacheGroup> groups_;
    const GroupMetadata metadata_;
    ActiveParents active_parents_;
    CacheMutationSink mutation_sink_;
};

}  // namespace tokenspeed
