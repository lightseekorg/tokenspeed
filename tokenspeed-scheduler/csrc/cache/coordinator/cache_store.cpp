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

#include "cache/coordinator/cache_store.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "cache/prefix/prefix_matcher.h"
#include "utils.h"

namespace tokenspeed {

namespace {

BoundaryTree::Config boundaryTreeConfig(const std::vector<CacheGroup>& groups, std::int32_t prefix_granularity,
                                        bool write_back_evictions, bool has_host_tier) {
    BoundaryTree::Config config{
        .prefix_granularity = prefix_granularity,
        .closed_blocks_per_page = 0,
        .num_state_groups = 0,
        .has_prefix_closed_groups = false,
        .has_host_tier = has_host_tier,
        .queue_dead_candidates = write_back_evictions,
    };
    for (const CacheGroup& group : groups) {
        config.num_state_groups += group.Spec().attn_kind == AttnKind::kState ? 1 : 0;
        if (group.Matcher().IsPrefixClosed()) {
            _assert(group.Spec().block_granularity > 0, "a prefix-closed group spans tokens");
            config.has_prefix_closed_groups = true;
            config.closed_blocks_per_page += prefix_granularity / group.Spec().block_granularity;
        }
    }
    return config;
}

}  // namespace

CacheStore::GroupMetadata CacheStore::buildGroupMetadata(std::span<const CacheGroup> groups,
                                                         std::int32_t prefix_granularity) {
    _assert(prefix_granularity > 0, "coordinator needs positive prefix_granularity");
    GroupMetadata metadata;
    metadata.geometry.reserve(groups.size());
    for (std::size_t i = 0; i < groups.size(); ++i) {
        const CacheGroup& group = groups[i];
        const CacheGroupSpec& spec = group.Spec();
        _assert(group.Id() == static_cast<std::uint32_t>(i), "cache group id must equal its group index");
        if (spec.attn_kind == AttnKind::kState) {
            _assert(spec.block_granularity == 0, "a State group has no token span");
            ++metadata.num_state_groups;
        } else {
            _assert(spec.block_granularity > 0 && prefix_granularity % spec.block_granularity == 0,
                    "group block_granularity must be a positive divisor of the prefix granularity");
        }
        _assert(group.Allocator().CacheBlocksPerLcmBlock() == spec.cache_blocks_per_lcm_block,
                "group allocator packing must match its group spec");
        _assert(group.Allocator().ShardCount() == spec.shard_count,
                "group allocator shard count must match its group spec");
        _assert(spec.checkpoint_granularity >= 1, "checkpoint grain must be positive");
        if (spec.checkpoint_granularity > 1) {
            _assert(spec.attn_kind == AttnKind::kState, "only State groups constrain checkpoints");
            _assert(
                metadata.checkpoint_granularity == 1 || metadata.checkpoint_granularity == spec.checkpoint_granularity,
                "restricted State groups must use the same checkpoint grain");
            metadata.checkpoint_granularity = spec.checkpoint_granularity;
        }
        metadata.geometry.emplace_back(spec.block_granularity);
        if (spec.replayable) {
            _assert(spec.attn_kind == AttnKind::kSlidingWindow, "a replayable group must be a sliding window");
            metadata.replay_window_tokens = std::max(metadata.replay_window_tokens, spec.sliding_window);
        }
        if (group.Matcher().IsPrefixClosed()) {
            metadata.routing_history_groups.push_back(i);
            metadata.match_order.push_back(i);
        }
    }
    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (!groups[i].Matcher().IsPrefixClosed() && !groups[i].Spec().replayable) metadata.match_order.push_back(i);
    }
    return metadata;
}

CacheStore::CacheStore(std::vector<CacheGroup> groups, std::int32_t prefix_granularity, BlockPool& pool,
                       BlockPool* host_pool, BlockPool* snapshot_pool, bool write_back_evictions,
                       std::int32_t checkpoint_retention_leaf_distance)
    : pool_{pool},
      host_pool_{host_pool},
      snapshot_pool_{snapshot_pool},
      prefix_granularity_{prefix_granularity},
      checkpoints_{prefix_granularity, checkpoint_retention_leaf_distance},
      boundary_tree_{boundaryTreeConfig(groups, prefix_granularity, write_back_evictions, host_pool != nullptr),
                     &checkpoints_},
      groups_{std::move(groups)},
      metadata_{buildGroupMetadata(groups_, prefix_granularity)},
      active_parents_{pool} {
    // These bindings change pool/index state only after all group metadata is valid.
    _assert(!write_back_evictions || groups_.size() <= 64, "writing evictions back supports up to 64 cache groups");
    for (CacheGroup& group : groups_) {
        const CacheGroupSpec& spec = group.Spec();
        pool_.RegisterGroup(group.Id(), spec.cache_blocks_per_lcm_block, spec.shard_count);
        if (host_pool_ != nullptr) {
            host_pool_->RegisterGroup(group.Id(), host_pool_->Packing(group.Id()), spec.shard_count);
        }
        // Packing belongs to the pool; cyclic owner count stays the same across tiers.
        if (snapshot_pool_ != nullptr) {
            snapshot_pool_->RegisterGroup(group.Id(), snapshot_pool_->Packing(group.Id()), spec.shard_count);
        }
        if (group.Matcher().IsPrefixClosed() && host_pool_ != nullptr) {
            group.Index().TrackCommonEntries(pool_, *host_pool_);
        }
        group.Index().TrackBoundaries(boundary_tree_, pool_, host_pool_, spec.attn_kind == AttnKind::kState,
                                      group.Matcher().IsPrefixClosed());
        // Only admissions without State groups read them
        // (AdmissionPlanner::candidatesThroughParentFit), and a replayable
        // group publishes nothing for its parents to hold.
        if (metadata_.num_state_groups == 0 && !spec.replayable) {
            group.Index().TrackParentEntries(pool_);
        }
    }
}

std::int32_t CacheStore::BlocksPerBoundary(std::int32_t i) const {
    const auto index = static_cast<std::size_t>(i);
    return GroupIsState(index) ? 1 : prefix_granularity_ / metadata_.geometry[index].BlockGranularity();
}

std::int32_t CacheStore::TailBlocksFor(std::int32_t i, std::int32_t tail_tokens) const {
    _assert(tail_tokens > 0 && tail_tokens < prefix_granularity_, "a tail lies strictly inside a prefix page");
    if (GroupIsState(static_cast<std::size_t>(i))) {
        return 1;
    }
    const std::int32_t block_granularity = metadata_.geometry[static_cast<std::size_t>(i)].BlockGranularity();
    return (tail_tokens + block_granularity - 1) / block_granularity;
}

CacheStore::BlockRange CacheStore::SlidingWindowBlocks(std::int32_t i, std::int32_t boundary_tokens) const {
    const CacheGroupSpec& spec = groups_[static_cast<std::size_t>(i)].Spec();
    _assert(spec.attn_kind == AttnKind::kSlidingWindow, "only a sliding group has a window behind a boundary");
    const std::int32_t block_granularity = metadata_.geometry[static_cast<std::size_t>(i)].BlockGranularity();
    const std::int32_t past_end = CeilDiv(boundary_tokens, block_granularity);
    const std::int32_t first_row = std::max(0, boundary_tokens - spec.sliding_window + 1);
    if (first_row >= boundary_tokens) {
        return BlockRange{.first = past_end, .past_end = past_end};  // W == 1: nothing behind the query
    }
    return BlockRange{.first = first_row / block_granularity, .past_end = past_end};
}

CacheStore::BlockRange CacheStore::SinkBlocksBelow(std::int32_t i, const BlockRange& window) const {
    const std::int32_t sink_blocks = groups_[static_cast<std::size_t>(i)].Allocator().KeptHeadBlocks();
    return BlockRange{.first = 0, .past_end = std::min(sink_blocks, window.first)};
}

CacheStore::BlockRange CacheStore::CopyInSlots(std::size_t i, std::int32_t tokens) const {
    _assert(i < groups_.size() && tokens >= 0, "copy-in slots of a group at a non-negative frontier");
    const auto group = static_cast<std::int32_t>(i);
    const std::int32_t tail_tokens = tokens % prefix_granularity_;
    if (tail_tokens == 0 || GroupIsState(i) || GroupIsReplayable(group)) {
        return BlockRange{};
    }
    // The tail's page starts on a block edge: a page is whole blocks.
    const std::int32_t page_first = (tokens - tail_tokens) / metadata_.geometry[i].BlockGranularity();
    const std::int32_t past_end = page_first + TailBlocksFor(group, tail_tokens);
    if (groups_[i].Matcher().IsPrefixClosed()) {
        return BlockRange{.first = page_first, .past_end = past_end};
    }
    const BlockRange window = SlidingWindowBlocks(group, tokens);
    _assert(window.past_end == past_end, "a sliding window behind a tail ends with the tail's blocks");
    return BlockRange{.first = std::max(window.first, page_first), .past_end = past_end};
}

std::vector<CacheStore::NodeEntry> CacheStore::NodeEntries(const BlockPool& pool,
                                                           const BoundaryTree::Node& node) const {
    std::vector<NodeEntry> entries;
    entries.reserve(groups_.size());
    CacheKey boundary{.content_hash = node.hash};
    const auto append = [&](std::size_t i, const CacheKey& key) {
        if (const auto entry = groups_[i].Index().Inspect(pool, key)) {
            entries.push_back(NodeEntry{
                .pool = &pool, .group_id = groups_[i].Id(), .location = entry->location, .in_use = entry->pinned});
        }
    };
    const auto tier = &pool == host_pool_ ? BoundaryTree::kHost : BoundaryTree::kDevice;
    // Preserve group and block order within each kind without a temporary
    // partition buffer for every inspected boundary.
    for (const bool state : {true, false}) {
        // Partial checkpoints and pending Host stores keep the mark set;
        // only its absence proves that every State lookup would miss.
        if (state && !node.HasCheckpoint(tier)) {
            continue;
        }
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (GroupIsState(i) == state) {
                ForEachBoundaryKeyIn(boundary, node.tail_tokens, std::array{i}, append);
            }
        }
    }
    return entries;
}

CacheBlockRef CacheStore::AcquireHostCachedBlock(const CacheKey& key) const {
    if (host_pool_ == nullptr) {
        return {};
    }
    _assert(key.group_id < groups_.size(), "host cache key group id out of range");
    return groups_[key.group_id].Index().Find(*host_pool_, key);
}

std::vector<CacheStore::PublishedRange> CacheStore::CompletedBlockRanges(std::size_t group_index,
                                                                         const CompletedPages& completed) const {
    std::vector<PublishedRange> ranges;
    if (GroupIsReplayable(static_cast<std::int32_t>(group_index))) {
        return ranges;
    }
    const std::int32_t blocks_per_boundary = BlocksPerBoundary(static_cast<std::int32_t>(group_index));
    const std::int32_t hashed_prefix_pages = static_cast<std::int32_t>(completed.prefix_hashes.size());
    // A tail the owner keeps publishes no KV block: only the State groups
    // publish there, under its hash.
    const std::int32_t tail_blocks =
        completed.tail && completed.owner_tail != OwnerTail::kKept
            ? TailBlocksFor(static_cast<std::int32_t>(group_index), completed.tail->tail_tokens)
            : 0;
    const auto history_range = [&](std::int32_t first_block, std::int32_t past_end_block) {
        PublishedRange range{.first_cache_block = first_block, .boundary_kind = completed.boundary_kind};
        for (std::int32_t block = first_block; block < past_end_block; ++block) {
            range.keys.push_back(historyBlockKey(group_index, completed, block));
        }
        return range;
    };
    if (groups_[group_index].Matcher().IsPrefixClosed()) {
        // The new full pages and the tail's blocks are one contiguous run.
        const std::int32_t first_block = completed.first_new_prefix_page * blocks_per_boundary;
        const std::int32_t past_end_block = hashed_prefix_pages * blocks_per_boundary + tail_blocks;
        if (past_end_block > first_block) {
            ranges.push_back(history_range(first_block, past_end_block));
        }
        return ranges;
    }
    if (GroupIsState(group_index)) {
        // A natural checkpoint: the state the forward that ended at this
        // point wrote (slot 1, or slot 0 behind a padded first decode), keyed
        // like the point itself.
        if (!completed.publish_state || completed.prefix_hashes.empty() && !completed.tail) {
            return ranges;
        }
        const std::int32_t boundary_tokens =
            hashed_prefix_pages * prefix_granularity_ + (completed.tail ? completed.tail->tail_tokens : 0);
        if (boundary_tokens % groups_[group_index].Spec().checkpoint_granularity != 0) {
            return ranges;
        }
        const ContentHash hash = completed.tail ? completed.tail->hash : ContentHash{completed.prefix_hashes.back()};
        ranges.push_back(PublishedRange{
            .first_cache_block = completed.state_slot,
            .keys = {CacheKey{.group_id = groups_[group_index].Id(), .content_hash = hash, .page_offset = 0}},
            .boundary_kind = completed.boundary_kind,
        });
        return ranges;
    }
    // The in-forward checkpoint can precede the newest page boundary. Publish
    // its own window before the request reclaims the blocks behind its frontier.
    // That window is an endpoint's, like the checkpoint, even when a chunk end
    // publishes it.
    std::vector<std::int32_t> boundaries;
    if (completed.state_checkpoint_pages > 0) {
        boundaries.push_back(completed.state_checkpoint_pages * prefix_granularity_);
    }
    if (completed.first_new_prefix_page < hashed_prefix_pages) {
        const std::int32_t boundary_tokens = hashed_prefix_pages * prefix_granularity_;
        if (boundaries.empty() || boundaries.back() != boundary_tokens) {
            boundaries.push_back(boundary_tokens);
        }
    }
    if (tail_blocks > 0) {
        boundaries.push_back(hashed_prefix_pages * prefix_granularity_ + completed.tail->tail_tokens);
    }
    const std::int32_t state_checkpoint_tokens = completed.state_checkpoint_pages * prefix_granularity_;
    for (const std::int32_t boundary_tokens : boundaries) {
        const BlockRange window = SlidingWindowBlocks(static_cast<std::int32_t>(group_index), boundary_tokens);
        const BlockRange sink_blocks = SinkBlocksBelow(static_cast<std::int32_t>(group_index), window);
        if (window.first >= window.past_end && sink_blocks.first >= sink_blocks.past_end) {
            continue;
        }
        PublishedRange range = history_range(window.first, window.past_end);
        if (state_checkpoint_tokens > 0 && boundary_tokens == state_checkpoint_tokens) {
            range.boundary_kind = CacheBoundaryKind::kEndpoint;
        }
        if (sink_blocks.first < sink_blocks.past_end) {
            PublishedRange sink = history_range(sink_blocks.first, sink_blocks.past_end);
            sink.boundary_kind = range.boundary_kind;
            sink.sink = true;
            ranges.push_back(std::move(sink));
        }
        if (!range.keys.empty()) {
            ranges.push_back(std::move(range));
        }
    }
    return ranges;
}

CacheKey CacheStore::historyBlockKey(std::size_t group_index, const CompletedPages& completed,
                                     std::int32_t block) const {
    const std::int32_t blocks_per_boundary = BlocksPerBoundary(static_cast<std::int32_t>(group_index));
    const std::int32_t full_blocks = static_cast<std::int32_t>(completed.prefix_hashes.size()) * blocks_per_boundary;
    if (block < full_blocks) {
        return CacheKey{
            .group_id = groups_[group_index].Id(),
            .content_hash = ContentHash{completed.prefix_hashes[static_cast<std::size_t>(block / blocks_per_boundary)]},
            .page_offset = block % blocks_per_boundary};
    }
    _assert(completed.tail.has_value() && block - full_blocks < TailBlocksFor(static_cast<std::int32_t>(group_index),
                                                                              completed.tail->tail_tokens),
            "a published slot beyond the full pages belongs to the tail");
    return CacheKey{.group_id = groups_[group_index].Id(),
                    .content_hash = completed.tail->hash,
                    .page_offset = block - full_blocks};
}

std::optional<CacheBlockLocation> CacheStore::LocationOf(const BlockPool& pool, std::size_t group_index,
                                                         const CacheKey& key) const {
    // The reference from Find must not outlive this lookup: an eviction that
    // follows requires the index to be the block's only owner.
    const CacheBlockRef block = groups_[group_index].Index().Find(pool, key);
    return block ? std::optional<CacheBlockLocation>{block->Location()} : std::nullopt;
}

void CacheStore::ValidateProgress(const RequestProgress& progress) const {
    if (!progress.completed_pages) {
        return;
    }
    const CompletedPages& completed = *progress.completed_pages;
    const auto hashed_pages = static_cast<std::int32_t>(completed.prefix_hashes.size());
    _assert(completed.first_new_prefix_page >= 0 && completed.first_new_prefix_page <= hashed_pages,
            "completed page range lies inside the hashed prefix");
    _assert(completed.first_new_prefix_page < hashed_pages || completed.tail.has_value() ||
                (completed.publish_state && hashed_pages > 0) || completed.state_checkpoint_pages > 0,
            "a publication point without new pages publishes a tail or a state checkpoint");
    _assert(completed.state_checkpoint_pages >= 0 && completed.state_checkpoint_pages <= hashed_pages,
            "an in-forward State checkpoint lies inside the hashed prefix");
    if (completed.tail) {
        _assert(completed.tail->tail_tokens > 0 && completed.tail->tail_tokens < prefix_granularity_,
                "a tail lies strictly inside a prefix page");
        _assert(!completed.tail->hash.empty(), "a tail is published under its hash");
    }
    _assert(progress.num_computed_tokens.has_value(), "completed pages require the request's computed progress");
    if (completed.tail) {
        _assert(*progress.num_computed_tokens == hashed_pages * prefix_granularity_ + completed.tail->tail_tokens,
                "the tail ends at the request's computed frontier");
    }
}

std::vector<GroupDemand> CacheStore::ReplayedDemands(const PrefixProbe& prefix,
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
    if (metadata_.replay_window_tokens > 0 && hit_tokens > 0) {
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
                .first_block = replay_begin / metadata_.geometry[i].BlockGranularity(),
            };
        }
    }
    return replayed;
}

std::vector<CacheKey> CacheStore::KeysForGroup(std::span<const std::string> content_hashes,
                                               std::uint32_t group_id) const {
    _assert(group_id < groups_.size(), "cache key group id out of range");
    const std::int32_t blocks_per_boundary = BlocksPerBoundary(static_cast<std::int32_t>(group_id));
    _assert(content_hashes.size() <=
                std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(blocks_per_boundary),
            "expanded cache key count exceeds size_t range");
    std::vector<CacheKey> keys;
    keys.reserve(content_hashes.size() * static_cast<std::size_t>(blocks_per_boundary));
    for (const std::string& content_hash : content_hashes) {
        for (std::int32_t offset = 0; offset < blocks_per_boundary; ++offset) {
            keys.push_back(CacheKey{
                .group_id = group_id,
                .content_hash = ContentHash{content_hash},
                .page_offset = offset,
            });
        }
    }
    return keys;
}

std::int32_t CacheStore::NumBoundaryKeys(std::int32_t tail_tokens) const {
    _assert(tail_tokens >= 0 && tail_tokens < prefix_granularity_, "a tail lies strictly inside a prefix page");
    std::int32_t total = 0;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const auto group = static_cast<std::int32_t>(i);
        // A replayable group has no key here, so it never keeps a boundary
        // partial.
        if (GroupIsReplayable(group)) {
            continue;
        }
        total += tail_tokens > 0 ? TailBlocksFor(group, tail_tokens) - FirstTailBlockNeeded(group, tail_tokens)
                                 : BlocksPerBoundary(group);
    }
    return total;
}

CacheStore::BoundaryResidency CacheStore::DeviceBoundaryResidency(const CacheKey& boundary,
                                                                  std::int32_t tail_tokens) const {
    // Every Device entry under the hash is one of the boundary's keys, and
    // holds its node.
    _assert(boundary.namespace_id == kDefaultCacheNamespaceId, "a tracked boundary lies in the default namespace");
    const BoundaryTree::Node* node = boundary_tree_.Find(boundary.content_hash);
    const std::int32_t cached = node == nullptr ? 0 : node->Entries(BoundaryTree::kDevice);
    if (cached == 0) {
        return BoundaryResidency::kNone;
    }
    const std::int32_t total = NumBoundaryKeys(tail_tokens);
    _assert(cached <= total, "a boundary's Device entries lie among its keys");
    return cached == total ? BoundaryResidency::kComplete : BoundaryResidency::kPartial;
}

std::vector<CacheKey> CacheStore::TailKeysForGroup(std::size_t i, const ContentHash& hash,
                                                   std::int32_t tail_tokens) const {
    std::vector<CacheKey> keys;
    const std::int32_t blocks = TailBlocksFor(static_cast<std::int32_t>(i), tail_tokens);
    keys.reserve(static_cast<std::size_t>(blocks));
    for (std::int32_t offset = 0; offset < blocks; ++offset) {
        keys.push_back(CacheKey{.group_id = groups_[i].Id(), .content_hash = hash, .page_offset = offset});
    }
    return keys;
}

std::vector<CacheKey> CacheStore::ExpandPrefixKeys(std::span<const std::string> content_hashes) const {
    std::vector<CacheKey> keys;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        std::vector<CacheKey> group_keys = KeysForGroup(content_hashes, groups_[i].Id());
        keys.insert(keys.end(), std::make_move_iterator(group_keys.begin()), std::make_move_iterator(group_keys.end()));
    }
    return keys;
}

}  // namespace tokenspeed
