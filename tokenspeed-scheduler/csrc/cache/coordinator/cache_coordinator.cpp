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

#include "cache/coordinator/eviction_plan.h"

#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <iterator>
#include <utility>

#include "cache/coordinator/shadow_occupancy.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache/prefix/prefix_matcher.h"
#include "utils.h"

namespace tokenspeed {

SessionCacheOwner CacheCoordinator::sessionTag(const std::string& id) {
    auto& weak = session_tags_[id];
    if (auto tag = weak.lock(); tag && !tag->ended) {
        tag->demoted = false;
        return tag;
    }
    auto tag = std::make_shared<SessionCacheTag>();
    tag->id = id;
    weak = tag;
    if (++session_tags_since_sweep_ >= 4096) {
        session_tags_since_sweep_ = 0;
        std::erase_if(session_tags_, [](const auto& entry) { return entry.second.expired(); });
    }
    return tag;
}

CacheCoordinator::SessionAccess::SessionAccess(CacheCoordinator& coordinator, std::uint64_t access_epoch,
                                               SessionCacheOwner session)
    : coordinator_{&coordinator}, access_epoch_{access_epoch} {
    const bool inserted = coordinator.session_access_.emplace(access_epoch, std::move(session)).second;
    _assert(inserted, "an access epoch registers one session");
}

CacheCoordinator::SessionAccess::SessionAccess(SessionAccess&& other) noexcept
    : coordinator_{std::exchange(other.coordinator_, nullptr)}, access_epoch_{std::exchange(other.access_epoch_, 0)} {}

CacheCoordinator::SessionAccess& CacheCoordinator::SessionAccess::operator=(SessionAccess&& other) noexcept {
    if (this != &other) {
        release();
        coordinator_ = std::exchange(other.coordinator_, nullptr);
        access_epoch_ = std::exchange(other.access_epoch_, 0);
    }
    return *this;
}

CacheCoordinator::SessionAccess::~SessionAccess() {
    release();
}

void CacheCoordinator::SessionAccess::release() noexcept {
    if (CacheCoordinator* coordinator = std::exchange(coordinator_, nullptr)) {
        coordinator->session_access_.erase(access_epoch_);
    }
}

std::size_t CacheCoordinator::DemoteSession(const std::string& session_id, bool ended) {
    return DemoteSessions({session_id}, ended);
}

std::size_t CacheCoordinator::DemoteSessions(const std::vector<std::string>& ids, bool ended) {
    std::vector<SessionCacheOwner> owners;
    std::unordered_set<const SessionCacheTag*> tags;
    for (const auto& id : ids) {
        auto it = session_tags_.find(id);
        if (it == session_tags_.end()) continue;
        auto tag = it->second.lock();
        if (!tag) {
            session_tags_.erase(it);
            continue;
        }
        tag->demoted = true;
        tag->ended = tag->ended || ended;
        tags.insert(tag.get());
        owners.push_back(std::move(tag));
    }
    if (tags.empty()) return 0;
    std::size_t changed = 0;
    for (auto& group : groups_) {
        changed += group.Index().DemoteSessions(pool_, tags);
        if (host_pool_) changed += group.Index().DemoteSessions(*host_pool_, tags);
    }
    return changed;
}

std::vector<CacheCoordinator::StoreCandidate> CacheCoordinator::TakePendingStores() {
    auto pending = std::exchange(pending_stores_, {});
    std::erase_if(pending, [this](const StoreCandidate& candidate) {
        const auto& index = groups_[candidate.key.group_id].Index();
        auto location = locationOf(pool_, candidate.key.group_id, candidate.key);
        if (!location) return false;
        auto metadata = index.MetadataFor(pool_, *location);
        return metadata && metadata->session && metadata->session->ended;
    });
    return pending;
}

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

CacheCoordinator::CacheCoordinator(std::vector<CacheGroup> groups, std::int32_t prefix_granularity, BlockPool& pool,
                                   bool enable_l3_storage, BlockPool* host_pool, BlockPool* snapshot_pool,
                                   bool stream_device_cache_to_host, bool write_back_evictions,
                                   std::int64_t checkpoint_retention_requests,
                                   std::int32_t checkpoint_retention_leaf_distance)
    : groups_{std::move(groups)},
      pool_{pool},
      host_pool_{host_pool},
      snapshot_pool_{snapshot_pool},
      stream_device_cache_to_host_{stream_device_cache_to_host && host_pool != nullptr},
      write_back_evictions_{write_back_evictions},
      enable_l3_storage_{enable_l3_storage && host_pool != nullptr},
      prefix_granularity_{prefix_granularity},
      checkpoints_{prefix_granularity, checkpoint_retention_leaf_distance},
      boundary_tree_{boundaryTreeConfig(groups_, prefix_granularity, write_back_evictions, host_pool != nullptr),
                     &checkpoints_},
      checkpoint_retention_requests_{checkpoint_retention_requests} {
    _assert(prefix_granularity_ > 0, "coordinator needs positive prefix_granularity");
    _assert(checkpoint_retention_requests_ >= 0 && checkpoint_retention_requests_ <= kMaxCheckpointRetentionRequests,
            "checkpoint retention requests out of range");
    // Enough admissions to place the retention window and to tell the resume
    // gaps a choice of it reads.
    admission_epochs_.resize(
        std::max<std::size_t>(static_cast<std::size_t>(checkpoint_retention_requests_), kResumeGapAdmissions));
    checkpoint_resume_gaps_.resize(static_cast<std::size_t>(std::bit_width(admission_epochs_.size() - 1)) + 2);
    _assert(!write_back_evictions_ || host_pool_ != nullptr, "writing evictions back needs a Host pool");
    _assert(!(stream_device_cache_to_host_ && write_back_evictions_),
            "the Host tier is fed by publication or by eviction, not both");
    // BoundaryTree::Node::written_back_group_bits keeps one bit per group.
    _assert(!write_back_evictions_ || groups_.size() <= 64, "writing evictions back supports up to 64 cache groups");
    geometry_.reserve(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        _assert(groups_[i].Id() == static_cast<std::uint32_t>(i), "cache group id must equal its group index");
        const std::int32_t group_block_granularity = groups_[i].Spec().block_granularity;
        if (groups_[i].Spec().attn_kind == AttnKind::kState) {
            _assert(group_block_granularity == 0, "a State group has no token span");
            ++num_state_groups_;
        } else {
            _assert(group_block_granularity > 0 && prefix_granularity_ % group_block_granularity == 0,
                    "group block_granularity must be a positive divisor of the prefix granularity");
        }
        _assert(groups_[i].Allocator().CacheBlocksPerLcmBlock() == groups_[i].Spec().cache_blocks_per_lcm_block,
                "group allocator packing must match its group spec");
        _assert(groups_[i].Allocator().ShardCount() == groups_[i].Spec().shard_count,
                "group allocator shard count must match its group spec");
        const auto& spec = groups_[i].Spec();
        _assert(spec.checkpoint_granularity >= 1, "checkpoint grain must be positive");
        if (spec.checkpoint_granularity > 1) {
            _assert(spec.attn_kind == AttnKind::kState, "only State groups constrain checkpoints");
            _assert(checkpoint_granularity_ == 1 || checkpoint_granularity_ == spec.checkpoint_granularity,
                    "restricted State groups must use the same checkpoint grain");
            checkpoint_granularity_ = spec.checkpoint_granularity;
        }
        pool_.RegisterGroup(groups_[i].Id(), spec.cache_blocks_per_lcm_block, spec.shard_count);
        if (host_pool_ != nullptr) {
            host_pool_->RegisterGroup(groups_[i].Id(), host_pool_->Packing(groups_[i].Id()), spec.shard_count);
        }
        // Packing belongs to the pool. Cyclic owner count stays the same on
        // both tiers, so a snapshot keeps the Device block's owner bucket.
        if (snapshot_pool_ != nullptr) {
            snapshot_pool_->RegisterGroup(groups_[i].Id(), snapshot_pool_->Packing(groups_[i].Id()), spec.shard_count);
        }
        geometry_.emplace_back(group_block_granularity);
        if (spec.replayable) {
            _assert(spec.attn_kind == AttnKind::kSlidingWindow, "a replayable group must be a sliding window");
            replay_window_tokens_ = std::max(replay_window_tokens_, spec.sliding_window);
        }
        if (groups_[i].Matcher().IsPrefixClosed()) {
            match_order_.push_back(i);
            has_prefix_closed_groups_ = true;
            if (host_pool_ != nullptr) {
                groups_[i].Index().TrackCommonEntries(pool_, *host_pool_);
            }
        }
        groups_[i].Index().TrackBoundaries(boundary_tree_, pool_, host_pool_, GroupIsState(i),
                                           groups_[i].Matcher().IsPrefixClosed());
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!groups_[i].Matcher().IsPrefixClosed() && !GroupIsReplayable(static_cast<std::int32_t>(i))) {
            match_order_.push_back(i);
        }
    }
}

CacheCoordinator::~CacheCoordinator() {
    // The recorded write-backs and the index entries point at boundary_tree_
    // and at pending_eviction_keys_: drop them while both still exist.
    evicted_segments_.clear();
    groups_.clear();
}

bool CacheCoordinator::GroupIsState(std::size_t i) const {
    return groups_[i].Spec().attn_kind == AttnKind::kState;
}

std::int32_t CacheCoordinator::BlocksPerBoundary(std::int32_t i) const {
    const auto index = static_cast<std::size_t>(i);
    return GroupIsState(index) ? 1 : prefix_granularity_ / geometry_[index].BlockGranularity();
}

bool CacheCoordinator::deviceCacheIsClearable() const {
    for (const CacheGroup& group : groups_) {
        if (group.Index().NumPinnedEntries(pool_) != 0) {
            return false;
        }
    }
    return true;
}

bool CacheCoordinator::hostCacheIsClearable() const {
    if (host_pool_ == nullptr) {
        return true;
    }
    for (const CacheGroup& group : groups_) {
        if (group.Index().NumPinnedEntries(*host_pool_) != 0) {
            return false;
        }
    }
    return true;
}

bool CacheCoordinator::CacheIsClearable(bool include_host) const {
    if (!deviceCacheIsClearable()) {
        return false;
    }
    if (!include_host) {
        return true;
    }
    return hostCacheIsClearable();
}

bool CacheCoordinator::ClearDeviceCache() {
    // Every plan drains what its admissions and restores recorded; a clear
    // runs between plans and drops what it evicts.
    _assert(evicted_segments_.empty(), "a cache clear runs with no eviction left undrained");
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> cached_locations;
    for (const CacheGroup& group : groups_) {
        const PrefixCacheIndex& index = group.Index();
        const std::vector<PrefixCacheIndex::EvictionCandidate> group_entries = index.EvictableCandidates(pool_);
        if (static_cast<std::int32_t>(group_entries.size()) != index.NumEntries(pool_)) {
            return false;
        }
        for (const PrefixCacheIndex::EvictionCandidate& entry : group_entries) {
            cached_locations.emplace_back(group.Id(), entry.location);
        }
    }

    store_candidates_dropped_by_flush_ += pending_stores_.size();
    pending_stores_.clear();
    for (const auto& [group_id, location] : cached_locations) {
        // The cascade of an earlier eviction may already have taken this one.
        evictCachedBlock(group_id, location, &DeviceEvictionStats::flush, /*write_back_segment=*/nullptr);
        FatalCheck(!pool_.IsOccupied(location), "clearable Device cache entry disappeared");
    }
    // A Device-only clear leaves Host L2 untouched: the Host pages under a
    // Device prefix it emptied are not collected for it.
    boundary_tree_.ClearDeadCandidates();
    return true;
}

bool CacheCoordinator::ClearCache() {
    if (host_pool_ == nullptr) {
        return ClearDeviceCache();
    }

    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> host_locations;
    for (const CacheGroup& group : groups_) {
        const PrefixCacheIndex& index = group.Index();
        const std::vector<PrefixCacheIndex::EvictionCandidate> group_entries = index.EvictableCandidates(*host_pool_);
        if (static_cast<std::int32_t>(group_entries.size()) != index.NumEntries(*host_pool_)) {
            return false;
        }
        for (const PrefixCacheIndex::EvictionCandidate& entry : group_entries) {
            host_locations.emplace_back(group.Id(), entry.location);
        }
    }

    // ClearDeviceCache performs its complete pin check before mutation. Since
    // Host was checked above, a false return leaves both tiers unchanged.
    if (!ClearDeviceCache()) {
        return false;
    }
    for (const auto& [group_id, location] : host_locations) {
        _assert(groups_[group_id].Index().Evict(*host_pool_, location).has_value(),
                "clearable Host cache entry disappeared");
        ++host_evictions_.flush.blocks;
    }
    // With both tiers empty no node is left: no eviction store holds one, as
    // the scheduler refuses a clear while a store is in flight. What the Host
    // evictions queued for collection goes, and so do the tail hints.
    boundary_tree_.ClearAfterCacheClear();
    checkpoints_.Clear();
    return true;
}

std::int32_t CacheCoordinator::TailBlocksFor(std::int32_t i, std::int32_t tail_tokens) const {
    _assert(tail_tokens > 0 && tail_tokens < prefix_granularity_, "a tail lies strictly inside a prefix page");
    if (GroupIsState(static_cast<std::size_t>(i))) {
        return 1;
    }
    const std::int32_t block_granularity = geometry_[static_cast<std::size_t>(i)].BlockGranularity();
    return (tail_tokens + block_granularity - 1) / block_granularity;
}

CacheCoordinator::BlockRange CacheCoordinator::SlidingWindowBlocks(std::int32_t i, std::int32_t boundary_tokens) const {
    const CacheGroupSpec& spec = groups_[static_cast<std::size_t>(i)].Spec();
    _assert(spec.attn_kind == AttnKind::kSlidingWindow, "only a sliding group has a window behind a boundary");
    const std::int32_t block_granularity = geometry_[static_cast<std::size_t>(i)].BlockGranularity();
    const std::int32_t past_end = (boundary_tokens + block_granularity - 1) / block_granularity;
    const std::int32_t first_row = std::max(0, boundary_tokens - spec.sliding_window + 1);
    if (first_row >= boundary_tokens) {
        return BlockRange{.first = past_end, .past_end = past_end};  // W == 1: nothing behind the query
    }
    return BlockRange{.first = first_row / block_granularity, .past_end = past_end};
}

CacheCoordinator::BlockRange CacheCoordinator::CopyInSlots(std::size_t i, std::int32_t tokens) const {
    _assert(i < groups_.size() && tokens >= 0, "copy-in slots of a group at a non-negative frontier");
    const auto group = static_cast<std::int32_t>(i);
    const std::int32_t tail_tokens = tokens % prefix_granularity_;
    if (tail_tokens == 0 || GroupIsState(i) || GroupIsReplayable(group)) {
        return BlockRange{};
    }
    // The tail's page starts on a block edge: a page is whole blocks.
    const std::int32_t page_first = (tokens - tail_tokens) / geometry_[i].BlockGranularity();
    const std::int32_t past_end = page_first + TailBlocksFor(group, tail_tokens);
    if (groups_[i].Matcher().IsPrefixClosed()) {
        return BlockRange{.first = page_first, .past_end = past_end};
    }
    const BlockRange window = SlidingWindowBlocks(group, tokens);
    _assert(window.past_end == past_end, "a sliding window behind a tail ends with the tail's blocks");
    return BlockRange{.first = std::max(window.first, page_first), .past_end = past_end};
}

CacheCoordinator::BoundaryResidency CacheCoordinator::DeviceBoundaryResidency(const CacheKey& boundary,
                                                                              std::int32_t tail_tokens) const {
    // Every Device entry under the hash is one of the boundary's keys, and
    // holds its node.
    _assert(boundary.namespace_id == kDefaultCacheNamespaceId, "a tracked boundary lies in the default namespace");
    const BoundaryTree::Node* node = boundary_tree_.Find(boundary.content_hash);
    const std::int32_t cached = node == nullptr ? 0 : node->Entries(BoundaryTree::kDevice);
    if (cached == 0) {
        return BoundaryResidency::kNone;
    }
    const std::int32_t total = numBoundaryKeys(tail_tokens);
    _assert(cached <= total, "a boundary's Device entries lie among its keys");
    return cached == total ? BoundaryResidency::kComplete : BoundaryResidency::kPartial;
}

bool CacheCoordinator::HoldsStateCheckpoint(const ContentHash& content_hash) const {
    // A probe takes each State group's checkpoint from whichever tier holds
    // it (probeHostSuffix), so every State group must hold one on either.
    bool any_state_group = false;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!GroupIsState(i)) {
            continue;
        }
        const CacheKey key{.group_id = groups_[i].Id(), .content_hash = content_hash, .page_offset = 0};
        const PrefixCacheIndex& index = groups_[i].Index();
        if (!index.Contains(pool_, key) && (host_pool_ == nullptr || !index.Contains(*host_pool_, key))) {
            return false;
        }
        any_state_group = true;
    }
    return any_state_group;
}

std::int32_t CacheCoordinator::numBoundaryKeys(std::int32_t tail_tokens) const {
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

std::vector<CacheCoordinator::NodeEntry> CacheCoordinator::NodeEntries(const BlockPool& pool,
                                                                       const BoundaryTree::Node& node) const {
    std::vector<NodeEntry> entries;
    entries.reserve(groups_.size());
    CacheKey boundary{.content_hash = *node.hash};
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
                forEachBoundaryKey(boundary, node.tail_tokens, std::array{i}, append);
            }
        }
    }
    return entries;
}

std::vector<CacheKey> CacheCoordinator::keysForGroup(std::span<const std::string> content_hashes,
                                                     std::uint32_t group_id) const {
    _assert(group_id < groups_.size(), "cache key group id out of range");
    const std::int32_t pages_per_prefix_hash = BlocksPerBoundary(static_cast<std::int32_t>(group_id));
    _assert(content_hashes.size() <=
                std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(pages_per_prefix_hash),
            "expanded cache key count exceeds size_t range");
    std::vector<CacheKey> keys;
    keys.reserve(content_hashes.size() * static_cast<std::size_t>(pages_per_prefix_hash));
    for (const std::string& content_hash : content_hashes) {
        for (std::int32_t offset = 0; offset < pages_per_prefix_hash; ++offset) {
            keys.push_back(CacheKey{
                .group_id = group_id,
                .content_hash = content_hash,
                .page_offset = offset,
            });
        }
    }
    return keys;
}

namespace {

// Shared match skeleton over the token-spanned groups: one ordered sweep
// (closed groups first), then re-match any window group left above the
// settled bound -- with 2+ window groups a later group can shrink the bound
// UNDER an earlier one's boundary-dependent match. A re-matched group lands at
// or under the current bound and only a further bound drop can lift it back
// above, so re-matches are finite; the result is the greatest page boundary
// every group supports. Bounds align down to the shared prefix granularity;
// the candidate phase that follows refines them to token boundaries.
template <typename MatchGroup, typename ExtentTokens>
std::int32_t SweepThenConverge(std::span<const std::size_t> order, const std::vector<CacheGroup>& groups,
                               std::int32_t bound_tokens, std::int32_t align_tokens, const MatchGroup& match,
                               const ExtentTokens& extent) {
    const auto align_down = [align_tokens](std::int32_t tokens) { return tokens - tokens % align_tokens; };
    bound_tokens = align_down(bound_tokens);
    for (std::size_t i : order) {
        match(i, bound_tokens);
        bound_tokens = std::min(bound_tokens, align_down(extent(i)));
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i : order) {
            if (groups[i].Matcher().IsPrefixClosed() || extent(i) <= bound_tokens) {
                continue;
            }
            match(i, bound_tokens);
            bound_tokens = std::min(bound_tokens, align_down(extent(i)));
            changed = true;
        }
    }
    return bound_tokens;
}

}  // namespace

std::vector<std::vector<CacheKey>> CacheCoordinator::buildGroupKeys(std::span<const std::string> content_hashes) const {
    std::vector<std::vector<CacheKey>> group_keys(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        group_keys[i] = keysForGroup(content_hashes, groups_[i].Id());
    }
    return group_keys;
}

template <CacheTier Tier>
BlockPool& CacheCoordinator::tierPool() {
    if constexpr (Tier == CacheTier::kDevice) {
        return pool_;
    }
    FatalCheck(host_pool_ != nullptr, "Host cache tier is not configured");
    return *host_pool_;
}

template <CacheTier Tier>
const BlockPool& CacheCoordinator::tierPool() const {
    if constexpr (Tier == CacheTier::kDevice) {
        return pool_;
    }
    FatalCheck(host_pool_ != nullptr, "Host cache tier is not configured");
    return *host_pool_;
}

// The aligned phase of one tier probe, over the token-spanned groups of
// match_order: slots below floor_tokens are assumed valid in a lower tier;
// per_group blocks are relative to the floor, num_common_tokens is the
// absolute converged page boundary in TOKENS. num_prefix_pages bounds the walk
// in prefix pages; each group key vector is expanded to that group's
// block_granularity. State groups take no part in this phase; probeTier
// checks State and sliding-window requirements at each candidate boundary.
template <CacheTier Tier>
CacheCoordinator::PrefixProbe::Tier CacheCoordinator::probeTierWithKeys(
    std::span<const std::vector<CacheKey>> group_keys, std::span<const std::size_t> match_order,
    std::int32_t num_prefix_pages, std::int32_t floor_tokens,
    const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits) const {
    const BlockPool& pool = tierPool<Tier>();
    PrefixProbe::Tier out;
    out.num_common_tokens = floor_tokens;
    out.full_page_tokens = floor_tokens;
    out.kv_tokens = floor_tokens;
    out.per_group.resize(groups_.size());
    std::vector<std::size_t> spanned_order;
    for (const std::size_t i : match_order) {
        if (!GroupIsState(i)) {
            spanned_order.push_back(i);
        }
    }
    if (spanned_order.empty()) {
        return out;
    }
    const std::int32_t common_tokens = SweepThenConverge(
        spanned_order, groups_, num_prefix_pages * prefix_granularity_, prefix_granularity_,
        [&](std::size_t i, std::int32_t bound_tokens) {
            const std::int32_t group_block_granularity = geometry_[i].BlockGranularity();
            out.per_group[i] = groups_[i].Matcher().Probe(groups_[i].Index(), pool, group_keys[i],
                                                          floor_tokens / group_block_granularity,
                                                          bound_tokens / group_block_granularity, extra_hits);
        },
        [&](std::size_t i) {
            return floor_tokens +
                   static_cast<std::int32_t>(out.per_group[i].hits.size()) * geometry_[i].BlockGranularity();
        });

    // A matcher can find a q-aligned resume point above the final P-aligned
    // boundary. Only acquire the CacheBlocks covered by the shared boundary.
    for (const std::size_t i : spanned_order) {
        GroupPrefixProbe& probe = out.per_group[i];
        const std::int32_t covered_pages = (common_tokens - floor_tokens) / geometry_[i].BlockGranularity();
        if (static_cast<std::int32_t>(probe.hits.size()) > covered_pages) {
            probe.hits.resize(static_cast<std::size_t>(covered_pages));
        }
    }
    out.num_common_tokens = common_tokens;
    out.full_page_tokens = common_tokens;
    out.kv_tokens = common_tokens;
    return out;
}

template <CacheTier Tier>
CoordinatorMatch CacheCoordinator::acquireTierWithKeys(std::span<const std::vector<CacheKey>> group_keys,
                                                       std::int32_t floor_tokens, PrefixProbe::Tier&& probe,
                                                       std::uint64_t access_epoch) {
    BlockPool& pool = tierPool<Tier>();
    CoordinatorMatch out;
    out.num_common_tokens = probe.num_common_tokens;
    out.per_group.resize(groups_.size());
    bool seen_state_group = false;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        // A State group's hits count from page 0 (PrefixProbe::Tier).
        const std::int32_t floor_pages = GroupIsState(i) ? 0 : floor_tokens / geometry_[i].BlockGranularity();
        out.per_group[i] = groups_[i].Index().AcquireMatched(pool, group_keys[i], floor_pages, probe.per_group[i],
                                                             access_epoch, SessionForAccess(access_epoch));
        if constexpr (Tier == CacheTier::kDevice) {
            if (host_pool_ != nullptr) {
                groups_[i].Index().TouchExisting(*host_pool_, std::span{group_keys[i]}.subspan(floor_pages),
                                                 out.per_group[i].blocks, access_epoch, SessionForAccess(access_epoch));
            }
        }
        if (probe.tail && !probe.tail->group_keys[i].empty()) {
            // The tail's blocks follow the full pages in the same table.
            PrefixMatch tail_match = groups_[i].Index().AcquireMatched(pool, probe.tail->group_keys[i],
                                                                       /*begin_blocks=*/0, probe.tail->per_group[i],
                                                                       access_epoch, SessionForAccess(access_epoch));
            if constexpr (Tier == CacheTier::kDevice) {
                if (host_pool_ != nullptr) {
                    groups_[i].Index().TouchExisting(*host_pool_, probe.tail->group_keys[i], tail_match.blocks,
                                                     access_epoch, SessionForAccess(access_epoch));
                }
            }
            out.per_group[i].blocks.insert(out.per_group[i].blocks.end(),
                                           std::make_move_iterator(tail_match.blocks.begin()),
                                           std::make_move_iterator(tail_match.blocks.end()));
        }
        if (GroupIsState(i)) {
            const bool first_state_group = !seen_state_group;
            seen_state_group = true;
            for (const CacheBlockRef& block : out.per_group[i].blocks) {
                if (const std::optional<CacheKey> key = groups_[i].Index().KeyOf(pool, block)) {
                    if (BoundaryTree::Node* node = boundary_tree_.Find(key->content_hash)) {
                        if (first_state_group) {
                            out.resumed_checkpoint_last_use = node->latest_state_access_epoch;
                        }
                        node->latest_state_access_epoch = std::max(node->latest_state_access_epoch, access_epoch);
                    }
                }
            }
        }
    }
    return out;
}

std::vector<CacheKey> CacheCoordinator::tailKeysForGroup(std::size_t i, const ContentHash& hash,
                                                         std::int32_t tail_tokens) const {
    std::vector<CacheKey> keys;
    const std::int32_t blocks = TailBlocksFor(static_cast<std::int32_t>(i), tail_tokens);
    keys.reserve(static_cast<std::size_t>(blocks));
    for (std::int32_t offset = 0; offset < blocks; ++offset) {
        keys.push_back(CacheKey{.group_id = groups_[i].Id(), .content_hash = hash, .page_offset = offset});
    }
    return keys;
}

// Whether group i can resume at the boundary full_tokens + tail_tokens on
// this tier, and the hits over the boundary's keys: a closed group needs
// every block of the tail; a sliding group the blocks covering its lookback
// behind the boundary (tail blocks, and page blocks above floor_tokens --
// blocks below the floor are a lower tier's); a State group the one state
// published at the boundary, its page hit counted from page 0 whatever the
// floor (a Host checkpoint may sit on the Device's last full page).
template <CacheTier Tier>
bool CacheCoordinator::groupAcceptsBoundary(std::size_t i, std::span<const std::string> prefix_hashes,
                                            std::int32_t floor_tokens, std::int32_t full_tokens,
                                            std::int32_t tail_tokens, std::span<const CacheKey> tail_keys,
                                            GroupPrefixProbe& tail_hits, GroupPrefixProbe* page_hits,
                                            const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits) const {
    const BlockPool& pool = tierPool<Tier>();
    const CacheGroup& group = groups_[i];
    const PrefixCacheIndex& index = group.Index();
    const auto contains = [&](const CacheKey& key) {
        return index.Contains(pool, key) || (extra_hits != nullptr && extra_hits->contains(key));
    };
    tail_hits.hits.assign(tail_keys.size(), 0);
    if (GroupIsState(i)) {
        if ((full_tokens + tail_tokens) % group.Spec().checkpoint_granularity != 0) {
            return false;
        }
        if (tail_tokens > 0) {
            _assert(tail_keys.size() == 1, "a State group has one key per boundary");
            if (!contains(tail_keys[0])) {
                return false;
            }
            tail_hits.hits[0] = 1;
            return true;
        }
        // An aligned boundary is keyed by the chain hash of its last page.
        _assert(full_tokens > 0 && full_tokens % prefix_granularity_ == 0, "a page boundary past page 0");
        const auto page = static_cast<std::size_t>(full_tokens / prefix_granularity_) - 1;
        const CacheKey key{.group_id = group.Id(), .content_hash = prefix_hashes[page], .page_offset = 0};
        if (!contains(key)) {
            return false;
        }
        if (page_hits != nullptr) {
            page_hits->hits.assign(page + 1, 0);
            page_hits->hits.back() = 1;
        }
        return true;
    }
    if (group.Matcher().IsPrefixClosed()) {
        for (std::size_t j = 0; j < tail_keys.size(); ++j) {
            if (!contains(tail_keys[j])) {
                return false;
            }
            tail_hits.hits[j] = 1;
        }
        return true;
    }
    // A sliding window resumes at T when the blocks holding rows
    // [T - W + 1, T) are cached.
    const std::int32_t block_granularity = geometry_[i].BlockGranularity();
    const std::int32_t blocks_per_page = prefix_granularity_ / block_granularity;
    const std::int32_t floor_blocks = floor_tokens / block_granularity;
    const std::int32_t full_blocks = full_tokens / block_granularity;
    const BlockRange window = SlidingWindowBlocks(static_cast<std::int32_t>(i), full_tokens + tail_tokens);
    _assert(window.past_end == full_blocks + static_cast<std::int32_t>(tail_keys.size()),
            "the tail's keys end where the window does");
    if (page_hits != nullptr) {
        page_hits->hits.assign(static_cast<std::size_t>(std::max(full_blocks - floor_blocks, 0)), 0);
    }
    for (std::int32_t block = window.first; block < window.past_end; ++block) {
        if (block >= full_blocks) {
            const auto j = static_cast<std::size_t>(block - full_blocks);
            if (!contains(tail_keys[j])) {
                return false;
            }
            tail_hits.hits[j] = 1;
            continue;
        }
        if (block < floor_blocks) {
            continue;
        }
        const CacheKey page_key{.group_id = group.Id(),
                                .content_hash = prefix_hashes[static_cast<std::size_t>(block / blocks_per_page)],
                                .page_offset = block % blocks_per_page};
        if (!contains(page_key)) {
            return false;
        }
        if (page_hits != nullptr) {
            page_hits->hits[static_cast<std::size_t>(block - floor_blocks)] = 1;
        }
    }
    return true;
}

bool CacheCoordinator::stateGroupsAcceptBoundary(std::span<const std::size_t> match_order,
                                                 std::span<const std::string> prefix_hashes, std::int32_t full_tokens,
                                                 std::int32_t tail_tokens, const ContentHash& tail_hash,
                                                 bool allow_host,
                                                 const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits) const {
    CacheKey tail_key{.content_hash = tail_hash};
    GroupPrefixProbe tail_hits;
    for (const std::size_t i : match_order) {
        if (!GroupIsState(i)) {
            continue;
        }
        tail_key.group_id = groups_[i].Id();
        const std::span<const CacheKey> keys =
            tail_tokens > 0 ? std::span<const CacheKey>{&tail_key, 1} : std::span<const CacheKey>{};
        if (groupAcceptsBoundary<CacheTier::kDevice>(i, prefix_hashes, /*floor_tokens=*/0, full_tokens, tail_tokens,
                                                     keys, tail_hits, /*page_hits=*/nullptr,
                                                     /*extra_hits=*/nullptr)) {
            continue;
        }
        if (!allow_host ||
            !groupAcceptsBoundary<CacheTier::kHost>(i, prefix_hashes, /*floor_tokens=*/0, full_tokens, tail_tokens,
                                                    keys, tail_hits, /*page_hits=*/nullptr, extra_hits)) {
            return false;
        }
    }
    return true;
}

// The boundaries a tier may resume at, longest first: the tails published
// inside each page under its prior that the prompt's own tokens reach and the
// page boundaries themselves, from the page containing `from_tokens` down to
// the one containing the floor. The floor need not be aligned: the Host suffix
// is probed above the Device-only hit, which may be a tail with longer tails
// beside it in its page.
template <typename Visit>
void CacheCoordinator::forEachCandidateBoundary(const PromptPrefix& prompt, std::int32_t floor_tokens,
                                                std::int32_t from_tokens, const Visit& visit) const {
    for (std::int32_t page = from_tokens / prefix_granularity_; (page + 1) * prefix_granularity_ > floor_tokens;
         --page) {
        const std::int32_t page_tokens = page * prefix_granularity_;
        if (static_cast<std::size_t>(page) <= prompt.prefix_hashes.size() && !prompt.tokens.empty()) {
            const ContentHash prior =
                page == 0 ? ContentHash{} : prompt.prefix_hashes[static_cast<std::size_t>(page) - 1];
            const std::span<const TailTable::Tail> published = boundary_tree_.Tails().Tails(prior);
            std::vector<TailTable::Tail> candidates(published.begin(), published.end());
            if (prompt.tail && page == prompt.max_tokens / prefix_granularity_) {
                const auto& tail = *prompt.tail;
                if (std::ranges::none_of(candidates, [&](const auto& candidate) {
                        return candidate.tail_tokens == tail.tail_tokens && candidate.hash == tail.hash;
                    })) {
                    candidates.push_back({.tail_tokens = tail.tail_tokens, .hash = tail.hash});
                }
            }
            std::ranges::stable_sort(candidates, std::greater<>{}, &TailTable::Tail::tail_tokens);
            // The prompt's own hash at a tail length, reused while the
            // candidates (sorted by length) keep that length.
            ContentHash hash;
            std::int32_t hashed_tail_tokens = 0;
            for (const TailTable::Tail& tail : candidates) {
                const std::int32_t boundary = page_tokens + tail.tail_tokens;
                if (boundary > from_tokens || boundary > prompt.max_tokens || boundary <= floor_tokens ||
                    static_cast<std::size_t>(boundary) > prompt.tokens.size()) {
                    continue;
                }
                if (tail.tail_tokens != hashed_tail_tokens) {
                    hash = HashPrefixPage(prompt.tokens.subspan(static_cast<std::size_t>(page_tokens),
                                                                static_cast<std::size_t>(tail.tail_tokens)),
                                          prior);
                    hashed_tail_tokens = tail.tail_tokens;
                }
                if (hash != tail.hash) {
                    continue;  // published by a prompt whose tokens diverge inside the page
                }
                if (!visit(page_tokens, tail.tail_tokens, tail.hash)) {
                    return;
                }
            }
        }
        if (page_tokens > floor_tokens && page_tokens <= from_tokens) {
            if (!visit(page_tokens, 0, ContentHash{})) {
                return;
            }
        }
        if (page == 0) {
            break;
        }
    }
}

// The Device's probe: the aligned phase over the prefix-closed groups, then
// the candidate phase -- longest first, a tail inside the page after the
// closed coverage, the closed coverage itself, then every published tail and
// page boundary below -- until every matched group accepts one. A closed
// group's acceptance is monotone in T, so its page walk bounds the candidates;
// a sliding group's is not (it can hold the lookback behind one boundary and
// miss it behind the one below), so sliding and State groups are judged per
// candidate by groupAcceptsBoundary. Without State groups the first candidate
// the KV groups accept wins; with them the walk descends to the longest
// published state checkpoint. kv_tokens records the KV groups' own boundary
// for the junction cut.
CacheCoordinator::PrefixProbe::Tier CacheCoordinator::probeTier(const PromptPrefix& prompt,
                                                                std::span<const std::vector<CacheKey>> group_keys,
                                                                std::span<const std::size_t> match_order,
                                                                bool claim_partial) const {
    const std::int32_t num_prefix_pages =
        std::min(static_cast<std::int32_t>(prompt.prefix_hashes.size()), prompt.max_tokens / prefix_granularity_);
    std::vector<std::size_t> closed_order;
    for (const std::size_t i : match_order) {
        if (groups_[i].Matcher().IsPrefixClosed()) {
            closed_order.push_back(i);
        }
    }
    PrefixProbe::Tier aligned = probeTierWithKeys<CacheTier::kDevice>(group_keys, closed_order, num_prefix_pages,
                                                                      /*floor_tokens=*/0, /*extra_hits=*/nullptr);
    PrefixProbe::Tier out = aligned;
    out.closed_kv_tokens = aligned.num_common_tokens;
    out.num_common_tokens = 0;
    out.full_page_tokens = 0;
    out.kv_tokens = 0;
    // Without a KV group there is no KV boundary to record: a junction
    // exists only where cached KV outruns the state.
    bool kv_settled = std::ranges::none_of(match_order, [&](std::size_t i) { return !GroupIsState(i); });
    PrefixProbe::TailHit tail{.group_keys = std::vector<std::vector<CacheKey>>(groups_.size()),
                              .per_group = std::vector<GroupPrefixProbe>(groups_.size())};
    std::vector<GroupPrefixProbe> page_hits(groups_.size());
    const auto visit = [&](std::int32_t page_tokens, std::int32_t tail_tokens, const ContentHash& published_hash) {
        const std::int32_t boundary = page_tokens + tail_tokens;
        // Record the longest KV reach first: the junction still needs it
        // when State is missing. Shorter boundaries can reject State before
        // filling history hit arrays.
        if (kv_settled && !stateGroupsAcceptBoundary(match_order, prompt.prefix_hashes, page_tokens, tail_tokens,
                                                     published_hash, /*allow_host=*/false, /*extra_hits=*/nullptr)) {
            return true;
        }
        tail.tail_tokens = tail_tokens;
        tail.hash = published_hash;
        for (auto& keys : tail.group_keys) {
            keys.clear();
        }
        bool kv_accepts = true;
        bool state_accepts = true;
        for (const std::size_t i : match_order) {
            if (tail_tokens > 0) {
                tail.group_keys[i] = tailKeysForGroup(i, published_hash, tail_tokens);
            }
            const bool accepts = groupAcceptsBoundary<CacheTier::kDevice>(
                i, prompt.prefix_hashes, /*floor_tokens=*/0, page_tokens, tail_tokens, tail.group_keys[i],
                tail.per_group[i], &page_hits[i], /*extra_hits=*/nullptr);
            // Once KV reach is known, a rejected boundary contributes nothing.
            if (!accepts && kv_settled) {
                return true;
            }
            (GroupIsState(i) ? state_accepts : kv_accepts) = accepts && (GroupIsState(i) ? state_accepts : kv_accepts);
        }
        if (kv_accepts && !kv_settled) {
            out.kv_tokens = boundary;
            kv_settled = true;
        }
        if (!kv_accepts || !state_accepts) {
            return true;
        }
        out.num_common_tokens = boundary;
        out.full_page_tokens = page_tokens;
        for (const std::size_t i : match_order) {
            if (GroupIsState(i)) {
                out.per_group[i] = tail_tokens > 0 ? GroupPrefixProbe{} : std::move(page_hits[i]);
                continue;
            }
            if (groups_[i].Matcher().IsPrefixClosed()) {
                // The aligned walk covered every page below the boundary.
                const std::int32_t covered = page_tokens / geometry_[i].BlockGranularity();
                out.per_group[i].hits.resize(static_cast<std::size_t>(covered));
            } else {
                out.per_group[i] = std::move(page_hits[i]);
            }
        }
        if (tail_tokens > 0) {
            if (!claim_partial) {
                for (const std::size_t i : match_order) {
                    if (!GroupIsState(i) && tail_tokens % geometry_[i].BlockGranularity() != 0 &&
                        !tail.per_group[i].hits.empty()) {
                        tail.per_group[i].hits.back() = 0;
                    }
                }
            }
            out.tail = std::move(tail);
        }
        return false;
    };
    // Candidates start inside the page after the closed groups' coverage (a
    // tail there extends it) and descend from that coverage itself. With no
    // closed group every published boundary below the bound is a candidate.
    const std::int32_t from_tokens =
        closed_order.empty() ? prompt.max_tokens
                             : std::min(aligned.num_common_tokens + prefix_granularity_ - 1, prompt.max_tokens);
    forEachCandidateBoundary(prompt, /*floor_tokens=*/0, from_tokens, visit);
    if (out.num_common_tokens == 0) {
        // No boundary matched: no group claims anything on the Device.
        for (GroupPrefixProbe& hits : out.per_group) {
            hits.hits.clear();
        }
        out.tail.reset();
    }
    return out;
}

// The Host-suffix rule (a hit is a Device KV prefix plus a Host suffix). A
// candidate T = p + t (longest first, above above_tokens) splits at F
// (split_tokens): the Device's full KV pages up to L (device_kv_tokens), the
// Device's own KV coverage in full pages, lowered to the page of the first
// block a sliding group's window behind T needs below it and the Device lacks.
// The KV pages [F, p) come from the Host (the closed groups' Host run from L, C
// (host_kv_end_tokens), bounds the candidates), the KV tail whole from the
// Device when F == p and the Device holds it, else from the Host, and each
// State group's checkpoint at T from the Device, else from the Host. A gap
// stops a tier's run (L or C), so a hit is always one contiguous Device prefix
// plus one Host suffix. A Host-only probe from the Device's full pages is the
// special case F = D.full_page_tokens <= L: every boundary it would accept is
// accepted here.
void CacheCoordinator::probeHostSuffix(const PromptPrefix& prompt, std::span<const std::vector<CacheKey>> group_keys,
                                       std::span<const std::size_t> match_order, std::int32_t above_tokens,
                                       const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits,
                                       PrefixProbe::Tier& device, PrefixProbe::Tier& host) const {
    const std::int32_t num_prefix_pages =
        std::min(static_cast<std::int32_t>(prompt.prefix_hashes.size()), prompt.max_tokens / prefix_granularity_);
    std::vector<std::size_t> closed_order;
    for (const std::size_t i : match_order) {
        if (groups_[i].Matcher().IsPrefixClosed()) {
            closed_order.push_back(i);
        }
    }
    const bool has_kv = std::ranges::any_of(match_order, [&](std::size_t i) { return !GroupIsState(i); });
    // A Host window may complete Device Full KV even when no window on the
    // Device can resume there. Start from the closed groups' coverage; each
    // candidate below lowers the split for its missing Device window blocks.
    const std::int32_t closed_device_tokens = closed_order.empty() ? device.kv_tokens : device.closed_kv_tokens;
    const std::int32_t device_kv_tokens = std::min(closed_device_tokens - closed_device_tokens % prefix_granularity_,
                                                   num_prefix_pages * prefix_granularity_);
    const std::int32_t host_kv_end_tokens =
        probeTierWithKeys<CacheTier::kHost>(group_keys, closed_order, num_prefix_pages, device_kv_tokens, extra_hits)
            .num_common_tokens;
    host = PrefixProbe::Tier{};
    host.per_group.resize(groups_.size());
    bool kv_settled = !has_kv;
    const auto visit = [&](std::int32_t page_tokens, std::int32_t tail_tokens, const ContentHash& published_hash) {
        const std::int32_t boundary = page_tokens + tail_tokens;
        if (kv_settled && !stateGroupsAcceptBoundary(match_order, prompt.prefix_hashes, page_tokens, tail_tokens,
                                                     published_hash, /*allow_host=*/true, extra_hits)) {
            return true;
        }
        std::int32_t split_tokens = std::min(device_kv_tokens, page_tokens);
        for (const std::size_t i : match_order) {
            if (GroupIsState(i) || groups_[i].Matcher().IsPrefixClosed()) {
                continue;
            }
            const std::int32_t block_granularity = geometry_[i].BlockGranularity();
            const BlockRange window = SlidingWindowBlocks(static_cast<std::int32_t>(i), boundary);
            for (std::int32_t block = window.first; block < std::min(window.past_end, split_tokens / block_granularity);
                 ++block) {
                if (!groups_[i].Index().Contains(pool_, group_keys[i][static_cast<std::size_t>(block)])) {
                    split_tokens = block * block_granularity - block * block_granularity % prefix_granularity_;
                    break;
                }
            }
        }
        // C covers the closed pages from L; a split lowered below L needs
        // the Host's closed pages up to L as well.
        for (const std::size_t i : closed_order) {
            const std::int32_t block_granularity = geometry_[i].BlockGranularity();
            for (std::int32_t block = split_tokens / block_granularity;
                 block < std::min(device_kv_tokens, page_tokens) / block_granularity; ++block) {
                if (!PrefixKeyIsHit(groups_[i].Index(), *host_pool_, group_keys[i][static_cast<std::size_t>(block)],
                                    extra_hits)) {
                    return true;
                }
            }
        }
        const auto make_tail = [&] {
            return PrefixProbe::TailHit{.tail_tokens = tail_tokens,
                                        .hash = published_hash,
                                        .group_keys = std::vector<std::vector<CacheKey>>(groups_.size()),
                                        .per_group = std::vector<GroupPrefixProbe>(groups_.size())};
        };
        PrefixProbe::TailHit device_tail = make_tail();
        PrefixProbe::TailHit host_tail = make_tail();
        std::vector<GroupPrefixProbe> device_hits(groups_.size());
        std::vector<GroupPrefixProbe> host_hits(groups_.size());
        // The KV tail comes whole from one tier.
        bool kv_tail_on_device = has_kv && tail_tokens > 0 && split_tokens == page_tokens;
        for (const std::size_t i : match_order) {
            if (GroupIsState(i) || !kv_tail_on_device) {
                continue;
            }
            device_tail.group_keys[i] = tailKeysForGroup(i, published_hash, tail_tokens);
            kv_tail_on_device = groupAcceptsBoundary<CacheTier::kDevice>(
                i, prompt.prefix_hashes, split_tokens, page_tokens, tail_tokens, device_tail.group_keys[i],
                device_tail.per_group[i], /*page_hits=*/nullptr, /*extra_hits=*/nullptr);
        }
        bool kv_accepts = true;
        for (const std::size_t i : match_order) {
            if (GroupIsState(i)) {
                continue;
            }
            if (!kv_tail_on_device) {
                device_tail.group_keys[i].clear();
                device_tail.per_group[i] = GroupPrefixProbe{};
                if (tail_tokens > 0) {
                    host_tail.group_keys[i] = tailKeysForGroup(i, published_hash, tail_tokens);
                }
            }
            // With the tail on the Device the split is the page: a sliding
            // group needs no Host block either.
            if (!groupAcceptsBoundary<CacheTier::kHost>(i, prompt.prefix_hashes, split_tokens, page_tokens,
                                                        kv_tail_on_device ? 0 : tail_tokens, host_tail.group_keys[i],
                                                        host_tail.per_group[i], &host_hits[i], extra_hits)) {
                kv_accepts = false;
                break;
            }
            const std::int32_t block_granularity = geometry_[i].BlockGranularity();
            if (groups_[i].Matcher().IsPrefixClosed()) {
                device_hits[i].hits.assign(static_cast<std::size_t>(split_tokens / block_granularity), 1);
                host_hits[i].hits.assign(static_cast<std::size_t>((page_tokens - split_tokens) / block_granularity), 1);
                continue;
            }
            // The window blocks below the split, all on the Device by the split.
            const BlockRange window = SlidingWindowBlocks(static_cast<std::int32_t>(i), boundary);
            device_hits[i].hits.assign(static_cast<std::size_t>(split_tokens / block_granularity), 0);
            for (std::int32_t block = window.first; block < split_tokens / block_granularity; ++block) {
                device_hits[i].hits[static_cast<std::size_t>(block)] = 1;
            }
        }
        // The longest boundary whose KV the two tiers hold: the Host's KV
        // reach when the Host holds part of it, past the Device's own.
        if (kv_accepts && !kv_settled) {
            kv_settled = true;
            if (split_tokens < page_tokens || (tail_tokens > 0 && !kv_tail_on_device)) {
                host.kv_tokens = boundary;
            }
        }
        if (!kv_accepts) {
            return true;
        }
        for (const std::size_t i : match_order) {
            if (!GroupIsState(i)) {
                continue;
            }
            const std::vector<CacheKey> keys =
                tail_tokens > 0 ? tailKeysForGroup(i, published_hash, tail_tokens) : std::vector<CacheKey>{};
            if (groupAcceptsBoundary<CacheTier::kDevice>(i, prompt.prefix_hashes, split_tokens, page_tokens,
                                                         tail_tokens, keys, device_tail.per_group[i], &device_hits[i],
                                                         /*extra_hits=*/nullptr)) {
                device_tail.group_keys[i] = keys;
                continue;
            }
            device_tail.per_group[i] = GroupPrefixProbe{};
            if (!groupAcceptsBoundary<CacheTier::kHost>(i, prompt.prefix_hashes, split_tokens, page_tokens, tail_tokens,
                                                        keys, host_tail.per_group[i], &host_hits[i], extra_hits)) {
                return true;
            }
            host_tail.group_keys[i] = keys;
        }
        const auto holds_tail = [](const PrefixProbe::TailHit& tail) {
            return std::ranges::any_of(tail.group_keys, [](const auto& keys) { return !keys.empty(); });
        };
        device.num_common_tokens = kv_tail_on_device ? boundary : split_tokens;
        device.full_page_tokens = split_tokens;
        device.per_group = std::move(device_hits);
        device.tail = holds_tail(device_tail) ? std::optional{std::move(device_tail)} : std::nullopt;
        host.num_common_tokens = boundary;
        host.full_page_tokens = page_tokens;
        host.per_group = std::move(host_hits);
        host.tail = holds_tail(host_tail) ? std::optional{std::move(host_tail)} : std::nullopt;
        return false;
    };
    const std::int32_t from_tokens = closed_order.empty()
                                         ? prompt.max_tokens
                                         : std::min(host_kv_end_tokens + prefix_granularity_ - 1, prompt.max_tokens);
    forEachCandidateBoundary(prompt, above_tokens, from_tokens, visit);
}

CacheCoordinator::PrefixProbe CacheCoordinator::probePrefixWithOrder(const PromptPrefix& prompt,
                                                                     std::span<const std::size_t> match_order,
                                                                     bool claim_partial) const {
    _assert(prompt.prefix_hashes.size() <=
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / prefix_granularity_),
            "prefix length exceeds int32 token range");
    _assert(prompt.tokens.empty() || static_cast<std::int64_t>(prompt.tokens.size()) >= prompt.max_tokens,
            "the probe bound lies inside the supplied tokens");
    _assert(!prompt.tail || (!prompt.tokens.empty() && prompt.tail->tail_tokens > 0 &&
                             prompt.tail->tail_tokens == prompt.max_tokens % prefix_granularity_),
            "an explicit tail describes the probe bound inside its last page");
    PrefixProbe out;
    out.group_keys = prompt.prepared_keys ? prompt.prepared_keys : PreparePrefixKeys(prompt.prefix_hashes);
    out.device = probeTier(prompt, *out.group_keys, match_order, claim_partial);
    if (host_pool_ == nullptr) {
        return out;
    }
    const std::int32_t device_tokens = out.device.num_common_tokens;
    out.device_only_hit_tokens = device_tokens;
    probeHostSuffix(prompt, *out.group_keys, match_order, device_tokens, /*extra_hits=*/nullptr, out.device, out.host);
    if (out.host.num_common_tokens <= device_tokens) {
        // The Device alone resumes furthest. The Host tier keeps the Host's
        // KV reach: the junction still counts it, so a chunk cut there
        // publishes the checkpoint that KV lacks.
        const std::int32_t host_kv_tokens = out.host.kv_tokens;
        out.host = PrefixProbe::Tier{};
        out.host.per_group.resize(groups_.size());
        out.host.num_common_tokens = out.device.num_common_tokens;
        out.host.full_page_tokens = out.device.full_page_tokens;
        out.host.kv_tokens = host_kv_tokens;
    }
    return out;
}

PreparedPrefixKeys CacheCoordinator::PreparePrefixKeys(std::span<const std::string> content_hashes) const {
    return std::make_shared<const std::vector<std::vector<CacheKey>>>(buildGroupKeys(content_hashes));
}

CacheCoordinator::PrefixProbe CacheCoordinator::ProbePromptPrefix(const PromptPrefix& prompt) const {
    return probePrefixWithOrder(prompt, match_order_, /*claim_partial=*/true);
}

bool CacheCoordinator::EvictionStoresExtendPrefix(const PromptPrefix& prompt, const PrefixProbe& visible) const {
    const std::int32_t hit = std::max(visible.device.num_common_tokens, visible.host.num_common_tokens);
    if (host_pool_ == nullptr || pending_eviction_keys_.empty() || prompt.max_tokens <= hit) {
        return false;
    }
    PrefixProbe::Tier device = visible.device;
    PrefixProbe::Tier host;
    probeHostSuffix(prompt, *visible.group_keys, match_order_, hit, &pending_eviction_keys_, device, host);
    return host.num_common_tokens > hit;
}

std::vector<CacheCoordinator::PrefixBoundaryAvailability> CacheCoordinator::InspectPrefixAvailability(
    const PromptPrefix& prompt) const {
    _assert(groups_.size() <= 64, "prefix availability uses one bit per cache group");
    const auto keys = buildGroupKeys(prompt.prefix_hashes);
    const auto contains = [&](std::size_t i, const CacheKey& key, std::size_t tier) {
        const PrefixCacheIndex& index = groups_[i].Index();
        const bool device = tier != 1 && index.Contains(pool_, key);
        const bool host = tier != 0 && host_pool_ != nullptr && index.Contains(*host_pool_, key);
        return device || host;
    };
    // A closed group's first hole bounds every later boundary. Compute it
    // once; rescanning the entire prefix per boundary would be quadratic.
    std::vector<std::array<std::int32_t, 3>> first_missing(groups_.size());
    for (const std::size_t i : match_order_) {
        if (!groups_[i].Matcher().IsPrefixClosed()) {
            continue;
        }
        for (std::size_t tier = 0; tier < 3; ++tier) {
            std::int32_t block = 0;
            while (static_cast<std::size_t>(block) < keys[i].size() &&
                   contains(i, keys[i][static_cast<std::size_t>(block)], tier)) {
                ++block;
            }
            first_missing[i][tier] = block;
        }
    }
    std::vector<PrefixBoundaryAvailability> out;
    forEachCandidateBoundary(
        prompt, /*floor_tokens=*/0, prompt.max_tokens,
        [&](std::int32_t page_tokens, std::int32_t tail_tokens, const ContentHash& hash) {
            std::array<std::uint64_t, 3> missing{};
            const std::int32_t boundary = page_tokens + tail_tokens;
            for (const std::size_t i : match_order_) {
                const auto tail_keys =
                    tail_tokens > 0 ? tailKeysForGroup(i, hash, tail_tokens) : std::vector<CacheKey>{};
                for (std::size_t tier = 0; tier < 3; ++tier) {
                    bool available = true;
                    if (GroupIsState(i)) {
                        const CacheKey key =
                            tail_tokens > 0 ? tail_keys[0]
                                            : CacheKey{.group_id = groups_[i].Id(),
                                                       .content_hash = prompt.prefix_hashes[static_cast<std::size_t>(
                                                           page_tokens / prefix_granularity_ - 1)],
                                                       .page_offset = 0};
                        available = contains(i, key, tier);
                    } else {
                        const std::int32_t granularity = geometry_[i].BlockGranularity();
                        const std::int32_t full_blocks = page_tokens / granularity;
                        if (groups_[i].Matcher().IsPrefixClosed()) {
                            available = first_missing[i][tier] >= full_blocks;
                            for (const CacheKey& key : tail_keys) {
                                available = available && contains(i, key, tier);
                            }
                        } else {
                            const BlockRange window = SlidingWindowBlocks(static_cast<std::int32_t>(i), boundary);
                            for (std::int32_t block = window.first; available && block < window.past_end; ++block) {
                                const CacheKey& key = block < full_blocks
                                                          ? keys[i][static_cast<std::size_t>(block)]
                                                          : tail_keys[static_cast<std::size_t>(block - full_blocks)];
                                available = contains(i, key, tier);
                            }
                        }
                    }
                    if (!available) {
                        missing[tier] |= std::uint64_t{1} << i;
                    }
                }
            }
            out.push_back({.tokens = boundary,
                           .device_missing = missing[0],
                           .host_missing = missing[1],
                           .union_missing = missing[2]});
            return true;
        });
    return out;
}

CacheCoordinator::PrefixProbe CacheCoordinator::ProbePrefix(std::span<const std::string> content_hashes) const {
    return ProbePromptPrefix(PromptPrefix{
        .prefix_hashes = content_hashes,
        .tokens = {},
        .max_tokens = static_cast<std::int32_t>(content_hashes.size()) * prefix_granularity_,
    });
}

CacheCoordinator::PrefixProbe CacheCoordinator::ProbeDecodeDevicePrefix(const PromptPrefix& prompt,
                                                                        std::int32_t prompt_end_tokens) const {
    _assert(prompt.tokens.empty() || prompt_end_tokens <= static_cast<std::int32_t>(prompt.tokens.size()),
            "the prompt end lies inside the supplied tokens");
    std::vector<std::size_t> history_match_order;
    history_match_order.reserve(match_order_.size());
    for (std::size_t group_index : match_order_) {
        if (!GroupIsState(group_index)) {
            history_match_order.push_back(group_index);
        }
    }
    // The D role probes the Device alone: the peer's transfer, not a Host
    // load, fills what the Device lacks. The partial block of a KV tail is
    // left a hole for the transfer to fill whole.
    PrefixProbe probe;
    probe.group_keys = prompt.prepared_keys ? prompt.prepared_keys : PreparePrefixKeys(prompt.prefix_hashes);
    probe.device = probeTier(prompt, *probe.group_keys, history_match_order, /*claim_partial=*/false);
    // The peer's final state supersedes any shorter checkpoint, so a State
    // group's checkpoint counts only at exactly the prompt end: then the
    // transfer omits the group and the D claims the checkpoint as slot 0.
    PrefixProbe::TailHit state_hit{.tail_tokens = prompt_end_tokens % prefix_granularity_,
                                   .group_keys = std::vector<std::vector<CacheKey>>(groups_.size()),
                                   .per_group = std::vector<GroupPrefixProbe>(groups_.size())};
    const std::int32_t page_tokens = prompt_end_tokens - state_hit.tail_tokens;
    const auto pages = static_cast<std::size_t>(page_tokens / prefix_granularity_);
    if (prompt.tokens.empty() || pages > prompt.prefix_hashes.size()) {
        return probe;  // nothing to match: the prefix cache is off, or the last page is not hashed
    }
    const ContentHash prior = pages == 0 ? ContentHash{} : prompt.prefix_hashes[pages - 1];
    if (state_hit.tail_tokens > 0) {
        state_hit.hash = HashPrefixPage(prompt.tokens.subspan(static_cast<std::size_t>(page_tokens),
                                                              static_cast<std::size_t>(state_hit.tail_tokens)),
                                        prior);
    } else if (pages == 0) {
        return probe;
    } else {
        state_hit.hash = prior;
    }
    bool every_state_group_holds = false;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!GroupIsState(i)) {
            continue;
        }
        state_hit.group_keys[i] = {
            CacheKey{.group_id = groups_[i].Id(), .content_hash = state_hit.hash, .page_offset = 0}};
        if (!groups_[i].Index().Contains(pool_, state_hit.group_keys[i].front())) {
            return probe;
        }
        state_hit.per_group[i].hits = {1};
        every_state_group_holds = true;
    }
    if (every_state_group_holds) {
        probe.state_at_prompt_end = std::move(state_hit);
    }
    return probe;
}

CacheCoordinator::AcquiredPrefix CacheCoordinator::acquirePrefix(PrefixProbe&& probe, std::uint64_t access_epoch) {
    AcquiredPrefix out;
    // The Host floor is the Device's full-page coverage: the Host's KV pages
    // start there (a Device KV tail means the Host holds none, only State
    // checkpoints at the tail).
    const std::int32_t device_full_tokens = probe.device.full_page_tokens;
    // A hit keeps its tail young under the per-prior cap.
    for (const PrefixProbe::Tier* tier : {&probe.device, &probe.host}) {
        if (tier->tail && tier->tail->tail_tokens > 0) {
            boundary_tree_.TouchTail(tier->tail->hash);
        }
    }
    out.device = acquireTierWithKeys<CacheTier::kDevice>(*probe.group_keys, /*floor_tokens=*/0, std::move(probe.device),
                                                         access_epoch);
    // The storage tier is never acquired: an admission loads Host entries
    // only; asynchronous storage publishes Host entries before admission.
    if (host_pool_ != nullptr && !probe.host.per_group.empty()) {
        out.host = acquireTierWithKeys<CacheTier::kHost>(*probe.group_keys, device_full_tokens, std::move(probe.host),
                                                         access_epoch);
    }
    if (const std::uint64_t last_use = out.device.resumed_checkpoint_last_use != 0
                                           ? out.device.resumed_checkpoint_last_use
                                           : out.host.resumed_checkpoint_last_use;
        last_use != 0) {
        recordResumeGap(last_use);
    }
    if (probe.state_at_prompt_end) {
        out.state_at_prompt_end.resize(groups_.size());
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (probe.state_at_prompt_end->group_keys[i].empty()) {
                continue;
            }
            PrefixMatch match =
                groups_[i].Index().AcquireMatched(pool_, probe.state_at_prompt_end->group_keys[i],
                                                  /*begin_blocks=*/0, probe.state_at_prompt_end->per_group[i],
                                                  access_epoch, SessionForAccess(access_epoch));
            if (host_pool_ != nullptr) {
                groups_[i].Index().TouchExisting(*host_pool_, probe.state_at_prompt_end->group_keys[i], match.blocks,
                                                 access_epoch, SessionForAccess(access_epoch));
            }
            out.state_at_prompt_end[i] = std::move(match.blocks.front());
            if (BoundaryTree::Node* node = boundary_tree_.Find(probe.state_at_prompt_end->hash)) {
                node->latest_state_access_epoch = std::max(node->latest_state_access_epoch, access_epoch);
            }
        }
    }
    return out;
}

// A full-pool census, and deliberately so: whether a parent is reclaimable
// depends on whether a request still owns one of its children, and a reference
// being dropped is not an event the pool or the index can observe. This is an
// observability call, not part of admission or placement; those answer their
// capacity questions from the pool's own counters.
std::int32_t CacheCoordinator::NumAvailableLcmBlocks() const {
    std::int32_t available = 0;
    for (std::int32_t parent_id = 1; parent_id <= pool_.NumLcmBlocks(); ++parent_id) {
        const std::optional<std::uint32_t> group_id = pool_.BoundGroup(parent_id);
        if (!group_id || groups_[*group_id].Index().ParentIsFullyEvictable(
                             pool_, parent_id, groups_[*group_id].Allocator().CacheBlocksPerLcmBlock())) {
            ++available;
        }
    }
    return available;
}

std::int32_t CacheCoordinator::NumActiveLcmBlocks(std::span<const std::span<const BlockTable>> request_tables) const {
    // Parent ids are dense in [1, NumLcmBlocks], so byte flags dedupe shared
    // prefixes without hashing every block reference of every live request.
    // The walk reads each table's page ids, kept contiguous beside its
    // references, rather than following every reference to its block: at long
    // contexts that pointer chase was most of the walk. A Device page id is
    // 1 + (parent - 1) * packing + slot (BlockPool::ResolveCacheBlockId), and 0
    // for a null slot.
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(pool_.NumLcmBlocks()) + 1, 0);
    std::vector<std::uint32_t> packing(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        packing[i] = static_cast<std::uint32_t>(pool_.Packing(groups_[i].Id()));
    }
    std::int32_t active = 0;
    for (std::span<const BlockTable> tables : request_tables) {
        _assert(tables.size() <= groups_.size(), "a request has one table per cache group");
        for (std::size_t i = 0; i < tables.size(); ++i) {
            const BlockTable& table = tables[i];
            for (const std::int32_t page_id : table.PageIds().subspan(table.ReclaimedPrefixBlocks())) {
                if (page_id == 0) {
                    continue;
                }
                const std::uint32_t parent = (static_cast<std::uint32_t>(page_id) - 1) / packing[i] + 1;
                std::uint8_t& parent_seen = seen[parent];
                if (!parent_seen) {
                    parent_seen = 1;
                    ++active;
                }
            }
        }
    }
    return active;
}

std::int32_t CacheCoordinator::GroupAvailablePages(std::int32_t group_index) const {
    _assert(group_index >= 0 && static_cast<std::size_t>(group_index) < groups_.size(),
            "cache group index out of range");
    const std::int32_t slots_per_parent =
        groups_[static_cast<std::size_t>(group_index)].Allocator().CacheBlocksPerLcmBlock();
    return pool_.NumEmptyLcmBlocks() * slots_per_parent + pool_.NumFreeSlots(static_cast<std::uint32_t>(group_index));
}

std::int32_t CacheCoordinator::NumNewlyReleasableLcmBlocks(std::span<const BlockTable> tables) const {
    _assert(tables.size() == groups_.size(), "release estimate requires one table per cache group");

    struct ReleasedRefs {
        const CacheBlockRef* block_ref{};
        std::uint32_t count{0};
    };
    std::vector<std::unordered_map<CacheBlockLocation, ReleasedRefs, CacheBlockLocationHash>> released_by_group(
        groups_.size());
    std::unordered_set<std::int32_t> referenced_parents;
    for (std::size_t group_id = 0; group_id < tables.size(); ++group_id) {
        for (const CacheBlockRef& block_ref : tables[group_id].Blocks()) {
            if (!block_ref) {
                continue;
            }
            const CacheBlockLocation location = block_ref->Location();
            ReleasedRefs& refs = released_by_group[group_id][location];
            refs.block_ref = &block_ref;
            ++refs.count;
            referenced_parents.insert(location.lcm_block_id);
        }
    }

    std::int32_t count = 0;
    for (std::int32_t parent_id : referenced_parents) {
        const std::optional<std::uint32_t> group_id = pool_.BoundGroup(parent_id);
        _assert(group_id.has_value(), "request table references an unbound LCM block");
        const PrefixCacheIndex& index = groups_[*group_id].Index();
        bool parent_becomes_reclaimable = true;
        for (CacheBlockLocation location : pool_.OccupiedLocations(parent_id)) {
            const auto released = released_by_group[*group_id].find(location);
            if (released == released_by_group[*group_id].end()) {
                parent_becomes_reclaimable = false;
                break;
            }
            const std::uint32_t owners = released->second.block_ref->use_count();
            _assert(owners >= released->second.count, "request-owned reference count exceeds total owners");
            const std::uint32_t retained_owners = owners - released->second.count;
            const std::uint32_t allowed_owners = index.Contains(pool_, location) ? 1U : 0U;
            if (retained_owners != allowed_owners) {
                parent_becomes_reclaimable = false;
                break;
            }
        }
        if (parent_becomes_reclaimable) {
            ++count;
        }
    }
    return count;
}

void CacheCoordinator::CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                                       std::uint64_t access_epoch, std::int32_t first_slot,
                                       CacheBoundaryKind boundary_kind) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    if (content_hashes.empty()) {
        return;  // hot decode rounds usually fill no page
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (GroupIsState(i)) {
            continue;  // a state checkpoint is published by its natural point, never by slot
        }
        std::vector<CacheKey> keys = keysForGroup(content_hashes, groups_[i].Id());
        const std::int32_t pages_per_prefix_hash = BlocksPerBoundary(static_cast<std::int32_t>(i));
        cacheFullBlocksForGroup<CacheTier::kDevice>(i, tables[i], keys, first_slot * pages_per_prefix_hash,
                                                    access_epoch, boundary_kind, /*stream_completed_to_host=*/false);
    }
}

void CacheCoordinator::QueueCachedBlocksForStore(std::span<const std::string> content_hashes) {
    if (host_pool_ == nullptr) {
        return;
    }
    for (const CacheGroup& group : groups_) {
        const PrefixCacheIndex& index = group.Index();
        // A tail hash holds fewer blocks than a page; Contains filters.
        for (CacheKey& key : keysForGroup(content_hashes, group.Id())) {
            if (!index.Contains(pool_, key)) {
                continue;
            }
            // Read only with demotion on: off, this stays one lookup.
            if (demotes_checkpoints_ && isDemotedEntry(index.Inspect(pool_, key)->metadata)) {
                continue;
            }
            pending_stores_.push_back(StoreCandidate{.key = std::move(key)});
        }
    }
}

void CacheCoordinator::validateProgress(const RequestProgress& progress) const {
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

void CacheCoordinator::validateCopyInBlocks(const RequestProgress& progress,
                                            std::span<const std::vector<CacheBlockRef>> copy_in_blocks) const {
    const bool copies_in = progress.completed_pages && progress.completed_pages->tail &&
                           progress.completed_pages->owner_tail == OwnerTail::kCopiedIn;
    if (!copies_in) {
        _assert(copy_in_blocks.empty(), "only an owner copying in past its tail spends copy-in blocks");
        return;
    }
    _assert(copy_in_blocks.size() == groups_.size(), "an owner copying in past its tail holds its copy-in blocks");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const BlockRange slots = CopyInSlots(i, *progress.num_computed_tokens);
        _assert(copy_in_blocks[i].size() == static_cast<std::size_t>(slots.past_end - slots.first) &&
                    std::ranges::all_of(copy_in_blocks[i], [](const CacheBlockRef& block) { return bool(block); }),
                "the owner holds a copy-in block for every copy-in slot");
    }
}

void CacheCoordinator::CacheCompletedBlocks(std::span<BlockTable> tables, const RequestProgress& progress,
                                            std::uint64_t access_epoch) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    _assert(progress.completed_pages.has_value(), "publication requires completed pages");
    _assert(progress.completed_pages->owner_tail == OwnerTail::kFinal,
            "a publication outside an admission publishes a final frontier");
    validateProgress(progress);
    const CompletedPages& completed = *progress.completed_pages;
    BoundaryTree::Hold publication = boundary_tree_.LinkPublication(completed);
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        cacheDeviceCompletedBlocksForGroup(i, tables[i], completed, access_epoch);
    }
    // A model without a prefix-closed group may register nothing under the
    // deepest boundary linked (a chunk end of State groups alone): it leaves
    // now.
    publication.reset();
    recordCheckpoints(completed, access_epoch);
    recordPublishedTail(completed);
}

template <CacheTier Tier>
void CacheCoordinator::cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table,
                                               std::span<const CacheKey> keys, std::int32_t first_cache_block,
                                               std::uint64_t access_epoch, CacheBoundaryKind boundary_kind,
                                               bool stream_completed_to_host) {
    if (GroupIsReplayable(static_cast<std::int32_t>(group_index))) {
        return;  // regenerated by the model, never published
    }
    // The window re-resolves its page ids once the index replaced deduped
    // blocks in it.
    BlockTable::PublishWindow window =
        groups_[group_index].Allocator().BlocksToPublish(table, first_cache_block, keys.size());
    cacheBlocksForGroup<Tier>(group_index, window.Blocks(), keys, first_cache_block, access_epoch, boundary_kind,
                              stream_completed_to_host);
}

template <CacheTier Tier>
void CacheCoordinator::cacheBlocksForGroup(std::size_t group_index, std::span<CacheBlockRef> blocks,
                                           std::span<const CacheKey> keys, std::int32_t first_logical_block,
                                           std::uint64_t access_epoch, CacheBoundaryKind boundary_kind,
                                           bool stream_completed_to_host) {
    std::vector<std::pair<CacheKey, CacheBlockRef>> newly_cached;
    const SessionCacheOwner session = SessionForAccess(access_epoch);
    const bool automatically_streams_to_host =
        stream_device_cache_to_host_ &&
        (groups_[group_index].Spec().attn_kind == AttnKind::kSlidingWindow || stream_completed_to_host);
    // Each entry it registers holds the tree node of its hash, which a
    // publication linked before registering.
    groups_[group_index].Index().RegisterFullBlocks(tierPool<Tier>(), blocks, keys, access_epoch, first_logical_block,
                                                    boundary_kind, Tier == CacheTier::kDevice ? &newly_cached : nullptr,
                                                    session);
    if constexpr (Tier == CacheTier::kHost) {
        return;
    }
    if (host_pool_ != nullptr) {
        groups_[group_index].Index().TouchExisting(*host_pool_, keys, blocks, access_epoch, session);
    }
    for (auto& [key, block_ref] : newly_cached) {
        if (cache_mutation_sink_) {
            cache_mutation_sink_(key, CacheMutation::kStored);
        }
        if (!automatically_streams_to_host || (session && session->ended)) {
            continue;
        }
        pending_stores_.push_back(StoreCandidate{
            .key = std::move(key),
        });
    }
    if (GroupIsState(group_index)) {
        for (std::size_t j = 0; j < keys.size(); ++j) {
            if (blocks[j]) {
                BoundaryTree::Node* node = boundary_tree_.Find(keys[j].content_hash);
                _assert(node != nullptr, "a published State checkpoint has a boundary");
                node->latest_state_access_epoch = std::max(node->latest_state_access_epoch, access_epoch);
            }
        }
    }
}

CacheBlockRef CacheCoordinator::AcquireDeviceCachedBlock(const CacheKey& key) const {
    if (key.group_id >= groups_.size()) {
        return {};
    }
    return groups_[key.group_id].Index().Find(pool_, key);
}

namespace {

class HostEvictionPlanner {
public:
    struct Member {
        std::uint32_t group_id;
        CacheBlockLocation location;
    };
    struct VictimSet {
        std::vector<Member> members;
    };

    HostEvictionPlanner(const CacheCoordinator& coordinator, const std::vector<CacheGroup>& groups,
                        const BlockPool& pool)
        : coordinator_{coordinator},
          groups_{groups},
          pool_{pool},
          plan_{pool, groups},
          no_dense_need_(groups.size()),
          no_pair_need_(groups.size()) {
        plan_.StartCandidates(&taken_, nullptr, coordinator.CheckpointAgingFor(CacheTier::kHost));
    }

    bool Fits(std::span<const std::vector<std::int64_t>> needed) const {
        return plan_.Fits(needed, no_dense_need_, no_pair_need_);
    }

    void Seed(std::uint32_t group_id, CacheBlockLocation location) {
        taken_.insert(location);
        plan_.Remove(group_id, location);
    }

    bool Fit(std::span<const std::vector<std::int64_t>> needed) {
        return plan_.SelectUntilFits(needed, no_dense_need_, no_pair_need_, [&] {
            while (const auto candidate = plan_.Next({})) {
                if (taken_.contains(candidate->location)) continue;
                VictimSet set;
                const auto take = [&](std::uint32_t group_id, CacheBlockLocation location) {
                    Seed(group_id, location);
                    set.members.push_back({group_id, location});
                };
                take(candidate->group_id, candidate->location);
                // Host State eviction takes the other unpinned State groups at this boundary.
                if (coordinator_.GroupIsState(candidate->group_id)) {
                    const auto key = groups_[candidate->group_id].Index().KeyOf(pool_, candidate->location);
                    coordinator_.ForEachBoundaryKey(
                        *key, /*tail_tokens=*/0, [&](std::size_t i, const CacheKey& sibling_key) {
                            if (!coordinator_.GroupIsState(i)) return;
                            const auto sibling = groups_[i].Index().Inspect(pool_, sibling_key);
                            if (sibling && !sibling->pinned && !taken_.contains(sibling->location)) {
                                take(groups_[i].Id(), sibling->location);
                            }
                        });
                }
                victims_.push_back(std::move(set));
                return true;
            }
            return false;
        });
    }

    void Minimize(std::span<const std::vector<std::int64_t>> needed) {
        plan_.Minimize(victims_, needed, no_dense_need_, no_pair_need_);
    }
    const std::vector<VictimSet>& Victims() const { return victims_; }
    std::vector<VictimSet> TakeVictims() { return std::move(victims_); }
    CacheCoordinator::HostAllocationStats Stats() const {
        return {.group_scans = plan_.Candidates().GroupScans(),
                .block_candidates = plan_.Candidates().ScannedCandidates(),
                .parent_candidates = plan_.TouchedParents()};
    }

private:
    const CacheCoordinator& coordinator_;
    const std::vector<CacheGroup>& groups_;
    const BlockPool& pool_;
    CacheEvictionPlan plan_;
    // Host rows land in their source's bucket: no dense blocks, no free-bucket pairs.
    std::vector<std::int64_t> no_dense_need_;
    std::vector<std::int64_t> no_pair_need_;
    CacheEvictionCandidates::LocationSet taken_;
    std::vector<VictimSet> victims_;
};

}  // namespace

CacheCoordinator::HostAllocationBatch CacheCoordinator::AcquireHostBlocks(std::span<const std::uint32_t> group_ids,
                                                                          std::span<const std::int32_t> buckets) {
    _assert(host_pool_ != nullptr, "AcquireHostBlocks requires a host pool");
    _assert(group_ids.size() == buckets.size(), "Host allocation needs one bucket per requested block");
    HostAllocationBatch batch;
    batch.blocks.resize(group_ids.size());
    batch.stats.requested = group_ids.size();
    _assert(group_ids.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "Host allocation batch request exceeds int32 range");
    const auto acquire_in_bucket = [&](std::uint32_t group_id, std::int32_t bucket) {
        const std::array<std::int32_t, 1> one{bucket};
        std::vector<CacheBlockRef> refs = host_pool_->AcquireBlocksInBuckets(group_id, one);
        return refs.empty() ? CacheBlockRef{} : std::move(refs.front());
    };
    std::vector<std::size_t> unresolved;
    for (std::size_t row = 0; row < group_ids.size(); ++row) {
        _assert(group_ids[row] < groups_.size(), "Host block group id out of range");
        _assert(buckets[row] >= 0 && buckets[row] < groups_[group_ids[row]].Allocator().ShardCount(),
                "Host block bucket is outside the group's shard count");
        batch.blocks[row] = acquire_in_bucket(group_ids[row], buckets[row]);
        if (!batch.blocks[row]) unresolved.push_back(row);
    }
    if (!unresolved.empty()) {
        std::vector<HostEvictionPlanner::VictimSet> victims;
        std::vector<std::size_t> accepted;
        {
            HostEvictionPlanner planner{*this, groups_, *host_pool_};
            std::vector<std::vector<std::int64_t>> needed(groups_.size());
            for (std::size_t i = 0; i < groups_.size(); ++i) needed[i].resize(groups_[i].Allocator().ShardCount());
            for (const std::size_t row : unresolved) {
                auto& need = needed[group_ids[row]][buckets[row]];
                ++need;
                if (planner.Fit(needed))
                    accepted.push_back(row);
                else
                    --need;
            }
            planner.Minimize(needed);
            victims = planner.TakeVictims();
            batch.stats = planner.Stats();
            batch.stats.requested = group_ids.size();
        }
        for (const auto& set : victims) {
            for (const auto& [group_id, location] : set.members) {
                if (groups_[group_id].Index().Contains(*host_pool_, location)) {
                    FatalCheck(evictHostCachedBlock(group_id, location, &HostEvictionStats::lru),
                               "a planned Host victim became pinned");
                }
            }
        }
        for (const std::size_t row : accepted) {
            batch.blocks[row] = acquire_in_bucket(group_ids[row], buckets[row]);
            FatalCheck(static_cast<bool>(batch.blocks[row]), "planned Host placement failed");
        }
    }
    batch.stats.allocated = static_cast<std::size_t>(
        std::ranges::count_if(batch.blocks, [](const CacheBlockRef& block) { return static_cast<bool>(block); }));
    batch.stats.unallocated = batch.stats.requested - batch.stats.allocated;
    return batch;
}

CacheBlockRef CacheCoordinator::AcquireHostBlock(std::uint32_t group_id, std::int32_t bucket) {
    const std::array groups{group_id};
    const std::array buckets{bucket};
    HostAllocationBatch batch = AcquireHostBlocks(groups, buckets);
    return batch.blocks.empty() ? CacheBlockRef{} : std::move(batch.blocks.front());
}

CacheCoordinator::HostPlacementPlan CacheCoordinator::planHostBoundaries(std::span<const std::uint32_t> group_ids,
                                                                         std::span<const std::int32_t> buckets,
                                                                         std::span<const HostSegmentRows> boundaries,
                                                                         bool place_by_segments) const {
    _assert(host_pool_ != nullptr, "Host placement requires a host pool");
    _assert(group_ids.size() == buckets.size(), "Host allocation needs one bucket per requested block");
    std::size_t next_row = 0;
    for (const HostSegmentRows& boundary : boundaries) {
        _assert(boundary.first_row == next_row && boundary.past_end_row >= boundary.first_row,
                "Host boundaries cover the rows in order");
        next_row = boundary.past_end_row;
    }
    _assert(next_row == group_ids.size(), "Host boundaries cover every row");
    std::vector<std::vector<std::int64_t>> needed(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) needed[i].resize(groups_[i].Allocator().ShardCount());
    for (std::size_t row = 0; row < group_ids.size(); ++row) {
        _assert(group_ids[row] < groups_.size(), "Host block group id out of range");
        _assert(buckets[row] >= 0 && buckets[row] < groups_[group_ids[row]].Allocator().ShardCount(),
                "Host block bucket is outside the group's shard count");
        ++needed[group_ids[row]][buckets[row]];
    }
    HostEvictionPlanner planner{*this, groups_, *host_pool_};
    HostPlacementPlan plan;
    if (planner.Fits(needed)) {
        plan.num_placed = boundaries.size();
        return plan;
    }
    if (place_by_segments) {
        plan = planHostSegments(group_ids, buckets, boundaries);
        if (plan.num_placed == boundaries.size()) return plan;
    }
    for (auto& group_need : needed) std::ranges::fill(group_need, 0);
    const std::size_t segment_rows = plan.num_placed == 0 ? 0 : boundaries[plan.num_placed - 1].past_end_row;
    for (std::size_t row = 0; row < segment_rows; ++row) ++needed[group_ids[row]][buckets[row]];
    for (const auto& victim : plan.victims) planner.Seed(victim.group_id, victim.location);

    for (const HostSegmentRows& boundary : boundaries.subspan(plan.num_placed)) {
        for (std::size_t row = boundary.first_row; row < boundary.past_end_row; ++row)
            ++needed[group_ids[row]][buckets[row]];
        if (!planner.Fit(needed)) {
            for (std::size_t row = boundary.first_row; row < boundary.past_end_row; ++row)
                --needed[group_ids[row]][buckets[row]];
            break;
        }
        ++plan.num_placed;
    }
    planner.Minimize(needed);
    for (const auto& set : planner.Victims()) {
        for (const auto& [group_id, location] : set.members) {
            plan.victims.push_back({.group_id = group_id, .location = location, .reason = &HostEvictionStats::lru});
        }
    }
    return plan;
}

std::size_t CacheCoordinator::NumHostBoundariesThatFit(std::span<const std::uint32_t> group_ids,
                                                       std::span<const std::int32_t> buckets,
                                                       std::span<const HostSegmentRows> boundaries) const {
    return planHostBoundaries(group_ids, buckets, boundaries, write_back_evictions_).num_placed;
}

CacheCoordinator::HostAllocationBatch CacheCoordinator::AcquireHostBoundaries(
    std::span<const std::uint32_t> group_ids, std::span<const std::int32_t> buckets,
    std::span<const HostSegmentRows> boundaries) {
    return acquireHostPlacement(planHostBoundaries(group_ids, buckets, boundaries, write_back_evictions_), group_ids,
                                buckets, boundaries);
}

CacheBlockRef CacheCoordinator::FindHostCachedBlock(const CacheKey& key) const {
    if (host_pool_ == nullptr) {
        return {};
    }
    _assert(key.group_id < groups_.size(), "host cache key group id out of range");
    return groups_[key.group_id].Index().Find(*host_pool_, key);
}

std::optional<CacheKey> CacheCoordinator::evictCachedBlockAlone(std::uint32_t group_id, CacheBlockLocation location,
                                                                DeviceEvictionCounts DeviceEvictionStats::*reason,
                                                                EvictedSegment* write_back_segment) {
    PrefixCacheIndex& index = groups_[group_id].Index();
    // Read while the entry is still there: the Host entry inherits it.
    const std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata = index.MetadataFor(pool_, location);
    // A demoted entry never rides an eviction write-back, whatever evicts it
    // -- an admission's or a restore's victim, a request's own released slot
    // or a stranded suffix: it is dropped, counted under what evicted it. A
    // demoted session's entry or a superseded input shares its epoch without
    // being demoted: the rule below takes it.
    const bool demoted = metadata && isDemotedEntry(*metadata);
    const bool drops_demoted = demoted && write_back_segment != nullptr;
    // A superseded input an admission, rotation or restore evicts leaves as the
    // reclaim of the request that superseded it, under either write policy,
    // and is never written back. It is never a stranded suffix: a tail at
    // epoch 0 is a candidate of its own before any page above it.
    if (demoted) {
        write_back_segment = nullptr;
    } else if (metadata && metadata->last_access_epoch == kSupersededAccessEpoch &&
               (reason == &DeviceEvictionStats::admission || reason == &DeviceEvictionStats::restore ||
                reason == &DeviceEvictionStats::request_reclaim)) {
        reason = &DeviceEvictionStats::request_reclaim;
        write_back_segment = nullptr;
    }
    const std::optional<CacheKey> key = index.UnpinnedKeyAt(pool_, location);
    if (!key) {
        return key;  // pinned or gone: nothing to record
    }
    if (drops_demoted) {
        ++demoted_entries_not_written_back_;
        // A request's own released slot the drain would have considered is
        // dropped like a dead one, and counted with them, so the drain's
        // outcomes still add up to the evictions it considered.
        if (reason == &DeviceEvictionStats::request_reclaim) {
            ++device_evictions_.request_reclaim_for_write_back;
        }
    }
    if (write_back_segment != nullptr) {
        if (reason == &DeviceEvictionStats::request_reclaim) {
            ++device_evictions_.request_reclaim_for_write_back;
        }
        BoundaryTree::Node* node = boundary_tree_.Find(key->content_hash);
        _assert(node != nullptr, "an indexed entry has a tree node");
        // Judged and recorded while the entry still holds the node, so the
        // checkpoints of a checkpoint boundary are recorded and hold it on the
        // Host before the pages above it are judged: no probe could resume
        // from a tail whose hint the cap dropped, nor, with State groups, from
        // a boundary with no checkpoint boundary at or under it on either tier
        // and none on its way there.
        const bool dead =
            (node->tail_tokens > 0 && !boundary_tree_.Tails().Contains(key->content_hash)) ||
            (num_state_groups_ > 0 && !node->HasCheckpointBoundaryInSubtree() && node->PendingHostStores() == 0);
        if (dead) {
            ++dead_entries_not_written_back_;
        } else {
            write_back_segment->members.push_back(
                EvictedBlock{.key = *key,
                             .device_location = location,
                             .metadata = *metadata,
                             .host_block = FindHostCachedBlock(*key),
                             .written_back_before = (node->written_back_group_bits >> key->group_id & 1U) != 0,
                             .pending = holdPendingStore(*node, *key, GroupIsState(group_id))});
            write_back_segment->newest_access_epoch =
                std::max(write_back_segment->newest_access_epoch, metadata->last_access_epoch);
        }
    }
    // The entry's hold on its node goes with it.
    FatalCheck(index.Evict(pool_, location).has_value(), "an unpinned cache entry stays evictable");
    if (cache_mutation_sink_) {
        cache_mutation_sink_(*key, CacheMutation::kRemoved);
    }
    DeviceEvictionCounts& evicted = device_evictions_.*reason;
    ++evicted.blocks;
    // Contains takes no reference, so it pins nothing on the Host tier.
    if (host_pool_ != nullptr && index.Contains(*host_pool_, *key)) {
        ++evicted.host_resident;
    }
    return key;
}

CacheCoordinator::PendingHostStore CacheCoordinator::holdPendingStore(BoundaryTree::Node& node, const CacheKey& key,
                                                                      bool checkpoint) {
    const auto [key_it, inserted] = pending_eviction_keys_.insert(key);
    if (!inserted) {
        ++pending_eviction_extra_refs_[key];
    }
    return PendingHostStore{*this, boundary_tree_.HoldPendingStore(node, checkpoint), *key_it};
}

void CacheCoordinator::releasePendingEvictionKey(const CacheKey& key) noexcept {
    // `key` may be the set's own element: find before erasing.
    const auto extra = pending_eviction_extra_refs_.find(key);
    if (extra == pending_eviction_extra_refs_.end()) {
        pending_eviction_keys_.erase(pending_eviction_keys_.find(key));
    } else if (--extra->second == 0) {
        pending_eviction_extra_refs_.erase(extra);
    }
}

CacheCoordinator::PendingHostStore::PendingHostStore(PendingHostStore&& other) noexcept
    : coordinator_{std::exchange(other.coordinator_, nullptr)},
      node_{std::move(other.node_)},
      key_{std::exchange(other.key_, nullptr)} {}

CacheCoordinator::PendingHostStore& CacheCoordinator::PendingHostStore::operator=(PendingHostStore&& other) noexcept {
    if (this != &other) {
        reset();
        coordinator_ = std::exchange(other.coordinator_, nullptr);
        node_ = std::move(other.node_);
        key_ = std::exchange(other.key_, nullptr);
    }
    return *this;
}

void CacheCoordinator::PendingHostStore::reset() noexcept {
    if (coordinator_ != nullptr) {
        coordinator_->releasePendingEvictionKey(*key_);
        coordinator_ = nullptr;
        key_ = nullptr;
    }
    node_.reset();
}

bool CacheCoordinator::evictCachedBlock(std::uint32_t group_id, CacheBlockLocation location,
                                        DeviceEvictionCounts DeviceEvictionStats::*reason,
                                        EvictedSegment* write_back_segment) {
    // Its own entry first: when that was the last holder of the hash's node,
    // the node leaves with its tail hint, and the cascade finds nothing.
    const std::optional<CacheKey> removed = evictCachedBlockAlone(group_id, location, reason, write_back_segment);
    if (!removed) {
        return false;
    }
    if (const std::optional<std::int32_t> tail_tokens = boundary_tree_.Tails().TailTokensOf(removed->content_hash)) {
        evictTailEntries(removed->content_hash, *tail_tokens, reason, write_back_segment);
    } else {
        // A state checkpoint is matched only with the KV below it and with
        // the checkpoints every other State group holds at its hash: a KV
        // block of its page gone, or one of those checkpoints, it is dead
        // weight and leaves too. A checkpoint takes no KV along: the page
        // still serves the boundaries past it.
        ForEachBoundaryKey(*removed, /*tail_tokens=*/0, [&](std::size_t i, const CacheKey& key) {
            if (!GroupIsState(i)) {
                return;
            }
            if (const std::optional<CacheBlockLocation> location = locationOf(pool_, i, key)) {
                evictCachedBlockAlone(groups_[i].Id(), *location, reason, write_back_segment);
            }
        });
    }
    return true;
}

bool CacheCoordinator::evictHostCachedBlock(std::uint32_t group_id, CacheBlockLocation location,
                                            HostEvictionCounts HostEvictionStats::*reason) {
    const auto evict_entry = [&](std::uint32_t entry_group, CacheBlockLocation entry_location) {
        std::optional<CacheKey> removed = groups_[entry_group].Index().Evict(*host_pool_, entry_location);
        if (!removed) {
            return removed;
        }
        HostEvictionCounts& evicted = host_evictions_.*reason;
        ++evicted.blocks;
        // Contains takes no reference, so it pins nothing on the Device tier.
        if (groups_[entry_group].Index().Contains(pool_, *removed)) {
            ++evicted.device_resident;
        }
        return removed;
    };
    const std::optional<CacheKey> removed = evict_entry(group_id, location);
    if (!removed) {
        return false;
    }
    if (GroupIsState(group_id)) {
        // A checkpoint resumes only with every State group's block at its
        // hash: the others leave with it unless pinned. A Host checkpoint
        // can resume on the Device's KV of its page, so no Host KV block
        // takes a checkpoint along.
        ForEachBoundaryKey(*removed, /*tail_tokens=*/0, [&](std::size_t i, const CacheKey& key) {
            if (!GroupIsState(i)) {
                return;
            }
            if (const std::optional<CacheBlockLocation> sibling = locationOf(*host_pool_, i, key)) {
                evict_entry(groups_[i].Id(), *sibling);
            }
        });
    }
    return true;
}

std::optional<CacheBlockLocation> CacheCoordinator::locationOf(const BlockPool& pool, std::size_t group_index,
                                                               const CacheKey& key) const {
    // The reference from Find must not outlive this lookup: an eviction that
    // follows requires the index to be the block's only owner.
    const CacheBlockRef block = groups_[group_index].Index().Find(pool, key);
    return block ? std::optional<CacheBlockLocation>{block->Location()} : std::nullopt;
}

void CacheCoordinator::evictTailEntries(const ContentHash& hash, std::int32_t tail_tokens,
                                        DeviceEvictionCounts DeviceEvictionStats::*reason,
                                        EvictedSegment* write_back_segment) {
    ForEachBoundaryKey(CacheKey{.content_hash = hash}, tail_tokens, [&](std::size_t i, const CacheKey& key) {
        // A companion still pinned by a request table stays indexed; it
        // ages out through the ordinary eviction order.
        if (const std::optional<CacheBlockLocation> location = locationOf(pool_, i, key)) {
            evictCachedBlockAlone(groups_[i].Id(), *location, reason, write_back_segment);
        }
    });
}

void CacheCoordinator::CollectDeadEntries() {
    // A node's entries, the Device's before the Host's.
    const auto entries_of = [&](const BoundaryTree::Node& node) {
        std::vector<NodeEntry> entries = NodeEntries(pool_, node);
        if (host_pool_ != nullptr) {
            std::ranges::copy(NodeEntries(*host_pool_, node), std::back_inserter(entries));
        }
        return entries;
    };
    // Judged on the state at collection, not at queueing. A node no
    // prefix-closed group holds on either tier is a gap, whatever State
    // entries or children it keeps. Nothing is dead while a store is
    // bringing it to the Host: its settlement queues it again.
    const auto dead = [&](const BoundaryTree::Node& node) {
        return node.PendingHostStores() == 0 &&
               ((num_state_groups_ > 0 && !node.HasCheckpointBoundaryInSubtree()) ||
                (node.tail_tokens > 0 && !boundary_tree_.Tails().Contains(*node.hash)) ||
                (has_prefix_closed_groups_ && std::ranges::none_of(entries_of(node), [&](const NodeEntry& entry) {
                     return groups_[entry.group_id].Matcher().IsPrefixClosed();
                 })));
    };
    // The roots left with an entry in use, each once, for the next call.
    std::vector<ContentHash> waiting;
    std::unordered_set<ContentHash> waiting_roots;
    // The evictions below may queue more; they are collected in this call too.
    while (const std::optional<ContentHash> candidate = boundary_tree_.PopDeadCandidate()) {
        const BoundaryTree::Node* top = boundary_tree_.Find(*candidate);
        if (top == nullptr || !dead(*top)) {
            continue;
        }
        // A live request's table holds its whole prefix, so the climb stops
        // below the pages it reads. It keeps the entries of each node it
        // passes.
        std::vector<std::pair<const BoundaryTree::Node*, std::vector<NodeEntry>>> climbed;
        while (top->parent != nullptr && dead(*top->parent)) {
            std::vector<NodeEntry> parent_entries = entries_of(*top->parent);
            if (std::ranges::any_of(parent_entries, &NodeEntry::in_use)) {
                break;
            }
            top = top->parent;
            climbed.emplace_back(top, std::move(parent_entries));
        }
        if (waiting_roots.contains(*top->hash)) {
            continue;
        }
        // Post-order over the subtree, a node after the nodes under it, so
        // each node leaves with its last entry; it completes the climbed
        // nodes in climb order.
        std::vector<NodeEntry> entries_to_evict;
        bool in_use = false;
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
                stack.push_back(Frame{.node = child, .next_child = child->first_child});
                continue;
            }
            const std::vector<NodeEntry> node_entries =
                next_climbed < climbed.size() && climbed[next_climbed].first == frame.node
                    ? std::move(climbed[next_climbed++].second)
                    : entries_of(*frame.node);
            for (const NodeEntry& entry : node_entries) {
                in_use = in_use || entry.in_use;
                if (!entry.in_use) {
                    entries_to_evict.push_back(entry);
                }
            }
            // A store still on its way lands here at its ACK, published when
            // its checkpoint boundary is held: the root waits for it like for a
            // pinned entry.
            in_use = in_use || frame.node->PendingHostStores() > 0;
            stack.pop_back();
        }
        // The evictions below may take the node away.
        ContentHash root = *top->hash;
        for (const NodeEntry& entry : entries_to_evict) {
            if (entry.pool == &pool_) {
                FatalCheck(
                    evictCachedBlockAlone(entry.group_id, entry.location, &DeviceEvictionStats::dead_entry_collection,
                                          /*write_back_segment=*/nullptr)
                        .has_value(),
                    "a dead Device entry became pinned");
            } else if (groups_[entry.group_id].Index().Contains(*host_pool_, entry.location)) {
                // A checkpoint takes the other State groups' at its hash along.
                FatalCheck(evictHostCachedBlock(entry.group_id, entry.location, &HostEvictionStats::dead_entry),
                           "a dead Host entry became pinned");
            }
        }
        if (in_use) {
            waiting_roots.insert(root);
            waiting.push_back(std::move(root));
        }
    }
    for (const ContentHash& root : waiting) {
        boundary_tree_.QueueDeadCandidate(root);
    }
}

std::vector<CacheCoordinator::EvictedSegment> CacheCoordinator::TakeEvictedSegments() {
    std::deque<EvictedSegment> recorded = std::exchange(evicted_segments_, {});
    // Union-find over the recording order: merged_into[s] leads to the
    // segment s was merged with.
    std::vector<std::size_t> merged_into(recorded.size());
    std::iota(merged_into.begin(), merged_into.end(), std::size_t{0});
    const auto root_of = [&](std::size_t s) {
        while (merged_into[s] != s) {
            s = merged_into[s] = merged_into[merged_into[s]];
        }
        return s;
    };
    // The segment that carries each tree node's KV page. A deeper checkpoint
    // boundary depends on the KV above it, never on a checkpoint above it.
    std::unordered_map<const BoundaryTree::Node*, std::size_t> segment_with_kv_page;
    // The first segment recording each key. A key evicted again after it was
    // published again is copied once, for both records, so their segments are
    // merged.
    std::unordered_map<CacheKey, std::size_t, CacheKeyHash> segment_with_key;
    std::vector<const BoundaryTree::Node*> segment_top_of(recorded.size(), nullptr);
    for (std::size_t s = 0; s < recorded.size(); ++s) {
        for (const EvictedBlock& member : recorded[s].members) {
            // The record holds the node.
            const BoundaryTree::Node* node = boundary_tree_.Find(member.key.content_hash);
            if (!GroupIsState(member.key.group_id)) {
                segment_with_kv_page.emplace(node, s);
            }
            if (const auto [it, inserted] = segment_with_key.try_emplace(member.key, s); !inserted) {
                merged_into[root_of(it->second)] = root_of(s);
            }
            if (segment_top_of[s] == nullptr || node->boundary_tokens < segment_top_of[s]->boundary_tokens) {
                segment_top_of[s] = node;
            }
        }
    }
    // A segment depends on another whose KV page lies at or above its
    // shallowest boundary; every boundary it holds is under that one.
    for (std::size_t s = 0; s < recorded.size(); ++s) {
        for (const BoundaryTree::Node* ancestor = segment_top_of[s]; ancestor != nullptr; ancestor = ancestor->parent) {
            if (const auto it = segment_with_kv_page.find(ancestor); it != segment_with_kv_page.end()) {
                merged_into[root_of(it->second)] = root_of(s);
            }
        }
    }
    // Members concatenated in recording order: a segment another one hangs
    // under was recorded after it (its climb took a page the earlier segment's
    // checkpoint boundary still depended on), so deeper checkpoint boundaries
    // still come first.
    std::vector<EvictedSegment> segments;
    std::vector<std::size_t> output_index_of_root(recorded.size(), recorded.size());
    for (std::size_t s = 0; s < recorded.size(); ++s) {
        std::size_t& output_index = output_index_of_root[root_of(s)];
        if (output_index == recorded.size()) {
            output_index = segments.size();
            segments.emplace_back();
        }
        EvictedSegment& segment = segments[output_index];
        segment.newest_access_epoch = std::max(segment.newest_access_epoch, recorded[s].newest_access_epoch);
        segment.members.insert(segment.members.end(), std::make_move_iterator(recorded[s].members.begin()),
                               std::make_move_iterator(recorded[s].members.end()));
    }
    return segments;
}

std::vector<HostPublication> CacheCoordinator::SettleEvictedSegment(EvictedSegment&& segment, bool placed_on_host) {
    std::vector<HostPublication> published;
    // In eviction order, each hold dropped as soon as its member is decided: a
    // checkpoint boundary is decided before the pages above it, which judge it
    // as the Host then holds it. Within a boundary the order does not matter:
    // its members are cancelled alike, and while one of its checkpoints is
    // still held or was published the boundary is a checkpoint boundary on the
    // Host.
    for (EvictedBlock& member : segment.members) {
        const ContentHash& hash = member.key.content_hash;
        _assert(static_cast<bool>(member.pending), "a recorded store holds its tree node until it settles");
        BoundaryTree::Node& node = member.pending.node();
        // The cap dropped the tail's hint while the copy was out, or no
        // checkpoint boundary at or under the boundary is left on either tier:
        // no probe could resume from it on the Host.
        const bool cancelled = (node.tail_tokens > 0 && !boundary_tree_.Tails().Contains(hash)) ||
                               (num_state_groups_ > 0 && !node.HasCheckpointBoundaryInSubtree());
        const bool publishes = placed_on_host && !cancelled;
        if (publishes) {
            _assert(static_cast<bool>(member.host_block), "a placed segment carries a Host block for every member");
            CacheHostBlock(member.host_block, member.key, member.metadata);
            node.written_back_group_bits |= std::uint64_t{1} << member.key.group_id;
            published.push_back(HostPublication{.key = member.key, .block = member.host_block});
        } else if (placed_on_host && member.host_block &&
                   groups_[member.key.group_id].Index().Contains(*host_pool_, member.host_block->Location())) {
            // A cancelled member the Host held already leaves it once nothing
            // else holds it; one in use waits for the collection.
            const CacheBlockLocation location = member.host_block->Location();
            member.host_block.reset();
            evictHostCachedBlock(member.key.group_id, location, &HostEvictionStats::dead_entry);
        }
        // The node may leave here; it is not read below.
        member.pending.reset();
        // What was not published is judged again by the next collection.
        if (!publishes) {
            boundary_tree_.QueueDeadCandidate(hash);
        }
    }
    return published;
}

void CacheCoordinator::MarkSupersededStateInput(const ContentHash& resumed_tail_hash, std::uint64_t access_epoch) {
    const std::optional<std::int32_t> tail_tokens = boundary_tree_.Tails().TailTokensOf(resumed_tail_hash);
    if (!tail_tokens) {
        return;
    }
    // A later request's hit still needs them: the whole tail stays unmarked,
    // so its checkpoints and blocks keep leaving together. The newest touch
    // decides, not the eviction epoch: an older request's publication of the
    // same tail since then dated them back, but the later hit stands.
    bool touched_since = false;
    ForEachBoundaryKey(
        CacheKey{.content_hash = resumed_tail_hash}, *tail_tokens, [&](std::size_t i, const CacheKey& key) {
            if (CacheBlockRef block = groups_[i].Index().Find(pool_, key)) {
                touched_since |=
                    groups_[i].Index().MetadataFor(pool_, block->Location())->max_access_epoch > access_epoch;
            }
        });
    if (touched_since) {
        return;
    }
    // Registering touches an existing entry with the epoch given (and keeps
    // its position and boundary kind): the tail's blocks and checkpoints
    // leave the eviction order's newest end for its oldest.
    ForEachBoundaryKey(CacheKey{.content_hash = resumed_tail_hash}, *tail_tokens,
                       [&](std::size_t i, const CacheKey& key) {
                           if (CacheBlockRef block = groups_[i].Index().Find(pool_, key)) {
                               groups_[i].Index().Register(pool_, block, key, kSupersededAccessEpoch,
                                                           /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                                                           /*newly_cached=*/nullptr, SessionForAccess(access_epoch));
                           }
                       });
}

static_assert(CacheCoordinator::kDemotedAccessEpoch == 0, "a demoted entry carries an epoch no access issues");

std::vector<CacheKey> CacheCoordinator::BoundaryCheckpointKeys(std::span<const std::string> prefix_hashes,
                                                               std::span<const std::int32_t> kept_boundaries) const {
    std::vector<CacheKey> keys;
    if (prefix_hashes.empty()) {
        return keys;
    }
    // What a natural checkpoint at the boundary publishes in the groups that
    // are not prefix-closed; the KV pages below it stay the path to the
    // deeper boundaries.
    const CompletedPages boundary{
        .prefix_hashes = prefix_hashes,
        .first_new_prefix_page = static_cast<std::int32_t>(prefix_hashes.size()) - 1,
        .publish_state = true,
    };
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (groups_[i].Matcher().IsPrefixClosed()) {
            continue;
        }
        const bool sliding = groups_[i].Spec().attn_kind == AttnKind::kSlidingWindow;
        for (PublishedRange& range : completedBlockRanges(i, boundary)) {
            for (std::size_t j = 0; j < range.keys.size(); ++j) {
                // A window block is keyed by its place on the chain, so a kept
                // boundary whose window covers it resumes with this very entry.
                const std::int32_t block = range.first_cache_block + static_cast<std::int32_t>(j);
                const auto behind = [&](std::int32_t kept_tokens) {
                    const BlockRange window = SlidingWindowBlocks(static_cast<std::int32_t>(i), kept_tokens);
                    return block >= window.first && block < window.past_end;
                };
                if (sliding && std::ranges::any_of(kept_boundaries, behind)) {
                    continue;
                }
                keys.push_back(std::move(range.keys[j]));
            }
        }
    }
    return keys;
}

std::int32_t CacheCoordinator::DemoteEntries(std::span<const CacheKey> keys) {
    std::int32_t demoted = 0;
    for (const CacheKey& key : keys) {
        PrefixCacheIndex& index = groups_[key.group_id].Index();
        demoted += index.Demote(pool_, key, kDemotedAccessEpoch, /*write_back_holds=*/0) ? 1 : 0;
        if (host_pool_ != nullptr) {
            demoted += index.Demote(*host_pool_, key, kDemotedAccessEpoch, /*write_back_holds=*/0) ? 1 : 0;
        }
    }
    return demoted;
}

bool CacheCoordinator::DemoteHostEntryHeldByWriteBack(const CacheKey& key, std::uint32_t write_back_holds) {
    return host_pool_ != nullptr &&
           groups_[key.group_id].Index().Demote(*host_pool_, key, kDemotedAccessEpoch, write_back_holds);
}

std::uint64_t CacheCoordinator::LatestStateAccessEpoch(const ContentHash& hash) const {
    const BoundaryTree::Node* node = boundary_tree_.Find(hash);
    return node == nullptr ? 0 : node->latest_state_access_epoch;
}

void CacheCoordinator::recordCheckpoints(const CompletedPages& completed, std::uint64_t access_epoch) {
    if (num_state_groups_ == 0 || completed.tokens.empty()) {
        return;
    }
    // The in-forward checkpoint first: it lies on the path to the point's.
    if (completed.state_checkpoint_pages > 0) {
        const auto pages = static_cast<std::size_t>(completed.state_checkpoint_pages);
        if (checkpoints_.Observe(completed.tokens.first(pages * static_cast<std::size_t>(prefix_granularity_)),
                                 completed.prefix_hashes.first(pages), std::nullopt, access_epoch)) {
            syncCheckpoint(completed.prefix_hashes[pages - 1]);
        }
    }
    if (completed.publish_state &&
        checkpoints_.Observe(completed.tokens, completed.prefix_hashes, completed.tail, access_epoch)) {
        syncCheckpoint(completed.tail ? completed.tail->hash : completed.prefix_hashes.back());
    }
}

void CacheCoordinator::syncCheckpoint(const ContentHash& hash) {
    if (checkpoints_.Size() == 0 || !checkpoints_.Contains(hash)) {
        return;
    }
    // A checkpoint nothing holds any more leaves at once.
    const BoundaryTree::Node* node = boundary_tree_.Find(hash);
    checkpoints_.Update(hash,
                        node == nullptr ? CheckpointTree::Residency{} : boundary_tree_.CheckpointResidency(*node));
}

CheckpointAging CacheCoordinator::CheckpointAgingFor(CacheTier tier) const {
    if (checkpoint_retention_requests_ == 0 || checkpoints_.Size() == 0) {
        return CheckpointAging{.tier = tier};
    }
    const auto window = static_cast<std::uint64_t>(checkpoint_retention_requests_);
    // Before that many admissions every entry is recent; epoch 0 stays first.
    const std::uint64_t cutoff_epoch =
        num_admissions_ < window ? 1 : admission_epochs_[(num_admissions_ - window) % admission_epochs_.size()];
    return CheckpointAging{.tree = &checkpoints_, .tier = tier, .cutoff_epoch = cutoff_epoch};
}

void CacheCoordinator::recordAdmission(std::uint64_t access_epoch) {
    admission_epochs_[num_admissions_ % admission_epochs_.size()] = access_epoch;
    ++num_admissions_;
}

void CacheCoordinator::recordResumeGap(std::uint64_t last_used_epoch) {
    const std::uint64_t kept = std::min<std::uint64_t>(num_admissions_, admission_epochs_.size());
    const std::uint64_t oldest = num_admissions_ - kept;
    // The kept admissions' epochs increase from the oldest: find the first
    // one after the checkpoint's last use.
    std::uint64_t low = 0;
    std::uint64_t high = kept;
    while (low < high) {
        const std::uint64_t mid = low + (high - low) / 2;
        if (admission_epochs_[(oldest + mid) % admission_epochs_.size()] > last_used_epoch) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    const std::uint64_t gap = kept - low;
    const bool counted = (low > 0 || oldest == 0) && gap < admission_epochs_.size();
    ++checkpoint_resume_gaps_[counted ? static_cast<std::size_t>(std::bit_width(gap))
                                      : checkpoint_resume_gaps_.size() - 1];
}

void CacheCoordinator::recordPublishedTail(const CompletedPages& completed) {
    if (!completed.tail) {
        return;
    }
    const ContentHash prior = completed.prefix_hashes.empty() ? ContentHash{} : completed.prefix_hashes.back();
    recordTail(prior, *completed.tail);
}

void CacheCoordinator::recordTail(const ContentHash& prior, const TailPublication& tail) {
    const std::optional<TailTable::Tail> dropped = boundary_tree_.RecordTail(prior, tail.tail_tokens, tail.hash);
    if (!dropped) {
        return;
    }
    evictTailEntries(dropped->hash, dropped->tail_tokens, &DeviceEvictionStats::tail_cap,
                     /*write_back_segment=*/nullptr);
    if (host_pool_ == nullptr) {
        return;
    }
    // No probe enumerates a dropped tail, so its Host entries could never be
    // matched again either. One still in use (by a retracted request's image,
    // or as a copy's source) stays, and so does what a write-through store
    // still in flight publishes at its ACK; both age out through ordinary
    // Host eviction. An eviction write-back still in flight publishes nothing
    // of the tail: its settlement finds the hint gone and cancels it. Under
    // write-back what stays is collected as a dead entry once released.
    ForEachBoundaryKey(CacheKey{.content_hash = dropped->hash}, dropped->tail_tokens,
                       [&](std::size_t i, const CacheKey& key) {
                           if (const std::optional<CacheBlockLocation> location = locationOf(*host_pool_, i, key)) {
                               evictHostCachedBlock(groups_[i].Id(), *location, &HostEvictionStats::tail_cap);
                           }
                       });
    // What stayed in use on either tier goes once released.
    boundary_tree_.QueueDeadCandidate(dropped->hash);
}

CacheKey CacheCoordinator::historyBlockKey(std::size_t group_index, const CompletedPages& completed,
                                           std::int32_t block) const {
    const std::int32_t pages_per_prefix_hash = BlocksPerBoundary(static_cast<std::int32_t>(group_index));
    const std::int32_t full_blocks = static_cast<std::int32_t>(completed.prefix_hashes.size()) * pages_per_prefix_hash;
    if (block < full_blocks) {
        return CacheKey{
            .group_id = groups_[group_index].Id(),
            .content_hash = completed.prefix_hashes[static_cast<std::size_t>(block / pages_per_prefix_hash)],
            .page_offset = block % pages_per_prefix_hash};
    }
    _assert(completed.tail.has_value() && block - full_blocks < TailBlocksFor(static_cast<std::int32_t>(group_index),
                                                                              completed.tail->tail_tokens),
            "a published slot beyond the full pages belongs to the tail");
    return CacheKey{.group_id = groups_[group_index].Id(),
                    .content_hash = completed.tail->hash,
                    .page_offset = block - full_blocks};
}

std::vector<CacheCoordinator::PublishedRange> CacheCoordinator::completedBlockRanges(
    std::size_t group_index, const CompletedPages& completed) const {
    std::vector<PublishedRange> ranges;
    if (GroupIsReplayable(static_cast<std::int32_t>(group_index))) {
        return ranges;
    }
    const std::int32_t pages_per_prefix_hash = BlocksPerBoundary(static_cast<std::int32_t>(group_index));
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
        const std::int32_t first_block = completed.first_new_prefix_page * pages_per_prefix_hash;
        const std::int32_t past_end_block = hashed_prefix_pages * pages_per_prefix_hash + tail_blocks;
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
        const ContentHash hash = completed.tail ? completed.tail->hash : completed.prefix_hashes.back();
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
        if (window.first >= window.past_end) {
            continue;
        }
        PublishedRange range = history_range(window.first, window.past_end);
        if (state_checkpoint_tokens > 0 && boundary_tokens == state_checkpoint_tokens) {
            range.boundary_kind = CacheBoundaryKind::kEndpoint;
        }
        ranges.push_back(std::move(range));
    }
    return ranges;
}

template <CacheTier Tier>
void CacheCoordinator::cacheCompletedBlocksForGroup(std::size_t group_index, BlockTable& table,
                                                    const CompletedPages& completed, std::uint64_t access_epoch) {
    // An owner that keeps writing past a tail registers its own tail blocks
    // but never takes another request's canonical block into its table: the
    // copy-in moves it off each block it registered, into the block reserved
    // in that block's bucket, and it keeps writing a block whose key another
    // request published first. Every other published block dedupes onto the
    // canonical one, which the table then holds.
    const BlockRange own_tail =
        completed.tail && completed.owner_tail == OwnerTail::kCopiedIn
            ? CopyInSlots(group_index, static_cast<std::int32_t>(completed.prefix_hashes.size()) * prefix_granularity_ +
                                           completed.tail->tail_tokens)
            : BlockRange{};
    for (const PublishedRange& range : completedBlockRanges(group_index, completed)) {
        const std::int32_t past_end = range.first_cache_block + static_cast<std::int32_t>(range.keys.size());
        const std::int32_t own_first = own_tail.first < own_tail.past_end
                                           ? std::clamp(own_tail.first, range.first_cache_block, past_end)
                                           : past_end;
        const std::span<const CacheKey> keys{range.keys};
        const auto shared = static_cast<std::size_t>(own_first - range.first_cache_block);
        if (shared > 0) {
            cacheFullBlocksForGroup<Tier>(group_index, table, keys.first(shared), range.first_cache_block, access_epoch,
                                          range.boundary_kind, completed.stream_completed_to_host);
        }
        if (own_first < past_end) {
            _assert(own_first == own_tail.first && past_end == own_tail.past_end,
                    "the owner's tail blocks are registered whole");
            // Registration may replace these references, never the table's.
            std::vector<CacheBlockRef> own(table.Blocks().begin() + own_first, table.Blocks().begin() + past_end);
            cacheBlocksForGroup<Tier>(group_index, own, keys.subspan(shared), own_first, access_epoch,
                                      range.boundary_kind, completed.stream_completed_to_host);
        }
    }
    // An in-forward State checkpoint sits outside the table: its block is
    // published like a natural checkpoint's slot 1, under its page's hash.
    if constexpr (Tier == CacheTier::kDevice) {
        if (GroupIsState(group_index) && completed.state_checkpoint_pages > 0 &&
            group_index < completed.state_checkpoint_blocks.size() && completed.state_checkpoint_blocks[group_index]) {
            _assert(completed.state_checkpoint_pages <= static_cast<std::int32_t>(completed.prefix_hashes.size()),
                    "an in-forward State checkpoint lies inside the hashed prefix");
            const CacheKey key{
                .group_id = groups_[group_index].Id(),
                .content_hash = completed.prefix_hashes[static_cast<std::size_t>(completed.state_checkpoint_pages - 1)],
                .page_offset = 0};
            cacheBlocksForGroup<Tier>(group_index, completed.state_checkpoint_blocks.subspan(group_index, 1),
                                      std::span<const CacheKey>(&key, 1), /*first_logical_block=*/1, access_epoch,
                                      CacheBoundaryKind::kEndpoint, completed.stream_completed_to_host);
        }
    }
}

void CacheCoordinator::cacheDeviceCompletedBlocksForGroup(std::size_t group_index, BlockTable& table,
                                                          const CompletedPages& completed, std::uint64_t access_epoch) {
    if (completed.preserve_source_blocks) {
        // Register may replace duplicate refs. Only the publication view may
        // change: the request pins the actual PD sources until its terminal ACK.
        BlockTable publication_table = table;
        std::vector<CacheBlockRef> checkpoints(completed.state_checkpoint_blocks.begin(),
                                               completed.state_checkpoint_blocks.end());
        CompletedPages publication = completed;
        publication.state_checkpoint_blocks = checkpoints;
        cacheCompletedBlocksForGroup<CacheTier::kDevice>(group_index, publication_table, publication, access_epoch);
    } else {
        cacheCompletedBlocksForGroup<CacheTier::kDevice>(group_index, table, completed, access_epoch);
    }
}

std::int32_t CacheCoordinator::FirstPublishedPage(const CompletedPages& completed) const {
    const auto hashed_prefix_pages = static_cast<std::int32_t>(completed.prefix_hashes.size());
    std::int32_t first_page = completed.first_new_prefix_page;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        for (const PublishedRange& range : completedBlockRanges(i, completed)) {
            // A State group's one key is the point itself: the deepest page
            // unless a tail is published. A history group's run starts at
            // slot first_cache_block.
            const std::int32_t page = GroupIsState(i)
                                          ? hashed_prefix_pages - (completed.tail ? 0 : 1)
                                          : range.first_cache_block / BlocksPerBoundary(static_cast<std::int32_t>(i));
            first_page = std::min(first_page, page);
        }
    }
    // The block an in-forward State checkpoint publishes under its page adds
    // no page: the chunk that wrote it started on a page boundary at least
    // one page below it (writesInForwardCheckpoint), so that page is never
    // below the first new one. A sliding window behind it is one of the
    // ranges read above.
    return first_page;
}

void CacheCoordinator::ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Allocator().ReclaimExpired(pool_, tables[i],
                                              groupExpiredBlocksAt(static_cast<std::int32_t>(i), num_computed_tokens));
    }
}

std::vector<CacheBlockRef> CacheCoordinator::RetainInputCheckpoint(std::span<const std::string> prefix_hashes,
                                                                   std::int32_t boundary_tokens) const {
    _assert(boundary_tokens > 0 && boundary_tokens % prefix_granularity_ == 0,
            "an input checkpoint must lie on a prefix boundary");
    const std::int32_t boundary_pages = boundary_tokens / prefix_granularity_;
    _assert(boundary_pages <= static_cast<std::int32_t>(prefix_hashes.size()),
            "a retained checkpoint must have a published hash");
    std::vector<CacheBlockRef> blocks;
    const auto retain = [&](const CacheKey& key) {
        const PrefixCacheIndex& index = groups_[key.group_id].Index();
        CacheBlockRef block = index.Find(pool_, key);
        if (!block && host_pool_ != nullptr) {
            block = index.Find(*host_pool_, key);
        }
        if (!block) {
            return false;
        }
        blocks.push_back(std::move(block));
        return true;
    };
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (GroupIsReplayable(static_cast<std::int32_t>(i)) || groups_[i].Matcher().IsPrefixClosed()) {
            continue;
        }
        if (GroupIsState(i)) {
            if (!retain(CacheKey{.group_id = groups_[i].Id(),
                                 .content_hash = prefix_hashes[static_cast<std::size_t>(boundary_pages - 1)]})) {
                return {};
            }
            continue;
        }
        const BlockRange window = SlidingWindowBlocks(static_cast<std::int32_t>(i), boundary_tokens);
        const std::int32_t blocks_per_boundary = BlocksPerBoundary(static_cast<std::int32_t>(i));
        for (std::int32_t slot = window.first; slot < window.past_end; ++slot) {
            if (!retain(CacheKey{.group_id = groups_[i].Id(),
                                 .content_hash = prefix_hashes[static_cast<std::size_t>(slot / blocks_per_boundary)],
                                 .page_offset = slot % blocks_per_boundary})) {
                return {};
            }
        }
    }
    return blocks;
}

void CacheCoordinator::RefreshRetainedCheckpoint(std::span<CacheBlockRef> blocks, std::uint64_t request_access_epoch) {
    if (blocks.empty()) {
        return;
    }
    const SessionCacheOwner session = SessionForAccess(request_access_epoch);
    const std::uint64_t access_epoch = ++next_access_epoch_;
    for (CacheBlockRef& block : blocks) {
        _assert(block.IsOwnedBy(pool_) || host_pool_ != nullptr && block.IsOwnedBy(*host_pool_),
                "a retained checkpoint belongs to a cache tier");
        const BlockPool& pool = block.IsOwnedBy(pool_) ? pool_ : *host_pool_;
        const auto group_id = pool.BoundGroup(block->Location().lcm_block_id);
        _assert(group_id.has_value(), "an occupied cache block belongs to a group");
        PrefixCacheIndex& index = groups_[*group_id].Index();
        const auto key = index.KeyOf(pool, block);
        if (!key) {
            continue;
        }
        const auto metadata = index.MetadataFor(pool, block->Location());
        _assert(metadata.has_value(), "a cached block has eviction metadata");
        index.Register(pool, block, *key, access_epoch, metadata->logical_block_index, metadata->boundary_kind,
                       /*newly_cached=*/nullptr, session);
        if (GroupIsState(key->group_id)) {
            BoundaryTree::Node* node = boundary_tree_.Find(key->content_hash);
            _assert(node != nullptr, "a cached checkpoint has a boundary-tree node");
            node->latest_state_access_epoch = std::max(node->latest_state_access_epoch, access_epoch);
        }
    }
}

void CacheCoordinator::ConsumeReservedTokens(std::span<BlockTable> tables, std::int32_t num_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (GroupIsState(i)) {
            continue;  // a decode step writes the live slot in place
        }
        groups_[i].Allocator().ConsumeReservedTokens(tables[i], num_tokens);
    }
}

void CacheCoordinator::Free(std::span<BlockTable> tables) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Allocator().Free(tables[i]);
    }
}

namespace {

// Slots covering [0, num_tokens) of a table and the capacity left inside the
// last of them. The computed tokens always lie inside the table: the
// admission that scheduled them acquired their blocks first. A State table's
// two slots are both data.
CacheCoordinator::DataSpan dataSpan(const BlockTable& table, const GroupGeometry& geometry, std::int32_t num_tokens) {
    if (!geometry.HasTokenSpan()) {
        return CacheCoordinator::DataSpan{.blocks = table.NumBlocks(), .tail_tokens = 0};
    }
    const std::int32_t block_granularity = geometry.BlockGranularity();
    const std::int32_t covering = (num_tokens + block_granularity - 1) / block_granularity;
    _assert(covering <= table.NumBlocks(), "snapshot token count exceeds the table's logical fill");
    return CacheCoordinator::DataSpan{.blocks = covering, .tail_tokens = covering * block_granularity - num_tokens};
}

}  // namespace

std::vector<std::vector<ImageSlot>> CacheCoordinator::PublishedDataSlots(std::span<const BlockTable> tables,
                                                                         std::int32_t num_tokens) const {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    _assert(num_tokens >= 0, "snapshot token count must be non-negative");
    std::vector<std::vector<ImageSlot>> published(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const DataSpan span = dataSpan(tables[i], geometry_[i], num_tokens);
        const std::span<const CacheBlockRef> blocks = tables[i].Blocks();
        for (std::int32_t slot = 0; slot < span.blocks; ++slot) {
            const CacheBlockRef& block = blocks[static_cast<std::size_t>(slot)];
            if (std::optional<CacheKey> key = groups_[i].Index().KeyOf(pool_, block)) {
                const std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata =
                    groups_[i].Index().MetadataFor(pool_, block->Location());
                _assert(metadata.has_value(), "a published slot has an index entry");
                published[i].push_back(ImageSlot{.slot_index = slot,
                                                 .block = block,
                                                 .key = std::move(*key),
                                                 .logical_block_index = metadata->logical_block_index,
                                                 .boundary_kind = metadata->boundary_kind});
            }
        }
    }
    return published;
}

std::optional<CacheCoordinator::SnapshotPoolPlan> CacheCoordinator::planSnapshotPool(
    std::span<const BlockTable> tables, std::int32_t num_tokens,
    std::span<const std::vector<ImageSlot>> host_served_slots) const {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    _assert(host_served_slots.size() == groups_.size(), "host slots/groups size mismatch");
    _assert(num_tokens >= 0, "snapshot token count must be non-negative");
    if (snapshot_pool_ == nullptr) {
        return std::nullopt;
    }
    // Every group's bucket demand against the pool's empty parents, with no
    // acquisition: a retraction that does not fit must take nothing, since
    // releasing a partial acquisition would reorder the pool's FIFO.
    SnapshotPoolPlan plan{
        .spans = std::vector<DataSpan>(groups_.size()),
        .private_slots = std::vector<std::vector<std::int32_t>>(groups_.size()),
        .private_buckets = std::vector<std::vector<std::int32_t>>(groups_.size()),
    };
    std::int64_t parents_needed = 0;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const GroupAllocator& allocator = groups_[i].Allocator();
        plan.spans[i] = dataSpan(tables[i], geometry_[i], num_tokens);
        std::vector<std::uint8_t> served(static_cast<std::size_t>(plan.spans[i].blocks), 0);
        for (const ImageSlot& slot : host_served_slots[i]) {
            _assert(slot.InHostCache(), "a Host-served snapshot slot carries the key it is published under");
            _assert(slot.slot_index >= 0 && slot.slot_index < plan.spans[i].blocks,
                    "a Host-served snapshot slot must be a data slot");
            served[static_cast<std::size_t>(slot.slot_index)] = 1;
        }
        std::vector<std::int64_t> need(static_cast<std::size_t>(allocator.ShardCount()), 0);
        const std::span<const CacheBlockRef> blocks = tables[i].Blocks();
        for (std::int32_t slot = 0; slot < plan.spans[i].blocks; ++slot) {
            const CacheBlockRef& block = blocks[static_cast<std::size_t>(slot)];
            if (!block || served[static_cast<std::size_t>(slot)]) {
                continue;
            }
            const std::int32_t bucket = allocator.BucketOf(block->Location());
            plan.private_slots[i].push_back(slot);
            plan.private_buckets[i].push_back(bucket);
            ++need[static_cast<std::size_t>(bucket)];
        }
        parents_needed += ParentsNeededForBuckets(need, snapshot_pool_->FreeSlotsByBucket(groups_[i].Id()),
                                                  /*dense_need=*/0, snapshot_pool_->Packing(groups_[i].Id()));
    }
    if (parents_needed > snapshot_pool_->NumEmptyLcmBlocks()) {
        return std::nullopt;
    }
    return plan;
}

bool CacheCoordinator::SnapshotPoolHolds(std::span<const BlockTable> tables, std::int32_t num_tokens,
                                         std::span<const std::vector<ImageSlot>> host_served_slots) const {
    return planSnapshotPool(tables, num_tokens, host_served_slots).has_value();
}

std::vector<std::vector<ImageSlot>> CacheCoordinator::PublishedDataSlotsAfter(std::span<const BlockTable> tables,
                                                                              std::int32_t num_tokens,
                                                                              const RequestProgress& progress) const {
    std::vector<std::vector<ImageSlot>> published = PublishedDataSlots(tables, num_tokens);
    if (!progress.completed_pages) {
        return published;
    }
    const CompletedPages& completed = *progress.completed_pages;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::vector<PublishedRange> ranges = completedBlockRanges(i, completed);
        if (ranges.empty()) {
            continue;
        }
        const DataSpan span = dataSpan(tables[i], geometry_[i], num_tokens);
        const std::span<const CacheBlockRef> blocks = tables[i].Blocks();
        std::vector<ImageSlot>& slots = published[i];
        for (const PublishedRange& range : ranges) {
            for (std::size_t j = 0; j < range.keys.size(); ++j) {
                const std::int32_t slot = range.first_cache_block + static_cast<std::int32_t>(j);
                const auto index = static_cast<std::size_t>(slot);
                // A slot beyond the data span, a null hole, or one already
                // published is not added: the publication would register
                // nothing new there.
                if (slot >= span.blocks || !blocks[index] ||
                    std::ranges::any_of(slots, [slot](const ImageSlot& s) { return s.slot_index == slot; })) {
                    continue;
                }
                // Register re-points a slot whose key the index holds already
                // to that canonical block.
                CacheBlockRef canonical = groups_[i].Index().Find(pool_, range.keys[j]);
                slots.push_back(ImageSlot{.slot_index = slot,
                                          .block = canonical ? std::move(canonical) : blocks[index],
                                          .key = range.keys[j],
                                          .logical_block_index = slot,
                                          .boundary_kind = range.boundary_kind});
            }
        }
        std::ranges::sort(slots, {}, &ImageSlot::slot_index);
    }
    return published;
}

std::optional<CacheCoordinator::ImageTaken> CacheCoordinator::TakeImage(
    std::span<const BlockTable> tables, std::int32_t num_tokens,
    std::span<const std::vector<ImageSlot>> host_cached_slots) {
    for (std::size_t i = 0; i < host_cached_slots.size(); ++i) {
        for (const ImageSlot& slot : host_cached_slots[i]) {
            _assert(slot.block && host_pool_ != nullptr && slot.block.IsOwnedBy(*host_pool_),
                    "a Host-cached snapshot slot must carry its Host L2 block");
        }
    }
    std::optional<SnapshotPoolPlan> plan = planSnapshotPool(tables, num_tokens, host_cached_slots);
    if (!plan) {
        return std::nullopt;
    }

    ImageTaken image;
    image.image.tables.reserve(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        ImageTable shape{
            .num_blocks = plan->spans[i].blocks,
            .reclaimed_prefix_blocks = std::min(tables[i].ReclaimedPrefixBlocks(), plan->spans[i].blocks),
            .available_tokens = plan->spans[i].tail_tokens,
            .slots = host_cached_slots[i],
        };
        std::vector<CacheBlockRef> snapshot_blocks =
            snapshot_pool_->AcquireBlocksInBuckets(groups_[i].Id(), plan->private_buckets[i]);
        FatalCheck(snapshot_blocks.size() == plan->private_buckets[i].size(),
                   "snapshot pool no longer fits a probed image");
        for (std::size_t j = 0; j < snapshot_blocks.size(); ++j) {
            const CacheBlockRef& source = tables[i].Blocks()[static_cast<std::size_t>(plan->private_slots[i][j])];
            image.store_pairs.push_back(BlockTransfer{
                .group_id = groups_[i].Id(),
                .source = source,
                .destination = snapshot_blocks[j],
            });
            shape.slots.push_back(
                ImageSlot{.slot_index = plan->private_slots[i][j], .block = std::move(snapshot_blocks[j])});
        }
        std::ranges::sort(shape.slots, {}, &ImageSlot::slot_index);
        image.image.tables.push_back(std::move(shape));
    }
    return image;
}

bool CacheCoordinator::ContainsHostCachedBlock(const CacheKey& key) const {
    if (host_pool_ == nullptr) {
        return false;
    }
    _assert(key.group_id < groups_.size(), "host cache key group id out of range");
    return groups_[key.group_id].Index().Contains(*host_pool_, key);
}

bool CacheCoordinator::IsHostCachedBlock(CacheBlockLocation location) const {
    if (host_pool_ == nullptr) {
        return false;
    }
    return std::ranges::any_of(groups_,
                               [&](const CacheGroup& group) { return group.Index().Contains(*host_pool_, location); });
}

std::int32_t CacheCoordinator::NumHostCachedBlocks() const {
    if (host_pool_ == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : groups_) {
        count += group.Index().NumEntries(*host_pool_);
    }
    return count;
}

std::int32_t CacheCoordinator::NumPinnedHostCachedBlocks() const {
    if (host_pool_ == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : groups_) {
        count += group.Index().NumPinnedEntries(*host_pool_);
    }
    return count;
}

void CacheCoordinator::CacheHostBlock(CacheBlockRef& block_ref, const CacheKey& key,
                                      std::optional<PrefixCacheIndex::CachedBlockMetadata> inherited_metadata) {
    _assert(host_pool_ != nullptr, "CacheHostBlock requires a host pool");
    _assert(key.group_id < groups_.size(), "CacheHostBlock group id out of range");
    PrefixCacheIndex& index = groups_[key.group_id].Index();
    PrefixCacheIndex::CachedBlockMetadata metadata{
        .logical_block_index = -1,
        .boundary_kind = CacheBoundaryKind::kChunk,
    };
    if (inherited_metadata) {
        metadata = *inherited_metadata;
        // Registering touches an existing entry with the epoch given: never
        // age one a Host hit of this very build may just have touched.
        if (const std::optional<CacheBlockLocation> existing = locationOf(*host_pool_, key.group_id, key)) {
            metadata.last_access_epoch =
                std::max(metadata.last_access_epoch, index.MetadataFor(*host_pool_, *existing)->last_access_epoch);
        }
    } else {
        metadata.last_access_epoch = ++next_access_epoch_;
    }
    // The entry holds the node of its hash: a store's the node its Device
    // publication linked, while either tier holds the hash; an L3 GET's, or a
    // store's whose Device entry left before the ACK, one it starts without a
    // parent, until a publication through its hash gives it its boundary (and,
    // past page 0, links it).
    index.Register(*host_pool_, block_ref, key, metadata.last_access_epoch, metadata.logical_block_index,
                   metadata.boundary_kind, /*newly_cached=*/nullptr, metadata.session);
    // A copy a demotion reached on its way (TierTransferManager::
    // DemoteInFlightEvictions) lands demoted, unless the entry is newer.
    if (metadata.demoted) {
        index.MarkDemoted(*host_pool_, key, kDemotedAccessEpoch);
    }
    if (GroupIsState(key.group_id)) {
        BoundaryTree::Node* node = boundary_tree_.Find(key.content_hash);
        _assert(node != nullptr, "an indexed checkpoint has a tree node");
        node->latest_state_access_epoch =
            std::max({node->latest_state_access_epoch, metadata.last_access_epoch, metadata.max_access_epoch});
    }
}

void CacheCoordinator::CacheHostTail(const ContentHash& prior, const TailPublication& tail) {
    // A failed/partial GET can leave only some groups present. Record a hint
    // only when Host holds data under it; ordinary matching still validates
    // every group's exact boundary before claiming anything.
    bool on_host = false;
    ForEachBoundaryKey(CacheKey{.content_hash = tail.hash}, tail.tail_tokens,
                       [&](std::size_t, const CacheKey& key) { on_host = on_host || ContainsHostCachedBlock(key); });
    if (on_host) {
        recordTail(prior, tail);
    }
}

void CacheCoordinator::CacheDeviceBlock(CacheBlockRef& block_ref, const CacheKey& key, std::int32_t logical_block_index,
                                        CacheBoundaryKind boundary_kind, std::uint64_t access_epoch) {
    _assert(static_cast<bool>(block_ref), "CacheDeviceBlock requires a destination block");
    _assert(key.group_id < groups_.size(), "CacheDeviceBlock group id out of range");
    std::vector<std::pair<CacheKey, CacheBlockRef>> newly_cached;
    // The entry holds the node of its hash. The node the restored entry was
    // published under is gone when no tier held the hash meanwhile; then the
    // entry starts one without a parent, until a publication through its hash
    // gives it its boundary (and, past page 0, links it).
    groups_[key.group_id].Index().Register(pool_, block_ref, key, access_epoch, logical_block_index, boundary_kind,
                                           &newly_cached, SessionForAccess(access_epoch));
    if (GroupIsState(key.group_id)) {
        BoundaryTree::Node* node = boundary_tree_.Find(key.content_hash);
        _assert(node != nullptr, "an indexed checkpoint has a tree node");
        node->latest_state_access_epoch = std::max(node->latest_state_access_epoch, access_epoch);
    }
    if (!cache_mutation_sink_) {
        return;
    }
    for (const auto& [cached_key, cached_block] : newly_cached) {
        (void)cached_block;
        cache_mutation_sink_(cached_key, CacheMutation::kStored);
    }
}

std::vector<CacheKey> CacheCoordinator::ExpandPrefixKeys(std::span<const std::string> content_hashes) const {
    std::vector<CacheKey> keys;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        std::vector<CacheKey> group_keys = keysForGroup(content_hashes, groups_[i].Id());
        keys.insert(keys.end(), std::make_move_iterator(group_keys.begin()), std::make_move_iterator(group_keys.end()));
    }
    return keys;
}

std::unique_ptr<PrefixMatcher> MakePrefixMatcher(const CacheGroupSpec& spec) {
    switch (spec.attn_kind) {
        case AttnKind::kFull:
            _assert(spec.block_granularity > 0, "group block_granularity must be > 0");
            return std::make_unique<FullAttnMatcher>();
        case AttnKind::kState:
            _assert(spec.block_granularity == 0, "a State group has no token span");
            return std::make_unique<StateMatcher>();
        case AttnKind::kSlidingWindow:
            _assert(spec.block_granularity > 0, "group block_granularity must be > 0");
            _assert(spec.sliding_window > 0, "sliding window group requires a positive window");
            return std::make_unique<SwaMatcher>(spec.block_granularity, spec.sliding_window);
        default:
            FatalCheck(false, "unknown AttnKind in coordinator group spec");
            return nullptr;
    }
}

CacheCoordinator MakeCoordinator(std::span<const CacheGroupSpec> specs, std::int32_t prefix_granularity,
                                 BlockPool& pool, bool enable_l3_storage, BlockPool* host_pool,
                                 BlockPool* snapshot_pool, bool stream_device_cache_to_host, bool write_back_evictions,
                                 std::int64_t checkpoint_retention_requests,
                                 std::int32_t checkpoint_retention_leaf_distance) {
    _assert(!specs.empty(), "MakeCoordinator requires at least one spec");
    _assert(prefix_granularity > 0, "prefix_granularity must be > 0");
    _assert(specs.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "number of cache groups exceeds int32 range");
    std::vector<CacheGroup> groups;
    groups.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const CacheGroupSpec& spec = specs[i];
        const std::uint32_t group_id = static_cast<std::uint32_t>(i);
        _assert(spec.cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        if (spec.attn_kind == AttnKind::kState) {
            _assert(spec.block_granularity == 0, "a State group has no token span");
        } else {
            _assert(spec.block_granularity > 0 && prefix_granularity % spec.block_granularity == 0,
                    "group block_granularity must be a positive divisor of the prefix granularity");
        }
        auto allocator = std::make_unique<GroupAllocator>(spec.cache_blocks_per_lcm_block, group_id, spec.shard_count);
        groups.emplace_back(spec, std::move(allocator), MakePrefixMatcher(spec));
    }
    return CacheCoordinator{std::move(groups),
                            prefix_granularity,
                            pool,
                            enable_l3_storage,
                            host_pool,
                            snapshot_pool,
                            stream_device_cache_to_host,
                            write_back_evictions,
                            checkpoint_retention_requests,
                            checkpoint_retention_leaf_distance};
}

CacheBlockRef CacheCoordinator::AcquireHostCachedBlock(const CacheKey& key) {
    return host_pool_ == nullptr ? CacheBlockRef{} : groups_.at(key.group_id).Index().Find(*host_pool_, key);
}

template <typename Visit>
void CacheCoordinator::forEachStorageDependency(const PrefixProbe::Tier& probe,
                                                std::span<const std::vector<CacheKey>> group_keys,
                                                std::int32_t floor_tokens, const Visit& visit) const {
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        // A State group's hits count from page 0 (PrefixProbe::Tier).
        const auto begin =
            GroupIsState(i) ? 0 : floor_tokens / prefix_granularity_ * BlocksPerBoundary(static_cast<std::int32_t>(i));
        for (std::size_t j = 0; j < probe.per_group[i].hits.size(); ++j) {
            if (probe.per_group[i].hits[j]) {
                visit(group_keys[i][static_cast<std::size_t>(begin) + j], begin + static_cast<std::int32_t>(j));
            }
        }
        if (!probe.tail) continue;
        const auto& tail = *probe.tail;
        const auto full_blocks =
            probe.full_page_tokens / (GroupIsState(i) ? prefix_granularity_ : geometry_[i].BlockGranularity());
        for (std::size_t j = 0; j < tail.per_group[i].hits.size(); ++j) {
            if (tail.per_group[i].hits[j]) {
                visit(tail.group_keys[i][j], full_blocks + static_cast<std::int32_t>(j));
            }
        }
    }
}

CacheCoordinator::StorageQuery CacheCoordinator::MissingStorageKeys(const PromptPrefix& prompt,
                                                                    std::size_t max_keys) const {
    const PrefixProbe local = ProbePromptPrefix(prompt);
    StorageQuery query;
    const auto local_tokens = std::max(local.device.num_common_tokens, local.host.num_common_tokens);
    const auto floor = std::max(local.device.full_page_tokens, local.host.full_page_tokens);
    query.max_tokens = local_tokens;
    if (max_keys == 0 || prompt.max_tokens <= local_tokens) return query;
    struct Candidate {
        std::int32_t full_tokens;
        std::int32_t tail_tokens;
        ContentHash hash;
    };
    std::vector<Candidate> candidates;
    forEachCandidateBoundary(prompt, floor, prompt.max_tokens,
                             [&](std::int32_t full_tokens, std::int32_t tail_tokens, const ContentHash& hash) {
                                 if (full_tokens + tail_tokens <= local_tokens) return true;
                                 candidates.push_back({full_tokens, tail_tokens, hash});
                                 return true;
                             });

    // Temporary assumptions let the ordinary matcher enumerate a complete
    // candidate's Full history, SWA window and endpoint State. They never
    // become published hits or acquire any Host references. They cover every
    // page: the Device decides where a candidate's Host suffix starts, and the
    // matcher consults them only from there on.
    std::unordered_set<CacheKey, CacheKeyHash> possible;
    for (const std::size_t i : match_order_) {
        const auto per_prefix = BlocksPerBoundary(static_cast<std::int32_t>(i));
        const auto end = static_cast<std::size_t>(prompt.max_tokens / prefix_granularity_ * per_prefix);
        for (std::size_t j = 0; j < end; ++j) possible.insert((*local.group_keys)[i][j]);
        for (const auto& candidate : candidates) {
            if (candidate.tail_tokens > 0) {
                for (const auto& key : tailKeysForGroup(i, candidate.hash, candidate.tail_tokens)) possible.insert(key);
            }
        }
    }

    // Each candidate above the local hit resumes at the Device's KV prefix
    // plus a Host suffix: only the suffix's keys missing from Host are asked
    // for, never a Device-resident checkpoint or KV page below the split.
    std::unordered_set<CacheKey, CacheKeyHash> selected;
    for (const auto& boundary : candidates) {
        auto candidate = prompt;
        candidate.max_tokens = boundary.full_tokens + boundary.tail_tokens;
        candidate.tail =
            boundary.tail_tokens > 0
                ? std::optional{TailPublication{.tail_tokens = boundary.tail_tokens, .hash = boundary.hash}}
                : std::nullopt;
        PrefixProbe::Tier device = local.device;
        PrefixProbe::Tier probe;
        probeHostSuffix(candidate, *local.group_keys, match_order_, local_tokens, &possible, device, probe);
        _assert(probe.num_common_tokens == candidate.max_tokens, "an optimistic candidate must match completely");
        std::vector<CacheKey> additional;
        std::vector<std::int32_t> additional_tokens;
        std::unordered_set<CacheKey, CacheKeyHash> distinct;
        bool fits = true;
        forEachStorageDependency(
            probe, *local.group_keys, device.full_page_tokens, [&](const CacheKey& key, std::int32_t logical_block) {
                if (!fits || ContainsHostCachedBlock(key) || selected.contains(key) || !distinct.insert(key).second)
                    return;
                if (additional.size() == max_keys - query.keys.size()) {
                    fits = false;
                    return;
                }
                additional.push_back(key);
                const auto grain =
                    GroupIsState(key.group_id) ? prefix_granularity_ : geometry_[key.group_id].BlockGranularity();
                additional_tokens.push_back(std::min(grain, candidate.max_tokens - logical_block * grain));
            });
        if (!fits) continue;
        query.max_tokens = std::max(query.max_tokens, candidate.max_tokens);
        selected.insert(additional.begin(), additional.end());
        query.keys.insert(query.keys.end(), additional.begin(), additional.end());
        query.key_tokens.insert(query.key_tokens.end(), additional_tokens.begin(), additional_tokens.end());
        if (query.keys.size() == max_keys) break;
    }
    return query;
}

CacheCoordinator::StorageReservation CacheCoordinator::ReserveStoragePrefix(const PromptPrefix& prompt,
                                                                            std::span<const CacheKey> hits,
                                                                            std::size_t max_blocks) {
    const std::unordered_set<CacheKey, CacheKeyHash> remote(hits.begin(), hits.end());
    // The Device-only probe: the reservation matches the Host suffix (Host and
    // remote) above its boundary again, under the same Host-suffix rule as
    // admission.
    const std::vector<std::vector<CacheKey>> group_keys = buildGroupKeys(prompt.prefix_hashes);
    const PrefixProbe::Tier device_only = probeTier(prompt, group_keys, match_order_, /*claim_partial=*/true);
    auto candidate = prompt;
    while (candidate.max_tokens > device_only.num_common_tokens) {
        PrefixProbe::Tier device = device_only;
        PrefixProbe::Tier probe;
        probeHostSuffix(candidate, group_keys, match_order_, device_only.num_common_tokens, &remote, device, probe);
        if (probe.num_common_tokens <= device_only.num_common_tokens) return {};
        StorageReservation result;
        std::vector<CacheBlockRef> warm_blocks;
        std::vector<std::uint32_t> groups;
        std::vector<std::int32_t> buckets;
        const auto pin = [&](const CacheKey& key, std::int32_t logical_block) {
            auto ref = AcquireHostCachedBlock(key);
            if (ref) {
                warm_blocks.push_back(std::move(ref));
            } else {
                result.keys.push_back(key);
                groups.push_back(key.group_id);
                buckets.push_back(logical_block % groups_[key.group_id].Spec().shard_count);
            }
        };
        // Protect warm dependencies only while allocating destinations. They
        // stay evictable during I/O; ordinary admission matches the cache again.
        forEachStorageDependency(probe, group_keys, device.full_page_tokens, pin);
        if (probe.tail) {
            result.tail_prior = probe.full_page_tokens == 0
                                    ? ContentHash{}
                                    : prompt.prefix_hashes[probe.full_page_tokens / prefix_granularity_ - 1];
            result.tail = TailPublication{.tail_tokens = probe.tail->tail_tokens, .hash = probe.tail->hash};
        }
        if (result.keys.empty()) return {};
        if (result.keys.size() + warm_blocks.size() <= max_blocks) {
            auto allocation = AcquireHostBlocks(groups, buckets);
            if (allocation.stats.allocated == groups.size()) {
                result.blocks = std::move(allocation.blocks);
                return result;
            }
        }
        // Re-run the matcher; a window/state mask cannot be truncated safely.
        candidate.max_tokens = probe.num_common_tokens - 1;
        candidate.tail.reset();
    }
    return {};
}

}  // namespace tokenspeed
