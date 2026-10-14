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

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <iterator>
#include <utility>

#include "cache/prefix/prefix_hasher.h"
#include "cache/prefix/prefix_matcher.h"
#include "utils.h"

namespace tokenspeed {

CacheCoordinator::CacheCoordinator(std::vector<CacheGroup> groups, std::int32_t prefix_granularity, BlockPool& pool,
                                   bool enable_l3_storage, BlockPool* host_pool, BlockPool* snapshot_pool,
                                   bool stream_device_cache_to_host)
    : groups_{std::move(groups)},
      pool_{pool},
      host_pool_{host_pool},
      snapshot_pool_{snapshot_pool},
      stream_device_cache_to_host_{stream_device_cache_to_host && host_pool != nullptr},
      enable_l3_storage_{enable_l3_storage && host_pool != nullptr},
      prefix_granularity_{prefix_granularity} {
    _assert(prefix_granularity_ > 0, "coordinator needs positive prefix_granularity");
    geometry_.reserve(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        _assert(groups_[i].Id() == static_cast<std::uint32_t>(i), "cache group id must equal its group index");
        const std::int32_t group_block_granularity = groups_[i].Spec().block_granularity;
        if (groups_[i].Spec().attn_kind == AttnKind::kState) {
            _assert(group_block_granularity == 0, "a State group has no token span");
        } else {
            _assert(group_block_granularity > 0 && prefix_granularity_ % group_block_granularity == 0,
                    "group block_granularity must be a positive divisor of the prefix granularity");
        }
        _assert(groups_[i].Allocator().CacheBlocksPerLcmBlock() == groups_[i].Spec().cache_blocks_per_lcm_block,
                "group allocator packing must match its group spec");
        _assert(groups_[i].Allocator().ShardCount() == groups_[i].Spec().shard_count,
                "group allocator shard count must match its group spec");
        const auto& spec = groups_[i].Spec();
        pool_.RegisterGroup(groups_[i].Id(), spec.cache_blocks_per_lcm_block, spec.shard_count);
        if (host_pool_ != nullptr) {
            host_pool_->RegisterGroup(groups_[i].Id(), spec.cache_blocks_per_lcm_block, spec.shard_count);
        }
        // The same packing and shard count as the Device pool, so a snapshot
        // block's virtual id has the same owner as the Device block it images.
        if (snapshot_pool_ != nullptr) {
            snapshot_pool_->RegisterGroup(groups_[i].Id(), spec.cache_blocks_per_lcm_block, spec.shard_count);
        }
        geometry_.emplace_back(group_block_granularity);
        if (spec.replayable) {
            _assert(spec.attn_kind == AttnKind::kSlidingWindow, "a replayable group must be a sliding window");
            replay_window_tokens_ = std::max(replay_window_tokens_, spec.sliding_window);
        }
        // Every Host copy keeps the bucket of its Device end, so a page-cyclic
        // owner translation selects the same rows on both sides. An L3
        // prefetch allocates its Host page before any Device destination
        // exists and would have to choose the bucket blind, so L3 stays
        // replicated-only.
        _assert(!enable_l3_storage_ || spec.shard_count == 1,
                "L3 storage is not supported for page-cyclic sharded cache groups");
        if (groups_[i].Matcher().IsPrefixClosed()) {
            match_order_.push_back(i);
        }
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!groups_[i].Matcher().IsPrefixClosed() && !GroupIsReplayable(static_cast<std::int32_t>(i))) {
            match_order_.push_back(i);
        }
    }
    if (enable_l3_storage_ && host_pool_ != nullptr) {
        std::size_t capacity = 0;
        for (const CacheGroup& group : groups_) {
            capacity += static_cast<std::size_t>(host_pool_->NumLcmBlocks()) *
                        static_cast<std::size_t>(group.Spec().cache_blocks_per_lcm_block);
        }
        storage_key_limit_ = std::max(capacity, std::size_t{1});
    }
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

    pending_stores_.clear();
    storage_keys_.clear();
    storage_key_order_.clear();
    for (const auto& [group_id, location] : cached_locations) {
        // The cascade of an earlier eviction may already have taken this one.
        evictCachedBlock(group_id, location);
        FatalCheck(!pool_.IsOccupied(location), "clearable Device cache entry disappeared");
    }
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
    }
    tails_.Clear();
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

CacheCoordinator::BoundaryResidency CacheCoordinator::DeviceBoundaryResidency(const CacheKey& boundary,
                                                                              std::int32_t tail_tokens) const {
    _assert(tail_tokens >= 0 && tail_tokens < prefix_granularity_, "a tail lies strictly inside a prefix page");
    std::int32_t cached = 0;
    std::int32_t total = 0;
    for (std::size_t group_index = 0; group_index < groups_.size(); ++group_index) {
        // A replayable group never holds an entry, so it would keep every
        // boundary permanently partial.
        if (GroupIsReplayable(static_cast<std::int32_t>(group_index))) {
            continue;
        }
        const std::int32_t pages_per_prefix_hash =
            tail_tokens > 0 ? TailBlocksFor(static_cast<std::int32_t>(group_index), tail_tokens)
                            : BlocksPerBoundary(static_cast<std::int32_t>(group_index));
        const std::int32_t first_offset =
            tail_tokens > 0 ? FirstTailBlockNeeded(static_cast<std::int32_t>(group_index), tail_tokens) : 0;
        for (std::int32_t offset = first_offset; offset < pages_per_prefix_hash; ++offset) {
            const CacheKey key{
                .namespace_id = boundary.namespace_id,
                .group_id = groups_[group_index].Id(),
                .content_hash = boundary.content_hash,
                .page_offset = offset,
            };
            cached += groups_[group_index].Index().Contains(pool_, key) ? 1 : 0;
            ++total;
        }
    }
    if (cached == 0) {
        return BoundaryResidency::kNone;
    }
    return cached == total ? BoundaryResidency::kComplete : BoundaryResidency::kPartial;
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
// block_granularity. State groups take no part here: their boundaries are
// candidates of probeTier's candidate phase.
template <CacheTier Tier>
CacheCoordinator::PrefixProbe::Tier CacheCoordinator::probeTierWithKeys(
    std::span<const std::vector<CacheKey>> group_keys, std::span<const std::size_t> match_order,
    std::int32_t num_prefix_pages, std::int32_t floor_tokens, bool with_storage_keys) const {
    _assert(!with_storage_keys || (Tier == CacheTier::kHost && enable_l3_storage_),
            "the storage shadow extends the Host tier, and only with L3 enabled");
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
    const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits = with_storage_keys ? &storage_keys_ : nullptr;
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
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t floor_pages =
            floor_tokens / (GroupIsState(i) ? prefix_granularity_ : geometry_[i].BlockGranularity());
        out.per_group[i] =
            groups_[i].Index().AcquireMatched(pool, group_keys[i], floor_pages, probe.per_group[i], access_epoch);
        if (!probe.tail || probe.tail->group_keys[i].empty()) {
            continue;
        }
        // The tail's blocks follow the full pages in the same table.
        PrefixMatch tail_match = groups_[i].Index().AcquireMatched(pool, probe.tail->group_keys[i], /*begin_blocks=*/0,
                                                                   probe.tail->per_group[i], access_epoch);
        out.per_group[i].blocks.insert(out.per_group[i].blocks.end(),
                                       std::make_move_iterator(tail_match.blocks.begin()),
                                       std::make_move_iterator(tail_match.blocks.end()));
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
// published at the boundary.
template <CacheTier Tier>
bool CacheCoordinator::groupAcceptsBoundary(std::size_t i, std::span<const std::string> prefix_hashes,
                                            std::int32_t floor_tokens, std::int32_t full_tokens,
                                            std::int32_t tail_tokens, std::span<const CacheKey> tail_keys,
                                            GroupPrefixProbe& tail_hits, GroupPrefixProbe* page_hits) const {
    const BlockPool& pool = tierPool<Tier>();
    const CacheGroup& group = groups_[i];
    const PrefixCacheIndex& index = group.Index();
    tail_hits.hits.assign(tail_keys.size(), 0);
    if (GroupIsState(i)) {
        if (tail_tokens > 0) {
            _assert(tail_keys.size() == 1, "a State group has one key per boundary");
            if (!index.Contains(pool, tail_keys[0])) {
                return false;
            }
            tail_hits.hits[0] = 1;
            return true;
        }
        // An aligned boundary is keyed by the chain hash of its last page.
        _assert(full_tokens > floor_tokens && full_tokens % prefix_granularity_ == 0,
                "a page boundary above the floor");
        const auto page = static_cast<std::size_t>(full_tokens / prefix_granularity_) - 1;
        const CacheKey key{.group_id = group.Id(), .content_hash = prefix_hashes[page], .page_offset = 0};
        if (!index.Contains(pool, key)) {
            return false;
        }
        if (page_hits != nullptr) {
            const auto floor_pages = static_cast<std::size_t>(floor_tokens / prefix_granularity_);
            page_hits->hits.assign(page + 1 - floor_pages, 0);
            page_hits->hits.back() = 1;
        }
        return true;
    }
    if (group.Matcher().IsPrefixClosed()) {
        for (std::size_t j = 0; j < tail_keys.size(); ++j) {
            if (!index.Contains(pool, tail_keys[j])) {
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
            if (!index.Contains(pool, tail_keys[j])) {
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
        if (!index.Contains(pool, page_key)) {
            return false;
        }
        if (page_hits != nullptr) {
            page_hits->hits[static_cast<std::size_t>(block - floor_blocks)] = 1;
        }
    }
    return true;
}

// The boundaries a tier may resume at, longest first: the tails published
// inside each page under its prior and the page boundaries themselves, from
// the page containing `from_tokens` down to the floor.
template <typename Visit>
void CacheCoordinator::forEachCandidateBoundary(const PromptPrefix& prompt, std::int32_t floor_tokens,
                                                std::int32_t from_tokens, const Visit& visit) const {
    for (std::int32_t page = from_tokens / prefix_granularity_; page * prefix_granularity_ >= floor_tokens; --page) {
        const std::int32_t page_tokens = page * prefix_granularity_;
        if (static_cast<std::size_t>(page) <= prompt.prefix_hashes.size() && !prompt.tokens.empty()) {
            const ContentHash prior =
                page == 0 ? ContentHash{} : prompt.prefix_hashes[static_cast<std::size_t>(page) - 1];
            const std::span<const TailTable::Tail> published = tails_.Tails(prior);
            std::vector<TailTable::Tail> candidates(published.begin(), published.end());
            std::ranges::stable_sort(candidates, std::greater<>{}, &TailTable::Tail::tail_tokens);
            for (const TailTable::Tail& tail : candidates) {
                const std::int32_t boundary = page_tokens + tail.tail_tokens;
                if (boundary > from_tokens || boundary > prompt.max_tokens || boundary <= floor_tokens ||
                    static_cast<std::size_t>(boundary) > prompt.tokens.size()) {
                    continue;
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

// One tier of a probe: the aligned phase over the prefix-closed groups, then
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
template <CacheTier Tier>
CacheCoordinator::PrefixProbe::Tier CacheCoordinator::probeTier(const PromptPrefix& prompt,
                                                                std::span<const std::vector<CacheKey>> group_keys,
                                                                std::span<const std::size_t> match_order,
                                                                std::int32_t floor_tokens, bool claim_partial) const {
    const std::int32_t num_prefix_pages =
        std::min(static_cast<std::int32_t>(prompt.prefix_hashes.size()), prompt.max_tokens / prefix_granularity_);
    std::vector<std::size_t> closed_order;
    for (const std::size_t i : match_order) {
        if (!GroupIsState(i) && groups_[i].Matcher().IsPrefixClosed()) {
            closed_order.push_back(i);
        }
    }
    PrefixProbe::Tier aligned = probeTierWithKeys<Tier>(group_keys, closed_order, num_prefix_pages, floor_tokens,
                                                        /*with_storage_keys=*/false);
    PrefixProbe::Tier out = aligned;
    out.num_common_tokens = floor_tokens;
    out.full_page_tokens = floor_tokens;
    out.kv_tokens = floor_tokens;
    ContentHash hash;
    std::int32_t hashed_page_tokens = -1;
    std::int32_t hashed_tail_tokens = -1;
    // Without a KV group there is no KV boundary to record: a junction
    // exists only where cached KV outruns the state.
    bool kv_settled = std::ranges::none_of(match_order, [&](std::size_t i) { return !GroupIsState(i); });
    const auto visit = [&](std::int32_t page_tokens, std::int32_t tail_tokens, const ContentHash& published_hash) {
        const std::int32_t boundary = page_tokens + tail_tokens;
        if (tail_tokens > 0) {
            if (page_tokens != hashed_page_tokens || tail_tokens != hashed_tail_tokens) {
                const ContentHash prior =
                    page_tokens == 0
                        ? ContentHash{}
                        : prompt.prefix_hashes[static_cast<std::size_t>(page_tokens / prefix_granularity_) - 1];
                hash = HashPrefixPage(
                    prompt.tokens.subspan(static_cast<std::size_t>(page_tokens), static_cast<std::size_t>(tail_tokens)),
                    prior);
                hashed_page_tokens = page_tokens;
                hashed_tail_tokens = tail_tokens;
            }
            if (hash != published_hash) {
                return true;  // published by a prompt whose tokens diverge inside the page
            }
        }
        PrefixProbe::TailHit tail{.tail_tokens = tail_tokens,
                                  .hash = tail_tokens > 0 ? hash : ContentHash{},
                                  .group_keys = std::vector<std::vector<CacheKey>>(groups_.size()),
                                  .per_group = std::vector<GroupPrefixProbe>(groups_.size())};
        std::vector<GroupPrefixProbe> page_hits(groups_.size());
        bool kv_accepts = true;
        bool state_accepts = true;
        for (const std::size_t i : match_order) {
            if (tail_tokens > 0) {
                tail.group_keys[i] = tailKeysForGroup(i, hash, tail_tokens);
            }
            const bool accepts =
                groupAcceptsBoundary<Tier>(i, prompt.prefix_hashes, floor_tokens, page_tokens, tail_tokens,
                                           tail.group_keys[i], tail.per_group[i], &page_hits[i]);
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
                const std::int32_t covered = (page_tokens - floor_tokens) / geometry_[i].BlockGranularity();
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
    forEachCandidateBoundary(prompt, floor_tokens, from_tokens, visit);
    if (out.num_common_tokens == floor_tokens) {
        // Nothing beyond the floor: no group claims anything on this tier.
        for (GroupPrefixProbe& hits : out.per_group) {
            hits.hits.clear();
        }
        out.tail.reset();
    }
    return out;
}

CacheCoordinator::PrefixProbe CacheCoordinator::probePrefixWithOrder(const PromptPrefix& prompt,
                                                                     std::span<const std::size_t> match_order,
                                                                     bool claim_partial) const {
    _assert(prompt.prefix_hashes.size() <=
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / prefix_granularity_),
            "prefix length exceeds int32 token range");
    _assert(prompt.tokens.empty() || static_cast<std::int64_t>(prompt.tokens.size()) >= prompt.max_tokens,
            "the probe bound lies inside the supplied tokens");
    PrefixProbe out;
    out.group_keys = buildGroupKeys(prompt.prefix_hashes);
    out.device = probeTier<CacheTier::kDevice>(prompt, out.group_keys, match_order, /*floor_tokens=*/0, claim_partial);
    if (host_pool_ == nullptr) {
        return out;
    }
    // The Host tier starts where the Device's full pages end and replaces the
    // Device's result when it reaches further.
    out.host =
        probeTier<CacheTier::kHost>(prompt, out.group_keys, match_order, out.device.full_page_tokens, claim_partial);
    if (out.host.num_common_tokens > out.device.num_common_tokens) {
        // The Device keeps its full pages below the Host's floor; its tail and
        // its state hit (a shorter checkpoint) give way to the Host's.
        out.device.tail.reset();
        out.device.num_common_tokens = out.device.full_page_tokens;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (GroupIsState(i)) {
                out.device.per_group[i].hits.clear();
            }
        }
    } else {
        out.host = PrefixProbe::Tier{};
        out.host.per_group.resize(groups_.size());
        out.host.num_common_tokens = out.device.num_common_tokens;
        out.host.full_page_tokens = out.device.full_page_tokens;
        out.host.kv_tokens = out.device.kv_tokens;
    }
    if (enable_l3_storage_) {
        out.storage = probeTierWithKeys<CacheTier::kHost>(
            out.group_keys, match_order,
            std::min(static_cast<std::int32_t>(prompt.prefix_hashes.size()), prompt.max_tokens / prefix_granularity_),
            /*floor_tokens=*/out.host.full_page_tokens, /*with_storage_keys=*/true);
    }
    return out;
}

CacheCoordinator::PrefixProbe CacheCoordinator::ProbePromptPrefix(const PromptPrefix& prompt) const {
    return probePrefixWithOrder(prompt, match_order_, /*claim_partial=*/true);
}

CacheCoordinator::PrefixProbe CacheCoordinator::ProbePrefix(std::span<const std::string> content_hashes) const {
    return ProbePromptPrefix(PromptPrefix{
        .prefix_hashes = content_hashes,
        .tokens = {},
        .max_tokens = static_cast<std::int32_t>(content_hashes.size()) * prefix_granularity_,
    });
}

std::optional<CacheCoordinator::PrefetchPlan> CacheCoordinator::PlanPrefetch(const PrefixProbe& probe,
                                                                             std::int32_t min_pages) {
    _assert(min_pages >= 1, "a prefetch below one page is no prefetch");
    if (!enable_l3_storage_ || host_pool_ == nullptr || probe.storage.per_group.empty()) {
        return std::nullopt;
    }
    // The fill starts where the Host's full pages end; a Host tail beyond
    // that is dropped once the pages behind it have landed and the request
    // is probed again.
    const std::int32_t floor_tokens = probe.host.full_page_tokens;
    const std::int32_t extension_pages = (probe.storage.num_common_tokens - floor_tokens) / prefix_granularity_;
    if (extension_pages < min_pages) {
        return std::nullopt;
    }
    // The rows each page needs: a storage-tier hit the matcher asked for
    // whose object is not already on Host.
    struct PageRows {
        std::vector<std::uint32_t> groups;
        std::vector<CacheKey> keys;
    };
    std::vector<PageRows> pages(static_cast<std::size_t>(extension_pages));
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t blocks_per_page = BlocksPerBoundary(static_cast<std::int32_t>(i));
        const std::int32_t floor_blocks = floor_tokens / prefix_granularity_ * blocks_per_page;
        const GroupPrefixProbe& hits = probe.storage.per_group[i];
        for (std::int32_t page = 0; page < extension_pages; ++page) {
            for (std::int32_t b = page * blocks_per_page; b < (page + 1) * blocks_per_page; ++b) {
                const auto hit_index = static_cast<std::size_t>(b);
                if (hit_index >= hits.hits.size() || hits.hits[hit_index] == 0) {
                    continue;  // not needed by this group's matcher (a window's older pages)
                }
                const CacheKey& key = probe.group_keys[i][static_cast<std::size_t>(floor_blocks + b)];
                if (groups_[i].Index().Contains(*host_pool_, key)) {
                    continue;  // landed meanwhile: an ordinary Host hit at admission
                }
                _assert(storage_keys_.contains(key), "storage probe hit without a Host or L3 entry");
                pages[static_cast<std::size_t>(page)].groups.push_back(groups_[i].Id());
                pages[static_cast<std::size_t>(page)].keys.push_back(key);
            }
        }
    }
    // The threshold is decided before anything is acquired: a fill that would
    // stop short of min_pages must evict nothing for the attempt. Counted per
    // group against its free slots, its unpinned entries and the pool's empty
    // parents (conservative: another group's evictable parents are not
    // counted, so a plan this refuses might have fit by evicting them -- the
    // request then computes the pages, which never blocks it).
    {
        std::vector<std::int64_t> rows_by_group(groups_.size(), 0);
        for (std::int32_t page = 0; page < min_pages; ++page) {
            for (const std::uint32_t group_id : pages[static_cast<std::size_t>(page)].groups) {
                ++rows_by_group[group_id];
            }
        }
        std::int64_t parents_needed = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::int64_t evictable = groups_[i].Index().EvictableCandidates(*host_pool_).size();
            const std::vector<std::int64_t> need{std::max<std::int64_t>(0, rows_by_group[i] - evictable)};
            parents_needed +=
                ParentsNeededForBuckets(need, host_pool_->FreeSlotsByBucket(groups_[i].Id()),
                                        /*dense_need=*/0, groups_[i].Allocator().CacheBlocksPerLcmBlock());
        }
        if (parents_needed > host_pool_->NumEmptyLcmBlocks()) {
            return std::nullopt;
        }
    }
    PrefetchPlan plan{.first_page = floor_tokens / prefix_granularity_};
    for (std::int32_t page = 0; page < extension_pages; ++page) {
        // Every group's rows of the page are acquired in one batch (one Host
        // allocation pass per page, not per row); a shortfall leaves a row
        // empty, and the page -- its other blocks returning -- is not planned.
        // L3 is accepted only for replicated groups (one bucket), see
        // Validate; a prefetch destination has no Device counterpart whose
        // bucket it would have to follow.
        PageRows& rows = pages[static_cast<std::size_t>(page)];
        const std::vector<std::int32_t> buckets(rows.groups.size(), 0);
        HostAllocationBatch host_blocks = AcquireHostBlocks(rows.groups, buckets);
        if (host_blocks.stats.unallocated != 0) {
            break;  // the Host pool is pinned full: the fill stops before this page
        }
        for (std::size_t r = 0; r < rows.keys.size(); ++r) {
            plan.rows.push_back(PrefetchRow{.group_id = rows.groups[r],
                                            .key = std::move(rows.keys[r]),
                                            .host_block = std::move(host_blocks.blocks[r]),
                                            .page_index = plan.first_page + page});
        }
        plan.page_row_ends.push_back(plan.rows.size());
    }
    if (static_cast<std::int32_t>(plan.page_row_ends.size()) < min_pages || plan.rows.empty()) {
        return std::nullopt;  // the refs drop with the plan
    }
    return plan;
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
    probe.group_keys = buildGroupKeys(prompt.prefix_hashes);
    probe.device = probeTier<CacheTier::kDevice>(prompt, probe.group_keys, history_match_order, /*floor_tokens=*/0,
                                                 /*claim_partial=*/false);
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
    // The Host floor is the Device's full-page coverage: a Device tail exists
    // only when the Host tier adds nothing beyond it.
    const std::int32_t device_full_tokens = probe.device.full_page_tokens;
    // A hit keeps its tail young under the per-prior cap.
    for (const PrefixProbe::Tier* tier : {&probe.device, &probe.host}) {
        if (tier->tail && tier->tail->tail_tokens > 0) {
            tails_.Touch(tier->tail->hash);
        }
    }
    out.device = acquireTierWithKeys<CacheTier::kDevice>(probe.group_keys, /*floor_tokens=*/0, std::move(probe.device),
                                                         access_epoch);
    // The storage tier is never acquired: an admission loads Host entries
    // only, and PlanPrefetch is what turns L3 objects into Host entries first.
    if (host_pool_ != nullptr && !probe.host.per_group.empty()) {
        out.host = acquireTierWithKeys<CacheTier::kHost>(probe.group_keys, device_full_tokens, std::move(probe.host),
                                                         access_epoch);
    }
    if (probe.state_at_prompt_end) {
        out.state_at_prompt_end.resize(groups_.size());
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (probe.state_at_prompt_end->group_keys[i].empty()) {
                continue;
            }
            PrefixMatch match = groups_[i].Index().AcquireMatched(
                pool_, probe.state_at_prompt_end->group_keys[i],
                /*begin_blocks=*/0, probe.state_at_prompt_end->per_group[i], access_epoch);
            out.state_at_prompt_end[i] = std::move(match.blocks.front());
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
    // Parent ids are dense in [1, NumLcmBlocks], so a bitmap dedupes shared
    // prefixes without hashing every block reference of every live request.
    std::vector<bool> seen(static_cast<std::size_t>(pool_.NumLcmBlocks()) + 1, false);
    std::int32_t active = 0;
    for (std::span<const BlockTable> tables : request_tables) {
        for (const BlockTable& table : tables) {
            for (const CacheBlockRef& block_ref : table.Blocks()) {
                if (!block_ref) {
                    continue;
                }
                std::vector<bool>::reference parent_seen =
                    seen[static_cast<std::size_t>(block_ref->Location().lcm_block_id)];
                if (!parent_seen) {
                    parent_seen = true;
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
        // A tail hash holds fewer blocks than a page; Contains filters.
        for (CacheKey& key : keysForGroup(content_hashes, group.Id())) {
            if (group.Index().Contains(pool_, key)) {
                pending_stores_.push_back(StoreCandidate{.key = std::move(key)});
            }
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
                (completed.publish_state && hashed_pages > 0),
            "a publication point without new pages publishes a tail or a state checkpoint");
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

void CacheCoordinator::CacheCompletedBlocks(std::span<BlockTable> tables, const RequestProgress& progress,
                                            std::uint64_t access_epoch) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    _assert(progress.completed_pages.has_value(), "publication requires completed pages");
    _assert(!progress.completed_pages->owner_continues,
            "a publication outside an admission has no capacity for the owner's fresh block");
    validateProgress(progress);
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        cacheDeviceCompletedBlocksForGroup(i, tables[i], *progress.completed_pages, access_epoch);
    }
    recordPublishedTail(*progress.completed_pages);
}

template <CacheTier Tier>
void CacheCoordinator::cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table,
                                               std::span<const CacheKey> keys, std::int32_t first_cache_block,
                                               std::uint64_t access_epoch, CacheBoundaryKind boundary_kind,
                                               bool stream_completed_to_host) {
    if (GroupIsReplayable(static_cast<std::int32_t>(group_index))) {
        return;  // regenerated by the model, never published
    }
    std::vector<std::pair<CacheKey, CacheBlockRef>> newly_cached;
    const bool automatically_streams_to_host =
        stream_device_cache_to_host_ &&
        (groups_[group_index].Spec().attn_kind == AttnKind::kSlidingWindow || stream_completed_to_host);
    auto* inserted = [&]() -> std::vector<std::pair<CacheKey, CacheBlockRef>>* {
        if constexpr (Tier == CacheTier::kDevice) {
            return automatically_streams_to_host || cache_mutation_sink_ ? &newly_cached : nullptr;
        }
        return nullptr;
    }();
    CacheGroup& group = groups_[group_index];
    group.Index().RegisterFullBlocks(tierPool<Tier>(),
                                     group.Allocator().BlocksToPublish(table, first_cache_block, keys.size()), keys,
                                     access_epoch, first_cache_block, boundary_kind, inserted);
    if constexpr (Tier == CacheTier::kHost) {
        return;
    }
    for (auto& [key, block_ref] : newly_cached) {
        if (cache_mutation_sink_) {
            cache_mutation_sink_(key, CacheMutation::kStored);
        }
        if (!automatically_streams_to_host) {
            continue;
        }
        pending_stores_.push_back(StoreCandidate{
            .key = std::move(key),
        });
    }
}

CacheBlockRef CacheCoordinator::AcquireDeviceCachedBlock(const CacheKey& key) const {
    if (key.group_id >= groups_.size()) {
        return {};
    }
    return groups_[key.group_id].Index().Find(pool_, key);
}

CacheCoordinator::HostAllocationBatch CacheCoordinator::AcquireHostBlocks(std::span<const std::uint32_t> group_ids,
                                                                          std::span<const std::int32_t> buckets) {
    _assert(host_pool_ != nullptr, "AcquireHostBlocks requires a host pool");
    _assert(group_ids.size() == buckets.size(), "Host allocation needs one bucket per requested block");
    HostAllocationBatch batch;
    batch.blocks.resize(group_ids.size());
    batch.stats.requested = group_ids.size();
    _assert(group_ids.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "Host allocation batch request exceeds int32 range");

    // A Host block lands in the bucket of the Device block it mirrors, so under
    // page-cyclic sharding both ends of a copy have the same owner.
    const auto acquire_in_bucket = [&](std::uint32_t group_id, std::int32_t bucket) {
        const std::array<std::int32_t, 1> one{bucket};
        std::vector<CacheBlockRef> refs = host_pool_->AcquireBlocksInBuckets(group_id, one);
        return refs.empty() ? CacheBlockRef{} : std::move(refs.front());
    };

    std::vector<std::uint32_t> group_order;
    group_order.reserve(groups_.size());
    std::vector<bool> seen_group(groups_.size(), false);
    std::vector<std::vector<std::size_t>> unresolved_by_group(groups_.size());
    for (std::size_t i = 0; i < group_ids.size(); ++i) {
        _assert(group_ids[i] < groups_.size(), "Host block group id out of range");
        _assert(buckets[i] >= 0 && buckets[i] < groups_[group_ids[i]].Allocator().ShardCount(),
                "Host block bucket is outside the group's shard count");
        if (!seen_group[group_ids[i]]) {
            seen_group[group_ids[i]] = true;
            group_order.push_back(group_ids[i]);
        }
        // Free capacity first, in candidate order.
        batch.blocks[i] = acquire_in_bucket(group_ids[i], buckets[i]);
        if (!batch.blocks[i]) {
            unresolved_by_group[group_ids[i]].push_back(i);
        }
    }

    // Retention value of one Host cache entry: keep what a request has already
    // proven useful, then the most recently accessed, then a stable tie-break.
    const auto value = [](std::uint32_t candidate_group, CacheBlockLocation location,
                          const PrefixCacheIndex::CachedBlockMetadata& metadata) {
        return std::tuple{metadata.was_acquired, metadata.last_access_epoch, candidate_group, location.lcm_block_id,
                          location.slot_index};
    };
    const auto lookup_value = [&](std::uint32_t candidate_group, CacheBlockLocation location) {
        const auto metadata = groups_[candidate_group].Index().MetadataFor(*host_pool_, location);
        _assert(metadata.has_value(), "evictable Host block has no cache metadata");
        return value(candidate_group, location, *metadata);
    };
    using HostCacheValue = decltype(lookup_value(std::uint32_t{}, CacheBlockLocation{}));

    for (std::uint32_t group_id : group_order) {
        std::vector<std::size_t>& unresolved = unresolved_by_group[group_id];
        if (unresolved.empty()) {
            continue;
        }
        ++batch.stats.same_group_scans;
        // Host retention ranks was_acquired before epoch, unlike the eviction
        // index, so finding the best victims needs a full group scan; each
        // unresolved block then evicts the least valuable entry in its own
        // bucket, which is exactly the hole it needs.
        std::vector<PrefixCacheIndex::EvictionCandidate> local_victims =
            groups_[group_id].Index().EvictableCandidates(*host_pool_);
        std::ranges::sort(local_victims, {}, [&](const PrefixCacheIndex::EvictionCandidate& candidate) {
            return value(group_id, candidate.location, candidate.metadata);
        });
        const GroupAllocator& allocator = groups_[group_id].Allocator();
        std::vector<bool> consumed(local_victims.size(), false);
        std::erase_if(unresolved, [&](std::size_t index) {
            const std::int32_t bucket = buckets[index];
            for (std::size_t v = 0; v < local_victims.size(); ++v) {
                if (consumed[v] || allocator.BucketOf(local_victims[v].location) != bucket) {
                    continue;
                }
                consumed[v] = true;
                _assert(evictHostCachedBlock(group_id, local_victims[v].location),
                        "selected Host child is not evictable");
                batch.blocks[index] = acquire_in_bucket(group_id, bucket);
                _assert(static_cast<bool>(batch.blocks[index]), "evicting a Host child did not free its bucket");
                return true;
            }
            return false;
        });
    }
    const bool has_unresolved = std::ranges::any_of(
        unresolved_by_group, [](const std::vector<std::size_t>& unresolved) { return !unresolved.empty(); });
    if (has_unresolved) {
        ++batch.stats.cross_group_scans;
        std::vector<std::pair<HostCacheValue, std::int32_t>> victim_parents;
        victim_parents.reserve(static_cast<std::size_t>(host_pool_->NumLcmBlocks()));
        for (std::int32_t parent_id = 1; parent_id <= host_pool_->NumLcmBlocks(); ++parent_id) {
            const std::optional<std::uint32_t> bound_group = host_pool_->BoundGroup(parent_id);
            if (!bound_group ||
                !groups_[*bound_group].Index().ParentIsFullyEvictable(
                    *host_pool_, parent_id, groups_[*bound_group].Allocator().CacheBlocksPerLcmBlock())) {
                continue;
            }
            std::optional<HostCacheValue> parent_value;
            for (std::int32_t slot = 0; slot < groups_[*bound_group].Allocator().CacheBlocksPerLcmBlock(); ++slot) {
                const CacheBlockLocation location{.lcm_block_id = parent_id, .slot_index = slot};
                if (!host_pool_->IsOccupied(location)) {
                    continue;
                }
                const auto child_value = lookup_value(*bound_group, location);
                parent_value = parent_value ? std::max(*parent_value, child_value) : child_value;
            }
            _assert(parent_value.has_value(), "evictable Host parent has no children");
            victim_parents.emplace_back(*parent_value, parent_id);
        }
        std::ranges::sort(victim_parents);

        std::size_t victim_index = 0;
        for (std::size_t result_index = 0; result_index < group_ids.size() && victim_index < victim_parents.size();
             ++result_index) {
            if (batch.blocks[result_index]) {
                continue;
            }
            const std::uint32_t target_group = group_ids[result_index];
            std::vector<std::size_t>& unresolved = unresolved_by_group[target_group];
            while (!unresolved.empty() && victim_index < victim_parents.size()) {
                const std::int32_t victim_parent = victim_parents[victim_index++].second;
                const std::optional<std::uint32_t> bound_group = host_pool_->BoundGroup(victim_parent);
                if (!bound_group ||
                    !groups_[*bound_group].Index().ParentIsFullyEvictable(
                        *host_pool_, victim_parent, groups_[*bound_group].Allocator().CacheBlocksPerLcmBlock())) {
                    continue;
                }
                const std::int32_t victim_packing = groups_[*bound_group].Allocator().CacheBlocksPerLcmBlock();
                for (std::int32_t slot = 0; slot < victim_packing; ++slot) {
                    const CacheBlockLocation location{.lcm_block_id = victim_parent, .slot_index = slot};
                    if (host_pool_->IsOccupied(location)) {
                        _assert(evictHostCachedBlock(*bound_group, location),
                                "selected Host parent changed before eviction");
                    }
                }
                // The freed parent is the pool's only empty one, so each
                // bucket may take its share of it; the rest stay unresolved.
                const std::size_t before = unresolved.size();
                std::erase_if(unresolved, [&](std::size_t index) {
                    batch.blocks[index] = acquire_in_bucket(target_group, buckets[index]);
                    return static_cast<bool>(batch.blocks[index]);
                });
                _assert(unresolved.size() < before, "evicting a Host parent did not free a placement");
                break;
            }
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

CacheBlockRef CacheCoordinator::FindHostCachedBlock(const CacheKey& key) const {
    if (host_pool_ == nullptr) {
        return {};
    }
    _assert(key.group_id < groups_.size(), "host cache key group id out of range");
    return groups_[key.group_id].Index().Find(*host_pool_, key);
}

bool CacheCoordinator::evictCachedBlockAlone(std::uint32_t group_id, CacheBlockLocation location) {
    std::optional<CacheKey> removed = groups_[group_id].Index().Evict(pool_, location);
    if (!removed) {
        return false;
    }
    if (cache_mutation_sink_) {
        cache_mutation_sink_(*removed, CacheMutation::kRemoved);
    }
    pruneTailIfUnreferenced(removed->content_hash);
    return true;
}

bool CacheCoordinator::evictCachedBlock(std::uint32_t group_id, CacheBlockLocation location) {
    std::optional<CacheKey> removed = groups_[group_id].Index().Evict(pool_, location);
    if (!removed) {
        return false;
    }
    if (cache_mutation_sink_) {
        cache_mutation_sink_(*removed, CacheMutation::kRemoved);
    }
    if (const std::optional<std::int32_t> tail_tokens = tails_.TailTokensOf(removed->content_hash)) {
        evictTailEntries(removed->content_hash, *tail_tokens);
    } else if (!GroupIsState(group_id)) {
        // A state checkpoint is matched only with the KV below it: a KV block
        // of its page gone, it is dead weight and leaves too.
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (!GroupIsState(i)) {
                continue;
            }
            const CacheKey key{.group_id = groups_[i].Id(), .content_hash = removed->content_hash, .page_offset = 0};
            if (const std::optional<CacheBlockLocation> location = deviceLocationOf(i, key)) {
                evictCachedBlockAlone(groups_[i].Id(), *location);
            }
        }
    }
    pruneTailIfUnreferenced(removed->content_hash);
    return true;
}

bool CacheCoordinator::evictHostCachedBlock(std::uint32_t group_id, CacheBlockLocation location) {
    std::optional<CacheKey> removed = groups_[group_id].Index().Evict(*host_pool_, location);
    if (!removed) {
        return false;
    }
    pruneTailIfUnreferenced(removed->content_hash);
    return true;
}

std::optional<CacheBlockLocation> CacheCoordinator::deviceLocationOf(std::size_t group_index,
                                                                     const CacheKey& key) const {
    // The reference from Find must not outlive this lookup: an eviction that
    // follows requires the index to be the block's only owner.
    const CacheBlockRef block = groups_[group_index].Index().Find(pool_, key);
    return block ? std::optional<CacheBlockLocation>{block->Location()} : std::nullopt;
}

void CacheCoordinator::evictTailEntries(const ContentHash& hash, std::int32_t tail_tokens) {
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t blocks = TailBlocksFor(static_cast<std::int32_t>(i), tail_tokens);
        for (std::int32_t offset = 0; offset < blocks; ++offset) {
            const CacheKey key{.group_id = groups_[i].Id(), .content_hash = hash, .page_offset = offset};
            // A companion still pinned by a request table stays indexed; it
            // ages out through the ordinary eviction order.
            if (const std::optional<CacheBlockLocation> location = deviceLocationOf(i, key)) {
                evictCachedBlockAlone(groups_[i].Id(), *location);
            }
        }
    }
}

void CacheCoordinator::pruneTailIfUnreferenced(const ContentHash& hash) {
    if (!tails_.Contains(hash)) {
        return;
    }
    const std::int32_t tail_tokens = *tails_.TailTokensOf(hash);
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t blocks = TailBlocksFor(static_cast<std::int32_t>(i), tail_tokens);
        for (std::int32_t offset = 0; offset < blocks; ++offset) {
            const CacheKey key{.group_id = groups_[i].Id(), .content_hash = hash, .page_offset = offset};
            if (groups_[i].Index().Contains(pool_, key) ||
                (host_pool_ != nullptr && groups_[i].Index().Contains(*host_pool_, key))) {
                return;
            }
        }
    }
    tails_.Remove(hash);
}

void CacheCoordinator::recordPublishedTail(const CompletedPages& completed) {
    if (!completed.tail) {
        return;
    }
    const ContentHash prior = completed.prefix_hashes.empty() ? ContentHash{} : completed.prefix_hashes.back();
    if (const std::optional<TailTable::Tail> dropped =
            tails_.Add(prior, completed.tail->tail_tokens, completed.tail->hash)) {
        evictTailEntries(dropped->hash, dropped->tail_tokens);
    }
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
    const std::int32_t tail_blocks =
        completed.tail ? TailBlocksFor(static_cast<std::int32_t>(group_index), completed.tail->tail_tokens) : 0;
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
        // A natural checkpoint: slot 1 holds the state the forward that ended
        // at this point wrote, keyed like the point itself.
        if (!completed.publish_state || completed.prefix_hashes.empty() && !completed.tail) {
            return ranges;
        }
        const ContentHash hash = completed.tail ? completed.tail->hash : completed.prefix_hashes.back();
        ranges.push_back(PublishedRange{
            .first_cache_block = 1,
            .keys = {CacheKey{.group_id = groups_[group_index].Id(), .content_hash = hash, .page_offset = 0}},
            .boundary_kind = completed.boundary_kind,
        });
        return ranges;
    }
    // A sliding window resumes from the newest page boundary and, at an
    // unaligned point, from the tail as well: both publish the blocks holding
    // the window behind them.
    std::vector<std::int32_t> boundaries;
    if (completed.first_new_prefix_page < hashed_prefix_pages) {
        boundaries.push_back(hashed_prefix_pages * prefix_granularity_);
    }
    if (tail_blocks > 0) {
        boundaries.push_back(hashed_prefix_pages * prefix_granularity_ + completed.tail->tail_tokens);
    }
    for (const std::int32_t boundary_tokens : boundaries) {
        const BlockRange window = SlidingWindowBlocks(static_cast<std::int32_t>(group_index), boundary_tokens);
        if (window.first >= window.past_end) {
            continue;
        }
        ranges.push_back(history_range(window.first, window.past_end));
    }
    return ranges;
}

template <CacheTier Tier>
void CacheCoordinator::cacheCompletedBlocksForGroup(std::size_t group_index, BlockTable& table,
                                                    const CompletedPages& completed, std::uint64_t access_epoch) {
    for (const PublishedRange& range : completedBlockRanges(group_index, completed)) {
        cacheFullBlocksForGroup<Tier>(group_index, table, range.keys, range.first_cache_block, access_epoch,
                                      range.boundary_kind, completed.stream_completed_to_host);
    }
}

void CacheCoordinator::cacheDeviceCompletedBlocksForGroup(std::size_t group_index, BlockTable& table,
                                                          const CompletedPages& completed, std::uint64_t access_epoch) {
    cacheCompletedBlocksForGroup<CacheTier::kDevice>(group_index, table, completed, access_epoch);
}

void CacheCoordinator::ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Allocator().ReclaimExpired(pool_, tables[i],
                                              groupExpiredBlocksAt(static_cast<std::int32_t>(i), num_computed_tokens));
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
                                                  /*dense_need=*/0, allocator.CacheBlocksPerLcmBlock());
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
                slots.push_back(ImageSlot{.slot_index = slot,
                                          .block = blocks[index],
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

void CacheCoordinator::CacheHostBlock(CacheBlockRef& block_ref, const CacheKey& key) {
    _assert(host_pool_ != nullptr, "CacheHostBlock requires a host pool");
    _assert(key.group_id < groups_.size(), "CacheHostBlock group id out of range");
    groups_[key.group_id].Index().Register(*host_pool_, block_ref, key, ++next_access_epoch_,
                                           /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                                           /*newly_cached=*/nullptr);
    rememberStorageKey(key);
}

void CacheCoordinator::CacheDeviceBlock(CacheBlockRef& block_ref, const CacheKey& key, std::int32_t logical_block_index,
                                        CacheBoundaryKind boundary_kind) {
    _assert(static_cast<bool>(block_ref), "CacheDeviceBlock requires a destination block");
    _assert(key.group_id < groups_.size(), "CacheDeviceBlock group id out of range");
    std::vector<std::pair<CacheKey, CacheBlockRef>> newly_cached;
    groups_[key.group_id].Index().Register(pool_, block_ref, key, ++next_access_epoch_, logical_block_index,
                                           boundary_kind, &newly_cached);
    if (!cache_mutation_sink_) {
        return;
    }
    for (const auto& [cached_key, cached_block] : newly_cached) {
        (void)cached_block;
        cache_mutation_sink_(cached_key, CacheMutation::kStored);
    }
}

void CacheCoordinator::RegisterStorageKeys(std::span<const CacheKey> keys) {
    if (!enable_l3_storage_ || storage_key_limit_ == 0) {
        return;
    }
    std::vector<CacheKey> ordered(keys.begin(), keys.end());
    std::unordered_map<ContentHash, std::size_t> hash_index;
    for (const CacheKey& key : ordered) {
        hash_index.emplace(key.content_hash, hash_index.size());
        _assert(key.group_id < groups_.size(), "storage key group id out of range");
    }
    std::stable_sort(ordered.begin(), ordered.end(), [&hash_index](const CacheKey& left, const CacheKey& right) {
        const std::size_t left_hash = hash_index.at(left.content_hash);
        const std::size_t right_hash = hash_index.at(right.content_hash);
        if (left_hash != right_hash) {
            return left_hash < right_hash;
        }
        if (left.page_offset != right.page_offset) {
            return left.page_offset < right.page_offset;
        }
        return left.group_id < right.group_id;
    });
    std::unordered_set<CacheKey, CacheKeyHash> keep;
    for (const CacheKey& key : ordered) {
        if (keep.size() >= storage_key_limit_) {
            break;
        }
        keep.insert(key);
    }
    for (const CacheKey& key : ordered) {
        if (!keep.contains(key)) {
            storage_keys_.erase(key);
        }
    }
    compactStorageKeyOrder();
    for (const CacheKey& key : ordered) {
        if (!keep.contains(key)) {
            break;
        }
        if (storage_keys_.contains(key)) {
            continue;
        }
        while (storage_keys_.size() >= storage_key_limit_) {
            if (!evictOldestUnprotectedKey(keep)) {
                break;
            }
        }
        if (storage_keys_.size() >= storage_key_limit_) {
            break;
        }
        storage_keys_.insert(key);
        storage_key_order_.push_back(key);
    }
}

void CacheCoordinator::UnregisterStorageKeys(std::span<const CacheKey> keys) {
    for (const CacheKey& key : keys) {
        storage_keys_.erase(key);
    }
    compactStorageKeyOrder();
}

void CacheCoordinator::rememberStorageKey(const CacheKey& key) {
    if (!enable_l3_storage_ || storage_key_limit_ == 0) {
        return;
    }
    if (!storage_keys_.insert(key).second) {
        return;
    }
    storage_key_order_.push_back(key);
    evictStorageKeysToLimit();
}

void CacheCoordinator::evictStorageKeysToLimit() {
    compactStorageKeyOrder();
    while (storage_keys_.size() > storage_key_limit_) {
        if (storage_key_order_.empty()) {
            break;
        }
        storage_keys_.erase(storage_key_order_.front());
        storage_key_order_.pop_front();
    }
}

void CacheCoordinator::compactStorageKeyOrder() {
    std::erase_if(storage_key_order_, [this](const CacheKey& key) { return !storage_keys_.contains(key); });
}

bool CacheCoordinator::evictOldestUnprotectedKey(const std::unordered_set<CacheKey, CacheKeyHash>& protected_keys) {
    for (auto it = storage_key_order_.begin(); it != storage_key_order_.end();) {
        if (!storage_keys_.contains(*it)) {
            it = storage_key_order_.erase(it);
            continue;
        }
        if (protected_keys.contains(*it)) {
            ++it;
            continue;
        }
        storage_keys_.erase(*it);
        storage_key_order_.erase(it);
        return true;
    }
    return false;
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
                                 BlockPool* snapshot_pool, bool stream_device_cache_to_host) {
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
    return CacheCoordinator{std::move(groups), prefix_granularity,         pool, enable_l3_storage, host_pool,
                            snapshot_pool,     stream_device_cache_to_host};
}

}  // namespace tokenspeed
