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

#include "cache/coordinator/publication_preview.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "cache/coordinator/recovery_route.h"
#include "cache/prefix/prefix_hasher.h"

namespace tokenspeed {
PublicationPreview::PublicationPreview(const CacheStore& store, std::span<const GroupDemand> demands,
                                       const CheckpointPublication& publication)
    : tokens_{publication.tokens.begin(), publication.tokens.end()} {
    for (const auto& hash : publication.prefix_hashes) hashes_.emplace_back(hash);
    std::unordered_map<CacheKey, CacheBlockLocation, CacheKeyHash> produced;
    const auto add = [&](const CacheKey& key, const CacheBlockRef& block) {
        if (!block) return;  // A sliding window can already contain holes.
        produced.emplace(key, block->Location());
        publications_.push_back({key, CacheTier::kDevice, true});
        protected_.emplace_back(CacheTier::kDevice, block->Location());
        // Publication deduplicates onto this canonical copy during commit.
        if (auto canonical = store.LocationOf(store.DevicePool(), key.group_id, key))
            protected_.emplace_back(CacheTier::kDevice, *canonical);
    };
    for (std::size_t g = 0; g < demands.size(); ++g) {
        for (const auto& range : store.CompletedBlockRanges(g, publication)) {
            const auto blocks = demands[g].table->Blocks();
            for (std::size_t i = 0; i < range.keys.size(); ++i) {
                const auto slot = static_cast<std::size_t>(range.first_cache_block) + i;
                if (slot < blocks.size()) add(range.keys[i], blocks[slot]);
            }
        }
        if (store.GroupIsState(g) && publication.state_checkpoint_pages > 0 &&
            g < publication.state_checkpoint_blocks.size()) {
            add(CacheKey{.group_id = static_cast<std::uint32_t>(g),
                         .content_hash = hashes_[publication.state_checkpoint_pages - 1]},
                publication.state_checkpoint_blocks[g]);
        }
    }
    if (publication.state_checkpoint_pages > 0)
        points_.push_back({hashes_[publication.state_checkpoint_pages - 1],
                           CheckedInt32(std::int64_t{publication.state_checkpoint_pages} * store.PrefixGranularity())});
    if (publication.publish_state && (!hashes_.empty() || publication.tail))
        points_.push_back({publication.tail ? publication.tail->hash : hashes_.back(), CheckedInt32(tokens_.size())});
    const auto contains = [&](const CacheKey& key, CacheTier tier) {
        if (tier == CacheTier::kDevice && produced.contains(key)) return true;
        const auto* pool = tier == CacheTier::kDevice ? &store.DevicePool() : store.HostPool();
        return pool && store.Groups()[key.group_id].Index().Contains(*pool, key);
    };
    const auto key_at = [&](const Point& point, std::size_t group, std::int32_t block) {
        if (store.GroupIsState(group))
            return CacheKey{.group_id = static_cast<std::uint32_t>(group), .content_hash = point.hash};
        const auto per_page = store.BlocksPerBoundary(static_cast<std::int32_t>(group));
        return CacheKey{.group_id = static_cast<std::uint32_t>(group),
                        .content_hash = block / per_page < point.tokens / store.PrefixGranularity()
                                            ? hashes_[block / per_page]
                                            : point.hash,
                        .page_offset = block % per_page};
    };
    std::unordered_set<CacheKey, CacheKeyHash> known;
    for (const auto& point : points_) {
        const auto route =
            ResolveRecoveryRanges(store, point.tokens, store.HostPool() != nullptr, true,
                                  [&](std::size_t group, std::int32_t first, std::int32_t end, CacheTier tier) {
                                      while (first < end && contains(key_at(point, group, first), tier)) ++first;
                                      return first;
                                  });
        if (!route) continue;  // Incomplete output never becomes a usable fallback.
        RecoveryIndex::Entry entry{.ready = true, .device_ready = route->device_only};
        for (const auto& range : route->dependencies) {
            auto& paths = range.tier == CacheTier::kDevice ? entry.device_paths : entry.host_paths;
            paths.push_back({key_at(point, range.group, range.first), key_at(point, range.group, range.past_end - 1)});
            for (auto b = range.first; b < range.past_end; ++b) {
                const auto key = key_at(point, range.group, b);
                if (known.insert(key).second)
                    leaves_.push_back({key, !store.GroupIsState(range.group) && b > 0
                                                ? std::optional{key_at(point, range.group, b - 1)}
                                                : std::nullopt});
                const auto& pool = range.tier == CacheTier::kDevice ? store.DevicePool() : *store.HostPool();
                if (auto location = store.LocationOf(pool, range.group, key))
                    protected_.emplace_back(range.tier, *location);
            }
        }
        routes_.emplace(point.hash, std::move(entry));
    }
    std::erase_if(points_, [&](const auto& point) { return !routes_.contains(point.hash); });
}

std::vector<RecoveryIndex::Leaf> PublicationPreview::Leaves(const CacheStore& store) const {
    auto result = store.RecoveryDependencies().Leaves();
    std::unordered_set<CacheKey, CacheKeyHash> known;
    for (const auto& leaf : result) known.insert(leaf.key);
    for (const auto& leaf : leaves_)
        if (known.insert(leaf.key).second) result.push_back(leaf);
    return result;
}

std::vector<CheckpointTree::RecoveryPoint> PublicationPreview::Points(const CacheStore& store) const {
    auto result = store.Checkpoints().RecoveryPoints();
    std::unordered_set<ContentHash> existing;
    std::uint64_t sequence = 0;
    for (const auto& point : result) {
        existing.insert(point.hash);
        sequence = std::max(sequence, point.sequence);
    }
    const auto identity = [&](std::int32_t tokens) {
        const auto full = tokens / store.PrefixGranularity();
        const auto tail = tokens % store.PrefixGranularity();
        const auto prior = full > 0 ? hashes_[full - 1] : ContentHash{};
        return tail ? HashPrefixPage(std::span{tokens_}.subspan(tokens - tail, tail), prior) : prior;
    };
    for (const auto& preview : points_) {
        if (std::ranges::find(result, preview.hash, &CheckpointTree::RecoveryPoint::hash) != result.end()) continue;
        FatalCheck(sequence < std::numeric_limits<std::uint64_t>::max(), "publication sequence overflow");
        CheckpointTree::RecoveryPoint added{preview.hash,
                                            std::nullopt,
                                            preview.tokens,
                                            store.Checkpoints().Watermark(),
                                            store.Checkpoints().HostWatermark(),
                                            true,
                                            ++sequence};
        std::int32_t parent_tokens = 0;
        for (const auto& point : result)
            if (point.tokens < added.tokens && point.tokens > parent_tokens && identity(point.tokens) == point.hash) {
                added.parent = point.hash;
                parent_tokens = point.tokens;
            }
        for (auto& point : result) {
            if (point.tokens <= added.tokens || !existing.contains(point.hash) ||
                store.Checkpoints().PrefixIdentityAt(point.hash, added.tokens) != added.hash)
                continue;
            const auto parent =
                std::ranges::find(result, point.parent.value_or(ContentHash{}), &CheckpointTree::RecoveryPoint::hash);
            if (parent == result.end() || parent->tokens < added.tokens) point.parent = added.hash;
        }
        result.push_back(std::move(added));
    }
    return result;
}

const RecoveryIndex::Entry& PublicationPreview::At(const ContentHash& hash, const RecoveryIndex::View& view) const {
    const auto found = routes_.find(hash);
    return found == routes_.end() ? view.At(hash) : found->second;
}

RecoveryIndex::Entry PublicationPreview::Resolve(const ContentHash& hash, const RecoveryIndex::View& view,
                                                 bool allow_device_copy) const {
    if (auto found = routes_.find(hash); found != routes_.end()) return found->second;
    return view.Resolve(hash, allow_device_copy);
}
}  // namespace tokenspeed
