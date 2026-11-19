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

#include "cache/coordinator/storage_queries.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils.h"

namespace tokenspeed {

template <typename Visit>
void StorageQueries::forEachStorageDependency(const PrefixProbe::Tier& probe,
                                              std::span<const std::vector<CacheKey>> group_keys,
                                              std::int32_t floor_tokens, const Visit& visit) const {
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        // A State group's hits count from page 0 (PrefixProbe::Tier).
        const auto begin = store_.GroupIsState(i) ? 0
                                                  : floor_tokens / store_.PrefixGranularity() *
                                                        store_.BlocksPerBoundary(static_cast<std::int32_t>(i));
        for (std::size_t j = 0; j < probe.per_group[i].hits.size(); ++j) {
            if (probe.per_group[i].hits[j]) {
                visit(group_keys[i][static_cast<std::size_t>(begin) + j], begin + static_cast<std::int32_t>(j));
            }
        }
        if (!probe.tail) continue;
        const auto& tail = *probe.tail;
        const auto full_blocks =
            probe.full_page_tokens /
            (store_.GroupIsState(i) ? store_.PrefixGranularity() : store_.Geometry()[i].BlockGranularity());
        for (std::size_t j = 0; j < tail.per_group[i].hits.size(); ++j) {
            if (tail.per_group[i].hits[j]) {
                visit(tail.group_keys[i][j], full_blocks + static_cast<std::int32_t>(j));
            }
        }
    }
}

StorageQuery StorageQueries::SharedPrefixKeys(const PromptPrefix& prompt, std::size_t max_keys) const {
    StorageQuery result;
    if (prompt.max_tokens <= 0) return result;
    const auto full_tokens = prompt.max_tokens / store_.PrefixGranularity() * store_.PrefixGranularity();
    const auto tail_tokens = prompt.max_tokens - full_tokens;
    if (static_cast<std::size_t>(full_tokens / store_.PrefixGranularity()) > prompt.prefix_hashes.size() ||
        (tail_tokens && (!prompt.tail || prompt.tail->tail_tokens != tail_tokens)))
        return result;
    const auto add = [&](CacheKey key) {
        if (result.keys.size() == max_keys) return false;
        result.keys.push_back(std::move(key));
        return true;
    };
    for (const auto i : store_.MatchOrder()) {
        const auto& group = store_.Groups()[i];
        if (store_.GroupIsState(i)) {
            if (prompt.max_tokens % group.Spec().checkpoint_granularity != 0) return {};
            const ContentHash hash =
                tail_tokens ? prompt.tail->hash
                            : ContentHash{prompt.prefix_hashes[full_tokens / store_.PrefixGranularity() - 1]};
            if (!add(CacheKey{.group_id = group.Id(), .content_hash = hash, .page_offset = 0})) {
                result.max_tokens = -1;
                return result;
            }
            continue;
        }
        const auto grain = store_.Geometry()[i].BlockGranularity();
        const auto per_prefix = store_.PrefixGranularity() / grain;
        const auto full_blocks = full_tokens / grain;
        const auto first = group.Matcher().IsPrefixClosed()
                               ? 0
                               : store_.SlidingWindowBlocks(static_cast<std::int32_t>(i), prompt.max_tokens).first;
        for (auto block = first; block < full_blocks; ++block) {
            if (!add(CacheKey{.group_id = group.Id(),
                              .content_hash = ContentHash{prompt.prefix_hashes[block / per_prefix]},
                              .page_offset = block % per_prefix})) {
                result.max_tokens = -1;
                return result;
            }
        }
        if (tail_tokens) {
            const auto keys = store_.TailKeysForGroup(i, prompt.tail->hash, tail_tokens);
            for (std::size_t j = 0; j < keys.size(); ++j) {
                if (full_blocks + static_cast<std::int32_t>(j) < first) continue;
                if (!add(keys[j])) {
                    result.max_tokens = -1;
                    return result;
                }
            }
        }
    }
    result.max_tokens = prompt.max_tokens;
    return result;
}

StorageQuery StorageQueries::MissingStorageKeys(const PromptPrefix& prompt, std::size_t max_keys) const {
    _assert(prompt.max_tokens >= 0 &&
                static_cast<std::size_t>(prompt.max_tokens / store_.PrefixGranularity()) <= prompt.prefix_hashes.size(),
            "storage query bound exceeds the supplied prefix hashes");
    const PrefixProbe local = prober_.ProbePromptPrefix(prompt);
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
    prober_.ForEachCandidateBoundary(prompt, floor, prompt.max_tokens,
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
    for (const std::size_t i : store_.MatchOrder()) {
        const auto per_prefix = store_.BlocksPerBoundary(static_cast<std::int32_t>(i));
        const auto end = static_cast<std::size_t>(prompt.max_tokens / store_.PrefixGranularity() * per_prefix);
        for (std::size_t j = 0; j < end; ++j) possible.insert((*local.group_keys)[i][j]);
        for (const auto& candidate : candidates) {
            if (candidate.tail_tokens > 0) {
                for (const auto& key : store_.TailKeysForGroup(i, candidate.hash, candidate.tail_tokens))
                    possible.insert(key);
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
        const auto suffix = prober_.ProbeHostSuffix(candidate, *local.group_keys, store_.MatchOrder(), local_tokens,
                                                    &possible, local.device);
        const PrefixProbe::Tier& probe = suffix.host;
        _assert(probe.num_common_tokens == candidate.max_tokens, "an optimistic candidate must match completely");
        std::vector<CacheKey> additional;
        std::vector<std::int32_t> additional_tokens;
        std::unordered_set<CacheKey, CacheKeyHash> distinct;
        bool fits = true;
        forEachStorageDependency(
            probe, *local.group_keys, suffix.device.full_page_tokens,
            [&](const CacheKey& key, std::int32_t logical_block) {
                if (!fits || host_.ContainsHostCachedBlock(key) || selected.contains(key) ||
                    !distinct.insert(key).second)
                    return;
                if (additional.size() == max_keys - query.keys.size()) {
                    fits = false;
                    return;
                }
                additional.push_back(key);
                const auto grain = store_.GroupIsState(key.group_id)
                                       ? store_.PrefixGranularity()
                                       : store_.Geometry()[key.group_id].BlockGranularity();
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

StorageReservation StorageQueries::ReserveStoragePrefix(const PromptPrefix& prompt, std::span<const CacheKey> hits,
                                                        std::size_t max_blocks) {
    const std::unordered_set<CacheKey, CacheKeyHash> remote(hits.begin(), hits.end());
    // The Device-only probe: the reservation matches the Host suffix (Host and
    // remote) above its boundary again, under the same Host-suffix rule as
    // admission.
    const PreparedPrefixKeys keys = prober_.ResolvePreparedPrefixKeys(prompt);
    const auto& group_keys = *keys;
    const PrefixProbe::Tier device_only =
        prober_.ProbeTier(prompt, group_keys, store_.MatchOrder(), /*claim_partial=*/true);
    auto candidate = prompt;
    StorageReservation result;
    std::vector<CacheBlockRef> warm_blocks;
    std::vector<std::uint32_t> groups;
    std::vector<std::int32_t> buckets;
    while (candidate.max_tokens > device_only.num_common_tokens) {
        // Release rejected candidates' warm pins before the next probe.
        warm_blocks.clear();
        groups.clear();
        buckets.clear();
        result.keys.clear();
        result.blocks.clear();
        result.tail.reset();
        result.tail_prior = ContentHash{};
        const auto suffix = prober_.ProbeHostSuffix(candidate, group_keys, store_.MatchOrder(),
                                                    device_only.num_common_tokens, &remote, device_only);
        const PrefixProbe::Tier& probe = suffix.host;
        if (probe.num_common_tokens <= device_only.num_common_tokens) return {};
        const auto pin = [&](const CacheKey& key, std::int32_t logical_block) {
            auto ref = store_.AcquireHostCachedBlock(key);
            if (ref) {
                warm_blocks.push_back(std::move(ref));
            } else {
                result.keys.push_back(key);
                groups.push_back(key.group_id);
                buckets.push_back(logical_block % store_.Groups()[key.group_id].Spec().shard_count);
            }
        };
        // Protect warm dependencies only while allocating destinations. They
        // stay evictable during I/O; ordinary admission matches the cache again.
        forEachStorageDependency(probe, group_keys, suffix.device.full_page_tokens, pin);
        if (probe.tail) {
            result.tail_prior =
                probe.full_page_tokens == 0
                    ? ContentHash{}
                    : ContentHash{prompt.prefix_hashes[probe.full_page_tokens / store_.PrefixGranularity() - 1]};
            result.tail = TailPublication{.tail_tokens = probe.tail->tail_tokens, .hash = probe.tail->hash};
        }
        if (result.keys.empty()) return {};
        if (result.keys.size() + warm_blocks.size() <= max_blocks) {
            auto allocation = host_.AcquireHostBlocks(groups, buckets);
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
