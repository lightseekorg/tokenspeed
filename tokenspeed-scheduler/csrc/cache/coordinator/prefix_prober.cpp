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

#include "cache/coordinator/prefix_prober.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cache/prefix/prefix_hasher.h"
#include "cache/prefix/prefix_matcher.h"
#include "core/profile_span.h"
#include "utils.h"

namespace tokenspeed {

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

PreparedPrefixKeys PrefixProber::PreparePrefixKeys(std::span<const std::string> content_hashes) const {
    ProfileSpan profile_span{"PrefixProber::PreparePrefixKeys"};
    return std::make_shared<const std::vector<std::vector<CacheKey>>>(buildGroupKeys(content_hashes));
}

PrefixProbe PrefixProber::ProbePromptPrefix(const PromptPrefix& prompt) const {
    ProfileSpan profile_span{"PrefixProber::ProbePromptPrefix"};
    return probePrefixWithOrder(prompt, store_.MatchOrder(), /*claim_partial=*/true);
}

bool PrefixProber::EvictionStoresExtendPrefix(const PromptPrefix& prompt, const PrefixProbe& visible) const {
    const std::int32_t hit = std::max(visible.device.num_common_tokens, visible.host.num_common_tokens);
    if (store_.HostPool() == nullptr || evictor_.PendingKeys().empty() || prompt.max_tokens <= hit) {
        return false;
    }
    const HostSuffixProbe suffix =
        ProbeHostSuffix(prompt, *visible.group_keys, store_.MatchOrder(), hit, &evictor_.PendingKeys(), visible.device);
    return suffix.host.num_common_tokens > hit;
}

std::vector<PrefixProber::PrefixBoundaryAvailability> PrefixProber::InspectPrefixAvailability(
    const PromptPrefix& prompt) const {
    _assert(store_.Groups().size() <= 64, "prefix availability uses one bit per cache group");
    const auto prepared_keys = ResolvePreparedPrefixKeys(prompt);
    const auto& keys = *prepared_keys;
    const auto contains = [&](std::size_t i, const CacheKey& key, std::size_t tier) {
        const PrefixCacheIndex& index = store_.Groups()[i].Index();
        const bool device = tier != 1 && index.Contains(store_.DevicePool(), key);
        const bool host = tier != 0 && store_.HostPool() != nullptr && index.Contains(*store_.HostPool(), key);
        return device || host;
    };
    // A closed group's first hole bounds every later boundary. Compute it
    // once; rescanning the entire prefix per boundary would be quadratic.
    std::vector<std::array<std::int32_t, 3>> first_missing(store_.Groups().size());
    for (const std::size_t i : store_.MatchOrder()) {
        if (!store_.Groups()[i].Matcher().IsPrefixClosed()) {
            continue;
        }
        const auto block_limit = prompt.prefix_hashes.size() *
                                 static_cast<std::size_t>(store_.BlocksPerBoundary(static_cast<std::int32_t>(i)));
        for (std::size_t tier = 0; tier < 3; ++tier) {
            std::int32_t block = 0;
            while (static_cast<std::size_t>(block) < block_limit &&
                   contains(i, keys[i][static_cast<std::size_t>(block)], tier)) {
                ++block;
            }
            first_missing[i][tier] = block;
        }
    }
    std::vector<PrefixBoundaryAvailability> out;
    ForEachCandidateBoundary(
        prompt, /*floor_tokens=*/0, prompt.max_tokens,
        [&](std::int32_t page_tokens, std::int32_t tail_tokens, const ContentHash& hash) {
            std::array<std::uint64_t, 3> missing{};
            const std::int32_t boundary = page_tokens + tail_tokens;
            for (const std::size_t i : store_.MatchOrder()) {
                const auto tail_keys =
                    tail_tokens > 0 ? store_.TailKeysForGroup(i, hash, tail_tokens) : std::vector<CacheKey>{};
                for (std::size_t tier = 0; tier < 3; ++tier) {
                    bool available = true;
                    if (store_.GroupIsState(i)) {
                        const CacheKey key =
                            tail_tokens > 0
                                ? tail_keys[0]
                                : CacheKey{.group_id = store_.Groups()[i].Id(),
                                           .content_hash = ContentHash{prompt.prefix_hashes[static_cast<std::size_t>(
                                               page_tokens / store_.PrefixGranularity() - 1)]},
                                           .page_offset = 0};
                        available = contains(i, key, tier);
                    } else {
                        const std::int32_t granularity = store_.Geometry()[i].BlockGranularity();
                        const std::int32_t full_blocks = page_tokens / granularity;
                        if (store_.Groups()[i].Matcher().IsPrefixClosed()) {
                            available = first_missing[i][tier] >= full_blocks;
                            for (const CacheKey& key : tail_keys) {
                                available = available && contains(i, key, tier);
                            }
                        } else {
                            const CacheStore::BlockRange window =
                                store_.SlidingWindowBlocks(static_cast<std::int32_t>(i), boundary);
                            for (const CacheStore::BlockRange& needed :
                                 {store_.SinkBlocksBelow(static_cast<std::int32_t>(i), window), window}) {
                                for (std::int32_t block = needed.first; available && block < needed.past_end; ++block) {
                                    const CacheKey& key =
                                        block < full_blocks ? keys[i][static_cast<std::size_t>(block)]
                                                            : tail_keys[static_cast<std::size_t>(block - full_blocks)];
                                    available = contains(i, key, tier);
                                }
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

PrefixProbe PrefixProber::ProbePrefix(std::span<const std::string> content_hashes) const {
    return ProbePromptPrefix(PromptPrefix{
        .prefix_hashes = content_hashes,
        .tokens = {},
        .max_tokens = CheckedInt32(static_cast<std::int64_t>(content_hashes.size()) * store_.PrefixGranularity(),
                                   "a probed prefix exceeds int32 token positions"),
    });
}

PrefixProbe PrefixProber::ProbeDecodeDevicePrefix(const PromptPrefix& prompt, std::int32_t prompt_end_tokens) const {
    ProfileSpan profile_span{"PrefixProber::ProbeDecodeDevicePrefix"};
    _assert(prompt.tokens.empty() || prompt_end_tokens <= static_cast<std::int32_t>(prompt.tokens.size()),
            "the prompt end lies inside the supplied tokens");
    std::vector<std::size_t> history_match_order;
    history_match_order.reserve(store_.MatchOrder().size());
    for (std::size_t group_index : store_.MatchOrder()) {
        if (!store_.GroupIsState(group_index)) {
            history_match_order.push_back(group_index);
        }
    }
    // The D role probes the Device alone: the peer's transfer, not a Host
    // load, fills what the Device lacks. The partial block of a KV tail is
    // left a hole for the transfer to fill whole.
    PrefixProbe probe;
    probe.group_keys = ResolvePreparedPrefixKeys(prompt);
    probe.device = ProbeTier(prompt, *probe.group_keys, history_match_order, /*claim_partial=*/false);
    // The peer's final state supersedes any shorter checkpoint, so a State
    // group's checkpoint counts only at exactly the prompt end: then the
    // transfer omits the group and the D claims the checkpoint as slot 0.
    PrefixProbe::TailHit state_hit{.tail_tokens = prompt_end_tokens % store_.PrefixGranularity(),
                                   .group_keys = std::vector<std::vector<CacheKey>>(store_.Groups().size()),
                                   .per_group = std::vector<GroupPrefixProbe>(store_.Groups().size())};
    const std::int32_t page_tokens = prompt_end_tokens - state_hit.tail_tokens;
    const auto pages = static_cast<std::size_t>(page_tokens / store_.PrefixGranularity());
    if (prompt.tokens.empty() || pages > prompt.prefix_hashes.size()) {
        return probe;  // nothing to match: the prefix cache is off, or the last page is not hashed
    }
    const ContentHash prior = pages == 0 ? ContentHash{} : ContentHash{prompt.prefix_hashes[pages - 1]};
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
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        if (!store_.GroupIsState(i)) {
            continue;
        }
        state_hit.group_keys[i] = {
            CacheKey{.group_id = store_.Groups()[i].Id(), .content_hash = state_hit.hash, .page_offset = 0}};
        if (!store_.Groups()[i].Index().Contains(store_.DevicePool(), state_hit.group_keys[i].front())) {
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

PreparedPrefixKeys PrefixProber::ResolvePreparedPrefixKeys(const PromptPrefix& prompt) const {
    if (!prompt.prepared_keys) {
        return PreparePrefixKeys(prompt.prefix_hashes);
    }
    const auto& keys = *prompt.prepared_keys;
    _assert(keys.size() == store_.Groups().size(), "prepared prefix keys must cover every cache group");
    for (const auto i : store_.MatchOrder()) {
        const auto blocks_per_boundary =
            static_cast<std::size_t>(store_.BlocksPerBoundary(static_cast<std::int32_t>(i)));
        _assert(keys[i].size() / blocks_per_boundary >= prompt.prefix_hashes.size(),
                "prepared prefix keys do not cover the prompt hash chain");
    }
    return prompt.prepared_keys;
}

std::vector<std::vector<CacheKey>> PrefixProber::buildGroupKeys(std::span<const std::string> content_hashes) const {
    std::vector<std::vector<CacheKey>> group_keys(store_.Groups().size());
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        // A replayable group is never matched (store_.MatchOrder() leaves it out)
        // and holds no hit on either tier, so no probe reads its keys. They
        // would be most of the keys: a 2-token window group expands every
        // prefix page into prefix_granularity / 2 of them.
        if (store_.GroupIsReplayable(static_cast<std::int32_t>(i))) {
            continue;
        }
        group_keys[i] = store_.KeysForGroup(content_hashes, store_.Groups()[i].Id());
    }
    return group_keys;
}

// The aligned phase of one tier probe, over the token-spanned groups of
// match_order: slots below floor_tokens are assumed valid in a lower tier;
// per_group blocks are relative to the floor, num_common_tokens is the
// absolute converged page boundary in TOKENS. num_prefix_pages bounds the walk
// in prefix pages; each group key vector is expanded to that group's
// block_granularity. State groups take no part in this phase; ProbeTier
// checks State and sliding-window requirements at each candidate boundary.
template <CacheTier Tier>
PrefixProbe::Tier PrefixProber::probeTierWithKeys(std::span<const std::vector<CacheKey>> group_keys,
                                                  std::span<const std::size_t> match_order,
                                                  std::int32_t num_prefix_pages, std::int32_t floor_tokens,
                                                  const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits) const {
    const BlockPool& pool = store_.TierPool<Tier>();
    PrefixProbe::Tier out;
    out.num_common_tokens = floor_tokens;
    out.full_page_tokens = floor_tokens;
    out.kv_tokens = floor_tokens;
    out.per_group.resize(store_.Groups().size());
    std::vector<std::size_t> spanned_order;
    for (const std::size_t i : match_order) {
        if (!store_.GroupIsState(i)) {
            spanned_order.push_back(i);
        }
    }
    if (spanned_order.empty()) {
        return out;
    }
    const std::int32_t common_tokens = SweepThenConverge(
        spanned_order, store_.Groups(), num_prefix_pages * store_.PrefixGranularity(), store_.PrefixGranularity(),
        [&](std::size_t i, std::int32_t bound_tokens) {
            const std::int32_t group_block_granularity = store_.Geometry()[i].BlockGranularity();
            out.per_group[i] = store_.Groups()[i].Matcher().Probe(store_.Groups()[i].Index(), pool, group_keys[i],
                                                                  floor_tokens / group_block_granularity,
                                                                  bound_tokens / group_block_granularity, extra_hits);
        },
        [&](std::size_t i) {
            return floor_tokens +
                   static_cast<std::int32_t>(out.per_group[i].hits.size()) * store_.Geometry()[i].BlockGranularity();
        });

    // A matcher can find a q-aligned resume point above the final P-aligned
    // boundary. Only acquire the CacheBlocks covered by the shared boundary.
    for (const std::size_t i : spanned_order) {
        GroupPrefixProbe& probe = out.per_group[i];
        const std::int32_t covered_blocks = (common_tokens - floor_tokens) / store_.Geometry()[i].BlockGranularity();
        if (static_cast<std::int32_t>(probe.hits.size()) > covered_blocks) {
            probe.hits.resize(static_cast<std::size_t>(covered_blocks));
        }
    }
    out.num_common_tokens = common_tokens;
    out.full_page_tokens = common_tokens;
    out.kv_tokens = common_tokens;
    return out;
}

// Whether group i can resume at the boundary full_tokens + tail_tokens on
// this tier, and the hits over the boundary's keys: a closed group needs
// every block of the tail; a sliding group the blocks covering its lookback
// behind the boundary (tail blocks, and page blocks above floor_tokens --
// blocks below the floor are a lower tier's); a State group the one state
// published at the boundary, its page hit counted from page 0 whatever the
// floor (a Host checkpoint may sit on the Device's last full page).
template <CacheTier Tier>
PrefixProber::GroupBoundaryProbe PrefixProber::probeGroupBoundary(
    std::size_t i, std::span<const std::string> prefix_hashes, std::int32_t floor_tokens, std::int32_t full_tokens,
    std::int32_t tail_tokens, std::span<const CacheKey> tail_keys, bool include_page_hits, GroupBoundaryHits hits,
    const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits) const {
    const bool accepts = [&] {
        const BlockPool& pool = store_.TierPool<Tier>();
        const CacheGroup& group = store_.Groups()[i];
        const PrefixCacheIndex& index = group.Index();
        const auto contains = [&](const CacheKey& key) {
            return index.Contains(pool, key) || (extra_hits != nullptr && extra_hits->contains(key));
        };
        hits.tail.hits.assign(tail_keys.size(), 0);
        if (store_.GroupIsState(i)) {
            if ((full_tokens + tail_tokens) % group.Spec().checkpoint_granularity != 0) {
                return false;
            }
            if (tail_tokens > 0) {
                _assert(tail_keys.size() == 1, "a State group has one key per boundary");
                if (!contains(tail_keys[0])) {
                    return false;
                }
                hits.tail.hits[0] = 1;
                return true;
            }
            // An aligned boundary is keyed by the chain hash of its last page.
            _assert(full_tokens > 0 && full_tokens % store_.PrefixGranularity() == 0, "a page boundary past page 0");
            const auto page = static_cast<std::size_t>(full_tokens / store_.PrefixGranularity()) - 1;
            const CacheKey key{
                .group_id = group.Id(), .content_hash = ContentHash{prefix_hashes[page]}, .page_offset = 0};
            if (!contains(key)) {
                return false;
            }
            if (include_page_hits) {
                hits.pages.hits.assign(page + 1, 0);
                hits.pages.hits.back() = 1;
            }
            return true;
        }
        if (group.Matcher().IsPrefixClosed()) {
            for (std::size_t j = 0; j < tail_keys.size(); ++j) {
                if (!contains(tail_keys[j])) {
                    return false;
                }
                hits.tail.hits[j] = 1;
            }
            return true;
        }
        // A sliding window resumes at T when the blocks holding rows
        // [T - W + 1, T) are cached, and with a sink the blocks holding it.
        const std::int32_t block_granularity = store_.Geometry()[i].BlockGranularity();
        const std::int32_t blocks_per_page = store_.PrefixGranularity() / block_granularity;
        const std::int32_t floor_blocks = floor_tokens / block_granularity;
        const std::int32_t full_blocks = full_tokens / block_granularity;
        const CacheStore::BlockRange window =
            store_.SlidingWindowBlocks(static_cast<std::int32_t>(i), full_tokens + tail_tokens);
        _assert(window.past_end == full_blocks + static_cast<std::int32_t>(tail_keys.size()),
                "the tail's keys end where the window does");
        if (include_page_hits) {
            hits.pages.hits.assign(static_cast<std::size_t>(std::max(full_blocks - floor_blocks, 0)), 0);
        }
        const auto holds = [&](std::int32_t block) {
            if (block >= full_blocks) {
                const auto j = static_cast<std::size_t>(block - full_blocks);
                if (!contains(tail_keys[j])) {
                    return false;
                }
                hits.tail.hits[j] = 1;
                return true;
            }
            if (block < floor_blocks) {
                return true;
            }
            const CacheKey page_key{
                .group_id = group.Id(),
                .content_hash = ContentHash{prefix_hashes[static_cast<std::size_t>(block / blocks_per_page)]},
                .page_offset = block % blocks_per_page};
            if (!contains(page_key)) {
                return false;
            }
            if (include_page_hits) {
                hits.pages.hits[static_cast<std::size_t>(block - floor_blocks)] = 1;
            }
            return true;
        };
        for (const CacheStore::BlockRange& needed :
             {store_.SinkBlocksBelow(static_cast<std::int32_t>(i), window), window}) {
            for (std::int32_t block = needed.first; block < needed.past_end; ++block) {
                if (!holds(block)) {
                    return false;
                }
            }
        }
        return true;
    }();
    return GroupBoundaryProbe{.accepts = accepts, .hits = std::move(hits)};
}

bool PrefixProber::stateGroupsAcceptBoundary(std::span<const std::size_t> match_order,
                                             std::span<const std::string> prefix_hashes, std::int32_t full_tokens,
                                             std::int32_t tail_tokens, const ContentHash& tail_hash, bool allow_host,
                                             const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits) const {
    CacheKey tail_key{.content_hash = tail_hash};
    GroupBoundaryHits hits;
    for (const std::size_t i : match_order) {
        if (!store_.GroupIsState(i)) {
            continue;
        }
        tail_key.group_id = store_.Groups()[i].Id();
        const std::span<const CacheKey> keys =
            tail_tokens > 0 ? std::span<const CacheKey>{&tail_key, 1} : std::span<const CacheKey>{};
        auto probe = probeGroupBoundary<CacheTier::kDevice>(i, prefix_hashes, /*floor_tokens=*/0, full_tokens,
                                                            tail_tokens, keys, /*include_page_hits=*/false,
                                                            std::move(hits), /*extra_hits=*/nullptr);
        hits = std::move(probe.hits);
        if (probe.accepts) continue;
        if (!allow_host) return false;
        probe = probeGroupBoundary<CacheTier::kHost>(i, prefix_hashes, /*floor_tokens=*/0, full_tokens, tail_tokens,
                                                     keys, /*include_page_hits=*/false, std::move(hits), extra_hits);
        hits = std::move(probe.hits);
        if (!probe.accepts) return false;
    }
    return true;
}

// The Device's probe: the aligned phase over the prefix-closed groups, then
// the candidate phase -- longest first, a tail inside the page after the
// closed coverage, the closed coverage itself, then every published tail and
// page boundary below -- until every matched group accepts one. A closed
// group's acceptance is monotone in T, so its page walk bounds the candidates;
// a sliding group's is not (it can hold the lookback behind one boundary and
// miss it behind the one below), so sliding and State groups are judged per
// candidate by probeGroupBoundary. Without State groups the first candidate
// the KV groups accept wins; with them the walk descends to the longest
// published state checkpoint. kv_tokens records the KV groups' own boundary
// for the junction cut.
PrefixProbe::Tier PrefixProber::ProbeTier(const PromptPrefix& prompt, std::span<const std::vector<CacheKey>> group_keys,
                                          std::span<const std::size_t> match_order, bool claim_partial) const {
    const std::int32_t num_prefix_pages = std::min(static_cast<std::int32_t>(prompt.prefix_hashes.size()),
                                                   prompt.max_tokens / store_.PrefixGranularity());
    std::vector<std::size_t> closed_order;
    for (const std::size_t i : match_order) {
        if (store_.Groups()[i].Matcher().IsPrefixClosed()) {
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
    bool kv_settled = std::ranges::none_of(match_order, [&](std::size_t i) { return !store_.GroupIsState(i); });
    PrefixProbe::TailHit tail{.group_keys = std::vector<std::vector<CacheKey>>(store_.Groups().size()),
                              .per_group = std::vector<GroupPrefixProbe>(store_.Groups().size())};
    std::vector<GroupPrefixProbe> page_hits(store_.Groups().size());
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
                tail.group_keys[i] = store_.TailKeysForGroup(i, published_hash, tail_tokens);
            }
            auto probe = probeGroupBoundary<CacheTier::kDevice>(
                i, prompt.prefix_hashes, /*floor_tokens=*/0, page_tokens, tail_tokens, tail.group_keys[i],
                /*include_page_hits=*/true,
                GroupBoundaryHits{.tail = std::move(tail.per_group[i]), .pages = std::move(page_hits[i])},
                /*extra_hits=*/nullptr);
            const bool accepts = probe.accepts;
            tail.per_group[i] = std::move(probe.hits.tail);
            page_hits[i] = std::move(probe.hits.pages);
            // Once KV reach is known, a rejected boundary contributes nothing.
            if (!accepts && kv_settled) {
                return true;
            }
            (store_.GroupIsState(i) ? state_accepts : kv_accepts) =
                accepts && (store_.GroupIsState(i) ? state_accepts : kv_accepts);
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
            if (store_.GroupIsState(i)) {
                out.per_group[i] = tail_tokens > 0 ? GroupPrefixProbe{} : std::move(page_hits[i]);
                continue;
            }
            if (store_.Groups()[i].Matcher().IsPrefixClosed()) {
                // The aligned walk covered every page below the boundary.
                const std::int32_t covered = page_tokens / store_.Geometry()[i].BlockGranularity();
                out.per_group[i].hits.resize(static_cast<std::size_t>(covered));
            } else {
                out.per_group[i] = std::move(page_hits[i]);
            }
        }
        if (tail_tokens > 0) {
            if (!claim_partial) {
                for (const std::size_t i : match_order) {
                    if (!store_.GroupIsState(i) && tail_tokens % store_.Geometry()[i].BlockGranularity() != 0 &&
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
        closed_order.empty()
            ? prompt.max_tokens
            : static_cast<std::int32_t>(std::min<std::int64_t>(
                  std::int64_t{aligned.num_common_tokens} + store_.PrefixGranularity() - 1, prompt.max_tokens));
    ForEachCandidateBoundary(prompt, /*floor_tokens=*/0, from_tokens, visit);
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
PrefixProber::HostSuffixProbe PrefixProber::ProbeHostSuffix(
    const PromptPrefix& prompt, std::span<const std::vector<CacheKey>> group_keys,
    std::span<const std::size_t> match_order, std::int32_t above_tokens,
    const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits, PrefixProbe::Tier device) const {
    PrefixProbe::Tier host;
    const std::int32_t num_prefix_pages = std::min(static_cast<std::int32_t>(prompt.prefix_hashes.size()),
                                                   prompt.max_tokens / store_.PrefixGranularity());
    std::vector<std::size_t> closed_order;
    for (const std::size_t i : match_order) {
        if (store_.Groups()[i].Matcher().IsPrefixClosed()) {
            closed_order.push_back(i);
        }
    }
    const bool has_kv = std::ranges::any_of(match_order, [&](std::size_t i) { return !store_.GroupIsState(i); });
    // A Host window may complete Device Full KV even when no window on the
    // Device can resume there. Start from the closed groups' coverage; each
    // candidate below lowers the split for its missing Device window blocks.
    const std::int32_t closed_device_tokens = closed_order.empty() ? device.kv_tokens : device.closed_kv_tokens;
    const std::int32_t device_kv_tokens =
        std::min(closed_device_tokens - closed_device_tokens % store_.PrefixGranularity(),
                 num_prefix_pages * store_.PrefixGranularity());
    const std::int32_t host_kv_end_tokens =
        probeTierWithKeys<CacheTier::kHost>(group_keys, closed_order, num_prefix_pages, device_kv_tokens, extra_hits)
            .num_common_tokens;
    host = PrefixProbe::Tier{};
    host.per_group.resize(store_.Groups().size());
    bool kv_settled = !has_kv;
    const auto visit = [&](std::int32_t page_tokens, std::int32_t tail_tokens, const ContentHash& published_hash) {
        const std::int32_t boundary = page_tokens + tail_tokens;
        if (kv_settled && !stateGroupsAcceptBoundary(match_order, prompt.prefix_hashes, page_tokens, tail_tokens,
                                                     published_hash, /*allow_host=*/true, extra_hits)) {
            return true;
        }
        std::int32_t split_tokens = std::min(device_kv_tokens, page_tokens);
        for (const std::size_t i : match_order) {
            if (store_.GroupIsState(i) || store_.Groups()[i].Matcher().IsPrefixClosed()) {
                continue;
            }
            const std::int32_t block_granularity = store_.Geometry()[i].BlockGranularity();
            const CacheStore::BlockRange window = store_.SlidingWindowBlocks(static_cast<std::int32_t>(i), boundary);
            // Lowest first: a missing sink block lowers the split under the window.
            for (const CacheStore::BlockRange& needed :
                 {store_.SinkBlocksBelow(static_cast<std::int32_t>(i), window), window}) {
                for (std::int32_t block = needed.first;
                     block < std::min(needed.past_end, split_tokens / block_granularity); ++block) {
                    if (!store_.Groups()[i].Index().Contains(store_.DevicePool(),
                                                             group_keys[i][static_cast<std::size_t>(block)])) {
                        split_tokens =
                            block * block_granularity - block * block_granularity % store_.PrefixGranularity();
                        break;
                    }
                }
            }
        }
        // C covers the closed pages from L; a split lowered below L needs
        // the Host's closed pages up to L as well.
        for (const std::size_t i : closed_order) {
            const std::int32_t block_granularity = store_.Geometry()[i].BlockGranularity();
            for (std::int32_t block = split_tokens / block_granularity;
                 block < std::min(device_kv_tokens, page_tokens) / block_granularity; ++block) {
                if (!PrefixKeyIsHit(store_.Groups()[i].Index(), *store_.HostPool(),
                                    group_keys[i][static_cast<std::size_t>(block)], extra_hits)) {
                    return true;
                }
            }
        }
        const auto make_tail = [&] {
            return PrefixProbe::TailHit{.tail_tokens = tail_tokens,
                                        .hash = published_hash,
                                        .group_keys = std::vector<std::vector<CacheKey>>(store_.Groups().size()),
                                        .per_group = std::vector<GroupPrefixProbe>(store_.Groups().size())};
        };
        PrefixProbe::TailHit device_tail = make_tail();
        PrefixProbe::TailHit host_tail = make_tail();
        std::vector<GroupPrefixProbe> device_hits(store_.Groups().size());
        std::vector<GroupPrefixProbe> host_hits(store_.Groups().size());
        // The KV tail comes whole from one tier.
        bool kv_tail_on_device = has_kv && tail_tokens > 0 && split_tokens == page_tokens;
        for (const std::size_t i : match_order) {
            if (store_.GroupIsState(i) || !kv_tail_on_device) {
                continue;
            }
            device_tail.group_keys[i] = store_.TailKeysForGroup(i, published_hash, tail_tokens);
            auto probe = probeGroupBoundary<CacheTier::kDevice>(
                i, prompt.prefix_hashes, split_tokens, page_tokens, tail_tokens, device_tail.group_keys[i],
                /*include_page_hits=*/false, GroupBoundaryHits{.tail = std::move(device_tail.per_group[i])},
                /*extra_hits=*/nullptr);
            kv_tail_on_device = probe.accepts;
            device_tail.per_group[i] = std::move(probe.hits.tail);
        }
        bool kv_accepts = true;
        for (const std::size_t i : match_order) {
            if (store_.GroupIsState(i)) {
                continue;
            }
            if (!kv_tail_on_device) {
                device_tail.group_keys[i].clear();
                device_tail.per_group[i] = GroupPrefixProbe{};
                if (tail_tokens > 0) {
                    host_tail.group_keys[i] = store_.TailKeysForGroup(i, published_hash, tail_tokens);
                }
            }
            // With the tail on the Device the split is the page: a sliding
            // group needs no Host block either.
            auto probe = probeGroupBoundary<CacheTier::kHost>(
                i, prompt.prefix_hashes, split_tokens, page_tokens, kv_tail_on_device ? 0 : tail_tokens,
                host_tail.group_keys[i], /*include_page_hits=*/true,
                GroupBoundaryHits{.tail = std::move(host_tail.per_group[i]), .pages = std::move(host_hits[i])},
                extra_hits);
            host_tail.per_group[i] = std::move(probe.hits.tail);
            host_hits[i] = std::move(probe.hits.pages);
            if (!probe.accepts) {
                kv_accepts = false;
                break;
            }
            const std::int32_t block_granularity = store_.Geometry()[i].BlockGranularity();
            if (store_.Groups()[i].Matcher().IsPrefixClosed()) {
                device_hits[i].hits.assign(static_cast<std::size_t>(split_tokens / block_granularity), 1);
                host_hits[i].hits.assign(static_cast<std::size_t>((page_tokens - split_tokens) / block_granularity), 1);
                continue;
            }
            // The sink and window blocks below the split, all on the Device by the split.
            const CacheStore::BlockRange window = store_.SlidingWindowBlocks(static_cast<std::int32_t>(i), boundary);
            device_hits[i].hits.assign(static_cast<std::size_t>(split_tokens / block_granularity), 0);
            for (const CacheStore::BlockRange& needed :
                 {store_.SinkBlocksBelow(static_cast<std::int32_t>(i), window), window}) {
                for (std::int32_t block = needed.first;
                     block < std::min(needed.past_end, split_tokens / block_granularity); ++block) {
                    device_hits[i].hits[static_cast<std::size_t>(block)] = 1;
                }
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
            if (!store_.GroupIsState(i)) {
                continue;
            }
            const std::vector<CacheKey> keys =
                tail_tokens > 0 ? store_.TailKeysForGroup(i, published_hash, tail_tokens) : std::vector<CacheKey>{};
            auto probe = probeGroupBoundary<CacheTier::kDevice>(
                i, prompt.prefix_hashes, split_tokens, page_tokens, tail_tokens, keys, /*include_page_hits=*/true,
                GroupBoundaryHits{.tail = std::move(device_tail.per_group[i]), .pages = std::move(device_hits[i])},
                /*extra_hits=*/nullptr);
            device_tail.per_group[i] = std::move(probe.hits.tail);
            device_hits[i] = std::move(probe.hits.pages);
            if (probe.accepts) {
                device_tail.group_keys[i] = keys;
                continue;
            }
            device_tail.per_group[i] = GroupPrefixProbe{};
            probe = probeGroupBoundary<CacheTier::kHost>(
                i, prompt.prefix_hashes, split_tokens, page_tokens, tail_tokens, keys, /*include_page_hits=*/true,
                GroupBoundaryHits{.tail = std::move(host_tail.per_group[i]), .pages = std::move(host_hits[i])},
                extra_hits);
            host_tail.per_group[i] = std::move(probe.hits.tail);
            host_hits[i] = std::move(probe.hits.pages);
            if (!probe.accepts) return true;
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
    const std::int32_t from_tokens =
        closed_order.empty()
            ? prompt.max_tokens
            : static_cast<std::int32_t>(std::min<std::int64_t>(
                  std::int64_t{host_kv_end_tokens} + store_.PrefixGranularity() - 1, prompt.max_tokens));
    ForEachCandidateBoundary(prompt, above_tokens, from_tokens, visit);
    return HostSuffixProbe{.device = std::move(device), .host = std::move(host)};
}

PrefixProbe PrefixProber::probePrefixWithOrder(const PromptPrefix& prompt, std::span<const std::size_t> match_order,
                                               bool claim_partial) const {
    _assert(prompt.prefix_hashes.size() <=
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / store_.PrefixGranularity()),
            "prefix length exceeds int32 token range");
    _assert(prompt.tokens.empty() || static_cast<std::int64_t>(prompt.tokens.size()) >= prompt.max_tokens,
            "the probe bound lies inside the supplied tokens");
    _assert(!prompt.tail || (!prompt.tokens.empty() && prompt.tail->tail_tokens > 0 &&
                             prompt.tail->tail_tokens == prompt.max_tokens % store_.PrefixGranularity()),
            "an explicit tail describes the probe bound inside its last page");
    PrefixProbe out;
    out.group_keys = ResolvePreparedPrefixKeys(prompt);
    out.device = ProbeTier(prompt, *out.group_keys, match_order, claim_partial);
    if (store_.HostPool() == nullptr) {
        return out;
    }
    const std::int32_t device_tokens = out.device.num_common_tokens;
    out.device_only_hit_tokens = device_tokens;
    HostSuffixProbe suffix = ProbeHostSuffix(prompt, *out.group_keys, match_order, device_tokens,
                                             /*extra_hits=*/nullptr, std::move(out.device));
    out.device = std::move(suffix.device);
    out.host = std::move(suffix.host);
    if (out.host.num_common_tokens <= device_tokens) {
        // The Device alone resumes furthest. The Host tier keeps the Host's
        // KV reach: the junction still counts it, so a chunk cut there
        // publishes the checkpoint that KV lacks.
        const std::int32_t host_kv_tokens = out.host.kv_tokens;
        out.host = PrefixProbe::Tier{};
        out.host.per_group.resize(store_.Groups().size());
        out.host.num_common_tokens = out.device.num_common_tokens;
        out.host.full_page_tokens = out.device.full_page_tokens;
        out.host.kv_tokens = host_kv_tokens;
    }
    return out;
}

}  // namespace tokenspeed
