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

// PrefixProber: read-only prefix matching over both tiers -- what a prompt's
// prefix hits on the Device and in Host L2, converged across the groups to
// one resumable boundary (cache-concepts.md, ProbePromptPrefix), what each
// boundary has physically, and the probing steps the L3 storage queries
// build on. It reads the store, and takes the keys an eviction write-back is
// bringing to the Host as held there (EvictionStoresExtendPrefix).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "cache/coordinator/admission_types.h"
#include "cache/coordinator/cache_evictor.h"
#include "cache/coordinator/cache_store.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache/prefix/tail_table.h"

namespace tokenspeed {

class PrefixProber {
public:
    PrefixProber(const CacheStore& store, const CacheEvictor& evictor) : store_{store}, evictor_{evictor} {}
    PrefixProber(const PrefixProber&) = delete;
    PrefixProber& operator=(const PrefixProber&) = delete;

    // A probe is read-only. Cache state must not change before its result
    // is passed to an admission (CacheCoordinator::Admit), except by a
    // lone-request compaction (LoneRequestAdmission::Compact), whose caller
    // retries with its copy of the probe taken before it: every hit stays
    // under its key, held and relocated with its entry, but the coverage the
    // probe counts past its hit (kv_tokens, closed_kv_tokens, so the
    // junction) may then be gone from the Device.
    PreparedPrefixKeys PreparePrefixKeys(std::span<const std::string> content_hashes) const;
    PrefixProbe ProbePromptPrefix(const PromptPrefix& prompt) const;
    // Whether recorded or in-flight eviction stores would extend the visible
    // hit using ordinary matching rules. Never acquires unpublished blocks.
    bool EvictionStoresExtendPrefix(const PromptPrefix& prompt, const PrefixProbe& visible) const;

    // Read-only physical availability, before admission can evict anything.
    // A bit names a group missing blocks at this boundary. The union permits
    // arbitrary tier mixing; it is an upper bound, not a resumable prefix.
    struct PrefixBoundaryAvailability {
        std::int32_t tokens{0};
        std::uint64_t device_missing{0};
        std::uint64_t host_missing{0};
        std::uint64_t union_missing{0};
    };
    std::vector<PrefixBoundaryAvailability> InspectPrefixAvailability(const PromptPrefix& prompt) const;
    // Full pages only: a probe that enumerates no tails.
    PrefixProbe ProbePrefix(std::span<const std::string> content_hashes) const;

    // Decode-side PD reuses local history pages; the KV boundary trims the
    // peer's transfer and the partial block of a tail hit is left a hole for
    // the transfer to fill whole. A State group's checkpoint counts only at
    // exactly prompt_end_tokens (the peer's final state supersedes any
    // shorter one): then state_at_prompt_end claims it as slot 0 and the
    // transfer omits the state groups. prompt.prefix_hashes must cover the
    // prompt's last full page and prompt.tokens the whole prompt.
    PrefixProbe ProbeDecodeDevicePrefix(const PromptPrefix& prompt, std::int32_t prompt_end_tokens) const;

    // The pages of a probe's hit, as CacheCoordinator::ExportPrefix would
    // export them, without acquiring or refreshing anything: what a peer is
    // told it may read (docs/rfcs/0004-peer-prefix-sharing.md).
    struct ListedPage {
        CacheKey key;
        std::int32_t slot{0};
        CacheTier tier{CacheTier::kDevice};
        std::int32_t block_id{0};

        bool operator==(const ListedPage&) const = default;
    };
    struct PrefixListing {
        std::int32_t num_tokens{0};
        std::vector<ListedPage> pages;
    };
    PrefixListing ListPrefix(const PrefixProbe& probe) const;

    // The probing steps the L3 storage queries build on (StorageQueries).
    PreparedPrefixKeys ResolvePreparedPrefixKeys(const PromptPrefix& prompt) const;
    // The Device's probe: the aligned phase, then the candidate phase.
    // `claim_partial` false leaves the partial block of a tail a hole in the
    // hits.
    PrefixProbe::Tier ProbeTier(const PromptPrefix& prompt, std::span<const std::vector<CacheKey>> group_keys,
                                std::span<const std::size_t> match_order, bool claim_partial) const;
    struct HostSuffixProbe {
        PrefixProbe::Tier device;
        PrefixProbe::Tier host;
    };
    // The Host-suffix probe over boundaries beyond above_tokens: the Device's KV
    // prefix (device.kv_tokens, of the Device-only probe, bounds it), then a
    // Host suffix, the State checkpoints from either tier. extra_hits are
    // assumed on the Host tier only. Returns both tiers together. On a hit the Device part
    // is trimmed to its share; otherwise it keeps the supplied Device probe
    // and host.num_common_tokens is 0. Either way host.kv_tokens is the longest
    // boundary whose KV the two tiers hold together when the Host holds part of
    // it -- the KV reach the Host adds past the Device's own -- and 0
    // otherwise.
    HostSuffixProbe ProbeHostSuffix(const PromptPrefix& prompt, std::span<const std::vector<CacheKey>> group_keys,
                                    std::span<const std::size_t> match_order, std::int32_t above_tokens,
                                    const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits,
                                    PrefixProbe::Tier device) const;
    // The boundaries a tier may resume at, longest first: the tails published
    // inside each page under its prior that the prompt's own tokens reach and the
    // page boundaries themselves, from the page containing `from_tokens` down to
    // the one containing the floor. The floor need not be aligned: the Host suffix
    // is probed above the Device-only hit, which may be a tail with longer tails
    // beside it in its page. Visits each as (page_tokens, tail_tokens,
    // published hash); the visitor returns false to stop.
    template <typename Visit>
    void ForEachCandidateBoundary(const PromptPrefix& prompt, std::int32_t floor_tokens, std::int32_t from_tokens,
                                  const Visit& visit) const;

private:
    std::vector<std::vector<CacheKey>> buildGroupKeys(std::span<const std::string> content_hashes) const;
    // Aligned matching over token-spanned groups; no State checkpoint hits.
    // extra_hits counts additional assumed hits beside the tier's entries.
    template <CacheTier Tier>
    PrefixProbe::Tier probeTierWithKeys(std::span<const std::vector<CacheKey>> group_keys,
                                        std::span<const std::size_t> match_order, std::int32_t num_prefix_pages,
                                        std::int32_t floor_tokens,
                                        const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits) const;
    struct GroupBoundaryHits {
        GroupPrefixProbe tail;
        GroupPrefixProbe pages;
    };
    struct GroupBoundaryProbe {
        bool accepts;
        GroupBoundaryHits hits;
    };
    // Return the boundary decision and its hit arrays together. Consuming the
    // previous arrays retains their capacity while successive boundaries are tried.
    template <CacheTier Tier>
    GroupBoundaryProbe probeGroupBoundary(std::size_t i, std::span<const std::string> prefix_hashes,
                                          std::int32_t floor_tokens, std::int32_t full_tokens, std::int32_t tail_tokens,
                                          std::span<const CacheKey> tail_keys, bool include_page_hits,
                                          GroupBoundaryHits hits,
                                          const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits) const;
    bool stateGroupsAcceptBoundary(std::span<const std::size_t> match_order, std::span<const std::string> prefix_hashes,
                                   std::int32_t full_tokens, std::int32_t tail_tokens, const ContentHash& tail_hash,
                                   bool allow_host, const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits) const;
    PrefixProbe probePrefixWithOrder(const PromptPrefix& prompt, std::span<const std::size_t> match_order,
                                     bool claim_partial) const;

    const CacheStore& store_;
    const CacheEvictor& evictor_;
};

template <typename Visit>
void PrefixProber::ForEachCandidateBoundary(const PromptPrefix& prompt, std::int32_t floor_tokens,
                                            std::int32_t from_tokens, const Visit& visit) const {
    // The end of the page containing from_tokens can pass INT32_MAX.
    for (std::int32_t page = from_tokens / store_.PrefixGranularity();
         (std::int64_t{page} + 1) * store_.PrefixGranularity() > floor_tokens; --page) {
        const std::int32_t page_tokens = page * store_.PrefixGranularity();
        if (static_cast<std::size_t>(page) <= prompt.prefix_hashes.size() && !prompt.tokens.empty()) {
            const ContentHash prior =
                page == 0 ? ContentHash{} : ContentHash{prompt.prefix_hashes[static_cast<std::size_t>(page) - 1]};
            const std::span<const TailTable::Tail> published = store_.Boundaries().Tails().Tails(prior);
            std::vector<TailTable::Tail> candidates(published.begin(), published.end());
            if (prompt.tail && page == prompt.max_tokens / store_.PrefixGranularity()) {
                const auto& tail = *prompt.tail;
                if (std::ranges::none_of(candidates, [&](const auto& candidate) {
                        return candidate.tail_tokens == tail.tail_tokens && candidate.hash == tail.hash;
                    })) {
                    candidates.push_back({.tail_tokens = tail.tail_tokens, .hash = tail.hash});
                }
            }
            for (const auto boundary : prompt.shared_prefix_tokens) {
                const auto count = boundary - page_tokens;
                if (count <= 0 || count >= store_.PrefixGranularity() || boundary > from_tokens ||
                    boundary > prompt.max_tokens || boundary <= floor_tokens ||
                    static_cast<std::size_t>(boundary) > prompt.tokens.size())
                    continue;
                const auto hash = HashPrefixPage(prompt.tokens.subspan(page_tokens, count), prior);
                if (std::ranges::none_of(candidates, [&](const auto& candidate) {
                        return candidate.tail_tokens == count && candidate.hash == hash;
                    }))
                    candidates.push_back({.tail_tokens = count, .hash = hash});
            }
            std::ranges::stable_sort(candidates, std::greater<>{}, &TailTable::Tail::tail_tokens);
            // The prompt's own hash at a tail length, reused while the
            // candidates (sorted by length) keep that length.
            ContentHash hash;
            std::int32_t hashed_tail_tokens = 0;
            for (const TailTable::Tail& tail : candidates) {
                const std::int32_t boundary = page_tokens + tail.tail_tokens;
                if (boundary % store_.CheckpointGranularity() != 0 || boundary > from_tokens ||
                    boundary > prompt.max_tokens || boundary <= floor_tokens ||
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
        if (page_tokens > floor_tokens && page_tokens <= from_tokens &&
            page_tokens % store_.CheckpointGranularity() == 0) {
            if (!visit(page_tokens, 0, ContentHash{})) {
                return;
            }
        }
        if (page == 0) {
            break;
        }
    }
}

}  // namespace tokenspeed
