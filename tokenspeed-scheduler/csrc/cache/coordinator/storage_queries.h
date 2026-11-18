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

// StorageQueries: the L3 storage side of prefix matching -- which keys a
// prompt's prefix asks storage for (MissingStorageKeys, SharedPrefixKeys),
// and the Host blocks a storage read lands in (ReserveStoragePrefix). It
// probes with the PrefixProber, taking keys storage holds as Host hits.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "cache/coordinator/admission_types.h"
#include "cache/coordinator/cache_store.h"
#include "cache/coordinator/host_cache.h"
#include "cache/coordinator/prefix_prober.h"
#include "cache/core/cache_block_ref.h"
#include "cache/core/cache_types.h"

namespace tokenspeed {

struct StorageReservation {
    std::vector<CacheKey> keys;
    std::vector<CacheBlockRef> blocks;
    ContentHash tail_prior;
    std::optional<TailPublication> tail;
};
struct StorageQuery {
    std::vector<CacheKey> keys;
    // Tokens each key covers, parallel to `keys`; the scheduler counts a
    // found key's tokens as an L3 hit.
    std::vector<std::int32_t> key_tokens;
    std::int32_t max_tokens{0};
};

class StorageQueries {
public:
    // `enabled` says whether the engine runs an L3 storage tier (it needs a
    // Host pool).
    StorageQueries(const CacheStore& store, const PrefixProber& prober, HostCache& host, bool enabled)
        : store_{store}, prober_{prober}, host_{host}, enabled_{enabled} {}
    StorageQueries(const StorageQueries&) = delete;
    StorageQueries& operator=(const StorageQueries&) = delete;

    bool Enabled() const { return enabled_; }

    // The keys to ask storage for: for each published boundary above the
    // local (Device and Host) hit, longest first, the keys its Host suffix
    // needs that the Host lacks, while they fit in max_keys. max_tokens is the
    // longest boundary the returned keys complete (the local hit when none).
    StorageQuery MissingStorageKeys(const PromptPrefix& prompt, std::size_t max_keys) const;
    // All immutable objects needed at exactly this boundary, including objects
    // already in Device/Host. max_tokens is 0 for an illegal boundary and -1
    // when the bounded key budget cannot describe the complete dependency.
    StorageQuery SharedPrefixKeys(const PromptPrefix& prompt, std::size_t max_keys) const;
    // Host destinations for a storage read: the longest boundary that the
    // storage `hits` (with the Host) extend the Device-only hit to, under the
    // Host-suffix rule, whose missing keys get Host blocks within max_blocks
    // (warm Host hits counted). Empty when no boundary fits.
    StorageReservation ReserveStoragePrefix(const PromptPrefix& prompt, std::span<const CacheKey> hits,
                                            std::size_t max_blocks);

private:
    // The complete dependency set selected by a matcher, with each key's
    // absolute logical block index (including holes before an SWA window);
    // floor_tokens is the tier's floor, where a token-spanned group's hits
    // start.
    template <typename Visit>
    void forEachStorageDependency(const PrefixProbe::Tier& probe, std::span<const std::vector<CacheKey>> group_keys,
                                  std::int32_t floor_tokens, const Visit& visit) const;

    const CacheStore& store_;
    const PrefixProber& prober_;
    HostCache& host_;
    const bool enabled_;
};

}  // namespace tokenspeed
