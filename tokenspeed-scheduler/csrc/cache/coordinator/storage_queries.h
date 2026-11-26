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
    // The boundary the reservation resumes the prompt at; 0 when empty.
    std::int32_t num_tokens{0};
    std::vector<CacheKey> keys;
    // Each key's block index in its group's request table, parallel to
    // `keys` (holes before a sliding window included).
    std::vector<std::int32_t> logical_blocks;
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
    // When `source_pool` is non-null (local Disk tier), each missing key's Host
    // destination is allocated in the same sharded owner bucket as its source
    // block in `*source_pool` so every rank reads only its own local Disk file.
    StorageReservation ReserveStoragePrefix(const PromptPrefix& prompt, std::span<const CacheKey> hits,
                                            std::size_t max_blocks, const BlockPool* source_pool = nullptr);
    // Longest boundary that `hits` (with Device and Host) can resume under the
    // Host-suffix rule, without allocating blocks or spending Host credits.
    std::int32_t ResumableStorageTokens(const PromptPrefix& prompt, std::span<const CacheKey> hits) const;
    // Reserve only a complete peer landing. A transient shortage sets retry;
    // an illegal boundary or a suffix exceeding the absolute limit does not.
    StorageReservation ReserveExactStoragePrefix(const PromptPrefix& prompt, std::span<const CacheKey> hits,
                                                 std::size_t available, std::size_t limit, bool& retry);
    // Peer sharing with a Disk tier (docs/rfcs/0006-disk-tier.md): what a
    // holder lists when `disk_hits`, keys its Disk holds, extend its local
    // hit -- the longest boundary the Device, the Host and the Disk resume
    // together under the Host-suffix rule, each page from the fastest tier
    // holding it, at the slots PrefixProber::ListPrefix uses. Reads only;
    // num_tokens is 0 when the Disk adds nothing to the Device's hit.
    PrefixProber::PrefixListing ListWithDisk(const PromptPrefix& prompt, std::span<const CacheKey> disk_hits) const;

private:
    StorageReservation reserveStoragePrefix(const PromptPrefix& prompt, std::span<const CacheKey> hits,
                                            std::size_t available, std::size_t limit, bool* exact_retry,
                                            const BlockPool* source_pool = nullptr);
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
