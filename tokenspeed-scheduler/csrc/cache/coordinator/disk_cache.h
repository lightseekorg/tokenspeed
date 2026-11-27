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

// DiskCache: the local SSD tier below Host (docs/rfcs/0006-disk-tier.md). Its
// blocks are acquired here -- one per (group, bucket), free capacity first,
// then the Disk's least recently used entries -- by a cleaning write that
// copies Host entries to it; Disk entries are published, found, counted and
// dropped here; and the keys of a prompt's prefix the Disk holds are listed
// for a prefetch to read into Host. Disk feeds only Host: no probe an
// admission takes reads it. It works on the store, evicts through the
// evictor and dates by the admission history.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "cache/coordinator/admission_history.h"
#include "cache/coordinator/admission_types.h"
#include "cache/coordinator/cache_evictor.h"
#include "cache/coordinator/cache_store.h"
#include "cache/coordinator/prefix_prober.h"
#include "cache/core/cache_block_ref.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/prefix_index.h"

namespace tokenspeed {

class DiskCache {
public:
    DiskCache(CacheStore& store, CacheEvictor& evictor, AdmissionHistory& history, const PrefixProber& prober)
        : store_{store}, evictor_{evictor}, history_{history}, prober_{prober} {}
    DiskCache(const DiskCache&) = delete;
    DiskCache& operator=(const DiskCache&) = delete;
    DiskCache(DiskCache&&) = delete;
    DiskCache& operator=(DiskCache&&) = delete;

    // Whether there is a Disk tier at all.
    bool HasPool() const { return store_.DiskPool() != nullptr; }

    // One Disk block per (group, bucket), each row's in its bucket: free
    // capacity first, in row order, then the Disk's eviction order funds the
    // remaining rows (TierEvictionPlanner); a row that cannot fit holds an
    // empty reference and evicts nothing it alone needed.
    struct DiskAllocationBatch {
        std::vector<CacheBlockRef> blocks;
        std::size_t allocated{0};
    };
    DiskAllocationBatch AcquireDiskBlocks(std::span<const std::uint32_t> group_ids,
                                          std::span<const std::int32_t> buckets);
    // Publishes a cleaning write's Disk block under key, with the metadata of
    // the Host entry it copied (access epoch, logical block index, boundary
    // kind, session). A key with a Disk entry already re-points block_ref to
    // it.
    void CacheDiskBlock(CacheBlockRef& block_ref, const CacheKey& key,
                        const PrefixCacheIndex::CachedBlockMetadata& metadata);
    bool ContainsDiskCachedBlock(const CacheKey& key) const;
    // The Disk entry under key, held; empty without one.
    CacheBlockRef AcquireDiskCachedBlock(const CacheKey& key) const;
    // Drops key's Disk entry (and, for a State checkpoint, its State siblings)
    // after a read of it failed. False when it is held or gone.
    bool DropUnreadableEntry(const CacheKey& key);
    bool DropUnreadableEntry(const CacheKey& key, const CacheBlockRef& source);
    std::int32_t NumDiskCachedBlocks() const;
    std::int32_t NumPinnedDiskCachedBlocks() const;
    std::int32_t NumFreeLcmBlocks() const {
        return store_.DiskPool() == nullptr ? 0 : store_.DiskPool()->NumEmptyLcmBlocks();
    }
    // The keys of the prompt's prefix up to its max_tokens the Disk holds:
    // every group's page keys and the keys of each published tail the
    // prompt's own tokens reach. A prefetch's Host-suffix probe takes them as
    // Host hits (StorageQueries::ReserveStoragePrefix).
    std::vector<CacheKey> PrefixHits(const PromptPrefix& prompt) const;

private:
    CacheStore& store_;
    CacheEvictor& evictor_;
    AdmissionHistory& history_;
    const PrefixProber& prober_;
};

}  // namespace tokenspeed
