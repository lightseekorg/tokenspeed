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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <list>
#include <map>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_block_ref.h"
#include "cache/core/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// One cache group's prefix-reuse index: CacheKey -> canonical CacheBlock.
// It decides WHAT is reusable; placement and allocation stay in BlockPool and
// GroupAllocator. Indices are pool-scoped because the same group serves the
// Device and Host tiers; every referenced BlockPool must outlive this index.
class PrefixCacheIndex {
private:
    struct CacheEntry;
    struct CacheEntries;
    using CacheEntryList = std::list<CacheEntry>;
    using CacheEntryIterator = CacheEntryList::iterator;
    using ConstCacheEntryIterator = CacheEntryList::const_iterator;

    // Index key: oldest access epoch first, with location as a stable tie-break.
    struct EvictionOrder {
        std::uint64_t last_access_epoch{0};
        std::int32_t lcm_block_id{0};
        std::int32_t slot_index{0};

        auto operator<=>(const EvictionOrder&) const noexcept = default;
    };

    using EvictionOrderMap = std::map<EvictionOrder, CacheEntryIterator>;

public:
    // Read-only admission snapshot from one index lookup; owns no block.
    struct CachedBlockMetadata {
        std::uint64_t last_access_epoch{0};
        std::int32_t logical_block_index{-1};
        CacheBoundaryKind boundary_kind{CacheBoundaryKind::kChunk};
        bool was_acquired{false};
        // The largest epoch any acquisition or publication touched the entry
        // with. last_access_epoch can move back below it (a publication under
        // an older epoch, CacheCoordinator::MarkSupersededStateInput); this
        // one never does.
        std::uint64_t max_access_epoch{0};
    };

    // Non-owning traversal state, bound on first use to one index/pool. The
    // index and pool must outlive it; do not insert, erase, or re-key entries
    // in that tier until traversal is finished. Use a fresh cursor per plan.
    class EvictionCursor {
        friend class PrefixCacheIndex;
        const CacheEntries* cache_index_{nullptr};
        EvictionOrderMap::const_iterator next_{};
    };

    // Traverses only keys present in both configured pools, in the source
    // pool's newest-first order. Like EvictionCursor, it borrows the index
    // and pools for one read-only planning pass.
    class CommonCursor {
        friend class PrefixCacheIndex;
        const CacheEntries* cache_index_{nullptr};
        EvictionOrderMap::const_reverse_iterator next_{};
    };

    // One unpinned entry, paired with the metadata an eviction policy reads.
    struct EvictionCandidate {
        CacheBlockLocation location;
        CachedBlockMetadata metadata;
    };

    // A read-only policy snapshot; inspecting it neither pins nor touches the
    // entry. Take a fresh snapshot after any ownership or index mutation.
    struct CachedBlockSnapshot {
        CacheBlockLocation location;
        CachedBlockMetadata metadata;
        bool pinned;
    };

    struct CommonEntry {
        const CacheKey* key;
        CachedBlockSnapshot source;
        CachedBlockSnapshot peer;
    };

    explicit PrefixCacheIndex(std::uint32_t group_id) : group_id_{group_id} {}

    PrefixCacheIndex(const PrefixCacheIndex&) = delete;
    PrefixCacheIndex& operator=(const PrefixCacheIndex&) = delete;
    PrefixCacheIndex(PrefixCacheIndex&&) = default;
    PrefixCacheIndex& operator=(PrefixCacheIndex&&) = default;

    // Maintain the intersection of one source/peer pair. The secondary index
    // owns no blocks and follows source recency; ownership is read by cursors.
    void TrackCommonEntries(const BlockPool& source_pool, const BlockPool& peer_pool) {
        _assert(&source_pool != &peer_pool, "common entries need distinct pools");
        CacheEntries& source = cacheEntries(source_pool);
        CacheEntries& peer = cacheEntries(peer_pool);
        _assert(source.common_peer == nullptr && peer.common_peer == nullptr, "common pools are configured once");
        source.common_peer = &peer;
        source.common_source = true;
        peer.common_peer = &source;
        for (auto it = source.entries.begin(); it != source.entries.end(); ++it) {
            linkCommonEntry(source, it);
        }
    }

    // Registers block_ref under key. If key already has a canonical block,
    // block_ref is replaced with a reference to that block.
    void Register(const BlockPool& pool, CacheBlockRef& block_ref, const CacheKey& key, std::uint64_t access_epoch,
                  std::int32_t logical_block_index, CacheBoundaryKind boundary_kind,
                  std::vector<std::pair<CacheKey, CacheBlockRef>>* newly_cached) {
        _assert(block_ref && block_ref.IsOwnedBy(pool), "cache block must belong to the target pool");
        _assert(pool.BoundGroup(block_ref->Location().lcm_block_id) == group_id_,
                "cache block must belong to the prefix index group");
        validateKey(key);
        CacheEntries& cache_index = cacheEntries(pool);
        CacheEntryIterator existing_it = findEntry(cache_index, block_ref->Location());
        if (existing_it != cache_index.entries.end()) {
            _assert(existing_it->key == key, "one cache block location cannot change cache key");
            if (existing_it->boundary_kind < boundary_kind) {
                existing_it->boundary_kind = boundary_kind;
            }
            touchEntry(cache_index, existing_it, access_epoch);
            return;
        }
        CacheEntryIterator canonical_it = findEntry(cache_index, key);
        if (canonical_it != cache_index.entries.end()) {
            if (canonical_it->boundary_kind < boundary_kind) {
                canonical_it->boundary_kind = boundary_kind;
            }
            touchEntry(cache_index, canonical_it, access_epoch);
            block_ref = canonical_it->block_ref;
            return;
        }

        CacheEntry& entry = cache_index.entries.emplace_back();
        entry.key = key;
        entry.block_ref = block_ref;
        entry.last_access_epoch = access_epoch;
        entry.logical_block_index = logical_block_index;
        entry.boundary_kind = boundary_kind;
        entry.max_access_epoch = access_epoch;
        entry.available_order = &cache_index.available_order;
        CacheEntryIterator entry_it = std::prev(cache_index.entries.end());
        cache_index.by_key.emplace(entry_it->key, entry_it);
        cache_index.by_location.emplace(entry_it->block_ref->Location(), entry_it);
        entry_it->available_it = cache_index.available_order.emplace(evictionOrder(*entry_it), entry_it).first;
        linkCommonEntry(cache_index, entry_it);
        if (newly_cached != nullptr) {
            newly_cached->emplace_back(key, block_ref);
        }
    }

    // Registers blocks[j] under keys[j]. blocks is the caller's own storage
    // (GroupAllocator::BlocksToPublish for a request table): a block that
    // dedupes against an existing canonical entry is replaced in place.
    // first_logical_block is the logical prefix position of blocks[0].
    void RegisterFullBlocks(const BlockPool& pool, std::span<CacheBlockRef> blocks, std::span<const CacheKey> keys,
                            std::uint64_t access_epoch, std::int32_t first_logical_block,
                            CacheBoundaryKind boundary_kind,
                            std::vector<std::pair<CacheKey, CacheBlockRef>>* newly_cached) {
        _assert(first_logical_block >= 0, "first_logical_block must be >= 0");
        _assert(blocks.size() == keys.size(), "one key per published block");
        for (std::size_t j = 0; j < keys.size(); ++j) {
            CacheBlockRef& block_ref = blocks[j];
            if (!block_ref) {
                continue;
            }
            Register(pool, block_ref, keys[j], access_epoch, first_logical_block + static_cast<std::int32_t>(j),
                     boundary_kind, newly_cached);
        }
    }

    bool Contains(const BlockPool& pool, const CacheKey& key) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        return cache_index != nullptr && findEntry(*cache_index, key) != cache_index->entries.end();
    }
    bool Contains(const BlockPool& pool, CacheBlockLocation location) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        return cache_index != nullptr && findEntry(*cache_index, location) != cache_index->entries.end();
    }
    // Any-tier lookup that also checks identity, not just location.
    bool Contains(const CacheBlockRef& block_ref) const {
        if (!block_ref) {
            return false;
        }
        return std::ranges::any_of(cache_entries_by_pool_, [&](const auto& item) {
            auto index_it = item.second.by_location.find(block_ref->Location());
            return index_it != item.second.by_location.end() && index_it->second->block_ref == block_ref;
        });
    }

    CacheBlockRef Find(const BlockPool& pool, const CacheKey& key) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return {};
        }
        ConstCacheEntryIterator entry_it = findEntry(*cache_index, key);
        return entry_it == cache_index->entries.end() ? CacheBlockRef{} : entry_it->block_ref;
    }

    std::optional<CachedBlockSnapshot> Inspect(const BlockPool& pool, const CacheKey& key) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return std::nullopt;
        }
        const ConstCacheEntryIterator entry_it = findEntry(*cache_index, key);
        if (entry_it == cache_index->entries.end()) {
            return std::nullopt;
        }
        return CachedBlockSnapshot{
            .location = entry_it->block_ref->Location(),
            .metadata = metadataOf(*entry_it),
            .pinned = !entry_it->block_ref.unique(),
        };
    }

    // The key this tier publishes `block_ref` under, when the entry's block is
    // this very block (identity, not just location); valid until the entry is
    // evicted. A retraction asks this of each data slot to decide which tier
    // images it.
    const CacheKey* KeyOf(const BlockPool& pool, const CacheBlockRef& block_ref) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr || !block_ref) {
            return nullptr;
        }
        ConstCacheEntryIterator entry_it = findEntry(*cache_index, block_ref->Location());
        if (entry_it == cache_index->entries.end() || entry_it->block_ref != block_ref) {
            return nullptr;
        }
        return &entry_it->key;
    }
    // The key of the entry at `location` on this tier, if one is there.
    std::optional<CacheKey> KeyOf(const BlockPool& pool, CacheBlockLocation location) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return std::nullopt;
        }
        ConstCacheEntryIterator entry_it = findEntry(*cache_index, location);
        if (entry_it == cache_index->entries.end()) {
            return std::nullopt;
        }
        return entry_it->key;
    }

    std::optional<CachedBlockMetadata> MetadataFor(const BlockPool& pool, CacheBlockLocation location) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return std::nullopt;
        }
        ConstCacheEntryIterator entry_it = findEntry(*cache_index, location);
        if (entry_it == cache_index->entries.end()) {
            return std::nullopt;
        }
        return metadataOf(*entry_it);
    }

    // The next potentially evictable epoch. Entries known to be pinned have
    // already left available_order and rejoin it when their last owner leaves.
    // Reading the order key does not inspect owners or advance the cursor.
    std::optional<std::uint64_t> PeekEvictionEpoch(const BlockPool& pool, EvictionCursor& cursor) const {
        if (cursor.cache_index_ == nullptr) {
            cursor.cache_index_ = findCacheEntries(pool);
            if (cursor.cache_index_ == nullptr) {
                return std::nullopt;
            }
            cursor.next_ = cursor.cache_index_->available_order.begin();
        }
        if (cursor.next_ == cursor.cache_index_->available_order.end()) {
            return std::nullopt;
        }
        return cursor.next_->first.last_access_epoch;
    }

    // Appends a complete unpinned epoch, skipping fully pinned epochs up to
    // max_access_epoch. False means either exhaustion or a later epoch; Peek
    // distinguishes them. A later epoch stays unread for the next call.
    bool NextEvictionEpoch(const BlockPool& pool, EvictionCursor& cursor, std::uint64_t max_access_epoch,
                           std::vector<EvictionCandidate>& out) const {
        const auto next_epoch = PeekEvictionEpoch(pool, cursor);
        if (!next_epoch || *next_epoch > max_access_epoch) {
            return false;
        }
        auto& available = cursor.cache_index_->available_order;
        while (cursor.next_ != available.end() && cursor.next_->first.last_access_epoch <= max_access_epoch) {
            const std::uint64_t epoch = cursor.next_->first.last_access_epoch;
            const std::size_t initial_size = out.size();
            while (cursor.next_ != available.end() && cursor.next_->first.last_access_epoch == epoch) {
                const auto current = cursor.next_++;
                CacheEntry& cache_entry = *current->second;
                if (cache_entry.block_ref.unique()) {
                    out.push_back(EvictionCandidate{
                        .location = cache_entry.block_ref->Location(),
                        .metadata = metadataOf(cache_entry),
                    });
                } else {
                    // Keep the node for allocation-free restoration when the
                    // last request/transfer owner releases this block. Ordinary
                    // temporary Find() references need no index maintenance.
                    cache_entry.pinned_order = available.extract(current);
                    cache_entry.unique_observer.Observe(cache_entry.block_ref, restoreAvailable, &cache_entry);
                }
            }
            if (out.size() != initial_size) {
                return true;
            }
        }
        return false;
    }

    // Borrowed keys and snapshots neither pin blocks nor update recency.
    // Start a fresh cursor after any index or ownership change.
    std::optional<CommonEntry> NextCommonRecentEntry(const BlockPool& source_pool, CommonCursor& cursor) const {
        if (cursor.cache_index_ == nullptr) {
            cursor.cache_index_ = findCacheEntries(source_pool);
            if (cursor.cache_index_ == nullptr) {
                return std::nullopt;
            }
            cursor.next_ = cursor.cache_index_->common_order.rbegin();
        }
        if (cursor.next_ == cursor.cache_index_->common_order.rend()) {
            return std::nullopt;
        }
        const CacheEntry& source = *(cursor.next_++)->second;
        const CacheEntry& peer = *source.common_peer;
        return CommonEntry{
            .key = &source.key,
            .source = CachedBlockSnapshot{.location = source.block_ref->Location(),
                                          .metadata = metadataOf(source),
                                          .pinned = !source.block_ref.unique()},
            .peer = CachedBlockSnapshot{.location = peer.block_ref->Location(),
                                        .metadata = metadataOf(peer),
                                        .pinned = !peer.block_ref.unique()},
        };
    }

    // Visit all entries without ordering or filtering ownership. Borrowed
    // keys and snapshots neither pin blocks nor touch recency. The visitor
    // must not mutate this index or block ownership during traversal.
    template <typename Visit>
    void ForEachEntry(const BlockPool& pool, const Visit& visit) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return;
        }
        for (const CacheEntry& entry : cache_index->entries) {
            visit(entry.key, CachedBlockSnapshot{.location = entry.block_ref->Location(),
                                                 .metadata = metadataOf(entry),
                                                 .pinned = !entry.block_ref.unique()});
        }
    }

    std::int32_t NumEntries(const BlockPool& pool) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        return cache_index == nullptr ? 0 : static_cast<std::int32_t>(cache_index->entries.size());
    }
    std::int32_t NumPinnedEntries(const BlockPool& pool) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return 0;
        }
        return static_cast<std::int32_t>(std::ranges::count_if(
            cache_index->entries, [](const CacheEntry& cache_entry) { return cache_entry.block_ref.use_count() > 1; }));
    }

    // An upper bound: traversal removes pinned entries lazily, and ownership
    // notifications restore an entry as soon as it becomes unpinned again.
    // This can overestimate eviction capacity, never underestimate it.
    std::size_t PotentiallyEvictableEntries(const BlockPool& pool) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        return cache_index == nullptr ? 0 : cache_index->available_order.size();
    }

    // Every unpinned entry of one tier, in no particular order. Callers that
    // only want the oldest few should use NextEvictionEpoch() instead; this is
    // for the ones that genuinely need the whole set. The metadata rides along
    // because recovering it afterwards would cost a lookup per entry.
    std::vector<EvictionCandidate> EvictableCandidates(const BlockPool& pool) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return {};
        }
        std::vector<EvictionCandidate> candidates;
        candidates.reserve(cache_index->entries.size());
        for (const CacheEntry& cache_entry : cache_index->entries) {
            if (cache_entry.block_ref.unique()) {
                candidates.push_back(EvictionCandidate{
                    .location = cache_entry.block_ref->Location(),
                    .metadata = metadataOf(cache_entry),
                });
            }
        }
        return candidates;
    }

    std::optional<CacheKey> Evict(const BlockPool& pool, CacheBlockLocation location) {
        CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return std::nullopt;
        }
        CacheEntryIterator entry_it = findEntry(*cache_index, location);
        if (entry_it == cache_index->entries.end() || !entry_it->block_ref.unique()) {
            return std::nullopt;
        }
        CacheKey key = entry_it->key;
        eraseEntry(*cache_index, entry_it);
        return key;
    }

    // True when every occupied child of the LCM parent is an unpinned entry of
    // this index, i.e. evicting the parent loses only reusable cache.
    bool ParentIsFullyEvictable(const BlockPool& pool, std::int32_t lcm_block_id,
                                std::int32_t cache_blocks_per_lcm_block) const {
        if (pool.OccupiedCount(lcm_block_id) == 0) {
            return false;
        }
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return false;
        }
        for (std::int32_t slot = 0; slot < cache_blocks_per_lcm_block; ++slot) {
            const CacheBlockLocation location{.lcm_block_id = lcm_block_id, .slot_index = slot};
            if (!pool.IsOccupied(location)) {
                continue;
            }
            ConstCacheEntryIterator entry_it = findEntry(*cache_index, location);
            if (entry_it == cache_index->entries.end() || !entry_it->block_ref.unique()) {
                return false;
            }
        }
        return true;
    }

    // Raises key's entry in pool to access_epoch, never dating it back; false
    // when pool does not cache key.
    bool Touch(const BlockPool& pool, const CacheKey& key, std::uint64_t access_epoch) {
        CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return false;
        }
        CacheEntryIterator entry_it = findEntry(*cache_index, key);
        if (entry_it == cache_index->entries.end()) {
            return false;
        }
        if (entry_it->last_access_epoch < access_epoch) {
            touchEntry(*cache_index, entry_it, access_epoch);
        }
        return true;
    }

    // Pins the probed hits: marks them acquired at access_epoch and returns
    // owning references aligned with probe.hits.
    PrefixMatch AcquireMatched(const BlockPool& pool, std::span<const CacheKey> keys, std::int32_t begin_blocks,
                               const GroupPrefixProbe& probe, std::uint64_t access_epoch) {
        _assert(begin_blocks >= 0 && static_cast<std::size_t>(begin_blocks) + probe.hits.size() <= keys.size(),
                "matched block range is out of bounds");
        PrefixMatch match;
        match.blocks.resize(probe.hits.size());
        CacheEntries* cache_index = findCacheEntries(pool);
        for (std::size_t i = 0; i < probe.hits.size(); ++i) {
            if (probe.hits[i] == 0) {
                continue;
            }
            _assert(cache_index != nullptr, "cached pool disappeared between match probe and acquisition");
            CacheEntryIterator entry_it = findEntry(*cache_index, keys[static_cast<std::size_t>(begin_blocks) + i]);
            _assert(entry_it != cache_index->entries.end(),
                    "cached block disappeared between match probe and acquisition");
            entry_it->was_acquired = true;
            touchEntry(*cache_index, entry_it, access_epoch);
            match.blocks[i] = entry_it->block_ref;
        }
        return match;
    }

    std::vector<CacheBlockLocation> MatchedLocations(const BlockPool& pool, std::span<const CacheKey> keys,
                                                     std::int32_t begin_blocks, const GroupPrefixProbe& probe) const {
        _assert(begin_blocks >= 0 && static_cast<std::size_t>(begin_blocks) + probe.hits.size() <= keys.size(),
                "matched block range is out of bounds");
        std::vector<CacheBlockLocation> locations;
        locations.reserve(static_cast<std::size_t>(std::ranges::count(probe.hits, std::uint8_t{1})));
        const CacheEntries* cache_index = findCacheEntries(pool);
        for (std::size_t i = 0; i < probe.hits.size(); ++i) {
            if (probe.hits[i] == 0) {
                continue;
            }
            _assert(cache_index != nullptr, "cached pool disappeared between match probes");
            ConstCacheEntryIterator entry_it =
                findEntry(*cache_index, keys[static_cast<std::size_t>(begin_blocks) + i]);
            _assert(entry_it != cache_index->entries.end(), "cached block disappeared between match probes");
            locations.push_back(entry_it->block_ref->Location());
        }
        return locations;
    }

private:
    struct CacheEntry {
        CacheKey key;
        CacheBlockRef block_ref;
        std::uint64_t last_access_epoch{0};
        // Position in the request's logical prefix. Host-only entries may not
        // have a device-table position yet.
        std::int32_t logical_block_index{-1};
        CacheBoundaryKind boundary_kind{CacheBoundaryKind::kChunk};
        // Set only after a successful request admission acquires this entry.
        bool was_acquired{false};
        // See CachedBlockMetadata::max_access_epoch.
        std::uint64_t max_access_epoch{0};
        EvictionOrderMap* available_order{nullptr};
        // This entry's node in available_order while it is there (pinned_order
        // empty), so re-keying needs no search.
        EvictionOrderMap::iterator available_it{};
        EvictionOrderMap::node_type pinned_order;
        CacheBlockUniqueObserver unique_observer;
        CacheEntry* common_peer{nullptr};
    };

    struct CacheEntries {
        // Owns each CacheEntry once. The maps are non-owning secondary indices
        // into stable list nodes. available_order keeps eviction candidates
        // sorted by age; common_order keeps shared keys in source age order.
        // AdmissionPlanner owns the policy that ranks one access epoch.
        CacheEntryList entries;
        std::unordered_map<CacheKey, CacheEntryIterator, CacheKeyHash> by_key;
        std::unordered_map<CacheBlockLocation, CacheEntryIterator, CacheBlockLocationHash> by_location;
        // Entries known to be pinned are removed lazily during traversal.
        // Ownership notifications restore them without changing their age.
        mutable EvictionOrderMap available_order;
        // Only the source keeps the order. Entries in both pools point at
        // each other while their keys coexist; erasing either unlinks them.
        CacheEntries* common_peer{nullptr};
        bool common_source{false};
        EvictionOrderMap common_order;
    };

    void linkCommonEntry(CacheEntries& cache_index, CacheEntryIterator entry) {
        if (cache_index.common_peer == nullptr) {
            return;
        }
        CacheEntries& peer = *cache_index.common_peer;
        const auto other = findEntry(peer, entry->key);
        if (other == peer.entries.end()) {
            return;
        }
        entry->common_peer = &*other;
        other->common_peer = &*entry;
        CacheEntries& source = cache_index.common_source ? cache_index : peer;
        const auto source_entry = cache_index.common_source ? entry : other;
        source.common_order.emplace(evictionOrder(*source_entry), source_entry);
    }

    static void restoreAvailable(void* context) noexcept {
        auto& entry = *static_cast<CacheEntry*>(context);
        entry.available_it = entry.available_order->insert(std::move(entry.pinned_order)).position;
    }

    static CachedBlockMetadata metadataOf(const CacheEntry& cache_entry) noexcept {
        return CachedBlockMetadata{
            .last_access_epoch = cache_entry.last_access_epoch,
            .logical_block_index = cache_entry.logical_block_index,
            .boundary_kind = cache_entry.boundary_kind,
            .was_acquired = cache_entry.was_acquired,
            .max_access_epoch = cache_entry.max_access_epoch,
        };
    }
    static EvictionOrder evictionOrder(const CacheEntry& cache_entry) {
        const CacheBlockLocation location = cache_entry.block_ref->Location();
        return EvictionOrder{
            .last_access_epoch = cache_entry.last_access_epoch,
            .lcm_block_id = location.lcm_block_id,
            .slot_index = location.slot_index,
        };
    }
    // The only writer of last_access_epoch: an entry's location never moves, so
    // re-keying moves its available_order node (or re-keys its pinned_order) --
    // amortized O(1) only when the node becomes the map's maximum (a hit or a
    // Host refresh at a new newest epoch); O(log N), like an erase and
    // re-insert, when an entry sharing its epoch orders after it or the epoch
    // is older, and for a common_order node, found by key -- in exchange for
    // admission's ordered traversal and early exit. Continued requests can
    // republish under an older epoch, so moving entries to the back of an LRU
    // list would not preserve this order.
    void touchEntry(CacheEntries& cache_index, CacheEntryIterator entry_it, std::uint64_t access_epoch) {
        entry_it->max_access_epoch = std::max(entry_it->max_access_epoch, access_epoch);
        if (entry_it->last_access_epoch == access_epoch) {
            return;
        }
        // An entry found pinned waits in pinned_order, outside available_order.
        auto available_node = entry_it->pinned_order ? EvictionOrderMap::node_type{}
                                                     : cache_index.available_order.extract(entry_it->available_it);
        auto common_node = cache_index.common_source && entry_it->common_peer != nullptr
                               ? cache_index.common_order.extract(evictionOrder(*entry_it))
                               : EvictionOrderMap::node_type{};
        entry_it->last_access_epoch = access_epoch;
        if (common_node) {
            common_node.key() = evictionOrder(*entry_it);
            cache_index.common_order.insert(std::move(common_node));
        }
        if (available_node) {
            available_node.key() = evictionOrder(*entry_it);
            entry_it->available_it =
                cache_index.available_order.insert(cache_index.available_order.end(), std::move(available_node));
        } else {
            entry_it->pinned_order.key() = evictionOrder(*entry_it);
        }
    }

    CacheEntries& cacheEntries(const BlockPool& pool) {
        return cache_entries_by_pool_.try_emplace(&pool).first->second;
    }
    CacheEntries* findCacheEntries(const BlockPool& pool) {
        auto it = cache_entries_by_pool_.find(&pool);
        return it == cache_entries_by_pool_.end() ? nullptr : &it->second;
    }
    const CacheEntries* findCacheEntries(const BlockPool& pool) const {
        auto it = cache_entries_by_pool_.find(&pool);
        return it == cache_entries_by_pool_.end() ? nullptr : &it->second;
    }
    void validateKey(const CacheKey& key) const {
        _assert(key.group_id == group_id_, "cache key group does not match index");
        _assert(!key.content_hash.empty(), "cache key content hash must not be empty");
    }
    CacheEntryIterator findEntry(CacheEntries& cache_index, const CacheKey& key) {
        validateKey(key);
        auto index_it = cache_index.by_key.find(key);
        return index_it == cache_index.by_key.end() ? cache_index.entries.end() : index_it->second;
    }
    CacheEntryIterator findEntry(CacheEntries& cache_index, CacheBlockLocation location) {
        auto index_it = cache_index.by_location.find(location);
        return index_it == cache_index.by_location.end() ? cache_index.entries.end() : index_it->second;
    }
    ConstCacheEntryIterator findEntry(const CacheEntries& cache_index, const CacheKey& key) const {
        validateKey(key);
        auto index_it = cache_index.by_key.find(key);
        return index_it == cache_index.by_key.end() ? cache_index.entries.end() : index_it->second;
    }
    ConstCacheEntryIterator findEntry(const CacheEntries& cache_index, CacheBlockLocation location) const {
        auto index_it = cache_index.by_location.find(location);
        return index_it == cache_index.by_location.end() ? cache_index.entries.end() : index_it->second;
    }
    void eraseEntry(CacheEntries& cache_index, CacheEntryIterator entry_it) {
        if (CacheEntry* peer = entry_it->common_peer) {
            CacheEntries& source = cache_index.common_source ? cache_index : *cache_index.common_peer;
            source.common_order.erase(evictionOrder(cache_index.common_source ? *entry_it : *peer));
            peer->common_peer = nullptr;
        }
        entry_it->unique_observer.Reset();
        if (!entry_it->pinned_order) {
            cache_index.available_order.erase(entry_it->available_it);
        }
        cache_index.by_key.erase(entry_it->key);
        cache_index.by_location.erase(entry_it->block_ref->Location());
        cache_index.entries.erase(entry_it);
    }

    std::uint32_t group_id_;
    std::unordered_map<const BlockPool*, CacheEntries> cache_entries_by_pool_;
};

}  // namespace tokenspeed
