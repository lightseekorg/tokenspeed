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
#include <memory>
#include <optional>
#include <set>
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
    // Host retention differs from Device admission: proven reuse precedes age.
    // The maximum child value ranks a parent when it must change groups.
    struct HostRetentionOrder {
        bool was_acquired{false};
        std::uint64_t last_access_epoch{0};
        std::uint32_t group_id{0};
        std::int32_t lcm_block_id{0};
        std::int32_t slot_index{0};

        auto operator<=>(const HostRetentionOrder&) const noexcept = default;
    };
    using HostParentIndex = std::set<HostRetentionOrder>;

    // Read-only admission snapshot from one index lookup; owns no block.
    struct CachedBlockMetadata {
        std::uint64_t last_access_epoch{0};
        std::int32_t logical_block_index{-1};
        CacheBoundaryKind boundary_kind{CacheBoundaryKind::kChunk};
        bool was_acquired{false};
    };

    // Non-owning traversal state, bound on first use to one index/pool. The
    // index and pool must outlive it; do not insert, erase, or re-key entries
    // in that tier until traversal is finished. Use a fresh cursor per plan.
    class EvictionCursor {
        friend class PrefixCacheIndex;
        const CacheEntries* cache_index_{nullptr};
        EvictionOrderMap::const_iterator next_;
    };

    // One unpinned entry, paired with the metadata an eviction policy reads.
    struct EvictionCandidate {
        CacheBlockLocation location;
        CachedBlockMetadata metadata;
    };

    explicit PrefixCacheIndex(std::uint32_t group_id) : group_id_{group_id} {}

    PrefixCacheIndex(const PrefixCacheIndex&) = delete;
    PrefixCacheIndex& operator=(const PrefixCacheIndex&) = delete;
    PrefixCacheIndex(PrefixCacheIndex&&) = default;
    PrefixCacheIndex& operator=(PrefixCacheIndex&&) = default;

    // All Host groups share the parent index; shared ownership keeps it stable
    // when a coordinator or group is moved. Existing publications are indexed
    // once here, before allocation can traverse the new ordering.
    void EnableHostRetention(const BlockPool& pool, std::shared_ptr<HostParentIndex> parents, std::int32_t packing,
                             std::int32_t shard_count) {
        _assert(parents && packing > 0 && shard_count > 0 && packing % shard_count == 0,
                "Host retention needs valid packing and bucket geometry");
        CacheEntries& entries = cacheEntries(pool);
        _assert(!entries.host_parents, "Host retention is already configured");
        // Reject another subscriber before marking this state for detachment.
        pool.SetBlockObserver(group_id_, &entries);
        entries.host_parents = std::move(parents);
        entries.host_pool = &pool;
        entries.group_id = group_id_;
        entries.host_packing = packing;
        entries.host_by_bucket.resize(static_cast<std::size_t>(shard_count));
        for (auto entry = entries.entries.begin(); entry != entries.entries.end(); ++entry) {
            insertHostEntry(entries, entry);
            const auto parent = entry->block_ref->Location().lcm_block_id;
            if (!entries.host_parent_values.contains(parent)) {
                refreshHostParent(entries, parent);
            }
        }
    }

    // A fresh cursor per bucket and allocation batch. Reference transitions
    // keep pinned entries out of this non-owning ordering.
    std::optional<CacheBlockLocation> NextHostVictim(const BlockPool& pool, std::int32_t bucket,
                                                     std::optional<HostRetentionOrder>& cursor) const {
        const CacheEntries* entries = findCacheEntries(pool);
        _assert(entries != nullptr && entries->host_parents, "Host retention is not configured");
        const auto& order = entries->host_by_bucket.at(static_cast<std::size_t>(bucket));
        auto next = cursor ? order.upper_bound(*cursor) : order.begin();
        if (next != order.end()) {
            cursor = next->first;
            _assert(next->second->block_ref.unique(), "Host victim index contains a pinned entry");
            return next->second->block_ref->Location();
        }
        return std::nullopt;
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
            touchEntry(cache_index, existing_it, access_epoch, /*acquired=*/false);
            return;
        }
        CacheEntryIterator canonical_it = findEntry(cache_index, key);
        if (canonical_it != cache_index.entries.end()) {
            if (canonical_it->boundary_kind < boundary_kind) {
                canonical_it->boundary_kind = boundary_kind;
            }
            touchEntry(cache_index, canonical_it, access_epoch, /*acquired=*/false);
            block_ref = canonical_it->block_ref;
            return;
        }

        cache_index.entries.push_back(CacheEntry{
            .key = key,
            .block_ref = block_ref,
            .last_access_epoch = access_epoch,
            .logical_block_index = logical_block_index,
            .boundary_kind = boundary_kind,
        });
        CacheEntryIterator entry_it = std::prev(cache_index.entries.end());
        cache_index.by_key.emplace(entry_it->key, entry_it);
        cache_index.by_location.emplace(entry_it->block_ref->Location(), entry_it);
        cache_index.by_eviction_order.emplace(evictionOrder(*entry_it), entry_it);
        insertHostEntry(cache_index, entry_it);
        refreshHostParent(cache_index, entry_it->block_ref->Location().lcm_block_id);
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

    // The key this tier publishes `block_ref` under, when the entry's block is
    // this very block (identity, not just location). A retraction asks this
    // of each data slot to decide which tier images it.
    std::optional<CacheKey> KeyOf(const BlockPool& pool, const CacheBlockRef& block_ref) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr || !block_ref) {
            return std::nullopt;
        }
        ConstCacheEntryIterator entry_it = findEntry(*cache_index, block_ref->Location());
        if (entry_it == cache_index->entries.end() || entry_it->block_ref != block_ref) {
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

    // Appends all unpinned entries of the next epoch containing candidates.
    // Skips fully pinned epochs and returns false at exhaustion. The cursor
    // advances continuously, without a tree lookup for each skipped epoch.
    // Complete epochs let admission apply its policy's tie-breaks locally.
    bool NextEvictionEpoch(const BlockPool& pool, EvictionCursor& cursor, std::vector<EvictionCandidate>& out) const {
        if (cursor.cache_index_ == nullptr) {
            cursor.cache_index_ = findCacheEntries(pool);
            if (cursor.cache_index_ == nullptr) {
                return false;
            }
            cursor.next_ = cursor.cache_index_->by_eviction_order.begin();
        }
        const auto end = cursor.cache_index_->by_eviction_order.end();
        while (cursor.next_ != end) {
            const std::uint64_t epoch = cursor.next_->first.last_access_epoch;
            const std::size_t initial_size = out.size();
            for (; cursor.next_ != end && cursor.next_->first.last_access_epoch == epoch; ++cursor.next_) {
                const CacheEntry& cache_entry = *cursor.next_->second;
                if (cache_entry.block_ref.unique()) {
                    out.push_back(EvictionCandidate{
                        .location = cache_entry.block_ref->Location(),
                        .metadata = metadataOf(cache_entry),
                    });
                }
            }
            if (out.size() != initial_size) {
                return true;
            }
        }
        return false;
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
            touchEntry(*cache_index, entry_it, access_epoch, /*acquired=*/true);
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
    };

    struct CacheEntries : BlockPool::BlockObserver {
        ~CacheEntries() override {
            if (host_pool) {
                host_pool->SetBlockObserver(group_id, nullptr);
                for (const auto& [parent, value] : host_parent_values) {
                    host_parents->erase(value);
                }
            }
        }

        void BlockChanged(CacheBlockLocation location) override {
            const auto child = by_location.find(location);
            if (child != by_location.end()) {
                eraseHostEntry(*this, *child->second);
                insertHostEntry(*this, child->second);
            }
            refreshHostParent(*this, location.lcm_block_id);
        }

        // Owns each CacheEntry once. The maps are non-owning secondary indices
        // into stable list nodes. by_eviction_order additionally keeps the
        // entries sorted by age, so a caller reaches the oldest ones without
        // visiting the rest; AdmissionPlanner still owns the policy that ranks
        // entries within one access epoch.
        CacheEntryList entries;
        std::unordered_map<CacheKey, CacheEntryIterator, CacheKeyHash> by_key;
        std::unordered_map<CacheBlockLocation, CacheEntryIterator, CacheBlockLocationHash> by_location;
        EvictionOrderMap by_eviction_order;
        std::shared_ptr<HostParentIndex> host_parents;
        const BlockPool* host_pool{nullptr};
        std::uint32_t group_id{0};
        std::int32_t host_packing{0};
        std::vector<std::map<HostRetentionOrder, CacheEntryIterator>> host_by_bucket;
        std::unordered_map<std::int32_t, HostParentIndex::iterator> host_parent_values;
    };

    static CachedBlockMetadata metadataOf(const CacheEntry& cache_entry) noexcept {
        return CachedBlockMetadata{
            .last_access_epoch = cache_entry.last_access_epoch,
            .logical_block_index = cache_entry.logical_block_index,
            .boundary_kind = cache_entry.boundary_kind,
            .was_acquired = cache_entry.was_acquired,
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
    // re-keying costs O(log N) per changed epoch, including on prefix hits, in
    // exchange for admission's ordered traversal and early exit. Continued
    // requests can republish under an older epoch, so moving entries to the
    // back of an LRU list would not preserve this order.
    void touchEntry(CacheEntries& cache_index, CacheEntryIterator entry_it, std::uint64_t access_epoch, bool acquired) {
        if (entry_it->last_access_epoch == access_epoch && (!acquired || entry_it->was_acquired)) {
            return;
        }
        eraseHostEntry(cache_index, *entry_it);
        if (entry_it->last_access_epoch != access_epoch) {
            cache_index.by_eviction_order.erase(evictionOrder(*entry_it));
            entry_it->last_access_epoch = access_epoch;
            cache_index.by_eviction_order.emplace(evictionOrder(*entry_it), entry_it);
        }
        entry_it->was_acquired |= acquired;
        insertHostEntry(cache_index, entry_it);
        refreshHostParent(cache_index, entry_it->block_ref->Location().lcm_block_id);
    }

    static HostRetentionOrder hostOrder(const CacheEntry& entry) {
        const auto location = entry.block_ref->Location();
        return {entry.was_acquired, entry.last_access_epoch, entry.key.group_id, location.lcm_block_id,
                location.slot_index};
    }
    static void insertHostEntry(CacheEntries& entries, CacheEntryIterator entry) {
        if (entries.host_parents && entry->block_ref.unique()) {
            const auto bucket =
                static_cast<std::size_t>(entry->block_ref->Location().slot_index) % entries.host_by_bucket.size();
            entries.host_by_bucket[bucket].emplace(hostOrder(*entry), entry);
        }
    }
    static void eraseHostEntry(CacheEntries& entries, const CacheEntry& entry) {
        if (entries.host_parents) {
            const auto bucket =
                static_cast<std::size_t>(entry.block_ref->Location().slot_index) % entries.host_by_bucket.size();
            entries.host_by_bucket[bucket].erase(hostOrder(entry));
        }
    }
    // Only this parent's fixed packing is visited, never the whole Host pool.
    // Occupancy and ownership notifications also cover unpublished transfer
    // destinations, their completion/publication, and cancellation.
    static void refreshHostParent(CacheEntries& entries, std::int32_t parent) {
        if (!entries.host_parents) {
            return;
        }
        auto previous = entries.host_parent_values.find(parent);
        std::optional<HostRetentionOrder> value;
        for (std::int32_t slot = 0; slot < entries.host_packing; ++slot) {
            if (!entries.host_pool->IsOccupied({.lcm_block_id = parent, .slot_index = slot})) {
                continue;
            }
            const auto child = entries.by_location.find({.lcm_block_id = parent, .slot_index = slot});
            if (child == entries.by_location.end() || !child->second->block_ref.unique()) {
                value.reset();
                break;
            }
            const auto order = hostOrder(*child->second);
            value = value ? std::max(*value, order) : order;
        }
        if (previous != entries.host_parent_values.end()) {
            if (value && *previous->second == *value) {
                return;
            }
            entries.host_parents->erase(previous->second);
        }
        if (value) {
            entries.host_parent_values.insert_or_assign(parent, entries.host_parents->insert(*value).first);
        } else if (previous != entries.host_parent_values.end()) {
            entries.host_parent_values.erase(previous);
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
        const std::int32_t parent = entry_it->block_ref->Location().lcm_block_id;
        eraseHostEntry(cache_index, *entry_it);
        cache_index.by_eviction_order.erase(evictionOrder(*entry_it));
        cache_index.by_key.erase(entry_it->key);
        cache_index.by_location.erase(entry_it->block_ref->Location());
        cache_index.entries.erase(entry_it);
        refreshHostParent(cache_index, parent);
    }

    std::uint32_t group_id_;
    std::unordered_map<const BlockPool*, CacheEntries> cache_entries_by_pool_;
};

}  // namespace tokenspeed
