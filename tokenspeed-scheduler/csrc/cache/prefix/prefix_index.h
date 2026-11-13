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
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_block_ref.h"
#include "cache/core/cache_types.h"
#include "cache/core/fixed_slot_resource.h"
#include "cache/core/prefetch.h"
#include "cache/core/session_cache_tag.h"
#include "cache/prefix/boundary_tree.h"
#include "cache/prefix/flat_pointer_index.h"
#include "cache/prefix/segmented_location_index.h"
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
    using CacheEntryList = std::pmr::list<CacheEntry>;
    using CacheEntryIterator = CacheEntryList::iterator;
    using ConstCacheEntryIterator = CacheEntryList::const_iterator;

    // Index key: oldest access epoch first, with location as a stable tie-break.
    // The key's location is the entry's block location: every change of
    // block_ref or last_access_epoch re-keys through reorderEntry, and
    // NextEvictionEpoch reports the key's location as the candidate's.
    struct EvictionOrder {
        std::uint64_t last_access_epoch{0};
        std::int32_t lcm_block_id{0};
        std::int32_t slot_index{0};

        auto operator<=>(const EvictionOrder&) const noexcept = default;
    };

    using EvictionOrderMap = std::map<EvictionOrder, CacheEntryIterator>;

    // All mutations maintain the insertion hint, the last insertion, including
    // erasing its node. std::map checks both neighbours of the hint, so
    // appends and consecutive locations within an epoch, in either direction,
    // insert in amortized constant time; arbitrary epochs/locations still use
    // std::map's ordering.
    class OrderIndex {
    public:
        OrderIndex() = default;
        // The hint points into this very map: a copy or a move would leave it
        // pointing at the original.
        OrderIndex(const OrderIndex&) = delete;
        OrderIndex& operator=(const OrderIndex&) = delete;
        OrderIndex(OrderIndex&&) = delete;
        OrderIndex& operator=(OrderIndex&&) = delete;

        auto begin() const { return entries_.begin(); }
        auto end() const { return entries_.end(); }
        auto rbegin() const { return entries_.rbegin(); }
        auto rend() const { return entries_.rend(); }
        auto upper_bound(EvictionOrder key) const { return entries_.upper_bound(key); }
        std::size_t size() const noexcept { return entries_.size(); }

        EvictionOrderMap::iterator Insert(EvictionOrder key, CacheEntryIterator entry) {
            const auto position = entries_.emplace_hint(hint_, key, entry);
            FatalCheck(position->second == entry, "an ordered entry has a unique epoch and location");
            hint_ = position;
            return position;
        }
        EvictionOrderMap::iterator Insert(EvictionOrderMap::node_type& node) noexcept {
            const auto position = entries_.insert(hint_, std::move(node));
            FatalCheck(node.empty(), "an extracted node restores its unique order key");
            hint_ = position;
            return position;
        }
        EvictionOrderMap::node_type Extract(EvictionOrderMap::iterator position) noexcept {
            if (hint_ == position) hint_ = successor(position);
            return entries_.extract(position);
        }
        void Erase(EvictionOrderMap::iterator position) noexcept {
            if (hint_ == position) hint_ = successor(position);
            entries_.erase(position);
        }

    private:
        // Re-keying to a new epoch leaves the hint on the last node, whose
        // std::next climbs the tree's whole right edge; the map reaches that
        // node in O(1) as end()'s predecessor.
        EvictionOrderMap::const_iterator successor(EvictionOrderMap::const_iterator position) const noexcept {
            return position == std::prev(entries_.end()) ? entries_.end() : std::next(position);
        }

        EvictionOrderMap entries_;
        EvictionOrderMap::const_iterator hint_{entries_.end()};
    };

    // Owns one ordered-index registration: its node leaves the order with it.
    // The map outlives all registrations; stable entries never move this handle.
    class OrderRegistration {
    public:
        OrderRegistration() = default;
        OrderRegistration(const OrderRegistration&) = delete;
        OrderRegistration& operator=(const OrderRegistration&) = delete;
        ~OrderRegistration() { Reset(); }

        void Insert(OrderIndex& order, EvictionOrder key, CacheEntryIterator entry) {
            FatalCheck(order_ == nullptr, "an order registration is inserted once");
            position_ = order.Insert(key, entry);
            order_ = &order;
        }
        bool Registered() const noexcept { return order_ != nullptr; }
        void Rekey(EvictionOrder key) noexcept {
            FatalCheck(Registered(), "only a registered entry can change age or location");
            auto node = order_->Extract(position_);
            node.key() = key;
            position_ = order_->Insert(node);
        }
        void Reset() noexcept {
            if (order_ == nullptr) return;
            order_->Erase(position_);
            order_ = nullptr;
        }
        // A hint for Reset or Rekey: prefetches the registered node's value.
        void Prefetch() const noexcept {
            if (order_ != nullptr) PrefetchObject(*position_);
        }

    protected:
        OrderIndex* order_{nullptr};
        EvictionOrderMap::iterator position_{};
    };

    // The available-order registration, which a pinned entry suspends: it
    // keeps the extracted node, so restoring the order never allocates during
    // a block destructor. Only this one pays for a node handle.
    class SuspendableOrderRegistration : private OrderRegistration {
    public:
        SuspendableOrderRegistration() = default;
        // A suspended node is not in the order: it leaves with the handle.
        ~SuspendableOrderRegistration() {
            if (Suspended()) order_ = nullptr;
        }

        using OrderRegistration::Insert;
        bool Suspended() const noexcept { return !suspended_.empty(); }
        void Suspend() noexcept {
            FatalCheck(Registered(), "only a registered entry can be suspended");
            if (!Suspended()) suspended_ = order_->Extract(position_);
        }
        void Resume() noexcept {
            FatalCheck(Suspended(), "only a suspended order node can be restored");
            position_ = order_->Insert(suspended_);
        }
        void Rekey(EvictionOrder key) noexcept {
            if (Suspended()) {
                suspended_.key() = key;
                return;
            }
            OrderRegistration::Rekey(key);
        }
        // A suspended node is not in the order: nothing to prefetch.
        void Prefetch() const noexcept {
            if (!Suspended()) OrderRegistration::Prefetch();
        }

    private:
        EvictionOrderMap::node_type suspended_{};
    };

public:
    // The part of an entry's metadata an eviction policy ranks it by. Epoch
    // traversals copy it for every entry they scan; the session owner would
    // cost each one another cache line and a reference count. A field the
    // policy starts ranking by belongs here.
    struct EvictionMetadata {
        std::uint64_t last_access_epoch{0};
        std::int32_t logical_block_index{-1};
        CacheBoundaryKind boundary_kind{CacheBoundaryKind::kChunk};
        bool was_acquired{false};
    };

    // Read-only admission snapshot from one index lookup; owns no block.
    struct CachedBlockMetadata {
        EvictionMetadata eviction;
        // The largest epoch any acquisition or publication touched the entry
        // with. eviction.last_access_epoch can move back below it (a
        // publication under an older epoch,
        // CacheCoordinator::MarkSupersededStateInput); this one never does.
        std::uint64_t max_access_epoch{0};
        SessionCacheOwner session;
        // Demote dated the entry back and no access has touched it since. A
        // session's demotion (DemoteSessions) and a superseded input date an
        // entry back too, but never set this.
        bool demoted{false};
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
        EvictionMetadata metadata;
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

    // The entry at one location while Evict may take it: the index is its
    // block's only owner (FindEvictable). It borrows the entry, so it is
    // neither copied nor moved, and lasts only until this tier's entries or
    // the block's owners change: read what the eviction needs, then hand it
    // to Evict. Its reads and Evict check that no entry joined, left or
    // moved (Relocate) in the tier since FindEvictable, and Evict that the
    // block is still unshared.
    // A failed read throws; a failed Evict aborts, since its caller may have
    // recorded the eviction already. An empty handle -- no entry there, a
    // pinned one, or one already evicted -- is false.
    class EvictableEntry {
    public:
        EvictableEntry(const EvictableEntry&) = delete;
        EvictableEntry& operator=(const EvictableEntry&) = delete;

        explicit operator bool() const noexcept { return cache_index_ != nullptr; }
        const CacheKey& key() const { return entry().key; }
        CachedBlockMetadata metadata() const { return metadataOf(entry()); }
        // The boundary-tree node the entry holds (TrackBoundaries), still
        // counting the entry; null in an index never bound.
        BoundaryTree::Node* boundary_node() const {
            const CacheEntry& found = entry();
            return found.boundary ? &found.boundary.node() : nullptr;
        }

    private:
        friend class PrefixCacheIndex;
        EvictableEntry() = default;
        EvictableEntry(CacheEntries& cache_index, CacheEntryIterator entry)
            : cache_index_{&cache_index}, entry_{entry}, membership_{cache_index.membership} {}

        const CacheEntry& entry() const {
            _assert(cache_index_ != nullptr, "an empty eviction handle has no entry");
            _assert(cache_index_->membership == membership_, "an eviction handle outlived a change to its tier");
            return *entry_;
        }

        CacheEntries* cache_index_{nullptr};
        CacheEntryIterator entry_{};
        // The tier's CacheEntries::membership when the entry was found.
        std::uint64_t membership_{0};
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

    // Every entry of these tiers holds the boundary-tree node of its hash
    // (BoundaryTree::Hold) for as long as it is indexed, so the node follows
    // the entries: present on a tier while some group indexes the hash there,
    // a checkpoint there while a State group does. `state` and `closed` give
    // this group's kind. The coordinator binds each index once, before
    // anything is registered; an index never bound holds nothing.
    void TrackBoundaries(BoundaryTree& tree, const BlockPool& device_pool, const BlockPool* host_pool, bool state,
                         bool closed) {
        const auto bind = [&](const BlockPool& pool, CacheTier tier) {
            CacheEntries& cache_index = cacheEntries(pool);
            _assert(cache_index.boundary_tree == nullptr && cache_index.entries.empty(),
                    "an index tier binds its boundary tree once, before anything is registered");
            cache_index.boundary_tree = &tree;
            cache_index.boundary_kind = BoundaryTree::EntryKind{.tier = tier, .state = state, .closed = closed};
        };
        bind(device_pool, CacheTier::kDevice);
        if (host_pool != nullptr) {
            bind(*host_pool, CacheTier::kHost);
        }
    }

    // Registers block_ref under key. If key already has a canonical block,
    // block_ref is replaced with a reference to that block.
    void Register(const BlockPool& pool, CacheBlockRef& block_ref, const CacheKey& key, std::uint64_t access_epoch,
                  std::int32_t logical_block_index, CacheBoundaryKind boundary_kind,
                  std::vector<std::pair<CacheKey, CacheBlockRef>>* newly_cached, SessionCacheOwner session) {
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
            mergeSession(*existing_it, session);
            touchEntry(existing_it, access_epoch);
            return;
        }
        // One key lookup finds the canonical entry or indexes a new one,
        // created with its key; a throw on the way leaves neither.
        const auto [entry_it, inserted] = cache_index.by_key.FindOrInsert(
            key, [&] { return cache_index.entries.emplace(cache_index.entries.end(), key); });
        if (!inserted) {
            if (entry_it->boundary_kind < boundary_kind) {
                entry_it->boundary_kind = boundary_kind;
            }
            mergeSession(*entry_it, session);
            touchEntry(entry_it, access_epoch);
            block_ref = entry_it->block_ref;
            return;
        }

        ++cache_index.membership;
        CacheEntry& entry = *entry_it;
        entry.block_ref = block_ref;
        entry.last_access_epoch = session && session->demoted ? 0 : access_epoch;
        entry.logical_block_index = logical_block_index;
        entry.boundary_kind = boundary_kind;
        entry.max_access_epoch = access_epoch;
        entry.session = std::move(session);
        cache_index.by_location.Set(entry_it->block_ref->Location(), entry_it);
        entry.available_order.Insert(cache_index.available_order, evictionOrder(entry), entry_it);
        if (cache_index.boundary_tree != nullptr) {
            // The tree keys its nodes by content hash alone.
            _assert(key.namespace_id == kDefaultCacheNamespaceId, "a tracked boundary lies in the default namespace");
            entry_it->boundary = cache_index.boundary_tree->HoldEntry(key.content_hash, cache_index.boundary_kind);
        }
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
                            std::vector<std::pair<CacheKey, CacheBlockRef>>* newly_cached, SessionCacheOwner session) {
        _assert(first_logical_block >= 0, "first_logical_block must be >= 0");
        _assert(blocks.size() == keys.size(), "one key per published block");
        for (std::size_t j = 0; j < keys.size(); ++j) {
            CacheBlockRef& block_ref = blocks[j];
            if (!block_ref) {
                continue;
            }
            Register(pool, block_ref, keys[j], access_epoch, first_logical_block + static_cast<std::int32_t>(j),
                     boundary_kind, newly_cached, session);
        }
    }

    bool Contains(const BlockPool& pool, const CacheKey& key) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        return cache_index != nullptr && findEntry(*cache_index, key) != cache_index->entries.end();
    }
    std::optional<CacheBlockLocation> LocationOf(const BlockPool& pool, const CacheKey& key) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) return std::nullopt;
        const auto entry = findEntry(*cache_index, key);
        if (entry == cache_index->entries.end()) return std::nullopt;
        return entry->block_ref->Location();
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
            const auto entry = item.second->by_location.Find(block_ref->Location());
            return entry != item.second->entries.end() && entry->block_ref == block_ref;
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

    // Resume after replaying a saved prefix of complete epochs. Seek by key:
    // a pre-scan can extract pinned nodes, invalidating a saved next iterator.
    void SeekAfterEvictionEpoch(const BlockPool& pool, EvictionCursor& cursor, std::uint64_t epoch) const {
        cursor.cache_index_ = findCacheEntries(pool);
        if (cursor.cache_index_ != nullptr) {
            cursor.next_ = cursor.cache_index_->available_order.upper_bound(EvictionOrder{
                epoch, std::numeric_limits<std::int32_t>::max(), std::numeric_limits<std::int32_t>::max()});
        }
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
                    // The order key holds the entry's location (EvictionOrder),
                    // so the block is read only for its owner count.
                    out.push_back(EvictionCandidate{
                        .location = {.lcm_block_id = current->first.lcm_block_id,
                                     .slot_index = current->first.slot_index},
                        .metadata = evictionMetadataOf(cache_entry),
                    });
                } else {
                    // Keep the node for allocation-free restoration when the
                    // last request/transfer owner releases this block. Ordinary
                    // temporary Find() references need no index maintenance.
                    cache_entry.available_order.Suspend();
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
                    .metadata = evictionMetadataOf(cache_entry),
                });
            }
        }
        return candidates;
    }

    // Adjust LRU only. Active requests and transfers keep their references;
    // shared or untagged prefixes keep their existing eviction order.
    std::size_t DemoteSession(const BlockPool& pool, const SessionCacheOwner& session) {
        return DemoteSessions(pool, {session.get()});
    }
    std::size_t DemoteSessions(const BlockPool& pool, const std::unordered_set<const SessionCacheTag*>& sessions) {
        CacheEntries* cache_index = findCacheEntries(pool);
        if (!cache_index) return 0;
        std::size_t changed = 0;
        for (auto it = cache_index->entries.begin(); it != cache_index->entries.end(); ++it) {
            if (!it->session || !sessions.contains(it->session.get()) || !it->block_ref.unique() ||
                it->last_access_epoch == 0) {
                continue;
            }
            touchEntry(it, 0);
            ++changed;
        }
        return changed;
    }

    // Dates key's entry back to `epoch`, never forward, marks it demoted
    // (CachedBlockMetadata::demoted) and forgets its acquisition
    // (CacheCoordinator::DemoteEntries); an entry a session's demotion dated
    // there already is marked too. False when the tier does not cache key, the
    // entry is demoted already or older than `epoch`, or anything holds its
    // block beyond the index and the write_back_holds references of an
    // eviction write-back that publishes into this very entry. A later hit,
    // touch or republication dates it forward again and ends the demotion.
    bool Demote(const BlockPool& pool, const CacheKey& key, std::uint64_t epoch, std::uint32_t write_back_holds) {
        CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return false;
        }
        const CacheEntryIterator entry_it = findEntry(*cache_index, key);
        if (entry_it == cache_index->entries.end() || entry_it->demoted || entry_it->last_access_epoch < epoch ||
            entry_it->block_ref.use_count() > 1 + write_back_holds) {
            return false;
        }
        markDemoted(entry_it, epoch);
        return true;
    }

    // Marks key's entry demoted, as Demote does, when it carries `epoch`
    // already, whatever holds it: an eviction write-back publishes a copy a
    // demotion reached on its way (CacheCoordinator::CacheHostBlock).
    void MarkDemoted(const BlockPool& pool, const CacheKey& key, std::uint64_t epoch) {
        CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return;
        }
        const CacheEntryIterator entry_it = findEntry(*cache_index, key);
        if (entry_it != cache_index->entries.end() && entry_it->last_access_epoch == epoch) {
            markDemoted(entry_it, epoch);
        }
    }

    // The key of the entry at `location` when Evict would take it: the index
    // is its block's only owner.
    std::optional<CacheKey> UnpinnedKeyAt(const BlockPool& pool, CacheBlockLocation location) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return std::nullopt;
        }
        const ConstCacheEntryIterator entry_it = findEntry(*cache_index, location);
        if (entry_it == cache_index->entries.end() || !entry_it->block_ref.unique()) {
            return std::nullopt;
        }
        return entry_it->key;
    }

    std::optional<CacheKey> Evict(const BlockPool& pool, CacheBlockLocation location) {
        EvictableEntry evictable = FindEvictable(pool, location);
        if (!evictable) {
            return std::nullopt;
        }
        return Evict(std::move(evictable));
    }

    // The entry at `location` when Evict may take it, located and checked
    // with one lookup; an empty handle when there is none or it is pinned.
    // A caller reads what the entry's eviction records -- key, metadata, the
    // tree node still counting it -- before the entry leaves.
    EvictableEntry FindEvictable(const BlockPool& pool, CacheBlockLocation location) {
        CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return EvictableEntry{};
        }
        const CacheEntryIterator entry_it = findEntry(*cache_index, location);
        if (entry_it == cache_index->entries.end() || !entry_it->block_ref.unique()) {
            return EvictableEntry{};
        }
        return EvictableEntry{*cache_index, entry_it};
    }

    // Takes the entry `evictable` found out of the index and leaves the
    // handle empty. The returned key owns its bytes. A handle that no longer
    // vouches for its entry aborts instead of throwing: the caller may have
    // recorded this eviction (counters, a write-back member) already, and a
    // throw would leave that record for an entry still indexed.
    CacheKey Evict(EvictableEntry&& evictable) {
        FatalCheck(evictable.cache_index_ != nullptr, "an empty eviction handle has no entry");
        FatalCheck(evictable.cache_index_->membership == evictable.membership_,
                   "an eviction handle outlived a change to its tier");
        const CacheEntry& entry = *evictable.entry_;
        FatalCheck(entry.block_ref.unique(), "an evictable entry stays unpinned until it is evicted");
        CacheKey key = entry.key;
        eraseEntry(*evictable.cache_index_, evictable.entry_);
        evictable.cache_index_ = nullptr;
        return key;
    }

    // A hint for evicting entries of `pool`'s tier later, in three stages,
    // one victim each, issued furthest ahead first so that each stage reads
    // only lines an earlier call prefetched: for `slot`, its slot in the
    // location index; for `entry`, the entry found there; for `removal`, what
    // removing the entry found there touches beyond it -- its key's slot, its
    // block's control, its order nodes, and its boundary-tree node with that
    // node's slot. A null location skips its stage. Changes nothing. Each
    // stage finds its entry afresh and reads only that live entry and what it
    // holds, so an entry evicted in between, or never indexed, is skipped;
    // nothing is kept across calls.
    void PrefetchEvictions(const BlockPool& pool, const CacheBlockLocation* slot, const CacheBlockLocation* entry,
                           const CacheBlockLocation* removal) const noexcept {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) return;
        if (slot != nullptr) cache_index->by_location.Prefetch(*slot);
        if (entry != nullptr) {
            const ConstCacheEntryIterator entry_it = findEntry(*cache_index, *entry);
            if (entry_it != cache_index->entries.end()) PrefetchObject(*entry_it);
        }
        if (removal == nullptr) return;
        const ConstCacheEntryIterator entry_it = findEntry(*cache_index, *removal);
        if (entry_it == cache_index->entries.end()) return;
        cache_index->by_key.Prefetch(entry_it->key);
        entry_it->block_ref.Prefetch();
        entry_it->available_order.Prefetch();
        entry_it->common_order.Prefetch();
        if (cache_index->boundary_tree != nullptr) {
            cache_index->boundary_tree->PrefetchRelease(entry_it->boundary, entry_it->key.content_hash);
        }
    }

    // Moves the entry at `from` onto `to`, a block of the same tier the caller
    // acquired for a copy of the entry's block: a lone request's compaction
    // (CacheCoordinator::CompactLoneRequest), which keeps the bucket. The
    // entry keeps its key, epochs, logical block index, boundary kind,
    // session, acquisition and demotion marks, pairing and its hold on the
    // boundary-tree node -- nothing a hash, KV event or checkpoint depends on
    // changes -- and only its block, location and place in the orders
    // follow. It is pinned while the caller's reference to `to` lives; the
    // old block is the caller's to release.
    void Relocate(const BlockPool& pool, CacheBlockLocation from, const CacheBlockRef& to) {
        _assert(to && to.IsOwnedBy(pool), "the destination block must belong to the entry's pool");
        _assert(pool.BoundGroup(to->Location().lcm_block_id) == group_id_,
                "the destination block must belong to the prefix index group");
        CacheEntries* cache_index = findCacheEntries(pool);
        _assert(cache_index != nullptr, "a moved block is indexed on its tier");
        CacheEntryIterator entry_it = findEntry(*cache_index, from);
        _assert(entry_it != cache_index->entries.end(), "a moved block is indexed at its location");
        _assert(findEntry(*cache_index, to->Location()) == cache_index->entries.end(),
                "an entry moves onto an unindexed location");
        // Counted before the move, so that a throw below cannot leave an
        // eviction handle vouching for a moved entry.
        ++cache_index->membership;
        // The pin watch, if any, follows the block (restored below).
        cache_index->by_location.Set(to->Location(), entry_it);
        entry_it->unique_observer.Reset();
        cache_index->by_location.Erase(from);
        entry_it->block_ref = to;
        reorderEntry(entry_it);
        if (entry_it->available_order.Suspended()) {
            entry_it->unique_observer.Observe(entry_it->block_ref, restoreAvailable, &*entry_it);
        }
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
                               const GroupPrefixProbe& probe, std::uint64_t access_epoch, SessionCacheOwner session) {
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
            CacheEntryIterator entry_it =
                findMatchedEntry(*cache_index, pool, keys[static_cast<std::size_t>(begin_blocks) + i], probe, i);
            _assert(entry_it != cache_index->entries.end(),
                    "cached block disappeared between match probe and acquisition");
            // The returned RAII owner pins the block. As for Find(), traversal
            // suspends its order node only if it actually reaches this entry.
            match.blocks[i] = entry_it->block_ref;
            entry_it->was_acquired = true;
            mergeSession(*entry_it, session);
            touchEntry(entry_it, access_epoch);
        }
        return match;
    }

    // Present source blocks refresh matching entries without pinning or inserting them.
    // Reuse by another or untagged owner also clears exclusivity on this tier.
    void TouchExisting(const BlockPool& pool, std::span<const CacheKey> keys, std::span<const CacheBlockRef> blocks,
                       std::uint64_t access_epoch, SessionCacheOwner session) {
        _assert(blocks.size() <= keys.size(), "source block span exceeds key span");
        CacheEntries* entries = findCacheEntries(pool);
        if (entries == nullptr) {
            return;
        }
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            if (!blocks[i]) {
                continue;
            }
            auto entry = findEntry(*entries, keys[i]);
            if (entry != entries->entries.end()) {
                mergeSession(*entry, session);
                touchEntry(entry, std::max(entry->last_access_epoch, access_epoch));
            }
        }
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
            const auto entry_it =
                findMatchedEntry(*cache_index, pool, keys[static_cast<std::size_t>(begin_blocks) + i], probe, i);
            _assert(entry_it != cache_index->entries.end(), "cached block disappeared between match probes");
            locations.push_back(entry_it->block_ref->Location());
        }
        return locations;
    }

private:
    struct CacheEntry {
        explicit CacheEntry(const CacheKey& key) : key{key} {}

        // The entry's key in by_key, fixed for its lifetime.
        const CacheKey key;
        CacheBlockRef block_ref;
        std::uint64_t last_access_epoch{0};
        // Position in the request's logical prefix. Host-only entries may not
        // have a device-table position yet.
        std::int32_t logical_block_index{-1};
        CacheBoundaryKind boundary_kind{CacheBoundaryKind::kChunk};
        // Set only after a successful request admission acquires this entry.
        bool was_acquired{false};
        // See CachedBlockMetadata::demoted.
        bool demoted{false};
        // See CachedBlockMetadata::max_access_epoch.
        std::uint64_t max_access_epoch{0};
        SuspendableOrderRegistration available_order;
        OrderRegistration common_order;
        // Cancel callbacks before either registration or the block is destroyed.
        CacheBlockUniqueObserver unique_observer;
        CacheEntry* common_peer{nullptr};
        SessionCacheOwner session;
        // The entry's claim on the boundary-tree node of its hash, in a bound
        // index (TrackBoundaries): dropped with the entry.
        BoundaryTree::Hold boundary;
    };

    static void mergeSession(CacheEntry& entry, const SessionCacheOwner& session) {
        if (entry.session != session) entry.session.reset();
    }

    struct EntryKey {
        const CacheKey& operator()(CacheEntryIterator entry) const noexcept { return entry->key; }
    };

    struct CacheEntries {
        CacheEntries(std::int32_t packing, std::int32_t parents) : by_location{packing, parents, entries.end()} {}
        // The entry list allocates from entry_memory, the order indices and
        // lookup tables point into the list, and cursors and common peers
        // point at this object: it never copies or moves.
        CacheEntries(const CacheEntries&) = delete;
        CacheEntries& operator=(const CacheEntries&) = delete;
        CacheEntries(CacheEntries&&) = delete;
        CacheEntries& operator=(CacheEntries&&) = delete;
        // Owns each CacheEntry once. The maps are non-owning secondary indices
        // into stable list nodes. available_order keeps eviction candidates
        // sorted by age; common_order keeps shared keys in source age order.
        // AdmissionPlanner owns the policy that ranks one access epoch.
        FixedSlotResource entry_memory{256};
        // Registration destructors erase their nodes, so maps outlive entries.
        OrderIndex available_order;
        OrderIndex common_order;
        CacheEntryList entries{&entry_memory};
        // Both lookups answer entries.end() for an absent entry.
        FlatPointerIndex<CacheEntryIterator, CacheKey, EntryKey, CacheKeyHash> by_key{entries.end()};
        SegmentedLocationIndex<CacheEntryIterator> by_location;
        // Traversal suspends entries it finds pinned by matched acquisition,
        // Find(), or any other owner.
        // Ownership notifications restore them without changing their age.
        // Only the source keeps the order. Entries in both pools point at
        // each other while their keys coexist; erasing either unlinks them.
        CacheEntries* common_peer{nullptr};
        bool common_source{false};
        // TrackBoundaries: the tree each entry holds a node of, and what an
        // entry of this tier stands for there.
        BoundaryTree* boundary_tree{nullptr};
        BoundaryTree::EntryKind boundary_kind{};
        // Counts the entries that joined, left or moved, so an EvictableEntry
        // can tell that the entry it borrows may be gone or elsewhere.
        std::uint64_t membership{0};
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
        source_entry->common_order.Insert(source.common_order, evictionOrder(*source_entry), source_entry);
        // The source side counts the pair in the node.
        source_entry->boundary.SetPaired(true);
    }

    static void restoreAvailable(void* context) noexcept {
        auto& entry = *static_cast<CacheEntry*>(context);
        entry.available_order.Resume();
    }

    static CachedBlockMetadata metadataOf(const CacheEntry& cache_entry) noexcept {
        return CachedBlockMetadata{
            .eviction = evictionMetadataOf(cache_entry),
            .max_access_epoch = cache_entry.max_access_epoch,
            .session = cache_entry.session,
            .demoted = cache_entry.demoted,
        };
    }
    static EvictionMetadata evictionMetadataOf(const CacheEntry& cache_entry) noexcept {
        return EvictionMetadata{
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
    // The only writer of last_access_epoch: an entry's location changes only
    // in Relocate, re-keyed the same way. Suspended nodes change their key
    // in place; registered nodes use the map's hint or O(log N) insertion.
    // This preserves admission's ordered traversal and early exit. Continued requests can republish
    // under an older epoch, so moving entries to the back of an LRU list
    // would not preserve this order.
    void touchEntry(CacheEntryIterator entry_it, std::uint64_t access_epoch) {
        entry_it->max_access_epoch = std::max(entry_it->max_access_epoch, access_epoch);
        // A touch ends a demotion; markDemoted marks the entry after its own.
        entry_it->demoted = false;
        if (entry_it->session && entry_it->session->demoted) access_epoch = 0;
        if (entry_it->last_access_epoch == access_epoch) {
            return;
        }
        entry_it->last_access_epoch = access_epoch;
        reorderEntry(entry_it);
    }
    // Re-keys the entry's order nodes to its current key: in available_order or, while pinned, in the
    // node kept aside, and in common_order when this tier keeps it.
    void reorderEntry(CacheEntryIterator entry_it) {
        const EvictionOrder key = evictionOrder(*entry_it);
        entry_it->available_order.Rekey(key);
        if (entry_it->common_order.Registered()) entry_it->common_order.Rekey(key);
    }
    // Demote's and MarkDemoted's common step.
    void markDemoted(CacheEntryIterator entry_it, std::uint64_t epoch) {
        entry_it->was_acquired = false;
        touchEntry(entry_it, epoch);
        entry_it->demoted = true;
    }

    CacheEntries& cacheEntries(const BlockPool& pool) {
        if (CacheEntries* cache_index = findCacheEntries(pool)) {
            return *cache_index;
        }
        return *cache_entries_by_pool_
                    .emplace_back(&pool, std::make_unique<CacheEntries>(pool.Packing(group_id_), pool.NumLcmBlocks()))
                    .second;
    }
    CacheEntries* findCacheEntries(const BlockPool& pool) {
        return const_cast<CacheEntries*>(std::as_const(*this).findCacheEntries(pool));
    }
    const CacheEntries* findCacheEntries(const BlockPool& pool) const {
        for (const auto& [owner, cache_index] : cache_entries_by_pool_) {
            if (owner == &pool) {
                return cache_index.get();
            }
        }
        return nullptr;
    }
    void validateKey(const CacheKey& key) const {
        _assert(key.group_id == group_id_, "cache key group does not match index");
        _assert(!key.content_hash.empty(), "cache key content hash must not be empty");
    }
    CacheEntryIterator findMatchedEntry(const CacheEntries& entries, const BlockPool& pool, const CacheKey& key,
                                        const GroupPrefixProbe& probe, std::size_t offset) const {
        validateKey(key);
        if (probe.location_pool == &pool && offset < probe.locations.size() &&
            probe.locations[offset].lcm_block_id > 0) {
            const auto entry = entries.by_location.Find(probe.locations[offset]);
            if (entry != entries.entries.end() && entry->key == key) return entry;
        }
        return entries.by_key.Find(key);
    }
    // An absent entry is entries.end().
    CacheEntryIterator findEntry(CacheEntries& cache_index, const CacheKey& key) {
        validateKey(key);
        return cache_index.by_key.Find(key);
    }
    CacheEntryIterator findEntry(CacheEntries& cache_index, CacheBlockLocation location) {
        return cache_index.by_location.Find(location);
    }
    ConstCacheEntryIterator findEntry(const CacheEntries& cache_index, const CacheKey& key) const {
        validateKey(key);
        return cache_index.by_key.Find(key);
    }
    ConstCacheEntryIterator findEntry(const CacheEntries& cache_index, CacheBlockLocation location) const {
        return cache_index.by_location.Find(location);
    }
    void eraseEntry(CacheEntries& cache_index, CacheEntryIterator entry_it) {
        if (CacheEntry* peer = entry_it->common_peer) {
            CacheEntries& source = cache_index.common_source ? cache_index : *cache_index.common_peer;
            CacheEntry& source_entry = cache_index.common_source ? *entry_it : *peer;
            source_entry.common_order.Reset();
            source_entry.boundary.SetPaired(false);
            peer->common_peer = nullptr;
        }
        entry_it->unique_observer.Reset();
        FatalCheck(cache_index.by_key.ExtractObject(entry_it->key, *entry_it) == entry_it,
                   "an indexed cache entry keeps its key");
        cache_index.by_location.Erase(entry_it->block_ref->Location());
        cache_index.entries.erase(entry_it);
        ++cache_index.membership;
    }

    std::uint32_t group_id_;
    // One per pool this index serves (the coordinator's Device and Host
    // tiers): scanning these few addresses costs less than hashing one on
    // every lookup. Each CacheEntries keeps its address: cursors and peers
    // point at it.
    std::vector<std::pair<const BlockPool*, std::unique_ptr<CacheEntries>>> cache_entries_by_pool_;
};

}  // namespace tokenspeed
