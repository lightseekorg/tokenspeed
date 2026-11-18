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

#include "cache/coordinator/cache_coordinator.h"

#include "core/profile_span.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <memory_resource>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <iterator>
#include <utility>

#include "cache/prefix/prefix_matcher.h"
#include "utils.h"

namespace tokenspeed {

std::size_t CacheCoordinator::DemoteSession(const std::string& session_id, bool ended) {
    return DemoteSessions({session_id}, ended);
}

std::size_t CacheCoordinator::DemoteSessions(const std::vector<std::string>& ids, bool ended) {
    return demoteOwners(sessions_.DemoteSessions(ids, ended));
}

std::size_t CacheCoordinator::DemoteCacheGroups(const std::vector<std::string>& ids, bool ended) {
    return demoteOwners(sessions_.DemoteCacheGroups(ids, ended));
}

std::size_t CacheCoordinator::demoteOwners(const SessionRegistry::DemotedOwners& demoted) {
    if (demoted.tags.empty()) return 0;
    std::size_t changed = 0;
    for (auto& group : store_.Groups()) {
        changed += group.Index().DemoteSessions(store_.DevicePool(), demoted.tags);
        if (store_.HostPool()) changed += group.Index().DemoteSessions(*store_.HostPool(), demoted.tags);
    }
    return changed;
}

CacheCoordinator::CacheCoordinator(std::vector<CacheGroup> groups, std::int32_t prefix_granularity, BlockPool& pool,
                                   bool enable_l3_storage, BlockPool* host_pool, BlockPool* snapshot_pool,
                                   bool stream_device_cache_to_host, bool write_back_evictions,
                                   std::int64_t checkpoint_retention_requests,
                                   std::int32_t checkpoint_retention_leaf_distance)
    : store_{std::move(groups),
             prefix_granularity,
             pool,
             host_pool,
             snapshot_pool,
             write_back_evictions,
             checkpoint_retention_leaf_distance},
      stream_device_cache_to_host_{stream_device_cache_to_host && host_pool != nullptr},
      write_back_evictions_{write_back_evictions},
      history_{store_.Checkpoints(), checkpoint_retention_requests},
      storage_{store_, prober_, host_, enable_l3_storage && host_pool != nullptr} {
    _assert(!write_back_evictions_ || host_pool != nullptr, "writing evictions back needs a Host pool");
    _assert(!(stream_device_cache_to_host_ && write_back_evictions_),
            "the Host tier is fed by publication or by eviction, not both");
}

CacheCoordinator::~CacheCoordinator() {
    // The recorded write-backs and the index entries point at the store's
    // boundary tree and at the evictor's pending keys: drop them while both
    // still exist.
    evictor_.ClearSegments();
    store_.ClearGroups();
}

bool CacheCoordinator::GroupIsState(std::size_t i) const {
    return store_.GroupIsState(i);
}

bool CacheCoordinator::deviceCacheIsClearable() const {
    for (const CacheGroup& group : store_.Groups()) {
        if (group.Index().NumPinnedEntries(store_.DevicePool()) != 0) {
            return false;
        }
    }
    return true;
}

bool CacheCoordinator::CacheIsClearable(bool include_host) const {
    if (!deviceCacheIsClearable()) {
        return false;
    }
    if (!include_host) {
        return true;
    }
    return host_.IsClearable();
}

bool CacheCoordinator::ClearDeviceCache() {
    // Every plan drains what its admissions, restores and compactions recorded; a clear
    // runs between plans and drops what it evicts.
    _assert(!evictor_.HasSegments(), "a cache clear runs with no eviction left undrained");
    VictimSet cached{std::pmr::get_default_resource()};
    for (const CacheGroup& group : store_.Groups()) {
        const PrefixCacheIndex& index = group.Index();
        const std::vector<PrefixCacheIndex::EvictionCandidate> group_entries =
            index.EvictableCandidates(store_.DevicePool());
        if (static_cast<std::int32_t>(group_entries.size()) != index.NumEntries(store_.DevicePool())) {
            return false;
        }
        for (const PrefixCacheIndex::EvictionCandidate& entry : group_entries) {
            cached.members.push_back(VictimSet::Member{.group_id = group.Id(),
                                                       .location = entry.location,
                                                       .in_stranded_suffix = false,
                                                       .is_request_reclaim = false});
        }
    }

    host_.DropPendingStores();
    // The cascade of an earlier eviction may already have taken one.
    for (const CacheEvictor::UnevictedVictim& victim : evictor_.EvictPlanned(
             std::span{&cached, 1}, /*write_back=*/false,
             CacheEvictor::VictimReasons{.victim = &CacheEvictor::DeviceEvictionStats::flush,
                                         .stranded_suffix = &CacheEvictor::DeviceEvictionStats::flush,
                                         .request_reclaim = &CacheEvictor::DeviceEvictionStats::flush})) {
        FatalCheck(!store_.DevicePool().IsOccupied(victim.location), "clearable Device cache entry disappeared");
    }
    // A Device-only clear leaves Host L2 untouched: the Host pages under a
    // Device prefix it emptied are not collected for it.
    store_.Boundaries().ClearDeadCandidates();
    return true;
}

bool CacheCoordinator::ClearCache() {
    if (store_.HostPool() == nullptr) {
        return ClearDeviceCache();
    }

    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> host_locations;
    for (const CacheGroup& group : store_.Groups()) {
        const PrefixCacheIndex& index = group.Index();
        const std::vector<PrefixCacheIndex::EvictionCandidate> group_entries =
            index.EvictableCandidates(*store_.HostPool());
        if (static_cast<std::int32_t>(group_entries.size()) != index.NumEntries(*store_.HostPool())) {
            return false;
        }
        for (const PrefixCacheIndex::EvictionCandidate& entry : group_entries) {
            host_locations.emplace_back(group.Id(), entry.location);
        }
    }

    // ClearDeviceCache performs its complete pin check before mutation. Since
    // Host was checked above, a false return leaves both tiers unchanged.
    if (!ClearDeviceCache()) {
        return false;
    }
    for (const auto& [group_id, location] : host_locations) {
        evictor_.FlushHostEntry(group_id, location);
    }
    // With both tiers empty no node is left: no eviction store holds one, as
    // the scheduler refuses a clear while a store is in flight. What the Host
    // evictions queued for collection goes, and so do the tail hints.
    store_.Boundaries().ClearAfterCacheClear();
    store_.Checkpoints().Clear();
    return true;
}

CacheCoordinator::BoundaryEventStatus CacheCoordinator::GetBoundaryEventStatus(const CacheKey& boundary,
                                                                               std::int32_t tail_tokens) const {
    _assert(boundary.namespace_id == kDefaultCacheNamespaceId, "a tracked boundary lies in the default namespace");
    _assert(tail_tokens >= 0 && tail_tokens < store_.PrefixGranularity(), "a tail lies strictly inside a prefix page");
    const BoundaryTree::Node* node = store_.Boundaries().Find(boundary.content_hash);
    if (node == nullptr) return {.publishable = false, .has_device_entries = false};
    const std::int32_t device_entries = node->Entries(BoundaryTree::kDevice);
    // A continuous history group represents locality without requiring the
    // auxiliary sliding windows or sparse checkpoints at every ancestor.
    if (!store_.RoutingHistoryGroups().empty()) {
        // The index already maintains one closed-entry hold per cached key.
        // Count expected keys by group, not by probing every history block.
        std::int32_t expected = 0;
        for (const std::size_t group : store_.RoutingHistoryGroups()) {
            const auto i = static_cast<std::int32_t>(group);
            expected += tail_tokens > 0 ? store_.TailBlocksFor(i, tail_tokens) : store_.BlocksPerBoundary(i);
        }
        const std::int32_t cached = node->ClosedEntries(BoundaryTree::kDevice);
        _assert(cached <= expected, "a boundary's Device history entries lie among its keys");
        return {.publishable = cached == expected, .has_device_entries = device_entries > 0};
    }
    // No continuous history means no ordinary prefix-residency events.
    // Scheduler configuration rejects enabling this event stream in that case.
    return {.publishable = false, .has_device_entries = device_entries > 0};
}

bool CacheCoordinator::HoldsStateCheckpoint(const ContentHash& content_hash) const {
    // A probe takes each State group's checkpoint from whichever tier holds
    // it (PrefixProber::ProbeHostSuffix), so every State group must hold one on either.
    bool any_state_group = false;
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        if (!GroupIsState(i)) {
            continue;
        }
        const CacheKey key{.group_id = store_.Groups()[i].Id(), .content_hash = content_hash, .page_offset = 0};
        const PrefixCacheIndex& index = store_.Groups()[i].Index();
        if (!index.Contains(store_.DevicePool(), key) &&
            (store_.HostPool() == nullptr || !index.Contains(*store_.HostPool(), key))) {
            return false;
        }
        any_state_group = true;
    }
    return any_state_group;
}

template <CacheTier Tier>
CoordinatorMatch CacheCoordinator::acquireTierWithKeys(std::span<const std::vector<CacheKey>> group_keys,
                                                       std::int32_t floor_tokens, PrefixProbe::Tier&& probe,
                                                       std::uint64_t access_epoch) {
    const BlockPool& pool = store_.TierPool<Tier>();
    CoordinatorMatch out;
    out.num_common_tokens = probe.num_common_tokens;
    out.per_group.resize(store_.Groups().size());
    bool seen_state_group = false;
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        if (store_.GroupIsReplayable(static_cast<std::int32_t>(i))) {
            // Never matched, so it has no keys (PrefixProber::PreparePrefixKeys) and nothing
            // to acquire: its match stays empty.
            _assert(probe.per_group[i].hits.empty() && (!probe.tail || probe.tail->group_keys[i].empty()),
                    "a replayable group holds no prefix hit");
            continue;
        }
        // A State group's hits count from page 0 (PrefixProbe::Tier).
        const std::int32_t begin_block = GroupIsState(i) ? 0 : floor_tokens / store_.Geometry()[i].BlockGranularity();
        out.per_group[i] = store_.Groups()[i].Index().AcquireMatched(
            pool, group_keys[i], begin_block, probe.per_group[i], access_epoch, SessionForAccess(access_epoch));
        if constexpr (Tier == CacheTier::kDevice) {
            if (store_.HostPool() != nullptr) {
                store_.Groups()[i].Index().TouchExisting(
                    *store_.HostPool(), std::span{group_keys[i]}.subspan(begin_block), out.per_group[i].blocks,
                    access_epoch, SessionForAccess(access_epoch));
            }
        }
        if (probe.tail && !probe.tail->group_keys[i].empty()) {
            // The tail's blocks follow the full pages in the same table.
            PrefixMatch tail_match = store_.Groups()[i].Index().AcquireMatched(
                pool, probe.tail->group_keys[i],
                /*begin_block=*/0, probe.tail->per_group[i], access_epoch, SessionForAccess(access_epoch));
            if constexpr (Tier == CacheTier::kDevice) {
                if (store_.HostPool() != nullptr) {
                    store_.Groups()[i].Index().TouchExisting(*store_.HostPool(), probe.tail->group_keys[i],
                                                             tail_match.blocks, access_epoch,
                                                             SessionForAccess(access_epoch));
                }
            }
            out.per_group[i].blocks.insert(out.per_group[i].blocks.end(),
                                           std::make_move_iterator(tail_match.blocks.begin()),
                                           std::make_move_iterator(tail_match.blocks.end()));
        }
        if (GroupIsState(i)) {
            const bool first_state_group = !seen_state_group;
            seen_state_group = true;
            for (const CacheBlockRef& block : out.per_group[i].blocks) {
                if (const std::optional<CacheKey> key = store_.Groups()[i].Index().KeyOf(pool, block)) {
                    if (BoundaryTree::Node* node = store_.Boundaries().Find(key->content_hash)) {
                        if (first_state_group) {
                            out.resumed_checkpoint_last_use = node->latest_state_access_epoch;
                        }
                        node->latest_state_access_epoch = std::max(node->latest_state_access_epoch, access_epoch);
                    }
                }
            }
        }
    }
    return out;
}

CacheCoordinator::AcquiredPrefix CacheCoordinator::acquirePrefix(PrefixProbe&& probe, std::uint64_t access_epoch) {
    ProfileSpan profile_span{"CacheCoordinator::acquirePrefix"};
    AcquiredPrefix out;
    // The Host floor is the Device's full-page coverage: the Host's KV pages
    // start there (a Device KV tail means the Host holds none, only State
    // checkpoints at the tail).
    const std::int32_t device_full_tokens = probe.device.full_page_tokens;
    // A hit keeps its tail young under the per-prior cap.
    for (const PrefixProbe::Tier* tier : {&probe.device, &probe.host}) {
        if (tier->tail && tier->tail->tail_tokens > 0) {
            store_.Boundaries().TouchTail(tier->tail->hash);
        }
    }
    out.device = acquireTierWithKeys<CacheTier::kDevice>(*probe.group_keys, /*floor_tokens=*/0, std::move(probe.device),
                                                         access_epoch);
    // The storage tier is never acquired: an admission loads Host entries
    // only; asynchronous storage publishes Host entries before admission.
    if (store_.HostPool() != nullptr && !probe.host.per_group.empty()) {
        out.host = acquireTierWithKeys<CacheTier::kHost>(*probe.group_keys, device_full_tokens, std::move(probe.host),
                                                         access_epoch);
    }
    if (const std::uint64_t last_use = out.device.resumed_checkpoint_last_use != 0
                                           ? out.device.resumed_checkpoint_last_use
                                           : out.host.resumed_checkpoint_last_use;
        last_use != 0) {
        history_.RecordResumeGap(last_use);
    }
    if (probe.state_at_prompt_end) {
        out.state_at_prompt_end.resize(store_.Groups().size());
        for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
            if (probe.state_at_prompt_end->group_keys[i].empty()) {
                continue;
            }
            PrefixMatch match =
                store_.Groups()[i].Index().AcquireMatched(store_.DevicePool(), probe.state_at_prompt_end->group_keys[i],
                                                          /*begin_block=*/0, probe.state_at_prompt_end->per_group[i],
                                                          access_epoch, SessionForAccess(access_epoch));
            if (store_.HostPool() != nullptr) {
                store_.Groups()[i].Index().TouchExisting(*store_.HostPool(), probe.state_at_prompt_end->group_keys[i],
                                                         match.blocks, access_epoch, SessionForAccess(access_epoch));
            }
            out.state_at_prompt_end[i] = std::move(match.blocks.front());
            if (BoundaryTree::Node* node = store_.Boundaries().Find(probe.state_at_prompt_end->hash)) {
                node->latest_state_access_epoch = std::max(node->latest_state_access_epoch, access_epoch);
            }
        }
    }
    return out;
}

// A full-pool census, and deliberately so: whether a parent is reclaimable
// depends on whether a request still owns one of its children, and a reference
// being dropped is not an event the pool or the index can observe. This is an
// observability call, not part of admission or placement; those answer their
// capacity questions from the pool's own counters.
std::int32_t CacheCoordinator::NumAvailableLcmBlocks() const {
    std::int32_t available = 0;
    // Ids 1..NumLcmBlocks, counted so that INT32_MAX ends the loop.
    for (std::int32_t i = 0; i < store_.DevicePool().NumLcmBlocks(); ++i) {
        const std::int32_t parent_id = i + 1;
        const std::optional<std::uint32_t> group_id = store_.DevicePool().BoundGroup(parent_id);
        if (!group_id ||
            store_.Groups()[*group_id].Index().ParentIsFullyEvictable(
                store_.DevicePool(), parent_id, store_.Groups()[*group_id].Allocator().CacheBlocksPerLcmBlock())) {
            ++available;
        }
    }
    return available;
}

std::int32_t CacheCoordinator::CountActiveLcmBlocks(std::span<const std::span<const BlockTable>> request_tables) const {
    // Parent ids are dense in [1, NumLcmBlocks], so byte flags dedupe shared
    // prefixes without hashing every block reference of every live request.
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(store_.DevicePool().NumLcmBlocks()) + 1, 0);
    std::int32_t active = 0;
    for (std::span<const BlockTable> tables : request_tables) {
        _assert(tables.empty() || tables.size() == store_.Groups().size(), "tables/groups size mismatch");
        for (std::size_t i = 0; i < tables.size(); ++i) {
            const BlockTable& table = tables[i];
            const std::int32_t released = store_.Groups()[i].Allocator().LeadingReleasedBlocks(table);
            for (const CacheBlockRef& block_ref : table.Blocks().subspan(static_cast<std::size_t>(released))) {
                if (!block_ref) {
                    continue;
                }
                std::uint8_t& parent_seen = seen[static_cast<std::size_t>(block_ref->Location().lcm_block_id)];
                if (!parent_seen) {
                    parent_seen = 1;
                    ++active;
                }
            }
        }
    }
    return active;
}

std::int32_t CacheCoordinator::GroupAvailablePages(std::int32_t group_index) const {
    _assert(group_index >= 0 && static_cast<std::size_t>(group_index) < store_.Groups().size(),
            "cache group index out of range");
    const std::int32_t slots_per_parent =
        store_.Groups()[static_cast<std::size_t>(group_index)].Allocator().CacheBlocksPerLcmBlock();
    return store_.DevicePool().NumEmptyLcmBlocks() * slots_per_parent +
           store_.DevicePool().NumFreeSlots(static_cast<std::uint32_t>(group_index));
}

std::int32_t CacheCoordinator::NumNewlyReleasableLcmBlocks(std::span<const BlockTable> tables) const {
    _assert(tables.size() == store_.Groups().size(), "release estimate requires one table per cache group");

    struct ReleasedRefs {
        const CacheBlockRef* block_ref{};
        std::uint32_t count{0};
    };
    std::vector<std::unordered_map<CacheBlockLocation, ReleasedRefs, CacheBlockLocationHash>> released_by_group(
        store_.Groups().size());
    std::unordered_set<std::int32_t> referenced_parents;
    for (std::size_t group_id = 0; group_id < tables.size(); ++group_id) {
        for (const CacheBlockRef& block_ref : tables[group_id].Blocks()) {
            if (!block_ref) {
                continue;
            }
            const CacheBlockLocation location = block_ref->Location();
            ReleasedRefs& refs = released_by_group[group_id][location];
            refs.block_ref = &block_ref;
            ++refs.count;
            referenced_parents.insert(location.lcm_block_id);
        }
    }

    std::int32_t count = 0;
    for (std::int32_t parent_id : referenced_parents) {
        const std::optional<std::uint32_t> group_id = store_.DevicePool().BoundGroup(parent_id);
        _assert(group_id.has_value(), "request table references an unbound LCM block");
        const PrefixCacheIndex& index = store_.Groups()[*group_id].Index();
        bool parent_becomes_reclaimable = true;
        for (CacheBlockLocation location : store_.DevicePool().OccupiedLocations(parent_id)) {
            const auto released = released_by_group[*group_id].find(location);
            if (released == released_by_group[*group_id].end()) {
                parent_becomes_reclaimable = false;
                break;
            }
            const std::uint32_t owners = released->second.block_ref->use_count();
            _assert(owners >= released->second.count, "request-owned reference count exceeds total owners");
            const std::uint32_t retained_owners = owners - released->second.count;
            const std::uint32_t allowed_owners = index.Contains(store_.DevicePool(), location) ? 1U : 0U;
            if (retained_owners != allowed_owners) {
                parent_becomes_reclaimable = false;
                break;
            }
        }
        if (parent_becomes_reclaimable) {
            ++count;
        }
    }
    return count;
}

void CacheCoordinator::CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                                       std::uint64_t access_epoch, std::int32_t first_prefix_page,
                                       CacheBoundaryKind boundary_kind) {
    _assert(tables.size() == store_.Groups().size(), "tables/groups size mismatch");
    if (content_hashes.empty()) {
        return;  // hot decode rounds usually fill no page
    }
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        if (GroupIsState(i)) {
            continue;  // a state checkpoint is published by its natural point, never by slot
        }
        std::vector<CacheKey> keys = store_.KeysForGroup(content_hashes, store_.Groups()[i].Id());
        const std::int32_t blocks_per_boundary = store_.BlocksPerBoundary(static_cast<std::int32_t>(i));
        cacheFullBlocksForGroup<CacheTier::kDevice>(i, tables[i], keys, first_prefix_page * blocks_per_boundary,
                                                    access_epoch, boundary_kind, /*stream_completed_to_host=*/false);
    }
}

void CacheCoordinator::validateCopyInBlocks(const RequestProgress& progress,
                                            std::span<const std::vector<CacheBlockRef>> copy_in_blocks) const {
    const bool copies_in = progress.completed_pages && progress.completed_pages->tail &&
                           progress.completed_pages->owner_tail == OwnerTail::kCopiedIn;
    if (!copies_in) {
        _assert(copy_in_blocks.empty(), "only an owner copying in past its tail spends copy-in blocks");
        return;
    }
    _assert(copy_in_blocks.size() == store_.Groups().size(),
            "an owner copying in past its tail holds its copy-in blocks");
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        const CacheStore::BlockRange slots = store_.CopyInSlots(i, *progress.num_computed_tokens);
        _assert(copy_in_blocks[i].size() == static_cast<std::size_t>(slots.past_end - slots.first) &&
                    std::ranges::all_of(copy_in_blocks[i], [](const CacheBlockRef& block) { return bool(block); }),
                "the owner holds a copy-in block for every copy-in slot");
    }
}

void CacheCoordinator::CacheCompletedBlocks(std::span<BlockTable> tables, const RequestProgress& progress,
                                            std::uint64_t access_epoch) {
    _assert(tables.size() == store_.Groups().size(), "tables/groups size mismatch");
    _assert(progress.completed_pages.has_value(), "publication requires completed pages");
    _assert(progress.completed_pages->owner_tail == OwnerTail::kFinal,
            "a publication outside an admission publishes a final frontier");
    store_.ValidateProgress(progress);
    const CompletedPages& completed = *progress.completed_pages;
    BoundaryTree::Hold publication = store_.Boundaries().LinkPublication(completed);
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        cacheDeviceCompletedBlocksForGroup(i, tables[i], completed, access_epoch);
    }
    // A model without a prefix-closed group may register nothing under the
    // deepest boundary linked (a chunk end of State groups alone): it leaves
    // now.
    publication.reset();
    recordCheckpoints(completed, access_epoch);
    recordPublishedTail(completed);
}

template <CacheTier Tier>
void CacheCoordinator::cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table,
                                               std::span<const CacheKey> keys, std::int32_t first_cache_block,
                                               std::uint64_t access_epoch, CacheBoundaryKind boundary_kind,
                                               bool stream_completed_to_host) {
    if (store_.GroupIsReplayable(static_cast<std::int32_t>(group_index))) {
        return;  // regenerated by the model, never published
    }
    // The window re-resolves its page ids once the index replaced deduped
    // blocks in it.
    BlockTable::PublishWindow window =
        store_.Groups()[group_index].Allocator().BlocksToPublish(table, first_cache_block, keys.size());
    cacheBlocksForGroup<Tier>(group_index, window.Blocks(), keys, first_cache_block, access_epoch, boundary_kind,
                              stream_completed_to_host);
}

template <CacheTier Tier>
void CacheCoordinator::cacheBlocksForGroup(std::size_t group_index, std::span<CacheBlockRef> blocks,
                                           std::span<const CacheKey> keys, std::int32_t first_logical_block,
                                           std::uint64_t access_epoch, CacheBoundaryKind boundary_kind,
                                           bool stream_completed_to_host) {
    std::vector<std::pair<CacheKey, CacheBlockRef>> newly_cached;
    const SessionCacheOwner session = SessionForAccess(access_epoch);
    const bool automatically_streams_to_host =
        stream_device_cache_to_host_ &&
        (store_.Groups()[group_index].Spec().attn_kind == AttnKind::kSlidingWindow || stream_completed_to_host);
    // Each entry it registers holds the tree node of its hash, which a
    // publication linked before registering.
    store_.Groups()[group_index].Index().RegisterFullBlocks(
        store_.TierPool<Tier>(), blocks, keys, access_epoch, first_logical_block, boundary_kind,
        Tier == CacheTier::kDevice ? &newly_cached : nullptr, session);
    if constexpr (Tier == CacheTier::kHost) {
        return;
    }
    if (store_.HostPool() != nullptr) {
        store_.Groups()[group_index].Index().TouchExisting(*store_.HostPool(), keys, blocks, access_epoch, session);
    }
    for (auto& [key, block_ref] : newly_cached) {
        if (store_.MutationSink()) {
            store_.MutationSink()(key, CacheMutation::kStored);
        }
        if (!automatically_streams_to_host || (session && session->ended)) {
            continue;
        }
        host_.QueueStore(std::move(key));
    }
    if (GroupIsState(group_index)) {
        for (std::size_t j = 0; j < keys.size(); ++j) {
            if (blocks[j]) {
                BoundaryTree::Node* node = store_.Boundaries().Find(keys[j].content_hash);
                _assert(node != nullptr, "a published State checkpoint has a boundary");
                node->latest_state_access_epoch = std::max(node->latest_state_access_epoch, access_epoch);
            }
        }
    }
}

CacheBlockRef CacheCoordinator::AcquireDeviceCachedBlock(const CacheKey& key) const {
    if (key.group_id >= store_.Groups().size()) {
        return {};
    }
    return store_.Groups()[key.group_id].Index().Find(store_.DevicePool(), key);
}

void CacheCoordinator::CollectDeadEntries() {
    ProfileSpan profile_span{"CacheCoordinator::CollectDeadEntries"};
    // A node's entries, the Device's before the Host's.
    const auto entries_of = [&](const BoundaryTree::Node& node) {
        std::vector<CacheStore::NodeEntry> entries = store_.NodeEntries(store_.DevicePool(), node);
        if (store_.HostPool() != nullptr) {
            std::ranges::copy(store_.NodeEntries(*store_.HostPool(), node), std::back_inserter(entries));
        }
        return entries;
    };
    // Judged on the state at collection, not at queueing. A node no
    // prefix-closed group holds on either tier is a gap, whatever State
    // entries or children it keeps. Nothing is dead while a store is
    // bringing it to the Host: its settlement queues it again.
    const auto dead = [&](const BoundaryTree::Node& node) {
        return node.PendingHostStores() == 0 &&
               ((store_.NumStateGroups() > 0 && !node.HasCheckpointBoundaryInSubtree()) ||
                (node.tail_tokens > 0 && !store_.Boundaries().Tails().Contains(node.hash)) ||
                (store_.HasPrefixClosedGroups() &&
                 std::ranges::none_of(entries_of(node), [&](const CacheStore::NodeEntry& entry) {
                     return store_.Groups()[entry.group_id].Matcher().IsPrefixClosed();
                 })));
    };
    // The roots left with an entry in use, each once, for the next call.
    std::vector<ContentHash> waiting;
    std::unordered_set<ContentHash> waiting_roots;
    // The evictions below may queue more; they are collected in this call too.
    while (const std::optional<ContentHash> candidate = store_.Boundaries().PopDeadCandidate()) {
        const BoundaryTree::Node* top = store_.Boundaries().Find(*candidate);
        if (top == nullptr || !dead(*top)) {
            continue;
        }
        // A live request's table holds its whole prefix, so the climb stops
        // below the pages it reads. It keeps the entries of each node it
        // passes.
        std::vector<std::pair<const BoundaryTree::Node*, std::vector<CacheStore::NodeEntry>>> climbed;
        while (top->parent != nullptr && dead(*top->parent)) {
            std::vector<CacheStore::NodeEntry> parent_entries = entries_of(*top->parent);
            if (std::ranges::any_of(parent_entries, &CacheStore::NodeEntry::in_use)) {
                break;
            }
            top = top->parent;
            climbed.emplace_back(top, std::move(parent_entries));
        }
        if (waiting_roots.contains(top->hash)) {
            continue;
        }
        // Post-order over the subtree, a node after the nodes under it, so
        // each node leaves with its last entry; it completes the climbed
        // nodes in climb order.
        std::vector<CacheStore::NodeEntry> entries_to_evict;
        bool in_use = false;
        struct Frame {
            const BoundaryTree::Node* node;
            const BoundaryTree::Node* next_child;
        };
        std::vector<Frame> stack{Frame{.node = top, .next_child = top->first_child}};
        std::size_t next_climbed = 0;
        while (!stack.empty()) {
            Frame& frame = stack.back();
            if (const BoundaryTree::Node* child = frame.next_child) {
                frame.next_child = child->next_sibling;
                stack.push_back(Frame{.node = child, .next_child = child->first_child});
                continue;
            }
            const std::vector<CacheStore::NodeEntry> node_entries =
                next_climbed < climbed.size() && climbed[next_climbed].first == frame.node
                    ? std::move(climbed[next_climbed++].second)
                    : entries_of(*frame.node);
            for (const CacheStore::NodeEntry& entry : node_entries) {
                in_use = in_use || entry.in_use;
                if (!entry.in_use) {
                    entries_to_evict.push_back(entry);
                }
            }
            // A store still on its way lands here at its ACK, published when
            // its checkpoint boundary is held: the root waits for it like for a
            // pinned entry.
            in_use = in_use || frame.node->PendingHostStores() > 0;
            stack.pop_back();
        }
        // The evictions below may take the node away.
        ContentHash root = top->hash;
        for (const CacheStore::NodeEntry& entry : entries_to_evict) {
            if (entry.pool == &store_.DevicePool()) {
                FatalCheck(evictor_
                               .EvictCachedBlockAlone(entry.group_id, entry.location,
                                                      &CacheEvictor::DeviceEvictionStats::dead_entry_collection,
                                                      /*write_back_segment=*/nullptr)
                               .has_value(),
                           "a dead Device entry became pinned");
            } else if (store_.Groups()[entry.group_id].Index().Contains(*store_.HostPool(), entry.location)) {
                // A checkpoint takes the other State groups' at its hash along.
                FatalCheck(evictor_.EvictHostCachedBlock(entry.group_id, entry.location,
                                                         &CacheEvictor::HostEvictionStats::dead_entry),
                           "a dead Host entry became pinned");
            }
        }
        if (in_use) {
            waiting_roots.insert(root);
            waiting.push_back(std::move(root));
        }
    }
    for (const ContentHash& root : waiting) {
        store_.Boundaries().QueueDeadCandidate(root);
    }
}

void CacheCoordinator::MarkSupersededStateInput(const ContentHash& resumed_tail_hash, std::uint64_t access_epoch) {
    const std::optional<std::int32_t> tail_tokens = store_.Boundaries().Tails().TailTokensOf(resumed_tail_hash);
    if (!tail_tokens) {
        return;
    }
    // A later request's hit still needs them: the whole tail stays unmarked,
    // so its checkpoints and blocks keep leaving together. The newest touch
    // decides, not the eviction epoch: an older request's publication of the
    // same tail since then dated them back, but the later hit stands.
    bool touched_since = false;
    store_.ForEachBoundaryKey(
        CacheKey{.content_hash = resumed_tail_hash}, *tail_tokens, [&](std::size_t i, const CacheKey& key) {
            if (CacheBlockRef block = store_.Groups()[i].Index().Find(store_.DevicePool(), key)) {
                touched_since |=
                    store_.Groups()[i].Index().MetadataFor(store_.DevicePool(), block->Location())->max_access_epoch >
                    access_epoch;
            }
        });
    if (touched_since) {
        return;
    }
    // Registering touches an existing entry with the epoch given (and keeps
    // its position and boundary kind): the tail's blocks and checkpoints
    // leave the eviction order's newest end for its oldest.
    store_.ForEachBoundaryKey(
        CacheKey{.content_hash = resumed_tail_hash}, *tail_tokens, [&](std::size_t i, const CacheKey& key) {
            if (CacheBlockRef block = store_.Groups()[i].Index().Find(store_.DevicePool(), key)) {
                store_.Groups()[i].Index().Register(store_.DevicePool(), block, key, CacheStore::kSupersededAccessEpoch,
                                                    /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                                                    /*newly_cached=*/nullptr, SessionForAccess(access_epoch));
            }
        });
}

static_assert(CacheStore::kDemotedAccessEpoch == 0, "a demoted entry carries an epoch no access issues");

std::vector<CacheKey> CacheCoordinator::BoundaryCheckpointKeys(std::span<const std::string> prefix_hashes,
                                                               std::span<const std::int32_t> kept_boundaries) const {
    std::vector<CacheKey> keys;
    if (prefix_hashes.empty()) {
        return keys;
    }
    // What a natural checkpoint at the boundary publishes in the groups that
    // are not prefix-closed; the KV pages below it stay the path to the
    // deeper boundaries.
    const CompletedPages boundary{
        .prefix_hashes = prefix_hashes,
        .first_new_prefix_page = static_cast<std::int32_t>(prefix_hashes.size()) - 1,
        .publish_state = true,
    };
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        if (store_.Groups()[i].Matcher().IsPrefixClosed()) {
            continue;
        }
        const bool sliding = store_.Groups()[i].Spec().attn_kind == AttnKind::kSlidingWindow;
        for (CacheStore::PublishedRange& range : store_.CompletedBlockRanges(i, boundary)) {
            for (std::size_t j = 0; j < range.keys.size(); ++j) {
                // A window block is keyed by its place on the chain, so a kept
                // boundary whose window covers it resumes with this very entry.
                const std::int32_t block = range.first_cache_block + static_cast<std::int32_t>(j);
                const auto behind = [&](std::int32_t kept_tokens) {
                    const CacheStore::BlockRange window =
                        store_.SlidingWindowBlocks(static_cast<std::int32_t>(i), kept_tokens);
                    return (block >= window.first && block < window.past_end) ||
                           block < store_.SinkBlocksBelow(static_cast<std::int32_t>(i), window).past_end;
                };
                if (sliding && std::ranges::any_of(kept_boundaries, behind)) {
                    continue;
                }
                keys.push_back(std::move(range.keys[j]));
            }
        }
    }
    return keys;
}

std::int32_t CacheCoordinator::DemoteEntries(std::span<const CacheKey> keys) {
    std::int32_t demoted = 0;
    for (const CacheKey& key : keys) {
        PrefixCacheIndex& index = store_.Groups()[key.group_id].Index();
        demoted +=
            index.Demote(store_.DevicePool(), key, CacheStore::kDemotedAccessEpoch, /*write_back_holds=*/0) ? 1 : 0;
        if (store_.HostPool() != nullptr) {
            demoted +=
                index.Demote(*store_.HostPool(), key, CacheStore::kDemotedAccessEpoch, /*write_back_holds=*/0) ? 1 : 0;
        }
    }
    return demoted;
}

std::uint64_t CacheCoordinator::LatestStateAccessEpoch(const ContentHash& hash) const {
    const BoundaryTree::Node* node = store_.Boundaries().Find(hash);
    return node == nullptr ? 0 : node->latest_state_access_epoch;
}

void CacheCoordinator::recordCheckpoints(const CompletedPages& completed, std::uint64_t access_epoch) {
    if (store_.NumStateGroups() == 0 || completed.tokens.empty()) {
        return;
    }
    // The in-forward checkpoint first: it lies on the path to the point's.
    if (completed.state_checkpoint_pages > 0) {
        const auto pages = static_cast<std::size_t>(completed.state_checkpoint_pages);
        if (store_.Checkpoints().Observe(
                completed.tokens.first(pages * static_cast<std::size_t>(store_.PrefixGranularity())),
                completed.prefix_hashes.first(pages), std::nullopt, access_epoch)) {
            syncCheckpoint(ContentHash{completed.prefix_hashes[pages - 1]});
        }
    }
    if (completed.publish_state &&
        store_.Checkpoints().Observe(completed.tokens, completed.prefix_hashes, completed.tail, access_epoch)) {
        syncCheckpoint(completed.tail ? completed.tail->hash : ContentHash{completed.prefix_hashes.back()});
    }
}

void CacheCoordinator::syncCheckpoint(const ContentHash& hash) {
    if (store_.Checkpoints().Size() == 0 || !store_.Checkpoints().Contains(hash)) {
        return;
    }
    // A checkpoint nothing holds any more leaves at once.
    const BoundaryTree::Node* node = store_.Boundaries().Find(hash);
    store_.Checkpoints().Update(
        hash, node == nullptr ? CheckpointTree::Residency{} : store_.Boundaries().CheckpointResidency(*node));
}

void CacheCoordinator::recordPublishedTail(const CompletedPages& completed) {
    if (!completed.tail) {
        return;
    }
    const ContentHash prior =
        completed.prefix_hashes.empty() ? ContentHash{} : ContentHash{completed.prefix_hashes.back()};
    evictor_.RecordTail(prior, *completed.tail);
}

template <CacheTier Tier>
void CacheCoordinator::cacheCompletedBlocksForGroup(std::size_t group_index, BlockTable& table,
                                                    const CompletedPages& completed, std::uint64_t access_epoch) {
    // An owner that keeps writing past a tail registers its own tail blocks
    // but never takes another request's canonical block into its table: the
    // copy-in moves it off each block it registered, into the block reserved
    // in that block's bucket, and it keeps writing a block whose key another
    // request published first. Every other published block dedupes onto the
    // canonical one, which the table then holds.
    const CacheStore::BlockRange own_tail =
        completed.tail && completed.owner_tail == OwnerTail::kCopiedIn
            ? store_.CopyInSlots(
                  group_index, static_cast<std::int32_t>(completed.prefix_hashes.size()) * store_.PrefixGranularity() +
                                   completed.tail->tail_tokens)
            : CacheStore::BlockRange{};
    for (const CacheStore::PublishedRange& range : store_.CompletedBlockRanges(group_index, completed)) {
        const std::int32_t past_end = range.first_cache_block + static_cast<std::int32_t>(range.keys.size());
        const std::int32_t own_first = own_tail.first < own_tail.past_end
                                           ? std::clamp(own_tail.first, range.first_cache_block, past_end)
                                           : past_end;
        const std::span<const CacheKey> keys{range.keys};
        const auto shared = static_cast<std::size_t>(own_first - range.first_cache_block);
        if (shared > 0) {
            cacheFullBlocksForGroup<Tier>(group_index, table, keys.first(shared), range.first_cache_block, access_epoch,
                                          range.boundary_kind, completed.stream_completed_to_host);
        }
        if (own_first < past_end) {
            _assert(own_first == own_tail.first && past_end == own_tail.past_end,
                    "the owner's tail blocks are registered whole");
            // Registration may replace these references, never the table's.
            std::vector<CacheBlockRef> own(table.Blocks().begin() + own_first, table.Blocks().begin() + past_end);
            cacheBlocksForGroup<Tier>(group_index, own, keys.subspan(shared), own_first, access_epoch,
                                      range.boundary_kind, completed.stream_completed_to_host);
        }
    }
    // An in-forward State checkpoint sits outside the table: its block is
    // published like a natural checkpoint's slot 1, under its page's hash.
    if constexpr (Tier == CacheTier::kDevice) {
        if (GroupIsState(group_index) && completed.state_checkpoint_pages > 0 &&
            group_index < completed.state_checkpoint_blocks.size() && completed.state_checkpoint_blocks[group_index]) {
            _assert(completed.state_checkpoint_pages <= static_cast<std::int32_t>(completed.prefix_hashes.size()),
                    "an in-forward State checkpoint lies inside the hashed prefix");
            const CacheKey key{
                .group_id = store_.Groups()[group_index].Id(),
                .content_hash =
                    ContentHash{
                        completed.prefix_hashes[static_cast<std::size_t>(completed.state_checkpoint_pages - 1)]},
                .page_offset = 0};
            cacheBlocksForGroup<Tier>(group_index, completed.state_checkpoint_blocks.subspan(group_index, 1),
                                      std::span<const CacheKey>(&key, 1), /*first_logical_block=*/1, access_epoch,
                                      CacheBoundaryKind::kEndpoint, completed.stream_completed_to_host);
        }
    }
}

void CacheCoordinator::cacheDeviceCompletedBlocksForGroup(std::size_t group_index, BlockTable& table,
                                                          const CompletedPages& completed, std::uint64_t access_epoch) {
    if (completed.preserve_source_blocks) {
        // Register may replace duplicate refs. Only the publication view may
        // change: the request pins the actual PD sources until its terminal ACK.
        BlockTable publication_table = table;
        std::vector<CacheBlockRef> checkpoints(completed.state_checkpoint_blocks.begin(),
                                               completed.state_checkpoint_blocks.end());
        CompletedPages publication = completed;
        publication.state_checkpoint_blocks = checkpoints;
        cacheCompletedBlocksForGroup<CacheTier::kDevice>(group_index, publication_table, publication, access_epoch);
    } else {
        cacheCompletedBlocksForGroup<CacheTier::kDevice>(group_index, table, completed, access_epoch);
    }
}

std::int32_t CacheCoordinator::FirstPublishedPage(const CompletedPages& completed) const {
    const auto hashed_prefix_pages = static_cast<std::int32_t>(completed.prefix_hashes.size());
    std::int32_t first_page = completed.first_new_prefix_page;
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        for (const CacheStore::PublishedRange& range : store_.CompletedBlockRanges(i, completed)) {
            if (range.sink) {
                continue;  // PublishedSinkPages names these pages
            }
            // A State group's one key is the point itself: the deepest page
            // unless a tail is published. A history group's run starts at
            // slot first_cache_block.
            const std::int32_t page =
                GroupIsState(i) ? hashed_prefix_pages - (completed.tail ? 0 : 1)
                                : range.first_cache_block / store_.BlocksPerBoundary(static_cast<std::int32_t>(i));
            first_page = std::min(first_page, page);
        }
    }
    // The block an in-forward State checkpoint publishes under its page adds
    // no page: the chunk that wrote it started on a page boundary at least
    // one page below it (writesInForwardCheckpoint), so that page is never
    // below the first new one. A sliding window behind it is one of the
    // ranges read above.
    return first_page;
}

std::int32_t CacheCoordinator::PublishedSinkPages(const CompletedPages& completed) const {
    std::int32_t pages = 0;
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        if (store_.Groups()[i].Allocator().KeptHeadBlocks() == 0) {
            continue;
        }
        const std::int32_t blocks_per_page = store_.BlocksPerBoundary(static_cast<std::int32_t>(i));
        for (const CacheStore::PublishedRange& range : store_.CompletedBlockRanges(i, completed)) {
            if (range.sink) {
                const std::int32_t past_end = range.first_cache_block + static_cast<std::int32_t>(range.keys.size());
                pages = std::max(pages, (past_end + blocks_per_page - 1) / blocks_per_page);
            }
        }
    }
    return pages;
}

void CacheCoordinator::ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens) {
    _assert(tables.size() == store_.Groups().size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        store_.Groups()[i].Allocator().ReclaimExpired(
            store_.DevicePool(), tables[i], groupExpiredBlockEndAt(static_cast<std::int32_t>(i), num_computed_tokens));
    }
}

std::vector<CacheBlockRef> CacheCoordinator::RetainInputCheckpoint(std::span<const std::string> prefix_hashes,
                                                                   std::int32_t boundary_tokens) const {
    _assert(boundary_tokens > 0 && boundary_tokens % store_.PrefixGranularity() == 0,
            "an input checkpoint must lie on a prefix boundary");
    const std::int32_t boundary_pages = boundary_tokens / store_.PrefixGranularity();
    _assert(boundary_pages <= static_cast<std::int32_t>(prefix_hashes.size()),
            "a retained checkpoint must have a published hash");
    std::vector<CacheBlockRef> blocks;
    const auto retain = [&](const CacheKey& key) {
        const PrefixCacheIndex& index = store_.Groups()[key.group_id].Index();
        CacheBlockRef block = index.Find(store_.DevicePool(), key);
        if (!block && store_.HostPool() != nullptr) {
            block = index.Find(*store_.HostPool(), key);
        }
        if (!block) {
            return false;
        }
        blocks.push_back(std::move(block));
        return true;
    };
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        if (store_.GroupIsReplayable(static_cast<std::int32_t>(i)) || store_.Groups()[i].Matcher().IsPrefixClosed()) {
            continue;
        }
        if (GroupIsState(i)) {
            if (!retain(CacheKey{
                    .group_id = store_.Groups()[i].Id(),
                    .content_hash = ContentHash{prefix_hashes[static_cast<std::size_t>(boundary_pages - 1)]}})) {
                return {};
            }
            continue;
        }
        const CacheStore::BlockRange window = store_.SlidingWindowBlocks(static_cast<std::int32_t>(i), boundary_tokens);
        const std::int32_t blocks_per_boundary = store_.BlocksPerBoundary(static_cast<std::int32_t>(i));
        for (const CacheStore::BlockRange& needed :
             {store_.SinkBlocksBelow(static_cast<std::int32_t>(i), window), window}) {
            for (std::int32_t slot = needed.first; slot < needed.past_end; ++slot) {
                if (!retain(
                        CacheKey{.group_id = store_.Groups()[i].Id(),
                                 .content_hash =
                                     ContentHash{prefix_hashes[static_cast<std::size_t>(slot / blocks_per_boundary)]},
                                 .page_offset = slot % blocks_per_boundary})) {
                    return {};
                }
            }
        }
    }
    return blocks;
}

void CacheCoordinator::RefreshRetainedCheckpoint(std::span<CacheBlockRef> blocks, std::uint64_t request_access_epoch) {
    if (blocks.empty()) {
        return;
    }
    const SessionCacheOwner session = SessionForAccess(request_access_epoch);
    const std::uint64_t access_epoch = history_.IssueAccessEpoch();
    for (CacheBlockRef& block : blocks) {
        _assert(
            block.IsOwnedBy(store_.DevicePool()) || store_.HostPool() != nullptr && block.IsOwnedBy(*store_.HostPool()),
            "a retained checkpoint belongs to a cache tier");
        const BlockPool& pool = block.IsOwnedBy(store_.DevicePool()) ? store_.DevicePool() : *store_.HostPool();
        const auto group_id = pool.BoundGroup(block->Location().lcm_block_id);
        _assert(group_id.has_value(), "an occupied cache block belongs to a group");
        PrefixCacheIndex& index = store_.Groups()[*group_id].Index();
        const auto key = index.KeyOf(pool, block);
        if (!key) {
            continue;
        }
        const auto metadata = index.MetadataFor(pool, block->Location());
        _assert(metadata.has_value(), "a cached block has eviction metadata");
        index.Register(pool, block, *key, access_epoch, metadata->eviction.logical_block_index,
                       metadata->eviction.boundary_kind,
                       /*newly_cached=*/nullptr, session);
        if (GroupIsState(key->group_id)) {
            BoundaryTree::Node* node = store_.Boundaries().Find(key->content_hash);
            _assert(node != nullptr, "a cached checkpoint has a boundary-tree node");
            node->latest_state_access_epoch = std::max(node->latest_state_access_epoch, access_epoch);
        }
    }
}

void CacheCoordinator::ConsumeReservedTokens(std::span<BlockTable> tables, std::int32_t num_tokens) {
    _assert(tables.size() == store_.Groups().size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        if (GroupIsState(i)) {
            continue;  // a decode step writes the live slot in place
        }
        store_.Groups()[i].Allocator().ConsumeReservedTokens(tables[i], num_tokens);
    }
}

void CacheCoordinator::Free(std::span<BlockTable> tables) {
    _assert(tables.size() == store_.Groups().size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        store_.Groups()[i].Allocator().Free(tables[i]);
    }
}

namespace {

// Slots covering [0, num_tokens) of a table and the capacity left inside the
// last of them. The computed tokens always lie inside the table: the
// admission that scheduled them acquired their blocks first. A State table's
// two slots are both data.
CacheCoordinator::DataSpan dataSpan(const BlockTable& table, const GroupGeometry& geometry, std::int32_t num_tokens) {
    if (!geometry.HasTokenSpan()) {
        return CacheCoordinator::DataSpan{.num_blocks = table.NumBlocks(), .available_tokens = 0};
    }
    const std::int32_t block_granularity = geometry.BlockGranularity();
    const std::int32_t covering = CeilDiv(num_tokens, block_granularity);
    _assert(covering <= table.NumBlocks(), "snapshot token count exceeds the table's logical fill");
    // Less than one block, though the covering blocks' capacity can pass INT32_MAX.
    return CacheCoordinator::DataSpan{
        .num_blocks = covering,
        .available_tokens = static_cast<std::int32_t>(std::int64_t{covering} * block_granularity - num_tokens)};
}

}  // namespace

std::vector<std::vector<ImageSlot>> CacheCoordinator::PublishedDataSlots(std::span<const BlockTable> tables,
                                                                         std::int32_t num_tokens) const {
    _assert(tables.size() == store_.Groups().size(), "tables/groups size mismatch");
    _assert(num_tokens >= 0, "snapshot token count must be non-negative");
    std::vector<std::vector<ImageSlot>> published(store_.Groups().size());
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        const DataSpan span = dataSpan(tables[i], store_.Geometry()[i], num_tokens);
        const std::span<const CacheBlockRef> blocks = tables[i].Blocks();
        for (std::int32_t slot = 0; slot < span.num_blocks; ++slot) {
            const CacheBlockRef& block = blocks[static_cast<std::size_t>(slot)];
            if (std::optional<CacheKey> key = store_.Groups()[i].Index().KeyOf(store_.DevicePool(), block)) {
                const std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata =
                    store_.Groups()[i].Index().MetadataFor(store_.DevicePool(), block->Location());
                _assert(metadata.has_value(), "a published slot has an index entry");
                published[i].push_back(ImageSlot{.slot_index = slot,
                                                 .block = block,
                                                 .key = std::move(*key),
                                                 .logical_block_index = metadata->eviction.logical_block_index,
                                                 .boundary_kind = metadata->eviction.boundary_kind});
            }
        }
    }
    return published;
}

std::optional<CacheCoordinator::SnapshotPoolPlan> CacheCoordinator::planSnapshotPool(
    std::span<const BlockTable> tables, std::int32_t num_tokens,
    std::span<const std::vector<ImageSlot>> host_served_slots) const {
    _assert(tables.size() == store_.Groups().size(), "tables/groups size mismatch");
    _assert(host_served_slots.size() == store_.Groups().size(), "host slots/groups size mismatch");
    _assert(num_tokens >= 0, "snapshot token count must be non-negative");
    if (store_.SnapshotPool() == nullptr) {
        return std::nullopt;
    }
    // Every group's bucket demand against the pool's empty parents, with no
    // acquisition: a retraction that does not fit must take nothing, since
    // releasing a partial acquisition would reorder the pool's FIFO.
    SnapshotPoolPlan plan{
        .spans = std::vector<DataSpan>(store_.Groups().size()),
        .private_slots = std::vector<std::vector<std::int32_t>>(store_.Groups().size()),
        .private_buckets = std::vector<std::vector<std::int32_t>>(store_.Groups().size()),
    };
    std::int64_t parents_needed = 0;
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        const GroupAllocator& allocator = store_.Groups()[i].Allocator();
        plan.spans[i] = dataSpan(tables[i], store_.Geometry()[i], num_tokens);
        std::vector<std::uint8_t> served(static_cast<std::size_t>(plan.spans[i].num_blocks), 0);
        for (const ImageSlot& slot : host_served_slots[i]) {
            _assert(slot.InHostCache(), "a Host-served snapshot slot carries the key it is published under");
            _assert(slot.slot_index >= 0 && slot.slot_index < plan.spans[i].num_blocks,
                    "a Host-served snapshot slot must be a data slot");
            served[static_cast<std::size_t>(slot.slot_index)] = 1;
        }
        std::vector<std::int64_t> blocks_by_bucket(static_cast<std::size_t>(allocator.ShardCount()), 0);
        const std::span<const CacheBlockRef> blocks = tables[i].Blocks();
        for (std::int32_t slot = 0; slot < plan.spans[i].num_blocks; ++slot) {
            const CacheBlockRef& block = blocks[static_cast<std::size_t>(slot)];
            if (!block || served[static_cast<std::size_t>(slot)]) {
                continue;
            }
            const std::int32_t bucket = allocator.BucketOf(block->Location());
            plan.private_slots[i].push_back(slot);
            plan.private_buckets[i].push_back(bucket);
            ++blocks_by_bucket[static_cast<std::size_t>(bucket)];
        }
        parents_needed +=
            ParentsNeededForBuckets(blocks_by_bucket, store_.SnapshotPool()->FreeSlotsByBucket(store_.Groups()[i].Id()),
                                    /*any_bucket_blocks=*/0, store_.SnapshotPool()->Packing(store_.Groups()[i].Id()));
    }
    if (parents_needed > store_.SnapshotPool()->NumEmptyLcmBlocks()) {
        return std::nullopt;
    }
    return plan;
}

bool CacheCoordinator::SnapshotPoolHolds(std::span<const BlockTable> tables, std::int32_t num_tokens,
                                         std::span<const std::vector<ImageSlot>> host_served_slots) const {
    return planSnapshotPool(tables, num_tokens, host_served_slots).has_value();
}

std::vector<std::vector<ImageSlot>> CacheCoordinator::PublishedDataSlotsAfter(std::span<const BlockTable> tables,
                                                                              std::int32_t num_tokens,
                                                                              const RequestProgress& progress) const {
    std::vector<std::vector<ImageSlot>> published = PublishedDataSlots(tables, num_tokens);
    if (!progress.completed_pages) {
        return published;
    }
    const CompletedPages& completed = *progress.completed_pages;
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        const std::vector<CacheStore::PublishedRange> ranges = store_.CompletedBlockRanges(i, completed);
        if (ranges.empty()) {
            continue;
        }
        const DataSpan span = dataSpan(tables[i], store_.Geometry()[i], num_tokens);
        const std::span<const CacheBlockRef> blocks = tables[i].Blocks();
        std::vector<ImageSlot>& slots = published[i];
        for (const CacheStore::PublishedRange& range : ranges) {
            for (std::size_t j = 0; j < range.keys.size(); ++j) {
                const std::int32_t slot = range.first_cache_block + static_cast<std::int32_t>(j);
                const auto index = static_cast<std::size_t>(slot);
                // A slot beyond the data span, a null hole, or one already
                // published is not added: the publication would register
                // nothing new there.
                if (slot >= span.num_blocks || !blocks[index] ||
                    std::ranges::any_of(slots, [slot](const ImageSlot& s) { return s.slot_index == slot; })) {
                    continue;
                }
                // Register re-points a slot whose key the index holds already
                // to that canonical block.
                CacheBlockRef canonical = store_.Groups()[i].Index().Find(store_.DevicePool(), range.keys[j]);
                slots.push_back(ImageSlot{.slot_index = slot,
                                          .block = canonical ? std::move(canonical) : blocks[index],
                                          .key = range.keys[j],
                                          .logical_block_index = slot,
                                          .boundary_kind = range.boundary_kind});
            }
        }
        std::ranges::sort(slots, {}, &ImageSlot::slot_index);
    }
    return published;
}

std::optional<CacheCoordinator::ImageTaken> CacheCoordinator::TakeImage(
    std::span<const BlockTable> tables, std::int32_t num_tokens,
    std::span<const std::vector<ImageSlot>> host_cached_slots) {
    for (std::size_t i = 0; i < host_cached_slots.size(); ++i) {
        for (const ImageSlot& slot : host_cached_slots[i]) {
            _assert(slot.block && store_.HostPool() != nullptr && slot.block.IsOwnedBy(*store_.HostPool()),
                    "a Host-cached snapshot slot must carry its Host L2 block");
        }
    }
    std::optional<SnapshotPoolPlan> plan = planSnapshotPool(tables, num_tokens, host_cached_slots);
    if (!plan) {
        return std::nullopt;
    }

    ImageTaken image;
    image.image.tables.reserve(store_.Groups().size());
    for (std::size_t i = 0; i < store_.Groups().size(); ++i) {
        ImageTable shape{
            .num_blocks = plan->spans[i].num_blocks,
            .reclaimed_prefix_blocks = std::min(tables[i].ReclaimedPrefixBlocks(), plan->spans[i].num_blocks),
            .available_tokens = plan->spans[i].available_tokens,
            .slots = host_cached_slots[i],
        };
        std::vector<CacheBlockRef> snapshot_blocks =
            store_.SnapshotPool()->AcquireBlocksInBuckets(store_.Groups()[i].Id(), plan->private_buckets[i]);
        FatalCheck(snapshot_blocks.size() == plan->private_buckets[i].size(),
                   "snapshot pool no longer fits a probed image");
        for (std::size_t j = 0; j < snapshot_blocks.size(); ++j) {
            const CacheBlockRef& source = tables[i].Blocks()[static_cast<std::size_t>(plan->private_slots[i][j])];
            image.store_pairs.push_back(BlockTransfer{
                .group_id = store_.Groups()[i].Id(),
                .source = source,
                .destination = snapshot_blocks[j],
            });
            shape.slots.push_back(
                ImageSlot{.slot_index = plan->private_slots[i][j], .block = std::move(snapshot_blocks[j])});
        }
        std::ranges::sort(shape.slots, {}, &ImageSlot::slot_index);
        image.image.tables.push_back(std::move(shape));
    }
    return image;
}

void CacheCoordinator::CacheDeviceBlock(CacheBlockRef& block_ref, const CacheKey& key, std::int32_t logical_block_index,
                                        CacheBoundaryKind boundary_kind, std::uint64_t access_epoch) {
    _assert(static_cast<bool>(block_ref), "CacheDeviceBlock requires a destination block");
    _assert(key.group_id < store_.Groups().size(), "CacheDeviceBlock group id out of range");
    std::vector<std::pair<CacheKey, CacheBlockRef>> newly_cached;
    // The entry holds the node of its hash. The node the restored entry was
    // published under is gone when no tier held the hash meanwhile; then the
    // entry starts one without a parent, until a publication through its hash
    // gives it its boundary (and, past page 0, links it).
    store_.Groups()[key.group_id].Index().Register(store_.DevicePool(), block_ref, key, access_epoch,
                                                   logical_block_index, boundary_kind, &newly_cached,
                                                   SessionForAccess(access_epoch));
    if (GroupIsState(key.group_id)) {
        BoundaryTree::Node* node = store_.Boundaries().Find(key.content_hash);
        _assert(node != nullptr, "an indexed checkpoint has a tree node");
        node->latest_state_access_epoch = std::max(node->latest_state_access_epoch, access_epoch);
    }
    if (!store_.MutationSink()) {
        return;
    }
    for (const auto& [cached_key, cached_block] : newly_cached) {
        (void)cached_block;
        store_.MutationSink()(cached_key, CacheMutation::kStored);
    }
}

std::unique_ptr<PrefixMatcher> MakePrefixMatcher(const CacheGroupSpec& spec) {
    switch (spec.attn_kind) {
        case AttnKind::kFull:
            _assert(spec.block_granularity > 0, "group block_granularity must be > 0");
            return std::make_unique<FullAttnMatcher>();
        case AttnKind::kState:
            _assert(spec.block_granularity == 0, "a State group has no token span");
            return std::make_unique<StateMatcher>();
        case AttnKind::kSlidingWindow:
            _assert(spec.block_granularity > 0, "group block_granularity must be > 0");
            _assert(spec.sliding_window > 0, "sliding window group requires a positive window");
            return std::make_unique<SwaMatcher>(spec.block_granularity, spec.sliding_window, spec.sink_tokens);
        default:
            FatalCheck(false, "unknown AttnKind in coordinator group spec");
            return nullptr;
    }
}

CacheCoordinator MakeCoordinator(std::span<const CacheGroupSpec> specs, std::int32_t prefix_granularity,
                                 BlockPool& pool, bool enable_l3_storage, BlockPool* host_pool,
                                 BlockPool* snapshot_pool, bool stream_device_cache_to_host, bool write_back_evictions,
                                 std::int64_t checkpoint_retention_requests,
                                 std::int32_t checkpoint_retention_leaf_distance) {
    _assert(!specs.empty(), "MakeCoordinator requires at least one spec");
    _assert(prefix_granularity > 0, "prefix_granularity must be > 0");
    _assert(specs.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "number of cache groups exceeds int32 range");
    std::vector<CacheGroup> groups;
    groups.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const CacheGroupSpec& spec = specs[i];
        const std::uint32_t group_id = static_cast<std::uint32_t>(i);
        _assert(spec.cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        if (spec.attn_kind == AttnKind::kState) {
            _assert(spec.block_granularity == 0, "a State group has no token span");
        } else {
            _assert(spec.block_granularity > 0 && prefix_granularity % spec.block_granularity == 0,
                    "group block_granularity must be a positive divisor of the prefix granularity");
        }
        std::int32_t sink_blocks = 0;
        if (spec.sink_tokens > 0) {
            _assert(spec.attn_kind == AttnKind::kSlidingWindow && !spec.replayable,
                    "only a cached sliding-window group keeps a sink");
            // A window of at least a prefix page needs every block of a tail,
            // so a tail in the sink's page holds the sink blocks as well.
            _assert(spec.sliding_window >= prefix_granularity, "a sink group's window must cover a prefix page");
            sink_blocks = spec.sink_tokens / spec.block_granularity + (spec.sink_tokens % spec.block_granularity != 0);
        }
        auto allocator =
            std::make_unique<GroupAllocator>(spec.cache_blocks_per_lcm_block, group_id, spec.shard_count, sink_blocks);
        groups.emplace_back(spec, std::move(allocator), MakePrefixMatcher(spec));
    }
    return CacheCoordinator{std::move(groups),
                            prefix_granularity,
                            pool,
                            enable_l3_storage,
                            host_pool,
                            snapshot_pool,
                            stream_device_cache_to_host,
                            write_back_evictions,
                            checkpoint_retention_requests,
                            checkpoint_retention_leaf_distance};
}

}  // namespace tokenspeed
