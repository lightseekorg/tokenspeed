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
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/core/active_parents.h"
#include "cache/core/block_pool.h"
#include "cache/core/cache_block_ref.h"
#include "cache/coordinator/admission_history.h"
#include "cache/coordinator/admission_types.h"
#include "cache/coordinator/cache_evictor.h"
#include "cache/coordinator/cache_store.h"
#include "cache/coordinator/host_cache.h"
#include "cache/coordinator/prefix_prober.h"
#include "cache/coordinator/storage_queries.h"
#include "cache/coordinator/group_geometry.h"
#include "cache/coordinator/session_registry.h"
#include "cache/coordinator/victim_set.h"
#include "cache/cache_group.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/boundary_tree.h"
#include "cache/prefix/checkpoint_tree.h"
#include "cache/prefix/tail_table.h"
#include "utils.h"

namespace tokenspeed {

struct CacheCoordinatorTestAccess;
// What the admission planner hands the commit (admission_planner.h).
struct AdmissionPlan;

// num_common_tokens is in tokens, a multiple of the shared prefix
// granularity. per_group[i] is group i's PrefixMatch at exactly that length.
struct CoordinatorMatch {
    std::int32_t num_common_tokens{0};
    std::vector<PrefixMatch> per_group;
    // The access epoch the resumed State checkpoint had before this match, 0
    // when this tier resumes none.
    std::uint64_t resumed_checkpoint_last_use{0};
};

// Multi-group fan-out over the per-attention groups, one shared BlockPool. Holds no per-request
// state; the request access clock is global, while each request carries its issued epoch.
class CacheCoordinator {
public:
    // The Host pool is available to explicit tier operations. Streaming controls
    // whether ordinary Device prefix publication also feeds the Host tier;
    // write_back_evictions whether an admission or a restore evicts Device
    // cache by segments, the evictions of those and of a lone-request
    // compaction are recorded for the plan's write-back drain
    // (TakeEvictedSegments), dead prefix entries are collected on both tiers,
    // and a retraction image's Host leg takes room by whole segments first
    // (the Host write-back policy; it needs a Host pool and excludes
    // streaming). The snapshot pool holds retracted requests' private KV
    // images (Snapshot / Restore); nullptr means the engine never retracts.
    // checkpoint_retention_requests and checkpoint_retention_leaf_distance are
    // SchedulerConfig's: a State checkpoint more than the leaf distance from a
    // leaf ranks, under pressure, as if last used that many new requests ago
    // (0 requests ranks every entry by its own use).
    // enable_l3_storage, host_pool, snapshot_pool, stream_device_cache_to_host,
    // write_back_evictions and checkpoint_retention_requests have no defaults:
    // a shorter call would convert a pool pointer to bool and silently disable
    // a tier instead of failing to compile. A leaf distance of 0 ranks only
    // the newest checkpoint on a branch as a leaf.
    CacheCoordinator(std::vector<CacheGroup> groups, std::int32_t prefix_granularity, BlockPool& pool,
                     bool enable_l3_storage, BlockPool* host_pool, BlockPool* snapshot_pool,
                     bool stream_device_cache_to_host, bool write_back_evictions,
                     std::int64_t checkpoint_retention_requests, std::int32_t checkpoint_retention_leaf_distance = 0);
    // Index entries and recorded write-backs point at the store's boundary
    // tree and at the evictor, so the coordinator stays where it was built,
    // and its destructor drops them before those members go.
    ~CacheCoordinator();
    CacheCoordinator(const CacheCoordinator&) = delete;
    CacheCoordinator& operator=(const CacheCoordinator&) = delete;
    CacheCoordinator(CacheCoordinator&&) = delete;
    CacheCoordinator& operator=(CacheCoordinator&&) = delete;
    std::size_t DemoteSession(const std::string& session_id, bool ended);
    std::size_t DemoteSessions(const std::vector<std::string>& ids, bool ended);
    std::size_t DemoteCacheGroups(const std::vector<std::string>& ids, bool ended);

    // The cache owner registered under an access epoch (SessionRegistry).
    SessionCacheOwner SessionForAccess(std::uint64_t epoch) const { return sessions_.OwnerForAccess(epoch); }
    std::int32_t NumGroups() const { return static_cast<std::int32_t>(store_.Groups().size()); }

    std::int32_t PrefixGranularity() const noexcept { return store_.PrefixGranularity(); }
    bool GroupIsState(std::size_t i) const;
    std::int32_t CheckpointGranularity() const noexcept { return store_.CheckpointGranularity(); }
    // The replay window: the largest sliding window over the replayable
    // groups, so a hit regenerates every page such a group retains; 0 when
    // none is replayable.
    std::int32_t ReplayWindowTokens() const noexcept { return store_.ReplayWindowTokens(); }
    // Tokens re-fed ahead of a prefix hit so every replayable group resumes
    // there: the window before the hit, or the whole hit when shorter.
    std::int32_t ReplayTokens(std::int32_t hit_tokens) const { return store_.ReplayTokens(hit_tokens); }

    GroupAllocator& Allocator(std::int32_t i) { return store_.Groups()[static_cast<std::size_t>(i)].Allocator(); }
    const GroupAllocator& Allocator(std::int32_t i) const {
        return store_.Groups()[static_cast<std::size_t>(i)].Allocator();
    }
    PrefixCacheIndex& GroupPrefixIndex(std::int32_t i) { return store_.Groups()[static_cast<std::size_t>(i)].Index(); }
    const PrefixCacheIndex& GroupPrefixIndex(std::int32_t i) const {
        return store_.Groups()[static_cast<std::size_t>(i)].Index();
    }
    const PrefixMatcher& GroupMatcher(std::int32_t i) const {
        return store_.Groups()[static_cast<std::size_t>(i)].Matcher();
    }
    // Match-policy and geometry views for scheduling code, in logical page
    // units. The managers are token-free; every token -> page conversion goes
    // through the per-group GroupGeometry the store holds.
    std::int32_t GroupBlockGranularity(std::int32_t i) const {
        return store_.Geometry()[static_cast<std::size_t>(i)].BlockGranularity();
    }
    // Fresh blocks a token-spanned group needs for num_tokens more; a State
    // group's table grows by StateStep alone.
    std::int32_t GroupBlocksNeededFor(std::int32_t i, const BlockTable& table, std::int32_t num_tokens) const {
        return GroupIsState(static_cast<std::size_t>(i))
                   ? 0
                   : store_.Geometry()[static_cast<std::size_t>(i)].BlocksNeededFor(table, num_tokens);
    }
    bool GroupHasReclaimableBlocksAt(std::int32_t i, const BlockTable& table, std::int32_t num_computed_tokens) const {
        const CacheGroup& group = store_.Groups()[static_cast<std::size_t>(i)];
        return !group.Allocator()
                    .ReclaimableBlockLocationsAt(group.Index(), table, groupExpiredBlockEndAt(i, num_computed_tokens))
                    .empty();
    }
    // Drops every State table's slot 0 (a decode step that writes slot 1 in
    // place), outside an admission.
    void ReleaseStateInputs(std::span<BlockTable> tables);
    // Hold one already-published, aligned input checkpoint across either
    // tier, including every cached sliding window it needs. Full history
    // is already held by the request. Missing any required block returns
    // an empty hold; replayable groups need no cached window.
    std::vector<CacheBlockRef> RetainInputCheckpoint(std::span<const std::string> prefix_hashes,
                                                     std::int32_t boundary_tokens) const;
    // A finishing producer makes its retained input available to the next
    // turn now. Refresh its recency without recording a prefix-cache hit or
    // changing the finishing request's session ownership.
    void RefreshRetainedCheckpoint(std::span<CacheBlockRef> blocks, std::uint64_t request_access_epoch);
    std::int32_t GroupBlocksReclaimableAt(std::int32_t i, const BlockTable& table, std::int32_t num_computed_tokens,
                                          bool count_uncached) const {
        const CacheGroup& group = store_.Groups()[static_cast<std::size_t>(i)];
        return group.Allocator().BlocksReclaimableAt(group.Index(), table,
                                                     groupExpiredBlockEndAt(i, num_computed_tokens), count_uncached);
    }

    // Admit consumes the probe (PrefixProber) whether or not it admits; a
    // caller that retries after a failure keeps a copy. A missing epoch
    // starts a new request; a supplied epoch continues that request. Once
    // commit starts, an internal plan/pool mismatch is fatal because partial
    // commit is not rolled back.
    // demands say what each group needs for the round ahead; progress says
    // what the request completed since its previous admission. Completed
    // pages are published and expired slots reclaimed inside this
    // transaction, before the new pages are acquired. An owner that keeps
    // writing past the point it publishes (OwnerTail::kCopiedIn) moves off
    // its published tail blocks into copy_in_blocks, the blocks the request
    // holds for that point (aligned like AdmissionResult::copy_in_blocks,
    // every slot filled): Admit swaps in the ones it uses and releases the
    // rest, leaving copy_in_blocks empty, so the publication needs no block
    // the request did not reserve. Any other admission passes none.
    // copy_in_point is the request's next such point: this admission reserves
    // its copy-in blocks for the slots whose blocks it allocates or fills
    // from the prefix, each with the block it serves
    // (AdmissionResult::copy_in_blocks). Under write_back_evictions each set
    // of cache victims is recorded
    // for the plan's drain (TakeEvictedSegments); each call erases the records
    // it added that ended up empty.
    std::optional<AdmissionResult> Admit(PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                         const RequestProgress& progress,
                                         std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                         std::optional<CopyInPoint> copy_in_point,
                                         std::optional<std::uint64_t> request_access_epoch,
                                         const std::string& session_id, const std::string& cache_group_id);
    // Admit, planned by `Planner`: AdmissionPlanner, or a subclass of it that
    // goes on past the segment rules (LoneRequestAdmission's). Defined in
    // admission_planner.h, which a caller includes.
    template <typename Planner>
    std::optional<AdmissionResult> AdmitWith(PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                             const RequestProgress& progress,
                                             std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                             std::optional<CopyInPoint> copy_in_point,
                                             std::optional<std::uint64_t> request_access_epoch,
                                             const std::string& session_id, const std::string& cache_group_id);

    // The data slots of a retracted request are the non-null slots covering
    // [0, num_tokens) in every group -- the unaligned tail block included;
    // reserve blocks beyond are not imaged, a restore re-reserves them. The
    // geometry turns num_tokens into each group's slot count here, so the
    // allocator stays token-free.
    //
    // Those data slots whose Device block is a published prefix entry, with
    // the key it is published under: they can ride Host L2 as that entry
    // (ImageSlot::block is the Device block here; the transfer manager
    // swaps in the Host block). Everything else goes to the snapshot pool.
    std::vector<std::vector<ImageSlot>> PublishedDataSlots(std::span<const BlockTable> tables,
                                                           std::int32_t num_tokens) const;
    // The same slots once `progress` has been published as well: the entries
    // that exist now plus the slots CacheCompletedBlocks(progress) would
    // register (the single publication rule, CacheStore::CompletedBlockRanges), each on
    // the block the publication leaves it on: a key the Device index holds
    // already re-points the slot to that entry's block. The victim probe
    // reads the image's L2/pool split off this before the retraction
    // publishes. Without completed pages it is PublishedDataSlots.
    std::vector<std::vector<ImageSlot>> PublishedDataSlotsAfter(std::span<const BlockTable> tables,
                                                                std::int32_t num_tokens,
                                                                const RequestProgress& progress) const;

    // A retracted request's KV image plus the Device -> snapshot-pool copies
    // that fill its snapshot-pool slots. The pairs pin the Device sources
    // only until the caller resolves them into a wire op; the image's
    // references keep the Host blocks for as long as the request may be
    // restored.
    struct ImageTaken {
        RetractionImage image;
        std::vector<BlockTransfer> store_pairs;
    };
    // Images every data slot not already served by Host L2 (host_cached_slots,
    // per group, each with its pinned Host block) into the snapshot pool, in
    // the bucket of the Device block it images, and merges both legs into
    // one shape per group. Returns nullopt without any mutation when the pool
    // cannot place some group's bucket demand (or no snapshot pool exists):
    // the probe over every group precedes the first acquisition.
    std::optional<ImageTaken> TakeImage(std::span<const BlockTable> tables, std::int32_t num_tokens,
                                        std::span<const std::vector<ImageSlot>> host_cached_slots);
    // TakeImage's fit probe alone, no mutation: whether the snapshot pool
    // holds every data slot of [0, num_tokens) that host_served_slots (per
    // group, by slot_index and key; the blocks are not read) does not cover.
    // The victim policy asks this before choosing a victim.
    bool SnapshotPoolHolds(std::span<const BlockTable> tables, std::int32_t num_tokens,
                           std::span<const std::vector<ImageSlot>> host_served_slots) const;
    // Slots covering [0, num_tokens) of a table and the capacity left inside
    // the last of them.
    struct DataSpan {
        std::int32_t num_blocks{0};
        std::int32_t available_tokens{0};
    };
    // Rebuilds every group's table from its image -- identical block count,
    // null holes, tail capacity and reclaimed prefix, one fresh Device block
    // per imaged slot in the imaged bucket -- and then admits `demands` on
    // the rebuilt tables (a dense growth, normally zero, plus each group's
    // reserve) inside the same AdmissionPlanner pass, so cache-only blocks
    // are evicted exactly as for an admission and a failed restore leaves
    // nothing allocated. load_pairs carries the Host L2 -> Device copies
    // (keyed and republished at the ACK, except the row of the partial page
    // the request keeps writing, which stays its private copy),
    // snapshot_pairs the snapshot pool -> Device copies, new_page_ids only
    // the reserve pages appended
    // beyond the imaged shape (the copies fill their destinations whole, so
    // the plan zeroes nothing it restores). copy_in_point as for Admit: the
    // retraction released the copy-in blocks the victim held, and the restore
    // reserves those of the slots it rebuilds or grows.
    // request_access_epoch is the request's first-admission epoch, carried
    // on. Under write_back_evictions each set of cache victims is recorded for
    // the plan's drain as an admission's is; each call erases the records it
    // added that ended up empty. An L2 slot whose key the Device index still
    // holds is claimed instead of copied; when the claims leave no plan, the
    // restore plans once more claiming nothing under write_back_evictions
    // (docs/design/scheduler.md §4).
    std::optional<AdmissionResult> Restore(const RetractionImage& snapshot, std::span<const GroupDemand> demands,
                                           std::optional<CopyInPoint> copy_in_point,
                                           std::uint64_t request_access_epoch);
    // Restore that, whatever the eviction policy, plans once more claiming
    // nothing when the claims leave no plan: every imaged slot then a copy.
    // After a failed Restore it is the same as having called it instead.
    std::optional<AdmissionResult> RestoreWithoutClaimsIfUnfit(const RetractionImage& snapshot,
                                                               std::span<const GroupDemand> demands,
                                                               std::optional<CopyInPoint> copy_in_point,
                                                               std::uint64_t request_access_epoch);
    // The coordinator's components, for what works on the cache beside it
    // through their own interfaces (the scheduler, the tier managers,
    // LoneRequestAdmission).
    CacheStore& Store() { return store_; }
    const CacheStore& Store() const { return store_; }
    CacheEvictor& Evictor() { return evictor_; }
    const CacheEvictor& Evictor() const { return evictor_; }
    const PrefixProber& Prober() const { return prober_; }
    StorageQueries& Storage() { return storage_; }
    const StorageQueries& Storage() const { return storage_; }
    const AdmissionHistory& History() const { return history_; }
    HostCache& Host() { return host_; }
    const HostCache& Host() const { return host_; }
    // Whether Device eviction writes back by segments (the Host write-back
    // policy).
    bool WritesBackEvictions() const { return write_back_evictions_; }
    bool HasSnapshotPool() const { return store_.SnapshotPool() != nullptr; }
    std::int32_t NumFreeSnapshotLcmBlocks() const {
        return store_.SnapshotPool() == nullptr ? 0 : store_.SnapshotPool()->NumEmptyLcmBlocks();
    }
    // Capacity views for scheduling code, counted in LCM parent blocks. The
    // counts are opaque capacity units to the scheduler: all packing/geometry
    // arithmetic stays behind these methods.
    //
    // Number of LCM blocks that become reclaimable after dropping the exact
    // request-owned refs in tables. Used only to rank Retraction victims.
    std::int32_t NumNewlyReleasableLcmBlocks(std::span<const BlockTable> tables) const;
    // Empty parents plus parents whose every child is an unpinned cache entry.
    // Scans the pool and every group's index: a diagnostic, not a per-step
    // gauge. Per-step accounting composes NumEmptyLcmBlocks and
    // NumActiveLcmBlocks instead.
    std::int32_t NumAvailableLcmBlocks() const;
    std::int32_t NumEmptyLcmBlocks() const { return store_.DevicePool().NumEmptyLcmBlocks(); }
    std::int32_t TotalLcmBlocks() const { return store_.DevicePool().NumLcmBlocks(); }
    // Distinct LCM blocks the request tables hold, counted by the tables'
    // own ParentRefs as they change, so reading it walks nothing.
    std::int32_t NumActiveLcmBlocks() const noexcept { return store_.ActiveParentGauge().Count(); }
    // The same count taken by walking the given per-request table sets:
    // the reference debug builds check NumActiveLcmBlocks against.
    std::int32_t CountActiveLcmBlocks(std::span<const std::span<const BlockTable>> request_tables) const;
    // Free pages (group page units) this group could still place, counting its
    // partially filled parents and every empty parent.
    std::int32_t GroupAvailablePages(std::int32_t group_index) const;

    // Registers an exact range, used for transferred prefix blocks and tests.
    // first_prefix_page is the logical prefix-page index of content_hashes[0].
    // Runtime publication during Admit follows each group's boundary contract.
    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::uint64_t access_epoch, std::int32_t first_prefix_page, CacheBoundaryKind boundary_kind);
    // Publication without admission, for finish, retraction and remote
    // completion. progress.completed_pages must be present.
    void CacheCompletedBlocks(std::span<BlockTable> tables, const RequestProgress& progress,
                              std::uint64_t access_epoch);
    // The first page of completed.prefix_hashes publishing `completed`
    // registers an entry under: the first new page, or a page below it --
    // the deepest one, where a State checkpoint lands when no page is new,
    // or the first a sliding window behind one of its boundaries reaches
    // back into.
    std::int32_t FirstPublishedPage(const CompletedPages& completed) const;
    // The leading pages [0, result) a publication registers sink blocks under
    // again, below FirstPublishedPage; 0 when it publishes no sink block.
    std::int32_t PublishedSinkPages(const CompletedPages& completed) const;
    void ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens);
    void ConsumeReservedTokens(std::span<BlockTable> tables, std::int32_t num_tokens);
    void Free(std::span<BlockTable> tables);
    // Clears only the Device prefix index. Returns false without mutation when
    // any cached block still has an owner outside its prefix index.
    bool ClearDeviceCache();
    // Clears both Device and Host prefix indexes. Returns false without
    // mutation when either tier still has a pinned cached block.
    bool ClearCache();
    // Same pin checks as ClearDeviceCache / ClearCache, with no mutation.
    // include_host selects both tiers (Host is a no-op without a host pool).
    bool CacheIsClearable(bool include_host) const;
    // Evicts, on both tiers, the prefix entries no probe can resume from any
    // more (write_back_evictions only; nothing is queued otherwise). A node no
    // eviction write-back is bringing to the Host is dead when no checkpoint
    // boundary is left in its subtree on either tier (with State groups), when
    // it is a tail no probe enumerates (its hint is gone), or when it is a gap
    // a prefix-closed group cannot cross: no prefix-closed group holds it on
    // either tier, whatever State entries or children it keeps. The boundary
    // tree queues candidates as holders leave it (the node a removal stops at
    // once no checkpoint boundary is left under it, a gap); a tail the cap
    // left behind and a boundary a settlement did not publish are queued too.
    // Each is checked again when collected -- a live request may have
    // published a checkpoint boundary under it since -- then the collection
    // climbs while the parent is dead and none of its entries is in use, and
    // takes every entry of that subtree not in use, deepest boundary first. A
    // root with an entry in use waits for the next call. The scheduler calls it
    // once per plan, after the build and its write-back drain.
    void CollectDeadEntries();

    // The segments this plan's admissions and restores recorded
    // (CacheEvictor::TakeSegments); the write-back drain is their one consumer.
    std::vector<CacheEvictor::EvictedSegment> TakeEvictedSegments() { return evictor_.TakeSegments(); }
    // Marks as superseded the Device entries of the tail a request resumed from
    // (AdmissionResult::resumed_tail_hash) -- every State group's checkpoint
    // there and the tail's blocks -- once the end state its finish published
    // supersedes them (docs/design/scheduler.md §1.2 gives why): they stay
    // cached and matchable, re-registered at CacheStore::kSupersededAccessEpoch, first in
    // the Device's eviction order (CacheEvictor::EvictCachedBlockAlone counts them as request
    // reclaim). A request that acquires them, or a publication of the same
    // key, gives them its own epoch. Nothing is marked when the tail table no
    // longer lists the tail (the cap dropped it meanwhile), or when any of its
    // entries was touched after access_epoch, the finishing request's: another
    // request acquired or published it since.
    void MarkSupersededStateInput(const ContentHash& resumed_tail_hash, std::uint64_t access_epoch);

    // Set once, by the scheduler, at construction. On, an entry a demotion
    // marked counts as demoted until a hit, touch or republication ends it:
    // no store or eviction write-back copies it to the Host
    // (DemotedEntriesNotWrittenBack), and a Host segment plan takes a
    // Host-only checkpoint boundary whose checkpoints are all demoted after
    // the Host copies of KV a request holds on the Device and before those of
    // unheld Device KV. Off, no entry counts as demoted. Either way a
    // session's demoted entries and superseded inputs keep their own rules.
    void SetCheckpointDemotion(bool enabled) { evictor_.SetCheckpointDemotion(enabled); }
    // Each State group's checkpoint at the page boundary that ends
    // prefix_hashes and the sliding-window blocks behind it, less the blocks
    // behind any of kept_boundaries: deeper boundaries on the same chain, in
    // tokens, that stay resumable with their windows.
    std::vector<CacheKey> BoundaryCheckpointKeys(std::span<const std::string> prefix_hashes,
                                                 std::span<const std::int32_t> kept_boundaries) const;
    // Demotes the entries of `keys` on the Device and in Host L2: dates each
    // back to CacheStore::kDemotedAccessEpoch, marks it demoted and forgets its
    // acquisition (PrefixCacheIndex::Demote), skipping one a request or a
    // transfer holds or one demoted there already. A later hit, touch or
    // republication dates an entry forward again and ends its demotion.
    // Returns the entries demoted.
    std::int32_t DemoteEntries(std::span<const CacheKey> keys);
    // The newest access epoch any acquisition or publication of a State
    // checkpoint under `hash` carried, on either tier, while either tier
    // holds the boundary; 0 when neither does.
    std::uint64_t LatestStateAccessEpoch(const ContentHash& hash) const;

    CacheBlockRef AcquireDeviceCachedBlock(const CacheKey& key) const;
    // Collection/pinning follows host-tier presence, so the slide credit flips count_uncached on this.
    bool StreamsDeviceCacheToHost() const { return stream_device_cache_to_host_; }

    // The Device entries an admission, a restore or a compaction evicted
    // under the write-back policy without recording them: no probe could resume from
    // them on the Host (a tail whose hint is gone, a boundary with no
    // checkpoint boundary at or under it).
    std::uint64_t DeadEntriesNotWrittenBack() const { return evictor_.DeadEntriesNotWrittenBack(); }
    // The demoted Device entries (SetCheckpointDemotion) an admission, a
    // restore or a compaction evicted under the write-back policy -- as a victim, a request's
    // own released slot or in a stranded suffix -- and dropped instead of
    // recording them. Like the dead ones, they count under the eviction's
    // cause.
    std::uint64_t DemotedEntriesNotWrittenBack() const { return evictor_.DemotedEntriesNotWrittenBack(); }

    // Cumulative, never reset (accounting only).
    const CacheEvictor::DeviceEvictionStats& DeviceEvictions() const { return evictor_.DeviceEvictions(); }

    // Publishes a Device block a restore filled from Host L2
    // (CompleteSnapshotRestore): the entry comes back with the logical block
    // index and boundary kind it was published under before the retraction,
    // so eviction ranks it as it did the original, and dated with the
    // restored request's access epoch, as its admission and its restore's
    // Device hits dated the rest of its prefix -- not as a later request's
    // hit, so the request's finish still marks its input superseded
    // (MarkSupersededStateInput).
    void CacheDeviceBlock(CacheBlockRef& block_ref, const CacheKey& key, std::int32_t logical_block_index,
                          CacheBoundaryKind boundary_kind, std::uint64_t access_epoch);

    // Reports real device-cache entry insertions and removals. The scheduler
    // folds the per-group mutations into one externally visible prefix event.
    void SetCacheMutationSink(CacheMutationSink sink) { store_.SetMutationSink(std::move(sink)); }
    struct BoundaryEventStatus {
        bool publishable;
        bool has_device_entries;
    };
    // Routing requires every continuous-history group on Device; auxiliary
    // windows and State do not gate publication. Without continuous history,
    // no boundary is publishable. Any Device entry retains the descriptor.
    BoundaryEventStatus GetBoundaryEventStatus(const CacheKey& boundary, std::int32_t tail_tokens) const;
    // Whether every State group caches its checkpoint published under the
    // chain hash content_hash, on the Device or the Host, as the probe takes
    // each from whichever tier holds it (false for a model without State
    // groups).
    bool HoldsStateCheckpoint(const ContentHash& content_hash) const;

    const CheckpointTree& Checkpoints() const { return store_.Checkpoints(); }

    // Published tail boundaries the probe can enumerate, and how many the
    // per-prior cap has dropped.
    std::size_t NumTailEntries() const { return store_.Boundaries().Tails().Size(); }
    std::int64_t TailCapEvictions() const { return store_.Boundaries().Tails().CapEvictions(); }

private:
    friend struct CacheCoordinatorTestAccess;

    struct AcquiredPrefix {
        CoordinatorMatch device;
        CoordinatorMatch host;
        // The D role's claimed prompt-end checkpoint per State group (empty
        // otherwise), aligned by group index.
        std::vector<CacheBlockRef> state_at_prompt_end;
    };

    // The snapshot pool's share of an image, probed without acquiring: per
    // group the data span and the slots (with their buckets) that ride the pool.
    struct SnapshotPoolPlan {
        std::vector<DataSpan> spans;
        std::vector<std::vector<std::int32_t>> private_slots;
        std::vector<std::vector<std::int32_t>> private_buckets;
    };
    std::optional<SnapshotPoolPlan> planSnapshotPool(std::span<const BlockTable> tables, std::int32_t num_tokens,
                                                     std::span<const std::vector<ImageSlot>> host_served_slots) const;
    // What every AdmitWith shares around its plan: checking
    // the request (and binding its tables to the active-parent gauge), the
    // demands it plans with when a replayable group regenerates its window
    // (CacheStore::ReplayedDemands; empty otherwise), and committing a plan.
    std::vector<GroupDemand> prepareAdmission(const PrefixProbe& prefix, std::span<const GroupDemand> demands,
                                              const RequestProgress& progress,
                                              std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                              std::optional<CopyInPoint> copy_in_point);
    AdmissionResult commitAdmission(AdmissionPlan&& plan, std::span<const GroupDemand> demands,
                                    const RequestProgress& progress,
                                    std::span<std::vector<CacheBlockRef>> copy_in_blocks,
                                    std::optional<CopyInPoint> copy_in_point,
                                    std::optional<std::uint64_t> request_access_epoch, const std::string& session_id,
                                    const std::string& cache_group_id);
    // Demotes the entries the demoted owners own, on both tiers.
    std::size_t demoteOwners(const SessionRegistry::DemotedOwners& demoted);
    // Restore and RestoreWithoutClaimsIfUnfit: whether claims that leave no plan are
    // dropped for a second plan claiming nothing.
    std::optional<AdmissionResult> restore(const RetractionImage& snapshot, std::span<const GroupDemand> demands,
                                           std::optional<CopyInPoint> copy_in_point, std::uint64_t request_access_epoch,
                                           bool plan_without_claims_if_unfit);
    template <CacheTier Tier>
    CoordinatorMatch acquireTierWithKeys(std::span<const std::vector<CacheKey>> group_keys, std::int32_t floor_tokens,
                                         PrefixProbe::Tier&& probe, std::uint64_t access_epoch);
    AcquiredPrefix acquirePrefix(PrefixProbe&& probe, std::uint64_t access_epoch);
    template <CacheTier Tier>
    void cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table, std::span<const CacheKey> keys,
                                 std::int32_t first_cache_block, std::uint64_t access_epoch,
                                 CacheBoundaryKind boundary_kind, bool stream_completed_to_host);
    // Registers blocks[j] under keys[j] (logical block first_logical_block + j)
    // and reports and streams what was newly cached; cacheFullBlocksForGroup
    // passes a table window, an in-forward State checkpoint its own block.
    template <CacheTier Tier>
    void cacheBlocksForGroup(std::size_t group_index, std::span<CacheBlockRef> blocks, std::span<const CacheKey> keys,
                             std::int32_t first_logical_block, std::uint64_t access_epoch,
                             CacheBoundaryKind boundary_kind, bool stream_completed_to_host);
    template <CacheTier Tier>
    void cacheCompletedBlocksForGroup(std::size_t group_index, BlockTable& table, const CompletedPages& completed,
                                      std::uint64_t access_epoch);
    void cacheDeviceCompletedBlocksForGroup(std::size_t group_index, BlockTable& table, const CompletedPages& completed,
                                            std::uint64_t access_epoch);
    // Records a published tail in the tail table and evicts the tail the
    // per-prior cap dropped, on both tiers.
    void recordPublishedTail(const CompletedPages& completed);
    // The checkpoint tree follows the State entries each tier holds (the
    // boundary tree hands it each residency change); a checkpoint a
    // publication observes gets its residency here.
    void recordCheckpoints(const CompletedPages& completed, std::uint64_t access_epoch);
    void syncCheckpoint(const ContentHash& hash);
    // Whether an imaged slot belongs to the page the computed frontier ends
    // inside: the restored request writes and later republishes those blocks,
    // so they are copied, never claimed. Every slot below is immutable.
    bool writableImageSlot(std::size_t group_index, const ImageTable& shape, std::int32_t slot_index) const;
    bool deviceCacheIsClearable() const;
    // Admit's copy_in_blocks: every slot's block for an owner copying in past
    // its tail, none otherwise.
    void validateCopyInBlocks(const RequestProgress& progress,
                              std::span<const std::vector<CacheBlockRef>> copy_in_blocks) const;
    std::int32_t groupExpiredBlockEndAt(std::int32_t i, std::int32_t num_computed_tokens) const {
        return store_.Geometry()[static_cast<std::size_t>(i)].ExpiredBlockEndAt(
            store_.Groups()[static_cast<std::size_t>(i)].Spec(), num_computed_tokens);
    }
    // The cache's groups, geometry, pools and boundary tree.
    CacheStore store_;
    // Device eviction and the write-back records it leaves.
    CacheEvictor evictor_{store_};
    // Read-only prefix matching over both tiers.
    PrefixProber prober_{store_, evictor_};
    bool stream_device_cache_to_host_{false};
    // An admission or a restore evicts Device cache by segments: a checkpoint
    // boundary with the pages only it depends on and everything under them, so
    // eviction keeps the Device's coverage closed from the root (a restore's
    // republication can open it: cache-concepts.md); records what its cache
    // victims (and a lone-request compaction's) leave behind for this plan's
    // write-back drain; and the dead
    // prefix entries either tier is left with are collected.
    bool write_back_evictions_{false};
    // The access epoch clock and the admissions checkpoint retention reads.
    AdmissionHistory history_;
    // Host L2: allocation and placement, publication, write-back settlement.
    HostCache host_{store_, evictor_, history_, write_back_evictions_};
    // What L3 storage is asked for a prompt's prefix, and the Host blocks a
    // read lands in.
    StorageQueries storage_;
    // The cache owners of the admitted requests.
    SessionRegistry sessions_;
};

// The prefix-match policy of one spec: full attention is prefix-closed with
// no lookback; a sliding window resumes only behind enough cached pages; a
// State group resumes at a boundary it holds a checkpoint for. The
// coordinator's groups and the capacity model ask the same matcher.
std::unique_ptr<PrefixMatcher> MakePrefixMatcher(const CacheGroupSpec& spec);

// One CacheGroup per spec (group_id = index), sharing one scheduler prefix
// domain P while each group may use a smaller cache-page token count.
CacheCoordinator MakeCoordinator(std::span<const CacheGroupSpec> specs, std::int32_t prefix_granularity,
                                 BlockPool& pool, bool enable_l3_storage, BlockPool* host_pool,
                                 BlockPool* snapshot_pool, bool stream_device_cache_to_host, bool write_back_evictions,
                                 std::int64_t checkpoint_retention_requests,
                                 std::int32_t checkpoint_retention_leaf_distance = 0);
}  // namespace tokenspeed
