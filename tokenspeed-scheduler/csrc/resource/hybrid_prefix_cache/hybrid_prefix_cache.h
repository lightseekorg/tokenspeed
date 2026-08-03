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

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "resource/allocator/paged_cache_group.h"
#include "resource/hybrid_prefix_cache/mamba_eviction_manager.h"
#include "resource/radix_tree/mamba_slot.h"
#include "scheduler/operations/cache.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/types.h"

namespace tokenspeed {

class MambaChunkAllocator;
class MambaHostAllocator;
class ForwardOperationBase;
struct PagedCacheSnapshot;

class HybridPrefixCache {
public:
    // `mamba_allocator` may be null; paged-cache adjunct is enabled separately.
    HybridPrefixCache(KVPrefixCache& prefix_cache, MambaChunkAllocator* allocator, std::int32_t mamba_cache_chunk_size,
                      MambaHostAllocator* mamba_host_allocator = nullptr);
    ~HybridPrefixCache();

    MatchResult Match(const token_vec_t& token_ids, MatchIntent intent = MatchIntent::PrefixReuse);
    MatchResult Match(const std::vector<std::span<const std::int32_t>>& token_pages,
                      MatchIntent intent = MatchIntent::PrefixReuse);

    bool EnsureMambaCapacityByEvict(std::int32_t num_slots, TreeNode* protected_node = nullptr);
    void InsertMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot);
    std::int32_t AlignMambaCacheSeqlen(std::int32_t seqlen) const;
    TreeNode* FindLastMambaNode(TreeNode* from) const;
    TreeNode* FindLastMambaHostNode(TreeNode* from) const;
    bool EnsureMambaHostCapacityByEvict(std::int32_t num_slots, TreeNode* protected_node = nullptr);
    std::vector<TransferPair> PrepareMambaHostWriteBack(const std::vector<TreeNode*>& nodes);
    std::vector<TransferPair> PrepareMambaDeviceLoadBack(const std::vector<TreeNode*>& nodes);
    void OnKVHostEvict(TreeNode* node);
    void OnKVDeviceDemote(TreeNode* node);
    void OnMambaHostWriteBackDone(TreeNode* last_node, bool success = true);
    void OnMambaHostWriteBackDone(const std::vector<TreeNode*>& nodes, bool success = true);
    void DemoteIdleMambaDeviceCopiesPresentOnHost();

    // Takes ownership. Duplicate group_id throws std::invalid_argument.
    void RegisterPagedCacheGroup(std::unique_ptr<PagedCacheGroupAllocator> allocator);
    void RegisterPagedCacheHostGroup(std::unique_ptr<PagedCacheGroupAllocator> allocator);

    // History alignment is the LCM of RawTokensPerPage() over the required
    // History-family groups. Continuation state groups only need their trailing
    // window and are intentionally outside prefix-reuse admission.
    void EnablePagedCacheAdjunct(std::vector<std::string> required_groups,
                                 std::unordered_map<std::string, std::int32_t> sliding_window_per_group);

    bool HasMambaAdjunct() const { return mamba_allocator_ != nullptr; }
    bool HasPagedCacheAdjunct() const { return paged_cache_history_alignment_tokens_ > 0; }
    bool HasPagedCacheHostAdjunct() const { return !paged_cache_host_allocators_.empty(); }
    std::int32_t PagedCacheHistoryAlignmentTokens() const { return paged_cache_history_alignment_tokens_; }
    const std::vector<std::string>& PagedCacheRequiredGroups() const { return paged_cache_required_groups_; }

    // Group introspection: throws std::out_of_range on unknown group_id.
    std::vector<std::string> PagedCacheGroupIds() const;
    std::int32_t PagedCacheGroupTotalPages(const std::string& group_id) const;
    std::int32_t PagedCacheGroupAvailablePages(const std::string& group_id) const;
    std::int64_t PagedCacheGroupFailedAllocCount(const std::string& group_id) const;
    std::int32_t PagedCacheHostGroupTotalPages(const std::string& group_id) const;
    std::int32_t PagedCacheHostGroupAvailablePages(const std::string& group_id) const;
    std::int64_t PagedCacheHostGroupFailedAllocCount(const std::string& group_id) const;

    // Per-request introspection: unknown group_id throws; unknown request_id returns empty.
    std::vector<std::int32_t> GetRequestPagedCachePageIds(const std::string& request_id,
                                                          const std::string& group_id) const;
    std::int32_t GetRequestPagedCacheBaseLogicalPage(const std::string& request_id, const std::string& group_id) const;
    bool HasRequestPagedCacheTables(const std::string& request_id) const;

    // Unified paged-cache lifecycle surface used by the Scheduler. All methods
    // below are no-ops when no paged-cache groups are registered.

    // Initial per-group simulated_free budget mirroring live allocator state.
    std::map<std::string, std::int32_t> InitialSimulatedFree() const;

    // Ensure tables exist and cover [first_raw_position_of_op, target_raw_tokens_exclusive).
    // Borrowed prefix is imported BEFORE any fresh allocation on a fresh table.
    void AcquireForRequest(const std::string& request_id, std::int32_t first_raw_position_of_op,
                           std::int32_t target_raw_tokens_exclusive,
                           const MatchResult::PagedCache& paged_cache_hit = {});

    // Owned pages return to the pool via OwnedPages RAII; borrowed ids are dropped.
    void ReleaseRequest(const std::string& request_id);

    // Reclaim request-local paged-cache tail slots beyond the accepted token
    // length while retaining slots referenced by an already-dispatched
    // overlapped decode.
    void RewindRequest(const std::string& request_id, std::int32_t accepted_raw_tokens,
                       std::int32_t protected_tail_tokens = 0);

    // Allocate request-local device pages for a host paged-cache hit and import
    // them as owned prefix pages. Returns the HOST->DEVICE transfer specs.
    std::vector<PagedCacheTransferPair> PreparePagedCacheDeviceLoadBack(const std::string& request_id,
                                                                        const MatchResult::PagedCache& host_hit);

    // Allocate pending host snapshots from device snapshots. Host metadata is
    // invisible to Match() until OnPagedCacheHostWriteBackDone(..., true).
    std::vector<PagedCacheTransferPair> PreparePagedCacheHostWriteBack(const std::vector<TreeNode*>& nodes);
    void OnPagedCacheHostWriteBackDone(const std::vector<TreeNode*>& nodes, bool success = true);

    // Fill op.paged_cache_pages / op.paged_cache_page_base_offsets from the tables.
    void PopulateOp(ForwardOperationBase& op_base) const;

    // Run admission against `simulated_free`; prunes evictable snapshots on
    // group-pool pressure, then applies the debit on success. When present,
    // `commit_target_raw_tokens` is the terminal depth CommitChunk will publish
    // immediately before AcquireForRequest for this chunk. `commit_token_pages`
    // identifies an already-existing exact terminal so admission can account
    // for continuation-state groups that CommitChunk will reuse rather than
    // checkpoint again.
    bool AdmitChunk(const std::string& request_id, std::int32_t first_raw_position_of_op,
                    std::int32_t target_raw_tokens_exclusive, std::map<std::string, std::int32_t>& simulated_free,
                    const MatchResult::PagedCache& paged_cache_hit = {},
                    std::optional<std::int32_t> commit_target_raw_tokens = std::nullopt,
                    std::span<const std::span<const std::int32_t>> commit_token_pages = {});

    // Commit newly-written full LCM segments into TreeNode PagedCacheSnapshots.
    void CommitChunk(const std::string& request_id, TreeNode* terminal);

    // Attach a snapshot to `node`, computing `complete_families` from which
    // required-per-family group ids are present and registering the node in
    // `paged_cache_snapshot_nodes_`. Returns false when either argument is
    // null (defensive no-op). Accepts partial snapshots; the per-policy
    // "snapshot must be full" invariant is enforced upstream by CommitChunk.
    bool AttachPagedCacheSnapshotToNode(TreeNode* node, std::unique_ptr<PagedCacheSnapshot> snapshot);
    bool AttachPagedCacheHostSnapshotToNode(TreeNode* node, std::unique_ptr<PagedCacheSnapshot> snapshot);

    // Drops `node` from the membership set, then detaches and returns the snapshot.
    std::unique_ptr<PagedCacheSnapshot> DetachPagedCacheSnapshotFromNode(TreeNode* node);
    std::unique_ptr<PagedCacheSnapshot> DetachPagedCacheHostSnapshotFromNode(TreeNode* node);

    // Callback from KV prefix-cache eviction.
    void OnKVEvict(TreeNode* node);

    // Callback from RadixTree just before prune destroys a node. Detaches its
    // adjunct resources while their allocators are alive and removes every raw
    // TreeNode* bookkeeping entry. A paged snapshot must have no table borrower
    // at this point; violating that lifetime invariant is an error.
    void OnNodeDestroyed(TreeNode* node);

    std::int32_t AvailableSlots() const;
    KVPrefixCache& GetKVPrefixCache() { return kv_prefix_cache_; }

private:
    friend class HybridPrefixCacheTestPeer;

    // Per-family classification of admission failure; drives state-only vs
    // full prune strategy.
    enum class AdmissionFailureKind { kNone, kHistoryStarved, kStateStarved, kBothStarved };

    struct PagedCacheGroupAdmission {
        bool ok{true};
        std::unordered_set<std::string> failed_groups{};
        std::map<std::string, std::int32_t> releasable_owned_pages{};
        std::map<std::string, std::int32_t> new_pages_needed{};
    };

    struct PagedCacheAdmissionContext {
        std::optional<std::int32_t> commit_target_raw_tokens{};
        std::span<const std::span<const std::int32_t>> commit_token_pages{};
    };

    // Classify which family caused `admission.ok == false`.
    AdmissionFailureKind ClassifyAdmissionFailure(const PagedCacheGroupAdmission& admission) const;

    // Drop only state-family groups from `node`'s snapshot; history portion
    // remains and the node stays registered. Returns true iff state groups removed.
    bool DetachStateSnapshotFromNode(TreeNode* node);
    std::unique_ptr<PagedCacheSnapshot> DetachPagedCachePendingHostSnapshotFromNode(TreeNode* node);

    void RefreshPagedCacheSnapshotCompleteness(PagedCacheSnapshot& snapshot) const;
    bool isPagedCacheSnapshotBorrowed(const TreeNode* node,
                                      std::optional<PagedCacheGroupFamily> family = std::nullopt) const;
    bool adoptExistingPagedCacheSnapshot(PagedCacheSnapshot& existing,
                                         std::map<std::string, PagedCacheGroupTable>& tables, std::int32_t target);
    bool commitTerminalContinuationSnapshot(std::map<std::string, PagedCacheGroupTable>& tables, TreeNode* terminal,
                                            std::int32_t target);

    void augmentMatch(MatchResult& match) const;
    void augmentMatchPagedCache(MatchResult& match, MatchIntent intent) const;

    bool admitPagedCacheChunk(const std::string& request_id, std::int32_t first_raw_position_of_op,
                              std::int32_t target_raw_tokens_exclusive,
                              std::map<std::string, std::int32_t>& simulated_free,
                              const MatchResult::PagedCache& paged_cache_hit,
                              const PagedCacheAdmissionContext& context);

    // Build admission record without mutating any table.
    PagedCacheGroupAdmission checkPagedCacheGroupAdmission(const std::string& request_id,
                                                           std::int32_t first_raw_position_of_op,
                                                           std::int32_t target_raw_tokens_exclusive,
                                                           const std::map<std::string, std::int32_t>& simulated_free,
                                                           const MatchResult::PagedCache& paged_cache_hit,
                                                           const PagedCacheAdmissionContext& context) const;

    // Owned releases credit, new-page needs debit.
    static void applyPagedCacheGroupAdmissionDebit(std::map<std::string, std::int32_t>& simulated_free,
                                                   const PagedCacheGroupAdmission& admission);
    void refreshPagedCacheSimulatedFree(std::map<std::string, std::int32_t>& simulated_free) const;

    KVPrefixCache& kv_prefix_cache_;
    MambaChunkAllocator* mamba_allocator_;
    MambaHostAllocator* mamba_host_allocator_;
    MambaEvictionManager mamba_eviction_manager_;
    std::int32_t mamba_cache_chunk_size_;
    std::unordered_set<TreeNode*> mamba_host_nodes_;
    std::unordered_map<TreeNode*, std::unique_ptr<MambaSlot>> pending_mamba_host_writebacks_;
    std::unordered_set<TreeNode*> mamba_host_writeback_done_nodes_;

    // `paged_cache_history_alignment_tokens_ == 0` means adjunct disabled; tables still work.
    std::map<std::string, std::unique_ptr<PagedCacheGroupAllocator>> paged_cache_allocators_;
    std::map<std::string, std::unique_ptr<PagedCacheGroupAllocator>> paged_cache_host_allocators_;
    std::unordered_map<std::string, std::map<std::string, PagedCacheGroupTable>> request_paged_cache_tables_;
    std::int32_t paged_cache_history_alignment_tokens_{0};
    std::vector<std::string> paged_cache_required_groups_;
    std::unordered_map<std::string, std::int32_t> paged_cache_sliding_window_per_group_;
    // Subset of `paged_cache_required_groups_` partitioned by family.
    std::vector<std::string> paged_cache_history_groups_;
    std::vector<std::string> paged_cache_state_groups_;
    std::vector<std::string> paged_cache_continuation_state_groups_;
    // Fast hot-path lookup mirrors of the above (filled in EnablePagedCacheAdjunct).
    std::unordered_set<std::string> paged_cache_history_group_set_;
    std::unordered_set<std::string> paged_cache_state_group_set_;
    std::unordered_set<std::string> paged_cache_continuation_state_group_set_;

    std::unordered_set<TreeNode*> paged_cache_snapshot_nodes_;
    std::unordered_set<TreeNode*> paged_cache_host_snapshot_nodes_;
    std::unordered_set<TreeNode*> paged_cache_pending_host_snapshot_nodes_;
};

}  // namespace tokenspeed
