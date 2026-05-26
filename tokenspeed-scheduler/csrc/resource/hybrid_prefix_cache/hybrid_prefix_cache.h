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

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "resource/allocator/owned_pages.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache_types.h"
#include "resource/hybrid_prefix_cache/mamba_eviction_manager.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "scheduler/operations/cache.h"
#include "resource/types.h"

namespace tokenspeed {

class MambaChunkAllocator;
class MambaHostAllocator;
class LocalKVAllocator;
class LocalMambaAllocator;
class MambaSlot;
class PageAllocator;
class ForwardOperationBase;

class HybridPrefixCache {
public:
    // `mamba_allocator` may be null; paged-cache adjunct is enabled separately.
    HybridPrefixCache(KVPrefixCache& prefix_cache, PageAllocator& device_allocator, MambaChunkAllocator* allocator,
                      std::int32_t mamba_cache_chunk_size, MambaHostAllocator* mamba_host_allocator = nullptr);
    ~HybridPrefixCache();

    RecoveryPlan MatchPrefix(const token_vec_t& token_ids, MatchIntent intent = MatchIntent::PrefixReuse);
    RecoveryPlan MatchPrefix(const std::vector<std::span<const std::int32_t>>& token_pages,
                             MatchIntent intent = MatchIntent::PrefixReuse);
    [[nodiscard]] AdmissionVerdict Admit(const AdmissionRequest& request,
                                         std::map<std::string, std::int32_t>& simulated_free);
    StepCommitResult StepCommit(StepCommitRequest request);
    [[nodiscard]] CacheStatsSnapshot Stats(const StatsRequest& request = {}) const;

    struct RawHostStorageHashSeed {
        std::int32_t host_matched_pages{0};
        std::string prior_hash_seed{};
    };

    // Cold-path storage rolling-hash seed lookup. This intentionally returns
    // only raw host KV match depth plus the terminal host page-hash seed; it
    // does not apply Mamba/paged-cache recovery augmentation.
    RawHostStorageHashSeed LookupRawHostStorageHashSeed(const std::vector<std::span<const std::int32_t>>& token_pages);

    cache_op_id AllocateCacheOpId();
    void SetKvEventSink(KvEventSink sink);

    // Startup-only scheduler configuration facade. Copies and validates group
    // configs, registers concrete paged-cache group allocators, and optionally
    // enables prefix-cache adjunct state for the required groups.
    void ConfigurePagedCacheAdjunct(std::span<const PagedCacheGroupConfig> group_configs,
                                    std::optional<std::span<const std::string>> required_groups);

    // Unified paged-cache lifecycle surface used by the Scheduler. All methods
    // below are no-ops when no paged-cache groups are registered.

    // Initial per-group simulated_free budget mirroring live allocator state.
    std::map<std::string, std::int32_t> InitialSimulatedFree() const;

    // Finish scheduler lifecycle for a request: release request-local
    // paged-cache tables/state only. Shared TreeNode attachments remain owned by
    // prefix-cache refcount/LRU/eviction paths.
    void FinishRequest(const std::string& request_id);

    void OnKVDeviceDemote(TreeNode* node);
    void OnMambaHostWriteBackDone(TreeNode* last_node);
    void OnMambaHostWriteBackDone(const std::vector<TreeNode*>& nodes);
    void DemoteIdleMambaDeviceCopiesPresentOnHost();

private:
    friend class HybridPrefixCacheTestPeer;

    struct DecodeFromRetractedRecovery {
        bool ok{true};
        TreeNode* protected_source_node{nullptr};
    };

    struct CacheAdmissionRequest {
        AdmissionRequestKind kind{AdmissionRequestKind::kDecodeChunk};
        std::string request_id{};
        std::int32_t device_pages_needed{0};
        std::int32_t host_pages_needed{0};
        std::int32_t tokens_this_round{0};
        std::int32_t first_raw_position_of_op{0};
        std::int32_t target_raw_tokens_exclusive{0};
        const MatchResult* match_result{nullptr};
        TreeNode* mamba_recovery_node{nullptr};
        bool refresh_mamba_checkpoint{false};
    };

    struct CacheAdmissionResult {
        bool admitted{false};
        std::optional<std::int32_t> mamba_branching_seqlen{};
        std::optional<std::int32_t> mamba_cow_src_index{};
        std::vector<TransferPair> cache_transfer_pairs{};
    };

    enum class RequestLocalKVKind {
        kPrefillFirstChunk,
        kPrefillChunk,
    };

    struct RequestLocalKVRequest {
        RequestLocalKVKind kind{RequestLocalKVKind::kPrefillChunk};
        LocalKVAllocator* local_kv_allocator{nullptr};
        std::int32_t tokens_this_round{0};
        std::int32_t decode_input_tokens{0};
    };

    struct RequestLocalKVResult {
        std::unique_ptr<LocalKVAllocator> local_kv_allocator{};
    };

    enum class RequestLocalMambaKind {
        kPrefillFirstChunk,
        kDecodeFromRetracted,
        kNextCheckpoint,
    };

    struct RequestLocalMambaRequest {
        RequestLocalMambaKind kind{RequestLocalMambaKind::kNextCheckpoint};
        LocalMambaAllocator* local_mamba_allocator{nullptr};
        std::optional<std::int32_t> checkpoint_raw_position{};
    };

    struct RequestLocalMambaResult {
        std::unique_ptr<LocalMambaAllocator> local_mamba_allocator{};
    };

    enum class CachePublicationKind {
        kForwardChunk,
        kFinishChunk,
        kRetractDeviceInsertPageCount,
        kRetractChunk,
    };

    struct CachePublicationRequest {
        CachePublicationKind kind{CachePublicationKind::kForwardChunk};
        const std::vector<std::span<const std::int32_t>>* full_paged_tokens{nullptr};
        std::unique_ptr<DeviceNodeRef>* device_node_ref{nullptr};
        const TreeNode* current_device_node{nullptr};
        LocalKVAllocator* local_kv_allocator{nullptr};
        LocalMambaAllocator* local_mamba_allocator{nullptr};
        const std::vector<std::string>* page_hashes{nullptr};
        OwnedPages pages_to_insert{};
    };

    struct CachePublicationResult {
        MatchResult match_result{};
        std::int32_t device_insert_page_count{0};
    };

    enum class CacheMaterializationKind {
        kPrefillHostPrefixOnDevice,
        kDecodeRecoveryHostPrefixOnDevice,
        kFinishWritebackHostPages,
        kRetractWritebackHostPages,
    };

    struct CacheMaterializationRequest {
        CacheMaterializationKind kind{CacheMaterializationKind::kPrefillHostPrefixOnDevice};
        const MatchResult* match_result{nullptr};
        const std::vector<TreeNode*>* write_diff{nullptr};
    };

    struct CacheMaterializationResult {
        bool ok{true};
    };

    struct CacheOpPrepareRequest {
        WorkerCompatibilityCommitKind kind{WorkerCompatibilityCommitKind::kDecodeChunk};
        TreeNode* terminal{nullptr};
        std::int32_t first_raw_position_of_op{0};
        std::int32_t target_raw_tokens_exclusive{0};
        const MatchResult* match_result{nullptr};
        const LocalMambaAllocator* local_mamba_allocator{nullptr};
        MatchResult::PagedCache paged_cache_hit{};
        bool commit_prior_chunk{false};
    };

    [[nodiscard]] CacheAdmissionResult Admit(const CacheAdmissionRequest& request,
                                             std::map<std::string, std::int32_t>& simulated_free);
    DecodeFromRetractedRecovery PrepareDecodeFromRetractedRecovery(MatchResult& match_result) const;
    [[nodiscard]] RequestLocalKVResult PrepareRequestLocalKV(const RequestLocalKVRequest& request) const;
    [[nodiscard]] RequestLocalMambaResult PrepareRequestLocalMamba(const RequestLocalMambaRequest& request) const;
    CachePublicationResult Publish(CachePublicationRequest request);
    [[nodiscard]] CacheMaterializationResult Materialize(const CacheMaterializationRequest& request);
    void PublishRetractMambaState(TreeNode* terminal, std::unique_ptr<LocalMambaAllocator>& local_mamba_allocator);
    void PrepareForwardOp(ForwardOperationBase& op_base, const CacheOpPrepareRequest& request);

    bool HasMambaAdjunct() const { return mamba_allocator_ != nullptr; }
    bool HasPagedCacheAdjunct() const { return paged_cache_history_alignment_tokens_ > 0; }
    RecoveryPlan BuildRecoveryPlan(MatchResult raw_match, MatchIntent intent) const;

    // Takes ownership. Duplicate group_id throws std::invalid_argument.
    void RegisterPagedCacheGroup(std::unique_ptr<PagedCacheGroupAllocator> allocator);

    // History alignment is the LCM of RawTokensPerPage() over the History-family
    // groups; state groups only need the trailing window. Sliding groups must
    // have a window entry; full-history groups must not.
    void EnablePagedCacheAdjunct(std::vector<std::string> required_groups,
                                 std::unordered_map<std::string, std::int32_t> sliding_window_per_group,
                                 StateRestorePolicy policy = StateRestorePolicy::kSnapshotRequired);

    void PopulateMambaMatchCompatibilityFields(ForwardOperationBase& op_base, const MatchResult& match_result) const;
    void PopulateMambaRecoveryCompatibilityFields(ForwardOperationBase& op_base, const MatchResult& match_result) const;
    void PopulateMambaRequestLocalCompatibilityFields(ForwardOperationBase& op_base,
                                                      const LocalMambaAllocator* local_mamba_allocator) const;
    bool EnsureMambaCapacityByEvict(std::int32_t num_slots, TreeNode* protected_node = nullptr);
    bool EnsureMambaHostCapacityByEvict(std::int32_t num_slots, TreeNode* protected_node = nullptr);
    std::int32_t AlignMambaCacheSeqlen(std::int32_t seqlen) const;
    void InsertMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot);
    TreeNode* FindLastMambaNode(TreeNode* from) const;
    TreeNode* FindLastMambaHostNode(TreeNode* from) const;
    std::vector<TransferPair> PrepareMambaHostWriteBack(const std::vector<TreeNode*>& nodes);
    std::vector<TransferPair> PrepareMambaDeviceLoadBack(const std::vector<TreeNode*>& nodes);
    // Callback from KV prefix-cache eviction.
    void OnKVEvict(TreeNode* node);
    void OnKVHostEvict(TreeNode* node);
    // Publish request-local Mamba state for finish after the caller has inserted
    // new terminal KV pages. The caller owns the "new KV pages were inserted"
    // gate so finish publication remains coupled to successful KV insertion.
    void PublishFinishMambaState(const std::vector<std::span<const std::int32_t>>& full_paged_tokens,
                                 LocalMambaAllocator* local_mamba_allocator);

    // Fill op.paged_cache_pages / op.paged_cache_page_base_offsets from the tables.
    void PopulateOp(ForwardOperationBase& op_base) const;
    std::unique_ptr<LocalMambaAllocator> allocateRequestLocalMambaState(
        std::optional<std::int32_t> checkpoint_raw_position = {}) const;

    // Per-family classification of admission failure; drives state-only vs
    // full prune strategy.
    enum class AdmissionFailureKind { kNone, kHistoryStarved, kStateStarved, kBothStarved };

    struct PagedCacheGroupAdmission {
        bool ok{true};
        std::map<std::string, std::int32_t> releasable_owned_pages{};
        std::map<std::string, std::int32_t> new_pages_needed{};
        std::map<std::string, std::int32_t> shortfall_pages{};
    };

    struct PagedCacheAdmissionContext {
        bool fresh_table_view{false};
        std::map<std::string, std::int32_t> owned_release_credit{};
    };

    // Classify which family caused `admission.ok == false`.
    AdmissionFailureKind ClassifyAdmissionFailure(const PagedCacheGroupAdmission& admission) const;

    // Drop only state-family groups from `node`'s snapshot; history portion
    // remains and the node stays registered. Returns true iff state groups removed.
    bool DetachStateSnapshotFromNode(TreeNode* node);

    // Ensure tables exist and cover [first_raw_position_of_op, target_raw_tokens_exclusive).
    // Borrowed prefix is imported BEFORE any fresh allocation on a fresh table.
    void AcquireForRequest(const std::string& request_id, std::int32_t first_raw_position_of_op,
                           std::int32_t target_raw_tokens_exclusive,
                           const MatchResult::PagedCache& paged_cache_hit = {});

    // Owned pages return to the pool via OwnedPages RAII; borrowed ids are dropped.
    void ReleaseRequest(const std::string& request_id);

    // Run paged-cache admission against `simulated_free`; prunes evictable
    // snapshots on group-pool pressure, then applies the debit on success.
    bool AdmitChunk(const std::string& request_id, std::int32_t first_raw_position_of_op,
                    std::int32_t target_raw_tokens_exclusive, std::map<std::string, std::int32_t>& simulated_free,
                    const MatchResult::PagedCache& paged_cache_hit = {});

    // Retract-decode variant: admission uses a fresh-table view and credits
    // pages owned by the stale table before it is released.
    bool AdmitChunkFromRetracted(const std::string& request_id, std::int32_t target_raw_tokens_exclusive,
                                 std::map<std::string, std::int32_t>& simulated_free,
                                 const MatchResult::PagedCache& paged_cache_hit);

    // Commit newly-written full LCM segments into TreeNode PagedCacheSnapshots.
    void CommitChunk(const std::string& request_id, TreeNode* terminal);

    // Attach a snapshot to `node`, computing `complete_families` from which
    // required-per-family group ids are present and registering the node in
    // `paged_cache_snapshot_nodes_`. Returns false when either argument is
    // null (defensive no-op). Accepts partial snapshots; the per-policy
    // "snapshot must be full" invariant is enforced upstream by CommitChunk.
    bool AttachPagedCacheSnapshotToNode(TreeNode* node, std::unique_ptr<PagedCacheSnapshot> snapshot);

    // Drops `node` from the membership set, then detaches and returns the snapshot.
    std::unique_ptr<PagedCacheSnapshot> DetachPagedCacheSnapshotFromNode(TreeNode* node);

    void augmentMatch(MatchResult& match) const;
    void augmentMatchPagedCache(MatchResult& match) const;
    bool publishRequestMambaState(TreeNode* terminal, LocalMambaAllocator* local_mamba_allocator);

    // Detach oldest evictable snapshot to free pool pages. State-only path is
    // used only on kStateStarved; history/both go to full cascade.
    bool tryPrunePagedCacheSnapshot(AdmissionFailureKind kind);

    bool admitPagedCacheChunk(const std::string& request_id, std::int32_t first_raw_position_of_op,
                              std::int32_t target_raw_tokens_exclusive,
                              std::map<std::string, std::int32_t>& simulated_free,
                              const MatchResult::PagedCache& paged_cache_hit,
                              const PagedCacheAdmissionContext& context);
    void acquireAndPopulateOp(ForwardOperationBase& op_base, std::int32_t first_raw_position_of_op,
                              std::int32_t target_raw_tokens_exclusive, const MatchResult::PagedCache& paged_cache_hit);

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
    PageAllocator& device_allocator_;
    MambaChunkAllocator* mamba_allocator_;
    MambaHostAllocator* mamba_host_allocator_;
    MambaEvictionManager mamba_eviction_manager_;
    std::int32_t mamba_cache_chunk_size_;
    std::unordered_set<TreeNode*> mamba_host_nodes_;
    std::unordered_map<TreeNode*, std::unique_ptr<MambaSlot>> pending_mamba_host_writebacks_;
    std::unordered_set<TreeNode*> mamba_host_writeback_done_nodes_;
    bool has_facade_kv_event_sink_{false};

    // `paged_cache_history_alignment_tokens_ == 0` means adjunct disabled; tables still work.
    std::map<std::string, std::unique_ptr<PagedCacheGroupAllocator>> paged_cache_allocators_;
    std::unordered_map<std::string, std::map<std::string, PagedCacheGroupTable>> request_paged_cache_tables_;
    std::int32_t paged_cache_history_alignment_tokens_{0};
    std::vector<std::string> paged_cache_required_groups_;
    std::unordered_map<std::string, std::int32_t> paged_cache_sliding_window_per_group_;
    // Subset of `paged_cache_required_groups_` partitioned by family.
    std::vector<std::string> paged_cache_history_groups_;
    std::vector<std::string> paged_cache_state_groups_;
    // Fast hot-path lookup mirrors of the above (filled in EnablePagedCacheAdjunct).
    std::unordered_set<std::string> paged_cache_history_group_set_;
    std::unordered_set<std::string> paged_cache_state_group_set_;
    StateRestorePolicy paged_cache_state_policy_{StateRestorePolicy::kSnapshotRequired};

    // TODO(snapshot-lru-perf): O(N log N) per prune; swap in LRU index if profiling shows it matters.
    std::unordered_set<TreeNode*> paged_cache_snapshot_nodes_;
};

}  // namespace tokenspeed
