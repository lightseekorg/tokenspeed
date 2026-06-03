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

#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/allocator/kv_allocator.h"
#include "resource/allocator/local_mamba_allocator.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/mamba_host_allocator.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/page_container.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/radix_tree/node_range.h"
#include "resource/radix_tree/paged_cache_snapshot.h"
#include "resource/radix_tree/radix_tree.h"
#include "resource/radix_tree/tree_node.h"
#include "scheduler/operations/forward.h"
#include "utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace tokenspeed {

RequestLocalCacheState::RequestLocalCacheState(std::unique_ptr<LocalKVAllocator> local_kv_allocator,
                                               std::unique_ptr<LocalMambaAllocator> adjunct_state)
    : local_kv_allocator_{std::move(local_kv_allocator)}, adjunct_state_{std::move(adjunct_state)} {
    if (local_kv_allocator_ == nullptr) {
        throw std::invalid_argument("RequestLocalCacheState requires local KV state");
    }
}

RequestLocalCacheState::~RequestLocalCacheState() = default;
RequestLocalCacheState::RequestLocalCacheState(RequestLocalCacheState&&) noexcept = default;
RequestLocalCacheState& RequestLocalCacheState::operator=(RequestLocalCacheState&&) noexcept = default;

std::vector<std::int32_t> RequestLocalCacheState::OccupiedPages(const TreeNode* device_node) const {
    return PageContainer{device_node, local_kv_allocator_.get()}.Pages();
}

void RequestLocalCacheState::AcquireKV(std::int32_t tokens) {
    local_kv_allocator_->Acquire(tokens);
}

OwnedPages RequestLocalCacheState::TakeFullKVPages() {
    return local_kv_allocator_->TakeFullPages();
}

OwnedPages RequestLocalCacheState::TakeFirstKVPages(std::int32_t n) {
    return local_kv_allocator_->TakeFirst(n);
}

std::vector<std::int32_t> RequestLocalCacheState::LocalKVPages() const {
    return local_kv_allocator_->Pages();
}

std::int32_t RequestLocalCacheState::TailPageAvailableTokens() const {
    return local_kv_allocator_->TailPageAvailableTokens();
}

LocalMambaAllocator* RequestLocalCacheState::AdjunctState() {
    return adjunct_state_.get();
}

const LocalMambaAllocator* RequestLocalCacheState::AdjunctState() const {
    return adjunct_state_.get();
}

void RequestLocalCacheState::SetAdjunctState(std::unique_ptr<LocalMambaAllocator> adjunct_state) {
    adjunct_state_ = std::move(adjunct_state);
}

void RequestLocalCacheState::ResetAdjunctState() {
    adjunct_state_.reset();
}

HybridPrefixCache::HybridPrefixCache(KVPrefixCache& kv_prefix_cache, PageAllocator& device_allocator,
                                     MambaChunkAllocator* mamba_allocator, std::int32_t mamba_cache_chunk_size,
                                     MambaHostAllocator* mamba_host_allocator)
    : kv_prefix_cache_{kv_prefix_cache},
      device_allocator_{device_allocator},
      mamba_allocator_{mamba_allocator},
      mamba_host_allocator_{mamba_host_allocator},
      mamba_eviction_manager_{mamba_allocator},
      mamba_cache_chunk_size_{mamba_cache_chunk_size} {
    if (HasMambaAdjunct() || mamba_host_allocator_ != nullptr) {
        EnsureKvEvictionCallbacksInstalled();
    }
}

HybridPrefixCache::~HybridPrefixCache() {
    SetKvEventSink({});
    if (has_kv_eviction_callbacks_) {
        kv_prefix_cache_.GetDeviceManager().SetEvictionCallback({});
        kv_prefix_cache_.GetHostManager().SetEvictionCallback({});
        has_kv_eviction_callbacks_ = false;
    }
}

RecoveryPlan HybridPrefixCache::MatchPrefix(const std::vector<std::span<const std::int32_t>>& token_pages,
                                            MatchIntent intent) {
    return BuildRecoveryPlan(kv_prefix_cache_.Match(token_pages, intent), intent);
}

RecoveryPlan HybridPrefixCache::BuildRecoveryPlan(MatchResult raw_match, MatchIntent intent) const {
    RecoveryPlan plan{};
    if (intent == MatchIntent::StateRecovery) {
        MatchResult raw_recovery_match = raw_match;
        plan.compat_match = std::move(raw_match);
        augmentMatch(plan.compat_match);
        augmentMatchPagedCache(plan.compat_match);

        MatchResult& recovery_match = HasPagedCacheAdjunct() ? plan.compat_match : raw_recovery_match;
        const DecodeFromRetractedRecovery recovery = PrepareDecodeFromRetractedRecovery(recovery_match);
        plan.recovery_state_available = recovery.ok;
        plan.protected_recovery_node = recovery.protected_source_node;
        plan.compat_match.mamba_cow_src_index = recovery_match.mamba_cow_src_index;
        plan.compat_match.mamba_host_src_index = recovery_match.mamba_host_src_index;
        return plan;
    }

    plan.compat_match = std::move(raw_match);
    augmentMatch(plan.compat_match);
    augmentMatchPagedCache(plan.compat_match);
    return plan;
}

HybridPrefixCache::RawHostStorageHashSeed HybridPrefixCache::LookupRawHostStorageHashSeed(
    const std::vector<std::span<const std::int32_t>>& token_pages) {
    MatchResult match = kv_prefix_cache_.Match(token_pages);
    const std::int32_t host_matched_pages = match.host.DepthInPage();
    const auto& page_hashes = match.host.last_node->PageHashes();
    return RawHostStorageHashSeed{
        .host_matched_pages = host_matched_pages,
        .prior_hash_seed = page_hashes.empty() ? std::string{} : page_hashes.back(),
    };
}

cache_op_id HybridPrefixCache::AllocateCacheOpId() {
    return kv_prefix_cache_.AllocateCacheOpId();
}

void HybridPrefixCache::SetKvEventSink(KvEventSink sink) {
    if (!sink) {
        if (has_facade_kv_event_sink_) {
            kv_prefix_cache_.SetKvEventSink({});
            has_facade_kv_event_sink_ = false;
        }
        return;
    }

    kv_prefix_cache_.SetKvEventSink(std::move(sink));
    has_facade_kv_event_sink_ = true;
}

void HybridPrefixCache::EnsureKvEvictionCallbacksInstalled() {
    if (has_kv_eviction_callbacks_) return;
    kv_prefix_cache_.GetDeviceManager().SetEvictionCallback([this](TreeNode* node) { OnKVEvict(node); });
    kv_prefix_cache_.GetHostManager().SetEvictionCallback([this](TreeNode* node) { OnKVHostEvict(node); });
    has_kv_eviction_callbacks_ = true;
}

cache::publish::DevicePrefix::Result HybridPrefixCache::Apply(const cache::publish::DevicePrefix& op) {
    cache::publish::DevicePrefix::Result result{};
    if (op.device_node_ref == nullptr) {
        throw std::invalid_argument("HybridPrefixCache::Apply(DevicePrefix) requires device_node_ref");
    }
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(op.device_node_ref->Node());
    const std::int32_t new_page_count =
        static_cast<std::int32_t>(op.full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());
    const std::int32_t page_size = kv_prefix_cache_.PageSize();
    const std::int32_t last_inserted_len = static_cast<std::int32_t>(op.full_paged_tokens.size()) * page_size;
    LocalMambaAllocator* adjunct_state = op.local_cache.AdjunctState();
    auto should_publish_mamba_checkpoint = [&]() {
        if (adjunct_state == nullptr || !adjunct_state->HasCheckpoint()) {
            return false;
        }
        const std::int32_t checkpoint_position = adjunct_state->CheckpointPosition();
        if (checkpoint_position < 0 || checkpoint_position == last_inserted_len) {
            return true;
        }
        if (!op.chunk_begin.has_value() || page_size <= 0) {
            return false;
        }
        const std::int32_t chunk_begin = *op.chunk_begin;
        if (last_inserted_len <= chunk_begin || last_inserted_len >= checkpoint_position) {
            return false;
        }
        const std::int32_t track_len = last_inserted_len - chunk_begin;
        return AlignMambaCacheSeqlen(track_len) == track_len;
    };
    const bool publish_mamba_checkpoint = should_publish_mamba_checkpoint();
    if (new_page_count <= 0) {
        if (adjunct_state != nullptr && adjunct_state->HasCheckpoint()) {
            (void)adjunct_state->DetachCheckpoint();
        }
        return result;
    }

    OwnedPages pages_to_insert = op.local_cache.TakeFirstKVPages(new_page_count);
    auto insert_result =
        kv_prefix_cache_.Insert<ResourceType::Device>(op.full_paged_tokens, prefix_pages, std::move(pages_to_insert));

    if (adjunct_state != nullptr && adjunct_state->HasCheckpoint()) {
        std::unique_ptr<MambaSlot> checkpoint = adjunct_state->DetachCheckpoint();
        if (publish_mamba_checkpoint) {
            InsertMamba(insert_result.last_node, std::move(checkpoint));
        }
    }
    op.device_node_ref = std::make_unique<DeviceNodeRef>(insert_result.last_node);
    result.device_insert_page_count = new_page_count;
    return result;
}

cache::publish::FinishedRequest::Result HybridPrefixCache::Apply(const cache::publish::FinishedRequest& op) {
    cache::publish::FinishedRequest::Result result{};
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(&op.current_device_node);
    const std::int32_t alloc_count =
        static_cast<std::int32_t>(op.full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());

    if (alloc_count > 0) {
        OwnedPages alloc_pages = op.local_cache.TakeFirstKVPages(alloc_count);
        kv_prefix_cache_.Insert<ResourceType::Device>(op.full_paged_tokens, prefix_pages, std::move(alloc_pages),
                                                      op.page_hashes);
        PublishFinishMambaState(op.full_paged_tokens, op.local_cache);
    }

    result.device_insert_page_count = std::max(0, alloc_count);
    result.match_result = kv_prefix_cache_.Match(op.full_paged_tokens);
    return result;
}

cache::publish::RetractPrefixCommit::Result HybridPrefixCache::Apply(const cache::publish::RetractPrefixCommit& op) {
    cache::publish::RetractPrefixCommit::Result result{};
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(&op.current_device_node);
    const std::int32_t alloc_count =
        static_cast<std::int32_t>(op.full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());
    if (alloc_count < 0) {
        throw std::logic_error(
            "HybridPrefixCache::Apply(RetractPrefixCommit): current device prefix exceeds available full token pages");
    }
    OwnedPages pages_to_insert = op.local_cache.TakeFirstKVPages(alloc_count);
    if (pages_to_insert.Size() != alloc_count) {
        throw std::logic_error("HybridPrefixCache::Apply(RetractPrefixCommit): request-local page count mismatch");
    }

    kv_prefix_cache_.Insert<ResourceType::Device>(op.full_paged_tokens, prefix_pages, std::move(pages_to_insert));
    result.device_insert_page_count = alloc_count;
    result.match_result = kv_prefix_cache_.Match(op.full_paged_tokens, MatchIntent::StateRecovery);
    return result;
}

cache::materialize::PrefixOnDevice::Result HybridPrefixCache::Apply(const cache::materialize::PrefixOnDevice& op) {
    cache::materialize::PrefixOnDevice::Result result{};
    const std::vector<TreeNode*> nodes = op.compat_match.NodesWithout<ResourceType::Device>();
    if (op.require_all_pages) {
        result.ok = kv_prefix_cache_.AllocateResourceOfType<ResourceType::Device>(nodes);
        return result;
    }
    (void)kv_prefix_cache_.AllocateResourceOfType<ResourceType::Device>(nodes);
    return result;
}

cache::materialize::HostWritebackPages::Result HybridPrefixCache::Apply(
    const cache::materialize::HostWritebackPages& op) {
    cache::materialize::HostWritebackPages::Result result{};
    if (op.ensure_capacity_before_allocate) {
        std::int32_t host_pages_num = 0;
        for (TreeNode* node : op.write_diff) {
            host_pages_num += node->Device().NumPages();
        }
        if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Host>(host_pages_num)) {
            result.ok = false;
            return result;
        }
    }
    result.ok = kv_prefix_cache_.AllocateResourceOfType<ResourceType::Host>(op.write_diff);
    if (!result.ok) return result;
    result.cache_transfer_pairs = PrepareMambaHostWriteBack(op.write_diff);
    for (const TransferPair& transfer : result.cache_transfer_pairs) {
        if (transfer.kind != CacheKind::kMamba) continue;
        for (TreeNode* node : op.write_diff) {
            if (node != nullptr && node->HasMamba() && node->MambaSlotIndex() == transfer.src) {
                result.pending_state.adjunct_nodes.push_back(node);
                break;
            }
        }
    }
    return result;
}

void HybridPrefixCache::OnKVEvict(TreeNode* node) {
    if (node == nullptr) return;
    if (mamba_allocator_ != nullptr && node->HasMamba()) {
        mamba_eviction_manager_.UntrackNode(node);
        node->DetachMamba();
        if (node->Parent() != nullptr) {
            mamba_eviction_manager_.UpdateLeaf(node->Parent());
        }
    }
    // Passive paged-cache detach on KV LRU drop: returns OwnedPages via RAII;
    // the chain scan sees the gap because `HasPagedCacheSnapshot()` is false.
    // Route through DetachPagedCacheSnapshotFromNode to keep membership set in sync.
    if (node->HasPagedCacheSnapshot()) {
        DetachPagedCacheSnapshotFromNode(node);
    }
}

void HybridPrefixCache::OnKVHostEvict(TreeNode* node) {
    if (node == nullptr || mamba_host_allocator_ == nullptr) return;
    pending_mamba_host_writebacks_.erase(node);
    if (node->HasMambaOnHost()) {
        node->DetachMambaHost();
        mamba_host_nodes_.erase(node);
        mamba_host_writeback_done_nodes_.erase(node);
    }
}

void HybridPrefixCache::DemoteIdleMambaDeviceCopiesPresentOnHost() {
    if (mamba_allocator_ == nullptr || mamba_host_allocator_ == nullptr) return;

    std::int32_t demoted = 0;
    std::vector<TreeNode*> nodes(mamba_host_writeback_done_nodes_.begin(), mamba_host_writeback_done_nodes_.end());
    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->HasMambaOnHost()) {
            mamba_host_writeback_done_nodes_.erase(node);
            continue;
        }
        if (!node->HasMamba()) {
            mamba_host_writeback_done_nodes_.erase(node);
            continue;
        }
        if (node->OnDevice() && node->Device().RefCount() != 0) {
            continue;
        }
        OnKVDeviceDemote(node);
        mamba_host_writeback_done_nodes_.erase(node);
        ++demoted;
    }
    if (demoted > 0) {
        spdlog::debug("[HybridPrefixCache][mamba_l2] demoted device copies after host writeback count={}", demoted);
    }
}

void HybridPrefixCache::OnMambaHostWriteBackDone(TreeNode* last_node) {
    if (last_node == nullptr) return;
    std::vector<TreeNode*> nodes;
    for (TreeNode* node : LeafToRoot(last_node)) {
        if (node == nullptr || !node->OnHost()) break;
        nodes.push_back(node);
    }
    OnMambaHostWriteBackDone(nodes);
}

void HybridPrefixCache::OnMambaHostWriteBackDone(const std::vector<TreeNode*>& nodes) {
    if (mamba_allocator_ == nullptr || mamba_host_allocator_ == nullptr) return;

    std::int32_t attached = 0;
    std::int32_t completed = 0;
    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->OnHost()) continue;
        auto pending = pending_mamba_host_writebacks_.find(node);
        if (pending != pending_mamba_host_writebacks_.end()) {
            node->AttachMambaHost(std::move(pending->second));
            pending_mamba_host_writebacks_.erase(pending);
            mamba_host_nodes_.insert(node);
            ++attached;
        }
        if (node->HasMambaOnHost()) {
            mamba_host_writeback_done_nodes_.insert(node);
            ++completed;
        }
    }
    if (attached > 0 || completed > 0) {
        spdlog::debug("[HybridPrefixCache][mamba_l2] host writeback done attach_count={} completed_nodes={}", attached,
                      completed);
    }
    DemoteIdleMambaDeviceCopiesPresentOnHost();
}

void HybridPrefixCache::OnKVDeviceDemote(TreeNode* node) {
    if (node == nullptr || mamba_allocator_ == nullptr) return;
    if (node->HasMamba() && node->HasMambaOnHost()) {
        mamba_eviction_manager_.UntrackNode(node);
        node->DetachMamba();
        if (node->Parent() != nullptr) {
            mamba_eviction_manager_.UpdateLeaf(node->Parent());
        }
    }
}

void HybridPrefixCache::CompleteHostWriteBack(const PendingHostWritebackState& pending_state, TreeNode* device_node) {
    OnMambaHostWriteBackDone(pending_state.adjunct_nodes);
    if (device_node != nullptr) {
        kv_prefix_cache_.ReleaseDeviceResourcesPresentOnHost(device_node,
                                                             [this](TreeNode* node) { OnKVDeviceDemote(node); });
    }
    DemoteIdleMambaDeviceCopiesPresentOnHost();
}

std::size_t HybridPrefixCache::AvailableDevicePages() const {
    return static_cast<std::size_t>(device_allocator_.AvailablePages());
}

std::vector<std::string> HybridPrefixCache::PagedCacheGroupIds() const {
    std::vector<std::string> ids;
    ids.reserve(paged_cache_allocators_.size());
    for (const auto& [gid, _] : paged_cache_allocators_) {
        ids.push_back(gid);
    }
    return ids;
}

std::int32_t HybridPrefixCache::PagedCacheGroupTotalPages(std::string_view group_id) const {
    auto alloc_it = std::find_if(paged_cache_allocators_.begin(), paged_cache_allocators_.end(),
                                 [group_id](const auto& entry) { return entry.first == group_id; });
    if (alloc_it == paged_cache_allocators_.end() || alloc_it->second == nullptr) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheGroupTotalPages: group_id not configured");
    }
    return alloc_it->second->TotalPages();
}

std::int32_t HybridPrefixCache::PagedCacheGroupAvailablePages(std::string_view group_id) const {
    auto alloc_it = std::find_if(paged_cache_allocators_.begin(), paged_cache_allocators_.end(),
                                 [group_id](const auto& entry) { return entry.first == group_id; });
    if (alloc_it == paged_cache_allocators_.end() || alloc_it->second == nullptr) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheGroupAvailablePages: group_id not configured");
    }
    return alloc_it->second->AvailablePages();
}

std::int64_t HybridPrefixCache::PagedCacheGroupFailedAllocCount(std::string_view group_id) const {
    auto alloc_it = std::find_if(paged_cache_allocators_.begin(), paged_cache_allocators_.end(),
                                 [group_id](const auto& entry) { return entry.first == group_id; });
    if (alloc_it == paged_cache_allocators_.end() || alloc_it->second == nullptr) {
        throw std::out_of_range("HybridPrefixCache::PagedCacheGroupFailedAllocCount: group_id not configured");
    }
    return alloc_it->second->FailedAllocCount();
}

std::vector<std::int32_t> HybridPrefixCache::GetRequestPagedCachePageIds(std::string_view request_id,
                                                                         std::string_view group_id) const {
    (void)PagedCacheGroupTotalPages(group_id);
    auto req_it = std::find_if(request_paged_cache_tables_.begin(), request_paged_cache_tables_.end(),
                               [request_id](const auto& entry) { return entry.first == request_id; });
    if (req_it == request_paged_cache_tables_.end()) return {};
    auto group_it = std::find_if(req_it->second.begin(), req_it->second.end(),
                                 [group_id](const auto& entry) { return entry.first == group_id; });
    if (group_it == req_it->second.end()) return {};
    return group_it->second.PageIds();
}

std::int32_t HybridPrefixCache::GetRequestPagedCacheBaseLogicalPage(std::string_view request_id,
                                                                    std::string_view group_id) const {
    (void)PagedCacheGroupTotalPages(group_id);
    auto req_it = std::find_if(request_paged_cache_tables_.begin(), request_paged_cache_tables_.end(),
                               [request_id](const auto& entry) { return entry.first == request_id; });
    if (req_it == request_paged_cache_tables_.end()) return 0;
    auto group_it = std::find_if(req_it->second.begin(), req_it->second.end(),
                                 [group_id](const auto& entry) { return entry.first == group_id; });
    if (group_it == req_it->second.end()) return 0;
    return group_it->second.BaseLogicalPage();
}

CacheStatsSnapshot HybridPrefixCache::Stats(const StatsRequest& request) const {
    CacheStatsSnapshot snapshot{
        .available_device_pages = AvailableDevicePages(),
    };

    snapshot.paged_cache_group_ids = PagedCacheGroupIds();

    std::vector<std::string> requested_groups = request.paged_cache_group_ids;
    if (requested_groups.empty()) {
        requested_groups = snapshot.paged_cache_group_ids;
    }
    for (const auto& gid : requested_groups) {
        snapshot.paged_cache_total_pages[gid] = PagedCacheGroupTotalPages(gid);
        snapshot.paged_cache_available_pages[gid] = PagedCacheGroupAvailablePages(gid);
        snapshot.paged_cache_failed_alloc_count[gid] = PagedCacheGroupFailedAllocCount(gid);

        if (request.request_id.has_value()) {
            snapshot.request_paged_cache_page_ids[gid] = GetRequestPagedCachePageIds(*request.request_id, gid);
            snapshot.request_paged_cache_base_logical_page[gid] =
                GetRequestPagedCacheBaseLogicalPage(*request.request_id, gid);
        }
    }

    if (request.include_device_memory_diagnostics) {
        snapshot.device_memory_diagnostics = CacheDeviceMemoryDiagnosticsSnapshot{
            .tree_device_pages = kv_prefix_cache_.CollectAllPages<ResourceType::Device>(),
            .free_device_pages = device_allocator_.AvailablePages(),
            .total_device_pages = device_allocator_.TotalPages() - 1,
        };
    }

    return snapshot;
}

cache::worker::CommitPrefillFirstChunkMetadata::Result HybridPrefixCache::Apply(
    const cache::worker::CommitPrefillFirstChunkMetadata& op) {
    PopulateMambaRequestLocalCompatibilityFields(op.op_base, op.local_cache);
    PopulateMambaMatchCompatibilityFields(op.op_base, op.match_result);
    CommitChunk(op.op_base.request_id, &op.tree_prefix_to_commit);
    acquireAndPopulateOp(op.op_base, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                         op.match_result.paged_cache);
    return {};
}

cache::worker::CommitPrefillMetadata::Result HybridPrefixCache::Apply(const cache::worker::CommitPrefillMetadata& op) {
    PopulateMambaRequestLocalCompatibilityFields(op.op_base, op.local_cache);
    CommitChunk(op.op_base.request_id, &op.tree_prefix_to_commit);
    acquireAndPopulateOp(op.op_base, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                         MatchResult::PagedCache{});
    return {};
}

cache::worker::CommitDecodeAfterPrefillMetadata::Result HybridPrefixCache::Apply(
    const cache::worker::CommitDecodeAfterPrefillMetadata& op) {
    PopulateMambaRequestLocalCompatibilityFields(op.op_base, op.local_cache);
    CommitChunk(op.op_base.request_id, &op.tree_prefix_to_commit);
    acquireAndPopulateOp(op.op_base, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                         MatchResult::PagedCache{});
    return {};
}

cache::worker::CommitDecodeMetadata::Result HybridPrefixCache::Apply(const cache::worker::CommitDecodeMetadata& op) {
    PopulateMambaRequestLocalCompatibilityFields(op.op_base, op.local_cache);
    acquireAndPopulateOp(op.op_base, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                         MatchResult::PagedCache{});
    return {};
}

cache::worker::CommitDecodeRecoveryMetadata::Result HybridPrefixCache::Apply(
    const cache::worker::CommitDecodeRecoveryMetadata& op) {
    PopulateMambaRequestLocalCompatibilityFields(op.op_base, op.local_cache);
    PopulateMambaRecoveryCompatibilityFields(op.op_base, op.match_result);
    ReleaseRequest(op.op_base.request_id);
    acquireAndPopulateOp(op.op_base, 0, op.target_raw_tokens_exclusive, op.match_result.paged_cache);
    return {};
}

std::vector<TreeNode*> HybridPrefixCache::MambaDeviceLoadbackNodes(const MatchResult& match_result,
                                                                   TreeNode* preferred_source) const {
    std::vector<TreeNode*> nodes;
    if (mamba_host_allocator_ == nullptr || match_result.mamba_host_src_index < 0 ||
        match_result.mamba_cow_src_index >= 0) {
        return nodes;
    }
    TreeNode* host_mamba_node =
        preferred_source != nullptr ? preferred_source : FindLastMambaHostNode(match_result.host.last_node);
    if (host_mamba_node != nullptr && host_mamba_node->HasMambaOnHost() && !host_mamba_node->HasMamba()) {
        nodes.push_back(host_mamba_node);
    }
    return nodes;
}

cache::admit::PrefillFirstChunk::Result HybridPrefixCache::Apply(const cache::admit::PrefillFirstChunk& op,
                                                                 std::map<std::string, std::int32_t>& simulated_free) {
    cache::admit::PrefillFirstChunk::Result result{};
    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(op.match_result.device.last_node);
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(op.device_pages_needed)) {
        return result;
    }
    std::vector<TreeNode*> loadback_nodes = MambaDeviceLoadbackNodes(op.match_result);
    std::optional<std::int32_t> mamba_branching_seqlen;
    if (HasMambaAdjunct()) {
        if (op.match_result.mamba_branching_seqlen == -1) {
            const std::int32_t aligned = AlignMambaCacheSeqlen(op.tokens_this_round);
            if (aligned > 0) {
                mamba_branching_seqlen = aligned;
            }
        }
        const std::int32_t slots_needed = 2 + static_cast<std::int32_t>(loadback_nodes.size());
        if (!EnsureMambaCapacityByEvict(slots_needed)) {
            return result;
        }
    }
    result.admitted = AdmitChunk(op.request_id, op.first_raw_position_of_op, op.target_raw_tokens_exclusive,
                                 simulated_free, op.match_result.paged_cache);
    if (result.admitted) {
        if (mamba_branching_seqlen.has_value()) {
            op.match_result.mamba_branching_seqlen = *mamba_branching_seqlen;
        }
        result.cache_transfer_pairs = PrepareMambaDeviceLoadBack(loadback_nodes);
        if (!loadback_nodes.empty() && loadback_nodes.front()->HasMamba()) {
            op.match_result.mamba_cow_src_index = loadback_nodes.front()->MambaSlotIndex();
        }
    }
    return result;
}

cache::admit::PrefillChunk::Result HybridPrefixCache::Apply(const cache::admit::PrefillChunk& op,
                                                            std::map<std::string, std::int32_t>& simulated_free) {
    cache::admit::PrefillChunk::Result result{};
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(op.device_pages_needed)) {
        return result;
    }
    if (HasMambaAdjunct() && !EnsureMambaCapacityByEvict(1)) {
        return result;
    }
    result.admitted =
        AdmitChunk(op.request_id, op.first_raw_position_of_op, op.target_raw_tokens_exclusive, simulated_free);
    return result;
}

cache::admit::Decode::Result HybridPrefixCache::Apply(const cache::admit::Decode& op,
                                                      std::map<std::string, std::int32_t>& simulated_free) {
    cache::admit::Decode::Result result{};
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(op.device_pages_needed)) {
        return result;
    }
    if (op.refresh_local_cache_state && HasMambaAdjunct() && !EnsureMambaCapacityByEvict(1)) {
        return result;
    }
    result.admitted =
        AdmitChunk(op.request_id, op.first_raw_position_of_op, op.target_raw_tokens_exclusive, simulated_free);
    return result;
}

cache::admit::DecodeFromRetracted::Result HybridPrefixCache::Apply(
    const cache::admit::DecodeFromRetracted& op, std::map<std::string, std::int32_t>& simulated_free) {
    cache::admit::DecodeFromRetracted::Result result{};
    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(op.match_result.device.last_node);
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(op.device_pages_needed)) {
        return result;
    }
    std::vector<TreeNode*> loadback_nodes = MambaDeviceLoadbackNodes(op.match_result, op.protected_recovery_node);
    if (HasMambaAdjunct()) {
        if (op.protected_recovery_node == nullptr) {
            return result;
        }
        // Recovery COWs the tree-owned Mamba state into fresh request-local
        // working/checkpoint slots. Protect the source node only for this
        // allocation; retracted Mamba states are otherwise normal evictable
        // tree-owned cache entries.
        const std::int32_t slots_needed = 2 + static_cast<std::int32_t>(loadback_nodes.size());
        if (!EnsureMambaCapacityByEvict(slots_needed, op.protected_recovery_node)) {
            return result;
        }
    }
    result.admitted = AdmitChunkFromRetracted(op.request_id, op.target_raw_tokens_exclusive, simulated_free,
                                              op.match_result.paged_cache);
    if (result.admitted) {
        result.cache_transfer_pairs = PrepareMambaDeviceLoadBack(loadback_nodes);
        if (!loadback_nodes.empty() && loadback_nodes.front()->HasMamba()) {
            op.match_result.mamba_cow_src_index = loadback_nodes.front()->MambaSlotIndex();
        }
    }
    return result;
}

cache::admit::Retract::Result HybridPrefixCache::Apply(const cache::admit::Retract& op,
                                                       std::map<std::string, std::int32_t>&) {
    cache::admit::Retract::Result result{};
    std::unique_ptr<HostNodeRef> temp_lock = std::make_unique<HostNodeRef>(op.match_result.host.last_node);
    result.admitted = kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Host>(op.host_pages_needed);
    return result;
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult&, cache::EmptyResult) {}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result, cache::publish::DevicePrefix::Result op_result) {
    result.device_insert_page_count = op_result.device_insert_page_count;
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result,
                                             cache::publish::FinishedRequest::Result&& op_result) {
    result.match_result = std::move(op_result.match_result);
    result.device_insert_page_count = op_result.device_insert_page_count;
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result,
                                             cache::publish::RetractPrefixCommit::Result&& op_result) {
    result.match_result = std::move(op_result.match_result);
    result.device_insert_page_count = op_result.device_insert_page_count;
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result,
                                             cache::materialize::PrefixOnDevice::Result op_result) {
    result.ok = op_result.ok;
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result,
                                             cache::materialize::HostWritebackPages::Result&& op_result) {
    result.ok = op_result.ok;
    if (!result.ok) return;
    result.cache_transfer_pairs = std::move(op_result.cache_transfer_pairs);
    result.pending_host_writeback = std::move(op_result.pending_state);
}

void HybridPrefixCache::AccumulateStepResult(StepCommitResult& result,
                                             cache::state::CreateRequestLocalCache::Result&& op_result) {
    result.local_cache = std::move(op_result.local_cache);
}

}  // namespace tokenspeed
