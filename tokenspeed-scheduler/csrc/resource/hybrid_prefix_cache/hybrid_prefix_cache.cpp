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

HybridPrefixCache::HybridPrefixCache(KVPrefixCache& kv_prefix_cache, PageAllocator& device_allocator,
                                     MambaChunkAllocator* mamba_allocator, std::int32_t mamba_cache_chunk_size,
                                     MambaHostAllocator* mamba_host_allocator)
    : kv_prefix_cache_{kv_prefix_cache},
      device_allocator_{device_allocator},
      mamba_allocator_{mamba_allocator},
      mamba_host_allocator_{mamba_host_allocator},
      mamba_eviction_manager_{mamba_allocator},
      mamba_cache_chunk_size_{mamba_cache_chunk_size} {
    kv_prefix_cache_.GetDeviceManager().SetEvictionCallback([this](TreeNode* node) { OnKVEvict(node); });
    kv_prefix_cache_.GetHostManager().SetEvictionCallback([this](TreeNode* node) { OnKVHostEvict(node); });
}

HybridPrefixCache::~HybridPrefixCache() {
    SetKvEventSink({});
    kv_prefix_cache_.GetDeviceManager().SetEvictionCallback({});
    kv_prefix_cache_.GetHostManager().SetEvictionCallback({});
}

RecoveryPlan HybridPrefixCache::MatchPrefix(const std::vector<std::span<const std::int32_t>>& token_pages,
                                            MatchIntent intent) {
    DemoteIdleMambaDeviceCopiesPresentOnHost();
    return BuildRecoveryPlan(kv_prefix_cache_.Match(token_pages, intent), intent);
}

RecoveryPlan HybridPrefixCache::BuildRecoveryPlan(MatchResult raw_match, MatchIntent intent) const {
    RecoveryPlan plan{};
    plan.compat_match = std::move(raw_match);
    augmentMatch(plan.compat_match);
    augmentMatchPagedCache(plan.compat_match);
    if (intent == MatchIntent::StateRecovery) {
        const DecodeFromRetractedRecovery recovery = PrepareDecodeFromRetractedRecovery(plan.compat_match);
        plan.recovery_state_available = recovery.ok;
        plan.protected_recovery_node = recovery.protected_source_node;
    }
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

HybridPrefixCache::RequestLocalKVResult HybridPrefixCache::PrepareRequestLocalKV(
    const RequestLocalKVRequest& request) const {
    RequestLocalKVResult result{};
    auto require_allocator = [&]() -> LocalKVAllocator& {
        if (request.local_kv_allocator == nullptr) {
            throw std::invalid_argument("HybridPrefixCache::PrepareRequestLocalKV requires local_kv_allocator");
        }
        return *request.local_kv_allocator;
    };

    switch (request.kind) {
        case RequestLocalKVKind::kPrefillFirstChunk:
            result.local_kv_allocator =
                std::make_unique<LocalKVAllocator>(&device_allocator_, request.tokens_this_round);
            result.local_kv_allocator->Acquire(request.decode_input_tokens);
            return result;
        case RequestLocalKVKind::kPrefillChunk:
            require_allocator().Acquire(request.tokens_this_round);
            return result;
    }
    return result;
}

HybridPrefixCache::CachePublicationResult HybridPrefixCache::Publish(CachePublicationRequest request) {
    CachePublicationResult result{};
    auto require_tokens = [&]() -> const std::vector<std::span<const std::int32_t>>& {
        if (request.full_paged_tokens == nullptr) {
            throw std::invalid_argument("HybridPrefixCache::Publish requires full_paged_tokens");
        }
        return *request.full_paged_tokens;
    };
    auto require_device_ref = [&]() -> std::unique_ptr<DeviceNodeRef>& {
        if (request.device_node_ref == nullptr || *request.device_node_ref == nullptr) {
            throw std::invalid_argument("HybridPrefixCache::Publish requires device_node_ref");
        }
        return *request.device_node_ref;
    };
    auto require_current_device_node = [&]() -> const TreeNode* {
        if (request.current_device_node == nullptr) {
            throw std::invalid_argument("HybridPrefixCache::Publish requires current_device_node");
        }
        return request.current_device_node;
    };
    auto require_local_kv_allocator = [&]() -> LocalKVAllocator& {
        if (request.local_kv_allocator == nullptr) {
            throw std::invalid_argument("HybridPrefixCache::Publish requires local_kv_allocator");
        }
        return *request.local_kv_allocator;
    };
    auto require_page_hashes = [&]() -> const std::vector<std::string>& {
        if (request.page_hashes == nullptr) {
            throw std::invalid_argument("HybridPrefixCache::Publish requires page_hashes");
        }
        return *request.page_hashes;
    };

    switch (request.kind) {
        case CachePublicationKind::kForwardChunk: {
            const auto& full_paged_tokens = require_tokens();
            auto& device_node_ref = require_device_ref();
            LocalKVAllocator& local_kv_allocator = require_local_kv_allocator();
            std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(device_node_ref->Node());
            const std::int32_t new_page_count =
                static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());
            const std::int32_t page_size = kv_prefix_cache_.PageSize();
            const std::int32_t last_inserted_len = static_cast<std::int32_t>(full_paged_tokens.size()) * page_size;
            auto should_publish_mamba_checkpoint = [&]() {
                if (request.local_mamba_allocator == nullptr || !request.local_mamba_allocator->HasCheckpoint()) {
                    return false;
                }
                const std::int32_t checkpoint_position = request.local_mamba_allocator->CheckpointPosition();
                if (checkpoint_position < 0 || checkpoint_position == last_inserted_len) {
                    return true;
                }
                if (!request.chunk_begin.has_value() || page_size <= 0) {
                    return false;
                }
                const std::int32_t chunk_begin = *request.chunk_begin;
                if (last_inserted_len <= chunk_begin || last_inserted_len >= checkpoint_position) {
                    return false;
                }
                const std::int32_t track_len = last_inserted_len - chunk_begin;
                return AlignMambaCacheSeqlen(track_len) == track_len;
            };
            const bool publish_mamba_checkpoint = should_publish_mamba_checkpoint();
            if (new_page_count <= 0) {
                if (request.local_mamba_allocator != nullptr && request.local_mamba_allocator->HasCheckpoint()) {
                    (void)request.local_mamba_allocator->DetachCheckpoint();
                }
                return result;
            }

            OwnedPages pages_to_insert = local_kv_allocator.TakeFirst(new_page_count);
            auto insert_result = kv_prefix_cache_.Insert<ResourceType::Device>(full_paged_tokens, prefix_pages,
                                                                               std::move(pages_to_insert));

            if (request.local_mamba_allocator != nullptr && request.local_mamba_allocator->HasCheckpoint()) {
                std::unique_ptr<MambaSlot> checkpoint = request.local_mamba_allocator->DetachCheckpoint();
                if (publish_mamba_checkpoint) {
                    InsertMamba(insert_result.last_node, std::move(checkpoint));
                }
            }
            device_node_ref = std::make_unique<DeviceNodeRef>(insert_result.last_node);
            result.device_insert_page_count = new_page_count;
            return result;
        }
        case CachePublicationKind::kFinishChunk: {
            const auto& full_paged_tokens = require_tokens();
            const TreeNode* current_device_node = require_current_device_node();
            LocalKVAllocator& local_kv_allocator = require_local_kv_allocator();
            const std::vector<std::string>& page_hashes = require_page_hashes();
            std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(current_device_node);
            const std::int32_t alloc_count =
                static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());

            if (alloc_count > 0) {
                OwnedPages alloc_pages = local_kv_allocator.TakeFirst(alloc_count);
                kv_prefix_cache_.Insert<ResourceType::Device>(full_paged_tokens, prefix_pages, std::move(alloc_pages),
                                                              page_hashes);
                PublishFinishMambaState(full_paged_tokens, request.local_mamba_allocator);
            }

            result.device_insert_page_count = std::max(0, alloc_count);
            result.match_result = kv_prefix_cache_.Match(full_paged_tokens);
            return result;
        }
        case CachePublicationKind::kRetractDeviceInsertPageCount: {
            const auto& full_paged_tokens = require_tokens();
            const TreeNode* current_device_node = require_current_device_node();
            std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(current_device_node);
            const std::int32_t full_page_count = static_cast<std::int32_t>(full_paged_tokens.size());
            const std::int32_t prefix_page_count = static_cast<std::int32_t>(prefix_pages.size());
            if (full_page_count < prefix_page_count) {
                throw std::logic_error(
                    "HybridPrefixCache::Publish retract plan: current device prefix exceeds "
                    "available full token pages");
            }
            result.device_insert_page_count = full_page_count - prefix_page_count;
            return result;
        }
        case CachePublicationKind::kRetractChunk: {
            const auto& full_paged_tokens = require_tokens();
            const TreeNode* current_device_node = require_current_device_node();
            std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(current_device_node);
            const std::int32_t alloc_count =
                static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());
            if (alloc_count < 0) {
                throw std::logic_error(
                    "HybridPrefixCache::Publish retract chunk: current device prefix exceeds "
                    "available full token pages");
            }
            if (request.pages_to_insert.Size() != alloc_count) {
                throw std::logic_error("HybridPrefixCache::Publish retract chunk: request-local page count mismatch");
            }

            kv_prefix_cache_.Insert<ResourceType::Device>(full_paged_tokens, prefix_pages,
                                                          std::move(request.pages_to_insert));
            result.device_insert_page_count = alloc_count;
            result.match_result = kv_prefix_cache_.Match(full_paged_tokens, MatchIntent::StateRecovery);
            return result;
        }
    }
    return result;
}

HybridPrefixCache::CacheMaterializationResult HybridPrefixCache::Materialize(
    const CacheMaterializationRequest& request) {
    CacheMaterializationResult result{};
    auto require_match = [&]() -> const MatchResult& {
        if (request.match_result == nullptr) {
            throw std::invalid_argument("HybridPrefixCache::Materialize requires match_result");
        }
        return *request.match_result;
    };
    auto require_write_diff = [&]() -> const std::vector<TreeNode*>& {
        if (request.write_diff == nullptr) {
            throw std::invalid_argument("HybridPrefixCache::Materialize requires write_diff");
        }
        return *request.write_diff;
    };

    switch (request.kind) {
        case CacheMaterializationKind::kPrefillHostPrefixOnDevice: {
            const MatchResult& match_result = require_match();
            (void)kv_prefix_cache_.AllocateResourceOfType<ResourceType::Device>(
                match_result.NodesWithout<ResourceType::Device>());
            return result;
        }
        case CacheMaterializationKind::kDecodeRecoveryHostPrefixOnDevice: {
            const MatchResult& match_result = require_match();
            result.ok = kv_prefix_cache_.AllocateResourceOfType<ResourceType::Device>(
                match_result.NodesWithout<ResourceType::Device>());
            return result;
        }
        case CacheMaterializationKind::kFinishWritebackHostPages: {
            const std::vector<TreeNode*>& write_diff = require_write_diff();
            std::int32_t host_pages_num = 0;
            for (TreeNode* node : write_diff) {
                host_pages_num += node->Device().NumPages();
            }
            if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Host>(host_pages_num)) {
                result.ok = false;
                return result;
            }
            (void)kv_prefix_cache_.AllocateResourceOfType<ResourceType::Host>(write_diff);
            return result;
        }
        case CacheMaterializationKind::kRetractWritebackHostPages: {
            const std::vector<TreeNode*>& write_diff = require_write_diff();
            result.ok = kv_prefix_cache_.AllocateResourceOfType<ResourceType::Host>(write_diff);
            return result;
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

CacheStatsSnapshot HybridPrefixCache::Stats(const StatsRequest& request) const {
    CacheStatsSnapshot snapshot{
        .available_device_pages = static_cast<std::size_t>(device_allocator_.AvailablePages()),
    };

    snapshot.paged_cache_group_ids.reserve(paged_cache_allocators_.size());
    for (const auto& [gid, _] : paged_cache_allocators_) {
        snapshot.paged_cache_group_ids.push_back(gid);
    }

    std::vector<std::string> requested_groups = request.paged_cache_group_ids;
    if (requested_groups.empty()) {
        requested_groups = snapshot.paged_cache_group_ids;
    }
    for (const auto& gid : requested_groups) {
        auto alloc_it = paged_cache_allocators_.find(gid);
        if (alloc_it == paged_cache_allocators_.end() || alloc_it->second == nullptr) {
            throw std::out_of_range("HybridPrefixCache::Stats: group_id not configured");
        }
        snapshot.paged_cache_total_pages[gid] = alloc_it->second->TotalPages();
        snapshot.paged_cache_available_pages[gid] = alloc_it->second->AvailablePages();
        snapshot.paged_cache_failed_alloc_count[gid] = alloc_it->second->FailedAllocCount();

        if (request.request_id.has_value()) {
            std::vector<std::int32_t> pages;
            std::int32_t base_logical_page = 0;
            auto req_it = request_paged_cache_tables_.find(*request.request_id);
            if (req_it != request_paged_cache_tables_.end()) {
                auto group_it = req_it->second.find(gid);
                if (group_it != req_it->second.end()) {
                    pages = group_it->second.PageIds();
                    base_logical_page = group_it->second.BaseLogicalPage();
                }
            }
            snapshot.request_paged_cache_page_ids[gid] = std::move(pages);
            snapshot.request_paged_cache_base_logical_page[gid] = base_logical_page;
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

void HybridPrefixCache::PrepareForwardOp(ForwardOperationBase& op_base, const CacheOpPrepareRequest& request) {
    auto require_match = [&]() -> const MatchResult& {
        if (request.match_result == nullptr) {
            throw std::invalid_argument(
                "HybridPrefixCache::PrepareForwardOp requires match_result for this prepare kind");
        }
        return *request.match_result;
    };

    PopulateMambaRequestLocalCompatibilityFields(op_base, request.local_mamba_allocator);

    switch (request.kind) {
        case WorkerCompatibilityCommitKind::kPrefillFirstChunk: {
            const MatchResult& match_result = require_match();
            PopulateMambaMatchCompatibilityFields(op_base, match_result);
            CommitChunk(op_base.request_id, request.terminal);
            acquireAndPopulateOp(op_base, request.first_raw_position_of_op, request.target_raw_tokens_exclusive,
                                 match_result.paged_cache);
            return;
        }
        case WorkerCompatibilityCommitKind::kPrefillChunk:
            CommitChunk(op_base.request_id, request.terminal);
            acquireAndPopulateOp(op_base, request.first_raw_position_of_op, request.target_raw_tokens_exclusive,
                                 request.paged_cache_hit);
            return;
        case WorkerCompatibilityCommitKind::kDecodeChunk:
            if (request.commit_prior_chunk) {
                CommitChunk(op_base.request_id, request.terminal);
            }
            acquireAndPopulateOp(op_base, request.first_raw_position_of_op, request.target_raw_tokens_exclusive, {});
            return;
        case WorkerCompatibilityCommitKind::kDecodeFromRetracted: {
            const MatchResult& match_result = require_match();
            PopulateMambaRecoveryCompatibilityFields(op_base, match_result);
            ReleaseRequest(op_base.request_id);
            acquireAndPopulateOp(op_base, 0, request.target_raw_tokens_exclusive, match_result.paged_cache);
            return;
        }
    }
}

AdmissionVerdict HybridPrefixCache::Admit(const AdmissionRequest& request,
                                          std::map<std::string, std::int32_t>& simulated_free) {
    const MatchResult* match = request.compat_match;
    if (match == nullptr && request.recovery_plan != nullptr) {
        match = &request.recovery_plan->compat_match;
    }

    CacheAdmissionRequest compat_request{
        .kind = request.kind,
        .request_id = request.request_id,
        .device_pages_needed = request.device_pages_needed,
        .host_pages_needed = request.host_pages_needed,
        .tokens_this_round = request.tokens_this_round,
        .first_raw_position_of_op = request.first_raw_position_of_op,
        .target_raw_tokens_exclusive = request.target_raw_tokens_exclusive,
        .match_result = match,
        .mamba_recovery_node = request.protected_recovery_node,
        .refresh_mamba_checkpoint = request.refresh_mamba_checkpoint,
    };
    CacheAdmissionResult compat_result = Admit(compat_request, simulated_free);
    return AdmissionVerdict{
        .admitted = compat_result.admitted,
        .mamba_branching_seqlen = compat_result.mamba_branching_seqlen,
        .mamba_cow_src_index = compat_result.mamba_cow_src_index,
        .cache_transfer_pairs = std::move(compat_result.cache_transfer_pairs),
    };
}

StepCommitResult HybridPrefixCache::StepCommit(StepCommitRequest request) {
    StepCommitResult result{};

    if (request.materialize_prefix.has_value()) {
        const auto& materialize = *request.materialize_prefix;
        const CacheMaterializationKind kind = materialize.require_all_pages
                                                  ? CacheMaterializationKind::kDecodeRecoveryHostPrefixOnDevice
                                                  : CacheMaterializationKind::kPrefillHostPrefixOnDevice;
        result.ok = Materialize({
                                    .kind = kind,
                                    .match_result = materialize.compat_match,
                                })
                        .ok;
        if (!result.ok) return result;
    }

    if (request.publish_device_prefix.has_value()) {
        const auto& publish = *request.publish_device_prefix;
        (void)Publish({
            .kind = CachePublicationKind::kForwardChunk,
            .full_paged_tokens = publish.full_paged_tokens,
            .chunk_begin = publish.chunk_begin,
            .device_node_ref = publish.device_node_ref,
            .local_kv_allocator = publish.local_kv_allocator,
            .local_mamba_allocator = publish.local_mamba_allocator,
        });
    }

    if (request.publish_finished_request.has_value()) {
        const auto& publish = *request.publish_finished_request;
        auto publish_result = Publish({
            .kind = CachePublicationKind::kFinishChunk,
            .full_paged_tokens = publish.full_paged_tokens,
            .current_device_node = publish.current_device_node,
            .local_kv_allocator = publish.local_kv_allocator,
            .local_mamba_allocator = publish.local_mamba_allocator,
            .page_hashes = publish.page_hashes,
        });
        result.match_result = std::move(publish_result.match_result);
        result.device_insert_page_count = publish_result.device_insert_page_count;
    }

    if (request.plan_device_prefix_insertion.has_value()) {
        const auto& plan = *request.plan_device_prefix_insertion;
        auto publish_result = Publish({
            .kind = CachePublicationKind::kRetractDeviceInsertPageCount,
            .full_paged_tokens = plan.full_paged_tokens,
            .current_device_node = plan.current_device_node,
        });
        result.device_insert_page_count = publish_result.device_insert_page_count;
    }

    if (request.publish_device_prefix_insertion.has_value()) {
        auto& publish = *request.publish_device_prefix_insertion;
        auto publish_result = Publish({
            .kind = CachePublicationKind::kRetractChunk,
            .full_paged_tokens = publish.full_paged_tokens,
            .current_device_node = publish.current_device_node,
            .pages_to_insert = std::move(publish.pages_to_insert),
        });
        result.match_result = std::move(publish_result.match_result);
        result.device_insert_page_count = publish_result.device_insert_page_count;
    }

    if (request.materialize_host_writeback.has_value()) {
        const auto& materialize = *request.materialize_host_writeback;
        const CacheMaterializationKind kind = materialize.ensure_capacity_before_allocate
                                                  ? CacheMaterializationKind::kFinishWritebackHostPages
                                                  : CacheMaterializationKind::kRetractWritebackHostPages;
        result.ok = Materialize({
                                    .kind = kind,
                                    .write_diff = materialize.write_diff,
                                })
                        .ok;
        if (!result.ok) return result;
        if (materialize.write_diff != nullptr) {
            result.cache_transfer_pairs = PrepareMambaHostWriteBack(*materialize.write_diff);
            for (const TransferPair& transfer : result.cache_transfer_pairs) {
                if (transfer.kind != CacheKind::kMamba) continue;
                for (TreeNode* node : *materialize.write_diff) {
                    if (node != nullptr && node->HasMamba() && node->MambaSlotIndex() == transfer.src) {
                        result.mamba_writeback_nodes.push_back(node);
                        break;
                    }
                }
            }
        }
    }

    if (request.publish_tree_owned_request_state.has_value()) {
        const auto& publish = *request.publish_tree_owned_request_state;
        if (publish.local_mamba_allocator_owner == nullptr) {
            throw std::invalid_argument(
                "HybridPrefixCache::StepCommit publish_tree_owned_request_state requires "
                "local_mamba_allocator_owner");
        }
        PublishRetractMambaState(publish.terminal, *publish.local_mamba_allocator_owner);
    }

    const bool strict_mamba_create = request.request_local_mamba.has_value() &&
                                     request.request_local_mamba->create_allocator &&
                                     request.request_local_mamba->require_allocator;

    auto apply_request_local_mamba = [&]() {
        if (!request.request_local_mamba.has_value()) return;
        const auto& mamba = *request.request_local_mamba;
        if (mamba.create_allocator) {
            auto mamba_result = PrepareRequestLocalMamba({
                .kind = mamba.require_allocator ? RequestLocalMambaKind::kDecodeFromRetracted
                                                : RequestLocalMambaKind::kPrefillFirstChunk,
                .checkpoint_raw_position = mamba.checkpoint_raw_position,
            });
            result.local_mamba_allocator = std::move(mamba_result.local_mamba_allocator);
        }
        if (mamba.refresh_checkpoint_allocator != nullptr) {
            (void)PrepareRequestLocalMamba({
                .kind = RequestLocalMambaKind::kNextCheckpoint,
                .local_mamba_allocator = mamba.refresh_checkpoint_allocator,
                .checkpoint_raw_position = mamba.checkpoint_raw_position,
            });
        }
    };

    auto apply_request_local_kv = [&]() {
        if (!request.request_local_kv.has_value()) return;
        const auto& kv = *request.request_local_kv;
        if (kv.create_allocator) {
            auto kv_result = PrepareRequestLocalKV({
                .kind = RequestLocalKVKind::kPrefillFirstChunk,
                .tokens_this_round = kv.initial_tokens,
                .decode_input_tokens = kv.acquire_tokens,
            });
            result.local_kv_allocator = std::move(kv_result.local_kv_allocator);
            return;
        }
        if (kv.allocator != nullptr || kv.acquire_tokens > 0) {
            (void)PrepareRequestLocalKV({
                .kind = RequestLocalKVKind::kPrefillChunk,
                .local_kv_allocator = kv.allocator,
                .tokens_this_round = kv.acquire_tokens,
            });
        }
    };

    if (strict_mamba_create) {
        apply_request_local_mamba();
        apply_request_local_kv();
    } else {
        apply_request_local_kv();
        apply_request_local_mamba();
    }

    if (request.worker_metadata.has_value()) {
        const auto& worker = *request.worker_metadata;
        if (worker.op_base == nullptr) {
            throw std::invalid_argument("HybridPrefixCache::StepCommit PrepareWorkerOp requires op_base");
        }
        PrepareForwardOp(*worker.op_base,
                         {
                             .kind = worker.kind,
                             .terminal = worker.terminal,
                             .first_raw_position_of_op = worker.first_raw_position_of_op,
                             .target_raw_tokens_exclusive = worker.target_raw_tokens_exclusive,
                             .match_result = worker.compat_match,
                             .local_mamba_allocator = worker.local_mamba_allocator_view,
                             .paged_cache_hit = worker.compat_match == nullptr ? worker.paged_cache_hit
                                                                               : worker.compat_match->paged_cache,
                             .commit_prior_chunk = worker.commit_tree_prefix_before_acquire,
                         });
    }

    return result;
}

HybridPrefixCache::CacheAdmissionResult HybridPrefixCache::Admit(const CacheAdmissionRequest& request,
                                                                 std::map<std::string, std::int32_t>& simulated_free) {
    CacheAdmissionResult result{};
    auto require_match = [&]() -> const MatchResult& {
        if (request.match_result == nullptr) {
            throw std::invalid_argument("HybridPrefixCache::Admit requires match_result for this admission kind");
        }
        return *request.match_result;
    };
    auto mamba_device_loadback_nodes = [this](const MatchResult& match_result, TreeNode* preferred_source = nullptr) {
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
    };

    switch (request.kind) {
        case AdmissionRequestKind::kPrefillFirstChunk: {
            const MatchResult& match_result = require_match();
            std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);
            if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(request.device_pages_needed)) {
                return result;
            }
            std::vector<TreeNode*> loadback_nodes = mamba_device_loadback_nodes(match_result);
            std::optional<std::int32_t> mamba_branching_seqlen;
            if (HasMambaAdjunct()) {
                if (match_result.mamba_branching_seqlen == -1) {
                    const std::int32_t aligned = AlignMambaCacheSeqlen(request.tokens_this_round);
                    if (aligned > 0) {
                        mamba_branching_seqlen = aligned;
                    }
                }
                const std::int32_t slots_needed = 2 + static_cast<std::int32_t>(loadback_nodes.size());
                if (!EnsureMambaCapacityByEvict(slots_needed)) {
                    return result;
                }
            }
            result.admitted = AdmitChunk(request.request_id, request.first_raw_position_of_op,
                                         request.target_raw_tokens_exclusive, simulated_free, match_result.paged_cache);
            if (result.admitted) {
                result.mamba_branching_seqlen = mamba_branching_seqlen;
                result.cache_transfer_pairs = PrepareMambaDeviceLoadBack(loadback_nodes);
                if (!loadback_nodes.empty() && loadback_nodes.front()->HasMamba()) {
                    result.mamba_cow_src_index = loadback_nodes.front()->MambaSlotIndex();
                }
            }
            return result;
        }
        case AdmissionRequestKind::kPrefillChunk:
            if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(request.device_pages_needed)) {
                return result;
            }
            if (HasMambaAdjunct() && !EnsureMambaCapacityByEvict(1)) {
                return result;
            }
            result.admitted = AdmitChunk(request.request_id, request.first_raw_position_of_op,
                                         request.target_raw_tokens_exclusive, simulated_free);
            return result;
        case AdmissionRequestKind::kDecodeChunk:
            if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(request.device_pages_needed)) {
                return result;
            }
            if (request.refresh_mamba_checkpoint && HasMambaAdjunct() && !EnsureMambaCapacityByEvict(1)) {
                return result;
            }
            result.admitted = AdmitChunk(request.request_id, request.first_raw_position_of_op,
                                         request.target_raw_tokens_exclusive, simulated_free);
            return result;
        case AdmissionRequestKind::kDecodeFromRetracted: {
            const MatchResult& match_result = require_match();
            std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);
            if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(request.device_pages_needed)) {
                return result;
            }
            std::vector<TreeNode*> loadback_nodes =
                mamba_device_loadback_nodes(match_result, request.mamba_recovery_node);
            if (HasMambaAdjunct()) {
                if (request.mamba_recovery_node == nullptr) {
                    return result;
                }
                // Recovery COWs the tree-owned Mamba state into fresh
                // request-local working/checkpoint slots. Protect the source
                // node only for this allocation; retracted Mamba states are
                // otherwise normal evictable tree-owned cache entries.
                const std::int32_t slots_needed = 2 + static_cast<std::int32_t>(loadback_nodes.size());
                if (!EnsureMambaCapacityByEvict(slots_needed, request.mamba_recovery_node)) {
                    return result;
                }
            }
            result.admitted = AdmitChunkFromRetracted(request.request_id, request.target_raw_tokens_exclusive,
                                                      simulated_free, match_result.paged_cache);
            if (result.admitted) {
                result.cache_transfer_pairs = PrepareMambaDeviceLoadBack(loadback_nodes);
                if (!loadback_nodes.empty() && loadback_nodes.front()->HasMamba()) {
                    result.mamba_cow_src_index = loadback_nodes.front()->MambaSlotIndex();
                }
            }
            return result;
        }
        case AdmissionRequestKind::kRetract: {
            const MatchResult& match_result = require_match();
            std::unique_ptr<HostNodeRef> temp_lock = std::make_unique<HostNodeRef>(match_result.host.last_node);
            result.admitted = kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Host>(request.host_pages_needed);
            return result;
        }
    }
    return result;
}

}  // namespace tokenspeed
