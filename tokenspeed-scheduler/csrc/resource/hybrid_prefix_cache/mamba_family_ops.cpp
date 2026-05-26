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

#include "resource/allocator/local_mamba_allocator.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/mamba_host_allocator.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/radix_tree/mamba_slot.h"
#include "resource/radix_tree/node_range.h"
#include "resource/radix_tree/tree_node.h"
#include "scheduler/operations/forward.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tokenspeed {
namespace {

std::int32_t AlignMambaCacheSeqlenImpl(std::int32_t seqlen, std::int32_t chunk_size) {
    if (chunk_size <= 0) return seqlen;
    return (seqlen / chunk_size) * chunk_size;
}

TreeNode* FindLastMambaNodeImpl(TreeNode* from) {
    for (TreeNode* cur = from; cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
        if (cur->HasMamba()) return cur;
    }
    return nullptr;
}

TreeNode* FindLastMambaHostNodeImpl(TreeNode* from) {
    for (TreeNode* cur = from; cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
        if (cur->HasMambaOnHost()) return cur;
    }
    return nullptr;
}

}  // namespace

HybridPrefixCache::DecodeFromRetractedRecovery HybridPrefixCache::PrepareDecodeFromRetractedRecovery(
    MatchResult& match_result) const {
    if (!HasMambaAdjunct()) return {};

    TreeNode* mamba_recovery_node = FindLastMambaNode(match_result.host.last_node);
    if (mamba_recovery_node != nullptr) {
        match_result.mamba_cow_src_index = mamba_recovery_node->MambaSlotIndex();
        return {.protected_source_node = mamba_recovery_node};
    }

    TreeNode* host_mamba_recovery_node = FindLastMambaHostNode(match_result.host.last_node);
    if (host_mamba_recovery_node == nullptr) {
        return {.ok = false};
    }
    match_result.mamba_host_src_index = host_mamba_recovery_node->MambaHostSlotIndex();
    match_result.mamba_cow_src_index = -1;
    return {.protected_source_node = host_mamba_recovery_node};
}

void HybridPrefixCache::PopulateMambaMatchCompatibilityFields(ForwardOperationBase& op_base,
                                                              const MatchResult& match_result) const {
    if (!HasMambaAdjunct()) return;
    op_base.mamba_cow_src_idx = match_result.mamba_cow_src_index;
    op_base.mamba_branching_seqlen = match_result.mamba_branching_seqlen;
}

void HybridPrefixCache::PopulateMambaRecoveryCompatibilityFields(ForwardOperationBase& op_base,
                                                                 const MatchResult& match_result) const {
    if (!HasMambaAdjunct()) return;
    op_base.mamba_cow_src_idx = match_result.mamba_cow_src_index;
}

void HybridPrefixCache::PopulateMambaRequestLocalCompatibilityFields(
    ForwardOperationBase& op_base, const LocalMambaAllocator* local_mamba_allocator) const {
    if (!HasMambaAdjunct() || local_mamba_allocator == nullptr) return;
    if (local_mamba_allocator->HasWorking()) {
        op_base.mamba_working_idx = local_mamba_allocator->WorkingIndex();
    }
    if (local_mamba_allocator->HasCheckpoint()) {
        op_base.mamba_checkpoint_dst_idx = local_mamba_allocator->CheckpointIndex();
    }
}

HybridPrefixCache::RequestLocalMambaResult HybridPrefixCache::PrepareRequestLocalMamba(
    const RequestLocalMambaRequest& request) const {
    RequestLocalMambaResult result{};
    const auto should_materialize_checkpoint = [&]() {
        if (!request.checkpoint_raw_position.has_value()) return true;
        const std::int32_t position = *request.checkpoint_raw_position;
        return position > 0 && AlignMambaCacheSeqlen(position) == position;
    };
    switch (request.kind) {
        case RequestLocalMambaKind::kPrefillFirstChunk:
            result.local_mamba_allocator = allocateRequestLocalMambaState(request.checkpoint_raw_position);
            return result;
        case RequestLocalMambaKind::kDecodeFromRetracted:
            result.local_mamba_allocator = allocateRequestLocalMambaState();
            if (HasMambaAdjunct() && result.local_mamba_allocator == nullptr) {
                throw std::logic_error("ScheduleDecodeFromRetractedEvent: failed to allocate Mamba recovery slots");
            }
            return result;
        case RequestLocalMambaKind::kNextCheckpoint:
            if (!HasMambaAdjunct() || request.local_mamba_allocator == nullptr) return result;
            if (!should_materialize_checkpoint()) {
                (void)request.local_mamba_allocator->DetachCheckpoint();
                return result;
            }
            (void)request.local_mamba_allocator->DetachCheckpoint();
            if (!request.local_mamba_allocator->AllocateCheckpoint(request.checkpoint_raw_position.value_or(-1))) {
                throw std::logic_error("SchedulePrefillEvent: failed to allocate Mamba checkpoint slot");
            }
            return result;
    }
    return result;
}

std::unique_ptr<LocalMambaAllocator> HybridPrefixCache::allocateRequestLocalMambaState(
    std::optional<std::int32_t> checkpoint_raw_position) const {
    if (!HasMambaAdjunct()) return nullptr;

    auto local_mamba_allocator = std::make_unique<LocalMambaAllocator>(mamba_allocator_);
    if (!local_mamba_allocator->AllocateWorking() ||
        !local_mamba_allocator->AllocateCheckpoint(checkpoint_raw_position.value_or(-1))) {
        return nullptr;
    }
    return local_mamba_allocator;
}

void HybridPrefixCache::PublishFinishMambaState(const std::vector<std::span<const std::int32_t>>& full_paged_tokens,
                                                LocalMambaAllocator* local_mamba_allocator) {
    if (!HasMambaAdjunct() || local_mamba_allocator == nullptr ||
        (!local_mamba_allocator->HasCheckpoint() && !local_mamba_allocator->HasWorking())) {
        return;
    }
    MatchResult post_match = kv_prefix_cache_.Match(full_paged_tokens);
    TreeNode* terminal = post_match.device.last_node;
    if (terminal == nullptr || terminal->HasMamba()) return;

    std::unique_ptr<MambaSlot> slot_to_publish;
    if (local_mamba_allocator->HasCheckpoint()) {
        const std::int32_t checkpoint_position = local_mamba_allocator->CheckpointPosition();
        if (checkpoint_position < 0 || checkpoint_position == static_cast<std::int32_t>(terminal->DepthInTokens())) {
            slot_to_publish = local_mamba_allocator->DetachCheckpoint();
        }
    }
    if (slot_to_publish == nullptr && local_mamba_allocator->HasWorking()) {
        slot_to_publish = local_mamba_allocator->DetachWorking();
    }
    if (slot_to_publish == nullptr) return;
    InsertMamba(terminal, std::move(slot_to_publish));
}

void HybridPrefixCache::PublishRetractMambaState(TreeNode* terminal,
                                                 std::unique_ptr<LocalMambaAllocator>& local_mamba_allocator) {
    if (local_mamba_allocator == nullptr) return;

    const bool had_request_local_mamba = local_mamba_allocator->HasCheckpoint() || local_mamba_allocator->HasWorking();
    if (!had_request_local_mamba) return;

    if (HasMambaAdjunct()) {
        publishRequestMambaState(terminal, local_mamba_allocator.get());
    }

    // Once retracted, any recoverable Mamba state is tree-owned and therefore
    // evictable by HybridPrefixCache. Do not keep request-local slots alive in
    // Retracting/Retracted.
    local_mamba_allocator.reset();
}

bool HybridPrefixCache::publishRequestMambaState(TreeNode* terminal, LocalMambaAllocator* local_mamba_allocator) {
    if (!HasMambaAdjunct() || terminal == nullptr || terminal->HasMamba() || local_mamba_allocator == nullptr) {
        return false;
    }
    if (local_mamba_allocator->HasCheckpoint()) {
        InsertMamba(terminal, local_mamba_allocator->DetachCheckpoint());
        return true;
    }
    if (local_mamba_allocator->HasWorking()) {
        InsertMamba(terminal, local_mamba_allocator->DetachWorking());
        return true;
    }
    return false;
}

void HybridPrefixCache::augmentMatch(MatchResult& match) const {
    if (mamba_allocator_ == nullptr) return;
    TreeNode* root = match.device.last_node;
    while (root != nullptr && !root->IsRoot()) root = root->Parent();
    if (root == nullptr) return;

    if (mamba_host_allocator_ == nullptr) {
        const std::int32_t page_size = match.device.page_size > 0 ? match.device.page_size : match.host.page_size;
        const std::int32_t kv_depth = std::max(match.device.DepthInPage(), match.host.DepthInPage());
        TreeNode* device_mamba_node = FindLastMambaNode(match.device.last_node);
        TreeNode* host_mamba_node = device_mamba_node == nullptr ? FindLastMambaNode(match.host.last_node) : nullptr;
        TreeNode* mamba_node = device_mamba_node != nullptr ? device_mamba_node : host_mamba_node;
        if (mamba_node == nullptr) {
            const std::int32_t aligned_seqlen = AlignMambaCacheSeqlen(kv_depth * page_size);
            if (aligned_seqlen > 0) {
                match.mamba_branching_seqlen = aligned_seqlen;
            }
            match.device.last_node = root;
            match.host.last_node = root;
            return;
        }

        const std::int32_t mamba_depth = mamba_node->DepthInPage(page_size);
        match.mamba_cow_src_index = mamba_node->MambaSlotIndex();
        if (kv_depth > mamba_depth) {
            const std::int32_t aligned_seqlen = AlignMambaCacheSeqlen(kv_depth * page_size);
            if (aligned_seqlen > mamba_depth * page_size) {
                match.mamba_branching_seqlen = aligned_seqlen;
            }
        }
        match.device.last_node = device_mamba_node != nullptr ? mamba_node : root;
        match.host.last_node = mamba_node;
        return;
    }

    const std::int32_t page_size = match.device.page_size > 0 ? match.device.page_size : match.host.page_size;
    const std::int32_t kv_depth = std::max(match.device.DepthInPage(), match.host.DepthInPage());
    TreeNode* device_mamba_node = FindLastMambaNode(match.device.last_node);
    TreeNode* host_mamba_node = FindLastMambaHostNode(match.host.last_node);
    const std::int32_t device_mamba_depth =
        device_mamba_node == nullptr ? 0 : device_mamba_node->DepthInPage(page_size);
    const std::int32_t host_mamba_depth = host_mamba_node == nullptr ? 0 : host_mamba_node->DepthInPage(page_size);
    const bool prefer_host_mamba = host_mamba_depth > device_mamba_depth;
    std::int32_t mamba_depth = 0;

    if (device_mamba_node != nullptr) {
        match.device.last_node = device_mamba_node;
        if (!prefer_host_mamba) {
            match.mamba_cow_src_index = device_mamba_node->MambaSlotIndex();
        }
        mamba_depth = std::max(mamba_depth, device_mamba_depth);
    } else {
        match.device.last_node = root;
    }

    if (host_mamba_node != nullptr) {
        match.host.last_node = host_mamba_node;
        match.mamba_host_src_index = host_mamba_node->MambaHostSlotIndex();
        if (prefer_host_mamba) {
            match.mamba_cow_src_index = -1;
        }
        mamba_depth = std::max(mamba_depth, host_mamba_depth);
    } else {
        match.host.last_node = root;
    }

    if (kv_depth > mamba_depth) {
        const std::int32_t aligned_seqlen = AlignMambaCacheSeqlen(kv_depth * page_size);
        if (aligned_seqlen > mamba_depth * page_size) {
            match.mamba_branching_seqlen = aligned_seqlen;
        }
    }
}

std::int32_t HybridPrefixCache::AlignMambaCacheSeqlen(std::int32_t seqlen) const {
    return AlignMambaCacheSeqlenImpl(seqlen, mamba_cache_chunk_size_);
}

TreeNode* HybridPrefixCache::FindLastMambaNode(TreeNode* from) const {
    return FindLastMambaNodeImpl(from);
}

TreeNode* HybridPrefixCache::FindLastMambaHostNode(TreeNode* from) const {
    return FindLastMambaHostNodeImpl(from);
}

bool HybridPrefixCache::EnsureMambaCapacityByEvict(std::int32_t num_slots, TreeNode* protected_node) {
    if (mamba_allocator_ == nullptr) return num_slots <= 0;
    return mamba_eviction_manager_.EnsureCapacity(num_slots, protected_node);
}

bool HybridPrefixCache::EnsureMambaHostCapacityByEvict(std::int32_t num_slots, TreeNode* protected_node) {
    if (mamba_host_allocator_ == nullptr) return num_slots <= 0;
    if (mamba_host_allocator_->AvailableSlots() >= num_slots) return true;

    std::vector<TreeNode*> candidates;
    candidates.reserve(mamba_host_nodes_.size());
    for (TreeNode* node : mamba_host_nodes_) {
        if (node == nullptr || node == protected_node || !node->HasMambaOnHost()) continue;
        if (node->OnHost() && GetResource<ResourceType::Host>(node).RefCount() > 0) continue;
        candidates.push_back(node);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const TreeNode* lhs, const TreeNode* rhs) { return lhs->Time() < rhs->Time(); });

    for (TreeNode* node : candidates) {
        if (mamba_host_allocator_->AvailableSlots() >= num_slots) break;
        node->DetachMambaHost();
        mamba_host_nodes_.erase(node);
    }
    if (mamba_host_allocator_->AvailableSlots() < num_slots) {
        spdlog::warn("[HybridPrefixCache] mamba host capacity exhausted required={} after_evict_available={}",
                     num_slots, mamba_host_allocator_->AvailableSlots());
    }
    return mamba_host_allocator_->AvailableSlots() >= num_slots;
}

std::vector<TransferPair> HybridPrefixCache::PrepareMambaHostWriteBack(const std::vector<TreeNode*>& nodes) {
    std::vector<TransferPair> transfers;
    if (mamba_allocator_ == nullptr || mamba_host_allocator_ == nullptr) return transfers;

    std::int32_t needed = 0;
    for (TreeNode* node : nodes) {
        if (node != nullptr && node->HasMamba() && !node->HasMambaOnHost() &&
            pending_mamba_host_writebacks_.find(node) == pending_mamba_host_writebacks_.end()) {
            ++needed;
        }
    }
    if (!EnsureMambaHostCapacityByEvict(needed)) return transfers;

    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->HasMamba() || node->HasMambaOnHost()) continue;
        if (pending_mamba_host_writebacks_.find(node) != pending_mamba_host_writebacks_.end()) continue;
        auto slot = mamba_host_allocator_->Allocate();
        if (!slot.has_value()) break;
        const std::int32_t device_idx = node->MambaSlotIndex();
        const std::int32_t host_idx = slot->Index();
        pending_mamba_host_writebacks_.emplace(node, std::make_unique<MambaSlot>(std::move(*slot)));
        transfers.push_back(TransferPair{CacheKind::kMamba, device_idx, host_idx});
    }
    return transfers;
}

std::vector<TransferPair> HybridPrefixCache::PrepareMambaDeviceLoadBack(const std::vector<TreeNode*>& nodes) {
    std::vector<TransferPair> transfers;
    if (mamba_allocator_ == nullptr || mamba_host_allocator_ == nullptr) return transfers;

    for (TreeNode* node : nodes) {
        if (node == nullptr || !node->HasMambaOnHost() || node->HasMamba()) continue;
        auto slot = mamba_allocator_->Allocate();
        if (!slot.has_value()) break;
        const std::int32_t host_idx = node->MambaHostSlotIndex();
        const std::int32_t device_idx = slot->Index();
        node->AttachMamba(std::make_unique<MambaSlot>(std::move(*slot)));
        mamba_eviction_manager_.TrackNode(node);
        transfers.push_back(TransferPair{CacheKind::kMamba, host_idx, device_idx});
    }
    return transfers;
}

void HybridPrefixCache::InsertMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot) {
    if (terminal_node == nullptr || slot == nullptr) return;
    if (mamba_allocator_ == nullptr) {
        throw std::logic_error("HybridPrefixCache::InsertMamba: mamba adjunct not enabled");
    }
    const std::int32_t page_size = kv_prefix_cache_.PageSize();
    if (page_size <= 0 || terminal_node->DepthInTokens() % static_cast<std::size_t>(page_size) != 0) {
        throw std::logic_error("HybridPrefixCache::InsertMamba: terminal node is not block-aligned");
    }
    terminal_node->AttachMamba(std::move(slot));
    mamba_eviction_manager_.TrackNode(terminal_node);
}

}  // namespace tokenspeed
