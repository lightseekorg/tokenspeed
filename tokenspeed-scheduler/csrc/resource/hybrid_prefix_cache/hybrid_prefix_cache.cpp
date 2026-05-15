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
#include "resource/allocator/mamba_chunk_allocator.h"

#include <cstddef>
#include <stdexcept>

namespace tokenspeed {

HybridPrefixCache::HybridPrefixCache(KVPrefixCache& kv_prefix_cache, MambaChunkAllocator* mamba_allocator,
                                     MambaChunkAllocator* host_mamba_allocator, std::int32_t mamba_cache_chunk_size)
    : kv_prefix_cache_{kv_prefix_cache},
      mamba_allocator_{mamba_allocator},
      host_mamba_allocator_{host_mamba_allocator},
      mamba_eviction_manager_{mamba_allocator, ResourceType::Device},
      host_mamba_eviction_manager_{host_mamba_allocator, ResourceType::Host,
                                   [this](TreeNode* node) { detachHostMamba(node); }},
      mamba_cache_chunk_size_{mamba_cache_chunk_size} {}

MatchResult HybridPrefixCache::Match(const token_vec_t& token_ids, MatchIntent intent) {
    auto match = kv_prefix_cache_.Match(token_ids, intent);
    augmentMatch(match);
    return match;
}

MatchResult HybridPrefixCache::Match(const std::vector<std::span<const std::int32_t>>& token_pages,
                                     MatchIntent intent) {
    auto match = kv_prefix_cache_.Match(token_pages, intent);
    augmentMatch(match);
    return match;
}

void HybridPrefixCache::augmentMatch(MatchResult& match) const {
    TreeNode* device_terminal = match.device.last_node;
    TreeNode* host_terminal = match.host.last_node;
    if (device_terminal == nullptr || device_terminal->IsRoot()) return;

    TreeNode* device_mamba = FindLastMambaNode(device_terminal, ResourceType::Device);
    TreeNode* host_mamba = FindLastMambaNode(host_terminal, ResourceType::Host);
    TreeNode* usable = device_mamba != nullptr ? device_mamba : host_mamba;

    if (usable == nullptr) {
        const std::int32_t kv_depth = match.device.DepthInPage();
        const std::int32_t aligned_seqlen = AlignMambaCacheSeqlen(kv_depth * match.device.page_size);
        if (aligned_seqlen > 0) {
            match.mamba_branching_seqlen = aligned_seqlen;
        }
        TreeNode* root = device_terminal;
        while (!root->IsRoot()) root = root->Parent();
        match.device.last_node = root;
        match.host.last_node = root;
        return;
    }

    std::int32_t page_size = match.device.page_size;
    std::int32_t kv_depth = match.device.DepthInPage();
    std::int32_t mamba_depth = usable->DepthInPage(page_size);

    if (usable->HasMambaDevice()) {
        match.mamba_cow_src_index = usable->MambaSlotIndex();
    }

    if (kv_depth > mamba_depth) {
        const std::int32_t aligned_seqlen = AlignMambaCacheSeqlen(kv_depth * page_size);
        if (aligned_seqlen > mamba_depth * page_size) {
            match.mamba_branching_seqlen = aligned_seqlen;
        }
    }

    match.device.last_node = device_mamba != nullptr ? device_mamba : usable->Parent();
    if (match.device.last_node == nullptr) match.device.last_node = usable;
    match.host.last_node = usable;
}

std::int32_t HybridPrefixCache::AlignMambaCacheSeqlen(std::int32_t seqlen) const {
    if (mamba_cache_chunk_size_ <= 0) return seqlen;
    return (seqlen / mamba_cache_chunk_size_) * mamba_cache_chunk_size_;
}

TreeNode* HybridPrefixCache::FindLastMambaNode(TreeNode* from, ResourceType residency,
                                                 bool require_exact_depth) const {
    for (TreeNode* cur = from; cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
        const bool has_mamba = residency == ResourceType::Device ? cur->HasMambaDevice() : cur->HasMambaHost();
        if (!has_mamba) continue;
        if (!require_exact_depth) return cur;
        const bool exact = residency == ResourceType::Device ? cur->HasAlignedMambaDevice() : cur->HasAlignedMambaHost();
        if (exact) return cur;
    }
    return nullptr;
}

bool HybridPrefixCache::EnsureMambaCapacityByEvict(std::int32_t num_slots, TreeNode* protected_node) {
    return mamba_eviction_manager_.EnsureCapacity(num_slots, protected_node);
}

bool HybridPrefixCache::EnsureMambaHostCapacityByEvict(std::int32_t num_slots, TreeNode* protected_node) {
    return host_mamba_eviction_manager_.EnsureCapacity(num_slots, protected_node);
}

std::unique_ptr<MambaSlot> HybridPrefixCache::AllocateDeviceMamba() {
    auto slot = mamba_allocator_->Allocate();
    if (!slot.has_value()) return nullptr;
    return std::make_unique<MambaSlot>(std::move(*slot));
}

std::unique_ptr<MambaSlot> HybridPrefixCache::AllocateHostMamba() {
    auto slot = host_mamba_allocator_->Allocate();
    if (!slot.has_value()) return nullptr;
    return std::make_unique<MambaSlot>(std::move(*slot));
}

void HybridPrefixCache::InsertMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot) {
    if (terminal_node == nullptr) return;
    InsertMamba(terminal_node, std::move(slot), terminal_node->DepthInTokens());
}

void HybridPrefixCache::InsertMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot,
                                    std::size_t depth_tokens) {
    if (terminal_node == nullptr || slot == nullptr) return;
    const std::int32_t page_size = kv_prefix_cache_.PageSize();
    if (page_size <= 0 || terminal_node->DepthInTokens() % static_cast<std::size_t>(page_size) != 0) {
        throw std::logic_error("HybridPrefixCache::InsertMamba: terminal node is not block-aligned");
    }
    terminal_node->AttachMambaDevice(std::move(slot), depth_tokens);
    mamba_eviction_manager_.TrackNode(terminal_node);
}

void HybridPrefixCache::LoadBackMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot) {
    if (terminal_node == nullptr || !terminal_node->HasMambaHost()) return;
    InsertMamba(terminal_node, std::move(slot), terminal_node->MambaHostDepthTokens());
}

void HybridPrefixCache::AttachHostMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot) {
    if (terminal_node == nullptr) return;
    AttachHostMamba(terminal_node, std::move(slot), terminal_node->DepthInTokens());
}

void HybridPrefixCache::AttachHostMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot,
                                        std::size_t depth_tokens) {
    if (terminal_node == nullptr || slot == nullptr) return;
    terminal_node->AttachMambaHost(std::move(slot), depth_tokens);
    host_mamba_eviction_manager_.TrackNode(terminal_node);
}

void HybridPrefixCache::detachHostMamba(TreeNode* node) {
    node->DetachMambaHost();
}

void HybridPrefixCache::OnKVEvict(TreeNode* node) {
    if (node == nullptr || !node->HasMambaDevice()) return;
    mamba_eviction_manager_.UntrackNode(node);
    node->DetachMambaDevice();
    if (node->Parent() != nullptr) {
        mamba_eviction_manager_.UpdateLeaf(node->Parent());
    }
}

std::int32_t HybridPrefixCache::AvailableSlots() const {
    return mamba_allocator_->AvailableSlots();
}

std::int32_t HybridPrefixCache::AvailableHostSlots() const {
    return host_mamba_allocator_->AvailableSlots();
}

}  // namespace tokenspeed
