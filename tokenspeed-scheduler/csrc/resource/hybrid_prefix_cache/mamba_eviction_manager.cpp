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

#include "resource/hybrid_prefix_cache/mamba_eviction_manager.h"
#include "resource/allocator/mamba_chunk_allocator.h"

#include <queue>

namespace tokenspeed {

MambaEvictionManager::MambaEvictionManager(MambaChunkAllocator* allocator, ResourceType residency, DetachFn detach_fn)
    : allocator_{allocator}, residency_{residency}, detach_fn_{std::move(detach_fn)} {}

bool MambaEvictionManager::hasMamba(const TreeNode* node) const {
    return residency_ == ResourceType::Device ? node->HasMambaDevice() : node->HasMambaHost();
}

bool MambaEvictionManager::isLocked(const TreeNode* node) const {
    if (residency_ != ResourceType::Device) return false;
    return node->OnDevice() && GetResource<ResourceType::Device>(node).RefCount() > 0;
}

void MambaEvictionManager::detach(TreeNode* node) const {
    if (detach_fn_) {
        detach_fn_(node);
        return;
    }
    if (residency_ == ResourceType::Device) {
        node->DetachMambaDevice();
    } else {
        node->DetachMambaHost();
    }
}

bool MambaEvictionManager::hasChildWithMamba(const TreeNode* node) const {
    for (const auto& [_, child] : node->Children()) {
        if (child != nullptr && hasMamba(child.get())) return true;
    }
    return false;
}

bool MambaEvictionManager::isMambaLeaf(const TreeNode* node) const {
    return !node->IsRoot() && hasMamba(node) && !hasChildWithMamba(node);
}

void MambaEvictionManager::TrackNode(TreeNode* node) {
    if (node == nullptr || node->IsRoot()) return;
    if (isMambaLeaf(node)) {
        mamba_leaves_.insert(node);
    }
    if (node->Parent() != nullptr && !node->Parent()->IsRoot() && !isMambaLeaf(node->Parent())) {
        mamba_leaves_.erase(node->Parent());
    }
}

void MambaEvictionManager::UntrackNode(TreeNode* node) {
    mamba_leaves_.erase(node);
}

void MambaEvictionManager::UpdateLeaf(TreeNode* node) {
    if (node == nullptr || node->IsRoot()) return;
    if (isMambaLeaf(node)) {
        mamba_leaves_.insert(node);
    } else {
        mamba_leaves_.erase(node);
    }
}

std::int32_t MambaEvictionManager::Evict(std::int32_t num_slots, TreeNode* protected_node) {
    auto older = [](const TreeNode* a, const TreeNode* b) { return a->Time() > b->Time(); };
    std::priority_queue<TreeNode*, std::vector<TreeNode*>, decltype(older)> candidates(older);

    for (TreeNode* n : mamba_leaves_) {
        if (n == protected_node || isLocked(n)) continue;
        candidates.push(n);
    }

    std::int32_t evicted = 0;
    while (evicted < num_slots && !candidates.empty()) {
        TreeNode* leaf = candidates.top();
        candidates.pop();

        detach(leaf);
        evicted++;
        mamba_leaves_.erase(leaf);

        TreeNode* parent = leaf->Parent();
        if (parent != nullptr && !parent->IsRoot() && isMambaLeaf(parent)) {
            mamba_leaves_.insert(parent);
            if (!isLocked(parent) && parent != protected_node) {
                candidates.push(parent);
            }
        }
    }
    return evicted;
}

bool MambaEvictionManager::EnsureCapacity(std::int32_t required_slots, TreeNode* protected_node) {
    std::int32_t available = allocator_->AvailableSlots();
    if (available >= required_slots) return true;
    Evict(required_slots - available, protected_node);
    return allocator_->AvailableSlots() >= required_slots;
}

std::int32_t MambaEvictionManager::EvictableSlots() const {
    std::int32_t total = 0;
    for (const TreeNode* n : mamba_leaves_) {
        if (isLocked(n)) continue;
        total++;
    }
    return total;
}

}  // namespace tokenspeed
