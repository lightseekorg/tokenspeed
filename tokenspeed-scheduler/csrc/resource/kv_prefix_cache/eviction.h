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
#include <cstdlib>
#include <queue>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>

#include "resource/allocator/owned_pages.h"
#include "resource/radix_tree/tree_node.h"

namespace tokenspeed {

template <ResourceType RType>
inline bool HasChildWithPages(const TreeNode* node) {
    for (const auto& [_, child] : node->Children()) {
        if (child == nullptr) continue;
        // A child without this resource type attached (device_ / host_ is nullptr)
        // cannot have pages — calling GetResource on it would dereference nullptr.
        if constexpr (RType == ResourceType::Device) {
            if (!child->OnDevice()) continue;
        } else {
            if (!child->OnHost()) continue;
        }
        const auto& resource = GetResource<RType>(child.get());
        if (!resource.IsEmpty()) return true;
    }
    return false;
}

template <ResourceType RType>
inline bool IsLeaf(const TreeNode* node) {
    // A node without this resource type (device_ / host_ is nullptr) cannot be a leaf.
    if constexpr (RType == ResourceType::Device) {
        if (!node->OnDevice()) return false;
    } else {
        if (!node->OnHost()) return false;
    }
    const auto& resource = GetResource<RType>(node);
    return !node->IsRoot() && !resource.IsEmpty() && !HasChildWithPages<RType>(node);
}

template <ResourceType RType>
void ResourceManager<RType>::updateLeaf(TreeNode* node) {
    if (node == nullptr || node->IsRoot()) return;
    if (IsLeaf<RType>(node)) {
        leaves_.insert(node);
    } else {
        leaves_.erase(node);
    }
}

template <ResourceType RType>
void ResourceManager<RType>::UpdateLeaves(TreeNode* node) {
    updateLeaf(node);
    if (node != nullptr) {
        updateLeaf(node->Parent());
    }
}

template <ResourceType RType>
std::vector<TreeNode*> ResourceManager<RType>::Evict(std::int32_t num_pages) {
    std::vector<TreeNode*> evicted_nodes;
    if (num_pages <= 0) {
        return evicted_nodes;
    }

    auto older = [](const TreeNode* a, const TreeNode* b) { return a->Time() > b->Time(); };
    std::priority_queue<TreeNode*, std::vector<TreeNode*>, decltype(older)> candidates(older);

    for (TreeNode* n : leaves_) {
        if (GetResource<RType>(n).IsEvictable()) {
            candidates.push(n);
        }
    }

    std::int32_t evicted = 0;
    while (evicted < num_pages && !candidates.empty()) {
        TreeNode* leaf = candidates.top();
        candidates.pop();
        if constexpr (RType == ResourceType::Device) {
            if (std::getenv("DEBUG_MEM")) {
                spdlog::debug("  evict node pages: [{}]", fmt::join(GetResource<RType>(leaf).Pages(), ", "));
            }
        }

        auto resource_ptr = leaf->DetachResource<RType>();
        if (eviction_callback_) {
            eviction_callback_(leaf);
        }
        OwnedPages pages = resource_ptr->TakePages();
        evicted += pages.Size();
        leaves_.erase(leaf);
        evicted_nodes.push_back(leaf);

        TreeNode* parent = leaf->Parent();
        updateLeaf(parent);
        if (parent != nullptr && IsLeaf<RType>(parent) && GetResource<RType>(parent).IsEvictable()) {
            candidates.push(parent);
        }
    }

    return evicted_nodes;
}

template <ResourceType RType>
void ResourceManager<RType>::EvictSubtree(const std::vector<TreeNode*>& nodes) {
    for (TreeNode* node : nodes) {
        bool has_resource;
        if constexpr (RType == ResourceType::Device) {
            has_resource = node->OnDevice();
        } else {
            has_resource = node->OnHost();
        }
        if (!has_resource) continue;

        const auto& res = GetResource<RType>(node);
        if (!res.IsEvictable()) continue;  // skip locked nodes; freed when request finishes

        leaves_.erase(node);
        auto resource_ptr = node->DetachResource<RType>();
        if (eviction_callback_) {
            eviction_callback_(node);
        }
        // OwnedPages RAII: pages returned to allocator on scope exit.
    }
}

template <ResourceType RType>
std::vector<TreeNode*> ResourceManager<RType>::EnsureCapacity(std::int32_t required_num_pages) {
    if (required_num_pages <= 0) {
        return {};
    }
    const std::int32_t available = allocator_->AvailablePages();
    if (available >= required_num_pages) {
        return {};
    }
    return Evict(required_num_pages - available);
}

}  // namespace tokenspeed
