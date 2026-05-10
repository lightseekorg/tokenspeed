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

    // Remove stale entry (if any) using the stored sort key for O(1) erase.
    auto it = node_time_.find(node);
    if (it != node_time_.end()) {
        lru_leaves_.erase({it->second, node});
        node_time_.erase(it);
        GetResource<RType>(node).ClearEvictableNotifier();
    }

    if (IsLeaf<RType>(node)) {
        auto ts = node->Time();
        lru_leaves_.insert({ts, node});
        node_time_[node] = ts;
        // When the last lock on this node is released, OnNodeEvictable refreshes
        // the LRU sort key so Touch() calls made while locked are reflected.
        GetResource<RType>(node).BindEvictableNotifier(this, node);
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

    // Leaf nodes locked by active requests: deferred until after the eviction loop.
    // In practice at most max_batch_size nodes are locked at once.
    std::vector<std::pair<timestamp_t, TreeNode*>> deferred_locked;

    std::int32_t evicted = 0;
    while (evicted < num_pages && !lru_leaves_.empty()) {
        auto it = lru_leaves_.begin();  // oldest (LRU) first
        TreeNode* leaf = it->second;
        timestamp_t ts = it->first;
        lru_leaves_.erase(it);
        node_time_.erase(leaf);

        if (!GetResource<RType>(leaf).IsEvictable()) {
            // Locked by an active request — skip for now, restore afterward.
            deferred_locked.push_back({ts, leaf});
            continue;
        }

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
        evicted_nodes.push_back(leaf);

        // Parent may have become a new leaf — updateLeaf inserts it into lru_leaves_
        // with its current timestamp so the outer loop picks it up naturally.
        TreeNode* parent = leaf->Parent();
        updateLeaf(parent);
    }

    // Restore locked nodes so they remain candidates for future eviction calls.
    // Use node->Time() (not the saved ts) so any Touch() that occurred while
    // the node was locked is reflected in LRU order immediately.
    for (auto& [ts, node] : deferred_locked) {
        auto current_ts = node->Time();
        lru_leaves_.insert({current_ts, node});
        node_time_[node] = current_ts;
        GetResource<RType>(node).BindEvictableNotifier(this, node);
    }

    return evicted_nodes;
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
