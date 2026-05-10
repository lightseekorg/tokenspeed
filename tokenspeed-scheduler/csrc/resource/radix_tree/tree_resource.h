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
#include <functional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/types.h"

namespace tokenspeed {

class TreeNode;

// Forward declaration so NodeResource can store a typed back-pointer without
// a std::function closure (avoids heap allocation and the layering ambiguity).
template <ResourceType RType>
class ResourceManager;

template <ResourceType RType>
class NodeResource {
public:
    explicit NodeResource(OwnedPages pages) : pages_{std::move(pages)} {}

    NodeResource(OwnedPages pages, std::int32_t ref_count) : pages_{std::move(pages)}, ref_count_{ref_count} {}

    void Lock() {
        _assert(ref_count_ >= 0, "ref_count must >= 0");
        ref_count_ = ref_count_ + 1;
    }

    // Defined below after ResourceManager is complete (needs OnNodeEvictable).
    void Unlock();

    // Called by ResourceManager when this node enters lru_leaves_.
    // On the last Unlock(), OnNodeEvictable is called so the manager can
    // refresh the LRU sort key with the post-Touch timestamp.
    void BindEvictableNotifier(ResourceManager<RType>* mgr, TreeNode* node) {
        evict_notifier_ = mgr;
        owner_node_ = node;
    }
    void ClearEvictableNotifier() {
        evict_notifier_ = nullptr;
        owner_node_ = nullptr;
    }

    bool IsEmpty() const { return pages_.Empty(); }
    const std::vector<std::int32_t>& Pages() const { return pages_.Ids(); }
    std::int32_t NumPages() const { return pages_.Size(); }
    std::int32_t RefCount() const { return ref_count_; }

    bool IsEvictable() const { return ref_count_ == 0; }

    OwnedPages TakePages() { return std::move(pages_); }

    OwnedPages SplitFirst(std::int32_t n) { return pages_.TakeFirst(n); }

    NodeResource(const NodeResource& other) = delete;
    NodeResource& operator=(const NodeResource& other) = delete;
    NodeResource(NodeResource&& other) = default;
    NodeResource& operator=(NodeResource&& other) = default;

private:
    OwnedPages pages_{};
    std::int32_t ref_count_{0};
    ResourceManager<RType>* evict_notifier_{nullptr};
    TreeNode* owner_node_{nullptr};
};

using DeviceResource = NodeResource<ResourceType::Device>;
using HostResource = NodeResource<ResourceType::Host>;

template <ResourceType RType>
class ResourceManager {
public:
    using EvictionCallback = std::function<void(TreeNode*)>;
    using timestamp_t = std::chrono::steady_clock::time_point;

    explicit ResourceManager(PageAllocator* allocator) : allocator_(allocator) {}

    void SetEvictionCallback(EvictionCallback cb) { eviction_callback_ = std::move(cb); }

    void UpdateLeaves(TreeNode* node);
    std::vector<TreeNode*> Evict(std::int32_t num_pages);
    std::vector<TreeNode*> EnsureCapacity(std::int32_t required_num_pages);

    // Called by NodeResource::Unlock() when ref_count transitions 1→0.
    void OnNodeEvictable(TreeNode* node) { updateLeaf(node); }

    OwnedPages Allocate(std::int32_t num_pages) { return allocator_->Allocate(num_pages); }
    std::int32_t AvailablePages() const { return allocator_->AvailablePages(); }

    // O(N) scan — locked leaves are in lru_leaves_ but not evictable.
    std::int32_t EvictablePagesNum() const {
        std::int32_t total = 0;
        for (const auto& [ts, node] : lru_leaves_) {
            const auto& res = GetResource<RType>(node);
            if (res.IsEvictable()) {
                total += res.NumPages();
            }
        }
        return total;
    }

private:
    void updateLeaf(TreeNode* node);

    PageAllocator* allocator_;
    // Leaf nodes sorted by last-access time (oldest first = LRU eviction order).
    // node_time_ holds each node's sort key for O(1) keyed removal.
    std::set<std::pair<timestamp_t, TreeNode*>> lru_leaves_;
    std::unordered_map<TreeNode*, timestamp_t> node_time_;
    EvictionCallback eviction_callback_{};
};

using DeviceManager = ResourceManager<ResourceType::Device>;
using HostManager = ResourceManager<ResourceType::Host>;

// Defined here (after ResourceManager is complete) because Unlock() calls
// OnNodeEvictable(), which requires the full ResourceManager definition.
template <ResourceType RType>
inline void NodeResource<RType>::Unlock() {
    _assert(ref_count_ >= 1, "ref_count must >= 1");
    ref_count_ = ref_count_ - 1;
    if (ref_count_ == 0 && evict_notifier_) {
        evict_notifier_->OnNodeEvictable(owner_node_);
    }
}

}  // namespace tokenspeed
