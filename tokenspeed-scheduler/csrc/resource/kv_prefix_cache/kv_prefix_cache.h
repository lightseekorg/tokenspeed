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
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// kLoraNone is the lora_id value meaning "base model, no adapter".
// Adapter IDs are positive integers assigned by LoraRegistry.
static constexpr std::int32_t kLoraNone = 0;

#include "resource/radix_tree/radix_tree.h"
#include "resource/radix_tree/tree_resource.h"
#include "resource/types.h"
#include "scheduler/kv_cache_events.h"

namespace tokenspeed {

class OwnedPages;
class PageAllocator;
class TreeNode;

using KvEventSink = std::function<void(KvCacheEvent)>;

class KVPrefixCache {
public:
    KVPrefixCache(PageAllocator* device_allocator, PageAllocator* host_allocator, bool enable_l3_storage = false);

    void SetKvEventSink(KvEventSink sink);

    // lora_id = kLoraNone (0) → base model, uses the shared radix tree root.
    // lora_id > 0          → adapter namespace; a per-adapter virtual root is
    //                        created on demand so same-adapter requests share the
    //                        prefix cache while cross-adapter requests never collide.
    MatchResult Match(const token_vec_t& token_ids, std::int32_t lora_id = kLoraNone);
    MatchResult Match(const std::vector<std::span<const std::int32_t>>& token_pages, std::int32_t lora_id = kLoraNone);

    template <ResourceType RType>
    InsertResult Insert(const token_vec_t& token_ids, const std::vector<std::int32_t>& prefix_pages,
                        OwnedPages allocator_pages = {}, const std::vector<std::string>& page_hashs = {},
                        TreeNode* start_node = nullptr, std::int32_t lora_id = kLoraNone);

    template <ResourceType RType>
    InsertResult Insert(const std::vector<std::span<const std::int32_t>>& token_pages,
                        const std::vector<std::int32_t>& prefix_pages, OwnedPages allocator_pages = {},
                        const std::vector<std::string>& page_hashs = {}, TreeNode* start_node = nullptr,
                        std::int32_t lora_id = kLoraNone);

    cache_op_id AllocateCacheOpId();

    template <ResourceType RType>
    bool EnsureCapacityByEvict(std::int32_t required_num_pages);

    void EnqueueTransfer(TreeNode* last_node);

    template <ResourceType RType>
    bool AllocateResourceOfType(const std::vector<TreeNode*>& nodes);

    // DFS-traverse the entire radix tree and collect all page IDs for the given resource type.
    // Returns a map from page_id to occurrence count (should always be 1 for a valid tree).
    template <ResourceType RType>
    std::unordered_map<std::int32_t, int> CollectAllPages() const;

    std::int32_t PageSize() const { return tree_.PageSize(); }
    DeviceManager& GetDeviceManager() { return device_; }

    // Evict all KV pages cached under the given adapter's namespace and remove
    // the virtual root from the tree. Call this when an adapter is unloaded so
    // its pages are freed immediately rather than waiting for LRU pressure.
    // Locked pages (in-flight requests) are skipped and freed when those
    // requests finish.
    void EvictLoraNamespace(std::int32_t lora_id);

private:
    template <ResourceType RType>
    void pruneEvicted(const std::vector<TreeNode*>& evicted);

    void recordDeviceBlockStored(TreeNode* node);
    void recordDeviceBlockRemoved(TreeNode* node);

    template <ResourceType RType>
    auto& getResourceManager() {
        if constexpr (RType == ResourceType::Device) {
            return device_;
        } else {
            return host_;
        }
    }

    // Returns (or creates) the virtual root node for the given LoRA adapter.
    // The virtual root is a child of the real root keyed by a sentinel page
    // [-lora_id, 0, ..., 0] that is outside any real vocabulary range.
    // An empty DeviceResource is attached so PruneEmptyByNode never removes it.
    TreeNode* getOrCreateLoraRoot(std::int32_t lora_id);

    // Resolve the start_node for Match/Insert: nullptr for base model,
    // per-adapter virtual root for LoRA.
    TreeNode* resolveStartNode(std::int32_t lora_id) {
        return (lora_id == kLoraNone) ? nullptr : getOrCreateLoraRoot(lora_id);
    }

    RadixTree tree_;
    DeviceManager device_;
    HostManager host_;
    cache_op_id next_op_id_{1};
    bool enable_l3_storage_{false};
    // Per-adapter virtual root nodes; keyed by lora_id (> 0).
    std::unordered_map<std::int32_t, TreeNode*> lora_virtual_roots_;
    KvEventSink kv_event_sink_{};
    std::unordered_set<std::uint64_t> published_device_blocks_;
};

}  // namespace tokenspeed
