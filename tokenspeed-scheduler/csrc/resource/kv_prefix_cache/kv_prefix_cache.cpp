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

#include "resource/kv_prefix_cache/kv_prefix_cache.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>

#include "resource/types.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/kv_prefix_cache/eviction.h"
#include "resource/radix_tree/node_range.h"
#include "resource/radix_tree/radix_tree.h"
#include "resource/radix_tree/tree_node.h"

namespace tokenspeed {
namespace {

std::int32_t PageCount(const TreeNode* node, std::int32_t page_size) {
    return static_cast<std::int32_t>(node->Tokens().size()) / page_size;
}

std::optional<std::uint64_t> ParentBlockHash(const TreeNode* node) {
    const TreeNode* parent = node->Parent();
    if (parent == nullptr || parent->IsRoot()) {
        return std::nullopt;
    }
    return parent->BlockHash();
}

std::vector<std::uint64_t> BuildBlockHashesForTokens(const token_vec_t& tokens, std::int32_t page_size,
                                                     std::optional<std::uint64_t> parent_hash) {
    std::vector<std::uint64_t> block_hashes;
    const auto page_count = static_cast<std::int32_t>(tokens.size()) / page_size;
    block_hashes.reserve(page_count);
    for (std::int32_t page = 0; page < page_count; ++page) {
        const auto begin = tokens.begin() + page * page_size;
        token_slice token_page{&*begin, static_cast<std::size_t>(page_size)};
        parent_hash = HashKvBlock(token_page, parent_hash);
        block_hashes.push_back(*parent_hash);
    }
    return block_hashes;
}

void EnsureBlockHashesForNode(TreeNode* node, std::int32_t page_size) {
    if (node == nullptr || node->IsRoot()) {
        return;
    }
    if (node->BlockHashes().size() == static_cast<std::size_t>(PageCount(node, page_size))) {
        return;
    }
    node->SetBlockHashes(BuildBlockHashesForTokens(node->Tokens(), page_size, ParentBlockHash(node)));
}

void EnsureBlockHashesTo(TreeNode* target, std::int32_t page_size) {
    for (TreeNode* node : RootToLeaf(target)) {
        EnsureBlockHashesForNode(node, page_size);
    }
}

std::vector<KvBlockStoredEvent> BuildBlockEventsForNode(TreeNode* target, std::int32_t page_size) {
    std::vector<KvBlockStoredEvent> events;
    if (target == nullptr || target->IsRoot()) {
        return events;
    }

    EnsureBlockHashesForNode(target, page_size);

    std::optional<std::uint64_t> parent_hash = ParentBlockHash(target);
    const auto& tokens = target->Tokens();
    const auto& block_hashes = target->BlockHashes();
    const std::int32_t page_count = PageCount(target, page_size);
    events.reserve(page_count);
    for (std::int32_t page = 0; page < page_count; ++page) {
        const auto begin = tokens.begin() + page * page_size;
        const std::uint64_t block_hash = block_hashes[page];
        events.push_back(KvBlockStoredEvent{
            .block_hashes = {block_hash},
            .parent_block_hash = parent_hash,
            .token_ids = token_vec_t(begin, begin + page_size),
            .block_size = page_size,
        });
        parent_hash = block_hash;
    }
    return events;
}

}  // namespace

KVPrefixCache::KVPrefixCache(PageAllocator* device_allocator, PageAllocator* host_allocator, bool enable_l3_storage)
    : tree_(device_allocator->PageSize()),
      device_(device_allocator),
      host_(host_allocator),
      enable_l3_storage_(enable_l3_storage) {}

TreeNode* KVPrefixCache::getOrCreateLoraRoot(std::int32_t lora_id) {
    auto& slot = lora_virtual_roots_[lora_id];
    // Re-create if null or if the node was pruned from the tree (parent == nullptr
    // while not the real root means it was removed by PruneEmptyByNode).
    if (slot != nullptr && slot->Parent() != nullptr) {
        return slot;
    }
    // Sentinel page: [-lora_id, 0, ..., 0].  Negative token IDs never appear in
    // real vocabularies (which are always non-negative), so there is no collision.
    const std::int32_t page_size = tree_.PageSize();
    token_vec_t sentinel(page_size, 0);
    sentinel[0] = -lora_id;
    auto node = std::make_unique<TreeNode>(sentinel, std::chrono::steady_clock::now());
    TreeNode* raw = node.get();
    // Attach an empty DeviceResource so OnDevice() returns true.
    // This prevents PruneEmptyByNode from removing the virtual root even when
    // all adapter sequences have been evicted.
    raw->AttachResource<ResourceType::Device>(std::make_unique<NodeResource<ResourceType::Device>>(OwnedPages{}));
    token_vec_t key(sentinel.begin(), sentinel.begin() + page_size);
    tree_.Root()->AddChild(key, std::move(node));
    slot = raw;
    return raw;
}

void KVPrefixCache::SetKvEventSink(KvEventSink sink) {
    kv_event_sink_ = std::move(sink);
    if (!kv_event_sink_) {
        published_device_blocks_.clear();
    }
}

void KVPrefixCache::recordDeviceBlockStored(TreeNode* node) {
    if (!kv_event_sink_) {
        return;
    }
    for (auto& event : BuildBlockEventsForNode(node, tree_.PageSize())) {
        const std::uint64_t block_hash = event.block_hashes.front();
        if (published_device_blocks_.insert(block_hash).second) {
            kv_event_sink_(KvCacheEvent{std::move(event)});
        }
    }
}

void KVPrefixCache::recordDeviceBlockRemoved(TreeNode* node) {
    if (!kv_event_sink_) {
        return;
    }
    std::vector<std::uint64_t> removed_hashes;
    for (const auto& event : BuildBlockEventsForNode(node, tree_.PageSize())) {
        const std::uint64_t block_hash = event.block_hashes.front();
        if (published_device_blocks_.erase(block_hash) > 0) {
            removed_hashes.push_back(block_hash);
        }
    }
    if (!removed_hashes.empty()) {
        kv_event_sink_(KvCacheEvent{KvBlockRemovedEvent{.block_hashes = std::move(removed_hashes)}});
    }
}

MatchResult KVPrefixCache::Match(const token_vec_t& token_ids, std::int32_t lora_id) {
    const auto access_time = std::chrono::steady_clock::now();
    const std::int32_t page_size = tree_.PageSize();
    if (token_ids.size() % page_size != 0) {
        throw std::runtime_error("KVPrefixCache::Match: token count must be divisible by page_size; token_count=" +
                                 std::to_string(token_ids.size()) + "; page_size=" + std::to_string(page_size));
    }

    TreeNode* start_node = resolveStartNode(lora_id);
    WalkResult walk_result = tree_.WalkDownUtilMismatch(token_ids, access_time, start_node);
    MatchResult& match = walk_result.match;
    match.device.page_size = page_size;
    match.host.page_size = page_size;
    if (lora_id != kLoraNone) {
        // The virtual namespace root contributes 1 sentinel page to absolute tree
        // depth.  Subtract it so callers see the number of real matched token pages.
        match.device.namespace_depth_offset = 1;
        match.host.namespace_depth_offset = 1;
    }
    return match;
}

MatchResult KVPrefixCache::Match(const std::vector<std::span<const std::int32_t>>& token_pages, std::int32_t lora_id) {
    return Match(FlattenPages(token_pages, 0, token_pages.size()), lora_id);
}

template <ResourceType RType>
InsertResult KVPrefixCache::Insert(const token_vec_t& token_ids, const std::vector<std::int32_t>& prefix_pages,
                                   OwnedPages allocator_pages, const std::vector<std::string>& page_hashs,
                                   TreeNode* start_node, std::int32_t lora_id) {
    const std::int32_t page_size = tree_.PageSize();
    auto insert_result = InsertResult{
        .last_node = tree_.Root(),
        .inserted_num_pages = 0,
    };
    if (token_ids.size() % page_size != 0) {
        throw std::runtime_error("KVPrefixCache::Insert: token count must be divisible by page_size; token_count=" +
                                 std::to_string(token_ids.size()) + "; page_size=" + std::to_string(page_size));
    }
    std::size_t total_pages = token_ids.size() / page_size;
    if (total_pages == 0) {
        return insert_result;
    }
    auto access_time = std::chrono::steady_clock::now();

    std::vector<std::int32_t> page_ids = prefix_pages;
    const auto& alloc_ids = allocator_pages.Ids();
    page_ids.insert(page_ids.end(), alloc_ids.begin(), alloc_ids.end());

    // When start_node is nullptr (no prior match), resolve the LoRA namespace root.
    // When start_node is provided (continuation from a prior match), the caller
    // already points into the correct namespace subtree.
    TreeNode* effective_start = (start_node != nullptr) ? start_node : resolveStartNode(lora_id);
    WalkResult walk_result = tree_.WalkDownUtilMismatch(token_slice{token_ids.data(), total_pages * page_size},
                                                        access_time, effective_start);

    token_slice mistmatched_tokens = walk_result.remaining_tokens;
    TreeNode* current = walk_result.terminal;

    if (!mistmatched_tokens.empty()) {
        auto node =
            std::make_unique<TreeNode>(token_vec_t(mistmatched_tokens.begin(), mistmatched_tokens.end()), access_time);
        TreeNode* last_node = node.get();
        current->AddChild(token_vec_t(mistmatched_tokens.begin(), mistmatched_tokens.begin() + page_size),
                          std::move(node));
        current = last_node;
    }

    insert_result.last_node = current;
    if constexpr (RType == ResourceType::Device) {
        if (kv_event_sink_) {
            EnsureBlockHashesTo(current, page_size);
        }
    }

    auto already_has_pages = [](TreeNode* node) -> bool {
        return (RType == ResourceType::Device) ? node->OnDevice() : node->OnHost();
    };

    auto update_leaves_set = [this](TreeNode* node) {
        if constexpr (RType == ResourceType::Device) {
            device_.UpdateLeaves(node);
        } else {
            host_.UpdateLeaves(node);
        }
    };

    const std::int32_t alloc_start = static_cast<std::int32_t>(prefix_pages.size());
    OwnedPages unconsumed;
    std::vector<TreeNode*> newly_stored_device_nodes;

    std::int32_t remaining_pages = total_pages;
    for (TreeNode* node : LeafToRoot(current)) {
        std::int32_t node_num_pages = node->Tokens().size() / page_size;
        if (remaining_pages < node_num_pages) {
            std::int32_t alloc_overlap = std::max(0, remaining_pages - std::max(0, alloc_start));
            if (alloc_overlap > 0) {
                unconsumed.Append(allocator_pages.TakeFirst(alloc_overlap));
            }
            break;
        }
        remaining_pages -= node_num_pages;

        std::int32_t node_end = remaining_pages + node_num_pages;
        std::int32_t overlap_start = std::max(remaining_pages, alloc_start);
        std::int32_t alloc_overlap = std::max(0, node_end - overlap_start);

        if (already_has_pages(node)) {
            if (alloc_overlap > 0) {
                unconsumed.Append(allocator_pages.TakeLast(alloc_overlap));
            }
            continue;
        }

        if (!page_hashs.empty()) {
            node->SetPageHashes(SliceStrings(page_hashs, remaining_pages, node_num_pages));
        }

        if constexpr (RType == ResourceType::Device) {
            if (std::getenv("DEBUG_MEM")) {
                auto inserted_pages = SliceInts(page_ids, remaining_pages, node_num_pages);
                spdlog::debug("[InsertDevice] node inserted pages: [{}]", fmt::join(inserted_pages, ", "));
            }
        }

        assert(alloc_overlap == node_num_pages);
        OwnedPages node_owned = allocator_pages.TakeLast(node_num_pages);
        node->AttachResource(std::make_unique<NodeResource<RType>>(std::move(node_owned)));

        update_leaves_set(node);
        insert_result.inserted_num_pages += node_num_pages;
        if constexpr (RType == ResourceType::Device) {
            if (kv_event_sink_) {
                newly_stored_device_nodes.push_back(node);
            }
        }
    }

    if constexpr (RType == ResourceType::Device) {
        std::reverse(newly_stored_device_nodes.begin(), newly_stored_device_nodes.end());
        for (TreeNode* node : newly_stored_device_nodes) {
            recordDeviceBlockStored(node);
        }
    }

    return insert_result;
}

template <ResourceType RType>
InsertResult KVPrefixCache::Insert(const std::vector<std::span<const std::int32_t>>& token_pages,
                                   const std::vector<std::int32_t>& prefix_pages, OwnedPages allocator_pages,
                                   const std::vector<std::string>& page_hashs, TreeNode* start_node,
                                   std::int32_t lora_id) {
    return Insert<RType>(FlattenPages(token_pages, 0, token_pages.size()), prefix_pages, std::move(allocator_pages),
                         page_hashs, start_node, lora_id);
}

template <ResourceType RType>
void KVPrefixCache::pruneEvicted(const std::vector<TreeNode*>& evicted) {
    std::unordered_set<TreeNode*> evicted_set(evicted.begin(), evicted.end());
    for (TreeNode* node : evicted) {
        if (evicted_set.count(node->Parent())) {
            continue;
        }
        TreeNode* survivor = tree_.PruneEmptyByNode(node);
        if (survivor != nullptr) {
            if constexpr (RType == ResourceType::Device) {
                device_.UpdateLeaves(survivor);
            } else {
                host_.UpdateLeaves(survivor);
            }
        }
    }
}

template <ResourceType RType>
bool KVPrefixCache::EnsureCapacityByEvict(std::int32_t required_num_pages) {
    auto& manager = getResourceManager<RType>();
    auto evicted = manager.EnsureCapacity(required_num_pages);
    if constexpr (RType == ResourceType::Device) {
        for (TreeNode* node : evicted) {
            recordDeviceBlockRemoved(node);
        }
    }
    pruneEvicted<RType>(evicted);
    return manager.AvailablePages() >= required_num_pages;
}

cache_op_id KVPrefixCache::AllocateCacheOpId() {
    return next_op_id_++;
}

void KVPrefixCache::EvictLoraNamespace(std::int32_t lora_id) {
    auto it = lora_virtual_roots_.find(lora_id);
    if (it == lora_virtual_roots_.end() || it->second == nullptr) {
        return;
    }
    TreeNode* vroot = it->second;

    // Collect all descendant nodes via DFS (excluding the virtual root itself,
    // which holds no real KV pages).
    std::vector<TreeNode*> descendants;
    std::function<void(TreeNode*)> collect = [&](TreeNode* node) {
        for (auto& [key, child] : node->Children()) {
            if (!child) continue;
            descendants.push_back(child.get());
            collect(child.get());
        }
    };
    collect(vroot);

    // Evict device and host pages. OwnedPages RAII returns them to the allocator.
    device_.EvictSubtree(descendants);
    host_.EvictSubtree(descendants);

    // Remove the virtual root from the tree. The unique_ptr cascade destroys the
    // entire subtree (including any mamba slots attached to those nodes).
    token_vec_t sentinel(tree_.PageSize(), 0);
    sentinel[0] = -lora_id;
    tree_.Root()->RemoveChild(sentinel);

    lora_virtual_roots_.erase(it);
}

template InsertResult KVPrefixCache::Insert<ResourceType::Device>(const token_vec_t&, const std::vector<std::int32_t>&,
                                                                  OwnedPages, const std::vector<std::string>&,
                                                                  TreeNode*, std::int32_t);
template InsertResult KVPrefixCache::Insert<ResourceType::Host>(const token_vec_t&, const std::vector<std::int32_t>&,
                                                                OwnedPages, const std::vector<std::string>&, TreeNode*,
                                                                std::int32_t);
template InsertResult KVPrefixCache::Insert<ResourceType::Device>(const std::vector<std::span<const std::int32_t>>&,
                                                                  const std::vector<std::int32_t>&, OwnedPages,
                                                                  const std::vector<std::string>&, TreeNode*,
                                                                  std::int32_t);
template InsertResult KVPrefixCache::Insert<ResourceType::Host>(const std::vector<std::span<const std::int32_t>>&,
                                                                const std::vector<std::int32_t>&, OwnedPages,
                                                                const std::vector<std::string>&, TreeNode*,
                                                                std::int32_t);

template bool KVPrefixCache::EnsureCapacityByEvict<ResourceType::Device>(std::int32_t required_num_pages);
template bool KVPrefixCache::EnsureCapacityByEvict<ResourceType::Host>(std::int32_t required_num_pages);

template <ResourceType RType>
bool KVPrefixCache::AllocateResourceOfType(const std::vector<TreeNode*>& nodes) {
    constexpr ResourceType ResourceTypeAlreadyHas =
        (RType == ResourceType::Device) ? ResourceType::Host : ResourceType::Device;

    std::int32_t total_pages = 0;
    for (TreeNode* node : nodes) {
        total_pages += GetResource<ResourceTypeAlreadyHas>(node).NumPages();
    }
    OwnedPages all_pages = getResourceManager<RType>().Allocate(total_pages);
    if (all_pages.Size() < total_pages) {
        spdlog::error("[AllocateResourceOfType] Allocate returned {} pages, expected {}; aborting node attachment",
                      all_pages.Size(), total_pages);
        return false;
    }

    for (TreeNode* node : nodes) {
        const std::int32_t n = GetResource<ResourceTypeAlreadyHas>(node).NumPages();
        OwnedPages node_pages = all_pages.TakeFirst(n);
        node->AttachResource(std::make_unique<NodeResource<RType>>(std::move(node_pages)));
        if constexpr (RType == ResourceType::Device) {
            device_.UpdateLeaves(node);
        } else {
            host_.UpdateLeaves(node);
        }
    }

    if constexpr (RType == ResourceType::Device) {
        if (kv_event_sink_ && !nodes.empty()) {
            std::vector<TreeNode*> published_nodes = nodes;
            std::sort(published_nodes.begin(), published_nodes.end(), [](const TreeNode* lhs, const TreeNode* rhs) {
                return lhs->DepthInTokens() < rhs->DepthInTokens();
            });
            EnsureBlockHashesTo(published_nodes.back(), tree_.PageSize());
            for (TreeNode* node : published_nodes) {
                recordDeviceBlockStored(node);
            }
        }
    }
    return true;
}

template bool KVPrefixCache::AllocateResourceOfType<ResourceType::Device>(const std::vector<TreeNode*>&);
template bool KVPrefixCache::AllocateResourceOfType<ResourceType::Host>(const std::vector<TreeNode*>&);

// Helpers: DFS from a node and collect pages for a given resource type.
template <ResourceType RType>
static void collectTreePages(const TreeNode* node, std::unordered_map<std::int32_t, int>& out) {
    if (node == nullptr) return;
    if (!node->IsRoot()) {
        bool has_resource = (RType == ResourceType::Device) ? node->OnDevice() : node->OnHost();
        if (has_resource) {
            const auto& pages = GetResource<RType>(node).Pages();
            for (std::int32_t p : pages) {
                out[p]++;
            }
        }
    }
    for (const auto& [key, child] : node->Children()) {
        collectTreePages<RType>(child.get(), out);
    }
}

template <ResourceType RType>
std::unordered_map<std::int32_t, int> KVPrefixCache::CollectAllPages() const {
    std::unordered_map<std::int32_t, int> result;
    collectTreePages<RType>(tree_.Root(), result);
    return result;
}

template std::unordered_map<std::int32_t, int> KVPrefixCache::CollectAllPages<ResourceType::Device>() const;
template std::unordered_map<std::int32_t, int> KVPrefixCache::CollectAllPages<ResourceType::Host>() const;

}  // namespace tokenspeed
