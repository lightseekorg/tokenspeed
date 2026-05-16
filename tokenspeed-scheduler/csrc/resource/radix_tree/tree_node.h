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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "resource/radix_tree/mamba_slot.h"
#include "resource/radix_tree/tree_resource.h"
#include "resource/types.h"

namespace tokenspeed {

// normal fast hash
struct TokenVecHash {
    std::size_t operator()(const token_vec_t& vec) const {
        std::size_t hash = 0;
        for (const auto& token : vec) {
            hash ^= token + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

class TreeNode {
public:
    using children_map_t = std::unordered_map<token_vec_t, std::unique_ptr<TreeNode>, TokenVecHash>;
    using timestamp_t = std::chrono::steady_clock::time_point;

    explicit TreeNode(token_vec_t tokens = {}, timestamp_t access_time = std::chrono::steady_clock::now());
    ~TreeNode() = default;

    TreeNode(const TreeNode&) = delete;
    TreeNode& operator=(const TreeNode&) = delete;

    bool IsRoot() const { return parent_ == nullptr; }
    bool OnDevice() const { return device_resource_ != nullptr; }
    bool OnHost() const { return host_resource_ != nullptr; }

    std::int32_t DepthInPage(std::int32_t page_size) const {
        return static_cast<std::int32_t>(depth_in_tokens_) / page_size;
    }
    std::size_t DepthInTokens() const { return depth_in_tokens_; }

    TreeNode* Parent() const { return parent_; }
    const token_vec_t& Tokens() const { return tokens_; }
    const std::vector<std::string>& PageHashes() const { return page_hashes_; }
    const std::vector<std::uint64_t>& BlockHashes() const { return block_hashes_; }
    std::optional<std::uint64_t> BlockHash() const;
    const std::size_t NumChildren() const { return children_.size(); }
    const children_map_t& Children() const { return children_; }
    timestamp_t Time() const { return last_access_time_; }

    template <ResourceType RType>
    void AttachResource(std::unique_ptr<NodeResource<RType>>&& resource) {
        if constexpr (RType == ResourceType::Device) {
            device_resource_ = std::move(resource);
        } else if constexpr (RType == ResourceType::Host) {
            host_resource_ = std::move(resource);
        }
    }

    // Detach and return the resource unique_ptr, making OnDevice()/OnHost() return false.
    template <ResourceType RType>
    std::unique_ptr<NodeResource<RType>> DetachResource() {
        if constexpr (RType == ResourceType::Device) {
            return std::move(device_resource_);
        } else {
            return std::move(host_resource_);
        }
    }

    DeviceResource& Device() { return *device_resource_; }
    const DeviceResource& Device() const { return *device_resource_; }
    HostResource& Host() { return *host_resource_; }
    const HostResource& Host() const { return *host_resource_; }

    bool HasMamba() const { return HasMambaDevice(); }
    bool HasMambaDevice() const { return device_mamba_slot_ != nullptr; }
    bool HasMambaHost() const { return host_mamba_slot_ != nullptr; }
    bool HasAlignedMambaDevice() const { return HasMambaDevice() && device_mamba_depth_tokens_ == depth_in_tokens_; }
    bool HasAlignedMambaHost() const { return HasMambaHost() && host_mamba_depth_tokens_ == depth_in_tokens_; }
    std::int32_t MambaSlotIndex() const;
    std::int32_t MambaHostSlotIndex() const;
    std::size_t MambaDeviceDepthTokens() const { return device_mamba_depth_tokens_; }
    std::size_t MambaHostDepthTokens() const { return host_mamba_depth_tokens_; }
    void AttachMamba(std::unique_ptr<MambaSlot> slot) { AttachMambaDevice(std::move(slot)); }
    void AttachMambaDevice(std::unique_ptr<MambaSlot> slot) { AttachMambaDevice(std::move(slot), depth_in_tokens_); }
    void AttachMambaDevice(std::unique_ptr<MambaSlot> slot, std::size_t depth_tokens) {
        device_mamba_slot_ = std::move(slot);
        device_mamba_depth_tokens_ = depth_tokens;
    }
    void AttachMambaHost(std::unique_ptr<MambaSlot> slot) { AttachMambaHost(std::move(slot), depth_in_tokens_); }
    void AttachMambaHost(std::unique_ptr<MambaSlot> slot, std::size_t depth_tokens) {
        host_mamba_slot_ = std::move(slot);
        host_mamba_depth_tokens_ = depth_tokens;
    }
    std::unique_ptr<MambaSlot> DetachMamba() { return DetachMambaDevice(); }
    std::unique_ptr<MambaSlot> DetachMambaDevice() {
        device_mamba_depth_tokens_ = 0;
        return std::move(device_mamba_slot_);
    }
    std::unique_ptr<MambaSlot> DetachMambaHost() {
        host_mamba_depth_tokens_ = 0;
        return std::move(host_mamba_slot_);
    }

    std::optional<cache_op_id> CacheOpId() const;

    void SetPersisted(bool persisted = true);
    void Touch(timestamp_t now = std::chrono::steady_clock::now());
    void SetPageHashes(std::vector<std::string> page_hashes);
    void SetBlockHashes(std::vector<std::uint64_t> block_hashes);
    void AddChild(const token_vec_t& key, std::unique_ptr<TreeNode>&& child);
    std::unique_ptr<TreeNode> RemoveChild(const token_vec_t& key);

    void SplitSelfInto(TreeNode& prefix, std::size_t prefix_pages, std::int32_t page_size);

private:
    TreeNode* parent_{};
    children_map_t children_{};
    token_vec_t tokens_{};
    std::size_t depth_in_tokens_{0};
    std::vector<std::string> page_hashes_{};
    std::vector<std::uint64_t> block_hashes_{};
    timestamp_t last_access_time_{std::chrono::steady_clock::now()};
    bool storage_persisted_{false};
    std::unique_ptr<DeviceResource> device_resource_{};
    std::unique_ptr<HostResource> host_resource_{};
    std::unique_ptr<MambaSlot> device_mamba_slot_{};
    std::unique_ptr<MambaSlot> host_mamba_slot_{};
    std::size_t device_mamba_depth_tokens_{0};
    std::size_t host_mamba_depth_tokens_{0};
};

template <ResourceType RType>
class NodeRef {
public:
    explicit NodeRef(TreeNode* node) : node_{node} { Lock(); }

    ~NodeRef() { Unlock(); }

    NodeRef(const NodeRef&) = delete;
    NodeRef& operator=(const NodeRef&) = delete;

    NodeRef(NodeRef&& other) noexcept : node_{std::exchange(other.node_, nullptr)} {}

    NodeRef& operator=(NodeRef&& other) noexcept {
        if (this != &other) {
            Unlock();
            node_ = std::exchange(other.node_, nullptr);
        }
        return *this;
    }
    TreeNode* Node() const { return node_; };

private:
    void Lock() {
        for (TreeNode* c = node_; c != nullptr && !c->IsRoot(); c = c->Parent()) {
            GetResource<RType>(c).Lock();
        }
    }

    void Unlock() {
        for (TreeNode* c = node_; c != nullptr && !c->IsRoot(); c = c->Parent()) {
            auto& resource = GetResource<RType>(c);
            if (resource.RefCount() > 0) {
                resource.Unlock();
            }
        }
    }

private:
    TreeNode* node_{};
};

using DeviceNodeRef = NodeRef<ResourceType::Device>;
using HostNodeRef = NodeRef<ResourceType::Host>;

template <ResourceType RType>
inline NodeResource<RType>& GetResource(TreeNode* node) {
    if constexpr (RType == ResourceType::Device) {
        return node->Device();
    } else {
        return node->Host();
    }
}
template <ResourceType RType>
const NodeResource<RType>& GetResource(const TreeNode* node) {
    return GetResource<RType>(const_cast<TreeNode*>(node));
}

inline TreeNode* FindChild(const TreeNode* parent, const token_vec_t& key) {
    auto& children = parent->Children();
    auto iter = children.find(key);
    return iter != children.end() ? iter->second.get() : nullptr;
}
}  // namespace tokenspeed
