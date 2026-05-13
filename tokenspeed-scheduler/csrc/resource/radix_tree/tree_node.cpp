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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "resource/allocator/owned_pages.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/radix_tree/radix_tree.h"
#include "resource/types.h"
namespace tokenspeed {

TreeNode::TreeNode(token_vec_t tokens, timestamp_t access_time)
    : tokens_{std::move(tokens)}, last_access_time_{access_time} {}

void TreeNode::AddChild(const token_vec_t& key, std::unique_ptr<TreeNode>&& child) {
    if (child == nullptr) [[unlikely]] {
        return;
    }
    child->parent_ = this;
    child->depth_in_tokens_ = depth_in_tokens_ + child->tokens_.size();
    children_[key] = std::move(child);
}

std::unique_ptr<TreeNode> TreeNode::RemoveChild(const token_vec_t& key) {
    auto iter = children_.find(key);
    if (iter == children_.end()) {
        return nullptr;
    }
    std::unique_ptr<TreeNode> child = std::move(iter->second);
    children_.erase(iter);
    return child;
}

void TreeNode::SplitSelfInto(TreeNode& prefix, std::size_t prefix_pages, std::int32_t page_size) {
    const std::size_t total_pages = tokens_.size() / page_size;
    const auto split_at = static_cast<std::ptrdiff_t>(prefix_pages * page_size);

    const std::size_t prefix_token_count = static_cast<std::size_t>(split_at);
    const std::size_t suffix_token_count = tokens_.size() - prefix_token_count;
    const std::size_t this_depth_start = depth_in_tokens_ - tokens_.size();

    prefix.tokens_ = token_vec_t(tokens_.begin(), tokens_.begin() + split_at);
    tokens_ = token_vec_t(tokens_.begin() + split_at, tokens_.end());

    prefix.depth_in_tokens_ = this_depth_start + prefix_token_count;
    depth_in_tokens_ = prefix.depth_in_tokens_ + suffix_token_count;

    if (!page_hashes_.empty()) {
        const std::vector<std::string> old_hashes = page_hashes_;
        prefix.page_hashes_ = SliceStrings(old_hashes, 0, prefix_pages);
        page_hashes_ = SliceStrings(old_hashes, prefix_pages, total_pages - prefix_pages);
    }

    if (block_hashes_.size() == total_pages) {
        const std::vector<std::uint64_t> old_hashes = block_hashes_;
        prefix.block_hashes_ = std::vector<std::uint64_t>(old_hashes.begin(), old_hashes.begin() + prefix_pages);
        block_hashes_ = std::vector<std::uint64_t>(old_hashes.begin() + prefix_pages, old_hashes.end());
    } else {
        prefix.block_hashes_.clear();
        block_hashes_.clear();
    }

    prefix.storage_persisted_ = storage_persisted_;
    prefix.last_access_time_ = last_access_time_;

    if (device_resource_ != nullptr) {
        std::int32_t ref_count = device_resource_->RefCount();
        prefix.AttachResource(std::make_unique<DeviceResource>(device_resource_->SplitFirst(prefix_pages), ref_count));
    }

    if (host_resource_ != nullptr) {
        std::int32_t ref_count = host_resource_->RefCount();
        prefix.AttachResource(std::make_unique<HostResource>(host_resource_->SplitFirst(prefix_pages), ref_count));
    }
    // Mamba stays in suffix.
    // Invariant: snapshot-bearing nodes are never split (RadixTree refuses).
    // A split here would dangle borrowed ids in active requests.
    _assert(paged_cache_snapshot_ == nullptr,
            "TreeNode::SplitSelfInto called on a node with an attached paged-cache snapshot; "
            "splitting would invalidate borrowed page id references in active requests");
}

void TreeNode::AttachPagedCacheSnapshot(std::unique_ptr<PagedCacheSnapshot> snapshot) {
    paged_cache_snapshot_ = std::move(snapshot);
}

std::unique_ptr<PagedCacheSnapshot> TreeNode::DetachPagedCacheSnapshot() {
    return std::move(paged_cache_snapshot_);
}

void TreeNode::SetPersisted(bool persisted) {
    storage_persisted_ = persisted;
}

void TreeNode::Touch(timestamp_t now) {
    last_access_time_ = now;
}

void TreeNode::SetPageHashes(std::vector<std::string> page_hashes) {
    page_hashes_ = std::move(page_hashes);
}

std::optional<std::uint64_t> TreeNode::BlockHash() const {
    if (block_hashes_.empty()) {
        return std::nullopt;
    }
    return block_hashes_.back();
}

void TreeNode::SetBlockHashes(std::vector<std::uint64_t> block_hashes) {
    block_hashes_ = std::move(block_hashes);
}

std::optional<cache_op_id> TreeNode::CacheOpId() const {
    return 0;
}

std::int32_t TreeNode::MambaSlotIndex() const {
    _assert(mamba_slot_ != nullptr, "MambaSlotIndex called on node without mamba");
    return mamba_slot_->Index();
}

}  // namespace tokenspeed
