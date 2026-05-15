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
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "resource/types.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/radix_tree/radix_tree.h"

namespace tokenspeed {

inline token_vec_t getFirstPage(const token_vec_t& tokens, std::int32_t page_size) {
    return token_vec_t(tokens.begin(), tokens.begin() + page_size);
}

token_vec_t FlattenPages(const std::vector<std::span<const std::int32_t>>& pages, std::size_t start_page,
                         std::size_t num_pages) {
    token_vec_t out;
    for (std::size_t i = 0; i < num_pages; ++i) {
        const auto& page = pages[start_page + i];
        out.insert(out.end(), page.begin(), page.end());
    }
    return out;
}

std::size_t calcMatchedPages(TreeNode* node, token_slice remaining_tokens, std::int32_t page_size) {
    const auto& node_tokens = node->Tokens();
    const std::size_t comparable_tokens = std::min(node_tokens.size(), remaining_tokens.size());
    if (comparable_tokens == 0) {
        return 0;
    }

    auto [node_it, _] =
        std::mismatch(node_tokens.begin(), node_tokens.begin() + static_cast<std::ptrdiff_t>(comparable_tokens),
                      remaining_tokens.begin());
    const std::size_t matched_tokens = static_cast<std::size_t>(std::distance(node_tokens.begin(), node_it));
    return matched_tokens / page_size;
}

RadixTree::RadixTree(std::int32_t page_size) : page_size_{page_size}, root_{std::make_unique<TreeNode>()} {}

SplitResult RadixTree::splitChild(TreeNode* parent, const token_vec_t& child_key, std::size_t prefix_pages) {
    std::unique_ptr<TreeNode> old_node = parent->RemoveChild(child_key);
    TreeNode* suffix_node = old_node.get();

    auto prefix_node = std::make_unique<TreeNode>();
    suffix_node->SplitSelfInto(*prefix_node, prefix_pages, page_size_);

    TreeNode* prefix_ptr = prefix_node.get();
    prefix_node->AddChild(getFirstPage(suffix_node->Tokens(), page_size_), std::move(old_node));
    parent->AddChild(getFirstPage(prefix_node->Tokens(), page_size_), std::move(prefix_node));

    return SplitResult{.prefix = prefix_ptr, .suffix = suffix_node};
}

TreeNode* RadixTree::PruneEmptyByNode(TreeNode* node) {
    TreeNode* current = node;
    while (current != nullptr && !current->IsRoot()) {
        if (current->NumChildren() != 0 || current->OnDevice() || current->OnHost()) {
            break;
        }

        TreeNode* parent = current->Parent();
        parent->RemoveChild(getFirstPage(current->Tokens(), page_size_));
        current = parent;
    }
    return current;
}

TreeNode* RadixTree::SplitAt(TreeNode* descendant, std::int32_t depth_in_tokens) {
    if (descendant == nullptr) {
        return nullptr;
    }
    if (depth_in_tokens <= 0 || depth_in_tokens % page_size_ != 0) {
        return nullptr;
    }
    if (depth_in_tokens > static_cast<std::int32_t>(descendant->DepthInTokens())) {
        return nullptr;
    }

    // Find the ancestor range covering depth_in_tokens.
    // Exact match returns the node; an interior split returns the prefix.
    TreeNode* current = descendant;
    while (current != nullptr && !current->IsRoot()) {
        const std::int32_t this_depth = static_cast<std::int32_t>(current->DepthInTokens());
        const std::int32_t parent_depth = this_depth - static_cast<std::int32_t>(current->Tokens().size());
        if (depth_in_tokens == this_depth) {
            return current;
        }
        if (depth_in_tokens > parent_depth && depth_in_tokens < this_depth) {
            // Refuse to split a snapshot-bearing node (would dangle borrowed ids).
            if (current->HasPagedCacheSnapshot()) {
                return nullptr;
            }
            TreeNode* parent = current->Parent();
            const token_vec_t child_key = getFirstPage(current->Tokens(), page_size_);
            const std::size_t prefix_pages =
                static_cast<std::size_t>((depth_in_tokens - parent_depth) / page_size_);
            SplitResult sr = splitChild(parent, child_key, prefix_pages);
            return sr.prefix;
        }
        current = current->Parent();
    }
    return nullptr;
}

WalkResult RadixTree::WalkDownUtilMismatch(token_slice aligned_tokens, TreeNode::timestamp_t access_time,
                                           TreeNode* start_node) {
    TreeNode* current = (start_node != nullptr) ? start_node : root_.get();

    WalkResult result{
        .terminal = current,
        .remaining_tokens = aligned_tokens,
        .match =
            {
                .device = {.last_node = current},
                .host = {.last_node = current},
            },
    };

    bool device_alive = true;
    bool host_alive = true;
    token_vec_t walk_key_cache;
    walk_key_cache.reserve(page_size_);

    // Update a single tier's match info; clears alive flag if the node is no longer on that tier.
    auto update_tier = [](bool& alive, auto& info, TreeNode* child, bool on_tier) {
        if (!alive) {
            return;
        }
        if (on_tier) {
            info.last_node = child;
        } else {
            alive = false;
        }
    };

    while (result.remaining_tokens.size() >= static_cast<std::size_t>(page_size_)) {
        walk_key_cache.assign(result.remaining_tokens.begin(), result.remaining_tokens.begin() + page_size_);
        TreeNode* child = FindChild(current, walk_key_cache);
        if (child == nullptr) {
            break;
        }
        const std::int32_t matched_num_pages = calcMatchedPages(child, result.remaining_tokens, page_size_);
        if (matched_num_pages == 0) {
            break;
        }
        if (matched_num_pages != static_cast<std::int32_t>(child->Tokens().size() / page_size_)) {
            // Refuse to split a snapshot-bearing node; borrowed ids rely on it.
            if (child->HasPagedCacheSnapshot()) {
                break;
            }
            SplitResult split = splitChild(current, walk_key_cache, matched_num_pages);
            child = split.prefix;
        }

        child->Touch(access_time);

        update_tier(device_alive, result.match.device, child, child->OnDevice());
        update_tier(host_alive, result.match.host, child, child->OnHost());

        current = child;
        result.terminal = child;
        result.remaining_tokens = result.remaining_tokens.subspan(matched_num_pages * page_size_);
    }

    return result;
}

}  // namespace tokenspeed
