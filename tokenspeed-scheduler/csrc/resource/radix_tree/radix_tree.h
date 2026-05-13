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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "resource/types.h"
#include "resource/radix_tree/tree_node.h"

namespace tokenspeed {

inline std::vector<std::int32_t> SliceInts(const std::vector<std::int32_t>& values, std::size_t start_page,
                                           std::size_t num_pages) {
    return std::vector<std::int32_t>(values.begin() + start_page, values.begin() + start_page + num_pages);
}

inline std::vector<std::string> SliceStrings(const std::vector<std::string>& values, std::size_t start_page,
                                             std::size_t num_pages) {
    return std::vector<std::string>(values.begin() + start_page, values.begin() + start_page + num_pages);
}

token_vec_t FlattenPages(const std::vector<std::span<const std::int32_t>>& pages, std::size_t start_page,
                         std::size_t num_pages);

class RadixTree {
public:
    RadixTree(std::int32_t page_size);

    std::int32_t PageSize() const { return page_size_; }
    TreeNode* Root() const { return root_.get(); }

    WalkResult WalkDownUtilMismatch(token_slice aligned_tokens, TreeNode::timestamp_t access_time,
                                    TreeNode* start_node = nullptr);

    TreeNode* PruneEmptyByNode(TreeNode* node);

    // Find or create the node at depth_in_tokens on descendant's root path.
    // depth_in_tokens must be page-aligned and within descendant's depth.
    // Returns nullptr for root depth or invalid inputs.
    TreeNode* SplitAt(TreeNode* descendant, std::int32_t depth_in_tokens);

private:
    SplitResult splitChild(TreeNode* parent, const token_vec_t& child_key, std::size_t prefix_pages);

private:
    std::int32_t page_size_;
    std::unique_ptr<TreeNode> root_;
};

}  // namespace tokenspeed
