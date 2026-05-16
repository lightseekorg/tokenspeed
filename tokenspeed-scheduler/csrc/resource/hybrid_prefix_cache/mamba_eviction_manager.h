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
#include <queue>
#include <unordered_set>
#include <vector>

#include "resource/radix_tree/tree_node.h"

namespace tokenspeed {

class MambaChunkAllocator;

class MambaEvictionManager {
public:
    using DetachFn = std::function<void(TreeNode*)>;

    MambaEvictionManager(MambaChunkAllocator* allocator, ResourceType residency, DetachFn detach_fn = {});

    void TrackNode(TreeNode* node);
    void UntrackNode(TreeNode* node);
    void UpdateLeaf(TreeNode* node);

    std::int32_t Evict(std::int32_t num_slots, TreeNode* protected_node = nullptr);
    bool EnsureCapacity(std::int32_t required_slots, TreeNode* protected_node = nullptr);

    std::int32_t EvictableSlots() const;

private:
    bool isMambaLeaf(const TreeNode* node) const;
    bool hasChildWithMamba(const TreeNode* node) const;

    bool hasMamba(const TreeNode* node) const;
    bool isLocked(const TreeNode* node) const;
    void detach(TreeNode* node) const;

    MambaChunkAllocator* allocator_;
    ResourceType residency_;
    DetachFn detach_fn_;
    std::unordered_set<TreeNode*> mamba_leaves_;
};

}  // namespace tokenspeed
