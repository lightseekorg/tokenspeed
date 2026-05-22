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
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace tokenspeed {

class TreeNode;

using token_t = std::int32_t;
using token_vec_t = std::vector<token_t>;
using token_slice = std::span<const token_t>;
using cache_op_id = std::uint32_t;

enum class ResourceType {
    Device,
    Host,
};

enum class MatchIntent {
    PrefixReuse,
    StateRecovery,
};

template <ResourceType RType>
class NodeRef;
using DeviceNodeRef = NodeRef<ResourceType::Device>;
using HostNodeRef = NodeRef<ResourceType::Host>;

struct MatchResult {
    struct Device {
        TreeNode* last_node;
        std::int32_t page_size{0};
        std::int32_t DepthInPage() const;
    } device;

    struct Host {
        TreeNode* last_node;
        std::int32_t page_size{0};
        std::int32_t DepthInPage() const;
    } host;

    template <ResourceType RType>
    std::vector<TreeNode*> NodesWithout() const;

    // Mamba extension (default: no mamba cache, -1 = inactive)
    std::int32_t mamba_branching_seqlen{-1};
    std::int32_t mamba_cow_src_index{-1};

    // Paged-cache adjunct hit. Null last_node or zero prefix means no hit.
    // When hit, device/host last_node also sit at or before prefix_len_tokens.
    // base_logical_page is 0 for full-history groups; > 0 for sliding windows.
    // TODO(match-result-pagedcache-zero-copy): return snapshot+depth and walk on demand.
    struct PagedCache {
        // Phase 1 hit kinds; Phase 2 will add kReplay variants.
        enum class RestoreKind { kSnapshotComplete };
        TreeNode* last_node{nullptr};
        std::int32_t prefix_len_tokens{0};
        std::map<std::string, std::vector<std::int32_t>> per_group_page_ids;
        std::map<std::string, std::int32_t> per_group_base_logical_page;
        RestoreKind restore_kind{RestoreKind::kSnapshotComplete};
        // Phase 2 placeholder; Phase 1 always 0.
        std::int32_t replay_start_tokens{0};
    } paged_cache;
};

struct InsertResult {
    TreeNode* last_node;
    std::int32_t inserted_num_pages;
};

struct SplitResult {
    TreeNode* parent;
    TreeNode* prefix;
    TreeNode* suffix;
};

struct WalkResult {
    TreeNode* terminal;
    token_slice remaining_tokens;
    MatchResult match;
};

struct CacheOpSpec {
    std::string request_id;
    TreeNode* last_node{nullptr};
    std::vector<TreeNode*> nodes;

    CacheOpSpec();
    ~CacheOpSpec();
    CacheOpSpec(CacheOpSpec&&) noexcept;
    CacheOpSpec& operator=(CacheOpSpec&&) noexcept;
    CacheOpSpec(const CacheOpSpec&) = delete;
    CacheOpSpec& operator=(const CacheOpSpec&) = delete;
};

}  // namespace tokenspeed
