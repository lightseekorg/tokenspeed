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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

#pragma once

// Test-only friend of HybridPrefixCache; exposes narrow hooks needed to seed or
// inspect internals while production callers use the scheduler-facing facades.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/radix_tree/mamba_slot.h"
#include "resource/radix_tree/tree_node.h"

namespace tokenspeed {

class HybridPrefixCacheTestPeer {
public:
    static void InsertMamba(HybridPrefixCache& cache, TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot) {
        cache.InsertMamba(terminal_node, std::move(slot));
    }

    static std::vector<TransferPair> PrepareMambaHostWriteBack(HybridPrefixCache& cache,
                                                               const std::vector<TreeNode*>& nodes) {
        return cache.PrepareMambaHostWriteBack(nodes);
    }

    static std::vector<TransferPair> PrepareMambaDeviceLoadBack(HybridPrefixCache& cache,
                                                                const std::vector<TreeNode*>& nodes) {
        return cache.PrepareMambaDeviceLoadBack(nodes);
    }

    static void RegisterPagedCacheGroup(HybridPrefixCache& cache, std::unique_ptr<PagedCacheGroupAllocator> allocator) {
        cache.RegisterPagedCacheGroup(std::move(allocator));
    }

    static void EnablePagedCacheAdjunct(HybridPrefixCache& cache, std::vector<std::string> required_groups,
                                        std::unordered_map<std::string, std::int32_t> sliding_window_per_group,
                                        StateRestorePolicy policy = StateRestorePolicy::kSnapshotRequired) {
        cache.EnablePagedCacheAdjunct(std::move(required_groups), std::move(sliding_window_per_group), policy);
    }

    static void AcquireForRequest(HybridPrefixCache& cache, const std::string& request_id,
                                  std::int32_t first_raw_position_of_op, std::int32_t target_raw_tokens_exclusive,
                                  const MatchResult::PagedCache& paged_cache_hit = {}) {
        cache.AcquireForRequest(request_id, first_raw_position_of_op, target_raw_tokens_exclusive, paged_cache_hit);
    }

    static void ReleaseRequest(HybridPrefixCache& cache, const std::string& request_id) {
        cache.ReleaseRequest(request_id);
    }

    static void CommitChunk(HybridPrefixCache& cache, const std::string& request_id, TreeNode* terminal) {
        cache.CommitChunk(request_id, terminal);
    }

    static bool AttachPagedCacheSnapshotToNode(HybridPrefixCache& cache, TreeNode* node,
                                               std::unique_ptr<PagedCacheSnapshot> snapshot) {
        return cache.AttachPagedCacheSnapshotToNode(node, std::move(snapshot));
    }

    static std::unique_ptr<PagedCacheSnapshot> DetachPagedCacheSnapshotFromNode(HybridPrefixCache& cache,
                                                                                TreeNode* node) {
        return cache.DetachPagedCacheSnapshotFromNode(node);
    }
};

}  // namespace tokenspeed
