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
#include <memory>
#include <span>
#include <vector>

#include "resource/hybrid_prefix_cache/mamba_eviction_manager.h"
#include "resource/radix_tree/mamba_slot.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/types.h"

namespace tokenspeed {

class MambaChunkAllocator;

class HybridPrefixCache {
public:
    HybridPrefixCache(KVPrefixCache& prefix_cache, MambaChunkAllocator* allocator, std::int32_t mamba_cache_chunk_size);

    MatchResult Match(const token_vec_t& token_ids, std::int32_t lora_id = kLoraNone);
    MatchResult Match(const std::vector<std::span<const std::int32_t>>& token_pages, std::int32_t lora_id = kLoraNone);

    bool EnsureMambaCapacityByEvict(std::int32_t num_slots);
    void InsertMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot);
    std::int32_t AlignMambaCacheSeqlen(std::int32_t seqlen) const;
    TreeNode* FindLastMambaNode(TreeNode* from) const;

    // CallBack on KV Prefix Cache Eviction
    void OnKVEvict(TreeNode* node);

    std::int32_t AvailableSlots() const;
    KVPrefixCache& GetKVPrefixCache() { return kv_prefix_cache_; }

private:
    void augmentMatch(MatchResult& match) const;

    KVPrefixCache& kv_prefix_cache_;
    MambaChunkAllocator* mamba_allocator_;
    MambaEvictionManager mamba_eviction_manager_;
    std::int32_t mamba_cache_chunk_size_;
};

}  // namespace tokenspeed
