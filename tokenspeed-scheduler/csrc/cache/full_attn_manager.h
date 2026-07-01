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

#include <span>
#include <string>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "cache/kv_cache_manager.h"

namespace tokenspeed {

// Full-attention KV manager: prefix match walks the page-hash keys left-to-right
// until the first miss (a full-attention hit is a contiguous prefix from token 0,
// with no holes). Shared allocation/claim/cache/free come from KvCacheManager.
class FullAttnManager : public KvCacheManager {
public:
    using KvCacheManager::KvCacheManager;

    // Read-only prefix match: walk the pre-computed page-hash keys until the
    // first cache miss. Returns the hit blocks in order (no holes). Does NOT
    // change ref counts -- callers claim hits via ClaimHitBlocks.
    PrefixMatch MatchPrefix(std::span<const std::string> block_hashes) const override {
        PrefixMatch match;
        for (const std::string& hash : block_hashes) {
            CacheBlock* block = pool_.GetCachedBlock(hash);
            if (block == nullptr) {
                break;
            }
            match.blocks.push_back(block);
        }
        match.num_hit_blocks = static_cast<std::int32_t>(match.blocks.size());
        return match;
    }
};

}  // namespace tokenspeed
