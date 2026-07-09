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
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "cache/kv_cache_manager.h"

namespace tokenspeed {

// Full-attention KV manager: a hit is a contiguous run with no holes, so both
// the device and the host lookup walk left-to-right until the first miss.
// Shared allocation/claim/cache/free come from KvCacheManager.
class FullAttnManager : public KvCacheManager {
public:
    using KvCacheManager::KvCacheManager;

    bool MatchIsPrefixClosed() const override { return true; }

    // Walk the keys until the first miss: contiguous hits, no holes.
    PrefixMatch MatchPrefix(std::span<const std::string> block_hashes,
                            std::int32_t max_blocks) const override {
        const std::int32_t bound = static_cast<std::int32_t>(
            std::min(block_hashes.size(), static_cast<std::size_t>(std::max(max_blocks, 0))));
        PrefixMatch match;
        match.blocks = contiguousRun(pool_, block_hashes, 0, bound);
        match.num_hit_blocks = static_cast<std::int32_t>(match.blocks.size());
        return match;
    }

    // Host-pool lookup: the same contiguous walk, starting at `begin_blocks`.
    std::vector<CacheBlock*> MatchHostPages(BlockPool& host_pool, std::span<const std::string> keys,
                                            std::int32_t begin_blocks, std::int32_t max_blocks) const override {
        return contiguousRun(host_pool, keys, begin_blocks, max_blocks);
    }

private:
    static std::vector<CacheBlock*> contiguousRun(BlockPool& pool, std::span<const std::string> keys,
                                                  std::int32_t begin_blocks, std::int32_t end_blocks) {
        std::vector<CacheBlock*> blocks;
        for (std::int32_t j = begin_blocks; j < end_blocks; ++j) {
            CacheBlock* block = pool.GetCachedBlock(keys[static_cast<std::size_t>(j)]);
            if (block == nullptr) {
                break;
            }
            blocks.push_back(block);
        }
        return blocks;
    }
};

}  // namespace tokenspeed
