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

// Full attention: a hit is a contiguous run with no holes, so both the device and
// the host lookup walk left-to-right until the first miss.
class FullAttnManager : public KvCacheManager {
public:
    using KvCacheManager::KvCacheManager;

    bool MatchIsPrefixClosed() const override { return true; }

    PrefixProbe Probe(const BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                      std::int32_t max_blocks) const override {
        const std::int32_t end_blocks =
            static_cast<std::int32_t>(std::min(keys.size(), static_cast<std::size_t>(std::max(max_blocks, 0))));
        PrefixProbe probe;
        for (std::int32_t j = begin_blocks; j < end_blocks; ++j) {
            if (!pool.ContainsCachedBlock(keys[static_cast<std::size_t>(j)])) {
                break;
            }
            probe.hits.push_back(1);
        }
        return probe;
    }
};

}  // namespace tokenspeed
