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

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/block_ref.h"
#include "utils.h"

namespace tokenspeed {

enum class AttnKind { kFull, kSlidingWindow, kMambaState };

struct KvCacheSpec {
    AttnKind kind;
    std::int32_t block_size;
    std::int32_t sliding_window;  // 0 for full attention
};

// Per-request logical-page -> physical-page mapping.
class BlockTable {
public:
    std::span<const BlockRef> Blocks() const noexcept { return blocks_; }
    std::int32_t NumBlocks() const { return static_cast<std::int32_t>(blocks_.size()); }
    std::int32_t TailAvailableTokens() const { return tail_avail_; }

    BlockRef EvictToNull(std::int32_t index) {
        _assert(0 <= index && index < static_cast<std::int32_t>(blocks_.size()), "EvictToNull index out of range");
        return std::exchange(blocks_[static_cast<std::size_t>(index)], {});
    }

private:
    friend class KvCacheManager;

    std::vector<BlockRef> blocks_{};
    std::int32_t tail_avail_{0};
};

inline std::vector<std::int32_t> BlockTablePageIds(const BlockTable& table) {
    std::vector<std::int32_t> ids;
    ids.reserve(static_cast<std::size_t>(table.NumBlocks()));
    for (const BlockRef& block : table.Blocks()) {
        ids.push_back(block ? block->BlockId() : 0);
    }
    return ids;
}

struct PrefixMatch {
    std::vector<BlockRef> blocks{};
    std::int32_t num_hit_blocks{0};
};

// Non-owning match shape. A nonzero slot is acquired only after the coordinator
// converges every group to the final common boundary.
struct PrefixProbe {
    std::vector<std::uint8_t> hits{};
};

// Pinned source/destination pages for one asynchronous cache transfer.
struct BlockTransfer {
    BlockRef source;
    BlockRef destination;
};

}  // namespace tokenspeed
