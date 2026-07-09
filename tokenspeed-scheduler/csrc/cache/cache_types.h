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
#include <ranges>
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
    // Read-only view of the pages as CacheBlock*; refcounts stay sealed inside BlockRef.
    using BlockView = std::ranges::transform_view<std::span<const BlockRef>, decltype(&BlockRef::Get)>;

    BlockView Blocks() const { return BlockView{std::span{blocks_}, &BlockRef::Get}; }
    std::int32_t NumBlocks() const { return static_cast<std::int32_t>(blocks_.size()); }
    std::int32_t TailAvailableTokens() const { return tail_avail_; }

    // Replace slot `index` with a null hole (slot alignment kept) and return the
    // displaced block for the caller to free; nullptr if already a hole.
    CacheBlock* EvictToNull(std::int32_t index, CacheBlock* null_block) {
        _assert(0 <= index && index < static_cast<std::int32_t>(blocks_.size()),
                "EvictToNull index out of range");
        BlockRef& slot = blocks_[static_cast<std::size_t>(index)];
        CacheBlock* old = slot.Get();
        _assert(old != nullptr, "EvictToNull on a moved-out slot");
        if (old == null_block) {
            return nullptr;
        }
        // Order is load-bearing: surrender the displaced ref BEFORE the move-assign,
        // or the assignment would double-decrement it.
        BlockRef hole = BlockRef::Share(*slot.pool_, null_block);
        slot.Release();
        slot = std::move(hole);
        return old;
    }

private:
    friend class KvCacheManager;

    std::vector<BlockRef> blocks_{};
    std::int32_t tail_avail_{0};
};

// The single flattening authority: BlockId() per logical slot, null holes written as 0, no compaction.
inline std::vector<std::int32_t> BlockTablePageIds(const BlockTable& table) {
    std::vector<std::int32_t> ids;
    ids.reserve(static_cast<std::size_t>(table.NumBlocks()));
    for (CacheBlock* b : table.Blocks()) {
        ids.push_back(b->IsNull() ? 0 : b->BlockId());
    }
    return ids;
}

// blocks maps logical page -> physical page, unmatched / out-of-window slots as null_block
// holes; num_hit_blocks counts only the real cached pages (holes excluded).
struct PrefixMatch {
    std::vector<CacheBlock*> blocks{};
    std::int32_t num_hit_blocks{0};
};

}  // namespace tokenspeed
