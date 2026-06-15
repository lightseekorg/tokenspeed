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
#include <span>
#include <vector>

#include "cache/block_pool.h"

namespace tokenspeed {

// Per-group KV cache configuration (immutable). One spec per attention group;
// GPT-OSS uses two: a full-attention group and a sliding-window group.
enum class AttnKind { kFull, kSlidingWindow };

struct KvCacheSpec {
    AttnKind kind;
    std::int32_t page_size;
    std::int32_t sliding_window;  // 0 for full attention
};

// Shared per-request / match data types for the flat KV-cache layer. Consumed
// by the per-type managers (FullAttnManager, and later SwaManager /
// MambaManager), the KVCacheCoordinator, and the FSM ForwardState. Kept here --
// not inside any one manager header -- so consumers depend on the data contract
// rather than on a concrete manager.

// Per-request logical-page -> physical-page mapping plus the incremental
// allocation cursor. Pure value type, movable. Takes the role LocalKVAllocator
// holds today; its final home is the FSM ForwardState (a later increment).
class BlockTable {
public:
    std::span<CacheBlock* const> Blocks() const { return blocks_; }
    std::int32_t NumBlocks() const { return static_cast<std::int32_t>(blocks_.size()); }
    // Tokens still fillable in the last (tail) page before a new page is needed.
    std::int32_t TailAvailableTokens() const { return tail_avail_; }

private:
    friend class KvCacheManager;

    std::vector<CacheBlock*> blocks_{};
    std::int32_t tail_avail_{0};
};

// Unified prefix-match result across all managers (mirrors vLLM computed_blocks).
// blocks maps logical page -> physical page; an unmatched / out-of-window slot is
// a null_block hole. blocks.size() is the compute coverage (every slot, real or
// hole, is a computed page); num_hit_blocks is the count of real cached pages
// that need claiming (excludes holes). For full attention there are no holes, so
// blocks.size() == num_hit_blocks.
struct PrefixMatch {
    std::vector<CacheBlock*> blocks{};
    std::int32_t num_hit_blocks{0};
};

}  // namespace tokenspeed
