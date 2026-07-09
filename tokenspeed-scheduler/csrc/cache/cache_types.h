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
#include <iterator>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/block_ref.h"
#include "utils.h"

namespace tokenspeed {

enum class AttnKind { kFull, kSlidingWindow };

// Per-group KV cache configuration (immutable). One spec per attention group;
// GPT-OSS uses two: a full-attention group and a sliding-window group.
struct KvCacheSpec {
    AttnKind kind;
    std::int32_t page_size;
    std::int32_t sliding_window;  // 0 for full attention
};

// Shared per-request / match data types for the flat KV-cache layer. Consumed
// by the per-type managers (FullAttnManager, SwaManager), the
// KVCacheCoordinator, and the FSM ForwardState. Kept here --
// not inside any one manager header -- so consumers depend on the data contract
// rather than on a concrete manager.

// Per-request logical-page -> physical-page mapping plus the incremental
// allocation cursor. Pure value type, movable; owned by the FSM ForwardState
// (block_tables_) for the request's lifetime.
class BlockTable {
public:
    class BlockView {
    public:
        class Iterator {
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type = CacheBlock*;
            using difference_type = std::ptrdiff_t;
            using pointer = CacheBlock* const*;
            using reference = CacheBlock*;

            explicit Iterator(const BlockRef* ref) : ref_{ref} {}
            CacheBlock* operator*() const { return ref_->Get(); }
            Iterator& operator++() {
                ++ref_;
                return *this;
            }
            Iterator operator++(int) {
                Iterator tmp = *this;
                ++ref_;
                return tmp;
            }
            bool operator==(const Iterator&) const = default;

        private:
            const BlockRef* ref_;
        };

        BlockView(const BlockRef* data, std::size_t size) : data_{data}, size_{size} {}
        CacheBlock* operator[](std::size_t i) const { return data_[i].Get(); }
        std::size_t size() const { return size_; }
        bool empty() const { return size_ == 0; }
        Iterator begin() const { return Iterator{data_}; }
        Iterator end() const { return Iterator{data_ + size_}; }

    private:
        const BlockRef* data_;
        std::size_t size_;
    };

    BlockView Blocks() const { return BlockView{blocks_.data(), blocks_.size()}; }
    std::int32_t NumBlocks() const { return static_cast<std::int32_t>(blocks_.size()); }
    // Tokens still fillable in the last (tail) page before a new page is needed.
    std::int32_t TailAvailableTokens() const { return tail_avail_; }

    // Evict the physical page at logical slot `index`, replacing it with a null
    // hole, and return the displaced block so the caller can free it. Returns
    // nullptr if the slot is already a null hole (idempotent). The table's length
    // is unchanged -- the hole preserves logical-page -> slot alignment.
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

// Physical page ids of one BlockTable: BlockId() per logical slot, with
// null-block holes written as 0, in absolute logical-page order (no
// compaction). The single source of truth for flattening a BlockTable's pages.
inline std::vector<std::int32_t> BlockTablePageIds(const BlockTable& table) {
    std::vector<std::int32_t> ids;
    ids.reserve(static_cast<std::size_t>(table.NumBlocks()));
    for (CacheBlock* b : table.Blocks()) {
        ids.push_back(b->IsNull() ? 0 : b->BlockId());
    }
    return ids;
}

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
