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
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/block_pool_set.h"
#include "cache/block_ref.h"
#include "utils.h"

namespace tokenspeed {

enum class AttnKind { kFull, kSlidingWindow, kMambaState };

// Scheduler-local copies of model-facing prefix and table policies. Keeping
// them here avoids a resource/allocator dependency in the cache substrate.
enum class KvPrefixRole { kHistoryAnchor, kContinuationState, kNone };
enum class KvTableLayout { kAbsolute, kBoundedWindow };

struct KvCacheSpec {
    AttnKind kind;
    std::int32_t block_size;
    std::int32_t sliding_window;  // 0 for full attention
    PoolIndex pool_index{0};
    KvPrefixRole prefix_role{KvPrefixRole::kHistoryAnchor};
    KvTableLayout table_layout{KvTableLayout::kAbsolute};
    std::uint32_t owner_mask{0};
    // Model/config-facing identifiers and physical row geometry. Defaults
    // preserve the legacy three-field aggregate used by MakeCoordinator tests
    // and callers; normalization resolves them into a complete runtime schema.
    std::string group_id{};
    std::int32_t rows_per_page{0};
    std::int32_t entry_stride_tokens{1};
};

// Coordinator-owned, immutable cache-local description of one group. Every
// scheduler/cache hot path indexes this vector by schema_index; external
// group_id is used only at the public keyed-table/query boundary.
struct KvCacheGroupSchema {
    std::string group_id;
    AttnKind kind;
    std::int32_t block_size;
    std::int32_t rows_per_page;
    std::int32_t entry_stride_tokens;
    std::int32_t sliding_window;
    PoolIndex pool_index;
    KvPrefixRole prefix_role;
    KvTableLayout table_layout;
    std::uint32_t owner_mask;
};

// Per-request logical-page -> physical-page mapping. Empty BlockRef entries
// are absolute-layout holes and flatten to page id 0; bounded layouts discard
// expired prefixes and advance base_logical_page_ instead.
class BlockTable {
public:
    std::span<const BlockRef> Blocks() const noexcept { return blocks_; }
    std::int32_t NumBlocks() const noexcept { return static_cast<std::int32_t>(blocks_.size()); }
    std::int32_t TailAvailableTokens() const noexcept { return tail_avail_; }

    std::int32_t BaseLogicalPage() const noexcept { return base_logical_page_; }
    std::int32_t LiveSize() const noexcept { return NumBlocks(); }
    std::int32_t LogicalEnd() const {
        const std::int64_t end = static_cast<std::int64_t>(base_logical_page_) + LiveSize();
        _assert(end <= std::numeric_limits<std::int32_t>::max(), "BlockTable logical end overflows int32");
        return static_cast<std::int32_t>(end);
    }
    bool ContainsLogical(std::int32_t abs_page) const {
        return abs_page >= base_logical_page_ && abs_page < LogicalEnd();
    }
    std::int32_t ToLocal(std::int32_t abs_page) const {
        _assert(ContainsLogical(abs_page), "BlockTable logical page is outside the live range");
        return abs_page - base_logical_page_;
    }
    BlockRef& AtLogical(std::int32_t abs_page) { return blocks_[static_cast<std::size_t>(ToLocal(abs_page))]; }
    const BlockRef& AtLogical(std::int32_t abs_page) const {
        return blocks_[static_cast<std::size_t>(ToLocal(abs_page))];
    }

    // Staging seam for coordinator transactions. Reserve before any pool or
    // table mutation so commit cannot fail due to vector growth.
    void ReserveLiveSize(std::int32_t live_capacity) {
        _assert(live_capacity >= LiveSize(), "BlockTable reserve cannot shrink below live size");
        blocks_.reserve(static_cast<std::size_t>(live_capacity));
    }

    void InitRange(std::int32_t base_logical_page, std::vector<BlockRef>&& ordered_refs) {
        _assert(blocks_.empty(), "BlockTable::InitRange requires an empty table");
        _assert(tail_avail_ == 0, "BlockTable::InitRange requires empty tail state");
        _assert(base_logical_page >= 0, "BlockTable::InitRange base must be >= 0");
        _assert(ordered_refs.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
                "BlockTable::InitRange live size exceeds int32");
        const auto live_size = static_cast<std::int32_t>(ordered_refs.size());
        _assert(base_logical_page <= std::numeric_limits<std::int32_t>::max() - live_size,
                "BlockTable::InitRange logical end overflows int32");
        base_logical_page_ = base_logical_page;
        blocks_ = std::move(ordered_refs);
    }

    void DropBefore(std::int32_t abs_page) {
        _assert(abs_page >= base_logical_page_, "BlockTable::DropBefore cannot move the base backward");
        _assert(abs_page <= LogicalEnd(), "BlockTable::DropBefore exceeds the live logical range");
        dropBeforeValid(abs_page);
    }

    void DropBeforeNoexcept(std::int32_t abs_page) noexcept {
        _assert(abs_page >= base_logical_page_, "BlockTable::DropBefore cannot move the base backward");
        _assert(abs_page <= LogicalEnd(), "BlockTable::DropBefore exceeds the live logical range");
        dropBeforeValid(abs_page);
    }

    // End the range, preserving the pool's tail-first recycling policy.
    void Reset() noexcept {
        for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
            it->reset();
        }
        blocks_.clear();
        base_logical_page_ = 0;
        tail_avail_ = 0;
    }

    // Replace one absolute-layout slot with an empty-handle hole and return
    // the displaced owner. The caller controls its release order.
    BlockRef EvictToNull(std::int32_t index) {
        _assert(0 <= index && index < LiveSize(), "EvictToNull index out of range");
        return std::exchange(blocks_[static_cast<std::size_t>(index)], {});
    }

    BlockRef EvictToNullNoexcept(std::int32_t index) noexcept {
        _assert(0 <= index && index < LiveSize(), "EvictToNull index out of range");
        return std::exchange(blocks_[static_cast<std::size_t>(index)], {});
    }

private:
    void dropBeforeValid(std::int32_t abs_page) noexcept {
        const std::int32_t drop_count = abs_page - base_logical_page_;
        blocks_.erase(blocks_.begin(), blocks_.begin() + drop_count);
        base_logical_page_ = abs_page;
        if (blocks_.empty()) {
            tail_avail_ = 0;
        }
    }

    friend class KvCacheManager;

    std::vector<BlockRef> blocks_{};
    std::int32_t base_logical_page_{0};
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
    std::int32_t base_logical_page{0};

    std::int32_t LogicalEnd() const {
        const std::int64_t end =
            static_cast<std::int64_t>(base_logical_page) + static_cast<std::int64_t>(blocks.size());
        _assert(end <= std::numeric_limits<std::int32_t>::max(), "PrefixMatch logical end overflows int32");
        return static_cast<std::int32_t>(end);
    }
};

// Non-owning match shape. Handles are acquired only after all groups converge
// on a common endpoint. Each state also records whether acquiring the hit will
// consume an evictable page, without a second cache-index lookup.
struct PrefixProbe {
    std::vector<CachedBlockState> hits{};
    std::int32_t base_logical_page{0};

    std::int32_t LogicalEnd() const {
        const std::int64_t end = static_cast<std::int64_t>(base_logical_page) + static_cast<std::int64_t>(hits.size());
        _assert(end <= std::numeric_limits<std::int32_t>::max(), "PrefixProbe logical end overflows int32");
        return static_cast<std::int32_t>(end);
    }
};

// Pinned source/destination pages for one asynchronous cache transfer.
struct BlockTransfer {
    BlockRef source;
    BlockRef destination;
};

}  // namespace tokenspeed
