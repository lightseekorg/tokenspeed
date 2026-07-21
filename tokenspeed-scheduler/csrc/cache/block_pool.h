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
#include <list>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache/block_ref.h"
#include "utils.h"

namespace tokenspeed {

// Per-block metadata. A zero BlockControl strong count does NOT destroy the
// block: it re-enters the free list hash-intact, so it is both prefix-reusable
// and an eviction candidate.
class CacheBlock {
public:
    explicit CacheBlock(std::int32_t block_id) : block_id_{block_id} {}

    // Copy deleted (free_pos_ points into BlockPool::free_); move defaulted for
    // vector::emplace_back but never runs (blocks_ is reserve()'d).
    CacheBlock(const CacheBlock&) = delete;
    CacheBlock& operator=(const CacheBlock&) = delete;
    CacheBlock(CacheBlock&&) = default;
    CacheBlock& operator=(CacheBlock&&) = default;

    std::int32_t BlockId() const { return block_id_; }
    bool IsNull() const { return is_null_; }
    bool IsCached() const { return !block_hash_.empty(); }

    // BlockHashWithGroupId key (page_hasher.h); empty when uncached.
    const std::string& BlockHash() const { return block_hash_; }

private:
    friend class BlockPool;

    void SetHash(std::string hash) {
        _assert(block_hash_.empty(), "block already has a hash");
        block_hash_ = std::move(hash);
    }
    void ResetHash() { block_hash_.clear(); }

    std::int32_t block_id_{0};
    std::string block_hash_{};
    bool is_null_{false};
    // Valid only while in_free_: this block's node in BlockPool::free_, for O(1) removal on a prefix hit.
    bool in_free_{false};
    std::list<CacheBlock*>::iterator free_pos_{};
};

namespace detail {

inline auto AllCachedControls(const std::unordered_map<std::string, std::vector<BlockControl*>>& cached) {
    return cached | std::views::values | std::views::join;
}

}  // namespace detail

// Flat prefix-cache block pool; owns all blocks for their whole lifetime; the free list and hash map only track state.
class BlockPool {
public:
    explicit BlockPool(std::int32_t total_num_blocks, bool enable_caching = true)
        : total_num_blocks_{total_num_blocks}, enable_caching_{enable_caching} {
        _assert(total_num_blocks > 0, "total_num_blocks must be > 0");
        blocks_.reserve(total_num_blocks);
        controls_.reserve(total_num_blocks);
        for (std::int32_t i = 0; i < total_num_blocks; ++i) {
            blocks_.emplace_back(i);
            controls_.emplace_back();
            controls_.back().owner_ = this;
            controls_.back().object_ = &blocks_.back();
        }
        // Block 0 is the null placeholder: never cached, ref not tracked; all others start free.
        null_block_ = &blocks_[0];
        null_block_->is_null_ = true;
        for (std::int32_t i = 1; i < total_num_blocks; ++i) {
            pushToFree(&blocks_[i]);
        }
    }

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    std::int32_t TotalBlocks() const { return total_num_blocks_; }
    std::int32_t NumFreeBlocks() const { return static_cast<std::int32_t>(free_.size()); }
    CacheBlock* NullBlock() const { return null_block_; }
    BlockRef NullBlockRef() { return BlockRef{ControlFor(null_block_)}; }

    // All-or-nothing (empty on shortfall). Returned handles own the blocks.
    std::vector<BlockRef> AcquireBlocks(std::int32_t num) {
        std::vector<BlockRef> out;
        if (num <= 0 || static_cast<std::int32_t>(free_.size()) < num) {
            return out;
        }
        out.reserve(num);
        for (std::int32_t i = 0; i < num; ++i) {
            out.push_back(AcquireBlock());
        }
        return out;
    }

    // Empty handle on shortfall.
    BlockRef AcquireBlock() {
        if (free_.empty()) {
            return {};
        }
        CacheBlock* block = popFromFree();
        if (block->IsCached()) {
            evictCachedBlock(block);
        }
        return BlockRef{ControlFor(block)};
    }

    // Owning prefix lookup: a hit is pinned before it leaves the pool.
    BlockRef FindCachedBlock(const std::string& block_hash_with_group) {
        BlockControl* control = lookupCachedControl(block_hash_with_group);
        if (control == nullptr) {
            return {};
        }
        if (control->strong_count_ == 0) {
            removeFromFree(control->object_);
        }
        return BlockRef{control};
    }

    bool ContainsCachedBlock(const std::string& block_hash_with_group) const {
        return lookupCachedControl(block_hash_with_group) != nullptr;
    }

    void CacheFullBlock(const BlockRef& block_ref, const std::string& block_hash_with_group) {
        BlockControl* control = ControlFor(block_ref);
        CacheBlock* block = control->object_;
        if (!enable_caching_ || block->is_null_) {
            return;
        }
        _assert(control->strong_count_ > 0, "CacheFullBlock requires an owning reference");
        block->SetHash(block_hash_with_group);
        cached_hash_to_controls_[block_hash_with_group].push_back(control);
    }

    // Test probes (off the hot path).
    std::int32_t NumCachedBlocks() const {
        return static_cast<std::int32_t>(std::ranges::distance(detail::AllCachedControls(cached_hash_to_controls_)));
    }
    std::int32_t NumCachedFreeBlocks() const {
        return static_cast<std::int32_t>(
            std::ranges::count_if(detail::AllCachedControls(cached_hash_to_controls_),
                                  [](const BlockControl* control) { return control->strong_count_ == 0; }));
    }
    std::int32_t NumPinnedCachedBlocks() const {
        return static_cast<std::int32_t>(
            std::ranges::count_if(detail::AllCachedControls(cached_hash_to_controls_),
                                  [](const BlockControl* control) { return control->strong_count_ > 0; }));
    }

private:
    friend class BlockRef;

    BlockControl* ControlFor(CacheBlock* block) {
        _assert(block != nullptr, "block must not be null");
        const std::int32_t block_id = block->BlockId();
        _assert(0 <= block_id && block_id < static_cast<std::int32_t>(controls_.size()),
                "block belongs to another pool");
        BlockControl* control = &controls_[static_cast<std::size_t>(block_id)];
        _assert(control->owner_ == this && control->object_ == block, "block belongs to another pool");
        return control;
    }

    BlockControl* ControlFor(const BlockRef& block_ref) {
        _assert(block_ref.control_ != nullptr && block_ref.control_->owner_ == this,
                "block reference belongs to another pool");
        return block_ref.control_;
    }

    void OnLastRef(BlockControl* control) {
        _assert(control != nullptr && control->owner_ == this, "control belongs to another pool");
        _assert(control->strong_count_ == 0, "OnLastRef requires zero strong_count");
        pushToFree(control->object_);
    }

    BlockControl* lookupCachedControl(const std::string& block_hash_with_group) const {
        if (!enable_caching_) {
            return nullptr;
        }
        auto it = cached_hash_to_controls_.find(block_hash_with_group);
        if (it == cached_hash_to_controls_.end() || it->second.empty()) {
            return nullptr;
        }
        return it->second.front();
    }

    // std::list gives the O(1) stored-iterator erase a prefix cache needs; a vector/deque stack cannot.
    void pushToFree(CacheBlock* block) {
        _assert(block != nullptr && !block->is_null_, "only real blocks enter the free list");
        _assert(ControlFor(block)->strong_count_ == 0, "only unowned blocks enter the free list");
        _assert(!block->in_free_, "block is already in the free list");
        block->free_pos_ = free_.insert(free_.end(), block);
        block->in_free_ = true;
    }

    CacheBlock* popFromFree() {
        CacheBlock* block = free_.front();
        _assert(block->in_free_ && ControlFor(block)->strong_count_ == 0, "free-list block invariant violated");
        free_.pop_front();
        block->in_free_ = false;
        return block;
    }

    void removeFromFree(CacheBlock* block) {
        _assert(block->in_free_, "block is not in the free list");
        free_.erase(block->free_pos_);
        block->in_free_ = false;
    }

    void evictCachedBlock(CacheBlock* block) {
        auto it = cached_hash_to_controls_.find(block->block_hash_);
        if (it != cached_hash_to_controls_.end()) {
            std::erase(it->second, ControlFor(block));
            if (it->second.empty()) {
                cached_hash_to_controls_.erase(it);
            }
        }
        block->ResetHash();
    }

    std::int32_t total_num_blocks_{0};
    bool enable_caching_{true};
    std::vector<CacheBlock> blocks_{};
    std::vector<BlockControl> controls_{};
    // LRU-ordered: front = next to evict.
    std::list<CacheBlock*> free_{};

    // One key may map to several physical duplicates: never de-duplicated so handed-out block ids stay stable.
    std::unordered_map<std::string, std::vector<BlockControl*>> cached_hash_to_controls_{};
    CacheBlock* null_block_{nullptr};
};

}  // namespace tokenspeed
