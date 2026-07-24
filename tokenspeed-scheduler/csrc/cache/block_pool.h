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
#include <deque>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache/block_ref.h"
#include "utils.h"

namespace tokenspeed {

class BlockPoolSet;
class KvCacheManager;
class Scheduler;

enum class CachedBlockState : std::uint8_t {
    kMiss,
    kPinnedHit,
    kFreeHit,
};

constexpr bool IsCachedBlockHit(CachedBlockState state) noexcept {
    return state != CachedBlockState::kMiss;
}

// Pool-scoped page allocator and prefix-cache index. BlockRef is the only
// public page-lifetime primitive: every non-empty handle contributes one owner
// and the last owner returns the page to the LRU free list without allocation.
class BlockPool {
public:
    explicit BlockPool(std::int32_t total_num_blocks, bool enable_caching = true)
        : total_num_blocks_{total_num_blocks}, enable_caching_{enable_caching} {
        initialize();
    }

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;
    BlockPool(BlockPool&&) = delete;
    BlockPool& operator=(BlockPool&&) = delete;

    ~BlockPool() noexcept { FatalCheck(in_use_.empty(), "BlockPool destroyed with live block references"); }

    std::int32_t TotalBlocks() const noexcept { return total_num_blocks_; }
    std::int32_t NumFreeBlocks() const noexcept { return static_cast<std::int32_t>(free_list_.size()); }

    // All-or-nothing (empty on shortfall), returning fresh shared owners.
    std::vector<BlockRef> AcquireBlocks(std::int32_t num) {
        std::vector<BlockRef> out;
        if (num <= 0 || static_cast<std::size_t>(num) > free_list_.size()) {
            return out;
        }
        out.reserve(static_cast<std::size_t>(num));
        for (std::int32_t i = 0; i < num; ++i) {
            out.push_back(acquireBlock());
        }
        return out;
    }

    // Allocation-free commit twin for pre-sized coordinator transactions.
    bool AcquireBlocksInto(std::span<BlockRef> out) {
        if (out.size() > free_list_.size()) {
            return false;
        }
        if (std::ranges::any_of(out, [](const BlockRef& block) { return static_cast<bool>(block); })) {
            throw std::invalid_argument("AcquireBlocksInto requires empty output handles");
        }
        for (BlockRef& block : out) {
            block = acquireBlock();
        }
        return true;
    }

    BlockRef AcquireBlock() { return acquireBlock(); }

    // A cache hit returns another shared owner. Active hits stay active; an
    // evictable hit is first removed from the LRU free list.
    BlockRef AcquireCachedBlock(const std::string& block_hash_with_group) {
        internal_block_ref::BlockControl* control = lookupCachedControl(block_hash_with_group);
        if (control == nullptr) {
            return {};
        }
        if (control->UseCount() == 0) {
            removeFree(*control);
            FatalCheck(num_cached_free_blocks_ > 0, "cached-free block counter underflow");
            --num_cached_free_blocks_;
            ++num_pinned_cached_blocks_;
        }
        return BlockRef{*control};
    }

    CachedBlockState ProbeCachedBlock(const std::string& block_hash_with_group) const {
        const internal_block_ref::BlockControl* control = lookupCachedControl(block_hash_with_group);
        if (control == nullptr) {
            return CachedBlockState::kMiss;
        }
        return control->UseCount() == 0 ? CachedBlockState::kFreeHit : CachedBlockState::kPinnedHit;
    }

    bool ContainsCachedBlock(const std::string& block_hash_with_group) const {
        return IsCachedBlockHit(ProbeCachedBlock(block_hash_with_group));
    }

    void CacheFullBlock(const BlockRef& block_ref, const std::string& block_hash_with_group) {
        _assert(block_ref && block_ref.IsOwnedBy(*this), "block reference belongs to another pool");
        if (!enable_caching_) {
            return;
        }
        _assert(!block_hash_with_group.empty(), "block hash must not be empty");

        internal_block_ref::BlockControl& control = *block_ref.control_;
        _assert(!control.Object().IsCached(), "block already has a hash");
        control.Object().SetHash(block_hash_with_group);
        try {
            cache_index_[block_hash_with_group].push_back(&control);
        } catch (...) {
            control.Object().ResetHash();
            auto it = cache_index_.find(block_hash_with_group);
            if (it != cache_index_.end() && it->second.empty()) {
                cache_index_.erase(it);
            }
            throw;
        }
        ++num_cache_bindings_;
        ++num_pinned_cached_blocks_;
    }

    // O(1) observability counters maintained at mutation time.
    std::int32_t NumCachedBlocks() const noexcept { return num_cache_bindings_; }
    std::int32_t NumCacheBindings() const { return NumCachedBlocks(); }
    std::size_t NumCacheHashKeys() const noexcept { return cache_index_.size(); }
    std::int32_t NumCachedFreeBlocks() const noexcept { return num_cached_free_blocks_; }
    std::int32_t NumPinnedCachedBlocks() const noexcept { return num_pinned_cached_blocks_; }

    bool IsQuiescent() const noexcept { return isQuiescent(); }

    void ResetQuiescent() {
        if (!isQuiescent()) {
            throw std::logic_error("cannot reset flat block pool with live refs");
        }
        cache_index_.clear();
        for (internal_block_ref::BlockControl& control : controls_) {
            control.Object().ResetHash();
        }
        num_cache_bindings_ = 0;
        num_cached_free_blocks_ = 0;
        num_pinned_cached_blocks_ = 0;

        // Restore constructor order without allocating or replacing controls.
        for (internal_block_ref::BlockControl& control : controls_) {
            free_list_.splice(free_list_.end(), free_list_, control.Position());
        }
    }

private:
    void initialize() {
        _assert(total_num_blocks_ > 0, "total_num_blocks must be > 0");
        auto return_to_pool = [](BlockPool& pool, internal_block_ref::BlockControl& control) noexcept {
            pool.returnToPool(control);
        };
        for (std::int32_t block_id = 1; block_id < total_num_blocks_; ++block_id) {
            controls_.emplace_back(block_id, *this, return_to_pool);
        }
        for (internal_block_ref::BlockControl& control : controls_) {
            free_list_.push_back(&control);
            control.SetPosition(std::prev(free_list_.end()));
            control.MarkFree();
        }
    }

    BlockRef acquireBlock() {
        if (free_list_.empty()) {
            return {};
        }
        internal_block_ref::BlockControl& control = popFree();
        if (control.Object().IsCached()) {
            evictCached(control);
        }
        return BlockRef{control};
    }

    void returnToPool(internal_block_ref::BlockControl& control) noexcept {
        FatalCheck(control.IsOwnedBy(*this) && !control.InFreeList() && control.UseCount() == 0,
                   "BlockPool can only reclaim its own unowned in-use block");
        if (control.Object().IsCached()) {
            FatalCheck(num_pinned_cached_blocks_ > 0, "pinned-cached block counter underflow");
            --num_pinned_cached_blocks_;
            ++num_cached_free_blocks_;
        }
        free_list_.splice(free_list_.end(), in_use_, control.Position());
        control.MarkFree();
    }

    internal_block_ref::BlockControl& popFree() noexcept {
        FatalCheck(!free_list_.empty(), "BlockPool cannot pop an empty free list");
        internal_block_ref::BlockControl& control = *free_list_.front();
        in_use_.splice(in_use_.end(), free_list_, free_list_.begin());
        control.MarkInUse();
        return control;
    }

    void removeFree(internal_block_ref::BlockControl& control) noexcept {
        FatalCheck(control.InFreeList() && control.UseCount() == 0, "BlockPool can only remove an unowned free block");
        in_use_.splice(in_use_.end(), free_list_, control.Position());
        control.MarkInUse();
    }

    void evictCached(internal_block_ref::BlockControl& control) noexcept {
        CacheBlock& block = control.Object();
        auto it = cache_index_.find(block.BlockHash());
        FatalCheck(it != cache_index_.end(), "cached block is missing from the cache index");
        const std::size_t old_size = it->second.size();
        std::erase(it->second, &control);
        FatalCheck(it->second.size() + 1 == old_size, "cached block has a duplicate or missing index binding");
        if (it->second.empty()) {
            cache_index_.erase(it);
        }
        FatalCheck(num_cache_bindings_ > 0, "cache binding counter underflow");
        --num_cache_bindings_;
        if (control.UseCount() == 0) {
            FatalCheck(num_cached_free_blocks_ > 0, "cached-free block counter underflow");
            --num_cached_free_blocks_;
        } else {
            FatalCheck(num_pinned_cached_blocks_ > 0, "pinned-cached block counter underflow");
            --num_pinned_cached_blocks_;
        }
        block.ResetHash();
    }

    internal_block_ref::BlockControl* lookupCachedControl(const std::string& block_hash_with_group) const {
        if (!enable_caching_) {
            return nullptr;
        }
        auto it = cache_index_.find(block_hash_with_group);
        if (it == cache_index_.end()) {
            return nullptr;
        }
        FatalCheck(!it->second.empty(), "cache index contains an empty binding set");
        return it->second.front();
    }

    bool isQuiescent() const noexcept {
        if (!in_use_.empty() || free_list_.size() != controls_.size()) {
            return false;
        }
        return std::ranges::all_of(controls_, [](const internal_block_ref::BlockControl& control) {
            return control.UseCount() == 0 && control.InFreeList();
        });
    }

    std::int32_t total_num_blocks_{0};
    bool enable_caching_{true};
    std::deque<internal_block_ref::BlockControl> controls_{};
    internal_block_ref::BlockControl::ControlList free_list_{};
    internal_block_ref::BlockControl::ControlList in_use_{};
    std::unordered_map<std::string, std::vector<internal_block_ref::BlockControl*>> cache_index_{};
    std::int32_t num_cache_bindings_{0};
    std::int32_t num_cached_free_blocks_{0};
    std::int32_t num_pinned_cached_blocks_{0};

    friend class BlockPoolSet;
    friend class KvCacheManager;
    friend class Scheduler;
};

}  // namespace tokenspeed
