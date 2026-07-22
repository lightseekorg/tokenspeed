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
#include <string>
#include <unordered_map>
#include <vector>

#include "cache/block_ref.h"
#include "utils.h"

namespace tokenspeed {

class BlockPool {
public:
    explicit BlockPool(std::int32_t total_num_blocks, bool enable_caching = true)
        : total_num_blocks_{total_num_blocks}, enable_caching_{enable_caching} {
        _assert(total_num_blocks > 0, "total_num_blocks must be > 0");
        auto return_to_pool = [](BlockPool& pool, internal_block_ref::BlockControl& control) noexcept {
            pool.returnToPool(control);
        };
        for (std::int32_t block_id = 1; block_id < total_num_blocks; ++block_id) {
            controls_.emplace_back(block_id, *this, return_to_pool);
        }
        for (internal_block_ref::BlockControl& control : controls_) {
            free_list_.push_back(&control);
            control.SetPosition(std::prev(free_list_.end()));
            control.MarkFree();
        }
    }

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;
    ~BlockPool() noexcept { FatalCheck(in_use_.empty(), "BlockPool destroyed with live block references"); }

    std::int32_t TotalBlocks() const noexcept { return total_num_blocks_; }
    std::int32_t NumFreeBlocks() const noexcept { return static_cast<std::int32_t>(free_list_.size()); }

    std::vector<BlockRef> AcquireBlocks(std::int32_t num) {
        std::vector<BlockRef> out;
        if (num <= 0 || static_cast<std::int32_t>(free_list_.size()) < num) {
            return out;
        }
        out.reserve(static_cast<std::size_t>(num));
        for (std::int32_t i = 0; i < num; ++i) {
            out.push_back(AcquireBlock());
        }
        return out;
    }

    BlockRef AcquireBlock() {
        if (free_list_.empty()) {
            return {};
        }
        internal_block_ref::BlockControl& control = popFree();
        if (control.Object().IsCached()) {
            evictCached(control);
        }
        return BlockRef{control};
    }

    BlockRef AcquireCachedBlock(const std::string& block_hash_with_group) {
        internal_block_ref::BlockControl* control = lookupCachedControl(block_hash_with_group);
        if (control == nullptr) {
            return {};
        }
        if (control->UseCount() == 0) {
            removeFree(*control);
        }
        return BlockRef{*control};
    }

    bool ContainsCachedBlock(const std::string& block_hash_with_group) const {
        return lookupCachedControl(block_hash_with_group) != nullptr;
    }

    bool IsCachedBlockFree(const std::string& block_hash_with_group) const {
        const internal_block_ref::BlockControl* control = lookupCachedControl(block_hash_with_group);
        return control != nullptr && control->UseCount() == 0;
    }

    void CacheFullBlock(const BlockRef& block_ref, const std::string& block_hash_with_group) {
        _assert(block_ref && block_ref.IsOwnedBy(*this), "block reference belongs to another pool");
        internal_block_ref::BlockControl& control = controlAt(block_ref->BlockId());
        CacheBlock& block = control.Object();
        if (!enable_caching_) {
            return;
        }
        _assert(!block_hash_with_group.empty(), "block hash must not be empty");
        block.SetHash(block_hash_with_group);
        try {
            cache_index_[block_hash_with_group].push_back(&control);
        } catch (...) {
            block.ResetHash();
            throw;
        }
    }

    std::int32_t NumCachedBlocks() const {
        std::int32_t count = 0;
        for (const auto& [_, controls] : cache_index_) {
            count += static_cast<std::int32_t>(controls.size());
        }
        return count;
    }

    std::int32_t NumCachedFreeBlocks() const {
        return countCached([](const internal_block_ref::BlockControl& control) { return control.UseCount() == 0; });
    }

    std::int32_t NumPinnedCachedBlocks() const {
        return countCached([](const internal_block_ref::BlockControl& control) { return control.UseCount() > 0; });
    }

private:
    internal_block_ref::BlockControl& controlAt(std::int32_t block_id) {
        _assert(0 < block_id && block_id < total_num_blocks_, "block id out of range");
        return controls_[static_cast<std::size_t>(block_id - 1)];
    }

    void returnToPool(internal_block_ref::BlockControl& control) noexcept {
        FatalCheck(control.IsOwnedBy(*this) && !control.InFreeList() && control.UseCount() == 0,
                   "BlockPool can only reclaim its own unowned in-use block");
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

    void evictCached(internal_block_ref::BlockControl& control) {
        CacheBlock& block = control.Object();
        auto it = cache_index_.find(block.BlockHash());
        if (it != cache_index_.end()) {
            std::erase(it->second, &control);
            if (it->second.empty()) {
                cache_index_.erase(it);
            }
        }
        block.ResetHash();
    }

    internal_block_ref::BlockControl* lookupCachedControl(const std::string& block_hash_with_group) const {
        if (!enable_caching_) {
            return nullptr;
        }
        auto it = cache_index_.find(block_hash_with_group);
        if (it == cache_index_.end() || it->second.empty()) {
            return nullptr;
        }
        return it->second.front();
    }

    template <class Predicate>
    std::int32_t countCached(Predicate predicate) const {
        std::int32_t count = 0;
        for (const auto& [_, controls] : cache_index_) {
            count += static_cast<std::int32_t>(std::ranges::count_if(
                controls, [&](const internal_block_ref::BlockControl* control) { return predicate(*control); }));
        }
        return count;
    }

    std::int32_t total_num_blocks_{0};
    bool enable_caching_{true};
    std::deque<internal_block_ref::BlockControl> controls_{};
    // Keep every control in an allocated list node so acquire/release can use allocation-free splice().
    internal_block_ref::BlockControl::ControlList free_list_{};
    internal_block_ref::BlockControl::ControlList in_use_{};
    std::unordered_map<std::string, std::vector<internal_block_ref::BlockControl*>> cache_index_{};
};

}  // namespace tokenspeed
