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

#include <utility>

#include "cache/block_pool.h"

namespace tokenspeed {

// Move-only RAII reference to one CacheBlock. Adopt takes over an already-counted ref (Acquire's);
// Share claims a new one (TouchBlock; null blocks are held but never counted). The destructor
// releases; Release() surrenders the block so batch sites keep one ordered pool FreeBlocks call.
class BlockRef {
public:
    BlockRef() = default;

    static BlockRef Adopt(BlockPool& pool, CacheBlock* block) { return BlockRef{&pool, block}; }

    static BlockRef Share(BlockPool& pool, CacheBlock* block) {
        pool.TouchBlock(block);
        return BlockRef{&pool, block};
    }

    BlockRef(BlockRef&& other) noexcept
        : pool_{std::exchange(other.pool_, nullptr)}, block_{std::exchange(other.block_, nullptr)} {}

    BlockRef& operator=(BlockRef&& other) noexcept {
        if (this != &other) {
            reset();
            pool_ = std::exchange(other.pool_, nullptr);
            block_ = std::exchange(other.block_, nullptr);
        }
        return *this;
    }

    BlockRef(const BlockRef&) = delete;
    BlockRef& operator=(const BlockRef&) = delete;

    ~BlockRef() { reset(); }

    CacheBlock* Get() const { return block_; }
    CacheBlock* operator->() const { return block_; }

    CacheBlock* Release() {
        pool_ = nullptr;
        return std::exchange(block_, nullptr);
    }

private:
    friend class BlockTable;

    BlockRef(BlockPool* pool, CacheBlock* block) : pool_{pool}, block_{block} {}

    void reset() {
        if (block_ != nullptr) {
            pool_->FreeBlock(block_);
        }
        pool_ = nullptr;
        block_ = nullptr;
    }

    BlockPool* pool_{nullptr};
    CacheBlock* block_{nullptr};
};

}  // namespace tokenspeed
