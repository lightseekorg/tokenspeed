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

namespace tokenspeed {

class BlockPool;
class CacheBlock;
class BlockTable;

// Stable control block owned by BlockPool. The scheduler is the sole writer, so
// the count deliberately stays non-atomic.
struct BlockControl {
    BlockPool* owner{nullptr};
    CacheBlock* object{nullptr};
    std::uint32_t strong_count{0};
};

// Shared owning handle to one pool-owned CacheBlock. Like std::shared_ptr,
// copying shares ownership, moving transfers it, and the last reset/destructor
// returns the block to its pool. The CacheBlock object itself is never deleted,
// and every handle must be destroyed before its owning BlockPool.
class BlockRef {
public:
    BlockRef() = default;
    BlockRef(const BlockRef& other);
    BlockRef& operator=(const BlockRef& other);
    BlockRef(BlockRef&& other) noexcept;
    BlockRef& operator=(BlockRef&& other) noexcept;
    ~BlockRef();

    CacheBlock* get() const;
    CacheBlock* operator->() const { return get(); }
    explicit operator bool() const { return control_ != nullptr; }

    std::uint32_t use_count() const;
    bool unique() const { return use_count() == 1; }
    void reset();

private:
    friend class BlockPool;
    friend class BlockTable;

    explicit BlockRef(BlockControl* control);
    void Retain();

    BlockControl* control_{nullptr};
};

}  // namespace tokenspeed
