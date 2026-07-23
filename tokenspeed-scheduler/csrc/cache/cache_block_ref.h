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
class CacheBlockRef;

using GroupId = std::uint32_t;

// Stable logical placement of one cache block inside an LCM-sized physical
// block. LCM block 0 remains reserved as the kernel null page.
struct CacheBlockLocation {
    std::int32_t lcm_block_id{0};
    std::int32_t slot_index{0};

    bool operator==(const CacheBlockLocation&) const noexcept = default;
};

class CacheBlock {
public:
    CacheBlock(BlockPool& pool, CacheBlockLocation location) noexcept : pool_{&pool}, location_{location} {}
    CacheBlock(const CacheBlock&) = delete;
    CacheBlock& operator=(const CacheBlock&) = delete;
    ~CacheBlock() noexcept;

    CacheBlockLocation Location() const noexcept { return location_; }
    bool IsOwnedBy(const BlockPool& pool) const noexcept { return pool_ == &pool; }

private:
    BlockPool* pool_{nullptr};
    CacheBlockLocation location_{};
};

namespace internal_cache_block_ref {

// Dynamically allocated shared-ownership control. The embedded CacheBlock is
// destroyed together with the control when the last CacheBlockRef releases it.
class CacheBlockControl {
public:
    CacheBlockControl(BlockPool& owner_pool, CacheBlockLocation location) noexcept : object_{owner_pool, location} {}
    CacheBlockControl(const CacheBlockControl&) = delete;
    CacheBlockControl& operator=(const CacheBlockControl&) = delete;
    CacheBlockControl(CacheBlockControl&&) = delete;
    CacheBlockControl& operator=(CacheBlockControl&&) = delete;
    ~CacheBlockControl() = default;

private:
    friend class ::tokenspeed::BlockPool;
    friend class ::tokenspeed::CacheBlockRef;

    void retain() noexcept;
    void release() noexcept;
    std::uint32_t useCount() const noexcept { return strong_count_; }

    CacheBlock& object() noexcept { return object_; }
    const CacheBlock& object() const noexcept { return object_; }
    bool isOwnedBy(const BlockPool& pool) const noexcept;

    CacheBlock object_;
    std::uint32_t strong_count_{1};
};

}  // namespace internal_cache_block_ref

// Pool-scoped shared owner. Its BlockPool must outlive every non-empty copy.
class CacheBlockRef {
public:
    CacheBlockRef() noexcept = default;
    CacheBlockRef(const CacheBlockRef& other) noexcept;
    CacheBlockRef& operator=(const CacheBlockRef& other) noexcept;
    CacheBlockRef(CacheBlockRef&& other) noexcept;
    CacheBlockRef& operator=(CacheBlockRef&& other) noexcept;
    ~CacheBlockRef() noexcept;

    const CacheBlock* operator->() const noexcept;
    const CacheBlock& operator*() const noexcept;
    explicit operator bool() const noexcept { return control_ != nullptr; }

    std::uint32_t use_count() const noexcept;
    bool unique() const noexcept { return use_count() == 1; }
    bool IsOwnedBy(const BlockPool& pool) const noexcept;
    void reset() noexcept;
    void swap(CacheBlockRef& other) noexcept;

    bool operator==(const CacheBlockRef&) const noexcept = default;

private:
    friend class BlockPool;

    explicit CacheBlockRef(internal_cache_block_ref::CacheBlockControl& control) noexcept : control_{&control} {}

    internal_cache_block_ref::CacheBlockControl* control_{nullptr};
};

inline void swap(CacheBlockRef& lhs, CacheBlockRef& rhs) noexcept {
    lhs.swap(rhs);
}

}  // namespace tokenspeed
