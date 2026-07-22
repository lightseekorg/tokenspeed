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
#include <list>
#include <string>

namespace tokenspeed {

class BlockPool;
class BlockRef;

class CacheBlock {
public:
    explicit CacheBlock(std::int32_t block_id) : block_id_{block_id} {}

    std::int32_t BlockId() const noexcept { return block_id_; }
    bool IsCached() const noexcept { return !block_hash_.empty(); }
    const std::string& BlockHash() const noexcept { return block_hash_; }

    void SetHash(std::string hash);
    void ResetHash() noexcept { block_hash_.clear(); }

private:
    std::int32_t block_id_{0};
    std::string block_hash_{};
};

namespace internal_block_ref {

// A lightweight std::shared_ptr-like control block that also embeds the CacheBlock.
// BlockPool owns and recycles the control after the last BlockRef releases it.
class BlockControl {
public:
    using ReturnToPoolHandler = void (*)(BlockPool&, BlockControl&) noexcept;

    BlockControl(std::int32_t block_id, BlockPool& owner_pool, ReturnToPoolHandler return_to_pool) noexcept;
    BlockControl(const BlockControl&) = delete;
    BlockControl& operator=(const BlockControl&) = delete;
    BlockControl(BlockControl&&) = delete;
    BlockControl& operator=(BlockControl&&) = delete;
    ~BlockControl() = default;

private:
    friend class ::tokenspeed::BlockPool;
    friend class ::tokenspeed::BlockRef;

    using ControlList = std::list<BlockControl*>;
    using ListPosition = ControlList::iterator;

    void Retain() noexcept;
    void Release() noexcept;
    std::uint32_t UseCount() const noexcept { return strong_count_; }

    CacheBlock& Object() noexcept { return object_; }
    const CacheBlock& Object() const noexcept { return object_; }
    bool IsOwnedBy(const BlockPool& pool) const noexcept { return owner_pool_ == &pool; }

    bool InFreeList() const noexcept { return in_free_list_; }
    ListPosition Position() const noexcept { return position_; }
    void SetPosition(ListPosition position) noexcept { position_ = position; }
    void MarkFree() noexcept;
    void MarkInUse() noexcept;

private:
    CacheBlock object_;
    BlockPool* owner_pool_{nullptr};
    ReturnToPoolHandler return_to_pool_{nullptr};
    std::uint32_t strong_count_{0};
    bool in_free_list_{false};
    ListPosition position_{};
};

}  // namespace internal_block_ref

// Pool-scoped shared owner. Its BlockPool must outlive every non-empty copy.
class BlockRef {
public:
    BlockRef() noexcept = default;
    BlockRef(const BlockRef& other) noexcept;
    BlockRef& operator=(const BlockRef& other) noexcept;
    BlockRef(BlockRef&& other) noexcept;
    BlockRef& operator=(BlockRef&& other) noexcept;
    ~BlockRef() noexcept;

    const CacheBlock* operator->() const noexcept;
    const CacheBlock& operator*() const noexcept;
    explicit operator bool() const noexcept { return control_ != nullptr; }

    std::uint32_t use_count() const noexcept;
    bool unique() const noexcept { return use_count() == 1; }
    bool IsOwnedBy(const BlockPool& pool) const noexcept;
    void reset() noexcept;
    void swap(BlockRef& other) noexcept;

    bool operator==(const BlockRef&) const noexcept = default;

private:
    friend class BlockPool;

    explicit BlockRef(internal_block_ref::BlockControl& control) noexcept;

    internal_block_ref::BlockControl* control_{nullptr};
};

inline void swap(BlockRef& lhs, BlockRef& rhs) noexcept {
    lhs.swap(rhs);
}

}  // namespace tokenspeed
