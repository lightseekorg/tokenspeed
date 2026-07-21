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

#include "cache/block_ref.h"

#include <exception>
#include <utility>

#include "utils.h"

namespace tokenspeed {

void CacheBlock::SetHash(std::string hash) {
    _assert(block_hash_.empty(), "block already has a hash");
    block_hash_ = std::move(hash);
}

namespace detail {

BlockControl::BlockControl(std::int32_t block_id, BlockPool& owner_pool, ReturnToPoolHandler return_to_pool) noexcept
    : object_{block_id}, owner_pool_{&owner_pool}, return_to_pool_{return_to_pool} {
    if (return_to_pool_ == nullptr) {
        std::terminate();
    }
}

void BlockControl::Retain() noexcept {
    if (owner_pool_ == nullptr || (strong_count_ == 0 && in_free_list_)) {
        std::terminate();
    }
    ++strong_count_;
}

void BlockControl::Release() noexcept {
    if (strong_count_ == 0 || in_free_list_) {
        std::terminate();
    }
    --strong_count_;
    if (strong_count_ == 0) {
        return_to_pool_(*owner_pool_, *this);
    }
}

void BlockControl::MarkFree() noexcept {
    if (in_free_list_ || strong_count_ != 0) {
        std::terminate();
    }
    in_free_list_ = true;
}

void BlockControl::MarkInUse() noexcept {
    if (!in_free_list_ || strong_count_ != 0) {
        std::terminate();
    }
    in_free_list_ = false;
}

}  // namespace detail

BlockRef::BlockRef(detail::BlockControl& control) noexcept : control_{&control} {
    control_->Retain();
}

BlockRef::BlockRef(const BlockRef& other) noexcept : control_{other.control_} {
    if (control_ != nullptr) {
        control_->Retain();
    }
}

BlockRef& BlockRef::operator=(const BlockRef& other) noexcept {
    if (this != &other) {
        BlockRef copy{other};
        swap(copy);
    }
    return *this;
}

BlockRef::BlockRef(BlockRef&& other) noexcept : control_{std::exchange(other.control_, nullptr)} {}

BlockRef& BlockRef::operator=(BlockRef&& other) noexcept {
    if (this != &other) {
        reset();
        control_ = std::exchange(other.control_, nullptr);
    }
    return *this;
}

BlockRef::~BlockRef() noexcept {
    reset();
}

const CacheBlock* BlockRef::operator->() const noexcept {
    return control_ == nullptr ? nullptr : &control_->Object();
}

const CacheBlock& BlockRef::operator*() const noexcept {
    return control_->Object();
}

std::uint32_t BlockRef::use_count() const noexcept {
    return control_ == nullptr ? 0 : control_->UseCount();
}

bool BlockRef::IsOwnedBy(const BlockPool& pool) const noexcept {
    return control_ != nullptr && control_->IsOwnedBy(pool);
}

void BlockRef::reset() noexcept {
    detail::BlockControl* control = std::exchange(control_, nullptr);
    if (control != nullptr) {
        control->Release();
    }
}

void BlockRef::swap(BlockRef& other) noexcept {
    std::swap(control_, other.control_);
}

}  // namespace tokenspeed
