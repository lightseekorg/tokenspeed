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

#include "cache/cache_block_ref.h"

#include <limits>
#include <utility>

#include "cache/block_pool.h"
#include "utils.h"

namespace tokenspeed {

CacheBlock::~CacheBlock() noexcept {
    FatalCheck(pool_ != nullptr, "CacheBlock requires a live BlockPool");
    pool_->Release(location_);
}

namespace internal_cache_block_ref {

void CacheBlockControl::retain() noexcept {
    FatalCheck(strong_count_ != 0 && strong_count_ != std::numeric_limits<std::uint32_t>::max(),
               "cannot retain a destroyed or saturated CacheBlockControl");
    ++strong_count_;
}

void CacheBlockControl::release() noexcept {
    FatalCheck(strong_count_ != 0, "CacheBlockControl release requires a live reference");
    --strong_count_;
    if (strong_count_ == 0) {
        delete this;
    }
}

bool CacheBlockControl::isOwnedBy(const BlockPool& pool) const noexcept {
    return object_.IsOwnedBy(pool);
}

}  // namespace internal_cache_block_ref

CacheBlockRef::CacheBlockRef(const CacheBlockRef& other) noexcept : control_{other.control_} {
    if (control_ != nullptr) {
        control_->retain();
    }
}

CacheBlockRef& CacheBlockRef::operator=(const CacheBlockRef& other) noexcept {
    if (this != &other) {
        CacheBlockRef copy{other};
        swap(copy);
    }
    return *this;
}

CacheBlockRef::CacheBlockRef(CacheBlockRef&& other) noexcept : control_{std::exchange(other.control_, nullptr)} {}

CacheBlockRef& CacheBlockRef::operator=(CacheBlockRef&& other) noexcept {
    if (this != &other) {
        reset();
        control_ = std::exchange(other.control_, nullptr);
    }
    return *this;
}

CacheBlockRef::~CacheBlockRef() noexcept {
    reset();
}

const CacheBlock* CacheBlockRef::operator->() const noexcept {
    return control_ == nullptr ? nullptr : &control_->object();
}

const CacheBlock& CacheBlockRef::operator*() const noexcept {
    return control_->object();
}

std::uint32_t CacheBlockRef::use_count() const noexcept {
    return control_ == nullptr ? 0 : control_->useCount();
}

bool CacheBlockRef::IsOwnedBy(const BlockPool& pool) const noexcept {
    return control_ != nullptr && control_->isOwnedBy(pool);
}

void CacheBlockRef::reset() noexcept {
    internal_cache_block_ref::CacheBlockControl* control = std::exchange(control_, nullptr);
    if (control != nullptr) {
        control->release();
    }
}

void CacheBlockRef::swap(CacheBlockRef& other) noexcept {
    std::swap(control_, other.control_);
}

}  // namespace tokenspeed
