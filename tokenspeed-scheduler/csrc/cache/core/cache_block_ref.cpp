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

#include "cache/core/cache_block_ref.h"

#include <limits>
#include <utility>

#include "cache/core/block_pool.h"
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
        while (unique_observers_ != nullptr) {
            unique_observers_->Reset();
        }
        delete this;
        return;
    }
    if (strong_count_ == 1 && unique_observers_ != nullptr) {
        // Detach before callbacks: a callback can cancel another observer or
        // release the last owner, including destroying this control.
        CacheBlockUniqueObserver* pending = std::exchange(unique_observers_, nullptr);
        for (auto* observer = pending; observer != nullptr; observer = observer->next_) {
            observer->head_ = &pending;
        }
        while (pending != nullptr) {
            const auto callback = pending->callback_;
            void* context = pending->context_;
            pending->Reset();
            callback(context);
        }
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

void CacheBlockUniqueObserver::Observe(const CacheBlockRef& block, Callback callback, void* context) noexcept {
    Reset();
    if (!block) {
        return;
    }
    FatalCheck(callback != nullptr, "a unique-owner observer requires a callback");
    if (block.unique()) {
        callback(context);
        return;
    }
    head_ = &block.control_->unique_observers_;
    next_ = *head_;
    if (next_ != nullptr) {
        next_->previous_ = this;
    }
    *head_ = this;
    callback_ = callback;
    context_ = context;
}

void CacheBlockUniqueObserver::Reset() noexcept {
    if (head_ != nullptr) {
        if (previous_ != nullptr) {
            previous_->next_ = next_;
        } else {
            *head_ = next_;
        }
        if (next_ != nullptr) {
            next_->previous_ = previous_;
        }
    }
    head_ = nullptr;
    previous_ = nullptr;
    next_ = nullptr;
    callback_ = nullptr;
    context_ = nullptr;
}

}  // namespace tokenspeed
