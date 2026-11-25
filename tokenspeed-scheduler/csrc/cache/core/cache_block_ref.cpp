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
    // The caller already owns a live reference. Publication and final retirement
    // synchronize separately, so incrementing requires only atomicity.
    const auto count = strong_count_.fetch_add(1, std::memory_order_relaxed) & ~kPending;
    FatalCheck(count != 0 && count < kPending - 1, "cannot retain a destroyed or saturated CacheBlockControl");
}

void CacheBlockControl::release() noexcept {
    const auto prior = strong_count_.fetch_sub(1, std::memory_order_acq_rel);
    FatalCheck((prior & ~kPending) != 0, "CacheBlockControl release requires a live reference");
    finishRelease(prior - 1);
}

void CacheBlockControl::finishRelease(std::uint32_t state) noexcept {
    if (state == 0) {
        while (unique_observers_ != nullptr) {
            unique_observers_->Reset();
        }
        object_.OwnerPool().destroyBlockControl(this);
        return;
    }
    if (state == 1 && unique_observers_ != nullptr) {
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

bool CacheBlockRef::IsOwnedBy(const BlockPool& pool) const noexcept {
    return control_ != nullptr && control_->isOwnedBy(pool);
}

void CacheBlockRef::reset() noexcept {
    internal_cache_block_ref::CacheBlockControl* control = std::exchange(control_, nullptr);
    if (control != nullptr) {
        control->release();
    }
}

// Only the peer holding this particular ref calls this. The one ref that
// sets kPending retains the control address for the cache owner's callback.
void CacheBlockRef::releaseFromPeer() noexcept {
    if (!control_) return;
    auto& count = control_->strong_count_;
    const auto pending = control_->kPending;
    auto prior = count.load(std::memory_order_relaxed);
    do {
        FatalCheck((prior & ~pending) != 0, "cannot unpin a released block");
    } while (!count.compare_exchange_weak(prior, (prior - 1) | pending, std::memory_order_acq_rel,
                                          std::memory_order_relaxed));
    if (prior & pending) control_ = nullptr;
}

void CacheBlockRef::finishPeerRelease() noexcept {
    if (auto* control = std::exchange(control_, nullptr)) {
        const auto pending = control->kPending;
        const auto prior = control->strong_count_.fetch_and(~pending, std::memory_order_acq_rel);
        FatalCheck((prior & pending) != 0, "peer release needs a pending callback");
        control->finishRelease(prior & ~pending);
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

void CacheBlockUniqueObserver::unlink() noexcept {
    if (previous_ != nullptr) {
        previous_->next_ = next_;
    } else {
        *head_ = next_;
    }
    if (next_ != nullptr) {
        next_->previous_ = previous_;
    }
    head_ = nullptr;
    previous_ = nullptr;
    next_ = nullptr;
    callback_ = nullptr;
    context_ = nullptr;
}

}  // namespace tokenspeed
