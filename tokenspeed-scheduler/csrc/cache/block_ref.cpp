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

#include <utility>

#include "cache/block_pool.h"
#include "utils.h"

namespace tokenspeed {

BlockRef::BlockRef(BlockControl* control) : control_{control} {
    Retain();
}

BlockRef::BlockRef(const BlockRef& other) : control_{other.control_} {
    Retain();
}

BlockRef& BlockRef::operator=(const BlockRef& other) {
    if (this != &other) {
        BlockRef copy{other};
        std::swap(control_, copy.control_);
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

BlockRef::~BlockRef() {
    reset();
}

CacheBlock* BlockRef::get() const {
    return control_ == nullptr ? nullptr : control_->object_;
}

std::uint32_t BlockRef::use_count() const {
    return control_ == nullptr ? 0 : control_->strong_count_;
}

void BlockRef::reset() {
    BlockControl* control = std::exchange(control_, nullptr);
    if (control == nullptr || control->object_->IsNull()) {
        return;
    }
    _assert(control->strong_count_ > 0, "BlockRef strong_count underflow");
    --control->strong_count_;
    if (control->strong_count_ == 0) {
        control->owner_->OnLastRef(control);
    }
}

bool BlockRef::SharesPoolWith(const BlockRef& other) const {
    return control_ != nullptr && other.control_ != nullptr && control_->owner_ == other.control_->owner_;
}

void BlockRef::Retain() {
    if (control_ == nullptr) {
        return;
    }
    _assert(control_->owner_ != nullptr && control_->object_ != nullptr, "BlockRef requires a valid control");
    if (!control_->object_->IsNull()) {
        ++control_->strong_count_;
    }
}

}  // namespace tokenspeed
