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

#include "resource/allocator/owned_pages.h"

#include "resource/allocator/page_allocator.h"

#include <algorithm>
#include <stdexcept>

namespace tokenspeed {

OwnedPages::~OwnedPages() {
    if (allocator_ && !ids_.empty()) {
        allocator_->Deallocate(ids_);
    }
}

OwnedPages& OwnedPages::operator=(OwnedPages&& other) noexcept {
    if (this != &other) {
        if (allocator_ && !ids_.empty()) {
            allocator_->Deallocate(ids_);
        }
        allocator_ = std::exchange(other.allocator_, nullptr);
        ids_ = std::move(other.ids_);
    }
    return *this;
}

OwnedPages OwnedPages::TakeFirst(std::int32_t n) {
    if (n < 0 || n > Size()) {
        throw std::out_of_range("OwnedPages::TakeFirst: count out of range; count=" + std::to_string(n) +
                                "; size=" + std::to_string(Size()));
    }
    std::vector<std::int32_t> taken(ids_.begin(), ids_.begin() + n);
    ids_.erase(ids_.begin(), ids_.begin() + n);
    return OwnedPages{allocator_, std::move(taken)};
}

OwnedPages OwnedPages::TakeLast(std::int32_t n) {
    if (n < 0 || n > Size()) {
        throw std::out_of_range("OwnedPages::TakeLast: count out of range; count=" + std::to_string(n) +
                                "; size=" + std::to_string(Size()));
    }
    std::vector<std::int32_t> taken(ids_.end() - n, ids_.end());
    ids_.erase(ids_.end() - n, ids_.end());
    return OwnedPages{allocator_, std::move(taken)};
}

void OwnedPages::Append(OwnedPages other) {
    if (other.ids_.empty()) return;
    if (allocator_ == nullptr) {
        allocator_ = other.allocator_;
    } else if (other.allocator_ != nullptr && allocator_ != other.allocator_) {
        throw std::logic_error("OwnedPages::Append: allocator mismatch between page owners");
    }
    ids_.insert(ids_.end(), other.ids_.begin(), other.ids_.end());
    other.allocator_ = nullptr;
    other.ids_.clear();
}

void OwnedPages::ReleaseOwnershipByID(const std::vector<std::int32_t>& ids) {
    for (std::int32_t id : ids) {
        auto it = std::find(ids_.begin(), ids_.end(), id);
        if (it != ids_.end()) {
            ids_.erase(it);
        }
    }
}

}  // namespace tokenspeed
