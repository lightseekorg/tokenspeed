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

#include "req_pool_allocator.h"

#include <stdexcept>

namespace tokenspeed {

ReqPoolIndex::ReqPoolIndex(std::int32_t slot, ReqPoolAllocator* allocator) : slot_(slot), allocator_(allocator) {}

ReqPoolIndex::ReqPoolIndex(ReqPoolIndex&& other) noexcept : slot_(other.slot_), allocator_(other.allocator_) {
    other.allocator_ = nullptr;
    other.slot_ = -1;
}

ReqPoolIndex& ReqPoolIndex::operator=(ReqPoolIndex&& other) noexcept {
    if (this != &other) {
        if (allocator_) {
            allocator_->deAllocate(slot_);
        }
        slot_ = other.slot_;
        allocator_ = other.allocator_;
        other.slot_ = -1;
        other.allocator_ = nullptr;
    }
    return *this;
}

ReqPoolIndex::~ReqPoolIndex() {
    if (allocator_) {
        allocator_->deAllocate(slot_);
    }
}

bool ReqPoolIndex::valid() const {
    return allocator_ != nullptr;
}

ReqPoolAllocator::ReqPoolAllocator(std::int32_t size) : size_(size) {
    // Slot 0 is conventionally reserved (matches Python which starts from index 1).
    for (std::int32_t i = 1; i < size + 1; ++i) {
        free_slots_.push_back(i);
    }
}

ReqPoolIndex ReqPoolAllocator::Allocate() {
    if (free_slots_.empty()) {
        throw std::runtime_error("ReqPoolAllocator::Allocate: no request pool slots available; capacity=" +
                                 std::to_string(size_));
    }
    std::int32_t slot = free_slots_.front();
    free_slots_.pop_front();
    return ReqPoolIndex{slot, this};
}

std::int32_t ReqPoolAllocator::Size() const {
    return size_;
}

std::int32_t ReqPoolAllocator::AvailableSlots() const {
    return static_cast<std::int32_t>(free_slots_.size());
}

void ReqPoolAllocator::deAllocate(std::int32_t slot) {
    free_slots_.push_back(slot);
}

}  // namespace tokenspeed
