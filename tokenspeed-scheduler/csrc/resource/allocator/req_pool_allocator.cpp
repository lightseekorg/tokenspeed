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

#include <limits>
#include <stdexcept>
#include <utility>

namespace tokenspeed {

ReqPoolIndex::ReqPoolIndex(std::int32_t slot, std::uint64_t generation, std::uint64_t previous_admission_id,
                           ReqPoolAllocator* allocator)
    : slot_(slot), allocator_(allocator), generation_(generation), previous_admission_id_(previous_admission_id) {}

ReqPoolIndex::ReqPoolIndex(ReqPoolIndex&& other) noexcept
    : slot_(std::exchange(other.slot_, -1)),
      allocator_(std::exchange(other.allocator_, nullptr)),
      generation_(std::exchange(other.generation_, 0)),
      previous_admission_id_(other.previous_admission_id_),
      admission_id_(other.admission_id_),
      request_id_(std::move(other.request_id_)),
      bound_(other.bound_) {}

ReqPoolIndex& ReqPoolIndex::operator=(ReqPoolIndex&& other) noexcept {
    if (this != &other) {
        release();
        slot_ = std::exchange(other.slot_, -1);
        allocator_ = std::exchange(other.allocator_, nullptr);
        generation_ = std::exchange(other.generation_, 0);
        previous_admission_id_ = other.previous_admission_id_;
        admission_id_ = other.admission_id_;
        request_id_ = std::move(other.request_id_);
        bound_ = other.bound_;
    }
    return *this;
}

ReqPoolIndex::~ReqPoolIndex() {
    release();
}

SlotOwnershipEvent ReqPoolIndex::ownership() const {
    return {slot_, generation_, admission_id_, previous_admission_id_, request_id_};
}

void ReqPoolIndex::release() noexcept {
    if (allocator_) {
        if (bound_ && allocator_->event_sink_) {
            allocator_->event_sink_->OnSlotReleased(ownership());
        }
        allocator_->deallocate(slot_);
        allocator_ = nullptr;
    }
}

void ReqPoolIndex::BindOwner(std::string_view request_id, std::uint64_t admission_id) {
    if (!valid() || bound_) {
        throw std::logic_error("request pool ownership requires a fresh allocated handle");
    }
    if (allocator_->event_sink_ && admission_id == 0) {
        throw std::invalid_argument("observed request pool ownership requires a nonzero admission identity");
    }
    admission_id_ = admission_id;
    allocator_->slot_history_[slot_ - 1].admission_id = admission_id;
    bound_ = true;
    if (allocator_->event_sink_) {
        request_id_ = request_id;
        allocator_->event_sink_->OnSlotAssigned(ownership());
    }
}

bool ReqPoolIndex::valid() const {
    return allocator_ != nullptr;
}

ReqPoolAllocator::ReqPoolAllocator(std::int32_t size) : size_(size) {
    if (size < 0) {
        throw std::invalid_argument("request pool capacity must be nonnegative");
    }
    slot_history_.resize(static_cast<std::size_t>(size));
    // Slot 0 is conventionally reserved (matches Python which starts from
    // index 1): slots 1..size, counted so that INT32_MAX ends the loop.
    for (std::int32_t i = 0; i < size; ++i) {
        free_slots_.push_back(i + 1);
    }
}

void ReqPoolAllocator::SetEventSink(SlotEventSink* sink) {
    if (allocated_) {
        throw std::logic_error("install the slot event sink before allocating any slots");
    }
    event_sink_ = sink;
}

ReqPoolIndex ReqPoolAllocator::Allocate() {
    if (free_slots_.empty()) {
        throw std::runtime_error("ReqPoolAllocator::Allocate: no request pool slots available; capacity=" +
                                 std::to_string(size_));
    }
    const std::int32_t slot = free_slots_.front();
    SlotHistory& history = slot_history_[slot - 1];
    if (history.generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("request pool slot generation exhausted");
    }
    ++history.generation;
    allocated_ = true;
    free_slots_.pop_front();
    return ReqPoolIndex{slot, history.generation, history.admission_id, this};
}

std::int32_t ReqPoolAllocator::Size() const {
    return size_;
}
std::int32_t ReqPoolAllocator::AvailableSlots() const {
    return static_cast<std::int32_t>(free_slots_.size());
}

void ReqPoolAllocator::deallocate(std::int32_t slot) {
    free_slots_.push_back(slot);
}

}  // namespace tokenspeed
