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

#include "resource/allocator/mamba_host_allocator.h"

#include <algorithm>

namespace tokenspeed {

MambaHostAllocator::MambaHostAllocator(std::int32_t num_slots) : total_slots_{num_slots} {
    free_list_.reserve(num_slots);
    released_idx_queue_.reserve(num_slots);
    for (std::int32_t i = num_slots - 1; i >= 0; --i) {
        free_list_.push_back(i);
    }
}

std::optional<MambaSlot> MambaHostAllocator::Allocate() {
    if (free_list_.empty()) {
        return std::nullopt;
    }
    std::int32_t index = free_list_.back();
    free_list_.pop_back();
    return MambaSlot{index, [this](std::int32_t i) { Free(i); }};
}

void MambaHostAllocator::Free(std::int32_t index) {
    if (index < 0 || index >= total_slots_) {
        return;
    }
    free_list_.push_back(index);
    released_idx_queue_.push_back(index);
}

std::vector<std::int32_t> MambaHostAllocator::DrainReleased(std::size_t max) {
    std::size_t n = std::min(max, released_idx_queue_.size());
    std::vector<std::int32_t> released(released_idx_queue_.begin(), released_idx_queue_.begin() + n);
    released_idx_queue_.erase(released_idx_queue_.begin(), released_idx_queue_.begin() + n);
    return released;
}

}  // namespace tokenspeed
