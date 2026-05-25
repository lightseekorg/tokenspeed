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

#include "resource/allocator/mamba_chunk_allocator.h"

namespace tokenspeed {

MambaSlot::MambaSlot(std::int32_t index, MambaChunkAllocator* allocator)
    : MambaSlot(index, [allocator](std::int32_t i) {
          if (allocator != nullptr) allocator->Free(i);
      }) {}

MambaChunkAllocator::MambaChunkAllocator(std::int32_t num_slots) : total_slots_{num_slots} {
    free_list_.reserve(num_slots);
    for (std::int32_t i = num_slots - 1; i >= 0; --i) {
        free_list_.push_back(i);
    }
}

std::optional<MambaSlot> MambaChunkAllocator::Allocate() {
    if (free_list_.empty()) {
        return std::nullopt;
    }
    std::int32_t index = free_list_.back();
    free_list_.pop_back();
    return MambaSlot{index, this};
}

void MambaChunkAllocator::Free(std::int32_t index) {
    free_list_.push_back(index);
}

MambaSlot::~MambaSlot() {
    release();
}

void MambaSlot::release() {
    if (index_ >= 0 && releaser_) {
        releaser_(index_);
        index_ = -1;
        releaser_ = {};
    }
}

}  // namespace tokenspeed
