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

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <vector>

#include "resource/radix_tree/mamba_slot.h"

namespace tokenspeed {

class MambaHostAllocator {
public:
    explicit MambaHostAllocator(std::int32_t num_slots);

    std::optional<MambaSlot> Allocate();
    void Free(std::int32_t index);
    std::vector<std::int32_t> DrainReleased(std::size_t max = 1024);

    std::int32_t AvailableSlots() const { return static_cast<std::int32_t>(free_list_.size()); }
    std::int32_t TotalSlots() const { return total_slots_; }

private:
    // See mamba_chunk_allocator.h for the rationale: min-heap free list keeps
    // Allocate() output independent of Free() ordering so TP ranks stay in sync
    // even when slot-release callbacks fire at different times per rank.
    std::priority_queue<std::int32_t, std::vector<std::int32_t>, std::greater<std::int32_t>> free_list_;
    // DrainReleased() sorts before returning, so insertion order here doesn't
    // affect cross-rank determinism.
    std::vector<std::int32_t> released_idx_queue_;
    std::int32_t total_slots_;
};

}  // namespace tokenspeed
