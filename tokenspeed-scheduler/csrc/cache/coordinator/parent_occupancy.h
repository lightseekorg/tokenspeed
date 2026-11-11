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

#include <cstdint>
#include <memory_resource>
#include <vector>

#include "utils.h"

namespace tokenspeed {

// Tentative counts for touched physical parents. Slots own only integers,
// never blocks. Storage scales with touched parents, not pool capacity, and
// belongs to the shadow's scratch lease. No individual node allocations or
// deletions are needed: a restored parent keeps its count until the plan dies.
class ParentOccupancy {
public:
    explicit ParentOccupancy(std::pmr::memory_resource* resource) : slots_{resource} {}

    std::int32_t& GetOrInsert(std::int32_t parent, std::int32_t initial) {
        _assert(parent > 0, "the reserved null parent has no shadow occupancy");
        if (slots_.empty()) grow();
        auto position = find(slots_, parent);
        if (slots_[position].parent == parent) return slots_[position].occupied;
        if ((size_ + 1) * 4 > slots_.size() * 3) {
            grow();
            position = find(slots_, parent);
        }
        auto& slot = slots_[position];
        slot = Slot{parent, initial};
        ++size_;
        return slot.occupied;
    }

    std::int32_t* Find(std::int32_t parent) {
        if (slots_.empty()) return nullptr;
        auto& slot = slots_[find(slots_, parent)];
        return slot.parent == parent ? &slot.occupied : nullptr;
    }

    std::size_t size() const noexcept { return size_; }

private:
    struct Slot {
        std::int32_t parent{0};
        std::int32_t occupied{0};
    };

    static std::size_t find(const std::pmr::vector<Slot>& slots, std::int32_t parent) {
        auto hash = static_cast<std::uint32_t>(parent);
        // Keep consecutive physical parents consecutive in this small table,
        // while folding upper bits for sparse, regularly strided locations.
        hash ^= hash >> 16U;
        const auto mask = slots.size() - 1;
        auto position = static_cast<std::size_t>(hash) & mask;
        while (slots[position].parent != 0 && slots[position].parent != parent) {
            position = (position + 1) & mask;
        }
        return position;
    }

    void grow() {
        std::pmr::vector<Slot> next{slots_.get_allocator()};
        next.resize(slots_.empty() ? 16 : 2 * slots_.size());
        for (const Slot& slot : slots_) {
            if (slot.parent != 0) next[find(next, slot.parent)] = slot;
        }
        slots_.swap(next);
    }

    std::pmr::vector<Slot> slots_;
    std::size_t size_{0};
};

}  // namespace tokenspeed
