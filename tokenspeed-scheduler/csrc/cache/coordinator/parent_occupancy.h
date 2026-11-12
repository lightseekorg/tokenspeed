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
// belongs to the shadow's scratch lease: one array of parent/count slots that
// doubles at 75% occupancy. No individual node allocations or deletions are
// needed: a restored parent keeps its count until the plan dies.
class ParentOccupancy {
public:
    explicit ParentOccupancy(std::pmr::memory_resource* resource) : slots_{resource} {}

    std::int32_t& GetOrInsert(std::int32_t parent, std::int32_t initial) {
        _assert(parent > 0, "the reserved null parent has no shadow occupancy");
        if (slots_.empty()) grow();
        auto found = find(slots_, parent, mixed_hash_);
        // Dense ranges benefit from adjacent slots. If a strided range starts
        // clustering, rehash once with mixed IDs for the rest of this shadow.
        if (!mixed_hash_ && found.probes > 8) {
            rehash(slots_.size(), true);
            found = find(slots_, parent, mixed_hash_);
        }
        auto position = found.position;
        if (slots_[position].parent == parent) return slots_[position].occupied;
        if ((size_ + 1) * 4 > slots_.size() * 3) {
            grow();
            position = find(slots_, parent, mixed_hash_).position;
        }
        auto& slot = slots_[position];
        slot = Slot{parent, initial};
        ++size_;
        return slot.occupied;
    }

    std::int32_t* Find(std::int32_t parent) {
        if (slots_.empty()) return nullptr;
        auto& slot = slots_[find(slots_, parent, mixed_hash_).position];
        return slot.parent == parent ? &slot.occupied : nullptr;
    }

    std::size_t size() const noexcept { return size_; }

private:
    struct Slot {
        std::int32_t parent{0};
        std::int32_t occupied{0};
    };

    struct Search {
        std::size_t position;
        std::size_t probes;
    };

    static Search find(const std::pmr::vector<Slot>& slots, std::int32_t parent, bool mixed) {
        auto hash = static_cast<std::uint64_t>(parent);
        if (mixed) {
            hash *= 0x9e3779b97f4a7c15ULL;
            hash ^= hash >> 32U;
        } else {
            hash ^= hash >> 16U;
        }
        const auto mask = slots.size() - 1;
        auto position = static_cast<std::size_t>(hash) & mask;
        std::size_t probes = 1;
        while (slots[position].parent != 0 && slots[position].parent != parent) {
            position = (position + 1) & mask;
            ++probes;
        }
        return Search{position, probes};
    }

    void grow() { rehash(slots_.empty() ? 16 : 2 * slots_.size(), mixed_hash_); }

    void rehash(std::size_t capacity, bool mixed) {
        std::pmr::vector<Slot> next{slots_.get_allocator()};
        next.resize(capacity);
        for (const Slot& slot : slots_) {
            if (slot.parent != 0) next[find(next, slot.parent, mixed).position] = slot;
        }
        slots_.swap(next);
        mixed_hash_ = mixed;
    }

    std::pmr::vector<Slot> slots_;
    std::size_t size_{0};
    bool mixed_hash_{false};
};

}  // namespace tokenspeed
