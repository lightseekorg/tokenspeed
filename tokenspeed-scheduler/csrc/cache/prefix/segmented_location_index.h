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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cache/core/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

// A pool/group-scoped, sparse direct index. Packing is fixed for this group
// in this pool; parent rebinding cannot alias another group's child slots.
// Only touched segments allocate values. T{} denotes an absent entry.
template <typename T>
class SegmentedLocationIndex {
public:
    SegmentedLocationIndex(std::int32_t packing, std::int32_t parents) : packing_{packing}, parents_{parents} {}

    T Find(CacheBlockLocation location) const noexcept {
        if (!valid(location)) return {};
        const auto id = offset(location);
        const auto outer = id / (kSegmentSize * kDirectorySize);
        if (outer >= directories_.size() || !directories_[outer]) return {};
        const auto& segment = directories_[outer]->segments[id / kSegmentSize % kDirectorySize];
        return segment ? segment->values[id % kSegmentSize] : T{};
    }

    void Set(CacheBlockLocation location, T value) {
        _assert(valid(location) && value != T{}, "a location index stores a live entry at a valid slot");
        const auto id = offset(location);
        const auto outer = id / (kSegmentSize * kDirectorySize);
        if (outer >= directories_.size()) directories_.resize(outer + 1);
        auto& directory = directories_[outer];
        if (!directory) directory = std::make_unique<Directory>();
        auto& segment = directory->segments[id / kSegmentSize % kDirectorySize];
        if (!segment) {
            segment = std::make_unique<Segment>();
            ++directory->size;
        }
        auto& entry = segment->values[id % kSegmentSize];
        if (entry == T{}) ++segment->size;
        entry = value;
    }

    void Erase(CacheBlockLocation location) noexcept {
        if (!valid(location)) return;
        const auto id = offset(location);
        const auto outer = id / (kSegmentSize * kDirectorySize);
        if (outer >= directories_.size()) return;
        auto& directory = directories_[outer];
        if (!directory) return;
        auto& segment = directory->segments[id / kSegmentSize % kDirectorySize];
        if (!segment || segment->values[id % kSegmentSize] == T{}) return;
        segment->values[id % kSegmentSize] = T{};
        if (--segment->size == 0) {
            segment.reset();
            if (--directory->size == 0) directory.reset();
        }
    }

private:
    static constexpr std::size_t kSegmentSize = 1024;
    static constexpr std::size_t kDirectorySize = 256;
    struct Segment {
        std::array<T, kSegmentSize> values{};
        std::size_t size{0};
    };
    struct Directory {
        std::array<std::unique_ptr<Segment>, kDirectorySize> segments{};
        std::size_t size{0};
    };
    bool valid(CacheBlockLocation location) const noexcept {
        return location.lcm_block_id > 0 && location.lcm_block_id <= parents_ && location.slot_index >= 0 &&
               location.slot_index < packing_;
    }
    std::size_t offset(CacheBlockLocation location) const noexcept {
        return static_cast<std::size_t>(location.lcm_block_id - 1) * static_cast<std::size_t>(packing_) +
               static_cast<std::size_t>(location.slot_index);
    }

    std::int32_t packing_;
    std::int32_t parents_;
    std::vector<std::unique_ptr<Directory>> directories_;
};

}  // namespace tokenspeed
