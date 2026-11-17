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

#include "cache/core/cache_block_ref.h"

namespace tokenspeed {

// Device blocks an admission or restore plan evicts together, in the order
// they leave: one block, or, when the coordinator evicts by segments, a
// whole segment -- deepest node first, a node's State checkpoints before its
// other blocks.
struct VictimSet {
    struct Member {
        std::uint32_t group_id;
        CacheBlockLocation location;
        // Whether it lies strictly under the boundary of the candidate the
        // set was chosen for, taken along so no Device page outlives its
        // parent (counted as a stranded suffix).
        bool in_stranded_suffix;
        // Kept per block when a segment absorbs the request's released slots.
        bool is_request_reclaim;
    };
    explicit VictimSet(std::pmr::memory_resource* resource) : members{resource} {}
    std::pmr::vector<Member> members;
};

}  // namespace tokenspeed
