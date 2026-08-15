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

namespace tokenspeed {

// A token-free placement order for GroupAllocator. All token -> page
// arithmetic happens on the logical side (GroupGeometry in the coordinator
// layer); the allocator only executes block counts and bookkeeping values.
struct AcquirePlan {
    // Fresh blocks to acquire from the pool.
    std::int32_t num_blocks{0};
    // Dense append when < 0. Otherwise the logical slot the sparse suffix
    // starts at; slots below it stay null holes.
    std::int32_t suffix_start{-1};
    // Sparse only: the table's logical block count after placement.
    std::int32_t table_blocks_after{0};
    // Unconsumed tail capacity after placement, in tokens. The allocator
    // stores it verbatim; it never derives it.
    std::int32_t available_tokens_after{0};
};

}  // namespace tokenspeed
