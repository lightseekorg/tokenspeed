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
#include <vector>

namespace tokenspeed {

// Additional cache blocks one group must place. The three counts are disjoint.
struct BlockPlacementDemand {
    // Blocks in each specified bucket, such as Host loads or restored blocks.
    std::vector<std::int64_t> blocks_by_bucket;
    // Individual blocks that may each take any bucket.
    std::int64_t any_bucket_blocks{0};
    // Pairs whose bucket is chosen by the fit; both blocks must share it.
    // An admission pairs a growing tail block with its reserved copy-in block.
    std::int64_t same_bucket_pairs{0};

    std::int64_t TotalBlocks() const {
        std::int64_t blocks = any_bucket_blocks + 2 * same_bucket_pairs;
        for (const std::int64_t bucket_blocks : blocks_by_bucket) {
            blocks += bucket_blocks;
        }
        return blocks;
    }
};

}  // namespace tokenspeed
