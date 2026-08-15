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
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

namespace tokenspeed::test {

// Generate a consecutive token sequence for test inputs.
inline std::vector<std::int32_t> MakeTokens(int32_t count, std::int32_t start = 1) {
    std::vector<std::int32_t> tokens(count);
    std::iota(tokens.begin(), tokens.end(), start);
    return tokens;
}

// Generate a token sequence aligned to granularity.
inline std::vector<std::int32_t> MakeAlignedTokens(int32_t num_pages, int32_t granularity, std::int32_t start = 1) {
    return MakeTokens(num_pages * granularity, start);
}

}  // namespace tokenspeed::test
