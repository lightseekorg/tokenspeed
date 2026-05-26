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
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/types.h"
#include "resource/allocator/page_allocator.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"

namespace tokenspeed::test {

// Generate a consecutive token sequence for test inputs.
inline token_vec_t MakeTokens(int32_t count, token_t start = 1) {
    token_vec_t tokens(count);
    std::iota(tokens.begin(), tokens.end(), start);
    return tokens;
}

// Generate a token sequence aligned to page_size.
inline token_vec_t MakeAlignedTokens(int32_t num_pages, int32_t page_size, token_t start = 1) {
    return MakeTokens(num_pages * page_size, start);
}

inline std::vector<std::span<const std::int32_t>> MakePagedTokenSpans(const token_vec_t& tokens,
                                                                      std::int32_t page_size) {
    std::vector<std::span<const std::int32_t>> pages;
    const auto page_count = static_cast<std::int32_t>(tokens.size()) / page_size;
    pages.reserve(static_cast<std::size_t>(page_count));
    for (std::int32_t page = 0; page < page_count; ++page) {
        pages.emplace_back(tokens.data() + page * page_size, static_cast<std::size_t>(page_size));
    }
    return pages;
}

inline RecoveryPlan MatchPrefix(HybridPrefixCache& cache, const token_vec_t& tokens, std::int32_t page_size,
                                MatchIntent intent = MatchIntent::PrefixReuse) {
    auto pages = MakePagedTokenSpans(tokens, page_size);
    return cache.MatchPrefix(pages, intent);
}

// Generate page hashes of the requested length.
inline std::vector<std::string> MakePageHashes(std::size_t num_pages, std::string_view prefix = "h") {
    std::vector<std::string> page_hashes;
    page_hashes.reserve(num_pages);
    const std::string prefix_string(prefix);
    for (std::size_t i = 0; i < num_pages; ++i) {
        page_hashes.emplace_back(prefix_string + std::to_string(i + 1));
    }
    return page_hashes;
}

}  // namespace tokenspeed::test
