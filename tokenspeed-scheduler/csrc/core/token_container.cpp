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

#include "core/token_container.h"

#include <algorithm>
#include <cstddef>

#include "utils.h"

namespace tokenspeed {

void TokenContainer::Extend(const std::vector<std::int32_t>& new_tokens) {
    tokens_.insert(tokens_.end(), new_tokens.begin(), new_tokens.end());
}

std::vector<std::span<const std::int32_t>> TokenContainer::FullPrefixPages(
    std::int32_t prefix_granularity, bool except_last, std::int32_t prefix_hash_lookahead_tokens) const {
    _assert(prefix_granularity > 0, "prefix_granularity must be positive");
    _assert(prefix_hash_lookahead_tokens == 0 || prefix_hash_lookahead_tokens == 1,
            "prefix_hash_lookahead_tokens must be 0 or 1");
    std::vector<std::span<const std::int32_t>> result;

    if (tokens_.empty()) {
        return result;
    }

    // tokens_ contains only prompt and committed output, never unaccepted drafts.
    // A sampled last token can identify the preceding page without having its own KV.
    const std::int32_t token_size =
        Size() - std::max(static_cast<std::int32_t>(except_last), prefix_hash_lookahead_tokens);
    const std::size_t num_full_pages = token_size / prefix_granularity;
    result.reserve(num_full_pages);
    for (std::size_t i = 0; i < num_full_pages; ++i) {
        std::size_t start = i * prefix_granularity;
        result.emplace_back(tokens_.data() + start,
                            static_cast<std::size_t>(prefix_granularity) + prefix_hash_lookahead_tokens);
    }

    return result;
}

std::span<const std::int32_t> TokenContainer::TokenSlice(Window window) const {
    return {tokens_.data() + window.begin, static_cast<std::size_t>(window.size)};
}

}  // namespace tokenspeed
