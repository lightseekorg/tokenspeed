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

#include <cstddef>
#include <cstdint>

#include "utils.h"

namespace tokenspeed {

void TokenContainer::Extend(const std::vector<std::int32_t>& new_tokens) {
    tokens_.insert(tokens_.end(), new_tokens.begin(), new_tokens.end());
}

std::vector<std::span<const std::int32_t>> TokenContainer::FullPrefixPages(std::int32_t prefix_granularity,
                                                                           bool except_last) const {
    if (tokens_.empty()) {
        return {};
    }
    const std::int32_t token_size = except_last ? Size() - 1 : Size();
    return PrefixPages(prefix_granularity, 0, token_size / prefix_granularity);
}

std::vector<std::span<const std::int32_t>> TokenContainer::PrefixPages(std::int32_t prefix_granularity,
                                                                       std::int32_t first_page,
                                                                       std::int32_t past_end_page) const {
    _assert(0 <= first_page && first_page <= past_end_page, "prefix page range is inverted");
    _assert(static_cast<std::int64_t>(past_end_page) * prefix_granularity <= Size(), "prefix pages must be complete");
    std::vector<std::span<const std::int32_t>> result;
    result.reserve(static_cast<std::size_t>(past_end_page - first_page));
    for (std::int32_t page = first_page; page < past_end_page; ++page) {
        result.emplace_back(tokens_.data() + static_cast<std::size_t>(page) * prefix_granularity,
                            static_cast<std::size_t>(prefix_granularity));
    }
    return result;
}

std::span<const std::int32_t> TokenContainer::TokenSlice(Window window) const {
    return {tokens_.data() + window.begin, static_cast<std::size_t>(window.size)};
}

}  // namespace tokenspeed
