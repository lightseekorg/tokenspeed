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
#include <exception>
#include <limits>
#include <stdexcept>

namespace tokenspeed {

void TokenContainer::Extend(const std::vector<std::int32_t>& new_tokens) {
    tokens_.insert(tokens_.end(), new_tokens.begin(), new_tokens.end());
}

void TokenContainer::ReserveAdditional(std::size_t additional_tokens) {
    const std::size_t int32_max = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (additional_tokens > tokens_.max_size() - tokens_.size() || additional_tokens > int32_max - tokens_.size()) {
        throw std::length_error("TokenContainer prepared extension exceeds its size domain");
    }
    tokens_.reserve(tokens_.size() + additional_tokens);
}

void TokenContainer::ExtendPrepared(std::span<const std::int32_t> new_tokens) noexcept {
    if (new_tokens.size() > tokens_.capacity() - tokens_.size() ||
        new_tokens.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) - tokens_.size()) {
        std::terminate();
    }
    for (const std::int32_t token : new_tokens) {
        // Capacity was fixed by ReserveAdditional. int32 copy construction is
        // no-throw, so this cannot enter vector growth.
        tokens_.push_back(token);
    }
}

std::vector<std::span<const std::int32_t>> TokenContainer::GetFullPagedTokens(std::int32_t page_size,
                                                                              bool except_last) const {
    std::vector<std::span<const std::int32_t>> result;

    if (tokens_.empty()) {
        return result;
    }

    std::int32_t token_size = except_last ? tokens_.size() - 1 : tokens_.size();
    std::size_t num_full_pages = token_size / page_size;
    result.reserve(num_full_pages);
    for (std::size_t i = 0; i < num_full_pages; ++i) {
        std::size_t start = i * page_size;
        result.emplace_back(tokens_.data() + start, page_size);
    }

    return result;
}

std::span<const std::int32_t> TokenContainer::GetTokenSlice(Window window) const {
    return {tokens_.data() + window.begin, static_cast<std::size_t>(window.size)};
}

}  // namespace tokenspeed
