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
#include <span>
#include <vector>

namespace tokenspeed {

// Design principle:
// only one tokens_ member variable, no more vectors storing tokens, use offsets to determine its meaning
class TokenContainer {
public:
    struct Window {
        std::int32_t begin;
        std::int32_t size;
    };

    TokenContainer(const std::vector<std::int32_t>& new_tokens)
        : tokens_{new_tokens}, num_prefill_tokens_(new_tokens.size()){};

    TokenContainer(const TokenContainer&) = delete;
    TokenContainer& operator=(const TokenContainer&) = delete;

    void Extend(const std::vector<std::int32_t>& new_tokens);

    // Flat retract: fold generated tokens into the prefill window so the
    // requeued request prefills prompt + generated as one fresh extend.
    void RebasePrefill() { num_prefill_tokens_ = static_cast<std::int32_t>(tokens_.size()); }

    // constant methods
    std::vector<std::span<const std::int32_t>> GetFullPagedTokens(std::int32_t page_size, bool except_last) const;
    std::int32_t Size() const { return tokens_.size(); }
    std::int32_t PrefillSize() const { return num_prefill_tokens_; }
    std::span<const std::int32_t> GetTokenSlice(Window window) const;
    std::int32_t LastToken() const { return tokens_.back(); }

private:
    std::vector<std::int32_t> tokens_;
    std::int32_t num_prefill_tokens_{};
};

}  // namespace tokenspeed
