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

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace tokenspeed {

// A fitted rank-local prefill cost, in nanoseconds. Coefficients include
// the layers executed on this rank; they are specific to the model,
// attention backend, parallel mapping and hardware, not FLOP counts.
struct PrefillComputeConfig {
    double budget_ns;
    double token_ns;
    double pair_ns;
    double request_ns;

    void Validate() const {
        if (!std::isfinite(budget_ns) || budget_ns <= 0 || !std::isfinite(token_ns) || token_ns < 0 ||
            !std::isfinite(pair_ns) || pair_ns < 0 || !std::isfinite(request_ns) || request_ns < 0 ||
            (token_ns == 0 && pair_ns == 0)) {
            throw std::invalid_argument(
                "prefill compute budget must be finite and positive; coefficients must be finite and "
                "non-negative, with at least one positive token or pair coefficient");
        }
    }

    double Cost(std::int32_t query_tokens, std::int32_t prefix_tokens) const {
        if (query_tokens == 0) return 0;
        const double q = query_tokens;
        return request_ns + token_ns * q + pair_ns * q * (prefix_tokens + (q + 1) / 2);
    }

    // Largest input length fitting the remaining time AND token budgets.
    // Binary search keeps the boundary conservative without cancellation in
    // the quadratic root, and is bounded by 31 iterations for int32 tokens.
    std::int32_t TokenLimit(std::int32_t prefix_tokens, std::int32_t token_limit, double spent_ns) const {
        const double remaining = budget_ns - spent_ns;
        std::int32_t lo = 0;
        std::int32_t hi = token_limit;
        while (lo < hi) {
            const std::int32_t mid = lo + (static_cast<std::int64_t>(hi) - lo + 1) / 2;
            if (Cost(mid, prefix_tokens) <= remaining) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }
};

}  // namespace tokenspeed
