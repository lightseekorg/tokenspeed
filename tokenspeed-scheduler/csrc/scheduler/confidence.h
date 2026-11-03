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

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tokenspeed {

// Host-only budget selection. Callers supply completed, older-round scores;
// this object never holds device state or waits for a forward in flight.
class ConfidenceBudgetPlanner {
public:
    explicit ConfidenceBudgetPlanner(int width) : width_(width) {
        if (width < 2) throw std::invalid_argument("confidence requires a draft window");
    }

    void Observe(int batch, int queries, double milliseconds) {
        if (batch < 1 || queries < batch || queries > batch * width_ || !std::isfinite(milliseconds) ||
            milliseconds <= 0) {
            throw std::invalid_argument("invalid confidence timing sample");
        }
        auto& samples = timings_[{batch, queries}];
        samples.push_back(milliseconds);
        if (samples.size() > 9) samples.erase(samples.begin());
    }

    int Select(int batch, const std::vector<std::vector<float>>& rates) const {
        if (batch < 1 || rates.size() > static_cast<size_t>(batch)) {
            throw std::invalid_argument("invalid confidence batch");
        }
        std::vector<float> scores;
        for (const auto& row : rates) {
            if (row.size() != static_cast<size_t>(width_ - 1)) {
                throw std::invalid_argument("confidence width mismatch");
            }
            float prefix = 1;
            for (float rate : row) {
                prefix *= std::isfinite(rate) ? std::clamp(rate, 0.0f, 1.0f) : 0.0f;
                scores.push_back(prefix);
            }
        }
        std::stable_sort(scores.begin(), scores.end(), std::greater<float>());
        int best_queries = batch * width_;
        double best_throughput = -1;
        for (int multiplier = 1; multiplier <= width_; ++multiplier) {
            const int queries = batch * multiplier;
            auto found = timings_.find({batch, queries});
            if (found == timings_.end()) continue;
            auto sorted = found->second;
            std::sort(sorted.begin(), sorted.end());
            const double latency = sorted[sorted.size() / 2];
            // The graph includes padding requests; only live requests earn
            // base output tokens. Remaining graph capacity can go to any live
            // request, bounded by its number of draft positions.
            double expected = static_cast<double>(rates.size());
            const size_t extras = std::min(scores.size(), static_cast<size_t>(queries - rates.size()));
            for (size_t i = 0; i < extras; ++i) expected += scores[i];
            const double throughput = expected / latency;
            if (throughput > best_throughput) {
                best_throughput = throughput;
                best_queries = queries;
            }
        }
        return best_queries;
    }

private:
    int width_;
    std::map<std::pair<int, int>, std::vector<double>> timings_;
};

}  // namespace tokenspeed
