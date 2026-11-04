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
#include <deque>
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

    void SetBaseline(int batch, int queries, double milliseconds) {
        ValidateTiming(batch, queries, milliseconds);
        baselines_[{batch, queries}] = milliseconds;
    }

    void Observe(int batch, int queries, double milliseconds) {
        ValidateTiming(batch, queries, milliseconds);
        auto& samples = timings_[{batch, queries}];
        samples.push_back(milliseconds);
        if (samples.size() > 9) samples.erase(samples.begin());
        auto baseline = baselines_.find({batch, queries});
        if (baseline != baselines_.end()) {
            corrections_.push_back({queries, (milliseconds - baseline->second) / queries});
            if (corrections_.size() > 5000) corrections_.pop_front();
        }
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
        std::vector<double> expected(scores.size() + 1, static_cast<double>(rates.size()));
        for (size_t i = 0; i < scores.size(); ++i) expected[i + 1] = expected[i] + scores[i];
        const double correction = CorrectionPerQuery();
        int best_queries = batch;
        double best_throughput = -1;
        // Target graphs are keyed by Q alone. Every captured Q in range is
        // usable, including those captured for a different request capacity.
        for (int queries = batch; queries <= batch * width_; ++queries) {
            const auto key = std::make_pair(batch, queries);
            const auto baseline = baselines_.find(key);
            const auto online = timings_.find(key);
            if (baseline == baselines_.end() && online == timings_.end()) continue;
            double latency = 0;
            if (baseline != baselines_.end()) {
                // Empty-cache capture underestimates live attention. Apply
                // completed measurements to the entire curve, so unvisited
                // large graphs cannot retain their optimistic startup cost.
                latency = baseline->second + correction * queries;
            }
            if (online != timings_.end()) latency = std::max(latency, Median(online->second));
            const size_t extras = std::min(scores.size(), static_cast<size_t>(queries - rates.size()));
            const double throughput = expected[extras] / latency;
            if (throughput > best_throughput) {
                best_throughput = throughput;
                best_queries = queries;
            }
        }
        return best_queries;
    }

private:
    void ValidateTiming(int batch, int queries, double milliseconds) const {
        if (batch < 1 || queries < batch || queries > batch * width_ || !std::isfinite(milliseconds) ||
            milliseconds <= 0) {
            throw std::invalid_argument("invalid confidence timing sample");
        }
    }

    static double Median(std::vector<double> samples) {
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }

    double CorrectionPerQuery() const {
        std::map<int, std::vector<double>> by_queries;
        for (const auto& [queries, correction] : corrections_) by_queries[queries].push_back(correction);
        std::vector<double> medians;
        for (auto& [queries, samples] : by_queries) {
            if (samples.size() >= 10) medians.push_back(Median(samples));
        }
        return medians.empty() ? 0.0 : std::max(0.0, Median(medians));
    }

    int width_;
    std::map<std::pair<int, int>, double> baselines_;
    std::map<std::pair<int, int>, std::vector<double>> timings_;
    std::deque<std::pair<int, double>> corrections_;
};

}  // namespace tokenspeed
