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

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "cache/coordinator/admission_order.h"

namespace tokenspeed {
namespace {

void CheckOrder(std::vector<AdmissionVictimCandidate> candidates) {
    auto expected = candidates;
    const auto key = [](const auto& c) {
        return std::tuple{c.last_access_epoch, c.eviction_tier,         c.position_rank,
                          c.group_id,          c.location.lcm_block_id, c.location.slot_index};
    };
    std::ranges::sort(expected, {}, key);
    SortAdmissionGroupEpoch(candidates);
    ASSERT_EQ(candidates.size(), expected.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        ASSERT_EQ(key(candidates[i]), key(expected[i])) << "index " << i;
    }
}

TEST(AdmissionOrder, ClosedPrefixBoundariesAndStableLocationTies) {
    for (const int size : {0, 1, 127, 128, 129, 255, 256, 257, 1024}) {
        for (const std::int64_t extent : {0LL, 1LL, 255LL, 256LL, 65535LL, 65536LL, 2147483647LL}) {
            std::vector<AdmissionVictimCandidate> candidates;
            for (int i = 0; i < size; ++i) {
                candidates.push_back({.group_id = 2,
                                      .location = {.lcm_block_id = i / 4, .slot_index = i % 4},
                                      .last_access_epoch = 42,
                                      .eviction_tier = AdmissionEvictionTier::kClosedPrefix,
                                      .position_rank = i % 3 == 0 ? 0 : -extent});
            }
            CheckOrder(candidates);
        }
    }
}

TEST(AdmissionOrder, RandomizedGroupEpochMatchesFullTupleOrder) {
    std::mt19937_64 random(20261101);
    for (int trial = 0; trial < 60000; ++trial) {
        std::vector<AdmissionVictimCandidate> candidates;
        const int size = random() % 513;
        const bool mixed_tiers = trial % 4 == 0;
        for (int i = 0; i < size; ++i) {
            const auto tier =
                mixed_tiers ? static_cast<AdmissionEvictionTier>(random() % 4) : AdmissionEvictionTier::kClosedPrefix;
            const std::int64_t position = trial % 2 ? random() % 17 : random() % 2147483648ULL;
            candidates.push_back(
                {.group_id = 7,
                 .location = {.lcm_block_id = i / 8, .slot_index = i % 8},
                 .last_access_epoch = 77,
                 .eviction_tier = tier,
                 .position_rank = tier == AdmissionEvictionTier::kClosedPrefix ? -position : position});
        }
        CheckOrder(candidates);
    }
}

TEST(AdmissionOrder, SmallAndMixedTierFallbackKeepsAllTupleFields) {
    std::mt19937 random(9);
    for (int size : {127, 128, 1024}) {
        std::vector<AdmissionVictimCandidate> candidates;
        for (int i = 0; i < size; ++i) {
            candidates.push_back({.group_id = static_cast<std::uint32_t>(random() % 4),
                                  .location = {.lcm_block_id = i / 4, .slot_index = i % 4},
                                  .last_access_epoch = random() % 8,
                                  .eviction_tier = AdmissionEvictionTier::kProbationaryBoundary,
                                  .position_rank = static_cast<std::int64_t>(random() % 100)});
        }
        std::shuffle(candidates.begin(), candidates.end(), random);
        CheckOrder(candidates);
    }
}

}  // namespace
}  // namespace tokenspeed
