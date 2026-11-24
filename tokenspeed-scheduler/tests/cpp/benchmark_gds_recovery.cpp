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
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "cache/coordinator/gds_planner.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache_test_access.h"

using namespace tokenspeed;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: gds_benchmark checkpoint_count iterations demanded_state_blocks\n";
        return 2;
    }
    const auto points = std::stoi(argv[1]);
    const auto iterations = std::stoi(argv[2]);
    const auto demand_count = std::stoi(argv[3]);
    if (points <= 0 || points > 100000 || iterations <= 0 || demand_count <= 0 || demand_count > points) return 2;
    constexpr std::int32_t page = 4;
    const auto micros = [](auto begin) {
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
    };
    BlockPool device{points * 4 + 32, {1, 1}};
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = page},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
    auto cache = MakeCoordinator(specs, page, device, false, nullptr, nullptr, nullptr, false, false, 0);
    cache.ConfigureEviction(PrefixCacheEvictionConfig::Gds({0.25, 0}));
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(points) * page);
    std::iota(tokens.begin(), tokens.end(), 1);
    std::vector<std::string> hashes;
    ContentHash prior;
    for (int p = 0; p < points; ++p) {
        prior = HashPrefixPage(std::span{tokens}.subspan(static_cast<std::size_t>(p) * page, page), prior);
        hashes.push_back(prior.ToString());
    }
    const auto setup_begin = std::chrono::steady_clock::now();
    for (int p = 1; p <= points; ++p) {
        std::vector<BlockTable> tables(2);
        auto admission = AdmitForTest(cache, tables, p * page);
        FatalCheck(admission.has_value(), "benchmark publication fits");
        cache.CacheCompletedBlocks(
            tables,
            RequestProgress{
                .completed_pages = CompletedPages{.tokens = std::span{tokens}.first(static_cast<std::size_t>(p) * page),
                                                  .prefix_hashes = std::span{hashes}.first(p),
                                                  .publish_state = true,
                                                  .boundary_kind = CacheBoundaryKind::kEndpoint},
                .num_computed_tokens = p * page},
            admission->access_epoch);
        cache.Free(tables);
    }
    const double publication_us = micros(setup_begin);
    std::vector<CacheBlockRef> pins;
    while (device.NumEmptyLcmBlocks() > 0) pins.push_back(device.AcquireBlock(0));
    const auto& store = CacheCoordinatorTestAccess::Store(cache);
#ifdef TOKENSPEED_PERSISTENT_RECOVERY_BENCHMARK
    const auto before = store.RecoveryDependencies().GetStats();
#endif
    std::uint64_t checksum = 0;
    std::vector<double> timings;
    for (int repeat = 0; repeat < iterations + 2; ++repeat) {
        const auto begin = std::chrono::steady_clock::now();
        std::pmr::monotonic_buffer_resource arena;
        AdmissionCandidateConstraints constraints{&arena};
        CacheEvictionPlan capacity{device, store.Groups()};
        std::array<BlockPlacementDemand, 2> demands{};
        for (auto& demand : demands) demand.blocks_by_bucket = {0};
        demands[1].any_bucket_blocks = demand_count;
        auto plan = PlanGdsEviction(store, constraints, demands, capacity, &arena);
        FatalCheck(plan.has_value(), "benchmark pressure plan fits");
        for (const auto& victim : *plan)
            for (const auto& member : victim.members)
                checksum += static_cast<std::uint64_t>(member.location.lcm_block_id);
        if (repeat >= 2) timings.push_back(micros(begin));
    }
    std::ranges::sort(timings);
    std::cout
        << "points,demand,iterations,publication_us,plan_p50_us,plan_p95_us,ranges,pressure_route_updates,checksum\n";
    std::cout << points << ',' << demand_count << ',' << iterations << ',' << publication_us << ','
              << timings[timings.size() / 2] << ',' << timings[(timings.size() - 1) * 95 / 100] << ',';
#ifdef TOKENSPEED_PERSISTENT_RECOVERY_BENCHMARK
    const auto after = store.RecoveryDependencies().GetStats();
    std::cout << after.ranges << ',' << after.route_updates - before.route_updates;
#else
    std::cout << "NA,NA";
#endif
    std::cout << ',' << checksum << '\n';
}
