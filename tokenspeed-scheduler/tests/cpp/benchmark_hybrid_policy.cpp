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

#include "cache/coordinator/admission_planner.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache_test_access.h"

using namespace tokenspeed;

// Compare read-only admission plans on an unchanged populated checkpoint chain.
// "blocked" has no reclaimable Host capacity; it measures migration/loss trials,
// not a populated large Host cache or a complete scheduler round.
int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: hybrid_policy_benchmark lru|gds none|blocked|free points iterations kv_blocks\n";
        return 2;
    }
    const std::string policy = argv[1], host_mode = argv[2];
    const int points = std::stoi(argv[3]), iterations = std::stoi(argv[4]), kv_blocks = std::stoi(argv[5]);
    if ((policy != "lru" && policy != "gds") ||
        (host_mode != "none" && host_mode != "blocked" && host_mode != "free") || points < 2 || points > 100000 ||
        iterations <= 0 || iterations > 100000 || kv_blocks < 1 || kv_blocks > 100000)
        return 2;
    const bool write_back = host_mode != "none";
    constexpr std::int32_t page = 4;
    BlockPool device{points * 4 + 32, {1, 1}}, host{16, {1, 1}};
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = page},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
    auto cache = MakeCoordinator(specs, page, device, false, write_back ? &host : nullptr, nullptr, nullptr, false,
                                 write_back, 0);
    if (policy == "gds") cache.ConfigureEviction(PrefixCacheEvictionConfig::Gds({0.25, 0}));
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(points) * page);
    std::iota(tokens.begin(), tokens.end(), 1);
    std::vector<std::string> hashes;
    ContentHash prior;
    for (int i = 0; i < points; ++i) {
        prior = HashPrefixPage(std::span{tokens}.subspan(static_cast<std::size_t>(i) * page, page), prior);
        hashes.push_back(prior.ToString());
    }
    for (int i = 1; i <= points; ++i) {
        std::vector<BlockTable> tables(2);
        auto admission = AdmitForTest(cache, tables, i * page);
        FatalCheck(admission.has_value(), "benchmark publication fits");
        cache.CacheCompletedBlocks(
            tables,
            RequestProgress{
                .completed_pages = CompletedPages{.tokens = std::span{tokens}.first(static_cast<std::size_t>(i) * page),
                                                  .prefix_hashes = std::span{hashes}.first(i),
                                                  .publish_state = true,
                                                  .boundary_kind = CacheBoundaryKind::kEndpoint},
                .num_computed_tokens = i * page},
            admission->access_epoch);
        cache.Free(tables);
    }
    std::vector<CacheBlockRef> device_pins, host_pins;
    while (device.NumEmptyLcmBlocks()) device_pins.push_back(device.AcquireBlock(0));
    if (host_mode == "blocked")
        while (host.NumEmptyLcmBlocks()) host_pins.push_back(host.AcquireBlock(0));
    const auto& store = CacheCoordinatorTestAccess::Store(cache);
    std::array<BlockTable, 2> tables;
    const std::array<GroupDemand, 2> demand{
        GroupDemand{.table = &tables[0], .extent = DenseGrowth{page * kv_blocks}},
        GroupDemand{.table = &tables[1], .extent = StateStep{StateStep::Kind::kRotate}}};
    std::vector<double> timings;
    std::uint64_t victims = 0;
    for (int repeat = 0; repeat < iterations + 2; ++repeat) {
        const auto begin = std::chrono::steady_clock::now();
        {
            auto plan = planAdmission(store, CheckpointAging{}, cache.Prober().ProbePrefix({}), demand, std::nullopt,
                                      std::nullopt, {}, {}, write_back);
            FatalCheck(plan.has_value(), "benchmark pressure plan fits");
            for (const auto& set : plan->victims) victims += set.members.size();
        }
        if (repeat >= 2)
            timings.push_back(
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count());
    }
    std::ranges::sort(timings);
    const auto quantile = [&](std::size_t percent) { return timings[(timings.size() - 1) * percent / 100]; };
    std::cout
        << "policy,host,points,iterations,kv_blocks,p50_us,p95_us,p99_us,max_us,dependency_ranges,victim_checksum\n"
        << policy << ',' << host_mode << ',' << points << ',' << iterations << ',' << kv_blocks << ',' << quantile(50)
        << ',' << quantile(95) << ',' << quantile(99) << ',' << timings.back() << ','
        << store.RecoveryDependencies().GetStats().ranges << ',' << victims << '\n';
}
