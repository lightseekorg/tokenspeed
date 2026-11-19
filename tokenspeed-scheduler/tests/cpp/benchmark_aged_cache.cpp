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

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "benchmark_report.h"

namespace tokenspeed::benchmark {
namespace {

// DeepSeek V4.1 Flash's cache groups and their children per LCM parent
// (deepseek_v41_geometry.py): two cached global groups beside two replayable
// windows that never cache, so a prefill chunk takes empty parents.
struct Packing {
    std::int32_t swa;
    std::int32_t global_r2;
    std::int32_t global_r1;
    std::int32_t compressor_tail;
};
constexpr Packing kPlainPacking{.swa = 1, .global_r2 = 20, .global_r1 = 60, .compressor_tail = 54};
constexpr Packing kDSparkPacking{.swa = 2, .global_r2 = 45, .global_r1 = 140, .compressor_tail = 126};
constexpr std::uint32_t kGlobalR2 = 1;
constexpr std::uint32_t kGlobalR1 = 2;

struct AgedCacheConfig {
    std::int32_t parents;
    std::int32_t prefix_granularity;
    std::int32_t history_tokens;
    std::int32_t age_rounds;
    std::int32_t evict_percent;
    std::int32_t chunk_tokens;
    Packing packing;

    void Validate() const {
        Require(parents > 0 && history_tokens > 0 && chunk_tokens > 0, "sizes must be positive");
        Require(prefix_granularity > 0 && prefix_granularity % 128 == 0, "pages must be whole 128-token blocks");
        Require(history_tokens % prefix_granularity == 0, "histories must be page aligned");
        Require(evict_percent >= 0 && evict_percent <= 100, "evict percent must be in [0, 100]");
    }
};

SchedulerConfig MakeSchedulerConfig(const AgedCacheConfig& workload) {
    workload.Validate();
    SchedulerConfig config{};
    config.prefix_granularity = workload.prefix_granularity;
    config.device_allocator.total_pages = workload.parents + 1;
    config.host_allocator.total_pages = 1;
    config.snapshot_allocator.total_pages = 1;
    config.max_retracted_requests = 0;
    config.max_scheduled_tokens = workload.chunk_tokens;
    config.max_batch_size = 1;
    config.max_resident_requests = 1;
    config.decode_input_tokens = 1;
    config.overlap_schedule_depth = 0;
    config.disable_l2_cache = true;
    config.enable_l3_storage = false;
    config.enable_kv_cache_events = false;
    config.enable_short_prefill_first = false;
    config.role = Role::kFused;
    config.disable_prefix_cache = false;
    config.prefix_replay_tokens = 128;
    config.checkpoint_retention_requests = 0;
    const auto group = [&](const char* id, std::int32_t granularity, std::int32_t packing,
                           std::optional<std::int32_t> window) {
        config.cache_groups.push_back(CacheGroupConfig{
            .group_id = id,
            .block_granularity = granularity,
            .total_pages = workload.parents * packing + 1,
            .cache_blocks_per_lcm_block = packing,
            .attn_kind = window ? AttnKind::kSlidingWindow : AttnKind::kFull,
            .sliding_window_tokens = window,
            .replayable = window.has_value(),
        });
    };
    group("swa", 64, workload.packing.swa, 128);
    group("global_r2", 128, workload.packing.global_r2, std::nullopt);
    group("global_r1", 64, workload.packing.global_r1, std::nullopt);
    group("compressor_tail", 2, workload.packing.compressor_tail, 2);
    return config;
}

struct AgedSample {
    Duration admit;
    std::uint64_t evicted_blocks{0};
};

// A full cache whose parents hold entries of many access epochs, the state a
// long-running server reaches: histories fill it, then rounds of scattered
// evictions leave holes that later histories take hole-first.
class AgedCacheWorkload {
public:
    explicit AgedCacheWorkload(const AgedCacheConfig& config, std::mt19937& random)
        : config_{config}, scheduler_{MakeSchedulerConfig(config)} {
        FillUp();
        for (std::int32_t round = 0; round < config_.age_rounds; ++round) {
            EvictScattered(random);
            FillUp();
        }
    }

    std::int64_t CachedBlocks() {
        return Coordinator().GroupPrefixIndex(kGlobalR1).NumEntries(Pool()) +
               Coordinator().GroupPrefixIndex(kGlobalR2).NumEntries(Pool());
    }

    // One prefill's first chunk of a fresh prompt, every group growing by the
    // chunk. Its blocks are released unpublished and the cache refilled
    // outside timing.
    AgedSample RunSample() {
        const std::vector<std::string> hashes = FreshHashes(config_.chunk_tokens + config_.prefix_granularity);
        const std::span<const std::string> probed = std::span<const std::string>{hashes}.first(hashes.size() - 1);
        std::vector<BlockTable> tables(static_cast<std::size_t>(Coordinator().NumGroups()));
        const std::vector<GroupDemand> demands = Demands(tables, config_.chunk_tokens, /*cached_only=*/false);
        AgedSample sample;
        const auto evictions = Coordinator().DeviceEvictions().admission.blocks;
        std::optional<AdmissionResult> result;
        sample.admit = Measure([&] {
            result = Coordinator().Admit(Coordinator().Prober().ProbePrefix(probed), demands, RequestProgress{}, {},
                                         std::nullopt, std::nullopt, /*session_id=*/{}, /*cache_group_id=*/{});
        });
        Require(result.has_value(), "the chunk's admission failed");
        sample.evicted_blocks = Coordinator().DeviceEvictions().admission.blocks - evictions;
        Require(sample.evicted_blocks > 0, "the chunk must evict from a full cache");
        tables.clear();
        FillUp();
        return sample;
    }

private:
    CacheCoordinator& Coordinator() { return SchedulerTestAccess::Coordinator(scheduler_); }
    const BlockPool& Pool() { return SchedulerTestAccess::Pool(scheduler_); }

    // Every group grows by `tokens`, or with `cached_only` the cached ones.
    static std::vector<GroupDemand> Demands(std::vector<BlockTable>& tables, std::int32_t tokens, bool cached_only) {
        std::vector<GroupDemand> demands;
        for (std::uint32_t g = 0; g < tables.size(); ++g) {
            const bool grows = !cached_only || g == kGlobalR1 || g == kGlobalR2;
            demands.push_back(
                GroupDemand{.table = &tables[g], .extent = DenseGrowth{grows ? tokens : 0}, .reserve_tokens = 0});
        }
        return demands;
    }

    // Page hashes of a prompt no earlier one shares a page with.
    std::vector<std::string> FreshHashes(std::int32_t tokens) {
        std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens));
        std::iota(ids.begin(), ids.end(), 1);
        ids.front() = -(++prompts_);
        const TokenContainer container(ids);
        return ComputePrefixHashes(container.FullPrefixPages(config_.prefix_granularity, false), "");
    }

    // Publishes fresh histories into the cached groups until one no longer
    // fits without eviction.
    void FillUp() {
        while (true) {
            const auto parents_short = [&](std::uint32_t g, std::int64_t blocks) {
                const std::int64_t short_by = std::max<std::int64_t>(blocks - Pool().NumFreeSlots(g), 0);
                return (short_by + Pool().Packing(g) - 1) / Pool().Packing(g);
            };
            if (parents_short(kGlobalR1, config_.history_tokens / 64) +
                    parents_short(kGlobalR2, config_.history_tokens / 128) >
                Pool().NumEmptyLcmBlocks()) {
                return;
            }
            const std::vector<std::string> hashes = FreshHashes(config_.history_tokens);
            std::vector<BlockTable> tables(static_cast<std::size_t>(Coordinator().NumGroups()));
            const std::vector<GroupDemand> demands = Demands(tables, config_.history_tokens, /*cached_only=*/true);
            const auto evictions = Coordinator().DeviceEvictions().admission.blocks;
            const auto result = Coordinator().Admit(Coordinator().Prober().ProbePrefix(hashes), demands,
                                                    RequestProgress{}, {}, std::nullopt, std::nullopt,
                                                    /*session_id=*/{}, /*cache_group_id=*/{});
            Require(result.has_value() && Coordinator().DeviceEvictions().admission.blocks == evictions,
                    "a fill admits without eviction");
            CacheCompletedBlocksForTest(Coordinator(), tables, hashes, result->access_epoch, 0, config_.history_tokens,
                                        CacheBoundaryKind::kEndpoint,
                                        /*stream_completed_to_host=*/false, /*publish_state=*/false);
        }
    }

    // Evicts a random share of the cached blocks: scattered holes.
    void EvictScattered(std::mt19937& random) {
        std::vector<std::pair<std::uint32_t, CacheBlockLocation>> victims;
        std::uniform_int_distribution<std::int32_t> percent(0, 99);
        for (const std::uint32_t g : {kGlobalR2, kGlobalR1}) {
            Coordinator().GroupPrefixIndex(g).ForEachEntry(Pool(), [&](const CacheKey&, const auto& entry) {
                if (!entry.pinned && percent(random) < config_.evict_percent) victims.emplace_back(g, entry.location);
            });
        }
        for (const auto& [g, location] : victims) {
            CacheCoordinatorTestAccess::EvictCachedBlock(Coordinator(), g, location);
        }
    }

    AgedCacheConfig config_;
    Scheduler scheduler_;
    std::int32_t prompts_{0};
};

void Run(const AgedCacheConfig& config, std::string_view packing, int warmups, int repeats) {
    Require(repeats > 0, "repeats must be positive");
    std::cerr << "Filling and aging " << config.parents << " parents...\n";
    std::mt19937 random(20261118);
    const auto fill_start = std::chrono::steady_clock::now();
    AgedCacheWorkload workload(config, random);
    const double fill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - fill_start).count();
    const std::int64_t cached_blocks = workload.CachedBlocks();
    std::cerr << "Ready in " << fill_seconds << " s with " << cached_blocks << " cached blocks; measuring " << repeats
              << " admissions.\n";
    std::vector<AgedSample> samples;
    for (int i = 0; i < warmups + repeats; ++i) {
        const AgedSample sample = workload.RunSample();
        if (i >= warmups) samples.push_back(sample);
    }
    std::cout << std::setprecision(9) << "{\"benchmark\":\"aged_cache\",\"parents\":" << config.parents
              << ",\"packing\":\"" << packing << "\",\"prefix_granularity\":" << config.prefix_granularity
              << ",\"history_tokens\":" << config.history_tokens << ",\"age_rounds\":" << config.age_rounds
              << ",\"evict_percent\":" << config.evict_percent << ",\"chunk_tokens\":" << config.chunk_tokens
              << ",\"cached_blocks\":" << cached_blocks
              << ",\"evicted_blocks_per_admission\":" << samples.front().evicted_blocks << ",\"warmups\":" << warmups
              << ",\"repeats\":" << repeats << ",\"fill_seconds\":" << fill_seconds;
    PrintTiming("admit_including_eviction", samples, &AgedSample::admit);
    std::cout << "}\n" << std::flush;
}

}  // namespace
}  // namespace tokenspeed::benchmark

int main(int argc, char** argv) {
    try {
        namespace bench = tokenspeed::benchmark;
        bench::Require(argc == 10,
                       "usage: benchmark PARENTS plain|dspark PREFIX_GRANULARITY HISTORY_TOKENS AGE_ROUNDS "
                       "EVICT_PERCENT CHUNK_TOKENS WARMUPS REPEATS");
        const std::string_view packing = argv[2];
        bench::Require(packing == "plain" || packing == "dspark", "packing must be plain or dspark");
        bench::Run(
            bench::AgedCacheConfig{.parents = bench::ParseNonnegative(argv[1]),
                                   .prefix_granularity = bench::ParseNonnegative(argv[3]),
                                   .history_tokens = bench::ParseNonnegative(argv[4]),
                                   .age_rounds = bench::ParseNonnegative(argv[5]),
                                   .evict_percent = bench::ParseNonnegative(argv[6]),
                                   .chunk_tokens = bench::ParseNonnegative(argv[7]),
                                   .packing = packing == "dspark" ? bench::kDSparkPacking : bench::kPlainPacking},
            packing, bench::ParseNonnegative(argv[8]), bench::ParseNonnegative(argv[9]));
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "aged_cache benchmark: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
