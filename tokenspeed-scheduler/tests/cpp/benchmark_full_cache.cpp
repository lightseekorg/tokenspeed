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

#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

#include "full_cache_benchmark.h"

namespace tokenspeed::benchmark {
namespace {

int ParseNonnegative(std::string_view value) {
    int result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    Require(error == std::errc{} && end == value.data() + value.size() && result >= 0, "expected nonnegative integer");
    return result;
}

void PrintDistribution(std::vector<double> values) {
    std::cout << "{\"samples\":[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << values[i];
    }
    std::ranges::sort(values);
    const auto n = values.size();
    std::cout << "],\"median\":" << (values[(n - 1) / 2] + values[n / 2]) / 2
              << ",\"p95\":" << values[(n * 95 + 99) / 100 - 1] << ",\"min\":" << values.front()
              << ",\"max\":" << values.back() << '}';
}

template <typename Sample>
void PrintTiming(const char* name, const std::vector<Sample>& samples, Duration Sample::*member) {
    std::vector<double> wall, cpu;
    for (const auto& sample : samples) {
        wall.push_back((sample.*member).wall_ms);
        cpu.push_back((sample.*member).cpu_ms);
    }
    std::cout << ",\"" << name << "\":{\"wall_ms\":";
    PrintDistribution(std::move(wall));
    std::cout << ",\"thread_cpu_ms\":";
    PrintDistribution(std::move(cpu));
    std::cout << '}';
}

void Run(const FullCacheConfig& config, int warmups, int repeats, std::string_view mode, std::string_view layout) {
    Require(repeats > 0, "repeats must be positive");
    Require(mode == "first" || mode == "lifecycle" || mode == "events", "unknown measurement mode");
    Require(layout == "sequential" || layout == "shuffled", "unknown physical layout");
    std::cerr << "Filling " << config.pool_pages << " pages for " << config.hit_percent << "% hits...\n";
    const auto fill_start = std::chrono::steady_clock::now();
    FullCacheWorkload workload(config, mode == "events",
                               layout == "shuffled" ? CacheLayout::kShuffled : CacheLayout::kSequential);
    const bool lifecycle = mode == "lifecycle" || mode == "events";
    const double fill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - fill_start).count();
    std::cerr << "Filled in " << fill_seconds << " s; measuring " << repeats << " batches per measurement.\n";
    if (lifecycle) {
        std::vector<LifecycleSample> samples;
        for (int i = 0; i < warmups + repeats; ++i) {
            const auto sample = workload.RunLifecycleSample();
            if (i >= warmups) samples.push_back(sample);
        }
        std::cout << std::setprecision(9) << "{\"benchmark\":\"" << mode << "\",\"layout\":\"" << layout
                  << "\",\"hit_percent\":" << config.hit_percent << ",\"pool_pages\":" << config.pool_pages
                  << ",\"published_prompt_pages\":" << samples.front().published_prompt_pages
                  << ",\"stored_event_blocks\":" << samples.front().stored_event_blocks;
        PrintTiming("submit", samples, &LifecycleSample::submit);
        PrintTiming("first_plan", samples, &LifecycleSample::first_plan);
        PrintTiming("advance", samples, &LifecycleSample::advance);
        PrintTiming("publish_plan", samples, &LifecycleSample::publish_plan);
        PrintTiming("finish", samples, &LifecycleSample::finish);
        PrintTiming("cleanup_plan", samples, &LifecycleSample::cleanup_plan);
        PrintTiming("events", samples, &LifecycleSample::events);
        std::cout << "}\n" << std::flush;
        return;
    }
    std::vector<CacheSample> cache;
    std::vector<SchedulerSample> scheduler;
    for (int i = 0; i < warmups + repeats; ++i) {
        const auto sample = workload.RunCacheSample();
        if (i >= warmups) cache.push_back(sample);
    }
    for (int i = 0; i < warmups + repeats; ++i) {
        const auto sample = workload.RunSchedulerSample();
        if (i >= warmups) scheduler.push_back(sample);
    }
    std::cout << std::setprecision(9) << "{\"benchmark\":\"full_prefix_cache\",\"layout\":\"" << layout
              << "\",\"pool_pages\":" << config.pool_pages
              << ",\"cached_tokens\":" << std::int64_t{config.pool_pages} * config.page_size
              << ",\"page_size\":" << config.page_size << ",\"request_tokens\":" << config.request_tokens
              << ",\"batch_size\":" << config.batch_size << ",\"target_hit_percent\":" << config.hit_percent
              << ",\"hit_tokens_per_request\":" << config.HitTokens()
              << ",\"actual_hit_percent\":" << 100.0 * config.HitTokens() / config.request_tokens
              << ",\"evicted_pages_per_batch\":" << cache.front().evicted_pages
              << ",\"scheduler_evicted_pages_per_batch\":" << scheduler.front().evicted_pages
              << ",\"warmups\":" << warmups << ",\"repeats\":" << repeats << ",\"fill_seconds\":" << fill_seconds;
    PrintTiming("hash", cache, &CacheSample::hash);
    PrintTiming("prepare_keys", cache, &CacheSample::prepare_keys);
    PrintTiming("prefix_probe", cache, &CacheSample::probe);
    PrintTiming("admit_including_eviction", cache, &CacheSample::admit);
    PrintTiming("submit_requests", scheduler, &SchedulerSample::submit);
    PrintTiming("next_execution_plan", scheduler, &SchedulerSample::plan);
    std::cout << "}\n" << std::flush;
}

}  // namespace
}  // namespace tokenspeed::benchmark

int main(int argc, char** argv) {
    try {
        using namespace tokenspeed::benchmark;
        Require(argc >= 8 && argc <= 10,
                "usage: benchmark POOL_PAGES PAGE_SIZE REQUEST_TOKENS BATCH_SIZE HIT_PERCENT WARMUPS REPEATS "
                "[first|lifecycle|events] [sequential|shuffled]");
        Run(FullCacheConfig{ParseNonnegative(argv[1]), ParseNonnegative(argv[2]), ParseNonnegative(argv[3]),
                            ParseNonnegative(argv[4]), ParseNonnegative(argv[5])},
            ParseNonnegative(argv[6]), ParseNonnegative(argv[7]), argc >= 9 ? argv[8] : "first",
            argc == 10 ? argv[9] : "sequential");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "full_cache benchmark: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
