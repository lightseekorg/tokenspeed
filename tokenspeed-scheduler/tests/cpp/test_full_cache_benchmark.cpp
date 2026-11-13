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

#include "full_cache_benchmark.h"

namespace tokenspeed::benchmark {
namespace {

// Measure reads the thread CPU clock just outside its wall-clock reads, so a
// call's CPU time may exceed its wall time by about what those reads take.
constexpr double kClockReadsMs = 0.05;

// `duration` sums `calls` measured calls, and neither clock may report more
// time than the other allows.
void ExpectConsistent(const Duration& duration, int calls, const char* phase) {
    EXPECT_GE(duration.wall_ms, 0) << phase;
    EXPECT_GE(duration.cpu_ms, 0) << phase;
    EXPECT_LE(duration.cpu_ms, duration.wall_ms + calls * kClockReadsMs) << phase;
}

// As ExpectConsistent, for calls that do real work, so both clocks advance.
void ExpectMeasured(const Duration& duration, int calls, const char* phase) {
    ExpectConsistent(duration, calls, phase);
    EXPECT_GT(duration.wall_ms, 0) << phase;
    EXPECT_GT(duration.cpu_ms, 0) << phase;
}

TEST(FullCacheBenchmark, RestoresRealPrefixesAfterCacheAndSchedulerAdmissions) {
    for (int percent : {0, 30, 60, 90}) {
        SCOPED_TRACE(percent);
        // Include a short final history and evict across history boundaries.
        const FullCacheConfig config{
            .pool_pages = 257, .page_size = 4, .request_tokens = 80, .batch_size = 4, .hit_percent = percent};
        FullCacheWorkload workload(config, false, CacheLayout::kSequential);
        for (int repeat = 0; repeat < 3; ++repeat) {
            const auto cache = workload.RunCacheSample();
            const auto scheduler = workload.RunSchedulerSample();
            EXPECT_EQ(cache.hit_tokens, static_cast<std::uint64_t>(config.HitTokens()) * config.batch_size);
            EXPECT_EQ(scheduler.hit_tokens, cache.hit_tokens);
            EXPECT_EQ(cache.evicted_pages, config.EvictedPages());
            EXPECT_EQ(scheduler.evicted_pages, cache.evicted_pages);
            EXPECT_EQ(scheduler.admitted, config.batch_size);
            // The cache phases sum one call per request.
            ExpectMeasured(cache.hash, config.batch_size, "hash");
            ExpectMeasured(cache.prepare_keys, config.batch_size, "prepare_keys");
            ExpectMeasured(cache.probe, config.batch_size, "probe");
            ExpectMeasured(cache.admit, config.batch_size, "admit");
            ExpectMeasured(scheduler.submit, 1, "submit");
            ExpectMeasured(scheduler.plan, 1, "plan");
            EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(workload.Coordinator()).empty());
            for (const auto& hashes : workload.Histories()) {
                const auto probe = workload.Coordinator().ProbePrefix(hashes);
                EXPECT_EQ(probe.device.num_common_tokens, hashes.size() * config.page_size);
            }
        }
    }
}

TEST(FullCacheBenchmark, LifecycleRestoresPrefixesWithAndWithoutEvents) {
    for (bool events : {false, true}) {
        for (int percent : {0, 30, 60, 90}) {
            const FullCacheConfig config{257, 4, 80, 4, percent};
            FullCacheWorkload workload(config, events, CacheLayout::kSequential);
            for (int repeat = 0; repeat < 3; ++repeat) {
                const auto sample = workload.RunLifecycleSample();
                EXPECT_EQ(sample.published_prompt_pages, config.EvictedPages() - config.batch_size);
                EXPECT_EQ(sample.stored_event_blocks > 0, events);
                ExpectMeasured(sample.submit, 1, "submit");
                ExpectMeasured(sample.first_plan, 1, "first_plan");
                ExpectMeasured(sample.advance, 1, "advance");
                ExpectMeasured(sample.publish_plan, 1, "publish_plan");
                ExpectMeasured(sample.finish, 1, "finish");
                ExpectMeasured(sample.cleanup_plan, 1, "cleanup_plan");
                // Without KV events the drain may find nothing to do.
                if (events) {
                    ExpectMeasured(sample.events, 1, "events");
                } else {
                    ExpectConsistent(sample.events, 1, "events");
                }
                EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(workload.Coordinator()).empty());
                for (const auto& hashes : workload.Histories()) {
                    EXPECT_EQ(workload.Coordinator().ProbePrefix(hashes).device.num_common_tokens,
                              hashes.size() * config.page_size);
                }
            }
        }
    }
}

TEST(FullCacheBenchmark, ShuffledPhysicalPagesRestoreTheSamePrefixes) {
    for (int percent : {0, 30, 60, 90}) {
        const FullCacheConfig config{257, 4, 80, 4, percent};
        FullCacheWorkload workload(config, true, CacheLayout::kShuffled);
        for (int repeat = 0; repeat < 3; ++repeat) {
            const auto sample = workload.RunLifecycleSample();
            EXPECT_EQ(sample.published_prompt_pages, config.EvictedPages() - config.batch_size);
            EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(workload.Coordinator()).empty());
            for (const auto& hashes : workload.Histories()) {
                EXPECT_EQ(workload.Coordinator().ProbePrefix(hashes).device.num_common_tokens,
                          hashes.size() * config.page_size);
            }
        }
    }
}

TEST(FullCacheBenchmark, AlignsFractionalHitRatesToFullPages) {
    const FullCacheConfig config{
        .pool_pages = 3770160, .page_size = 64, .request_tokens = 131072, .batch_size = 16, .hit_percent = 30};
    EXPECT_EQ(config.HitTokens(), 39296);
    EXPECT_EQ(config.EvictedPages(), 22960);
    EXPECT_NO_THROW(config.Validate());
    EXPECT_THROW((FullCacheConfig{100, 64, 131072, 16, 30}.Validate()), std::runtime_error);
}

}  // namespace
}  // namespace tokenspeed::benchmark
