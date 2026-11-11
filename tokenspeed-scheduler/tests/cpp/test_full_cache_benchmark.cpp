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
