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

// How a local prefill chunk is cut (scheduler/operations/prefill_chunk.h):
// the junction cut, the bounded-replay budget debit and final-window rule,
// and the mixed-mode reserve, all as pure arithmetic on the coordinator's
// facts.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache/coordinator/cache_coordinator.h"
#include "scheduler/operations/prefill_chunk.h"

namespace tokenspeed::test {
namespace {

TEST(ChunkKeepingFinalWindowTest, NeverLeavesAShortFinalChunk) {
    constexpr std::int32_t kW = 8;
    // Completing chunk: unchanged.
    EXPECT_EQ(ChunkKeepingFinalWindow(30, 30, kW), 30);
    // Remainder at least a window: unchanged.
    EXPECT_EQ(ChunkKeepingFinalWindow(64, 72, kW), 64);
    EXPECT_EQ(ChunkKeepingFinalWindow(64, 100, kW), 64);
    // Remainder inside (0, W): shorten so exactly W remain.
    EXPECT_EQ(ChunkKeepingFinalWindow(64, 70, kW), 62);
    EXPECT_EQ(ChunkKeepingFinalWindow(10, 11, kW), 3);
    // Everything left must go in one chunk and does not fit: wait.
    EXPECT_EQ(ChunkKeepingFinalWindow(4, 6, kW), 0);
    EXPECT_EQ(ChunkKeepingFinalWindow(0, 6, kW), 0);
    // No replayable group: identity.
    EXPECT_EQ(ChunkKeepingFinalWindow(64, 70, 0), 64);
}

// full (closed, g=4) + replayable swa (window 8) + replayable tail (window 2):
// the coordinator's replay window is the largest replayable window, 8.
std::vector<CacheGroupSpec> ReplayableSpecs() {
    return {
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 8,
                       .replayable = true,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 4},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 2,
                       .replayable = true,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2}};
}

std::vector<CacheGroupSpec> FullOnlySpecs() {
    return {CacheGroupSpec{
        .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
}

TEST(CoordinatorReplayTest, WindowIsTheLargestDeclaredAndAHitReplaysAtMostAWindow) {
    BlockPool pool(16, {1, 1, 1});
    const CacheCoordinator coord = MakeCoordinator(ReplayableSpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr,
                                                   /*snapshot_pool=*/nullptr, false);
    EXPECT_EQ(coord.ReplayWindowTokens(), 8);
    EXPECT_EQ(coord.ReplayTokens(0), 0);
    EXPECT_EQ(coord.ReplayTokens(5), 5);
    EXPECT_EQ(coord.ReplayTokens(8), 8);
    EXPECT_EQ(coord.ReplayTokens(64), 8);

    BlockPool plain_pool(16, {1});
    const CacheCoordinator plain = MakeCoordinator(FullOnlySpecs(), 4, plain_pool, /*enable_l3_storage=*/false, nullptr,
                                                   /*snapshot_pool=*/nullptr, false);
    EXPECT_EQ(plain.ReplayWindowTokens(), 0);
    EXPECT_EQ(plain.ReplayTokens(64), 0);
}

TEST(PrefillChunkTokensTest, DebitsTheHitReplayAndKeepsTheFinalWindow) {
    BlockPool pool(64, {1, 1, 1});
    const CacheCoordinator coord = MakeCoordinator(ReplayableSpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr,
                                                   /*snapshot_pool=*/nullptr, false);
    // A hit at 16 re-feeds 8 tokens first: 40 of budget leave 32 for new
    // tokens; a hit shorter than the window re-feeds only itself.
    EXPECT_EQ(
        PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/true, 100, 40, 0, /*enable_prefix_cache=*/true),
        32);
    EXPECT_EQ(
        PrefillChunkTokens(coord, /*first_pos=*/4, /*resumes_hit=*/true, 100, 40, 0, /*enable_prefix_cache=*/true), 36);
    // Nothing left for a new token: wait for a fresher budget.
    EXPECT_EQ(
        PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/true, 100, 8, 0, /*enable_prefix_cache=*/true), 0);
    // A later chunk starts where the previous one ended and re-feeds nothing.
    EXPECT_EQ(
        PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/false, 100, 40, 0, /*enable_prefix_cache=*/true),
        40);
    // The final chunk is never shorter than the window: 70 unscheduled under
    // a budget of 64 leaves 62 so that 8 remain; the remainder itself fits.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/0, /*resumes_hit=*/true, 70, 64, 0, /*enable_prefix_cache=*/true),
              62);
    EXPECT_EQ(
        PrefillChunkTokens(coord, /*first_pos=*/62, /*resumes_hit=*/false, 8, 64, 0, /*enable_prefix_cache=*/true), 8);
    // A junction cuts the chunk at any token, under the debited budget.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/true, 100, 40, /*junction=*/27,
                                 /*enable_prefix_cache=*/true),
              11);
    // A junction inside the final window cannot be honored without leaving a
    // short final chunk: the chunk passes it and completes the prompt instead
    // of waiting forever; when the whole remainder does not fit the budget
    // yet, it waits like any other final window.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/48, /*resumes_hit=*/false, /*unscheduled=*/8, 64,
                                 /*junction=*/52, /*enable_prefix_cache=*/true),
              8);
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/48, /*resumes_hit=*/false, /*unscheduled=*/8, 6,
                                 /*junction=*/52, /*enable_prefix_cache=*/true),
              0);

    // Without replayable groups the helper is min(budget, unscheduled), cut
    // at a pending junction.
    BlockPool plain_pool(16, {1});
    const CacheCoordinator plain = MakeCoordinator(FullOnlySpecs(), 4, plain_pool, /*enable_l3_storage=*/false, nullptr,
                                                   /*snapshot_pool=*/nullptr, false);
    EXPECT_EQ(PrefillChunkTokens(plain, 16, true, 100, 40, 0, /*enable_prefix_cache=*/true), 40);
    EXPECT_EQ(PrefillChunkTokens(plain, 16, true, 30, 40, 0, /*enable_prefix_cache=*/true), 30);
    EXPECT_EQ(PrefillChunkTokens(plain, 16, false, 100, 40, /*junction=*/21, /*enable_prefix_cache=*/true), 5);
    EXPECT_EQ(PrefillChunkTokens(plain, 21, false, 95, 40, /*junction=*/21, /*enable_prefix_cache=*/true), 40)
        << "a junction already passed";
    // With prefix caching disabled, State chunks are cut by budget alone.
    BlockPool state_pool(16, {1, 1});
    const std::vector<CacheGroupSpec> with_state = {
        FullOnlySpecs()[0],
        CacheGroupSpec{.attn_kind = AttnKind::kState,
                       .sliding_window = 0,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 0},
    };
    const CacheCoordinator hybrid = MakeCoordinator(with_state, 4, state_pool, /*enable_l3_storage=*/false, nullptr,
                                                    /*snapshot_pool=*/nullptr, false);
    EXPECT_EQ(PrefillChunkTokens(hybrid, 3, false, 100, 7, 0, /*enable_prefix_cache=*/false), 7);
    EXPECT_EQ(PrefillChunkTokens(hybrid, 3, false, 5, 7, 0, /*enable_prefix_cache=*/false), 5);
    // With caching, retain complete-page states before the changing prompt
    // tail. An exact-page prompt still needs a preceding checkpoint.
    EXPECT_EQ(PrefillChunkTokens(hybrid, 3, false, 100, 7, 0, /*enable_prefix_cache=*/true), 5);
    EXPECT_EQ(PrefillChunkTokens(hybrid, 3, false, 5, 7, 0, /*enable_prefix_cache=*/true), 1);
    EXPECT_EQ(PrefillChunkTokens(hybrid, 4, false, 4, 7, 0, /*enable_prefix_cache=*/true), 4);
    EXPECT_EQ(PrefillChunkTokens(hybrid, 0, false, 8, 8, 0, /*enable_prefix_cache=*/true), 4);
    // No usable boundary in this budget: make progress rather than waiting.
    EXPECT_EQ(PrefillChunkTokens(hybrid, 4, false, 100, 2, 0, /*enable_prefix_cache=*/true), 2);
    // An exact unaligned KV junction takes precedence over page alignment.
    EXPECT_EQ(PrefillChunkTokens(hybrid, 4, false, 100, 8, 7, /*enable_prefix_cache=*/true), 3);
}

TEST(PrefillChunkTokensTest, StateCheckpointsPreserveTheFinalReplayWindowAndProgress) {
    std::vector<CacheGroupSpec> groups = ReplayableSpecs();
    groups.push_back(CacheGroupSpec{
        .attn_kind = AttnKind::kState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 0});
    BlockPool pool(32, {1, 1, 1, 1});
    const CacheCoordinator coordinator = MakeCoordinator(groups, 4, pool, /*enable_l3_storage=*/false, nullptr,
                                                         /*snapshot_pool=*/nullptr, false);
    EXPECT_EQ(PrefillChunkTokens(coordinator, 0, false, 70, 64, 0, true), 60);
    EXPECT_EQ(PrefillChunkTokens(coordinator, 60, false, 10, 64, 0, true), 10);
    EXPECT_EQ(PrefillChunkTokens(coordinator, 48, false, 8, 64, 52, true), 8);
    for (const bool caching : {false, true}) {
        for (const bool resumes_hit : {false, true}) {
            for (std::int32_t first = 0; first <= 16; ++first) {
                for (std::int32_t remaining = 1; remaining <= 70; ++remaining) {
                    for (std::int32_t budget = 0; budget <= 32; ++budget) {
                        const auto length =
                            PrefillChunkTokens(coordinator, first, resumes_hit, remaining, budget, 0, caching);
                        const auto available =
                            std::max(0, budget - (resumes_hit ? coordinator.ReplayTokens(first) : 0));
                        ASSERT_GE(length, 0);
                        ASSERT_LE(length, std::min(remaining, available));
                        if (length > 0 && length < remaining) {
                            ASSERT_GE(remaining - length, coordinator.ReplayWindowTokens());
                        }
                        if (available >= remaining) {
                            ASSERT_GT(length, 0) << "a sufficient budget must advance the prompt";
                        }
                    }
                }
            }
        }
    }
}

TEST(MinPrefillChunkTokensTest, ReservesReplayPlusAWindowOrPage) {
    BlockPool pool(16, {1, 1, 1});
    // W = 8 > P = 4: replay plus another window.
    EXPECT_EQ(MinPrefillChunkTokens(MakeCoordinator(ReplayableSpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr,
                                                    /*snapshot_pool=*/nullptr, false)),
              16);
    // W = 8 < P = 16: replay plus one prefix page, so an aligned chunk fits.
    BlockPool wide_pool(16, {1, 1, 1});
    EXPECT_EQ(MinPrefillChunkTokens(MakeCoordinator(ReplayableSpecs(), 16, wide_pool, /*enable_l3_storage=*/false,
                                                    nullptr, /*snapshot_pool=*/nullptr, false)),
              24);
    BlockPool plain_pool(16, {1});
    EXPECT_EQ(MinPrefillChunkTokens(MakeCoordinator(FullOnlySpecs(), 4, plain_pool, /*enable_l3_storage=*/false,
                                                    nullptr, /*snapshot_pool=*/nullptr, false)),
              0);
    // A State group reserves nothing: its chunks are cut by budget alone.
    BlockPool state_pool(16, {1, 1});
    const std::vector<CacheGroupSpec> with_state = {
        FullOnlySpecs()[0],
        CacheGroupSpec{.attn_kind = AttnKind::kState,
                       .sliding_window = 0,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 0},
    };
    EXPECT_EQ(MinPrefillChunkTokens(MakeCoordinator(with_state, 4, state_pool, /*enable_l3_storage=*/false, nullptr,
                                                    /*snapshot_pool=*/nullptr, false)),
              0);
}

}  // namespace
}  // namespace tokenspeed::test
