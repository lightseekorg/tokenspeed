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
// the alignment rules for snapshot-state and promotion boundaries, the
// bounded-replay budget debit and final-window rule, and the mixed-mode
// reserve, all as pure arithmetic on the coordinator's facts.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache/coordinator/cache_coordinator.h"
#include "scheduler/operations/prefill_chunk.h"

namespace tokenspeed::test {
namespace {

TEST(AlignPrefillChunkTest, RespectsBudgetPromotionAndFinalExtent) {
    const struct {
        const char* name;
        std::int32_t first_pos;
        std::int32_t unscheduled;
        std::int32_t token_budget;
        std::int32_t prefix_granularity;
        std::int32_t promotion_boundary;
        std::int32_t expected_tokens;
    } cases[] = {
        {"first chunk reaches promotion", 16, 24, 24, 4, 32, 16},
        {"budget precedes promotion", 16, 24, 8, 4, 32, 8},
        {"later chunk reaches promotion", 24, 16, 16, 4, 32, 8},
        {"prompt ends before promotion", 24, 4, 16, 4, 32, 4},
        {"promotion already reached", 32, 16, 10, 4, 32, 8},
        {"final extent crosses checkpoint", 50432, 868, 868, 128, 0, 868},
        {"budget truncates final extent", 50432, 868, 800, 128, 0, 768},
        {"aligned final extent", 51200, 768, 868, 128, 0, 768},
        {"final extent has no internal checkpoint", 51200, 100, 868, 128, 0, 100},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        EXPECT_EQ(
            AlignPrefillChunk(c.first_pos, c.unscheduled, c.token_budget, c.prefix_granularity, c.promotion_boundary),
            c.expected_tokens);
    }
}

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
    return {CacheGroupSpec{
                .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
            CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                           .sliding_window = 8,
                           .replayable = true,
                           .cache_blocks_per_lcm_block = 1,
                           .block_granularity = 4},
            CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                           .sliding_window = 2,
                           .replayable = true,
                           .cache_blocks_per_lcm_block = 1,
                           .block_granularity = 2}};
}

std::vector<CacheGroupSpec> FullOnlySpecs() {
    return {CacheGroupSpec{
        .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
}

TEST(CoordinatorReplayTest, WindowIsTheLargestDeclaredAndAHitReplaysAtMostAWindow) {
    BlockPool pool(16, {1, 1, 1});
    const CacheCoordinator coord =
        MakeCoordinator(ReplayableSpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr, false);
    EXPECT_EQ(coord.ReplayWindowTokens(), 8);
    EXPECT_EQ(coord.ReplayTokens(0), 0);
    EXPECT_EQ(coord.ReplayTokens(5), 5);
    EXPECT_EQ(coord.ReplayTokens(8), 8);
    EXPECT_EQ(coord.ReplayTokens(64), 8);

    BlockPool plain_pool(16, {1});
    const CacheCoordinator plain =
        MakeCoordinator(FullOnlySpecs(), 4, plain_pool, /*enable_l3_storage=*/false, nullptr, false);
    EXPECT_EQ(plain.ReplayWindowTokens(), 0);
    EXPECT_EQ(plain.ReplayTokens(64), 0);
}

TEST(PrefillChunkTokensTest, DebitsTheHitReplayAndKeepsTheFinalWindow) {
    BlockPool pool(64, {1, 1, 1});
    const CacheCoordinator coord =
        MakeCoordinator(ReplayableSpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr, false);
    // A hit at 16 re-feeds 8 tokens first: 40 of budget leave 32 for new
    // tokens; a hit shorter than the window re-feeds only itself.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/true, 100, 40, 0), 32);
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/4, /*resumes_hit=*/true, 100, 40, 0), 36);
    // Nothing left for a new token: wait for a fresher budget.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/true, 100, 8, 0), 0);
    // A later chunk starts where the previous one ended and re-feeds nothing.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/false, 100, 40, 0), 40);
    // The final chunk is never shorter than the window: 70 unscheduled under
    // a budget of 64 leaves 62 so that 8 remain; the remainder itself fits.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/0, /*resumes_hit=*/true, 70, 64, 0), 62);
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/62, /*resumes_hit=*/false, 8, 64, 0), 8);
    // A promotion boundary still aligns the chunk (host tier), under the
    // debited budget.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/true, 100, 40, /*promotion=*/24), 8);
    // A promotion boundary inside the final window cannot be honored without
    // leaving a short final chunk: the chunk passes it and completes the
    // prompt instead of waiting forever; when the whole remainder does not
    // fit the budget yet, it waits like any other final window.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/48, /*resumes_hit=*/false, /*unscheduled=*/8, 64,
                                 /*promotion=*/52),
              8);
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/48, /*resumes_hit=*/false, /*unscheduled=*/8, 6,
                                 /*promotion=*/52),
              0);

    // Without replayable groups the helper is min(budget, unscheduled) plus
    // the alignment the state/promotion cases already had.
    BlockPool plain_pool(16, {1});
    const CacheCoordinator plain =
        MakeCoordinator(FullOnlySpecs(), 4, plain_pool, /*enable_l3_storage=*/false, nullptr, false);
    EXPECT_EQ(PrefillChunkTokens(plain, 16, true, 100, 40, 0), 40);
    EXPECT_EQ(PrefillChunkTokens(plain, 16, true, 30, 40, 0), 30);
    EXPECT_EQ(PrefillChunkTokens(plain, 16, false, 100, 40, /*promotion=*/24), 8);
}

TEST(MinPrefillChunkTokensTest, ReservesACheckpointPageOrReplayPlusAWindowOrPage) {
    BlockPool pool(16, {1, 1, 1});
    // W = 8 > P = 4: replay plus another window.
    EXPECT_EQ(
        MinPrefillChunkTokens(MakeCoordinator(ReplayableSpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr, false)),
        16);
    // W = 8 < P = 16: replay plus one prefix page, so an aligned chunk fits.
    BlockPool wide_pool(16, {1, 1, 1});
    EXPECT_EQ(MinPrefillChunkTokens(
                  MakeCoordinator(ReplayableSpecs(), 16, wide_pool, /*enable_l3_storage=*/false, nullptr, false)),
              24);
    BlockPool plain_pool(16, {1});
    EXPECT_EQ(MinPrefillChunkTokens(
                  MakeCoordinator(FullOnlySpecs(), 4, plain_pool, /*enable_l3_storage=*/false, nullptr, false)),
              0);
    BlockPool state_pool(16, {1, 1});
    const std::vector<CacheGroupSpec> with_state = {
        FullOnlySpecs()[0],
        CacheGroupSpec{.kind = AttnKind::kMambaState,
                       .sliding_window = 0,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 4},
    };
    EXPECT_EQ(
        MinPrefillChunkTokens(MakeCoordinator(with_state, 4, state_pool, /*enable_l3_storage=*/false, nullptr, false)),
        4);
}

}  // namespace
}  // namespace tokenspeed::test
