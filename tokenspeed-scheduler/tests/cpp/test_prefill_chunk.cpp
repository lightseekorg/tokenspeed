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
// the mixed-mode reserve and the budget a chunk gives way with, all as pure
// arithmetic on the coordinator's facts.

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
                                                   /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false,
                                                   /*checkpoint_retention_requests=*/0);
    EXPECT_EQ(coord.ReplayWindowTokens(), 8);
    EXPECT_EQ(coord.ReplayTokens(0), 0);
    EXPECT_EQ(coord.ReplayTokens(5), 5);
    EXPECT_EQ(coord.ReplayTokens(8), 8);
    EXPECT_EQ(coord.ReplayTokens(64), 8);

    BlockPool plain_pool(16, {1});
    const CacheCoordinator plain = MakeCoordinator(FullOnlySpecs(), 4, plain_pool, /*enable_l3_storage=*/false, nullptr,
                                                   /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false,
                                                   /*checkpoint_retention_requests=*/0);
    EXPECT_EQ(plain.ReplayWindowTokens(), 0);
    EXPECT_EQ(plain.ReplayTokens(64), 0);
}

TEST(PrefillChunkTokensTest, DebitsTheHitReplayAndKeepsTheFinalWindow) {
    BlockPool pool(64, {1, 1, 1});
    const CacheCoordinator coord = MakeCoordinator(ReplayableSpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr,
                                                   /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false,
                                                   /*checkpoint_retention_requests=*/0);
    // A hit at 16 re-feeds 8 tokens first: 40 of budget leave 32 for new
    // tokens; a hit shorter than the window re-feeds only itself.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/true, 100, 40, 0, std::nullopt, 0), 32);
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/4, /*resumes_hit=*/true, 100, 40, 0, std::nullopt, 0), 36);
    // Nothing left for a new token: wait for a fresher budget.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/true, 100, 8, 0, std::nullopt, 0), 0);
    // A later chunk starts where the previous one ended and re-feeds nothing.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/false, 100, 40, 0, std::nullopt, 0), 40);
    // The final chunk is never shorter than the window: 70 unscheduled under
    // a budget of 64 leaves 62 so that 8 remain; the remainder itself fits.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/0, /*resumes_hit=*/true, 70, 64, 0, std::nullopt, 0), 62);
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/62, /*resumes_hit=*/false, 8, 64, 0, std::nullopt, 0), 8);
    // A junction cuts the chunk at any token, under the debited budget.
    EXPECT_EQ(
        PrefillChunkTokens(coord, /*first_pos=*/16, /*resumes_hit=*/true, 100, 40, /*junction=*/27, std::nullopt, 0),
        11);
    // A junction inside the final window cannot be honored without leaving a
    // short final chunk: the chunk passes it and completes the prompt instead
    // of waiting forever; when the whole remainder does not fit the budget
    // yet, it waits like any other final window.
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/48, /*resumes_hit=*/false, /*unscheduled=*/8, 64,
                                 /*junction=*/52, std::nullopt, 0),
              8);
    EXPECT_EQ(PrefillChunkTokens(coord, /*first_pos=*/48, /*resumes_hit=*/false, /*unscheduled=*/8, 6,
                                 /*junction=*/52, std::nullopt, 0),
              0);

    // Without replayable groups the helper is min(budget, unscheduled), cut
    // at a pending junction.
    BlockPool plain_pool(16, {1});
    const CacheCoordinator plain = MakeCoordinator(FullOnlySpecs(), 4, plain_pool, /*enable_l3_storage=*/false, nullptr,
                                                   /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false,
                                                   /*checkpoint_retention_requests=*/0);
    EXPECT_EQ(PrefillChunkTokens(plain, 16, true, 100, 40, 0, std::nullopt, 0), 40);
    EXPECT_EQ(PrefillChunkTokens(plain, 16, true, 30, 40, 0, std::nullopt, 0), 30);
    EXPECT_EQ(PrefillChunkTokens(plain, 16, false, 100, 40, /*junction=*/21, std::nullopt, 0), 5);
    EXPECT_EQ(PrefillChunkTokens(plain, 21, false, 95, 40, /*junction=*/21, std::nullopt, 0), 40)
        << "a junction already passed";
    // A State group adds no alignment: chunks are cut by budget alone.
    BlockPool state_pool(16, {1, 1});
    const std::vector<CacheGroupSpec> with_state = {
        FullOnlySpecs()[0],
        CacheGroupSpec{.attn_kind = AttnKind::kState,
                       .sliding_window = 0,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 0},
    };
    const CacheCoordinator hybrid = MakeCoordinator(with_state, 4, state_pool, /*enable_l3_storage=*/false, nullptr,
                                                    /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false,
                                                    /*checkpoint_retention_requests=*/0);
    EXPECT_EQ(PrefillChunkTokens(hybrid, 3, false, 100, 7, 0, std::nullopt, 0), 7);
    EXPECT_EQ(PrefillChunkTokens(hybrid, 3, false, 5, 7, 0, std::nullopt, 0), 5);
}

// A chunk that cannot finish gives way to whole prompts behind it
// (SchedulerConfig::prefill_chunk_yield_tokens): it ends on the last page at
// least the yielded tokens before its budget's end.
TEST(YieldingChunkBudgetTest, AChunkThatCannotFinishEndsOnAPageBeforeTheYield) {
    BlockPool plain_pool(16, {1});
    const CacheCoordinator plain = MakeCoordinator(FullOnlySpecs(), 4, plain_pool, /*enable_l3_storage=*/false, nullptr,
                                                   /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false,
                                                   /*checkpoint_retention_requests=*/0);
    // (first_pos, resumes_hit, unscheduled, budget, yield)
    EXPECT_EQ(YieldingChunkBudget(plain, 16, false, 100, 40, 0), 40) << "nothing to yield";
    EXPECT_EQ(YieldingChunkBudget(plain, 16, false, 40, 40, 12), 40) << "the chunk finishes within the budget";
    EXPECT_EQ(YieldingChunkBudget(plain, 16, false, 100, 40, 10), 28) << "16 + 40 - 10 = 46, down to the page at 44";
    EXPECT_EQ(YieldingChunkBudget(plain, 16, false, 100, 40, 8), 32) << "a whole page yields exactly";
    EXPECT_EQ(YieldingChunkBudget(plain, 17, false, 100, 40, 10), 27) << "from inside a page, still ending at 44";
    EXPECT_EQ(YieldingChunkBudget(plain, 0, true, 100, 40, 10), 28) << "a hit's first chunk without replay";
    // A cut leaving less than a page of new tokens takes the whole budget.
    EXPECT_EQ(YieldingChunkBudget(plain, 16, false, 100, 8, 4), 4);
    EXPECT_EQ(YieldingChunkBudget(plain, 17, false, 100, 8, 4), 8) << "17 + 8 - 4 = 21: the page at 20 leaves 3";
    EXPECT_EQ(YieldingChunkBudget(plain, 16, false, 100, 8, 8), 8);
    // Every end such a budget allows is a page for PrefillChunkTokens too.
    EXPECT_EQ(PrefillChunkTokens(plain, 17, false, 100, YieldingChunkBudget(plain, 17, false, 100, 40, 10), 0,
                                 std::nullopt, 0),
              27);

    // A hit's replay window comes out of the budget first: it counts towards
    // whether the chunk finishes, and the window rides on top of the new
    // tokens that end on a page.
    BlockPool pool(64, {1, 1, 1});
    const CacheCoordinator replay = MakeCoordinator(ReplayableSpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr,
                                                    /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false,
                                                    /*checkpoint_retention_requests=*/0);
    EXPECT_EQ(YieldingChunkBudget(replay, 16, true, 36, 40, 10), 28) << "36 + 8 replayed > 40: 16 + 40 - 8 - 10 = 38";
    EXPECT_EQ(PrefillChunkTokens(replay, 16, true, 36, 28, 0, std::nullopt, 0), 20) << "ends on the page at 36";
    EXPECT_EQ(YieldingChunkBudget(replay, 16, true, 32, 40, 10), 40) << "32 + 8 replayed fit";
    EXPECT_EQ(YieldingChunkBudget(replay, 16, false, 100, 40, 10), 28) << "a later chunk replays nothing";
    // The least a chunk computes here is MinPrefillChunkTokens = 16.
    EXPECT_EQ(YieldingChunkBudget(replay, 16, true, 100, 36, 10), 24) << "16 new tokens, 32 - 16";
    EXPECT_EQ(YieldingChunkBudget(replay, 16, true, 100, 36, 14), 36) << "12 new tokens would be too few";

    // Restricted State groups: a chunk holds whole checkpoint segments.
    std::vector<CacheGroupSpec> restricted = FullOnlySpecs();
    restricted.push_back(CacheGroupSpec{.attn_kind = AttnKind::kState,
                                        .cache_blocks_per_lcm_block = 1,
                                        .block_granularity = 0,
                                        .checkpoint_granularity = 8});
    BlockPool restricted_pool(16, {1, 1});
    const CacheCoordinator aligned = MakeCoordinator(restricted, 4, restricted_pool, /*enable_l3_storage=*/false,
                                                     nullptr, /*snapshot_pool=*/nullptr, false,
                                                     /*write_back_evictions=*/false,
                                                     /*checkpoint_retention_requests=*/0);
    EXPECT_EQ(YieldingChunkBudget(aligned, 8, false, 100, 40, 10), 24) << "8 + 40 - 10 = 38, down to the segment at 32";
    EXPECT_EQ(PrefillChunkTokens(aligned, 8, false, 100, 24, 0, std::nullopt, 0), 24);
    EXPECT_EQ(YieldingChunkBudget(aligned, 8, false, 100, 16, 8), 8) << "one segment";
    EXPECT_EQ(YieldingChunkBudget(aligned, 8, false, 100, 16, 9), 16) << "less than a segment: the whole budget";
}

TEST(PrefillStateCheckpointTest, LongPromptsLeaveAReusablePageBeforeTheirReplayTail) {
    auto specs = FullOnlySpecs();
    specs.push_back(
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0});
    BlockPool pool(64, {1, 1});
    const CacheCoordinator hybrid =
        MakeCoordinator(specs, 4, pool, false, nullptr, nullptr, false, /*write_back_evictions=*/false,
                        /*checkpoint_retention_requests=*/0);
    EXPECT_EQ(PrefillStateCheckpointTokens(hybrid, 26, 8, 0), 24);
    EXPECT_EQ(PrefillStateCheckpointTokens(hybrid, 24, 8, 0), 20) << "leave the last logits token";
    EXPECT_EQ(PrefillStateCheckpointTokens(hybrid, 8, 8, 0), 0) << "a prompt within one budget needs none";
    EXPECT_EQ(PrefillStateCheckpointTokens(hybrid, 26, 8, 7), 16) << "leave the configured replay tail";
    EXPECT_EQ(PrefillStateCheckpointTokens(hybrid, 4, 2, 8), 0) << "no positive checkpoint fits";

    BlockPool plain_pool(64, {1});
    const CacheCoordinator plain = MakeCoordinator(FullOnlySpecs(), 4, plain_pool, false, nullptr, nullptr, false,
                                                   /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    EXPECT_EQ(PrefillStateCheckpointTokens(plain, 26, 8, 0), 0) << "no State group, no checkpoint";

    auto replay_specs = ReplayableSpecs();
    replay_specs.push_back(specs.back());
    BlockPool replay_pool(64, {1, 1, 1, 1});
    const CacheCoordinator replay =
        MakeCoordinator(replay_specs, 4, replay_pool, false, nullptr, nullptr, false, /*write_back_evictions=*/false,
                        /*checkpoint_retention_requests=*/0);
    EXPECT_EQ(PrefillStateCheckpointTokens(replay, 30, 16, 0), 20) << "leave the largest replayable window";
}

TEST(MinPrefillChunkTokensTest, ReservesReplayPlusAWindowOrPage) {
    BlockPool pool(16, {1, 1, 1});
    // W = 8 > P = 4: replay plus another window.
    EXPECT_EQ(MinPrefillChunkTokens(MakeCoordinator(ReplayableSpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr,
                                                    /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false,
                                                    /*checkpoint_retention_requests=*/0)),
              16);
    // W = 8 < P = 16: replay plus one prefix page, so an aligned chunk fits.
    BlockPool wide_pool(16, {1, 1, 1});
    EXPECT_EQ(
        MinPrefillChunkTokens(MakeCoordinator(ReplayableSpecs(), 16, wide_pool, /*enable_l3_storage=*/false, nullptr,
                                              /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false,
                                              /*checkpoint_retention_requests=*/0)),
        24);
    BlockPool plain_pool(16, {1});
    EXPECT_EQ(
        MinPrefillChunkTokens(MakeCoordinator(FullOnlySpecs(), 4, plain_pool, /*enable_l3_storage=*/false, nullptr,
                                              /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false,
                                              /*checkpoint_retention_requests=*/0)),
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
                                                    /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false,
                                                    /*checkpoint_retention_requests=*/0)),
              0);
}

TEST(PrefillComputeTest, ConservativeBoundaryForLongPrefixesAndLinearCosts) {
    const PrefillComputeConfig quadratic{.budget_ns = 36, .token_ns = 0, .pair_ns = 1, .request_ns = 0};
    EXPECT_EQ(quadratic.TokenLimit(0, 100, 0), 8);
    EXPECT_EQ(quadratic.TokenLimit(8, 100, 0), 3);
    EXPECT_EQ(quadratic.TokenLimit(0, 4, 0), 4);
    EXPECT_EQ(quadratic.TokenLimit(0, 100, 36), 0);
    const PrefillComputeConfig linear{.budget_ns = 20, .token_ns = 2, .pair_ns = 0, .request_ns = 4};
    EXPECT_EQ(linear.TokenLimit(1000000000, 100, 0), 8);
    EXPECT_EQ(linear.TokenLimit(1000000000, 100, 6), 5);
    EXPECT_EQ(linear.Cost(0, 100), 0);
}

TEST(PrefillComputeTest, LargeTokenCountsDoNotOverflowIntegerArithmetic) {
    constexpr std::int32_t limit = 2147483647;
    const PrefillComputeConfig compute{.budget_ns = 1e30, .token_ns = 1, .pair_ns = 1, .request_ns = 0};
    EXPECT_EQ(compute.TokenLimit(limit, limit, 0), limit);
    EXPECT_GT(compute.Cost(limit, limit), 6e18);
}

TEST(PrefillComputeTest, AHitStillFitsTheHardTokenBudgetWhenTimeBudgetIsTiny) {
    BlockPool pool(64, {1, 1, 1});
    const CacheCoordinator coord =
        MakeCoordinator(ReplayableSpecs(), 4, pool, false, nullptr, nullptr, false, false, 0);
    const PrefillComputeConfig compute{.budget_ns = 1, .token_ns = 100, .pair_ns = 1, .request_ns = 0};
    EXPECT_EQ(PrefillChunkTokens(coord, 16, true, 100, 8, 0, compute, 0), 0);
    EXPECT_EQ(PrefillChunkTokens(coord, 16, true, 100, 16, 0, compute, 0), 8);
    EXPECT_EQ(PrefillChunkTokens(coord, 16, true, 100, 32, 0, compute, 1), 0);
    // A junction remains a natural checkpoint even when the compute cap is active.
    EXPECT_EQ(PrefillChunkTokens(coord, 16, true, 100, 16, 20, compute, 0), 4);
}

}  // namespace
}  // namespace tokenspeed::test
