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

// What a scheduled step asks the coordinator for, per cache group
// (scheduler/operations/group_demands.h): dense or sparse page shapes and the
// reserve each group holds beyond the tokens it computes.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_config.h"
#include "cache/core/cache_types.h"
#include "cache/coordinator/cache_coordinator.h"
#include "scheduler/operations/group_demands.h"

namespace tokenspeed::test {
namespace {

TEST(StateCheckpointMaterializationStartTest, SelectsLatestBoundaryOrEndpoint) {
    const struct {
        const char* name;
        std::int32_t before_tokens;
        std::int32_t after_tokens;
        std::int32_t expected_start;
    } cases[] = {
        {"internal checkpoint and continuation", 50432, 51300, 51200},
        {"continuation without internal checkpoint", 51200, 51300, 51300},
        {"checkpoint coincides with endpoint", 50432, 51200, 51200},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        EXPECT_EQ(StateCheckpointMaterializationStart(c.before_tokens, c.after_tokens, /*prefix_granularity=*/128),
                  c.expected_start);
    }
}

TEST(SnapshotStateReserveTokensTest, CoversGrowthAndDecodeWidth) {
    EXPECT_EQ(SnapshotStateReserveTokens(/*block_granularity=*/128, /*decode_tokens=*/1), 128);
    EXPECT_EQ(SnapshotStateReserveTokens(/*block_granularity=*/2, /*decode_tokens=*/3), 3);
}

CacheGroupConfig Group(const char* id, CacheGroupConfig::Retention retention, CacheGroupFamily family) {
    CacheGroupConfig group;
    group.group_id = id;
    group.block_granularity = 4;
    group.total_pages = 8;
    group.retention = retention;
    group.family = family;
    if (retention == CacheGroupConfig::Retention::SlidingWindow) {
        group.sliding_window_tokens = 8;
    }
    return group;
}

TEST(ReservePrefillDemandsTest, EachRetentionHoldsItsOwnShareOfTheRound) {
    const std::vector<CacheGroupConfig> groups = {
        Group("full", CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
        Group("swa", CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::History),
        Group("state", CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::State),
    };
    std::vector<BlockTable> tables(3);
    // A first chunk that does not complete the prompt: full history prepays
    // the prompt headroom, the window and the state hold nothing yet.
    std::vector<GroupDemand> demands = MakeGroupDemands(tables, GroupDemand{.extent = DenseGrowth{6}});
    ReservePrefillDemands(demands, groups,
                          PrefillReserve{.decode_input_tokens = 2,
                                         .completes_prefill = false,
                                         .prompt_headroom_tokens = 30,
                                         .reserve_snapshot_state_growth = false});
    EXPECT_EQ(demands[0].reserve_tokens, 30);
    EXPECT_EQ(demands[1].reserve_tokens, 0);
    EXPECT_EQ(demands[2].reserve_tokens, 0);
    for (const GroupDemand& demand : demands) {
        EXPECT_EQ(demand.extent, (GroupExtent{DenseGrowth{6}}));
    }
    // The completing chunk: every group holds the decode slot, the state
    // group at least one growth block.
    demands = MakeGroupDemands(tables, GroupDemand{.extent = DenseGrowth{6}});
    ReservePrefillDemands(demands, groups,
                          PrefillReserve{.decode_input_tokens = 2,
                                         .completes_prefill = true,
                                         .prompt_headroom_tokens = 0,
                                         .reserve_snapshot_state_growth = true});
    EXPECT_EQ(demands[0].reserve_tokens, 2);
    EXPECT_EQ(demands[1].reserve_tokens, 2);
    EXPECT_EQ(demands[2].reserve_tokens, 4) << "max(block_granularity, decode)";
}

TEST(MakeSnapshotStatePrefillSparseTest, MaterializesOnlyTheStateGroupsFromTheLastCheckpoint) {
    const std::vector<CacheGroupConfig> groups = {
        Group("full", CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
        Group("state", CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::State),
    };
    BlockPool pool(16, {1, 1});
    const std::vector<CacheGroupSpec> specs = {
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        CacheGroupSpec{.kind = AttnKind::kMambaState,
                       .sliding_window = 0,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 4},
    };
    const CacheCoordinator coord = MakeCoordinator(specs, 8, pool, /*enable_l3_storage=*/false, nullptr, false);
    std::vector<BlockTable> tables(2);
    std::vector<GroupDemand> demands = MakeGroupDemands(tables, GroupDemand{.extent = DenseGrowth{10}});
    // (0, 10] crosses the checkpoint at 8: the state group covers 10 tokens
    // from the checkpoint's slot (8 - 1) / 4 = 1; the history group is dense.
    MakeSnapshotStatePrefillSparse(demands, groups, coord, /*before_tokens=*/0, /*after_tokens=*/10);
    EXPECT_EQ(demands[0].extent, (GroupExtent{DenseGrowth{10}}));
    EXPECT_EQ(demands[1].extent, (GroupExtent{SparseSuffix{.extent_tokens = 10, .first_block = 1}}));
    // No checkpoint crossed: only the endpoint, slot (10 - 1) / 4 = 2.
    demands = MakeGroupDemands(tables, GroupDemand{.extent = DenseGrowth{2}});
    MakeSnapshotStatePrefillSparse(demands, groups, coord, /*before_tokens=*/8, /*after_tokens=*/10);
    EXPECT_EQ(demands[1].extent, (GroupExtent{SparseSuffix{.extent_tokens = 10, .first_block = 2}}));
}

}  // namespace
}  // namespace tokenspeed::test
