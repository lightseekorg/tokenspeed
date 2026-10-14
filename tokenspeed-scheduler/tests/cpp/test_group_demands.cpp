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
// (scheduler/operations/group_demands.h): dense or sparse page shapes for the
// token-spanned groups, a StateStep for a State group's two slots, and the
// reserve each History group holds beyond the tokens it computes.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "cache/core/cache_config.h"
#include "cache/core/cache_types.h"
#include "scheduler/operations/group_demands.h"

namespace tokenspeed::test {
namespace {

CacheGroupConfig Group(const char* id, AttnKind kind) {
    CacheGroupConfig group;
    group.group_id = id;
    group.block_granularity = kind == AttnKind::kState ? 0 : 4;
    group.total_pages = 8;
    group.attn_kind = kind;
    if (kind == AttnKind::kSlidingWindow) {
        group.sliding_window_tokens = 8;
    }
    return group;
}

TEST(MakeGroupDemandsTest, HistoryGroupsTakeThePrototypeAndStateGroupsTheStep) {
    const std::vector<CacheGroupConfig> groups = {
        Group("full", AttnKind::kFull),
        Group("state", AttnKind::kState),
    };
    std::vector<BlockTable> tables(2);
    const std::vector<GroupDemand> demands =
        MakeGroupDemands(tables, groups, GroupDemand{.extent = DenseGrowth{6}, .reserve_tokens = 3},
                         StateStep{StateStep::Kind::kRelease});
    ASSERT_EQ(demands.size(), 2u);
    EXPECT_EQ(demands[0].table, &tables[0]);
    EXPECT_EQ(demands[0].extent, (GroupExtent{DenseGrowth{6}}));
    EXPECT_EQ(demands[0].reserve_tokens, 3);
    EXPECT_EQ(demands[1].table, &tables[1]);
    EXPECT_EQ(demands[1].extent, (GroupExtent{StateStep{StateStep::Kind::kRelease}}));
    EXPECT_EQ(demands[1].reserve_tokens, 0) << "a State group's slots hold no token reserve";
}

TEST(ReservePrefillDemandsTest, EachKindHoldsItsOwnShareOfTheRound) {
    const std::vector<CacheGroupConfig> groups = {
        Group("full", AttnKind::kFull),
        Group("swa", AttnKind::kSlidingWindow),
        Group("state", AttnKind::kState),
    };
    std::vector<BlockTable> tables(3);
    // A first chunk that does not complete the prompt: full history prepays
    // the prompt headroom, the window holds nothing yet.
    std::vector<GroupDemand> demands =
        MakeGroupDemands(tables, groups, GroupDemand{.extent = DenseGrowth{6}}, StateStep{StateStep::Kind::kRotate});
    ReservePrefillDemands(
        demands, groups,
        PrefillReserve{.decode_input_tokens = 2, .completes_prefill = false, .prompt_headroom_tokens = 30});
    EXPECT_EQ(demands[0].reserve_tokens, 30);
    EXPECT_EQ(demands[1].reserve_tokens, 0);
    EXPECT_EQ(demands[2].reserve_tokens, 0);
    EXPECT_EQ(demands[0].extent, (GroupExtent{DenseGrowth{6}}));
    EXPECT_EQ(demands[1].extent, (GroupExtent{DenseGrowth{6}}));
    EXPECT_EQ(demands[2].extent, (GroupExtent{StateStep{StateStep::Kind::kRotate}}));
    // The completing chunk: every History group holds the decode slot; the
    // State group's step costs its fresh slot and nothing beyond.
    demands =
        MakeGroupDemands(tables, groups, GroupDemand{.extent = DenseGrowth{6}}, StateStep{StateStep::Kind::kRotate});
    ReservePrefillDemands(
        demands, groups,
        PrefillReserve{.decode_input_tokens = 2, .completes_prefill = true, .prompt_headroom_tokens = 0});
    EXPECT_EQ(demands[0].reserve_tokens, 2);
    EXPECT_EQ(demands[1].reserve_tokens, 2);
    EXPECT_EQ(demands[2].reserve_tokens, 0);
}

}  // namespace
}  // namespace tokenspeed::test
