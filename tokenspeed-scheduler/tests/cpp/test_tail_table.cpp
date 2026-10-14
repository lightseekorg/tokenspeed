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

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cache/prefix/tail_table.h"

namespace tokenspeed::test {
namespace {

TEST(TailTableTest, RecordsTailsUnderTheirPriorInPublicationOrder) {
    TailTable table;
    EXPECT_FALSE(table.Add("p", 5, "h5").has_value());
    EXPECT_FALSE(table.Add("p", 2, "h2").has_value());
    EXPECT_FALSE(table.Add("", 7, "h7").has_value());

    const auto tails = table.Tails("p");
    ASSERT_EQ(tails.size(), 2u);
    EXPECT_EQ(tails[0].tail_tokens, 5);
    EXPECT_EQ(tails[0].hash, "h5");
    EXPECT_EQ(tails[1].tail_tokens, 2);
    EXPECT_EQ(table.Tails("").size(), 1u);
    EXPECT_TRUE(table.Tails("unknown").empty());
    EXPECT_EQ(table.TailTokensOf("h2"), 2);
    EXPECT_EQ(table.TailTokensOf("h7"), 7);
    EXPECT_FALSE(table.TailTokensOf("chain-hash").has_value());
    EXPECT_EQ(table.Size(), 3u);
}

TEST(TailTableTest, SameOffsetDifferentTokensAreTwoTails) {
    // Two prompts sharing the prior but diverging inside the page end at the
    // same offset under different hashes; both stay enumerable.
    TailTable table;
    table.Add("p", 3, "a");
    table.Add("p", 3, "b");
    EXPECT_EQ(table.Tails("p").size(), 2u);
    EXPECT_FALSE(table.Add("p", 3, "a").has_value()) << "re-publishing a hash adds nothing";
    EXPECT_EQ(table.Tails("p").size(), 2u);
    EXPECT_EQ(table.Tails("p").back().hash, "a") << "but makes it the most recently used";
}

TEST(TailTableTest, AHitOrARepublicationOutlivesEightNewerTails) {
    // The hot tail "h" is touched before each newcomer, so the cap drops the
    // newcomers in turn and never "h"; untouched, it would have gone first.
    TailTable table;
    table.Add("p", 1, "h");
    for (std::int32_t t = 2; t <= TailTable::kMaxTailsPerPrior + 3; ++t) {
        EXPECT_TRUE(table.Touch("h"));
        const auto dropped = table.Add("p", t, "n" + std::to_string(t));
        if (t <= TailTable::kMaxTailsPerPrior) {
            EXPECT_FALSE(dropped.has_value());
        } else {
            ASSERT_TRUE(dropped.has_value());
            EXPECT_EQ(dropped->hash, "n" + std::to_string(t - TailTable::kMaxTailsPerPrior + 1));
            EXPECT_EQ(dropped->tail_tokens, t - TailTable::kMaxTailsPerPrior + 1);
        }
    }
    EXPECT_TRUE(table.Contains("h"));
    EXPECT_FALSE(table.Touch("unknown"));
}

TEST(TailTableTest, RemoveForgetsTheHashAndEmptiesThePrior) {
    TailTable table;
    table.Add("p", 1, "a");
    table.Add("p", 2, "b");
    table.Remove("a");
    EXPECT_FALSE(table.Contains("a"));
    ASSERT_EQ(table.Tails("p").size(), 1u);
    EXPECT_EQ(table.Tails("p")[0].hash, "b");
    table.Remove("b");
    EXPECT_TRUE(table.Tails("p").empty());
    EXPECT_EQ(table.Size(), 0u);
    table.Remove("never-recorded");
}

TEST(TailTableTest, CapKeepsTheNewestEightPerPriorAndReportsTheDropped) {
    TailTable table;
    for (std::int32_t t = 1; t <= TailTable::kMaxTailsPerPrior; ++t) {
        EXPECT_FALSE(table.Add("p", t, "h" + std::to_string(t)).has_value());
    }
    EXPECT_EQ(table.CapEvictions(), 0);
    const auto dropped = table.Add("p", 9, "h9");
    ASSERT_TRUE(dropped.has_value());
    EXPECT_EQ(dropped->hash, "h1") << "the least recently used goes first";
    EXPECT_EQ(dropped->tail_tokens, 1);
    EXPECT_FALSE(table.Contains("h1"));
    EXPECT_EQ(table.Tails("p").size(), static_cast<std::size_t>(TailTable::kMaxTailsPerPrior));
    EXPECT_EQ(table.CapEvictions(), 1);
    EXPECT_TRUE(table.Tails("other").empty()) << "the cap is per prior";
    table.Add("other", 4, "o4");
    EXPECT_EQ(table.CapEvictions(), 1);
}

}  // namespace
}  // namespace tokenspeed::test
