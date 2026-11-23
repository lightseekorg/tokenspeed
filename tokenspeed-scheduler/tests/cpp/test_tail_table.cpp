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
    EXPECT_FALSE(table.Add(ContentHash{"p"}, 5, ContentHash{"h5"}).has_value());
    EXPECT_FALSE(table.Add(ContentHash{"p"}, 2, ContentHash{"h2"}).has_value());
    EXPECT_FALSE(table.Add(ContentHash{""}, 7, ContentHash{"h7"}).has_value());

    const auto tails = table.Tails(ContentHash{"p"});
    ASSERT_EQ(tails.size(), 2u);
    EXPECT_EQ(tails[0].tail_tokens, 5);
    EXPECT_EQ(tails[0].hash, ContentHash{"h5"});
    EXPECT_EQ(tails[1].tail_tokens, 2);
    EXPECT_EQ(table.Tails(ContentHash{""}).size(), 1u);
    EXPECT_TRUE(table.Tails(ContentHash{"unknown"}).empty());
    EXPECT_EQ(table.TailTokensOf(ContentHash{"h2"}), 2);
    EXPECT_EQ(table.TailTokensOf(ContentHash{"h7"}), 7);
    EXPECT_FALSE(table.TailTokensOf(ContentHash{"chain-hash"}).has_value());
    EXPECT_EQ(table.Size(), 3u);
}

TEST(TailTableTest, SameOffsetDifferentTokensAreTwoTails) {
    // Two prompts sharing the prior but diverging inside the page end at the
    // same offset under different hashes; both stay enumerable.
    TailTable table;
    table.Add(ContentHash{"p"}, 3, ContentHash{"a"});
    table.Add(ContentHash{"p"}, 3, ContentHash{"b"});
    EXPECT_EQ(table.Tails(ContentHash{"p"}).size(), 2u);
    EXPECT_FALSE(table.Add(ContentHash{"p"}, 3, ContentHash{"a"}).has_value()) << "re-publishing a hash adds nothing";
    EXPECT_EQ(table.Tails(ContentHash{"p"}).size(), 2u);
    EXPECT_EQ(table.Tails(ContentHash{"p"}).back().hash, ContentHash{"a"}) << "but makes it the most recently used";
}

TEST(TailTableTest, AHitOrARepublicationOutlivesEightNewerTails) {
    // The hot tail "h" is touched before each newcomer, so the cap drops the
    // newcomers in turn and never "h"; untouched, it would have gone first.
    TailTable table;
    table.Add(ContentHash{"p"}, 1, ContentHash{"h"});
    for (std::int32_t t = 2; t <= TailTable::kMaxTailsPerPrior + 3; ++t) {
        EXPECT_TRUE(table.Touch(ContentHash{"h"}));
        const auto dropped = table.Add(ContentHash{"p"}, t, ContentHash{"n" + std::to_string(t)});
        if (t <= TailTable::kMaxTailsPerPrior) {
            EXPECT_FALSE(dropped.has_value());
        } else {
            ASSERT_TRUE(dropped.has_value());
            EXPECT_EQ(dropped->hash, ContentHash{"n" + std::to_string(t - TailTable::kMaxTailsPerPrior + 1)});
            EXPECT_EQ(dropped->tail_tokens, t - TailTable::kMaxTailsPerPrior + 1);
        }
    }
    EXPECT_TRUE(table.Contains(ContentHash{"h"}));
    EXPECT_FALSE(table.Touch(ContentHash{"unknown"}));
}

TEST(TailTableTest, RemoveForgetsTheHashAndEmptiesThePrior) {
    TailTable table;
    table.Add(ContentHash{"p"}, 1, ContentHash{"a"});
    table.Add(ContentHash{"p"}, 2, ContentHash{"b"});
    table.Remove(ContentHash{"a"});
    EXPECT_FALSE(table.Contains(ContentHash{"a"}));
    ASSERT_EQ(table.Tails(ContentHash{"p"}).size(), 1u);
    EXPECT_EQ(table.Tails(ContentHash{"p"})[0].hash, ContentHash{"b"});
    table.Remove(ContentHash{"b"});
    EXPECT_TRUE(table.Tails(ContentHash{"p"}).empty());
    EXPECT_EQ(table.Size(), 0u);
    table.Remove(ContentHash{"never-recorded"});
}

TEST(TailTableTest, AnEmptyTableAnswersEveryLookupAndFillsAgain) {
    TailTable table;
    table.Remove(ContentHash{"never-recorded"});
    EXPECT_FALSE(table.Contains(ContentHash{"never-recorded"}));
    EXPECT_FALSE(table.TailTokensOf(ContentHash{"never-recorded"}).has_value());
    table.Add(ContentHash{"p"}, 3, ContentHash{"a"});
    table.Remove(ContentHash{"a"});
    EXPECT_EQ(table.Size(), 0u);
    EXPECT_FALSE(table.Contains(ContentHash{"a"})) << "emptied again";
    EXPECT_FALSE(table.TailTokensOf(ContentHash{"a"}).has_value());
    table.Add(ContentHash{"q"}, 2, ContentHash{"b"});
    EXPECT_TRUE(table.Contains(ContentHash{"b"}));
    EXPECT_EQ(table.TailTokensOf(ContentHash{"b"}), 2);
    EXPECT_FALSE(table.TailTokensOf(ContentHash{"a"}).has_value());
    table.Remove(ContentHash{"a"});
    EXPECT_EQ(table.Size(), 1u) << "removing an absent hash keeps the others";
}

TEST(TailTableTest, CapKeepsTheNewestEightPerPriorAndReportsTheDropped) {
    TailTable table;
    for (std::int32_t t = 1; t <= TailTable::kMaxTailsPerPrior; ++t) {
        EXPECT_FALSE(table.Add(ContentHash{"p"}, t, ContentHash{"h" + std::to_string(t)}).has_value());
    }
    EXPECT_EQ(table.CapEvictions(), 0);
    const auto dropped = table.Add(ContentHash{"p"}, 9, ContentHash{"h9"});
    ASSERT_TRUE(dropped.has_value());
    EXPECT_EQ(dropped->hash, ContentHash{"h1"}) << "the least recently used goes first";
    EXPECT_EQ(dropped->tail_tokens, 1);
    EXPECT_FALSE(table.Contains(ContentHash{"h1"}));
    EXPECT_EQ(table.Tails(ContentHash{"p"}).size(), static_cast<std::size_t>(TailTable::kMaxTailsPerPrior));
    EXPECT_EQ(table.CapEvictions(), 1);
    EXPECT_TRUE(table.Tails(ContentHash{"other"}).empty()) << "the cap is per prior";
    table.Add(ContentHash{"other"}, 4, ContentHash{"o4"});
    EXPECT_EQ(table.CapEvictions(), 1);
}

}  // namespace
}  // namespace tokenspeed::test
