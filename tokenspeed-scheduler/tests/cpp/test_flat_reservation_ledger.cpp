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

#include "cache/flat_reservation_ledger.h"

namespace tokenspeed {
namespace {

TEST(FlatReservationLedgerTest, PreparedKeysMaintainAggregateAcrossActiveLifetime) {
    FlatReservationLedger ledger(/*pool_count=*/2);
    ledger.ReservePrepared(/*additional=*/2);

    EXPECT_TRUE(ledger.Prepare("a"));
    EXPECT_TRUE(ledger.Prepare("b"));
    EXPECT_FALSE(ledger.Prepare("a"));
    ASSERT_NE(ledger.Find("a"), nullptr);
    EXPECT_EQ(*ledger.Find("a"), (PoolDemand{0, 0}));
    EXPECT_TRUE(ledger.Empty());

    ledger.SetPrepared("a", PoolDemand{3, 4});
    ledger.SetPrepared("b", PoolDemand{5, 6});
    EXPECT_EQ(ledger.Total(), (PoolDemand{8, 10}));
    EXPECT_FALSE(ledger.Empty());

    ledger.SetPrepared("a", PoolDemand{1, 2});
    EXPECT_EQ(ledger.Total(), (PoolDemand{6, 8}));

    ledger.ClearPrepared("a");
    ASSERT_NE(ledger.Find("a"), nullptr)
        << "an active zero-demand request keeps its preallocated map node";
    EXPECT_EQ(*ledger.Find("a"), (PoolDemand{0, 0}));
    EXPECT_EQ(ledger.Total(), (PoolDemand{5, 6}));

    EXPECT_TRUE(ledger.Erase("b"));
    EXPECT_FALSE(ledger.Erase("b"));
    EXPECT_TRUE(ledger.Empty());
    EXPECT_TRUE(ledger.Erase("a"));
}

}  // namespace
}  // namespace tokenspeed
