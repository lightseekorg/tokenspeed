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

#include "cache/block_pool.h"
#include "cache/full_attn_manager.h"
#include "cache/swa_manager.h"

namespace tokenspeed::test {
namespace {

// Publish a host page for `key` (the scheduler's store path minus the D2H write):
// allocate -> hash -> free leaves it cached-and-evictable, exactly like a committed store.
CacheBlock* Put(BlockPool& host_pool, const std::string& key) {
    CacheBlock* block = host_pool.AllocateBlocks(1).front();
    host_pool.CacheFullBlock(block, key);
    host_pool.FreeBlocks({block});
    return block;
}

TEST(MatchHostPagesTest, FullWalksContiguousRunFromBegin) {
    BlockPool pool(4);
    FullAttnManager mgr(pool, /*page_size=*/4);
    EXPECT_TRUE(mgr.MatchIsPrefixClosed());
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3", "k4"};
    std::vector<CacheBlock*> put;
    for (std::size_t j = 1; j <= 4; ++j) {
        put.push_back(Put(host_pool, keys[j]));
    }
    // Slots below begin=1 are device-valid; the run covers all extension slots, no holes.
    std::vector<CacheBlock*> pages = mgr.MatchHostPages(host_pool, keys, /*begin_blocks=*/1, /*max_blocks=*/5);
    EXPECT_EQ(pages, put);
}

TEST(MatchHostPagesTest, FullStopsAtFirstMiss) {
    BlockPool pool(4);
    FullAttnManager mgr(pool, 4);
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3"};
    CacheBlock* p0 = Put(host_pool, keys[0]);
    CacheBlock* p1 = Put(host_pool, keys[1]);
    (void)Put(host_pool, keys[3]);  // beyond the gap at k2: unreachable
    std::vector<CacheBlock*> pages = mgr.MatchHostPages(host_pool, keys, 0, 4);
    EXPECT_EQ(pages, (std::vector<CacheBlock*>{p0, p1}));
}

TEST(MatchHostPagesTest, FullEmptyOnBeginMissOrEmptyRange) {
    BlockPool pool(4);
    FullAttnManager mgr(pool, 4);
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1"};
    (void)Put(host_pool, keys[1]);
    EXPECT_TRUE(mgr.MatchHostPages(host_pool, keys, 0, 2).empty());  // miss right at begin
    EXPECT_TRUE(mgr.MatchHostPages(host_pool, keys, 2, 2).empty());  // empty extension range
}

TEST(MatchHostPagesTest, SwaTrailingRunAtEnd) {
    BlockPool pool(4);
    // page_size 4, window 10 -> contiguous_needed = ceil(9/4) = 3.
    SwaManager mgr(pool, 4, /*sliding_window=*/10);
    EXPECT_FALSE(mgr.MatchIsPrefixClosed());
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3", "k4"};
    CacheBlock* p2 = Put(host_pool, keys[2]);
    CacheBlock* p3 = Put(host_pool, keys[3]);
    CacheBlock* p4 = Put(host_pool, keys[4]);
    // Trailing run [2, 5) covers the window at boundary 5; slots below stay holes.
    std::vector<CacheBlock*> pages = mgr.MatchHostPages(host_pool, keys, 0, 5);
    EXPECT_EQ(pages, (std::vector<CacheBlock*>{nullptr, nullptr, p2, p3, p4}));
}

TEST(MatchHostPagesTest, SwaInteriorBoundaryShrink) {
    BlockPool pool(4);
    SwaManager mgr(pool, 4, 10);  // needed = 3
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3", "k4"};
    CacheBlock* p1 = Put(host_pool, keys[1]);
    CacheBlock* p2 = Put(host_pool, keys[2]);
    CacheBlock* p3 = Put(host_pool, keys[3]);
    // Miss at 4 invalidates boundary 5; boundary 4 needs [1, 4), which hits.
    std::vector<CacheBlock*> pages = mgr.MatchHostPages(host_pool, keys, 0, 5);
    EXPECT_EQ(pages, (std::vector<CacheBlock*>{nullptr, p1, p2, p3}));
}

TEST(MatchHostPagesTest, SwaShortRunAtBottomSuffices) {
    BlockPool pool(4);
    SwaManager mgr(pool, 4, 10);  // needed = 3, but only 2 extension slots exist
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1"};
    CacheBlock* p0 = Put(host_pool, keys[0]);
    CacheBlock* p1 = Put(host_pool, keys[1]);
    // The window clamps to begin: a full 2-run from the bottom is a valid boundary 2.
    std::vector<CacheBlock*> pages = mgr.MatchHostPages(host_pool, keys, 0, 2);
    EXPECT_EQ(pages, (std::vector<CacheBlock*>{p0, p1}));
}

TEST(MatchHostPagesTest, SwaBeginAboveZeroInteriorBoundary) {
    BlockPool pool(4);
    SwaManager mgr(pool, 4, /*sliding_window=*/9);  // needed = ceil(8/4) = 2
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3", "k4", "k5", "k6"};
    CacheBlock* p3 = Put(host_pool, keys[3]);
    CacheBlock* p4 = Put(host_pool, keys[4]);
    CacheBlock* p5 = Put(host_pool, keys[5]);
    (void)p3;  // hit at slot 3 sits below the winning run's window and stays a hole
    // Miss at 6 invalidates boundary 7; boundary 6 needs [4, 6), which hits -> vector
    // covers [3, 6): hole at slot 3, pages for 4 and 5.
    std::vector<CacheBlock*> pages = mgr.MatchHostPages(host_pool, keys, /*begin_blocks=*/3, /*max_blocks=*/7);
    EXPECT_EQ(pages, (std::vector<CacheBlock*>{nullptr, p4, p5}));
}

TEST(MatchHostPagesTest, SwaAllMissReturnsEmpty) {
    BlockPool pool(4);
    SwaManager mgr(pool, 4, 10);
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3", "k4"};
    EXPECT_TRUE(mgr.MatchHostPages(host_pool, keys, 1, 5).empty());
}

TEST(MatchHostPagesTest, SwaZeroNeededWindowAcceptsAllAsHoles) {
    BlockPool pool(4);
    SwaManager mgr(pool, 4, /*sliding_window=*/1);  // needed = 0
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2"};
    // Zero needed pages: every boundary is resumable with no host page at all.
    std::vector<CacheBlock*> pages = mgr.MatchHostPages(host_pool, keys, 1, 3);
    EXPECT_EQ(pages, (std::vector<CacheBlock*>{nullptr, nullptr}));
}

}  // namespace
}  // namespace tokenspeed::test
