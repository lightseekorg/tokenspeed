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

std::vector<std::int32_t> BlockIds(const std::vector<BlockRef>& refs) {
    std::vector<std::int32_t> ids;
    ids.reserve(refs.size());
    for (const BlockRef& ref : refs) {
        ids.push_back(ref ? ref->BlockId() : 0);
    }
    return ids;
}

// The unified Match with a raised floor (begin_blocks > 0 over a non-device pool) is the
// host-tier lookup: slots below the floor are device-valid, holes come back as the queried
// pool's null block.

// Publish a host page for `key` (the scheduler's store path minus the D2H write):
// allocate -> hash -> free leaves it cached-and-evictable, exactly like a committed store.
std::int32_t Put(BlockPool& host_pool, const std::string& key) {
    BlockRef block = host_pool.AcquireBlock();
    const std::int32_t id = block->BlockId();
    host_pool.CacheFullBlock(block, key);
    block.reset();
    return id;
}

TEST(HostTierMatchTest, FullWalksContiguousRunFromBegin) {
    BlockPool pool(4);
    FullAttnManager mgr(/*block_size=*/4);
    EXPECT_TRUE(mgr.MatchIsPrefixClosed());
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3", "k4"};
    std::vector<std::int32_t> put;
    for (std::size_t j = 1; j <= 4; ++j) {
        put.push_back(Put(host_pool, keys[j]));
    }
    // Slots below begin=1 are device-valid; the run covers all extension slots, no holes.
    PrefixMatch m = mgr.Match(host_pool, keys, /*begin_blocks=*/1, /*max_blocks=*/5);
    EXPECT_EQ(BlockIds(m.blocks), put);
    EXPECT_EQ(m.num_hit_blocks, 4);
}

TEST(HostTierMatchTest, FullStopsAtFirstMiss) {
    BlockPool pool(4);
    FullAttnManager mgr(4);
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3"};
    const std::int32_t p0 = Put(host_pool, keys[0]);
    const std::int32_t p1 = Put(host_pool, keys[1]);
    (void)Put(host_pool, keys[3]);  // beyond the gap at k2: unreachable
    EXPECT_EQ(BlockIds(mgr.Match(host_pool, keys, 0, 4).blocks), (std::vector<std::int32_t>{p0, p1}));
}

TEST(HostTierMatchTest, FullEmptyOnBeginMissOrEmptyRange) {
    BlockPool pool(4);
    FullAttnManager mgr(4);
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1"};
    (void)Put(host_pool, keys[1]);
    EXPECT_TRUE(mgr.Match(host_pool, keys, 0, 2).blocks.empty());  // miss right at begin
    EXPECT_TRUE(mgr.Match(host_pool, keys, 2, 2).blocks.empty());  // empty extension range
}

TEST(HostTierMatchTest, SwaTrailingRunAtEnd) {
    BlockPool pool(4);
    // block_size 4, window 10 -> pages_needed = ceil(9/4) = 3.
    SwaManager mgr(4, /*sliding_window=*/10);
    EXPECT_FALSE(mgr.MatchIsPrefixClosed());
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3", "k4"};
    const std::int32_t p2 = Put(host_pool, keys[2]);
    const std::int32_t p3 = Put(host_pool, keys[3]);
    const std::int32_t p4 = Put(host_pool, keys[4]);
    // Trailing run [2, 5) covers the window at boundary 5; slots below stay holes.
    PrefixMatch m = mgr.Match(host_pool, keys, 0, 5);
    EXPECT_EQ(BlockIds(m.blocks), (std::vector<std::int32_t>{0, 0, p2, p3, p4}));
    EXPECT_EQ(m.num_hit_blocks, 3);
}

TEST(HostTierMatchTest, SwaInteriorBoundaryShrink) {
    BlockPool pool(4);
    SwaManager mgr(4, 10);  // pages_needed = 3
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3", "k4"};
    const std::int32_t p1 = Put(host_pool, keys[1]);
    const std::int32_t p2 = Put(host_pool, keys[2]);
    const std::int32_t p3 = Put(host_pool, keys[3]);
    // Miss at 4 invalidates boundary 5; boundary 4 needs [1, 4), which hits.
    EXPECT_EQ(BlockIds(mgr.Match(host_pool, keys, 0, 5).blocks), (std::vector<std::int32_t>{0, p1, p2, p3}));
}

TEST(HostTierMatchTest, SwaShortRunAtBottomSuffices) {
    BlockPool pool(4);
    SwaManager mgr(4, 10);  // pages_needed = 3, but only 2 extension slots exist
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1"};
    const std::int32_t p0 = Put(host_pool, keys[0]);
    const std::int32_t p1 = Put(host_pool, keys[1]);
    // The window clamps to begin: a full 2-run from the bottom is a valid boundary 2.
    EXPECT_EQ(BlockIds(mgr.Match(host_pool, keys, 0, 2).blocks), (std::vector<std::int32_t>{p0, p1}));
}

TEST(HostTierMatchTest, SwaBeginAboveZeroInteriorBoundary) {
    BlockPool pool(4);
    SwaManager mgr(4, /*sliding_window=*/9);  // pages_needed = ceil(8/4) = 2
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3", "k4", "k5", "k6"};
    const std::int32_t p3 = Put(host_pool, keys[3]);
    const std::int32_t p4 = Put(host_pool, keys[4]);
    const std::int32_t p5 = Put(host_pool, keys[5]);
    (void)p3;  // hit at slot 3 sits below the winning run's window and stays a hole
    // Miss at 6 invalidates boundary 7; boundary 6 needs [4, 6), which hits -> vector
    // covers [3, 6): hole at slot 3, pages for 4 and 5.
    PrefixMatch m = mgr.Match(host_pool, keys, /*begin_blocks=*/3, /*max_blocks=*/7);
    EXPECT_EQ(BlockIds(m.blocks), (std::vector<std::int32_t>{0, p4, p5}));
    EXPECT_EQ(m.num_hit_blocks, 2);
}

TEST(HostTierMatchTest, SwaAllMissReturnsEmpty) {
    BlockPool pool(4);
    SwaManager mgr(4, 10);
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2", "k3", "k4"};
    EXPECT_TRUE(mgr.Match(host_pool, keys, 1, 5).blocks.empty());
}

TEST(HostTierMatchTest, SwaZeroNeededWindowAcceptsAllAsHoles) {
    BlockPool pool(4);
    SwaManager mgr(4, /*sliding_window=*/1);  // pages_needed = 0
    BlockPool host_pool(9);
    std::vector<std::string> keys{"k0", "k1", "k2"};
    // Zero needed pages: every boundary is resumable with no host page at all.
    PrefixMatch m = mgr.Match(host_pool, keys, 1, 3);
    EXPECT_EQ(BlockIds(m.blocks), (std::vector<std::int32_t>{0, 0}));
    EXPECT_EQ(m.num_hit_blocks, 0);
}

}  // namespace
}  // namespace tokenspeed::test
