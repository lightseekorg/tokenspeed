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

// Coordinator arithmetic on token positions within one block or prefix page
// of INT32_MAX: rounding a window, its lookback, a tail or a retraction image
// up to whole blocks, and walking a prefix probe's candidate boundaries from
// the end of the page that holds its last token, all pass 2^31 before the
// result is back in range.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/prefix_matcher.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kInt32Max = std::numeric_limits<std::int32_t>::max();

std::vector<std::string> PageHashes(std::int32_t pages) {
    std::vector<std::string> hashes;
    hashes.reserve(static_cast<std::size_t>(pages));
    for (std::int32_t page = 0; page < pages; ++page) {
        hashes.push_back("page-" + std::to_string(page));
    }
    return hashes;
}

TEST(TokenPositionBoundsTest, ASlidingWindowNearInt32MaxEndsAtItsLastBlock) {
    BlockPool pool(4, {1});
    const std::array specs{CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                                          .sliding_window = 4096,
                                          .cache_blocks_per_lcm_block = 1,
                                          .block_granularity = 1024}};
    const auto coordinator = MakeCoordinator(specs, 1024, pool, false, nullptr, nullptr, false, false, 0);
    // T = 2^31 - 1001: rows [T - 4095, T) lie in blocks 2097147 to 2097151.
    const CacheCoordinator::BlockRange window = coordinator.SlidingWindowBlocks(0, kInt32Max - 1000);
    EXPECT_EQ(window.first, 2097147);
    EXPECT_EQ(window.past_end, 2097152);
    // The 23-token tail at T copies into the window's last block.
    const CacheCoordinator::BlockRange copy_in = coordinator.CopyInSlots(0, kInt32Max - 1000);
    EXPECT_EQ(copy_in.first, 2097151);
    EXPECT_EQ(copy_in.past_end, 2097152);
}

// One request spanning INT32_MAX - 1000 tokens of a full group whose blocks
// and prefix pages are 49152 tokens: 43691 blocks, 2^31 + 16384 tokens of
// capacity, with its 43690 full pages published.
class LongestRequestTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kPage = 49152;
    static constexpr std::int32_t kTokens = kInt32Max - 1000;
    static constexpr std::int32_t kBlocks = 43691;
    static constexpr std::int32_t kFullPages = 43690;

    void SetUp() override {
        std::array<BlockTable, 1>& tables = tables_.emplace();
        const std::array demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{kTokens}}};
        const std::optional<CacheCoordinator::AdmissionResult> admitted =
            coordinator_.Admit(coordinator_.ProbePrefix({}), demands, RequestProgress{}, /*copy_in_blocks=*/{},
                               /*copy_in_point=*/std::nullopt, std::nullopt, false, /*session_id=*/{},
                               /*cache_group_id=*/{});
        ASSERT_TRUE(admitted);
        ASSERT_EQ(tables[0].NumBlocks(), kBlocks);
        ASSERT_EQ(tables[0].AvailableTokens(), kBlocks * std::int64_t{kPage} - kTokens);
        coordinator_.CacheFullBlocks(tables, hashes_, admitted->access_epoch, 0, CacheBoundaryKind::kChunk);
    }

    void TearDown() override {
        if (tables_) {
            coordinator_.Free(*tables_);
        }
    }

    static constexpr std::array kSpecs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kPage}};
    BlockPool pool_{kBlocks, {1}};
    CacheCoordinator coordinator_ = MakeCoordinator(kSpecs, kPage, pool_, false, nullptr, nullptr, false, false, 0);
    const std::vector<std::string> hashes_ = PageHashes(kFullPages);
    std::optional<std::array<BlockTable, 1>> tables_;
};

TEST_F(LongestRequestTest, ItsImageCoversEveryBlockUpToItsLastToken) {
    const std::vector<std::vector<ImageSlot>> published = coordinator_.PublishedDataSlots(*tables_, kTokens);
    ASSERT_EQ(published.size(), 1u);
    EXPECT_EQ(published[0].size(), static_cast<std::size_t>(kFullPages)) << "every full page, not its partial tail";
    EXPECT_EQ(coordinator_.PublishedDataSlots(*tables_, kInt32Max)[0].size(), static_cast<std::size_t>(kFullPages))
        << "INT32_MAX itself lies in the last block";
}

TEST_F(LongestRequestTest, AProbeMatchesItsWholePublishedPrefix) {
    coordinator_.Free(*tables_);
    tables_.reset();
    // The page after the hit ends at 2^31 + 16384, past INT32_MAX.
    EXPECT_EQ(coordinator_.ProbePrefix(hashes_).device.num_common_tokens, kFullPages * kPage);
    // One page more would end past INT32_MAX.
    std::vector<std::string> longer = hashes_;
    longer.push_back("page-past-the-limit");
    EXPECT_THROW((void)coordinator_.ProbePrefix(longer), std::runtime_error);
}

TEST(TokenPositionBoundsTest, AnInt32MaxWindowsLookbackRoundsUpToWholeBlocks) {
    // Its lookback, INT32_MAX - 1 tokens, rounds up to 2^31 / 1024 blocks.
    const std::unique_ptr<PrefixMatcher> matcher = MakePrefixMatcher(CacheGroupSpec{
        .attn_kind = AttnKind::kSlidingWindow,
        .sliding_window = kInt32Max,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 1024,
    });
    EXPECT_EQ(matcher->BoundaryLookbackPages(), 2097152);
}

}  // namespace
}  // namespace tokenspeed::test
