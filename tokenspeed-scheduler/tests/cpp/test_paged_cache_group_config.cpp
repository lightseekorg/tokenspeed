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
#include <limits>
#include <stdexcept>

#include "resource/allocator/paged_cache_group.h"

namespace tokenspeed::test {
namespace {

PagedCacheGroupConfig MakeValidGroup() {
    PagedCacheGroupConfig config{};
    config.group_id = "v4.c4.history";
    config.rows_per_page = 64;
    config.entry_stride_tokens = 4;
    config.total_pages = 32;
    return config;
}

TEST(PagedCacheGroupConfigTest, AdditiveMetadataDefaultsPreserveLegacyContract) {
    const PagedCacheGroupConfig config = MakeValidGroup();

    EXPECT_NO_THROW(config.Validate());
    EXPECT_EQ(config.block_size, 0);
    EXPECT_TRUE(config.pool_id.empty());
    EXPECT_EQ(config.prefix_role, PrefixRole::HistoryAnchor);
    EXPECT_EQ(config.table_layout, TableLayout::Absolute);
    EXPECT_EQ(config.required_producer_domain_mask, 0u);
    EXPECT_EQ(config.owner_mask, 0u);
}

TEST(PagedCacheGroupConfigTest, FlatGeometryAcceptsExplicitRawTokenSpanAndMetadata) {
    PagedCacheGroupConfig config = MakeValidGroup();
    config.group_id = "v4.state";
    config.block_size = 256;
    config.pool_id = "v4.state";
    config.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    config.sliding_window_tokens = 256;
    config.family = PagedCacheGroupFamily::State;
    config.prefix_role = PrefixRole::ContinuationState;
    config.table_layout = TableLayout::BoundedWindow;
    config.required_producer_domain_mask = 0b0011;
    config.owner_mask = 0b0011;

    EXPECT_NO_THROW(config.ValidateFlatBlockGeometry());
    EXPECT_EQ(config.RawTokensPerPage(), 256);
    EXPECT_EQ(config.prefix_role, PrefixRole::ContinuationState);
    EXPECT_EQ(config.table_layout, TableLayout::BoundedWindow);
}

TEST(PagedCacheGroupConfigTest, FlatGeometryRejectsLegacyFallbackAndMismatchedSpan) {
    PagedCacheGroupConfig config = MakeValidGroup();

    EXPECT_THROW(config.ValidateFlatBlockGeometry(), std::invalid_argument);
    EXPECT_NO_THROW(config.Validate());

    config.block_size = 64;
    EXPECT_THROW(config.ValidateFlatBlockGeometry(), std::invalid_argument);

    config.block_size = 256;
    EXPECT_NO_THROW(config.ValidateFlatBlockGeometry());
}

TEST(PagedCacheGroupConfigTest, FlatGeometryRejectsRawTokenSpanOverflow) {
    PagedCacheGroupConfig config = MakeValidGroup();
    config.rows_per_page = std::numeric_limits<std::int32_t>::max();
    config.entry_stride_tokens = 2;
    config.block_size = std::numeric_limits<std::int32_t>::max();

    EXPECT_THROW(config.ValidateFlatBlockGeometry(), std::invalid_argument);
}

TEST(PagedCacheGroupConfigTest, ContinuationRequiresBoundedAlignedOwnedProducerContract) {
    PagedCacheGroupConfig config = MakeValidGroup();
    config.block_size = 256;
    config.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    config.sliding_window_tokens = 256;
    config.family = PagedCacheGroupFamily::State;
    config.prefix_role = PrefixRole::ContinuationState;
    config.table_layout = TableLayout::BoundedWindow;
    config.required_producer_domain_mask = 1;
    config.owner_mask = 1;
    EXPECT_NO_THROW(config.ValidateFlatBlockGeometry());

    config.sliding_window_tokens = 128;
    EXPECT_THROW(config.ValidateFlatBlockGeometry(), std::invalid_argument);
    config.sliding_window_tokens = 256;

    config.required_producer_domain_mask = 0;
    EXPECT_THROW(config.ValidateFlatBlockGeometry(), std::invalid_argument);
    config.required_producer_domain_mask = 1;
    config.owner_mask = 0;
    EXPECT_THROW(config.ValidateFlatBlockGeometry(), std::invalid_argument);
    config.owner_mask = 1;

    config.table_layout = TableLayout::Absolute;
    EXPECT_THROW(config.ValidateFlatBlockGeometry(), std::invalid_argument);
}

}  // namespace
}  // namespace tokenspeed::test
