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
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "resource/allocator/page_allocator.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kPageSize = 4;
constexpr std::int32_t kMambaChunkSize = 8;

PagedCacheGroupConfig MakePagedGroup(std::string group_id, PagedCacheGroupFamily family,
                                     PagedCacheGroupConfig::Retention retention,
                                     std::optional<std::int32_t> sliding_window_tokens = std::nullopt) {
    return PagedCacheGroupConfig{
        .group_id = std::move(group_id),
        .rows_per_page = 4,
        .entry_stride_tokens = 2,
        .total_pages = 16,
        .retention = retention,
        .sliding_window_tokens = sliding_window_tokens,
        .family = family,
    };
}

}  // namespace

TEST(HybridPrefixCacheStatsTest, PagedGroupsReportPublicStatsAndMissingRequestState) {
    PageAllocator device_allocator{kPageSize, /*total_pages=*/8};
    PageAllocator host_allocator{kPageSize, /*total_pages=*/0};
    KVPrefixCache prefix_cache{&device_allocator, &host_allocator};
    HybridPrefixCache hybrid_prefix_cache{prefix_cache, device_allocator, /*allocator=*/nullptr, kMambaChunkSize};
    const std::vector<PagedCacheGroupConfig> groups = {
        MakePagedGroup("v4.history", PagedCacheGroupFamily::History, PagedCacheGroupConfig::Retention::FullHistory),
        MakePagedGroup("v4.swa", PagedCacheGroupFamily::State, PagedCacheGroupConfig::Retention::SlidingWindow,
                       /*sliding_window_tokens=*/16),
    };

    hybrid_prefix_cache.ConfigurePagedCacheAdjunct(std::span<const PagedCacheGroupConfig>{groups}, std::nullopt);

    const CacheStatsSnapshot all_stats = hybrid_prefix_cache.Stats();
    EXPECT_EQ(all_stats.available_device_pages, 7u);
    EXPECT_EQ(all_stats.paged_cache_group_ids, std::vector<std::string>({"v4.history", "v4.swa"}));

    const CacheStatsSnapshot history_stats =
        hybrid_prefix_cache.Stats({.request_id = "missing", .paged_cache_group_ids = {"v4.history"}});
    EXPECT_EQ(history_stats.paged_cache_total_pages.at("v4.history"), 16);
    EXPECT_EQ(history_stats.paged_cache_available_pages.at("v4.history"), 15);
    EXPECT_EQ(history_stats.paged_cache_failed_alloc_count.at("v4.history"), 0);
    EXPECT_TRUE(history_stats.request_paged_cache_page_ids.at("v4.history").empty());
    EXPECT_EQ(history_stats.request_paged_cache_base_logical_page.at("v4.history"), 0);

    EXPECT_THROW((void)hybrid_prefix_cache.Stats({.paged_cache_group_ids = {"missing"}}), std::out_of_range);
}

}  // namespace tokenspeed::test
