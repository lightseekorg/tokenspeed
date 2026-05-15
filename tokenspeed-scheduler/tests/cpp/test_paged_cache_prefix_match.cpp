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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

// Coverage: HybridPrefixCache::Match paged-cache adjunct branch.

#include "paged_cache_test_fixture.h"

namespace tokenspeed::test {

using PagedCachePrefixMatchTest = PagedCacheLargeFixture;

// 320 tokens: no snapshot caps to root; snapshot at 256 caps to 256.
TEST_F(PagedCachePrefixMatchTest, CapVsNoCap320) {
    const std::int32_t num_pages = 320 / kPageSize;  // 5 pages
    TreeNode* terminal = InsertDevicePages(num_pages, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);
    EXPECT_EQ(terminal->DepthInTokens(), 320u);

    const auto tokens = MakeAlignedTokens(num_pages, kPageSize, /*start=*/1);

    // No snapshot: paged_cache empty; device/host capped to root.
    auto match = hybrid_->Match(tokens);
    EXPECT_EQ(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 0);
    ASSERT_NE(match.device.last_node, nullptr);
    EXPECT_TRUE(match.device.last_node->IsRoot())
        << "device terminal must be capped to root when adjunct is enabled but no snapshot exists";
    ASSERT_NE(match.host.last_node, nullptr);
    EXPECT_TRUE(match.host.last_node->IsRoot())
        << "host terminal must be capped to root when adjunct is enabled but no snapshot exists";

    // A complete paged-cache snapshot at depth 256 caps to 256.
    TreeNode* boundary_256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    ASSERT_NE(boundary_256, nullptr);
    EXPECT_EQ(boundary_256->DepthInTokens(), 256u);
    hybrid_->AttachPagedCacheSnapshotToNode(boundary_256, MakeCompleteSnapshot(256));
    ASSERT_TRUE(boundary_256->HasPagedCacheSnapshot());
    EXPECT_TRUE(boundary_256->GetPagedCacheSnapshot()->IsCompleteFor(
        PagedCacheGroupFamily::History));
    EXPECT_TRUE(boundary_256->GetPagedCacheSnapshot()->IsCompleteFor(
        PagedCacheGroupFamily::State));

    match = hybrid_->Match(tokens);
    ASSERT_NE(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache.last_node, boundary_256);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 256);
    ASSERT_NE(match.device.last_node, nullptr);
    EXPECT_EQ(match.device.last_node->DepthInTokens(), 256u)
        << "device terminal must be capped to the deepest contiguous paged-cache node";
}

// Snapshots at 256/512/768; detaching 512 makes Match fall back to 256.
TEST_F(PagedCachePrefixMatchTest, ContiguousChainBreakMid) {
    const std::int32_t num_pages = 768 / kPageSize;  // 12 pages
    TreeNode* terminal = InsertDevicePages(num_pages, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);

    TreeNode* n256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    TreeNode* n512 = kv_cache_->GetRadixTree().SplitAt(terminal, 512);
    TreeNode* n768 = kv_cache_->GetRadixTree().SplitAt(terminal, 768);
    ASSERT_NE(n256, nullptr);
    ASSERT_NE(n512, nullptr);
    ASSERT_NE(n768, nullptr);

    hybrid_->AttachPagedCacheSnapshotToNode(n256, MakeCompleteSnapshot(256));
    hybrid_->AttachPagedCacheSnapshotToNode(n512, MakeCompleteSnapshot(512));
    hybrid_->AttachPagedCacheSnapshotToNode(n768, MakeCompleteSnapshot(768));

    // Drop the middle snapshot; chain scan must stop at the gap.
    auto dropped = hybrid_->DetachPagedCacheSnapshotFromNode(n512);
    EXPECT_TRUE(dropped != nullptr);
    EXPECT_FALSE(n512->HasPagedCacheSnapshot());
    ASSERT_TRUE(n768->HasPagedCacheSnapshot());
    EXPECT_TRUE(n768->GetPagedCacheSnapshot()->IsCompleteFor(
        PagedCacheGroupFamily::History));

    auto match = hybrid_->Match(MakeAlignedTokens(num_pages, kPageSize, /*start=*/1));
    ASSERT_NE(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache.last_node, n256);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 256);
}

}  // namespace tokenspeed::test
