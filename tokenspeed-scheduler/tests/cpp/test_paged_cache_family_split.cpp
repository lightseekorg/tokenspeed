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

// Coverage: V4 family-split prefix match. History (chain) and State
// (trailing window) families are now scanned independently; a missing
// State snapshot falls back to a shallower depth without killing the
// History chain.

#include "hybrid_prefix_cache_test_peer.h"
#include "paged_cache_test_fixture.h"

namespace tokenspeed::test {

using PagedCacheFamilySplitTest = PagedCacheLargeFixture;
using PagedCacheFamilyWideWindowTest = PagedCacheWideWindowFixture;

// kSlidingWindow=128 < kLcm=256 -> segments_needed=1.
// Dropping state-completeness at the deepest boundary falls back one segment.
TEST_F(PagedCacheFamilySplitTest, HistoryCompleteStateMissingFallback) {
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

    // Downgrade only the deepest snapshot: history-only at 768.
    DowngradeSnapshotToHistoryOnly(n768);
    ASSERT_TRUE(n768->HasPagedCacheSnapshot());
    EXPECT_TRUE(n768->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_FALSE(n768->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::State));

    auto match = hybrid_->Match(MakeAlignedTokens(num_pages, kPageSize, /*start=*/1));
    ASSERT_NE(match.paged_cache.last_node, nullptr);
    // History chain reaches 768 but state at 768 is missing; segments_needed=1
    // forces fallback to 512.
    EXPECT_EQ(match.paged_cache.last_node, n512);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 512);
}

// segments_needed=2 (window=512, align=256). State missing at 512 breaks
// both end_idx=2 (trailing 512+768) and end_idx=1 (trailing 256+512); only
// end_idx=0 (single segment 256) remains.
TEST_F(PagedCacheFamilyWideWindowTest, StateWindowDiscontinuityFallback) {
    const std::int32_t num_pages = 768 / kPageSize;
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

    DowngradeSnapshotToHistoryOnly(n512);

    auto match = hybrid_->Match(MakeAlignedTokens(num_pages, kPageSize, /*start=*/1));
    ASSERT_NE(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache.last_node, n256);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 256);
}

// segments_needed=1: detaching state at mid-chain does not break the history
// chain; deepest state-complete boundary (768) remains usable.
TEST_F(PagedCacheFamilySplitTest, StateDetachDoesNotBreakHistoryChain) {
    const std::int32_t num_pages = 768 / kPageSize;
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

    DowngradeSnapshotToHistoryOnly(n512);
    ASSERT_TRUE(n512->HasPagedCacheSnapshot());
    EXPECT_TRUE(n512->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_FALSE(n512->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::State));
    EXPECT_TRUE(n768->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::State));

    auto match = hybrid_->Match(MakeAlignedTokens(num_pages, kPageSize, /*start=*/1));
    ASSERT_NE(match.paged_cache.last_node, nullptr);
    // History chain unbroken; state at 768 (only the trailing segment) is fine.
    EXPECT_EQ(match.paged_cache.last_node, n768);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 768);
}

}  // namespace tokenspeed::test
