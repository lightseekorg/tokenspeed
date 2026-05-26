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

// Coverage: a State-family CheckpointStateToSnapshot following a prefix-cache
// hit on a wide window (window >= LCM) must not throw. Each State snapshot
// stores only its own LCM segment's owned delta; the trailing window is
// reconstructed across the snapshot chain at match time. Regression for the
// "not enough owned pages for window" overflow that incorrectly conflated
// "this commit's delta" with "the whole trailing window".

#include "paged_cache_test_fixture.h"

namespace tokenspeed::test {

using PagedCachePrefixHitCommitTest = PagedCacheWideWindowFixture;

// kSlidingWindow=512, kLcm=256 -> window = 2*LCM. With a 256-token prefix-hit,
// the import covers exactly the first LCM segment as borrowed; the second LCM
// segment must commit through CheckpointStateToSnapshot without trying to
// claim the borrowed half from owned_pages_.
TEST_F(PagedCachePrefixHitCommitTest, PrefixHitFollowedByCheckpointDoesNotOverflowWindow) {
    static_assert(kSlidingWindow >= kLcm, "this test exercises window >= LCM");

    // Seed a chain reaching 512 tokens and attach a complete snapshot at 256
    // so the second request gets a one-LCM-segment prefix-cache hit.
    const std::int32_t num_pages = 512 / kPageSize;  // 8 pages
    TreeNode* terminal = InsertDevicePages(num_pages, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);

    TreeNode* n256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    ASSERT_NE(n256, nullptr);
    HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, n256, MakeCompleteSnapshot(256));
    ASSERT_TRUE(n256->HasPagedCacheSnapshot());

    const auto tokens = MakeAlignedTokens(num_pages, kPageSize, /*start=*/1);

    // The second request: prefix-cache match returns the depth-256 hit.
    auto pre_match = MatchPrefix(*hybrid_, tokens, kPageSize).compat_match;
    ASSERT_NE(pre_match.paged_cache.last_node, nullptr);
    EXPECT_EQ(pre_match.paged_cache.last_node, n256);
    EXPECT_EQ(pre_match.paged_cache.prefix_len_tokens, 256);

    // Import borrowed prefix + acquire fresh pages for the remaining LCM segment.
    const std::string request_id = "r-prefix-hit";
    HybridPrefixCacheTestPeer::AcquireForRequest(*hybrid_, request_id,
                                                 /*first_raw_position_of_op=*/256,
                                                 /*target_raw_tokens_exclusive=*/512, pre_match.paged_cache);

    // Trigger CheckpointStateToSnapshot at the next LCM boundary. Pre-fix this
    // throws std::logic_error("not enough owned pages for window"); post-fix it
    // commits only the new LCM segment's delta to the snapshot.
    ASSERT_NO_THROW(HybridPrefixCacheTestPeer::CommitChunk(*hybrid_, request_id, terminal));

    // After commit, n512 (=terminal) must hold a complete snapshot covering
    // both required families.
    ASSERT_TRUE(terminal->HasPagedCacheSnapshot());
    const auto* committed_snap = terminal->GetPagedCacheSnapshot();
    ASSERT_NE(committed_snap, nullptr);
    EXPECT_TRUE(committed_snap->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_TRUE(committed_snap->IsCompleteFor(PagedCacheGroupFamily::State));

    // Observable: a fresh Match now reconstructs the full trailing window
    // (state_span = [n256, n512]) and exposes window/raw_per_page page ids
    // for the sliding "swa" group.
    auto post_match = MatchPrefix(*hybrid_, tokens, kPageSize).compat_match;
    ASSERT_NE(post_match.paged_cache.last_node, nullptr);
    EXPECT_EQ(post_match.paged_cache.prefix_len_tokens, 512);

    auto swa_it = post_match.paged_cache.per_group_page_ids.find("swa");
    ASSERT_NE(swa_it, post_match.paged_cache.per_group_page_ids.end());
    const auto& swa_ids = swa_it->second;
    ASSERT_FALSE(swa_ids.empty());

    const PagedCacheGroupSnapshot& swa_at_256 = n256->GetPagedCacheSnapshot()->groups.at("swa");
    const std::int32_t raw_per_page = swa_at_256.pages.Size() > 0 ? (kLcm / swa_at_256.pages.Size()) : 0;
    ASSERT_GT(raw_per_page, 0);
    const std::int32_t committed_depth = post_match.paged_cache.prefix_len_tokens;
    const std::int32_t expected_state_pages = std::min(kSlidingWindow / raw_per_page, committed_depth / raw_per_page);
    EXPECT_EQ(static_cast<std::int32_t>(swa_ids.size()), expected_state_pages);

    // Clean up the request tables; owned pages return via RAII / ReleaseAll.
    HybridPrefixCacheTestPeer::ReleaseRequest(*hybrid_, request_id);
}

}  // namespace tokenspeed::test
