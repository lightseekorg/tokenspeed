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
    EXPECT_TRUE(n768->GetPagedCacheSnapshot()->IsCompleteFor(
        PagedCacheGroupFamily::History));
    EXPECT_FALSE(n768->GetPagedCacheSnapshot()->IsCompleteFor(
        PagedCacheGroupFamily::State));

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

// State checkpoint over a chunk spanning multiple LCM segments must leave the
// table with only the live window (no accumulation of stale ids), and each
// snapshot must own exactly its [D-window, D) slice. raw_per_page=2, window=4,
// LCM=16. A 48-token chunk produces 3 snapshots at D=16,32,48.
TEST(PagedCacheGroupCheckpointTest, ChunkSpansMultiLcmStateGroupHoldsOnlyLiveWindow) {
    PagedCacheGroupConfig cfg{};
    cfg.group_id = "swa_multi";
    cfg.rows_per_page = 2;
    cfg.entry_stride_tokens = 1;  // raw_per_page = 2
    cfg.total_pages = 64;
    cfg.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    cfg.sliding_window_tokens = 4;
    cfg.family = PagedCacheGroupFamily::State;
    auto alloc = std::make_unique<PagedCacheGroupAllocator>(cfg);

    const std::int32_t kLcm = 16;
    const std::int32_t kRawPerPage = 2;
    const std::int32_t kWindow = 4;
    const std::int32_t kWindowPages = kWindow / kRawPerPage;  // 2
    PagedCacheGroupTable table{alloc.get()};

    // Single chunk spans 3 LCM boundaries [16, 32, 48].
    table.Acquire(48);

    std::vector<std::vector<std::int32_t>> snapshot_ids;
    std::vector<std::int32_t> snapshot_base_pages;
    for (std::int32_t target : {16, 32, 48}) {
        auto result = table.CheckpointStateToSnapshot(target);
        snapshot_ids.push_back(result.pages.Ids());
        snapshot_base_pages.push_back(result.segment_base_logical_page);
    }

    // After all three checkpoints, the table sees only the trailing live window.
    EXPECT_EQ(table.Size(), kWindowPages);
    EXPECT_EQ(table.BaseLogicalPage(), (48 - kWindow) / kRawPerPage);  // page 22
    EXPECT_EQ(table.OwnedPagesCount(), 0);
    EXPECT_EQ(table.BorrowedPagesCount(), kWindowPages);
    EXPECT_EQ(table.CommittedPrefixLenTokens(), 48);

    // borrowed_page_ids_ post-final-checkpoint must equal the most recent
    // snapshot's ids (no stale prefix carry-over from earlier snapshots).
    EXPECT_EQ(table.PageIds(), snapshot_ids.back());

    // Each snapshot's segment covers exactly [D-window, D) for its boundary.
    for (std::size_t i = 0; i < snapshot_ids.size(); ++i) {
        const std::int32_t target = (static_cast<std::int32_t>(i) + 1) * kLcm;
        EXPECT_EQ(snapshot_base_pages[i], (target - kWindow) / kRawPerPage);
        EXPECT_EQ(static_cast<std::int32_t>(snapshot_ids[i].size()), kWindowPages);
    }
}

// Fix 1: sliding-window state group at LCM boundary trims dead pages so the
// snapshot only retains the live window. raw_per_page=2, window=4, LCM=16.
TEST(PagedCacheGroupCommitTrimTest, StateSnapshotTrimsDeadPagesAtCommit) {
    PagedCacheGroupConfig cfg{};
    cfg.group_id = "swa_tight";
    cfg.rows_per_page = 2;
    cfg.entry_stride_tokens = 1;  // raw_per_page = 2
    cfg.total_pages = 16;
    cfg.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    cfg.sliding_window_tokens = 4;
    cfg.family = PagedCacheGroupFamily::State;
    auto alloc = std::make_unique<PagedCacheGroupAllocator>(cfg);

    const std::int32_t kLcm = 16;  // raw_per_page=2 -> 8 pages
    const std::int32_t available_before = alloc->AvailablePages();
    PagedCacheGroupTable table{alloc.get()};
    table.Acquire(kLcm);
    EXPECT_EQ(alloc->AvailablePages(), available_before - 8);

    auto result = table.CheckpointStateToSnapshot(kLcm);
    // window/raw_per_page = 4/2 = 2 pages live; remaining 6 pages return to pool.
    EXPECT_EQ(result.pages.Size(), 2);
    EXPECT_EQ(result.segment_base_logical_page, (kLcm - 4) / 2);
    EXPECT_EQ(alloc->AvailablePages(), available_before - 2);
    EXPECT_EQ(table.CommittedPrefixLenTokens(), kLcm);
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
    EXPECT_TRUE(n512->GetPagedCacheSnapshot()->IsCompleteFor(
        PagedCacheGroupFamily::History));
    EXPECT_FALSE(n512->GetPagedCacheSnapshot()->IsCompleteFor(
        PagedCacheGroupFamily::State));
    EXPECT_TRUE(n768->GetPagedCacheSnapshot()->IsCompleteFor(
        PagedCacheGroupFamily::State));

    auto match = hybrid_->Match(MakeAlignedTokens(num_pages, kPageSize, /*start=*/1));
    ASSERT_NE(match.paged_cache.last_node, nullptr);
    // History chain unbroken; state at 768 (only the trailing segment) is fine.
    EXPECT_EQ(match.paged_cache.last_node, n768);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 768);
}

// Fix 2: state-only detach drops state pages but leaves the history snapshot
// (and the history pool's available count) intact; history Match chain stays
// reachable through the same node.
TEST_F(PagedCacheFamilySplitTest, StateOnlyPruneDoesNotBreakHistoryChain) {
    const std::int32_t num_pages = 768 / kPageSize;
    TreeNode* terminal = InsertDevicePages(num_pages, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);

    TreeNode* n256 = kv_cache_->GetRadixTree().SplitAt(terminal, 256);
    TreeNode* n512 = kv_cache_->GetRadixTree().SplitAt(terminal, 512);
    TreeNode* n768 = kv_cache_->GetRadixTree().SplitAt(terminal, 768);
    ASSERT_NE(n256, nullptr);
    ASSERT_NE(n512, nullptr);
    ASSERT_NE(n768, nullptr);

    const std::int32_t swa_before = swa_alloc_->AvailablePages();
    hybrid_->AttachPagedCacheSnapshotToNode(n256, MakeCompleteSnapshot(256));
    hybrid_->AttachPagedCacheSnapshotToNode(n512, MakeCompleteSnapshot(512));
    hybrid_->AttachPagedCacheSnapshotToNode(n768, MakeCompleteSnapshot(768));
    const std::int32_t fh_after_attach = fh_alloc_->AvailablePages();
    const std::int32_t swa_after_attach = swa_alloc_->AvailablePages();
    EXPECT_LT(swa_after_attach, swa_before);

    // State-only detach on the oldest node returns its state pages, leaves
    // history pages intact, and keeps the snapshot present (history-only).
    EXPECT_TRUE(HybridPrefixCacheTestPeer::DetachStateSnapshotFromNode(*hybrid_, n256));
    EXPECT_TRUE(n256->HasPagedCacheSnapshot());
    EXPECT_TRUE(n256->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_FALSE(n256->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::State));
    EXPECT_EQ(fh_alloc_->AvailablePages(), fh_after_attach);
    EXPECT_GT(swa_alloc_->AvailablePages(), swa_after_attach);

    // History chain root->768 stays intact because n512 and n768 keep both
    // families; Match still reaches the deepest snapshot.
    auto match = hybrid_->Match(MakeAlignedTokens(num_pages, kPageSize, /*start=*/1));
    EXPECT_EQ(match.paged_cache.last_node, n768);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 768);
}

}  // namespace tokenspeed::test
