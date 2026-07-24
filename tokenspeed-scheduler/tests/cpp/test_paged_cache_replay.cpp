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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include "integration_test_helper.h"
#include "paged_cache_test_fixture.h"

namespace tokenspeed::test {
namespace {

class PagedCacheTerminalSchedulerTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.block_size = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;

        PagedCacheGroupConfig fh{};
        fh.group_id = "fh";
        fh.rows_per_page = 4;
        fh.entry_stride_tokens = 1;
        fh.total_pages = 32;
        fh.retention = PagedCacheGroupConfig::Retention::FullHistory;
        fh.family = PagedCacheGroupFamily::History;
        cfg.paged_cache_groups.push_back(fh);

        PagedCacheGroupConfig swa{};
        swa.group_id = "swa";
        swa.rows_per_page = 2;
        swa.entry_stride_tokens = 1;
        swa.total_pages = 32;
        swa.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
        swa.sliding_window_tokens = 8;
        swa.family = PagedCacheGroupFamily::State;
        cfg.paged_cache_groups.push_back(swa);

        PrefixCacheAdjunctSpec spec{};
        spec.required_groups = {"fh"};
        cfg.prefix_cache_adjunct = spec;
        return cfg;
    }
};

class PagedCacheTerminalMixedSchedulerTest : public PagedCacheTerminalSchedulerTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PagedCacheTerminalSchedulerTest::MakeConfig();
        cfg.device_allocator.total_pages = 256;
        cfg.max_scheduled_tokens = 128;
        cfg.enable_mixed_prefill_decode = true;
        for (auto& group : cfg.paged_cache_groups) {
            group.total_pages = 256;
        }
        return cfg;
    }
};

class PagedCacheDecodePublishTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.block_size = 1;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.decode_input_tokens = 4;
        cfg.enable_l3_storage = false;

        PagedCacheGroupConfig history{};
        history.group_id = "fh";
        history.rows_per_page = 2;
        history.entry_stride_tokens = 1;
        history.total_pages = 64;
        history.retention = PagedCacheGroupConfig::Retention::FullHistory;
        history.family = PagedCacheGroupFamily::History;
        cfg.paged_cache_groups.push_back(history);

        PrefixCacheAdjunctSpec spec{};
        spec.required_groups = {"fh"};
        cfg.prefix_cache_adjunct = spec;
        return cfg;
    }
};

class PagedCacheOverlapSchedulerTest
    : public SchedulerTestSuite,
      public ::testing::WithParamInterface<std::tuple<std::int32_t, std::int32_t, std::int32_t>> {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.block_size = 64;
        cfg.device_allocator.total_pages = 256;
        cfg.host_allocator.total_pages = 256;
        cfg.max_scheduled_tokens = 256;
        cfg.max_batch_size = 4;
        cfg.decode_input_tokens = std::get<0>(GetParam());
        cfg.overlap_schedule_depth = std::get<2>(GetParam());
        cfg.disable_l2_cache = true;

        PagedCacheGroupConfig history{};
        history.group_id = "overlap.history";
        history.rows_per_page = 1;
        history.entry_stride_tokens = 1;
        history.total_pages = 256;
        history.retention = PagedCacheGroupConfig::Retention::FullHistory;
        history.family = PagedCacheGroupFamily::History;
        cfg.paged_cache_groups.push_back(history);

        PagedCacheGroupConfig state{};
        state.group_id = "overlap.state";
        state.rows_per_page = 1;
        state.entry_stride_tokens = 1;
        state.total_pages = 64;
        state.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
        state.sliding_window_tokens = 8;
        state.family = PagedCacheGroupFamily::State;
        cfg.paged_cache_groups.push_back(state);
        return cfg;
    }
};

class PagedCacheOverlapRetractTest
    : public SchedulerTestSuite,
      public ::testing::WithParamInterface<std::tuple<std::int32_t, std::int32_t, bool>> {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.block_size = 2;
        cfg.device_allocator.total_pages = 10;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 4;
        cfg.decode_input_tokens = 4;
        cfg.overlap_schedule_depth = std::get<0>(GetParam());
        cfg.enable_l3_storage = false;

        PagedCacheGroupConfig history{};
        history.group_id = "retract.history";
        history.rows_per_page = 2;
        history.entry_stride_tokens = 1;
        history.total_pages = 128;
        history.retention = PagedCacheGroupConfig::Retention::FullHistory;
        history.family = PagedCacheGroupFamily::History;
        cfg.paged_cache_groups.push_back(history);

        PagedCacheGroupConfig state{};
        state.group_id = "retract.state";
        state.rows_per_page = 1;
        state.entry_stride_tokens = 1;
        state.total_pages = 64;
        state.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
        state.sliding_window_tokens = 8;
        state.family = PagedCacheGroupFamily::State;
        cfg.paged_cache_groups.push_back(state);

        PrefixCacheAdjunctSpec adjunct{};
        adjunct.required_groups = {"retract.history"};
        cfg.prefix_cache_adjunct = adjunct;
        return cfg;
    }

    void SendReserveNumTokens(std::int32_t value) {
        ExecutionEvent event;
        event.With(ForwardEvent{forward::UpdateReserveNumTokens{
            .request_id = "r",
            .reserve_num_tokens_in_next_schedule_event = value,
        }});
        scheduler_->Advance(std::move(event));
    }
};

class PagedCacheTerminalContinuationTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kPageSize = 64;
    static constexpr std::int32_t kDevicePages = 64;
    static constexpr std::int32_t kRequiredStateRows = 4;
    static constexpr std::int32_t kRequiredStateWindow = 8;
    static constexpr std::int32_t kWindowStateTokens = 128;
    static constexpr const char* kHistoryGroup = "history";
    static constexpr const char* kRequiredStateGroup = "required_state";
    static constexpr const char* kWindowStateGroup = "window_state";

    void SetUp() override {
        device_alloc_ = std::make_unique<PageAllocator>(kPageSize, kDevicePages);
        kv_cache_ = std::make_unique<KVPrefixCache>(device_alloc_.get(), /*host=*/nullptr);

        auto history_owner = std::make_unique<PagedCacheGroupAllocator>(
            MakeGroup(kHistoryGroup, /*rows_per_page=*/64, /*stride=*/1, PagedCacheGroupConfig::Retention::FullHistory,
                      std::nullopt, PagedCacheGroupFamily::History,
                      /*total_pages=*/64));
        auto required_state_owner = std::make_unique<PagedCacheGroupAllocator>(MakeGroup(
            kRequiredStateGroup, kRequiredStateRows, /*stride=*/1, PagedCacheGroupConfig::Retention::SlidingWindow,
            kRequiredStateWindow, PagedCacheGroupFamily::State,
            /*total_pages=*/128));
        // Page 0 is reserved; the longest borrower/HostRef chain holds eight
        // transport-only window pages concurrently.
        auto window_owner = std::make_unique<PagedCacheGroupAllocator>(MakeGroup(
            kWindowStateGroup, /*rows_per_page=*/64, /*stride=*/1, PagedCacheGroupConfig::Retention::SlidingWindow,
            kWindowStateTokens, PagedCacheGroupFamily::State, /*total_pages=*/9));

        hybrid_ = std::make_unique<HybridPrefixCache>(*kv_cache_, /*mamba=*/nullptr,
                                                      /*mamba_chunk_size=*/0);
        hybrid_->RegisterPagedCacheGroup(std::move(history_owner));
        hybrid_->RegisterPagedCacheGroup(std::move(required_state_owner));
        hybrid_->RegisterPagedCacheGroup(std::move(window_owner));
        hybrid_->EnablePagedCacheAdjunct({kHistoryGroup, kRequiredStateGroup},
                                         {{kRequiredStateGroup, kRequiredStateWindow}});
        kv_cache_->GetDeviceManager().SetEvictionCallback([this](TreeNode* node) { hybrid_->OnKVEvict(node); });
    }

    TreeNode* InsertDeviceTokens(std::int32_t raw_tokens, token_t token_start = 1) {
        const std::int32_t num_pages = raw_tokens / kPageSize;
        auto tokens = MakeAlignedTokens(num_pages, kPageSize, token_start);
        OwnedPages pages = device_alloc_->Allocate(num_pages);
        auto res = kv_cache_->Insert<ResourceType::Device>(tokens, /*prefix_pages=*/{}, std::move(pages),
                                                           /*page_hashes=*/{}, /*start_node=*/nullptr);
        return res.last_node;
    }

    MatchResult MatchTokens(std::int32_t raw_tokens, token_t token_start = 1) {
        return hybrid_->Match(MakeAlignedTokens(raw_tokens / kPageSize, kPageSize, token_start));
    }

    void CommitRequest(const std::string& request_id, std::int32_t first_token, std::int32_t target, TreeNode* terminal,
                       const MatchResult::PagedCache& hit = {}) {
        hybrid_->AcquireForRequest(request_id, first_token, target, hit);
        hybrid_->CommitChunk(request_id, terminal);
    }

    static PagedCacheGroupConfig MakeGroup(std::string group_id, std::int32_t rows_per_page, std::int32_t stride,
                                           PagedCacheGroupConfig::Retention retention,
                                           std::optional<std::int32_t> window, PagedCacheGroupFamily family,
                                           std::int32_t total_pages) {
        PagedCacheGroupConfig cfg{};
        cfg.group_id = std::move(group_id);
        cfg.rows_per_page = rows_per_page;
        cfg.entry_stride_tokens = stride;
        cfg.total_pages = total_pages;
        cfg.retention = retention;
        cfg.sliding_window_tokens = window;
        cfg.family = family;
        return cfg;
    }

    std::unique_ptr<PageAllocator> device_alloc_;
    std::unique_ptr<KVPrefixCache> kv_cache_;
    std::unique_ptr<HybridPrefixCache> hybrid_;
};

const PagedCacheGroupConfig& FindGroupConfig(const SchedulerConfig& cfg, const std::string& gid) {
    for (const auto& group : cfg.paged_cache_groups) {
        if (group.group_id == gid) return group;
    }
    throw std::logic_error("test group config missing: " + gid);
}

void ExpectPagedGroupCoversRange(const FlatForwardOperation& fwd, const SchedulerConfig& cfg, const std::string& gid,
                                 std::size_t row, std::int32_t first_token, std::int32_t token_count) {
    const auto& group_cfg = FindGroupConfig(cfg, gid);
    const std::int32_t raw_per_page = group_cfg.RawTokensPerPage();
    ASSERT_GT(raw_per_page, 0);

    auto table_it = fwd.paged_cache_block_tables.find(gid);
    ASSERT_NE(table_it, fwd.paged_cache_block_tables.end()) << "group=" << gid;
    ASSERT_LT(row, table_it->second.size()) << "group=" << gid;
    const auto& pages = table_it->second[row];

    std::int32_t base_logical_page = 0;
    auto base_map_it = fwd.paged_cache_block_table_base_offsets.find(gid);
    if (base_map_it != fwd.paged_cache_block_table_base_offsets.end()) {
        ASSERT_LT(row, base_map_it->second.size()) << "group=" << gid;
        base_logical_page = base_map_it->second[row];
    }

    for (std::int32_t pos = first_token; pos < first_token + token_count; ++pos) {
        const std::int32_t logical_page = pos / raw_per_page;
        const std::int32_t table_page = logical_page - base_logical_page;
        ASSERT_GE(table_page, 0) << "group=" << gid << " row=" << row << " pos=" << pos;
        ASSERT_LT(table_page, static_cast<std::int32_t>(pages.size()))
            << "group=" << gid << " row=" << row << " pos=" << pos;
        const std::int32_t physical_page = pages[static_cast<std::size_t>(table_page)];
        EXPECT_GT(physical_page, 0) << "group=" << gid << " row=" << row << " pos=" << pos;
        EXPECT_LT(physical_page, group_cfg.total_pages) << "group=" << gid << " row=" << row << " pos=" << pos;
    }
}

}  // namespace

TEST_F(PagedCacheTerminalContinuationTest, ExactTerminalHitUsesContinuationStateWithoutReplay) {
    TreeNode* n256 = InsertDeviceTokens(256);
    ASSERT_NE(n256, nullptr);
    CommitRequest("r1", /*first_token=*/0, /*target=*/256, n256);
    hybrid_->ReleaseRequest("r1");

    ASSERT_TRUE(n256->HasPagedCacheSnapshot());
    ASSERT_TRUE(n256->GetPagedCacheSnapshot()->continuation_state_complete);

    auto first_match = MatchTokens(256);
    EXPECT_EQ(first_match.paged_cache.history_hit_tokens, 256);
    EXPECT_EQ(first_match.paged_cache.prefix_len_tokens, 256);
    EXPECT_EQ(first_match.paged_cache.last_node, n256);
    EXPECT_EQ(first_match.paged_cache.per_group_base_logical_page.at(kWindowStateGroup), 2);
    EXPECT_EQ(first_match.paged_cache.per_group_page_ids.at(kWindowStateGroup).size(), 2u);
    EXPECT_EQ(first_match.paged_cache.per_group_base_logical_page.at(kRequiredStateGroup), 62);
    EXPECT_EQ(first_match.paged_cache.per_group_page_ids.at(kRequiredStateGroup).size(), 2u);

    TreeNode* n320 = InsertDeviceTokens(320);
    ASSERT_NE(n320, nullptr);
    CommitRequest("r2", /*first_token=*/256, /*target=*/320, n320, first_match.paged_cache);

    ASSERT_TRUE(n320->HasPagedCacheSnapshot());
    ASSERT_TRUE(n320->GetPagedCacheSnapshot()->continuation_state_complete);
    const auto& n320_window = n320->GetPagedCacheSnapshot()->groups.at(kWindowStateGroup);
    EXPECT_EQ(n320_window.base_logical_page, 4);
    EXPECT_EQ(n320_window.pages.Size(), 1);

    auto second_match = MatchTokens(320);
    EXPECT_EQ(second_match.paged_cache.history_hit_tokens, 320);
    EXPECT_EQ(second_match.paged_cache.prefix_len_tokens, 320);
    EXPECT_EQ(second_match.paged_cache.last_node, n320);
    EXPECT_EQ(second_match.paged_cache.per_group_base_logical_page.at(kWindowStateGroup), 3);
    EXPECT_EQ(second_match.paged_cache.per_group_page_ids.at(kWindowStateGroup).size(), 2u);
    EXPECT_EQ(second_match.paged_cache.per_group_base_logical_page.at(kRequiredStateGroup), 78);
    EXPECT_EQ(second_match.paged_cache.per_group_page_ids.at(kRequiredStateGroup).size(), 2u);
}

TEST_F(PagedCacheTerminalContinuationTest, StateRecoveryDoesNotUseEarlierContinuationCheckpoint) {
    TreeNode* checkpoint = InsertDeviceTokens(256);
    ASSERT_NE(checkpoint, nullptr);
    CommitRequest("seed", /*first_token=*/0, /*target=*/256, checkpoint);
    hybrid_->ReleaseRequest("seed");

    TreeNode* terminal = InsertDeviceTokens(320);
    ASSERT_NE(terminal, nullptr);
    ASSERT_FALSE(terminal->HasPagedCacheSnapshot());

    auto tokens = MakeAlignedTokens(/*num_pages=*/5, kPageSize, /*start=*/1);
    auto match = hybrid_->Match(tokens, MatchIntent::StateRecovery);

    EXPECT_EQ(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache_host.last_node, nullptr);
    ASSERT_NE(match.device.last_node, nullptr);
    EXPECT_TRUE(match.device.last_node->IsRoot());
}

TEST_F(PagedCacheTerminalContinuationTest, SnapshotCursorStopsAtCheckpointBeforeReservedTail) {
    constexpr std::int32_t kCheckpoint = 256;
    constexpr std::int32_t kVerifyWidth = 4;
    TreeNode* terminal = InsertDeviceTokens(kCheckpoint);
    ASSERT_NE(terminal, nullptr);

    hybrid_->AcquireForRequest("r", /*first_raw_position_of_op=*/0,
                               /*target_raw_tokens_exclusive=*/kCheckpoint + 2 * kVerifyWidth);
    hybrid_->CommitChunk("r", terminal);

    ASSERT_TRUE(terminal->HasPagedCacheSnapshot());
    const auto* snapshot = terminal->GetPagedCacheSnapshot();
    ASSERT_NE(snapshot, nullptr);
    EXPECT_EQ(snapshot->prefix_len_tokens, kCheckpoint);
    ASSERT_TRUE(snapshot->continuation_state_complete);
    for (const auto& [group_id, group] : snapshot->groups) {
        EXPECT_EQ(group.raw_token_cursor, kCheckpoint) << "group=" << group_id;
    }

    // The request still owns the physical lookahead page, but the snapshot
    // advertises only the accepted checkpoint.
    EXPECT_EQ(hybrid_->GetRequestPagedCachePageIds("r", kHistoryGroup).size(), 5u);
}

TEST_F(PagedCacheTerminalContinuationTest, RewindRejectsInvalidProtectedTail) {
    EXPECT_THROW(hybrid_->RewindRequest("missing", /*accepted_raw_tokens=*/0,
                                        /*protected_tail_tokens=*/-1),
                 std::invalid_argument);
    EXPECT_THROW(hybrid_->RewindRequest("missing", std::numeric_limits<std::int32_t>::max(),
                                        /*protected_tail_tokens=*/1),
                 std::overflow_error);
}

TEST_F(PagedCacheTerminalContinuationTest, TableBorrowPinsStateButHostRefDoesNot) {
    TreeNode* n256 = InsertDeviceTokens(256);
    ASSERT_NE(n256, nullptr);
    CommitRequest("seed", /*first_token=*/0, /*target=*/256, n256);
    hybrid_->ReleaseRequest("seed");
    auto hit256 = MatchTokens(256);
    ASSERT_EQ(hit256.paged_cache.prefix_len_tokens, 256);

    // A retracted request can keep borrowing n256 after its DeviceNodeRef is
    // gone. That table, not a coarse host-ref count, must pin the State pages.
    hybrid_->AcquireForRequest("borrower", /*first_raw_position_of_op=*/0,
                               /*target_raw_tokens_exclusive=*/256, hit256.paged_cache);

    TreeNode* n512 = InsertDeviceTokens(512);
    ASSERT_NE(n512, nullptr);
    {
        DeviceNodeRef writer_ref{n512};
        CommitRequest("writer", /*first_token=*/256, /*target=*/512, n512, hit256.paged_cache);
    }
    ASSERT_TRUE(n256->HasPagedCacheSnapshot());
    EXPECT_TRUE(n256->GetPagedCacheSnapshot()->continuation_state_complete);

    hybrid_->ReleaseRequest("borrower");

    // Model the writer's own recovery HostNodeRef. The former HostRef-based
    // heuristic treated this as another sharer and leaked old State snapshots.
    PageAllocator host_alloc{kPageSize, /*total_pages=*/64};
    std::vector<TreeNode*> attached_host_nodes;
    for (TreeNode* cur = n256; cur != nullptr && !cur->IsRoot(); cur = cur->Parent()) {
        if (cur->OnHost()) continue;
        const auto host_pages = static_cast<std::int32_t>(cur->Tokens().size()) / kPageSize;
        cur->AttachResource(std::make_unique<NodeResource<ResourceType::Host>>(host_alloc.Allocate(host_pages)));
        attached_host_nodes.push_back(cur);
    }
    auto writer_host_ref = std::make_unique<HostNodeRef>(n256);

    TreeNode* n768 = InsertDeviceTokens(768);
    ASSERT_NE(n768, nullptr);
    {
        DeviceNodeRef writer_ref{n768};
        CommitRequest("writer", /*first_token=*/512, /*target=*/768, n768);
    }

    ASSERT_TRUE(n256->HasPagedCacheSnapshot());
    EXPECT_FALSE(n256->GetPagedCacheSnapshot()->continuation_state_complete);
    EXPECT_EQ(n256->GetPagedCacheSnapshot()->groups.find(kRequiredStateGroup),
              n256->GetPagedCacheSnapshot()->groups.end());
    EXPECT_EQ(n256->GetPagedCacheSnapshot()->groups.find(kWindowStateGroup),
              n256->GetPagedCacheSnapshot()->groups.end());

    hybrid_->ReleaseRequest("writer");
    writer_host_ref.reset();
    for (TreeNode* node : attached_host_nodes) {
        auto host_resource = node->DetachResource<ResourceType::Host>();
        ASSERT_NE(host_resource, nullptr);
    }
}

TEST_F(PagedCacheTerminalContinuationTest, CurrentBorrowPinsAncestorWhenTerminalSnapshotAlreadyExists) {
    TreeNode* n64 = InsertDeviceTokens(64);
    ASSERT_NE(n64, nullptr);
    CommitRequest("seed", /*first_token=*/0, /*target=*/64, n64);
    hybrid_->ReleaseRequest("seed");

    auto hit64 = MatchTokens(64);
    ASSERT_EQ(hit64.paged_cache.prefix_len_tokens, 64);

    // Keep n64's transport-only State page borrowed by the request that will
    // later adopt an already-existing terminal snapshot at n192.
    hybrid_->AcquireForRequest("current", /*first_raw_position_of_op=*/64,
                               /*target_raw_tokens_exclusive=*/192, hit64.paged_cache);

    TreeNode* n192 = InsertDeviceTokens(192);
    ASSERT_NE(n192, nullptr);
    {
        DeviceNodeRef writer_ref{n192};
        CommitRequest("writer", /*first_token=*/64, /*target=*/192, n192, hit64.paged_cache);
    }
    hybrid_->ReleaseRequest("writer");

    ASSERT_TRUE(n64->HasPagedCacheSnapshot());
    ASSERT_TRUE(n192->HasPagedCacheSnapshot());
    ASSERT_TRUE(n192->GetPagedCacheSnapshot()->continuation_state_complete);
    const auto& ancestor_state_pages = n64->GetPagedCacheSnapshot()->groups.at(kWindowStateGroup).pages.Ids();
    const auto& current_state_pages = hybrid_->GetRequestPagedCachePageIds("current", kWindowStateGroup);
    ASSERT_TRUE(std::any_of(ancestor_state_pages.begin(), ancestor_state_pages.end(), [&](std::int32_t page_id) {
        return std::find(current_state_pages.begin(), current_state_pages.end(), page_id) != current_state_pages.end();
    }));

    // Required groups adopt n192's existing snapshot, while the
    // transport-only group is already present and therefore is not adopted.
    // The current request still borrows n64, so ancestor cleanup must retain
    // that State snapshot.
    {
        DeviceNodeRef current_ref{n192};
        hybrid_->CommitChunk("current", n192);
    }

    ASSERT_TRUE(n64->HasPagedCacheSnapshot());
    EXPECT_TRUE(n64->GetPagedCacheSnapshot()->continuation_state_complete);
    EXPECT_NE(n64->GetPagedCacheSnapshot()->groups.find(kWindowStateGroup), n64->GetPagedCacheSnapshot()->groups.end());

    hybrid_->ReleaseRequest("current");
}

TEST_F(PagedCacheTerminalContinuationTest, StatePruneDropsContinuationAndDisablesPrefixReuse) {
    TreeNode* n256 = InsertDeviceTokens(256);
    ASSERT_NE(n256, nullptr);
    CommitRequest("r1", /*first_token=*/0, /*target=*/256, n256);
    hybrid_->ReleaseRequest("r1");
    ASSERT_TRUE(n256->HasPagedCacheSnapshot());
    ASSERT_TRUE(n256->GetPagedCacheSnapshot()->continuation_state_complete);

    auto simulated_free = hybrid_->InitialSimulatedFree();
    simulated_free[kWindowStateGroup] = 0;
    ASSERT_TRUE(hybrid_->AdmitChunk("pressure", /*first_raw_position_of_op=*/0,
                                    /*target_raw_tokens_exclusive=*/64, simulated_free));

    ASSERT_TRUE(n256->HasPagedCacheSnapshot());
    ASSERT_FALSE(n256->GetPagedCacheSnapshot()->continuation_state_complete);
    EXPECT_EQ(n256->GetPagedCacheSnapshot()->groups.find(kWindowStateGroup),
              n256->GetPagedCacheSnapshot()->groups.end());

    auto match = MatchTokens(256);
    EXPECT_EQ(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 0);
    ASSERT_NE(match.device.last_node, nullptr);
    EXPECT_TRUE(match.device.last_node->IsRoot());
}

TEST_F(PagedCacheTerminalContinuationTest, StateOnlyPruneIgnoresHistoryOnlySideTableBorrow) {
    TreeNode* n64 = InsertDeviceTokens(64);
    ASSERT_NE(n64, nullptr);
    CommitRequest("seed", /*first_token=*/0, /*target=*/64, n64);
    hybrid_->ReleaseRequest("seed");

    auto hit = MatchTokens(64);
    ASSERT_EQ(hit.paged_cache.prefix_len_tokens, 64);
    hybrid_->AcquireForRequest("borrower", /*first_raw_position_of_op=*/192,
                               /*target_raw_tokens_exclusive=*/192, hit.paged_cache);

    auto shares_snapshot_pages = [&](const std::string& group_id) {
        const auto& snapshot_pages = n64->GetPagedCacheSnapshot()->groups.at(group_id).pages.Ids();
        const auto& request_pages = hybrid_->GetRequestPagedCachePageIds("borrower", group_id);
        return std::any_of(snapshot_pages.begin(), snapshot_pages.end(), [&](std::int32_t page_id) {
            return std::find(request_pages.begin(), request_pages.end(), page_id) != request_pages.end();
        });
    };

    ASSERT_TRUE(n64->HasPagedCacheSnapshot());
    EXPECT_TRUE(shares_snapshot_pages(kHistoryGroup));
    EXPECT_FALSE(shares_snapshot_pages(kRequiredStateGroup));
    EXPECT_FALSE(shares_snapshot_pages(kWindowStateGroup));
    const auto history_pages_before = hybrid_->GetRequestPagedCachePageIds("borrower", kHistoryGroup);

    auto simulated_free = hybrid_->InitialSimulatedFree();
    simulated_free[kWindowStateGroup] = 0;
    ASSERT_TRUE(hybrid_->AdmitChunk("pressure", /*first_raw_position_of_op=*/0,
                                    /*target_raw_tokens_exclusive=*/64, simulated_free));

    ASSERT_TRUE(n64->HasPagedCacheSnapshot());
    const auto* snapshot = n64->GetPagedCacheSnapshot();
    ASSERT_NE(snapshot, nullptr);
    EXPECT_NE(snapshot->groups.find(kHistoryGroup), snapshot->groups.end());
    EXPECT_EQ(snapshot->groups.find(kRequiredStateGroup), snapshot->groups.end());
    EXPECT_EQ(snapshot->groups.find(kWindowStateGroup), snapshot->groups.end());
    EXPECT_FALSE(snapshot->continuation_state_complete);
    EXPECT_EQ(hybrid_->GetRequestPagedCachePageIds("borrower", kHistoryGroup), history_pages_before);

    hybrid_->ReleaseRequest("borrower");
}

TEST(PagedCacheHistoryOnlyTest, HistoryOnlyPrefixHitRemainsUsable) {
    auto device_alloc = std::make_unique<PageAllocator>(64, 64);
    auto kv_cache = std::make_unique<KVPrefixCache>(device_alloc.get(), /*host_allocator=*/nullptr);

    PagedCacheGroupConfig history{};
    history.group_id = "fh";
    history.rows_per_page = 64;
    history.entry_stride_tokens = 1;
    history.total_pages = 16;
    history.retention = PagedCacheGroupConfig::Retention::FullHistory;
    history.family = PagedCacheGroupFamily::History;

    auto history_owner = std::make_unique<PagedCacheGroupAllocator>(history);
    HybridPrefixCache hybrid(*kv_cache, /*mamba=*/nullptr, /*mamba_chunk_size=*/0);
    hybrid.RegisterPagedCacheGroup(std::move(history_owner));
    hybrid.EnablePagedCacheAdjunct({"fh"}, {});

    auto tokens = MakeAlignedTokens(/*num_pages=*/4, /*page_size=*/64, /*start=*/1);
    OwnedPages pages = device_alloc->Allocate(4);
    auto inserted = kv_cache->Insert<ResourceType::Device>(tokens, /*prefix_pages=*/{}, std::move(pages),
                                                           /*page_hashes=*/{}, /*start_node=*/nullptr);
    ASSERT_NE(inserted.last_node, nullptr);

    hybrid.AcquireForRequest("r1", /*first_raw_position_of_op=*/0, /*target_raw_tokens_exclusive=*/256);
    hybrid.CommitChunk("r1", inserted.last_node);
    hybrid.ReleaseRequest("r1");

    auto match = hybrid.Match(MakeAlignedTokens(/*num_pages=*/4, /*page_size=*/64, /*start=*/1));
    EXPECT_EQ(match.paged_cache.history_hit_tokens, 256);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 256);
    ASSERT_NE(match.paged_cache.last_node, nullptr);
    ASSERT_EQ(match.paged_cache.per_group_page_ids.at("fh").size(), 4u);
}

TEST(PagedCacheAdmissionTest, ExistingTransportStateGroupUsesSlidingWindowCredit) {
    auto device_alloc = std::make_unique<PageAllocator>(64, 128);
    auto kv_cache = std::make_unique<KVPrefixCache>(device_alloc.get(), /*host_allocator=*/nullptr);

    PagedCacheGroupConfig history{};
    history.group_id = "fh";
    history.rows_per_page = 8;
    history.entry_stride_tokens = 4;
    history.total_pages = 32;
    history.retention = PagedCacheGroupConfig::Retention::FullHistory;
    history.family = PagedCacheGroupFamily::History;

    PagedCacheGroupConfig swa{};
    swa.group_id = "swa";
    swa.rows_per_page = 4;
    swa.entry_stride_tokens = 1;
    swa.total_pages = 10;
    swa.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    swa.sliding_window_tokens = 16;
    swa.family = PagedCacheGroupFamily::State;

    auto history_owner = std::make_unique<PagedCacheGroupAllocator>(history);
    auto swa_owner = std::make_unique<PagedCacheGroupAllocator>(swa);
    HybridPrefixCache hybrid(*kv_cache, /*mamba=*/nullptr, /*mamba_chunk_size=*/0);
    hybrid.RegisterPagedCacheGroup(std::move(history_owner));
    hybrid.RegisterPagedCacheGroup(std::move(swa_owner));
    hybrid.EnablePagedCacheAdjunct({"fh"}, {});

    hybrid.AcquireForRequest("r", /*first_raw_position_of_op=*/0, /*target_raw_tokens_exclusive=*/32);

    auto simulated_free = hybrid.InitialSimulatedFree();
    ASSERT_EQ(simulated_free.at("swa"), 1);
    EXPECT_FALSE(hybrid.AdmitChunk("r", /*first_raw_position_of_op=*/32,
                                   /*target_raw_tokens_exclusive=*/64, simulated_free));
    EXPECT_THROW(hybrid.AcquireForRequest("r", /*first_raw_position_of_op=*/32,
                                          /*target_raw_tokens_exclusive=*/64),
                 std::runtime_error);
}

TEST(PagedCacheRewindTest, RewindRequestReleasesRejectedTailAndKeepsCommittedPrefix) {
    PageAllocator device_alloc(/*page_size=*/2, /*total_pages=*/16);
    KVPrefixCache kv_cache(&device_alloc, /*host=*/nullptr);

    PagedCacheGroupConfig history{};
    history.group_id = "fh";
    history.rows_per_page = 2;
    history.entry_stride_tokens = 1;
    history.total_pages = 8;
    history.retention = PagedCacheGroupConfig::Retention::FullHistory;
    history.family = PagedCacheGroupFamily::History;

    auto history_owner = std::make_unique<PagedCacheGroupAllocator>(history);
    HybridPrefixCache hybrid(kv_cache, /*mamba=*/nullptr, /*mamba_chunk_size=*/0);
    hybrid.RegisterPagedCacheGroup(std::move(history_owner));
    hybrid.EnablePagedCacheAdjunct({"fh"}, {});

    ASSERT_EQ(hybrid.PagedCacheGroupAvailablePages("fh"), 7);
    hybrid.AcquireForRequest("r", /*first_raw_position_of_op=*/0, /*target_raw_tokens_exclusive=*/8);
    ASSERT_EQ(hybrid.GetRequestPagedCachePageIds("r", "fh").size(), 4u);
    ASSERT_EQ(hybrid.PagedCacheGroupAvailablePages("fh"), 3);

    auto tokens = MakeAlignedTokens(/*num_pages=*/2, /*page_size=*/2, /*start=*/1);
    OwnedPages pages = device_alloc.Allocate(/*num_pages=*/2);
    auto inserted =
        kv_cache.Insert<ResourceType::Device>(tokens, /*prefix_pages=*/{}, std::move(pages), /*page_hashes=*/{});
    ASSERT_NE(inserted.last_node, nullptr);
    hybrid.CommitChunk("r", inserted.last_node);
    ASSERT_TRUE(inserted.last_node->HasPagedCacheSnapshot());

    hybrid.RewindRequest("r", /*accepted_raw_tokens=*/5);

    EXPECT_EQ(hybrid.GetRequestPagedCachePageIds("r", "fh").size(), 3u);
    EXPECT_EQ(hybrid.PagedCacheGroupAvailablePages("fh"), 4);
    auto match = hybrid.Match(MakeAlignedTokens(/*num_pages=*/2, /*page_size=*/2, /*start=*/1));
    EXPECT_EQ(match.paged_cache.history_hit_tokens, 4);
}

TEST_F(PagedCacheDecodePublishTest, ContinuingDecodePublishesAcceptedPagesOnly) {
    Submit(RequestSpec{.request_id = "r1", .tokens = {1, 2}});
    ASSERT_NE(GetForwardOp(PlanOnce()), nullptr);

    SendForwardDone("r1", {3});
    ASSERT_NE(GetForwardOp(PlanOnce()), nullptr);

    // Accepted 2 tokens from a 4-token speculative reserve. The accepted truth is
    // {1,2,3,4,5}; with except-last KV semantics, only prefix {1,2,3,4}
    // can be published. Reserved/draft tail slots beyond that must stay local.
    SendForwardDone("r1", {4, 5});
    EXPECT_EQ(scheduler_->GetRequestPagedCachePageIds("r1", "fh").size(), 3u);

    Submit({
        RequestSpec{.request_id = "hit4", .tokens = {1, 2, 3, 4, 5}},
        RequestSpec{.request_id = "probe_tail", .tokens = {1, 2, 3, 4, 5, 6}},
    });
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_GE(fwd->extend_prefix_lens.size(), 2u);

    std::unordered_map<std::string, std::int32_t> prefix_by_request;
    for (std::size_t row = 0; row < fwd->extend_prefix_lens.size(); ++row) {
        ASSERT_LT(row, fwd->request_ids.size());
        prefix_by_request.emplace(fwd->request_ids[row], fwd->extend_prefix_lens[row]);
    }

    ASSERT_TRUE(prefix_by_request.contains("hit4"));
    ASSERT_TRUE(prefix_by_request.contains("probe_tail"));
    EXPECT_EQ(prefix_by_request.at("hit4"), 4);
    EXPECT_EQ(prefix_by_request.at("probe_tail"), 4);
}

TEST_P(PagedCacheOverlapSchedulerTest, DynamicVerifyWidthRetainsAlreadyDispatchedRange) {
    const auto [verify_width, accepted_length, overlap_depth] = GetParam();
    Submit(RequestSpec{.request_id = "r", .tokens = {1, 2}});
    ASSERT_NE(GetForwardOp(PlanOnce()), nullptr);

    // Commit the prefill result so the request can enter decode at C=3.
    SendForwardDone("r", {3});
    const std::int32_t committed_before = scheduler_->GetRequestTokenSize("r");
    ASSERT_EQ(committed_before, 3);

    auto current_plan = PlanOnce();
    auto* current = GetForwardOp(current_plan);
    ASSERT_NE(current, nullptr);
    ASSERT_EQ(current->request_ids.size(), 1u);
    EXPECT_EQ(current->input_lengths.at(0), verify_width);
    ExpectPagedGroupCoversRange(*current, Config(), "overlap.history", /*row=*/0, committed_before,
                                /*token_count=*/(overlap_depth + 1) * verify_width);
    ExpectPagedGroupCoversRange(*current, Config(), "overlap.state", /*row=*/0, committed_before,
                                /*token_count=*/(overlap_depth + 1) * verify_width);

    // The overlapped event loop dispatches the next plan before committing
    // this result. Its CPU token length is still C, while the GPU will use the
    // accepted length A after valid_cache_lengths is updated.
    if (overlap_depth == 1) {
        auto stale_plan = PlanOnce();
        auto* stale = GetForwardOp(stale_plan);
        ASSERT_NE(stale, nullptr);
        ExpectPagedGroupCoversRange(*stale, Config(), "overlap.history", /*row=*/0, committed_before,
                                    /*token_count=*/2 * verify_width);
    }

    const auto history_pages_before = scheduler_->GetRequestPagedCachePageIds("r", "overlap.history");
    const auto state_pages_before = scheduler_->GetRequestPagedCachePageIds("r", "overlap.state");
    const auto reserved_end = committed_before + (overlap_depth + 1) * verify_width;
    ASSERT_EQ(history_pages_before.size(), static_cast<std::size_t>(reserved_end));
    ASSERT_EQ(state_pages_before.size(), static_cast<std::size_t>(reserved_end));

    std::vector<std::int32_t> accepted(static_cast<std::size_t>(accepted_length), 1000 + accepted_length);
    SendForwardDone("r", accepted);
    const std::int32_t accepted_end = committed_before + accepted_length;
    ASSERT_EQ(scheduler_->GetRequestTokenSize("r"), accepted_end);

    const auto history_pages_after = scheduler_->GetRequestPagedCachePageIds("r", "overlap.history");
    const auto state_pages_after = scheduler_->GetRequestPagedCachePageIds("r", "overlap.state");
    const auto protected_end = accepted_end + overlap_depth * verify_width;
    ASSERT_EQ(history_pages_after.size(), static_cast<std::size_t>(protected_end));
    ASSERT_EQ(state_pages_after.size(), static_cast<std::size_t>(protected_end));
    for (std::int32_t pos = accepted_end; pos < accepted_end + overlap_depth * verify_width; ++pos) {
        ASSERT_LT(static_cast<std::size_t>(pos), history_pages_before.size());
        ASSERT_LT(static_cast<std::size_t>(pos), history_pages_after.size());
        EXPECT_EQ(history_pages_after[static_cast<std::size_t>(pos)],
                  history_pages_before[static_cast<std::size_t>(pos)])
            << "history pos=" << pos;
        ASSERT_LT(static_cast<std::size_t>(pos), state_pages_before.size());
        ASSERT_LT(static_cast<std::size_t>(pos), state_pages_after.size());
        EXPECT_EQ(state_pages_after[static_cast<std::size_t>(pos)], state_pages_before[static_cast<std::size_t>(pos)])
            << "pos=" << pos;
    }

    auto next_plan = PlanOnce();
    auto* next = GetForwardOp(next_plan);
    ASSERT_NE(next, nullptr);
    ExpectPagedGroupCoversRange(*next, Config(), "overlap.history", /*row=*/0, accepted_end,
                                /*token_count=*/(overlap_depth + 1) * verify_width);
    ExpectPagedGroupCoversRange(*next, Config(), "overlap.state", /*row=*/0, accepted_end,
                                /*token_count=*/(overlap_depth + 1) * verify_width);
}

TEST_P(PagedCacheOverlapRetractTest, LateResultRecoveryRebuildsDynamicHorizon) {
    const auto [overlap_depth, accepted_length, writeback_first] = GetParam();
    Submit(RequestSpec{.request_id = "r", .tokens = {1, 2}});
    ASSERT_NE(GetForwardOp(PlanOnce()), nullptr);
    SendForwardDone("r", {3});

    // Dispatch a decode at C=3, then let the next scheduler pass retract it
    // before this in-flight result is committed.
    ASSERT_NE(GetForwardOp(PlanOnce()), nullptr);
    const auto history_pages_before = scheduler_->GetRequestPagedCachePageIds("r", "retract.history");
    const auto state_pages_before = scheduler_->GetRequestPagedCachePageIds("r", "retract.state");
    SendReserveNumTokens(/*value=*/100);
    auto retract_plan = PlanOnce();
    EXPECT_TRUE(retract_plan.SchedulerAborts().empty());
    const auto* writeback = GetWriteBack(retract_plan);
    ASSERT_NE(writeback, nullptr);
    ASSERT_FALSE(writeback->op_ids.empty());
    if (writeback_first) {
        SendWriteBackDone(writeback->op_ids.front());
        ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    }

    std::vector<std::int32_t> accepted;
    for (std::int32_t i = 0; i < accepted_length; ++i) {
        accepted.push_back(4 + i);
    }
    SendForwardDone("r", accepted);
    ASSERT_EQ(scheduler_->GetRequestTokenSize("r"), 3 + accepted_length);
    const auto retained_end = 3 + accepted_length + overlap_depth * 4;
    const auto history_pages_after_late_result = scheduler_->GetRequestPagedCachePageIds("r", "retract.history");
    const auto state_pages_after_late_result = scheduler_->GetRequestPagedCachePageIds("r", "retract.state");
    ASSERT_EQ(history_pages_after_late_result.size(), static_cast<std::size_t>((retained_end + 1) / 2));
    ASSERT_EQ(state_pages_after_late_result.size(), static_cast<std::size_t>(retained_end));
    EXPECT_TRUE(std::equal(history_pages_after_late_result.begin(), history_pages_after_late_result.end(),
                           history_pages_before.begin()));
    EXPECT_TRUE(std::equal(state_pages_after_late_result.begin(), state_pages_after_late_result.end(),
                           state_pages_before.begin()));
    if (!writeback_first) {
        SendWriteBackDone(writeback->op_ids.front());
    }
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);

    auto recovery_plan = PlanOnce();
    EXPECT_TRUE(recovery_plan.SchedulerAborts().empty());
    const auto* recovery = GetForwardOp(recovery_plan);
    ASSERT_NE(recovery, nullptr);
    ASSERT_EQ(recovery->request_ids.size(), 1u);
    ASSERT_EQ(recovery->request_ids.front(), "r");
    ASSERT_EQ(recovery->hist_token_lens.size(), 1u);
    EXPECT_EQ(recovery->hist_token_lens.front(), 2 + accepted_length);
    EXPECT_EQ(recovery->input_lengths.front(), 4);
    ExpectPagedGroupCoversRange(*recovery, Config(), "retract.history", /*row=*/0,
                                /*first_token=*/2 + accepted_length, /*token_count=*/(overlap_depth + 1) * 4);
    ExpectPagedGroupCoversRange(*recovery, Config(), "retract.state", /*row=*/0,
                                /*first_token=*/2 + accepted_length, /*token_count=*/(overlap_depth + 1) * 4);
    const auto history_pages_after = scheduler_->GetRequestPagedCachePageIds("r", "retract.history");
    const auto state_pages_after = scheduler_->GetRequestPagedCachePageIds("r", "retract.state");
    ASSERT_GE(history_pages_after.size(), history_pages_after_late_result.size());
    ASSERT_GE(state_pages_after.size(), state_pages_after_late_result.size());
    EXPECT_TRUE(std::equal(history_pages_after_late_result.begin(), history_pages_after_late_result.end(),
                           history_pages_after.begin()));
    EXPECT_TRUE(std::equal(state_pages_after_late_result.begin(), state_pages_after_late_result.end(),
                           state_pages_after.begin()));
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
}

INSTANTIATE_TEST_SUITE_P(OverlapDepthsAndAcceptLengths, PagedCacheOverlapRetractTest,
                         ::testing::Values(std::make_tuple(0, 0, false), std::make_tuple(0, 2, false),
                                           std::make_tuple(0, 4, false), std::make_tuple(1, 0, false),
                                           std::make_tuple(1, 2, false), std::make_tuple(1, 4, false),
                                           std::make_tuple(1, 0, true)));

INSTANTIATE_TEST_SUITE_P(VerifyWidthsAndAcceptLengths, PagedCacheOverlapSchedulerTest,
                         ::testing::Values(std::make_tuple(1, 1, 0), std::make_tuple(2, 1, 0), std::make_tuple(4, 1, 0),
                                           std::make_tuple(8, 1, 0), std::make_tuple(1, 1, 1), std::make_tuple(2, 1, 1),
                                           std::make_tuple(2, 2, 1), std::make_tuple(4, 0, 1), std::make_tuple(4, 2, 1),
                                           std::make_tuple(4, 4, 1), std::make_tuple(8, 0, 1), std::make_tuple(8, 4, 1),
                                           std::make_tuple(8, 7, 1), std::make_tuple(8, 8, 1)));

TEST_F(PagedCacheTerminalMixedSchedulerTest, MixedPrefillDecodePagedTablesCoverScheduledTokens) {
    std::vector<std::string> decode_ids;
    for (int i = 0; i < 5; ++i) {
        decode_ids.push_back("decode_" + std::to_string(i));
        Submit(MakeRequestSpec(decode_ids.back(), /*num_pages=*/4, static_cast<token_t>(1000 + i * 100)));
    }
    PlanOnce();
    for (const auto& id : decode_ids) {
        SendForwardDone(id, {900});
    }
    PlanOnce();
    for (const auto& id : decode_ids) {
        SendForwardDone(id, {901});
    }

    std::unordered_map<std::string, std::int32_t> decode_first_pos;
    for (const auto& id : decode_ids) {
        decode_first_pos.emplace(id, scheduler_->GetRequestTokenSize(id));
    }

    Submit({
        MakeRequestSpec("prefill_0", /*num_pages=*/16, /*start=*/1),
        MakeRequestSpec("prefill_1", /*num_pages=*/16, /*start=*/100),
        MakeRequestSpec("prefill_2", /*num_pages=*/16, /*start=*/200),
    });

    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->request_ids.size(), 8u);
    ASSERT_EQ(fwd->extend_prefix_lens.size(), 3u);

    for (std::size_t row = 0; row < fwd->request_ids.size(); ++row) {
        std::int32_t first_token = 0;
        if (row < fwd->extend_prefix_lens.size()) {
            first_token = fwd->extend_prefix_lens[row];
        } else {
            auto it = decode_first_pos.find(fwd->request_ids[row]);
            ASSERT_NE(it, decode_first_pos.end()) << "request_id=" << fwd->request_ids[row];
            first_token = it->second;
        }
        ASSERT_LT(row, fwd->input_lengths.size());
        ExpectPagedGroupCoversRange(*fwd, Config(), "fh", row, first_token, fwd->input_lengths[row]);
        ExpectPagedGroupCoversRange(*fwd, Config(), "swa", row, first_token, fwd->input_lengths[row]);
    }
}

}  // namespace tokenspeed::test
