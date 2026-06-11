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

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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
        cfg.page_size = 2;
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

    static const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
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
        cfg.page_size = 1;
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

    static const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
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
        auto window_owner = std::make_unique<PagedCacheGroupAllocator>(MakeGroup(
            kWindowStateGroup, /*rows_per_page=*/64, /*stride=*/1, PagedCacheGroupConfig::Retention::SlidingWindow,
            kWindowStateTokens, PagedCacheGroupFamily::State, /*total_pages=*/6));

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

TEST_F(PagedCacheTerminalContinuationTest, StatePruneDropsContinuationAndFallsBackToColdPrefill) {
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
    EXPECT_EQ(match.paged_cache.history_hit_tokens, 0);
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 0);
    EXPECT_TRUE(match.paged_cache.per_group_page_ids.empty());
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
