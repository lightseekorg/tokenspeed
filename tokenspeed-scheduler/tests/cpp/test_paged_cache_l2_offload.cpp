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

#include "paged_cache_test_fixture.h"
#include "integration_test_helper.h"
#include "resource/radix_tree/tree_resource.h"
#include "scheduler/operations/forward.h"

#include <map>
#include <set>

namespace tokenspeed::test {

class PagedCacheL2OffloadTest : public PagedCacheSmallFixture {
protected:
    void SetUp() override {
        PagedCacheSmallFixture::SetUp();

        auto fh_host_cfg = MakeGroupConfig("fh", kSmallFixtureParams.fh_rows_per_page, kSmallFixtureParams.fh_stride,
                                           PagedCacheGroupConfig::Retention::FullHistory, /*window=*/0,
                                           PagedCacheGroupFamily::History);
        fh_host_cfg.total_pages = 3;
        auto swa_host_cfg = MakeGroupConfig(
            "swa", kSmallFixtureParams.swa_rows_per_page, kSmallFixtureParams.swa_stride,
            PagedCacheGroupConfig::Retention::SlidingWindow, kSlidingWindow, PagedCacheGroupFamily::State);
        swa_host_cfg.total_pages = 3;
        auto fh_host = std::make_unique<PagedCacheGroupAllocator>(std::move(fh_host_cfg));
        auto swa_host = std::make_unique<PagedCacheGroupAllocator>(std::move(swa_host_cfg));
        hybrid_->RegisterPagedCacheHostGroup(std::move(fh_host));
        hybrid_->RegisterPagedCacheHostGroup(std::move(swa_host));
    }

    TreeNode* SeedCompleteDeviceSnapshot() {
        TreeNode* terminal = InsertDevicePages(/*num_pages=*/2, /*token_start=*/1);
        EXPECT_NE(terminal, nullptr);
        hybrid_->AttachPagedCacheSnapshotToNode(terminal, MakeCompleteSnapshot(kLcm));
        return terminal;
    }

    TreeNode* SeedCompleteDeviceSnapshot(token_t token_start) {
        TreeNode* terminal = InsertDevicePages(/*num_pages=*/2, token_start);
        EXPECT_NE(terminal, nullptr);
        hybrid_->AttachPagedCacheSnapshotToNode(terminal, MakeCompleteSnapshot(kLcm));
        return terminal;
    }

    void AttachHostResource(TreeNode* node) {
        ASSERT_NE(node, nullptr);
        node->AttachResource<ResourceType::Host>(std::make_unique<NodeResource<ResourceType::Host>>(OwnedPages{}));
    }
};

class PagedCacheL2SchedulerTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.page_size = kSmallFixtureParams.page_size;
        cfg.device_allocator.total_pages = 32;
        cfg.host_allocator.total_pages = 32;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = false;

        cfg.paged_cache_groups = {
            MakeSchedulerGroup("fh", kSmallFixtureParams.fh_rows_per_page, kSmallFixtureParams.fh_stride,
                               PagedCacheGroupConfig::Retention::FullHistory, /*window=*/0,
                               PagedCacheGroupFamily::History),
            MakeSchedulerGroup("swa", kSmallFixtureParams.swa_rows_per_page, kSmallFixtureParams.swa_stride,
                               PagedCacheGroupConfig::Retention::SlidingWindow,
                               kSmallFixtureParams.sliding_window_tokens, PagedCacheGroupFamily::State),
        };
        cfg.paged_cache_host_group_pages = {{"fh", 16}, {"swa", 16}};
        PrefixCacheAdjunctSpec adjunct;
        adjunct.required_groups = {"fh"};
        cfg.prefix_cache_adjunct = adjunct;
        return cfg;
    }

    static PagedCacheGroupConfig MakeSchedulerGroup(std::string group_id, std::int32_t rows_per_page,
                                                    std::int32_t stride, PagedCacheGroupConfig::Retention retention,
                                                    std::int32_t window, PagedCacheGroupFamily family) {
        PagedCacheGroupConfig cfg{};
        cfg.group_id = std::move(group_id);
        cfg.rows_per_page = rows_per_page;
        cfg.entry_stride_tokens = stride;
        cfg.total_pages = 16;
        cfg.retention = retention;
        if (retention == PagedCacheGroupConfig::Retention::SlidingWindow) {
            cfg.sliding_window_tokens = window;
        }
        cfg.family = family;
        return cfg;
    }

    void BringToDecoding(const std::string& id, token_t start = 1) {
        Submit(MakeRequestSpec(id, /*num_pages=*/2, start));
        PlanOnce();
        SendForwardDone(id, {42});
        PlanOnce();
    }

    static const FlatWriteBackOperation* GetWriteBack(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cache_op = std::get_if<CacheOperation>(&op)) {
                if (auto* wb = std::get_if<FlatWriteBackOperation>(cache_op)) {
                    return wb;
                }
            }
        }
        return nullptr;
    }

    static const FlatLoadBackOperation* GetLoadBack(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cache_op = std::get_if<CacheOperation>(&op)) {
                if (auto* lb = std::get_if<FlatLoadBackOperation>(cache_op)) {
                    return lb;
                }
            }
        }
        return nullptr;
    }

    static const FlatForwardOperation* GetForward(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* fwd = std::get_if<FlatForwardOperation>(&op)) {
                return fwd;
            }
        }
        return nullptr;
    }

    static std::map<std::string, std::int32_t> UniqueDstPagesByGroup(const FlatWriteBackOperation& writeback) {
        std::map<std::string, std::set<std::int32_t>> pages_by_group;
        for (const auto& transfers : writeback.paged_cache_transfers) {
            for (const auto& transfer : transfers) {
                auto& pages = pages_by_group[transfer.group_id];
                pages.insert(transfer.dst_pages.begin(), transfer.dst_pages.end());
            }
        }
        std::map<std::string, std::int32_t> counts;
        for (const auto& [group_id, pages] : pages_by_group) {
            counts[group_id] = static_cast<std::int32_t>(pages.size());
        }
        return counts;
    }

    static std::int32_t UniqueKvDstPages(const FlatWriteBackOperation& writeback) {
        std::set<std::int32_t> pages;
        for (const auto& op_pages : writeback.dst_pages) {
            pages.insert(op_pages.begin(), op_pages.end());
        }
        return static_cast<std::int32_t>(pages.size());
    }
};

class PagedCacheHistoryOnlyL2OffloadTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kPageSize = 2;
    static constexpr std::int32_t kLcm = 4;

    void SetUp() override {
        device_alloc_ = std::make_unique<PageAllocator>(kPageSize, /*total_pages=*/64);
        kv_cache_ = std::make_unique<KVPrefixCache>(device_alloc_.get(), /*host=*/nullptr);

        auto history_cfg = MakeHistoryGroup(/*total_pages=*/32);
        auto history_owner = std::make_unique<PagedCacheGroupAllocator>(history_cfg);
        history_alloc_ = history_owner.get();
        hybrid_ = std::make_unique<HybridPrefixCache>(*kv_cache_, /*mamba=*/nullptr, /*mamba_chunk_size=*/0);
        hybrid_->RegisterPagedCacheGroup(std::move(history_owner));

        auto host_history_cfg = MakeHistoryGroup(/*total_pages=*/32);
        auto host_history = std::make_unique<PagedCacheGroupAllocator>(host_history_cfg);
        hybrid_->RegisterPagedCacheHostGroup(std::move(host_history));
        hybrid_->EnablePagedCacheAdjunct({"fh"}, {});
        kv_cache_->GetDeviceManager().SetEvictionCallback([this](TreeNode* node) { hybrid_->OnKVEvict(node); });
    }

    TreeNode* InsertDeviceTokens(std::int32_t num_pages, token_t token_start = 1) {
        auto tokens = MakeAlignedTokens(num_pages, kPageSize, token_start);
        OwnedPages pages = device_alloc_->Allocate(num_pages);
        auto inserted =
            kv_cache_->Insert<ResourceType::Device>(tokens, /*prefix_pages=*/{}, std::move(pages), /*page_hashes=*/{});
        return inserted.last_node;
    }

    std::unique_ptr<PagedCacheSnapshot> MakeHistorySnapshot(std::int32_t prefix_len_tokens) {
        PagedCacheGroupTable table{history_alloc_};
        table.Acquire(prefix_len_tokens);
        auto committed = table.CommitHistoryToSnapshot(prefix_len_tokens);

        PagedCacheGroupSnapshot group{};
        group.pages = std::move(committed.pages);
        group.base_logical_page = committed.segment_base_logical_page;
        group.raw_token_cursor = prefix_len_tokens;
        group.sliding = false;

        auto snapshot = std::make_unique<PagedCacheSnapshot>();
        snapshot->prefix_len_tokens = prefix_len_tokens;
        snapshot->groups.emplace("fh", std::move(group));
        return snapshot;
    }

    static PagedCacheGroupConfig MakeHistoryGroup(std::int32_t total_pages) {
        PagedCacheGroupConfig cfg{};
        cfg.group_id = "fh";
        cfg.rows_per_page = 4;
        cfg.entry_stride_tokens = 1;
        cfg.total_pages = total_pages;
        cfg.retention = PagedCacheGroupConfig::Retention::FullHistory;
        cfg.family = PagedCacheGroupFamily::History;
        return cfg;
    }

    std::unique_ptr<PageAllocator> device_alloc_;
    std::unique_ptr<KVPrefixCache> kv_cache_;
    PagedCacheGroupAllocator* history_alloc_{nullptr};
    std::unique_ptr<HybridPrefixCache> hybrid_;
};

TEST_F(PagedCacheL2OffloadTest, PendingHostSnapshotInvisibleUntilWriteBackAck) {
    TreeNode* terminal = SeedCompleteDeviceSnapshot();
    ASSERT_NE(terminal, nullptr);
    AttachHostResource(terminal);
    const auto tokens = MakeAlignedTokens(/*num_pages=*/2, kPageSize, /*start=*/1);

    auto transfers = hybrid_->PreparePagedCacheHostWriteBack({terminal});
    ASSERT_EQ(transfers.size(), 2u);
    EXPECT_TRUE(terminal->HasPagedCachePendingHostSnapshot());
    EXPECT_FALSE(terminal->HasPagedCacheHostSnapshot());

    auto pending_match = hybrid_->Match(tokens);
    EXPECT_EQ(pending_match.paged_cache_host.last_node, nullptr);
    EXPECT_EQ(pending_match.paged_cache_host.prefix_len_tokens, 0);

    hybrid_->OnPagedCacheHostWriteBackDone({terminal}, /*success=*/true);
    EXPECT_FALSE(terminal->HasPagedCachePendingHostSnapshot());
    ASSERT_TRUE(terminal->HasPagedCacheHostSnapshot());

    auto visible_match = hybrid_->Match(tokens);
    ASSERT_NE(visible_match.paged_cache_host.last_node, nullptr);
    EXPECT_EQ(visible_match.paged_cache_host.last_node, terminal);
    EXPECT_EQ(visible_match.paged_cache_host.prefix_len_tokens, kLcm);
}

TEST_F(PagedCacheL2OffloadTest, FailedHostWriteBackReleasesPendingSnapshotWithoutPublishing) {
    TreeNode* terminal = SeedCompleteDeviceSnapshot();
    ASSERT_NE(terminal, nullptr);
    AttachHostResource(terminal);
    const auto tokens = MakeAlignedTokens(/*num_pages=*/2, kPageSize, /*start=*/1);
    const std::int32_t fh_before = hybrid_->PagedCacheHostGroupAvailablePages("fh");
    const std::int32_t swa_before = hybrid_->PagedCacheHostGroupAvailablePages("swa");

    auto transfers = hybrid_->PreparePagedCacheHostWriteBack({terminal});
    ASSERT_FALSE(transfers.empty());
    ASSERT_TRUE(terminal->HasPagedCachePendingHostSnapshot());
    ASSERT_LT(hybrid_->PagedCacheHostGroupAvailablePages("fh"), fh_before);
    ASSERT_LT(hybrid_->PagedCacheHostGroupAvailablePages("swa"), swa_before);

    hybrid_->OnPagedCacheHostWriteBackDone({terminal}, /*success=*/false);

    EXPECT_FALSE(terminal->HasPagedCachePendingHostSnapshot());
    EXPECT_FALSE(terminal->HasPagedCacheHostSnapshot());
    EXPECT_TRUE(terminal->HasPagedCacheSnapshot());
    EXPECT_EQ(hybrid_->PagedCacheHostGroupAvailablePages("fh"), fh_before);
    EXPECT_EQ(hybrid_->PagedCacheHostGroupAvailablePages("swa"), swa_before);
    auto match = hybrid_->Match(tokens);
    EXPECT_EQ(match.paged_cache_host.last_node, nullptr);
    EXPECT_EQ(match.paged_cache_host.prefix_len_tokens, 0);
}

TEST_F(PagedCacheL2OffloadTest, HostHitMaterializesDeviceDestinationPagesForForwardMetadata) {
    TreeNode* terminal = SeedCompleteDeviceSnapshot();
    ASSERT_NE(terminal, nullptr);
    AttachHostResource(terminal);
    auto writeback = hybrid_->PreparePagedCacheHostWriteBack({terminal});
    ASSERT_FALSE(writeback.empty());
    hybrid_->OnPagedCacheHostWriteBackDone({terminal}, /*success=*/true);
    auto detached_device = hybrid_->DetachPagedCacheSnapshotFromNode(terminal);
    ASSERT_NE(detached_device, nullptr);

    const auto tokens = MakeAlignedTokens(/*num_pages=*/2, kPageSize, /*start=*/1);
    auto match = hybrid_->Match(tokens);
    ASSERT_EQ(match.paged_cache.last_node, nullptr);
    ASSERT_NE(match.paged_cache_host.last_node, nullptr);

    auto loadback = hybrid_->PreparePagedCacheDeviceLoadBack("r-load", match.paged_cache_host);
    ASSERT_EQ(loadback.size(), 2u);

    PrefillOperation op{{
        .request_id = "r-load",
        .request_pool_index = 0,
        .input_length = 0,
        .occupied_pages = {},
        .begin = 0,
        .size = 0,
        .prefill_length = kLcm,
    }};
    hybrid_->AcquireForRequest("r-load", /*first_raw_position_of_op=*/0, /*target_raw_tokens_exclusive=*/kLcm);
    hybrid_->PopulateOp(op);

    for (const auto& transfer : loadback) {
        auto pages_it = op.paged_cache_pages.find(transfer.group_id);
        ASSERT_NE(pages_it, op.paged_cache_pages.end()) << transfer.group_id;
        EXPECT_EQ(pages_it->second, transfer.dst_pages) << transfer.group_id;
    }

    hybrid_->ReleaseRequest("r-load");
}

TEST_F(PagedCacheHistoryOnlyL2OffloadTest, HostLoadedHistoryPrefixCanPublishContinuationSnapshot) {
    TreeNode* prefix = InsertDeviceTokens(/*num_pages=*/2);
    ASSERT_NE(prefix, nullptr);
    ASSERT_TRUE(hybrid_->AttachPagedCacheSnapshotToNode(prefix, MakeHistorySnapshot(kLcm)));
    prefix->AttachResource<ResourceType::Host>(std::make_unique<NodeResource<ResourceType::Host>>(OwnedPages{}));

    auto writeback = hybrid_->PreparePagedCacheHostWriteBack({prefix});
    ASSERT_EQ(writeback.size(), 1u);
    hybrid_->OnPagedCacheHostWriteBackDone({prefix}, /*success=*/true);
    auto detached_device = hybrid_->DetachPagedCacheSnapshotFromNode(prefix);
    ASSERT_NE(detached_device, nullptr);

    auto host_match = hybrid_->Match(MakeAlignedTokens(/*num_pages=*/2, kPageSize, /*start=*/1));
    ASSERT_EQ(host_match.paged_cache.last_node, nullptr);
    ASSERT_NE(host_match.paged_cache_host.last_node, nullptr);
    EXPECT_EQ(host_match.paged_cache_host.prefix_len_tokens, kLcm);

    auto loadback = hybrid_->PreparePagedCacheDeviceLoadBack("r-load", host_match.paged_cache_host);
    ASSERT_EQ(loadback.size(), 1u);

    TreeNode* extended = InsertDeviceTokens(/*num_pages=*/4);
    ASSERT_NE(extended, nullptr);
    hybrid_->AcquireForRequest("r-load", /*first_raw_position_of_op=*/kLcm,
                               /*target_raw_tokens_exclusive=*/2 * kLcm);
    hybrid_->CommitChunk("r-load", extended);

    ASSERT_TRUE(extended->HasPagedCacheSnapshot());
    auto device_match = hybrid_->Match(MakeAlignedTokens(/*num_pages=*/4, kPageSize, /*start=*/1));
    ASSERT_NE(device_match.paged_cache.last_node, nullptr);
    EXPECT_EQ(device_match.paged_cache.prefix_len_tokens, 2 * kLcm);
    ASSERT_EQ(device_match.paged_cache.per_group_page_ids.count("fh"), 1u);
    EXPECT_EQ(device_match.paged_cache.per_group_page_ids.at("fh").size(), 2u);
}

TEST_F(PagedCacheL2OffloadTest, HostStateRecoveryRequiresExactTerminalStateCompleteness) {
    TreeNode* terminal = SeedCompleteDeviceSnapshot();
    ASSERT_NE(terminal, nullptr);
    AttachHostResource(terminal);
    auto writeback = hybrid_->PreparePagedCacheHostWriteBack({terminal});
    ASSERT_FALSE(writeback.empty());
    hybrid_->OnPagedCacheHostWriteBackDone({terminal}, /*success=*/true);
    auto detached_device = hybrid_->DetachPagedCacheSnapshotFromNode(terminal);
    ASSERT_NE(detached_device, nullptr);

    const auto tokens = MakeAlignedTokens(/*num_pages=*/2, kPageSize, /*start=*/1);
    auto match = hybrid_->Match(tokens, MatchIntent::StateRecovery);

    ASSERT_EQ(match.paged_cache.last_node, nullptr);
    ASSERT_NE(match.paged_cache_host.last_node, nullptr);
    EXPECT_EQ(match.paged_cache_host.last_node, terminal);
    EXPECT_EQ(match.paged_cache_host.prefix_len_tokens, kLcm);
}

TEST_F(PagedCacheL2OffloadTest, HostStateRecoveryCapsWhenTerminalStateGroupMissing) {
    TreeNode* terminal = SeedCompleteDeviceSnapshot();
    ASSERT_NE(terminal, nullptr);
    AttachHostResource(terminal);
    auto writeback = hybrid_->PreparePagedCacheHostWriteBack({terminal});
    ASSERT_FALSE(writeback.empty());
    hybrid_->OnPagedCacheHostWriteBackDone({terminal}, /*success=*/true);
    auto detached_device = hybrid_->DetachPagedCacheSnapshotFromNode(terminal);
    ASSERT_NE(detached_device, nullptr);

    auto host_snapshot = hybrid_->DetachPagedCacheHostSnapshotFromNode(terminal);
    ASSERT_NE(host_snapshot, nullptr);
    host_snapshot->groups.erase("swa");
    ASSERT_TRUE(hybrid_->AttachPagedCacheHostSnapshotToNode(terminal, std::move(host_snapshot)));

    const auto tokens = MakeAlignedTokens(/*num_pages=*/2, kPageSize, /*start=*/1);
    auto match = hybrid_->Match(tokens, MatchIntent::StateRecovery);

    EXPECT_EQ(match.paged_cache.last_node, nullptr);
    EXPECT_EQ(match.paged_cache_host.last_node, nullptr);
    ASSERT_NE(match.device.last_node, nullptr);
    EXPECT_TRUE(match.device.last_node->IsRoot());
    ASSERT_NE(match.host.last_node, nullptr);
    EXPECT_TRUE(match.host.last_node->IsRoot());
}

TEST_F(PagedCacheL2OffloadTest, KVHostEvictReleasesPagedCacheHostSnapshotPages) {
    TreeNode* terminal = SeedCompleteDeviceSnapshot();
    ASSERT_NE(terminal, nullptr);
    AttachHostResource(terminal);
    auto writeback = hybrid_->PreparePagedCacheHostWriteBack({terminal});
    ASSERT_FALSE(writeback.empty());
    hybrid_->OnPagedCacheHostWriteBackDone({terminal}, /*success=*/true);
    ASSERT_TRUE(terminal->HasPagedCacheHostSnapshot());
    EXPECT_EQ(hybrid_->PagedCacheHostGroupAvailablePages("fh"), 1);
    EXPECT_EQ(hybrid_->PagedCacheHostGroupAvailablePages("swa"), 0);

    hybrid_->OnKVHostEvict(terminal);

    EXPECT_FALSE(terminal->HasPagedCacheHostSnapshot());
    EXPECT_EQ(hybrid_->PagedCacheHostGroupAvailablePages("fh"), 2);
    EXPECT_EQ(hybrid_->PagedCacheHostGroupAvailablePages("swa"), 2);
}

TEST_F(PagedCacheL2OffloadTest, HostWriteBackPrunesOnlyUnpinnedPagedCacheHostSnapshots) {
    TreeNode* pinned = SeedCompleteDeviceSnapshot(/*token_start=*/1);
    ASSERT_NE(pinned, nullptr);
    AttachHostResource(pinned);
    auto first_writeback = hybrid_->PreparePagedCacheHostWriteBack({pinned});
    ASSERT_FALSE(first_writeback.empty());
    hybrid_->OnPagedCacheHostWriteBackDone({pinned}, /*success=*/true);
    ASSERT_TRUE(pinned->HasPagedCacheHostSnapshot());

    TreeNode* next = SeedCompleteDeviceSnapshot(/*token_start=*/101);
    ASSERT_NE(next, nullptr);
    AttachHostResource(next);
    {
        HostNodeRef lock(pinned);
        auto blocked = hybrid_->PreparePagedCacheHostWriteBack({next});
        EXPECT_TRUE(blocked.empty());
        EXPECT_TRUE(pinned->HasPagedCacheHostSnapshot());
        EXPECT_FALSE(next->HasPagedCachePendingHostSnapshot());
    }

    auto writeback = hybrid_->PreparePagedCacheHostWriteBack({next});
    ASSERT_FALSE(writeback.empty());
    EXPECT_FALSE(pinned->HasPagedCacheHostSnapshot());
    EXPECT_TRUE(next->HasPagedCachePendingHostSnapshot());
}

TEST_F(PagedCacheL2SchedulerTest, WriteBackAckDemotesDeviceSnapshotAndNextHitLoadsFromHost) {
    const auto request_tokens = MakeAlignedTokens(/*num_pages=*/3, kSmallFixtureParams.page_size, /*start=*/1);
    BringToDecoding("r1");
    SendFinish("r1");

    auto writeback_plan = PlanOnce();
    const auto* writeback = GetWriteBack(writeback_plan);
    ASSERT_NE(writeback, nullptr);
    ASSERT_EQ(writeback->op_ids.size(), 1u);
    ASSERT_EQ(writeback->paged_cache_transfers.size(), 1u);
    ASSERT_FALSE(writeback->paged_cache_transfers[0].empty());

    SendWriteBackDone(writeback->op_ids[0]);
    PlanOnce();
    EXPECT_EQ(scheduler_->PagedCacheGroupAvailablePages("fh"), 15);
    EXPECT_EQ(scheduler_->PagedCacheGroupAvailablePages("swa"), 15);
    EXPECT_EQ(scheduler_->PagedCacheHostGroupTotalPages("fh"), 16);
    EXPECT_EQ(scheduler_->PagedCacheHostGroupTotalPages("swa"), 16);
    const std::int32_t fh_host_after_writeback = scheduler_->PagedCacheHostGroupAvailablePages("fh");
    const std::int32_t swa_host_after_writeback = scheduler_->PagedCacheHostGroupAvailablePages("swa");
    EXPECT_LT(fh_host_after_writeback, scheduler_->PagedCacheHostGroupTotalPages("fh"));
    EXPECT_LT(swa_host_after_writeback, scheduler_->PagedCacheHostGroupTotalPages("swa"));

    Submit(RequestSpec{
        .request_id = "r2",
        .tokens = request_tokens,
    });
    auto loadback_plan = PlanOnce();
    const auto* loadback = GetLoadBack(loadback_plan);
    ASSERT_NE(loadback, nullptr);
    ASSERT_EQ(loadback->paged_cache_transfers.size(), 1u);
    ASSERT_FALSE(loadback->paged_cache_transfers[0].empty());

    std::set<std::string> groups;
    for (const auto& transfer : loadback->paged_cache_transfers[0]) {
        groups.insert(transfer.group_id);
        EXPECT_EQ(transfer.src_pages.size(), transfer.dst_pages.size());
        EXPECT_FALSE(transfer.src_pages.empty());
    }
    EXPECT_EQ(groups, (std::set<std::string>{"fh", "swa"}));
    EXPECT_EQ(scheduler_->PagedCacheHostGroupAvailablePages("fh"), fh_host_after_writeback);
    EXPECT_EQ(scheduler_->PagedCacheHostGroupAvailablePages("swa"), swa_host_after_writeback);
}

TEST_F(PagedCacheL2SchedulerTest, FailedWriteBackDoesNotPublishHostSnapshotOrDemoteDeviceSnapshot) {
    const auto request_tokens = MakeAlignedTokens(/*num_pages=*/3, kSmallFixtureParams.page_size, /*start=*/1);
    BringToDecoding("r1");
    SendFinish("r1");

    auto writeback_plan = PlanOnce();
    const auto* writeback = GetWriteBack(writeback_plan);
    ASSERT_NE(writeback, nullptr);
    ASSERT_EQ(writeback->op_ids.size(), 1u);
    ASSERT_EQ(writeback->paged_cache_transfers.size(), 1u);
    ASSERT_FALSE(writeback->paged_cache_transfers[0].empty());
    const auto transferred_host_pages = UniqueDstPagesByGroup(*writeback);
    const std::int32_t transferred_kv_host_pages = UniqueKvDstPages(*writeback);
    const std::size_t kv_host_before_ack = scheduler_->AvailableHostKvPages();
    const std::int32_t fh_host_before_ack = scheduler_->PagedCacheHostGroupAvailablePages("fh");
    const std::int32_t swa_host_before_ack = scheduler_->PagedCacheHostGroupAvailablePages("swa");
    ASSERT_GT(transferred_kv_host_pages, 0);
    ASSERT_GT(transferred_host_pages.at("fh"), 0);
    ASSERT_GT(transferred_host_pages.at("swa"), 0);

    SendWriteBackDone(writeback->op_ids[0], /*success=*/false);
    PlanOnce();

    EXPECT_EQ(scheduler_->AvailableHostKvPages(), kv_host_before_ack + transferred_kv_host_pages);
    EXPECT_EQ(scheduler_->PagedCacheHostGroupAvailablePages("fh"),
              fh_host_before_ack + transferred_host_pages.at("fh"));
    EXPECT_EQ(scheduler_->PagedCacheHostGroupAvailablePages("swa"),
              swa_host_before_ack + transferred_host_pages.at("swa"));

    Submit(RequestSpec{
        .request_id = "r2",
        .tokens = request_tokens,
    });
    auto reuse_plan = PlanOnce();
    EXPECT_EQ(GetLoadBack(reuse_plan), nullptr);
    ASSERT_NE(GetForward(reuse_plan), nullptr);
}

}  // namespace tokenspeed::test
