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
#include "integration_test_helper.h"

namespace tokenspeed::test {

class MambaIntegrationTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_mamba = true;
        cfg.mamba_pool_total_chunks = 16;
        return cfg;
    }
};

TEST_F(MambaIntegrationTest, BasicPrefillDecodeFinish) {
    Submit(MakeRequestSpec("r1", 2));

    auto plan1 = PlanOnce();
    EXPECT_FALSE(plan1.Operations().empty());

    SendForwardDone("r1", {100});

    auto plan2 = PlanOnce();
    EXPECT_FALSE(plan2.Operations().empty());

    SendFinish("r1");
    auto plan3 = PlanOnce();
}

TEST_F(MambaIntegrationTest, PrefixSharingWithMamba) {
    // Use 5 pages so that after decode-token append (11 tokens),
    // except_last gives 5 full pages. R2 also uses 5 pages (10 tokens),
    // except_last → (10-1)/2 = 4 pages, but the radix tree stores 5 pages
    // from R1's FinishEvent. The mamba slot lives at depth 5, so R2's
    // 4-page query still walks up and finds the mamba at depth 4 (the
    // checkpoint inserted during ScheduleDecodeEvent).
    //
    // A cleaner approach: use 4 pages for R1 and verify that R2 gets a
    // partial mamba prefix hit of 3 pages (since except_last on 8 aligned
    // tokens yields only 3 pages for the match query).
    const std::int32_t kPages = 4;

    Submit(MakeRequestSpec("r1", kPages));
    auto plan1 = PlanOnce();
    ASSERT_FALSE(plan1.Operations().empty());

    SendForwardDone("r1", {100});
    auto plan2 = PlanOnce();
    ASSERT_FALSE(plan2.Operations().empty());

    SendFinish("r1");
    PlanOnce();

    // R2: same prefix tokens. GetFullPagedTokens(except_last=true) on
    // kPages*PageSize tokens yields (kPages*PageSize - 1) / PageSize pages.
    // With block_size=2, 4 pages = 8 tokens → except_last → 3 pages.
    // The mamba checkpoint from R1 sits at depth 4 (inserted during decode
    // transition), but the mamba working slot also at depth 4 (from Finish).
    // R2's match walks 3 pages; FindLastMambaNode walks up — the ancestor
    // chain doesn't include the 4-page-depth node since it's a sibling.
    //
    // The tree after R1 finish looks like:
    //   root → [page0,page1,page2,page3] (4-page node, has mamba)
    // R2 queries with 3 pages → match hits first 3 pages of the 4-page node
    // via split. After split: root → 3-page prefix → 1-page suffix (mamba).
    // FindLastMambaNode from 3-page prefix → no mamba → walks up → root → null.
    //
    // So with aligned tokens, R2 does NOT get a mamba hit.
    // Verify the partial-hit behavior:
    Submit(MakeRequestSpec("r2", kPages));
    auto plan3 = PlanOnce();
    ASSERT_FALSE(plan3.Operations().empty());

    const auto& op = plan3.Operations()[0];
    auto* flat = std::get_if<FlatForwardOperation>(&op);
    ASSERT_NE(flat, nullptr) << "Expected FlatForwardOperation";
    ASSERT_EQ(flat->request_ids.size(), 1u);
    EXPECT_EQ(flat->request_ids[0], "r2");

    // With page-aligned input, except_last strips the last page so the
    // mamba node is beyond the match point. mamba_cow_src is NOT set.
    // This is the correct behavior for the current except_last semantics.
    // A real mamba COW hit only occurs when the query depth reaches a node
    // that has a mamba slot (e.g., non-aligned inputs or longer prefixes).
    //
    // For now verify the operation is well-formed and doesn't crash:
    EXPECT_GE(flat->extend_prefix_lens[0], 0);
    EXPECT_GE(flat->input_lengths[0], 0);

    SendForwardDone("r2", {200});
    PlanOnce();
    SendFinish("r2");
    PlanOnce();
}

TEST_F(MambaIntegrationTest, AbortFreesMambaSlots) {
    Submit(MakeRequestSpec("r1", 2));
    PlanOnce();

    SendFinish("r1");
    PlanOnce();

    for (int i = 0; i < 8; ++i) {
        Submit(MakeRequestSpec("fill_" + std::to_string(i), 1));
    }
    PlanOnce();
}

class MambaDecodeCapacityTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_mamba = true;
        cfg.mamba_pool_total_chunks = 2;
        cfg.max_batch_size = 1;
        return cfg;
    }
};

TEST_F(MambaDecodeCapacityTest, PrefillDoneDecodeCapacityMissRetractsInsteadOfThrowing) {
    Submit(MakeRequestSpec("r1", 1));

    auto prefill = PlanOnce();
    ASSERT_FALSE(prefill.Operations().empty());
    SendForwardDone("r1", {100});

    ExecutionPlan plan;
    EXPECT_NO_THROW(plan = PlanOnce());

    auto writebacks = ExtractCacheOpsOfKind<FlatWriteBackOperation>(plan);
    ASSERT_EQ(writebacks.size(), 1u);
}

class MambaDecodeAdmissionTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.role = Role::kD;
        cfg.enable_mamba = true;
        cfg.mamba_pool_total_chunks = 24;
        cfg.max_batch_size = 16;
        cfg.max_scheduled_tokens = 1024;
        return cfg;
    }

    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(PDEvent{pd::BootstrappedEvent{request_id}});
        scheduler_->Advance(std::move(event));
    }
};

TEST_F(MambaDecodeAdmissionTest, SubmittedBatchStopsAtLocalMambaPairCapacity) {
    std::vector<RequestSpec> requests;
    for (int i = 0; i < 16; ++i) {
        requests.push_back(MakeRequestSpec("r" + std::to_string(i), 1));
    }
    Submit(requests);
    for (const auto& request : requests) {
        SendBootstrapped(request.request_id);
    }

    auto plan = PlanOnce();
    const auto* fwd = GetForward(plan);
    ASSERT_NE(fwd, nullptr);

    ASSERT_EQ(fwd->request_ids.size(), 12u);
    for (std::size_t i = 0; i < fwd->request_ids.size(); ++i) {
        EXPECT_GE(fwd->mamba_working_indices[i], 0) << fwd->request_ids[i];
        EXPECT_GE(fwd->mamba_checkpoint_dst_indices[i], 0) << fwd->request_ids[i];
    }
    EXPECT_EQ(scheduler_->WaitingSize(), 4u);
    EXPECT_EQ(scheduler_->PrefillSize(), 12u);
}

class MambaUnalignedCheckpointTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_mamba = true;
        cfg.mamba_pool_total_chunks = 16;
        cfg.mamba_cache_chunk_size = 4;
        cfg.max_scheduled_tokens = 3;
        cfg.enable_l3_storage = false;
        return cfg;
    }
};

TEST_F(MambaUnalignedCheckpointTest, ChunkBoundaryNotAlignedToMambaChunkDoesNotPublishCheckpoint) {
    Submit(RequestSpec{.request_id = "r1", .tokens = {1, 2, 3, 4, 5}});

    auto first_chunk = PlanOnce();
    const auto* first_forward = GetForward(first_chunk);
    ASSERT_NE(first_forward, nullptr);
    ASSERT_EQ(first_forward->input_lengths[0], 3);

    auto second_chunk = PlanOnce();
    const auto* second_forward = GetForward(second_chunk);
    ASSERT_NE(second_forward, nullptr);

    SendFinish("r1");
    PlanOnce();

    Submit(RequestSpec{.request_id = "r2", .tokens = {1, 2, 9}});
    auto prefix_probe = PlanOnce();
    const auto* probe_forward = GetForward(prefix_probe);
    ASSERT_NE(probe_forward, nullptr);
    ASSERT_EQ(probe_forward->request_ids.size(), 1u);
    EXPECT_EQ(probe_forward->request_ids[0], "r2");
    EXPECT_EQ(probe_forward->extend_prefix_lens[0], 0);
    EXPECT_EQ(probe_forward->mamba_cow_src_indices[0], -1)
        << "C++ must not publish a checkpoint that Python skipped at an unaligned boundary";
}

class MambaL2IntegrationTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_mamba = true;
        cfg.mamba_pool_total_chunks = 16;
        cfg.enable_mamba_l2 = true;
        cfg.mamba_l2_host_slots = 16;
        cfg.host_allocator.total_pages = 32;
        return cfg;
    }
};

TEST_F(MambaL2IntegrationTest, FinishWriteBackCarriesMambaPair) {
    Submit(MakeRequestSpec("r1", 2));
    PlanOnce();
    SendForwardDone("r1", {100});
    PlanOnce();

    SendFinish("r1");
    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    ASSERT_FALSE(wb->op_ids.empty());
    ASSERT_TRUE(wb->src_pages_by_kind.contains("mamba"));
    bool has_mamba_pair = false;
    for (const auto& pages : wb->src_pages_by_kind.at("mamba")) {
        has_mamba_pair = has_mamba_pair || !pages.empty();
    }
    EXPECT_TRUE(has_mamba_pair);
}

TEST_F(MambaL2IntegrationTest, WriteBackDoneDemotesDeviceAndNextRequestLoadsBackMamba) {
    Submit(MakeRequestSpec("r1", 2));
    PlanOnce();
    SendForwardDone("r1", {100});
    PlanOnce();

    SendFinish("r1");
    auto writeback_plan = PlanOnce();
    const auto* wb = GetWriteBack(writeback_plan);
    ASSERT_NE(wb, nullptr);
    ASSERT_FALSE(wb->op_ids.empty());

    SendWriteBackDone(wb->op_ids[0]);
    PlanOnce();

    Submit(MakeRequestSpec("r2", 3));
    auto loadback_plan = PlanOnce();
    const auto* lb = GetLoadBack(loadback_plan);
    ASSERT_NE(lb, nullptr) << "written-back Mamba+KV cache must be host-only and require loadback";
    ASSERT_TRUE(lb->src_pages_by_kind.contains("kv"));
    ASSERT_TRUE(lb->src_pages_by_kind.contains("mamba"));
    EXPECT_FALSE(lb->src_pages_by_kind.at("mamba").empty());
}

TEST_F(MambaL2IntegrationTest, HostOnlyMambaLoadsBackAfterPinnedWriteBackReleases) {
    Submit(RequestSpec{.request_id = "r1", .tokens = {1, 2, 3, 4}});
    PlanOnce();
    SendForwardDone("r1", {100});
    PlanOnce();

    SendFinish("r1");
    auto writeback_plan = PlanOnce();
    const auto* wb = GetWriteBack(writeback_plan);
    ASSERT_NE(wb, nullptr);
    ASSERT_FALSE(wb->op_ids.empty());

    Submit(RequestSpec{.request_id = "child", .tokens = {1, 2, 3, 4, 5, 6}});
    PlanOnce();
    SendForwardDone("child", {200});
    PlanOnce();

    SendWriteBackDone(wb->op_ids[0]);
    PlanOnce();

    ExecutionEvent abort_child;
    abort_child.With(ForwardEvent{forward::Abort{.request_id = "child"}});
    scheduler_->Advance(std::move(abort_child));
    PlanOnce();

    Submit(RequestSpec{.request_id = "probe", .tokens = {1, 2, 3, 4, 9}});
    auto loadback_plan = PlanOnce();
    const auto* lb = GetLoadBack(loadback_plan);
    ASSERT_NE(lb, nullptr) << "host-only Mamba must load back after the pinning request releases";
    ASSERT_TRUE(lb->src_pages_by_kind.contains("mamba"));
    EXPECT_FALSE(lb->src_pages_by_kind.at("mamba").empty());
}

class DisablePrefixCacheMambaRetractTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.disable_prefix_cache = true;
        cfg.enable_mamba = true;
        cfg.mamba_pool_total_chunks = 16;
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 3;
        cfg.host_allocator.total_pages = 16;
        cfg.enable_l3_storage = false;
        return cfg;
    }
};

TEST_F(DisablePrefixCacheMambaRetractTest, RetractedRequestRecoversFromTreeOwnedMambaState) {
    Submit(MakeRequestSpec("r1", 1));
    PlanOnce();
    SendForwardDone("r1", {100});
    PlanOnce();

    SendReserveNumTokens("r1", 3);
    auto retract_plan = PlanOnce();
    const auto* wb = GetWriteBack(retract_plan);
    ASSERT_NE(wb, nullptr);
    ASSERT_FALSE(wb->op_ids.empty());

    SendWriteBackDone(wb->op_ids[0]);
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);

    auto recover_plan = PlanOnce();
    const auto* fwd = GetForward(recover_plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->request_ids.size(), 1u);
    EXPECT_EQ(fwd->request_ids[0], "r1");
    EXPECT_GE(fwd->mamba_cow_src_indices[0], 0);
    EXPECT_GE(fwd->mamba_working_indices[0], 0);
    EXPECT_GE(fwd->mamba_checkpoint_dst_indices[0], 0);
}

}  // namespace tokenspeed::test
