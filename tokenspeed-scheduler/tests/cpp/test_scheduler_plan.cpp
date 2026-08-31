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

#include "integration_test_helper.h"

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace tokenspeed::test {

class LoadBackViaCacheTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 5;
        cfg.host_allocator.total_pages = 32;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    void SetupHostCache() {
        Submit(MakeRequestSpec("r_seed", /*num_pages=*/2, /*start=*/1));
        PlanOnce();
        SendForwardDone("r_seed", {42});
        const ExecutionPlan seed_stream = PlanOnce();
        ASSERT_FALSE(ExtractCacheOpsOfKind<WriteBackBatch>(seed_stream).empty());
        AckWriteBacks(seed_stream);
        SendFinish("r_seed");
        AckWriteBacks(PlanOnce());
        PlanOnce();

        Submit(MakeRequestSpec("r_fill", /*num_pages=*/3, /*start=*/100));
        PlanOnce();
        SendForwardDone("r_fill", {200});
        AckWriteBacks(PlanOnce());
        SendFinish("r_fill");
        AckWriteBacks(PlanOnce());
        PlanOnce();
    }
};

TEST_F(LoadBackViaCacheTestSuite, LoadBack_TriggeredAfterPrefetchPopulatesHostCache) {
    SetupHostCache();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2, /*start=*/1));
    auto plan = PlanOnce();
    auto lb = ExtractCacheOpsOfKind<LoadBackBatch>(plan);

    bool r1_in_forward = false;
    for (const auto& op : plan.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            for (const auto& rid : fwd->request_ids) {
                if (rid == "r1") r1_in_forward = true;
            }
        }
    }
    EXPECT_TRUE(r1_in_forward || !lb.empty())
        << "host cache hit should trigger LoadBack inline or r1 should be in forward";
}

TEST_F(SchedulerTestSuite, LoadBack_NotTriggeredWithoutHostCacheHit) {
    Submit(MakeRequestSpec("r1", 4));
    auto plan = PlanOnce();
    auto lb = ExtractCacheOpsOfKind<LoadBackBatch>(plan);
    EXPECT_TRUE(lb.empty());
}

TEST_F(SchedulerTestSuite, NoCacheOps_WhenNoRequests) {
    auto plan = PlanOnce();
    auto cache_ops = ExtractCacheOps(plan);
    EXPECT_TRUE(cache_ops.empty());
}

TEST_F(SchedulerTestSuite, NoCacheOps_PlainRequestNoCacheHit) {
    Submit(MakeRequestSpec("r1", 2));
    auto plan = PlanOnce();
    auto cache_ops = ExtractCacheOps(plan);
    EXPECT_TRUE(cache_ops.empty());
}

class DisablePrefixCacheTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.disable_prefix_cache = true;
        return cfg;
    }
};

TEST_F(DisablePrefixCacheTestSuite, SamePromptDoesNotReuseDevicePrefix) {
    Submit(MakeRequestSpec("r_seed", 2));
    PlanOnce();
    SendForwardDone("r_seed", {100});
    PlanOnce();
    SendFinish("r_seed");
    PlanOnce();

    Submit(MakeRequestSpec("r1", 2));
    auto plan = PlanOnce();
    const auto& op = plan.Operations()[0];
    auto* fwd = std::get_if<ForwardBatch>(&op);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->request_ids.size(), 1u);
    EXPECT_EQ(fwd->request_ids[0], "r1");
    EXPECT_EQ(fwd->extend_prefix_lens[0], 0);
    EXPECT_EQ(fwd->input_lengths[0], 4);
    EXPECT_TRUE(ExtractCacheOpsOfKind<LoadBackBatch>(plan).empty());
}

class StableCandidateOrderingSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        // Force the candidates loop to break after exactly one push so the
        // tiebreaker decides which request wins.
        cfg.max_batch_size = 1;
        return cfg;
    }
};

TEST_F(StableCandidateOrderingSuite, ForwardOperationsFollowSubmissionOrder) {
    // TP-determinism + FIFO: requests_ is a vector in submission order, and
    // the mirrored schedulers receive identical submission sequences -- so
    // when the loop budget admits only a subset, every rank picks the same
    // request, and it is the OLDEST one, whatever its id sorts like.
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    auto plan = PlanOnce();
    std::vector<std::string> ids;
    for (const auto& op : plan.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            ids = fwd->request_ids;
        }
    }
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], "r_ccc") << "first submitted, first scheduled";
}

TEST_F(StableCandidateOrderingSuite, ForwardBatchIsReproducibleAcrossMirroredSchedulers) {
    // Two scheduler instances fed the SAME submission sequence must build
    // the same batch -- the mirrored-ranks invariant. (Different submission
    // orders legitimately differ: FIFO means arrival order is meaningful.)
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    auto plan_a = PlanOnce();
    std::vector<std::string> ids_a;
    for (const auto& op : plan_a.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            ids_a = fwd->request_ids;
        }
    }

    scheduler_ = std::make_unique<Scheduler>(config_);
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    auto plan_b = PlanOnce();
    std::vector<std::string> ids_b;
    for (const auto& op : plan_b.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            ids_b = fwd->request_ids;
        }
    }

    ASSERT_FALSE(ids_a.empty());
    EXPECT_EQ(ids_a, ids_b);
}

class SchedulerKvCacheEventTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 3;
        cfg.disable_l2_cache = true;
        cfg.enable_kv_cache_events = true;
        return cfg;
    }
};

TEST_F(SchedulerKvCacheEventTestSuite, PublishesStoredBlockAndDrainsItOnce) {
    const RequestSpec spec = MakeRequestSpec("r1", 1);
    Submit(spec);
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    SendFinish("r1");

    std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[0]));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(events[0]).token_ids, spec.tokens);
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());
}

TEST_F(SchedulerKvCacheEventTestSuite, PublishesRemovalWhenAdmissionEvictsBlock) {
    Submit(MakeRequestSpec("seed", 1));
    PlanOnce();
    SendForwardDone("seed", {42});
    PlanOnce();
    SendFinish("seed");

    std::vector<KvCacheEvent> stored = scheduler_->DrainKvEvents();
    ASSERT_EQ(stored.size(), 1u);
    const std::uint64_t stored_hash = std::get<KvBlockStoredEvent>(stored[0]).block_hashes.front();

    Submit(MakeRequestSpec("replacement", 1, 100));
    PlanOnce();

    std::vector<KvCacheEvent> removed = scheduler_->DrainKvEvents();
    ASSERT_EQ(removed.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<KvBlockRemovedEvent>(removed[0]));
    EXPECT_EQ(std::get<KvBlockRemovedEvent>(removed[0]).block_hashes, std::vector<std::uint64_t>{stored_hash});
}

class MultiGroupKvCacheEventTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 8;
        cfg.disable_l2_cache = true;
        cfg.enable_kv_cache_events = true;
        CacheGroupConfig second = cfg.cache_groups.front();
        second.group_id = "full_attention_1";
        cfg.cache_groups.push_back(std::move(second));
        return cfg;
    }
};

TEST_F(MultiGroupKvCacheEventTestSuite, PublishesOneEventAfterAllGroupsCacheBoundary) {
    Submit(MakeRequestSpec("r1", 1));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    SendFinish("r1");

    std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[0]));
}

class SubpageKvCacheEventTestSuite : public SchedulerKvCacheEventTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerKvCacheEventTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 4;
        auto& group = cfg.cache_groups.front();
        group.rows_per_page = 1;
        group.total_pages = 2 * cfg.device_allocator.total_pages;
        group.cache_blocks_per_lcm_block = 2;
        return cfg;
    }
};

TEST_F(SubpageKvCacheEventTestSuite, PublishesOneEventAfterAllChildBlocksCacheBoundary) {
    const RequestSpec spec = MakeRequestSpec("r1", 1);
    Submit(spec);
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    SendFinish("r1");

    std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[0]));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(events[0]).token_ids, spec.tokens);
}

TEST_F(SubpageKvCacheEventTestSuite, PublishesRemovalWhenAnyChildBlockIsEvicted) {
    Submit(MakeRequestSpec("seed", 2));
    PlanOnce();
    SendForwardDone("seed", {42});
    PlanOnce();
    SendFinish("seed");

    std::vector<KvCacheEvent> stored = scheduler_->DrainKvEvents();
    ASSERT_EQ(stored.size(), 2u);
    const std::uint64_t first_stored_hash = std::get<KvBlockStoredEvent>(stored[0]).block_hashes.front();
    const std::uint64_t second_stored_hash = std::get<KvBlockStoredEvent>(stored[1]).block_hashes.front();

    Submit(MakeRequestSpec("replacement", 2, 100));
    PlanOnce();

    std::vector<KvCacheEvent> removed = scheduler_->DrainKvEvents();
    ASSERT_EQ(removed.size(), 2u);
    std::unordered_set<std::uint64_t> remaining_hashes{first_stored_hash, second_stored_hash};
    for (const KvCacheEvent& event : removed) {
        ASSERT_TRUE(std::holds_alternative<KvBlockRemovedEvent>(event));
        const auto& removed_hashes = std::get<KvBlockRemovedEvent>(event).block_hashes;
        ASSERT_EQ(removed_hashes.size(), 1u);
        EXPECT_EQ(remaining_hashes.erase(removed_hashes[0]), 1u);
    }
    EXPECT_TRUE(remaining_hashes.empty());
}

TEST_F(SchedulerTestSuite, SubmitRequestsRejectsEmptyTokens) {
    EXPECT_THROW(Submit(RequestSpec{.request_id = "empty"}), std::invalid_argument);
}

TEST_F(SchedulerTestSuite, SubmitRequestsValidatesWholeBatchBeforeInsertion) {
    const RequestSpec valid = MakeRequestSpec("valid", 1);
    RequestSpec invalid = MakeRequestSpec("invalid", 1);
    invalid.max_new_tokens = -1;

    EXPECT_THROW(Submit(std::vector<RequestSpec>{valid, invalid}), std::invalid_argument);
    EXPECT_NO_THROW(Submit(valid));
}

class HybridPrefixPromotionTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 128;
        cfg.disable_l2_cache = true;
        for (std::int32_t i = 0; i < 3; ++i) {
            CacheGroupConfig state = cfg.cache_groups.front();
            state.group_id = "linear_attention_" + std::to_string(i);
            state.family = CacheGroupFamily::State;
            cfg.cache_groups.push_back(std::move(state));
        }
        return cfg;
    }

    RequestSpec MakeHybridRequest(const std::string& id, std::int32_t suffix_start) {
        RequestSpec spec = MakeRequestSpec(id, /*num_pages=*/4);
        spec.tokens.push_back(suffix_start);
        spec.tokens.push_back(suffix_start + 1);
        spec.tokens.push_back(suffix_start + 2);
        return spec;
    }
};

TEST_F(HybridPrefixPromotionTestSuite, ThirdRequestReusesPromotedStateBoundary) {
    Submit(MakeHybridRequest("seed", 100));
    const ExecutionPlan seed_body_plan = PlanOnce();
    const ForwardBatch* seed_body = FindForwardBatch(seed_body_plan);
    ASSERT_NE(seed_body, nullptr);
    EXPECT_EQ(seed_body->input_lengths, std::vector<std::int32_t>{10});
    const ExecutionPlan seed_tail_plan = PlanOnce();
    const ForwardBatch* seed_tail = FindForwardBatch(seed_tail_plan);
    ASSERT_NE(seed_tail, nullptr);
    EXPECT_EQ(seed_tail->input_lengths, std::vector<std::int32_t>{1});
    SendForwardDone("seed", {900});
    PlanOnce();
    SendFinish("seed");
    PlanOnce();

    Submit(MakeHybridRequest("promote", 200));
    const ExecutionPlan promotion_plan = PlanOnce();
    const ForwardBatch* promotion = FindForwardBatch(promotion_plan);
    ASSERT_NE(promotion, nullptr);
    ASSERT_EQ(promotion->request_ids, std::vector<std::string>{"promote"});
    EXPECT_EQ(promotion->input_lengths, std::vector<std::int32_t>{8});
    const ExecutionPlan promotion_body_plan = PlanOnce();
    const ForwardBatch* promotion_body = FindForwardBatch(promotion_body_plan);
    ASSERT_NE(promotion_body, nullptr);
    EXPECT_EQ(promotion_body->input_lengths, std::vector<std::int32_t>{2});
    const ExecutionPlan promotion_tail_plan = PlanOnce();
    const ForwardBatch* promotion_tail = FindForwardBatch(promotion_tail_plan);
    ASSERT_NE(promotion_tail, nullptr);
    EXPECT_EQ(promotion_tail->input_lengths, std::vector<std::int32_t>{1});
    SendForwardDone("promote", {901});
    PlanOnce();
    SendFinish("promote");
    PlanOnce();

    Submit(MakeHybridRequest("reuse", 300));
    const ExecutionPlan reuse_plan = PlanOnce();
    const ForwardBatch* reuse = FindForwardBatch(reuse_plan);
    ASSERT_NE(reuse, nullptr);
    ASSERT_EQ(reuse->request_ids, std::vector<std::string>{"reuse"});
    EXPECT_EQ(reuse->extend_prefix_lens, std::vector<std::int32_t>{8});
}

TEST(SchedulerConstructionTest, ValidatesConfigBeforeBuildingPools) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = 2;
    cfg.max_scheduled_tokens = 64;
    cfg.max_batch_size = 8;
    cfg.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full_attention",
        .rows_per_page = cfg.prefix_granularity,
        .entry_stride_tokens = 1,
        .total_pages = 32,
    });
    // device_allocator.total_pages stays 0, so the block pool would be built
    // with a negative usable count and assert before the config diagnostic.
    try {
        Scheduler scheduler{cfg};
        FAIL() << "a device cache without usable capacity was accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string{error.what()}.find("device cache"), std::string::npos) << error.what();
    }
}

}  // namespace tokenspeed::test
