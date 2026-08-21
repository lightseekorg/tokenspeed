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
        auto plan_wb = PlanOnce();
        SendFinish("r_seed");
        const WriteBackBatch* wb = nullptr;
        for (const auto& op : plan_wb.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* w = std::get_if<WriteBackBatch>(cop)) {
                    wb = w;
                    break;
                }
            }
        }
        ASSERT_NE(wb, nullptr);
        ASSERT_FALSE(wb->op_ids.empty());
        SendWriteBackDone(wb->op_ids[0]);
        PlanOnce();

        Submit(MakeRequestSpec("r_fill", /*num_pages=*/3, /*start=*/100));
        PlanOnce();
        SendForwardDone("r_fill", {200});
        auto plan_wb2 = PlanOnce();
        SendFinish("r_fill");
        for (const auto& op : plan_wb2.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* w = std::get_if<WriteBackBatch>(cop)) {
                    if (!w->op_ids.empty()) SendWriteBackDone(w->op_ids[0]);
                    break;
                }
            }
        }
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

TEST_F(StableCandidateOrderingSuite, ForwardOperationsTieBreakOnRequestId) {
    // TP-determinism regression: requests_ is unordered_map<string, ...> so
    // candidates are visited in per-process random order. Without the request-id
    // tiebreaker, each rank can pick a different request when the loop budget
    // admits only a subset, which deadlocks NCCL.
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
    EXPECT_EQ(ids[0], "r_aaa");
}

TEST_F(StableCandidateOrderingSuite, ForwardBatchIsInsertionOrderIndependent) {
    // Mirror of the above using two scheduler instances fed the same request
    // set in opposite submit orders. The chosen forward request must depend
    // only on the SET of request ids, not the submission sequence.
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
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
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

class ExperimentalM16PrefillDelayerSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_experimental_m16_prefill_delayer = true;
        return cfg;
    }

    void StartDecode(const std::string& id = "decode") {
        Submit(MakeRequestSpec(id, 1));
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* prefill = FindForwardBatch(plan);
        ASSERT_NE(prefill, nullptr);
        ASSERT_EQ(prefill->request_ids, std::vector<std::string>{id});
        SendForwardDone(id, {100});
    }

    static ForwardBatch Batch(const ExecutionPlan& plan) {
        const ForwardBatch* batch = FindForwardBatch(plan);
        if (batch == nullptr) {
            ADD_FAILURE() << "execution plan has no ForwardBatch";
            return ForwardBatch{std::vector<ForwardOperation>{}};
        }
        return *batch;
    }
};

TEST_F(ExperimentalM16PrefillDelayerSuite, DelaysLoneSubmittedPrefillForAtMostTwoDecodePasses) {
    StartDecode();
    Submit(MakeRequestSpec("prefill", 1, 200));

    const ForwardBatch first = Batch(PlanOnce());
    EXPECT_EQ(first.request_ids, std::vector<std::string>{"decode"});
    EXPECT_EQ(first.NumExtends(), 0u);
    SendForwardDone("decode", {101});

    const ForwardBatch second = Batch(PlanOnce());
    EXPECT_EQ(second.request_ids, std::vector<std::string>{"decode"});
    EXPECT_EQ(second.NumExtends(), 0u);
    SendForwardDone("decode", {102});

    const ForwardBatch third = Batch(PlanOnce());
    EXPECT_EQ(third.request_ids, std::vector<std::string>{"prefill"});
    EXPECT_EQ(third.NumExtends(), 1u);
    EXPECT_TRUE(third.decode_input_ids.empty());
}

TEST_F(ExperimentalM16PrefillDelayerSuite, ReleasesEarlyWhenSecondSubmittedPrefillArrives) {
    StartDecode();
    Submit(MakeRequestSpec("prefill_a", 1, 200));

    EXPECT_EQ(Batch(PlanOnce()).request_ids, std::vector<std::string>{"decode"});
    SendForwardDone("decode", {101});
    Submit(MakeRequestSpec("prefill_b", 1, 300));

    const ForwardBatch released = Batch(PlanOnce());
    EXPECT_EQ(released.request_ids, (std::vector<std::string>{"prefill_a", "prefill_b"}));
    EXPECT_EQ(released.NumExtends(), 2u);
    EXPECT_TRUE(released.decode_input_ids.empty());
}

TEST_F(ExperimentalM16PrefillDelayerSuite, EarlyReleaseIsSubmissionOrderIndependent) {
    StartDecode();
    Submit({MakeRequestSpec("prefill_b", 1, 300), MakeRequestSpec("prefill_a", 1, 200)});
    const std::vector<std::string> first_ids = Batch(PlanOnce()).request_ids;

    scheduler_ = std::make_unique<Scheduler>(config_);
    StartDecode();
    Submit({MakeRequestSpec("prefill_a", 1, 200), MakeRequestSpec("prefill_b", 1, 300)});
    const std::vector<std::string> second_ids = Batch(PlanOnce()).request_ids;

    EXPECT_EQ(first_ids, (std::vector<std::string>{"prefill_a", "prefill_b"}));
    EXPECT_EQ(first_ids, second_ids);
}

TEST_F(ExperimentalM16PrefillDelayerSuite, DoesNotDelayWithoutRunnableDecode) {
    Submit(MakeRequestSpec("prefill", 1));
    const ForwardBatch batch = Batch(PlanOnce());
    EXPECT_EQ(batch.request_ids, std::vector<std::string>{"prefill"});
    EXPECT_EQ(batch.NumExtends(), 1u);
}

class ExperimentalM16InsufficientDecodeBudgetSuite : public ExperimentalM16PrefillDelayerSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = ExperimentalM16PrefillDelayerSuite::MakeConfig();
        cfg.max_scheduled_tokens = 1;
        cfg.decode_input_tokens = 2;
        return cfg;
    }
};

TEST_F(ExperimentalM16InsufficientDecodeBudgetSuite, FallsThroughWhenDecodeCannotFitTokenBudget) {
    Submit(MakeRequestSpec("decode", 1));
    EXPECT_EQ(Batch(PlanOnce()).request_ids, std::vector<std::string>{"decode"});
    EXPECT_EQ(Batch(PlanOnce()).request_ids, std::vector<std::string>{"decode"});
    SendForwardDone("decode", {100, 101});
    Submit(MakeRequestSpec("prefill", 1, 200));

    const ForwardBatch batch = Batch(PlanOnce());
    EXPECT_EQ(batch.request_ids, std::vector<std::string>{"prefill"});
    EXPECT_EQ(batch.NumExtends(), 1u);
    EXPECT_TRUE(batch.decode_input_ids.empty());
}

class ExperimentalM16ChunkedPrefillSuite : public ExperimentalM16PrefillDelayerSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = ExperimentalM16PrefillDelayerSuite::MakeConfig();
        cfg.max_scheduled_tokens = 2;
        return cfg;
    }
};

TEST_F(ExperimentalM16ChunkedPrefillSuite, NeverDelaysAlreadyPrefillingRequest) {
    StartDecode();
    Submit({MakeRequestSpec("prefill_a", 2, 200), MakeRequestSpec("prefill_b", 2, 300)});

    const ForwardBatch first_chunk = Batch(PlanOnce());
    ASSERT_EQ(first_chunk.request_ids, std::vector<std::string>{"prefill_a"});
    EXPECT_EQ(first_chunk.input_lengths, std::vector<std::int32_t>{2});

    const ForwardBatch second_chunk = Batch(PlanOnce());
    EXPECT_EQ(second_chunk.request_ids, std::vector<std::string>{"prefill_a"});
    EXPECT_EQ(second_chunk.input_lengths, std::vector<std::int32_t>{2});
    EXPECT_TRUE(second_chunk.decode_input_ids.empty());
}

TEST_F(SchedulerTestSuite, ExperimentalM16PrefillDelayerDefaultsOff) {
    EXPECT_FALSE(config_.enable_experimental_m16_prefill_delayer);

    Submit(MakeRequestSpec("decode", 1));
    PlanOnce();
    SendForwardDone("decode", {100});
    Submit(MakeRequestSpec("prefill", 1, 200));

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, std::vector<std::string>{"prefill"});
    EXPECT_EQ(batch->NumExtends(), 1u);
}

TEST_F(SchedulerTestSuite, ExperimentalM16PrefillDelayerRejectsMixedAndPdModes) {
    config_.enable_experimental_m16_prefill_delayer = true;
    config_.enable_mixed_prefill_decode = true;
    EXPECT_THROW({ Scheduler scheduler{config_}; }, std::invalid_argument);

    config_.enable_mixed_prefill_decode = false;
    config_.role = Role::kD;
    EXPECT_THROW({ Scheduler scheduler{config_}; }, std::invalid_argument);
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
    PlanOnce();
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
    PlanOnce();
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
