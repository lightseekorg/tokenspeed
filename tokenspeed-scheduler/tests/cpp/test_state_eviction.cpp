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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {

class StatePublicationSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 4;
        cfg.device_allocator.total_pages = 257;
        cfg.host_allocator.total_pages = 257;
        cfg.max_scheduled_tokens = 8;
        cfg.max_batch_size = 1;
        cfg.disable_l2_cache = true;
        for (const std::string& id : {"full", "state0", "state1", "state2"}) {
            cfg.cache_groups.push_back(CacheGroupConfig{
                .group_id = id,
                .block_granularity = 4,
                .total_pages = cfg.device_allocator.total_pages,
                .retention = CacheGroupConfig::Retention::FullHistory,
                .family = id == "full" ? CacheGroupFamily::History : CacheGroupFamily::State,
            });
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void Reset(bool host_cache, std::int32_t decode_width, std::int32_t overlap_depth) {
        config_ = MakeConfig();
        config_.disable_l2_cache = !host_cache;
        config_.decode_input_tokens = decode_width;
        config_.overlap_schedule_depth = overlap_depth;
        scheduler_ = std::make_unique<Scheduler>(config_);
    }

    RequestSpec RequestWithTokens(const std::string& id, std::vector<std::int32_t> tokens) const {
        return RequestSpec{.request_id = id, .tokens = std::move(tokens), .max_new_tokens = 32};
    }

    std::int32_t ResidentBlocks() const {
        return config_.device_allocator.NumUsableBlocks() - scheduler_->EmptyLcmBlocks();
    }

    std::int32_t StateResidentBlocks(const ForwardBatch& batch) const {
        const auto& history = batch.block_tables.at("full").at(0);
        const auto history_blocks =
            std::count_if(history.begin(), history.end(), [](std::int32_t page) { return page > 0; });
        return ResidentBlocks() - static_cast<std::int32_t>(history_blocks);
    }

    // Decodes `survivor` until no unpinned Device cache entry is left: its
    // growth evicts the suspended victim's still-cached blocks, so the restore
    // that follows must copy them back from Host instead of claiming them. (A
    // flush is refused while a request is suspended, so this is the way to
    // reach that state.) Streams the survivor publishes are acknowledged as
    // they appear.
    void DecodeUntilDeviceCacheIsConsumed(const std::string& survivor, std::int32_t& next_token) {
        for (std::int32_t round = 0; round < 64 && scheduler_->AvailableLcmBlocks() > 0; ++round) {
            const ExecutionPlan plan = PlanOnce();
            AckWriteBacks(plan);
            const ForwardBatch* batch = FindForwardBatch(plan);
            ASSERT_NE(batch, nullptr);
            if (std::ranges::find(batch->request_ids, survivor) != batch->request_ids.end()) {
                SendForwardDone(survivor, {next_token++});
            }
        }
        ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0) << "the survivor took every evictable block";
    }

    static std::int32_t StateStoreCount(const ExecutionPlan& plan) {
        std::int32_t count = 0;
        for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
            for (const auto& group_ids : std::get<WriteBackBatch>(operation).group_ids) {
                count += static_cast<std::int32_t>(
                    std::count_if(group_ids.begin(), group_ids.end(), [](std::uint32_t group) { return group > 0; }));
            }
        }
        return count;
    }

    // Fresh prompts only. Each plan is retained so callers can inspect every
    // store, including a checkpoint published before a sub-page final tail.
    void Prefill(const RequestSpec& request, std::vector<ExecutionPlan>& plans) {
        Submit(request);
        std::int32_t computed = 0;
        while (computed < static_cast<std::int32_t>(request.tokens.size())) {
            ExecutionPlan plan = PlanOnce();
            const ForwardBatch* batch = FindForwardBatch(plan);
            ASSERT_NE(batch, nullptr);
            ASSERT_EQ(batch->request_ids, std::vector<std::string>{request.request_id});
            ASSERT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{computed});
            ASSERT_GT(batch->input_lengths.at(0), 0);
            computed += batch->input_lengths.at(0);
            AckWriteBacks(plan);
            SendForwardDone(request.request_id, computed == static_cast<std::int32_t>(request.tokens.size())
                                                    ? std::vector<std::int32_t>{101}
                                                    : std::vector<std::int32_t>{});
            plans.push_back(std::move(plan));
        }
    }

    void ExpectReplay(const std::string& id, std::vector<std::int32_t> tokens, std::int32_t expected_prefix) {
        ExpectReplay(id, std::move(tokens), expected_prefix, 32);
    }

    void ExpectReplay(const std::string& id, std::vector<std::int32_t> tokens, std::int32_t expected_prefix,
                      std::int32_t max_new_tokens) {
        RequestSpec request = RequestWithTokens(id, std::move(tokens));
        request.max_new_tokens = max_new_tokens;
        Submit(request);
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{id});
        EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{expected_prefix});
        for (const CacheOperation& operation : ExtractCacheOpsOfKind<LoadBackBatch>(plan)) {
            for (std::uint32_t op_id : std::get<LoadBackBatch>(operation).op_ids) {
                SendLoadBackDone(op_id, /*success=*/true);
            }
        }
        AckWriteBacks(plan);
        SendAbortEvent(id);
    }

    static std::vector<std::int32_t> ConversationPrefix(std::int32_t computed_tokens) {
        std::vector<std::int32_t> tokens = MakeTokens(4, 1);
        const std::vector<std::int32_t> response = MakeTokens(computed_tokens - 4, 101);
        tokens.insert(tokens.end(), response.begin(), response.end());
        tokens.push_back(999);
        return tokens;
    }
};

TEST_F(StatePublicationSuite, ReclaimDropsOldPrefillStateWithoutCapacityPressure) {
    Submit(RequestWithTokens("source", MakeTokens(40, 1)));
    const ExecutionPlan first = PlanOnce();
    ASSERT_NE(FindForwardBatch(first), nullptr);
    SendForwardDone("source", {});
    const ExecutionPlan second = PlanOnce();
    ASSERT_NE(FindForwardBatch(second), nullptr);
    SendForwardDone("source", {});

    const ExecutionPlan third = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(third);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{16});
    ASSERT_GT(scheduler_->EmptyLcmBlocks(), 200);
    EXPECT_EQ(StateResidentBlocks(*batch), 6) << "expired input is released even while the pool has ample capacity";
    for (const std::string& group : {"state0", "state1", "state2"}) {
        const auto& row = batch->block_tables.at(group).at(0);
        EXPECT_EQ(row.at(1), 0) << "the token-8 input has expired";
        EXPECT_GT(row.at(3), 0) << "token-16 is still this chunk's input";
        EXPECT_GT(row.at(5), 0) << "token-24 is this chunk's output";
    }
}

TEST_F(StatePublicationSuite, CapacityPressureReusesOldPrefillBlocksAndKeepsTheFinalResumeBoundary) {
    config_.device_allocator.total_pages = 33;
    config_.max_scheduled_tokens = 4;
    for (CacheGroupConfig& group : config_.cache_groups) {
        group.total_pages = config_.device_allocator.total_pages;
    }
    scheduler_ = std::make_unique<Scheduler>(config_);
    RequestSpec source = RequestWithTokens("source", MakeTokens(62, 1));
    source.max_new_tokens = 4;
    std::vector<ExecutionPlan> plans;
    Prefill(source, plans);
    ASSERT_EQ(plans.size(), 16u);

    const ForwardBatch* first = FindForwardBatch(plans.front());
    const ForwardBatch* third = FindForwardBatch(plans.at(2));
    ASSERT_NE(first, nullptr);
    ASSERT_NE(third, nullptr);
    for (const std::string& group : {"state0", "state1", "state2"}) {
        const std::int32_t original = first->block_tables.at(group).at(0).at(0);
        ASSERT_GT(original, 0);
        EXPECT_EQ(third->block_tables.at(group).at(0).at(0), 0) << "the expired request-local chunk has been released";
        bool reused = false;
        for (const ExecutionPlan& plan : plans) {
            const ForwardBatch* batch = FindForwardBatch(plan);
            ASSERT_NE(batch, nullptr);
            const std::int32_t end = batch->extend_prefix_lens.at(0) + batch->input_lengths.at(0);
            if (end > 12) {
                reused |= batch->block_tables.at(group).at(0).at((end - 1) / 4) == original;
            }
        }
        EXPECT_TRUE(reused) << group << " must recycle its first request-local chunk under real pool pressure";
    }

    SendFinish("source");
    PlanOnce();
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0);
    ExpectReplay("evicted_chunk", MakeTokens(5, 1), 0, 4);
    auto resume = source.tokens;
    resume.push_back(999);
    ExpectReplay("final_boundary", std::move(resume), 60, 4);
}

TEST_F(StatePublicationSuite, FinalPrefillBoundarySurvivesAlignedAndUnalignedEndings) {
    for (std::int32_t length : {24, 26, 30}) {
        SCOPED_TRACE(length);
        Reset(false, 1, 0);
        const RequestSpec request = RequestWithTokens("source", MakeTokens(length, 1));
        std::vector<ExecutionPlan> plans;
        Prefill(request, plans);
        SendFinish("source");
        PlanOnce();

        EXPECT_EQ(ResidentBlocks(), length / 4 + 3);
        ExpectReplay("earlier_chunk", MakeTokens(9, 1), 0);
        auto replay_tokens = request.tokens;
        replay_tokens.push_back(999);
        ExpectReplay("next_turn", std::move(replay_tokens), length / 4 * 4);
    }
}

TEST_F(StatePublicationSuite, HostStoresOnlyFinalPrefillSnapshotIncludingShortTail) {
    for (std::int32_t length : {26, 30}) {
        SCOPED_TRACE(length);
        Reset(true, 1, 0);
        const RequestSpec request = RequestWithTokens("source", MakeTokens(length, 1));
        std::vector<ExecutionPlan> plans;
        Prefill(request, plans);
        SendFinish("source");
        plans.push_back(PlanOnce());
        AckWriteBacks(plans.back());
        std::int32_t state_stores = 0;
        for (const ExecutionPlan& plan : plans) {
            state_stores += StateStoreCount(plan);
        }
        EXPECT_EQ(state_stores, 3) << "ordinary computed Chunks must not be queued for L2";
        EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), length / 4 + 3);
        ASSERT_TRUE(scheduler_->ClearL1Cache());
        auto replay_tokens = request.tokens;
        replay_tokens.push_back(999);
        ExpectReplay("host_replay", std::move(replay_tokens), length / 4 * 4);
    }
}

TEST_F(StatePublicationSuite, DecodeAndFinishKeepOnlyThePrefillStateBoundary) {
    Reset(false, 1, 0);
    std::vector<ExecutionPlan> plans;
    Prefill(RequestWithTokens("source", MakeTokens(4, 1)), plans);
    PlanOnce();
    for (std::int32_t token = 102; token <= 110; ++token) {
        SendForwardDone("source", {token});
        PlanOnce();
    }
    // The last schedule has already hashed the completed prefix pages;
    // Finish adds no hash and must not supplement a state snapshot.
    SendFinish("source");
    PlanOnce();
    EXPECT_EQ(ResidentBlocks(), 3 + 3);
    ExpectReplay("old_boundary", ConversationPrefix(8), 4);
    ExpectReplay("generated_boundary", ConversationPrefix(12), 4);
}

TEST_F(StatePublicationSuite, FinishAddsHistoryButNoDecodeStateToHost) {
    Reset(true, 1, 0);
    std::vector<ExecutionPlan> plans;
    Prefill(RequestWithTokens("source", MakeTokens(4, 1)), plans);
    const ExecutionPlan first_decode = PlanOnce();
    EXPECT_EQ(StateStoreCount(first_decode), 3) << "the final prompt snapshot keeps its ordinary store";
    AckWriteBacks(first_decode);
    for (std::int32_t token = 102; token <= 110; ++token) {
        SendForwardDone("source", {token});
        const ExecutionPlan decode = PlanOnce();
        EXPECT_EQ(StateStoreCount(decode), 0) << "decode snapshots are not streamed to L2";
        AckWriteBacks(decode);
    }
    SendFinish("source");
    const ExecutionPlan finish = PlanOnce();
    EXPECT_EQ(StateStoreCount(finish), 0) << "finish does not create a decode Endpoint";
    AckWriteBacks(finish);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6) << "three history pages plus the prefill state";
    ASSERT_TRUE(scheduler_->ClearL1Cache());
    ExpectReplay("old_host_boundary", ConversationPrefix(8), 4);
    ExpectReplay("generated_host_boundary", ConversationPrefix(12), 4);
}

TEST_F(StatePublicationSuite, DecodeReclaimsExpiredWorkingSlotsWithoutKeepingALatestSnapshot) {
    Reset(false, 1, 0);
    std::vector<ExecutionPlan> plans;
    Prefill(RequestWithTokens("source", MakeTokens(4, 1)), plans);
    PlanOnce();
    for (std::int32_t token = 102; token <= 113; ++token) {
        SendForwardDone("source", {token});
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        if (token >= 110) {
            for (const std::string& group : {"state0", "state1", "state2"}) {
                EXPECT_EQ(batch->block_tables.at(group).at(0).at(1), 0);
            }
        }
    }
    SendAbortEvent("source");
    ExpectReplay("decode_is_not_reusable", ConversationPrefix(12), 4);
}

TEST_F(StatePublicationSuite, AbortReleasesConsumedPrefillStateWithoutPublishingItsOutput) {
    Submit(RequestWithTokens("source", MakeTokens(40, 1)));
    PlanOnce();
    SendForwardDone("source", {});
    const ExecutionPlan second = PlanOnce();
    ASSERT_NE(FindForwardBatch(second), nullptr);
    SendForwardDone("source", {});
    SendAbortEvent("source");
    EXPECT_EQ(ResidentBlocks(), 2) << "only the already published token-8 history remains";
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0);
    ExpectReplay("completed_chunk", MakeTokens(17, 1), 0);
}

TEST_F(StatePublicationSuite, AlignedDecodeFeedbackThenAbortDoesNotPublishItsNewSnapshot) {
    Reset(true, 1, 0);
    std::vector<ExecutionPlan> plans;
    Prefill(RequestWithTokens("source", MakeTokens(4, 1)), plans);
    const ExecutionPlan first_decode = PlanOnce();
    ASSERT_EQ(StateStoreCount(first_decode), 3);
    AckWriteBacks(first_decode);
    for (std::int32_t token = 102; token <= 104; ++token) {
        SendForwardDone("source", {token});
        const ExecutionPlan decode = PlanOnce();
        ASSERT_NE(FindForwardBatch(decode), nullptr);
        EXPECT_EQ(StateStoreCount(decode), 0);
        AckWriteBacks(decode);
    }
    ASSERT_EQ(scheduler_->RequestTokenSize("source"), 8);

    // A sanitized result and Abort in one packet must expose no decode checkpoint.
    ExecutionEvent terminated;
    terminated.With(forward::ExtendResult{
        .request_id = "source",
        .tokens = {105},
    });
    terminated.With(forward::Abort{.request_id = "source"});
    scheduler_->Advance(terminated);
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_EQ(ResidentBlocks(), 4) << "only the valid prompt's history and three state blocks remain";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 4);
    const ExecutionPlan after_abort = PlanOnce();
    const ForwardBatch* idle = FindForwardBatch(after_abort);
    ASSERT_NE(idle, nullptr);
    EXPECT_TRUE(idle->request_ids.empty());
    EXPECT_EQ(StateStoreCount(after_abort), 0);

    ExpectReplay("new_snapshot_is_not_reusable", ConversationPrefix(8), 4);
    ExpectReplay("valid_prompt_is_reusable", ConversationPrefix(4), 4);
}

TEST_F(StatePublicationSuite, OverlapKeepsInputUntilItsLastScheduledConsumer) {
    Reset(false, 1, 1);
    Submit(RequestWithTokens("source", MakeTokens(40, 1)));
    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* first_batch = FindForwardBatch(first);
    ASSERT_NE(first_batch, nullptr);
    const ExecutionPlan second = PlanOnce();
    const ForwardBatch* second_batch = FindForwardBatch(second);
    ASSERT_NE(second_batch, nullptr);
    for (const std::string& group : {"state0", "state1", "state2"}) {
        EXPECT_EQ(second_batch->block_tables.at(group).at(0).at(1), first_batch->block_tables.at(group).at(0).at(1));
    }
    SendForwardDone("source", {});
    const ExecutionPlan third = PlanOnce();
    const ForwardBatch* third_batch = FindForwardBatch(third);
    ASSERT_NE(third_batch, nullptr);
    EXPECT_EQ(StateResidentBlocks(*third_batch), 6);
    for (const std::string& group : {"state0", "state1", "state2"}) {
        EXPECT_EQ(third_batch->block_tables.at(group).at(0).at(1), 0);
        EXPECT_EQ(third_batch->block_tables.at(group).at(0).at(3), second_batch->block_tables.at(group).at(0).at(3));
    }
}

TEST_F(StatePublicationSuite, SpeculationPublishesNeitherCrossedNorAlignedDecodeBoundaries) {
    Reset(false, 3, 0);
    std::vector<ExecutionPlan> plans;
    const RequestSpec request = RequestWithTokens("source", MakeTokens(6, 1));
    Prefill(request, plans);
    PlanOnce();
    SendForwardDone("source", {102, 103, 104});  // state advances from 6 to 9, not 8
    PlanOnce();
    SendForwardDone("source", {105, 106, 107});  // a real token-12 checkpoint
    SendFinish("source");
    PlanOnce();

    auto skipped = request.tokens;
    skipped.insert(skipped.end(), {101, 102, 999});
    ExpectReplay("skipped_boundary", std::move(skipped), 4);
    auto aligned_decode = request.tokens;
    aligned_decode.insert(aligned_decode.end(), {101, 102, 103, 104, 105, 106, 999});
    ExpectReplay("actual_boundary", std::move(aligned_decode), 4);
}

TEST_F(StatePublicationSuite, TruncatedSpeculativeOutputDoesNotInventAnAlignedState) {
    Reset(false, 3, 0);
    std::vector<ExecutionPlan> plans;
    Prefill(RequestWithTokens("source", MakeTokens(4, 1)), plans);
    PlanOnce();
    SendForwardDone("source", {102, 103, 104});  // actual state endpoint 7
    PlanOnce();
    ExecutionEvent truncated;
    truncated.With(forward::ExtendResult{
        .request_id = "source",
        .tokens = {105},
    });
    scheduler_->Advance(truncated);  // the visible endpoint is aligned, regardless of the GPU's accepted count
    SendFinish("source");
    PlanOnce();
    ExpectReplay("no_token8_state", ConversationPrefix(8), 4);
}

TEST_F(StatePublicationSuite, OverlappedSpeculativeDecodeKeepsOnlyThePrefillCacheBoundary) {
    Reset(true, 3, 1);
    Submit(RequestWithTokens("source", MakeTokens(4, 1)));
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    const ExecutionPlan first_decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(first_decode), nullptr);
    EXPECT_EQ(StateStoreCount(first_decode), 3);
    AckWriteBacks(first_decode);
    SendForwardDone("source", {101});

    for (const auto& tokens :
         std::vector<std::vector<std::int32_t>>{{102, 103, 104}, {105}, {106, 107, 108}, {109}, {110, 111, 112}}) {
        const ExecutionPlan next = PlanOnce();
        ASSERT_NE(FindForwardBatch(next), nullptr);
        EXPECT_EQ(StateStoreCount(next), 0);
        AckWriteBacks(next);
        SendForwardDone("source", tokens);
    }
    SendFinish("source");
    const ExecutionPlan finish = PlanOnce();
    EXPECT_EQ(StateStoreCount(finish), 0);
    AckWriteBacks(finish);
    ASSERT_TRUE(scheduler_->ClearL1Cache());
    ExpectReplay("old_host_boundary", ConversationPrefix(8), 4);
    ExpectReplay("new_host_boundary", ConversationPrefix(12), 4);
}

TEST_F(StatePublicationSuite, LiveDecodeDoesNotExposeGeneratedStateToAnotherRequest) {
    config_.max_batch_size = 2;
    scheduler_ = std::make_unique<Scheduler>(config_);
    std::vector<ExecutionPlan> plans;
    Prefill(RequestWithTokens("source", MakeTokens(4, 1)), plans);
    PlanOnce();
    for (std::int32_t token = 102; token <= 105; ++token) {
        SendForwardDone("source", {token});
        PlanOnce();
    }
    ExpectReplay("reader_during_decode", ConversationPrefix(8), 4);
    for (std::int32_t token = 106; token <= 109; ++token) {
        SendForwardDone("source", {token});
        PlanOnce();
    }
    SendFinish("source");
    PlanOnce();
    ExpectReplay("reader_after_finish", ConversationPrefix(12), 4);
}

TEST_F(StatePublicationSuite, AbortingOneProducerPreservesAnotherRequestsWorkingState) {
    Reset(false, 3, 1);
    config_.max_batch_size = 3;
    scheduler_ = std::make_unique<Scheduler>(config_);
    Submit({RequestWithTokens("a", MakeTokens(4, 1)), RequestWithTokens("b", MakeTokens(4, 1))});
    const ExecutionPlan prefill = PlanOnce();
    ASSERT_NE(FindForwardBatch(prefill), nullptr);
    ASSERT_EQ(FindForwardBatch(prefill)->request_ids, (std::vector<std::string>{"a", "b"}));
    PlanOnce();
    SendForwardDone("a", {101});
    SendForwardDone("b", {101});
    for (const auto& tokens : std::vector<std::vector<std::int32_t>>{{102, 103, 104}, {105}, {106, 107, 108}}) {
        PlanOnce();
        SendForwardDone("a", tokens);
        SendForwardDone("b", tokens);
    }
    const ExecutionPlan next = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(next);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, (std::vector<std::string>{"a", "b"}));
    for (const std::string& group : {"state0", "state1", "state2"}) {
        EXPECT_GT(batch->block_tables.at(group).at(1).at(2), 0);
    }
    SendAbortEvent("a");
    SendForwardDone("b", {109});
    const ExecutionPlan continued = PlanOnce();
    ASSERT_NE(FindForwardBatch(continued), nullptr);
    EXPECT_EQ(FindForwardBatch(continued)->request_ids, std::vector<std::string>{"b"});
    SendForwardDone("b", {110});
    SendAbortEvent("b");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    ExpectReplay("prefill_remains_reusable", ConversationPrefix(8), 4);
}

TEST_F(StatePublicationSuite, LongUnalignedDecodeKeepsOnlyItsBoundedWorkingState) {
    Reset(false, 3, 0);
    config_.max_scheduled_tokens = 4;
    scheduler_ = std::make_unique<Scheduler>(config_);
    std::vector<ExecutionPlan> plans;
    RequestSpec request = RequestWithTokens("source", MakeTokens(4, 1));
    request.max_new_tokens = 96;
    Prefill(request, plans);
    const auto send_decode_result = [&](const std::vector<std::int32_t>& tokens) {
        ExecutionEvent event;
        event.With(forward::ExtendResult{.request_id = "source", .tokens = tokens});
        event.With(forward::UpdateReserveNumTokens{
            .request_id = "source",
            .reserve_num_tokens_in_next_schedule_event = static_cast<std::int32_t>(tokens.size())});
        scheduler_->Advance(std::move(event));
    };
    PlanOnce();
    send_decode_result({102, 103, 104});
    PlanOnce();
    send_decode_result({105});

    std::int32_t next_token = 106;
    for (std::int32_t round = 0; round < 25; ++round) {
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{"source"});
        for (const std::string& group : {"state0", "state1", "state2"}) {
            const auto& row = batch->block_tables.at(group).at(0);
            if (round > 3) {
                EXPECT_EQ(row.at(1), 0);
                EXPECT_EQ(row.at(2), 0);
            }
            EXPECT_LE(std::count_if(row.begin(), row.end(), [](std::int32_t block) { return block > 0; }), 4);
        }
        EXPECT_LE(StateResidentBlocks(*batch), 15) << "prefill cache plus working slots, with no retained decode slot";
        const std::int32_t accepted = round == 0 ? 1 : 2;
        std::vector<std::int32_t> tokens;
        for (std::int32_t i = 0; i < accepted; ++i) {
            tokens.push_back(next_token++);
        }
        send_decode_result(tokens);
    }
    SendFinish("source");
    PlanOnce();
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    ExpectReplay("decode_boundary_not_cached", ConversationPrefix(8), 4);
}

TEST_F(StatePublicationSuite, IncompletePrefillRetractionPublishesItsComputedStateAndResumesFromIt) {
    Reset(true, 1, 0);
    config_.device_allocator.total_pages = 11;
    config_.max_batch_size = 2;
    config_.cache_groups.resize(2);
    for (CacheGroupConfig& group : config_.cache_groups) {
        group.total_pages = config_.device_allocator.total_pages;
    }
    SetTestSnapshotPool(config_);
    // The test knob: at the fourth plan, retract the one Prefilling request
    // between its chunks.
    config_.debug_force_retraction_interval = -4;
    scheduler_ = std::make_unique<Scheduler>(config_);

    Submit(RequestSpec{.request_id = "resident", .tokens = MakeTokens(8, 1)});
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    SendForwardDone("resident", {41});
    const ExecutionPlan first_decode = PlanOnce();
    ASSERT_EQ(StateStoreCount(first_decode), 1);
    AckWriteBacks(first_decode);
    SendForwardDone("resident", {42});
    ASSERT_EQ(ResidentBlocks(), 5);
    ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 3);

    Submit(RequestSpec{.request_id = "partial", .tokens = MakeTokens(20, 101)});
    const ExecutionPlan first_chunk = PlanOnce();
    const ForwardBatch* chunk = FindForwardBatch(first_chunk);
    ASSERT_NE(chunk, nullptr);
    ASSERT_EQ(chunk->request_ids, std::vector<std::string>{"partial"});
    ASSERT_EQ(chunk->extend_prefix_lens, std::vector<std::int32_t>{0});
    ASSERT_EQ(chunk->input_lengths, std::vector<std::int32_t>{8});
    ASSERT_EQ(chunk->prefill_lengths, std::vector<std::int32_t>{20});
    EXPECT_EQ(StateStoreCount(first_chunk), 0);
    SendForwardDone("partial", {});
    ASSERT_EQ(ResidentBlocks(), 8);

    // The knob retracts the mid-prompt prefill between its chunks while the
    // resident decodes on: its computed token-8 state is published as an
    // Endpoint and, with its two history pages, rides Host L2 as the image's
    // L2 leg; nothing is left for the snapshot pool.
    const ExecutionPlan retract = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    ASSERT_EQ(scheduler_->WaitingSize(), 1u);
    const ForwardBatch* resident_decode = FindForwardBatch(retract);
    ASSERT_NE(resident_decode, nullptr);
    ASSERT_EQ(resident_decode->request_ids, std::vector<std::string>{"resident"});
    EXPECT_EQ(StateStoreCount(retract), 1);
    std::int32_t history_stores = 0;
    for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(retract)) {
        const auto& stores = std::get<WriteBackBatch>(operation);
        for (std::size_t i = 0; i < stores.op_ids.size(); ++i) {
            EXPECT_FALSE(stores.source_pinned.at(i));
            history_stores +=
                static_cast<std::int32_t>(std::count(stores.group_ids.at(i).begin(), stores.group_ids.at(i).end(), 0u));
        }
    }
    EXPECT_EQ(history_stores, 2);
    const SnapshotStoreBatch* tail = FindSnapshotStore(retract);
    ASSERT_NE(tail, nullptr);
    EXPECT_TRUE(tail->src_pages.at(0).empty()) << "8 computed tokens end on a page boundary";
    SendForwardDone("resident", {43});
    EXPECT_FALSE(scheduler_->ClearL1Cache()) << "a flush is refused while a request is suspended with an image";
    // The image has not landed, so no restore is attempted (it would claim
    // the victim's still-cached Device blocks) while the resident's growth
    // evicts them.
    std::int32_t next_token = 44;
    DecodeUntilDeviceCacheIsConsumed("resident", next_token);
    AckImageStores(retract);
    EXPECT_GE(scheduler_->HostPoolCachedBlocks(), 6) << "the ACK publishes the image's entries beside the resident's";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 3) << "the suspended prompt pins its three Host entries";
    SendAbortEvent("resident");

    // With the victim's Device entries evicted the restore copies the three
    // published blocks back from Host -- history and the state checkpoint
    // alike -- and the prompt then runs its second chunk from the checkpoint.
    const ExecutionPlan recovery = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(recovery);
    ASSERT_NE(restore, nullptr);
    ASSERT_EQ(restore->request_ids, std::vector<std::string>{"partial"});
    EXPECT_EQ(std::count(restore->group_ids.at(0).begin(), restore->group_ids.at(0).end(), 0u), 2);
    EXPECT_EQ(std::count(restore->group_ids.at(0).begin(), restore->group_ids.at(0).end(), 1u), 1)
        << "the published state checkpoint comes back from Host";
    for (const std::uint8_t tier : restore->source_tiers.at(0)) {
        EXPECT_EQ(tier, static_cast<std::uint8_t>(HostTier::kL2));
    }
    EXPECT_TRUE(FindForwardBatch(recovery)->request_ids.empty());
    AckRestores(recovery);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);

    const ExecutionPlan resumed = PlanOnce();
    const ForwardBatch* recovered = FindForwardBatch(resumed);
    ASSERT_NE(recovered, nullptr);
    ASSERT_EQ(recovered->request_ids, std::vector<std::string>{"partial"});
    EXPECT_EQ(recovered->extend_prefix_lens, std::vector<std::int32_t>{8});
    EXPECT_EQ(recovered->input_lengths, std::vector<std::int32_t>{8});
    EXPECT_EQ(recovered->prefill_lengths, std::vector<std::int32_t>{20});
    EXPECT_GT(recovered->block_tables.at("state0").at(0).at(1), 0) << "the restored checkpoint is the chunk's input";
    AckWriteBacks(resumed);
    SendForwardDone("partial", {});
    SendAbortEvent("partial");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

TEST_F(StatePublicationSuite, PrefillDoneRetractionUsesActualPrefillEndWithSpeculativeDecodeWidth) {
    Reset(true, 5, 0);
    config_.device_allocator.total_pages = 15;
    config_.max_scheduled_tokens = 8;
    config_.max_batch_size = 3;
    config_.enable_mixed_prefill_decode = true;
    config_.cache_groups.resize(2);
    for (CacheGroupConfig& group : config_.cache_groups) {
        group.total_pages = config_.device_allocator.total_pages;
    }
    SetTestSnapshotPool(config_);
    scheduler_ = std::make_unique<Scheduler>(config_);
    Submit({RequestSpec{.request_id = "a", .tokens = MakeTokens(8, 1)},
            RequestSpec{.request_id = "b", .tokens = MakeTokens(8, 101)},
            RequestSpec{.request_id = "c", .tokens = MakeTokens(8, 201)}});
    for (const std::string& id : {"a", "b"}) {
        const ExecutionPlan prefill = PlanOnce();
        const ForwardBatch* initial_batch = FindForwardBatch(prefill);
        ASSERT_NE(initial_batch, nullptr);
        ASSERT_EQ(initial_batch->request_ids, std::vector<std::string>{id});
        SendForwardDone(id, {id == "a" ? 41 : 141});
    }

    // The mixed-mode prefill reserve leaves too little token budget for a
    // decode, so the third request forces a retraction while both residents
    // are still PrefillDone. All 8 prompt tokens are computed, even though
    // TokenSize() - decode_input_tokens is only 9 - 5 = 4.
    const ExecutionPlan retract = PlanOnce();
    const ForwardBatch* replacement = FindForwardBatch(retract);
    ASSERT_NE(replacement, nullptr);
    ASSERT_EQ(replacement->request_ids, std::vector<std::string>{"c"});
    ASSERT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(StateStoreCount(retract), 1);
    std::int32_t history_stores = 0;
    for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(retract)) {
        const auto& stores = std::get<WriteBackBatch>(operation);
        for (std::size_t i = 0; i < stores.op_ids.size(); ++i) {
            EXPECT_FALSE(stores.source_pinned.at(i));
            history_stores +=
                static_cast<std::int32_t>(std::count(stores.group_ids.at(i).begin(), stores.group_ids.at(i).end(), 0u));
        }
    }
    EXPECT_EQ(history_stores, 2);
    AckImageStores(retract);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 3);
    SendForwardDone("c", {241});
    for (const std::string& id : {"a", "b", "c"}) {
        SendAbortEvent(id);
    }
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0) << "an aborted victim releases its Host pins";
    ASSERT_TRUE(scheduler_->ClearL1Cache());
    auto replay_tokens = MakeTokens(8, 1);
    replay_tokens.push_back(999);
    ExpectReplay("recovered_prefill", std::move(replay_tokens), 8, 0);
}

TEST_F(StatePublicationSuite, ADecodingVictimResumesDecodingWithOrWithoutHostCache) {
    for (const bool host_cache : {false, true}) {
        SCOPED_TRACE(host_cache);
        Reset(host_cache, 1, 0);
        config_.device_allocator.total_pages = 11;
        config_.max_scheduled_tokens = 64;
        config_.max_batch_size = 2;
        config_.cache_groups.resize(2);
        for (CacheGroupConfig& group : config_.cache_groups) {
            group.total_pages = config_.device_allocator.total_pages;
        }
        SetTestSnapshotPool(config_);
        scheduler_ = std::make_unique<Scheduler>(config_);
        Submit({RequestSpec{.request_id = "a", .tokens = MakeTokens(8, 1)},
                RequestSpec{.request_id = "b", .tokens = MakeTokens(8, 101)}});
        const ExecutionPlan prefill = PlanOnce();
        ASSERT_NE(FindForwardBatch(prefill), nullptr);
        ASSERT_EQ(FindForwardBatch(prefill)->request_ids, (std::vector<std::string>{"a", "b"}));
        SendForwardDone("a", {41});
        SendForwardDone("b", {141});

        // The two residents decode until one needs a page the other holds;
        // the victim is suspended with its image. Its L2 leg carries only
        // what prefill published -- decode records no state checkpoint, so
        // the live state block rides the snapshot pool.
        bool retracted = false;
        ExecutionPlan retract;
        for (std::int32_t round = 0; round < 32 && !retracted; ++round) {
            retract = PlanOnce();
            for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(retract)) {
                const auto& stores = std::get<WriteBackBatch>(operation);
                for (std::size_t i = 0; i < stores.op_ids.size(); ++i) {
                    if (!stores.source_pinned.at(i)) {
                        EXPECT_EQ(std::ranges::find(stores.group_ids.at(i), 1u), stores.group_ids.at(i).end())
                            << "retraction must not publish or store a decode snapshot";
                    }
                }
            }
            AckImageStores(retract);
            retracted = scheduler_->RetractedSize() == 1u;
            const ForwardBatch* batch = FindForwardBatch(retract);
            ASSERT_NE(batch, nullptr);
            for (const std::string& id : batch->request_ids) {
                SendForwardDone(id, {id == "a" ? 42 + round : 142 + round});
            }
        }
        ASSERT_TRUE(retracted);
        const std::string victim = scheduler_->DecodingSize() == 1u && FindForwardBatch(retract) != nullptr &&
                                           std::ranges::find(FindForwardBatch(retract)->request_ids, "a") !=
                                               FindForwardBatch(retract)->request_ids.end()
                                       ? "b"
                                       : "a";
        const std::string survivor = victim == "a" ? "b" : "a";
        const std::int32_t token_count = scheduler_->RequestTokenSize(victim);
        ASSERT_GT(token_count, 9);
        const SnapshotStoreBatch* tail = FindSnapshotStore(retract);
        ASSERT_NE(tail, nullptr);
        EXPECT_GE(std::count(tail->group_ids.at(0).begin(), tail->group_ids.at(0).end(), 1u), 1)
            << "the live state block is private to the request and rides the snapshot pool";
        EXPECT_EQ(ExtractCacheOpsOfKind<WriteBackBatch>(retract).empty(), !host_cache)
            << "the published history pages ride Host L2 exactly when there is one";

        EXPECT_FALSE(scheduler_->ClearL1Cache()) << "a flush is refused while a request is suspended with an image";
        EXPECT_FALSE(scheduler_->CanClearCache());
        std::int32_t next_token = 1000;
        DecodeUntilDeviceCacheIsConsumed(survivor, next_token);
        SendAbortEvent(survivor);
        const ExecutionPlan recovery = PlanOnce();
        const SnapshotRestoreBatch* restore = FindRestore(recovery);
        ASSERT_NE(restore, nullptr);
        ASSERT_EQ(restore->request_ids, std::vector<std::string>{victim});
        EXPECT_TRUE(FindForwardBatch(recovery)->request_ids.empty()) << "nothing is recomputed";
        const auto& tiers = restore->source_tiers.at(0);
        EXPECT_EQ(std::ranges::count(tiers, static_cast<std::uint8_t>(HostTier::kL2)) > 0, host_cache);
        EXPECT_GT(std::ranges::count(tiers, static_cast<std::uint8_t>(HostTier::kSnapshotPool)), 0);
        AckRestores(recovery);
        EXPECT_EQ(scheduler_->RequestTokenSize(victim), token_count) << "the same tokens, resumed where it stopped";
        EXPECT_EQ(scheduler_->DecodingSize(), 1u);

        const ExecutionPlan decode = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(decode);
        ASSERT_NE(batch, nullptr);
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{victim});
        EXPECT_EQ(batch->NumExtends(), 0u);
        SendForwardDone(victim, {99});
        SendFinish(victim);
        AckWriteBacks(PlanOnce());
        EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), config_.snapshot_allocator.NumUsableBlocks());
    }
}

}  // namespace tokenspeed::test
