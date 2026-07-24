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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

const FlatForwardOperation* FindCompletionForward(const ExecutionPlan& plan) {
    for (const auto& op : plan.Operations()) {
        if (const auto* flat = std::get_if<FlatForwardOperation>(&op)) {
            return flat;
        }
    }
    return nullptr;
}

const FlatKVCompletionInput& InputForRequest(const ExecutionPlan& plan, const std::string& request_id) {
    const FlatForwardOperation* op = FindCompletionForward(plan);
    if (op == nullptr) {
        throw std::logic_error("execution plan has no flat forward operation");
    }
    for (std::size_t row = 0; row < op->request_ids.size(); ++row) {
        if (op->request_ids[row] == request_id) {
            return op->flat_kv_completion_inputs.at(row);
        }
    }
    throw std::logic_error("execution plan has no completion input for request " + request_id);
}

std::int32_t ExtendPrefixForRequest(const ExecutionPlan& plan, const std::string& request_id) {
    const FlatForwardOperation* op = FindCompletionForward(plan);
    if (op == nullptr) {
        throw std::logic_error("execution plan has no flat forward operation");
    }
    for (std::size_t row = 0; row < op->num_extends(); ++row) {
        if (op->request_ids[row] == request_id) {
            return op->extend_prefix_lens.at(row);
        }
    }
    throw std::logic_error("execution plan has no prefill row for request " + request_id);
}

std::vector<std::int32_t> FlatTableForRequest(const ExecutionPlan& plan, const std::string& request_id,
                                              const std::string& group_id) {
    const FlatForwardOperation* op = FindCompletionForward(plan);
    if (op == nullptr) {
        throw std::logic_error("execution plan has no flat forward operation");
    }
    for (std::size_t row = 0; row < op->request_ids.size(); ++row) {
        if (op->request_ids[row] == request_id) {
            const std::span<const std::int32_t> pages = op->flat_block_tables.at(group_id).Row(row);
            return {pages.begin(), pages.end()};
        }
    }
    throw std::logic_error("execution plan has no flat row for request " + request_id);
}

bool PlanContainsRequest(const ExecutionPlan& plan, const std::string& request_id) {
    const FlatForwardOperation* op = FindCompletionForward(plan);
    return op != nullptr && std::ranges::find(op->request_ids, request_id) != op->request_ids.end();
}

forward::FlatKVCompletion Ready(const FlatKVCompletionInput& input, std::int32_t accepted_raw_end) {
    return forward::FlatKVCompletion{
        .table_generation = input.table_generation,
        .dispatch_seq = input.dispatch_seq,
        .accepted_raw_end = accepted_raw_end,
    };
}

void SendCompletion(Scheduler& scheduler, const std::string& request_id, const FlatKVCompletionInput& input,
                    std::vector<std::int32_t> tokens, std::int32_t accepted_raw_end) {
    ExecutionEvent event;
    event.With(ForwardEvent{forward::ExtendResult{
        .request_id = request_id,
        .tokens = std::move(tokens),
        .flat_kv_completion = Ready(input, accepted_raw_end),
    }});
    scheduler.Advance(std::move(event));
}

#if TOKENSPEED_FLAT_KVCACHE

PagedCacheGroupConfig CompletionGroup(std::string id) {
    PagedCacheGroupConfig group;
    group.group_id = std::move(id);
    group.rows_per_page = 2;
    group.entry_stride_tokens = 1;
    group.total_pages = 128;
    group.block_size = 2;
    group.retention = PagedCacheGroupConfig::Retention::FullHistory;
    group.family = PagedCacheGroupFamily::History;
    return group;
}

std::vector<PagedCacheGroupConfig> HeterogeneousCompletionGroups(std::int32_t total_pages) {
    PagedCacheGroupConfig history = CompletionGroup("history");
    history.total_pages = total_pages;
    history.pool_id = "history-pool";
    history.prefix_role = PrefixRole::HistoryAnchor;
    history.owner_mask = 0b0001;

    PagedCacheGroupConfig state = CompletionGroup("state");
    state.total_pages = total_pages;
    state.pool_id = "state-pool";
    state.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    state.sliding_window_tokens = 8;
    state.family = PagedCacheGroupFamily::State;
    state.prefix_role = PrefixRole::ContinuationState;
    state.table_layout = TableLayout::BoundedWindow;
    state.owner_mask = 0b0001;
    return {std::move(history), std::move(state)};
}

SchedulerConfig HeterogeneousCompletionConfig(std::int32_t verify_width, std::int32_t overlap_depth,
                                              bool disable_prefix_cache = true, std::int32_t total_blocks = 512) {
    SchedulerConfig cfg{};
    cfg.block_size = 2;
    cfg.device_allocator.total_pages = 0;
    cfg.host_allocator.total_pages = 1;
    cfg.max_scheduled_tokens = 64;
    cfg.max_batch_size = 8;
    cfg.decode_input_tokens = verify_width;
    cfg.overlap_schedule_depth = overlap_depth;
    cfg.disable_l2_cache = true;
    cfg.disable_prefix_cache = disable_prefix_cache;
    cfg.flat_block_pools = {
        FlatBlockPoolConfig{.pool_id = "history-pool", .total_blocks = total_blocks, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state-pool", .total_blocks = total_blocks, .bytes_per_block = 32},
    };
    cfg.paged_cache_groups = HeterogeneousCompletionGroups(total_blocks);
    return cfg;
}

std::size_t FlatRowForRequest(const ExecutionPlan& plan, const std::string& request_id) {
    const FlatForwardOperation* op = FindCompletionForward(plan);
    if (op == nullptr) {
        throw std::logic_error("execution plan has no flat forward operation");
    }
    const auto row = std::ranges::find(op->request_ids, request_id);
    if (row == op->request_ids.end()) {
        throw std::logic_error("execution plan has no row for request " + request_id);
    }
    return static_cast<std::size_t>(std::distance(op->request_ids.begin(), row));
}

void ExpectFlatGroupCoversRawRange(const ExecutionPlan& plan, const std::string& request_id,
                                   const std::string& group_id, std::int32_t raw_tokens_per_page,
                                   std::int32_t raw_begin, std::int32_t raw_end) {
    const FlatForwardOperation* op = FindCompletionForward(plan);
    ASSERT_NE(op, nullptr);
    const std::size_t row = FlatRowForRequest(plan, request_id);
    const auto table = op->flat_block_tables.find(group_id);
    ASSERT_NE(table, op->flat_block_tables.end());
    ASSERT_LT(row, table->second.size());
    ASSERT_LT(row, table->second.bases.size());
    const std::span<const std::int32_t> pages = table->second.Row(row);
    const std::int32_t base = table->second.bases[row];
    for (std::int32_t raw = raw_begin; raw < raw_end; ++raw) {
        const std::int32_t column = raw / raw_tokens_per_page - base;
        ASSERT_GE(column, 0) << "group=" << group_id << " raw=" << raw;
        ASSERT_LT(column, static_cast<std::int32_t>(pages.size())) << "group=" << group_id << " raw=" << raw;
        EXPECT_GT(pages[static_cast<std::size_t>(column)], 0) << "group=" << group_id << " raw=" << raw;
    }
}

class FlatKVCompletionSchedulerTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = HeterogeneousCompletionConfig(/*verify_width=*/4, /*overlap_depth=*/1);
        cfg.max_scheduled_tokens = 4;
        return cfg;
    }

    const FlatKVCompletionInput& OnlyInput(const ExecutionPlan& plan) {
        const FlatForwardOperation* op = FindCompletionForward(plan);
        EXPECT_NE(op, nullptr);
        EXPECT_EQ(op == nullptr ? 0u : op->flat_kv_completion_inputs.size(), 1u);
        return op->flat_kv_completion_inputs.at(0);
    }
};

class FlatKVReadyPublicationSchedulerTest : public FlatKVCompletionSchedulerTest {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FlatKVCompletionSchedulerTest::MakeConfig();
        cfg.disable_prefix_cache = false;
        return cfg;
    }
};

class FlatKVResetSchedulerTest : public FlatKVCompletionSchedulerTest {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FlatKVCompletionSchedulerTest::MakeConfig();
        return cfg;
    }
};

class FlatKVShortAcceptReservationTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        return HeterogeneousCompletionConfig(/*verify_width=*/1, /*overlap_depth=*/1,
                                             /*disable_prefix_cache=*/true, /*total_blocks=*/4);
    }
};

class FlatKVLegacyOverlapSchedulerTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.block_size = 2;
        cfg.device_allocator.total_pages = 6;
        cfg.host_allocator.total_pages = 6;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.decode_input_tokens = 2;
        cfg.overlap_schedule_depth = 1;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        cfg.paged_cache_groups = {CompletionGroup("full")};
        cfg.paged_cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        return cfg;
    }
};

class FlatKVStarvationFenceSchedulerTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = HeterogeneousCompletionConfig(/*verify_width=*/3, /*overlap_depth=*/0);
        cfg.max_scheduled_tokens = 64;
        for (FlatBlockPoolConfig& pool : cfg.flat_block_pools) {
            pool.total_blocks = pool.pool_id == "history-pool" ? 10 : 5;
        }
        for (PagedCacheGroupConfig& group : cfg.paged_cache_groups) {
            group.total_pages = group.pool_id == "history-pool" ? 10 : 5;
        }
        PagedCacheGroupConfig& history = cfg.paged_cache_groups.front();
        history.rows_per_page = 4;
        history.block_size = 4;
        return cfg;
    }
};

enum class TerminalEvent { kAbort, kFinish, kPdSucceeded };

class FlatKVTerminalSchedulerTest : public FlatKVCompletionSchedulerTest,
                                    public ::testing::WithParamInterface<TerminalEvent> {};

class FlatKVAcceptedPublicationTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        return HeterogeneousCompletionConfig(/*verify_width=*/4, /*overlap_depth=*/0,
                                             /*disable_prefix_cache=*/false);
    }
};

class FlatKVExplicitMixedBatchTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = HeterogeneousCompletionConfig(/*verify_width=*/4, /*overlap_depth=*/0);
        cfg.enable_mixed_prefill_decode = true;
        return cfg;
    }
};

TEST(FlatKVSlidingExportBoundTest, FinalPrefillCarryAndOverlappedDecodeStayWithinCaptureBound) {
    constexpr std::int32_t kPrefillTokens = 64;
    constexpr std::int32_t kVerifyWidth = 1;
    constexpr std::int32_t kRawTokensPerPage = 2;
    constexpr std::int32_t kWindowTokens = 8;
    constexpr std::size_t kCaptureCols =
        (kWindowTokens + kPrefillTokens + 2 * kVerifyWidth + kRawTokensPerPage - 1) / kRawTokensPerPage + 1;
    constexpr std::size_t kColsWithoutPrefillCarry =
        (kWindowTokens + 2 * kVerifyWidth + kRawTokensPerPage - 1) / kRawTokensPerPage + 1;

    SchedulerConfig config = HeterogeneousCompletionConfig(kVerifyWidth, /*overlap_depth=*/1);
    config.max_batch_size = 1;
    Scheduler scheduler(std::move(config));
    scheduler.SubmitRequests({RequestSpec{
        .request_id = "r",
        .tokens = std::vector<std::int32_t>(2 * kPrefillTokens, 7),
    }});

    const FlatKVCompletionInput first_prefill = InputForRequest(scheduler.NextExecutionPlan(), "r");
    ASSERT_EQ(first_prefill.dispatch_raw_end, kPrefillTokens);
    SendCompletion(scheduler, "r", first_prefill, {}, first_prefill.dispatch_raw_end);

    const FlatKVCompletionInput final_prefill = InputForRequest(scheduler.NextExecutionPlan(), "r");
    ASSERT_EQ(final_prefill.dispatch_raw_end, 2 * kPrefillTokens);
    const ExecutionPlan first_decode_plan = scheduler.NextExecutionPlan();
    FlatKVCompletionInput predecessor = InputForRequest(first_decode_plan, "r");
    const std::size_t carry_cols = FlatTableForRequest(first_decode_plan, "r", "state").size();
    EXPECT_GT(carry_cols, kColsWithoutPrefillCarry);
    EXPECT_LE(carry_cols, kCaptureCols);

    SendCompletion(scheduler, "r", final_prefill, {9}, final_prefill.dispatch_raw_end);
    for (std::int32_t round = 0; round < 4; ++round) {
        const ExecutionPlan successor_plan = scheduler.NextExecutionPlan();
        const FlatKVCompletionInput successor = InputForRequest(successor_plan, "r");
        EXPECT_LE(FlatTableForRequest(successor_plan, "r", "state").size(), kCaptureCols) << "round=" << round;
        SendCompletion(scheduler, "r", predecessor, {100 + round}, predecessor.dispatch_raw_end);
        predecessor = successor;
    }
    SendCompletion(scheduler, "r", predecessor, {200}, predecessor.dispatch_raw_end);
}

TEST_F(FlatKVAcceptedPublicationTest, DecodePublishesOnlyAcceptedHistoryAndExactContinuationBundle) {
    Submit(RequestSpec{.request_id = "producer", .tokens = {1, 2}});
    const FlatKVCompletionInput prefill = InputForRequest(PlanOnce(), "producer");
    SendCompletion(*scheduler_, "producer", prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);

    const FlatKVCompletionInput decode = InputForRequest(PlanOnce(), "producer");
    const std::int32_t decode_start = decode.dispatch_raw_end - Config().decode_input_tokens;
    ASSERT_EQ(decode_start, 2);
    SendCompletion(*scheduler_, "producer", decode, /*tokens=*/{4, 5}, /*accepted_raw_end=*/4);

    Submit({RequestSpec{.request_id = "accepted", .tokens = {1, 2, 3, 4, 99}},
            RequestSpec{.request_id = "rejected-tail", .tokens = {1, 2, 3, 4, 5, 8, 9}}});
    const ExecutionPlan probe_plan = PlanOnce();
    EXPECT_EQ(ExtendPrefixForRequest(probe_plan, "accepted"), 4);
    EXPECT_EQ(ExtendPrefixForRequest(probe_plan, "rejected-tail"), 4);
}

TEST_F(FlatKVExplicitMixedBatchTest, PrefillAndDecodeRowsCarryEveryGroupTableAndCompletionSeed) {
    Submit(RequestSpec{.request_id = "decode", .tokens = {1, 2}});
    const FlatKVCompletionInput prefill = InputForRequest(PlanOnce(), "decode");
    SendCompletion(*scheduler_, "decode", prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);

    Submit(RequestSpec{.request_id = "prefill", .tokens = {11, 12, 13, 14, 15, 16, 17, 18}});
    const ExecutionPlan mixed_plan = PlanOnce();
    const FlatForwardOperation* mixed = FindCompletionForward(mixed_plan);
    ASSERT_NE(mixed, nullptr);
    ASSERT_EQ(mixed->request_ids.size(), 2u);
    ASSERT_EQ(mixed->extend_prefix_lens.size(), 1u);
    ASSERT_EQ(mixed->flat_kv_completion_inputs.size(), 2u);
    EXPECT_EQ(mixed->request_ids.front(), "prefill");
    EXPECT_EQ(mixed->request_ids.back(), "decode");
    EXPECT_EQ(mixed->flat_block_tables.size(), 2u);
    EXPECT_EQ(mixed->flat_block_tables.at("history").bases.size(), 2u);
    EXPECT_EQ(mixed->flat_block_tables.at("state").bases.size(), 2u);

    const FlatKVCompletionInput prefill_input = InputForRequest(mixed_plan, "prefill");
    const FlatKVCompletionInput decode_input = InputForRequest(mixed_plan, "decode");
    ExpectFlatGroupCoversRawRange(mixed_plan, "prefill", "history", /*raw_tokens_per_page=*/2, 0,
                                  prefill_input.dispatch_raw_end);
    ExpectFlatGroupCoversRawRange(mixed_plan, "prefill", "state", /*raw_tokens_per_page=*/2, 0,
                                  prefill_input.dispatch_raw_end);
    ExpectFlatGroupCoversRawRange(mixed_plan, "decode", "history", /*raw_tokens_per_page=*/2,
                                  decode_input.dispatch_raw_end - Config().decode_input_tokens,
                                  decode_input.dispatch_raw_end);
    ExpectFlatGroupCoversRawRange(mixed_plan, "decode", "state", /*raw_tokens_per_page=*/2,
                                  decode_input.dispatch_raw_end - Config().decode_input_tokens,
                                  decode_input.dispatch_raw_end);
}

TEST_F(FlatKVResetSchedulerTest, QuiescentResetAdvancesGenerationMonotonically) {
    EXPECT_TRUE(scheduler_->FlatKVQuiescent());
    const std::uint64_t initial = scheduler_->FlatKVGeneration();

    const std::uint64_t first = scheduler_->ResetFlatKVCache();
    const std::uint64_t second = scheduler_->ResetFlatKVCache();

    EXPECT_TRUE(scheduler_->FlatKVQuiescent());
    EXPECT_GT(first, initial);
    EXPECT_GT(second, first);
    EXPECT_EQ(second, scheduler_->FlatKVGeneration());

    Submit(MakeRequestSpec("post-reset", /*num_pages=*/1));
    const ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* forward = FindCompletionForward(plan);
    ASSERT_NE(forward, nullptr);
    EXPECT_EQ(forward->cache_generation, second);
}

TEST_F(FlatKVResetSchedulerTest, InflightCompletionRejectsReset) {
    Submit(MakeRequestSpec("r", /*num_pages=*/1));
    const ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* forward = FindCompletionForward(plan);
    ASSERT_NE(forward, nullptr);
    EXPECT_EQ(forward->cache_generation, scheduler_->FlatKVGeneration());
    const FlatKVCompletionInput in_flight = OnlyInput(plan);
    EXPECT_FALSE(scheduler_->FlatKVQuiescent());

    EXPECT_THROW(scheduler_->ResetFlatKVCache(), std::logic_error);
    EXPECT_EQ(scheduler_->FlatKVGeneration(), 0u);
    (void)in_flight;
}

TEST_F(FlatKVCompletionSchedulerTest, UnfencedRejectedSuccessorQuarantinesTablesUntilFenceArrives) {
    Submit(MakeRequestSpec("r", /*num_pages=*/1));
    const FlatKVCompletionInput prefill = OnlyInput(PlanOnce());
    SendCompletion(*scheduler_, "r", prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);

    const FlatKVCompletionInput first_decode = OnlyInput(PlanOnce());
    const ExecutionPlan successor_plan = PlanOnce();
    const FlatKVCompletionInput successor = InputForRequest(successor_plan, "r");
    const std::size_t protected_cols = FlatTableForRequest(successor_plan, "r", "history").size();

    SendCompletion(*scheduler_, "r", first_decode, /*tokens=*/{10, 11}, first_decode.dispatch_raw_end - 2);
    EXPECT_FALSE(PlanContainsRequest(PlanOnce(), "r"));

    // The canceled successor carries no logical result, but its completion is
    // the execution fence that makes physical rewind/reuse safe.
    SendCompletion(*scheduler_, "r", successor, /*tokens=*/{20, 21, 22, 23}, successor.dispatch_raw_end);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 5);

    const ExecutionPlan recovered_plan = PlanOnce();
    const FlatKVCompletionInput recovered = InputForRequest(recovered_plan, "r");
    EXPECT_GT(recovered.dispatch_seq, successor.dispatch_seq);
    EXPECT_LE(FlatTableForRequest(recovered_plan, "r", "history").size(), protected_cols);
}

TEST_F(FlatKVShortAcceptReservationTest, RewindRefreshesReservationBeforeCompetingExactFit) {
    EXPECT_EQ(SnapshotFreeBlocks(*scheduler_), (std::vector<std::int32_t>{3, 3}));
    Submit(RequestSpec{.request_id = "a", .tokens = {1, 2}});
    const FlatKVCompletionInput prefill = InputForRequest(PlanOnce(), "a");
    EXPECT_EQ(SnapshotFreeBlocks(*scheduler_), (std::vector<std::int32_t>{2, 2}));
    EXPECT_EQ(scheduler_->FlatReservedBlocksByPool(), (PoolDemand{1, 1}));
    SendCompletion(*scheduler_, "a", prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);

    const FlatKVCompletionInput decode = InputForRequest(PlanOnce(), "a");
    EXPECT_EQ(SnapshotFreeBlocks(*scheduler_), (std::vector<std::int32_t>{1, 1}));
    EXPECT_EQ(scheduler_->FlatReservedBlocksByPool(), (PoolDemand{0, 0}));
    SendCompletion(*scheduler_, "a", decode, /*tokens=*/{}, /*accepted_raw_end=*/2);
    EXPECT_EQ(SnapshotFreeBlocks(*scheduler_), (std::vector<std::int32_t>{2, 2}));
    EXPECT_EQ(scheduler_->FlatReservedBlocksByPool(), (PoolDemand{1, 1}));

    Submit(RequestSpec{.request_id = "b", .tokens = {101, 102}});
    const ExecutionPlan next_plan = PlanOnce();
    const FlatForwardOperation* next = FindCompletionForward(next_plan);
    ASSERT_NE(next, nullptr);
    EXPECT_EQ(next->request_ids, (std::vector<std::string>{"a"}));
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
}

TEST_F(FlatKVLegacyOverlapSchedulerTest, CompletionDebtDoesNotGateLegacyDispatch) {
    Submit(RequestSpec{.request_id = "r", .tokens = {1, 2}});
    PlanOnce();
    EXPECT_EQ(SnapshotFreeBlocks(*scheduler_), (std::vector<std::int32_t>{4}));
    EXPECT_EQ(scheduler_->FlatReservedBlocksByPool(), (PoolDemand{2}));
    SendForwardDone("r", {3});

    const ExecutionPlan first_plan = PlanOnce();
    const FlatForwardOperation* first = FindCompletionForward(first_plan);
    ASSERT_NE(first, nullptr);
    ASSERT_EQ(first->request_ids, (std::vector<std::string>{"r"}));
    EXPECT_EQ(SnapshotFreeBlocks(*scheduler_), (std::vector<std::int32_t>{3}));
    EXPECT_EQ(scheduler_->FlatReservedBlocksByPool(), (PoolDemand{1}));

    const ExecutionPlan second_plan = PlanOnce();
    const FlatForwardOperation* second = FindCompletionForward(second_plan);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(second->request_ids, (std::vector<std::string>{"r"}));
    EXPECT_EQ(SnapshotFreeBlocks(*scheduler_), (std::vector<std::int32_t>{2}));
    EXPECT_EQ(scheduler_->FlatReservedBlocksByPool(), (PoolDemand{0}));

    const ExecutionPlan successor_plan = PlanOnce();
    const FlatForwardOperation* successor = FindCompletionForward(successor_plan);
    ASSERT_NE(successor, nullptr);
    EXPECT_EQ(successor->request_ids, (std::vector<std::string>{"r"}))
        << "legacy Flat tracks completion debt for lifetime only; main never used it as a dispatch window";
}

TEST(FlatKVAdmissionOverflowTest, RejectsBeforePoolOrReservationMutation) {
    SchedulerConfig config =
        HeterogeneousCompletionConfig(std::numeric_limits<std::int32_t>::max(), /*overlap_depth=*/0,
                                      /*disable_prefix_cache=*/true, /*total_blocks=*/4);
    Scheduler scheduler(std::move(config));
    scheduler.SubmitRequests({RequestSpec{.request_id = "r", .tokens = {1, 2}}});
    const std::vector<std::int32_t> free_before = SnapshotFreeBlocks(scheduler);
    const PoolDemand reserved_before = scheduler.FlatReservedBlocksByPool();

    EXPECT_THROW(scheduler.NextExecutionPlan(), std::overflow_error);
    EXPECT_EQ(SnapshotFreeBlocks(scheduler), free_before);
    EXPECT_EQ(scheduler.FlatReservedBlocksByPool(), reserved_before);
    EXPECT_EQ(scheduler.WaitingSize(), 1u);
}

TEST(FlatKVCompletionRetirementTest, FinalizationFailureDoesNotRetireCompletionTicket) {
    SchedulerConfig config = HeterogeneousCompletionConfig(/*verify_width=*/4, /*overlap_depth=*/1);
    config.max_scheduled_tokens = 4;
    Scheduler scheduler(std::move(config));
    const std::vector<std::int32_t> initial_free = SnapshotFreeBlocks(scheduler);
    scheduler.SubmitRequests({RequestSpec{.request_id = "r", .tokens = {1, 2}}});
    const FlatKVCompletionInput prefill = InputForRequest(scheduler.NextExecutionPlan(), "r");
    SendCompletion(scheduler, "r", prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);
    const FlatKVCompletionInput decode = InputForRequest(scheduler.NextExecutionPlan(), "r");

    ExecutionEvent reserve_update;
    reserve_update.With(ForwardEvent{forward::UpdateReserveNumTokens{
        .request_id = "r",
        .reserve_num_tokens_in_next_schedule_event = -1,
    }});
    scheduler.Advance(std::move(reserve_update));

    EXPECT_THROW(SendCompletion(scheduler, "r", decode, /*tokens=*/{4, 5, 6, 7}, decode.dispatch_raw_end),
                 std::runtime_error);
    const std::vector<std::int32_t> held_after_failure = SnapshotFreeBlocks(scheduler);
    EXPECT_NE(held_after_failure, initial_free);

    // Production exits the event loop after the exception. This abort is only
    // an observation: a still-live ticket defers terminal release, whereas a
    // prematurely retired ticket would free every table here.
    ExecutionEvent abort;
    abort.With(ForwardEvent{forward::Abort{.request_id = "r"}});
    scheduler.Advance(std::move(abort));
    EXPECT_EQ(SnapshotFreeBlocks(scheduler), held_after_failure);
}

TEST_F(FlatKVStarvationFenceSchedulerTest, StarvationCannotRetractLiveTablesBeforeCompletionFence) {
    Submit({
        RequestSpec{.request_id = "a", .tokens = {1, 2, 3, 4}},
        RequestSpec{.request_id = "b", .tokens = {101, 102}},
    });
    const ExecutionPlan prefill_plan = PlanOnce();
    const FlatForwardOperation* prefill = FindCompletionForward(prefill_plan);
    ASSERT_NE(prefill, nullptr);
    ASSERT_EQ(prefill->request_ids, (std::vector<std::string>{"a"}));
    SendCompletion(*scheduler_, "a", InputForRequest(prefill_plan, "a"), {5}, /*accepted_raw_end=*/4);

    const ExecutionPlan decode_plan = PlanOnce();
    const FlatForwardOperation* decode = FindCompletionForward(decode_plan);
    ASSERT_NE(decode, nullptr);
    ASSERT_EQ(decode->request_ids, (std::vector<std::string>{"a"}));
    const FlatKVCompletionInput in_flight = InputForRequest(decode_plan, "a");
    const std::vector<std::int32_t> fenced_free = SnapshotFreeBlocks(*scheduler_);

    for (int starved_round = 0; starved_round < 3; ++starved_round) {
        const ExecutionPlan quiet_plan = PlanOnce();
        const FlatForwardOperation* quiet = FindCompletionForward(quiet_plan);
        ASSERT_NE(quiet, nullptr);
        EXPECT_TRUE(quiet->request_ids.empty());
        EXPECT_EQ(SnapshotFreeBlocks(*scheduler_), fenced_free);
        EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "the waiter must not steal or trigger reuse of live pages";
    }

    SendCompletion(*scheduler_, "a", in_flight, {6, 7, 8}, in_flight.dispatch_raw_end);
    const std::vector<std::int32_t> retired_free = SnapshotFreeBlocks(*scheduler_);
    EXPECT_EQ(retired_free, fenced_free);

    const ExecutionPlan first_starved_plan = PlanOnce();
    const FlatForwardOperation* first_starved = FindCompletionForward(first_starved_plan);
    ASSERT_NE(first_starved, nullptr);
    EXPECT_TRUE(first_starved->request_ids.empty());
    EXPECT_EQ(SnapshotFreeBlocks(*scheduler_), retired_free);

    const ExecutionPlan retract_plan = PlanOnce();
    const FlatForwardOperation* retract = FindCompletionForward(retract_plan);
    ASSERT_NE(retract, nullptr);
    EXPECT_TRUE(retract->request_ids.empty());
    EXPECT_NE(SnapshotFreeBlocks(*scheduler_), retired_free)
        << "after the fence retires, the second eligible starved round may retract the holder";
    EXPECT_EQ(scheduler_->WaitingSize(), 2u);
}

TEST_P(FlatKVTerminalSchedulerTest, WaitsForFenceAndReadmitNeverReusesGeneration) {
    const std::vector<std::int32_t> initial_free = SnapshotFreeBlocks(*scheduler_);
    Submit(MakeRequestSpec("r", /*num_pages=*/1));
    const FlatKVCompletionInput prefill = OnlyInput(PlanOnce());
    SendCompletion(*scheduler_, "r", prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);
    const FlatKVCompletionInput stale_decode = OnlyInput(PlanOnce());
    if (GetParam() == TerminalEvent::kAbort) {
        ExecutionEvent abort;
        abort.With(ForwardEvent{forward::Abort{.request_id = "r"}});
        scheduler_->Advance(std::move(abort));
    } else if (GetParam() == TerminalEvent::kFinish) {
        SendFinish("r");
    } else {
        ExecutionEvent succeeded;
        succeeded.With(PDEvent{pd::SucceededEvent{"r"}});
        scheduler_->Advance(std::move(succeeded));
    }
    EXPECT_FALSE(PlanContainsRequest(PlanOnce(), "r"));
    EXPECT_NE(SnapshotFreeBlocks(*scheduler_), initial_free)
        << "terminal handling must not recycle tables before the execution fence";

    SendCompletion(*scheduler_, "r", stale_decode, /*tokens=*/{77}, stale_decode.dispatch_raw_end);
    EXPECT_EQ(SnapshotFreeBlocks(*scheduler_), initial_free)
        << "retiring the final canceled completion fence releases terminal tables";
    PlanOnce();

    Submit(MakeRequestSpec("r", /*num_pages=*/1, /*start=*/101));
    const FlatKVCompletionInput readmitted = OnlyInput(PlanOnce());
    EXPECT_NE(readmitted.table_generation, stale_decode.table_generation);
    SendCompletion(*scheduler_, "r", stale_decode, /*tokens=*/{77}, stale_decode.dispatch_raw_end);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 2);
    SendCompletion(*scheduler_, "r", readmitted, /*tokens=*/{103}, readmitted.dispatch_raw_end);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 3);
}

INSTANTIATE_TEST_SUITE_P(AllTerminalSources, FlatKVTerminalSchedulerTest,
                         ::testing::Values(TerminalEvent::kAbort, TerminalEvent::kFinish, TerminalEvent::kPdSucceeded));

TEST_F(FlatKVReadyPublicationSchedulerTest, OverlappedAbortNeverPublishesCanceledPages) {
    Submit(RequestSpec{.request_id = "producer", .tokens = {1, 2}});
    const FlatKVCompletionInput prefill = OnlyInput(PlanOnce());
    SendCompletion(*scheduler_, "producer", prefill, {3}, prefill.dispatch_raw_end);

    const FlatKVCompletionInput first_decode = OnlyInput(PlanOnce());
    const FlatKVCompletionInput successor = OnlyInput(PlanOnce());

    ExecutionEvent abort_then_first;
    abort_then_first.With(ForwardEvent{forward::Abort{.request_id = "producer"}});
    abort_then_first.With(ForwardEvent{forward::ExtendResult{
        .request_id = "producer",
        .tokens = {10, 11, 12, 13},
        .flat_kv_completion = Ready(first_decode, first_decode.dispatch_raw_end),
    }});
    scheduler_->Advance(std::move(abort_then_first));
    SendCompletion(*scheduler_, "producer", successor, {20, 21, 22, 23}, successor.dispatch_raw_end);

    Submit(RequestSpec{.request_id = "probe", .tokens = {1, 2, 3, 10, 11, 12, 99}});
    EXPECT_EQ(ExtendPrefixForRequest(PlanOnce(), "probe"), 2);
}

TEST_F(FlatKVReadyPublicationSchedulerTest, PrefixIsInvisibleBeforeFenceAndHittableAfterCompletion) {
    Submit(MakeRequestSpec("producer", /*num_pages=*/2));
    const ExecutionPlan producer_plan = PlanOnce();
    const FlatKVCompletionInput producer = InputForRequest(producer_plan, "producer");

    Submit(MakeRequestSpec("before", /*num_pages=*/2));
    const ExecutionPlan before_plan = PlanOnce();
    EXPECT_EQ(ExtendPrefixForRequest(before_plan, "before"), 0);

    SendCompletion(*scheduler_, "producer", producer, /*tokens=*/{99}, producer.dispatch_raw_end);

    Submit(MakeRequestSpec("after", /*num_pages=*/2));
    const ExecutionPlan after_plan = PlanOnce();
    // Match excludes the final prompt token, so a four-token request can reuse
    // exactly its first two-token base page.
    EXPECT_EQ(ExtendPrefixForRequest(after_plan, "after"), 2);
}

#endif

}  // namespace
}  // namespace tokenspeed::test
