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
#include <bit>
#include <cstdint>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
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
        .request_id = input.request_id,
        .table_generation = input.table_generation,
        .dispatch_seq = input.dispatch_seq,
        .accepted_raw_end = accepted_raw_end,
        .protected_raw_end = input.protected_raw_end,
        .groups =
            {
                forward::FlatKVGroupCompletion{
                    .group_id = "history",
                    .completed_domain_mask = 0b0011,
                    .domain_valid_ends = {input.dispatch_raw_end, input.dispatch_raw_end},
                },
                forward::FlatKVGroupCompletion{
                    .group_id = "state",
                    .completed_domain_mask = 0b0101,
                    .domain_valid_ends = {input.dispatch_raw_end, input.dispatch_raw_end},
                },
            },
    };
}

void SendCompletion(Scheduler& scheduler, const FlatKVCompletionInput& input, std::vector<std::int32_t> tokens,
                    std::int32_t accepted_raw_end) {
    ExecutionEvent event;
    event.With(ForwardEvent{forward::ExtendResult{
        .request_id = input.request_id,
        .tokens = std::move(tokens),
        .flat_kv_completion = Ready(input, accepted_raw_end),
    }});
    scheduler.Advance(std::move(event));
}

#if TOKENSPEED_FLAT_KVCACHE

PagedCacheGroupConfig CompletionGroup(std::string id, std::uint32_t required_domain_mask) {
    PagedCacheGroupConfig group;
    group.group_id = std::move(id);
    group.rows_per_page = 2;
    group.entry_stride_tokens = 1;
    group.total_pages = 128;
    group.block_size = 2;
    group.retention = PagedCacheGroupConfig::Retention::FullHistory;
    group.family = PagedCacheGroupFamily::History;
    group.required_producer_domain_mask = required_domain_mask;
    return group;
}

std::vector<PagedCacheGroupConfig> HeterogeneousCompletionGroups(std::int32_t total_pages) {
    PagedCacheGroupConfig history = CompletionGroup("history", 0b0011);
    history.total_pages = total_pages;
    history.pool_id = "history-pool";
    history.prefix_role = PrefixRole::HistoryAnchor;
    history.owner_mask = 0b0001;

    PagedCacheGroupConfig state = CompletionGroup("state", 0b0101);
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

constexpr std::uint32_t kV4TargetMain = 1u << 0;
constexpr std::uint32_t kV4TargetIndexer = 1u << 1;

PagedCacheGroupConfig V4CompletionGroup(std::string group_id, std::int32_t rows_per_page,
                                        std::int32_t entry_stride_tokens, std::int32_t sliding_window_tokens,
                                        PrefixRole prefix_role, std::uint32_t required_producer_domain_mask) {
    constexpr std::int32_t kTotalBlocks = 1024;
    PagedCacheGroupConfig group;
    group.group_id = std::move(group_id);
    group.rows_per_page = rows_per_page;
    group.entry_stride_tokens = entry_stride_tokens;
    group.total_pages = kTotalBlocks;
    group.block_size = rows_per_page * entry_stride_tokens;
    group.pool_id = group.group_id + ".pool";
    group.prefix_role = prefix_role;
    group.required_producer_domain_mask = required_producer_domain_mask;
    group.owner_mask = 1;
    if (prefix_role == PrefixRole::ContinuationState) {
        group.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
        group.sliding_window_tokens = sliding_window_tokens;
        group.family = PagedCacheGroupFamily::State;
        group.table_layout = TableLayout::BoundedWindow;
    }
    return group;
}

std::vector<PagedCacheGroupConfig> V4CompletionGroups() {
    return {
        V4CompletionGroup("v4.swa_kv", /*rows_per_page=*/64, /*entry_stride_tokens=*/1,
                          /*sliding_window_tokens=*/128, PrefixRole::ContinuationState, kV4TargetMain),
        V4CompletionGroup("v4.c4a.compressor_state", /*rows_per_page=*/4, /*entry_stride_tokens=*/1,
                          /*sliding_window_tokens=*/8, PrefixRole::ContinuationState, kV4TargetMain),
        V4CompletionGroup("v4.c4a.compressed_kv", /*rows_per_page=*/64, /*entry_stride_tokens=*/4,
                          /*sliding_window_tokens=*/0, PrefixRole::HistoryAnchor, kV4TargetMain | kV4TargetIndexer),
        V4CompletionGroup("v4.c128a.compressor_state", /*rows_per_page=*/8, /*entry_stride_tokens=*/1,
                          /*sliding_window_tokens=*/128, PrefixRole::ContinuationState, kV4TargetMain),
        V4CompletionGroup("v4.c128a.compressed_kv", /*rows_per_page=*/2, /*entry_stride_tokens=*/128,
                          /*sliding_window_tokens=*/0, PrefixRole::HistoryAnchor, kV4TargetMain),
        V4CompletionGroup("v4.c4a.indexer_compressor_state", /*rows_per_page=*/4,
                          /*entry_stride_tokens=*/1, /*sliding_window_tokens=*/8, PrefixRole::ContinuationState,
                          kV4TargetIndexer),
    };
}

SchedulerConfig V4CompletionConfig(std::int32_t legacy_block_size, bool disable_prefix_cache) {
    constexpr std::int32_t kTotalBlocks = 1024;
    SchedulerConfig cfg{};
    cfg.block_size = legacy_block_size;
    cfg.device_allocator.total_pages = 0;
    cfg.host_allocator.total_pages = 1;
    cfg.max_scheduled_tokens = 512;
    cfg.max_batch_size = 8;
    cfg.decode_input_tokens = 1;
    cfg.overlap_schedule_depth = 0;
    cfg.disable_l2_cache = true;
    cfg.disable_prefix_cache = disable_prefix_cache;
    cfg.enable_structured_flat_kv_completion = true;
    cfg.paged_cache_groups = V4CompletionGroups();
    if (!disable_prefix_cache) {
        cfg.prefix_cache_adjunct = PrefixCacheAdjunctSpec{
            .required_groups = {"v4.c4a.compressed_kv", "v4.c128a.compressed_kv"},
        };
    }
    cfg.flat_block_pools.reserve(cfg.paged_cache_groups.size());
    for (const PagedCacheGroupConfig& group : cfg.paged_cache_groups) {
        cfg.flat_block_pools.push_back(FlatBlockPoolConfig{
            .pool_id = group.pool_id,
            .total_blocks = kTotalBlocks,
            .bytes_per_block = 128,
        });
    }
    return cfg;
}

std::vector<std::int32_t> V4Prompt() {
    std::vector<std::int32_t> prompt;
    prompt.reserve(585);
    for (std::int32_t token = 1; token <= 585; ++token) {
        prompt.push_back(token);
    }
    return prompt;
}

forward::FlatKVCompletion V4Ready(const FlatKVCompletionInput& input) {
    forward::FlatKVCompletion completion{
        .request_id = input.request_id,
        .table_generation = input.table_generation,
        .dispatch_seq = input.dispatch_seq,
        .accepted_raw_end = input.dispatch_raw_end,
        .protected_raw_end = input.protected_raw_end,
    };
    for (const PagedCacheGroupConfig& group : V4CompletionGroups()) {
        const std::int32_t valid_raw_end =
            input.dispatch_raw_end / group.entry_stride_tokens * group.entry_stride_tokens;
        completion.groups.push_back(forward::FlatKVGroupCompletion{
            .group_id = group.group_id,
            .completed_domain_mask = group.required_producer_domain_mask,
            .domain_valid_ends =
                std::vector<std::int32_t>(std::popcount(group.required_producer_domain_mask), valid_raw_end),
        });
    }
    return completion;
}

void SendV4Completion(Scheduler& scheduler, const FlatKVCompletionInput& input) {
    ExecutionEvent event;
    event.With(ForwardEvent{forward::ExtendResult{
        .request_id = input.request_id,
        .tokens = {},
        .flat_kv_completion = V4Ready(input),
    }});
    scheduler.Advance(std::move(event));
}

SchedulerConfig HeterogeneousCompletionConfig(std::int32_t verify_width, std::int32_t overlap_depth,
                                              bool disable_prefix_cache = true) {
    constexpr std::int32_t kTotalBlocks = 512;
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
    cfg.enable_structured_flat_kv_completion = true;
    cfg.flat_block_pools = {
        FlatBlockPoolConfig{.pool_id = "history-pool", .total_blocks = kTotalBlocks, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state-pool", .total_blocks = kTotalBlocks, .bytes_per_block = 32},
    };
    cfg.paged_cache_groups = HeterogeneousCompletionGroups(kTotalBlocks);
    return cfg;
}

TEST(FlatKVStructuredOwnershipTest, V4ColdAdmissionIsIndependentOfGlobalBlockSizeAndPrefixSwitch) {
    struct PlanShape {
        std::int32_t extend_prefix_len{};
        std::int32_t input_length{};
        std::vector<std::size_t> group_columns;
        PoolDemand free_blocks;

        bool operator==(const PlanShape&) const = default;
    };

    std::optional<PlanShape> baseline;
    for (const bool disable_prefix_cache : {false, true}) {
        for (const std::int32_t global_block_size : {128, 256}) {
            Scheduler scheduler(V4CompletionConfig(global_block_size, disable_prefix_cache));
            scheduler.SubmitRequests({RequestSpec{.request_id = "request", .tokens = V4Prompt()}});
            const ExecutionPlan plan = scheduler.NextExecutionPlan();
            const FlatForwardOperation* op = FindCompletionForward(plan);
            ASSERT_NE(op, nullptr);
            ASSERT_EQ(op->request_ids, (std::vector<std::string>{"request"}));

            PlanShape shape{
                .extend_prefix_len = op->extend_prefix_lens.at(0),
                .input_length = op->input_lengths.at(0),
                .free_blocks = scheduler.FlatPoolFreeBlocksByPool(),
            };
            for (const PagedCacheGroupConfig& group : V4CompletionGroups()) {
                shape.group_columns.push_back(op->flat_block_tables.at(group.group_id).Row(0).size());
            }
            EXPECT_EQ(shape.extend_prefix_len, 0);
            EXPECT_EQ(shape.input_length, 512);
            if (!baseline.has_value()) {
                baseline = shape;
            } else {
                EXPECT_EQ(shape, *baseline) << "structured flat admission must not observe SchedulerConfig.block_size";
            }
        }
    }
}

TEST(FlatKVStructuredOwnershipTest, V4CompletionRejectsMissingIndexerDomain) {
    Scheduler scheduler(V4CompletionConfig(/*legacy_block_size=*/256, /*disable_prefix_cache=*/true));
    scheduler.SubmitRequests({RequestSpec{.request_id = "request", .tokens = V4Prompt()}});
    const FlatKVCompletionInput input = InputForRequest(scheduler.NextExecutionPlan(), "request");
    forward::FlatKVCompletion completion = V4Ready(input);
    const auto c4_history = std::ranges::find_if(
        completion.groups, [](const auto& group) { return group.group_id == "v4.c4a.compressed_kv"; });
    ASSERT_NE(c4_history, completion.groups.end());
    ASSERT_EQ(c4_history->completed_domain_mask, kV4TargetMain | kV4TargetIndexer);
    c4_history->completed_domain_mask = kV4TargetMain;
    c4_history->domain_valid_ends = {input.dispatch_raw_end};

    ExecutionEvent event;
    event.With(ForwardEvent{forward::ExtendResult{
        .request_id = input.request_id,
        .tokens = {},
        .flat_kv_completion = std::move(completion),
    }});

    EXPECT_THROW(scheduler.Advance(std::move(event)), std::invalid_argument);
    EXPECT_EQ(scheduler.FlatKVCompletionOutstandingCount("request"), 1u);
}

TEST(FlatKVStructuredOwnershipTest, V4CompletionPublishesWarmPrefixAtHistoryAlignment) {
    Scheduler scheduler(V4CompletionConfig(/*legacy_block_size=*/256, /*disable_prefix_cache=*/false));
    const std::vector<std::int32_t> prompt = V4Prompt();
    scheduler.SubmitRequests({RequestSpec{.request_id = "producer", .tokens = prompt}});
    const ExecutionPlan producer_plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(ExtendPrefixForRequest(producer_plan, "producer"), 0);
    const FlatKVCompletionInput producer = InputForRequest(producer_plan, "producer");
    ASSERT_EQ(producer.dispatch_raw_end, 512);
    SendV4Completion(scheduler, producer);

    scheduler.SubmitRequests({RequestSpec{.request_id = "consumer", .tokens = prompt}});
    const ExecutionPlan consumer_plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(ExtendPrefixForRequest(consumer_plan, "consumer"), 512);
    const FlatForwardOperation* consumer_op = FindCompletionForward(consumer_plan);
    ASSERT_NE(consumer_op, nullptr);
    const auto consumer = std::ranges::find(consumer_op->request_ids, "consumer");
    ASSERT_NE(consumer, consumer_op->request_ids.end());
    const std::size_t consumer_row =
        static_cast<std::size_t>(std::distance(consumer_op->request_ids.begin(), consumer));
    EXPECT_EQ(consumer_op->input_lengths.at(consumer_row), 73);
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
        SchedulerConfig cfg{};
        cfg.block_size = 2;
        cfg.device_allocator.total_pages = 128;
        cfg.host_allocator.total_pages = 1;
        cfg.max_scheduled_tokens = 4;
        cfg.max_batch_size = 8;
        cfg.decode_input_tokens = 4;
        cfg.overlap_schedule_depth = 1;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        cfg.enable_structured_flat_kv_completion = true;
        cfg.paged_cache_groups = {CompletionGroup("history", 0b0011), CompletionGroup("state", 0b0101)};
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
        cfg.device_allocator.total_pages = 0;
        cfg.flat_block_pools = {
            FlatBlockPoolConfig{.pool_id = "history-pool", .total_blocks = 128, .bytes_per_block = 128},
            FlatBlockPoolConfig{.pool_id = "state-pool", .total_blocks = 128, .bytes_per_block = 32},
        };
        cfg.paged_cache_groups[0].pool_id = "history-pool";
        cfg.paged_cache_groups[1].pool_id = "state-pool";
        return cfg;
    }
};

enum class TerminalEvent { kAbort, kFinish };

class FlatKVTerminalSchedulerTest : public FlatKVCompletionSchedulerTest,
                                    public ::testing::WithParamInterface<TerminalEvent> {};

class FlatKVCompletionOverlapMatrixTest
    : public SchedulerTestSuite,
      public ::testing::WithParamInterface<std::tuple<std::int32_t, std::int32_t, std::int32_t>> {
protected:
    SchedulerConfig MakeConfig() override {
        const auto [verify_width, _, overlap_depth] = GetParam();
        return HeterogeneousCompletionConfig(verify_width, overlap_depth);
    }
};

class FlatKVAcceptedPublicationTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        return HeterogeneousCompletionConfig(/*verify_width=*/4, /*overlap_depth=*/0,
                                             /*disable_prefix_cache=*/false);
    }
};

class FlatKVStructuredMixedBatchTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = HeterogeneousCompletionConfig(/*verify_width=*/4, /*overlap_depth=*/0);
        cfg.enable_mixed_prefill_decode = true;
        return cfg;
    }
};

TEST_P(FlatKVCompletionOverlapMatrixTest, AcceptedLengthAndOverlapKeepProjectedTablesAndProgressContiguous) {
    const auto [verify_width, accepted_length, overlap_depth] = GetParam();
    Submit(RequestSpec{.request_id = "r", .tokens = {1, 2}});
    const FlatKVCompletionInput prefill = InputForRequest(PlanOnce(), "r");
    SendCompletion(*scheduler_, prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);
    ASSERT_EQ(scheduler_->GetRequestTokenSize("r"), 3);

    const ExecutionPlan first_plan = PlanOnce();
    const FlatKVCompletionInput first = InputForRequest(first_plan, "r");
    const std::int32_t first_start = first.dispatch_raw_end - verify_width;
    ASSERT_EQ(first_start, 2);
    ExpectFlatGroupCoversRawRange(first_plan, "r", "history", /*raw_tokens_per_page=*/2, first_start,
                                  first.dispatch_raw_end);
    ExpectFlatGroupCoversRawRange(first_plan, "r", "state", /*raw_tokens_per_page=*/2, first_start,
                                  first.dispatch_raw_end);

    std::optional<FlatKVCompletionInput> successor;
    if (overlap_depth == 1) {
        const ExecutionPlan successor_plan = PlanOnce();
        successor = InputForRequest(successor_plan, "r");
        EXPECT_EQ(successor->dispatch_raw_end, first.dispatch_raw_end + verify_width);
        EXPECT_EQ(successor->protected_raw_end, first.protected_raw_end);
        ExpectFlatGroupCoversRawRange(successor_plan, "r", "history", /*raw_tokens_per_page=*/2, first.dispatch_raw_end,
                                      successor->dispatch_raw_end);
        ExpectFlatGroupCoversRawRange(successor_plan, "r", "state", /*raw_tokens_per_page=*/2, first.dispatch_raw_end,
                                      successor->dispatch_raw_end);

        std::vector<std::int32_t> successor_tokens(static_cast<std::size_t>(verify_width), 2000 + verify_width);
        SendCompletion(*scheduler_, *successor, std::move(successor_tokens), successor->dispatch_raw_end);
        EXPECT_EQ(scheduler_->FlatKVCompletionBufferedCount("r"), 1u);
    }

    std::vector<std::int32_t> accepted(static_cast<std::size_t>(accepted_length), 1000 + accepted_length);
    const std::int32_t accepted_raw_end = first_start + accepted_length;
    SendCompletion(*scheduler_, first, std::move(accepted), accepted_raw_end);

    const bool successor_is_valid = successor.has_value() && accepted_length == verify_width;
    const std::int32_t expected_token_size = 3 + accepted_length + (successor_is_valid ? verify_width : 0);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), expected_token_size);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);

    const ExecutionPlan recovered_plan = PlanOnce();
    const FlatKVCompletionInput recovered = InputForRequest(recovered_plan, "r");
    const std::int32_t recovered_start = expected_token_size - 1;
    EXPECT_EQ(recovered.dispatch_raw_end, recovered_start + verify_width);
    ExpectFlatGroupCoversRawRange(recovered_plan, "r", "history", /*raw_tokens_per_page=*/2, recovered_start,
                                  recovered.dispatch_raw_end);
    ExpectFlatGroupCoversRawRange(recovered_plan, "r", "state", /*raw_tokens_per_page=*/2, recovered_start,
                                  recovered.dispatch_raw_end);
}

INSTANTIATE_TEST_SUITE_P(FlatV4VerifyWidthsAndAcceptLengths, FlatKVCompletionOverlapMatrixTest,
                         ::testing::Values(std::make_tuple(1, 1, 0), std::make_tuple(2, 1, 0), std::make_tuple(4, 1, 0),
                                           std::make_tuple(8, 1, 0), std::make_tuple(1, 1, 1), std::make_tuple(2, 1, 1),
                                           std::make_tuple(2, 2, 1), std::make_tuple(4, 0, 1), std::make_tuple(4, 2, 1),
                                           std::make_tuple(4, 4, 1), std::make_tuple(8, 0, 1), std::make_tuple(8, 4, 1),
                                           std::make_tuple(8, 7, 1), std::make_tuple(8, 8, 1)));

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
    SendCompletion(scheduler, first_prefill, {}, first_prefill.dispatch_raw_end);

    const FlatKVCompletionInput final_prefill = InputForRequest(scheduler.NextExecutionPlan(), "r");
    ASSERT_EQ(final_prefill.dispatch_raw_end, 2 * kPrefillTokens);
    const ExecutionPlan first_decode_plan = scheduler.NextExecutionPlan();
    FlatKVCompletionInput predecessor = InputForRequest(first_decode_plan, "r");
    const std::size_t carry_cols = FlatTableForRequest(first_decode_plan, "r", "state").size();
    EXPECT_GT(carry_cols, kColsWithoutPrefillCarry);
    EXPECT_LE(carry_cols, kCaptureCols);

    SendCompletion(scheduler, final_prefill, {9}, final_prefill.dispatch_raw_end);
    for (std::int32_t round = 0; round < 4; ++round) {
        const ExecutionPlan successor_plan = scheduler.NextExecutionPlan();
        const FlatKVCompletionInput successor = InputForRequest(successor_plan, "r");
        EXPECT_LE(FlatTableForRequest(successor_plan, "r", "state").size(), kCaptureCols) << "round=" << round;
        SendCompletion(scheduler, predecessor, {100 + round}, predecessor.dispatch_raw_end);
        predecessor = successor;
    }
    SendCompletion(scheduler, predecessor, {200}, predecessor.dispatch_raw_end);
}

TEST_F(FlatKVAcceptedPublicationTest, DecodePublishesOnlyAcceptedHistoryAndExactContinuationBundle) {
    Submit(RequestSpec{.request_id = "producer", .tokens = {1, 2}});
    const FlatKVCompletionInput prefill = InputForRequest(PlanOnce(), "producer");
    SendCompletion(*scheduler_, prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);

    const FlatKVCompletionInput decode = InputForRequest(PlanOnce(), "producer");
    const std::int32_t decode_start = decode.dispatch_raw_end - Config().decode_input_tokens;
    ASSERT_EQ(decode_start, 2);
    SendCompletion(*scheduler_, decode, /*tokens=*/{4, 5}, /*accepted_raw_end=*/4);

    Submit({RequestSpec{.request_id = "accepted", .tokens = {1, 2, 3, 4, 99}},
            RequestSpec{.request_id = "rejected-tail", .tokens = {1, 2, 3, 4, 5, 8, 9}}});
    const ExecutionPlan probe_plan = PlanOnce();
    EXPECT_EQ(ExtendPrefixForRequest(probe_plan, "accepted"), 4);
    EXPECT_EQ(ExtendPrefixForRequest(probe_plan, "rejected-tail"), 4);
}

TEST_F(FlatKVStructuredMixedBatchTest, PrefillAndDecodeRowsCarryEveryGroupTableAndCompletionSeed) {
    Submit(RequestSpec{.request_id = "decode", .tokens = {1, 2}});
    const FlatKVCompletionInput prefill = InputForRequest(PlanOnce(), "decode");
    SendCompletion(*scheduler_, prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);

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

TEST_F(FlatKVResetSchedulerTest, QuiescentResetSucceeds) {
    EXPECT_TRUE(scheduler_->FlatKVQuiescent());
    EXPECT_EQ(scheduler_->FlatKVGeneration(), 0u);

    EXPECT_EQ(scheduler_->ResetFlatKVCache(), 1u);

    EXPECT_TRUE(scheduler_->FlatKVQuiescent());
    EXPECT_EQ(scheduler_->FlatKVGeneration(), 1u);
}

TEST_F(FlatKVResetSchedulerTest, InflightCompletionRejectsReset) {
    Submit(MakeRequestSpec("r", /*num_pages=*/1));
    const FlatKVCompletionInput in_flight = OnlyInput(PlanOnce());
    ASSERT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 1u);
    EXPECT_FALSE(scheduler_->FlatKVQuiescent());

    EXPECT_THROW(scheduler_->ResetFlatKVCache(), std::logic_error);
    EXPECT_EQ(scheduler_->FlatKVGeneration(), 0u);
    (void)in_flight;
}

TEST_F(FlatKVResetSchedulerTest, GenerationAdvancesMonotonically) {
    const std::uint64_t initial = scheduler_->FlatKVGeneration();
    const std::uint64_t first = scheduler_->ResetFlatKVCache();
    const std::uint64_t second = scheduler_->ResetFlatKVCache();

    EXPECT_GT(first, initial);
    EXPECT_GT(second, first);
    EXPECT_EQ(second, scheduler_->FlatKVGeneration());
}

TEST_F(FlatKVCompletionSchedulerTest, UnfencedRejectedSuccessorQuarantinesTablesUntilFenceArrives) {
    Submit(MakeRequestSpec("r", /*num_pages=*/1));
    const FlatKVCompletionInput prefill = OnlyInput(PlanOnce());
    SendCompletion(*scheduler_, prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);

    const FlatKVCompletionInput first_decode = OnlyInput(PlanOnce());
    const ExecutionPlan successor_plan = PlanOnce();
    const FlatKVCompletionInput successor = InputForRequest(successor_plan, "r");
    const std::size_t protected_cols = FlatTableForRequest(successor_plan, "r", "history").size();

    SendCompletion(*scheduler_, first_decode, /*tokens=*/{10, 11}, first_decode.dispatch_raw_end - 2);
    EXPECT_TRUE(scheduler_->FlatKVCompletionOutstandingCount("r") > 0);
    EXPECT_FALSE(PlanContainsRequest(PlanOnce(), "r"));

    // The canceled successor carries no logical result, but its completion is
    // the execution fence that makes physical rewind/reuse safe.
    SendCompletion(*scheduler_, successor, /*tokens=*/{20, 21, 22, 23}, successor.dispatch_raw_end);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 5);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);

    const ExecutionPlan recovered_plan = PlanOnce();
    const FlatKVCompletionInput recovered = InputForRequest(recovered_plan, "r");
    EXPECT_GT(recovered.dispatch_seq, successor.dispatch_seq);
    EXPECT_LE(FlatTableForRequest(recovered_plan, "r", "history").size(), protected_cols);
}

TEST_P(FlatKVTerminalSchedulerTest, WaitsForFenceAndReadmitNeverReusesGeneration) {
    Submit(MakeRequestSpec("r", /*num_pages=*/1));
    const FlatKVCompletionInput prefill = OnlyInput(PlanOnce());
    SendCompletion(*scheduler_, prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);
    const FlatKVCompletionInput stale_decode = OnlyInput(PlanOnce());
    if (GetParam() == TerminalEvent::kAbort) {
        ExecutionEvent abort;
        abort.With(ForwardEvent{forward::Abort{.request_id = "r"}});
        scheduler_->Advance(std::move(abort));
    } else {
        SendFinish("r");
    }
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 1u);
    EXPECT_FALSE(PlanContainsRequest(PlanOnce(), "r"));

    SendCompletion(*scheduler_, stale_decode, /*tokens=*/{77}, stale_decode.dispatch_raw_end);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);
    EXPECT_EQ(scheduler_->FlatKVCompletionStats().canceled_results, 1u);
    PlanOnce();

    Submit(MakeRequestSpec("r", /*num_pages=*/1, /*start=*/101));
    const FlatKVCompletionInput readmitted = OnlyInput(PlanOnce());
    EXPECT_NE(readmitted.table_generation, stale_decode.table_generation);
    SendCompletion(*scheduler_, stale_decode, /*tokens=*/{77}, stale_decode.dispatch_raw_end);
    EXPECT_EQ(scheduler_->FlatKVCompletionStats().stale_generation_results, 1u);
}

INSTANTIATE_TEST_SUITE_P(AbortAndFinish, FlatKVTerminalSchedulerTest,
                         ::testing::Values(TerminalEvent::kAbort, TerminalEvent::kFinish));

TEST_F(FlatKVReadyPublicationSchedulerTest, OverlappedAbortNeverPublishesCanceledPages) {
    Submit(RequestSpec{.request_id = "producer", .tokens = {1, 2}});
    const FlatKVCompletionInput prefill = OnlyInput(PlanOnce());
    SendCompletion(*scheduler_, prefill, {3}, prefill.dispatch_raw_end);

    std::vector<std::uint64_t> published;
    scheduler_->SetFlatKVCompletionPublisher(
        [&](const FlatKVReadyCompletion& ready) { published.push_back(ready.input.dispatch_seq); });
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
    SendCompletion(*scheduler_, successor, {20, 21, 22, 23}, successor.dispatch_raw_end);

    EXPECT_TRUE(published.empty());
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("producer"), 0u);
    EXPECT_EQ(scheduler_->FlatKVCompletionStats().canceled_results, 2u);
    Submit(RequestSpec{.request_id = "probe", .tokens = {1, 2, 3, 10, 11, 12, 99}});
    EXPECT_EQ(ExtendPrefixForRequest(PlanOnce(), "probe"), 2);
}

TEST_F(FlatKVReadyPublicationSchedulerTest, PrepareFailureRollsBackAndExactRetryCommitsOnce) {
    Submit(MakeRequestSpec("r", /*num_pages=*/2));
    const FlatKVCompletionInput input = OnlyInput(PlanOnce());
    const std::vector<BlockPoolSnapshot> before = scheduler_->FlatPoolSnapshots();
    const std::int32_t token_size_before = scheduler_->GetRequestTokenSize("r");
    std::int32_t attempts = 0;
    scheduler_->SetFlatKVCompletionPublisher([&](const FlatKVReadyCompletion&) {
        if (++attempts == 1) {
            throw std::bad_alloc{};
        }
    });

    EXPECT_THROW(SendCompletion(*scheduler_, input, {99}, input.dispatch_raw_end), std::bad_alloc);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), token_size_before);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 1u);
    EXPECT_EQ(scheduler_->FlatKVCompletionStats().applied_results, 0u);
    const std::vector<BlockPoolSnapshot> after_failure = scheduler_->FlatPoolSnapshots();
    ASSERT_EQ(after_failure.size(), before.size());
    for (std::size_t i = 0; i < before.size(); ++i) {
        EXPECT_EQ(after_failure[i].free_blocks, before[i].free_blocks);
        EXPECT_EQ(after_failure[i].active_blocks, before[i].active_blocks);
        EXPECT_EQ(after_failure[i].cached_evictable_blocks, before[i].cached_evictable_blocks);
        EXPECT_EQ(after_failure[i].pinned_cached_blocks, before[i].pinned_cached_blocks);
    }

    SendCompletion(*scheduler_, input, {99}, input.dispatch_raw_end);
    EXPECT_EQ(attempts, 2);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), token_size_before + 1);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);
    EXPECT_EQ(scheduler_->FlatKVCompletionStats().applied_results, 1u);
    Submit(MakeRequestSpec("probe", /*num_pages=*/2));
    EXPECT_EQ(ExtendPrefixForRequest(PlanOnce(), "probe"), 2);
}

TEST_F(FlatKVReadyPublicationSchedulerTest, PrefixIsInvisibleBeforeFenceAndHittableAfterCompletion) {
    Submit(MakeRequestSpec("producer", /*num_pages=*/2));
    const ExecutionPlan producer_plan = PlanOnce();
    const FlatKVCompletionInput producer = InputForRequest(producer_plan, "producer");

    Submit(MakeRequestSpec("before", /*num_pages=*/2));
    const ExecutionPlan before_plan = PlanOnce();
    EXPECT_EQ(ExtendPrefixForRequest(before_plan, "before"), 0);

    SendCompletion(*scheduler_, producer, /*tokens=*/{99}, producer.dispatch_raw_end);

    Submit(MakeRequestSpec("after", /*num_pages=*/2));
    const ExecutionPlan after_plan = PlanOnce();
    // Match excludes the final prompt token, so a four-token request can reuse
    // exactly its first two-token base page.
    EXPECT_EQ(ExtendPrefixForRequest(after_plan, "after"), 2);
}

#endif

}  // namespace
}  // namespace tokenspeed::test
