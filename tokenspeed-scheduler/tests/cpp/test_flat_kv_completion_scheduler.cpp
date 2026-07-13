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
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <thread>
#include <utility>
#include <vector>

#include "integration_test_helper.h"
#include "resource/radix_tree/tree_node.h"

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

forward::FlatKVCompletion ReadySingleState(const FlatKVCompletionInput& input, std::int32_t accepted_raw_end) {
    return forward::FlatKVCompletion{
        .request_id = input.request_id,
        .table_generation = input.table_generation,
        .dispatch_seq = input.dispatch_seq,
        .accepted_raw_end = accepted_raw_end,
        .protected_raw_end = input.protected_raw_end,
        .groups = {forward::FlatKVGroupCompletion{
            .group_id = "state",
            .completed_domain_mask = 1,
            .domain_valid_ends = {input.dispatch_raw_end},
        }},
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

void SendSingleStateCompletion(Scheduler& scheduler, const FlatKVCompletionInput& input,
                               std::vector<std::int32_t> tokens, std::int32_t accepted_raw_end) {
    ExecutionEvent event;
    event.With(ForwardEvent{forward::ExtendResult{
        .request_id = input.request_id,
        .tokens = std::move(tokens),
        .flat_kv_completion = ReadySingleState(input, accepted_raw_end),
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

PagedCacheGroupConfig V4CompletionGroup(std::string group_id, std::int32_t rows_per_page,
                                        std::int32_t entry_stride_tokens, std::int32_t sliding_window_tokens,
                                        PrefixRole prefix_role) {
    constexpr std::int32_t kTotalBlocks = 1024;
    PagedCacheGroupConfig group;
    group.group_id = std::move(group_id);
    group.rows_per_page = rows_per_page;
    group.entry_stride_tokens = entry_stride_tokens;
    group.total_pages = kTotalBlocks;
    group.block_size = rows_per_page * entry_stride_tokens;
    group.pool_id = group.group_id + ".pool";
    group.prefix_role = prefix_role;
    group.required_producer_domain_mask = 1;
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
                          /*sliding_window_tokens=*/128, PrefixRole::ContinuationState),
        V4CompletionGroup("v4.c4a.compressor_state", /*rows_per_page=*/4, /*entry_stride_tokens=*/1,
                          /*sliding_window_tokens=*/8, PrefixRole::ContinuationState),
        V4CompletionGroup("v4.c4a.compressed_kv", /*rows_per_page=*/64, /*entry_stride_tokens=*/4,
                          /*sliding_window_tokens=*/0, PrefixRole::HistoryAnchor),
        V4CompletionGroup("v4.c128a.compressor_state", /*rows_per_page=*/8, /*entry_stride_tokens=*/1,
                          /*sliding_window_tokens=*/128, PrefixRole::ContinuationState),
        V4CompletionGroup("v4.c128a.compressed_kv", /*rows_per_page=*/2, /*entry_stride_tokens=*/128,
                          /*sliding_window_tokens=*/0, PrefixRole::HistoryAnchor),
        V4CompletionGroup("v4.c4a.indexer_compressor_state", /*rows_per_page=*/4,
                          /*entry_stride_tokens=*/1, /*sliding_window_tokens=*/8, PrefixRole::ContinuationState),
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
        completion.groups.push_back(forward::FlatKVGroupCompletion{
            .group_id = group.group_id,
            .completed_domain_mask = 1,
            .domain_valid_ends = {input.dispatch_raw_end},
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

TEST(FlatKVStructuredOwnershipTest, V4ColdAdmissionIsIndependentOfLegacyBlockSizeAndPrefixSwitch) {
    struct PlanShape {
        std::int32_t extend_prefix_len{};
        std::int32_t input_length{};
        std::vector<std::size_t> group_columns;
        PoolDemand free_blocks;

        bool operator==(const PlanShape&) const = default;
    };

    std::optional<PlanShape> baseline;
    for (const bool disable_prefix_cache : {false, true}) {
        for (const std::int32_t legacy_block_size : {128, 256}) {
            Scheduler scheduler(V4CompletionConfig(legacy_block_size, disable_prefix_cache));
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
                EXPECT_EQ(shape, *baseline)
                    << "structured flat admission must not observe legacy SchedulerConfig.block_size";
            }
        }
    }
}

TEST(FlatKVStructuredOwnershipTest, StructuredSchedulerConstructsNoRadixRoot) {
    TreeNode before;
    Scheduler scheduler(V4CompletionConfig(/*legacy_block_size=*/256, /*disable_prefix_cache=*/false));
    TreeNode after;

    EXPECT_EQ(after.SeqId(), before.SeqId() + 1)
        << "structured explicit Flat construction must not create a radix root TreeNode";
    EXPECT_EQ(scheduler.CalcRollingHash({1, 2, 3, 4, 5, 6, 7, 8}, /*apply_match=*/false).size(), 2u)
        << "pure hashing must use the coordinator's four-token base geometry";
    EXPECT_THROW(scheduler.CalcRollingHash({1, 2, 3, 4}, /*apply_match=*/true), std::runtime_error);
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

TEST(FlatKVLegacyProducerDomainTest, ZeroRequiredMaskCanonicalizesToSingleProducerBit) {
    SchedulerConfig cfg = HeterogeneousCompletionConfig(/*verify_width=*/1, /*overlap_depth=*/0);
    cfg.flat_block_pools.resize(1);
    cfg.paged_cache_groups.resize(1);
    cfg.paged_cache_groups.front().required_producer_domain_mask = 0;
    Scheduler scheduler(std::move(cfg));
    scheduler.SubmitRequests({RequestSpec{.request_id = "r", .tokens = {1, 2}}});
    const FlatKVCompletionInput input = InputForRequest(scheduler.NextExecutionPlan(), "r");

    ExecutionEvent event;
    event.With(ForwardEvent{forward::ExtendResult{
        .request_id = "r",
        .tokens = {3},
        .flat_kv_completion =
            forward::FlatKVCompletion{
                .request_id = input.request_id,
                .table_generation = input.table_generation,
                .dispatch_seq = input.dispatch_seq,
                .accepted_raw_end = input.dispatch_raw_end,
                .protected_raw_end = input.protected_raw_end,
                .groups = {forward::FlatKVGroupCompletion{
                    .group_id = "history",
                    .completed_domain_mask = 1,
                    .domain_valid_ends = {input.dispatch_raw_end},
                }},
            },
    }});

    EXPECT_NO_THROW(scheduler.Advance(std::move(event)));
    EXPECT_EQ(scheduler.FlatKVCompletionOutstandingCount("r"), 0u);
}

PagedCacheGroupConfig SlidingCompletionGroup(std::int32_t total_blocks) {
    PagedCacheGroupConfig state;
    state.group_id = "state";
    state.rows_per_page = 4;
    state.entry_stride_tokens = 1;
    state.total_pages = total_blocks;
    state.block_size = 4;
    state.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    state.sliding_window_tokens = 8;
    state.family = PagedCacheGroupFamily::State;
    state.prefix_role = PrefixRole::ContinuationState;
    state.table_layout = TableLayout::BoundedWindow;
    state.required_producer_domain_mask = 1;
    state.owner_mask = 1;
    state.pool_id = "state-pool";
    return state;
}

SchedulerConfig SlidingCompletionConfig(std::int32_t total_blocks, std::int32_t max_scheduled_tokens,
                                        std::int32_t max_batch_size, std::int32_t verify_width,
                                        bool enable_mixed_prefill_decode) {
    SchedulerConfig cfg{};
    cfg.block_size = 4;
    cfg.device_allocator.total_pages = 0;
    cfg.host_allocator.total_pages = 1;
    cfg.max_scheduled_tokens = max_scheduled_tokens;
    cfg.max_batch_size = max_batch_size;
    cfg.decode_input_tokens = verify_width;
    cfg.overlap_schedule_depth = 1;
    cfg.disable_l2_cache = true;
    cfg.disable_prefix_cache = true;
    cfg.enable_mixed_prefill_decode = enable_mixed_prefill_decode;
    cfg.enable_structured_flat_kv_completion = true;
    cfg.flat_block_pools = {
        FlatBlockPoolConfig{.pool_id = "state-pool", .total_blocks = total_blocks, .bytes_per_block = 32},
    };
    cfg.paged_cache_groups = {SlidingCompletionGroup(total_blocks)};
    return cfg;
}

TEST(SchedulerThreadDomainTest, RejectsForeignMutableEntriesButAllowsImmutableSchemaQueries) {
    Scheduler scheduler(HeterogeneousCompletionConfig(/*verify_width=*/1, /*overlap_depth=*/0));
    std::exception_ptr mutation_error;
    std::exception_ptr read_error;
    std::exception_ptr schema_error;
    std::vector<std::string> group_ids;
    std::vector<std::string> pool_ids;
    std::int32_t total_pages = 0;

    std::thread worker([&] {
        try {
            scheduler.SubmitRequests({RequestSpec{.request_id = "foreign", .tokens = {1, 2}}});
        } catch (...) {
            mutation_error = std::current_exception();
        }
        try {
            (void)scheduler.WaitingSize();
        } catch (...) {
            read_error = std::current_exception();
        }
        try {
            group_ids = scheduler.PagedCacheGroupIds();
            pool_ids = scheduler.FlatPoolIds();
            total_pages = scheduler.PagedCacheGroupTotalPages("history");
        } catch (...) {
            schema_error = std::current_exception();
        }
    });
    worker.join();

    ASSERT_NE(mutation_error, nullptr);
    ASSERT_NE(read_error, nullptr);
    EXPECT_THROW(std::rethrow_exception(mutation_error), std::logic_error);
    EXPECT_THROW(std::rethrow_exception(read_error), std::logic_error);
    EXPECT_EQ(schema_error, nullptr);
    EXPECT_EQ(group_ids, (std::vector<std::string>{"history", "state"}));
    EXPECT_EQ(pool_ids, (std::vector<std::string>{"history-pool", "state-pool"}));
    EXPECT_EQ(total_pages, 512);
    EXPECT_EQ(scheduler.WaitingSize(), 0u);
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

class FlatKVLegacyCompletionSchedulerTest : public FlatKVCompletionSchedulerTest {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FlatKVCompletionSchedulerTest::MakeConfig();
        cfg.enable_structured_flat_kv_completion = false;
        cfg.overlap_schedule_depth = 0;
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

class FlatKVFirstCompletionStarvationTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        constexpr std::int32_t kTotalBlocks = 3;  // null page 0 + exactly two usable pages per pool
        SchedulerConfig cfg = HeterogeneousCompletionConfig(/*verify_width=*/4, /*overlap_depth=*/0);
        cfg.max_scheduled_tokens = 4;
        cfg.flat_block_pools = {
            FlatBlockPoolConfig{.pool_id = "history-pool", .total_blocks = kTotalBlocks, .bytes_per_block = 128},
            FlatBlockPoolConfig{.pool_id = "state-pool", .total_blocks = kTotalBlocks, .bytes_per_block = 32},
        };
        cfg.paged_cache_groups = HeterogeneousCompletionGroups(kTotalBlocks);
        return cfg;
    }
};

TEST(FlatKVSlidingCompletionBoundTest, FullAcceptedMixedSuccessorsDoNotAccumulateUnboundedRows) {
    constexpr std::int32_t kMaxScheduledTokens = 64;
    constexpr std::int32_t kVerifyWidth = 4;
    constexpr std::int32_t kOverlapDepth = 1;
    constexpr std::int32_t kRawTokensPerPage = 4;
    constexpr std::int32_t kSlidingWindowTokens = 8;
    constexpr std::size_t kExportCols = (kSlidingWindowTokens + (kOverlapDepth + 1) * kMaxScheduledTokens +
                                         (kOverlapDepth + 1) * kVerifyWidth + kRawTokensPerPage - 1) /
                                            kRawTokensPerPage +
                                        1;

    Scheduler scheduler(SlidingCompletionConfig(/*total_blocks=*/512, kMaxScheduledTokens,
                                                /*max_batch_size=*/9, kVerifyWidth,
                                                /*enable_mixed_prefill_decode=*/true));

    std::vector<std::string> helper_ids;
    std::vector<RequestSpec> helper_specs;
    for (std::int32_t i = 0; i < 8; ++i) {
        const std::string request_id = "decode-" + std::to_string(i);
        helper_ids.push_back(request_id);
        helper_specs.push_back(RequestSpec{.request_id = request_id, .tokens = {1, 2, 3, 4}});
    }
    scheduler.SubmitRequests(std::move(helper_specs));

    auto complete_full_plan = [&](const ExecutionPlan& plan) {
        const FlatForwardOperation* op = FindCompletionForward(plan);
        ASSERT_NE(op, nullptr);
        ExecutionEvent event;
        for (std::size_t row = 0; row < op->request_ids.size(); ++row) {
            const FlatKVCompletionInput& input = op->flat_kv_completion_inputs.at(row);
            std::vector<std::int32_t> tokens;
            if (input.request_id != "prefill") {
                tokens.assign(static_cast<std::size_t>(kVerifyWidth), 1000 + static_cast<std::int32_t>(row));
            }
            event.With(ForwardEvent{forward::ExtendResult{
                .request_id = input.request_id,
                .tokens = std::move(tokens),
                .flat_kv_completion = ReadySingleState(input, input.dispatch_raw_end),
            }});
        }
        scheduler.Advance(std::move(event));
    };

    const ExecutionPlan helper_prefill = scheduler.NextExecutionPlan();
    const FlatForwardOperation* helper_op = FindCompletionForward(helper_prefill);
    ASSERT_NE(helper_op, nullptr);
    ASSERT_EQ(helper_op->request_ids.size(), helper_ids.size());
    complete_full_plan(helper_prefill);

    scheduler.SubmitRequests({RequestSpec{
        .request_id = "prefill",
        .tokens = std::vector<std::int32_t>(256, 7),
    }});

    auto expect_bounded_mixed_plan = [&](const ExecutionPlan& plan, std::int32_t round) {
        SCOPED_TRACE(round);
        const FlatForwardOperation* op = FindCompletionForward(plan);
        ASSERT_NE(op, nullptr);
        ASSERT_EQ(op->request_ids.size(), 9u);
        ASSERT_EQ(op->num_extends(), 1u);
        ASSERT_TRUE(PlanContainsRequest(plan, "prefill"));
        EXPECT_LE(FlatTableForRequest(plan, "prefill", "state").size(), kExportCols);
    };

    std::vector<ExecutionPlan> in_flight;
    in_flight.push_back(scheduler.NextExecutionPlan());
    expect_bounded_mixed_plan(in_flight.back(), /*round=*/0);
    in_flight.push_back(scheduler.NextExecutionPlan());
    expect_bounded_mixed_plan(in_flight.back(), /*round=*/1);

    // Keep exactly one full-accepted successor outstanding while repeatedly
    // retiring its predecessor. The bounded row may carry at most two prefill
    // chunks plus the sliding retention and protected decode horizon; retired
    // prefixes must not accumulate merely because the successor fence is live.
    for (std::int32_t round = 2; round <= 4; ++round) {
        complete_full_plan(in_flight.at(static_cast<std::size_t>(round - 2)));
        ASSERT_EQ(scheduler.FlatKVCompletionOutstandingCount("prefill"), 1u);
        in_flight.push_back(scheduler.NextExecutionPlan());
        expect_bounded_mixed_plan(in_flight.back(), round);
    }
}

TEST(FlatKVSlidingCompletionBoundTest, FinalPrefillCarryFitsCudaGraphCaptureWidth) {
    constexpr std::int32_t kPrefillChunkTokens = 64;
    constexpr std::int32_t kVerifyWidth = 1;
    constexpr std::int32_t kOverlapDepth = 1;
    constexpr std::int32_t kRawTokensPerPage = 4;
    constexpr std::int32_t kSlidingWindowTokens = 8;
    constexpr std::size_t kCaptureColsWithoutPrefillCarry = 4;
    constexpr std::size_t kCaptureCols = (kSlidingWindowTokens + kOverlapDepth * kPrefillChunkTokens +
                                          (kOverlapDepth + 1) * kVerifyWidth + kRawTokensPerPage - 1) /
                                             kRawTokensPerPage +
                                         1;

    // This is the exact state-pool budget for one live request: two retained
    // pages, sixteen scheduled pages, one row tail, one protected page, and
    // the null page.
    Scheduler scheduler(SlidingCompletionConfig(/*total_blocks=*/21, kPrefillChunkTokens,
                                                /*max_batch_size=*/1, kVerifyWidth,
                                                /*enable_mixed_prefill_decode=*/false));
    scheduler.SubmitRequests({RequestSpec{
        .request_id = "r",
        .tokens = std::vector<std::int32_t>(2 * kPrefillChunkTokens, 7),
    }});

    const ExecutionPlan first_prefill_plan = scheduler.NextExecutionPlan();
    const FlatKVCompletionInput first_prefill = InputForRequest(first_prefill_plan, "r");
    ASSERT_EQ(first_prefill.dispatch_raw_end, kPrefillChunkTokens);
    SendSingleStateCompletion(scheduler, first_prefill, /*tokens=*/{}, first_prefill.dispatch_raw_end);

    const ExecutionPlan final_prefill_plan = scheduler.NextExecutionPlan();
    const FlatForwardOperation* final_prefill_op = FindCompletionForward(final_prefill_plan);
    ASSERT_NE(final_prefill_op, nullptr);
    ASSERT_EQ(final_prefill_op->num_extends(), 1u);
    const FlatKVCompletionInput final_prefill = InputForRequest(final_prefill_plan, "r");
    ASSERT_EQ(final_prefill.dispatch_raw_end, 2 * kPrefillChunkTokens);

    // PrefillDone is schedulable before the final prefill fence arrives. The
    // decode row therefore still carries that in-flight chunk, and its CUDA
    // Graph capture bound must include overlap_depth * prefill_chunk_tokens.
    const ExecutionPlan first_decode_plan = scheduler.NextExecutionPlan();
    const FlatForwardOperation* first_decode_op = FindCompletionForward(first_decode_plan);
    ASSERT_NE(first_decode_op, nullptr);
    ASSERT_EQ(first_decode_op->num_extends(), 0u);
    ASSERT_TRUE(PlanContainsRequest(first_decode_plan, "r"));
    ASSERT_EQ(scheduler.FlatKVCompletionOutstandingCount("r"), 2u);
    const std::size_t actual_cols = FlatTableForRequest(first_decode_plan, "r", "state").size();
    EXPECT_GT(actual_cols, kCaptureColsWithoutPrefillCarry);
    EXPECT_LE(actual_cols, kCaptureCols);

    SendSingleStateCompletion(scheduler, final_prefill, /*tokens=*/{9}, final_prefill.dispatch_raw_end);
    ASSERT_EQ(scheduler.FlatKVCompletionOutstandingCount("r"), 1u);

    const ExecutionPlan successor_decode_plan = scheduler.NextExecutionPlan();
    const FlatForwardOperation* successor_decode_op = FindCompletionForward(successor_decode_plan);
    ASSERT_NE(successor_decode_op, nullptr);
    const std::size_t row = FlatRowForRequest(successor_decode_plan, "r");
    const auto& state = successor_decode_op->flat_block_tables.at("state");
    // Full-accept front reclaim may advance through page 29, but the first
    // in-flight decode can still read the trailing window beginning on page
    // 30. Keep that earliest successor page while bounding the exported row.
    EXPECT_EQ(state.bases.at(row), 30);
    ExpectFlatGroupCoversRawRange(successor_decode_plan, "r", "state", kRawTokensPerPage,
                                  /*raw_begin=*/121, /*raw_end=*/129);
    EXPECT_LE(state.Row(row).size(), kCaptureCols);
}

TEST(FlatKVOverlapReservationTest, ReservesFullProtectedHorizonThenConsumesOneDispatchAtATime) {
    Scheduler scheduler(HeterogeneousCompletionConfig(/*verify_width=*/4, /*overlap_depth=*/1));
    scheduler.SubmitRequests({RequestSpec{.request_id = "r", .tokens = {1, 2}}});

    const ExecutionPlan prefill_plan = scheduler.NextExecutionPlan();
    const FlatKVCompletionInput prefill = InputForRequest(prefill_plan, "r");
    // Formula: verify_width * (overlap_depth + 1) = 4 * 2 = 8
    // protected decode tokens. With two-token pages, each pool reserves four
    // pages beyond the one prompt page.
    EXPECT_EQ(scheduler.FlatReservedBlocksByPool(), (PoolDemand{4, 4}));
    EXPECT_EQ(FlatTableForRequest(prefill_plan, "r", "history").size(), 1u);
    EXPECT_EQ(FlatTableForRequest(prefill_plan, "r", "state").size(), 1u);

    SendCompletion(scheduler, prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);
    const ExecutionPlan first_decode_plan = scheduler.NextExecutionPlan();
    const FlatKVCompletionInput first_decode = InputForRequest(first_decode_plan, "r");
    EXPECT_EQ(first_decode.dispatch_raw_end, 6);
    EXPECT_EQ(first_decode.protected_raw_end, 10);
    EXPECT_EQ(FlatTableForRequest(first_decode_plan, "r", "history").size(), 3u);
    EXPECT_EQ(FlatTableForRequest(first_decode_plan, "r", "state").size(), 3u);
    // The first physical Acquire advances by one verify width (two pages per
    // pool); only the one not-yet-materialized overlap slot remains reserved.
    EXPECT_EQ(scheduler.FlatReservedBlocksByPool(), (PoolDemand{2, 2}));

    const ExecutionPlan successor_plan = scheduler.NextExecutionPlan();
    const FlatKVCompletionInput successor = InputForRequest(successor_plan, "r");
    EXPECT_EQ(successor.dispatch_raw_end, 10);
    EXPECT_EQ(successor.protected_raw_end, first_decode.protected_raw_end);
    EXPECT_EQ(FlatTableForRequest(successor_plan, "r", "history").size(), 5u);
    EXPECT_EQ(FlatTableForRequest(successor_plan, "r", "state").size(), 5u);
    EXPECT_EQ(scheduler.FlatReservedBlocksByPool(), (PoolDemand{0, 0}));
}

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

TEST_F(FlatKVLegacyCompletionSchedulerTest, DefaultOmitsCompletionAbiAndAcceptsLegacyResult) {
    EXPECT_FALSE(SchedulerConfig{}.enable_structured_flat_kv_completion);
    Submit(MakeRequestSpec("r", /*num_pages=*/1));

    const ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindCompletionForward(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_TRUE(op->flat_kv_completion_inputs.empty());

    ExecutionEvent event;
    event.With(ForwardEvent{forward::ExtendResult{.request_id = "r", .tokens = {99}}});
    scheduler_->Advance(std::move(event));
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 3);
}

TEST_F(FlatKVCompletionSchedulerTest, MidPrefillEmptyCompletionClearsDebtAndKeepsGeneration) {
    std::vector<std::uint64_t> published;
    scheduler_->SetFlatKVCompletionPublisher(
        [&](const FlatKVReadyCompletion& ready) { published.push_back(ready.input.dispatch_seq); });
    Submit(MakeRequestSpec("r", /*num_pages=*/4));

    const FlatKVCompletionInput first = OnlyInput(PlanOnce());
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 1u);
    SendCompletion(*scheduler_, first, /*tokens=*/{}, first.dispatch_raw_end);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);

    const FlatKVCompletionInput second = OnlyInput(PlanOnce());
    EXPECT_EQ(second.table_generation, first.table_generation);
    EXPECT_GT(second.dispatch_seq, first.dispatch_seq);
    SendCompletion(*scheduler_, second, /*tokens=*/{99}, second.dispatch_raw_end);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 9);
    EXPECT_EQ(published, (std::vector<std::uint64_t>{first.dispatch_seq, second.dispatch_seq}));
}

TEST_F(FlatKVCompletionSchedulerTest, ThrowingPrepareObserverLeavesTokensCacheAndCompletionRetryable) {
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

    EXPECT_THROW(SendCompletion(*scheduler_, input, /*tokens=*/{99}, input.dispatch_raw_end), std::bad_alloc);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), token_size_before);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 1u);
    const std::vector<BlockPoolSnapshot> after_failure = scheduler_->FlatPoolSnapshots();
    ASSERT_EQ(after_failure.size(), before.size());
    for (std::size_t i = 0; i < before.size(); ++i) {
        EXPECT_EQ(after_failure[i].free_blocks, before[i].free_blocks);
        EXPECT_EQ(after_failure[i].active_blocks, before[i].active_blocks);
        EXPECT_EQ(after_failure[i].cached_evictable_blocks, before[i].cached_evictable_blocks);
        EXPECT_EQ(after_failure[i].pinned_cached_blocks, before[i].pinned_cached_blocks);
    }

    SendCompletion(*scheduler_, input, /*tokens=*/{99}, input.dispatch_raw_end);
    EXPECT_EQ(attempts, 2);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), token_size_before + 1);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);
}

TEST_F(FlatKVCompletionSchedulerTest, AbortAfterPrepareFailureSynchronouslyAppliesTerminal) {
    Submit(MakeRequestSpec("r", /*num_pages=*/2));
    const FlatKVCompletionInput input = OnlyInput(PlanOnce());
    const std::int32_t token_size_before = scheduler_->GetRequestTokenSize("r");
    std::int32_t attempts = 0;
    scheduler_->SetFlatKVCompletionPublisher([&](const FlatKVReadyCompletion&) {
        if (++attempts == 1) {
            throw std::bad_alloc{};
        }
    });
    EXPECT_THROW(SendCompletion(*scheduler_, input, /*tokens=*/{99}, input.dispatch_raw_end), std::bad_alloc);
    ASSERT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 1u);

    ExecutionEvent abort;
    abort.With(ForwardEvent{forward::Abort{.request_id = "r"}});
    scheduler_->Advance(std::move(abort));

    EXPECT_EQ(attempts, 1);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), token_size_before);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);
    EXPECT_EQ(scheduler_->FlatKVCompletionStats().canceled_results, 1u);
    EXPECT_FALSE(PlanContainsRequest(PlanOnce(), "r"));
    EXPECT_TRUE(scheduler_->FlatKVQuiescent());
}

TEST_F(FlatKVCompletionSchedulerTest, FinishAfterPrepareFailureSynchronouslyAppliesTerminal) {
    Submit(MakeRequestSpec("r", /*num_pages=*/1));
    const FlatKVCompletionInput prefill = OnlyInput(PlanOnce());
    SendCompletion(*scheduler_, prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);
    const FlatKVCompletionInput decode = OnlyInput(PlanOnce());
    const std::int32_t token_size_before = scheduler_->GetRequestTokenSize("r");
    std::int32_t attempts = 0;
    scheduler_->SetFlatKVCompletionPublisher([&](const FlatKVReadyCompletion&) {
        if (++attempts == 1) {
            throw std::bad_alloc{};
        }
    });
    EXPECT_THROW(SendCompletion(*scheduler_, decode, /*tokens=*/{99, 100, 101, 102}, decode.dispatch_raw_end),
                 std::bad_alloc);
    ASSERT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 1u);

    SendFinish("r");

    EXPECT_EQ(attempts, 1);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), token_size_before);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);
    EXPECT_EQ(scheduler_->FlatKVCompletionStats().canceled_results, 1u);
    EXPECT_FALSE(PlanContainsRequest(PlanOnce(), "r"));
    EXPECT_TRUE(scheduler_->FlatKVQuiescent());
}

TEST_F(FlatKVCompletionSchedulerTest, OutOfOrderPrefillResultsDoNotMutateFsmUntilGapCloses) {
    Submit(MakeRequestSpec("r", /*num_pages=*/4));
    const FlatKVCompletionInput first = OnlyInput(PlanOnce());
    const FlatKVCompletionInput second = OnlyInput(PlanOnce());

    SendCompletion(*scheduler_, second, /*tokens=*/{99}, second.dispatch_raw_end);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 8);
    EXPECT_EQ(scheduler_->FlatKVCompletionBufferedCount("r"), 1u);

    SendCompletion(*scheduler_, first, /*tokens=*/{}, first.dispatch_raw_end);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 9);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);
    EXPECT_THROW(SendCompletion(*scheduler_, second, /*tokens=*/{100}, second.dispatch_raw_end), std::invalid_argument);
}

TEST_F(FlatKVCompletionSchedulerTest, ShortDecodeInvalidatesBufferedSuccessorWithoutPublishingAcrossGap) {
    std::vector<std::uint64_t> published;
    scheduler_->SetFlatKVCompletionPublisher(
        [&](const FlatKVReadyCompletion& ready) { published.push_back(ready.input.dispatch_seq); });
    Submit(MakeRequestSpec("r", /*num_pages=*/1));
    const FlatKVCompletionInput prefill = OnlyInput(PlanOnce());
    SendCompletion(*scheduler_, prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);

    const FlatKVCompletionInput first_decode = OnlyInput(PlanOnce());
    const FlatKVCompletionInput successor = OnlyInput(PlanOnce());
    ASSERT_EQ(successor.dispatch_raw_end, first_decode.dispatch_raw_end + 4);
    ASSERT_EQ(successor.protected_raw_end, first_decode.protected_raw_end);
    SendCompletion(*scheduler_, successor, /*tokens=*/{20, 21, 22, 23}, successor.dispatch_raw_end);
    SendCompletion(*scheduler_, first_decode, /*tokens=*/{10, 11}, first_decode.dispatch_raw_end - 2);

    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 5);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);
    EXPECT_EQ(published, (std::vector<std::uint64_t>{prefill.dispatch_seq, first_decode.dispatch_seq}));
    const FlatKVCompletionInput recovered = OnlyInput(PlanOnce());
    EXPECT_GT(recovered.dispatch_seq, successor.dispatch_seq);
}

TEST_F(FlatKVCompletionSchedulerTest, FullOverlapResultsCommitProjectedIntervalsInOrder) {
    Submit(MakeRequestSpec("r", /*num_pages=*/1));
    const FlatKVCompletionInput prefill = OnlyInput(PlanOnce());
    SendCompletion(*scheduler_, prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);

    const FlatKVCompletionInput first_decode = OnlyInput(PlanOnce());
    const FlatKVCompletionInput successor = OnlyInput(PlanOnce());
    ASSERT_EQ(successor.dispatch_raw_end, first_decode.dispatch_raw_end + 4);
    ASSERT_EQ(successor.protected_raw_end, first_decode.protected_raw_end);

    SendCompletion(*scheduler_, successor, /*tokens=*/{20, 21, 22, 23}, successor.dispatch_raw_end);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 3);
    EXPECT_EQ(scheduler_->FlatKVCompletionBufferedCount("r"), 1u);

    SendCompletion(*scheduler_, first_decode, /*tokens=*/{10, 11, 12, 13}, first_decode.dispatch_raw_end);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 11);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);

    const FlatKVCompletionInput next = OnlyInput(PlanOnce());
    EXPECT_EQ(next.dispatch_raw_end, successor.dispatch_raw_end + 4);
}

TEST_F(FlatKVFirstCompletionStarvationTest, InflightFirstChunkCannotBeOomReclaimedBeforeItsFence) {
    Submit(RequestSpec{.request_id = "r", .tokens = std::vector<std::int32_t>(12, 7)});
    const ExecutionPlan first_plan = PlanOnce();
    const FlatKVCompletionInput first = InputForRequest(first_plan, "r");
    ASSERT_EQ(first.dispatch_raw_end, 4);
    ASSERT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 1u);
    const auto history_pages = scheduler_->GetRequestPagedCachePageIds("r", "history");
    const auto state_pages = scheduler_->GetRequestPagedCachePageIds("r", "state");
    const auto snapshots = scheduler_->FlatPoolSnapshots();

    for (int round = 0; round < 3; ++round) {
        const ExecutionPlan starved = PlanOnce();
        const FlatForwardOperation* op = FindCompletionForward(starved);
        ASSERT_NE(op, nullptr);
        EXPECT_TRUE(op->request_ids.empty());
        EXPECT_TRUE(starved.flat_oom_request_ids.empty());
        EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 1u);
        EXPECT_EQ(scheduler_->GetRequestPagedCachePageIds("r", "history"), history_pages);
        EXPECT_EQ(scheduler_->GetRequestPagedCachePageIds("r", "state"), state_pages);
        const auto current = scheduler_->FlatPoolSnapshots();
        ASSERT_EQ(current.size(), snapshots.size());
        for (std::size_t i = 0; i < current.size(); ++i) {
            EXPECT_EQ(current[i].free_blocks, snapshots[i].free_blocks);
            EXPECT_EQ(current[i].active_blocks, snapshots[i].active_blocks);
        }
    }

    SendCompletion(*scheduler_, first, /*tokens=*/{}, first.dispatch_raw_end);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);
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

TEST_F(FlatKVCompletionSchedulerTest, AbortQuarantinesPagesUntilLateCompletionFenceArrives) {
    Submit(MakeRequestSpec("r", /*num_pages=*/2));
    const FlatKVCompletionInput in_flight = OnlyInput(PlanOnce());
    const std::int32_t free_before_abort = scheduler_->FlatPoolFreeBlocks();
    ExecutionEvent abort;
    abort.With(ForwardEvent{forward::Abort{.request_id = "r"}});
    scheduler_->Advance(std::move(abort));
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 1u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_before_abort);
    EXPECT_FALSE(PlanContainsRequest(PlanOnce(), "r"));

    SendCompletion(*scheduler_, in_flight, /*tokens=*/{}, in_flight.dispatch_raw_end);

    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 0u);
    EXPECT_EQ(scheduler_->FlatKVCompletionStats().canceled_results, 1u);
    EXPECT_EQ(scheduler_->FlatKVCompletionStats().stale_generation_results, 0u);
    EXPECT_GT(scheduler_->FlatPoolFreeBlocks(), free_before_abort);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r"), 4);
}

TEST_F(FlatKVReadyPublicationSchedulerTest, AbortBeforeOverlappedCompletionsNeverPublishesCanceledPages) {
    Submit(RequestSpec{.request_id = "producer", .tokens = {1, 2}});
    const FlatKVCompletionInput prefill = OnlyInput(PlanOnce());
    SendCompletion(*scheduler_, prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);

    std::vector<std::uint64_t> published;
    scheduler_->SetFlatKVCompletionPublisher(
        [&](const FlatKVReadyCompletion& ready) { published.push_back(ready.input.dispatch_seq); });
    const FlatKVCompletionInput first_decode = OnlyInput(PlanOnce());
    const FlatKVCompletionInput successor = OnlyInput(PlanOnce());
    ASSERT_EQ(scheduler_->FlatKVCompletionOutstandingCount("producer"), 2u);
    const std::int32_t free_while_inflight = scheduler_->FlatPoolFreeBlocks();

    // This is the exact Python event order for a structured NaN terminal: the
    // abort cancels every exported table before the current execution fence is
    // retired later in the same scheduler advance.
    ExecutionEvent abort_then_first;
    abort_then_first.With(ForwardEvent{forward::Abort{.request_id = "producer"}});
    abort_then_first.With(ForwardEvent{forward::ExtendResult{
        .request_id = "producer",
        .tokens = {10, 11, 12, 13},
        .flat_kv_completion = Ready(first_decode, first_decode.dispatch_raw_end),
    }});
    scheduler_->Advance(std::move(abort_then_first));
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("producer"), 1u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_while_inflight);
    EXPECT_FALSE(PlanContainsRequest(PlanOnce(), "producer"));
    SendCompletion(*scheduler_, successor, /*tokens=*/{20, 21, 22, 23}, successor.dispatch_raw_end);

    EXPECT_TRUE(published.empty());
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("producer"), 0u);
    EXPECT_EQ(scheduler_->FlatKVCompletionStats().canceled_results, 2u);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("producer"), 3);

    Submit(RequestSpec{.request_id = "probe", .tokens = {1, 2, 3, 10, 11, 12, 99}});
    const ExecutionPlan probe_plan = PlanOnce();
    // The healthy prefill page remains reusable, but neither canceled decode
    // dispatch may extend the visible prefix with suspect KV.
    EXPECT_EQ(ExtendPrefixForRequest(probe_plan, "probe"), 2);
}

TEST_F(FlatKVCompletionSchedulerTest, FinishWaitsForFenceAndReadmitNeverReusesGeneration) {
    Submit(MakeRequestSpec("r", /*num_pages=*/1));
    const FlatKVCompletionInput prefill = OnlyInput(PlanOnce());
    SendCompletion(*scheduler_, prefill, /*tokens=*/{3}, prefill.dispatch_raw_end);
    const FlatKVCompletionInput stale_decode = OnlyInput(PlanOnce());
    SendFinish("r");
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

TEST_F(FlatKVCompletionSchedulerTest, OutstandingDispatchesAreBoundedByOverlapDepth) {
    Submit(MakeRequestSpec("r", /*num_pages=*/4));
    const FlatKVCompletionInput first = OnlyInput(PlanOnce());
    const FlatKVCompletionInput second = OnlyInput(PlanOnce());
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 2u);
    EXPECT_FALSE(PlanContainsRequest(PlanOnce(), "r"));

    SendCompletion(*scheduler_, first, /*tokens=*/{}, first.dispatch_raw_end);
    EXPECT_EQ(scheduler_->FlatKVCompletionOutstandingCount("r"), 1u);
    EXPECT_TRUE(PlanContainsRequest(PlanOnce(), "r"));
    (void)second;
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

TEST_F(FlatKVReadyPublicationSchedulerTest, PrefixRestoreSeedsGenerationProgressAtClaimedBoundary) {
    Submit(MakeRequestSpec("producer", /*num_pages=*/2));
    const FlatKVCompletionInput producer = InputForRequest(PlanOnce(), "producer");
    SendCompletion(*scheduler_, producer, /*tokens=*/{99}, producer.dispatch_raw_end);

    std::vector<std::vector<std::int32_t>> published_ready_ends;
    scheduler_->SetFlatKVCompletionPublisher(
        [&](const FlatKVReadyCompletion& ready) { published_ready_ends.push_back(ready.ready_raw_ends); });
    Submit(MakeRequestSpec("restored", /*num_pages=*/2));
    const ExecutionPlan restored_plan = PlanOnce();
    EXPECT_EQ(ExtendPrefixForRequest(restored_plan, "restored"), 2);
    const FlatKVCompletionInput restored = InputForRequest(restored_plan, "restored");
    ASSERT_EQ(restored.dispatch_raw_start, 2);

    auto completion = Ready(restored, restored.dispatch_raw_end);
    for (forward::FlatKVGroupCompletion& group : completion.groups) {
        std::ranges::fill(group.domain_valid_ends, 0);
    }
    ExecutionEvent event;
    event.With(ForwardEvent{forward::ExtendResult{
        .request_id = restored.request_id,
        .tokens = {100},
        .flat_kv_completion = std::move(completion),
    }});
    scheduler_->Advance(std::move(event));

    ASSERT_EQ(published_ready_ends.size(), 1u);
    EXPECT_EQ(published_ready_ends.front(), (std::vector<std::int32_t>{2, 2}));
}

#else

TEST(FlatKVCompletionRadixNoOpTest, OptionalCompletionPreservesLegacyTokenResult) {
    SchedulerConfig config{};
    config.block_size = 2;
    config.device_allocator.total_pages = 32;
    config.host_allocator.total_pages = 1;
    config.max_scheduled_tokens = 16;
    config.max_batch_size = 4;
    config.disable_l2_cache = true;
    Scheduler scheduler(config);
    scheduler.SubmitRequests({RequestSpec{.request_id = "r", .tokens = {1, 2}}});
    const FlatForwardOperation* op = FindCompletionForward(scheduler.NextExecutionPlan());
    ASSERT_NE(op, nullptr);
    EXPECT_TRUE(op->flat_kv_completion_inputs.empty());

    ExecutionEvent event;
    event.With(ForwardEvent{forward::ExtendResult{
        .request_id = "r",
        .tokens = {3},
        .flat_kv_completion =
            forward::FlatKVCompletion{
                .request_id = "r",
                .table_generation = 99,
                .dispatch_seq = 77,
            },
    }});
    scheduler.Advance(std::move(event));
    EXPECT_EQ(scheduler.GetRequestTokenSize("r"), 3);
}

#endif

}  // namespace
}  // namespace tokenspeed::test
