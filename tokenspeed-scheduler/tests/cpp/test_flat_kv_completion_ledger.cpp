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

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "scheduler/flat_kv_completion_ledger.h"

namespace tokenspeed::test {
namespace {

static_assert(std::is_same_v<decltype(FlatKVReadyCompletion::request_id), const std::string&>);
static_assert(std::is_same_v<decltype(FlatKVReadyCompletion::tokens), std::span<const std::int32_t>>);

std::vector<KvCacheGroupSchema> CompletionSchema(std::initializer_list<std::int32_t> strides) {
    std::vector<KvCacheGroupSchema> schema;
    schema.reserve(strides.size());
    std::size_t index = 0;
    for (const std::int32_t stride : strides) {
        schema.push_back(KvCacheGroupSchema{
            .group_id = std::to_string(index++),
            .kind = AttnKind::kFull,
            .block_size = stride,
            .rows_per_page = 1,
            .entry_stride_tokens = stride,
            .sliding_window = 0,
            .pool_index = 0,
            .prefix_role = KvPrefixRole::kHistoryAnchor,
            .table_layout = KvTableLayout::kAbsolute,
            .owner_mask = 0,
        });
    }
    return schema;
}

forward::FlatKVCompletion Complete(const FlatKVCompletionInput& input, std::int32_t accepted_raw_end) {
    return forward::FlatKVCompletion{
        .table_generation = input.table_generation,
        .dispatch_seq = input.dispatch_seq,
        .accepted_raw_end = accepted_raw_end,
    };
}

FlatKVCompletionInput Dispatch(FlatKVCompletionLedger& ledger, FlatKVCompletionState& state, std::int32_t raw_start,
                               std::int32_t raw_end, bool apply_fsm_result = true) {
    return ledger.RecordDispatch(state, FlatKVDispatchSpec{
                                            .dispatch_raw_start = raw_start,
                                            .dispatch_raw_end = raw_end,
                                            .protected_raw_end = raw_end + 8,
                                            .apply_fsm_result = apply_fsm_result,
                                        });
}

std::size_t CommitPrepared(FlatKVCompletionLedger& ledger, FlatKVCompletionState& state,
                           const FlatKVCompletionPrepareResult& prepared) {
    EXPECT_NE(prepared.disposition, FlatKVCompletionDisposition::kStale);
    EXPECT_TRUE(prepared.ticket.has_value());
    return ledger.Commit(state, *prepared.ticket);
}

TEST(FlatKVCompletionLedgerTest, ValidatesFixedDepthAndGroupStrides) {
    const std::vector<KvCacheGroupSchema> valid = CompletionSchema({1});
    EXPECT_THROW((FlatKVCompletionLedger{0, valid}), std::invalid_argument);
    EXPECT_THROW((FlatKVCompletionLedger{3, valid}), std::invalid_argument);
    EXPECT_THROW((FlatKVCompletionLedger{1, std::span<const KvCacheGroupSchema>{}}), std::invalid_argument);
    std::vector<KvCacheGroupSchema> invalid = CompletionSchema({1});
    invalid[0].entry_stride_tokens = 0;
    EXPECT_THROW((FlatKVCompletionLedger{1, invalid}), std::invalid_argument);
}

TEST(FlatKVCompletionLedgerTest, PrepareDerivesStaticGroupBoundariesWithoutRetiringFront) {
    struct Case {
        std::int32_t accepted;
        std::vector<std::int32_t> expected;
    };
    const std::vector<KvCacheGroupSchema> schema = CompletionSchema({1, 4, 128});
    for (const Case& test :
         std::vector<Case>{{0, {0, 0, 0}}, {7, {7, 4, 0}}, {128, {128, 128, 128}}, {585, {585, 584, 512}}}) {
        FlatKVCompletionLedger ledger(/*max_outstanding_per_request=*/1, schema);
        FlatKVCompletionState state;
        const FlatKVCompletionInput input = Dispatch(ledger, state, 0, 585);

        const FlatKVCompletionPrepareResult prepared = ledger.Prepare(state, "r", Complete(input, test.accepted), {});

        ASSERT_EQ(prepared.disposition, FlatKVCompletionDisposition::kApplied);
        ASSERT_TRUE(prepared.ready.has_value());
        EXPECT_EQ(
            (std::vector<std::int32_t>{prepared.ready->ready_raw_ends.begin(), prepared.ready->ready_raw_ends.end()}),
            test.expected);
        EXPECT_EQ(prepared.ready->accepted_raw_end, test.accepted);
        EXPECT_EQ(state.Snapshot().outstanding_count, 1u) << "Prepare must not cross the FIFO progress boundary";
        EXPECT_EQ(CommitPrepared(ledger, state, prepared), 0u);
    }
}

TEST(FlatKVCompletionLedgerTest, RejectsOutOfOrderAndRetiresStrictFifo) {
    const std::vector<KvCacheGroupSchema> schema = CompletionSchema({1});
    FlatKVCompletionLedger ledger(/*max_outstanding_per_request=*/2, schema);
    FlatKVCompletionState state;
    const FlatKVCompletionInput first = Dispatch(ledger, state, 0, 4);
    const FlatKVCompletionInput second = Dispatch(ledger, state, 4, 8);

    EXPECT_THROW(ledger.Prepare(state, "r", Complete(second, 8), {}), std::invalid_argument);
    EXPECT_EQ(state.Snapshot().outstanding_count, 2u);
    const auto first_prepared = ledger.Prepare(state, "r", Complete(first, 4), {10});
    ASSERT_TRUE(first_prepared.ready.has_value());
    EXPECT_EQ(first_prepared.ready->dispatch_seq, first.dispatch_seq);
    EXPECT_EQ(CommitPrepared(ledger, state, first_prepared), 1u);
    const auto second_prepared = ledger.Prepare(state, "r", Complete(second, 8), {11});
    ASSERT_TRUE(second_prepared.ready.has_value());
    EXPECT_EQ(second_prepared.ready->dispatch_seq, second.dispatch_seq);
    EXPECT_EQ(CommitPrepared(ledger, state, second_prepared), 0u);
}

TEST(FlatKVCompletionLedgerTest, RejectsStaleGenerationAndRetiredSequenceWithoutMutation) {
    const std::vector<KvCacheGroupSchema> schema = CompletionSchema({1});
    FlatKVCompletionLedger ledger(/*max_outstanding_per_request=*/1, schema);
    FlatKVCompletionState retired_state;
    const FlatKVCompletionInput stale_generation = Dispatch(ledger, retired_state, 0, 4);
    FlatKVCompletionState current_state;
    const FlatKVCompletionInput current = Dispatch(ledger, current_state, 0, 4);
    ASSERT_NE(stale_generation.table_generation, current.table_generation);

    EXPECT_EQ(ledger.Prepare(current_state, "r", Complete(stale_generation, 4), {}).disposition,
              FlatKVCompletionDisposition::kStale);
    EXPECT_EQ(current_state.Snapshot().outstanding_count, 1u);
    const auto prepared = ledger.Prepare(current_state, "r", Complete(current, 4), {});
    CommitPrepared(ledger, current_state, prepared);
    EXPECT_EQ(ledger.Prepare(current_state, "r", Complete(current, 4), {}).disposition,
              FlatKVCompletionDisposition::kStale);
}

TEST(FlatKVCompletionLedgerTest, ShortAcceptanceQuarantinesSuccessorUntilItsFence) {
    const std::vector<KvCacheGroupSchema> schema = CompletionSchema({1, 4});
    FlatKVCompletionLedger ledger(/*max_outstanding_per_request=*/2, schema);
    FlatKVCompletionState state;
    const FlatKVCompletionInput first = Dispatch(ledger, state, 0, 4);
    const FlatKVCompletionInput successor = Dispatch(ledger, state, 4, 8);

    const auto short_prepared = ledger.Prepare(state, "r", Complete(first, 2), {10, 11});
    EXPECT_FALSE(state.Snapshot().has_canceled_outstanding);
    EXPECT_EQ(short_prepared.disposition, FlatKVCompletionDisposition::kApplied);
    EXPECT_EQ(CommitPrepared(ledger, state, short_prepared), 1u);
    const FlatKVCompletionRequestSnapshot quarantined = state.Snapshot();
    EXPECT_TRUE(quarantined.has_canceled_outstanding);
    ASSERT_TRUE(quarantined.last_dispatch_raw_end.has_value());
    EXPECT_EQ(*quarantined.last_dispatch_raw_end, successor.dispatch_raw_end);
    EXPECT_THROW(Dispatch(ledger, state, 2, 6), std::logic_error);

    const auto canceled = ledger.Prepare(state, "r", Complete(successor, 8), {20, 21, 22, 23});
    EXPECT_EQ(canceled.disposition, FlatKVCompletionDisposition::kCanceled);
    EXPECT_FALSE(canceled.ready.has_value());
    EXPECT_EQ(CommitPrepared(ledger, state, canceled), 0u);
    EXPECT_EQ(Dispatch(ledger, state, 2, 6).dispatch_seq, successor.dispatch_seq + 1);
}

TEST(FlatKVCompletionLedgerTest, TerminalCancellationRetainsEveryExecutionFence) {
    const std::vector<KvCacheGroupSchema> schema = CompletionSchema({1});
    FlatKVCompletionLedger ledger(/*max_outstanding_per_request=*/2, schema);
    FlatKVCompletionState state;
    const FlatKVCompletionInput first = Dispatch(ledger, state, 0, 4);
    const FlatKVCompletionInput second = Dispatch(ledger, state, 4, 8);

    EXPECT_EQ(state.CancelOutstanding(), 2u);
    EXPECT_EQ(state.CancelOutstanding(), 0u);
    EXPECT_EQ(state.Snapshot().outstanding_count, 2u);
    const auto first_canceled = ledger.Prepare(state, "r", Complete(first, 4), {10});
    EXPECT_EQ(first_canceled.disposition, FlatKVCompletionDisposition::kCanceled);
    EXPECT_EQ(CommitPrepared(ledger, state, first_canceled), 1u);
    const auto second_canceled = ledger.Prepare(state, "r", Complete(second, 8), {11});
    EXPECT_EQ(second_canceled.disposition, FlatKVCompletionDisposition::kCanceled);
    EXPECT_EQ(CommitPrepared(ledger, state, second_canceled), 0u);
}

TEST(FlatKVCompletionLedgerTest, RejectsInvalidCompletionPayloadBeforeCommit) {
    struct Case {
        const char* name;
        std::int32_t accepted;
        std::vector<std::int32_t> tokens;
        bool canceled;
    };
    const Case cases[] = {
        {"active before dispatch", 7, {}, false},       {"active after dispatch", 13, {}, false},
        {"canceled before dispatch", 7, {}, true},      {"canceled after dispatch", 13, {}, true},
        {"mid-prefill result tokens", 12, {99}, false},
    };
    const std::vector<KvCacheGroupSchema> schema = CompletionSchema({1});
    for (const Case& test : cases) {
        SCOPED_TRACE(test.name);
        FlatKVCompletionLedger ledger(/*max_outstanding_per_request=*/1, schema);
        FlatKVCompletionState state;
        const FlatKVCompletionInput input = Dispatch(ledger, state, 8, 12, /*apply_fsm_result=*/false);
        if (test.canceled) {
            ASSERT_EQ(state.CancelOutstanding(), 1u);
        }
        EXPECT_THROW(ledger.Prepare(state, "r", Complete(input, test.accepted), test.tokens), std::invalid_argument);
        EXPECT_EQ(state.Snapshot().outstanding_count, 1u);
    }
}

}  // namespace
}  // namespace tokenspeed::test
