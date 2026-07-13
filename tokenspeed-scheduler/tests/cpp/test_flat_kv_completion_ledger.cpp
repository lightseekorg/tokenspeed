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

#include <bit>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "scheduler/flat_kv_completion_ledger.h"

namespace tokenspeed::test {
namespace {

void NoopCommit(void*) noexcept {}

template <typename Observer>
class ObserverCallbacks {
public:
    explicit ObserverCallbacks(Observer observer) : observer_{std::move(observer)} {}

    operator FlatKVCompletionCallbacks() {
        return FlatKVCompletionCallbacks{
            .context = this,
            .prepare = [](void* opaque, const FlatKVReadyCompletion& ready) {
                static_cast<ObserverCallbacks*>(opaque)->observer_(ready);
            },
            .commit = &NoopCommit,
        };
    }

private:
    Observer observer_;
};

template <typename Observer>
ObserverCallbacks<Observer> PrepareWith(Observer observer) {
    return ObserverCallbacks<Observer>{std::move(observer)};
}

void ThrowBadAlloc(void*, const FlatKVReadyCompletion&) { throw std::bad_alloc{}; }

FlatKVCompletionCallbacks FailingPrepare() {
    return FlatKVCompletionCallbacks{.prepare = &ThrowBadAlloc, .commit = &NoopCommit};
}

std::vector<FlatKVCompletionGroupSchema> Schemas() {
    return {
        FlatKVCompletionGroupSchema{
            .group_id = "history",
            .required_domain_mask = 0b0011,
            .entry_stride_tokens = 1,
        },
        FlatKVCompletionGroupSchema{
            .group_id = "state",
            .required_domain_mask = 0b0101,
            .entry_stride_tokens = 1,
        },
    };
}

forward::FlatKVCompletion Complete(const FlatKVCompletionInput& input, std::int32_t accepted_raw_end) {
    std::vector<forward::FlatKVGroupCompletion> groups;
    groups.reserve(Schemas().size());
    for (const FlatKVCompletionGroupSchema& group : Schemas()) {
        groups.push_back(forward::FlatKVGroupCompletion{
            .group_id = group.group_id,
            .completed_domain_mask = group.required_domain_mask,
            .domain_valid_ends =
                std::vector<std::int32_t>(std::popcount(group.required_domain_mask), input.dispatch_raw_end),
        });
    }
    return forward::FlatKVCompletion{
        .request_id = input.request_id,
        .table_generation = input.table_generation,
        .dispatch_seq = input.dispatch_seq,
        .accepted_raw_end = accepted_raw_end,
        .protected_raw_end = input.protected_raw_end,
        .groups = std::move(groups),
    };
}

forward::FlatKVCompletion CompleteOneGroup(const FlatKVCompletionInput& input, std::int32_t accepted_raw_end,
                                           std::uint32_t completed_domain_mask,
                                           std::vector<std::int32_t> domain_valid_ends) {
    return forward::FlatKVCompletion{
        .request_id = input.request_id,
        .table_generation = input.table_generation,
        .dispatch_seq = input.dispatch_seq,
        .accepted_raw_end = accepted_raw_end,
        .protected_raw_end = input.protected_raw_end,
        .groups = {forward::FlatKVGroupCompletion{
            .group_id = "compressed",
            .completed_domain_mask = completed_domain_mask,
            .domain_valid_ends = std::move(domain_valid_ends),
        }},
    };
}

FlatKVCompletionInput Dispatch(FlatKVCompletionLedger& ledger, std::int32_t raw_start, std::int32_t raw_end,
                               bool legacy_result_expected = true) {
    return ledger.RecordDispatch(FlatKVDispatchSpec{
        .request_id = "r",
        .dispatch_raw_start = raw_start,
        .dispatch_raw_end = raw_end,
        .protected_raw_end = raw_end + 4,
        .legacy_result_expected = legacy_result_expected,
    });
}

TEST(FlatKVCompletionLedgerTest, UnpublishedPlanBatchRollsBackEveryDispatch) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());

    try {
        FlatKVDispatchBatch batch(ledger, /*max_dispatches=*/2);
        const FlatKVCompletionInput first = batch.Record(FlatKVDispatchSpec{
            .request_id = "r",
            .dispatch_raw_start = 8,
            .dispatch_raw_end = 12,
            .protected_raw_end = 16,
            .legacy_result_expected = true,
        });
        const FlatKVCompletionInput second = batch.Record(FlatKVDispatchSpec{
            .request_id = "r",
            .dispatch_raw_start = 12,
            .dispatch_raw_end = 16,
            .protected_raw_end = 20,
            .legacy_result_expected = true,
        });
        EXPECT_EQ(first.dispatch_seq, 0u);
        EXPECT_EQ(second.dispatch_seq, 1u);
        EXPECT_EQ(ledger.OutstandingCount("r"), 2u);
        throw std::bad_alloc{};  // stand-in for a later plan/export allocation
    } catch (const std::bad_alloc&) {
    }

    EXPECT_EQ(ledger.OutstandingCount("r"), 0u);
    EXPECT_FALSE(ledger.HasOutstanding("r"));
    const FlatKVCompletionInput recovered = Dispatch(ledger, /*raw_start=*/20, /*raw_end=*/24);
    EXPECT_EQ(recovered.dispatch_seq, 0u);
    EXPECT_EQ(recovered.dispatch_raw_start, 20);
}

TEST(FlatKVCompletionLedgerTest, PublishedPlanBatchKeepsDispatches) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    FlatKVCompletionInput input;
    {
        FlatKVDispatchBatch batch(ledger, /*max_dispatches=*/1);
        input = batch.Record(FlatKVDispatchSpec{
            .request_id = "r",
            .dispatch_raw_start = 4,
            .dispatch_raw_end = 8,
            .protected_raw_end = 12,
            .legacy_result_expected = true,
        });
        batch.Commit();
    }
    EXPECT_EQ(ledger.OutstandingCount("r"), 1u);
    EXPECT_EQ(input.dispatch_seq, 0u);
}

TEST(FlatKVCompletionLedgerTest, BatchCapacityFailureDoesNotRecordDispatch) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    FlatKVDispatchBatch batch(ledger, /*max_dispatches=*/0);
    EXPECT_THROW(batch.Record(FlatKVDispatchSpec{
                     .request_id = "r",
                     .dispatch_raw_start = 0,
                     .dispatch_raw_end = 4,
                     .protected_raw_end = 8,
                     .legacy_result_expected = true,
                 }),
                 std::length_error);
    EXPECT_EQ(ledger.OutstandingCount("r"), 0u);
}

TEST(FlatKVCompletionLedgerTest, AppliesInOrderAndClampsRequiredDomainMinimumToAcceptedEnd) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput input = Dispatch(ledger, /*raw_start=*/4, /*raw_end=*/8);
    std::vector<FlatKVReadyCompletion> applied;

    auto completion = Complete(input, /*accepted_raw_end=*/6);
    completion.groups.at(1).domain_valid_ends = {7, 5};
    const FlatKVCompletionSubmitResult result =
        ledger.Submit(std::move(completion), /*tokens=*/{10, 11},
                      PrepareWith([&](const FlatKVReadyCompletion& ready) { applied.push_back(ready); }));

    EXPECT_EQ(result.disposition, FlatKVCompletionDisposition::kApplied);
    EXPECT_EQ(result.applied_count, 1u);
    ASSERT_EQ(applied.size(), 1u);
    EXPECT_EQ(applied.front().tokens, (std::vector<std::int32_t>{10, 11}));
    EXPECT_EQ(applied.front().ready_raw_ends, (std::vector<std::int32_t>{6, 5}));
    EXPECT_FALSE(ledger.HasOutstanding("r"));
}

TEST(FlatKVCompletionLedgerTest, RejectsProducerDomainSupersetThatDriftsFromSchema) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput input = Dispatch(ledger, /*raw_start=*/4, /*raw_end=*/8);

    auto completion = Complete(input, /*accepted_raw_end=*/8);
    completion.groups.at(0).completed_domain_mask = 0b0111;
    completion.groups.at(0).domain_valid_ends = {8, 6, 4};
    EXPECT_THROW(ledger.Submit(std::move(completion), /*tokens=*/{10, 11},
                               PrepareWith([](const FlatKVReadyCompletion&) {})),
                 std::invalid_argument);
    EXPECT_TRUE(ledger.HasOutstanding("r"));
}

TEST(FlatKVCompletionLedgerTest, AcceptsCompressedEndBeforeDispatchAndDraftEndBeyondDispatchWithinProtection) {
    FlatKVCompletionLedger ledger(
        /*max_buffered_results=*/2,
        {FlatKVCompletionGroupSchema{"compressed", /*required_domain_mask=*/0b0011, /*entry_stride_tokens=*/4}});
    const FlatKVCompletionInput input = ledger.RecordDispatch(FlatKVDispatchSpec{
        .request_id = "r",
        .dispatch_raw_start = 8,
        .dispatch_raw_end = 12,
        .protected_raw_end = 16,
        .legacy_result_expected = true,
    });
    std::vector<FlatKVReadyCompletion> applied;

    const auto result =
        ledger.Submit(CompleteOneGroup(input, /*accepted_raw_end=*/12, 0b0011, {4, 16}), {},
                      PrepareWith([&](const FlatKVReadyCompletion& ready) { applied.push_back(ready); }));

    EXPECT_EQ(result.disposition, FlatKVCompletionDisposition::kApplied);
    ASSERT_EQ(applied.size(), 1u);
    EXPECT_EQ(applied.front().ready_raw_ends, (std::vector<std::int32_t>{8}));
}

TEST(FlatKVCompletionLedgerTest, MergesProducerDomainsAcrossOrderedCompletionsBeforeTakingRequiredMinimum) {
    FlatKVCompletionLedger ledger(
        /*max_buffered_results=*/2,
        {FlatKVCompletionGroupSchema{"compressed", /*required_domain_mask=*/0b0011, /*entry_stride_tokens=*/1}});
    const FlatKVCompletionInput first = ledger.RecordDispatch(FlatKVDispatchSpec{
        .request_id = "r",
        .dispatch_raw_start = 0,
        .dispatch_raw_end = 8,
        .protected_raw_end = 16,
        .legacy_result_expected = true,
    });
    const FlatKVCompletionInput second = ledger.RecordDispatch(FlatKVDispatchSpec{
        .request_id = "r",
        .dispatch_raw_start = 8,
        .dispatch_raw_end = 12,
        .protected_raw_end = 16,
        .legacy_result_expected = true,
    });
    std::vector<std::int32_t> ready_ends;
    auto collect = [&](const FlatKVReadyCompletion& ready) { ready_ends.push_back(ready.ready_raw_ends.front()); };

    ledger.Submit(CompleteOneGroup(first, /*accepted_raw_end=*/8, 0b0011, {8, 4}), {}, PrepareWith(collect));
    ledger.Submit(CompleteOneGroup(second, /*accepted_raw_end=*/12, 0b0011, {6, 12}), {}, PrepareWith(collect));

    EXPECT_EQ(ready_ends, (std::vector<std::int32_t>{4, 8}));
}

TEST(FlatKVCompletionLedgerTest, ShortAcceptanceClampsStrideProgressAndKeepsReadyBoundaryMonotonic) {
    FlatKVCompletionLedger ledger(
        /*max_buffered_results=*/2,
        {FlatKVCompletionGroupSchema{"compressed", /*required_domain_mask=*/0b0001, /*entry_stride_tokens=*/4}});
    const FlatKVCompletionInput first = ledger.RecordDispatch(FlatKVDispatchSpec{
        .request_id = "r",
        .dispatch_raw_start = 0,
        .dispatch_raw_end = 12,
        .protected_raw_end = 16,
        .legacy_result_expected = true,
    });
    std::vector<std::int32_t> ready_ends;
    auto collect = [&](const FlatKVReadyCompletion& ready) { ready_ends.push_back(ready.ready_raw_ends.front()); };
    ledger.Submit(CompleteOneGroup(first, /*accepted_raw_end=*/10, 0b0001, {16}), {}, PrepareWith(collect));

    const FlatKVCompletionInput recovered = ledger.RecordDispatch(FlatKVDispatchSpec{
        .request_id = "r",
        .dispatch_raw_start = 10,
        .dispatch_raw_end = 14,
        .protected_raw_end = 18,
        .legacy_result_expected = true,
    });
    ledger.Submit(CompleteOneGroup(recovered, /*accepted_raw_end=*/14, 0b0001, {8}), {}, PrepareWith(collect));

    EXPECT_EQ(ready_ends, (std::vector<std::int32_t>{8, 8}));
}

TEST(FlatKVCompletionLedgerTest, BuffersOutOfOrderAndAppliesContiguousSequence) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput first = Dispatch(ledger, 0, 4);
    const FlatKVCompletionInput second = Dispatch(ledger, 4, 8);
    std::vector<std::uint64_t> order;
    auto apply = [&](const FlatKVReadyCompletion& ready) { order.push_back(ready.input.dispatch_seq); };

    const auto buffered = ledger.Submit(Complete(second, 8), {}, PrepareWith(apply));
    EXPECT_EQ(buffered.disposition, FlatKVCompletionDisposition::kBuffered);
    EXPECT_TRUE(order.empty());
    EXPECT_EQ(ledger.BufferedResultCount("r"), 1u);

    const auto drained = ledger.Submit(Complete(first, 4), {}, PrepareWith(apply));
    EXPECT_EQ(drained.disposition, FlatKVCompletionDisposition::kApplied);
    EXPECT_EQ(drained.applied_count, 2u);
    EXPECT_EQ(order, (std::vector<std::uint64_t>{first.dispatch_seq, second.dispatch_seq}));
    EXPECT_EQ(ledger.BufferedResultCount("r"), 0u);
}

TEST(FlatKVCompletionLedgerTest, RejectsDuplicateWithoutReapplying) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput input = Dispatch(ledger, 0, 4);
    std::size_t applies = 0;
    auto apply = [&](const FlatKVReadyCompletion&) { ++applies; };

    EXPECT_EQ(ledger.Submit(Complete(input, 4), {}, PrepareWith(apply)).disposition,
              FlatKVCompletionDisposition::kApplied);
    EXPECT_THROW(ledger.Submit(Complete(input, 4), {}, PrepareWith(apply)), std::invalid_argument);
    EXPECT_EQ(applies, 1u);
    EXPECT_EQ(ledger.Stats().duplicate_results, 1u);
}

TEST(FlatKVCompletionLedgerTest, IgnoresAndCountsCrossGenerationLateResult) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput stale = Dispatch(ledger, 0, 4);
    EXPECT_EQ(ledger.Invalidate("r"), 1u);
    const FlatKVCompletionInput current = Dispatch(ledger, 0, 4);
    ASSERT_NE(stale.table_generation, current.table_generation);
    std::size_t applies = 0;

    const auto result =
        ledger.Submit(Complete(stale, 4), {}, PrepareWith([&](const FlatKVReadyCompletion&) { ++applies; }));

    EXPECT_EQ(result.disposition, FlatKVCompletionDisposition::kStaleGeneration);
    EXPECT_EQ(applies, 0u);
    EXPECT_EQ(ledger.Stats().stale_generation_results, 1u);
    EXPECT_TRUE(ledger.HasOutstanding("r"));
}

TEST(FlatKVCompletionLedgerTest, ShortAcceptanceInvalidatesEveryDispatchedSuccessor) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput first = Dispatch(ledger, 0, 4);
    const FlatKVCompletionInput second = Dispatch(ledger, 4, 8);
    std::vector<std::uint64_t> order;
    std::vector<std::int32_t> history_ready_ends;
    auto apply = [&](const FlatKVReadyCompletion& ready) {
        order.push_back(ready.input.dispatch_seq);
        history_ready_ends.push_back(ready.ready_raw_ends.front());
    };
    EXPECT_EQ(ledger.Submit(Complete(second, 8), {}, PrepareWith(apply)).disposition,
              FlatKVCompletionDisposition::kBuffered);

    const auto rejected = ledger.Submit(Complete(first, 2), {}, PrepareWith(apply));

    EXPECT_EQ(rejected.applied_count, 1u);
    EXPECT_EQ(rejected.invalidated_dispatches, 1u);
    EXPECT_EQ(order, (std::vector<std::uint64_t>{first.dispatch_seq}));
    EXPECT_FALSE(ledger.HasOutstanding("r"));
    const FlatKVCompletionInput recovered = Dispatch(ledger, 2, 6);
    EXPECT_GT(recovered.dispatch_seq, second.dispatch_seq);
    auto recovered_completion = Complete(recovered, /*accepted_raw_end=*/3);
    recovered_completion.groups.at(0).domain_valid_ends = {3, 3};
    recovered_completion.groups.at(1).domain_valid_ends = {3, 3};
    ledger.Submit(std::move(recovered_completion), {}, PrepareWith(apply));
    EXPECT_EQ(history_ready_ends, (std::vector<std::int32_t>{2, 3}));
}

TEST(FlatKVCompletionLedgerTest, UnfencedCanceledSuccessorBlocksRedispatchUntilItsCompletionRetires) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput first = Dispatch(ledger, 0, 4);
    const FlatKVCompletionInput successor = Dispatch(ledger, 4, 8);
    std::vector<std::uint64_t> applied;
    auto apply = [&](const FlatKVReadyCompletion& ready) { applied.push_back(ready.input.dispatch_seq); };

    const auto rejected = ledger.Submit(Complete(first, 2), {}, PrepareWith(apply));
    EXPECT_EQ(rejected.invalidated_dispatches, 1u);
    EXPECT_EQ(rejected.retired_canceled_dispatches, 0u);
    EXPECT_TRUE(ledger.HasCanceledOutstanding("r"));
    EXPECT_EQ(ledger.OutstandingCount("r"), 1u);
    EXPECT_THROW(Dispatch(ledger, 2, 6), std::logic_error);

    const auto retired = ledger.Submit(Complete(successor, 8), {99}, PrepareWith(apply));
    EXPECT_EQ(retired.disposition, FlatKVCompletionDisposition::kCanceled);
    EXPECT_EQ(retired.retired_canceled_dispatches, 1u);
    EXPECT_EQ(applied, (std::vector<std::uint64_t>{first.dispatch_seq}));
    EXPECT_FALSE(ledger.HasCanceledOutstanding("r"));
    EXPECT_FALSE(ledger.HasOutstanding("r"));
    const FlatKVCompletionInput recovered = Dispatch(ledger, 2, 6);
    EXPECT_EQ(recovered.dispatch_seq, successor.dispatch_seq + 1);
}

TEST(FlatKVCompletionLedgerTest, TerminalCancellationRetainsEveryDispatchUntilItsFenceArrives) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput first = Dispatch(ledger, 0, 4);
    const FlatKVCompletionInput second = Dispatch(ledger, 4, 8);
    std::size_t applies = 0;
    auto apply = [&](const FlatKVReadyCompletion&) { ++applies; };

    EXPECT_EQ(ledger.CancelOutstanding("r"), 2u);
    EXPECT_EQ(ledger.CancelOutstanding("r"), 0u);
    EXPECT_TRUE(ledger.HasCanceledOutstanding("r"));
    EXPECT_EQ(ledger.OutstandingCount("r"), 2u);
    EXPECT_THROW(Dispatch(ledger, 0, 4), std::logic_error);

    const auto buffered = ledger.Submit(Complete(second, 8), {99}, PrepareWith(apply));
    EXPECT_EQ(buffered.disposition, FlatKVCompletionDisposition::kBuffered);
    EXPECT_EQ(ledger.OutstandingCount("r"), 2u);
    EXPECT_EQ(applies, 0u);

    const auto retired = ledger.Submit(Complete(first, 4), {98}, PrepareWith(apply));
    EXPECT_EQ(retired.disposition, FlatKVCompletionDisposition::kCanceled);
    EXPECT_EQ(retired.retired_canceled_dispatches, 2u);
    EXPECT_EQ(applies, 0u);
    EXPECT_FALSE(ledger.HasOutstanding("r"));
    EXPECT_EQ(ledger.Stats().invalidated_dispatches, 2u);
    EXPECT_EQ(ledger.Stats().canceled_results, 2u);
}

TEST(FlatKVCompletionLedgerTest, FailsClosedWhenOutOfOrderBufferWouldExceedBound) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput first = Dispatch(ledger, 0, 4);
    const FlatKVCompletionInput second = Dispatch(ledger, 4, 8);
    const FlatKVCompletionInput third = Dispatch(ledger, 8, 12);
    const FlatKVCompletionInput fourth = Dispatch(ledger, 12, 16);
    auto apply = [](const FlatKVReadyCompletion&) {};
    EXPECT_EQ(ledger.Submit(Complete(second, 8), {}, PrepareWith(apply)).disposition,
              FlatKVCompletionDisposition::kBuffered);
    EXPECT_EQ(ledger.Submit(Complete(third, 12), {}, PrepareWith(apply)).disposition,
              FlatKVCompletionDisposition::kBuffered);

    EXPECT_THROW(ledger.Submit(Complete(fourth, 16), {}, PrepareWith(apply)), std::overflow_error);
    EXPECT_EQ(ledger.BufferedResultCount("r"), 2u);
    EXPECT_EQ(ledger.Stats().buffer_overflows, 1u);
    EXPECT_TRUE(ledger.HasOutstanding("r"));
    (void)first;
}

TEST(FlatKVCompletionLedgerTest, RejectsMissingDomainAndInvalidGroupWatermark) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput input = Dispatch(ledger, 0, 4);
    auto missing_domain = Complete(input, 4);
    missing_domain.groups.at(0).completed_domain_mask = 0b0001;
    missing_domain.groups.at(0).domain_valid_ends = {4};
    EXPECT_THROW(ledger.Submit(std::move(missing_domain), {}, PrepareWith([](const FlatKVReadyCompletion&) {})),
                 std::invalid_argument);

    auto invalid_watermark = Complete(input, 4);
    invalid_watermark.groups.at(1).domain_valid_ends.at(1) = 9;
    EXPECT_THROW(ledger.Submit(std::move(invalid_watermark), {}, PrepareWith([](const FlatKVReadyCompletion&) {})),
                 std::invalid_argument);
    EXPECT_TRUE(ledger.HasOutstanding("r"));
    EXPECT_THROW(ledger.RetireLegacyResult("r"), std::invalid_argument);
}

TEST(FlatKVCompletionLedgerTest, RejectsDomainWatermarkCardinalityMismatch) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput input = Dispatch(ledger, 0, 4);
    auto completion = Complete(input, 4);
    completion.groups.at(0).domain_valid_ends.pop_back();

    EXPECT_THROW(ledger.Submit(std::move(completion), {}, PrepareWith([](const FlatKVReadyCompletion&) {})),
                 std::invalid_argument);
    EXPECT_TRUE(ledger.HasOutstanding("r"));
}

TEST(FlatKVCompletionLedgerTest, RejectsDomainWatermarkOutsideEntryStride) {
    FlatKVCompletionLedger ledger(
        /*max_buffered_results=*/2,
        {FlatKVCompletionGroupSchema{"compressed", /*required_domain_mask=*/1, /*entry_stride_tokens=*/4}});
    const FlatKVCompletionInput input = ledger.RecordDispatch(FlatKVDispatchSpec{
        .request_id = "r",
        .dispatch_raw_start = 0,
        .dispatch_raw_end = 4,
        .protected_raw_end = 8,
        .legacy_result_expected = true,
    });

    EXPECT_THROW(
        ledger.Submit(CompleteOneGroup(input, /*accepted_raw_end=*/4, 1, {3}), {},
                      PrepareWith([](const FlatKVReadyCompletion&) {})),
        std::invalid_argument);
}

TEST(FlatKVCompletionLedgerTest, InvalidFirstDispatchDoesNotBurnGenerationOrSequence) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    FlatKVDispatchSpec invalid{
        .request_id = "r",
        .dispatch_raw_start = 0,
        .dispatch_raw_end = 4,
        .protected_raw_end = 3,
        .legacy_result_expected = true,
    };
    EXPECT_THROW(ledger.RecordDispatch(std::move(invalid)), std::invalid_argument);

    const FlatKVCompletionInput first = Dispatch(ledger, 0, 4);
    EXPECT_NE(first.table_generation, 0u);
    EXPECT_EQ(first.dispatch_seq, 0u);
}

TEST(FlatKVCompletionLedgerTest, ThrowingPrepareKeepsCurrentAndBufferedSuccessorRetryable) {
    FlatKVCompletionLedger ledger(
        /*max_buffered_results=*/2,
        {FlatKVCompletionGroupSchema{"compressed", /*required_domain_mask=*/0b0011, /*entry_stride_tokens=*/1}});
    const FlatKVCompletionInput first = ledger.RecordDispatch(FlatKVDispatchSpec{
        .request_id = "r",
        .dispatch_raw_start = 0,
        .dispatch_raw_end = 8,
        .protected_raw_end = 16,
        .legacy_result_expected = true,
    });
    const FlatKVCompletionInput second = ledger.RecordDispatch(FlatKVDispatchSpec{
        .request_id = "r",
        .dispatch_raw_start = 8,
        .dispatch_raw_end = 12,
        .protected_raw_end = 16,
        .legacy_result_expected = true,
    });
    const forward::FlatKVCompletion first_completion =
        CompleteOneGroup(first, /*accepted_raw_end=*/8, 0b0011, {8, 4});
    ledger.Submit(CompleteOneGroup(second, /*accepted_raw_end=*/12, 0b0011, {6, 12}), {},
                  PrepareWith([](const FlatKVReadyCompletion&) {}));

    EXPECT_THROW(ledger.Submit(first_completion, {}, FailingPrepare()),
                 std::bad_alloc);
    EXPECT_EQ(ledger.OutstandingCount("r"), 2u);
    EXPECT_EQ(ledger.BufferedResultCount("r"), 1u);
    EXPECT_EQ(ledger.Stats().applied_results, 0u);

    std::vector<std::int32_t> ready_ends;
    const auto retried = ledger.Submit(
        first_completion, {},
        PrepareWith([&](const FlatKVReadyCompletion& ready) { ready_ends.push_back(ready.ready_raw_ends.front()); }));
    EXPECT_EQ(retried.applied_count, 2u);
    EXPECT_EQ(ready_ends, (std::vector<std::int32_t>{4, 8}));
    EXPECT_FALSE(ledger.HasOutstanding("r"));
}

TEST(FlatKVCompletionLedgerTest, FailedPrepareRejectsDifferentRetryPayload) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput input = Dispatch(ledger, 0, 4);
    const forward::FlatKVCompletion completion = Complete(input, 4);
    EXPECT_THROW(ledger.Submit(completion, {}, FailingPrepare()),
                 std::bad_alloc);

    auto changed = completion;
    changed.groups.front().domain_valid_ends.front() = 3;
    EXPECT_THROW(ledger.Submit(std::move(changed), {}, PrepareWith([](const FlatKVReadyCompletion&) {})),
                 std::invalid_argument);
    EXPECT_EQ(ledger.OutstandingCount("r"), 1u);
}

struct RecordingCommitContext {
    std::vector<std::uint64_t>* committed{};
    std::uint64_t seq{};
    std::uint64_t fail_seq{std::numeric_limits<std::uint64_t>::max()};
};

void PrepareRecordingCommit(void* opaque, const FlatKVReadyCompletion& ready) {
    auto& context = *static_cast<RecordingCommitContext*>(opaque);
    if (ready.input.dispatch_seq == context.fail_seq) {
        throw std::bad_alloc{};
    }
    context.seq = ready.input.dispatch_seq;
}

void RecordCommit(void* opaque) noexcept {
    auto& context = *static_cast<RecordingCommitContext*>(opaque);
    context.committed->push_back(context.seq);
}

TEST(FlatKVCompletionLedgerTest, AppliedPredecessorRetransmissionRetriesFailedBufferedSuccessor) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput first = Dispatch(ledger, 0, 4);
    const FlatKVCompletionInput second = Dispatch(ledger, 4, 8);
    ledger.Submit(Complete(second, 8), {}, PrepareWith([](const FlatKVReadyCompletion&) {}));

    std::vector<std::uint64_t> committed;
    committed.reserve(2);
    RecordingCommitContext context{.committed = &committed, .fail_seq = second.dispatch_seq};
    const FlatKVCompletionCallbacks callbacks{
        .context = &context,
        .prepare = &PrepareRecordingCommit,
        .commit = &RecordCommit,
    };
    EXPECT_THROW(
        ledger.Submit(Complete(first, 4), {}, callbacks),
        std::bad_alloc);
    EXPECT_EQ(committed, (std::vector<std::uint64_t>{first.dispatch_seq}));
    EXPECT_EQ(ledger.Stats().applied_results, 1u);
    EXPECT_EQ(ledger.OutstandingCount("r"), 1u);

    context.fail_seq = std::numeric_limits<std::uint64_t>::max();
    const auto retried = ledger.Submit(Complete(first, 4), {}, callbacks);
    EXPECT_EQ(retried.disposition, FlatKVCompletionDisposition::kApplied);
    EXPECT_EQ(retried.applied_count, 1u);
    EXPECT_EQ(committed, (std::vector<std::uint64_t>{first.dispatch_seq, second.dispatch_seq}));
    EXPECT_EQ(ledger.Stats().duplicate_results, 0u);
    EXPECT_FALSE(ledger.HasOutstanding("r"));
}

TEST(FlatKVCompletionLedgerTest, NewSubmitDrainsFailedFrontBeforeCheckingFullBuffer) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/1, Schemas());
    const FlatKVCompletionInput first = Dispatch(ledger, 0, 4);
    const FlatKVCompletionInput second = Dispatch(ledger, 4, 8);
    const FlatKVCompletionInput third = Dispatch(ledger, 8, 12);
    ledger.Submit(Complete(second, 8), {}, PrepareWith([](const FlatKVReadyCompletion&) {}));
    EXPECT_THROW(ledger.Submit(Complete(first, 4), {}, FailingPrepare()),
                 std::bad_alloc);

    std::vector<std::uint64_t> committed;
    committed.reserve(3);
    RecordingCommitContext context{.committed = &committed};
    const auto result = ledger.Submit(
        Complete(third, 12), {},
        FlatKVCompletionCallbacks{
            .context = &context,
            .prepare = &PrepareRecordingCommit,
            .commit = &RecordCommit,
        });

    EXPECT_EQ(result.disposition, FlatKVCompletionDisposition::kApplied);
    EXPECT_EQ(result.applied_count, 3u);
    EXPECT_EQ(committed, (std::vector<std::uint64_t>{first.dispatch_seq, second.dispatch_seq, third.dispatch_seq}));
    EXPECT_EQ(ledger.Stats().buffer_overflows, 0u);
    EXPECT_FALSE(ledger.HasOutstanding("r"));
}

TEST(FlatKVCompletionLedgerTest, CancellationRetiresFailedFrontAndContiguousBufferedSuccessors) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    const FlatKVCompletionInput first = Dispatch(ledger, 0, 4);
    const FlatKVCompletionInput second = Dispatch(ledger, 4, 8);
    ledger.Submit(Complete(second, 8), {}, PrepareWith([](const FlatKVReadyCompletion&) {}));
    EXPECT_THROW(ledger.Submit(Complete(first, 4), {}, FailingPrepare()), std::bad_alloc);
    ASSERT_EQ(ledger.OutstandingCount("r"), 2u);
    ASSERT_EQ(ledger.BufferedResultCount("r"), 1u);

    EXPECT_EQ(ledger.CancelOutstanding("r"), 2u);
    EXPECT_FALSE(ledger.HasOutstanding("r"));
    EXPECT_EQ(ledger.BufferedResultCount("r"), 0u);
    EXPECT_EQ(ledger.Stats().applied_results, 0u);
    EXPECT_EQ(ledger.Stats().canceled_results, 2u);
}

void TerminatingCommit(void*) noexcept { std::terminate(); }

TEST(FlatKVCompletionLedgerTest, CommitContractViolationFailsStop) {
    EXPECT_EXIT(
        {
            FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
            const FlatKVCompletionInput input = Dispatch(ledger, 0, 4);
            std::set_terminate([] { std::_Exit(87); });
            (void)ledger.Submit(Complete(input, 4), {},
                                FlatKVCompletionCallbacks{
                                    .prepare = [](void*, const FlatKVReadyCompletion&) {},
                                    .commit = &TerminatingCommit,
                                });
        },
        ::testing::ExitedWithCode(87), "");
}

TEST(FlatKVCompletionLedgerTest, LegacyResultRetiresPrecedingMidChunkDebt) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());
    Dispatch(ledger, 0, 4, /*legacy_result_expected=*/false);
    Dispatch(ledger, 4, 8, /*legacy_result_expected=*/true);

    EXPECT_EQ(ledger.RetireLegacyResult("r"), 2u);
    EXPECT_FALSE(ledger.HasOutstanding("r"));
    EXPECT_EQ(ledger.Stats().legacy_results, 1u);
}

TEST(FlatKVCompletionLedgerTest, FirstMidPrefillDispatchBlocksReclaimBeforeAnyCompletion) {
    FlatKVCompletionLedger ledger(/*max_buffered_results=*/2, Schemas());

    Dispatch(ledger, 0, 4, /*legacy_result_expected=*/false);

    EXPECT_TRUE(ledger.HasOutstanding("r"));
    EXPECT_TRUE(ledger.HasBlockingOutstanding())
        << "dispatch lifetime, not the observed completion format, protects in-flight pages";
}

}  // namespace
}  // namespace tokenspeed::test
