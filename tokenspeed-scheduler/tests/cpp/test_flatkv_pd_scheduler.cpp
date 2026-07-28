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
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/kv_cache_coordinator.h"
#include "integration_test_helper.h"
#include "scheduler/page_hasher.h"

namespace tokenspeed::test {
namespace {

#if TOKENSPEED_FLAT_KVCACHE

PagedCacheGroupConfig MakePdGroup(std::string group_id, PagedCacheGroupFamily family,
                                  PagedCacheTransferPolicy transfer_policy, std::int32_t block_size = 2,
                                  std::int32_t total_pages = 64) {
    PagedCacheGroupConfig group;
    group.group_id = std::move(group_id);
    group.rows_per_page = block_size;
    group.entry_stride_tokens = 1;
    group.total_pages = total_pages;
    group.block_size = block_size;
    group.retention = PagedCacheGroupConfig::Retention::FullHistory;
    group.family = family;
    group.transfer_policy = transfer_policy;
    return group;
}

SchedulerConfig MakeFlatPdConfig(Role role, std::int32_t total_pages = 64, std::int32_t overlap_schedule_depth = 0) {
    SchedulerConfig config;
    config.block_size = 2;
    config.device_allocator.total_pages = total_pages;
    config.host_allocator.total_pages = 1;
    config.max_scheduled_tokens = 64;
    config.max_batch_size = 8;
    config.decode_input_tokens = 1;
    config.overlap_schedule_depth = overlap_schedule_depth;
    config.disable_l2_cache = true;
    config.disable_prefix_cache = true;
    config.role = role;
    config.enable_flatkv_pd = true;
    config.paged_cache_groups = {
        MakePdGroup("history", PagedCacheGroupFamily::History, PagedCacheTransferPolicy::FullSuffix, 2, total_pages),
        MakePdGroup("state_0", PagedCacheGroupFamily::State, PagedCacheTransferPolicy::LatestSnapshot, 2, total_pages),
        MakePdGroup("state_1", PagedCacheGroupFamily::State, PagedCacheTransferPolicy::LatestSnapshot, 2, total_pages),
        MakePdGroup("state_2", PagedCacheGroupFamily::State, PagedCacheTransferPolicy::LatestSnapshot, 2, total_pages),
    };
    return config;
}

RequestSpec MakePdRequest(std::string request_id, std::int32_t prompt_tokens) {
    return RequestSpec{
        .request_id = std::move(request_id),
        .tokens = MakeTokens(prompt_tokens),
    };
}

const FlatForwardOperation* FindForward(const ExecutionPlan& plan) {
    for (const Operation& operation : plan.Operations()) {
        if (const auto* forward = std::get_if<FlatForwardOperation>(&operation)) {
            return forward;
        }
    }
    return nullptr;
}

template <typename Event>
void SendPdEvent(Scheduler& scheduler, Event event) {
    ExecutionEvent execution_event;
    execution_event.With(PDEvent{std::move(event)});
    scheduler.Advance(execution_event);
}

void SendForwardAbort(Scheduler& scheduler, const std::string& request_id) {
    ExecutionEvent execution_event;
    execution_event.With(ForwardEvent{forward::Abort{.request_id = request_id}});
    scheduler.Advance(execution_event);
}

void SendForwardResultAndFinish(Scheduler& scheduler, const std::string& request_id, std::vector<std::int32_t> tokens) {
    ExecutionEvent execution_event;
    execution_event.With(ForwardEvent{forward::ExtendResult{
        .request_id = request_id,
        .tokens = std::move(tokens),
    }});
    execution_event.With(ForwardEvent{forward::Finish{.request_id = request_id}});
    scheduler.Advance(execution_event);
}

void SendForwardResult(Scheduler& scheduler, const std::string& request_id, std::vector<std::int32_t> tokens) {
    ExecutionEvent execution_event;
    execution_event.With(ForwardEvent{forward::ExtendResult{
        .request_id = request_id,
        .tokens = std::move(tokens),
    }});
    scheduler.Advance(execution_event);
}

std::size_t CountRealPages(const std::map<std::string, std::vector<std::vector<std::int32_t>>>& tables) {
    std::size_t count = 0;
    for (const auto& [_, rows] : tables) {
        for (const auto& row : rows) {
            count += static_cast<std::size_t>(
                std::count_if(row.begin(), row.end(), [](std::int32_t page) { return page > 0; }));
        }
    }
    return count;
}

void ExpectSparseDestination(std::int32_t prompt_tokens, std::size_t expected_history_pages,
                             std::size_t expected_state_pages, std::size_t expected_total_pages) {
    Scheduler scheduler{MakeFlatPdConfig(Role::kD)};
    const std::int32_t free_at_start = scheduler.FlatPoolFreeBlocks();
    scheduler.SubmitRequests({MakePdRequest("decode-request", prompt_tokens)});

    EXPECT_EQ(scheduler.WaitingSize(), 1u);
    const ExecutionPlan before_bootstrap = scheduler.NextExecutionPlan();
    ASSERT_NE(FindForward(before_bootstrap), nullptr);
    EXPECT_TRUE(FindForward(before_bootstrap)->empty());
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start);

    SendPdEvent(scheduler, pd::BootstrappedEvent{"decode-request"});
    const ExecutionPlan destination = scheduler.NextExecutionPlan();
    const FlatForwardOperation* forward = FindForward(destination);
    ASSERT_NE(forward, nullptr);
    ASSERT_EQ(forward->request_ids, (std::vector<std::string>{"decode-request"}));

    const auto& history = forward->flat_block_tables.at("history").at(0);
    EXPECT_EQ(history.size(), expected_history_pages);
    EXPECT_TRUE(std::all_of(history.begin(), history.end(), [](std::int32_t page) { return page > 0; }));

    std::set<std::int32_t> real_page_ids(history.begin(), history.end());
    for (const char* group_id : {"state_0", "state_1", "state_2"}) {
        const auto& state = forward->flat_block_tables.at(group_id).at(0);
        ASSERT_EQ(state.size(), expected_history_pages);
        ASSERT_EQ(static_cast<std::size_t>(
                      std::count_if(state.begin(), state.end(), [](std::int32_t page) { return page > 0; })),
                  expected_state_pages);
        const std::size_t first_real = state.size() - expected_state_pages;
        for (std::size_t i = 0; i < first_real; ++i) {
            EXPECT_EQ(state[i], 0) << group_id << " slot " << i << " must remain a logical null hole";
        }
        for (std::size_t i = first_real; i < state.size(); ++i) {
            EXPECT_GT(state[i], 0);
            EXPECT_TRUE(real_page_ids.insert(state[i]).second) << "physical page reused across cache groups";
        }
    }

    EXPECT_EQ(CountRealPages(forward->flat_block_tables), expected_total_pages);
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start - static_cast<std::int32_t>(expected_total_pages));
    EXPECT_TRUE(scheduler.FlatPdTransferPinned("decode-request"));

    SendForwardAbort(scheduler, "decode-request");
    EXPECT_FALSE(scheduler.FlatPdTransferPinned("decode-request"));
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start);
    scheduler.NextExecutionPlan();
}

TEST(FlatKvPdDestination, AlignedPromptUsesFinalAndReservedStatePages) {
    // N=8, page=2: history keeps 5 pages (prompt + first decode input);
    // each state group keeps slots 3 and 4, with slots 0..2 as null holes.
    ExpectSparseDestination(
        /*prompt_tokens=*/8,
        /*expected_history_pages=*/5,
        /*expected_state_pages=*/2,
        /*expected_total_pages=*/11);
}

TEST(FlatKvPdDestination, UnalignedPromptSharesStatePageWithDecodeInput) {
    // N=7, page=2: prompt-final state and first decode input share slot 3.
    ExpectSparseDestination(
        /*prompt_tokens=*/7,
        /*expected_history_pages=*/4,
        /*expected_state_pages=*/1,
        /*expected_total_pages=*/7);
}

TEST(FlatKvPdCoordinator, DestinationAcquireIsAtomicAcrossGroups) {
    const std::vector<KvCacheSpec> specs{
        {AttnKind::kFull, 2, 0},
        {AttnKind::kMambaState, 2, 0},
        {AttnKind::kMambaState, 2, 0},
        {AttnKind::kMambaState, 2, 0},
    };

    BlockPool short_pool(/*total_num_blocks=*/11);  // 10 usable < 11.
    KvCacheCoordinator short_coordinator = MakeCoordinator(specs, short_pool);
    std::vector<BlockTable> short_tables(static_cast<std::size_t>(short_coordinator.NumGroups()));
    EXPECT_EQ(short_coordinator.BlocksNeededForDecodeDestination(
                  /*prompt_tokens=*/8,
                  /*remaining_prompt_tokens=*/8,
                  /*reserve_tokens=*/1),
              11);
    EXPECT_FALSE(short_coordinator.AcquireDecodeDestination(short_tables, 8, 8, 1));
    EXPECT_EQ(short_pool.NumFreeBlocks(), 10);
    for (const BlockTable& table : short_tables) {
        EXPECT_EQ(table.NumBlocks(), 0);
        EXPECT_EQ(table.TailAvailableTokens(), 0);
    }

    BlockPool exact_pool(/*total_num_blocks=*/12);  // 11 usable.
    KvCacheCoordinator exact_coordinator = MakeCoordinator(specs, exact_pool);
    std::vector<BlockTable> exact_tables(static_cast<std::size_t>(exact_coordinator.NumGroups()));
    ASSERT_TRUE(exact_coordinator.AcquireDecodeDestination(exact_tables, 8, 8, 1));
    EXPECT_EQ(exact_pool.NumFreeBlocks(), 0);
    EXPECT_EQ(BlockTablePageIds(exact_tables[0]).size(), 5u);
    for (std::size_t group = 1; group < exact_tables.size(); ++group) {
        EXPECT_EQ(BlockTablePageIds(exact_tables[group]).size(), 5u);
        EXPECT_EQ(BlockTablePageIds(exact_tables[group])[0], 0);
        EXPECT_EQ(BlockTablePageIds(exact_tables[group])[1], 0);
        EXPECT_EQ(BlockTablePageIds(exact_tables[group])[2], 0);
    }
    exact_coordinator.Free(exact_tables);
    EXPECT_EQ(exact_pool.NumFreeBlocks(), 11);
}

TEST(FlatKvPdCoordinator, DecodePrefixMatchUsesDenseGroupLcmOnly) {
    BlockPool pool(/*total_num_blocks=*/32);
    const std::vector<KvCacheSpec> specs{
        {AttnKind::kFull, 2, 0},
        {AttnKind::kMambaState, 8, 0},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    ASSERT_EQ(coordinator.BaseBlockSize(), 2);
    ASSERT_EQ(coordinator.LcmBlockSize(), 8);
    ASSERT_EQ(coordinator.DecodeMatchLcmBlockSize(), 2);

    const std::vector<std::vector<std::int32_t>> pages{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
    std::vector<std::span<const std::int32_t>> spans;
    spans.reserve(pages.size());
    for (const auto& page : pages) {
        spans.emplace_back(page.data(), page.size());
    }
    const std::vector<std::string> hashes = ComputePagedHashes(spans, "");
    for (const std::string& hash : hashes) {
        BlockRef block = pool.AcquireBlock();
        ASSERT_TRUE(block);
        pool.CacheFullBlock(block, MakeKeyWithGroupId(hash, /*group_id=*/0));
        block.reset();
    }

    EXPECT_EQ(coordinator.MatchPrefix(hashes).device.num_common_tokens, 0);
    auto probe = coordinator.ProbeDecodeDestinationPrefix(hashes);
    const CoordinatorMatch decode = coordinator.AcquirePrefix(hashes, std::move(probe)).device;
    EXPECT_EQ(decode.num_common_tokens, 10);
    ASSERT_EQ(decode.per_group.size(), 2u);
    EXPECT_EQ(decode.per_group[0].blocks.size(), 5u);
    ASSERT_EQ(decode.per_group[1].blocks.size(), 1u);
    EXPECT_FALSE(decode.per_group[1].blocks[0]);
}

TEST(FlatKvPdLifecycle, RemoteDonePreservesDestinationAndBootstrapToken) {
    constexpr std::int32_t kBootstrapToken = 9001;
    Scheduler scheduler{MakeFlatPdConfig(Role::kD)};
    const std::int32_t free_at_start = scheduler.FlatPoolFreeBlocks();
    scheduler.SubmitRequests({MakePdRequest("r", /*prompt_tokens=*/8)});
    SendPdEvent(scheduler, pd::BootstrappedEvent{"r"});
    const ExecutionPlan destination_plan = scheduler.NextExecutionPlan();
    const FlatForwardOperation* destination = FindForward(destination_plan);
    ASSERT_NE(destination, nullptr);
    const auto destination_tables = destination->flat_block_tables;
    const std::int32_t free_with_destination = scheduler.FlatPoolFreeBlocks();

    EXPECT_THROW(SendPdEvent(scheduler, pd::RemotePrefillDoneEvent{"r", -1}), std::invalid_argument);
    EXPECT_TRUE(scheduler.FlatPdTransferPinned("r"));
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_with_destination);

    SendPdEvent(scheduler, pd::RemotePrefillDoneEvent{"r", kBootstrapToken});
    EXPECT_FALSE(scheduler.FlatPdTransferPinned("r"));
    EXPECT_EQ(scheduler.GetRequestTokenSize("r"), 9);

    const ExecutionPlan decode_plan = scheduler.NextExecutionPlan();
    const FlatForwardOperation* decode = FindForward(decode_plan);
    ASSERT_NE(decode, nullptr);
    ASSERT_EQ(decode->request_ids, (std::vector<std::string>{"r"}));
    EXPECT_EQ(decode->flat_block_tables, destination_tables);
    ASSERT_EQ(decode->decode_input_ids.size(), 1u);
    EXPECT_EQ(decode->decode_input_ids[0], kBootstrapToken);
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_with_destination);

    SendPdEvent(scheduler, pd::SucceededEvent{"r"});
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start);
    scheduler.NextExecutionPlan();
    EXPECT_EQ(scheduler.GetRequestTokenSize("r"), -1);
}

TEST(FlatKvPdOverlap, DecodeProtectsSpeculativeStepUntilFinish) {
    constexpr std::int32_t kBootstrapToken = 9001;
    Scheduler scheduler{MakeFlatPdConfig(Role::kD, /*total_pages=*/64, /*overlap_schedule_depth=*/1)};
    const std::int32_t free_at_start = scheduler.FlatPoolFreeBlocks();
    scheduler.SubmitRequests({MakePdRequest("r", /*prompt_tokens=*/9)});
    SendPdEvent(scheduler, pd::BootstrappedEvent{"r"});

    const ExecutionPlan destination_plan = scheduler.NextExecutionPlan();
    const FlatForwardOperation* destination = FindForward(destination_plan);
    ASSERT_NE(destination, nullptr);
    ASSERT_EQ(destination->request_ids, (std::vector<std::string>{"r"}));
    SendPdEvent(scheduler, pd::RemotePrefillDoneEvent{"r", kBootstrapToken});

    // Dispatch the first local decode, then schedule the speculative next
    // decode before committing the first result, matching event_loop_overlap.
    const ExecutionPlan first_decode_plan = scheduler.NextExecutionPlan();
    const FlatForwardOperation* first_decode = FindForward(first_decode_plan);
    ASSERT_NE(first_decode, nullptr);
    ASSERT_EQ(first_decode->request_ids, (std::vector<std::string>{"r"}));
    ASSERT_EQ(first_decode->decode_input_ids, (std::vector<std::int32_t>{kBootstrapToken}));
    const auto first_history = first_decode->flat_block_tables.at("history").at(0);

    const ExecutionPlan speculative_plan = scheduler.NextExecutionPlan();
    const FlatForwardOperation* speculative = FindForward(speculative_plan);
    ASSERT_NE(speculative, nullptr);
    ASSERT_EQ(speculative->request_ids, (std::vector<std::string>{"r"}));
    const auto speculative_history = speculative->flat_block_tables.at("history").at(0);
    ASSERT_GT(speculative_history.size(), first_history.size());
    EXPECT_TRUE(std::equal(first_history.begin(), first_history.end(), speculative_history.begin()));
    EXPECT_LT(scheduler.FlatPoolFreeBlocks(), free_at_start);

    // A normal decode finish retires both the committed result and the
    // speculative result debt. All FlatKV pages become available again.
    SendForwardResultAndFinish(scheduler, "r", {9002});
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start);
    scheduler.NextExecutionPlan();
    EXPECT_EQ(scheduler.GetRequestTokenSize("r"), -1);

    // The already-dispatched speculative result can arrive after reaping. It
    // must be ignored and must not perturb the fully reusable pool.
    EXPECT_NO_THROW(SendForwardResult(scheduler, "r", {9003}));
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start);

    scheduler.SubmitRequests({MakePdRequest("replacement", /*prompt_tokens=*/9)});
    SendPdEvent(scheduler, pd::BootstrappedEvent{"replacement"});
    const ExecutionPlan replacement_plan = scheduler.NextExecutionPlan();
    const FlatForwardOperation* replacement = FindForward(replacement_plan);
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(replacement->request_ids, (std::vector<std::string>{"replacement"}));
}

TEST(FlatKvPdOverlap, AbortRetiresSpeculativeResultDebtAndPages) {
    Scheduler scheduler{MakeFlatPdConfig(Role::kD, /*total_pages=*/64, /*overlap_schedule_depth=*/1)};
    const std::int32_t free_at_start = scheduler.FlatPoolFreeBlocks();
    scheduler.SubmitRequests({MakePdRequest("aborted", /*prompt_tokens=*/9)});
    SendPdEvent(scheduler, pd::BootstrappedEvent{"aborted"});
    scheduler.NextExecutionPlan();
    SendPdEvent(scheduler, pd::RemotePrefillDoneEvent{"aborted", /*bootstrap_token=*/9001});
    ASSERT_NE(FindForward(scheduler.NextExecutionPlan()), nullptr);
    ASSERT_NE(FindForward(scheduler.NextExecutionPlan()), nullptr);

    EXPECT_NO_THROW(SendForwardAbort(scheduler, "aborted"));
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start);
    scheduler.NextExecutionPlan();
    EXPECT_EQ(scheduler.GetRequestTokenSize("aborted"), -1);
    EXPECT_NO_THROW(SendForwardResult(scheduler, "aborted", {9002}));
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start);
}

TEST(FlatKvPdLifecycle, PrefillHandoffRemainsPinnedUntilTerminal) {
    Scheduler scheduler{MakeFlatPdConfig(Role::kP)};
    const std::int32_t free_at_start = scheduler.FlatPoolFreeBlocks();
    scheduler.SubmitRequests({MakePdRequest("p", /*prompt_tokens=*/8)});
    SendPdEvent(scheduler, pd::BootstrappedEvent{"p"});

    const ExecutionPlan prefill_plan = scheduler.NextExecutionPlan();
    const FlatForwardOperation* prefill = FindForward(prefill_plan);
    ASSERT_NE(prefill, nullptr);
    ASSERT_EQ(prefill->request_ids, (std::vector<std::string>{"p"}));

    const ExecutionPlan handoff_plan = scheduler.NextExecutionPlan();
    const FlatForwardOperation* handoff = FindForward(handoff_plan);
    ASSERT_NE(handoff, nullptr);
    ASSERT_EQ(handoff->request_ids, (std::vector<std::string>{"p"}));
    EXPECT_TRUE(scheduler.FlatPdTransferPinned("p"));

    SendPdEvent(scheduler, pd::SucceededEvent{"p"});
    EXPECT_FALSE(scheduler.FlatPdTransferPinned("p"));
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start);
}

TEST(FlatKvPdLifecycle, GenericAbortReleasesTransferPins) {
    Scheduler scheduler{MakeFlatPdConfig(Role::kD)};
    const std::int32_t free_at_start = scheduler.FlatPoolFreeBlocks();
    scheduler.SubmitRequests({MakePdRequest("guarded", 8)});
    SendPdEvent(scheduler, pd::BootstrappedEvent{"guarded"});
    scheduler.NextExecutionPlan();
    ASSERT_TRUE(scheduler.FlatPdTransferPinned("guarded"));
    const std::int32_t free_while_pinned = scheduler.FlatPoolFreeBlocks();

    EXPECT_THROW(SendPdEvent(scheduler, pd::SucceededEvent{"guarded"}), std::logic_error);
    EXPECT_TRUE(scheduler.FlatPdTransferPinned("guarded"));
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_while_pinned);
    EXPECT_NO_THROW(SendForwardAbort(scheduler, "guarded"));
    EXPECT_FALSE(scheduler.FlatPdTransferPinned("guarded"));
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start);
}

TEST(FlatKvPdLifecycle, QuiescentFailureAndDuplicateAbortAreIdempotent) {
    Scheduler scheduler{MakeFlatPdConfig(Role::kD)};
    const std::int32_t free_at_start = scheduler.FlatPoolFreeBlocks();
    scheduler.SubmitRequests({MakePdRequest("failed", 7)});
    SendPdEvent(scheduler, pd::BootstrappedEvent{"failed"});
    scheduler.NextExecutionPlan();
    ASSERT_TRUE(scheduler.FlatPdTransferPinned("failed"));

    ExecutionEvent terminal;
    terminal.With(PDEvent{pd::FailedEvent{"failed"}});
    terminal.With(ForwardEvent{forward::Abort{.request_id = "failed"}});
    EXPECT_NO_THROW(scheduler.Advance(terminal));
    EXPECT_FALSE(scheduler.FlatPdTransferPinned("failed"));
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start);
    scheduler.NextExecutionPlan();
    EXPECT_EQ(scheduler.GetRequestTokenSize("failed"), -1);
}

TEST(FlatKvPdLifecycle, DuplicateSubmissionIsRejectedAtomically) {
    Scheduler scheduler{MakeFlatPdConfig(Role::kD)};
    EXPECT_THROW(scheduler.SubmitRequests({
                     MakePdRequest("duplicate", 4),
                     MakePdRequest("duplicate", 4),
                 }),
                 std::invalid_argument);
    EXPECT_EQ(scheduler.WaitingSize(), 0u);
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), 63);
}

TEST(FlatKvPdConfig, TransferPolicyMustMatchSchedulerLayout) {
    SchedulerConfig unspecified = MakeFlatPdConfig(Role::kD);
    unspecified.paged_cache_groups[0].transfer_policy = PagedCacheTransferPolicy::Unspecified;
    EXPECT_THROW((void)Scheduler{unspecified}, std::invalid_argument);

    SchedulerConfig dense_mismatch = MakeFlatPdConfig(Role::kD);
    dense_mismatch.paged_cache_groups[0].transfer_policy = PagedCacheTransferPolicy::LatestSnapshot;
    EXPECT_THROW((void)Scheduler{dense_mismatch}, std::invalid_argument);

    SchedulerConfig state_mismatch = MakeFlatPdConfig(Role::kD);
    state_mismatch.paged_cache_groups[1].transfer_policy = PagedCacheTransferPolicy::FullSuffix;
    EXPECT_THROW((void)Scheduler{state_mismatch}, std::invalid_argument);

    SchedulerConfig standalone = MakeFlatPdConfig(Role::kFused);
    standalone.enable_flatkv_pd = false;
    for (PagedCacheGroupConfig& group : standalone.paged_cache_groups) {
        group.transfer_policy = PagedCacheTransferPolicy::Unspecified;
    }
    EXPECT_NO_THROW((void)Scheduler{standalone});
}

TEST(FlatKvPdConfig, OverlapIsDecodeOnly) {
    EXPECT_NO_THROW((void)Scheduler{MakeFlatPdConfig(Role::kD, /*total_pages=*/64, /*overlap_schedule_depth=*/1)});
    EXPECT_THROW((void)Scheduler{MakeFlatPdConfig(Role::kP, /*total_pages=*/64, /*overlap_schedule_depth=*/1)},
                 std::invalid_argument);
}

TEST(FlatKvPdOom, OversizedDecodeDestinationFailsWithStructuredReason) {
    Scheduler scheduler{MakeFlatPdConfig(Role::kD, /*total_pages=*/8)};
    const std::int32_t free_at_start = scheduler.FlatPoolFreeBlocks();
    scheduler.SubmitRequests({MakePdRequest("too-large", 8)});
    SendPdEvent(scheduler, pd::BootstrappedEvent{"too-large"});
    const ExecutionPlan plan = scheduler.NextExecutionPlan();
    ASSERT_EQ(plan.flat_terminal_errors.size(), 1u);
    const FlatTerminalError& error = plan.flat_terminal_errors.front();
    EXPECT_EQ(error.request_id, "too-large");
    EXPECT_EQ(error.reason, FlatTerminalReason::kPromptExceedsPoolCapacity);
    EXPECT_EQ(error.required_pages, 11);
    EXPECT_EQ(error.capacity_pages, 7);
    EXPECT_EQ(plan.flat_oom_request_ids, (std::vector<std::string>{"too-large"}));
    EXPECT_EQ(scheduler.FlatPoolFreeBlocks(), free_at_start);
    EXPECT_FALSE(scheduler.FlatPdTransferPinned("too-large"));
}

#endif  // TOKENSPEED_FLAT_KVCACHE

}  // namespace
}  // namespace tokenspeed::test
