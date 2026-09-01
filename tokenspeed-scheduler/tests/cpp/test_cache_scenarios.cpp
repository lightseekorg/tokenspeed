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

// End-to-end scenario tests for the two-level KV-cache FSM path. Fused
// scheduling uses release-and-requeue (see RetractSuite); Decode PD uses Host
// writeback and recovery.

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include "scheduler/operations/cache.h"
#include "cache_test_access.h"
#include "integration_test_helper.h"

namespace tokenspeed::test {

namespace {

static_assert(
    std::is_same_v<decltype(std::declval<fsm::SchedulePrefillFirstChunkEvent&>()(std::declval<fsm::Submitted&&>())),
                   std::variant<fsm::PrefillDone, fsm::PrefillAwaitingResult, fsm::Prefilling, fsm::RemotePrefilling>>);
static_assert(
    std::is_same_v<decltype(std::declval<fsm::SchedulePrefillFirstChunkEvent&>()(std::declval<fsm::Retracted&&>())),
                   std::variant<fsm::PrefillDone, fsm::PrefillAwaitingResult, fsm::Prefilling, fsm::RemotePrefilling>>);
static_assert(std::is_same_v<decltype(std::declval<fsm::SchedulePrefillEvent&>()(std::declval<fsm::Prefilling&&>())),
                             std::variant<fsm::PrefillDone, fsm::PrefillAwaitingResult, fsm::Prefilling>>);

std::pair<bool, std::string> ClearL1CacheWithCapturedLog(Scheduler* scheduler) {
    std::ostringstream output;
    auto previous_logger = spdlog::default_logger();
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(output);
    auto logger = std::make_shared<spdlog::logger>("clear-l1-cache-test", std::move(sink));
    logger->set_pattern("%v");
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(std::move(logger));
    const bool cleared = scheduler->ClearL1Cache();
    spdlog::set_default_logger(std::move(previous_logger));
    return {cleared, output.str()};
}

CacheGroupConfig MakeGroup(const std::string& id, std::int32_t block_granularity, std::int32_t total_pages,
                           CacheGroupConfig::Retention retention, CacheGroupFamily family,
                           std::int32_t sliding_window_tokens = 0) {
    CacheGroupConfig g;
    g.group_id = id;
    g.rows_per_page = block_granularity;
    g.entry_stride_tokens = 1;
    g.total_pages = total_pages;
    g.retention = retention;
    g.family = family;
    if (sliding_window_tokens > 0) {
        g.sliding_window_tokens = sliding_window_tokens;
    }
    return g;
}

// Collect every real (>0) physical page id across all rows of a group.
std::vector<std::int32_t> RealPages(const std::vector<std::vector<std::int32_t>>& group) {
    std::vector<std::int32_t> out;
    for (const auto& row : group) {
        for (std::int32_t id : row) {
            if (id > 0) out.push_back(id);
        }
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Chunked prefill: first-chunk admission followed by one admission per chunk.
// ---------------------------------------------------------------------------
class ChunkedPrefillSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 4;  // 4 tokens = 2 pages per chunk
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                      /*sliding_window_tokens=*/4),
        };
        return cfg;
    }
};

TEST_F(ChunkedPrefillSuite, MultiChunkPrefillGrowsFullTableThenDecodes) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    // 8 tokens (4 pages) with max_scheduled_tokens=4 -> 2 prefill chunks.
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));

    ExecutionPlan chunk1 = PlanOnce();
    const ForwardBatch* op1 = FindForwardBatch(chunk1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->block_tables.count("full"), 1u);
    const std::size_t full_after_c1 = op1->block_tables.at("full").at(0).size();
    EXPECT_GT(full_after_c1, 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);

    ExecutionPlan chunk2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(chunk2);
    ASSERT_NE(op2, nullptr);
    const auto& full_c2 = op2->block_tables.at("full").at(0);
    EXPECT_GT(full_c2.size(), full_after_c1) << "second chunk should extend the full-history block table";
    for (std::int32_t id : full_c2) {
        EXPECT_GT(id, 0) << "full-history row must have no null hole";
    }

    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    SendForwardDone("r1", {100});

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start)
        << "all pages returned to the pool after a chunked-prefill request finishes";
}

class MambaChunkAlignmentSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 4;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 6;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("state", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::State),
        };
        return cfg;
    }
};

TEST_F(MambaChunkAlignmentSuite, PartialPrefillEndsAtStatePageBoundary) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/3));  // 12 tokens

    for (std::int32_t expected_prefix : {0, 4, 8}) {
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(plan);
        ASSERT_NE(op, nullptr);
        ASSERT_EQ(op->input_lengths.size(), 1u);
        ASSERT_EQ(op->extend_prefix_lens.size(), 1u);
        EXPECT_EQ(op->input_lengths[0], 4);
        EXPECT_EQ(op->extend_prefix_lens[0], expected_prefix);
    }
}

class MambaStateCheckpointSplitSuite : public MambaChunkAlignmentSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = MambaChunkAlignmentSuite::MakeConfig();
        cfg.max_scheduled_tokens = 64;
        cfg.disable_prefix_cache = false;
        return cfg;
    }
};

TEST_F(MambaStateCheckpointSplitSuite, ReservesAndBatchesDependentTails) {
    RequestSpec first = MakeRequestSpec("a", /*num_pages=*/3);
    RequestSpec second = MakeRequestSpec("b", /*num_pages=*/3, /*start=*/100);
    first.tokens.resize(10);
    second.tokens.resize(10);
    Submit({first, second});

    ExecutionPlan bodies = PlanOnce();
    const ForwardBatch* body_op = FindForwardBatch(bodies);
    ASSERT_NE(body_op, nullptr);
    EXPECT_EQ(body_op->request_ids, (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(body_op->input_lengths, (std::vector<std::int32_t>{8, 8}));
    for (const auto& row : body_op->block_tables.at("state")) {
        EXPECT_EQ(row.size(), 3u);
    }

    ExecutionPlan tails = PlanOnce();
    const ForwardBatch* tail_op = FindForwardBatch(tails);
    ASSERT_NE(tail_op, nullptr);
    EXPECT_EQ(tail_op->request_ids, (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(tail_op->extend_prefix_lens, (std::vector<std::int32_t>{8, 8}));
    EXPECT_EQ(tail_op->input_lengths, (std::vector<std::int32_t>{2, 2}));
}

TEST(MambaStateCheckpointCapacityTest, CountsInputBodyAndReservedTailCheckpoints) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = 4;
    cfg.device_allocator.total_pages = 3;  // null + two usable state blocks
    cfg.host_allocator.total_pages = 0;
    cfg.max_scheduled_tokens = 8;
    cfg.max_batch_size = 1;
    cfg.disable_l2_cache = true;
    cfg.cache_groups = {
        MakeGroup("state", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                  CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::State),
    };

    Scheduler scheduler{std::move(cfg)};

    // Up to eight prompt tokens plus the decode reservation fit in two
    // blocks. A longer prompt can reuse one input checkpoint while exposing an
    // aligned body and reserving its sub-page tail, which needs three.
    EXPECT_EQ(scheduler.MaxSingleRequestTokens(), 9);
    RequestSpec too_long{
        .request_id = "too-long",
        .tokens = std::vector<std::int32_t>(14, 1),
    };
    EXPECT_THROW(scheduler.SubmitRequests({too_long}), std::invalid_argument);
}

TEST(MambaStateCheckpointCapacityTest, CountsFirstChunkBodyAndSubPageTail) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = 4;
    cfg.device_allocator.total_pages = 4;  // null + three usable state blocks
    cfg.host_allocator.total_pages = 0;
    cfg.max_scheduled_tokens = 8;
    cfg.max_batch_size = 1;
    cfg.disable_l2_cache = true;
    cfg.cache_groups = {
        MakeGroup("state", /*block_granularity=*/1, cfg.device_allocator.total_pages,
                  CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::State),
    };

    Scheduler scheduler{std::move(cfg)};

    // A seven-token prompt would split into a four-token body and a
    // three-token tail. The body checkpoint plus the reserved tail need four
    // state blocks, so three usable blocks cannot admit it and its decode.
    EXPECT_EQ(scheduler.MaxSingleRequestTokens(), 7);
    RequestSpec too_long{
        .request_id = "too-long",
        .tokens = std::vector<std::int32_t>(7, 1),
    };
    EXPECT_THROW(scheduler.SubmitRequests({too_long}), std::invalid_argument);
}

class MambaStateCheckpointNoPrefixCacheSuite : public MambaStateCheckpointSplitSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = MambaStateCheckpointSplitSuite::MakeConfig();
        cfg.disable_prefix_cache = true;
        return cfg;
    }
};

TEST_F(MambaStateCheckpointNoPrefixCacheSuite, KeepsSingleFinalChunk) {
    RequestSpec spec = MakeRequestSpec("r1", /*num_pages=*/3);
    spec.tokens.resize(10);
    Submit(spec);

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->input_lengths, std::vector<std::int32_t>{10});
}

class MambaStateCheckpointPrefillRoleSuite : public MambaStateCheckpointSplitSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = MambaStateCheckpointSplitSuite::MakeConfig();
        cfg.role = Role::kP;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.transfer_policy =
                group.IsSnapshotStateGroup() ? CacheTransferPolicy::LatestSnapshot : CacheTransferPolicy::FullSuffix;
        }
        return cfg;
    }

    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{request_id});
        scheduler_->Advance(std::move(event));
    }
};

TEST_F(MambaStateCheckpointPrefillRoleSuite, SplitsLocalPrefillOnPrefillWorker) {
    RequestSpec spec = MakeRequestSpec("r1", /*num_pages=*/3);
    spec.tokens.resize(10);
    Submit(spec);
    SendBootstrapped("r1");

    ExecutionPlan body_plan = PlanOnce();
    const ForwardBatch* body = FindForwardBatch(body_plan);
    ASSERT_NE(body, nullptr);
    EXPECT_EQ(body->extend_prefix_lens, std::vector<std::int32_t>{0});
    EXPECT_EQ(body->input_lengths, std::vector<std::int32_t>{8});

    ExecutionPlan tail_plan = PlanOnce();
    const ForwardBatch* tail = FindForwardBatch(tail_plan);
    ASSERT_NE(tail, nullptr);
    EXPECT_EQ(tail->extend_prefix_lens, std::vector<std::int32_t>{8});
    EXPECT_EQ(tail->input_lengths, std::vector<std::int32_t>{2});
}

TEST_F(MambaStateCheckpointPrefillRoleSuite, HoldsRemoteDecodeUntilTheFinalChunkResultLands) {
    // A request turns PrefillDone when its last chunk is SCHEDULED, and its
    // remote decode needs the bootstrap token that lands with that chunk's
    // result.  The plan must hold the remote decode until then -- and emit
    // it on its own stream (plan.remote_decode), never as forward work.
    RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/3);
    first.tokens.resize(10);
    Submit(first);
    SendBootstrapped("r1");

    PlanOnce();  // r1 body chunk
    ExecutionPlan tail_plan = PlanOnce();
    const ForwardBatch* tail = FindForwardBatch(tail_plan);
    ASSERT_NE(tail, nullptr);
    // r1 is PrefillDone from here on.
    EXPECT_FALSE(tail_plan.remote_decode.has_value());

    // Its ExtendResult has not arrived, so the plan holds the remote decode.
    ExecutionPlan while_pending = PlanOnce();
    const ForwardBatch* pending = FindForwardBatch(while_pending);
    ASSERT_NE(pending, nullptr);
    EXPECT_TRUE(pending->request_ids.empty());
    EXPECT_FALSE(while_pending.remote_decode.has_value());

    ExecutionEvent result;
    result.With(forward::ExtendResult{.request_id = "r1", .tokens = {42}, .spec_candidate_ids = {42, 7, 9}});
    scheduler_->Advance(std::move(result));

    // Result in hand: the remote decode goes out on the plan's own stream,
    // self-contained -- it carries the bootstrap token (the sampled first
    // decode token, now LastToken()) and the drafter candidates, so the
    // transfer peer needs no side channel.
    ExecutionPlan ready = PlanOnce();
    ASSERT_TRUE(ready.remote_decode.has_value());
    EXPECT_EQ(ready.remote_decode->request_ids, std::vector<std::string>{"r1"});
    EXPECT_EQ(ready.remote_decode->NumExtends(), 0u);
    EXPECT_EQ(ready.remote_decode->decode_input_ids, std::vector<std::int32_t>{42});
    EXPECT_EQ(ready.remote_decode->spec_candidate_ids, (std::vector<std::vector<std::int32_t>>{{42, 7, 9}}));
    const ForwardBatch* forward = FindForwardBatch(ready);
    ASSERT_NE(forward, nullptr);
    EXPECT_TRUE(forward->request_ids.empty());
}

TEST_F(MambaStateCheckpointPrefillRoleSuite, EmitsRemoteDecodeAlongsideOngoingPrefillWork) {
    // Everything dispatchable dispatches in one round: a ready remote decode
    // rides plan.remote_decode while another request's prefill keeps filling
    // the ForwardBatch.  Neither waits for the other, and a remote decode
    // still awaiting its result never leaks into the forward work.
    RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/3);
    first.tokens.resize(10);
    Submit(first);
    SendBootstrapped("r1");

    PlanOnce();  // r1 body chunk
    PlanOnce();  // r1 tail chunk -> r1 PrefillDone, result pending

    RequestSpec second = MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/100);
    second.tokens.resize(10);
    Submit(second);
    SendBootstrapped("r2");

    // r1's result is still pending: its remote decode is held, and r2's
    // prefill proceeds -- the pipeline never stalls for a completed prompt.
    ExecutionPlan held = PlanOnce();
    EXPECT_FALSE(held.remote_decode.has_value());
    const ForwardBatch* body = FindForwardBatch(held);
    ASSERT_NE(body, nullptr);
    EXPECT_EQ(body->request_ids, std::vector<std::string>{"r2"});
    EXPECT_EQ(body->input_lengths, std::vector<std::int32_t>{8});

    ExecutionEvent result;
    result.With(forward::ExtendResult{.request_id = "r1", .tokens = {42}});
    scheduler_->Advance(std::move(result));

    // One round, both streams: r1's remote decode and r2's tail chunk.
    ExecutionPlan combined = PlanOnce();
    ASSERT_TRUE(combined.remote_decode.has_value());
    EXPECT_EQ(combined.remote_decode->request_ids, std::vector<std::string>{"r1"});
    EXPECT_EQ(combined.remote_decode->decode_input_ids, std::vector<std::int32_t>{42});
    const ForwardBatch* tail = FindForwardBatch(combined);
    ASSERT_NE(tail, nullptr);
    EXPECT_EQ(tail->request_ids, std::vector<std::string>{"r2"});
    EXPECT_EQ(tail->extend_prefix_lens, std::vector<std::int32_t>{8});
    EXPECT_EQ(tail->input_lengths, std::vector<std::int32_t>{2});
}

class MambaStateCheckpointDecodeRoleSuite : public MambaStateCheckpointPrefillRoleSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = MambaStateCheckpointPrefillRoleSuite::MakeConfig();
        cfg.role = Role::kD;
        return cfg;
    }
};

TEST_F(MambaStateCheckpointDecodeRoleSuite, KeepsRemoteAdmissionWhole) {
    RequestSpec spec = MakeRequestSpec("r1", /*num_pages=*/3);
    spec.tokens.resize(10);
    Submit(spec);
    SendBootstrapped("r1");

    ExecutionPlan admission_plan = PlanOnce();
    // Landing on the remote-prefill stream IS the statement that the peer
    // prefills this prompt; the model batch carries only local work.
    const ForwardBatch* admission = FindRemoteAdmission(admission_plan);
    ASSERT_NE(admission, nullptr);
    EXPECT_EQ(admission->extend_prefix_lens, std::vector<std::int32_t>{0});
    EXPECT_EQ(admission->input_lengths, std::vector<std::int32_t>{10});
}

class MambaSparsePrefillSuite : public MambaChunkAlignmentSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = MambaChunkAlignmentSuite::MakeConfig();
        cfg.max_scheduled_tokens = 12;
        return cfg;
    }
};

TEST_F(MambaSparsePrefillSuite, LocalChunkDefersStateDecodeReservationUntilCompletion) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/3));  // 12 tokens

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->input_lengths, std::vector<std::int32_t>{12});

    EXPECT_EQ(RealPages(op->block_tables.at("full")).size(), 4u);

    const auto& state = op->block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 3u);
    EXPECT_EQ(state[0], 0);
    EXPECT_EQ(state[1], 0);
    EXPECT_GT(state[2], 0);  // final prefill checkpoint
    EXPECT_EQ(RealPages(op->block_tables.at("state")).size(), 1u);

    ASSERT_EQ(plan.pages_to_zero.count("state"), 1u);
    EXPECT_EQ(plan.pages_to_zero.at("state").size(), 1u);

    SendForwardDone("r1", {42});
    ExecutionPlan decode_plan = PlanOnce();
    const ForwardBatch* decode = FindForwardBatch(decode_plan);
    ASSERT_NE(decode, nullptr);
    const auto& decode_state = decode->block_tables.at("state").at(0);
    ASSERT_EQ(decode_state.size(), 4u);
    EXPECT_GT(decode_state[2], 0);
    EXPECT_GT(decode_state[3], 0);
    ASSERT_EQ(decode_plan.pages_to_zero.count("state"), 1u);
    EXPECT_EQ(decode_plan.pages_to_zero.at("state").size(), 1u);
}

class MambaOverlapRollingStateSuite : public MambaSparsePrefillSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = MambaSparsePrefillSuite::MakeConfig();
        cfg.overlap_schedule_depth = 1;
        return cfg;
    }
};

TEST_F(MambaOverlapRollingStateSuite, FinalPrefillUsesInputAndOutputWithoutDecodeReserve) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));  // 16 tokens

    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    ExecutionPlan final_prefill = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(final_prefill);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->input_lengths, std::vector<std::int32_t>{4});

    const auto& state = op->block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 4u);
    EXPECT_EQ(state[0], 0);
    EXPECT_EQ(state[1], 0);
    EXPECT_GT(state[2], 0);  // rolling input
    EXPECT_GT(state[3], 0);  // final prefill output
    EXPECT_EQ(RealPages(op->block_tables.at("state")).size(), 2u);
}

class MambaMixedBudgetSuite : public MambaChunkAlignmentSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = MambaChunkAlignmentSuite::MakeConfig();
        cfg.max_scheduled_tokens = cfg.prefix_granularity;
        cfg.enable_mixed_prefill_decode = true;
        return cfg;
    }
};

TEST_F(MambaMixedBudgetSuite, StatePrefillKeepsOnePageOfMixedBudget) {
    Submit(MakeRequestSpec("decode", /*num_pages=*/1));
    PlanOnce();
    SendForwardDone("decode", {42});

    Submit(MakeRequestSpec("prefill", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    EXPECT_EQ(op->request_ids.front(), "prefill");
    EXPECT_EQ(op->input_lengths.front(), config_.prefix_granularity);
}

class MambaMixedSpareBudgetSuite : public MambaMixedBudgetSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = MambaMixedBudgetSuite::MakeConfig();
        cfg.max_scheduled_tokens = cfg.prefix_granularity + cfg.decode_input_tokens;
        return cfg;
    }
};

TEST_F(MambaMixedSpareBudgetSuite, DecodeUsesOnlyBudgetAboveReservedStatePage) {
    Submit(MakeRequestSpec("decode", /*num_pages=*/1));
    PlanOnce();
    SendForwardDone("decode", {42});

    Submit(MakeRequestSpec("prefill", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 2u);
    const auto decode = std::ranges::find(op->request_ids, "decode");
    const auto prefill = std::ranges::find(op->request_ids, "prefill");
    ASSERT_NE(decode, op->request_ids.end());
    ASSERT_NE(prefill, op->request_ids.end());
    EXPECT_EQ(op->input_lengths[std::distance(op->request_ids.begin(), decode)], config_.decode_input_tokens);
    EXPECT_EQ(op->input_lengths[std::distance(op->request_ids.begin(), prefill)], config_.prefix_granularity);
}

TEST(MambaChunkAlignmentConfigTest, RejectsBudgetSmallerThanStatePage) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = 4;
    cfg.device_allocator.total_pages = 64;
    cfg.host_allocator.total_pages = 64;
    cfg.max_scheduled_tokens = 3;
    cfg.max_batch_size = 8;
    cfg.disable_l2_cache = true;
    cfg.disable_prefix_cache = true;
    cfg.cache_groups = {
        MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                  CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
        MakeGroup("state", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                  CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::State),
    };

    EXPECT_THROW((void)Scheduler(std::move(cfg)), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Three cache groups: full + two sliding windows. Group 0 stays full-history
// to honor the batch consumer's block_tables_[0] contract.
// ---------------------------------------------------------------------------
class ThreeGroupSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 96;
        cfg.host_allocator.total_pages = 96;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("swa_small", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                      /*sliding_window_tokens=*/4),
            MakeGroup("swa_big", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                      /*sliding_window_tokens=*/8),
        };
        return cfg;
    }
};

TEST_F(ThreeGroupSuite, ThreeGroupsEachEmitARowAndReclaim) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(prefill);
    ASSERT_NE(op, nullptr);

    ASSERT_EQ(op->block_tables.count("full"), 1u);
    ASSERT_EQ(op->block_tables.count("swa_small"), 1u);
    ASSERT_EQ(op->block_tables.count("swa_big"), 1u);
    EXPECT_EQ(op->block_tables.at("full").size(), 1u);
    EXPECT_EQ(op->block_tables.at("swa_small").size(), 1u);
    EXPECT_EQ(op->block_tables.at("swa_big").size(), 1u);

    auto full_pages = RealPages(op->block_tables.at("full"));
    auto small_pages = RealPages(op->block_tables.at("swa_small"));
    auto big_pages = RealPages(op->block_tables.at("swa_big"));
    std::set<std::int32_t> all(full_pages.begin(), full_pages.end());
    all.insert(small_pages.begin(), small_pages.end());
    all.insert(big_pages.begin(), big_pages.end());
    EXPECT_EQ(all.size(), full_pages.size() + small_pages.size() + big_pages.size())
        << "groups must not share physical pages";

    SendForwardDone("r1", {42});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Sub-page (w=3 < P=4) and page-straddling (w=5 = P+1) windows (M14): pins
// per-group slide independence and the <=2-real-page steady state.
// ---------------------------------------------------------------------------
class SubPageWindowSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 4;
        cfg.device_allocator.total_pages = 96;
        cfg.host_allocator.total_pages = 96;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("swa_w3", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                      /*sliding_window_tokens=*/3),
            MakeGroup("swa_w5", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                      /*sliding_window_tokens=*/5),
        };
        return cfg;
    }
};

TEST_F(SubPageWindowSuite, SubPageWindowsPlateauAtTwoRealPages) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    ExecutionPlan prefill = PlanOnce();
    ASSERT_NE(FindForwardBatch(prefill), nullptr);
    SendForwardDone("r1", {1000});

    for (std::int32_t step = 0; step < 24; ++step) {
        ExecutionPlan decode = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(decode);
        ASSERT_NE(op, nullptr) << "decode step " << step;
        // fullySlidOutBlocks frees only FULLY slid-out pages: 1 <= real pages <= 2.
        const std::size_t w3_real = RealPages(op->block_tables.at("swa_w3")).size();
        const std::size_t w5_real = RealPages(op->block_tables.at("swa_w5")).size();
        EXPECT_GE(w3_real, 1u) << "w=3 lost its live tail page at step " << step;
        EXPECT_LE(w3_real, 2u) << "w=3 working set exceeded 2 pages at step " << step;
        EXPECT_GE(w5_real, 1u) << "w=5 lost its live tail page at step " << step;
        EXPECT_LE(w5_real, 2u) << "w=5 working set exceeded 2 pages at step " << step;
        SendForwardDone("r1", {1001 + step});
    }

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

TEST_F(SubPageWindowSuite, StraddlingWindowHoldsPreviousPage) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    PlanOnce();
    SendForwardDone("r1", {1000});

    bool diverged = false;
    for (std::int32_t step = 0; step < 8; ++step) {
        ExecutionPlan decode = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(decode);
        ASSERT_NE(op, nullptr);
        const std::size_t w3_real = RealPages(op->block_tables.at("swa_w3")).size();
        const std::size_t w5_real = RealPages(op->block_tables.at("swa_w5")).size();
        EXPECT_LE(w3_real, w5_real) << "a smaller window can never hold more pages, step " << step;
        if (w3_real < w5_real) {
            diverged = true;  // the straddling window (w=5) holds one more real page
        }
        SendForwardDone("r1", {1001 + step});
    }
    EXPECT_TRUE(diverged) << "w=3 and w=5 never diverged: per-group slides are not independent";

    SendFinish("r1");
    PlanOnce();
}

// ---------------------------------------------------------------------------
// Two full-history groups (no sliding window at all).
// ---------------------------------------------------------------------------
class AllFullTwoGroupSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full_a", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("full_b", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
        };
        return cfg;
    }
};

TEST_F(AllFullTwoGroupSuite, BothFullGroupsKeepHistoryNoHoles) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();  // prefill
    SendForwardDone("r1", {42});

    std::optional<ExecutionPlan> last;
    int tok = 43;
    for (int i = 0; i < 4; ++i) {
        last = PlanOnce();
        ASSERT_NE(FindForwardBatch(*last), nullptr);
        SendForwardDone("r1", {tok++});
    }
    const ForwardBatch* op = FindForwardBatch(*last);
    ASSERT_NE(op, nullptr);
    for (const char* key : {"full_a", "full_b"}) {
        const auto& row = op->block_tables.at(key).at(0);
        for (std::int32_t id : row) {
            EXPECT_GT(id, 0) << key << " (full-history) must not develop a null hole";
        }
    }

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Shared-pool accounting: out-of-order finishes each return exactly their pages.
// ---------------------------------------------------------------------------
class PoolAccountingSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                      /*sliding_window_tokens=*/4),
        };
        return cfg;
    }
};

TEST_F(PoolAccountingSuite, ThreeRequestsOutOfOrderFinishReclaimExactly) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/4, /*start=*/101));
    Submit(MakeRequestSpec("r3", /*num_pages=*/3, /*start=*/201));
    PlanOnce();  // prefill all three (max_scheduled_tokens=64 covers them)
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    const std::int32_t free_after_prefill = scheduler_->PoolFreeBlocks();
    EXPECT_LT(free_after_prefill, free_at_start) << "prefill must consume pages from the shared pool";

    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    SendForwardDone("r3", {242});

    SendFinish("r2");
    PlanOnce();
    SendFinish("r1");
    PlanOnce();
    EXPECT_LT(scheduler_->PoolFreeBlocks(), free_at_start) << "pool not fully reclaimed while r3 is still live";
    SendFinish("r3");
    PlanOnce();

    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "every page returns to the pool once all requests finish";
}

// Chunked prefill slides the SWA window DURING prefill, then decode keeps
// sliding. Window convention used below: with N = tokens computed BEFORE a
// round's forward, the pending query at N attends keys [N-W+1, N], so the
// first kept page is (N-W+1)/block_granularity and everything below it is freed.
TEST_F(ChunkedPrefillSuite, ChunkedPrefillThenSwaSlidesToNullHole) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    // 12 tokens (6 pages), max_scheduled_tokens=4 -> 3 prefill chunks.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    PlanOnce();  // chunk 1
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    // Chunk 2: N=4 -> first kept token 4-4+1=1 -> first kept page 0: no hole.
    ExecutionPlan chunk2 = PlanOnce();
    const ForwardBatch* c2op = FindForwardBatch(chunk2);
    ASSERT_NE(c2op, nullptr);
    {
        const auto& swa_c2 = c2op->block_tables.at("swa").at(0);
        ASSERT_EQ(swa_c2.size(), 4u);
        EXPECT_EQ(std::count(swa_c2.begin(), swa_c2.end(), 0), 0)
            << "N=4, W=4: no page fully below token 1, so chunk 2 punches nothing";
    }
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    const std::int32_t free_after_c2 = scheduler_->PoolFreeBlocks();

    // Chunk 3: N=8 -> first kept token 5 -> page 5/2=2: slots 0,1 punched MID-PREFILL.
    ExecutionPlan chunk3 = PlanOnce();  // chunk 3 (last)
    const ForwardBatch* c3op = FindForwardBatch(chunk3);
    ASSERT_NE(c3op, nullptr);
    {
        const auto& swa_c3 = c3op->block_tables.at("swa").at(0);
        ASSERT_EQ(swa_c3.size(), 7u);
        for (int s = 0; s <= 1; ++s) EXPECT_EQ(swa_c3[s], 0) << "slot " << s << " punched during prefill";
        for (int s = 2; s <= 6; ++s) EXPECT_GT(swa_c3[s], 0) << "slot " << s;
        for (std::int32_t id : c3op->block_tables.at("full").at(0)) {
            EXPECT_GT(id, 0) << "full group keeps every chunk-built page";
        }
    }
    // Chunk-3 balance: slide frees 2 SWA pages, the chunk takes 2/group and
    // the physically-backed decode reservation takes 1/group.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_after_c2 + 2 - 4 - 2)
        << "the mid-prefill slide must return the out-of-window pages to the pool";

    SendForwardDone("r1", {99});  // container size 13 (12 prompt + 1 sampled)

    // swa_rows[i] = the swa row round i's op carried (after slide + acquire).
    std::vector<std::vector<std::int32_t>> swa_rows;
    int tok = 100;
    for (int i = 0; i < 4; ++i) {
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(plan);
        ASSERT_NE(op, nullptr);
        for (std::int32_t id : op->block_tables.at("full").at(0)) {
            EXPECT_GT(id, 0) << "full group must keep chunk-built history without holes (round " << i << ")";
        }
        swa_rows.push_back(op->block_tables.at("swa").at(0));
        SendForwardDone("r1", {tok++});
    }

    auto null_count = [](const std::vector<std::int32_t>& row) { return std::count(row.begin(), row.end(), 0); };

    // Round 0 (finalize): N=12 -> first kept page 4; + reserve page -> 7 slots, 4 holes.
    ASSERT_EQ(swa_rows[0].size(), 7u);
    EXPECT_EQ(null_count(swa_rows[0]), 4) << "finalize slides at the full prefill length";
    for (int s = 0; s <= 3; ++s) EXPECT_EQ(swa_rows[0][s], 0) << "slot " << s;
    for (int s = 4; s <= 6; ++s) EXPECT_GT(swa_rows[0][s], 0) << "slot " << s;

    // Round 1: N=13 -> first kept page 5; tail room absorbs the acquire.
    ASSERT_EQ(swa_rows[1].size(), 7u);
    EXPECT_EQ(null_count(swa_rows[1]), 5);
    for (int s = 0; s <= 4; ++s) EXPECT_EQ(swa_rows[1][s], 0) << "slot " << s;
    for (int s = 5; s <= 6; ++s) EXPECT_GT(swa_rows[1][s], 0) << "slot " << s;

    // Round 2: N=14 -> first kept token 11 -> page 5 (unchanged); acquire adds
    // page 7. Sliding at the container size 15 instead would free slot 5 early.
    ASSERT_EQ(swa_rows[2].size(), 8u);
    EXPECT_EQ(null_count(swa_rows[2]), 5);
    EXPECT_GT(swa_rows[2][5], 0) << "slot 5 must survive round 2: key 11 of the pending query lives there";
    for (int s = 6; s <= 7; ++s) EXPECT_GT(swa_rows[2][s], 0) << "slot " << s;

    // Round 3: N=15 -> first kept token 12 -> first kept page 6.
    ASSERT_EQ(swa_rows[3].size(), 8u);
    EXPECT_EQ(null_count(swa_rows[3]), 6);
    EXPECT_EQ(swa_rows[3][5], 0) << "slot 5 slides out once the query window has moved past key 11";
    for (int s = 6; s <= 7; ++s) EXPECT_GT(swa_rows[3][s], 0) << "slot " << s;

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

TEST_F(ThreeGroupSuite, TwoRequestsBatchedAcrossThreeGroupsNoCollision) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(prefill);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 2u);

    for (const char* key : {"full", "swa_small", "swa_big"}) {
        ASSERT_EQ(op->block_tables.count(key), 1u) << key;
        EXPECT_EQ(op->block_tables.at(key).size(), 2u) << key;
    }

    std::vector<std::int32_t> every;
    for (const char* key : {"full", "swa_small", "swa_big"}) {
        auto pages = RealPages(op->block_tables.at(key));
        every.insert(every.end(), pages.begin(), pages.end());
    }
    std::vector<std::int32_t> sorted = every;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end())
        << "no physical page may be shared across requests or groups";

    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    SendFinish("r1");
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Mixed batch: with enable_mixed_prefill_decode a decode and a prefill share
// one SoA op; stable_partition puts prefill rows ahead of decode rows.
// ---------------------------------------------------------------------------
class MixedBatchSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        cfg.enable_mixed_prefill_decode = true;  // decode + prefill in one plan

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                      /*sliding_window_tokens=*/4),
        };
        return cfg;
    }
};

TEST_F(MixedBatchSuite, PrefillAndDecodeShareOnePlan) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();                   // r1 prefill
    SendForwardDone("r1", {42});  // r1 -> decode

    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan mixed = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(mixed);
    ASSERT_NE(op, nullptr);

    ASSERT_EQ(op->request_ids.size(), 2u);
    EXPECT_EQ(op->NumExtends(), 1u) << "exactly one prefill row (r2)";
    EXPECT_EQ(op->decode_input_ids.size(), 1u) << "exactly one decode row (r1)";

    EXPECT_EQ(op->request_ids.at(0), "r2") << "prefill partitioned first";
    EXPECT_EQ(op->request_ids.at(1), "r1") << "decode after prefill";

    for (const char* key : {"full", "swa"}) {
        ASSERT_EQ(op->block_tables.count(key), 1u) << key;
        ASSERT_EQ(op->block_tables.at(key).size(), 2u) << key;
        auto pages = RealPages(op->block_tables.at(key));
        std::vector<std::int32_t> sorted = pages;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end())
            << key << ": two requests must not share a physical page";
    }

    SendForwardDone("r1", {43});
    SendForwardDone("r2", {142});
    SendFinish("r1");
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// Swa eviction state is tracked independently per request, not batch-wide.
TEST_F(MixedBatchSuite, PerRequestSwaHoleAtDifferentDecodeDepths) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    PlanOnce();  // both prefill together (mixed batch)
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});

    // r1 goes well past the window (W=4 = 2 pages); r2 advances once, staying inside it.
    std::optional<ExecutionPlan> last;
    int t1 = 43, t2 = 143;
    for (int step = 0; step < 5; ++step) {
        last = PlanOnce();
        ASSERT_NE(FindForwardBatch(*last), nullptr);
        SendForwardDone("r1", {t1++});
        if (step == 0) {
            SendForwardDone("r2", {t2++});  // r2 advances only once
        }
    }
    const ForwardBatch* op = FindForwardBatch(*last);
    ASSERT_NE(op, nullptr);

    // Row order within the op is not guaranteed.
    const auto& ids = op->request_ids;
    auto row_of = [&](const std::string& id) -> std::size_t {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == id) return i;
        }
        ADD_FAILURE() << "request " << id << " not in op";
        return 0;
    };

    // r2 may or may not remain in the batch; assert only on rows present.
    const auto& swa = op->block_tables.at("swa");
    const auto& full = op->block_tables.at("full");
    if (std::find(ids.begin(), ids.end(), "r1") != ids.end()) {
        std::size_t r1 = row_of("r1");
        EXPECT_NE(std::find(swa.at(r1).begin(), swa.at(r1).end(), 0), swa.at(r1).end())
            << "r1 drove past the window -> swa row must have a null hole";
        for (std::int32_t id : full.at(r1)) {
            EXPECT_GT(id, 0) << "r1 full-history row must stay hole-free";
        }
    }

    SendFinish("r1");
    if (scheduler_->DecodingSize() > 0) SendFinish("r2");
    PlanOnce();
}

// ---------------------------------------------------------------------------
// prefix_granularity = 1: the batch path is not hard-wired to prefix_granularity=2.
// ---------------------------------------------------------------------------
class PrefixGranularityOneSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 1;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                      /*sliding_window_tokens=*/2),
        };
        return cfg;
    }
};

TEST_F(PrefixGranularityOneSuite, TokenGranularPagesSlideAndReclaim) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* pop = FindForwardBatch(prefill);
    ASSERT_NE(pop, nullptr);
    EXPECT_EQ(pop->block_tables.at("full").at(0).size(), 4u) << "three prompt pages plus one preallocated decode page";

    SendForwardDone("r1", {42});

    std::optional<ExecutionPlan> last;
    int tok = 43;
    for (int i = 0; i < 4; ++i) {
        last = PlanOnce();
        ASSERT_NE(FindForwardBatch(*last), nullptr);
        SendForwardDone("r1", {tok++});
    }
    const ForwardBatch* op = FindForwardBatch(*last);
    ASSERT_NE(op, nullptr);
    for (std::int32_t id : op->block_tables.at("full").at(0)) {
        EXPECT_GT(id, 0) << "full group hole-free at prefix_granularity=1";
    }
    const auto& swa = op->block_tables.at("swa").at(0);
    EXPECT_NE(std::find(swa.begin(), swa.end(), 0), swa.end())
        << "swa group must develop a null hole at prefix_granularity=1 too";

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

namespace {

void SendAbort(Scheduler& scheduler, const std::string& id) {
    ExecutionEvent event;
    event.With(forward::Abort{.request_id = id});
    scheduler.Advance(std::move(event));
}

}  // namespace

// ---------------------------------------------------------------------------
// Pool-exhaustion admission. The first-chunk gate charges prompt + decode
// reserve = groups * ceil((tokens + 1) / prefix_granularity) blocks.
// ---------------------------------------------------------------------------
class TinyPoolSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        // 11 physical pages -> 10 usable (page 0 is the null placeholder):
        // one 4-page prompt over 2 groups (8 prefill + 2 reserve) = the pool.
        cfg.device_allocator.total_pages = 11;
        cfg.host_allocator.total_pages = 11;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                      /*sliding_window_tokens=*/4),
        };
        return cfg;
    }
};

TEST_F(TinyPoolSuite, ExhaustedPoolDefersSecondRequestUntilFirstFinishes) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    ASSERT_EQ(free_at_start, 10);

    // r1 exact admission acquires 8 prefill + 2 reserve blocks: free 0.
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    ExecutionPlan plan1 = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan1), nullptr);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 0);

    // r2 needs 4 blocks while r1 owns the whole pool: deferred.
    Submit(MakeRequestSpec("r2", /*num_pages=*/1, /*start=*/101));
    SendForwardDone("r1", {99});
    ExecutionPlan blocked = PlanOnce();
    const ForwardBatch* blocked_op = FindForwardBatch(blocked);
    ASSERT_NE(blocked_op, nullptr);
    ASSERT_EQ(blocked_op->request_ids.size(), 1u) << "only r1's reserved decode step fits this round";
    EXPECT_EQ(blocked_op->request_ids.at(0), "r1");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "deferred r2 stays intact in the waiting set";
    // Finalize consumes the reservation and exposes two slid-out SWA parents.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 2);

    SendForwardDone("r1", {100});
    SendFinish("r1");
    ExecutionPlan plan2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(plan2);
    ASSERT_NE(op2, nullptr) << "deferred request must be schedulable after pages free up";
    ASSERT_EQ(op2->request_ids.size(), 1u);
    EXPECT_EQ(op2->request_ids.at(0), "r2");
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    SendForwardDone("r2", {142});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start)
        << "pool back to baseline after the deferred request completes";
}

// ---------------------------------------------------------------------------
// Prefill-slide admission: a long chunked prompt fits ONLY because the gate
// credits the slide the chunk itself performs (BlocksFreedByAdvance).
// ---------------------------------------------------------------------------
class PrefillSlideAdmissionSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 13;
        cfg.host_allocator.total_pages = 14;  // 13 usable + the null placeholder (page 0)
        cfg.max_scheduled_tokens = 4;         // 4-token prefill chunks
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                      /*sliding_window_tokens=*/4),
        };
        return cfg;
    }
};

TEST_F(PrefillSlideAdmissionSuite, LongPromptAdmittedOnlyBecausePrefillSlides) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    ASSERT_EQ(free_at_start, 12);

    // page=2, W=4, 4-token chunks: c1 charges 4 blocks (2/group), 12 -> 8;
    // c2 (slide credit 0) charges 4, acquires 4 -> free 4.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    ExecutionPlan c1 = PlanOnce();
    ASSERT_NE(FindForwardBatch(c1), nullptr);
    ASSERT_EQ(FindForwardBatch(c1)->request_ids.size(), 1u);
    ExecutionPlan c2 = PlanOnce();
    ASSERT_NE(FindForwardBatch(c2), nullptr);
    ASSERT_EQ(FindForwardBatch(c2)->request_ids.size(), 1u);
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 4);

    // c3 gate: chunk + reserve = 3 blocks/group = 6 vs raw free 4; the pending
    // slide at N=8 frees the 2 swa pages below token 5 -> 4 + 2 = 6, admitted.
    ExecutionPlan c3 = PlanOnce();
    const ForwardBatch* c3op = FindForwardBatch(c3);
    ASSERT_NE(c3op, nullptr);
    ASSERT_EQ(c3op->request_ids.size(), 1u) << "final chunk must be admitted via the prefill slide credit";
    // Op balance: punch 2, acquire 2/group plus 1 reserved/group -> exact fit.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 0);

    // Decode transition: gate needs 2, finalize-slide credit at N=12 gives 2.
    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    ASSERT_EQ(FindForwardBatch(decode)->request_ids.size(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 2);

    SendForwardDone("r1", {100});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

TEST_F(PrefillSlideAdmissionSuite, InFlightStoreDoesNotDeferAdmission) {
    // Sink ON over the LongPromptAdmittedOnlyBecausePrefillSlides math: device
    // 13 -> 12 usable, c1+c2 charge 8, c3 needs 6 = free 4 + slide credit 2.
    // A store ticket pins no device source (the forward thread's stream orders the
    // copy before any reuse), so c3 admits WHILE op1 is still in flight.
    config_.disable_l2_cache = false;
    config_.host_allocator.total_pages = 13;
    scheduler_ = std::make_unique<Scheduler>(config_);
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    ASSERT_EQ(free_at_start, 12);

    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    ExecutionPlan c1 = PlanOnce();
    ASSERT_NE(FindForwardBatch(c1), nullptr);
    ASSERT_EQ(FindForwardBatch(c1)->request_ids.size(), 1u);

    ExecutionPlan c2 = PlanOnce();  // registers pages 0,1 both groups: streaming op1
    ASSERT_NE(FindForwardBatch(c2), nullptr);
    ASSERT_EQ(FindForwardBatch(c2)->request_ids.size(), 1u);
    auto wb1 = ExtractCacheOpsOfKind<WriteBackBatch>(c2);
    ASSERT_EQ(wb1.size(), 1u);
    const auto op1 = std::get<WriteBackBatch>(wb1.front());
    ASSERT_EQ(op1.op_ids.size(), 1u);
    EXPECT_EQ(op1.src_pages.at(0).size(), 4u) << "the first completed Full+SWA pages stream together";
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 4);

    ExecutionPlan c3 = PlanOnce();  // slide credit 2: admitted with op1 in flight; emits op2
    ASSERT_NE(FindForwardBatch(c3), nullptr);
    ASSERT_EQ(FindForwardBatch(c3)->request_ids.size(), 1u)
        << "an in-flight store must not defer admission: its sources are not pinned";
    auto wb2 = ExtractCacheOpsOfKind<WriteBackBatch>(c3);
    ASSERT_EQ(wb2.size(), 1u);
    const auto op2 = std::get<WriteBackBatch>(wb2.front());
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 0);

    SendWriteBackDone(op1.op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 4);

    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();  // last prefill pages stream on PrefillDone
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    ASSERT_EQ(FindForwardBatch(decode)->request_ids.size(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    auto wb3 = ExtractCacheOpsOfKind<WriteBackBatch>(decode);
    ASSERT_EQ(wb3.size(), 1u);
    const auto op3 = std::get<WriteBackBatch>(wb3.front());
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 2)
        << "the finalize slide's pages return at once: store tickets hold no device pins";

    SendForwardDone("r1", {100});
    SendFinish("r1");
    PlanOnce();  // reap: no device pins, the pool balances immediately
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
    SendWriteBackDone(op2.op_ids.at(0));
    SendWriteBackDone(op3.op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 12);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// Pool 17 -> 16 usable: swa at full prompt length would need 10+10+2 = 22
// (infeasible); the plateau ceil((chunk+W-1)/P) = ceil(7/2) = 4 keeps the peak
// at full 10 + swa 4 + reserve 2 = 16 (exact fit) -- the batch-swa-alloc contract.
class PrefillPlateauSuite : public PrefillSlideAdmissionSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = PrefillSlideAdmissionSuite::MakeConfig();
        cfg.device_allocator.total_pages = 17;
        cfg.host_allocator.total_pages = 17;
        return cfg;
    }
};

TEST_F(PrefillPlateauSuite, SwaWorkingSetPlateausWhileFullGrowsToPromptLength) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    ASSERT_EQ(free_at_start, 16);

    Submit(MakeRequestSpec("r1", /*num_pages=*/10));  // 20 tokens, 5 chunks of 4
    std::size_t swa_peak = 0;
    std::size_t full_last = 0;
    for (std::int32_t chunk = 0; chunk < 5; ++chunk) {
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(plan);
        ASSERT_NE(op, nullptr) << "chunk " << chunk;
        ASSERT_EQ(op->request_ids.size(), 1u) << "chunk " << chunk << " must be admitted";
        const std::size_t swa_real = RealPages(op->block_tables.at("swa")).size();
        const std::size_t full_real = RealPages(op->block_tables.at("full")).size();
        EXPECT_LE(swa_real, 5u) << "swa exceeded its four-page window plus decode reserve at chunk " << chunk;
        EXPECT_GE(full_real, full_last) << "full group must grow monotonically, chunk " << chunk;
        swa_peak = std::max(swa_peak, swa_real);
        full_last = full_real;
    }
    EXPECT_EQ(swa_peak, 5u) << "the four-page window plus decode reserve must be reached";
    EXPECT_EQ(full_last, 11u) << "ten prompt pages plus one preallocated decode page";

    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);

    SendForwardDone("r1", {100});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Capacity blocking: once no work or result is in flight, the scheduler
// immediately retracts the largest running request.
// ---------------------------------------------------------------------------
class CapacityBlockSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        // 13 physical pages -> 12 usable: two 2-page prompts charge
        // 2*ceil(5/2) = 6 blocks each at admission = exactly the pool.
        cfg.device_allocator.total_pages = 13;
        cfg.host_allocator.total_pages = 14;  // 13 usable + the null placeholder (page 0)
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full_a", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("full_b", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
        };
        return cfg;
    }
};

TEST_F(CapacityBlockSuite, RetractsLargestRunningRequestImmediately) {
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 12);

    // Round 1: both exact admissions include physical decode reservations.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* op1 = FindForwardBatch(prefill);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 2u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 0);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});

    // Round 2: both decode transitions consume their 2-block reservations.
    ExecutionPlan round2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(round2);
    ASSERT_NE(op2, nullptr);
    ASSERT_EQ(op2->request_ids.size(), 2u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 0);
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});

    // Round 3: both next steps fit their tail pages (0 fresh blocks).
    ExecutionPlan round3 = PlanOnce();
    const ForwardBatch* op3 = FindForwardBatch(round3);
    ASSERT_NE(op3, nullptr);
    ASSERT_EQ(op3->request_ids.size(), 2u);

    // A blocked round with r2's decode result STILL IN FLIGHT must stay quiet.
    SendForwardDone("r1", {44});
    ExecutionPlan quiet = PlanOnce();
    const ForwardBatch* quiet_op = FindForwardBatch(quiet);
    ASSERT_NE(quiet_op, nullptr);
    EXPECT_TRUE(quiet_op->request_ids.empty());

    // Nothing is in flight now, so this blocked round retracts immediately.
    SendForwardDone("r2", {144});
    ExecutionPlan retract_round = PlanOnce();
    const ForwardBatch* retract_op = FindForwardBatch(retract_round);
    ASSERT_NE(retract_op, nullptr);
    EXPECT_TRUE(retract_op->request_ids.empty());
    // r1 and r2 tie at 7 tokens; deterministic candidate order picks r1.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 6) << "r1's 3 pages x 2 groups return to the pool";
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "r1 requeues as a fresh prefill";
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);

    // The remaining request can decode on the released pages.
    ExecutionPlan resumed = PlanOnce();
    const ForwardBatch* resumed_op = FindForwardBatch(resumed);
    ASSERT_NE(resumed_op, nullptr);
    ASSERT_EQ(resumed_op->request_ids.size(), 1u);
    EXPECT_EQ(resumed_op->request_ids.at(0), "r2");
    SendForwardDone("r2", {145});
    SendFinish("r2");

    // With r2 reaped, r1 re-admits: its prefill covers prompt + generated.
    ExecutionPlan readmit = PlanOnce();
    const ForwardBatch* readmit_op = FindForwardBatch(readmit);
    ASSERT_NE(readmit_op, nullptr);
    ASSERT_EQ(readmit_op->request_ids.size(), 1u);
    EXPECT_EQ(readmit_op->request_ids.at(0), "r1");
    // Rebased to prompt + generated; whatever prefix survived is matched
    // back and only the rest is re-computed.
    EXPECT_EQ(readmit_op->prefill_lengths.at(0), 7);
    EXPECT_EQ(readmit_op->extend_prefix_lens.at(0) + readmit_op->input_lengths.at(0), 7);
    EXPECT_EQ(readmit_op->prefill_lengths.at(0), 7);

    SendForwardDone("r1", {45});
    PlanOnce();  // decode transition
    SendForwardDone("r1", {46});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 12);
}

class MambaFusedRetractionDrainSuite : public MambaSparsePrefillSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = MambaSparsePrefillSuite::MakeConfig();
        cfg.device_allocator.total_pages = 9;
        cfg.host_allocator.total_pages = 9;
        cfg.max_scheduled_tokens = 64;
        for (auto& group : cfg.cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }
};

TEST_F(MambaFusedRetractionDrainSuite, RetractionFreesCapacityWithoutPausingTheEngine) {
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 8);

    Submit(MakeRequestSpec("a", /*num_pages=*/2));
    Submit(MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101));
    Submit(MakeRequestSpec("c", /*num_pages=*/2, /*start=*/201));

    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* prefill_op = FindForwardBatch(prefill);
    ASSERT_NE(prefill_op, nullptr);
    ASSERT_EQ(prefill_op->request_ids, (std::vector<std::string>{"a", "b"}));
    SendForwardDone("a", {42});
    SendForwardDone("b", {142});

    // The blocked round retracts a victim and grants the freed capacity to
    // the blocked request within the same plan: no free page ever waits for
    // whoever asks first next round.
    ExecutionPlan retract_round = PlanOnce();
    const ForwardBatch* retract_op = FindForwardBatch(retract_round);
    ASSERT_NE(retract_op, nullptr);
    EXPECT_FALSE(retract_op->request_ids.empty()) << "the freed capacity is put to work in the same round";
    ASSERT_EQ(scheduler_->WaitingSize(), 1u) << "only the victim waits";
}

class FusedRetractionL2TestSuite : public CapacityBlockSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = CapacityBlockSuite::MakeConfig();
        cfg.disable_l2_cache = false;
        cfg.disable_prefix_cache = false;
        return cfg;
    }

    void CompleteStores(const ExecutionPlan& plan) {
        for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
            for (std::uint32_t op_id : std::get<WriteBackBatch>(operation).op_ids) {
                SendWriteBackDone(op_id);
            }
        }
    }
};

TEST_F(FusedRetractionL2TestSuite, RetractionStoresTheLatestCompletedBoundary) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    CompleteStores(prefill);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});

    ExecutionPlan first_decode = PlanOnce();
    CompleteStores(first_decode);
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});

    ExecutionPlan second_decode = PlanOnce();
    CompleteStores(second_decode);
    SendForwardDone("r1", {44});
    PlanOnce();  // r2 still has a forward result in flight, so retraction waits.
    SendForwardDone("r2", {144});

    const ExecutionPlan retraction = PlanOnce();
    EXPECT_FALSE(ExtractCacheOpsOfKind<WriteBackBatch>(retraction).empty())
        << "fused retraction must store the completed boundary before releasing request ownership";
}

TEST_F(FusedRetractionL2TestSuite, AReadmissionThatDoesNotFitWaitsWithoutRetracting) {
    // Drive r1 into Retracted with an L2 snapshot while r2 keeps decoding.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    CompleteStores(prefill);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    CompleteStores(PlanOnce());
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});
    CompleteStores(PlanOnce());
    SendForwardDone("r1", {44});
    SendForwardDone("r2", {144});
    const ExecutionPlan retraction = PlanOnce();  // retracts r1; the grant serves r2's blocked decode
    CompleteStores(retraction);
    ASSERT_EQ(scheduler_->WaitingSize(), 1u);
    const auto decoding_before = scheduler_->DecodingSize();

    // r1's readmission cannot fit while r2 holds the pool. It must WAIT --
    // retracting r2 to readmit r1 would be pure thrash -- and r2's decodes
    // keep running unstalled.
    for (std::int32_t token = 145; token < 149; ++token) {
        const ExecutionPlan round = PlanOnce();
        EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "the readmission waits; nothing new is retracted";
        EXPECT_EQ(scheduler_->DecodingSize(), decoding_before)
            << "resident decodes are not sacrificed for a readmission";
        const ForwardBatch* op = FindForwardBatch(round);
        ASSERT_NE(op, nullptr);
        if (!op->request_ids.empty()) {
            SendForwardDone("r2", {token});
        }
        CompleteStores(round);
    }
}

// ---------------------------------------------------------------------------
// Cache retract: a blocked round picks the largest Decoding/PrefillDone
// request, releases every page and requeues it as a fresh prefill. Accepted
// request lengths must obey Scheduler::MaxSingleRequestTokens().
// ---------------------------------------------------------------------------
class RetractSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        // 15 physical pages -> 14 usable: "a" (3-page prompt) charges
        // 2*ceil(7/2) = 8 and "b" (2-page prompt) 2*ceil(5/2) = 6 = the pool.
        cfg.device_allocator.total_pages = 15;
        cfg.host_allocator.total_pages = 16;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full_a", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("full_b", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
        };
        return cfg;
    }

    // Drives "a" (6-token prompt) and "b" (4-token prompt) into the exact-fit
    // capacity block that retracts "a".
    // Post: "a" Submitted with 9 tokens, "b" Decoding with 7 tokens, free = 8.
    void DriveToRetractOfA() {
        ASSERT_EQ(scheduler_->PoolFreeBlocks(), 14);
        Submit(MakeRequestSpec("a", /*num_pages=*/3));
        Submit(MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101));

        ExecutionPlan prefill = PlanOnce();
        const ForwardBatch* prefill_op = FindForwardBatch(prefill);
        ASSERT_NE(prefill_op, nullptr);
        ASSERT_EQ(prefill_op->request_ids.size(), 2u);
        ASSERT_EQ(scheduler_->PoolFreeBlocks(), 0);
        SendForwardDone("a", {42});
        SendForwardDone("b", {142});

        // Both decode transitions consume their reservations: free 0.
        ExecutionPlan decode = PlanOnce();
        const ForwardBatch* decode_op = FindForwardBatch(decode);
        ASSERT_NE(decode_op, nullptr);
        ASSERT_EQ(decode_op->request_ids.size(), 2u);
        ASSERT_EQ(scheduler_->PoolFreeBlocks(), 0);
        SendForwardDone("a", {43});   // 8 tokens = a's capacity
        SendForwardDone("b", {143});  // 6 tokens = b's capacity

        // Both next steps still fit their tail pages (0 fresh blocks).
        ExecutionPlan tail_round = PlanOnce();
        const ForwardBatch* tail_op = FindForwardBatch(tail_round);
        ASSERT_NE(tail_op, nullptr);
        ASSERT_EQ(tail_op->request_ids.size(), 2u);
        SendForwardDone("a", {44});   // 9 tokens: past capacity
        SendForwardDone("b", {144});  // 7 tokens: past capacity

        // The first fully blocked round retracts "a" (9 tokens > b's 7).
        ExecutionPlan retract_round = PlanOnce();
        const ForwardBatch* retract_op = FindForwardBatch(retract_round);
        ASSERT_NE(retract_op, nullptr);
        ASSERT_TRUE(retract_op->request_ids.empty());
        ASSERT_EQ(scheduler_->PoolFreeBlocks(), 8) << "a's 4 pages x 2 groups return to the pool";
        ASSERT_EQ(scheduler_->WaitingSize(), 1u) << "a requeues as a fresh prefill";
        ASSERT_EQ(scheduler_->DecodingSize(), 1u);
    }
};

TEST_F(RetractSuite, DecodingRequestReleasesPagesAndRequeues) {
    DriveToRetractOfA();

    // The survivor proceeds on the freed pages while a waits for re-admission.
    ExecutionPlan resumed = PlanOnce();
    const ForwardBatch* resumed_op = FindForwardBatch(resumed);
    ASSERT_NE(resumed_op, nullptr);
    ASSERT_EQ(resumed_op->request_ids.size(), 1u);
    EXPECT_EQ(resumed_op->request_ids.at(0), "b");
    SendForwardDone("b", {145});
    SendFinish("b");

    // b reaped -> a re-admits with its FULL length and completes.
    ExecutionPlan readmit = PlanOnce();
    const ForwardBatch* readmit_op = FindForwardBatch(readmit);
    ASSERT_NE(readmit_op, nullptr);
    ASSERT_EQ(readmit_op->request_ids.size(), 1u);
    EXPECT_EQ(readmit_op->request_ids.at(0), "a");
    // The readmission matches back whatever prefix survived and re-computes
    // the rest; together they cover the rebased prompt+generated length.
    EXPECT_EQ(readmit_op->extend_prefix_lens.at(0) + readmit_op->input_lengths.at(0),
              readmit_op->prefill_lengths.at(0));
    EXPECT_GT(readmit_op->input_lengths.at(0), 0);
    SendForwardDone("a", {45});
    PlanOnce();  // decode transition
    SendForwardDone("a", {46});
    SendFinish("a");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 14) << "pool balances after the full retract cycle";
}

TEST_F(RetractSuite, RetractedRequestPrefillCoversOldTokens) {
    DriveToRetractOfA();
    EXPECT_EQ(scheduler_->RequestTokenSize("a"), 9);

    // Free the running request so the retracted request re-admits immediately.
    SendFinish("b");
    ExecutionPlan readmit = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(readmit);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    ASSERT_EQ(op->request_ids.at(0), "a");
    EXPECT_EQ(op->prefill_lengths.at(0), 9) << "PrefillSize rebased to the full token count";
    // Whatever prefix survived is matched back; the rest is re-computed.
    EXPECT_EQ(op->extend_prefix_lens.at(0) + op->input_lengths.at(0), 9)
        << "RebasePrefill: the new prefill covers prompt + generated";
}

// Chunked re-admission after a retract: with max_scheduled_tokens = 4 the
// the retracted request's 9-token rebased prefill (prompt + generated) takes
// three chunks. EVERY chunk reports back; an intermediate one carries no
// token (the FSM stays Prefilling), and the op exposes the rebased
// prefill_lengths so the runtime can tell which kind it is.
class RetractChunkedReadmitSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        cfg.max_scheduled_tokens = 4;
        return cfg;
    }

    // Chunked-prefill twin of DriveToRetractOfA: same capacity block and retract of "a"
    // (9 tokens > b's 7), but "a"'s 6-token prompt prefills in two chunks.
    // Post: "a" requeued with 9 rebased tokens, "b" finished, pool fully free.
    void DriveToRetractOfAChunkedAndFreePool() {
        ASSERT_EQ(scheduler_->PoolFreeBlocks(), 14);
        Submit(MakeRequestSpec("a", /*num_pages=*/3));
        Submit(MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101));

        // Chunk 1 of "a" (4 of 6 prompt tokens) exhausts the round's budget;
        // an intermediate chunk reports back with no token.
        ExecutionPlan p1 = PlanOnce();
        const ForwardBatch* op1 = FindForwardBatch(p1);
        ASSERT_NE(op1, nullptr);
        ASSERT_EQ(op1->request_ids.size(), 1u);
        ASSERT_EQ(op1->request_ids.at(0), "a");
        ASSERT_LT(op1->input_lengths.at(0), op1->prefill_lengths.at(0)) << "mid-chunk";
        SendForwardDone("a");

        // Chunk 2 completes "a" (owes a result); leftover budget starts "b".
        ExecutionPlan p2 = PlanOnce();
        const ForwardBatch* op2 = FindForwardBatch(p2);
        ASSERT_NE(op2, nullptr);
        ASSERT_EQ(op2->request_ids.size(), 2u);
        SendForwardDone("a", {42});  // 7 tokens
        SendForwardDone("b");        // b's first chunk is intermediate: KV only

        // "b"'s completing chunk; "a" (PrefillDone) waits behind the prefill.
        ExecutionPlan p3 = PlanOnce();
        const ForwardBatch* op3 = FindForwardBatch(p3);
        ASSERT_NE(op3, nullptr);
        ASSERT_EQ(op3->request_ids.size(), 1u);
        ASSERT_EQ(op3->request_ids.at(0), "b");
        SendForwardDone("b", {142});  // 5 tokens

        // Both decode transitions consume their reservations: free 0.
        ExecutionPlan p4 = PlanOnce();
        const ForwardBatch* op4 = FindForwardBatch(p4);
        ASSERT_NE(op4, nullptr);
        ASSERT_EQ(op4->request_ids.size(), 2u);
        ASSERT_EQ(scheduler_->PoolFreeBlocks(), 0);
        SendForwardDone("a", {43});   // 8 tokens = a's capacity
        SendForwardDone("b", {143});  // 6 tokens = b's capacity

        // Tail-page decodes (0 fresh blocks).
        ExecutionPlan p5 = PlanOnce();
        const ForwardBatch* op5 = FindForwardBatch(p5);
        ASSERT_NE(op5, nullptr);
        ASSERT_EQ(op5->request_ids.size(), 2u);
        SendForwardDone("a", {44});   // 9 tokens: past capacity
        SendForwardDone("b", {144});  // 7 tokens: past capacity

        // The first fully blocked round retracts "a" (9 tokens > b's 7).
        ExecutionPlan retract_round = PlanOnce();
        ASSERT_TRUE(FindForwardBatch(retract_round)->request_ids.empty());
        ASSERT_EQ(scheduler_->PoolFreeBlocks(), 8);
        ASSERT_EQ(scheduler_->WaitingSize(), 1u);
        ASSERT_EQ(scheduler_->RequestTokenSize("a"), 9);

        // Free the survivor so a re-admits alone.
        SendFinish("b");
        ASSERT_EQ(scheduler_->PoolFreeBlocks(), 14);
    }
};

TEST_F(RetractChunkedReadmitSuite, MidChunkReadmitOwesNoToken) {
    DriveToRetractOfAChunkedAndFreePool();

    // The readmission exposes the REBASED prompt+generated length, and the
    // runtime decides "does this op produce a token" from the op alone:
    // prefix + input < prefill means mid-chunk, so the result is empty.
    ExecutionPlan readmit = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(readmit);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    ASSERT_EQ(op->request_ids.at(0), "a");
    EXPECT_EQ(op->prefill_lengths.at(0), 9) << "rebased prompt+generated length exposed on the op";
    EXPECT_GT(op->input_lengths.at(0), 0);
    EXPECT_LE(op->extend_prefix_lens.at(0) + op->input_lengths.at(0), op->prefill_lengths.at(0));
}

// Regression pin for the crash: a forward-done ExtendResult for a mid-chunk
// re-prefill slot must land (it clears the in-flight count that protects the
// chunk's pages), but it must not smuggle a token into a prompt that is
// still being prefilled.
TEST_F(RetractChunkedReadmitSuite, MidChunkReadmitTakesEmptyResultOnly) {
    DriveToRetractOfAChunkedAndFreePool();

    ExecutionPlan readmit = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(readmit);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.at(0), "a");
    if (op->extend_prefix_lens.at(0) + op->input_lengths.at(0) < op->prefill_lengths.at(0)) {
        EXPECT_THROW(SendForwardDone("a", {45}), std::logic_error)
            << "the FSM is still Prefilling; a mid-chunk result carries no token";
        EXPECT_NO_THROW(SendForwardDone("a"));
    } else {
        // This readmission completed in one round, so a token IS owed.
        EXPECT_NO_THROW(SendForwardDone("a", {45}));
    }
}

TEST_F(RetractChunkedReadmitSuite, ChunkedReadmitCompletes) {
    DriveToRetractOfAChunkedAndFreePool();

    // Drive the readmission to completion, whatever number of chunks it
    // takes: every chunk reports, only the completing one carries a token.
    for (int round = 0; round < 8; ++round) {
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(plan);
        ASSERT_NE(op, nullptr);
        ASSERT_EQ(op->request_ids.size(), 1u);
        ASSERT_EQ(op->request_ids.at(0), "a");
        if (op->extend_prefix_lens.at(0) + op->input_lengths.at(0) == op->prefill_lengths.at(0)) {
            SendForwardDone("a", {45});
            SUCCEED();
            return;
        }
        SendForwardDone("a");
    }
    FAIL() << "the readmission never reached its completing chunk";
}

class PrefillHeadOfLineSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        // 19 physical pages -> 18 usable. "holder" owns 6 parents and the
        // first chunk of "active" owns 8, leaving just enough for "queued".
        cfg.device_allocator.total_pages = 19;
        cfg.host_allocator.total_pages = 20;
        cfg.max_scheduled_tokens = 8;
        for (auto& group : cfg.cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }
};

TEST_F(PrefillHeadOfLineSuite, RetractingAnIncompletePrefillPublishesOnlyComputedPages) {
    // The victim is a prefill that has been through some chunks but not all.
    // Only those chunks may reach the prefix cache: publishing the whole
    // prompt would hand later requests pages that were never computed.
    Submit(MakeRequestSpec("done", /*num_pages=*/2));
    ExecutionPlan p1 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p1)->request_ids, std::vector<std::string>{"done"});
    SendForwardDone("done", {42});

    Submit(MakeRequestSpec("half", /*num_pages=*/8, /*start=*/101));
    ExecutionPlan p2 = PlanOnce();
    const ForwardBatch* chunk = FindForwardBatch(p2);
    ASSERT_EQ(chunk->request_ids, std::vector<std::string>{"half"});
    const std::int32_t computed = chunk->extend_prefix_lens.at(0) + chunk->input_lengths.at(0);
    ASSERT_LT(computed, chunk->prefill_lengths.at(0)) << "the victim is mid-prefill";
    SendForwardDone("half");  // the chunk landed, so the victim is retractable

    // Force the retraction, then re-submit the SAME prompt: it may only match
    // back the pages the victim actually computed.
    Submit(MakeRequestSpec("waiting", /*num_pages=*/1, /*start=*/201));
    PlanOnce();
    ASSERT_GT(scheduler_->WaitingSize(), 1u) << "the incomplete prefill gave way";

    for (int round = 0; round < 6; ++round) {
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(plan);
        if (op == nullptr) {
            continue;
        }
        const auto row = std::ranges::find(op->request_ids, "half");
        if (row == op->request_ids.end()) {
            continue;
        }
        const auto index = static_cast<std::size_t>(std::distance(op->request_ids.begin(), row));
        EXPECT_LE(op->extend_prefix_lens.at(index), computed)
            << "the readmission cannot match back pages the victim never computed";
        return;
    }
}

TEST(ExtendResultEvent, AwaitingResultAbsorbsEmptyIntermediateResults) {
    // Under the PP chunk pipeline, older intermediate chunk results (empty
    // by contract) can land AFTER the final chunk was scheduled, i.e. while
    // the request already sits in PrefillAwaitingResult. Only the final
    // chunk's result carries a token; an empty arrival must keep waiting,
    // or the handoff batch goes out before the bootstrap token is real.
    BlockPool pool(/*num_lcm_blocks=*/8);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0, fsm::CacheProgress{},
                                                      /*load_pairs=*/{},
                                                      /*awaits_result=*/true});
    ASSERT_TRUE(request.Is<fsm::PrefillAwaitingResult>());

    request.Apply(fsm::ExtendResultEvent{{}});  // an older intermediate chunk's empty result
    EXPECT_TRUE(request.Is<fsm::PrefillAwaitingResult>()) << "an empty result must not end the wait";

    request.Apply(fsm::ExtendResultEvent{{42}});  // the final chunk's token
    EXPECT_TRUE(request.Is<fsm::PrefillDone>());
    EXPECT_EQ(request.LastToken(), 42);
}

TEST(RetractEvent, StampsResumePriorityFromGeneratedOutput) {
    // The readmission order is derived off the Retracted states: a victim
    // with generated output a client is reading resumes ahead of one that
    // had produced nothing, whatever their retraction epochs say
    // (nextReadmission reads resumes_generation first, then the epoch).
    BlockPool pool(/*num_lcm_blocks=*/8);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/2,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0, fsm::CacheProgress{},
                                                      /*load_pairs=*/{}});
    ASSERT_TRUE(request.Is<fsm::Prefilling>());

    request.Apply(
        fsm::RetractEvent{&coordinator, /*epoch=*/1, /*has_recoverable_snapshot=*/true, request.HasGeneratedOutput()});
    const auto* retracted = request.GetIf<fsm::Retracted>();
    ASSERT_NE(retracted, nullptr);
    EXPECT_FALSE(retracted->ResumesGeneration()) << "no output yet: it resumes behind decode-origin victims";
}

TEST(RetractionHeadroom, EscalatesPerRetractionAndStopsAtTheGenerationBudget) {
    // Every admission secures one safe-step window of decode headroom up
    // front; each retraction raises the bar -- the previous admission was
    // still too optimistic -- until it reaches the budget the request could
    // ever use.
    RequestSpec spec;
    spec.request_id = "r";
    spec.tokens = {1, 2, 3, 4};
    spec.max_new_tokens = 6000;
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};

    constexpr std::int32_t kSafeSteps = 4096;
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 4096) << "a fresh request secures one window";

    request.NoteRetracted();
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 6000)
        << "capped by the generation budget, so the escalation terminates";

    request.NoteRetracted();
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 6000) << "and stays there";
}

TEST(RetractionHeadroom, ReservesOnlyTheRemainingGenerationBudget) {
    // Retraction rebases generated tokens into the prompt. A readmission
    // that still reserved the full declared budget on top of them would
    // demand prompt + generated + max_new -- more than the request can ever
    // write, and near the single-request limit more than the pool holds,
    // leaving it Retracted forever.
    BlockPool pool(/*num_lcm_blocks=*/16);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    spec.max_new_tokens = 6000;
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0, fsm::CacheProgress{},
                                                      /*load_pairs=*/{}});
    request.Apply(fsm::ExtendResultEvent{{42}});
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, request.CacheProgress()});
    request.Apply(fsm::ExtendResultEvent{std::vector<std::int32_t>(999, 7)});
    ASSERT_EQ(request.GeneratedTokens(), 1000);

    constexpr std::int32_t kSafeSteps = 4096;
    EXPECT_EQ(request.RemainingNewTokens(), 5000);
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 4096) << "one window, not yet the remaining budget";
    request.NoteRetracted();
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 5000)
        << "capped by the REMAINING budget: the generated 1000 are part of the rebased prompt now";
    EXPECT_TRUE(request.ReserveCoversGeneration(kSafeSteps));

    request.Apply(
        fsm::RetractEvent{&coordinator, /*epoch=*/1, /*has_recoverable_snapshot=*/true, request.HasGeneratedOutput()});
    EXPECT_EQ(request.PrefillSize(), 1004) << "rebase folded prompt + generated into the prefill window";
    EXPECT_EQ(request.RemainingNewTokens(), 5000) << "rebasing does not change the remaining budget";
}

TEST(RetractionHeadroom, AnUndeclaredGenerationBudgetDemandsNone) {
    // max_new_tokens == 0 is the opt-out: with no declared budget there is
    // nothing to prepay, and no cap either -- an escalation without a cap
    // could demand more than the pool holds and make the request
    // unreadmittable, which is worse than admitting it optimistically. So it
    // stays optimistic and relies on the victim policy for progress.
    RequestSpec spec;
    spec.request_id = "r";
    spec.tokens = {1, 2, 3, 4};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};

    constexpr std::int32_t kSafeSteps = 4096;
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 0);
    request.NoteRetracted();
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 0) << "even after a retraction";
}

TEST_F(RetractSuite, AFreshAdmissionPrepaysItsGenerationBudget) {
    // Without declared budgets "a" (6 tokens) and "b" (4 tokens) pack into
    // one round: 2*ceil(7/2) + 2*ceil(5/2) = 14 = the pool. A 5-token budget
    // on "a" is prepaid at admission -- 2*ceil((6+5)/2) = 12 -- so "b"
    // (needing 6) no longer fits beside it and waits.
    RequestSpec a = MakeRequestSpec("a", /*num_pages=*/3);
    a.max_new_tokens = 5;
    Submit(a);
    Submit(MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101));

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->request_ids, std::vector<std::string>{"a"});
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 2);
}

class AdmissionHeadroomPrefillRoleSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        cfg.role = Role::kP;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.transfer_policy = CacheTransferPolicy::FullSuffix;
        }
        return cfg;
    }

    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{request_id});
        scheduler_->Advance(std::move(event));
    }
};

TEST_F(AdmissionHeadroomPrefillRoleSuite, ThePrefillRoleDoesNotPrepayDecodeHeadroom) {
    // The P role never decodes locally and never retracts, so a declared
    // generation budget -- even one far beyond this pool -- charges nothing
    // at admission: the prompt's own 2*ceil(6/2) = 6 blocks and no more.
    RequestSpec heavy = MakeRequestSpec("heavy", /*num_pages=*/3);
    heavy.max_new_tokens = 6000;
    Submit(heavy);
    SendBootstrapped("heavy");

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->request_ids, std::vector<std::string>{"heavy"});
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 8);
}

TEST_F(PrefillHeadOfLineSuite, AnIncompletePrefillGivesWayBeforeACompletedOne) {
    // Two retraction candidates: "done" finished its prefill and owns a
    // sampled token a client is waiting on; "half" has produced nothing. The
    // one with no committed output gives way, even though it is larger.
    Submit(MakeRequestSpec("done", /*num_pages=*/2));
    ExecutionPlan p1 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p1)->request_ids, std::vector<std::string>{"done"});
    SendForwardDone("done", {42});

    Submit(MakeRequestSpec("half", /*num_pages=*/8, /*start=*/101));
    ExecutionPlan p2 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p2)->request_ids, std::vector<std::string>{"half"});
    SendForwardDone("half");  // its chunk landed; the pages are no longer in use

    // "half" cannot get its next chunk, so capacity must be freed.
    Submit(MakeRequestSpec("waiting", /*num_pages=*/1, /*start=*/201));
    PlanOnce();

    EXPECT_EQ(scheduler_->DecodingSize(), 1u) << "the completed request kept its pages and took its first decode step";
    EXPECT_EQ(scheduler_->WaitingSize(), 2u) << "the incomplete prefill was the victim";
}

TEST_F(PrefillHeadOfLineSuite, AnIncompletePrefillIsNotRetractedWhileItsChunkIsInFlight) {
    // Regression pin: an incomplete prefill is the preferred victim, but a
    // chunk forward writes KV into its pages. Retract it before that write
    // lands and the result hits pages another request now owns -- which the
    // attention backend sees as a batch whose metadata does not match.
    Submit(MakeRequestSpec("done", /*num_pages=*/2));
    ExecutionPlan p1 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p1)->request_ids, std::vector<std::string>{"done"});
    SendForwardDone("done", {42});

    Submit(MakeRequestSpec("half", /*num_pages=*/8, /*start=*/101));
    ExecutionPlan p2 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p2)->request_ids, std::vector<std::string>{"half"});
    // Deliberately do NOT report the chunk: it is still on the GPU.

    Submit(MakeRequestSpec("waiting", /*num_pages=*/1, /*start=*/201));
    const std::int32_t waiting_before = static_cast<std::int32_t>(scheduler_->WaitingSize());
    PlanOnce();
    EXPECT_EQ(static_cast<std::int32_t>(scheduler_->WaitingSize()), waiting_before)
        << "nothing may be retracted while a forward is out against its pages";

    // Once the chunk lands, the same round retracts as before.
    SendForwardDone("half");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 1u) << "the completed request kept its pages";
    EXPECT_EQ(scheduler_->WaitingSize(), 2u) << "the incomplete prefill was the victim";
}

TEST_F(PrefillHeadOfLineSuite, BlockedLaterChunkDoesNotStartSubmittedRequest) {
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 18);

    Submit(MakeRequestSpec("holder", /*num_pages=*/2));
    ExecutionPlan holder_prefill = PlanOnce();
    ASSERT_EQ(FindForwardBatch(holder_prefill)->request_ids, std::vector<std::string>{"holder"});
    SendForwardDone("holder", {42});

    Submit(MakeRequestSpec("active", /*num_pages=*/8, /*start=*/101));
    ExecutionPlan first_chunk = PlanOnce();
    ASSERT_EQ(FindForwardBatch(first_chunk)->request_ids, std::vector<std::string>{"active"});
    ASSERT_EQ(FindForwardBatch(first_chunk)->input_lengths.at(0), 8);
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 4);
    SendForwardDone("active");  // the chunk landed; its pages are retractable

    Submit(MakeRequestSpec("queued", /*num_pages=*/1, /*start=*/201));
    ExecutionPlan blocked = PlanOnce();
    // The stalled resident prefill seals new-prompt admission ("queued"
    // must not strand it further), but the completed "holder" still takes
    // its decode step: decodes are never hostage to a stalled prefill.
    ASSERT_EQ(FindForwardBatch(blocked)->request_ids, std::vector<std::string>{"holder"})
        << "a blocked active prefill must prevent lower-priority admission";
    // The stalled prefill is the victim, not the completed "holder": it has
    // produced no output a client is reading, so it gives way and retries
    // once the capacity is there.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 12) << "the incomplete prefill released its pages";
    EXPECT_EQ(scheduler_->WaitingSize(), 2u) << "the retracted prefill and the untouched submitted request wait";
    EXPECT_EQ(scheduler_->DecodingSize(), 1u) << "the completed request keeps its pages";
}

// Exact-fit re-admission after a retract: the whole freed budget (pages AND any
// stale decode reserve) must be spendable by the next request.
class RetractExactFitSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        // 9 physical pages -> 8 usable: one 3-page prompt charges exactly the pool.
        cfg.device_allocator.total_pages = 9;
        cfg.host_allocator.total_pages = 10;
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }
};

TEST_F(RetractExactFitSuite, ReportsSingleRequestTokenCapacity) {
    // Two full-history groups share eight usable parents. Each group needs
    // ceil(tokens / 2) parents, so one request can address eight tokens.
    EXPECT_EQ(scheduler_->MaxSingleRequestTokens(), 8);
}

TEST_F(RetractExactFitSuite, ReportsCapacityUsingEachGroupsBlockGranularity) {
    SchedulerConfig config = MakeConfig();
    config.cache_groups[1].rows_per_page = 1;
    Scheduler scheduler{std::move(config)};

    // Eight parents fit ceil(tokens / 2) pages for the first group and one
    // page per token for the second group. Five tokens use 3 + 5 parents.
    EXPECT_EQ(scheduler.MaxSingleRequestTokens(), 5);
}

TEST_F(RetractExactFitSuite, IncludesOverlapDecodeReserveInTokenCapacity) {
    SchedulerConfig config = MakeConfig();
    config.overlap_schedule_depth = 1;
    Scheduler scheduler{std::move(config)};
    // The extra decode token shares the fourth page with token seven. Counting
    // it as a separately rounded page would incorrectly report only six.
    EXPECT_EQ(scheduler.MaxSingleRequestTokens(), 7);
}

TEST(PdSlidingCapacityTest, CountsPrefixIslandPhasePageAndGroupPacking) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = 4;
    cfg.device_allocator.total_pages = 3;  // null + two usable LCM parents
    cfg.host_allocator.total_pages = 0;
    cfg.max_scheduled_tokens = 16;
    cfg.max_batch_size = 1;
    cfg.decode_input_tokens = 1;
    cfg.role = Role::kD;
    cfg.disable_l2_cache = true;

    CacheGroupConfig sliding;
    sliding.group_id = "sliding";
    sliding.rows_per_page = 2;  // group q=2 while scheduler P=4
    sliding.entry_stride_tokens = 1;
    sliding.total_pages = 5;  // null + two parents packing two children each
    sliding.cache_blocks_per_lcm_block = 2;
    sliding.retention = CacheGroupConfig::Retention::SlidingWindow;
    sliding.sliding_window_tokens = 4;
    sliding.family = CacheGroupFamily::State;
    sliding.transfer_policy = CacheTransferPolicy::FullSuffix;
    cfg.cache_groups = {sliding};

    Scheduler scheduler{std::move(cfg)};

    // W=4, q=2 and a one-token decode reserve can require two cached
    // lookback pages plus a three-page phase-shifted remote tail. The dense
    // cap makes eight the exact maximum with four available child pages.
    EXPECT_EQ(scheduler.MaxSingleRequestTokens(), 8);
}

TEST_F(RetractExactFitSuite, ReserveRefundBalances) {
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 8);
    Submit(MakeRequestSpec("a", /*num_pages=*/3));  // charge 2*ceil(7/2) = 8: exact fit
    ExecutionPlan prefill = PlanOnce();
    ASSERT_EQ(FindForwardBatch(prefill)->request_ids.size(), 1u);
    SendForwardDone("a", {42});
    PlanOnce();  // decode transition consumes the reserve: free 0
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 0);
    SendForwardDone("a", {43});  // 8 tokens = capacity
    PlanOnce();                  // tail-page decode (0 fresh blocks)
    SendForwardDone("a", {44});  // 9 tokens: past capacity

    ExecutionPlan retract_round = PlanOnce();  // first blocked round retracts "a"
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 8);
    ASSERT_EQ(scheduler_->WaitingSize(), 1u);

    // "d" needs EXACTLY the released capacity: a leaked reservation from a
    // would make this admission fail. (The round right after a retraction
    // withholds new prompts, so this is the round after that one.)
    Submit(MakeRequestSpec("d", /*num_pages=*/3, /*start=*/201));
    PlanOnce();
    ExecutionPlan admitted = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(admitted);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    EXPECT_EQ(op->request_ids.at(0), "d") << "exact-fit admission proves the full budget was refunded";

    SendForwardDone("d", {99});
    PlanOnce();
    SendForwardDone("d", {100});
    SendFinish("d");
    SendAbort(*scheduler_, "a");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 8);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
}

// Two capacity-block cycles on one pool: each cycle retracts a different
// largest running request; the smallest request rides both frees to completion.
class RetractTrioSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        // 25 physical pages -> 24 usable: r1 charges 10, r2 8, r3 6 = the pool.
        cfg.device_allocator.total_pages = 25;
        cfg.host_allocator.total_pages = 26;
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }
};

TEST_F(RetractTrioSuite, TwoCapacityBlocksRetractDifferentRequests) {
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 24);
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    Submit(MakeRequestSpec("r3", /*num_pages=*/2, /*start=*/201));

    ExecutionPlan prefill = PlanOnce();
    ASSERT_EQ(FindForwardBatch(prefill)->request_ids.size(), 3u);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    SendForwardDone("r3", {242});

    ExecutionPlan decode = PlanOnce();  // all three consume their reserves
    ASSERT_EQ(FindForwardBatch(decode)->request_ids.size(), 3u);
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 0);
    SendForwardDone("r1", {43});   // 10 = capacity
    SendForwardDone("r2", {143});  // 8 = capacity
    SendForwardDone("r3", {243});  // 6 = capacity
    PlanOnce();                    // tail-page decodes (0 fresh blocks)
    SendForwardDone("r1", {44});   // 11: past capacity
    SendForwardDone("r2", {144});  // 9: past capacity
    SendForwardDone("r3", {244});  // 7: past capacity

    // Cycle 1 immediately retracts r1 (11 tokens, the largest).
    ExecutionPlan first_retract = PlanOnce();
    ASSERT_TRUE(FindForwardBatch(first_retract)->request_ids.empty());
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 10) << "r1's 5 pages x 2 groups return";
    ASSERT_EQ(scheduler_->WaitingSize(), 1u);

    // r2 and r3 ride the freed pages until capacity blocks again with r2 the
    // largest running request (r1's 12-block re-admission charge never fits meanwhile).
    ExecutionPlan p6 = PlanOnce();  // both acquire a page pair: free 6
    ASSERT_EQ(FindForwardBatch(p6)->request_ids.size(), 2u);
    SendForwardDone("r2", {145});
    SendForwardDone("r3", {245});
    PlanOnce();  // tail-page decodes
    SendForwardDone("r2", {146});
    SendForwardDone("r3", {246});
    PlanOnce();  // both acquire a page pair: free 2
    SendForwardDone("r2", {147});
    SendForwardDone("r3", {247});
    PlanOnce();  // tail-page decodes
    SendForwardDone("r2", {148});
    SendForwardDone("r3", {248});
    ExecutionPlan p10 = PlanOnce();  // r2 takes the last pair; r3 defers
    ASSERT_EQ(FindForwardBatch(p10)->request_ids.size(), 1u);
    ASSERT_EQ(FindForwardBatch(p10)->request_ids.at(0), "r2");
    SendForwardDone("r2", {149});
    ExecutionPlan p11 = PlanOnce();  // r2 tail-page decode
    ASSERT_EQ(FindForwardBatch(p11)->request_ids.size(), 1u);
    SendForwardDone("r2", {150});  // 15 tokens: past capacity

    // Cycle 2 retracts a second, DIFFERENT request -- the cycles must not
    // keep picking the same victim -- and grants the freed capacity to the
    // blocked r1 readmission within the same round.
    ExecutionPlan second_retract = PlanOnce();
    const ForwardBatch* second_op = FindForwardBatch(second_retract);
    ASSERT_NE(second_op, nullptr);
    EXPECT_EQ(second_op->request_ids, std::vector<std::string>{"r1"})
        << "the freed capacity reaches the blocked readmission in the same round";
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "only the fresh victim waits";
    SendForwardDone("r1");  // the granted chunk's KV lands; r1 is abortable

    // Third proceeds: drop the two waiting requests and let r3 finish.
    SendAbort(*scheduler_, "r1");
    SendAbort(*scheduler_, "r2");
    ExecutionPlan survivor = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(survivor);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    EXPECT_EQ(op->request_ids.at(0), "r3");
    SendForwardDone("r3", {249});
    SendFinish("r3");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 24);
}

// A retracted request whose config carries a mamba-style state group
// (family=State, FullHistory retention) must release state pages too.
class RetractStateGroupSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        cfg.device_allocator.total_pages = 9;  // 8 usable
        cfg.host_allocator.total_pages = 10;
        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("state", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::State),
        };
        return cfg;
    }
};

TEST_F(RetractStateGroupSuite, StateGroupRequestRetractsCleanly) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    Submit(MakeRequestSpec("a", /*num_pages=*/2));
    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* prefill_op = FindForwardBatch(prefill);
    ASSERT_NE(prefill_op, nullptr);
    ASSERT_EQ(prefill_op->request_ids.size(), 1u) << "the prompt must admit into the state-group config";
    ASSERT_EQ(prefill_op->block_tables.count("state"), 1u);
    SendForwardDone("a", {1000});

    // The lone grower decodes until capacity blocks, then retracts immediately.
    std::int32_t tok = 1001;
    bool retracted = false;
    for (int round = 0; round < 64 && !retracted; ++round) {
        ExecutionPlan p = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(p);
        ASSERT_NE(op, nullptr);
        if (!op->request_ids.empty()) {
            SendForwardDone("a", {tok++});
        } else if (scheduler_->WaitingSize() == 1u) {
            retracted = true;
        }
    }
    ASSERT_TRUE(retracted) << "the lone grower must exhaust capacity and retract";
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start)
        << "retract must return full-history AND state pages to the pool";
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);

    SendAbort(*scheduler_, "a");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

TEST(CacheProgressTest, PromotionBoundarySurvivesPrefillRounds) {
    BlockPool pool(/*num_lcm_blocks=*/8);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.kind = AttnKind::kMambaState,
                       .sliding_window = 0,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/6, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> admission =
        AdmitForTest(coordinator, tables, /*num_tokens=*/4);
    ASSERT_TRUE(admission);

    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/0, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0,
                                                      fsm::CacheProgress{
                                                          .access_epoch = admission->access_epoch,
                                                          .promotion_boundary_tokens = 8,
                                                      },
                                                      /*load_pairs=*/{}});
    ASSERT_TRUE(request.Is<fsm::Prefilling>());
    EXPECT_EQ(request.CacheProgress().promotion_boundary_tokens, 8);

    request.Apply(fsm::SchedulePrefillEvent{
        /*tokens_this_round=*/4,
        /*reserve_num_tokens_in_next_schedule_event=*/1,
        request.CacheProgress(),
    });
    ASSERT_TRUE(request.Is<fsm::Prefilling>());
    EXPECT_EQ(request.CacheProgress().promotion_boundary_tokens, 8);
}

class PromotionBoundaryHeadOfLineSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 8;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = false;
        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("state", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::State),
        };
        return cfg;
    }
};

TEST_F(PromotionBoundaryHeadOfLineSuite, DoesNotStartSecondIncompletePrefill) {
    RequestSpec seed = MakeRequestSpec("seed", /*num_pages=*/6);
    Submit(seed);
    PlanOnce();  // chunk boundary at token 8
    PlanOnce();  // endpoint at token 12
    SendForwardDone("seed", {42});
    SendFinish("seed");
    PlanOnce();
    ASSERT_EQ(scheduler_->WaitingSize(), 0u);
    ASSERT_EQ(scheduler_->DecodingSize(), 0u);

    RequestSpec first = seed;
    first.request_id = "first";
    for (std::size_t i = 6; i < first.tokens.size(); ++i) {
        first.tokens[i] += 1000;
    }
    RequestSpec second = MakeRequestSpec("second", /*num_pages=*/6, /*start=*/2001);
    Submit({first, second});
    ASSERT_EQ(scheduler_->WaitingSize(), 2u);

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"first"});
    EXPECT_EQ(batch->input_lengths, std::vector<std::int32_t>{6})
        << "the full-history hit promotes token 6 before the remaining prompt";
}

TEST(CacheProgressTest, RemotePrefillPreservesDecodeReserve) {
    BlockPool pool(/*num_lcm_blocks=*/8);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kD};
    request.Apply(fsm::BootstrappedEvent{});
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> admission =
        AdmitForTest(coordinator, tables, GroupDemand{.num_tokens = 4, .reserve_tokens = 3});
    ASSERT_TRUE(admission);

    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/3, &req_pool,
                                                      fsm::PrefillSource::kRemote, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0,
                                                      fsm::CacheProgress{.access_epoch = admission->access_epoch},
                                                      /*load_pairs=*/{}});
    ASSERT_TRUE(request.Is<fsm::RemotePrefilling>());

    request.Apply(fsm::RemotePrefillDoneEvent{/*token=*/42});

    ASSERT_TRUE(request.Is<fsm::PrefillDone>());
    EXPECT_EQ(request.ReserveNumTokensInNextScheduleEvent(), 3);
}

TEST(RetractionStateFsmTest, RetractionTransitionsImmediatelyAndRebasesPrefill) {
    BlockPool device_pool(/*num_lcm_blocks=*/12);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool);
    ReqPoolAllocator req_pool{4};
    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kD};
    request.Apply(fsm::BootstrappedEvent{});
    std::vector<BlockTable> tables(coordinator.NumGroups());
    auto admission = AdmitForTest(coordinator, tables, GroupDemand{.num_tokens = 4, .reserve_tokens = 1});
    ASSERT_TRUE(admission);
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        /*tokens_this_round=*/4,
        /*reserve_num_tokens_in_next_schedule_event=*/1,
        &req_pool,
        fsm::PrefillSource::kRemote,
        &coordinator,
        std::move(tables),
        /*hit_tokens=*/0,
        fsm::CacheProgress{.access_epoch = admission->access_epoch},
        /*load_pairs=*/{},
    });
    request.Apply(fsm::RemotePrefillDoneEvent{/*token=*/42});
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, request.CacheProgress()});
    ASSERT_TRUE(request.Is<fsm::Decoding>());

    request.Apply(
        fsm::RetractEvent{&coordinator, /*epoch=*/1, /*has_recoverable_snapshot=*/true, request.HasGeneratedOutput()});

    ASSERT_TRUE(request.Is<fsm::Retracted>());
    EXPECT_EQ(request.PrefillSize(), request.TokenSize());
    EXPECT_EQ(device_pool.NumEmptyLcmBlocks(), device_pool.NumLcmBlocks());

    std::vector<BlockTable> recovery_tables(coordinator.NumGroups());
    auto recovery_admission = AdmitForTest(coordinator, recovery_tables,
                                           GroupDemand{.num_tokens = request.PrefillSize(), .reserve_tokens = 1});
    ASSERT_TRUE(recovery_admission);
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        request.PrefillSize(),
        /*reserve_num_tokens_in_next_schedule_event=*/1,
        &req_pool,
        fsm::PrefillSource::kLocal,
        &coordinator,
        std::move(recovery_tables),
        /*hit_tokens=*/0,
        fsm::CacheProgress{.access_epoch = recovery_admission->access_epoch},
        /*load_pairs=*/{},
    });
    EXPECT_TRUE(request.Is<fsm::PrefillDone>());
}

// Drive the FSM directly to pin the PrefillDone retract overload.
TEST(RetractEvent, PrefillDoneVictimReleasesPagesAndRequeues) {
    BlockPool pool(/*num_lcm_blocks=*/8);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> admission =
        AdmitForTest(coordinator, tables, /*num_tokens=*/4);
    ASSERT_TRUE(admission);

    // Whole 4-token prompt in one chunk -> PrefillDone: holds pages, no decode yet.
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0,
                                                      fsm::CacheProgress{.access_epoch = admission->access_epoch},
                                                      /*load_pairs=*/{}});
    ASSERT_TRUE(request.Is<fsm::PrefillDone>());
    EXPECT_EQ(request.CacheProgress().access_epoch, admission->access_epoch);
    ASSERT_LT(pool.NumEmptyLcmBlocks(), 8);

    // The last chunk's ExtendResult lands while still PrefillDone.
    request.Apply(fsm::ExtendResultEvent{{42}});

    request.Apply(
        fsm::RetractEvent{&coordinator, /*epoch=*/1, /*has_recoverable_snapshot=*/false, request.HasGeneratedOutput()});
    // Retracted, not Submitted: "was retracted" and "never ran" are
    // different situations even when there is no snapshot to recover.
    EXPECT_TRUE(request.Is<fsm::Retracted>());
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8) << "the retract must release every page";
    EXPECT_EQ(request.TokenSize(), 5);
    EXPECT_EQ(request.PrefillSize(), 5) << "prompt + generated rebase into the prefill window";
}

// ---------------------------------------------------------------------------
// Abort-mid-flight pool balance: abort mid-chunked-prefill or mid-decode must
// return every page to the pool.
// ---------------------------------------------------------------------------
TEST_F(ChunkedPrefillSuite, AbortMidPrefillRestoresPoolBaseline) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    // 12 tokens (6 pages), max_scheduled_tokens=4 -> abort lands mid-prefill.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    PlanOnce();  // chunk 1
    PlanOnce();  // chunk 2 -> still Prefilling
    EXPECT_LT(scheduler_->PoolFreeBlocks(), free_at_start);

    SendAbort(*scheduler_, "r1");
    PlanOnce();  // reap the aborted request
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start)
        << "abort mid-prefill must return every page (both groups) to the pool";
}

TEST_F(ChunkedPrefillSuite, AbortDuringDecodeRestoresPoolBaseline) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();  // single-chunk prefill (4 tokens)
    SendForwardDone("r1", {42});
    PlanOnce();  // decode step
    SendForwardDone("r1", {43});
    EXPECT_LT(scheduler_->PoolFreeBlocks(), free_at_start);

    SendAbort(*scheduler_, "r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "abort during decode must return every page to the pool";
}

// Admission owns the prepared refs before the event. If the independent request
// slot allocation fails, event destruction releases those refs through RAII.
TEST(EventFailurePath, ReqPoolExhaustionAtFirstChunkLeavesPoolBalanced) {
    BlockPool pool(/*num_lcm_blocks=*/31);  // Pages are not the constraint.
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
    ReqPoolAllocator req_pool{1};
    ReqPoolIndex held = req_pool.Allocate();  // exhaust the single slot
    ASSERT_EQ(req_pool.AvailableSlots(), 0);

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 27);

    EXPECT_THROW(
        request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                          /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                          fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                          /*hit_tokens=*/0,
                                                          /*cache_progress=*/{},
                                                          /*load_pairs=*/{}}),
        std::runtime_error);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 31) << "a failed req-pool Allocate must not leak block-pool pages";

    EXPECT_NO_THROW(request.Apply(fsm::AbortEvent{&coordinator}));
    EXPECT_TRUE(request.Is<fsm::Finished>());
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 31);
}

// ---------------------------------------------------------------------------
// SWA off-by-one regression: decode admission slides at the number of tokens
// computed before the pending query.
// ---------------------------------------------------------------------------
TEST(SwaWindowBoundary, DecodeStepKeepsOldestInWindowPageAtPageBoundary) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .num_tokens = 1,
                                 .num_computed_tokens = 4,
                             }));

    const auto swa_slot_null = [&](std::int32_t i) { return !tables[1].Blocks()[i]; };
    ASSERT_EQ(tables[1].NumBlocks(), 3);
    EXPECT_FALSE(swa_slot_null(0));

    // N=5; keys [2,5] -> page 0 out: slot 0 punched, slot 1 kept.
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .num_tokens = 1,
                                 .num_computed_tokens = 5,
                             }));
    EXPECT_TRUE(swa_slot_null(0));
    EXPECT_FALSE(swa_slot_null(1));

    // N=6; keys [3,6]: key 3 still lives in page 1, so slot 1 survives.
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .num_tokens = 1,
                                 .num_computed_tokens = 6,
                             }));
    EXPECT_FALSE(swa_slot_null(1)) << "key 3 of the pending query lives in page 1; freeing it is the off-by-one";
    EXPECT_TRUE(swa_slot_null(0));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before - 2);

    // N=7; keys [4,7] -> page 1 fully out, punched exactly now.
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .num_tokens = 1,
                                 .num_computed_tokens = 7,
                             }));
    EXPECT_TRUE(swa_slot_null(1));
    EXPECT_FALSE(swa_slot_null(2));

    for (const CacheBlockRef& block : tables[0].Blocks()) {
        EXPECT_TRUE(block);
    }
    FreeRequest(coordinator, tables);
}

// ---------------------------------------------------------------------------
// Physically-backed decode reservations cannot be stolen before consumption.
// ---------------------------------------------------------------------------
class PhysicalReserveSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 11;
        cfg.host_allocator.total_pages = 11;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full_a", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("full_b", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
        };
        return cfg;
    }
};

TEST_F(PhysicalReserveSuite, LaterRequestCannotStealReservedDecodeHeadroom) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    ASSERT_EQ(free_at_start, 10);

    // a: exact admission acquires 6 prefill blocks plus 2 decode-reserve
    // blocks, leaving 2. b needs 4 and must defer.
    Submit(MakeRequestSpec("a", /*num_pages=*/3));
    Submit(MakeRequestSpec("b", /*num_pages=*/1, /*start=*/101));
    ExecutionPlan round1 = PlanOnce();
    const ForwardBatch* op1 = FindForwardBatch(round1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 1u) << "b must not be admitted into a's reserved decode blocks";
    EXPECT_EQ(op1->request_ids.at(0), "a");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 2);

    // a's decode transition consumes its already-owned reservation.
    SendForwardDone("a", {99});
    ExecutionPlan round2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(round2);
    ASSERT_NE(op2, nullptr);
    ASSERT_EQ(op2->request_ids.size(), 1u) << "a's decode must proceed into its reserved pages";
    EXPECT_EQ(op2->request_ids.at(0), "a");
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 2);
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);

    SendForwardDone("a", {100});
    SendFinish("a");
    ExecutionPlan round3 = PlanOnce();
    const ForwardBatch* op3 = FindForwardBatch(round3);
    ASSERT_NE(op3, nullptr);
    ASSERT_EQ(op3->request_ids.size(), 1u);
    EXPECT_EQ(op3->request_ids.at(0), "b");

    SendForwardDone("b", {142});
    SendFinish("b");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

TEST_F(PhysicalReserveSuite, AbortWithOutstandingReservationLeavesNoPhantom) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    ASSERT_EQ(free_at_start, 10);

    // a owns a 2-block physical decode reservation (see above).
    Submit(MakeRequestSpec("a", /*num_pages=*/3));
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 2);

    // Abort before the reserve is consumed: RAII must release it too.
    SendAbort(*scheduler_, "a");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);

    // b needs the whole pool: gate 2*ceil(9/2) = 10 <= 10 only without a phantom.
    Submit(MakeRequestSpec("b", /*num_pages=*/4, /*start=*/101));
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u) << "a leaked reservation would defer b forever";
    EXPECT_EQ(op->request_ids.at(0), "b");
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 0);

    SendForwardDone("b", {142});
    SendFinish("b");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Cross-request prefix hits, end to end: admission match -> FSM claim -> input
// window starts past the hit (disable_prefix_cache=false, W=32).
// ---------------------------------------------------------------------------
class PrefixHitSuite : public SchedulerTestSuite {
protected:
    virtual std::int32_t SlidingWindowTokens() const { return 32; }
    virtual bool DisablePrefixCache() const { return false; }
    virtual std::int32_t TotalPages() const { return 64; }

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = TotalPages();
        cfg.host_allocator.total_pages = TotalPages();
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = DisablePrefixCache();

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State, SlidingWindowTokens()),
        };
        return cfg;
    }

    RequestSpec MakeSpecWithTokens(const std::string& id, std::vector<std::int32_t> tokens) {
        return RequestSpec{.request_id = id, .tokens = std::move(tokens)};
    }

    // Prefill -> one decode round -> finish; returns the PREFILL op's per-group
    // rows. The decode round is load-bearing: the finalize registers the page
    // hashes, and finish frees the blocks WITH hashes intact (still matchable).
    std::map<std::string, std::vector<std::int32_t>> RunLifecycle(const RequestSpec& spec) {
        Submit(spec);
        ExecutionPlan prefill = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(prefill);
        EXPECT_NE(op, nullptr);
        std::map<std::string, std::vector<std::int32_t>> rows;
        if (op != nullptr) {
            for (const auto& [gid, table] : op->block_tables) {
                rows[gid] = table.at(0);
            }
        }
        SendForwardDone(spec.request_id, {9001});
        PlanOnce();  // PrefillDone -> Decoding: finalize registers the hashes
        SendForwardDone(spec.request_id, {9002});
        SendFinish(spec.request_id);
        PlanOnce();  // reap
        return rows;
    }

    static void ExpectRowPrefixEq(const std::vector<std::int32_t>& row,
                                  const std::vector<std::int32_t>& expected_prefix, const char* what) {
        ASSERT_GE(row.size(), expected_prefix.size()) << what;
        for (std::size_t i = 0; i < expected_prefix.size(); ++i) {
            EXPECT_EQ(row[i], expected_prefix[i]) << what << " slot " << i;
        }
    }
};

TEST_F(PrefixHitSuite, TwoRequestsSharePrefixReusePages) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "r1 must fully reclaim before r2 runs";
    ASSERT_EQ(r1_rows.at("full").size(), 5u);
    ASSERT_EQ(r1_rows.at("swa").size(), 5u);

    // r2: 12 tokens, first 8 == r1's. Hit: cap = (12-1)/2 = 5 pages; r1
    // registered 4, r2's page-4 hash chains off different tail tokens -> full
    // hits 4; swa (W=32, needed 16 > 4) keeps 4 -> fixpoint 4 blocks = 8 tokens.
    std::vector<std::int32_t> r2_tokens =
        MakeAlignedTokens(/*num_pages=*/4, PrefixGranularity());  // tokens 1..8 == r1's
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/4, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 4);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->prefill_lengths.at(0), 12);
    EXPECT_EQ(op->input_ids, tail);
    // Complete group rows contain 4 claimed + ceil(4/2) fresh + 1
    // preallocated decode page = 7.
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 7u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 7u);

    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    const std::vector<std::int32_t> swa_prefix(r1_rows.at("swa").begin(), r1_rows.at("swa").begin() + 4);
    ExpectRowPrefixEq(op->block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op->block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: 4 hit blocks/group remain active, 2 fresh blocks/group are
    // acquired, and 1 decode-reserve block/group is physically held.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 14);

    // Finalize registers pages 4..5 and consumes the existing reservation.
    SendForwardDone("r2", {199});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 14);

    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "pool back to baseline after r2 finishes";
}

TEST_F(PrefixHitSuite, FinishPublishesPagesFromLastForward) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    Submit(first);
    const ExecutionPlan first_plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(first_plan), nullptr);

    // Finish before another scheduling round can publish the prefill pages.
    SendForwardDone("r1", {9001});
    SendFinish("r1");
    PlanOnce();

    Submit(RequestSpec{.request_id = "r2", .tokens = first.tokens});
    const ExecutionPlan second_plan = PlanOnce();
    const ForwardBatch* second = FindForwardBatch(second_plan);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(second->request_ids.size(), 1u);
    EXPECT_EQ(second->input_lengths.at(0), 2);
    EXPECT_EQ(second->extend_prefix_lens.at(0), 6);
}

TEST_F(PrefixHitSuite, ClearL1CacheRemovesAnIdlePrefix) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    RunLifecycle(first);

    const auto [cleared, log] = ClearL1CacheWithCapturedLog(scheduler_.get());
    ASSERT_TRUE(cleared);
    EXPECT_NE(log.find("flush L1 cache completed"), std::string::npos);
    Submit(RequestSpec{.request_id = "r2", .tokens = first.tokens});
    const ExecutionPlan second_plan = PlanOnce();
    const ForwardBatch* second = FindForwardBatch(second_plan);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(second->request_ids.size(), 1u);
    EXPECT_EQ(second->input_lengths.at(0), 8);
    EXPECT_EQ(second->extend_prefix_lens.at(0), 0);
}

TEST_F(PrefixHitSuite, ClearL1CacheRejectsAnActiveRequestAndPreservesItsPrefix) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    Submit(first);
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    SendForwardDone("r1", {9001});
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);

    // The active request's pages are pinned, so the coordinator refuses the
    // clear and its prefix survives for the next request to match.
    const auto [cleared, log] = ClearL1CacheWithCapturedLog(scheduler_.get());
    EXPECT_FALSE(cleared);
    EXPECT_NE(log.find("cached blocks are still pinned"), std::string::npos);
    SendForwardDone("r1", {9002});
    SendFinish("r1");
    PlanOnce();

    Submit(RequestSpec{.request_id = "r2", .tokens = first.tokens});
    const ExecutionPlan second_plan = PlanOnce();
    const ForwardBatch* second = FindForwardBatch(second_plan);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(second->request_ids.size(), 1u);
    EXPECT_EQ(second->input_lengths.at(0), 2);
    EXPECT_EQ(second->extend_prefix_lens.at(0), 6);
}

// The hit is capped at (PrefillSize-1)/prefix_granularity pages so the last token is
// always recomputed to produce logits.
TEST_F(PrefixHitSuite, FullHitCapsAtLastToken) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    const RequestSpec r1 = MakeRequestSpec("r1", /*num_pages=*/4);  // 8 tokens
    RunLifecycle(r1);
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);

    // r2 = the same 8 tokens: cap = (8-1)/2 = 3 pages -> hit 3 = 6 tokens.
    Submit(MakeSpecWithTokens("r2", r1.tokens));
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 6);
    // input = tokens [6, 8) of the 1..8 sequence.
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/7));
    // 3 claimed + 1 fresh + 1 preallocated decode page per group.
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 5u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 5u);
    // Pool: 3 hit + 1 fresh + 1 reserved block per group.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 10);

    // Reserve: 1 fresh page per group (tail full).
    SendForwardDone("r2", {199});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 10);
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

class PrefixReplaySuite : public PrefixHitSuite {
protected:
    virtual std::int32_t PrefixReplayTokens() const { return 4; }

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = PrefixHitSuite::MakeConfig();
        cfg.prefix_replay_tokens = PrefixReplayTokens();
        return cfg;
    }
};

TEST_F(PrefixReplaySuite, FullHitReplaysPrivateTailPages) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);  // 8 tokens
    const auto first_rows = RunLifecycle(first);
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);

    Submit(MakeSpecWithTokens("r2", first.tokens));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    // Four replay tokens cap the hit at two 2-token pages. The remaining
    // prompt tail is ordinary prefill input, not a claimed shared page.
    EXPECT_EQ(op->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(op->input_lengths.at(0), 4);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/4, /*start=*/5));
    for (const char* group_id : {"full", "swa"}) {
        const auto& row = op->block_tables.at(group_id).at(0);
        ASSERT_EQ(row.size(), 5u);  // 2 hit + 2 private replay + 1 decode reserve
        EXPECT_EQ(row[0], first_rows.at(group_id)[0]);
        EXPECT_EQ(row[1], first_rows.at(group_id)[1]);
        EXPECT_NE(row[2], first_rows.at(group_id)[2])
            << group_id << " replay page must not alias the cached shared page";
        EXPECT_GT(row[2], 0);
        EXPECT_GT(row[3], 0);
        EXPECT_GT(row[4], 0);
    }
}

TEST_F(PrefixReplaySuite, ExistingUncachedSuffixSatisfiesReplayRequirement) {
    RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));  // tokens 1..8

    // Only four tokens match. The eight-token uncached suffix already exceeds
    // prefix_replay_tokens=4, so the ordinary partial hit stays unchanged.
    std::vector<std::int32_t> tokens = MakeTokens(/*count=*/4);
    const std::vector<std::int32_t> suffix = MakeTokens(/*count=*/8, /*start=*/801);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    Submit(MakeSpecWithTokens("r2", tokens));

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->input_ids, suffix);
}

TEST_F(PrefixReplaySuite, ReplayedPagesArePublishedForTheNextRequest) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    RunLifecycle(first);

    // r2 recomputes and republishes the tail behind the capped four-token hit.
    RunLifecycle(MakeSpecWithTokens("r2", first.tokens));

    // Extending the prompt lets r3 probe beyond its own four-token replay tail.
    // It must hit all eight tokens from r2, including r2's replayed pages.
    std::vector<std::int32_t> extended = first.tokens;
    const std::vector<std::int32_t> suffix = MakeTokens(/*count=*/4, /*start=*/801);
    extended.insert(extended.end(), suffix.begin(), suffix.end());
    Submit(MakeSpecWithTokens("r3", extended));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->input_lengths.at(0), 4);
    EXPECT_EQ(op->input_ids, suffix);
}

class PrefixReplayLargerThanPromptSuite : public PrefixReplaySuite {
protected:
    std::int32_t PrefixReplayTokens() const override { return 32; }
};

TEST_F(PrefixReplayLargerThanPromptSuite, FallsBackToFullPromptPrefill) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    RunLifecycle(first);

    Submit(MakeSpecWithTokens("r2", first.tokens));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->input_ids, first.tokens);
}

class PrefixReplayDisabledSuite : public PrefixReplaySuite {
protected:
    bool DisablePrefixCache() const override { return true; }
};

TEST_F(PrefixReplayDisabledSuite, DisabledPrefixCacheStillPrefillsTheFullPrompt) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    RunLifecycle(first);

    Submit(MakeSpecWithTokens("r2", first.tokens));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->input_ids, first.tokens);
}

class PrefixReplayHeterogeneousSuite : public PrefixHitSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 8;
        cfg.device_allocator.total_pages = 128;
        cfg.host_allocator.total_pages = 0;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = false;
        cfg.prefix_replay_tokens = 8;

        CacheGroupConfig history = MakeGroup("history", /*block_granularity=*/8, cfg.device_allocator.total_pages,
                                             CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History);
        CacheGroupConfig state = MakeGroup("state", /*block_granularity=*/2, cfg.device_allocator.total_pages,
                                           CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                                           /*sliding_window_tokens=*/32);
        state.cache_blocks_per_lcm_block = 4;
        cfg.cache_groups = {history, state};
        return cfg;
    }
};

TEST_F(PrefixReplayHeterogeneousSuite, ReplayTailIsPrivateAcrossPackedGroups) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);  // 32 tokens
    const auto first_rows = RunLifecycle(first);

    Submit(MakeSpecWithTokens("r2", first.tokens));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 24);
    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/8, /*start=*/25));

    const auto& history = op->block_tables.at("history").at(0);
    ASSERT_EQ(history.size(), 5u);  // 3 hit + 1 replay + 1 decode reserve
    EXPECT_EQ(history[0], first_rows.at("history")[0]);
    EXPECT_EQ(history[1], first_rows.at("history")[1]);
    EXPECT_EQ(history[2], first_rows.at("history")[2]);
    EXPECT_NE(history[3], first_rows.at("history")[3]);

    const auto& state = op->block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 17u);  // 12 hit + 4 replay + 1 decode reserve
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(state[i], first_rows.at("state")[i]);
    }
    for (std::size_t i = 12; i < 16; ++i) {
        EXPECT_NE(state[i], first_rows.at("state")[i]);
        EXPECT_GT(state[i], 0);
    }
    EXPECT_GT(state[16], 0);
}

TEST(PrefixReplayConfigTest, RejectsNegativeReplayTokens) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = 2;
    cfg.device_allocator.total_pages = 8;
    cfg.max_scheduled_tokens = 8;
    cfg.max_batch_size = 1;
    cfg.prefix_replay_tokens = -1;
    cfg.cache_groups = {
        MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                  CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
    };
    EXPECT_THROW((void)Scheduler(std::move(cfg)), std::invalid_argument);
}

class PrefixHitDisabledSuite : public PrefixHitSuite {
protected:
    bool DisablePrefixCache() const override { return true; }
};

TEST_F(PrefixHitDisabledSuite, DisablePrefixCacheSkipsMatch) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);

    std::vector<std::int32_t> r2_tokens = MakeAlignedTokens(/*num_pages=*/4, PrefixGranularity());
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/4, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 12) << "no hit -> the whole prompt is the input";
    EXPECT_EQ(op->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(op->input_ids, r2_tokens);
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 7u) << "six prompt pages plus one preallocated decode page";
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 7u);
    // Pool: 6 live pages plus one physically reserved decode page per group.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 14);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

TEST_F(PrefixHitSuite, PartialHit) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));  // tokens 1..8
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);

    // r2: 12 tokens, only the first 4 match r1 (pages 0..1); the hash chain
    // propagates the divergence to every later page. Hit = 2 pages = 4 tokens.
    std::vector<std::int32_t> r2_tokens = MakeTokens(/*count=*/4);  // 1..4 == r1's first 4
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/8, /*start=*/801);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(op->input_ids, tail);
    // 2 claimed + 4 fresh + 1 preallocated decode page per group.
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 7u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 7u);

    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 2);
    const std::vector<std::int32_t> swa_prefix(r1_rows.at("swa").begin(), r1_rows.at("swa").begin() + 2);
    ExpectRowPrefixEq(op->block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op->block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: 2 hit + 4 fresh + 1 reserved block per group.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 14);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// Small window: the SWA group's bounded right-to-left scan stops once its
// contiguous run is satisfied, claiming r1's punched slots as null holes.
class PrefixHitSmallWindowSuite : public PrefixHitSuite {
protected:
    std::int32_t SlidingWindowTokens() const override { return 4; }
};

TEST_F(PrefixHitSmallWindowSuite, SwaGroupHitRespectsWindow) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    // r1's finalize REGISTERS all 4 swa hashes BEFORE ReclaimExpired(8) punches
    // slots 0,1 -- punched blocks reach the free list with hashes, matchable.
    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
    ASSERT_EQ(r1_rows.at("swa").size(), 5u);

    // r2: 10 tokens, first 8 == r1's. Fixpoint (W=4, page=2, pages_needed
    // = ceil(3/2) = 2): cap = (10-1)/2 = 4, full matches 4; swa scan stops at
    // run 2 -> keep 4 with 2 holes -> common stays 4 = 8 hit tokens.
    std::vector<std::int32_t> r2_tokens = MakeAlignedTokens(/*num_pages=*/4, PrefixGranularity());  // 1..8 == r1's
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/2, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->input_ids, tail);
    // 4 claimed slots (real or hole) + 1 fresh + 1 preallocated decode page.
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 6u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 6u);

    const auto& full_row = op->block_tables.at("full").at(0);
    ASSERT_EQ(full_row.size(), 6u);
    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    ExpectRowPrefixEq(full_row, full_prefix, "full row");
    EXPECT_GT(full_row[4], 0);
    EXPECT_GT(full_row[5], 0);

    const auto& swa_row = op->block_tables.at("swa").at(0);
    ASSERT_EQ(swa_row.size(), 6u);
    EXPECT_EQ(swa_row[0], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[1], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[2], r1_rows.at("swa")[2]);
    EXPECT_EQ(swa_row[3], r1_rows.at("swa")[3]);
    EXPECT_GT(swa_row[4], 0);
    EXPECT_GT(swa_row[5], 0);
    // Window invariant (mirrors ExpectSwaWindowIntact): the last
    // pages_needed = 2 slots of the claimed prefix must be real.
    for (std::size_t i = 2; i < 4; ++i) {
        EXPECT_GT(swa_row[i], 0) << "null hole inside the last window of the claimed prefix at slot " << i;
    }

    // Pool: full claims 4 + swa claims 2 (holes claim nothing) + 1 fresh
    // page/group + one physically reserved decode page/group = 10.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 10);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// Near capacity, exact admission must protect prefix-hit parents while finding
// placements for the fresh suffix and decode reservation.
class PrefixHitTightPoolSuite : public PrefixHitSuite {
protected:
    std::int32_t TotalPages() const override { return 11; }
};

TEST_F(PrefixHitTightPoolSuite, ProtectedHitAndFreshDemandMustFitTogether) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    ASSERT_EQ(free_at_start, 10);

    // r1 leaves 4 cached, evictable parents plus 6 empty parents. The capacity
    // metric reports all 10 as available, but only the latter are unbound.
    RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/2));
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);

    // r3 holds 1 prefill page plus 1 decode-reserve page per group: 10 -> 6.
    Submit(MakeRequestSpec("r3", /*num_pages=*/1, /*start=*/501));
    ExecutionPlan r3_prefill = PlanOnce();
    ASSERT_NE(FindForwardBatch(r3_prefill), nullptr);
    ASSERT_EQ(FindForwardBatch(r3_prefill)->request_ids.size(), 1u);
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 6);

    // r3's finalize consumes its already physical decode reservation.
    SendForwardDone("r3", {599});
    PlanOnce();
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 6);

    // r2: 8 tokens, first 4 == r1's. The 4 cached hit parents are protected.
    // Its suffix and reserve need 6 empty parents, but r3 pins 4 and leaves only
    // 2 empty, so the whole admission defers without acquiring the hits.
    std::vector<std::int32_t> r2_tokens =
        MakeAlignedTokens(/*num_pages=*/2, PrefixGranularity());  // tokens 1..4 == r1's
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/4, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));
    ExecutionPlan blocked = PlanOnce();
    const ForwardBatch* blocked_op = FindForwardBatch(blocked);
    ASSERT_NE(blocked_op, nullptr);
    ASSERT_EQ(blocked_op->request_ids.size(), 1u) << "r2 must be deferred, not admitted into a short pool";
    EXPECT_EQ(blocked_op->request_ids.at(0), "r3");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "deferred r2 stays intact in the waiting set";
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 6) << "a deferred first chunk must not touch the pool";

    // r3 finishes -> 10 available parents. r2 protects 4 hit parents and
    // acquires 4 fresh plus 2 physically reserved parents: exact fit.
    SendForwardDone("r3", {600});
    SendFinish("r3");
    ExecutionPlan plan2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(plan2);
    ASSERT_NE(op2, nullptr) << "deferred request must be schedulable after r3 releases its pages";
    ASSERT_EQ(op2->request_ids.size(), 1u);
    EXPECT_EQ(op2->request_ids.at(0), "r2");
    EXPECT_EQ(op2->input_lengths.at(0), 4) << "only the 4-token remainder is computed";
    EXPECT_EQ(op2->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 0);

    // r2's finalize consumes the existing reservation, so capacity stays 0.
    SendForwardDone("r2", {699});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 0);

    SendForwardDone("r2", {700});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "pool back to baseline after both complete";
}

// ---------------------------------------------------------------------------
// M13 decode-block caching: pages filled DURING decode register via the hash
// chain (admission: register -> slide -> acquire), so a later turn hits PAST
// the previous prompt boundary. Fill timing: a round at container Size s has
// N = s - 1 computed and registers pages up to N/prefix_granularity -- a tail page
// registers one round late (finishing earlier frees its block hashless).
// ---------------------------------------------------------------------------
class DecodeCachingSuite : public PrefixHitSuite {
protected:
    // Deliver one sampled token and run the next schedule round, returning the
    // per-group rows the round's op carried. Single-request rounds only.
    std::map<std::string, std::vector<std::int32_t>> AdvanceOneRound(const std::string& id, std::int32_t token) {
        SendForwardDone(id, {token});
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(plan);
        EXPECT_NE(op, nullptr);
        std::map<std::string, std::vector<std::int32_t>> rows;
        if (op != nullptr) {
            for (const auto& [gid, table] : op->block_tables) {
                rows[gid] = table.at(0);
            }
        }
        return rows;
    }

    // Turn 1: prompt {1,2,3,4}, generated 101..105 (page=2). Finalize registers
    // prompt pages 0,1; +103 (N=6) registers page 2; +105 (N=8) registers page
    // 3 (tail one round late: 105 exists only to push N past 8). Returns the
    // last round's rows: 5 slots, the first 4 = the conversation's pages 0..3.
    std::map<std::string, std::vector<std::int32_t>> RunTurnOne() {
        Submit(MakeRequestSpec("r1", /*num_pages=*/2));
        ExecutionPlan prefill = PlanOnce();
        EXPECT_NE(FindForwardBatch(prefill), nullptr);
        AdvanceOneRound("r1", 101);
        AdvanceOneRound("r1", 102);
        AdvanceOneRound("r1", 103);
        AdvanceOneRound("r1", 104);
        auto rows = AdvanceOneRound("r1", 105);
        SendFinish("r1");
        PlanOnce();  // reap
        return rows;
    }

    // Turn-2 prompt: r1's 4 prompt tokens + first 4 generated + 2 new = 10;
    // pages 0..3 match r1's registration by content.
    std::vector<std::int32_t> MakeTurnTwoPrompt() {
        std::vector<std::int32_t> tokens =
            MakeAlignedTokens(/*num_pages=*/2, PrefixGranularity());                        // {1,2,3,4} == r1's prompt
        const std::vector<std::int32_t> response = MakeTokens(/*count=*/4, /*start=*/101);  // r1's generated 101..104
        tokens.insert(tokens.end(), response.begin(), response.end());
        const std::vector<std::int32_t> fresh = MakeTokens(/*count=*/2, /*start=*/901);
        tokens.insert(tokens.end(), fresh.begin(), fresh.end());
        return tokens;
    }

    // Turn-3 prompt: turn 2's full 13-token stream + 3 new tokens = 16.
    std::vector<std::int32_t> MakeTurnThreePrompt() {
        std::vector<std::int32_t> tokens = MakeTurnTwoPrompt();
        const std::vector<std::int32_t> r2_response = MakeTokens(/*count=*/3, /*start=*/201);
        tokens.insert(tokens.end(), r2_response.begin(), r2_response.end());
        const std::vector<std::int32_t> fresh = MakeTokens(/*count=*/3, /*start=*/951);
        tokens.insert(tokens.end(), fresh.begin(), fresh.end());
        return tokens;
    }
};

TEST_F(DecodeCachingSuite, DecodeFilledPageBecomesHittable) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    const auto r1_rows = RunTurnOne();
    ASSERT_EQ(r1_rows.at("full").size(), 5u);
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "r1 must fully reclaim before r2 runs";

    // Hit: cap = (10-1)/2 = 4 -> pages 0..3, all registered by r1 (RunTurnOne);
    // swa (W=32, needed 16 > 4) keeps 4 -> fixpoint 4 blocks = 8 hit tokens.
    Submit(MakeSpecWithTokens("r2", MakeTurnTwoPrompt()));
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->prefill_lengths.at(0), 10);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/901));
    // 4 claimed + 1 fresh + 1 preallocated decode page.
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 6u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 6u);

    // Slots 2,3 are the pages r1's decode filled, beyond its prompt boundary.
    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    const std::vector<std::int32_t> swa_prefix(r1_rows.at("swa").begin(), r1_rows.at("swa").begin() + 4);
    ExpectRowPrefixEq(op->block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op->block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: claim 4/group (8) + 1 fresh/group (2) + 1 reserved/group (2).
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 12);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "pool back to baseline after r2 finishes";
}

TEST_F(DecodeCachingSuite, MultiTurnConversationReusesResponsePages) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    RunTurnOne();  // registers conversation pages 0..3
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);

    // Turn 2: hit 4 pages, then decode 201..203: +201 finalize registers page
    // 4 = {901,902}; +203 (N=12) registers page 5 (tail one round late).
    Submit(MakeSpecWithTokens("r2", MakeTurnTwoPrompt()));
    ExecutionPlan turn2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(turn2);
    ASSERT_NE(op2, nullptr);
    EXPECT_EQ(op2->extend_prefix_lens.at(0), 8) << "turn 2 hits r1's prompt + response pages";
    AdvanceOneRound("r2", 201);
    AdvanceOneRound("r2", 202);
    const auto r2_rows = AdvanceOneRound("r2", 203);
    ASSERT_EQ(r2_rows.at("full").size(), 7u);  // ceil(13/2)
    SendFinish("r2");
    PlanOnce();  // reap
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);

    // Turn 3 hit: cap = (16-1)/2 = 7; pages 0..5 registered (0..3 by r1, 4..5
    // by r2), page 6 never full in any request -> fixpoint 6 blocks = 12 hit
    // tokens, into r2's response (page 5).
    Submit(MakeSpecWithTokens("r3", MakeTurnThreePrompt()));

    ExecutionPlan turn3 = PlanOnce();
    const ForwardBatch* op3 = FindForwardBatch(turn3);
    ASSERT_NE(op3, nullptr);
    ASSERT_EQ(op3->request_ids.size(), 1u);
    EXPECT_EQ(op3->extend_prefix_lens.at(0), 12) << "hit grows across turns: 8 -> 12 tokens";
    EXPECT_EQ(op3->input_lengths.at(0), 4);
    EXPECT_EQ(op3->prefill_lengths.at(0), 16);
    EXPECT_EQ(op3->input_ids, (std::vector<std::int32_t>{203, 951, 952, 953}));
    // 6 claimed + 2 fresh + 1 preallocated decode page.
    EXPECT_EQ(op3->block_tables.at("full").at(0).size(), 9u);
    EXPECT_EQ(op3->block_tables.at("swa").at(0).size(), 9u);

    // Slots 0..3 are r1's blocks (re-freed cached by r2), 4..5 r2's own pages.
    const std::vector<std::int32_t> full_prefix(r2_rows.at("full").begin(), r2_rows.at("full").begin() + 6);
    const std::vector<std::int32_t> swa_prefix(r2_rows.at("swa").begin(), r2_rows.at("swa").begin() + 6);
    ExpectRowPrefixEq(op3->block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op3->block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: 6 claimed/group (12) + 2 fresh/group (4) + 1 reserved/group (2).
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 18);

    SendForwardDone("r3", {299});
    PlanOnce();
    SendForwardDone("r3", {300});
    SendFinish("r3");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "pool back to baseline after all three turns";
}

// A decode page registers before the admission slide, and a later
// ReclaimExpired punches it: the punch frees the block WITH its hash intact.
class DecodeCachingSmallWindowSuite : public DecodeCachingSuite {
protected:
    std::int32_t SlidingWindowTokens() const override { return 4; }
};

TEST_F(DecodeCachingSmallWindowSuite, SwaPunchedDecodePageStillHittable) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    // RunTurnOne's fill timing, inlined because the punch round +106 must land
    // BEFORE finish. W=4 slides on top (punched pages = (N-3)/2): +102 punches
    // slot 0, +103 registers page 2, +104 punches slot 1, +105 registers page 3.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    ExecutionPlan r1_prefill = PlanOnce();
    ASSERT_NE(FindForwardBatch(r1_prefill), nullptr);
    AdvanceOneRound("r1", 101);
    AdvanceOneRound("r1", 102);
    AdvanceOneRound("r1", 103);
    AdvanceOneRound("r1", 104);
    const auto r1_rows = AdvanceOneRound("r1", 105);
    ASSERT_EQ(r1_rows.at("swa").size(), 5u);
    EXPECT_EQ(r1_rows.at("swa")[0], 0);
    EXPECT_EQ(r1_rows.at("swa")[1], 0);
    ASSERT_GT(r1_rows.at("swa")[2], 0) << "page 2 is registered AND still live after the +105 round";
    ASSERT_GT(r1_rows.at("swa")[3], 0);

    // +106 -> N=9 -> first kept page 3: slot 2 (REGISTERED at +103) is punched;
    // its block reaches the free list with the hash intact.
    const auto punched = AdvanceOneRound("r1", 106);
    EXPECT_EQ(punched.at("swa")[2], 0) << "the registered decode page must be punched by now";
    SendFinish("r1");
    PlanOnce();  // reap
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);

    // r2: same 8-token prefix + 2 new. Fixpoint (W=4, needed 2): cap =
    // (10-1)/2 = 4, all four hashes cached (0,1,2 punched WITH hash); full
    // matches 4, swa bounded scan keeps 4 (2 holes) -> common 4 = 8 hit tokens.
    std::vector<std::int32_t> r2_tokens = MakeTurnTwoPrompt();
    Submit(MakeSpecWithTokens("r2", r2_tokens));
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/901));
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 6u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 6u);

    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    ExpectRowPrefixEq(op->block_tables.at("full").at(0), full_prefix, "full row");

    // Slot 2's expected id was captured at the +105 round, before the punch.
    const auto& swa_row = op->block_tables.at("swa").at(0);
    ASSERT_EQ(swa_row.size(), 6u);
    EXPECT_EQ(swa_row[0], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[1], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[2], r1_rows.at("swa")[2]) << "punched decode page claimed back by hash";
    EXPECT_EQ(swa_row[3], r1_rows.at("swa")[3]);
    EXPECT_GT(swa_row[4], 0);
    EXPECT_GT(swa_row[5], 0);

    // Pool: full claims 4 + swa claims 2 + 1 fresh/group + 1 reserved/group = 10.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 10);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// Registration writes hashes only -- never refcounts.
TEST_F(DecodeCachingSuite, PoolBalanceAcrossDecodeCaching) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    RunTurnOne();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "turn 1: decode registration must not hold refs";

    Submit(MakeSpecWithTokens("r2", MakeTurnTwoPrompt()));
    ExecutionPlan turn2 = PlanOnce();
    ASSERT_NE(FindForwardBatch(turn2), nullptr);
    EXPECT_LT(scheduler_->PoolFreeBlocks(), free_at_start) << "turn 2 holds claimed + fresh pages while live";
    AdvanceOneRound("r2", 201);
    AdvanceOneRound("r2", 202);
    AdvanceOneRound("r2", 203);
    SendFinish("r2");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "turn 2: claimed and fresh pages all return";

    Submit(MakeSpecWithTokens("r3", MakeTurnThreePrompt()));
    ExecutionPlan turn3 = PlanOnce();
    const ForwardBatch* op3 = FindForwardBatch(turn3);
    ASSERT_NE(op3, nullptr);
    EXPECT_EQ(op3->extend_prefix_lens.at(0), 12);
    SendForwardDone("r3", {299});
    PlanOnce();
    SendForwardDone("r3", {300});
    SendFinish("r3");
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "baseline restored after the whole conversation";
}

// ---------------------------------------------------------------------------
// M15 streaming L2 sink: pages registered by a planning round batch into ONE
// D2H write-back; WriteBackDone commits the host index and unpins the source
// blocks. Byte movement itself is Phase D.
// ---------------------------------------------------------------------------
class StreamingSinkSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 7;  // 6 usable + the null placeholder (page 0, device convention)
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = false;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::FullHistory, CacheGroupFamily::History),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages,
                      CacheGroupConfig::Retention::SlidingWindow, CacheGroupFamily::State,
                      /*sliding_window_tokens=*/4),
        };
        return cfg;
    }

    // Prefill -> finalize; the finalize round registers the prompt's page
    // hashes and streams every completed Full+SWA page.
    ExecutionPlan RunToFinalize(const RequestSpec& spec) {
        Submit(spec);
        PlanOnce();  // prefill
        SendForwardDone(spec.request_id, {9001});
        return PlanOnce();  // PrefillDone -> Decoding: registration + drain
    }

    ExecutionPlan FinishAndReap(const std::string& id) {
        SendForwardDone(id, {9002});
        SendFinish(id);
        return PlanOnce();  // leftover decode pages + latest snapshots, if any
    }

    static std::optional<WriteBackBatch> FindWriteBack(const ExecutionPlan& plan) {
        auto ops = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
        if (ops.empty()) {
            return std::nullopt;
        }
        EXPECT_EQ(ops.size(), 1u) << "the plan must carry at most one merged write-back list";
        return std::get<WriteBackBatch>(ops.front());
    }
};

TEST_F(StreamingSinkSuite, RegisteredPagesEmitWriteBackAndIndexOnDone) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    ExecutionPlan finalize = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value()) << "finalize must stream the 4 Full + 2 SWA completed pages";
    ASSERT_EQ(stream_wb->op_ids.size(), 1u);
    EXPECT_EQ(stream_wb->src_pages.at(0).size(), 6u);
    EXPECT_EQ(stream_wb->dst_pages.at(0).size(), 6u);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0) << "nothing indexed until WriteBackDone";
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 0);

    ExecutionPlan finish = FinishAndReap("r1");
    EXPECT_FALSE(FindWriteBack(finish).has_value()) << "already-streamed prefill pages must not rewrite at finish";
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start)
        << "sources are not pinned: the forward thread's stream orders the copy before any reuse";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0);

    SendWriteBackDone(stream_wb->op_ids.at(0));
    PlanOnce();
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

TEST_F(StreamingSinkSuite, DuplicateRegistrationsAreDroppedAtDrain) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    ExecutionPlan finalize1 = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb1 = FindWriteBack(finalize1);
    ASSERT_TRUE(stream_wb1.has_value());
    EXPECT_FALSE(FindWriteBack(FinishAndReap("r1")).has_value());
    SendWriteBackDone(stream_wb1->op_ids.at(0));
    PlanOnce();
    ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);

    ExecutionPlan finalize2 = RunToFinalize(MakeRequestSpec("r2", /*num_pages=*/4));  // identical tokens
    EXPECT_FALSE(FindWriteBack(finalize2).has_value()) << "already-indexed keys must not re-emit a write-back";
    ExecutionPlan finish2 = FinishAndReap("r2");
    EXPECT_FALSE(FindWriteBack(finish2).has_value()) << "finish must also dedupe already-indexed keys";
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start)
        << "duplicate candidates are unpinned at drain, pool back to baseline";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
}

TEST_F(StreamingSinkSuite, HostPoolExhaustionSkipsSilently) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    ExecutionPlan finalize1 = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb1 = FindWriteBack(finalize1);
    ASSERT_TRUE(stream_wb1.has_value());
    EXPECT_FALSE(FindWriteBack(FinishAndReap("r1")).has_value());
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
    ASSERT_EQ(scheduler_->HostPoolFreeBlocks(), 0) << "r1 holds all 6 host pages in flight";

    ExecutionPlan finalize2 = RunToFinalize(MakeRequestSpec("r2", /*num_pages=*/4, /*start=*/501));
    EXPECT_FALSE(FindWriteBack(finalize2).has_value())
        << "a fully-consumed host pool drops every candidate: no op at all";
    ExecutionPlan finish2 = FinishAndReap("r2");
    EXPECT_FALSE(FindWriteBack(finish2).has_value())
        << "finish-created candidates must also skip a fully-consumed host pool";
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);

    SendWriteBackDone(stream_wb1->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "everything balances after r1's commit";
}

TEST_F(StreamingSinkSuite, CommittedColdEntriesAreReplacedWhenHostPoolIsFull) {
    ExecutionPlan finalize1 = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb1 = FindWriteBack(finalize1);
    ASSERT_TRUE(stream_wb1.has_value());
    EXPECT_FALSE(FindWriteBack(FinishAndReap("r1")).has_value());
    SendWriteBackDone(stream_wb1->op_ids.at(0));
    ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    ASSERT_EQ(scheduler_->HostPoolFreeBlocks(), 0);

    ExecutionPlan finalize2 = RunToFinalize(MakeRequestSpec("r2", /*num_pages=*/4, /*start=*/501));
    auto stream_wb2 = FindWriteBack(finalize2);
    ASSERT_TRUE(stream_wb2.has_value()) << "committed, unpinned Host entries are replaceable";
    EXPECT_EQ(stream_wb2->src_pages.at(0).size(), 6u);
    EXPECT_FALSE(FindWriteBack(FinishAndReap("r2")).has_value());
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0) << "old keys are removed before replacement D2H";
    SendWriteBackDone(stream_wb2->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
}

TEST_F(StreamingSinkSuite, SameRoundDuplicateKeysDedupeAtDrain) {
    // Host pool with headroom (12 usable) so duplicates are dropped by the drain's batch
    // dedupe, NOT by pool exhaustion: two IDENTICAL prompts registering in one round drain
    // 12 candidates into 6 pairs.
    config_.host_allocator.total_pages = 13;
    scheduler_ = std::make_unique<Scheduler>(config_);
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    Submit(MakeRequestSpec("r2", /*num_pages=*/4));
    PlanOnce();  // both prefill (batch 2 <= max_batch_size 8, 16 tokens <= budget 64)
    SendForwardDone("r1", {9001});
    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();  // both register the same completed pages, one merged drain
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value());
    ASSERT_EQ(stream_wb->op_ids.size(), 1u);
    EXPECT_EQ(stream_wb->src_pages.at(0).size(), 6u) << "each Full+SWA key must be emitted at most once";
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 6) << "duplicates must not consume extra host pages";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0);

    EXPECT_FALSE(FindWriteBack(FinishAndReap("r1")).has_value());
    EXPECT_FALSE(FindWriteBack(FinishAndReap("r2")).has_value()) << "same-round duplicates must be dropped";
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start)
        << "no source pins: the forward thread's stream orders the copies before any reuse";

    SendWriteBackDone(stream_wb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 6) << "the six cached host pages remain occupied";
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

TEST_F(StreamingSinkSuite, MidDrainPoolFillEmitsPartialOp) {
    // 4 usable host pages against 6 candidates: the drain emits the 4 that fit and drops the
    // rest -- a partial op IS the contract when the pool fills mid-batch.
    config_.host_allocator.total_pages = 5;
    scheduler_ = std::make_unique<Scheduler>(config_);
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    ExecutionPlan finalize = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value());
    EXPECT_EQ(stream_wb->src_pages.at(0).size(), 4u) << "the drain emits the 4 Full pages that fit first";
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 0);

    ExecutionPlan finish = FinishAndReap("r1");
    EXPECT_FALSE(FindWriteBack(finish).has_value())
        << "in-flight Full pages still occupy the host pool, so leftover SWA must skip";
    SendWriteBackDone(stream_wb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 4);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "dropped candidates unpinned at drain";
}

TEST_F(StreamingSinkSuite, DuplicateWriteBackDoneIsIgnored) {
    ExecutionPlan finalize = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value());
    EXPECT_FALSE(FindWriteBack(FinishAndReap("r1")).has_value());
    const std::int32_t free_after_reap = scheduler_->PoolFreeBlocks();

    SendWriteBackDone(stream_wb->op_ids.at(0));
    ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    const std::int32_t free_after_ack = scheduler_->PoolFreeBlocks();
    EXPECT_EQ(free_after_ack, free_after_reap) << "the ack publishes host entries; no device pins to return";

    // A replayed ack must be a no-op (the ledger already retired the op).
    SendWriteBackDone(stream_wb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_after_ack);
}

// ---------------------------------------------------------------------------
// M15 host-hit load-back: an admission whose device match ends inside the host
// index extends it from the host tier; the plan carries one H2D load-back and
// LoadBackDone releases the host load pins and the destination-page pins.
// ---------------------------------------------------------------------------
class HostHitSuite : public StreamingSinkSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StreamingSinkSuite::MakeConfig();
        cfg.disable_prefix_cache = false;
        // 13 device pages -> 12 free (page 0 is null): the 5-page churn request's peak
        // (10 prefill + 2 reserve) spans the whole free list, recycling r1's 6 cached pages.
        cfg.device_allocator.total_pages = 13;
        cfg.host_allocator.total_pages = 33;  // ample (+null page 0): r1's 6 + the churn's 7 entries fit un-evicted
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }

    static std::optional<LoadBackBatch> FindLoadBack(const ExecutionPlan& plan) {
        auto ops = ExtractCacheOpsOfKind<LoadBackBatch>(plan);
        if (ops.empty()) {
            return std::nullopt;
        }
        EXPECT_EQ(ops.size(), 1u) << "the plan must carry at most one merged load-back list";
        return std::get<LoadBackBatch>(ops.front());
    }

    // Full sink lifecycle: completed Full+SWA pages stream at finalize; finish
    // only emits leftover decode pages or latest snapshots.
    std::vector<WriteBackBatch> RunSinkLifecycle(const RequestSpec& spec) {
        ExecutionPlan finalize = RunToFinalize(spec);
        ExecutionPlan finish = FinishAndReap(spec.request_id);
        std::vector<WriteBackBatch> write_backs;
        if (auto wb = FindWriteBack(finalize)) {
            write_backs.push_back(*wb);
        }
        if (auto wb = FindWriteBack(finish)) {
            write_backs.push_back(*wb);
        }
        return write_backs;
    }

    // r1 (tokens 1..8) indexes 6 host entries (4 Full + 2 SWA); the churn request
    // then floods the free list so r1's pages survive ONLY on the host tier.
    void SeedHostThenEvictDevice() {
        auto wb1 = RunSinkLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
        ASSERT_EQ(wb1.size(), 1u);
        for (const auto& wb : wb1) {
            SendWriteBackDone(wb.op_ids.at(0));
        }
        ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
        // Host cache entries retain their host blocks until host eviction.
        ASSERT_EQ(scheduler_->HostPoolFreeBlocks(), 26);

        auto wb3 = RunSinkLifecycle(MakeRequestSpec("churn", /*num_pages=*/5, /*start=*/501));
        ASSERT_EQ(wb3.size(), 1u);
        for (const auto& wb : wb3) {
            SendWriteBackDone(wb.op_ids.at(0));
        }
        ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 13);
        ASSERT_EQ(scheduler_->PoolFreeBlocks(), 12) << "both seeding requests fully retired";
    }
};

TEST_F(HostHitSuite, HostHitLoadsBackAfterDeviceEviction) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    ASSERT_EQ(free_at_start, 12);
    SeedHostThenEvictDevice();

    // r2 extends r1 by one page, so r1's 8-token endpoint is a matchable boundary:
    // full extension 4 blocks + SWA resume tail 2 blocks = 6 transfer pairs.
    Submit(MakeRequestSpec("r2", /*num_pages=*/5));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(lb->op_ids.size(), 1u);
    ASSERT_EQ(lb->src_pages.at(0).size(), 6u);
    ASSERT_EQ(lb->dst_pages.at(0).size(), 6u);

    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    // The input window skips the 8 host-hit tokens exactly as a device hit would.
    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->prefill_lengths.at(0), 10);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/9));
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 6u) << "4 extension + 2 fresh pages, all new to the table";
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 6u);

    // Wire pairs are group-major: full ext slots 0..3, then SWA slots 2..3.
    const auto& full_row = op->block_tables.at("full").at(0);
    const auto& swa_row = op->block_tables.at("swa").at(0);
    ASSERT_EQ(full_row.size(), 6u);
    ASSERT_EQ(swa_row.size(), 6u);
    const auto& dst = lb->dst_pages.at(0);
    EXPECT_EQ(dst.at(0), full_row.at(0));
    EXPECT_EQ(dst.at(1), full_row.at(1));
    EXPECT_EQ(dst.at(2), full_row.at(2));
    EXPECT_EQ(dst.at(3), full_row.at(3));
    EXPECT_EQ(swa_row.at(0), 0) << "swa slot 0 is the pre-window hole";
    EXPECT_EQ(swa_row.at(1), 0) << "swa slot 1 is the pre-window hole";
    EXPECT_EQ(dst.at(4), swa_row.at(2));
    EXPECT_EQ(dst.at(5), swa_row.at(3));

    // The 6 matched host entries stay load-pinned until LoadBackDone retires the op.
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6);
    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);

    // r2 holds 6 loaded blocks and 4 fresh blocks.
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 10);

    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value()) << "r2's newly completed prefill pages must stream";
    SendWriteBackDone(stream_wb->op_ids.at(0));
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    AckWriteBacks(PlanOnce());
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "pool balances after the host-hit request";
}

TEST_F(HostHitSuite, EmptyHostIndexEmitsNoLoadBack) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    ExecutionPlan plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan), nullptr);
    EXPECT_FALSE(FindLoadBack(plan).has_value()) << "an empty host index must emit no load-back";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
}

TEST_F(HostHitSuite, AbandonedAdmissionUnpins) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    SeedHostThenEvictDevice();

    // Filler: 5 pages -> 10 prefill + 2 reserve = the whole pool while it decodes.
    Submit(MakeRequestSpec("filler", /*num_pages=*/5, /*start=*/701));
    PlanOnce();
    SendForwardDone("filler", {9001});
    ExecutionPlan filler_finalize = PlanOnce();  // finalization slides three expired SWA pages: free = 3
    auto filler_wb = FindWriteBack(filler_finalize);
    ASSERT_TRUE(filler_wb.has_value());
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), 3);

    // r2's host match takes 6 pins, but the gate needs 4 + 6 ext > 3 free: the
    // abandoning return must give the pins back.
    Submit(MakeRequestSpec("r2", /*num_pages=*/5));
    ExecutionPlan blocked = PlanOnce();
    EXPECT_FALSE(FindLoadBack(blocked).has_value());
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0) << "an abandoned admission must unpin its host match";
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);

    // Free the filler (its write-back pins included) -> r2 admits with the load-back.
    SendWriteBackDone(filler_wb->op_ids.at(0));
    SendForwardDone("filler", {9002});
    SendFinish("filler");
    ExecutionPlan after_finish = PlanOnce();
    AckWriteBacks(after_finish);
    auto lb = FindLoadBack(after_finish);
    if (!lb.has_value()) {
        after_finish = PlanOnce();
        lb = FindLoadBack(after_finish);
    }
    ASSERT_TRUE(lb.has_value());
    EXPECT_EQ(lb->src_pages.at(0).size(), 6u);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value()) << "r2's newly completed prefill pages must stream";
    SendWriteBackDone(stream_wb->op_ids.at(0));
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    AckWriteBacks(PlanOnce());
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "pool balances after the deferred host hit";
}

TEST_F(HostHitSuite, AbortDuringLoadKeepsPagesPinned) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    SeedHostThenEvictDevice();

    Submit(MakeRequestSpec("r2", /*num_pages=*/5));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(lb->dst_pages.at(0).size(), 6u);
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 10);

    // Abort while the H2D copy is in flight: the reap returns only the 4 fresh pages;
    // the 6 load destinations must stay off the free list until LoadBackDone.
    SendAbort(*scheduler_, "r2");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 6) << "in-flight load destinations must not be reusable";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6) << "the host sources stay pinned too";

    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "LoadBackDone releases the destinations";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
}

// An abort-during-load leaves pages owned only by the load ledger. Capacity
// handling must wait for LoadBackDone rather than retract or abort a request
// blocked on those pages.
TEST_F(HostHitSuite, CapacityBlockWaitsForInFlightLoads) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    SeedHostThenEvictDevice();

    // Same shape as AbortDuringLoadKeepsPagesPinned: 6 destinations stay ticket-held.
    Submit(MakeRequestSpec("r2", /*num_pages=*/5));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    SendAbort(*scheduler_, "r2");
    PlanOnce();  // reap
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 6);

    // r3 (fresh tokens, no host hit) charges 8 prefill + 2 reserve = 10 > 6 free: deferred.
    Submit(MakeRequestSpec("r3", /*num_pages=*/4, /*start=*/901));
    ExecutionPlan blocked1 = PlanOnce();
    ASSERT_NE(FindForwardBatch(blocked1), nullptr);
    EXPECT_TRUE(FindForwardBatch(blocked1)->request_ids.empty());
    // A second blocked round is also quiet while the load ledger owns pages.
    ExecutionPlan blocked2 = PlanOnce();
    ASSERT_NE(FindForwardBatch(blocked2), nullptr);
    EXPECT_TRUE(FindForwardBatch(blocked2)->request_ids.empty());
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "deferred r3 stays intact in the waiting set";

    // LoadBackDone frees the 6 destinations: r3's 10-block gate now clears.
    SendLoadBackDone(lb->op_ids.at(0));
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
    ExecutionPlan admitted = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(admitted);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    EXPECT_EQ(op->request_ids.at(0), "r3");
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
}

// A LoadBackDone whose op_id was already retired must hit the silent-ignore arm:
// no crash, no double UnpinLoad, no double-free of the destination pages.
TEST_F(HostHitSuite, DuplicateLoadBackDoneIsIgnored) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    SeedHostThenEvictDevice();

    Submit(MakeRequestSpec("r2", /*num_pages=*/5));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 10);

    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
    const std::int32_t free_after_first = scheduler_->PoolFreeBlocks();
    EXPECT_EQ(free_after_first, free_at_start - 10) << "destinations still table-held: no free-list change";

    SendLoadBackDone(lb->op_ids.at(0));  // duplicate
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_after_first) << "a duplicate Done must not double-free";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);

    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value()) << "r2's newly completed prefill pages must stream";
    SendWriteBackDone(stream_wb->op_ids.at(0));
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    AckWriteBacks(PlanOnce());
    PlanOnce();
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "pool balances despite the duplicate event";
}

// ---------------------------------------------------------------------------
// M15 host hit + chunked prefill: later chunks must count the host extension as
// computed tokens, and an SWA slide that punches a still-loading destination
// page must leave it ticket-protected until LoadBackDone.
// ---------------------------------------------------------------------------
class ChunkedHostHitSuite : public HostHitSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = HostHitSuite::MakeConfig();
        cfg.max_scheduled_tokens = 4;  // 4-token prefill chunks
        // 21 -> 20 free: r2's first chunk holds 10 (6 ext + 4 fresh) and its second
        // chunk charges 6 with zero slide credit (ticket-held punches don't count).
        cfg.device_allocator.total_pages = 21;
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }

    // Chunked twin of RunSinkLifecycle: ACK prefill Full+SWA streams round by round,
    // then ACK any leftover finish write-back.
    void RunChunkedSinkLifecycle(const RequestSpec& spec, std::int32_t prefill_rounds) {
        Submit(spec);
        for (std::int32_t i = 0; i < prefill_rounds; ++i) {
            ExecutionPlan plan = PlanOnce();
            ASSERT_NE(FindForwardBatch(plan), nullptr) << "chunk " << i;
            ASSERT_EQ(FindForwardBatch(plan)->request_ids.size(), 1u) << "chunk " << i << " must be admitted";
            AckWriteBacks(plan);
        }
        SendForwardDone(spec.request_id, {9001});
        AckWriteBacks(PlanOnce());  // finalize: registration + drain
        AckWriteBacks(FinishAndReap(spec.request_id));
    }

    // r1 (4 pages) indexes 8 host entries over 2 chunks; the churn request must pop
    // 22 free-list entries (full 10 + swa 10 + reserve 2) = the 12 fresh blocks
    // still unused plus ALL 10 of r1's cached blocks, so r1 survives host-only.
    void SeedHostThenEvictDeviceChunked() {
        RunChunkedSinkLifecycle(MakeRequestSpec("r1", /*num_pages=*/4), /*prefill_rounds=*/2);
        ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 8);
        RunChunkedSinkLifecycle(MakeRequestSpec("churn", /*num_pages=*/10, /*start=*/501), /*prefill_rounds=*/5);
        ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 28);
        ASSERT_EQ(scheduler_->PoolFreeBlocks(), 20) << "both seeding requests fully retired";
    }
};

TEST_F(ChunkedHostHitSuite, ChunkedPrefillAfterHostHit) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();
    ASSERT_EQ(free_at_start, 20);
    SeedHostThenEvictDeviceChunked();

    // r2: 16 tokens sharing r1's first 8. Host extension = 4 blocks (full run 0..3;
    // swa tail ceil((W-1)/P)=2 at the boundary) -> real pages full 4 + swa 2 = 6.
    Submit(MakeRequestSpec("r2", /*num_pages=*/8));
    ExecutionPlan c1 = PlanOnce();
    auto lb = FindLoadBack(c1);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(lb->src_pages.at(0).size(), 6u);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6);

    const ForwardBatch* op1 = FindForwardBatch(c1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 1u);
    // First chunk: the 8 host-hit tokens are computed; chunk covers tokens [8,12).
    EXPECT_EQ(op1->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op1->input_lengths.at(0), 4);
    EXPECT_EQ(op1->prefill_lengths.at(0), 16);
    // 6 ext + 2 fresh/group: 10 blocks held.
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 10);

    // Chunk 2 completes prefill; its slide at num_computed=12 punches swa ext slots
    // 2,3 = LOADED destinations mid-copy. The ticket must keep them off the free list.
    ExecutionPlan c2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(c2);
    ASSERT_NE(op2, nullptr);
    ASSERT_EQ(op2->request_ids.size(), 1u);
    EXPECT_EQ(op2->extend_prefix_lens.at(0), 12) << "chunk 2 must see ext(8) + chunk1(4) as computed";
    EXPECT_EQ(op2->input_lengths.at(0), 4);
    AckWriteBacks(c2);  // pages 4,5 registered this round; ack so only the ticket pins remain
    // Four input tokens plus one decode-reserve token require 3 new pages per
    // group; the 2 punched load destinations stay ticket-held.
    ASSERT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 16);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6) << "the copy is still in flight";

    // LoadBackDone releases exactly the 2 punched destinations (the other 4 stay table-held).
    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start - 14);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);

    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();
    ASSERT_NE(FindForwardBatch(finalize), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    AckWriteBacks(finalize);
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    AckWriteBacks(PlanOnce());  // any remaining decode write-back + reap
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start) << "pool balances after the chunked host hit";
}

}  // namespace tokenspeed::test
