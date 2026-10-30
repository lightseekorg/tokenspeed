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

// Recurrent-state groups through the Scheduler: the two-slot table (slot 0
// the checkpoint a forward reads, slot 1 the block it writes) rotating through
// prefill chunks, the first decode and later decodes; natural checkpoints
// published at prompt end, finish and retraction only; hits resuming from
// them; the two-blocks-per-request capacity; and the retraction image of
// both slots.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {

#define ASSERT_TRUE_OR_RETURN(cond, ret) \
    do {                                 \
        EXPECT_TRUE(cond);               \
        if (!(cond)) return ret;         \
    } while (false)

class StateCheckpointSuite : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kP = 4;

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = kP;
        cfg.device_allocator.total_pages = 257;
        cfg.host_allocator.total_pages = 257;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 8;
        cfg.max_batch_size = 2;
        cfg.disable_l2_cache = true;
        for (const std::string& id : {"full", "state0", "state1", "state2"}) {
            const bool is_state = id != "full";
            cfg.cache_groups.push_back(CacheGroupConfig{
                .group_id = id,
                .block_granularity = is_state ? 0 : kP,
                .total_pages = cfg.device_allocator.total_pages,
                .attn_kind = is_state ? AttnKind::kState : AttnKind::kFull,
            });
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void Reset(bool host_cache, std::int32_t decode_width) {
        config_ = WithHostPackingFromDevice(MakeConfig());
        config_.disable_l2_cache = !host_cache;
        config_.decode_input_tokens = decode_width;
        scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    }

    static RequestSpec RequestWithTokens(const std::string& id, std::vector<std::int32_t> tokens) {
        return RequestSpec{.request_id = id, .tokens = std::move(tokens), .max_new_tokens = 32};
    }

    // The P role's peer events: the bootstrap a prompt waits for before its
    // admission, and the ACK of the KV transfer that finishes it.
    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{request_id});
        scheduler_->Advance(std::move(event));
    }

    void SendPdSucceeded(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::SucceededEvent{request_id});
        scheduler_->Advance(std::move(event));
    }

    static std::vector<std::int32_t> StateRow(const ForwardBatch& batch, const std::string& group,
                                              const std::string& id) {
        const auto it = std::ranges::find(batch.request_ids, id);
        ASSERT_TRUE_OR_RETURN(it != batch.request_ids.end(), {});
        return batch.block_tables.at(group).at(static_cast<std::size_t>(it - batch.request_ids.begin()));
    }

    // Plans one round and returns the state0 row of `id`, which must be in
    // the batch.
    std::vector<std::int32_t> PlanAndStateRow(const std::string& id, ExecutionPlan* plan_out = nullptr) {
        ExecutionPlan plan = PlanOnce();
        AckWriteBacks(plan);
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_TRUE_OR_RETURN(batch != nullptr, {});
        std::vector<std::int32_t> row = StateRow(*batch, "state0", id);
        if (plan_out != nullptr) {
            *plan_out = std::move(plan);
        }
        return row;
    }

    // Submits a prompt, plans one round and reports where its first chunk
    // resumed and the state0 row it was given, then aborts it.
    struct Resume {
        std::int32_t prefix_tokens{-1};
        std::vector<std::int32_t> state_row;
        std::int32_t load_rows{0};
    };
    Resume Probe(const std::string& id, std::vector<std::int32_t> tokens) {
        Submit(RequestWithTokens(id, std::move(tokens)));
        if (config_.role == Role::kP) {
            SendBootstrapped(id);
        }
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        Resume resume;
        if (batch == nullptr) {
            return resume;
        }
        const auto it = std::ranges::find(batch->request_ids, id);
        if (it == batch->request_ids.end()) {
            return resume;
        }
        const auto row = static_cast<std::size_t>(it - batch->request_ids.begin());
        resume.prefix_tokens = batch->extend_prefix_lens.at(row);
        resume.state_row = batch->block_tables.at("state0").at(row);
        for (const CacheOperation& operation : ExtractCacheOpsOfKind<LoadBackBatch>(plan)) {
            const auto& load = std::get<LoadBackBatch>(operation);
            for (std::size_t i = 0; i < load.op_ids.size(); ++i) {
                resume.load_rows += static_cast<std::int32_t>(load.src_pages[i].size());
                SendLoadBackDone(load.op_ids[i]);
            }
        }
        AckWriteBacks(plan);
        SendAbortEvent(id);
        PlanOnce();
        return resume;
    }
};

class AlignedStateCheckpointSuite : public StateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointSuite::MakeConfig();
        cfg.prefix_granularity = 128;
        cfg.max_scheduled_tokens = 16384;
        // An aligned grain requires the consumer that writes the checkpoint block.
        cfg.state_checkpoint_in_forward = true;
        for (auto& group : cfg.cache_groups) {
            if (group.attn_kind == AttnKind::kState) {
                group.checkpoint_granularity = 4096;
            } else {
                group.block_granularity = 128;
            }
        }
        return cfg;
    }
};

TEST_F(AlignedStateCheckpointSuite, FullPromptAlwaysRecomputesTheLastSegmentAndNeverHitsATail) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    Submit(RequestWithTokens("seed", MakeTokens(8192)));
    std::map<std::int32_t, std::int32_t> checkpoints;
    {
        ExecutionPlan plan;
        auto state = PlanAndStateRow("seed", &plan);
        const auto* batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        ASSERT_EQ(batch->extend_prefix_lens.size(), 1u);
        EXPECT_EQ(batch->extend_prefix_lens[0], 0);
        EXPECT_EQ(batch->input_lengths[0], 8192);
        ASSERT_EQ(state.size(), 3u);
        EXPECT_NE(state[2], state[0]);
        EXPECT_NE(state[2], state[1]);
        EXPECT_EQ(std::ranges::count(plan.pages_to_zero.at("state0"), state[2]), 1);
        checkpoints[4096] = state.at(2);
        checkpoints[8192] = state.at(1);
        SendForwardDone("seed", {99});
    }
    SendFinish("seed");
    PlanOnce();
    // Cache lookup is real: no manufactured hits or injected checkpoint keys.
    for (const auto [length, expected] : std::vector<std::pair<int, int>>{
             {4096, 0}, {4097, 4096}, {8191, 4096}, {8192, 4096}, {8193, 8192}, {12288, 8192}}) {
        const Resume hit = Probe("probe-" + std::to_string(length), MakeTokens(length));
        EXPECT_EQ(hit.prefix_tokens, expected) << "prompt=" << length;
        if (expected != 0) {
            EXPECT_EQ(hit.state_row.at(0), checkpoints.at(expected));
        }
    }
}

TEST_F(AlignedStateCheckpointSuite, OnlyTheLastSegmentEntranceIsPublishedBeforeThePromptEnd) {
    for (const std::int32_t budget : {4096, 16384, 32768}) {
        SCOPED_TRACE(budget);
        config_ = MakeConfig();
        config_.max_scheduled_tokens = budget;
        config_.device_allocator.total_pages = 1025;
        config_.host_allocator.total_pages = 1025;
        for (auto& group : config_.cache_groups) {
            group.total_pages = 1025;
        }
        SetTestSnapshotPool(config_);
        scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
        Submit(RequestWithTokens("seed", MakeTokens(16385)));
        std::int32_t saved_state = -1;
        for (std::int32_t begin = 0; begin < 16385;) {
            ExecutionPlan plan;
            const auto state = PlanAndStateRow("seed", &plan);
            const auto* batch = FindForwardBatch(plan);
            ASSERT_NE(batch, nullptr);
            EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{begin});
            const std::int32_t length = std::min(budget, 16385 - begin);
            EXPECT_EQ(batch->input_lengths, std::vector<std::int32_t>{length});
            begin += length;
            if (begin >= 16384 && saved_state < 0) {
                saved_state = state.at(begin == 16384 ? 1 : 2);
            }
            SendForwardDone("seed", begin == 16385 ? std::vector<std::int32_t>{99} : std::vector<std::int32_t>{});
        }
        SendFinish("seed");
        PlanOnce();
        for (const auto [length, expected] : std::vector<std::pair<int, int>>{
                 {4097, 0}, {8193, 0}, {12289, 0}, {16384, 0}, {16385, 16384}, {20480, 16384}}) {
            const Resume hit = Probe("probe-" + std::to_string(length), MakeTokens(length));
            EXPECT_EQ(hit.prefix_tokens, expected) << "prompt=" << length;
            if (expected != 0) {
                EXPECT_EQ(hit.state_row.at(0), saved_state);
            }
        }
    }
}

TEST_F(AlignedStateCheckpointSuite, A4097TokenRequestSavesItsInteriorCheckpointInOneForward) {
    config_ = MakeConfig();
    config_.max_scheduled_tokens = 8192;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    Submit(RequestWithTokens("a", MakeTokens(1808)));
    Submit(RequestWithTokens("b", MakeTokens(4097, 100000)));
    const ExecutionPlan first = PlanOnce();
    const auto* batch = FindForwardBatch(first);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, std::vector<std::string>({"a", "b"}));
    EXPECT_EQ(batch->input_lengths, std::vector<std::int32_t>({1808, 4097}));
    EXPECT_LE(StateRow(*batch, "state0", "a").at(2), 0) << "padding is not an allocated checkpoint";
    const auto checkpoint = StateRow(*batch, "state0", "b").at(2);
    SendForwardDone("a", {99});
    SendFinish("a");
    SendForwardDone("b", {99});
    SendFinish("b");
    PlanOnce();
    const Resume hit = Probe("probe", MakeTokens(4097, 100000));
    EXPECT_EQ(hit.prefix_tokens, 4096);
    EXPECT_EQ(hit.state_row.at(0), checkpoint);
}

TEST_F(AlignedStateCheckpointSuite, RejectsBudgetSmallerThanOneSegment) {
    auto cfg = MakeConfig();
    cfg.max_scheduled_tokens = 4095;
    EXPECT_THROW(Scheduler{cfg}, std::invalid_argument);
}

TEST_F(AlignedStateCheckpointSuite, InteriorCheckpointSurvivesRotationAndHostOffload) {
    for (const bool host_cache : {false, true}) {
        Reset(host_cache, /*decode_width=*/1);
        Submit(RequestWithTokens("seed", MakeTokens(8192)));
        ExecutionPlan plan;
        const auto first = PlanAndStateRow("seed", &plan);
        ASSERT_EQ(first.size(), 3u);
        SendForwardDone("seed", {99});
        const auto decode = PlanAndStateRow("seed", &plan);
        ASSERT_EQ(decode.size(), 2u) << "the transient output leaves the request after publication";
        EXPECT_EQ(decode[0], first[1]);
        EXPECT_NE(decode[1], first[2]) << "the published checkpoint is immutable";
        SendForwardDone("seed", {100});
        SendFinish("seed");
        AckWriteBacks(PlanOnce());
        if (host_cache) {
            ASSERT_TRUE(scheduler_->ClearL1Cache());
        }
        const Resume hit = Probe("hit", MakeTokens(8192));
        EXPECT_EQ(hit.prefix_tokens, 4096);
        if (host_cache) {
            EXPECT_GE(hit.load_rows, 32 + 3) << "KV pages plus the interior state of every group";
        } else {
            EXPECT_EQ(hit.state_row.at(0), first[2]);
        }
    }
}

TEST_F(AlignedStateCheckpointSuite, AbortDoesNotPublishTheInteriorOutput) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    const auto empty = scheduler_->EmptyLcmBlocks();
    Submit(RequestWithTokens("seed", MakeTokens(8192)));
    ASSERT_EQ(PlanAndStateRow("seed").size(), 3u);
    SendForwardDone("seed", {99});
    SendAbortEvent("seed");
    PlanOnce();
    EXPECT_EQ(scheduler_->EmptyLcmBlocks(), empty);
    EXPECT_EQ(Probe("miss", MakeTokens(8192)).prefix_tokens, 0);
}

TEST_F(AlignedStateCheckpointSuite, PrefillResultsDoNotPublishWithoutAFollowingAdmission) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    Submit(RequestWithTokens("seed", MakeTokens(8192)));
    ASSERT_EQ(PlanAndStateRow("seed").size(), 3u);
    SendForwardDone("seed", {8193});
    // A result advances the request, but publication remains in admission.
    // Abort before the next admission drops the still-private checkpoint.
    SendAbortEvent("seed");
    PlanOnce();
    EXPECT_EQ(Probe("miss", MakeTokens(8193)).prefix_tokens, 0);
}

TEST_F(AlignedStateCheckpointSuite, FirstDecodeAdmissionDeliversThePrecedingPrefillCheckpoints) {
    for (const bool prefill_landed : {false, true}) {
        SCOPED_TRACE(prefill_landed);
        Reset(/*host_cache=*/false, /*decode_width=*/1);
        Submit(RequestWithTokens("seed", MakeTokens(8192)));
        const auto prefill_state = PlanAndStateRow("seed");
        ASSERT_EQ(prefill_state.size(), 3u);
        if (prefill_landed) {
            SendForwardDone("seed", {8193});
        }
        const auto decode_state = PlanAndStateRow("seed");
        ASSERT_EQ(decode_state.size(), 2u);
        EXPECT_EQ(decode_state[0], prefill_state[1]);
        EXPECT_NE(decode_state[1], prefill_state[1]);
        // Stream order permits publication of the previous plan's outputs
        // before its CPU result lands. No checkpoint travels on that result.
        SendAbortEvent("seed");
        PlanOnce();
        const Resume interior = Probe("interior", MakeTokens(8192));
        EXPECT_EQ(interior.prefix_tokens, 4096);
        ASSERT_FALSE(interior.state_row.empty());
        EXPECT_EQ(interior.state_row[0], prefill_state[2]);
        const Resume endpoint = Probe("endpoint", MakeTokens(8193));
        EXPECT_EQ(endpoint.prefix_tokens, 8192);
        ASSERT_FALSE(endpoint.state_row.empty());
        EXPECT_EQ(endpoint.state_row[0], prefill_state[1]);
    }
}

TEST_F(AlignedStateCheckpointSuite, DecodeRoleDoesNotPublishTheTransferredPromptState) {
    config_ = MakeConfig();
    config_.role = Role::kD;
    for (auto& group : config_.cache_groups) {
        group.transfer_policy =
            group.attn_kind == AttnKind::kState ? CacheTransferPolicy::LatestSnapshot : CacheTransferPolicy::FullSuffix;
    }
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    Submit(RequestWithTokens("seed", MakeTokens(8192)));
    ExecutionEvent bootstrap;
    bootstrap.With(pd::BootstrappedEvent{"seed"});
    scheduler_->Advance(std::move(bootstrap));
    ASSERT_TRUE(PlanOnce().remote_prefill.has_value());
    ExecutionEvent remote_done;
    remote_done.With(pd::RemotePrefillDoneEvent{"seed", 8193});
    scheduler_->Advance(std::move(remote_done));
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    SendForwardDone("seed", {8194});
    SendFinish("seed");
    PlanOnce();
    Submit(RequestWithTokens("probe", MakeTokens(8192)));
    ExecutionEvent probe_bootstrap;
    probe_bootstrap.With(pd::BootstrappedEvent{"probe"});
    scheduler_->Advance(std::move(probe_bootstrap));
    const ExecutionPlan probe = PlanOnce();
    ASSERT_TRUE(probe.remote_prefill.has_value());
    // D may still reuse KV pages. A cached prompt-end State would instead
    // appear in slot 0 and cause that group's transfer to be omitted.
    EXPECT_LE(StateRow(*probe.remote_prefill, "state0", "probe").at(0), 0);
}

TEST_F(AlignedStateCheckpointSuite, DecodeCannotPublishAnAlignedFrontierAtFinish) {
    for (const bool host_cache : {false, true}) {
        for (const bool finish_in_flight : {false, true}) {
            for (const std::int32_t prompt : {4095, 8191}) {
                SCOPED_TRACE(::testing::Message()
                             << "host=" << host_cache << " in_flight=" << finish_in_flight << " prompt=" << prompt);
                config_ = MakeConfig();
                config_.disable_l2_cache = !host_cache;
                config_.enable_mixed_prefill_decode = true;
                config_.decode_input_tokens = 1;
                scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
                Submit(RequestWithTokens("seed", MakeTokens(prompt)));
                const ExecutionPlan prefill = PlanOnce();
                ASSERT_NE(FindForwardBatch(prefill), nullptr);
                SendForwardDone("seed", {prompt + 1});
                AckWriteBacks(PlanOnce());  // may publish the preceding prefill checkpoint only
                if (finish_in_flight) {
                    SendFinish("seed");
                }
                SendForwardDone("seed", {prompt + 2});  // decode consumed one token: frontier 4096 or 8192
                if (!finish_in_flight) {
                    SendFinish("seed");
                }
                AckWriteBacks(PlanOnce());
                if (host_cache) {
                    ASSERT_TRUE(scheduler_->ClearL1Cache());
                }
                EXPECT_EQ(Probe("hit", MakeTokens(prompt + 2)).prefix_tokens, prompt / 4096 * 4096)
                    << "only the prefill-created entrance may be reused, not the aligned decode state";
            }
        }
    }
}

TEST_F(AlignedStateCheckpointSuite, RetractedDecodeStateRestoresPrivatelyWithoutBecomingAPrefixHit) {
    config_ = MakeConfig();
    config_.disable_l2_cache = false;
    config_.debug_force_retraction_interval = 3;
    config_.decode_input_tokens = 1;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    Submit(RequestWithTokens("seed", MakeTokens(4095)));
    const ExecutionPlan prefill = PlanOnce();  // plan 1: prefill
    ASSERT_NE(FindForwardBatch(prefill), nullptr);
    AckWriteBacks(prefill);
    SendForwardDone("seed", {4096});
    const ExecutionPlan decode = PlanOnce();  // plan 2: decode consumes token 4096
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    AckWriteBacks(decode);
    SendForwardDone("seed", {4097});
    const ExecutionPlan retraction = PlanOnce();  // plan 3: frontier 4096
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    ASSERT_NE(FindSnapshotStore(retraction), nullptr);
    AckImageStores(retraction);
    const ExecutionPlan readmit = PlanOnce();
    ASSERT_NE(FindRestore(readmit), nullptr);
    AckRestores(readmit);
    const auto restored_state = PlanAndStateRow("seed");
    ASSERT_GT(restored_state.at(1), 0) << "the request can still resume its private decode state";
    SendForwardDone("seed", {4098});
    SendAbortEvent("seed");
    EXPECT_EQ(Probe("miss", MakeTokens(4097)).prefix_tokens, 0)
        << "the decode image is not a canonical prefill checkpoint";
}

TEST_F(AlignedStateCheckpointSuite, AFinalTailDoesNotAlignTheNextRequestsPackedOffset) {
    for (const bool resumed : {false, true}) {
        for (const std::int32_t budget : {4096, 8192}) {
            for (const std::int32_t tail : {1, 63, 64, 65, 1808, 4095, 4096}) {
                SCOPED_TRACE(::testing::Message() << "resumed=" << resumed << " budget=" << budget << " tail=" << tail);
                config_ = MakeConfig();
                config_.max_scheduled_tokens = budget;
                scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
                if (resumed) {
                    Submit(RequestWithTokens("seed", MakeTokens(4096)));
                    const ExecutionPlan first = PlanOnce();
                    const auto* batch = FindForwardBatch(first);
                    ASSERT_NE(batch, nullptr);
                    EXPECT_EQ(batch->request_ids, std::vector<std::string>{"seed"});
                    EXPECT_EQ(batch->input_lengths, std::vector<std::int32_t>{4096});
                    SendForwardDone("seed", {99});
                    SendFinish("seed");
                    PlanOnce();
                }
                Submit(RequestWithTokens("a", MakeTokens((resumed ? 4096 : 0) + tail)));
                Submit(RequestWithTokens("b", MakeTokens(8193, 100000)));
                const ExecutionPlan plan = PlanOnce();
                const auto* batch = FindForwardBatch(plan);
                ASSERT_NE(batch, nullptr);
                const bool b_fits = budget - tail >= 4096;
                EXPECT_EQ(batch->request_ids,
                          b_fits ? std::vector<std::string>({"a", "b"}) : std::vector<std::string>({"a"}));
                EXPECT_EQ(batch->input_lengths,
                          b_fits ? std::vector<std::int32_t>({tail, 4096}) : std::vector<std::int32_t>({tail}));
                EXPECT_EQ(batch->extend_prefix_lens, b_fits ? std::vector<std::int32_t>({resumed ? 4096 : 0, 0})
                                                            : std::vector<std::int32_t>({resumed ? 4096 : 0}));
                auto expected_ids = MakeTokens(tail, resumed ? 4097 : 1);
                if (b_fits) {
                    const auto b_ids = MakeTokens(4096, 100000);
                    expected_ids.insert(expected_ids.end(), b_ids.begin(), b_ids.end());
                }
                EXPECT_EQ(batch->input_ids, expected_ids);
                const std::string key = std::string(resumed ? "resumed" : "cold") + "_budget_" +
                                        std::to_string(budget) + "_tail_" + std::to_string(tail);
                RecordProperty(key, b_fits ? "a=" + std::to_string(tail) +
                                                 ",b=4096,b_packed_offset=" + std::to_string(tail) + ",b_prefix=0"
                                           : "a=" + std::to_string(tail) + ",b=wait");
                // There is no batch padding after A. B waits only when its
                // complete next segment cannot fit in the remaining budget.
                SendForwardDone("a", {99});
                SendFinish("a");
                if (b_fits) {
                    SendForwardDone("b");
                }
                const ExecutionPlan next = PlanOnce();
                const auto* next_batch = FindForwardBatch(next);
                ASSERT_NE(next_batch, nullptr);
                EXPECT_EQ(next_batch->request_ids, std::vector<std::string>{"b"});
                const std::int32_t next_length = b_fits && budget == 8192 ? 4097 : (budget == 8192 ? 8192 : 4096);
                EXPECT_EQ(next_batch->input_lengths, std::vector<std::int32_t>{next_length});
                EXPECT_EQ(next_batch->extend_prefix_lens, std::vector<std::int32_t>{b_fits ? 4096 : 0});
                EXPECT_EQ(next_batch->input_ids, MakeTokens(next_length, b_fits ? 104096 : 100000));
                SendAbortEvent("b");
                PlanOnce();
            }
        }
    }
}

TEST_F(StateCheckpointSuite, TwoSlotsRotateThroughChunksTheFirstDecodeAndLaterDecodes) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    // A 10-token prompt under an 8-token chunk: chunk [0, 8), chunk [8, 10),
    // then decode steps.
    Submit(RequestWithTokens("r", MakeTokens(10)));
    ExecutionPlan plan;
    const std::vector<std::int32_t> chunk1 = PlanAndStateRow("r", &plan);
    ASSERT_EQ(chunk1.size(), 2u);
    EXPECT_GT(chunk1[0], 0) << "no checkpoint: the first chunk starts from a zeroed block";
    EXPECT_GT(chunk1[1], 0);
    EXPECT_NE(chunk1[0], chunk1[1]);
    for (const std::int32_t slot : chunk1) {
        EXPECT_EQ(std::ranges::count(plan.pages_to_zero.at("state0"), slot), 1) << "both fresh slots are zeroed";
    }
    SendForwardDone("r");

    // The rotation swaps the released zero state for the fresh live slot: the
    // request never holds more than two blocks.
    const std::vector<std::int32_t> chunk2 = PlanAndStateRow("r", &plan);
    EXPECT_EQ(chunk2[0], chunk1[1]) << "the first chunk's output is the second chunk's input";
    EXPECT_GT(chunk2[1], 0);
    EXPECT_NE(chunk2[1], chunk2[0]);
    EXPECT_EQ(plan.pages_to_zero.at("state0"), std::vector<std::int32_t>{chunk2[1]});
    EXPECT_TRUE(plan.pages_to_copy.at("full").empty()) << "an ordinary chunk end publishes nothing";
    EXPECT_EQ(scheduler_->TailEntries(), 0u);
    SendForwardDone("r", {101});

    // The first decode is the prompt end: the state the second chunk wrote
    // is published and rotated into slot 0, the decode writes a fresh slot
    // 1, and the partial KV page is published with a copy-in.
    const std::vector<std::int32_t> decode1 = PlanAndStateRow("r", &plan);
    EXPECT_EQ(decode1[0], chunk2[1]) << "the prompt-end state is the first decode's input";
    EXPECT_GT(decode1[1], 0);
    EXPECT_NE(decode1[1], decode1[0]);
    EXPECT_EQ(plan.pages_to_copy.at("full").size(), 1u) << "the partial page [8, 10) moves to a private copy";
    EXPECT_TRUE(plan.pages_to_copy.at("state0").empty()) << "a state block is never copied";
    EXPECT_EQ(scheduler_->TailEntries(), 1u);
    SendForwardDone("r", {102});

    // Every later decode writes slot 1 in place; slot 0 is released.
    const std::vector<std::int32_t> decode2 = PlanAndStateRow("r", &plan);
    EXPECT_EQ(decode2, (std::vector<std::int32_t>{0, decode1[1]}));
    EXPECT_TRUE(plan.pages_to_zero.at("state0").empty());
    SendForwardDone("r", {103});
    EXPECT_EQ(PlanAndStateRow("r"), decode2);
    SendForwardDone("r", {104});

    // Another prompt sharing the ten prompt tokens resumes at the published
    // prompt-end state; one sharing only the first chunk finds no
    // checkpoint at 8 and starts over (with the junction at 8).
    const Resume at_prompt_end = Probe("hit", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 78});
    EXPECT_EQ(at_prompt_end.prefix_tokens, 10);
    EXPECT_EQ(at_prompt_end.state_row.at(0), decode1[0]) << "resumes from the published checkpoint";
    const Resume at_chunk_end = Probe("miss", {1, 2, 3, 4, 5, 6, 7, 8, 77, 78, 79, 80});
    EXPECT_EQ(at_chunk_end.prefix_tokens, 0);
    EXPECT_NE(at_chunk_end.state_row.at(0), decode1[0]) << "a fresh zero state, not a checkpoint";

    // The two probes above each planned a decode of r; both land before the
    // finish, which publishes the frontier (15 computed tokens: the prompt
    // and the five consumed generated tokens) and its state.
    SendForwardDone("r", {105});
    SendForwardDone("r", {106});
    SendFinish("r");
    PlanOnce();
    EXPECT_EQ(scheduler_->TailEntries(), 2u);
    const Resume at_finish = Probe("finish-hit", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 101, 102, 103, 104, 105, 77});
    EXPECT_EQ(at_finish.prefix_tokens, 15);
    EXPECT_EQ(at_finish.state_row.at(0), decode1[1]) << "the live slot published at finish";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks())
        << "every block is cache-only or free";
}

// The in-forward State checkpoint (SchedulerConfig::state_checkpoint_in_forward):
// the forward that spans a long prompt's last page boundary before its
// replay tail also writes the state there, into a third block of every State
// group's row, without cutting a chunk; the request's next publication point
// publishes that block under the boundary's page hash.
class StateCheckpointInForwardSuite : public StateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointSuite::MakeConfig();
        cfg.state_checkpoint_in_forward = true;
        return cfg;
    }

    // One prefill forward of a request, as the planner shaped it.
    struct Chunk {
        std::int32_t prefix_tokens{0};
        std::int32_t tokens{0};
        std::int32_t checkpoint_len{0};
        // The state0..state2 rows.
        std::vector<std::vector<std::int32_t>> state_rows;
        std::vector<std::int32_t> zeroed_state0;
    };

    // Plans `id`'s prefill from its hit to its end, alone in each batch,
    // landing every forward (the last with a sampled token).
    std::vector<Chunk> RunPrefill(const std::string& id, std::int32_t prompt_tokens) {
        std::vector<Chunk> chunks;
        std::int32_t scheduled = 0;
        for (int round = 0; scheduled < prompt_tokens && round < 16; ++round) {
            const ExecutionPlan plan = PlanOnce();
            const ForwardBatch* batch = FindForwardBatch(plan);
            ASSERT_TRUE_OR_RETURN(batch != nullptr && batch->request_ids == std::vector<std::string>{id}, chunks);
            Chunk chunk{.prefix_tokens = batch->extend_prefix_lens.at(0),
                        .tokens = batch->input_lengths.at(0),
                        .checkpoint_len = batch->extend_state_checkpoint_lens.at(0),
                        .zeroed_state0 = plan.pages_to_zero.at("state0")};
            for (const std::string& group : {"state0", "state1", "state2"}) {
                chunk.state_rows.push_back(StateRow(*batch, group, id));
            }
            scheduled = chunk.prefix_tokens + chunk.tokens;
            chunks.push_back(std::move(chunk));
            SendForwardDone(id,
                            scheduled == prompt_tokens ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
            AckWriteBacks(plan);
        }
        return chunks;
    }
};

class ReservedDecodeCheckpointSuite : public StateCheckpointInForwardSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = StateCheckpointInForwardSuite::MakeConfig();
        cfg.prefix_granularity = 64;
        cfg.max_scheduled_tokens = 256;
        cfg.decode_input_tokens = 4;
        cfg.overlap_schedule_depth = 1;
        cfg.disable_l2_cache = false;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
        for (auto& group : cfg.cache_groups) {
            if (group.attn_kind != AttnKind::kState) {
                group.block_granularity = 64;
            }
        }
        return cfg;
    }
};

TEST_F(ReservedDecodeCheckpointSuite, ReservedStepsKeepGeneratedHashesAndTheFinalStateReusable) {
    std::vector<std::int32_t> tokens = MakeTokens(385);
    auto owner = RequestWithTokens("owner", tokens);
    owner.max_new_tokens = 512;
    Submit(owner);
    const auto chunks = RunPrefill("owner", 385);
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks.back().checkpoint_len, 128);
    tokens.push_back(900);

    std::int32_t live_state = 0;
    for (std::int32_t round = 0; round < 80; ++round) {
        const auto row = PlanAndStateRow("owner");
        ASSERT_EQ(row.size(), 2u);
        ASSERT_GT(row[1], 0);
        live_state = row[1];
        const std::vector<std::int32_t> accepted{1000 + 2 * round, 1001 + 2 * round};
        SendForwardDone("owner", accepted);
        tokens.insert(tokens.end(), accepted.begin(), accepted.end());
    }
    SendFinish("owner");
    AckWriteBacks(PlanOnce());

    const auto computed_tokens = static_cast<std::int32_t>(tokens.size()) - 1;
    tokens.push_back(2000);
    const auto resumed = Probe("reuse", tokens);
    EXPECT_EQ(resumed.prefix_tokens, computed_tokens);
    ASSERT_EQ(resumed.state_row.size(), 2u);
    EXPECT_EQ(resumed.state_row[0], live_state);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks());
}

// Under the 8-token budget a 22-token prompt keeps its plain chunks 8, 8 and
// 6. The last one spans the boundary 20 and writes the state after its first
// 4 tokens into a third, freshly zeroed block of each State group's row. The
// finish publishes it, so the same prompt sent again, or one that changes
// only its last page, resumes at 20 -- by default both start over, like the
// "miss" probe of TwoSlotsRotateThroughChunksTheFirstDecodeAndLaterDecodes.
TEST_F(StateCheckpointInForwardSuite, ALongPromptWritesItsCheckpointInsideTheForwardThatSpansIt) {
    for (const bool host_cache : {false, true}) {
        SCOPED_TRACE(host_cache);
        Reset(host_cache, /*decode_width=*/1);
        std::vector<std::int32_t> tokens = MakeTokens(22);
        Submit(RequestWithTokens("seed", tokens));
        const std::vector<Chunk> chunks = RunPrefill("seed", 22);
        ASSERT_EQ(chunks.size(), 3u) << "no extra chunk";
        EXPECT_EQ(chunks[0].tokens, 8);
        EXPECT_EQ(chunks[1].tokens, 8);
        EXPECT_EQ(chunks[2].tokens, 6);
        for (std::size_t i = 0; i < 2; ++i) {
            EXPECT_EQ(chunks[i].checkpoint_len, 0);
            EXPECT_EQ(chunks[i].state_rows[0].size(), 2u);
        }
        EXPECT_EQ(chunks[2].checkpoint_len, 4) << "the state after position 16 + 4";
        std::vector<std::int32_t> checkpoint_blocks;
        for (const std::vector<std::int32_t>& row : chunks[2].state_rows) {
            ASSERT_EQ(row.size(), 3u);
            EXPECT_GT(row[2], 0);
            EXPECT_NE(row[2], row[0]);
            EXPECT_NE(row[2], row[1]);
            checkpoint_blocks.push_back(row[2]);
        }
        EXPECT_EQ(std::ranges::count(chunks[2].zeroed_state0, checkpoint_blocks[0]), 1);
        SendFinish("seed");
        AckWriteBacks(PlanOnce());

        const Resume resent = Probe("resent", tokens);
        EXPECT_EQ(resent.prefix_tokens, 20);
        EXPECT_EQ(resent.state_row.at(0), checkpoint_blocks[0]) << "resumes from the in-forward checkpoint";
        tokens[20] = 991;
        tokens[21] = 992;
        const Resume changed = Probe("changed", tokens);
        EXPECT_EQ(changed.prefix_tokens, 20);
        EXPECT_EQ(changed.state_row.at(0), checkpoint_blocks[0]);
        if (!host_cache) {
            EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks())
                << "every block is cache-only or free";
        }
    }
}

// A 26-token prompt's boundary 24 is a budget chunk end already: that chunk
// leaves the state in slot 1 and the next chunk's admission publishes it as
// a natural checkpoint. No forward carries a third block.
TEST_F(StateCheckpointInForwardSuite, ABoundaryAtAChunkEndIsPublishedFromTheLiveSlot) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    const std::vector<std::int32_t> tokens = MakeTokens(26);
    Submit(RequestWithTokens("seed", tokens));
    const std::vector<Chunk> chunks = RunPrefill("seed", 26);
    ASSERT_EQ(chunks.size(), 4u);
    for (const Chunk& chunk : chunks) {
        EXPECT_EQ(chunk.checkpoint_len, 0);
        EXPECT_EQ(chunk.state_rows[0].size(), 2u);
    }
    EXPECT_EQ(chunks[3].prefix_tokens, 24);
    SendFinish("seed");
    AckWriteBacks(PlanOnce());

    const Resume resent = Probe("resent", tokens);
    EXPECT_EQ(resent.prefix_tokens, 24);
    EXPECT_EQ(resent.state_row.at(0), chunks[2].state_rows[0][1]) << "the live slot of the chunk ending at 24";
}

// Multi-turn traffic where each turn keeps all but the previous prompt's
// final token, as when re-rendering the previous reply re-tokenizes the end
// of its generation prompt. turn2 resumes at the seed's last full page 24,
// and its rest takes one 6-token forward that also writes turn2's own
// checkpoint at 28. Its first decode publishes that with the prompt end, so
// turn3 resumes at 28 while turn2 is still decoding.
TEST_F(StateCheckpointInForwardSuite, EachTurnResumesAtThePreviousPromptsLastFullPage) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    const std::vector<std::int32_t> seed = MakeTokens(26);
    Submit(RequestWithTokens("seed", seed));
    RunPrefill("seed", 26);
    SendFinish("seed");
    AckWriteBacks(PlanOnce());

    std::vector<std::int32_t> turn2(seed.begin(), seed.end() - 1);
    for (std::int32_t i = 0; i < 5; ++i) {
        turn2.push_back(500 + i);
    }
    Submit(RequestWithTokens("turn2", turn2));
    const std::vector<Chunk> chunks = RunPrefill("turn2", 30);
    ASSERT_EQ(chunks.size(), 1u) << "one forward";
    EXPECT_EQ(chunks[0].prefix_tokens, 24) << "turn2 resumes at the seed's last full page";
    EXPECT_EQ(chunks[0].tokens, 6);
    EXPECT_EQ(chunks[0].checkpoint_len, 4);
    ASSERT_EQ(chunks[0].state_rows[0].size(), 3u);
    const std::int32_t checkpoint_block = chunks[0].state_rows[0][2];

    ExecutionPlan plan;
    PlanAndStateRow("turn2", &plan);
    SendForwardDone("turn2", {901});
    std::vector<std::int32_t> turn3(turn2.begin(), turn2.end() - 1);
    for (std::int32_t i = 0; i < 3; ++i) {
        turn3.push_back(700 + i);
    }
    const Resume resumed = Probe("turn3", turn3);
    EXPECT_EQ(resumed.prefix_tokens, 28) << "turn3 resumes at turn2's last full page";
    EXPECT_EQ(resumed.state_row.at(0), checkpoint_block);

    // The probe planned a second decode of turn2; it lands before the finish.
    SendForwardDone("turn2", {902});
    SendFinish("turn2");
    AckWriteBacks(PlanOnce());
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks())
        << "every block is cache-only or free";
}

// A chunk that starts inside a page -- a hit on a published prompt-end tail
// -- has no boundary whole pages past its start. The 30-token prompt's first
// chunk would span its boundary 28 from 22, so it ends there instead, as a
// cut: the next chunk's admission publishes its live slot at 28.
TEST_F(StateCheckpointInForwardSuite, AChunkSpanningTheBoundaryFromInsideAPageEndsAtIt) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    const std::vector<std::int32_t> seed = MakeTokens(22);
    Submit(RequestWithTokens("seed", seed));
    RunPrefill("seed", 22);
    SendFinish("seed");
    AckWriteBacks(PlanOnce());

    std::vector<std::int32_t> longer = seed;
    for (std::int32_t i = 0; i < 8; ++i) {
        longer.push_back(600 + i);
    }
    Submit(RequestWithTokens("longer", longer));
    const std::vector<Chunk> chunks = RunPrefill("longer", 30);
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0].prefix_tokens, 22) << "resumes at the seed's prompt-end tail";
    EXPECT_EQ(chunks[0].tokens, 6) << "ends at the boundary";
    EXPECT_EQ(chunks[1].tokens, 2);
    for (const Chunk& chunk : chunks) {
        EXPECT_EQ(chunk.checkpoint_len, 0);
        EXPECT_EQ(chunk.state_rows[0].size(), 2u);
    }
    SendFinish("longer");
    AckWriteBacks(PlanOnce());

    const Resume resent = Probe("resent", longer);
    EXPECT_EQ(resent.prefix_tokens, 28);
    EXPECT_EQ(resent.state_row.at(0), chunks[0].state_rows[0][1]) << "the live slot of the chunk ending at 28";
}

// The P role computes prompts for a remote decode and publishes at its
// remote decode's admission and its finish: the 22-token prompt's forward
// [16, 22) writes the checkpoint at 20, and the same prompt sent again to
// this prefill engine resumes there while the first is still handing off its
// KV (before the PD ACK), as a follow-up does while the first decodes.
class StateCheckpointInForwardPrefillRoleSuite : public StateCheckpointInForwardSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointInForwardSuite::MakeConfig();
        cfg.role = Role::kP;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.transfer_policy = group.attn_kind == AttnKind::kState ? CacheTransferPolicy::LatestSnapshot
                                                                        : CacheTransferPolicy::FullSuffix;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(StateCheckpointInForwardPrefillRoleSuite, TheNextPromptResumesAtTheCheckpointTheRemoteDecodePublished) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    const std::vector<std::int32_t> tokens = MakeTokens(22);
    Submit(RequestWithTokens("seed", tokens));
    SendBootstrapped("seed");
    const std::vector<Chunk> chunks = RunPrefill("seed", 22);
    ASSERT_EQ(chunks.size(), 3u);
    EXPECT_EQ(chunks[2].checkpoint_len, 4);
    ASSERT_EQ(chunks[2].state_rows[0].size(), 3u);
    const ExecutionPlan handoff = PlanOnce();
    ASSERT_TRUE(handoff.remote_decode.has_value());
    AckWriteBacks(handoff);

    Submit(RequestWithTokens("resent", tokens));
    SendBootstrapped("resent");
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"resent"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{20});
    EXPECT_EQ(StateRow(*batch, "state0", "resent").at(0), chunks[2].state_rows[0][2])
        << "resumes from the in-forward checkpoint before the seed's PD ACK";
    AckWriteBacks(plan);
    SendPdSucceeded("seed");
    AckWriteBacks(PlanOnce());
}

// Mixed batches beside one decoding request, which takes one token of every
// round's 8 and leaves 7 for the prefill.
class StateCheckpointInForwardMixedSuite : public StateCheckpointInForwardSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointInForwardSuite::MakeConfig();
        cfg.enable_mixed_prefill_decode = true;
        return cfg;
    }

    void StartDecoding() {
        Submit(RequestWithTokens("decoding", {501, 502, 503}));
        AckWriteBacks(PlanOnce());
        SendForwardDone("decoding", {800});
    }

    struct MixedPrefill {
        std::vector<std::pair<std::int32_t, std::int32_t>> chunks;
        std::int32_t checkpoint_len{0};
        std::int32_t checkpoint_block{0};
    };

    // Plans `id`'s prefill beside the decode to its end, landing every round.
    MixedPrefill RunPrefillBesideTheDecode(const std::string& id, std::int32_t prompt_tokens) {
        MixedPrefill out;
        for (std::int32_t round = 0; round < 12; ++round) {
            const ExecutionPlan plan = PlanOnce();
            const ForwardBatch* batch = FindForwardBatch(plan);
            ASSERT_TRUE_OR_RETURN(batch != nullptr && batch->request_ids.size() == 2u, out);
            const auto row =
                static_cast<std::size_t>(std::ranges::find(batch->request_ids, id) - batch->request_ids.begin());
            ASSERT_TRUE_OR_RETURN(row < 2u, out);
            const std::int32_t start = batch->extend_prefix_lens.at(row);
            const std::int32_t end = start + batch->input_lengths.at(row);
            out.chunks.emplace_back(start, end);
            if (batch->extend_state_checkpoint_lens.at(row) > 0) {
                out.checkpoint_len = batch->extend_state_checkpoint_lens.at(row);
                out.checkpoint_block = StateRow(*batch, "state0", id).at(2);
            }
            AckWriteBacks(plan);
            SendForwardDone("decoding", {801 + round});
            SendForwardDone(id, end == prompt_tokens ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
            if (end == prompt_tokens) {
                break;
            }
        }
        return out;
    }
};

// Plain chunks [0, 7), [7, 14), [14, 21) of a 22-token prompt would span its
// boundary 20 from inside a page and write nothing. A chunk ending within a
// chunk of the boundary ends on a page instead -- [7, 12), [12, 16) -- so the
// chunk [16, 22) writes the checkpoint 4 tokens in.
TEST_F(StateCheckpointInForwardMixedSuite, ChunksNearTheBoundaryEndOnAPageUnderABudgetCutShort) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    StartDecoding();
    const std::vector<std::int32_t> tokens = MakeTokens(22);
    Submit(RequestWithTokens("seed", tokens));
    const MixedPrefill prefill = RunPrefillBesideTheDecode("seed", 22);
    EXPECT_EQ(prefill.chunks,
              (std::vector<std::pair<std::int32_t, std::int32_t>>{{0, 7}, {7, 12}, {12, 16}, {16, 22}}));
    EXPECT_EQ(prefill.checkpoint_len, 4);
    SendFinish("seed");
    AckWriteBacks(PlanOnce());

    const Resume resent = Probe("resent", tokens);
    EXPECT_EQ(resent.prefix_tokens, 20);
    EXPECT_EQ(resent.state_row.at(0), prefill.checkpoint_block) << "resumes from the in-forward checkpoint";
}

// A junction before the boundary fixes where a later chunk starts: the
// 22-token prompt shares a cached 24-token prompt's first four pages but no
// state checkpoint below them, so it resumes at 0 with the junction at 16.
// [7, 14) keeps its budget rather than end on the page at 12, the junction
// cuts [14, 16), and the chunk [16, 22) writes the checkpoint.
TEST_F(StateCheckpointInForwardMixedSuite, AJunctionBeforeTheBoundaryLeavesTheChunksBeforeItWhole) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    const std::vector<std::int32_t> cached = MakeTokens(24);
    Submit(RequestWithTokens("cached", cached));
    RunPrefill("cached", 24);
    SendFinish("cached");
    AckWriteBacks(PlanOnce());

    StartDecoding();
    std::vector<std::int32_t> tokens(cached.begin(), cached.begin() + 16);
    for (std::int32_t i = 0; i < 6; ++i) {
        tokens.push_back(700 + i);
    }
    Submit(RequestWithTokens("seed", tokens));
    const MixedPrefill prefill = RunPrefillBesideTheDecode("seed", 22);
    EXPECT_EQ(prefill.chunks,
              (std::vector<std::pair<std::int32_t, std::int32_t>>{{0, 7}, {7, 14}, {14, 16}, {16, 22}}));
    EXPECT_EQ(prefill.checkpoint_len, 4);
}

// The checkpoint's block is an extra the pool may not hold right now. In the
// smallest pool that fits the 22-token prompt and its 32-token budget, the
// chunk spanning the boundary instead ends at it, as a cut would: one more
// round, and the next chunk's admission publishes that chunk's live slot at
// 20, which the same prompt sent again resumes from.
class StateCheckpointInForwardTightPoolSuite : public StateCheckpointInForwardSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointInForwardSuite::MakeConfig();
        cfg.device_allocator.total_pages = 22;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }
};

TEST_F(StateCheckpointInForwardTightPoolSuite, AChunkEndsAtTheBoundaryWhenTheCheckpointBlockDoesNotFit) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    const std::vector<std::int32_t> tokens = MakeTokens(22);
    Submit(RequestWithTokens("seed", tokens));
    const std::vector<Chunk> chunks = RunPrefill("seed", 22);
    ASSERT_EQ(chunks.size(), 4u) << "no room for the checkpoint's block: a cut at 20";
    EXPECT_EQ(chunks[2].prefix_tokens, 16);
    EXPECT_EQ(chunks[2].tokens, 4);
    EXPECT_EQ(chunks[3].tokens, 2);
    for (const Chunk& chunk : chunks) {
        EXPECT_EQ(chunk.checkpoint_len, 0);
        EXPECT_EQ(chunk.state_rows[0].size(), 2u);
    }
    SendFinish("seed");
    AckWriteBacks(PlanOnce());

    const Resume resent = Probe("resent", tokens);
    EXPECT_EQ(resent.prefix_tokens, 20);
    EXPECT_EQ(resent.state_row.at(0), chunks[2].state_rows[0][1]) << "the live slot of the chunk ending at 20";
}

// A retraction right after the forward that wrote the checkpoint is the
// request's next publication point: it publishes the checkpoint, and the
// suspended request holds no Device block. With a 6-token replay tail the
// 26-token prompt's boundary is 20, inside the chunk [16, 24); the knob
// retracts the prefilling request at the fourth plan, once that chunk landed.
class StateCheckpointInForwardRetractionSuite : public StateCheckpointInForwardSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointInForwardSuite::MakeConfig();
        cfg.prefix_replay_tokens = 6;
        cfg.debug_force_retraction_interval = -4;
        return cfg;
    }
};

TEST_F(StateCheckpointInForwardRetractionSuite, ARetractionPublishesTheCheckpointAndHoldsNoDeviceBlock) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    const std::vector<std::int32_t> tokens = MakeTokens(26);
    Submit(RequestWithTokens("seed", tokens));
    std::int32_t checkpoint_block = 0;
    for (const std::int32_t end : {8, 16, 24}) {
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{"seed"});
        ASSERT_EQ(batch->extend_prefix_lens.at(0) + batch->input_lengths.at(0), end);
        if (end == 24) {
            EXPECT_EQ(batch->extend_state_checkpoint_lens.at(0), 4) << "the state after position 16 + 4";
            const std::vector<std::int32_t> row = StateRow(*batch, "state0", "seed");
            ASSERT_EQ(row.size(), 3u);
            checkpoint_block = row[2];
        }
        SendForwardDone("seed");
    }

    const ExecutionPlan retraction = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    AckImageStores(retraction);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks())
        << "the suspended request holds no Device block; the index keeps the checkpoint";

    const ExecutionPlan readmit = PlanOnce();
    ASSERT_NE(FindRestore(readmit), nullptr);
    AckRestores(readmit);
    const ExecutionPlan last = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(last);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{24});
    EXPECT_EQ(batch->extend_state_checkpoint_lens, std::vector<std::int32_t>{0});
    SendForwardDone("seed", {900});
    SendFinish("seed");
    AckWriteBacks(PlanOnce());

    const Resume resent = Probe("resent", tokens);
    EXPECT_EQ(resent.prefix_tokens, 20);
    EXPECT_EQ(resent.state_row.at(0), checkpoint_block) << "resumes from the checkpoint the retraction published";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks())
        << "every block is cache-only or free";
}

// A request that finishes right after its prefill (a one-token budget) never
// rotates: finish publishes slot 1 as the prefill wrote it -- the state after
// the prompt -- and the partial KV block in place, and a longer prompt resumes
// from exactly those blocks.
TEST_F(StateCheckpointSuite, FinishRightAfterPrefillPublishesThePrefillsLiveSlotInPlace) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    Submit(RequestSpec{.request_id = "r", .tokens = MakeTokens(7), .max_new_tokens = 1});
    const ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(prefill);
    ASSERT_NE(batch, nullptr);
    const std::vector<std::int32_t> state_row = StateRow(*batch, "state0", "r");
    const std::vector<std::int32_t> full_row = batch->block_tables.at("full").at(0);
    ASSERT_EQ(state_row.size(), 2u);
    ASSERT_EQ(full_row.size(), 2u);  // [0,4) [4,8): seven prompt tokens and the decode slot
    SendForwardDone("r", {101});
    SendFinish("r");  // no decode step ran: computed tokens 7, frontier inside block 1
    PlanOnce();
    EXPECT_EQ(scheduler_->TailEntries(), 1u);

    // [1..7, 77] resumes at 7: slot 0 is the block the prefill wrote as its
    // live slot, and the KV tail copy is made from the prefill's own block.
    Submit(RequestWithTokens("hit", {1, 2, 3, 4, 5, 6, 7, 77}));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* hit = FindForwardBatch(plan);
    ASSERT_NE(hit, nullptr);
    EXPECT_EQ(hit->extend_prefix_lens, std::vector<std::int32_t>{7});
    EXPECT_EQ(StateRow(*hit, "state0", "hit").at(0), state_row.at(1)) << "the prefill's live slot, not its input";
    ASSERT_EQ(plan.pages_to_copy.at("full").size(), 1u);
    EXPECT_EQ(plan.pages_to_copy.at("full").front().first, full_row.at(1)) << "the partial block as written";
    EXPECT_EQ(hit->block_tables.at("full").at(0).at(0), full_row.at(0));
}

// A finish lands on the sampled token while, under the overlap schedule, the
// decode consuming that token is already dispatched. The request parks in
// Finishing with its pages; nothing is published at 8, where the live slot is
// being written. When that decode's result lands the frontier is 9 (102
// consumed, 103 the unconsumed sample) and the request settles there in the
// same advance: the state the decode wrote, the KV tail [8, 9), the pages
// released.
TEST_F(StateCheckpointSuite, FinishWithAForwardInFlightSettlesAtTheFrontierThatForwardLands) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    Submit(RequestSpec{.request_id = "r", .tokens = MakeTokens(7), .max_new_tokens = 8});
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    SendForwardDone("r", {101});
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);  // first decode: publishes the state at 7, rotates
    SendForwardDone("r", {102});
    const ExecutionPlan second = PlanOnce();  // second decode, in flight: writes slot 1 in place
    const ForwardBatch* batch = FindForwardBatch(second);
    ASSERT_NE(batch, nullptr);
    const std::vector<std::int32_t> live = StateRow(*batch, "state0", "r");
    const std::int32_t held = scheduler_->AvailableLcmBlocks();
    SendFinish("r");  // decided on 102, which the decode in flight consumes
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), held) << "the pages stay held while a forward is out";
    EXPECT_EQ(scheduler_->TailEntries(), 1u) << "only the prompt end's tail; nothing published at 8";

    SendForwardDone("r", {103});  // lands: the request settles at 9
    EXPECT_EQ(scheduler_->TailEntries(), 2u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks())
        << "every block is cache-only or free";
    const Resume resume = Probe("hit", {1, 2, 3, 4, 5, 6, 7, 101, 102, 77});
    EXPECT_EQ(resume.prefix_tokens, 9);
    EXPECT_EQ(resume.state_row.at(0), live.at(1)) << "the live slot the in-flight decode wrote";
}

// A finish with no forward out settles with the Finish event itself, as it
// always did: the frontier's tail and state are published and the pages
// released before the next plan.
TEST_F(StateCheckpointSuite, AQuiescentFinishSettlesWithTheFinishEvent) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    Submit(RequestWithTokens("r", MakeTokens(6)));
    PlanOnce();
    SendForwardDone("r", {101});
    PlanOnce();  // first decode
    SendForwardDone("r", {102});
    EXPECT_EQ(scheduler_->TailEntries(), 1u);
    SendFinish("r");
    EXPECT_EQ(scheduler_->TailEntries(), 2u) << "the frontier's tail";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks());
    EXPECT_EQ(Probe("hit", {1, 2, 3, 4, 5, 6, 101, 77}).prefix_tokens, 7);
}

// An abort of a finishing request releases its pages without publishing.
TEST_F(StateCheckpointSuite, AnAbortWhileFinishingPublishesNothing) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    Submit(RequestWithTokens("r", MakeTokens(6)));
    PlanOnce();
    SendForwardDone("r", {101});
    PlanOnce();  // decode in flight
    SendFinish("r");
    SendAbortEvent("r");
    EXPECT_EQ(scheduler_->TailEntries(), 1u) << "the prompt end only";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks());
    EXPECT_EQ(Probe("miss", {1, 2, 3, 4, 5, 6, 101, 77}).prefix_tokens, 6);
}

TEST_F(StateCheckpointSuite, AbortReleasesBothSlotsAndPublishesNothing) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    const std::int32_t empty = scheduler_->EmptyLcmBlocks();
    Submit(RequestWithTokens("r", MakeTokens(6)));
    PlanOnce();
    SendForwardDone("r", {101});
    PlanOnce();  // first decode: prompt end published (page + tail + state)
    SendForwardDone("r", {102});
    PlanOnce();  // slot 0 released
    SendAbortEvent("r");
    PlanOnce();
    // Only the published entries stay: page [1 2 3 4], the tail [5 6] in the
    // full group, and the prompt-end state in each state group.
    EXPECT_EQ(scheduler_->EmptyLcmBlocks(), empty - 2 - 3);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), empty) << "all of them evictable";
}

TEST_F(StateCheckpointSuite, SpeculativeDecodeWritesTheLiveSlotInPlace) {
    Reset(/*host_cache=*/false, /*decode_width=*/3);
    Submit(RequestWithTokens("r", MakeTokens(6)));
    PlanOnce();
    SendForwardDone("r", {101});
    const std::vector<std::int32_t> first = PlanAndStateRow("r");
    ASSERT_GT(first[0], 0);
    ASSERT_GT(first[1], 0);
    std::int32_t token = 102;
    for (const std::int32_t accepted : {3, 1, 2, 3}) {
        std::vector<std::int32_t> tokens;
        for (std::int32_t i = 0; i < accepted; ++i) {
            tokens.push_back(token++);
        }
        SendForwardDone("r", tokens);
        ExecutionEvent reserve;
        reserve.With(
            forward::UpdateReserveNumTokens{.request_id = "r", .reserve_num_tokens_in_next_schedule_event = 3});
        scheduler_->Advance(reserve);
        EXPECT_EQ(PlanAndStateRow("r"), (std::vector<std::int32_t>{0, first[1]}))
            << "verify widths grow the KV table, never the state table";
    }
}

TEST_F(StateCheckpointSuite, AStateCheckpointLeavesWithItsKvTail) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    Submit(RequestWithTokens("r", MakeTokens(6)));
    PlanOnce();
    SendForwardDone("r", {101});
    PlanOnce();  // prompt end: tail [5 6] and the state at 6 published
    SendForwardDone("r", {102});
    SendFinish("r");
    PlanOnce();
    ASSERT_EQ(scheduler_->TailEntries(), 2u);
    EXPECT_EQ(Probe("hit", {1, 2, 3, 4, 5, 6, 77}).prefix_tokens, 6);

    // Flooding the pool evicts the tails' KV blocks; the states published
    // with them go too, so nothing orphaned is ever matched.
    for (std::int32_t i = 0; i < 40; ++i) {
        const std::string id = "flood" + std::to_string(i);
        Submit(RequestWithTokens(id, MakeTokens(24, 1000 * (i + 1))));
        PlanOnce();
        SendForwardDone(id);
        PlanOnce();
        SendForwardDone(id);
        PlanOnce();
        SendForwardDone(id, {7});
        SendFinish(id);
        PlanOnce();
    }
    EXPECT_EQ(scheduler_->TailEntries(), 0u) << "r's tails were evicted with their blocks";
    EXPECT_EQ(Probe("miss", {1, 2, 3, 4, 5, 6, 77}).prefix_tokens, 0);
}

TEST_F(StateCheckpointSuite, AHostCheckpointLoadsIntoSlotZero) {
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    Submit(RequestWithTokens("r", MakeTokens(6)));
    AckWriteBacks(PlanOnce());
    SendForwardDone("r", {101});
    AckWriteBacks(PlanOnce());  // prompt end: page, tail and the three states stream to Host
    SendForwardDone("r", {102});
    SendFinish("r");
    AckWriteBacks(PlanOnce());
    EXPECT_GE(scheduler_->HostPoolCachedBlocks(), 1 + 1 + 3);
    ASSERT_TRUE(scheduler_->ClearL1Cache());

    // The Host tier matches the prompt-end checkpoint: the state loads into a
    // fresh slot 0 and the KV tail into private pages.
    const Resume host_hit = Probe("hit", {1, 2, 3, 4, 5, 6, 77});
    EXPECT_EQ(host_hit.prefix_tokens, 6);
    EXPECT_GT(host_hit.state_row.at(0), 0);
    EXPECT_GT(host_hit.state_row.at(1), 0);
    EXPECT_EQ(host_hit.load_rows, 1 + 1 + 3) << "page, tail block and one state per group";
}

// The Host-suffix rule (a hit is a Device KV prefix plus a Host suffix) through
// the scheduler. A Host hit publishes the KV pages it loads on the Device but
// keeps its loaded checkpoint private, so the next prompt finds the KV of h0,
// h1 on the Device and the checkpoint at 8 on the Host alone. It resumes at 8
// and loads only the three checkpoints -- the Device's pages are claimed, not
// reloaded -- and counts as a Host win that loads no KV and gains the whole
// hit.
TEST_F(StateCheckpointSuite, ADeviceKvHitLoadsOnlyTheHostCheckpoint) {
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    Submit(RequestWithTokens("r", MakeTokens(8)));
    AckWriteBacks(PlanOnce());
    SendForwardDone("r", {101});
    AckWriteBacks(PlanOnce());  // prompt end: two pages and the three states stream to Host
    SendForwardDone("r", {102});
    SendFinish("r");
    AckWriteBacks(PlanOnce());
    ASSERT_TRUE(scheduler_->ClearL1Cache());

    std::vector<std::int32_t> prompt = MakeTokens(8);
    prompt.push_back(77);
    const Resume warm = Probe("warm", prompt);
    EXPECT_EQ(warm.prefix_tokens, 8);
    EXPECT_EQ(warm.load_rows, 2 + 3) << "two pages and one state per group";
    const HostCacheStats before = scheduler_->GetHostCacheStats();
    const Resume hit = Probe("hit", prompt);
    const HostCacheStats after = scheduler_->GetHostCacheStats();
    EXPECT_EQ(hit.prefix_tokens, 8);
    EXPECT_GT(hit.state_row.at(0), 0);
    EXPECT_EQ(hit.load_rows, 3) << "one state per group";
    EXPECT_EQ(after.prefix_loads.ops - before.prefix_loads.ops, 1u);
    EXPECT_EQ(after.prefix_hits.host_hit_tokens - before.prefix_hits.host_hit_tokens, 0u);
    EXPECT_EQ(after.prefix_hits.host_gain_tokens - before.prefix_hits.host_gain_tokens, 8u);
}

// A Host hit publishes its extension pages on the Device; the next prompt
// resumes on them and loads only the Host pages and checkpoints beyond them
// (BoundaryTreeCoordinatorTest.AHostExtensionKeepsItsPagesLinkedToTheirParents
// checks the tree nodes they leave).
TEST_F(StateCheckpointSuite, AHostHitResumesOnAnEarlierHostHitsPages) {
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    // r1's prompt end publishes pages 0 and 1 with the checkpoint at 8; r2
    // resumes there and its prompt end publishes pages 2 and 3 with the
    // checkpoint at 16. All of it streams to the Host.
    for (const std::int32_t prompt_tokens : {8, 16}) {
        const std::string id = "r" + std::to_string(prompt_tokens);
        Submit(RequestWithTokens(id, MakeTokens(prompt_tokens)));
        AckWriteBacks(PlanOnce());
        SendForwardDone(id, {101});
        AckWriteBacks(PlanOnce());  // prompt end
        SendForwardDone(id, {102});
        SendFinish(id);
        AckWriteBacks(PlanOnce());
    }
    ASSERT_TRUE(scheduler_->ClearL1Cache());

    // A Host hit at 8 brings pages 0 and 1 back to the Device; the next
    // prompt resumes at 16 on them plus the Host's pages 2 and 3, which its
    // admission publishes on the Device.
    std::vector<std::int32_t> prompt = MakeTokens(8);
    prompt.push_back(77);
    ASSERT_EQ(Probe("warm", prompt).prefix_tokens, 8);
    prompt = MakeTokens(16);
    prompt.push_back(77);
    const Resume hit = Probe("hit", prompt);
    EXPECT_EQ(hit.prefix_tokens, 16);
    EXPECT_EQ(hit.load_rows, 2 + 3) << "pages 2 and 3 and one state per group";
}

class StateOnlySuite : public StateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointSuite::MakeConfig();
        std::erase_if(cfg.cache_groups, [](const CacheGroupConfig& group) { return group.group_id == "full"; });
        return cfg;
    }
};

// A model with no KV group has no KV boundary to cut at: a prompt a page
// boundary could be matched at is not a junction, and the first chunk runs
// to the budget.
TEST_F(StateOnlySuite, NoKvGroupMeansNoJunctionCut) {
    Reset(/*host_cache=*/false, /*decode_width=*/1);
    Submit(RequestWithTokens("r", MakeTokens(7)));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->input_lengths, std::vector<std::int32_t>{7});
    const std::vector<std::int32_t> row = StateRow(*batch, "state0", "r");
    EXPECT_GT(row.at(0), 0) << "a zeroed block is the first chunk's input";
    EXPECT_EQ(std::ranges::count(plan.pages_to_zero.at("state0"), row.at(0)), 1);
    SendForwardDone("r", {101});
    SendFinish("r");
    PlanOnce();

    // The prompt-end checkpoint at 7 is the only one: a longer prompt sharing
    // the seven tokens resumes there, one sharing fewer resumes at zero.
    EXPECT_EQ(Probe("hit", {1, 2, 3, 4, 5, 6, 7, 88, 89}).prefix_tokens, 7);
    EXPECT_EQ(Probe("shorter", {1, 2, 3, 4, 5, 66, 67}).prefix_tokens, 0);
}

// A pool sized at two blocks per State group per live request: four one-page
// prompts fill it (2 History pages + 3 x 2 state blocks each).
class StateCapacitySuite : public StateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointSuite::MakeConfig();
        cfg.device_allocator.total_pages = 1 + 4 * (2 + 3 * 2);
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        cfg.max_batch_size = 8;
        cfg.decode_input_tokens = 1;
        return cfg;
    }
};

// Admission back-pressures instead of wedging: the zero state is a block the
// request already holds, so the first decode's rotation needs nothing from
// the pool, and every request runs to its end.
TEST_F(StateCapacitySuite, PromptsFillingThePoolNeverWedgeAtTheirFirstDecode) {
    constexpr std::int32_t kRequests = 6;
    constexpr std::int32_t kMaxNewTokens = 4;
    std::vector<RequestSpec> specs;
    for (std::int32_t i = 0; i < kRequests; ++i) {
        specs.push_back(RequestSpec{.request_id = "r" + std::to_string(i),
                                    .tokens = MakeTokens(kP, 100 * (i + 1)),
                                    .max_new_tokens = kMaxNewTokens});
    }
    Submit(specs);
    std::map<std::string, std::int32_t> generated;
    std::set<std::string> finished;
    std::int32_t idle_rounds = 0;
    for (std::int32_t round = 0; round < 200 && finished.size() < specs.size() && idle_rounds < 10; ++round) {
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        if (batch == nullptr || batch->request_ids.empty()) {
            ++idle_rounds;
            continue;
        }
        idle_rounds = 0;
        for (const std::string& id : batch->request_ids) {
            const std::int32_t n = ++generated[id];
            SendForwardDone(id, {1000 + n});
            if (n >= kMaxNewTokens) {
                SendFinish(id);
                finished.insert(id);
            }
        }
    }
    EXPECT_EQ(finished.size(), specs.size()) << "every request finished";
    PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

class StateRetractionSuite : public StateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointSuite::MakeConfig();
        cfg.disable_l2_cache = false;
        cfg.debug_force_retraction_interval = 4;
        return cfg;
    }
};

TEST_F(StateRetractionSuite, TheImageCarriesBothSlotsAndRestoresTheLiveOneAsACopy) {
    Submit(RequestWithTokens("r", MakeTokens(6)));
    AckWriteBacks(PlanOnce());  // plan 1: prefill
    SendForwardDone("r", {101});
    ExecutionPlan plan;
    const std::vector<std::int32_t> first = PlanAndStateRow("r", &plan);  // plan 2: first decode
    SendForwardDone("r", {102});
    const std::vector<std::int32_t> second = PlanAndStateRow("r", &plan);  // plan 3: slot 0 released
    EXPECT_EQ(second, (std::vector<std::int32_t>{0, first[1]}));
    SendForwardDone("r", {103});

    // Plan 4 retracts r = [1..6, 101, 102, 103], frontier 8: the page
    // [101 102] completes, the state the decodes wrote is published as the
    // checkpoint at 8 and rides L2 like the KV pages; the pool carries only
    // the blob.
    const ExecutionPlan retraction = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    const SnapshotStoreBatch* store = FindSnapshotStore(retraction);
    ASSERT_NE(store, nullptr);
    EXPECT_TRUE(store->src_pages.at(0).empty());
    AckImageStores(retraction);

    const ExecutionPlan readmit = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr);
    ASSERT_EQ(restore->request_ids, std::vector<std::string>{"r"});
    // Every state row is slot 1 restored as a private copy: an unkeyed L2
    // row, since the resumed request keeps writing it.
    std::int32_t state_rows = 0;
    for (std::size_t i = 0; i < restore->group_ids.at(0).size(); ++i) {
        if (restore->group_ids.at(0)[i] == 0) {
            continue;
        }
        ++state_rows;
        EXPECT_EQ(restore->source_tiers.at(0)[i], static_cast<std::uint8_t>(HostTier::kL2));
        EXPECT_TRUE(restore->content_hashes.at(0)[i].empty());
    }
    EXPECT_EQ(state_rows, 3);
    AckRestores(readmit);
    const std::vector<std::int32_t> resumed = PlanAndStateRow("r");
    EXPECT_EQ(resumed[0], 0);
    EXPECT_GT(resumed[1], 0);
    EXPECT_NE(resumed[1], first[1]) << "a fresh block, not the published one";
    SendForwardDone("r", {104});
    EXPECT_EQ(PlanAndStateRow("r"), resumed);
    SendForwardDone("r", {105});
    EXPECT_EQ(Probe("hit", {1, 2, 3, 4, 5, 6, 101, 102, 77}).prefix_tokens, 8) << "the retraction's checkpoint";
}

// A restore whose planned victim is a tail block cascades to the state
// checkpoint published under the same hash -- the very block the image
// resumes on. Claiming before evicting pins it, as an admission's hit is.
class StateRestoreClaimSuite : public StateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointSuite::MakeConfig();
        cfg.disable_l2_cache = false;
        cfg.debug_force_retraction_interval = 3;
        // One full and one State group over six usable blocks: the retraction
        // leaves five cache-only entries and one empty block, fewer than the
        // restore's two fresh ones.
        std::erase_if(cfg.cache_groups, [](const CacheGroupConfig& group) {
            return group.group_id == "state1" || group.group_id == "state2";
        });
        cfg.device_allocator.total_pages = 7;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }
};

TEST_F(StateRestoreClaimSuite, TheRestoreClaimsTheCheckpointBeforeEvictingItsTailCompanions) {
    Reset(/*host_cache=*/true, /*decode_width=*/2);
    // r = [1..6]: plan 1 prefills; plan 2 (the first decode) publishes the tail
    // at 6 and the checkpoint at 6, which rotates into slot 0 while the tail's
    // block is copied in; plan 3 retracts r at frontier 7 and publishes the
    // tail at 7 and the checkpoint at 7. The image: page 0 and slot 0 (the
    // checkpoint at 6) are published slots the restore claims from the
    // Device; the tail block and slot 1 are writable and come back as copies.
    // The two-token decode width makes the full group need a growth block
    // beyond the tail copy, so it evicts within its own index.
    Submit(RequestSpec{.request_id = "r", .tokens = MakeTokens(6), .max_new_tokens = 2});
    AckWriteBacks(PlanOnce());
    SendForwardDone("r", {101});
    ExecutionPlan plan;
    const std::vector<std::int32_t> first = PlanAndStateRow("r", &plan);
    ASSERT_GT(first[0], 0) << "the checkpoint at 6 is the first decode's input";
    SendForwardDone("r", {102});
    const ExecutionPlan retraction = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    AckImageStores(retraction);
    ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 1);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 6)
        << "five cache-only entries: page 0, the two tails' full blocks, the two checkpoints";

    // The restore claims page 0 and the checkpoint at 6 and needs fresh blocks
    // for the tail copy, the full group's growth and the live slot: more than
    // the one empty, so each group evicts within its own index -- the full
    // group the tail at 6's block, the oldest. Its cascade reaches the state
    // entry at 6, which the restore has already claimed and pinned; evicting
    // first would have taken it, and slot 0 would have come back as a keyed
    // Host load the plan never counted a block for.
    const ExecutionPlan readmit = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr);
    ASSERT_EQ(restore->request_ids, std::vector<std::string>{"r"});
    std::int32_t keyed_state_rows = 0;
    std::int32_t unkeyed_state_rows = 0;
    for (std::size_t i = 0; i < restore->group_ids.at(0).size(); ++i) {
        if (restore->group_ids.at(0)[i] == 0) {
            continue;
        }
        (restore->content_hashes.at(0)[i].empty() ? unkeyed_state_rows : keyed_state_rows) += 1;
    }
    EXPECT_EQ(unkeyed_state_rows, 1) << "slot 1 is a private copy";
    EXPECT_EQ(keyed_state_rows, 0) << "slot 0 was claimed on the Device, not loaded";
    EXPECT_EQ(restore->group_ids.at(0).size(), 2u) << "the tail copy and the live slot";
    // Three blocks count as a restore's: the full group's tail block at 6,
    // whose claimed checkpoint stays and is not counted, the State group's
    // checkpoint at 7, and the tail block at 7 it takes along. The stream to
    // Host and the image's L2 leg left Host holding all three.
    const CacheCoordinator::DeviceEvictionCounts restored = scheduler_->GetHostCacheStats().device_evictions.restore;
    EXPECT_EQ(restored.blocks, 3u);
    EXPECT_EQ(restored.host_resident, 3u);
    AckRestores(readmit);
    // The resumed decode releases the claimed slot 0 and writes its copy of
    // slot 1 in place.
    const std::vector<std::int32_t> resumed = PlanAndStateRow("r");
    EXPECT_EQ(resumed[0], 0);
    EXPECT_GT(resumed[1], 0);
}

// The same geometry under Host write-back, where a victim set leaves whole:
// r = [1..7] is retracted at frontier 8, a page boundary. Its restore could
// claim page 0, page 1 (full, never written again) and the checkpoint at the
// tail 7 from the Device, and needs fresh blocks for the checkpoint at 8
// (written in place) and the decode reserve: two, with one empty. The cache
// left is the tail 7's full block, beside the checkpoint at 7, and the
// checkpoint at 8, beside page 1 -- each set holding a block the claims
// protect, so with the claims nothing could leave. The restore claims nothing
// then: every slot comes back as a copy and r's own cache leaves whole. r
// then runs to its end, and nothing is left pinned or in flight.
TEST_F(StateRestoreClaimSuite, ARetractionAtAPageBoundaryRestoresUnderWriteBack) {
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    config_.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    Submit(RequestSpec{.request_id = "r", .tokens = MakeTokens(7), .max_new_tokens = 4});
    AckWriteBacks(PlanOnce());  // plan 1: the prefill
    SendForwardDone("r", {101});
    AckWriteBacks(PlanOnce());  // plan 2: the first decode publishes the tail at 7 and its checkpoint
    SendForwardDone("r", {102});
    const ExecutionPlan retraction = PlanOnce();  // plan 3: the knob retracts r at frontier 8
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    AckImageStores(retraction);
    ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 1);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 6);

    const HostCacheStats before = scheduler_->GetHostCacheStats();
    const ExecutionPlan readmit = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr) << "the restore fits";
    EXPECT_EQ(restore->group_ids.at(0).size(), 4u) << "both pages and both State slots: nothing is claimed";
    const HostCacheStats after = scheduler_->GetHostCacheStats();
    EXPECT_EQ(after.device_evictions.restore.blocks - before.device_evictions.restore.blocks, 4u)
        << "r's own cache leaves";
    EXPECT_EQ(after.eviction_stores.ops, before.eviction_stores.ops + 1) << "what the image left on the Device alone";
    AckWriteBacks(readmit);
    AckRestores(readmit);

    std::int32_t token = 103;
    for (std::int32_t round = 0; round < 16 && token <= 104; ++round) {
        const ExecutionPlan plan = PlanOnce();
        AckImageStores(plan);
        AckRestores(plan);
        const ForwardBatch* batch = FindForwardBatch(plan);
        if (batch == nullptr || batch->request_ids.empty()) {
            continue;
        }
        SendForwardDone("r", {token});
        if (token++ == 104) {
            SendFinish("r");
        }
    }
    AckImageStores(PlanOnce());
    EXPECT_EQ(scheduler_->RequestTokenSize("r"), -1);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
    EXPECT_TRUE(scheduler_->CanClearCache());
}

// Retraction after the forward that wrote the in-forward checkpoint, before
// the request's next publication point: the retraction is that point, and the
// Retracted request holds no Device block (fsm::Retracted). Its image carries
// the rest; only the index keeps the published checkpoint.
class StateCheckpointInForwardRetractionPoolSuite : public StateCheckpointInForwardSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointInForwardSuite::MakeConfig();
        cfg.device_allocator.total_pages = pool_pages_;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        cfg.prefix_replay_tokens = prefix_replay_tokens_;
        cfg.debug_force_retraction_interval = force_retraction_interval_;
        cfg.state_checkpoint_in_forward = in_forward_;
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    struct Outcome {
        bool wrote_checkpoint{false};
        std::int32_t retractions{0};
        std::int32_t restores{0};
        bool finished{false};
        bool stalled{false};
    };

    // Runs one request until it finishes, or until ten rounds pass with no
    // forward, image store or restore: lands every forward (with a sampled
    // token at the prompt's end and at every decode), acknowledges every
    // store and restore, and finishes the request after `finish_after`
    // tokens.
    Outcome DriveOne(const RequestSpec& spec, std::int32_t finish_after) {
        Submit(spec);
        Outcome out;
        std::int32_t generated = 0;
        for (std::int32_t round = 0, idle = 0; round < 60 && !out.finished; ++round) {
            const ExecutionPlan plan = PlanOnce();
            const bool stores = FindSnapshotStore(plan) != nullptr;
            const bool restores = FindRestore(plan) != nullptr;
            out.retractions += stores ? 1 : 0;
            out.restores += restores ? 1 : 0;
            AckImageStores(plan);
            AckRestores(plan);
            const ForwardBatch* batch = FindForwardBatch(plan);
            const bool forwards = batch != nullptr && !batch->request_ids.empty();
            idle = forwards || stores || restores ? 0 : idle + 1;
            if (idle == 10) {
                out.stalled = true;
                break;
            }
            if (!forwards) {
                continue;
            }
            if (!batch->extend_prefix_lens.empty()) {
                out.wrote_checkpoint = out.wrote_checkpoint || batch->extend_state_checkpoint_lens.at(0) > 0;
                if (batch->extend_prefix_lens.at(0) + batch->input_lengths.at(0) < batch->prefill_lengths.at(0)) {
                    SendForwardDone(spec.request_id);
                    continue;
                }
            }
            SendForwardDone(spec.request_id, {5000 + ++generated});
            if (generated == finish_after) {
                SendFinish(spec.request_id);
                out.finished = true;
            }
        }
        AckImageStores(PlanOnce());
        return out;
    }

    // Every Device and snapshot block is free or an unpinned cache entry, and
    // no Host entry is pinned.
    void ExpectEveryBlockFreeOrCacheOnly() {
        EXPECT_EQ(scheduler_->RetractedSize(), 0u);
        EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
        EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks());
        EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), config_.snapshot_allocator.NumUsableBlocks());
        EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
    }

    std::int32_t pool_pages_{257};
    // An 8-token replay tail puts a 30-token prompt's boundary at 20, inside
    // its third chunk [16, 24), which is not its last.
    std::int32_t prefix_replay_tokens_{8};
    std::int32_t force_retraction_interval_{-4};
    bool in_forward_{true};
};

// The 30-token prompt's chunks are [0, 8), [8, 16), [16, 24) -- which writes
// the checkpoint at 20 -- and [24, 30). Plan 4 retracts it in Prefilling after
// [16, 24), so the retraction publishes the checkpoint; the restore resumes it
// at 24.
TEST_F(StateCheckpointInForwardRetractionPoolSuite, APrefillingVictimHoldsNoDeviceBlockWhileRetracted) {
    for (const bool host_cache : {false, true}) {
        SCOPED_TRACE(host_cache);
        Reset(host_cache, /*decode_width=*/1);
        Submit(RequestWithTokens("r", MakeTokens(30)));
        for (std::int32_t chunk = 0; chunk < 3; ++chunk) {
            const ExecutionPlan plan = PlanOnce();
            const ForwardBatch* batch = FindForwardBatch(plan);
            ASSERT_NE(batch, nullptr);
            ASSERT_EQ(batch->request_ids, std::vector<std::string>{"r"});
            EXPECT_EQ(batch->extend_prefix_lens.at(0), 8 * chunk);
            EXPECT_EQ(batch->input_lengths.at(0), 8);
            EXPECT_EQ(batch->extend_state_checkpoint_lens.at(0), chunk == 2 ? 4 : 0);
            AckWriteBacks(plan);
            SendForwardDone("r");
        }

        const ExecutionPlan retraction = PlanOnce();
        ASSERT_EQ(scheduler_->RetractedSize(), 1u) << "plan 4 retracts r in Prefilling";
        AckImageStores(retraction);
        EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
        EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks())
            << "a Retracted request holds no Device block, so every one is cache-only or free";

        const ExecutionPlan readmit = PlanOnce();
        ASSERT_NE(FindRestore(readmit), nullptr);
        AckRestores(readmit);
        const ExecutionPlan resumed = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(resumed);
        ASSERT_NE(batch, nullptr);
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{"r"});
        EXPECT_EQ(batch->extend_prefix_lens.at(0), 24);
        EXPECT_EQ(batch->input_lengths.at(0), 6);
        AckWriteBacks(resumed);
        SendForwardDone("r", {900});
        AckWriteBacks(PlanOnce());
        SendForwardDone("r", {901});
        SendFinish("r");
        AckWriteBacks(PlanOnce());
        ExpectEveryBlockFreeOrCacheOnly();

        std::vector<std::int32_t> changed = MakeTokens(30);
        changed[24] = 991;
        EXPECT_EQ(Probe("changed", changed).prefix_tokens, 20) << "the retraction published the checkpoint";
    }
}

// A Decoding victim: the 22-token prompt's last chunk [16, 22) writes the
// checkpoint at 20, the first decode publishes it, and plan 5 retracts r.
TEST_F(StateCheckpointInForwardRetractionPoolSuite, ADecodingVictimHoldsNoDeviceBlockWhileRetracted) {
    prefix_replay_tokens_ = 0;
    force_retraction_interval_ = 5;
    for (const bool host_cache : {false, true}) {
        SCOPED_TRACE(host_cache);
        Reset(host_cache, /*decode_width=*/1);
        Submit(RequestWithTokens("r", MakeTokens(22)));
        for (std::int32_t chunk = 0; chunk < 3; ++chunk) {
            const ExecutionPlan plan = PlanOnce();
            const ForwardBatch* batch = FindForwardBatch(plan);
            ASSERT_NE(batch, nullptr);
            EXPECT_EQ(batch->extend_state_checkpoint_lens.at(0), chunk == 2 ? 4 : 0);
            AckWriteBacks(plan);
            SendForwardDone("r", chunk == 2 ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
        }
        AckWriteBacks(PlanOnce());
        SendForwardDone("r", {901});

        const ExecutionPlan retraction = PlanOnce();
        ASSERT_EQ(scheduler_->RetractedSize(), 1u) << "plan 5 retracts r in Decoding";
        AckImageStores(retraction);
        EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
        EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks())
            << "a Retracted request holds no Device block, so every one is cache-only or free";

        const ExecutionPlan readmit = PlanOnce();
        ASSERT_NE(FindRestore(readmit), nullptr);
        AckRestores(readmit);
        AckWriteBacks(PlanOnce());
        SendForwardDone("r", {902});
        SendFinish("r");
        AckWriteBacks(PlanOnce());
        ExpectEveryBlockFreeOrCacheOnly();

        std::vector<std::int32_t> changed = MakeTokens(22);
        changed[21] = 991;
        EXPECT_EQ(Probe("changed", changed).prefix_tokens, 20);
    }
}

// A restore after one retraction re-reserves the whole remaining budget (one
// more 4096-token safe-step window than the first admission prepaid), and the
// capacity model promises that a request's limit fits an otherwise empty
// pool. In the smallest pool that admits a 30-token prompt with a 4108-token
// budget, the first admission leaves room for the checkpoint's three blocks
// beside the chunk [16, 24), and plan 4 retracts r after that chunk. Nothing
// else lives, so its restore must follow, with the checkpoint as without it.
TEST_F(StateCheckpointInForwardRetractionPoolSuite, ARetractedVictimIsRestoredInTheSmallestPoolThatAdmitsIt) {
    constexpr std::int32_t kPrompt = 30;
    constexpr std::int32_t kBudget = 4108;
    for (const bool in_forward : {false, true}) {
        SCOPED_TRACE(testing::Message() << "in_forward " << in_forward);
        in_forward_ = in_forward;
        for (pool_pages_ = (kPrompt + kBudget) / kP;; ++pool_pages_) {
            Reset(/*host_cache=*/false, /*decode_width=*/1);
            if (scheduler_->MaxSingleRequestTokens() >= kPrompt + kBudget) {
                break;
            }
        }
        const Outcome out =
            DriveOne(RequestSpec{.request_id = "r", .tokens = MakeTokens(kPrompt), .max_new_tokens = kBudget},
                     /*finish_after=*/2);
        EXPECT_EQ(out.wrote_checkpoint, in_forward);
        EXPECT_EQ(out.retractions, 1);
        EXPECT_FALSE(out.stalled) << "r is Retracted (RetractedSize " << scheduler_->RetractedSize()
                                  << ") and no request holds Device pages (ActiveLcmBlocks "
                                  << scheduler_->ActiveLcmBlocks() << "), yet only " << scheduler_->AvailableLcmBlocks()
                                  << " of " << config_.device_allocator.NumUsableBlocks()
                                  << " blocks are available to its restore";
        EXPECT_EQ(out.restores, 1);
        EXPECT_TRUE(out.finished);
        ExpectEveryBlockFreeOrCacheOnly();
    }
}

// ---------------------------------------------------------------------------
// The Host write-back policy on the P role, and on the fused role below
// (StateCheckpointFusedHostSuite): nothing streams at publication; what an
// admission or a restore evicts reaches Host L2 in the same plan, by whole
// segments, and publishes there once its copy lands. The full group packs four
// blocks per parent, each State group one. A 22-token session leaves 5 pages, a
// tail of 2 and the three checkpoints at its tail: 9 blocks in 2 + 3 parents.
// The Device has 8 parents, so a second prompt's first chunk (6 State parents)
// needs room the session holds.
// ---------------------------------------------------------------------------
class StateCheckpointPrefillHostSuite : public StateCheckpointInForwardPrefillRoleSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointInForwardPrefillRoleSuite::MakeConfig();
        cfg.role = role_;
        cfg.state_checkpoint_in_forward = in_forward_;
        cfg.device_allocator.total_pages = device_pages_;
        cfg.host_allocator.total_pages = host_pages_;
        cfg.disable_l2_cache = false;
        cfg.host_write_policy = write_policy_;
        cfg.enable_kv_cache_events = kv_events_;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            const bool is_state = group.attn_kind == AttnKind::kState;
            group.cache_blocks_per_lcm_block = is_state ? 1 : 4;
            group.total_pages = (cfg.device_allocator.total_pages - 1) * group.cache_blocks_per_lcm_block + 1;
            group.shard_count = is_state ? 1 : kv_shard_count_;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // A P request from submission to its PD ACK: bootstrap, prefill (each
    // plan's write-backs acknowledged at once), the remote decode that hands
    // it off, the ACK that finishes it and publishes its prompt end.
    // Returns the hand-off plan.
    ExecutionPlan CompleteSession(const std::string& id, const std::vector<std::int32_t>& tokens) {
        Submit(RequestWithTokens(id, tokens));
        SendBootstrapped(id);
        RunPrefill(id, static_cast<std::int32_t>(tokens.size()));
        ExecutionPlan handoff = PlanOnce();
        EXPECT_TRUE(handoff.remote_decode.has_value());
        AckWriteBacks(handoff);
        SendPdSucceeded(id);
        return handoff;
    }

    std::int32_t device_pages_{9};
    std::int32_t host_pages_{13};
    bool in_forward_{false};
    bool kv_events_{false};
    // KV parallelism: the full group's pages dealt page-cyclically over this
    // many shards (either policy).
    std::int32_t kv_shard_count_{1};
    Role role_{Role::kP};
    SchedulerConfig::HostWritePolicy write_policy_{SchedulerConfig::HostWritePolicy::kWriteBack};
};

// The cases that hold under any shard count, with the full group's pages dealt
// over 1, 2 and 4 KV-parallel shards.
class StateCheckpointPrefillHostShardSuite : public StateCheckpointPrefillHostSuite,
                                             public ::testing::WithParamInterface<std::int32_t> {
protected:
    StateCheckpointPrefillHostShardSuite() { kv_shard_count_ = GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(KvShards, StateCheckpointPrefillHostShardSuite, ::testing::Values(1, 2, 4));

// The fused role under Host write-back: the geometry of
// StateCheckpointPrefillHostSuite (4-token pages, the full group packed four
// blocks to a parent, three State groups) on an engine that decodes its own
// prompts and retracts into the snapshot pool SetTestSnapshotPool sizes to
// the Device. A session completes by its own forwards: its prefill, its first
// decode, its finish.
class StateCheckpointFusedHostSuite : public StateCheckpointPrefillHostSuite {
protected:
    StateCheckpointFusedHostSuite() { role_ = Role::kFused; }
};

// f1's first chunk needs the room of s1's checkpoints: s1's one segment --
// everything it published, the checkpoints first -- leaves the Device and is
// copied to the Host in the plan that evicts it, stream-ordered, its blocks
// already granted to f1. Its Host entries appear at the ACK.
TEST_P(StateCheckpointPrefillHostShardSuite, EvictionWritesBackInTheSamePlan) {
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    CompleteSession("s1", MakeTokens(22));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0) << "nothing streams at publication";

    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    SendBootstrapped("f1");
    const ExecutionPlan plan = PlanOnce();
    const std::vector<CacheOperation> stores = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
    ASSERT_EQ(stores.size(), 1u);
    const WriteBackBatch& store = std::get<WriteBackBatch>(stores.front());
    ASSERT_EQ(store.op_ids.size(), 1u);
    EXPECT_EQ(store.source_pinned, std::vector<bool>{false});
    std::map<std::uint32_t, std::int32_t> rows_by_group;
    for (const std::uint32_t group : store.group_ids[0]) {
        ++rows_by_group[group];
    }
    EXPECT_EQ(rows_by_group, (std::map<std::uint32_t, std::int32_t>{{0, 6}, {1, 1}, {2, 1}, {3, 1}}));
    const std::array<std::string, 4> group_names{"full", "state0", "state1", "state2"};
    std::int32_t reused = 0;
    for (std::size_t row = 0; row < store.src_pages[0].size(); ++row) {
        reused += static_cast<std::int32_t>(
            std::ranges::count(plan.pages_to_zero.at(group_names[store.group_ids[0][row]]), store.src_pages[0][row]));
    }
    EXPECT_EQ(reused, 2) << "the parents s1's group-2 and group-3 checkpoints left hold f1's fresh blocks";
    const HostCacheStats stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.publication_stores.ops + stats.finish_stores.ops, 0u);
    EXPECT_EQ(stats.eviction_stores.ops, 1u);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0) << "published at the ACK, not before";
    AckWriteBacks(plan);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 9);
}

// After the write-back lands, the session resumes from the Host alone: its
// pages, its tail and its three checkpoints load back.
TEST_P(StateCheckpointPrefillHostShardSuite, AWrittenBackSessionResumesFromHost) {
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> tokens = MakeTokens(22);
    CompleteSession("s1", tokens);
    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    SendBootstrapped("f1");
    AckWriteBacks(PlanOnce());
    ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 9);
    SendAbortEvent("f1");
    AckWriteBacks(PlanOnce());

    tokens.push_back(77);
    const Resume resumed = Probe("s1-next", tokens);
    EXPECT_EQ(resumed.prefix_tokens, 22);
    EXPECT_EQ(resumed.load_rows, 9) << "five pages, the tail block and one checkpoint per State group";
}

// A and B share an 8-token prompt S; B, published after A, keeps S's pages
// young (B's first chunk is cut at the junction 8, so S's last page also
// holds B's checkpoint). A's segment climbs no further than S, which B's
// checkpoint boundaries still read: only A's own pages, tail and checkpoints go
// to the Host, and A's next turn claims S and the checkpoint at 8 on the Device
// and loads the rest. 15 Device parents hold both sessions -- B's prefill needs
// nine State parents beside A's three -- but not f1's first chunk as well.
TEST_P(StateCheckpointPrefillHostShardSuite, ASharedPromptStaysOnDeviceWhileItsSessionGoesToHost) {
    device_pages_ = 16;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> a = MakeTokens(22);
    std::vector<std::int32_t> b = MakeTokens(22, 500);
    std::copy_n(a.begin(), 8, b.begin());
    CompleteSession("a", a);
    CompleteSession("b", b);
    const std::vector<std::string> a_pages = scheduler_->PrefixHashesForTokens(a);
    const std::vector<std::string> b_pages = scheduler_->PrefixHashesForTokens(b);

    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    SendBootstrapped("f1");
    const ExecutionPlan plan = PlanOnce();
    const std::vector<CacheOperation> stores = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
    ASSERT_EQ(stores.size(), 1u);
    const WriteBackBatch& store = std::get<WriteBackBatch>(stores.front());
    ASSERT_EQ(store.op_ids.size(), 1u);
    EXPECT_EQ(store.src_pages[0].size(), 7u) << "three pages, the tail block and three checkpoints";
    for (const std::string& hash : store.content_hashes[0]) {
        EXPECT_NE(hash, a_pages[0]);
        EXPECT_NE(hash, a_pages[1]);
        for (const std::string& b_page : b_pages) {
            EXPECT_NE(hash, b_page);
        }
    }
    AckWriteBacks(plan);
    SendAbortEvent("f1");
    AckWriteBacks(PlanOnce());

    a.push_back(77);
    const Resume resumed = Probe("a-next", a);
    EXPECT_EQ(resumed.prefix_tokens, 22);
    EXPECT_EQ(resumed.load_rows, 7) << "S is claimed on the Device; the rest loads";
}

// A Device-only clear drops what only the Device held: the session never
// reaches the Host, and nothing resumes it.
TEST_P(StateCheckpointPrefillHostShardSuite, ClearL1CacheDropsDeviceOnlyContent) {
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> tokens = MakeTokens(22);
    CompleteSession("s1", tokens);
    ASSERT_TRUE(scheduler_->ClearL1Cache());
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(PlanOnce()).empty());
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0);
    tokens.push_back(77);
    EXPECT_EQ(Probe("s1-next", tokens).prefix_tokens, 0);
}

// While the write-back is out, the tail's hint stays, so the ACK publishes a
// tail the probe can still enumerate.
TEST_P(StateCheckpointPrefillHostShardSuite, TailHintSurvivesTheWriteBackWindow) {
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> tokens = MakeTokens(22);
    CompleteSession("s1", tokens);
    ASSERT_EQ(scheduler_->TailEntries(), 1u);
    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    SendBootstrapped("f1");
    const ExecutionPlan plan = PlanOnce();
    ASSERT_EQ(ExtractCacheOpsOfKind<WriteBackBatch>(plan).size(), 1u);
    EXPECT_EQ(scheduler_->TailEntries(), 1u) << "no tier holds the tail now, but a store is bringing it";
    AckWriteBacks(plan);
    EXPECT_EQ(scheduler_->TailEntries(), 1u);
    SendAbortEvent("f1");
    AckWriteBacks(PlanOnce());
    tokens.push_back(77);
    EXPECT_EQ(Probe("s1-next", tokens).prefix_tokens, 22);
}

// Each plan drains what its own admissions evicted: f1's first chunk evicts
// s1, f2's s2, and no batch after the first carries any of s1's keys.
TEST_P(StateCheckpointPrefillHostShardSuite, NoEvictionRecordCrossesAPlan) {
    device_pages_ = 16;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    CompleteSession("s1", MakeTokens(22));
    CompleteSession("s2", MakeTokens(22, 500));
    const std::vector<std::string> s1_pages = scheduler_->PrefixHashesForTokens(MakeTokens(22));
    const std::vector<std::string> s2_pages = scheduler_->PrefixHashesForTokens(MakeTokens(22, 500));
    // Per plan with a write-back: the rows of s1's pages and of s2's.
    std::vector<std::pair<std::int32_t, std::int32_t>> rows_by_plan;
    const auto plan_and_ack = [&] {
        ExecutionPlan plan = PlanOnce();
        for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
            std::pair<std::int32_t, std::int32_t> rows{0, 0};
            for (const std::vector<std::string>& hashes : std::get<WriteBackBatch>(operation).content_hashes) {
                for (const std::string& hash : hashes) {
                    rows.first += static_cast<std::int32_t>(std::ranges::count(s1_pages, hash));
                    rows.second += static_cast<std::int32_t>(std::ranges::count(s2_pages, hash));
                }
            }
            rows_by_plan.push_back(rows);
        }
        AckWriteBacks(plan);
        return plan;
    };
    for (const auto& [id, start] : {std::pair{"f1", 1000}, std::pair{"f2", 2000}}) {
        Submit(RequestWithTokens(id, MakeTokens(22, start)));
        SendBootstrapped(id);
        for (std::int32_t chunk = 0; chunk < 3; ++chunk) {
            plan_and_ack();
            SendForwardDone(id, chunk == 2 ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
        }
        ASSERT_TRUE(plan_and_ack().remote_decode.has_value());
        SendPdSucceeded(id);
    }
    EXPECT_EQ(rows_by_plan, (std::vector<std::pair<std::int32_t, std::int32_t>>{{5, 0}, {0, 5}}));
}

// h resumes at s1's checkpoint on the Device and prefills [22, 38) in two
// chunks. The second chunk's rotation releases that checkpoint, and with no
// other room in the pool the admission reclaims it -- with its tail's KV
// block, the request's own copy of which it keeps -- as the request's own:
// dropped, never written back.
TEST_P(StateCheckpointPrefillHostShardSuite, RequestReclaimedCheckpointIsDropped) {
    device_pages_ = 10;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> tokens = MakeTokens(22);
    CompleteSession("s1", tokens);
    for (std::int32_t i = 0; i < 16; ++i) {
        tokens.push_back(600 + i);
    }
    Submit(RequestWithTokens("h", tokens));
    SendBootstrapped("h");
    const std::vector<Chunk> chunks = RunPrefill("h", 38);
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0].prefix_tokens, 22);
    const HostCacheStats stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.eviction_stores.ops, 0u);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0);
    // The tail's KV block and its three checkpoints.
    EXPECT_EQ(stats.device_evictions.request_reclaim.blocks, 4u);
    EXPECT_EQ(stats.device_evictions.admission.blocks, 0u);
}

// A single-chunk turn: h resumes at s1's checkpoint at 22 and computes
// [22, 27). Once the transfer completes, h's finish publishes the prompt end
// at 27, which supersedes that checkpoint. It stays cached while the Device
// has room, so the same prompt sent again -- a retry -- still resumes at 22,
// on the Device.
TEST_P(StateCheckpointPrefillHostShardSuite, AnIdenticalResendResumesAtTheSupersededInput) {
    device_pages_ = 16;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> tokens = MakeTokens(22);
    CompleteSession("s1", tokens);
    for (std::int32_t i = 0; i < 5; ++i) {
        tokens.push_back(600 + i);
    }
    CompleteSession("h", tokens);
    EXPECT_EQ(scheduler_->TailEntries(), 2u) << "s1's tail at 22 and h's at 27";
    const Resume resent = Probe("resent", tokens);
    EXPECT_EQ(resent.prefix_tokens, 22);
    EXPECT_EQ(resent.load_rows, 0);
    const CacheCoordinator::DeviceEvictionStats evictions = scheduler_->GetHostCacheStats().device_evictions;
    EXPECT_EQ(evictions.admission.blocks + evictions.request_reclaim.blocks, 0u);
}

// o, then s1 and its next turn h, leave three of the Device's 16 parents
// empty; h's finish marks s1's checkpoint at 22 superseded. f1's first chunk
// needs three more: the checkpoint goes first, ahead of o, the least recently
// used session, and is enough. It leaves with its tail's KV block as h's
// reclaim, under either write policy; o and h still resume on the Device.
// Under write-back it is written back nowhere, so the same prompt as h no
// longer resumes at 22; under write-through its finish stored it on the Host,
// so that prompt loads it back from there.
TEST_P(StateCheckpointPrefillHostShardSuite, TheSupersededInputIsTheFirstVictim) {
    device_pages_ = 17;
    for (const auto policy :
         {SchedulerConfig::HostWritePolicy::kWriteBack, SchedulerConfig::HostWritePolicy::kWriteThrough}) {
        SCOPED_TRACE(static_cast<int>(policy));
        const bool write_back = policy == SchedulerConfig::HostWritePolicy::kWriteBack;
        write_policy_ = policy;
        Reset(/*host_cache=*/true, /*decode_width=*/1);
        std::vector<std::int32_t> o = MakeTokens(22, 3000);
        CompleteSession("o", o);
        std::vector<std::int32_t> tokens = MakeTokens(22);
        CompleteSession("s1", tokens);
        for (std::int32_t i = 0; i < 5; ++i) {
            tokens.push_back(600 + i);
        }
        CompleteSession("h", tokens);
        // f1's admission, not h's finish, takes the marked checkpoint.
        EXPECT_EQ(scheduler_->TailEntries(), 3u) << "o's and s1's tails at 22 and h's at 27";

        Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
        SendBootstrapped("f1");
        const ExecutionPlan plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        const HostCacheStats stats = scheduler_->GetHostCacheStats();
        // The checkpoint's tail: its KV block and three checkpoints.
        EXPECT_EQ(stats.device_evictions.request_reclaim.blocks, 4u);
        EXPECT_EQ(stats.device_evictions.admission.blocks, 0u);
        if (write_back) {
            EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(plan).empty());
        }
        AckWriteBacks(plan);
        SendAbortEvent("f1");
        AckWriteBacks(PlanOnce());

        o.push_back(77);
        const Resume o_next = Probe("o-next", o);
        EXPECT_EQ(o_next.prefix_tokens, 22);
        EXPECT_EQ(o_next.load_rows, 0);
        std::vector<std::int32_t> next = tokens;
        next.push_back(77);
        const Resume h_next = Probe("h-next", next);
        EXPECT_EQ(h_next.prefix_tokens, 27);
        EXPECT_EQ(h_next.load_rows, 0);
        const Resume resent = Probe("resent", tokens);
        EXPECT_EQ(resent.prefix_tokens, write_back ? 0 : 22);
        EXPECT_EQ(resent.load_rows, write_back ? 0 : 4) << "the tail block and one checkpoint per State group";
    }
}

// The tie the mark breaks: h acquires s1's checkpoint at 22 and publishes its
// end state at 27 under one access epoch, so with s1 and h alone on the
// Device the two tails' checkpoints rank level and block location decides,
// here for the one at 27. f1's first chunk needs the room of one tail's
// checkpoints. Under write-through -- with a Host tier or without one, the
// full group sharded (KV parallelism) or not -- h's finish marked the
// checkpoint at 22: it leaves, as h's reclaim, and h's next turn resumes at
// 27 on the Device. (Unmarked, the end state at 27 leaves: that turn loads
// it back from the Host, or without one resumes at 22.) The finish's own
// store lands first: it pins the end state until then.
TEST_F(StateCheckpointPrefillHostSuite, TheEndStateOutlivesTheInputItSupersedesUnderWriteThrough) {
    device_pages_ = 15;
    write_policy_ = SchedulerConfig::HostWritePolicy::kWriteThrough;
    for (const auto& [host_cache, shards] :
         {std::pair{true, 1}, std::pair{false, 1}, std::pair{true, 2}, std::pair{false, 2}}) {
        SCOPED_TRACE(testing::Message() << "host " << host_cache << ", KV shards " << shards);
        kv_shard_count_ = shards;
        Reset(host_cache, /*decode_width=*/1);
        std::vector<std::int32_t> tokens = MakeTokens(22);
        CompleteSession("s1", tokens);
        for (std::int32_t i = 0; i < 5; ++i) {
            tokens.push_back(600 + i);
        }
        CompleteSession("h", tokens);
        AckWriteBacks(PlanOnce());

        Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
        SendBootstrapped("f1");
        const ExecutionPlan plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        const CacheCoordinator::DeviceEvictionStats evictions = scheduler_->GetHostCacheStats().device_evictions;
        // The input's tail: its KV block and three checkpoints.
        EXPECT_EQ(evictions.request_reclaim.blocks, 4u);
        EXPECT_EQ(evictions.admission.blocks, 0u);
        AckWriteBacks(plan);
        SendAbortEvent("f1");
        AckWriteBacks(PlanOnce());

        tokens.push_back(77);
        const Resume next = Probe("h-next", tokens);
        EXPECT_EQ(next.prefix_tokens, 27);
        EXPECT_EQ(next.load_rows, 0);
    }
}

// On the fused role a turn publishes two tails: its prompt end at its first
// decode and its end state at its finish. s1 ends at 23 (its 22 tokens and a
// decode); h, its next turn, resumes at one of s1's tails -- at s1's end state
// when the client sends the reply back as it was generated, at its prompt end
// when it renders the reply anew -- and publishes its prompt end at 28 and its
// end state at 29, both at the epoch its admission dated that input with.
// Under either write policy, with a Host tier or (write-through) without one,
// h's finish marks the input. f1's first chunk needs the room of two tails'
// checkpoints: the input leaves first, as h's reclaim, then s1's other tail,
// the least recently used; h's next turn resumes at 29 and a prompt that
// leaves h's turn at its prompt end resumes at 28, both on the Device.
// (Unmarked, the input ranks level with h's two tails: after s1's other tail,
// both of h's leave here, and the next turn loads its end state back from the
// Host, or without one resumes at 23.) h's own two tails still rank level with
// each other: which one the next turn resumes at depends on whether the client
// sends the reply back as it was generated. Under write-back the input is
// dropped and s1's other tail, which no turn continued, is written back like
// any cache: a prompt that leaves s1 there loads it back from the Host.
TEST_F(StateCheckpointFusedHostSuite, TheFusedRoleMarksTheInputAheadOfBothTailsOfTheTurn) {
    device_pages_ = 20;
    for (const auto& [policy, host_cache] : {std::pair{SchedulerConfig::HostWritePolicy::kWriteThrough, true},
                                             std::pair{SchedulerConfig::HostWritePolicy::kWriteThrough, false},
                                             std::pair{SchedulerConfig::HostWritePolicy::kWriteBack, true}}) {
        for (const bool as_generated : {true, false}) {
            SCOPED_TRACE(testing::Message() << "policy " << static_cast<int>(policy) << ", host " << host_cache
                                            << ", reply as generated " << as_generated);
            write_policy_ = policy;
            Reset(host_cache, /*decode_width=*/1);
            const std::vector<std::int32_t> s1 = MakeTokens(22);
            std::vector<std::int32_t> h = s1;
            h.push_back(as_generated ? 900 : 500);
            for (std::int32_t i = 0; i < 5; ++i) {
                h.push_back(600 + i);
            }
            // The prefill samples 900, the first decode 901; the finish leaves
            // the request quiescent, so it publishes the state after 900.
            for (const auto& [id, tokens] : {std::pair{"s1", s1}, std::pair{"h", h}}) {
                Submit(RequestWithTokens(id, tokens));
                const std::vector<Chunk> chunks = RunPrefill(id, static_cast<std::int32_t>(tokens.size()));
                ASSERT_FALSE(chunks.empty()) << id;
                EXPECT_EQ(chunks[0].prefix_tokens, std::string{id} == "h" ? (as_generated ? 23 : 22) : 0) << id;
                AckWriteBacks(PlanOnce());
                SendForwardDone(id, {901});
                SendFinish(id);
                AckWriteBacks(PlanOnce());
            }

            Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
            const ExecutionPlan plan = PlanOnce();
            ASSERT_NE(FindForwardBatch(plan), nullptr);
            // Each tail leaves with its KV block and three checkpoints: the
            // input as h's reclaim, s1's other tail as an admission victim.
            const HostCacheStats stats = scheduler_->GetHostCacheStats();
            EXPECT_EQ(stats.device_evictions.request_reclaim.blocks, 4u);
            EXPECT_EQ(stats.device_evictions.admission.blocks, 4u);
            std::uint64_t written_back = 0;
            for (const std::uint64_t blocks : stats.eviction_stores.blocks) {
                written_back += blocks;
            }
            EXPECT_EQ(written_back, policy == SchedulerConfig::HostWritePolicy::kWriteBack ? 4u : 0u);
            AckWriteBacks(plan);
            SendAbortEvent("f1");
            AckWriteBacks(PlanOnce());

            std::vector<std::int32_t> next = h;
            next.push_back(900);
            next.push_back(77);
            const Resume resumed = Probe("h-next", next);
            EXPECT_EQ(resumed.prefix_tokens, 29);
            EXPECT_EQ(resumed.load_rows, 0);
            std::vector<std::int32_t> branch = h;
            branch.push_back(555);
            const Resume branched = Probe("branch", branch);
            EXPECT_EQ(branched.prefix_tokens, 28);
            EXPECT_EQ(branched.load_rows, 0);
            if (policy == SchedulerConfig::HostWritePolicy::kWriteBack) {
                std::vector<std::int32_t> other = s1;
                other.push_back(as_generated ? 555 : 900);
                other.push_back(77);
                const Resume left = Probe("other", other);
                EXPECT_EQ(left.prefix_tokens, as_generated ? 22 : 23);
                EXPECT_EQ(left.load_rows, 4) << "the tail block and three checkpoints";
            }
        }
    }
}

// Only an admission's eviction counts a superseded input as the reclaim of
// the request that superseded it: a clear counts it as flushed, as every
// other entry -- s1's five pages and the tail block at 22, h's page at 20
// and the tail block at 27, and the two tails' checkpoints.
TEST_P(StateCheckpointPrefillHostShardSuite, AClearFlushesTheSupersededInputLikeAnyEntry) {
    device_pages_ = 16;
    for (const auto policy :
         {SchedulerConfig::HostWritePolicy::kWriteBack, SchedulerConfig::HostWritePolicy::kWriteThrough}) {
        SCOPED_TRACE(static_cast<int>(policy));
        write_policy_ = policy;
        Reset(/*host_cache=*/true, /*decode_width=*/1);
        std::vector<std::int32_t> tokens = MakeTokens(22);
        CompleteSession("s1", tokens);
        for (std::int32_t i = 0; i < 5; ++i) {
            tokens.push_back(600 + i);
        }
        CompleteSession("h", tokens);
        AckWriteBacks(PlanOnce());
        ASSERT_TRUE(scheduler_->ClearL1Cache());
        const CacheCoordinator::DeviceEvictionStats evictions = scheduler_->GetHostCacheStats().device_evictions;
        EXPECT_EQ(evictions.flush.blocks, 14u) << "eight KV blocks and two tails' three checkpoints";
        EXPECT_EQ(evictions.request_reclaim.blocks, 0u);
    }
}

// As in TheSupersededInputIsTheFirstVictim, but with one Device parent fewer:
// the marked checkpoint is still chosen first, yet its room is not enough, and
// o, chosen next, makes the room alone. The plan keeps only the victims it
// needs, so the checkpoint stays: o is written back whole, nothing is
// dropped, and the same prompt as h still resumes at 22 on the Device.
TEST_P(StateCheckpointPrefillHostShardSuite, TheSupersededInputStaysWhenTheNextVictimMakesTheRoomAlone) {
    device_pages_ = 16;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> o = MakeTokens(22, 3000);
    CompleteSession("o", o);
    std::vector<std::int32_t> tokens = MakeTokens(22);
    CompleteSession("s1", tokens);
    for (std::int32_t i = 0; i < 5; ++i) {
        tokens.push_back(600 + i);
    }
    CompleteSession("h", tokens);

    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    SendBootstrapped("f1");
    const ExecutionPlan plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan), nullptr);
    const std::vector<CacheOperation> stores = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
    ASSERT_EQ(stores.size(), 1u);
    std::map<std::uint32_t, std::int32_t> rows_by_group;
    for (const std::uint32_t group : std::get<WriteBackBatch>(stores.front()).group_ids[0]) {
        ++rows_by_group[group];
    }
    EXPECT_EQ(rows_by_group, (std::map<std::uint32_t, std::int32_t>{{0, 6}, {1, 1}, {2, 1}, {3, 1}}));
    // The admission's evictions: o's five pages and tail block, and its three
    // checkpoints.
    const CacheCoordinator::DeviceEvictionStats evictions = scheduler_->GetHostCacheStats().device_evictions;
    EXPECT_EQ(evictions.admission.blocks, 9u);
    EXPECT_EQ(evictions.request_reclaim.blocks, 0u);
    AckWriteBacks(plan);
    SendAbortEvent("f1");
    AckWriteBacks(PlanOnce());

    const Resume resent = Probe("resent", tokens);
    EXPECT_EQ(resent.prefix_tokens, 22);
    EXPECT_EQ(resent.load_rows, 0);
    o.push_back(77);
    const Resume o_next = Probe("o-next", o);
    EXPECT_EQ(o_next.prefix_tokens, 22);
    EXPECT_EQ(o_next.load_rows, 9) << "five pages, the tail block and one checkpoint per State group";
}

// A single-chunk turn on a full Device: h resumes at s1's checkpoint at 22,
// computes [22, 27) and marks that checkpoint superseded at its finish. f1's
// first chunk needs more room than the checkpoint holds: the checkpoint
// leaves first, with its tail's KV block, as h's reclaim, and h's own
// segment -- its page at 20 and its tail at 27 with the checkpoints -- is
// written back without it. The climb from 27 stops below s1's five pages,
// which the checkpoint at 22 still hangs under while the plan is chosen, so
// they stay on the Device: h's next turn resumes on them and loads the rest.
TEST_F(StateCheckpointPrefillHostSuite, TheSupersededInputOfASingleChunkTurnIsDroppedWhenEvicted) {
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> tokens = MakeTokens(22);
    CompleteSession("s1", tokens);
    for (std::int32_t i = 0; i < 5; ++i) {
        tokens.push_back(600 + i);
    }
    CompleteSession("h", tokens);
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.request_reclaim.blocks, 0u);
    EXPECT_EQ(scheduler_->TailEntries(), 2u);

    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    SendBootstrapped("f1");
    const ExecutionPlan plan = PlanOnce();
    const std::vector<CacheOperation> stores = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
    ASSERT_EQ(stores.size(), 1u);
    const WriteBackBatch& store = std::get<WriteBackBatch>(stores.front());
    std::map<std::uint32_t, std::int32_t> rows_by_group;
    for (const std::uint32_t group : store.group_ids[0]) {
        ++rows_by_group[group];
    }
    EXPECT_EQ(rows_by_group, (std::map<std::uint32_t, std::int32_t>{{0, 2}, {1, 1}, {2, 1}, {3, 1}}));
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.request_reclaim.blocks, 4u)
        << "the checkpoint's tail: its KV block and three checkpoints";
    AckWriteBacks(plan);
    EXPECT_EQ(scheduler_->TailEntries(), 1u) << "only h's tail at 27";
    SendAbortEvent("f1");
    AckWriteBacks(PlanOnce());

    tokens.push_back(77);
    const Resume next = Probe("h-next", tokens);
    EXPECT_EQ(next.prefix_tokens, 27);
    EXPECT_EQ(next.load_rows, 5) << "h's page at 20, its tail block and one checkpoint per State group";
}

// A two-chunk turn: h resumes at s1's checkpoint at 22 and prefills
// [22, 38). The second chunk's rotation releases that checkpoint, and no
// admission needs its room, so it stays cached; h's finish, publishing the
// prompt end at 38, marks it superseded. f1's first chunk needs room: the
// checkpoint goes first, with its tail's KV block, and is enough. Nothing is
// written back, and a prompt that leaves h's turn after s1's 22 tokens finds
// no checkpoint to resume at any more.
TEST_P(StateCheckpointPrefillHostShardSuite, TheSupersededInputOfATwoChunkTurnIsMarkedAtItsFinish) {
    device_pages_ = 16;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    const std::vector<std::int32_t> s1 = MakeTokens(22);
    CompleteSession("s1", s1);
    std::vector<std::int32_t> tokens = s1;
    for (std::int32_t i = 0; i < 16; ++i) {
        tokens.push_back(600 + i);
    }
    Submit(RequestWithTokens("h", tokens));
    SendBootstrapped("h");
    const std::vector<Chunk> chunks = RunPrefill("h", 38);
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0].prefix_tokens, 22);
    const ExecutionPlan handoff = PlanOnce();
    ASSERT_TRUE(handoff.remote_decode.has_value());
    AckWriteBacks(handoff);
    SendPdSucceeded("h");
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.request_reclaim.blocks, 0u) << "nothing needed its room";
    EXPECT_EQ(scheduler_->TailEntries(), 2u) << "s1's tail at 22 and h's at 38";

    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    SendBootstrapped("f1");
    const ExecutionPlan plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan), nullptr);
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(plan).empty());
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.request_reclaim.blocks, 4u)
        << "the checkpoint's tail: its KV block and three checkpoints";
    EXPECT_EQ(scheduler_->TailEntries(), 1u) << "only h's tail at 38";
    AckWriteBacks(plan);
    SendAbortEvent("f1");
    AckWriteBacks(PlanOnce());
    std::vector<std::int32_t> other = s1;
    other.push_back(77);
    EXPECT_EQ(Probe("other", other).prefix_tokens, 0);
}

// As in TheSupersededInputIsTheFirstVictim, h's finish marks s1's
// checkpoint at 22. The same prompt as h sent again resumes there, which
// unmarks it: it is an ordinary cache entry again, dated by that admission.
// Aborted, the resend marks nothing, so f1's room comes from o, now the least
// recently used session, written back whole -- its five pages, its tail and
// its checkpoints -- and the checkpoint at 22 stays. Finished, the resend
// supersedes it once more, and f1 drops it and evicts nothing else. (The
// finished resend's first chunk is cut at the junction 24, where h's page at
// 20 ends, and the checkpoint it publishes there -- whose room evicts o --
// is no tail and stays.)
TEST_F(StateCheckpointPrefillHostSuite, AHitUnmarksTheSupersededInputUntilTheHitterFinishes) {
    device_pages_ = 17;
    for (const bool resend_finishes : {false, true}) {
        SCOPED_TRACE(resend_finishes);
        Reset(/*host_cache=*/true, /*decode_width=*/1);
        CompleteSession("o", MakeTokens(22, 3000));
        std::vector<std::int32_t> tokens = MakeTokens(22);
        CompleteSession("s1", tokens);
        for (std::int32_t i = 0; i < 5; ++i) {
            tokens.push_back(600 + i);
        }
        CompleteSession("h", tokens);
        if (resend_finishes) {
            Submit(RequestWithTokens("resent", tokens));
            SendBootstrapped("resent");
            const std::vector<Chunk> chunks = RunPrefill("resent", 27);
            ASSERT_EQ(chunks.size(), 2u);
            EXPECT_EQ(chunks[0].prefix_tokens, 22);
            EXPECT_EQ(chunks[1].prefix_tokens, 24);
            const ExecutionPlan handoff = PlanOnce();
            ASSERT_TRUE(handoff.remote_decode.has_value());
            AckWriteBacks(handoff);
            SendPdSucceeded("resent");
        } else {
            EXPECT_EQ(Probe("resent", tokens).prefix_tokens, 22);
        }
        const HostCacheStats before = scheduler_->GetHostCacheStats();

        Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
        SendBootstrapped("f1");
        const ExecutionPlan plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        const std::vector<CacheOperation> stores = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
        const HostCacheStats stats = scheduler_->GetHostCacheStats();
        if (resend_finishes) {
            EXPECT_TRUE(stores.empty());
            const CacheCoordinator::DeviceEvictionStats& now = stats.device_evictions;
            const CacheCoordinator::DeviceEvictionStats& then = before.device_evictions;
            EXPECT_EQ(now.admission.blocks, then.admission.blocks);
            // The checkpoint's tail, its KV block and three checkpoints, and
            // nothing before.
            EXPECT_EQ(then.request_reclaim.blocks, 0u);
            EXPECT_EQ(now.request_reclaim.blocks - then.request_reclaim.blocks, 4u);
        } else {
            ASSERT_EQ(stores.size(), 1u);
            std::map<std::uint32_t, std::int32_t> rows_by_group;
            for (const std::uint32_t group : std::get<WriteBackBatch>(stores.front()).group_ids[0]) {
                ++rows_by_group[group];
            }
            EXPECT_EQ(rows_by_group, (std::map<std::uint32_t, std::int32_t>{{0, 6}, {1, 1}, {2, 1}, {3, 1}}));
            EXPECT_EQ(stats.device_evictions.request_reclaim.blocks, 0u);
        }
        AckWriteBacks(plan);
        SendAbortEvent("f1");
        AckWriteBacks(PlanOnce());
        const Resume again = Probe("again", tokens);
        EXPECT_EQ(again.prefix_tokens, resend_finishes ? 24 : 22);
        EXPECT_EQ(again.load_rows, 0);
    }
}

// The transfer of a turn that resumed at a tail checkpoint fails: the
// request publishes no prompt end and marks nothing, so the checkpoint it
// resumed from stays an ordinary cache entry. When f1 needs the room, s1's
// whole chain, the checkpoint included, is written back, and the turn tried
// again resumes there from the Host.
TEST_F(StateCheckpointPrefillHostSuite, AFailedTransferMarksNothing) {
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> tokens = MakeTokens(22);
    CompleteSession("s1", tokens);
    for (std::int32_t i = 0; i < 5; ++i) {
        tokens.push_back(600 + i);
    }
    Submit(RequestWithTokens("h", tokens));
    SendBootstrapped("h");
    RunPrefill("h", 27);
    const ExecutionPlan handoff = PlanOnce();
    ASSERT_TRUE(handoff.remote_decode.has_value());
    AckWriteBacks(handoff);
    ExecutionEvent failed;
    failed.With(pd::FailedEvent{"h"});
    scheduler_->Advance(std::move(failed));

    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    SendBootstrapped("f1");
    const ExecutionPlan plan = PlanOnce();
    const std::vector<CacheOperation> stores = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
    ASSERT_EQ(stores.size(), 1u);
    std::map<std::uint32_t, std::int32_t> rows_by_group;
    for (const std::uint32_t group : std::get<WriteBackBatch>(stores.front()).group_ids[0]) {
        ++rows_by_group[group];
    }
    EXPECT_EQ(rows_by_group, (std::map<std::uint32_t, std::int32_t>{{0, 6}, {1, 1}, {2, 1}, {3, 1}}));
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.request_reclaim.blocks, 0u);
    AckWriteBacks(plan);
    SendAbortEvent("f1");
    AckWriteBacks(PlanOnce());
    const Resume again = Probe("h-again", tokens);
    EXPECT_EQ(again.prefix_tokens, 22);
    EXPECT_EQ(again.load_rows, 9) << "five pages, the tail block and one checkpoint per State group";
}

// h and g, two next turns of s1, both resume at its checkpoint at 22, g
// after h. h's finish comes after g's hit, so it leaves the checkpoint
// unmarked: it is g's input until g finishes. g's transfer fails and marks
// nothing either. f1's first chunk needs room: the oldest entries are h's,
// dated by h's admission before g's hit, and its page at 24 with its tail
// and checkpoints at 27 are written back; the checkpoint at 22 stays on the
// Device for g tried again. Had g finished, its finish would have marked the
// checkpoint, and f1 would have dropped it instead, so the same prompt as g
// sent again would start over. Under write-through, with a Host tier or
// without one, the same holds: f1 evicts h's tail and its checkpoints at 27
// while g's input is unmarked, and the checkpoint at 22 as well, as g's
// reclaim, once g finished -- the same prompt as g then loads it back from
// the Host, or without one starts over.
TEST_P(StateCheckpointPrefillHostShardSuite, AnEarlierFinishLeavesALaterHittersInputUnmarked) {
    device_pages_ = 15;
    for (const auto& [policy, host_cache] : {std::pair{SchedulerConfig::HostWritePolicy::kWriteBack, true},
                                             std::pair{SchedulerConfig::HostWritePolicy::kWriteThrough, true},
                                             std::pair{SchedulerConfig::HostWritePolicy::kWriteThrough, false}}) {
        for (const bool g_finishes : {false, true}) {
            SCOPED_TRACE(testing::Message() << "policy " << static_cast<int>(policy) << ", host " << host_cache
                                            << ", g finishes " << g_finishes);
            write_policy_ = policy;
            const bool write_back = policy == SchedulerConfig::HostWritePolicy::kWriteBack;
            Reset(host_cache, /*decode_width=*/1);
            const std::vector<std::int32_t> s1 = MakeTokens(22);
            CompleteSession("s1", s1);
            std::vector<std::int32_t> h = s1;
            std::vector<std::int32_t> g = s1;
            for (std::int32_t i = 0; i < 5; ++i) {
                h.push_back(600 + i);
                g.push_back(700 + i);
            }
            for (const auto& [id, tokens] : {std::pair{"h", h}, std::pair{"g", g}}) {
                Submit(RequestWithTokens(id, tokens));
                SendBootstrapped(id);
                const std::vector<Chunk> chunks = RunPrefill(id, 27);
                ASSERT_EQ(chunks.size(), 1u) << id;
                EXPECT_EQ(chunks[0].prefix_tokens, 22) << id;
                const ExecutionPlan handoff = PlanOnce();
                ASSERT_TRUE(handoff.remote_decode.has_value()) << id;
                AckWriteBacks(handoff);
            }
            SendPdSucceeded("h");
            if (g_finishes) {
                SendPdSucceeded("g");
            } else {
                ExecutionEvent failed;
                failed.With(pd::FailedEvent{"g"});
                scheduler_->Advance(std::move(failed));
            }
            // Under write-through the finishes' stores land first: each pins
            // its end state until then.
            AckWriteBacks(PlanOnce());

            Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
            SendBootstrapped("f1");
            const ExecutionPlan plan = PlanOnce();
            ASSERT_NE(FindForwardBatch(plan), nullptr);
            const std::vector<CacheOperation> stores = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
            ASSERT_EQ(stores.size(), write_back ? 1u : 0u);
            const std::uint64_t reclaimed = scheduler_->GetHostCacheStats().device_evictions.request_reclaim.blocks;
            EXPECT_EQ(reclaimed, g_finishes ? 4u : 0u) << "the tail's KV block and three checkpoints";
            if (write_back && !g_finishes) {
                std::map<std::uint32_t, std::int32_t> rows_by_group;
                for (const std::uint32_t group : std::get<WriteBackBatch>(stores.front()).group_ids[0]) {
                    ++rows_by_group[group];
                }
                EXPECT_EQ(rows_by_group, (std::map<std::uint32_t, std::int32_t>{{0, 2}, {1, 1}, {2, 1}, {3, 1}}));
            }
            AckWriteBacks(plan);
            SendAbortEvent("f1");
            AckWriteBacks(PlanOnce());
            const Resume again = Probe("g-again", g);
            const bool from_host = g_finishes && !write_back && host_cache;
            EXPECT_EQ(again.prefix_tokens, g_finishes && !from_host ? 0 : 22);
            EXPECT_EQ(again.load_rows, from_host ? 4 : 0) << "the tail block and one checkpoint per State group";
        }
    }
}

// As in AnEarlierFinishLeavesALaterHittersInputUnmarked, h and then g resume
// at s1's checkpoint at 22, g's transfer fails, and h's finish leaves the
// checkpoint unmarked for g tried again -- also when x, s1's prompt sent again
// before h came, publishes that checkpoint again after g's hit and before
// h's finish: x's finish dates it back to x's epoch, below h's, but g's hit
// still counts. Unmarked, it ranks after o's tail, the least recently used:
// f1's prefill evicts o's checkpoints with o's tail block (under write-back
// o's whole segment, written back), nothing as a reclaim, and g tried again
// resumes at 22 on the Device. o's next turn loads its checkpoints back from
// the Host (its whole session under write-back), or without one starts over.
// (Marked, the checkpoint at 22 would leave first, as h's reclaim.)
TEST_P(StateCheckpointPrefillHostShardSuite, AnOlderPublicationLeavesALaterHittersInputUnmarked) {
    device_pages_ = 24;
    host_pages_ = 40;
    for (const auto& [policy, host_cache] : {std::pair{SchedulerConfig::HostWritePolicy::kWriteBack, true},
                                             std::pair{SchedulerConfig::HostWritePolicy::kWriteThrough, true},
                                             std::pair{SchedulerConfig::HostWritePolicy::kWriteThrough, false}}) {
        SCOPED_TRACE(testing::Message() << "policy " << static_cast<int>(policy) << ", host " << host_cache);
        write_policy_ = policy;
        const bool write_back = policy == SchedulerConfig::HostWritePolicy::kWriteBack;
        Reset(host_cache, /*decode_width=*/1);
        config_.max_batch_size = 3;  // x, h and g in flight together
        scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
        std::vector<std::int32_t> o = MakeTokens(22, 3000);
        CompleteSession("o", o);
        const std::vector<std::int32_t> s1 = MakeTokens(22);
        CompleteSession("s1", s1);
        AckWriteBacks(PlanOnce());
        std::vector<std::int32_t> h = s1;
        std::vector<std::int32_t> g = s1;
        for (std::int32_t i = 0; i < 5; ++i) {
            h.push_back(600 + i);
            g.push_back(700 + i);
        }
        for (const auto& [id, tokens] : {std::pair{"x", s1}, std::pair{"h", h}, std::pair{"g", g}}) {
            Submit(RequestWithTokens(id, tokens));
            SendBootstrapped(id);
            const std::vector<Chunk> chunks = RunPrefill(id, static_cast<std::int32_t>(tokens.size()));
            ASSERT_FALSE(chunks.empty()) << id;
            EXPECT_EQ(chunks[0].prefix_tokens, std::string{id} == "x" ? 0 : 22) << id;
            const ExecutionPlan handoff = PlanOnce();
            ASSERT_TRUE(handoff.remote_decode.has_value()) << id;
            AckWriteBacks(handoff);
        }
        SendPdSucceeded("x");
        AckWriteBacks(PlanOnce());
        SendPdSucceeded("h");
        ExecutionEvent failed;
        failed.With(pd::FailedEvent{"g"});
        scheduler_->Advance(std::move(failed));
        AckWriteBacks(PlanOnce());

        Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
        SendBootstrapped("f1");
        ASSERT_EQ(RunPrefill("f1", 22).size(), 3u);
        // o's tail: its KV block and three checkpoints, and under write-back
        // the rest of o's segment, its five pages.
        const CacheCoordinator::DeviceEvictionStats evictions = scheduler_->GetHostCacheStats().device_evictions;
        EXPECT_EQ(evictions.admission.blocks, write_back ? 9u : 4u);
        EXPECT_EQ(evictions.request_reclaim.blocks, 0u);
        AckWriteBacks(PlanOnce());
        SendAbortEvent("f1");
        AckWriteBacks(PlanOnce());
        const Resume again = Probe("g-again", g);
        EXPECT_EQ(again.prefix_tokens, 22);
        EXPECT_EQ(again.load_rows, 0);
        o.push_back(77);
        const Resume o_next = Probe("o-next", o);
        EXPECT_EQ(o_next.prefix_tokens, host_cache ? 22 : 0);
        EXPECT_EQ(o_next.load_rows, write_back ? 9 : host_cache ? 4 : 0);
    }
}

// On the fused role h resumes at s1's end state at 23 and prefills [23, 39)
// in two chunks. The forced retraction of plan 7 suspends it after the first,
// with that input still in its slot 0, and f0, admitted in the same plan,
// evicts the input from the Device. h's restore loads it back from the Host
// and republishes it dated with h's own access epoch, as its admission dated
// it, so h's finish still marks it superseded: f1's first chunk drops it, as
// h's reclaim, and h's next turn resumes at its end state at 40 on the
// Device. (Dated as a later request's hit, the input would stay unmarked and
// outrank h's tails: f1 would evict two older tails instead, h's end state
// among them, and that turn would load its checkpoints back from the Host.)
TEST_F(StateCheckpointPrefillHostSuite, ARestoredInputIsStillMarkedByItsOwnFinish) {
    device_pages_ = 18;
    write_policy_ = SchedulerConfig::HostWritePolicy::kWriteThrough;
    role_ = Role::kFused;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    config_.debug_force_retraction_interval = -7;  // the oldest Prefilling request, every seventh plan
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    const std::vector<std::int32_t> s1 = MakeTokens(22);
    Submit(RequestWithTokens("s1", s1));
    ASSERT_EQ(RunPrefill("s1", 22).size(), 3u);  // plans 1-3
    AckWriteBacks(PlanOnce());                   // plan 4: the first decode
    SendForwardDone("s1", {901});
    SendFinish("s1");
    AckWriteBacks(PlanOnce());  // plan 5
    std::vector<std::int32_t> h = s1;
    h.push_back(900);
    for (std::int32_t i = 0; i < 16; ++i) {
        h.push_back(600 + i);
    }
    Submit(RequestWithTokens("h", h));
    const ExecutionPlan first = PlanOnce();  // plan 6
    const ForwardBatch* chunk = FindForwardBatch(first);
    ASSERT_NE(chunk, nullptr);
    EXPECT_EQ(chunk->extend_prefix_lens, std::vector<std::int32_t>{23});
    AckWriteBacks(first);
    SendForwardDone("h");

    Submit(RequestWithTokens("f0", MakeTokens(30, 1000)));
    const ExecutionPlan retraction = PlanOnce();  // plan 7
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    ASSERT_NE(FindForwardBatch(retraction), nullptr) << "f0's first chunk";
    AckImageStores(retraction);
    AckWriteBacks(retraction);
    SendForwardDone("f0");
    SendAbortEvent("f0");
    const ExecutionPlan readmit = PlanOnce();
    ASSERT_NE(FindRestore(readmit), nullptr);
    AckRestores(readmit);
    AckWriteBacks(readmit);
    const ExecutionPlan second = PlanOnce();
    chunk = FindForwardBatch(second);
    ASSERT_NE(chunk, nullptr);
    EXPECT_EQ(chunk->extend_prefix_lens, std::vector<std::int32_t>{31});
    AckWriteBacks(second);
    SendForwardDone("h", {900});
    AckWriteBacks(PlanOnce());
    SendForwardDone("h", {901});
    SendFinish("h");
    AckWriteBacks(PlanOnce());

    const HostCacheStats before = scheduler_->GetHostCacheStats();
    Submit(RequestWithTokens("f1", MakeTokens(4, 2000)));
    const ExecutionPlan plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan), nullptr);
    // In f1's plan alone: the input's three checkpoints, as h's reclaim.
    const HostCacheStats after = scheduler_->GetHostCacheStats();
    EXPECT_EQ(after.device_evictions.admission.blocks - before.device_evictions.admission.blocks, 0u);
    EXPECT_EQ(after.device_evictions.request_reclaim.blocks - before.device_evictions.request_reclaim.blocks, 3u);
    AckWriteBacks(plan);
    SendAbortEvent("f1");
    AckWriteBacks(PlanOnce());

    std::vector<std::int32_t> next = h;
    next.push_back(900);
    next.push_back(77);
    const Resume resumed = Probe("h-next", next);
    EXPECT_EQ(resumed.prefix_tokens, 40);
    EXPECT_EQ(resumed.load_rows, 0);
}

// A junction checkpoint is no superseded input: other prompts resume there.
// B shares A's first 8 tokens, so its first chunk is cut at the junction 8 and
// publishes the checkpoint there, which its second and last chunk [8, 14)
// reads; C resumes at that checkpoint and prefills [8, 14) in one chunk. Each
// finish marks nothing -- the checkpoint read is a page boundary, not a
// turn's tail -- so when f1 needs room the Device evicts by recency, B's own
// page and tail (C read the junction after B), nothing is dropped, and the
// next prompt sharing the 8 tokens still resumes at 8 on the Device.
TEST_P(StateCheckpointPrefillHostShardSuite, AJunctionCheckpointOutlivesTheHandOffsThatReadIt) {
    device_pages_ = 16;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    const std::vector<std::int32_t> a = MakeTokens(22);
    const auto sharing = [&](std::int32_t start) {
        std::vector<std::int32_t> tokens = MakeTokens(14, start);
        std::copy_n(a.begin(), 8, tokens.begin());
        return tokens;
    };
    CompleteSession("a", a);
    CompleteSession("b", sharing(500));
    EXPECT_EQ(Probe("after-b", sharing(700)).prefix_tokens, 8);
    CompleteSession("c", sharing(900));
    EXPECT_EQ(Probe("after-c", sharing(1100)).prefix_tokens, 8);

    Submit(RequestWithTokens("f1", MakeTokens(22, 2000)));
    SendBootstrapped("f1");
    const ExecutionPlan plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan), nullptr);
    const std::vector<CacheOperation> stores = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
    ASSERT_EQ(stores.size(), 1u);
    const WriteBackBatch& store = std::get<WriteBackBatch>(stores.front());
    std::map<std::uint32_t, std::int32_t> rows_by_group;
    for (const std::uint32_t group : store.group_ids[0]) {
        ++rows_by_group[group];
    }
    EXPECT_EQ(rows_by_group, (std::map<std::uint32_t, std::int32_t>{{0, 2}, {1, 1}, {2, 1}, {3, 1}}));
    EXPECT_EQ(std::ranges::count(store.content_hashes[0], scheduler_->PrefixHashesForTokens(sharing(500)).back()), 1);
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.request_reclaim.blocks, 0u);
    AckWriteBacks(plan);
    SendAbortEvent("f1");
    AckWriteBacks(PlanOnce());
    const Resume after = Probe("after-f1", sharing(1300));
    EXPECT_EQ(after.prefix_tokens, 8);
    EXPECT_EQ(after.load_rows, 0);
}

// The Host is s1's alone and a load of it is in flight: the segment f2's
// admission evicts from the Device cannot be placed without evicting a
// pinned entry, so it is dropped whole -- no copy, no Host eviction, its tail
// hint dropped -- and the admission proceeds.
TEST_P(StateCheckpointPrefillHostShardSuite, ASegmentHostCannotPlaceIsDroppedWhole) {
    host_pages_ = 6;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> tokens = MakeTokens(22);
    CompleteSession("s1", tokens);
    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    SendBootstrapped("f1");
    AckWriteBacks(PlanOnce());
    ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 9);
    ASSERT_EQ(scheduler_->HostPoolFreeBlocks(), 0);
    SendAbortEvent("f1");
    AckWriteBacks(PlanOnce());
    CompleteSession("s2", MakeTokens(22, 2000));
    ASSERT_EQ(scheduler_->TailEntries(), 2u);

    tokens.push_back(77);
    Submit(RequestWithTokens("h", tokens));
    SendBootstrapped("h");
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{22}) << "the admission proceeds";
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(plan).empty());
    const HostCacheStats stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.eviction_stores.no_placement, 9u);
    EXPECT_EQ(stats.eviction_stores.ops, 1u) << "s1's only";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 9);
    EXPECT_EQ(scheduler_->TailEntries(), 1u) << "s2's tail hint left with its blocks";
    for (const CacheOperation& operation : ExtractCacheOpsOfKind<LoadBackBatch>(plan)) {
        for (const std::uint32_t id : std::get<LoadBackBatch>(operation).op_ids) {
            SendLoadBackDone(id);
        }
    }
}

// h is s1's next turn, resumed from the Host: its pages are published on the
// Device again while the Host keeps them. Evicted after h's finish, only what
// h added is copied -- its page at 20, its tail at 26 and its checkpoints --
// and s1's five pages ride as the Host entries they already are.
TEST_P(StateCheckpointPrefillHostShardSuite, ASessionResumedFromHostIsNotCopiedAgain) {
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> tokens = MakeTokens(22);
    CompleteSession("s1", tokens);
    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    SendBootstrapped("f1");
    AckWriteBacks(PlanOnce());
    SendAbortEvent("f1");
    AckWriteBacks(PlanOnce());

    for (std::int32_t i = 0; i < 4; ++i) {
        tokens.push_back(600 + i);
    }
    Submit(RequestWithTokens("h", tokens));
    SendBootstrapped("h");
    const ExecutionPlan first = PlanOnce();
    std::int32_t load_rows = 0;
    for (const CacheOperation& operation : ExtractCacheOpsOfKind<LoadBackBatch>(first)) {
        for (std::size_t i = 0; i < std::get<LoadBackBatch>(operation).op_ids.size(); ++i) {
            load_rows += static_cast<std::int32_t>(std::get<LoadBackBatch>(operation).src_pages[i].size());
            SendLoadBackDone(std::get<LoadBackBatch>(operation).op_ids[i]);
        }
    }
    ASSERT_EQ(load_rows, 9);
    AckWriteBacks(first);
    SendForwardDone("h", {900});
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    SendPdSucceeded("h");
    const std::uint64_t host_cached_before = scheduler_->GetHostCacheStats().eviction_stores.host_cached;

    Submit(RequestWithTokens("f2", MakeTokens(22, 2000)));
    SendBootstrapped("f2");
    const ExecutionPlan plan = PlanOnce();
    const std::vector<CacheOperation> stores = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
    ASSERT_EQ(stores.size(), 1u);
    const WriteBackBatch& store = std::get<WriteBackBatch>(stores.front());
    std::map<std::uint32_t, std::int32_t> rows_by_group;
    for (const std::uint32_t group : store.group_ids[0]) {
        ++rows_by_group[group];
    }
    EXPECT_EQ(rows_by_group, (std::map<std::uint32_t, std::int32_t>{{0, 2}, {1, 1}, {2, 1}, {3, 1}}));
    EXPECT_EQ(scheduler_->GetHostCacheStats().eviction_stores.host_cached - host_cached_before, 5u);
    AckWriteBacks(plan);
}

// KV events keep the Device's view: a written-back boundary is removed when it
// leaves the Device, and its Host publication reports nothing. (A boundary is
// published while every group holds it, so of s1's boundaries only the tail
// with its checkpoints is.)
TEST_P(StateCheckpointPrefillHostShardSuite, KvEventsKeepDeviceSemantics) {
    kv_events_ = true;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    CompleteSession("s1", MakeTokens(22));
    const std::vector<KvCacheEvent> stored = scheduler_->DrainKvEvents();
    ASSERT_EQ(stored.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(stored[0]));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(stored[0]).token_ids, (std::vector<std::int32_t>{21, 22}));
    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    SendBootstrapped("f1");
    const ExecutionPlan plan = PlanOnce();
    ASSERT_EQ(ExtractCacheOpsOfKind<WriteBackBatch>(plan).size(), 1u);
    const std::vector<KvCacheEvent> removed = scheduler_->DrainKvEvents();
    ASSERT_EQ(removed.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<KvBlockRemovedEvent>(removed[0]));
    EXPECT_EQ(std::get<KvBlockRemovedEvent>(removed[0]).block_hashes,
              std::get<KvBlockStoredEvent>(stored[0]).block_hashes);
    AckWriteBacks(plan);
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());
}

// s1's prompt is handed off and its KV transfer is out: its table pins its
// pages. s2 sends s1's prompt and four tokens more; its third chunk is cut
// at the junction 20, so the checkpoint published there hangs under s1's
// pinned pages. f1's first chunk writes s2's suffix back, the segment's climb
// stopping below them. The junction's checkpoint boundary is then the only
// cache left and cannot leave without a pinned page: neither f1's next chunk
// nor f2's first runs, and nothing is written back. Once the transfer finishes,
// s1's pages go to the Host like any cache: f1's next chunk writes s1's prompt
// end back, and the plan after the junction's segment, s1's pages with it.
TEST_P(StateCheckpointPrefillHostShardSuite, PdTransferPinnedPagesAreNeverVictims) {
    device_pages_ = 15;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    const std::array<std::string, 4> group_names{"full", "state0", "state1", "state2"};
    Submit(RequestWithTokens("s1", MakeTokens(22)));
    SendBootstrapped("s1");
    RunPrefill("s1", 22);
    const ExecutionPlan handoff = PlanOnce();
    ASSERT_TRUE(handoff.remote_decode.has_value());
    AckWriteBacks(handoff);
    std::vector<std::int32_t> s2 = MakeTokens(22);
    s2.insert(s2.end(), {500, 501, 502, 503});
    Submit(RequestWithTokens("s2", s2));
    SendBootstrapped("s2");
    const std::vector<Chunk> s2_chunks = RunPrefill("s2", 26);
    const ExecutionPlan s2_handoff = PlanOnce();
    ASSERT_TRUE(s2_handoff.remote_decode.has_value());
    AckWriteBacks(s2_handoff);
    SendPdSucceeded("s2");
    // s2's third chunk ends at the junction: the next chunk's admission
    // publishes its live State blocks there.
    ASSERT_EQ(s2_chunks.size(), 4u);
    ASSERT_EQ(s2_chunks[2].prefix_tokens + s2_chunks[2].tokens, 20);
    // Each hand-off's block table by group: the full group's pages and tail
    // block, each State group's end state in slot 1.
    std::map<std::string, std::vector<std::int32_t>> s1_table;
    std::map<std::string, std::vector<std::int32_t>> s2_table;
    for (const std::string& group : group_names) {
        s1_table[group] = handoff.remote_decode->block_tables.at(group).at(0);
        s2_table[group] = s2_handoff.remote_decode->block_tables.at(group).at(0);
    }
    ASSERT_EQ(s1_table["full"].size(), 6u);
    ASSERT_EQ(s2_table["full"].size(), 7u);
    EXPECT_TRUE(std::equal(s1_table["full"].begin(), s1_table["full"].begin() + 5, s2_table["full"].begin()))
        << "s2 resumes on s1's five pages";

    // By plan: the requests that run, and the rows written back as (group,
    // Device page).
    using Rows = std::vector<std::pair<std::string, std::int32_t>>;
    std::vector<std::vector<std::string>> ran;
    std::vector<Rows> written_back;
    std::map<std::string, std::int32_t> computed;
    for (std::int32_t step = 0; step < 4; ++step) {
        if (step < 2) {
            const std::string id = step == 0 ? "f1" : "f2";
            Submit(RequestWithTokens(id, MakeTokens(22, 1000 * (step + 1))));
            SendBootstrapped(id);
        } else if (step == 2) {
            EXPECT_TRUE(scheduler_->PdTransferPinned("s1"));
            SendPdSucceeded("s1");
        }
        const ExecutionPlan plan = PlanOnce();
        ran.emplace_back();
        if (const ForwardBatch* batch = FindForwardBatch(plan)) {
            ran.back() = batch->request_ids;
            for (std::size_t row = 0; row < batch->request_ids.size(); ++row) {
                const std::string& id = batch->request_ids[row];
                computed[id] = batch->extend_prefix_lens[row] + batch->input_lengths[row];
                SendForwardDone(id, computed[id] == 22 ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
            }
        }
        written_back.emplace_back();
        for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
            const WriteBackBatch& store = std::get<WriteBackBatch>(operation);
            for (std::size_t op = 0; op < store.op_ids.size(); ++op) {
                for (std::size_t row = 0; row < store.src_pages[op].size(); ++row) {
                    written_back.back().emplace_back(group_names[store.group_ids[op][row]], store.src_pages[op][row]);
                }
            }
        }
        AckWriteBacks(plan);
    }
    EXPECT_EQ(ran, (std::vector<std::vector<std::string>>{{"f1"}, {}, {"f1"}, {"f1", "f2"}}));
    // A segment's checkpoints first, a tail's KV block leaving with the first
    // of them, then its pages from the deepest up.
    const Rows s2_suffix{{"state0", s2_table["state0"][1]},
                         {"full", s2_table["full"][6]},
                         {"state1", s2_table["state1"][1]},
                         {"state2", s2_table["state2"][1]},
                         {"full", s2_table["full"][5]}};
    const Rows s1_prompt_end{{"state0", s1_table["state0"][1]},
                             {"full", s1_table["full"][5]},
                             {"state1", s1_table["state1"][1]},
                             {"state2", s1_table["state2"][1]}};
    const Rows junction{{"state0", s2_chunks[2].state_rows[0][1]},
                        {"state1", s2_chunks[2].state_rows[1][1]},
                        {"state2", s2_chunks[2].state_rows[2][1]},
                        {"full", s1_table["full"][4]},
                        {"full", s1_table["full"][3]},
                        {"full", s1_table["full"][2]},
                        {"full", s1_table["full"][1]},
                        {"full", s1_table["full"][0]}};
    EXPECT_EQ(written_back, (std::vector<Rows>{s2_suffix, {}, s1_prompt_end, junction}))
        << "s1's pages leave only once its transfer finished";
}

// s1's 8-token session leaves its prompt-end checkpoint at 8. big sends s1's
// prompt and more, up to the single-request bound of 11 usable parents: it
// resumes at 8, and its last chunk needs every parent, five for its KV and six
// for its State slots. Once big's KV reaches its third parent, the checkpoint
// at 8 is the only cache left, and its segment holds big's own pages at and
// under 8, in use: no whole segment can leave. Alone, big has no release to
// wait for, so the checkpoint leaves block by block -- every State group's at
// once, written back -- the pages big holds stay, and big runs to its hand-off
// with no idle round. The checkpoint lands on the Host, where s1's next turn
// resumes from it over the Device's KV.
TEST_F(StateCheckpointPrefillHostSuite, APromptAloneEvictsTheCheckpointOnItsOwnPathBlockByBlock) {
    device_pages_ = 12;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> s1 = MakeTokens(8);
    const ExecutionPlan s1_handoff = CompleteSession("s1", s1);
    const std::int32_t prompt = scheduler_->MaxSingleRequestTokens() - 1;
    std::vector<std::int32_t> big = s1;
    const std::vector<std::int32_t> rest = MakeTokens(prompt - 8, 3000);
    big.insert(big.end(), rest.begin(), rest.end());
    Submit(RequestSpec{.request_id = "big", .tokens = big, .max_new_tokens = 1});
    SendBootstrapped("big");

    using Rows = std::vector<std::pair<std::string, std::int32_t>>;
    const std::array<std::string, 4> group_names{"full", "state0", "state1", "state2"};
    Rows written_back;
    std::int32_t resumed_at = -1;
    std::int32_t computed = 0;
    std::optional<ExecutionPlan> handoff;
    for (std::int32_t round = 0; round < 16 && !handoff; ++round) {
        ExecutionPlan plan = PlanOnce();
        for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
            const WriteBackBatch& store = std::get<WriteBackBatch>(operation);
            for (std::size_t op = 0; op < store.op_ids.size(); ++op) {
                for (std::size_t row = 0; row < store.src_pages[op].size(); ++row) {
                    written_back.emplace_back(group_names[store.group_ids[op][row]], store.src_pages[op][row]);
                }
            }
        }
        AckWriteBacks(plan);
        if (plan.remote_decode.has_value()) {
            handoff = std::move(plan);
            break;
        }
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_TRUE(batch != nullptr && batch->request_ids == std::vector<std::string>{"big"})
            << "big waits at " << computed << " of " << prompt << " tokens";
        if (resumed_at < 0) {
            resumed_at = batch->extend_prefix_lens[0];
        }
        computed = batch->extend_prefix_lens[0] + batch->input_lengths[0];
        SendForwardDone("big", computed == prompt ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
    }
    ASSERT_TRUE(handoff.has_value());
    EXPECT_EQ(resumed_at, 8);
    const Rows checkpoint{{"state0", s1_handoff.remote_decode->block_tables.at("state0").at(0)[1]},
                          {"state1", s1_handoff.remote_decode->block_tables.at("state1").at(0)[1]},
                          {"state2", s1_handoff.remote_decode->block_tables.at("state2").at(0)[1]}};
    EXPECT_EQ(written_back, checkpoint) << "only the checkpoint at 8 leaves";
    SendPdSucceeded("big");
    AckWriteBacks(PlanOnce());

    s1.push_back(77);
    const Resume resumed = Probe("s1-next", s1);
    EXPECT_EQ(resumed.prefix_tokens, 8);
    EXPECT_EQ(resumed.load_rows, 3) << "the three checkpoints load from the Host";
}

// a's prompt is handed off and its transfer is out while s1, s2 and then b
// run: b's prefill needs room, and of the cache it evicts s1 and writes it
// back, writing back no page a's transfer pins. Then a's transfer fails
// and b is aborted after its own hand-off: both free their tables at once
// (their prompt pages stay cached; the prompt ends were never published).
// In the next plan f1 and f2 take every block the two released -- each
// one's KV block after its prompt and its three State blocks -- and the room
// of s2, evicted and written back by a stream-ordered store in that same
// plan, which reads the parents of s2's group-1 and group-2 checkpoints
// while f1 and f2 are granted them. The plan after carries no record.
TEST_F(StateCheckpointPrefillHostSuite, AFailedAndAnAbortedHandOffFreeTheirPagesForTheSamePlansWriteBack) {
    device_pages_ = 19;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    const std::array<std::string, 4> group_names{"full", "state0", "state1", "state2"};
    // Each hand-off's block table by group: the pages its transfer pins.
    std::map<std::string, std::map<std::string, std::vector<std::int32_t>>> pinned;
    for (const auto& [id, start] : {std::pair{"a", 500}, std::pair{"b", 700}}) {
        if (std::string(id) == "b") {
            CompleteSession("s1", MakeTokens(22));
            CompleteSession("s2", MakeTokens(22, 300));
        }
        Submit(RequestWithTokens(id, MakeTokens(20, start)));
        SendBootstrapped(id);
        std::vector<std::int32_t> written_back;
        for (std::int32_t chunk = 0; chunk < 3; ++chunk) {
            const ExecutionPlan plan = PlanOnce();
            for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
                const WriteBackBatch& store = std::get<WriteBackBatch>(operation);
                ASSERT_EQ(store.op_ids.size(), 1u);
                for (std::size_t row = 0; row < store.src_pages[0].size(); ++row) {
                    written_back.push_back(store.src_pages[0][row]);
                    if (pinned.contains("a")) {
                        EXPECT_TRUE(scheduler_->PdTransferPinned("a"));
                        EXPECT_EQ(std::ranges::count(pinned["a"][group_names[store.group_ids[0][row]]],
                                                     store.src_pages[0][row]),
                                  0)
                            << "b's admission writes back a page a's transfer pins";
                    }
                }
            }
            SendForwardDone(id, chunk == 2 ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
            AckWriteBacks(plan);
        }
        const ExecutionPlan handoff = PlanOnce();
        ASSERT_TRUE(handoff.remote_decode.has_value());
        for (const std::string& group : group_names) {
            pinned[id][group] = handoff.remote_decode->block_tables.at(group).at(0);
        }
        AckWriteBacks(handoff);
        const std::vector<std::int32_t> s1_written_back{8, 24, 9, 10, 23, 22, 21, 44, 43};
        EXPECT_EQ(written_back, std::string(id) == "a" ? std::vector<std::int32_t>{} : s1_written_back)
            << "b's prefill writes s1 back";
    }
    EXPECT_EQ(pinned["a"]["full"], (std::vector<std::int32_t>{1, 2, 3, 4, 41, 42}));
    EXPECT_EQ(pinned["b"]["full"], (std::vector<std::int32_t>{43, 44, 63, 64, 25, 26}));
    EXPECT_EQ(scheduler_->PdTransferSize(), 2u);
    EXPECT_EQ(scheduler_->EmptyLcmBlocks(), 4);

    ExecutionEvent failed;
    failed.With(pd::FailedEvent{"a"});
    scheduler_->Advance(std::move(failed));
    SendAbortEvent("b");
    EXPECT_EQ(scheduler_->PdTransferSize(), 0u);
    EXPECT_EQ(scheduler_->EmptyLcmBlocks(), 10) << "each frees its three State parents";

    for (const auto& [id, start] : {std::pair{"f1", 1000}, std::pair{"f2", 2000}}) {
        Submit(RequestWithTokens(id, MakeTokens(4, start)));
        SendBootstrapped(id);
    }
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"f1", "f2"}));
    const std::vector<CacheOperation> stores = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
    ASSERT_EQ(stores.size(), 1u);
    const WriteBackBatch& store = std::get<WriteBackBatch>(stores.front());
    ASSERT_EQ(store.op_ids.size(), 1u);
    EXPECT_EQ(store.source_pinned, std::vector<bool>{false});
    EXPECT_EQ(store.group_ids[0], (std::vector<std::uint32_t>{1, 0, 2, 3, 0, 0, 0, 0, 0}));
    EXPECT_EQ(store.src_pages[0], (std::vector<std::int32_t>{4, 62, 2, 18, 61, 60, 59, 58, 57}))
        << "s2: its three checkpoints and six KV blocks";
    // Every State block is a parent of its own, so a page id names the
    // parent whichever State group takes it.
    std::set<std::int32_t> state_zeroed;
    for (const std::string& group : {"state0", "state1", "state2"}) {
        state_zeroed.insert(plan.pages_to_zero.at(group).begin(), plan.pages_to_zero.at(group).end());
    }
    EXPECT_EQ(state_zeroed, (std::set<std::int32_t>{2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17}));
    EXPECT_EQ(plan.pages_to_zero.at("full"), (std::vector<std::int32_t>{42, 26, 27, 28}))
        << "the KV blocks after a's and b's prompts, then the free slots beside b's last page";
    for (const std::string& id : {"a", "b"}) {
        EXPECT_EQ(std::ranges::count(plan.pages_to_zero.at("full"), pinned[id]["full"].back()), 1) << id;
        for (const std::string& group : {"state0", "state1", "state2"}) {
            EXPECT_EQ(pinned[id][group].at(0), 0) << id << " " << group;
            EXPECT_TRUE(state_zeroed.contains(pinned[id][group].at(1))) << id << " " << group;
        }
    }
    EXPECT_TRUE(state_zeroed.contains(store.src_pages[0][0])) << "s2's group-1 checkpoint";
    EXPECT_TRUE(state_zeroed.contains(store.src_pages[0][2])) << "s2's group-2 checkpoint";
    const HostCacheStats stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.eviction_stores.ops, 2u);
    EXPECT_EQ(stats.eviction_stores.no_placement, 0u);
    AckWriteBacks(plan);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 18) << "s1 and s2, nine blocks each";

    const ExecutionPlan next = PlanOnce();
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(next).empty());
    EXPECT_EQ(scheduler_->GetHostCacheStats().eviction_stores.ops, 2u);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 18);
}

// Two schedulers fed the same sessions and events build the same write-back
// batches, byte for byte: every decision rests on scheduler state alone.
TEST_P(StateCheckpointPrefillHostShardSuite, WriteBackPlansMatchOnMirroredSchedulers) {
    device_pages_ = 16;
    const auto run = [&] {
        Reset(/*host_cache=*/true, /*decode_width=*/1);
        std::vector<WriteBackBatch> batches;
        CompleteSession("a", MakeTokens(22));
        std::vector<std::int32_t> b = MakeTokens(26, 500);
        std::copy_n(MakeTokens(8).begin(), 8, b.begin());
        CompleteSession("b", b);
        for (const auto& [id, start] : {std::pair{"f1", 1000}, std::pair{"f2", 2000}}) {
            Submit(RequestWithTokens(id, MakeTokens(22, start)));
            SendBootstrapped(id);
            for (std::int32_t chunk = 0; chunk < 3; ++chunk) {
                const ExecutionPlan plan = PlanOnce();
                for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
                    batches.push_back(std::get<WriteBackBatch>(operation));
                }
                AckWriteBacks(plan);
                SendForwardDone(id, chunk == 2 ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
            }
            AckWriteBacks(PlanOnce());
            SendPdSucceeded(id);
        }
        return batches;
    };
    const std::vector<WriteBackBatch> first = run();
    const std::vector<WriteBackBatch> second = run();
    ASSERT_EQ(first.size(), 3u);
    ASSERT_EQ(second.size(), first.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_EQ(second[i].op_ids, first[i].op_ids);
        EXPECT_EQ(second[i].group_ids, first[i].group_ids);
        EXPECT_EQ(second[i].src_pages, first[i].src_pages);
        EXPECT_EQ(second[i].dst_pages, first[i].dst_pages);
        EXPECT_EQ(second[i].content_hashes, first[i].content_hashes);
        EXPECT_EQ(second[i].page_offsets, first[i].page_offsets);
        EXPECT_EQ(second[i].source_pinned, first[i].source_pinned);
    }
}

// Six sessions take turns, each turn its previous prompt plus four tokens,
// over a Device of 22 parents and a Host of 18. Either tier alone holds three
// of them beside the turn running (with write-through, a fourth session's
// growing turns already miss); together they hold seven. So with write-back
// every turn after the first lap resumes exactly where its session's
// previous turn ended, while write-through misses every one of them: its
// Host keeps the newest sessions, a copy of what the Device holds.
TEST_P(StateCheckpointPrefillHostShardSuite, WriteBackHoldsMoreSessionsThanEitherTier) {
    constexpr std::int32_t kSessions = 6;
    constexpr std::int32_t kLaps = 4;
    device_pages_ = 23;
    host_pages_ = 19;
    for (const auto policy :
         {SchedulerConfig::HostWritePolicy::kWriteBack, SchedulerConfig::HostWritePolicy::kWriteThrough}) {
        SCOPED_TRACE(static_cast<int>(policy));
        write_policy_ = policy;
        Reset(/*host_cache=*/true, /*decode_width=*/1);
        std::vector<std::vector<std::int32_t>> prompts;
        for (std::int32_t j = 0; j < kSessions; ++j) {
            prompts.push_back(MakeTokens(10, 1000 * (j + 1)));
        }
        // Per turn after the first lap: where it resumed and where its
        // session's previous turn ended.
        std::vector<std::int32_t> hits;
        std::vector<std::int32_t> previous_ends;
        for (std::int32_t lap = 0; lap < kLaps; ++lap) {
            for (std::int32_t j = 0; j < kSessions; ++j) {
                std::vector<std::int32_t>& prompt = prompts[static_cast<std::size_t>(j)];
                const auto previous_end = static_cast<std::int32_t>(prompt.size());
                if (lap > 0) {
                    for (std::int32_t i = 0; i < 4; ++i) {
                        prompt.push_back(100000 * (j + 1) + 10 * lap + i);
                    }
                }
                const std::string id = "s" + std::to_string(j) + "-" + std::to_string(lap);
                Submit(RequestWithTokens(id, prompt));
                SendBootstrapped(id);
                std::optional<std::int32_t> hit;
                // Prefill to the hand-off, every copy acknowledged at once.
                for (std::int32_t round = 0; round < 4; ++round) {
                    const ExecutionPlan plan = PlanOnce();
                    for (const CacheOperation& operation : ExtractCacheOpsOfKind<LoadBackBatch>(plan)) {
                        for (const std::uint32_t op : std::get<LoadBackBatch>(operation).op_ids) {
                            SendLoadBackDone(op);
                        }
                    }
                    AckWriteBacks(plan);
                    if (plan.remote_decode.has_value()) {
                        break;
                    }
                    const ForwardBatch* batch = FindForwardBatch(plan);
                    ASSERT_NE(batch, nullptr);
                    ASSERT_EQ(batch->request_ids, std::vector<std::string>{id});
                    hit = hit.value_or(batch->extend_prefix_lens.at(0));
                    const bool completes =
                        batch->extend_prefix_lens.at(0) + batch->input_lengths.at(0) == batch->prefill_lengths.at(0);
                    SendForwardDone(id, completes ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
                }
                SendPdSucceeded(id);
                AckWriteBacks(PlanOnce());
                if (lap > 0) {
                    hits.push_back(*hit);
                    previous_ends.push_back(previous_end);
                }
            }
        }
        if (policy == SchedulerConfig::HostWritePolicy::kWriteBack) {
            EXPECT_EQ(hits, previous_ends);
        } else {
            EXPECT_EQ(hits, std::vector<std::int32_t>(hits.size(), 0));
        }
    }
}

// s1's first decode publishes its prompt end, 22 -- the tail block and three
// checkpoints -- and nothing streams to the Host. One decode later s1 no
// longer reads those checkpoints, and f1's first chunk, beside s1's decode in
// one mixed batch, takes their room: the plan that evicts the tail copies it
// to the Host, stream-ordered, and the ACK publishes its four blocks there. A
// prompt that leaves s1 at 22 loads them back.
TEST_F(StateCheckpointFusedHostSuite, ThePromptEndTheFirstDecodePublishesLeavesWithItsWriteBack) {
    device_pages_ = 16;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    config_.enable_mixed_prefill_decode = true;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    const std::vector<std::int32_t> s1 = MakeTokens(22);
    Submit(RequestWithTokens("s1", s1));
    RunPrefill("s1", 22);
    const ExecutionPlan first_decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(first_decode), nullptr);
    ASSERT_EQ(FindForwardBatch(first_decode)->request_ids, std::vector<std::string>{"s1"});
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(first_decode).empty()) << "nothing streams at publication";
    SendForwardDone("s1", {901});

    Submit(RequestWithTokens("f1", MakeTokens(4, 1000)));
    const ExecutionPlan plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan), nullptr);
    EXPECT_EQ(FindForwardBatch(plan)->request_ids, (std::vector<std::string>{"f1", "s1"}));
    const std::vector<CacheOperation> stores = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
    ASSERT_EQ(stores.size(), 1u);
    const WriteBackBatch& store = std::get<WriteBackBatch>(stores.front());
    ASSERT_EQ(store.op_ids.size(), 1u);
    EXPECT_EQ(store.source_pinned, std::vector<bool>{false});
    EXPECT_EQ(store.group_ids[0], (std::vector<std::uint32_t>{1, 0, 2, 3})) << "the tail block and its checkpoints";
    const HostCacheStats stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.publication_stores.ops + stats.finish_stores.ops, 0u);
    EXPECT_EQ(stats.eviction_stores.ops, 1u);
    EXPECT_EQ(stats.device_evictions.admission.blocks, 4u);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0) << "published at the ACK, not before";
    AckWriteBacks(plan);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 4);

    SendForwardDone("s1", {902});
    SendForwardDone("f1", {903});
    SendFinish("s1");
    SendFinish("f1");
    AckWriteBacks(PlanOnce());
    std::vector<std::int32_t> branch = s1;
    branch.push_back(77);
    const Resume resumed = Probe("branch", branch);
    EXPECT_EQ(resumed.prefix_tokens, 22);
    EXPECT_EQ(resumed.load_rows, 4);
}

// A decoding request holds the KV pages under its prompt end. s1's prompt ends
// on a page boundary, 20: its first decode publishes the State checkpoints
// there, and one decode later s1 no longer reads them, but under write-back
// they leave only with the pages they need, which s1 holds while it decodes.
// f1's first chunk passes them over though they are older than s0's session,
// which leaves instead; once s1 finishes, f2's first chunk takes the three
// checkpoints at s1's prompt end, with s1's end-state tail, and writes them
// back; the prompt's KV pages stay.
TEST_F(StateCheckpointFusedHostSuite, APageAlignedPromptEndLeavesOnlyAfterItsRequestFinishes) {
    device_pages_ = 20;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    const std::vector<std::int32_t> s1 = MakeTokens(20);
    std::vector<std::int32_t> s1_next = s1;
    s1_next.push_back(77);
    const std::string prompt_end = scheduler_->PrefixHashesForTokens(s1_next).at(4);
    Submit(RequestWithTokens("s1", s1));
    RunPrefill("s1", 20);
    AckWriteBacks(PlanOnce());
    SendForwardDone("s1", {901});
    AckWriteBacks(PlanOnce());
    SendForwardDone("s1", {902});

    // Every plan's copies acknowledged at once; each landed row gets a token,
    // and a request in finish_on_token finishes at it. Returns how many of
    // the plans' write-back rows carry prompt_end.
    std::set<std::string> finish_on_token{"s0"};
    const auto run = [&](std::int32_t plans) {
        std::int64_t at_prompt_end = 0;
        for (std::int32_t i = 0; i < plans; ++i) {
            const ExecutionPlan plan = PlanOnce();
            for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
                for (const std::vector<std::string>& hashes : std::get<WriteBackBatch>(operation).content_hashes) {
                    at_prompt_end += std::ranges::count(hashes, prompt_end);
                }
            }
            AckWriteBacks(plan);
            const ForwardBatch* batch = FindForwardBatch(plan);
            for (std::size_t row = 0; batch != nullptr && row < batch->request_ids.size(); ++row) {
                const std::string& id = batch->request_ids[row];
                if (row < batch->NumExtends() &&
                    batch->extend_prefix_lens[row] + batch->input_lengths[row] < batch->prefill_lengths[row]) {
                    SendForwardDone(id);
                    continue;
                }
                SendForwardDone(id, {903});
                if (finish_on_token.contains(id)) {
                    SendFinish(id);
                }
            }
        }
        return at_prompt_end;
    };
    Submit(RequestWithTokens("s0", MakeTokens(6, 500)));
    run(4);
    ASSERT_EQ(scheduler_->RequestTokenSize("s0"), -1);
    const HostCacheStats before = scheduler_->GetHostCacheStats();
    Submit(RequestWithTokens("f1", MakeTokens(8, 1000)));
    EXPECT_EQ(run(2), 0) << "s1 decodes on the pages its checkpoints need";
    const HostCacheStats after = scheduler_->GetHostCacheStats();
    EXPECT_GT(after.eviction_stores.ops, before.eviction_stores.ops) << "f1 took s0's room instead";
    finish_on_token = {"s1", "f1"};
    run(1);
    Submit(RequestWithTokens("f2", MakeTokens(30, 2000)));
    EXPECT_EQ(run(1), 3) << "s1's prompt left with its three checkpoints";
}

// A clear refuses while an eviction write-back is out -- its Host blocks are
// taken and its tree node held until the ACK publishes it -- and goes ahead
// once the ACK lands. f1 took s1's end state at 23, the tail no turn has read
// yet: the Host holds it, and s1's next turn resumes there, loading it. The
// clear drops s1's prompt, which only the Device held, so the Host tail
// continues a prefix no probe reaches and the next turn starts over.
TEST_F(StateCheckpointFusedHostSuite, AClearWaitsForTheEvictionWriteBackInFlight) {
    device_pages_ = 16;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    const std::vector<std::int32_t> s1 = MakeTokens(22);
    Submit(RequestWithTokens("s1", s1));
    RunPrefill("s1", 22);
    AckWriteBacks(PlanOnce());
    SendForwardDone("s1", {901});
    SendFinish("s1");
    AckWriteBacks(PlanOnce());

    Submit(RequestWithTokens("f1", MakeTokens(22, 1000)));
    const ExecutionPlan eviction = PlanOnce();
    ASSERT_EQ(ExtractCacheOpsOfKind<WriteBackBatch>(eviction).size(), 1u);
    SendForwardDone("f1");
    SendAbortEvent("f1");
    const ExecutionPlan after_abort = PlanOnce();
    EXPECT_FALSE(scheduler_->CanClearCache());
    EXPECT_FALSE(scheduler_->ClearL1Cache()) << "the eviction write-back is in flight";
    AckWriteBacks(eviction);
    AckWriteBacks(after_abort);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 4) << "the tail block and three checkpoints";
    std::vector<std::int32_t> next = s1;
    next.push_back(900);
    next.push_back(77);
    const Resume before_clear = Probe("s1-next", next);
    EXPECT_EQ(before_clear.prefix_tokens, 23);
    EXPECT_EQ(before_clear.load_rows, 4);

    EXPECT_TRUE(scheduler_->ClearL1Cache());
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 4) << "a Device-only clear leaves the Host as it is";
    const Resume after_clear = Probe("s1-next-again", next);
    EXPECT_EQ(after_clear.prefix_tokens, 0);
    EXPECT_EQ(after_clear.load_rows, 0);
}

// A request as long as the Device holds, behind a finished session's cache:
// its admissions evict that cache, written back plan by plan, and it runs to
// its last token without a retraction or an abort; then nothing is pinned
// or in flight.
TEST_F(StateCheckpointFusedHostSuite, ARequestTheSizeOfTheDeviceRunsToItsEnd) {
    device_pages_ = 16;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    Submit(RequestWithTokens("s1", MakeTokens(22)));
    RunPrefill("s1", 22);
    AckWriteBacks(PlanOnce());
    SendForwardDone("s1", {901});
    SendFinish("s1");
    AckWriteBacks(PlanOnce());

    // The prompt and its 32 generated tokens fill the request token limit.
    const std::int32_t prompt = scheduler_->MaxSingleRequestTokens() - 32;
    Submit(RequestWithTokens("big", MakeTokens(prompt, 3000)));
    std::int32_t generated = 0;
    for (std::int32_t round = 0; round < 64 && generated < 32; ++round) {
        const ExecutionPlan plan = PlanOnce();
        EXPECT_TRUE(plan.aborts.empty());
        EXPECT_EQ(FindSnapshotStore(plan), nullptr) << "nothing is retracted";
        AckWriteBacks(plan);
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{"big"});
        if (batch->NumExtends() == 1 &&
            batch->extend_prefix_lens[0] + batch->input_lengths[0] < batch->prefill_lengths[0]) {
            SendForwardDone("big");
            continue;
        }
        SendForwardDone("big", {4000 + generated});
        if (++generated == 32) {
            SendFinish("big");
        }
    }
    EXPECT_EQ(generated, 32);
    AckWriteBacks(PlanOnce());
    EXPECT_EQ(scheduler_->RequestTokenSize("big"), -1);
    EXPECT_GT(scheduler_->GetHostCacheStats().eviction_stores.ops, 0u);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
    EXPECT_TRUE(scheduler_->CanClearCache());
}

// The in-forward checkpoint's block is optional, and alone under write-back
// it never makes blocks leave one by one: a chunk that could hold it only so
// ends at the boundary instead, as in a pool with no room for it
// (AChunkEndsAtTheBoundaryWhenTheCheckpointBlockDoesNotFit). s0's 24-token
// session leaves its checkpoints at 20 and 24 and its end state at 25 on a
// 14-parent Device. big sends s0's prompt and more, 46 tokens, and resumes at
// 24; its chunks at 24 and 32 evict the checkpoint at 20 and the end state,
// each a set of its own. The chunk at 40 spans the boundary 44, and its
// checkpoint's blocks could only be the checkpoint at 24, whose segment holds
// big's pages under it: the chunk ends at 44, nothing more leaves, and a
// prompt continuing s0's still resumes at 24 on the Device.
TEST_F(StateCheckpointFusedHostSuite, AnInForwardCheckpointNeverEvictsBlockByBlock) {
    in_forward_ = true;
    device_pages_ = 14;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    std::vector<std::int32_t> s0 = MakeTokens(24);
    Submit(RequestWithTokens("s0", s0));
    RunPrefill("s0", 24);
    AckWriteBacks(PlanOnce());
    SendForwardDone("s0", {901});
    SendFinish("s0");
    AckWriteBacks(PlanOnce());

    Submit(RequestSpec{.request_id = "big", .tokens = MakeTokens(46), .max_new_tokens = 4});
    const HostCacheStats before = scheduler_->GetHostCacheStats();
    const std::vector<Chunk> chunks = RunPrefill("big", 46);
    std::vector<std::array<std::int32_t, 3>> shapes;
    for (const Chunk& chunk : chunks) {
        shapes.push_back({chunk.prefix_tokens, chunk.tokens, chunk.checkpoint_len});
    }
    EXPECT_EQ(shapes, (std::vector<std::array<std::int32_t, 3>>{{24, 8, 0}, {32, 8, 0}, {40, 4, 0}, {44, 2, 0}}))
        << "{prefix, tokens, checkpoint} per chunk: a cut at 44";
    const HostCacheStats after = scheduler_->GetHostCacheStats();
    EXPECT_EQ(after.eviction_stores.ops - before.eviction_stores.ops, 2u);
    EXPECT_EQ(after.device_evictions.admission.blocks - before.device_evictions.admission.blocks, 7u)
        << "the checkpoint at 20 and the end state's tail block and checkpoints";
    SendAbortEvent("big");
    AckWriteBacks(PlanOnce());

    s0.push_back(77);
    const Resume resumed = Probe("s0-next", s0);
    EXPECT_EQ(resumed.prefix_tokens, 24);
    EXPECT_EQ(resumed.load_rows, 0) << "the checkpoint at 24 stayed on the Device";
}

// A restore that evicts cache only the Device holds writes it back in its own
// plan, the store placed ahead of the restore that reuses its pages: the
// runtime runs a plan's stores before its loads. r is retracted by the knob
// (every fifth plan) and its image held; s's session finishes while r is
// out, leaving Device-only cache, and q runs. When r's image lands, r's
// restore takes s's pages beside q's decode.
TEST_F(StateCheckpointFusedHostSuite, ARestoresEvictionStoreRunsBeforeTheRestoreInTheSamePlan) {
    device_pages_ = 16;
    Reset(/*host_cache=*/true, /*decode_width=*/1);
    config_.debug_force_retraction_interval = 5;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    Submit(RequestWithTokens("r", MakeTokens(14)));
    std::optional<ExecutionPlan> image;
    std::optional<ExecutionPlan> restore;
    for (std::int32_t round = 0; round < 12 && !restore; ++round) {
        if (round == 5) {
            Submit(RequestWithTokens("s", MakeTokens(10, 700)));
        }
        if (round == 9) {
            Submit(RequestWithTokens("q", MakeTokens(4, 900)));
        }
        if (round == 11) {
            ASSERT_TRUE(image.has_value()) << "the knob retracted r";
            AckImageStores(*image);
        }
        const HostCacheStats before = scheduler_->GetHostCacheStats();
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        for (std::size_t row = 0; batch != nullptr && row < batch->request_ids.size(); ++row) {
            const std::string& id = batch->request_ids[row];
            const bool chunk_left =
                row < batch->NumExtends() &&
                batch->extend_prefix_lens[row] + batch->input_lengths[row] < batch->prefill_lengths[row];
            SendForwardDone(id, chunk_left ? std::vector<std::int32_t>{} : std::vector<std::int32_t>{903});
            if (!chunk_left && id == "s") {
                SendFinish("s");
            }
        }
        if (FindSnapshotStore(plan) != nullptr) {
            ASSERT_EQ(FindSnapshotStore(plan)->request_ids, std::vector<std::string>{"r"});
            image = std::move(plan);
            continue;
        }
        AckWriteBacks(plan);
        if (FindRestore(plan) != nullptr) {
            const HostCacheStats after = scheduler_->GetHostCacheStats();
            EXPECT_GT(after.device_evictions.restore.blocks, before.device_evictions.restore.blocks);
            EXPECT_EQ(after.eviction_stores.ops, before.eviction_stores.ops + 1);
            restore = std::move(plan);
        }
    }
    ASSERT_TRUE(restore.has_value());
    EXPECT_EQ(FindRestore(*restore)->request_ids, std::vector<std::string>{"r"});
    std::optional<std::size_t> store_at;
    std::optional<std::size_t> restore_at;
    for (std::size_t i = 0; i < restore->Operations().size(); ++i) {
        if (const auto* cache = std::get_if<CacheOperation>(&restore->Operations()[i])) {
            if (std::holds_alternative<WriteBackBatch>(*cache)) {
                store_at = i;
            } else if (std::holds_alternative<SnapshotRestoreBatch>(*cache)) {
                restore_at = i;
            }
        }
    }
    ASSERT_TRUE(store_at.has_value() && restore_at.has_value());
    EXPECT_LT(*store_at, *restore_at);
    const WriteBackBatch& store = std::get<WriteBackBatch>(std::get<CacheOperation>(restore->Operations()[*store_at]));
    ASSERT_EQ(store.op_ids.size(), 1u);
    EXPECT_EQ(store.source_pinned, std::vector<bool>{false});
    // Some restored page is one the store reads.
    std::set<std::pair<std::uint32_t, std::int32_t>> read;
    for (std::size_t row = 0; row < store.src_pages[0].size(); ++row) {
        read.emplace(store.group_ids[0][row], store.src_pages[0][row]);
    }
    const SnapshotRestoreBatch& restored = *FindRestore(*restore);
    bool reused = false;
    for (std::size_t row = 0; row < restored.dst_pages[0].size(); ++row) {
        reused = reused || read.contains({restored.group_ids[0][row], restored.dst_pages[0][row]});
    }
    EXPECT_TRUE(reused);
}

}  // namespace tokenspeed::test
