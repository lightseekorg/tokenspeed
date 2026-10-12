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
        cfg.max_scheduled_tokens = 8;
        cfg.max_batch_size = 2;
        cfg.disable_l2_cache = true;
        for (const std::string& id : {"full", "state0", "state1", "state2"}) {
            const bool is_state = id != "full";
            cfg.cache_groups.push_back(CacheGroupConfig{
                .group_id = id,
                .block_granularity = is_state ? 0 : kP,
                .total_pages = cfg.device_allocator.total_pages,
                .retention = CacheGroupConfig::Retention::FullHistory,
                .family = is_state ? CacheGroupFamily::State : CacheGroupFamily::History,
            });
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void Reset(bool host_cache, std::int32_t decode_width) {
        config_ = MakeConfig();
        config_.disable_l2_cache = !host_cache;
        config_.decode_input_tokens = decode_width;
        scheduler_ = std::make_unique<Scheduler>(config_);
    }

    static RequestSpec RequestWithTokens(const std::string& id, std::vector<std::int32_t> tokens) {
        return RequestSpec{.request_id = id, .tokens = std::move(tokens), .max_new_tokens = 32};
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

    // Finish publishes the frontier (14 computed tokens: the prompt and the
    // four consumed generated tokens) and its state.
    SendForwardDone("r", {105});
    SendFinish("r");
    PlanOnce();
    EXPECT_EQ(scheduler_->TailEntries(), 2u);
    const Resume at_finish = Probe("finish-hit", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 101, 102, 103, 104, 77});
    EXPECT_EQ(at_finish.prefix_tokens, 14);
    EXPECT_EQ(at_finish.state_row.at(0), decode1[1]) << "the live slot published at finish";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks())
        << "every block is cache-only or free";
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
        reserve.With(forward::UpdateReserveNumTokens{.request_id = "r", .reserve_num_tokens_in_next_schedule_event = 3});
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
        PlanOnce();
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
        specs.push_back(RequestSpec{
            .request_id = "r" + std::to_string(i), .tokens = MakeTokens(kP, 100 * (i + 1)), .max_new_tokens = kMaxNewTokens});
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

}  // namespace tokenspeed::test
