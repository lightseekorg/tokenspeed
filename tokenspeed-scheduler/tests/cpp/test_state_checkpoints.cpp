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
                .attn_kind = is_state ? AttnKind::kState : AttnKind::kFull,
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
    AckRestores(readmit);
    // The resumed decode releases the claimed slot 0 and writes its copy of
    // slot 1 in place.
    const std::vector<std::int32_t> resumed = PlanAndStateRow("r");
    EXPECT_EQ(resumed[0], 0);
    EXPECT_GT(resumed[1], 0);
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

}  // namespace tokenspeed::test
