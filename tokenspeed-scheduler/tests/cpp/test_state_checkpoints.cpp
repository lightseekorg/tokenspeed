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
// published at prompt end, finish and retraction; the opt-in in-forward and
// frequent checkpoints; hits resuming from them; the two-blocks-per-request
// capacity; and the retraction image of both slots.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
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

    // The first `shared` tokens of `tokens`, then two of the prompt's own:
    // a prompt that diverges from `tokens` after `shared` tokens.
    static std::vector<std::int32_t> Diverging(const std::vector<std::int32_t>& tokens, std::int32_t shared) {
        std::vector<std::int32_t> out(tokens.begin(), tokens.begin() + shared);
        out.push_back(990);
        out.push_back(991);
        return out;
    }

    // The (group id, content hash) rows of every write-back a plan issues.
    static std::vector<std::pair<std::uint32_t, std::string>> WriteBackRows(const ExecutionPlan& plan) {
        std::vector<std::pair<std::uint32_t, std::string>> rows;
        for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
            const auto& batch = std::get<WriteBackBatch>(operation);
            for (std::size_t op = 0; op < batch.op_ids.size(); ++op) {
                for (std::size_t row = 0; row < batch.group_ids[op].size(); ++row) {
                    rows.emplace_back(batch.group_ids[op][row], batch.content_hashes[op][row]);
                }
            }
        }
        return rows;
    }

    // One prefill forward of a request, as the planner shaped it.
    struct Chunk {
        std::int32_t prefix_tokens{0};
        std::int32_t tokens{0};
        std::int32_t checkpoint_len{0};
        // The state0..state2 rows.
        std::vector<std::vector<std::int32_t>> state_rows;
        std::vector<std::int32_t> zeroed_state0;
        // The rows of the plan's write-backs (WriteBackRows).
        std::vector<std::pair<std::uint32_t, std::string>> write_backs;
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
                        .zeroed_state0 = plan.pages_to_zero.at("state0"),
                        .write_backs = WriteBackRows(plan)};
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

    // Mixed mode (SchedulerConfig::enable_mixed_prefill_decode): one decoding
    // request takes one token of every round's 8 and leaves 7 for a prefill.
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

// Frequent State checkpoints (SchedulerConfig::frequent_state_checkpoints):
// a prefill chunk end on a page at least one chunk budget before the prompt
// end publishes every State group's slot 1 at the next chunk's admission, so
// a later prompt that diverges before the prompt's last budget resumes within
// a chunk of its divergence. Each one is unproven -- trimmed to
// max_unproven_state_checkpoints, oldest request first, shallowest first --
// until a hit claims it or an endpoint republishes it.
class FrequentStateCheckpointSuite : public StateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StateCheckpointSuite::MakeConfig();
        cfg.frequent_state_checkpoints = frequent_;
        cfg.max_unproven_state_checkpoints = frequent_ ? cap_ : 0;
        cfg.state_checkpoint_in_forward = in_forward_;
        cfg.prefix_replay_tokens = prefix_replay_tokens_;
        return cfg;
    }

    // Rebuilds the scheduler with the mode on or off and the given cap.
    void ResetFrequent(bool frequent, std::int32_t cap, bool host_cache = false) {
        frequent_ = frequent;
        cap_ = cap;
        Reset(host_cache, /*decode_width=*/1);
    }

    // Runs `id`'s prefill alone to its prompt end and finishes it there.
    std::vector<Chunk> RunAndFinish(const std::string& id, std::vector<std::int32_t> tokens) {
        const auto prompt_tokens = static_cast<std::int32_t>(tokens.size());
        Submit(RequestWithTokens(id, std::move(tokens)));
        std::vector<Chunk> chunks = RunPrefill(id, prompt_tokens);
        SendFinish(id);
        AckWriteBacks(PlanOnce());
        return chunks;
    }

    bool frequent_{true};
    std::int32_t cap_{8};
    bool in_forward_{false};
    std::int32_t prefix_replay_tokens_{0};
};

// A cold 34-token prompt keeps its 8-token chunks; 8, 16 and 24 are chunk
// ends at least a budget before its end, 32 is not. Prompts diverging after
// 13, 21, 29 and 33 tokens resume at the checkpoint before their divergence,
// from the live slot the chunk ending there wrote; without the mode they
// start over.
TEST_F(FrequentStateCheckpointSuite, EveryChunkEndBeforeThePromptsLastBudgetIsACheckpoint) {
    for (const bool frequent : {true, false}) {
        SCOPED_TRACE(frequent);
        ResetFrequent(frequent, /*cap=*/8);
        const std::vector<std::int32_t> tokens = MakeTokens(34);
        const std::vector<Chunk> chunks = RunAndFinish("seed", tokens);
        ASSERT_EQ(chunks.size(), 5u);
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            EXPECT_EQ(chunks[i].prefix_tokens, 8 * static_cast<std::int32_t>(i));
            EXPECT_EQ(chunks[i].tokens, i < 4 ? 8 : 2);
            EXPECT_EQ(chunks[i].checkpoint_len, 0);
            EXPECT_EQ(chunks[i].state_rows[0].size(), 2u) << "no third block";
        }
        const std::vector<std::pair<std::int32_t, std::int32_t>> shared_and_checkpoint = {
            {13, 8}, {21, 16}, {29, 24}, {33, 24}};
        for (const auto& [shared, checkpoint] : shared_and_checkpoint) {
            SCOPED_TRACE(shared);
            const Resume resume = Probe("diverges" + std::to_string(shared), Diverging(tokens, shared));
            EXPECT_EQ(resume.prefix_tokens, frequent ? checkpoint : 0);
            if (frequent) {
                EXPECT_EQ(resume.state_row.at(0), chunks[static_cast<std::size_t>(checkpoint / 8 - 1)].state_rows[0][1])
                    << "the live slot of the chunk ending there";
            }
        }
        EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 0);
    }
}

// The cap counts unproven checkpoints across requests. Under a cap of two the
// first prompt keeps 16 and 24. The second prompt's checkpoints then evict
// the older request's first -- 16, then 24 -- and then its own shallowest.
// The first prompt's prompt-end checkpoint is established and stays, and so
// does its KV.
TEST_F(FrequentStateCheckpointSuite, TheCapEvictsTheOldestRequestsShallowestCheckpointsFirst) {
    ResetFrequent(true, /*cap=*/2);
    const std::vector<std::int32_t> first = MakeTokens(34);
    RunAndFinish("first", first);
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 1) << "8 left when 24 was published";
    // A hit promotes the checkpoint it claims, so the misses run first.
    EXPECT_EQ(Probe("first13", Diverging(first, 13)).prefix_tokens, 0);

    const std::vector<std::int32_t> second = MakeTokens(34, 1000);
    RunAndFinish("second", second);
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 4);
    EXPECT_EQ(Probe("first21", Diverging(first, 21)).prefix_tokens, 0);
    EXPECT_EQ(Probe("first29", Diverging(first, 29)).prefix_tokens, 0);
    EXPECT_EQ(Probe("second13", Diverging(second, 13)).prefix_tokens, 0);
    EXPECT_EQ(Probe("first34", Diverging(first, 34)).prefix_tokens, 34) << "the first prompt's end";
    EXPECT_EQ(Probe("second21", Diverging(second, 21)).prefix_tokens, 16);
    EXPECT_EQ(Probe("second29", Diverging(second, 29)).prefix_tokens, 24);
}

// Pool pressure across requests: the planner evicts unproven checkpoints
// before any established entry, whatever its access epoch.
class FrequentStateCheckpointPressureSuite : public FrequentStateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FrequentStateCheckpointSuite::MakeConfig();
        cfg.device_allocator.total_pages = pool_pages_;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.total_pages = pool_pages_;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    std::int32_t pool_pages_{257};
};

// A second, unrelated prompt that needs room takes the first prompt's
// unproven checkpoints, then its own, shallowest first, before the first
// prompt's prompt-end checkpoint: that one stays down to the same pool size
// as without the mode (36 pages, where the second prompt evicts all six
// unproven checkpoints).
TEST_F(FrequentStateCheckpointPressureSuite, UnprovenCheckpointsLeaveBeforeAnOlderPromptsEstablishedCache) {
    const std::vector<std::int32_t> first = MakeTokens(34);
    const std::vector<std::int32_t> second = MakeTokens(34, 1000);
    // Runs both prompts on a fresh scheduler with `pages` Device pages and
    // reports where a prompt diverging from `prompt` after `shared` tokens
    // resumes.
    const auto resume_after_both = [&](bool frequent, std::int32_t pages, const std::vector<std::int32_t>& prompt,
                                       std::int32_t shared) {
        pool_pages_ = pages;
        ResetFrequent(frequent, /*cap=*/8);
        RunAndFinish("first", first);
        RunAndFinish("second", second);
        return Probe("probe", Diverging(prompt, shared)).prefix_tokens;
    };
    for (const std::int32_t pages : {44, 40, 36}) {
        SCOPED_TRACE(pages);
        EXPECT_EQ(resume_after_both(/*frequent=*/false, pages, first, 34), 34) << "without the mode";
        EXPECT_EQ(resume_after_both(/*frequent=*/true, pages, first, 34), 34) << "the first prompt's end stays";
        EXPECT_EQ(resume_after_both(/*frequent=*/true, pages, first, 29), 0) << "its unproven 24 left first";
    }
    EXPECT_EQ(resume_after_both(/*frequent=*/true, 44, second, 13), 0) << "the second prompt's 8 left next";
    EXPECT_EQ(resume_after_both(/*frequent=*/true, 44, second, 21), 16);
    EXPECT_EQ(resume_after_both(/*frequent=*/true, 44, second, 29), 24);
    EXPECT_EQ(resume_after_both(/*frequent=*/true, 36, second, 29), 0) << "and at 36 pages its deepest too";
}

// A checkpoint leaves the cap's count once proven: a hit claims it, or an
// endpoint republishes it (here the prompt end of a prompt ending on it).
// Under a cap of two, a hit on the first prompt's 24 keeps it through a later
// prompt's checkpoints, which evict its unproven 16. A prompt ending at the
// second prompt's 24 republishes that one as its prompt end, and a third
// prompt's checkpoints no longer evict it.
TEST_F(FrequentStateCheckpointSuite, AHitOrAnEndpointRepublicationPromotesACheckpointOutOfTheCap) {
    ResetFrequent(true, /*cap=*/2);
    const std::vector<std::int32_t> first = MakeTokens(34);
    RunAndFinish("first", first);
    EXPECT_EQ(Probe("hit24", Diverging(first, 29)).prefix_tokens, 24);

    const std::vector<std::int32_t> second = MakeTokens(34, 1000);
    RunAndFinish("second", second);
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 3) << "first's 8 and 16, second's 8";
    EXPECT_EQ(Probe("again24", Diverging(first, 29)).prefix_tokens, 24) << "the promoted checkpoint stays";
    EXPECT_EQ(Probe("gone16", Diverging(first, 21)).prefix_tokens, 0) << "its unproven sibling did not";

    // second[:24] resumes from second's 16 (its probe bound is 23) with the
    // junction at 20; its first decode publishes its prompt end at 24.
    const std::vector<std::int32_t> ends_at_24(second.begin(), second.begin() + 24);
    Submit(RequestWithTokens("ends_at_24", ends_at_24));
    const std::vector<Chunk> chunks = RunPrefill("ends_at_24", 24);
    ASSERT_FALSE(chunks.empty());
    EXPECT_EQ(chunks[0].prefix_tokens, 16);
    AckWriteBacks(PlanOnce());
    SendForwardDone("ends_at_24", {901});
    SendFinish("ends_at_24");
    AckWriteBacks(PlanOnce());

    RunAndFinish("third", MakeTokens(34, 2000));
    EXPECT_EQ(Probe("second24", Diverging(second, 29)).prefix_tokens, 24) << "republished as an endpoint";
}

// A chunk that ends at a junction publishes the junction's endpoint, also on
// a frequent position. The second prompt shares 18 tokens of a cached prompt
// whose checkpoints at 8 and 16 the cap of one has evicted, so it resumes at
// 0 with its junction at 16: its 8 and 24 are unproven, its 16 is not.
TEST_F(FrequentStateCheckpointSuite, AJunctionOnAFrequentPositionStaysAnEndpoint) {
    ResetFrequent(true, /*cap=*/1);
    const std::vector<std::int32_t> cached = MakeTokens(34);
    RunAndFinish("cached", cached);
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 2);

    std::vector<std::int32_t> tokens(cached.begin(), cached.begin() + 18);
    for (std::int32_t i = 0; i < 16; ++i) {
        tokens.push_back(700 + i);
    }
    const std::vector<Chunk> chunks = RunAndFinish("junction", tokens);
    ASSERT_EQ(chunks.size(), 5u);
    EXPECT_EQ(chunks[0].prefix_tokens, 0);
    EXPECT_EQ(chunks[2].prefix_tokens, 16) << "a chunk ended at the junction";
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 4) << "its 8 evicted cached's 24; its 24 its own 8";

    RunAndFinish("other", MakeTokens(34, 2000));
    EXPECT_EQ(Probe("diverges21", Diverging(tokens, 21)).prefix_tokens, 16) << "the junction's endpoint";
    EXPECT_EQ(Probe("diverges13", Diverging(tokens, 13)).prefix_tokens, 0);
}

// With the in-forward checkpoint (the recommended setup), chunk ends before
// the prompt's last budget are frequent checkpoints and the forward spanning
// the last page before the replay tail still writes that one inside it. The
// 30-token prompt's chunk ends 8 and 16 are frequent checkpoints, 24 is
// inside its last budget, and its final chunk [24, 30) writes 28. Only the
// frequent ones are unproven: an unrelated prompt never evicts 28.
TEST_F(FrequentStateCheckpointSuite, TheInForwardCheckpointCoversThePromptsLastBudget) {
    in_forward_ = true;
    ResetFrequent(true, /*cap=*/1);
    const std::vector<std::int32_t> tokens = MakeTokens(30);
    const std::vector<Chunk> chunks = RunAndFinish("seed", tokens);
    ASSERT_EQ(chunks.size(), 4u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(chunks[i].prefix_tokens, 8 * static_cast<std::int32_t>(i));
        EXPECT_EQ(chunks[i].checkpoint_len, 0);
    }
    EXPECT_EQ(chunks[3].prefix_tokens, 24);
    EXPECT_EQ(chunks[3].tokens, 6);
    EXPECT_EQ(chunks[3].checkpoint_len, 4) << "the in-forward checkpoint at 28";
    ASSERT_EQ(chunks[3].state_rows[0].size(), 3u);
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 1) << "8 left when 16 was published";
    EXPECT_EQ(Probe("diverges25", Diverging(tokens, 25)).prefix_tokens, 16) << "no checkpoint at 24";

    RunAndFinish("other", MakeTokens(34, 1000));
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 3) << "other's 8 and 16";
    const Resume resent = Probe("resent", tokens);
    EXPECT_EQ(resent.prefix_tokens, 28);
    EXPECT_EQ(resent.state_row.at(0), chunks[3].state_rows[0][2]) << "the in-forward checkpoint's block";
}

// A chunk end that is both a frequent position and the in-forward boundary
// (an 8-token replay tail puts the 34-token prompt's boundary at 24, a chunk
// end) is published as the boundary's endpoint, never as an unproven one.
TEST_F(FrequentStateCheckpointSuite, ABoundaryAtAChunkEndStaysAnEndpoint) {
    in_forward_ = true;
    prefix_replay_tokens_ = 8;
    ResetFrequent(true, /*cap=*/1);
    const std::vector<std::int32_t> tokens = MakeTokens(34);
    const std::vector<Chunk> chunks = RunAndFinish("seed", tokens);
    ASSERT_EQ(chunks.size(), 5u);
    for (const Chunk& chunk : chunks) {
        EXPECT_EQ(chunk.checkpoint_len, 0) << "the boundary is a chunk end";
    }
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 1) << "8 left when 16 was published";

    RunAndFinish("other", MakeTokens(42, 1000));
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 4) << "seed's 16, other's 8 and 16";
    // The replay tail bounds the probe 8 tokens before its end.
    std::vector<std::int32_t> probe = Diverging(tokens, 29);
    probe.push_back(992);
    EXPECT_EQ(Probe("diverges29", probe).prefix_tokens, 24);
}

// Mixed batches beside one decoding request, which takes one token of every
// round's 8 and leaves 7 for the prefill.
class FrequentStateCheckpointMixedSuite : public FrequentStateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FrequentStateCheckpointSuite::MakeConfig();
        cfg.enable_mixed_prefill_decode = true;
        return cfg;
    }
};

// A 7-token budget rarely ends on a page. A chunk passing a checkpoint's page
// ends on it, so the 30-token prompt runs [0, 4) ... [16, 20), each end a
// checkpoint; its last budget keeps the chunks [20, 27) and [27, 30) whole.
TEST_F(FrequentStateCheckpointMixedSuite, ChunksEndOnACheckpointsPageUnderABudgetCutShort) {
    ResetFrequent(true, /*cap=*/8);
    StartDecoding();
    const std::vector<std::int32_t> tokens = MakeTokens(30);
    Submit(RequestWithTokens("seed", tokens));
    const MixedPrefill prefill = RunPrefillBesideTheDecode("seed", 30);
    EXPECT_EQ(prefill.chunks, (std::vector<std::pair<std::int32_t, std::int32_t>>{
                                  {0, 4}, {4, 8}, {8, 12}, {12, 16}, {16, 20}, {20, 27}, {27, 30}}));
    SendFinish("seed");
    AckWriteBacks(PlanOnce());
    for (const std::int32_t checkpoint : {4, 8, 12, 16, 20}) {
        SCOPED_TRACE(checkpoint);
        EXPECT_EQ(Probe("diverges" + std::to_string(checkpoint), Diverging(tokens, checkpoint + 1)).prefix_tokens,
                  checkpoint);
    }
}

// A follow-up computing less than about two budgets past its hit is chunked
// and published as without the mode: its chunk ends lie in its last budget,
// so no chunk is rounded and no checkpoint is published inside it.
TEST_F(FrequentStateCheckpointMixedSuite, AShortFollowUpIsChunkedAsWithoutTheMode) {
    std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>> follow_up_chunks;
    for (const bool frequent : {false, true}) {
        SCOPED_TRACE(frequent);
        ResetFrequent(frequent, /*cap=*/8);
        StartDecoding();
        const std::vector<std::int32_t> seed = MakeTokens(30);
        Submit(RequestWithTokens("seed", seed));
        RunPrefillBesideTheDecode("seed", 30);
        SendFinish("seed");
        AckWriteBacks(PlanOnce());

        // The seed's prompt, whose end is a checkpoint, and 12 new tokens.
        std::vector<std::int32_t> follow_up = seed;
        for (std::int32_t i = 0; i < 12; ++i) {
            follow_up.push_back(600 + i);
        }
        Submit(RequestWithTokens("follow_up", follow_up));
        follow_up_chunks.push_back(RunPrefillBesideTheDecode("follow_up", 42).chunks);
        SendFinish("follow_up");
        AckWriteBacks(PlanOnce());
        EXPECT_EQ(Probe("diverges", Diverging(follow_up, 41)).prefix_tokens, 30)
            << "no checkpoint inside the follow-up";
    }
    ASSERT_EQ(follow_up_chunks.size(), 2u);
    EXPECT_EQ(follow_up_chunks[0], (std::vector<std::pair<std::int32_t, std::int32_t>>{{30, 37}, {37, 42}}));
    EXPECT_EQ(follow_up_chunks[1], follow_up_chunks[0]);
}

// The P role publishes its chunk ends' checkpoints too, and the remote decode
// hands off the prompt-end state in slot 1 as without the mode.
class FrequentStateCheckpointPrefillRoleSuite : public FrequentStateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FrequentStateCheckpointSuite::MakeConfig();
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

TEST_F(FrequentStateCheckpointPrefillRoleSuite, ChunkEndsPublishCheckpointsAndTheHandOffIsUnchanged) {
    ResetFrequent(true, /*cap=*/8);
    const std::vector<std::int32_t> tokens = MakeTokens(34);
    Submit(RequestWithTokens("seed", tokens));
    SendBootstrapped("seed");
    const std::vector<Chunk> chunks = RunPrefill("seed", 34);
    ASSERT_EQ(chunks.size(), 5u);
    const ExecutionPlan handoff = PlanOnce();
    ASSERT_TRUE(handoff.remote_decode.has_value());
    EXPECT_EQ(StateRow(*handoff.remote_decode, "state0", "seed"),
              (std::vector<std::int32_t>{0, chunks[4].state_rows[0][1]}))
        << "the prompt-end state in slot 1";
    AckWriteBacks(handoff);

    Submit(RequestWithTokens("follow_up", Diverging(tokens, 29)));
    SendBootstrapped("follow_up");
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"follow_up"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{24});
    EXPECT_EQ(StateRow(*batch, "state0", "follow_up").at(0), chunks[2].state_rows[0][1])
        << "resumes from the checkpoint before the seed's PD ACK";
    AckWriteBacks(plan);
    SendPdSucceeded("seed");
    AckWriteBacks(PlanOnce());
}

// Unproven checkpoints stay on the Device: a chunk end that publishes one
// streams its KV pages but not its state, and the publisher's finish stores
// its prompt-end and frontier states but none of them. A hit promotes the
// checkpoint it claims, and the hitter's finish stores that one.
TEST_F(FrequentStateCheckpointSuite, UnprovenCheckpointsStayOnTheDeviceUntilAHitPromotesThem) {
    ResetFrequent(true, /*cap=*/8, /*host_cache=*/true);
    const std::vector<std::int32_t> tokens = MakeTokens(34);
    // The chain hashes of the pages ending at 4, 8, ..., 32.
    const std::vector<std::string> chain = scheduler_->PrefixHashesForTokens(tokens);
    ASSERT_EQ(chain.size(), 8u);
    // The State groups' write-back rows by content hash (the full group is 0).
    const auto state_rows = [](const std::vector<std::pair<std::uint32_t, std::string>>& rows) {
        std::map<std::string, std::int32_t> by_hash;
        for (const auto& [group_id, hash] : rows) {
            if (group_id != 0) {
                ++by_hash[hash];
            }
        }
        return by_hash;
    };

    Submit(RequestWithTokens("seed", tokens));
    const std::vector<Chunk> chunks = RunPrefill("seed", 34);
    ASSERT_EQ(chunks.size(), 5u);
    for (std::size_t i = 1; i < chunks.size(); ++i) {
        EXPECT_FALSE(chunks[i].write_backs.empty()) << "chunk " << i << " streams the pages before it";
        EXPECT_TRUE(state_rows(chunks[i].write_backs).empty()) << "chunk " << i << " streams no checkpoint";
    }
    const ExecutionPlan first_decode = PlanOnce();
    const std::map<std::string, std::int32_t> prompt_end = state_rows(WriteBackRows(first_decode));
    ASSERT_EQ(prompt_end.size(), 1u);
    EXPECT_EQ(prompt_end.begin()->second, 3) << "the prompt end's state streams in every State group";
    AckWriteBacks(first_decode);
    SendForwardDone("seed", {901});
    SendFinish("seed");
    const ExecutionPlan finish = PlanOnce();
    const std::map<std::string, std::int32_t> finish_rows = state_rows(WriteBackRows(finish));
    ASSERT_EQ(finish_rows.size(), 1u) << "the frontier's state alone";
    for (const std::size_t page : {1, 3, 5}) {
        EXPECT_FALSE(finish_rows.contains(chain[page])) << "page " << page;
    }
    AckWriteBacks(finish);

    // A hit on 24 promotes it; the hitter's finish stores it, and still none
    // of the unproven 8 and 16.
    std::vector<std::int32_t> hitter(tokens.begin(), tokens.begin() + 29);
    for (std::int32_t i = 0; i < 6; ++i) {
        hitter.push_back(700 + i);
    }
    Submit(RequestWithTokens("hitter", hitter));
    const std::vector<Chunk> hit_chunks = RunPrefill("hitter", 35);
    ASSERT_FALSE(hit_chunks.empty());
    EXPECT_EQ(hit_chunks[0].prefix_tokens, 24);
    AckWriteBacks(PlanOnce());
    SendForwardDone("hitter", {902});
    SendFinish("hitter");
    const std::map<std::string, std::int32_t> hitter_rows = state_rows(WriteBackRows(PlanOnce()));
    EXPECT_EQ(hitter_rows.contains(chain[5]) ? hitter_rows.at(chain[5]) : 0, 3) << "the promoted checkpoint at 24";
    EXPECT_FALSE(hitter_rows.contains(chain[1]));
    EXPECT_FALSE(hitter_rows.contains(chain[3]));
}

// With KV events a chunk end becomes BlockStored once its checkpoint makes
// every group hold its page, and BlockRemoved when a trim takes the
// checkpoint again; the page's KV stays.
class FrequentStateCheckpointKvEventSuite : public FrequentStateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FrequentStateCheckpointSuite::MakeConfig();
        cfg.enable_kv_cache_events = true;
        return cfg;
    }

    struct Drained {
        std::vector<std::uint64_t> stored;
        std::vector<std::uint64_t> removed;
    };
    Drained Drain() {
        Drained out;
        for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
            if (const auto* stored = std::get_if<KvBlockStoredEvent>(&event)) {
                out.stored.push_back(stored->block_hashes.front());
            } else {
                out.removed.push_back(std::get<KvBlockRemovedEvent>(event).block_hashes.front());
            }
        }
        return out;
    }
};

TEST_F(FrequentStateCheckpointKvEventSuite, ACheckpointedChunkEndIsStoredAndATrimRemovesIt) {
    ResetFrequent(true, /*cap=*/1);
    const std::vector<std::int32_t> tokens = MakeTokens(34);
    const std::span<const std::int32_t> all(tokens);
    // The KV event hash of each page [4i, 4i + 4).
    std::vector<std::uint64_t> page_hash;
    for (std::size_t page = 0; page < 4; ++page) {
        page_hash.push_back(HashKvBlock(all.subspan(page * 4, 4),
                                        page == 0 ? std::nullopt : std::optional<std::uint64_t>{page_hash.back()}));
    }
    Submit(RequestWithTokens("seed", tokens));
    // [0, 8): nothing is published yet.
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    SendForwardDone("seed");
    EXPECT_TRUE(Drain().stored.empty());
    // [8, 16): its admission publishes the pages through 8 and the checkpoint
    // at 8, so every group holds the page [4, 8).
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    SendForwardDone("seed");
    const Drained at_8 = Drain();
    EXPECT_EQ(at_8.stored, std::vector<std::uint64_t>{page_hash[1]});
    EXPECT_TRUE(at_8.removed.empty());
    // [16, 24): the checkpoint at 16 is published and the cap of one evicts 8.
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    SendForwardDone("seed");
    const Drained at_16 = Drain();
    EXPECT_EQ(at_16.stored, std::vector<std::uint64_t>{page_hash[3]});
    EXPECT_EQ(at_16.removed, std::vector<std::uint64_t>{page_hash[1]});
}

// A request retracted on a page-aligned frontier and restored with no Host
// cache resumes on private copies of its pages. Here the pool is so tight
// that the restore evicts their cache entries, and a drain between the
// restore's ACK and the next chunk drops their descriptors. The next chunk's
// admission publishes the state at the frontier again without hashing a new
// page -- a frequent checkpoint (mode on) or the junction the chunk before it
// was cut at (mode off). That publication must not trip over the dropped
// descriptor, and with the KV through the frontier gone nothing is reported
// stored.
class StateCheckpointRestoreKvEventSuite : public FrequentStateCheckpointKvEventSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FrequentStateCheckpointKvEventSuite::MakeConfig();
        cfg.debug_force_retraction_interval = -4;
        // One full and one State group over twelve usable blocks.
        std::erase_if(cfg.cache_groups, [](const CacheGroupConfig& group) {
            return group.group_id == "state1" || group.group_id == "state2";
        });
        cfg.device_allocator.total_pages = 13;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Plans one round in which `id` runs a prefill chunk alone, lands it and
    // returns where it started and how many tokens it computed.
    std::pair<std::int32_t, std::int32_t> RunChunk(const std::string& id, bool completes = false) {
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_TRUE_OR_RETURN(batch != nullptr && batch->request_ids == std::vector<std::string>{id},
                              (std::pair<std::int32_t, std::int32_t>{-1, -1}));
        SendForwardDone(id, completes ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
        return {batch->extend_prefix_lens.at(0), batch->input_lengths.at(0)};
    }

    // The forced retraction of the plan after `id`'s last chunk landed, and
    // the restore of the plan after it, with a drain after each ACK.
    void RetractAndRestore() {
        const ExecutionPlan retraction = PlanOnce();
        ASSERT_EQ(scheduler_->RetractedSize(), 1u);
        AckImageStores(retraction);
        Drain();
        const ExecutionPlan readmit = PlanOnce();
        ASSERT_NE(FindRestore(readmit), nullptr);
        AckRestores(readmit);
        Drain();
    }
};

// The 34-token prompt is retracted at 24, a frequent position whose pages the
// retraction hashed; the chunk [24, 32) resumes there.
TEST_F(StateCheckpointRestoreKvEventSuite, AFrequentCheckpointAfterARestoreFindsItsDescriptor) {
    ResetFrequent(true, /*cap=*/8);
    const std::vector<std::int32_t> tokens = MakeTokens(34);
    Submit(RequestSpec{.request_id = "seed", .tokens = tokens, .max_new_tokens = 2});
    // Plans 1-3 run [0, 8), [8, 16) and [16, 24); plan 4 retracts, plan 5
    // restores.
    for (std::int32_t chunk = 0; chunk < 3; ++chunk) {
        EXPECT_EQ(RunChunk("seed").first, 8 * chunk);
    }
    Drain();
    RetractAndRestore();
    EXPECT_EQ(RunChunk("seed"), (std::pair<std::int32_t, std::int32_t>{24, 8}));
    EXPECT_TRUE(Drain().stored.empty()) << "nothing new is complete: the KV through 24 is gone";
    EXPECT_EQ(RunChunk("seed", /*completes=*/true), (std::pair<std::int32_t, std::int32_t>{32, 2}));
}

// Mode off: the second prompt shares the first one's first 8 tokens and no
// checkpoint, so its first chunk is cut at the junction at 8; it is retracted
// there, and the chunk [8, 16) resumes at the junction.
TEST_F(StateCheckpointRestoreKvEventSuite, AJunctionAfterARestoreFindsItsDescriptor) {
    ResetFrequent(false, /*cap=*/0);
    const std::vector<std::int32_t> seed = MakeTokens(12);
    Submit(RequestSpec{.request_id = "seed", .tokens = seed, .max_new_tokens = 1});
    RunChunk("seed");
    RunChunk("seed", /*completes=*/true);
    SendFinish("seed");
    // Plans 3-6 run nothing, so the second prompt's first chunk is plan 7 and
    // the forced retraction of plan 8 takes it.
    for (std::int32_t plan = 3; plan <= 6; ++plan) {
        PlanOnce();
    }
    Drain();
    std::vector<std::int32_t> tokens(seed.begin(), seed.begin() + 8);
    for (std::int32_t i = 0; i < 26; ++i) {
        tokens.push_back(500 + i);
    }
    Submit(RequestSpec{.request_id = "r", .tokens = tokens, .max_new_tokens = 1});
    EXPECT_EQ(RunChunk("r"), (std::pair<std::int32_t, std::int32_t>{0, 8})) << "cut at the junction";
    RetractAndRestore();
    EXPECT_EQ(RunChunk("r"), (std::pair<std::int32_t, std::int32_t>{8, 8}));
    EXPECT_TRUE(Drain().stored.empty()) << "the KV through 8 is gone";
}

// A prefill retracted between chunks publishes its page-aligned frontier as
// an endpoint, as every retraction does. The restored request's next chunk
// republishes its live slot there, which leaves the endpoint as it was, and
// the request runs to its finish.
class FrequentStateCheckpointRetractionSuite : public FrequentStateCheckpointSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FrequentStateCheckpointSuite::MakeConfig();
        cfg.debug_force_retraction_interval = -4;
        return cfg;
    }
};

TEST_F(FrequentStateCheckpointRetractionSuite, ARetractedPrefillPublishesItsFrontierAsAnEndpoint) {
    ResetFrequent(true, /*cap=*/1);
    const std::vector<std::int32_t> tokens = MakeTokens(34);
    Submit(RequestWithTokens("seed", tokens));
    // Plans 1-3 run [0, 8), [8, 16) and [16, 24); plan 4 retracts the request.
    for (std::int32_t chunk = 0; chunk < 3; ++chunk) {
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_TRUE(batch != nullptr && batch->request_ids == std::vector<std::string>{"seed"});
        EXPECT_EQ(batch->extend_prefix_lens.at(0), 8 * chunk);
        SendForwardDone("seed");
    }
    const ExecutionPlan retraction = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    AckImageStores(retraction);
    const ExecutionPlan readmit = PlanOnce();
    ASSERT_NE(FindRestore(readmit), nullptr);
    AckRestores(readmit);
    const std::vector<Chunk> rest = RunPrefill("seed", 34);
    ASSERT_EQ(rest.size(), 2u);
    EXPECT_EQ(rest[0].prefix_tokens, 24);
    SendFinish("seed");
    AckWriteBacks(PlanOnce());
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 1) << "8 left when 16 was published";

    EXPECT_EQ(Probe("diverges29", Diverging(tokens, 29)).prefix_tokens, 24) << "the retraction's endpoint";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), config_.device_allocator.NumUsableBlocks())
        << "every block is cache-only or free";
}

// With a Host cache the image's L2 leg takes the published slots, but an
// unproven checkpoint stays off Host there too: slot 0 -- the checkpoint at
// 16 the chunk [16, 24) read -- rides the snapshot pool like an unpublished
// slot, and only the endpoint the retraction publishes at 24 rides L2. The
// restore copies slot 0 back from the pool instead of claiming (and so
// promoting) the Device entry, which stays unproven: a later prompt's
// checkpoints evict it under the cap.
TEST_F(FrequentStateCheckpointRetractionSuite, AnUnprovenSlotZeroRidesTheSnapshotPoolNotHost) {
    ResetFrequent(true, /*cap=*/2, /*host_cache=*/true);
    const std::vector<std::int32_t> tokens = MakeTokens(34);
    // The chain hashes of the pages ending at 4, 8, ..., 32.
    const std::vector<std::string> chain = scheduler_->PrefixHashesForTokens(tokens);
    ASSERT_EQ(chain.size(), 8u);
    Submit(RequestWithTokens("seed", tokens));
    for (std::int32_t chunk = 0; chunk < 3; ++chunk) {
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_TRUE(batch != nullptr && batch->request_ids == std::vector<std::string>{"seed"});
        AckWriteBacks(plan);
        SendForwardDone("seed");
    }
    const std::int32_t host_before = scheduler_->HostPoolCachedBlocks();
    const ExecutionPlan retraction = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    std::map<std::string, std::int32_t> l2_state_rows;
    for (const auto& [group_id, hash] : WriteBackRows(retraction)) {
        if (group_id != 0) {
            ++l2_state_rows[hash];
        }
    }
    EXPECT_EQ(l2_state_rows, (std::map<std::string, std::int32_t>{{chain[5], 3}}))
        << "the endpoint at 24 in every State group, and not the unproven 16";
    const SnapshotStoreBatch* store = FindSnapshotStore(retraction);
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(std::ranges::count_if(store->group_ids.at(0), [](std::uint32_t group_id) { return group_id != 0; }), 3)
        << "slot 0 of every State group rides the pool";
    AckImageStores(retraction);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), host_before + 2 + 3) << "the pages [16, 24) and the endpoint";

    const ExecutionPlan readmit = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr);
    std::map<std::uint8_t, std::int32_t> state_rows_by_tier;
    for (std::size_t i = 0; i < restore->group_ids.at(0).size(); ++i) {
        if (restore->group_ids.at(0)[i] != 0) {
            ++state_rows_by_tier[restore->source_tiers.at(0)[i]];
        }
    }
    EXPECT_EQ(state_rows_by_tier,
              (std::map<std::uint8_t, std::int32_t>{{static_cast<std::uint8_t>(HostTier::kL2), 3},
                                                    {static_cast<std::uint8_t>(HostTier::kSnapshotPool), 3}}))
        << "slot 1 from L2 and slot 0 from the pool, each a private copy";
    AckRestores(readmit);
    const std::vector<Chunk> rest = RunPrefill("seed", 34);
    ASSERT_EQ(rest.size(), 2u);
    EXPECT_EQ(rest[0].prefix_tokens, 24);
    SendFinish("seed");
    AckWriteBacks(PlanOnce());
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 0) << "8 and 16, within the cap";

    // Another prompt's 8 and 16 evict the seed's 8 and 16, the oldest. (Its
    // 24 tokens prefill in plans 9-11, between two forced retractions.)
    RunAndFinish("other", MakeTokens(24, 1000));
    EXPECT_EQ(scheduler_->UnprovenStateCheckpointTrims(), 2);
    EXPECT_EQ(Probe("diverges21", Diverging(tokens, 21)).prefix_tokens, 0) << "16 was evicted, not promoted";
    EXPECT_EQ(Probe("diverges29", Diverging(tokens, 29)).prefix_tokens, 24) << "the retraction's endpoint";
}

}  // namespace tokenspeed::test
