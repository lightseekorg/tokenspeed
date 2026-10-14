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

#include "integration_test_helper.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace tokenspeed::test {

namespace {

struct ScheduledPrefill {
    std::int32_t extend_prefix_len{0};
    std::int32_t input_length{0};
};

// The prefill rows of the plan's model batch, by request id.
std::map<std::string, ScheduledPrefill> ScheduledPrefills(const ExecutionPlan& plan) {
    std::map<std::string, ScheduledPrefill> prefills;
    const ForwardBatch* batch = FindForwardBatch(plan);
    if (batch == nullptr) {
        return prefills;
    }
    for (std::size_t i = 0; i < batch->NumExtends(); ++i) {
        prefills[batch->request_ids[i]] = {batch->extend_prefix_lens[i], batch->input_lengths[i]};
    }
    return prefills;
}

std::vector<std::int32_t> Concat(std::vector<std::int32_t> head, const std::vector<std::int32_t>& tail) {
    head.insert(head.end(), tail.begin(), tail.end());
    return head;
}

RequestSpec Prompt(const std::string& id, std::vector<std::int32_t> tokens) {
    return RequestSpec{.request_id = id, .tokens = std::move(tokens)};
}

// Three pages (prefix granularity 2) shared by the prompts below; each prompt
// completes a fourth page of its own and ends one token into a fifth.
std::vector<std::int32_t> SharedPrefix() {
    return MakeTokens(6, 1);
}
std::vector<std::int32_t> PromptA() {
    return Concat(SharedPrefix(), {100, 101, 102});
}
std::vector<std::int32_t> PromptB() {
    return Concat(SharedPrefix(), {200, 201, 202});
}

}  // namespace

using InflightPrefixSharingTest = SchedulerTestSuite;

TEST_F(InflightPrefixSharingTest, SameRoundPromptWaitsForTheSharedPrefixAndHitsIt) {
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB())});

    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 1u) << "b would compute the pages a computes in the same forward";
    EXPECT_EQ(first.at("a").extend_prefix_len, 0);
    EXPECT_EQ(first.at("a").input_length, 9);

    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.at("b").extend_prefix_len, 6);
    EXPECT_EQ(second.at("b").input_length, 3);
}

TEST_F(InflightPrefixSharingTest, CompletedPrefillPublishesBeforeItsFirstDecode) {
    Submit(Prompt("a", PromptA()));
    ASSERT_EQ(ScheduledPrefills(PlanOnce()).size(), 1u);

    // Outside mixed mode a's first decode cannot run beside b's prefill, yet
    // b hits the prompt pages a computed.
    Submit(Prompt("b", PromptB()));
    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.at("b").extend_prefix_len, 6);
}

TEST_F(InflightPrefixSharingTest, PromptsDivergingAfterTheCachedPrefixShareARound) {
    Submit(Prompt("seed", Concat(SharedPrefix(), {7})));
    PlanOnce();
    SendForwardDone("seed", {8});
    PlanOnce();
    SendFinish("seed");
    PlanOnce();

    Submit({Prompt("a", PromptA()), Prompt("b", PromptB())});
    const auto round = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(round.size(), 2u) << "each computes only its own fourth page";
    EXPECT_EQ(round.at("a").extend_prefix_len, 6);
    EXPECT_EQ(round.at("b").extend_prefix_len, 6);
}

TEST_F(InflightPrefixSharingTest, IdenticalPromptsComputeThePromptOnce) {
    Submit({Prompt("a", MakeTokens(9, 1)), Prompt("b", MakeTokens(9, 1))});
    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 1u);
    EXPECT_TRUE(first.contains("a"));

    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.at("b").extend_prefix_len, 8) << "only the final token is recomputed";
    EXPECT_EQ(second.at("b").input_length, 1);
}

TEST_F(InflightPrefixSharingTest, PromptAlreadyHittingThePageAWriterRecomputesDoesNotWait) {
    // A page-aligned prompt recomputes its cached last page -- a probe never
    // claims the page holding the final token. A prompt extending it already
    // hits that page, so it has nothing to wait for.
    const std::vector<std::int32_t> aligned = MakeTokens(8, 1);
    Submit(Prompt("seed", aligned));
    PlanOnce();
    SendForwardDone("seed", {9});
    PlanOnce();
    SendFinish("seed");
    PlanOnce();

    Submit({Prompt("a", aligned), Prompt("b", Concat(aligned, {50, 51, 52}))});
    const auto round = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(round.size(), 2u);
    EXPECT_EQ(round.at("a").extend_prefix_len, 6);
    EXPECT_EQ(round.at("a").input_length, 2);
    EXPECT_EQ(round.at("b").extend_prefix_len, 8);
}

TEST_F(InflightPrefixSharingTest, LaterPromptsOfASharedPrefixHitTogether) {
    // C-Eval-shaped: many prompts share a 5-shot prefix. The first computes
    // it; the rest wait this round and hit together on the next.
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB()), Prompt("c", Concat(SharedPrefix(), {300, 301, 302}))});
    ASSERT_EQ(ScheduledPrefills(PlanOnce()).size(), 1u);

    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 2u);
    EXPECT_EQ(second.at("b").extend_prefix_len, 6);
    EXPECT_EQ(second.at("c").extend_prefix_len, 6);
}

TEST_F(InflightPrefixSharingTest, WaitingPromptDoesNotHoldBackUnrelatedPrompts) {
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB()), Prompt("c", MakeTokens(9, 500))});
    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 2u);
    EXPECT_TRUE(first.contains("a"));
    EXPECT_TRUE(first.contains("c"));
}

class InflightPrefixSharingWithoutPrefixCacheTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.disable_prefix_cache = true;
        return cfg;
    }
};

TEST_F(InflightPrefixSharingWithoutPrefixCacheTest, PromptsSharingAPrefixShareARound) {
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB())});
    EXPECT_EQ(ScheduledPrefills(PlanOnce()).size(), 2u) << "without prefix reuse there is nothing to wait for";
}

class InflightPrefixSharingStateCacheTest : public KimiFourGroupSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = KimiFourGroupSuite::MakeConfig();
        cfg.device_allocator.total_pages = 129;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }
};

class InflightPrefixSharingMixedTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_mixed_prefill_decode = true;
        return cfg;
    }
};

TEST_F(InflightPrefixSharingMixedTest, SameRoundPromptWaitsForTheSharedPrefixAndHitsIt) {
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB())});

    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 1u);
    EXPECT_EQ(first.at("a").extend_prefix_len, 0);

    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.at("b").extend_prefix_len, 6);
}

TEST_F(InflightPrefixSharingMixedTest, PrefillHitJoinsTheWritersFirstDecode) {
    Submit(Prompt("a", PromptA()));
    ASSERT_EQ(ScheduledPrefills(PlanOnce()).size(), 1u);

    Submit(Prompt("b", PromptB()));
    const ExecutionPlan plan = PlanOnce();
    const auto prefills = ScheduledPrefills(plan);
    ASSERT_EQ(prefills.size(), 1u);
    EXPECT_EQ(prefills.at("b").extend_prefix_len, 6);
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->decode_input_ids.size(), 1u) << "a's first decode rides the same mixed plan";
}

class InflightPrefixSharingChunkedTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        // Four tokens leave a 9-token prompt incomplete; the writer holds the
        // line, so the shared-prefix waiter cannot sneak into this round.
        cfg.max_scheduled_tokens = 4;
        return cfg;
    }
};

TEST_F(InflightPrefixSharingChunkedTest, IncompleteChunkHoldsTheLineThenTheSharedPrefixHits) {
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB())});

    const ExecutionPlan first = PlanOnce();
    const auto first_prefills = ScheduledPrefills(first);
    ASSERT_EQ(first_prefills.size(), 1u);
    EXPECT_TRUE(first_prefills.contains("a"));
    EXPECT_LT(first_prefills.at("a").input_length, 9);

    // Incomplete chunks hold the line. The finishing chunk becomes PrefillDone
    // in the same applyEvent, so HOL lifts and b joins that plan (1.1) and
    // hits the pages a's Admit already published. There is no b-only prefill.
    SendForwardDone("a");
    std::map<std::string, ScheduledPrefill> later;
    bool b_joined_finishing_chunk = false;
    for (int step = 0; step < 8; ++step) {
        later = ScheduledPrefills(PlanOnce());
        if (later.contains("b")) {
            ASSERT_TRUE(later.contains("a"));
            ASSERT_EQ(later.size(), 2u);
            EXPECT_EQ(later.at("b").extend_prefix_len, 6);
            b_joined_finishing_chunk = true;
            break;
        }
        ASSERT_EQ(later.size(), 1u);
        ASSERT_TRUE(later.contains("a"));
        SendForwardDone("a");
    }
    EXPECT_TRUE(b_joined_finishing_chunk);
}

TEST_F(InflightPrefixSharingStateCacheTest, OnlyAPromptSharingTheWritersCheckpointWaits) {
    // A state group resumes only where a checkpoint was published: a's
    // completed prompt ends at its aligned natural checkpoint, 8 tokens in.
    // b diverges before it and cannot resume there, so it does not wait;
    // c extends that exact checkpoint and does.
    std::vector<std::int32_t> a = PromptA();
    a.resize(8);
    const std::vector<std::int32_t> c = Concat({a.begin(), a.begin() + 8}, {300, 301, 302});
    Submit({Prompt("a", a), Prompt("b", PromptB()), Prompt("c", c)});

    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 2u);
    EXPECT_TRUE(first.contains("a"));
    EXPECT_TRUE(first.contains("b"));

    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.at("c").extend_prefix_len, 8);
    EXPECT_EQ(second.at("c").input_length, 3);
}

TEST_F(InflightPrefixSharingStateCacheTest, UnalignedStateDoesNotCreateAnInteriorCheckpoint) {
    const std::vector<std::int32_t> a = PromptA();
    const std::vector<std::int32_t> shorter = Concat({a.begin(), a.begin() + 8}, {300, 301, 302});
    Submit({Prompt("writer", a), Prompt("reader", shorter)});
    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 2u) << "no state checkpoint exists at token 8 of a 9-token forward";
    EXPECT_EQ(first.at("reader").extend_prefix_len, 0);
}

class InflightStateSpeculationTest : public InflightPrefixSharingStateCacheTest {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = InflightPrefixSharingStateCacheTest::MakeConfig();
        cfg.decode_input_tokens = 5;
        return cfg;
    }
};

TEST_F(InflightStateSpeculationTest, PublishesThePromptFrontierNotTheSpeculativeWidth) {
    const auto prefix = MakeTokens(8, 1);
    Submit({Prompt("writer", prefix), Prompt("reader", Concat(prefix, {300, 301, 302}))});
    ASSERT_EQ(ScheduledPrefills(PlanOnce()).size(), 1u);
    SendForwardDone("writer", {41});
    const auto next = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(next.size(), 1u);
    EXPECT_EQ(next.at("reader").extend_prefix_len, 8);
    EXPECT_EQ(next.at("reader").input_length, 3);
}

TEST_F(InflightPrefixSharingTest, WritableTailIsNotSharedBeforeDecodeAdmission) {
    const auto prompt = PromptA();
    Submit(Prompt("writer", prompt));
    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* writer = FindForwardBatch(first);
    ASSERT_NE(writer, nullptr);
    const auto writer_tail = writer->block_tables.at("full_attention").at(0).at(4);
    SendForwardDone("writer", {400});
    Submit(Prompt("reader", Concat(prompt, {300, 301, 302})));
    const ExecutionPlan next = PlanOnce();
    const auto prefills = ScheduledPrefills(next);
    ASSERT_EQ(prefills.size(), 1u);
    EXPECT_EQ(prefills.at("reader").extend_prefix_len, 8);
    const ForwardBatch* reader = FindForwardBatch(next);
    ASSERT_NE(reader, nullptr);
    EXPECT_NE(reader->block_tables.at("full_attention").at(0).at(4), writer_tail);
    SendForwardDone("reader", {401});
    SendFinish("reader");
    const ExecutionPlan decode = PlanOnce();
    const auto& copies = decode.pages_to_copy.at("full_attention");
    EXPECT_TRUE(std::any_of(copies.begin(), copies.end(), [writer_tail](const auto& copy) {
        return copy.first == writer_tail && copy.second != writer_tail;
    })) << "the owner's tail moves only through the first decode admission";
}

class InflightPrefixLifetimeTest : public SchedulerTestSuite, public ::testing::WithParamInterface<bool> {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.disable_l2_cache = true;
        cfg.host_allocator.total_pages = 0;
        return cfg;
    }
};

TEST_P(InflightPrefixLifetimeTest, ReaderPinsSharedPagesAfterWriterExits) {
    const std::int32_t available = scheduler_->AvailableLcmBlocks();
    Submit({Prompt("writer", PromptA()), Prompt("reader", PromptB())});
    const ExecutionPlan writer_plan = PlanOnce();
    const ForwardBatch* writer = FindForwardBatch(writer_plan);
    ASSERT_NE(writer, nullptr);
    ASSERT_EQ(writer->request_ids, std::vector<std::string>{"writer"});

    // Publish while the writer's forward is still in flight. The reader
    // must own its hit, not merely borrow the writer's request lifetime.
    const ExecutionPlan reader_plan = PlanOnce();
    const ForwardBatch* reader = FindForwardBatch(reader_plan);
    ASSERT_NE(reader, nullptr);
    ASSERT_EQ(reader->request_ids, std::vector<std::string>{"reader"});
    ASSERT_EQ(reader->extend_prefix_lens.at(0), 6);
    const auto& writer_pages = writer->block_tables.at("full_attention").at(0);
    const auto& reader_pages = reader->block_tables.at("full_attention").at(0);
    ASSERT_GE(writer_pages.size(), 3u);
    ASSERT_GE(reader_pages.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_GT(reader_pages[i], 0);
        EXPECT_EQ(reader_pages[i], writer_pages[i]);
    }
    const std::set<std::int32_t> pinned(reader_pages.begin(), reader_pages.end());
    if (GetParam()) {
        SendAbortEvent("writer");
    } else {
        SendFinish("writer");
    }

    // New unrelated writes cannot reuse any physical page the reader holds.
    Submit(Prompt("other", MakeTokens(9, 500)));
    const ExecutionPlan other_plan = PlanOnce();
    const ForwardBatch* other = FindForwardBatch(other_plan);
    ASSERT_NE(other, nullptr);
    ASSERT_EQ(other->request_ids, std::vector<std::string>{"other"});
    for (std::int32_t page : other->block_tables.at("full_attention").at(0)) {
        if (page > 0) {
            EXPECT_FALSE(pinned.contains(page));
        }
    }
    SendForwardDone("writer", {400});
    SendForwardDone("reader", {401});
    SendForwardDone("other", {402});
    SendFinish("reader");
    SendFinish("other");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), available);

    // Reuse most of the pool after the last reader exits: shared references
    // must be released exactly once, and old unpinned entries may be evicted.
    Submit(Prompt("churn", MakeTokens(55, 1000)));
    const ExecutionPlan churn_plan = PlanOnce();
    const ForwardBatch* churn = FindForwardBatch(churn_plan);
    ASSERT_NE(churn, nullptr);
    ASSERT_EQ(churn->request_ids, std::vector<std::string>{"churn"});
    ASSERT_EQ(churn->input_lengths.at(0), 55);
    std::set<std::int32_t> allocated;
    for (std::int32_t page : churn->block_tables.at("full_attention").at(0)) {
        if (page > 0) {
            EXPECT_TRUE(allocated.insert(page).second);
        }
    }
    SendForwardDone("churn", {403});
    SendFinish("churn");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), available);
}

INSTANTIATE_TEST_SUITE_P(FinishAndAbort, InflightPrefixLifetimeTest, ::testing::Bool());

TEST_F(InflightPrefixSharingTest, ExplicitReplayBoundCapsTheSharedHit) {
    RequestSpec reader = Prompt("reader", PromptB());
    reader.max_cached_prefix_tokens = 2;
    Submit({Prompt("writer", PromptA()), reader});
    ASSERT_EQ(ScheduledPrefills(PlanOnce()).size(), 1u);
    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.at("reader").extend_prefix_len, 2);
    EXPECT_EQ(second.at("reader").input_length, 7);
}

TEST_F(InflightPrefixSharingTest, ARequestRecomputingItsWholePromptDoesNotWait) {
    RequestSpec reader = Prompt("reader", PromptB());
    reader.max_cached_prefix_tokens = 0;
    Submit({Prompt("writer", PromptA()), reader});
    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 2u);
    EXPECT_EQ(first.at("reader").extend_prefix_len, 0);
    EXPECT_EQ(first.at("reader").input_length, 9);
}

}  // namespace tokenspeed::test
