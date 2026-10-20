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

// Token-level prefix hits end to end through the Scheduler: requests publish
// their partial pages at prompt end and finish, later prompts resume at the
// longest published boundary that is a prefix of theirs, and the admission
// carries the copy of the block containing that boundary.

#include <gtest/gtest.h>

#include <map>
#include <set>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

// One published request as the brute-force oracle sees it: its token stream
// and the boundaries it published -- every full page below its computed
// frontier, its prompt end, and its frontier at finish.
struct PublishedStream {
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> boundaries;
};

std::int32_t CommonPrefix(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b) {
    std::int32_t n = 0;
    while (static_cast<std::size_t>(n) < a.size() && static_cast<std::size_t>(n) < b.size() && a[n] == b[n]) {
        ++n;
    }
    return n;
}

// The longest published boundary B <= max_tokens that is a prefix of `prompt`.
std::int32_t OracleHit(const std::vector<PublishedStream>& published, const std::vector<std::int32_t>& prompt,
                       std::int32_t max_tokens) {
    std::int32_t best = 0;
    for (const PublishedStream& stream : published) {
        const std::int32_t common = CommonPrefix(stream.tokens, prompt);
        for (const std::int32_t boundary : stream.boundaries) {
            if (boundary <= common && boundary <= max_tokens) {
                best = std::max(best, boundary);
            }
        }
    }
    return best;
}

#define ASSERT_NE_RETURN(a, b, ret) \
    do {                            \
        EXPECT_NE(a, b);            \
        if ((a) == (b)) return ret; \
    } while (false)
#define ASSERT_EQ_RETURN(a, b, ret) \
    do {                            \
        EXPECT_EQ(a, b);            \
        if ((a) != (b)) return ret; \
    } while (false)

class TokenLevelHitSuite : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kP = 4;

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = kP;
        cfg.device_allocator.total_pages = 257;  // ample: nothing is ever evicted
        cfg.host_allocator.total_pages = 0;
        cfg.disable_l2_cache = true;
        cfg.max_scheduled_tokens = 256;
        cfg.max_batch_size = 16;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full",
            .block_granularity = 2,  // two blocks per prefix page
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kFull,
        });
        if (WithSlidingGroup()) {
            cfg.cache_groups.push_back(CacheGroupConfig{
                .group_id = "swa",
                .block_granularity = kP,
                .total_pages = cfg.device_allocator.total_pages,
                .attn_kind = AttnKind::kSlidingWindow,
                .sliding_window_tokens = 5,

            });
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }
    virtual bool WithSlidingGroup() const { return false; }

    // Runs one request to completion: prefill, `generated` decode steps, finish.
    // Returns the boundary its first chunk resumed at.
    std::int32_t Run(const std::string& id, const std::vector<std::int32_t>& prompt, std::int32_t generated,
                     std::int32_t next_token) {
        Submit(RequestSpec{.request_id = id, .tokens = prompt, .max_new_tokens = generated + 1});
        const ExecutionPlan prefill = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(prefill);
        ASSERT_NE_RETURN(batch, nullptr, -1);
        ASSERT_EQ_RETURN(batch->request_ids, std::vector<std::string>{id}, -1);
        const std::int32_t hit = batch->extend_prefix_lens.at(0);
        for (std::int32_t step = 0; step < generated; ++step) {
            SendForwardDone(id, {next_token + step});
            PlanOnce();
        }
        SendForwardDone(id, {next_token + generated});
        SendFinish(id);
        PlanOnce();
        return hit;
    }
};

TEST_F(TokenLevelHitSuite, RandomStreamsResumeAtTheLongestPublishedBoundary) {
    std::mt19937 rng{20261012};
    for (std::int32_t trial = 0; trial < 40; ++trial) {
        SCOPED_TRACE(trial);
        SetUp();  // a fresh scheduler per trial
        std::vector<PublishedStream> published;
        std::int32_t next_token = 1000 * (trial + 1);
        // Four producers: at most two tails each, so no prior ever holds
        // more than the table's cap.
        for (std::int32_t r = 0; r < 4; ++r) {
            std::vector<std::int32_t> prompt;
            if (!published.empty() && rng() % 2 == 0) {
                // Share a prefix with an earlier stream, then diverge.
                const PublishedStream& base = published[rng() % published.size()];
                const auto shared = static_cast<std::int32_t>(rng() % base.tokens.size()) + 1;
                prompt.assign(base.tokens.begin(), base.tokens.begin() + shared);
            }
            const auto extra = static_cast<std::int32_t>(rng() % 9) + 1;
            for (std::int32_t i = 0; i < extra; ++i) {
                prompt.push_back(next_token++);
            }
            const auto generated = static_cast<std::int32_t>(rng() % 6);
            const std::string id = "r" + std::to_string(trial) + "_" + std::to_string(r);
            const auto prompt_len = static_cast<std::int32_t>(prompt.size());
            const std::int32_t expected = OracleHit(published, prompt, prompt_len - 1);
            const std::int32_t hit = Run(id, prompt, generated, next_token);
            EXPECT_EQ(hit, expected) << id;
            next_token += generated + 1;

            // What this stream published: full pages below its frontier
            // (finish computes every token but the last), its prompt end,
            // and its frontier.
            std::vector<std::int32_t> tokens = prompt;
            for (std::int32_t step = 0; step <= generated; ++step) {
                tokens.push_back(next_token - generated - 1 + step);
            }
            const auto frontier = static_cast<std::int32_t>(tokens.size()) - 1;
            PublishedStream stream{.tokens = tokens};
            for (std::int32_t page = 1; page * kP <= frontier; ++page) {
                stream.boundaries.push_back(page * kP);
            }
            stream.boundaries.push_back(prompt_len);
            stream.boundaries.push_back(frontier);
            published.push_back(std::move(stream));
        }
        // Probes over prefixes of every stream, with and without a diverging
        // last token.
        for (const PublishedStream& stream : published) {
            for (std::int32_t len = 1; len <= static_cast<std::int32_t>(stream.tokens.size()); ++len) {
                std::vector<std::int32_t> prompt(stream.tokens.begin(), stream.tokens.begin() + len);
                if (rng() % 3 == 0) {
                    prompt.back() = -prompt.back();
                }
                prompt.push_back(next_token++);
                const auto prompt_len = static_cast<std::int32_t>(prompt.size());
                const std::int32_t expected = OracleHit(published, prompt, prompt_len - 1);
                const std::string id = "q" + std::to_string(trial) + "_" + std::to_string(next_token);
                Submit(RequestSpec{.request_id = id, .tokens = prompt, .max_new_tokens = 1});
                const ExecutionPlan plan = PlanOnce();
                const ForwardBatch* batch = FindForwardBatch(plan);
                ASSERT_NE(batch, nullptr);
                ASSERT_EQ(batch->request_ids, std::vector<std::string>{id});
                EXPECT_EQ(batch->extend_prefix_lens.at(0), expected) << id << " len " << len;
                // A boundary inside a page arrives with copies of the tail's
                // blocks -- the page's blocks below the boundary -- whose
                // destinations are the table's slots for them and not zeroed.
                for (const auto& [group_id, pairs] : plan.pages_to_copy) {
                    const std::int32_t block_granularity = group_id == "full" ? 2 : kP;
                    const std::int32_t tail_tokens = expected % kP;
                    const std::int32_t tail_blocks = (tail_tokens + block_granularity - 1) / block_granularity;
                    ASSERT_EQ(pairs.size(), static_cast<std::size_t>(tail_blocks)) << id << " " << group_id;
                    const auto& row = batch->block_tables.at(group_id).at(0);
                    const std::int32_t first_slot = (expected - tail_tokens) / block_granularity;
                    for (std::int32_t j = 0; j < tail_blocks; ++j) {
                        EXPECT_EQ(row.at(static_cast<std::size_t>(first_slot + j)),
                                  pairs[static_cast<std::size_t>(j)].second);
                        const auto& zeroed = plan.pages_to_zero.at(group_id);
                        EXPECT_EQ(std::ranges::find(zeroed, pairs[static_cast<std::size_t>(j)].second), zeroed.end());
                    }
                }
                SendAbortEvent(id);
                PlanOnce();
            }
        }
    }
}

class TokenLevelHitWithSlidingGroupSuite : public TokenLevelHitSuite {
protected:
    bool WithSlidingGroup() const override { return true; }
};

TEST_F(TokenLevelHitWithSlidingGroupSuite, ASlidingGroupFollowsTheSameBoundaries) {
    // With an ample pool every published block of the sliding group stays
    // cached, so its lookback behind any published boundary is intact and the
    // joint boundary equals the closed group's.
    std::mt19937 rng{7};
    SetUp();
    std::vector<PublishedStream> published;
    std::int32_t next_token = 1;
    for (std::int32_t r = 0; r < 3; ++r) {
        std::vector<std::int32_t> prompt;
        if (!published.empty()) {
            const PublishedStream& base = published.back();
            const auto shared = static_cast<std::int32_t>(rng() % base.tokens.size()) + 1;
            prompt.assign(base.tokens.begin(), base.tokens.begin() + shared);
        }
        for (std::int32_t i = 0; i < 7; ++i) {
            prompt.push_back(next_token++);
        }
        const std::int32_t generated = 3;
        const auto prompt_len = static_cast<std::int32_t>(prompt.size());
        const std::int32_t expected = OracleHit(published, prompt, prompt_len - 1);
        EXPECT_EQ(Run("r" + std::to_string(r), prompt, generated, next_token), expected);
        std::vector<std::int32_t> tokens = prompt;
        for (std::int32_t step = 0; step <= generated; ++step) {
            tokens.push_back(next_token + step);
        }
        next_token += generated + 1;
        const auto frontier = static_cast<std::int32_t>(tokens.size()) - 1;
        PublishedStream stream{.tokens = tokens};
        for (std::int32_t page = 1; page * kP <= frontier; ++page) {
            stream.boundaries.push_back(page * kP);
        }
        stream.boundaries.push_back(prompt_len);
        stream.boundaries.push_back(frontier);
        published.push_back(std::move(stream));
    }
    for (const PublishedStream& stream : published) {
        for (std::int32_t len = 1; len <= static_cast<std::int32_t>(stream.tokens.size()); ++len) {
            std::vector<std::int32_t> prompt(stream.tokens.begin(), stream.tokens.begin() + len);
            prompt.push_back(next_token++);
            const std::string id = "q" + std::to_string(next_token);
            Submit(RequestSpec{.request_id = id, .tokens = prompt, .max_new_tokens = 1});
            const ExecutionPlan plan = PlanOnce();
            const ForwardBatch* batch = FindForwardBatch(plan);
            ASSERT_NE(batch, nullptr);
            ASSERT_EQ(batch->request_ids, std::vector<std::string>{id});
            EXPECT_EQ(batch->extend_prefix_lens.at(0), OracleHit(published, prompt, len)) << id;
            SendAbortEvent(id);
            PlanOnce();
        }
    }
}

// The gpt-oss shape: full attention and a 128-token sliding window over
// 64-token pages, no State group. A cold prompt decodes and finishes; a
// prompt one token shorter resumes at the longest boundary both groups hold.
class GptOssShapeSuite : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kP = 64;
    virtual std::int32_t BlockGranularity() const { return 16; }

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = kP;
        cfg.device_allocator.total_pages = 257;
        cfg.host_allocator.total_pages = 0;
        cfg.disable_l2_cache = true;
        cfg.max_scheduled_tokens = 1024;
        cfg.max_batch_size = 16;
        cfg.overlap_schedule_depth = 1;
        cfg.enable_mixed_prefill_decode = true;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full",
            .block_granularity = BlockGranularity(),
            .total_pages = 1 + 256 * (kP / BlockGranularity()),
            .cache_blocks_per_lcm_block = kP / BlockGranularity(),
            .attn_kind = AttnKind::kFull,
        });
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "sliding",
            .block_granularity = BlockGranularity(),
            .total_pages = 1 + 256 * (kP / BlockGranularity()),
            .cache_blocks_per_lcm_block = kP / BlockGranularity(),
            .attn_kind = AttnKind::kSlidingWindow,
            .sliding_window_tokens = 128,

        });
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(GptOssShapeSuite, APromptOneTokenShorterThanAFinishedOneResumesAtItsLastPage) {
    const std::vector<std::int32_t> prompt = MakeTokens(276);
    Submit(RequestSpec{.request_id = "x", .tokens = prompt, .max_new_tokens = 32});
    const ExecutionPlan prefill = PlanOnce();
    ASSERT_NE(FindForwardBatch(prefill), nullptr);
    std::int32_t next_token = 5000;
    SendForwardDone("x", {next_token++});
    for (std::int32_t step = 0; step < 24; ++step) {
        ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr) << "decode step " << step;
        SendForwardDone("x", {next_token++});
    }
    SendFinish("x");
    PlanOnce();

    // X + Y hits the prompt end exactly, decodes four steps and finishes.
    std::vector<std::int32_t> extended = prompt;
    for (std::int32_t i = 0; i < 10; ++i) {
        extended.push_back(9000 + i);
    }
    Submit(RequestSpec{.request_id = "xy", .tokens = extended, .max_new_tokens = 8});
    const ExecutionPlan xy_plan = PlanOnce();
    const ForwardBatch* xy = FindForwardBatch(xy_plan);
    ASSERT_NE(xy, nullptr);
    EXPECT_EQ(xy->extend_prefix_lens, std::vector<std::int32_t>{276});
    SendForwardDone("xy", {next_token++});
    for (std::int32_t step = 0; step < 4; ++step) {
        ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr) << "xy decode step " << step;
        SendForwardDone("xy", {next_token++});
    }
    SendFinish("xy");
    PlanOnce();

    // X[:-1]: 275 tokens, probed up to 274. Pages 0..3 of the full group are
    // cached and the sliding group holds the lookback behind 256, so the hit
    // is the page boundary 256 (the tails at 276 and 299 lie beyond the bound).
    std::vector<std::int32_t> shorter(prompt.begin(), prompt.end() - 1);
    Submit(RequestSpec{.request_id = "y", .tokens = shorter, .max_new_tokens = 4});
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"y"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{256});
    SendAbortEvent("y");
    PlanOnce();

    // X[:-37]: 239 tokens, probed up to 238. The full group holds pages 0..2,
    // but the sliding group publishes only the lookback behind its newest
    // boundary (256) and its tails; the blocks behind 192 slid out of X's
    // table unpublished, so no boundary below 256 is resumable.
    std::vector<std::int32_t> cut(prompt.begin(), prompt.end() - 37);
    Submit(RequestSpec{.request_id = "z", .tokens = cut, .max_new_tokens = 4});
    const ExecutionPlan cut_plan = PlanOnce();
    const ForwardBatch* cut_batch = FindForwardBatch(cut_plan);
    ASSERT_NE(cut_batch, nullptr);
    ASSERT_EQ(cut_batch->request_ids, std::vector<std::string>{"z"});
    EXPECT_EQ(cut_batch->extend_prefix_lens, std::vector<std::int32_t>{0});
}

class GptOssShapeWithWholePagesSuite : public GptOssShapeSuite {
protected:
    std::int32_t BlockGranularity() const override { return kP; }
};

TEST_F(GptOssShapeWithWholePagesSuite, APromptOneTokenShorterThanAFinishedOneResumesAtItsLastPage) {
    const std::vector<std::int32_t> prompt = MakeTokens(276);
    Submit(RequestSpec{.request_id = "x", .tokens = prompt, .max_new_tokens = 32});
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    std::int32_t next_token = 5000;
    SendForwardDone("x", {next_token++});
    for (std::int32_t step = 0; step < 24; ++step) {
        ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr) << "decode step " << step;
        SendForwardDone("x", {next_token++});
    }
    SendFinish("x");
    PlanOnce();
    std::vector<std::int32_t> shorter(prompt.begin(), prompt.end() - 1);
    Submit(RequestSpec{.request_id = "y", .tokens = shorter, .max_new_tokens = 4});
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{256});
    SendAbortEvent("y");
    PlanOnce();

    // A prompt one token longer than X hits X's prompt end at 276, and the
    // first query after it (position 276) attends the window [149, 276): the
    // sliding row must hold blocks 2, 3 and 4 -- one more than the lookback
    // behind a page-aligned boundary.
    std::vector<std::int32_t> longer = prompt;
    longer.push_back(77);
    Submit(RequestSpec{.request_id = "z", .tokens = longer, .max_new_tokens = 4});
    const ExecutionPlan z_plan = PlanOnce();
    const ForwardBatch* z = FindForwardBatch(z_plan);
    ASSERT_NE(z, nullptr);
    EXPECT_EQ(z->extend_prefix_lens, std::vector<std::int32_t>{276});
    const auto& swa = z->block_tables.at("sliding").at(0);
    ASSERT_GE(swa.size(), 5u);
    EXPECT_EQ(swa.at(0), 0);
    EXPECT_EQ(swa.at(1), 0);
    EXPECT_GT(swa.at(2), 0) << "[128, 192) holds rows 149..191 of the window";
    EXPECT_GT(swa.at(3), 0);
    EXPECT_GT(swa.at(4), 0) << "the copied tail block [256, 276)";
}

// A pool filled by prompts whose generation budget their admission prepaid
// (retraction-exempt), each ending inside a page: their first decodes would
// each copy the published partial block into a fresh block, and the pool has
// none. The owner then keeps its partial block private and publishes nothing
// at the prompt end, so every request still decodes and finishes.
class ExemptOwnersFillThePoolSuite : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kP = 16;

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = kP;
        cfg.device_allocator.total_pages = 1 + 24;
        cfg.host_allocator.total_pages = 0;
        cfg.disable_l2_cache = true;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_mixed_prefill_decode = false;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full",
            .block_granularity = kP,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kFull,
        });
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "sliding",
            .block_granularity = kP,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kSlidingWindow,
            .sliding_window_tokens = 2 * kP,

        });
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(ExemptOwnersFillThePoolSuite, FirstDecodesProceedWithoutTheirCopyIn) {
    // Four 23-token prompts with a two-token budget: 2 full blocks (23 + 1
    // decode slot + 1 headroom) + 2 sliding blocks each -> 16 blocks; three
    // more such prompts queue behind them on the remaining 8 and 0 blocks.
    std::vector<std::string> ids;
    for (std::int32_t i = 0; i < 7; ++i) {
        ids.push_back("r" + std::to_string(i));
        Submit(RequestSpec{.request_id = ids.back(), .tokens = MakeTokens(23, 1000 * (i + 1)), .max_new_tokens = 2});
    }
    std::map<std::string, std::int32_t> produced;
    std::map<std::string, std::int32_t> computed;
    std::set<std::string> finished;
    std::int32_t idle = 0;
    std::int32_t next_token = 50000;
    for (std::int32_t round = 0; round < 200 && finished.size() < ids.size() && idle < 10; ++round) {
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        if (batch == nullptr || batch->request_ids.empty()) {
            ++idle;
            continue;
        }
        idle = 0;
        for (std::size_t row = 0; row < batch->request_ids.size(); ++row) {
            const std::string& id = batch->request_ids[row];
            if (row < batch->extend_prefix_lens.size()) {
                computed[id] = batch->extend_prefix_lens[row] + batch->input_lengths[row];
                if (computed[id] < 23) {
                    SendForwardDone(id);  // an intermediate chunk produces no token
                    continue;
                }
            }
            SendForwardDone(id, {next_token++});
            if (++produced[id] >= 2) {
                SendFinish(id);
                finished.insert(id);
            }
        }
    }
    EXPECT_EQ(finished.size(), ids.size()) << "idle rounds: " << idle << ", waiting " << scheduler_->WaitingSize();
    PlanOnce();
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

}  // namespace
}  // namespace tokenspeed::test
