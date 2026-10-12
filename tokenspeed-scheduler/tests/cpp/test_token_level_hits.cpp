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
            .retention = CacheGroupConfig::Retention::FullHistory,
            .family = CacheGroupFamily::History,
        });
        if (WithSlidingGroup()) {
            cfg.cache_groups.push_back(CacheGroupConfig{
                .group_id = "swa",
                .block_granularity = kP,
                .total_pages = cfg.device_allocator.total_pages,
                .retention = CacheGroupConfig::Retention::SlidingWindow,
                .sliding_window_tokens = 5,
                .family = CacheGroupFamily::History,
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

}  // namespace
}  // namespace tokenspeed::test
