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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "integration_test_helper.h"
#include "unit_test_helper.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kP = 2;
constexpr std::int64_t kLookupTimeoutUs = 100;

// One full-history group, and with `hybrid` a sliding window and a State
// group too.
SchedulerConfig PeerConfigFor(const std::string& name, bool hybrid = false) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = kP;
    cfg.device_allocator.total_pages = 32;
    cfg.host_allocator.total_pages = 32;
    cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
    cfg.max_scheduled_tokens = 64;
    cfg.max_batch_size = 8;
    cfg.max_resident_requests = cfg.max_batch_size;
    cfg.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full_attention",
        .block_granularity = kP,
        .total_pages = cfg.device_allocator.total_pages,
        .attn_kind = AttnKind::kFull,
    });
    if (hybrid) {
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "sliding_attention",
            .block_granularity = kP,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kSlidingWindow,
            .sliding_window_tokens = 2 * kP,
        });
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "linear_attention",
            .block_granularity = 0,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kState,
        });
    }
    cfg.peer_config = PeerConfig{.name = name,
                                 .max_leases = 8,
                                 .soft_pin_us = 1000,
                                 .hard_pin_us = 1000,
                                 .lookup_timeout_us = kLookupTimeoutUs,
                                 .min_fetch_tokens = kP};
    SetTestSnapshotPool(cfg);
    return WithHostPackingFromDevice(cfg);
}

ExecutionPlan Plan(Scheduler& scheduler) {
    return scheduler.NextExecutionPlan(Scheduler::PlanOptions{.allow_prefill = true, .apply_prefill_watermark = false});
}

template <typename Event>
void Send(Scheduler& scheduler, Event event) {
    ExecutionEvent execution_event;
    execution_event.With(std::move(event));
    scheduler.Advance(std::move(execution_event));
}

// The prefill rows of `plan`'s batch, as (request id, prefix it extends).
std::vector<std::pair<std::string, std::int32_t>> PrefillRows(const ExecutionPlan& plan) {
    std::vector<std::pair<std::string, std::int32_t>> rows;
    if (const ForwardBatch* batch = FindForwardBatch(plan)) {
        for (std::size_t i = 0; i < batch->request_ids.size(); ++i) {
            if (i < batch->extend_prefix_lens.size()) {
                rows.emplace_back(batch->request_ids[i], batch->extend_prefix_lens[i]);
            }
        }
    }
    return rows;
}

// Two replicas of one deployment: a holder that computed a prompt and a
// reader that is asked to continue it; the test relays their messages as the
// runtime would.
class PeerSchedulerTest : public ::testing::TestWithParam<bool> {
protected:
    // The holder prefills `tokens` and finishes, publishing them.
    void HolderComputes(const std::vector<std::int32_t>& tokens) {
        holder_.SubmitRequests({RequestSpec{.request_id = "h", .tokens = tokens, .max_new_tokens = 1}});
        const ExecutionPlan plan = Plan(holder_);
        ASSERT_EQ(PrefillRows(plan).size(), 1u);
        Send(holder_, forward::ExtendResult{.request_id = "h", .tokens = {7}});
        Send(holder_, forward::Finish{.request_id = "h"});
        Plan(holder_);
    }

    // Relays the reader's lookups to the holder and the holder's answers back.
    void RelayLookups() {
        for (const PeerLookupRequest& lookup : reader_.TakePeerLookups()) {
            EXPECT_EQ(lookup.holder, "holder");
            Send(holder_, cache::PeerLookup{.lease = lookup.lease,
                                            .prefix_hashes = lookup.prefix_hashes,
                                            .tokens = lookup.tokens,
                                            .max_tokens = lookup.max_tokens});
        }
        for (const PeerAnswer& answer : holder_.TakePeerAnswers()) {
            ASSERT_EQ(answer.kind, PeerAnswerKind::kLookup);
            Send(reader_, cache::PeerLookupAnswered{.lease = answer.lease, .num_tokens = answer.num_tokens});
        }
    }

    // Runs a landing as the runtime does: pins it at the holder, (reads,)
    // releases it and reports the boundary confirmed.
    std::int32_t Fetch(const PeerFetchOperation& fetch, bool read_succeeds) {
        Send(holder_, cache::PeerPin{.lease = fetch.lease});
        std::vector<PeerAnswer> answers = holder_.TakePeerAnswers();
        EXPECT_EQ(answers.size(), 1u);
        const PeerAnswer pinned = answers.front();
        EXPECT_EQ(pinned.kind, PeerAnswerKind::kPin);
        // Every destination has a holder page at its slot.
        for (const PeerPage& destination : fetch.destinations) {
            EXPECT_TRUE(std::ranges::any_of(pinned.pages,
                                            [&](const PeerPage& page) {
                                                return page.group_id == destination.group_id &&
                                                       page.slot == destination.slot;
                                            }))
                << "slot " << destination.slot;
        }
        Send(holder_, cache::PeerRelease{.lease = fetch.lease, .demote = read_succeeds});
        answers = holder_.TakePeerAnswers();
        const bool held = answers.size() == 1 && answers.front().held;
        const std::int32_t confirmed =
            read_succeeds && held && pinned.num_tokens == fetch.end_tokens ? fetch.end_tokens : fetch.begin_tokens;
        Send(reader_, cache::PeerFetchDone{.request_id = fetch.request_id, .num_tokens = confirmed});
        return confirmed;
    }

    Scheduler holder_{PeerConfigFor("holder", GetParam())};
    Scheduler reader_{PeerConfigFor("reader", GetParam())};
};

INSTANTIATE_TEST_SUITE_P(Groups, PeerSchedulerTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "Hybrid" : "Full"; });

TEST_P(PeerSchedulerTest, AReaderLandsTheHoldersPrefixAndComputesTheRest) {
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(8, kP);
    HolderComputes(tokens);
    std::vector<std::int32_t> longer = tokens;
    longer.insert(longer.end(), {901, 902, 903, 904});
    reader_.SubmitRequests({RequestSpec{.request_id = "r", .peer_holder = "holder", .tokens = longer}});

    // The first plan asks the holder and holds the prompt back meanwhile.
    EXPECT_TRUE(PrefillRows(Plan(reader_)).empty());
    EXPECT_EQ(reader_.GetPeerStats().looking_requests, 1u);
    RelayLookups();

    // The landing rides beside the batch, which does not carry the prompt.
    const ExecutionPlan landing_plan = Plan(reader_);
    EXPECT_TRUE(PrefillRows(landing_plan).empty());
    ASSERT_EQ(landing_plan.peer_fetches.size(), 1u);
    const PeerFetchOperation fetch = landing_plan.peer_fetches.front();
    EXPECT_EQ(fetch.request_id, "r");
    EXPECT_EQ(fetch.holder, "holder");
    EXPECT_EQ(fetch.begin_tokens, 0);
    EXPECT_EQ(fetch.end_tokens, static_cast<std::int32_t>(tokens.size()));
    const auto destinations_of = [&](std::uint32_t group) {
        return std::ranges::count_if(fetch.destinations,
                                     [group](const PeerPage& page) { return page.group_id == group; });
    };
    EXPECT_EQ(destinations_of(0), static_cast<std::ptrdiff_t>(tokens.size() / kP)) << "every full page";
    if (GetParam()) {
        EXPECT_EQ(destinations_of(1), 2) << "the window behind the landing's end";
        EXPECT_EQ(destinations_of(2), 1) << "the State group's live slot";
        EXPECT_TRUE(std::ranges::any_of(fetch.destinations, [](const PeerPage& page) {
            return page.group_id == 2 && page.slot == kPeerStateSlot;
        }));
    }
    EXPECT_EQ(reader_.GetPeerStats().landing_requests, 1u);
    EXPECT_TRUE(PrefillRows(Plan(reader_)).empty()) << "nothing reads a landing's pages";

    EXPECT_EQ(Fetch(fetch, /*read_succeeds=*/true), fetch.end_tokens);
    EXPECT_EQ(PrefillRows(Plan(reader_)), (std::vector<std::pair<std::string, std::int32_t>>{{"r", fetch.end_tokens}}));
    EXPECT_EQ(reader_.GetPeerStats().landed_tokens, static_cast<std::uint64_t>(fetch.end_tokens));
    EXPECT_EQ(holder_.GetPeerStats().leases, 0u);
}

TEST_P(PeerSchedulerTest, AShortLandingIsAdmittedAgainFromTheLocalHit) {
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(8, kP);
    HolderComputes(tokens);
    std::vector<std::int32_t> longer = tokens;
    longer.insert(longer.end(), {901, 902});
    reader_.SubmitRequests({RequestSpec{.request_id = "r", .peer_holder = "holder", .tokens = longer}});
    Plan(reader_);
    RelayLookups();
    const ExecutionPlan landing_plan = Plan(reader_);
    ASSERT_EQ(landing_plan.peer_fetches.size(), 1u);

    EXPECT_EQ(Fetch(landing_plan.peer_fetches.front(), /*read_succeeds=*/false), 0);
    const ExecutionPlan plan = Plan(reader_);
    EXPECT_EQ(PrefillRows(plan), (std::vector<std::pair<std::string, std::int32_t>>{{"r", 0}}));
    EXPECT_TRUE(plan.peer_fetches.empty()) << "it asked its holder once";
    EXPECT_TRUE(reader_.TakePeerLookups().empty());
    EXPECT_EQ(reader_.GetPeerStats().failed_landings, 1u);
}

TEST_P(PeerSchedulerTest, AMissOrAnUnansweredLookupIsAdmittedLocally) {
    reader_.SubmitRequests(
        {RequestSpec{.request_id = "miss", .peer_holder = "holder", .tokens = MakeAlignedTokens(4, kP)},
         RequestSpec{
             .request_id = "late", .peer_holder = "holder", .tokens = MakeAlignedTokens(4, kP, /*start=*/500)}});
    Send(reader_, cache::PeerTick{.now_us = 0});
    Plan(reader_);
    const std::vector<PeerLookupRequest> lookups = reader_.TakePeerLookups();
    ASSERT_EQ(lookups.size(), 2u);
    Send(reader_, cache::PeerLookupAnswered{.lease = lookups[0].lease, .num_tokens = 0});
    EXPECT_EQ(PrefillRows(Plan(reader_)), (std::vector<std::pair<std::string, std::int32_t>>{{"miss", 0}}));

    Send(reader_, cache::PeerTick{.now_us = kLookupTimeoutUs});
    EXPECT_EQ(PrefillRows(Plan(reader_)), (std::vector<std::pair<std::string, std::int32_t>>{{"late", 0}}));
    // An answer after the wait ended changes nothing.
    Send(reader_, cache::PeerLookupAnswered{.lease = lookups[1].lease, .num_tokens = 4 * kP});
    EXPECT_TRUE(Plan(reader_).peer_fetches.empty());
}

TEST_P(PeerSchedulerTest, NoLookupWithoutAHintOrForTooShortAGain) {
    reader_.SubmitRequests(
        {RequestSpec{.request_id = "plain", .tokens = MakeAlignedTokens(4, kP)},
         RequestSpec{.request_id = "self", .peer_holder = "reader", .tokens = MakeAlignedTokens(4, kP, 200)},
         // Less than a page before the replay tail: nothing worth a fetch.
         RequestSpec{.request_id = "short", .peer_holder = "holder", .tokens = {1, 2}}});
    EXPECT_EQ(PrefillRows(Plan(reader_)).size(), 3u);
    EXPECT_TRUE(reader_.TakePeerLookups().empty());
}

TEST_P(PeerSchedulerTest, AnAbortWhileLandingWaitsForTheFetchToEnd) {
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(8, kP);
    HolderComputes(tokens);
    std::vector<std::int32_t> longer = tokens;
    longer.insert(longer.end(), {901, 902});
    reader_.SubmitRequests({RequestSpec{.request_id = "r", .peer_holder = "holder", .tokens = longer}});
    Plan(reader_);
    RelayLookups();
    const ExecutionPlan landing_plan = Plan(reader_);
    ASSERT_EQ(landing_plan.peer_fetches.size(), 1u);
    ASSERT_GT(reader_.ActiveLcmBlocks(), 0);

    Send(reader_, forward::Abort{.request_id = "r"});
    Plan(reader_);
    EXPECT_GT(reader_.ActiveLcmBlocks(), 0) << "the reads may still write its pages";
    Fetch(landing_plan.peer_fetches.front(), /*read_succeeds=*/true);
    Plan(reader_);
    EXPECT_EQ(reader_.ActiveLcmBlocks(), 0);
    EXPECT_EQ(reader_.GetPeerStats().landing_requests, 0u);
}

TEST_P(PeerSchedulerTest, AHolderAnswersFromItsOwnCacheAndHoldsAPinUntilTheRelease) {
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(4, kP);
    HolderComputes(tokens);
    const std::vector<std::int32_t> prefix(tokens.begin(), tokens.end() - 1);
    Send(holder_, cache::PeerTick{.now_us = 0});
    Send(holder_,
         cache::PeerLookup{.lease = {.reader = "x", .id = 1}, .prefix_hashes = {}, .tokens = {}, .max_tokens = 0});
    std::vector<PeerAnswer> answers = holder_.TakePeerAnswers();
    ASSERT_EQ(answers.size(), 1u);
    EXPECT_EQ(answers.front().num_tokens, 0) << "a malformed lookup is a miss";
    EXPECT_EQ(holder_.GetPeerStats().leases, 0u);
}

TEST(PeerSchedulerConfigTest, PeerSharingNeedsPrefixCachingOutsideTheDRoleAndNoReplayableGroup) {
    SchedulerConfig decode = PeerConfigFor("d");
    decode.role = Role::kD;
    EXPECT_THROW(decode.Validate(), std::invalid_argument);
    SchedulerConfig uncached = PeerConfigFor("u");
    uncached.disable_prefix_cache = true;
    EXPECT_THROW(uncached.Validate(), std::invalid_argument);
    SchedulerConfig replayable = PeerConfigFor("r");
    replayable.cache_groups.front().replayable = true;
    EXPECT_THROW(replayable.Validate(), std::invalid_argument);
    EXPECT_NO_THROW(PeerConfigFor("ok").Validate());
}

}  // namespace
}  // namespace tokenspeed::test
