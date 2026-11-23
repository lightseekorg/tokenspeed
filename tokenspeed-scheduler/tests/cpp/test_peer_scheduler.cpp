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
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
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
                                 .incarnation = 7,
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

template <typename Batch>
std::vector<std::uint32_t> CacheOpIds(const ExecutionPlan& plan) {
    std::vector<std::uint32_t> ids;
    for (const Operation& op : plan.Operations()) {
        if (const auto* cache = std::get_if<CacheOperation>(&op)) {
            if (const auto* batch = std::get_if<Batch>(cache)) {
                ids.insert(ids.end(), batch->op_ids.begin(), batch->op_ids.end());
            }
        }
    }
    return ids;
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

// The holder's runtime half, as the engine host runs it: every message is
// answered at once from the scheduler's reads, and what it changes is queued
// for the next round's head, ahead of every other event.
class HolderServicer {
public:
    explicit HolderServicer(Scheduler& holder) : holder_{holder} {}

    PeerAnswer Lookup(const PeerLookupRequest& request) {
        const cache::PeerLookup lookup{.lease = request.lease,
                                       .prefix_hashes = request.prefix_hashes,
                                       .tokens = request.tokens,
                                       .max_tokens = request.max_tokens};
        const PeerAnswer answer = holder_.AnswerPeerLookup(lookup);
        lookups_.insert_or_assign(request.lease.id, std::pair{lookup, answer});
        pending_.With(lookup);
        return answer;
    }
    PeerAnswer Pin(const PeerLeaseId& lease) {
        const auto& [lookup, looked_up] = lookups_.at(lease.id);
        const bool matches = holder_.PeerPagesMatch(lookup, looked_up.num_tokens, looked_up.pages);
        if (matches) {
            pending_.With(cache::PeerPin{.lease = lease, .num_tokens = looked_up.num_tokens, .pages = looked_up.pages});
            pinned_at_.insert_or_assign(lease.id, holder_.PlanCalls());
        }
        return PeerAnswer{.lease = lease, .kind = PeerAnswerKind::kPin, .held = matches};
    }
    PeerAnswer Release(const PeerLeaseId& lease, bool demote) {
        // Untouched from the pin: no plan ran since it, or its pin held.
        const auto pinned = pinned_at_.find(lease.id);
        const bool held =
            pinned != pinned_at_.end() && (holder_.PlanCalls() == pinned->second || holder_.PeerLeaseHeld(lease));
        pending_.With(cache::PeerRelease{.lease = lease, .demote = demote && held});
        return PeerAnswer{.lease = lease, .kind = PeerAnswerKind::kRelease, .held = held};
    }
    // The next round's head: the clock, then the queued changes in order.
    void Head(std::int64_t now_us = 0) {
        ExecutionEvent event;
        event.With(cache::PeerTick{.now_us = now_us});
        for (auto& queued : pending_.Events()) {
            std::visit([&](auto& inner) { event.With(std::move(inner)); }, queued);
        }
        pending_ = ExecutionEvent{};
        holder_.Advance(std::move(event));
    }
    const PeerAnswer& Answer(const PeerLeaseId& lease) const { return lookups_.at(lease.id).second; }

private:
    Scheduler& holder_;
    ExecutionEvent pending_;
    std::map<std::uint64_t, std::pair<cache::PeerLookup, PeerAnswer>> lookups_;
    std::map<std::uint64_t, std::int64_t> pinned_at_;
};

// Two replicas of one deployment: a holder that computed a prompt and a
// reader that is asked to continue it; the test relays their messages as the
// runtime would.
class PeerSchedulerTest : public ::testing::TestWithParam<bool> {
protected:
    // The holder prefills `tokens` and finishes, publishing them.
    ExecutionPlan HolderComputes(const std::vector<std::int32_t>& tokens) {
        holder_.SubmitRequests({RequestSpec{.request_id = "h", .tokens = tokens, .max_new_tokens = 1}});
        const ExecutionPlan plan = Plan(holder_);
        EXPECT_EQ(PrefillRows(plan).size(), 1u);
        Send(holder_, forward::ExtendResult{.request_id = "h", .tokens = {7}});
        Send(holder_, forward::Finish{.request_id = "h"});
        return Plan(holder_);
    }

    // Relays the reader's lookups to the holder and the answers back.
    void RelayLookups() {
        for (const PeerLookupRequest& lookup : reader_.TakePeerLookups()) {
            EXPECT_EQ(lookup.holder, "holder");
            const PeerAnswer answer = servicer_.Lookup(lookup);
            Send(reader_, cache::PeerLookupAnswered{.lease = answer.lease, .num_tokens = answer.num_tokens});
        }
    }

    // Runs a landing as the runtime does: pins it at the holder, (reads,)
    // releases it and reports the boundary confirmed. The holder's head runs
    // between the pin and the release when `head_between` is set.
    std::int32_t Fetch(const PeerFetchOperation& fetch, bool read_succeeds, bool head_between = false) {
        const PeerAnswer pinned = servicer_.Pin(fetch.lease);
        // Every destination has a holder page at its slot.
        const PeerAnswer& looked_up = servicer_.Answer(fetch.lease);
        for (const PeerPage& destination : fetch.destinations) {
            EXPECT_TRUE(std::ranges::any_of(looked_up.pages,
                                            [&](const PeerPage& page) {
                                                return page.group_id == destination.group_id &&
                                                       page.slot == destination.slot;
                                            }))
                << "slot " << destination.slot;
        }
        if (head_between) {
            servicer_.Head();
            Plan(holder_);
        }
        const PeerAnswer released = servicer_.Release(fetch.lease, read_succeeds && pinned.held);
        const std::int32_t confirmed =
            read_succeeds && pinned.held && released.held && looked_up.num_tokens == fetch.end_tokens
                ? fetch.end_tokens
                : fetch.begin_tokens;
        Send(reader_, cache::PeerFetchDone{.request_id = fetch.request_id, .num_tokens = confirmed});
        servicer_.Head();
        return confirmed;
    }

    Scheduler holder_{PeerConfigFor("holder", GetParam())};
    Scheduler reader_{PeerConfigFor("reader", GetParam())};
    HolderServicer servicer_{holder_};
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
    // The plan's device work leaves the destinations alone: the reads are not
    // ordered with it.
    const std::vector<std::string> group_ids{"full_attention", "sliding_attention", "linear_attention"};
    for (const PeerPage& destination : fetch.destinations) {
        const std::string& group = group_ids[destination.group_id];
        EXPECT_EQ(std::ranges::count(landing_plan.pages_to_zero.at(group), destination.block_id), 0) << group;
        for (const auto& [source, copied] : landing_plan.pages_to_copy.at(group)) {
            EXPECT_NE(copied, destination.block_id) << group;
        }
    }
    EXPECT_TRUE(PrefillRows(Plan(reader_)).empty()) << "nothing reads a landing's pages";

    EXPECT_EQ(Fetch(fetch, /*read_succeeds=*/true, /*head_between=*/true), fetch.end_tokens);
    EXPECT_EQ(PrefillRows(Plan(reader_)), (std::vector<std::pair<std::string, std::int32_t>>{{"r", fetch.end_tokens}}));
    EXPECT_EQ(reader_.GetPeerStats().landed_tokens, static_cast<std::uint64_t>(fetch.end_tokens));
    EXPECT_EQ(holder_.GetPeerStats().leases, 0u);
}

TEST_P(PeerSchedulerTest, ALandedPromptEndingInsideAPagePublishesItsEnd) {
    // The prompt ends one token past the landed pages: its first decode
    // publishes the prompt end and keeps writing past it, through the
    // copy-in blocks the landing's admission reserved as a local prompt's
    // would.
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(8, kP);
    HolderComputes(tokens);
    std::vector<std::int32_t> longer = tokens;
    longer.push_back(901);
    // Decode room makes the landing's admission prepay the rest of the
    // prompt: the tail's block, and its copy-in slot, come with the landing.
    reader_.SubmitRequests(
        {RequestSpec{.request_id = "r", .peer_holder = "holder", .tokens = longer, .max_new_tokens = 4}});
    Plan(reader_);
    RelayLookups();
    const ExecutionPlan landing_plan = Plan(reader_);
    ASSERT_EQ(landing_plan.peer_fetches.size(), 1u);
    const PeerFetchOperation fetch = landing_plan.peer_fetches.front();
    EXPECT_EQ(Fetch(fetch, /*read_succeeds=*/true), fetch.end_tokens);
    EXPECT_EQ(PrefillRows(Plan(reader_)), (std::vector<std::pair<std::string, std::int32_t>>{{"r", fetch.end_tokens}}));
    Send(reader_, forward::ExtendResult{.request_id = "r", .tokens = {77}});
    Plan(reader_);  // the first decode publishes the prompt end
    Send(reader_, forward::ExtendResult{.request_id = "r", .tokens = {78}});
    Plan(reader_);
    Send(reader_, forward::Finish{.request_id = "r"});
    Plan(reader_);

    // The landed prompt, its end included, is the reader's own now.
    std::vector<std::int32_t> again = longer;
    again.insert(again.end(), {77, 600, 601, 602});
    reader_.SubmitRequests({RequestSpec{.request_id = "again", .tokens = again}});
    const auto rows = PrefillRows(Plan(reader_));
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_GE(rows.front().second, static_cast<std::int32_t>(tokens.size()));
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
    EXPECT_EQ(lookups[0].lease.id, (std::uint64_t{7} << 32) + 1) << "counted from the reader's incarnation";
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

TEST_P(PeerSchedulerTest, AHolderAnswersBetweenRoundsAndItsPinHoldsAcrossAPlan) {
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(8, kP);
    HolderComputes(tokens);
    std::vector<std::int32_t> longer = tokens;
    longer.insert(longer.end(), {901, 902});
    reader_.SubmitRequests({RequestSpec{.request_id = "r", .peer_holder = "holder", .tokens = longer}});
    Plan(reader_);
    RelayLookups();
    EXPECT_EQ(holder_.GetPeerStats().leases, 0u) << "answering changes nothing until the head";
    servicer_.Head();
    EXPECT_EQ(holder_.GetPeerStats().leases, 1u);
    const ExecutionPlan landing_plan = Plan(reader_);
    ASSERT_EQ(landing_plan.peer_fetches.size(), 1u);
    const PeerLeaseId lease = landing_plan.peer_fetches.front().lease;

    ASSERT_TRUE(servicer_.Pin(lease).held);
    servicer_.Head();
    EXPECT_GT(holder_.GetPeerStats().pinned_blocks, 0u);
    // A plan between the pin and the release: the pin held the pages.
    Plan(holder_);
    EXPECT_TRUE(servicer_.Release(lease, /*demote=*/false).held);
    servicer_.Head();
    EXPECT_EQ(holder_.GetPeerStats().leases, 0u);
    EXPECT_EQ(holder_.GetPeerStats().pinned_blocks, 0u);
}

TEST_P(PeerSchedulerTest, AReleaseAfterAPlanThePinMissedIsNotHeld) {
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(8, kP);
    HolderComputes(tokens);
    std::vector<std::int32_t> longer = tokens;
    longer.insert(longer.end(), {901, 902});
    reader_.SubmitRequests({RequestSpec{.request_id = "r", .peer_holder = "holder", .tokens = longer}});
    Plan(reader_);
    RelayLookups();
    servicer_.Head();
    const ExecutionPlan landing_plan = Plan(reader_);
    ASSERT_EQ(landing_plan.peer_fetches.size(), 1u);
    const PeerLeaseId lease = landing_plan.peer_fetches.front().lease;
    ASSERT_TRUE(servicer_.Pin(lease).held);
    // The lookup's soft pin expired before the head applied the pin, which
    // then holds nothing; a plan ran since the pin's answer.
    servicer_.Head(/*now_us=*/1'000'000);
    Plan(holder_);
    EXPECT_FALSE(servicer_.Release(lease, /*demote=*/true).held);
}

TEST_P(PeerSchedulerTest, AHolderPinsLoadedDevicePagesOnlyAfterTheirLoadAcknowledgment) {
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(4, kP);
    const ExecutionPlan stored = HolderComputes(tokens);
    const auto stores = CacheOpIds<WriteBackBatch>(stored);
    ASSERT_FALSE(stores.empty());
    for (const std::uint32_t id : stores) {
        Send(holder_, cache::WriteBackDone{.op_id = id});
    }
    ASSERT_TRUE(holder_.ClearL1Cache());

    // An unrelated Device prefix stays readable while this prefix loads.
    const std::vector<std::int32_t> ready_tokens = MakeAlignedTokens(2, kP, 501);
    HolderComputes(ready_tokens);
    std::vector<std::int32_t> longer = tokens;
    longer.insert(longer.end(), {901, 902});
    holder_.SubmitRequests({RequestSpec{.request_id = "load", .tokens = longer}});
    const ExecutionPlan loading = Plan(holder_);
    const auto loads = CacheOpIds<LoadBackBatch>(loading);
    ASSERT_FALSE(loads.empty());
    ASSERT_EQ(PrefillRows(loading),
              (std::vector<std::pair<std::string, std::int32_t>>{{"load", static_cast<std::int32_t>(tokens.size())}}));

    std::vector<std::int32_t> ready_longer = ready_tokens;
    ready_longer.insert(ready_longer.end(), {903, 904});
    reader_.SubmitRequests({RequestSpec{.request_id = "r", .peer_holder = "holder", .tokens = longer},
                            RequestSpec{.request_id = "ready", .peer_holder = "holder", .tokens = ready_longer}});
    Plan(reader_);
    const auto lookups = reader_.TakePeerLookups();
    ASSERT_EQ(lookups.size(), 2u);
    for (const PeerLookupRequest& lookup : lookups) {
        const PeerAnswer answer = servicer_.Lookup(lookup);
        ASSERT_GT(answer.num_tokens, 0);
        ASSERT_TRUE(
            std::ranges::any_of(answer.pages, [](const PeerPage& page) { return page.tier == CacheTier::kDevice; }));
        const bool ready = lookup.tokens.front() == ready_tokens.front();
        EXPECT_EQ(servicer_.Pin(lookup.lease).held, ready)
            << "the Device index alone does not establish that H2D has completed";
    }
    for (const std::uint32_t id : loads) {
        Send(holder_, cache::LoadBackDone{.op_id = id});
    }
    for (const PeerLookupRequest& lookup : lookups) {
        EXPECT_TRUE(servicer_.Pin(lookup.lease).held);
    }
}

// A local hit that ends inside a page (a tail) ends inside a block, which a
// landing would fill as well as the hit's copy: the landing starts at the
// page boundary below instead.
TEST(PeerLandingAlignmentTest, ALocalTailHitLandsFromThePageBoundaryBelow) {
    Scheduler holder{PeerConfigFor("holder")};
    Scheduler reader{PeerConfigFor("reader")};
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(8, kP);
    for (Scheduler* scheduler : {&holder, &reader}) {
        // The holder computes the whole prompt, the reader a page and a tail.
        const auto computed =
            scheduler == &holder ? tokens : std::vector<std::int32_t>(tokens.begin(), tokens.begin() + kP + 1);
        scheduler->SubmitRequests({RequestSpec{.request_id = "warm", .tokens = computed, .max_new_tokens = 1}});
        ASSERT_EQ(PrefillRows(Plan(*scheduler)).size(), 1u);
        Send(*scheduler, forward::ExtendResult{.request_id = "warm", .tokens = {7}});
        Send(*scheduler, forward::Finish{.request_id = "warm"});
        Plan(*scheduler);
    }
    reader.SubmitRequests({RequestSpec{.request_id = "r", .peer_holder = "holder", .tokens = tokens}});
    Plan(reader);
    HolderServicer servicer{holder};
    for (const PeerLookupRequest& lookup : reader.TakePeerLookups()) {
        const PeerAnswer answer = servicer.Lookup(lookup);
        ASSERT_GT(answer.num_tokens, 2 * kP);
        Send(reader, cache::PeerLookupAnswered{.lease = answer.lease, .num_tokens = answer.num_tokens});
    }
    const ExecutionPlan plan = Plan(reader);
    ASSERT_EQ(plan.peer_fetches.size(), 1u);
    EXPECT_EQ(plan.peer_fetches.front().begin_tokens % kP, 0);
    for (const auto& [group, copies] : plan.pages_to_copy) {
        EXPECT_TRUE(copies.empty()) << group;
    }
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
