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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "integration_test_helper.h"
#include "cache_event_test_sink.h"
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
                                 .min_fetch_tokens = kP,
                                 .landing_tier = PeerLandingTier::kDevice,
                                 .max_host_landing_blocks = 0,
                                 .host_ready_timeout_us = 0};
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
        const std::size_t pending_new =
            unsynced_new_leases_.size() - (unsynced_new_leases_.contains(request.lease) ? 1u : 0u);
        const PeerAnswer answer = holder_.AnswerPeerLookup(lookup, pending_new);
        if (answer.num_tokens > 0 && !holder_.PeerLeaseCanPin(request.lease) && !holder_.PeerLeaseHeld(request.lease)) {
            unsynced_new_leases_.insert(request.lease);
        }
        lookups_.insert_or_assign(request.lease.id, Record{.lookup = lookup, .answer = answer, .lookup_synced = false});
        pinned_at_.erase(request.lease.id);
        if (answer.num_tokens > 0 || holder_.PeerLeaseCanPin(request.lease)) {
            pending_.With(lookup);
        }
        return answer;
    }
    PeerAnswer Pin(const PeerLeaseId& lease) {
        const auto& record = lookups_.at(lease.id);
        const bool can_pin =
            record.answer.num_tokens > 0 &&
            (!record.lookup_synced || holder_.PeerLeaseCanPin(lease) || holder_.PeerLeaseHeld(lease)) &&
            holder_.PeerPagesMatch(record.lookup, record.answer.num_tokens, record.answer.pages);
        if (can_pin && pinned_at_.try_emplace(lease.id, holder_.PlanCalls()).second) {
            pending_.With(
                cache::PeerPin{.lease = lease, .num_tokens = record.answer.num_tokens, .pages = record.answer.pages});
        }
        return PeerAnswer{.lease = lease,
                          .kind = PeerAnswerKind::kPin,
                          .num_tokens = record.answer.num_tokens,
                          .pages = record.answer.pages,
                          .held = can_pin};
    }
    PeerAnswer Release(const PeerLeaseId& lease, bool demote) {
        // Untouched from the pin: no plan or cache clear ran since it, or its pin held.
        const auto pinned = pinned_at_.find(lease.id);
        const bool held =
            pinned != pinned_at_.end() && (holder_.PlanCalls() == pinned->second || holder_.PeerLeaseHeld(lease));
        lookups_.erase(lease.id);
        pinned_at_.erase(lease.id);
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
        for (auto& [id, record] : lookups_) {
            record.lookup_synced = true;
        }
        unsynced_new_leases_.clear();
        holder_.Advance(std::move(event));
    }
    const PeerAnswer& Answer(const PeerLeaseId& lease) const { return lookups_.at(lease.id).answer; }

private:
    struct Record {
        cache::PeerLookup lookup;
        PeerAnswer answer;
        bool lookup_synced{false};
    };
    Scheduler& holder_;
    ExecutionEvent pending_;
    std::map<std::uint64_t, Record> lookups_;
    std::map<std::uint64_t, std::int64_t> pinned_at_;
    std::unordered_set<PeerLeaseId, PeerLeaseIdHash> unsynced_new_leases_;
};

// Two replicas of one deployment: a holder that holds a prefix and a reader
// that is asked to continue it; the test relays their messages as the
// runtime would.
class PeerPair {
protected:
    PeerPair(SchedulerConfig holder, SchedulerConfig reader)
        : holder_{std::move(holder)}, reader_{std::move(reader)}, servicer_{holder_} {}

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

    Scheduler holder_;
    Scheduler reader_;
    HolderServicer servicer_;
};

// A holder that computed the prompt itself, on full-history and hybrid group
// sets.
class PeerSchedulerTest : public ::testing::TestWithParam<bool>, protected PeerPair {
protected:
    PeerSchedulerTest() : PeerPair(PeerConfigFor("holder", GetParam()), PeerConfigFor("reader", GetParam())) {}

    // The holder prefills `tokens` and finishes, publishing them.
    ExecutionPlan HolderComputes(const std::vector<std::int32_t>& tokens) {
        holder_.SubmitRequests({RequestSpec{.request_id = "h", .tokens = tokens, .max_new_tokens = 1}});
        const ExecutionPlan plan = Plan(holder_);
        EXPECT_EQ(PrefillRows(plan).size(), 1u);
        Send(holder_, forward::ExtendResult{.request_id = "h", .tokens = {7}});
        Send(holder_, forward::Finish{.request_id = "h"});
        return Plan(holder_);
    }
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

// A retraction publishes a decoding victim's frontier. When the State group
// defers its decode state, the runtime commits it into the published block
// only while it runs that plan, so no PIN exports the block until the plan's
// commit fence is acknowledged; without deferral the endpoint is readable at
// once.
TEST(PeerStateCommitTest, ARetractedDecodeEndpointIsPinnedOnlyAfterItsCommit) {
    for (const bool deferred : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "deferred=" << deferred);
        auto cfg = PeerConfigFor("holder", /*hybrid=*/true);
        cfg.cache_groups.back().deferred_decode_state = deferred;
        cfg.debug_force_retraction_interval = 1;
        Scheduler holder{std::move(cfg)};
        Scheduler reader{PeerConfigFor("reader", /*hybrid=*/true)};
        HolderServicer servicer{holder};

        holder.SubmitRequests({RequestSpec{.request_id = "seed", .tokens = {11, 12, 13}, .max_new_tokens = 8}});
        ASSERT_EQ(PrefillRows(Plan(holder)).size(), 1u);
        Send(holder, forward::ExtendResult{.request_id = "seed", .tokens = {14}});
        // The first decode consumes 14; the retraction then publishes the
        // state after four tokens.
        ASSERT_NE(FindForwardBatch(Plan(holder)), nullptr);
        Send(holder, forward::ExtendResult{.request_id = "seed", .tokens = {15}});
        std::optional<ExecutionPlan> retraction;
        for (int round = 0; round < 3 && holder.RetractedSize() == 0; ++round) {
            retraction = Plan(holder);
        }
        ASSERT_EQ(holder.RetractedSize(), 1u);
        EXPECT_EQ(retraction->state_commit_fence.has_value(), deferred);

        reader.SubmitRequests(
            {RequestSpec{.request_id = "hit", .peer_holder = "holder", .tokens = {11, 12, 13, 14, 80, 81}}});
        Plan(reader);
        const auto lookups = reader.TakePeerLookups();
        ASSERT_EQ(lookups.size(), 1u);
        ASSERT_EQ(servicer.Lookup(lookups.front()).num_tokens, 4);
        EXPECT_EQ(servicer.Pin(lookups.front().lease).held, !deferred);
        if (deferred) {
            Send(holder, cache::StateCommitDone{.op_id = *retraction->state_commit_fence});
            EXPECT_TRUE(servicer.Pin(lookups.front().lease).held);
        }
    }
}

// A local hit that ends inside a page (a tail) ends inside a block, which a
// landing would fill as well as the hit's copy: the landing starts at the
// page boundary below instead.
TEST(PeerLoadOwnershipTest, HostPrefixLoadUsesTheAllocatedLandingSlot) {
    Scheduler holder{PeerConfigFor("holder")};
    auto reader_config = PeerConfigFor("reader");
    reader_config.max_batch_size = 1;
    reader_config.max_resident_requests = 1;
    Scheduler reader{reader_config};
    auto sink = std::make_shared<RecordingCacheEventSink>();
    reader.SetCacheEventSink(sink);
    const auto tokens = MakeAlignedTokens(8, kP);
    for (Scheduler* scheduler : {&holder, &reader}) {
        const auto computed =
            scheduler == &holder ? tokens : std::vector<std::int32_t>(tokens.begin(), tokens.begin() + 2 * kP);
        scheduler->SubmitRequests(
            {RequestSpec{.request_id = "warm", .admission_id = 11, .tokens = computed, .max_new_tokens = 1}});
        Plan(*scheduler);
        Send(*scheduler, forward::ExtendResult{.request_id = "warm", .tokens = {7}});
        Send(*scheduler, forward::Finish{.request_id = "warm"});
        const auto finish = Plan(*scheduler);
        for (const auto id : CacheOpIds<WriteBackBatch>(finish)) Send(*scheduler, cache::WriteBackDone{.op_id = id});
    }
    ASSERT_TRUE(reader.ClearL1Cache());
    reader.SubmitRequests(
        {RequestSpec{.request_id = "r", .admission_id = 22, .peer_holder = "holder", .tokens = tokens}});
    Plan(reader);
    HolderServicer servicer{holder};
    for (const auto& lookup : reader.TakePeerLookups()) {
        const auto answer = servicer.Lookup(lookup);
        Send(reader, cache::PeerLookupAnswered{.lease = answer.lease, .num_tokens = answer.num_tokens});
    }
    const auto plan = Plan(reader);
    ASSERT_EQ(plan.peer_fetches.size(), 1u);
    ASSERT_EQ(sink->prefix_loads.size(), 1u);
    const auto& load = sink->prefix_loads.front();
    EXPECT_EQ(load.request_id, "r");
    EXPECT_EQ(load.telemetry.admission_id, 22u);
    EXPECT_EQ(load.telemetry.request_pool_index, sink->slots.back().slot);
    EXPECT_EQ(load.telemetry.slot_generation, sink->slots.back().generation);
    EXPECT_EQ(load.telemetry.slot_generation, 2u);
    EXPECT_EQ(load.telemetry.op_generation, 0u);
    EXPECT_FALSE(load.transfers.empty());
}

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

// A decode replica of a disaggregated deployment as the holder
// (docs/rfcs/0005-runtime-pd-roles.md): it holds the history it decoded, the
// prompt its PD peer sent and the reply it generated, and serves it as any
// holder does, but never reads.
SchedulerConfig DecodePeerConfigFor(const std::string& name) {
    SchedulerConfig cfg = PeerConfigFor(name);
    cfg.role = Role::kD;
    cfg.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
    return cfg;
}

class PeerDecodeHolderTest : public ::testing::Test, protected PeerPair {
protected:
    PeerDecodeHolderTest() : PeerPair(DecodePeerConfigFor("holder"), PeerConfigFor("reader")) {}

    // The holder lands `prompt` from its PD peer with `first` as the peer's
    // first token, decodes `replies` and finishes, publishing the history.
    void HolderDecodes(const std::vector<std::int32_t>& prompt, std::int32_t first,
                       const std::vector<std::int32_t>& replies) {
        holder_.SubmitRequests({RequestSpec{.request_id = "d", .tokens = prompt, .max_new_tokens = 8}});
        Send(holder_, pd::BootstrappedEvent{"d"});
        ASSERT_TRUE(Plan(holder_).remote_prefill.has_value());
        Send(holder_, pd::RemotePrefillDoneEvent{"d", first});
        for (const std::int32_t token : replies) {
            ASSERT_NE(FindForwardBatch(Plan(holder_)), nullptr);
            Send(holder_, forward::ExtendResult{.request_id = "d", .tokens = {token}});
        }
        Send(holder_, forward::Finish{.request_id = "d"});
        Plan(holder_);
    }
};

TEST_F(PeerDecodeHolderTest, TheNextTurnReadsThePreviousReplyFromTheDecodeReplica) {
    // The peer's first token 801, then 802 and 803 decoded here: 801 and 802
    // were fed back, so the history the replica computed is the prompt and
    // those two.
    const std::vector<std::int32_t> prompt = MakeAlignedTokens(8, kP);
    const auto history = static_cast<std::int32_t>(prompt.size()) + 2;
    HolderDecodes(prompt, /*first=*/801, /*replies=*/{802, 803});
    std::vector<std::int32_t> next_turn = prompt;
    next_turn.insert(next_turn.end(), {801, 802, 803, 901, 902, 903});
    reader_.SubmitRequests({RequestSpec{.request_id = "r", .peer_holder = "holder", .tokens = next_turn}});

    EXPECT_TRUE(PrefillRows(Plan(reader_)).empty());
    RelayLookups();
    const ExecutionPlan landing_plan = Plan(reader_);
    ASSERT_EQ(landing_plan.peer_fetches.size(), 1u);
    const PeerFetchOperation fetch = landing_plan.peer_fetches.front();
    EXPECT_EQ(fetch.begin_tokens, 0);
    EXPECT_EQ(fetch.end_tokens, history) << "the prompt and the reply the replica computed";
    EXPECT_EQ(Fetch(fetch, /*read_succeeds=*/true, /*head_between=*/true), history);
    EXPECT_EQ(PrefillRows(Plan(reader_)), (std::vector<std::pair<std::string, std::int32_t>>{{"r", history}}));
    EXPECT_EQ(holder_.GetPeerStats().leases, 0u);
}

TEST_F(PeerDecodeHolderTest, ADecodeReplicaNeverReads) {
    holder_.SubmitRequests({RequestSpec{
        .request_id = "d", .peer_holder = "reader", .tokens = MakeAlignedTokens(8, kP), .max_new_tokens = 4}});
    Send(holder_, pd::BootstrappedEvent{"d"});
    EXPECT_TRUE(Plan(holder_).remote_prefill.has_value()) << "its PD peer sends the whole prompt";
    EXPECT_TRUE(holder_.TakePeerLookups().empty());
}

// A reader that lands into Host (PeerLandingTier::kHost).
SchedulerConfig HostLandingConfigFor(const std::string& name, bool hybrid) {
    SchedulerConfig cfg = PeerConfigFor(name, hybrid);
    cfg.peer_config->landing_tier = PeerLandingTier::kHost;
    cfg.peer_config->max_host_landing_blocks = 24;
    cfg.peer_config->host_ready_timeout_us = 1000;
    return cfg;
}

class PeerHostLandingTest : public ::testing::TestWithParam<bool>, protected PeerPair {
protected:
    PeerHostLandingTest() : PeerPair(PeerConfigFor("holder", GetParam()), HostLandingConfigFor("reader", GetParam())) {}

    void HolderComputes(const std::vector<std::int32_t>& tokens) {
        holder_.SubmitRequests({RequestSpec{.request_id = "h", .tokens = tokens, .max_new_tokens = 1}});
        EXPECT_EQ(PrefillRows(Plan(holder_)).size(), 1u);
        Send(holder_, forward::ExtendResult{.request_id = "h", .tokens = {7}});
        Send(holder_, forward::Finish{.request_id = "h"});
        Plan(holder_);
    }

    // The reader asks for `longer`, whose prefix the holder computed, and
    // returns its landing.
    PeerFetchOperation StartLanding(const std::vector<std::int32_t>& longer) {
        reader_.SubmitRequests({RequestSpec{.request_id = "r", .peer_holder = "holder", .tokens = longer}});
        EXPECT_TRUE(PrefillRows(Plan(reader_)).empty());
        RelayLookups();
        const ExecutionPlan plan = Plan(reader_);
        EXPECT_TRUE(PrefillRows(plan).empty()) << "the prompt waits for its landing before its admission";
        EXPECT_EQ(plan.peer_fetches.size(), 1u);
        return plan.peer_fetches.empty() ? PeerFetchOperation{} : plan.peer_fetches.front();
    }
};

INSTANTIATE_TEST_SUITE_P(Groups, PeerHostLandingTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "Hybrid" : "Full"; });

TEST_P(PeerHostLandingTest, ALandingIntoHostIsAdmittedAsAHostHit) {
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(8, kP);
    HolderComputes(tokens);
    std::vector<std::int32_t> longer = tokens;
    longer.insert(longer.end(), {901, 902, 903, 904});
    const std::int32_t free_host = reader_.HostPoolFreeBlocks();
    const PeerFetchOperation fetch = StartLanding(longer);
    EXPECT_EQ(fetch.end_tokens, static_cast<std::int32_t>(tokens.size()));
    EXPECT_TRUE(
        std::ranges::all_of(fetch.destinations, [](const PeerPage& page) { return page.tier == CacheTier::kHost; }));
    const auto destinations_of = [&](std::uint32_t group) {
        return std::ranges::count_if(fetch.destinations,
                                     [group](const PeerPage& page) { return page.group_id == group; });
    };
    EXPECT_EQ(destinations_of(0), static_cast<std::ptrdiff_t>(tokens.size() / kP)) << "every full page";
    if (GetParam()) {
        EXPECT_EQ(destinations_of(1), 2) << "the window behind the landing's end";
        EXPECT_TRUE(std::ranges::any_of(fetch.destinations, [](const PeerPage& page) {
            return page.group_id == 2 && page.slot == kPeerStateSlot;
        })) << "the State checkpoint fills the live slot";
    }
    EXPECT_LT(reader_.HostPoolFreeBlocks(), free_host) << "the destinations are reserved";
    EXPECT_EQ(reader_.GetPeerStats().landing_requests, 1u);
    EXPECT_TRUE(reader_.StoragePending()) << "a clear waits for the landing";
    EXPECT_TRUE(PrefillRows(Plan(reader_)).empty()) << "still landing";

    EXPECT_EQ(Fetch(fetch, /*read_succeeds=*/true), fetch.end_tokens);
    EXPECT_FALSE(reader_.StorageProgressPending()) << "landed: held for the admission";
    const ExecutionPlan admitted = Plan(reader_);
    EXPECT_EQ(PrefillRows(admitted), (std::vector<std::pair<std::string, std::int32_t>>{{"r", fetch.end_tokens}}))
        << "the landed prefix is a Host hit";
    EXPECT_FALSE(CacheOpIds<LoadBackBatch>(admitted).empty()) << "loaded to the Device";
    EXPECT_EQ(reader_.GetPeerStats().landed_tokens, static_cast<std::uint64_t>(fetch.end_tokens));
    EXPECT_EQ(holder_.GetPeerStats().leases, 0u);
}

TEST_P(PeerHostLandingTest, AFailedLandingReturnsItsHostBlocksAndIsAdmittedLocally) {
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(8, kP);
    HolderComputes(tokens);
    std::vector<std::int32_t> longer = tokens;
    longer.insert(longer.end(), {901, 902});
    const std::int32_t free_host = reader_.HostPoolFreeBlocks();
    const PeerFetchOperation fetch = StartLanding(longer);
    EXPECT_EQ(Fetch(fetch, /*read_succeeds=*/false), fetch.begin_tokens);
    EXPECT_EQ(reader_.HostPoolFreeBlocks(), free_host);
    EXPECT_EQ(PrefillRows(Plan(reader_)), (std::vector<std::pair<std::string, std::int32_t>>{{"r", 0}}));
    EXPECT_EQ(reader_.GetPeerStats().failed_landings, 1u);
}

TEST_P(PeerHostLandingTest, AnAbortWhileLandingKeepsTheBlocksUntilTheLandingEnds) {
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(8, kP);
    HolderComputes(tokens);
    std::vector<std::int32_t> longer = tokens;
    longer.insert(longer.end(), {901, 902});
    const std::int32_t free_host = reader_.HostPoolFreeBlocks();
    const PeerFetchOperation fetch = StartLanding(longer);
    Send(reader_, forward::Abort{.request_id = "r"});
    Plan(reader_);
    EXPECT_LT(reader_.HostPoolFreeBlocks(), free_host) << "the reads may still write its blocks";
    EXPECT_TRUE(reader_.StoragePending());
    Fetch(fetch, /*read_succeeds=*/true);
    Plan(reader_);
    EXPECT_FALSE(reader_.StoragePending());
    EXPECT_EQ(reader_.GetPeerStats().landing_requests, 0u);
}

TEST_P(PeerHostLandingTest, ALandedPrefixNotAdmittedInTimeIsNoLongerHeld) {
    const std::vector<std::int32_t> tokens = MakeAlignedTokens(8, kP);
    HolderComputes(tokens);
    std::vector<std::int32_t> longer = tokens;
    longer.insert(longer.end(), {901, 902});
    const PeerFetchOperation fetch = StartLanding(longer);
    Fetch(fetch, /*read_succeeds=*/true);
    // The Ready hold keeps the peer clock: past it, the entries stay in Host
    // as any other, and the admission still finds them.
    Send(reader_, cache::PeerTick{.now_us = 5000});
    EXPECT_EQ(PrefillRows(Plan(reader_)), (std::vector<std::pair<std::string, std::int32_t>>{{"r", fetch.end_tokens}}));
}

TEST(PeerSchedulerConfigTest, HostLandingsNeedAHostPoolAndCredits) {
    EXPECT_NO_THROW(HostLandingConfigFor("ok", /*hybrid=*/false).Validate());
    SchedulerConfig no_credits = HostLandingConfigFor("c", false);
    no_credits.peer_config->max_host_landing_blocks = 0;
    EXPECT_THROW(no_credits.Validate(), std::invalid_argument);
    SchedulerConfig device_with_credits = PeerConfigFor("d");
    device_with_credits.peer_config->max_host_landing_blocks = 4;
    EXPECT_THROW(device_with_credits.Validate(), std::invalid_argument) << "Device landings use no Host credits";
    SchedulerConfig too_many = HostLandingConfigFor("t", false);
    too_many.peer_config->max_host_landing_blocks = 32;
    EXPECT_THROW(too_many.Validate(), std::invalid_argument) << "more credits than Host blocks";
}

TEST(PeerSchedulerConfigTest, PeerSharingNeedsPrefixCachingAndNoReplayableGroup) {
    EXPECT_NO_THROW(DecodePeerConfigFor("d").Validate());
    SchedulerConfig uncached = PeerConfigFor("u");
    uncached.disable_prefix_cache = true;
    EXPECT_THROW(uncached.Validate(), std::invalid_argument);
    SchedulerConfig replayable = PeerConfigFor("r");
    replayable.cache_groups.front().replayable = true;
    EXPECT_THROW(replayable.Validate(), std::invalid_argument);
    EXPECT_NO_THROW(PeerConfigFor("ok").Validate());
}

TEST(PeerLifecycleAccountingTest, PeerLookingAndPeerLandingCountInSizesAndCanAbort) {
    Scheduler holder{PeerConfigFor("holder")};
    Scheduler reader{PeerConfigFor("reader")};
    const std::vector<std::int32_t> prefix = MakeAlignedTokens(6, kP);
    holder.SubmitRequests({RequestSpec{.request_id = "seed", .tokens = prefix, .max_new_tokens = 1}});
    ASSERT_EQ(PrefillRows(Plan(holder)).size(), 1u);
    Send(holder, forward::ExtendResult{.request_id = "seed", .tokens = {7}});
    Send(holder, forward::Finish{.request_id = "seed"});
    Plan(holder);

    std::vector<std::int32_t> prompt = prefix;
    prompt.insert(prompt.end(), {901, 902, 903, 904});
    reader.SubmitRequests({RequestSpec{.request_id = "r1", .peer_holder = "holder", .tokens = prompt},
                           RequestSpec{.request_id = "r2", .peer_holder = "holder", .tokens = prompt}});
    EXPECT_TRUE(PrefillRows(Plan(reader)).empty());
    EXPECT_EQ(reader.WaitingSize(), 2u);
    EXPECT_EQ(reader.PendingPrefillTokens(), 2 * static_cast<std::int64_t>(prompt.size()));
    EXPECT_TRUE(reader.CanAbortBeforeAdmission("r2"));
    Send(reader, forward::Abort{.request_id = "r2"});
    EXPECT_EQ(reader.WaitingSize(), 1u);

    HolderServicer servicer{holder};
    for (const PeerLookupRequest& lookup : reader.TakePeerLookups()) {
        if (lookup.request_id == "r1") {
            const PeerAnswer ans = servicer.Lookup(lookup);
            Send(reader, cache::PeerLookupAnswered{.lease = ans.lease, .num_tokens = ans.num_tokens});
        }
    }
    const ExecutionPlan landing = Plan(reader);
    ASSERT_EQ(landing.peer_fetches.size(), 1u);
    EXPECT_EQ(reader.WaitingSize(), 0u);
    EXPECT_EQ(reader.PrefillSize(), 1u);
    EXPECT_FALSE(reader.CanAbortBeforeAdmission("r1"));
    EXPECT_EQ(reader.PendingPrefillTokens(),
              static_cast<std::int64_t>(prompt.size()) - landing.peer_fetches.front().end_tokens);
}

TEST(PeerEndpointNormalizationTest, Dp0AliasesNormalizeForSelfAndCrossRankHints) {
    EXPECT_EQ(CanonicalPeerEndpoint("10.0.0.1:7100/0"), "10.0.0.1:7100");
    EXPECT_EQ(CanonicalPeerEndpoint("10.0.0.1:7100/00"), "10.0.0.1:7100");
    EXPECT_EQ(CanonicalPeerEndpoint("10.0.0.1:7100/+0"), "10.0.0.1:7100");
    EXPECT_EQ(CanonicalPeerEndpoint("[::1]:7100/00"), "[::1]:7100");
    EXPECT_EQ(CanonicalPeerEndpoint("10.0.0.1:7100/1"), "10.0.0.1:7100/1");

    Scheduler holder{PeerConfigFor("10.0.0.1:8680")};
    Scheduler reader{PeerConfigFor("10.0.0.1:8680/1")};
    const std::vector<std::int32_t> prefix = MakeAlignedTokens(6, kP);
    holder.SubmitRequests({RequestSpec{.request_id = "seed", .tokens = prefix, .max_new_tokens = 1}});
    ASSERT_EQ(PrefillRows(Plan(holder)).size(), 1u);
    Send(holder, forward::ExtendResult{.request_id = "seed", .tokens = {7}});
    Send(holder, forward::Finish{.request_id = "seed"});
    Plan(holder);

    std::vector<std::int32_t> prompt = prefix;
    prompt.insert(prompt.end(), {801, 802});
    holder.SubmitRequests({RequestSpec{.request_id = "self0", .peer_holder = "10.0.0.1:8680/0", .tokens = prompt}});
    EXPECT_EQ(PrefillRows(Plan(holder)).size(), 1u);
    EXPECT_TRUE(holder.TakePeerLookups().empty());

    reader.SubmitRequests({RequestSpec{.request_id = "cross0", .peer_holder = "10.0.0.1:8680/00", .tokens = prompt}});
    EXPECT_TRUE(PrefillRows(Plan(reader)).empty());
    const auto lookups = reader.TakePeerLookups();
    ASSERT_EQ(lookups.size(), 1u);
    EXPECT_EQ(lookups[0].holder, "10.0.0.1:8680");
}

namespace {
std::vector<DiskOperation> ExtractDiskOps(const ExecutionPlan& plan) {
    std::vector<DiskOperation> ops;
    for (const Operation& op : plan.Operations()) {
        if (const auto* c = std::get_if<CacheOperation>(&op)) {
            if (const auto* d = std::get_if<DiskOperation>(c)) ops.push_back(*d);
        }
    }
    return ops;
}

void WarmOnHolder(Scheduler& holder, const std::vector<std::int32_t>& tokens) {
    holder.SubmitRequests({RequestSpec{.request_id = "h-warm", .tokens = tokens, .max_new_tokens = 1}});
    const ExecutionPlan p1 = Plan(holder);
    Send(holder, forward::ExtendResult{.request_id = "h-warm", .tokens = {7}});
    Send(holder, forward::Finish{.request_id = "h-warm"});
    const ExecutionPlan p2 = Plan(holder);
    for (const auto wb : CacheOpIds<WriteBackBatch>(p1)) Send(holder, cache::WriteBackDone{.op_id = wb});
    for (const auto wb : CacheOpIds<WriteBackBatch>(p2)) Send(holder, cache::WriteBackDone{.op_id = wb});
}

void SeedAndCleanToDisk(Scheduler& reader,
                        const std::vector<std::pair<std::string, std::vector<std::int32_t>>>& seeds) {
    for (const auto& [id, toks] : seeds) {
        reader.SubmitRequests({RequestSpec{.request_id = id, .tokens = toks, .max_new_tokens = 1}});
        const ExecutionPlan p1 = Plan(reader);
        ASSERT_EQ(PrefillRows(p1).size(), 1u);
        Send(reader, forward::ExtendResult{.request_id = id, .tokens = {7}});
        Send(reader, forward::Finish{.request_id = id});
        const ExecutionPlan p2 = Plan(reader);
        for (const auto wb : CacheOpIds<WriteBackBatch>(p1)) Send(reader, cache::WriteBackDone{.op_id = wb});
        for (const auto wb : CacheOpIds<WriteBackBatch>(p2)) Send(reader, cache::WriteBackDone{.op_id = wb});
    }
    for (int i = 0; i < 3; ++i) {
        const std::string pid = "press-" + std::to_string(i);
        reader.SubmitRequests(
            {RequestSpec{.request_id = pid, .tokens = MakeAlignedTokens(7, kP, 2000 + i * 100), .max_new_tokens = 1}});
        const ExecutionPlan p1 = Plan(reader);
        for (const auto& w : ExtractDiskOps(p1)) Send(reader, cache::DiskDone{.op_id = w.op_id, .success = true});
        Send(reader, forward::ExtendResult{.request_id = pid, .tokens = {7}});
        Send(reader, forward::Finish{.request_id = pid});
        const ExecutionPlan p2 = Plan(reader);
        for (const auto wb : CacheOpIds<WriteBackBatch>(p1)) Send(reader, cache::WriteBackDone{.op_id = wb});
        for (const auto wb : CacheOpIds<WriteBackBatch>(p2)) Send(reader, cache::WriteBackDone{.op_id = wb});
        for (const auto& w : ExtractDiskOps(p2)) Send(reader, cache::DiskDone{.op_id = w.op_id, .success = true});
    }
    while (reader.GetDiskStats().queued_cleanings > 0) {
        for (const auto& w : ExtractDiskOps(Plan(reader)))
            Send(reader, cache::DiskDone{.op_id = w.op_id, .success = true});
    }
}

void EvictHostTokens(Scheduler& reader, const std::vector<std::int32_t>& toks) {
    CacheCoordinator& coord = SchedulerTestAccess::Coordinator(reader);
    std::vector<std::span<const std::int32_t>> pages;
    for (std::size_t off = 0; off < toks.size(); off += kP) pages.emplace_back(toks.data() + off, kP);
    for (const CacheKey& key : coord.Store().KeysForGroup(ComputePrefixHashes(pages, ""), 0)) {
        if (const auto loc = coord.Store().LocationOf(*coord.Store().HostPool(), 0, key)) {
            ASSERT_TRUE(coord.Evictor().EvictHostCachedBlock(0, *loc, &CacheEvictor::HostEvictionStats::lru));
        }
    }
}
}  // namespace

TEST(PeerDiskInterleavingTest, DiskCreditsAndLookingReadyHoldCoordinateCleanlyWithPeerLanding) {
    Scheduler holder{PeerConfigFor("holder")};
    const std::vector<std::int32_t> full_prefix = MakeAlignedTokens(8, kP);
    WarmOnHolder(holder, full_prefix);

    SchedulerConfig cfg = HostLandingConfigFor("reader", /*hybrid=*/false);
    cfg.device_allocator.total_pages = 17;
    cfg.cache_groups.front().total_pages = 17;
    cfg.host_allocator.total_pages = 33;
    cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
    cfg.disk_allocator.total_pages = 33;
    cfg.disk_config = DiskConfig{.max_operations = 4,
                                 .max_blocks_per_operation = 32,
                                 .max_host_blocks = 4,
                                 .max_write_host_blocks = 4,
                                 .max_queued_cleanings = 32,
                                 .min_prefetch_tokens = kP,
                                 .wait_timeout_us = 10'000,
                                 .ready_timeout_us = 50};
    cfg.max_batch_size = 1;
    cfg.max_resident_requests = 1;
    SetTestSnapshotPool(cfg);
    Scheduler reader{cfg};

    Send(reader, cache::StorageTick{.now_us = 1, .allow_forward = true, .allow_storage = true});
    const std::vector<std::int32_t> short_prefix(full_prefix.begin(), full_prefix.begin() + 2 * kP);
    const std::vector<std::int32_t> other_disk = MakeAlignedTokens(4, kP, 5000);
    SeedAndCleanToDisk(reader, {{"s1", short_prefix}, {"s2", other_disk}});
    ASSERT_TRUE(SchedulerTestAccess::Coordinator(reader).ClearDeviceCache());
    EvictHostTokens(reader, short_prefix);
    EvictHostTokens(reader, other_disk);

    // Occupy the 1 resident slot and exhaust all 4 Disk read credits with `disk-hog`.
    reader.SubmitRequests(
        {RequestSpec{.request_id = "blk", .tokens = MakeAlignedTokens(2, kP, 8000), .max_new_tokens = 4}});
    ASSERT_EQ(PrefillRows(Plan(reader)).size(), 1u);
    std::vector<std::int32_t> hog = other_disk;
    hog.insert(hog.end(), {7001, 7002});
    std::vector<std::int32_t> peer_toks = full_prefix;
    peer_toks.insert(peer_toks.end(), {9001, 9002});
    reader.SubmitRequests(
        {RequestSpec{.request_id = "hog", .tokens = hog, .max_new_tokens = 1},
         RequestSpec{.request_id = "r-peer", .peer_holder = "holder", .tokens = peer_toks, .max_new_tokens = 1}});

    const auto reads_a = ExtractDiskOps(Plan(reader));
    ASSERT_EQ(reads_a.size(), 1u);
    EXPECT_TRUE(reader.TakePeerLookups().empty()) << "preparePeer waits while Disk is credit-exhausted";

    // Free credits -> r-peer prefetches [0, 4) from Disk, carries Ready hold into PeerLooking.
    Send(reader, cache::DiskDone{.op_id = reads_a[0].op_id, .success = true});
    Send(reader, forward::Abort{.request_id = "hog"});
    const auto reads_b = ExtractDiskOps(Plan(reader));
    ASSERT_EQ(reads_b.size(), 1u);
    Send(reader, cache::DiskDone{.op_id = reads_b[0].op_id, .success = true});
    Plan(reader);
    EXPECT_EQ(reader.GetDiskStats().ready_requests, 1u);
    EXPECT_EQ(reader.GetDiskStats().held_host_blocks, 2u);
    const auto lookups = reader.TakePeerLookups();
    ASSERT_EQ(lookups.size(), 1u);

    // Holder answers 16 tokens -> startPeerHostLanding clears Disk storage_hold (no dual leases!).
    HolderServicer servicer{holder};
    const PeerAnswer ans = servicer.Lookup(lookups[0]);
    Send(reader, cache::PeerLookupAnswered{.lease = ans.lease, .num_tokens = ans.num_tokens});
    for (int i = 0; i < 4; ++i) Send(reader, forward::ExtendResult{.request_id = "blk", .tokens = {7}});
    Send(reader, forward::Finish{.request_id = "blk"});

    const ExecutionPlan p_land = Plan(reader);
    EXPECT_TRUE(ExtractDiskOps(p_land).empty());
    ASSERT_EQ(p_land.peer_fetches.size(), 1u);
    EXPECT_EQ(p_land.peer_fetches[0].begin_tokens, 4);
    EXPECT_EQ(p_land.peer_fetches[0].end_tokens, 16);
    EXPECT_EQ(reader.GetDiskStats().held_host_blocks, 0u);

    // PeerFetchDone Ready hold allows CanPrepareStorageClear() -> ClearCache().
    Send(reader, cache::PeerFetchDone{.request_id = "r-peer", .num_tokens = 16});
    ASSERT_TRUE(reader.StoragePending());
    EXPECT_TRUE(reader.CanPrepareStorageClear());
    reader.CancelStorageForClear();
    EXPECT_TRUE(reader.ClearCache());
}

TEST(PeerSchedulerRound7Test, ExpiredSoftPinMaxLeasesOverflowAndFusedAbortSafety) {
    // 1. max_leases overflow & expired soft pin never phantom-pin or vouch across ClearCache.
    SchedulerConfig cfg = PeerConfigFor("holder");
    cfg.peer_config->max_leases = 2;
    cfg.peer_config->soft_pin_us = 10;
    cfg.peer_config->hard_pin_us = 100;
    Scheduler holder{cfg};
    HolderServicer servicer{holder};

    const std::vector<std::int32_t> prefix = MakeAlignedTokens(4, kP, 100);
    std::vector<std::int32_t> warm = prefix;
    warm.push_back(999);
    WarmOnHolder(holder, warm);
    const auto hashes = holder.PrefixHashesForTokens(warm);
    const auto mk = [&](std::uint64_t id) {
        return PeerLookupRequest{.request_id = "r" + std::to_string(id),
                                 .deadline_us = 1000,
                                 .lease = PeerLeaseId{.reader = "reader", .id = id},
                                 .holder = "holder",
                                 .prefix_hashes = hashes,
                                 .tokens = prefix,
                                 .max_tokens = 8};
    };
    const PeerAnswer a1 = servicer.Lookup(mk(1));
    EXPECT_EQ(a1.num_tokens, 8);
    EXPECT_EQ(servicer.Lookup(mk(2)).num_tokens, 8);
    EXPECT_EQ(servicer.Lookup(mk(3)).num_tokens, 0) << "3rd unsynced lookup past max_leases=2";
    servicer.Head(5);
    EXPECT_EQ(servicer.Lookup(mk(4)).num_tokens, 0) << "synced max_leases=2 overflow";
    servicer.Head(20);  // past soft_pin_us = 10
    EXPECT_FALSE(servicer.Pin(a1.lease).held) << "expired soft pin must not return held=true";

    const PeerAnswer a5 = servicer.Lookup(mk(5));
    ASSERT_TRUE(servicer.Pin(a5.lease).held);
    const std::int64_t before = holder.PlanCalls();
    ASSERT_TRUE(holder.ClearL1Cache());
    ASSERT_TRUE(holder.ClearCache());
    EXPECT_GT(holder.PlanCalls(), before);
    EXPECT_FALSE(servicer.Release(a5.lease, true).held);

    // 2. Aborting PeerLanding on Role::kFused keeps PdTransferInFlight until PeerFetchDone.
    Scheduler h2{PeerConfigFor("holder")};
    WarmOnHolder(h2, warm);
    Scheduler reader{PeerConfigFor("reader")};
    std::vector<std::int32_t> prompt = prefix;
    prompt.push_back(502);
    reader.SubmitRequests(
        {RequestSpec{.request_id = "ab", .peer_holder = "holder/00", .tokens = prompt, .max_new_tokens = 1}});
    Plan(reader);
    const auto lookups = reader.TakePeerLookups();
    ASSERT_EQ(lookups.size(), 1u);
    HolderServicer s2{h2};
    const PeerAnswer ans = s2.Lookup(lookups[0]);
    Send(reader, cache::PeerLookupAnswered{.lease = ans.lease, .num_tokens = ans.num_tokens});
    ASSERT_EQ(Plan(reader).peer_fetches.size(), 1u);
    Send(reader, forward::Abort{.request_id = "ab"});
    EXPECT_EQ(reader.PdTransferSize(), 1u);
    EXPECT_TRUE(reader.PdTransferPinned("ab"));
    EXPECT_FALSE(reader.ClearL1Cache());
    Send(reader, cache::PeerFetchDone{.request_id = "ab", .num_tokens = 0});
    EXPECT_EQ(reader.PdTransferSize(), 0u);
    EXPECT_TRUE(reader.ClearL1Cache());
}

}  // namespace
}  // namespace tokenspeed::test
