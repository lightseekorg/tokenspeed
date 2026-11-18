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

// Peer prefix sharing (docs/rfcs/0004-peer-prefix-sharing.md): the holder
// answers lookups, pins and releases from its own cache; the reader asks
// once per prompt, admits a landing for what its holder can add, and either
// keeps the landed prefix or is admitted again from its local hit.

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

#include "core/profile_span.h"
#include "fsm/peer_events.h"
#include "scheduler/scheduler.h"

namespace tokenspeed {

bool Scheduler::PeerPagesMatch(const cache::PeerLookup& lookup, std::int32_t num_tokens,
                               std::span<const PeerPage> pages) const {
    if (!PeerExports::Matches(coordinator_, lookup, num_tokens, pages)) {
        return false;
    }
    if (!tier_transfers_.HasLoadBacksInFlight()) {
        return true;
    }
    // Local forwards wait on layerwise H2D fences; a remote reader cannot.
    // A PIN may export a load's Device destinations only after every rank
    // acknowledged it. Other Device pages and Host sources remain readable.
    const auto loading = tier_transfers_.LoadBackDestinations();
    return std::ranges::none_of(pages, [&](const PeerPage& page) {
        return page.tier == CacheTier::kDevice && loading[page.group_id].contains(page.block_id);
    });
}
namespace {

std::int64_t Deadline(std::int64_t now_us, std::int64_t lifetime_us) {
    return now_us + std::min(lifetime_us, std::numeric_limits<std::int64_t>::max() - now_us);
}

}  // namespace

Scheduler::PeerStats Scheduler::GetPeerStats() const {
    PeerStats stats{.landed_tokens = peer_landed_tokens_, .failed_landings = peer_failed_landings_};
    if (peer_exports_) {
        stats.leases = peer_exports_->NumLeases();
        stats.pinned_blocks = peer_exports_->NumPinnedBlocks();
    }
    for (const auto& request : requests_) {
        stats.looking_requests += request->Is<fsm::PeerLooking>();
        stats.landing_requests += request->Is<fsm::PeerLanding>();
    }
    return stats;
}

void Scheduler::handleEvent(const cache::PeerTick& event) {
    if (!peer_exports_) throw std::invalid_argument("a peer tick without configured peer sharing");
    if (event.now_us < peer_now_us_) throw std::invalid_argument("the peer clock moved backwards");
    peer_now_us_ = event.now_us;
    peer_exports_->Expire(peer_now_us_);
    for (const auto& request : requests_) {
        if (const auto* looking = request->GetIf<fsm::PeerLooking>();
            looking != nullptr && peer_now_us_ >= looking->deadline_us) {
            spdlog::debug("[Peer] rid={} lookup {} unanswered; admitted from the local hit", request->Id(),
                          looking->lease.id);
            request->Apply(fsm::FinishPeerLookupEvent{std::nullopt});
        }
    }
}

void Scheduler::handleEvent(const cache::PeerLookup& event) {
    if (!peer_exports_) throw std::invalid_argument("a peer lookup without configured peer sharing");
    peer_exports_->Lookup(coordinator_, event, peer_now_us_);
}

void Scheduler::handleEvent(const cache::PeerPin& event) {
    if (!peer_exports_) throw std::invalid_argument("a peer pin without configured peer sharing");
    peer_exports_->Pin(coordinator_, event, peer_now_us_);
}

void Scheduler::handleEvent(const cache::PeerRelease& event) {
    if (!peer_exports_) throw std::invalid_argument("a peer release without configured peer sharing");
    peer_exports_->Release(coordinator_, event);
}

void Scheduler::handleEvent(const cache::PeerLookupAnswered& event) {
    for (const auto& request : requests_) {
        const auto* looking = request->GetIf<fsm::PeerLooking>();
        if (looking == nullptr || looking->lease != event.lease) {
            continue;
        }
        request->Apply(fsm::FinishPeerLookupEvent{
            event.num_tokens > 0 ? std::optional{fsm::PeerHit{.lease = event.lease, .num_tokens = event.num_tokens}}
                                 : std::nullopt});
        return;
    }
    // The request gave the wait up or ended: the soft pin expires at the holder.
}

void Scheduler::handleEvent(const cache::PeerFetchDone& event) {
    Request* request = findRequest(event.request_id);
    if (request == nullptr) {
        return;
    }
    if (request->Is<fsm::PdTransferring>()) {
        // It ended while landing: the reads have stopped, its pages go back.
        handleEvent(pd::TransferStoppedEvent{event.request_id});
        return;
    }
    const auto* landing = request->GetIf<fsm::PeerLanding>();
    if (landing == nullptr) {
        throw std::logic_error("PeerFetchDone for a request that is not landing; state=" + request->StateName());
    }
    const std::int32_t end = landing->window.begin + landing->window.size;
    if (event.num_tokens == end) {
        peer_landed_tokens_ += static_cast<std::uint64_t>(landing->window.size);
        request->Apply(fsm::PeerLandedEvent{});
        return;
    }
    spdlog::debug("[Peer] rid={} landing confirmed {} of [{}, {}); admitted again from the local hit", request->Id(),
                  event.num_tokens, landing->window.begin, end);
    ++peer_failed_landings_;
    request->Apply(fsm::PeerLandingFailedEvent{&coordinator_});
}

std::vector<PeerPage> Scheduler::peerLandingDestinations(std::span<const BlockTable> tables, std::int32_t hit_tokens,
                                                         std::int32_t end_tokens) const {
    std::vector<PeerPage> destinations;
    for (std::int32_t i = 0; i < coordinator_.NumGroups(); ++i) {
        const BlockTable& table = tables[static_cast<std::size_t>(i)];
        const std::span<const CacheBlockRef> blocks = table.Blocks();
        const auto add = [&](std::int32_t slot) {
            destinations.push_back(PeerPage{
                .group_id = static_cast<std::uint32_t>(i),
                .slot = slot,
                .tier = CacheTier::kDevice,
                .block_id = coordinator_.Allocator(i).ResolveCacheBlockId(*blocks[static_cast<std::size_t>(slot)]),
            });
        };
        if (coordinator_.GroupIsState(static_cast<std::size_t>(i))) {
            add(kPeerStateSlot);
            continue;
        }
        const std::int32_t block_granularity = coordinator_.GroupBlockGranularity(i);
        const std::int32_t past_end = std::min(CeilDiv(end_tokens, block_granularity), table.NumBlocks());
        for (std::int32_t slot = hit_tokens / block_granularity; slot < past_end; ++slot) {
            if (blocks[static_cast<std::size_t>(slot)]) {
                add(slot);
            }
        }
    }
    return destinations;
}

void Scheduler::excludeLandedPages(AdmissionResult& admission, std::span<const PeerPage> destinations) const {
    const auto landed = [&](std::size_t group, std::int32_t block_id) {
        return std::ranges::any_of(
            destinations, [&](const PeerPage& page) { return page.group_id == group && page.block_id == block_id; });
    };
    for (std::size_t i = 0; i < admission.new_page_ids.size(); ++i) {
        std::erase_if(admission.new_page_ids[i], [&](std::int32_t block_id) { return landed(i, block_id); });
        for (const auto& [source, destination] : admission.pages_to_copy[i]) {
            _assert(!landed(i, destination), "a landing's destination is no copy destination");
        }
    }
    for (const BlockTransfer& pair : admission.load_pairs) {
        _assert(!landed(pair.group_id, coordinator_.Allocator(static_cast<std::int32_t>(pair.group_id))
                                           .ResolveCacheBlockId(*pair.destination)),
                "a landing's destination is no load destination");
    }
}

void Scheduler::preparePeer(Request& request) {
    auto* submitted = request.GetIf<fsm::Submitted>();
    if (submitted == nullptr || submitted->admission.peer_prepared) {
        return;
    }
    submitted->admission.peer_prepared = true;
    const PeerConfig& limits = *config_.peer_config;
    const std::string& holder = request.PeerHolder();
    if (holder.empty() || holder == limits.name) {
        return;
    }
    const std::int32_t grain = config_.prefix_granularity;
    const std::int32_t max_tokens = cacheablePrefixTokens(request);
    const std::span<const std::string> hashes = request.PromptPrefixHashes(max_tokens / grain);
    const std::span<const std::int32_t> tokens = request.TokenSlice(0, max_tokens);
    const PrefixProbe probe =
        coordinator_.Prober().ProbePromptPrefix({.prefix_hashes = hashes,
                                                 .tokens = tokens,
                                                 .max_tokens = max_tokens,
                                                 .shared_prefix_tokens = request.SharedPrefixTokens()});
    const std::int32_t local = std::max(probe.device.num_common_tokens, probe.host.num_common_tokens);
    if (max_tokens - local < limits.min_fetch_tokens) {
        return;
    }
    PeerLeaseId lease{.reader = limits.name, .id = ++next_peer_lease_};
    peer_lookups_.push_back(PeerLookupRequest{.lease = lease,
                                              .holder = holder,
                                              .prefix_hashes = {hashes.begin(), hashes.end()},
                                              .tokens = {tokens.begin(), tokens.end()},
                                              .max_tokens = max_tokens});
    spdlog::debug("[Peer] rid={} lookup {} to {} local_tokens={} bound_tokens={}", request.Id(), lease.id, holder,
                  local, max_tokens);
    request.Apply(fsm::StartPeerLookupEvent{std::move(lease), Deadline(peer_now_us_, limits.lookup_timeout_us)});
}

void Scheduler::schedulePeerLandings(AdmissionFeedback& feedback, PlanBuild& build,
                                     std::span<Request* const> candidates, bool new_prompts_sealed) {
    if (!peer_exports_ || new_prompts_sealed) {
        return;
    }
    ProfileSpan profile_span{"Scheduler::schedulePeerLandings"};
    const std::int64_t kv_token_limit =
        config_.max_kv_page_num == 0 ? std::numeric_limits<std::int64_t>::max()
                                     : static_cast<std::int64_t>(build.kv_page_budget) * config_.prefix_granularity;
    for (Request* request : candidates) {
        if (req_pool_allocator_.AvailableSlots() == 0 || lone_requests_.ActedInPlan()) {
            return;
        }
        auto* submitted = request->GetIf<fsm::Submitted>();
        if (submitted == nullptr || !submitted->admission.peer_hit || build.Scheduled(*request)) {
            continue;
        }
        if (!prefillAllowed(*request, build)) {
            return;
        }
        const fsm::PeerHit hit = *submitted->admission.peer_hit;
        feedback.admission_failed = false;
        FirstChunkOutcome outcome = schedulePrefillFirstChunk(
            build.plan, feedback, request, /*remaining=*/std::numeric_limits<std::int32_t>::max(),
            config_.decode_input_tokens, kv_token_limit, /*yield_tokens=*/0, /*must_complete=*/false, hit.num_tokens);
        if (!outcome.event) {
            if (feedback.admission_failed) {
                // It waits for room, keeping its hit; its soft pin may lapse.
                feedback.NoteCapacityBlocked(request);
                return;
            }
            // Nothing worth landing (the local hit grew, or the landing would
            // pass the KV budget): it is admitted locally.
            submitted->admission.peer_hit.reset();
            continue;
        }
        std::vector<BlockTransfer> load_pairs = outcome.event->TakeLoadPairs();
        if (!load_pairs.empty()) {
            build.load_backs.push_back(tier_transfers_.StartPrefixLoad(std::move(load_pairs)));
        }
        request->Apply(std::move(*outcome.event));
        build.scheduled.insert(request);

        const auto& landing = *request->GetIf<fsm::PeerLanding>();
        PeerFetchOperation fetch{.request_id = request->Id(),
                                 .lease = hit.lease,
                                 .holder = request->PeerHolder(),
                                 .begin_tokens = landing.window.begin,
                                 .end_tokens = landing.window.begin + landing.window.size,
                                 .destinations = std::move(outcome.peer_destinations)};
        spdlog::debug("[Peer] rid={} lands [{}, {}) from {} ({} blocks)", request->Id(), fetch.begin_tokens,
                      fetch.end_tokens, fetch.holder, fetch.destinations.size());
        build.plan.peer_fetches.push_back(std::move(fetch));
    }
}

}  // namespace tokenspeed
