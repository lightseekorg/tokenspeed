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

#include "cache/tier/peer_host_landings.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

#include "utils.h"

namespace tokenspeed {

PeerHostLandings::PeerHostLandings(CacheCoordinator& coordinator, std::int32_t max_host_blocks)
    : coordinator_{coordinator}, max_host_blocks_{max_host_blocks} {
    FatalCheck(coordinator_.Store().HostPool() != nullptr, "peer Host landings need a Host pool");
    FatalCheck(max_host_blocks_ > 0, "peer Host landings need positive Host block credits");
}

std::uint64_t PeerHostLandings::nextId() {
    if (next_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("peer Host landing ids exhausted");
    }
    return next_id_++;
}

std::size_t PeerHostLandings::NumHeldHostBlocks() const {
    std::size_t held = 0;
    for (const auto& [id, landing] : landings_) held += landing.host_blocks.size();
    return held;
}

std::optional<PeerHostLandings::Started> PeerHostLandings::Start(const PromptPrefix& prompt,
                                                                 const std::string& request_id,
                                                                 const PeerLeaseId& lease, const std::string& holder,
                                                                 std::int32_t local_tokens, bool* retry) {
    FatalCheck(!by_request_.contains(request_id), "a request lands into Host once at a time");
    const std::size_t held = NumHeldHostBlocks();
    const auto limit = static_cast<std::size_t>(max_host_blocks_);
    if (retry) *retry = false;
    if (held >= limit) {
        if (retry) *retry = true;
        return std::nullopt;
    }
    // The holder answered the boundary whole, so every key it needs stands in
    // as a Host hit: the reservation then gives the keys the Host lacks above
    // the Device's hit blocks, under the same Host-suffix rule as admission.
    const StorageQuery boundary =
        coordinator_.Storage().SharedPrefixKeys(prompt, std::numeric_limits<std::size_t>::max());
    if (boundary.max_tokens != prompt.max_tokens) return std::nullopt;
    StorageReservation reservation = coordinator_.Storage().ReserveStoragePrefix(prompt, boundary.keys, limit - held);
    if (reservation.keys.empty() || reservation.num_tokens != prompt.max_tokens) {
        if (retry) *retry = true;
        return std::nullopt;
    }

    PeerFetchOperation fetch{.request_id = request_id,
                             .lease = lease,
                             .holder = holder,
                             .begin_tokens = local_tokens,
                             .end_tokens = prompt.max_tokens};
    fetch.destinations.reserve(reservation.keys.size());
    for (std::size_t i = 0; i < reservation.keys.size(); ++i) {
        const std::uint32_t group = reservation.keys[i].group_id;
        fetch.destinations.push_back(PeerPage{
            .group_id = group,
            // A State checkpoint fills the reader's live slot, as the holder
            // lists it.
            .slot = coordinator_.GroupIsState(group) ? kPeerStateSlot : reservation.logical_blocks[i],
            .tier = CacheTier::kHost,
            .block_id =
                coordinator_.Allocator(static_cast<std::int32_t>(group)).ResolveCacheBlockId(*reservation.blocks[i]),
        });
    }
    const std::uint64_t id = nextId();
    landings_.emplace(id, Landing{.request_id = request_id,
                                  .begin_tokens = local_tokens,
                                  .end_tokens = prompt.max_tokens,
                                  .keys = std::move(reservation.keys),
                                  .host_blocks = std::move(reservation.blocks),
                                  .tail_prior = std::move(reservation.tail_prior),
                                  .tail = std::move(reservation.tail)});
    by_request_.emplace(request_id, id);
    return Started{.lease = StorageLease{*this, id}, .fetch = std::move(fetch)};
}

bool PeerHostLandings::HasProgressPending() const {
    return std::ranges::any_of(landings_, [](const auto& entry) { return !entry.second.completed; });
}

bool PeerHostLandings::Completed(std::uint64_t id) const {
    const auto it = landings_.find(id);
    return it == landings_.end() || it->second.completed;
}

std::optional<std::int32_t> PeerHostLandings::Finish(const std::string& request_id, std::int32_t num_tokens) {
    const auto by = by_request_.find(request_id);
    if (by == by_request_.end()) return std::nullopt;
    const auto it = landings_.find(by->second);
    by_request_.erase(by);
    FatalCheck(it != landings_.end() && !it->second.completed, "a peer Host landing finishes once");
    Landing& landing = it->second;
    landing.completed = true;
    const bool landed = num_tokens == landing.end_tokens;
    const std::int32_t landed_tokens = landed ? landing.end_tokens - landing.begin_tokens : 0;
    if (landed) {
        if (!landing.detached) {
            for (std::size_t i = 0; i < landing.keys.size(); ++i) {
                coordinator_.Host().CacheHostBlock(landing.host_blocks[i], landing.keys[i],
                                                   /*inherited_metadata=*/std::nullopt);
            }
            if (landing.tail) coordinator_.Host().CacheHostTail(landing.tail_prior, *landing.tail);
        }
    } else {
        spdlog::debug("[Peer] rid={} Host landing confirmed {} of {} tokens; its blocks go back", request_id,
                      num_tokens, landing.end_tokens);
        landing.host_blocks.clear();
    }
    // A request still holds the published entries until its admission takes
    // them (the Ready hold); one that let go is done with them.
    if (landing.detached) landings_.erase(it);
    return landed_tokens;
}

void PeerHostLandings::detach(std::uint64_t id) noexcept {
    const auto it = landings_.find(id);
    FatalCheck(it != landings_.end() && !it->second.detached, "a peer Host landing lease names an attached landing");
    it->second.detached = true;
    if (it->second.completed) landings_.erase(it);
}

}  // namespace tokenspeed
