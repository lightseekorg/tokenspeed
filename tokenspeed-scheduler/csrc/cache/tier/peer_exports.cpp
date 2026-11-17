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

#include "cache/tier/peer_exports.h"

#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

namespace tokenspeed {
namespace {

std::int64_t Deadline(std::int64_t now_us, std::int64_t lifetime_us) {
    return now_us + std::min(lifetime_us, std::numeric_limits<std::int64_t>::max() - now_us);
}

PromptPrefix Prompt(std::span<const std::string> prefix_hashes, std::span<const std::int32_t> tokens,
                    std::int32_t max_tokens) {
    return PromptPrefix{.prefix_hashes = prefix_hashes, .tokens = tokens, .max_tokens = max_tokens};
}

// A lookup comes off the wire: one whose prompt does not cover its bound is
// answered with a miss rather than trusted.
bool WellFormed(const cache::PeerLookup& lookup, std::int32_t prefix_granularity) {
    return lookup.max_tokens > 0 && static_cast<std::size_t>(lookup.max_tokens) <= lookup.tokens.size() &&
           lookup.prefix_hashes.size() == static_cast<std::size_t>(lookup.max_tokens / prefix_granularity);
}

}  // namespace

void PeerConfig::Validate() const {
    if (self.empty() || max_leases <= 0 || soft_pin_us <= 0 || hard_pin_us <= 0 || lookup_timeout_us <= 0 ||
        min_fetch_tokens <= 0) {
        throw std::invalid_argument("peer sharing needs a name, and positive lease limits, lifetimes and minimum");
    }
}

PeerExports::PeerExports(const PeerConfig& config)
    : max_leases_{config.max_leases}, soft_pin_us_{config.soft_pin_us}, hard_pin_us_{config.hard_pin_us} {}

PeerAnswer PeerExports::Lookup(CacheCoordinator& coordinator, const cache::PeerLookup& lookup, std::int64_t now_us) {
    PeerAnswer answer{.lease = lookup.lease, .kind = PeerAnswerKind::kLookup};
    // A lookup repeated under the same lease starts it over.
    leases_.erase(lookup.lease);
    if (!WellFormed(lookup, coordinator.PrefixGranularity())) {
        spdlog::warn("[Peer] lookup {}#{} does not cover its bound; answered as a miss", lookup.lease.reader,
                     lookup.lease.id);
        return answer;
    }
    if (static_cast<std::int64_t>(leases_.size()) >= max_leases_) {
        return answer;
    }
    CacheCoordinator::PrefixExport exported =
        coordinator.ExportPrefix(coordinator.Prober().ProbePromptPrefix(Prompt(lookup.prefix_hashes, lookup.tokens,
                                                                      lookup.max_tokens)));
    if (exported.num_tokens == 0) {
        return answer;
    }
    const auto boundary = static_cast<std::size_t>(exported.num_tokens);
    const auto full_pages = boundary / static_cast<std::size_t>(coordinator.PrefixGranularity());
    leases_.emplace(lookup.lease,
                    Lease{.deadline_us = Deadline(now_us, soft_pin_us_),
                          .prefix_hashes = {lookup.prefix_hashes.begin(),
                                            lookup.prefix_hashes.begin() + static_cast<std::ptrdiff_t>(full_pages)},
                          .tokens = {lookup.tokens.begin(), lookup.tokens.begin() + static_cast<std::ptrdiff_t>(boundary)},
                          .num_tokens = exported.num_tokens});
    // The exported blocks go here: the lookup only refreshed their entries.
    answer.num_tokens = exported.num_tokens;
    return answer;
}

PeerAnswer PeerExports::Pin(CacheCoordinator& coordinator, const cache::PeerPin& pin, std::int64_t now_us) {
    PeerAnswer answer{.lease = pin.lease, .kind = PeerAnswerKind::kPin};
    const auto it = leases_.find(pin.lease);
    if (it == leases_.end() || it->second.phase != Phase::kSoft) {
        return answer;
    }
    Lease& lease = it->second;
    CacheCoordinator::PrefixExport exported = coordinator.ExportPrefix(
        coordinator.Prober().ProbePromptPrefix(Prompt(lease.prefix_hashes, lease.tokens, lease.num_tokens)));
    if (exported.num_tokens == 0) {
        leases_.erase(it);
        return answer;
    }
    lease.phase = Phase::kHard;
    lease.deadline_us = Deadline(now_us, hard_pin_us_);
    lease.num_tokens = exported.num_tokens;
    lease.access_epoch = exported.access_epoch;
    answer.num_tokens = exported.num_tokens;
    answer.pages.reserve(exported.blocks.size());
    lease.pins.reserve(exported.blocks.size());
    lease.keys.reserve(exported.blocks.size());
    for (CacheCoordinator::ExportedBlock& block : exported.blocks) {
        const auto group = static_cast<std::int32_t>(block.key.group_id);
        answer.pages.push_back(PeerPage{
            .group_id = block.key.group_id,
            .slot = block.slot,
            .tier = block.tier,
            .block_id = coordinator.Allocator(group).ResolveCacheBlockId(*block.block),
        });
        lease.pins.push_back(std::move(block.block));
        lease.keys.push_back(std::move(block.key));
    }
    return answer;
}

PeerAnswer PeerExports::Release(CacheCoordinator& coordinator, const cache::PeerRelease& release) {
    PeerAnswer answer{.lease = release.lease, .kind = PeerAnswerKind::kRelease};
    const auto it = leases_.find(release.lease);
    if (it == leases_.end()) {
        return answer;
    }
    answer.held = it->second.phase == Phase::kHard;
    // The pins go first: a demotion skips an entry anything holds.
    Lease lease = std::move(it->second);
    leases_.erase(it);
    lease.pins.clear();
    if (answer.held && release.demote) {
        coordinator.DemoteExportedChain(lease.keys, lease.access_epoch);
    }
    return answer;
}

void PeerExports::Expire(std::int64_t now_us) {
    std::erase_if(leases_, [&](auto& entry) {
        Lease& lease = entry.second;
        if (now_us < lease.deadline_us) {
            return false;
        }
        if (lease.phase != Phase::kHard) {
            return true;
        }
        lease.phase = Phase::kExpired;
        lease.pins.clear();
        lease.deadline_us = Deadline(now_us, hard_pin_us_);
        return false;
    });
}

std::size_t PeerExports::NumPinnedBlocks() const {
    std::size_t pinned = 0;
    for (const auto& [id, lease] : leases_) {
        pinned += lease.pins.size();
    }
    return pinned;
}

}  // namespace tokenspeed
