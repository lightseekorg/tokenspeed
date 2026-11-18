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
    if (name.empty() || max_leases <= 0 || soft_pin_us <= 0 || hard_pin_us <= 0 || lookup_timeout_us <= 0 ||
        min_fetch_tokens <= 0) {
        throw std::invalid_argument("peer sharing needs a name, and positive lease limits, lifetimes and minimum");
    }
    // Lease ids count up from incarnation << 32: a wider incarnation would lose
    // the bits that tell two runs of the reader apart.
    if (incarnation == 0 || incarnation > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("a peer incarnation is a nonzero 32-bit value");
    }
}

PeerExports::PeerExports(const PeerConfig& config)
    : max_leases_{config.max_leases}, soft_pin_us_{config.soft_pin_us}, hard_pin_us_{config.hard_pin_us} {}

namespace {

// The lookup's prompt up to `num_tokens` (at most its own bound).
PromptPrefix PromptUpTo(const cache::PeerLookup& lookup, std::int32_t num_tokens, std::int32_t prefix_granularity) {
    const auto pages = static_cast<std::size_t>(num_tokens / prefix_granularity);
    return Prompt(std::span{lookup.prefix_hashes}.first(pages),
                  std::span{lookup.tokens}.first(static_cast<std::size_t>(num_tokens)), num_tokens);
}

std::vector<PeerPage> Pages(const PrefixProber::PrefixListing& listing) {
    std::vector<PeerPage> pages;
    pages.reserve(listing.pages.size());
    for (const PrefixProber::ListedPage& page : listing.pages) {
        pages.push_back(
            PeerPage{.group_id = page.key.group_id, .slot = page.slot, .tier = page.tier, .block_id = page.block_id});
    }
    return pages;
}

}  // namespace

PeerAnswer PeerExports::AnswerLookup(const CacheCoordinator& coordinator, const cache::PeerLookup& lookup) {
    PeerAnswer answer{.lease = lookup.lease, .kind = PeerAnswerKind::kLookup};
    if (!WellFormed(lookup, coordinator.PrefixGranularity())) {
        spdlog::warn("[Peer] lookup {}#{} does not cover its bound; answered as a miss", lookup.lease.reader,
                     lookup.lease.id);
        return answer;
    }
    const PrefixProber::PrefixListing listing = coordinator.Prober().ListPrefix(
        coordinator.Prober().ProbePromptPrefix(Prompt(lookup.prefix_hashes, lookup.tokens, lookup.max_tokens)));
    answer.num_tokens = listing.num_tokens;
    answer.pages = Pages(listing);
    return answer;
}

bool PeerExports::Matches(const CacheCoordinator& coordinator, const cache::PeerLookup& lookup, std::int32_t num_tokens,
                          std::span<const PeerPage> pages) {
    if (!WellFormed(lookup, coordinator.PrefixGranularity()) || num_tokens <= 0 || num_tokens > lookup.max_tokens) {
        return false;
    }
    const PrefixProber::PrefixListing listing = coordinator.Prober().ListPrefix(
        coordinator.Prober().ProbePromptPrefix(PromptUpTo(lookup, num_tokens, coordinator.PrefixGranularity())));
    return listing.num_tokens == num_tokens && std::ranges::equal(Pages(listing), pages);
}

bool PeerExports::Held(const PeerLeaseId& lease) const {
    const auto it = leases_.find(lease);
    return it != leases_.end() && it->second.phase == Phase::kHard;
}

void PeerExports::Lookup(CacheCoordinator& coordinator, const cache::PeerLookup& lookup, std::int64_t now_us) {
    // A lookup repeated under the same lease starts it over.
    leases_.erase(lookup.lease);
    if (!WellFormed(lookup, coordinator.PrefixGranularity()) ||
        static_cast<std::int64_t>(leases_.size()) >= max_leases_) {
        return;
    }
    CacheCoordinator::PrefixExport exported = coordinator.ExportPrefix(
        coordinator.Prober().ProbePromptPrefix(Prompt(lookup.prefix_hashes, lookup.tokens, lookup.max_tokens)));
    if (exported.num_tokens == 0) {
        return;
    }
    // The exported blocks go here: the soft pin only refreshed their entries.
    const auto boundary = static_cast<std::size_t>(exported.num_tokens);
    const auto full_pages = boundary / static_cast<std::size_t>(coordinator.PrefixGranularity());
    leases_.emplace(
        lookup.lease,
        Lease{.deadline_us = Deadline(now_us, soft_pin_us_),
              .prefix_hashes = {lookup.prefix_hashes.begin(),
                                lookup.prefix_hashes.begin() + static_cast<std::ptrdiff_t>(full_pages)},
              .tokens = {lookup.tokens.begin(), lookup.tokens.begin() + static_cast<std::ptrdiff_t>(boundary)},
              .num_tokens = exported.num_tokens});
}

void PeerExports::Pin(CacheCoordinator& coordinator, const cache::PeerPin& pin, std::int64_t now_us) {
    const auto it = leases_.find(pin.lease);
    if (it == leases_.end() || it->second.phase != Phase::kSoft) {
        return;
    }
    Lease& lease = it->second;
    if (pin.num_tokens <= 0 || pin.num_tokens > lease.num_tokens) {
        leases_.erase(it);
        return;
    }
    const auto pages = static_cast<std::size_t>(pin.num_tokens / coordinator.PrefixGranularity());
    CacheCoordinator::PrefixExport exported = coordinator.ExportPrefix(coordinator.Prober().ProbePromptPrefix(
        Prompt(std::span{lease.prefix_hashes}.first(pages),
               std::span{lease.tokens}.first(static_cast<std::size_t>(pin.num_tokens)), pin.num_tokens)));
    // The pin holds what its answer named, or nothing: the blocks it just
    // acquired go back with the lease.
    std::vector<PeerPage> acquired;
    acquired.reserve(exported.blocks.size());
    for (const CacheCoordinator::ExportedBlock& block : exported.blocks) {
        acquired.push_back(PeerPage{
            .group_id = block.key.group_id,
            .slot = block.slot,
            .tier = block.tier,
            .block_id =
                coordinator.Allocator(static_cast<std::int32_t>(block.key.group_id)).ResolveCacheBlockId(*block.block),
        });
    }
    if (exported.num_tokens != pin.num_tokens || acquired != pin.pages) {
        leases_.erase(it);
        return;
    }
    lease.phase = Phase::kHard;
    lease.deadline_us = Deadline(now_us, hard_pin_us_);
    lease.access_epoch = exported.access_epoch;
    for (CacheCoordinator::ExportedBlock& block : exported.blocks) {
        lease.pins.push_back(std::move(block.block));
        lease.keys.push_back(std::move(block.key));
    }
}

void PeerExports::Release(CacheCoordinator& coordinator, const cache::PeerRelease& release) {
    const auto it = leases_.find(release.lease);
    if (it == leases_.end()) {
        return;
    }
    const bool held = it->second.phase == Phase::kHard;
    // The pins go first: a demotion skips an entry anything holds.
    Lease lease = std::move(it->second);
    leases_.erase(it);
    lease.pins.clear();
    if (held && release.demote) {
        coordinator.DemoteExportedChain(lease.keys, lease.access_epoch);
    }
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
