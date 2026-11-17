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

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/tier/peer.h"

namespace tokenspeed {

// The holder's half of peer prefix sharing: one lease per reader's lookup.
// A lookup soft-pins the prefix it answers -- it refreshes the entries, so
// they leave last, and keeps the prompt -- and a pin turns that into block
// references, which hold the pages on their tier until the release or the
// pin's deadline. A pin that expired leaves a tombstone, so the release that
// follows reports that it did not hold. The scheduler applies every call in
// a plan, from events its replica agreed on, so mirrored ranks hold the same
// leases.
class PeerExports {
public:
    explicit PeerExports(const PeerConfig& config);

    PeerAnswer Lookup(CacheCoordinator& coordinator, const cache::PeerLookup& lookup, std::int64_t now_us);
    PeerAnswer Pin(CacheCoordinator& coordinator, const cache::PeerPin& pin, std::int64_t now_us);
    // A release that asks for it, after a pin that held, demotes the pinned
    // prefix's leaf chain (CacheCoordinator::DemoteExportedChain): the
    // reader has its own copy now.
    PeerAnswer Release(CacheCoordinator& coordinator, const cache::PeerRelease& release);
    // Lets every lease past its deadline go: a soft pin is forgotten, a hard
    // pin's blocks go back and its tombstone stays for one more hard-pin
    // lifetime.
    void Expire(std::int64_t now_us);

    std::size_t NumLeases() const { return leases_.size(); }
    std::size_t NumPinnedBlocks() const;

private:
    enum class Phase { kSoft, kHard, kExpired };
    struct Lease {
        Phase phase{Phase::kSoft};
        std::int64_t deadline_us{0};
        // The prompt up to the looked-up boundary: a pin probes it again, so
        // it never pins past what the reader admitted.
        std::vector<std::string> prefix_hashes;
        std::vector<std::int32_t> tokens;
        std::int32_t num_tokens{0};
        // A hard pin's blocks, their entries' keys and the access epoch the
        // pin dated them with.
        std::vector<CacheBlockRef> pins;
        std::vector<CacheKey> keys;
        std::uint64_t access_epoch{0};
    };

    std::int32_t max_leases_;
    std::int64_t soft_pin_us_;
    std::int64_t hard_pin_us_;
    std::unordered_map<PeerLeaseId, Lease, PeerLeaseIdHash> leases_;
};

}  // namespace tokenspeed
