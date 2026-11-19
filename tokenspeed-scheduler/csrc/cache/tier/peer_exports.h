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
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/tier/peer.h"

namespace tokenspeed {

// The holder's half of peer prefix sharing. Answers are reads: the
// replica's first member gives them as soon as a message arrives, while the
// scheduler is idle between rounds, from the cache as it stands. What the
// messages change -- a lookup's soft pin, a pin's block references, a
// release's demotion -- every member applies at the head of the next round,
// before anything else, in the order the answers were given, so mirrored
// ranks hold the same leases and the answers held in between: nothing leaves
// the cache outside a plan, and no plan runs between an answer and the head
// that applies it.
//
// A soft pin refreshes the looked-up entries, so they leave last, and keeps
// the prompt; a pin holds the exact pages its answer named, or nothing if
// they changed. A pin that expired leaves a tombstone, so a release that
// follows does not count it as held.
class PeerExports {
public:
    explicit PeerExports(const PeerConfig& config);

    // The answers.
    static PeerAnswer AnswerLookup(const CacheCoordinator& coordinator, const cache::PeerLookup& lookup);
    // Whether the lookup's prompt, up to num_tokens, still resolves to
    // exactly `pages`.
    static bool Matches(const CacheCoordinator& coordinator, const cache::PeerLookup& lookup, std::int32_t num_tokens,
                        std::span<const PeerPage> pages);
    // Whether the lease's pin holds its pages now.
    bool Held(const PeerLeaseId& lease) const;

    // The changes, applied at the head.
    void Lookup(CacheCoordinator& coordinator, const cache::PeerLookup& lookup, std::int64_t now_us);
    void Pin(CacheCoordinator& coordinator, const cache::PeerPin& pin, std::int64_t now_us);
    // A release that asks for it, after a pin that held, demotes the pinned
    // prefix's leaf chain (CacheCoordinator::DemoteExportedChain): the
    // reader has its own copy now.
    void Release(CacheCoordinator& coordinator, const cache::PeerRelease& release);
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
        // The prompt up to the looked-up boundary; a pin probes it again up
        // to the boundary its answer named.
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
