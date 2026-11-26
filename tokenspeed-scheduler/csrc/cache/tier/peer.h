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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "cache/core/cache_types.h"

namespace tokenspeed {

// Canonicalize DP0's non-canonical "<host>:<port>/0", "/00", "/+0", etc.
// aliases to "<host>:<port>", matching Rust `u32::parse() == Ok(0)`.
inline std::string_view CanonicalPeerEndpoint(std::string_view endpoint) {
    const std::size_t slash = endpoint.find('/');
    if (slash == 0 || slash == std::string_view::npos) {
        return endpoint;
    }
    std::string_view rank = endpoint.substr(slash + 1);
    if (rank.starts_with('+')) {
        rank.remove_prefix(1);
    }
    if (!rank.empty() && std::ranges::all_of(rank, [](char c) { return c == '0'; })) {
        return endpoint.substr(0, slash);
    }
    return endpoint;
}

// Sharing cached prefixes between engine replicas
// (docs/rfcs/0004-peer-prefix-sharing.md). A reader asks a holder for a
// prefix (lookup), pins what the holder still has (pin), reads it into its
// own Device pages and lets the pin go (release). The holder's scheduler owns
// every pin as lease state; the reader's scheduler owns the landing.

// Where a reader lands a peer prefix (docs/rfcs/0006-disk-tier.md, Peer
// sharing): in the Device pages its admission allocates, the request
// waiting out of forward candidates (RFC 0004); or in Host blocks before
// its admission, the request then admitted as an ordinary Host hit.
enum class PeerLandingTier { kDevice, kHost };

// Explicit at the runtime/scheduler boundary; nothing here has a default.
struct PeerConfig {
    // The name other replicas address this one by, as the router's holder
    // hint carries it. A lease names its reader by it.
    std::string name;
    // This process's incarnation, which a restart changes: a reader's lease
    // ids count from it, so no lease of a restarted reader meets a holder's
    // lease of the reader it replaced. Nonzero and 32-bit: the ids count up
    // from incarnation << 32.
    std::uint64_t incarnation;
    // Holder: leases held at once. A lookup past it is answered with a miss.
    std::int32_t max_leases;
    // Holder: how long a soft pin (a lookup's) and a hard pin last.
    std::int64_t soft_pin_us;
    std::int64_t hard_pin_us;
    // Reader: how long a request waits for a lookup's answer.
    std::int64_t lookup_timeout_us;
    // Reader: the smallest extension of the local hit worth a fetch.
    std::int32_t min_fetch_tokens;
    // Reader: where landings go.
    PeerLandingTier landing_tier;
    // Reader, Host landings: the Host blocks landings hold at once, and how
    // long a landed prefix is held for its request's admission. Zero with
    // Device landings, which use neither.
    std::int32_t max_host_landing_blocks;
    std::int64_t host_ready_timeout_us;
    bool versioned_shards{false};
    void Validate() const;
};

// A lease is named by its reader and the reader's own counter.
struct PeerLeaseId {
    std::string reader;
    std::uint64_t id{0};

    bool operator==(const PeerLeaseId&) const = default;
};

struct PeerLeaseIdHash {
    std::size_t operator()(const PeerLeaseId& lease) const noexcept {
        return std::hash<std::string>{}(lease.reader) ^ (std::hash<std::uint64_t>{}(lease.id) * 0x9E3779B97F4A7C15ULL);
    }
};

// One block of a prefix: on the holder, a cached block a pin holds (its tier
// and block id there); on the reader, a destination block (tier kDevice).
// `slot` is the block's position in the group's request table, so a holder's
// page and a reader's destination of the same group and slot pair up. A
// State group's checkpoint fills the reader's live slot, 1.
struct PeerPage {
    std::uint32_t group_id{0};
    std::int32_t slot{0};
    CacheTier tier{CacheTier::kDevice};
    std::int32_t block_id{0};

    bool operator==(const PeerPage&) const = default;
};

inline constexpr std::int32_t kPeerStateSlot = 1;

// Reader -> holder: a waiting prompt's question, sent once.
struct PeerLookupRequest {
    std::string request_id;
    std::int64_t deadline_us{0};
    PeerLeaseId lease;
    // The holder's name (PeerConfig::name there).
    std::string holder;
    // The chain hashes of the prompt's full pages below max_tokens, and its
    // tokens up to max_tokens, from which the holder hashes tails.
    std::vector<std::string> prefix_hashes;
    std::vector<std::int32_t> tokens;
    std::int32_t max_tokens{0};

    bool operator==(const PeerLookupRequest&) const = default;
};

enum class PeerAnswerKind { kLookup, kPin, kRelease };

// Holder -> reader, answered at once from the holder's cache as it stands,
// without a scheduling round: a lookup's answer carries the boundary the
// holder can resume the prompt at (0: a miss) and the pages to read; a pin's,
// whether those pages are still the ones (held: true); a release's, whether
// the pages stayed untouched from the pin to the release.
struct PeerAnswer {
    PeerLeaseId lease;
    PeerAnswerKind kind{PeerAnswerKind::kLookup};
    std::int32_t num_tokens{0};
    std::vector<PeerPage> pages;
    bool held{false};

    bool operator==(const PeerAnswer&) const = default;
};

// Reader: an admitted prompt's landing. Tokens [begin_tokens, end_tokens)
// come from the holder into `destinations`; the runtime pins, reads and
// releases, then reports the boundary it confirmed (PeerFetchDone).
struct PeerFetchOperation {
    std::string request_id;
    PeerLeaseId lease;
    std::string holder;
    std::int32_t begin_tokens{0};
    std::int32_t end_tokens{0};
    std::vector<PeerPage> destinations;
    std::vector<std::uint8_t> checkpoint;
    std::vector<std::string> prefix_hashes;
    std::vector<std::int32_t> tokens;

    bool operator==(const PeerFetchOperation&) const = default;
};

namespace cache {

// The peer clock, agreed by the replica's members; it times leases and waits.
// Setting expire=false advances peer_now_us_ without sweeping leases/waits yet
// so holder events answered against the pre-tick state (Lookup, Pin, Release)
// can be applied at now_us before expiration runs.
struct PeerTick {
    std::int64_t now_us{0};
    bool expire{true};
};

// Holder: a reader's lookup, pin and release, as every member applies them
// at the head of the round after the holder answered them: the lookup's soft
// pin, the pin's block references, the release's demotion.
struct PeerLookup {
    PeerLeaseId lease;
    std::vector<std::string> prefix_hashes;
    std::vector<std::int32_t> tokens;
    std::int32_t max_tokens{0};
};
struct PeerPin {
    PeerLeaseId lease;
    // The boundary and pages the holder told the reader it may read: the pin
    // holds exactly these, or nothing.
    std::int32_t num_tokens{0};
    std::vector<PeerPage> pages;
};
struct PeerRelease {
    PeerLeaseId lease;
    // The reader read and published the whole pinned prefix: the holder's
    // copy becomes the spare one.
    bool demote{false};
};

// Reader: a lookup's answer, and a landing's outcome: the end of the prefix
// every member read in full under a pin the holder confirmed held, or the
// landing's begin when nothing is usable.
struct PeerLookupAnswered {
    PeerLeaseId lease;
    std::int32_t num_tokens{0};
    unsigned source_mask{0};  // Device=1, Host=2, Disk=4; mixed sources combine bits.
    std::vector<std::uint8_t> checkpoint;
};
struct PeerFetchDone {
    std::string request_id;
    std::int32_t num_tokens{0};
    std::int64_t dispatch_us{0};
    std::int64_t writable_us{0};
    std::int64_t terminal_us{0};
    PeerLeaseId lease;
    bool frozen{false};
};

}  // namespace cache

}  // namespace tokenspeed
