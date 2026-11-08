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

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "cache/core/cache_config.h"
#include "scheduler/kv_cache_events.h"

namespace tokenspeed {

class KvEventFrameWriter;  // the msgpack cursor of kv_event_ledger.cpp

// The KV-event publisher's bookkeeping, kept so that a plan step costs it
// nothing beyond a few counter updates: no prefix-index query, no allocation
// in steady state, one msgpack frame reused across drains.
//
// A boundary (a prefix page or a tail, named by its SHA-256 content hash) has
// one descriptor: its external block hash and parent, its tokens, its chain
// depth, and per cache group how many of the group's Device entries for it
// are indexed. The coordinator's mutation sink moves those counts as pages
// are stored and evicted, so a group "holds" the boundary exactly while the
// count equals the group's block count for it (a State group's one checkpoint
// block, a window group's blocks of the tail it needs). Drain publishes the
// net change since the previous drain per boundary: one BlockStored carrying
// the mask of the groups that hold it when a group newly holds it, one
// BlockRemoved carrying the mask of the groups that dropped it. A store and a
// removal of the same boundary inside one plan cancel: nothing is published.
//
// Descriptors live in a slab sized to the Device cache (one boundary per
// Device page at most), their tokens in an arena beside it, and the hash
// index is open addressing over fixed-size keys: after the slab filled once,
// registering, mutating and draining allocate nothing. A slab that still
// overflows grows and says so once.
class KvEventLedger {
public:
    static constexpr std::size_t kMaxGroups = 64;
    static constexpr std::size_t kHashLength = 64;  // a SHA-256 digest in hex
    static constexpr std::uint32_t kNoSlot = UINT32_MAX;

    struct GroupKind {
        AttnKind attn_kind{AttnKind::kFull};
        // The window's tokens for a kSlidingWindow group, else 0.
        std::int32_t sliding_window{0};
    };

    // `groups` in the scheduler's cache-group index order; `prefix_granularity`
    // bounds a descriptor's tokens; `capacity` is the boundaries the slab holds
    // before it grows.
    KvEventLedger(std::vector<GroupKind> groups, std::int32_t prefix_granularity, std::size_t capacity);

    // Registers the descriptor of a boundary, or confirms an existing one
    // (the same hash names the same block and depth), and marks it for the
    // next drain. `blocks_per_group[i]` is how many Device entries group i
    // indexes for this boundary when it holds it (0 for a group that never
    // holds one, a replayable window).
    void Register(std::string_view content_hash, std::uint64_t block_hash, std::optional<std::uint64_t> parent_hash,
                  std::span<const std::int32_t> tokens, std::int32_t depth,
                  std::span<const std::int32_t> blocks_per_group);

    // A Device entry of `group` for the boundary was indexed (`stored`) or
    // evicted. The boundary has a descriptor: registration precedes every
    // mutation of a tracked boundary.
    void Mutate(std::string_view content_hash, std::uint32_t group, bool stored);

    struct Descriptor {
        std::int32_t depth{0};
        std::int32_t tokens{0};
    };
    // The chain depth and token count of a tracked boundary.
    std::optional<Descriptor> Find(std::string_view content_hash) const;

    // The publisher's batch `[timestamp, events, attn_dp_rank]` for the net
    // change since the previous drain, encoded as the reference engine's
    // publisher encodes its events (every slot written, nil where this engine
    // has nothing): removals leaf-first, then stores parent-first. Empty when
    // nothing changed. The bytes live until the next Drain.
    std::span<const std::uint8_t> Drain(double timestamp, std::int64_t dp_rank);

    // A frame's events as structs (tests, the Python binding).
    static std::vector<KvCacheEvent> DecodeFrame(std::span<const std::uint8_t> frame);

    // Tracked boundaries.
    std::size_t Size() const { return live_; }
    // How many times the slab grew past its initial capacity.
    std::uint32_t Growths() const { return growths_; }

private:
    struct Slot {
        std::array<char, kHashLength> hash{};
        // HashOfDigest(hash), kept so a probe compares eight bytes before the
        // sixty-four and an erase never rehashes its neighbours.
        std::uint64_t digest_hash{0};
        std::uint64_t block_hash{0};
        std::uint64_t parent_hash{0};
        bool has_parent{false};
        bool live{false};
        bool marked{false};
        std::int32_t depth{0};
        std::int32_t tokens{0};
        // Groups whose every block of this boundary is indexed on the Device.
        std::uint64_t complete_mask{0};
        // Groups whose BlockStored for this boundary is out.
        std::uint64_t published_mask{0};
        // Device entries indexed for this boundary, all groups.
        std::int32_t resident_total{0};
        std::uint32_t next_free{kNoSlot};
    };

    std::uint32_t find(std::string_view content_hash) const;
    std::uint32_t insert(std::string_view content_hash, std::uint64_t digest_hash);
    void erase(std::uint32_t slot);
    void grow();
    void rebuildIndex();
    std::span<std::uint16_t> resident(std::uint32_t slot) {
        return {&resident_[slot * groups_.size()], groups_.size()};
    }
    std::span<std::uint16_t> blocks(std::uint32_t slot) { return {&blocks_[slot * groups_.size()], groups_.size()}; }
    std::span<std::int32_t> tokens(std::uint32_t slot) {
        return {&tokens_[slot * static_cast<std::size_t>(prefix_granularity_)],
                static_cast<std::size_t>(slots_[slot].tokens)};
    }
    void mark(std::uint32_t slot);
    void encodeStore(KvEventFrameWriter& writer, std::uint32_t slot, std::uint64_t newly, std::uint64_t holding);
    void encodeRemoval(KvEventFrameWriter& writer, std::uint32_t slot, std::uint64_t dropped);

    std::vector<GroupKind> groups_;
    std::int32_t prefix_granularity_;
    std::vector<Slot> slots_;
    std::vector<std::uint16_t> resident_;  // slots x groups
    std::vector<std::uint16_t> blocks_;    // slots x groups
    std::vector<std::int32_t> tokens_;     // slots x prefix_granularity
    std::vector<std::uint32_t> index_;     // open addressing: slot + 1, 0 empty
    std::uint32_t free_head_{kNoSlot};
    std::size_t live_{0};
    std::uint32_t growths_{0};
    std::vector<std::uint32_t> marked_;
    // Drain scratch, capacity retained. `order` is the boundary's position in
    // the mark list, so an in-place sort by (depth, order) is stable.
    struct Pending {
        std::int32_t depth;
        std::uint32_t order;
        std::uint32_t slot;
        std::uint64_t mask;
        std::uint64_t holding;
    };
    std::vector<Pending> stores_;
    std::vector<Pending> removals_;
    std::vector<std::uint8_t> frame_;
};

}  // namespace tokenspeed
