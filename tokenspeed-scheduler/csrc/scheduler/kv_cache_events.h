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

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "cache/core/cache_config.h"

namespace tokenspeed {

// One event per boundary: a BlockStored when a cache group newly holds the
// boundary, carrying group_mask, the groups that hold it after the drain (bit
// i for cache group i), and a BlockRemoved when groups dropped it, carrying
// the dropped groups. group_idx, the kind and the window name the lowest
// group the event is about, so a consumer that reads one group per event
// (the reference engine's shape) sees that group's store or removal; a
// consumer that reads the mask sees the hybrid model's History chain, its
// State checkpoints and its sliding windows under the same block hashes.
struct KvBlockStoredEvent {
    std::vector<std::uint64_t> block_hashes;
    std::optional<std::uint64_t> parent_block_hash;
    std::vector<std::int32_t> token_ids;
    std::int32_t block_size;
    std::int32_t group_idx{0};
    AttnKind attn_kind{AttnKind::kFull};
    // The window's tokens for a kSlidingWindow group, none otherwise.
    std::optional<std::int32_t> sliding_window;
    // The groups that hold the boundary after this event, bit i for group i.
    std::uint64_t group_mask{0};
};

struct KvBlockRemovedEvent {
    std::vector<std::uint64_t> block_hashes;
    std::int32_t group_idx{0};
    // The groups that dropped the boundary, bit i for group i.
    std::uint64_t group_mask{0};
};

using KvCacheEvent = std::variant<KvBlockStoredEvent, KvBlockRemovedEvent>;

// The group's attention kind as the event wire spells it, the names a
// consumer already reads from the reference engine's publisher:
// full_attention, sliding_window and mamba (a recurrent-state group).
std::string_view KvCacheSpecKind(AttnKind attn_kind);

std::uint64_t HashKvBlock(std::span<const std::int32_t> token_ids, std::optional<std::uint64_t> parent_hash);

}  // namespace tokenspeed
