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

// Every cache group publishes its own events for a boundary, under the same
// block hashes: a hybrid model's History group reports its whole chain, its
// State group the boundaries that hold a checkpoint, a sliding-window group
// the boundaries its window still covers. A consumer tells the groups apart
// by group_idx (the scheduler's cache-group index) and reads what each one
// means from the group's attention kind.
struct KvBlockStoredEvent {
    std::vector<std::uint64_t> block_hashes;
    std::optional<std::uint64_t> parent_block_hash;
    std::vector<std::int32_t> token_ids;
    std::int32_t block_size;
    std::int32_t group_idx{0};
    AttnKind attn_kind{AttnKind::kFull};
    // The window's tokens for a kSlidingWindow group, none otherwise.
    std::optional<std::int32_t> sliding_window;
};

struct KvBlockRemovedEvent {
    std::vector<std::uint64_t> block_hashes;
    std::int32_t group_idx{0};
};

using KvCacheEvent = std::variant<KvBlockStoredEvent, KvBlockRemovedEvent>;

// The group's attention kind as the event wire spells it, the names a
// consumer already reads from the reference engine's publisher:
// full_attention, sliding_window and mamba (a recurrent-state group).
std::string_view KvCacheSpecKind(AttnKind attn_kind);

std::uint64_t HashKvBlock(std::span<const std::int32_t> token_ids, std::optional<std::uint64_t> parent_hash);

}  // namespace tokenspeed
