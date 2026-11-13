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
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace tokenspeed {

struct KvBlockStoredEvent {
    std::vector<std::uint64_t> block_hashes;
    std::optional<std::uint64_t> parent_block_hash;
    std::vector<std::int32_t> token_ids;
    std::int32_t block_size;
};

struct KvBlockRemovedEvent {
    std::vector<std::uint64_t> block_hashes;
};

using KvCacheEvent = std::variant<KvBlockStoredEvent, KvBlockRemovedEvent>;

// Projects per-boundary Device residency onto a prefix-closed routing view.
// This owns only event metadata, never cache blocks or eviction decisions.
class KvEventPrefixView {
public:
    // Scheduler Stored events each describe one boundary, including short tails.
    // Return net visibility changes: removals leaf-first, stores parent-first.
    std::vector<KvCacheEvent> Apply(std::vector<KvCacheEvent> events);

private:
    struct Node {
        std::optional<KvBlockStoredEvent> resident;
        std::optional<std::uint64_t> parent;
        std::unordered_set<std::uint64_t> children;
        std::int32_t depth{0};
        bool registered{false};
        bool visible{false};
    };
    void Reconcile(std::uint64_t hash, std::unordered_map<std::uint64_t, bool>& before);
    void Prune(std::uint64_t hash);
    std::unordered_map<std::uint64_t, Node> nodes_;
};

std::uint64_t HashKvBlock(std::span<const std::int32_t> token_ids, std::optional<std::uint64_t> parent_hash);

}  // namespace tokenspeed
