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

#include "scheduler/kv_cache_events.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace tokenspeed {
namespace {

constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

void MixByte(std::uint64_t& hash, std::uint8_t byte) {
    hash ^= byte;
    hash *= kFnvPrime;
}

void MixUint64(std::uint64_t& hash, std::uint64_t value) {
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        MixByte(hash, static_cast<std::uint8_t>((value >> (i * 8)) & 0xffu));
    }
}

void MixInt32(std::uint64_t& hash, std::int32_t value) {
    const auto raw = static_cast<std::uint32_t>(value);
    for (std::size_t i = 0; i < sizeof(raw); ++i) {
        MixByte(hash, static_cast<std::uint8_t>((raw >> (i * 8)) & 0xffu));
    }
}

}  // namespace

std::uint64_t HashKvBlock(std::span<const std::int32_t> token_ids, std::optional<std::uint64_t> parent_hash) {
    std::uint64_t hash = kFnvOffsetBasis;
    MixByte(hash, parent_hash.has_value() ? 1 : 0);
    if (parent_hash.has_value()) {
        MixUint64(hash, *parent_hash);
    }
    MixUint64(hash, static_cast<std::uint64_t>(token_ids.size()));
    for (std::int32_t token_id : token_ids) {
        MixInt32(hash, token_id);
    }
    return hash;
}

void KvEventPrefixView::Reconcile(std::uint64_t hash, std::unordered_map<std::uint64_t, bool>& before) {
    std::vector<std::uint64_t> pending{hash};
    while (!pending.empty()) {
        const auto current = pending.back();
        pending.pop_back();
        auto& node = nodes_.at(current);
        const bool visible = node.resident.has_value() && (!node.parent || nodes_.at(*node.parent).visible);
        if (visible == node.visible) continue;
        before.try_emplace(current, node.visible);
        node.visible = visible;
        if (visible) node.depth = node.parent ? nodes_.at(*node.parent).depth + 1 : 0;
        // Only a visibility transition can affect descendants. Hidden subtrees
        // stay untouched until their first missing ancestor becomes visible.
        pending.insert(pending.end(), node.children.begin(), node.children.end());
    }
}

void KvEventPrefixView::Prune(std::uint64_t hash) {
    while (true) {
        const auto it = nodes_.find(hash);
        if (it == nodes_.end() || it->second.resident || !it->second.children.empty()) return;
        const auto parent = it->second.parent;
        nodes_.erase(it);
        if (!parent) return;
        nodes_.at(*parent).children.erase(hash);
        hash = *parent;
    }
}

std::vector<KvCacheEvent> KvEventPrefixView::Apply(std::vector<KvCacheEvent> events) {
    std::unordered_map<std::uint64_t, bool> before;
    std::vector<std::uint64_t> removed_candidates;
    for (auto& event : events) {
        if (auto* stored = std::get_if<KvBlockStoredEvent>(&event)) {
            if (stored->block_hashes.size() != 1) {
                throw std::invalid_argument("routing residency events must describe one boundary");
            }
            const auto hash = stored->block_hashes.front();
            auto& node = nodes_[hash];
            if (node.registered && node.parent != stored->parent_block_hash) {
                throw std::invalid_argument("a routing boundary cannot change its parent");
            }
            node.parent = stored->parent_block_hash;
            node.registered = true;
            if (node.parent) nodes_[*node.parent].children.insert(hash);
            node.resident = std::move(*stored);
            Reconcile(hash, before);
        } else {
            for (const auto hash : std::get<KvBlockRemovedEvent>(event).block_hashes) {
                const auto it = nodes_.find(hash);
                if (it == nodes_.end()) continue;
                it->second.resident.reset();
                Reconcile(hash, before);
                removed_candidates.push_back(hash);
            }
        }
    }
    std::vector<std::pair<std::int32_t, std::uint64_t>> removed, stored;
    for (const auto& [hash, was_visible] : before) {
        const auto& node = nodes_.at(hash);
        if (was_visible == node.visible) continue;
        (node.visible ? stored : removed).emplace_back(node.depth, hash);
    }
    std::sort(removed.begin(), removed.end(), std::greater<>());
    std::sort(stored.begin(), stored.end());
    std::vector<KvCacheEvent> result;
    result.reserve(removed.size() + stored.size());
    for (const auto& [depth, hash] : removed) result.emplace_back(KvBlockRemovedEvent{.block_hashes = {hash}});
    for (const auto& [depth, hash] : stored) result.emplace_back(*nodes_.at(hash).resident);
    // Keep missing ancestors only while a resident descendant needs their link.
    for (const auto hash : removed_candidates) Prune(hash);
    return result;
}

}  // namespace tokenspeed
