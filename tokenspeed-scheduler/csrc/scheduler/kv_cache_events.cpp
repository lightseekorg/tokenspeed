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
#include <limits>
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

void KvEventPrefixView::Reconcile(std::uint64_t hash) {
    pending_.clear();
    pending_.push_back(hash);
    while (!pending_.empty()) {
        const auto current = pending_.back();
        pending_.pop_back();
        auto& node = nodes_.at(current);
        const bool visible = node.resident != nullptr && (!node.parent || nodes_.at(*node.parent).visible);
        if (visible == node.visible) continue;
        if (node.changed_round != round_) {
            node.changed_round = round_;
            node.was_visible = node.visible;
            changed_.push_back(current);
        }
        node.visible = visible;
        if (visible) node.depth = node.parent ? nodes_.at(*node.parent).depth + 1 : 0;
        pending_.insert(pending_.end(), node.children.begin(), node.children.end());
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
    std::vector<KvEventMutation> mutations;
    mutations.reserve(events.size());
    for (auto& event : events) {
        if (auto* stored = std::get_if<KvBlockStoredEvent>(&event)) {
            if (stored->block_hashes.size() != 1) {
                throw std::invalid_argument("routing residency events must describe one boundary");
            }
            const auto hash = stored->block_hashes.front();
            mutations.push_back({hash, std::make_shared<const KvEventDescriptor>(
                                           KvEventDescriptor{.block_hash = hash,
                                                             .parent_block_hash = stored->parent_block_hash,
                                                             .token_ids = std::move(stored->token_ids),
                                                             .block_size = stored->block_size})});
        } else {
            for (const auto hash : std::get<KvBlockRemovedEvent>(event).block_hashes) {
                mutations.push_back({hash, {}});
            }
        }
    }
    return ApplyMutations(mutations).ToEvents();
}

KvEventBatch KvEventPrefixView::ApplyMutations(std::span<const KvEventMutation> mutations) {
    // A wrapped generation must not alias one retained by a live node.
    if (round_ == std::numeric_limits<std::uint64_t>::max()) {
        for (auto& [hash, node] : nodes_) node.changed_round = 0;
        round_ = 0;
    }
    ++round_;
    changed_.clear();
    std::vector<std::uint64_t> removed_candidates;
    for (const auto& mutation : mutations) {
        const auto hash = mutation.block_hash;
        if (mutation.stored) {
            const auto& stored = *mutation.stored;
            if (stored.block_hash != hash) throw std::invalid_argument("event descriptor hash mismatch");
            auto& node = nodes_[hash];
            if (node.registered && node.parent != stored.parent_block_hash) {
                throw std::invalid_argument("a routing boundary cannot change its parent");
            }
            node.parent = stored.parent_block_hash;
            node.registered = true;
            if (node.parent) nodes_[*node.parent].children.insert(hash);
            node.resident = mutation.stored;
            Reconcile(hash);
        } else {
            const auto it = nodes_.find(hash);
            if (it == nodes_.end()) continue;
            it->second.resident.reset();
            Reconcile(hash);
            removed_candidates.push_back(hash);
        }
    }
    std::vector<std::pair<std::int32_t, std::uint64_t>> removed, stored;
    std::size_t token_count = 0;
    for (const auto hash : changed_) {
        const auto& node = nodes_.at(hash);
        if (node.was_visible == node.visible) continue;
        (node.visible ? stored : removed).emplace_back(node.depth, hash);
        if (node.visible) {
            const auto count = node.resident->token_ids.size();
            if (count > std::numeric_limits<std::size_t>::max() - token_count) {
                throw std::length_error("KV event token count overflows size_t");
            }
            token_count += count;
        }
    }
    std::sort(removed.begin(), removed.end(), std::greater<>());
    std::sort(stored.begin(), stored.end());
    KvEventBatch result;
    const auto count = removed.size() + stored.size();
    result.Reserve(count, count, token_count);
    for (const auto& [depth, hash] : removed) result.Remove(std::span{&hash, 1});
    for (const auto& [depth, hash] : stored) {
        const auto& descriptor = *nodes_.at(hash).resident;
        result.Store(std::span{&descriptor.block_hash, 1}, descriptor.parent_block_hash, descriptor.token_ids,
                     descriptor.block_size);
    }
    for (const auto hash : removed_candidates) Prune(hash);
    return result;
}

void KvEventBatch::Reserve(std::size_t events, std::size_t hashes, std::size_t tokens) {
    if (events == 0) return;
    if (events == std::numeric_limits<std::size_t>::max()) throw std::length_error("too many KV events");
    stored_.reserve(events);
    has_parent_.reserve(events);
    parents_.reserve(events);
    block_sizes_.reserve(events);
    hash_offsets_.reserve(events + 1);
    token_offsets_.reserve(events + 1);
    hashes_.reserve(hashes);
    tokens_.reserve(tokens);
}

void KvEventBatch::Append(bool stored, std::span<const std::uint64_t> hashes, std::optional<std::uint64_t> parent,
                          std::span<const std::int32_t> tokens, std::int32_t block_size) {
    if (stored_.empty()) {
        hash_offsets_.push_back(0);
        token_offsets_.push_back(0);
    }
    stored_.push_back(stored);
    has_parent_.push_back(parent.has_value());
    parents_.push_back(parent.value_or(0));
    block_sizes_.push_back(block_size);
    hashes_.insert(hashes_.end(), hashes.begin(), hashes.end());
    tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
    hash_offsets_.push_back(hashes_.size());
    token_offsets_.push_back(tokens_.size());
}

void KvEventBatch::Store(std::span<const std::uint64_t> hashes, std::optional<std::uint64_t> parent,
                         std::span<const std::int32_t> tokens, std::int32_t block_size) {
    Append(true, hashes, parent, tokens, block_size);
}
void KvEventBatch::Remove(std::span<const std::uint64_t> hashes) {
    Append(false, hashes, std::nullopt, {}, 0);
}

std::size_t KvEventBatch::ByteSize() const noexcept {
    return stored_.size() * sizeof(std::uint8_t) + has_parent_.size() * sizeof(std::uint8_t) +
           parents_.size() * sizeof(std::uint64_t) + block_sizes_.size() * sizeof(std::int32_t) +
           hash_offsets_.size() * sizeof(std::size_t) + token_offsets_.size() * sizeof(std::size_t) +
           hashes_.size() * sizeof(std::uint64_t) + tokens_.size() * sizeof(std::int32_t);
}

KvEventBatch KvEventBatch::FromEvents(std::span<const KvCacheEvent> events) {
    KvEventBatch result;
    for (const auto& event : events) {
        if (const auto* stored = std::get_if<KvBlockStoredEvent>(&event)) {
            result.Store(stored->block_hashes, stored->parent_block_hash, stored->token_ids, stored->block_size);
        } else {
            result.Remove(std::get<KvBlockRemovedEvent>(event).block_hashes);
        }
    }
    return result;
}

std::vector<KvCacheEvent> KvEventBatch::ToEvents() const {
    std::vector<KvCacheEvent> result;
    result.reserve(Size());
    for (std::size_t i = 0; i < Size(); ++i) {
        std::vector<std::uint64_t> hashes(hashes_.begin() + hash_offsets_[i], hashes_.begin() + hash_offsets_[i + 1]);
        if (stored_[i]) {
            result.emplace_back(KvBlockStoredEvent{
                .block_hashes = std::move(hashes),
                .parent_block_hash = has_parent_[i] ? std::optional{parents_[i]} : std::nullopt,
                .token_ids = {tokens_.begin() + token_offsets_[i], tokens_.begin() + token_offsets_[i + 1]},
                .block_size = block_sizes_[i]});
        } else {
            result.emplace_back(KvBlockRemovedEvent{.block_hashes = std::move(hashes)});
        }
    }
    return result;
}

}  // namespace tokenspeed
