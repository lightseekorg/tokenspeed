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
#include <cstddef>
#include <memory>
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

// One immutable scheduler boundary. The scheduler and routing view share its
// descriptor; neither owns a second token/hash vector for the same boundary.
struct KvEventDescriptor {
    std::uint64_t block_hash;
    std::optional<std::uint64_t> parent_block_hash;
    std::vector<std::int32_t> token_ids;
    std::int32_t block_size;
};
using KvEventDescriptorRef = std::shared_ptr<const KvEventDescriptor>;

struct KvEventMutation {
    std::uint64_t block_hash;
    // Null means that this boundary is no longer locally publishable.
    KvEventDescriptorRef stored;
};

// An independent, owned snapshot. Offsets address contiguous columns, so moving
// a batch across the engine bridge never allocates per-event vectors. It holds
// no descriptor, cache block, request, or scheduler reference.
class KvEventBatch {
public:
    std::size_t Size() const noexcept { return stored_.size(); }
    bool Empty() const noexcept { return stored_.empty(); }
    std::size_t ByteSize() const noexcept;
    void Reserve(std::size_t events, std::size_t hashes, std::size_t tokens);
    void Store(std::span<const std::uint64_t> hashes, std::optional<std::uint64_t> parent,
               std::span<const std::int32_t> tokens, std::int32_t block_size);
    void Remove(std::span<const std::uint64_t> hashes);
    // Compatibility at the Python/diagnostic boundary; the engine consumes
    // the columns directly. All event forms, including multi-hash events,
    // retain the existing wire representation.
    static KvEventBatch FromEvents(std::span<const KvCacheEvent> events);
    std::vector<KvCacheEvent> ToEvents() const;

    std::span<const std::uint8_t> Stored() const noexcept { return stored_; }
    std::span<const std::uint8_t> HasParent() const noexcept { return has_parent_; }
    std::span<const std::uint64_t> Parents() const noexcept { return parents_; }
    std::span<const std::int32_t> BlockSizes() const noexcept { return block_sizes_; }
    std::span<const std::size_t> HashOffsets() const noexcept { return hash_offsets_; }
    std::span<const std::size_t> TokenOffsets() const noexcept { return token_offsets_; }
    std::span<const std::uint64_t> Hashes() const noexcept { return hashes_; }
    std::span<const std::int32_t> Tokens() const noexcept { return tokens_; }

private:
    void Append(bool stored, std::span<const std::uint64_t> hashes, std::optional<std::uint64_t> parent,
                std::span<const std::int32_t> tokens, std::int32_t block_size);
    std::vector<std::uint8_t> stored_;
    std::vector<std::uint8_t> has_parent_;
    std::vector<std::uint64_t> parents_;
    std::vector<std::int32_t> block_sizes_;
    std::vector<std::size_t> hash_offsets_;
    std::vector<std::size_t> token_offsets_;
    std::vector<std::uint64_t> hashes_;
    std::vector<std::int32_t> tokens_;
};

// Projects per-boundary Device residency onto a prefix-closed routing view.
// This owns only event metadata, never cache blocks or eviction decisions.
class KvEventPrefixView {
public:
    // One ordering point: removals leaf-first, stores parent-first. Mutations
    // may arrive in any order, including children before missing parents.
    KvEventBatch ApplyMutations(std::span<const KvEventMutation> mutations);
    std::vector<KvCacheEvent> Apply(std::vector<KvCacheEvent> events);

private:
    struct Node {
        KvEventDescriptorRef resident;
        std::optional<std::uint64_t> parent;
        std::unordered_set<std::uint64_t> children;
        std::int32_t depth{0};
        bool registered{false};
        bool visible{false};
        bool was_visible{false};
        std::uint64_t changed_round{0};
    };
    void Reconcile(std::uint64_t hash);
    void Prune(std::uint64_t hash);
    std::unordered_map<std::uint64_t, Node> nodes_;
    std::uint64_t round_{0};
    // Reused work arrays replace a temporary hash-map entry/allocation for
    // every visibility change and a fresh traversal vector per mutation.
    std::vector<std::uint64_t> changed_;
    std::vector<std::uint64_t> pending_;
};

std::uint64_t HashKvBlock(std::span<const std::int32_t> token_ids, std::optional<std::uint64_t> parent_hash);

}  // namespace tokenspeed
