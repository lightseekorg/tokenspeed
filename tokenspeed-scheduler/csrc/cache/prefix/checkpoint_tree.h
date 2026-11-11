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
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cache/core/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// State checkpoints ordered by token prefix, including checkpoints inside a
// page, with each checkpoint's distance to the nearest leaf below it: the
// newest checkpoint on that branch. Distance counts the later requests that
// published checkpoints present on either tier: the checkpoints one request
// publishes (in-forward, prompt end, end state) are no step apart. A
// checkpoint more than leaf_distance from a leaf is aged (CheckpointAging);
// one within it ranks as a leaf does. The tree holds metadata only, never a
// cache block.
class CheckpointTree {
public:
    // Distances from here on rank alike, which bounds every update to the
    // checkpoints of this many earlier requests above it.
    static constexpr std::int32_t kMaxDistance = kMaxCheckpointDistance;

    struct Residency {
        // Every State group holds the checkpoint, per CacheTier.
        std::array<bool, 2> complete{};
        bool pending_host_store{false};
        // Some State entry or pending store still holds it.
        bool held{false};

        bool operator==(const Residency&) const = default;
    };

    CheckpointTree(std::int32_t prefix_granularity, std::int32_t leaf_distance)
        : prefix_granularity_{prefix_granularity}, leaf_distance_{leaf_distance} {
        _assert(leaf_distance_ >= 0 && leaf_distance_ < kMaxDistance, "checkpoint leaf distance out of range");
    }
    CheckpointTree(const CheckpointTree&) = delete;
    CheckpointTree& operator=(const CheckpointTree&) = delete;
    CheckpointTree(CheckpointTree&&) = default;
    CheckpointTree& operator=(CheckpointTree&&) = default;

    // Records the checkpoint at the end of `tokens`, whose full pages are
    // `prefix_hashes` and partial page `tail`, published by the request of
    // `access_epoch`. Its residency starts empty until Update. False when the
    // checkpoint is already known.
    bool Observe(std::span<const std::int32_t> tokens, std::span<const std::string> prefix_hashes,
                 const std::optional<TailPublication>& tail, std::uint64_t access_epoch) {
        const std::size_t full_tokens = prefix_hashes.size() * static_cast<std::size_t>(prefix_granularity_);
        _assert(
            !tokens.empty() && tokens.size() == full_tokens + static_cast<std::size_t>(tail ? tail->tail_tokens : 0),
            "checkpoint tokens cover exactly its full pages and tail");
        const ContentHash& hash = tail ? tail->hash : ContentHash{prefix_hashes.back()};
        if (nodes_.contains(hash)) {
            return false;
        }
        const auto [full_prefix, existing_pages] = internPrefix(tokens, prefix_hashes);
        Node* parent = findParent(tokens, prefix_hashes, existing_pages);
        const auto it = nodes_.try_emplace(hash).first;
        Node& node = it->second;
        node.hash = &it->first;
        node.full_prefix = full_prefix;
        if (full_prefix != nullptr) {
            ++full_prefix->references;
        }
        node.num_tokens = static_cast<std::int32_t>(tokens.size());
        node.access_epoch = access_epoch;
        node.tail.assign(tokens.begin() + static_cast<std::ptrdiff_t>(full_tokens), tokens.end());
        // A shorter checkpoint published after longer ones on its path takes
        // them as children. Only an already interned prefix can have them.
        if (prefix_hashes.size() <= existing_pages) {
            std::vector<Node*>& siblings = parent != nullptr ? parent->children : roots_;
            for (std::size_t i = 0; i < siblings.size();) {
                Node& sibling = *siblings[i];
                if (extends(sibling, node)) {
                    detach(sibling);
                    attach(sibling, &node);
                } else {
                    ++i;
                }
            }
        }
        attach(node, parent);
        if (!node.tail.empty()) {
            tails_by_prior_[prior(node)].insert(&node);
        }
        refresh(&node);
        return true;
    }

    // A checkpoint nothing holds leaves; its children take its place.
    void Update(const ContentHash& hash, const Residency& residency) {
        const auto it = nodes_.find(hash);
        if (it == nodes_.end()) {
            return;
        }
        Node& node = it->second;
        const bool was_present = present(node);
        node.residency = residency;
        if (residency.held) {
            if (present(node) != was_present) {
                for (Node* child : node.children) {
                    reanchor(*child);
                }
            }
            refresh(&node);
        } else {
            remove(it);
        }
    }

    // The distance of a checkpoint present on either tier.
    std::optional<std::int32_t> Distance(const ContentHash& hash) const {
        const auto it = nodes_.find(hash);
        if (it == nodes_.end() || !present(it->second)) {
            return std::nullopt;
        }
        return distanceOf(it->second);
    }

    // The distance of an aged checkpoint complete on `tier`, else 0. A
    // junction, with checkpoints on two or more branches, has been resumed
    // from more than once and keeps its rank.
    std::int32_t AgedDistance(const ContentHash& hash, CacheTier tier) const {
        const auto it = nodes_.find(hash);
        return it == nodes_.end() ? 0 : it->second.aged_distance[static_cast<std::size_t>(tier)];
    }

    // Visits the hashes AgedDistance(hash, tier) == distance, in no order.
    template <typename Visit>
    void ForEachAged(CacheTier tier, std::int32_t distance, const Visit& visit) const {
        _assert(distance > leaf_distance_ && distance <= kMaxDistance, "aged distance out of range");
        for (const Node* node : aged_[static_cast<std::size_t>(tier)][static_cast<std::size_t>(distance)]) {
            visit(*node->hash);
        }
    }

    // Whether a checkpoint stays more than leaf_distance from a leaf once the
    // checkpoints `removed(hash)` reports leave `tier`.
    template <typename Removed>
    bool StaysAged(const ContentHash& hash, CacheTier tier, const Removed& removed) const {
        const auto it = nodes_.find(hash);
        return it != nodes_.end() && !leafWithin(it->second, leaf_distance_, static_cast<std::size_t>(tier), removed);
    }

    void Clear() {
        nodes_.clear();
        roots_.clear();
        tails_by_prior_.clear();
        prefixes_.clear();
        for (auto& by_distance : aged_) {
            for (auto& hashes : by_distance) {
                hashes.clear();
            }
        }
    }

    std::int32_t LeafDistance() const { return leaf_distance_; }
    std::size_t Size() const { return nodes_.size(); }
    std::size_t PrefixSegments() const { return prefixes_.size(); }
    bool Contains(const ContentHash& hash) const { return nodes_.contains(hash); }

private:
    // One full page of tokens, shared by every checkpoint whose prefix holds it.
    struct PrefixSegment {
        ContentHash hash;
        std::vector<std::int32_t> tokens;
        PrefixSegment* parent{nullptr};
        std::int32_t depth{0};
        std::size_t references{0};
        // Ancestors at distances 1, 2, 4, ...; the parent reference keeps them.
        std::vector<PrefixSegment*> ancestors;
    };

    struct Node {
        const ContentHash* hash{nullptr};
        PrefixSegment* full_prefix{nullptr};
        std::int32_t num_tokens{0};
        // The publishing request's access epoch.
        std::uint64_t access_epoch{0};
        std::vector<std::int32_t> tail;
        Node* parent{nullptr};
        std::vector<Node*> children;
        // Position in the parent's children (or the roots).
        std::size_t slot{0};
        Residency residency{};
        // How many children report each value (index 0 unused).
        std::array<std::uint32_t, kMaxDistance + 2> child_values{};
        // What the parent sees, 0 for no checkpoint: a present checkpoint
        // reports 1 + its distance as seen from its nearest present ancestor
        // (one step more unless that ancestor's request published it); an
        // absent one passes on its children's smallest value.
        std::uint8_t value{0};
        std::array<std::int32_t, 2> aged_distance{};
    };

    static bool present(const Node& node) {
        return node.residency.complete[0] || node.residency.complete[1] || node.residency.pending_host_store;
    }

    static std::uint8_t smallestChildValue(const Node& node) {
        for (std::size_t value = 1; value < node.child_values.size(); ++value) {
            if (node.child_values[value] != 0) {
                return static_cast<std::uint8_t>(value);
            }
        }
        return 0;
    }

    static std::uint32_t branchesWithCheckpoints(const Node& node) {
        std::uint32_t branches = 0;
        for (std::size_t value = 1; value < node.child_values.size(); ++value) {
            branches += node.child_values[value];
        }
        return branches;
    }

    static std::int32_t distanceOf(const Node& node) {
        const std::uint8_t smallest = smallestChildValue(node);
        return smallest == 0 ? 0 : smallest - 1;
    }

    // Whether `node` is a step from its nearest present ancestor: another
    // request published it, or it has no such ancestor.
    static bool stepsFromAnchor(const Node& node) {
        const Node* anchor = node.parent;
        while (anchor != nullptr && !present(*anchor)) {
            anchor = anchor->parent;
        }
        return anchor == nullptr || anchor->access_epoch != node.access_epoch;
    }

    static std::uint8_t valueOf(const Node& node) {
        if (!present(node)) {
            return smallestChildValue(node);
        }
        const std::int32_t seen = distanceOf(node) + (stepsFromAnchor(node) ? 1 : 0);
        return static_cast<std::uint8_t>(std::min(kMaxDistance, seen) + 1);
    }

    static ContentHash prior(const Node& node) {
        return node.full_prefix == nullptr ? ContentHash{} : node.full_prefix->hash;
    }

    void reindex(Node& node) {
        const std::int32_t distance =
            branchesWithCheckpoints(node) == 1 && distanceOf(node) > leaf_distance_ ? distanceOf(node) : 0;
        for (std::size_t tier = 0; tier < node.aged_distance.size(); ++tier) {
            const std::int32_t wanted = node.residency.complete[tier] ? distance : 0;
            std::int32_t& current = node.aged_distance[tier];
            if (wanted == current) {
                continue;
            }
            if (current != 0) {
                aged_[tier][static_cast<std::size_t>(current)].erase(&node);
            }
            if (wanted != 0) {
                aged_[tier][static_cast<std::size_t>(wanted)].insert(&node);
            }
            current = wanted;
        }
    }

    // Whether a leaf lies within `hops` steps below `node`, `node` itself at
    // 0, counting the checkpoints present once `removed` ones leave `tier`.
    // A checkpoint `node`'s own request published is no step.
    template <typename Removed>
    static bool leafWithin(const Node& node, std::int32_t hops, std::size_t tier, const Removed& removed) {
        std::vector<const Node*> nearest;
        std::vector<const Node*> pending(node.children.begin(), node.children.end());
        while (!pending.empty()) {
            const Node* below = pending.back();
            pending.pop_back();
            const Residency& residency = below->residency;
            if (residency.pending_host_store || residency.complete[1 - tier] ||
                (residency.complete[tier] && !removed(*below->hash))) {
                nearest.push_back(below);
            } else {
                pending.insert(pending.end(), below->children.begin(), below->children.end());
            }
        }
        if (nearest.empty()) {
            return true;
        }
        return std::ranges::any_of(nearest, [&](const Node* below) {
            const std::int32_t left = hops - (below->access_epoch != node.access_epoch ? 1 : 0);
            return left >= 0 && leafWithin(*below, left, tier, removed);
        });
    }

    // Recomputes a node and its ancestors until a reported value holds.
    void refresh(Node* node) {
        while (node != nullptr) {
            reindex(*node);
            const std::uint8_t value = valueOf(*node);
            if (value == node->value) {
                return;
            }
            if (node->parent != nullptr) {
                if (node->value != 0) {
                    --node->parent->child_values[node->value];
                }
                if (value != 0) {
                    ++node->parent->child_values[value];
                }
            }
            node->value = value;
            node = node->parent;
        }
    }

    // Recomputes the present checkpoints at and below `node` whose nearest
    // present ancestor may have changed, and the absent nodes above them.
    void reanchor(Node& node) {
        if (!present(node)) {
            for (Node* child : node.children) {
                reanchor(*child);
            }
        }
        refresh(&node);
    }

    void attach(Node& node, Node* parent) {
        node.parent = parent;
        std::vector<Node*>& siblings = parent != nullptr ? parent->children : roots_;
        node.slot = siblings.size();
        siblings.push_back(&node);
        if (parent != nullptr && node.value != 0) {
            ++parent->child_values[node.value];
        }
    }

    void detach(Node& node) {
        std::vector<Node*>& siblings = node.parent != nullptr ? node.parent->children : roots_;
        Node* last = siblings.back();
        siblings[node.slot] = last;
        last->slot = node.slot;
        siblings.pop_back();
        if (node.parent != nullptr && node.value != 0) {
            --node.parent->child_values[node.value];
        }
    }

    void remove(std::unordered_map<ContentHash, Node>::iterator it) {
        Node& node = it->second;
        Node* parent = node.parent;
        detach(node);
        for (Node* child : node.children) {
            attach(*child, parent);
        }
        for (Node* child : node.children) {
            reanchor(*child);
        }
        node.children.clear();
        node.child_values = {};
        reindex(node);
        if (!node.tail.empty()) {
            const auto tails = tails_by_prior_.find(prior(node));
            tails->second.erase(&node);
            if (tails->second.empty()) {
                tails_by_prior_.erase(tails);
            }
        }
        releasePrefix(node.full_prefix);
        nodes_.erase(it);
        refresh(parent);
    }

    // The segment of the last full page, and how many leading pages were
    // interned already: a checkpoint's ancestors keep theirs.
    std::pair<PrefixSegment*, std::size_t> internPrefix(std::span<const std::int32_t> tokens,
                                                        std::span<const std::string> prefix_hashes) {
        std::size_t first = prefix_hashes.size();
        PrefixSegment* parent = nullptr;
        while (first > 0) {
            if (const auto it = prefixes_.find(prefix_hashes[first - 1]); it != prefixes_.end()) {
                parent = it->second.get();
                break;
            }
            --first;
        }
        const std::size_t existing_pages = first;
        const auto page = static_cast<std::size_t>(prefix_granularity_);
        for (std::size_t i = first; i < prefix_hashes.size(); ++i) {
            const auto page_tokens = tokens.subspan(i * page, page);
            auto segment = std::make_unique<PrefixSegment>(PrefixSegment{
                .hash = prefix_hashes[i],
                .tokens = {page_tokens.begin(), page_tokens.end()},
                .parent = parent,
                .depth = static_cast<std::int32_t>(i + 1),
            });
            if (parent != nullptr) {
                ++parent->references;
                segment->ancestors.push_back(parent);
                for (std::size_t level = 1;; ++level) {
                    const PrefixSegment* previous = segment->ancestors[level - 1];
                    if (previous->ancestors.size() < level) {
                        break;
                    }
                    segment->ancestors.push_back(previous->ancestors[level - 1]);
                }
            }
            PrefixSegment* next = segment.get();
            prefixes_.emplace(segment->hash, std::move(segment));
            parent = next;
        }
        return {parent, existing_pages};
    }

    void releasePrefix(PrefixSegment* segment) {
        while (segment != nullptr) {
            // Fatal rather than thrown: Update gets here from a boundary-tree
            // hold's release, which runs in destructors.
            FatalCheck(segment->references > 0, "checkpoint prefix reference underflow");
            if (--segment->references != 0) {
                return;
            }
            PrefixSegment* parent = segment->parent;
            const ContentHash hash = segment->hash;
            prefixes_.erase(hash);
            segment = parent;
        }
    }

    static const PrefixSegment* segmentAt(const Node& node, std::int32_t depth) {
        if (depth == 0) {
            return nullptr;
        }
        const PrefixSegment* segment = node.full_prefix;
        _assert(segment != nullptr && segment->depth >= depth, "checkpoint prefix depth out of range");
        auto distance = static_cast<std::uint32_t>(segment->depth - depth);
        for (std::size_t bit = 0; distance != 0; ++bit, distance >>= 1) {
            if ((distance & 1) != 0) {
                segment = segment->ancestors[bit];
            }
        }
        return segment;
    }

    // The deepest known checkpoint strictly inside these tokens. It keeps its
    // prefix interned, so it lies within the first existing_pages pages.
    Node* findParent(std::span<const std::int32_t> tokens, std::span<const std::string> prefix_hashes,
                     std::size_t existing_pages) {
        const auto length = static_cast<std::int32_t>(tokens.size());
        for (auto full = static_cast<std::int32_t>(existing_pages); full >= 0; --full) {
            const ContentHash prior =
                full == 0 ? ContentHash{} : ContentHash{prefix_hashes[static_cast<std::size_t>(full - 1)]};
            const std::int32_t page_start = full * prefix_granularity_;
            Node* best = nullptr;
            if (const auto it = tails_by_prior_.find(prior); it != tails_by_prior_.end()) {
                for (Node* candidate : it->second) {
                    if (candidate->num_tokens >= length ||
                        (best != nullptr && candidate->num_tokens <= best->num_tokens)) {
                        continue;
                    }
                    if (std::ranges::equal(tokens.subspan(static_cast<std::size_t>(page_start), candidate->tail.size()),
                                           candidate->tail)) {
                        best = candidate;
                    }
                }
            }
            if (best != nullptr) {
                return best;
            }
            if (full > 0 && page_start < length) {
                if (const auto it = nodes_.find(prior); it != nodes_.end()) {
                    return &it->second;
                }
            }
        }
        return nullptr;
    }

    bool extends(const Node& longer, const Node& prefix) const {
        if (prefix.num_tokens >= longer.num_tokens) {
            return false;
        }
        const std::int32_t full = prefix.num_tokens / prefix_granularity_;
        if (full > 0 && segmentAt(longer, full) != prefix.full_prefix) {
            return false;
        }
        if (prefix.tail.empty()) {
            return true;
        }
        const std::span<const std::int32_t> page =
            longer.num_tokens / prefix_granularity_ > full
                ? std::span<const std::int32_t>{segmentAt(longer, full + 1)->tokens}
                : std::span<const std::int32_t>{longer.tail};
        return std::ranges::equal(page.first(prefix.tail.size()), prefix.tail);
    }

    std::int32_t prefix_granularity_;
    std::int32_t leaf_distance_;
    std::unordered_map<ContentHash, Node> nodes_;
    std::vector<Node*> roots_;
    std::unordered_map<ContentHash, std::unordered_set<Node*>> tails_by_prior_;
    std::unordered_map<ContentHash, std::unique_ptr<PrefixSegment>> prefixes_;
    // Per tier and distance, the aged checkpoints complete there.
    std::array<std::array<std::unordered_set<const Node*>, kMaxDistance + 1>, 2> aged_{};
};

// How an eviction plan on `tier` ranks aged State checkpoints: one used at or
// after cutoff_epoch ranks as if last used then, the larger distance first.
// Without a tree every entry ranks by its own access epoch.
struct CheckpointAging {
    const CheckpointTree* tree{nullptr};
    CacheTier tier{CacheTier::kDevice};
    std::uint64_t cutoff_epoch{0};
};

}  // namespace tokenspeed
