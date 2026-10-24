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
#include <cstdint>
#include <set>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/core/cache_types.h"
#include "cache/prefix/prefix_hasher.h"

namespace tokenspeed {

// Checkpoint ancestry shares immutable prefix segments. Its references own
// metadata only; they never pin a cache block or a request.
class CheckpointTree {
public:
    explicit CheckpointTree(std::int32_t prefix_granularity) : prefix_granularity_{prefix_granularity} {}

    void Observe(const CompletedPages& completed) {
        if (!completed.publish_state || completed.tokens.empty()) {
            return;
        }
        _assert(
            completed.tokens.size() == completed.prefix_hashes.size() * prefix_granularity_ +
                                           static_cast<std::size_t>(completed.tail ? completed.tail->tail_tokens : 0),
            "checkpoint tokens must cover exactly the published full prefix and tail");
        const ContentHash hash = completed.tail ? completed.tail->hash : completed.prefix_hashes.back();
        if (nodes_.contains(hash)) {
            return;
        }
        Node node{.full_prefix = InternPrefix(completed),
                  .num_tokens = static_cast<std::int32_t>(completed.tokens.size())};
        if (node.full_prefix != nullptr) {
            ++node.full_prefix->references;
        }
        const auto full_tokens = completed.prefix_hashes.size() * static_cast<std::size_t>(prefix_granularity_);
        node.tail.assign(completed.tokens.begin() + full_tokens, completed.tokens.end());
        node.parent = FindParent(completed);
        auto& siblings = node.parent.empty() ? roots_ : nodes_.at(node.parent).children;
        std::vector<ContentHash> adopted;
        for (const auto& child_hash : siblings) {
            if (IsPrefix(hash, node, nodes_.at(child_hash))) {
                adopted.push_back(child_hash);
            }
        }
        for (const auto& child_hash : adopted) {
            Node& child = nodes_.at(child_hash);
            auto split = child.branch_points.lower_bound(node.num_tokens);
            node.branch_points.insert(child.branch_points.begin(), split);
            if (split != child.branch_points.end() && *split == node.num_tokens) {
                node.was_branch = true;
                ++split;
            }
            child.branch_points.erase(child.branch_points.begin(), split);
            child.parent = hash;
            siblings.erase(child_hash);
            node.children.insert(child_hash);
        }
        siblings.insert(hash);
        if (!node.tail.empty()) {
            tails_by_prior_[Prior(node)].insert(hash);
        }
        nodes_.emplace(hash, std::move(node));
    }

    template <typename IsResident>
    bool Demoted(const ContentHash& hash, const IsResident& resident) const {
        auto it = nodes_.find(hash);
        if (it == nodes_.end()) {
            return false;
        }
        std::int32_t descendants = 0;
        while (descendants < 2) {
            const Node& node = it->second;
            if (node.was_branch || node.children.size() != 1) {
                return false;
            }
            it = nodes_.find(*node.children.begin());
            if (!it->second.branch_points.empty()) {
                return false;
            }
            descendants += resident(it->first) ? 1 : 0;
        }
        return true;
    }

    template <typename IsResident>
    std::unordered_set<ContentHash> DemotedHashes(const IsResident& resident) const {
        std::unordered_set<ContentHash> complete;
        for (const auto& [hash, node] : nodes_) {
            if (resident(hash)) complete.insert(hash);
        }
        std::unordered_set<ContentHash> result;
        for (const auto& [hash, node] : nodes_) {
            if (Demoted(hash, [&](const ContentHash& child) { return complete.contains(child); })) {
                result.insert(hash);
            }
        }
        return result;
    }

    template <typename IsResident>
    void Prune(const IsResident& resident) {
        std::vector<ContentHash> pending;
        for (const auto& [hash, node] : nodes_) {
            if (!resident(hash)) pending.push_back(hash);
        }
        while (!pending.empty()) {
            ContentHash hash = std::move(pending.back());
            pending.pop_back();
            const auto it = nodes_.find(hash);
            if (it == nodes_.end() || resident(hash) || it->second.children.size() > 1) continue;
            Node& node = it->second;
            const ContentHash parent = node.parent;
            auto& siblings = parent.empty() ? roots_ : nodes_.at(parent).children;
            if (node.children.empty() && siblings.size() > 1) {
                const auto parent_depth = parent.empty() ? 0 : nodes_.at(parent).num_tokens;
                for (const ContentHash& sibling_hash : siblings) {
                    if (sibling_hash == hash) continue;
                    Node& sibling = nodes_.at(sibling_hash);
                    const auto fork = CommonPrefix(node, sibling);
                    sibling.branch_points.insert(node.branch_points.begin(), node.branch_points.lower_bound(fork));
                    if (fork > parent_depth) {
                        sibling.branch_points.insert(fork);
                    } else if (!parent.empty()) {
                        nodes_.at(parent).was_branch = true;
                    }
                }
            }
            siblings.erase(hash);
            if (!node.children.empty()) {
                const ContentHash child_hash = *node.children.begin();
                Node& child = nodes_.at(child_hash);
                child.parent = parent;
                child.branch_points.insert(node.branch_points.begin(), node.branch_points.end());
                if (node.was_branch) child.branch_points.insert(node.num_tokens);
                siblings.insert(child_hash);
            }
            if (!node.tail.empty()) {
                auto tails = tails_by_prior_.find(Prior(node));
                tails->second.erase(hash);
                if (tails->second.empty()) tails_by_prior_.erase(tails);
            }
            ReleasePrefix(node.full_prefix);
            nodes_.erase(it);
            if (!parent.empty()) pending.push_back(parent);
        }
    }

    void Clear() {
        nodes_.clear();
        roots_.clear();
        tails_by_prior_.clear();
        prefixes_.clear();
    }
    std::size_t Size() const { return nodes_.size(); }
    std::size_t PrefixSegments() const { return prefixes_.size(); }
    bool Contains(const ContentHash& hash) const { return nodes_.contains(hash); }

private:
    struct PrefixSegment {
        ContentHash hash;
        std::vector<std::int32_t> tokens;
        PrefixSegment* parent{nullptr};
        std::int32_t depth{0};
        std::size_t references{0};
        // Ancestors at distances 1, 2, 4, ...; the parent reference owns them.
        std::vector<PrefixSegment*> ancestors;
    };
    struct Node {
        PrefixSegment* full_prefix{nullptr};
        std::int32_t num_tokens{0};
        std::vector<std::int32_t> tail;
        ContentHash parent{};
        std::unordered_set<ContentHash> children{};
        bool was_branch{false};
        // Historical fork positions between this node and its parent.
        std::set<std::int32_t> branch_points{};
    };

    static ContentHash Prior(const Node& node) {
        return node.full_prefix == nullptr ? ContentHash{} : node.full_prefix->hash;
    }

    PrefixSegment* InternPrefix(const CompletedPages& completed) {
        std::size_t first = completed.prefix_hashes.size();
        PrefixSegment* parent = nullptr;
        while (first > 0) {
            const auto it = prefixes_.find(completed.prefix_hashes[first - 1]);
            if (it != prefixes_.end()) {
                parent = it->second.get();
                break;
            }
            --first;
        }
        for (std::size_t i = first; i < completed.prefix_hashes.size(); ++i) {
            const auto tokens = completed.tokens.subspan(i * prefix_granularity_, prefix_granularity_);
            auto segment = std::make_unique<PrefixSegment>(PrefixSegment{.hash = completed.prefix_hashes[i],
                                                                         .tokens = {tokens.begin(), tokens.end()},
                                                                         .parent = parent,
                                                                         .depth = static_cast<std::int32_t>(i + 1)});
            if (parent != nullptr) {
                ++parent->references;
                segment->ancestors.push_back(parent);
                for (std::size_t level = 1;; ++level) {
                    const auto* previous = segment->ancestors[level - 1];
                    if (previous->ancestors.size() < level) break;
                    segment->ancestors.push_back(previous->ancestors[level - 1]);
                }
            }
            auto* next = segment.get();
            prefixes_.emplace(segment->hash, std::move(segment));
            parent = next;
        }
        return parent;
    }

    void ReleasePrefix(PrefixSegment* segment) {
        while (segment != nullptr) {
            _assert(segment->references > 0, "prefix metadata reference underflow");
            if (--segment->references != 0) break;
            auto* parent = segment->parent;
            const ContentHash hash = segment->hash;
            prefixes_.erase(hash);
            segment = parent;
        }
    }

    static const PrefixSegment* SegmentAt(const Node& node, std::int32_t depth) {
        if (depth == 0) return nullptr;
        const auto* segment = node.full_prefix;
        _assert(segment != nullptr && segment->depth >= depth, "checkpoint prefix depth out of bounds");
        auto distance = static_cast<std::uint32_t>(segment->depth - depth);
        for (std::size_t bit = 0; distance != 0; ++bit, distance >>= 1) {
            if (distance & 1) segment = segment->ancestors[bit];
        }
        return segment;
    }

    ContentHash FindParent(const CompletedPages& completed) const {
        const auto length = static_cast<std::int32_t>(completed.tokens.size());
        for (std::int32_t full = static_cast<std::int32_t>(completed.prefix_hashes.size()); full >= 0; --full) {
            const ContentHash prior = full == 0 ? ContentHash{} : completed.prefix_hashes[full - 1];
            ContentHash best;
            std::int32_t best_length = full * prefix_granularity_;
            if (const auto it = tails_by_prior_.find(prior); it != tails_by_prior_.end()) {
                for (const ContentHash& hash : it->second) {
                    const Node& node = nodes_.at(hash);
                    if (node.num_tokens <= best_length || node.num_tokens >= length) continue;
                    const auto tail = completed.tokens.subspan(full * prefix_granularity_, node.tail.size());
                    if (HashPrefixPage(tail, prior) == hash) {
                        best = hash;
                        best_length = node.num_tokens;
                    }
                }
            }
            if (!best.empty()) return best;
            if (full > 0 && full * prefix_granularity_ < length && nodes_.contains(prior)) return prior;
        }
        return {};
    }

    std::int32_t CommonPrefix(const Node& a, const Node& b) const {
        std::int32_t low = 0;
        std::int32_t high = std::min(a.num_tokens, b.num_tokens) / prefix_granularity_;
        while (low < high) {
            const auto mid = low + (high - low + 1) / 2;
            if (SegmentAt(a, mid)->hash == SegmentAt(b, mid)->hash)
                low = mid;
            else
                high = mid - 1;
        }
        const auto partial = [&](const Node& node) -> std::span<const std::int32_t> {
            if (node.num_tokens / prefix_granularity_ > low) return SegmentAt(node, low + 1)->tokens;
            return node.tail;
        };
        const auto left = partial(a), right = partial(b);
        std::size_t shared = 0;
        while (shared < std::min(left.size(), right.size()) && left[shared] == right[shared]) ++shared;
        return low * prefix_granularity_ + static_cast<std::int32_t>(shared);
    }

    bool IsPrefix(const ContentHash& hash, const Node& prefix, const Node& longer) const {
        if (prefix.num_tokens >= longer.num_tokens) return false;
        const auto full = prefix.num_tokens / prefix_granularity_;
        if (full > 0 && SegmentAt(longer, full)->hash != prefix.full_prefix->hash) return false;
        if (prefix.tail.empty()) return true;
        const auto longer_full = longer.num_tokens / prefix_granularity_;
        const std::span<const std::int32_t> tokens =
            longer_full > full ? std::span<const std::int32_t>{SegmentAt(longer, full + 1)->tokens}
                               : std::span<const std::int32_t>{longer.tail};
        return HashPrefixPage(tokens.first(prefix.tail.size()), Prior(prefix)) == hash;
    }

    std::int32_t prefix_granularity_;
    std::unordered_map<ContentHash, Node> nodes_;
    std::unordered_set<ContentHash> roots_;
    std::unordered_map<ContentHash, std::unordered_set<ContentHash>> tails_by_prior_;
    std::unordered_map<ContentHash, std::unique_ptr<PrefixSegment>> prefixes_;
};

}  // namespace tokenspeed
