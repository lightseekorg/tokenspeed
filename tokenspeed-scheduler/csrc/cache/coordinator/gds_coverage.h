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
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/coordinator/recovery_index.h"
#include "utils.h"

namespace tokenspeed {
struct RecoveryInterval {
    std::size_t first;
    std::size_t past_end;
};

// Heavy-path coordinates for the current dependency forest. A checkpoint's
// history/window path becomes O(log(leaves)) intervals, not one edge per KV.
// Coordinates are plan-local; cache residency and dependencies remain indexed.
class RecoveryLayout {
public:
    explicit RecoveryLayout(std::span<const RecoveryIndex::Leaf> leaves) : positions_(leaves.size()) {
        const auto n = leaves.size();
        std::vector<std::vector<std::size_t>> children(n);
        parents_.assign(n, n);
        heads_.resize(n);
        for (std::size_t i = 0; i < n; ++i) by_key_.emplace(leaves[i].key, i);
        std::vector<std::size_t> roots;
        for (std::size_t i = 0; i < n; ++i) {
            auto parent = leaves[i].previous ? by_key_.find(*leaves[i].previous) : by_key_.end();
            if (parent == by_key_.end())
                roots.push_back(i);
            else {
                parents_[i] = parent->second;
                children[parent->second].push_back(i);
            }
        }
        std::vector<std::size_t> order = roots;
        for (std::size_t i = 0; i < order.size(); ++i)
            order.insert(order.end(), children[order[i]].begin(), children[order[i]].end());
        FatalCheck(order.size() == n, "recovery dependencies form a forest");
        std::vector<std::size_t> sizes(n, 1), heavy(n, n);
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const auto v = *it;
            for (auto child : children[v]) {
                sizes[v] += sizes[child];
                if (heavy[v] == n || sizes[child] > sizes[heavy[v]]) heavy[v] = child;
            }
        }
        std::vector<std::pair<std::size_t, std::size_t>> stack;
        for (auto root : roots) stack.emplace_back(root, root);
        std::size_t position = 0;
        while (!stack.empty()) {
            auto [v, head] = stack.back();
            stack.pop_back();
            for (; v != n; v = heavy[v]) {
                heads_[v] = head;
                positions_[v] = position++;
                for (auto child : children[v])
                    if (child != heavy[v]) stack.emplace_back(child, child);
            }
        }
        keys_.resize(n);
        for (std::size_t i = 0; i < n; ++i) keys_[positions_[i]] = leaves[i].key;
    }
    std::size_t Size() const { return positions_.size(); }
    const CacheKey& KeyAt(std::size_t position) const { return keys_[position]; }
    std::optional<std::size_t> Position(const CacheKey& key) const {
        const auto it = by_key_.find(key);
        return it == by_key_.end() ? std::nullopt : std::optional{positions_[it->second]};
    }
    std::vector<RecoveryInterval> Intervals(std::span<const RecoveryIndex::Path> paths) const {
        std::vector<RecoveryInterval> result;
        for (const auto& path : paths) {
            const auto first = by_key_.at(path.first);
            auto last = by_key_.at(path.last);
            while (heads_[first] != heads_[last]) {
                result.push_back({positions_[heads_[last]], positions_[last] + 1});
                last = parents_[heads_[last]];
                FatalCheck(last < Size(), "a recovery path's first key is an ancestor of its last key");
            }
            FatalCheck(positions_[first] <= positions_[last], "recovery path order is forward");
            result.push_back({positions_[first], positions_[last] + 1});
        }
        std::ranges::sort(result, {}, &RecoveryInterval::first);
        std::size_t out = 0;
        for (auto range : result) {
            if (out > 0 && result[out - 1].past_end >= range.first)
                result[out - 1].past_end = std::max(result[out - 1].past_end, range.past_end);
            else
                result[out++] = range;
        }
        result.resize(out);
        return result;
    }

private:
    std::unordered_map<CacheKey, std::size_t, CacheKeyHash> by_key_;
    std::vector<std::size_t> positions_, parents_, heads_;
    std::vector<CacheKey> keys_;
};

// Count and XOR of live users over the layout. Lazy range subtraction skips
// shared history until a location becomes exclusive or orphaned. Ownership
// eligibility is supplied by the admission's real candidate census.
class RecoveryCoverage {
public:
    struct Leaf {
        bool eligible;
        std::int64_t users;
        std::size_t owners;
    };
    explicit RecoveryCoverage(std::span<const Leaf> leaves) : size_{leaves.size()}, nodes_(4 * leaves.size()) {
        if (size_ > 0) build(1, 0, size_, leaves);
    }
    template <typename Visit>
    void ForEachExclusive(RecoveryInterval range, std::size_t owner, const Visit& visit) {
        exclusive(1, 0, size_, range, owner, visit);
    }
    template <typename Changed>
    void Remove(RecoveryInterval range, std::size_t owner, const Changed& changed) {
        remove(1, 0, size_, range, owner, changed);
    }
    void Disable(std::size_t position) { disable(1, 0, size_, position); }

private:
    static constexpr auto kInactive = std::numeric_limits<std::int64_t>::max();
    struct Node {
        std::int64_t minimum{kInactive};
        std::int64_t delta{0};
        std::size_t owners{0};
        std::size_t xor_delta{0};
    };
    std::size_t size_;
    std::vector<Node> nodes_;
    void build(std::size_t node, std::size_t first, std::size_t end, std::span<const Leaf> leaves) {
        if (end - first == 1) {
            nodes_[node].minimum = leaves[first].eligible ? leaves[first].users : kInactive;
            nodes_[node].owners = leaves[first].owners;
            return;
        }
        const auto mid = first + (end - first) / 2;
        build(node * 2, first, mid, leaves);
        build(node * 2 + 1, mid, end, leaves);
        pull(node);
    }
    void apply(std::size_t node, std::int64_t delta, std::size_t owners) {
        auto& value = nodes_[node];
        if (value.minimum != kInactive) value.minimum += delta;
        value.delta += delta;
        value.owners ^= owners;
        value.xor_delta ^= owners;
    }
    void push(std::size_t node) {
        const auto delta = std::exchange(nodes_[node].delta, 0);
        const auto owners = std::exchange(nodes_[node].xor_delta, 0);
        apply(node * 2, delta, owners);
        apply(node * 2 + 1, delta, owners);
    }
    void pull(std::size_t node) {
        nodes_[node].minimum = std::min(nodes_[node * 2].minimum, nodes_[node * 2 + 1].minimum);
    }
    template <typename Visit>
    void exclusive(std::size_t node, std::size_t first, std::size_t end, RecoveryInterval range, std::size_t owner,
                   const Visit& visit) {
        if (end <= range.first || first >= range.past_end || nodes_[node].minimum > 1) return;
        if (end - first == 1) {
            FatalCheck(nodes_[node].minimum == 1 && nodes_[node].owners == owner,
                       "exclusive recovery location has its sole owner");
            visit(first);
            return;
        }
        push(node);
        const auto mid = first + (end - first) / 2;
        exclusive(node * 2, first, mid, range, owner, visit);
        exclusive(node * 2 + 1, mid, end, range, owner, visit);
    }
    template <typename Changed>
    void remove(std::size_t node, std::size_t first, std::size_t end, RecoveryInterval range, std::size_t owner,
                const Changed& changed) {
        if (end <= range.first || first >= range.past_end) return;
        if (range.first <= first && end <= range.past_end && nodes_[node].minimum > 2) {
            apply(node, -1, owner);
            return;
        }
        if (end - first == 1) {
            FatalCheck(nodes_[node].minimum > 0, "removing a recovery user cannot underflow coverage");
            apply(node, -1, owner);
            changed(first, nodes_[node].minimum, nodes_[node].owners);
            return;
        }
        push(node);
        const auto mid = first + (end - first) / 2;
        remove(node * 2, first, mid, range, owner, changed);
        remove(node * 2 + 1, mid, end, range, owner, changed);
        pull(node);
    }
    void disable(std::size_t node, std::size_t first, std::size_t end, std::size_t position) {
        if (end - first == 1) {
            nodes_[node].minimum = kInactive;
            return;
        }
        push(node);
        const auto mid = first + (end - first) / 2;
        if (position < mid)
            disable(node * 2, first, mid, position);
        else
            disable(node * 2 + 1, mid, end, position);
        pull(node);
    }
};
}  // namespace tokenspeed
