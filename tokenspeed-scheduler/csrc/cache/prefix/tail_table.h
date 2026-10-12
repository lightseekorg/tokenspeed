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
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/core/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// The published tail boundaries, indexed by the chain hash of the last full
// page before them ("" for page 0). A probe cannot enumerate hash@T without
// knowing T, so it reads the offsets published under its last matched page
// here and hashes its own tokens up to each one. The table is a lookup
// accelerator over the prefix indexes, not a tier of its own: an offset whose
// entries were evicted is a stale hint that costs one hash and a miss, so
// pruning is lazy (the coordinator removes a hash when its last Device entry
// leaves and no Host entry remains). Per prior at most kMaxTailsPerPrior
// tails are kept in least-recently-used order: a publication or a hit moves
// a tail to the back, the cap drops the front.
class TailTable {
public:
    static constexpr std::int32_t kMaxTailsPerPrior = 8;

    struct Tail {
        std::int32_t tail_tokens{0};  // T mod prefix_granularity, in (0, P)
        ContentHash hash{};           // hash@T
    };

    // Records a published tail; a hash already recorded becomes the most
    // recently used. Returns the tail the per-prior cap dropped, if any; the
    // caller evicts its entries.
    std::optional<Tail> Add(const ContentHash& prior, std::int32_t tail_tokens, const ContentHash& hash) {
        _assert(tail_tokens > 0, "a tail boundary lies strictly inside a prefix page");
        _assert(!hash.empty(), "a tail is published under a content hash");
        if (Touch(hash)) {
            return std::nullopt;
        }
        std::vector<Tail>& tails = tails_by_prior_[prior];
        tails.push_back(Tail{.tail_tokens = tail_tokens, .hash = hash});
        prior_of_.emplace(hash, prior);
        if (static_cast<std::int32_t>(tails.size()) <= kMaxTailsPerPrior) {
            return std::nullopt;
        }
        Tail dropped = std::move(tails.front());
        tails.erase(tails.begin());
        prior_of_.erase(dropped.hash);
        ++cap_evictions_;
        return dropped;
    }

    // A hit on a recorded tail makes it the most recently used of its prior.
    // Returns whether the hash was recorded.
    bool Touch(const ContentHash& hash) {
        const auto prior_it = prior_of_.find(hash);
        if (prior_it == prior_of_.end()) {
            return false;
        }
        std::vector<Tail>& tails = tails_by_prior_.at(prior_it->second);
        const auto it = std::ranges::find(tails, hash, &Tail::hash);
        _assert(it != tails.end(), "a recorded tail hash is in its prior's list");
        std::rotate(it, it + 1, tails.end());
        return true;
    }

    void Remove(const ContentHash& hash) {
        const auto prior_it = prior_of_.find(hash);
        if (prior_it == prior_of_.end()) {
            return;
        }
        const auto tails_it = tails_by_prior_.find(prior_it->second);
        _assert(tails_it != tails_by_prior_.end(), "a recorded tail hash names a recorded prior");
        std::erase_if(tails_it->second, [&hash](const Tail& tail) { return tail.hash == hash; });
        if (tails_it->second.empty()) {
            tails_by_prior_.erase(tails_it);
        }
        prior_of_.erase(prior_it);
    }

    // The tails published under `prior`, least recently used first.
    std::span<const Tail> Tails(const ContentHash& prior) const {
        const auto it = tails_by_prior_.find(prior);
        return it == tails_by_prior_.end() ? std::span<const Tail>{} : std::span<const Tail>{it->second};
    }

    // The tail offset a hash was published with; nullopt for a chain hash.
    std::optional<std::int32_t> TailTokensOf(const ContentHash& hash) const {
        const auto prior_it = prior_of_.find(hash);
        if (prior_it == prior_of_.end()) {
            return std::nullopt;
        }
        for (const Tail& tail : tails_by_prior_.at(prior_it->second)) {
            if (tail.hash == hash) {
                return tail.tail_tokens;
            }
        }
        FatalCheck(false, "a recorded tail hash is missing from its prior's list");
        return std::nullopt;
    }

    bool Contains(const ContentHash& hash) const { return prior_of_.contains(hash); }
    std::size_t Size() const { return prior_of_.size(); }
    std::int64_t CapEvictions() const { return cap_evictions_; }

    void Clear() {
        tails_by_prior_.clear();
        prior_of_.clear();
    }

private:
    std::unordered_map<ContentHash, std::vector<Tail>> tails_by_prior_;
    std::unordered_map<ContentHash, ContentHash> prior_of_;
    std::int64_t cap_evictions_{0};
};

}  // namespace tokenspeed
