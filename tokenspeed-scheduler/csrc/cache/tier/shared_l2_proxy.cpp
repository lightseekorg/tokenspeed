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

#include "cache/tier/shared_l2_proxy.h"

#include <stdexcept>

#include <spdlog/spdlog.h>

#include "utils.h"

namespace tokenspeed {

SharedL2LookupHold::SharedL2LookupHold(SharedL2LookupHold&& other) noexcept
    : proxy_{std::exchange(other.proxy_, nullptr)},
      lookup_id_{std::exchange(other.lookup_id_, 0)},
      blocks_{std::move(other.blocks_)} {}

SharedL2LookupHold& SharedL2LookupHold::operator=(SharedL2LookupHold&& other) noexcept {
    if (this != &other) {
        reset();
        proxy_ = std::exchange(other.proxy_, nullptr);
        lookup_id_ = std::exchange(other.lookup_id_, 0);
        blocks_ = std::move(other.blocks_);
    }
    return *this;
}

SharedL2LookupHold::~SharedL2LookupHold() {
    reset();
}

void SharedL2LookupHold::reset() noexcept {
    blocks_.clear();
    if (SharedL2Proxy* proxy = std::exchange(proxy_, nullptr)) {
        proxy->endLookup(std::exchange(lookup_id_, 0));
    }
}

SharedL2Proxy::SharedL2Proxy(CacheCoordinator& coordinator) : coordinator_{coordinator} {
    _assert(coordinator_.Host().HasPool(), "shared L2 needs its proxy Host tier");
}

SharedL2LookupHold SharedL2Proxy::StartLookup(const std::string& request_id) {
    const std::uint64_t lookup_id = ++next_lookup_id_;
    pending_.emplace(lookup_id, request_id);
    return SharedL2LookupHold{*this, lookup_id};
}

const std::string* SharedL2Proxy::PendingRequest(std::uint64_t lookup_id) const {
    const auto it = pending_.find(lookup_id);
    return it == pending_.end() ? nullptr : &it->second;
}

void SharedL2Proxy::endLookup(std::uint64_t lookup_id) noexcept {
    // Answered lookups left pending_ when their answer was applied.
    pending_.erase(lookup_id);
}

std::vector<SharedL2ProxyPage> SharedL2Proxy::ApplyAnswer(const SharedL2LookupResult& result,
                                                          const ContentHash& tail_prior, SharedL2LookupHold& hold) {
    _assert(hold.proxy_ == this && hold.lookup_id_ == result.lookup_id && pending_.contains(result.lookup_id),
            "a pool answer is applied to the pending lookup it answers");
    if (result.keys.size() != result.pages.size() || result.keys.size() != result.buckets.size()) {
        throw std::invalid_argument("a shared L2 answer names one bucket and one page per key");
    }
    if ((result.ticket == 0) != result.keys.empty()) {
        throw std::invalid_argument("a shared L2 answer pins the pages it names under a ticket");
    }
    for (std::size_t i = 0; i < result.keys.size(); ++i) {
        const CacheKey& key = result.keys[i];
        if (key.namespace_id != kDefaultCacheNamespaceId || key.content_hash.empty() ||
            key.group_id >= static_cast<std::uint32_t>(coordinator_.NumGroups()) || result.buckets[i] < 0 ||
            result.buckets[i] >= coordinator_.Allocator(static_cast<std::int32_t>(key.group_id)).ShardCount()) {
            throw std::invalid_argument("a shared L2 answer names keys of the rank's cache groups and their buckets");
        }
    }
    if (result.tail && (result.tail->tail_tokens <= 0 || result.tail->tail_tokens >= coordinator_.PrefixGranularity() ||
                        result.tail->hash.empty())) {
        throw std::invalid_argument("a shared L2 answer's tail lies strictly inside a page, under its hash");
    }
    // The answer is taken from here on: the checks above cover what the
    // coordinator calls below would refuse.
    pending_.erase(result.lookup_id);
    std::vector<SharedL2ProxyPage> created;
    if (result.keys.empty()) {
        return created;
    }
    // A key a proxy entry stands for already is pinned at the owner by the
    // ticket that entry was created under, which stays until the entry goes.
    std::vector<CacheBlockRef> shared;
    std::vector<std::size_t> fresh;
    std::vector<std::uint32_t> groups;
    std::vector<std::int32_t> buckets;
    for (std::size_t i = 0; i < result.keys.size(); ++i) {
        if (CacheBlockRef block = coordinator_.Store().AcquireHostCachedBlock(result.keys[i])) {
            shared.push_back(std::move(block));
            continue;
        }
        fresh.push_back(i);
        groups.push_back(result.keys[i].group_id);
        buckets.push_back(result.buckets[i]);
    }
    HostCache::HostAllocationBatch batch = coordinator_.Host().AcquireHostBlocks(groups, buckets);
    if (batch.stats.unallocated != 0) {
        // The proxy tier is sized past the pool, so only a mis-sized rank
        // gets here: the prompt computes its prefix instead.
        spdlog::warn("[SharedL2] lookup {}: the proxy Host tier has no room for {} of {} pool pages; taken as a miss",
                     result.lookup_id, batch.stats.unallocated, result.keys.size());
        commands_.push_back(SharedL2ReleaseOperation{result.ticket});
        return created;
    }
    TicketEntries ticket{.id = result.ticket};
    created.reserve(fresh.size());
    for (std::size_t j = 0; j < fresh.size(); ++j) {
        const CacheKey& key = result.keys[fresh[j]];
        CacheBlockRef& block = batch.blocks[j];
        coordinator_.Host().CacheHostBlock(block, key, /*inherited_metadata=*/std::nullopt);
        created.push_back(SharedL2ProxyPage{
            .group_id = key.group_id,
            .block = coordinator_.Allocator(static_cast<std::int32_t>(key.group_id)).ResolveCacheBlockId(*block),
            .bucket = result.buckets[fresh[j]],
            .page = result.pages[fresh[j]],
        });
        ticket.keys.push_back(key);
    }
    if (result.tail) {
        coordinator_.Host().CacheHostTail(tail_prior, *result.tail);
    }
    hold.blocks_ = std::move(batch.blocks);
    hold.blocks_.insert(hold.blocks_.end(), std::make_move_iterator(shared.begin()),
                        std::make_move_iterator(shared.end()));
    if (ticket.keys.empty()) {
        // Every page was pinned under earlier tickets already.
        commands_.push_back(SharedL2ReleaseOperation{result.ticket});
    } else {
        tickets_.push_back(std::move(ticket));
    }
    return created;
}

void SharedL2Proxy::Touch(std::vector<CacheKey> keys) {
    if (!keys.empty()) {
        commands_.push_back(SharedL2TouchOperation{std::move(keys)});
    }
}

void SharedL2Proxy::ReleaseUnheld() {
    for (TicketEntries& ticket : tickets_) {
        // From the answer's last entry back. One still held -- by a waiting
        // prompt's lookup, or by the load-back reading it -- ends the pass
        // for its ticket, so a ticket in use costs one check per plan.
        while (!ticket.keys.empty() && coordinator_.Host().DropHostEntry(ticket.keys.back())) {
            ticket.keys.pop_back();
        }
        if (ticket.keys.empty()) {
            commands_.push_back(SharedL2ReleaseOperation{ticket.id});
        }
    }
    std::erase_if(tickets_, [](const TicketEntries& ticket) { return ticket.keys.empty(); });
}

}  // namespace tokenspeed
