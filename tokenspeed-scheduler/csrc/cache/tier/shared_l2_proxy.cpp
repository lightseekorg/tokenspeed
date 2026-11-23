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
#include <utility>

#include "utils.h"

namespace tokenspeed {

SharedL2Proxy::SharedL2Proxy(CacheCoordinator& coordinator) : coordinator_{coordinator} {
    _assert(coordinator_.Host().HasPool(), "shared L2 needs its proxy Host tier");
}

std::vector<SharedL2ProxyBlock> SharedL2Proxy::MirrorAnswer(const SharedL2Answer& answer, const ContentHash& tail_prior,
                                                            std::vector<CacheBlockRef>& hit_blocks) {
    if (answer.keys.size() != answer.pool_pages.size() || answer.keys.size() != answer.buckets.size()) {
        throw std::invalid_argument("a shared L2 answer names one bucket and one page per key");
    }
    if ((answer.ticket == 0) != answer.keys.empty()) {
        throw std::invalid_argument("a shared L2 answer pins the pages it names under a ticket");
    }
    for (std::size_t i = 0; i < answer.keys.size(); ++i) {
        const CacheKey& key = answer.keys[i];
        if (key.namespace_id != kDefaultCacheNamespaceId || key.content_hash.empty() ||
            key.group_id >= static_cast<std::uint32_t>(coordinator_.NumGroups()) || answer.buckets[i] < 0 ||
            answer.buckets[i] >= coordinator_.Allocator(static_cast<std::int32_t>(key.group_id)).ShardCount()) {
            throw std::invalid_argument("a shared L2 answer names keys of the rank's cache groups and their buckets");
        }
    }
    if (answer.tail && (answer.tail->tail_tokens <= 0 || answer.tail->tail_tokens >= coordinator_.PrefixGranularity() ||
                        answer.tail->hash.empty())) {
        throw std::invalid_argument("a shared L2 answer's tail lies strictly inside a page, under its hash");
    }
    // The answer is taken from here on: the checks above cover what the
    // coordinator calls below would refuse.
    std::vector<SharedL2ProxyBlock> created;
    if (answer.keys.empty()) {
        return created;
    }
    // A key a proxy entry stands for already is pinned at the owner by the
    // ticket that entry was created under, which stays until the entry goes.
    std::vector<CacheBlockRef> shared;
    std::vector<std::size_t> fresh;
    std::vector<std::uint32_t> groups;
    std::vector<std::int32_t> buckets;
    for (std::size_t i = 0; i < answer.keys.size(); ++i) {
        if (CacheBlockRef block = coordinator_.Store().AcquireHostCachedBlock(answer.keys[i])) {
            shared.push_back(std::move(block));
            continue;
        }
        fresh.push_back(i);
        groups.push_back(answer.keys[i].group_id);
        buckets.push_back(answer.buckets[i]);
    }
    HostCache::HostAllocationBatch batch = coordinator_.Host().AcquireHostBlocks(groups, buckets);
    // The proxy holds at most what the pool pins, and it is sized past the
    // pool.
    FatalCheck(batch.stats.unallocated == 0, "the shared L2 proxy Host tier is full");
    TicketEntries entries{.ticket = answer.ticket};
    created.reserve(fresh.size());
    for (std::size_t j = 0; j < fresh.size(); ++j) {
        const CacheKey& key = answer.keys[fresh[j]];
        CacheBlockRef& block = batch.blocks[j];
        coordinator_.Host().CacheHostBlock(block, key, /*inherited_metadata=*/std::nullopt);
        created.push_back(SharedL2ProxyBlock{
            .group_id = key.group_id,
            .host_block = coordinator_.Allocator(static_cast<std::int32_t>(key.group_id)).ResolveCacheBlockId(*block),
            .bucket = answer.buckets[fresh[j]],
            .pool_page = answer.pool_pages[fresh[j]],
        });
        entries.keys.push_back(key);
    }
    if (answer.tail) {
        coordinator_.Host().CacheHostTail(tail_prior, *answer.tail);
    }
    hit_blocks = std::move(batch.blocks);
    hit_blocks.insert(hit_blocks.end(), std::make_move_iterator(shared.begin()), std::make_move_iterator(shared.end()));
    // A ticket without an entry (every page was pinned under earlier
    // tickets already) is released by the next plan's operations.
    tickets_.push_back(std::move(entries));
    return created;
}

std::vector<SharedL2Operation> SharedL2Proxy::NextOperations(std::vector<CacheKey> device_hits) {
    std::vector<SharedL2Operation> operations;
    if (!device_hits.empty()) {
        operations.push_back(SharedL2TouchOperation{std::move(device_hits)});
    }
    for (TicketEntries& entries : tickets_) {
        // From the answer's last entry back. One still held -- by a waiting
        // prompt's lookup, or by the load-back reading it -- ends the pass
        // for its ticket, so a ticket in use costs one check per plan.
        while (!entries.keys.empty() && coordinator_.Host().TryEvictHostEntry(entries.keys.back())) {
            entries.keys.pop_back();
        }
        if (entries.keys.empty()) {
            operations.push_back(SharedL2ReleaseOperation{entries.ticket});
        }
    }
    std::erase_if(tickets_, [](const TicketEntries& entries) { return entries.keys.empty(); });
    return operations;
}

}  // namespace tokenspeed
