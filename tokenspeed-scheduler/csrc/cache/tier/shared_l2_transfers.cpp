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

#include "cache/tier/shared_l2_transfers.h"

#include <stdexcept>

#include "utils.h"

namespace tokenspeed {

void SharedL2Config::Validate() const {
    if (max_writes <= 0 || max_write_keys <= 0) {
        throw std::invalid_argument("shared L2 needs positive write limits");
    }
}

SharedL2Transfers::SharedL2Transfers(CacheCoordinator& coordinator, SharedL2Config config)
    : coordinator_{coordinator}, config_{config} {
    _assert(coordinator_.HasHostPool(), "shared L2 needs its proxy Host tier");
}

std::vector<SharedL2ProxyPage> SharedL2Transfers::ApplyLookup(const SharedL2LookupResult& result,
                                                              const ContentHash& prior_hash) {
    _assert(!lookups_.contains(result.request_id), "a request is looked up once");
    _assert(result.keys.size() == result.pages.size(), "a pool hit names one page per key");
    std::vector<std::size_t> fresh;
    std::vector<std::uint32_t> groups;
    for (std::size_t i = 0; i < result.keys.size(); ++i) {
        const auto it = proxy_.find(result.keys[i]);
        if (it == proxy_.end()) {
            fresh.push_back(i);
            groups.push_back(result.keys[i].group_id);
        } else {
            FatalCheck(it->second.page == result.pages[i], "the pool moved a pinned page");
        }
    }
    std::vector<SharedL2ProxyPage> created;
    if (!fresh.empty()) {
        _assert(result.ticket != 0, "a pool hit is pinned");
        const std::vector<std::int32_t> buckets(groups.size(), 0);
        CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostBlocks(groups, buckets);
        // The proxy holds at most what the pool pins, and it is sized past
        // the pool.
        FatalCheck(batch.stats.unallocated == 0, "the shared L2 proxy Host tier is full");
        created.reserve(fresh.size());
        for (std::size_t j = 0; j < fresh.size(); ++j) {
            const CacheKey& key = result.keys[fresh[j]];
            CacheBlockRef& block = batch.blocks[j];
            coordinator_.CacheHostBlock(block, key);
            const std::int32_t block_id =
                coordinator_.Allocator(static_cast<std::int32_t>(key.group_id)).ResolveCacheBlockId(block->Location());
            created.push_back(
                SharedL2ProxyPage{.group_id = key.group_id, .block = block_id, .page = result.pages[fresh[j]]});
            proxy_.emplace(
                key,
                ProxyEntry{
                    .ticket = result.ticket, .page = result.pages[fresh[j]], .block_id = block_id, .block = block});
            keys_by_block_.emplace(blockKey(key.group_id, block_id), key);
        }
        ticket_entries_[result.ticket] += static_cast<std::int32_t>(fresh.size());
    } else if (result.ticket != 0) {
        commands_.push_back(SharedL2ReleaseOperation{result.ticket});
    }
    if (result.tail) coordinator_.CacheHostTail(prior_hash, *result.tail);
    hold(result.keys);
    lookups_.emplace(result.request_id, result.keys);
    return created;
}

void SharedL2Transfers::ReleaseLookup(const std::string& request_id) {
    const auto it = lookups_.find(request_id);
    if (it == lookups_.end()) return;
    const std::vector<CacheKey> keys = std::move(it->second);
    lookups_.erase(it);
    release(keys);
}

void SharedL2Transfers::HoldLoad(const LoadBackOperation& load) {
    std::vector<CacheKey> keys;
    keys.reserve(load.transfers.size());
    for (const CacheTransfer& transfer : load.transfers) {
        const auto it = keys_by_block_.find(blockKey(transfer.group_id, transfer.source_page));
        FatalCheck(it != keys_by_block_.end(), "a shared L2 load-back reads outside the proxy");
        keys.push_back(it->second);
    }
    hold(keys);
    loads_.emplace(load.op_id, std::move(keys));
}

void SharedL2Transfers::ReleaseLoad(std::uint32_t op_id) {
    const auto it = loads_.find(op_id);
    _assert(it != loads_.end(), "a shared L2 load-back is held until its completion");
    const std::vector<CacheKey> keys = std::move(it->second);
    loads_.erase(it);
    release(keys);
}

void SharedL2Transfers::hold(std::span<const CacheKey> keys) {
    for (const CacheKey& key : keys) ++proxy_.at(key).holders;
}

void SharedL2Transfers::release(std::span<const CacheKey> keys) {
    for (const CacheKey& key : keys) {
        const auto it = proxy_.find(key);
        _assert(it != proxy_.end() && it->second.holders > 0, "a released proxy entry is held");
        if (--it->second.holders > 0) continue;
        const std::uint64_t ticket = it->second.ticket;
        keys_by_block_.erase(blockKey(key.group_id, it->second.block_id));
        proxy_.erase(it);
        coordinator_.DropHostEntry(key);
        const auto entries = ticket_entries_.find(ticket);
        if (--entries->second == 0) {
            ticket_entries_.erase(entries);
            commands_.push_back(SharedL2ReleaseOperation{ticket});
        }
    }
}

void SharedL2Transfers::Collect() {
    for (auto& published : coordinator_.TakePoolPublications()) {
        // A block loaded from the pool is there already.
        if (proxy_.contains(published.first)) continue;
        queued_writes_.push_back(std::move(published));
    }
    if (std::vector<CacheKey> touched = coordinator_.TakePoolTouches(); !touched.empty()) {
        commands_.push_back(SharedL2TouchOperation{std::move(touched)});
    }
}

void SharedL2Transfers::Publish(const CompletedPages& completed) {
    if (!completed.publish_state || (completed.prefix_hashes.empty() && !completed.tail)) return;
    publications_.push_back(SharedL2Publication{
        .tokens = {completed.tokens.begin(), completed.tokens.end()},
        .prefix_hashes = {completed.prefix_hashes.begin(), completed.prefix_hashes.end()},
        .tail = completed.tail,
    });
}

void SharedL2Transfers::StartWrites() {
    while (static_cast<std::int32_t>(writes_.size()) < config_.max_writes &&
           (!queued_writes_.empty() || !publications_.empty())) {
        SharedL2WriteOperation write{.op_id = nextOpId(), .publications = std::exchange(publications_, {})};
        std::vector<CacheBlockRef> sources;
        while (!queued_writes_.empty() && static_cast<std::int32_t>(write.keys.size()) < config_.max_write_keys) {
            auto [key, block] = std::move(queued_writes_.front());
            queued_writes_.pop_front();
            write.blocks.push_back(
                coordinator_.Allocator(static_cast<std::int32_t>(key.group_id)).ResolveCacheBlockId(block->Location()));
            write.keys.push_back(std::move(key));
            sources.push_back(std::move(block));
        }
        writes_.emplace(write.op_id, std::move(sources));
        commands_.push_back(std::move(write));
    }
}

void SharedL2Transfers::DropWrites() {
    queued_writes_.clear();
    publications_.clear();
    coordinator_.TakePoolPublications();
}

void SharedL2Transfers::Complete(const cache::SharedL2WriteDone& event) {
    if (writes_.erase(event.op_id) != 1) throw std::invalid_argument("completion of an unknown shared L2 write");
}

bool SharedL2Transfers::HasPending() const {
    return !writes_.empty() || !queued_writes_.empty() || !publications_.empty() || !commands_.empty();
}

}  // namespace tokenspeed
