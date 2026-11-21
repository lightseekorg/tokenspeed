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

#include "cache/tier/disk_manager.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

#include "utils.h"

namespace tokenspeed {

void DiskConfig::Validate() const {
    if (max_operations <= 0 || max_blocks_per_operation <= 0 || max_host_blocks <= 0 || max_write_host_blocks <= 0 ||
        max_queued_cleanings <= 0 || min_prefetch_tokens <= 0 || wait_timeout_us <= 0 || ready_timeout_us <= 0) {
        throw std::invalid_argument("Disk tier limits and deadlines must be positive");
    }
    if (max_write_host_blocks > max_host_blocks) {
        throw std::invalid_argument("Disk tier write credits must not exceed its Host credits");
    }
}

DiskTransferManager::DiskTransferManager(CacheCoordinator& coordinator, DiskConfig config)
    : coordinator_{coordinator}, config_{config} {
    config_.Validate();
    FatalCheck(coordinator_.Disk().HasPool(), "the Disk tier's transfers need a Disk pool");
}

std::uint64_t DiskTransferManager::nextId() {
    if (next_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("Disk operation ids exhausted");
    }
    return next_id_++;
}

bool DiskTransferManager::HasProgressPending() const {
    return std::ranges::any_of(operations_, [](const auto& entry) { return !entry.second.completed; });
}

std::size_t DiskTransferManager::NumHeldHostBlocks() const {
    return heldHostBlocks(DiskOpKind::kRead) + heldHostBlocks(DiskOpKind::kWrite);
}

std::size_t DiskTransferManager::heldHostBlocks(DiskOpKind kind) const {
    std::size_t count = 0;
    for (const auto& [id, ticket] : operations_) {
        if (ticket.kind == kind) count += ticket.host_blocks.size();
    }
    return count;
}

std::vector<const CacheBlockRef*> DiskTransferManager::HeldHostBlockRefs() const {
    std::vector<const CacheBlockRef*> refs;
    for (const auto& [id, ticket] : operations_) {
        for (const CacheBlockRef& block : ticket.host_blocks) {
            if (block) refs.push_back(&block);
        }
    }
    return refs;
}

bool DiskTransferManager::Completed(std::uint64_t id) const {
    const auto it = operations_.find(id);
    return it == operations_.end() || it->second.completed;
}

DiskOperation DiskTransferManager::command(std::uint64_t id, const Ticket& ticket) const {
    DiskOperation op{.kind = ticket.kind, .op_id = id};
    for (std::size_t i = 0; i < ticket.keys.size(); ++i) {
        const auto group = static_cast<std::int32_t>(ticket.keys[i].group_id);
        op.group_ids.push_back(ticket.keys[i].group_id);
        op.host_blocks.push_back(coordinator_.Allocator(group).ResolveCacheBlockId(*ticket.host_blocks[i]));
        op.disk_blocks.push_back(coordinator_.Allocator(group).ResolveCacheBlockId(*ticket.disk_blocks[i]));
    }
    return op;
}

StorageLease DiskTransferManager::StartRead(const PromptPrefix& prompt) {
    if (operations_.size() >= static_cast<std::size_t>(config_.max_operations)) return {};
    const std::size_t held = NumHeldHostBlocks();
    const auto limit = static_cast<std::size_t>(config_.max_host_blocks);
    if (held >= limit) return {};
    const std::vector<CacheKey> hits = coordinator_.Disk().PrefixHits(prompt);
    if (hits.empty()) return {};
    // The Disk's keys stand in as Host hits for the Host-suffix probe; every
    // key the boundary needs and the Host lacks is then one the Disk holds.
    StorageReservation reservation = coordinator_.Storage().ReserveStoragePrefix(
        prompt, hits, std::min(limit - held, static_cast<std::size_t>(config_.max_blocks_per_operation)));
    if (reservation.keys.empty()) return {};
    Ticket ticket{.kind = DiskOpKind::kRead,
                  .keys = std::move(reservation.keys),
                  .host_blocks = std::move(reservation.blocks),
                  .tail_prior = std::move(reservation.tail_prior),
                  .tail = std::move(reservation.tail)};
    ticket.disk_blocks.reserve(ticket.keys.size());
    for (const CacheKey& key : ticket.keys) {
        CacheBlockRef source = coordinator_.Disk().AcquireDiskCachedBlock(key);
        // The reservation evicts Host entries only.
        FatalCheck(static_cast<bool>(source), "a prefetch's key left the Disk while its Host blocks were reserved");
        ticket.disk_blocks.push_back(std::move(source));
    }
    const std::uint64_t id = nextId();
    commands_.push_back(command(id, ticket));
    operations_.emplace(id, std::move(ticket));
    return StorageLease{*this, id};
}

void DiskTransferManager::QueueCleaning(std::span<const CacheKey> keys) {
    for (const CacheKey& key : keys) {
        if (cleaning_.size() == static_cast<std::size_t>(config_.max_queued_cleanings)) {
            cleaning_.pop_front();
            ++dropped_cleanings_;
        }
        cleaning_.push_back(key);
    }
}

void DiskTransferManager::DropCleaning() {
    dropped_cleanings_ += cleaning_.size();
    cleaning_.clear();
}

void DiskTransferManager::StartCleaning() {
    if (cleaning_.empty() || operations_.size() >= static_cast<std::size_t>(config_.max_operations)) return;
    const std::size_t held = NumHeldHostBlocks();
    const std::size_t write_held = heldHostBlocks(DiskOpKind::kWrite);
    const auto limit = static_cast<std::size_t>(config_.max_host_blocks);
    const auto write_limit = static_cast<std::size_t>(config_.max_write_host_blocks);
    if (held >= limit || write_held >= write_limit) return;
    const std::size_t credits =
        std::min({limit - held, write_limit - write_held, static_cast<std::size_t>(config_.max_blocks_per_operation)});

    CacheStore& store = coordinator_.Store();
    const BlockPool& host_pool = *store.HostPool();
    std::unordered_set<CacheKey, CacheKeyHash> writing;
    for (const auto& [id, ticket] : operations_) {
        if (ticket.kind == DiskOpKind::kWrite) writing.insert(ticket.keys.begin(), ticket.keys.end());
    }
    const auto dirty = [&](const CacheKey& key) {
        return coordinator_.Host().ContainsHostCachedBlock(key) && !coordinator_.Disk().ContainsDiskCachedBlock(key) &&
               !writing.contains(key);
    };
    // The dirty Host entries of a boundary node, every group's.
    const auto dirty_keys_of = [&](const BoundaryTree::Node& node) {
        std::vector<CacheKey> keys;
        for (const CacheStore::NodeEntry& entry : store.NodeEntries(host_pool, node)) {
            std::optional<CacheKey> key = store.Groups()[entry.group_id].Index().KeyOf(host_pool, entry.location);
            if (key && dirty(*key)) keys.push_back(std::move(*key));
        }
        return keys;
    };

    Ticket ticket{.kind = DiskOpKind::kWrite};
    std::unordered_set<CacheKey, CacheKeyHash> chosen;
    while (!cleaning_.empty() && ticket.keys.size() < credits) {
        const CacheKey front = cleaning_.front();
        const BoundaryTree::Node* node = store.Boundaries().Find(front.content_hash);
        if (node == nullptr || !dirty(front) || chosen.contains(front)) {
            cleaning_.pop_front();
            continue;
        }
        // The pages above it first, nearest the root first, so a cleaned page
        // never waits on the Disk for a parent only the Host holds: a Host
        // placement dropping that parent would leave it unresumable. The
        // climb stops at a page on the Disk, whose own parents cleaning
        // wrote before it, or one the Host does not hold.
        std::vector<const BoundaryTree::Node*> chain{node};
        for (const BoundaryTree::Node* above = node->parent; above != nullptr && above->OnHost() && !above->OnDisk();
             above = above->parent) {
            chain.push_back(above);
        }
        std::vector<CacheKey> batch;
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            for (CacheKey& key : dirty_keys_of(**it)) {
                if (!chosen.contains(key)) batch.push_back(std::move(key));
            }
        }
        if (!ticket.keys.empty() && ticket.keys.size() + batch.size() > credits) break;
        cleaning_.pop_front();
        if (batch.size() > credits) {
            // A chain longer than one operation: its root-most part now, the
            // rest when the key comes up again.
            batch.resize(credits);
            cleaning_.push_front(front);
        }
        for (CacheKey& key : batch) {
            chosen.insert(key);
            ticket.keys.push_back(std::move(key));
        }
    }
    if (ticket.keys.empty()) return;

    std::vector<std::uint32_t> groups;
    std::vector<std::int32_t> buckets;
    for (const CacheKey& key : ticket.keys) {
        CacheBlockRef source = store.AcquireHostCachedBlock(key);
        FatalCheck(static_cast<bool>(source), "a dirty Host entry has a block");
        const auto group = static_cast<std::int32_t>(key.group_id);
        groups.push_back(key.group_id);
        // The Disk copy lives in its Host source's bucket (same owner).
        buckets.push_back(coordinator_.Allocator(group).BucketOf(source->Location()));
        ticket.metadata.push_back(*store.Groups()[key.group_id].Index().MetadataFor(host_pool, source->Location()));
        ticket.host_blocks.push_back(std::move(source));
    }
    DiskCache::DiskAllocationBatch allocation = coordinator_.Disk().AcquireDiskBlocks(groups, buckets);
    // A row the Disk cannot place is not cleaned now.
    std::size_t kept = 0;
    for (std::size_t row = 0; row < ticket.keys.size(); ++row) {
        if (!allocation.blocks[row]) {
            ++dropped_cleanings_;
            continue;
        }
        ticket.keys[kept] = std::move(ticket.keys[row]);
        ticket.host_blocks[kept] = std::move(ticket.host_blocks[row]);
        ticket.metadata[kept] = std::move(ticket.metadata[row]);
        ticket.disk_blocks.push_back(std::move(allocation.blocks[row]));
        ++kept;
    }
    ticket.keys.resize(kept);
    ticket.host_blocks.resize(kept);
    ticket.metadata.resize(kept);
    if (kept == 0) return;
    const std::uint64_t id = nextId();
    commands_.push_back(command(id, ticket));
    operations_.emplace(id, std::move(ticket));
}

void DiskTransferManager::Complete(const cache::DiskDone& event) {
    const auto it = operations_.find(event.op_id);
    if (it == operations_.end() || it->second.completed) {
        throw std::invalid_argument("a Disk completion names no pending operation");
    }
    Ticket& ticket = it->second;
    ticket.completed = true;
    if (ticket.kind == DiskOpKind::kRead) {
        // The sources go first: a held entry is not dropped.
        ticket.disk_blocks.clear();
        if (event.success) {
            read_blocks_ += ticket.keys.size();
            if (!ticket.detached) {
                for (std::size_t i = 0; i < ticket.keys.size(); ++i) {
                    coordinator_.Host().CacheHostBlock(ticket.host_blocks[i], ticket.keys[i],
                                                       /*inherited_metadata=*/std::nullopt);
                }
                if (ticket.tail) {
                    coordinator_.Host().CacheHostTail(ticket.tail_prior, *ticket.tail);
                }
            }
        } else {
            ++failed_reads_;
            spdlog::warn("[Disk] read op_id={} of {} blocks failed; dropping their Disk entries", event.op_id,
                         ticket.keys.size());
            for (const CacheKey& key : ticket.keys) coordinator_.Disk().DropUnreadableEntry(key);
            ticket.host_blocks.clear();
        }
        // A request still holds the published Host entries until its
        // admission acquires them (the Ready hold); one that gave up let go.
        if (ticket.detached) operations_.erase(it);
        return;
    }
    if (event.success) {
        written_blocks_ += ticket.keys.size();
        for (std::size_t i = 0; i < ticket.keys.size(); ++i) {
            // A Host entry still there still describes these bytes.
            if (coordinator_.Host().ContainsHostCachedBlock(ticket.keys[i])) {
                coordinator_.Disk().CacheDiskBlock(ticket.disk_blocks[i], ticket.keys[i], ticket.metadata[i]);
            }
        }
    } else {
        ++failed_writes_;
        spdlog::warn("[Disk] write op_id={} of {} blocks failed", event.op_id, ticket.keys.size());
    }
    operations_.erase(it);
}

void DiskTransferManager::detach(std::uint64_t id) noexcept {
    const auto it = operations_.find(id);
    FatalCheck(it != operations_.end() && !it->second.detached, "a Disk lease names an attached operation");
    Ticket& ticket = it->second;
    ticket.detached = true;
    if (ticket.completed) {
        operations_.erase(it);
        return;
    }
    // Nothing has seen a command still in this scheduler-owned outbox. One
    // already issued keeps its blocks until it retires.
    const auto queued = std::ranges::find_if(commands_, [id](const DiskOperation& op) { return op.op_id == id; });
    if (queued != commands_.end()) {
        commands_.erase(queued);
        operations_.erase(it);
    }
}

std::vector<DiskOperation> DiskTransferManager::TakeCommands() {
    return std::exchange(commands_, {});
}

}  // namespace tokenspeed
