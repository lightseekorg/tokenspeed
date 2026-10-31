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

#include "cache/tier/storage_manager.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tokenspeed {

void StoragePressure::Validate() const {
    if (queued_operations < 0 || queued_bytes < 0 || oldest_operation_us < 0) {
        throw std::invalid_argument("L3 pressure observations cannot be negative");
    }
}

void StorageConfig::Validate() const {
    if (max_operations <= 0 || max_keys_per_operation <= 0 || max_host_blocks <= 0 || metadata_entries <= 0 ||
        max_prefetch_tokens <= 0 || min_prefetch_tokens <= 0 || min_prefetch_tokens > max_prefetch_tokens ||
        wait_timeout_us <= 0 || ready_timeout_us <= 0) {
        throw std::invalid_argument("L3 limits and deadlines must be positive and the minimum must fit the maximum");
    }
    pressure_low.Validate();
    if (pressure_high.queued_operations <= pressure_low.queued_operations ||
        pressure_high.queued_bytes <= pressure_low.queued_bytes ||
        pressure_high.oldest_operation_us <= pressure_low.oldest_operation_us) {
        throw std::invalid_argument("L3 pressure high watermarks must exceed nonnegative low watermarks");
    }
}

StorageTransferManager::StorageTransferManager(CacheCoordinator& coordinator, StorageConfig config)
    : coordinator_{coordinator}, config_{config} {
    config_.Validate();
}

std::uint64_t StorageTransferManager::nextId() {
    if (next_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("L3 operation ids exhausted");
    }
    return next_id_++;
}

bool StorageTransferManager::CanStart() const {
    return !pressure_gate_closed_ && operations_.size() < static_cast<std::size_t>(config_.max_operations) &&
           NumQuarantined() == 0;
}

void StorageTransferManager::UpdatePressure(const StoragePressure& pressure) {
    pressure.Validate();
    if (pressure_gate_closed_) {
        const auto& low = config_.pressure_low;
        if (pressure.queued_operations <= low.queued_operations && pressure.queued_bytes <= low.queued_bytes &&
            pressure.oldest_operation_us <= low.oldest_operation_us) {
            pressure_gate_closed_ = false;
        }
    } else {
        const auto& high = config_.pressure_high;
        pressure_gate_closed_ = pressure.queued_operations >= high.queued_operations ||
                                pressure.queued_bytes >= high.queued_bytes ||
                                pressure.oldest_operation_us >= high.oldest_operation_us;
    }
    if (pressure_gate_closed_) {
        pressure_dropped_backup_keys_ += backup_candidates_.size();
        backup_candidates_.clear();
    }
}

std::size_t StorageTransferManager::NumHeldBlocks() const {
    std::size_t count = 0;
    for (const auto& [id, ticket] : operations_) {
        count += ticket.blocks.size();
    }
    return count;
}

std::size_t StorageTransferManager::NumQuarantined() const {
    return std::ranges::count_if(operations_, [](const auto& entry) { return entry.second.quarantined; });
}

std::size_t StorageTransferManager::availableHostCredits() const {
    const auto used = NumHeldBlocks();
    const auto limit = static_cast<std::size_t>(config_.max_host_blocks);
    return used < limit ? limit - used : 0;
}

std::uint64_t StorageTransferManager::StartQuery(std::vector<CacheKey> keys) {
    if (keys.empty() || !CanStart() || availableHostCredits() == 0 ||
        keys.size() > static_cast<std::size_t>(config_.max_keys_per_operation)) {
        return 0;
    }
    StorageQueryOperation op{nextId(), std::move(keys)};
    const auto id = op.op_id;
    operations_.emplace(id, Ticket{decltype(op)::kKind, op.keys, {}, {}});
    commands_.push_back(std::move(op));
    return id;
}

bool StorageTransferManager::Completed(std::uint64_t id) const {
    const auto it = operations_.find(id);
    return it == operations_.end() || it->second.completed;
}

bool StorageTransferManager::IsQuery(std::uint64_t id) const {
    const auto it = operations_.find(id);
    return it != operations_.end() && it->second.kind == StorageKind::kQuery;
}

std::vector<CacheKey> StorageTransferManager::QueryHits(std::uint64_t id) const {
    const Ticket& ticket = operations_.at(id);
    std::vector<CacheKey> keys;
    for (std::size_t i = 0; i < ticket.success.size(); ++i) {
        if (ticket.success[i] && !IsNegative(ticket.keys[i])) keys.push_back(ticket.keys[i]);
    }
    return keys;
}

std::uint64_t StorageTransferManager::StartRead(const CacheCoordinator::PromptPrefix& prompt,
                                                std::span<const CacheKey> hits) {
    if (!CanStart() || hits.empty()) return 0;
    auto reservation = coordinator_.ReserveStoragePrefix(prompt, hits, availableHostCredits());
    if (reservation.keys.empty()) return 0;
    StorageGetOperation op{nextId(), reservation.keys, {}};
    for (std::size_t i = 0; i < reservation.blocks.size(); ++i) {
        op.host_blocks.push_back(coordinator_.Allocator(static_cast<std::int32_t>(op.keys[i].group_id))
                                     .ResolveCacheBlockId(*reservation.blocks[i]));
    }
    const auto id = op.op_id;
    operations_.emplace(id, Ticket{decltype(op)::kKind,
                                   op.keys,
                                   std::move(reservation.blocks),
                                   {},
                                   std::move(reservation.tail_prior),
                                   std::move(reservation.tail)});
    commands_.push_back(std::move(op));
    return id;
}

void StorageTransferManager::rememberNegative(const CacheKey& key) {
    if (!negative_.insert(key).second) return;
    negative_order_.push_back(key);
    while (negative_order_.size() > static_cast<std::size_t>(config_.metadata_entries)) {
        negative_.erase(negative_order_.front());
        negative_order_.pop_front();
    }
}

void StorageTransferManager::Complete(const cache::StorageDone& event) {
    const auto it = operations_.find(event.op_id);
    if (it == operations_.end() || it->second.completed) return;
    Ticket& ticket = it->second;
    if (event.success.size() != ticket.keys.size() || event.object_failures.size() != event.success.size()) {
        throw std::invalid_argument("L3 completion size differs from its immutable operation");
    }
    for (std::size_t i = 0; i < event.success.size(); ++i) {
        if (event.success[i] > 1 || event.object_failures[i] > 1 || (event.success[i] && event.object_failures[i]) ||
            (ticket.kind != StorageKind::kGet && event.object_failures[i])) {
            throw std::invalid_argument("Invalid L3 object completion facts");
        }
    }
    ticket.success = event.success;
    ticket.completed = true;
    ticket.quarantined = !event.retired;
    const auto kind = ticket.kind;
    if (kind == StorageKind::kGet) {
        for (std::size_t i = 0; i < ticket.success.size(); ++i) {
            if (!ticket.detached && event.object_failures[i]) {
                rememberNegative(ticket.keys[i]);
            }
            if (event.retired && !ticket.detached && ticket.success[i]) {
                coordinator_.CacheHostBlock(ticket.blocks[i], ticket.keys[i], /*inherited_metadata=*/std::nullopt);
            }
        }
        if (event.retired && !ticket.detached && ticket.tail) {
            coordinator_.CacheHostTail(ticket.tail_prior, *ticket.tail);
        }
    }
    if (!event.retired) return;  // Quarantine is charged until a process/transport teardown proves safety.
    if (ticket.detached || kind == StorageKind::kPut) operations_.erase(it);
}

void StorageTransferManager::Detach(std::uint64_t id) {
    auto it = operations_.find(id);
    if (it == operations_.end()) return;
    Ticket& ticket = it->second;
    if (ticket.detached) return;
    ticket.detached = true;
    if (ticket.quarantined) return;
    if (ticket.completed) {
        operations_.erase(it);
        return;
    }
    // Nothing has seen a command still in this scheduler-owned outbox.
    const auto queued = std::ranges::find_if(commands_, [id](const auto& operation) {
        return !std::holds_alternative<StorageCancelOperation>(operation) &&
               std::visit([id](const auto& op) { return op.op_id == id; }, operation);
    });
    if (queued != commands_.end()) {
        commands_.erase(queued);
        operations_.erase(it);
    } else {
        commands_.push_back(StorageCancelOperation{id});
    }
}

void StorageTransferManager::QueueBackups(std::span<const CacheKey> keys) {
    // A permanently zero write budget cannot drain, even after every read retires.
    if (config_.max_operations < 2 || config_.max_host_blocks < 2) {
        disabled_dropped_backup_keys_ += keys.size();
        return;
    }
    if (pressure_gate_closed_) {
        pressure_dropped_backup_keys_ += keys.size();
        return;
    }
    for (const auto& key : keys) {
        if (backup_candidates_.size() == static_cast<std::size_t>(config_.metadata_entries)) {
            backup_candidates_.pop_front();
            ++overflow_dropped_backup_keys_;
        }
        backup_candidates_.push_back(key);
    }
}

void StorageTransferManager::StartBackups() {
    if (!CanStart()) return;
    std::size_t backup_operations = 0;
    std::size_t backup_blocks = 0;
    std::unordered_set<CacheKey, CacheKeyHash> pending;
    for (const auto& [id, ticket] : operations_) {
        if (ticket.kind == StorageKind::kPut) {
            ++backup_operations;
            backup_blocks += ticket.blocks.size();
            pending.insert(ticket.keys.begin(), ticket.keys.end());
        }
    }
    // Backups are best-effort. Even a blocked PUT must leave capacity for
    // queries and prefetch; readers may use the entire shared budget.
    if (backup_operations >= static_cast<std::size_t>(config_.max_operations / 2)) return;
    const auto backup_limit = static_cast<std::size_t>(config_.max_host_blocks / 2);
    const auto backup_available = backup_blocks < backup_limit ? backup_limit - backup_blocks : 0;
    const auto limit =
        std::min({availableHostCredits(), backup_available, static_cast<std::size_t>(config_.max_keys_per_operation)});
    // Credits held by other tickets are temporary. Candidates own no Host
    // references; retain them within the queue bound and recheck residency later.
    if (limit == 0) return;
    StoragePutOperation op{0, {}, {}};
    std::vector<CacheBlockRef> refs;
    while (!backup_candidates_.empty() && refs.size() < limit) {
        CacheKey key = std::move(backup_candidates_.front());
        backup_candidates_.pop_front();
        if (!pending.insert(key).second) continue;
        CacheBlockRef ref = coordinator_.AcquireHostCachedBlock(key);
        if (!ref) {
            ++unavailable_dropped_backup_keys_;
            continue;
        }
        op.host_blocks.push_back(
            coordinator_.Allocator(static_cast<std::int32_t>(key.group_id)).ResolveCacheBlockId(*ref));
        op.keys.push_back(std::move(key));
        refs.push_back(std::move(ref));
    }
    if (refs.empty()) return;
    op.op_id = nextId();
    operations_.emplace(op.op_id, Ticket{decltype(op)::kKind, op.keys, std::move(refs), {}});
    commands_.push_back(std::move(op));
}

void StorageTransferManager::DropBackups() {
    cleanup_dropped_backup_keys_ += backup_candidates_.size();
    backup_candidates_.clear();
}

void StorageTransferManager::ClearMetadata() {
    if (HasPending()) throw std::logic_error("cannot clear L3 metadata while operations hold resources");
    DropBackups();
    negative_.clear();
    negative_order_.clear();
}

std::vector<StorageOperation> StorageTransferManager::TakeCommands() {
    return std::exchange(commands_, {});
}
}  // namespace tokenspeed
