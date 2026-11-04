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

#include <deque>
#include <map>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/tier/storage.h"
#include "cache/tier/storage_lease.h"

namespace tokenspeed {

// Owns Host references, never request scheduling or SDK calls. All methods run
// on the scheduler thread. Runtime workers receive only immutable descriptors.
// A query or GET is started for a request and returned as its StorageLease;
// backups (PUT) belong to no request and retire on their own.
class StorageTransferManager {
public:
    StorageTransferManager(CacheCoordinator& coordinator, StorageConfig config);
    const StorageConfig& Config() const { return config_; }
    bool CanStart() const;
    void UpdatePressure(const StoragePressure& pressure);
    bool PressureGateClosed() const { return pressure_gate_closed_; }
    std::uint64_t NumPressureDroppedBackups() const { return pressure_dropped_backup_keys_; }
    std::uint64_t NumOverflowDroppedBackups() const { return overflow_dropped_backup_keys_; }
    std::uint64_t NumUnavailableDroppedBackups() const { return unavailable_dropped_backup_keys_; }
    std::uint64_t NumDisabledDroppedBackups() const { return disabled_dropped_backup_keys_; }
    std::uint64_t NumCleanupDroppedBackups() const { return cleanup_dropped_backup_keys_; }
    bool HasPending() const { return !operations_.empty(); }
    std::size_t NumOperations() const { return operations_.size(); }
    std::size_t NumHeldBlocks() const;
    std::size_t NumQuarantined() const;
    std::size_t NumNegativeKeys() const { return negative_.size(); }
    std::size_t NumBackupCandidates() const { return backup_candidates_.size(); }
    bool IsNegative(const CacheKey& key) const { return negative_.contains(key); }
    // Returns an empty lease when nothing was started.
    StorageLease StartQuery(std::vector<CacheKey> keys, std::vector<std::int32_t> key_tokens);
    bool Completed(std::uint64_t id) const;
    bool IsQuery(std::uint64_t id) const;
    // Successful query hits, and the cumulative tokens they cover. Each hit counts once per operation.
    std::vector<CacheKey> QueryHits(std::uint64_t id);
    std::uint64_t QueryHitTokens() const { return query_hit_tokens_; }
    // Returns an empty lease when nothing was started.
    StorageLease StartRead(const CacheCoordinator::PromptPrefix& prompt, std::span<const CacheKey> hits);
    void Complete(const cache::StorageDone& event);
    void QueueBackups(std::span<const CacheKey> keys);
    void StartBackups();
    void DropBackups();
    void ClearMetadata();
    std::vector<StorageOperation> TakeCommands();

private:
    friend class StorageLease;

    struct Ticket {
        StorageKind kind;
        std::vector<CacheKey> keys;
        // Query tickets: tokens each key covers, parallel to `keys`.
        std::vector<std::int32_t> key_tokens;
        std::vector<CacheBlockRef> blocks;
        std::vector<std::uint8_t> success;
        ContentHash tail_prior;
        std::optional<TailPublication> tail;
        bool completed{false};
        bool detached{false};
        bool quarantined{false};
    };
    std::uint64_t nextId();
    // What dropping the operation's lease does.
    void detach(std::uint64_t id) noexcept;
    void rememberNegative(const CacheKey& key);
    std::size_t availableHostCredits() const;
    CacheCoordinator& coordinator_;
    StorageConfig config_;
    std::map<std::uint64_t, Ticket> operations_;
    std::deque<CacheKey> backup_candidates_;
    std::unordered_set<CacheKey, CacheKeyHash> negative_;
    std::deque<CacheKey> negative_order_;
    std::vector<StorageOperation> commands_;
    std::uint64_t next_id_{1};
    bool pressure_gate_closed_{false};
    std::uint64_t pressure_dropped_backup_keys_{0};
    std::uint64_t overflow_dropped_backup_keys_{0};
    std::uint64_t unavailable_dropped_backup_keys_{0};
    std::uint64_t disabled_dropped_backup_keys_{0};
    std::uint64_t cleanup_dropped_backup_keys_{0};
    // Cumulative tokens of keys storage queries found (L3 hits).
    std::uint64_t query_hit_tokens_{0};
};
}  // namespace tokenspeed
