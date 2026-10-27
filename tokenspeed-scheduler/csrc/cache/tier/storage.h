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

#include <cstdint>
#include <string>
#include <vector>
#include <variant>

#include "cache/core/cache_types.h"

namespace tokenspeed {

enum class StorageKind { kQuery, kGet, kPut, kCancel };

// Runtime facts, reduced by MAX across cache owners. Ages are durations, not
// timestamps from different hosts. Completed native work is excluded.
struct StoragePressure {
    std::int64_t queued_operations;
    std::int64_t queued_bytes;
    std::int64_t oldest_operation_us;
    void Validate() const;
};

// All limits are explicit at the runtime/scheduler boundary. Host credits count
// CacheBlock references conservatively: each can prevent one parent recycling.
struct StorageConfig {
    std::int32_t max_operations;
    std::int32_t max_keys_per_operation;
    std::int32_t max_host_blocks;
    std::int32_t metadata_entries;
    std::int32_t max_prefetch_tokens;
    std::int32_t min_prefetch_tokens;
    std::int64_t wait_timeout_us;
    std::int64_t ready_timeout_us;
    StoragePressure pressure_high;
    StoragePressure pressure_low;
    void Validate() const;
};

struct StorageQueryOperation {
    static constexpr StorageKind kKind = StorageKind::kQuery;
    std::uint64_t op_id;
    std::vector<CacheKey> keys;
};

struct StorageGetOperation {
    static constexpr StorageKind kKind = StorageKind::kGet;
    std::uint64_t op_id;
    std::vector<CacheKey> keys;
    std::vector<std::int32_t> host_blocks;  // Destinations held until native retirement.
};

struct StoragePutOperation {
    static constexpr StorageKind kKind = StorageKind::kPut;
    std::uint64_t op_id;
    std::vector<CacheKey> keys;
    std::vector<std::int32_t> host_blocks;  // Sources held until native retirement.
};

struct StorageCancelOperation {
    static constexpr StorageKind kKind = StorageKind::kCancel;
    std::uint64_t op_id;  // The original operation; no new ticket or completion.
};

using StorageOperation =
    std::variant<StorageQueryOperation, StorageGetOperation, StoragePutOperation, StorageCancelOperation>;

// Scheduler-owned observations. Runtime queue and transport counters live in
// the executor; these values describe logical leases and preparation instead.
struct StorageStats {
    std::uint64_t operations{0};
    std::uint64_t held_blocks{0};
    std::uint64_t quarantined{0};
    std::uint64_t negative_keys{0};
    std::uint64_t backup_candidates{0};
    std::uint64_t waiting_requests{0};
    std::uint64_t ready_requests{0};
    std::uint64_t wait_timeouts{0};
    std::uint64_t ready_expirations{0};
    bool pressure_gate_closed{false};
    std::uint64_t pressure_bypasses{0};
    std::uint64_t pressure_dropped_backup_keys{0};
    std::uint64_t overflow_dropped_backup_keys{0};
    std::uint64_t unavailable_dropped_backup_keys{0};
    std::uint64_t disabled_dropped_backup_keys{0};
    std::uint64_t cleanup_dropped_backup_keys{0};
};

namespace cache {
// The runtime agrees on this clock and execution permission before each plan.
struct StorageTick {
    std::int64_t now_us;
    bool allow_forward;
    bool allow_storage;
    StoragePressure pressure;
};

struct StorageDetach {
    std::string request_id;
};

struct StorageDone {
    std::uint64_t op_id;
    std::vector<std::uint8_t> success;
    // A false value quarantines every source/destination: failure does not
    // prove a native transfer has stopped accessing its registered memory.
    bool retired;
    // Confirmed missing/invalid shards (OR across owners). A skipped call or
    // transport failure carries no object evidence and must not poison keys.
    std::vector<std::uint8_t> object_failures;
};
}  // namespace cache
}  // namespace tokenspeed
