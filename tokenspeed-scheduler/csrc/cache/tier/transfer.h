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

// Scheduler-to-runtime wire types for transfers between cache tiers.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "utils.h"

namespace tokenspeed {

struct CacheTransfer {
    std::uint32_t group_id{0};
    std::int32_t source_page{-1};
    std::int32_t destination_page{-1};
    std::string content_hash{};
    std::int32_t page_offset{0};
    bool prefetch_from_storage{false};

    bool operator==(const CacheTransfer&) const = default;
};

struct CacheTransferHash {
    std::size_t operator()(const CacheTransfer& transfer) const {
        std::size_t seed = std::hash<std::uint32_t>{}(transfer.group_id);
        const auto combine = [&seed](std::int32_t value) {
            const std::size_t hash = std::hash<std::int32_t>{}(value);
            seed ^= hash + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        };
        combine(transfer.source_page);
        combine(transfer.destination_page);
        return seed;
    }
};

// How a store's Device source is protected while its D2H copy is in flight.
enum class StoreSourceGuard : std::uint8_t {
    // The scheduler pins the Device block until the runtime acknowledges the
    // copy: it stays cached and unevictable, so the runtime may copy it on
    // any stream, off the forward's critical path. Ordinary publication.
    kPinnedUntilAck,
    // The Device block is released -- and may be re-granted -- the moment the
    // store issues; the runtime must order the copy on the forward thread's
    // stream ahead of the plan's page reuse. A retraction's snapshot: the
    // victim's pages are granted away in the same round.
    kStreamOrdered,
};

struct WriteBackOperation {
    std::uint32_t op_id{0};
    std::vector<CacheTransfer> transfers;  // DEVICE→HOST.
    // False is the safe reading: the runtime orders the copy ahead of reuse.
    bool source_pinned{false};
};

// Every op on the wire carries at least one transfer, and no (group, source,
// destination) repeats within one plan: a store skips keys already in flight
// and a load targets freshly acquired pages. The runtime relies on both -- an
// op is acknowledged by its copy's completion event, so an empty op would
// never be acknowledged and its tickets would leak. A violation is a
// scheduler bug and fails here rather than being papered over.
struct WriteBackBatch {
    std::vector<std::uint32_t> op_ids;
    std::vector<std::vector<std::uint32_t>> group_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;
    std::vector<std::vector<std::string>> content_hashes;
    std::vector<std::vector<std::int32_t>> page_offsets;
    // Per op: whether the scheduler holds the Device sources until the ACK
    // (see StoreSourceGuard). The runtime must order an unpinned op's copy
    // ahead of the plan's page zeroing; a pinned op may ride any stream.
    std::vector<bool> source_pinned;

    explicit WriteBackBatch(const std::vector<WriteBackOperation>& ops) {
        std::unordered_set<CacheTransfer, CacheTransferHash> seen;
        for (const auto& op : ops) {
            _assert(!op.transfers.empty(), "write-back op carries no transfers");
            std::vector<std::uint32_t> operation_groups;
            std::vector<std::int32_t> operation_sources;
            std::vector<std::int32_t> operation_destinations;
            std::vector<std::string> operation_hashes;
            std::vector<std::int32_t> operation_offsets;
            for (const auto& transfer : op.transfers) {
                _assert(seen.insert(transfer).second, "duplicate write-back transfer within one plan");
                operation_groups.push_back(transfer.group_id);
                operation_sources.push_back(transfer.source_page);
                operation_destinations.push_back(transfer.destination_page);
                operation_hashes.push_back(transfer.content_hash);
                operation_offsets.push_back(transfer.page_offset);
            }

            op_ids.push_back(op.op_id);
            group_ids.push_back(std::move(operation_groups));
            src_pages.push_back(std::move(operation_sources));
            dst_pages.push_back(std::move(operation_destinations));
            content_hashes.push_back(std::move(operation_hashes));
            page_offsets.push_back(std::move(operation_offsets));
            source_pinned.push_back(op.source_pinned);
        }
    }
};

struct LoadBackOperation {
    std::uint32_t op_id{0};
    std::vector<CacheTransfer> transfers;  // HOST→DEVICE.
};

struct LoadBackBatch {
    std::vector<std::uint32_t> op_ids;
    std::vector<std::vector<std::uint32_t>> group_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;
    std::vector<std::vector<std::string>> content_hashes;
    std::vector<std::vector<std::int32_t>> page_offsets;
    std::vector<std::vector<std::uint8_t>> prefetch_from_storage;

    explicit LoadBackBatch(const std::vector<LoadBackOperation>& ops) {
        std::unordered_set<CacheTransfer, CacheTransferHash> seen;
        for (const auto& op : ops) {
            _assert(!op.transfers.empty(), "load-back op carries no transfers");
            std::vector<std::uint32_t> operation_groups;
            std::vector<std::int32_t> operation_sources;
            std::vector<std::int32_t> operation_destinations;
            std::vector<std::string> operation_hashes;
            std::vector<std::int32_t> operation_offsets;
            std::vector<std::uint8_t> operation_prefetch;
            for (const auto& transfer : op.transfers) {
                _assert(seen.insert(transfer).second, "duplicate load-back transfer within one plan");
                operation_groups.push_back(transfer.group_id);
                operation_sources.push_back(transfer.source_page);
                operation_destinations.push_back(transfer.destination_page);
                operation_hashes.push_back(transfer.content_hash);
                operation_offsets.push_back(transfer.page_offset);
                operation_prefetch.push_back(transfer.prefetch_from_storage ? std::uint8_t{1} : std::uint8_t{0});
            }

            op_ids.push_back(op.op_id);
            group_ids.push_back(std::move(operation_groups));
            src_pages.push_back(std::move(operation_sources));
            dst_pages.push_back(std::move(operation_destinations));
            content_hashes.push_back(std::move(operation_hashes));
            page_offsets.push_back(std::move(operation_offsets));
            prefetch_from_storage.push_back(std::move(operation_prefetch));
        }
    }
};

// Which Host buffer a restore row reads: the Host L2 prefix tier or the
// request-private snapshot pool.
enum class HostTier : std::uint8_t { kL2, kSnapshotPool };

// The tail leg of a retraction image, DEVICE -> snapshot pool: the data slots
// that no published prefix entry covers, plus the request's slot-state blob
// (exported from request_pool_index into snapshot_slot of the runtime's blob
// arena). The Device sources are re-granted in the same round, so the runtime
// orders this copy on the forward thread's stream ahead of the plan's page
// reuse, exactly like a stream-ordered write-back; the two ride one fence.
// `transfers` may be empty: the blob row is always there, so the op still
// completes a copy and is acknowledged.
struct SnapshotStoreOperation {
    std::uint32_t op_id{0};
    std::string request_id;
    std::int32_t request_pool_index{-1};
    std::int32_t snapshot_slot{-1};
    std::vector<CacheTransfer> transfers;
};

// The way back, Host -> DEVICE in one op: every imaged slot that was not
// claimed from the Device prefix index, each row naming its source tier, plus
// the slot-state blob imported into the new request_pool_index. L2 rows carry
// their key (content_hash / page_offset) so the ACK republishes the
// destination into the Device prefix index; pool rows carry none. Runs after
// the plan's zeroing; nothing in the round's batch reads the destinations,
// so no layerwise tracking is involved.
struct SnapshotRestoreOperation {
    std::uint32_t op_id{0};
    std::string request_id;
    std::int32_t request_pool_index{-1};
    std::int32_t snapshot_slot{-1};
    std::vector<CacheTransfer> transfers;
    std::vector<HostTier> source_tier;  // parallel to transfers
};

namespace internal_transfer {

// No (group, source, destination) repeats within one plan; an op's row set
// may be empty only when it carries another payload (the slot-state blob).
inline void AppendTransfers(std::unordered_set<CacheTransfer, CacheTransferHash>& seen,
                            const std::vector<CacheTransfer>& transfers, std::vector<std::uint32_t>& group_ids,
                            std::vector<std::int32_t>& src_pages, std::vector<std::int32_t>& dst_pages,
                            const char* what) {
    for (const CacheTransfer& transfer : transfers) {
        _assert(seen.insert(transfer).second, what);
        group_ids.push_back(transfer.group_id);
        src_pages.push_back(transfer.source_page);
        dst_pages.push_back(transfer.destination_page);
    }
}

}  // namespace internal_transfer

struct SnapshotStoreBatch {
    std::vector<std::uint32_t> op_ids;
    std::vector<std::string> request_ids;
    std::vector<std::int32_t> request_pool_indices;
    std::vector<std::int32_t> snapshot_slots;
    std::vector<std::vector<std::uint32_t>> group_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;

    explicit SnapshotStoreBatch(const std::vector<SnapshotStoreOperation>& ops) {
        std::unordered_set<CacheTransfer, CacheTransferHash> seen;
        for (const SnapshotStoreOperation& op : ops) {
            _assert(op.request_pool_index >= 0 && op.snapshot_slot >= 0,
                    "snapshot store op must name the victim's request slot and its blob slot");
            std::vector<std::uint32_t> operation_groups;
            std::vector<std::int32_t> operation_sources;
            std::vector<std::int32_t> operation_destinations;
            internal_transfer::AppendTransfers(seen, op.transfers, operation_groups, operation_sources,
                                               operation_destinations,
                                               "duplicate snapshot store transfer within one plan");
            op_ids.push_back(op.op_id);
            request_ids.push_back(op.request_id);
            request_pool_indices.push_back(op.request_pool_index);
            snapshot_slots.push_back(op.snapshot_slot);
            group_ids.push_back(std::move(operation_groups));
            src_pages.push_back(std::move(operation_sources));
            dst_pages.push_back(std::move(operation_destinations));
        }
    }
};

struct SnapshotRestoreBatch {
    std::vector<std::uint32_t> op_ids;
    std::vector<std::string> request_ids;
    std::vector<std::int32_t> request_pool_indices;
    std::vector<std::int32_t> snapshot_slots;
    std::vector<std::vector<std::uint32_t>> group_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;
    std::vector<std::vector<std::string>> content_hashes;
    std::vector<std::vector<std::int32_t>> page_offsets;
    // Per row: HostTier as its integer value (0 = L2, 1 = snapshot pool).
    std::vector<std::vector<std::uint8_t>> source_tiers;

    explicit SnapshotRestoreBatch(const std::vector<SnapshotRestoreOperation>& ops) {
        std::unordered_set<CacheTransfer, CacheTransferHash> seen;
        for (const SnapshotRestoreOperation& op : ops) {
            _assert(op.request_pool_index >= 0 && op.snapshot_slot >= 0,
                    "snapshot restore op must name the new request slot and its blob slot");
            _assert(op.source_tier.size() == op.transfers.size(), "every restore row names its source tier");
            std::vector<std::uint32_t> operation_groups;
            std::vector<std::int32_t> operation_sources;
            std::vector<std::int32_t> operation_destinations;
            std::vector<std::string> operation_hashes;
            std::vector<std::int32_t> operation_offsets;
            std::vector<std::uint8_t> operation_tiers;
            internal_transfer::AppendTransfers(seen, op.transfers, operation_groups, operation_sources,
                                               operation_destinations,
                                               "duplicate snapshot restore transfer within one plan");
            for (std::size_t i = 0; i < op.transfers.size(); ++i) {
                operation_hashes.push_back(op.transfers[i].content_hash);
                operation_offsets.push_back(op.transfers[i].page_offset);
                operation_tiers.push_back(static_cast<std::uint8_t>(op.source_tier[i]));
            }
            op_ids.push_back(op.op_id);
            request_ids.push_back(op.request_id);
            request_pool_indices.push_back(op.request_pool_index);
            snapshot_slots.push_back(op.snapshot_slot);
            group_ids.push_back(std::move(operation_groups));
            src_pages.push_back(std::move(operation_sources));
            dst_pages.push_back(std::move(operation_destinations));
            content_hashes.push_back(std::move(operation_hashes));
            page_offsets.push_back(std::move(operation_offsets));
            source_tiers.push_back(std::move(operation_tiers));
        }
    }
};

using CacheOperation = std::variant<LoadBackBatch, WriteBackBatch, SnapshotStoreBatch, SnapshotRestoreBatch>;

}  // namespace tokenspeed
