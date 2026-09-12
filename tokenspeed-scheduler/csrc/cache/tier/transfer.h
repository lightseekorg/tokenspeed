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
            for (const auto& transfer : op.transfers) {
                _assert(seen.insert(transfer).second, "duplicate write-back transfer within one plan");
                operation_groups.push_back(transfer.group_id);
                operation_sources.push_back(transfer.source_page);
                operation_destinations.push_back(transfer.destination_page);
            }

            op_ids.push_back(op.op_id);
            group_ids.push_back(std::move(operation_groups));
            src_pages.push_back(std::move(operation_sources));
            dst_pages.push_back(std::move(operation_destinations));
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

    explicit LoadBackBatch(const std::vector<LoadBackOperation>& ops) {
        std::unordered_set<CacheTransfer, CacheTransferHash> seen;
        for (const auto& op : ops) {
            _assert(!op.transfers.empty(), "load-back op carries no transfers");
            std::vector<std::uint32_t> operation_groups;
            std::vector<std::int32_t> operation_sources;
            std::vector<std::int32_t> operation_destinations;
            for (const auto& transfer : op.transfers) {
                _assert(seen.insert(transfer).second, "duplicate load-back transfer within one plan");
                operation_groups.push_back(transfer.group_id);
                operation_sources.push_back(transfer.source_page);
                operation_destinations.push_back(transfer.destination_page);
            }

            op_ids.push_back(op.op_id);
            group_ids.push_back(std::move(operation_groups));
            src_pages.push_back(std::move(operation_sources));
            dst_pages.push_back(std::move(operation_destinations));
        }
    }
};

using CacheOperation = std::variant<LoadBackBatch, WriteBackBatch>;

}  // namespace tokenspeed
