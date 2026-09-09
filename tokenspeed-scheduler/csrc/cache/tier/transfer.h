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
            std::vector<std::uint32_t> operation_groups;
            std::vector<std::int32_t> operation_sources;
            std::vector<std::int32_t> operation_destinations;
            std::vector<std::string> operation_hashes;
            std::vector<std::int32_t> operation_offsets;
            for (const auto& transfer : op.transfers) {
                if (seen.insert(transfer).second) {
                    operation_groups.push_back(transfer.group_id);
                    operation_sources.push_back(transfer.source_page);
                    operation_destinations.push_back(transfer.destination_page);
                    operation_hashes.push_back(transfer.content_hash);
                    operation_offsets.push_back(transfer.page_offset);
                }
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
            std::vector<std::uint32_t> operation_groups;
            std::vector<std::int32_t> operation_sources;
            std::vector<std::int32_t> operation_destinations;
            std::vector<std::string> operation_hashes;
            std::vector<std::int32_t> operation_offsets;
            std::vector<std::uint8_t> operation_prefetch;
            for (const auto& transfer : op.transfers) {
                if (seen.insert(transfer).second) {
                    operation_groups.push_back(transfer.group_id);
                    operation_sources.push_back(transfer.source_page);
                    operation_destinations.push_back(transfer.destination_page);
                    operation_hashes.push_back(transfer.content_hash);
                    operation_offsets.push_back(transfer.page_offset);
                    operation_prefetch.push_back(transfer.prefetch_from_storage ? std::uint8_t{1} : std::uint8_t{0});
                }
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

using CacheOperation = std::variant<LoadBackBatch, WriteBackBatch>;

}  // namespace tokenspeed
