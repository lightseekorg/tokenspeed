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

// Scheduler-to-runtime wire types of the Disk tier (docs/rfcs/0006-disk-tier.md):
// the limits a DiskTransferManager runs under, the operations a plan starts on
// every cache owner's disk executor, and the completion the owners agree on.

#include <cstdint>
#include <vector>

namespace tokenspeed {

// All limits are explicit at the runtime/scheduler boundary.
struct DiskConfig {
    // Disk operations in flight at once, reads and writes together.
    std::int32_t max_operations;
    // Blocks one operation moves.
    std::int32_t max_blocks_per_operation;
    // Host blocks Disk operations hold at once: a prefetch's destinations, a
    // cleaning write's sources. Each can keep one Host parent from recycling.
    std::int32_t max_host_blocks;
    // Of those, at most this many for cleaning writes, so a slow drive leaves
    // prefetches room.
    std::int32_t max_write_host_blocks;
    // Host entries waiting to be cleaned. The queue holds keys, not blocks;
    // past the bound its oldest key is dropped.
    std::int32_t max_queued_cleanings;
    // A prefetch is worth starting only when Disk extends the local hit by
    // at least this many tokens.
    std::int32_t min_prefetch_tokens;
    // How long a request waits for its prefetch before admitting from its
    // local hit, and how long the prefetched Host entries are held for it.
    std::int64_t wait_timeout_us;
    std::int64_t ready_timeout_us;
    void Validate() const;
};

enum class DiskOpKind : std::uint8_t {
    // Disk to Host: a prefetch's reads.
    kRead,
    // Host to Disk: a cleaning write.
    kWrite,
};

// One Disk operation, run on every cache owner. Row i moves group_ids[i]'s
// block between Host block host_blocks[i] and Disk block disk_blocks[i]; the
// scheduler holds both until the owners agree it completed.
struct DiskOperation {
    DiskOpKind kind{DiskOpKind::kRead};
    std::uint64_t op_id{0};
    std::vector<std::uint32_t> group_ids;
    std::vector<std::int32_t> host_blocks;
    std::vector<std::int32_t> disk_blocks;

    bool operator==(const DiskOperation&) const = default;
};

// Scheduler-owned observations of the Disk tier.
struct DiskStats {
    std::uint64_t operations{0};
    std::uint64_t held_host_blocks{0};
    std::uint64_t queued_cleanings{0};
    std::uint64_t waiting_requests{0};
    std::uint64_t ready_requests{0};
    std::uint64_t disk_blocks{0};
    std::uint64_t free_disk_parents{0};
    // Cumulative.
    std::uint64_t written_blocks{0};
    std::uint64_t read_blocks{0};
    std::uint64_t failed_reads{0};
    std::uint64_t failed_writes{0};
    std::uint64_t dropped_cleanings{0};
    std::uint64_t wait_timeouts{0};
    std::uint64_t ready_expirations{0};
};

namespace cache {
// Every cache owner finished the operation; `success` only when all of them
// moved every row. A finished operation retired: nothing still touches its
// blocks.
struct DiskDone {
    std::uint64_t op_id;
    bool success;
};
}  // namespace cache

}  // namespace tokenspeed
