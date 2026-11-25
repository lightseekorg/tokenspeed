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

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/tier/disk.h"
#include "cache/tier/storage_lease.h"

namespace tokenspeed {

// The Disk tier's transfers (docs/rfcs/0006-disk-tier.md): a request's
// prefetch, which reads the Disk part of its prefix into Host before its
// admission, and cleaning, which writes dirty Host entries to Disk so that a
// Host placement can drop them without loss. It owns the blocks every
// operation holds, never a request's scheduling or an I/O call: the runtime
// receives each operation's block ids and reports when every cache owner
// finished it. All methods run on the scheduler thread.
//
// A prefetch is started for a request and returned as its StorageLease: the
// request waits in Prefetching, then holds the published Host entries as its
// Ready hold until its admission acquires them. A cleaning write belongs to
// no request and retires on its own.
class DiskTransferManager : public StorageLeaseIssuer {
public:
    DiskTransferManager(CacheCoordinator& coordinator, DiskConfig config);
    DiskTransferManager(const DiskTransferManager&) = delete;
    DiskTransferManager& operator=(const DiskTransferManager&) = delete;

    const DiskConfig& Config() const { return config_; }

    struct ReadStartResult {
        StorageLease lease;
        // True when Disk had hits (or credits were exhausted before probing)
        // and the read could not start only because operation or Host-block
        // credits were temporarily insufficient.
        bool retryable{false};
    };

    // Whether a read can start now within the operation and Host-block bounds.
    bool HasReadCredits() const;
    // Reads the keys of the prompt's prefix that only the Disk holds -- for
    // the longest boundary Disk extends the local hit to under the
    // Host-suffix rule -- into Host blocks it allocates, holding the Disk
    // sources. Distinguishes a permanent miss from transient credit exhaustion.
    ReadStartResult TryStartRead(const PromptPrefix& prompt, std::int32_t local_tokens);
    // Convenience wrapper returning only the lease.
    StorageLease StartRead(const PromptPrefix& prompt);
    // Whether the operation finished; an unknown operation counts as finished.
    bool Completed(std::uint64_t id) const;

    // Queues Host entries to be cleaned, oldest first. They are checked
    // again when a cleaning write takes them.
    void QueueCleaning(std::span<const CacheKey> keys);
    // Writes the oldest queued keys that are still dirty Host entries, each
    // with the dirty Host entries of the pages above it, into Disk blocks it
    // allocates, within the write credits. At most one write per plan.
    void StartCleaning();
    // Forgets the queued keys (a flush, a frozen round).
    void DropCleaning();

    // Publishes what a finished operation moved: a read's Host entries, a
    // write's Disk entries. A failed read drops the Disk entries it read.
    void Complete(const cache::DiskDone& event);
    std::vector<DiskOperation> TakeCommands();

    bool HasPending() const { return !operations_.empty(); }
    // Pending work a waiting request can still see finish.
    bool HasProgressPending() const;
    std::size_t NumOperations() const { return operations_.size(); }
    std::size_t NumHeldHostBlocks() const;
    // The Host blocks the operations hold: a write's indexed sources, a
    // read's unpublished targets. A cache clear drains them
    // (CacheCoordinator::CacheIsClearableAfterStorageDrain).
    std::vector<const CacheBlockRef*> HeldHostBlockRefs() const;
    std::size_t NumQueuedCleanings() const { return cleaning_.size(); }
    std::uint64_t WrittenBlocks() const { return written_blocks_; }
    std::uint64_t ReadBlocks() const { return read_blocks_; }
    std::uint64_t FailedReads() const { return failed_reads_; }
    std::uint64_t FailedWrites() const { return failed_writes_; }
    std::uint64_t DroppedCleanings() const { return dropped_cleanings_; }

private:
    struct Ticket {
        DiskOpKind kind{DiskOpKind::kRead};
        std::vector<CacheKey> keys;
        std::vector<CacheBlockRef> host_blocks;
        std::vector<CacheBlockRef> disk_blocks;
        // A cleaning write: each key's Host entry metadata, which its Disk
        // entry is published with.
        std::vector<PrefixCacheIndex::CachedBlockMetadata> metadata;
        // A read: the tail it brings to Host, whose hint the Host keeps.
        ContentHash tail_prior;
        std::optional<TailPublication> tail;
        bool detached{false};
        bool completed{false};
    };

    std::uint64_t nextId();
    void detach(std::uint64_t id) noexcept override;
    std::size_t numOperations(DiskOpKind kind) const;
    std::size_t heldHostBlocks(DiskOpKind kind) const;
    DiskOperation command(std::uint64_t id, const Ticket& ticket) const;

    CacheCoordinator& coordinator_;
    const DiskConfig config_;
    std::map<std::uint64_t, Ticket> operations_;
    std::deque<CacheKey> cleaning_;
    std::vector<DiskOperation> commands_;
    std::uint64_t next_id_{1};
    std::uint64_t written_blocks_{0};
    std::uint64_t read_blocks_{0};
    std::uint64_t failed_reads_{0};
    std::uint64_t failed_writes_{0};
    std::uint64_t dropped_cleanings_{0};
};

}  // namespace tokenspeed
