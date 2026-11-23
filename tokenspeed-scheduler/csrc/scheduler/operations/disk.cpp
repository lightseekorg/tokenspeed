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

// The scheduler's side of the Disk tier (docs/rfcs/0006-disk-tier.md): a
// waiting prompt's prefetch, and the completions of Disk operations.

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "cache/prefix/prefix_hasher.h"
#include "fsm/storage_events.h"
#include "scheduler/scheduler.h"

namespace tokenspeed {

DiskStats Scheduler::GetDiskStats() const {
    DiskStats stats;
    if (!disk_transfers_) return stats;
    stats.operations = disk_transfers_->NumOperations();
    stats.held_host_blocks = disk_transfers_->NumHeldHostBlocks();
    stats.queued_cleanings = disk_transfers_->NumQueuedCleanings();
    stats.disk_blocks = static_cast<std::uint64_t>(coordinator_.Disk().NumDiskCachedBlocks());
    stats.free_disk_parents = static_cast<std::uint64_t>(coordinator_.Disk().NumFreeLcmBlocks());
    stats.written_blocks = disk_transfers_->WrittenBlocks();
    stats.read_blocks = disk_transfers_->ReadBlocks();
    stats.failed_reads = disk_transfers_->FailedReads();
    stats.failed_writes = disk_transfers_->FailedWrites();
    stats.dropped_cleanings = disk_transfers_->DroppedCleanings();
    stats.wait_timeouts = storage_wait_timeouts_;
    stats.ready_expirations = storage_ready_expirations_;
    for (const auto& request : requests_) {
        const auto* prefetch = request->GetIf<fsm::Prefetching>();
        stats.waiting_requests += prefetch != nullptr && prefetch->lease.IssuedBy(*disk_transfers_);
        const auto* admission = request->AdmissionResourcesPtr();
        stats.ready_requests += admission != nullptr && admission->storage_hold &&
                                admission->storage_hold->lease.IssuedBy(*disk_transfers_);
    }
    return stats;
}

void Scheduler::handleEvent(const cache::DiskDone& event) {
    if (!disk_transfers_) throw std::invalid_argument("a Disk completion without a configured Disk tier");
    disk_transfers_->Complete(event);
}

void Scheduler::prepareDisk(Request& request) {
    const DiskConfig& limits = disk_transfers_->Config();
    if (auto* state = request.GetIf<fsm::Prefetching>()) {
        if (!state->lease.IssuedBy(*disk_transfers_)) return;
        const std::uint64_t operation = state->lease.OperationId();
        if (disk_transfers_->Completed(operation)) {
            const auto deadline =
                storage_now_us_ +
                std::min(limits.ready_timeout_us, std::numeric_limits<std::int64_t>::max() - storage_now_us_);
            spdlog::debug("[Disk] rid={} op_id={} prefetch read completed", request.Id(), operation);
            request.Apply(fsm::FinishPrefetchEvent{deadline});
            return;
        }
        if (storage_now_us_ >= state->deadline_us) {
            expireStorage(request);
        }
        return;
    }

    auto* admission = request.AdmissionResourcesPtr();
    if (admission == nullptr || admission->storage_prepared || !storage_allowed_) return;
    admission->storage_prepared = true;
    // Decode receives its prompt from its PD peer.
    if (config_.role == Role::kD) return;
    const auto grain = config_.prefix_granularity;
    const auto max_tokens = cacheablePrefixTokens(request);
    const auto cached_hashes = request.PromptPrefixHashes(max_tokens / grain);
    std::vector<std::string> hashes(cached_hashes.begin(), cached_hashes.end());
    std::optional<TailPublication> tail;
    if (const auto tail_tokens = max_tokens % grain; tail_tokens > 0) {
        const ContentHash prior = hashes.empty() ? ContentHash{} : ContentHash{hashes.back()};
        tail =
            TailPublication{.tail_tokens = tail_tokens,
                            .hash = HashPrefixPage(request.TokenSlice(max_tokens - tail_tokens, tail_tokens), prior)};
    }
    PromptPrefix prompt{.prefix_hashes = hashes,
                        .tokens = request.TokenSlice(0, max_tokens),
                        .max_tokens = max_tokens,
                        .tail = tail,
                        .shared_prefix_tokens = request.SharedPrefixTokens()};
    const PrefixProbe probe = coordinator_.Prober().ProbePromptPrefix(prompt);
    const auto local = std::max(probe.device.num_common_tokens, probe.host.num_common_tokens);
    if (max_tokens - local < limits.min_prefetch_tokens) return;
    prompt.prepared_keys = probe.group_keys;
    StorageLease lease = disk_transfers_->StartRead(prompt);
    if (!lease) return;
    spdlog::debug("[Disk] rid={} op_id={} local_tokens={} bound_tokens={} prefetch read enqueued", request.Id(),
                  lease.OperationId(), local, max_tokens);
    const auto deadline =
        storage_now_us_ + std::min(limits.wait_timeout_us, std::numeric_limits<std::int64_t>::max() - storage_now_us_);
    request.Apply(fsm::StartPrefetchEvent{std::move(lease), deadline, std::move(hashes), max_tokens, std::move(tail)});
}

}  // namespace tokenspeed
