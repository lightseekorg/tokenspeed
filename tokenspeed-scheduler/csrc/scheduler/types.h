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
#include <optional>

#include "cache/tier/storage.h"
#include <vector>

#include "cache/core/cache_config.h"
#include "utils.h"

namespace tokenspeed {

struct SchedulerConfig {
    std::int32_t prefix_granularity{};
    struct AllocatorConfig {
        // Page 0 is the null placeholder, so usable = total - 1.
        std::int32_t total_pages{};
        std::int32_t NumUsableBlocks() const { return total_pages - 1; }
    };
    AllocatorConfig host_allocator;
    AllocatorConfig device_allocator;
    // The request-private pool a retracted request's image tail lives in
    // (the unaligned tail pages, groups that never publish, and -- with no
    // Host cache -- the whole image). Never prefix-indexed, never evicted.
    // Required: total_pages == 1 (the null page alone) means the engine never
    // retracts; anything above it enables retraction on the fused and D roles.
    AllocatorConfig snapshot_allocator;
    // Slot-state blob slots in the runtime's arena, hence the most requests
    // that can be retracted at once. Required with a snapshot pool above the
    // null page, and 0 without one.
    std::int32_t max_retracted_requests{};
    // Test knob: every |N| plans the fused/D grammar retracts the oldest
    // quiescent Decoding request (N > 0) or the one Prefilling request
    // between its chunks (N < 0), bypassing ReserveCoversGeneration but not
    // the image-fit refusal. 0 is off; non-zero requires a snapshot pool.
    // Lives here so every mirrored rank decides identically.
    std::int32_t debug_force_retraction_interval{0};

    std::vector<CacheGroupConfig> cache_groups{};
    // Virtual children per compact Host parent, shared by L2 and snapshots.
    // Required when either Host pool exists; no capacity is reserved by group.
    std::vector<std::int32_t> host_group_packing{};

    bool HasHostCache() const { return !disable_l2_cache && host_allocator.total_pages > 1; }
    bool HasSnapshotPool() const { return snapshot_allocator.total_pages > 1; }

    // What feeds Host L2 from ordinary Device prefix entries; required with a
    // Host tier (Validate says where kWriteBack may run).
    //   kWriteThrough  published entries stream to Host at chunk ends, the
    //                  prompt end and finish.
    //   kWriteBack     nothing streams at publication: what an admission
    //                  or a restore evicts is written back to Host in the
    //                  same plan, by whole segments.
    enum class HostWritePolicy : std::uint8_t { kUnspecified, kWriteThrough, kWriteBack };
    HostWritePolicy host_write_policy{HostWritePolicy::kUnspecified};

    // The D role streams no ordinary Device cache entry to Host; its Host
    // cache holds only the published pages of retracted requests (the image's
    // L2 leg) until they are restored.
    bool StreamsDeviceCacheToHost() const {
        return HasHostCache() && role != Role::kD && host_write_policy == HostWritePolicy::kWriteThrough;
    }

    std::int32_t max_scheduled_tokens{};
    // Sum of ceil((cached prefix + this chunk) / prefix_granularity) across
    // all local prefill requests in one batch. Zero leaves this unbounded.
    std::int32_t max_kv_page_num{0};
    std::int32_t max_batch_size{};
    std::int32_t decode_input_tokens{1};
    // Number of scheduler iterations that may be dispatched before the
    // accepted decode length is committed. The current event loop supports
    // only the non-overlapped (0) and one-step-overlapped (1) contracts.
    std::int32_t overlap_schedule_depth{0};
    bool disable_l2_cache{false};
    bool enable_l3_storage{false};
    std::optional<StorageConfig> storage_config;
    bool enable_kv_cache_events{false};
    bool enable_mixed_prefill_decode{false};

    // The P and D roles ARE the cache-transfer PD protocol: there is no
    // disaggregated deployment without it, so everything PD-specific
    // (transfer pins, destination layouts, transfer_policy validation) keys
    // on the role directly.
    Role role{Role::kFused};

    bool disable_prefix_cache{false};
    // Minimum prompt tail that must be recomputed after a prefix-cache hit.
    // Zero preserves the default logits contract, which already recomputes at
    // least the final prompt token. The effective hit is page-aligned down.
    std::int32_t prefix_replay_tokens{0};
    // Write a reusable State checkpoint inside the prefill forward that spans
    // the page boundary before a long prompt's replay tail
    // (PrefillStateCheckpointTokens), without cutting a chunk there: that
    // forward holds one more block per State group, and the request's next
    // publication point publishes it under the boundary's page hash. The
    // runtime writes the state at ForwardBatch::extend_state_checkpoint_lens
    // into the third entry of every State group's row. Requires prefix
    // caching, except on the D role, which never computes a prompt.
    bool state_checkpoint_in_forward{false};

    // The single validation entry point for a scheduler configuration: every
    // scheduler scalar, every group's own invariants, and the cross-checks
    // between them. Throws std::invalid_argument on the first violation.
    //
    // The Scheduler runs this before constructing any member, because the pools
    // and the cache coordinator assert on the same fields and their assertions
    // would otherwise preempt these diagnostics.
    void Validate() const;
    // The subset of Validate() the CapacityModel needs: every field it reads
    // to size a pool, none of the page counts that describe a sized one.
    void ValidateCapacityInputs() const;
};

}  // namespace tokenspeed
