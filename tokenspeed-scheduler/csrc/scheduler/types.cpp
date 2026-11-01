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

#include "scheduler/types.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace tokenspeed {

namespace {

void validateGroupCapacityInputs(const SchedulerConfig& config, const CacheGroupConfig& group) {
    group.ValidateCapacityInputs();
    const std::string where = "Cache group '" + group.group_id + "': ";
    if (group.attn_kind != AttnKind::kState && config.prefix_granularity % group.block_granularity != 0) {
        throw std::invalid_argument(where + "block_granularity must divide the scheduler prefix_granularity");
    }
    if (group.checkpoint_granularity > 1 && (group.checkpoint_granularity % config.prefix_granularity != 0 ||
                                             config.max_scheduled_tokens < group.checkpoint_granularity)) {
        throw std::invalid_argument(where +
                                    "checkpoint_granularity must be a multiple of prefix_granularity "
                                    "and fit max_scheduled_tokens");
    }
    if (config.role == Role::kFused) {
        return;
    }
    // A group's transfer policy is dictated by the destination layout the
    // scheduler builds for it, so it cannot be chosen independently.
    const CacheTransferPolicy expected =
        group.attn_kind == AttnKind::kState ? CacheTransferPolicy::LatestSnapshot : CacheTransferPolicy::FullSuffix;
    if (group.transfer_policy == CacheTransferPolicy::Unspecified) {
        throw std::invalid_argument(where + "PD cache requires an explicit transfer_policy");
    }
    if (group.transfer_policy != expected) {
        throw std::invalid_argument(where + "transfer_policy does not match its scheduler destination layout");
    }
}

}  // namespace

void SchedulerConfig::Validate() const {
    ValidateCapacityInputs();
    if (max_kv_page_num < 0) {
        throw std::invalid_argument("Scheduler: max_kv_page_num must be >= 0");
    }
    if (device_allocator.total_pages <= 1) {
        throw std::invalid_argument("Scheduler: device cache must contain a null page and usable capacity");
    }
    for (const CacheGroupConfig& group : cache_groups) {
        group.Validate();
        // A grain only shapes where the checkpoint lands; the capability to write it stays a hard requirement, so
        // an aligned group never schedules a checkpoint its consumer cannot fill.
        if (role != Role::kD && group.checkpoint_granularity > 1 && !state_checkpoint_in_forward) {
            throw std::invalid_argument(
                "Scheduler: a State group with checkpoint_granularity > 1 requires state_checkpoint_in_forward");
        }
    }
    if (prefix_replay_tokens < 0) {
        throw std::invalid_argument("Scheduler: prefix_replay_tokens must be >= 0");
    }
    if (state_checkpoint_in_forward && disable_prefix_cache && role != Role::kD) {
        throw std::invalid_argument("Scheduler: state_checkpoint_in_forward requires prefix caching");
    }
    if (!enable_l3_storage && storage_config) {
        throw std::invalid_argument("StorageConfig requires L3 storage to be enabled");
    }
    if (enable_l3_storage) {
        if (!storage_config) throw std::invalid_argument("L3 requires an explicit StorageConfig");
        storage_config->Validate();
        if (storage_config->max_host_blocks > host_allocator.NumUsableBlocks()) {
            throw std::invalid_argument("L3 Host credits must not exceed usable Host parents");
        }
    }
    if (enable_l3_storage && !HasHostCache()) {
        throw std::invalid_argument("Scheduler: L3 storage requires Host L2 cache");
    }
    if (HasHostCache() && host_write_policy == HostWritePolicy::kUnspecified) {
        throw std::invalid_argument("Scheduler: a Host L2 cache requires an explicit host_write_policy");
    }
    // Write-back needs no KV-parallel check: each Host block takes its Device
    // block's bucket (docs/design/scheduler.md §5).
    //
    // Device copies read their sources before any copy in the plan overwrites
    // those bytes, including when another group reuses the same LCM block.
    if (host_write_policy == HostWritePolicy::kWriteBack) {
        if (!HasHostCache()) {
            throw std::invalid_argument("Scheduler: host_write_policy WriteBack requires a Host L2 cache");
        }
        if (role == Role::kD) {
            throw std::invalid_argument(
                "Scheduler: host_write_policy WriteBack is not available on the D role: its Host holds retraction "
                "images only and remote admission probes the Device alone");
        }
        if (enable_l3_storage) {
            throw std::invalid_argument("Scheduler: host_write_policy WriteBack excludes L3 storage");
        }
        // BoundaryTree::Node::written_back_group_bits keeps one bit per group.
        if (cache_groups.size() > 64) {
            throw std::invalid_argument("Scheduler: host_write_policy WriteBack supports at most 64 cache groups");
        }
    }
    // The snapshot pool is stated explicitly on every role: the null page
    // alone (and no slots) says "never retract", anything more needs slots.
    // The messages name the binding field and the server arg that sets it,
    // because the runtime surfaces them verbatim at startup.
    if (snapshot_allocator.total_pages < 1) {
        throw std::invalid_argument(
            "Scheduler: snapshot_allocator.total_pages (num_snapshot_pages, from --retraction-snapshot-host-gb) "
            "must include the null page: 1 means the engine never retracts");
    }
    if (max_retracted_requests < 0) {
        throw std::invalid_argument(
            "Scheduler: max_retracted_requests (--retraction-snapshot-max-requests) must be >= 0");
    }
    if (HasSnapshotPool() != (max_retracted_requests > 0)) {
        throw std::invalid_argument(
            "Scheduler: a snapshot pool above the null page (num_snapshot_pages > 1, from "
            "--retraction-snapshot-host-gb) requires max_retracted_requests > 0 "
            "(--retraction-snapshot-max-requests), and max_retracted_requests > 0 requires a snapshot pool");
    }
    if (role == Role::kP && HasSnapshotPool()) {
        throw std::invalid_argument(
            "Scheduler: the P role never retracts and takes no snapshot pool (--retraction-snapshot-host-gb "
            "must be 0 on the prefill role)");
    }
    // The knob retracts through the same path capacity pressure uses, and
    // that path needs somewhere to put the image.
    if (debug_force_retraction_interval != 0 && !HasSnapshotPool()) {
        throw std::invalid_argument(
            "Scheduler: debug_force_retraction_interval (--debug-force-retraction-interval) requires a snapshot "
            "pool (--retraction-snapshot-host-gb > 0): without one nothing can be retracted");
    }
    if (HasHostCache() || HasSnapshotPool()) {
        if (host_group_packing.size() != cache_groups.size()) {
            throw std::invalid_argument("Scheduler: Host pools require explicit packing for every cache group");
        }
        for (std::size_t i = 0; i < cache_groups.size(); ++i) {
            const std::int32_t packing = host_group_packing[i];
            if (packing <= 0 || packing % cache_groups[i].shard_count != 0) {
                throw std::invalid_argument("Scheduler: Host packing must be positive and divisible by shard_count");
            }
            const std::int64_t parents =
                std::max(host_allocator.NumUsableBlocks(), snapshot_allocator.NumUsableBlocks());
            if (parents * packing > std::numeric_limits<std::int32_t>::max()) {
                throw std::invalid_argument("Scheduler: Host cache block ids exceed int32 range");
            }
        }
    } else if (!host_group_packing.empty()) {
        throw std::invalid_argument("Scheduler: Host packing requires an L2 or snapshot pool");
    }
}

void SchedulerConfig::ValidateCapacityInputs() const {
    if (prefix_granularity <= 0) {
        throw std::invalid_argument("Scheduler: prefix_granularity must be > 0");
    }
    if (cache_groups.empty()) {
        throw std::invalid_argument("Scheduler: at least one cache group is required");
    }
    if (max_batch_size < 0) {
        throw std::invalid_argument("Scheduler: max_batch_size must be >= 0");
    }
    if (decode_input_tokens < 0) {
        throw std::invalid_argument("Scheduler: decode_input_tokens must be >= 0");
    }
    if (max_scheduled_tokens <= 0) {
        throw std::invalid_argument("Scheduler: max_scheduled_tokens must be > 0");
    }
    if (overlap_schedule_depth < 0 || overlap_schedule_depth > 1) {
        throw std::invalid_argument("Scheduler: overlap_schedule_depth must be 0 or 1");
    }
    if (overlap_schedule_depth > 0 && decode_input_tokens == 0) {
        throw std::invalid_argument("Scheduler: overlapped decode requires decode_input_tokens > 0");
    }
    std::int32_t replay_window_tokens = 0;
    std::int32_t checkpoint_granularity = 1;
    for (const CacheGroupConfig& group : cache_groups) {
        validateGroupCapacityInputs(*this, group);
        if (group.checkpoint_granularity > 1) {
            if (checkpoint_granularity != 1 && checkpoint_granularity != group.checkpoint_granularity) {
                throw std::invalid_argument("restricted State groups must use the same checkpoint_granularity");
            }
            checkpoint_granularity = group.checkpoint_granularity;
        }
        if (group.replayable) {
            replay_window_tokens = std::max(replay_window_tokens, *group.sliding_window_tokens);
        }
    }
    if (replay_window_tokens > 0 && checkpoint_granularity > 1) {
        throw std::invalid_argument("aligned State checkpoints do not support bounded replay groups");
    }
    if (replay_window_tokens > 0) {
        // A prefix hit re-feeds up to one replay window and must still carry
        // the prompt's final window whole, or at least a prefix page of new
        // tokens when the window is smaller than a page.
        if (max_scheduled_tokens < replay_window_tokens + std::max(replay_window_tokens, prefix_granularity)) {
            throw std::invalid_argument(
                "Scheduler: max_scheduled_tokens must cover the replayable groups' largest sliding_window_tokens "
                "plus max(that window, prefix_granularity)");
        }
    }
}

}  // namespace tokenspeed
