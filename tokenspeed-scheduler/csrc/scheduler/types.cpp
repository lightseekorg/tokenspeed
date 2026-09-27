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
#include <stdexcept>
#include <string>

namespace tokenspeed {

namespace {

void validateGroupCapacityInputs(const SchedulerConfig& config, const CacheGroupConfig& group) {
    group.ValidateCapacityInputs();
    const std::string where = "Cache group '" + group.group_id + "': ";
    if (config.prefix_granularity % group.block_granularity != 0) {
        throw std::invalid_argument(where + "block_granularity must divide the scheduler prefix_granularity");
    }
    if (config.role == Role::kFused) {
        return;
    }
    // A group's transfer policy is dictated by the destination layout the
    // scheduler builds for it, so it cannot be chosen independently.
    const CacheTransferPolicy expected =
        group.IsSnapshotStateGroup() ? CacheTransferPolicy::LatestSnapshot : CacheTransferPolicy::FullSuffix;
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
    if (device_allocator.total_pages <= 1) {
        throw std::invalid_argument("Scheduler: device cache must contain a null page and usable capacity");
    }
    for (const CacheGroupConfig& group : cache_groups) {
        group.Validate();
    }
    if (prefix_replay_tokens < 0) {
        throw std::invalid_argument("Scheduler: prefix_replay_tokens must be >= 0");
    }
    if (enable_l3_storage && !HasHostCache()) {
        throw std::invalid_argument("Scheduler: L3 storage requires Host L2 cache");
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
    for (const CacheGroupConfig& group : cache_groups) {
        validateGroupCapacityInputs(*this, group);
        // A recurrent state advances one whole checkpoint at a time, so a chunk
        // must be able to cover one cache block.
        if (group.IsSnapshotStateGroup() && max_scheduled_tokens < prefix_granularity) {
            throw std::invalid_argument("Scheduler: Mamba max_scheduled_tokens must cover one cache block");
        }
        if (group.replayable) {
            replay_window_tokens = std::max(replay_window_tokens, *group.sliding_window_tokens);
        }
    }
    if (replay_window_tokens > 0) {
        // A prefix hit re-feeds up to one replay window and must still advance:
        // by every new token when fewer than a window remain, or by one prefix
        // page when a promotion boundary aligns the chunk.
        if (max_scheduled_tokens < replay_window_tokens + std::max(replay_window_tokens, prefix_granularity)) {
            throw std::invalid_argument(
                "Scheduler: max_scheduled_tokens must cover the replayable groups' largest sliding_window_tokens "
                "plus max(that window, prefix_granularity)");
        }
        // The final-chunk window rule and the state-checkpoint chunk alignment
        // would each reshape the other's chunk; no model needs both.
        for (const CacheGroupConfig& group : cache_groups) {
            if (group.IsSnapshotStateGroup()) {
                throw std::invalid_argument(
                    "Scheduler: bounded-replay cache groups cannot be combined with snapshot-state groups");
            }
        }
    }
}

}  // namespace tokenspeed
