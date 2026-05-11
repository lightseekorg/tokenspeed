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

#include <optional>
#include <variant>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#include "fsm/forward_events.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/types.h"
#include "scheduler/operations/inc.h"

namespace tokenspeed {

class TreeNode;

enum class DisaggregationMode {
    kNone,
    kPrefill,
    kDecode,
};

template <ResourceType>
class NodeRef;
using HostNodeRef = NodeRef<ResourceType::Host>;
using DeviceNodeRef = NodeRef<ResourceType::Device>;

struct SchedulerStats {
    std::int64_t total_batches = 0;
    std::int64_t mixed_batches = 0;
    std::int64_t retract_count = 0;
    std::int64_t abort_count = 0;
    std::int64_t schedule_latency_count = 0;
    std::int64_t schedule_latency_sum_us = 0;
    std::int64_t schedule_latency_max_us = 0;
    std::int64_t prefix_cache_hit_tokens = 0;
    std::int64_t prefix_cache_req_tokens = 0;

    std::int64_t pending_queue_size = 0;
    std::int64_t plan_queue_size = 0;
    std::int64_t event_queue_size = 0;
    std::int64_t active_requests = 0;
};

struct SchedulerConfig {
    std::int32_t page_size{};
    struct {
        std::int32_t total_pages{};
    } host_allocator;

    struct {
        std::int32_t total_pages{};
    } device_allocator;

    std::vector<PagedCacheGroupConfig> paged_cache_groups{};

    std::int32_t max_scheduled_tokens{};
    std::int32_t max_batch_size{};
    std::int32_t decode_input_tokens{1};
    bool disable_l2_cache{false};
    bool enable_l3_storage{false};
    std::int32_t prefetch_threshold{4};  // num pages

    std::int32_t num_pages_reserved_for_retracted_or_running{};
    Role role{Role::kFused};

    std::int32_t num_mamba_slots{0};
    bool disable_prefix_cache{false};
    bool enable_mamba{false};
    std::int32_t mamba_cache_chunk_size{64};
    std::int32_t mamba_pool_total_chunks{0};
};

}  // namespace tokenspeed
