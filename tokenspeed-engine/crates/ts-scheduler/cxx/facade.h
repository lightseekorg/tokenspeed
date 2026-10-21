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

// The control thread's view of the C++ scheduler: plain-value requests and
// events in, plain-value copies of what the control plane reads out.

#pragma once

#include <memory>

#include "rust/cxx.h"
#include "scheduler/execution_plan.h"
#include "scheduler/scheduler.h"

namespace ts_scheduler {

struct RequestSpec;
struct ForwardEvent;
struct CacheEvent;
struct CacheOpView;
struct ForwardBatchView;
struct SchedulerAbortView;
struct SchedulerSizes;
struct CacheGroupPages;
struct KvEventView;
struct CacheKeyView;

void submit_requests(tokenspeed::Scheduler& scheduler, rust::Slice<const RequestSpec> specs);
std::unique_ptr<tokenspeed::ExecutionPlan> next_execution_plan(tokenspeed::Scheduler& scheduler);
void advance(tokenspeed::Scheduler& scheduler, rust::Slice<const ForwardEvent> events);
void advance_cache(tokenspeed::Scheduler& scheduler, rust::Slice<const CacheEvent> events);
SchedulerSizes sizes(const tokenspeed::Scheduler& scheduler);
CacheGroupPages cache_group_pages(const tokenspeed::Scheduler& scheduler, rust::Str group_id);
rust::Vec<KvEventView> drain_kv_events(tokenspeed::Scheduler& scheduler);
rust::Vec<rust::String> prefix_hashes_for_tokens(const tokenspeed::Scheduler& scheduler,
                                                 rust::Slice<const std::int32_t> tokens);
rust::Vec<CacheKeyView> expand_prefix_keys(const tokenspeed::Scheduler& scheduler,
                                           rust::Slice<const rust::String> content_hashes);
void register_storage_keys(tokenspeed::Scheduler& scheduler, rust::Slice<const CacheKeyView> keys);
bool clear_cache(tokenspeed::Scheduler& scheduler);
bool clear_l1_cache(tokenspeed::Scheduler& scheduler);
bool can_clear_cache(const tokenspeed::Scheduler& scheduler);
std::int32_t max_single_request_tokens(const tokenspeed::Scheduler& scheduler);

rust::Vec<ForwardBatchView> plan_forward_batches(const tokenspeed::ExecutionPlan& plan);
bool plan_needs_device(const tokenspeed::ExecutionPlan& plan);
rust::Vec<ForwardBatchView> plan_remote_decode(const tokenspeed::ExecutionPlan& plan);
rust::Vec<ForwardBatchView> plan_remote_prefill(const tokenspeed::ExecutionPlan& plan);
rust::Vec<SchedulerAbortView> plan_aborts(const tokenspeed::ExecutionPlan& plan);
rust::Vec<CacheOpView> plan_cache_ops(const tokenspeed::ExecutionPlan& plan);

}  // namespace ts_scheduler
