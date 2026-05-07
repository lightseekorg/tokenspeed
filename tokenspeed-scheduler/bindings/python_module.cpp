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

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "scheduler/outside_events/inc.h"
#include "scheduler/operations/inc.h"
#include "scheduler/execution_event.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"

/*
Writable types:
1. SchedulerConfig
2. RequestSpec
3. ForwardEvent
4. AbortEvent
5. cache::*DoneEvent

All other types are produced by the scheduler and consumed by Python, so they do
not need writable properties.
*/

namespace nb = nanobind;

namespace {

template <typename Op, typename Cls>
void BindForwardCommonFields(Cls& cls) {
    cls.def_prop_ro(
           "request_ids", [](const Op& op) -> const std::vector<std::string>& { return op.request_ids; },
           nb::rv_policy::reference_internal)
        .def_prop_ro(
            "request_pool_indices",
            [](const Op& op) -> const std::vector<std::int32_t>& { return op.request_pool_indices; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "input_lengths", [](const Op& op) -> const std::vector<std::int32_t>& { return op.input_lengths; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "occupied_pages",
            [](const Op& op) -> const std::vector<std::vector<std::int32_t>>& { return op.occupied_pages; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "begins", [](const Op& op) -> const std::vector<std::int32_t>& { return op.begins; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "sizes", [](const Op& op) -> const std::vector<std::int32_t>& { return op.sizes; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "new_occupied_pages",
            [](const Op& op) {
                std::vector<std::vector<std::int32_t>> result;
                result.reserve(op.occupied_pages.size());
                for (std::size_t i = 0; i < op.occupied_pages.size(); ++i) {
                    const auto& pages = op.occupied_pages[i];
                    std::int32_t b = op.begins[i];
                    std::int32_t s = op.sizes[i];
                    result.emplace_back(pages.begin() + b, pages.begin() + b + s);
                }
                return result;
            },
            nb::rv_policy::copy);
}

template <typename Op, typename Cls>
void BindCacheCommonFields(Cls& cls) {
    cls.def_prop_ro(
           "op_id", [](const Op& op) -> const tokenspeed::cache_op_id& { return op.op_id; },
           nb::rv_policy::reference_internal)
        .def_prop_ro(
            "src_pages", [](const Op& op) -> const std::vector<std::int32_t>& { return op.src_pages; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "dst_pages", [](const Op& op) -> const std::vector<std::int32_t>& { return op.dst_pages; },
            nb::rv_policy::reference_internal);
}

}  // namespace

NB_MODULE(tokenspeed_scheduler_ext, m) {
    m.doc() = "TokenSpeed scheduler bindings";

    nb::class_<tokenspeed::SchedulerStats>(m, "SchedulerStats")
        .def(nb::init<>())
        .def_ro("total_batches", &tokenspeed::SchedulerStats::total_batches)
        .def_ro("mixed_batches", &tokenspeed::SchedulerStats::mixed_batches)
        .def_ro("retract_count", &tokenspeed::SchedulerStats::retract_count)
        .def_ro("abort_count", &tokenspeed::SchedulerStats::abort_count)
        .def_ro("schedule_latency_count", &tokenspeed::SchedulerStats::schedule_latency_count)
        .def_ro("schedule_latency_sum_us", &tokenspeed::SchedulerStats::schedule_latency_sum_us)
        .def_ro("schedule_latency_max_us", &tokenspeed::SchedulerStats::schedule_latency_max_us)
        .def_ro("prefix_cache_hit_tokens", &tokenspeed::SchedulerStats::prefix_cache_hit_tokens)
        .def_ro("prefix_cache_req_tokens", &tokenspeed::SchedulerStats::prefix_cache_req_tokens)
        .def_ro("pending_queue_size", &tokenspeed::SchedulerStats::pending_queue_size)
        .def_ro("plan_queue_size", &tokenspeed::SchedulerStats::plan_queue_size)
        .def_ro("event_queue_size", &tokenspeed::SchedulerStats::event_queue_size)
        .def_ro("active_requests", &tokenspeed::SchedulerStats::active_requests);

    nb::enum_<tokenspeed::DisaggregationMode>(m, "DisaggregationMode")
        .value("none", tokenspeed::DisaggregationMode::kNone)
        .value("prefill", tokenspeed::DisaggregationMode::kPrefill)
        .value("decode", tokenspeed::DisaggregationMode::kDecode);

    auto scheduler_config = nb::class_<tokenspeed::SchedulerConfig>(m, "SchedulerConfig");

    nb::enum_<tokenspeed::Role>(scheduler_config, "Role")
        .value("P", tokenspeed::Role::kP)
        .value("D", tokenspeed::Role::kD)
        .value("Fused", tokenspeed::Role::kFused);

    scheduler_config.def(nb::init<>())
        .def_rw("page_size", &tokenspeed::SchedulerConfig::page_size)
        .def_rw("max_scheduled_tokens", &tokenspeed::SchedulerConfig::max_scheduled_tokens)
        .def_rw("max_batch_size", &tokenspeed::SchedulerConfig::max_batch_size)
        .def_rw("decode_input_tokens", &tokenspeed::SchedulerConfig::decode_input_tokens)
        .def_rw("role", &tokenspeed::SchedulerConfig::role)
        .def_prop_rw(
            "num_device_pages", [](const tokenspeed::SchedulerConfig& c) { return c.device_allocator.total_pages; },
            [](tokenspeed::SchedulerConfig& c, std::int32_t v) { c.device_allocator.total_pages = v; })
        .def_prop_rw(
            "num_host_pages", [](const tokenspeed::SchedulerConfig& c) { return c.host_allocator.total_pages; },
            [](tokenspeed::SchedulerConfig& c, std::int32_t v) { c.host_allocator.total_pages = v; })
        .def_rw("disable_l2_cache", &tokenspeed::SchedulerConfig::disable_l2_cache)
        .def_rw("enable_l3_storage", &tokenspeed::SchedulerConfig::enable_l3_storage)
        .def_rw("prefetch_threshold", &tokenspeed::SchedulerConfig::prefetch_threshold)
        .def_rw("num_mamba_slots", &tokenspeed::SchedulerConfig::num_mamba_slots);

    nb::class_<tokenspeed::RequestSpec>(m, "RequestSpec")
        .def(nb::init<>())
        .def_rw("request_id", &tokenspeed::RequestSpec::request_id)
        .def_rw("tokens", &tokenspeed::RequestSpec::tokens)
        .def_rw("rolling_hashes", &tokenspeed::RequestSpec::rolling_hashes)
        .def_rw("storage_hit_pages", &tokenspeed::RequestSpec::storage_hit_pages)
        .def_rw("lora_id", &tokenspeed::RequestSpec::lora_id);

    nb::module_ forward_event = m.def_submodule("ForwardEvent");
    nb::class_<tokenspeed::forward::ExtendResult>(forward_event, "ExtendResult")
        .def(nb::init<>())
        .def_rw("request_id", &tokenspeed::forward::ExtendResult::request_id)
        .def_rw("tokens", &tokenspeed::forward::ExtendResult::tokens);

    nb::class_<tokenspeed::forward::Finish>(forward_event, "Finish")
        .def(nb::init<>())
        .def_rw("request_id", &tokenspeed::forward::Finish::request_id);

    nb::class_<tokenspeed::forward::Abort>(forward_event, "Abort")
        .def(nb::init<>())
        .def_rw("request_id", &tokenspeed::forward::Abort::request_id);

    nb::class_<tokenspeed::forward::UpdateReserveNumTokens>(forward_event, "UpdateReserveNumTokens")
        .def(nb::init<>())
        .def_rw("request_id", &tokenspeed::forward::UpdateReserveNumTokens::request_id)
        .def_rw("reserve_num_tokens_in_next_schedule_event",
                &tokenspeed::forward::UpdateReserveNumTokens::reserve_num_tokens_in_next_schedule_event);

    // ─── ExecutionEvent ─────────────────────────────────────────────

    nb::module_ pd = m.def_submodule("PD");
    nb::module_ cache = m.def_submodule("Cache");

    nb::class_<tokenspeed::cache::PrefetchDone>(cache, "PrefetchDoneEvent")
        .def(nb::init<>())
        .def_rw("success", &tokenspeed::cache::PrefetchDone::success)
        .def_rw("op_id", &tokenspeed::cache::PrefetchDone::op_id)
        .def_rw("request_id", &tokenspeed::cache::PrefetchDone::request_id)
        .def_rw("completed_pages", &tokenspeed::cache::PrefetchDone::completed_pages);

    nb::class_<tokenspeed::cache::WriteBackDone>(cache, "WriteBackDoneEvent")
        .def(nb::init<>())
        .def_rw("op_id", &tokenspeed::cache::WriteBackDone::op_id)
        .def_rw("success", &tokenspeed::cache::WriteBackDone::success);

    nb::class_<tokenspeed::pd::BootstrappedEvent>(pd, "BootstrappedEvent")
        .def(nb::init<std::string>(), nb::arg("request_id"))
        .def_ro("request_id", &tokenspeed::pd::BootstrappedEvent::request_id);

    nb::class_<tokenspeed::pd::FailedEvent>(pd, "FailedEvent")
        .def(nb::init<std::string>(), nb::arg("request_id"))
        .def_ro("request_id", &tokenspeed::pd::FailedEvent::request_id);

    nb::class_<tokenspeed::pd::SucceededEvent>(pd, "SucceededEvent")
        .def(nb::init<std::string>(), nb::arg("request_id"))
        .def_ro("request_id", &tokenspeed::pd::SucceededEvent::request_id);

    nb::class_<tokenspeed::pd::RemotePrefillDoneEvent>(pd, "RemotePrefillDoneEvent")
        .def(nb::init<std::string, int32_t>(), nb::arg("request_id"), nb::arg("bootstrap_token"))
        .def_ro("request_id", &tokenspeed::pd::RemotePrefillDoneEvent::request_id)
        .def_rw("bootstrap_token", &tokenspeed::pd::RemotePrefillDoneEvent::bootstrap_token);

    nb::class_<tokenspeed::ExecutionEvent>(m, "ExecutionEvent")
        .def(nb::init<>())
        .def(
            "add_event",
            [](tokenspeed::ExecutionEvent& self, tokenspeed::Event e) -> tokenspeed::ExecutionEvent& {
                return self.With(std::move(e));
            },
            nb::arg("event"), nb::rv_policy::reference);

    nb::module_ forward = m.def_submodule("Forward");

    auto flat_fwd_op = nb::class_<tokenspeed::FlatForwardOperation>(forward, "FlatForwardOp");
    BindForwardCommonFields<tokenspeed::FlatForwardOperation>(flat_fwd_op);
    flat_fwd_op.def_ro("input_ids", &tokenspeed::FlatForwardOperation::input_ids)
        .def_ro("shifted_input_ids", &tokenspeed::FlatForwardOperation::shifted_input_ids)
        .def_ro("extend_prefix_lens", &tokenspeed::FlatForwardOperation::extend_prefix_lens)
        .def_prop_ro(
            "prefill_lengths",
            [](const tokenspeed::FlatForwardOperation& op) -> const std::vector<std::int32_t>& {
                return op.prefill_lengths;
            },
            nb::rv_policy::reference_internal)
        .def_ro("decode_input_ids", &tokenspeed::FlatForwardOperation::decode_input_ids)
        .def_rw("hist_token_lens", &tokenspeed::FlatForwardOperation::hist_token_lens)
        .def("num_extends", &tokenspeed::FlatForwardOperation::num_extends)
        .def_ro("mamba_pool_indices", &tokenspeed::FlatForwardOperation::mamba_working_indices)
        .def_ro("mamba_track_pool_indices", &tokenspeed::FlatForwardOperation::mamba_checkpoint_dst_indices)
        .def_ro("mamba_cow_src_indices", &tokenspeed::FlatForwardOperation::mamba_cow_src_indices)
        .def_ro("mamba_branching_seqlens", &tokenspeed::FlatForwardOperation::mamba_branching_seqlens);

    // ─── CacheOperation (attached to the Cache submodule) ──────────
    auto prefetch_op = nb::class_<tokenspeed::PrefetchOperation>(cache, "PrefetchOp");
    BindCacheCommonFields<tokenspeed::PrefetchOperation>(prefetch_op);
    prefetch_op.def(nb::init<>())
        .def_ro("request_id", &tokenspeed::PrefetchOperation::request_id)
        .def_ro("rolling_page_hashes", &tokenspeed::PrefetchOperation::rolling_page_hashes);

    auto backup_op = nb::class_<tokenspeed::BackUpOperation>(cache, "BackUpOp");
    BindCacheCommonFields<tokenspeed::BackUpOperation>(backup_op);
    backup_op.def(nb::init<>()).def_ro("rolling_page_hashes", &tokenspeed::BackUpOperation::rolling_page_hashes);

    nb::class_<tokenspeed::FlatLoadBackOperation>(cache, "LoadBackOp")
        .def_ro("op_ids", &tokenspeed::FlatLoadBackOperation::op_ids)
        .def_ro("src_pages", &tokenspeed::FlatLoadBackOperation::src_pages)
        .def_ro("dst_pages", &tokenspeed::FlatLoadBackOperation::dst_pages);

    nb::class_<tokenspeed::FlatWriteBackOperation>(cache, "WriteBackOp")
        .def_ro("op_ids", &tokenspeed::FlatWriteBackOperation::op_ids)
        .def_ro("src_pages", &tokenspeed::FlatWriteBackOperation::src_pages)
        .def_ro("dst_pages", &tokenspeed::FlatWriteBackOperation::dst_pages)
        .def_ro("is_retract", &tokenspeed::FlatWriteBackOperation::is_retract);

    auto collect_forward = [](const tokenspeed::ExecutionPlan& plan) -> nb::list {
        nb::list result;
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<tokenspeed::FlatForwardOperation>(&op)) {
                result.append(nb::cast(*f, nb::rv_policy::copy));
            }
        }
        return result;
    };

    auto collect_cache = [](const tokenspeed::ExecutionPlan& plan) -> nb::list {
        nb::list result;
        for (const auto& op : plan.Operations()) {
            if (auto* c = std::get_if<tokenspeed::CacheOperation>(&op)) {
                std::visit([&result](const auto& inner) { result.append(nb::cast(inner, nb::rv_policy::copy)); }, *c);
            }
        }
        return result;
    };

    nb::class_<tokenspeed::ExecutionPlan>(m, "ExecutionPlan")
        .def(nb::init<>())
        .def_prop_ro("forward", collect_forward)
        .def_prop_ro("cache", collect_cache);

    nb::class_<tokenspeed::Scheduler>(m, "Scheduler")
        .def(nb::init<tokenspeed::SchedulerConfig>(), nb::arg("config") = tokenspeed::SchedulerConfig{})
        .def("submit_requests",
             nb::overload_cast<const std::vector<tokenspeed::RequestSpec>&>(&tokenspeed::Scheduler::SubmitRequests),
             nb::arg("request_specs"))
        .def("next_execution_plan", [](tokenspeed::Scheduler& s) { return s.NextExecutionPlan(); })
        .def("advance", &tokenspeed::Scheduler::Advance, nb::arg("event"))
        .def("waiting_size", &tokenspeed::Scheduler::WaitingSize)
        .def("decoding_size", &tokenspeed::Scheduler::DecodingSize)
        .def("prefilling_size", &tokenspeed::Scheduler::PrefillSize)
        .def("retract_count", &tokenspeed::Scheduler::RetractedSize)
        .def("available_kv_pages", &tokenspeed::Scheduler::AvailableKvPages)
        .def("active_kv_pages", &tokenspeed::Scheduler::ActiveKvPages)
        .def("get_request_token_size", &tokenspeed::Scheduler::GetRequestTokenSize, nb::arg("id"))
        .def("calc_rolling_hash", &tokenspeed::Scheduler::CalcRollingHash, nb::arg("input_tokens"),
             nb::arg("apply_match") = false);
}
