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
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "scheduler/outside_events/inc.h"
#include "scheduler/operations/inc.h"
#include "scheduler/execution_event.h"
#include "scheduler/kv_cache_events.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"
#include "utils.h"

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
            nb::rv_policy::reference_internal);
}

}  // namespace

NB_MODULE(tokenspeed_scheduler_ext, m) {
    m.doc() = "TokenSpeed scheduler bindings";

    nb::module_ kv_event = m.def_submodule("KVEvent");
    nb::class_<tokenspeed::KvBlockStoredEvent>(kv_event, "BlockStored")
        .def_prop_ro("kind", [](const tokenspeed::KvBlockStoredEvent&) { return "BlockStored"; })
        .def_ro("block_hashes", &tokenspeed::KvBlockStoredEvent::block_hashes)
        .def_ro("parent_block_hash", &tokenspeed::KvBlockStoredEvent::parent_block_hash)
        .def_ro("token_ids", &tokenspeed::KvBlockStoredEvent::token_ids)
        .def_ro("block_size", &tokenspeed::KvBlockStoredEvent::block_size);

    nb::class_<tokenspeed::KvBlockRemovedEvent>(kv_event, "BlockRemoved")
        .def_prop_ro("kind", [](const tokenspeed::KvBlockRemovedEvent&) { return "BlockRemoved"; })
        .def_ro("block_hashes", &tokenspeed::KvBlockRemovedEvent::block_hashes);

    auto scheduler_config = nb::class_<tokenspeed::SchedulerConfig>(m, "SchedulerConfig");

    nb::enum_<tokenspeed::Role>(scheduler_config, "Role")
        .value("P", tokenspeed::Role::kP)
        .value("D", tokenspeed::Role::kD)
        .value("Fused", tokenspeed::Role::kFused);

    nb::enum_<tokenspeed::CacheGroupConfig::Retention>(m, "CacheRetention")
        .value("FullHistory", tokenspeed::CacheGroupConfig::Retention::FullHistory)
        .value("SlidingWindow", tokenspeed::CacheGroupConfig::Retention::SlidingWindow);

    nb::enum_<tokenspeed::CacheGroupFamily>(m, "CacheGroupFamily")
        .value("History", tokenspeed::CacheGroupFamily::History)
        .value("State", tokenspeed::CacheGroupFamily::State);

    nb::enum_<tokenspeed::CacheTransferPolicy>(m, "CacheTransferPolicy")
        .value("Unspecified", tokenspeed::CacheTransferPolicy::Unspecified)
        .value("FullSuffix", tokenspeed::CacheTransferPolicy::FullSuffix)
        .value("LatestSnapshot", tokenspeed::CacheTransferPolicy::LatestSnapshot);

    nb::class_<tokenspeed::CacheGroupConfig>(m, "CacheGroupConfig")
        .def(nb::init<>())
        .def(
            "__init__",
            [](tokenspeed::CacheGroupConfig* self, std::string group_id, std::int32_t rows_per_page,
               std::int32_t entry_stride_tokens, std::int32_t total_pages,
               tokenspeed::CacheGroupConfig::Retention retention, std::optional<std::int32_t> sliding_window_tokens,
               tokenspeed::CacheGroupFamily family, std::int32_t cache_blocks_per_lcm_block,
               tokenspeed::CacheTransferPolicy transfer_policy) {
                new (self) tokenspeed::CacheGroupConfig{std::move(group_id),
                                                        rows_per_page,
                                                        entry_stride_tokens,
                                                        total_pages,
                                                        cache_blocks_per_lcm_block,
                                                        retention,
                                                        sliding_window_tokens,
                                                        family,
                                                        transfer_policy};
            },
            nb::arg("group_id"), nb::arg("rows_per_page"), nb::arg("entry_stride_tokens"), nb::arg("total_pages"),
            nb::arg("retention") = tokenspeed::CacheGroupConfig::Retention::FullHistory,
            nb::arg("sliding_window_tokens") = std::nullopt, nb::arg("family") = tokenspeed::CacheGroupFamily::History,
            nb::arg("cache_blocks_per_lcm_block") = 1,
            nb::arg("transfer_policy") = tokenspeed::CacheTransferPolicy::Unspecified)
        .def_rw("group_id", &tokenspeed::CacheGroupConfig::group_id)
        .def_rw("rows_per_page", &tokenspeed::CacheGroupConfig::rows_per_page)
        .def_rw("entry_stride_tokens", &tokenspeed::CacheGroupConfig::entry_stride_tokens)
        .def_rw("total_pages", &tokenspeed::CacheGroupConfig::total_pages)
        .def_rw("cache_blocks_per_lcm_block", &tokenspeed::CacheGroupConfig::cache_blocks_per_lcm_block)
        .def_rw("retention", &tokenspeed::CacheGroupConfig::retention)
        .def_rw("sliding_window_tokens", &tokenspeed::CacheGroupConfig::sliding_window_tokens)
        .def_rw("family", &tokenspeed::CacheGroupConfig::family)
        .def_rw("transfer_policy", &tokenspeed::CacheGroupConfig::transfer_policy)
        .def("validate", &tokenspeed::CacheGroupConfig::Validate);

    scheduler_config.def(nb::init<>())
        .def_rw("prefix_granularity", &tokenspeed::SchedulerConfig::prefix_granularity)
        .def_rw("max_scheduled_tokens", &tokenspeed::SchedulerConfig::max_scheduled_tokens)
        .def_rw("max_batch_size", &tokenspeed::SchedulerConfig::max_batch_size)
        .def_rw("decode_input_tokens", &tokenspeed::SchedulerConfig::decode_input_tokens)
        .def_rw("overlap_schedule_depth", &tokenspeed::SchedulerConfig::overlap_schedule_depth)
        .def_rw("role", &tokenspeed::SchedulerConfig::role)
        .def_prop_rw(
            "num_device_pages", [](const tokenspeed::SchedulerConfig& c) { return c.device_allocator.total_pages; },
            [](tokenspeed::SchedulerConfig& c, std::int32_t v) { c.device_allocator.total_pages = v; })
        .def_prop_rw(
            "num_host_pages", [](const tokenspeed::SchedulerConfig& c) { return c.host_allocator.total_pages; },
            [](tokenspeed::SchedulerConfig& c, std::int32_t v) { c.host_allocator.total_pages = v; })
        .def_rw("cache_groups", &tokenspeed::SchedulerConfig::cache_groups)
        .def_rw("disable_l2_cache", &tokenspeed::SchedulerConfig::disable_l2_cache)
        .def_rw("enable_l3_storage", &tokenspeed::SchedulerConfig::enable_l3_storage)
        .def_rw("enable_kv_cache_events", &tokenspeed::SchedulerConfig::enable_kv_cache_events)
        .def_rw("enable_mixed_prefill_decode", &tokenspeed::SchedulerConfig::enable_mixed_prefill_decode)
        .def_rw("disable_prefix_cache", &tokenspeed::SchedulerConfig::disable_prefix_cache)
        .def_rw("prefix_replay_tokens", &tokenspeed::SchedulerConfig::prefix_replay_tokens);

    nb::class_<tokenspeed::RequestSpec>(m, "RequestSpec")
        .def(nb::init<>())
        .def_rw("request_id", &tokenspeed::RequestSpec::request_id)
        .def_rw("tokens", &tokenspeed::RequestSpec::tokens)
        .def_rw("max_new_tokens", &tokenspeed::RequestSpec::max_new_tokens);

    nb::module_ forward_event = m.def_submodule("ForwardEvent");
    nb::class_<tokenspeed::forward::ExtendResult>(forward_event, "ExtendResult")
        .def(nb::init<>())
        .def_rw("request_id", &tokenspeed::forward::ExtendResult::request_id)
        .def_rw("tokens", &tokenspeed::forward::ExtendResult::tokens)
        .def_rw("spec_candidate_ids", &tokenspeed::forward::ExtendResult::spec_candidate_ids);

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

    nb::class_<tokenspeed::cache::WriteBackDone>(cache, "WriteBackDoneEvent")
        .def(nb::init<>())
        .def_rw("op_id", &tokenspeed::cache::WriteBackDone::op_id);

    nb::class_<tokenspeed::cache::LoadBackDone>(cache, "LoadBackDoneEvent")
        .def(nb::init<>())
        .def_rw("op_id", &tokenspeed::cache::LoadBackDone::op_id);

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

    auto forward_batch = nb::class_<tokenspeed::ForwardBatch>(forward, "Batch");
    BindForwardCommonFields<tokenspeed::ForwardBatch>(forward_batch);
    forward_batch.def_ro("input_ids", &tokenspeed::ForwardBatch::input_ids)
        .def_ro("shifted_input_ids", &tokenspeed::ForwardBatch::shifted_input_ids)
        .def_ro("extend_prefix_lens", &tokenspeed::ForwardBatch::extend_prefix_lens)
        .def_prop_ro(
            "prefill_lengths",
            [](const tokenspeed::ForwardBatch& op) -> const std::vector<std::int32_t>& { return op.prefill_lengths; },
            nb::rv_policy::reference_internal)
        .def_ro("decode_input_ids", &tokenspeed::ForwardBatch::decode_input_ids)
        .def_ro("spec_candidate_ids", &tokenspeed::ForwardBatch::spec_candidate_ids)
        .def_prop_ro(
            "block_tables",
            [](const tokenspeed::ForwardBatch& op)
                -> const std::map<std::string, std::vector<std::vector<std::int32_t>>>& { return op.block_tables; },
            nb::rv_policy::reference_internal)
        .def("block_tables_arrays",
             [](nb::handle self) {
                 // Zero-copy 2-D int32 views; `self` keeps the backing
                 // ForwardBatch alive for the lifetime of each ndarray.
                 auto& op = nb::cast<tokenspeed::ForwardBatch&>(self);
                 nb::dict out;
                 for (auto& [gid, buf] : op.block_tables_contig) {
                     const std::size_t rows = op.request_ids.size();
                     tokenspeed::FatalCheck(rows == 0 || buf.size() % rows == 0,
                                            "block-table buffer must contain complete rows");
                     const std::size_t columns = rows == 0 ? 0 : buf.size() / rows;
                     out[nb::str(gid.c_str())] =
                         nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<2>>(buf.data(), {rows, columns}, self);
                 }
                 return out;
             })
        .def("num_extends", &tokenspeed::ForwardBatch::NumExtends);

    // ─── CacheOperation (attached to the Cache submodule) ──────────
    nb::class_<tokenspeed::LoadBackBatch>(cache, "LoadBackOp")
        .def_ro("op_ids", &tokenspeed::LoadBackBatch::op_ids)
        .def_ro("group_ids", &tokenspeed::LoadBackBatch::group_ids)
        .def_ro("src_pages", &tokenspeed::LoadBackBatch::src_pages)
        .def_ro("dst_pages", &tokenspeed::LoadBackBatch::dst_pages)
        .def_ro("content_hashes", &tokenspeed::LoadBackBatch::content_hashes)
        .def_ro("page_offsets", &tokenspeed::LoadBackBatch::page_offsets)
        .def_ro("prefetch_from_storage", &tokenspeed::LoadBackBatch::prefetch_from_storage);

    nb::class_<tokenspeed::WriteBackBatch>(cache, "WriteBackOp")
        .def_ro("op_ids", &tokenspeed::WriteBackBatch::op_ids)
        .def_ro("group_ids", &tokenspeed::WriteBackBatch::group_ids)
        .def_ro("src_pages", &tokenspeed::WriteBackBatch::src_pages)
        .def_ro("dst_pages", &tokenspeed::WriteBackBatch::dst_pages)
        .def_ro("content_hashes", &tokenspeed::WriteBackBatch::content_hashes)
        .def_ro("page_offsets", &tokenspeed::WriteBackBatch::page_offsets);

    auto collect_forward = [](const tokenspeed::ExecutionPlan& plan) -> nb::list {
        nb::list result;
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<tokenspeed::ForwardBatch>(&op)) {
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
        .def_prop_ro("cache", collect_cache)
        .def_prop_ro("remote_decode",
                     [](const tokenspeed::ExecutionPlan& plan) -> nb::object {
                         if (!plan.remote_decode) {
                             return nb::none();
                         }
                         return nb::cast(*plan.remote_decode, nb::rv_policy::copy);
                     })
        .def_prop_ro("remote_prefill",
                     [](const tokenspeed::ExecutionPlan& plan) -> nb::object {
                         if (!plan.remote_prefill) {
                             return nb::none();
                         }
                         return nb::cast(*plan.remote_prefill, nb::rv_policy::copy);
                     })
        .def_ro("pages_to_zero", &tokenspeed::ExecutionPlan::pages_to_zero);

    nb::class_<tokenspeed::Scheduler>(m, "Scheduler")
        .def(nb::init<tokenspeed::SchedulerConfig>(), nb::arg("config") = tokenspeed::SchedulerConfig{})
        .def("submit_requests",
             nb::overload_cast<const std::vector<tokenspeed::RequestSpec>&>(&tokenspeed::Scheduler::SubmitRequests),
             nb::arg("request_specs"))
        .def(
            "next_execution_plan", [](tokenspeed::Scheduler& s) { return s.NextExecutionPlan(); },
            nb::call_guard<nb::gil_scoped_release>())
        .def("advance", &tokenspeed::Scheduler::Advance, nb::arg("event"))
        .def("drain_kv_events",
             [](tokenspeed::Scheduler& s) {
                 nb::list result;
                 for (auto& event : s.DrainKvEvents()) {
                     std::visit([&result](auto& inner) { result.append(nb::cast(inner, nb::rv_policy::copy)); }, event);
                 }
                 return result;
             })
        .def("waiting_size", &tokenspeed::Scheduler::WaitingSize)
        .def("decoding_size", &tokenspeed::Scheduler::DecodingSize)
        .def("prefilling_size", &tokenspeed::Scheduler::PrefillSize)
        .def("pd_transfer_pinned", &tokenspeed::Scheduler::PdTransferPinned, nb::arg("request_id"))
        .def("available_kv_pages", &tokenspeed::Scheduler::AvailableKvPages)
        .def("active_kv_pages", &tokenspeed::Scheduler::ActiveKvPages)
        .def("request_token_size", &tokenspeed::Scheduler::RequestTokenSize, nb::arg("id"))
        .def("max_single_request_tokens", &tokenspeed::Scheduler::MaxSingleRequestTokens)
        .def("clear_l1_cache", &tokenspeed::Scheduler::ClearL1Cache)
        .def("clear_cache", &tokenspeed::Scheduler::ClearCache)
        .def("cache_group_total_pages", &tokenspeed::Scheduler::CacheGroupTotalPages, nb::arg("group_id"))
        .def("cache_group_available_pages", &tokenspeed::Scheduler::CacheGroupAvailablePages, nb::arg("group_id"))
        .def("prefix_hashes_for_tokens", &tokenspeed::Scheduler::PrefixHashesForTokens, nb::arg("tokens"))
        .def(
            "expand_prefix_keys",
            [](const tokenspeed::Scheduler& scheduler, const std::vector<std::string>& content_hashes) {
                const std::vector<tokenspeed::CacheKey> keys = scheduler.ExpandPrefixKeys(content_hashes);
                std::vector<std::uint32_t> group_ids;
                std::vector<std::string> hashes;
                std::vector<std::int32_t> page_offsets;
                group_ids.reserve(keys.size());
                hashes.reserve(keys.size());
                page_offsets.reserve(keys.size());
                for (const tokenspeed::CacheKey& key : keys) {
                    group_ids.push_back(key.group_id);
                    hashes.push_back(key.content_hash);
                    page_offsets.push_back(key.page_offset);
                }
                return std::tuple{std::move(group_ids), std::move(hashes), std::move(page_offsets)};
            },
            nb::arg("content_hashes"))
        .def(
            "register_storage_keys",
            [](tokenspeed::Scheduler& scheduler, const std::vector<std::uint32_t>& group_ids,
               const std::vector<std::string>& content_hashes, const std::vector<std::int32_t>& page_offsets) {
                if (group_ids.size() != content_hashes.size() || group_ids.size() != page_offsets.size()) {
                    throw std::invalid_argument("register_storage_keys requires aligned group/hash/offset lists");
                }
                std::vector<tokenspeed::CacheKey> keys;
                keys.reserve(group_ids.size());
                for (std::size_t i = 0; i < group_ids.size(); ++i) {
                    keys.push_back(tokenspeed::CacheKey{
                        .group_id = group_ids[i],
                        .content_hash = content_hashes[i],
                        .page_offset = page_offsets[i],
                    });
                }
                scheduler.RegisterStorageKeys(keys);
            },
            nb::arg("group_ids"), nb::arg("content_hashes"), nb::arg("page_offsets"));
}
