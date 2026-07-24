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
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "scheduler/outside_events/inc.h"
#include "scheduler/operations/inc.h"
#include "scheduler/execution_event.h"
#include "scheduler/kv_cache_events.h"
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

using WritableHostInt32Vector = nb::ndarray<nb::pytorch, std::int32_t, nb::ndim<1>, nb::c_contig>;
using HostInt64Vector = nb::ndarray<nb::pytorch, std::int64_t, nb::ndim<1>, nb::c_contig>;
using WritableHostInt64Matrix = nb::ndarray<nb::pytorch, std::int64_t, nb::ndim<2>, nb::c_contig>;

bool IsHostArray(const auto& array) {
    const int device_type = array.device_type();
    return device_type == nb::device::cpu::value || device_type == nb::device::cuda_host::value;
}

std::size_t CopyFlatBlockTablesTo(const tokenspeed::FlatForwardOperation& op, WritableHostInt32Vector destination,
                                  HostInt64Vector page_id_upper_bounds, WritableHostInt64Matrix copy_metadata,
                                  std::size_t payload_offset) {
    if (!IsHostArray(destination) || !IsHostArray(page_id_upper_bounds) || !IsHostArray(copy_metadata)) {
        throw std::invalid_argument("flat block table packed staging inputs must be host tensors");
    }
    if (page_id_upper_bounds.shape(0) != op.flat_block_tables.size()) {
        throw std::invalid_argument("flat block table packed staging bounds must match the exported groups");
    }
    if (copy_metadata.shape(0) != op.flat_block_tables.size() || copy_metadata.shape(1) != 4) {
        throw std::invalid_argument("flat block table packed staging metadata must have shape [groups, 4]");
    }

    const std::size_t capacity = destination.shape(0);
    if (payload_offset > capacity) {
        throw std::invalid_argument("flat block table packed staging payload offset exceeds the destination");
    }
    std::size_t cursor = payload_offset;
    std::size_t group_index = 0;
    for (const auto& [group_id, table] : op.flat_block_tables) {
        const std::size_t table_offset = cursor;
        if (table.values.size() > capacity - cursor) {
            throw std::invalid_argument("flat block table packed staging table capacity exceeded for group: " +
                                        group_id);
        }
        cursor += table.values.size();
        const std::size_t base_offset = cursor;
        if (table.rows > capacity - cursor) {
            throw std::invalid_argument("flat block table packed staging base capacity exceeded for group: " +
                                        group_id);
        }
        cursor += table.rows;
        const tokenspeed::FlatBlockTableExport::CopyResult copied =
            table.CopyTo(std::span<std::int32_t>{destination.data() + table_offset, table.values.size()},
                         std::span<std::int32_t>{destination.data() + base_offset, table.rows},
                         page_id_upper_bounds.data()[group_index]);
        std::int64_t* metadata = copy_metadata.data() + group_index * 4;
        metadata[0] = static_cast<std::int64_t>(copied.rows);
        metadata[1] = static_cast<std::int64_t>(copied.cols);
        metadata[2] = static_cast<std::int64_t>(table_offset);
        metadata[3] = static_cast<std::int64_t>(base_offset);
        ++group_index;
    }
    return cursor;
}

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

    // Build-time KV-cache path of this extension: true when compiled with
    // TOKENSPEED_FLAT_KVCACHE (flat KvCacheCoordinator FSM path), false for the
    // default radix LocalKVAllocator build. Python gates paged-cache group
    // publication — and therefore the flat CUDA-graph capture path — on this
    // flag; a radix build never populates flat_block_tables.
#if TOKENSPEED_FLAT_KVCACHE
    m.attr("FLAT_KVCACHE") = true;
#else
    m.attr("FLAT_KVCACHE") = false;
#endif

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

    nb::enum_<tokenspeed::PagedCacheGroupConfig::Retention>(m, "PagedCacheRetention")
        .value("FullHistory", tokenspeed::PagedCacheGroupConfig::Retention::FullHistory)
        .value("SlidingWindow", tokenspeed::PagedCacheGroupConfig::Retention::SlidingWindow);

    nb::enum_<tokenspeed::PagedCacheGroupFamily>(m, "PagedCacheGroupFamily")
        .value("History", tokenspeed::PagedCacheGroupFamily::History)
        .value("State", tokenspeed::PagedCacheGroupFamily::State);

    nb::enum_<tokenspeed::PrefixRole>(m, "PagedCachePrefixRole")
        .value("HistoryAnchor", tokenspeed::PrefixRole::HistoryAnchor)
        .value("ContinuationState", tokenspeed::PrefixRole::ContinuationState)
        .value("None_", tokenspeed::PrefixRole::None);

    nb::enum_<tokenspeed::TableLayout>(m, "PagedCacheTableLayout")
        .value("Absolute", tokenspeed::TableLayout::Absolute)
        .value("BoundedWindow", tokenspeed::TableLayout::BoundedWindow);

    nb::class_<tokenspeed::FlatBlockPoolConfig>(m, "FlatBlockPoolConfig")
        .def(nb::init<>())
        .def_rw("pool_id", &tokenspeed::FlatBlockPoolConfig::pool_id)
        .def_rw("total_blocks", &tokenspeed::FlatBlockPoolConfig::total_blocks)
        .def_rw("bytes_per_block", &tokenspeed::FlatBlockPoolConfig::bytes_per_block);

    nb::class_<tokenspeed::BlockPoolSnapshot>(m, "BlockPoolSnapshot")
        .def_ro("pool_id", &tokenspeed::BlockPoolSnapshot::pool_id)
        .def_ro("total_blocks", &tokenspeed::BlockPoolSnapshot::total_blocks)
        .def_ro("usable_blocks", &tokenspeed::BlockPoolSnapshot::usable_blocks)
        .def_ro("free_blocks", &tokenspeed::BlockPoolSnapshot::free_blocks)
        .def_ro("active_blocks", &tokenspeed::BlockPoolSnapshot::active_blocks)
        .def_ro("cached_evictable_blocks", &tokenspeed::BlockPoolSnapshot::cached_evictable_blocks)
        .def_ro("pinned_cached_blocks", &tokenspeed::BlockPoolSnapshot::pinned_cached_blocks)
        .def_ro("reserved_blocks", &tokenspeed::BlockPoolSnapshot::reserved_blocks)
        .def_ro("bytes_per_block", &tokenspeed::BlockPoolSnapshot::bytes_per_block);

    nb::class_<tokenspeed::FlatPoolAggregate>(m, "FlatPoolAggregate")
        .def_ro("active_bytes", &tokenspeed::FlatPoolAggregate::active_bytes)
        .def_ro("capacity_bytes", &tokenspeed::FlatPoolAggregate::capacity_bytes)
        .def_ro("pressure_numerator", &tokenspeed::FlatPoolAggregate::pressure_numerator)
        .def_ro("pressure_denominator", &tokenspeed::FlatPoolAggregate::pressure_denominator);

    nb::class_<tokenspeed::PagedCacheGroupConfig>(m, "PagedCacheGroupConfig")
        .def(nb::init<>())
        .def(
            "__init__",
            [](tokenspeed::PagedCacheGroupConfig* self, std::string group_id, std::int32_t rows_per_page,
               std::int32_t entry_stride_tokens, std::int32_t total_pages,
               tokenspeed::PagedCacheGroupConfig::Retention retention,
               std::optional<std::int32_t> sliding_window_tokens, tokenspeed::PagedCacheGroupFamily family,
               std::int32_t block_size, std::string pool_id, tokenspeed::PrefixRole prefix_role,
               tokenspeed::TableLayout table_layout, std::uint32_t owner_mask) {
                new (self) tokenspeed::PagedCacheGroupConfig{
                    std::move(group_id),   rows_per_page, entry_stride_tokens, total_pages, block_size,   retention,
                    sliding_window_tokens, family,        std::move(pool_id),  prefix_role, table_layout, owner_mask};
            },
            nb::arg("group_id"), nb::arg("rows_per_page"), nb::arg("entry_stride_tokens"), nb::arg("total_pages"),
            nb::arg("retention") = tokenspeed::PagedCacheGroupConfig::Retention::FullHistory,
            nb::arg("sliding_window_tokens") = std::nullopt,
            nb::arg("family") = tokenspeed::PagedCacheGroupFamily::History, nb::arg("block_size") = 0,
            nb::arg("pool_id") = "", nb::arg("prefix_role") = tokenspeed::PrefixRole::HistoryAnchor,
            nb::arg("table_layout") = tokenspeed::TableLayout::Absolute, nb::arg("owner_mask") = 0)
        .def_rw("group_id", &tokenspeed::PagedCacheGroupConfig::group_id)
        .def_rw("rows_per_page", &tokenspeed::PagedCacheGroupConfig::rows_per_page)
        .def_rw("entry_stride_tokens", &tokenspeed::PagedCacheGroupConfig::entry_stride_tokens)
        .def_rw("total_pages", &tokenspeed::PagedCacheGroupConfig::total_pages)
        .def_rw("block_size", &tokenspeed::PagedCacheGroupConfig::block_size)
        .def_rw("retention", &tokenspeed::PagedCacheGroupConfig::retention)
        .def_rw("sliding_window_tokens", &tokenspeed::PagedCacheGroupConfig::sliding_window_tokens)
        .def_rw("family", &tokenspeed::PagedCacheGroupConfig::family)
        .def_rw("pool_id", &tokenspeed::PagedCacheGroupConfig::pool_id)
        .def_rw("prefix_role", &tokenspeed::PagedCacheGroupConfig::prefix_role)
        .def_rw("table_layout", &tokenspeed::PagedCacheGroupConfig::table_layout)
        .def_rw("owner_mask", &tokenspeed::PagedCacheGroupConfig::owner_mask)
        .def("raw_tokens_per_page", &tokenspeed::PagedCacheGroupConfig::RawTokensPerPage)
        .def("validate", &tokenspeed::PagedCacheGroupConfig::Validate)
        .def("validate_flat_block_geometry", &tokenspeed::PagedCacheGroupConfig::ValidateFlatBlockGeometry);

    nb::class_<tokenspeed::PagedCacheGroupAllocator>(m, "PagedCacheGroupAllocator")
        .def(nb::init<tokenspeed::PagedCacheGroupConfig>(), nb::arg("config"))
        .def("allocate", &tokenspeed::PagedCacheGroupAllocator::Allocate, nb::arg("num_pages"))
        .def("deallocate", &tokenspeed::PagedCacheGroupAllocator::Deallocate, nb::arg("pages"))
        .def("config", &tokenspeed::PagedCacheGroupAllocator::Config, nb::rv_policy::reference_internal)
        .def("total_pages", &tokenspeed::PagedCacheGroupAllocator::TotalPages)
        .def("available_pages", &tokenspeed::PagedCacheGroupAllocator::AvailablePages)
        .def("allocated_pages_total", &tokenspeed::PagedCacheGroupAllocator::AllocatedPagesTotal)
        .def("released_pages_total", &tokenspeed::PagedCacheGroupAllocator::ReleasedPagesTotal)
        .def("failed_alloc_count", &tokenspeed::PagedCacheGroupAllocator::FailedAllocCount);

    nb::class_<tokenspeed::PagedCacheGroupTable>(m, "PagedCacheGroupTable")
        .def(nb::init<tokenspeed::PagedCacheGroupAllocator*>(), nb::arg("allocator"), nb::keep_alive<1, 2>())
        .def("acquire", &tokenspeed::PagedCacheGroupTable::Acquire, nb::arg("target_raw_tokens_exclusive"))
        .def("release_skipped", &tokenspeed::PagedCacheGroupTable::ReleaseSkipped, nb::arg("window_lower_bound"))
        .def("release_all", &tokenspeed::PagedCacheGroupTable::ReleaseAll)
        .def("page_ids", &tokenspeed::PagedCacheGroupTable::PageIds, nb::rv_policy::reference_internal)
        .def("size", &tokenspeed::PagedCacheGroupTable::Size)
        .def("active_pages_count", &tokenspeed::PagedCacheGroupTable::ActivePagesCount)
        .def("owned_pages_count", &tokenspeed::PagedCacheGroupTable::OwnedPagesCount)
        .def("borrowed_pages_count", &tokenspeed::PagedCacheGroupTable::BorrowedPagesCount)
        .def("released_pages_count", &tokenspeed::PagedCacheGroupTable::ReleasedPagesCount)
        .def("base_logical_page", &tokenspeed::PagedCacheGroupTable::BaseLogicalPage)
        .def("raw_token_cursor", &tokenspeed::PagedCacheGroupTable::RawTokenCursor)
        .def("rows_per_page", &tokenspeed::PagedCacheGroupTable::RowsPerPage)
        .def("entry_stride_tokens", &tokenspeed::PagedCacheGroupTable::EntryStrideTokens)
        .def("raw_tokens_per_page", &tokenspeed::PagedCacheGroupTable::RawTokensPerPage)
        .def("is_sliding", &tokenspeed::PagedCacheGroupTable::IsSliding)
        .def("sliding_window_tokens", &tokenspeed::PagedCacheGroupTable::SlidingWindowTokens);

    // Python declares the required group ids only. Scheduler derives LCM and
    // sliding-window metadata from the matching PagedCacheGroupConfig entries.
    nb::class_<tokenspeed::PrefixCacheAdjunctSpec>(m, "PrefixCacheAdjunctSpec")
        .def(nb::init<>())
        .def_rw("required_groups", &tokenspeed::PrefixCacheAdjunctSpec::required_groups);

    scheduler_config.def(nb::init<>())
        .def_rw("block_size", &tokenspeed::SchedulerConfig::block_size)
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
        .def_rw("flat_block_pools", &tokenspeed::SchedulerConfig::flat_block_pools)
        .def_rw("paged_cache_groups", &tokenspeed::SchedulerConfig::paged_cache_groups)
        .def_rw("prefix_cache_adjunct", &tokenspeed::SchedulerConfig::prefix_cache_adjunct)
        .def_rw("disable_l2_cache", &tokenspeed::SchedulerConfig::disable_l2_cache)
        .def_rw("enable_l3_storage", &tokenspeed::SchedulerConfig::enable_l3_storage)
        .def_rw("prefetch_threshold", &tokenspeed::SchedulerConfig::prefetch_threshold)
        .def_rw("enable_kv_cache_events", &tokenspeed::SchedulerConfig::enable_kv_cache_events)
        .def_rw("enable_mixed_prefill_decode", &tokenspeed::SchedulerConfig::enable_mixed_prefill_decode)
        .def_prop_ro("uses_explicit_flat_pools", &tokenspeed::SchedulerConfig::UsesExplicitFlatPools)
        .def_rw("disable_prefix_cache", &tokenspeed::SchedulerConfig::disable_prefix_cache)
        .def_rw("enable_mamba", &tokenspeed::SchedulerConfig::enable_mamba)
        .def_rw("mamba_cache_chunk_size", &tokenspeed::SchedulerConfig::mamba_cache_chunk_size)
        .def_rw("mamba_pool_total_chunks", &tokenspeed::SchedulerConfig::mamba_pool_total_chunks)
        .def_rw("enable_mamba_l2", &tokenspeed::SchedulerConfig::enable_mamba_l2)
        .def_rw("mamba_l2_host_slots", &tokenspeed::SchedulerConfig::mamba_l2_host_slots);

    nb::class_<tokenspeed::RequestSpec>(m, "RequestSpec")
        .def(nb::init<>())
        .def_rw("request_id", &tokenspeed::RequestSpec::request_id)
        .def_rw("tokens", &tokenspeed::RequestSpec::tokens)
        .def_rw("rolling_hashes", &tokenspeed::RequestSpec::rolling_hashes)
        .def_rw("storage_hit_pages", &tokenspeed::RequestSpec::storage_hit_pages);

    nb::module_ forward_event = m.def_submodule("ForwardEvent");
    nb::class_<tokenspeed::forward::FlatKVCompletion>(forward_event, "FlatKVCompletion")
        .def(nb::init<>())
        .def_rw("table_generation", &tokenspeed::forward::FlatKVCompletion::table_generation)
        .def_rw("dispatch_seq", &tokenspeed::forward::FlatKVCompletion::dispatch_seq)
        .def_rw("accepted_raw_end", &tokenspeed::forward::FlatKVCompletion::accepted_raw_end);

    nb::class_<tokenspeed::forward::ExtendResult>(forward_event, "ExtendResult")
        .def(nb::init<>())
        .def_rw("request_id", &tokenspeed::forward::ExtendResult::request_id)
        .def_rw("tokens", &tokenspeed::forward::ExtendResult::tokens)
        .def_rw("flat_kv_completion", &tokenspeed::forward::ExtendResult::flat_kv_completion);

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

    nb::class_<tokenspeed::cache::LoadBackDone>(cache, "LoadBackDoneEvent")
        .def(nb::init<>())
        .def_rw("op_id", &tokenspeed::cache::LoadBackDone::op_id)
        .def_rw("success", &tokenspeed::cache::LoadBackDone::success);

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

    nb::class_<tokenspeed::FlatKVCompletionInput>(forward, "FlatKVCompletionInput")
        .def_ro("table_generation", &tokenspeed::FlatKVCompletionInput::table_generation)
        .def_ro("dispatch_seq", &tokenspeed::FlatKVCompletionInput::dispatch_seq)
        .def_ro("dispatch_raw_start", &tokenspeed::FlatKVCompletionInput::dispatch_raw_start)
        .def_ro("dispatch_raw_end", &tokenspeed::FlatKVCompletionInput::dispatch_raw_end)
        .def_ro("protected_raw_end", &tokenspeed::FlatKVCompletionInput::protected_raw_end);

    auto flat_fwd_op = nb::class_<tokenspeed::FlatForwardOperation>(forward, "FlatForwardOp");
    BindForwardCommonFields<tokenspeed::FlatForwardOperation>(flat_fwd_op);
    flat_fwd_op.def_ro("cache_generation", &tokenspeed::FlatForwardOperation::cache_generation)
        .def_ro("input_ids", &tokenspeed::FlatForwardOperation::input_ids)
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
        .def_prop_ro(
            "paged_cache_block_tables",
            [](const tokenspeed::FlatForwardOperation& op)
                -> const std::map<std::string, std::vector<std::vector<std::int32_t>>>& {
                return op.paged_cache_block_tables;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "paged_cache_block_table_base_offsets",
            [](const tokenspeed::FlatForwardOperation& op) -> const std::map<std::string, std::vector<std::int32_t>>& {
                return op.paged_cache_block_table_base_offsets;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro("flat_block_tables",
                     [](const tokenspeed::FlatForwardOperation& op) { return op.MaterializeFlatBlockTables(); })
        .def_prop_ro(
            "flat_block_table_base_offsets",
            [](const tokenspeed::FlatForwardOperation& op) { return op.MaterializeFlatBlockTableBaseOffsets(); })
        .def("flat_block_table_group_ids",
             [](const tokenspeed::FlatForwardOperation& op) {
                 std::vector<std::string> group_ids;
                 group_ids.reserve(op.flat_block_tables.size());
                 for (const auto& [group_id, _] : op.flat_block_tables) {
                     group_ids.push_back(group_id);
                 }
                 return group_ids;
             })
        .def("flat_block_tables_arrays",
             [](nb::handle self) {
                 // Zero-copy 2-D int32 views over the structured contiguous
                 // export owners; `self` pins the op and its execution plan.
                 auto& op = nb::cast<tokenspeed::FlatForwardOperation&>(self);
                 nb::dict out;
                 for (auto& [gid, export_owner] : op.flat_block_tables) {
                     out[nb::str(gid.c_str())] = nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<2>>(
                         export_owner.values.data(), {export_owner.rows, export_owner.cols}, self);
                 }
                 return out;
             })
        .def("copy_flat_block_tables_to", &CopyFlatBlockTablesTo, nb::arg("destination").noconvert(),
             nb::arg("page_id_upper_bounds").noconvert(), nb::arg("copy_metadata").noconvert(),
             nb::arg("payload_offset"))
        .def_ro("flat_kv_completion_inputs", &tokenspeed::FlatForwardOperation::flat_kv_completion_inputs)
        .def("num_extends", &tokenspeed::FlatForwardOperation::num_extends)
        .def_ro("mamba_pool_indices", &tokenspeed::FlatForwardOperation::mamba_working_indices)
        .def_ro("mamba_checkpoint_dst_indices", &tokenspeed::FlatForwardOperation::mamba_checkpoint_dst_indices)
        .def_ro("mamba_track_pool_indices", &tokenspeed::FlatForwardOperation::mamba_checkpoint_dst_indices)
        .def_ro("mamba_cow_src_indices", &tokenspeed::FlatForwardOperation::mamba_cow_src_indices)
        .def_ro("mamba_branching_seqlens", &tokenspeed::FlatForwardOperation::mamba_branching_seqlens);

    // ─── CacheOperation (attached to the Cache submodule) ──────────
    nb::enum_<tokenspeed::CacheKind>(cache, "CacheKind")
        .value("KV", tokenspeed::CacheKind::kKV)
        .value("MAMBA", tokenspeed::CacheKind::kMamba);

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
        .def_ro("dst_pages", &tokenspeed::FlatLoadBackOperation::dst_pages)
        .def_ro("src_pages_by_kind", &tokenspeed::FlatLoadBackOperation::src_pages_by_kind)
        .def_ro("dst_pages_by_kind", &tokenspeed::FlatLoadBackOperation::dst_pages_by_kind);

    nb::class_<tokenspeed::FlatWriteBackOperation>(cache, "WriteBackOp")
        .def_ro("op_ids", &tokenspeed::FlatWriteBackOperation::op_ids)
        .def_ro("src_pages", &tokenspeed::FlatWriteBackOperation::src_pages)
        .def_ro("dst_pages", &tokenspeed::FlatWriteBackOperation::dst_pages)
        .def_ro("src_pages_by_kind", &tokenspeed::FlatWriteBackOperation::src_pages_by_kind)
        .def_ro("dst_pages_by_kind", &tokenspeed::FlatWriteBackOperation::dst_pages_by_kind)
        .def_ro("is_retract", &tokenspeed::FlatWriteBackOperation::is_retract);

    auto collect_forward = [](nb::pointer_and_handle<tokenspeed::ExecutionPlan> self) -> nb::list {
        nb::list result;
        for (const auto& op : self.p->Operations()) {
            if (auto* f = std::get_if<tokenspeed::FlatForwardOperation>(&op)) {
                result.append(nb::cast(f, nb::rv_policy::reference_internal, self.h));
            }
        }
        return result;
    };

    auto collect_cache = [](nb::pointer_and_handle<tokenspeed::ExecutionPlan> self) -> nb::list {
        nb::list result;
        for (const auto& op : self.p->Operations()) {
            if (auto* c = std::get_if<tokenspeed::CacheOperation>(&op)) {
                std::visit(
                    [&result, parent = self.h](const auto& inner) {
                        result.append(nb::cast(&inner, nb::rv_policy::reference_internal, parent));
                    },
                    *c);
            }
        }
        return result;
    };

    nb::class_<tokenspeed::ExecutionPlan>(m, "ExecutionPlan")
        .def(nb::init<>())
        .def_prop_ro("forward", collect_forward)
        .def_prop_ro("cache", collect_cache)
        .def_ro("flat_oom_request_ids", &tokenspeed::ExecutionPlan::flat_oom_request_ids);

    auto scheduler = nb::class_<tokenspeed::Scheduler>(m, "Scheduler");
    scheduler.def(nb::init<tokenspeed::SchedulerConfig>(), nb::arg("config") = tokenspeed::SchedulerConfig{})
        .def("submit_requests",
             nb::overload_cast<const std::vector<tokenspeed::RequestSpec>&>(&tokenspeed::Scheduler::SubmitRequests),
             nb::arg("request_specs"))
        .def("next_execution_plan", [](tokenspeed::Scheduler& s) { return s.NextExecutionPlan(); })
        .def("advance", &tokenspeed::Scheduler::Advance, nb::arg("event"))
        .def(
            "drain_kv_events",
            [](tokenspeed::Scheduler& s) {
                nb::list result;
                for (auto& event : s.DrainKvEvents()) {
                    std::visit([&result](auto& inner) { result.append(nb::cast(inner, nb::rv_policy::copy)); }, event);
                }
                return result;
            },
            nb::rv_policy::move)
        .def("waiting_size", &tokenspeed::Scheduler::WaitingSize)
        .def("decoding_size", &tokenspeed::Scheduler::DecodingSize)
        .def("prefilling_size", &tokenspeed::Scheduler::PrefillSize)
        .def("retract_count", &tokenspeed::Scheduler::RetractedSize)
        .def("available_kv_pages", &tokenspeed::Scheduler::AvailableKvPages)
        .def("active_kv_pages", &tokenspeed::Scheduler::ActiveKvPages)
        .def("get_request_token_size", &tokenspeed::Scheduler::GetRequestTokenSize, nb::arg("id"))
        .def("calc_rolling_hash", &tokenspeed::Scheduler::CalcRollingHash, nb::arg("input_tokens"),
             nb::arg("apply_match") = false)
        .def("paged_cache_group_ids", &tokenspeed::Scheduler::PagedCacheGroupIds)
        .def("paged_cache_group_total_pages", &tokenspeed::Scheduler::PagedCacheGroupTotalPages, nb::arg("group_id"))
        .def("paged_cache_group_available_pages", &tokenspeed::Scheduler::PagedCacheGroupAvailablePages,
             nb::arg("group_id"))
        .def("paged_cache_group_failed_alloc_count", &tokenspeed::Scheduler::PagedCacheGroupFailedAllocCount,
             nb::arg("group_id"))
        .def("get_request_paged_cache_page_ids", &tokenspeed::Scheduler::GetRequestPagedCachePageIds,
             nb::arg("request_id"), nb::arg("group_id"))
        .def("get_request_paged_cache_base_logical_page", &tokenspeed::Scheduler::GetRequestPagedCacheBaseLogicalPage,
             nb::arg("request_id"), nb::arg("group_id"));
#if TOKENSPEED_FLAT_KVCACHE
    scheduler.def("flat_pool_aggregate", &tokenspeed::Scheduler::FlatPoolAggregateStats)
        .def("flat_pool_snapshots", &tokenspeed::Scheduler::FlatPoolSnapshots)
        .def("flat_kv_generation", &tokenspeed::Scheduler::FlatKVGeneration)
        .def("flat_kv_cache_quiescent", &tokenspeed::Scheduler::FlatKVQuiescent)
        .def("reset_flat_kv_cache", &tokenspeed::Scheduler::ResetFlatKVCache);
#endif
}
