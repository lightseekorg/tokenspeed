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
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "scheduler/outside_events/inc.h"
#include "scheduler/operations/inc.h"
#include "scheduler/capacity_model.h"
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
3. ForwardEvent (ExtendResult, Finish, Abort, UpdateReserveNumTokens)
4. PD::*Event
5. cache::* ACKs (WriteBackDone, LoadBackDone, SnapshotDone, RestoreDone)

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

    nb::enum_<tokenspeed::SchedulerConfig::HostWritePolicy>(scheduler_config, "HostWritePolicy")
        .value("Unspecified", tokenspeed::SchedulerConfig::HostWritePolicy::kUnspecified)
        .value("WriteThrough", tokenspeed::SchedulerConfig::HostWritePolicy::kWriteThrough)
        .value("WriteBack", tokenspeed::SchedulerConfig::HostWritePolicy::kWriteBack);

    nb::enum_<tokenspeed::AttnKind>(m, "AttnKind")
        .value("Full", tokenspeed::AttnKind::kFull)
        .value("SlidingWindow", tokenspeed::AttnKind::kSlidingWindow)
        .value("State", tokenspeed::AttnKind::kState);

    nb::enum_<tokenspeed::CacheTransferPolicy>(m, "CacheTransferPolicy")
        .value("Unspecified", tokenspeed::CacheTransferPolicy::Unspecified)
        .value("FullSuffix", tokenspeed::CacheTransferPolicy::FullSuffix)
        .value("LatestSnapshot", tokenspeed::CacheTransferPolicy::LatestSnapshot);

    nb::class_<tokenspeed::CacheGroupConfig>(m, "CacheGroupConfig")
        .def(nb::init<>())
        .def(
            "__init__",
            [](tokenspeed::CacheGroupConfig* self, std::string group_id, std::int32_t block_granularity,
               std::int32_t total_pages, tokenspeed::AttnKind attn_kind,
               std::optional<std::int32_t> sliding_window_tokens, std::int32_t cache_blocks_per_lcm_block,
               tokenspeed::CacheTransferPolicy transfer_policy, std::int32_t shard_count, bool replayable,
               std::int32_t checkpoint_granularity) {
                new (self) tokenspeed::CacheGroupConfig{
                    std::move(group_id), block_granularity,      total_pages, cache_blocks_per_lcm_block,
                    attn_kind,           sliding_window_tokens,  replayable,  transfer_policy,
                    shard_count,         checkpoint_granularity,
                };
            },
            nb::arg("group_id"), nb::arg("block_granularity"), nb::arg("total_pages"), nb::arg("attn_kind"),
            nb::arg("sliding_window_tokens") = std::nullopt, nb::arg("cache_blocks_per_lcm_block") = 1,
            nb::arg("transfer_policy") = tokenspeed::CacheTransferPolicy::Unspecified, nb::arg("shard_count") = 1,
            nb::arg("replayable") = false, nb::arg("checkpoint_granularity") = 1)
        .def_rw("group_id", &tokenspeed::CacheGroupConfig::group_id)
        .def_rw("block_granularity", &tokenspeed::CacheGroupConfig::block_granularity)
        .def_rw("total_pages", &tokenspeed::CacheGroupConfig::total_pages)
        .def_rw("cache_blocks_per_lcm_block", &tokenspeed::CacheGroupConfig::cache_blocks_per_lcm_block)
        .def_rw("shard_count", &tokenspeed::CacheGroupConfig::shard_count)
        .def_rw("attn_kind", &tokenspeed::CacheGroupConfig::attn_kind)
        .def_rw("sliding_window_tokens", &tokenspeed::CacheGroupConfig::sliding_window_tokens)
        .def_rw("replayable", &tokenspeed::CacheGroupConfig::replayable)
        .def_rw("checkpoint_granularity", &tokenspeed::CacheGroupConfig::checkpoint_granularity)
        .def_rw("transfer_policy", &tokenspeed::CacheGroupConfig::transfer_policy)
        .def("validate", &tokenspeed::CacheGroupConfig::Validate);

    scheduler_config.def(nb::init<>())
        .def_rw("prefix_granularity", &tokenspeed::SchedulerConfig::prefix_granularity)
        .def_rw("max_scheduled_tokens", &tokenspeed::SchedulerConfig::max_scheduled_tokens)
        .def_rw("max_kv_page_num", &tokenspeed::SchedulerConfig::max_kv_page_num)
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
        // The request-private snapshot pool (virtual block count including
        // the null page): 1 means the engine never retracts.
        .def_prop_rw(
            "num_snapshot_pages", [](const tokenspeed::SchedulerConfig& c) { return c.snapshot_allocator.total_pages; },
            [](tokenspeed::SchedulerConfig& c, std::int32_t v) { c.snapshot_allocator.total_pages = v; })
        .def_rw("max_retracted_requests", &tokenspeed::SchedulerConfig::max_retracted_requests)
        .def_rw("debug_force_retraction_interval", &tokenspeed::SchedulerConfig::debug_force_retraction_interval)
        .def_rw("cache_groups", &tokenspeed::SchedulerConfig::cache_groups)
        .def_rw("host_group_packing", &tokenspeed::SchedulerConfig::host_group_packing)
        .def_rw("disable_l2_cache", &tokenspeed::SchedulerConfig::disable_l2_cache)
        .def_rw("host_write_policy", &tokenspeed::SchedulerConfig::host_write_policy)
        .def_rw("enable_l3_storage", &tokenspeed::SchedulerConfig::enable_l3_storage)
        .def_rw("enable_kv_cache_events", &tokenspeed::SchedulerConfig::enable_kv_cache_events)
        .def_rw("enable_mixed_prefill_decode", &tokenspeed::SchedulerConfig::enable_mixed_prefill_decode)
        .def_rw("disable_prefix_cache", &tokenspeed::SchedulerConfig::disable_prefix_cache)
        .def_rw("prefix_replay_tokens", &tokenspeed::SchedulerConfig::prefix_replay_tokens)
        .def_rw("state_checkpoint_in_forward", &tokenspeed::SchedulerConfig::state_checkpoint_in_forward);

    // The config-only sizing model. Python builds it from a SchedulerConfig
    // whose page counts are still zero, sizes the pool from its answers, and
    // the Scheduler later bounds requests against that pool with the same
    // model. Group results are indexed like config.cache_groups.
    nb::class_<tokenspeed::CapacityModel>(m, "CapacityModel")
        .def(nb::init<const tokenspeed::SchedulerConfig&>(), nb::arg("config"))
        .def_prop_ro("num_groups", &tokenspeed::CapacityModel::NumGroups)
        .def("single_request_group_pages", &tokenspeed::CapacityModel::SingleRequestGroupPages, nb::arg("token_limit"))
        .def("concurrent_group_pages", &tokenspeed::CapacityModel::ConcurrentGroupPages, nb::arg("max_total_tokens"),
             nb::arg("max_context_len"))
        .def(
            "lcm_blocks_needed_for",
            [](const tokenspeed::CapacityModel& model, const std::vector<std::int64_t>& group_pages) {
                return model.LcmBlocksNeededFor(group_pages);
            },
            nb::arg("group_pages"))
        .def("max_single_request_tokens", &tokenspeed::CapacityModel::MaxSingleRequestTokens,
             nb::arg("usable_lcm_blocks"));

    nb::class_<tokenspeed::RequestSpec>(m, "RequestSpec")
        .def(nb::init<>())
        .def_rw("request_id", &tokenspeed::RequestSpec::request_id)
        .def_rw("tokens", &tokenspeed::RequestSpec::tokens)
        .def_rw("max_new_tokens", &tokenspeed::RequestSpec::max_new_tokens)
        .def_rw("max_cached_prefix_tokens", &tokenspeed::RequestSpec::max_cached_prefix_tokens);

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

    nb::class_<tokenspeed::StoragePressure>(m, "StoragePressure")
        .def(nb::init<std::int64_t, std::int64_t, std::int64_t>(), nb::arg("queued_operations"),
             nb::arg("queued_bytes"), nb::arg("oldest_operation_us"))
        .def_ro("queued_operations", &tokenspeed::StoragePressure::queued_operations)
        .def_ro("queued_bytes", &tokenspeed::StoragePressure::queued_bytes)
        .def_ro("oldest_operation_us", &tokenspeed::StoragePressure::oldest_operation_us);
    nb::class_<tokenspeed::StorageConfig>(m, "StorageConfig")
        .def(nb::init<std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int64_t,
                      std::int64_t, tokenspeed::StoragePressure, tokenspeed::StoragePressure>(),
             nb::arg("max_operations"), nb::arg("max_keys_per_operation"), nb::arg("max_host_blocks"),
             nb::arg("metadata_entries"), nb::arg("max_prefetch_tokens"), nb::arg("min_prefetch_tokens"),
             nb::arg("wait_timeout_us"), nb::arg("ready_timeout_us"), nb::arg("pressure_high"), nb::arg("pressure_low"))
        .def_ro("max_operations", &tokenspeed::StorageConfig::max_operations)
        .def_ro("max_keys_per_operation", &tokenspeed::StorageConfig::max_keys_per_operation)
        .def_ro("max_host_blocks", &tokenspeed::StorageConfig::max_host_blocks)
        .def_ro("metadata_entries", &tokenspeed::StorageConfig::metadata_entries)
        .def_ro("max_prefetch_tokens", &tokenspeed::StorageConfig::max_prefetch_tokens)
        .def_ro("min_prefetch_tokens", &tokenspeed::StorageConfig::min_prefetch_tokens)
        .def_ro("wait_timeout_us", &tokenspeed::StorageConfig::wait_timeout_us)
        .def_ro("ready_timeout_us", &tokenspeed::StorageConfig::ready_timeout_us)
        .def_ro("pressure_high", &tokenspeed::StorageConfig::pressure_high)
        .def_ro("pressure_low", &tokenspeed::StorageConfig::pressure_low);
    scheduler_config.def_rw("storage_config", &tokenspeed::SchedulerConfig::storage_config);
    nb::enum_<tokenspeed::StorageKind>(m, "StorageKind")
        .value("Query", tokenspeed::StorageKind::kQuery)
        .value("Get", tokenspeed::StorageKind::kGet)
        .value("Put", tokenspeed::StorageKind::kPut)
        .value("Cancel", tokenspeed::StorageKind::kCancel);
    nb::class_<tokenspeed::StorageStats>(m, "StorageStats")
        .def_ro("operations", &tokenspeed::StorageStats::operations)
        .def_ro("held_blocks", &tokenspeed::StorageStats::held_blocks)
        .def_ro("quarantined", &tokenspeed::StorageStats::quarantined)
        .def_ro("negative_keys", &tokenspeed::StorageStats::negative_keys)
        .def_ro("backup_candidates", &tokenspeed::StorageStats::backup_candidates)
        .def_ro("waiting_requests", &tokenspeed::StorageStats::waiting_requests)
        .def_ro("ready_requests", &tokenspeed::StorageStats::ready_requests)
        .def_ro("wait_timeouts", &tokenspeed::StorageStats::wait_timeouts)
        .def_ro("ready_expirations", &tokenspeed::StorageStats::ready_expirations)
        .def_ro("pressure_gate_closed", &tokenspeed::StorageStats::pressure_gate_closed)
        .def_ro("pressure_bypasses", &tokenspeed::StorageStats::pressure_bypasses)
        .def_ro("pressure_dropped_backup_keys", &tokenspeed::StorageStats::pressure_dropped_backup_keys)
        .def_ro("overflow_dropped_backup_keys", &tokenspeed::StorageStats::overflow_dropped_backup_keys)
        .def_ro("unavailable_dropped_backup_keys", &tokenspeed::StorageStats::unavailable_dropped_backup_keys)
        .def_ro("disabled_dropped_backup_keys", &tokenspeed::StorageStats::disabled_dropped_backup_keys)
        .def_ro("cleanup_dropped_backup_keys", &tokenspeed::StorageStats::cleanup_dropped_backup_keys);

    // Host L2 counters (Scheduler.host_cache_stats), read-only. Every by-group
    // list is indexed like SchedulerConfig.cache_groups.
    nb::class_<tokenspeed::CacheCoordinator::HostEvictionCounts>(m, "HostEvictionCounts")
        .def_ro("blocks", &tokenspeed::CacheCoordinator::HostEvictionCounts::blocks)
        .def_ro("device_resident", &tokenspeed::CacheCoordinator::HostEvictionCounts::device_resident);
    nb::class_<tokenspeed::CacheCoordinator::HostEvictionStats>(m, "HostEvictionStats")
        .def_ro("lru", &tokenspeed::CacheCoordinator::HostEvictionStats::lru)
        .def_ro("duplicate", &tokenspeed::CacheCoordinator::HostEvictionStats::duplicate)
        .def_ro("superseded", &tokenspeed::CacheCoordinator::HostEvictionStats::superseded)
        .def_ro("dead_entry", &tokenspeed::CacheCoordinator::HostEvictionStats::dead_entry)
        .def_ro("tail_cap", &tokenspeed::CacheCoordinator::HostEvictionStats::tail_cap)
        .def_ro("flush", &tokenspeed::CacheCoordinator::HostEvictionStats::flush)
        .def_ro("duplicate_kv", &tokenspeed::CacheCoordinator::HostEvictionStats::duplicate_kv);
    nb::class_<tokenspeed::CacheCoordinator::DeviceEvictionCounts>(m, "DeviceEvictionCounts")
        .def_ro("blocks", &tokenspeed::CacheCoordinator::DeviceEvictionCounts::blocks)
        .def_ro("host_resident", &tokenspeed::CacheCoordinator::DeviceEvictionCounts::host_resident);
    nb::class_<tokenspeed::CacheCoordinator::DeviceEvictionStats>(m, "DeviceEvictionStats")
        .def_ro("admission", &tokenspeed::CacheCoordinator::DeviceEvictionStats::admission)
        .def_ro("request_reclaim", &tokenspeed::CacheCoordinator::DeviceEvictionStats::request_reclaim)
        .def_ro("stranded_suffix", &tokenspeed::CacheCoordinator::DeviceEvictionStats::stranded_suffix)
        .def_ro("restore", &tokenspeed::CacheCoordinator::DeviceEvictionStats::restore)
        .def_ro("flush", &tokenspeed::CacheCoordinator::DeviceEvictionStats::flush)
        .def_ro("tail_cap", &tokenspeed::CacheCoordinator::DeviceEvictionStats::tail_cap)
        .def_ro("dead_entry_collection", &tokenspeed::CacheCoordinator::DeviceEvictionStats::dead_entry_collection);
    nb::class_<tokenspeed::TierTransferManager::WriteBackStats>(m, "WriteBackStats")
        .def_ro("ops", &tokenspeed::TierTransferManager::WriteBackStats::ops)
        .def_ro("blocks", &tokenspeed::TierTransferManager::WriteBackStats::blocks)
        .def_ro("host_cached", &tokenspeed::TierTransferManager::WriteBackStats::host_cached)
        .def_ro("in_flight", &tokenspeed::TierTransferManager::WriteBackStats::in_flight)
        .def_ro("evicted_before_drain", &tokenspeed::TierTransferManager::WriteBackStats::evicted_before_drain)
        .def_ro("no_placement", &tokenspeed::TierTransferManager::WriteBackStats::no_placement)
        .def_ro("snapshot_fallback", &tokenspeed::TierTransferManager::WriteBackStats::snapshot_fallback)
        .def_ro("image_refused", &tokenspeed::TierTransferManager::WriteBackStats::image_refused)
        .def_ro("recopied", &tokenspeed::TierTransferManager::WriteBackStats::recopied)
        .def_ro("cancelled", &tokenspeed::TierTransferManager::WriteBackStats::cancelled);
    nb::class_<tokenspeed::TierTransferManager::LoadBackStats>(m, "LoadBackStats")
        .def_ro("ops", &tokenspeed::TierTransferManager::LoadBackStats::ops)
        .def_ro("blocks", &tokenspeed::TierTransferManager::LoadBackStats::blocks);
    nb::class_<tokenspeed::PrefixHitStats>(m, "PrefixHitStats")
        .def_ro("prompt_tokens", &tokenspeed::PrefixHitStats::prompt_tokens)
        .def_ro("device_hit_tokens", &tokenspeed::PrefixHitStats::device_hit_tokens)
        .def_ro("host_hit_tokens", &tokenspeed::PrefixHitStats::host_hit_tokens)
        .def_ro("host_gain_tokens", &tokenspeed::PrefixHitStats::host_gain_tokens)
        .def_ro("unresumable_kv_tokens", &tokenspeed::PrefixHitStats::unresumable_kv_tokens);
    nb::class_<tokenspeed::HostCacheStats>(m, "HostCacheStats")
        .def_ro("prefix_hits", &tokenspeed::HostCacheStats::prefix_hits)
        .def_ro("publication_stores", &tokenspeed::HostCacheStats::publication_stores)
        .def_ro("finish_stores", &tokenspeed::HostCacheStats::finish_stores)
        .def_ro("retraction_stores", &tokenspeed::HostCacheStats::retraction_stores)
        .def_ro("eviction_stores", &tokenspeed::HostCacheStats::eviction_stores)
        .def_ro("prefix_loads", &tokenspeed::HostCacheStats::prefix_loads)
        .def_ro("restore_loads", &tokenspeed::HostCacheStats::restore_loads)
        .def_ro("host_evictions", &tokenspeed::HostCacheStats::host_evictions)
        .def_ro("store_candidates_dropped_by_flush", &tokenspeed::HostCacheStats::store_candidates_dropped_by_flush)
        .def_ro("dead_entries_not_written_back", &tokenspeed::HostCacheStats::dead_entries_not_written_back)
        .def_ro("device_evictions", &tokenspeed::HostCacheStats::device_evictions);

    // ─── ExecutionEvent ─────────────────────────────────────────────

    nb::module_ pd = m.def_submodule("PD");
    nb::module_ cache = m.def_submodule("Cache");

    const auto storage_keys = [](const std::vector<tokenspeed::CacheKey>& values) {
        nb::list keys;
        for (const auto& key : values) keys.append(nb::make_tuple(key.group_id, key.content_hash, key.page_offset));
        return keys;
    };

    nb::class_<tokenspeed::StorageQueryOperation>(cache, "StorageQueryOp")
        .def_ro("op_id", &tokenspeed::StorageQueryOperation::op_id)
        .def_prop_ro("kind",
                     [](const tokenspeed::StorageQueryOperation&) { return tokenspeed::StorageQueryOperation::kKind; })
        .def_prop_ro("keys",
                     [storage_keys](const tokenspeed::StorageQueryOperation& op) { return storage_keys(op.keys); });

    nb::class_<tokenspeed::StorageGetOperation>(cache, "StorageGetOp")
        .def_ro("op_id", &tokenspeed::StorageGetOperation::op_id)
        .def_prop_ro("kind",
                     [](const tokenspeed::StorageGetOperation&) { return tokenspeed::StorageGetOperation::kKind; })
        .def_prop_ro("keys",
                     [storage_keys](const tokenspeed::StorageGetOperation& op) { return storage_keys(op.keys); })
        .def_ro("host_blocks", &tokenspeed::StorageGetOperation::host_blocks);

    nb::class_<tokenspeed::StoragePutOperation>(cache, "StoragePutOp")
        .def_ro("op_id", &tokenspeed::StoragePutOperation::op_id)
        .def_prop_ro("kind",
                     [](const tokenspeed::StoragePutOperation&) { return tokenspeed::StoragePutOperation::kKind; })
        .def_prop_ro("keys",
                     [storage_keys](const tokenspeed::StoragePutOperation& op) { return storage_keys(op.keys); })
        .def_ro("host_blocks", &tokenspeed::StoragePutOperation::host_blocks);

    nb::class_<tokenspeed::StorageCancelOperation>(cache, "StorageCancelOp")
        .def_ro("op_id", &tokenspeed::StorageCancelOperation::op_id)
        .def_prop_ro("kind", [](const tokenspeed::StorageCancelOperation&) {
            return tokenspeed::StorageCancelOperation::kKind;
        });

    nb::class_<tokenspeed::cache::StorageTick>(cache, "StorageTick")
        .def(nb::init<std::int64_t, bool, bool, tokenspeed::StoragePressure>(), nb::arg("now_us"),
             nb::arg("allow_forward"), nb::arg("allow_storage"), nb::arg("pressure"))
        .def_ro("pressure", &tokenspeed::cache::StorageTick::pressure);
    nb::class_<tokenspeed::cache::StorageDetach>(cache, "StorageDetach")
        .def(nb::init<std::string>(), nb::arg("request_id"))
        .def_ro("request_id", &tokenspeed::cache::StorageDetach::request_id);
    nb::class_<tokenspeed::cache::StorageDone>(cache, "StorageDone")
        .def(nb::init<std::uint64_t, std::vector<std::uint8_t>, bool, std::vector<std::uint8_t>>(), nb::arg("op_id"),
             nb::arg("success"), nb::arg("retired"), nb::arg("object_failures"))
        .def_ro("op_id", &tokenspeed::cache::StorageDone::op_id)
        .def_ro("success", &tokenspeed::cache::StorageDone::success)
        .def_ro("retired", &tokenspeed::cache::StorageDone::retired)
        .def_ro("object_failures", &tokenspeed::cache::StorageDone::object_failures);

    nb::class_<tokenspeed::cache::WriteBackDone>(cache, "WriteBackDoneEvent")
        .def(nb::init<>())
        .def_rw("op_id", &tokenspeed::cache::WriteBackDone::op_id);

    // A prefix load landed. Every source is a Host-warm entry, so there is no
    // failed load-back: an L3 object that is gone is discovered by the
    // pre-admission storage transaction, never by a load.
    nb::class_<tokenspeed::cache::LoadBackDone>(cache, "LoadBackDoneEvent")
        .def(nb::init<>())
        .def(nb::init<std::uint32_t>(), nb::arg("op_id"))
        .def_rw("op_id", &tokenspeed::cache::LoadBackDone::op_id);

    // The tail pages and slot-state blob of a retraction image landed.
    nb::class_<tokenspeed::cache::SnapshotDone>(cache, "SnapshotDoneEvent")
        .def(nb::init<>())
        .def_rw("op_id", &tokenspeed::cache::SnapshotDone::op_id);

    nb::class_<tokenspeed::cache::RestoreDone>(cache, "RestoreDoneEvent")
        .def(nb::init<>())
        .def_rw("op_id", &tokenspeed::cache::RestoreDone::op_id);

    // RestoreOp.source_tiers carries these as plain integers per row.
    nb::enum_<tokenspeed::HostTier>(cache, "HostTier", nb::is_arithmetic())
        .value("L2", tokenspeed::HostTier::kL2)
        .value("SnapshotPool", tokenspeed::HostTier::kSnapshotPool);

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
        .def_ro("extend_replay_lens", &tokenspeed::ForwardBatch::extend_replay_lens)
        // Per extend row (the D role's remote admission only): every State
        // group's prompt-end checkpoint sits in slot 0, so the peer omits the
        // latest-snapshot transfer.
        .def_ro("state_prefix_cached", &tokenspeed::ForwardBatch::state_prefix_cached)
        // Per extend row: tokens of the chunk before its in-forward State
        // checkpoint (0 for none); the State groups' rows then carry the
        // checkpoint's block as a third entry.
        .def_ro("extend_state_checkpoint_lens", &tokenspeed::ForwardBatch::extend_state_checkpoint_lens)
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
        .def_ro("page_offsets", &tokenspeed::LoadBackBatch::page_offsets);

    nb::class_<tokenspeed::WriteBackBatch>(cache, "WriteBackOp")
        .def_ro("op_ids", &tokenspeed::WriteBackBatch::op_ids)
        .def_ro("group_ids", &tokenspeed::WriteBackBatch::group_ids)
        .def_ro("src_pages", &tokenspeed::WriteBackBatch::src_pages)
        .def_ro("dst_pages", &tokenspeed::WriteBackBatch::dst_pages)
        .def_ro("content_hashes", &tokenspeed::WriteBackBatch::content_hashes)
        .def_ro("page_offsets", &tokenspeed::WriteBackBatch::page_offsets)
        .def_ro("source_pinned", &tokenspeed::WriteBackBatch::source_pinned);

    // A retraction image's tail leg: DEVICE -> snapshot pool rows (possibly
    // none) plus the slot-state blob of request_pool_indices[i], exported
    // into snapshot_slots[i]. Stream-ordered like an unpinned write-back.
    nb::class_<tokenspeed::SnapshotStoreBatch>(cache, "SnapshotOp")
        .def_ro("op_ids", &tokenspeed::SnapshotStoreBatch::op_ids)
        .def_ro("request_ids", &tokenspeed::SnapshotStoreBatch::request_ids)
        .def_ro("request_pool_indices", &tokenspeed::SnapshotStoreBatch::request_pool_indices)
        .def_ro("snapshot_slots", &tokenspeed::SnapshotStoreBatch::snapshot_slots)
        .def_ro("group_ids", &tokenspeed::SnapshotStoreBatch::group_ids)
        .def_ro("src_pages", &tokenspeed::SnapshotStoreBatch::src_pages)
        .def_ro("dst_pages", &tokenspeed::SnapshotStoreBatch::dst_pages);

    // The way back: Host -> DEVICE rows, each naming its source tier
    // (HostTier as an integer: 0 = L2, 1 = snapshot pool; L2 rows carry their
    // key), plus the blob of snapshot_slots[i] imported into
    // request_pool_indices[i]. Runs after the plan's zeroing.
    nb::class_<tokenspeed::SnapshotRestoreBatch>(cache, "RestoreOp")
        .def_ro("op_ids", &tokenspeed::SnapshotRestoreBatch::op_ids)
        .def_ro("request_ids", &tokenspeed::SnapshotRestoreBatch::request_ids)
        .def_ro("request_pool_indices", &tokenspeed::SnapshotRestoreBatch::request_pool_indices)
        .def_ro("snapshot_slots", &tokenspeed::SnapshotRestoreBatch::snapshot_slots)
        .def_ro("group_ids", &tokenspeed::SnapshotRestoreBatch::group_ids)
        .def_ro("src_pages", &tokenspeed::SnapshotRestoreBatch::src_pages)
        .def_ro("dst_pages", &tokenspeed::SnapshotRestoreBatch::dst_pages)
        .def_ro("content_hashes", &tokenspeed::SnapshotRestoreBatch::content_hashes)
        .def_ro("page_offsets", &tokenspeed::SnapshotRestoreBatch::page_offsets)
        .def_ro("source_tiers", &tokenspeed::SnapshotRestoreBatch::source_tiers);

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

    // A request the scheduler finished on its own while building the plan:
    // the last resort of a capacity retraction whose victim's KV image did not
    // fit the host budgets. Already Finished on every rank; the runtime fails
    // it toward the client with `detail` as the message.
    nb::enum_<tokenspeed::AbortReason>(m, "AbortReason", nb::is_arithmetic())
        .value("ImageDoesNotFit", tokenspeed::AbortReason::kImageDoesNotFit);
    nb::class_<tokenspeed::SchedulerAbort>(m, "SchedulerAbort")
        .def_ro("request_id", &tokenspeed::SchedulerAbort::request_id)
        .def_ro("reason", &tokenspeed::SchedulerAbort::reason)
        .def_ro("detail", &tokenspeed::SchedulerAbort::detail);

    nb::class_<tokenspeed::ExecutionPlan>(m, "ExecutionPlan")
        .def(nb::init<>())
        .def_prop_ro("forward", collect_forward)
        .def_prop_ro("cache", collect_cache)
        .def_ro("aborts", &tokenspeed::ExecutionPlan::aborts)
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
        .def_ro("pages_to_zero", &tokenspeed::ExecutionPlan::pages_to_zero)
        .def("pages_to_zero_arrays",
             [](nb::handle self) {
                 // Zero-copy 1-D int32 views of pages_to_zero, one per cache group
                 // (empty groups included); `self` keeps the ExecutionPlan alive
                 // for the lifetime of each ndarray. The list export builds one
                 // Python int per page, which for a long prompt's admission is
                 // thousands of objects on the forward thread's critical path.
                 auto& plan = nb::cast<tokenspeed::ExecutionPlan&>(self);
                 nb::dict out;
                 for (auto& [gid, pages] : plan.pages_to_zero) {
                     out[nb::str(gid.c_str())] =
                         nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>>(pages.data(), {pages.size()}, self);
                 }
                 return out;
             })
        // Device -> Device whole-block copies per cache group, (source,
        // destination) virtual block ids. The runtime runs them on the forward
        // thread's default stream after the execution stream's in-flight
        // forwards and before this plan's zeroing; destinations are never in
        // pages_to_zero. No ACK.
        .def_ro("pages_to_copy", &tokenspeed::ExecutionPlan::pages_to_copy)
        .def("pages_to_copy_arrays", [](nb::handle self) {
            // Zero-copy (n, 2) int32 views of pages_to_copy, one per cache
            // group (empty groups included): column 0 the source, column 1
            // the destination. std::pair<int32, int32> is two contiguous
            // int32s, which the static_assert below holds the layout to.
            static_assert(sizeof(std::pair<std::int32_t, std::int32_t>) == 2 * sizeof(std::int32_t),
                          "a copy pair must be two contiguous int32 values");
            auto& plan = nb::cast<tokenspeed::ExecutionPlan&>(self);
            nb::dict out;
            for (auto& [gid, pairs] : plan.pages_to_copy) {
                const auto* data = pairs.empty() ? nullptr : reinterpret_cast<const std::int32_t*>(pairs.data());
                out[nb::str(gid.c_str())] =
                    nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<2>>(data, {pairs.size(), 2}, self);
            }
            return out;
        });

    nb::class_<tokenspeed::Scheduler>(m, "Scheduler")
        .def(nb::init<tokenspeed::SchedulerConfig>(), nb::arg("config"))
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
        .def("bootstrapping_size", &tokenspeed::Scheduler::BootstrappingSize,
             "Count requests waiting for their PD bootstrap handshake.")
        .def("waiting_size", &tokenspeed::Scheduler::WaitingSize,
             "Count Submitted, Prefetching, Retracted and Restoring requests awaiting admission or their restore.")
        .def("retracted_size", &tokenspeed::Scheduler::RetractedSize,
             "Count requests suspended with an image (Retracted or Restoring); a weight update must not flush "
             "while this is non-zero.")
        .def("decoding_size", &tokenspeed::Scheduler::DecodingSize, "Count requests in the Decoding FSM state.")
        .def("prefilling_size", &tokenspeed::Scheduler::PrefillSize,
             "Count local/remote prefills, PrefillAwaitingResult, and PrefillDone requests.")
        .def("remote_prefilling_size", &tokenspeed::Scheduler::RemotePrefillSize,
             "Count RemotePrefilling requests; these are also included in prefilling_size().")
        .def("pd_transfer_size", &tokenspeed::Scheduler::PdTransferSize,
             "Count requests with PD-pinned pages; this resource count overlaps lifecycle states.")
        .def("available_lcm_blocks", &tokenspeed::Scheduler::AvailableLcmBlocks)
        .def("empty_lcm_blocks", &tokenspeed::Scheduler::EmptyLcmBlocks)
        .def("active_lcm_blocks", &tokenspeed::Scheduler::ActiveLcmBlocks)
        .def("snapshot_pool_free_blocks", &tokenspeed::Scheduler::SnapshotPoolFreeBlocks,
             "Empty parents of the request-private snapshot pool (a leak check: every block belongs to some "
             "suspended request's image).")
        .def("host_pool_pinned_blocks", &tokenspeed::Scheduler::HostPoolPinnedBlocks,
             "Host L2 entries pinned by a request -- a suspended request's image, or an in-flight transfer.")
        .def("tail_entries", &tokenspeed::Scheduler::TailEntries,
             "Published partial-page boundaries (hash@T) the prefix probe can resume at.")
        .def("tail_cap_evictions", &tokenspeed::Scheduler::TailCapEvictions,
             "Tails dropped because more than eight were published under one prior page.")
        .def("can_abort_before_admission", &tokenspeed::Scheduler::CanAbortBeforeAdmission, nb::arg("id"))
        .def("request_token_size", &tokenspeed::Scheduler::RequestTokenSize, nb::arg("id"))
        .def("max_single_request_tokens", &tokenspeed::Scheduler::MaxSingleRequestTokens)
        .def("clear_l1_cache", &tokenspeed::Scheduler::ClearL1Cache)
        .def("clear_cache", &tokenspeed::Scheduler::ClearCache)
        .def("can_clear_cache", &tokenspeed::Scheduler::CanClearCache)
        .def("host_pool_cached_blocks", &tokenspeed::Scheduler::HostPoolCachedBlocks)
        .def("host_pool_free_blocks", &tokenspeed::Scheduler::HostPoolFreeBlocks,
             "Empty Host L2 pool parents; a parent with any occupied slot counts as used.")
        .def("host_pool_total_blocks", &tokenspeed::Scheduler::HostPoolTotalBlocks,
             "Host L2 pool parents (0 without a Host pool).")
        .def("host_cache_stats", &tokenspeed::Scheduler::GetHostCacheStats,
             "Host L2 counters and the Device evictions they are read against, cumulative since start. "
             "Accounting only, for tests and inspection: no scheduling decision reads them, the runtime "
             "exports none, and reading them changes nothing.")
        .def("storage_pending", &tokenspeed::Scheduler::StoragePending)
        .def("storage_stats", &tokenspeed::Scheduler::GetStorageStats)
        .def("storage_held_blocks", &tokenspeed::Scheduler::StorageHeldBlocks)
        .def("storage_quarantined", &tokenspeed::Scheduler::StorageQuarantined)
        .def("storage_negative_keys", &tokenspeed::Scheduler::StorageNegativeKeys)
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
            nb::arg("content_hashes"));
}
