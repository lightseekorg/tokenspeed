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

#include "ts-scheduler/cxx/facade.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "scheduler/execution_event.h"
#include "ts-scheduler/src/lib.rs.h"

namespace ts_scheduler {

namespace {

template <typename T>
rust::Vec<T> to_rust(const std::vector<T>& values) {
    rust::Vec<T> out;
    out.reserve(values.size());
    for (const T& value : values) {
        out.push_back(value);
    }
    return out;
}

std::vector<std::int32_t> to_std(rust::Slice<const std::int32_t> values) {
    return {values.begin(), values.end()};
}

}  // namespace

void submit_requests(tokenspeed::Scheduler& scheduler, rust::Slice<const RequestSpec> specs) {
    std::vector<tokenspeed::RequestSpec> converted;
    converted.reserve(specs.size());
    for (const RequestSpec& spec : specs) {
        converted.push_back(tokenspeed::RequestSpec{
            .request_id = std::string(spec.request_id),
            .tokens = to_std({spec.tokens.data(), spec.tokens.size()}),
            .max_new_tokens = spec.max_new_tokens,
            .max_cached_prefix_tokens = spec.max_cached_prefix_tokens,
        });
    }
    scheduler.SubmitRequests(converted);
}

std::unique_ptr<tokenspeed::ExecutionPlan> next_execution_plan(tokenspeed::Scheduler& scheduler) {
    return std::make_unique<tokenspeed::ExecutionPlan>(scheduler.NextExecutionPlan());
}

void advance(tokenspeed::Scheduler& scheduler, rust::Slice<const ForwardEvent> events) {
    tokenspeed::ExecutionEvent execution_event;
    for (const ForwardEvent& event : events) {
        std::string request_id(event.request_id);
        switch (event.kind) {
            case ForwardEventKind::ExtendResult:
                execution_event.With(tokenspeed::forward::ExtendResult{
                    .request_id = std::move(request_id),
                    .tokens = to_std({event.tokens.data(), event.tokens.size()}),
                    .spec_candidate_ids =
                        to_std({event.spec_candidate_ids.data(), event.spec_candidate_ids.size()}),
                });
                break;
            case ForwardEventKind::Finish:
                execution_event.With(tokenspeed::forward::Finish{.request_id = std::move(request_id)});
                break;
            case ForwardEventKind::Abort:
                execution_event.With(tokenspeed::forward::Abort{.request_id = std::move(request_id)});
                break;
            case ForwardEventKind::UpdateReserveNumTokens:
                execution_event.With(tokenspeed::forward::UpdateReserveNumTokens{
                    .request_id = std::move(request_id),
                    .reserve_num_tokens_in_next_schedule_event = event.reserve_num_tokens,
                });
                break;
            case ForwardEventKind::PdBootstrapped:
                execution_event.With(tokenspeed::pd::BootstrappedEvent(std::move(request_id)));
                break;
            case ForwardEventKind::PdFailed:
                execution_event.With(tokenspeed::pd::FailedEvent(std::move(request_id)));
                break;
            case ForwardEventKind::PdSucceeded:
                execution_event.With(tokenspeed::pd::SucceededEvent(std::move(request_id)));
                break;
            case ForwardEventKind::PdRemotePrefillDone:
                execution_event.With(
                    tokenspeed::pd::RemotePrefillDoneEvent(std::move(request_id), event.bootstrap_token));
                break;
            default:
                throw std::invalid_argument("unknown forward event kind");
        }
    }
    scheduler.Advance(execution_event);
}

void advance_head(tokenspeed::Scheduler& scheduler, rust::Slice<const StorageTickView> ticks,
                  rust::Slice<const StorageCancellationView> cancellations,
                  rust::Slice<const StorageDoneView> done, rust::Slice<const CacheEvent> cache_events) {
    // One batch, in the order the Python hooks built it: the storage clock,
    // the cancellations, the storage completions, then the Host copies' ACKs.
    tokenspeed::ExecutionEvent execution_event;
    for (const StorageTickView& tick : ticks) {
        execution_event.With(tokenspeed::cache::StorageTick{
            .now_us = tick.now_us,
            .allow_forward = tick.allow_forward,
            .allow_storage = tick.allow_storage,
            .pressure =
                tokenspeed::StoragePressure{
                    .queued_operations = tick.queued_operations,
                    .queued_bytes = tick.queued_bytes,
                    .oldest_operation_us = tick.oldest_operation_us,
                },
        });
    }
    for (const StorageCancellationView& cancellation : cancellations) {
        std::string request_id(cancellation.request_id);
        if (cancellation.abort) {
            execution_event.With(tokenspeed::forward::Abort{.request_id = std::move(request_id)});
        } else {
            execution_event.With(tokenspeed::cache::StorageDetach{.request_id = std::move(request_id)});
        }
    }
    for (const StorageDoneView& completion : done) {
        execution_event.With(tokenspeed::cache::StorageDone{
            .op_id = completion.op_id,
            .success = std::vector<std::uint8_t>(completion.success.begin(), completion.success.end()),
            .retired = completion.retired,
            .object_failures = std::vector<std::uint8_t>(completion.object_failures.begin(),
                                                         completion.object_failures.end()),
        });
    }
    for (const CacheEvent& event : cache_events) {
        switch (event.kind) {
            case CacheEventKind::WriteBackDone:
                execution_event.With(tokenspeed::cache::WriteBackDone{.op_id = event.op_id});
                break;
            case CacheEventKind::LoadBackDone:
                execution_event.With(tokenspeed::cache::LoadBackDone{.op_id = event.op_id});
                break;
            case CacheEventKind::SnapshotDone:
                execution_event.With(tokenspeed::cache::SnapshotDone{.op_id = event.op_id});
                break;
            case CacheEventKind::RestoreDone:
                execution_event.With(tokenspeed::cache::RestoreDone{.op_id = event.op_id});
                break;
            default:
                throw std::invalid_argument("unknown cache event kind");
        }
    }
    scheduler.Advance(execution_event);
}

SchedulerSizes sizes(const tokenspeed::Scheduler& scheduler) {
    return SchedulerSizes{
        .waiting = scheduler.WaitingSize(),
        .prefilling = scheduler.PrefillSize(),
        .decoding = scheduler.DecodingSize(),
        .retracted = scheduler.RetractedSize(),
        .bootstrapping = scheduler.BootstrappingSize(),
        .remote_prefilling = scheduler.RemotePrefillSize(),
        .pd_transfer = scheduler.PdTransferSize(),
        .empty_lcm_blocks = scheduler.EmptyLcmBlocks(),
        .active_lcm_blocks = scheduler.ActiveLcmBlocks(),
    };
}

CacheGroupPages cache_group_pages(const tokenspeed::Scheduler& scheduler, rust::Str group_id) {
    const std::string id(group_id);
    return CacheGroupPages{
        .total = scheduler.CacheGroupTotalPages(id),
        .available = scheduler.CacheGroupAvailablePages(id),
    };
}

rust::Vec<KvEventView> drain_kv_events(tokenspeed::Scheduler& scheduler) {
    rust::Vec<KvEventView> views;
    for (auto& event : scheduler.DrainKvEvents()) {
        KvEventView view{};
        if (auto* stored = std::get_if<tokenspeed::KvBlockStoredEvent>(&event)) {
            view.stored = true;
            for (auto hash : stored->block_hashes) view.block_hashes.push_back(hash);
            view.has_parent = stored->parent_block_hash.has_value();
            view.parent_block_hash = stored->parent_block_hash.value_or(0);
            for (auto token : stored->token_ids) view.token_ids.push_back(token);
            view.block_size = stored->block_size;
        } else {
            const auto& removed = std::get<tokenspeed::KvBlockRemovedEvent>(event);
            view.stored = false;
            for (auto hash : removed.block_hashes) view.block_hashes.push_back(hash);
        }
        views.push_back(std::move(view));
    }
    return views;
}

std::int32_t request_token_size(const tokenspeed::Scheduler& scheduler, rust::Str request_id) {
    return scheduler.RequestTokenSize(std::string(request_id));
}

bool can_abort_before_admission(const tokenspeed::Scheduler& scheduler, rust::Str request_id) {
    return scheduler.CanAbortBeforeAdmission(std::string(request_id));
}

bool storage_pending(const tokenspeed::Scheduler& scheduler) { return scheduler.StoragePending(); }

StorageStatsView storage_stats(const tokenspeed::Scheduler& scheduler) {
    const tokenspeed::StorageStats stats = scheduler.GetStorageStats();
    return StorageStatsView{
        .operations = stats.operations,
        .held_blocks = stats.held_blocks,
        .quarantined = stats.quarantined,
        .negative_keys = stats.negative_keys,
        .backup_candidates = stats.backup_candidates,
        .waiting_requests = stats.waiting_requests,
        .ready_requests = stats.ready_requests,
        .wait_timeouts = stats.wait_timeouts,
        .ready_expirations = stats.ready_expirations,
        .pressure_gate_closed = stats.pressure_gate_closed,
        .pressure_bypasses = stats.pressure_bypasses,
        .pressure_dropped_backup_keys = stats.pressure_dropped_backup_keys,
        .overflow_dropped_backup_keys = stats.overflow_dropped_backup_keys,
        .unavailable_dropped_backup_keys = stats.unavailable_dropped_backup_keys,
        .disabled_dropped_backup_keys = stats.disabled_dropped_backup_keys,
        .cleanup_dropped_backup_keys = stats.cleanup_dropped_backup_keys,
    };
}

bool clear_cache(tokenspeed::Scheduler& scheduler) { return scheduler.ClearCache(); }

bool clear_l1_cache(tokenspeed::Scheduler& scheduler) { return scheduler.ClearL1Cache(); }

bool can_clear_cache(const tokenspeed::Scheduler& scheduler) { return scheduler.CanClearCache(); }

std::int32_t max_single_request_tokens(const tokenspeed::Scheduler& scheduler) {
    return scheduler.MaxSingleRequestTokens();
}

namespace {

ForwardBatchView batch_view(const tokenspeed::ForwardBatch& batch) {
    rust::Vec<rust::String> request_ids;
    request_ids.reserve(batch.request_ids.size());
    for (const std::string& id : batch.request_ids) {
        request_ids.push_back(rust::String(id));
    }
    return ForwardBatchView{
        .request_ids = std::move(request_ids),
        .request_pool_indices = to_rust(batch.request_pool_indices),
        .input_lengths = to_rust(batch.input_lengths),
        .prefill_lengths = to_rust(batch.prefill_lengths),
        .extend_prefix_lens = to_rust(batch.extend_prefix_lens),
        .extend_replay_lens = to_rust(batch.extend_replay_lens),
        .num_extends = batch.NumExtends(),
    };
}

TransferBatchView transfer_view(const tokenspeed::ForwardBatch& batch) {
    TransferBatchView view;
    view.request_ids.reserve(batch.request_ids.size());
    for (const std::string& id : batch.request_ids) {
        view.request_ids.push_back(rust::String(id));
    }
    view.request_pool_indices = to_rust(batch.request_pool_indices);
    view.input_lengths = to_rust(batch.input_lengths);
    view.prefill_lengths = to_rust(batch.prefill_lengths);
    view.extend_prefix_lens = to_rust(batch.extend_prefix_lens);
    view.state_prefix_cached.reserve(batch.state_prefix_cached.size());
    for (const bool cached : batch.state_prefix_cached) {
        view.state_prefix_cached.push_back(cached);
    }
    view.decode_input_ids = to_rust(batch.decode_input_ids);
    view.spec_candidate_offsets.push_back(0);
    for (const std::vector<std::int32_t>& candidates : batch.spec_candidate_ids) {
        for (const std::int32_t id : candidates) {
            view.spec_candidate_ids.push_back(id);
        }
        view.spec_candidate_offsets.push_back(view.spec_candidate_ids.size());
    }
    view.num_extends = batch.NumExtends();
    const std::size_t rows = batch.request_ids.size();
    for (const auto& [group_id, blocks] : batch.block_tables_contig) {
        view.block_tables.push_back(BlockTableView{
            .group_id = rust::String(group_id),
            .columns = rows == 0 ? 0 : blocks.size() / rows,
            .blocks = to_rust(blocks),
        });
    }
    return view;
}

rust::Vec<TransferBatchView> optional_transfer(const std::optional<tokenspeed::ForwardBatch>& batch) {
    rust::Vec<TransferBatchView> out;
    if (batch.has_value()) {
        out.push_back(transfer_view(*batch));
    }
    return out;
}

rust::Vec<ForwardBatchView> optional_batch(const std::optional<tokenspeed::ForwardBatch>& batch) {
    rust::Vec<ForwardBatchView> out;
    if (batch.has_value()) {
        out.push_back(batch_view(*batch));
    }
    return out;
}

}  // namespace

rust::Vec<ForwardBatchView> plan_forward_batches(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<ForwardBatchView> out;
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        const auto* batch = std::get_if<tokenspeed::ForwardBatch>(&operation);
        if (batch != nullptr) {
            out.push_back(batch_view(*batch));
        }
    }
    return out;
}

bool plan_needs_device(const tokenspeed::ExecutionPlan& plan) {
    if (plan.remote_decode.has_value() || plan.remote_prefill.has_value()) {
        return true;
    }
    for (const auto& [group_id, pages] : plan.pages_to_zero) {
        if (!pages.empty()) {
            return true;
        }
    }
    for (const auto& [group_id, pairs] : plan.pages_to_copy) {
        if (!pairs.empty()) {
            return true;
        }
    }
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        if (const auto* batch = std::get_if<tokenspeed::ForwardBatch>(&operation)) {
            if (!batch->request_ids.empty()) {
                return true;
            }
            continue;
        }
        // Storage operations run on the host's L3 executor, not the device.
        const auto* cache = std::get_if<tokenspeed::CacheOperation>(&operation);
        if (cache == nullptr || !std::holds_alternative<tokenspeed::StorageOperation>(*cache)) {
            return true;
        }
    }
    return false;
}

rust::Vec<ForwardBatchView> plan_remote_decode(const tokenspeed::ExecutionPlan& plan) {
    return optional_batch(plan.remote_decode);
}

rust::Vec<ForwardBatchView> plan_remote_prefill(const tokenspeed::ExecutionPlan& plan) {
    return optional_batch(plan.remote_prefill);
}

rust::Vec<TransferBatchView> plan_forward_transfer(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<TransferBatchView> out;
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        if (const auto* batch = std::get_if<tokenspeed::ForwardBatch>(&operation)) {
            out.push_back(transfer_view(*batch));
            break;
        }
    }
    return out;
}

rust::Vec<TransferBatchView> plan_remote_decode_transfer(const tokenspeed::ExecutionPlan& plan) {
    return optional_transfer(plan.remote_decode);
}

rust::Vec<TransferBatchView> plan_remote_prefill_transfer(const tokenspeed::ExecutionPlan& plan) {
    return optional_transfer(plan.remote_prefill);
}

rust::Vec<SchedulerAbortView> plan_aborts(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<SchedulerAbortView> out;
    out.reserve(plan.aborts.size());
    for (const tokenspeed::SchedulerAbort& abort : plan.aborts) {
        out.push_back(SchedulerAbortView{
            .request_id = rust::String(abort.request_id),
            .detail = rust::String(abort.detail),
        });
    }
    return out;
}

rust::Vec<CacheOpView> plan_cache_ops(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<CacheOpView> out;
    // One acknowledgment per op id, whatever the batch kind: an L2 batch
    // lists its ops, and a snapshot or restore batch has one op per request.
    // Storage operations are the L3 executor's, completed through the
    // storage agreement instead.
    auto append = [&out](CacheOpKind kind, const std::vector<std::uint32_t>& op_ids) {
        for (std::uint32_t op_id : op_ids) {
            out.push_back(CacheOpView{.kind = kind, .op_id = op_id});
        }
    };
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        const auto* cache = std::get_if<tokenspeed::CacheOperation>(&operation);
        if (cache == nullptr) {
            continue;
        }
        std::visit(
            [&append](const auto& batch) {
                using Batch = std::decay_t<decltype(batch)>;
                if constexpr (std::is_same_v<Batch, tokenspeed::WriteBackBatch>) {
                    append(CacheOpKind::WriteBack, batch.op_ids);
                } else if constexpr (std::is_same_v<Batch, tokenspeed::LoadBackBatch>) {
                    append(CacheOpKind::LoadBack, batch.op_ids);
                } else if constexpr (std::is_same_v<Batch, tokenspeed::SnapshotStoreBatch>) {
                    append(CacheOpKind::SnapshotStore, batch.op_ids);
                } else if constexpr (std::is_same_v<Batch, tokenspeed::SnapshotRestoreBatch>) {
                    append(CacheOpKind::SnapshotRestore, batch.op_ids);
                } else {
                    static_assert(std::is_same_v<Batch, tokenspeed::StorageOperation>, "a new cache batch kind");
                }
            },
            *cache);
    }
    return out;
}

rust::Vec<StorageOpView> plan_storage_ops(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<StorageOpView> out;
    const auto keys = [](const std::vector<tokenspeed::CacheKey>& cache_keys) {
        rust::Vec<StorageKeyView> views;
        views.reserve(cache_keys.size());
        for (const tokenspeed::CacheKey& key : cache_keys) {
            views.push_back(StorageKeyView{.group_id = key.group_id,
                                           .content_hash = rust::String(key.content_hash),
                                           .page_offset = key.page_offset});
        }
        return views;
    };
    const auto blocks = [](const std::vector<std::int32_t>& host_blocks) {
        rust::Vec<std::int32_t> out;
        out.reserve(host_blocks.size());
        for (const std::int32_t block : host_blocks) {
            out.push_back(block);
        }
        return out;
    };
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        const auto* cache = std::get_if<tokenspeed::CacheOperation>(&operation);
        if (cache == nullptr) {
            continue;
        }
        const auto* storage = std::get_if<tokenspeed::StorageOperation>(cache);
        if (storage == nullptr) {
            continue;
        }
        std::visit(
            [&](const auto& op) {
                using Op = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<Op, tokenspeed::StorageCancelOperation>) {
                    out.push_back(StorageOpView{
                        .kind = StorageOpKind::Cancel, .op_id = op.op_id, .keys = {}, .host_blocks = {}});
                } else if constexpr (std::is_same_v<Op, tokenspeed::StorageQueryOperation>) {
                    out.push_back(StorageOpView{
                        .kind = StorageOpKind::Query, .op_id = op.op_id, .keys = keys(op.keys), .host_blocks = {}});
                } else {
                    const StorageOpKind kind = std::is_same_v<Op, tokenspeed::StorageGetOperation>
                                                   ? StorageOpKind::Get
                                                   : StorageOpKind::Put;
                    out.push_back(StorageOpView{.kind = kind,
                                                .op_id = op.op_id,
                                                .keys = keys(op.keys),
                                                .host_blocks = blocks(op.host_blocks)});
                }
            },
            *storage);
    }
    return out;
}

}  // namespace ts_scheduler
