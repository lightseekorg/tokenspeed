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

void advance_cache(tokenspeed::Scheduler& scheduler, rust::Slice<const CacheEvent> events) {
    tokenspeed::ExecutionEvent execution_event;
    for (const CacheEvent& event : events) {
        switch (event.kind) {
            case CacheEventKind::WriteBackDone:
                execution_event.With(tokenspeed::cache::WriteBackDone{.op_id = event.op_id});
                break;
            case CacheEventKind::LoadBackDone:
                execution_event.With(tokenspeed::cache::LoadBackDone{.op_id = event.op_id});
                break;
            case CacheEventKind::PrefetchDone:
                execution_event.With(tokenspeed::cache::PrefetchDone(event.op_id, event.landed_pages));
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

rust::Vec<rust::String> prefix_hashes_for_tokens(const tokenspeed::Scheduler& scheduler,
                                                 rust::Slice<const std::int32_t> tokens) {
    const std::vector<std::int32_t> owned(tokens.begin(), tokens.end());
    rust::Vec<rust::String> hashes;
    for (const auto& hash : scheduler.PrefixHashesForTokens(owned)) {
        hashes.push_back(rust::String(hash));
    }
    return hashes;
}

rust::Vec<CacheKeyView> expand_prefix_keys(const tokenspeed::Scheduler& scheduler,
                                           rust::Slice<const rust::String> content_hashes) {
    std::vector<std::string> owned;
    owned.reserve(content_hashes.size());
    for (const auto& hash : content_hashes) owned.emplace_back(hash);
    rust::Vec<CacheKeyView> keys;
    for (const auto& key : scheduler.ExpandPrefixKeys(owned)) {
        keys.push_back(CacheKeyView{
            .group_id = key.group_id,
            .content_hash = rust::String(key.content_hash),
            .page_offset = key.page_offset,
        });
    }
    return keys;
}

void register_storage_keys(tokenspeed::Scheduler& scheduler, rust::Slice<const CacheKeyView> keys) {
    std::vector<tokenspeed::CacheKey> owned;
    owned.reserve(keys.size());
    for (const auto& key : keys) {
        owned.push_back(tokenspeed::CacheKey{
            .group_id = key.group_id,
            .content_hash = std::string(key.content_hash),
            .page_offset = key.page_offset,
        });
    }
    scheduler.RegisterStorageKeys(owned);
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
        const auto* batch = std::get_if<tokenspeed::ForwardBatch>(&operation);
        if (batch == nullptr || !batch->request_ids.empty()) {
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
    // lists its ops, and a snapshot, restore or prefetch batch has one op
    // per request.
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
                    static_assert(std::is_same_v<Batch, tokenspeed::PrefetchBatch>, "a new cache batch kind");
                    append(CacheOpKind::Prefetch, batch.op_ids);
                }
            },
            *cache);
    }
    return out;
}

}  // namespace ts_scheduler
