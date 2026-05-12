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

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <spdlog/spdlog.h>

#include "fsm/cache_states.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/radix_tree/node_range.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "scheduler/operations/cache.h"
#include "scheduler/operations/forward.h"
#include "scheduler/request.h"
#include "scheduler/request_spec.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"
#include "utils.h"

namespace {

std::int32_t CeilDivPositive(std::int32_t numer, std::int32_t denom) {
    if (numer <= 0) return 0;
    return (numer + denom - 1) / denom;
}

}  // namespace

namespace tokenspeed {

std::optional<fsm::SchedulePrefillFirstChunkEvent> Scheduler::schedulePrefillFirstChunk(
    Request* request, std::int32_t remaining, std::int32_t decode_input_tokens, bool disable_l2_cache,
    std::map<std::string, std::int32_t>& simulated_free) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};
    MatchResult match_result = hybrid_prefix_cache_ ? hybrid_prefix_cache_->Match(request->GetFullPagedTokens(true))
                                                    : kv_prefix_cache_.Match(request->GetFullPagedTokens(true));
    std::int32_t loadback_tokens = 0;
    std::int32_t unscheduled = 0;
    std::vector<TreeNode*> loadback_diff;

    const std::int32_t device_matched = match_result.device.DepthInPage();
    const std::int32_t host_matched = match_result.host.DepthInPage();
    if (disable_l2_cache) {
        unscheduled = request->PrefillSize() - device_matched * config_.page_size;
    } else {
        loadback_diff = match_result.NodesWithout<ResourceType::Device>();
        if (host_matched > device_matched) {
            loadback_tokens = config_.page_size * (host_matched - device_matched);
        }
        unscheduled = request->PrefillSize() - std::max(device_matched, host_matched) * config_.page_size;
    }

    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    if (hybrid_prefix_cache_ && match_result.mamba_branching_seqlen == -1) {
        const std::int32_t aligned = hybrid_prefix_cache_->AlignMambaCacheSeqlen(tokens_this_round);
        if (aligned > 0) {
            match_result.mamba_branching_seqlen = aligned;
        }
    }

    std::int32_t num_tokens = loadback_tokens + tokens_this_round + decode_input_tokens;
    std::int32_t device_pages_needed = (num_tokens + config_.page_size - 1) / config_.page_size;

    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);

    // Eviction happens Here: evicts unlocked prefix-cache nodes to free device_pages_needed pages.
    if (!(kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed))) {
        return {};
    }

    if (hybrid_prefix_cache_ && !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(2)) {
        return {};
    }

    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    const std::int32_t target = first_pos + tokens_this_round;
    auto admission = checkPagedCacheGroupAdmission(request->Id(), first_pos, target, simulated_free);
    if (!admission.ok) {
        return {};
    }
    applyPagedCacheGroupAdmissionDebit(simulated_free, admission);

    return fsm::SchedulePrefillFirstChunkEvent{
        tokens_this_round,
        decode_input_tokens,
        &device_allocator_,
        &req_pool_allocator_,
        match_result,
        config_.role,
        &kv_prefix_cache_,
        disable_l2_cache,
        std::move(loadback_diff),
        hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr,
        mamba_allocator_ ? &*mamba_allocator_ : nullptr,
    };
}

std::optional<fsm::SchedulePrefillEvent> Scheduler::schedulePrefill(
    Request* request, std::int32_t remaining, std::int32_t reserve_num_tokens_in_next_schedule_event,
    std::map<std::string, std::int32_t>& simulated_free) {
    std::int32_t unscheduled = request->UnScheduledPrefillSize();
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);

    std::int32_t pages_needed = (tokens_this_round + config_.page_size - 1) / config_.page_size;

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
        return {};
    }

    if (hybrid_prefix_cache_ && !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(1)) {
        return {};
    }

    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    const std::int32_t target = first_pos + tokens_this_round;
    auto admission = checkPagedCacheGroupAdmission(request->Id(), first_pos, target, simulated_free);
    if (!admission.ok) {
        return {};
    }
    applyPagedCacheGroupAdmissionDebit(simulated_free, admission);

    return fsm::SchedulePrefillEvent{tokens_this_round, reserve_num_tokens_in_next_schedule_event,
                                     hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
}

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(Request* request,
                                                                  std::map<std::string, std::int32_t>& simulated_free) {
    std::int32_t tail_available = request->TailPageAvailableTokens();
    std::int32_t extra_tokens = std::max(0, request->GetReserveNumTokensInNextScheduleEvent() - tail_available);
    std::int32_t pages_needed = (extra_tokens + config_.page_size - 1) / config_.page_size;

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
        return {};
    }

    const std::int32_t first_pos = request->TokenSize();
    const std::int32_t target = first_pos + config_.decode_input_tokens;
    auto admission = checkPagedCacheGroupAdmission(request->Id(), first_pos, target, simulated_free);
    if (!admission.ok) {
        return {};
    }
    applyPagedCacheGroupAdmissionDebit(simulated_free, admission);

    return fsm::ScheduleDecodeEvent{config_.decode_input_tokens,
                                    hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
}

std::optional<fsm::ScheduleDecodeFromRetractedEvent> Scheduler::scheduleDecodeFromRetracted(
    Request* request, std::map<std::string, std::int32_t>& simulated_free) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};

    MatchResult match_result = kv_prefix_cache_.Match(request->GetFullPagedTokens(true));
    std::vector<TreeNode*> loadback_diff = match_result.NodesWithout<ResourceType::Device>();

    const std::int32_t device_matched2 = match_result.device.DepthInPage();
    const std::int32_t host_matched2 = match_result.host.DepthInPage();
    // Pages needed: LoadBack nodes (host→device) + pages for decode step itself.
    std::int32_t num_tokens = 0;
    if (host_matched2 > device_matched2) {
        num_tokens += (config_.page_size * (host_matched2 - device_matched2)) + config_.decode_input_tokens;
    } else {
        num_tokens += config_.decode_input_tokens;
    }
    std::int32_t device_pages_needed = (num_tokens + config_.page_size - 1) / config_.page_size;

    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed)) {
        return {};
    }

    std::map<std::string, std::int32_t> released_back;
    auto req_it = request_paged_cache_tables_.find(request->Id());
    if (req_it != request_paged_cache_tables_.end()) {
        for (const auto& [gid, table] : req_it->second) {
            released_back[gid] = table.ActivePagesCount();
        }
    }
    auto simulated_after_release = simulated_free;
    for (const auto& [gid, n] : released_back) {
        simulated_after_release[gid] += n;
    }

    PagedCacheGroupAdmission admission;
    const std::int32_t target = request->TokenSize();
    if (!paged_cache_allocators_.empty() && target >= 0) {
        for (const auto& [gid, allocator] : paged_cache_allocators_) {
            const auto& cfg = allocator->Config();
            if (cfg.entry_stride_tokens <= 0 || cfg.rows_per_page <= 0) continue;
            const std::int32_t entries = CeilDivPositive(target, cfg.entry_stride_tokens);
            const std::int32_t required = (entries + cfg.rows_per_page - 1) / cfg.rows_per_page;
            const std::int32_t free =
                simulated_after_release.count(gid) ? simulated_after_release.at(gid) : allocator->AvailablePages();
            admission.releasable_pages[gid] = 0;
            admission.new_pages_needed[gid] = required;
            if (free < required) {
                admission.ok = false;
            }
        }
    }
    if (!admission.ok) {
        return {};
    }
    for (const auto& [gid, n] : released_back) {
        simulated_free[gid] += n;
    }
    applyPagedCacheGroupAdmissionDebit(simulated_free, admission);

    return fsm::ScheduleDecodeFromRetractedEvent{config_.decode_input_tokens, &device_allocator_,
                                                 &req_pool_allocator_,        &kv_prefix_cache_,
                                                 std::move(match_result),     loadback_diff};
}

std::optional<fsm::ScheduleRetractEvent> Scheduler::scheduleRetract(Request* request) {
    auto full_paged_tokens = request->GetFullPagedTokens(true);
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(request->GetDeviceNode());
    std::int32_t total_available = static_cast<std::int32_t>(request->GetOccupiedPages().size());

    // Overlap scheduling: ExtendResult may grow the token container before the
    // next Acquire runs. Clamp to the pages we actually have.
    if (total_available < static_cast<std::int32_t>(full_paged_tokens.size())) {
        full_paged_tokens.resize(total_available);
    }

    std::int32_t alloc_count =
        static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());

    OwnedPages alloc_pages = request->TakeFirstPages(alloc_count);

    kv_prefix_cache_.Insert<ResourceType::Device>(full_paged_tokens, prefix_pages, std::move(alloc_pages));

    MatchResult match_result = kv_prefix_cache_.Match(full_paged_tokens);

    std::unique_ptr<HostNodeRef> temp_lock = std::make_unique<HostNodeRef>(match_result.host.last_node);
    const std::int32_t device_matched3 = match_result.device.DepthInPage();
    const std::int32_t host_matched3 = match_result.host.DepthInPage();
    std::int32_t host_pages_needed = 0;
    if (device_matched3 > host_matched3) {
        host_pages_needed = device_matched3 - host_matched3;
    }

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Host>(host_pages_needed)) {
        return {};
    }
    return fsm::ScheduleRetractEvent{&kv_prefix_cache_, &host_allocator_, match_result};
}

LoadBackOperation GenerateLoadBackOp(const std::vector<TreeNode*>& diff, cache_op_id op_id) {
    std::vector<std::tuple<std::int32_t, std::int32_t>> pages_to_transfer;

    for (TreeNode* node : diff) {
        const auto& host_pages = node->Host().Pages();
        const auto& device_pages = node->Device().Pages();
        for (std::size_t i = 0; i < host_pages.size(); ++i) {
            pages_to_transfer.emplace_back(host_pages[i], device_pages[i]);
        }
    }
    return LoadBackOperation{op_id, std::move(pages_to_transfer)};
}

std::optional<WriteBackOperation> Scheduler::applyEventAndGenerateOp(Request* request,
                                                                     fsm::ScheduleRetractEvent event) {
    // ScheduleRetractEvent::operator() already builds the (device_page, host_page) pairs
    // inside the state transition (consistent with FinishEvent→Draining path).
    // We just apply the event and read back the pre-computed pairs.
    request->Apply(std::move(event));

    const auto& pages_to_transfer = request->GetPagesToTransfer<fsm::Retracting>();
    if (pages_to_transfer.empty()) {
        // device.matched == host.matched: no device→host copy needed.
        // Fire WriteBackDoneEvent immediately so the request transitions
        // Retracting → Retracted without registering a dangling op_id.
        request->Apply(fsm::WriteBackDoneEvent{});
        return std::nullopt;
    }
    // Register in cache_op_tracker_ so WriteBackDone can route back to the request.
    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
    CacheOpSpec spec;
    spec.request_id = request->Id();
    cache_op_tracker_[op_id] = std::move(spec);
    return WriteBackOperation{
        op_id, std::vector<std::tuple<std::int32_t, std::int32_t>>(pages_to_transfer.begin(), pages_to_transfer.end()),
        true};
}

std::optional<WriteBackOperation> Scheduler::newRetractOperation(Request* retract_request) {
    if (auto event = scheduleRetract(retract_request)) {
        if (auto op = applyEventAndGenerateOp(retract_request, std::move(*event))) {
            return std::move(*op);
        }
    } else {
        spdlog::warn("[Scheduler] Retract failed for request {}: host capacity exhausted, aborting request",
                     retract_request->Id());
        retract_request->Apply(fsm::AbortEvent{});
    }
    return std::nullopt;
}

// Apply event: state transfer + resource allocation
template <typename Event>
    requires(std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent> || std::same_as<Event, fsm::SchedulePrefillEvent>)
static PrefillOperation applyPrefillEvent(Request* request, Event event) {
    std::int32_t begin = static_cast<std::int32_t>(request->GetOccupiedPages().size());
    request->Apply(event);
    std::vector<std::int32_t> all_pages = request->GetOccupiedPages();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size()) - begin;

    auto info = request->GetPrefillInfo();
    auto op = PrefillOperation{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = info.extend_len,
        .occupied_pages = std::move(all_pages),
        .begin = begin,
        .size = sz,
        .prefill_length = request->PrefillSize(),
    }};
    op.input_ids = std::vector<std::int32_t>(info.input_ids.begin(), info.input_ids.end());
    op.shifted_input_ids = std::move(info.shifted_input_ids);
    op.extend_prefix_len = info.already_scheduled_len;

    auto* mamba = request->GetLocalMambaAllocator();
    if (mamba != nullptr && mamba->HasWorking()) {
        op.mamba_working_idx = mamba->WorkingIndex();
        if (mamba->HasCheckpoint()) {
            op.mamba_checkpoint_dst_idx = mamba->CheckpointIndex();
        }
    }

    return op;
}

PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event) {
    auto match = event.GetMatchResult();
    auto op = applyPrefillEvent(request, std::move(event));
    op.mamba_cow_src_idx = match.mamba_cow_src_index;
    op.mamba_branching_seqlen = match.mamba_branching_seqlen;
    acquirePagedCachePagesForRequest(op.request_id, op.extend_prefix_len, op.extend_prefix_len + op.input_length);
    populatePagedCachePagesForOp(op);
    return op;
}

PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event) {
    auto op = applyPrefillEvent(request, std::move(event));
    acquirePagedCachePagesForRequest(op.request_id, op.extend_prefix_len, op.extend_prefix_len + op.input_length);
    populatePagedCachePagesForOp(op);
    return op;
}

template <typename Event>
    requires(std::same_as<Event, fsm::ScheduleDecodeEvent> ||
             std::same_as<Event, fsm::ScheduleDecodeFromRetractedEvent>)
static DecodeOperation applyDecodeEvent(Request* request, Event event, std::int32_t decode_input_tokens) {
    std::int32_t begin = static_cast<std::int32_t>(request->GetOccupiedPages().size());
    request->Apply(std::move(event));
    std::vector<std::int32_t> all_pages = request->GetOccupiedPages();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size()) - begin;

    auto op = DecodeOperation{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = decode_input_tokens,
        .occupied_pages = std::move(all_pages),
        .begin = begin,
        .size = sz,
        .prefill_length = request->PrefillSize(),
    }};

    auto* mamba = request->GetLocalMambaAllocator();
    if (mamba != nullptr && mamba->HasWorking()) {
        op.mamba_working_idx = mamba->WorkingIndex();
        if (mamba->HasCheckpoint()) {
            op.mamba_checkpoint_dst_idx = mamba->CheckpointIndex();
        }
    }

    return op;
}

DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event) {
    const bool need_bootstrap_token = request->Is<fsm::PrefillDone>() && config_.role == Role::kD;
    std::int32_t bootstrap_token = need_bootstrap_token ? request->GetLastToken() : -1;
    const std::int32_t first_pos = request->TokenSize();

    auto op = applyDecodeEvent(request, std::move(event), config_.decode_input_tokens);
    if (need_bootstrap_token) {
        op.decode_input_id = bootstrap_token;
    }
    acquirePagedCachePagesForRequest(op.request_id, first_pos, first_pos + op.input_length);
    populatePagedCachePagesForOp(op);
    return op;
}

DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent event) {
    request->Apply(std::move(event));
    if (!request->Is<fsm::Decoding>()) {
        throw std::logic_error(
            "Scheduler::applyEventAndGenerateOp: expected state=Decoding after loadback recovery; got state=" +
            request->StateName());
    }
    std::vector<std::int32_t> all_pages = request->GetOccupiedPages();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size());
    DecodeOperation op{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = config_.decode_input_tokens,
        .occupied_pages = std::move(all_pages),
        .begin = 0,
        .size = sz,
    }};
    op.decode_input_id = request->GetLastToken();
    op.hist_token_len = request->TokenSize() - 1;

    auto* mamba = request->GetLocalMambaAllocator();
    if (mamba != nullptr && mamba->HasWorking()) {
        op.mamba_working_idx = mamba->WorkingIndex();
        if (mamba->HasCheckpoint()) {
            op.mamba_checkpoint_dst_idx = mamba->CheckpointIndex();
        }
    }

    releasePagedCachePagesForRequest(op.request_id);
    acquirePagedCachePagesForRequest(op.request_id, 0, request->TokenSize());
    populatePagedCachePagesForOp(op);
    return op;
}

std::tuple<std::vector<ForwardOperation>, std::variant<std::vector<LoadBackOperation>, std::vector<WriteBackOperation>>>
Scheduler::newForwardOperation(std::vector<Request*> candidates) {
    auto priority = [&](const Request* req) -> int {
        if (req->Is<fsm::Prefilling>()) return 0;
        if (req->Is<fsm::Submitted>()) return 1;
        if (req->Is<fsm::Decoding>() || req->Is<fsm::PrefillDone>()) return 2;
        if (req->Is<fsm::Retracted>()) return 3;
        return 4;
    };
    std::sort(candidates.begin(), candidates.end(),
              [&](const auto& a, const auto& b) { return priority(a) < priority(b); });

    std::vector<ForwardOperation> ops;
    std::int32_t token_budget = config_.max_scheduled_tokens;
    auto push_op = [&](auto op, bool uses_pool_slot = false) {
        if (config_.role != Role::kD) {
            token_budget -= op.input_length;
        }
        ops.push_back(std::move(op));
    };
    std::vector<LoadBackOperation> loadback_ops;
    auto simulated_free = initialPagedCacheGroupSimulatedFree();
    for (Request* request : candidates) {
        if (token_budget <= 0 || config_.max_batch_size == ops.size()) break;

        if (request->Is<fsm::Prefilling>() && config_.role != Role::kD) {
            std::int32_t reserver_num_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            if (auto ev = schedulePrefill(request, token_budget, reserver_num_tokens, simulated_free)) {
                push_op(applyEventAndGenerateOp(request, *ev));
            }
        } else if (request->Is<fsm::Submitted>() || request->Is<fsm::PrefetchDone>()) {
            // PrefetchDone: host cache populated; treat same as Submitted for forward scheduling.
            std::int32_t decode_input_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;

            if (auto ev = schedulePrefillFirstChunk(request, token_budget, decode_input_tokens,
                                                    config_.disable_l2_cache, simulated_free)) {
                std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                push_op(applyEventAndGenerateOp(request, std::move(*ev)), true);
                // will be empty when disable_l2_cache
                if (!loadback_diff.empty()) {
                    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
                    loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, op_id));
                }
            }
        } else if (request->Is<fsm::PrefillDone>() || (request->Is<fsm::Decoding>() && config_.role != Role::kP)) {
            // Prefill-first: skip ALL decode if any prefill was scheduled this round.
            if (!ops.empty() && std::holds_alternative<PrefillOperation>(ops.back())) break;

            if (auto ev = scheduleDecode(request, simulated_free)) {
                push_op(applyEventAndGenerateOp(request, *ev));
            }
        } else if (request->Is<fsm::Retracted>() && config_.role != Role::kP) {
            if (!ops.empty() && std::holds_alternative<PrefillOperation>(ops.back())) break;

            if (auto ev = scheduleDecodeFromRetracted(request, simulated_free)) {
                std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                push_op(applyEventAndGenerateOp(request, std::move(*ev)));
                if (!loadback_diff.empty()) {
                    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
                    loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, op_id));
                }
            }
        }
    }

    // If all active decode requests failed, device memory is exhausted: retract the longest one.
    if (ops.empty() && !candidates.empty()) {
        std::vector<Request*> retract_candidates;
        for (Request* req : candidates) {
            if ((req->Is<fsm::Decoding>() || (req->Is<fsm::PrefillDone>() && config_.role != Role::kD)) &&
                config_.role != Role::kP) {
                retract_candidates.push_back(req);
            }
        }
        if (!retract_candidates.empty()) {
            Request* victim =
                *std::max_element(retract_candidates.begin(), retract_candidates.end(),
                                  [](const Request* a, const Request* b) { return a->TokenSize() < b->TokenSize(); });
            std::vector<WriteBackOperation> wb_ops;
            if (auto op = newRetractOperation(victim)) {
                wb_ops.push_back(std::move(*op));
            }
            return {std::vector<ForwardOperation>{}, std::move(wb_ops)};
        }
    }

    return {std::move(ops), std::move(loadback_ops)};
}

}  // namespace tokenspeed
