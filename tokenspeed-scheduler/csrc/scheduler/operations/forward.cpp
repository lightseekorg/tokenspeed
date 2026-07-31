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

#include "scheduler/scheduler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "cache/forward_cache_ops.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "scheduler/operations/cache.h"
#include "scheduler/operations/forward.h"
#include "scheduler/page_hasher.h"
#include "scheduler/request.h"
#include "utils.h"

namespace tokenspeed {

namespace {

template <typename Operation>
void fillBlockTables(Operation& operation, Request& request, const KvCacheCoordinator& coordinator,
                     std::span<const std::string> group_ids) {
    operation.block_tables = BuildBlockTables(coordinator, request.BlockTablesRef(), group_ids);
}

std::vector<GroupDemand> makeGroupDemands(std::vector<BlockTable>& tables, GroupDemand prototype) {
    std::vector<GroupDemand> demands;
    demands.reserve(tables.size());
    for (BlockTable& table : tables) {
        prototype.table = &table;
        demands.push_back(prototype);
    }
    return demands;
}

void appendCompletedPageHashes(std::vector<std::string>& page_hashes,
                               const std::vector<std::span<const std::int32_t>>& paged_tokens,
                               std::int32_t filled_pages) {
    const std::int32_t first_new_page = static_cast<std::int32_t>(page_hashes.size());
    _assert(filled_pages > first_new_page, "caller must pre-check page-hash progress");
    const std::string previous_hash = page_hashes.empty() ? std::string{} : page_hashes.back();
    std::vector<std::string> new_hashes =
        AdvancePagedHashes(paged_tokens, first_new_page, previous_hash, filled_pages);
    page_hashes.insert(page_hashes.end(), std::make_move_iterator(new_hashes.begin()),
                       std::make_move_iterator(new_hashes.end()));
}

bool canConsumeReservedTokensInPlace(const KvCacheCoordinator& coordinator, std::span<const BlockTable> tables,
                                     std::int32_t num_tokens, std::int32_t num_computed_tokens) {
    for (std::int32_t i = 0; i < coordinator.NumGroups(); ++i) {
        const KvCacheManager& manager = coordinator.GroupManager(i);
        const BlockTable& table = tables[static_cast<std::size_t>(i)];
        if (manager.BlocksNeededFor(table, num_tokens) != 0 ||
            !manager.ReclaimableBlockLocationsAt(table, num_computed_tokens).empty()) {
            return false;
        }
    }
    return true;
}

CacheBoundaryKind consumeCompletedBoundaryKind(fsm::CacheProgress& cache_progress,
                                               std::int32_t num_computed_tokens,
                                               std::int32_t prefill_size) {
    if (cache_progress.promotion_boundary_tokens > 0 &&
        num_computed_tokens >= cache_progress.promotion_boundary_tokens) {
        const bool reached_exactly =
            num_computed_tokens == cache_progress.promotion_boundary_tokens;
        cache_progress.promotion_boundary_tokens = 0;
        if (reached_exactly) {
            return CacheBoundaryKind::kPromoted;
        }
    }
    return num_computed_tokens == prefill_size ? CacheBoundaryKind::kEndpoint : CacheBoundaryKind::kChunk;
}

template <typename Event>
    requires(std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent> ||
             std::same_as<Event, fsm::SchedulePrefillEvent>)
PrefillOperation applyPrefillEvent(Request& request, Event& event, const KvCacheCoordinator& coordinator,
                                   std::span<const std::string> group_ids) {
    request.Apply(event);
    const PrefillInfo info = request.CurrentPrefillInfo();

    PrefillOperation operation{{
        .request_id = request.Id(),
        .request_pool_index = request.RequestPoolIndex(),
        .input_length = info.extend_len,
        .prefill_length = request.PrefillSize(),
    }};
    operation.input_ids.assign(info.input_ids.begin(), info.input_ids.end());
    operation.shifted_input_ids = info.shifted_input_ids;
    operation.extend_prefix_len = info.already_scheduled_len;
    fillBlockTables(operation, request, coordinator, group_ids);
    return operation;
}

DecodeOperation applyDecodeEvent(Request& request, fsm::ScheduleDecodeEvent event,
                                 std::int32_t decode_input_tokens, const KvCacheCoordinator& coordinator,
                                 std::span<const std::string> group_ids) {
    request.Apply(std::move(event));

    DecodeOperation operation{{
        .request_id = request.Id(),
        .request_pool_index = request.RequestPoolIndex(),
        .input_length = decode_input_tokens,
        .prefill_length = request.PrefillSize(),
    }};
    fillBlockTables(operation, request, coordinator, group_ids);
    return operation;
}

}  // namespace

Scheduler::AdmissionMatch Scheduler::matchPrefixAtAdmission(Request* request) {
    const auto probe = [this](std::span<const std::string> hashes) {
        if (config_.enable_pd_cache && config_.role == Role::kD) {
            return coordinator_.ProbeDecodeDestinationPrefix(hashes);
        }
        return coordinator_.ProbePrefix(hashes);
    };
    if (config_.disable_prefix_cache) {
        AdmissionMatch match;
        match.probe = probe({});
        return match;
    }

    const std::int32_t cache_block_tokens = coordinator_.CacheBlockTokens();
    const std::int32_t cacheable_pages = std::max((request->PrefillSize() - 1) / cache_block_tokens, 0);
    std::vector<std::span<const std::int32_t>> paged_tokens = request->FullPagedTokens(false);
    paged_tokens.resize(std::min(paged_tokens.size(), static_cast<std::size_t>(cacheable_pages)));
    std::vector<std::string> hashes = ComputePagedHashes(paged_tokens, "");

    AdmissionMatch match;
    match.probe = probe(hashes);
    const std::int32_t hit_pages =
        std::max(match.probe.device.num_common_tokens, match.probe.host.num_common_tokens) / cache_block_tokens;
    match.page_hashes.assign(hashes.begin(), hashes.begin() + hit_pages);

    const std::int32_t extension_pages =
        std::max(match.probe.host.num_common_tokens - match.probe.device.num_common_tokens, 0) / cache_block_tokens;
    const auto extension_begin = hashes.begin() + match.probe.device.num_common_tokens / cache_block_tokens;
    match.extension_hashes.assign(extension_begin, extension_begin + extension_pages);
    return match;
}

std::optional<KvCacheCoordinator::AdmissionResult> Scheduler::admit(
    KvCacheCoordinator::PrefixProbe&& prefix, std::span<const GroupDemand> demands,
    std::optional<std::uint64_t> request_access_epoch) {
    std::optional<KvCacheCoordinator::AdmissionResult> result =
        coordinator_.Admit(std::move(prefix), demands, request_access_epoch);
    if (!result) {
        lcm_admission_failed_ = true;
        return std::nullopt;
    }

    _assert(result->new_page_ids.size() == cache_group_ids_.size(),
            "admission fresh-page groups must match scheduler config");
    for (std::size_t i = 0; i < result->new_page_ids.size(); ++i) {
        auto& page_ids = result->new_page_ids[i];
        auto& pending = new_page_ids_[cache_group_ids_[i]];
        pending.insert(pending.end(), page_ids.begin(), page_ids.end());
    }
    return result;
}

std::optional<KvCacheCoordinator::AdmissionResult> Scheduler::admit(
    std::span<const GroupDemand> demands, std::uint64_t request_access_epoch) {
    return admit(coordinator_.ProbePrefix({}), demands, request_access_epoch);
}

std::optional<fsm::SchedulePrefillFirstChunkEvent> Scheduler::schedulePrefillFirstChunk(
    Request* request, std::int32_t remaining, std::int32_t decode_input_tokens) {
    if (req_pool_allocator_.AvailableSlots() == 0) {
        return std::nullopt;
    }

    AdmissionMatch match = matchPrefixAtAdmission(request);
    const std::int32_t hit_tokens =
        std::max(match.probe.device.num_common_tokens, match.probe.host.num_common_tokens);
    const std::int32_t promotion_boundary_tokens =
        !coordinator_.HasHostTier() && match.probe.device.prefix_closed_tokens > match.probe.device.num_common_tokens
            ? match.probe.device.prefix_closed_tokens
            : 0;
    _assert(promotion_boundary_tokens == 0 ||
                (promotion_boundary_tokens % coordinator_.CacheBlockTokens() == 0 &&
                 promotion_boundary_tokens > hit_tokens &&
                 promotion_boundary_tokens < request->PrefillSize()),
            "promotion boundary must be page-aligned and inside the unmatched prompt");

    const std::int32_t unscheduled = request->PrefillSize() - hit_tokens;
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    if (coordinator_.HasMambaStateGroup() || promotion_boundary_tokens > 0) {
        tokens_this_round = AlignPrefillChunk(hit_tokens, unscheduled, remaining,
                                                  coordinator_.CacheBlockTokens(), promotion_boundary_tokens);
        if (tokens_this_round == 0) {
            return std::nullopt;
        }
    }

    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t decode_reserve = completes_prefill ? decode_input_tokens : 0;
    std::vector<BlockTable> tables(static_cast<std::size_t>(coordinator_.NumGroups()));
    std::vector<GroupDemand> demands = makeGroupDemands(
        tables, GroupDemand{.num_tokens = tokens_this_round, .reserve_tokens = decode_reserve});

    if (config_.enable_pd_cache && config_.role == Role::kD) {
        const std::int32_t final_prompt_block =
            (request->PrefillSize() - 1) / coordinator_.CacheBlockTokens();
        for (std::size_t i = 0; i < demands.size(); ++i) {
            if (config_.paged_cache_groups[i].transfer_policy == PagedCacheTransferPolicy::LatestSnapshot) {
                demands[i].num_tokens = request->PrefillSize();
                demands[i].materialized_suffix_start = final_prompt_block;
            }
        }
    }

    std::optional<KvCacheCoordinator::AdmissionResult> admission =
        admit(std::move(match.probe), demands);
    if (!admission) {
        return std::nullopt;
    }
    _assert(admission->promotion_boundary_tokens == promotion_boundary_tokens,
            "promotion boundary changed between probe and admission");

    if (!match.extension_hashes.empty()) {
        coordinator_.CacheFullBlocks(tables, match.extension_hashes, admission->access_epoch,
                                     admission->device_prefix_tokens / coordinator_.CacheBlockTokens());
    }
    return fsm::SchedulePrefillFirstChunkEvent{
        tokens_this_round,
        decode_reserve,
        &req_pool_allocator_,
        config_.role,
        &coordinator_,
        std::move(tables),
        hit_tokens,
        fsm::CacheProgress{
            .page_hashes = std::move(match.page_hashes),
            .access_epoch = admission->access_epoch,
            .promotion_boundary_tokens = admission->promotion_boundary_tokens,
        },
        std::move(admission->load_pairs),
    };
}

std::optional<fsm::SchedulePrefillEvent> Scheduler::schedulePrefill(
    Request* request, std::int32_t remaining, std::int32_t reserve_num_tokens_in_next_schedule_event) {
    const std::int32_t unscheduled = request->UnscheduledPrefillSize();
    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    fsm::CacheProgress cache_progress = request->CacheProgress();
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    if (coordinator_.HasMambaStateGroup() || cache_progress.promotion_boundary_tokens > 0) {
        tokens_this_round = AlignPrefillChunk(first_pos, unscheduled, remaining,
                                                  coordinator_.CacheBlockTokens(),
                                                  cache_progress.promotion_boundary_tokens);
        if (tokens_this_round == 0) {
            return std::nullopt;
        }
    }

    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t decode_reserve =
        completes_prefill ? reserve_num_tokens_in_next_schedule_event : 0;
    const PrefillInfo previous = request->CurrentPrefillInfo();
    const std::int32_t num_computed_tokens = previous.already_scheduled_len + previous.extend_len;
    const std::int32_t new_page_hash_begin =
        static_cast<std::int32_t>(cache_progress.page_hashes.size());
    const std::int32_t filled_pages = num_computed_tokens / coordinator_.CacheBlockTokens();
    if (filled_pages > static_cast<std::int32_t>(cache_progress.page_hashes.size())) {
        appendCompletedPageHashes(cache_progress.page_hashes, request->FullPagedTokens(false), filled_pages);
    }
    std::optional<CacheBoundaryKind> completed_boundary_kind;
    if (new_page_hash_begin < static_cast<std::int32_t>(cache_progress.page_hashes.size())) {
        completed_boundary_kind =
            consumeCompletedBoundaryKind(cache_progress, num_computed_tokens, request->PrefillSize());
    }

    std::vector<BlockTable>& tables = request->BlockTablesRef();
    std::vector<GroupDemand> demands = makeGroupDemands(
        tables,
        GroupDemand{
            .num_tokens = tokens_this_round,
            .page_hashes = cache_progress.page_hashes,
            .new_page_hash_begin = new_page_hash_begin,
            .completed_boundary_kind = completed_boundary_kind,
            .num_computed_tokens = num_computed_tokens,
            .reserve_tokens = decode_reserve,
        });
    if (!admit(demands, cache_progress.access_epoch)) {
        return std::nullopt;
    }

    return fsm::SchedulePrefillEvent{
        tokens_this_round,
        decode_reserve,
        std::move(cache_progress),
    };
}

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(Request* request) {
    std::vector<BlockTable>& tables = request->BlockTablesRef();
    const std::int32_t reserve_tokens = request->ReserveNumTokensInNextScheduleEvent();
    fsm::CacheProgress cache_progress = request->CacheProgress();
    std::int32_t num_computed_tokens = 0;
    if (request->Is<fsm::PrefillDone>()) {
        const PrefillInfo previous = request->CurrentPrefillInfo();
        num_computed_tokens = previous.already_scheduled_len + previous.extend_len;
    } else {
        num_computed_tokens = request->TokenSize() - config_.decode_input_tokens;
    }

    const std::int32_t new_page_hash_begin =
        static_cast<std::int32_t>(cache_progress.page_hashes.size());
    const std::int32_t filled_pages = num_computed_tokens / coordinator_.CacheBlockTokens();
    if (filled_pages > static_cast<std::int32_t>(cache_progress.page_hashes.size())) {
        appendCompletedPageHashes(cache_progress.page_hashes, request->FullPagedTokens(false), filled_pages);
    }
    std::optional<CacheBoundaryKind> completed_boundary_kind;
    if (new_page_hash_begin < static_cast<std::int32_t>(cache_progress.page_hashes.size())) {
        completed_boundary_kind =
            consumeCompletedBoundaryKind(cache_progress, num_computed_tokens, request->PrefillSize());
    }

    if (new_page_hash_begin == static_cast<std::int32_t>(cache_progress.page_hashes.size()) &&
        canConsumeReservedTokensInPlace(coordinator_, tables, reserve_tokens, num_computed_tokens)) {
        coordinator_.ConsumeReservedTokens(tables, reserve_tokens);
    } else {
        std::vector<GroupDemand> demands = makeGroupDemands(
            tables,
            GroupDemand{
                .num_tokens = reserve_tokens,
                .page_hashes = cache_progress.page_hashes,
                .new_page_hash_begin = new_page_hash_begin,
                .completed_boundary_kind = completed_boundary_kind,
                .num_computed_tokens = num_computed_tokens,
            });
        if (!admit(demands, cache_progress.access_epoch)) {
            return std::nullopt;
        }
    }

    return fsm::ScheduleDecodeEvent{config_.decode_input_tokens, std::move(cache_progress)};
}

PrefillOperation Scheduler::applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                                        std::vector<LoadBackOperation>& load_back_operations) {
    PrefillOperation operation = applyPrefillEvent(*request, event, coordinator_, cache_group_ids_);
    std::vector<BlockTransfer> load_pairs = event.TakeLoadPairs();
    if (load_pairs.empty()) {
        return operation;
    }

    std::vector<TransferPair> transfers;
    transfers.reserve(load_pairs.size());
    LoadTicket ticket;
    ticket.host_pins.reserve(load_pairs.size());
    ticket.device_blocks.reserve(load_pairs.size());
    for (BlockTransfer& pair : load_pairs) {
        _assert(coordinator_.IsHostCachedBlock(pair.source->Location()),
                "pinned host page lost its cache entry before load emission");
        const KvCacheManager& manager =
            coordinator_.GroupManager(static_cast<std::int32_t>(pair.group_id));
        transfers.push_back(TransferPair{
            manager.ResolveKernelPageId(pair.source->Location()),
            manager.ResolveKernelPageId(pair.destination->Location()),
        });
        ticket.host_pins.push_back(std::move(pair.source));
        ticket.device_blocks.push_back(std::move(pair.destination));
    }
    const cache_op_id op_id = allocateCacheOpId();
    const bool inserted = load_ops_.emplace(op_id, std::move(ticket)).second;
    _assert(inserted, "duplicate loadback op id");
    load_back_operations.emplace_back(op_id, std::move(transfers));
    return operation;
}

PrefillOperation Scheduler::applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillEvent event) {
    return applyPrefillEvent(*request, event, coordinator_, cache_group_ids_);
}

DecodeOperation Scheduler::applyEventAndBuildOperation(Request* request, fsm::ScheduleDecodeEvent event) {
    const bool needs_bootstrap_token =
        request->Is<fsm::PrefillDone>() && config_.role == Role::kD;
    const std::int32_t bootstrap_token = needs_bootstrap_token ? request->LastToken() : -1;
    DecodeOperation operation =
        applyDecodeEvent(*request, std::move(event), config_.decode_input_tokens, coordinator_, cache_group_ids_);
    if (needs_bootstrap_token) {
        operation.decode_input_id = bootstrap_token;
    }
    return operation;
}

void Scheduler::retractForCapacity(const std::vector<Request*>& candidates) {
    if (config_.role != Role::kFused || !lcm_admission_failed_ ||
        !pending_forward_results_.empty() || !store_ops_.Empty() || !load_ops_.empty()) {
        return;
    }

    Request* request_to_retract = nullptr;
    for (Request* request : candidates) {
        if (request->Is<fsm::Decoding>() || request->Is<fsm::PrefillDone>()) {
            if (request_to_retract == nullptr || request->TokenSize() > request_to_retract->TokenSize()) {
                request_to_retract = request;
            }
        }
    }

    FatalCheck(request_to_retract != nullptr, "LCM admission failed without a retractable request");
    request_to_retract->Apply(fsm::RetractEvent{&coordinator_});
    ++retract_count_;
    spdlog::info("[Scheduler] retract: released request {} ({} tokens) for LCM capacity",
                 request_to_retract->Id(), request_to_retract->TokenSize());
}

std::pair<std::vector<ForwardOperation>, std::vector<LoadBackOperation>> Scheduler::buildForwardOperations(
    std::vector<Request*> candidates) {
    lcm_admission_failed_ = false;
    const auto priority = [this](const Request* request) {
        if (request->Is<fsm::Prefilling>()) {
            return 1;
        }
        if (request->Is<fsm::Submitted>()) {
            return 2;
        }
        if (request->Is<fsm::Decoding>() || request->Is<fsm::PrefillDone>()) {
            return config_.enable_mixed_prefill_decode ? 0 : 3;
        }
        return 9;
    };
    std::ranges::sort(candidates, [&](const Request* lhs, const Request* rhs) {
        const int lhs_priority = priority(lhs);
        const int rhs_priority = priority(rhs);
        return lhs_priority != rhs_priority ? lhs_priority < rhs_priority
                                            : lhs->Id() < rhs->Id();
    });

    const bool has_local_prefill =
        config_.role != Role::kD &&
        std::ranges::any_of(candidates, [](const Request* request) {
            return request->Is<fsm::Prefilling>() || request->Is<fsm::Submitted>();
        });
    const std::int32_t state_prefill_reserve =
        config_.enable_mixed_prefill_decode && coordinator_.HasMambaStateGroup() &&
                has_local_prefill
            ? coordinator_.CacheBlockTokens()
            : 0;

    std::vector<ForwardOperation> operations;
    std::vector<LoadBackOperation> load_back_operations;
    std::int32_t token_budget = config_.max_scheduled_tokens;
    bool pushed_prefill = false;
    auto push_operation = [&](auto operation) {
        if (config_.role != Role::kD) {
            token_budget -= operation.input_length;
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(operation)>, PrefillOperation>) {
            pushed_prefill = true;
        }
        operations.push_back(std::move(operation));
    };
    const auto trackPendingForwardResult = [&](const Request* request) {
        // Intermediate prefill and D-side cache admission do not produce a
        // forward::ExtendResult.
        if (request->Is<fsm::PrefillDone>() || request->Is<fsm::Decoding>()) {
            ++pending_forward_results_[request->Id()];
        }
    };

    for (Request* request : candidates) {
        if (token_budget <= 0 ||
            operations.size() == static_cast<std::size_t>(config_.max_batch_size)) {
            break;
        }

        if (request->Is<fsm::Prefilling>() && config_.role != Role::kD) {
            const std::int32_t reserve =
                config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            if (auto event = schedulePrefill(request, token_budget, reserve)) {
                push_operation(applyEventAndBuildOperation(request, std::move(*event)));
                if (config_.enable_pd_cache) {
                    pd_transfer_pins_.insert(request->Id());
                }
                trackPendingForwardResult(request);
            } else if (lcm_admission_failed_) {
                break;
            }
            continue;
        }

        if (request->Is<fsm::Submitted>()) {
            const std::int32_t decode_input_tokens =
                config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            const std::int32_t prefill_budget =
                config_.role == Role::kD ? request->PrefillSize() : token_budget;
            if (auto event =
                    schedulePrefillFirstChunk(request, prefill_budget, decode_input_tokens)) {
                push_operation(applyEventAndBuildOperation(request, std::move(*event), load_back_operations));
                if (config_.enable_pd_cache) {
                    pd_transfer_pins_.insert(request->Id());
                }
                trackPendingForwardResult(request);
            }
            continue;
        }

        if (request->Is<fsm::PrefillDone>() ||
            (request->Is<fsm::Decoding>() && config_.role != Role::kP)) {
            if (!config_.enable_mixed_prefill_decode && pushed_prefill) {
                break;
            }
            if (token_budget <
                state_prefill_reserve + config_.decode_input_tokens) {
                continue;
            }
            if (auto event = scheduleDecode(request)) {
                push_operation(applyEventAndBuildOperation(request, std::move(*event)));
                trackPendingForwardResult(request);
            }
        }
    }

    if (operations.empty()) {
        retractForCapacity(candidates);
    }
    return {std::move(operations), std::move(load_back_operations)};
}

}  // namespace tokenspeed
