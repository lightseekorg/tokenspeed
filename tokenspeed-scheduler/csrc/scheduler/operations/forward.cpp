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
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "cache/tier/transfer.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "scheduler/operations/cache.h"
#include "scheduler/operations/forward.h"
#include "cache/prefix/prefix_hasher.h"
#include "scheduler/request.h"
#include "utils.h"

namespace tokenspeed {

namespace {

constexpr std::int32_t kM16MaxDecodeDelayPasses = 2;
constexpr std::size_t kM16UsefulPrefillBatchSize = 2;

template <typename Operation>
void fillBlockTables(Operation& operation, Request& request, const CacheCoordinator& coordinator,
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

void appendCompletedPrefixHashes(std::vector<std::string>& prefix_hashes,
                                 const std::vector<std::span<const std::int32_t>>& prefix_pages,
                                 std::int32_t filled_prefix_pages) {
    const std::int32_t first_new_prefix_page = static_cast<std::int32_t>(prefix_hashes.size());
    _assert(filled_prefix_pages > first_new_prefix_page, "caller must pre-check page-hash progress");
    const std::string previous_hash = prefix_hashes.empty() ? std::string{} : prefix_hashes.back();
    std::vector<std::string> new_hashes =
        AdvancePrefixHashes(prefix_pages, first_new_prefix_page, previous_hash, filled_prefix_pages);
    prefix_hashes.insert(prefix_hashes.end(), std::make_move_iterator(new_hashes.begin()),
                         std::make_move_iterator(new_hashes.end()));
}

bool canConsumeReservedTokensInPlace(const CacheCoordinator& coordinator, std::span<const BlockTable> tables,
                                     std::int32_t num_tokens, std::int32_t num_computed_tokens) {
    for (std::int32_t i = 0; i < coordinator.NumGroups(); ++i) {
        const BlockTable& table = tables[static_cast<std::size_t>(i)];
        if (coordinator.GroupBlocksNeededFor(i, table, num_tokens) != 0 ||
            coordinator.GroupHasReclaimableBlocksAt(i, table, num_computed_tokens)) {
            return false;
        }
    }
    return true;
}

CacheBoundaryKind consumeCompletedBoundaryKind(fsm::CacheProgress& cache_progress, std::int32_t num_computed_tokens,
                                               std::int32_t prefill_size) {
    if (cache_progress.promotion_boundary_tokens > 0 &&
        num_computed_tokens >= cache_progress.promotion_boundary_tokens) {
        const bool reached_exactly = num_computed_tokens == cache_progress.promotion_boundary_tokens;
        cache_progress.promotion_boundary_tokens = 0;
        if (reached_exactly) {
            return CacheBoundaryKind::kPromoted;
        }
    }
    return num_computed_tokens == prefill_size ? CacheBoundaryKind::kEndpoint : CacheBoundaryKind::kChunk;
}

struct CompletedPrefixPages {
    std::int32_t first_new_prefix_page{0};
    std::optional<CacheBoundaryKind> boundary_kind;
};

CompletedPrefixPages updateCompletedPrefixHashes(Request& request, fsm::CacheProgress& cache_progress,
                                                 std::int32_t num_computed_tokens, std::int32_t prefix_granularity) {
    CompletedPrefixPages completed{
        .first_new_prefix_page = static_cast<std::int32_t>(cache_progress.prefix_hashes.size()),
    };
    const std::int32_t filled_prefix_pages = num_computed_tokens / prefix_granularity;
    if (filled_prefix_pages > static_cast<std::int32_t>(cache_progress.prefix_hashes.size())) {
        appendCompletedPrefixHashes(cache_progress.prefix_hashes, request.FullPrefixPages(false), filled_prefix_pages);
    }
    if (completed.first_new_prefix_page < static_cast<std::int32_t>(cache_progress.prefix_hashes.size())) {
        completed.boundary_kind =
            consumeCompletedBoundaryKind(cache_progress, num_computed_tokens, request.PrefillSize());
    }
    return completed;
}

template <typename Event>
    requires(std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent> || std::same_as<Event, fsm::SchedulePrefillEvent>)
PrefillOperation applyPrefillEvent(Request& request, Event& event, const CacheCoordinator& coordinator,
                                   std::span<const std::string> group_ids) {
    const fsm::PrefillSource source = [&] {
        if constexpr (std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent>) {
            return event.Source();
        }
        return request.PrefillSource();
    }();
    request.Apply(event);
    const PrefillInfo info = request.CurrentPrefillInfo();

    PrefillOperation operation;
    operation.request_id = request.Id();
    operation.request_pool_index = request.RequestPoolIndex();
    operation.input_length = info.extend_len;
    operation.prefill_length = request.PrefillSize();
    operation.input_ids.assign(info.input_ids.begin(), info.input_ids.end());
    operation.shifted_input_ids = info.shifted_input_ids;
    operation.extend_prefix_len = info.already_scheduled_len;
    operation.local_prefill = source == fsm::PrefillSource::kLocal;
    fillBlockTables(operation, request, coordinator, group_ids);
    return operation;
}

DecodeOperation applyDecodeEvent(Request& request, fsm::ScheduleDecodeEvent event, std::int32_t decode_input_tokens,
                                 const CacheCoordinator& coordinator, std::span<const std::string> group_ids) {
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
    const auto probe = [this, request](std::span<const std::string> hashes) {
        if (config_.role == Role::kD && !request->Is<fsm::Retracted>()) {
            return coordinator_.ProbeDecodeDevicePrefix(hashes);
        }
        return coordinator_.ProbePrefix(hashes);
    };
    const std::int32_t prefix_granularity = coordinator_.PrefixGranularity();
    // The final prompt token is always recomputed to produce logits. Some
    // consumers additionally require a larger prompt tail (for example, to
    // rebuild request-persistent state that is not stored in the KV cache).
    // Limit the probe itself so excluded hit pages are never claimed: admission
    // will allocate private writable pages for the replayed suffix.
    const std::int32_t replay_tokens = std::max(config_.prefix_replay_tokens, 1);
    const std::int32_t max_cacheable_tokens = std::max(request->PrefillSize() - replay_tokens, 0);
    const std::int32_t probe_prefix_pages = max_cacheable_tokens / prefix_granularity;
    const std::int32_t candidate_prefix_pages = std::max((request->PrefillSize() - 1) / prefix_granularity, 0);
    std::vector<std::span<const std::int32_t>> prefix_pages = request->FullPrefixPages(false);
    prefix_pages.resize(std::min(prefix_pages.size(), static_cast<std::size_t>(candidate_prefix_pages)));
    std::vector<std::string> hashes = ComputePrefixHashes(prefix_pages, "");
    const auto probe_hashes = std::span<const std::string>(hashes).first(
        std::min(hashes.size(), static_cast<std::size_t>(probe_prefix_pages)));

    AdmissionMatch match;
    match.candidate_prefix_hashes = hashes;
    // Retraction recovery may reuse its own L2 snapshot even when ordinary
    // request-to-request prefix reuse is disabled.
    if (config_.disable_prefix_cache && !request->Is<fsm::Retracted>()) {
        match.probe = probe({});
        return match;
    }
    match.probe = probe(probe_hashes);
    const std::int32_t hit_prefix_pages =
        std::max(match.probe.device.num_common_tokens, match.probe.host.num_common_tokens) / prefix_granularity;
    match.prefix_hashes.assign(hashes.begin(), hashes.begin() + hit_prefix_pages);

    const std::int32_t extension_pages =
        std::max(match.probe.host.num_common_tokens - match.probe.device.num_common_tokens, 0) / prefix_granularity;
    const auto extension_begin = hashes.begin() + match.probe.device.num_common_tokens / prefix_granularity;
    match.extension_hashes.assign(extension_begin, extension_begin + extension_pages);
    return match;
}

std::optional<CacheCoordinator::AdmissionResult> Scheduler::admit(PlanBuildContext& context,
                                                                  CacheCoordinator::PrefixProbe&& prefix,
                                                                  std::span<const GroupDemand> demands,
                                                                  std::optional<std::uint64_t> request_access_epoch) {
    std::optional<CacheCoordinator::AdmissionResult> result =
        coordinator_.Admit(std::move(prefix), demands, request_access_epoch);
    if (!result) {
        context.admission_failed = true;
        if (tier_transfers_.HasStoresInFlight()) {
            const auto pending_store_releases = tier_transfers_.DeviceLocationsReleasedOnStoreAck();
            context.waits_for_store_ack = context.waits_for_store_ack ||
                                          coordinator_.CanAdmitAfterReleasing(prefix, demands, pending_store_releases);
        }
        return std::nullopt;
    }

    _assert(result->new_page_ids.size() == cache_group_ids_.size(),
            "admission fresh-page groups must match scheduler config");
    for (std::size_t i = 0; i < result->new_page_ids.size(); ++i) {
        auto& page_ids = result->new_page_ids[i];
        auto& pending = context.plan.pages_to_zero[cache_group_ids_[i]];
        pending.insert(pending.end(), page_ids.begin(), page_ids.end());
    }
    return result;
}

std::optional<CacheCoordinator::AdmissionResult> Scheduler::admit(PlanBuildContext& context,
                                                                  std::span<const GroupDemand> demands,
                                                                  std::uint64_t request_access_epoch) {
    return admit(context, coordinator_.ProbePrefix({}), demands, request_access_epoch);
}

bool Scheduler::admitWithKvEventTracking(PlanBuildContext& context, Request& request,
                                         const fsm::CacheProgress& cache_progress, std::int32_t new_prefix_hash_begin,
                                         std::span<const GroupDemand> demands) {
    std::vector<CacheKey> event_keys =
        registerKvEventPrefixPages(request, cache_progress.prefix_hashes, new_prefix_hash_begin);
    const bool admitted = admit(context, demands, cache_progress.access_epoch).has_value();
    discardUncachedKvEventPages(event_keys);
    return admitted;
}

std::optional<fsm::SchedulePrefillFirstChunkEvent> Scheduler::schedulePrefillFirstChunk(
    PlanBuildContext& context, Request* request, std::int32_t remaining, std::int32_t decode_input_tokens) {
    if (req_pool_allocator_.AvailableSlots() == 0) {
        return std::nullopt;
    }

    AdmissionMatch match = matchPrefixAtAdmission(request);
    const std::int32_t hit_tokens = std::max(match.probe.device.num_common_tokens, match.probe.host.num_common_tokens);
    const std::int32_t promotion_boundary_tokens = coordinator_.PromotionBoundaryTokens(match.probe);
    _assert(promotion_boundary_tokens == 0 ||
                (promotion_boundary_tokens % coordinator_.PrefixGranularity() == 0 &&
                 promotion_boundary_tokens > hit_tokens && promotion_boundary_tokens < request->PrefillSize()),
            "promotion boundary must be page-aligned and inside the unmatched prompt");

    const std::int32_t unscheduled = request->PrefillSize() - hit_tokens;
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    if (coordinator_.HasMambaStateGroup() || promotion_boundary_tokens > 0) {
        tokens_this_round = AlignPrefillChunk(hit_tokens, unscheduled, remaining, coordinator_.PrefixGranularity(),
                                              promotion_boundary_tokens);
        if (tokens_this_round == 0) {
            return std::nullopt;
        }
    }

    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t decode_reserve = completes_prefill ? decode_input_tokens : 0;
    std::vector<BlockTable> tables(static_cast<std::size_t>(coordinator_.NumGroups()));
    std::vector<GroupDemand> demands =
        makeGroupDemands(tables, GroupDemand{.num_tokens = tokens_this_round, .reserve_tokens = decode_reserve});

    const fsm::PrefillSource source = config_.role == Role::kD && request->Is<fsm::Submitted>()
                                          ? fsm::PrefillSource::kRemote
                                          : fsm::PrefillSource::kLocal;
    if (config_.enable_pd_cache && source == fsm::PrefillSource::kRemote) {
        for (std::size_t i = 0; i < demands.size(); ++i) {
            const CacheGroupConfig& group = config_.cache_groups[i];
            const std::int32_t block_granularity = coordinator_.GroupBlockGranularity(i);
            if (group.transfer_policy == CacheTransferPolicy::LatestSnapshot) {
                demands[i].num_tokens = request->PrefillSize();
                demands[i].materialized_suffix_start = (request->PrefillSize() - 1) / block_granularity;
            } else if (group.retention == CacheGroupConfig::Retention::SlidingWindow) {
                const std::int32_t retained_begin =
                    std::max(0, request->PrefillSize() - *group.sliding_window_tokens + 1);
                demands[i].num_tokens = request->PrefillSize();
                demands[i].materialized_suffix_start =
                    std::max(hit_tokens / block_granularity, retained_begin / block_granularity);
            }
        }
    }

    std::vector<CacheKey> event_keys = registerKvEventPrefixPages(*request, match.candidate_prefix_hashes, 0);
    std::optional<CacheCoordinator::AdmissionResult> admission = admit(context, std::move(match.probe), demands);
    if (!admission) {
        context.capacity_blocker = request->Id();
        discardUncachedKvEventPages(event_keys);
        return std::nullopt;
    }
    _assert(admission->promotion_boundary_tokens == promotion_boundary_tokens,
            "promotion boundary changed between probe and admission");

    if (!match.extension_hashes.empty()) {
        coordinator_.CacheFullBlocks(tables, match.extension_hashes, admission->access_epoch,
                                     admission->device_prefix_tokens / coordinator_.PrefixGranularity());
    }
    discardUncachedKvEventPages(event_keys);
    return fsm::SchedulePrefillFirstChunkEvent{
        tokens_this_round,
        decode_reserve,
        &req_pool_allocator_,
        source,
        &coordinator_,
        std::move(tables),
        hit_tokens,
        fsm::CacheProgress{
            .prefix_hashes = std::move(match.prefix_hashes),
            .access_epoch = admission->access_epoch,
            .promotion_boundary_tokens = admission->promotion_boundary_tokens,
        },
        std::move(admission->load_pairs),
    };
}

std::optional<fsm::SchedulePrefillEvent> Scheduler::schedulePrefill(
    PlanBuildContext& context, Request* request, std::int32_t remaining,
    std::int32_t reserve_num_tokens_in_next_schedule_event) {
    const std::int32_t unscheduled = request->UnscheduledPrefillSize();
    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    fsm::CacheProgress cache_progress = request->CacheProgress();
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    if (coordinator_.HasMambaStateGroup() || cache_progress.promotion_boundary_tokens > 0) {
        tokens_this_round = AlignPrefillChunk(first_pos, unscheduled, remaining, coordinator_.PrefixGranularity(),
                                              cache_progress.promotion_boundary_tokens);
        if (tokens_this_round == 0) {
            return std::nullopt;
        }
    }

    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t decode_reserve = completes_prefill ? reserve_num_tokens_in_next_schedule_event : 0;
    const PrefillInfo previous = request->CurrentPrefillInfo();
    const std::int32_t num_computed_tokens = previous.already_scheduled_len + previous.extend_len;
    const CompletedPrefixPages completed =
        updateCompletedPrefixHashes(*request, cache_progress, num_computed_tokens, coordinator_.PrefixGranularity());

    std::vector<BlockTable>& tables = request->BlockTablesRef();
    std::vector<GroupDemand> demands =
        makeGroupDemands(tables, GroupDemand{
                                     .num_tokens = tokens_this_round,
                                     .prefix_hashes = cache_progress.prefix_hashes,
                                     .new_prefix_hash_begin = completed.first_new_prefix_page,
                                     .completed_boundary_kind = completed.boundary_kind,
                                     .num_computed_tokens = num_computed_tokens,
                                     .reserve_tokens = decode_reserve,
                                 });
    if (!admitWithKvEventTracking(context, *request, cache_progress, completed.first_new_prefix_page, demands)) {
        context.capacity_blocker = request->Id();
        return std::nullopt;
    }

    return fsm::SchedulePrefillEvent{
        tokens_this_round,
        decode_reserve,
        std::move(cache_progress),
    };
}

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(PlanBuildContext& context, Request* request) {
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

    const CompletedPrefixPages completed =
        updateCompletedPrefixHashes(*request, cache_progress, num_computed_tokens, coordinator_.PrefixGranularity());

    if (completed.first_new_prefix_page == static_cast<std::int32_t>(cache_progress.prefix_hashes.size()) &&
        canConsumeReservedTokensInPlace(coordinator_, tables, reserve_tokens, num_computed_tokens)) {
        coordinator_.ConsumeReservedTokens(tables, reserve_tokens);
    } else {
        std::vector<GroupDemand> demands =
            makeGroupDemands(tables, GroupDemand{
                                         .num_tokens = reserve_tokens,
                                         .prefix_hashes = cache_progress.prefix_hashes,
                                         .new_prefix_hash_begin = completed.first_new_prefix_page,
                                         .completed_boundary_kind = completed.boundary_kind,
                                         .num_computed_tokens = num_computed_tokens,
                                     });
        if (!admitWithKvEventTracking(context, *request, cache_progress, completed.first_new_prefix_page, demands)) {
            context.capacity_blocker = request->Id();
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

    load_back_operations.push_back(tier_transfers_.StartPrefixLoad(std::move(load_pairs)));
    return operation;
}

PrefillOperation Scheduler::applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillEvent event) {
    return applyPrefillEvent(*request, event, coordinator_, cache_group_ids_);
}

DecodeOperation Scheduler::applyEventAndBuildOperation(Request* request, fsm::ScheduleDecodeEvent event) {
    const bool needs_bootstrap_token = request->Is<fsm::PrefillDone>() && config_.role == Role::kD;
    const std::int32_t bootstrap_token = needs_bootstrap_token ? request->LastToken() : -1;
    DecodeOperation operation =
        applyDecodeEvent(*request, std::move(event), config_.decode_input_tokens, coordinator_, cache_group_ids_);
    if (needs_bootstrap_token) {
        operation.decode_input_id = bootstrap_token;
    }
    return operation;
}

std::optional<WriteBackOperation> Scheduler::beginRetraction(Request& request) {
    fsm::CacheProgress cache_progress = request.CacheProgress();
    const std::int32_t num_computed_tokens = request.TokenSize() - config_.decode_input_tokens;
    const CompletedPrefixPages completed =
        updateCompletedPrefixHashes(request, cache_progress, num_computed_tokens, coordinator_.PrefixGranularity());
    if (completed.boundary_kind) {
        coordinator_.CacheCompletedBlocks(request.BlockTablesRef(), cache_progress.prefix_hashes,
                                          cache_progress.access_epoch, completed.first_new_prefix_page,
                                          num_computed_tokens, *completed.boundary_kind);
    }
    coordinator_.QueueCachedBlocksForStore(cache_progress.prefix_hashes);
    std::optional<WriteBackOperation> write_back = tier_transfers_.StartPendingStores();
    request.Apply(fsm::RetractionEvent{&coordinator_});
    recovery_queue_.push_back(request.Id());
    return write_back;
}

void Scheduler::retractForCapacity(PlanBuildContext& context, const std::vector<Request*>& candidates,
                                   std::vector<WriteBackOperation>& write_back_operations) {
    if ((config_.role != Role::kFused && config_.role != Role::kD) || !context.admission_failed ||
        !pending_forward_results_.empty() || !pd_transfer_pins_.empty() || context.waits_for_store_ack ||
        tier_transfers_.HasLoadBacksInFlight()) {
        return;
    }

    if (config_.role == Role::kD) {
        Request* victim = nullptr;
        std::optional<std::tuple<std::int32_t, std::int32_t, std::string>> victim_rank;
        for (Request* request : candidates) {
            if (!request->Is<fsm::Decoding>()) {
                continue;
            }
            auto rank = std::tuple{-coordinator_.NumNewlyReleasableLcmBlocks(request->BlockTablesRef()),
                                   request->TokenSize(), request->Id()};
            if (!victim_rank || rank < *victim_rank) {
                victim = request;
                victim_rank = std::move(rank);
            }
        }
        FatalCheck(victim != nullptr,
                   "cache admission failed without a retractable Decode request or asynchronous capacity release");
        recovery_barrier_ = context.capacity_blocker;
        if (!recovery_barrier_ || *recovery_barrier_ == victim->Id()) {
            const auto waiting = std::ranges::find_if(candidates, [victim](const Request* request) {
                return request != victim && (request->Is<fsm::Submitted>() || request->Is<fsm::Prefilling>());
            });
            recovery_barrier_ =
                waiting == candidates.end() ? std::nullopt : std::optional<std::string>{(*waiting)->Id()};
        }
        if (auto operation = beginRetraction(*victim)) {
            write_back_operations.push_back(std::move(*operation));
        }
        spdlog::info("[Scheduler] retract: released request {} ({} tokens) with best-effort L2 store", victim->Id(),
                     victim->TokenSize());
        return;
    }

    Request* request_to_retract = nullptr;
    for (Request* request : candidates) {
        if ((request->Is<fsm::Decoding>() || request->Is<fsm::PrefillDone>()) &&
            (request_to_retract == nullptr || request->TokenSize() > request_to_retract->TokenSize())) {
            request_to_retract = request;
        }
    }
    FatalCheck(request_to_retract != nullptr, "cache admission failed without a retractable request");
    if (config_.HasHostCache() && request_to_retract->Is<fsm::Decoding>()) {
        if (auto operation = beginRetraction(*request_to_retract)) {
            write_back_operations.push_back(std::move(*operation));
        }
        spdlog::info("[Scheduler] retract: released request {} ({} tokens) with best-effort L2 store",
                     request_to_retract->Id(), request_to_retract->TokenSize());
        return;
    }
    request_to_retract->Apply(fsm::RetractEvent{&coordinator_});
    spdlog::info("[Scheduler] retract: released request {} ({} tokens) for cache capacity", request_to_retract->Id(),
                 request_to_retract->TokenSize());
}

std::pair<std::vector<ForwardOperation>, std::vector<LoadBackOperation>> Scheduler::buildForwardOperations(
    ExecutionPlan& plan, std::vector<Request*> candidates, std::vector<WriteBackOperation>& write_back_operations) {
    PlanBuildContext context{plan};
    while (!recovery_queue_.empty()) {
        Request* request = findRequest(recovery_queue_.front());
        if (request != nullptr && !request->Is<fsm::Finished>()) {
            break;
        }
        recovery_queue_.pop_front();
    }
    if (recovery_barrier_) {
        Request* request = findRequest(*recovery_barrier_);
        if (request == nullptr || request->Is<fsm::Finished>() || request->Is<fsm::Retracted>()) {
            recovery_barrier_.reset();
        }
    }
    bool delay_submitted_prefills = false;
    if (config_.enable_experimental_m16_prefill_delayer) {
        std::erase_if(newly_submitted_prefills_, [this](const std::string& id) {
            Request* request = findRequest(id);
            return request == nullptr || !request->Is<fsm::Submitted>();
        });
        std::erase_if(submitted_prefill_delay_passes_, [this](const auto& item) {
            Request* request = findRequest(item.first);
            return request == nullptr || !request->Is<fsm::Submitted>() ||
                   !newly_submitted_prefills_.contains(item.first);
        });
        const std::size_t submitted_prefills =
            static_cast<std::size_t>(std::ranges::count_if(candidates, [this](const Request* request) {
                return request->Is<fsm::Submitted>() && newly_submitted_prefills_.contains(request->Id());
            }));
        const bool has_decode_work = std::ranges::any_of(candidates, [](const Request* request) {
            return request->Is<fsm::PrefillDone>() || request->Is<fsm::Decoding>();
        });
        const bool has_in_progress_prefill = std::ranges::any_of(candidates, [this](const Request* request) {
            return request->Is<fsm::Prefilling>() || request->Is<fsm::Retracted>() ||
                   (request->Is<fsm::Submitted>() && !newly_submitted_prefills_.contains(request->Id()));
        });
        const std::size_t useful_prefill_batch_size =
            std::min(kM16UsefulPrefillBatchSize, static_cast<std::size_t>(config_.max_batch_size));
        const bool delay_budget_remaining = std::ranges::all_of(candidates, [this](const Request* request) {
            if (!request->Is<fsm::Submitted>() || !newly_submitted_prefills_.contains(request->Id())) {
                return true;
            }
            const auto it = submitted_prefill_delay_passes_.find(request->Id());
            return it == submitted_prefill_delay_passes_.end() || it->second < kM16MaxDecodeDelayPasses;
        });
        delay_submitted_prefills = submitted_prefills > 0 && submitted_prefills < useful_prefill_batch_size &&
                                   has_decode_work && !has_in_progress_prefill && delay_budget_remaining;
    }

    const auto priority = [this, delay_submitted_prefills](const Request* request) {
        const bool recovery_front = !recovery_queue_.empty() && request->Id() == recovery_queue_.front();
        const bool local_decode_prefill =
            request->Is<fsm::Prefilling>() && request->PrefillSource() == fsm::PrefillSource::kLocal;
        if (config_.role == Role::kP && request->Is<fsm::PrefillDone>()) {
            // Completed P-side prefills hold their KV pages until the remote
            // transfer finishes, so start their handoffs before admitting
            // more prompt tokens.
            return 0;
        }
        if (config_.role == Role::kD && (local_decode_prefill || request->Is<fsm::PrefillDone>() ||
                                         (recovery_front && request->Is<fsm::Decoding>()))) {
            // Keep the oldest retracted request at the head of line through
            // local recovery and Decode completion. Starting another recovery
            // earlier can make the two requests repeatedly evict each other.
            return 0;
        }
        if (recovery_barrier_ && request->Id() == *recovery_barrier_) {
            return 1;
        }
        if (request->Is<fsm::Retracted>() && !recovery_queue_.empty() && request->Id() == recovery_queue_.front()) {
            return 2;
        }
        if (request->Is<fsm::Prefilling>()) {
            return 3;
        }
        if (request->Is<fsm::Submitted>()) {
            return delay_submitted_prefills ? 6 : 4;
        }
        if (request->Is<fsm::Decoding>() || request->Is<fsm::PrefillDone>()) {
            return config_.enable_mixed_prefill_decode ? 3 : 5;
        }
        return 10;
    };
    std::ranges::sort(candidates, [&](const Request* lhs, const Request* rhs) {
        const int lhs_priority = priority(lhs);
        const int rhs_priority = priority(rhs);
        return lhs_priority != rhs_priority ? lhs_priority < rhs_priority : lhs->Id() < rhs->Id();
    });

    // EventLoop selects the P-side KV-transfer path only for an all-zero-extend
    // ForwardBatch.  Once the highest-priority candidate selects that batch
    // kind, collect every consecutive PrefillDone candidate but do not mix in
    // local prefill work.  PrefillDone candidates are contiguous because they
    // all have priority zero above.
    const bool build_prefill_handoff_batch =
        config_.role == Role::kP && !candidates.empty() && candidates.front()->Is<fsm::PrefillDone>();

    const bool has_local_prefill = std::ranges::any_of(candidates, [this](const Request* request) {
        return (request->Is<fsm::Prefilling>() && request->PrefillSource() == fsm::PrefillSource::kLocal) ||
               (config_.role != Role::kD && request->Is<fsm::Submitted>()) ||
               (request->Is<fsm::Retracted>() && !recovery_queue_.empty() && request->Id() == recovery_queue_.front());
    });
    const std::int32_t state_prefill_reserve = !build_prefill_handoff_batch && config_.enable_mixed_prefill_decode &&
                                                       coordinator_.HasMambaStateGroup() && has_local_prefill
                                                   ? coordinator_.PrefixGranularity()
                                                   : 0;

    std::vector<ForwardOperation> operations;
    std::vector<LoadBackOperation> load_back_operations;
    std::int32_t token_budget = config_.max_scheduled_tokens;
    bool pushed_prefill = false;
    bool pushed_decode = false;
    auto push_operation = [&](auto operation) {
        if (recovery_barrier_ && operation.request_id == *recovery_barrier_) {
            recovery_barrier_.reset();
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(operation)>, PrefillOperation>) {
            if (config_.role != Role::kD || operation.local_prefill) {
                token_budget -= operation.input_length;
            }
            pushed_prefill = true;
        } else if (config_.role != Role::kD) {
            token_budget -= operation.input_length;
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(operation)>, DecodeOperation>) {
            pushed_decode = true;
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
        if (token_budget <= 0 || operations.size() == static_cast<std::size_t>(config_.max_batch_size)) {
            break;
        }
        if (build_prefill_handoff_batch && !request->Is<fsm::PrefillDone>()) {
            break;
        }

        if (request->Is<fsm::Prefilling>() &&
            (config_.role != Role::kD || request->PrefillSource() == fsm::PrefillSource::kLocal)) {
            if (config_.role == Role::kD && pushed_decode) {
                break;
            }
            const std::int32_t reserve = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            if (auto event = schedulePrefill(context, request, token_budget, reserve)) {
                push_operation(applyEventAndBuildOperation(request, std::move(*event)));
                // P-side pages stay pinned until the PD transfer completes.
                // D-side local recovery has no corresponding PD ACK; its
                // Host/Device lifetime is already owned by the L2 load ticket.
                if (config_.enable_pd_cache && config_.role != Role::kD) {
                    pd_transfer_pins_.insert(request->Id());
                }
                trackPendingForwardResult(request);
                if (config_.role == Role::kD) {
                    // Decode-side recovery prefill runs in its own batch.
                    break;
                }
                if (request->Is<fsm::Prefilling>()) {
                    // Admission reserves only this chunk, not the request's
                    // remaining prompt. Keep one incomplete prefill as the
                    // head of line so another request cannot strand it by
                    // consuming the capacity it needs to finish.
                    break;
                }
            } else if (context.admission_failed) {
                break;
            }
            continue;
        }

        if (request->Is<fsm::Retracted>() && !recovery_queue_.empty() && request->Id() == recovery_queue_.front()) {
            if (config_.role == Role::kD && pushed_decode) {
                break;
            }
            if (auto event = schedulePrefillFirstChunk(context, request, token_budget, config_.decode_input_tokens)) {
                push_operation(applyEventAndBuildOperation(request, std::move(*event), load_back_operations));
                trackPendingForwardResult(request);
                if (config_.role == Role::kD || request->Is<fsm::Prefilling>()) {
                    // Keep a Decode-side recovery batch local-only.
                    break;
                }
                continue;
            }
            break;
        }

        if (request->Is<fsm::Submitted>()) {
            if (delay_submitted_prefills && pushed_decode) {
                break;
            }
            if (config_.role == Role::kD && pushed_decode) {
                break;
            }
            const std::int32_t decode_input_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            const std::int32_t prefill_budget = config_.role == Role::kD ? request->PrefillSize() : token_budget;
            if (auto event = schedulePrefillFirstChunk(context, request, prefill_budget, decode_input_tokens)) {
                push_operation(applyEventAndBuildOperation(request, std::move(*event), load_back_operations));
                if (config_.enable_pd_cache) {
                    pd_transfer_pins_.insert(request->Id());
                }
                trackPendingForwardResult(request);
                if (request->Is<fsm::Prefilling>()) {
                    break;
                }
            }
            continue;
        }

        if (request->Is<fsm::PrefillDone>() || (request->Is<fsm::Decoding>() && config_.role != Role::kP)) {
            if ((config_.role == Role::kD || !config_.enable_mixed_prefill_decode) && pushed_prefill) {
                break;
            }
            if (token_budget < state_prefill_reserve + config_.decode_input_tokens) {
                continue;
            }
            if (auto event = scheduleDecode(context, request)) {
                push_operation(applyEventAndBuildOperation(request, std::move(*event)));
                trackPendingForwardResult(request);
            }
        }
    }

    if (operations.empty() && context.admission_failed) {
        retractForCapacity(context, candidates, write_back_operations);
    }
    if (delay_submitted_prefills && pushed_decode) {
        for (const Request* request : candidates) {
            if (request->Is<fsm::Submitted>() && newly_submitted_prefills_.contains(request->Id())) {
                ++submitted_prefill_delay_passes_[request->Id()];
            }
        }
    }
    return {std::move(operations), std::move(load_back_operations)};
}

}  // namespace tokenspeed
