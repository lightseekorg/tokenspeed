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

// Decode headroom every admission secures up front, and again per retraction
// the request has suffered, until it reaches the request's own generation
// budget. Large enough that a request escapes the retract/readmit cycle
// within a couple of rounds rather than inching toward safety.
constexpr std::int32_t kRetractionSafeSteps = 4096;

// A retracted request with no L2 snapshot re-prefills from scratch: for
// admission it behaves exactly like a newly submitted prompt.
bool admitsLikeNewPrompt(const Request& request) {
    if (request.Is<fsm::Submitted>()) {
        return true;
    }
    const auto* retracted = request.GetIf<fsm::Retracted>();
    return retracted != nullptr && !retracted->HasRecoverableSnapshot();
}

// An incomplete prefill keeps the head of line: admission reserved only this
// chunk, and a later candidate could strand it by consuming the capacity it
// needs to finish. (A reserved mamba state-checkpoint tail already holds its
// pages, so the line may move on.)
bool holdsHeadOfLine(const Request& request) {
    const auto* prefilling = request.GetIf<fsm::Prefilling>();
    return prefilling != nullptr && !prefilling->TailCheckpointReserved();
}

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

void makeSnapshotStatePrefillSparse(std::span<GroupDemand> demands, std::span<const CacheGroupConfig> cache_groups,
                                    const CacheCoordinator& coordinator, std::int32_t after_tokens) {
    _assert(demands.size() == cache_groups.size(), "demands/cache groups size mismatch");
    _assert(after_tokens > 0, "snapshot-state prefill requires a positive endpoint");
    for (std::size_t i = 0; i < demands.size(); ++i) {
        if (!cache_groups[i].IsSnapshotStateGroup()) {
            continue;
        }
        const std::int32_t block_granularity = coordinator.GroupBlockGranularity(static_cast<std::int32_t>(i));
        demands[i].num_tokens = after_tokens;
        demands[i].materialized_suffix_start = (after_tokens - 1) / block_granularity;
    }
}

void setSnapshotStatePrefillReserve(std::span<GroupDemand> demands, std::span<const CacheGroupConfig> cache_groups,
                                    std::int32_t reserve_tokens) {
    _assert(demands.size() == cache_groups.size(), "demands/cache groups size mismatch");
    for (std::size_t i = 0; i < demands.size(); ++i) {
        if (cache_groups[i].IsSnapshotStateGroup()) {
            demands[i].reserve_tokens = reserve_tokens;
        }
    }
}

// Only a local prefill splits off its final state checkpoint -- and outside
// the D role every prefill is local (a D-role admission is the peer's work,
// riding plan.remote_prefill).
bool shouldSplitFinalStateCheckpoint(const SchedulerConfig& config, const CacheCoordinator& coordinator) {
    return config.role != Role::kD && !config.disable_prefix_cache && coordinator.HasMambaStateGroup();
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

std::optional<CacheCoordinator::AdmissionResult> Scheduler::admit(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                                  CacheCoordinator::PrefixProbe&& prefix,
                                                                  std::span<const GroupDemand> demands,
                                                                  std::optional<std::uint64_t> request_access_epoch) {
    std::optional<CacheCoordinator::AdmissionResult> result =
        coordinator_.Admit(std::move(prefix), demands, request_access_epoch);
    if (!result) {
        feedback.admission_failed = true;
        return std::nullopt;
    }

    _assert(result->new_page_ids.size() == cache_group_ids_.size(),
            "admission fresh-page groups must match scheduler config");
    for (std::size_t i = 0; i < result->new_page_ids.size(); ++i) {
        auto& page_ids = result->new_page_ids[i];
        auto& pending = plan.pages_to_zero[cache_group_ids_[i]];
        pending.insert(pending.end(), page_ids.begin(), page_ids.end());
    }
    return result;
}

std::optional<CacheCoordinator::AdmissionResult> Scheduler::admit(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                                  std::span<const GroupDemand> demands,
                                                                  std::uint64_t request_access_epoch) {
    return admit(plan, feedback, coordinator_.ProbePrefix({}), demands, request_access_epoch);
}

bool Scheduler::admitWithKvEventTracking(ExecutionPlan& plan, AdmissionFeedback& feedback, Request& request,
                                         const fsm::CacheProgress& cache_progress, std::int32_t new_prefix_hash_begin,
                                         std::span<const GroupDemand> demands) {
    std::vector<CacheKey> event_keys =
        registerKvEventPrefixPages(request, cache_progress.prefix_hashes, new_prefix_hash_begin);
    const bool admitted = admit(plan, feedback, demands, cache_progress.access_epoch).has_value();
    discardUncachedKvEventPages(event_keys);
    return admitted;
}

std::optional<fsm::SchedulePrefillFirstChunkEvent> Scheduler::schedulePrefillFirstChunk(
    ExecutionPlan& plan, AdmissionFeedback& feedback, Request* request, std::int32_t remaining,
    std::int32_t decode_input_tokens) {
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

    const fsm::PrefillSource source = config_.role == Role::kD && request->Is<fsm::Submitted>()
                                          ? fsm::PrefillSource::kRemote
                                          : fsm::PrefillSource::kLocal;
    const std::int32_t unscheduled = request->PrefillSize() - hit_tokens;
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    std::optional<std::int32_t> final_tail_tokens;
    if (coordinator_.HasMambaStateGroup() || promotion_boundary_tokens > 0) {
        if (shouldSplitFinalStateCheckpoint(config_, coordinator_)) {
            final_tail_tokens = FinalAlignedTailTokens(hit_tokens, unscheduled, remaining,
                                                       coordinator_.PrefixGranularity(), promotion_boundary_tokens);
        }
        tokens_this_round = final_tail_tokens
                                ? unscheduled - *final_tail_tokens
                                : AlignPrefillChunk(hit_tokens, unscheduled, remaining,
                                                    coordinator_.PrefixGranularity(), promotion_boundary_tokens);
        if (tokens_this_round == 0) {
            return std::nullopt;
        }
    }

    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t decode_reserve = completes_prefill ? decode_input_tokens : 0;
    const std::int32_t split_tail_tokens = final_tail_tokens.value_or(0);
    std::int32_t admission_reserve = split_tail_tokens > 0 ? split_tail_tokens + decode_input_tokens : decode_reserve;
    // Every admission on a decoding role (D, Fused) secures real headroom
    // before the prefill starts: the rest of the prompt plus decode room
    // that starts at one safe-step window and grows with each retraction
    // (Request::AdmissionHeadroom). Pages only -- the request still computes
    // one chunk per round, because the chunk size is a forward-pass limit
    // rather than a capacity one. The P role is exempt: it never decodes
    // locally and never retracts, so there is no decode room to prepay.
    if (const std::int32_t headroom = config_.role == Role::kP ? 0 : request->AdmissionHeadroom(kRetractionSafeSteps);
        headroom > 0) {
        admission_reserve = std::max(admission_reserve, unscheduled - tokens_this_round + headroom);
    }
    std::vector<BlockTable> tables(static_cast<std::size_t>(coordinator_.NumGroups()));
    std::vector<GroupDemand> demands =
        makeGroupDemands(tables, GroupDemand{.num_tokens = tokens_this_round, .reserve_tokens = admission_reserve});
    if (source == fsm::PrefillSource::kLocal) {
        makeSnapshotStatePrefillSparse(demands, config_.cache_groups, coordinator_, hit_tokens + tokens_this_round);
    }

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
    setSnapshotStatePrefillReserve(demands, config_.cache_groups, split_tail_tokens);
    std::vector<CacheKey> event_keys = registerKvEventPrefixPages(*request, match.candidate_prefix_hashes, 0);
    std::optional<CacheCoordinator::AdmissionResult> admission = admit(plan, feedback, std::move(match.probe), demands);
    if (!admission) {
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
            .state_checkpoint_tail_reserved = split_tail_tokens > 0,
        },
        std::move(admission->load_pairs),
        // The P role holds a completed prompt until its result lands: the
        // remote decode that hands it off carries the bootstrap token.
        config_.role == Role::kP,
    };
}

std::optional<fsm::SchedulePrefillEvent> Scheduler::schedulePrefill(
    ExecutionPlan& plan, AdmissionFeedback& feedback, Request* request, std::int32_t remaining,
    std::int32_t reserve_num_tokens_in_next_schedule_event) {
    const std::int32_t unscheduled = request->UnscheduledPrefillSize();
    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    fsm::CacheProgress cache_progress = request->CacheProgress();
    // Spend a state-checkpoint tail banked by the previous round's admission
    // (read-and-clear at entry, so a tail banked BELOW is untouched): its
    // pages are already held, so this round must complete it -- asserted
    // once the chunk size is fixed -- and skips its sparse re-shaping.
    const bool consumes_reserved_tail = std::exchange(cache_progress.state_checkpoint_tail_reserved, false);
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    std::optional<std::int32_t> final_tail_tokens;
    if (coordinator_.HasMambaStateGroup() || cache_progress.promotion_boundary_tokens > 0) {
        if (shouldSplitFinalStateCheckpoint(config_, coordinator_)) {
            final_tail_tokens =
                FinalAlignedTailTokens(first_pos, unscheduled, remaining, coordinator_.PrefixGranularity(),
                                       cache_progress.promotion_boundary_tokens);
        }
        tokens_this_round = final_tail_tokens
                                ? unscheduled - *final_tail_tokens
                                : AlignPrefillChunk(first_pos, unscheduled, remaining, coordinator_.PrefixGranularity(),
                                                    cache_progress.promotion_boundary_tokens);
        if (final_tail_tokens) {
            _assert(!consumes_reserved_tail, "cannot nest reserved state-checkpoint tails");
            cache_progress.state_checkpoint_tail_reserved = true;
        }
        if (tokens_this_round == 0) {
            return std::nullopt;
        }
    }

    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t decode_reserve = completes_prefill ? reserve_num_tokens_in_next_schedule_event : 0;
    const std::int32_t checkpoint_tail_reserve = final_tail_tokens.value_or(0);
    const std::int32_t admission_reserve =
        final_tail_tokens ? checkpoint_tail_reserve + reserve_num_tokens_in_next_schedule_event : decode_reserve;
    _assert(!consumes_reserved_tail || tokens_this_round == unscheduled,
            "reserved state-checkpoint tail must complete in one round");
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
                                     .reserve_tokens = admission_reserve,
                                 });
    if (!consumes_reserved_tail) {
        makeSnapshotStatePrefillSparse(demands, config_.cache_groups, coordinator_, first_pos + tokens_this_round);
    }
    setSnapshotStatePrefillReserve(demands, config_.cache_groups, checkpoint_tail_reserve);
    if (!admitWithKvEventTracking(plan, feedback, *request, cache_progress, completed.first_new_prefix_page, demands)) {
        return std::nullopt;
    }

    return fsm::SchedulePrefillEvent{
        tokens_this_round,
        decode_reserve,
        std::move(cache_progress),
        config_.role == Role::kP,
    };
}

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                                  Request* request) {
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
        if (!admitWithKvEventTracking(plan, feedback, *request, cache_progress, completed.first_new_prefix_page,
                                      demands)) {
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
    // A decode op carries its token when its executor cannot otherwise know
    // it: the D side's first decode (the token crossed the wire with
    // RemotePrefillDoneEvent) and the P side's remote decode (the peer sends
    // it on as the bootstrap token, and the P grammar holds the op until the
    // result lands). Fused stays -1 on purpose: overlap plans the decode
    // BEFORE the result lands, and the device fills the input from its
    // in-flight capture.
    const bool needs_bootstrap_token = request->Is<fsm::PrefillDone>() && config_.role != Role::kFused;
    const std::int32_t bootstrap_token = needs_bootstrap_token ? request->LastToken() : -1;
    std::vector<std::int32_t> spec_candidate_ids =
        config_.role == Role::kP && needs_bootstrap_token ? request->TakeSpecCandidates() : std::vector<std::int32_t>{};
    DecodeOperation operation =
        applyDecodeEvent(*request, std::move(event), config_.decode_input_tokens, coordinator_, cache_group_ids_);
    if (needs_bootstrap_token) {
        operation.decode_input_id = bootstrap_token;
        operation.spec_candidate_ids = std::move(spec_candidate_ids);
    }
    return operation;
}

std::optional<PrefillOperation> Scheduler::schedulePrefillCandidate(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                                    Request* request, std::int32_t token_budget,
                                                                    std::int32_t decode_reserve,
                                                                    std::vector<LoadBackOperation>& load_backs) {
    if (request->Is<fsm::Prefilling>()) {
        if (auto event = schedulePrefill(plan, feedback, request, token_budget, decode_reserve)) {
            return applyEventAndBuildOperation(request, std::move(*event));
        }
        return std::nullopt;
    }
    if (auto event = schedulePrefillFirstChunk(plan, feedback, request, token_budget, decode_reserve)) {
        return applyEventAndBuildOperation(request, std::move(*event), load_backs);
    }
    return std::nullopt;
}

// Who gives way. An incomplete prefill first -- it has produced no output a
// client is reading, and its computed chunks survive as a prefix for the
// retry -- largest first, freeing the most at once. Then decode work, by
// most newly releasable blocks and fewest tokens: the most capacity for the
// least lost work. (On the D role the first rule reaches only a local
// recovery chunk mid-prompt; everything else resident is decoding.)
//
// Exempt: a request whose reserve already covers its whole generation --
// retracting it frees exactly what its readmission must take back, pure
// thrash. Transient obstacles (a forward still out, a PD transfer pin) do
// NOT redirect the choice; the caller waits for the chosen victim to
// quiesce rather than sacrificing a worse-ranked request.
Request* Scheduler::chooseVictim(std::span<Request* const> candidates) const {
    Request* victim = nullptr;
    for (Request* request : candidates) {
        if (request->Is<fsm::Prefilling>() && !request->ReserveCoversGeneration(kRetractionSafeSteps) &&
            (victim == nullptr || request->TokenSize() > victim->TokenSize())) {
            victim = request;
        }
    }
    if (victim != nullptr) {
        return victim;
    }

    std::optional<std::tuple<std::int32_t, std::int32_t, std::string>> victim_rank;
    for (Request* request : candidates) {
        if ((!request->Is<fsm::Decoding>() && !request->Is<fsm::PrefillDone>()) ||
            request->ReserveCoversGeneration(kRetractionSafeSteps)) {
            continue;
        }
        auto rank = std::tuple{-coordinator_.NumNewlyReleasableLcmBlocks(request->BlockTablesRef()),
                               request->TokenSize(), request->Id()};
        if (!victim_rank || rank < *victim_rank) {
            victim = request;
            victim_rank = std::move(rank);
        }
    }
    return victim;
}

void Scheduler::retractVictim(Request& victim, std::vector<WriteBackOperation>& write_back_operations) {
    victim.NoteRetracted();
    // A host cache turns the retraction into an L2 snapshot the readmission
    // loads back; without one the victim re-prefills from scratch. On the
    // fused role that means competing for admission like a newcomer, but a
    // D-role victim always recovers through the ordered local-prefill path
    // -- there is no other way back on that role -- so it stays a
    // readmission even when there is no snapshot to load.
    const bool store_snapshot = config_.HasHostCache();
    const bool recovers_as_readmission = store_snapshot || config_.role == Role::kD;
    if (store_snapshot) {
        fsm::CacheProgress cache_progress = victim.CacheProgress();
        // Only what has actually been computed may be published as a prefix.
        // A decoding request has its whole prompt plus generated tokens bar
        // the one it is about to write; an incomplete prefill has only the
        // chunks it has been through -- taking TokenSize() there would
        // publish pages that were never computed.
        const std::int32_t num_computed_tokens = [&] {
            if (const auto* prefilling = victim.GetIf<fsm::Prefilling>()) {
                return prefilling->window.begin + prefilling->window.size;
            }
            return victim.TokenSize() - config_.decode_input_tokens;
        }();
        const CompletedPrefixPages completed =
            updateCompletedPrefixHashes(victim, cache_progress, num_computed_tokens, coordinator_.PrefixGranularity());
        if (completed.boundary_kind) {
            coordinator_.CacheCompletedBlocks(victim.BlockTablesRef(), cache_progress.prefix_hashes,
                                              cache_progress.access_epoch, completed.first_new_prefix_page,
                                              num_computed_tokens, *completed.boundary_kind);
        }
        coordinator_.QueueCachedBlocksForStore(cache_progress.prefix_hashes);
        if (auto write_back = tier_transfers_.StartPendingStores()) {
            write_back_operations.push_back(std::move(*write_back));
        }
    }
    victim.Apply(fsm::RetractEvent{&coordinator_, next_retraction_epoch_++, recovers_as_readmission,
                                   victim.HasGeneratedOutput()});
    spdlog::info("[Scheduler] retract: released request {} ({} tokens){}", victim.Id(), victim.TokenSize(),
                 store_snapshot ? " with best-effort L2 store" : " for cache capacity");
}

// Fires only when no prefill progressed this round and an admission failed
// for capacity (decode steps do not count as progress -- they release no
// capacity, so a round of pure decode leaves a stalled prefill exactly as
// stuck as an empty one). Retracts victims and RETRIES the blocked admission
// in the same plan build: the freed capacity reaches the request it was
// freed for within this very round, so there is never a free page waiting
// for whoever asks first next round -- which is what used to require a
// cross-round capacity barrier.
//
// The victim's own device blocks return to the pool immediately; the L2
// snapshot copy is ordered on the forward thread's stream BEFORE anything
// this round writes to those pages -- the same stream carries the plan's
// zeroing and fences its forwards (see DeviceHandle.execute) -- so releasing
// them under the still-uncopied snapshot is safe.
//
// A readmission's failed admission never reaches here (its phase records no
// blocker): when the readmission needs a victim, the two simply do not fit
// together, and swapping them is pure thrash -- it waits for a completion
// instead.
//
// A grant that cannot join its round (a local prefill grant beside an
// already-built decode batch, where the role's grammar keeps them apart)
// still retracts one victim: the next round's phase order tries the blocker
// before any other claim on the freed pages.
void Scheduler::maybeRetractForCapacity(AdmissionFeedback& feedback, PlanBuild& build,
                                        std::span<Request* const> candidates,
                                        std::vector<WriteBackOperation>& write_back_operations) {
    Request* blocker = std::exchange(feedback.capacity_blocker, nullptr);
    if (!build.NoPrefillProgress() || blocker == nullptr) {
        return;
    }
    // A load-back mid-flight is writing pages its readmission owns; the
    // victim policy cannot see that write, so no retraction until it lands.
    if (tier_transfers_.HasLoadBacksInFlight()) {
        return;
    }

    while (true) {
        Request* victim = chooseVictim(candidates);
        if (victim == nullptr) {
            return;  // everything resident is exempt; only a completion can free capacity
        }
        if (victim->ResultsInFlight() > 0 || pd_transfer_pins_.contains(victim->Id())) {
            // The victim is chosen but not quiescent: a forward's KV write or
            // a PD transfer is still landing on its pages. Wait for it
            // rather than sacrificing a worse-ranked request.
            return;
        }
        retractVictim(*victim, write_back_operations);

        if (blocker == victim) {
            // The victim blocked on its own next page; it comes back through
            // the readmission phase. Its freed capacity goes to the first
            // waiting prompt instead -- granting it back to the victim's own
            // readmission is the loop the grant exists to break.
            const auto waiting = std::ranges::find_if(
                candidates, [victim](Request* request) { return request != victim && admitsLikeNewPrompt(*request); });
            if (waiting == candidates.end()) {
                return;
            }
            blocker = *waiting;
        }
        if (build.Full(config_.max_batch_size)) {
            return;
        }

        const bool blocked_on_decode = blocker->Is<fsm::Decoding>() || blocker->Is<fsm::PrefillDone>();
        const bool remote_grant = config_.role == Role::kD && !blocked_on_decode && !blocker->Is<fsm::Prefilling>();
        if (!blocked_on_decode && !remote_grant && build.pushed_decode &&
            !(config_.role == Role::kFused && config_.enable_mixed_prefill_decode)) {
            // A local prefill grant cannot join an already-built decode
            // batch (a D-role recovery chunk runs alone; fused mixes only in
            // mixed mode). The victim is still retracted: the next round's
            // phase order tries the blocker before any other claim.
            return;
        }

        feedback.admission_failed = false;
        if (blocked_on_decode) {
            if (auto event = scheduleDecode(build.plan, feedback, blocker)) {
                pushOperation(build, *blocker, applyEventAndBuildOperation(blocker, std::move(*event)));
                blocker->TrackScheduledForward();
                return;
            }
        } else {
            // A D-role prompt admission is remote by construction: the whole
            // prompt admits at once and the peer's prefill rides
            // plan.remote_prefill beside whatever batch this round built.
            // Everything else is local prefill work joining the model batch.
            const std::int32_t budget = remote_grant ? blocker->PrefillSize() : build.token_budget;
            if (auto operation = schedulePrefillCandidate(build.plan, feedback, blocker, budget,
                                                          config_.decode_input_tokens, build.load_backs)) {
                if (remote_grant) {
                    build.scheduled.insert(blocker);
                    build.remote_prefill.emplace_back(std::move(*operation));
                } else {
                    pushOperation(build, *blocker, std::move(*operation));
                    blocker->TrackScheduledForward();
                }
                // A D-role LOCAL recovery grant must not pin: no PD ACK ever
                // arrives to erase it (its lifetime is the L2 load ticket).
                if (config_.enable_pd_cache && (remote_grant || config_.role != Role::kD)) {
                    pd_transfer_pins_.insert(blocker->Id());
                }
                return;
            }
        }
        if (!feedback.admission_failed) {
            return;  // not a capacity failure (zero-token alignment); stop retracting
        }
    }
}

Request* Scheduler::nextReadmission(std::span<Request* const> candidates) {
    const auto rank = [](const fsm::Retracted& retracted) {
        return std::pair{!retracted.ResumesGeneration(), retracted.RetractionEpoch()};
    };
    Request* next = nullptr;
    const fsm::Retracted* next_state = nullptr;
    for (Request* request : candidates) {
        const auto* retracted = request->GetIf<fsm::Retracted>();
        if (retracted == nullptr || !retracted->HasRecoverableSnapshot()) {
            continue;
        }
        if (next_state == nullptr || rank(*retracted) < rank(*next_state)) {
            next = request;
            next_state = retracted;
        }
    }
    return next;
}

// Local prefill phases shared by the P and fused grammars: the readmission
// first (fused only -- it resumes a client's generation and precedes fresh
// work), then resident chunks (they hold pages), then new prompts. One loop
// per tier so the order is visible.
//
// Head-of-line: an incomplete prefill that scheduled stops further prefill
// work (nothing may consume the capacity it still needs), and a resident
// chunk that FAILED admission stops it too -- admitting behind it would
// strand it. A readmission that fails admission waits without becoming the
// capacity blocker (retracting a victim for it is pure thrash -- the two
// simply do not fit together), but it does seal new-prompt admission:
// a newcomer taking the pages it is waiting for would starve it.
void Scheduler::scheduleLocalPrefillWork(AdmissionFeedback& feedback, PlanBuild& build,
                                         std::span<Request* const> candidates, Request* readmission,
                                         std::int32_t decode_reserve) {
    bool new_prompts_sealed = false;
    if (readmission != nullptr) {
        feedback.admission_failed = false;
        if (auto operation = schedulePrefillCandidate(build.plan, feedback, readmission, build.token_budget,
                                                      decode_reserve, build.load_backs)) {
            pushOperation(build, *readmission, std::move(*operation));
            readmission->TrackScheduledForward();
            if (holdsHeadOfLine(*readmission)) {
                return;
            }
        } else {
            new_prompts_sealed = feedback.admission_failed;
        }
    }
    for (const bool resident : {true, false}) {
        if (!resident && new_prompts_sealed) {
            return;
        }
        for (Request* request : candidates) {
            if (build.Full(config_.max_batch_size)) {
                return;
            }
            if ((resident ? !request->Is<fsm::Prefilling>() : !admitsLikeNewPrompt(*request)) ||
                build.Scheduled(*request)) {
                continue;
            }
            feedback.admission_failed = false;
            if (auto operation = schedulePrefillCandidate(build.plan, feedback, request, build.token_budget,
                                                          decode_reserve, build.load_backs)) {
                pushOperation(build, *request, std::move(*operation));
                if (config_.enable_pd_cache) {
                    // Pages stay pinned until the PD transfer completes.
                    pd_transfer_pins_.insert(request->Id());
                }
                request->TrackScheduledForward();
                if (holdsHeadOfLine(*request)) {
                    return;
                }
            } else if (feedback.admission_failed) {
                if (feedback.capacity_blocker == nullptr) {
                    feedback.capacity_blocker = request;
                }
                if (resident) {
                    return;
                }
            }
        }
    }
}

// The decode batch shared by the D and fused grammars: every PrefillDone
// (its first decode) and Decoding candidate. The budget guard protects the
// mamba state reserve of a prefill scheduled beside them in mixed mode; on
// the D role decodes consume no budget, so it never binds there.
void Scheduler::scheduleDecodeBatch(AdmissionFeedback& feedback, PlanBuild& build,
                                    std::span<Request* const> candidates) {
    for (Request* request : candidates) {
        if (build.Full(config_.max_batch_size) ||
            build.token_budget < build.state_prefill_reserve + config_.decode_input_tokens) {
            return;
        }
        if ((!request->Is<fsm::PrefillDone>() && !request->Is<fsm::Decoding>()) || build.Scheduled(*request)) {
            continue;
        }
        feedback.admission_failed = false;
        if (auto event = scheduleDecode(build.plan, feedback, request)) {
            pushOperation(build, *request, applyEventAndBuildOperation(request, std::move(*event)));
            request->TrackScheduledForward();
        } else if (feedback.admission_failed && feedback.capacity_blocker == nullptr) {
            feedback.capacity_blocker = request;
        }
    }
}

// P role: prefill worker. Completed prompts leave on plan.remote_decode
// first -- their KV pages stay pinned until the transfer finishes, so
// releasing them outranks feeding more prompt work -- then the prefill
// phases run with no decode reserve (this role never decodes locally). No
// retraction either: a P node's pressure valve is the transfer itself, so
// this grammar never calls maybeRetractForCapacity and nothing here is ever
// readmitted.
void Scheduler::buildPrefillWorkerPlan(AdmissionFeedback& feedback, PlanBuild& build,
                                       std::span<Request* const> candidates) {
    // The prompt decodes on the peer node: its KV goes out on the plan's own
    // stream, occupying no token budget and no batch slot. A prompt still
    // waiting for its final chunk's result is in PrefillAwaitingResult, not
    // PrefillDone, so the bootstrap token the transfer needs is real. No
    // TrackScheduledForward: the counter guards this engine's own pages
    // against its own forwards, and the peer's decode writes none of them;
    // its fence is the PD ACK.
    for (Request* request : candidates) {
        if (request->Is<fsm::PrefillDone>()) {
            if (auto event = scheduleDecode(build.plan, feedback, request)) {
                build.scheduled.insert(request);
                build.remote_decode.emplace_back(applyEventAndBuildOperation(request, std::move(*event)));
            }
        }
    }

    scheduleLocalPrefillWork(feedback, build, candidates, /*readmission=*/nullptr, /*decode_reserve=*/0);
}

// D role: decode worker. Local recovery work runs alone in its batch;
// otherwise the round is a decode batch, beside which at most ONE remote
// admission rides plan.remote_prefill. Retraction picks decode victims and
// retries the blocked admission in the same round.
void Scheduler::buildDecodeWorkerPlan(AdmissionFeedback& feedback, PlanBuild& build,
                                      std::span<Request* const> candidates,
                                      std::vector<WriteBackOperation>& write_back_operations) {
    // Phase 1: local recovery, alone in its batch -- a resident chunk if one
    // is mid-prompt (always recovery here: a remote prompt is
    // RemotePrefilling, which schedules nothing until the peer is done),
    // else the one readmission this round may start. No PD pin: local
    // recovery has no PD ACK; its Host/Device lifetime is owned by the L2
    // load ticket. A readmission that fails admission simply waits -- it
    // never triggers retraction (swapping it with a victim is pure thrash)
    // and never stalls the decodes below.
    const auto resident = std::ranges::find_if(candidates, &Request::Is<fsm::Prefilling>);
    Request* recovery = resident != candidates.end() ? *resident : nextReadmission(candidates);
    if (recovery != nullptr) {
        feedback.admission_failed = false;
        if (auto operation = schedulePrefillCandidate(build.plan, feedback, recovery, build.token_budget,
                                                      config_.decode_input_tokens, build.load_backs)) {
            pushOperation(build, *recovery, std::move(*operation));
            recovery->TrackScheduledForward();
            return;  // recovery runs alone
        }
        if (recovery->Is<fsm::Prefilling>() && feedback.admission_failed && feedback.capacity_blocker == nullptr) {
            feedback.capacity_blocker = recovery;
        }
    }

    // Phase 2: the decode batch. Completed prefills' first decodes go ahead
    // of the running ones; neither consumes token budget on this role.
    scheduleDecodeBatch(feedback, build, candidates);

    // Phase 3: at most one remote admission -- the whole prompt reserves at
    // once, so admitting a queue's worth in one round would drain the pool
    // before any of their KV arrives. It rides plan.remote_prefill beside
    // the decode batch: no token budget, no batch slot. No
    // TrackScheduledForward: the peer runs this prefill, so no forward of
    // this engine's is out against the pages; the PD pin holds them.
    for (Request* request : candidates) {
        if (!admitsLikeNewPrompt(*request)) {
            continue;
        }
        feedback.admission_failed = false;
        if (auto operation = schedulePrefillCandidate(build.plan, feedback, request, request->PrefillSize(),
                                                      config_.decode_input_tokens, build.load_backs)) {
            build.scheduled.insert(request);
            build.remote_prefill.emplace_back(std::move(*operation));
            if (config_.enable_pd_cache) {
                pd_transfer_pins_.insert(request->Id());
            }
            break;
        }
        if (feedback.admission_failed && feedback.capacity_blocker == nullptr) {
            feedback.capacity_blocker = request;
        }
    }

    maybeRetractForCapacity(feedback, build, candidates, write_back_operations);
}

// Fused role: one engine does everything locally. In mixed mode resident
// decodes take their token budget first -- a client is streaming them, and a
// long prefill chunk must not starve them -- leaving one state-checkpoint
// page of budget for a pending local mamba prefill; the prefill phases spend
// the rest. Outside mixed mode prefill work runs alone, and decodes get a
// round only when no prefill scheduled. Recovery readmission is live when a
// host cache gives victims a way back.
void Scheduler::buildFusedPlan(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates,
                               std::vector<WriteBackOperation>& write_back_operations) {
    Request* readmission = nextReadmission(candidates);
    if (config_.enable_mixed_prefill_decode) {
        const bool has_local_prefill =
            readmission != nullptr || std::ranges::any_of(candidates, [](const Request* request) {
                return request->Is<fsm::Prefilling>() || admitsLikeNewPrompt(*request);
            });
        build.state_prefill_reserve =
            coordinator_.HasMambaStateGroup() && has_local_prefill ? coordinator_.PrefixGranularity() : 0;
        scheduleDecodeBatch(feedback, build, candidates);
    }

    scheduleLocalPrefillWork(feedback, build, candidates, readmission, config_.decode_input_tokens);

    if (!config_.enable_mixed_prefill_decode && !build.pushed_prefill) {
        scheduleDecodeBatch(feedback, build, candidates);
    }

    maybeRetractForCapacity(feedback, build, candidates, write_back_operations);
}

std::pair<std::vector<ForwardOperation>, std::vector<LoadBackOperation>> Scheduler::buildForwardOperations(
    ExecutionPlan& plan, std::vector<Request*> candidates, std::vector<WriteBackOperation>& write_back_operations) {
    // The candidates arrive in submission order (requests_ is the FIFO),
    // identical on every rank -- so within a phase, older requests win.
    AdmissionFeedback feedback;
    PlanBuild build{plan};
    build.token_budget = config_.max_scheduled_tokens;
    switch (config_.role) {
        case Role::kP:
            buildPrefillWorkerPlan(feedback, build, candidates);
            break;
        case Role::kD:
            buildDecodeWorkerPlan(feedback, build, candidates, write_back_operations);
            break;
        case Role::kFused:
            buildFusedPlan(feedback, build, candidates, write_back_operations);
            break;
    }

    if (!build.remote_decode.empty()) {
        plan.remote_decode.emplace(std::move(build.remote_decode));
    }
    if (!build.remote_prefill.empty()) {
        plan.remote_prefill.emplace(std::move(build.remote_prefill));
    }
    return {std::move(build.operations), std::move(build.load_backs)};
}

}  // namespace tokenspeed
