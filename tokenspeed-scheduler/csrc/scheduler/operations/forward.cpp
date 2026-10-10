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
#include <cstdlib>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "cache/core/cache_types.h"
#include "cache/tier/transfer.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "scheduler/operations/cache.h"
#include "scheduler/operations/forward.h"
#include "scheduler/operations/group_demands.h"
#include "scheduler/operations/prefill_chunk.h"
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

// Landed images the restore phase tries per round before giving up: a large
// image that does not fit must not seal the queue while a smaller one behind
// it would, but every failed attempt is an admission-planner pass, so the
// scan is bounded.
constexpr std::int32_t kMaxRestoreAttemptsPerRound = 4;

// An incomplete prefill keeps the head of line: admission reserved only this
// chunk, and a later candidate could strand it by consuming the capacity it
// needs to finish.
bool holdsHeadOfLine(const Request& request) {
    return request.Is<fsm::Prefilling>();
}

template <typename Operation>
void fillBlockTables(Operation& operation, Request& request, const CacheCoordinator& coordinator,
                     std::span<const std::string> group_ids) {
    operation.block_tables = BuildBlockTables(coordinator, request.BlockTablesRef(), group_ids);
}

void classifyCompletedStateBoundaries(CompletedPages& completed, std::int32_t endpoint_tokens,
                                      std::int32_t prefix_granularity) {
    if (completed.boundary_kind != CacheBoundaryKind::kChunk) {
        return;
    }
    const std::int32_t endpoint_boundary = endpoint_tokens / prefix_granularity * prefix_granularity;
    if (endpoint_boundary > 0 && std::ranges::find(completed.materialized_state_boundaries, endpoint_boundary) !=
                                     completed.materialized_state_boundaries.end()) {
        // Classify the last aligned prompt checkpoint without upgrading history.
        completed.state_boundary_kind = CacheBoundaryKind::kEndpoint;
    }
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

// What a scheduled prefill window proves about state: local prefill
// materializes its latest internal aligned checkpoint; a remote landing
// brings only the endpoint state, a checkpoint when the window ends aligned.
void recordPrefillStateCheckpoint(fsm::CacheProgress& cache_progress, fsm::PrefillSource source,
                                  std::int32_t after_tokens, std::int32_t prefix_granularity) {
    const std::int32_t aligned = after_tokens / prefix_granularity * prefix_granularity;
    if (source == fsm::PrefillSource::kLocal || aligned == after_tokens) {
        cache_progress.RecordMaterializedStateBoundary(aligned, prefix_granularity);
    }
}

// Hashes the prefix pages that num_computed_tokens has filled since the
// previous admission and states the request's progress for the coordinator.
// The returned spans view cache_progress, which must outlive their use.
RequestProgress advanceRequestProgress(Request& request, fsm::CacheProgress& cache_progress,
                                       std::int32_t num_computed_tokens, std::int32_t prefix_granularity,
                                       bool stream_completed_to_host) {
    const std::int32_t first_new_prefix_page = static_cast<std::int32_t>(cache_progress.prefix_hashes.size());
    const std::int32_t filled_prefix_pages = num_computed_tokens / prefix_granularity;
    if (filled_prefix_pages > first_new_prefix_page) {
        appendCompletedPrefixHashes(cache_progress.prefix_hashes, request.FullPrefixPages(false), filled_prefix_pages);
    }
    RequestProgress progress{.num_computed_tokens = num_computed_tokens};
    if (first_new_prefix_page < static_cast<std::int32_t>(cache_progress.prefix_hashes.size())) {
        progress.completed_pages = CompletedPages{
            .prefix_hashes = cache_progress.prefix_hashes,
            .first_new_prefix_page = first_new_prefix_page,
            .boundary_kind = consumeCompletedBoundaryKind(cache_progress, num_computed_tokens, request.PrefillSize()),
            .stream_completed_to_host = stream_completed_to_host,
            .materialized_state_boundaries = cache_progress.materialized_state_boundaries,
        };
        classifyCompletedStateBoundaries(*progress.completed_pages, request.PrefillSize(), prefix_granularity);
    }
    return progress;
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
    operation.extend_replay_len = info.replay_len;
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
    // Only a first admission probes: a readmission copies the request's own
    // image back and matches nothing.
    _assert(request->Is<fsm::Submitted>(), "the admission probe is for a Submitted request");
    const auto probe = [this](std::span<const std::string> hashes) {
        if (config_.role == Role::kD) {
            return coordinator_.ProbeDecodeDevicePrefix(hashes);
        }
        return coordinator_.ProbePrefix(hashes);
    };
    const std::int32_t prefix_granularity = coordinator_.PrefixGranularity();
    // The final prompt token is always recomputed to produce logits. Some
    // consumers additionally require a larger prompt tail (for example, to
    // rebuild request-persistent state that is not stored in the KV cache).
    // Limit the probe itself so excluded hit pages are never claimed: admission
    // will allocate private writable pages for the replayed suffix. A request
    // may tighten the bound further (RequestSpec::max_cached_prefix_tokens) so
    // the positions it needs logits for are recomputed rather than matched.
    const std::int32_t replay_tokens = std::max(config_.prefix_replay_tokens, 1);
    const std::int32_t max_cacheable_tokens =
        std::max(std::min(request->PrefillSize() - replay_tokens, request->MaxCachedPrefixTokens()), 0);
    const std::int32_t probe_prefix_pages = max_cacheable_tokens / prefix_granularity;
    const std::int32_t candidate_prefix_pages = std::max((request->PrefillSize() - 1) / prefix_granularity, 0);
    std::vector<std::span<const std::int32_t>> prefix_pages = request->FullPrefixPages(false);
    prefix_pages.resize(std::min(prefix_pages.size(), static_cast<std::size_t>(candidate_prefix_pages)));
    std::vector<std::string> hashes = ComputePrefixHashes(prefix_pages, "");
    const auto probe_hashes = std::span<const std::string>(hashes).first(
        std::min(hashes.size(), static_cast<std::size_t>(probe_prefix_pages)));

    AdmissionMatch match;
    match.candidate_prefix_hashes = hashes;
    if (config_.disable_prefix_cache) {
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
                                                                  const RequestProgress& progress,
                                                                  std::optional<std::uint64_t> request_access_epoch) {
    std::optional<CacheCoordinator::AdmissionResult> result =
        coordinator_.Admit(std::move(prefix), demands, progress, request_access_epoch);
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

bool Scheduler::admitWithKvEventTracking(ExecutionPlan& plan, AdmissionFeedback& feedback, Request& request,
                                         const fsm::CacheProgress& cache_progress, std::span<const GroupDemand> demands,
                                         const RequestProgress& progress) {
    const std::int32_t first_new_prefix_page = progress.completed_pages
                                                   ? progress.completed_pages->first_new_prefix_page
                                                   : static_cast<std::int32_t>(cache_progress.prefix_hashes.size());
    registerKvEventPrefixPages(request, cache_progress.prefix_hashes, first_new_prefix_page);
    return admit(plan, feedback, coordinator_.ProbePrefix({}), demands, progress, cache_progress.access_epoch)
        .has_value();
}

Scheduler::FirstChunkOutcome Scheduler::schedulePrefillFirstChunk(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                                  Request* request, std::int32_t remaining,
                                                                  std::int32_t decode_input_tokens,
                                                                  std::vector<PrefetchOperation>& prefetches) {
    AdmissionMatch match = matchPrefixAtAdmission(request);
    // L3 objects beyond the Host hit are fetched into Host BEFORE admission:
    // the request waits (holding only the Host blocks being filled, no Device
    // pages, no slot) and is admitted as an ordinary Host hit once they
    // landed, so nothing an admission loads can miss. Below the threshold
    // the pages are simply computed. The D role probes the Device alone and
    // never gets here with a storage tier.
    if (config_.enable_l3_storage && config_.role != Role::kD) {
        if (std::optional<CacheCoordinator::PrefetchPlan> prefetch =
                coordinator_.PlanPrefetch(match.probe, config_.l3_prefetch_min_pages)) {
            std::vector<CacheBlockRef> host_blocks;
            host_blocks.reserve(prefetch->rows.size());
            for (const CacheCoordinator::PrefetchRow& row : prefetch->rows) {
                host_blocks.push_back(row.host_block);
            }
            PrefetchOperation op = tier_transfers_.StartPrefetch(request->Id(), std::move(*prefetch));
            spdlog::info("[Scheduler] prefetch: request {} waits for {} L3 page(s) beyond its Host hit", request->Id(),
                         op.num_pages);
            request->Apply(fsm::SchedulePrefetchEvent{std::move(host_blocks), op.op_id});
            prefetches.push_back(std::move(op));
            return FirstChunkOutcome{.prefetching = true};
        }
    }
    if (req_pool_allocator_.AvailableSlots() == 0) {
        return FirstChunkOutcome{};
    }

    // The D role's prompts are the peer's work; every other first chunk is local.
    const fsm::PrefillSource source =
        config_.role == Role::kD ? fsm::PrefillSource::kRemote : fsm::PrefillSource::kLocal;
    const std::int32_t prefix_granularity = coordinator_.PrefixGranularity();
    registerKvEventPrefixPages(*request, match.candidate_prefix_hashes, 0);

    _assert(match.probe.host.num_common_tokens % prefix_granularity == 0, "a Host hit ends on a prefix boundary");
    const std::int32_t hit_tokens = std::max(match.probe.device.num_common_tokens, match.probe.host.num_common_tokens);
    const std::int32_t promotion_boundary_tokens = coordinator_.PromotionBoundaryTokens(match.probe);
    _assert(promotion_boundary_tokens == 0 ||
                (promotion_boundary_tokens % prefix_granularity == 0 && promotion_boundary_tokens > hit_tokens &&
                 promotion_boundary_tokens < request->PrefillSize()),
            "promotion boundary must be page-aligned and inside the unmatched prompt");

    const std::int32_t unscheduled = request->PrefillSize() - hit_tokens;
    const std::int32_t tokens_this_round = PrefillChunkTokens(coordinator_, hit_tokens, /*resumes_hit=*/true,
                                                              unscheduled, remaining, promotion_boundary_tokens);
    if (tokens_this_round == 0) {
        return FirstChunkOutcome{};
    }
    const std::int32_t after_tokens = hit_tokens + tokens_this_round;
    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t decode_reserve = completes_prefill ? decode_input_tokens : 0;
    // Every admission on a decoding role (D, Fused) secures real headroom
    // before the prefill starts: the rest of the prompt plus decode room
    // that starts at one safe-step window and grows with each retraction
    // (Request::AdmissionHeadroom). Pages only -- the request still computes
    // one chunk per round, because the chunk size is a forward-pass limit
    // rather than a capacity one. The P role is exempt: it never decodes
    // locally and never retracts, so there is no decode room to prepay.
    const std::int32_t headroom = config_.role == Role::kP ? 0 : request->AdmissionHeadroom(kRetractionSafeSteps);
    const PrefillReserve reserve{
        .decode_input_tokens = decode_input_tokens,
        .completes_prefill = completes_prefill,
        .prompt_headroom_tokens = headroom > 0 ? unscheduled - tokens_this_round + headroom : 0,
        // A remote landing always finishes shaping; the P role needs no local decode growth.
        .reserve_snapshot_state_growth =
            config_.role != Role::kP && (source == fsm::PrefillSource::kRemote || completes_prefill),
    };
    std::vector<BlockTable> tables(static_cast<std::size_t>(coordinator_.NumGroups()));
    std::vector<GroupDemand> demands = MakeGroupDemands(tables, GroupDemand{.extent = DenseGrowth{tokens_this_round}});
    ReservePrefillDemands(demands, config_.cache_groups, reserve);
    if (source == fsm::PrefillSource::kLocal) {
        MakeSnapshotStatePrefillSparse(demands, config_.cache_groups, coordinator_, hit_tokens, after_tokens);
    }

    if (source == fsm::PrefillSource::kRemote) {
        for (std::size_t i = 0; i < demands.size(); ++i) {
            const CacheGroupConfig& group = config_.cache_groups[i];
            const std::int32_t block_granularity = coordinator_.GroupBlockGranularity(i);
            if (group.transfer_policy == CacheTransferPolicy::LatestSnapshot) {
                // The peer lands only the endpoint snapshot, in slot (PrefillSize-1)/g.
                demands[i].extent = SparseSuffix{
                    .extent_tokens = request->PrefillSize(),
                    .first_block = (request->PrefillSize() - 1) / block_granularity,
                };
            } else if (group.Kind() == AttnKind::kSlidingWindow) {
                const std::int32_t retained_begin =
                    std::max(0, request->PrefillSize() - *group.sliding_window_tokens + 1);
                demands[i].extent = SparseSuffix{
                    .extent_tokens = request->PrefillSize(),
                    .first_block = std::max(hit_tokens / block_granularity, retained_begin / block_granularity),
                };
            }
        }
    }
    // First admission has computed nothing to publish or reclaim.
    std::optional<CacheCoordinator::AdmissionResult> admission =
        coordinator_.Admit(std::move(match.probe), demands, RequestProgress{}, /*request_access_epoch=*/std::nullopt);
    if (!admission) {
        feedback.admission_failed = true;
        return FirstChunkOutcome{};
    }
    // Every Host hit is an entry the admission acquired in the same step, so
    // the admitted prefix is the probed one.
    _assert(std::max(admission->device_prefix_tokens, admission->host_prefix_tokens) == hit_tokens,
            "an admission claims exactly the prefix it probed");
    _assert(admission->promotion_boundary_tokens == promotion_boundary_tokens,
            "promotion boundary changed between probe and admission");
    _assert(admission->new_page_ids.size() == cache_group_ids_.size(),
            "admission fresh-page groups must match scheduler config");
    for (std::size_t i = 0; i < admission->new_page_ids.size(); ++i) {
        auto& page_ids = admission->new_page_ids[i];
        auto& pending = plan.pages_to_zero[cache_group_ids_[i]];
        pending.insert(pending.end(), page_ids.begin(), page_ids.end());
    }

    if (!match.extension_hashes.empty()) {
        // The Host-warm H2D destinations are filled on Host already; publish
        // them on the Device now (the layer-wise load makes the bytes
        // available before any forward reads them).
        const std::int32_t first_extension_slot = admission->device_prefix_tokens / prefix_granularity;
        for (std::size_t i = 0; i < match.extension_hashes.size(); ++i) {
            coordinator_.CacheFullBlocks(tables, std::span<const std::string>(match.extension_hashes).subspan(i, 1),
                                         admission->access_epoch, first_extension_slot + static_cast<std::int32_t>(i),
                                         CacheBoundaryKind::kChunk);
        }
    }
    fsm::CacheProgress cache_progress{
        .prefix_hashes = std::move(match.prefix_hashes),
        .access_epoch = admission->access_epoch,
        .promotion_boundary_tokens = admission->promotion_boundary_tokens,
    };
    recordPrefillStateCheckpoint(cache_progress, source, hit_tokens + tokens_this_round, prefix_granularity);
    return FirstChunkOutcome{.event = fsm::SchedulePrefillFirstChunkEvent{
                                 tokens_this_round,
                                 decode_reserve,
                                 &req_pool_allocator_,
                                 source,
                                 &coordinator_,
                                 std::move(tables),
                                 hit_tokens,
                                 std::move(cache_progress),
                                 std::move(admission->load_pairs),
                                 // The P role holds a completed prompt until its result lands: the
                                 // remote decode that hands it off carries the bootstrap token.
                                 config_.role == Role::kP,
                             }};
}

std::optional<fsm::SchedulePrefillEvent> Scheduler::schedulePrefill(
    ExecutionPlan& plan, AdmissionFeedback& feedback, Request* request, std::int32_t remaining,
    std::int32_t reserve_num_tokens_in_next_schedule_event) {
    const std::int32_t unscheduled = request->UnscheduledPrefillSize();
    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    fsm::CacheProgress cache_progress = request->CacheProgress();
    const std::int32_t prefill_tokens = PrefillChunkTokens(coordinator_, first_pos, /*resumes_hit=*/false, unscheduled,
                                                           remaining, cache_progress.promotion_boundary_tokens);
    if (prefill_tokens == 0) {
        return std::nullopt;
    }

    const std::int32_t after_tokens = first_pos + prefill_tokens;
    const bool completes_prefill = prefill_tokens == unscheduled;
    const std::int32_t decode_reserve = completes_prefill ? reserve_num_tokens_in_next_schedule_event : 0;
    // The prompt headroom was prepaid at first-chunk admission.
    const PrefillReserve reserve{
        .decode_input_tokens = reserve_num_tokens_in_next_schedule_event,
        .completes_prefill = completes_prefill,
        .prompt_headroom_tokens = 0,
        .reserve_snapshot_state_growth = config_.role != Role::kP && completes_prefill,
    };
    const RequestProgress progress =
        advanceRequestProgress(*request, cache_progress, request->NumComputedTokens(), coordinator_.PrefixGranularity(),
                               config_.StreamsDeviceCacheToHost());

    std::vector<BlockTable>& tables = request->BlockTablesRef();
    std::vector<GroupDemand> demands = MakeGroupDemands(tables, GroupDemand{.extent = DenseGrowth{prefill_tokens}});
    ReservePrefillDemands(demands, config_.cache_groups, reserve);
    MakeSnapshotStatePrefillSparse(demands, config_.cache_groups, coordinator_, first_pos, after_tokens);
    if (!admitWithKvEventTracking(plan, feedback, *request, cache_progress, demands, progress)) {
        return std::nullopt;
    }

    cache_progress.DiscardHashedStateBoundaries(coordinator_.PrefixGranularity());
    recordPrefillStateCheckpoint(cache_progress, fsm::PrefillSource::kLocal, after_tokens,
                                 coordinator_.PrefixGranularity());
    request->CacheProgressRef() = std::move(cache_progress);
    return fsm::SchedulePrefillEvent{prefill_tokens, decode_reserve, config_.role == Role::kP};
}

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                                  Request* request) {
    std::vector<BlockTable>& tables = request->BlockTablesRef();
    fsm::CacheProgress cache_progress = request->CacheProgress();
    const std::int32_t num_computed_tokens = request->NumComputedTokens();
    std::int32_t reserve_tokens = request->ReserveNumTokensInNextScheduleEvent();
    if (config_.decode_input_tokens > 1) {
        const std::int32_t decode_width = std::max(config_.decode_input_tokens, reserve_tokens);
        const std::int32_t pending_decode_tokens =
            request->Is<fsm::Decoding>()
                ? std::min(request->ResultsInFlight(), config_.overlap_schedule_depth) * config_.decode_input_tokens
                : 0;
        // All groups consume the same logical token extent, including sparse tables.
        // Reuse reserved speculative slots after a prefill interrupts decode instead
        // of charging another verify span on every restart.
        const std::int32_t reserved_end =
            tables.front().NumBlocks() * coordinator_.GroupBlockGranularity(0) - tables.front().AvailableTokens();
        reserve_tokens = std::max(num_computed_tokens + pending_decode_tokens + decode_width - reserved_end, 0);
    }
    const RequestProgress progress =
        advanceRequestProgress(*request, cache_progress, num_computed_tokens, coordinator_.PrefixGranularity(),
                               config_.StreamsDeviceCacheToHost() && request->Is<fsm::PrefillDone>());

    if (!progress.completed_pages &&
        canConsumeReservedTokensInPlace(coordinator_, tables, reserve_tokens, num_computed_tokens)) {
        coordinator_.ConsumeReservedTokens(tables, reserve_tokens);
    } else {
        std::vector<GroupDemand> demands = MakeGroupDemands(tables, GroupDemand{.extent = DenseGrowth{reserve_tokens}});
        if (!admitWithKvEventTracking(plan, feedback, *request, cache_progress, demands, progress)) {
            return std::nullopt;
        }
    }

    cache_progress.DiscardHashedStateBoundaries(coordinator_.PrefixGranularity());
    request->CacheProgressRef() = std::move(cache_progress);
    return fsm::ScheduleDecodeEvent{config_.decode_input_tokens};
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
    // RemotePrefillDoneEvent), the P side's remote decode (the peer sends it
    // on as the bootstrap token, and the P grammar holds the op until the
    // result lands), and the first decode after a restore on any role (the
    // request sits in a new slot; the capture its last forward left belongs
    // to the slot it was retracted from). Otherwise fused stays -1 on
    // purpose: overlap plans the decode BEFORE the result lands, and the
    // device fills the input from its in-flight capture. A restored request
    // is quiescent at its first decode, so LastToken() is the landed input.
    const bool needs_bootstrap_token = request->Is<fsm::PrefillDone>() && config_.role != Role::kFused;
    const bool needs_explicit_token = needs_bootstrap_token || request->ResumedByRestore();
    const std::int32_t explicit_token = needs_explicit_token ? request->LastToken() : -1;
    std::vector<std::int32_t> spec_candidate_ids =
        config_.role == Role::kP && needs_bootstrap_token ? request->TakeSpecCandidates() : std::vector<std::int32_t>{};
    // The event builds a fresh Decoding, so the restore marker is consumed here.
    DecodeOperation operation =
        applyDecodeEvent(*request, std::move(event), config_.decode_input_tokens, coordinator_, cache_group_ids_);
    if (needs_explicit_token) {
        operation.decode_input_id = explicit_token;
        operation.spec_candidate_ids = std::move(spec_candidate_ids);
    }
    return operation;
}

std::optional<Scheduler::PrefillAdmission> Scheduler::schedulePrefillCandidate(
    ExecutionPlan& plan, AdmissionFeedback& feedback, Request* request, std::int32_t token_budget,
    std::int32_t decode_reserve, std::vector<LoadBackOperation>& load_backs,
    std::vector<PrefetchOperation>& prefetches) {
    if (request->Is<fsm::Prefilling>()) {
        if (auto event = schedulePrefill(plan, feedback, request, token_budget, decode_reserve)) {
            return PrefillAdmission{.operation = applyEventAndBuildOperation(request, std::move(*event))};
        }
        return std::nullopt;
    }
    FirstChunkOutcome outcome =
        schedulePrefillFirstChunk(plan, feedback, request, token_budget, decode_reserve, prefetches);
    if (outcome.prefetching) {
        return PrefillAdmission{};
    }
    if (outcome.event) {
        return PrefillAdmission{.operation =
                                    applyEventAndBuildOperation(request, std::move(*outcome.event), load_backs)};
    }
    return std::nullopt;
}

// Who gives way. Neither tier loses work any more -- a victim resumes exactly
// where it stopped -- so the cost of a retraction is only the image bytes
// (proportional to the pages held) and the client-visible interruption. An
// incomplete prefill first: no client is streaming it yet, and the mid-prompt
// prefill is usually the request that blocked on its own next page, so
// retracting it and granting its pages to a prompt that can finish is the
// shortest path out of head-of-line -- largest first, freeing the most at
// once. Then decode work, by most newly releasable blocks and fewest
// generated tokens: the needed pages with the fewest victims disturb the
// fewest clients, and among equal frees the client that has streamed least
// is interrupted. (On the D role everything resident is decoding.)
//
// Only candidates whose image fits the host budgets (imageFits) are ranked.
// When none does, the host side is out of room and the last resort is to
// abort a resident instead of imaging one: the newest retractable resident
// (the least work lost) is returned with image_fits = false. Without a
// snapshot pool nothing ever fits, so a capacity block there always ends in
// that abort rather than in a wait that could deadlock once every resident
// needs a page.
//
// Exempt in both tiers: a request whose reserve already covers its whole
// generation -- retracting it frees exactly what its readmission must take
// back, pure thrash -- and it is not aborted either: it completes on its own
// reserve and frees its pages then. Excluded by state: Retracted, Restoring,
// RemotePrefilling. Transient obstacles (a forward still out, a PD transfer
// pin) do NOT redirect the choice; the caller waits for the chosen victim to
// quiesce rather than sacrificing a worse-ranked request.
Scheduler::VictimChoice Scheduler::chooseVictim(std::span<Request* const> candidates) const {
    const auto retractable = [](const Request& request) {
        return request.IsAnyOf<fsm::Prefilling, fsm::PrefillDone, fsm::Decoding>() &&
               !request.ReserveCoversGeneration(kRetractionSafeSteps);
    };
    Request* victim = nullptr;
    for (Request* request : candidates) {
        if (request->Is<fsm::Prefilling>() && retractable(*request) &&
            (victim == nullptr || request->TokenSize() > victim->TokenSize()) && imageFits(*request)) {
            victim = request;
        }
    }
    if (victim != nullptr) {
        return VictimChoice{.victim = victim, .image_fits = true};
    }

    std::optional<std::tuple<std::int32_t, std::int32_t, std::string>> victim_rank;
    Request* newest = nullptr;
    for (Request* request : candidates) {
        if (!retractable(*request)) {
            continue;
        }
        newest = request;  // candidates arrive in submission order
        if (!request->IsAnyOf<fsm::Decoding, fsm::PrefillDone>()) {
            continue;
        }
        auto rank = std::tuple{-coordinator_.NumNewlyReleasableLcmBlocks(request->BlockTablesRef()),
                               request->GeneratedTokens(), request->Id()};
        if ((!victim_rank || rank < *victim_rank) && imageFits(*request)) {
            victim = request;
            victim_rank = std::move(rank);
        }
    }
    if (victim != nullptr) {
        return VictimChoice{.victim = victim, .image_fits = true};
    }
    return VictimChoice{.victim = newest, .image_fits = false};
}

bool Scheduler::imageFits(const Request& request) const {
    if (snapshot_slots_.AvailableSlots() == 0) {
        return false;
    }
    const std::int32_t num_computed_tokens = request.NumComputedTokens();
    // The published slots ride Host L2 (StartRetractionStores pins or copies
    // them there); only the rest must fit the pool. Pages the request has
    // completed but not yet hashed are published at retraction and join the
    // L2 leg too, so the probe is conservative only by those pages.
    const std::vector<std::vector<ImageSlot>> published =
        coordinator_.PublishedDataSlots(request.BlockTablesRef(), num_computed_tokens);
    return coordinator_.SnapshotPoolHolds(
        request.BlockTablesRef(), num_computed_tokens,
        coordinator_.HasHostPool() ? published : std::vector<std::vector<ImageSlot>>(published.size()));
}

// Suspends a quiescent victim with its image. First the completed prefix
// pages are published into the Device index (a finish-like publication other
// requests may hit; it costs only the hashes of pages not yet hashed, and the
// image's L2 leg is built from exactly those entries). Then the transfer
// manager takes the image -- published slots as pinned Host L2 entries,
// everything else in the snapshot pool, each in its Device block's bucket --
// and issues both store legs; the victim's Device pages are released by the
// retract event and may be granted away in this very round, because the
// runtime orders both copies on the forward thread's stream ahead of the
// plan's page reuse. A victim whose image cannot be held, or with no blob
// slot free, is not retracted: the shortfall is returned, and the publication
// it did is written back as the victim's progress so it is not redone.
std::optional<Scheduler::ImageShortfall> Scheduler::retractVictim(
    Request& victim, PlanBuild& build, std::vector<WriteBackOperation>& write_back_operations) {
    if (snapshot_slots_.AvailableSlots() == 0) {
        return ImageShortfall::kBlobSlot;
    }
    fsm::CacheProgress cache_progress = victim.CacheProgress();
    // Only what has actually been computed may be published as a prefix: an
    // incomplete prefill has only the chunks it has been through -- taking
    // TokenSize() there would publish pages that were never computed.
    const std::int32_t num_computed_tokens = victim.NumComputedTokens();
    RequestProgress progress =
        advanceRequestProgress(victim, cache_progress, num_computed_tokens, coordinator_.PrefixGranularity(),
                               /*stream_completed_to_host=*/false);
    if (progress.completed_pages) {
        classifyCompletedStateBoundaries(*progress.completed_pages, num_computed_tokens,
                                         coordinator_.PrefixGranularity());
        // A publication is a KV-event mutation: the newly hashed pages (decode
        // pages no admission has registered yet) need their token descriptors
        // first, exactly as publishCompletedPages registers before it caches.
        registerKvEventPrefixPages(victim, cache_progress.prefix_hashes,
                                   progress.completed_pages->first_new_prefix_page);
        coordinator_.CacheCompletedBlocks(victim.BlockTablesRef(), progress, cache_progress.access_epoch);
    }
    cache_progress.DiscardHashedStateBoundaries(coordinator_.PrefixGranularity());

    // The blob slot is shared with the store that exports into it (and later
    // the restore that imports from it): an abort before either ACK leaves
    // the slot with the op until the copy is done.
    auto blob_slot = std::make_shared<SnapshotSlotIndex>(snapshot_slots_.Allocate());
    std::optional<TierTransferManager::RetractionStores> stores = tier_transfers_.StartRetractionStores(
        victim.Id(), victim.RequestPoolIndex(), blob_slot, victim.BlockTablesRef(), num_computed_tokens);
    if (!stores) {
        // The publication stands; the victim keeps running with it recorded.
        victim.CacheProgressRef() = std::move(cache_progress);
        return ImageShortfall::kSnapshotPool;
    }
    victim.NoteRetracted();
    victim.CacheProgressRef() = std::move(cache_progress);
    if (stores->host_store) {
        write_back_operations.push_back(std::move(*stores->host_store));
    }
    build.snapshot_stores.push_back(std::move(stores->snapshot_store));
    victim.Apply(fsm::SnapshotRetractEvent{&coordinator_, next_retraction_epoch_++, victim.HasGeneratedOutput(),
                                           std::move(stores->image), std::move(blob_slot),
                                           std::move(stores->pending_store_ops)});
    spdlog::info("[Scheduler] retract: suspended request {} ({} tokens) with its image", victim.Id(),
                 victim.TokenSize());
    return std::nullopt;
}

void Scheduler::onImageDoesNotFit(Request& victim, ImageShortfall shortfall, PlanBuild& build) {
    std::string detail;
    switch (shortfall) {
        case ImageShortfall::kNoPool:
            detail =
                "the engine has no snapshot pool, so a capacity block aborts a request instead of imaging one "
                "(--retraction-snapshot-ratio 0 / --retraction-snapshot-host-gb 0 selects this; set either above 0 "
                "to retract and restore)";
            break;
        case ImageShortfall::kBlobSlot:
            detail = "no slot-state blob slot is free (max_retracted_requests=" +
                     std::to_string(config_.max_retracted_requests) + "; raise --retraction-snapshot-max-requests)";
            break;
        case ImageShortfall::kSnapshotPool:
            detail = "the snapshot pool cannot hold the image (num_snapshot_pages=" +
                     std::to_string(config_.snapshot_allocator.total_pages) +
                     "; raise --retraction-snapshot-ratio / --retraction-snapshot-host-gb)";
            break;
    }
    spdlog::warn(
        "[Scheduler] capacity abort: request {} ({} tokens) cannot be imaged -- {}; aborting it to free its "
        "pages for the blocked admission",
        victim.Id(), victim.TokenSize(), detail);
    build.plan.aborts.push_back(SchedulerAbort{
        .request_id = victim.Id(),
        .reason = AbortReason::kImageDoesNotFit,
        .detail = detail,
    });
    // Pages and request-pool slot return now; the grant proceeds on them.
    victim.Apply(fsm::AbortEvent{&coordinator_});
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
// The victim's own device blocks return to the pool immediately; both legs
// of the image copy are ordered on the forward thread's stream BEFORE
// anything this round writes to those pages -- the same stream carries the
// plan's zeroing and fences its forwards (see DeviceHandle.execute) -- so
// releasing them under the still-uncopied image is safe.
//
// A readmission's failed restore never reaches here (its phase records no
// blocker): when the readmission needs a victim, the two simply do not fit
// together, and swapping them is pure thrash -- it waits for a completion
// instead. Likewise while an ordinary (pinned) store is in flight: its
// Device pages come back at the ACK without anyone giving way, so retracting
// for capacity they hold would be the same thrash.
//
// The host side is finite too. When no candidate's image fits (no blob slot,
// or a snapshot pool too small for the tail -- always, without a pool) the
// last resort is to ABORT the newest retractable resident instead of imaging
// anyone (onImageDoesNotFit): its pages free in this round exactly as a
// retraction's would and the grant proceeds. The alternative, waiting, can
// deadlock once every resident needs a page.
//
// A grant that cannot join its round (a fused prefill beside an already-built
// decode batch outside mixed mode) still retracts one victim: the next
// round's phase order tries the blocker before any other claim on the freed
// pages.
void Scheduler::maybeRetractForCapacity(AdmissionFeedback& feedback, PlanBuild& build,
                                        std::span<Request* const> candidates,
                                        std::vector<WriteBackOperation>& write_back_operations) {
    Request* blocker = std::exchange(feedback.capacity_blocker, nullptr);
    if (!build.NoPrefillProgress() || blocker == nullptr) {
        return;
    }
    // A load-back mid-flight is writing pages its admission owns; the victim
    // policy cannot see that write, so no retraction until it lands. A
    // pinned store mid-flight holds pages the ACK is about to release; the
    // blocked admission retries against them next round before anyone is
    // sacrificed. (Stream-ordered stores -- both legs of an image -- hold
    // nothing and gate nothing; an in-flight restore gates nothing because
    // its request is Restoring, invisible to chooseVictim.)
    if (tier_transfers_.HasLoadBacksInFlight() || tier_transfers_.HasPinnedStoresInFlight()) {
        return;
    }

    while (true) {
        const VictimChoice choice = chooseVictim(candidates);
        Request* victim = choice.victim;
        if (victim == nullptr) {
            return;  // everything resident is exempt; only a completion can free capacity
        }
        if (victim->ResultsInFlight() > 0 || pdTransferInFlight(*victim)) {
            // The victim is chosen but not quiescent: a forward's KV write or
            // a PD transfer is still landing on its pages. Wait for it
            // rather than sacrificing a worse-ranked request.
            return;
        }
        if (blocker == victim) {
            // The victim blocked on its own next page; it comes back through
            // the readmission phase. Its freed capacity goes to the next
            // request whose admission failed this round, else to the first
            // waiting prompt -- granting it back to the victim's own
            // readmission is the loop the grant exists to break. With nobody
            // to serve, retracting it would only cost the image's copies and
            // its restore: it waits for a completion instead.
            const auto blocked = std::ranges::find_if(feedback.capacity_blocked, [victim](Request* request) {
                // An earlier iteration may have retracted one of them.
                return request != victim &&
                       request->IsAnyOf<fsm::Submitted, fsm::Prefilling, fsm::PrefillDone, fsm::Decoding>();
            });
            const auto waiting = std::ranges::find_if(
                candidates, [victim](Request* request) { return request != victim && request->Is<fsm::Submitted>(); });
            if (blocked != feedback.capacity_blocked.end()) {
                blocker = *blocked;
            } else if (waiting != candidates.end()) {
                blocker = *waiting;
            } else {
                return;
            }
        }
        // A victim nobody can image (or whose image turns out not to fit
        // once the L2 leg is tried) is aborted: the last resort frees its
        // pages for the grant all the same.
        std::optional<ImageShortfall> shortfall;
        if (choice.image_fits) {
            shortfall = retractVictim(*victim, build, write_back_operations);
        } else if (!config_.HasSnapshotPool()) {
            shortfall = ImageShortfall::kNoPool;
        } else {
            shortfall =
                snapshot_slots_.AvailableSlots() == 0 ? ImageShortfall::kBlobSlot : ImageShortfall::kSnapshotPool;
        }
        if (shortfall) {
            onImageDoesNotFit(*victim, *shortfall, build);
        }
        if (build.Full(config_.max_batch_size)) {
            return;
        }

        // On the D role every grant is a remote admission (a Submitted prompt
        // the peer prefills) or a blocked decode; a fused prefill grant can
        // join an already-built decode batch only in mixed mode.
        const bool blocked_on_decode = blocker->Is<fsm::Decoding>() || blocker->Is<fsm::PrefillDone>();
        const bool remote_grant = config_.role == Role::kD && !blocked_on_decode;
        if (!blocked_on_decode && !remote_grant && build.pushed_decode &&
            !(config_.role == Role::kFused && config_.enable_mixed_prefill_decode)) {
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
            if (auto admitted =
                    schedulePrefillCandidate(build.plan, feedback, blocker, budget, config_.decode_input_tokens,
                                             build.load_backs, build.prefetches)) {
                if (!admitted->operation) {
                    // The blocker went to prefetch its L3 prefix first; it is
                    // admitted on the freed pages once that landed.
                    build.scheduled.insert(blocker);
                } else if (remote_grant) {
                    // The blocker is now RemotePrefilling: the peer's prefill
                    // is out against its pages (pdTransferInFlight).
                    build.scheduled.insert(blocker);
                    build.remote_prefill.emplace_back(std::move(*admitted->operation));
                } else {
                    pushOperation(build, *blocker, std::move(*admitted->operation));
                    blocker->TrackScheduledForward();
                }
                return;
            }
        }
        if (!feedback.admission_failed) {
            return;  // not a capacity failure (zero-token alignment); stop retracting
        }
    }
}

// The debug knob: every |interval| plans, arm the oldest Decoding (interval >
// 0) or Prefilling (interval < 0) request. An armed request is kept out of
// this round's batch so that its outstanding forward lands, and is retracted
// at the first plan where it is quiescent -- bypassing the victim policy's
// exemption but not the image-fit refusal, which disarms it.
void Scheduler::maybeForceRetraction(PlanBuild& build, std::span<Request* const> candidates,
                                     std::vector<WriteBackOperation>& write_back_operations) {
    const std::int32_t interval = config_.debug_force_retraction_interval;
    if (interval == 0) {
        return;
    }
    const auto is_kind = [interval](const Request& request) {
        return interval > 0 ? request.Is<fsm::Decoding>() : request.Is<fsm::Prefilling>();
    };
    Request* armed = forced_victim_id_.empty() ? nullptr : findRequest(forced_victim_id_);
    if (armed != nullptr && !is_kind(*armed)) {
        armed = nullptr;  // finished, aborted, or moved on since it was armed
    }
    if (armed == nullptr) {
        forced_victim_id_.clear();
        if (plan_calls_ % std::abs(interval) != 0) {
            return;
        }
        const auto candidate = std::ranges::find_if(candidates, [&](Request* request) {
            return is_kind(*request) && !pdTransferInFlight(*request) && !build.Scheduled(*request);
        });
        if (candidate == candidates.end()) {
            return;
        }
        armed = *candidate;
        forced_victim_id_ = armed->Id();
    }
    if (armed->ResultsInFlight() > 0) {
        build.scheduled.insert(armed);  // sit this round out so the forward lands
        return;
    }
    forced_victim_id_.clear();
    // The knob is not capacity pressure: a victim it cannot image is left
    // running, never aborted (onImageDoesNotFit is the retraction loop's).
    if (retractVictim(*armed, build, write_back_operations)) {
        spdlog::info("[Scheduler] forced retraction of request {} refused: its image does not fit", armed->Id());
    }
}

std::vector<Request*> Scheduler::rankedReadmissions(std::span<Request* const> candidates) {
    const auto rank = [](const Request* request) {
        const fsm::Retracted& retracted = *request->GetIf<fsm::Retracted>();
        return std::pair{!retracted.ResumesGeneration(), retracted.RetractionEpoch()};
    };
    std::vector<Request*> landed;
    for (Request* request : candidates) {
        const auto* retracted = request->GetIf<fsm::Retracted>();
        if (retracted != nullptr && retracted->ImageLanded()) {
            landed.push_back(request);
        }
    }
    std::ranges::stable_sort(landed, [&rank](const Request* a, const Request* b) { return rank(a) < rank(b); });
    return landed;
}

// Head-of-line among readmissions: the first-ranked image may need more Device
// pages than are free while a smaller one behind it fits, so the ranked
// candidates are tried in turn (bounded) until one restores. Any landed image
// that waited for capacity seals new-prompt admission for the round, whether
// or not a later one restored: the pages it waits for must not go to a
// newcomer. A readmission that found no request-pool slot stops the scan
// without sealing -- nothing later would get a slot either, and neither would
// a newcomer.
bool Scheduler::scheduleReadmission(AdmissionFeedback& feedback, PlanBuild& build,
                                    std::span<Request* const> candidates) {
    bool new_prompts_sealed = false;
    std::int32_t attempts = 0;
    for (Request* readmission : rankedReadmissions(candidates)) {
        if (attempts == kMaxRestoreAttemptsPerRound) {
            break;
        }
        ++attempts;
        feedback.admission_failed = false;
        if (scheduleRestore(feedback, build, readmission)) {
            break;
        }
        if (!feedback.admission_failed) {
            break;
        }
        new_prompts_sealed = true;
    }
    return new_prompts_sealed;
}

// The readmission: fresh Device pages for the whole image, in the imaged
// buckets, plus the reserve the resumed state needs -- decided once per
// group, by kind, in ReservePrefillDemands exactly as a first chunk's is: the
// decode slot when the request resumes decoding or its completed prompt, the
// rest of the prompt plus the (escalated) admission headroom when it resumes
// mid-prefill. A restore takes no token budget and no batch slot: it is a
// cache op riding beside the batch, like a remote admission.
bool Scheduler::scheduleRestore(AdmissionFeedback& feedback, PlanBuild& build, Request* request) {
    const fsm::Retracted* retracted = request->GetIf<fsm::Retracted>();
    _assert(retracted != nullptr && retracted->ImageLanded(),
            "a restore resumes a Retracted request whose image landed");
    if (req_pool_allocator_.AvailableSlots() == 0) {
        return false;
    }
    const auto* resumes_prefilling = std::get_if<fsm::ResumePrefilling>(&retracted->shape);
    const std::int32_t resume_reserve =
        std::visit([](const auto& shape) { return shape.reserve_num_tokens_in_next_schedule_event; }, retracted->shape);
    // A mid-prefill victim still owes the prompt beyond its computed window.
    const std::int32_t unscheduled =
        resumes_prefilling == nullptr
            ? 0
            : request->PrefillSize() - (resumes_prefilling->window.begin + resumes_prefilling->window.size);
    const std::int32_t headroom = request->AdmissionHeadroom(kRetractionSafeSteps);
    const PrefillReserve reserve{
        .decode_input_tokens = std::max(config_.decode_input_tokens, resume_reserve),
        .completes_prefill = resumes_prefilling == nullptr,
        .prompt_headroom_tokens = headroom > 0 ? unscheduled + headroom : 0,
        .reserve_snapshot_state_growth = resumes_prefilling == nullptr,
    };
    std::vector<BlockTable> tables(static_cast<std::size_t>(coordinator_.NumGroups()));
    std::vector<GroupDemand> demands = MakeGroupDemands(tables, GroupDemand{.extent = DenseGrowth{0}});
    ReservePrefillDemands(demands, config_.cache_groups, reserve);

    std::optional<CacheCoordinator::AdmissionResult> result =
        coordinator_.Restore(retracted->image, demands, retracted->cache_progress.access_epoch);
    if (!result) {
        feedback.admission_failed = true;
        return false;
    }
    _assert(result->new_page_ids.size() == cache_group_ids_.size(),
            "restore fresh-page groups must match scheduler config");
    for (std::size_t i = 0; i < result->new_page_ids.size(); ++i) {
        auto& pending = build.plan.pages_to_zero[cache_group_ids_[i]];
        pending.insert(pending.end(), result->new_page_ids[i].begin(), result->new_page_ids[i].end());
    }
    // The restore op owns the request-pool row it imports the blob into until
    // its ACK hands it to the resumed state (an abort meanwhile must not
    // re-grant a row still being written), and shares the blob slot.
    SnapshotRestoreOperation op =
        tier_transfers_.StartSnapshotRestore(request->Id(), req_pool_allocator_.Allocate(), retracted->blob_slot,
                                             std::move(result->load_pairs), std::move(result->snapshot_pairs));
    const std::uint32_t restore_op = op.op_id;
    build.snapshot_restores.push_back(std::move(op));
    request->Apply(fsm::ScheduleRestoreEvent{&coordinator_, std::move(tables), restore_op});
    build.scheduled.insert(request);
    spdlog::info("[Scheduler] restore: request {} ({} tokens) copies its image back", request->Id(),
                 request->TokenSize());
    return true;
}

// Local prefill phases shared by the P and fused grammars: resident chunks
// (they hold pages), then new prompts. One loop per tier so the order is
// visible.
//
// Head-of-line: an incomplete prefill that scheduled stops further prefill
// work (nothing may consume the capacity it still needs), and a resident
// chunk that FAILED admission stops it too -- admitting behind it would
// strand it. A readmission whose restore failed for capacity waits without
// becoming the capacity blocker (retracting a victim for it is pure thrash
// -- the two simply do not fit together), but it does seal new-prompt
// admission (new_prompts_sealed): a newcomer taking the pages it is waiting
// for would starve it.
void Scheduler::scheduleLocalPrefillWork(AdmissionFeedback& feedback, PlanBuild& build,
                                         std::span<Request* const> candidates, bool new_prompts_sealed,
                                         std::int32_t decode_reserve) {
    for (const bool resident : {true, false}) {
        if (!resident && new_prompts_sealed) {
            return;
        }
        for (Request* request : candidates) {
            if (build.Full(config_.max_batch_size)) {
                return;
            }
            if ((resident ? !request->Is<fsm::Prefilling>() : !request->Is<fsm::Submitted>()) ||
                build.Scheduled(*request)) {
                continue;
            }
            feedback.admission_failed = false;
            if (auto admitted = schedulePrefillCandidate(build.plan, feedback, request, build.token_budget,
                                                         decode_reserve, build.load_backs, build.prefetches)) {
                if (!admitted->operation) {
                    // Gone to prefetch its L3 prefix (Prefetching): it holds
                    // no head of line, so the prompts behind it go on.
                    build.scheduled.insert(request);
                    continue;
                }
                pushOperation(build, *request, std::move(*admitted->operation));
                request->TrackScheduledForward();
                if (holdsHeadOfLine(*request)) {
                    return;
                }
            } else if (feedback.admission_failed) {
                feedback.NoteCapacityBlocked(request);
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
        } else if (feedback.admission_failed) {
            feedback.NoteCapacityBlocked(request);
        }
    }
}

// P role: prefill worker. Completed prompts leave on plan.remote_decode
// first -- their KV pages stay pinned until the transfer finishes
// (pdTransferInFlight: on this role every page-holding state is pinned, from
// the first scheduled chunk to the PD ACK), so releasing them outranks
// feeding more prompt work -- then the prefill phases run with the same
// completing-chunk decode reserve as every other role: this node never
// decodes locally, but the forward that completes a prompt drafts the first
// candidate window into that reserve before the remote decode ships it. No
// retraction either: a P node's pressure valve is the transfer itself, so
// this grammar never calls maybeRetractForCapacity and nothing here is ever
// restored.
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

    scheduleLocalPrefillWork(feedback, build, candidates, /*new_prompts_sealed=*/false, config_.decode_input_tokens);
}

// D role: decode worker. One restore rides beside the decode batch, then at
// most ONE remote admission rides plan.remote_prefill beside it too.
// Retraction picks decode victims and retries the blocked admission in the
// same round. Nothing on this role is ever Prefilling: a prompt is the peer's
// work, and a retracted request comes back by restore, never by a local
// prefill.
void Scheduler::buildDecodeWorkerPlan(AdmissionFeedback& feedback, PlanBuild& build,
                                      std::span<Request* const> candidates,
                                      std::vector<WriteBackOperation>& write_back_operations) {
    maybeForceRetraction(build, candidates, write_back_operations);

    // Phase 1: the one readmission this round may restore. It resumes a
    // streaming client, so it takes capacity ahead of fresh work. A restore
    // that does not fit simply waits -- it never triggers retraction
    // (swapping it with a victim is pure thrash) and never stalls the decodes
    // below -- but it seals the remote admission: a newcomer taking the pages
    // it waits for would starve it.
    const bool new_prompts_sealed = scheduleReadmission(feedback, build, candidates);

    // Phase 2: the decode batch. Completed prefills' first decodes go ahead
    // of the running ones; neither consumes token budget on this role.
    scheduleDecodeBatch(feedback, build, candidates);

    // Phase 3: at most one remote admission -- the whole prompt reserves at
    // once, so admitting a queue's worth in one round would drain the pool
    // before any of their KV arrives. It rides plan.remote_prefill beside
    // the decode batch: no token budget, no batch slot. No
    // TrackScheduledForward: the peer runs this prefill, so no forward of
    // this engine's is out against the pages; the RemotePrefilling state it
    // enters is what holds them (pdTransferInFlight).
    if (!new_prompts_sealed) {
        for (Request* request : candidates) {
            if (!request->Is<fsm::Submitted>()) {
                continue;
            }
            feedback.admission_failed = false;
            if (auto admitted =
                    schedulePrefillCandidate(build.plan, feedback, request, request->PrefillSize(),
                                             config_.decode_input_tokens, build.load_backs, build.prefetches)) {
                // The D role probes the Device alone, so no admission here
                // ever goes to prefetch.
                _assert(admitted->operation.has_value(), "a D-role admission never prefetches from L3");
                build.scheduled.insert(request);
                build.remote_prefill.emplace_back(std::move(*admitted->operation));
                break;
            }
            if (feedback.admission_failed) {
                feedback.NoteCapacityBlocked(request);
            }
        }
    }

    maybeRetractForCapacity(feedback, build, candidates, write_back_operations);
}

// Fused role: one engine does everything locally. The one readmission this
// round may restore goes first on every mode -- it resumes a streaming client
// and takes no budget. In mixed mode resident decodes then take their token
// budget -- a client is streaming them, and a long prefill chunk must not
// starve them -- leaving the budget a pending local prefill cannot advance
// without (MinPrefillChunkTokens); the prefill phases spend the rest. Outside
// mixed mode prefill work runs alone, and decodes get a round only when no
// prefill scheduled.
void Scheduler::buildFusedPlan(AdmissionFeedback& feedback, PlanBuild& build, std::span<Request* const> candidates,
                               std::vector<WriteBackOperation>& write_back_operations) {
    maybeForceRetraction(build, candidates, write_back_operations);

    const bool new_prompts_sealed = scheduleReadmission(feedback, build, candidates);
    if (config_.enable_mixed_prefill_decode) {
        const bool has_local_prefill = std::ranges::any_of(candidates, [](const Request* request) {
            return request->Is<fsm::Prefilling>() || request->Is<fsm::Submitted>();
        });
        build.state_prefill_reserve = has_local_prefill ? MinPrefillChunkTokens(coordinator_) : 0;
        scheduleDecodeBatch(feedback, build, candidates);
    }

    scheduleLocalPrefillWork(feedback, build, candidates, new_prompts_sealed, config_.decode_input_tokens);

    if (!config_.enable_mixed_prefill_decode && !build.pushed_prefill) {
        scheduleDecodeBatch(feedback, build, candidates);
    }

    maybeRetractForCapacity(feedback, build, candidates, write_back_operations);
}

Scheduler::BuiltOperations Scheduler::buildForwardOperations(ExecutionPlan& plan, std::vector<Request*> candidates,
                                                             std::vector<WriteBackOperation>& write_back_operations) {
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
    return BuiltOperations{
        .forward = std::move(build.operations),
        .load_backs = std::move(build.load_backs),
        .prefetches = std::move(build.prefetches),
        .snapshot_stores = std::move(build.snapshot_stores),
        .snapshot_restores = std::move(build.snapshot_restores),
    };
}

}  // namespace tokenspeed
