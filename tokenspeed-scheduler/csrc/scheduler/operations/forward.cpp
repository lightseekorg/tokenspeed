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
#include <limits>
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
#include "scheduler/capacity_model.h"
#include "scheduler/operations/cache.h"
#include "scheduler/operations/forward.h"
#include "scheduler/operations/group_demands.h"
#include "scheduler/operations/prefill_chunk.h"
#include "scheduler/operations/progress.h"
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

// The page boundary before a long prompt's replay tail where the forward that
// spans it also writes the State checkpoint
// (SchedulerConfig::state_checkpoint_in_forward); 0 means none. The D role
// computes no prompt, and without prefix caching nothing could reuse it.
std::int32_t inForwardStateCheckpoint(const SchedulerConfig& config, const CacheCoordinator& coordinator,
                                      std::int32_t prompt_tokens) {
    if (!config.state_checkpoint_in_forward || config.disable_prefix_cache || config.role == Role::kD) {
        return 0;
    }
    return PrefillStateCheckpointTokens(coordinator, prompt_tokens, config.max_scheduled_tokens,
                                        config.prefix_replay_tokens);
}

// Whether the forward computing [first_pos, first_pos + tokens) writes the
// in-forward checkpoint: the boundary lies strictly inside it (a forward
// ending there leaves the state in slot 1, published as a natural checkpoint)
// and whole pages past its start, where the runtime's chunked scan emits a
// state.
bool writesInForwardCheckpoint(std::int32_t checkpoint, std::int32_t first_pos, std::int32_t tokens,
                               std::int32_t prefix_granularity) {
    return checkpoint > first_pos && checkpoint < first_pos + tokens &&
           (checkpoint - first_pos) % prefix_granularity == 0;
}

// The tokens a chunk of a request with an in-forward checkpoint takes from
// the `tokens` its budget allows. The budget a batch's decodes leave rarely
// ends on a page, and a chunk starting inside one cannot write the boundary
// (writesInForwardCheckpoint). So a chunk ending less than a full chunk short
// of the boundary -- the next chunk may span it -- ends on a page when the
// budget reaches one, unless a junction at or past its end comes first and
// fixes where a later chunk starts anyway; and a chunk spanning the boundary
// from inside a page (a hit on a prompt-end tail, a budget short of a page)
// ends at the boundary, whose live slot the next chunk's admission publishes.
std::int32_t chunkTowardInForwardCheckpoint(std::int32_t checkpoint, std::int32_t first_pos, std::int32_t tokens,
                                            std::int32_t junction, std::int32_t max_chunk_tokens,
                                            std::int32_t prefix_granularity) {
    const std::int32_t end = first_pos + tokens;
    if (checkpoint <= first_pos || end == checkpoint || checkpoint - end >= max_chunk_tokens) {
        return tokens;
    }
    if (end < checkpoint) {
        if (junction >= end && junction < checkpoint) {
            return tokens;
        }
        const std::int32_t page_end = end / prefix_granularity * prefix_granularity;
        return page_end > first_pos ? page_end - first_pos : tokens;
    }
    return (checkpoint - first_pos) % prefix_granularity == 0 ? tokens : checkpoint - first_pos;
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

}  // namespace

RequestProgress AdvanceRequestProgress(const Request& request, fsm::CacheProgress& cache_progress,
                                       std::int32_t num_computed_tokens, std::int32_t prefix_granularity,
                                       bool stream_completed_to_host, PublicationPoint point,
                                       bool allow_state_checkpoint) {
    const std::int32_t first_new_prefix_page = static_cast<std::int32_t>(cache_progress.prefix_hashes.size());
    const std::int32_t filled_prefix_pages = num_computed_tokens / prefix_granularity;
    if (filled_prefix_pages > first_new_prefix_page) {
        appendCompletedPrefixHashes(cache_progress.prefix_hashes, request.FullPrefixPages(false), filled_prefix_pages);
    }
    RequestProgress progress{.num_computed_tokens = num_computed_tokens};
    std::optional<TailPublication> tail;
    if (point != PublicationPoint::kChunk) {
        const std::string& prior =
            cache_progress.prefix_hashes.empty() ? std::string{} : cache_progress.prefix_hashes.back();
        tail = request.TailPublicationAt(num_computed_tokens, prior);
    }
    // A final point publishes the frontier's state even when its pages were
    // all published by an earlier step (a D request leaving at a page-aligned
    // frontier): the checkpoint is keyed by the last page's chain hash.
    // A retained in-forward checkpoint below is always eligible. Publishing
    // the live slot additionally requires a prefill-produced frontier (or an
    // unrestricted State group); an aligned decode state is not a checkpoint.
    const bool publish_state = allow_state_checkpoint && point != PublicationPoint::kChunk;
    const bool state_only = publish_state && !tail && !cache_progress.prefix_hashes.empty() &&
                            num_computed_tokens == filled_prefix_pages * prefix_granularity;
    // The in-forward State checkpoint a scheduled forward wrote is published
    // at the first point past it; its blocks move to the published holder,
    // which the returned spans view.
    cache_progress.published_state_checkpoint_blocks.clear();
    std::int32_t state_checkpoint_pages = 0;
    if (cache_progress.state_checkpoint_tokens > 0 && num_computed_tokens >= cache_progress.state_checkpoint_tokens) {
        _assert(cache_progress.state_checkpoint_tokens % prefix_granularity == 0,
                "an in-forward State checkpoint sits at a page boundary");
        state_checkpoint_pages = cache_progress.state_checkpoint_tokens / prefix_granularity;
        cache_progress.published_state_checkpoint_blocks = std::move(cache_progress.state_checkpoint_blocks);
        cache_progress.state_checkpoint_blocks.clear();
        cache_progress.state_checkpoint_tokens = 0;
    }
    if (first_new_prefix_page < static_cast<std::int32_t>(cache_progress.prefix_hashes.size()) || tail || state_only ||
        state_checkpoint_pages > 0) {
        progress.completed_pages = CompletedPages{
            .prefix_hashes = cache_progress.prefix_hashes,
            .first_new_prefix_page = first_new_prefix_page,
            .tail = tail,
            .owner_continues = point == PublicationPoint::kOwnerContinues,
            .publish_state = publish_state,
            .state_checkpoint_pages = state_checkpoint_pages,
            .state_checkpoint_blocks = cache_progress.published_state_checkpoint_blocks,
            .boundary_kind =
                point == PublicationPoint::kChunk ? CacheBoundaryKind::kChunk : CacheBoundaryKind::kEndpoint,
            .stream_completed_to_host = stream_completed_to_host,
        };
    }
    return progress;
}

namespace {

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
    // The in-forward State checkpoint this forward writes: its position in
    // the chunk, and each State group's block as the row's third entry.
    const fsm::CacheProgress& cache_progress = request.CacheProgressRef();
    const std::int32_t checkpoint = cache_progress.state_checkpoint_tokens;
    if (checkpoint > info.already_scheduled_len && checkpoint < info.already_scheduled_len + info.extend_len) {
        operation.extend_state_checkpoint_len = checkpoint - info.already_scheduled_len;
        for (std::size_t i = 0; i < group_ids.size(); ++i) {
            if (const CacheBlockRef& block = cache_progress.state_checkpoint_blocks[i]) {
                operation.block_tables[group_ids[i]].push_back(
                    coordinator.Allocator(static_cast<std::int32_t>(i)).ResolveCacheBlockId(*block));
            }
        }
    }
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
    const auto probe = [this, request](const CacheCoordinator::PromptPrefix& prompt) {
        if (config_.role == Role::kD) {
            return coordinator_.ProbeDecodeDevicePrefix(prompt, request->PrefillSize());
        }
        return coordinator_.ProbePromptPrefix(prompt);
    };
    const std::int32_t prefix_granularity = coordinator_.PrefixGranularity();
    // The final prompt token is always recomputed to produce logits. Some
    // consumers additionally require a larger prompt tail (for example, to
    // rebuild request-persistent state that is not stored in the KV cache).
    // Limit the probe itself so excluded hit pages are never claimed: admission
    // will allocate private writable pages for the replayed suffix. A request
    // may tighten the bound further (RequestSpec::max_cached_prefix_tokens) so
    // the positions it needs logits for are recomputed rather than matched.
    // The bound is in tokens: a hit may end at it exactly.
    const std::int32_t replay_tokens = std::max(config_.prefix_replay_tokens, 1);
    const std::int32_t max_cacheable_tokens =
        std::max(std::min(request->PrefillSize() - replay_tokens, request->MaxCachedPrefixTokens()), 0);
    // The last token is recomputed, so a page the prompt fills exactly can
    // never be hit -- except on the D role, whose prompt-end state check
    // needs the prompt's last page hashed when the prompt ends on it.
    const std::int32_t candidate_prefix_pages = config_.role == Role::kD
                                                    ? request->PrefillSize() / prefix_granularity
                                                    : std::max((request->PrefillSize() - 1) / prefix_granularity, 0);
    const auto hashes = request->PromptPrefixHashes().first(static_cast<std::size_t>(candidate_prefix_pages));

    AdmissionMatch match;
    match.candidate_prefix_hashes = hashes;
    if (config_.disable_prefix_cache) {
        match.probe = probe(CacheCoordinator::PromptPrefix{});
        return match;
    }
    // The KV walk stops at the probe bound; the D role also reads the prompt's
    // last page and its final tokens for the prompt-end state check.
    const CacheCoordinator::PromptPrefix prompt{
        .prefix_hashes = hashes,
        .tokens = request->TokenSlice(0, request->PrefillSize()),
        .max_tokens = max_cacheable_tokens,
    };
    match.probe = probe(prompt);
    // Opt-in diagnostics deliberately walk more than matching needs. Emit
    // only a successful first admission, using the pre-admission snapshot.
    static const bool inspect_prefix = std::getenv("TOKENSPEED_PREFIX_CACHE_DIAGNOSTICS") != nullptr;
    if (inspect_prefix) {
        const auto boundaries = coordinator_.InspectPrefixAvailability(prompt);
        std::string rows;
        for (const auto& boundary : boundaries) {
            if (!rows.empty()) {
                rows += ',';
            }
            rows += fmt::format("[{},{},{},{}]", boundary.tokens, boundary.device_missing, boundary.host_missing,
                                boundary.union_missing);
        }
        std::string group_names;
        for (const std::string& id : cache_group_ids_) {
            if (!group_names.empty()) {
                group_names += ',';
            }
            group_names += id;
        }
        std::string prompt_hashes;
        for (const std::string& hash : hashes) {
            if (!prompt_hashes.empty()) {
                prompt_hashes += ',';
            }
            prompt_hashes += hash;
        }
        match.prefix_diagnostic = fmt::format(
            "request={} prompt={} bound={} resumed={} kv={} groups={} hashes={} boundaries=[{}]", request->Id(),
            request->PrefillSize(), prompt.max_tokens,
            std::max(match.probe.device.num_common_tokens, match.probe.host.num_common_tokens),
            std::max(match.probe.device.kv_tokens, match.probe.host.kv_tokens), group_names, prompt_hashes, rows);
    }
    // The chain progress counts full pages only; a tail is a boundary, not a
    // hashed page.
    const std::int32_t hit_prefix_pages =
        std::max(match.probe.device.full_page_tokens, match.probe.host.full_page_tokens) / prefix_granularity;
    match.prefix_hashes = hashes.first(static_cast<std::size_t>(hit_prefix_pages));

    const std::int32_t extension_pages =
        std::max(match.probe.host.full_page_tokens - match.probe.device.full_page_tokens, 0) / prefix_granularity;
    const auto extension_begin = static_cast<std::size_t>(match.probe.device.full_page_tokens / prefix_granularity);
    match.extension_hashes = hashes.subspan(extension_begin, static_cast<std::size_t>(extension_pages));
    return match;
}

void Scheduler::recordAdmittedPages(ExecutionPlan& plan, const CacheCoordinator::AdmissionResult& result) {
    _assert(
        result.new_page_ids.size() == cache_group_ids_.size() && result.pages_to_copy.size() == cache_group_ids_.size(),
        "admission page groups must match scheduler config");
    for (std::size_t i = 0; i < cache_group_ids_.size(); ++i) {
        auto& pending = plan.pages_to_zero[cache_group_ids_[i]];
        pending.insert(pending.end(), result.new_page_ids[i].begin(), result.new_page_ids[i].end());
        auto& copies = plan.pages_to_copy[cache_group_ids_[i]];
        copies.insert(copies.end(), result.pages_to_copy[i].begin(), result.pages_to_copy[i].end());
    }
}

std::optional<CacheCoordinator::AdmissionResult> Scheduler::admit(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                                  CacheCoordinator::PrefixProbe&& prefix,
                                                                  std::span<const GroupDemand> demands,
                                                                  const RequestProgress& progress,
                                                                  std::optional<std::uint64_t> request_access_epoch,
                                                                  bool evict_blocks_past_segments) {
    std::optional<CacheCoordinator::AdmissionResult> result =
        coordinator_.Admit(std::move(prefix), demands, progress, request_access_epoch, evict_blocks_past_segments);
    if (!result) {
        feedback.admission_failed = true;
        return std::nullopt;
    }
    recordAdmittedPages(plan, *result);
    return result;
}

std::optional<CacheCoordinator::AdmissionResult> Scheduler::admitWithKvEventTracking(
    ExecutionPlan& plan, AdmissionFeedback& feedback, Request& request, const fsm::CacheProgress& cache_progress,
    std::span<const GroupDemand> demands, const RequestProgress& progress, bool evict_blocks_past_segments) {
    if (progress.completed_pages) {
        registerKvEventPublication(request, cache_progress.prefix_hashes, *progress.completed_pages);
    } else {
        registerKvEventPrefixPages(request, cache_progress.prefix_hashes,
                                   static_cast<std::int32_t>(cache_progress.prefix_hashes.size()));
    }
    // Alone, the request has no release to wait for: when whole segments
    // cannot fund it, blocks leave one by one (docs/design/scheduler.md §1.4),
    // never for an attempt the caller drops instead (evict_blocks_past_segments
    // false).
    const bool others_hold_pages = std::ranges::any_of(requests_, [&request](const std::unique_ptr<Request>& other) {
        return other.get() != &request && other->HoldsPages();
    });
    return admit(plan, feedback, coordinator_.ProbePrefix(std::span<const std::string>{}), demands, progress,
                 cache_progress.access_epoch,
                 /*evict_blocks_past_segments=*/evict_blocks_past_segments && !others_hold_pages);
}

Scheduler::FirstChunkOutcome Scheduler::schedulePrefillFirstChunk(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                                  Request* request, std::int32_t remaining,
                                                                  std::int32_t decode_input_tokens,
                                                                  std::int64_t kv_token_limit) {
    // No admission is possible while slots are full. L3 preparation runs
    // independently before forward scheduling and needs no request slot.
    if (req_pool_allocator_.AvailableSlots() == 0) {
        return FirstChunkOutcome{};
    }
    AdmissionMatch match = matchPrefixAtAdmission(request);

    // The D role's prompts are the peer's work; every other first chunk is local.
    const fsm::PrefillSource source =
        config_.role == Role::kD ? fsm::PrefillSource::kRemote : fsm::PrefillSource::kLocal;
    const std::int32_t prefix_granularity = coordinator_.PrefixGranularity();
    registerKvEventPrefixPages(*request, match.candidate_prefix_hashes, 0);

    const std::int32_t hit_tokens = std::max(match.probe.device.num_common_tokens, match.probe.host.num_common_tokens);
    // KV coverage beyond the state checkpoint the hit resumes from is the
    // junction: the first chunk is cut there so its natural checkpoint lands
    // where the KV is already cached.
    const std::int32_t kv_tokens = std::max(match.probe.device.kv_tokens, match.probe.host.kv_tokens);
    const std::int32_t junction_tokens = kv_tokens > hit_tokens ? kv_tokens : 0;
    _assert(junction_tokens == 0 || junction_tokens < request->PrefillSize(),
            "a junction lies inside the unmatched prompt");

    const std::int32_t unscheduled = request->PrefillSize() - hit_tokens;
    remaining = static_cast<std::int32_t>(
        std::min<std::int64_t>(remaining, std::max<std::int64_t>(0, kv_token_limit - hit_tokens)));
    // A remote landing completes the prompt in one transfer; no junction cut
    // can shorten it.
    std::int32_t tokens_this_round = source == fsm::PrefillSource::kRemote
                                         ? unscheduled
                                         : PrefillChunkTokens(coordinator_, hit_tokens, /*resumes_hit=*/true,
                                                              unscheduled, remaining, junction_tokens);
    if (tokens_this_round == 0) {
        return FirstChunkOutcome{};
    }
    const std::int32_t after_tokens = hit_tokens + tokens_this_round;
    // Every admission on a decoding role (D, Fused) secures real headroom
    // before the prefill starts: the rest of the prompt plus decode room
    // that starts at one safe-step window and grows with each retraction
    // (Request::AdmissionHeadroom). Pages only -- the request still computes
    // one chunk per round, because the chunk size is a forward-pass limit
    // rather than a capacity one. The P role is exempt: it never decodes
    // locally and never retracts, so there is no decode room to prepay.
    const std::int32_t headroom = config_.role == Role::kP ? 0 : request->AdmissionHeadroom(kRetractionSafeSteps);
    std::vector<BlockTable> tables(static_cast<std::size_t>(coordinator_.NumGroups()));
    // Whether another request holds pages. Alone, the request asks for what
    // max_single_request_tokens counts and waits for no release (below).
    const bool others_hold_pages = std::ranges::any_of(requests_, [request](const std::unique_ptr<Request>& other) {
        return other.get() != request && other->HoldsPages();
    });
    // The first chunk's demands for a chunk of `tokens`, with or without an
    // in-forward State checkpoint.
    const auto chunk_demands = [&](std::int32_t tokens, bool checkpoint) {
        // Sliding-window prepay (docs/design/scheduler.md §1). A remote
        // landing has no later local chunk: the peer fills the whole prompt.
        const std::int32_t later_chunk_tokens = source == fsm::PrefillSource::kLocal ? unscheduled - tokens : 0;
        // Only beside another page holder. Any holder, not only one sharing
        // the hit: hit pages may become shared after this admission.
        const bool prepays = (hit_tokens > 0 || later_chunk_tokens > 0) && others_hold_pages;
        const PrefillReserve reserve{
            .decode_input_tokens = decode_input_tokens,
            .completes_prefill = tokens == unscheduled,
            .prompt_headroom_tokens = headroom > 0 ? unscheduled - tokens + headroom : 0,
            .swa_hit_tokens = prepays ? hit_tokens : 0,
            .swa_later_chunk_tokens = prepays ? later_chunk_tokens : 0,
            .swa_prompt_after_hit_tokens = unscheduled,
        };
        // Every State table starts as [checkpoint or zero state, fresh] for a
        // local chunk, which reads slot 0 and writes slot 1; the peer's
        // landing fills slot 1 and reads nothing, so it leaves slot 0 null
        // unless the prompt-end checkpoint was claimed.
        std::vector<GroupDemand> demands = MakeGroupDemands(
            tables, config_.cache_groups, GroupDemand{.extent = DenseGrowth{tokens}},
            StateStep{.kind = source == fsm::PrefillSource::kRemote ? StateStep::Kind::kLand : StateStep::Kind::kRotate,
                      .checkpoint = checkpoint});
        ReservePrefillDemands(demands, config_.cache_groups, reserve);
        if (source == fsm::PrefillSource::kRemote) {
            // The peer's retained window lands as a sparse suffix behind the
            // hit: the first landed block is the one after the hit's last
            // block (a partial tail block is already the hitter's own fresh
            // page), or the window's first block when that is later.
            for (std::size_t i = 0; i < demands.size(); ++i) {
                const CacheGroupConfig& group = config_.cache_groups[i];
                if (group.attn_kind == AttnKind::kSlidingWindow) {
                    const std::int32_t block_granularity =
                        coordinator_.GroupBlockGranularity(static_cast<std::int32_t>(i));
                    const std::int32_t hit_blocks = (hit_tokens + block_granularity - 1) / block_granularity;
                    const std::int32_t retained_begin =
                        std::max(0, request->PrefillSize() - *group.sliding_window_tokens + 1);
                    demands[i].extent = SparseSuffix{
                        .extent_tokens = request->PrefillSize(),
                        .first_block = std::max(hit_blocks, retained_begin / block_granularity),
                    };
                }
            }
        }
        return demands;
    };
    // The in-forward State checkpoint, when this chunk spans its boundary.
    const std::int32_t state_checkpoint = source == fsm::PrefillSource::kLocal
                                              ? inForwardStateCheckpoint(config_, coordinator_, request->PrefillSize())
                                              : 0;
    tokens_this_round = chunkTowardInForwardCheckpoint(state_checkpoint, hit_tokens, tokens_this_round, junction_tokens,
                                                       config_.max_scheduled_tokens, coordinator_.PrefixGranularity());
    bool writes_checkpoint =
        writesInForwardCheckpoint(state_checkpoint, hit_tokens, tokens_this_round, coordinator_.PrefixGranularity());
    // Accounting only, read before Admit takes the probe: what a Host suffix
    // adds when it carries the hit past the Device's own boundary (even when
    // it adds only a State checkpoint over Device KV).
    const bool host_won = match.probe.host.num_common_tokens > match.probe.device_only_hit_tokens;
    const std::int32_t host_gain_tokens = host_won ? hit_tokens - match.probe.device_only_hit_tokens : 0;
    // First admission has computed nothing to publish or reclaim. A failed
    // admission changes nothing; one that cannot also hold the in-forward
    // checkpoint's block ends the chunk at the boundary instead, so the next
    // chunk's admission publishes its live slot there, at the cost of a
    // round (a cache this full keeps no spare block). That optional block
    // never makes blocks leave one by one (docs/design/scheduler.md §1.4).
    std::optional<CacheCoordinator::AdmissionResult> admission;
    if (writes_checkpoint) {
        admission = coordinator_.Admit(CacheCoordinator::PrefixProbe{match.probe},
                                       chunk_demands(tokens_this_round, true), RequestProgress{},
                                       /*request_access_epoch=*/std::nullopt,
                                       /*evict_blocks_past_segments=*/false);
        if (!admission) {
            writes_checkpoint = false;
            tokens_this_round = state_checkpoint - hit_tokens;
        }
    }
    if (!admission) {
        admission = coordinator_.Admit(std::move(match.probe), chunk_demands(tokens_this_round, false),
                                       RequestProgress{}, /*request_access_epoch=*/std::nullopt,
                                       /*evict_blocks_past_segments=*/!others_hold_pages);
    }
    if (!admission) {
        feedback.admission_failed = true;
        return FirstChunkOutcome{};
    }
    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t decode_reserve = completes_prefill ? decode_input_tokens : 0;
    // Every Host hit is an entry the admission acquired in the same step, so
    // the admitted prefix is the probed one.
    _assert(std::max(admission->device_prefix_tokens, admission->host_prefix_tokens) == hit_tokens,
            "an admission claims exactly the prefix it probed");
    _assert(admission->junction_tokens == junction_tokens, "the junction changed between probe and admission");
    if (!match.prefix_diagnostic.empty()) {
        spdlog::info("[PrefixCacheDiagnostic] {}", match.prefix_diagnostic);
    }
    recordAdmittedPages(plan, *admission);
    // host_prefix_tokens is a boundary: the Host loads the KV that lies beyond
    // the Device's prefix (0 when only checkpoints come from Host), and
    // nothing when the Device won.
    const std::int32_t host_hit_tokens = std::max(admission->host_prefix_tokens - admission->device_prefix_tokens, 0);
    prefix_hit_stats_.prompt_tokens += request->PrefillSize();
    prefix_hit_stats_.device_hit_tokens += hit_tokens - host_hit_tokens;
    prefix_hit_stats_.host_hit_tokens += host_hit_tokens;
    prefix_hit_stats_.host_gain_tokens += host_gain_tokens;
    prefix_hit_stats_.unresumable_kv_tokens += junction_tokens > 0 ? junction_tokens - hit_tokens : 0;

    if (!match.extension_hashes.empty()) {
        // The Host-warm H2D destinations of full pages are filled on Host
        // already; publish them on the Device now (the layer-wise load makes
        // the bytes available before any forward reads them). A Host tail is
        // loaded into the hitter's private page and stays unpublished.
        const std::int32_t first_extension_slot = match.probe.device.full_page_tokens / prefix_granularity;
        for (std::size_t i = 0; i < match.extension_hashes.size(); ++i) {
            coordinator_.CacheFullBlocks(tables, std::span<const std::string>(match.extension_hashes).subspan(i, 1),
                                         admission->access_epoch, first_extension_slot + static_cast<std::int32_t>(i),
                                         CacheBoundaryKind::kChunk);
        }
    }
    fsm::CacheProgress cache_progress{
        .prefix_hashes = std::vector<std::string>(match.prefix_hashes.begin(), match.prefix_hashes.end()),
        .access_epoch = admission->access_epoch,
        .junction_tokens = admission->junction_tokens,
        .resumed_tail_hash = std::move(admission->resumed_tail_hash),
    };
    if (writes_checkpoint) {
        cache_progress.state_checkpoint_tokens = state_checkpoint;
        cache_progress.state_checkpoint_blocks = std::move(admission->state_checkpoint_blocks);
    }
    return FirstChunkOutcome{.state_prefix_cached = admission->state_prefix_cached,
                             .event = fsm::SchedulePrefillFirstChunkEvent{
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
    std::int32_t reserve_num_tokens_in_next_schedule_event, std::int64_t kv_token_limit) {
    const std::int32_t unscheduled = request->UnscheduledPrefillSize();
    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    remaining = static_cast<std::int32_t>(
        std::min<std::int64_t>(remaining, std::max<std::int64_t>(0, kv_token_limit - first_pos)));
    fsm::CacheProgress cache_progress = request->CacheProgress();
    std::int32_t prefill_tokens = PrefillChunkTokens(coordinator_, first_pos, /*resumes_hit=*/false, unscheduled,
                                                     remaining, cache_progress.junction_tokens);
    if (prefill_tokens == 0) {
        return std::nullopt;
    }
    const std::int32_t state_checkpoint = inForwardStateCheckpoint(config_, coordinator_, request->PrefillSize());
    prefill_tokens =
        chunkTowardInForwardCheckpoint(state_checkpoint, first_pos, prefill_tokens, cache_progress.junction_tokens,
                                       config_.max_scheduled_tokens, coordinator_.PrefixGranularity());
    bool writes_checkpoint =
        writesInForwardCheckpoint(state_checkpoint, first_pos, prefill_tokens, coordinator_.PrefixGranularity());
    // The chunk that ended at the junction publishes its natural checkpoint
    // (KV tail and state) and moves off the published blocks, and so does one
    // that happened to end at the in-forward checkpoint's boundary; every
    // other chunk end publishes only its completed full pages.
    const bool ends_at_junction = cache_progress.junction_tokens > 0 && first_pos == cache_progress.junction_tokens;
    const bool ends_at_state_checkpoint = state_checkpoint > 0 && first_pos == state_checkpoint;
    if (first_pos >= cache_progress.junction_tokens) {
        cache_progress.junction_tokens = 0;
    }
    const RequestProgress progress = AdvanceRequestProgress(
        *request, cache_progress, request->NumComputedTokens(), coordinator_.PrefixGranularity(),
        config_.StreamsDeviceCacheToHost(),
        ends_at_junction || ends_at_state_checkpoint ? PublicationPoint::kOwnerContinues : PublicationPoint::kChunk,
        /*allow_state_checkpoint=*/true);

    std::vector<BlockTable>& tables = request->BlockTablesRef();
    // A chunk reads the state its predecessor wrote (slot 1 -> slot 0) and
    // writes a fresh slot 1. The prompt headroom was prepaid at first-chunk
    // admission.
    const auto chunk_demands = [&](std::int32_t tokens, bool checkpoint) {
        std::vector<GroupDemand> demands =
            MakeGroupDemands(tables, config_.cache_groups, GroupDemand{.extent = DenseGrowth{tokens}},
                             StateStep{.kind = StateStep::Kind::kRotate, .checkpoint = checkpoint});
        ReservePrefillDemands(demands, config_.cache_groups,
                              PrefillReserve{
                                  .decode_input_tokens = reserve_num_tokens_in_next_schedule_event,
                                  .completes_prefill = tokens == unscheduled,
                                  .prompt_headroom_tokens = 0,
                              });
        return demands;
    };
    // As for a first chunk: without room for the in-forward checkpoint's
    // block the chunk ends at the boundary instead.
    std::optional<CacheCoordinator::AdmissionResult> admission;
    if (writes_checkpoint) {
        AdmissionFeedback checkpoint_feedback;
        admission = admitWithKvEventTracking(plan, checkpoint_feedback, *request, cache_progress,
                                             chunk_demands(prefill_tokens, true), progress,
                                             /*evict_blocks_past_segments=*/false);
        if (!admission) {
            writes_checkpoint = false;
            prefill_tokens = state_checkpoint - first_pos;
        }
    }
    if (!admission) {
        admission = admitWithKvEventTracking(plan, feedback, *request, cache_progress,
                                             chunk_demands(prefill_tokens, false), progress,
                                             /*evict_blocks_past_segments=*/true);
    }
    if (!admission) {
        return std::nullopt;
    }
    const bool completes_prefill = prefill_tokens == unscheduled;
    const std::int32_t decode_reserve = completes_prefill ? reserve_num_tokens_in_next_schedule_event : 0;
    if (writes_checkpoint) {
        cache_progress.state_checkpoint_tokens = state_checkpoint;
        cache_progress.state_checkpoint_blocks = std::move(admission->state_checkpoint_blocks);
    }

    request->CacheProgressRef() = std::move(cache_progress);
    return fsm::SchedulePrefillEvent{prefill_tokens, decode_reserve, config_.role == Role::kP};
}

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                                  Request* request) {
    std::vector<BlockTable>& tables = request->BlockTablesRef();
    const std::int32_t num_computed_tokens = request->NumComputedTokens();
    std::int32_t reserve_tokens = request->ReserveNumTokensInNextScheduleEvent();
    if (config_.decode_input_tokens > 1) {
        const std::int32_t decode_width = std::max(config_.decode_input_tokens, reserve_tokens);
        const std::int32_t pending_decode_tokens =
            request->Is<fsm::Decoding>()
                ? std::min(request->ResultsInFlight(), config_.overlap_schedule_depth) * config_.decode_input_tokens
                : 0;
        // Every token-spanned group covers the same logical extent. Reuse
        // reserved speculative slots after a prefill interrupts decode instead
        // of charging another verify span on every restart.
        const std::int32_t verify_end = num_computed_tokens + pending_decode_tokens + decode_width;
        // Draft writes land in the next verify window up to the token limit; admission covers none without a budget.
        const std::int64_t draft_end =
            request->HasDeclaredBudget()
                ? std::min({std::int64_t{verify_end} + CapacityModel::DraftWindowTokens(config_), request->TokenLimit(),
                            std::int64_t{std::numeric_limits<std::int32_t>::max()}})
                : verify_end;
        const auto decode_end = static_cast<std::int32_t>(std::max<std::int64_t>(verify_end, draft_end));
        std::int32_t reserved_end = decode_end;
        for (std::int32_t i = 0; i < coordinator_.NumGroups(); ++i) {
            if (coordinator_.GroupIsState(static_cast<std::size_t>(i))) {
                continue;
            }
            const BlockTable& table = tables[static_cast<std::size_t>(i)];
            reserved_end = table.NumBlocks() * coordinator_.GroupBlockGranularity(i) - table.AvailableTokens();
            break;
        }
        reserve_tokens = std::max(decode_end - reserved_end, 0);
    }
    // The first decode on the fused role is the owner's prompt end: the
    // partial page and the state are published and the owner moves to fresh
    // blocks (the state by rotation: the first decode reads slot 0 and writes
    // the fresh slot 1). The P and D roles publish at finish only -- the P
    // never writes again, the D's prompt-end state arrived by transfer or was
    // claimed. This delivers the preceding prefill's checkpoints; decode
    // never creates an in-forward checkpoint. Every later decode writes slot
    // 1 in place and lets slot 0 go. Restricted groups cannot publish that
    // decode-produced live state at finish or retraction either.
    const bool first_decode = request->Is<fsm::PrefillDone>();
    const bool prompt_end = first_decode && config_.role == Role::kFused && !config_.disable_prefix_cache;
    const StateStep state_step{prompt_end                                 ? StateStep::Kind::kRotate
                               : first_decode && config_.role == Role::kD ? StateStep::Kind::kKeep
                                                                          : StateStep::Kind::kRelease};
    fsm::CacheProgress& current_progress = request->CacheProgressRef();
    const bool completed_page = num_computed_tokens / coordinator_.PrefixGranularity() >
                                static_cast<std::int32_t>(current_progress.prefix_hashes.size());
    const bool completed_checkpoint =
        current_progress.state_checkpoint_tokens > 0 && num_computed_tokens >= current_progress.state_checkpoint_tokens;
    // Between publication points, consuming an existing reserve cannot fail
    // and does not change the hash chain. Keep the rollback copy for admission.
    if (!completed_page && !completed_checkpoint && state_step.kind != StateStep::Kind::kRotate &&
        canConsumeReservedTokensInPlace(coordinator_, tables, reserve_tokens, num_computed_tokens)) {
        coordinator_.ConsumeReservedTokens(tables, reserve_tokens);
        if (state_step.kind == StateStep::Kind::kRelease) {
            coordinator_.ReleaseStateInputs(tables);
        }
        current_progress.published_state_checkpoint_blocks.clear();
        return fsm::ScheduleDecodeEvent{config_.decode_input_tokens};
    }

    fsm::CacheProgress cache_progress = current_progress;
    const RequestProgress progress =
        AdvanceRequestProgress(*request, cache_progress, num_computed_tokens, coordinator_.PrefixGranularity(),
                               config_.StreamsDeviceCacheToHost() && first_decode,
                               prompt_end ? PublicationPoint::kOwnerContinues : PublicationPoint::kChunk,
                               /*allow_state_checkpoint=*/coordinator_.CheckpointGranularity() == 1 || prompt_end);

    std::vector<GroupDemand> demands =
        MakeGroupDemands(tables, config_.cache_groups, GroupDemand{.extent = DenseGrowth{reserve_tokens}}, state_step);
    if (!admitWithKvEventTracking(plan, feedback, *request, cache_progress, demands, progress,
                                  /*evict_blocks_past_segments=*/true)) {
        return std::nullopt;
    }

    request->CacheProgressRef() = std::move(cache_progress);
    return fsm::ScheduleDecodeEvent{config_.decode_input_tokens};
}

PrefillOperation Scheduler::applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                                        std::vector<LoadBackOperation>& load_back_operations) {
    const auto* admission = request->AdmissionResourcesPtr();
    const auto storage_id = admission->storage_hold ? admission->storage_hold->operation_id : 0;
    PrefillOperation operation = applyPrefillEvent(*request, event, coordinator_, cache_group_ids_);
    std::vector<BlockTransfer> load_pairs = event.TakeLoadPairs();
    if (!load_pairs.empty()) {
        load_back_operations.push_back(tier_transfers_.StartPrefixLoad(std::move(load_pairs)));
    }
    // Admission and the ordinary H2D ticket now own the Host sources. The
    // Submitted state was consumed above, so retire its saved Ready hold here.
    if (storage_id) storage_transfers_->Detach(storage_id);
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

std::optional<PrefillOperation> Scheduler::schedulePrefillCandidate(ExecutionPlan& plan, AdmissionFeedback& feedback,
                                                                    Request* request, std::int32_t token_budget,
                                                                    std::int32_t decode_reserve,
                                                                    std::vector<LoadBackOperation>& load_backs,
                                                                    std::int32_t kv_page_budget) {
    const std::int64_t kv_token_limit = config_.max_kv_page_num == 0 || config_.role == Role::kD
                                            ? std::numeric_limits<std::int64_t>::max()
                                            : static_cast<std::int64_t>(kv_page_budget) * config_.prefix_granularity;
    if (request->Is<fsm::Prefilling>()) {
        if (auto event = schedulePrefill(plan, feedback, request, token_budget, decode_reserve, kv_token_limit)) {
            return applyEventAndBuildOperation(request, std::move(*event));
        }
        return std::nullopt;
    }
    FirstChunkOutcome outcome =
        schedulePrefillFirstChunk(plan, feedback, request, token_budget, decode_reserve, kv_token_limit);
    if (outcome.event) {
        PrefillOperation operation = applyEventAndBuildOperation(request, std::move(*outcome.event), load_backs);
        operation.state_prefix_cached = outcome.state_prefix_cached;
        return operation;
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
// Only candidates whose image fits the host budgets (imageShortfall) are
// ranked. When none does, the host side is out of room and the last resort
// is to abort a resident instead of imaging one: the newest retractable
// resident (the least work lost) is returned with its shortfall. Without a
// snapshot pool nothing ever fits, so a capacity block there always ends in
// that abort rather than in a wait that could deadlock once every resident
// needs a page.
//
// Exempt in both tiers, unless `ignore_exemption` is set (the empty-round
// retraction of maybeRetractForCapacity): a request whose reserve already
// covers its whole generation -- retracting it frees exactly what its
// readmission must take back, pure thrash -- and it is not aborted either: it
// completes on its own reserve and frees its pages then. Excluded by state:
// Retracted, Restoring, RemotePrefilling. Transient obstacles (a forward still
// out, a PD transfer pin) do NOT redirect the choice; the caller waits for the
// chosen victim to quiesce rather than sacrificing a worse-ranked request.
Scheduler::VictimChoice Scheduler::chooseVictim(std::span<Request* const> candidates, bool ignore_exemption) const {
    const auto retractable = [ignore_exemption](const Request& request) {
        return request.IsAnyOf<fsm::Prefilling, fsm::PrefillDone, fsm::Decoding>() &&
               (ignore_exemption || !request.ReserveCoversGeneration(kRetractionSafeSteps));
    };
    // Rank first, then probe the host fit in rank order, so the probe (which
    // hashes the candidate's completed pages) runs for as few candidates as
    // possible -- normally one.
    std::vector<Request*> ranked;
    struct DecodeCandidate {
        Request* request;
        std::int32_t releasable_blocks;
    };
    std::vector<DecodeCandidate> decodes;
    Request* newest = nullptr;
    for (Request* request : candidates) {
        if (!retractable(*request)) {
            continue;
        }
        newest = request;  // candidates arrive in submission order
        if (request->Is<fsm::Prefilling>()) {
            ranked.push_back(request);
        } else {
            decodes.push_back(
                DecodeCandidate{request, coordinator_.NumNewlyReleasableLcmBlocks(request->BlockTablesRef())});
        }
    }
    std::ranges::stable_sort(ranked,
                             [](const Request* a, const Request* b) { return a->TokenSize() > b->TokenSize(); });
    const auto decode_rank = [](const DecodeCandidate& candidate) {
        return std::tuple{-candidate.releasable_blocks, candidate.request->GeneratedTokens(), candidate.request->Id()};
    };
    std::ranges::stable_sort(decodes, [&decode_rank](const DecodeCandidate& a, const DecodeCandidate& b) {
        return decode_rank(a) < decode_rank(b);
    });
    for (const DecodeCandidate& candidate : decodes) {
        ranked.push_back(candidate.request);
    }
    VictimChoice newest_choice{.victim = newest};
    for (Request* request : ranked) {
        const std::optional<ImageShortfall> shortfall = imageShortfall(*request);
        if (!shortfall) {
            return VictimChoice{.victim = request};
        }
        if (request == newest) {
            newest_choice.shortfall = shortfall;
        }
    }
    return newest_choice;
}

// The first half of a retraction: hashes the prefix pages the victim
// completed since its last admission and publishes them into the Device index
// -- exactly what its next admission would have published -- then writes the
// progress back. Only what has actually been computed is published (an
// incomplete prefill has only the chunks it has been through; TokenSize()
// would publish pages that were never computed). Device-only: the pages reach
// Host through the image's L2 leg.
void Scheduler::publishCompletedPrefix(Request& victim) {
    fsm::CacheProgress cache_progress = victim.CacheProgress();
    const std::int32_t num_computed_tokens = victim.NumComputedTokens();
    RequestProgress progress =
        AdvanceRequestProgress(victim, cache_progress, num_computed_tokens, coordinator_.PrefixGranularity(),
                               /*stream_completed_to_host=*/false, PublicationPoint::kFinal,
                               /*allow_state_checkpoint=*/coordinator_.CheckpointGranularity() == 1 ||
                                   (config_.role != Role::kD && num_computed_tokens <= victim.PrefillSize()));
    if (progress.completed_pages) {
        // A publication is a KV-event mutation: the newly hashed pages (decode
        // pages no admission has registered yet) and the tail need their token
        // descriptors first, exactly as publishCompletedPages registers before
        // it caches.
        registerKvEventPublication(victim, cache_progress.prefix_hashes, *progress.completed_pages);
        coordinator_.CacheCompletedBlocks(victim.BlockTablesRef(), progress, cache_progress.access_epoch);
    }
    // A suspended request holds no Device block: the index keeps the
    // in-forward State checkpoint this publication may have just published.
    cache_progress.published_state_checkpoint_blocks.clear();
    victim.CacheProgressRef() = std::move(cache_progress);
}

std::optional<ImageShortfall> Scheduler::imageShortfall(const Request& request) const {
    if (!config_.HasSnapshotPool()) {
        return ImageShortfall::kNoPool;
    }
    if (snapshot_slots_.AvailableSlots() == 0) {
        return ImageShortfall::kBlobSlot;
    }
    // The retraction publishes the candidate's completed pages before it
    // images it, and those pages ride the L2 leg. The probe projects that
    // publication on a copy of the progress (hashing the same pages
    // publishCompletedPrefix will) and splits the image off
    // PublishedDataSlotsAfter -- the publication's own slot rule -- so it
    // plans the L2 leg and counts against the pool what StartRetractionStores
    // will, unless the publication itself changes what that plan reads: a
    // tail it drops under the per-prior cap takes its Host entries along;
    // under the write-back policy the segment step ranks Host victims by
    // what the Device holds, which the publication adds to
    // (NumHostBoundariesThatFit); and under KV parallelism a slot it
    // re-points to the Device entry already holding its key rides the pool
    // in that entry's bucket, not the one counted here. The split
    // retractVictim makes after the publication can then refuse the image.
    // Nothing is mutated: a candidate that does not fit is left as it was,
    // and so is the Host.
    fsm::CacheProgress cache_progress = request.CacheProgress();
    const std::int32_t num_computed_tokens = request.NumComputedTokens();
    const RequestProgress progress =
        AdvanceRequestProgress(request, cache_progress, num_computed_tokens, coordinator_.PrefixGranularity(),
                               /*stream_completed_to_host=*/false, PublicationPoint::kFinal,
                               /*allow_state_checkpoint=*/coordinator_.CheckpointGranularity() == 1 ||
                                   (config_.role != Role::kD && num_computed_tokens <= request.PrefillSize()));
    return tier_transfers_
        .SplitRetractionImage(
            request.BlockTablesRef(), num_computed_tokens,
            coordinator_.PublishedDataSlotsAfter(request.BlockTablesRef(), num_computed_tokens, progress))
        .shortfall;
}

// Suspends a quiescent victim with its image. First the completed prefix
// pages are published into the Device index (publishCompletedPrefix: a
// finish-like publication other requests may hit; it costs only the hashes
// of pages not yet hashed, and the image's L2 leg is built from exactly
// those entries). Then the transfer manager splits the image again on what
// the publication left -- the published slots the Host serves or places by
// plan, everything else in the snapshot pool, each in its Device block's
// bucket -- takes it and issues both store legs; the victim's Device pages
// are released by the retract event and may be granted away in this very
// round, because the runtime orders both copies on the forward thread's
// stream ahead of the plan's page reuse. A victim whose image cannot be held,
// or with no blob slot free, is not retracted: the shortfall is returned
// before anything is acquired or evicted for the image, and the publication
// it did stands as its progress. Both callers then abort it rather than let
// it run on: the publication registered blocks the victim would write again (a
// partial page as a tail, and a State group's end state, which a decode step
// writes in place).
std::optional<ImageShortfall> Scheduler::retractVictim(Request& victim, PlanBuild& build,
                                                       std::vector<WriteBackOperation>& write_back_operations) {
    if (snapshot_slots_.AvailableSlots() == 0) {
        return ImageShortfall::kBlobSlot;
    }
    publishCompletedPrefix(victim);
    const std::int32_t num_computed_tokens = victim.NumComputedTokens();
    TierTransferManager::RetractionImageSplit split = tier_transfers_.SplitRetractionImage(
        victim.BlockTablesRef(), num_computed_tokens,
        coordinator_.PublishedDataSlots(victim.BlockTablesRef(), num_computed_tokens));
    if (split.shortfall) {
        return split.shortfall;  // the publication stands; the caller aborts the victim
    }

    // The blob slot is shared with the store that exports into it (and later
    // the restore that imports from it): an abort before either ACK leaves
    // the slot with the op until the copy is done.
    auto blob_slot = std::make_shared<SnapshotSlotIndex>(snapshot_slots_.Allocate());
    std::optional<TierTransferManager::RetractionStores> stores =
        tier_transfers_.StartRetractionStores(victim.Id(), victim.RequestPoolIndex(), blob_slot,
                                              victim.BlockTablesRef(), num_computed_tokens, std::move(split));
    if (!stores) {
        return ImageShortfall::kSnapshotPool;  // unreachable: the split held the image
    }
    victim.NoteRetracted();
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
        case ImageShortfall::kHostL2:
            detail =
                "the Host L2 cache cannot take the image's published pages the snapshot pool cannot hold "
                "(num_host_pages=" +
                std::to_string(config_.host_allocator.total_pages) + ", " +
                std::to_string(coordinator_.NumPinnedHostCachedBlocks()) +
                " Host entries pinned by images and copies, " +
                (tier_transfers_.HasAnyInFlight() ? "copies in flight" : "no copy in flight") +
                "; raise --kvstore-size (or --kvstore-ratio when no size is given))";
            break;
    }
    spdlog::warn(
        "[Scheduler] capacity abort: request {} ({} tokens) cannot be imaged -- {}; aborting it to free its "
        "pages",
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
// cross-round capacity barrier. The one exception is the empty-round
// retraction, when the round runs nothing: its freed pages go first to the
// round's decodes, none of which got a page, and to the blocker only when
// none fits.
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
// a snapshot pool too small for the tail, or a Host L2 that cannot take the
// published pages the pool cannot -- always, without a pool) the last resort
// is to ABORT the newest retractable resident instead of imaging anyone
// (onImageDoesNotFit): its pages free in this round exactly as a
// retraction's would and the grant proceeds. The alternative, waiting, can
// deadlock once every resident needs a page.
//
// A grant that cannot join its round (a fused prefill beside an already-built
// decode batch outside mixed mode) still retracts one victim, and the blocker
// waits for the next round, whose restore phase runs first: the victim's own
// readmission, once its image has landed, can take the freed pages back
// before the blocker tries. So while decodes keep running, the victim can be
// retracted and restored as often as every other round (the redirect below
// does not prevent it), until the decodes finish and the grant joins its
// round.
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
    // sacrificed. (Stream-ordered stores -- both legs of an image, and
    // eviction write-backs -- hold nothing and gate nothing; an in-flight
    // restore gates nothing because its request is Restoring, invisible to
    // chooseVictim.)
    if (tier_transfers_.HasLoadBacksInFlight() || tier_transfers_.HasPinnedStoresInFlight()) {
        return;
    }

    while (true) {
        // No ranked choice can run while every resident is still being
        // written. Avoid hashing and planning images that would all wait.
        // With any quiescent resident, retain the full ranking and fit probe:
        // a better-ranked busy victim must not be bypassed.
        if (std::ranges::none_of(candidates, [this](const Request* request) {
                return request->IsAnyOf<fsm::Prefilling, fsm::PrefillDone, fsm::Decoding>() &&
                       request->ResultsInFlight() == 0 && !pdTransferInFlight(*request);
            })) {
            return;
        }
        VictimChoice choice = chooseVictim(candidates, /*ignore_exemption=*/false);
        // A finished request whose last forward is still out frees its pages
        // when that forward lands (settleFinishedRequests), so a round that
        // waits for it is not empty either.
        const bool empty_round =
            choice.victim == nullptr && build.operations.empty() && build.snapshot_restores.empty() &&
            build.remote_prefill.empty() && build.remote_decode.empty() &&
            std::ranges::none_of(requests_,
                                 [](const std::unique_ptr<Request>& request) { return request->Is<fsm::Finishing>(); });
        if (empty_round) {
            // Nothing is retractable under the exemption and the round runs
            // nothing: the empty-round retraction (docs/design/scheduler.md §2).
            choice = chooseVictim(candidates, /*ignore_exemption=*/true);
        }
        Request* victim = choice.victim;
        if (victim == nullptr) {
            return;  // nothing to retract, or this round's work will free capacity when it completes
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
            // waiting prompt, else to a resident prefill the head-of-line
            // rule kept from trying this round -- granting it back to the
            // victim's own readmission is the loop the grant exists to break.
            // With nobody to serve, retracting it would only cost the image's
            // copies and its restore: it waits for a completion instead. An
            // empty round has no completion to wait for, and its decodes take
            // the freed pages first (below): there the victim is retracted
            // whenever a decode is left to serve.
            const auto blocked = std::ranges::find_if(feedback.capacity_blocked, [victim](Request* request) {
                // An earlier iteration may have retracted one of them.
                return request != victim &&
                       request->IsAnyOf<fsm::Submitted, fsm::Prefilling, fsm::PrefillDone, fsm::Decoding>();
            });
            const auto waiting = std::ranges::find_if(
                candidates, [victim](Request* request) { return request != victim && request->Is<fsm::Submitted>(); });
            // When the victim is a prefill, its chunk failed admission and the
            // head-of-line rule kept every resident prefill behind it from
            // trying this round (a forced victim sitting the round out is
            // marked scheduled and stays out).
            const auto held_back = std::ranges::find_if(candidates, [&build, victim](Request* request) {
                return victim->Is<fsm::Prefilling>() && request != victim && request->Is<fsm::Prefilling>() &&
                       !build.Scheduled(*request);
            });
            const bool decode_left = std::ranges::any_of(candidates, [victim](Request* request) {
                return request != victim && request->IsAnyOf<fsm::PrefillDone, fsm::Decoding>();
            });
            if (blocked != feedback.capacity_blocked.end()) {
                blocker = *blocked;
            } else if (waiting != candidates.end()) {
                blocker = *waiting;
            } else if (held_back != candidates.end()) {
                blocker = *held_back;
            } else if (!empty_round || !decode_left) {
                return;
            }
        }
        // A victim nobody can image, or one retractVictim refuses after
        // publishing, is aborted: the last resort frees its pages all the same.
        std::optional<ImageShortfall> shortfall = choice.shortfall;
        if (!shortfall) {
            shortfall = retractVictim(*victim, build, write_back_operations);
        }
        if (shortfall) {
            onImageDoesNotFit(*victim, *shortfall, build);
        }

        if (empty_round) {
            // The round's decodes, none of which got a page, take the freed
            // pages before the blocker; a victim that is still the blocker had
            // nobody else to serve (docs/design/scheduler.md §2).
            scheduleDecodeBatch(feedback, build, candidates);
            if (build.pushed_decode || blocker == victim) {
                return;
            }
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
                                             build.load_backs, build.kv_page_budget)) {
                if (remote_grant) {
                    // The blocker is now RemotePrefilling: the peer's prefill
                    // is out against its pages (pdTransferInFlight).
                    build.scheduled.insert(blocker);
                    build.remote_prefill.emplace_back(std::move(*admitted));
                } else {
                    pushOperation(build, *blocker, std::move(*admitted));
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
    // The knob is not capacity pressure: a victim the probe refuses runs on
    // with nothing published; one retractVictim refuses after its
    // publication is aborted (see retractVictim).
    if (imageShortfall(*armed)) {
        spdlog::info("[Scheduler] forced retraction of request {} refused: its image does not fit", armed->Id());
        return;
    }
    if (const std::optional<ImageShortfall> shortfall = retractVictim(*armed, build, write_back_operations)) {
        onImageDoesNotFit(*armed, *shortfall, build);
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
    };
    std::vector<BlockTable> tables(static_cast<std::size_t>(coordinator_.NumGroups()));
    // The image rebuilds the State tables' two slots as they were; the first
    // scheduled step after the restore takes its own StateStep.
    std::vector<GroupDemand> demands = MakeGroupDemands(
        tables, config_.cache_groups, GroupDemand{.extent = DenseGrowth{0}}, StateStep{StateStep::Kind::kKeep});
    ReservePrefillDemands(demands, config_.cache_groups, reserve);

    std::optional<CacheCoordinator::AdmissionResult> result =
        coordinator_.Restore(retracted->image, demands, retracted->cache_progress.access_epoch);
    if (!result) {
        feedback.admission_failed = true;
        return false;
    }
    recordAdmittedPages(build.plan, *result);
    // The restore op owns the request-pool row it imports the blob into until
    // its ACK hands it to the resumed state (an abort meanwhile must not
    // re-grant a row still being written), and shares the blob slot.
    SnapshotRestoreOperation op = tier_transfers_.StartSnapshotRestore(
        request->Id(), req_pool_allocator_.Allocate(), retracted->blob_slot, std::move(result->load_pairs),
        std::move(result->snapshot_pairs), retracted->cache_progress.access_epoch);
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
                                                         decode_reserve, build.load_backs, build.kv_page_budget)) {
                pushOperation(build, *request, std::move(*admitted));
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
// replay reserve of a prefill scheduled beside them in mixed mode; on the D
// role decodes consume no budget, so it never binds there.
void Scheduler::scheduleDecodeBatch(AdmissionFeedback& feedback, PlanBuild& build,
                                    std::span<Request* const> candidates) {
    for (Request* request : candidates) {
        if (build.Full(config_.max_batch_size) ||
            build.token_budget < build.pending_prefill_reserve + config_.decode_input_tokens) {
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
                                             config_.decode_input_tokens, build.load_backs, build.kv_page_budget)) {
                build.scheduled.insert(request);
                build.remote_prefill.emplace_back(std::move(*admitted));
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
        build.pending_prefill_reserve = has_local_prefill ? MinPrefillChunkTokens(coordinator_) : 0;
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
    build.kv_page_budget = config_.max_kv_page_num;
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
        .snapshot_stores = std::move(build.snapshot_stores),
        .snapshot_restores = std::move(build.snapshot_restores),
    };
}

}  // namespace tokenspeed
