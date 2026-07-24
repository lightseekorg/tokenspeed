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
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <spdlog/spdlog.h>

#include "cache/forward_cache_ops.h"
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
#include "scheduler/page_hasher.h"
#include "scheduler/request.h"
#include "scheduler/request_spec.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"
#include "utils.h"

namespace tokenspeed {

namespace {

constexpr std::int32_t kLocalMambaSlotsPerRequest = 2;

std::int32_t DecodePagedCacheReservationEnd(std::int32_t first_pos, std::int32_t verify_width,
                                            std::int32_t overlap_depth) {
    if (first_pos < 0 || verify_width < 0 || overlap_depth < 0 || overlap_depth > 1) {
        throw std::invalid_argument("invalid paged-cache decode reservation arguments");
    }
    const std::int64_t reservation_end =
        static_cast<std::int64_t>(first_pos) + static_cast<std::int64_t>(overlap_depth + 1) * verify_width;
    if (reservation_end > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("paged-cache decode reservation exceeds int32 range");
    }
    return static_cast<std::int32_t>(reservation_end);
}

std::int32_t DecodeProtectedReservationTokens(std::int32_t verify_width, std::int32_t overlap_depth) {
    return DecodePagedCacheReservationEnd(/*first_pos=*/0, verify_width, overlap_depth);
}

std::int32_t CheckedTokenCountSum(std::int32_t lhs, std::int32_t rhs) {
    if (lhs < 0 || rhs < 0) {
        throw std::invalid_argument("flat admission token counts must be non-negative");
    }
    const std::int64_t sum = static_cast<std::int64_t>(lhs) + rhs;
    if (sum > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("flat admission token count exceeds int32 range");
    }
    return static_cast<std::int32_t>(sum);
}

std::int32_t CountMambaDeviceLoadBackSlots(const std::vector<TreeNode*>& nodes) {
    std::int32_t slots = 0;
    for (TreeNode* node : nodes) {
        if (node != nullptr && node->HasMambaOnHost() && !node->HasMamba()) {
            ++slots;
        }
    }
    return slots;
}

void AddUniqueNode(std::vector<TreeNode*>& nodes, TreeNode* node) {
    if (node == nullptr) return;
    if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
        nodes.push_back(node);
    }
}

template <typename Op>
static void MaybeFillFlatBlockTables(Op& op, Request* request, std::span<const KvCacheGroupSchema> flat_schema) {
    if (!request->FlatBlockTablesEmpty()) {
        const std::vector<BlockTable>& tables = request->FlatBlockTablesRef();
        if (tables.size() != flat_schema.size()) {
            std::terminate();
        }
        op.flat_block_table_view = tables;
        op.flat_cache_schema = flat_schema;
    }
}

}  // namespace

#if TOKENSPEED_FLAT_KVCACHE
namespace {

// Slide credit is registration-aware: with candidate collection on, blocks the op will register
// get pinned and do not free (count_uncached=false path).
PoolDemand FlatSlideCredit(const KvCacheCoordinator& coordinator, std::span<const BlockTable> tables,
                           std::int32_t num_computed_tokens) {
    return coordinator.BlocksReclaimableByPool(tables, num_computed_tokens,
                                               /*count_uncached=*/!coordinator.HasHostTier());
}

// Decoding/PrefillDone requests hold pool pages a flat retract can release.
bool isFlatHolder(const Request* req) {
    return req->Is<fsm::Decoding>() || req->Is<fsm::PrefillDone>();
}

// Deferred = schedulable states the forward loop skipped this round for lack of pool pages.
bool isFlatDeferred(const Request* req) {
    return isFlatHolder(req) || req->Is<fsm::Submitted>() || req->Is<fsm::PrefetchDone>() || req->Is<fsm::Prefilling>();
}

}  // namespace

bool Scheduler::flatCanAdmit(const Request& request, const PoolDemand& need, const PoolDemand* credit,
                             PoolDemand* shortfall) noexcept {
    const std::size_t pool_count = block_pools_.Size();
    const PoolDemand& reserved = flat_reserved_pages_.Total();
    const PoolDemand& own_reservation = request.FlatReservation().Demand();
    if (need.Size() != pool_count || (credit != nullptr && credit->Size() != pool_count) ||
        reserved.Size() != pool_count || (shortfall != nullptr && shortfall->Size() != pool_count) ||
        own_reservation.Size() != pool_count) {
        std::terminate();
    }
    bool deferred = false;
    for (PoolIndex i = 0; i < pool_count; ++i) {
        const std::int64_t charged = static_cast<std::int64_t>(reserved[i]) - own_reservation[i] + need[i];
        const std::int64_t reclaim = credit == nullptr ? 0 : (*credit)[i];
        const std::int64_t capacity = static_cast<std::int64_t>(block_pools_.Pool(i).NumFreeBlocks()) + reclaim;
        const std::int64_t deficit = std::max<std::int64_t>(charged - capacity, 0);
        if (deficit > std::numeric_limits<std::int32_t>::max()) {
            std::terminate();
        }
        if (shortfall != nullptr) {
            (*shortfall)[i] = static_cast<std::int32_t>(deficit);
        }
        deferred = deferred || deficit != 0;
    }
    return !deferred;
}

void Scheduler::attachFlatKVCompletionInput(Request* request, FlatForwardOperation& op, std::size_t row,
                                            bool apply_fsm_result) {
    if (row >= op.request_ids.size() || row >= op.input_lengths.size() || op.flat_kv_completion_inputs.size() != row ||
        op.flat_kv_completion_inputs.capacity() < op.request_ids.size()) {
        std::terminate();
    }
    std::int32_t dispatch_raw_start = 0;
    const std::int32_t input_length = op.input_lengths[row];
    if (row < op.num_extends()) {
        dispatch_raw_start = op.extend_prefix_lens[row];
    } else {
        const std::size_t decode_row = row - op.num_extends();
        if (decode_row >= op.hist_token_lens.size()) {
            std::terminate();
        }
        // Decode consumes KV rows beginning at hist_token_len. The sampled
        // last token is already present in TokenContainer but has no KV yet.
        const std::int32_t hist_token_len = op.hist_token_lens[decode_row];
        dispatch_raw_start = hist_token_len >= 0 ? hist_token_len : request->TokenSize() - 1;
    }
    const std::int64_t initial_raw_end = static_cast<std::int64_t>(dispatch_raw_start) + input_length;
    if (initial_raw_end > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("flat KV dispatch raw end exceeds int32 range");
    }
    std::int32_t dispatch_raw_end = static_cast<std::int32_t>(initial_raw_end);
    if (dispatch_raw_start < 0 || dispatch_raw_end < dispatch_raw_start) {
        throw std::logic_error("flat KV dispatch has invalid operation raw bounds");
    }
    const FlatKVCompletionRequestSnapshot completion_state = request->FlatCompletionState().Snapshot();
    const std::size_t outstanding = completion_state.outstanding_count;
    if (completion_state.last_dispatch_raw_end.has_value()) {
        // The CPU FSM advances only when completions retire. An overlapped
        // successor is nevertheless a distinct device-side interval, so base
        // it on the newest exported dispatch rather than the stale CPU start.
        dispatch_raw_start = std::max(dispatch_raw_start, *completion_state.last_dispatch_raw_end);
        const std::int64_t projected_end = static_cast<std::int64_t>(dispatch_raw_start) + input_length;
        if (projected_end > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("flat KV dispatch raw end exceeds int32 range");
        }
        dispatch_raw_end = static_cast<std::int32_t>(projected_end);
    }
    const std::size_t overlap_depth = static_cast<std::size_t>(config_.overlap_schedule_depth);
    const std::size_t remaining_overlap = overlap_depth > outstanding ? overlap_depth - outstanding : 0;
    const std::int64_t protected_raw_end =
        static_cast<std::int64_t>(dispatch_raw_end) + static_cast<std::int64_t>(remaining_overlap) * input_length;
    if (protected_raw_end > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("flat KV protected raw end exceeds int32 range");
    }
    FlatKVCompletionInput input = flat_completion_ledger_.RecordDispatch(
        request->FlatCompletionState(), FlatKVDispatchSpec{
                                            .dispatch_raw_start = dispatch_raw_start,
                                            .dispatch_raw_end = dispatch_raw_end,
                                            .protected_raw_end = static_cast<std::int32_t>(protected_raw_end),
                                            .apply_fsm_result = apply_fsm_result,
                                        });
    static_assert(std::is_nothrow_move_constructible_v<FlatKVCompletionInput>);
    op.flat_kv_completion_inputs.push_back(std::move(input));
}

std::vector<std::string> Scheduler::flatPrefixHashesAtAdmission(Request* request) const {
    if (config_.disable_prefix_cache) {
        return {};
    }
    const std::int32_t base_block_size = coordinator_.BaseBlockSize();
    const std::int32_t cap_pages = std::max((request->PrefillSize() - 1) / base_block_size, 0);
    std::vector<std::span<const std::int32_t>> paged_tokens = request->GetFullPagedTokens(/*except_last=*/false);
    if (static_cast<std::size_t>(cap_pages) < paged_tokens.size()) {
        paged_tokens.resize(static_cast<std::size_t>(cap_pages));
    }
    return ComputePagedHashes(paged_tokens, "");
}

// One match, one hash pass at admission: the device match plus its host-tier extension share the
// token math, the gate charge and window.begin. Claiming in-flight pages is stream-ordering safe
// (forward_cache_ops.h).
Scheduler::FlatAdmissionMatch Scheduler::matchFlatPrefixAtAdmission(Request* request) {
    const std::vector<std::string> flat_hashes = flatPrefixHashesAtAdmission(request);
    const std::int32_t base_block_size = coordinator_.BaseBlockSize();
    FlatAdmissionMatch match;
    match.probe = coordinator_.ProbePrefix(flat_hashes);
    // Boundaries are in tokens; the extension hash offsets are in base pages (the granularity the
    // hashes were computed at). No host pool -> host boundary 0 -> empty slice.
    const std::int32_t ext_pages =
        std::max(match.probe.host.num_common_tokens - match.probe.device.num_common_tokens, 0) / base_block_size;
    const auto ext_begin = flat_hashes.begin() + match.probe.device.num_common_tokens / base_block_size;
    match.ext_hashes.assign(ext_begin, ext_begin + ext_pages);
    return match;
}

// Returns the decode-reserve pages to record when admitted (0 unless this chunk completes prefill); nullopt = defer.
std::optional<PoolDemand> Scheduler::flatAdmitFirstChunk(Request* request, const PoolDemand& claim_demand,
                                                         const KvCacheCoordinator::FreshDemandPlan& demand,
                                                         std::int32_t ext_real_pages, PoolDemand* shortfall) {
    // Charge chunk + reserve in one query: unreserved, an exactly-filling prompt's own decode defers forever.
    // ext_real_pages composes exactly: extension pages are FULL, so they leave tail_avail 0.
    PoolDemand blocks_needed = demand.ProtectedDemand();
    blocks_needed.AddInPlace(claim_demand);
    if (ext_real_pages > 0) {
        _assert(block_pools_.Size() == 1, "host extension requires the legacy single device pool");
        PoolDemand extension(block_pools_.Size(), 0);
        extension[0] = ext_real_pages;
        blocks_needed.AddInPlace(extension);
    }
    if (!flatCanAdmit(*request, blocks_needed, nullptr, shortfall)) {
        return std::nullopt;
    }
    return demand.ReservationDemand();
}

// Same contract as flatAdmitFirstChunk, on live tables; num_computed_tokens matches the transition's slide.
std::optional<PoolDemand> Scheduler::flatAdmitPrefillChunk(Request* request, std::int32_t chunk_tokens,
                                                           std::int32_t decode_reserve_tokens,
                                                           std::int32_t num_computed_tokens, PoolDemand* shortfall) {
    std::optional<PoolDemand> slide_credit;
    if (!config_.UsesExplicitFlatPools()) {
        slide_credit.emplace(FlatSlideCredit(coordinator_, request->FlatBlockTablesRef(), num_computed_tokens));
    }
    const std::int32_t protected_tokens = CheckedTokenCountSum(chunk_tokens, decode_reserve_tokens);
    const PoolDemand blocks_needed = coordinator_.BlocksNeededByPool(request->FlatBlockTablesRef(), protected_tokens);
    if (!flatCanAdmit(*request, blocks_needed, slide_credit ? &*slide_credit : nullptr, shortfall)) {
        return std::nullopt;
    }
    PoolDemand reservation(block_pools_.Size(), 0);
    if (decode_reserve_tokens > 0) {
        reservation = blocks_needed;
        reservation.SubtractInPlace(coordinator_.BlocksNeededByPool(request->FlatBlockTablesRef(), chunk_tokens));
    }
    return reservation;
}

// Gate for the PrefillDone reserve Acquire and each DecodeStep, composed from the transition's own primitives.
bool Scheduler::flatAdmitDecode(Request* request, std::size_t outstanding_dispatches, PoolDemand* shortfall) {
    // Same num_computed the transition slides with: Decoding's pending tail is not yet computed.
    const std::int32_t num_computed_tokens =
        request->Is<fsm::Decoding>() ? request->TokenSize() - config_.decode_input_tokens : request->PrefillSize();
    std::optional<PoolDemand> slide_credit;
    if (!config_.UsesExplicitFlatPools()) {
        slide_credit.emplace(FlatSlideCredit(coordinator_, request->FlatBlockTablesRef(), num_computed_tokens));
    }
    const std::size_t max_dispatches = static_cast<std::size_t>(config_.overlap_schedule_depth) + 1;
    const std::size_t outstanding = outstanding_dispatches;
    const std::size_t slots = request->Is<fsm::PrefillDone>()
                                  ? max_dispatches
                                  : (outstanding < max_dispatches ? max_dispatches - outstanding : 0);
    if (slots == 0) {
        return false;
    }
    const std::int32_t protected_tokens =
        decodeReservationTokensForSlots(request->GetReserveNumTokensInNextScheduleEvent(), slots);
    const PoolDemand blocks_needed = coordinator_.BlocksNeededByPool(request->FlatBlockTablesRef(), protected_tokens);
    return flatCanAdmit(*request, blocks_needed, slide_credit ? &*slide_credit : nullptr, shortfall);
}

std::int32_t Scheduler::decodeReservationTokensForSlots(std::int32_t verify_width, std::size_t slots) {
    if (verify_width < 0) {
        throw std::invalid_argument("decode reservation width must be non-negative");
    }
    const std::int64_t tokens = static_cast<std::int64_t>(verify_width) * static_cast<std::int64_t>(slots);
    if (tokens > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("decode reservation exceeds int32 range");
    }
    return static_cast<std::int32_t>(tokens);
}

void Scheduler::resolveFlatStarvation(const std::vector<Request*>& candidates, Request* target,
                                      const PoolDemand& shortfall,
                                      std::span<Request* const> scheduled_requests) noexcept {
    auto reset_starvation = [&]() {
        flat_starved_rounds_ = 0;
        flat_starved_request_id_.clear();
    };
    if (config_.role != Role::kFused || !cache_op_tracker_.empty() || !flat_store_ops_.Empty() ||
        !flat_load_ops_.empty()) {
        reset_starvation();
        return;
    }
    const std::size_t pool_count = block_pools_.Size();
    if (target == nullptr) {
        reset_starvation();
        return;
    }
    if (!isFlatDeferred(target) || shortfall.Size() != pool_count || !shortfall.AnyPositive()) {
        std::terminate();
    }

    const auto scheduled_in_plan = [&](Request* request) {
        return std::binary_search(scheduled_requests.begin(), scheduled_requests.end(), request, std::less<Request*>{});
    };
    const auto has_outstanding = [&](Request* request) {
        if (scheduled_in_plan(request)) {
            return true;
        }
        if (config_.UsesExplicitFlatPools()) {
            return request->FlatCompletionState().HasOutstanding();
        }
        auto it = pending_forward_results_.find(request->Id());
        return it != pending_forward_results_.end() && it->second > 0;
    };
    if (has_outstanding(target)) {
        reset_starvation();
        return;
    }

    struct VictimScore {
        Request* request{};
        std::int64_t bottleneck_blocks{};
        std::int64_t total_bytes{};
    };
    std::optional<VictimScore> victim;
    bool relevant_holder_in_flight = false;
    for (Request* req : candidates) {
        if (!isFlatHolder(req)) {
            continue;
        }
        PoolDemand released = coordinator_.BlocksReleasedByFreeByPool(req->FlatBlockTablesRef());
        released.AddInPlace(req->FlatReservation().Demand());
        std::int64_t bottleneck_blocks = 0;
        for (PoolIndex i = 0; i < pool_count; ++i) {
            if (shortfall[i] > 0) {
                bottleneck_blocks += std::min(released[i], shortfall[i]);
            }
        }
        if (has_outstanding(req)) {
            relevant_holder_in_flight = relevant_holder_in_flight || bottleneck_blocks > 0;
            continue;
        }
        if (bottleneck_blocks == 0) {
            continue;
        }
        VictimScore score{
            .request = req,
            .bottleneck_blocks = bottleneck_blocks,
            .total_bytes = block_pools_.BytesFor(released),
        };
        const bool better = !victim.has_value() || score.bottleneck_blocks > victim->bottleneck_blocks ||
                            (score.bottleneck_blocks == victim->bottleneck_blocks &&
                             (score.total_bytes > victim->total_bytes ||
                              (score.total_bytes == victim->total_bytes &&
                               (score.request->TokenSize() > victim->request->TokenSize() ||
                                (score.request->TokenSize() == victim->request->TokenSize() &&
                                 score.request->Id() < victim->request->Id())))));
        if (better) {
            victim = score;
        }
    }

    if (relevant_holder_in_flight) {
        reset_starvation();
        return;
    }
    if (flat_starved_request_id_ != target->Id()) {
        if (flat_starved_request_id_.capacity() < target->Id().size()) {
            std::terminate();
        }
        flat_starved_request_id_ = target->Id();
        flat_starved_rounds_ = 1;
    } else {
        ++flat_starved_rounds_;
    }
    if (flat_starved_rounds_ < 2) {
        return;
    }
    reset_starvation();

    if (victim.has_value()) {
        Request* request = victim->request;
        request->ResetFlatWriteProgress();
        request->FlatReservation().Clear();
        request->Apply(fsm::FlatRetractEvent{&coordinator_});
        spdlog::info(
            "[Scheduler] flat retract: released request {} "
            "({} bottleneck blocks, {} bytes)",
            request->Id(), victim->bottleneck_blocks, victim->total_bytes);
        return;
    }
    if (isFlatHolder(target) || !flat_oom_terminal_scratch_.empty() ||
        flat_oom_terminal_scratch_.capacity() < target->Id().size() ||
        flat_oom_request_ids_.size() == flat_oom_request_ids_.capacity()) {
        std::terminate();
    }
    flat_oom_terminal_scratch_.assign(target->Id());
    flat_oom_request_ids_.push_back(std::move(flat_oom_terminal_scratch_));
    pending_forward_results_.erase(flat_oom_request_ids_.back());
    target->ResetFlatWriteProgress();
    target->FlatReservation().Clear();
    target->Apply(fsm::AbortEvent{&coordinator_});
}
#endif

#if TOKENSPEED_FLAT_KVCACHE
std::optional<fsm::ScheduleFlatPrefillFirstChunkEvent> Scheduler::scheduleExplicitFlatPrefillFirstChunk(
    Request* request, std::int32_t remaining, std::int32_t decode_input_tokens, PoolDemand* flat_shortfall) {
    _assert(config_.UsesExplicitFlatPools(), "explicit flat first admission requires explicit pools");
    if (req_pool_allocator_.AvailableSlots() == 0) {
        return {};
    }

    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_.Allocate());
    KvCacheCoordinator::PreparedPrefix prepared = coordinator_.PreparePrefix(flatPrefixHashesAtAdmission(request));
    const std::int32_t hit_tokens = prepared.HitTokens();
    const PoolDemand& claim_demand = prepared.ClaimDemand();
    const std::int32_t unscheduled = request->PrefillSize() - hit_tokens;
    const std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t decode_reserve =
        completes_prefill ? DecodeProtectedReservationTokens(decode_input_tokens, config_.overlap_schedule_depth) : 0;
    const std::int32_t protected_tokens = CheckedTokenCountSum(tokens_this_round, decode_reserve);
    KvCacheCoordinator::FreshAllocationPlan allocation =
        coordinator_.PlanFreshAllocation(tokens_this_round, protected_tokens);
    std::optional<PoolDemand> reserve_pages =
        flatAdmitFirstChunk(request, claim_demand, allocation.Demand(), /*ext_real_pages=*/0, flat_shortfall);
    if (!reserve_pages.has_value()) {
        return {};
    }

    const bool owns_reservation = config_.role != Role::kD;
    std::optional<KvCacheCoordinator::CommittedPrefix> committed =
        coordinator_.CommitPreparedPrefix(std::move(prepared), std::move(allocation));
    if (!committed.has_value()) {
        return {};
    }
    if (owns_reservation) {
        _assert(request->FlatReservation().Empty(), "first flat admission found a stale decode reservation");
        request->FlatReservation().Set(std::move(*reserve_pages));
    }

    fsm::FlatCommittedAdmission admission{std::move(req_pool_index), std::move(committed->tables),
                                          committed->hit_tokens,
                                          owns_reservation ? &request->FlatReservation() : nullptr};
    return fsm::ScheduleFlatPrefillFirstChunkEvent{
        tokens_this_round,
        decode_input_tokens,
        config_.role,
        std::move(admission),
    };
}
#endif

std::optional<fsm::SchedulePrefillFirstChunkEvent> Scheduler::schedulePrefillFirstChunk(
    Request* request, std::int32_t remaining, std::int32_t decode_input_tokens, bool disable_l2_cache,
    std::map<std::string, std::int32_t>& simulated_free, [[maybe_unused]] PoolDemand* flat_shortfall) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(!config_.UsesExplicitFlatPools(), "explicit flat admission must use its dedicated helper");
#endif
    if (req_pool_allocator_.AvailableSlots() == 0) return {};
    KVPrefixCache& prefix_cache = radixPrefixCache();
    MatchResult match_result = hybrid_prefix_cache_ ? hybrid_prefix_cache_->Match(request->GetFullPagedTokens(true))
                                                    : prefix_cache.Match(request->GetFullPagedTokens(true));
    std::int32_t loadback_tokens = 0;
    std::int32_t unscheduled = 0;
    std::vector<TreeNode*> loadback_diff;
    std::vector<TreeNode*> mamba_loadback_nodes;

    const std::int32_t device_matched = match_result.device.DepthInPage();
    const std::int32_t host_matched = match_result.host.DepthInPage();
    if (disable_l2_cache) {
        unscheduled = request->PrefillSize() - device_matched * config_.block_size;
    } else {
        loadback_diff = match_result.NodesWithout<ResourceType::Device>();
        if (host_matched > device_matched) {
            loadback_tokens = config_.block_size * (host_matched - device_matched);
        }
        unscheduled = request->PrefillSize() - std::max(device_matched, host_matched) * config_.block_size;
    }

    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && match_result.mamba_branching_seqlen == -1) {
        const std::int32_t aligned = hybrid_prefix_cache_->AlignMambaCacheSeqlen(tokens_this_round);
        if (aligned > 0) {
            match_result.mamba_branching_seqlen = aligned;
        }
    }

#if !TOKENSPEED_FLAT_KVCACHE
    const std::int32_t num_tokens = loadback_tokens + tokens_this_round + decode_input_tokens;
    const std::int32_t device_pages_needed = (num_tokens + config_.block_size - 1) / config_.block_size;
    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);
    // Evict unlocked prefix-cache nodes before allocating request-local pages.
    if (!prefix_cache.EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed)) {
        return {};
    }
#endif

#if TOKENSPEED_FLAT_KVCACHE
    FlatAdmissionMatch flat_match = matchFlatPrefixAtAdmission(request);
    // Overwrite the radix-sourced locals: the radix tree is never written on flat builds.
    const std::int32_t flat_hit_tokens =
        std::max(flat_match.probe.device.num_common_tokens, flat_match.probe.host.num_common_tokens);
    unscheduled = request->PrefillSize() - flat_hit_tokens;
    tokens_this_round = std::min(remaining, unscheduled);

    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t flat_decode_reserve =
        completes_prefill ? DecodeProtectedReservationTokens(decode_input_tokens, config_.overlap_schedule_depth) : 0;
    const std::int32_t flat_protected_tokens = CheckedTokenCountSum(tokens_this_round, flat_decode_reserve);
    KvCacheCoordinator::FreshDemandPlan flat_demand =
        coordinator_.PlanFreshDemand(tokens_this_round, flat_protected_tokens);
    // One pin per real (non-hole) extension slot across all groups = the new device pages the load needs.
    std::int32_t flat_ext_real_pages = 0;
    for (const PrefixProbe& g : flat_match.probe.host.per_group) {
        flat_ext_real_pages += static_cast<std::int32_t>(
            std::ranges::count_if(g.hits, [](CachedBlockState state) { return IsCachedBlockHit(state); }));
    }
    std::optional<PoolDemand> flat_reserve_pages = flatAdmitFirstChunk(
        request, flat_match.probe.device.free_hit_blocks_by_pool, flat_demand, flat_ext_real_pages, flat_shortfall);
    if (!flat_reserve_pages) {
        return {};
    }
#endif

    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && match_result.mamba_host_src_index >= 0 &&
        match_result.mamba_cow_src_index < 0) {
        TreeNode* host_mamba_node = hybrid_prefix_cache_->FindLastMambaHostNode(match_result.host.last_node);
        if (host_mamba_node != nullptr && host_mamba_node->HasMambaOnHost() && !host_mamba_node->HasMamba()) {
            AddUniqueNode(mamba_loadback_nodes, host_mamba_node);
        }
    }
    const bool needs_mamba_loadback = !mamba_loadback_nodes.empty();
    const std::int32_t mamba_loadback_slots_needed =
        needs_mamba_loadback ? CountMambaDeviceLoadBackSlots(mamba_loadback_nodes) : 0;
    const std::int32_t mamba_slots_needed = 2 + mamba_loadback_slots_needed;
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() &&
        !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(mamba_slots_needed)) {
        return {};
    }

    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    const std::int32_t target = first_pos + tokens_this_round;
    if (hybrid_prefix_cache_ &&
        !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free, match_result.paged_cache)) {
        return {};
    }
    if (needs_mamba_loadback) {
        hybrid_prefix_cache_->PrepareMambaDeviceLoadBack(mamba_loadback_nodes);
        TreeNode* mamba_node = hybrid_prefix_cache_->FindLastMambaNode(match_result.host.last_node);
        if (mamba_node != nullptr) {
            match_result.mamba_cow_src_index = mamba_node->MambaSlotIndex();
        }
    }
    if (mamba_allocator_ && mamba_allocator_->AvailableSlots() < kLocalMambaSlotsPerRequest) {
        return {};
    }

#if TOKENSPEED_FLAT_KVCACHE
    KvCacheCoordinator::AdmissionMatch acquired_flat_match = coordinator_.AcquirePrefix(std::move(flat_match.probe));
    // Role kD reaches PrefillDone via RemotePrefillDoneEvent with reserve 0: recording would leave a phantom entry.
    if (config_.role != Role::kD) {
        _assert(request->FlatReservation().Empty(), "first flat admission found a stale decode reservation");
        request->FlatReservation().Set(std::move(*flat_reserve_pages));
    }
#endif
    return fsm::SchedulePrefillFirstChunkEvent{
        tokens_this_round,
        decode_input_tokens,
        &device_allocator_,
        &req_pool_allocator_,
        std::move(match_result),
        config_.role,
        &prefix_cache,
        disable_l2_cache,
        std::move(loadback_diff),
        mamba_allocator_ ? &*mamba_allocator_ : nullptr,
        std::move(mamba_loadback_nodes),
#if TOKENSPEED_FLAT_KVCACHE
        &coordinator_,
        std::move(acquired_flat_match.device),
        std::move(acquired_flat_match.host),
        std::move(flat_match.ext_hashes),
#endif
    };
}

std::optional<fsm::SchedulePrefillEvent> Scheduler::schedulePrefill(
    Request* request, std::int32_t remaining, std::int32_t reserve_num_tokens_in_next_schedule_event,
    std::map<std::string, std::int32_t>& simulated_free, [[maybe_unused]] PoolDemand* flat_shortfall) {
    std::int32_t unscheduled = request->UnScheduledPrefillSize();
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);

#if !TOKENSPEED_FLAT_KVCACHE
    const std::int32_t pages_needed = (tokens_this_round + config_.block_size - 1) / config_.block_size;
    if (!radixPrefixCache().EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
        return {};
    }
#endif

#if TOKENSPEED_FLAT_KVCACHE
    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t flat_decode_reserve =
        completes_prefill ? DecodeProtectedReservationTokens(reserve_num_tokens_in_next_schedule_event,
                                                             config_.overlap_schedule_depth)
                          : 0;
    const std::int32_t flat_num_computed = request->PrefillSize() - unscheduled;
    std::optional<PoolDemand> flat_reserve_pages =
        flatAdmitPrefillChunk(request, tokens_this_round, flat_decode_reserve, flat_num_computed, flat_shortfall);
    if (!flat_reserve_pages) {
        return {};
    }
#endif

    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() &&
        !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(1)) {
        return {};
    }

    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    const std::int32_t target = first_pos + tokens_this_round;
    if (hybrid_prefix_cache_) {
        const std::int32_t commit_target = (first_pos / config_.block_size) * config_.block_size;
        const auto commit_token_pages = request->GetFullPagedTokens(false);
        if (!hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free, {}, commit_target,
                                              commit_token_pages)) {
            return {};
        }
    }

#if TOKENSPEED_FLAT_KVCACHE
    // No kD gate needed: the planning loop never calls schedulePrefill for role kD.
    request->FlatReservation().Set(std::move(*flat_reserve_pages));
#endif

#if TOKENSPEED_FLAT_KVCACHE
    return fsm::SchedulePrefillEvent{tokens_this_round, reserve_num_tokens_in_next_schedule_event, &coordinator_};
#else
    return fsm::SchedulePrefillEvent{tokens_this_round, reserve_num_tokens_in_next_schedule_event,
                                     hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
#endif
}

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(Request* request,
                                                                  std::map<std::string, std::int32_t>& simulated_free,
                                                                  [[maybe_unused]] PoolDemand* flat_shortfall,
                                                                  [[maybe_unused]] std::size_t outstanding_dispatches) {
#if !TOKENSPEED_FLAT_KVCACHE
    const std::int32_t tail_available = request->TailPageAvailableTokens();
    const std::int32_t extra_tokens = std::max(0, request->GetReserveNumTokensInNextScheduleEvent() - tail_available);
    const std::int32_t pages_needed = (extra_tokens + config_.block_size - 1) / config_.block_size;
    if (!radixPrefixCache().EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
        return {};
    }
#endif

#if TOKENSPEED_FLAT_KVCACHE
    if (!flatAdmitDecode(request, outstanding_dispatches, flat_shortfall)) {
        return {};
    }
#endif

    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && mamba_allocator_ &&
        request->Is<fsm::PrefillDone>() && request->GetLocalMambaAllocator() != nullptr &&
        !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(1)) {
        return {};
    }

    const std::int32_t first_pos = request->TokenSize();
    const std::int32_t target =
        DecodePagedCacheReservationEnd(first_pos, config_.decode_input_tokens, config_.overlap_schedule_depth);
    if (hybrid_prefix_cache_) {
        std::optional<std::int32_t> commit_target;
        std::vector<std::span<const std::int32_t>> commit_token_pages;
        if (request->Is<fsm::PrefillDone>()) {
            commit_target = (request->PrefillSize() / config_.block_size) * config_.block_size;
            commit_token_pages = request->GetFullPagedTokens(false);
        }
        if (!hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free, {}, commit_target,
                                              commit_token_pages)) {
            return {};
        }
    }

#if TOKENSPEED_FLAT_KVCACHE
    return fsm::ScheduleDecodeEvent{config_.decode_input_tokens, &coordinator_};
#else
    return fsm::ScheduleDecodeEvent{config_.decode_input_tokens,
                                    hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
#endif
}

#if !TOKENSPEED_FLAT_KVCACHE
std::optional<fsm::ScheduleDecodeFromRetractedEvent> Scheduler::scheduleDecodeFromRetracted(
    Request* request, std::map<std::string, std::int32_t>& simulated_free) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};

    MatchResult match_result =
        hybrid_prefix_cache_
            ? hybrid_prefix_cache_->Match(request->GetFullPagedTokens(true), MatchIntent::StateRecovery)
            : radixPrefixCache().Match(request->GetFullPagedTokens(true), MatchIntent::StateRecovery);
    std::vector<TreeNode*> loadback_diff = match_result.NodesWithout<ResourceType::Device>();
    std::vector<TreeNode*> mamba_loadback_nodes;
    TreeNode* mamba_recovery_node = nullptr;
    bool needs_mamba_loadback = false;
    if (hybrid_prefix_cache_ && mamba_allocator_) {
        mamba_recovery_node = hybrid_prefix_cache_->FindLastMambaNode(match_result.host.last_node);
        if (mamba_recovery_node == nullptr) {
            mamba_recovery_node = hybrid_prefix_cache_->FindLastMambaHostNode(match_result.host.last_node);
            needs_mamba_loadback = mamba_recovery_node != nullptr;
            if (needs_mamba_loadback && !mamba_recovery_node->HasMamba()) {
                AddUniqueNode(mamba_loadback_nodes, mamba_recovery_node);
            }
        }
        if (mamba_recovery_node == nullptr) {
            spdlog::warn("[Scheduler] Retracted request {} lost tree-owned Mamba state, aborting request",
                         request->Id());
            request->Apply(fsm::AbortEvent{
#if TOKENSPEED_FLAT_KVCACHE
                &coordinator_
#endif
            });
            return {};
        }
        if (!needs_mamba_loadback) {
            match_result.mamba_cow_src_index = mamba_recovery_node->MambaSlotIndex();
        }
    }

    const std::int32_t device_matched2 = match_result.device.DepthInPage();
    const std::int32_t host_matched2 = match_result.host.DepthInPage();
    std::int32_t num_tokens = 0;
    if (host_matched2 > device_matched2) {
        num_tokens += (config_.block_size * (host_matched2 - device_matched2)) + config_.decode_input_tokens;
    } else {
        num_tokens += config_.decode_input_tokens;
    }
    std::int32_t device_pages_needed = (num_tokens + config_.block_size - 1) / config_.block_size;

    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);
    if (!radixPrefixCache().EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed)) {
        return {};
    }
    if (hybrid_prefix_cache_ && mamba_allocator_) {
        // Protect the COW source node only for this allocation; retracted Mamba states stay normal evictable entries.
        const std::int32_t mamba_slots_needed = 2 + CountMambaDeviceLoadBackSlots(mamba_loadback_nodes);
        if (!hybrid_prefix_cache_->EnsureMambaCapacityByEvict(mamba_slots_needed, mamba_recovery_node)) {
            return {};
        }
    }

    const std::int32_t first_pos = request->TokenSize() - 1;
    const std::int32_t target = std::max(
        request->TokenSize(),
        DecodePagedCacheReservationEnd(first_pos, config_.decode_input_tokens, config_.overlap_schedule_depth));
    if (hybrid_prefix_cache_ &&
        !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free, match_result.paged_cache)) {
        return {};
    }
    if (needs_mamba_loadback) {
        hybrid_prefix_cache_->PrepareMambaDeviceLoadBack(mamba_loadback_nodes);
        if (mamba_recovery_node->HasMamba()) {
            match_result.mamba_cow_src_index = mamba_recovery_node->MambaSlotIndex();
        }
    }
    if (mamba_allocator_ && mamba_allocator_->AvailableSlots() < kLocalMambaSlotsPerRequest) {
        return {};
    }

    return fsm::ScheduleDecodeFromRetractedEvent{
        config_.decode_input_tokens,
        &device_allocator_,
        &req_pool_allocator_,
        &radixPrefixCache(),
        std::move(match_result),
        loadback_diff,
        mamba_allocator_ ? &*mamba_allocator_ : nullptr,
        std::move(mamba_loadback_nodes),
    };
}

std::optional<fsm::ScheduleRetractEvent> Scheduler::scheduleRetract(Request* request) {
    auto full_paged_tokens = request->GetFullPagedTokens(true);
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(request->GetDeviceNode());
    std::int32_t total_available = static_cast<std::int32_t>(request->GetOccupiedPages().size());

    // Overlap scheduling: ExtendResult may grow the token container early; clamp to the pages we actually have.
    if (total_available < static_cast<std::int32_t>(full_paged_tokens.size())) {
        full_paged_tokens.resize(total_available);
    }

    std::int32_t alloc_count =
        static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());

    // Skip when alloc_count <= 0: a prefix deeper than total_available would make TakeFirstPages negative.
    if (alloc_count > 0) {
        OwnedPages alloc_pages = request->TakeFirstPages(alloc_count);
        radixPrefixCache().Insert<ResourceType::Device>(full_paged_tokens, prefix_pages, std::move(alloc_pages));
    }

    MatchResult match_result = radixPrefixCache().Match(full_paged_tokens, MatchIntent::StateRecovery);

    std::unique_ptr<HostNodeRef> temp_lock = std::make_unique<HostNodeRef>(match_result.host.last_node);
    const std::int32_t device_matched3 = match_result.device.DepthInPage();
    const std::int32_t host_matched3 = match_result.host.DepthInPage();
    std::int32_t host_pages_needed = 0;
    if (device_matched3 > host_matched3) {
        host_pages_needed = device_matched3 - host_matched3;
    }

    if (!radixPrefixCache().EnsureCapacityByEvict<ResourceType::Host>(host_pages_needed)) {
        return {};
    }
    return fsm::ScheduleRetractEvent{&radixPrefixCache(), &host_allocator_, match_result,
                                     hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
}
#endif

LoadBackOperation GenerateLoadBackOp(const std::vector<TreeNode*>& diff, const std::vector<TreeNode*>& mamba_nodes,
                                     cache_op_id op_id) {
    std::vector<TransferPair> transfers;

    for (TreeNode* node : diff) {
        const auto& host_pages = node->Host().Pages();
        const auto& device_pages = node->Device().Pages();
        for (std::size_t i = 0; i < host_pages.size(); ++i) {
            transfers.push_back(TransferPair{CacheKind::kKV, host_pages[i], device_pages[i]});
        }
    }
    for (TreeNode* node : mamba_nodes) {
        if (node != nullptr && node->HasMambaOnHost() && node->HasMamba()) {
            transfers.push_back(TransferPair{CacheKind::kMamba, node->MambaHostSlotIndex(), node->MambaSlotIndex()});
        }
    }
    return LoadBackOperation{op_id, std::move(transfers)};
}

#if !TOKENSPEED_FLAT_KVCACHE
std::optional<WriteBackOperation> Scheduler::applyEventAndGenerateOp(Request* request,
                                                                     fsm::ScheduleRetractEvent event) {
    request->Apply(std::move(event));

    const auto& pages_to_transfer = request->GetPagesToTransfer<fsm::Retracting>();
    if (pages_to_transfer.empty()) {
        // No copy needed; advance Retracting to Retracted without an op_id.
        request->Apply(
            fsm::WriteBackDoneEvent{&radixPrefixCache(), hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr});
        return std::nullopt;
    }
    cache_op_id op_id = radixPrefixCache().AllocateCacheOpId();
    CacheOpSpec spec;
    spec.request_id = request->Id();
    cache_op_tracker_[op_id] = std::move(spec);
    return WriteBackOperation{op_id, std::vector<TransferPair>(pages_to_transfer.begin(), pages_to_transfer.end()),
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
        retract_request->Apply(fsm::AbortEvent{
#if TOKENSPEED_FLAT_KVCACHE
            &coordinator_
#endif
        });
    }
    return std::nullopt;
}
#endif

#if !TOKENSPEED_FLAT_KVCACHE
void Scheduler::finalizeRadixPageTableEmission(Request* request, ForwardOperationBase& op, bool force_full) {
    // Without a hybrid cache, local pages are never published into the radix tree
    // during an active forward lease, so the builder's append-only delta is exact.
    if (!hybrid_prefix_cache_) return;

    if (op.request_pool_index <= 0 ||
        static_cast<std::size_t>(op.request_pool_index) >= radix_page_table_emissions_.size()) {
        throw std::logic_error("Scheduler::finalizeRadixPageTableEmission: invalid request-pool index=" +
                               std::to_string(op.request_pool_index));
    }

    RadixPageTableEmission& previous = radix_page_table_emissions_[op.request_pool_index];
    const std::int32_t current_size = static_cast<std::int32_t>(op.occupied_pages.size());
    const std::int32_t current_prefix = request->GetDeviceNode()->DepthInPage(config_.block_size);
    if (current_prefix < 0 || current_prefix > current_size) {
        throw std::logic_error("Scheduler::finalizeRadixPageTableEmission: invalid radix prefix size=" +
                               std::to_string(current_prefix) + "; page-table size=" + std::to_string(current_size));
    }

    std::int32_t begin = 0;
    if (!force_full && previous.prefix_pages >= 0) {
        const std::int32_t previous_size =
            previous.prefix_pages + static_cast<std::int32_t>(previous.local_pages.size());

        // The emitted table is an immutable, pinned radix prefix followed by an
        // append-only local tail. Publication can replace ids only in that old
        // tail. If a lifecycle or ordering invariant changes, refresh safely.
        const bool valid_incremental =
            current_prefix >= previous.prefix_pages && op.begin == previous_size && current_size >= previous_size;
        if (valid_incremental) {
            auto mismatch = std::mismatch(previous.local_pages.begin(), previous.local_pages.end(),
                                          op.occupied_pages.begin() + previous.prefix_pages,
                                          op.occupied_pages.begin() + previous_size);
            begin = op.begin;
            if (mismatch.first != previous.local_pages.end()) {
                begin = previous.prefix_pages +
                        static_cast<std::int32_t>(std::distance(previous.local_pages.begin(), mismatch.first));
            }
        }
    }

    op.begin = begin;
    op.size = current_size - begin;
    previous.prefix_pages = current_prefix;
    previous.local_pages.assign(op.occupied_pages.begin() + current_prefix, op.occupied_pages.end());
}
#endif

// By-reference so the first-chunk caller can harvest the transition's flat load pairs afterwards.
template <typename Event>
    requires(std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent> || std::same_as<Event, fsm::SchedulePrefillEvent>
#if TOKENSPEED_FLAT_KVCACHE
             || std::same_as<Event, fsm::ScheduleFlatPrefillFirstChunkEvent>
#endif
             )
static PrefillOperation applyPrefillEvent(Request* request, Event& event,
                                          std::span<const KvCacheGroupSchema> flat_schema) {
    // begin/size are PAGE-space: the req_to_page refresh slice for this operation.
    // The builder starts with appended pages; radix finalization may move begin
    // backward when publication canonicalizes an already-emitted physical page.
    // A first-chunk prefix hit enters during the event, so begin stays 0 and size counts the hit rows too;
    // the op's token-space INPUT window intentionally starts past the hit.
    // Multi-group Flat execution consumes only the explicit keyed tables.
    // Keep the scalar group-0 mirror solely for the single-group compatibility
    // ABI; exporting it for heterogeneous groups is ambiguous and copies an
    // O(sequence) page vector on every dispatch.
#if TOKENSPEED_FLAT_KVCACHE
    const bool emit_legacy_occupied_pages = flat_schema.size() == 1;
#else
    const bool emit_legacy_occupied_pages = true;
#endif
    const std::int32_t begin =
        emit_legacy_occupied_pages ? static_cast<std::int32_t>(request->GetOccupiedPages().size()) : 0;
    request->Apply(event);
    std::vector<std::int32_t> all_pages =
        emit_legacy_occupied_pages ? request->GetOccupiedPages() : std::vector<std::int32_t>{};
    const std::int32_t sz = static_cast<std::int32_t>(all_pages.size()) - begin;

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

    MaybeFillFlatBlockTables(op, request, flat_schema);

    return op;
}

#if TOKENSPEED_FLAT_KVCACHE
PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleFlatPrefillFirstChunkEvent event) {
    return applyPrefillEvent(request, event, FlatCacheSchema());
}
#endif

// TODO(radix-removal): the #if !TOKENSPEED_FLAT_KVCACHE publishing arms in these op-builders go with the radix path.
PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                                    std::vector<LoadBackOperation>& loadback_ops) {
#if !TOKENSPEED_FLAT_KVCACHE
    auto match = event.GetMatchResult();
#endif
    auto op = applyPrefillEvent(request, event, FlatCacheSchema());
#if TOKENSPEED_FLAT_KVCACHE
    // Host-loaded pages ride the same LoadBackOperation channel as radix loadbacks.
    std::vector<BlockTransfer> load_pairs = event.TakeFlatLoadPairs();
    if (!load_pairs.empty()) {
        std::vector<TransferPair> transfers;
        transfers.reserve(load_pairs.size());
        FlatLoadTicket ticket;
        ticket.host_pins.reserve(load_pairs.size());
        ticket.device_blocks.reserve(load_pairs.size());
        for (BlockTransfer& pair : load_pairs) {
            _assert(pair.source->IsCached(), "pinned host page lost its hash before load emission");
            transfers.push_back(TransferPair{CacheKind::kKV, pair.source->BlockId(), pair.destination->BlockId()});
            ticket.host_pins.push_back(std::move(pair.source));
            ticket.device_blocks.push_back(std::move(pair.destination));
        }
        const cache_op_id op_id = radixPrefixCache().AllocateCacheOpId();
        flat_load_ops_.emplace(op_id, std::move(ticket));
        loadback_ops.push_back(LoadBackOperation{op_id, std::move(transfers)});
    }
#else
    (void)loadback_ops;
#endif
#if !TOKENSPEED_FLAT_KVCACHE
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct()) {
        op.mamba_cow_src_idx = match.mamba_cow_src_index;
        op.mamba_branching_seqlen = match.mamba_branching_seqlen;
    }
    // CommitChunk before acquire: prior-chunk tail pages must commit into snapshots before ReleaseSkipped frees them.
    if (hybrid_prefix_cache_) {
        hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.extend_prefix_len,
                                                op.extend_prefix_len + op.input_length, match.paged_cache);
        hybrid_prefix_cache_->PopulateOp(op);
    }
    finalizeRadixPageTableEmission(request, op, /*force_full=*/true);
#endif
    return op;
}

PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event) {
    auto op = applyPrefillEvent(request, event, FlatCacheSchema());
#if !TOKENSPEED_FLAT_KVCACHE
    if (hybrid_prefix_cache_) {
        hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.extend_prefix_len,
                                                op.extend_prefix_len + op.input_length);
        hybrid_prefix_cache_->PopulateOp(op);
    }
    finalizeRadixPageTableEmission(request, op, /*force_full=*/false);
#endif
    return op;
}

template <typename Event>
    requires(std::same_as<Event, fsm::ScheduleDecodeEvent>
#if !TOKENSPEED_FLAT_KVCACHE
             || std::same_as<Event, fsm::ScheduleDecodeFromRetractedEvent>
#endif
             )
static DecodeOperation applyDecodeEvent(Request* request, Event event, std::int32_t decode_input_tokens,
                                        std::span<const KvCacheGroupSchema> flat_schema) {
#if TOKENSPEED_FLAT_KVCACHE
    const bool emit_legacy_occupied_pages = flat_schema.size() == 1;
#else
    const bool emit_legacy_occupied_pages = true;
#endif
    const std::int32_t begin =
        emit_legacy_occupied_pages ? static_cast<std::int32_t>(request->GetOccupiedPages().size()) : 0;
    request->Apply(std::move(event));
    std::vector<std::int32_t> all_pages =
        emit_legacy_occupied_pages ? request->GetOccupiedPages() : std::vector<std::int32_t>{};
    const std::int32_t sz = static_cast<std::int32_t>(all_pages.size()) - begin;

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

    MaybeFillFlatBlockTables(op, request, flat_schema);

    return op;
}

DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event,
                                                   [[maybe_unused]] std::size_t outstanding_dispatches) {
    const bool need_bootstrap_token = request->Is<fsm::PrefillDone>() && config_.role == Role::kD;
    std::int32_t bootstrap_token = need_bootstrap_token ? request->GetLastToken() : -1;
    const bool came_from_prefill_done = request->Is<fsm::PrefillDone>();
#if !TOKENSPEED_FLAT_KVCACHE
    const std::int32_t first_pos = request->TokenSize();
#endif
#if TOKENSPEED_FLAT_KVCACHE
    const std::int32_t scheduled_reserve_tokens = request->GetReserveNumTokensInNextScheduleEvent();
    const std::size_t outstanding_before = outstanding_dispatches;
#endif

    auto op = applyDecodeEvent(request, std::move(event), config_.decode_input_tokens, FlatCacheSchema());
    if (need_bootstrap_token) {
        op.decode_input_id = bootstrap_token;
    }
#if TOKENSPEED_FLAT_KVCACHE
    const std::size_t max_dispatches = static_cast<std::size_t>(config_.overlap_schedule_depth) + 1;
    const std::size_t remaining_slots =
        came_from_prefill_done
            ? static_cast<std::size_t>(config_.overlap_schedule_depth)
            : (outstanding_before + 1 < max_dispatches ? max_dispatches - (outstanding_before + 1) : 0);
    if (remaining_slots == 0) {
        request->FlatReservation().Clear();
    } else {
        const std::int32_t remaining_tokens =
            decodeReservationTokensForSlots(scheduled_reserve_tokens, remaining_slots);
        PoolDemand reservation = coordinator_.BlocksNeededByPool(request->FlatBlockTablesRef(), remaining_tokens);
        request->FlatReservation().Set(std::move(reservation));
    }
#endif
#if !TOKENSPEED_FLAT_KVCACHE
    if (hybrid_prefix_cache_) {
        if (came_from_prefill_done) {
            hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        }
        const std::int32_t target =
            DecodePagedCacheReservationEnd(first_pos, op.input_length, config_.overlap_schedule_depth);
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, first_pos, target);
        hybrid_prefix_cache_->PopulateOp(op);
    }
    finalizeRadixPageTableEmission(request, op, /*force_full=*/false);
#endif
    return op;
}

#if !TOKENSPEED_FLAT_KVCACHE
DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent event) {
    const std::int32_t mamba_cow_src_index = event.GetMatchResult().mamba_cow_src_index;
#if !TOKENSPEED_FLAT_KVCACHE
    auto paged_cache_hit = event.GetMatchResult().paged_cache;
#endif
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
    op.mamba_cow_src_idx = mamba_cow_src_index;

    auto* mamba = request->GetLocalMambaAllocator();
    if (mamba != nullptr && mamba->HasWorking()) {
        op.mamba_working_idx = mamba->WorkingIndex();
        if (mamba->HasCheckpoint()) {
            op.mamba_checkpoint_dst_idx = mamba->CheckpointIndex();
        }
    }

#if !TOKENSPEED_FLAT_KVCACHE
    if (hybrid_prefix_cache_) {
        const std::int32_t target = std::max(
            request->TokenSize(),
            DecodePagedCacheReservationEnd(op.hist_token_len, op.input_length, config_.overlap_schedule_depth));
        // Preserve the existing table across retraction. Its request-local
        // tail contains state after the last published prefix checkpoint and
        // cannot be reconstructed by importing that older snapshot alone.
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.hist_token_len, target, paged_cache_hit);
        hybrid_prefix_cache_->PopulateOp(op);
    }
    finalizeRadixPageTableEmission(request, op, /*force_full=*/true);
#endif

    MaybeFillFlatBlockTables(op, request, FlatCacheSchema());

    return op;
}
#endif

std::tuple<std::vector<ForwardOperation>, std::variant<std::vector<LoadBackOperation>, std::vector<WriteBackOperation>>>
Scheduler::newForwardOperation(std::vector<Request*> candidates) {
    auto priority = [&](const Request* req) -> int {
        if (req->Is<fsm::Prefilling>()) return 1;
        if (req->Is<fsm::Submitted>()) return 2;
        if (req->Is<fsm::Decoding>() || req->Is<fsm::PrefillDone>()) {
            // Decode-first if mixed-batch is enabled; prefill-first otherwise.
            return config_.enable_mixed_prefill_decode ? 0 : 3;
        }
#if !TOKENSPEED_FLAT_KVCACHE
        if (req->Is<fsm::Retracted>()) return 4;
#endif
        return 9;
    };
    // TP-determinism: tie-break on Id() so every rank schedules the same subset (a rank-varying op deadlocks NCCL).
    std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
        int pa = priority(a), pb = priority(b);
        return pa != pb ? pa < pb : a->Id() < b->Id();
    });

    const std::size_t scheduled_capacity =
        std::min(candidates.size(), static_cast<std::size_t>(config_.max_batch_size));
#if TOKENSPEED_FLAT_KVCACHE
    const std::size_t pool_count = block_pools_.Size();
    const bool uses_explicit_flat_pools = config_.UsesExplicitFlatPools();
    PoolDemand candidate_shortfall(pool_count, 0);
    PoolDemand starvation_shortfall(pool_count, 0);
    Request* starvation_target = nullptr;
    std::vector<Request*> scheduled_requests;
    scheduled_requests.reserve(scheduled_capacity);
#endif
    std::vector<ForwardOperation> ops;
    ops.reserve(scheduled_capacity);
    std::int32_t token_budget = config_.max_scheduled_tokens;
    bool pushed_prefill = false;
    auto push_op = [&](Request* request, auto op) {
        if (config_.role != Role::kD) {
            token_budget -= op.input_length;
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(op)>, PrefillOperation>) {
            pushed_prefill = true;
        }
#if TOKENSPEED_FLAT_KVCACHE
        scheduled_requests.push_back(request);
#else
        (void)request;
#endif
        ops.push_back(std::move(op));
    };
    std::vector<LoadBackOperation> loadback_ops;
    loadback_ops.reserve(scheduled_capacity);
    auto simulated_free =
        hybrid_prefix_cache_ ? hybrid_prefix_cache_->InitialSimulatedFree() : std::map<std::string, std::int32_t>{};
    for (Request* request : candidates) {
        if (token_budget <= 0 || config_.max_batch_size == ops.size()) break;
        std::size_t outstanding_dispatches = 0;
#if TOKENSPEED_FLAT_KVCACHE
        PoolDemand* flat_shortfall = starvation_target == nullptr ? &candidate_shortfall : nullptr;
        if (uses_explicit_flat_pools && request->HasPendingFlatTerminal()) {
            continue;
        }
        const std::size_t reservation_window = static_cast<std::size_t>(config_.overlap_schedule_depth) + 1;
        if (uses_explicit_flat_pools) {
            const FlatKVCompletionRequestSnapshot completion_state = request->FlatCompletionState().Snapshot();
            outstanding_dispatches = completion_state.outstanding_count;
            if (completion_state.has_canceled_outstanding) {
                continue;
            }
            if (outstanding_dispatches > reservation_window) {
                std::terminate();
            }
            if (outstanding_dispatches == reservation_window) {
                continue;
            }
        } else if (const auto pending = pending_forward_results_.find(request->Id());
                   pending != pending_forward_results_.end()) {
            if (pending->second <= 0) {
                std::terminate();
            }
            // Main's legacy Flat path tracks completion debt for lifetime but
            // never uses it as a dispatch fence. Clamp only the reservation
            // horizon so old callers keep scheduling without over-reserving.
            outstanding_dispatches = std::min(static_cast<std::size_t>(pending->second), reservation_window - 1);
        }
#else
        PoolDemand* flat_shortfall = nullptr;
#endif

        if (request->Is<fsm::Prefilling>() && config_.role != Role::kD) {
            std::int32_t reserver_num_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            if (auto ev = schedulePrefill(request, token_budget, reserver_num_tokens, simulated_free, flat_shortfall)) {
                push_op(request, applyEventAndGenerateOp(request, *ev));
            }
        } else if (request->Is<fsm::Submitted>() || request->Is<fsm::PrefetchDone>()) {
            // PrefetchDone: host cache populated; treat same as Submitted for forward scheduling.
            std::int32_t decode_input_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;

            // Role D only reserves the remote-prefill destination. A partial
            // first chunk cannot be completed locally, so admit the whole
            // prompt atomically without applying the prefill compute budget.
            const std::int32_t prefill_budget = config_.role == Role::kD ? request->PrefillSize() : token_budget;
#if TOKENSPEED_FLAT_KVCACHE
            if (uses_explicit_flat_pools) {
                if (auto ev = scheduleExplicitFlatPrefillFirstChunk(request, prefill_budget, decode_input_tokens,
                                                                    flat_shortfall)) {
                    push_op(request, applyEventAndGenerateOp(request, std::move(*ev)));
                }
            } else
#endif
            {
                if (auto ev = schedulePrefillFirstChunk(request, prefill_budget, decode_input_tokens,
                                                        config_.disable_l2_cache, simulated_free, flat_shortfall)) {
                    std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                    std::vector<TreeNode*> mamba_loadback_nodes = ev->GetMambaLoadbackNodes();
                    push_op(request, applyEventAndGenerateOp(request, std::move(*ev), loadback_ops));
                    if (!loadback_diff.empty() || !mamba_loadback_nodes.empty()) {
                        const cache_op_id op_id = radixPrefixCache().AllocateCacheOpId();
                        loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, mamba_loadback_nodes, op_id));
                    }
                }
            }
        } else if (request->Is<fsm::PrefillDone>() || (request->Is<fsm::Decoding>() && config_.role != Role::kP)) {
            // Mixed-batch disabled: skip ALL decode once a prefill was scheduled.
            if (!config_.enable_mixed_prefill_decode && pushed_prefill) break;

            if (auto ev = scheduleDecode(request, simulated_free, flat_shortfall, outstanding_dispatches)) {
                push_op(request, applyEventAndGenerateOp(request, *ev, outstanding_dispatches));
            }
#if !TOKENSPEED_FLAT_KVCACHE
        } else if (request->Is<fsm::Retracted>() && config_.role != Role::kP) {
            if (!config_.enable_mixed_prefill_decode && pushed_prefill) break;

            if (auto ev = scheduleDecodeFromRetracted(request, simulated_free)) {
                std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                std::vector<TreeNode*> mamba_loadback_nodes = ev->GetMambaLoadbackNodes();
                push_op(request, applyEventAndGenerateOp(request, std::move(*ev)));
                if (!loadback_diff.empty() || !mamba_loadback_nodes.empty()) {
                    cache_op_id op_id = radixPrefixCache().AllocateCacheOpId();
                    loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, mamba_loadback_nodes, op_id));
                }
            }
#endif
        }
#if TOKENSPEED_FLAT_KVCACHE
        if (starvation_target == nullptr && isFlatDeferred(request) && candidate_shortfall.AnyPositive()) {
            starvation_target = request;
            for (PoolIndex i = 0; i < pool_count; ++i) {
                starvation_shortfall[i] = candidate_shortfall[i];
            }
        }
#endif
    }

#if TOKENSPEED_FLAT_KVCACHE
    std::sort(scheduled_requests.begin(), scheduled_requests.end(), std::less<Request*>{});
    resolveFlatStarvation(candidates, starvation_target, starvation_shortfall, scheduled_requests);
#else
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
#endif

    return {std::move(ops), std::move(loadback_ops)};
}

}  // namespace tokenspeed
