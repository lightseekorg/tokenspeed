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
static void MaybeFillFlatBlockTables(Op& op, Request* request, std::span<const std::string> flat_group_ids) {
    if (!request->FlatBlockTablesEmpty()) {
        op.flat_block_tables = BuildFlatBlockTables(request->FlatBlockTablesRef(), flat_group_ids);
    }
}

template <ResourceType RType>
TreeNode* LastNodeWithResourceOrRoot(TreeNode* node) {
    for (TreeNode* candidate = node; candidate != nullptr; candidate = candidate->Parent()) {
        if (candidate->IsRoot()) return candidate;
        if constexpr (RType == ResourceType::Device) {
            if (candidate->OnDevice()) {
                return candidate;
            }
        } else {
            if (candidate->OnHost()) {
                return candidate;
            }
        }
    }
    return nullptr;
}

}  // namespace

#if TOKENSPEED_FLAT_KVCACHE
namespace {

// Slide credit is registration-aware: with candidate collection on, blocks the op will register
// get pinned and do not free (count_uncached=false path).
std::int32_t FlatSlideCredit(const KvCacheCoordinator& coordinator, std::span<const BlockTable> tables,
                             std::int32_t num_computed_tokens) {
    std::int32_t total_freed = 0;
    for (std::int32_t i = 0; i < coordinator.NumGroups(); ++i) {
        total_freed +=
            coordinator.GroupManager(i).BlocksReclaimableAt(tables[static_cast<std::size_t>(i)], num_computed_tokens,
                                                            /*count_uncached=*/!coordinator.HasHostTier());
    }
    return total_freed;
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

// One match, one hash pass at admission: the device match plus its host-tier extension share the
// token math, the gate charge and window.begin. Claiming in-flight pages is stream-ordering safe
// (forward_cache_ops.h).
Scheduler::FlatAdmissionMatch Scheduler::matchFlatPrefixAtAdmission(Request* request) {
    if (config_.disable_prefix_cache) {
        FlatAdmissionMatch match;
        match.probe = coordinator_.ProbePrefix({});
        return match;
    }
    // Hash input must be byte-identical to the REGISTRATION form (GetFullPagedTokens(false)); radix's
    // except_last rule (last prompt token recomputed for logits) becomes the page cap, also bounding SWA.
    // These offsets index the base-granular hash array (GetFullPagedTokens is cut at base).
    const std::int32_t base_block_size = coordinator_.BaseBlockSize();
    const std::int32_t cap_pages = std::max((request->PrefillSize() - 1) / base_block_size, 0);
    std::vector<std::span<const std::int32_t>> paged_tokens = request->GetFullPagedTokens(/*except_last=*/false);
    if (static_cast<std::size_t>(cap_pages) < paged_tokens.size()) {
        paged_tokens.resize(cap_pages);
    }
    const std::vector<std::string> flat_hashes = ComputePagedHashes(paged_tokens, "");
    FlatAdmissionMatch match;
    match.probe = coordinator_.ProbePrefix(flat_hashes);
    // Boundaries are in tokens; the extension hash offsets are in base pages (the granularity the
    // hashes were computed at). No host pool -> host boundary 0 -> empty slice.
    const std::int32_t ext_pages =
        std::max(match.probe.host.num_common_tokens - match.probe.device.num_common_tokens, 0) / base_block_size;
    const auto ext_begin = flat_hashes.begin() + match.probe.device.num_common_tokens / base_block_size;
    match.ext_hashes.assign(ext_begin, ext_begin + ext_pages);
    match.hashes = flat_hashes;
    return match;
}

// Returns the decode-reserve pages to record when admitted (0 unless this chunk completes prefill); nullopt = defer.
std::optional<std::int32_t> Scheduler::flatAdmitFirstChunk(Request* request, std::int32_t device_free_hit_blocks,
                                                           std::int32_t ext_real_pages, std::int32_t chunk_tokens,
                                                           std::int32_t decode_reserve_tokens) const {
    // Charge chunk + reserve in one query: unreserved, an exactly-filling prompt's own decode defers forever.
    // ext_real_pages composes exactly: extension pages are FULL, so they leave tail_avail 0.
    const std::int32_t blocks_needed = coordinator_.BlocksNeededFor(chunk_tokens + decode_reserve_tokens);
    // Device probe hits remain in the free list until admission commits; charge
    // the entries AcquirePrefix will remove. Host hits need fresh device pages.
    if (blocks_needed + device_free_hit_blocks + ext_real_pages > flatFreeBudget(request->Id())) {
        return std::nullopt;
    }
    // Reserve need is computed on the post-prefill table shape now, never recomputed against drifted state.
    return decode_reserve_tokens > 0 ? blocks_needed - coordinator_.BlocksNeededFor(chunk_tokens) : 0;
}

// Same contract as flatAdmitFirstChunk, on live tables; num_computed_tokens matches the transition's slide.
std::optional<std::int32_t> Scheduler::flatAdmitPrefillChunk(Request* request, std::int32_t chunk_tokens,
                                                             std::int32_t decode_reserve_tokens,
                                                             std::int32_t num_computed_tokens) const {
    const std::int32_t slide_credit = FlatSlideCredit(coordinator_, request->FlatBlockTablesRef(), num_computed_tokens);
    const std::int32_t blocks_needed =
        coordinator_.BlocksNeededFor(request->FlatBlockTablesRef(), chunk_tokens + decode_reserve_tokens);
    if (blocks_needed > flatFreeBudget(request->Id()) + slide_credit) {
        return std::nullopt;
    }
    // The pending slide cannot drift the reserve: ReclaimExpired punches front holes, BlocksNeededFor reads tail_avail.
    return decode_reserve_tokens > 0
               ? blocks_needed - coordinator_.BlocksNeededFor(request->FlatBlockTablesRef(), chunk_tokens)
               : 0;
}

// Gate for the PrefillDone reserve Acquire and each DecodeStep, composed from the transition's own primitives.
bool Scheduler::flatAdmitDecode(Request* request) const {
    // Same num_computed the transition slides with: Decoding's pending tail is not yet computed.
    const std::int32_t num_computed_tokens =
        request->Is<fsm::Decoding>() ? request->TokenSize() - config_.decode_input_tokens : request->PrefillSize();
    const std::int32_t slide_credit = FlatSlideCredit(coordinator_, request->FlatBlockTablesRef(), num_computed_tokens);
    const std::int32_t blocks_needed =
        coordinator_.BlocksNeededFor(request->FlatBlockTablesRef(), request->GetReserveNumTokensInNextScheduleEvent());
    return blocks_needed <= flatFreeBudget(request->Id()) + slide_credit;
}

// True when this round can never unwedge itself; fused-only (PD requests hold pages outside both ledgers).
bool Scheduler::flatPoolWedged(const std::vector<Request*>& candidates) const {
    const bool any_deferred = std::any_of(candidates.begin(), candidates.end(), isFlatDeferred);
    // Block 0 is the null placeholder, never allocated.
    const bool pool_pages_held = block_pool_.NumFreeBlocks() < block_pool_.TotalBlocks() - 1;
    // An in-flight D2H store OR H2D load still holds pool pages its Done event will free; both
    // ledgers must be empty. Dispatched mid-prefill chunk ops are invisible here and safe (they
    // free no pool pages, emit no event, and page reuse under them is stream-ordering safe).
    const bool nothing_in_flight = pending_forward_results_.empty() && cache_op_tracker_.empty() &&
                                   flat_store_ops_.Empty() && flat_load_ops_.empty();
    return config_.role == Role::kFused && any_deferred && pool_pages_held && nothing_in_flight;
}

// Wedge resolution, escalating on the SECOND consecutive wedged round (an in-flight Finish fakes one):
// flat-retract the largest Decoding/PrefillDone holder, or -- with no holder -- OOM-terminalize the
// head-of-line deferred non-holder. Sole owner of flat_starved_rounds_.
void Scheduler::resolveFlatStarvation(const std::vector<Request*>& candidates, bool made_progress) {
    if (made_progress || !flatPoolWedged(candidates)) {
        flat_starved_rounds_ = 0;
        return;
    }
    if (++flat_starved_rounds_ < 2) {
        return;
    }
    flat_starved_rounds_ = 0;
    std::vector<Request*> holders;
    for (Request* req : candidates) {
        if (isFlatHolder(req)) {
            holders.push_back(req);
        }
    }
    if (!holders.empty()) {
        Request* victim = *std::max_element(holders.begin(), holders.end(), [](const Request* a, const Request* b) {
            return a->TokenSize() < b->TokenSize();
        });
        // Wedge-gate guarantee (flatPoolWedged: pending_forward_results_.empty()): the runtime owes
        // the victim nothing, so its re-prefill can never race a stale ExtendResult. If the gate is
        // ever relaxed, fail loud here instead of corrupting the rebased prefill.
        _assert(pending_forward_results_.find(victim->Id()) == pending_forward_results_.end(),
                "retract victim must not owe a forward result");
        flat_reserved_pages_.erase(victim->Id());
        victim->Apply(fsm::FlatRetractEvent{&coordinator_});
        spdlog::info("[Scheduler] flat retract: released request {} ({} tokens) to unwedge the pool", victim->Id(),
                     victim->TokenSize());
        return;
    }
    // No retract victim: the pages are held mid-prefill (a wedged holder, or a mutual wedge among
    // prefilling requests). Terminalize the head-of-line deferred non-holder to unwedge the pool.
    for (Request* req : candidates) {
        if (isFlatDeferred(req) && !isFlatHolder(req)) {
            req->Apply(fsm::AbortEvent{&coordinator_});
            flat_oom_request_ids_.push_back(req->Id());
            spdlog::warn(
                "[Scheduler] flat OOM: pool wedged by unretractable mid-prefill holders; terminalized request {}",
                req->Id());
            return;
        }
    }
    _assert(false, "wedged with no holder and no deferred non-holder");
}
#endif

std::optional<fsm::SchedulePrefillFirstChunkEvent> Scheduler::schedulePrefillFirstChunk(
    Request* request, std::int32_t remaining, std::int32_t decode_input_tokens, bool disable_l2_cache,
    std::map<std::string, std::int32_t>& simulated_free) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};
    MatchResult match_result = hybrid_prefix_cache_ ? hybrid_prefix_cache_->Match(request->GetFullPagedTokens(true))
                                                    : kv_prefix_cache_.Match(request->GetFullPagedTokens(true));
    std::int32_t loadback_tokens = 0;
    std::int32_t unscheduled = 0;
    std::vector<TreeNode*> loadback_diff;
    std::vector<TreeNode*> mamba_loadback_nodes;
    std::vector<PagedCacheTransferPair> paged_cache_loadback_transfers;

    const std::int32_t device_matched = match_result.device.DepthInPage();
    const std::int32_t host_matched = match_result.host.DepthInPage();
    const bool has_paged_cache = hybrid_prefix_cache_ && hybrid_prefix_cache_->HasPagedCacheAdjunct();
    const std::int32_t paged_device_matched = match_result.paged_cache.prefix_len_tokens;
    const std::int32_t paged_host_matched = disable_l2_cache ? 0 : match_result.paged_cache_host.prefix_len_tokens;
    const bool use_paged_host_hit = has_paged_cache && paged_host_matched > paged_device_matched;
    std::int32_t matched_prefix_len_tokens = 0;
    if (disable_l2_cache) {
        matched_prefix_len_tokens = device_matched * config_.block_size;
        unscheduled = request->PrefillSize() - matched_prefix_len_tokens;
    } else {
        loadback_diff = match_result.NodesWithout<ResourceType::Device>();
        if (host_matched > device_matched) {
            loadback_tokens = config_.block_size * (host_matched - device_matched);
        }
        matched_prefix_len_tokens = has_paged_cache ? std::max(paged_device_matched, paged_host_matched)
                                                    : std::max(device_matched, host_matched) * config_.block_size;
        unscheduled = request->PrefillSize() - matched_prefix_len_tokens;
    }
    if (unscheduled < 0) {
        unscheduled = 0;
    }

    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && match_result.mamba_branching_seqlen == -1) {
        const std::int32_t aligned = hybrid_prefix_cache_->AlignMambaCacheSeqlen(tokens_this_round);
        if (aligned > 0) {
            match_result.mamba_branching_seqlen = aligned;
        }
    }

    std::int32_t num_tokens = loadback_tokens + tokens_this_round + decode_input_tokens;
    std::int32_t device_pages_needed = (num_tokens + config_.block_size - 1) / config_.block_size;

    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(
        LastNodeWithResourceOrRoot<ResourceType::Device>(match_result.device.last_node));

    // Evict unlocked prefix-cache nodes before allocating request-local pages.
    if (!(kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed))) {
        return {};
    }

#if TOKENSPEED_FLAT_KVCACHE
    FlatAdmissionMatch flat_match = matchFlatPrefixAtAdmission(request);
    // Overwrite the radix-sourced locals: the radix tree is never written on flat builds.
    const std::int32_t flat_hit_tokens =
        std::max(flat_match.probe.device.num_common_tokens, flat_match.probe.host.num_common_tokens);
    unscheduled = request->PrefillSize() - flat_hit_tokens;
    tokens_this_round = std::min(remaining, unscheduled);

    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t flat_decode_reserve = completes_prefill ? decode_input_tokens : 0;
    // One pin per real (non-hole) extension slot across all groups = the new device pages the load needs.
    std::int32_t flat_ext_real_pages = 0;
    for (const PrefixProbe& g : flat_match.probe.host.per_group) {
        flat_ext_real_pages += static_cast<std::int32_t>(std::ranges::count(g.hits, std::uint8_t{1}));
    }
    const std::optional<std::int32_t> flat_reserve_pages =
        flatAdmitFirstChunk(request, flat_match.probe.device.num_free_hit_blocks, flat_ext_real_pages,
                            tokens_this_round, flat_decode_reserve);
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

    const std::int32_t first_pos = disable_l2_cache ? request->PrefillSize() - unscheduled : matched_prefix_len_tokens;
    const std::int32_t target = first_pos + tokens_this_round;
    const MatchResult::PagedCache empty_paged_hit{};
    const MatchResult::PagedCache& paged_hit_for_admission =
        use_paged_host_hit ? empty_paged_hit : match_result.paged_cache;
    if (hybrid_prefix_cache_ &&
        !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free, paged_hit_for_admission)) {
        return {};
    }
    if (use_paged_host_hit) {
        paged_cache_loadback_transfers =
            hybrid_prefix_cache_->PreparePagedCacheDeviceLoadBack(request->Id(), match_result.paged_cache_host);
        if (paged_cache_loadback_transfers.empty()) {
            return {};
        }
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
    KvCacheCoordinator::AdmissionMatch acquired_flat_match =
        coordinator_.AcquirePrefix(flat_match.hashes, std::move(flat_match.probe));
    // Role kD reaches PrefillDone via RemotePrefillDoneEvent with reserve 0: recording would leave a phantom entry.
    if (*flat_reserve_pages > 0 && config_.role != Role::kD) {
        flat_reserved_pages_[request->Id()] = *flat_reserve_pages;
    }
#endif
    return fsm::SchedulePrefillFirstChunkEvent{
        tokens_this_round,
        decode_input_tokens,
        &device_allocator_,
        &req_pool_allocator_,
        std::move(match_result),
        config_.role,
        &kv_prefix_cache_,
        disable_l2_cache,
        std::move(loadback_diff),
        hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr,
        mamba_allocator_ ? &*mamba_allocator_ : nullptr,
        std::move(mamba_loadback_nodes),
#if TOKENSPEED_FLAT_KVCACHE
        &coordinator_,
        std::move(acquired_flat_match.device),
        std::move(acquired_flat_match.host),
        std::move(flat_match.ext_hashes),
#endif
        std::move(paged_cache_loadback_transfers),
        matched_prefix_len_tokens,
    };
}

std::optional<fsm::SchedulePrefillEvent> Scheduler::schedulePrefill(
    Request* request, std::int32_t remaining, std::int32_t reserve_num_tokens_in_next_schedule_event,
    std::map<std::string, std::int32_t>& simulated_free) {
    std::int32_t unscheduled = request->UnScheduledPrefillSize();
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);

    std::int32_t pages_needed = (tokens_this_round + config_.block_size - 1) / config_.block_size;

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
        return {};
    }

#if TOKENSPEED_FLAT_KVCACHE
    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t flat_decode_reserve = completes_prefill ? reserve_num_tokens_in_next_schedule_event : 0;
    const std::int32_t flat_num_computed = request->PrefillSize() - unscheduled;
    const std::optional<std::int32_t> flat_reserve_pages =
        flatAdmitPrefillChunk(request, tokens_this_round, flat_decode_reserve, flat_num_computed);
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
    if (*flat_reserve_pages > 0) {
        flat_reserved_pages_[request->Id()] = *flat_reserve_pages;
    }
#endif

    return fsm::SchedulePrefillEvent{tokens_this_round, reserve_num_tokens_in_next_schedule_event,
                                     hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr
#if TOKENSPEED_FLAT_KVCACHE
                                     ,
                                     &coordinator_
#endif
    };
}

Scheduler::ScheduleAttempt<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(
    Request* request, std::map<std::string, std::int32_t>& simulated_free) {
    std::int32_t tail_available = request->TailPageAvailableTokens();
    std::int32_t extra_tokens = std::max(0, request->GetReserveNumTokensInNextScheduleEvent() - tail_available);
    std::int32_t pages_needed = (extra_tokens + config_.block_size - 1) / config_.block_size;

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
        return {};
    }

#if TOKENSPEED_FLAT_KVCACHE
    if (!flatAdmitDecode(request)) {
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
            return {.event = std::nullopt, .failure = ScheduleFailure::kPagedCache};
        }
    }

    return {.event = fsm::ScheduleDecodeEvent{config_.decode_input_tokens,
                                              hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr
#if TOKENSPEED_FLAT_KVCACHE
                                              ,
                                              &coordinator_
#endif
            },
            .failure = ScheduleFailure::kNone};
}

Scheduler::ScheduleAttempt<fsm::ScheduleDecodeFromRetractedEvent> Scheduler::scheduleDecodeFromRetracted(
    Request* request, std::map<std::string, std::int32_t>& simulated_free) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};

    MatchResult match_result =
        hybrid_prefix_cache_
            ? hybrid_prefix_cache_->Match(request->GetFullPagedTokens(true), MatchIntent::StateRecovery)
            : kv_prefix_cache_.Match(request->GetFullPagedTokens(true), MatchIntent::StateRecovery);
    std::vector<TreeNode*> loadback_diff = match_result.NodesWithout<ResourceType::Device>();
    std::vector<TreeNode*> mamba_loadback_nodes;
    std::vector<PagedCacheTransferPair> paged_cache_loadback_transfers;
    const bool has_paged_cache = hybrid_prefix_cache_ && hybrid_prefix_cache_->HasPagedCacheAdjunct();
    const std::int32_t paged_device_matched = match_result.paged_cache.prefix_len_tokens;
    const std::int32_t paged_host_matched =
        config_.disable_l2_cache ? 0 : match_result.paged_cache_host.prefix_len_tokens;
    const bool use_paged_host_hit = has_paged_cache && paged_host_matched > paged_device_matched;
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
            abortRequest(request, "Retracted request lost tree-owned Mamba state");
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

    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(
        LastNodeWithResourceOrRoot<ResourceType::Device>(match_result.device.last_node));
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed)) {
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
    const MatchResult::PagedCache empty_paged_hit{};
    const MatchResult::PagedCache& paged_hit_for_admission =
        use_paged_host_hit ? empty_paged_hit : match_result.paged_cache;
    if (hybrid_prefix_cache_ &&
        !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free, paged_hit_for_admission)) {
        return {.event = std::nullopt, .failure = ScheduleFailure::kPagedCache};
    }
    if (use_paged_host_hit) {
        if (!hybrid_prefix_cache_->HasRequestPagedCacheTables(request->Id())) {
            paged_cache_loadback_transfers =
                hybrid_prefix_cache_->PreparePagedCacheDeviceLoadBack(request->Id(), match_result.paged_cache_host);
            if (paged_cache_loadback_transfers.empty()) {
                return {};
            }
        }
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

    return {.event =
                fsm::ScheduleDecodeFromRetractedEvent{
                    config_.decode_input_tokens,
                    &device_allocator_,
                    &req_pool_allocator_,
                    &kv_prefix_cache_,
                    std::move(match_result),
                    std::move(loadback_diff),
                    mamba_allocator_ ? &*mamba_allocator_ : nullptr,
                    std::move(mamba_loadback_nodes),
                    std::move(paged_cache_loadback_transfers),
                },
            .failure = ScheduleFailure::kNone};
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
        kv_prefix_cache_.Insert<ResourceType::Device>(full_paged_tokens, prefix_pages, std::move(alloc_pages));
    }

    MatchResult match_result = kv_prefix_cache_.Match(full_paged_tokens, MatchIntent::StateRecovery);

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
    return fsm::ScheduleRetractEvent{&kv_prefix_cache_, &host_allocator_, std::move(match_result),
                                     hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
}

LoadBackOperation Scheduler::newLoadBackOperation(const std::string& request_id, const std::vector<TreeNode*>& diff,
                                                  const std::vector<TreeNode*>& mamba_nodes,
                                                  std::vector<PagedCacheTransferPair> paged_cache_transfers,
                                                  TreeNode* paged_cache_host_node) {
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

    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
    if (!paged_cache_transfers.empty()) {
        if (paged_cache_host_node == nullptr || !paged_cache_host_node->OnHost()) {
            throw std::logic_error("paged-cache loadback requires a host-resident snapshot node");
        }
        CacheOpSpec spec;
        spec.request_id = request_id;
        spec.paged_loadback_host_pin = std::make_unique<HostNodeRef>(paged_cache_host_node);
        cache_op_tracker_[op_id] = std::move(spec);
    }
    return LoadBackOperation{op_id, std::move(transfers), std::move(paged_cache_transfers)};
}

std::optional<WriteBackOperation> Scheduler::applyEventAndGenerateOp(Request* request,
                                                                     fsm::ScheduleRetractEvent event) {
    request->Apply(std::move(event));

    const auto& pages_to_transfer = request->GetPagesToTransfer<fsm::Retracting>();
    const auto& paged_cache_transfers = request->GetPagedCacheWriteBackTransfers<fsm::Retracting>();
    if (pages_to_transfer.empty() && paged_cache_transfers.empty()) {
        // No copy needed; advance Retracting to Retracted without an op_id.
        request->Apply(
            fsm::WriteBackDoneEvent{&kv_prefix_cache_, hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr});
        return std::nullopt;
    }
    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
    CacheOpSpec spec;
    spec.request_id = request->Id();
    spec.paged_cache_nodes = request->GetPagedCacheWriteBackNodes<fsm::Retracting>();
    cache_op_tracker_[op_id] = std::move(spec);
    return WriteBackOperation{
        op_id, std::vector<TransferPair>(pages_to_transfer.begin(), pages_to_transfer.end()),
        std::vector<PagedCacheTransferPair>(paged_cache_transfers.begin(), paged_cache_transfers.end()), true};
}

bool Scheduler::hasInFlightCacheOp(const std::string& request_id) const {
    return std::any_of(cache_op_tracker_.begin(), cache_op_tracker_.end(),
                       [&](const auto& item) { return item.second.request_id == request_id; });
}

void Scheduler::deferAbort(const std::string& request_id, bool discard_writeback, std::string scheduler_message) {
    auto& deferred = deferred_aborts_[request_id];
    if (discard_writeback) {
        // A runtime-originated abort is already terminal to the client. It wins over
        // an internal abort that had not yet become observable.
        deferred.discard_writeback = true;
        deferred.scheduler_message.clear();
    } else if (!deferred.discard_writeback && deferred.scheduler_message.empty()) {
        deferred.scheduler_message = std::move(scheduler_message);
    }
}

void Scheduler::tryFinalizeDeferredAbort(const std::string& request_id) {
    if (deferred_aborts_.empty()) return;

    auto deferred_it = deferred_aborts_.find(request_id);
    if (deferred_it == deferred_aborts_.end() || pending_forward_results_.contains(request_id) ||
        hasInFlightCacheOp(request_id)) {
        return;
    }

    if (Request* request = find_request(request_id); request != nullptr && !request->Is<fsm::Finished>()) {
        request->Apply(fsm::AbortEvent{&kv_prefix_cache_, hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr
#if TOKENSPEED_FLAT_KVCACHE
                                       ,
                                       &coordinator_
#endif
        });
    }
    if (hybrid_prefix_cache_) {
        hybrid_prefix_cache_->ReleaseRequest(request_id);
    }

    std::string scheduler_message = std::move(deferred_it->second.scheduler_message);
    deferred_aborts_.erase(deferred_it);
    if (!scheduler_message.empty()) {
        spdlog::warn("[Scheduler] Aborting request {}: {}", request_id, scheduler_message);
        scheduler_aborts_.push_back(SchedulerAbort{request_id, std::move(scheduler_message)});
    }
}

void Scheduler::abortRequest(Request* request, std::string message) {
    if (request == nullptr || request->Is<fsm::Finished>()) return;

    const std::string request_id = request->Id();
    deferAbort(request_id, /*discard_writeback=*/false, std::move(message));
    tryFinalizeDeferredAbort(request_id);
}

std::optional<WriteBackOperation> Scheduler::newRetractOperation(Request* retract_request) {
    if (auto event = scheduleRetract(retract_request)) {
        if (auto op = applyEventAndGenerateOp(retract_request, std::move(*event))) {
            return std::move(*op);
        }
    } else {
        abortRequest(retract_request, "Retract failed because host cache capacity is exhausted");
    }
    return std::nullopt;
}

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
    requires(std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent> || std::same_as<Event, fsm::SchedulePrefillEvent>)
static PrefillOperation applyPrefillEvent(Request* request, Event& event, std::span<const std::string> flat_group_ids) {
    // begin/size are PAGE-space: the req_to_page refresh slice for this operation.
    // The builder starts with appended pages; radix finalization may move begin
    // backward when publication canonicalizes an already-emitted physical page.
    // A first-chunk prefix hit enters during the event, so begin stays 0 and size counts the hit rows too;
    // the op's token-space INPUT window intentionally starts past the hit.
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

    MaybeFillFlatBlockTables(op, request, flat_group_ids);

    return op;
}

// TODO(radix-removal): the #if !TOKENSPEED_FLAT_KVCACHE publishing arms in these op-builders go with the radix path.
PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                                    std::vector<LoadBackOperation>& loadback_ops) {
#if !TOKENSPEED_FLAT_KVCACHE
    auto match = event.GetMatchResult();
#endif
    auto op = applyPrefillEvent(request, event, FlatGroupIds());
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
        const cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
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
    auto op = applyPrefillEvent(request, event, FlatGroupIds());
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
    requires(std::same_as<Event, fsm::ScheduleDecodeEvent> ||
             std::same_as<Event, fsm::ScheduleDecodeFromRetractedEvent>)
static DecodeOperation applyDecodeEvent(Request* request, Event event, std::int32_t decode_input_tokens,
                                        std::span<const std::string> flat_group_ids) {
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

    MaybeFillFlatBlockTables(op, request, flat_group_ids);

    return op;
}

DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event) {
    const bool need_bootstrap_token = request->Is<fsm::PrefillDone>() && config_.role == Role::kD;
    std::int32_t bootstrap_token = need_bootstrap_token ? request->GetLastToken() : -1;
    const bool came_from_prefill_done = request->Is<fsm::PrefillDone>();
#if !TOKENSPEED_FLAT_KVCACHE
    const std::int32_t first_pos = request->TokenSize();
#endif

    auto op = applyDecodeEvent(request, std::move(event), config_.decode_input_tokens, FlatGroupIds());
    if (need_bootstrap_token) {
        op.decode_input_id = bootstrap_token;
    }
#if TOKENSPEED_FLAT_KVCACHE
    // FinalizePrefillAndReserveDecode just acquired the promised decode reserve: retire the ledger entry.
    if (came_from_prefill_done) {
        flat_reserved_pages_.erase(op.request_id);
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

DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent& event) {
    const MatchResult& match = event.GetMatchResult();
    const std::int32_t mamba_cow_src_index = match.mamba_cow_src_index;
#if !TOKENSPEED_FLAT_KVCACHE
    const bool has_paged_cache_loadback = !event.GetPagedCacheLoadbackTransfers().empty();
#endif
    request->Apply(event);
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
        if (!has_paged_cache_loadback && !hybrid_prefix_cache_->HasRequestPagedCacheTables(op.request_id)) {
            hybrid_prefix_cache_->ReleaseRequest(op.request_id);
        }
        // Preserve the existing table across retraction. Its request-local
        // tail contains state after the last published prefix checkpoint and
        // cannot be reconstructed by importing that older snapshot alone.
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.hist_token_len, target, match.paged_cache);
        hybrid_prefix_cache_->PopulateOp(op);
    }
    finalizeRadixPageTableEmission(request, op, /*force_full=*/true);
#endif

    MaybeFillFlatBlockTables(op, request, FlatGroupIds());

    return op;
}

std::tuple<std::vector<ForwardOperation>, std::variant<std::vector<LoadBackOperation>, std::vector<WriteBackOperation>>>
Scheduler::newForwardOperation(std::vector<Request*> candidates) {
    auto priority = [&](const Request* req) -> int {
        if (req->Is<fsm::Prefilling>()) return 1;
        if (req->Is<fsm::Submitted>()) return 2;
        if (req->Is<fsm::Decoding>() || req->Is<fsm::PrefillDone>()) {
            // Decode-first if mixed-batch is enabled; prefill-first otherwise.
            return config_.enable_mixed_prefill_decode ? 0 : 3;
        }
        if (req->Is<fsm::Retracted>()) return 4;
        return 9;
    };
    // TP-determinism: tie-break on Id() so every rank schedules the same subset (a rank-varying op deadlocks NCCL).
    std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
        int pa = priority(a), pb = priority(b);
        return pa != pb ? pa < pb : a->Id() < b->Id();
    });

    std::vector<ForwardOperation> ops;
    std::int32_t token_budget = config_.max_scheduled_tokens;
    bool pushed_prefill = false;
    auto push_op = [&](auto op) {
        if (config_.role != Role::kD) {
            token_budget -= op.input_length;
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(op)>, PrefillOperation>) {
            pushed_prefill = true;
        }
        ops.push_back(std::move(op));
    };
    // Mid-prefill chunk ops emit no ExtendResult; only decode and prefill-completing ops owe one.
    auto note_result_owed = [&](Request* request) {
        if (!request->Is<fsm::Prefilling>()) {
            ++pending_forward_results_[request->Id()];
        }
    };
    std::vector<LoadBackOperation> loadback_ops;
    std::vector<Request*> paged_cache_blocked;
    bool all_decode_failures_are_paged_cache = true;
    auto record_decode_failure = [&](Request* request, ScheduleFailure failure) {
        if (request->Is<fsm::Finished>() || failure != ScheduleFailure::kPagedCache) {
            all_decode_failures_are_paged_cache = false;
            return;
        }
        paged_cache_blocked.push_back(request);
    };
    auto simulated_free =
        hybrid_prefix_cache_ ? hybrid_prefix_cache_->InitialSimulatedFree() : std::map<std::string, std::int32_t>{};
    for (Request* request : candidates) {
        if (token_budget <= 0 || config_.max_batch_size == ops.size()) break;

        if (request->Is<fsm::Prefilling>() && config_.role != Role::kD) {
            std::int32_t reserver_num_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            if (auto ev = schedulePrefill(request, token_budget, reserver_num_tokens, simulated_free)) {
                push_op(applyEventAndGenerateOp(request, *ev));
                note_result_owed(request);
            }
        } else if (request->Is<fsm::Submitted>() || request->Is<fsm::PrefetchDone>()) {
            // PrefetchDone: host cache populated; treat same as Submitted for forward scheduling.
            std::int32_t decode_input_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;

            // Role D only reserves the remote-prefill destination. A partial
            // first chunk cannot be completed locally, so admit the whole
            // prompt atomically without applying the prefill compute budget.
            const std::int32_t prefill_budget = config_.role == Role::kD ? request->PrefillSize() : token_budget;
            if (auto ev = schedulePrefillFirstChunk(request, prefill_budget, decode_input_tokens,
                                                    config_.disable_l2_cache, simulated_free)) {
                TreeNode* paged_cache_host_node = ev->GetMatchResult().paged_cache_host.last_node;
                std::vector<TreeNode*> loadback_diff = ev->TakeLoadbackDiff();
                std::vector<TreeNode*> mamba_loadback_nodes = ev->TakeMambaLoadbackNodes();
                std::vector<PagedCacheTransferPair> paged_cache_loadback_transfers =
                    ev->TakePagedCacheLoadbackTransfers();
                push_op(applyEventAndGenerateOp(request, std::move(*ev), loadback_ops));
                note_result_owed(request);
                // will be empty when disable_l2_cache
                if (!loadback_diff.empty() || !mamba_loadback_nodes.empty() ||
                    !paged_cache_loadback_transfers.empty()) {
                    loadback_ops.push_back(newLoadBackOperation(request->Id(), loadback_diff, mamba_loadback_nodes,
                                                                std::move(paged_cache_loadback_transfers),
                                                                paged_cache_host_node));
                }
            }
        } else if (request->Is<fsm::PrefillDone>() || (request->Is<fsm::Decoding>() && config_.role != Role::kP)) {
            // Mixed-batch disabled: skip ALL decode once a prefill was scheduled.
            if (!config_.enable_mixed_prefill_decode && pushed_prefill) break;

            auto attempt = scheduleDecode(request, simulated_free);
            if (attempt.event) {
                push_op(applyEventAndGenerateOp(request, *attempt.event));
                note_result_owed(request);
            } else {
                record_decode_failure(request, attempt.failure);
            }
        } else if (request->Is<fsm::Retracted>() && config_.role != Role::kP) {
            if (!config_.enable_mixed_prefill_decode && pushed_prefill) break;

            auto attempt = scheduleDecodeFromRetracted(request, simulated_free);
            if (attempt.event) {
                TreeNode* paged_cache_host_node = attempt.event->GetMatchResult().paged_cache_host.last_node;
                push_op(applyEventAndGenerateOp(request, *attempt.event));
                note_result_owed(request);
                std::vector<TreeNode*> loadback_diff = attempt.event->TakeLoadbackDiff();
                std::vector<TreeNode*> mamba_loadback_nodes = attempt.event->TakeMambaLoadbackNodes();
                std::vector<PagedCacheTransferPair> paged_cache_loadback_transfers =
                    attempt.event->TakePagedCacheLoadbackTransfers();
                if (!loadback_diff.empty() || !mamba_loadback_nodes.empty() ||
                    !paged_cache_loadback_transfers.empty()) {
                    loadback_ops.push_back(newLoadBackOperation(request->Id(), loadback_diff, mamba_loadback_nodes,
                                                                std::move(paged_cache_loadback_transfers),
                                                                paged_cache_host_node));
                }
            } else {
                record_decode_failure(request, attempt.failure);
            }
        }
    }

#if TOKENSPEED_FLAT_KVCACHE
    resolveFlatStarvation(candidates, /*made_progress=*/!ops.empty());
#else
    const bool no_async_progress_pending = pending_forward_results_.empty() && cache_op_tracker_.empty();
    if (ops.empty() && no_async_progress_pending && all_decode_failures_are_paged_cache &&
        !paged_cache_blocked.empty()) {
        Request* victim = paged_cache_blocked.front();
        for (Request* request : paged_cache_blocked) {
            if (request->TokenSize() > victim->TokenSize() ||
                (request->TokenSize() == victim->TokenSize() && request->Id() < victim->Id())) {
                victim = request;
            }
        }
        abortRequest(victim, "Paged cache group capacity is exhausted");
        return {std::vector<ForwardOperation>{}, std::move(loadback_ops)};
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
#endif

    return {std::move(ops), std::move(loadback_ops)};
}

}  // namespace tokenspeed
