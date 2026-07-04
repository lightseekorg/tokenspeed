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

}  // namespace

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
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && match_result.mamba_branching_seqlen == -1) {
        const std::int32_t aligned = hybrid_prefix_cache_->AlignMambaCacheSeqlen(tokens_this_round);
        if (aligned > 0) {
            match_result.mamba_branching_seqlen = aligned;
        }
    }

    std::int32_t num_tokens = loadback_tokens + tokens_this_round + decode_input_tokens;
    std::int32_t device_pages_needed = (num_tokens + config_.page_size - 1) / config_.page_size;

    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);

    // Evict unlocked prefix-cache nodes before allocating request-local pages.
    if (!(kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed))) {
        return {};
    }

#if TOKENSPEED_FLAT_KVCACHE
    // M9: cross-request prefix hit. One match at admission (vLLM V1 shape);
    // the SAME num_common_blocks drives the token math here, the gate charge
    // below, and window.begin in the FSM event -- single source, so the
    // double-subtraction class of bugs cannot exist.
    // Hash input: GetFullPagedTokens(/*except_last=*/false) pages the whole
    // token buffer, which at admission holds exactly the prompt -- so the
    // chained page hashes are byte-identical to what the write side registers
    // (the prefill events hash state.GetFullPagedTokens(false); the radix
    // match above uses except_last=true, but admission hashes must equal the
    // REGISTRATION form or hits never occur). The rule except_last=true
    // encodes for radix -- the last prompt token is always recomputed to
    // produce logits (vLLM semantics) -- is applied here as the explicit
    // (PrefillSize-1)/page_size cap instead, which also bounds the SWA
    // fixpoint match input.
    // Safety of claiming pages whose KV write may still be in flight
    // (registration happens at schedule time): identical stream-ordering
    // invariant as the prefill window slide -- the claimer's read kernels
    // are enqueued after the writer's on the single execution stream, and
    // claimed pages are ref>1 so they cannot be freed and reloaded from
    // outside the stream. TODO(flat-l2): revisit when out-of-stream writers
    // (load-back H2D) join the flat path.
    CoordinatorMatch flat_hit;
    if (!config_.disable_prefix_cache) {
        const std::int32_t cap_pages = std::max((request->PrefillSize() - 1) / config_.page_size, 0);
        const std::vector<std::string> flat_hashes = FlatWindowPageHashes(
            request->GetFullPagedTokens(/*except_last=*/false), config_.page_size, 0, request->PrefillSize());
        const std::size_t bounded = std::min(flat_hashes.size(), static_cast<std::size_t>(cap_pages));
        flat_hit = coordinator_.MatchPrefix(std::span<const std::string>(flat_hashes).first(bounded));
    }
    // Overwrite the radix-sourced locals: the radix tree is never written on
    // flat builds, so the match at the top of this function is always empty
    // and both arrived holding the full prompt. Every consumer below
    // (completes_prefill, the pool gate, first_pos, the event) then reads the
    // hit-adjusted values naturally.
    const std::int32_t flat_hit_tokens = flat_hit.num_common_blocks * config_.page_size;
    unscheduled = request->PrefillSize() - flat_hit_tokens;
    tokens_this_round = std::min(remaining, unscheduled);

    // TODO(radix-removal): the EnsureCapacityByEvict gate above budgets the radix
    // device_allocator_, which the flat path never draws from -- it always
    // admits. The flat shared BlockPool is gated here instead: a request that
    // does not fit is simply not scheduled this round (mirrors the radix skip),
    // so the FSM transition's Acquire cannot fail in normal operation. The flat
    // first chunk claims the admission prefix hit (flat_hit, threaded through
    // the event) and acquires tokens_this_round.
    //
    // When this chunk completes the prefill, also charge the decode headroom the
    // PrefillDone->Decoding transition will acquire (decode_input_tokens becomes
    // that state's reserve; scheduleDecode gates on exactly that value). Without
    // it, a prompt that exactly fills the pool is admitted and its own decode
    // step defers forever -- unrecoverable while flat retract is unimplemented
    // (TODO(flat-retract)). No slide credit is possible for the first chunk
    // itself (fresh tables, num_computed = 0), and even when chunk >> window
    // the SWA group transiently allocates the FULL chunk before any later op
    // slides it (TODO(flat-swa-alloc), forward_cache_ops.h), so
    // BlocksNeededFor(tokens_this_round + reserve) charges that transient peak
    // -- correctness over post-slide optimism. With an admission prefix hit,
    // tokens_this_round above is already the hit-adjusted remainder, and the
    // charge stays exact for the NEW tokens only: claimed pages are full (no
    // tail credit), and the fresh-table BlocksNeededFor overload assumes no
    // tail credit either.
    // The charge is only kept if a reservation is recorded in
    // flat_reserved_pages_ (below): the reserve is not Acquired until the
    // PrefillDone->Decoding round, so without the ledger other candidates would
    // read the raw free count and be admitted into the promised pages. Every
    // flat gate subtracts the outstanding reservations of OTHER requests.
    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t flat_decode_reserve = completes_prefill ? decode_input_tokens : 0;
    const std::int32_t flat_blocks_needed = coordinator_.BlocksNeededFor(tokens_this_round + flat_decode_reserve);
    // The transition's ClaimCommonPrefix consumes free blocks BlocksNeededFor
    // never sees: TouchBlock on a ref-0 cached hit block REMOVES it from the
    // free list (block_pool.h), so the claim shrinks the free count by the
    // number of such blocks before the remainder Acquire runs. Charge them
    // too. The count is exact, not a bound: this gate and the event's apply
    // run back to back within the same planning-loop iteration for this
    // request (no other admission interleaves), so the set of hit blocks at
    // ref 0 now is precisely the set TouchBlock will pull. Not added to
    // flat_reserve_pages below: the ledger carries cross-round promises only
    // (the decode acquire lands rounds later), while the claim lands within
    // this same round's apply.
    const std::int32_t flat_claim_blocks = coordinator_.BlocksConsumedByClaim(flat_hit);
    if (flat_blocks_needed + flat_claim_blocks >
        block_pool_.NumFreeBlocks() - flatReservedPagesExcept(request->Id())) {
        return {};
    }
    // Exact reserve page need on the post-prefill table shape: BlocksNeededFor
    // is tail-associative, so needed(chunk + reserve) - needed(chunk) equals
    // BlocksNeededFor(post-prefill tables, reserve). Computed now, while the
    // future table shape is known, and stored -- never recomputed later against
    // drifted state. Recorded just before the event is returned (below), after
    // all remaining admission gates have passed.
    const std::int32_t flat_reserve_pages =
        flat_decode_reserve > 0
            ? flat_blocks_needed - coordinator_.BlocksNeededFor(tokens_this_round)
            : 0;
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
    // Admission commits here (the caller always applies a returned event).
    // Record the decode-reserve promise only when this request will consume it
    // locally: role kD reaches PrefillDone via RemotePrefillDoneEvent, which
    // builds the state with reserve 0 (pd_events.cpp), so its transition
    // acquires nothing and a recorded entry would be a phantom until Finish.
    // Erased when FinalizePrefillAndReserveDecode acquires, or on
    // Finish/Abort/PD-success.
    if (flat_reserve_pages > 0 && config_.role != Role::kD) {
        flat_reserved_pages_[request->Id()] = flat_reserve_pages;
    }
#endif

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
        std::move(mamba_loadback_nodes),
#if TOKENSPEED_FLAT_KVCACHE
        &coordinator_,
        std::move(flat_hit),
#endif
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

#if TOKENSPEED_FLAT_KVCACHE
    // TODO(radix-removal): the radix gate above is dead on the flat path (see
    // schedulePrefillFirstChunk); the flat pool is gated here. The final chunk
    // additionally charges the PrefillDone->Decoding decode headroom
    // (reserve_num_tokens_in_next_schedule_event), mirroring the first-chunk
    // gate: BlocksNeededFor is tail-associative, so gating chunk + reserve in
    // one query equals gating the two acquires the request will actually run.
    // Like the first-chunk gate, the reserve is kept as a flat_reserved_pages_
    // entry until the PrefillDone->Decoding transition acquires it, and other
    // requests' outstanding reservations are subtracted from the free count.
    //
    // Slide credit (mirrors scheduleDecode): PrefillChunk slides the SWA
    // window BEFORE its Acquire, with num_computed = the prior chunks' tokens
    // (chunks 0..k-1 when gating chunk k = PrefillSize() - unscheduled; the
    // SchedulePrefillEvent transition computes the identical window.begin +
    // window.size). Crediting BlocksFreedByAdvance here keeps long-prompt
    // admission from being over-conservative once chunk >> window, without a
    // second copy of the window math. The credit is exact: the op runs
    // CacheFullBlocks first, which never changes ref counts, so the pending
    // slide frees exactly what this query reports.
    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t flat_decode_reserve = completes_prefill ? reserve_num_tokens_in_next_schedule_event : 0;
    const std::int32_t flat_num_computed = request->PrefillSize() - unscheduled;
    const std::int32_t flat_slide_credit =
        coordinator_.BlocksFreedByAdvance(request->FlatBlockTablesRef(), flat_num_computed);
    const std::int32_t flat_blocks_needed =
        coordinator_.BlocksNeededFor(request->FlatBlockTablesRef(), tokens_this_round + flat_decode_reserve);
    if (flat_blocks_needed >
        block_pool_.NumFreeBlocks() + flat_slide_credit - flatReservedPagesExcept(request->Id())) {
        return {};
    }
    // Exact reserve page need on the post-prefill table shape (tail-associative
    // BlocksNeededFor; see schedulePrefillFirstChunk). Prefill sliding does not
    // perturb this: AdvanceWindow only punches front holes and never touches
    // the tail page or tail_avail, and BlocksNeededFor depends on tail_avail
    // alone -- the stored count stays exact on the post-slide table. Recorded
    // before the event is returned, after the remaining admission gates.
    const std::int32_t flat_reserve_pages =
        flat_decode_reserve > 0
            ? flat_blocks_needed - coordinator_.BlocksNeededFor(request->FlatBlockTablesRef(), tokens_this_round)
            : 0;
#endif

    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() &&
        !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(1)) {
        return {};
    }

    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    const std::int32_t target = first_pos + tokens_this_round;
    if (hybrid_prefix_cache_ && !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free)) {
        return {};
    }

#if TOKENSPEED_FLAT_KVCACHE
    // Admission commits here (see schedulePrefillFirstChunk). No role gate is
    // needed on the final chunk: the planning loop only calls schedulePrefill
    // for Prefilling requests when role != kD (newForwardOperation's
    // Is<Prefilling>() && role != kD gate), so kD can never record here.
    if (flat_reserve_pages > 0) {
        flat_reserved_pages_[request->Id()] = flat_reserve_pages;
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

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(Request* request,
                                                                  std::map<std::string, std::int32_t>& simulated_free) {
    std::int32_t tail_available = request->TailPageAvailableTokens();
    std::int32_t extra_tokens = std::max(0, request->GetReserveNumTokensInNextScheduleEvent() - tail_available);
    std::int32_t pages_needed = (extra_tokens + config_.page_size - 1) / config_.page_size;

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
        return {};
    }

#if TOKENSPEED_FLAT_KVCACHE
    // TODO(radix-removal): the radix gate above is dead on the flat path (see
    // schedulePrefillFirstChunk); the flat pool is gated here. Covers both the
    // PrefillDone->Decoding reserve Acquire and each Decoding DecodeStep.
    //
    // Gate design (one source of page math): the gate composes the same two
    // coordinator primitives the transition will run -- BlocksNeededFor for the
    // Acquire, and BlocksFreedByAdvance for the SWA slide the op performs
    // BEFORE that Acquire (slide-before-acquire; the slide's freed pages fund
    // the step). A Decoding request gets its slide credited here with the same
    // num_computed DecodeStep will use (TokenSize() minus the
    // decode_input_tokens tail still pending compute); a PrefillDone request
    // runs FinalizePrefillAndReserveDecode, which slides at the full prefill
    // length -- credit that slide with PrefillSize() (the transition slides at
    // window.begin + window.size, equal by its is_last_chunk condition).
    // Outstanding decode reservations of OTHER requests are subtracted from the
    // free count (a request consuming its own reservation is exactly what this
    // gate admits).
    const std::int32_t num_computed_tokens = request->Is<fsm::Decoding>()
                                                 ? request->TokenSize() - config_.decode_input_tokens
                                                 : request->PrefillSize();
    const std::int32_t slide_credit =
        coordinator_.BlocksFreedByAdvance(request->FlatBlockTablesRef(), num_computed_tokens);
    const std::int32_t flat_blocks_needed = coordinator_.BlocksNeededFor(
        request->FlatBlockTablesRef(), request->GetReserveNumTokensInNextScheduleEvent());
    if (flat_blocks_needed >
        block_pool_.NumFreeBlocks() + slide_credit - flatReservedPagesExcept(request->Id())) {
        return {};
    }
#endif

    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && mamba_allocator_ &&
        request->Is<fsm::PrefillDone>() && request->GetLocalMambaAllocator() != nullptr &&
        !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(1)) {
        return {};
    }

    const std::int32_t first_pos = request->TokenSize();
    const std::int32_t target = first_pos + config_.decode_input_tokens;
    if (hybrid_prefix_cache_ && !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free)) {
        return {};
    }

    return fsm::ScheduleDecodeEvent{config_.decode_input_tokens,
                                    hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr
#if TOKENSPEED_FLAT_KVCACHE
                                    ,
                                    &coordinator_
#endif
    };
}

std::optional<fsm::ScheduleDecodeFromRetractedEvent> Scheduler::scheduleDecodeFromRetracted(
    Request* request, std::map<std::string, std::int32_t>& simulated_free) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};

    MatchResult match_result =
        hybrid_prefix_cache_
            ? hybrid_prefix_cache_->Match(request->GetFullPagedTokens(true), MatchIntent::StateRecovery)
            : kv_prefix_cache_.Match(request->GetFullPagedTokens(true), MatchIntent::StateRecovery);
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
    if (hybrid_prefix_cache_ && mamba_allocator_) {
        // Recovery COWs the tree-owned Mamba state into fresh request-local
        // working/checkpoint slots. Protect the source node only for this
        // allocation; retracted Mamba states are otherwise normal evictable
        // tree-owned cache entries.
        const std::int32_t mamba_slots_needed = 2 + CountMambaDeviceLoadBackSlots(mamba_loadback_nodes);
        if (!hybrid_prefix_cache_->EnsureMambaCapacityByEvict(mamba_slots_needed, mamba_recovery_node)) {
            return {};
        }
    }

    const std::int32_t target = request->TokenSize();
    if (hybrid_prefix_cache_ && !hybrid_prefix_cache_->AdmitChunkFromRetracted(request->Id(), target, simulated_free,
                                                                               match_result.paged_cache)) {
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
        &kv_prefix_cache_,
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

    // Overlap scheduling: ExtendResult may grow the token container before the
    // next Acquire runs. Clamp to the pages we actually have.
    if (total_available < static_cast<std::int32_t>(full_paged_tokens.size())) {
        full_paged_tokens.resize(total_available);
    }

    std::int32_t alloc_count =
        static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());

    OwnedPages alloc_pages = request->TakeFirstPages(alloc_count);

    kv_prefix_cache_.Insert<ResourceType::Device>(full_paged_tokens, prefix_pages, std::move(alloc_pages));

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
    return fsm::ScheduleRetractEvent{&kv_prefix_cache_, &host_allocator_, match_result,
                                     hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
}

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

std::optional<WriteBackOperation> Scheduler::applyEventAndGenerateOp(Request* request,
                                                                     fsm::ScheduleRetractEvent event) {
    // Event applier builds the (device_page, host_page) pairs.
    request->Apply(std::move(event));

    const auto& pages_to_transfer = request->GetPagesToTransfer<fsm::Retracting>();
    if (pages_to_transfer.empty()) {
        // No copy needed; advance Retracting to Retracted without an op_id.
        request->Apply(
            fsm::WriteBackDoneEvent{&kv_prefix_cache_, hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr});
        return std::nullopt;
    }
    // Register op_id so WriteBackDone can route back.
    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
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

// Apply event: state transfer + resource allocation
template <typename Event>
    requires(std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent> || std::same_as<Event, fsm::SchedulePrefillEvent>)
static PrefillOperation applyPrefillEvent(Request* request, Event event,
                                          std::span<const std::string> flat_group_ids) {
    // begin/size are in PAGE space: the slice of occupied_pages that is new to
    // the REQUEST'S TABLE this round (Python copies exactly that slice into
    // req_to_page rows [begin, begin+size)). On a first chunk with a prefix hit
    // -- radix or flat -- the matched pages enter the table during the event
    // (radix: PageContainer prepends the device node's prefix pages; flat:
    // ClaimCommonPrefix appends the claimed blocks), so begin stays 0 and size
    // COUNTS THE PREFIX: req_to_page has no rows for this request yet and needs
    // the prefix mappings too. The op's INPUT window is token-space and comes
    // from the state's window below (input_ids/extend_len/extend_prefix_len),
    // which the transition already started past the hit -- the two spaces
    // intentionally differ on a hit.
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

// TODO(radix-removal): radix hybrid prefix-cache publishing is compile-cut on
// the flat build (#if !TOKENSPEED_FLAT_KVCACHE below): in the flat binary these
// op-builders are only reached by flat requests, which carry no radix device
// node. Reachable by radix requests only in the radix build. Remove these arms
// with the radix path.
PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event) {
#if !TOKENSPEED_FLAT_KVCACHE
    auto match = event.GetMatchResult();
#endif
    auto op = applyPrefillEvent(request, std::move(event), FlatGroupIds());
#if !TOKENSPEED_FLAT_KVCACHE
    // Mamba fields only when adjunct is active.
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct()) {
        op.mamba_cow_src_idx = match.mamba_cow_src_index;
        op.mamba_branching_seqlen = match.mamba_branching_seqlen;
    }
    // Order: attach, acquire, populate. Attach before acquire so prior-chunk
    // tail pages commit into snapshots before Acquire's ReleaseSkipped frees them.
    if (hybrid_prefix_cache_) {
        hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.extend_prefix_len,
                                                op.extend_prefix_len + op.input_length, match.paged_cache);
        hybrid_prefix_cache_->PopulateOp(op);
    }
#endif
    return op;
}

PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event) {
    auto op = applyPrefillEvent(request, std::move(event), FlatGroupIds());
#if !TOKENSPEED_FLAT_KVCACHE
    // Order: attach, acquire, populate (see SchedulePrefillFirstChunkEvent).
    if (hybrid_prefix_cache_) {
        hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.extend_prefix_len,
                                                op.extend_prefix_len + op.input_length);
        hybrid_prefix_cache_->PopulateOp(op);
    }
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
    // The PrefillDone->Decoding transition just ran FinalizePrefillAndReserveDecode,
    // turning the promised decode reserve into actually-acquired pages: retire
    // the reservation (see flat_reserved_pages_).
    if (came_from_prefill_done) {
        flat_reserved_pages_.erase(op.request_id);
    }
#endif
#if !TOKENSPEED_FLAT_KVCACHE
    // Order: attach, acquire, populate.
    if (hybrid_prefix_cache_) {
        if (came_from_prefill_done) {
            hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        }
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, first_pos, first_pos + op.input_length);
        hybrid_prefix_cache_->PopulateOp(op);
    }
#endif
    return op;
}

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
        hybrid_prefix_cache_->ReleaseRequest(op.request_id);
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, 0, request->TokenSize(), paged_cache_hit);
        hybrid_prefix_cache_->PopulateOp(op);
    }
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
    // TP-determinism: tie-break on Request::Id() so the relative order within a
    // priority class is identical across ranks. requests_ is an unordered_map
    // keyed by string id; libstdc++ randomizes string hashing per process, so
    // without the tiebreaker each rank visits candidates in a different order
    // and — when token_budget / page / mamba-slot constraints are tight — picks
    // a different subset to schedule. That made forward_op None on some ranks
    // and non-None on others, deadlocking the next NCCL collective.
    std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
        int pa = priority(a), pb = priority(b);
        return pa != pb ? pa < pb : a->Id() < b->Id();
    });

    std::vector<ForwardOperation> ops;
    std::int32_t token_budget = config_.max_scheduled_tokens;
    bool pushed_prefill = false;
    auto push_op = [&](auto op, bool uses_pool_slot = false) {
        if (config_.role != Role::kD) {
            token_budget -= op.input_length;
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(op)>, PrefillOperation>) {
            pushed_prefill = true;
        }
        ops.push_back(std::move(op));
    };
#if TOKENSPEED_FLAT_KVCACHE
    // The executor owes one ExtendResult per decode op and one per op that
    // completes a prefill; a mid-prefill chunk op produces no event (the driver
    // just schedules the next chunk). Record the debt so a starved round can
    // tell "results still in flight" from a genuine starvation deadlock (see the
    // check below the loop). Decrements live in the outside-event handlers.
    auto note_result_owed = [&](Request* request) {
        if (!request->Is<fsm::Prefilling>()) {
            ++pending_forward_results_[request->Id()];
        }
    };
#else
    auto note_result_owed = [](Request*) {};
#endif
    std::vector<LoadBackOperation> loadback_ops;
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

            if (auto ev = schedulePrefillFirstChunk(request, token_budget, decode_input_tokens,
                                                    config_.disable_l2_cache, simulated_free)) {
                std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                std::vector<TreeNode*> mamba_loadback_nodes = ev->GetMambaLoadbackNodes();
                push_op(applyEventAndGenerateOp(request, std::move(*ev)), true);
                note_result_owed(request);
                // will be empty when disable_l2_cache
                if (!loadback_diff.empty() || !mamba_loadback_nodes.empty()) {
                    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
                    loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, mamba_loadback_nodes, op_id));
                }
            }
        } else if (request->Is<fsm::PrefillDone>() || (request->Is<fsm::Decoding>() && config_.role != Role::kP)) {
            // If mixed-batch is disabled, skip ALL decode if any prefill was scheduled this round.
            // If mixed-batch is enabled, the priority sort puts decodes first, so this
            // branch is reached before any prefill push.
            if (!config_.enable_mixed_prefill_decode && pushed_prefill) break;

            if (auto ev = scheduleDecode(request, simulated_free)) {
                push_op(applyEventAndGenerateOp(request, *ev));
                note_result_owed(request);
            }
        } else if (request->Is<fsm::Retracted>() && config_.role != Role::kP) {
            if (!config_.enable_mixed_prefill_decode && pushed_prefill) break;

            if (auto ev = scheduleDecodeFromRetracted(request, simulated_free)) {
                std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                std::vector<TreeNode*> mamba_loadback_nodes = ev->GetMambaLoadbackNodes();
                push_op(applyEventAndGenerateOp(request, std::move(*ev)));
                note_result_owed(request);
                if (!loadback_diff.empty() || !mamba_loadback_nodes.empty()) {
                    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
                    loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, mamba_loadback_nodes, op_id));
                }
            }
        }
    }

#if TOKENSPEED_FLAT_KVCACHE
    // TODO(flat-retract): retract is unsupported on the flat path (C slice), so
    // a pool-starved round schedules nothing; the flat admission gates deferred
    // the requests intact and they retry when pages free up. Pages only free
    // when a request finishes, which requires a forward result still in flight
    // (pending_forward_results_) or a pending cache op (cache_op_tracker_). If
    // the deferred set itself holds the pool's pages with neither pending, no
    // free can ever arrive and the stall is permanent: fail loud instead of
    // spinning silently, until flat retract lands.
    //
    // The assert requires TWO consecutive such rounds (flat_starved_rounds_):
    // pending_forward_results_ decrements on ExtendResult but the pages free on
    // the request's later Finish, so a single starved round observed in that
    // gap is a false positive -- the queued Finish frees pages before the next
    // round, which then schedules and resets the counter. A genuine deadlock
    // stays starved and trips the assert one round later.
    bool starved_this_round = false;
    if (ops.empty() && !candidates.empty()) {
        std::size_t deferred = 0;
        for (const Request* req : candidates) {
            if (req->Is<fsm::Submitted>() || req->Is<fsm::PrefetchDone>() || req->Is<fsm::Prefilling>() ||
                req->Is<fsm::PrefillDone>() || req->Is<fsm::Decoding>() || req->Is<fsm::Retracted>()) {
                ++deferred;
            }
        }
        // Block 0 is the null placeholder, never allocated: fewer than
        // TotalBlocks()-1 free blocks means live requests hold pool pages.
        const bool pool_pages_held = block_pool_.NumFreeBlocks() < block_pool_.TotalBlocks() - 1;
        const bool nothing_in_flight = pending_forward_results_.empty() && cache_op_tracker_.empty();
        // Fused-only: under PD-disagg, requests awaiting pd::Succeeded /
        // remote-prefill events hold pages while appearing in neither ledger,
        // so this predicate would misfire on a system PD completion unwedges.
        if (config_.role == Role::kFused && deferred > 0 && pool_pages_held && nothing_in_flight) {
            starved_this_round = true;
            if (++flat_starved_rounds_ >= 2) {
                const std::string msg = "flat pool starvation deadlock: " + std::to_string(deferred) +
                                        " candidate(s) deferred, no request in flight to free pages; flat retract "
                                        "not yet implemented (TODO(flat-retract))";
                _assert(false, msg.c_str());
            }
        }
    }
    if (!starved_this_round) {
        flat_starved_rounds_ = 0;  // any progress (or in-flight work) resets the streak
    }
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
