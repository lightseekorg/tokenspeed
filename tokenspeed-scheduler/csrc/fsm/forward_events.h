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

#pragma once

#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "fsm/base_event.h"
#include "fsm/forward_states.h"
// The D and P role grammars schedule INTO the PD parking states
// (RemotePrefilling, PrefillAwaitingResult), so the forward events name them
// in their transition signatures.
#include "fsm/pd_states.h"
#include "utils.h"

namespace tokenspeed::fsm {

struct SchedulePrefillFirstChunkEvent : InvalidTransitionHandler<SchedulePrefillFirstChunkEvent> {
    using InvalidTransitionHandler<SchedulePrefillFirstChunkEvent>::operator();

    // `awaits_result`: a completed prefill parks in PrefillAwaitingResult
    // until its final chunk's result lands (the P role, whose remote decode
    // needs the bootstrap token that arrives with it).
    SchedulePrefillFirstChunkEvent(std::int32_t tokens_this_round,
                                   std::int32_t reserve_num_tokens_in_next_schedule_event,
                                   ReqPoolAllocator* req_pool_allocator, PrefillSource source,
                                   CacheCoordinator* coordinator, std::vector<BlockTable> block_tables,
                                   std::int32_t hit_tokens, CacheProgress cache_progress,
                                   std::vector<BlockTransfer> load_pairs, bool awaits_result)
        : tokens_this_round_{tokens_this_round},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event},
          req_pool_allocator_{req_pool_allocator},
          source_{source},
          coordinator_{coordinator},
          block_tables_{std::move(block_tables)},
          hit_tokens_{hit_tokens},
          cache_progress_{std::move(cache_progress)},
          load_pairs_{std::move(load_pairs)},
          awaits_result_{awaits_result} {}

    std::variant<PrefillDone, PrefillAwaitingResult, Prefilling, RemotePrefilling> operator()(Submitted&& state);
    std::vector<BlockTransfer> TakeLoadPairs() { return std::exchange(load_pairs_, {}); }

private:
    std::int32_t tokens_this_round_{};
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
    ReqPoolAllocator* req_pool_allocator_{};
    PrefillSource source_{PrefillSource::kLocal};
    CacheCoordinator* coordinator_{};
    std::vector<BlockTable> block_tables_;
    std::int32_t hit_tokens_{0};
    CacheProgress cache_progress_;
    std::vector<BlockTransfer> load_pairs_;
    bool awaits_result_{false};
};

struct SchedulePrefillEvent : InvalidTransitionHandler<SchedulePrefillEvent> {
    using InvalidTransitionHandler<SchedulePrefillEvent>::operator();

    SchedulePrefillEvent(std::int32_t tokens_this_round, std::int32_t reserve_num_tokens_in_next_schedule_event,
                         bool awaits_result)
        : tokens_this_round_{tokens_this_round},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event},
          awaits_result_{awaits_result} {}

    std::variant<PrefillDone, PrefillAwaitingResult, Prefilling> operator()(Prefilling&& state);

private:
    std::int32_t tokens_this_round_{};
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
    bool awaits_result_{false};
};

struct ScheduleDecodeEvent : InvalidTransitionHandler<ScheduleDecodeEvent> {
    using InvalidTransitionHandler<ScheduleDecodeEvent>::operator();

    explicit ScheduleDecodeEvent(std::int32_t decode_input_tokens) : decode_input_tokens_{decode_input_tokens} {}

    Decoding operator()(PrefillDone&& state);
    Decoding operator()(PrefillAwaitingResult&& state);
    Decoding operator()(Decoding&& state);

private:
    template <typename State>
    Decoding decode(State&& state);

    std::int32_t decode_input_tokens_{};
};

struct FinishEvent : InvalidTransitionHandler<FinishEvent> {
    using InvalidTransitionHandler<FinishEvent>::operator();

    explicit FinishEvent(CacheCoordinator* coordinator) : coordinator_{coordinator} {}

    Finished operator()(PrefillDone&& state);
    Finished operator()(PrefillAwaitingResult&& state);
    Finished operator()(Decoding&& state);
    // A suspended request's image dies with the state: its Host L2 pins drop
    // (the entries stay published, now evictable), its snapshot-pool blocks
    // and blob slot return.
    Finished operator()(Retracted&& state);
    Finished operator()(Restoring&& state);
    Finished operator()(Finished&& state) { return std::move(state); }

private:
    template <typename State>
    Finished finish(State&& state);

    CacheCoordinator* coordinator_{};
};

struct AbortEvent : InvalidTransitionHandler<AbortEvent> {
    using InvalidTransitionHandler<AbortEvent>::operator();

    explicit AbortEvent(CacheCoordinator* coordinator) : coordinator_{coordinator} {}

    Finished operator()(Bootstrapping&&);
    Finished operator()(Submitted&&);
    // The op in flight keeps the Host blocks until its ACK; only the
    // request's own pins drop here.
    Finished operator()(Prefetching&&);
    Finished operator()(Prefilling&& state);
    Finished operator()(RemotePrefilling&& state);
    Finished operator()(PrefillDone&& state);
    Finished operator()(PrefillAwaitingResult&& state);
    Finished operator()(Decoding&& state);
    Finished operator()(Retracted&& state);
    Finished operator()(Restoring&& state);
    Finished operator()(Finished&& state) { return std::move(state); }

private:
    template <typename State>
    Finished abortForward(State&& state);

    CacheCoordinator* coordinator_{};
};

// Capacity retraction: the victim's Device pages are released -- and granted
// away in this very round -- while its KV lives on in the image the caller
// took first (Host L2 pins for the published pages, snapshot-pool blocks for
// the rest). The state records exactly where the request stopped, so the
// restore continues it there: nothing is recomputed and the prefill window is
// not rebased. "Was retracted" and "never ran" stay different situations, and
// only the first escalates the readmission headroom (Request::NoteRetracted,
// called by the scheduler beside this event).
//
// `resumes_generation` is Request::HasGeneratedOutput() at retraction time: a
// victim with generated tokens a client is reading resumes ahead of one that
// had produced nothing. `pending_store_ops` are the store ops whose ACKs make
// the image landed. PrefillAwaitingResult exists only on the P role, which
// never retracts, and RemotePrefilling is PD-pinned: neither overload exists.
struct SnapshotRetractEvent : InvalidTransitionHandler<SnapshotRetractEvent> {
    using InvalidTransitionHandler<SnapshotRetractEvent>::operator();

    SnapshotRetractEvent(CacheCoordinator* coordinator, std::int64_t epoch, bool resumes_generation,
                         RetractionImage image, std::shared_ptr<SnapshotSlotIndex> blob_slot,
                         std::vector<std::uint32_t> pending_store_ops)
        : coordinator_{coordinator},
          epoch_{epoch},
          resumes_generation_{resumes_generation},
          image_{std::move(image)},
          blob_slot_{std::move(blob_slot)},
          pending_store_ops_{std::move(pending_store_ops)} {}

    Retracted operator()(Prefilling&& state);
    Retracted operator()(PrefillDone&& state);
    Retracted operator()(Decoding&& state);

private:
    template <typename State>
    Retracted retract(State&& state, ResumeShape shape);

    CacheCoordinator* coordinator_{};
    std::int64_t epoch_{0};
    bool resumes_generation_{false};
    RetractionImage image_;
    std::shared_ptr<SnapshotSlotIndex> blob_slot_;
    std::vector<std::uint32_t> pending_store_ops_;
};

// A Submitted request's L3 prefetch was issued: it waits for the fill before
// it can be admitted as a Host hit, pinning the blocks being filled.
struct SchedulePrefetchEvent : InvalidTransitionHandler<SchedulePrefetchEvent> {
    using InvalidTransitionHandler<SchedulePrefetchEvent>::operator();

    SchedulePrefetchEvent(std::vector<CacheBlockRef> host_blocks, std::uint32_t prefetch_op)
        : host_blocks_{std::move(host_blocks)}, prefetch_op_{prefetch_op} {}

    Prefetching operator()(Submitted&& state) {
        return Prefetching{state.TokenContainerPtr(), state.PrefixGranularity(), std::move(host_blocks_), prefetch_op_};
    }

private:
    std::vector<CacheBlockRef> host_blocks_;
    std::uint32_t prefetch_op_{0};
};

// The prefetch finished: the request is Submitted again, at its original
// queue position, holding the Host entries that landed (published by the
// scheduler before this event) until its admission claims them.
struct PrefetchDoneEvent : InvalidTransitionHandler<PrefetchDoneEvent> {
    using InvalidTransitionHandler<PrefetchDoneEvent>::operator();

    explicit PrefetchDoneEvent(std::vector<CacheBlockRef> published) : published_{std::move(published)} {}

    Submitted operator()(Prefetching&& state) {
        return Submitted{state.token_container, state.prefix_granularity, std::move(published_)};
    }

private:
    std::vector<CacheBlockRef> published_;
};

// One of the image's store ops was acknowledged (WriteBackDone for the L2
// leg, with the Host entries it published; SnapshotDone for the tail leg,
// with none); once none is pending the image has landed and the request may
// be restored. The published entries let the image follow a publication the
// Host index redirected to an existing canonical block.
struct StoreLandedEvent : InvalidTransitionHandler<StoreLandedEvent> {
    using InvalidTransitionHandler<StoreLandedEvent>::operator();

    StoreLandedEvent(std::uint32_t op_id, std::span<const HostPublication> published)
        : op_id_{op_id}, published_{published} {}

    Retracted operator()(Retracted&& state) {
        state.NoteStoreLanded(op_id_, published_);
        return std::move(state);
    }

private:
    std::uint32_t op_id_{0};
    std::span<const HostPublication> published_;
};

// The restore was admitted: fresh Device pages for the whole image (the
// scheduler ran CacheCoordinator::Restore). Mirrors
// SchedulePrefillFirstChunkEvent but returns no forward operation: the plan
// carries a SnapshotRestore cache op, and nothing is schedulable until its
// ACK. The request-pool row stays with that op until then (see Restoring).
// Nothing is out against the pages yet.
struct ScheduleRestoreEvent : InvalidTransitionHandler<ScheduleRestoreEvent> {
    using InvalidTransitionHandler<ScheduleRestoreEvent>::operator();

    ScheduleRestoreEvent(CacheCoordinator* coordinator, std::vector<BlockTable> block_tables, std::uint32_t restore_op)
        : coordinator_{coordinator}, block_tables_{std::move(block_tables)}, restore_op_{restore_op} {}

    Restoring operator()(Retracted&& state);

private:
    CacheCoordinator* coordinator_{};
    std::vector<BlockTable> block_tables_;
    std::uint32_t restore_op_{0};
};

// The restore's copies landed: the request continues in the state it left,
// with the same token count, window and reserve, in the request-pool row the
// op imported its slot-state blob into (handed over from the op's ticket).
// The image dies here -- Host L2 pins drop, snapshot-pool blocks and the blob
// slot return.
struct RestoreDoneEvent : InvalidTransitionHandler<RestoreDoneEvent> {
    using InvalidTransitionHandler<RestoreDoneEvent>::operator();

    explicit RestoreDoneEvent(ReqPoolIndex req_pool_index) : req_pool_index_{std::move(req_pool_index)} {}

    std::variant<Prefilling, PrefillDone, Decoding> operator()(Restoring&& state);

private:
    ReqPoolIndex req_pool_index_;
};

struct UpdateReserveNumTokensEvent : InvalidTransitionHandler<UpdateReserveNumTokensEvent> {
    using InvalidTransitionHandler<UpdateReserveNumTokensEvent>::operator();

    explicit UpdateReserveNumTokensEvent(std::int32_t value) : value_{value} {}

    Decoding operator()(Decoding&& state) {
        state.SetReserveNumTokensInNextScheduleEvent(value_);
        return std::move(state);
    }
    Finished operator()(Finished&& state) { return std::move(state); }

private:
    std::int32_t value_{};
};

struct ExtendResultEvent : InvalidTransitionHandler<ExtendResultEvent> {
    using InvalidTransitionHandler<ExtendResultEvent>::operator();

    explicit ExtendResultEvent(std::vector<std::int32_t> result_tokens) : result_tokens_{std::move(result_tokens)} {}

    template <typename State>
        requires CanExtendTokenContainer<State>
    std::remove_cvref_t<State> operator()(State&& state) {
        state.ExtendResultTokens(result_tokens_);
        return std::move(state);
    }

    // Only the FINAL chunk's result carries a token (an intermediate chunk
    // reports back empty -- see the Prefilling overload below), and under
    // the PP chunk pipeline older intermediate results may still be landing
    // after the final chunk was scheduled. So an empty arrival keeps
    // waiting, and the token-bearing one is the result this state was
    // waiting FOR: the prompt's last token is now real, so the request
    // becomes schedulable (P: its remote decode can carry the bootstrap
    // token).
    std::variant<PrefillDone, PrefillAwaitingResult> operator()(PrefillAwaitingResult&& state) {
        if (result_tokens_.empty()) {
            return std::move(state);
        }
        state.ExtendResultTokens(result_tokens_);
        // Older intermediate chunk results may still be landing on these
        // pages (see above): the bundle, in-flight count included, moves on.
        return PrefillDone{std::move(state.resources), state.window, state.ReserveNumTokensInNextScheduleEvent()};
    }

    // An intermediate chunk produces no token -- its result is KV written
    // into pages this request owns. The event still arrives (empty), because
    // the arrival is the point: it is what clears the in-flight count that
    // keeps the chunk's pages safe from retraction.
    Prefilling operator()(Prefilling&& state) {
        _assert(result_tokens_.empty(), "ExtendResultEvent: an intermediate prefill chunk produces no tokens");
        return std::move(state);
    }
    RemotePrefilling operator()(RemotePrefilling&& state) { return std::move(state); }

    Finished operator()(Finished&& state) { return std::move(state); }

private:
    std::vector<std::int32_t> result_tokens_;
};

}  // namespace tokenspeed::fsm
