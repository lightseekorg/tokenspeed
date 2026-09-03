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
                                   std::vector<BlockTransfer> load_pairs, bool awaits_result = false)
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
    std::variant<PrefillDone, PrefillAwaitingResult, Prefilling, RemotePrefilling> operator()(Retracted&& state);
    std::vector<BlockTransfer> TakeLoadPairs() { return std::exchange(load_pairs_, {}); }

private:
    std::variant<PrefillDone, PrefillAwaitingResult, Prefilling, RemotePrefilling> scheduleFirstChunk(
        TokenContainer* token_container, std::int32_t prefix_granularity);

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
                         CacheProgress cache_progress, bool awaits_result = false)
        : tokens_this_round_{tokens_this_round},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event},
          cache_progress_{std::move(cache_progress)},
          awaits_result_{awaits_result} {}

    std::variant<PrefillDone, PrefillAwaitingResult, Prefilling> operator()(Prefilling&& state);

private:
    std::int32_t tokens_this_round_{};
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
    CacheProgress cache_progress_;
    bool awaits_result_{false};
};

struct ScheduleDecodeEvent : InvalidTransitionHandler<ScheduleDecodeEvent> {
    using InvalidTransitionHandler<ScheduleDecodeEvent>::operator();

    ScheduleDecodeEvent(std::int32_t decode_input_tokens, CacheProgress cache_progress)
        : decode_input_tokens_{decode_input_tokens}, cache_progress_{std::move(cache_progress)} {}

    Decoding operator()(PrefillDone&& state);
    Decoding operator()(PrefillAwaitingResult&& state);
    Decoding operator()(Decoding&& state);

private:
    template <typename State>
    Decoding decode(State&& state);

    std::int32_t decode_input_tokens_{};
    CacheProgress cache_progress_;
};

struct FinishEvent : InvalidTransitionHandler<FinishEvent> {
    using InvalidTransitionHandler<FinishEvent>::operator();

    explicit FinishEvent(CacheCoordinator* coordinator) : coordinator_{coordinator} {}

    Finished operator()(PrefillDone&& state);
    Finished operator()(PrefillAwaitingResult&& state);
    Finished operator()(Decoding&& state);
    Finished operator()(Retracted&& state);
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
    Finished operator()(Prefilling&& state);
    Finished operator()(RemotePrefilling&& state);
    Finished operator()(PrefillDone&& state);
    Finished operator()(PrefillAwaitingResult&& state);
    Finished operator()(Decoding&& state);
    Finished operator()(Retracted&& state);
    Finished operator()(Finished&& state) { return std::move(state); }

private:
    template <typename State>
    Finished abortForward(State&& state);

    CacheCoordinator* coordinator_{};
};

// Capacity retraction: release every request-owned page and requeue all
// accepted tokens as prefill. With a host cache the caller stores the KV
// first (`has_recoverable_snapshot`) and the readmission loads it back;
// without one the readmission recomputes from whatever the prefix cache
// still holds. Either way the request lands in Retracted, not Submitted:
// "was retracted" and "never ran" are different situations, and only the
// first escalates its readmission headroom (Request::AdmissionHeadroom).
//
// `resumes_generation` is Request::HasGeneratedOutput() at retraction time:
// a victim with generated tokens a client is reading resumes ahead of one
// that had produced nothing -- including a victim taken mid-RECOVERY, whose
// generated tokens were rebased into its prefill by an earlier retraction.
struct RetractEvent : InvalidTransitionHandler<RetractEvent> {
    using InvalidTransitionHandler<RetractEvent>::operator();

    RetractEvent(CacheCoordinator* coordinator, std::int64_t epoch, bool has_recoverable_snapshot,
                 bool resumes_generation)
        : coordinator_{coordinator},
          epoch_{epoch},
          has_recoverable_snapshot_{has_recoverable_snapshot},
          resumes_generation_{resumes_generation} {}

    Retracted operator()(Prefilling&& state);
    Retracted operator()(PrefillDone&& state);
    Retracted operator()(Decoding&& state);

private:
    template <typename State>
    Retracted retract(State&& state);

    CacheCoordinator* coordinator_{};
    std::int64_t epoch_{0};
    bool has_recoverable_snapshot_{true};
    bool resumes_generation_{false};
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
        TokenContainer* token_container = state.TokenContainerPtr();
        const std::int32_t prefix_granularity = state.PrefixGranularity();
        const TokenContainer::Window window = state.window;
        const std::int32_t reserve = state.ReserveNumTokensInNextScheduleEvent();
        auto req_pool_index = std::move(state).TakeRequestPoolIndex();
        auto block_tables = std::move(state).TakeBlockTables();
        auto cache_progress = std::move(state).TakeCacheProgress();
        return PrefillDone{token_container, prefix_granularity,      std::move(req_pool_index), window,
                           reserve,         std::move(block_tables), std::move(cache_progress)};
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
