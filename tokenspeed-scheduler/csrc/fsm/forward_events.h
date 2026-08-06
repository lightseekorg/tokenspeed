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

#include "cache/coordinator/kv_cache_coordinator.h"
#include "fsm/base_event.h"
#include "fsm/forward_states.h"
#include "utils.h"

namespace tokenspeed::fsm {

struct Bootstrapping;

struct SchedulePrefillFirstChunkEvent : InvalidTransitionHandler<SchedulePrefillFirstChunkEvent> {
    using InvalidTransitionHandler<SchedulePrefillFirstChunkEvent>::operator();

    SchedulePrefillFirstChunkEvent(std::int32_t tokens_this_round,
                                   std::int32_t reserve_num_tokens_in_next_schedule_event,
                                   ReqPoolAllocator* req_pool_allocator, PrefillSource source,
                                   KvCacheCoordinator* coordinator,
                                   std::vector<BlockTable> block_tables, std::int32_t hit_tokens,
                                   CacheProgress cache_progress, std::vector<BlockTransfer> load_pairs)
        : tokens_this_round_{tokens_this_round},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event},
          req_pool_allocator_{req_pool_allocator},
          source_{source},
          coordinator_{coordinator},
          block_tables_{std::move(block_tables)},
          hit_tokens_{hit_tokens},
          cache_progress_{std::move(cache_progress)},
          load_pairs_{std::move(load_pairs)} {}

    std::variant<PrefillDone, Prefilling> operator()(Submitted&& state);
    std::variant<PrefillDone, Prefilling> operator()(Retracted&& state);
    std::vector<BlockTransfer> TakeLoadPairs() { return std::exchange(load_pairs_, {}); }
    PrefillSource Source() const { return source_; }

private:
    template <typename State>
    std::variant<PrefillDone, Prefilling> scheduleFirstChunk(State&& state);

    std::int32_t tokens_this_round_{};
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
    ReqPoolAllocator* req_pool_allocator_{};
    PrefillSource source_{PrefillSource::kLocal};
    KvCacheCoordinator* coordinator_{};
    std::vector<BlockTable> block_tables_;
    std::int32_t hit_tokens_{0};
    CacheProgress cache_progress_;
    std::vector<BlockTransfer> load_pairs_;
};

struct SchedulePrefillEvent : InvalidTransitionHandler<SchedulePrefillEvent> {
    using InvalidTransitionHandler<SchedulePrefillEvent>::operator();

    SchedulePrefillEvent(std::int32_t tokens_this_round, std::int32_t reserve_num_tokens_in_next_schedule_event,
                         CacheProgress cache_progress)
        : tokens_this_round_{tokens_this_round},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event},
          cache_progress_{std::move(cache_progress)} {}

    std::variant<PrefillDone, Prefilling> operator()(Prefilling&& state);

private:
    std::int32_t tokens_this_round_{};
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
    CacheProgress cache_progress_;
};

struct ScheduleDecodeEvent : InvalidTransitionHandler<ScheduleDecodeEvent> {
    using InvalidTransitionHandler<ScheduleDecodeEvent>::operator();

    ScheduleDecodeEvent(std::int32_t decode_input_tokens, CacheProgress cache_progress)
        : decode_input_tokens_{decode_input_tokens}, cache_progress_{std::move(cache_progress)} {}

    Decoding operator()(PrefillDone&& state);
    Decoding operator()(Decoding&& state);

private:
    template <typename State>
    Decoding decode(State&& state);

    std::int32_t decode_input_tokens_{};
    CacheProgress cache_progress_;
};

struct FinishEvent : InvalidTransitionHandler<FinishEvent> {
    using InvalidTransitionHandler<FinishEvent>::operator();

    explicit FinishEvent(KvCacheCoordinator* coordinator) : coordinator_{coordinator} {}

    Finished operator()(PrefillDone&& state);
    Finished operator()(Decoding&& state);
    Finished operator()(Retracted&& state);
    Finished operator()(Finished&& state) { return std::move(state); }

private:
    template <typename State>
    Finished finish(State&& state);

    KvCacheCoordinator* coordinator_{};
};

struct AbortEvent : InvalidTransitionHandler<AbortEvent> {
    using InvalidTransitionHandler<AbortEvent>::operator();

    explicit AbortEvent(KvCacheCoordinator* coordinator) : coordinator_{coordinator} {}

    Finished operator()(Bootstrapping&&);
    Finished operator()(Submitted&&);
    Finished operator()(Prefilling&& state);
    Finished operator()(PrefillDone&& state);
    Finished operator()(Decoding&& state);
    Finished operator()(Retracted&& state);
    Finished operator()(Finished&& state) { return std::move(state); }

private:
    template <typename State>
    Finished abortForward(State&& state);

    KvCacheCoordinator* coordinator_{};
};

// Decode Retraction releases Device ownership immediately. Cached blocks may
// remain in L1 or be pinned by an already-created best-effort Store ticket.
struct RetractionEvent : InvalidTransitionHandler<RetractionEvent> {
    using InvalidTransitionHandler<RetractionEvent>::operator();

    explicit RetractionEvent(KvCacheCoordinator* coordinator) : coordinator_{coordinator} {}

    Retracted operator()(Decoding&& state);

private:
    KvCacheCoordinator* coordinator_{};
};

// Release every request-owned page and requeue all accepted tokens as prefill.
struct RetractEvent : InvalidTransitionHandler<RetractEvent> {
    using InvalidTransitionHandler<RetractEvent>::operator();

    explicit RetractEvent(KvCacheCoordinator* coordinator) : coordinator_{coordinator} {}

    Submitted operator()(PrefillDone&& state);
    Submitted operator()(Decoding&& state);

private:
    template <typename State>
    Submitted retract(State&& state);

    KvCacheCoordinator* coordinator_{};
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

    Finished operator()(Finished&& state) { return std::move(state); }

private:
    std::vector<std::int32_t> result_tokens_;
};

}  // namespace tokenspeed::fsm
