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

#include "fsm/forward_events.h"

#include <memory>
#include <utility>

#include "scheduler/operations/cache.h"
#include "core/token_container.h"
#include "fsm/pd_states.h"

namespace tokenspeed::fsm {

std::variant<PrefillDone, PrefillAwaitingResult, Prefilling, RemotePrefilling>
SchedulePrefillFirstChunkEvent::scheduleFirstChunk(TokenContainer* token_container, std::int32_t prefix_granularity) {
    _assert(coordinator_ != nullptr, "SchedulePrefillFirstChunkEvent requires a cache coordinator");
    _assert(block_tables_.size() == static_cast<std::size_t>(coordinator_->NumGroups()),
            "SchedulePrefillFirstChunkEvent requires one admitted table per cache group");

    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());
    TokenContainer::Window window{.begin = hit_tokens_, .size = tokens_this_round_};
    if (source_ == PrefillSource::kRemote) {
        // The peer prefills the whole prompt; this engine only holds the
        // destination pages until RemotePrefillDone.
        return RemotePrefilling{token_container,
                                prefix_granularity,
                                std::move(req_pool_index),
                                window,
                                reserve_num_tokens_in_next_schedule_event_,
                                std::move(block_tables_),
                                std::move(cache_progress_)};
    }
    if (window.begin + window.size == token_container->PrefillSize()) {
        if (awaits_result_) {
            return PrefillAwaitingResult{token_container,
                                         prefix_granularity,
                                         std::move(req_pool_index),
                                         window,
                                         reserve_num_tokens_in_next_schedule_event_,
                                         std::move(block_tables_),
                                         std::move(cache_progress_)};
        }
        return PrefillDone{token_container,
                           prefix_granularity,
                           std::move(req_pool_index),
                           window,
                           reserve_num_tokens_in_next_schedule_event_,
                           std::move(block_tables_),
                           std::move(cache_progress_)};
    }
    return Prefilling{token_container,
                      prefix_granularity,
                      std::move(req_pool_index),
                      window,
                      reserve_num_tokens_in_next_schedule_event_,
                      std::move(block_tables_),
                      std::move(cache_progress_)};
}

std::variant<PrefillDone, PrefillAwaitingResult, Prefilling, RemotePrefilling>
SchedulePrefillFirstChunkEvent::operator()(Submitted&& state) {
    return scheduleFirstChunk(state.TokenContainerPtr(), state.PrefixGranularity());
}

std::variant<PrefillDone, PrefillAwaitingResult, Prefilling, RemotePrefilling>
SchedulePrefillFirstChunkEvent::operator()(Retracted&& state) {
    return scheduleFirstChunk(state.TokenContainerPtr(), state.PrefixGranularity());
}

std::variant<PrefillDone, PrefillAwaitingResult, Prefilling> SchedulePrefillEvent::operator()(Prefilling&& state) {
    TokenContainer* token_container = state.TokenContainerPtr();
    const std::int32_t prefix_granularity = state.PrefixGranularity();
    // The chunk forwards already out survive this transition -- see
    // ForwardState::CarryResultsInFlight.
    const std::int32_t results_in_flight = state.ResultsInFlight();
    TokenContainer::Window window{
        .begin = state.window.begin + state.window.size,
        .size = tokens_this_round_,
    };
    auto req_pool_index = std::move(state).TakeRequestPoolIndex();
    auto block_tables = std::move(state).TakeBlockTables();
    auto build = [&]<typename Next>(std::type_identity<Next>) {
        Next next{token_container,
                  prefix_granularity,
                  std::move(req_pool_index),
                  window,
                  reserve_num_tokens_in_next_schedule_event_,
                  std::move(block_tables),
                  std::move(cache_progress_)};
        next.CarryResultsInFlight(results_in_flight);
        return next;
    };
    if (window.begin + window.size == token_container->PrefillSize()) {
        if (awaits_result_) {
            return build(std::type_identity<PrefillAwaitingResult>{});
        }
        return build(std::type_identity<PrefillDone>{});
    }
    return build(std::type_identity<Prefilling>{});
}

template <typename State>
Decoding ScheduleDecodeEvent::decode(State&& state) {
    TokenContainer* token_container = state.TokenContainerPtr();
    const std::int32_t prefix_granularity = state.PrefixGranularity();
    // The overlap schedule plans this decode before the previous result
    // lands -- see ForwardState::CarryResultsInFlight.
    const std::int32_t results_in_flight = state.ResultsInFlight();
    auto req_pool_index = std::move(state).TakeRequestPoolIndex();
    auto block_tables = std::move(state).TakeBlockTables();
    Decoding decoding{token_container,      prefix_granularity,      std::move(req_pool_index),
                      decode_input_tokens_, std::move(block_tables), std::move(cache_progress_)};
    decoding.CarryResultsInFlight(results_in_flight);
    return decoding;
}

Decoding ScheduleDecodeEvent::operator()(PrefillDone&& state) {
    return decode(std::move(state));
}

Decoding ScheduleDecodeEvent::operator()(PrefillAwaitingResult&& state) {
    return decode(std::move(state));
}

Decoding ScheduleDecodeEvent::operator()(Decoding&& state) {
    return decode(std::move(state));
}

template <typename State>
Finished FinishEvent::finish(State&& state) {
    _assert(coordinator_ != nullptr, "FinishEvent requires a cache coordinator");
    auto block_tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, block_tables);
    return Finished{};
}

Finished FinishEvent::operator()(PrefillDone&& state) {
    return finish(std::move(state));
}

Finished FinishEvent::operator()(PrefillAwaitingResult&& state) {
    return finish(std::move(state));
}

Finished FinishEvent::operator()(Decoding&& state) {
    return finish(std::move(state));
}

Finished FinishEvent::operator()(Retracted&&) {
    return Finished{};
}

Finished AbortEvent::operator()(Bootstrapping&&) {
    return Finished{};
}

Finished AbortEvent::operator()(Submitted&&) {
    return Finished{};
}

template <typename State>
Finished AbortEvent::abortForward(State&& state) {
    _assert(coordinator_ != nullptr, "AbortEvent requires a cache coordinator");
    auto block_tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, block_tables);
    return Finished{};
}

Finished AbortEvent::operator()(Prefilling&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(RemotePrefilling&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(PrefillDone&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(PrefillAwaitingResult&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(Decoding&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(Retracted&&) {
    return Finished{};
}

template <typename State>
Retracted RetractEvent::retract(State&& state) {
    _assert(coordinator_ != nullptr, "RetractEvent requires a cache coordinator");
    TokenContainer* token_container = state.TokenContainerPtr();
    const std::int32_t prefix_granularity = state.PrefixGranularity();
    token_container->RebasePrefill();
    auto block_tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, block_tables);
    return Retracted{.token_container = token_container,
                     .prefix_granularity = prefix_granularity,
                     .retraction_epoch = epoch_,
                     .has_recoverable_snapshot = has_recoverable_snapshot_,
                     .resumes_generation = resumes_generation_};
}

Retracted RetractEvent::operator()(Prefilling&& state) {
    return retract(std::move(state));
}

Retracted RetractEvent::operator()(PrefillDone&& state) {
    return retract(std::move(state));
}

Retracted RetractEvent::operator()(Decoding&& state) {
    return retract(std::move(state));
}

}  // namespace tokenspeed::fsm
