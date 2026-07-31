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

#include "cache/forward_cache_ops.h"
#include "core/token_container.h"
#include "fsm/pd_states.h"

namespace tokenspeed::fsm {

std::variant<PrefillDone, Prefilling> SchedulePrefillFirstChunkEvent::operator()(Submitted&& state) {
    _assert(coordinator_ != nullptr, "SchedulePrefillFirstChunkEvent requires a cache coordinator");
    _assert(block_tables_.size() == static_cast<std::size_t>(coordinator_->NumGroups()),
            "SchedulePrefillFirstChunkEvent requires one admitted table per cache group");

    TokenContainer* token_container = state.GetTokenContainer();
    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());
    TokenContainer::Window window{.begin = hit_tokens_, .size = tokens_this_round_};
    const bool is_last_chunk = window.begin + window.size == token_container->PrefillSize();
    if (is_last_chunk && role_ != Role::kD) {
        return PrefillDone{token_container,
                           state.GetPageSize(),
                           std::move(req_pool_index),
                           window,
                           reserve_num_tokens_in_next_schedule_event_,
                           std::move(block_tables_),
                           std::move(cache_progress_)};
    }
    return Prefilling{token_container, state.GetPageSize(), std::move(req_pool_index), window,
                      reserve_num_tokens_in_next_schedule_event_, std::move(block_tables_),
                      std::move(cache_progress_)};
}

std::variant<PrefillDone, Prefilling> SchedulePrefillEvent::operator()(Prefilling&& state) {
    TokenContainer* token_container = state.GetTokenContainer();
    const std::int32_t page_size = state.GetPageSize();
    TokenContainer::Window window{
        .begin = state.window.begin + state.window.size,
        .size = tokens_this_round_,
    };
    auto req_pool_index = std::move(state).TakeReqPoolIndex();
    auto block_tables = std::move(state).TakeBlockTables();
    if (window.begin + window.size == token_container->PrefillSize()) {
        return PrefillDone{token_container,
                           page_size,
                           std::move(req_pool_index),
                           window,
                           reserve_num_tokens_in_next_schedule_event_,
                           std::move(block_tables),
                           std::move(cache_progress_)};
    }
    return Prefilling{token_container, page_size, std::move(req_pool_index), window,
                      reserve_num_tokens_in_next_schedule_event_, std::move(block_tables),
                      std::move(cache_progress_)};
}

Decoding ScheduleDecodeEvent::operator()(PrefillDone&& state) {
    TokenContainer* token_container = state.GetTokenContainer();
    const std::int32_t page_size = state.GetPageSize();
    auto req_pool_index = std::move(state).TakeReqPoolIndex();
    auto block_tables = std::move(state).TakeBlockTables();
    return Decoding{token_container, page_size, std::move(req_pool_index), decode_input_tokens_,
                    std::move(block_tables), std::move(cache_progress_)};
}

Decoding ScheduleDecodeEvent::operator()(Decoding&& state) {
    TokenContainer* token_container = state.GetTokenContainer();
    const std::int32_t page_size = state.GetPageSize();
    auto req_pool_index = std::move(state).TakeReqPoolIndex();
    auto block_tables = std::move(state).TakeBlockTables();
    return Decoding{token_container, page_size, std::move(req_pool_index), decode_input_tokens_,
                    std::move(block_tables), std::move(cache_progress_)};
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

Finished FinishEvent::operator()(Decoding&& state) {
    return finish(std::move(state));
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

Finished AbortEvent::operator()(PrefillDone&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(Decoding&& state) {
    return abortForward(std::move(state));
}

template <typename State>
Submitted RetractEvent::retract(State&& state) {
    _assert(coordinator_ != nullptr, "RetractEvent requires a cache coordinator");
    TokenContainer* token_container = state.GetTokenContainer();
    const std::int32_t page_size = state.GetPageSize();
    token_container->RebasePrefill();
    auto block_tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, block_tables);
    return Submitted{token_container, page_size};
}

Submitted RetractEvent::operator()(PrefillDone&& state) {
    return retract(std::move(state));
}

Submitted RetractEvent::operator()(Decoding&& state) {
    return retract(std::move(state));
}

}  // namespace tokenspeed::fsm
