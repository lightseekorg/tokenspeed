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

std::variant<PrefillDone, Prefilling> SchedulePrefillFirstChunkEvent::operator()(Submitted&& state) {
    _assert(coordinator_ != nullptr, "SchedulePrefillFirstChunkEvent requires a cache coordinator");
    _assert(block_tables_.size() == static_cast<std::size_t>(coordinator_->NumGroups()),
            "SchedulePrefillFirstChunkEvent requires one admitted table per cache group");

    TokenContainer* token_container = state.TokenContainerPtr();
    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());
    TokenContainer::Window window{.begin = hit_tokens_, .size = tokens_this_round_};
    const bool is_last_chunk = window.begin + window.size == token_container->PrefillSize();
    if (is_last_chunk && role_ != Role::kD) {
        return PrefillDone{token_container,
                           state.PageSize(),
                           std::move(req_pool_index),
                           window,
                           reserve_num_tokens_in_next_schedule_event_,
                           std::move(block_tables_),
                           std::move(cache_progress_)};
    }
    return Prefilling{token_container,
                      state.PageSize(),
                      std::move(req_pool_index),
                      window,
                      reserve_num_tokens_in_next_schedule_event_,
                      std::move(block_tables_),
                      std::move(cache_progress_)};
}

std::variant<PrefillDone, Prefilling> SchedulePrefillEvent::operator()(Prefilling&& state) {
    TokenContainer* token_container = state.TokenContainerPtr();
    const std::int32_t page_size = state.PageSize();
    TokenContainer::Window window{
        .begin = state.window.begin + state.window.size,
        .size = tokens_this_round_,
    };
    auto req_pool_index = std::move(state).TakeRequestPoolIndex();
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
    return Prefilling{token_container,
                      page_size,
                      std::move(req_pool_index),
                      window,
                      reserve_num_tokens_in_next_schedule_event_,
                      std::move(block_tables),
                      std::move(cache_progress_)};
}

template <typename State>
Decoding ScheduleDecodeEvent::decode(State&& state) {
    TokenContainer* token_container = state.TokenContainerPtr();
    const std::int32_t page_size = state.PageSize();
    auto req_pool_index = std::move(state).TakeRequestPoolIndex();
    auto block_tables = std::move(state).TakeBlockTables();
    return Decoding{token_container,           page_size,
                    std::move(req_pool_index), decode_input_tokens_,
                    std::move(block_tables),   std::move(cache_progress_)};
}

Decoding ScheduleDecodeEvent::operator()(PrefillDone&& state) {
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

Finished FinishEvent::operator()(Decoding&& state) {
    return finish(std::move(state));
}

Finished FinishEvent::operator()(Retracting&& state) {
    return finish(std::move(state.device_state));
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

Finished AbortEvent::operator()(PrefillDone&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(Decoding&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(Retracting&& state) {
    return abortForward(std::move(state.device_state));
}

Finished AbortEvent::operator()(Retracted&&) {
    return Finished{};
}

Retracted CompleteRetractionEvent::operator()(Retracting&& state) {
    _assert(coordinator_ != nullptr, "CompleteRetractionEvent requires a cache coordinator");
    Decoding device_state = std::move(state.device_state);
    const std::int32_t first_new_page = state.host_prefix_tokens / coordinator_->CacheBlockTokens();
    const CacheProgress& current_progress = state.cache_progress;
    if (first_new_page < static_cast<std::int32_t>(current_progress.page_hashes.size())) {
        coordinator_->CacheHostCompletedBlocks(state.host_tables, current_progress.page_hashes,
                                               current_progress.access_epoch, first_new_page,
                                               device_state.TokenContainerPtr()->Size() -
                                                   device_state.ReserveNumTokensInNextScheduleEvent(),
                                               CacheBoundaryKind::kChunk);
    }
    const std::int32_t decode_reserve_tokens = device_state.ReserveNumTokensInNextScheduleEvent();
    Retracted retracted{
        .host_tables = std::move(state.host_tables),
        .cache_progress = std::move(state.cache_progress),
        .decode_reserve_tokens = decode_reserve_tokens,
    };
    auto device_tables = std::move(device_state).TakeBlockTables();
    FreeRequest(*coordinator_, device_tables);
    return retracted;
}

Decoding RecoverEvent::operator()(Retracted&& state) {
    _assert(req_pool_allocator_ != nullptr, "RecoverEvent requires a request-pool allocator");
    _assert(token_container_ != nullptr, "RecoverEvent requires a token container");
    return Decoding{token_container_, page_size_, std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate()),
                    state.decode_reserve_tokens, std::move(device_tables_), std::move(state.cache_progress)};
}

template <typename State>
Submitted RetractEvent::retract(State&& state) {
    _assert(coordinator_ != nullptr, "RetractEvent requires a cache coordinator");
    TokenContainer* token_container = state.TokenContainerPtr();
    const std::int32_t page_size = state.PageSize();
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
