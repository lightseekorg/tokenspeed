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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "cache/core/cache_types.h"
#include "core/token_container.h"
#include "fsm/forward_states.h"
#include "resource/allocator/req_pool_allocator.h"
#include "scheduler/request_spec.h"

// States that exist only under PD disaggregation: a fused engine never
// enters any of them. The universal lifecycle lives in forward_states.h;
// these extend it with the P and D roles' parking states.

namespace tokenspeed {
namespace fsm {
struct Bootstrapping {
    Bootstrapping(TokenContainer* _token_container, std::int32_t _prefix_granularity)
        : token_container{_token_container}, prefix_granularity{_prefix_granularity} {}

    TokenContainer* token_container{};
    std::int32_t prefix_granularity{};
};

// The peer node is prefilling this prompt: this engine admitted and holds
// the destination pages, and nothing here is schedulable work -- the state
// advances only when RemotePrefillDoneEvent lands with the bootstrap token.
// A state of its own, not Prefilling with a source flag: the two share the
// data layout (hence the inheritance) but no lifecycle.
struct RemotePrefilling : public Prefilling {
    using Prefilling::Prefilling;
};

// P role: every chunk is scheduled, but the FINAL chunk's result has not
// landed yet -- and the remote decode that hands this prompt to the peer
// needs the bootstrap token that arrives with it. Nothing is schedulable
// here; the only event this state accepts is that result, which turns it
// into PrefillDone.
//
// A sibling of PrefillDone, not a subclass: `Is<PrefillDone>()` must mean
// "schedulable" with no exceptions, and inheritance would make the answer
// depend on how the caller dispatches (holds_alternative vs derived_from vs
// an overload set).
struct PrefillAwaitingResult : public ForwardState {
    PrefillAwaitingResult(TokenContainer* token_container, std::int32_t prefix_granularity,
                          std::unique_ptr<ReqPoolIndex> req_pool_index, TokenContainer::Window window,
                          std::int32_t reserve_num_tokens_in_next_schedule_event, std::vector<BlockTable> block_tables,
                          CacheProgress cache_progress)
        : ForwardState(token_container, prefix_granularity, std::move(req_pool_index), std::move(block_tables),
                       std::move(cache_progress)),
          window{window},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    std::int32_t ReserveNumTokensInNextScheduleEvent() const { return reserve_num_tokens_in_next_schedule_event_; }
    PrefillInfo CurrentPrefillInfo() const {
        return PrefillInfo{
            .input_ids = token_container_->TokenSlice(window),
            .shifted_input_ids = ComputeShiftedInputIds(token_container_, window),
            .already_scheduled_len = window.begin,
            .extend_len = window.size,
        };
    }
    void ExtendResultTokens(const std::vector<std::int32_t>& result_tokens) { token_container_->Extend(result_tokens); }

    TokenContainer::Window window{};

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
};

}  // namespace fsm
}  // namespace tokenspeed
