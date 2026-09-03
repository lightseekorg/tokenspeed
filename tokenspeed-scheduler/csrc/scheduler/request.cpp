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

#include "scheduler/request.h"

#include <stdexcept>

#include "fsm/forward_events.h"

namespace tokenspeed {

Request::Request(const RequestSpec& spec, std::int32_t prefix_granularity, Role role)
    : id_{spec.request_id},
      token_container_{spec.tokens},
      submitted_prompt_size_{static_cast<std::int32_t>(spec.tokens.size())},
      max_new_tokens_{spec.max_new_tokens},
      prefix_granularity_{prefix_granularity},
      state_{role == Role::kFused ? fsm::State{fsm::Submitted{&token_container_, prefix_granularity}}
                                  : fsm::State{fsm::Bootstrapping{&token_container_, prefix_granularity}}} {}

PrefillInfo Request::CurrentPrefillInfo() const {
    return std::visit(
        Overloaded{
            [](const fsm::Prefilling& state) { return state.CurrentPrefillInfo(); },
            [](const fsm::RemotePrefilling& state) { return state.CurrentPrefillInfo(); },
            [](const fsm::PrefillDone& state) { return state.CurrentPrefillInfo(); },
            [](const fsm::PrefillAwaitingResult& state) { return state.CurrentPrefillInfo(); },
            [this](const auto&) -> PrefillInfo {
                throw std::logic_error(
                    "Request::CurrentPrefillInfo: expected Prefilling, RemotePrefilling or PrefillDone; got " +
                    StateName());
            },
        },
        state_);
}

fsm::ForwardState& Request::forwardState(const char* operation) {
    fsm::ForwardState* result = std::visit(
        []<typename State>(State& state) -> fsm::ForwardState* {
            if constexpr (std::derived_from<State, fsm::ForwardState>) {
                return &state;
            }
            return nullptr;
        },
        state_);
    if (result == nullptr) {
        throw std::logic_error(std::string{"Request::"} + operation + ": expected a forward state; got " + StateName());
    }
    return *result;
}

const fsm::ForwardState& Request::forwardState(const char* operation) const {
    const fsm::ForwardState* result = std::visit(
        []<typename State>(const State& state) -> const fsm::ForwardState* {
            if constexpr (std::derived_from<State, fsm::ForwardState>) {
                return &state;
            }
            return nullptr;
        },
        state_);
    if (result == nullptr) {
        throw std::logic_error(std::string{"Request::"} + operation + ": expected a forward state; got " + StateName());
    }
    return *result;
}

}  // namespace tokenspeed
