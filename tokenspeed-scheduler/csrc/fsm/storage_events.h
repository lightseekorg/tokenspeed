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

#include <optional>
#include <utility>

#include "fsm/base_event.h"
#include "fsm/storage_states.h"

namespace tokenspeed::fsm {

struct StartPrefetchEvent : InvalidTransitionHandler<StartPrefetchEvent> {
    using InvalidTransitionHandler<StartPrefetchEvent>::operator();

    StartPrefetchEvent(std::uint64_t operation_id, std::int64_t deadline_us, std::vector<std::string> hashes,
                       std::int32_t max_tokens, std::optional<TailPublication> tail)
        : operation_id_{operation_id},
          deadline_us_{deadline_us},
          hashes_{std::move(hashes)},
          max_tokens_{max_tokens},
          tail_{std::move(tail)} {}

    Prefetching operator()(Submitted&& state) { return start(std::move(state)); }
    Prefetching operator()(Bootstrapping&& state) { return start(std::move(state)); }

private:
    template <typename State>
    Prefetching start(State&& state) {
        state.admission.storage_prepared = true;
        return Prefetching{std::move(state),   operation_id_, deadline_us_,
                           std::move(hashes_), max_tokens_,   std::move(tail_)};
    }
    std::uint64_t operation_id_;
    std::int64_t deadline_us_;
    std::vector<std::string> hashes_;
    std::int32_t max_tokens_;
    std::optional<TailPublication> tail_;
};

struct FinishPrefetchEvent : InvalidTransitionHandler<FinishPrefetchEvent> {
    using InvalidTransitionHandler<FinishPrefetchEvent>::operator();
    explicit FinishPrefetchEvent(std::optional<StorageReadyHold> hold) : hold_{std::move(hold)} {}

    Prefetching::Continuation operator()(Prefetching&& state) {
        state.Admission().storage_hold = std::move(hold_);
        return std::move(state.continuation);
    }

private:
    std::optional<StorageReadyHold> hold_;
};

}  // namespace tokenspeed::fsm
