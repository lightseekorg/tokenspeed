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
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "fsm/forward_states.h"
#include "fsm/pd_states.h"

namespace tokenspeed::fsm {

// No Device pages or request slot. The continuation records the independent
// PD dependency: its ACK changes Bootstrapping to Submitted without ending
// this wait. Native I/O and its Host references belong to the transfer manager.
struct Prefetching {
    using Continuation = std::variant<Bootstrapping, Submitted>;
    Continuation continuation;
    std::uint64_t operation_id;
    std::int64_t deadline_us;
    std::vector<std::string> hashes;
    std::int32_t max_tokens;
    std::optional<TailPublication> tail;

    AdmissionResources& Admission() {
        return std::visit([](auto& state) -> AdmissionResources& { return state.admission; }, continuation);
    }
    const AdmissionResources& Admission() const {
        return std::visit([](const auto& state) -> const AdmissionResources& { return state.admission; }, continuation);
    }
    bool AwaitsBootstrap() const { return std::holds_alternative<Bootstrapping>(continuation); }
};

}  // namespace tokenspeed::fsm
