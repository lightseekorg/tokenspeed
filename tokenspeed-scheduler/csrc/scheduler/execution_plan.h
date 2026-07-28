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
#include <string>
#include <variant>
#include <vector>

#include "scheduler/operations/inc.h"

namespace tokenspeed {

enum class FlatTerminalReason {
    kPromptExceedsPoolCapacity,
    kDecodeExceedsPoolCapacity,
};

struct FlatTerminalError {
    std::string request_id;
    FlatTerminalReason reason{FlatTerminalReason::kPromptExceedsPoolCapacity};
    std::int32_t required_pages{0};
    std::int32_t capacity_pages{0};
    std::string message;
};

class ExecutionPlan {
public:
    template <typename OperationType>
    ExecutionPlan& With(OperationType operation) {
        operations_.emplace_back(operation);
        return *this;
    }

    template <typename OperationType>
    ExecutionPlan& With(std::vector<OperationType> ops) {
        for (auto& op : ops) {
            operations_.emplace_back(std::move(op));
        }
        return *this;
    }

    const std::vector<Operation>& Operations() const { return operations_; }

    // Flat KV-cache: requests terminalized this round as OOM -- the pool was wedged by
    // unretractable mid-prefill holders (possibly the request itself, or a mutual wedge)
    // with no Decoding/PrefillDone victim to retract. Always empty on the radix path.
    std::vector<std::string> flat_oom_request_ids;
    // Structured companion to flat_oom_request_ids. The legacy ID vector is
    // retained for runtime compatibility while callers migrate to precise
    // phase/reason diagnostics.
    std::vector<FlatTerminalError> flat_terminal_errors;

    // Flat KV-cache physical pages newly assigned to an owner in this plan.
    // The runtime zeros the complete page set before cache transfers/forward.
    // Cached prefix hits are intentionally absent.
    std::vector<std::int32_t> flat_page_ids_to_zero;

private:
    std::vector<Operation> operations_;
};

}  // namespace tokenspeed
