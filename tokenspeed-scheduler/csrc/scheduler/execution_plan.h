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

#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "scheduler/operations/inc.h"

namespace tokenspeed {

class ExecutionPlan {
public:
    void ReserveOperations(std::size_t capacity) { operations_.reserve(capacity); }

    template <typename OperationType>
    ExecutionPlan& With(OperationType operation) noexcept {
        static_assert(std::is_nothrow_constructible_v<Operation, OperationType&&>);
        if (operations_.size() == operations_.capacity()) {
            std::terminate();
        }
        operations_.emplace_back(std::move(operation));
        return *this;
    }

    const std::vector<Operation>& Operations() const { return operations_; }

    // Flat KV-cache: requests terminalized this round as OOM -- the pool was wedged by
    // unretractable mid-prefill holders (possibly the request itself, or a mutual wedge)
    // with no Decoding/PrefillDone victim to retract. Always empty on the radix path.
    std::vector<std::string> flat_oom_request_ids;

    void TakeFlatOomRequestIds(std::vector<std::string>& outbox) noexcept { flat_oom_request_ids.swap(outbox); }

private:
    std::vector<Operation> operations_;
};

}  // namespace tokenspeed
