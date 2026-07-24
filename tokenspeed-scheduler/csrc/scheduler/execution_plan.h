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

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "scheduler/operations/inc.h"

namespace tokenspeed {

struct SchedulerAbort {
    std::string request_id;
    std::string message;
};

class ExecutionPlan {
public:
    template <typename OperationType>
    ExecutionPlan& With(OperationType operation) {
        operations_.emplace_back(std::move(operation));
        return *this;
    }

    template <typename OperationType>
    ExecutionPlan& With(std::vector<OperationType> ops) {
        for (auto& op : ops) {
            operations_.emplace_back(std::move(op));
        }
        return *this;
    }

    ExecutionPlan& WithSchedulerAborts(std::vector<SchedulerAbort> aborts) {
        scheduler_aborts_ = std::move(aborts);
        return *this;
    }

    const std::vector<Operation>& Operations() const { return operations_; }
    const std::vector<SchedulerAbort>& SchedulerAborts() const { return scheduler_aborts_; }

    // Flat KV-cache: requests terminalized this round as OOM -- the pool was wedged by
    // unretractable mid-prefill holders (possibly the request itself, or a mutual wedge)
    // with no Decoding/PrefillDone victim to retract. Always empty on the radix path.
    std::vector<std::string> flat_oom_request_ids;

private:
    std::vector<Operation> operations_;
    std::vector<SchedulerAbort> scheduler_aborts_;
};

}  // namespace tokenspeed
