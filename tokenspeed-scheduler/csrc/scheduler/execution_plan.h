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
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "scheduler/operations/inc.h"

namespace tokenspeed {

class ExecutionPlan {
public:
    template <typename OperationType>
    ExecutionPlan& With(OperationType operation) {
        operations_.emplace_back(std::move(operation));
        return *this;
    }

    const std::vector<Operation>& Operations() const { return operations_; }

    // Cache child pages newly assigned in this plan. Group identity is
    // required because one LCM parent can still contain live sibling children.
    // The runtime clears these exact byte ranges before transfers/forward.
    std::map<std::string, std::vector<std::int32_t>> pages_to_zero;

    // The plan separates its streams by executor, and the two remote
    // streams are the transfer peer's half of the role: work the peer runs
    // asynchronously, exactly as the model runs the ForwardBatch. Neither
    // occupies a batch slot or token budget, and the runtime submits them on
    // every round the plan carries one -- rounds with no batch included.
    //
    // remote_decode rides beside any forward work. remote_prefill does not:
    // a D-role round is either a decode batch, one remote admission, or a
    // local recovery prefill (see buildDecodeWorkerPlan).
    //
    // P role: completed prefills whose final chunk's result has landed; each
    // one's decode happens on the peer node, so its KV goes out. Rows are
    // self-contained (bootstrap token + drafter candidates).
    std::optional<ForwardBatch> remote_decode;
    // D role: admitted prompts whose prefill runs on the peer node; the
    // receive pulls their KV into the freshly admitted (and sanitized)
    // pages. The model sees the request only after RemotePrefillDone.
    std::optional<ForwardBatch> remote_prefill;

private:
    std::vector<Operation> operations_;
};

}  // namespace tokenspeed
