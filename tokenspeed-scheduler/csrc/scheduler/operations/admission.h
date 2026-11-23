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

// Admission helpers the scheduler and its lone-request recovery
// (LoneRequestRecovery) share.

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "scheduler/execution_plan.h"
#include "scheduler/request.h"
#include "scheduler/types.h"

namespace tokenspeed {

// Records an admission's fresh pages (to zero) and copies on the plan, a copy
// from a block an earlier copy of the plan fills reading that copy's source.
// Asserts that each copy destination is written once and never zeroed in the
// plan. cache_group_ids names the plan's groups in coordinator order.
void RecordAdmittedPages(ExecutionPlan& plan, std::span<const std::string> cache_group_ids,
                         const AdmissionResult& result);

// The copy-in blocks a step's admission deals with: every one the request
// holds (held: for a step that reserves for a copy-in point or publishes the
// one it holds them for), and the ones its publication spends.
struct StepCopyIns {
    std::span<std::vector<CacheBlockRef>> held;
    std::span<std::vector<CacheBlockRef>> spent;
};
StepCopyIns StepCopyInsOf(Request& request, const RequestProgress& progress, std::optional<CopyInPoint> copy_in_point);

// Whether a request other than `request` holds pages.
bool OthersHoldPages(std::span<const std::unique_ptr<Request>> requests, const Request& request);

// Whether a PD transfer is out against the request's pages on this role.
bool PdTransferInFlight(Role role, const Request& request);

}  // namespace tokenspeed
