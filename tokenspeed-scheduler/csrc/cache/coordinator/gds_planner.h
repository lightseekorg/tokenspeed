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

#include <memory_resource>
#include <optional>
#include <span>
#include <vector>

#include "cache/coordinator/admission_candidates.h"
#include "cache/coordinator/eviction_plan.h"
#include "cache/coordinator/victim_set.h"

namespace tokenspeed {

// All state is plan-local. A failed preview changes neither residency nor the
// GDS watermark. Real group/bucket placement remains CacheEvictionPlan's job.
std::optional<std::vector<VictimSet>> PlanGdsEviction(const CacheStore& store,
                                                      const AdmissionCandidateConstraints& constraints,
                                                      std::span<const BlockPlacementDemand> demands,
                                                      CacheEvictionPlan& capacity, std::pmr::memory_resource* resource);

}  // namespace tokenspeed
