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

#include <memory>
#include <vector>

#include "cache/coordinator/eviction_policy.h"
#include "cache/coordinator/gds_trace.h"
#include "cache/coordinator/victim_set.h"

namespace tokenspeed {
struct TieredGdsPlan;

// Commit data belongs to the whole selection, not to each physical victim.
// Only GDS plans allocate it; abandoning a plan has no cache side effects.
struct GdsEvictionCommit {
    GdsEvictionStats stats;
    GdsTrace trace;
    std::vector<ContentHash> fresh_rejections;
    std::shared_ptr<const TieredGdsPlan> tiered;
};

struct PlannedEviction {
    std::vector<VictimSet> victims;
    std::unique_ptr<GdsEvictionCommit> gds;
};

}  // namespace tokenspeed
