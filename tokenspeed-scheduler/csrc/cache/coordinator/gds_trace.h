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

#include <span>
#include <vector>

#include <spdlog/spdlog.h>

#include "cache/core/cache_types.h"

namespace tokenspeed {

// Optional plan-local diagnostics. Failed trials discard these with the plan;
// only committed decisions are logged. No per-key history survives a plan.
struct GdsScoreObservation {
    enum class Cause {
        kInitial,
        kPublication,
        kExclusiveCapacity,
        kFallback,
        kDependencies,
        kJointRoutes,
        kEviction,
        kRejection,
        kDemotion,
        kCapacityFallback
    };
    ContentHash hash;
    CacheTier tier;
    Cause cause;
    std::int32_t tokens;
    std::int32_t fallback_tokens;
    double stamp;
    double relief;
    double score;
    double watermark;
};
using GdsTrace = std::vector<GdsScoreObservation>;

inline void ObserveGdsScore(GdsTrace& trace, const GdsScoreObservation& event) {
    if (spdlog::should_log(spdlog::level::debug)) trace.push_back(event);
}

inline const char* GdsScoreCauseName(GdsScoreObservation::Cause cause) {
    using Cause = GdsScoreObservation::Cause;
    switch (cause) {
        case Cause::kInitial:
            return "initial";
        case Cause::kPublication:
            return "publication_preview";
        case Cause::kExclusiveCapacity:
            return "exclusive_capacity";
        case Cause::kFallback:
            return "fallback";
        case Cause::kDependencies:
            return "fallback_and_capacity";
        case Cause::kJointRoutes:
            return "joint_routes";
        case Cause::kEviction:
            return "eviction";
        case Cause::kRejection:
            return "retention_rejection";
        case Cause::kDemotion:
            return "explicit_demotion";
        case Cause::kCapacityFallback:
            return "capacity_fallback";
    }
    std::terminate();
}

inline void CommitGdsTrace(std::span<const GdsScoreObservation> trace) {
    for (const auto& event : trace) {
        spdlog::debug(
            "[GDS] committed hash={} tier={} cause={} tokens={} fallback={} distance={} stamp={} relief={} score={} "
            "watermark={}",
            event.hash.ToString(), event.tier == CacheTier::kDevice ? "device" : "host", GdsScoreCauseName(event.cause),
            event.tokens, event.fallback_tokens, event.tokens - event.fallback_tokens, event.stamp, event.relief,
            event.score, event.watermark);
    }
}
}  // namespace tokenspeed
