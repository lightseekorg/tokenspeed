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
#include <memory>

#include "utils.h"

namespace tokenspeed {
struct CheckpointProductionStats {
    std::uint64_t planned{0};
    std::uint64_t allocated{0};
    std::uint64_t materialized{0};
    std::uint64_t published{0};
    std::uint64_t cancelled{0};
    std::uint64_t extra_blocks{0};
    std::uint64_t chunk_end_fallbacks{0};
    std::uint64_t fresh_rejected{0};
    std::uint64_t rejected_extra_blocks{0};
    std::uint64_t rejected_unmeasured{0};
};

// Shared only across a staged admission and its rollback copy. It owns no
// blocks: CacheProgress and the publication holder own the producer output.
// Observing computed progress survives an admission retry; publication does not.
class CheckpointLifecycle {
public:
    CheckpointLifecycle(std::shared_ptr<CheckpointProductionStats> stats, std::uint64_t blocks)
        : stats_{std::move(stats)}, blocks_{blocks} {
        ++stats_->allocated;
        stats_->extra_blocks += blocks;
    }
    CheckpointLifecycle(const CheckpointLifecycle&) = delete;
    CheckpointLifecycle& operator=(const CheckpointLifecycle&) = delete;
    ~CheckpointLifecycle() {
        if (!published_) ++stats_->cancelled;
    }
    void Materialize() {
        if (!materialized_) {
            materialized_ = true;
            ++stats_->materialized;
        }
    }
    void Publish() {
        _assert(materialized_, "uncomputed output cannot be published");
        if (!published_) {
            published_ = true;
            ++stats_->published;
        }
    }

    // A receipt survives publication with its recovery point, but owns no
    // cache blocks. Only committed fresh logical loss charges it, once.
    void RejectFresh() {
        _assert(published_, "only published output can lose retention");
        if (rejected_) return;
        rejected_ = true;
        ++stats_->fresh_rejected;
        stats_->rejected_extra_blocks += blocks_;
        // Runtime bytes and marginal producer GPU time are not yet joined.
        // Unknown cost must remain visible rather than being reported as zero.
        ++stats_->rejected_unmeasured;
    }

private:
    std::shared_ptr<CheckpointProductionStats> stats_;
    const std::uint64_t blocks_;
    bool rejected_{false};
    bool materialized_{false};
    bool published_{false};
};
}  // namespace tokenspeed
