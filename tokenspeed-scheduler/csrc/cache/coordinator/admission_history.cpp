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

#include "cache/coordinator/admission_history.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "utils.h"

namespace tokenspeed {

AdmissionHistory::AdmissionHistory(const CheckpointTree& checkpoints, std::int64_t checkpoint_retention_requests)
    : checkpoints_{checkpoints}, checkpoint_retention_requests_{checkpoint_retention_requests} {
    _assert(checkpoint_retention_requests_ >= 0 && checkpoint_retention_requests_ <= kMaxCheckpointRetentionRequests,
            "checkpoint retention requests out of range");
    // Enough admissions to place the retention window and to tell the resume
    // gaps a choice of it reads.
    admission_epochs_.resize(
        std::max<std::size_t>(static_cast<std::size_t>(checkpoint_retention_requests_), kResumeGapAdmissions));
    checkpoint_resume_gaps_.resize(static_cast<std::size_t>(std::bit_width(admission_epochs_.size() - 1)) + 2);
}

CheckpointAging AdmissionHistory::CheckpointAgingFor(CacheTier tier) const {
    if (checkpoint_retention_requests_ == 0 || checkpoints_.Size() == 0) {
        return CheckpointAging{.tier = tier};
    }
    const auto window = static_cast<std::uint64_t>(checkpoint_retention_requests_);
    // Before that many admissions every entry is recent; epoch 0 stays first.
    const std::uint64_t cutoff_epoch =
        num_admissions_ < window ? 1 : admission_epochs_[(num_admissions_ - window) % admission_epochs_.size()];
    return CheckpointAging{.tree = &checkpoints_, .tier = tier, .cutoff_epoch = cutoff_epoch};
}

void AdmissionHistory::RecordAdmission(std::uint64_t access_epoch) {
    admission_epochs_[num_admissions_ % admission_epochs_.size()] = access_epoch;
    ++num_admissions_;
}

void AdmissionHistory::RecordResumeGap(std::uint64_t last_used_epoch) {
    const std::uint64_t kept = std::min<std::uint64_t>(num_admissions_, admission_epochs_.size());
    const std::uint64_t oldest = num_admissions_ - kept;
    // The kept admissions' epochs increase from the oldest: find the first
    // one after the checkpoint's last use.
    std::uint64_t low = 0;
    std::uint64_t high = kept;
    while (low < high) {
        const std::uint64_t mid = low + (high - low) / 2;
        if (admission_epochs_[(oldest + mid) % admission_epochs_.size()] > last_used_epoch) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    const std::uint64_t gap = kept - low;
    const bool counted = (low > 0 || oldest == 0) && gap < admission_epochs_.size();
    ++checkpoint_resume_gaps_[counted ? static_cast<std::size_t>(std::bit_width(gap))
                                      : checkpoint_resume_gaps_.size() - 1];
}

}  // namespace tokenspeed
