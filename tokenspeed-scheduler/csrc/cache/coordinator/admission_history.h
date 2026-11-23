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
#include <cstdint>
#include <vector>

#include "cache/core/cache_types.h"
#include "cache/prefix/checkpoint_tree.h"

namespace tokenspeed {

// AdmissionHistory: the access epoch clock every admission, publication and
// acquisition is dated by, and the new-request admissions checkpoint
// retention ranks by -- the retention window's cutoff (CheckpointAgingFor)
// and how many admissions passed between a resumed checkpoint's last use and
// its resume (CheckpointResumeGaps).
class AdmissionHistory {
public:
    // checkpoints: the tree whose checkpoints CheckpointAgingFor ranks.
    // checkpoint_retention_requests: the retention window, in new-request
    // admissions; 0 ranks every checkpoint by its own recency.
    AdmissionHistory(const CheckpointTree& checkpoints, std::int64_t checkpoint_retention_requests);
    AdmissionHistory(const AdmissionHistory&) = delete;
    AdmissionHistory& operator=(const AdmissionHistory&) = delete;

    // Issues the next access epoch. Issued epochs start at 1.
    std::uint64_t IssueAccessEpoch() { return ++next_access_epoch_; }
    // The newest epoch issued; 0 before any.
    std::uint64_t LatestAccessEpoch() const { return next_access_epoch_; }
    // A new request's admission at `access_epoch`.
    void RecordAdmission(std::uint64_t access_epoch);
    // A resume from a checkpoint last used at `last_used_epoch`.
    void RecordResumeGap(std::uint64_t last_used_epoch);
    // How an eviction plan on `tier` ranks State checkpoints that are not
    // leaves (cache-concepts.md, checkpoint retention).
    CheckpointAging CheckpointAgingFor(CacheTier tier) const;
    // New requests admitted between a resumed checkpoint's last use and the
    // resume: index 0 counts none, index b >= 1 counts [2^(b-1), 2^b), and
    // the last index counts gaps as long as the admissions kept, or longer.
    const std::vector<std::uint64_t>& CheckpointResumeGaps() const { return checkpoint_resume_gaps_; }

private:
    static constexpr std::size_t kResumeGapAdmissions{std::size_t{1} << 16};

    const CheckpointTree& checkpoints_;
    const std::int64_t checkpoint_retention_requests_;
    std::uint64_t next_access_epoch_{0};
    // The access epochs of the latest new-request admissions, oldest first
    // from num_admissions_ (a ring); it covers the retention window and the
    // resume gaps.
    std::vector<std::uint64_t> admission_epochs_;
    std::uint64_t num_admissions_{0};
    std::vector<std::uint64_t> checkpoint_resume_gaps_;
};

}  // namespace tokenspeed
