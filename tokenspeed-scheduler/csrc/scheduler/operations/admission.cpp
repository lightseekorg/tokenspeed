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

#include "scheduler/operations/admission.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fsm/forward_states.h"
#include "fsm/pd_events.h"
#include "fsm/peer_states.h"
#include "utils.h"

namespace tokenspeed {

void RecordAdmittedPages(ExecutionPlan& plan, std::span<const std::string> cache_group_ids,
                         const AdmissionResult& result) {
    _assert(
        result.new_page_ids.size() == cache_group_ids.size() && result.pages_to_copy.size() == cache_group_ids.size(),
        "admission page groups must match scheduler config");
    for (std::size_t i = 0; i < cache_group_ids.size(); ++i) {
        auto& pending = plan.pages_to_zero[cache_group_ids[i]];
        auto& copies = plan.pages_to_copy[cache_group_ids[i]];
        // The runtime copies before it zeroes: a page both copied into and
        // zeroed in one plan would lose the copy.
        for (const std::int32_t page : result.new_page_ids[i]) {
            _assert(std::ranges::find(copies, page, &std::pair<std::int32_t, std::int32_t>::second) == copies.end(),
                    "a copy destination is never zeroed");
        }
        pending.insert(pending.end(), result.new_page_ids[i].begin(), result.new_page_ids[i].end());
        // The runtime reads every copy's source as the plan found it, so a
        // copy from a block an earlier copy of the plan fills -- the retry's
        // private copy of a tail block its lone-request compaction just moved
        // -- reads that copy's source instead; composed back onto its own
        // source, the block holds those bytes already (the arena refuses a
        // self copy).
        for (auto [source, destination] : result.pages_to_copy[i]) {
            _assert(
                std::ranges::find(copies, destination, &std::pair<std::int32_t, std::int32_t>::second) == copies.end(),
                "a plan writes each copy destination once");
            _assert(std::ranges::find(pending, destination) == pending.end(), "a copy destination is never zeroed");
            if (const auto earlier = std::ranges::find(copies, source, &std::pair<std::int32_t, std::int32_t>::second);
                earlier != copies.end()) {
                source = earlier->first;
                if (source == destination) {
                    continue;
                }
            }
            copies.emplace_back(source, destination);
        }
    }
}

StepCopyIns StepCopyInsOf(Request& request, const RequestProgress& progress, std::optional<CopyInPoint> copy_in_point) {
    // The publication of a prompt end the owner keeps writing past spends the
    // copy-in blocks the request holds for it; no other admission does. The
    // request holds them from the admission that reserves for the point
    // (copy_in_point) to the one that spends them, and a compaction in
    // between moves them with their slots.
    const bool copies_in = progress.completed_pages && progress.completed_pages->tail &&
                           progress.completed_pages->owner_tail == OwnerTail::kCopiedIn;
    const std::span<std::vector<CacheBlockRef>> held =
        copies_in || copy_in_point ? std::span<std::vector<CacheBlockRef>>(request.CopyInBlocksRef())
                                   : std::span<std::vector<CacheBlockRef>>{};
    return StepCopyIns{.held = held, .spent = copies_in ? held : std::span<std::vector<CacheBlockRef>>{}};
}

bool OthersHoldPages(std::span<const std::unique_ptr<Request>> requests, const Request& request) {
    return std::ranges::any_of(requests, [&request](const std::unique_ptr<Request>& other) {
        return other.get() != &request && other->HoldsPages();
    });
}

bool PdTransferInFlight(Role role, const Request& request) {
    // A peer landing's reads write its pages on any role.
    if (request.Is<fsm::PeerLanding>()) {
        return true;
    }
    switch (role) {
        case Role::kD:
            return request.IsAnyOf<fsm::RemotePrefilling, fsm::PdTransferring>();
        case Role::kP:
            // Every page-holding state but Finishing, PdTransferring
            // included: SucceededEvent moves a request to Finishing once the
            // peer has read its pages.
            return request.HoldsPages() && !request.Is<fsm::Finishing>();
        case Role::kFused:
            return false;
    }
    return false;
}

}  // namespace tokenspeed
