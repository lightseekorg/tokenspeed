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

#include "scheduler/scheduler.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "fsm/pd_events.h"
#include "scheduler/outside_events/inc.h"
#include "cache/prefix/prefix_hasher.h"
#include "utils.h"

namespace tokenspeed {

void Scheduler::handleEvent(const pd::BootstrappedEvent& event) {
    Request* request = findRequest(event.request_id);
    if (request != nullptr && request->Is<fsm::Bootstrapping>()) {
        request->Apply(fsm::BootstrappedEvent{});
    }
}

void Scheduler::handleEvent(const pd::FailedEvent& event) {
    Request* request = findRequest(event.request_id);
    if (request == nullptr || request->Is<fsm::Finished>()) {
        return;
    }
    pd_transfer_pins_.erase(event.request_id);
    request->Apply(fsm::AbortEvent{&coordinator_});
}

void Scheduler::handleEvent(const pd::SucceededEvent& event) {
    Request* request = findRequest(event.request_id);
    if (request == nullptr || request->Is<fsm::Finished>()) {
        return;
    }
    if (!request->Is<fsm::PrefillDone>() && !request->Is<fsm::Decoding>()) {
        throw std::logic_error("PD SucceededEvent received in state " + request->StateName());
    }
    pd_transfer_pins_.erase(event.request_id);
    request->Apply(fsm::FinishEvent{&coordinator_});
}

void Scheduler::handleEvent(const pd::RemotePrefillDoneEvent& event) {
    Request* request = findRequest(event.request_id);
    if (request == nullptr) {
        return;
    }
    if (request->Is<fsm::RemotePrefilling>()) {
        if (event.bootstrap_token < 0) {
            throw std::invalid_argument("PD RemotePrefillDoneEvent requires a non-negative bootstrap token");
        }
        pd_transfer_pins_.erase(event.request_id);
        request->Apply(fsm::RemotePrefillDoneEvent{event.bootstrap_token});
        return;
    }
    if (request->Is<fsm::PrefillDone>() || request->Is<fsm::Decoding>() || request->Is<fsm::Finished>()) {
        return;
    }
    throw std::logic_error("PD RemotePrefillDoneEvent received before destination admission; state=" +
                           request->StateName());
}

void Scheduler::handleEvent(const forward::Finish& event) {
    if (config_.enable_pd_cache && pd_transfer_pins_.contains(event.request_id)) {
        throw std::logic_error("PD Finish received while transfer pages are pinned");
    }
    if (Request* request = findRequest(event.request_id)) {
        if (request->Is<fsm::PrefillDone>() || request->Is<fsm::Decoding>()) {
            publishCompletedPages(*request);
        }
        request->Apply(fsm::FinishEvent{&coordinator_});
    }
}

void Scheduler::publishCompletedPages(Request& request) {
    const std::vector<std::span<const std::int32_t>> stable_prefix_pages = request.FullPrefixPages(true);
    fsm::CacheProgress progress = request.CacheProgress();
    const std::int32_t first_new_prefix_page = static_cast<std::int32_t>(progress.prefix_hashes.size());
    const std::int32_t num_stable_prefix_pages = static_cast<std::int32_t>(stable_prefix_pages.size());
    _assert(first_new_prefix_page <= num_stable_prefix_pages, "cache progress exceeds completed request pages");
    if (first_new_prefix_page == num_stable_prefix_pages) {
        return;
    }

    const std::string previous_hash = progress.prefix_hashes.empty() ? std::string{} : progress.prefix_hashes.back();
    std::vector<std::string> new_hashes =
        AdvancePrefixHashes(stable_prefix_pages, first_new_prefix_page, previous_hash, num_stable_prefix_pages);
    progress.prefix_hashes.insert(progress.prefix_hashes.end(), std::make_move_iterator(new_hashes.begin()),
                                  std::make_move_iterator(new_hashes.end()));

    std::vector<CacheKey> event_keys =
        registerKvEventPrefixPages(request, progress.prefix_hashes, first_new_prefix_page);
    coordinator_.CacheCompletedBlocks(request.BlockTablesRef(), progress.prefix_hashes, progress.access_epoch,
                                      first_new_prefix_page, request.TokenSize() - 1, CacheBoundaryKind::kEndpoint);
    discardUncachedKvEventPages(event_keys);
}

void Scheduler::handleEvent(const forward::UpdateReserveNumTokens& event) {
    if (Request* request = findRequest(event.request_id)) {
        request->Apply(fsm::UpdateReserveNumTokensEvent{event.reserve_num_tokens_in_next_schedule_event});
    }
}

void Scheduler::handleEvent(const forward::ExtendResult& event) {
    if (Request* request = findRequest(event.request_id)) {
        request->NoteResultLanded();
        request->Apply(fsm::ExtendResultEvent{event.tokens});
        if (!event.spec_candidate_ids.empty()) {
            request->StoreSpecCandidates(event.spec_candidate_ids);
        }
    }
}

void Scheduler::handleEvent(const forward::Abort& event) {
    pd_transfer_pins_.erase(event.request_id);
    if (Request* request = findRequest(event.request_id)) {
        request->Apply(fsm::AbortEvent{&coordinator_});
    }
}

void Scheduler::handleEvent(const cache::WriteBackDone& event) {
    tier_transfers_.CompleteWriteBack(event.op_id);
}

void Scheduler::handleEvent(const cache::LoadBackDone& event) {
    tier_transfers_.CompleteLoadBack(event.op_id);
}

}  // namespace tokenspeed
