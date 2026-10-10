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

#include <algorithm>
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
    if (Request* request = findRequest(event.request_id)) {
        if (pdTransferInFlight(*request)) {
            throw std::logic_error("PD Finish received while transfer pages are pinned");
        }
        if (request->Is<fsm::PrefillDone>() || request->Is<fsm::Decoding>()) {
            if (auto store = publishCompletedPages(*request)) {
                pending_write_back_operations_.push_back(std::move(*store));
            }
        }
        request->Apply(fsm::FinishEvent{&coordinator_});
    }
}

std::optional<WriteBackOperation> Scheduler::publishCompletedPages(Request& request) {
    const std::vector<std::span<const std::int32_t>> stable_prefix_pages = request.FullPrefixPages(true);
    fsm::CacheProgress progress = request.CacheProgress();
    const std::int32_t first_new_prefix_page = static_cast<std::int32_t>(progress.prefix_hashes.size());
    const std::int32_t num_stable_prefix_pages = static_cast<std::int32_t>(stable_prefix_pages.size());
    _assert(first_new_prefix_page <= num_stable_prefix_pages, "cache progress exceeds completed request pages");
    if (first_new_prefix_page != num_stable_prefix_pages) {
        const std::string previous_hash =
            progress.prefix_hashes.empty() ? std::string{} : progress.prefix_hashes.back();
        std::vector<std::string> new_hashes =
            AdvancePrefixHashes(stable_prefix_pages, first_new_prefix_page, previous_hash, num_stable_prefix_pages);
        progress.prefix_hashes.insert(progress.prefix_hashes.end(), std::make_move_iterator(new_hashes.begin()),
                                      std::make_move_iterator(new_hashes.end()));

        registerKvEventPrefixPages(request, progress.prefix_hashes, first_new_prefix_page);
        coordinator_.CacheCompletedBlocks(
            request.BlockTablesRef(),
            RequestProgress{
                .completed_pages =
                    CompletedPages{
                        .prefix_hashes = progress.prefix_hashes,
                        .first_new_prefix_page = first_new_prefix_page,
                        .boundary_kind = CacheBoundaryKind::kEndpoint,
                        .stream_completed_to_host = false,
                        .materialized_state_boundaries = progress.materialized_state_boundaries,
                    },
                .num_computed_tokens = request.TokenSize() - 1,
            },
            progress.access_epoch);
    }
    if (!config_.StreamsDeviceCacheToHost()) {
        return std::nullopt;
    }
    coordinator_.QueueCachedBlocksForStore(progress.prefix_hashes);
    const auto prefill_hashes = std::span<const std::string>{progress.prefix_hashes}.first(
        std::min(progress.prefix_hashes.size(),
                 static_cast<std::size_t>(request.PrefillSize() / coordinator_.PrefixGranularity())));
    coordinator_.QueueLatestSnapshotBlocksForStore(prefill_hashes);
    // The request's pages are released right after this (FinishEvent); the
    // pinned ticket keeps them cached and unevictable until the copy ACKs.
    return tier_transfers_.StartPendingStores(StoreSourceGuard::kPinnedUntilAck);
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
    if (Request* request = findRequest(event.request_id)) {
        request->Apply(fsm::AbortEvent{&coordinator_});
    }
}

void Scheduler::handleEvent(const forward::RecomputeRetract& event) {
    Request* request = findRequest(event.request_id);
    // Only a request with a forward out against its pages can have had that
    // forward skipped; a suspended or finished one has nothing to recompute.
    if (request == nullptr || !request->HoldsPages() || request->Is<fsm::Restoring>()) {
        return;
    }
    // The skipped forward's destination pages were never filled: publishing
    // would cache empty KV, so the request re-prefills through ordinary
    // admission like a newcomer.
    request->Apply(fsm::RecomputeRetractEvent{&coordinator_});
}

void Scheduler::handleEvent(const cache::WriteBackDone& event) {
    const std::vector<HostPublication> published = tier_transfers_.CompleteWriteBack(event.op_id);
    // A retraction image's L2 leg: every suspended request waiting for this
    // store (its own, or an earlier one carrying one of its keys) notes it
    // and follows the Host entries as published, which may differ from the
    // ticket's blocks it pinned when the index redirected a key to an entry
    // an L3 prefetch landed first.
    for (const std::unique_ptr<Request>& request : requests_) {
        const auto* retracted = request->GetIf<fsm::Retracted>();
        if (retracted != nullptr && retracted->WaitsForStore(event.op_id)) {
            request->Apply(fsm::StoreLandedEvent{event.op_id, published});
        }
    }
}

void Scheduler::handleEvent(const cache::LoadBackDone& event) {
    tier_transfers_.CompleteLoadBack(event.op_id, event.success);
}

void Scheduler::handleEvent(const cache::SnapshotDone& event) {
    const std::optional<std::string> request_id = tier_transfers_.CompleteSnapshotStore(event.op_id);
    if (!request_id) {
        return;  // unknown or duplicate ACK
    }
    Request* request = findRequest(*request_id);
    if (request != nullptr && request->Is<fsm::Retracted>()) {
        request->Apply(fsm::StoreLandedEvent{event.op_id, /*published=*/{}});
    }
}

void Scheduler::handleEvent(const cache::RestoreDone& event) {
    const std::optional<std::string> request_id = tier_transfers_.SnapshotRestoreRequest(event.op_id);
    if (!request_id) {
        return;  // unknown or duplicate ACK
    }
    Request* request = findRequest(*request_id);
    const fsm::Restoring* restoring = request == nullptr ? nullptr : request->GetIf<fsm::Restoring>();
    // Finished or aborted while restoring: the ACK only drops the pins.
    const bool resumes = restoring != nullptr && restoring->restore_op == event.op_id;
    if (resumes) {
        // The ACK republishes the L2-tier destinations into the Device index,
        // a KV-event mutation of boundaries whose descriptors DrainKvEvents
        // dropped when the victim's pages left the Device: register them
        // again first, as a first chunk does for the load-backs it issues.
        registerKvEventPrefixPages(*request, restoring->resources.cache_progress.prefix_hashes, 0);
    }
    tier_transfers_.CompleteSnapshotRestore(event.op_id, /*publish=*/resumes);
    if (resumes) {
        request->Apply(fsm::RestoreDoneEvent{});
    }
}

}  // namespace tokenspeed
