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

#include <spdlog/spdlog.h>

#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "fsm/pd_events.h"
#include "scheduler/outside_events/inc.h"
#include "scheduler/operations/progress.h"
#include "utils.h"

namespace tokenspeed {

void Scheduler::handleEvent(const pd::BootstrappedEvent& event) {
    Request* request = findRequest(event.request_id);
    if (request != nullptr && request->IsAnyOf<fsm::Bootstrapping, fsm::Prefetching>()) {
        request->Apply(fsm::BootstrappedEvent{});
    }
}

void Scheduler::handleEvent(const pd::FailedEvent& event) {
    Request* request = findRequest(event.request_id);
    if (request == nullptr || request->Is<fsm::Finished>()) {
        return;
    }
    releaseStorage(*request);
    request->Apply(fsm::AbortEvent{&coordinator_});
}

void Scheduler::handleEvent(const pd::SucceededEvent& event) {
    Request* request = findRequest(event.request_id);
    if (request == nullptr || request->IsAnyOf<fsm::Finishing, fsm::Finished>()) {
        return;
    }
    if (!request->Is<fsm::PrefillDone>() && !request->Is<fsm::Decoding>()) {
        throw std::logic_error("PD SucceededEvent received in state " + request->StateName());
    }
    // The transfer is done and the request leaves this engine: its computed
    // frontier is final here, so it finishes like any request -- the P's
    // prompt-end state and partial page, the D's final state and tail are
    // published when it settles.
    releaseStorage(*request);
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
    if (request->IsAnyOf<fsm::PrefillDone, fsm::Decoding, fsm::Finishing, fsm::Finished>()) {
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
        releaseStorage(*request);
        request->Apply(fsm::FinishEvent{&coordinator_});
    }
}

void Scheduler::settleFinishedRequests() {
    for (const std::unique_ptr<Request>& request : requests_) {
        if (!request->Is<fsm::Finishing>() || request->ResultsInFlight() > 0) {
            continue;
        }
        if (auto store = publishCompletedPages(*request)) {
            pending_write_back_operations_.push_back(std::move(*store));
        }
        request->Apply(fsm::ReleaseEvent{&coordinator_});
    }
}

std::optional<WriteBackOperation> Scheduler::publishCompletedPages(Request& request) {
    fsm::CacheProgress progress = request.CacheProgress();
    // The frontier is final: every forward has landed, so the live state slot
    // holds the state after the last consumed token and the KV rows below it
    // are complete.
    const std::int32_t num_computed_tokens = request.NumComputedTokens();
    RequestProgress completed =
        AdvanceRequestProgress(request, progress, num_computed_tokens, coordinator_.PrefixGranularity(),
                               /*stream_completed_to_host=*/false, PublicationPoint::kFinal,
                               /*allow_state_checkpoint=*/coordinator_.CheckpointGranularity() == 1 ||
                                   (config_.role != Role::kD && num_computed_tokens <= request.PrefillSize()));
    std::vector<std::string> store_hashes = progress.prefix_hashes;
    if (completed.completed_pages) {
        registerKvEventPublication(request, progress.prefix_hashes, *completed.completed_pages);
        coordinator_.CacheCompletedBlocks(request.BlockTablesRef(), completed, progress.access_epoch);
        if (completed.completed_pages->tail) {
            store_hashes.push_back(completed.completed_pages->tail->hash);
        }
        // This end state supersedes the checkpoint the turn resumed from
        // (MarkSupersededStateInput). Marked at the finish only: a request
        // that fails earlier (a failed PD transfer) publishes and marks
        // nothing, and an end state that is the input itself (a D request
        // that claimed its prompt-end checkpoint and leaves before any
        // decode) supersedes nothing.
        if (progress.resumed_tail_hash && !(completed.completed_pages->tail &&
                                            completed.completed_pages->tail->hash == *progress.resumed_tail_hash)) {
            coordinator_.MarkSupersededStateInput(*progress.resumed_tail_hash, progress.access_epoch);
        }
    }
    if (!config_.StreamsDeviceCacheToHost()) {
        return std::nullopt;
    }
    // Every published entry of the request streams: its pages, its tail and
    // the state checkpoint published with the frontier.
    coordinator_.QueueCachedBlocksForStore(store_hashes);
    // The request's pages are released right after this (ReleaseEvent); the
    // pinned ticket keeps them cached and unevictable until the copy ACKs.
    return tier_transfers_.StartPendingStores(StoreSourceGuard::kPinnedUntilAck, HostStoreCause::kFinish);
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
        releaseStorage(*request);
        request->Apply(fsm::AbortEvent{&coordinator_});
    }
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
    if (storage_transfers_ && storage_allowed_) {
        std::vector<CacheKey> keys;
        keys.reserve(published.size());
        for (const auto& entry : published) keys.push_back(entry.key);
        storage_transfers_->QueueBackups(keys);
    }
}

void Scheduler::handleEvent(const cache::LoadBackDone& event) {
    tier_transfers_.CompleteLoadBack(event.op_id);
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
    ReqPoolIndex request_pool_index = tier_transfers_.CompleteSnapshotRestore(event.op_id, /*publish=*/resumes);
    if (resumes) {
        request->Apply(fsm::RestoreDoneEvent{std::move(request_pool_index)});
    }
    // Otherwise the row and the blob slot return with the ticket: the request
    // that would have used them is gone.
}

}  // namespace tokenspeed
