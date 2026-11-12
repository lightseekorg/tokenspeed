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
#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "fsm/pd_events.h"
#include "scheduler/outside_events/inc.h"
#include "scheduler/operations/prefill_chunk.h"
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
    abortRequest(*request);
}

void Scheduler::handleEvent(const pd::TransferStoppedEvent& event) {
    Request* request = findRequest(event.request_id);
    fsm::PdTransferring* transferring = request == nullptr ? nullptr : request->GetIf<fsm::PdTransferring>();
    if (transferring == nullptr) {
        return;
    }
    // Transport retirement does not fence this engine's queued forwards.
    if (request->ResultsInFlight() > 0) {
        transferring->transfer_stopped = true;
        return;
    }
    request->Apply(fsm::TransferStoppedEvent{&coordinator_});
}

void Scheduler::handleEvent(const pd::SucceededEvent& event) {
    Request* request = findRequest(event.request_id);
    if (request == nullptr || request->IsAnyOf<fsm::Finishing, fsm::Finished>()) {
        return;
    }
    if (request->Is<fsm::PdTransferring>()) {
        // A success also retires the transfer, but local forwards may remain.
        handleEvent(pd::TransferStoppedEvent{event.request_id});
        return;
    }
    if (!request->Is<fsm::PrefillDone>() && !request->Is<fsm::Decoding>()) {
        throw std::logic_error("PD SucceededEvent received in state " + request->StateName());
    }
    // The transfer is done and the request leaves this engine: its computed
    // frontier is final here, so it finishes like any request -- the P's
    // prompt-end state and partial page, the D's final state and tail are
    // published when it settles.
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
    if (request->Is<fsm::PdTransferring>()) {
        // A success also retires the transfer, but local forwards may remain.
        handleEvent(pd::TransferStoppedEvent{event.request_id});
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
        if (request->Is<fsm::PdTransferring>()) {
            return;  // it ended already; only its transfer is still stopping
        }
        if (pdTransferInFlight(*request)) {
            throw std::logic_error("PD Finish received while transfer pages are pinned");
        }
        request->Apply(fsm::FinishEvent{&coordinator_});
    }
}

void Scheduler::settleFinishedRequests() {
    for (const std::unique_ptr<Request>& request : requests_) {
        const auto* transferring = request->GetIf<fsm::PdTransferring>();
        if (transferring != nullptr && transferring->transfer_stopped && request->ResultsInFlight() == 0) {
            request->Apply(fsm::TransferStoppedEvent{&coordinator_});
            continue;
        }
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
    // The frontier is final: every forward has landed, so the State slot the
    // last kept token's forward wrote holds the state after it (slot 1, or
    // slot 0 behind a first decode that ran as a padding row:
    // CompletedPages::state_slot) and the KV rows below it are complete.
    const std::int32_t num_computed_tokens = request.NumComputedTokens();
    RequestProgress completed = AdvanceRequestProgress(
        request, progress, num_computed_tokens, coordinator_.PrefixGranularity(),
        /*stream_completed_to_host=*/false,
        finishCheckpointIsDead(request, progress) ? PublicationPoint::kChunk : PublicationPoint::kFinal,
        /*allow_state_checkpoint=*/coordinator_.CheckpointGranularity() == 1 ||
            (config_.role != Role::kD && num_computed_tokens <= request.PrefillSize()),
        /*preserve_source_blocks=*/false);
    if (completed.completed_pages) {
        registerKvEventPublication(request, progress.prefix_hashes, *completed.completed_pages);
        coordinator_.CacheCompletedBlocks(request.BlockTablesRef(), completed, progress.access_epoch);
    }
    // The end state this finish publishes supersedes the checkpoint the turn
    // resumed from (MarkSupersededStateInput). A finish that publishes no end
    // state (finishCheckpointIsDead) publishes like a chunk end, whether or
    // not it completes a page, and marks the input all the same: the prompt
    // end, or the pre-tail checkpoint the next turn resumes at instead, was
    // published or found cached earlier. Marked at the finish only: a request
    // that fails earlier (a failed PD transfer) publishes and marks nothing,
    // and an end state that is the input itself (a D request that claimed its
    // prompt-end checkpoint and leaves before any decode) supersedes nothing.
    // Nor does a short prompt end left unpublished (shortPromptEndIsDropped):
    // such a prompt writes no in-forward checkpoint either, so the input can
    // still be the deepest checkpoint a next turn extending the prompt resumes
    // at, and it ages like any cache entry.
    if (progress.resumed_tail_hash && !shortPromptEndIsDropped(request) &&
        !(completed.completed_pages && completed.completed_pages->tail &&
          completed.completed_pages->tail->hash == ContentHash{*progress.resumed_tail_hash})) {
        coordinator_.MarkSupersededStateInput(ContentHash{*progress.resumed_tail_hash}, progress.access_epoch);
    }
    if (checkpoint_lineage_) {
        demoteSupersededCheckpoint(progress);
        recordCheckpointPublication(progress);
    }
    coordinator_.RefreshRetainedCheckpoint(progress.retained_input_checkpoint, progress.access_epoch);
    if (!config_.StreamsDeviceCacheToHost()) {
        return std::nullopt;
    }
    // Every published entry of the request streams: its pages, its tail and
    // the state checkpoint published with the frontier; a demoted entry
    // under its pages is not stored.
    std::vector<std::string> store_hashes = progress.prefix_hashes;
    if (completed.completed_pages && completed.completed_pages->tail) {
        store_hashes.push_back(completed.completed_pages->tail->hash.ToString());
    }
    coordinator_.QueueCachedBlocksForStore(store_hashes);
    // The request's pages are released right after this (ReleaseEvent); the
    // pinned ticket keeps them cached and unevictable until the copy ACKs.
    return tier_transfers_.StartPendingStores(StoreSourceGuard::kPinnedUntilAck, HostStoreCause::kFinish);
}

// With the in-forward checkpoint (SchedulerConfig::state_checkpoint_in_forward)
// a prompt end T inside the page its pre-tail State checkpoint C starts
// (PrefillStateCheckpointTokens: a prompt longer than the chunk budget whose
// partial page holds at least the replay tail) is dead: an identical prompt
// cannot resume at T -- its probe stops short of the final token -- and
// resumes at C; only a longer prompt repeating T's partial page could match
// T, and it resumes at C instead, recomputing less than a page. C lies below
// T, so a page-aligned T, a chain boundary a longer prompt resumes from, is
// never dead. With aligned State checkpoints C is the last segment entrance
// of any prompt longer than one segment instead, and those groups neither
// publish nor match a state at a dead T, which is never aligned. A request
// that resumed from a tail beyond C keeps T: it extends an earlier prompt end
// in place, and the next request likely extends T the same way. Returns C,
// or zero when T is not dead (always on the D role, which publishes as
// ever). It reads only the prompt and the resume point, so the admissions of
// a fused prompt, made before C exists, decide as its first decode will
// (firstDecodePublishesPromptEnd). C normally exists by the first decode:
// the forward spanning C writes it, and a forward that starts or ends at C
// finds or leaves the state there cached (a hit at C, a chunk end at C).
std::int32_t Scheduler::promptEndCheckpoint(const Request& request, std::int32_t resume_tokens) const {
    if (!config_.state_checkpoint_in_forward || config_.role == Role::kD) {
        return 0;
    }
    const std::int32_t page = coordinator_.PrefixGranularity();
    const std::int32_t checkpoint = PrefillStateCheckpointTokens(
        coordinator_, request.PrefillSize(), config_.max_scheduled_tokens, config_.prefix_replay_tokens);
    if (checkpoint <= 0 || checkpoint != request.PrefillSize() / page * page || resume_tokens > checkpoint) {
        return 0;
    }
    return checkpoint;
}

// Whether the checkpoint C that makes a prompt end T dead
// (promptEndCheckpoint) exists, which a publication in place at T -- a finish
// there -- needs to skip T: the request's final forward wrote C and it is
// still pending (the publication at T then publishes it), or every State
// group's state at C is cached, on either tier (the KV below C is the
// request's own). Once an admission lets the request's optional hold on C go
// and evicts it, such a finish publishes T. The caller passes the progress
// before the publication at T, the request's computed frontier.
bool Scheduler::promptEndIsCoveredByCheckpoint(const Request& request, const fsm::CacheProgress& progress) const {
    const std::int32_t checkpoint = promptEndCheckpoint(request, progress.resume_tokens);
    if (checkpoint == 0) {
        return false;
    }
    if (progress.state_checkpoint_tokens == checkpoint) {
        return true;
    }
    const auto checkpoint_pages = static_cast<std::size_t>(checkpoint / coordinator_.PrefixGranularity());
    return checkpoint_pages <= progress.prefix_hashes.size() &&
           coordinator_.HoldsStateCheckpoint(ContentHash{progress.prefix_hashes[checkpoint_pages - 1]});
}

// With drop_short_prompt_end_checkpoints (P role, in-forward checkpoint): a
// prompt too short for a pre-tail checkpoint (PrefillStateCheckpointTokens
// is 0) that ends inside a page ends in a tail a re-rendered next turn does
// not resume at, and an identical prompt's probe stops short of its last
// token; its state is keyed by that tail. The finish publishes the full pages
// only. A page-aligned end keeps its state: it is a chain boundary a longer
// prompt resumes from. A model without State groups keeps its tails. Unlike
// a covered prompt end (promptEndIsCoveredByCheckpoint) it needs no
// checkpoint to exist: it reads only the prompt, so it holds whatever the
// cache holds. Nothing before the finish -- the PD ACK -- depends on it: the
// P role reserves no copy-in block for its prompt end
// (firstDecodePublishesPromptEnd), its remote decode publishes like a chunk
// end, and it never retracts.
bool Scheduler::shortPromptEndIsDropped(const Request& request) const {
    const std::int32_t prompt = request.PrefillSize();
    if (!config_.drop_short_prompt_end_checkpoints || request.NumComputedTokens() != prompt ||
        prompt % coordinator_.PrefixGranularity() == 0 ||
        PrefillStateCheckpointTokens(coordinator_, prompt, config_.max_scheduled_tokens,
                                     config_.prefix_replay_tokens) != 0) {
        return false;
    }
    for (std::int32_t i = 0; i < coordinator_.NumGroups(); ++i) {
        if (coordinator_.GroupIsState(static_cast<std::size_t>(i))) {
            return true;
        }
    }
    return false;
}

// With the in-forward checkpoint, a finish at a covered prompt end
// publishes like a chunk end: its full pages and a pending in-forward
// checkpoint, but no tail and no state. That is the P role's finish (the
// transfer ACK; the P never computes past its prompt), and a fused
// request's that no decode ran past. Both publish in place, so they keep a
// dead prompt end whose checkpoint has left the cache
// (promptEndIsCoveredByCheckpoint); a fused first decode, which would move
// off the tail it publishes into copy-in blocks its admissions reserved,
// decides as they did instead (firstDecodePublishesPromptEnd). A fused finish
// F past the prompt end publishes like a chunk end too: a next turn that
// re-renders the generated reply diverges before F, and without its state the
// tail is no point a hybrid model can resume at. The prompt end was the
// first decode's to publish or skip (scheduleDecode). With
// drop_short_prompt_end_checkpoints the P role's finish also publishes a
// short prompt's end like a chunk end (shortPromptEndIsDropped). A model
// without State groups keeps its tails: there they are resume points of
// their own. The D role publishes its frontier as ever. (Validate requires
// prefix caching with the in-forward checkpoint on the other roles.)
bool Scheduler::finishCheckpointIsDead(const Request& request, const fsm::CacheProgress& progress) const {
    if (!config_.state_checkpoint_in_forward || config_.role == Role::kD) {
        return false;
    }
    const std::int32_t frontier = request.NumComputedTokens();
    if (frontier == request.PrefillSize()) {
        return shortPromptEndIsDropped(request) || promptEndIsCoveredByCheckpoint(request, progress);
    }
    if (frontier < request.PrefillSize()) {
        return false;
    }
    for (std::int32_t i = 0; i < coordinator_.NumGroups(); ++i) {
        if (coordinator_.GroupIsState(static_cast<std::size_t>(i))) {
            return true;
        }
    }
    return false;
}

void Scheduler::demoteSupersededCheckpoint(const fsm::CacheProgress& progress) {
    const std::int32_t page = coordinator_.PrefixGranularity();
    const std::int32_t checkpoint = progress.published_state_checkpoint_tokens;
    const std::int32_t resumed = progress.resume_tokens;
    const std::span<const std::string> chain(progress.prefix_hashes);
    if (checkpoint <= resumed || checkpoint / page > static_cast<std::int32_t>(chain.size()) || resumed <= 0 ||
        resumed % page != 0) {
        return;  // no checkpoint past a page boundary the request resumed from
    }
    // R's publication, when R is a recorded in-forward checkpoint, names
    // where R's request resumed. The chain hash at R fixes every page before
    // it, so that boundary lies on this request's chain too.
    const CheckpointLineage::Publication* resumed_publication =
        checkpoint_lineage_->FindPublication(ContentHash{chain[static_cast<std::size_t>(resumed / page) - 1]});
    if (resumed_publication == nullptr) {
        return;
    }
    const std::int32_t superseded = resumed_publication->resumed_tokens;
    if (superseded <= 0 || superseded >= resumed || superseded % page != 0) {
        return;
    }
    const ContentHash superseded_hash{chain[static_cast<std::size_t>(superseded / page) - 1]};
    if (checkpoint_lineage_->FindPublication(superseded_hash) == nullptr) {
        return;  // two turns back is no recorded in-forward checkpoint
    }
    // Shared -- another first admission resumed there too, or an acquisition
    // or publication there is newer than the one R's request made when it
    // resumed there: a fork, or a system prompt every new conversation
    // resumes at. It stays.
    if (checkpoint_lineage_->Resumes(superseded_hash) != 1 ||
        coordinator_.LatestStateAccessEpoch(superseded_hash) > resumed_publication->access_epoch) {
        return;
    }
    // R and C stay resumable with the window blocks they resume with: a
    // sliding window wider than a turn reaches back behind the boundary.
    const std::array<std::int32_t, 2> kept_boundaries{resumed, checkpoint};
    const std::vector<CacheKey> keys =
        coordinator_.BoundaryCheckpointKeys(chain.first(static_cast<std::size_t>(superseded / page)), kept_boundaries);
    std::int32_t demoted = coordinator_.DemoteEntries(keys);
    // Under the write-back policy an eviction may still be carrying one of
    // them to the Host: it lands demoted as well.
    demoted += tier_transfers_.DemoteInFlightEvictions(keys);
    demoted_checkpoint_entries_ += static_cast<std::uint64_t>(demoted);
}

void Scheduler::recordCheckpointPublication(const fsm::CacheProgress& progress) {
    const std::int32_t page = coordinator_.PrefixGranularity();
    const std::int32_t checkpoint = progress.published_state_checkpoint_tokens;
    const std::int32_t resumed = progress.resume_tokens;
    if (checkpoint <= resumed || checkpoint / page > static_cast<std::int32_t>(progress.prefix_hashes.size())) {
        return;  // no in-forward checkpoint past the point it resumed from
    }
    checkpoint_lineage_->RecordPublication(
        ContentHash{progress.prefix_hashes[static_cast<std::size_t>(checkpoint / page) - 1]},
        CheckpointLineage::Publication{.resumed_tokens = resumed % page == 0 ? resumed : 0,
                                       .access_epoch = progress.access_epoch});
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
        abortRequest(*request);
    }
}

void Scheduler::abortRequest(Request& request) {
    if (pdTransferInFlight(request)) {
        // The transport stops the transfer; the pages wait for it.
        request.Apply(fsm::StopPdTransferEvent{});
    } else {
        request.Apply(fsm::AbortEvent{&coordinator_});
    }
}

void Scheduler::handleEvent(const cache::WriteBackDone& event) {
    // Completing the op lands it for every image waiting on it: their waits
    // expire with its record.
    const std::vector<HostPublication> published = tier_transfers_.CompleteWriteBack(event.op_id);
    // A retraction image's L2 leg: every suspended request that waited for
    // this store (its own, or an earlier one carrying one of its keys)
    // follows the Host entries as published, which may differ from the
    // ticket's blocks it pinned when the index redirected a key to an entry
    // an L3 prefetch landed first.
    for (const std::unique_ptr<Request>& request : requests_) {
        fsm::Retracted* retracted = request->GetIf<fsm::Retracted>();
        if (retracted != nullptr && retracted->WaitsForStore(event.op_id)) {
            retracted->image.FollowPublished(published);
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
    // The image's tail leg landed: its wait expires with the op's record.
    tier_transfers_.CompleteSnapshotStore(event.op_id);
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
        // again first -- the prefix chain and any tail a State checkpoint of
        // the image sits at -- as a first chunk does for the load-backs it
        // issues.
        registerKvEventRestore(*request, restoring->resources.cache_progress.prefix_hashes, restoring->kv_event_tails);
    }
    ReqPoolIndex request_pool_index = tier_transfers_.CompleteSnapshotRestore(event.op_id, /*publish=*/resumes);
    if (resumes) {
        request->Apply(fsm::RestoreDoneEvent{std::move(request_pool_index)});
    }
    // Otherwise the row and the blob slot return with the ticket: the request
    // that would have used them is gone.
}

}  // namespace tokenspeed
