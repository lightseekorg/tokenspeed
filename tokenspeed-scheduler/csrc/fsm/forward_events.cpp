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

#include "fsm/forward_events.h"

#include <utility>

#include "scheduler/operations/cache.h"
#include "core/token_container.h"
#include "fsm/pd_states.h"

namespace tokenspeed::fsm {

std::variant<PrefillDone, PrefillAwaitingResult, Prefilling, RemotePrefilling>
SchedulePrefillFirstChunkEvent::operator()(Submitted&& state) {
    TokenContainer* token_container = state.TokenContainerPtr();
    const std::int32_t prefix_granularity = state.PrefixGranularity();
    _assert(coordinator_ != nullptr, "SchedulePrefillFirstChunkEvent requires a cache coordinator");
    _assert(block_tables_.size() == static_cast<std::size_t>(coordinator_->NumGroups()),
            "SchedulePrefillFirstChunkEvent requires one admitted table per cache group");

    // The request's first page-holding state: nothing is out against these
    // pages yet.
    ForwardResources resources{
        .token_container = token_container,
        .prefix_granularity = prefix_granularity,
        .req_pool_index = req_pool_allocator_->Allocate(),
        .block_tables = std::move(block_tables_),
        .cache_progress = std::move(cache_progress_),
        .results_in_flight = 0,
    };
    // A local hit re-feeds the replay window before it (bounded replay). A
    // remote prefill computes nothing here: the peer's replayable pages land
    // as its retained tail, like any sliding group's.
    const TokenContainer::Window window{
        .begin = hit_tokens_,
        .size = tokens_this_round_,
        .replay = source_ == PrefillSource::kLocal ? coordinator_->ReplayTokens(hit_tokens_) : 0,
    };
    if (source_ == PrefillSource::kRemote) {
        // The peer prefills the whole prompt; this engine only holds the
        // destination pages until RemotePrefillDone.
        return RemotePrefilling{std::move(resources), window, reserve_num_tokens_in_next_schedule_event_};
    }
    if (window.begin + window.size == token_container->PrefillSize()) {
        if (awaits_result_) {
            return PrefillAwaitingResult{std::move(resources), window, reserve_num_tokens_in_next_schedule_event_};
        }
        return PrefillDone{std::move(resources), window, reserve_num_tokens_in_next_schedule_event_};
    }
    return Prefilling{std::move(resources), window, reserve_num_tokens_in_next_schedule_event_};
}

std::variant<PrefillDone, PrefillAwaitingResult, Prefilling> SchedulePrefillEvent::operator()(Prefilling&& state) {
    const TokenContainer::Window window{
        .begin = state.window.begin + state.window.size,
        .size = tokens_this_round_,
    };
    if (window.begin + window.size == state.resources.token_container->PrefillSize()) {
        if (awaits_result_) {
            return PrefillAwaitingResult{std::move(state.resources), window,
                                         reserve_num_tokens_in_next_schedule_event_};
        }
        return PrefillDone{std::move(state.resources), window, reserve_num_tokens_in_next_schedule_event_};
    }
    return Prefilling{std::move(state.resources), window, reserve_num_tokens_in_next_schedule_event_};
}

template <typename State>
Decoding ScheduleDecodeEvent::decode(State&& state) {
    return Decoding{std::move(state.resources), decode_input_tokens_};
}

Decoding ScheduleDecodeEvent::operator()(PrefillDone&& state) {
    return decode(std::move(state));
}

Decoding ScheduleDecodeEvent::operator()(PrefillAwaitingResult&& state) {
    return decode(std::move(state));
}

Decoding ScheduleDecodeEvent::operator()(Decoding&& state) {
    return decode(std::move(state));
}

template <typename State>
Finished FinishEvent::finish(State&& state) {
    _assert(coordinator_ != nullptr, "FinishEvent requires a cache coordinator");
    FreeRequest(*coordinator_, state.resources.block_tables);
    return Finished{};
}

Finished FinishEvent::operator()(PrefillDone&& state) {
    return finish(std::move(state));
}

Finished FinishEvent::operator()(PrefillAwaitingResult&& state) {
    return finish(std::move(state));
}

Finished FinishEvent::operator()(Decoding&& state) {
    return finish(std::move(state));
}

Finished FinishEvent::operator()(Retracted&&) {
    return Finished{};  // the image dies with the state
}

Finished FinishEvent::operator()(Restoring&& state) {
    return finish(std::move(state));
}

Finished AbortEvent::operator()(Bootstrapping&&) {
    return Finished{};
}

Finished AbortEvent::operator()(Submitted&&) {
    return Finished{};
}

template <typename State>
Finished AbortEvent::abortForward(State&& state) {
    _assert(coordinator_ != nullptr, "AbortEvent requires a cache coordinator");
    FreeRequest(*coordinator_, state.resources.block_tables);
    return Finished{};
}

Finished AbortEvent::operator()(Prefilling&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(RemotePrefilling&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(PrefillDone&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(PrefillAwaitingResult&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(Decoding&& state) {
    return abortForward(std::move(state));
}

Finished AbortEvent::operator()(Retracted&&) {
    return Finished{};  // the image dies with the state
}

Finished AbortEvent::operator()(Restoring&& state) {
    // The restore's copies may still be in flight: the transfer manager keeps
    // both ends of every pair pinned until the ACK, so freeing the tables
    // here only drops the request's own references.
    return abortForward(std::move(state));
}

template <typename State>
Retracted SnapshotRetractEvent::retract(State&& state, ResumeShape shape) {
    _assert(coordinator_ != nullptr, "SnapshotRetractEvent requires a cache coordinator");
    _assert(state.resources.results_in_flight == 0, "a retraction victim must be quiescent");
    ForwardResources& resources = state.resources;
    // The pages are released -- and may be granted away in this very round --
    // because the store reads them ahead of any reuse (stream-ordered).
    FreeRequest(*coordinator_, resources.block_tables);
    return Retracted{
        .token_container = resources.token_container,
        .prefix_granularity = resources.prefix_granularity,
        .cache_progress = std::move(resources.cache_progress),
        .image = std::move(image_),
        .blob_slot = std::move(blob_slot_),
        .shape = std::move(shape),
        .retraction_epoch = epoch_,
        .resumes_generation = resumes_generation_,
        .pending_store_ops = std::move(pending_store_ops_),
    };
}

Retracted SnapshotRetractEvent::operator()(Prefilling&& state) {
    const ResumePrefilling shape{
        .window = state.window,
        .reserve_num_tokens_in_next_schedule_event = state.ReserveNumTokensInNextScheduleEvent()};
    return retract(std::move(state), shape);
}

Retracted SnapshotRetractEvent::operator()(PrefillDone&& state) {
    const ResumePrefillDone shape{
        .window = state.window,
        .reserve_num_tokens_in_next_schedule_event = state.ReserveNumTokensInNextScheduleEvent()};
    return retract(std::move(state), shape);
}

Retracted SnapshotRetractEvent::operator()(Decoding&& state) {
    const ResumeDecoding shape{.reserve_num_tokens_in_next_schedule_event =
                                   state.ReserveNumTokensInNextScheduleEvent()};
    return retract(std::move(state), shape);
}

template <typename State>
Submitted RecomputeRetractEvent::recompute(State&& state) {
    _assert(coordinator_ != nullptr, "RecomputeRetractEvent requires a cache coordinator");
    ForwardResources& resources = state.resources;
    // Prompt + generated become one fresh prefill: the skipped forward wrote
    // nothing, so there is no checkpoint to continue from.
    resources.token_container->RebasePrefill();
    FreeRequest(*coordinator_, resources.block_tables);
    return Submitted{resources.token_container, resources.prefix_granularity};
}

Submitted RecomputeRetractEvent::operator()(Prefilling&& state) {
    return recompute(std::move(state));
}

Submitted RecomputeRetractEvent::operator()(PrefillDone&& state) {
    return recompute(std::move(state));
}

Submitted RecomputeRetractEvent::operator()(PrefillAwaitingResult&& state) {
    return recompute(std::move(state));
}

Submitted RecomputeRetractEvent::operator()(RemotePrefilling&& state) {
    return recompute(std::move(state));
}

Submitted RecomputeRetractEvent::operator()(Decoding&& state) {
    return recompute(std::move(state));
}

Restoring ScheduleRestoreEvent::operator()(Retracted&& state) {
    _assert(coordinator_ != nullptr, "ScheduleRestoreEvent requires a cache coordinator");
    _assert(state.ImageLanded(), "a restore is issued only after the image landed");
    _assert(req_pool_index_.valid(), "ScheduleRestoreEvent requires a request pool slot");
    _assert(block_tables_.size() == static_cast<std::size_t>(coordinator_->NumGroups()),
            "ScheduleRestoreEvent requires one rebuilt table per cache group");
    return Restoring{
        .resources =
            ForwardResources{
                .token_container = state.token_container,
                .prefix_granularity = state.prefix_granularity,
                .req_pool_index = std::move(req_pool_index_),
                .block_tables = std::move(block_tables_),
                .cache_progress = std::move(state.cache_progress),
                .results_in_flight = 0,
            },
        .image = std::move(state.image),
        .blob_slot = std::move(state.blob_slot),
        .shape = std::move(state.shape),
        .restore_op = restore_op_,
    };
}

std::variant<Prefilling, PrefillDone, Decoding> RestoreDoneEvent::operator()(Restoring&& state) {
    return std::visit(Overloaded{
                          [&](const ResumePrefilling& shape) -> std::variant<Prefilling, PrefillDone, Decoding> {
                              return Prefilling{std::move(state.resources), shape.window,
                                                shape.reserve_num_tokens_in_next_schedule_event};
                          },
                          [&](const ResumePrefillDone& shape) -> std::variant<Prefilling, PrefillDone, Decoding> {
                              return PrefillDone{std::move(state.resources), shape.window,
                                                 shape.reserve_num_tokens_in_next_schedule_event};
                          },
                          [&](const ResumeDecoding& shape) -> std::variant<Prefilling, PrefillDone, Decoding> {
                              return Decoding{std::move(state.resources),
                                              shape.reserve_num_tokens_in_next_schedule_event};
                          },
                      },
                      state.shape);
}

}  // namespace tokenspeed::fsm
