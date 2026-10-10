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

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "cache/core/cache_types.h"
#include "core/token_container.h"
#include "resource/allocator/req_pool_allocator.h"
#include "scheduler/request_spec.h"
#include "utils.h"

namespace tokenspeed::fsm {

enum class PrefillSource { kLocal, kRemote };

struct CacheProgress {
    // One source of truth for both the next hash-chain seed and the cumulative
    // history needed to publish a resumable boundary across chunk edges.
    std::vector<std::string> prefix_hashes;
    std::uint64_t access_epoch{0};
    // Pending closed-prefix boundary; zero once published or when absent.
    std::int32_t promotion_boundary_tokens{0};
    // Aligned prefill checkpoints written but not yet hashed, in token order.
    // Scheduled prefill windows record their materialized checkpoint;
    // decode results keep working state only and add no reusable boundary.
    std::vector<std::int32_t> materialized_state_boundaries;

    void RecordMaterializedStateBoundary(std::int32_t boundary, std::int32_t prefix_granularity) {
        if (boundary <= 0 || boundary % prefix_granularity != 0 ||
            boundary / prefix_granularity <= static_cast<std::int32_t>(prefix_hashes.size())) {
            return;
        }
        if (materialized_state_boundaries.empty() || materialized_state_boundaries.back() < boundary) {
            materialized_state_boundaries.push_back(boundary);
        }
    }

    // Only after the admission that hashed them succeeded: a failed attempt
    // retries their publication with the same hashes.
    void DiscardHashedStateBoundaries(std::int32_t prefix_granularity) {
        std::erase_if(materialized_state_boundaries, [&](std::int32_t boundary) {
            return boundary / prefix_granularity <= static_cast<std::int32_t>(prefix_hashes.size());
        });
    }
};

inline std::vector<std::int32_t> ComputeShiftedInputIds(const TokenContainer* token_container,
                                                        TokenContainer::Window window) {
    const std::int32_t shifted_start = window.begin + 1;
    const std::int32_t shifted_end = std::min(token_container->PrefillSize(), shifted_start + window.size);
    const std::int32_t shifted_size = std::max<std::int32_t>(0, shifted_end - shifted_start);

    std::vector<std::int32_t> shifted;
    shifted.reserve(static_cast<std::size_t>(window.size));
    if (shifted_size > 0) {
        auto slice = token_container->TokenSlice(TokenContainer::Window{shifted_start, shifted_size});
        shifted.insert(shifted.end(), slice.begin(), slice.end());
    }
    shifted.resize(static_cast<std::size_t>(window.size), -1);
    return shifted;
}

// Waiting for admission. A request whose L3 prefetch landed comes back here
// holding the Host entries it fetched (prefetched_host_entries), pinned so
// they are still there when its admission claims them as an ordinary Host hit;
// the pins drop with the state at admission, after the admission acquired its
// own references.
struct Submitted {
    Submitted(TokenContainer* token_container, std::int32_t prefix_granularity)
        : token_container_{token_container}, prefix_granularity_{prefix_granularity} {}
    Submitted(TokenContainer* token_container, std::int32_t prefix_granularity,
              std::vector<CacheBlockRef> prefetched_host_entries)
        : token_container_{token_container},
          prefix_granularity_{prefix_granularity},
          prefetched_host_entries_{std::move(prefetched_host_entries)} {}

    TokenContainer* TokenContainerPtr() const { return token_container_; }
    std::int32_t PrefixGranularity() const { return prefix_granularity_; }
    std::size_t NumPrefetchedHostEntries() const { return prefetched_host_entries_.size(); }

private:
    TokenContainer* token_container_{};
    std::int32_t prefix_granularity_{};
    std::vector<CacheBlockRef> prefetched_host_entries_;
};

// Before admission, with an L3 prefetch in flight: the request pins the Host
// blocks the fetch fills (nothing on the Device, no request-pool row), is
// skipped by admission -- later Submitted requests may be admitted past it,
// it holds no head of line -- and is never a victim. The op's ACK
// (PrefetchDone) returns it to Submitted holding the entries that landed; an
// abort drops its pins, and the op's own pins keep the blocks until the ACK.
struct Prefetching {
    Prefetching(TokenContainer* token_container, std::int32_t prefix_granularity,
                std::vector<CacheBlockRef> host_blocks, std::uint32_t prefetch_op)
        : token_container{token_container},
          prefix_granularity{prefix_granularity},
          host_blocks{std::move(host_blocks)},
          prefetch_op{prefetch_op} {}

    TokenContainer* token_container{};
    std::int32_t prefix_granularity{};
    std::vector<CacheBlockRef> host_blocks;
    std::uint32_t prefetch_op{};
};

// Everything a page-holding state owns on the request's behalf: the KV
// pages, the request-pool slot, the prefix-cache progress and the count of
// forwards still out against those pages. Move-only, and moved as ONE
// bundle: a transition hands it whole to exactly one successor state, or
// returns the pages to the coordinator and lets the empty bundle die. There
// is no third path, and no field that a transition could forget to carry.
struct ForwardResources {
    TokenContainer* token_container{};
    std::int32_t prefix_granularity{};
    ReqPoolIndex req_pool_index;
    std::vector<BlockTable> block_tables;
    CacheProgress cache_progress;
    // Forwards scheduled for this request whose results have not come back.
    // More than one is normal under the overlap schedule, which plans the
    // next step before committing the previous one.
    //
    // It lives here, not on the states that happen to consume a result: a
    // forward is out against the PAGES, and every page-holding state has
    // this bundle. A prefill chunk produces no ExtendResult, but its result
    // still writes KV into this request's tables -- retract it mid-flight
    // and the write lands on pages someone else now owns.
    std::int32_t results_in_flight{0};

    std::int32_t RequestPoolIndex() const { return req_pool_index.valid() ? req_pool_index.slot_ : -1; }
    void TrackScheduledForward() { ++results_in_flight; }
    void ResultLanded() {
        FatalCheck(results_in_flight > 0, "a forward result landed for a request with no forward in flight");
        --results_in_flight;
    }
    void ExtendTokens(const std::vector<std::int32_t>& tokens) { token_container->Extend(tokens); }
};

template <typename State>
concept HoldsForwardResources = requires(State& state) {
    { state.resources } -> std::same_as<ForwardResources&>;
};

// A prefill window's model inputs; shared by every state that still
// describes its prompt chunk. The model input starts `window.replay` tokens
// before the window (bounded replay); progress still ends at begin + size.
inline PrefillInfo MakePrefillInfo(const ForwardResources& resources, TokenContainer::Window window) {
    _assert(window.replay >= 0 && window.replay <= window.begin, "replay must re-feed computed prompt tokens");
    const TokenContainer::Window input{.begin = window.begin - window.replay, .size = window.size + window.replay};
    return PrefillInfo{
        .input_ids = resources.token_container->TokenSlice(input),
        .shifted_input_ids = ComputeShiftedInputIds(resources.token_container, input),
        .already_scheduled_len = input.begin,
        .extend_len = input.size,
        .replay_len = window.replay,
    };
}

struct Prefilling {
    Prefilling(ForwardResources resources, TokenContainer::Window window,
               std::int32_t reserve_num_tokens_in_next_schedule_event)
        : resources{std::move(resources)},
          window{window},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    PrefillInfo CurrentPrefillInfo() const { return MakePrefillInfo(resources, window); }
    std::int32_t ReserveNumTokensInNextScheduleEvent() const { return reserve_num_tokens_in_next_schedule_event_; }

    ForwardResources resources;
    TokenContainer::Window window{};

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
};

// Set by RestoreDoneEvent on the state a restore resumes into: the device
// holds no in-flight capture for this request's next input (its forward
// history belongs to the slot it left), so the first decode after a restore
// carries the token explicitly, as a D-role bootstrap decode does. The
// ScheduleDecodeEvent that consumes it builds a fresh state without it.
struct RestoreMarker {
    bool ResumedByRestore() const { return resumed_by_restore_; }
    void MarkResumedByRestore() { resumed_by_restore_ = true; }

private:
    bool resumed_by_restore_{false};
};

struct PrefillDone : RestoreMarker {
    PrefillDone(ForwardResources resources, TokenContainer::Window window,
                std::int32_t reserve_num_tokens_in_next_schedule_event)
        : resources{std::move(resources)},
          window{window},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    PrefillInfo CurrentPrefillInfo() const { return MakePrefillInfo(resources, window); }
    std::int32_t ReserveNumTokensInNextScheduleEvent() const { return reserve_num_tokens_in_next_schedule_event_; }
    void ExtendResultTokens(const std::vector<std::int32_t>& result_tokens) { resources.ExtendTokens(result_tokens); }

    ForwardResources resources;
    TokenContainer::Window window{};

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
};

struct Decoding : RestoreMarker {
    Decoding(ForwardResources resources, std::int32_t reserve_num_tokens_in_next_schedule_event)
        : resources{std::move(resources)},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    std::int32_t ReserveNumTokensInNextScheduleEvent() const {
        _assert(reserve_num_tokens_in_next_schedule_event_ >= 0);
        return reserve_num_tokens_in_next_schedule_event_;
    }
    void SetReserveNumTokensInNextScheduleEvent(std::int32_t value) {
        reserve_num_tokens_in_next_schedule_event_ = value;
    }
    void ExtendResultTokens(const std::vector<std::int32_t>& result_tokens) { resources.ExtendTokens(result_tokens); }

    ForwardResources resources;

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{-1};
};

// Where a retracted request resumes once its image is back on Device: the
// state it left, with what that state carried beyond its ForwardResources.
struct ResumePrefilling {
    TokenContainer::Window window{};
    std::int32_t reserve_num_tokens_in_next_schedule_event{};
};
struct ResumePrefillDone {
    TokenContainer::Window window{};
    std::int32_t reserve_num_tokens_in_next_schedule_event{};
};
struct ResumeDecoding {
    std::int32_t reserve_num_tokens_in_next_schedule_event{};
};
using ResumeShape = std::variant<ResumePrefilling, ResumePrefillDone, ResumeDecoding>;

// Suspended with its image: the request holds no Device pages and no request
// pool slot, but its KV lives on -- the published pages as pinned Host L2
// entries, everything else in the snapshot pool -- together with its cache
// progress and the exact point it stopped at, so the restore continues it
// where it was (a decoding victim decodes, a mid-prefill victim runs its next
// chunk). Nothing is recomputed and nothing is rebased.
struct Retracted {
    TokenContainer* token_container{};
    std::int32_t prefix_granularity{};
    CacheProgress cache_progress;
    RetractionImage image;
    // The slot-state blob's slot in the runtime's arena (RAII).
    SnapshotSlotIndex blob_slot;
    ResumeShape shape;
    // Monotonic stamp from the retraction that produced this state. The plan
    // builder derives the readmission order off the states themselves -- no
    // separate queue to keep in step with the FSM.
    std::int64_t retraction_epoch{0};
    // A victim with generated output a client is reading resumes ahead of
    // one that had produced nothing, whatever their retraction epochs say.
    bool resumes_generation{false};
    // The store ops the image waits for: the L2 leg's write-back(s) -- its
    // own and any earlier in-flight store carrying one of its keys -- and
    // the tail leg's snapshot store. A restore is issued only once every one
    // has been acknowledged.
    std::vector<std::uint32_t> pending_store_ops;

    TokenContainer* TokenContainerPtr() const { return token_container; }
    std::int32_t PrefixGranularity() const { return prefix_granularity; }
    std::int64_t RetractionEpoch() const { return retraction_epoch; }
    bool ResumesGeneration() const { return resumes_generation; }
    bool ImageLanded() const { return pending_store_ops.empty(); }
    bool WaitsForStore(std::uint32_t op_id) const {
        return std::find(pending_store_ops.begin(), pending_store_ops.end(), op_id) != pending_store_ops.end();
    }
    // The store landed; its publications say which Host block is canonical
    // for each key, and the image's L2 slots follow (an L3 prefetch of the
    // same key may have published first, leaving the ticket's block unindexed).
    void NoteStoreLanded(std::uint32_t op_id, std::span<const HostPublication> published) {
        std::erase(pending_store_ops, op_id);
        image.FollowPublished(published);
    }
};

// The image is being copied back into freshly allocated Device pages: the
// request holds pages and a request pool slot again (so the pages count as
// active and are never granted away), but nothing is schedulable until the
// restore's ACK. The image stays alive here because the copy reads it.
struct Restoring {
    ForwardResources resources;
    RetractionImage image;
    SnapshotSlotIndex blob_slot;
    ResumeShape shape;
    std::uint32_t restore_op{0};
};

struct Finished {};

}  // namespace tokenspeed::fsm
