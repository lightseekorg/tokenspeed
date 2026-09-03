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
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/core/cache_types.h"
#include "core/token_container.h"
#include "resource/allocator/req_pool_allocator.h"
#include "scheduler/request_spec.h"

namespace tokenspeed::fsm {

enum class PrefillSource { kLocal, kRemote };

struct CacheProgress {
    // One source of truth for both the next hash-chain seed and the cumulative
    // history needed to publish a resumable boundary across chunk edges.
    std::vector<std::string> prefix_hashes;
    std::uint64_t access_epoch{0};
    // Pending closed-prefix boundary; zero once published or when absent.
    std::int32_t promotion_boundary_tokens{0};
    // Whether cache storage for the final state-checkpoint tail was reserved.
    bool state_checkpoint_tail_reserved{false};
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

struct Submitted {
    Submitted(TokenContainer* token_container, std::int32_t prefix_granularity)
        : token_container_{token_container}, prefix_granularity_{prefix_granularity} {}

    TokenContainer* TokenContainerPtr() const { return token_container_; }
    std::int32_t PrefixGranularity() const { return prefix_granularity_; }

private:
    TokenContainer* token_container_{};
    std::int32_t prefix_granularity_{};
};

struct ForwardState {
    ForwardState(TokenContainer* token_container, std::int32_t prefix_granularity,
                 std::unique_ptr<ReqPoolIndex> req_pool_index, std::vector<BlockTable> block_tables,
                 CacheProgress cache_progress)
        : token_container_{token_container},
          prefix_granularity_{prefix_granularity},
          req_pool_index_{std::move(req_pool_index)},
          block_tables_{std::move(block_tables)},
          cache_progress_{std::move(cache_progress)} {}

    ForwardState(const ForwardState&) = delete;
    ForwardState& operator=(const ForwardState&) = delete;
    ForwardState(ForwardState&&) noexcept = default;
    ForwardState& operator=(ForwardState&&) noexcept = default;

    TokenContainer* TokenContainerPtr() const { return token_container_; }
    std::int32_t PrefixGranularity() const { return prefix_granularity_; }

    std::unique_ptr<ReqPoolIndex> TakeRequestPoolIndex() && { return std::move(req_pool_index_); }
    std::int32_t RequestPoolIndex() const { return req_pool_index_ ? req_pool_index_->slot_ : -1; }

    std::vector<BlockTable>& BlockTables() { return block_tables_; }
    const std::vector<BlockTable>& BlockTables() const { return block_tables_; }
    std::vector<BlockTable> TakeBlockTables() && { return std::move(block_tables_); }

    CacheProgress TakeCacheProgress() && { return std::move(cache_progress_); }
    const CacheProgress& CacheProgressRef() const { return cache_progress_; }

    // Forwards scheduled for this request whose results have not come back.
    // More than one is normal under the overlap schedule, which plans the
    // next step before committing the previous one.
    //
    // It lives on the base, not on the states that happen to consume a
    // result: a forward is out against the PAGES, and every forward state
    // owns pages. A prefill chunk produces no ExtendResult, but its result
    // still writes KV into this request's tables -- retract it mid-flight
    // and the write lands on pages someone else now owns.
    std::int32_t ResultsInFlight() const { return results_in_flight_; }
    void TrackScheduledForward() { ++results_in_flight_; }
    void ResultLanded() { results_in_flight_ = std::max(0, results_in_flight_ - 1); }
    // Carried across a state transition: a transition relabels the request,
    // and the forwards already out do not care what it is called.
    void CarryResultsInFlight(std::int32_t count) { results_in_flight_ = count; }

protected:
    TokenContainer* token_container_{};
    std::int32_t prefix_granularity_{};

private:
    std::unique_ptr<ReqPoolIndex> req_pool_index_;
    std::vector<BlockTable> block_tables_;
    CacheProgress cache_progress_;
    std::int32_t results_in_flight_{0};
};

struct Prefilling : public ForwardState {
    Prefilling(TokenContainer* token_container, std::int32_t prefix_granularity,
               std::unique_ptr<ReqPoolIndex> req_pool_index, TokenContainer::Window window,
               std::int32_t reserve_num_tokens_in_next_schedule_event, std::vector<BlockTable> block_tables,
               CacheProgress cache_progress)
        : ForwardState(token_container, prefix_granularity, std::move(req_pool_index), std::move(block_tables),
                       std::move(cache_progress)),
          window{window},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    std::span<const std::int32_t> PrefillInputIds() const { return token_container_->TokenSlice(window); }
    std::vector<std::int32_t> ShiftedInputIds() const { return ComputeShiftedInputIds(token_container_, window); }
    PrefillInfo CurrentPrefillInfo() const {
        return PrefillInfo{
            .input_ids = PrefillInputIds(),
            .shifted_input_ids = ShiftedInputIds(),
            .already_scheduled_len = window.begin,
            .extend_len = window.size,
        };
    }

    std::int32_t ReserveNumTokensInNextScheduleEvent() const { return reserve_num_tokens_in_next_schedule_event_; }
    // The final mamba state checkpoint's pages are already reserved, so this
    // request's remaining prompt is capacity-safe.
    bool TailCheckpointReserved() const { return CacheProgressRef().state_checkpoint_tail_reserved; }

    TokenContainer::Window window{};

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
};

struct PrefillDone : public ForwardState {
    PrefillDone(TokenContainer* token_container, std::int32_t prefix_granularity,
                std::unique_ptr<ReqPoolIndex> req_pool_index, TokenContainer::Window window,
                std::int32_t reserve_num_tokens_in_next_schedule_event, std::vector<BlockTable> block_tables,
                CacheProgress cache_progress)
        : ForwardState(token_container, prefix_granularity, std::move(req_pool_index), std::move(block_tables),
                       std::move(cache_progress)),
          window{window},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    std::int32_t ReserveNumTokensInNextScheduleEvent() const { return reserve_num_tokens_in_next_schedule_event_; }

    std::span<const std::int32_t> PrefillInputIds() const { return token_container_->TokenSlice(window); }
    std::vector<std::int32_t> ShiftedInputIds() const { return ComputeShiftedInputIds(token_container_, window); }
    PrefillInfo CurrentPrefillInfo() const {
        return PrefillInfo{
            .input_ids = PrefillInputIds(),
            .shifted_input_ids = ShiftedInputIds(),
            .already_scheduled_len = window.begin,
            .extend_len = window.size,
        };
    }
    void ExtendResultTokens(const std::vector<std::int32_t>& result_tokens) { token_container_->Extend(result_tokens); }

    TokenContainer::Window window{};

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
};

struct Decoding : public ForwardState {
    Decoding(TokenContainer* token_container, std::int32_t prefix_granularity,
             std::unique_ptr<ReqPoolIndex> req_pool_index, std::int32_t reserve_num_tokens_in_next_schedule_event,
             std::vector<BlockTable> block_tables, CacheProgress cache_progress)
        : ForwardState(token_container, prefix_granularity, std::move(req_pool_index), std::move(block_tables),
                       std::move(cache_progress)),
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    std::int32_t ReserveNumTokensInNextScheduleEvent() const {
        _assert(reserve_num_tokens_in_next_schedule_event_ >= 0);
        return reserve_num_tokens_in_next_schedule_event_;
    }
    void SetReserveNumTokensInNextScheduleEvent(std::int32_t value) {
        reserve_num_tokens_in_next_schedule_event_ = value;
    }
    void ExtendResultTokens(const std::vector<std::int32_t>& result_tokens) { token_container_->Extend(result_tokens); }

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{-1};
};

struct Retracted {
    TokenContainer* token_container{};
    std::int32_t prefix_granularity{};
    // Monotonic stamp from the retraction that produced this state. The plan
    // builder derives the readmission order off the states themselves -- no
    // separate queue to keep in step with the FSM.
    std::int64_t retraction_epoch{0};
    // False when the retraction had nowhere to store the KV (no host cache):
    // there is no snapshot to recover, so the request re-prefills like any
    // newcomer and does not queue behind other readmissions.
    bool has_recoverable_snapshot{true};
    // A victim with generated output a client is reading resumes ahead of
    // one that had produced nothing, whatever their retraction epochs say.
    bool resumes_generation{false};

    TokenContainer* TokenContainerPtr() const { return token_container; }
    std::int32_t PrefixGranularity() const { return prefix_granularity; }
    std::int64_t RetractionEpoch() const { return retraction_epoch; }
    bool HasRecoverableSnapshot() const { return has_recoverable_snapshot; }
    bool ResumesGeneration() const { return resumes_generation; }
};

struct Finished {};

}  // namespace tokenspeed::fsm
