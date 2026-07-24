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

#include "scheduler/flat_kv_completion_ledger.h"

#include <atomic>
#include <exception>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tokenspeed {
namespace {

std::atomic<std::uint64_t> g_next_flat_table_generation{1};

}  // namespace

FlatKVCompletionState::FlatKVCompletionState() : table_generation_{AllocateGeneration()} {}

std::uint64_t FlatKVCompletionState::AllocateGeneration() {
    const std::uint64_t generation = g_next_flat_table_generation.fetch_add(1, std::memory_order_relaxed);
    if (generation == 0 || generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("flat KV table generation space exhausted");
    }
    return generation;
}

void FlatKVCompletionState::PushBack(DispatchRecord record) noexcept {
    static_assert(std::is_nothrow_move_constructible_v<DispatchRecord>);
    if (size_ == kMaxDispatches) {
        std::terminate();
    }
    const std::size_t tail = (head_ + size_) % kMaxDispatches;
    if (dispatches_[tail].has_value()) {
        std::terminate();
    }
    dispatches_[tail].emplace(std::move(record));
    ++size_;
}

void FlatKVCompletionState::PopFront() noexcept {
    if (size_ == 0 || !dispatches_[head_].has_value()) {
        std::terminate();
    }
    dispatches_[head_].reset();
    head_ = (head_ + 1) % kMaxDispatches;
    --size_;
}

FlatKVCompletionRequestSnapshot FlatKVCompletionState::Snapshot() const noexcept {
    if (size_ == 0) {
        return {};
    }
    bool has_canceled = false;
    for (std::size_t i = 0; i < size_; ++i) {
        has_canceled = has_canceled || At(i).canceled;
    }
    return FlatKVCompletionRequestSnapshot{
        .outstanding_count = size_,
        .has_canceled_outstanding = has_canceled,
        .last_dispatch_raw_end = Back().dispatch_raw_end,
    };
}

std::size_t FlatKVCompletionState::CancelOutstanding() noexcept {
    std::size_t canceled = 0;
    for (std::size_t i = 0; i < size_; ++i) {
        DispatchRecord& record = At(i);
        if (!record.canceled) {
            record.canceled = true;
            ++canceled;
        }
    }
    return canceled;
}

FlatKVCompletionLedger::FlatKVCompletionLedger(std::size_t max_outstanding_per_request,
                                               std::span<const KvCacheGroupSchema> schema)
    : max_outstanding_per_request_{max_outstanding_per_request},
      schema_{schema},
      ready_raw_ends_scratch_(schema_.size()) {
    if (max_outstanding_per_request_ == 0 || max_outstanding_per_request_ > FlatKVCompletionState::kMaxDispatches) {
        throw std::invalid_argument("flat KV completion FIFO depth must be one or two");
    }
    if (schema_.empty()) {
        throw std::invalid_argument("flat KV completion requires a non-empty group schema");
    }
    for (const KvCacheGroupSchema& group : schema_) {
        if (group.entry_stride_tokens <= 0) {
            throw std::invalid_argument("flat KV completion group stride must be positive");
        }
    }
}

FlatKVCompletionInput FlatKVCompletionLedger::RecordDispatch(FlatKVCompletionState& state, FlatKVDispatchSpec spec) {
    if (spec.dispatch_raw_start < 0 || spec.dispatch_raw_end < spec.dispatch_raw_start ||
        spec.protected_raw_end < spec.dispatch_raw_end) {
        throw std::invalid_argument("flat KV dispatch raw bounds are invalid");
    }
    if (state.next_dispatch_seq_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("flat KV dispatch sequence exhausted");
    }
    if (state.size_ >= max_outstanding_per_request_) {
        throw std::length_error("flat KV request exceeded its configured overlap depth");
    }
    for (std::size_t i = 0; i < state.size_; ++i) {
        if (state.At(i).canceled) {
            throw std::logic_error("flat KV dispatch is blocked until canceled execution fences retire");
        }
    }

    FlatKVCompletionInput input{
        .table_generation = state.table_generation_,
        .dispatch_seq = state.next_dispatch_seq_,
        .dispatch_raw_start = spec.dispatch_raw_start,
        .dispatch_raw_end = spec.dispatch_raw_end,
        .protected_raw_end = spec.protected_raw_end,
    };
    state.PushBack(FlatKVCompletionState::DispatchRecord{
        .dispatch_seq = state.next_dispatch_seq_,
        .dispatch_raw_start = spec.dispatch_raw_start,
        .dispatch_raw_end = spec.dispatch_raw_end,
        .apply_fsm_result = spec.apply_fsm_result,
    });
    ++state.next_dispatch_seq_;
    return input;
}

FlatKVReadyCompletion FlatKVCompletionLedger::MakeReady(const std::string& request_id, std::uint64_t table_generation,
                                                        const FlatKVCompletionState::DispatchRecord& record,
                                                        const forward::FlatKVCompletion& completion,
                                                        std::span<const std::int32_t> tokens) {
    if (!record.apply_fsm_result && !tokens.empty()) {
        throw std::invalid_argument("mid-prefill flat KV completion must not carry result tokens");
    }
    for (std::size_t i = 0; i < schema_.size(); ++i) {
        const std::int32_t stride = schema_[i].entry_stride_tokens;
        ready_raw_ends_scratch_[i] = completion.accepted_raw_end / stride * stride;
    }
    return FlatKVReadyCompletion{
        .request_id = request_id,
        .table_generation = table_generation,
        .dispatch_seq = record.dispatch_seq,
        .dispatch_raw_start = record.dispatch_raw_start,
        .dispatch_raw_end = record.dispatch_raw_end,
        .accepted_raw_end = completion.accepted_raw_end,
        .tokens = tokens,
        .ready_raw_ends = ready_raw_ends_scratch_,
        .apply_fsm_result = record.apply_fsm_result,
    };
}

FlatKVCompletionPrepareResult FlatKVCompletionLedger::Prepare(const FlatKVCompletionState& state,
                                                              const std::string& request_id,
                                                              forward::FlatKVCompletion completion,
                                                              const std::vector<std::int32_t>& tokens) {
    if (state.table_generation_ != completion.table_generation) {
        return {
            .disposition = FlatKVCompletionDisposition::kStale,
        };
    }
    if (state.size_ == 0 || completion.dispatch_seq < state.next_retire_seq_) {
        return {
            .disposition = FlatKVCompletionDisposition::kStale,
        };
    }
    if (completion.dispatch_seq >= state.next_dispatch_seq_) {
        throw std::invalid_argument("flat KV completion references an undispatched sequence");
    }
    const FlatKVCompletionState::DispatchRecord& record = state.Front();
    if (completion.dispatch_seq != record.dispatch_seq) {
        throw std::invalid_argument("flat KV completion arrived out of FIFO order");
    }
    // A canceled row still represents a real executor fence. Validate its
    // accepted boundary before suppressing logical result publication.
    if (completion.accepted_raw_end < record.dispatch_raw_start ||
        completion.accepted_raw_end > record.dispatch_raw_end) {
        throw std::invalid_argument("flat KV accepted end is outside its dispatch interval");
    }

    const FlatKVCompletionDisposition disposition =
        record.canceled ? FlatKVCompletionDisposition::kCanceled : FlatKVCompletionDisposition::kApplied;
    FlatKVCompletionPrepareResult result{
        .disposition = disposition,
        .ticket = FlatKVCompletionTicket{state.table_generation_, record.dispatch_seq, completion.accepted_raw_end,
                                         disposition},
    };
    if (!record.canceled) {
        result.ready.emplace(MakeReady(request_id, state.table_generation_, record, completion, tokens));
    }
    return result;
}

std::size_t FlatKVCompletionLedger::Commit(FlatKVCompletionState& state,
                                           const FlatKVCompletionTicket& ticket) noexcept {
    if (ticket.disposition == FlatKVCompletionDisposition::kStale || state.size_ == 0 ||
        state.table_generation_ != ticket.table_generation || state.Front().dispatch_seq != ticket.dispatch_seq ||
        state.Front().canceled != (ticket.disposition == FlatKVCompletionDisposition::kCanceled)) {
        std::terminate();
    }
    const bool short_accept = ticket.disposition == FlatKVCompletionDisposition::kApplied &&
                              ticket.accepted_raw_end < state.Front().dispatch_raw_end;
    state.PopFront();
    ++state.next_retire_seq_;
    if (short_accept) {
        (void)state.CancelOutstanding();
    }
    return state.size_;
}

}  // namespace tokenspeed
