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

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "scheduler/operations/forward.h"
#include "scheduler/outside_events/forward.h"

namespace tokenspeed {

struct FlatKVCompletionGroupSchema {
    std::string group_id;
    std::uint32_t required_domain_mask{};
    std::int32_t entry_stride_tokens{1};
};

struct FlatKVDispatchSpec {
    std::string request_id;
    std::int32_t dispatch_raw_start{};
    std::int32_t dispatch_raw_end{};
    std::int32_t protected_raw_end{};
    bool legacy_result_expected{};
};

// Validated, ordered result passed to the scheduler's ready-only publisher
// callback before its token/FSM mutation is applied.
struct FlatKVReadyCompletion {
    FlatKVCompletionInput input;
    forward::FlatKVCompletion completion;
    std::vector<std::int32_t> tokens;
    // Monotonic group publication boundaries: persistent required-domain
    // minima, clamped to the stride-aligned accepted end.
    std::vector<std::int32_t> ready_raw_ends;
    bool apply_fsm_result{};
};

enum class FlatKVCompletionDisposition : std::uint8_t {
    kApplied,
    kBuffered,
    kCanceled,
    kStaleGeneration,
};

struct FlatKVCompletionSubmitResult {
    FlatKVCompletionDisposition disposition{FlatKVCompletionDisposition::kBuffered};
    std::size_t applied_count{};
    std::size_t invalidated_dispatches{};
    // Canceled successors whose execution fences have now arrived. Their
    // result tokens/progress were deliberately not applied.
    std::size_t retired_canceled_dispatches{};
};

// Allocation-free callbacks into one caller-owned prepared transaction slot.
// `prepare` is the last fallible step and populates that slot; `commit` crosses
// the ledger progress boundary and is therefore an explicitly noexcept
// function pointer. Keeping both in one value prevents returning a dangling
// action to stack-local preparation state.
struct FlatKVCompletionCallbacks {
    void* context{};
    void (*prepare)(void*, const FlatKVReadyCompletion&){};
    void (*commit)(void*) noexcept{};

    explicit operator bool() const noexcept { return prepare != nullptr && commit != nullptr; }
};

class FlatKVCompletionLedger;

// A scheduler-plan transaction for newly recorded dispatches.  The caller
// reserves the maximum batch size before the first ledger mutation, records
// each row through this object, and commits only after the complete execution
// plan has been materialized.  Stack unwinding rolls the rows back in reverse
// order without allocating, so a failed table export or plan publication
// cannot leave completion debt for device work that was never dispatched.
class FlatKVDispatchBatch {
public:
    FlatKVDispatchBatch(FlatKVCompletionLedger& ledger, std::size_t max_dispatches);
    ~FlatKVDispatchBatch() noexcept;

    FlatKVDispatchBatch(const FlatKVDispatchBatch&) = delete;
    FlatKVDispatchBatch& operator=(const FlatKVDispatchBatch&) = delete;
    FlatKVDispatchBatch(FlatKVDispatchBatch&&) = delete;
    FlatKVDispatchBatch& operator=(FlatKVDispatchBatch&&) = delete;

    FlatKVCompletionInput Record(FlatKVDispatchSpec spec);
    void Commit() noexcept { committed_ = true; }

private:
    struct RollbackHandle {
        const std::string* request_id{};
        std::uint64_t table_generation{};
        std::uint64_t dispatch_seq{};
    };

    FlatKVCompletionLedger* ledger_{};
    std::vector<RollbackHandle> handles_;
    std::size_t handle_count_{};
    bool committed_{};

    friend class FlatKVCompletionLedger;
};

struct FlatKVCompletionLedgerStats {
    std::uint64_t applied_results{};
    std::uint64_t buffered_results{};
    std::uint64_t stale_generation_results{};
    std::uint64_t duplicate_results{};
    std::uint64_t invalid_results{};
    std::uint64_t buffer_overflows{};
    std::uint64_t invalidated_dispatches{};
    std::uint64_t canceled_results{};
    std::uint64_t legacy_results{};
};

class FlatKVCompletionLedger {
public:
    // The prepare callback may validate and allocate, but must not mutate
    // irreversible scheduler/FSM/cache state. A failed preparation leaves the
    // ordered result buffered and retryable; its paired callback is the
    // no-throw commit seam.
    FlatKVCompletionLedger(std::size_t max_buffered_results, std::vector<FlatKVCompletionGroupSchema> group_schema);

    FlatKVCompletionInput RecordDispatch(FlatKVDispatchSpec spec);
    FlatKVCompletionSubmitResult Submit(forward::FlatKVCompletion completion, std::vector<std::int32_t> tokens,
                                        FlatKVCompletionCallbacks callbacks);

    // Legacy ExtendResult has no completion metadata. Retire it and every
    // preceding mid-prefill dispatch, which historically emitted no event.
    std::size_t RetireLegacyResult(const std::string& request_id);
    // Stop applying every outstanding dispatch while preserving each dispatch
    // until its execution fence arrives. This is the terminal/retract path:
    // callers must not free or reuse exported block-table pages before the
    // canceled completions retire.
    std::size_t CancelOutstanding(const std::string& request_id);
    std::size_t Invalidate(const std::string& request_id);

    bool HasOutstanding(const std::string& request_id) const;
    bool HasCanceledOutstanding(const std::string& request_id) const;
    std::size_t OutstandingCount(const std::string& request_id) const;
    // Exclusive raw end of the newest still-outstanding dispatch. Overlap
    // scheduling projects the next logical interval from this boundary while
    // the request FSM still reflects only completed results.
    std::optional<std::int32_t> LastOutstandingDispatchRawEnd(const std::string& request_id) const;
    std::size_t BufferedResultCount(const std::string& request_id) const;
    std::size_t TotalOutstandingCount() const;
    // Structured producers owe every dispatch, including first/mid-prefill
    // work before the first result establishes its response shape.
    bool HasBlockingOutstanding() const;
    const FlatKVCompletionLedgerStats& Stats() const { return stats_; }
    std::size_t MaxBufferedResults() const { return max_buffered_results_; }

private:
    using RollbackHandle = FlatKVDispatchBatch::RollbackHandle;

    struct PendingResult {
        FlatKVReadyCompletion ready;
        bool counted_as_buffered{};
        bool preparation_failed{};
    };

    struct DispatchRecord {
        FlatKVCompletionInput input;
        bool legacy_result_expected{};
        bool canceled{};
        std::optional<PendingResult> pending;
    };

    struct GenerationState {
        std::uint64_t table_generation{};
        std::uint64_t next_dispatch_seq{};
        std::uint64_t next_apply_seq{};
        // Group-aligned absolute producer watermarks. This ledger is their
        // single owner so out-of-order/canceled results cannot leak progress.
        std::vector<std::array<std::int32_t, 32>> domain_valid_ends;
        std::vector<std::int32_t> last_ready_raw_ends;
        std::deque<DispatchRecord> dispatches;
        std::size_t buffered_results{};
        bool saw_structured_completion{};
    };

    static std::uint64_t AllocateGeneration();
    std::pair<FlatKVCompletionInput, RollbackHandle> RecordDispatchTracked(FlatKVDispatchSpec spec);
    void RollbackDispatch(const RollbackHandle& handle) noexcept;
    FlatKVReadyCompletion ValidateAndMakeReady(const DispatchRecord& record, forward::FlatKVCompletion completion,
                                               std::vector<std::int32_t> tokens) const;
    FlatKVCompletionSubmitResult DrainReady(GenerationState& state, FlatKVCompletionCallbacks callbacks);
    std::size_t RetireCanceledReady(GenerationState& state) noexcept;
    void ComputeReady(const GenerationState& state, FlatKVReadyCompletion& ready) const noexcept;
    void CommitProgress(GenerationState& state, const FlatKVReadyCompletion& ready) const noexcept;
    void ClampProgressToAccepted(GenerationState& state, std::int32_t accepted_raw_end) const noexcept;
    [[noreturn]] void RejectDuplicate(const std::string& message);
    [[noreturn]] void RejectInvalid(const std::string& message);

    std::size_t max_buffered_results_;
    std::vector<FlatKVCompletionGroupSchema> group_schema_;
    std::unordered_map<std::string, GenerationState> active_;
    FlatKVCompletionLedgerStats stats_;

    friend class FlatKVDispatchBatch;
};

}  // namespace tokenspeed
