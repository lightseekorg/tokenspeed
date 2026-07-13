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

#include <algorithm>
#include <atomic>
#include <bit>
#include <exception>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tokenspeed {
namespace {

std::atomic<std::uint64_t> g_next_flat_table_generation{1};

bool SameCompletionPayload(const FlatKVReadyCompletion& pending, const forward::FlatKVCompletion& completion,
                           const std::vector<std::int32_t>& tokens) noexcept {
    if (pending.tokens != tokens || pending.completion.request_id != completion.request_id ||
        pending.completion.table_generation != completion.table_generation ||
        pending.completion.dispatch_seq != completion.dispatch_seq ||
        pending.completion.accepted_raw_end != completion.accepted_raw_end ||
        pending.completion.protected_raw_end != completion.protected_raw_end ||
        pending.completion.groups.size() != completion.groups.size()) {
        return false;
    }
    for (std::size_t i = 0; i < completion.groups.size(); ++i) {
        const forward::FlatKVGroupCompletion& lhs = pending.completion.groups[i];
        const forward::FlatKVGroupCompletion& rhs = completion.groups[i];
        if (lhs.group_id != rhs.group_id || lhs.completed_domain_mask != rhs.completed_domain_mask ||
            lhs.domain_valid_ends != rhs.domain_valid_ends) {
            return false;
        }
    }
    return true;
}

}  // namespace

FlatKVCompletionLedger::FlatKVCompletionLedger(std::size_t max_buffered_results,
                                               std::vector<FlatKVCompletionGroupSchema> group_schema)
    : max_buffered_results_{max_buffered_results}, group_schema_{std::move(group_schema)} {
    if (max_buffered_results_ == 0) {
        throw std::invalid_argument("flat KV completion buffer bound must be positive");
    }
    for (std::size_t i = 0; i < group_schema_.size(); ++i) {
        const FlatKVCompletionGroupSchema& group = group_schema_[i];
        if (group.group_id.empty() || group.required_domain_mask == 0 || group.entry_stride_tokens <= 0) {
            throw std::invalid_argument("flat KV completion group schema is invalid");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (group_schema_[j].group_id == group.group_id) {
                throw std::invalid_argument("flat KV completion group schema contains duplicate group_id");
            }
        }
    }
}

std::uint64_t FlatKVCompletionLedger::AllocateGeneration() {
    const std::uint64_t generation = g_next_flat_table_generation.fetch_add(1, std::memory_order_relaxed);
    if (generation == 0 || generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("flat KV table generation space exhausted");
    }
    return generation;
}

FlatKVDispatchBatch::FlatKVDispatchBatch(FlatKVCompletionLedger& ledger, std::size_t max_dispatches)
    : ledger_{&ledger}, handles_(max_dispatches) {}

FlatKVDispatchBatch::~FlatKVDispatchBatch() noexcept {
    if (committed_) {
        return;
    }
    while (handle_count_ != 0) {
        ledger_->RollbackDispatch(handles_[--handle_count_]);
    }
}

FlatKVCompletionInput FlatKVDispatchBatch::Record(FlatKVDispatchSpec spec) {
    if (committed_ || ledger_ == nullptr) {
        throw std::logic_error("flat KV dispatch batch is not open");
    }
    if (handle_count_ == handles_.size()) {
        throw std::length_error("flat KV dispatch batch exceeded its reserved size");
    }
    auto [input, handle] = ledger_->RecordDispatchTracked(std::move(spec));
    // handles_ was fully sized before the first ledger mutation. This plain
    // assignment cannot allocate or throw, so every successful record is
    // immediately covered by the rollback destructor.
    handles_[handle_count_++] = handle;
    return input;
}

FlatKVCompletionInput FlatKVCompletionLedger::RecordDispatch(FlatKVDispatchSpec spec) {
    return RecordDispatchTracked(std::move(spec)).first;
}

std::pair<FlatKVCompletionInput, FlatKVCompletionLedger::RollbackHandle> FlatKVCompletionLedger::RecordDispatchTracked(
    FlatKVDispatchSpec spec) {
    if (spec.request_id.empty()) {
        throw std::invalid_argument("flat KV dispatch request_id must be non-empty");
    }
    if (spec.dispatch_raw_start < 0 || spec.dispatch_raw_end < spec.dispatch_raw_start ||
        spec.protected_raw_end < spec.dispatch_raw_end) {
        throw std::invalid_argument("flat KV dispatch raw bounds are invalid");
    }
    if (group_schema_.empty()) {
        throw std::invalid_argument("flat KV dispatch requires a non-empty completion group schema");
    }
    auto it = active_.find(spec.request_id);
    if (it == active_.end()) {
        // Construct the complete first record in local state. Any string/deque
        // allocation failure therefore leaves active_ untouched rather than
        // publishing an empty generation with stale seed watermarks.
        const std::uint64_t generation = AllocateGeneration();
        GenerationState initial;
        initial.table_generation = generation;
        initial.domain_valid_ends.resize(group_schema_.size());
        initial.last_ready_raw_ends.reserve(group_schema_.size());
        for (std::size_t group_index = 0; group_index < group_schema_.size(); ++group_index) {
            const FlatKVCompletionGroupSchema& schema = group_schema_[group_index];
            const std::int32_t seed = spec.dispatch_raw_start / schema.entry_stride_tokens * schema.entry_stride_tokens;
            initial.last_ready_raw_ends.push_back(seed);
            for (std::uint32_t domains = schema.required_domain_mask; domains != 0; domains &= domains - 1) {
                initial.domain_valid_ends[group_index][std::countr_zero(domains)] = seed;
            }
        }
        FlatKVCompletionInput input{
            .request_id = spec.request_id,
            .table_generation = generation,
            .dispatch_seq = 0,
            .dispatch_raw_start = spec.dispatch_raw_start,
            .dispatch_raw_end = spec.dispatch_raw_end,
            .protected_raw_end = spec.protected_raw_end,
        };
        initial.dispatches.push_back(DispatchRecord{
            .input = input,
            .legacy_result_expected = spec.legacy_result_expected,
        });
        initial.next_dispatch_seq = 1;
        auto [new_it, inserted] = active_.try_emplace(spec.request_id, std::move(initial));
        if (!inserted) {
            throw std::logic_error("flat KV dispatch generation was inserted reentrantly");
        }
        return {
            std::move(input),
            RollbackHandle{
                .request_id = &new_it->first,
                .table_generation = generation,
                .dispatch_seq = 0,
            },
        };
    }
    GenerationState& state = it->second;
    if (state.next_dispatch_seq == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("flat KV dispatch sequence exhausted");
    }
    if (!state.dispatches.empty() && state.dispatches.front().canceled) {
        throw std::logic_error("flat KV dispatch is blocked until canceled successor fences retire");
    }
    FlatKVCompletionInput input{
        .request_id = spec.request_id,
        .table_generation = state.table_generation,
        .dispatch_seq = state.next_dispatch_seq,
        .dispatch_raw_start = spec.dispatch_raw_start,
        .dispatch_raw_end = spec.dispatch_raw_end,
        .protected_raw_end = spec.protected_raw_end,
    };
    DispatchRecord record{
        .input = input,
        .legacy_result_expected = spec.legacy_result_expected,
    };
    static_assert(std::is_nothrow_move_constructible_v<DispatchRecord>);
    // deque insertion is the commit point. A failed allocation leaves both the
    // sequence and dispatch queue unchanged; only a successful insertion burns
    // the sequence number.
    state.dispatches.push_back(std::move(record));
    ++state.next_dispatch_seq;
    return {
        std::move(input),
        RollbackHandle{
            .request_id = &it->first,
            .table_generation = state.table_generation,
            .dispatch_seq = state.next_dispatch_seq - 1,
        },
    };
}

void FlatKVCompletionLedger::RollbackDispatch(const RollbackHandle& handle) noexcept {
    if (handle.request_id == nullptr) {
        std::terminate();
    }
    auto it = active_.find(*handle.request_id);
    if (it == active_.end() || &it->first != handle.request_id) {
        std::terminate();
    }
    GenerationState& state = it->second;
    if (state.table_generation != handle.table_generation || state.next_dispatch_seq == 0 ||
        state.next_dispatch_seq - 1 != handle.dispatch_seq || state.dispatches.empty()) {
        std::terminate();
    }
    const DispatchRecord& record = state.dispatches.back();
    if (record.pending.has_value() || record.input.table_generation != handle.table_generation ||
        record.input.dispatch_seq != handle.dispatch_seq) {
        std::terminate();
    }
    state.dispatches.pop_back();
    --state.next_dispatch_seq;
    if (handle.dispatch_seq == 0) {
        if (!state.dispatches.empty() || state.next_dispatch_seq != 0 || state.next_apply_seq != 0) {
            std::terminate();
        }
        active_.erase(it);
    }
}

FlatKVReadyCompletion FlatKVCompletionLedger::ValidateAndMakeReady(const DispatchRecord& record,
                                                                   forward::FlatKVCompletion completion,
                                                                   std::vector<std::int32_t> tokens) const {
    if (completion.request_id != record.input.request_id ||
        completion.table_generation != record.input.table_generation ||
        completion.dispatch_seq != record.input.dispatch_seq) {
        throw std::invalid_argument("flat KV completion identity does not match its dispatch");
    }
    if (completion.accepted_raw_end < record.input.dispatch_raw_start ||
        completion.accepted_raw_end > record.input.dispatch_raw_end ||
        completion.protected_raw_end != record.input.protected_raw_end) {
        throw std::invalid_argument("flat KV completion raw bounds do not match its dispatch");
    }
    if (completion.groups.size() != group_schema_.size()) {
        throw std::invalid_argument("flat KV completion group count does not match its dispatch");
    }
    if (!record.legacy_result_expected && !tokens.empty()) {
        throw std::invalid_argument("mid-prefill flat KV completion must not carry result tokens");
    }

    for (std::size_t i = 0; i < completion.groups.size(); ++i) {
        const FlatKVCompletionGroupSchema& expected = group_schema_[i];
        const forward::FlatKVGroupCompletion& completed = completion.groups[i];
        if (completed.group_id != expected.group_id) {
            throw std::invalid_argument("flat KV completion group order/id does not match its dispatch");
        }
        if (completed.completed_domain_mask != expected.required_domain_mask) {
            throw std::invalid_argument("flat KV completion producer domains disagree with the dispatch schema");
        }
        if (completed.domain_valid_ends.size() != std::popcount(completed.completed_domain_mask)) {
            throw std::invalid_argument("flat KV completion domain watermark count does not match its mask");
        }
        for (const std::int32_t valid_raw_end : completed.domain_valid_ends) {
            if (valid_raw_end < 0 || valid_raw_end > record.input.protected_raw_end) {
                throw std::invalid_argument("flat KV completion domain watermark is outside the protected range");
            }
            if (valid_raw_end % expected.entry_stride_tokens != 0) {
                throw std::invalid_argument("flat KV completion domain watermark is not entry-stride aligned");
            }
        }
    }
    return FlatKVReadyCompletion{
        .input = record.input,
        .completion = std::move(completion),
        .tokens = std::move(tokens),
        .ready_raw_ends = std::vector<std::int32_t>(group_schema_.size()),
        .apply_fsm_result = record.legacy_result_expected,
    };
}

void FlatKVCompletionLedger::ComputeReady(const GenerationState& state, FlatKVReadyCompletion& ready) const noexcept {
    for (std::size_t group_index = 0; group_index < group_schema_.size(); ++group_index) {
        const FlatKVCompletionGroupSchema& schema = group_schema_[group_index];
        const forward::FlatKVGroupCompletion& completed = ready.completion.groups[group_index];
        std::size_t packed_index = 0;
        std::array<std::int32_t, 32> prospective_domain_ends = state.domain_valid_ends[group_index];
        for (std::uint32_t domains = completed.completed_domain_mask; domains != 0; domains &= domains - 1) {
            const std::size_t bit = std::countr_zero(domains);
            prospective_domain_ends[bit] =
                std::max(prospective_domain_ends[bit], completed.domain_valid_ends[packed_index++]);
        }

        std::int32_t required_min = ready.input.protected_raw_end;
        for (std::uint32_t domains = schema.required_domain_mask; domains != 0; domains &= domains - 1) {
            required_min = std::min(required_min, prospective_domain_ends[std::countr_zero(domains)]);
        }
        const std::int32_t aligned_accepted =
            ready.completion.accepted_raw_end / schema.entry_stride_tokens * schema.entry_stride_tokens;
        const std::int32_t publishable = std::min(required_min, aligned_accepted);
        ready.ready_raw_ends[group_index] = std::max(state.last_ready_raw_ends[group_index], publishable);
    }
}

void FlatKVCompletionLedger::CommitProgress(GenerationState& state,
                                            const FlatKVReadyCompletion& ready) const noexcept {
    for (std::size_t group_index = 0; group_index < group_schema_.size(); ++group_index) {
        const forward::FlatKVGroupCompletion& completed = ready.completion.groups[group_index];
        std::size_t packed_index = 0;
        for (std::uint32_t domains = completed.completed_domain_mask; domains != 0; domains &= domains - 1) {
            const std::size_t bit = std::countr_zero(domains);
            state.domain_valid_ends[group_index][bit] =
                std::max(state.domain_valid_ends[group_index][bit], completed.domain_valid_ends[packed_index++]);
        }
        state.last_ready_raw_ends[group_index] = ready.ready_raw_ends[group_index];
    }
}

void FlatKVCompletionLedger::ClampProgressToAccepted(GenerationState& state,
                                                     std::int32_t accepted_raw_end) const noexcept {
    for (std::size_t group_index = 0; group_index < group_schema_.size(); ++group_index) {
        const std::int32_t stride = group_schema_[group_index].entry_stride_tokens;
        const std::int32_t aligned_cap = accepted_raw_end / stride * stride;
        for (std::int32_t& valid_raw_end : state.domain_valid_ends[group_index]) {
            valid_raw_end = std::min(valid_raw_end, aligned_cap);
        }
    }
}

[[noreturn]] void FlatKVCompletionLedger::RejectDuplicate(const std::string& message) {
    ++stats_.duplicate_results;
    throw std::invalid_argument(message);
}

[[noreturn]] void FlatKVCompletionLedger::RejectInvalid(const std::string& message) {
    ++stats_.invalid_results;
    throw std::invalid_argument(message);
}

FlatKVCompletionSubmitResult FlatKVCompletionLedger::Submit(forward::FlatKVCompletion completion,
                                                            std::vector<std::int32_t> tokens,
                                                            FlatKVCompletionCallbacks callbacks) {
    if (!callbacks) {
        throw std::invalid_argument("flat KV completion callbacks must provide prepare and noexcept commit");
    }
    auto state_it = active_.find(completion.request_id);
    if (state_it == active_.end() || state_it->second.table_generation != completion.table_generation) {
        ++stats_.stale_generation_results;
        return FlatKVCompletionSubmitResult{.disposition = FlatKVCompletionDisposition::kStaleGeneration};
    }
    GenerationState& state = state_it->second;
    const bool failed_front = !state.dispatches.empty() && state.dispatches.front().pending.has_value() &&
                              state.dispatches.front().pending->preparation_failed;
    if (completion.dispatch_seq >= state.next_dispatch_seq) {
        RejectInvalid("flat KV completion references an undispatched sequence");
    }
    if (failed_front && completion.dispatch_seq >= state.next_apply_seq) {
        const std::uint64_t pre_drain_offset = completion.dispatch_seq - state.next_apply_seq;
        if (pre_drain_offset < state.dispatches.size()) {
            const DispatchRecord& pre_drain_record = state.dispatches[static_cast<std::size_t>(pre_drain_offset)];
            if (pre_drain_record.pending.has_value() &&
                !SameCompletionPayload(pre_drain_record.pending->ready, completion, tokens)) {
                RejectDuplicate("duplicate flat KV completion is already buffered");
            }
        }
    }
    FlatKVCompletionSubmitResult result;
    if (failed_front) {
        // Drain before considering capacity or duplicate status for the new
        // payload. Thus every valid Submit is also a retry trigger, including
        // when the out-of-order buffer was already full.
        result = DrainReady(state, callbacks);
    }
    if (completion.dispatch_seq < state.next_apply_seq) {
        if (result.applied_count != 0 || result.retired_canceled_dispatches != 0) {
            return result;
        }
        RejectDuplicate("flat KV completion was already applied or invalidated");
    }
    const std::uint64_t offset = completion.dispatch_seq - state.next_apply_seq;
    if (offset >= state.dispatches.size()) {
        RejectInvalid("flat KV completion sequence is absent from the active ledger");
    }
    DispatchRecord& record = state.dispatches[static_cast<std::size_t>(offset)];
    if (record.pending.has_value()) {
        // A preparation exception is recoverable: retain the original payload
        // and let an exact retransmission trigger drain again. A different
        // payload for the same sequence remains a protocol error.
        if (!SameCompletionPayload(record.pending->ready, completion, tokens)) {
            RejectDuplicate("duplicate flat KV completion is already buffered");
        }
        if (result.applied_count != 0 || result.retired_canceled_dispatches != 0) {
            return result;
        }
        if (!record.pending->preparation_failed) {
            RejectDuplicate("duplicate flat KV completion is already buffered");
        }
    } else {
        // Preserve the ABI-mode guard even when this structured result is invalid,
        // over capacity, or cannot be materialized: a later legacy result must not
        // silently retire structured completion debt.
        state.saw_structured_completion = true;

        if (offset != 0 && state.buffered_results >= max_buffered_results_) {
            ++stats_.buffer_overflows;
            throw std::overflow_error("flat KV out-of-order completion buffer is full");
        }

        FlatKVReadyCompletion ready;
        try {
            // Validate and materialize exactly once, before touching the ledger.
            // Allocation failure therefore leaves no duplicate-looking pending slot.
            ready = ValidateAndMakeReady(record, std::move(completion), std::move(tokens));
        } catch (const std::invalid_argument&) {
            ++stats_.invalid_results;
            throw;
        }
        static_assert(std::is_nothrow_move_constructible_v<FlatKVReadyCompletion>);
        record.pending.emplace(PendingResult{
            .ready = std::move(ready),
            .counted_as_buffered = offset != 0,
        });
        if (offset != 0) {
            ++state.buffered_results;
            ++stats_.buffered_results;
        }
    }

    FlatKVCompletionSubmitResult drained = DrainReady(state, callbacks);
    result.applied_count += drained.applied_count;
    result.invalidated_dispatches += drained.invalidated_dispatches;
    result.retired_canceled_dispatches += drained.retired_canceled_dispatches;
    if (result.applied_count != 0) {
        result.disposition = FlatKVCompletionDisposition::kApplied;
    } else if (result.retired_canceled_dispatches != 0) {
        result.disposition = FlatKVCompletionDisposition::kCanceled;
    } else {
        result.disposition = FlatKVCompletionDisposition::kBuffered;
    }
    return result;
}

FlatKVCompletionSubmitResult FlatKVCompletionLedger::DrainReady(GenerationState& state,
                                                                FlatKVCompletionCallbacks callbacks) {
    FlatKVCompletionSubmitResult result;
    while (!state.dispatches.empty() && state.dispatches.front().pending.has_value()) {
        DispatchRecord& next = state.dispatches.front();
        FlatKVReadyCompletion& next_ready = next.pending->ready;
        const bool canceled = next.canceled;
        bool rejected_tail = false;
        if (!canceled) {
            ComputeReady(state, next_ready);
            try {
                callbacks.prepare(callbacks.context, next_ready);
            } catch (...) {
                next.pending->preparation_failed = true;
                throw;
            }
            CommitProgress(state, next_ready);
            callbacks.commit(callbacks.context);
            rejected_tail = next_ready.completion.accepted_raw_end < next_ready.input.dispatch_raw_end;
            if (rejected_tail) {
                ClampProgressToAccepted(state, next_ready.completion.accepted_raw_end);
            }
        }
        if (next.pending->counted_as_buffered) {
            --state.buffered_results;
        }
        state.dispatches.pop_front();
        ++state.next_apply_seq;
        if (canceled) {
            ++result.retired_canceled_dispatches;
            ++stats_.canceled_results;
        } else {
            ++result.applied_count;
            ++stats_.applied_results;
        }
        if (rejected_tail) {
            result.invalidated_dispatches = state.dispatches.size();
            stats_.invalidated_dispatches += result.invalidated_dispatches;
            for (DispatchRecord& successor : state.dispatches) {
                successor.canceled = true;
            }
        }
    }
    if (result.applied_count != 0) {
        result.disposition = FlatKVCompletionDisposition::kApplied;
    } else if (result.retired_canceled_dispatches != 0) {
        result.disposition = FlatKVCompletionDisposition::kCanceled;
    } else {
        result.disposition = FlatKVCompletionDisposition::kBuffered;
    }
    return result;
}

std::size_t FlatKVCompletionLedger::RetireCanceledReady(GenerationState& state) noexcept {
    std::size_t retired = 0;
    while (!state.dispatches.empty() && state.dispatches.front().canceled &&
           state.dispatches.front().pending.has_value()) {
        if (state.dispatches.front().pending->counted_as_buffered) {
            --state.buffered_results;
        }
        state.dispatches.pop_front();
        ++state.next_apply_seq;
        ++retired;
        ++stats_.canceled_results;
    }
    return retired;
}

std::size_t FlatKVCompletionLedger::RetireLegacyResult(const std::string& request_id) {
    auto it = active_.find(request_id);
    if (it == active_.end()) {
        return 0;
    }
    GenerationState& state = it->second;
    if (state.saw_structured_completion) {
        RejectInvalid("legacy ExtendResult cannot follow structured flat KV completions");
    }
    std::size_t retire_count = 0;
    for (const DispatchRecord& record : state.dispatches) {
        ++retire_count;
        if (record.pending.has_value()) {
            RejectInvalid("legacy ExtendResult cannot retire a buffered structured completion");
        }
        if (record.legacy_result_expected) {
            break;
        }
    }
    if (retire_count == 0 || retire_count > state.dispatches.size() ||
        !state.dispatches[retire_count - 1].legacy_result_expected) {
        return 0;
    }
    state.dispatches.erase(state.dispatches.begin(), state.dispatches.begin() + retire_count);
    state.next_apply_seq += retire_count;
    ++stats_.legacy_results;
    return retire_count;
}

std::size_t FlatKVCompletionLedger::CancelOutstanding(const std::string& request_id) {
    auto it = active_.find(request_id);
    if (it == active_.end()) {
        return 0;
    }
    std::size_t canceled = 0;
    for (DispatchRecord& record : it->second.dispatches) {
        if (!record.canceled) {
            record.canceled = true;
            ++canceled;
        }
    }
    stats_.invalidated_dispatches += canceled;
    // A failed prepare leaves a completion fence buffered. Once terminal
    // cancellation makes its logical result irrelevant, retire every already
    // arrived contiguous fence immediately; no future Submit is guaranteed.
    (void)RetireCanceledReady(it->second);
    return canceled;
}

std::size_t FlatKVCompletionLedger::Invalidate(const std::string& request_id) {
    auto it = active_.find(request_id);
    if (it == active_.end()) {
        return 0;
    }
    const std::size_t invalidated = it->second.dispatches.size();
    stats_.invalidated_dispatches += invalidated;
    active_.erase(it);
    return invalidated;
}

bool FlatKVCompletionLedger::HasOutstanding(const std::string& request_id) const {
    return OutstandingCount(request_id) != 0;
}

bool FlatKVCompletionLedger::HasCanceledOutstanding(const std::string& request_id) const {
    auto it = active_.find(request_id);
    if (it == active_.end()) {
        return false;
    }
    return std::ranges::any_of(it->second.dispatches, [](const DispatchRecord& record) { return record.canceled; });
}

std::size_t FlatKVCompletionLedger::OutstandingCount(const std::string& request_id) const {
    auto it = active_.find(request_id);
    return it == active_.end() ? 0 : it->second.dispatches.size();
}

std::optional<std::int32_t> FlatKVCompletionLedger::LastOutstandingDispatchRawEnd(const std::string& request_id) const {
    auto it = active_.find(request_id);
    if (it == active_.end() || it->second.dispatches.empty()) {
        return std::nullopt;
    }
    return it->second.dispatches.back().input.dispatch_raw_end;
}

std::size_t FlatKVCompletionLedger::BufferedResultCount(const std::string& request_id) const {
    auto it = active_.find(request_id);
    return it == active_.end() ? 0 : it->second.buffered_results;
}

std::size_t FlatKVCompletionLedger::TotalOutstandingCount() const {
    std::size_t total = 0;
    for (const auto& [_, state] : active_) {
        total += state.dispatches.size();
    }
    return total;
}

bool FlatKVCompletionLedger::HasBlockingOutstanding() const {
    for (const auto& [_, state] : active_) {
        // Every recorded dispatch is an execution fence, including a
        // mid-prefill dispatch whose legacy token result is not expected and
        // the very first dispatch before any structured completion has been
        // observed. Reclaiming its tables based on response-format history
        // would let an in-flight GPU operation write into reused pages.
        if (!state.dispatches.empty()) {
            return true;
        }
    }
    return false;
}

}  // namespace tokenspeed
