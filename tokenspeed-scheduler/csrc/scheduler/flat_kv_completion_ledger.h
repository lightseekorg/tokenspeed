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
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cache/cache_types.h"
#include "scheduler/operations/forward.h"
#include "scheduler/outside_events/forward.h"

namespace tokenspeed {

struct FlatKVDispatchSpec {
    std::int32_t dispatch_raw_start{};
    std::int32_t dispatch_raw_end{};
    std::int32_t protected_raw_end{};
    bool apply_fsm_result{};
};

// One fence-ready dispatch. Every reference/view is borrowed until the next
// Prepare call and must be consumed synchronously by the scheduler.
struct FlatKVReadyCompletion {
    const std::string& request_id;
    std::uint64_t table_generation{};
    std::uint64_t dispatch_seq{};
    std::int32_t dispatch_raw_start{};
    std::int32_t dispatch_raw_end{};
    std::int32_t accepted_raw_end{};
    std::span<const std::int32_t> tokens;
    std::span<const std::int32_t> ready_raw_ends;
    bool apply_fsm_result{};
};

enum class FlatKVCompletionDisposition : std::uint8_t {
    kApplied,
    kCanceled,
    kStale,
};

struct FlatKVCompletionRequestSnapshot {
    std::size_t outstanding_count{};
    bool has_canceled_outstanding{};
    std::optional<std::int32_t> last_dispatch_raw_end;
};

// The FIFO itself belongs to Request. The ledger owns only immutable schema
// and reusable completion-boundary scratch.
class FlatKVCompletionState {
public:
    static constexpr std::size_t kMaxDispatches = 2;

    FlatKVCompletionState();
    FlatKVCompletionState(const FlatKVCompletionState&) = delete;
    FlatKVCompletionState& operator=(const FlatKVCompletionState&) = delete;
    FlatKVCompletionState(FlatKVCompletionState&&) = delete;
    FlatKVCompletionState& operator=(FlatKVCompletionState&&) = delete;

    bool HasOutstanding() const noexcept { return size_ != 0; }
    FlatKVCompletionRequestSnapshot Snapshot() const noexcept;

    // Logical cancellation does not retire physical work. Every row stays in
    // the FIFO until its executor fence arrives.
    std::size_t CancelOutstanding() noexcept;

private:
    struct DispatchRecord {
        std::uint64_t dispatch_seq{};
        std::int32_t dispatch_raw_start{};
        std::int32_t dispatch_raw_end{};
        bool apply_fsm_result{};
        bool canceled{};
    };

    static std::uint64_t AllocateGeneration();
    DispatchRecord& Front() noexcept { return *dispatches_[head_]; }
    const DispatchRecord& Front() const noexcept { return *dispatches_[head_]; }
    DispatchRecord& Back() noexcept { return *dispatches_[(head_ + size_ - 1) % kMaxDispatches]; }
    const DispatchRecord& Back() const noexcept { return *dispatches_[(head_ + size_ - 1) % kMaxDispatches]; }
    DispatchRecord& At(std::size_t offset) noexcept { return *dispatches_[(head_ + offset) % kMaxDispatches]; }
    const DispatchRecord& At(std::size_t offset) const noexcept {
        return *dispatches_[(head_ + offset) % kMaxDispatches];
    }
    void PushBack(DispatchRecord record) noexcept;
    void PopFront() noexcept;

    std::uint64_t table_generation_{};
    std::uint64_t next_dispatch_seq_{};
    std::uint64_t next_retire_seq_{};
    std::array<std::optional<DispatchRecord>, kMaxDispatches> dispatches_;
    std::size_t head_{};
    std::size_t size_{};

    friend class FlatKVCompletionLedger;
};

// An opaque proof that Prepare validated the current FIFO front. Scheduler
// publication is prepared fallibly between Prepare and Commit; Commit then
// retires exactly this sequence without throwing.
struct FlatKVCompletionTicket {
public:
    FlatKVCompletionTicket(const FlatKVCompletionTicket&) = default;
    FlatKVCompletionTicket& operator=(const FlatKVCompletionTicket&) = default;
    FlatKVCompletionTicket(FlatKVCompletionTicket&&) noexcept = default;
    FlatKVCompletionTicket& operator=(FlatKVCompletionTicket&&) noexcept = default;

private:
    FlatKVCompletionTicket(std::uint64_t generation, std::uint64_t sequence, std::int32_t accepted_end,
                           FlatKVCompletionDisposition completion_disposition) noexcept
        : table_generation{generation},
          dispatch_seq{sequence},
          accepted_raw_end{accepted_end},
          disposition{completion_disposition} {}

    std::uint64_t table_generation{};
    std::uint64_t dispatch_seq{};
    std::int32_t accepted_raw_end{};
    FlatKVCompletionDisposition disposition{FlatKVCompletionDisposition::kStale};

    friend class FlatKVCompletionLedger;
};

struct FlatKVCompletionPrepareResult {
    FlatKVCompletionDisposition disposition{FlatKVCompletionDisposition::kStale};
    std::optional<FlatKVReadyCompletion> ready;
    std::optional<FlatKVCompletionTicket> ticket;
};

class FlatKVCompletionLedger {
public:
    FlatKVCompletionLedger(std::size_t max_outstanding_per_request, std::span<const KvCacheGroupSchema> schema);

    FlatKVCompletionInput RecordDispatch(FlatKVCompletionState& state, FlatKVDispatchSpec spec);

    // Prepare validates but never mutates the request FIFO. Applied results
    // borrow tokens and the ledger's aligned-boundary scratch until Commit.
    FlatKVCompletionPrepareResult Prepare(const FlatKVCompletionState& state, const std::string& request_id,
                                          forward::FlatKVCompletion completion,
                                          const std::vector<std::int32_t>& tokens);
    std::size_t Commit(FlatKVCompletionState& state, const FlatKVCompletionTicket& ticket) noexcept;

private:
    FlatKVReadyCompletion MakeReady(const std::string& request_id, std::uint64_t table_generation,
                                    const FlatKVCompletionState::DispatchRecord& record,
                                    const forward::FlatKVCompletion& completion, std::span<const std::int32_t> tokens);

    const std::size_t max_outstanding_per_request_;
    const std::span<const KvCacheGroupSchema> schema_;
    std::vector<std::int32_t> ready_raw_ends_scratch_;
};

}  // namespace tokenspeed
