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

#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

namespace tokenspeed {

// Borrowed for the duration of a sink callback. Generation counts allocations
// of this slot, including assignments to other admissions; zero means unused.
struct SlotOwnershipEvent {
    std::int32_t slot;
    std::uint64_t generation;
    std::uint64_t admission_id;
    std::uint64_t previous_admission_id;
    std::string_view request_id;
};

// The sink must outlive the allocator and every handle. Callbacks must be
// nonblocking; a sink may drop records, but cannot affect resource ownership.
class SlotEventSink {
public:
    virtual ~SlotEventSink() = default;
    virtual void OnSlotAssigned(const SlotOwnershipEvent& event) noexcept = 0;
    virtual void OnSlotReleased(const SlotOwnershipEvent& event) noexcept = 0;
};

class ReqPoolAllocator;

struct ReqPoolIndex {
    // The allocator must outlive every index it creates.
    ReqPoolIndex() = default;

    // Forbid Copy construct
    ReqPoolIndex(const ReqPoolIndex&) = delete;
    ReqPoolIndex& operator=(const ReqPoolIndex&) = delete;

    // move construct
    ReqPoolIndex(ReqPoolIndex&& other) noexcept;

    // move assignment
    ReqPoolIndex& operator=(ReqPoolIndex&& other) noexcept;

    ~ReqPoolIndex();

    bool valid() const;
    std::uint64_t generation() const;
    std::uint64_t previous_admission_id() const;
    // Called once when an allocated request row is bound to its admission.
    // The external ID is retained only while observing this live ownership.
    void BindOwner(std::string_view request_id, std::uint64_t admission_id);

    // A moved-from index has slot_ == -1 and no longer owns a slot.
    std::int32_t slot_{};

private:
    friend class ReqPoolAllocator;

    ReqPoolIndex(std::int32_t slot, ReqPoolAllocator* allocator);
    void release() noexcept;
    ReqPoolAllocator* allocator_{nullptr};
};

class ReqPoolAllocator {
public:
    explicit ReqPoolAllocator(std::int32_t size);

    // Forbid copy construct
    ReqPoolAllocator(const ReqPoolAllocator&) = delete;
    ReqPoolAllocator& operator=(const ReqPoolAllocator&) = delete;

    // Forbid move construct
    ReqPoolAllocator(ReqPoolAllocator&&) = delete;
    ReqPoolAllocator& operator=(ReqPoolAllocator&&) = delete;

    ReqPoolIndex Allocate();
    // Install before the first allocation, including when replacing a sink.
    void SetEventSink(SlotEventSink* sink);

    std::int32_t Size() const;
    std::int32_t AvailableSlots() const;

private:
    void deallocate(std::int32_t slot);

    friend struct ReqPoolIndex;

    std::int32_t size_{};
    std::deque<std::int32_t> free_slots_;
    struct SlotHistory {
        std::uint64_t generation{0};
        std::uint64_t admission_id{0};
        std::uint64_t previous_admission_id{0};
        bool bound{false};
        // Only an observed live owner retains its external ID. Keeping it
        // here avoids moving a string on every request FSM transition (and
        // on decode-permit moves, which use the same RAII handle).
        std::string request_id;

        SlotOwnershipEvent ownership(std::int32_t slot) const {
            return {slot, generation, admission_id, previous_admission_id, request_id};
        }
    };
    std::vector<SlotHistory> slot_history_;
    SlotEventSink* event_sink_{nullptr};
    bool allocated_{false};
};

// The slot-state blob slots of retracted requests: the same RAII slot pool,
// sized by SchedulerConfig::max_retracted_requests, slots numbered from 1 like
// request-pool indices (the runtime's blob arena has max_retracted_requests + 1
// rows). A retracted request that finishes, aborts or is restored releases
// its slot by dropping the index.
using SnapshotSlotAllocator = ReqPoolAllocator;
using SnapshotSlotIndex = ReqPoolIndex;

// A decode keeps its execution permit across rounds, including finishing
// forwards and a restore awaiting its ACK. These indices never address data.
using DecodeSlotAllocator = ReqPoolAllocator;
using DecodeSlotIndex = ReqPoolIndex;

}  // namespace tokenspeed
