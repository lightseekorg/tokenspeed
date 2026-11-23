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

#include <string>
#include <vector>

#include "scheduler/cache_event_sink.h"

namespace tokenspeed::test {

struct RecordedSlotOwnership {
    bool assigned;
    std::int32_t slot;
    std::uint64_t generation;
    std::uint64_t admission_id;
    std::uint64_t previous_admission_id;
    std::string request_id;
};

class RecordingCacheEventSink : public CacheEventSink {
public:
    void OnSlotAssigned(const SlotOwnershipEvent& event) noexcept override { record(event, true); }
    void OnSlotReleased(const SlotOwnershipEvent& event) noexcept override { record(event, false); }
    void OnPrefixLoadScheduled(const LoadBackOperation& operation) noexcept override {
        prefix_loads.push_back(operation);
    }
    void OnRestoreScheduled(const SnapshotRestoreOperation& operation) noexcept override {
        restores.push_back(operation);
    }

    std::vector<RecordedSlotOwnership> slots;
    std::vector<SnapshotRestoreOperation> restores;
    std::vector<LoadBackOperation> prefix_loads;

private:
    void record(const SlotOwnershipEvent& event, bool assigned) {
        slots.push_back({assigned, event.slot, event.generation, event.admission_id, event.previous_admission_id,
                         std::string{event.request_id}});
    }
};

}  // namespace tokenspeed::test
