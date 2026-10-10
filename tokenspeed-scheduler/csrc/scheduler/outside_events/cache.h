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
#include <variant>

namespace tokenspeed {
namespace cache {
struct WriteBackDone {
    std::uint32_t op_id{};
};

// A prefix load-back landed: every row was copied. Its destinations were
// published at admission (Host-warm hits); the ACK drops the op's pins.
struct LoadBackDone {
    std::uint32_t op_id{};
};

// A pre-admission L3 prefetch (PrefetchOperation) finished: the first
// landed_pages prefix pages of its rows were fetched into their Host blocks
// (replica-converged), the rest were not. Both fields are constructor
// arguments so a caller cannot ACK an op without saying how much landed.
struct PrefetchDone {
    std::uint32_t op_id;
    std::int32_t landed_pages;

    PrefetchDone() = delete;
    PrefetchDone(std::uint32_t op_id, std::int32_t landed_pages) : op_id(op_id), landed_pages(landed_pages) {}
};

// A retraction image's tail leg (SnapshotStoreOperation) completed its D2H
// copies and the slot-state export.
struct SnapshotDone {
    std::uint32_t op_id{};
};

// A SnapshotRestoreOperation completed every H2D row and the slot-state
// import: the restored request is schedulable again.
struct RestoreDone {
    std::uint32_t op_id{};
};

};  // namespace cache

using CacheEvent = std::variant<cache::WriteBackDone, cache::LoadBackDone, cache::PrefetchDone, cache::SnapshotDone,
                                cache::RestoreDone>;
}  // namespace tokenspeed
