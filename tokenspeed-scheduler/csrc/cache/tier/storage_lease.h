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

namespace tokenspeed {

class StorageTransferManager;

// A request's claim on one L3 operation it started: a query, or a GET and
// the Host blocks its ticket holds. The pre-admission state that waits on
// the operation owns the lease, so whatever ends the wait -- admission, a
// soft deadline, an abort -- drops it with the state. Dropping it detaches
// the request from the operation: a finished ticket retires, a command still
// in the outbox is withdrawn, one already issued is cancelled. Move-only; an
// empty lease claims nothing. The manager must outlive every lease it issued.
class StorageLease {
public:
    StorageLease() noexcept = default;
    StorageLease(const StorageLease&) = delete;
    StorageLease& operator=(const StorageLease&) = delete;
    StorageLease(StorageLease&& other) noexcept;
    StorageLease& operator=(StorageLease&& other) noexcept;
    ~StorageLease();

    explicit operator bool() const noexcept { return manager_ != nullptr; }
    // The id the runtime reports the operation's completion under.
    std::uint64_t OperationId() const noexcept { return operation_id_; }
    // Detaches now and leaves the lease empty.
    void reset() noexcept;

private:
    friend class StorageTransferManager;

    StorageLease(StorageTransferManager& manager, std::uint64_t operation_id) noexcept
        : manager_{&manager}, operation_id_{operation_id} {}

    StorageTransferManager* manager_{nullptr};
    std::uint64_t operation_id_{0};
};

}  // namespace tokenspeed
