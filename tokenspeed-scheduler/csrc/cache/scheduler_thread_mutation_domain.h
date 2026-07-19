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

#include <exception>
#include <stdexcept>
#include <thread>

namespace tokenspeed {

// Immutable owner-thread token shared by every mutable object in one flat KV
// scheduler domain. The scheduler is single-threaded by contract; transaction
// boundaries validate that contract without synchronizing the hot path.
class SchedulerThreadMutationDomain {
public:
    SchedulerThreadMutationDomain() noexcept : owner_thread_id_{std::this_thread::get_id()} {}

    SchedulerThreadMutationDomain(const SchedulerThreadMutationDomain&) = delete;
    SchedulerThreadMutationDomain& operator=(const SchedulerThreadMutationDomain&) = delete;
    SchedulerThreadMutationDomain(SchedulerThreadMutationDomain&&) = delete;
    SchedulerThreadMutationDomain& operator=(SchedulerThreadMutationDomain&&) = delete;

    bool IsOwnerThread() const noexcept { return std::this_thread::get_id() == owner_thread_id_; }

    void AssertOwnerThread() const {
        if (!IsOwnerThread()) {
            throw std::logic_error("flat KV cache access outside the scheduler owner thread");
        }
    }

    // Destructors and final-reference release cannot report an ordinary
    // exception. Terminate before touching thread-confined state instead.
    void AssertOwnerThreadNoexcept() const noexcept {
        if (!IsOwnerThread()) {
            std::terminate();
        }
    }

    // RAII cleanup may run once per page inside an already-validated
    // transaction. Keep the local misuse check in debug builds without paying
    // for std::this_thread::get_id() once per page in release builds.
    void DebugAssertOwnerThreadNoexcept() const noexcept {
#ifndef NDEBUG
        AssertOwnerThreadNoexcept();
#endif
    }

private:
    const std::thread::id owner_thread_id_;
};

}  // namespace tokenspeed
