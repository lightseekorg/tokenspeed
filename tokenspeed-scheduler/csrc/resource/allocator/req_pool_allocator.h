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

namespace tokenspeed {

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

    std::int32_t slot_{};

private:
    friend class ReqPoolAllocator;

    ReqPoolIndex(std::int32_t slot, ReqPoolAllocator* allocator);
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

    std::int32_t Size() const;
    std::int32_t AvailableSlots() const;

private:
    void deallocate(std::int32_t slot);

    friend struct ReqPoolIndex;

    std::int32_t size_{};
    std::deque<std::int32_t> free_slots_;
};

}  // namespace tokenspeed
