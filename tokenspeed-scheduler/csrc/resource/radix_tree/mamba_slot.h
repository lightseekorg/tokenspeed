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
#include <functional>
#include <utility>

namespace tokenspeed {

class MambaChunkAllocator;

class MambaSlot {
public:
    using Releaser = std::function<void(std::int32_t)>;

    MambaSlot(std::int32_t index, MambaChunkAllocator* allocator);
    MambaSlot(std::int32_t index, Releaser releaser) : index_{index}, releaser_{std::move(releaser)} {}

    ~MambaSlot();

    MambaSlot(MambaSlot&& other) noexcept
        : index_{std::exchange(other.index_, -1)}, releaser_{std::move(other.releaser_)} {}

    MambaSlot& operator=(MambaSlot&& other) noexcept {
        if (this != &other) {
            release();
            index_ = std::exchange(other.index_, -1);
            releaser_ = std::move(other.releaser_);
        }
        return *this;
    }

    MambaSlot(const MambaSlot&) = delete;
    MambaSlot& operator=(const MambaSlot&) = delete;

    std::int32_t Index() const { return index_; }

private:
    void release();

    std::int32_t index_{-1};
    Releaser releaser_{};
};

}  // namespace tokenspeed
