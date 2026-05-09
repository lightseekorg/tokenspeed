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
#include <utility>
#include <vector>

namespace tokenspeed {

class PageAllocator;

class OwnedPages {
public:
    OwnedPages() = default;

    OwnedPages(PageAllocator* allocator, std::vector<std::int32_t> ids) : allocator_{allocator}, ids_{std::move(ids)} {}

    ~OwnedPages();

    OwnedPages(const OwnedPages&) = delete;
    OwnedPages& operator=(const OwnedPages&) = delete;

    OwnedPages(OwnedPages&& other) noexcept
        : allocator_{std::exchange(other.allocator_, nullptr)}, ids_{std::move(other.ids_)} {}

    OwnedPages& operator=(OwnedPages&& other) noexcept;

    const std::vector<std::int32_t>& Ids() const { return ids_; }
    std::int32_t Size() const { return static_cast<std::int32_t>(ids_.size()); }
    bool Empty() const { return ids_.empty(); }

    // Split off the first n pages; self keeps the rest.
    OwnedPages TakeFirst(std::int32_t n);

    // Split off the last n pages; self keeps the rest.
    OwnedPages TakeLast(std::int32_t n);

    // Absorb all pages from other into this. Both must share the same allocator.
    void Append(OwnedPages other);

    // Drop ownership without returning the pages to the allocator.
    void ReleaseOwnershipByID(const std::vector<std::int32_t>& ids);

private:
    PageAllocator* allocator_{nullptr};
    std::vector<std::int32_t> ids_;
};

}  // namespace tokenspeed
