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
#include <vector>

#include "resource/allocator/owned_pages.h"

namespace tokenspeed {

class PageAllocator;

class LocalKVAllocator {
public:
    LocalKVAllocator(PageAllocator* allocator, std::int32_t num_tokens);

    ~LocalKVAllocator() = default;

    LocalKVAllocator(const LocalKVAllocator&) = delete;
    LocalKVAllocator& operator=(const LocalKVAllocator&) = delete;

    void Acquire(std::int32_t num_tokens);

    // Take all fully-used pages out of this allocator.
    // After this call, only the tail (partially-used) page remains in pages_.
    OwnedPages TakeFullPages();

    OwnedPages TakeFirst(std::int32_t n) { return pages_.TakeFirst(n); }

    std::vector<std::int32_t> Pages() const { return pages_.Ids(); };
    std::int32_t TailPageAvailableTokens() const { return tail_page_available_tokens_; };
    void ReleaseOwnershipByID(const std::vector<std::int32_t>& pages) { pages_.ReleaseOwnershipByID(pages); }

private:
    PageAllocator* allocator_;
    std::int32_t page_size_;
    OwnedPages pages_;
    std::int32_t tail_page_available_tokens_{};
};

}  // namespace tokenspeed
