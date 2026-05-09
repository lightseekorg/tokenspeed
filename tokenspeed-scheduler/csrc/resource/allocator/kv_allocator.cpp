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

#include "resource/allocator/kv_allocator.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "resource/allocator/page_allocator.h"

namespace tokenspeed {

LocalKVAllocator::LocalKVAllocator(PageAllocator* allocator, std::int32_t num_tokens)
    : allocator_(allocator), page_size_{allocator_->PageSize()} {
    Acquire(num_tokens);
};

void LocalKVAllocator::Acquire(std::int32_t num_tokens) {
    if (num_tokens <= tail_page_available_tokens_) {
        tail_page_available_tokens_ -= num_tokens;
    } else {
        std::int32_t over_needed = num_tokens - tail_page_available_tokens_;
        std::int32_t num_pages = (over_needed + page_size_ - 1) / page_size_;
        OwnedPages new_pages = allocator_->Allocate(num_pages);
        if (new_pages.Size() < num_pages) {
            throw std::runtime_error("LocalKVAllocator::Acquire: insufficient KV pages; requested=" +
                                     std::to_string(num_pages) + "; allocated=" + std::to_string(new_pages.Size()));
        }
        pages_.Append(std::move(new_pages));

        std::int32_t used_in_tail = (num_tokens - tail_page_available_tokens_) % page_size_;
        tail_page_available_tokens_ = used_in_tail == 0 ? 0 : page_size_ - used_in_tail;
    }
}

OwnedPages LocalKVAllocator::TakeFullPages() {
    if (tail_page_available_tokens_ == 0 || pages_.Size() <= 1) {
        // All pages are full, or only one page (the tail) — take everything
        // Actually if tail_page_available_tokens_ == 0, all pages are full.
        // If pages_.Size() <= 1, there's at most one tail page, no full pages.
        if (tail_page_available_tokens_ == 0) {
            return std::move(pages_);
        }
        return OwnedPages{};
    }
    // Keep the last page (tail), return the rest
    std::int32_t full_count = pages_.Size() - 1;
    return pages_.TakeFirst(full_count);
}

}  // namespace tokenspeed
