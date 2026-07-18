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

#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>

namespace tokenspeed {

PageAllocator::PageAllocator(std::int32_t page_size, std::int32_t total_pages)
    : page_size_(page_size), total_pages_(total_pages) {
    free_pages_.reserve(static_cast<std::size_t>(total_pages_));
    for (std::int32_t i = 1; i < total_pages_; ++i) {
        free_pages_.push_back(i);
    }
}

OwnedPages PageAllocator::Allocate(std::int32_t num_pages) {
    if (num_pages <= 0 || static_cast<std::size_t>(num_pages) > free_pages_.size()) {
        return {};
    }
    std::vector<std::int32_t> pages;
    if (std::getenv("DEBUG_MEM")) {
        spdlog::debug("Free Pages Before Allocate: {}", free_pages_.size());
    }
    pages.reserve(static_cast<std::size_t>(num_pages));
    for (std::int32_t i = 0; i < num_pages; ++i) {
        pages.push_back(free_pages_.back());
        free_pages_.pop_back();
    }
    if (std::getenv("DEBUG_MEM")) {
        spdlog::debug("Free Pages After Allocate: {}", free_pages_.size());
        spdlog::debug("Allocated pages: [{}]", fmt::join(pages, ", "));
    }
    return OwnedPages{this, std::move(pages)};
}

void PageAllocator::Deallocate(const std::vector<std::int32_t>& pages) {
    if (std::getenv("DEBUG_MEM")) {
        spdlog::debug("Pages to Deallocate: [{}]", fmt::join(pages, ", "));
        spdlog::debug("Free Pages Before Deallocate: {}", free_pages_.size());
    }
    free_pages_.insert(free_pages_.end(), pages.begin(), pages.end());
    if (std::getenv("DEBUG_MEM")) {
        spdlog::debug("Free Pages After Deallocate: {}", free_pages_.size());
    }
}

}  // namespace tokenspeed
