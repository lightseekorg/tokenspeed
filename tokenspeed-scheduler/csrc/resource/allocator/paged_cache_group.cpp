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

#include "resource/allocator/paged_cache_group.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tokenspeed {

namespace {

std::vector<std::int32_t> CollectLivePageIds(const std::vector<std::int32_t>& page_ids) {
    std::vector<std::int32_t> live;
    live.reserve(page_ids.size());
    for (std::int32_t id : page_ids) {
        if (id >= 0) {
            live.push_back(id);
        }
    }
    return live;
}

std::int32_t CeilDivPositive(std::int32_t numer, std::int32_t denom) {
    if (numer <= 0) return 0;
    return (numer + denom - 1) / denom;
}

}  // namespace

void PagedCacheGroupConfig::Validate() const {
    if (group_id.empty()) {
        throw std::invalid_argument("PagedCacheGroupConfig: group_id must be non-empty");
    }
    if (rows_per_page <= 0) {
        throw std::invalid_argument("PagedCacheGroupConfig: rows_per_page must be > 0");
    }
    if (entry_stride_tokens <= 0) {
        throw std::invalid_argument("PagedCacheGroupConfig: entry_stride_tokens must be > 0");
    }
    if (total_pages < 1) {
        throw std::invalid_argument("PagedCacheGroupConfig: total_pages must include the dummy page");
    }
    if (retention == Retention::SlidingWindow &&
        (!sliding_window_tokens.has_value() || *sliding_window_tokens <= 0)) {
        throw std::invalid_argument(
            "PagedCacheGroupConfig: sliding_window_tokens must be > 0 for sliding groups");
    }
}

PagedCacheGroupAllocator::PagedCacheGroupAllocator(PagedCacheGroupConfig config) : config_(std::move(config)) {
    config_.Validate();
    free_pages_.reserve(static_cast<std::size_t>(config_.total_pages));
    for (std::int32_t i = 1; i < config_.total_pages; ++i) {
        free_pages_.push_back(i);
    }
}

std::vector<std::int32_t> PagedCacheGroupAllocator::Allocate(std::int32_t num_pages) {
    if (num_pages <= 0) {
        return {};
    }
    if (static_cast<std::size_t>(num_pages) > free_pages_.size()) {
        ++failed_alloc_count_;
        return {};
    }

    std::vector<std::int32_t> pages;
    pages.reserve(static_cast<std::size_t>(num_pages));
    for (std::int32_t i = 0; i < num_pages; ++i) {
        pages.push_back(free_pages_.back());
        free_pages_.pop_back();
    }
    allocated_pages_total_ += num_pages;
    return pages;
}

void PagedCacheGroupAllocator::Deallocate(const std::vector<std::int32_t>& pages) {
    free_pages_.insert(free_pages_.end(), pages.begin(), pages.end());
    released_pages_total_ += static_cast<std::int64_t>(pages.size());
}

PagedCacheGroupTable::~PagedCacheGroupTable() {
    if (allocator_ == nullptr || page_ids_.empty()) {
        return;
    }
    auto live = CollectLivePageIds(page_ids_);
    if (!live.empty()) {
        allocator_->Deallocate(live);
    }
}

PagedCacheGroupTable& PagedCacheGroupTable::operator=(PagedCacheGroupTable&& other) noexcept {
    if (this != &other) {
        if (allocator_ != nullptr && !page_ids_.empty()) {
            auto live = CollectLivePageIds(page_ids_);
            if (!live.empty()) {
                allocator_->Deallocate(live);
            }
        }
        allocator_ = std::exchange(other.allocator_, nullptr);
        page_ids_ = std::move(other.page_ids_);
        raw_token_cursor_ = std::exchange(other.raw_token_cursor_, 0);
        released_pages_count_ = std::exchange(other.released_pages_count_, 0);
    }
    return *this;
}

void PagedCacheGroupTable::Acquire(std::int32_t target_raw_tokens_exclusive) {
    if (allocator_ == nullptr) {
        throw std::logic_error("PagedCacheGroupTable::Acquire: no allocator bound");
    }
    if (target_raw_tokens_exclusive < 0) {
        throw std::invalid_argument("PagedCacheGroupTable::Acquire: target must be >= 0");
    }
    if (target_raw_tokens_exclusive <= raw_token_cursor_) {
        return;
    }

    const auto& cfg = allocator_->Config();
    const std::int32_t entries = CeilDivPositive(target_raw_tokens_exclusive, cfg.entry_stride_tokens);
    const std::int32_t pages_needed = (entries + cfg.rows_per_page - 1) / cfg.rows_per_page;
    const std::int32_t pages_to_allocate = pages_needed - Size();
    if (pages_to_allocate > 0) {
        auto fresh = allocator_->Allocate(pages_to_allocate);
        if (static_cast<std::int32_t>(fresh.size()) < pages_to_allocate) {
            throw std::runtime_error(
                "PagedCacheGroupTable::Acquire: failed to allocate pages for group " + cfg.group_id);
        }
        page_ids_.insert(page_ids_.end(), fresh.begin(), fresh.end());
    }
    raw_token_cursor_ = target_raw_tokens_exclusive;
}

std::vector<std::int32_t> PagedCacheGroupTable::ReleaseSkipped(std::int32_t window_lower_bound) {
    if (allocator_ == nullptr || page_ids_.empty() || window_lower_bound <= 0) {
        return {};
    }
    const auto& cfg = allocator_->Config();
    if (cfg.retention != PagedCacheGroupConfig::Retention::SlidingWindow) {
        return {};
    }

    const std::int32_t raw_per_page = cfg.RawTokensPerPage();
    if (raw_per_page <= 0) {
        return {};
    }
    const std::int32_t target = window_lower_bound / raw_per_page;
    if (target <= released_pages_count_) {
        return {};
    }

    const std::int32_t end = std::min(target, Size());
    std::vector<std::int32_t> released;
    released.reserve(static_cast<std::size_t>(end - released_pages_count_));
    for (std::int32_t i = released_pages_count_; i < end; ++i) {
        if (page_ids_[i] >= 0) {
            released.push_back(page_ids_[i]);
        }
        page_ids_[i] = -1;
    }
    released_pages_count_ = end;
    if (!released.empty()) {
        allocator_->Deallocate(released);
    }
    return released;
}

std::vector<std::int32_t> PagedCacheGroupTable::ReleaseAll() {
    std::vector<std::int32_t> live;
    if (allocator_ != nullptr && !page_ids_.empty()) {
        live = CollectLivePageIds(page_ids_);
        if (!live.empty()) {
            allocator_->Deallocate(live);
        }
    }
    page_ids_.clear();
    raw_token_cursor_ = 0;
    released_pages_count_ = 0;
    return live;
}

std::int32_t PagedCacheGroupTable::RowsPerPage() const {
    return allocator_ != nullptr ? allocator_->Config().rows_per_page : 0;
}

std::int32_t PagedCacheGroupTable::EntryStrideTokens() const {
    return allocator_ != nullptr ? allocator_->Config().entry_stride_tokens : 0;
}

std::int32_t PagedCacheGroupTable::RawTokensPerPage() const {
    return allocator_ != nullptr ? allocator_->Config().RawTokensPerPage() : 0;
}

bool PagedCacheGroupTable::IsSliding() const {
    return allocator_ != nullptr &&
           allocator_->Config().retention == PagedCacheGroupConfig::Retention::SlidingWindow;
}

std::int32_t PagedCacheGroupTable::SlidingWindowTokens() const {
    if (allocator_ == nullptr) {
        return 0;
    }
    const auto& cfg = allocator_->Config();
    if (cfg.retention != PagedCacheGroupConfig::Retention::SlidingWindow) {
        return 0;
    }
    return cfg.sliding_window_tokens.value_or(0);
}

}  // namespace tokenspeed
