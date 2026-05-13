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
    if (retention == Retention::SlidingWindow && (!sliding_window_tokens.has_value() || *sliding_window_tokens <= 0)) {
        throw std::invalid_argument("PagedCacheGroupConfig: sliding_window_tokens must be > 0 for sliding groups");
    }
}

PagedCacheGroupAllocator::PagedCacheGroupAllocator(PagedCacheGroupConfig config)
    : config_(std::move(config)), pool_(config_.RawTokensPerPage(), config_.total_pages) {
    config_.Validate();
}

OwnedPages PagedCacheGroupAllocator::AcquireOwned(std::int32_t num_pages) {
    if (num_pages <= 0) {
        return {};
    }
    OwnedPages owned = pool_.Allocate(num_pages);
    if (owned.Size() < num_pages) {
        ++failed_alloc_count_;
        return {};
    }
    allocated_pages_total_ += num_pages;
    return owned;
}

std::vector<std::int32_t> PagedCacheGroupAllocator::Allocate(std::int32_t num_pages) {
    OwnedPages owned = AcquireOwned(num_pages);
    if (owned.Empty()) {
        return {};
    }
    return owned.Detach();
}

void PagedCacheGroupAllocator::Deallocate(const std::vector<std::int32_t>& pages) {
    pool_.Deallocate(pages);
    released_pages_total_ += static_cast<std::int64_t>(pages.size());
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
    // Absolute pages already covered = base + live size after any
    // ReleaseSkipped compaction. Allocate only the delta.
    const std::int32_t pages_have = base_logical_page_ + Size();
    const std::int32_t pages_to_allocate = pages_needed - pages_have;
    if (pages_to_allocate > 0) {
        OwnedPages fresh = allocator_->AcquireOwned(pages_to_allocate);
        if (fresh.Size() < pages_to_allocate) {
            // fresh dtor returns any partial allocation to pool_.
            throw std::runtime_error("PagedCacheGroupTable::Acquire: failed to allocate pages for group " +
                                     cfg.group_id);
        }
        pages_.Append(std::move(fresh));
    }
    raw_token_cursor_ = target_raw_tokens_exclusive;
}

std::vector<std::int32_t> PagedCacheGroupTable::ReleaseSkipped(std::int32_t window_lower_bound) {
    if (allocator_ == nullptr || pages_.Empty() || window_lower_bound <= 0) {
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
    // Absolute logical-page index (exclusive) below which entries fall out of
    // the active window.
    const std::int32_t target = window_lower_bound / raw_per_page;
    if (target <= base_logical_page_) {
        return {};
    }
    const std::int32_t to_drop = std::min(target - base_logical_page_, Size());
    if (to_drop <= 0) {
        return {};
    }
    OwnedPages dropped = pages_.TakeFirst(to_drop);
    std::vector<std::int32_t> released = dropped.Ids();
    base_logical_page_ += to_drop;
    // dropped goes out of scope: OwnedPages dtor returns the pages to pool_.
    return released;
}

std::vector<std::int32_t> PagedCacheGroupTable::ReleaseAll() {
    OwnedPages dropped = pages_.TakeFirst(pages_.Size());
    std::vector<std::int32_t> released = dropped.Ids();
    raw_token_cursor_ = 0;
    base_logical_page_ = 0;
    return released;
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
    return allocator_ != nullptr && allocator_->Config().retention == PagedCacheGroupConfig::Retention::SlidingWindow;
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
