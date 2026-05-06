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
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tokenspeed {

// One model-defined paged cache group. The scheduler treats group_id as opaque:
// V4 uses ids like "compressed.4" and "v4.swa".
struct PagedCacheGroupConfig {
    enum class Retention {
        FullHistory,
        SlidingWindow,
    };

    std::string group_id;
    std::int32_t rows_per_page{};
    std::int32_t entry_stride_tokens{};
    std::int32_t total_pages{};
    Retention retention{Retention::FullHistory};
    std::optional<std::int32_t> sliding_window_tokens{};

    std::int32_t RawTokensPerPage() const {
        return rows_per_page * entry_stride_tokens;
    }

    void Validate() const;
};

class PagedCacheGroupAllocator {
public:
    explicit PagedCacheGroupAllocator(PagedCacheGroupConfig config);

    PagedCacheGroupAllocator(const PagedCacheGroupAllocator&) = delete;
    PagedCacheGroupAllocator& operator=(const PagedCacheGroupAllocator&) = delete;
    PagedCacheGroupAllocator(PagedCacheGroupAllocator&&) = delete;
    PagedCacheGroupAllocator& operator=(PagedCacheGroupAllocator&&) = delete;

    std::vector<std::int32_t> Allocate(std::int32_t num_pages);
    void Deallocate(const std::vector<std::int32_t>& pages);

    const PagedCacheGroupConfig& Config() const { return config_; }
    std::int32_t TotalPages() const { return config_.total_pages; }
    std::int32_t AvailablePages() const { return static_cast<std::int32_t>(free_pages_.size()); }
    std::int64_t AllocatedPagesTotal() const { return allocated_pages_total_; }
    std::int64_t ReleasedPagesTotal() const { return released_pages_total_; }
    std::int64_t FailedAllocCount() const { return failed_alloc_count_; }

private:
    PagedCacheGroupConfig config_;
    std::vector<std::int32_t> free_pages_;
    std::int64_t allocated_pages_total_{0};
    std::int64_t released_pages_total_{0};
    std::int64_t failed_alloc_count_{0};
};

class PagedCacheGroupTable {
public:
    PagedCacheGroupTable() = default;
    explicit PagedCacheGroupTable(PagedCacheGroupAllocator* allocator) : allocator_(allocator) {}
    ~PagedCacheGroupTable();

    PagedCacheGroupTable(const PagedCacheGroupTable&) = delete;
    PagedCacheGroupTable& operator=(const PagedCacheGroupTable&) = delete;

    PagedCacheGroupTable(PagedCacheGroupTable&& other) noexcept
        : allocator_(std::exchange(other.allocator_, nullptr)),
          page_ids_(std::move(other.page_ids_)),
          raw_token_cursor_(std::exchange(other.raw_token_cursor_, 0)),
          released_pages_count_(std::exchange(other.released_pages_count_, 0)) {}

    PagedCacheGroupTable& operator=(PagedCacheGroupTable&& other) noexcept;

    void Acquire(std::int32_t target_raw_tokens_exclusive);
    std::vector<std::int32_t> ReleaseSkipped(std::int32_t window_lower_bound);
    std::vector<std::int32_t> ReleaseAll();

    const std::vector<std::int32_t>& PageIds() const { return page_ids_; }
    std::int32_t Size() const { return static_cast<std::int32_t>(page_ids_.size()); }
    std::int32_t ActivePagesCount() const { return Size() - released_pages_count_; }
    std::int32_t ReleasedPagesCount() const { return released_pages_count_; }
    std::int32_t RawTokenCursor() const { return raw_token_cursor_; }

    bool IsEmpty() const { return allocator_ == nullptr || page_ids_.empty(); }
    std::int32_t RowsPerPage() const;
    std::int32_t EntryStrideTokens() const;
    std::int32_t RawTokensPerPage() const;
    bool IsSliding() const;
    std::int32_t SlidingWindowTokens() const;

private:
    PagedCacheGroupAllocator* allocator_{nullptr};
    std::vector<std::int32_t> page_ids_;
    std::int32_t raw_token_cursor_{0};
    std::int32_t released_pages_count_{0};
};

}  // namespace tokenspeed
