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
#include <vector>

#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"

namespace tokenspeed {

// One model-defined paged cache group. The scheduler treats group_id as opaque:
// V4 uses ids like "v4.c4a.compressed_kv" and "v4.swa_kv".
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

    std::int32_t RawTokensPerPage() const { return rows_per_page * entry_stride_tokens; }

    void Validate() const;
};

// Group-level allocator. Composes a PageAllocator that owns the free list, and
// adds group config + cumulative counters. PagedCacheGroupTable acquires pages
// through this allocator and stores them as OwnedPages bound to the inner
// pool, so all release paths go directly to the pool via RAII.
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
    std::int32_t TotalPages() const { return pool_.TotalPages(); }
    std::int32_t AvailablePages() const { return pool_.AvailablePages(); }

    std::int64_t AllocatedPagesTotal() const { return allocated_pages_total_; }
    // Counts pages explicitly returned via Deallocate(). Pages released through
    // PagedCacheGroupTable RAII (destructor / ReleaseSkipped / ReleaseAll) go
    // directly to the inner pool and are not counted here.
    std::int64_t ReleasedPagesTotal() const { return released_pages_total_; }
    std::int64_t FailedAllocCount() const { return failed_alloc_count_; }

private:
    friend class PagedCacheGroupTable;

    // Allocate a fresh batch as OwnedPages bound to pool_. Bumps stats; returns
    // an empty OwnedPages on insufficient capacity (and bumps failed counter).
    OwnedPages AcquireOwned(std::int32_t num_pages);

    PagedCacheGroupConfig config_;
    PageAllocator pool_;
    std::int64_t allocated_pages_total_{0};
    std::int64_t released_pages_total_{0};
    std::int64_t failed_alloc_count_{0};
};

// One per request, per group. Stores live pages as OwnedPages so destruction
// (and any TakeFirst-based release) automatically returns them to the pool.
// Acquire grows the cursor; ReleaseSkipped peels expired pages off the front
// (sliding-window groups only) and bumps the base logical page index so
// PageIds() always exposes only live entries; absolute logical page for
// column c is BaseLogicalPage() + c.
class PagedCacheGroupTable {
public:
    PagedCacheGroupTable() = default;
    explicit PagedCacheGroupTable(PagedCacheGroupAllocator* allocator) : allocator_(allocator) {}
    ~PagedCacheGroupTable() = default;

    PagedCacheGroupTable(const PagedCacheGroupTable&) = delete;
    PagedCacheGroupTable& operator=(const PagedCacheGroupTable&) = delete;
    PagedCacheGroupTable(PagedCacheGroupTable&&) noexcept = default;
    PagedCacheGroupTable& operator=(PagedCacheGroupTable&&) noexcept = default;

    void Acquire(std::int32_t target_raw_tokens_exclusive);

    // Returns physical ids of pages whose covered raw range is strictly below
    // `window_lower_bound`. Compacts the in-memory table and bumps the base
    // logical page so PageIds() always contains only live entries. Idempotent.
    // No-op (returns empty) for full-history groups.
    std::vector<std::int32_t> ReleaseSkipped(std::int32_t window_lower_bound);

    // Returns all live physical ids and clears the table. Used by
    // finish/abort/retract.
    std::vector<std::int32_t> ReleaseAll();

    // Compact view: column c here represents absolute logical page
    // BaseLogicalPage() + c. Released-from-front pages are NOT present.
    const std::vector<std::int32_t>& PageIds() const { return pages_.Ids(); }
    std::int32_t Size() const { return pages_.Size(); }
    std::int32_t ActivePagesCount() const { return Size(); }
    std::int32_t ReleasedPagesCount() const { return base_logical_page_; }
    std::int32_t BaseLogicalPage() const { return base_logical_page_; }
    std::int32_t RawTokenCursor() const { return raw_token_cursor_; }

    bool IsEmpty() const { return allocator_ == nullptr || pages_.Empty(); }
    std::int32_t RowsPerPage() const;
    std::int32_t EntryStrideTokens() const;
    std::int32_t RawTokensPerPage() const;
    bool IsSliding() const;
    std::int32_t SlidingWindowTokens() const;

private:
    PagedCacheGroupAllocator* allocator_{nullptr};
    OwnedPages pages_;
    std::int32_t raw_token_cursor_{0};
    // Absolute logical-page index of pages_.Ids()[0]. Bumped by ReleaseSkipped.
    std::int32_t base_logical_page_{0};
};

}  // namespace tokenspeed
