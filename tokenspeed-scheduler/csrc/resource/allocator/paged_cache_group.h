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

// Positive-only ceiling division; returns 0 for non-positive numerators.
// Lives here because paged-cache admission/table math is its only caller.
inline std::int32_t CeilDivPositive(std::int32_t numer, std::int32_t denom) {
    if (numer <= 0) return 0;
    return (numer + denom - 1) / denom;
}

// Paged-cache group families for V4 prefix-cache reuse.
// History: every page on [0, P) required (chain).
// State: only the trailing window at the hit depth required.
enum class PagedCacheGroupFamily { History, State };

// Phase 1 only has kSnapshotRequired; kReplayTail / kCheckpointThenReplay
// reserved for Phase 2 RecoveryPlan work.
enum class StateRestorePolicy { kSnapshotRequired };

// One model-defined paged cache group; scheduler treats group_id as opaque.
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
    // History groups form a chain; State groups only need the trailing window.
    PagedCacheGroupFamily family{PagedCacheGroupFamily::History};

    std::int32_t RawTokensPerPage() const { return rows_per_page * entry_stride_tokens; }

    void Validate() const;
};

// Group-level allocator: wraps PageAllocator + config + counters. Releases run
// via OwnedPages RAII directly to the pool.
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
    // Only counts explicit Deallocate(); RAII releases bypass this counter.
    std::int64_t ReleasedPagesTotal() const { return released_pages_total_; }
    std::int64_t FailedAllocCount() const { return failed_alloc_count_; }

private:
    friend class PagedCacheGroupTable;

    // Empty OwnedPages on insufficient capacity (bumps failed_alloc_count_).
    OwnedPages AcquireOwned(std::int32_t num_pages);

    PagedCacheGroupConfig config_;
    PageAllocator pool_;
    std::int64_t allocated_pages_total_{0};
    std::int64_t released_pages_total_{0};
    std::int64_t failed_alloc_count_{0};
};

// One per request, per group. Two storage segments (no refcounts):
//   - `borrowed_page_ids_`: page ids only; physical ownership lives in a
//     TreeNode's PagedCacheSnapshot, pinned via the request's DeviceNodeRef.
//   - `owned_pages_`: RAII back to the allocator on release or moved to a
//     snapshot via CommitHistoryToSnapshot / CheckpointStateToSnapshot.
// PageIds() = borrowed ++ owned, where column c == absolute logical page
// BaseLogicalPage() + c. ReleaseSkipped peels expired front pages (sliding only).
class PagedCacheGroupTable {
public:
    PagedCacheGroupTable() = default;
    explicit PagedCacheGroupTable(PagedCacheGroupAllocator* allocator) : allocator_(allocator) {}
    ~PagedCacheGroupTable() = default;

    PagedCacheGroupTable(const PagedCacheGroupTable&) = delete;
    PagedCacheGroupTable& operator=(const PagedCacheGroupTable&) = delete;
    PagedCacheGroupTable(PagedCacheGroupTable&&) noexcept = default;
    PagedCacheGroupTable& operator=(PagedCacheGroupTable&&) noexcept = default;

    // Grow pages to cover [base*RawTokensPerPage, target_raw_tokens_exclusive).
    void Acquire(std::int32_t target_raw_tokens_exclusive);

    // segment_base_logical_page is captured BEFORE the commit cursor advances
    // (sliding ReleaseSkipped may have already moved BaseLogicalPage() forward).
    struct CommitResult {
        OwnedPages pages;
        std::int32_t segment_base_logical_page{0};
    };

    // History append: move owned [committed, target) out and mirror ids to
    // borrowed_page_ids_. Throws for non-History family groups.
    CommitResult CommitHistoryToSnapshot(std::int32_t target_raw_tokens);

    // State checkpoint: snapshot the live trailing window [max(0,target-W),
    // target); drop stale prefix from both owned (back to pool) and borrowed
    // (index drop only; physical pages live on earlier snapshots). Throws for
    // non-State family groups or when sliding_window_tokens is missing/non-positive.
    CommitResult CheckpointStateToSnapshot(std::int32_t target_raw_tokens);

    // Adopt borrowed page ids from a prefix-cache hit on a fresh-empty table.
    // Throws std::logic_error if called after Acquire/Import/Commit.
    void ImportPrefixBorrowed(std::vector<std::int32_t> ids,
                              std::int32_t base_logical_page,
                              std::int32_t raw_tokens_covered);

    // Sliding-only: drop front pages strictly below `window_lower_bound`.
    // Bumps base_logical_page_; commit cursor untouched. Idempotent.
    std::vector<std::int32_t> ReleaseSkipped(std::int32_t window_lower_bound);

    // Release everything; owned via RAII, borrowed by clearing. Used by finish/abort/retract.
    std::vector<std::int32_t> ReleaseAll();

    // Compact: PageIds()[c] = absolute logical page BaseLogicalPage() + c.
    const std::vector<std::int32_t>& PageIds() const { return page_ids_view_; }
    std::int32_t Size() const {
        return static_cast<std::int32_t>(borrowed_page_ids_.size()) + owned_pages_.Size();
    }
    std::int32_t ActivePagesCount() const { return Size(); }
    std::int32_t OwnedPagesCount() const { return owned_pages_.Size(); }
    std::int32_t BorrowedPagesCount() const {
        return static_cast<std::int32_t>(borrowed_page_ids_.size());
    }
    std::int32_t ReleasedPagesCount() const { return base_logical_page_; }
    std::int32_t BaseLogicalPage() const { return base_logical_page_; }
    std::int32_t RawTokenCursor() const { return raw_token_cursor_; }

    // Independent of base_logical_page_; sliding ReleaseSkipped does not move this.
    std::int32_t CommittedPrefixLenTokens() const { return committed_prefix_len_tokens_; }

    bool IsEmpty() const { return allocator_ == nullptr || Size() == 0; }
    std::int32_t RowsPerPage() const;
    std::int32_t EntryStrideTokens() const;
    std::int32_t RawTokensPerPage() const;
    bool IsSliding() const;
    std::int32_t SlidingWindowTokens() const;

private:
    // Must be called after every mutation of borrowed_page_ids_ or owned_pages_.
    void RefreshPageIdsView();

    PagedCacheGroupAllocator* allocator_{nullptr};
    OwnedPages owned_pages_;
    std::vector<std::int32_t> borrowed_page_ids_;
    std::int32_t raw_token_cursor_{0};
    std::int32_t base_logical_page_{0};
    std::int32_t committed_prefix_len_tokens_{0};
    // Cached borrowed ++ owned, exposed by PageIds() as const ref for ABI shape.
    std::vector<std::int32_t> page_ids_view_;
};

}  // namespace tokenspeed
