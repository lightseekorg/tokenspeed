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

#include "resource/types.h"

namespace tokenspeed {

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

void PagedCacheGroupTable::RefreshPageIdsView() {
    page_ids_view_.clear();
    page_ids_view_.reserve(borrowed_page_ids_.size() + static_cast<std::size_t>(owned_pages_.Size()));
    page_ids_view_.insert(page_ids_view_.end(), borrowed_page_ids_.begin(), borrowed_page_ids_.end());
    const auto& owned_ids = owned_pages_.Ids();
    page_ids_view_.insert(page_ids_view_.end(), owned_ids.begin(), owned_ids.end());
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
    // Absolute pages have = base + borrowed + owned; allocate only the delta.
    const std::int32_t pages_have =
        base_logical_page_ + static_cast<std::int32_t>(borrowed_page_ids_.size()) + owned_pages_.Size();
    const std::int32_t pages_to_allocate = pages_needed - pages_have;
    if (pages_to_allocate > 0) {
        OwnedPages fresh = allocator_->AcquireOwned(pages_to_allocate);
        if (fresh.Size() < pages_to_allocate) {
            throw std::runtime_error("PagedCacheGroupTable::Acquire: failed to allocate pages for group " +
                                     cfg.group_id);
        }
        owned_pages_.Append(std::move(fresh));
        RefreshPageIdsView();
    }
    raw_token_cursor_ = target_raw_tokens_exclusive;
}

PagedCacheGroupTable::CommitResult PagedCacheGroupTable::CommitHistoryToSnapshot(
    std::int32_t target_raw_tokens) {
    if (allocator_ == nullptr) {
        throw std::logic_error("PagedCacheGroupTable::CommitHistoryToSnapshot: no allocator bound");
    }
    const auto& cfg = allocator_->Config();
    if (cfg.family != PagedCacheGroupFamily::History) {
        throw std::logic_error(
            "PagedCacheGroupTable::CommitHistoryToSnapshot: requires History family; group=" +
            cfg.group_id);
    }
    if (target_raw_tokens <= committed_prefix_len_tokens_) {
        return {};
    }
    if (target_raw_tokens > raw_token_cursor_) {
        throw std::invalid_argument(
            "PagedCacheGroupTable::CommitHistoryToSnapshot: target exceeds raw_token_cursor; target=" +
            std::to_string(target_raw_tokens) + "; cursor=" + std::to_string(raw_token_cursor_));
    }
    const std::int32_t raw_per_page = cfg.RawTokensPerPage();
    if (raw_per_page <= 0) {
        throw std::logic_error(
            "PagedCacheGroupTable::CommitHistoryToSnapshot: invalid group config (raw_per_page <= 0)");
    }
    if (committed_prefix_len_tokens_ % raw_per_page != 0) {
        throw std::logic_error(
            "PagedCacheGroupTable::CommitHistoryToSnapshot: committed cursor not page-aligned; committed=" +
            std::to_string(committed_prefix_len_tokens_) + "; raw_per_page=" + std::to_string(raw_per_page));
    }
    if (target_raw_tokens % raw_per_page != 0) {
        throw std::invalid_argument(
            "PagedCacheGroupTable::CommitHistoryToSnapshot: target not page-aligned; target=" +
            std::to_string(target_raw_tokens) + "; raw_per_page=" + std::to_string(raw_per_page));
    }

    const std::int32_t pages_to_commit =
        (target_raw_tokens - committed_prefix_len_tokens_) / raw_per_page;
    if (pages_to_commit <= 0) {
        committed_prefix_len_tokens_ = target_raw_tokens;
        RefreshPageIdsView();
        return {};
    }
    if (pages_to_commit > owned_pages_.Size()) {
        throw std::logic_error(
            "PagedCacheGroupTable::CommitHistoryToSnapshot: not enough owned pages; want=" +
            std::to_string(pages_to_commit) + "; have_owned=" + std::to_string(owned_pages_.Size()));
    }

    const std::int32_t segment_base_logical_page = committed_prefix_len_tokens_ / raw_per_page;
    OwnedPages segment = owned_pages_.TakeFirst(pages_to_commit);
    const auto& seg_ids = segment.Ids();
    borrowed_page_ids_.insert(borrowed_page_ids_.end(), seg_ids.begin(), seg_ids.end());
    committed_prefix_len_tokens_ = target_raw_tokens;
    RefreshPageIdsView();
    return CommitResult{std::move(segment), segment_base_logical_page};
}

PagedCacheGroupTable::CommitResult PagedCacheGroupTable::CheckpointStateToSnapshot(
    std::int32_t target_raw_tokens) {
    if (allocator_ == nullptr) {
        throw std::logic_error("PagedCacheGroupTable::CheckpointStateToSnapshot: no allocator bound");
    }
    const auto& cfg = allocator_->Config();
    if (cfg.family != PagedCacheGroupFamily::State) {
        throw std::logic_error(
            "PagedCacheGroupTable::CheckpointStateToSnapshot: requires State family; group=" +
            cfg.group_id);
    }
    if (!cfg.sliding_window_tokens.has_value() || *cfg.sliding_window_tokens <= 0) {
        throw std::logic_error(
            "PagedCacheGroupTable::CheckpointStateToSnapshot: State family requires positive"
            " sliding_window_tokens; group=" + cfg.group_id);
    }
    if (target_raw_tokens <= committed_prefix_len_tokens_) {
        return {};
    }
    if (target_raw_tokens > raw_token_cursor_) {
        throw std::invalid_argument(
            "PagedCacheGroupTable::CheckpointStateToSnapshot: target exceeds raw_token_cursor; target=" +
            std::to_string(target_raw_tokens) + "; cursor=" + std::to_string(raw_token_cursor_));
    }
    const std::int32_t raw_per_page = cfg.RawTokensPerPage();
    if (raw_per_page <= 0) {
        throw std::logic_error(
            "PagedCacheGroupTable::CheckpointStateToSnapshot: invalid group config (raw_per_page <= 0)");
    }
    if (target_raw_tokens % raw_per_page != 0) {
        throw std::invalid_argument(
            "PagedCacheGroupTable::CheckpointStateToSnapshot: target not page-aligned; target=" +
            std::to_string(target_raw_tokens) + "; raw_per_page=" + std::to_string(raw_per_page));
    }

    const std::int32_t window = *cfg.sliding_window_tokens;
    const std::int32_t live_lower_raw = std::max(0, target_raw_tokens - window);
    const std::int32_t live_lower_page = live_lower_raw / raw_per_page;
    const std::int32_t target_page = target_raw_tokens / raw_per_page;
    const std::int32_t window_pages = target_page - live_lower_page;

    // Drop stale borrowed entries: their physical pages live on earlier snapshots,
    // so the table just discards the index. Mirrors ReleaseSkipped semantics.
    if (live_lower_page > base_logical_page_) {
        const std::int32_t borrowed_to_drop = std::min<std::int32_t>(
            live_lower_page - base_logical_page_,
            static_cast<std::int32_t>(borrowed_page_ids_.size()));
        if (borrowed_to_drop > 0) {
            borrowed_page_ids_.erase(borrowed_page_ids_.begin(),
                                     borrowed_page_ids_.begin() + borrowed_to_drop);
            base_logical_page_ += borrowed_to_drop;
        }
    }

    // Drop stale owned-prefix pages (below live_lower) back to the pool via RAII.
    if (live_lower_page > base_logical_page_) {
        const std::int32_t owned_dead_pages = std::min<std::int32_t>(
            live_lower_page - base_logical_page_, owned_pages_.Size());
        if (owned_dead_pages > 0) {
            OwnedPages dropped = owned_pages_.TakeFirst(owned_dead_pages);
            base_logical_page_ += owned_dead_pages;
            // dropped dtor returns pages to pool.
        }
    }

    // Live window slice: take from owned, append ids to borrowed as new tail,
    // also hand it out via CommitResult for the snapshot.
    OwnedPages segment{};
    std::int32_t segment_base_logical_page = live_lower_page;
    if (window_pages > 0) {
        if (window_pages > owned_pages_.Size()) {
            throw std::logic_error(
                "PagedCacheGroupTable::CheckpointStateToSnapshot: not enough owned pages for window;"
                " want=" + std::to_string(window_pages) +
                "; have_owned=" + std::to_string(owned_pages_.Size()));
        }
        segment = owned_pages_.TakeFirst(window_pages);
        const auto& seg_ids = segment.Ids();
        borrowed_page_ids_.insert(borrowed_page_ids_.end(), seg_ids.begin(), seg_ids.end());
    }

    committed_prefix_len_tokens_ = target_raw_tokens;
    RefreshPageIdsView();
    return CommitResult{std::move(segment), segment_base_logical_page};
}

void PagedCacheGroupTable::ImportPrefixBorrowed(std::vector<std::int32_t> ids,
                                                std::int32_t base_logical_page,
                                                std::int32_t raw_tokens_covered) {
    if (allocator_ == nullptr) {
        throw std::logic_error("PagedCacheGroupTable::ImportPrefixBorrowed: no allocator bound");
    }
    if (!(borrowed_page_ids_.empty() && owned_pages_.Empty() && raw_token_cursor_ == 0 &&
          base_logical_page_ == 0 && committed_prefix_len_tokens_ == 0)) {
        throw std::logic_error(
            "PagedCacheGroupTable::ImportPrefixBorrowed: only legal on a fresh-empty table");
    }
    if (base_logical_page < 0) {
        throw std::invalid_argument("PagedCacheGroupTable::ImportPrefixBorrowed: base_logical_page must be >= 0");
    }
    if (raw_tokens_covered < 0) {
        throw std::invalid_argument(
            "PagedCacheGroupTable::ImportPrefixBorrowed: raw_tokens_covered must be >= 0");
    }
    borrowed_page_ids_ = std::move(ids);
    base_logical_page_ = base_logical_page;
    raw_token_cursor_ = raw_tokens_covered;
    committed_prefix_len_tokens_ = raw_tokens_covered;
    RefreshPageIdsView();
}

std::vector<std::int32_t> PagedCacheGroupTable::ReleaseSkipped(std::int32_t window_lower_bound) {
    if (allocator_ == nullptr || Size() == 0 || window_lower_bound <= 0) {
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
    if (target <= base_logical_page_) {
        return {};
    }
    const std::int32_t to_drop = std::min(target - base_logical_page_, Size());
    if (to_drop <= 0) {
        return {};
    }
    std::vector<std::int32_t> released;
    released.reserve(static_cast<std::size_t>(to_drop));

    // Drop from FRONT: borrowed first, then owned.
    const std::int32_t borrowed_drop =
        std::min(to_drop, static_cast<std::int32_t>(borrowed_page_ids_.size()));
    if (borrowed_drop > 0) {
        released.insert(released.end(), borrowed_page_ids_.begin(),
                        borrowed_page_ids_.begin() + borrowed_drop);
        borrowed_page_ids_.erase(borrowed_page_ids_.begin(), borrowed_page_ids_.begin() + borrowed_drop);
        // Borrowed pages stay owned by their TreeNode snapshot; only shrink index.
    }
    const std::int32_t owned_drop = to_drop - borrowed_drop;
    if (owned_drop > 0) {
        OwnedPages dropped = owned_pages_.TakeFirst(owned_drop);
        const auto& dropped_ids = dropped.Ids();
        released.insert(released.end(), dropped_ids.begin(), dropped_ids.end());
        // dropped dtor returns pages to pool.
    }
    base_logical_page_ += to_drop;
    RefreshPageIdsView();
    // committed_prefix_len_tokens_ intentionally untouched (logical vs physical).
    return released;
}

std::vector<std::int32_t> PagedCacheGroupTable::ReleaseAll() {
    std::vector<std::int32_t> released;
    released.reserve(borrowed_page_ids_.size() + static_cast<std::size_t>(owned_pages_.Size()));
    released.insert(released.end(), borrowed_page_ids_.begin(), borrowed_page_ids_.end());
    borrowed_page_ids_.clear();
    // Borrowed pages stay owned by their snapshots.

    OwnedPages dropped = owned_pages_.TakeFirst(owned_pages_.Size());
    const auto& dropped_ids = dropped.Ids();
    released.insert(released.end(), dropped_ids.begin(), dropped_ids.end());
    // dropped dtor returns pages to pool.

    raw_token_cursor_ = 0;
    base_logical_page_ = 0;
    committed_prefix_len_tokens_ = 0;
    RefreshPageIdsView();
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
