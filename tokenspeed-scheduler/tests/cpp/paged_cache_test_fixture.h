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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

#pragma once

// Shared fixture for HybridPrefixCache + two-group paged-cache tests.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/paged_cache_snapshot.h"
#include "resource/radix_tree/radix_tree.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "unit_test_helper.h"

namespace tokenspeed::test {

struct PagedCacheFixtureParams {
    std::int32_t page_size;
    std::int32_t device_pages;
    std::int32_t lcm_raw_tokens;
    std::int32_t sliding_window_tokens;
    std::int32_t fh_rows_per_page;
    std::int32_t fh_stride;
    std::int32_t swa_rows_per_page;
    std::int32_t swa_stride;
    std::int32_t group_total_pages;
};

template <PagedCacheFixtureParams kParams>
class PagedCacheTestFixtureT : public ::testing::Test {
protected:
    static constexpr std::int32_t kPageSize = kParams.page_size;
    static constexpr std::int32_t kDevicePages = kParams.device_pages;
    static constexpr std::int32_t kLcm = kParams.lcm_raw_tokens;
    static constexpr std::int32_t kSlidingWindow = kParams.sliding_window_tokens;

    void SetUp() override {
        device_alloc_ = std::make_unique<PageAllocator>(kPageSize, kDevicePages);
        kv_cache_ = std::make_unique<KVPrefixCache>(device_alloc_.get(), /*host=*/nullptr);

        auto fh_owner = std::make_unique<PagedCacheGroupAllocator>(MakeGroupConfig(
            "fh", kParams.fh_rows_per_page, kParams.fh_stride,
            PagedCacheGroupConfig::Retention::FullHistory, /*window=*/0,
            PagedCacheGroupFamily::History));
        auto swa_owner = std::make_unique<PagedCacheGroupAllocator>(MakeGroupConfig(
            "swa", kParams.swa_rows_per_page, kParams.swa_stride,
            PagedCacheGroupConfig::Retention::SlidingWindow, kSlidingWindow,
            PagedCacheGroupFamily::State));
        fh_alloc_ = fh_owner.get();
        swa_alloc_ = swa_owner.get();

        hybrid_ = std::make_unique<HybridPrefixCache>(*kv_cache_, /*mamba=*/nullptr,
                                                     /*mamba_chunk_size=*/0);
        hybrid_->RegisterPagedCacheGroup(std::move(fh_owner));
        hybrid_->RegisterPagedCacheGroup(std::move(swa_owner));
        std::unordered_map<std::string, std::int32_t> sliding{{"swa", kSlidingWindow}};
        hybrid_->EnablePagedCacheAdjunct(/*required=*/{"fh", "swa"}, std::move(sliding));
        kv_cache_->GetDeviceManager().SetEvictionCallback(
            [this](TreeNode* node) { hybrid_->OnKVEvict(node); });
    }

    // Insert pages from `start_node` (nullptr=root); returns terminal node.
    TreeNode* InsertDevicePages(std::int32_t num_pages, token_t token_start,
                                 TreeNode* start_node = nullptr) {
        auto tokens = MakeAlignedTokens(num_pages, kPageSize, token_start);
        OwnedPages pages = device_alloc_->Allocate(num_pages);
        auto res = kv_cache_->Insert<ResourceType::Device>(
            tokens, /*prefix_pages=*/{}, std::move(pages), /*page_hashes=*/{}, start_node);
        return res.last_node;
    }

    // Build a complete snapshot covering one LCM segment ending at prefix_len_tokens.
    std::unique_ptr<PagedCacheSnapshot> MakeCompleteSnapshot(
        std::int32_t prefix_len_tokens, std::int32_t swa_base_logical_page = 0) {
        auto snap = std::make_unique<PagedCacheSnapshot>();
        snap->prefix_len_tokens = prefix_len_tokens;
        snap->groups.emplace("fh", BuildGroupSnap(fh_alloc_, prefix_len_tokens,
                                                    /*base=*/0, /*sliding=*/false));
        snap->groups.emplace("swa", BuildGroupSnap(swa_alloc_, prefix_len_tokens,
                                                    swa_base_logical_page, /*sliding=*/true));
        return snap;
    }

    // History-only snapshot (state group omitted); used for fallback tests.
    std::unique_ptr<PagedCacheSnapshot> MakeHistoryOnlySnapshot(std::int32_t prefix_len_tokens) {
        auto snap = std::make_unique<PagedCacheSnapshot>();
        snap->prefix_len_tokens = prefix_len_tokens;
        snap->groups.emplace("fh", BuildGroupSnap(fh_alloc_, prefix_len_tokens,
                                                    /*base=*/0, /*sliding=*/false));
        return snap;
    }

    // Detach and reattach without the state group; re-attach recomputes
    // `complete_families` and leaves only History present.
    void DowngradeSnapshotToHistoryOnly(TreeNode* node) {
        auto snap = hybrid_->DetachPagedCacheSnapshotFromNode(node);
        ASSERT_NE(snap, nullptr);
        snap->groups.erase("swa");
        hybrid_->AttachPagedCacheSnapshotToNode(node, std::move(snap));
    }

    std::unique_ptr<PageAllocator> device_alloc_;
    std::unique_ptr<KVPrefixCache> kv_cache_;
    PagedCacheGroupAllocator* fh_alloc_{nullptr};
    PagedCacheGroupAllocator* swa_alloc_{nullptr};
    std::unique_ptr<HybridPrefixCache> hybrid_;

protected:
    static PagedCacheGroupConfig MakeGroupConfig(std::string group_id,
                                                  std::int32_t rows_per_page,
                                                  std::int32_t stride,
                                                  PagedCacheGroupConfig::Retention retention,
                                                  std::int32_t window,
                                                  PagedCacheGroupFamily family) {
        PagedCacheGroupConfig cfg{};
        cfg.group_id = std::move(group_id);
        cfg.rows_per_page = rows_per_page;
        cfg.entry_stride_tokens = stride;
        cfg.total_pages = kParams.group_total_pages;
        cfg.retention = retention;
        cfg.sliding_window_tokens = window;
        cfg.family = family;
        return cfg;
    }

private:

    PagedCacheGroupSnapshot BuildGroupSnap(PagedCacheGroupAllocator* alloc,
                                            std::int32_t prefix_len_tokens,
                                            std::int32_t base_logical_page, bool sliding) {
        PagedCacheGroupTable t{alloc};
        t.Acquire(kLcm);
        // Caller chooses absolute base; fresh table commits at 0.
        auto committed = sliding ? t.CheckpointStateToSnapshot(kLcm)
                                  : t.CommitHistoryToSnapshot(kLcm);
        PagedCacheGroupSnapshot g{};
        g.pages = std::move(committed.pages);
        g.base_logical_page = base_logical_page;
        g.raw_token_cursor = prefix_len_tokens;
        g.sliding = sliding;
        return g;
    }
};

inline constexpr PagedCacheFixtureParams kSmallFixtureParams{
    /*page_size=*/2,           /*device_pages=*/8,
    /*lcm_raw_tokens=*/4,      /*sliding_window_tokens=*/8,
    /*fh_rows_per_page=*/4,    /*fh_stride=*/1,
    /*swa_rows_per_page=*/2,   /*swa_stride=*/1,
    /*group_total_pages=*/16,
};
using PagedCacheSmallFixture = PagedCacheTestFixtureT<kSmallFixtureParams>;

inline constexpr PagedCacheFixtureParams kLargeFixtureParams{
    /*page_size=*/64,          /*device_pages=*/64,
    /*lcm_raw_tokens=*/256,    /*sliding_window_tokens=*/128,
    /*fh_rows_per_page=*/64,   /*fh_stride=*/4,
    /*swa_rows_per_page=*/64,  /*swa_stride=*/1,
    /*group_total_pages=*/32,
};
using PagedCacheLargeFixture = PagedCacheTestFixtureT<kLargeFixtureParams>;

// Wide-window variant: state window > history alignment so `segments_needed=2`.
inline constexpr PagedCacheFixtureParams kWideWindowFixtureParams{
    /*page_size=*/64,          /*device_pages=*/64,
    /*lcm_raw_tokens=*/256,    /*sliding_window_tokens=*/512,
    /*fh_rows_per_page=*/64,   /*fh_stride=*/4,
    /*swa_rows_per_page=*/64,  /*swa_stride=*/1,
    /*group_total_pages=*/64,
};
using PagedCacheWideWindowFixture = PagedCacheTestFixtureT<kWideWindowFixtureParams>;

}  // namespace tokenspeed::test
