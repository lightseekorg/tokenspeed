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

// Coverage: passive snapshot detach on KV LRU eviction returns pages via RAII.

#include "paged_cache_test_fixture.h"

namespace tokenspeed::test {

using PagedCacheEvictionTest = PagedCacheSmallFixture;

// Two branches, evict A with snapshot attached; pages must return via RAII.
TEST_F(PagedCacheEvictionTest, PassiveEvictionReleasesPagedCachePages) {
    InsertDevicePages(/*num_pages=*/2, /*token_start=*/1);                       // branch A
    auto* leaf_b = InsertDevicePages(/*num_pages=*/2, /*token_start=*/100);      // branch B
    ASSERT_NE(leaf_b, nullptr);

    TreeNode* attach_a = kv_cache_->GetRadixTree().SplitAt(
        kv_cache_->Match(MakeAlignedTokens(2, kPageSize, /*start=*/1)).device.last_node, kLcm);
    ASSERT_NE(attach_a, nullptr);

    const std::int32_t fh_before = fh_alloc_->AvailablePages();
    const std::int32_t swa_before = swa_alloc_->AvailablePages();
    hybrid_->AttachPagedCacheSnapshotToNode(attach_a, MakeCompleteSnapshot(kLcm));
    // 1 fh + 2 swa pages.
    EXPECT_EQ(fh_alloc_->AvailablePages(), fh_before - 1);
    EXPECT_EQ(swa_alloc_->AvailablePages(), swa_before - 2);
    EXPECT_TRUE(attach_a->HasPagedCacheSnapshot());

    // Pin branch B so eviction targets A.
    auto match_b = kv_cache_->Match(MakeAlignedTokens(2, kPageSize, /*start=*/100));
    DeviceNodeRef ref_b{match_b.device.last_node};

    const std::int32_t target_available = kDevicePages - 2;
    const bool ok = kv_cache_->EnsureCapacityByEvict<ResourceType::Device>(target_available);
    EXPECT_TRUE(ok);

    // Snapshot pages returned to pools.
    EXPECT_EQ(fh_alloc_->AvailablePages(), fh_before);
    EXPECT_EQ(swa_alloc_->AvailablePages(), swa_before);
}

}  // namespace tokenspeed::test
