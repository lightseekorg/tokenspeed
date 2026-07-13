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
//
// The contract under test is purely observable: after the snapshot-bearing KV
// node is evicted, every paged-cache page the snapshot was holding must be
// back in its group allocator's free list. We deliberately avoid hard-coding
// "how many pages a snapshot consumes" (that depends on commit-time semantics
// such as State-window trim) — instead, we capture the allocator state right
// before attach and require it to be restored after eviction.
TEST_F(PagedCacheEvictionTest, PassiveEvictionReleasesPagedCachePages) {
    InsertDevicePages(/*num_pages=*/2, /*token_start=*/1);                   // branch A
    auto* leaf_b = InsertDevicePages(/*num_pages=*/2, /*token_start=*/100);  // branch B
    ASSERT_NE(leaf_b, nullptr);

    TreeNode* attach_a = kv_cache_->GetRadixTree().SplitAt(
        kv_cache_->Match(MakeAlignedTokens(2, kPageSize, /*start=*/1)).device.last_node, kLcm);
    ASSERT_NE(attach_a, nullptr);

    // Baseline: paged-cache pools fully free, no snapshot attached.
    const std::int32_t fh_before = fh_alloc_->AvailablePages();
    const std::int32_t swa_before = swa_alloc_->AvailablePages();

    hybrid_->AttachPagedCacheSnapshotToNode(attach_a, MakeCompleteSnapshot(kLcm));
    EXPECT_TRUE(attach_a->HasPagedCacheSnapshot());
    // The snapshot must hold *some* pages from each group, otherwise the test
    // below ("eviction returns them") is vacuous. We do NOT assert the exact
    // count — that depends on snapshot-build semantics (e.g. State-window
    // trim) and is covered by dedicated build/commit tests.
    EXPECT_LT(fh_alloc_->AvailablePages(), fh_before);
    EXPECT_LT(swa_alloc_->AvailablePages(), swa_before);

    // Pin branch B so eviction targets A. Without this lock the LRU policy
    // could evict either branch.
    auto match_b = kv_cache_->Match(MakeAlignedTokens(2, kPageSize, /*start=*/100));
    DeviceNodeRef ref_b{match_b.device.last_node};

    // Force eviction of branch A by demanding one more page than the device
    // allocator currently has free. Branch B's 2 device pages are pinned by
    // `ref_b`, so the LRU must drop branch A — which carries our snapshot.
    const std::int32_t target_available = device_alloc_->AvailablePages() + 1;
    const bool ok = kv_cache_->EnsureCapacityByEvict<ResourceType::Device>(target_available);
    EXPECT_TRUE(ok);
    // Note: after eviction `attach_a` may be freed by tree pruning, so we do
    // not dereference it. The observable proof that OnKVEvict detached the
    // snapshot is that the paged-cache allocator pools are restored below.

    // Observable contract: every paged-cache page the snapshot held is now
    // back in its allocator's free list (OwnedPages RAII via OnKVEvict ->
    // DetachPagedCacheSnapshotFromNode).
    EXPECT_EQ(fh_alloc_->AvailablePages(), fh_before);
    EXPECT_EQ(swa_alloc_->AvailablePages(), swa_before);
}

TEST_F(PagedCacheEvictionTest, BorrowedSnapshotSurvivesDeviceEviction) {
    InsertDevicePages(/*num_pages=*/2, /*token_start=*/1);                   // branch A
    auto* leaf_b = InsertDevicePages(/*num_pages=*/2, /*token_start=*/100);  // branch B
    ASSERT_NE(leaf_b, nullptr);

    TreeNode* attach_a = kv_cache_->GetRadixTree().SplitAt(
        kv_cache_->Match(MakeAlignedTokens(2, kPageSize, /*start=*/1)).device.last_node, kLcm);
    ASSERT_NE(attach_a, nullptr);

    PageAllocator host_alloc{kPageSize, /*total_pages=*/8};
    const auto host_pages = static_cast<std::int32_t>(attach_a->Tokens().size()) / kPageSize;
    attach_a->AttachResource(std::make_unique<NodeResource<ResourceType::Host>>(host_alloc.Allocate(host_pages)));
    auto host_ref = std::make_unique<HostNodeRef>(attach_a);

    const std::int32_t fh_before = fh_alloc_->AvailablePages();
    const std::int32_t swa_before = swa_alloc_->AvailablePages();
    hybrid_->AttachPagedCacheSnapshotToNode(attach_a, MakeCompleteSnapshot(kLcm));
    const std::int32_t fh_pinned = fh_alloc_->AvailablePages();
    const std::int32_t swa_pinned = swa_alloc_->AvailablePages();
    ASSERT_LT(fh_pinned, fh_before);
    ASSERT_LT(swa_pinned, swa_before);

    auto match_a = hybrid_->Match(MakeAlignedTokens(2, kPageSize, /*start=*/1));
    ASSERT_EQ(match_a.paged_cache.prefix_len_tokens, kLcm);
    hybrid_->AcquireForRequest("retracted", /*first_raw_position_of_op=*/0,
                               /*target_raw_tokens_exclusive=*/kLcm, match_a.paged_cache);

    auto match_b = kv_cache_->Match(MakeAlignedTokens(2, kPageSize, /*start=*/100));
    DeviceNodeRef ref_b{match_b.device.last_node};
    const std::int32_t target_available = device_alloc_->AvailablePages() + 1;
    ASSERT_TRUE(kv_cache_->EnsureCapacityByEvict<ResourceType::Device>(target_available));

    // Device eviction must not free side-cache pages still borrowed by a
    // retracted/loadback request's retained table.
    EXPECT_FALSE(attach_a->OnDevice());
    EXPECT_TRUE(attach_a->HasPagedCacheSnapshot());
    EXPECT_EQ(fh_alloc_->AvailablePages(), fh_pinned);
    EXPECT_EQ(swa_alloc_->AvailablePages(), swa_pinned);
    EXPECT_THROW(hybrid_->OnNodeDestroyed(attach_a), std::runtime_error);
    EXPECT_TRUE(attach_a->HasPagedCacheSnapshot());

    // Once the request table drops the borrow, the normal eviction callback
    // can reclaim the snapshot and return its pages.
    hybrid_->ReleaseRequest("retracted");
    hybrid_->OnKVEvict(attach_a);
    EXPECT_FALSE(attach_a->HasPagedCacheSnapshot());
    EXPECT_EQ(fh_alloc_->AvailablePages(), fh_before);
    EXPECT_EQ(swa_alloc_->AvailablePages(), swa_before);

    // Destroy the manually attached host resource before its local allocator.
    host_ref.reset();
    auto host_resource = attach_a->DetachResource<ResourceType::Host>();
    ASSERT_NE(host_resource, nullptr);
}

}  // namespace tokenspeed::test
