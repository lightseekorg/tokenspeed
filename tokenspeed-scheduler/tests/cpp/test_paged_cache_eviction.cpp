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

#include <string>

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

    HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, attach_a, MakeCompleteSnapshot(kLcm));
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

TEST_F(PagedCacheEvictionTest, StatePressurePrunesOnlyStateWhenHistoryHasCapacity) {
    TreeNode* terminal = InsertDevicePages(/*num_pages=*/2, /*token_start=*/1);
    ASSERT_NE(terminal, nullptr);

    TreeNode* attach = kv_cache_->GetRadixTree().SplitAt(terminal, kLcm);
    ASSERT_NE(attach, nullptr);
    HybridPrefixCacheTestPeer::AttachPagedCacheSnapshotToNode(*hybrid_, attach, MakeCompleteSnapshot(kLcm));
    ASSERT_TRUE(attach->HasPagedCacheSnapshot());
    ASSERT_TRUE(attach->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::History));
    ASSERT_TRUE(attach->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::State));

    const std::int32_t fh_available_before_admit = fh_alloc_->AvailablePages();
    ASSERT_GE(fh_available_before_admit, 1);

    std::vector<std::int32_t> state_saturator = swa_alloc_->Allocate(swa_alloc_->AvailablePages());
    ASSERT_FALSE(state_saturator.empty());
    ASSERT_EQ(swa_alloc_->AvailablePages(), 0);

    auto simulated_free = hybrid_->InitialSimulatedFree();
    MatchResult match = kv_cache_->Match(token_vec_t{});
    const std::string request_id = "state-pressure";
    auto result = hybrid_->Apply(
        cache::admit::PrefillFirstChunk{
            .request_id = request_id,
            .match_result = match,
            .device_pages_needed = 0,
            .tokens_this_round = kLcm,
            .first_raw_position_of_op = 0,
            .target_raw_tokens_exclusive = kLcm,
        },
        simulated_free);

    ASSERT_TRUE(result.admitted);
    ASSERT_TRUE(attach->HasPagedCacheSnapshot())
        << "History had enough free pages; State pressure must not full-prune the snapshot";
    const auto* snap = attach->GetPagedCacheSnapshot();
    ASSERT_NE(snap, nullptr);
    EXPECT_TRUE(snap->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_FALSE(snap->IsCompleteFor(PagedCacheGroupFamily::State));
    EXPECT_NE(snap->groups.find("fh"), snap->groups.end());
    EXPECT_EQ(snap->groups.find("swa"), snap->groups.end());

    HybridPrefixCacheTestPeer::ReleaseRequest(*hybrid_, "state-pressure");
    swa_alloc_->Deallocate(state_saturator);
}

}  // namespace tokenspeed::test
