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

#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "unit_test_helper.h"
#include "resource/allocator/page_allocator.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"

namespace tokenspeed::test {

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class EvictionLRUTest : public ::testing::Test {
protected:
    static constexpr int32_t kPageSize = 4;
    static constexpr int32_t kTotalPages = 64;

    void SetUp() override {
        device_alloc_ = std::make_unique<PageAllocator>(kPageSize, kTotalPages);
        cache_ = std::make_unique<KVPrefixCache>(device_alloc_.get(), /*host=*/nullptr);
    }

    // Insert a sequence of `num_pages` pages starting with token `start`.
    InsertResult Insert(int32_t num_pages, token_t start = 1, TreeNode* from = nullptr) {
        auto tokens = MakeAlignedTokens(num_pages, kPageSize, start);
        auto pages = device_alloc_->Allocate(num_pages);
        return cache_->Insert<ResourceType::Device>(tokens, /*prefix_pages=*/{}, std::move(pages), {}, from);
    }

    std::unique_ptr<PageAllocator> device_alloc_;
    std::unique_ptr<KVPrefixCache> cache_;
};

// ---------------------------------------------------------------------------
// Locked-node deferral: a locked leaf is skipped and evicted in a later call
// ---------------------------------------------------------------------------

TEST_F(EvictionLRUTest, LockedLeafDeferredNotEvicted) {
    const int32_t initial = device_alloc_->AvailablePages();

    // Insert two independent leaves: A (tokens 1..4) and B (tokens 5..8).
    Insert(1, /*start=*/1);
    Insert(1, /*start=*/5);
    ASSERT_EQ(device_alloc_->AvailablePages(), initial - 2);

    // Lock leaf A via a NodeRef.
    auto match_a = cache_->Match(MakeAlignedTokens(1, kPageSize, 1));
    auto ref_a = DeviceNodeRef(match_a.device.last_node);

    // Evict 1 page — only B (unlocked) should be evicted; A is deferred.
    bool ok = cache_->EnsureCapacityByEvict<ResourceType::Device>(initial - 1);
    EXPECT_TRUE(ok);
    EXPECT_EQ(device_alloc_->AvailablePages(), initial - 1);

    // After releasing A's lock, evicting again should free A.
    ref_a = DeviceNodeRef(nullptr);  // triggers on_evictable → updateLeaf(A)
    ok = cache_->EnsureCapacityByEvict<ResourceType::Device>(initial);
    EXPECT_TRUE(ok);
    EXPECT_EQ(device_alloc_->AvailablePages(), initial);
}

// ---------------------------------------------------------------------------
// Cascade eviction: evicting both children exposes parent as a new leaf
// ---------------------------------------------------------------------------

TEST_F(EvictionLRUTest, CascadeEvictionFreesParent) {
    const int32_t initial = device_alloc_->AvailablePages();

    // Insert shared trunk (2 pages) then two unique leaves (1 page each).
    auto trunk = Insert(2, /*start=*/1);
    Insert(1, /*start=*/100, trunk.last_node);
    Insert(1, /*start=*/200, trunk.last_node);
    ASSERT_EQ(device_alloc_->AvailablePages(), initial - 4);

    // Evict enough to free both leaves AND the exposed trunk.
    bool ok = cache_->EnsureCapacityByEvict<ResourceType::Device>(initial);
    EXPECT_TRUE(ok);
    EXPECT_EQ(device_alloc_->AvailablePages(), initial);
}

// ---------------------------------------------------------------------------
// Exact LRU: Touch() while locked is reflected after unlock
// ---------------------------------------------------------------------------

TEST_F(EvictionLRUTest, ExactLRUAfterTouchWhileLocked) {
    // Insert OLD first, then NEW — OLD has an older timestamp.
    Insert(1, /*start=*/1);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    Insert(1, /*start=*/5);

    // Match OLD: Touch() updates its timestamp to "now", making it MRU.
    // Then lock it so it is deferred if eviction fires while locked.
    auto match_old = cache_->Match(MakeAlignedTokens(1, kPageSize, 1));
    auto ref_old = DeviceNodeRef(match_old.device.last_node);

    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    // Capture first-evicted token inside the callback (node may be freed by
    // pruneEvicted after EnsureCapacityByEvict returns).
    token_t first_evicted_start = -1;
    cache_->GetDeviceManager().SetEvictionCallback([&](TreeNode* n) {
        if (first_evicted_start == -1 && !n->Tokens().empty()) {
            first_evicted_start = n->Tokens().front();
        }
    });

    // Release OLD's lock — on_evictable fires, updateLeaf re-inserts OLD with
    // its current (post-Touch) timestamp, making OLD the MRU.
    ref_old = DeviceNodeRef(nullptr);

    // Evict 1 page: NEW is now the LRU (older insertion time), should go first.
    const int32_t initial = device_alloc_->AvailablePages();
    bool ok = cache_->EnsureCapacityByEvict<ResourceType::Device>(initial + 1);
    EXPECT_TRUE(ok);

    EXPECT_EQ(first_evicted_start, 5);  // NEW's tokens start at 5, not OLD's at 1
}

// ---------------------------------------------------------------------------
// EvictablePagesNum reflects unlocked leaf pages
// ---------------------------------------------------------------------------

TEST_F(EvictionLRUTest, EvictablePagesNumAccurate) {
    EXPECT_EQ(cache_->GetDeviceManager().EvictablePagesNum(), 0);

    Insert(1, /*start=*/1);
    EXPECT_EQ(cache_->GetDeviceManager().EvictablePagesNum(), 1);

    Insert(2, /*start=*/5);
    EXPECT_EQ(cache_->GetDeviceManager().EvictablePagesNum(), 3);

    cache_->EnsureCapacityByEvict<ResourceType::Device>(kTotalPages);
    EXPECT_EQ(cache_->GetDeviceManager().EvictablePagesNum(), 0);
}

}  // namespace tokenspeed::test
