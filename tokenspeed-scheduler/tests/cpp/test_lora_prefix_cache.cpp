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
#include <memory>

#include "unit_test_helper.h"
#include "resource/allocator/page_allocator.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"

namespace tokenspeed::test {

class LoraPrefixCacheTest : public ::testing::Test {
protected:
    static constexpr int32_t kPageSize = 4;
    static constexpr int32_t kTotalPages = 128;

    void SetUp() override {
        device_alloc_ = std::make_unique<PageAllocator>(kPageSize, kTotalPages);
        cache_ = std::make_unique<KVPrefixCache>(device_alloc_.get(), /*host=*/nullptr);
    }

    // Insert N pages for a given token sequence under a given lora_id.
    InsertResult DoInsert(int32_t num_pages, token_t start_token, int32_t lora_id) {
        auto tokens = MakeAlignedTokens(num_pages, kPageSize, start_token);
        auto pages = device_alloc_->Allocate(num_pages);
        return cache_->Insert<ResourceType::Device>(tokens, /*prefix_pages=*/{}, std::move(pages),
                                                    /*page_hashs=*/{}, /*start_node=*/nullptr, lora_id);
    }

    // Return the matched device depth (in pages) for a given sequence + lora_id.
    int32_t MatchDepth(int32_t num_pages, token_t start_token, int32_t lora_id) {
        auto tokens = MakeAlignedTokens(num_pages, kPageSize, start_token);
        return cache_->Match(tokens, lora_id).device.DepthInPage();
    }

    std::unique_ptr<PageAllocator> device_alloc_;
    std::unique_ptr<KVPrefixCache> cache_;
};

// ---------------------------------------------------------------------------
// Same adapter reuses prefix cache (intra-adapter sharing)
// ---------------------------------------------------------------------------

TEST_F(LoraPrefixCacheTest, SameAdapterReusesPrefixCache) {
    DoInsert(2, /*start_token=*/1, /*lora_id=*/1);
    // A second request with the same adapter and same tokens should hit the cache.
    EXPECT_EQ(MatchDepth(2, 1, /*lora_id=*/1), 2);
}

// ---------------------------------------------------------------------------
// Different adapters do not share cache entries (cross-adapter isolation)
// ---------------------------------------------------------------------------

TEST_F(LoraPrefixCacheTest, DifferentAdaptersDontShareCache) {
    // Insert tokens [1..8] under adapter 1.
    DoInsert(2, /*start_token=*/1, /*lora_id=*/1);
    // Adapter 2 has no entry for the same tokens — expect 0 hit.
    EXPECT_EQ(MatchDepth(2, 1, /*lora_id=*/2), 0);
}

// ---------------------------------------------------------------------------
// Base model (lora_id=0) is independent of any adapter namespace
// ---------------------------------------------------------------------------

TEST_F(LoraPrefixCacheTest, BaseModelIndependentOfAdapters) {
    // Insert under adapter 1 and the base model with the same tokens.
    DoInsert(2, /*start_token=*/1, /*lora_id=*/1);
    DoInsert(2, /*start_token=*/1, /*lora_id=*/kLoraNone);

    // Each namespace sees only its own entries.
    EXPECT_EQ(MatchDepth(2, 1, /*lora_id=*/1), 2);
    EXPECT_EQ(MatchDepth(2, 1, /*lora_id=*/kLoraNone), 2);

    // Adapter 2 still gets nothing for these tokens.
    EXPECT_EQ(MatchDepth(2, 1, /*lora_id=*/2), 0);
}

// ---------------------------------------------------------------------------
// Multiple adapters each cache independently
// ---------------------------------------------------------------------------

TEST_F(LoraPrefixCacheTest, MultipleAdaptersCacheIndependently) {
    // Insert different sequences for three different adapters.
    DoInsert(1, /*start_token=*/100, /*lora_id=*/1);
    DoInsert(1, /*start_token=*/200, /*lora_id=*/2);
    DoInsert(1, /*start_token=*/300, /*lora_id=*/3);

    EXPECT_EQ(MatchDepth(1, 100, /*lora_id=*/1), 1);
    EXPECT_EQ(MatchDepth(1, 200, /*lora_id=*/2), 1);
    EXPECT_EQ(MatchDepth(1, 300, /*lora_id=*/3), 1);

    // Cross-adapter: each adapter sees 0 for the others' tokens.
    EXPECT_EQ(MatchDepth(1, 200, /*lora_id=*/1), 0);
    EXPECT_EQ(MatchDepth(1, 100, /*lora_id=*/2), 0);
}

// ---------------------------------------------------------------------------
// InsertResult.last_node stays within the adapter namespace
// ---------------------------------------------------------------------------

TEST_F(LoraPrefixCacheTest, InsertLastNodeIsInAdapterNamespace) {
    auto result1 = DoInsert(2, /*start_token=*/1, /*lora_id=*/1);
    auto result2 = DoInsert(2, /*start_token=*/1, /*lora_id=*/2);
    // last_nodes should be distinct (different subtrees).
    EXPECT_NE(result1.last_node, result2.last_node);
    EXPECT_NE(result1.last_node, nullptr);
    EXPECT_NE(result2.last_node, nullptr);
}

// ---------------------------------------------------------------------------
// Eviction only evicts within the same namespace
// ---------------------------------------------------------------------------

TEST_F(LoraPrefixCacheTest, EvictionDoesNotCrossNamespaces) {
    const int32_t initial = device_alloc_->AvailablePages();
    DoInsert(2, /*start_token=*/1, /*lora_id=*/1);
    DoInsert(2, /*start_token=*/1, /*lora_id=*/2);
    ASSERT_EQ(device_alloc_->AvailablePages(), initial - 4);

    // Evict everything.
    cache_->EnsureCapacityByEvict<ResourceType::Device>(initial);
    EXPECT_EQ(device_alloc_->AvailablePages(), initial);

    // Both namespaces should now have empty caches.
    EXPECT_EQ(MatchDepth(2, 1, /*lora_id=*/1), 0);
    EXPECT_EQ(MatchDepth(2, 1, /*lora_id=*/2), 0);
}

}  // namespace tokenspeed::test
