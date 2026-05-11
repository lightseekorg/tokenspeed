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

#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/radix_tree/mamba_slot.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/node_range.h"
#include "resource/allocator/page_allocator.h"
#include "unit_test_helper.h"

namespace tokenspeed::test {

class MambaCacheTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kPageSize = 2;
    static constexpr std::int32_t kDevicePages = 32;
    static constexpr std::int32_t kHostPages = 0;
    static constexpr std::int32_t kMambaSlots = 8;
    static constexpr std::int32_t kMambaCacheChunkSize = 4;

    void SetUp() override {
        device_alloc_ = std::make_unique<PageAllocator>(kPageSize, kDevicePages);
        host_alloc_ = std::make_unique<PageAllocator>(kPageSize, kHostPages);
        prefix_cache_ = std::make_unique<KVPrefixCache>(device_alloc_.get(), host_alloc_.get());
        mamba_alloc_ = std::make_unique<MambaChunkAllocator>(kMambaSlots);
        hybrid_prefix_cache_ =
            std::make_unique<HybridPrefixCache>(*prefix_cache_, mamba_alloc_.get(), kMambaCacheChunkSize);
    }

    std::vector<std::int32_t> CollectPrefixPages(TreeNode* matched_node) {
        if (matched_node == nullptr || matched_node->IsRoot()) return {};
        return DevicePagesFromRoot(matched_node);
    }

    void InsertKVAndMamba(const token_vec_t& tokens) {
        auto match = prefix_cache_->Match(tokens);
        std::int32_t matched_pages = match.device.DepthInPage();
        std::int32_t total_pages = static_cast<std::int32_t>(tokens.size()) / kPageSize;
        std::int32_t new_pages = total_pages - matched_pages;
        if (new_pages > 0) {
            auto prefix_pages = CollectPrefixPages(match.device.last_node);
            auto result =
                prefix_cache_->Insert<ResourceType::Device>(tokens, prefix_pages, device_alloc_->Allocate(new_pages));
            auto slot = mamba_alloc_->Allocate();
            if (slot.has_value()) {
                hybrid_prefix_cache_->InsertMamba(result.last_node, std::make_unique<MambaSlot>(std::move(*slot)));
            }
        }
    }

    void InsertKVOnly(const token_vec_t& tokens) {
        auto match = prefix_cache_->Match(tokens);
        std::int32_t matched_pages = match.device.DepthInPage();
        std::int32_t total_pages = static_cast<std::int32_t>(tokens.size()) / kPageSize;
        std::int32_t new_pages = total_pages - matched_pages;
        if (new_pages > 0) {
            auto prefix_pages = CollectPrefixPages(match.device.last_node);
            prefix_cache_->Insert<ResourceType::Device>(tokens, prefix_pages, device_alloc_->Allocate(new_pages));
        }
    }

    std::unique_ptr<PageAllocator> device_alloc_;
    std::unique_ptr<PageAllocator> host_alloc_;
    std::unique_ptr<MambaChunkAllocator> mamba_alloc_;
    std::unique_ptr<KVPrefixCache> prefix_cache_;
    std::unique_ptr<HybridPrefixCache> hybrid_prefix_cache_;
};

TEST_F(MambaCacheTest, MatchWithoutMambaTruncatesToRoot) {
    auto tokens = MakeAlignedTokens(3, kPageSize);
    InsertKVOnly(tokens);

    auto match = hybrid_prefix_cache_->Match(tokens);
    EXPECT_EQ(match.device.DepthInPage(), 0);
    EXPECT_EQ(match.mamba_cow_src_index, -1);
    EXPECT_EQ(match.mamba_branching_seqlen, 4);
}

TEST_F(MambaCacheTest, MatchWithFullMambaKeepsDepth) {
    auto tokens = MakeAlignedTokens(3, kPageSize);
    InsertKVAndMamba(tokens);

    auto match = hybrid_prefix_cache_->Match(tokens);
    EXPECT_EQ(match.device.DepthInPage(), 3);
    EXPECT_NE(match.mamba_cow_src_index, -1);
    EXPECT_EQ(match.mamba_branching_seqlen, -1);
}

TEST_F(MambaCacheTest, MatchWithPartialMambaTruncatesToMambaDepth) {
    auto tokens2 = MakeAlignedTokens(2, kPageSize);
    InsertKVAndMamba(tokens2);

    auto tokens4 = MakeAlignedTokens(4, kPageSize);
    InsertKVOnly(tokens4);

    auto match = hybrid_prefix_cache_->Match(tokens4);
    EXPECT_EQ(match.device.DepthInPage(), 2);
    EXPECT_NE(match.mamba_cow_src_index, -1);
    EXPECT_NE(match.mamba_branching_seqlen, -1);
    EXPECT_EQ(match.mamba_branching_seqlen, 8);
}

TEST_F(MambaCacheTest, SplitPrefixWithoutMambaStillRequestsBranchingSnapshot) {
    auto tokens4 = MakeAlignedTokens(4, kPageSize);
    InsertKVAndMamba(tokens4);

    token_vec_t diverged = tokens4;
    diverged.resize(3 * kPageSize);
    diverged[2 * kPageSize] = 1001;
    diverged[2 * kPageSize + 1] = 1002;

    auto match = hybrid_prefix_cache_->Match(diverged);
    EXPECT_EQ(match.device.DepthInPage(), 0);
    EXPECT_EQ(match.mamba_cow_src_index, -1);
    EXPECT_EQ(match.mamba_branching_seqlen, 4);
}

TEST_F(MambaCacheTest, BranchingSeqlenIsSuppressedWhenAlignedInsideMambaPrefix) {
    auto tokens2 = MakeAlignedTokens(2, kPageSize);
    InsertKVAndMamba(tokens2);

    auto tokens3 = MakeAlignedTokens(3, kPageSize);
    InsertKVOnly(tokens3);

    auto match = hybrid_prefix_cache_->Match(tokens3);
    EXPECT_EQ(match.device.DepthInPage(), 2);
    EXPECT_NE(match.mamba_cow_src_index, -1);
    EXPECT_EQ(match.mamba_branching_seqlen, -1);
}

TEST_F(MambaCacheTest, OnKVEvictRemovesMamba) {
    auto tokens = MakeAlignedTokens(2, kPageSize);
    InsertKVAndMamba(tokens);

    auto match = prefix_cache_->Match(tokens);
    TreeNode* node = match.device.last_node;
    EXPECT_TRUE(node->HasMamba());

    hybrid_prefix_cache_->OnKVEvict(node);
    EXPECT_FALSE(node->HasMamba());
}

TEST_F(MambaCacheTest, FindLastMambaNodeWalksUp) {
    auto tokens2 = MakeAlignedTokens(2, kPageSize);
    InsertKVAndMamba(tokens2);

    auto tokens4 = MakeAlignedTokens(4, kPageSize);
    InsertKVOnly(tokens4);

    auto match = prefix_cache_->Match(tokens4);
    TreeNode* terminal = match.device.last_node;
    TreeNode* mamba_node = hybrid_prefix_cache_->FindLastMambaNode(terminal);

    ASSERT_NE(mamba_node, nullptr);
    EXPECT_TRUE(mamba_node->HasMamba());
    EXPECT_EQ(mamba_node->DepthInPage(kPageSize), 2);
}

TEST_F(MambaCacheTest, KVEvictionTriggersMambaEviction) {
    auto tokens = MakeAlignedTokens(2, kPageSize);
    InsertKVAndMamba(tokens);

    auto match = prefix_cache_->Match(tokens);
    TreeNode* node = match.device.last_node;
    EXPECT_TRUE(node->HasMamba());

    prefix_cache_->GetDeviceManager().SetEvictionCallback([this](TreeNode* n) { hybrid_prefix_cache_->OnKVEvict(n); });

    prefix_cache_->EnsureCapacityByEvict<ResourceType::Device>(kDevicePages);

    EXPECT_FALSE(node->HasMamba());
}

}  // namespace tokenspeed::test
