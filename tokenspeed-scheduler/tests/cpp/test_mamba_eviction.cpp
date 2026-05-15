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
#include <thread>

#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/hybrid_prefix_cache/mamba_eviction_manager.h"
#include "resource/radix_tree/mamba_slot.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/allocator/page_allocator.h"
#include "resource/types.h"

namespace tokenspeed::test {

class MambaEvictionTest : public ::testing::Test {
protected:
    void SetUp() override {
        mamba_alloc_ = std::make_unique<MambaChunkAllocator>(8);
        page_alloc_ = std::make_unique<PageAllocator>(2, 32);
        eviction_ = std::make_unique<MambaEvictionManager>(mamba_alloc_.get(), ResourceType::Device);
    }

    TreeNode* MakeNodeWithMamba(TreeNode* parent, const token_vec_t& tokens) {
        auto child = std::make_unique<TreeNode>(tokens);
        auto pages = page_alloc_->Allocate(static_cast<std::int32_t>(tokens.size() / 2));
        child->AttachResource<ResourceType::Device>(std::make_unique<DeviceResource>(std::move(pages)));
        auto slot = mamba_alloc_->Allocate();
        child->AttachMamba(std::make_unique<MambaSlot>(std::move(*slot)));
        TreeNode* raw = child.get();
        parent->AddChild(tokens, std::move(child));
        eviction_->TrackNode(raw);
        return raw;
    }

    std::unique_ptr<MambaChunkAllocator> mamba_alloc_;
    std::unique_ptr<PageAllocator> page_alloc_;
    std::unique_ptr<MambaEvictionManager> eviction_;
    TreeNode root_;
};

TEST_F(MambaEvictionTest, EvictFreesSlotButKeepsKV) {
    auto tokens = token_vec_t{1, 2, 3, 4};
    auto* node = MakeNodeWithMamba(&root_, tokens);
    EXPECT_TRUE(node->HasMamba());
    EXPECT_TRUE(node->OnDevice());

    std::int32_t freed = eviction_->Evict(1);
    EXPECT_EQ(freed, 1);
    EXPECT_FALSE(node->HasMamba());
    EXPECT_TRUE(node->OnDevice());
}

TEST_F(MambaEvictionTest, EvictLRUOrder) {
    auto t1 = token_vec_t{1, 2};
    auto* node1 = MakeNodeWithMamba(&root_, t1);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto t2 = token_vec_t{3, 4};
    auto* node2 = MakeNodeWithMamba(&root_, t2);

    std::int32_t freed = eviction_->Evict(1);
    EXPECT_EQ(freed, 1);
    EXPECT_FALSE(node1->HasMamba());
    EXPECT_TRUE(node2->HasMamba());
}

TEST_F(MambaEvictionTest, SkipsLockedNodes) {
    auto tokens = token_vec_t{1, 2, 3, 4};
    auto* node = MakeNodeWithMamba(&root_, tokens);

    DeviceNodeRef ref(node);

    std::int32_t freed = eviction_->Evict(1);
    EXPECT_EQ(freed, 0);
    EXPECT_TRUE(node->HasMamba());
}

TEST_F(MambaEvictionTest, EnsureCapacityEvictsEnough) {
    auto t1 = token_vec_t{1, 2};
    auto t2 = token_vec_t{3, 4};
    MakeNodeWithMamba(&root_, t1);
    MakeNodeWithMamba(&root_, t2);

    EXPECT_EQ(mamba_alloc_->AvailableSlots(), 6);
    bool ok = eviction_->EnsureCapacity(8);
    EXPECT_TRUE(ok);
    EXPECT_GE(mamba_alloc_->AvailableSlots(), 8);
}

TEST_F(MambaEvictionTest, LeafPromotion) {
    auto ta = token_vec_t{1, 2};
    auto* a = MakeNodeWithMamba(&root_, ta);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto tb = token_vec_t{3, 4};
    auto* b = MakeNodeWithMamba(a, tb);

    std::int32_t freed = eviction_->Evict(1);
    EXPECT_EQ(freed, 1);
    EXPECT_FALSE(b->HasMamba());
    EXPECT_TRUE(a->HasMamba());

    freed = eviction_->Evict(1);
    EXPECT_EQ(freed, 1);
    EXPECT_FALSE(a->HasMamba());
}

}  // namespace tokenspeed::test
