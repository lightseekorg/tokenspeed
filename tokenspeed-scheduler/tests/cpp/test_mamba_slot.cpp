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
#include <numeric>
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/mamba_host_allocator.h"
#include "resource/radix_tree/mamba_slot.h"
#include "resource/allocator/local_mamba_allocator.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/allocator/page_allocator.h"

namespace tokenspeed::test {

TEST(MambaChunkAllocatorTest, AllocateReturnsSequentialSlots) {
    MambaChunkAllocator allocator(4);
    EXPECT_EQ(allocator.TotalSlots(), 4);
    EXPECT_EQ(allocator.AvailableSlots(), 4);

    auto slot0 = allocator.Allocate();
    ASSERT_TRUE(slot0.has_value());
    EXPECT_EQ(allocator.AvailableSlots(), 3);

    auto slot1 = allocator.Allocate();
    ASSERT_TRUE(slot1.has_value());
    EXPECT_NE(slot0->Index(), slot1->Index());
}

TEST(MambaChunkAllocatorTest, ExhaustedPoolReturnsNullopt) {
    MambaChunkAllocator allocator(2);
    auto s0 = allocator.Allocate();
    auto s1 = allocator.Allocate();
    auto s2 = allocator.Allocate();
    EXPECT_FALSE(s2.has_value());
}

TEST(MambaHostAllocatorTest, AllocateFreeAndDrainReleased) {
    tokenspeed::MambaHostAllocator allocator(3);
    auto slot = allocator.Allocate();
    ASSERT_TRUE(slot.has_value());
    std::int32_t idx = slot->Index();
    EXPECT_EQ(allocator.AvailableSlots(), 2);

    slot.reset();
    EXPECT_EQ(allocator.AvailableSlots(), 3);
    auto released = allocator.DrainReleased();
    ASSERT_EQ(released.size(), 1);
    EXPECT_EQ(released[0], idx);
    EXPECT_TRUE(allocator.DrainReleased().empty());
}

TEST(MambaSlotTest, CustomReleaserRunsOnDestruction) {
    std::vector<std::int32_t> released;
    {
        tokenspeed::MambaSlot slot(7, [&released](std::int32_t idx) { released.push_back(idx); });
        EXPECT_EQ(slot.Index(), 7);
    }
    ASSERT_EQ(released.size(), 1);
    EXPECT_EQ(released[0], 7);
}

TEST(MambaSlotTest, RAIIFreesOnDestruction) {
    MambaChunkAllocator allocator(2);
    {
        auto slot = allocator.Allocate();
        ASSERT_TRUE(slot.has_value());
        EXPECT_EQ(allocator.AvailableSlots(), 1);
    }
    EXPECT_EQ(allocator.AvailableSlots(), 2);
}

TEST(MambaSlotTest, MoveTransfersOwnership) {
    MambaChunkAllocator allocator(2);
    auto slot = allocator.Allocate();
    ASSERT_TRUE(slot.has_value());
    std::int32_t idx = slot->Index();

    MambaSlot moved = std::move(*slot);
    EXPECT_EQ(moved.Index(), idx);
    EXPECT_EQ(allocator.AvailableSlots(), 1);
}

TEST(MambaSlotTest, UniquePtrOwnership) {
    MambaChunkAllocator allocator(2);
    {
        auto slot = allocator.Allocate();
        auto ptr = std::make_unique<MambaSlot>(std::move(*slot));
        EXPECT_EQ(allocator.AvailableSlots(), 1);
    }
    EXPECT_EQ(allocator.AvailableSlots(), 2);
}

TEST(TreeNodeMambaTest, DefaultHasNoMamba) {
    tokenspeed::TreeNode node;
    EXPECT_FALSE(node.HasMamba());
}

TEST(TreeNodeMambaTest, AttachAndDetachMamba) {
    tokenspeed::MambaChunkAllocator allocator(4);
    tokenspeed::TreeNode node;

    auto slot = allocator.Allocate();
    ASSERT_TRUE(slot.has_value());
    std::int32_t idx = slot->Index();

    node.AttachMamba(std::make_unique<tokenspeed::MambaSlot>(std::move(*slot)));
    EXPECT_TRUE(node.HasMamba());
    EXPECT_EQ(node.MambaSlotIndex(), idx);

    auto detached = node.DetachMamba();
    EXPECT_FALSE(node.HasMamba());
    EXPECT_EQ(detached->Index(), idx);
    EXPECT_EQ(allocator.AvailableSlots(), 3);
}

TEST(TreeNodeMambaTest, DestructorFreesMambaSlot) {
    tokenspeed::MambaChunkAllocator allocator(4);
    {
        tokenspeed::TreeNode node;
        auto slot = allocator.Allocate();
        node.AttachMamba(std::make_unique<tokenspeed::MambaSlot>(std::move(*slot)));
        EXPECT_EQ(allocator.AvailableSlots(), 3);
    }
    EXPECT_EQ(allocator.AvailableSlots(), 4);
}

TEST(TreeNodeMambaTest, AttachAndDetachMambaHost) {
    tokenspeed::MambaHostAllocator allocator(4);
    tokenspeed::TreeNode node;

    auto slot = allocator.Allocate();
    ASSERT_TRUE(slot.has_value());
    std::int32_t idx = slot->Index();

    node.AttachMambaHost(std::make_unique<tokenspeed::MambaSlot>(std::move(*slot)));
    EXPECT_TRUE(node.HasMambaOnHost());
    EXPECT_EQ(node.MambaHostSlotIndex(), idx);

    auto detached = node.DetachMambaHost();
    EXPECT_FALSE(node.HasMambaOnHost());
    EXPECT_EQ(detached->Index(), idx);
    EXPECT_EQ(allocator.AvailableSlots(), 3);
}

TEST(TreeNodeMambaTest, DestructorFreesMambaHostSlot) {
    tokenspeed::MambaHostAllocator allocator(4);
    {
        tokenspeed::TreeNode node;
        auto slot = allocator.Allocate();
        node.AttachMambaHost(std::make_unique<tokenspeed::MambaSlot>(std::move(*slot)));
        EXPECT_EQ(allocator.AvailableSlots(), 3);
    }
    EXPECT_EQ(allocator.AvailableSlots(), 4);
    EXPECT_EQ(allocator.DrainReleased().size(), 1);
}

TEST(TreeNodeMambaTest, SplitKeepsMambaOnSuffix) {
    tokenspeed::MambaChunkAllocator mamba_alloc(4);
    tokenspeed::PageAllocator page_alloc(2, 8);

    tokenspeed::TreeNode root;
    auto tokens = tokenspeed::token_vec_t(8);
    std::iota(tokens.begin(), tokens.end(), 1);

    auto child_ptr = std::make_unique<tokenspeed::TreeNode>(tokens);
    auto pages = page_alloc.Allocate(4);
    child_ptr->AttachResource<tokenspeed::ResourceType::Device>(
        std::make_unique<tokenspeed::DeviceResource>(std::move(pages)));
    auto slot = mamba_alloc.Allocate();
    std::int32_t idx = slot->Index();
    child_ptr->AttachMamba(std::make_unique<tokenspeed::MambaSlot>(std::move(*slot)));

    tokenspeed::TreeNode* child = child_ptr.get();
    root.AddChild(tokens, std::move(child_ptr));

    tokenspeed::TreeNode prefix;
    child->SplitSelfInto(prefix, 2, 2);

    EXPECT_TRUE(child->HasMamba());
    EXPECT_EQ(child->MambaSlotIndex(), idx);
    EXPECT_FALSE(prefix.HasMamba());
}

TEST(LocalMambaAllocatorTest, AllocateWorkingAndCheckpoint) {
    tokenspeed::MambaChunkAllocator allocator(8);
    tokenspeed::LocalMambaAllocator local(&allocator);

    EXPECT_FALSE(local.HasWorking());
    EXPECT_TRUE(local.AllocateWorking());
    EXPECT_TRUE(local.HasWorking());
    EXPECT_GE(local.WorkingIndex(), 0);

    EXPECT_FALSE(local.HasCheckpoint());
    EXPECT_TRUE(local.AllocateCheckpoint());
    EXPECT_TRUE(local.HasCheckpoint());
    EXPECT_GE(local.CheckpointIndex(), 0);
    EXPECT_NE(local.WorkingIndex(), local.CheckpointIndex());

    EXPECT_EQ(allocator.AvailableSlots(), 6);
}

TEST(LocalMambaAllocatorTest, DetachCheckpointTransfersOwnership) {
    tokenspeed::MambaChunkAllocator allocator(8);
    tokenspeed::LocalMambaAllocator local(&allocator);

    local.AllocateWorking();
    local.AllocateCheckpoint();
    std::int32_t cp_idx = local.CheckpointIndex();

    auto detached = local.DetachCheckpoint();
    EXPECT_FALSE(local.HasCheckpoint());
    EXPECT_EQ(detached->Index(), cp_idx);
    EXPECT_EQ(allocator.AvailableSlots(), 6);
}

TEST(LocalMambaAllocatorTest, DetachWorkingTransfersOwnership) {
    tokenspeed::MambaChunkAllocator allocator(8);
    tokenspeed::LocalMambaAllocator local(&allocator);

    local.AllocateWorking();
    std::int32_t w_idx = local.WorkingIndex();

    auto detached = local.DetachWorking();
    EXPECT_FALSE(local.HasWorking());
    EXPECT_EQ(detached->Index(), w_idx);
}

TEST(LocalMambaAllocatorTest, DestructorFreesAllSlots) {
    tokenspeed::MambaChunkAllocator allocator(8);
    {
        tokenspeed::LocalMambaAllocator local(&allocator);
        local.AllocateWorking();
        local.AllocateCheckpoint();
        EXPECT_EQ(allocator.AvailableSlots(), 6);
    }
    EXPECT_EQ(allocator.AvailableSlots(), 8);
}

}  // namespace tokenspeed::test
