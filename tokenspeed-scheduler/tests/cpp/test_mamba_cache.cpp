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

#include <stdexcept>
#include <variant>

#include "core/token_container.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"

#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/mamba_host_allocator.h"
#include "scheduler/operations/cache.h"
#include "resource/radix_tree/mamba_slot.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/node_range.h"
#include "resource/allocator/page_allocator.h"
#include "resource/allocator/req_pool_allocator.h"
#include "unit_test_helper.h"
#include "scheduler/types.h"

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

class MambaL2CacheTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kPageSize = 2;
    static constexpr std::int32_t kDevicePages = 32;
    static constexpr std::int32_t kHostPages = 32;
    static constexpr std::int32_t kMambaSlots = 8;
    static constexpr std::int32_t kMambaHostSlots = 8;
    static constexpr std::int32_t kMambaCacheChunkSize = 4;

    void SetUp() override {
        device_alloc_ = std::make_unique<PageAllocator>(kPageSize, kDevicePages);
        host_alloc_ = std::make_unique<PageAllocator>(kPageSize, kHostPages);
        prefix_cache_ = std::make_unique<KVPrefixCache>(device_alloc_.get(), host_alloc_.get());
        mamba_alloc_ = std::make_unique<MambaChunkAllocator>(kMambaSlots);
        mamba_host_alloc_ = std::make_unique<MambaHostAllocator>(kMambaHostSlots);
        hybrid_prefix_cache_ = std::make_unique<HybridPrefixCache>(*prefix_cache_, mamba_alloc_.get(),
                                                                   kMambaCacheChunkSize, mamba_host_alloc_.get());
    }

    TreeNode* InsertHostKV(const token_vec_t& tokens) {
        auto result = prefix_cache_->Insert<ResourceType::Host>(
            tokens, {}, host_alloc_->Allocate(static_cast<std::int32_t>(tokens.size()) / kPageSize));
        return result.last_node;
    }

    std::unique_ptr<PageAllocator> device_alloc_;
    std::unique_ptr<PageAllocator> host_alloc_;
    std::unique_ptr<MambaChunkAllocator> mamba_alloc_;
    std::unique_ptr<MambaHostAllocator> mamba_host_alloc_;
    std::unique_ptr<KVPrefixCache> prefix_cache_;
    std::unique_ptr<HybridPrefixCache> hybrid_prefix_cache_;
};

TEST_F(MambaL2CacheTest, HostKVRequiresHostMambaForHybridMatch) {
    auto tokens = MakeAlignedTokens(3, kPageSize);
    TreeNode* node = InsertHostKV(tokens);

    auto device_slot = mamba_alloc_->Allocate();
    ASSERT_TRUE(device_slot.has_value());
    node->AttachMamba(std::make_unique<MambaSlot>(std::move(*device_slot)));

    auto mismatch = hybrid_prefix_cache_->Match(tokens);
    EXPECT_EQ(mismatch.host.DepthInPage(), 0);
    EXPECT_EQ(mismatch.device.DepthInPage(), 0);

    node->DetachMamba();
    auto host_slot = mamba_host_alloc_->Allocate();
    ASSERT_TRUE(host_slot.has_value());
    const std::int32_t host_idx = host_slot->Index();
    node->AttachMambaHost(std::make_unique<MambaSlot>(std::move(*host_slot)));

    auto match = hybrid_prefix_cache_->Match(tokens);
    EXPECT_EQ(match.host.DepthInPage(), 3);
    EXPECT_EQ(match.device.DepthInPage(), 0);
    EXPECT_EQ(match.mamba_host_src_index, host_idx);
    EXPECT_EQ(match.mamba_cow_src_index, -1);
}

TEST_F(MambaL2CacheTest, DeeperHostMambaMatchTakesPriorityOverShallowDeviceMamba) {
    auto tokens2 = MakeAlignedTokens(2, kPageSize);
    auto device_result = prefix_cache_->Insert<ResourceType::Device>(tokens2, {}, device_alloc_->Allocate(2));
    TreeNode* device_node = device_result.last_node;
    auto device_slot = mamba_alloc_->Allocate();
    ASSERT_TRUE(device_slot.has_value());
    device_node->AttachMamba(std::make_unique<MambaSlot>(std::move(*device_slot)));

    auto tokens4 = MakeAlignedTokens(4, kPageSize);
    auto host_result = prefix_cache_->Insert<ResourceType::Host>(tokens4, {}, host_alloc_->Allocate(4));
    TreeNode* host_node = host_result.last_node;
    auto host_slot = mamba_host_alloc_->Allocate();
    ASSERT_TRUE(host_slot.has_value());
    const std::int32_t host_idx = host_slot->Index();
    host_node->AttachMambaHost(std::make_unique<MambaSlot>(std::move(*host_slot)));

    auto match = hybrid_prefix_cache_->Match(tokens4);

    EXPECT_EQ(match.device.DepthInPage(), 2);
    EXPECT_EQ(match.host.DepthInPage(), 4);
    EXPECT_EQ(match.mamba_host_src_index, host_idx);
    EXPECT_EQ(match.mamba_cow_src_index, -1) << "deeper host hit must trigger Mamba L2 loadback";
}

TEST_F(MambaL2CacheTest, PrefillFirstChunkRequiresCheckpointSlot) {
    MambaChunkAllocator one_slot_mamba_alloc(1);
    ReqPoolAllocator req_pool_alloc(1);
    auto tokens = MakeAlignedTokens(1, kPageSize);
    TokenContainer token_container(tokens);
    auto match = prefix_cache_->Match(token_container.GetFullPagedTokens(kPageSize, true));

    fsm::SchedulePrefillFirstChunkEvent event{
        static_cast<std::int32_t>(tokens.size()),
        0,
        device_alloc_.get(),
        &req_pool_alloc,
        match,
        Role::kP,
        prefix_cache_.get(),
        false,
        {},
        hybrid_prefix_cache_.get(),
        &one_slot_mamba_alloc,
    };

    EXPECT_THROW((void)event(fsm::Submitted{&token_container, kPageSize}), std::logic_error);
}

TEST_F(MambaL2CacheTest, PrepareMambaLoadBackAllocatesDeviceSlotAndTransferPair) {
    auto tokens = MakeAlignedTokens(2, kPageSize);
    TreeNode* node = InsertHostKV(tokens);
    auto host_slot = mamba_host_alloc_->Allocate();
    ASSERT_TRUE(host_slot.has_value());
    const std::int32_t host_idx = host_slot->Index();
    node->AttachMambaHost(std::make_unique<MambaSlot>(std::move(*host_slot)));

    auto transfers = hybrid_prefix_cache_->PrepareMambaDeviceLoadBack({node});

    ASSERT_TRUE(node->HasMamba());
    ASSERT_EQ(transfers.size(), 1u);
    EXPECT_EQ(transfers[0].kind, CacheKind::kMamba);
    EXPECT_EQ(transfers[0].src, host_idx);
    EXPECT_EQ(transfers[0].dst, node->MambaSlotIndex());
}

TEST_F(MambaL2CacheTest, ExactWriteBackAckDoesNotPublishUnackedAncestor) {
    auto tokens2 = MakeAlignedTokens(2, kPageSize);
    auto tokens4 = MakeAlignedTokens(4, kPageSize);
    auto result2 = prefix_cache_->Insert<ResourceType::Device>(tokens2, {}, device_alloc_->Allocate(2));
    auto result4 = prefix_cache_->Insert<ResourceType::Device>(tokens4, {}, device_alloc_->Allocate(4));
    TreeNode* ancestor = result2.last_node;
    TreeNode* descendant = result4.last_node;
    prefix_cache_->Insert<ResourceType::Host>(tokens4, {}, host_alloc_->Allocate(4));

    auto ancestor_slot = mamba_alloc_->Allocate();
    ASSERT_TRUE(ancestor_slot.has_value());
    ancestor->AttachMamba(std::make_unique<MambaSlot>(std::move(*ancestor_slot)));
    auto descendant_slot = mamba_alloc_->Allocate();
    ASSERT_TRUE(descendant_slot.has_value());
    descendant->AttachMamba(std::make_unique<MambaSlot>(std::move(*descendant_slot)));

    auto ancestor_transfers = hybrid_prefix_cache_->PrepareMambaHostWriteBack({ancestor});
    auto descendant_transfers = hybrid_prefix_cache_->PrepareMambaHostWriteBack({descendant});
    ASSERT_EQ(ancestor_transfers.size(), 1u);
    ASSERT_EQ(descendant_transfers.size(), 1u);

    hybrid_prefix_cache_->OnMambaHostWriteBackDone(std::vector<TreeNode*>{descendant});

    EXPECT_FALSE(ancestor->HasMambaOnHost())
        << "an ack for a descendant op must not publish a different pending ancestor";
    EXPECT_TRUE(descendant->HasMambaOnHost());

    hybrid_prefix_cache_->OnMambaHostWriteBackDone(std::vector<TreeNode*>{ancestor});
    EXPECT_TRUE(ancestor->HasMambaOnHost());
}

TEST_F(MambaL2CacheTest, PrepareMambaWriteBackPublishesHostSlotOnlyAfterAck) {
    auto tokens = MakeAlignedTokens(2, kPageSize);
    auto result = prefix_cache_->Insert<ResourceType::Device>(tokens, {}, device_alloc_->Allocate(2));
    TreeNode* node = result.last_node;
    prefix_cache_->Insert<ResourceType::Host>(tokens, {}, host_alloc_->Allocate(2));
    auto device_slot = mamba_alloc_->Allocate();
    ASSERT_TRUE(device_slot.has_value());
    const std::int32_t device_idx = device_slot->Index();
    node->AttachMamba(std::make_unique<MambaSlot>(std::move(*device_slot)));

    auto transfers = hybrid_prefix_cache_->PrepareMambaHostWriteBack({node});

    ASSERT_EQ(transfers.size(), 1u);
    EXPECT_EQ(transfers[0].kind, CacheKind::kMamba);
    EXPECT_EQ(transfers[0].src, device_idx);
    const std::int32_t host_idx = transfers[0].dst;
    EXPECT_FALSE(node->HasMambaOnHost()) << "host mamba must remain invisible until writeback ack";

    auto pending_match = hybrid_prefix_cache_->Match(tokens);
    EXPECT_EQ(pending_match.host.DepthInPage(), 0);

    hybrid_prefix_cache_->OnMambaHostWriteBackDone(node);

    ASSERT_TRUE(node->HasMambaOnHost());
    EXPECT_EQ(node->MambaHostSlotIndex(), host_idx);
    EXPECT_FALSE(node->HasMamba()) << "idle device mamba copy should demote once host writeback is acknowledged";
    auto host_match = hybrid_prefix_cache_->Match(tokens);
    EXPECT_EQ(host_match.host.DepthInPage(), 2);
    EXPECT_EQ(host_match.mamba_host_src_index, host_idx);
    EXPECT_EQ(host_match.mamba_cow_src_index, -1);
}

TEST_F(MambaL2CacheTest, HostWriteBackDemotesAfterDeviceRefUnlock) {
    auto tokens = MakeAlignedTokens(2, kPageSize);
    auto result = prefix_cache_->Insert<ResourceType::Device>(tokens, {}, device_alloc_->Allocate(2));
    TreeNode* node = result.last_node;
    prefix_cache_->Insert<ResourceType::Host>(tokens, {}, host_alloc_->Allocate(2));

    auto device_slot = mamba_alloc_->Allocate();
    ASSERT_TRUE(device_slot.has_value());
    node->AttachMamba(std::make_unique<MambaSlot>(std::move(*device_slot)));

    auto transfers = hybrid_prefix_cache_->PrepareMambaHostWriteBack({node});
    ASSERT_EQ(transfers.size(), 1u);

    {
        DeviceNodeRef device_ref(node);
        hybrid_prefix_cache_->OnMambaHostWriteBackDone(std::vector<TreeNode*>{node});
        EXPECT_TRUE(node->HasMamba()) << "device copy must stay pinned while DeviceNodeRef is live";
        EXPECT_TRUE(node->HasMambaOnHost());
    }

    EXPECT_TRUE(node->HasMamba()) << "device copy is still present before the post-unlock demote pass";

    hybrid_prefix_cache_->DemoteIdleMambaDeviceCopiesPresentOnHost();

    EXPECT_FALSE(node->HasMamba());
    EXPECT_TRUE(node->HasMambaOnHost());
}

TEST_F(MambaL2CacheTest, WriteBackDoneDropsDeviceMambaWhenKVChildKeepsDeviceNode) {
    auto tokens4 = MakeAlignedTokens(4, kPageSize);
    auto result = prefix_cache_->Insert<ResourceType::Device>(tokens4, {}, device_alloc_->Allocate(4));
    TreeNode* node = result.last_node;
    prefix_cache_->Insert<ResourceType::Host>(tokens4, {}, host_alloc_->Allocate(4));

    auto device_slot = mamba_alloc_->Allocate();
    ASSERT_TRUE(device_slot.has_value());
    node->AttachMamba(std::make_unique<MambaSlot>(std::move(*device_slot)));
    auto host_slot = mamba_host_alloc_->Allocate();
    ASSERT_TRUE(host_slot.has_value());
    node->AttachMambaHost(std::make_unique<MambaSlot>(std::move(*host_slot)));

    auto tokens5 = MakeAlignedTokens(5, kPageSize);
    prefix_cache_->Insert<ResourceType::Device>(tokens5, DevicePagesFromRoot(node), device_alloc_->Allocate(1));
    ASSERT_TRUE(node->OnDevice());
    ASSERT_TRUE(node->HasMamba());
    ASSERT_GT(node->NumChildren(), 0u);

    prefix_cache_->ReleaseDeviceResourcesPresentOnHost(
        node, [this](TreeNode* n) { hybrid_prefix_cache_->OnKVDeviceDemote(n); });

    EXPECT_TRUE(node->OnDevice()) << "KV device node is kept because a child still uses the device tier";
    EXPECT_FALSE(node->HasMamba()) << "Mamba device state must still demote to host after writeback";
    EXPECT_TRUE(node->HasMambaOnHost());
}

}  // namespace tokenspeed::test
