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

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "hybrid_prefix_cache_test_peer.h"
#include "resource/allocator/kv_allocator.h"
#include "resource/allocator/local_mamba_allocator.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/mamba_host_allocator.h"
#include "scheduler/operations/cache.h"
#include "resource/radix_tree/mamba_slot.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/node_range.h"
#include "resource/allocator/page_allocator.h"
#include "scheduler/operations/forward.h"
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
        hybrid_prefix_cache_ = std::make_unique<HybridPrefixCache>(*prefix_cache_, *device_alloc_, mamba_alloc_.get(),
                                                                   kMambaCacheChunkSize);
    }

    std::vector<std::int32_t> CollectPrefixPages(TreeNode* matched_node) {
        if (matched_node == nullptr || matched_node->IsRoot()) return {};
        return DevicePagesFromRoot(matched_node);
    }

    TreeNode* InsertKVAndMamba(const token_vec_t& tokens) {
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
                HybridPrefixCacheTestPeer::InsertMamba(*hybrid_prefix_cache_, result.last_node,
                                                       std::make_unique<MambaSlot>(std::move(*slot)));
            }
            return result.last_node;
        }
        return match.device.last_node;
    }

    TreeNode* InsertKVOnly(const token_vec_t& tokens) {
        auto match = prefix_cache_->Match(tokens);
        std::int32_t matched_pages = match.device.DepthInPage();
        std::int32_t total_pages = static_cast<std::int32_t>(tokens.size()) / kPageSize;
        std::int32_t new_pages = total_pages - matched_pages;
        if (new_pages > 0) {
            auto prefix_pages = CollectPrefixPages(match.device.last_node);
            auto result =
                prefix_cache_->Insert<ResourceType::Device>(tokens, prefix_pages, device_alloc_->Allocate(new_pages));
            return result.last_node;
        }
        return match.device.last_node;
    }

    std::vector<std::span<const std::int32_t>> PagedTokenSpans(const token_vec_t& tokens) const {
        std::vector<std::span<const std::int32_t>> pages;
        const auto page_count = static_cast<std::int32_t>(tokens.size()) / kPageSize;
        pages.reserve(static_cast<std::size_t>(page_count));
        for (std::int32_t page = 0; page < page_count; ++page) {
            pages.emplace_back(tokens.data() + page * kPageSize, static_cast<std::size_t>(kPageSize));
        }
        return pages;
    }

    std::vector<TreeNode*> FillMambaSlots(std::int32_t start_token) {
        std::vector<TreeNode*> nodes;
        nodes.reserve(kMambaSlots);
        for (std::int32_t i = 0; i < kMambaSlots; ++i) {
            TreeNode* node = InsertKVAndMamba(MakeAlignedTokens(1, kPageSize, /*start=*/start_token + i * 10));
            nodes.push_back(node);
        }
        return nodes;
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

    auto match = MatchPrefix(*hybrid_prefix_cache_, tokens, kPageSize).compat_match;
    EXPECT_EQ(match.device.DepthInPage(), 0);
    EXPECT_EQ(match.mamba_cow_src_index, -1);
    EXPECT_EQ(match.mamba_branching_seqlen, 4);
}

TEST(HybridPrefixCacheMambaRecoverablePrefixTest, HostOnlyKVWithoutMambaDoesNotProduceLoadBackPrefix) {
    static constexpr std::int32_t kPageSize = 2;
    PageAllocator device_alloc{kPageSize, 4};
    PageAllocator host_alloc{kPageSize, 16};
    KVPrefixCache prefix_cache{&device_alloc, &host_alloc, false};
    MambaChunkAllocator mamba_alloc{2};
    HybridPrefixCache hybrid_prefix_cache{prefix_cache, device_alloc, &mamba_alloc,
                                          /*mamba_cache_chunk_size=*/4};

    auto tokens = MakeAlignedTokens(/*num_pages=*/2, kPageSize);
    prefix_cache.Insert<ResourceType::Host>(tokens, /*prefix_pages=*/{}, host_alloc.Allocate(/*num_pages=*/2));
    auto raw_match = prefix_cache.Match(tokens);
    ASSERT_EQ(raw_match.device.DepthInPage(), 0);
    ASSERT_EQ(raw_match.host.DepthInPage(), 2);

    auto match = MatchPrefix(hybrid_prefix_cache, tokens, kPageSize).compat_match;

    EXPECT_EQ(match.device.DepthInPage(), 0);
    EXPECT_EQ(match.host.DepthInPage(), 0);
    EXPECT_TRUE(match.NodesWithout<ResourceType::Device>().empty())
        << "host-only KV without tree-owned Mamba state must not plan LoadBack";
    EXPECT_EQ(match.mamba_cow_src_index, -1);
    EXPECT_EQ(match.mamba_branching_seqlen, 4);

    RecoveryPlan recovery = MatchPrefix(hybrid_prefix_cache, tokens, kPageSize, MatchIntent::StateRecovery);
    EXPECT_FALSE(recovery.recovery_state_available);
    EXPECT_EQ(recovery.protected_recovery_node, nullptr);
    EXPECT_TRUE(recovery.compat_match.NodesWithout<ResourceType::Device>().empty());
}

TEST_F(MambaCacheTest, MatchWithFullMambaKeepsDepth) {
    auto tokens = MakeAlignedTokens(3, kPageSize);
    InsertKVAndMamba(tokens);

    auto match = MatchPrefix(*hybrid_prefix_cache_, tokens, kPageSize).compat_match;
    EXPECT_EQ(match.device.DepthInPage(), 3);
    EXPECT_NE(match.mamba_cow_src_index, -1);
    EXPECT_EQ(match.mamba_branching_seqlen, -1);
}

TEST_F(MambaCacheTest, MatchWithPartialMambaTruncatesToMambaDepth) {
    auto tokens2 = MakeAlignedTokens(2, kPageSize);
    InsertKVAndMamba(tokens2);

    auto tokens4 = MakeAlignedTokens(4, kPageSize);
    InsertKVOnly(tokens4);

    auto match = MatchPrefix(*hybrid_prefix_cache_, tokens4, kPageSize).compat_match;
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

    auto match = MatchPrefix(*hybrid_prefix_cache_, diverged, kPageSize).compat_match;
    EXPECT_EQ(match.device.DepthInPage(), 0);
    EXPECT_EQ(match.mamba_cow_src_index, -1);
    EXPECT_EQ(match.mamba_branching_seqlen, 4);
}

TEST_F(MambaCacheTest, BranchingSeqlenIsSuppressedWhenAlignedInsideMambaPrefix) {
    auto tokens2 = MakeAlignedTokens(2, kPageSize);
    InsertKVAndMamba(tokens2);

    auto tokens3 = MakeAlignedTokens(3, kPageSize);
    InsertKVOnly(tokens3);

    auto match = MatchPrefix(*hybrid_prefix_cache_, tokens3, kPageSize).compat_match;
    EXPECT_EQ(match.device.DepthInPage(), 2);
    EXPECT_NE(match.mamba_cow_src_index, -1);
    EXPECT_EQ(match.mamba_branching_seqlen, -1);
}

TEST_F(MambaCacheTest, MatchPrefixStateRecoveryReportsAvailableAndMissingMambaState) {
    auto recoverable_tokens = MakeAlignedTokens(2, kPageSize);
    TreeNode* recovery_source = InsertKVAndMamba(recoverable_tokens);
    ASSERT_NE(recovery_source, nullptr);
    ASSERT_TRUE(recovery_source->HasMamba());

    RecoveryPlan recovery =
        MatchPrefix(*hybrid_prefix_cache_, recoverable_tokens, kPageSize, MatchIntent::StateRecovery);

    EXPECT_TRUE(recovery.recovery_state_available);
    EXPECT_EQ(recovery.protected_recovery_node, recovery_source);
    EXPECT_EQ(recovery.compat_match.mamba_cow_src_index, recovery_source->MambaSlotIndex());

    auto missing_tokens = MakeAlignedTokens(2, kPageSize, /*start=*/100);
    InsertKVOnly(missing_tokens);
    RecoveryPlan missing = MatchPrefix(*hybrid_prefix_cache_, missing_tokens, kPageSize, MatchIntent::StateRecovery);

    EXPECT_FALSE(missing.recovery_state_available);
    EXPECT_EQ(missing.protected_recovery_node, nullptr);
    EXPECT_EQ(missing.compat_match.mamba_cow_src_index, -1);
}

TEST_F(MambaCacheTest, WorkerMetadataReceivesPrefixRecoveryAndRequestLocalInfo) {
    MatchResult prefix_match{};
    prefix_match.mamba_cow_src_index = 7;
    prefix_match.mamba_branching_seqlen = 12;
    ForwardOperationBase prefix_op{};

    hybrid_prefix_cache_->StepCommit({
        .worker_metadata =
            WorkerCompatibilityCommitRequest{
                .op_base = &prefix_op,
                .kind = WorkerCompatibilityCommitKind::kPrefillFirstChunk,
                .compat_match = &prefix_match,
            },
    });

    EXPECT_EQ(prefix_op.mamba_cow_src_idx, 7);
    EXPECT_EQ(prefix_op.mamba_branching_seqlen, 12);

    MatchResult recovery_match{};
    recovery_match.mamba_cow_src_index = 5;
    recovery_match.mamba_branching_seqlen = 12;
    ForwardOperationBase recovery_op{};
    recovery_op.mamba_branching_seqlen = 99;

    hybrid_prefix_cache_->StepCommit({
        .worker_metadata =
            WorkerCompatibilityCommitRequest{
                .op_base = &recovery_op,
                .kind = WorkerCompatibilityCommitKind::kDecodeFromRetracted,
                .compat_match = &recovery_match,
            },
    });

    EXPECT_EQ(recovery_op.mamba_cow_src_idx, 5);
    EXPECT_EQ(recovery_op.mamba_branching_seqlen, 99);

    LocalMambaAllocator local_mamba(mamba_alloc_.get());
    ASSERT_TRUE(local_mamba.AllocateWorking());
    ASSERT_TRUE(local_mamba.AllocateCheckpoint());
    ForwardOperationBase request_local_op{};

    hybrid_prefix_cache_->StepCommit({
        .worker_metadata =
            WorkerCompatibilityCommitRequest{
                .op_base = &request_local_op,
                .local_mamba_allocator_view = &local_mamba,
            },
    });

    EXPECT_EQ(request_local_op.mamba_working_idx, local_mamba.WorkingIndex());
    EXPECT_EQ(request_local_op.mamba_checkpoint_dst_idx, local_mamba.CheckpointIndex());
    EXPECT_EQ(request_local_op.mamba_cow_src_idx, -1);
    EXPECT_EQ(request_local_op.mamba_branching_seqlen, -1);
}

TEST(HybridPrefixCacheKVAllocationTest, PrefillCreatesAndRefreshesRequestLocalStateWithoutLeakingPages) {
    PageAllocator device_alloc(/*page_size=*/2, /*total_pages=*/5);
    PageAllocator host_alloc(/*page_size=*/2, /*total_pages=*/0);
    MambaChunkAllocator mamba_alloc(/*num_slots=*/4);
    KVPrefixCache prefix_cache(&device_alloc, &host_alloc);
    HybridPrefixCache hybrid_prefix_cache(prefix_cache, device_alloc, &mamba_alloc, /*mamba_cache_chunk_size=*/4);
    constexpr std::int32_t kInitialCheckpointPosition = 4;
    constexpr std::int32_t kRefreshedCheckpointPosition = 8;

    auto result = hybrid_prefix_cache.StepCommit({
        .request_local_kv =
            RequestLocalKVStateRequest{
                .create_allocator = true,
                .initial_tokens = 1,
                .acquire_tokens = 2,
            },
        .request_local_mamba =
            RequestLocalMambaStateRequest{
                .create_allocator = true,
                .checkpoint_raw_position = kInitialCheckpointPosition,
            },
    });
    auto local_kv = std::move(result.local_kv_allocator);
    auto local_mamba = std::move(result.local_mamba_allocator);

    ASSERT_NE(local_kv, nullptr);
    EXPECT_EQ(local_kv->Pages().size(), 2u);
    EXPECT_EQ(local_kv->TailPageAvailableTokens(), 1);
    EXPECT_EQ(device_alloc.AvailablePages(), 2);
    ASSERT_NE(local_mamba, nullptr);
    EXPECT_TRUE(local_mamba->HasWorking());
    EXPECT_TRUE(local_mamba->HasCheckpoint());
    EXPECT_EQ(local_mamba->CheckpointPosition(), kInitialCheckpointPosition);
    EXPECT_EQ(mamba_alloc.AvailableSlots(), 2);

    const std::vector<std::int32_t> first_chunk_pages = local_kv->Pages();

    (void)hybrid_prefix_cache.StepCommit({
        .request_local_kv =
            RequestLocalKVStateRequest{
                .allocator = local_kv.get(),
                .acquire_tokens = 2,
            },
        .request_local_mamba =
            RequestLocalMambaStateRequest{
                .refresh_checkpoint_allocator = local_mamba.get(),
                .checkpoint_raw_position = kRefreshedCheckpointPosition,
            },
    });

    ASSERT_EQ(local_kv->Pages().size(), first_chunk_pages.size() + 1);
    EXPECT_EQ(local_kv->Pages().front(), first_chunk_pages.front());
    EXPECT_EQ(local_kv->TailPageAvailableTokens(), 1);
    EXPECT_EQ(device_alloc.AvailablePages(), 1);
    EXPECT_TRUE(local_mamba->HasWorking());
    EXPECT_TRUE(local_mamba->HasCheckpoint());
    EXPECT_EQ(local_mamba->CheckpointPosition(), kRefreshedCheckpointPosition);
    EXPECT_NE(local_mamba->CheckpointIndex(), local_mamba->WorkingIndex());
    EXPECT_EQ(mamba_alloc.AvailableSlots(), 2);

    local_kv.reset();
    local_mamba.reset();
    EXPECT_EQ(device_alloc.AvailablePages(), 4);
    EXPECT_EQ(mamba_alloc.AvailableSlots(), 4);
}

TEST(HybridPrefixCacheMambaCheckpointTest, RefreshReusesOldCheckpointWhenNoExtraSlotIsFree) {
    PageAllocator device_alloc(/*page_size=*/2, /*total_pages=*/4);
    PageAllocator host_alloc(/*page_size=*/2, /*total_pages=*/0);
    MambaChunkAllocator mamba_alloc(/*num_slots=*/2);
    KVPrefixCache prefix_cache(&device_alloc, &host_alloc);
    HybridPrefixCache hybrid_prefix_cache(prefix_cache, device_alloc, &mamba_alloc, /*mamba_cache_chunk_size=*/4);

    auto result = hybrid_prefix_cache.StepCommit({
        .request_local_mamba =
            RequestLocalMambaStateRequest{
                .create_allocator = true,
                .checkpoint_raw_position = 4,
            },
    });
    auto local_mamba = std::move(result.local_mamba_allocator);

    ASSERT_NE(local_mamba, nullptr);
    ASSERT_TRUE(local_mamba->HasWorking());
    ASSERT_TRUE(local_mamba->HasCheckpoint());
    const std::int32_t working_index = local_mamba->WorkingIndex();
    const std::int32_t checkpoint_index = local_mamba->CheckpointIndex();
    EXPECT_EQ(mamba_alloc.AvailableSlots(), 0);

    (void)hybrid_prefix_cache.StepCommit({
        .request_local_mamba =
            RequestLocalMambaStateRequest{
                .refresh_checkpoint_allocator = local_mamba.get(),
                .checkpoint_raw_position = 8,
            },
    });

    EXPECT_TRUE(local_mamba->HasWorking());
    EXPECT_TRUE(local_mamba->HasCheckpoint());
    EXPECT_EQ(local_mamba->WorkingIndex(), working_index);
    EXPECT_EQ(local_mamba->CheckpointIndex(), checkpoint_index);
    EXPECT_EQ(local_mamba->CheckpointPosition(), 8);
    EXPECT_EQ(mamba_alloc.AvailableSlots(), 0);
}

TEST(HybridPrefixCacheMambaCheckpointTest, RefreshFailureIsExplicitWhenNoCheckpointSlotCanBeMadeAvailable) {
    PageAllocator device_alloc(/*page_size=*/2, /*total_pages=*/4);
    PageAllocator host_alloc(/*page_size=*/2, /*total_pages=*/0);
    MambaChunkAllocator mamba_alloc(/*num_slots=*/1);
    KVPrefixCache prefix_cache(&device_alloc, &host_alloc);
    HybridPrefixCache hybrid_prefix_cache(prefix_cache, device_alloc, &mamba_alloc, /*mamba_cache_chunk_size=*/4);
    LocalMambaAllocator local_mamba(&mamba_alloc);
    ASSERT_TRUE(local_mamba.AllocateWorking());

    EXPECT_THROW((void)hybrid_prefix_cache.StepCommit({
                     .request_local_mamba =
                         RequestLocalMambaStateRequest{
                             .refresh_checkpoint_allocator = &local_mamba,
                             .checkpoint_raw_position = 4,
                         },
                 }),
                 std::logic_error);

    EXPECT_TRUE(local_mamba.HasWorking());
    EXPECT_FALSE(local_mamba.HasCheckpoint());
    EXPECT_EQ(local_mamba.CheckpointPosition(), -1);
    EXPECT_EQ(mamba_alloc.AvailableSlots(), 0);
}

TEST(HybridPrefixCacheMambaCheckpointTest, ForwardPublishAttachesAlignedPartialPageCheckpoint) {
    static constexpr std::int32_t kPageSize = 8;
    PageAllocator device_alloc(kPageSize, /*total_pages=*/4);
    PageAllocator host_alloc(kPageSize, /*total_pages=*/0);
    MambaChunkAllocator mamba_alloc(/*num_slots=*/4);
    KVPrefixCache prefix_cache(&device_alloc, &host_alloc);
    HybridPrefixCache hybrid_prefix_cache(prefix_cache, device_alloc, &mamba_alloc, /*mamba_cache_chunk_size=*/4);

    const auto tokens = MakeTokens(kPageSize);
    const auto full_pages = MakePagedTokenSpans(tokens, kPageSize);
    auto device_node_ref = std::make_unique<DeviceNodeRef>(prefix_cache.Match(token_vec_t{}).device.last_node);
    LocalKVAllocator local_kv(&device_alloc, kPageSize);
    LocalMambaAllocator local_mamba(&mamba_alloc);
    ASSERT_TRUE(local_mamba.AllocateWorking());
    ASSERT_TRUE(local_mamba.AllocateCheckpoint(/*raw_position=*/12));
    const std::int32_t checkpoint_index = local_mamba.CheckpointIndex();

    (void)hybrid_prefix_cache.StepCommit({
        .publish_device_prefix =
            DevicePrefixPublicationRequest{
                .full_paged_tokens = &full_pages,
                .device_node_ref = &device_node_ref,
                .local_kv_allocator = &local_kv,
                .local_mamba_allocator = &local_mamba,
                .chunk_begin = 4,
            },
    });

    ASSERT_NE(device_node_ref, nullptr);
    TreeNode* terminal = device_node_ref->Node();
    ASSERT_NE(terminal, nullptr);
    EXPECT_EQ(terminal->DepthInTokens(), static_cast<std::size_t>(kPageSize));
    ASSERT_TRUE(terminal->HasMamba());
    EXPECT_EQ(terminal->MambaSlotIndex(), checkpoint_index);
    EXPECT_FALSE(local_mamba.HasCheckpoint());

    auto match = MatchPrefix(hybrid_prefix_cache, tokens, kPageSize).compat_match;
    EXPECT_EQ(match.device.DepthInPage(), 1);
    EXPECT_EQ(match.mamba_cow_src_index, checkpoint_index);
}

TEST(HybridPrefixCacheMambaCheckpointTest, ForwardPublishDropsUnalignedPartialPageCheckpoint) {
    static constexpr std::int32_t kPageSize = 8;
    PageAllocator device_alloc(kPageSize, /*total_pages=*/4);
    PageAllocator host_alloc(kPageSize, /*total_pages=*/0);
    MambaChunkAllocator mamba_alloc(/*num_slots=*/4);
    KVPrefixCache prefix_cache(&device_alloc, &host_alloc);
    HybridPrefixCache hybrid_prefix_cache(prefix_cache, device_alloc, &mamba_alloc, /*mamba_cache_chunk_size=*/4);

    const auto tokens = MakeTokens(kPageSize);
    const auto full_pages = MakePagedTokenSpans(tokens, kPageSize);
    auto device_node_ref = std::make_unique<DeviceNodeRef>(prefix_cache.Match(token_vec_t{}).device.last_node);
    LocalKVAllocator local_kv(&device_alloc, kPageSize);
    LocalMambaAllocator local_mamba(&mamba_alloc);
    ASSERT_TRUE(local_mamba.AllocateWorking());
    ASSERT_TRUE(local_mamba.AllocateCheckpoint(/*raw_position=*/12));

    (void)hybrid_prefix_cache.StepCommit({
        .publish_device_prefix =
            DevicePrefixPublicationRequest{
                .full_paged_tokens = &full_pages,
                .device_node_ref = &device_node_ref,
                .local_kv_allocator = &local_kv,
                .local_mamba_allocator = &local_mamba,
                .chunk_begin = 2,
            },
    });

    ASSERT_NE(device_node_ref, nullptr);
    TreeNode* terminal = device_node_ref->Node();
    ASSERT_NE(terminal, nullptr);
    EXPECT_EQ(terminal->DepthInTokens(), static_cast<std::size_t>(kPageSize));
    EXPECT_FALSE(terminal->HasMamba());
    EXPECT_FALSE(local_mamba.HasCheckpoint());
    EXPECT_EQ(local_mamba.CheckpointPosition(), -1);
    EXPECT_EQ(mamba_alloc.AvailableSlots(), 3);

    auto match = MatchPrefix(hybrid_prefix_cache, tokens, kPageSize).compat_match;
    EXPECT_EQ(match.device.DepthInPage(), 0);
    EXPECT_EQ(match.mamba_cow_src_index, -1);
}

TEST(HybridPrefixCacheKVAllocationTest, DecodeRetractFailureReleasesPartialResources) {
    PageAllocator device_alloc(/*page_size=*/2, /*total_pages=*/2);
    PageAllocator host_alloc(/*page_size=*/2, /*total_pages=*/0);
    KVPrefixCache prefix_cache(&device_alloc, &host_alloc);
    HybridPrefixCache hybrid_prefix_cache(prefix_cache, device_alloc, /*allocator=*/nullptr,
                                          /*mamba_cache_chunk_size=*/4);

    {
        LocalKVAllocator local_kv(&device_alloc, /*num_tokens=*/1);
        const std::vector<std::int32_t> original_pages = local_kv.Pages();
        ASSERT_EQ(original_pages.size(), 1u);
        ASSERT_EQ(local_kv.TailPageAvailableTokens(), 1);
        ASSERT_EQ(device_alloc.AvailablePages(), 0);

        EXPECT_THROW((void)hybrid_prefix_cache.StepCommit({
                         .request_local_kv =
                             RequestLocalKVStateRequest{
                                 .allocator = &local_kv,
                                 .acquire_tokens = 3,
                             },
                     }),
                     std::runtime_error);

        EXPECT_EQ(local_kv.Pages(), original_pages);
        EXPECT_EQ(local_kv.TailPageAvailableTokens(), 1);
        EXPECT_EQ(device_alloc.AvailablePages(), 0);
    }

    EXPECT_EQ(device_alloc.AvailablePages(), 1);

    PageAllocator mamba_device_alloc(/*page_size=*/2, /*total_pages=*/4);
    PageAllocator mamba_host_alloc(/*page_size=*/2, /*total_pages=*/0);
    KVPrefixCache mamba_prefix_cache(&mamba_device_alloc, &mamba_host_alloc);
    MambaChunkAllocator mamba_alloc(/*num_slots=*/1);
    HybridPrefixCache mamba_hybrid(mamba_prefix_cache, mamba_device_alloc, &mamba_alloc,
                                   /*mamba_cache_chunk_size=*/4);
    EXPECT_THROW((void)mamba_hybrid.StepCommit({
                     .request_local_mamba =
                         RequestLocalMambaStateRequest{
                             .create_allocator = true,
                             .require_allocator = true,
                         },
                 }),
                 std::logic_error);
    EXPECT_EQ(mamba_alloc.AvailableSlots(), 1);
}

TEST_F(MambaCacheTest, KVEvictionTriggersMambaEviction) {
    auto tokens = MakeAlignedTokens(2, kPageSize);
    InsertKVAndMamba(tokens);

    auto match = prefix_cache_->Match(tokens);
    TreeNode* node = match.device.last_node;
    EXPECT_TRUE(node->HasMamba());

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
        hybrid_prefix_cache_ = std::make_unique<HybridPrefixCache>(*prefix_cache_, *device_alloc_, mamba_alloc_.get(),
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

    auto mismatch = MatchPrefix(*hybrid_prefix_cache_, tokens, kPageSize).compat_match;
    EXPECT_EQ(mismatch.host.DepthInPage(), 0);
    EXPECT_EQ(mismatch.device.DepthInPage(), 0);

    node->DetachMamba();
    auto host_slot = mamba_host_alloc_->Allocate();
    ASSERT_TRUE(host_slot.has_value());
    const std::int32_t host_idx = host_slot->Index();
    node->AttachMambaHost(std::make_unique<MambaSlot>(std::move(*host_slot)));

    auto match = MatchPrefix(*hybrid_prefix_cache_, tokens, kPageSize).compat_match;
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

    auto match = MatchPrefix(*hybrid_prefix_cache_, tokens4, kPageSize).compat_match;

    EXPECT_EQ(match.device.DepthInPage(), 2);
    EXPECT_EQ(match.host.DepthInPage(), 4);
    EXPECT_EQ(match.mamba_host_src_index, host_idx);
    EXPECT_EQ(match.mamba_cow_src_index, -1) << "deeper host hit must trigger Mamba L2 loadback";
}

TEST_F(MambaL2CacheTest, PrepareMambaLoadBackAllocatesDeviceSlotAndTransferPair) {
    auto tokens = MakeAlignedTokens(2, kPageSize);
    TreeNode* node = InsertHostKV(tokens);
    auto host_slot = mamba_host_alloc_->Allocate();
    ASSERT_TRUE(host_slot.has_value());
    const std::int32_t host_idx = host_slot->Index();
    node->AttachMambaHost(std::make_unique<MambaSlot>(std::move(*host_slot)));

    auto transfers = HybridPrefixCacheTestPeer::PrepareMambaDeviceLoadBack(*hybrid_prefix_cache_, {node});

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

    auto ancestor_transfers = HybridPrefixCacheTestPeer::PrepareMambaHostWriteBack(*hybrid_prefix_cache_, {ancestor});
    auto descendant_transfers =
        HybridPrefixCacheTestPeer::PrepareMambaHostWriteBack(*hybrid_prefix_cache_, {descendant});
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

    auto transfers = HybridPrefixCacheTestPeer::PrepareMambaHostWriteBack(*hybrid_prefix_cache_, {node});

    ASSERT_EQ(transfers.size(), 1u);
    EXPECT_EQ(transfers[0].kind, CacheKind::kMamba);
    EXPECT_EQ(transfers[0].src, device_idx);
    const std::int32_t host_idx = transfers[0].dst;
    EXPECT_FALSE(node->HasMambaOnHost()) << "host mamba must remain invisible until writeback ack";

    auto pending_match = MatchPrefix(*hybrid_prefix_cache_, tokens, kPageSize).compat_match;
    EXPECT_EQ(pending_match.host.DepthInPage(), 0);

    hybrid_prefix_cache_->OnMambaHostWriteBackDone(node);

    ASSERT_TRUE(node->HasMambaOnHost());
    EXPECT_EQ(node->MambaHostSlotIndex(), host_idx);
    EXPECT_FALSE(node->HasMamba()) << "idle device mamba copy should demote once host writeback is acknowledged";
    auto host_match = MatchPrefix(*hybrid_prefix_cache_, tokens, kPageSize).compat_match;
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

    auto transfers = HybridPrefixCacheTestPeer::PrepareMambaHostWriteBack(*hybrid_prefix_cache_, {node});
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

TEST_F(MambaCacheTest, StepCommitFinishStateInsertsKvPagesHashesAndMamba) {
    auto tokens = MakeAlignedTokens(3, kPageSize);
    auto page_hashes = MakePageHashes(/*num_pages=*/3);
    TreeNode* root = prefix_cache_->Match(token_vec_t{}).device.last_node;

    LocalKVAllocator local_kv(device_alloc_.get(), static_cast<std::int32_t>(tokens.size()));
    LocalMambaAllocator local_mamba(mamba_alloc_.get());
    ASSERT_TRUE(local_mamba.AllocateWorking());
    const std::int32_t working_index = local_mamba.WorkingIndex();
    ASSERT_TRUE(local_mamba.AllocateCheckpoint());
    const std::int32_t checkpoint_index = local_mamba.CheckpointIndex();

    const auto full_pages = PagedTokenSpans(tokens);
    StepCommitRequest request{
        .publish_finished_request =
            FinishedRequestPublicationRequest{
                .full_paged_tokens = &full_pages,
                .current_device_node = root,
                .local_kv_allocator = &local_kv,
                .local_mamba_allocator = &local_mamba,
                .page_hashes = &page_hashes,
            },
    };
    MatchResult match = hybrid_prefix_cache_->StepCommit(std::move(request)).match_result;

    EXPECT_EQ(match.device.DepthInPage(), 3);
    TreeNode* terminal = match.device.last_node;
    ASSERT_NE(terminal, nullptr);
    ASSERT_TRUE(terminal->HasMamba());
    EXPECT_EQ(terminal->MambaSlotIndex(), checkpoint_index);
    EXPECT_EQ(terminal->PageHashes(), page_hashes);
    EXPECT_TRUE(local_kv.Pages().empty());
    EXPECT_FALSE(local_mamba.HasCheckpoint());
    EXPECT_TRUE(local_mamba.HasWorking());
    EXPECT_EQ(local_mamba.WorkingIndex(), working_index);
}

TEST_F(MambaCacheTest, StepCommitFinishStatePublishesWorkingWhenCheckpointWasDropped) {
    auto tokens = MakeAlignedTokens(3, kPageSize);
    auto page_hashes = MakePageHashes(/*num_pages=*/3);
    TreeNode* root = prefix_cache_->Match(token_vec_t{}).device.last_node;

    auto result = hybrid_prefix_cache_->StepCommit({
        .request_local_mamba =
            RequestLocalMambaStateRequest{
                .create_allocator = true,
                .checkpoint_raw_position = 4,
            },
    });
    auto local_mamba = std::move(result.local_mamba_allocator);
    ASSERT_NE(local_mamba, nullptr);
    ASSERT_TRUE(local_mamba->HasWorking());
    ASSERT_TRUE(local_mamba->HasCheckpoint());
    const std::int32_t working_index = local_mamba->WorkingIndex();

    (void)hybrid_prefix_cache_->StepCommit({
        .request_local_mamba =
            RequestLocalMambaStateRequest{
                .refresh_checkpoint_allocator = local_mamba.get(),
                .checkpoint_raw_position = 6,
            },
    });
    ASSERT_TRUE(local_mamba->HasWorking());
    ASSERT_FALSE(local_mamba->HasCheckpoint());

    LocalKVAllocator local_kv(device_alloc_.get(), static_cast<std::int32_t>(tokens.size()));
    const auto full_pages = PagedTokenSpans(tokens);
    MatchResult match = hybrid_prefix_cache_
                            ->StepCommit({
                                .publish_finished_request =
                                    FinishedRequestPublicationRequest{
                                        .full_paged_tokens = &full_pages,
                                        .current_device_node = root,
                                        .local_kv_allocator = &local_kv,
                                        .local_mamba_allocator = local_mamba.get(),
                                        .page_hashes = &page_hashes,
                                    },
                            })
                            .match_result;

    TreeNode* terminal = match.device.last_node;
    ASSERT_NE(terminal, nullptr);
    ASSERT_TRUE(terminal->HasMamba());
    EXPECT_EQ(terminal->MambaSlotIndex(), working_index);
    EXPECT_FALSE(local_mamba->HasWorking());
    EXPECT_FALSE(local_mamba->HasCheckpoint());
}

TEST_F(MambaCacheTest, StepCommitRetractPublicationInsertsKvAndReturnsRawStateRecoveryMatch) {
    auto tokens = MakeAlignedTokens(3, kPageSize);
    token_vec_t prefix_tokens(tokens.begin(), tokens.begin() + kPageSize);
    TreeNode* prefix = InsertKVOnly(prefix_tokens);
    ASSERT_NE(prefix, nullptr);
    ASSERT_FALSE(prefix->HasMamba());

    const auto full_pages = PagedTokenSpans(tokens);
    StepCommitRequest count_request{
        .plan_device_prefix_insertion =
            DevicePrefixInsertionPlanRequest{
                .full_paged_tokens = &full_pages,
                .current_device_node = prefix,
            },
    };
    EXPECT_EQ(hybrid_prefix_cache_->StepCommit(std::move(count_request)).device_insert_page_count, 2);

    const std::vector<std::int32_t> existing_pages = DevicePagesFromRoot(prefix);
    auto pages_to_insert = device_alloc_->Allocate(/*num_pages=*/2);
    const std::vector<std::int32_t> inserted_pages = pages_to_insert.Ids();

    StepCommitRequest request{
        .publish_device_prefix_insertion =
            DevicePrefixInsertionRequest{
                .full_paged_tokens = &full_pages,
                .current_device_node = prefix,
                .pages_to_insert = std::move(pages_to_insert),
            },
    };
    MatchResult match = hybrid_prefix_cache_->StepCommit(std::move(request)).match_result;

    EXPECT_EQ(match.device.DepthInPage(), 3);
    EXPECT_EQ(match.host.DepthInPage(), 0);
    EXPECT_EQ(match.mamba_cow_src_index, -1);
    EXPECT_EQ(match.mamba_branching_seqlen, -1);

    std::vector<std::int32_t> expected_pages = existing_pages;
    expected_pages.insert(expected_pages.end(), inserted_pages.begin(), inserted_pages.end());
    EXPECT_EQ(DevicePagesFromRoot(match.device.last_node), expected_pages);

    // The retract facade intentionally returns the raw KV state-recovery match
    // used for host writeback planning, not the Mamba-capped hybrid match.
    MatchResult hybrid_match = hybrid_prefix_cache_->MatchPrefix(full_pages, MatchIntent::StateRecovery).compat_match;
    EXPECT_EQ(hybrid_match.device.DepthInPage(), 0);

    TreeNode* terminal = match.device.last_node;
    ASSERT_NE(terminal, nullptr);
    ASSERT_FALSE(terminal->HasMamba());

    auto local_mamba = std::make_unique<LocalMambaAllocator>(mamba_alloc_.get());
    ASSERT_TRUE(local_mamba->AllocateWorking());
    ASSERT_TRUE(local_mamba->AllocateCheckpoint());
    const std::int32_t checkpoint_index = local_mamba->CheckpointIndex();

    hybrid_prefix_cache_->StepCommit({
        .publish_tree_owned_request_state =
            TreeOwnedRequestStatePublicationRequest{
                .terminal = terminal,
                .local_mamba_allocator_owner = &local_mamba,
            },
    });

    EXPECT_EQ(local_mamba, nullptr);
    ASSERT_TRUE(terminal->HasMamba());
    EXPECT_EQ(terminal->MambaSlotIndex(), checkpoint_index);
    EXPECT_EQ(mamba_alloc_->AvailableSlots(), kMambaSlots - 1);
}

TEST_F(MambaCacheTest, AdmitMambaCapacityForPrefillDecodeAndRecovery) {
    std::int32_t fill_start = 1000;
    auto saturate_mamba_slots = [this, &fill_start]() {
        auto nodes = FillMambaSlots(fill_start);
        fill_start += kMambaSlots * 10;
        EXPECT_EQ(mamba_alloc_->AvailableSlots(), 0);
        return nodes;
    };

    saturate_mamba_slots();
    MatchResult prefill_first_match{};
    std::map<std::string, std::int32_t> simulated_free;
    AdmissionVerdict prefill_first = hybrid_prefix_cache_->Admit(
        AdmissionRequest{
            .request_id = "r-prefill-first",
            .kind = AdmissionRequestKind::kPrefillFirstChunk,
            .tokens_this_round = 5,
            .first_raw_position_of_op = 0,
            .target_raw_tokens_exclusive = 5,
            .compat_match = &prefill_first_match,
        },
        simulated_free);

    EXPECT_TRUE(prefill_first.admitted);
    ASSERT_TRUE(prefill_first.mamba_branching_seqlen.has_value());
    EXPECT_EQ(*prefill_first.mamba_branching_seqlen, 4);
    EXPECT_EQ(prefill_first_match.mamba_branching_seqlen, -1);
    EXPECT_GE(mamba_alloc_->AvailableSlots(), 2);

    saturate_mamba_slots();
    simulated_free.clear();
    AdmissionVerdict prefill_continue = hybrid_prefix_cache_->Admit(
        AdmissionRequest{
            .request_id = "r-prefill-continue",
            .kind = AdmissionRequestKind::kPrefillChunk,
            .first_raw_position_of_op = 4,
            .target_raw_tokens_exclusive = 8,
        },
        simulated_free);

    EXPECT_TRUE(prefill_continue.admitted);
    EXPECT_GE(mamba_alloc_->AvailableSlots(), 1);

    auto decode_nodes = saturate_mamba_slots();
    simulated_free.clear();
    AdmissionVerdict decode = hybrid_prefix_cache_->Admit(
        AdmissionRequest{
            .request_id = "r-decode",
            .kind = AdmissionRequestKind::kDecodeChunk,
            .first_raw_position_of_op = 8,
            .target_raw_tokens_exclusive = 9,
        },
        simulated_free);

    EXPECT_TRUE(decode.admitted);
    EXPECT_EQ(mamba_alloc_->AvailableSlots(), 0);

    auto recovery_it = std::find_if(decode_nodes.begin(), decode_nodes.end(),
                                    [](const TreeNode* node) { return node != nullptr && node->HasMamba(); });
    ASSERT_NE(recovery_it, decode_nodes.end());
    TreeNode* recovery_source = *recovery_it;

    simulated_free.clear();
    MatchResult recovery_match{};
    AdmissionVerdict recovery = hybrid_prefix_cache_->Admit(
        AdmissionRequest{
            .request_id = "r-retracted",
            .kind = AdmissionRequestKind::kDecodeFromRetracted,
            .target_raw_tokens_exclusive = 8,
            .compat_match = &recovery_match,
            .protected_recovery_node = recovery_source,
        },
        simulated_free);

    EXPECT_TRUE(recovery.admitted);
    EXPECT_TRUE(recovery_source->HasMamba());
    EXPECT_GE(mamba_alloc_->AvailableSlots(), 2);
}

}  // namespace tokenspeed::test
