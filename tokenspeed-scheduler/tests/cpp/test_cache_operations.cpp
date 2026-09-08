#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache_test_access.h"
#include "cache/tier/transfer.h"
#include "cache/tier/transfer_manager.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"

namespace tokenspeed::test {

static_assert(std::is_aggregate_v<WriteBackOperation>);
static_assert(std::is_aggregate_v<LoadBackOperation>);

TEST(CacheOperationTest, WriteBackDeduplicatesTransfersAcrossBatch) {
    WriteBackOperation op;
    op.op_id = 7;
    op.transfers = {
        CacheTransfer{0, 1, 11},
        CacheTransfer{0, 2, 22},
        CacheTransfer{0, 1, 11},
    };
    WriteBackOperation duplicate;
    duplicate.op_id = 8;
    duplicate.transfers = {CacheTransfer{0, 2, 22}, CacheTransfer{0, 3, 33}};

    WriteBackBatch batch({op, duplicate});

    ASSERT_EQ(batch.op_ids, std::vector<std::uint32_t>({7, 8}));
    EXPECT_EQ(batch.group_ids[0], std::vector<std::uint32_t>({0, 0}));
    EXPECT_EQ(batch.src_pages[0], std::vector<std::int32_t>({1, 2}));
    EXPECT_EQ(batch.dst_pages[0], std::vector<std::int32_t>({11, 22}));
    EXPECT_EQ(batch.src_pages[1], std::vector<std::int32_t>({3}));
    EXPECT_EQ(batch.dst_pages[1], std::vector<std::int32_t>({33}));
}

TEST(CacheOperationTest, SamePagesInDifferentGroupsAreDistinctTransfers) {
    WriteBackOperation op;
    op.op_id = 10;
    op.transfers = {
        CacheTransfer{.group_id = 0, .source_page = 1, .destination_page = 11},
        CacheTransfer{.group_id = 1, .source_page = 1, .destination_page = 11},
    };

    WriteBackBatch batch({op});

    EXPECT_EQ(batch.group_ids[0], std::vector<std::uint32_t>({0, 1}));
    EXPECT_EQ(batch.src_pages[0], std::vector<std::int32_t>({1, 1}));
    EXPECT_EQ(batch.dst_pages[0], std::vector<std::int32_t>({11, 11}));
}

TEST(CacheOperationTest, LoadBackPreservesTransferOrder) {
    LoadBackOperation op;
    op.op_id = 9;
    op.transfers = {
        CacheTransfer{0, 10, 20},
        CacheTransfer{0, 30, 40},
    };

    LoadBackBatch batch({op});

    ASSERT_EQ(batch.op_ids, std::vector<std::uint32_t>({9}));
    EXPECT_EQ(batch.group_ids[0], std::vector<std::uint32_t>({0, 0}));
    EXPECT_EQ(batch.src_pages[0], std::vector<std::int32_t>({10, 30}));
    EXPECT_EQ(batch.dst_pages[0], std::vector<std::int32_t>({20, 40}));
}

TEST(CacheOperationTest, HostCacheAndContinuousStreamingAreSeparatePolicies) {
    SchedulerConfig config;
    config.host_allocator.total_pages = 2;

    config.role = Role::kFused;
    EXPECT_TRUE(config.HasHostCache());
    EXPECT_TRUE(config.StreamsDeviceCacheToHost());

    config.role = Role::kD;
    EXPECT_TRUE(config.HasHostCache());
    EXPECT_FALSE(config.StreamsDeviceCacheToHost());

    config.disable_l2_cache = true;
    EXPECT_FALSE(config.HasHostCache());
    EXPECT_FALSE(config.StreamsDeviceCacheToHost());
}

TEST(CacheOperationTest, DecodeCanStartWithoutHostL2) {
    const auto make_config = [] {
        SchedulerConfig config;
        config.prefix_granularity = 2;
        config.device_allocator.total_pages = 4;
        config.host_allocator.total_pages = 4;
        config.max_scheduled_tokens = 2;
        config.max_batch_size = 1;
        config.role = Role::kD;
        config.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full",
            .block_granularity = 2,
            .total_pages = 4,
            .retention = CacheGroupConfig::Retention::FullHistory,
            .family = CacheGroupFamily::History,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        return config;
    };

    SchedulerConfig disabled = make_config();
    disabled.disable_l2_cache = true;
    EXPECT_NO_THROW(Scheduler{std::move(disabled)});

    SchedulerConfig empty = make_config();
    empty.host_allocator.total_pages = 1;
    EXPECT_NO_THROW(Scheduler{std::move(empty)});
}

TEST(CacheOperationTest, DeviceRequestLimitDoesNotDependOnHostCapacity) {
    const auto make_config = [](std::int32_t host_pages) {
        SchedulerConfig config;
        config.prefix_granularity = 2;
        config.device_allocator.total_pages = 9;
        config.host_allocator.total_pages = host_pages;
        config.max_scheduled_tokens = 8;
        config.max_batch_size = 2;
        config.role = Role::kD;
        config.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full",
            .block_granularity = 2,
            .total_pages = 9,
            .retention = CacheGroupConfig::Retention::FullHistory,
            .family = CacheGroupFamily::History,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        return config;
    };

    Scheduler small_host{make_config(/*host_pages=*/2)};
    Scheduler large_host{make_config(/*host_pages=*/64)};

    EXPECT_EQ(small_host.MaxSingleRequestTokens(), large_host.MaxSingleRequestTokens());
}

TEST(CacheOperationTest, RetractionStoreIsBestEffortAndUsesOrdinaryTransferPins) {
    BlockPool device_pool{2};
    BlockPool host_pool{1};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*stream_device_cache_to_host=*/false);
    TierTransferManager transfers{coordinator};

    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .num_tokens = 2}};
    auto admission = coordinator.Admit(coordinator.ProbePrefix({}), demands);
    ASSERT_TRUE(admission);
    const std::array<std::string, 1> hashes{"h0"};
    coordinator.CacheFullBlocks(tables, hashes, admission->access_epoch);

    coordinator.QueueCachedBlocksForStore(hashes);
    auto write_back = transfers.StartPendingStores();
    ASSERT_TRUE(write_back);
    coordinator.Free(tables);
    // The ticket pins no Device source: the runtime orders the D2H copy on
    // the forward thread's stream ahead of any reuse, so the cache stays clearable.
    EXPECT_TRUE(coordinator.ClearDeviceCache());

    transfers.CompleteWriteBack(write_back->op_id);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(CacheKey{.group_id = 0, .content_hash = "h0"}));
}

TEST(CacheOperationTest, HostDestinationCannotBeReusedBeforeWriteBackAck) {
    BlockPool device_pool{2};
    BlockPool host_pool{1};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*stream_device_cache_to_host=*/false);
    TierTransferManager transfers{coordinator};
    const auto cache_device = [&](const CacheKey& key) {
        CacheBlockRef block = device_pool.AcquireBlock(key.group_id, /*packing=*/1);
        ASSERT_TRUE(block);
        coordinator.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id))
            .Register(device_pool, block, key, /*access_epoch=*/1);
    };

    const CacheKey first_key{.group_id = 0, .content_hash = "first"};
    cache_device(first_key);
    const std::array first_hashes{first_key.content_hash};
    coordinator.QueueCachedBlocksForStore(first_hashes);
    auto first = transfers.StartPendingStores();
    ASSERT_TRUE(first);
    ASSERT_EQ(first->transfers.size(), 1u);
    const std::int32_t destination = first->transfers.front().destination_page;

    const CacheKey second_key{.group_id = 0, .content_hash = "second"};
    cache_device(second_key);
    const std::array second_hashes{second_key.content_hash};
    coordinator.QueueCachedBlocksForStore(second_hashes);
    EXPECT_FALSE(transfers.StartPendingStores());

    transfers.CompleteWriteBack(first->op_id);
    coordinator.QueueCachedBlocksForStore(second_hashes);
    auto second = transfers.StartPendingStores();
    ASSERT_TRUE(second);
    ASSERT_EQ(second->transfers.size(), 1u);
    EXPECT_EQ(second->transfers.front().destination_page, destination);
    transfers.CompleteWriteBack(second->op_id);
}

TEST(CacheOperationTest, RetractionStoreSkipsWhenHostHasNoPlacement) {
    BlockPool device_pool{2};
    BlockPool host_pool{1};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*stream_device_cache_to_host=*/false);
    TierTransferManager transfers{coordinator};

    CacheBlockRef host_pin = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(host_pin);
    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .num_tokens = 2}};
    auto admission = coordinator.Admit(coordinator.ProbePrefix({}), demands);
    ASSERT_TRUE(admission);
    const std::array<std::string, 1> hashes{"h0"};
    coordinator.CacheFullBlocks(tables, hashes, admission->access_epoch);

    coordinator.QueueCachedBlocksForStore(hashes);
    EXPECT_FALSE(transfers.StartPendingStores());
    coordinator.Free(tables);
    EXPECT_TRUE(coordinator.ClearDeviceCache());
}

TEST(CacheOperationTest, PendingStoresUseBatchHostAllocation) {
    BlockPool device_pool{3};
    BlockPool host_pool{1};
    const std::array specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 2,
            .block_granularity = 2,
        },
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 1,
            .block_granularity = 2,
        },
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*stream_device_cache_to_host=*/false);
    TierTransferManager transfers{coordinator};

    const auto cache_block = [&](BlockPool& pool, const CacheKey& key) {
        GroupAllocator& allocator = coordinator.Allocator(static_cast<std::int32_t>(key.group_id));
        CacheBlockRef block = pool.AcquireBlock(key.group_id, allocator.CacheBlocksPerLcmBlock());
        EXPECT_TRUE(block);
        const std::int32_t page = allocator.ResolveCacheBlockId(block->Location());
        coordinator.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id))
            .Register(pool, block, key, /*access_epoch=*/1);
        block.reset();
        return page;
    };

    cache_block(host_pool, CacheKey{.group_id = 0, .content_hash = "old-0"});
    cache_block(host_pool, CacheKey{.group_id = 0, .content_hash = "old-1"});

    const CacheKey group_one{.group_id = 1, .content_hash = "new-1"};
    cache_block(device_pool, group_one);
    const std::array group_one_hashes{group_one.content_hash};
    coordinator.QueueCachedBlocksForStore(group_one_hashes);

    const CacheKey group_zero_first{.group_id = 0, .content_hash = "new-0"};
    const CacheKey group_zero_second{.group_id = 0, .content_hash = "new-2"};
    const std::int32_t first_source = cache_block(device_pool, group_zero_first);
    const std::int32_t second_source = cache_block(device_pool, group_zero_second);
    const std::array group_zero_hashes{group_zero_first.content_hash, group_zero_second.content_hash};
    coordinator.QueueCachedBlocksForStore(group_zero_hashes);

    auto write_back = transfers.StartPendingStores();

    ASSERT_TRUE(write_back);
    ASSERT_EQ(write_back->transfers.size(), 2u);
    EXPECT_EQ(write_back->transfers[0].group_id, 0u);
    EXPECT_EQ(write_back->transfers[0].source_page, first_source);
    EXPECT_EQ(write_back->transfers[1].group_id, 0u);
    EXPECT_EQ(write_back->transfers[1].source_page, second_source);

    transfers.CompleteWriteBack(write_back->op_id);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(group_one));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(group_zero_first));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(group_zero_second));
}

TEST(CacheOperationTest, RetractionReleaseEstimateExcludesBlocksOwnedByAnotherRequest) {
    BlockPool device_pool{2};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool,
                                                   /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);

    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .num_tokens = 4}};
    auto admission = coordinator.Admit(coordinator.ProbePrefix({}), demands);
    ASSERT_TRUE(admission);
    const std::array<std::string, 2> hashes{"h0", "h1"};
    coordinator.CacheFullBlocks(tables, hashes, admission->access_epoch);

    CacheBlockRef other_request_ref = tables[0].Blocks()[1];
    EXPECT_EQ(coordinator.NumNewlyReleasableLcmBlocks(tables), 1);
    other_request_ref.reset();
    EXPECT_EQ(coordinator.NumNewlyReleasableLcmBlocks(tables), 2);
}

TEST(CacheOperationTest, DecodeRejectsRequestWhoseMaximumExtentCannotFitDevice) {
    SchedulerConfig config;
    config.prefix_granularity = 2;
    config.device_allocator.total_pages = 4;
    config.host_allocator.total_pages = 10;
    config.max_scheduled_tokens = 8;
    config.max_batch_size = 2;
    config.role = Role::kD;
    config.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full",
        .block_granularity = 2,
        .total_pages = 4,
        .retention = CacheGroupConfig::Retention::FullHistory,
        .family = CacheGroupFamily::History,
        .transfer_policy = CacheTransferPolicy::FullSuffix,
    });
    Scheduler scheduler{std::move(config)};
    ASSERT_EQ(scheduler.MaxSingleRequestTokens(), 6);
    RequestSpec spec{
        .request_id = "too-large-for-device",
        .tokens = {1, 2, 3, 4},
        .max_new_tokens = 4,
    };

    EXPECT_THROW(scheduler.SubmitRequests({spec}), std::invalid_argument);
}

TEST(CacheOperationTest, PrefillAcceptsPromptThatFitsWithoutReservingDecodeTokens) {
    SchedulerConfig config;
    config.prefix_granularity = 2;
    config.device_allocator.total_pages = 4;
    config.host_allocator.total_pages = 10;
    config.max_scheduled_tokens = 8;
    config.max_batch_size = 2;
    config.role = Role::kP;
    config.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full",
        .block_granularity = 2,
        .total_pages = 4,
        .retention = CacheGroupConfig::Retention::FullHistory,
        .family = CacheGroupFamily::History,
        .transfer_policy = CacheTransferPolicy::FullSuffix,
    });
    Scheduler scheduler{std::move(config)};
    ASSERT_EQ(scheduler.MaxSingleRequestTokens(), 6);
    RequestSpec spec{
        .request_id = "prefill-only-capacity",
        .tokens = {1, 2, 3, 4, 5, 6},
        .max_new_tokens = 100,
    };

    EXPECT_NO_THROW(scheduler.SubmitRequests({spec}));
}

TEST(CacheOperationTest, L3StorageRequiresHostCache) {
    SchedulerConfig config;
    config.prefix_granularity = 2;
    config.device_allocator.total_pages = 4;
    config.host_allocator.total_pages = 1;
    config.max_scheduled_tokens = 2;
    config.max_batch_size = 1;
    config.enable_l3_storage = true;
    config.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full",
        .block_granularity = 2,
        .total_pages = 4,
        .retention = CacheGroupConfig::Retention::FullHistory,
        .family = CacheGroupFamily::History,
    });
    EXPECT_THROW(Scheduler{std::move(config)}, std::invalid_argument);
}

TEST(CacheOperationTest, L3StorageAcceptsHostCache) {
    SchedulerConfig config;
    config.prefix_granularity = 2;
    config.device_allocator.total_pages = 4;
    config.host_allocator.total_pages = 8;
    config.max_scheduled_tokens = 2;
    config.max_batch_size = 1;
    config.enable_l3_storage = true;
    config.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full",
        .block_granularity = 2,
        .total_pages = 4,
        .retention = CacheGroupConfig::Retention::FullHistory,
        .family = CacheGroupFamily::History,
    });
    EXPECT_NO_THROW(Scheduler{std::move(config)});
}

TEST(CacheOperationTest, L3StorageHitsAllocateHostPrefetch) {
    BlockPool device_pool{4};
    BlockPool host_pool{4};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    ASSERT_TRUE(coordinator.EnablesL3Storage());

    const CacheKey key{.group_id = 0, .content_hash = "h0"};
    coordinator.RegisterStorageKeys(std::array{key});
    EXPECT_TRUE(coordinator.ContainsStorageKey(key));

    auto probe = coordinator.ProbePrefix(std::array<std::string, 1>{"h0"});
    EXPECT_EQ(probe.host.num_common_tokens, 2);

    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .num_tokens = 2}};
    auto admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    ASSERT_EQ(admission->load_pairs.size(), 1u);
    EXPECT_TRUE(admission->load_pairs[0].prefetch_from_storage);
    EXPECT_EQ(admission->load_pairs[0].key.content_hash, "h0");
}

TEST(CacheOperationTest, FailedLoadBackDoesNotPublishPrefetchedHost) {
    BlockPool device_pool{4};
    BlockPool host_pool{4};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    const CacheKey key{.group_id = 0, .content_hash = "h0"};
    coordinator.RegisterStorageKeys(std::array{key});

    auto probe = coordinator.ProbePrefix(std::array<std::string, 1>{"h0"});
    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .num_tokens = 2}};
    auto admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    ASSERT_EQ(admission->load_pairs.size(), 1u);

    TierTransferManager transfers(coordinator);
    LoadBackOperation op = transfers.StartPrefixLoad(std::move(admission->load_pairs));
    transfers.CompleteLoadBack(op.op_id, false);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 0);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(key));
    EXPECT_FALSE(coordinator.AcquireDeviceCachedBlock(key))
        << "failed L3 prefetch must not leave empty Device prefix hits";
}

TEST(CacheOperationTest, SuccessfulLoadBackPublishesPrefetchedHost) {
    BlockPool device_pool{4};
    BlockPool host_pool{4};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    const CacheKey key{.group_id = 0, .content_hash = "h0"};
    coordinator.RegisterStorageKeys(std::array{key});

    auto probe = coordinator.ProbePrefix(std::array<std::string, 1>{"h0"});
    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .num_tokens = 2}};
    auto admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    ASSERT_EQ(admission->load_pairs.size(), 1u);

    TierTransferManager transfers(coordinator);
    LoadBackOperation op = transfers.StartPrefixLoad(std::move(admission->load_pairs));
    transfers.CompleteLoadBack(op.op_id, true);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key));
    EXPECT_TRUE(coordinator.AcquireDeviceCachedBlock(key))
        << "successful L3 prefetch must publish filled Device destinations";
}

TEST(CacheOperationTest, MixedHostAndL3LoadBackPublishesEveryDeviceDestination) {
    BlockPool device_pool{8};
    BlockPool host_pool{4};
    const std::array specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 1,
            .block_granularity = 2,
        },
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 1,
            .block_granularity = 2,
        },
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    CacheBlockRef host_block = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(host_block);
    const CacheKey host_key{.group_id = 0, .content_hash = "h0"};
    const CacheKey l3_key{.group_id = 1, .content_hash = "h0"};
    coordinator.CacheHostBlock(host_block, host_key);
    host_block.reset();
    coordinator.RegisterStorageKeys(std::array{l3_key});

    auto probe = coordinator.ProbePrefix(std::array<std::string, 1>{"h0"});
    EXPECT_EQ(probe.host.num_common_tokens, 2);
    std::vector<BlockTable> tables(2);
    std::vector<GroupDemand> demands{
        {.table = &tables[0], .num_tokens = 2},
        {.table = &tables[1], .num_tokens = 2},
    };
    auto admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    ASSERT_EQ(admission->load_pairs.size(), 2u);
    EXPECT_EQ(std::count_if(admission->load_pairs.begin(), admission->load_pairs.end(),
                            [](const BlockTransfer& transfer) { return transfer.prefetch_from_storage; }),
              1);

    TierTransferManager transfers(coordinator);
    LoadBackOperation op = transfers.StartPrefixLoad(std::move(admission->load_pairs));
    transfers.CompleteLoadBack(op.op_id, true);
    coordinator.Free(tables);
    EXPECT_TRUE(coordinator.AcquireDeviceCachedBlock(host_key))
        << "Host-warm sibling of an L3 prefetch must still enter the Device index";
    EXPECT_TRUE(coordinator.AcquireDeviceCachedBlock(l3_key));
    auto retry = coordinator.ProbePrefix(std::array<std::string, 1>{"h0"});
    EXPECT_EQ(retry.device.num_common_tokens, 2);
}

TEST(CacheOperationTest, HostHitsWithoutL3DoNotTagPrefetch) {
    BlockPool device_pool{4};
    BlockPool host_pool{4};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool,
                                                   /*enable_l3_storage=*/false, &host_pool,
                                                   /*stream_device_cache_to_host=*/false);
    ASSERT_FALSE(coordinator.EnablesL3Storage());

    CacheBlockRef host_block = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(host_block);
    const CacheKey key{.group_id = 0, .content_hash = "h0"};
    coordinator.CacheHostBlock(host_block, key);
    host_block.reset();

    auto probe = coordinator.ProbePrefix(std::array<std::string, 1>{"h0"});
    EXPECT_EQ(probe.host.num_common_tokens, 2);

    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .num_tokens = 2}};
    auto admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    ASSERT_EQ(admission->load_pairs.size(), 1u);
    EXPECT_FALSE(admission->load_pairs[0].prefetch_from_storage);
    EXPECT_TRUE(admission->load_pairs[0].key.content_hash.empty());

    TierTransferManager transfers(coordinator);
    LoadBackOperation op = transfers.StartPrefixLoad(std::move(admission->load_pairs));
    transfers.CompleteLoadBack(op.op_id, true);
}

TEST(CacheOperationTest, L3StorageMissCanBeUnregistered) {
    BlockPool device_pool{4};
    BlockPool host_pool{4};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool,
                                                   /*enable_l3_storage=*/true, &host_pool,
                                                   /*stream_device_cache_to_host=*/true);
    const CacheKey key{.group_id = 0, .content_hash = "h0"};
    coordinator.RegisterStorageKeys(std::array{key});
    ASSERT_TRUE(coordinator.ContainsStorageKey(key));

    coordinator.UnregisterStorageKeys(std::array{key});

    EXPECT_FALSE(coordinator.ContainsStorageKey(key));
    EXPECT_EQ(coordinator.ProbePrefix(std::array<std::string, 1>{"h0"}).host.num_common_tokens, 0);
}

TEST(CacheOperationTest, L3UnregisterPrunesStorageKeyOrder) {
    BlockPool device_pool{4};
    BlockPool host_pool{4};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool,
                                                   /*enable_l3_storage=*/true, &host_pool,
                                                   /*stream_device_cache_to_host=*/true);
    const CacheKey key{.group_id = 0, .content_hash = "h0"};
    for (int cycle = 0; cycle < 8; ++cycle) {
        coordinator.RegisterStorageKeys(std::array{key});
        ASSERT_TRUE(coordinator.ContainsStorageKey(key));
        coordinator.UnregisterStorageKeys(std::array{key});
        EXPECT_FALSE(coordinator.ContainsStorageKey(key));
        EXPECT_EQ(CacheCoordinatorTestAccess::NumStorageKeyOrder(coordinator), 0u)
            << "unregister must drop LRU tombstones, not leave them until the live set hits capacity";
    }
    coordinator.RegisterStorageKeys(std::array{key});
    EXPECT_EQ(coordinator.NumStorageKeys(), 1);
    EXPECT_EQ(CacheCoordinatorTestAccess::NumStorageKeyOrder(coordinator), 1u);
}

TEST(CacheOperationTest, MultiGroupL3AllocationFailureTrimsEarlierPins) {
    BlockPool device_pool{8};
    BlockPool host_pool{1};
    const std::array specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 1,
            .block_granularity = 2,
        },
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 1,
            .block_granularity = 2,
        },
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool,
                                                   /*enable_l3_storage=*/true, &host_pool,
                                                   /*stream_device_cache_to_host=*/true);
    const std::array keys{
        CacheKey{.group_id = 0, .content_hash = "h0"},
        CacheKey{.group_id = 1, .content_hash = "h0"},
    };
    coordinator.RegisterStorageKeys(keys);

    auto match = MatchPrefixForTest(coordinator, std::array<std::string, 1>{"h0"});

    EXPECT_EQ(match.host.num_common_tokens, 0);
    ASSERT_EQ(match.host.per_group.size(), 2u);
    EXPECT_TRUE(match.host.per_group[0].blocks.empty());
    EXPECT_TRUE(match.host.per_group[1].blocks.empty());
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 1);
}

TEST(CacheOperationTest, ExpandPrefixKeysCoversGroupsAndOffsets) {
    BlockPool device_pool{4};
    BlockPool host_pool{4};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    const std::vector<CacheKey> keys = coordinator.ExpandPrefixKeys(std::array<std::string, 1>{"h0"});
    ASSERT_EQ(keys.size(), 2u);
    EXPECT_EQ(keys[0].content_hash, "h0");
    EXPECT_EQ(keys[0].page_offset, 0);
    EXPECT_EQ(keys[1].page_offset, 1);
}

TEST(CacheOperationTest, WriteBackBatchCarriesStorageKeys) {
    WriteBackOperation op;
    op.op_id = 3;
    op.transfers = {CacheTransfer{
        .group_id = 0,
        .source_page = 1,
        .destination_page = 11,
        .content_hash = "h0",
        .page_offset = 0,
    }};

    WriteBackBatch batch({op});

    ASSERT_EQ(batch.content_hashes[0], std::vector<std::string>({"h0"}));
    ASSERT_EQ(batch.page_offsets[0], std::vector<std::int32_t>({0}));
}

TEST(CacheOperationTest, L3KeySurvivesHostEvictionAndPrefetches) {
    BlockPool device_pool{8};
    BlockPool host_pool{2};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);

    const CacheKey key_h0{.group_id = 0, .content_hash = "h0"};
    const CacheKey key_h1{.group_id = 0, .content_hash = "h1"};
    CacheBlockRef first = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    CacheBlockRef second = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    coordinator.CacheHostBlock(first, key_h0);
    coordinator.CacheHostBlock(second, key_h1);
    first.reset();
    second.reset();
    ASSERT_TRUE(coordinator.ContainsHostCachedBlock(key_h0));
    ASSERT_TRUE(coordinator.ContainsStorageKey(key_h0));

    CacheBlockRef replacement = coordinator.AcquireHostBlock(/*group_id=*/0);
    ASSERT_TRUE(replacement);
    replacement.reset();
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(key_h0)) << "Host eviction must drop the L2 index entry";
    EXPECT_TRUE(coordinator.ContainsStorageKey(key_h0))
        << "Host eviction must not drop Mooncake keys still inside the Host-capacity shadow";

    auto probe = coordinator.ProbePrefix(std::array<std::string, 1>{"h0"});
    EXPECT_EQ(probe.host.num_common_tokens, 2);

    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .num_tokens = 2}};
    auto admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    ASSERT_EQ(admission->load_pairs.size(), 1u);
    EXPECT_TRUE(admission->load_pairs[0].prefetch_from_storage);
    EXPECT_EQ(admission->load_pairs[0].key.content_hash, "h0");

    admission.reset();
    coordinator.Free(tables);
    EXPECT_TRUE(coordinator.ClearCache());
}

TEST(CacheOperationTest, L3StorageKeyShadowIsBoundedToHostCapacity) {
    BlockPool device_pool{8};
    BlockPool host_pool{2};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    const CacheKey key_h0{.group_id = 0, .content_hash = "h0"};
    const CacheKey key_h1{.group_id = 0, .content_hash = "h1"};
    const CacheKey key_h2{.group_id = 0, .content_hash = "h2"};
    coordinator.RegisterStorageKeys(std::array{key_h0, key_h1, key_h2});

    EXPECT_TRUE(coordinator.ContainsStorageKey(key_h0))
        << "a prompt longer than Host capacity must keep the prefix-start keys";
    EXPECT_TRUE(coordinator.ContainsStorageKey(key_h1));
    EXPECT_FALSE(coordinator.ContainsStorageKey(key_h2))
        << "the shadow must drop the tail once it exceeds Host page capacity";
    EXPECT_EQ(coordinator.NumStorageKeys(), 2);
    EXPECT_EQ(coordinator.ProbePrefix(std::array<std::string, 3>{"h0", "h1", "h2"}).host.num_common_tokens, 4)
        << "prefix-closed matching must still see the retained L3 prefix";

    coordinator.RegisterStorageKeys(std::array{key_h2});
    EXPECT_FALSE(coordinator.ContainsStorageKey(key_h0)) << "a later prompt may LRU-evict an older prefix key";
    EXPECT_TRUE(coordinator.ContainsStorageKey(key_h1));
    EXPECT_TRUE(coordinator.ContainsStorageKey(key_h2));

    coordinator.RegisterStorageKeys(std::array{key_h0});
    EXPECT_TRUE(coordinator.ContainsStorageKey(key_h0)) << "admit-time registration restores a shadow-evicted key";
    EXPECT_FALSE(coordinator.ContainsStorageKey(key_h1));
    EXPECT_TRUE(coordinator.ContainsStorageKey(key_h2));
}

TEST(CacheOperationTest, L3StorageKeyShadowKeepsSharedPrefixAcrossGroups) {
    BlockPool device_pool{8};
    BlockPool host_pool{2};
    const std::array specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 1,
            .block_granularity = 2,
        },
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 1,
            .block_granularity = 2,
        },
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    const std::array hashes{std::string{"h0"}, std::string{"h1"}, std::string{"h2"}};
    coordinator.RegisterStorageKeys(coordinator.ExpandPrefixKeys(hashes));

    EXPECT_EQ(coordinator.NumStorageKeys(), 4);
    EXPECT_TRUE(coordinator.ContainsStorageKey(CacheKey{.group_id = 0, .content_hash = "h0"}));
    EXPECT_TRUE(coordinator.ContainsStorageKey(CacheKey{.group_id = 1, .content_hash = "h0"}));
    EXPECT_TRUE(coordinator.ContainsStorageKey(CacheKey{.group_id = 0, .content_hash = "h1"}));
    EXPECT_TRUE(coordinator.ContainsStorageKey(CacheKey{.group_id = 1, .content_hash = "h1"}));
    EXPECT_FALSE(coordinator.ContainsStorageKey(CacheKey{.group_id = 0, .content_hash = "h2"}));
    EXPECT_FALSE(coordinator.ContainsStorageKey(CacheKey{.group_id = 1, .content_hash = "h2"}));
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 4)
        << "both groups must keep the same prefix-hash boundary";
}

TEST(CacheOperationTest, L3PrefetchShortensHostPrefixWhenHostPoolIsExhausted) {
    BlockPool device_pool{8};
    BlockPool host_pool{2};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    CacheBlockRef pinned = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(pinned);

    const CacheKey key_h0{.group_id = 0, .content_hash = "h0"};
    const CacheKey key_h1{.group_id = 0, .content_hash = "h1"};
    coordinator.RegisterStorageKeys(std::array{key_h0, key_h1});

    auto probe = coordinator.ProbePrefix(std::array<std::string, 2>{"h0", "h1"});
    EXPECT_EQ(probe.host.num_common_tokens, 4);

    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .num_tokens = 2}};
    auto admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    EXPECT_EQ(admission->host_prefix_tokens, 2)
        << "a pinned Host pool must shorten the L3 prefix instead of admitting stale KV";
    ASSERT_EQ(admission->load_pairs.size(), 1u);
    EXPECT_TRUE(admission->load_pairs[0].prefetch_from_storage);

    admission.reset();
    coordinator.Free(tables);
}

TEST(CacheOperationTest, SlidingWindowL3ShortageRematchesLookback) {
    BlockPool device_pool{8};
    BlockPool host_pool{2};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kSlidingWindow,
        // Resuming needs ceil((window - 1) / block_granularity) == 2 pages.
        .sliding_window = 5,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    CacheBlockRef pinned = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(pinned);

    const CacheKey key_h1{.group_id = 0, .content_hash = "h1"};
    const CacheKey key_h2{.group_id = 0, .content_hash = "h2"};
    coordinator.RegisterStorageKeys(std::array{key_h1, key_h2});

    auto probe = coordinator.ProbePrefix(std::array<std::string, 3>{"h0", "h1", "h2"});
    ASSERT_EQ(probe.host.num_common_tokens, 6);
    ASSERT_EQ(probe.host.per_group.size(), 1u);
    ASSERT_EQ(probe.host.per_group[0].hits, (std::vector<std::uint8_t>{0, 1, 1}));

    auto match = MatchPrefixForTest(coordinator, std::array<std::string, 3>{"h0", "h1", "h2"});
    EXPECT_EQ(match.host.num_common_tokens, 0)
        << "truncating [0, 1, 1] to [0, 1] would restore a window whose first live page is a hole";
    ASSERT_EQ(match.host.per_group.size(), 1u);
    EXPECT_TRUE(match.host.per_group[0].blocks.empty());
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 1);
}

TEST(CacheOperationTest, AdmissionLoadPairsKeepHostPinnedAfterTableFree) {
    BlockPool device_pool{8};
    BlockPool host_pool{2};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    const CacheKey key_h0{.group_id = 0, .content_hash = "h0"};
    const CacheKey key_h1{.group_id = 0, .content_hash = "h1"};
    coordinator.RegisterStorageKeys(std::array{key_h0, key_h1});

    auto probe = coordinator.ProbePrefix(std::array<std::string, 2>{"h0", "h1"});
    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .num_tokens = 2}};
    auto admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    ASSERT_FALSE(admission->load_pairs.empty());
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 0);

    coordinator.Free(tables);
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 0)
        << "Host sources in load_pairs stay pinned after Free(tables); retry must reset admission";
    admission.reset();
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 2);
}

TEST(CacheOperationTest, L3HostShortageRoundsDownToPrefixGranularity) {
    BlockPool device_pool{8};
    BlockPool host_pool{2};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    CacheBlockRef pinned = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(pinned);

    const std::vector<CacheKey> keys = coordinator.ExpandPrefixKeys(std::array<std::string, 1>{"h0"});
    ASSERT_EQ(keys.size(), 2u);
    coordinator.RegisterStorageKeys(keys);

    auto probe = coordinator.ProbePrefix(std::array<std::string, 1>{"h0"});
    EXPECT_EQ(probe.host.num_common_tokens, 4);

    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .num_tokens = 4}};
    auto admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    EXPECT_EQ(admission->host_prefix_tokens, 0)
        << "a mid-prefix Host shortage must round down to prefix_granularity, not keep 2 tokens";
    EXPECT_TRUE(admission->load_pairs.empty());

    admission.reset();
    coordinator.Free(tables);
}

TEST(CacheOperationTest, L3HostShortageDoesNotSkipACoarserGroup) {
    BlockPool device_pool{8};
    BlockPool host_pool{2};
    const std::array specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 1,
            .block_granularity = 2,
        },
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 1,
            .block_granularity = 4,
        },
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*stream_device_cache_to_host=*/true);
    CacheBlockRef pinned = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(pinned);

    const std::vector<CacheKey> keys = coordinator.ExpandPrefixKeys(std::array<std::string, 1>{"h0"});
    coordinator.RegisterStorageKeys(keys);

    auto probe = coordinator.ProbePrefix(std::array<std::string, 1>{"h0"});
    EXPECT_EQ(probe.host.num_common_tokens, 4);

    std::vector<BlockTable> tables(2);
    std::vector<GroupDemand> demands{
        {.table = &tables[0], .num_tokens = 4},
        {.table = &tables[1], .num_tokens = 4},
    };
    auto admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    EXPECT_EQ(admission->host_prefix_tokens, 0)
        << "rounding to prefix_granularity must drop the fine-group partial hit so the "
           "coarse group is not skipped without KV";
    EXPECT_TRUE(admission->load_pairs.empty());

    admission.reset();
    coordinator.Free(tables);
}

}  // namespace tokenspeed::test
