#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <type_traits>

#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
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
            .rows_per_page = 2,
            .entry_stride_tokens = 1,
            .total_pages = 4,
            .retention = CacheGroupConfig::Retention::FullHistory,
            .family = CacheGroupFamily::History,
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
            .rows_per_page = 2,
            .entry_stride_tokens = 1,
            .total_pages = 9,
            .retention = CacheGroupConfig::Retention::FullHistory,
            .family = CacheGroupFamily::History,
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
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, &host_pool,
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
    EXPECT_FALSE(coordinator.ClearDeviceCache()) << "the transfer ticket must pin its Device source";

    transfers.CompleteWriteBack(write_back->op_id);
    EXPECT_TRUE(coordinator.ClearDeviceCache());
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(CacheKey{.group_id = 0, .content_hash = "h0"}));
}

TEST(CacheOperationTest, RetractionStoreSkipsWhenHostHasNoPlacement) {
    BlockPool device_pool{2};
    BlockPool host_pool{1};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, &host_pool,
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

TEST(CacheOperationTest, RetractionReleaseEstimateExcludesBlocksOwnedByAnotherRequest) {
    BlockPool device_pool{2};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*host_pool=*/nullptr,
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
        .rows_per_page = 2,
        .entry_stride_tokens = 1,
        .total_pages = 4,
        .retention = CacheGroupConfig::Retention::FullHistory,
        .family = CacheGroupFamily::History,
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
        .rows_per_page = 2,
        .entry_stride_tokens = 1,
        .total_pages = 4,
        .retention = CacheGroupConfig::Retention::FullHistory,
        .family = CacheGroupFamily::History,
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

}  // namespace tokenspeed::test
