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

TEST(CacheOperationTest, PrefixL2IsDisabledForDecode) {
    SchedulerConfig config;
    config.host_allocator.total_pages = 2;

    config.role = Role::kFused;
    EXPECT_TRUE(config.PrefixL2Enabled());

    config.role = Role::kD;
    EXPECT_FALSE(config.PrefixL2Enabled());

    config.disable_l2_cache = true;
    EXPECT_FALSE(config.PrefixL2Enabled());
}

TEST(CacheOperationTest, DecodeRequiresEnabledHostRetractionPool) {
    const auto make_config = [] {
        SchedulerConfig config;
        config.block_size = 2;
        config.device_allocator.total_pages = 4;
        config.host_allocator.total_pages = 4;
        config.max_scheduled_tokens = 2;
        config.max_batch_size = 1;
        config.role = Role::kD;
        config.paged_cache_groups.push_back(PagedCacheGroupConfig{
            .group_id = "full",
            .rows_per_page = 2,
            .entry_stride_tokens = 1,
            .total_pages = 4,
            .retention = PagedCacheGroupConfig::Retention::FullHistory,
            .family = PagedCacheGroupFamily::History,
        });
        return config;
    };

    SchedulerConfig disabled = make_config();
    disabled.disable_l2_cache = true;
    EXPECT_THROW(Scheduler{std::move(disabled)}, std::invalid_argument);

    SchedulerConfig empty = make_config();
    empty.host_allocator.total_pages = 1;
    EXPECT_THROW(Scheduler{std::move(empty)}, std::invalid_argument);
}

TEST(CacheOperationTest, RetractionBlockTablePreservesHolesAndTailCapacity) {
    BlockPool pool{2};
    std::vector<CacheBlockRef> blocks;
    blocks.push_back(pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1));
    blocks.emplace_back();

    BlockTable table = BlockTable::FromBlocks(std::move(blocks), /*available_tokens=*/3);

    ASSERT_EQ(table.NumBlocks(), 2);
    EXPECT_TRUE(table.Blocks()[0]);
    EXPECT_FALSE(table.Blocks()[1]);
    EXPECT_EQ(table.AvailableTokens(), 3);
}

TEST(CacheOperationTest, RetractionReusesHostPrefixAndTransfersOnlyMissingBlocks) {
    BlockPool device_pool{4};
    BlockPool host_pool{4};
    const std::array specs{KvCacheSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .cache_block_tokens = 2,
    }};
    KvCacheCoordinator coordinator = MakeCoordinator(specs, /*cache_block_tokens=*/2, device_pool, &host_pool);
    TierTransferManager transfers{coordinator};
    const std::vector<std::string> hashes{"h0", "h1", "h2"};

    CacheBlockRef cached_host = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    const CacheBlockLocation cached_location = cached_host->Location();
    coordinator.CacheHostBlock(cached_host, CacheKey{.group_id = 0, .content_hash = hashes[0]});
    cached_host.reset();

    std::vector<CacheBlockRef> device_blocks;
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        device_blocks.push_back(device_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1));
    }
    std::vector<BlockTable> device_tables;
    device_tables.push_back(BlockTable::FromBlocks(std::move(device_blocks), /*available_tokens=*/1));

    auto prepared = transfers.PrepareRetraction(hashes, /*access_epoch=*/1, device_tables);

    ASSERT_TRUE(prepared);
    EXPECT_EQ(prepared->host_prefix_tokens, 2);
    ASSERT_EQ(prepared->host_tables.size(), 1u);
    ASSERT_EQ(prepared->host_tables[0].NumBlocks(), 3);
    EXPECT_EQ(prepared->host_tables[0].Blocks()[0]->Location(), cached_location);
    EXPECT_EQ(prepared->host_tables[0].AvailableTokens(), 1);
    EXPECT_EQ(prepared->transfers.size(), 2u);
}

TEST(CacheOperationTest, RetractionEvictsUnusedHostCacheForDestination) {
    BlockPool device_pool{2};
    BlockPool host_pool{2};
    const std::array specs{KvCacheSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .cache_block_tokens = 2,
    }};
    KvCacheCoordinator coordinator = MakeCoordinator(specs, /*cache_block_tokens=*/2, device_pool, &host_pool);
    TierTransferManager transfers{coordinator};

    const CacheKey old_key{.group_id = 0, .content_hash = "old"};
    const CacheKey matched_key{.group_id = 0, .content_hash = "h0"};
    CacheBlockRef old_host = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    const CacheBlockLocation old_location = old_host->Location();
    coordinator.CacheHostBlock(old_host, old_key);
    old_host.reset();
    CacheBlockRef matched_host = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    const CacheBlockLocation matched_location = matched_host->Location();
    coordinator.CacheHostBlock(matched_host, matched_key);
    matched_host.reset();

    std::vector<CacheBlockRef> device_blocks;
    device_blocks.push_back(device_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1));
    device_blocks.push_back(device_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1));
    std::vector<BlockTable> device_tables;
    device_tables.push_back(BlockTable::FromBlocks(std::move(device_blocks), /*available_tokens=*/0));
    const std::array<std::string, 2> hashes{"h0", "new"};

    auto prepared = transfers.PrepareRetraction(hashes, /*access_epoch=*/1, device_tables);

    ASSERT_TRUE(prepared);
    ASSERT_EQ(prepared->host_tables.size(), 1u);
    ASSERT_EQ(prepared->host_tables[0].NumBlocks(), 2);
    EXPECT_EQ(prepared->host_tables[0].Blocks()[0]->Location(), matched_location);
    EXPECT_EQ(prepared->host_tables[0].Blocks()[1]->Location(), old_location);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(old_key));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(matched_key));
    EXPECT_EQ(prepared->transfers.size(), 1u);
}

TEST(CacheOperationTest, FailedRetractionPreparationLeavesHostCacheUnchanged) {
    BlockPool device_pool{3};
    BlockPool host_pool{3};
    const std::array specs{KvCacheSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .cache_block_tokens = 2,
    }};
    KvCacheCoordinator coordinator = MakeCoordinator(specs, /*cache_block_tokens=*/2, device_pool, &host_pool);
    TierTransferManager transfers{coordinator};

    const CacheKey matched_key{.group_id = 0, .content_hash = "h0"};
    CacheBlockRef matched_host = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    const CacheBlockLocation matched_location = matched_host->Location();
    coordinator.CacheHostBlock(matched_host, matched_key);
    matched_host.reset();

    const CacheKey cold_key{.group_id = 0, .content_hash = "cold"};
    CacheBlockRef cold_host = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    coordinator.CacheHostBlock(cold_host, cold_key);
    cold_host.reset();

    CacheBlockRef pinned_host = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(pinned_host);
    const auto metadata_before = coordinator.GroupManager(0).CachedBlockMetadataFor(host_pool, matched_location);
    ASSERT_TRUE(metadata_before);

    std::vector<CacheBlockRef> device_blocks;
    for (std::int32_t i = 0; i < 3; ++i) {
        device_blocks.push_back(device_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1));
    }
    std::vector<BlockTable> device_tables;
    device_tables.push_back(BlockTable::FromBlocks(std::move(device_blocks), /*available_tokens=*/0));
    const std::array<std::string, 3> hashes{"h0", "new-1", "new-2"};

    EXPECT_FALSE(transfers.PrepareRetraction(hashes, /*access_epoch=*/2, device_tables));

    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(matched_key));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(cold_key));
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 2);
    EXPECT_EQ(host_pool.NumOccupiedSlots(), 3);
    const auto metadata_after = coordinator.GroupManager(0).CachedBlockMetadataFor(host_pool, matched_location);
    ASSERT_TRUE(metadata_after);
    EXPECT_EQ(metadata_after->last_access_epoch, metadata_before->last_access_epoch);
    EXPECT_EQ(metadata_after->was_acquired, metadata_before->was_acquired);
}

TEST(CacheOperationTest, RecoveryReusesDevicePrefixAndLoadsOnlyMissingBlocks) {
    BlockPool device_pool{6};
    BlockPool host_pool{4};
    const std::array specs{KvCacheSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .cache_block_tokens = 2,
    }};
    KvCacheCoordinator coordinator = MakeCoordinator(specs, /*cache_block_tokens=*/2, device_pool, &host_pool);
    TierTransferManager transfers{coordinator};
    const std::vector<std::string> hashes{"h0", "h1", "h2"};

    std::vector<BlockTable> cached_tables(1);
    std::vector<GroupDemand> cached_demands{{.table = &cached_tables[0], .num_tokens = 2}};
    auto admission = coordinator.Admit(coordinator.ProbePrefix({}), cached_demands);
    ASSERT_TRUE(admission);
    const CacheBlockLocation cached_device_location = cached_tables[0].Blocks()[0]->Location();
    coordinator.CacheFullBlocks(cached_tables, std::span<const std::string>{hashes}.first(1),
                                admission->access_epoch);
    coordinator.Free(cached_tables);

    std::vector<CacheBlockRef> host_blocks;
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        host_blocks.push_back(host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1));
    }
    std::vector<BlockTable> host_tables;
    host_tables.push_back(BlockTable::FromBlocks(std::move(host_blocks), /*available_tokens=*/0));

    auto prepared = transfers.PrepareRecovery(hashes, admission->access_epoch, host_tables,
                                              /*decode_reserve_tokens=*/1);

    ASSERT_TRUE(prepared);
    ASSERT_EQ(prepared->device_tables.size(), 1u);
    ASSERT_EQ(prepared->device_tables[0].NumBlocks(), 4);
    EXPECT_EQ(prepared->device_tables[0].Blocks()[0]->Location(), cached_device_location);
    EXPECT_EQ(prepared->transfers.size(), 2u);
    EXPECT_EQ(prepared->new_device_page_ids[0].size(), 3u);
}

TEST(CacheOperationTest, DecodeRejectsRequestWhoseRetractionStateCannotFitHostL2) {
    SchedulerConfig config;
    config.block_size = 2;
    config.device_allocator.total_pages = 9;
    config.host_allocator.total_pages = 4;
    config.max_scheduled_tokens = 8;
    config.max_batch_size = 2;
    config.role = Role::kD;
    config.paged_cache_groups.push_back(PagedCacheGroupConfig{
        .group_id = "full",
        .rows_per_page = 2,
        .entry_stride_tokens = 1,
        .total_pages = 9,
        .retention = PagedCacheGroupConfig::Retention::FullHistory,
        .family = PagedCacheGroupFamily::History,
    });
    Scheduler scheduler{std::move(config)};
    EXPECT_EQ(scheduler.MaxHostRetractionTokens(), 6);
    EXPECT_EQ(scheduler.MaxSingleRequestTokens(), 6)
        << "Decode must advertise the smaller of Device and retraction capacity";
    RequestSpec spec{
        .request_id = "too-large",
        .tokens = {1, 2, 3, 4},
        .max_new_tokens = 4,
    };

    EXPECT_THROW(scheduler.SubmitRequests({spec}), std::invalid_argument);
}

TEST(CacheOperationTest, DecodeRejectsRequestWhoseMaximumExtentCannotFitDevice) {
    SchedulerConfig config;
    config.block_size = 2;
    config.device_allocator.total_pages = 4;
    config.host_allocator.total_pages = 10;
    config.max_scheduled_tokens = 8;
    config.max_batch_size = 2;
    config.role = Role::kD;
    config.paged_cache_groups.push_back(PagedCacheGroupConfig{
        .group_id = "full",
        .rows_per_page = 2,
        .entry_stride_tokens = 1,
        .total_pages = 4,
        .retention = PagedCacheGroupConfig::Retention::FullHistory,
        .family = PagedCacheGroupFamily::History,
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

}  // namespace tokenspeed::test
