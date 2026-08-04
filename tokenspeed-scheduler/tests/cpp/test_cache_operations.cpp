#include <gtest/gtest.h>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "scheduler/operations/cache.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"

namespace tokenspeed::test {

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

    ASSERT_EQ(batch.op_ids, std::vector<cache_op_id>({7, 8}));
    EXPECT_EQ(batch.group_ids[0], std::vector<GroupId>({0, 0}));
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

    EXPECT_EQ(batch.group_ids[0], std::vector<GroupId>({0, 1}));
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

    ASSERT_EQ(batch.op_ids, std::vector<cache_op_id>({9}));
    EXPECT_EQ(batch.group_ids[0], std::vector<GroupId>({0, 0}));
    EXPECT_EQ(batch.src_pages[0], std::vector<std::int32_t>({10, 30}));
    EXPECT_EQ(batch.dst_pages[0], std::vector<std::int32_t>({20, 40}));
}

TEST(CacheOperationTest, L2RoleGatesSeparatePrefixCacheFromDecodeSnapshots) {
    SchedulerConfig config;
    config.host_allocator.total_pages = 2;

    config.role = Role::kFused;
    EXPECT_TRUE(config.OrdinaryL2Enabled());
    EXPECT_FALSE(config.DecodeSnapshotEnabled());

    config.role = Role::kD;
    EXPECT_FALSE(config.OrdinaryL2Enabled());
    EXPECT_TRUE(config.DecodeSnapshotEnabled());

    config.disable_l2_cache = true;
    EXPECT_FALSE(config.OrdinaryL2Enabled());
    EXPECT_FALSE(config.DecodeSnapshotEnabled());
}

TEST(CacheOperationTest, SnapshotBlockTablePreservesHolesAndTailCapacity) {
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

TEST(CacheOperationTest, DecodeRejectsRequestWhoseSnapshotCannotFitHostL2) {
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
    EXPECT_EQ(scheduler.MaxHostSnapshotTokens(), 6);
    EXPECT_EQ(scheduler.MaxSingleRequestTokens(), 6)
        << "Decode must advertise the smaller of Device and snapshot capacity";
    RequestSpec spec{
        .request_id = "too-large",
        .tokens = {1, 2, 3, 4},
        .max_new_tokens = 4,
    };

    EXPECT_THROW(scheduler.SubmitRequests({spec}), std::invalid_argument);
}

}  // namespace tokenspeed::test
