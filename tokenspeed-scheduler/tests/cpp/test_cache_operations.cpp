#include <gtest/gtest.h>

#include "scheduler/operations/cache.h"

namespace tokenspeed::test {

TEST(CacheOperationTest, WriteBackDeduplicatesTransfersAcrossBatch) {
    WriteBackOperation op;
    op.op_id = 7;
    op.transfers = {
        TransferPair{1, 11},
        TransferPair{2, 22},
        TransferPair{1, 11},
    };
    WriteBackOperation duplicate;
    duplicate.op_id = 8;
    duplicate.transfers = {TransferPair{2, 22}, TransferPair{3, 33}};

    WriteBackBatch batch({op, duplicate});

    ASSERT_EQ(batch.op_ids, std::vector<cache_op_id>({7, 8}));
    EXPECT_EQ(batch.src_pages[0], std::vector<std::int32_t>({1, 2}));
    EXPECT_EQ(batch.dst_pages[0], std::vector<std::int32_t>({11, 22}));
    EXPECT_EQ(batch.src_pages[1], std::vector<std::int32_t>({3}));
    EXPECT_EQ(batch.dst_pages[1], std::vector<std::int32_t>({33}));
}

TEST(CacheOperationTest, LoadBackPreservesTransferOrder) {
    LoadBackOperation op;
    op.op_id = 9;
    op.transfers = {
        TransferPair{10, 20},
        TransferPair{30, 40},
    };

    LoadBackBatch batch({op});

    ASSERT_EQ(batch.op_ids, std::vector<cache_op_id>({9}));
    EXPECT_EQ(batch.src_pages[0], std::vector<std::int32_t>({10, 30}));
    EXPECT_EQ(batch.dst_pages[0], std::vector<std::int32_t>({20, 40}));
}

}  // namespace tokenspeed::test
