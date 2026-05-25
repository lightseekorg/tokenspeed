#include <gtest/gtest.h>

#include "scheduler/operations/cache.h"

namespace tokenspeed::test {

TEST(CacheOperationKindTest, FlatWriteBackBucketsTransfersByKind) {
    WriteBackOperation op;
    op.op_id = 7;
    op.transfers = {
        TransferPair{CacheKind::kKV, 1, 11},
        TransferPair{CacheKind::kMamba, 2, 22},
        TransferPair{CacheKind::kKV, 1, 11},
        TransferPair{CacheKind::kMamba, 3, 23},
    };

    FlatWriteBackOperation flat({op});

    ASSERT_EQ(flat.op_ids, std::vector<cache_op_id>({7}));
    EXPECT_EQ(flat.src_pages[0], std::vector<std::int32_t>({1}));
    EXPECT_EQ(flat.dst_pages[0], std::vector<std::int32_t>({11}));
    EXPECT_EQ(flat.src_pages_by_kind.at("kv")[0], std::vector<std::int32_t>({1}));
    EXPECT_EQ(flat.dst_pages_by_kind.at("kv")[0], std::vector<std::int32_t>({11}));
    EXPECT_EQ(flat.src_pages_by_kind.at("mamba")[0], std::vector<std::int32_t>({2, 3}));
    EXPECT_EQ(flat.dst_pages_by_kind.at("mamba")[0], std::vector<std::int32_t>({22, 23}));
}

TEST(CacheOperationKindTest, FlatLoadBackBucketsTransfersByKind) {
    LoadBackOperation op;
    op.op_id = 9;
    op.transfers = {
        TransferPair{CacheKind::kKV, 10, 20},
        TransferPair{CacheKind::kMamba, 30, 40},
    };

    FlatLoadBackOperation flat({op});

    ASSERT_EQ(flat.op_ids, std::vector<cache_op_id>({9}));
    EXPECT_EQ(flat.src_pages[0], std::vector<std::int32_t>({10}));
    EXPECT_EQ(flat.dst_pages[0], std::vector<std::int32_t>({20}));
    EXPECT_EQ(flat.src_pages_by_kind.at("kv")[0], std::vector<std::int32_t>({10}));
    EXPECT_EQ(flat.dst_pages_by_kind.at("kv")[0], std::vector<std::int32_t>({20}));
    EXPECT_EQ(flat.src_pages_by_kind.at("mamba")[0], std::vector<std::int32_t>({30}));
    EXPECT_EQ(flat.dst_pages_by_kind.at("mamba")[0], std::vector<std::int32_t>({40}));
}

}  // namespace tokenspeed::test
