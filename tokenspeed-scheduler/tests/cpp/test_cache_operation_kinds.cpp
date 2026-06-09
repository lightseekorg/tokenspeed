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

TEST(CacheOperationKindTest, FlatWriteBackPreservesPagedCacheTransfersByOp) {
    WriteBackOperation first;
    first.op_id = 7;
    first.paged_cache_transfers = {
        PagedCacheTransferPair{"v4.c4a.compressed_kv", {1, 2}, {101, 102}},
        PagedCacheTransferPair{"v4.swa_kv", {3}, {103}},
    };
    WriteBackOperation second;
    second.op_id = 8;
    second.paged_cache_transfers = {
        PagedCacheTransferPair{"v4.c128a.compressed_kv", {4}, {104}},
    };

    FlatWriteBackOperation flat({first, second});

    ASSERT_EQ(flat.op_ids, std::vector<cache_op_id>({7, 8}));
    ASSERT_EQ(flat.paged_cache_transfers.size(), 2u);
    ASSERT_EQ(flat.paged_cache_transfers[0].size(), 2u);
    EXPECT_EQ(flat.paged_cache_transfers[0][0].group_id, "v4.c4a.compressed_kv");
    EXPECT_EQ(flat.paged_cache_transfers[0][0].src_pages, std::vector<std::int32_t>({1, 2}));
    EXPECT_EQ(flat.paged_cache_transfers[0][0].dst_pages, std::vector<std::int32_t>({101, 102}));
    EXPECT_EQ(flat.paged_cache_transfers[0][1].group_id, "v4.swa_kv");
    ASSERT_EQ(flat.paged_cache_transfers[1].size(), 1u);
    EXPECT_EQ(flat.paged_cache_transfers[1][0].group_id, "v4.c128a.compressed_kv");
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

TEST(CacheOperationKindTest, FlatLoadBackPreservesPagedCacheTransfersByOp) {
    LoadBackOperation op;
    op.op_id = 9;
    op.paged_cache_transfers = {
        PagedCacheTransferPair{"v4.c4a.compressed_kv", {101, 102}, {1, 2}},
        PagedCacheTransferPair{"v4.c4a.indexer_compressor_state", {103}, {3}},
    };

    FlatLoadBackOperation flat({op});

    ASSERT_EQ(flat.op_ids, std::vector<cache_op_id>({9}));
    ASSERT_EQ(flat.paged_cache_transfers.size(), 1u);
    ASSERT_EQ(flat.paged_cache_transfers[0].size(), 2u);
    EXPECT_EQ(flat.paged_cache_transfers[0][0].group_id, "v4.c4a.compressed_kv");
    EXPECT_EQ(flat.paged_cache_transfers[0][0].src_pages, std::vector<std::int32_t>({101, 102}));
    EXPECT_EQ(flat.paged_cache_transfers[0][0].dst_pages, std::vector<std::int32_t>({1, 2}));
    EXPECT_EQ(flat.paged_cache_transfers[0][1].group_id, "v4.c4a.indexer_compressor_state");
}

}  // namespace tokenspeed::test
