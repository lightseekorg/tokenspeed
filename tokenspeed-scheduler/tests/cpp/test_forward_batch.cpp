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

// Tests for ForwardBatch's struct-of-arrays batching constructor
// (csrc/scheduler/operations/forward.h): ragged-row -1 padding, null-hole(0)
// vs pad(-1), prefill-before-decode partition, group-key union.

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "scheduler/operations/forward.h"

namespace tokenspeed::test {
namespace {

using BlockTables = std::map<std::string, std::vector<std::int32_t>>;

PrefillOperation MakePrefill(std::string id, BlockTables block_tables, std::vector<std::int32_t> input_ids = {},
                             std::int32_t pool_index = 0, std::int32_t extend_prefix_len = 0) {
    PrefillOperation op;
    op.request_id = std::move(id);
    op.request_pool_index = pool_index;
    op.input_length = static_cast<std::int32_t>(input_ids.size());
    op.block_tables = std::move(block_tables);
    op.input_ids = std::move(input_ids);
    op.extend_prefix_len = extend_prefix_len;
    return op;
}

DecodeOperation MakeDecode(std::string id, BlockTables block_tables, std::int32_t decode_input_id = -1,
                           std::int32_t pool_index = 0) {
    DecodeOperation op;
    op.request_id = std::move(id);
    op.request_pool_index = pool_index;
    op.input_length = 1;
    op.block_tables = std::move(block_tables);
    op.decode_input_id = decode_input_id;
    return op;
}

TEST(ForwardBatch, EmptyOpsProducesEmpty) {
    ForwardBatch batch{std::vector<ForwardOperation>{}};

    EXPECT_TRUE(batch.empty());
    EXPECT_EQ(batch.num_extends(), 0u);
    EXPECT_TRUE(batch.request_ids.empty());
    EXPECT_TRUE(batch.block_tables.empty());
}

TEST(ForwardBatch, MultiRequestPadsRaggedRowsWithMinusOne) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", BlockTables{{"full", {10, 11, 12}}}));
    ops.emplace_back(MakePrefill("r1", BlockTables{{"full", {20}}}));

    ForwardBatch batch{std::move(ops)};

    ASSERT_EQ(batch.block_tables.count("full"), 1u);
    const auto& full = batch.block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    EXPECT_EQ(full.at(0), (std::vector<std::int32_t>{10, 11, 12}));
    EXPECT_EQ(full.at(1), (std::vector<std::int32_t>{20, -1, -1}));
}

// Cache contract: 0 = real null-block hole, -1 = absent (pad) column.
TEST(ForwardBatch, NullHoleZeroDistinctFromPadMinusOne) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", BlockTables{{"swa", {0, 31, 32}}}));
    ops.emplace_back(MakePrefill("r1", BlockTables{{"swa", {40}}}));

    ForwardBatch batch{std::move(ops)};

    const auto& swa = batch.block_tables.at("swa");
    ASSERT_EQ(swa.size(), 2u);
    EXPECT_EQ(swa.at(0), (std::vector<std::int32_t>{0, 31, 32}));
    EXPECT_EQ(swa.at(1), (std::vector<std::int32_t>{40, -1, -1}));
    EXPECT_EQ(swa.at(0).at(0), 0);
    EXPECT_EQ(swa.at(1).at(1), -1);
}

TEST(ForwardBatch, PrefillBeforeDecodeKeepsRowsAlignedWithRequests) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakeDecode("d", BlockTables{{"full", {20}}}, /*decode_input_id=*/99));
    ops.emplace_back(MakePrefill("p", BlockTables{{"full", {10, 11}}}, /*input_ids=*/{7, 8}));

    ForwardBatch batch{std::move(ops)};

    ASSERT_EQ(batch.request_ids.size(), 2u);
    EXPECT_EQ(batch.request_ids.at(0), "p");
    EXPECT_EQ(batch.request_ids.at(1), "d");

    const auto& full = batch.block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    EXPECT_EQ(full.at(0), (std::vector<std::int32_t>{10, 11}));
    EXPECT_EQ(full.at(1), (std::vector<std::int32_t>{20, -1}));

    EXPECT_EQ(batch.num_extends(), 1u);
    EXPECT_EQ(batch.input_ids, (std::vector<std::int32_t>{7, 8}));
    EXPECT_EQ(batch.decode_input_ids, (std::vector<std::int32_t>{99}));
}

TEST(ForwardBatch, GroupKeyUnionAcrossRequestsPadsMissingGroup) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", BlockTables{{"full", {10, 11}}}));     // no "swa"
    ops.emplace_back(MakePrefill("r1", BlockTables{{"swa", {20, 21, 22}}}));  // no "full"

    ForwardBatch batch{std::move(ops)};

    ASSERT_EQ(batch.block_tables.count("full"), 1u);
    ASSERT_EQ(batch.block_tables.count("swa"), 1u);

    const auto& full = batch.block_tables.at("full");
    const auto& swa = batch.block_tables.at("swa");
    ASSERT_EQ(full.size(), 2u);
    ASSERT_EQ(swa.size(), 2u);

    EXPECT_EQ(full.at(0), (std::vector<std::int32_t>{10, 11}));
    EXPECT_EQ(full.at(1), (std::vector<std::int32_t>{-1, -1}));
    EXPECT_EQ(swa.at(0), (std::vector<std::int32_t>{-1, -1, -1}));
    EXPECT_EQ(swa.at(1), (std::vector<std::int32_t>{20, 21, 22}));
}

TEST(ForwardBatch, ScalarFieldsTrackPerRequestRows) {
    std::vector<ForwardOperation> ops;
    auto p0 = MakePrefill("r0", BlockTables{{"full", {10}}}, /*input_ids=*/{1, 2, 3},
                          /*pool_index=*/5);
    auto p1 = MakePrefill("r1", BlockTables{{"full", {20, 21}}}, /*input_ids=*/{4, 5},
                          /*pool_index=*/7);
    ops.emplace_back(std::move(p0));
    ops.emplace_back(std::move(p1));

    ForwardBatch batch{std::move(ops)};

    EXPECT_EQ(batch.request_pool_indices, (std::vector<std::int32_t>{5, 7}));
    EXPECT_EQ(batch.input_lengths, (std::vector<std::int32_t>{3, 2}));
    EXPECT_EQ(batch.block_tables.at("full").at(0), (std::vector<std::int32_t>{10, -1}));
    EXPECT_EQ(batch.block_tables.at("full").at(1), (std::vector<std::int32_t>{20, 21}));
    EXPECT_EQ(batch.input_ids, (std::vector<std::int32_t>{1, 2, 3, 4, 5}));
}

TEST(ForwardBatch, EqualLengthRowsUnchanged) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", BlockTables{{"full", {10, 11}}}));
    ops.emplace_back(MakePrefill("r1", BlockTables{{"full", {20, 21}}}));

    ForwardBatch batch{std::move(ops)};

    const auto& full = batch.block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    EXPECT_EQ(full.at(0), (std::vector<std::int32_t>{10, 11}));
    EXPECT_EQ(full.at(1), (std::vector<std::int32_t>{20, 21}));
}

}  // namespace
}  // namespace tokenspeed::test
