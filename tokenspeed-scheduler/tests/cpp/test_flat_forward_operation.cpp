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

// Tests for FlatForwardOperation's struct-of-arrays batching constructor
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

using FlatTable = std::map<std::string, std::vector<std::int32_t>>;

PrefillOperation MakePrefill(std::string id, FlatTable flat, std::vector<std::int32_t> input_ids = {},
                             std::int32_t pool_index = 0, std::int32_t extend_prefix_len = 0) {
    PrefillOperation op;
    op.request_id = std::move(id);
    op.request_pool_index = pool_index;
    op.input_length = static_cast<std::int32_t>(input_ids.size());
    op.flat_block_tables = std::move(flat);
    op.input_ids = std::move(input_ids);
    op.extend_prefix_len = extend_prefix_len;
    return op;
}

DecodeOperation MakeDecode(std::string id, FlatTable flat, std::int32_t decode_input_id = -1,
                           std::int32_t pool_index = 0) {
    DecodeOperation op;
    op.request_id = std::move(id);
    op.request_pool_index = pool_index;
    op.input_length = 1;
    op.flat_block_tables = std::move(flat);
    op.decode_input_id = decode_input_id;
    return op;
}

TEST(FlatForwardOperation, EmptyOpsProducesEmpty) {
    FlatForwardOperation flat_op{std::vector<ForwardOperation>{}};

    EXPECT_TRUE(flat_op.empty());
    EXPECT_EQ(flat_op.num_extends(), 0u);
    EXPECT_TRUE(flat_op.request_ids.empty());
    EXPECT_TRUE(flat_op.flat_block_tables.empty());
}

TEST(FlatForwardOperation, MultiRequestPadsRaggedRowsWithMinusOne) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", FlatTable{{"full", {10, 11, 12}}}));
    ops.emplace_back(MakePrefill("r1", FlatTable{{"full", {20}}}));

    FlatForwardOperation flat_op{std::move(ops)};

    ASSERT_EQ(flat_op.flat_block_tables.count("full"), 1u);
    const auto& full = flat_op.flat_block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    EXPECT_EQ(full.at(0), (std::vector<std::int32_t>{10, 11, 12}));
    EXPECT_EQ(full.at(1), (std::vector<std::int32_t>{20, -1, -1}));
}

// Flat contract: 0 = real null-block hole, -1 = absent (pad) column.
TEST(FlatForwardOperation, NullHoleZeroDistinctFromPadMinusOne) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", FlatTable{{"swa", {0, 31, 32}}}));
    ops.emplace_back(MakePrefill("r1", FlatTable{{"swa", {40}}}));

    FlatForwardOperation flat_op{std::move(ops)};

    const auto& swa = flat_op.flat_block_tables.at("swa");
    ASSERT_EQ(swa.size(), 2u);
    EXPECT_EQ(swa.at(0), (std::vector<std::int32_t>{0, 31, 32}));
    EXPECT_EQ(swa.at(1), (std::vector<std::int32_t>{40, -1, -1}));
    EXPECT_EQ(swa.at(0).at(0), 0);
    EXPECT_EQ(swa.at(1).at(1), -1);
}

TEST(FlatForwardOperation, PrefillBeforeDecodeKeepsRowsAlignedWithRequests) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakeDecode("d", FlatTable{{"full", {20}}}, /*decode_input_id=*/99));
    ops.emplace_back(MakePrefill("p", FlatTable{{"full", {10, 11}}}, /*input_ids=*/{7, 8}));

    FlatForwardOperation flat_op{std::move(ops)};

    ASSERT_EQ(flat_op.request_ids.size(), 2u);
    EXPECT_EQ(flat_op.request_ids.at(0), "p");
    EXPECT_EQ(flat_op.request_ids.at(1), "d");

    const auto& full = flat_op.flat_block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    EXPECT_EQ(full.at(0), (std::vector<std::int32_t>{10, 11}));
    EXPECT_EQ(full.at(1), (std::vector<std::int32_t>{20, -1}));

    EXPECT_EQ(flat_op.num_extends(), 1u);
    EXPECT_EQ(flat_op.input_ids, (std::vector<std::int32_t>{7, 8}));
    EXPECT_EQ(flat_op.decode_input_ids, (std::vector<std::int32_t>{99}));
}

TEST(FlatForwardOperation, GroupKeyUnionAcrossRequestsPadsMissingGroup) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", FlatTable{{"full", {10, 11}}}));     // no "swa"
    ops.emplace_back(MakePrefill("r1", FlatTable{{"swa", {20, 21, 22}}}));  // no "full"

    FlatForwardOperation flat_op{std::move(ops)};

    ASSERT_EQ(flat_op.flat_block_tables.count("full"), 1u);
    ASSERT_EQ(flat_op.flat_block_tables.count("swa"), 1u);

    const auto& full = flat_op.flat_block_tables.at("full");
    const auto& swa = flat_op.flat_block_tables.at("swa");
    ASSERT_EQ(full.size(), 2u);
    ASSERT_EQ(swa.size(), 2u);

    EXPECT_EQ(full.at(0), (std::vector<std::int32_t>{10, 11}));
    EXPECT_EQ(full.at(1), (std::vector<std::int32_t>{-1, -1}));
    EXPECT_EQ(swa.at(0), (std::vector<std::int32_t>{-1, -1, -1}));
    EXPECT_EQ(swa.at(1), (std::vector<std::int32_t>{20, 21, 22}));
}

TEST(FlatForwardOperation, ScalarFieldsTrackPerRequestRows) {
    std::vector<ForwardOperation> ops;
    auto p0 = MakePrefill("r0", FlatTable{{"full", {10}}}, /*input_ids=*/{1, 2, 3},
                          /*pool_index=*/5);
    p0.occupied_pages = {10};
    auto p1 = MakePrefill("r1", FlatTable{{"full", {20, 21}}}, /*input_ids=*/{4, 5},
                          /*pool_index=*/7);
    p1.occupied_pages = {20, 21};
    ops.emplace_back(std::move(p0));
    ops.emplace_back(std::move(p1));

    FlatForwardOperation flat_op{std::move(ops)};

    EXPECT_EQ(flat_op.request_pool_indices, (std::vector<std::int32_t>{5, 7}));
    EXPECT_EQ(flat_op.input_lengths, (std::vector<std::int32_t>{3, 2}));
    ASSERT_EQ(flat_op.occupied_pages.size(), 2u);
    EXPECT_EQ(flat_op.occupied_pages.at(0), (std::vector<std::int32_t>{10}));
    EXPECT_EQ(flat_op.occupied_pages.at(1), (std::vector<std::int32_t>{20, 21}));
    EXPECT_EQ(flat_op.input_ids, (std::vector<std::int32_t>{1, 2, 3, 4, 5}));
}

TEST(FlatForwardOperation, EqualLengthRowsUnchanged) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", FlatTable{{"full", {10, 11}}}));
    ops.emplace_back(MakePrefill("r1", FlatTable{{"full", {20, 21}}}));

    FlatForwardOperation flat_op{std::move(ops)};

    const auto& full = flat_op.flat_block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    EXPECT_EQ(full.at(0), (std::vector<std::int32_t>{10, 11}));
    EXPECT_EQ(full.at(1), (std::vector<std::int32_t>{20, 21}));
}

}  // namespace
}  // namespace tokenspeed::test
