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

// Unit tests for FlatForwardOperation's struct-of-arrays batching constructor
// (csrc/scheduler/operations/forward.h). This logic is build-flag independent
// (it carries no #if TOKENSPEED_FLAT_KVCACHE guard) and runs in the default
// build, so these tests exercise the multi-request aggregation that the
// end-to-end lifecycle tests cannot reach with a single request: ragged-row -1
// padding, the null-hole(0)-vs-pad(-1) distinction, prefill-before-decode
// stable_partition row reordering, and group-key union across requests with
// differing key sets.

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "scheduler/operations/forward.h"

namespace tokenspeed::test {
namespace {

using FlatTable = std::map<std::string, std::vector<std::int32_t>>;

// Build a PrefillOperation carrying just the fields these tests assert on.
// input_ids let us check the per-token concatenation; flat seeds the per-group
// block-table row.
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

// Build a DecodeOperation with a flat row and the one decode token id.
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

// Empty input -> empty everything. Guards the degenerate batch (the scheduler
// can produce a FlatForwardOperation with no ops).
TEST(FlatForwardOperation, EmptyOpsProducesEmpty) {
    FlatForwardOperation flat_op{std::vector<ForwardOperation>{}};

    EXPECT_TRUE(flat_op.empty());
    EXPECT_EQ(flat_op.num_extends(), 0u);
    EXPECT_TRUE(flat_op.request_ids.empty());
    EXPECT_TRUE(flat_op.flat_block_tables.empty());
}

// Two requests, same single group, rows of unequal length -> the short row is
// right-padded with -1 to the batch's max column count (rectangular output).
TEST(FlatForwardOperation, MultiRequestPadsRaggedRowsWithMinusOne) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", FlatTable{{"full", {10, 11, 12}}}));
    ops.emplace_back(MakePrefill("r1", FlatTable{{"full", {20}}}));

    FlatForwardOperation flat_op{std::move(ops)};

    ASSERT_EQ(flat_op.flat_block_tables.count("full"), 1u);
    const auto& full = flat_op.flat_block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    EXPECT_EQ(full.at(0), (std::vector<std::int32_t>{10, 11, 12}));
    // Shorter row padded to width 3 with the -1 sentinel (NOT 0).
    EXPECT_EQ(full.at(1), (std::vector<std::int32_t>{20, -1, -1}));
}

// A null hole (id 0) inside a row must survive verbatim, and must be distinct
// from the -1 used to pad a shorter row. This is the core flat contract: 0 =
// real null-block hole, -1 = absent column.
TEST(FlatForwardOperation, NullHoleZeroDistinctFromPadMinusOne) {
    std::vector<ForwardOperation> ops;
    // Row with an interior null hole (slid-out sliding-window page).
    ops.emplace_back(MakePrefill("r0", FlatTable{{"swa", {0, 31, 32}}}));
    // Shorter row -> gets -1 padding, and its own real page.
    ops.emplace_back(MakePrefill("r1", FlatTable{{"swa", {40}}}));

    FlatForwardOperation flat_op{std::move(ops)};

    const auto& swa = flat_op.flat_block_tables.at("swa");
    ASSERT_EQ(swa.size(), 2u);
    EXPECT_EQ(swa.at(0), (std::vector<std::int32_t>{0, 31, 32}));
    EXPECT_EQ(swa.at(1), (std::vector<std::int32_t>{40, -1, -1}));
    // Pin the distinction explicitly: column 0 of row0 is a real hole (0), the
    // tail of row1 is padding (-1) -- never conflated.
    EXPECT_EQ(swa.at(0).at(0), 0);
    EXPECT_EQ(swa.at(1).at(1), -1);
}

// Mixed prefill+decode given in decode-first order: the constructor's
// stable_partition pulls prefills to the front, and every parallel array --
// request_ids AND the flat block-table rows -- must follow the same reorder so
// row r still describes request_ids[r].
TEST(FlatForwardOperation, PrefillBeforeDecodeKeepsRowsAlignedWithRequests) {
    std::vector<ForwardOperation> ops;
    // Submitted decode-first to force a non-trivial partition.
    ops.emplace_back(MakeDecode("d", FlatTable{{"full", {20}}}, /*decode_input_id=*/99));
    ops.emplace_back(MakePrefill("p", FlatTable{{"full", {10, 11}}}, /*input_ids=*/{7, 8}));

    FlatForwardOperation flat_op{std::move(ops)};

    // Prefill partitioned ahead of decode.
    ASSERT_EQ(flat_op.request_ids.size(), 2u);
    EXPECT_EQ(flat_op.request_ids.at(0), "p");
    EXPECT_EQ(flat_op.request_ids.at(1), "d");

    // Rows follow the same order: row0 is the prefill's, row1 the decode's.
    const auto& full = flat_op.flat_block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    EXPECT_EQ(full.at(0), (std::vector<std::int32_t>{10, 11}));
    EXPECT_EQ(full.at(1), (std::vector<std::int32_t>{20, -1}));

    // Prefill-only / decode-only SoA streams stay consistent with the partition.
    EXPECT_EQ(flat_op.num_extends(), 1u);
    EXPECT_EQ(flat_op.input_ids, (std::vector<std::int32_t>{7, 8}));
    EXPECT_EQ(flat_op.decode_input_ids, (std::vector<std::int32_t>{99}));
}

// Requests with disjoint group-key sets: the batch's group set is the UNION,
// and a request missing a group contributes an (all -1) padded row for it, so
// every group's table stays rectangular with num_reqs rows.
TEST(FlatForwardOperation, GroupKeyUnionAcrossRequestsPadsMissingGroup) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", FlatTable{{"full", {10, 11}}}));        // no "swa"
    ops.emplace_back(MakePrefill("r1", FlatTable{{"swa", {20, 21, 22}}}));     // no "full"

    FlatForwardOperation flat_op{std::move(ops)};

    ASSERT_EQ(flat_op.flat_block_tables.count("full"), 1u);
    ASSERT_EQ(flat_op.flat_block_tables.count("swa"), 1u);

    // Every group has one row per request (rectangular over the batch).
    const auto& full = flat_op.flat_block_tables.at("full");
    const auto& swa = flat_op.flat_block_tables.at("swa");
    ASSERT_EQ(full.size(), 2u);
    ASSERT_EQ(swa.size(), 2u);

    EXPECT_EQ(full.at(0), (std::vector<std::int32_t>{10, 11}));
    // r1 had no "full" -> its row is empty, padded to width 2 with -1.
    EXPECT_EQ(full.at(1), (std::vector<std::int32_t>{-1, -1}));
    // r0 had no "swa" -> its row is empty, padded to width 3 with -1.
    EXPECT_EQ(swa.at(0), (std::vector<std::int32_t>{-1, -1, -1}));
    EXPECT_EQ(swa.at(1), (std::vector<std::int32_t>{20, 21, 22}));
}

// Per-request scalar SoA fields land in the right row after partition: indices,
// input lengths and occupied_pages must travel with their request.
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
    // All-prefill batch: every input row concatenated head-to-tail.
    EXPECT_EQ(flat_op.input_ids, (std::vector<std::int32_t>{1, 2, 3, 4, 5}));
}

// Equal-length rows need no padding: the output equals the inputs verbatim.
// Guards against the padding pass mutating already-rectangular tables.
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
