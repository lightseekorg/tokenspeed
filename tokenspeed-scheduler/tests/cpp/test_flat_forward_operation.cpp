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
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache/kv_cache_coordinator.h"
#include "scheduler/operations/forward.h"

namespace tokenspeed::test {
namespace {

using FlatTable = std::map<std::string, std::vector<std::int32_t>>;
using FlatBase = std::map<std::string, std::int32_t>;

FlatBase ZeroBases(const FlatTable& flat) {
    FlatBase bases;
    for (const auto& [group_id, _] : flat) {
        bases.emplace(group_id, 0);
    }
    return bases;
}

void ExpectRowEq(std::span<const std::int32_t> actual, std::initializer_list<std::int32_t> expected) {
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
}

PrefillOperation MakePrefill(std::string id, FlatTable flat, std::vector<std::int32_t> input_ids = {},
                             std::int32_t pool_index = 0, std::int32_t extend_prefix_len = 0, FlatBase bases = {}) {
    PrefillOperation op;
    op.request_id = std::move(id);
    op.request_pool_index = pool_index;
    op.input_length = static_cast<std::int32_t>(input_ids.size());
    if (bases.empty()) {
        bases = ZeroBases(flat);
    }
    op.flat_block_tables = std::move(flat);
    op.flat_block_table_base_offsets = std::move(bases);
    op.input_ids = std::move(input_ids);
    op.extend_prefix_len = extend_prefix_len;
    return op;
}

DecodeOperation MakeDecode(std::string id, FlatTable flat, std::int32_t decode_input_id = -1,
                           std::int32_t pool_index = 0, FlatBase bases = {}) {
    DecodeOperation op;
    op.request_id = std::move(id);
    op.request_pool_index = pool_index;
    op.input_length = 1;
    if (bases.empty()) {
        bases = ZeroBases(flat);
    }
    op.flat_block_tables = std::move(flat);
    op.flat_block_table_base_offsets = std::move(bases);
    op.decode_input_id = decode_input_id;
    return op;
}

TEST(FlatForwardOperation, EmptyOpsProducesEmpty) {
    FlatForwardOperation flat_op{std::vector<ForwardOperation>{}};

    EXPECT_TRUE(flat_op.empty());
    EXPECT_EQ(flat_op.num_extends(), 0u);
    EXPECT_TRUE(flat_op.request_ids.empty());
    EXPECT_TRUE(flat_op.flat_block_tables.empty());
    EXPECT_TRUE(flat_op.MaterializeFlatBlockTableBaseOffsets().empty());
}

TEST(FlatForwardOperation, MultiRequestPadsRaggedRowsWithMinusOne) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", FlatTable{{"full", {10, 11, 12}}}));
    ops.emplace_back(MakePrefill("r1", FlatTable{{"full", {20}}}));

    FlatForwardOperation flat_op{std::move(ops)};

    ASSERT_EQ(flat_op.flat_block_tables.count("full"), 1u);
    const auto& full = flat_op.flat_block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    EXPECT_EQ(full.rows, 2u);
    EXPECT_EQ(full.cols, 3u);
    EXPECT_EQ(full.values, (std::vector<std::int32_t>{10, 11, 12, 20, -1, -1}));
    ExpectRowEq(full.Row(0), {10, 11, 12});
    ExpectRowEq(full.Row(1), {20, -1, -1});
    EXPECT_EQ(full.Row(1).data(), full.Row(0).data() + full.cols);
}

// Flat contract: 0 = real null-block hole, -1 = absent (pad) column.
TEST(FlatForwardOperation, NullHoleZeroDistinctFromPadMinusOne) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", FlatTable{{"swa", {0, 31, 32}}}));
    ops.emplace_back(MakePrefill("r1", FlatTable{{"swa", {40}}}));

    FlatForwardOperation flat_op{std::move(ops)};

    const auto& swa = flat_op.flat_block_tables.at("swa");
    ASSERT_EQ(swa.size(), 2u);
    ExpectRowEq(swa.Row(0), {0, 31, 32});
    ExpectRowEq(swa.Row(1), {40, -1, -1});
    EXPECT_EQ(swa.Row(0)[0], 0);
    EXPECT_EQ(swa.Row(1)[1], -1);
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
    ExpectRowEq(full.Row(0), {10, 11});
    ExpectRowEq(full.Row(1), {20, -1});

    EXPECT_EQ(flat_op.num_extends(), 1u);
    EXPECT_EQ(flat_op.input_ids, (std::vector<std::int32_t>{7, 8}));
    EXPECT_EQ(flat_op.decode_input_ids, (std::vector<std::int32_t>{99}));
}

TEST(FlatForwardOperation, CompletionInputsStayAlignedAcrossStablePartition) {
    std::vector<ForwardOperation> ops;
    auto decode = MakeDecode("d", FlatTable{{"full", {20}}});
    decode.flat_kv_completion_input = FlatKVCompletionInput{
        .request_id = "d",
        .table_generation = 11,
        .dispatch_seq = 7,
        .dispatch_raw_start = 7,
        .dispatch_raw_end = 8,
        .protected_raw_end = 9,
    };
    auto prefill = MakePrefill("p", FlatTable{{"full", {10}}});
    prefill.flat_kv_completion_input = FlatKVCompletionInput{
        .request_id = "p",
        .table_generation = 22,
        .dispatch_seq = 3,
        .dispatch_raw_start = 0,
        .dispatch_raw_end = 4,
        .protected_raw_end = 5,
    };
    ops.emplace_back(std::move(decode));
    ops.emplace_back(std::move(prefill));

    FlatForwardOperation flat_op{std::move(ops)};

    ASSERT_EQ(flat_op.flat_kv_completion_inputs.size(), 2u);
    EXPECT_EQ(flat_op.flat_kv_completion_inputs.at(0).request_id, "p");
    EXPECT_EQ(flat_op.flat_kv_completion_inputs.at(1).request_id, "d");
}

TEST(FlatForwardOperation, MixedPresenceOfCompletionInputsFailsClosed) {
    std::vector<ForwardOperation> ops;
    auto first = MakePrefill("a", FlatTable{{"full", {10}}});
    first.flat_kv_completion_input = FlatKVCompletionInput{
        .request_id = "a",
        .table_generation = 1,
        .dispatch_seq = 0,
        .dispatch_raw_start = 0,
        .dispatch_raw_end = 1,
        .protected_raw_end = 1,
    };
    ops.emplace_back(std::move(first));
    ops.emplace_back(MakePrefill("b", FlatTable{{"full", {20}}}));

    EXPECT_THROW(FlatForwardOperation(std::move(ops)), std::invalid_argument);
}

TEST(FlatForwardOperation, MissingRequiredGroupFailsClosed) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", FlatTable{{"full", {10, 11}}}));     // no "swa"
    ops.emplace_back(MakePrefill("r1", FlatTable{{"swa", {20, 21, 22}}}));  // no "full"

    EXPECT_THROW(FlatForwardOperation(std::move(ops)), std::invalid_argument);
}

TEST(FlatForwardOperation, BaseOffsetsStayAlignedAcrossStablePartition) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakeDecode("d", FlatTable{{"full", {20}}, {"state", {30}}}, /*decode_input_id=*/99,
                                /*pool_index=*/0, FlatBase{{"full", 0}, {"state", 7}}));
    ops.emplace_back(MakePrefill("p", FlatTable{{"full", {10}}, {"state", {40}}}, /*input_ids=*/{7},
                                 /*pool_index=*/0, /*extend_prefix_len=*/0, FlatBase{{"full", 0}, {"state", 3}}));

    FlatForwardOperation flat_op{std::move(ops)};

    EXPECT_EQ(flat_op.request_ids, (std::vector<std::string>{"p", "d"}));
    EXPECT_EQ(flat_op.flat_block_tables.at("full").bases, (std::vector<std::int32_t>{0, 0}));
    EXPECT_EQ(flat_op.flat_block_tables.at("state").bases, (std::vector<std::int32_t>{3, 7}));
}

TEST(FlatForwardOperation, MissingBaseForAnyFlatGroupFailsClosed) {
    PrefillOperation op;
    op.request_id = "r";
    op.flat_block_tables = {{"full", {10}}, {"state", {20}}};
    op.flat_block_table_base_offsets = {{"full", 0}};
    std::vector<ForwardOperation> ops;
    ops.emplace_back(std::move(op));

    EXPECT_THROW(FlatForwardOperation(std::move(ops)), std::invalid_argument);
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
    ExpectRowEq(full.Row(0), {10, 11});
    ExpectRowEq(full.Row(1), {20, 21});
}

TEST(FlatForwardOperation, LiveRowViewCopiesDirectlyIntoOneContiguousOwner) {
    BlockPool pool{/*total_num_blocks=*/16};
    const std::vector<KvCacheSpec> specs = {
        KvCacheSpec{.kind = AttnKind::kFull, .block_size = 2},
        KvCacheSpec{.kind = AttnKind::kFull, .block_size = 4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(static_cast<std::size_t>(coordinator.NumGroups()));
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/5));
    const std::vector<std::string> group_ids{"fine", "coarse"};
    const std::vector<std::int32_t> fine = BlockTablePageIds(tables[0]);
    const std::vector<std::int32_t> coarse = BlockTablePageIds(tables[1]);

    PrefillOperation op;
    op.request_id = "r";
    op.flat_block_table_view = tables;
    op.flat_block_table_group_ids = group_ids;
    ASSERT_TRUE(op.flat_block_tables.empty());
    ASSERT_TRUE(op.flat_block_table_base_offsets.empty());
    std::vector<ForwardOperation> ops;
    ops.emplace_back(std::move(op));

    FlatForwardOperation flat_op{std::move(ops)};

    const auto fine_row = flat_op.flat_block_tables.at("fine").Row(0);
    const auto coarse_row = flat_op.flat_block_tables.at("coarse").Row(0);
    EXPECT_EQ(std::vector<std::int32_t>(fine_row.begin(), fine_row.end()), fine);
    EXPECT_EQ(std::vector<std::int32_t>(coarse_row.begin(), coarse_row.end()), coarse);
    EXPECT_EQ(flat_op.flat_block_tables.at("fine").bases, (std::vector<std::int32_t>{0}));
    EXPECT_EQ(flat_op.flat_block_tables.at("coarse").bases, (std::vector<std::int32_t>{0}));
    coordinator.Free(tables);
}

TEST(FlatForwardOperation, EmptyRowsKeepZeroWidthAndCopyOnlyBases) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", FlatTable{{"full", {}}}, /*input_ids=*/{}, /*pool_index=*/0,
                                 /*extend_prefix_len=*/0, FlatBase{{"full", 5}}));
    ops.emplace_back(MakePrefill("r1", FlatTable{{"full", {}}}, /*input_ids=*/{}, /*pool_index=*/0,
                                 /*extend_prefix_len=*/0, FlatBase{{"full", 7}}));

    FlatForwardOperation flat_op{std::move(ops)};
    const FlatBlockTableExport& full = flat_op.flat_block_tables.at("full");
    EXPECT_EQ(full.rows, 2u);
    EXPECT_EQ(full.cols, 0u);
    EXPECT_TRUE(full.values.empty());
    EXPECT_TRUE(full.Row(0).empty());
    EXPECT_TRUE(full.Row(1).empty());

    std::vector<std::int32_t> table_destination{123};
    std::vector<std::int32_t> base_destination(3, -9);
    const FlatBlockTableExport::CopyResult copied =
        full.CopyTo(std::span<std::int32_t>{table_destination}, std::span<std::int32_t>{base_destination},
                    /*page_id_upper_bound=*/8);
    EXPECT_EQ(copied.rows, 2u);
    EXPECT_EQ(copied.cols, 0u);
    EXPECT_EQ(table_destination, (std::vector<std::int32_t>{123}));
    EXPECT_EQ(base_destination, (std::vector<std::int32_t>{5, 7, -9}));
}

TEST(FlatForwardOperation, CopyToUsesContiguousRectangleAndHonorsCapacity) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("r0", FlatTable{{"full", {0, 3, 4}}}, /*input_ids=*/{}, /*pool_index=*/0,
                                 /*extend_prefix_len=*/0, FlatBase{{"full", 0}}));
    ops.emplace_back(MakePrefill("r1", FlatTable{{"full", {5}}}, /*input_ids=*/{}, /*pool_index=*/0,
                                 /*extend_prefix_len=*/0, FlatBase{{"full", 7}}));

    FlatForwardOperation flat_op{std::move(ops)};
    const FlatBlockTableExport& full = flat_op.flat_block_tables.at("full");
    std::vector<std::int32_t> table_destination(8, 99);
    std::vector<std::int32_t> base_destination(3, 99);
    const FlatBlockTableExport::CopyResult copied =
        full.CopyTo(std::span<std::int32_t>{table_destination}, std::span<std::int32_t>{base_destination},
                    /*page_id_upper_bound=*/8);

    EXPECT_EQ(copied.rows, 2u);
    EXPECT_EQ(copied.cols, 3u);
    EXPECT_EQ(table_destination, (std::vector<std::int32_t>{0, 3, 4, 5, -1, -1, 99, 99}));
    EXPECT_EQ(base_destination, (std::vector<std::int32_t>{0, 7, 99}));

    std::vector<std::int32_t> short_table(5, 0);
    EXPECT_THROW(full.CopyTo(std::span<std::int32_t>{short_table}, std::span<std::int32_t>{base_destination}, 8),
                 std::invalid_argument);
    std::vector<std::int32_t> short_bases(1, 0);
    EXPECT_THROW(full.CopyTo(std::span<std::int32_t>{table_destination}, std::span<std::int32_t>{short_bases}, 8),
                 std::invalid_argument);
    EXPECT_THROW(full.CopyTo(std::span<std::int32_t>{table_destination}, std::span<std::int32_t>{base_destination}, 1),
                 std::invalid_argument);
}

TEST(FlatForwardOperation, CopyToRejectsOobAndMalformedPayloads) {
    FlatBlockTableExport export_owner{1, 2};
    export_owner.values = {1, 8};
    export_owner.bases = {0};
    std::vector<std::int32_t> table_destination(2, 0);
    std::vector<std::int32_t> base_destination(1, 0);
    EXPECT_THROW(
        export_owner.CopyTo(std::span<std::int32_t>{table_destination}, std::span<std::int32_t>{base_destination}, 8),
        std::invalid_argument);

    export_owner.values = {1, -2};
    EXPECT_THROW(
        export_owner.CopyTo(std::span<std::int32_t>{table_destination}, std::span<std::int32_t>{base_destination}, 8),
        std::invalid_argument);

    export_owner.values = {1, 2};
    export_owner.bases = {-1};
    EXPECT_THROW(
        export_owner.CopyTo(std::span<std::int32_t>{table_destination}, std::span<std::int32_t>{base_destination}, 8),
        std::invalid_argument);

    export_owner.bases = {0};
    export_owner.values.pop_back();
    EXPECT_THROW(
        export_owner.CopyTo(std::span<std::int32_t>{table_destination}, std::span<std::int32_t>{base_destination}, 8),
        std::logic_error);
}

}  // namespace
}  // namespace tokenspeed::test
