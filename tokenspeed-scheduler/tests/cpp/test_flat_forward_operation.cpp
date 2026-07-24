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

#include <algorithm>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cache/kv_cache_coordinator.h"
#include "scheduler/operations/forward.h"

namespace tokenspeed::test {
namespace {

struct TableShape {
    std::int32_t size{};
    std::int32_t base{};
    std::vector<std::int32_t> null_slots;
};

// Test rows use the same BlockTable view consumed by Scheduler production.
// The deque keeps every table vector stable until FlatForwardOperation has
// copied it into its contiguous export owner.
class LiveTableRows {
public:
    explicit LiveTableRows(std::vector<std::string> group_ids) : pool_{/*total_num_blocks=*/256} {
        schema_.reserve(group_ids.size());
        for (std::string& group_id : group_ids) {
            schema_.push_back(KvCacheGroupSchema{
                .group_id = std::move(group_id),
                .kind = AttnKind::kFull,
                .block_size = 1,
                .rows_per_page = 1,
                .entry_stride_tokens = 1,
                .sliding_window = 0,
                .pool_index = 0,
                .prefix_role = KvPrefixRole::kHistoryAnchor,
                .table_layout = KvTableLayout::kAbsolute,
                .owner_mask = 0,
            });
        }
    }

    template <typename Operation>
    void Attach(Operation& op, std::vector<TableShape> shapes) {
        if (shapes.size() != schema_.size()) {
            throw std::invalid_argument("test flat row shape differs from group schema");
        }
        rows_.emplace_back();
        std::vector<BlockTable>& tables = rows_.back();
        tables.reserve(shapes.size());
        for (const TableShape& shape : shapes) {
            if (shape.size < 0 || shape.base < 0) {
                throw std::invalid_argument("test flat table shape must be non-negative");
            }
            std::vector<bool> is_null(static_cast<std::size_t>(shape.size), false);
            for (std::int32_t slot : shape.null_slots) {
                if (slot < 0 || slot >= shape.size || is_null[static_cast<std::size_t>(slot)]) {
                    throw std::invalid_argument("test flat table has an invalid null slot");
                }
                is_null[static_cast<std::size_t>(slot)] = true;
            }
            const std::int32_t real_count = shape.size - static_cast<std::int32_t>(shape.null_slots.size());
            std::vector<BlockRef> blocks = pool_.AcquireBlocks(real_count);
            if (static_cast<std::int32_t>(blocks.size()) != real_count) {
                throw std::runtime_error("test flat block pool exhausted");
            }
            std::vector<BlockRef> refs;
            refs.reserve(static_cast<std::size_t>(shape.size));
            std::size_t block_index = 0;
            for (bool null_slot : is_null) {
                if (null_slot) {
                    refs.emplace_back();
                } else {
                    refs.push_back(std::move(blocks[block_index++]));
                }
            }
            BlockTable table;
            table.InitRange(shape.base, std::move(refs));
            tables.push_back(std::move(table));
        }
        op.flat_block_table_view = tables;
        op.flat_cache_schema = schema_;
    }

    std::vector<std::int32_t> PageIds(std::size_t row, std::size_t group) const {
        return BlockTablePageIds(rows_.at(row).at(group));
    }

private:
    BlockPool pool_;
    std::vector<KvCacheGroupSchema> schema_;
    std::deque<std::vector<BlockTable>> rows_;
};

void ExpectRowEq(std::span<const std::int32_t> actual, std::initializer_list<std::int32_t> expected) {
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
}

PrefillOperation MakePrefill(std::string id, std::vector<std::int32_t> input_ids = {}, std::int32_t pool_index = 0,
                             std::int32_t extend_prefix_len = 0) {
    PrefillOperation op;
    op.request_id = std::move(id);
    op.request_pool_index = pool_index;
    op.input_length = static_cast<std::int32_t>(input_ids.size());
    op.input_ids = std::move(input_ids);
    op.extend_prefix_len = extend_prefix_len;
    return op;
}

DecodeOperation MakeDecode(std::string id, std::int32_t decode_input_id = -1, std::int32_t pool_index = 0) {
    DecodeOperation op;
    op.request_id = std::move(id);
    op.request_pool_index = pool_index;
    op.input_length = 1;
    op.decode_input_id = decode_input_id;
    return op;
}

PrefillOperation MakeFlatPrefill(LiveTableRows& rows, std::string id, std::vector<TableShape> shapes,
                                 std::vector<std::int32_t> input_ids = {}, std::int32_t pool_index = 0,
                                 std::int32_t extend_prefix_len = 0) {
    PrefillOperation op = MakePrefill(std::move(id), std::move(input_ids), pool_index, extend_prefix_len);
    rows.Attach(op, std::move(shapes));
    return op;
}

DecodeOperation MakeFlatDecode(LiveTableRows& rows, std::string id, std::vector<TableShape> shapes,
                               std::int32_t decode_input_id = -1, std::int32_t pool_index = 0) {
    DecodeOperation op = MakeDecode(std::move(id), decode_input_id, pool_index);
    rows.Attach(op, std::move(shapes));
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
    LiveTableRows rows{{"full"}};
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakeFlatPrefill(rows, "r0", {{.size = 3}}));
    ops.emplace_back(MakeFlatPrefill(rows, "r1", {{.size = 1}}));
    const std::vector<std::int32_t> first = rows.PageIds(0, 0);
    const std::vector<std::int32_t> second = rows.PageIds(1, 0);

    FlatForwardOperation flat_op{std::move(ops)};

    ASSERT_EQ(flat_op.flat_block_tables.count("full"), 1u);
    const auto& full = flat_op.flat_block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    EXPECT_EQ(full.rows, 2u);
    EXPECT_EQ(full.cols, 3u);
    EXPECT_EQ(full.values, (std::vector<std::int32_t>{first[0], first[1], first[2], second[0], -1, -1}));
    EXPECT_TRUE(std::equal(full.Row(0).begin(), full.Row(0).end(), first.begin(), first.end()));
    ExpectRowEq(full.Row(1), {second[0], -1, -1});
    EXPECT_EQ(full.Row(1).data(), full.Row(0).data() + full.cols);
}

// Flat contract: 0 = real null-block hole, -1 = absent (pad) column.
TEST(FlatForwardOperation, NullHoleZeroDistinctFromPadMinusOne) {
    LiveTableRows rows{{"swa"}};
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakeFlatPrefill(rows, "r0", {{.size = 3, .null_slots = {0}}}));
    ops.emplace_back(MakeFlatPrefill(rows, "r1", {{.size = 1}}));
    const std::vector<std::int32_t> first = rows.PageIds(0, 0);
    const std::vector<std::int32_t> second = rows.PageIds(1, 0);

    FlatForwardOperation flat_op{std::move(ops)};

    const auto& swa = flat_op.flat_block_tables.at("swa");
    ASSERT_EQ(swa.size(), 2u);
    EXPECT_TRUE(std::equal(swa.Row(0).begin(), swa.Row(0).end(), first.begin(), first.end()));
    ExpectRowEq(swa.Row(1), {second[0], -1, -1});
    EXPECT_EQ(swa.Row(0)[0], 0);
    EXPECT_EQ(swa.Row(1)[1], -1);
}

TEST(FlatForwardOperation, PrefillBeforeDecodeKeepsRowsAlignedWithRequests) {
    LiveTableRows rows{{"full", "state"}};
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakeFlatDecode(rows, "d", {{.size = 1}, {.size = 1, .base = 7}},
                                    /*decode_input_id=*/99));
    ops.emplace_back(MakeFlatPrefill(rows, "p", {{.size = 2}, {.size = 1, .base = 3}},
                                     /*input_ids=*/{7, 8}));
    const std::vector<std::int32_t> decode = rows.PageIds(0, 0);
    const std::vector<std::int32_t> prefill = rows.PageIds(1, 0);

    FlatForwardOperation flat_op{std::move(ops)};

    ASSERT_EQ(flat_op.request_ids.size(), 2u);
    EXPECT_EQ(flat_op.request_ids.at(0), "p");
    EXPECT_EQ(flat_op.request_ids.at(1), "d");

    const auto& full = flat_op.flat_block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    ExpectRowEq(full.Row(0), {prefill[0], prefill[1]});
    ExpectRowEq(full.Row(1), {decode[0], -1});
    EXPECT_EQ(flat_op.flat_block_tables.at("state").bases, (std::vector<std::int32_t>{3, 7}));

    EXPECT_EQ(flat_op.num_extends(), 1u);
    EXPECT_EQ(flat_op.input_ids, (std::vector<std::int32_t>{7, 8}));
    EXPECT_EQ(flat_op.decode_input_ids, (std::vector<std::int32_t>{99}));
}

TEST(FlatForwardOperation, CompletionInputsStayAlignedAcrossStablePartition) {
    std::vector<ForwardOperation> ops;
    auto decode = MakeDecode("d");
    decode.flat_kv_completion_input = FlatKVCompletionInput{
        .table_generation = 11,
        .dispatch_seq = 7,
        .dispatch_raw_start = 7,
        .dispatch_raw_end = 8,
        .protected_raw_end = 9,
    };
    auto prefill = MakePrefill("p");
    prefill.flat_kv_completion_input = FlatKVCompletionInput{
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
    EXPECT_EQ(flat_op.request_ids, (std::vector<std::string>{"p", "d"}));
    EXPECT_EQ(flat_op.flat_kv_completion_inputs.at(0).table_generation, 22u);
    EXPECT_EQ(flat_op.flat_kv_completion_inputs.at(1).table_generation, 11u);
}

TEST(FlatForwardOperation, MixedPresenceOfCompletionInputsFailsClosed) {
    std::vector<ForwardOperation> ops;
    auto first = MakePrefill("a");
    first.flat_kv_completion_input = FlatKVCompletionInput{
        .table_generation = 1,
        .dispatch_seq = 0,
        .dispatch_raw_start = 0,
        .dispatch_raw_end = 1,
        .protected_raw_end = 1,
    };
    ops.emplace_back(std::move(first));
    ops.emplace_back(MakePrefill("b"));

    EXPECT_THROW(FlatForwardOperation(std::move(ops)), std::invalid_argument);
}

TEST(FlatForwardOperation, ScalarFieldsTrackPerRequestRows) {
    std::vector<ForwardOperation> ops;
    auto p0 = MakePrefill("r0", /*input_ids=*/{1, 2, 3}, /*pool_index=*/5);
    p0.occupied_pages = {10};
    auto p1 = MakePrefill("r1", /*input_ids=*/{4, 5}, /*pool_index=*/7);
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
    LiveTableRows rows{{"full"}};
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakeFlatPrefill(rows, "r0", {{.size = 2}}));
    ops.emplace_back(MakeFlatPrefill(rows, "r1", {{.size = 2}}));
    const std::vector<std::int32_t> first = rows.PageIds(0, 0);
    const std::vector<std::int32_t> second = rows.PageIds(1, 0);

    FlatForwardOperation flat_op{std::move(ops)};

    const auto& full = flat_op.flat_block_tables.at("full");
    ASSERT_EQ(full.size(), 2u);
    ExpectRowEq(full.Row(0), {first[0], first[1]});
    ExpectRowEq(full.Row(1), {second[0], second[1]});
}

TEST(FlatForwardOperation, CanonicalSchemaIndexAlignsRowsWhilePublicMapStaysLexical) {
    // Schema order deliberately differs from std::map key order. Production
    // rows all borrow this one coordinator-owned schema and never reorder.
    LiveTableRows rows{{"zeta", "alpha"}};
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakeFlatPrefill(rows, "r0", {{.size = 1}, {.size = 3}}));
    ops.emplace_back(MakeFlatPrefill(rows, "r1", {{.size = 2}, {.size = 1}}));
    const std::vector<std::int32_t> first_zeta = rows.PageIds(0, 0);
    const std::vector<std::int32_t> first_alpha = rows.PageIds(0, 1);
    const std::vector<std::int32_t> second_zeta = rows.PageIds(1, 0);
    const std::vector<std::int32_t> second_alpha = rows.PageIds(1, 1);

    FlatForwardOperation flat_op{std::move(ops)};

    ASSERT_EQ(flat_op.flat_block_tables.begin()->first, "alpha");
    const auto& zeta = flat_op.flat_block_tables.at("zeta");
    EXPECT_EQ(zeta.cols, 2u);
    ExpectRowEq(zeta.Row(0), {first_zeta[0], -1});
    ExpectRowEq(zeta.Row(1), {second_zeta[0], second_zeta[1]});

    const auto& alpha = flat_op.flat_block_tables.at("alpha");
    EXPECT_EQ(alpha.cols, 3u);
    ExpectRowEq(alpha.Row(0), {first_alpha[0], first_alpha[1], first_alpha[2]});
    ExpectRowEq(alpha.Row(1), {second_alpha[0], -1, -1});
}

TEST(FlatForwardOperation, CopyToUsesContiguousRectangle) {
    LiveTableRows rows{{"full"}};
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakeFlatPrefill(rows, "r0", {{.size = 3, .null_slots = {0}}}));
    ops.emplace_back(MakeFlatPrefill(rows, "r1", {{.size = 1, .base = 7}}));
    const std::vector<std::int32_t> first = rows.PageIds(0, 0);
    const std::vector<std::int32_t> second = rows.PageIds(1, 0);

    FlatForwardOperation flat_op{std::move(ops)};
    const FlatBlockTableExport& full = flat_op.flat_block_tables.at("full");
    std::vector<std::int32_t> table_destination(8, 99);
    std::vector<std::int32_t> base_destination(3, 99);
    const FlatBlockTableExport::CopyResult copied =
        full.CopyTo(std::span<std::int32_t>{table_destination}, std::span<std::int32_t>{base_destination},
                    /*page_id_upper_bound=*/8);

    EXPECT_EQ(copied.rows, 2u);
    EXPECT_EQ(copied.cols, 3u);
    EXPECT_EQ(table_destination, (std::vector<std::int32_t>{first[0], first[1], first[2], second[0], -1, -1, 99, 99}));
    EXPECT_EQ(base_destination, (std::vector<std::int32_t>{0, 7, 99}));
}

TEST(FlatForwardOperation, CopyToRejectsInvalidCapacityMatrix) {
    LiveTableRows rows{{"full"}};
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakeFlatPrefill(rows, "r0", {{.size = 3, .null_slots = {0}}}));
    ops.emplace_back(MakeFlatPrefill(rows, "r1", {{.size = 1, .base = 7}}));
    FlatForwardOperation flat_op{std::move(ops)};
    const FlatBlockTableExport& full = flat_op.flat_block_tables.at("full");

    struct Case {
        const char* name;
        std::size_t table_capacity;
        std::size_t base_capacity;
        std::int32_t page_id_upper_bound;
    };
    const Case cases[] = {
        {"short table destination", 5, 3, 8},
        {"short base destination", 8, 1, 8},
        {"page id exceeds pool bound", 8, 3, 1},
    };

    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        std::vector<std::int32_t> table_destination(test_case.table_capacity, 99);
        std::vector<std::int32_t> base_destination(test_case.base_capacity, 99);
        EXPECT_THROW(full.CopyTo(std::span<std::int32_t>{table_destination}, std::span<std::int32_t>{base_destination},
                                 test_case.page_id_upper_bound),
                     std::invalid_argument);
    }
}

}  // namespace
}  // namespace tokenspeed::test
