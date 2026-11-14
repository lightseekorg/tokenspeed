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

// ForwardBatch resolves borrowed cache tables into one owned row-major buffer.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cache/allocator/group_allocator.h"
#include "fsm/forward_states.h"
#include "scheduler/operations/forward.h"

namespace tokenspeed::test {
namespace {

struct GroupSource {
    explicit GroupSource(std::int32_t packing)
        : pool{16, {packing}}, allocator{packing, /*group_id=*/0, /*shard_count=*/1, /*kept_head_blocks=*/0} {}

    BlockPool pool;
    GroupAllocator allocator;
    std::deque<BlockTable> rows;
    std::vector<CacheBlockRef> checkpoint_blocks;

    BlockTable& AddRow(const std::vector<std::int32_t>& labels) {
        std::vector<CacheBlockRef> blocks;
        blocks.reserve(labels.size());
        for (const std::int32_t label : labels) {
            if (label == 0) {
                blocks.emplace_back();
            } else {
                blocks.push_back(pool.AcquireBlock(/*group_id=*/0));
                if (!blocks.back()) throw std::runtime_error("test block pool exhausted");
            }
        }
        rows.push_back(BlockTable::FromBlocks(/*group_id=*/0, std::move(blocks), /*available_tokens=*/0));
        return rows.back();
    }

    std::int32_t AddCheckpoint() {
        checkpoint_blocks.push_back(pool.AcquireBlock(/*group_id=*/0));
        if (!checkpoint_blocks.back()) throw std::runtime_error("test block pool exhausted");
        return allocator.ResolveCacheBlockId(*checkpoint_blocks.back());
    }
};

struct Sources {
    GroupSource& AddGroup(std::string group, std::int32_t packing = 1) {
        auto source = std::make_unique<GroupSource>(packing);
        auto& result = *source;
        groups.emplace(std::move(group), std::move(source));
        return result;
    }

    void Attach(ForwardOperationBase& op, const std::string& group, const std::vector<std::int32_t>& labels,
                std::int32_t checkpoint_page_id = 0) {
        GroupSource& source = *groups.at(group);
        BlockTable& table = source.AddRow(labels);
        op.block_tables.emplace(group, ForwardBlockTable{source.allocator, table, checkpoint_page_id});
    }

    std::map<std::string, std::unique_ptr<GroupSource>> groups;
};

std::vector<std::int32_t> PageIds(const GroupSource& source, const BlockTable& table) {
    return source.allocator.BlockTablePageIds(table);
}

TEST(ForwardBatch, EmptyOpsProducesEmpty) {
    ForwardBatch batch{std::vector<ForwardOperation>{}};

    EXPECT_TRUE(batch.empty());
    EXPECT_EQ(batch.NumExtends(), 0u);
    EXPECT_TRUE(batch.request_ids.empty());
    EXPECT_TRUE(batch.block_tables.empty());
    EXPECT_TRUE(batch.BlockTables().empty());
}

TEST(ForwardBatch, OwnsInputWindowsAndPadsAtThePromptBoundaryAfterSourcesDie) {
    std::optional<ForwardBatch> batch;
    {
        TokenContainer tokens(std::vector<std::int32_t>{1, 2, 3, 4, 5});
        tokens.Extend({99, 100});
        fsm::ForwardResources resources;
        resources.token_container = &tokens;
        const std::array windows{TokenContainer::Window{0, 3, 0}, TokenContainer::Window{3, 2, 1},
                                 TokenContainer::Window{5, 0, 0}};
        std::vector<ForwardOperation> operations;
        DecodeOperation decode;
        decode.request_id = "decode";
        decode.input_length = 1;
        decode.decode_input_id = 99;
        operations.emplace_back(std::move(decode));
        for (std::size_t i = 0; i < windows.size(); ++i) {
            const auto info = fsm::MakePrefillInfo(resources, windows[i]);
            PrefillOperation operation;
            operation.request_id = std::to_string(i);
            operation.input_length = info.extend_len;
            operation.input_ids = info.input_ids;
            operation.shifted_input_ids = info.shifted_input_ids;
            operation.extend_prefix_len = info.already_scheduled_len;
            operation.extend_replay_len = info.replay_len;
            operations.emplace_back(std::move(operation));
        }
        batch.emplace(std::move(operations));
        tokens.Extend(std::vector<std::int32_t>(10000, 101));
    }
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"0", "1", "2", "decode"}));
    EXPECT_EQ(batch->input_ids, (std::vector<std::int32_t>{1, 2, 3, 3, 4, 5}));
    EXPECT_EQ(batch->shifted_input_ids, (std::vector<std::int32_t>{2, 3, 4, 4, 5, -1}));
    EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{3, 3, 0, 1}));
    EXPECT_EQ(batch->extend_replay_lens, (std::vector<std::int32_t>{0, 1, 0}));
    EXPECT_EQ(batch->decode_input_ids, (std::vector<std::int32_t>{99}));
}

TEST(ForwardBatch, ContiguousAndNestedTablesMatchForRealAllocatorIds) {
    Sources sources;
    GroupSource& full = sources.AddGroup("full", /*packing=*/3);
    GroupSource& swa = sources.AddGroup("swa", /*packing=*/2);
    auto& full0 = full.AddRow({1, 2, 3, 4});
    auto& full1 = full.AddRow({5});
    auto& swa0 = swa.AddRow({0, 6, 7});
    auto& swa1 = swa.AddRow({8});

    PrefillOperation p0;
    p0.request_id = "r0";
    const std::vector<std::int32_t> p0_tokens{1, 2, 3};
    p0.input_ids = p0_tokens;
    p0.input_length = 3;
    p0.block_tables.emplace("full", ForwardBlockTable{full.allocator, full0, 0});
    p0.block_tables.emplace("swa", ForwardBlockTable{swa.allocator, swa0, 0});
    PrefillOperation p1;
    p1.request_id = "r1";
    const std::vector<std::int32_t> p1_tokens{4};
    p1.input_ids = p1_tokens;
    p1.input_length = 1;
    p1.block_tables.emplace("full", ForwardBlockTable{full.allocator, full1, 0});
    p1.block_tables.emplace("swa", ForwardBlockTable{swa.allocator, swa1, 0});

    std::vector<ForwardOperation> ops;
    ops.emplace_back(std::move(p0));
    ops.emplace_back(std::move(p1));
    const ForwardBatch batch{std::move(ops)};
    const auto nested = batch.BlockTables();

    const auto expected_full0 = PageIds(full, full0);
    const auto expected_full1 = PageIds(full, full1);
    const auto expected_swa0 = PageIds(swa, swa0);
    const auto expected_swa1 = PageIds(swa, swa1);
    EXPECT_EQ(batch.block_tables.at("full").AbsoluteRows(),
              (std::vector<std::int32_t>{expected_full0[0], expected_full0[1], expected_full0[2], expected_full0[3],
                                         expected_full1[0], -1, -1, -1}));
    EXPECT_EQ(
        batch.block_tables.at("swa").AbsoluteRows(),
        (std::vector<std::int32_t>{expected_swa0[0], expected_swa0[1], expected_swa0[2], expected_swa1[0], -1, -1}));
    EXPECT_EQ(nested.at("full"),
              (std::vector<std::vector<std::int32_t>>{expected_full0, {expected_full1[0], -1, -1, -1}}));
    EXPECT_EQ(nested.at("swa"), (std::vector<std::vector<std::int32_t>>{expected_swa0, {expected_swa1[0], -1, -1}}));
}

TEST(ForwardBatch, NullHoleZeroAndPaddingMinusOneRemainDistinct) {
    Sources sources;
    GroupSource& full = sources.AddGroup("full");
    auto& with_hole = full.AddRow({0, 1, 2});
    auto& short_row = full.AddRow({3});
    std::vector<ForwardOperation> ops;
    PrefillOperation p0;
    p0.request_id = "r0";
    p0.block_tables.emplace("full", ForwardBlockTable{full.allocator, with_hole, 0});
    PrefillOperation p1;
    p1.request_id = "r1";
    p1.block_tables.emplace("full", ForwardBlockTable{full.allocator, short_row, 0});
    ops.emplace_back(std::move(p0));
    ops.emplace_back(std::move(p1));

    const ForwardBatch batch{std::move(ops)};
    const auto nested = batch.BlockTables();
    const auto& rows = batch.block_tables.at("full");
    ASSERT_EQ(batch.block_tables.at("full").AbsoluteRows().size(), 6u);
    EXPECT_EQ(batch.block_tables.at("full").AbsoluteRows().at(0), 0);
    EXPECT_EQ(batch.block_tables.at("full").AbsoluteRows().at(4), -1);
    EXPECT_EQ(rows.Spans(), (std::vector<std::int32_t>{0, 3, 0, 1}));
    EXPECT_EQ(rows.WindowRows(), (std::vector<std::int32_t>{0, full.allocator.BlockTablePageIds(with_hole)[1],
                                                            full.allocator.BlockTablePageIds(with_hole)[2],
                                                            full.allocator.BlockTablePageIds(short_row)[0], -1, -1}));
    EXPECT_EQ(nested.at("full").at(0).at(0), 0);
    EXPECT_EQ(nested.at("full").at(1),
              (std::vector<std::int32_t>{full.allocator.BlockTablePageIds(short_row)[0], -1, -1}));
}

TEST(ForwardBatch, CheckpointPageIsThirdStateSlot) {
    Sources sources;
    GroupSource& state = sources.AddGroup("state", /*packing=*/2);
    auto& row = state.AddRow({1, 2});
    const std::int32_t checkpoint = state.AddCheckpoint();
    std::vector<ForwardOperation> ops;
    PrefillOperation prefill;
    prefill.request_id = "r0";
    prefill.block_tables.emplace("state", ForwardBlockTable{state.allocator, row, checkpoint});
    ops.emplace_back(std::move(prefill));

    const ForwardBatch batch{std::move(ops)};
    const auto nested = batch.BlockTables();
    const auto ids = PageIds(state, row);
    EXPECT_EQ(batch.block_tables.at("state").AbsoluteRows(), (std::vector<std::int32_t>{ids[0], ids[1], checkpoint}));
    EXPECT_EQ(nested.at("state"), (std::vector<std::vector<std::int32_t>>{{ids[0], ids[1], checkpoint}}));
}

TEST(ForwardBatch, MissingGroupAndEmptyTableRowsArePadded) {
    Sources sources;
    GroupSource& full = sources.AddGroup("full");
    GroupSource& state = sources.AddGroup("state");
    auto& full_row = full.AddRow({1, 2});
    auto& empty_row = state.AddRow({});
    auto& state_row = state.AddRow({3, 4});
    std::vector<ForwardOperation> ops;
    PrefillOperation p0;
    p0.request_id = "r0";
    p0.block_tables.emplace("full", ForwardBlockTable{full.allocator, full_row, 0});
    PrefillOperation p1;
    p1.request_id = "r1";
    p1.block_tables.emplace("state", ForwardBlockTable{state.allocator, empty_row, 0});
    PrefillOperation p2;
    p2.request_id = "r2";
    p2.block_tables.emplace("state", ForwardBlockTable{state.allocator, state_row, 0});
    ops.emplace_back(std::move(p0));
    ops.emplace_back(std::move(p1));
    ops.emplace_back(std::move(p2));

    const ForwardBatch batch{std::move(ops)};
    const auto nested = batch.BlockTables();
    EXPECT_EQ(nested.at("full").at(1), (std::vector<std::int32_t>{-1, -1}));
    EXPECT_EQ(nested.at("state").at(0), (std::vector<std::int32_t>{-1, -1}));
    EXPECT_EQ(nested.at("state").at(1), (std::vector<std::int32_t>{-1, -1}));
}

TEST(ForwardBatch, ReclaimedPrefixUsesCompactWindowAndRaggedRowsStayAligned) {
    Sources sources;
    GroupSource& full = sources.AddGroup("full");
    auto& reclaimed = full.AddRow({1, 2, 3, 4, 5, 6, 7, 8});
    const auto ids = PageIds(full, reclaimed);
    full.allocator.ReclaimExpired(full.pool, reclaimed, /*num_expired_blocks=*/6);

    std::vector<ForwardOperation> ops;
    PrefillOperation with_prefix;
    with_prefix.request_id = "reclaimed";
    with_prefix.block_tables.emplace("full", ForwardBlockTable{full.allocator, reclaimed, 0});
    PrefillOperation missing_group;
    missing_group.request_id = "missing";
    ops.emplace_back(std::move(with_prefix));
    ops.emplace_back(std::move(missing_group));
    const ForwardBatch batch{std::move(ops)};
    const auto& table = batch.block_tables.at("full");

    EXPECT_EQ(table.Rows(), 2u);
    EXPECT_EQ(table.Columns(), 8u);
    EXPECT_EQ(table.WindowColumns(), 2u);
    EXPECT_EQ(table.Spans(), (std::vector<std::int32_t>{6, 8, 0, 0}));
    EXPECT_EQ(table.WindowRows(), (std::vector<std::int32_t>{ids[6], ids[7], -1, -1}));
    EXPECT_EQ(table.AbsoluteRows(),
              (std::vector<std::int32_t>{0, 0, 0, 0, 0, 0, ids[6], ids[7], -1, -1, -1, -1, -1, -1, -1, -1}));
}

TEST(ForwardBatch, AllReclaimedRowsKeepLogicalZeroWidthWindow) {
    Sources sources;
    GroupSource& full = sources.AddGroup("full");
    auto& row = full.AddRow({1, 2, 3, 4});
    full.allocator.ReclaimExpired(full.pool, row, /*num_expired_blocks=*/4);
    std::vector<ForwardOperation> ops;
    PrefillOperation prefill;
    prefill.request_id = "all-reclaimed";
    prefill.block_tables.emplace("full", ForwardBlockTable{full.allocator, row, 0});
    ops.emplace_back(std::move(prefill));

    const ForwardBatch batch{std::move(ops)};
    const auto& table = batch.block_tables.at("full");
    EXPECT_EQ(table.Columns(), 4u);
    EXPECT_EQ(table.WindowColumns(), 0u);
    EXPECT_EQ(table.Spans(), (std::vector<std::int32_t>{4, 4}));
    EXPECT_TRUE(table.WindowRows().empty());
    EXPECT_EQ(table.AbsoluteRows(), (std::vector<std::int32_t>{0, 0, 0, 0}));
}

TEST(ForwardBatch, PrefillBeforeDecodeKeepsRowsAligned) {
    Sources sources;
    GroupSource& full = sources.AddGroup("full");
    auto& decode_row = full.AddRow({1});
    auto& prefill_row = full.AddRow({2, 3});
    DecodeOperation decode;
    decode.request_id = "d";
    decode.request_pool_index = 3;
    decode.input_length = 1;
    decode.prefill_length = 6;
    decode.decode_input_id = 99;
    decode.block_tables.emplace("full", ForwardBlockTable{full.allocator, decode_row, 0});
    PrefillOperation prefill;
    prefill.request_id = "p";
    prefill.request_pool_index = 5;
    prefill.prefill_length = 4;
    const std::vector<std::int32_t> prefill_tokens{7, 8};
    prefill.input_ids = prefill_tokens;
    prefill.input_length = 2;
    prefill.block_tables.emplace("full", ForwardBlockTable{full.allocator, prefill_row, 0});
    std::vector<ForwardOperation> ops;
    ops.emplace_back(std::move(decode));
    ops.emplace_back(std::move(prefill));

    const ForwardBatch batch{std::move(ops)};
    const auto nested = batch.BlockTables();
    EXPECT_EQ(batch.request_ids, (std::vector<std::string>{"p", "d"}));
    // The per-request fields follow the rows through the partition.
    EXPECT_EQ(batch.request_pool_indices, (std::vector<std::int32_t>{5, 3}));
    EXPECT_EQ(batch.input_lengths, (std::vector<std::int32_t>{2, 1}));
    EXPECT_EQ(batch.prefill_lengths, (std::vector<std::int32_t>{4, 6}));
    EXPECT_EQ(nested.at("full").at(0), PageIds(full, prefill_row));
    EXPECT_EQ(nested.at("full").at(1),
              (std::vector<std::int32_t>{full.allocator.BlockTablePageIds(decode_row)[0], -1}));
    EXPECT_EQ(batch.input_ids, (std::vector<std::int32_t>{7, 8}));
    EXPECT_EQ(batch.decode_input_ids, (std::vector<std::int32_t>{99}));
}

ForwardBatch BuildBatchAfterSourcesGoOutOfScope() {
    Sources sources;
    GroupSource& full = sources.AddGroup("full", /*packing=*/2);
    auto& row = full.AddRow({1, 2, 3});
    const auto ids = PageIds(full, row);
    std::vector<ForwardOperation> ops;
    PrefillOperation prefill;
    prefill.request_id = "r0";
    prefill.block_tables.emplace("full", ForwardBlockTable{full.allocator, row, 0});
    ops.emplace_back(std::move(prefill));
    ForwardBatch batch{std::move(ops)};
    EXPECT_EQ(batch.block_tables.at("full").AbsoluteRows(), ids);
    return batch;
}

TEST(ForwardBatch, OwnsResolvedSnapshotAfterSourcesAreDestroyed) {
    const ForwardBatch batch = BuildBatchAfterSourcesGoOutOfScope();
    EXPECT_EQ(batch.request_ids, (std::vector<std::string>{"r0"}));
    ASSERT_EQ(batch.block_tables.at("full").AbsoluteRows().size(), 3u);
    const auto nested = batch.BlockTables();
    EXPECT_EQ(nested.at("full").at(0), batch.block_tables.at("full").AbsoluteRows());
}

TEST(ForwardBatch, OwnsResolvedSnapshotAfterSourceTableIsFreed) {
    Sources sources;
    GroupSource& full = sources.AddGroup("full", /*packing=*/2);
    auto& row = full.AddRow({1, 2, 3});
    const auto ids = PageIds(full, row);
    std::vector<ForwardOperation> ops;
    PrefillOperation prefill;
    prefill.request_id = "r0";
    prefill.block_tables.emplace("full", ForwardBlockTable{full.allocator, row, 0});
    ops.emplace_back(std::move(prefill));
    const ForwardBatch batch{std::move(ops)};
    const ForwardBatch copied{batch};

    full.allocator.Free(row);
    EXPECT_TRUE(full.allocator.BlockTablePageIds(row).empty());
    EXPECT_EQ(batch.block_tables.at("full").AbsoluteRows(), ids);
    EXPECT_EQ(batch.BlockTables().at("full").at(0), ids);
    EXPECT_EQ(copied.block_tables.at("full").WindowRows(), ids);
    EXPECT_EQ(copied.block_tables.at("full").WindowRows().data(), batch.block_tables.at("full").WindowRows().data());
}

TEST(ForwardBatch, AbsoluteRowsMaterializeOnceAcrossThreads) {
    Sources sources;
    GroupSource& full = sources.AddGroup("full");
    auto& row = full.AddRow({1, 2, 3, 4, 5});
    full.allocator.ReclaimExpired(full.pool, row, /*num_expired_blocks=*/3);
    const auto expected = PageIds(full, row);
    std::vector<ForwardOperation> ops;
    PrefillOperation prefill;
    prefill.request_id = "threaded";
    prefill.block_tables.emplace("full", ForwardBlockTable{full.allocator, row, 0});
    ops.emplace_back(std::move(prefill));
    const ForwardBatch batch{std::move(ops)};

    constexpr std::size_t kThreads = 8;
    std::vector<const std::vector<std::int32_t>*> materialized(kThreads);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (std::size_t i = 0; i < kThreads; ++i) {
        threads.emplace_back(
            [&batch, &materialized, i] { materialized[i] = &batch.block_tables.at("full").AbsoluteRows(); });
    }
    for (auto& thread : threads) thread.join();

    ASSERT_NE(materialized[0], nullptr);
    for (const auto* rows : materialized) EXPECT_EQ(rows, materialized[0]);
    EXPECT_EQ(*materialized[0], (std::vector<std::int32_t>{0, 0, 0, expected[3], expected[4]}));
}

TEST(ForwardBatch, CarriesExtendReplayAndCheckpointLengthsAlongRows) {
    std::vector<ForwardOperation> ops;
    PrefillOperation replayed;
    replayed.request_id = "r0";
    const std::vector<std::int32_t> replayed_tokens{1, 2, 3, 4, 5};
    replayed.input_ids = replayed_tokens;
    replayed.input_length = 5;
    replayed.extend_prefix_len = 16;
    replayed.extend_replay_len = 3;
    replayed.extend_state_checkpoint_len = 2;
    PrefillOperation plain;
    plain.request_id = "r1";
    const std::vector<std::int32_t> plain_tokens{6, 7};
    plain.input_ids = plain_tokens;
    plain.input_length = 2;
    DecodeOperation decode;
    decode.request_id = "d0";
    decode.input_length = 1;
    ops.emplace_back(std::move(decode));
    ops.emplace_back(std::move(replayed));
    ops.emplace_back(std::move(plain));

    const ForwardBatch batch{std::move(ops)};
    EXPECT_EQ(batch.request_ids, (std::vector<std::string>{"r0", "r1", "d0"}));
    EXPECT_EQ(batch.extend_prefix_lens, (std::vector<std::int32_t>{16, 0}));
    EXPECT_EQ(batch.extend_replay_lens, (std::vector<std::int32_t>{3, 0}));
    EXPECT_EQ(batch.extend_state_checkpoint_lens, (std::vector<std::int32_t>{2, 0}));
    EXPECT_EQ(batch.NumExtends(), 2);
}

}  // namespace
}  // namespace tokenspeed::test
