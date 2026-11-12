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

#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/coordinator/eviction_candidates.h"
#include "cache_test_access.h"
#include "cache/prefix/prefix_hasher.h"
#include "scheduler/operations/forward.h"

namespace tokenspeed::test {
namespace {

const std::array kSpecs{CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                                       .sliding_window = 8,
                                       .sink_tokens = 2,
                                       .cache_blocks_per_lcm_block = 1,
                                       .block_granularity = 4}};
const std::vector<std::string> kHashes{"a", "b", "c", "d", "e", "f"};

void Put(CacheCoordinator& coordinator, BlockPool& pool, std::int32_t slot, std::uint64_t epoch) {
    auto block = pool.AcquireBlock(0);
    ASSERT_TRUE(block);
    coordinator.GroupPrefixIndex(0).Register(pool, block, CacheKey{.group_id = 0, .content_hash = kHashes.at(slot)},
                                             epoch, slot, CacheBoundaryKind::kChunk, nullptr, nullptr);
}

TEST(SwaSink, SinkIsPublishedWhenTheTrailingWindowIsEmpty) {
    BlockPool pool(8, {1});
    const std::array specs{CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                                          .sliding_window = 1,
                                          .sink_tokens = 1,
                                          .cache_blocks_per_lcm_block = 1,
                                          .block_granularity = 1}};
    auto coordinator = MakeCoordinator(specs, 1, pool, false, nullptr, nullptr, false, false, 0);
    std::vector<BlockTable> tables(1);
    const std::array demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{6}}};
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, {}, {}, std::nullopt, std::nullopt, false, {}));
    coordinator.CacheCompletedBlocks(
        tables, RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = kHashes}, .num_computed_tokens = 6},
        1);
    coordinator.ReclaimExpired(tables, 6);
    EXPECT_TRUE(tables[0].Blocks()[0]);
    coordinator.Free(tables);
    EXPECT_EQ(coordinator.ProbePrefix(kHashes).device.num_common_tokens, 6);
}

TEST(SwaSink, DeviceMatchNeedsHeadAndWindow) {
    BlockPool pool(16, {1});
    auto coordinator = MakeCoordinator(kSpecs, 4, pool, false, nullptr, nullptr, false, false, 0);
    Put(coordinator, pool, 4, 1);
    Put(coordinator, pool, 5, 1);
    EXPECT_EQ(coordinator.ProbePrefix(kHashes).device.num_common_tokens, 0);
    Put(coordinator, pool, 0, 2);
    auto probe = coordinator.ProbePrefix(kHashes);
    EXPECT_EQ(probe.device.num_common_tokens, 24);
    EXPECT_EQ(probe.device.per_group[0].hits, (std::vector<std::uint8_t>{1, 0, 0, 0, 1, 1}));
    std::vector<BlockTable> tables(1);
    const std::array demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{4}}};
    auto admitted = coordinator.Admit(std::move(probe), demands, {}, {}, std::nullopt, std::nullopt, false, {});
    ASSERT_TRUE(admitted);
    ASSERT_TRUE(tables[0].Blocks()[0]);
    EXPECT_FALSE(tables[0].Blocks()[1]);
    coordinator.Free(tables);
}

TEST(SwaSink, MatchedHeadFitsTheExactLandingCapacity) {
    for (const auto shards : {1, 2}) {
        for (const bool host_hit : {false, true}) {
            for (const auto sink_tokens : {2, 6}) {
                for (const auto hit_tokens : {4, 16}) {
                    SCOPED_TRACE(::testing::Message() << "shards=" << shards << " host=" << host_hit
                                                      << " sink=" << sink_tokens << " hit=" << hit_tokens);
                    auto specs = kSpecs;
                    specs[0].sink_tokens = sink_tokens;
                    specs[0].cache_blocks_per_lcm_block = shards;
                    specs[0].shard_count = shards;
                    const auto head_blocks = (sink_tokens + 3) / 4;
                    const auto hit_blocks = hit_tokens / 4;
                    const auto first_window = std::max(0, hit_tokens - 7) / 4;
                    std::vector<std::int32_t> hit_slots;
                    for (auto slot = 0; slot < hit_blocks; ++slot) {
                        if (slot < head_blocks || slot >= first_window) hit_slots.push_back(slot);
                    }
                    const auto fresh = 2 + std::max(0, head_blocks - hit_blocks);
                    BlockPool pool((static_cast<std::int32_t>(hit_slots.size()) + fresh + shards - 1) / shards,
                                   {shards});
                    BlockPool host(16, {shards});
                    auto coordinator =
                        MakeCoordinator(specs, 4, pool, false, host_hit ? &host : nullptr, nullptr, false, host_hit, 0);
                    for (const auto slot : hit_slots) {
                        if (host_hit) {
                            auto block = coordinator.AcquireHostBlock(0, slot % shards);
                            coordinator.CacheHostBlock(block, CacheKey{.content_hash = kHashes[slot]}, std::nullopt);
                        } else {
                            Put(coordinator, pool, slot, 1);
                        }
                    }
                    auto probe = coordinator.ProbePrefix(std::span{kHashes}.first(hit_blocks));
                    ASSERT_EQ(std::max(probe.device.num_common_tokens, probe.host.num_common_tokens), hit_tokens);
                    std::vector<BlockTable> tables(1);
                    const std::array demands{GroupDemand{
                        .table = &tables[0],
                        .extent = SparseSuffix{.extent_tokens = 24, .first_block = 4, .head_blocks = head_blocks}}};
                    auto admitted =
                        coordinator.Admit(std::move(probe), demands, {}, {}, std::nullopt, std::nullopt, false, {});
                    ASSERT_TRUE(admitted);
                    for (auto slot = 0; slot < head_blocks; ++slot) EXPECT_TRUE(tables[0].Blocks()[slot]);
                    EXPECT_TRUE(tables[0].Blocks()[4]);
                    EXPECT_TRUE(tables[0].Blocks()[5]);
                    coordinator.Free(tables);
                }
            }
        }
    }
}

TEST(SwaSink, PartialSinkTailCopiesFitTheExactLandingCapacity) {
    for (const bool host_hit : {false, true}) {
        SCOPED_TRACE(host_hit);
        auto specs = kSpecs;
        specs[0].sink_tokens = 6;
        BlockPool pool(host_hit ? 4 : 5, {1});
        BlockPool host(8, {1});
        auto coordinator = MakeCoordinator(specs, 4, pool, false, &host, nullptr, false, true, 0);
        std::vector<std::int32_t> tokens(25);
        std::iota(tokens.begin(), tokens.end(), 1);
        std::vector<std::span<const std::int32_t>> pages;
        for (std::size_t i = 0; i < 6; ++i) pages.push_back(std::span{tokens}.subspan(i * 4, 4));
        const auto hashes = ComputePrefixHashes(pages, "");
        const TailPublication tail{.tail_tokens = 2,
                                   .hash = HashPrefixPage(std::span{tokens}.subspan(4, 2), hashes[0])};
        const std::array keys{CacheKey{.content_hash = hashes[0]}, CacheKey{.content_hash = tail.hash}};
        std::vector<BlockTable> published(1);
        const std::array publish_demands{GroupDemand{.table = &published[0], .extent = DenseGrowth{6}}};
        ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), publish_demands, {}, {}, std::nullopt, std::nullopt,
                                      false, {}));
        coordinator.CacheCompletedBlocks(
            published,
            RequestProgress{
                .completed_pages = CompletedPages{.prefix_hashes = std::span{hashes}.first(1), .tail = tail},
                .num_computed_tokens = 6},
            1);
        const auto original_ids = coordinator.Allocator(0).BlockTablePageIds(published[0]);
        coordinator.Free(published);
        if (host_hit) {
            for (const auto& key : keys) {
                auto block = coordinator.AcquireHostBlock(0, 0);
                coordinator.CacheHostBlock(block, key, std::nullopt);
            }
            // Move the existing entries to Host, preserving their published tail identity.
            for (const auto& key : keys) {
                if (const auto location = CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, key)) {
                    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *location));
                }
            }
        }
        auto probe = coordinator.ProbePromptPrefix({.prefix_hashes = hashes, .tokens = tokens, .max_tokens = 6});
        ASSERT_EQ(std::max(probe.device.num_common_tokens, probe.host.num_common_tokens), 6);
        std::vector<BlockTable> tables(1);
        const std::array demands{GroupDemand{
            .table = &tables[0], .extent = SparseSuffix{.extent_tokens = 24, .first_block = 4, .head_blocks = 2}}};
        auto admitted = coordinator.Admit(std::move(probe), demands, {}, {}, std::nullopt, std::nullopt, false, {});
        ASSERT_TRUE(admitted);
        EXPECT_EQ(admitted->load_pairs.size(), host_hit ? 2 : 0);
        EXPECT_EQ(admitted->pages_to_copy[0].size(), host_hit ? 0 : 1);
        const auto ids = coordinator.Allocator(0).BlockTablePageIds(tables[0]);
        EXPECT_GT(ids[0], 0);
        EXPECT_GT(ids[1], 0);
        if (!host_hit) {
            EXPECT_EQ(ids[0], original_ids[0]);
            EXPECT_NE(ids[1], original_ids[1]);
        }
        coordinator.Free(tables);
    }
}

TEST(SwaSink, HostOnlyMatchLoadsHeadAndWindow) {
    BlockPool pool(16, {1});
    BlockPool host(16, {1});
    auto coordinator = MakeCoordinator(kSpecs, 4, pool, false, &host, nullptr, false, true, 0);
    for (const auto slot : {0, 4, 5}) {
        auto block = coordinator.AcquireHostBlock(0, 0);
        ASSERT_TRUE(block);
        coordinator.CacheHostBlock(block, CacheKey{.group_id = 0, .content_hash = kHashes[slot]}, std::nullopt);
    }
    auto probe = coordinator.ProbePrefix(kHashes);
    EXPECT_EQ(probe.host.num_common_tokens, 24);
    std::vector<BlockTable> tables(1);
    const std::array demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{4}}};
    auto admitted = coordinator.Admit(std::move(probe), demands, {}, {}, std::nullopt, std::nullopt, false, {});
    ASSERT_TRUE(admitted);
    EXPECT_EQ(admitted->load_pairs.size(), 3);
    ASSERT_TRUE(tables[0].Blocks()[0]);
    EXPECT_FALSE(tables[0].Blocks()[1]);
    EXPECT_TRUE(tables[0].Blocks()[4]);
    EXPECT_TRUE(tables[0].Blocks()[5]);
    coordinator.Free(tables);
}

TEST(SwaSink, MixedHostHitUsesTheOrdinarySuffixSplit) {
    for (const bool sink_on_device : {false, true}) {
        SCOPED_TRACE(sink_on_device);
        BlockPool pool(32, {1, 1});
        BlockPool host(32, {1, 1});
        const std::array specs{
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
            kSpecs[0]};
        auto coordinator = MakeCoordinator(specs, 4, pool, false, &host, nullptr, false, true, 0);
        for (std::int32_t slot = 0; slot < 6; ++slot) {
            Put(coordinator, pool, slot, 1);
            auto block = coordinator.AcquireHostBlock(0, 0);
            coordinator.CacheHostBlock(block, CacheKey{.group_id = 0, .content_hash = kHashes[slot]}, std::nullopt);
        }
        for (const auto slot : {0, 4, 5}) {
            if (sink_on_device && slot == 0) {
                auto block = pool.AcquireBlock(1);
                coordinator.GroupPrefixIndex(1).Register(pool, block,
                                                         CacheKey{.group_id = 1, .content_hash = kHashes[slot]}, 1,
                                                         slot, CacheBoundaryKind::kChunk, nullptr, nullptr);
            } else {
                auto block = coordinator.AcquireHostBlock(1, 0);
                coordinator.CacheHostBlock(block, CacheKey{.group_id = 1, .content_hash = kHashes[slot]}, std::nullopt);
            }
        }
        auto probe = coordinator.ProbePrefix(kHashes);
        EXPECT_EQ(probe.host.num_common_tokens, 24);
        EXPECT_EQ(probe.device.full_page_tokens, sink_on_device ? 16 : 0);
        std::vector<BlockTable> tables(2);
        const std::array demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{4}},
                                 GroupDemand{.table = &tables[1], .extent = DenseGrowth{4}}};
        auto admitted = coordinator.Admit(std::move(probe), demands, {}, {}, std::nullopt, std::nullopt, false, {});
        ASSERT_TRUE(admitted);
        EXPECT_EQ(admitted->load_pairs.size(), sink_on_device ? 4 : 9);
        EXPECT_TRUE(tables[1].Blocks()[0]);
        EXPECT_FALSE(tables[1].Blocks()[1]);
        coordinator.Free(tables);
    }
}

TEST(SwaSink, ReclaimExportAndRestorePreserveHead) {
    BlockPool pool(16, {1});
    BlockPool snapshot(16, {1});
    auto coordinator = MakeCoordinator(kSpecs, 4, pool, false, nullptr, &snapshot, false, false, 0);
    std::vector<BlockTable> tables(1);
    const std::array demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{24}}};
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, {}, {}, std::nullopt, std::nullopt, false, {}));
    coordinator.ReclaimExpired(tables, 24);
    const auto ids = coordinator.Allocator(0).BlockTablePageIds(tables[0]);
    ASSERT_EQ(ids.size(), 6);
    EXPECT_GT(ids[0], 0);
    EXPECT_EQ(ids[1], 0);
    EXPECT_EQ(ids[2], 0);
    EXPECT_EQ(ids[3], 0);
    EXPECT_GT(ids[4], 0);
    EXPECT_GT(ids[5], 0);
    const std::array<std::span<const BlockTable>, 1> requests{tables};
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(requests), 3);
    PrefillOperation op;
    op.request_id = "sink";
    op.block_tables.emplace("swa", ForwardBlockTable{coordinator.Allocator(0), tables[0], 0});
    std::vector<ForwardOperation> operations;
    operations.emplace_back(std::move(op));
    const ForwardBatch batch{std::move(operations)};
    EXPECT_EQ(batch.block_tables.at("swa").WindowColumns(), 6);
    EXPECT_EQ(batch.block_tables.at("swa").Spans(), (std::vector<std::int32_t>{0, 6}));
    EXPECT_EQ(batch.block_tables.at("swa").AbsoluteRows(), ids);

    auto image = coordinator.TakeImage(tables, 24, std::vector<std::vector<ImageSlot>>(1));
    ASSERT_TRUE(image);
    ASSERT_EQ(image->image.tables[0].slots.size(), 3);
    image->store_pairs.clear();
    coordinator.Free(tables);
    std::vector<BlockTable> restored(1);
    const std::array restore_demands{GroupDemand{.table = &restored[0], .extent = DenseGrowth{0}}};
    ASSERT_TRUE(coordinator.Restore(image->image, restore_demands, std::nullopt, 1, false));
    const auto restored_ids = coordinator.Allocator(0).BlockTablePageIds(restored[0]);
    ASSERT_EQ(restored_ids.size(), ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) EXPECT_EQ(restored_ids[i] > 0, ids[i] > 0);
    coordinator.Free(restored);
}

TEST(SwaSink, DemotedChainStillEvictsItsSinkLast) {
    BlockPool pool(6, {1});
    auto coordinator = MakeCoordinator(kSpecs, 4, pool, false, nullptr, nullptr, false, false, 0);
    std::vector<CacheKey> keys;
    for (std::int32_t slot = 0; slot < 6; ++slot) {
        Put(coordinator, pool, slot, 10);
        keys.push_back(CacheKey{.content_hash = kHashes[slot]});
    }
    EXPECT_EQ(coordinator.DemoteEntries(keys), 6);
    std::vector<BlockTable> tables(1);
    const std::array demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{20}}};
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, {}, {}, std::nullopt, std::nullopt, true, {}));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, keys[0]));
    for (std::size_t i = 1; i < keys.size(); ++i) {
        EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, keys[i]));
    }
    coordinator.Free(tables);
}

TEST(SwaSink, PublicationRefreshesHeadAndEvictionKeepsBothEnds) {
    BlockPool pool(6, {1});
    auto coordinator = MakeCoordinator(kSpecs, 4, pool, false, nullptr, nullptr, false, false, 0);
    std::vector<BlockTable> tables(1);
    const std::array demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{24}}};
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, {}, {}, std::nullopt, std::nullopt, false, {}));
    // Publish each boundary while the whole request is resident, at one epoch.
    for (std::size_t count = 1; count <= kHashes.size(); ++count) {
        coordinator.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = std::span{kHashes}.first(count)},
                            .num_computed_tokens = static_cast<std::int32_t>(count) * 4},
            10);
    }
    coordinator.CacheCompletedBlocks(
        tables,
        RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = kHashes, .first_new_prefix_page = 5},
                        .num_computed_tokens = 24},
        11);
    const auto sink_location = tables[0].Blocks()[0]->Location();
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).MetadataFor(pool, sink_location)->last_access_epoch, 11);
    coordinator.Free(tables);
    // Evict three pages: the endpoint's sink and newest window outlive middle history.
    std::vector<BlockTable> other(1);
    const std::array other_demands{GroupDemand{.table = &other[0], .extent = DenseGrowth{12}}};
    ASSERT_TRUE(
        coordinator.Admit(coordinator.ProbePrefix({}), other_demands, {}, {}, std::nullopt, std::nullopt, true, {}));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, CacheKey{.content_hash = kHashes[0]}));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, CacheKey{.content_hash = kHashes[5]}));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, CacheKey{.content_hash = kHashes[1]}));
    coordinator.Free(other);
}

}  // namespace
}  // namespace tokenspeed::test
