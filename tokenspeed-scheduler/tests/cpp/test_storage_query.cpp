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

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/core/block_pool.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache/tier/storage_manager.h"

namespace tokenspeed::test {
namespace {

struct QueryPrompt {
    QueryPrompt(std::int32_t size, std::int32_t grain) : grain{grain}, tokens(size) {
        std::iota(tokens.begin(), tokens.end(), 1);
        std::vector<std::span<const std::int32_t>> pages;
        for (std::int32_t start = 0; start + grain <= size; start += grain) {
            pages.emplace_back(tokens.data() + start, grain);
        }
        hashes = ComputePrefixHashes(pages, "");
    }

    CacheCoordinator::PromptPrefix Prefix(std::int32_t bound, bool include_tail) const {
        CacheCoordinator::PromptPrefix result{
            .prefix_hashes = std::span{hashes}.first(bound / grain), .tokens = tokens, .max_tokens = bound};
        if (include_tail && bound % grain != 0) {
            const auto count = bound % grain;
            const auto prior = result.prefix_hashes.empty() ? ContentHash{} : result.prefix_hashes.back();
            result.tail = TailPublication{
                .tail_tokens = count, .hash = HashPrefixPage(std::span{tokens}.subspan(bound - count, count), prior)};
        }
        return result;
    }

    std::int32_t grain;
    std::vector<std::int32_t> tokens;
    std::vector<std::string> hashes;
};

std::vector<CacheGroupSpec> FullAndStates(std::int32_t grain, std::int32_t full_block, std::int32_t states) {
    std::vector<CacheGroupSpec> specs{{.attn_kind = AttnKind::kFull,
                                       .cache_blocks_per_lcm_block = grain / full_block,
                                       .block_granularity = full_block}};
    for (std::int32_t i = 0; i < states; ++i) {
        specs.push_back({.attn_kind = AttnKind::kState, .block_granularity = 0});
    }
    return specs;
}

std::vector<std::int32_t> Packing(std::span<const CacheGroupSpec> specs) {
    std::vector<std::int32_t> result;
    for (const auto& spec : specs) result.push_back(spec.cache_blocks_per_lcm_block);
    return result;
}

class QueryCache {
public:
    QueryCache(std::int32_t grain, std::vector<CacheGroupSpec> specs, std::int32_t parents)
        : device{parents, Packing(specs)},
          host{parents, Packing(specs)},
          coordinator{MakeCoordinator(specs, grain, device, true, &host, nullptr, true)},
          storage{coordinator,
                  StorageConfig{8, 256, parents, 256, 8192, 1, 1000, 1000, {8, 1048576, 1000}, {4, 524288, 500}}} {}

    void Warm(std::span<const CacheKey> keys) {
        for (const auto& key : keys) {
            auto block = coordinator.AcquireHostBlock(key.group_id, 0);
            ASSERT_TRUE(block);
            coordinator.CacheHostBlock(block, key);
        }
    }

    std::vector<CacheKey> Restore(const CacheCoordinator::PromptPrefix& prefix, std::span<const CacheKey> hits) {
        const auto id = storage.StartRead(prefix, hits);
        EXPECT_NE(id, 0u);
        if (!id) return {};
        const auto commands = storage.TakeCommands();
        EXPECT_EQ(commands.size(), 1u);
        if (commands.size() != 1 || !std::holds_alternative<StorageGetOperation>(commands[0])) return {};
        const auto operation = std::get<StorageGetOperation>(commands[0]);
        storage.Complete(cache::StorageDone{id, std::vector<std::uint8_t>(operation.keys.size(), 1), true,
                                            std::vector<std::uint8_t>(operation.keys.size(), 0)});
        storage.Detach(id);
        return operation.keys;
    }

    BlockPool device;
    BlockPool host;
    CacheCoordinator coordinator;
    StorageTransferManager storage;
};

bool Has(std::span<const CacheKey> keys, const CacheKey& key) {
    return std::ranges::find(keys, key) != keys.end();
}

class StorageQueryHighestTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(StorageQueryHighestTest, RestoresTheHighestCheckpointWithinTheActualKeyBudget) {
    QueryCache cache(128, FullAndStates(128, 128, 7), 256);
    QueryPrompt prompt(4097, 128);
    const auto prefix = prompt.Prefix(4097, true);
    const auto query = cache.coordinator.MissingStorageKeys(prefix, GetParam());
    EXPECT_EQ(query.max_tokens, 4097);
    EXPECT_LE(query.keys.size(), GetParam());
    for (std::uint32_t group = 1; group < 8; ++group) {
        EXPECT_TRUE(Has(query.keys, {.group_id = group, .content_hash = prefix.tail->hash}));
    }
    const auto restored = cache.Restore(prefix, query.keys);
    // Thirty-two Full pages plus its partial page and seven exact states.
    EXPECT_EQ(restored.size(), 40u);
    EXPECT_EQ(cache.coordinator.ProbePromptPrefix(prefix).host.num_common_tokens, 4097);
}

INSTANTIATE_TEST_SUITE_P(KeyBudgets, StorageQueryHighestTest, ::testing::Values(40u, 256u));

TEST(StorageQueryTest, MissingHighestStateFallsBackToTheNextCompleteCheckpoint) {
    QueryCache cache(4, FullAndStates(4, 4, 1), 64);
    QueryPrompt prompt(11, 4);
    const auto prefix = prompt.Prefix(11, true);
    const auto query = cache.coordinator.MissingStorageKeys(prefix, 5);
    ASSERT_EQ(query.max_tokens, 11);
    ASSERT_TRUE(Has(query.keys, {.group_id = 1, .content_hash = prefix.tail->hash}));
    ASSERT_TRUE(Has(query.keys, {.group_id = 1, .content_hash = prompt.hashes[1]}));
    std::vector<CacheKey> hits;
    for (const auto& key : query.keys) {
        if (key.group_id != 1 || key.content_hash != prefix.tail->hash) hits.push_back(key);
    }
    const auto restored = cache.Restore(prefix, hits);
    EXPECT_EQ(restored.size(), 3u);
    EXPECT_EQ(cache.coordinator.ProbePromptPrefix(prefix).host.num_common_tokens, 8);
}

TEST(StorageQueryTest, OversizedHighestCandidateStillAllowsACompleteLowerBoundary) {
    QueryCache cache(4, FullAndStates(4, 4, 1), 64);
    QueryPrompt prompt(11, 4);
    const auto prefix = prompt.Prefix(11, true);
    const auto query = cache.coordinator.MissingStorageKeys(prefix, 3);
    // T=11 needs four objects; T=8 fits exactly, without the T=4 state.
    EXPECT_EQ(query.max_tokens, 8);
    EXPECT_EQ(query.keys.size(), 3u);
    EXPECT_TRUE(Has(query.keys, {.group_id = 1, .content_hash = prompt.hashes[1]}));
    EXPECT_FALSE(Has(query.keys, {.group_id = 1, .content_hash = prompt.hashes[0]}));
    EXPECT_FALSE(Has(query.keys, {.group_id = 1, .content_hash = prefix.tail->hash}));
    cache.Restore(prefix, query.keys);
    EXPECT_EQ(cache.coordinator.ProbePromptPrefix(prefix).host.num_common_tokens, 8);
}

class StorageQueryWindowTest : public ::testing::TestWithParam<std::int32_t> {};

TEST_P(StorageQueryWindowTest, AShorterCheckpointQueriesItsOwnSlidingWindow) {
    const std::vector<CacheGroupSpec> specs{{.attn_kind = AttnKind::kSlidingWindow,
                                             .sliding_window = GetParam(),
                                             .cache_blocks_per_lcm_block = 2,
                                             .block_granularity = 2},
                                            {.attn_kind = AttnKind::kState, .block_granularity = 0}};
    QueryCache cache(4, specs, 64);
    QueryPrompt prompt(11, 4);
    const auto prefix = prompt.Prefix(11, true);
    // W=8 needs five objects at each boundary, with eight in their union.
    const auto query = cache.coordinator.MissingStorageKeys(prefix, 8);
    ASSERT_EQ(query.max_tokens, 11);
    ASSERT_TRUE(Has(query.keys, {.group_id = 1, .content_hash = prefix.tail->hash}));
    ASSERT_TRUE(Has(query.keys, {.group_id = 1, .content_hash = prompt.hashes[1]}));
    ASSERT_TRUE(Has(query.keys, {.group_id = 0, .content_hash = prompt.hashes[1], .page_offset = 0}));
    std::vector<CacheKey> hits;
    for (const auto& key : query.keys) {
        if (key.group_id != 1 || key.content_hash != prefix.tail->hash) hits.push_back(key);
    }
    cache.Restore(prefix, hits);
    EXPECT_EQ(cache.coordinator.ProbePromptPrefix(prefix).host.num_common_tokens, 8);
}

INSTANTIATE_TEST_SUITE_P(WindowSizes, StorageQueryWindowTest, ::testing::Values(4, 8));

TEST(StorageQueryTest, AnOversizedMiddleFallbackDoesNotHideASmallerCompleteOne) {
    const std::vector<CacheGroupSpec> specs{{.attn_kind = AttnKind::kSlidingWindow,
                                             .sliding_window = 4,
                                             .cache_blocks_per_lcm_block = 2,
                                             .block_granularity = 2},
                                            {.attn_kind = AttnKind::kState, .block_granularity = 0}};
    QueryCache cache(4, specs, 64);
    QueryPrompt prompt(11, 4);
    const std::vector<CacheKey> warm{{.group_id = 0, .content_hash = prompt.hashes[0], .page_offset = 0}};
    cache.Warm(warm);
    const auto prefix = prompt.Prefix(11, true);
    const auto query = cache.coordinator.MissingStorageKeys(prefix, 5);
    // T=11 costs three. T=8 needs three more, but warm T=4 needs only two.
    ASSERT_EQ(query.max_tokens, 11);
    ASSERT_EQ(query.keys.size(), 5u);
    EXPECT_FALSE(Has(query.keys, {.group_id = 1, .content_hash = prompt.hashes[1]}));
    EXPECT_FALSE(Has(query.keys, {.group_id = 0, .content_hash = prompt.hashes[1], .page_offset = 0}));
    EXPECT_FALSE(Has(query.keys, {.group_id = 0, .content_hash = prompt.hashes[1], .page_offset = 1}));
    EXPECT_TRUE(Has(query.keys, {.group_id = 1, .content_hash = prompt.hashes[0]}));
    std::vector<CacheKey> hits;
    for (const auto& key : query.keys) {
        if (key.group_id != 1 || key.content_hash != prefix.tail->hash) hits.push_back(key);
    }
    const auto restored = cache.Restore(prefix, hits);
    EXPECT_EQ(restored.size(), 2u);
    EXPECT_EQ(cache.coordinator.ProbePromptPrefix(prefix).host.num_common_tokens, 4);
}

TEST(StorageQueryTest, EvictedWarmObjectsCannotBeAssumedPresentByTheReadReservation) {
    QueryCache cache(4, FullAndStates(4, 2, 0), 64);
    QueryPrompt prompt(8, 4);
    const std::vector<CacheKey> warm{{.group_id = 0, .content_hash = prompt.hashes[0], .page_offset = 0},
                                     {.group_id = 0, .content_hash = prompt.hashes[0], .page_offset = 1}};
    cache.Warm(warm);
    const auto prefix = prompt.Prefix(8, false);
    const auto query = cache.coordinator.MissingStorageKeys(prefix, 2);
    ASSERT_EQ(query.max_tokens, 8);
    ASSERT_EQ(query.keys.size(), 2u);
    for (const auto& key : warm) {
        auto block = cache.coordinator.AcquireHostCachedBlock(key);
        ASSERT_TRUE(block);
        const auto location = block->Location();
        block.reset();
        ASSERT_TRUE(cache.coordinator.GroupPrefixIndex(0).Evict(cache.host, location));
    }
    EXPECT_EQ(cache.coordinator.ProbePromptPrefix(prefix).host.num_common_tokens, 0);
    const auto reservation = cache.coordinator.ReserveStoragePrefix(prefix, query.keys, 64);
    EXPECT_TRUE(reservation.keys.empty());
    EXPECT_TRUE(reservation.blocks.empty());
    EXPECT_EQ(cache.device.NumEmptyLcmBlocks(), 64);
}

TEST(StorageQueryTest, APartialHostBoundaryUsesAlignedQueryFloorAndDeviceReservationFloor) {
    QueryCache cache(4, FullAndStates(4, 2, 1), 64);
    QueryPrompt prompt(11, 4);
    const auto warm_prefix = prompt.Prefix(7, true);
    const std::vector<CacheKey> warm{{.group_id = 0, .content_hash = prompt.hashes[0], .page_offset = 0},
                                     {.group_id = 0, .content_hash = prompt.hashes[0], .page_offset = 1},
                                     {.group_id = 0, .content_hash = warm_prefix.tail->hash, .page_offset = 0},
                                     {.group_id = 0, .content_hash = warm_prefix.tail->hash, .page_offset = 1},
                                     {.group_id = 1, .content_hash = warm_prefix.tail->hash}};
    cache.Warm(warm);
    cache.coordinator.CacheHostTail(prompt.hashes[0], *warm_prefix.tail);
    const auto prefix = prompt.Prefix(11, true);
    const auto local = cache.coordinator.ProbePromptPrefix(prefix);
    ASSERT_EQ(local.host.num_common_tokens, 7);
    ASSERT_EQ(local.host.full_page_tokens, 4);
    ASSERT_EQ(local.device.num_common_tokens, 0);
    const auto query = cache.coordinator.MissingStorageKeys(prefix, 5);
    ASSERT_EQ(query.max_tokens, 11);
    ASSERT_EQ(query.keys.size(), 5u);
    EXPECT_TRUE(Has(query.keys, {.group_id = 0, .content_hash = prompt.hashes[1], .page_offset = 0}));
    EXPECT_TRUE(Has(query.keys, {.group_id = 0, .content_hash = prompt.hashes[1], .page_offset = 1}));
    const auto reservation = cache.coordinator.ReserveStoragePrefix(prefix, query.keys, 64);
    EXPECT_EQ(reservation.keys.size(), 5u);
    EXPECT_EQ(cache.coordinator.NumPinnedHostCachedBlocks(), 0);
    EXPECT_EQ(cache.coordinator.ProbePromptPrefix(prefix).host.num_common_tokens, 7);
}

TEST(StorageQuerySwaFloorTest, QuerySkipsBelowHostFloorAndWarmPinsEndWithReservation) {
    const std::vector<CacheGroupSpec> specs{
        {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        {.attn_kind = AttnKind::kState, .block_granularity = 0},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .cache_blocks_per_lcm_block = 2,
         .block_granularity = 2}};
    QueryCache cache(4, specs, 64);
    QueryPrompt prompt(9, 4);
    const auto warm_prefix = prompt.Prefix(7, true);
    const std::vector<CacheKey> warm{{.group_id = 0, .content_hash = prompt.hashes[0], .page_offset = 0},
                                     {.group_id = 0, .content_hash = prompt.hashes[0], .page_offset = 1},
                                     {.group_id = 0, .content_hash = warm_prefix.tail->hash, .page_offset = 0},
                                     {.group_id = 0, .content_hash = warm_prefix.tail->hash, .page_offset = 1},
                                     {.group_id = 1, .content_hash = warm_prefix.tail->hash},
                                     {.group_id = 2, .content_hash = prompt.hashes[0], .page_offset = 0},
                                     {.group_id = 2, .content_hash = prompt.hashes[0], .page_offset = 1},
                                     {.group_id = 2, .content_hash = warm_prefix.tail->hash, .page_offset = 0},
                                     {.group_id = 2, .content_hash = warm_prefix.tail->hash, .page_offset = 1}};
    cache.Warm(warm);
    cache.coordinator.CacheHostTail(prompt.hashes[0], *warm_prefix.tail);
    const auto prefix = prompt.Prefix(9, true);
    const auto local = cache.coordinator.ProbePromptPrefix(prefix);
    ASSERT_EQ(local.host.num_common_tokens, 7);
    ASSERT_EQ(local.host.full_page_tokens, 4);
    ASSERT_EQ(local.device.num_common_tokens, 0);
    ASSERT_EQ(local.device.full_page_tokens, 0);
    const auto pool_facts = [](const BlockPool& pool) {
        return std::array{pool.NumEmptyLcmBlocks(), pool.NumFreeSlots(0), pool.NumFreeSlots(1), pool.NumFreeSlots(2)};
    };
    const auto device_before = pool_facts(cache.device);
    const auto host_before = pool_facts(cache.host);
    const auto cached_before = cache.coordinator.NumHostCachedBlocks();
    const auto pinned_before = cache.coordinator.NumPinnedHostCachedBlocks();
    const auto query = cache.coordinator.MissingStorageKeys(prefix, 7);
    EXPECT_EQ(pool_facts(cache.device), device_before);
    EXPECT_EQ(pool_facts(cache.host), host_before);
    EXPECT_EQ(cache.coordinator.NumHostCachedBlocks(), cached_before);
    EXPECT_EQ(cache.coordinator.NumPinnedHostCachedBlocks(), pinned_before);
    ASSERT_EQ(query.max_tokens, 9);
    ASSERT_EQ(query.keys.size(), 7u);
    const std::vector<CacheKey> expected_query{{.group_id = 0, .content_hash = prompt.hashes[1], .page_offset = 0},
                                               {.group_id = 0, .content_hash = prompt.hashes[1], .page_offset = 1},
                                               {.group_id = 0, .content_hash = prefix.tail->hash, .page_offset = 0},
                                               {.group_id = 1, .content_hash = prefix.tail->hash},
                                               {.group_id = 2, .content_hash = prompt.hashes[1], .page_offset = 0},
                                               {.group_id = 2, .content_hash = prompt.hashes[1], .page_offset = 1},
                                               {.group_id = 2, .content_hash = prefix.tail->hash, .page_offset = 0}};
    for (const auto& key : expected_query) EXPECT_TRUE(Has(query.keys, key));
    EXPECT_FALSE(Has(query.keys, {.group_id = 2, .content_hash = prompt.hashes[0], .page_offset = 0}));
    EXPECT_FALSE(Has(query.keys, {.group_id = 2, .content_hash = prompt.hashes[0], .page_offset = 1}));

    const auto reservation = cache.coordinator.ReserveStoragePrefix(prefix, query.keys, 64);
    EXPECT_EQ(pool_facts(cache.device), device_before);
    ASSERT_EQ(reservation.keys.size(), 7u);
    ASSERT_EQ(reservation.blocks.size(), 7u);
    EXPECT_EQ(cache.coordinator.NumPinnedHostCachedBlocks(), pinned_before);
    const std::vector<CacheKey> expected_warm{{.group_id = 0, .content_hash = prompt.hashes[0], .page_offset = 0},
                                              {.group_id = 0, .content_hash = prompt.hashes[0], .page_offset = 1},
                                              // T=9, W=8 needs [2, 9): this block is below Query's F=4.
                                              {.group_id = 2, .content_hash = prompt.hashes[0], .page_offset = 1}};
    for (const auto& key : expected_warm) {
        EXPECT_TRUE(cache.coordinator.ContainsHostCachedBlock(key));
    }
    for (const auto& key : expected_query) EXPECT_TRUE(Has(reservation.keys, key));
    // Reserving unfilled destinations is not Host publication.
    EXPECT_EQ(cache.coordinator.ProbePromptPrefix(prefix).host.num_common_tokens, 7);
}

enum class WarmReadStage { kRunning, kReady, kDetached, kQuarantined, kDetachBeforeQuarantine };

class StorageWarmLifetimeTest : public ::testing::TestWithParam<std::tuple<std::uint32_t, WarmReadStage>> {};

TEST_P(StorageWarmLifetimeTest, ExistingDependenciesRemainEvictableAcrossTheGetLifetime) {
    const std::vector<CacheGroupSpec> specs{
        {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kState, .block_granularity = 0},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .cache_blocks_per_lcm_block = 2,
         .block_granularity = 2}};
    QueryCache cache(4, specs, 32);
    QueryPrompt prompt(8, 4);
    const auto prefix = prompt.Prefix(8, false);
    const std::vector<CacheKey> warm{{.group_id = 0, .content_hash = prompt.hashes[0]},
                                     {.group_id = 1, .content_hash = prompt.hashes[1]},
                                     {.group_id = 2, .content_hash = prompt.hashes[0], .page_offset = 1}};
    cache.Warm(warm);
    const auto query = cache.coordinator.MissingStorageKeys(prefix, 64);
    const auto id = cache.storage.StartRead(prefix, query.keys);
    ASSERT_NE(id, 0u);
    const auto commands = cache.storage.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    const auto op = std::get<StorageGetOperation>(commands[0]);
    EXPECT_EQ(cache.storage.NumHeldBlocks(), op.keys.size());
    const auto [group, stage] = GetParam();
    const auto complete = [&](bool retired) {
        cache.storage.Complete(cache::StorageDone{id, std::vector<std::uint8_t>(op.keys.size(), 1), retired,
                                                  std::vector<std::uint8_t>(op.keys.size(), 0)});
    };
    if (stage == WarmReadStage::kReady) {
        complete(true);
        ASSERT_EQ(cache.coordinator.ProbePromptPrefix(prefix).host.num_common_tokens, 8);
    } else if (stage == WarmReadStage::kDetached || stage == WarmReadStage::kDetachBeforeQuarantine) {
        cache.storage.Detach(id);
        if (stage == WarmReadStage::kDetachBeforeQuarantine) complete(false);
    } else if (stage == WarmReadStage::kQuarantined) {
        complete(false);
        cache.storage.Detach(id);
    }

    // The old dependency is not an I/O destination, even when its group is
    // needed to resume at T=8. Its eviction must shorten the next real probe.
    auto block = cache.coordinator.AcquireHostCachedBlock(warm[group]);
    ASSERT_TRUE(block);
    const auto location = block->Location();
    block.reset();
    ASSERT_TRUE(cache.coordinator.GroupPrefixIndex(group).Evict(cache.host, location));
    if (stage == WarmReadStage::kRunning || stage == WarmReadStage::kDetached) complete(true);
    EXPECT_LT(cache.coordinator.ProbePromptPrefix(prefix).host.num_common_tokens, 8);
    if (stage == WarmReadStage::kDetached) {
        EXPECT_FALSE(cache.storage.HasPending());
        EXPECT_EQ(cache.storage.NumHeldBlocks(), 0u);
    } else {
        EXPECT_EQ(cache.storage.NumHeldBlocks(), op.keys.size());
    }
    if (stage == WarmReadStage::kQuarantined || stage == WarmReadStage::kDetachBeforeQuarantine) {
        EXPECT_EQ(cache.storage.NumQuarantined(), 1u);
        EXPECT_FALSE(cache.storage.CanStart());
    }
}

INSTANTIATE_TEST_SUITE_P(FullStateAndWindow, StorageWarmLifetimeTest,
                         ::testing::Combine(::testing::Values(0u, 1u, 2u),
                                            ::testing::Values(WarmReadStage::kRunning, WarmReadStage::kReady,
                                                              WarmReadStage::kDetached, WarmReadStage::kQuarantined,
                                                              WarmReadStage::kDetachBeforeQuarantine)));

}  // namespace
}  // namespace tokenspeed::test
