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
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/core/block_pool.h"
#include "cache_test_access.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kGrain = 4;
using Kind = CacheBoundaryKind;

CacheGroupSpec GroupSpec(AttnKind kind, std::int32_t packing) {
    return CacheGroupSpec{.kind = kind,
                          .sliding_window = kind == AttnKind::kSlidingWindow ? 8 : 0,
                          .cache_blocks_per_lcm_block = packing,
                          .block_granularity = kGrain};
}

std::vector<std::int32_t> PackingOf(const std::vector<CacheGroupSpec>& specs) {
    std::vector<std::int32_t> packing;
    for (const CacheGroupSpec& spec : specs) {
        packing.push_back(spec.cache_blocks_per_lcm_block);
    }
    return packing;
}

CacheKey Key(std::uint32_t group, const std::string& hash) {
    return CacheKey{.group_id = group, .content_hash = hash, .page_offset = 0};
}

struct PriorityCache {
    std::vector<CacheGroupSpec> specs;
    BlockPool pool;
    CacheCoordinator coordinator;

    PriorityCache(std::int32_t parents, std::vector<CacheGroupSpec> groups)
        : specs(std::move(groups)),
          pool(parents, PackingOf(specs)),
          coordinator(MakeCoordinator(specs, kGrain, pool, nullptr, false)) {}

    CacheBlockLocation Cache(std::uint32_t group, const std::string& hash, std::uint64_t epoch,
                             std::int32_t logical_index, Kind kind) {
        CacheBlockRef block = pool.AcquireBlock(group);
        EXPECT_TRUE(block);
        if (!block) {
            return {};
        }
        const CacheBlockLocation location = block->Location();
        coordinator.GroupPrefixIndex(group).Register(pool, block, Key(group, hash), epoch, logical_index, kind,
                                                     nullptr);
        return location;
    }

    bool Contains(std::uint32_t group, const std::string& hash) const {
        return coordinator.GroupPrefixIndex(group).Contains(pool, Key(group, hash));
    }

    std::vector<GroupDemand> Demands(std::vector<BlockTable>& tables, std::uint32_t group, std::int32_t tokens) const {
        std::vector<GroupDemand> demands;
        for (std::size_t i = 0; i < tables.size(); ++i) {
            demands.push_back(GroupDemand{.table = &tables[i], .extent = DenseGrowth{i == group ? tokens : 0}});
        }
        return demands;
    }

    std::vector<BlockTable> Hold(const StateSnapshot& snapshot) {
        std::vector<BlockTable> tables(specs.size());
        for (const CachedStateBlock& block : snapshot.blocks) {
            const std::int32_t slot = snapshot.boundary_tokens / specs[block.key.group_id].block_granularity - 1;
            std::vector<CacheBlockRef> refs(static_cast<std::size_t>(slot + 1));
            refs[static_cast<std::size_t>(slot)] = coordinator.AcquireDeviceCachedBlock(block.key);
            tables[block.key.group_id] = BlockTable::FromBlocks(std::move(refs), 0);
        }
        return tables;
    }

    bool Allocate(std::vector<BlockTable>& tables, std::uint32_t group, std::int32_t tokens) {
        const auto demands = Demands(tables, group, tokens);
        return coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt).has_value();
    }
};

TEST(StateEvictionPriorityTest, FreeCapacityPreservesUnrelatedChunks) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "chunk", 10, 0, Kind::kChunk);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    cache.coordinator.Free(tables);
    EXPECT_TRUE(cache.Contains(0, "chunk"));
}

TEST(StateEvictionPriorityTest, CapacityKeepsEpochBeforeBoundaryKind) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "endpoint", 1, 0, Kind::kEndpoint);
    cache.Cache(0, "promoted", 2, 1, Kind::kPromoted);
    cache.Cache(0, "chunk", 100, 2, Kind::kChunk);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_TRUE(cache.Contains(0, "chunk"));
    EXPECT_FALSE(cache.Contains(0, "endpoint"));
    EXPECT_TRUE(cache.Contains(0, "promoted"));
}

TEST(StateEvictionPriorityTest, AcquiredChunkDoesNotOverrideAnOlderEndpointEpoch) {
    PriorityCache cache(2, {GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "endpoint", 1, 0, Kind::kEndpoint);
    const auto chunk_location = cache.Cache(0, "chunk", 2, 1, Kind::kChunk);
    const std::vector<CacheKey> keys{Key(0, "chunk")};
    auto match =
        cache.coordinator.GroupPrefixIndex(0).AcquireMatched(cache.pool, keys, 0, GroupPrefixProbe{.hits = {1}}, 100);
    match.blocks.clear();
    ASSERT_TRUE(cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, chunk_location)->was_acquired);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_TRUE(cache.Contains(0, "chunk"));
    EXPECT_FALSE(cache.Contains(0, "endpoint"));
}

TEST(StateEvictionPriorityTest, StateChunksUseEpochAcrossGroupsAndAcquisitionOnlyWithinTheSameEpoch) {
    for (const std::uint64_t unacquired_epoch : {5u, 10u}) {
        SCOPED_TRACE(unacquired_epoch);
        PriorityCache cache(2, {GroupSpec(AttnKind::kMambaState, 1), GroupSpec(AttnKind::kMambaState, 1)});
        const auto acquired = cache.Cache(0, "acquired", 5, 0, Kind::kChunk);
        const auto unacquired = cache.Cache(1, "unacquired", unacquired_epoch, 0, Kind::kChunk);
        const std::vector<CacheKey> keys{Key(0, "acquired")};
        auto match =
            cache.coordinator.GroupPrefixIndex(0).AcquireMatched(cache.pool, keys, 0, GroupPrefixProbe{.hits = {1}}, 5);
        match.blocks.clear();
        const auto acquired_metadata = cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, acquired);
        const auto unacquired_metadata = cache.coordinator.GroupPrefixIndex(1).MetadataFor(cache.pool, unacquired);
        ASSERT_TRUE(acquired_metadata);
        ASSERT_TRUE(unacquired_metadata);
        ASSERT_TRUE(acquired_metadata->was_acquired);
        ASSERT_FALSE(unacquired_metadata->was_acquired);
        ASSERT_EQ(acquired_metadata->last_access_epoch, 5u);
        ASSERT_EQ(unacquired_metadata->last_access_epoch, unacquired_epoch);

        std::vector<BlockTable> tables(2);
        ASSERT_TRUE(cache.Allocate(tables, 0, 4));
        if (unacquired_epoch == 5) {
            EXPECT_TRUE(cache.Contains(0, "acquired")) << "a hit earns preference only within its epoch";
            EXPECT_FALSE(cache.Contains(1, "unacquired"));
        } else {
            EXPECT_FALSE(cache.Contains(0, "acquired")) << "the older epoch still loses across groups";
            EXPECT_TRUE(cache.Contains(1, "unacquired"));
        }
    }
}

TEST(StateEvictionPriorityTest, LatestKeepsItsOriginalTableReferenceWithoutChangingKind) {
    PriorityCache cache(2, {GroupSpec(AttnKind::kMambaState, 1)});
    const auto endpoint_location = cache.Cache(0, "endpoint", 7, 100, Kind::kEndpoint);
    const auto latest_location = cache.Cache(0, "latest", 7, 0, Kind::kChunk);
    ASSERT_LT(endpoint_location.lcm_block_id, latest_location.lcm_block_id);
    const std::vector<std::string> hashes{"latest"};
    const auto latest = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 4);
    ASSERT_TRUE(latest);
    auto source = cache.Hold(*latest);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, source, *latest));
    cache.coordinator.ReclaimExpired(source, 6);
    ASSERT_TRUE(source[0].Blocks()[0]);
    EXPECT_EQ(source[0].Blocks()[0].use_count(), 2u) << "protection reuses the table's existing reference";
    const auto copied_identity = *latest;
    EXPECT_EQ(source[0].Blocks()[0].use_count(), 2u) << "copying snapshot metadata must not add a pin";
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, copied_identity));
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_FALSE(cache.Contains(0, "endpoint"));
    ASSERT_TRUE(cache.Contains(0, "latest"));
    const auto metadata = cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, latest_location);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->boundary_kind, Kind::kChunk);
    EXPECT_EQ(metadata->generation, latest->blocks[0].generation);
    EXPECT_EQ(cache.coordinator.GroupPrefixIndex(0).NumPinnedEntries(cache.pool), 1);
}

TEST(StateEvictionPriorityTest, LatestRemainsPinnedWhenEveryOtherEpochIsNewer) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "latest", 1, 0, Kind::kChunk);
    cache.Cache(0, "endpoint", 2, 1, Kind::kEndpoint);
    cache.Cache(0, "ordinary", 100, 2, Kind::kChunk);
    const std::vector<std::string> hashes{"latest"};
    const auto latest = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 4);
    ASSERT_TRUE(latest);
    auto source = cache.Hold(*latest);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, source, *latest));
    cache.coordinator.ReclaimExpired(source, 6);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    EXPECT_FALSE(cache.Contains(0, "ordinary"));
    EXPECT_TRUE(cache.Contains(0, "latest"));
    EXPECT_FALSE(cache.Contains(0, "endpoint"));
}

TEST(StateEvictionPriorityTest, ReplacingOneRequestsLatestKeepsAnotherRequestsSharedSnapshot) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "shared", 1, 0, Kind::kChunk);
    cache.Cache(0, "replacement", 1, 1, Kind::kChunk);
    cache.Cache(0, "ordinary", 1, 2, Kind::kChunk);
    const std::vector<std::string> shared_hashes{"shared"};
    const std::vector<std::string> replacement_hashes{"shared", "replacement"};
    const auto shared = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, shared_hashes, 4);
    const auto replacement = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, replacement_hashes, 8);
    ASSERT_TRUE(shared);
    ASSERT_TRUE(replacement);
    std::vector<BlockTable> first(1);
    first[0] = BlockTable::FromBlocks({cache.coordinator.AcquireDeviceCachedBlock(Key(0, "shared")),
                                       cache.coordinator.AcquireDeviceCachedBlock(Key(0, "replacement"))},
                                      0);
    auto second = cache.Hold(*shared);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, first, *shared));
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, second, *shared));
    cache.coordinator.ReclaimExpired(first, 6);
    cache.coordinator.ReclaimExpired(second, 6);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, first, *replacement));
    EXPECT_FALSE(first[0].Blocks()[0]);
    EXPECT_TRUE(second[0].Blocks()[0]);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_FALSE(cache.Contains(0, "ordinary"));
    EXPECT_TRUE(cache.Contains(0, "shared"));
    EXPECT_TRUE(cache.Contains(0, "replacement"));
    cache.coordinator.ClearProtectedStateSnapshot(second);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_FALSE(cache.Contains(0, "shared"));
    EXPECT_TRUE(cache.Contains(0, "replacement"));
}

TEST(StateEvictionPriorityTest, StaleSameKeyGenerationCannotProtectANewRegistration) {
    PriorityCache cache(2, {GroupSpec(AttnKind::kMambaState, 1)});
    const auto original = cache.Cache(0, "latest", 10, 0, Kind::kChunk);
    const std::vector<std::string> hashes{"latest"};
    const auto stale = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 4);
    ASSERT_TRUE(stale);
    ASSERT_TRUE(cache.coordinator.GroupPrefixIndex(0).Evict(cache.pool, original));
    cache.Cache(0, "endpoint", 100, 1, Kind::kEndpoint);
    cache.Cache(0, "latest", 100, 0, Kind::kChunk);
    const auto current = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 4);
    ASSERT_TRUE(current);
    ASSERT_NE(current->blocks[0].generation, stale->blocks[0].generation);
    auto source = cache.Hold(*current);
    EXPECT_FALSE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, source, *stale));
    EXPECT_EQ(source[0].ProtectedSlot(), -1);
    cache.coordinator.Free(source);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_FALSE(cache.Contains(0, "latest"));
    EXPECT_TRUE(cache.Contains(0, "endpoint"));
}

TEST(StateEvictionPriorityTest, PartialSnapshotDoesNotProtectItsRemainingGroupAsACompleteLatest) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1), GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "latest", 100, 0, Kind::kChunk);
    const auto missing = cache.Cache(1, "latest", 100, 0, Kind::kChunk);
    cache.Cache(0, "endpoint", 100, 1, Kind::kEndpoint);
    const std::vector<std::string> hashes{"latest"};
    const auto latest = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 4);
    ASSERT_TRUE(latest);
    ASSERT_EQ(latest->blocks.size(), 2u);
    ASSERT_TRUE(cache.coordinator.GroupPrefixIndex(1).Evict(cache.pool, missing));
    cache.Cache(1, "other-endpoint", 100, 1, Kind::kEndpoint);
    auto source = cache.Hold(*latest);
    EXPECT_FALSE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, source, *latest));
    EXPECT_EQ(source[0].ProtectedSlot(), -1);
    EXPECT_EQ(source[1].ProtectedSlot(), -1);
    cache.coordinator.Free(source);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_FALSE(cache.Contains(0, "latest"));
    EXPECT_TRUE(cache.Contains(0, "endpoint"));
    EXPECT_TRUE(cache.Contains(1, "other-endpoint"));
}

TEST(StateEvictionPriorityTest, CompleteLatestProtectsEveryStateGroup) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1), GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "latest", 1, 0, Kind::kChunk);
    cache.Cache(1, "latest", 1, 0, Kind::kChunk);
    cache.Cache(1, "ordinary", 1, 1, Kind::kChunk);
    const std::vector<std::string> hashes{"latest"};
    const auto latest = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 4);
    ASSERT_TRUE(latest);
    ASSERT_EQ(latest->blocks.size(), 2u);
    auto source = cache.Hold(*latest);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, source, *latest));
    cache.coordinator.ReclaimExpired(source, 6);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_FALSE(cache.Contains(1, "ordinary"));
    EXPECT_TRUE(cache.Contains(0, "latest"));
    EXPECT_TRUE(cache.Contains(1, "latest"));
}

TEST(StateEvictionPriorityTest, StateGroupsLeaveFullHistoryEpochAndSuffixOrderingUnchanged) {
    PriorityCache cache(5, {GroupSpec(AttnKind::kFull, 1), GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "old-head", 1, 0, Kind::kChunk);
    cache.Cache(0, "old-tail", 1, 1, Kind::kChunk);
    cache.Cache(0, "new-head", 2, 0, Kind::kChunk);
    cache.Cache(1, "endpoint", 3, 0, Kind::kEndpoint);
    cache.Cache(1, "ordinary", 100, 1, Kind::kChunk);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    EXPECT_TRUE(cache.Contains(1, "ordinary"));
    EXPECT_FALSE(cache.Contains(0, "old-tail"));
    EXPECT_FALSE(cache.Contains(0, "old-head"));
    EXPECT_TRUE(cache.Contains(0, "new-head"));
    EXPECT_TRUE(cache.Contains(1, "endpoint"));
}

TEST(StateEvictionPriorityTest, NoStateGroupsKeepOriginalOrdering) {
    PriorityCache cache(2, {GroupSpec(AttnKind::kFull, 1), GroupSpec(AttnKind::kSlidingWindow, 1)});
    cache.Cache(0, "old-full", 1, 0, Kind::kChunk);
    cache.Cache(1, "new-window", 100, 0, Kind::kChunk);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_FALSE(cache.Contains(0, "old-full"));
    EXPECT_TRUE(cache.Contains(1, "new-window"));
}

TEST(StateEvictionPriorityTest, ZeroEpochCachedEntryRemainsVisibleToNormalIndexTraversal) {
    PriorityCache cache(2, {GroupSpec(AttnKind::kFull, 1)});
    cache.Cache(0, "zero-epoch", 0, 0, Kind::kChunk);
    cache.Cache(0, "later", 1, 1, Kind::kChunk);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_FALSE(cache.Contains(0, "zero-epoch"));
    EXPECT_TRUE(cache.Contains(0, "later"));
}

TEST(StateEvictionPriorityTest, CurrentPrefixHitIsProtectedEvenWhenItIsAnOrdinaryChunk) {
    PriorityCache cache(2, {GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "hit", 100, 0, Kind::kChunk);
    cache.Cache(0, "endpoint", 1, 1, Kind::kEndpoint);
    const std::vector<std::string> hashes{"hit"};
    std::vector<BlockTable> tables(1);
    const auto demands = cache.Demands(tables, 0, 4);
    ASSERT_TRUE(
        cache.coordinator.Admit(cache.coordinator.ProbePrefix(hashes), demands, RequestProgress{}, std::nullopt));
    EXPECT_TRUE(cache.Contains(0, "hit"));
    EXPECT_FALSE(cache.Contains(0, "endpoint"));
}

TEST(StateEvictionPriorityTest, ExtraOwnerKeepsAChunkOutOfCapacityEviction) {
    PriorityCache cache(2, {GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "chunk", 100, 0, Kind::kChunk);
    cache.Cache(0, "endpoint", 1, 1, Kind::kEndpoint);
    const CacheBlockRef pin = cache.coordinator.GroupPrefixIndex(0).Find(cache.pool, Key(0, "chunk"));
    ASSERT_TRUE(pin);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_TRUE(cache.Contains(0, "chunk"));
    EXPECT_FALSE(cache.Contains(0, "endpoint"));
}

TEST(StateEvictionPriorityTest, UncachedExpiredRequestBlockPrecedesAllCachedCandidates) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    const auto expired = tables[0].Blocks()[0]->Location();
    cache.Cache(0, "ordinary", 1, 0, Kind::kChunk);
    auto demands = cache.Demands(tables, 0, 4);
    const RequestProgress progress{.num_computed_tokens = 6};
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_TRUE(cache.Contains(0, "ordinary"));
    EXPECT_FALSE(tables[0].Blocks()[0]);
    EXPECT_EQ(tables[0].Blocks().back()->Location(), expired);
}

TEST(StateEvictionPriorityTest, RegisteredZeroEpochEndpointUsesTheOriginalLocationTieBreak) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 12));
    const auto registered = tables[0].Blocks()[0]->Location();
    const auto unregistered = tables[0].Blocks()[1]->Location();
    const std::vector<std::string> hashes{"zero-epoch"};
    ASSERT_TRUE(
        CacheCoordinatorTestAccess::PublishStateSnapshot(cache.coordinator, tables, hashes, 4, 0, Kind::kEndpoint));
    ASSERT_TRUE(cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, registered));
    ASSERT_FALSE(cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, unregistered));

    // Both old slots are request-reclaimable. With no state-first phase,
    // registered and unregistered epoch-zero entries share the original
    // location tie-break; the earlier Endpoint is still capacity-evictable.
    auto demands = cache.Demands(tables, 0, 4);
    const RequestProgress progress{.num_computed_tokens = 10};
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_FALSE(cache.Contains(0, "zero-epoch"));
    const auto metadata = cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, registered);
    EXPECT_FALSE(metadata);
    EXPECT_FALSE(tables[0].Blocks()[0]);
    EXPECT_FALSE(tables[0].Blocks()[1]);
    EXPECT_EQ(tables[0].Blocks().back()->Location(), unregistered);
}

TEST(StateEvictionPriorityTest, GuaranteedChunkCleanupPreservesAnOtherwiseUnneededEndpoint) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    const std::vector<std::string> hashes{"reclaimable"};
    auto snapshot =
        CacheCoordinatorTestAccess::PublishStateSnapshot(cache.coordinator, tables, hashes, 4, 100, Kind::kChunk);
    ASSERT_TRUE(snapshot);
    cache.Cache(0, "endpoint", 1, 1, Kind::kEndpoint);
    ASSERT_EQ(cache.pool.NumEmptyLcmBlocks(), 0);
    ASSERT_EQ(cache.coordinator.GroupPrefixIndex(0).NumEntries(cache.pool), 2);
    auto demands = cache.Demands(tables, 0, 4);
    const RequestProgress progress{.num_computed_tokens = 6};
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    // The expired Chunk is guaranteed to be released by this commit. Its
    // capacity must be counted before considering the older Endpoint.
    EXPECT_FALSE(cache.Contains(0, "reclaimable"));
    EXPECT_TRUE(cache.Contains(0, "endpoint"));
    EXPECT_EQ(cache.coordinator.GroupPrefixIndex(0).NumEntries(cache.pool), 1);
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 0);
}

TEST(StateEvictionPriorityTest, GuaranteedCleanupExcludesAnotherLiveRequestsLatest) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    const auto latest = CacheCoordinatorTestAccess::PublishStateSnapshot(
        cache.coordinator, tables, std::vector<std::string>{"latest"}, 4, 100, Kind::kChunk);
    ASSERT_TRUE(latest);
    auto other_request = cache.Hold(*latest);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, other_request, *latest));
    cache.coordinator.ReclaimExpired(other_request, 6);
    cache.Cache(0, "endpoint", 1, 1, Kind::kEndpoint);
    auto demands = cache.Demands(tables, 0, 4);
    const RequestProgress progress{.num_computed_tokens = 6};
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *latest));
    EXPECT_FALSE(cache.Contains(0, "endpoint")) << "the live latest was not guaranteed free capacity";
    EXPECT_FALSE(tables[0].Blocks()[0]);
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 0);
}

TEST(StateEvictionPriorityTest, SharedPinPreventsCleanupCreditAndFailedAdmissionIsAtomic) {
    PriorityCache cache(2, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    const auto snapshot = CacheCoordinatorTestAccess::PublishStateSnapshot(
        cache.coordinator, tables, std::vector<std::string>{"shared"}, 4, 1, Kind::kChunk);
    ASSERT_TRUE(snapshot);
    const auto first = tables[0].Blocks()[0]->Location();
    const auto second = tables[0].Blocks()[1]->Location();
    CacheBlockRef reader = cache.coordinator.AcquireDeviceCachedBlock(Key(0, "shared"));
    ASSERT_TRUE(reader);
    auto demands = cache.Demands(tables, 0, 4);
    const RequestProgress progress{.num_computed_tokens = 6};
    EXPECT_FALSE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    ASSERT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[0].ReclaimedPrefixBlocks(), 0);
    EXPECT_EQ(tables[0].Blocks()[0]->Location(), first);
    EXPECT_EQ(tables[0].Blocks()[1]->Location(), second);
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *snapshot));
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 0);

    const std::array<CacheBlockRef*, 1> refs{&reader};
    cache.coordinator.ReleaseDeviceBlockRefs(refs);
    auto oversized = cache.Demands(tables, 0, 8);
    // Even after counting the guaranteed release, two new pages do not fit.
    // A failed shadow plan must leave that credited Chunk untouched.
    EXPECT_FALSE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), oversized, progress, std::nullopt));
    ASSERT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[0].ReclaimedPrefixBlocks(), 0);
    EXPECT_EQ(tables[0].Blocks()[0]->Location(), first);
    EXPECT_EQ(tables[0].Blocks()[1]->Location(), second);
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *snapshot));
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 0);
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_FALSE(cache.Contains(0, "shared"));
    EXPECT_FALSE(tables[0].Blocks()[0]);
    EXPECT_EQ(tables[0].Blocks().back()->Location(), first);
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 0);
}

TEST(StateEvictionPriorityTest, CleanupCreditForOnePackedChildDoesNotInventAnEmptyParent) {
    PriorityCache cache(2, {GroupSpec(AttnKind::kMambaState, 2), GroupSpec(AttnKind::kFull, 1)});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    const auto first = tables[0].Blocks()[0]->Location();
    const auto working = tables[0].Blocks()[1]->Location();
    ASSERT_EQ(first.lcm_block_id, working.lcm_block_id);
    ASSERT_TRUE(CacheCoordinatorTestAccess::PublishStateSnapshot(
        cache.coordinator, tables, std::vector<std::string>{"expired"}, 4, 100, Kind::kChunk));
    const auto history = cache.Cache(1, "history", 1, 0, Kind::kChunk);
    auto demands = cache.Demands(tables, 1, 4);
    const RequestProgress progress{.num_computed_tokens = 6};
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_FALSE(cache.Contains(0, "expired"));
    EXPECT_FALSE(cache.Contains(1, "history")) << "one state child cannot fund a different group's parent";
    EXPECT_EQ(tables[0].Blocks()[1]->Location(), working);
    EXPECT_EQ(tables[1].Blocks()[0]->Location(), history);
    EXPECT_EQ(cache.pool.OccupiedCount(working.lcm_block_id), 1);
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 0);
}

TEST(StateEvictionPriorityTest, PendingChunkCanonicalThatGainsAWorkingReferenceIsNotCleanupCredit) {
    PriorityCache cache(4, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 12));
    const auto canonical = tables[0].Blocks()[0]->Location();
    const auto snapshot = CacheCoordinatorTestAccess::PublishStateSnapshot(
        cache.coordinator, tables, std::vector<std::string>{"same"}, 4, 100, Kind::kChunk);
    ASSERT_TRUE(snapshot);
    // Keep the duplicate source alive after Register replaces its table ref;
    // otherwise that incidental release could mask an invalid capacity credit.
    CacheBlockRef source_pin = tables[0].Blocks()[1];
    cache.Cache(0, "endpoint", 1, 2, Kind::kEndpoint);
    const std::vector<std::string> hashes{"same", "same"};
    auto demands = cache.Demands(tables, 0, 4);
    const std::array materialized{8};
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .first_new_prefix_page = 1,
                .boundary_kind = Kind::kChunk,
                .materialized_state_boundaries = materialized,
            },
        .num_computed_tokens = 8,
    };
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_FALSE(tables[0].Blocks()[0]);
    ASSERT_TRUE(tables[0].Blocks()[1]);
    EXPECT_EQ(tables[0].Blocks()[1]->Location(), canonical);
    EXPECT_TRUE(cache.Contains(0, "same"));
    EXPECT_FALSE(cache.Contains(0, "endpoint"));
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 0);
}

class PendingStateEvictionPriorityTest : public testing::TestWithParam<Kind> {};

TEST_P(PendingStateEvictionPriorityTest, FutureRetainedPublicationIsNotClassifiedAsUncached) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    const auto future_endpoint = tables[0].Blocks()[0]->Location();
    cache.Cache(0, "ordinary", 2, 1, Kind::kChunk);
    const std::vector<std::string> hashes{"pending"};
    auto demands = cache.Demands(tables, 0, 4);
    const std::array materialized{4};
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .boundary_kind = GetParam(),
                .materialized_state_boundaries = materialized,
            },
        .num_computed_tokens = 6,
    };
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_FALSE(cache.Contains(0, "ordinary"));
    ASSERT_TRUE(cache.Contains(0, "pending"));
    const auto metadata = cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, future_endpoint);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->boundary_kind, GetParam());
    EXPECT_FALSE(tables[0].Blocks()[0]);
}

TEST_P(PendingStateEvictionPriorityTest, CanonicalUpgradeDoesNotProtectTheDiscardedSecondProducer) {
    PriorityCache cache(4, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    const auto duplicate_source = tables[0].Blocks()[0]->Location();
    const auto canonical = cache.Cache(0, "pending", 1, 0, Kind::kChunk);
    cache.Cache(0, "ordinary", 100, 1, Kind::kChunk);
    const std::vector<std::string> hashes{"pending"};
    auto demands = cache.Demands(tables, 0, 4);
    const std::array materialized{4};
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .boundary_kind = GetParam(),
                .materialized_state_boundaries = materialized,
            },
        .num_computed_tokens = 6,
    };
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_TRUE(cache.Contains(0, "ordinary"));
    ASSERT_TRUE(cache.Contains(0, "pending"));
    const auto metadata = cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, canonical);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->boundary_kind, GetParam());
    EXPECT_EQ(tables[0].Blocks().back()->Location(), duplicate_source);
}

TEST_P(PendingStateEvictionPriorityTest, ExistingReclaimableChunkUsesItsPendingUpgradeDuringPlanning) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    const auto future_endpoint = tables[0].Blocks()[0]->Location();
    const std::vector<std::string> hashes{"pending"};
    ASSERT_TRUE(
        CacheCoordinatorTestAccess::PublishStateSnapshot(cache.coordinator, tables, hashes, 4, 1, Kind::kChunk));
    cache.Cache(0, "ordinary", 1, 1, Kind::kChunk);
    auto demands = cache.Demands(tables, 0, 4);
    const std::array materialized{4};
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .boundary_kind = GetParam(),
                .materialized_state_boundaries = materialized,
            },
        .num_computed_tokens = 6,
    };
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_FALSE(cache.Contains(0, "ordinary"));
    ASSERT_TRUE(cache.Contains(0, "pending"));
    const auto metadata = cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, future_endpoint);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->boundary_kind, GetParam());
}

TEST_P(PendingStateEvictionPriorityTest, CanonicalReceivesRetainedPriorityBeforeItsProducerDeduplicates) {
    PriorityCache cache(4, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    const auto canonical = cache.Cache(0, "pending", 1, 0, Kind::kChunk);
    cache.Cache(0, "ordinary", 1, 1, Kind::kChunk);
    const std::vector<std::string> hashes{"pending"};
    // The duplicate source alone is insufficient: a second victim is needed.
    // The future canonical endpoint gets the retained tie-break in its epoch.
    auto demands = cache.Demands(tables, 0, 8);
    const std::array materialized{4};
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .boundary_kind = GetParam(),
                .materialized_state_boundaries = materialized,
            },
        .num_computed_tokens = 6,
    };
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_FALSE(cache.Contains(0, "ordinary"));
    ASSERT_TRUE(cache.Contains(0, "pending"));
    const auto metadata = cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, canonical);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->boundary_kind, GetParam());
}

INSTANTIATE_TEST_SUITE_P(RetainedKinds, PendingStateEvictionPriorityTest,
                         testing::Values(Kind::kEndpoint, Kind::kPromoted));

TEST(StateEvictionPriorityTest, LatestAndPendingPromotedKeepPromotionWithoutChangingHistoryKind) {
    PriorityCache cache(5, {GroupSpec(AttnKind::kFull, 1), GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(cache.Allocate(tables, 1, 8));
    // Publication progress is request-wide. This state-only fixture keeps the
    // History extent as holes without acquiring or protecting its cached entry.
    tables[0] = BlockTable::FromBlocks(std::vector<CacheBlockRef>(2), 0);
    const auto canonical = cache.Cache(1, "pending", 1, 0, Kind::kChunk);
    const auto history = cache.Cache(0, "history", 1, 0, Kind::kChunk);
    const auto history_before = cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, history);
    ASSERT_TRUE(history_before);
    cache.Cache(1, "ordinary", 1, 1, Kind::kChunk);
    const std::vector<std::string> hashes{"pending"};
    const auto latest = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 4);
    ASSERT_TRUE(latest);
    auto other_request = cache.Hold(*latest);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, other_request, *latest));
    auto demands = cache.Demands(tables, 1, 8);
    const std::array materialized{4};
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .boundary_kind = Kind::kChunk,
                .state_boundary_kind = Kind::kPromoted,
                .materialized_state_boundaries = materialized,
            },
        .num_computed_tokens = 6,
    };
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_FALSE(cache.Contains(1, "ordinary"));
    const auto state_after = cache.coordinator.GroupPrefixIndex(1).MetadataFor(cache.pool, canonical);
    ASSERT_TRUE(state_after);
    EXPECT_EQ(state_after->boundary_kind, Kind::kPromoted) << "table protection must not downgrade publication";
    EXPECT_EQ(state_after->generation, latest->blocks[0].generation);
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *latest));
    const auto history_after = cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, history);
    ASSERT_TRUE(history_after);
    EXPECT_EQ(history_after->boundary_kind, Kind::kChunk);
    EXPECT_EQ(history_after->generation, history_before->generation);
    EXPECT_EQ(history_after->last_access_epoch, history_before->last_access_epoch);
    EXPECT_EQ(history_after->was_acquired, history_before->was_acquired);
}

TEST(StateEvictionPriorityTest, MissingProvenanceCannotProtectAnUnwrittenPendingBoundary) {
    PriorityCache cache(3, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    cache.Cache(0, "ordinary", 100, 1, Kind::kChunk);
    const std::vector<std::string> hashes{"unwritten"};
    auto demands = cache.Demands(tables, 0, 4);
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .boundary_kind = Kind::kEndpoint,
                .materialized_state_boundaries = {},
            },
        .num_computed_tokens = 6,
    };
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_TRUE(cache.Contains(0, "ordinary"));
    EXPECT_FALSE(cache.Contains(0, "unwritten"));
}

TEST(StateEvictionPriorityTest, ReversePruningRestoresAChunkThatCannotReleaseItsPinnedParent) {
    PriorityCache cache(2, {GroupSpec(AttnKind::kFull, 1), GroupSpec(AttnKind::kMambaState, 2)});
    cache.Cache(0, "full", 2, 0, Kind::kChunk);
    const auto chunk = cache.Cache(1, "ordinary", 1, 0, Kind::kChunk);
    const auto endpoint = cache.Cache(1, "endpoint", 1, 1, Kind::kEndpoint);
    ASSERT_EQ(chunk.lcm_block_id, endpoint.lcm_block_id);
    const CacheBlockRef pin = cache.coordinator.GroupPrefixIndex(1).Find(cache.pool, Key(1, "endpoint"));
    ASSERT_TRUE(pin);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(cache.Allocate(tables, 0, 4));
    EXPECT_FALSE(cache.Contains(0, "full"));
    EXPECT_TRUE(cache.Contains(1, "ordinary"));
    EXPECT_TRUE(cache.Contains(1, "endpoint"));
    EXPECT_EQ(cache.pool.OccupiedCount(chunk.lcm_block_id), 2);
}

TEST(StateEvictionPriorityTest, LastSharedPinReclaimsAcquiredChunkWithoutReleasingItsPackedSibling) {
    PriorityCache cache(4, {GroupSpec(AttnKind::kMambaState, 2)});
    const auto chunk = cache.Cache(0, "chunk", 1, 0, Kind::kChunk);
    const auto endpoint = cache.Cache(0, "endpoint", 1, 1, Kind::kEndpoint);
    ASSERT_EQ(chunk.lcm_block_id, endpoint.lcm_block_id);
    auto& index = cache.coordinator.GroupPrefixIndex(0);
    const std::vector<CacheKey> keys{Key(0, "chunk")};
    auto acquired = index.AcquireMatched(cache.pool, keys, 0, GroupPrefixProbe{.hits = {1}}, 2);
    CacheBlockRef another_owner = index.Find(cache.pool, keys[0]);
    CacheBlockRef sibling_owner = index.Find(cache.pool, Key(0, "endpoint"));
    ASSERT_TRUE(index.MetadataFor(cache.pool, chunk)->was_acquired);
    ASSERT_EQ(index.MetadataFor(cache.pool, chunk)->boundary_kind, Kind::kChunk);
    const std::array first_release{&acquired.blocks[0]};
    cache.coordinator.ReleaseDeviceBlockRefs(first_release);
    EXPECT_TRUE(cache.Contains(0, "chunk"));
    const std::array last_release{&another_owner};
    cache.coordinator.ReleaseDeviceBlockRefs(last_release);
    EXPECT_FALSE(cache.Contains(0, "chunk"));
    EXPECT_TRUE(cache.Contains(0, "endpoint"));
    EXPECT_EQ(sibling_owner->Location(), endpoint);
    EXPECT_EQ(cache.pool.OccupiedCount(chunk.lcm_block_id), 1);
    EXPECT_EQ(cache.pool.BoundGroup(chunk.lcm_block_id), 0u);
    CacheBlockRef replacement = cache.pool.AcquireBlock(0);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement->Location(), chunk);
}

TEST(StateEvictionPriorityTest, ReclaimKeepsLatestUntilFreeClearsAllStateOwnership) {
    PriorityCache cache(12, {GroupSpec(AttnKind::kMambaState, 1), GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(2);
    std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{8}},
                                     {.table = &tables[1], .extent = DenseGrowth{8}}};
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));
    const std::vector<std::string> hashes{"old", "latest"};
    cache.coordinator.CacheFullBlocks(tables, hashes, 1, 0, Kind::kChunk);
    const auto latest = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 8);
    ASSERT_TRUE(latest);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *latest));

    cache.coordinator.ReclaimExpired(tables, 6);
    for (std::uint32_t group = 0; group < 2; ++group) {
        EXPECT_FALSE(cache.Contains(group, "old"));
        EXPECT_TRUE(cache.Contains(group, "latest"));
        EXPECT_TRUE(tables[group].Blocks()[1]);
    }
    cache.coordinator.ReclaimExpired(tables, 10);
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *latest));
    for (std::uint32_t group = 0; group < 2; ++group) {
        EXPECT_EQ(tables[group].ReclaimedPrefixBlocks(), 2);
        EXPECT_EQ(tables[group].ProtectedSlot(), 1);
        EXPECT_TRUE(tables[group].Blocks()[1]);
        EXPECT_EQ(cache.coordinator.GroupPrefixIndex(group).NumPinnedEntries(cache.pool), 1);
    }
    cache.coordinator.Free(tables);
    EXPECT_FALSE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *latest));
    EXPECT_EQ(tables[0].ProtectedSlot(), -1);
    EXPECT_EQ(tables[1].ProtectedSlot(), -1);
    EXPECT_FALSE(cache.Contains(0, "latest"));
    EXPECT_FALSE(cache.Contains(1, "latest"));
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), cache.pool.NumLcmBlocks());
}

TEST(StateEvictionPriorityTest, AnotherRequestsProtectedSlotPreventsCleanupUntilItsLastReferenceIsCleared) {
    PriorityCache cache(4, {GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "shared", 1, 0, Kind::kChunk);
    cache.Cache(0, "replacement", 1, 1, Kind::kChunk);
    const auto shared =
        CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, std::vector<std::string>{"shared"}, 4);
    const auto replacement = CacheCoordinatorTestAccess::CaptureStateSnapshot(
        cache.coordinator, std::vector<std::string>{"shared", "replacement"}, 8);
    ASSERT_TRUE(shared);
    ASSERT_TRUE(replacement);
    auto first = cache.Hold(*shared);
    auto second = cache.Hold(*shared);
    auto next = cache.Hold(*replacement);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, first, *shared));
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, second, *shared));
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, next, *replacement));
    cache.coordinator.ReclaimExpired(first, 6);
    cache.coordinator.ReclaimExpired(second, 6);
    cache.coordinator.ClearProtectedStateSnapshot(first);
    EXPECT_TRUE(cache.Contains(0, "shared"));
    cache.coordinator.ClearProtectedStateSnapshot(second);
    EXPECT_FALSE(cache.Contains(0, "shared"));
    ASSERT_TRUE(cache.Contains(0, "replacement"));
    EXPECT_EQ(cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, Key(0, "replacement"))->boundary_kind,
              Kind::kChunk);
}

TEST(StateEvictionPriorityTest, ClearingProtectionPreservesRetainedKindsAndRejectsStaleIdentity) {
    PriorityCache cache(5, {GroupSpec(AttnKind::kMambaState, 1)});
    const auto original = cache.Cache(0, "reused", 1, 0, Kind::kChunk);
    const std::vector<std::string> hashes{"reused"};
    const auto stale = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 4);
    ASSERT_TRUE(stale);
    ASSERT_TRUE(cache.coordinator.GroupPrefixIndex(0).Evict(cache.pool, original));
    cache.Cache(0, "reused", 1, 0, Kind::kChunk);
    const auto replacement = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 4);
    ASSERT_TRUE(replacement);
    ASSERT_NE(replacement->blocks[0].generation, stale->blocks[0].generation);
    auto tables = cache.Hold(*replacement);
    EXPECT_FALSE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *stale));
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *replacement));
    cache.coordinator.ReclaimExpired(tables, 6);
    EXPECT_TRUE(cache.Contains(0, "reused"));
    ASSERT_TRUE(cache.coordinator.RetainLatestStateSnapshot(hashes, Kind::kEndpoint));
    cache.coordinator.ClearProtectedStateSnapshot(tables);
    EXPECT_FALSE(tables[0].Blocks()[0]);
    EXPECT_TRUE(cache.Contains(0, "reused"));
    tables = cache.Hold(*replacement);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *replacement));
    ASSERT_TRUE(cache.coordinator.RetainLatestStateSnapshot(hashes, Kind::kPromoted));
    ASSERT_TRUE(cache.coordinator.RetainLatestStateSnapshot(hashes, Kind::kEndpoint));
    cache.coordinator.ReclaimExpired(tables, 6);
    cache.coordinator.ClearProtectedStateSnapshot(tables);
    EXPECT_EQ(cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, Key(0, "reused"))->boundary_kind,
              Kind::kPromoted);
}

TEST(StateEvictionPriorityTest, PartialSnapshotCleanupRemovesItsRemainingOrdinaryChunk) {
    PriorityCache cache(4, {GroupSpec(AttnKind::kMambaState, 1), GroupSpec(AttnKind::kMambaState, 1)});
    cache.Cache(0, "partial", 1, 0, Kind::kChunk);
    const auto missing = cache.Cache(1, "partial", 1, 0, Kind::kChunk);
    const auto snapshot =
        CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, std::vector<std::string>{"partial"}, 4);
    ASSERT_TRUE(snapshot);
    ASSERT_TRUE(cache.coordinator.GroupPrefixIndex(1).Evict(cache.pool, missing));
    auto tables = cache.Hold(*snapshot);
    EXPECT_FALSE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *snapshot));
    cache.coordinator.Free(tables);
    EXPECT_FALSE(cache.Contains(0, "partial"));
    EXPECT_FALSE(cache.Contains(1, "partial"));
}

TEST(StateEvictionPriorityTest, ProtectedSlotDoesNotDelayTheReclaimFrontierOrKeepOtherExpiredSlots) {
    PriorityCache cache(6, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 16));
    const std::vector<std::string> hashes{"latest", "expired1", "expired2", "next"};
    cache.coordinator.CacheFullBlocks(tables, hashes, 1, 0, Kind::kChunk);
    const auto latest = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 4);
    const auto next = CacheCoordinatorTestAccess::CaptureStateSnapshot(cache.coordinator, hashes, 16);
    ASSERT_TRUE(latest);
    ASSERT_TRUE(next);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *latest));

    cache.coordinator.ReclaimExpired(tables, 14);
    EXPECT_EQ(tables[0].ReclaimedPrefixBlocks(), 3);
    EXPECT_EQ(tables[0].ProtectedSlot(), 0);
    EXPECT_TRUE(tables[0].Blocks()[0]);
    EXPECT_FALSE(tables[0].Blocks()[1]);
    EXPECT_FALSE(tables[0].Blocks()[2]);
    EXPECT_TRUE(tables[0].Blocks()[3]);
    EXPECT_FALSE(cache.Contains(0, "expired1"));
    EXPECT_FALSE(cache.Contains(0, "expired2"));
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 4);

    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *next));
    EXPECT_EQ(tables[0].ReclaimedPrefixBlocks(), 3);
    EXPECT_EQ(tables[0].ProtectedSlot(), 3);
    EXPECT_FALSE(tables[0].Blocks()[0]) << "replacement must revisit the old slot behind the frontier";
    EXPECT_FALSE(cache.Contains(0, "latest"));
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *next));
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 5);

    cache.coordinator.ClearProtectedStateSnapshot(tables);
    EXPECT_TRUE(tables[0].Blocks()[3]) << "clearing protection does not end the ordinary working lifetime";
    cache.coordinator.ReclaimExpired(tables, 18);
    EXPECT_FALSE(tables[0].Blocks()[3]);
    EXPECT_FALSE(cache.Contains(0, "next"));
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 6);
}

TEST(StateEvictionPriorityTest, FailedReplacementLeavesEveryOldProtectedSlotIntact) {
    PriorityCache cache(8, {GroupSpec(AttnKind::kMambaState, 1), GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(2);
    const std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{12}},
                                           {.table = &tables[1], .extent = DenseGrowth{12}}};
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));
    const std::vector<std::string> hashes{"old", "next"};
    const auto old =
        CacheCoordinatorTestAccess::PublishStateSnapshot(cache.coordinator, tables, hashes, 4, 1, Kind::kChunk);
    const auto next =
        CacheCoordinatorTestAccess::PublishStateSnapshot(cache.coordinator, tables, hashes, 8, 1, Kind::kChunk);
    ASSERT_TRUE(old);
    ASSERT_TRUE(next);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *old));
    cache.coordinator.ReclaimExpired(tables, 6);
    CacheBlockRef missing = tables[1].EvictToNull(1);
    ASSERT_TRUE(missing);
    EXPECT_FALSE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *next));
    for (std::uint32_t group = 0; group < 2; ++group) {
        EXPECT_EQ(tables[group].ProtectedSlot(), 0);
        ASSERT_TRUE(tables[group].Blocks()[0]);
        EXPECT_EQ(tables[group].Blocks()[0].use_count(), 2u);
    }
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *old));
    cache.coordinator.ReclaimExpired(tables, 10);
    for (std::uint32_t group = 0; group < 2; ++group) {
        EXPECT_EQ(tables[group].ReclaimedPrefixBlocks(), 2);
        EXPECT_TRUE(tables[group].Blocks()[0]);
    }
    EXPECT_FALSE(cache.Contains(0, "next")) << "a failed update must not partially protect the first group";
    const std::array refs{&missing};
    cache.coordinator.ReleaseDeviceBlockRefs(refs);
    cache.coordinator.ClearProtectedStateSnapshot(tables);
    EXPECT_FALSE(cache.Contains(0, "old"));
    EXPECT_FALSE(cache.Contains(1, "old"));
}

TEST(StateEvictionPriorityTest, PublishAndProtectSwitchesEveryGroupBeforeCleaningExpiredChunks) {
    PriorityCache cache(8, {GroupSpec(AttnKind::kMambaState, 1), GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(2);
    const std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{12}},
                                           {.table = &tables[1], .extent = DenseGrowth{12}}};
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));
    const std::vector<std::string> hashes{"old", "next"};
    const auto old = cache.coordinator.PublishAndProtectStateSnapshot(tables, hashes, 4, 1, Kind::kChunk);
    ASSERT_TRUE(old);
    cache.coordinator.ReclaimExpired(tables, 6);
    for (const BlockTable& table : tables) {
        ASSERT_EQ(table.ProtectedSlot(), 0);
        ASSERT_EQ(table.ReclaimedPrefixBlocks(), 1);
        ASSERT_TRUE(table.Blocks()[0]);
        ASSERT_EQ(table.Blocks()[0].use_count(), 2u);
    }

    const auto redundant_source = tables[0].Blocks()[1]->Location();
    const auto second_source = tables[1].Blocks()[1]->Location();
    const auto canonical = cache.Cache(0, "next", 2, 1, Kind::kChunk);
    ASSERT_NE(canonical, redundant_source);
    const auto canonical_identity = cache.coordinator.GroupPrefixIndex(0).IdentityFor(cache.pool, canonical);
    ASSERT_TRUE(canonical_identity);
    ASSERT_EQ(cache.pool.NumEmptyLcmBlocks(), 1);
    std::int32_t old_removals = 0;
    cache.coordinator.SetCacheMutationSink([&](const CacheKey& key, CacheCoordinator::CacheMutation mutation) {
        if (key.content_hash != "old" || mutation != CacheCoordinator::CacheMutation::kRemoved) {
            return;
        }
        ++old_removals;
        for (const BlockTable& table : tables) {
            EXPECT_EQ(table.ProtectedSlot(), 1) << "all groups must switch before any old Chunk is cleaned";
            EXPECT_FALSE(table.Blocks()[0]);
            ASSERT_TRUE(table.Blocks()[1]);
            EXPECT_EQ(table.Blocks()[1].use_count(), 2u);
        }
    });

    const auto next = cache.coordinator.PublishAndProtectStateSnapshot(tables, hashes, 8, 3, Kind::kChunk);
    ASSERT_TRUE(next);
    ASSERT_EQ(next->blocks.size(), 2u);
    EXPECT_EQ(next->boundary_tokens, 8);
    EXPECT_EQ(next->blocks[0].generation, canonical_identity->generation);
    EXPECT_EQ(tables[0].Blocks()[1]->Location(), canonical);
    EXPECT_EQ(tables[1].Blocks()[1]->Location(), second_source);
    EXPECT_FALSE(cache.pool.IsOccupied(redundant_source));
    EXPECT_EQ(old_removals, 2);
    EXPECT_FALSE(cache.Contains(0, "old"));
    EXPECT_FALSE(cache.Contains(1, "old"));
    EXPECT_FALSE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *old));
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *next));
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 4) << "two old Chunks and the redundant producer return to the pool";

    const auto repeated = cache.coordinator.PublishAndProtectStateSnapshot(tables, hashes, 8, 4, Kind::kChunk);
    ASSERT_TRUE(repeated);
    ASSERT_EQ(repeated->blocks.size(), 2u);
    for (std::uint32_t group = 0; group < 2; ++group) {
        EXPECT_EQ(repeated->blocks[group].generation, next->blocks[group].generation);
        EXPECT_EQ(tables[group].ProtectedSlot(), 1);
        EXPECT_EQ(tables[group].ReclaimedPrefixBlocks(), 1);
        EXPECT_EQ(tables[group].Blocks()[1].use_count(), 2u) << "returned identities must not add a pin";
    }
    EXPECT_EQ(old_removals, 2);
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 4);
}

TEST(StateEvictionPriorityTest, PublishAndProtectMissingGroupPreservesOldProtectionAndUnpublishedTables) {
    PriorityCache cache(12, {GroupSpec(AttnKind::kMambaState, 1), GroupSpec(AttnKind::kMambaState, 1),
                             GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(3);
    const std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{12}},
                                           {.table = &tables[1], .extent = DenseGrowth{12}},
                                           {.table = &tables[2], .extent = DenseGrowth{12}}};
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));
    const std::vector<std::string> hashes{"old", "next"};
    const auto old = cache.coordinator.PublishAndProtectStateSnapshot(tables, hashes, 4, 1, Kind::kChunk);
    ASSERT_TRUE(old);
    ASSERT_EQ(old->blocks.size(), 3u);
    cache.coordinator.ReclaimExpired(tables, 6);
    const auto canonical = cache.Cache(0, "next", 2, 1, Kind::kChunk);
    const auto canonical_before = cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, canonical);
    ASSERT_TRUE(canonical_before);
    CacheBlockRef missing = tables[2].EvictToNull(1);
    ASSERT_TRUE(missing);
    ASSERT_EQ(cache.pool.NumEmptyLcmBlocks(), 2);

    std::array<std::array<CacheBlockLocation, 3>, 3> locations{};
    std::array<std::array<std::uint32_t, 3>, 3> use_counts{};
    for (std::size_t group = 0; group < tables.size(); ++group) {
        for (std::size_t slot = 0; slot < tables[group].Blocks().size(); ++slot) {
            const CacheBlockRef& ref = tables[group].Blocks()[slot];
            locations[group][slot] = ref ? ref->Location() : CacheBlockLocation{};
            use_counts[group][slot] = ref.use_count();
        }
    }

    EXPECT_FALSE(cache.coordinator.PublishAndProtectStateSnapshot(tables, hashes, 8, 9, Kind::kPromoted));
    for (std::size_t group = 0; group < tables.size(); ++group) {
        EXPECT_EQ(tables[group].ProtectedSlot(), 0);
        EXPECT_EQ(tables[group].ReclaimedPrefixBlocks(), 1);
        ASSERT_EQ(tables[group].NumBlocks(), 3);
        for (std::size_t slot = 0; slot < tables[group].Blocks().size(); ++slot) {
            const CacheBlockRef& ref = tables[group].Blocks()[slot];
            EXPECT_EQ(ref ? ref->Location() : CacheBlockLocation{}, locations[group][slot]);
            EXPECT_EQ(ref.use_count(), use_counts[group][slot]);
        }
        const auto old_metadata = cache.coordinator.GroupPrefixIndex(group).MetadataFor(cache.pool, Key(group, "old"));
        ASSERT_TRUE(old_metadata);
        EXPECT_EQ(old_metadata->generation, old->blocks[group].generation);
        EXPECT_EQ(old_metadata->boundary_kind, Kind::kChunk);
        EXPECT_EQ(old_metadata->last_access_epoch, 1u);
    }
    const auto canonical_after = cache.coordinator.GroupPrefixIndex(0).MetadataFor(cache.pool, canonical);
    ASSERT_TRUE(canonical_after);
    EXPECT_EQ(canonical_after->generation, canonical_before->generation);
    EXPECT_EQ(canonical_after->boundary_kind, canonical_before->boundary_kind);
    EXPECT_EQ(canonical_after->last_access_epoch, canonical_before->last_access_epoch);
    EXPECT_EQ(canonical_after->was_acquired, canonical_before->was_acquired);
    EXPECT_FALSE(cache.Contains(1, "next")) << "validate the last group before publishing any preceding group";
    EXPECT_FALSE(cache.Contains(2, "next"));
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *old));
    EXPECT_EQ(missing.use_count(), 1u);
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 2);
}

TEST(StateEvictionPriorityTest, OnlyLatestCapacityCannotFundAnAdmissionAndFailureIsAtomic) {
    PriorityCache cache(2, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 8));
    const auto latest = CacheCoordinatorTestAccess::PublishStateSnapshot(
        cache.coordinator, tables, std::vector<std::string>{"latest"}, 4, 1, Kind::kChunk);
    ASSERT_TRUE(latest);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *latest));
    const auto first = tables[0].Blocks()[0]->Location();
    const auto second = tables[0].Blocks()[1]->Location();
    auto demands = cache.Demands(tables, 0, 4);
    const RequestProgress progress{.num_computed_tokens = 6};
    EXPECT_FALSE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[0].ReclaimedPrefixBlocks(), 0);
    EXPECT_EQ(tables[0].ProtectedSlot(), 0);
    EXPECT_EQ(tables[0].Blocks()[0]->Location(), first);
    EXPECT_EQ(tables[0].Blocks()[1]->Location(), second);
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *latest));
    EXPECT_EQ(cache.pool.NumEmptyLcmBlocks(), 0);

    cache.coordinator.ReclaimExpired(tables, 6);
    EXPECT_FALSE(cache.coordinator.GroupHasReclaimableBlocksAt(0, tables[0], 6));
    EXPECT_EQ(cache.coordinator.GroupBlocksReclaimableAt(0, tables[0], 6, true), 0);
    cache.coordinator.ClearProtectedStateSnapshot(tables);
    ASSERT_TRUE(cache.coordinator.Admit(cache.coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_FALSE(cache.Contains(0, "latest"));
    EXPECT_FALSE(tables[0].Blocks()[0]);
    EXPECT_EQ(tables[0].Blocks().back()->Location(), first);
}

TEST(StateEvictionPriorityTest, ReplacementKeepsThePreviousSnapshotUntilItsWorkingConsumerExpires) {
    PriorityCache cache(4, {GroupSpec(AttnKind::kMambaState, 1)});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(cache.Allocate(tables, 0, 12));
    const std::vector<std::string> hashes{"old", "next"};
    const auto old =
        CacheCoordinatorTestAccess::PublishStateSnapshot(cache.coordinator, tables, hashes, 4, 1, Kind::kChunk);
    const auto next =
        CacheCoordinatorTestAccess::PublishStateSnapshot(cache.coordinator, tables, hashes, 8, 1, Kind::kChunk);
    ASSERT_TRUE(old);
    ASSERT_TRUE(next);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *old));
    cache.coordinator.ReclaimExpired(tables, 4);
    ASSERT_TRUE(CacheCoordinatorTestAccess::ProtectStateSnapshot(cache.coordinator, tables, *next));
    EXPECT_EQ(tables[0].ProtectedSlot(), 1);
    EXPECT_TRUE(tables[0].Blocks()[0]);
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *old));
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *next));

    cache.coordinator.ReclaimExpired(tables, 6);
    EXPECT_FALSE(tables[0].Blocks()[0]);
    EXPECT_FALSE(cache.Contains(0, "old"));
    EXPECT_TRUE(tables[0].Blocks()[1]);
    EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(cache.coordinator, *next));
}

}  // namespace
}  // namespace tokenspeed::test
