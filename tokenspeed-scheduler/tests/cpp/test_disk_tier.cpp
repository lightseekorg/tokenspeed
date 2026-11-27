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

// The Disk tier below Host (docs/rfcs/0006-disk-tier.md) on the coordinator:
// cleaning writes dirty Host entries to Disk, a prefetch reads a prompt's
// Disk-only prefix back into Host, and Disk entries count in the boundary
// tree like the other tiers'.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/coordinator/gds_tiered.h"
#include "cache/core/block_pool.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache/tier/disk_manager.h"
#include "cache/tier/peer_exports.h"
#include "cache_event_test_sink.h"
#include "cache_test_access.h"
#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kGrain = 4;

DiskConfig Limits() {
    return DiskConfig{.max_operations = 4,
                      .max_blocks_per_operation = 64,
                      .max_host_blocks = 32,
                      .max_write_host_blocks = 32,
                      .max_queued_cleanings = 64,
                      .min_prefetch_tokens = 1,
                      .wait_timeout_us = 1000,
                      .ready_timeout_us = 1000};
}

std::vector<std::int32_t> Packing(std::span<const CacheGroupSpec> specs) {
    std::vector<std::int32_t> packing;
    for (const auto& spec : specs) packing.push_back(spec.cache_blocks_per_lcm_block);
    return packing;
}

std::vector<CacheGroupSpec> FullOnly() {
    return {{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kGrain}};
}

std::vector<CacheGroupSpec> FullAndState() {
    return {{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kGrain},
            {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
}

// A prompt of `pages` full pages, distinct per `seed`.
struct Prompt {
    Prompt(std::int32_t pages, std::int32_t seed) : tokens(static_cast<std::size_t>(pages * kGrain)) {
        std::iota(tokens.begin(), tokens.end(), seed * 1000 + 1);
        std::vector<std::span<const std::int32_t>> spans;
        for (std::int32_t page = 0; page < pages; ++page) {
            spans.emplace_back(tokens.data() + page * kGrain, kGrain);
        }
        hashes = ComputePrefixHashes(spans, "");
    }

    PromptPrefix Prefix() const {
        return PromptPrefix{
            .prefix_hashes = hashes, .tokens = tokens, .max_tokens = static_cast<std::int32_t>(tokens.size())};
    }

    std::vector<std::int32_t> tokens;
    std::vector<std::string> hashes;
};

class DiskTierCache {
public:
    DiskTierCache(std::vector<CacheGroupSpec> specs, std::int32_t disk_parents, DiskConfig limits = Limits(),
                  std::int32_t host_parents = 64)
        : device{64, Packing(specs)},
          host{host_parents, Packing(specs)},
          disk{disk_parents, Packing(specs)},
          coordinator{MakeCoordinator(specs, kGrain, device, /*enable_l3_storage=*/false, &host, &disk,
                                      /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                      /*write_back_evictions=*/true, /*checkpoint_retention_requests=*/0)},
          disks{coordinator, limits} {}

    std::vector<BlockTable> OnDevice(const Prompt& prompt) {
        std::vector<BlockTable> tables(coordinator.NumGroups());
        EXPECT_TRUE(AdmitForTest(coordinator, tables, static_cast<std::int32_t>(prompt.tokens.size())));
        coordinator.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.tokens = prompt.tokens,
                                                              .prefix_hashes = prompt.hashes,
                                                              .publish_state = true,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint,
                                                              .stream_completed_to_host = false},
                            .num_computed_tokens = static_cast<std::int32_t>(prompt.tokens.size())},
            CacheCoordinatorTestAccess::NextAccessEpoch(coordinator));
        return tables;
    }

    // Publishes the prompt on the Device -- a checkpoint at its end with State
    // groups -- copies every entry to a fresh Host block, as a write-back would,
    // and empties the Device: the prompt then lives in dirty Host entries.
    std::vector<CacheKey> OnHostOnly(const Prompt& prompt) {
        auto tables = OnDevice(prompt);
        coordinator.Free(tables);
        std::vector<CacheKey> keys = DeviceKeys(prompt);
        for (const CacheKey& key : keys) {
            CacheBlockRef block = coordinator.Host().AcquireHostBlock(key.group_id, 0);
            EXPECT_TRUE(block);
            coordinator.Host().CacheHostBlock(block, key, std::nullopt);
        }
        EXPECT_TRUE(coordinator.ClearDeviceCache());
        return keys;
    }

    // A write-back's Host placement of `rows` Full blocks as one segment.
    HostCache::HostAllocationBatch PlaceSegment(std::size_t rows) {
        const std::vector<std::uint32_t> groups(rows, 0);
        const std::vector<std::int32_t> buckets(rows, 0);
        const std::array segments{HostCache::HostSegmentRows{.first_row = 0, .past_end_row = rows}};
        return coordinator.Host().AcquireHostSegments(groups, buckets, segments);
    }

    // Writes keys to the Disk: their Host entries become clean.
    void CleanAll(std::span<const CacheKey> keys) {
        const DiskOperation write = Clean(keys);
        disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    }

    bool OnHost(const CacheKey& key) const { return coordinator.Host().ContainsHostCachedBlock(key); }

    // The prompt's keys the Device indexes, Full pages first.
    std::vector<CacheKey> DeviceKeys(const Prompt& prompt) const {
        std::vector<CacheKey> keys;
        for (std::uint32_t group = 0; group < static_cast<std::uint32_t>(coordinator.NumGroups()); ++group) {
            for (CacheKey& key : coordinator.Store().KeysForGroup(prompt.hashes, group)) {
                if (coordinator.Store().LocationOf(device, group, key)) keys.push_back(std::move(key));
            }
        }
        return keys;
    }

    void EvictFromHost(std::span<const CacheKey> keys) {
        for (const CacheKey& key : keys) {
            const auto location = coordinator.Store().LocationOf(host, key.group_id, key);
            ASSERT_TRUE(location);
            ASSERT_TRUE(coordinator.Evictor().EvictHostCachedBlock(key.group_id, *location,
                                                                   &CacheEvictor::HostEvictionStats::lru));
        }
    }

    // Starts one cleaning write and returns it.
    DiskOperation Clean(std::span<const CacheKey> keys) {
        disks.QueueCleaning(keys);
        disks.StartCleaning();
        const std::vector<DiskOperation> commands = disks.TakeCommands();
        EXPECT_EQ(commands.size(), 1u);
        return commands.empty() ? DiskOperation{} : commands.front();
    }

    std::int32_t HostHitTokens(const Prompt& prompt) const {
        const PrefixProbe probe = coordinator.Prober().ProbePromptPrefix(prompt.Prefix());
        return std::max(probe.device.num_common_tokens, probe.host.num_common_tokens);
    }

    BlockPool device;
    BlockPool host;
    BlockPool disk;
    CacheCoordinator coordinator;
    DiskTransferManager disks;
};

TEST(DiskTierTest, CleaningWritesDirtyHostEntriesAndPublishesThemOnDisk) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(4, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    ASSERT_EQ(keys.size(), 4u);

    const DiskOperation write = cache.Clean(keys);
    EXPECT_EQ(write.kind, DiskOpKind::kWrite);
    ASSERT_EQ(write.group_ids.size(), 4u);
    EXPECT_TRUE(std::ranges::all_of(write.host_blocks, [](std::int32_t block) { return block > 0; }));
    EXPECT_TRUE(std::ranges::all_of(write.disk_blocks, [](std::int32_t block) { return block > 0; }));
    // Nothing is on the Disk before every owner reports the write.
    EXPECT_FALSE(std::ranges::any_of(
        keys, [&](const CacheKey& key) { return cache.coordinator.Disk().ContainsDiskCachedBlock(key); }));
    EXPECT_EQ(cache.coordinator.Host().NumPinnedHostCachedBlocks(), 4) << "the write holds its Host sources";

    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    EXPECT_TRUE(std::ranges::all_of(keys, [&](const CacheKey& key) {
        return cache.coordinator.Disk().ContainsDiskCachedBlock(key) &&
               cache.coordinator.Host().ContainsHostCachedBlock(key);
    })) << "the Host entries stay, now clean";
    EXPECT_EQ(cache.coordinator.Host().NumPinnedHostCachedBlocks(), 0);
    EXPECT_FALSE(cache.disks.HasPending());
    EXPECT_EQ(cache.disks.WrittenBlocks(), 4u);
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(cache.coordinator).empty());

    // A clean entry is not written again.
    cache.disks.QueueCleaning(keys);
    cache.disks.StartCleaning();
    EXPECT_TRUE(cache.disks.TakeCommands().empty());
}

TEST(DiskTierTest, CleaningAPageWritesTheDirtyPagesAboveItFirst) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(4, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    const DiskOperation write = cache.Clean(std::span{keys}.last(1));
    ASSERT_EQ(write.group_ids.size(), 4u);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    for (const CacheKey& key : keys) {
        EXPECT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(key));
    }
}

TEST(DiskTierTest, AFailedWritePublishesNothing) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(2, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    const DiskOperation write = cache.Clean(keys);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = false});
    EXPECT_EQ(cache.coordinator.Disk().NumDiskCachedBlocks(), 0);
    EXPECT_EQ(cache.disk.NumOccupiedSlots(), 0) << "the unpublished Disk blocks go back";
    EXPECT_EQ(cache.disks.FailedWrites(), 1u);
}

TEST(DiskTierTest, APrefetchReadsACleanedPrefixTheHostDroppedBackIntoHost) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(4, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    const DiskOperation write = cache.Clean(keys);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    cache.EvictFromHost(keys);
    ASSERT_EQ(cache.HostHitTokens(prompt), 0);

    StorageLease read = cache.disks.StartRead(prompt.Prefix());
    ASSERT_TRUE(read);
    const std::vector<DiskOperation> commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(commands[0].kind, DiskOpKind::kRead);
    EXPECT_EQ(commands[0].disk_blocks, write.disk_blocks) << "the read's sources are the cleaned blocks";
    EXPECT_EQ(cache.HostHitTokens(prompt), 0) << "nothing is published before the read completes";
    EXPECT_FALSE(cache.disks.Completed(read.OperationId()));

    cache.disks.Complete(cache::DiskDone{.op_id = read.OperationId(), .success = true});
    EXPECT_TRUE(cache.disks.Completed(read.OperationId()));
    EXPECT_EQ(cache.HostHitTokens(prompt), 16);
    EXPECT_EQ(cache.coordinator.Host().NumPinnedHostCachedBlocks(), 4) << "the lease holds them for its request";
    read.reset();
    EXPECT_EQ(cache.coordinator.Host().NumPinnedHostCachedBlocks(), 0);
    EXPECT_FALSE(cache.disks.HasPending());
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(cache.coordinator).empty());
}

TEST(DiskTierTest, APrefetchReadsOnlyWhatTheHostLacks) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(4, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    const DiskOperation write = cache.Clean(keys);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    cache.EvictFromHost(std::span{keys}.last(2));
    ASSERT_EQ(cache.HostHitTokens(prompt), 8);

    StorageLease read = cache.disks.StartRead(prompt.Prefix());
    ASSERT_TRUE(read);
    const std::vector<DiskOperation> commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(commands[0].group_ids.size(), 2u);
    cache.disks.Complete(cache::DiskDone{.op_id = read.OperationId(), .success = true});
    EXPECT_EQ(cache.HostHitTokens(prompt), 16);
}

TEST(DiskTierTest, AFailedReadDropsTheDiskEntriesItRead) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(2, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    const DiskOperation write = cache.Clean(keys);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    cache.EvictFromHost(keys);

    StorageLease read = cache.disks.StartRead(prompt.Prefix());
    ASSERT_TRUE(read);
    cache.disks.TakeCommands();
    cache.disks.Complete(cache::DiskDone{.op_id = read.OperationId(), .success = false});
    EXPECT_EQ(cache.HostHitTokens(prompt), 0);
    EXPECT_EQ(cache.coordinator.Disk().NumDiskCachedBlocks(), 0);
    EXPECT_EQ(cache.coordinator.Evictor().DiskEvictions().read_failure.blocks, 2u);
    EXPECT_EQ(cache.coordinator.Host().NumPinnedHostCachedBlocks(), 0) << "a failed read holds nothing";
}

TEST(DiskTierTest, ARequestThatGivesUpBeforeTheReadIsIssuedWithdrawsIt) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(2, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    const DiskOperation write = cache.Clean(keys);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    cache.EvictFromHost(keys);

    StorageLease read = cache.disks.StartRead(prompt.Prefix());
    ASSERT_TRUE(read);
    read.reset();
    EXPECT_TRUE(cache.disks.TakeCommands().empty());
    EXPECT_FALSE(cache.disks.HasPending());
    EXPECT_EQ(cache.coordinator.Disk().NumPinnedDiskCachedBlocks(), 0);
}

TEST(DiskTierTest, ARequestThatGivesUpWhileTheReadRunsKeepsItsBlocksUntilItRetires) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(2, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    const DiskOperation write = cache.Clean(keys);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    cache.EvictFromHost(keys);

    StorageLease read = cache.disks.StartRead(prompt.Prefix());
    ASSERT_TRUE(read);
    const std::uint64_t id = read.OperationId();
    ASSERT_EQ(cache.disks.TakeCommands().size(), 1u);
    read.reset();
    // The device may still write the destinations.
    EXPECT_TRUE(cache.disks.HasPending());
    EXPECT_EQ(cache.disks.NumHeldHostBlocks(), 2u);
    cache.disks.Complete(cache::DiskDone{.op_id = id, .success = true});
    EXPECT_FALSE(cache.disks.HasPending());
    EXPECT_EQ(cache.HostHitTokens(prompt), 0) << "a read its request gave up publishes nothing";
}

TEST(DiskTierTest, ADiskPlacementEvictsTheLeastRecentlyUsedEntries) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/4);
    const Prompt first(4, 1);
    const Prompt second(4, 2);
    const std::vector<CacheKey> first_keys = cache.OnHostOnly(first);
    const DiskOperation first_write = cache.Clean(first_keys);
    cache.disks.Complete(cache::DiskDone{.op_id = first_write.op_id, .success = true});
    const std::vector<CacheKey> second_keys = cache.OnHostOnly(second);
    const DiskOperation second_write = cache.Clean(second_keys);
    ASSERT_EQ(second_write.group_ids.size(), 4u);
    cache.disks.Complete(cache::DiskDone{.op_id = second_write.op_id, .success = true});
    EXPECT_EQ(cache.coordinator.Evictor().DiskEvictions().lru.blocks, 4u);
    EXPECT_EQ(cache.coordinator.Evictor().DiskEvictions().lru.host_resident, 4u)
        << "the first prompt's Host entries are dirty again";
    for (const CacheKey& key : second_keys) {
        EXPECT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(key));
    }
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(cache.coordinator).empty());
}

TEST(DiskTierTest, AnEntryAReadHoldsIsNotEvictedForCleaning) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/2);
    const Prompt first(2, 1);
    const Prompt second(2, 2);
    const std::vector<CacheKey> first_keys = cache.OnHostOnly(first);
    const DiskOperation first_write = cache.Clean(first_keys);
    cache.disks.Complete(cache::DiskDone{.op_id = first_write.op_id, .success = true});
    cache.EvictFromHost(first_keys);
    StorageLease read = cache.disks.StartRead(first.Prefix());
    ASSERT_TRUE(read);
    ASSERT_EQ(cache.disks.TakeCommands().size(), 1u);

    const std::vector<CacheKey> second_keys = cache.OnHostOnly(second);
    cache.disks.QueueCleaning(second_keys);
    cache.disks.StartCleaning();
    EXPECT_TRUE(cache.disks.TakeCommands().empty()) << "no Disk block is free of the read";
    EXPECT_EQ(cache.disks.DroppedCleanings(), 2u);
    cache.disks.Complete(cache::DiskDone{.op_id = read.OperationId(), .success = true});
    EXPECT_EQ(cache.HostHitTokens(first), 8);
}

TEST(DiskTierTest, APageOnTheDiskKeepsTheHostPagesUnderItResumable) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(4, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    const DiskOperation write = cache.Clean(std::span{keys}.first(1));
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    // The Host drops the clean root page: on the Disk alone, it is no gap.
    cache.EvictFromHost(std::span{keys}.first(1));
    cache.coordinator.CollectDeadEntries();
    for (const CacheKey& key : std::span{keys}.subspan(1)) {
        EXPECT_TRUE(cache.coordinator.Host().ContainsHostCachedBlock(key));
    }
    // A prefetch brings the root page back; the prompt then resumes in full.
    StorageLease read = cache.disks.StartRead(prompt.Prefix());
    ASSERT_TRUE(read);
    const std::vector<DiskOperation> commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(commands[0].group_ids.size(), 1u);
    cache.disks.Complete(cache::DiskDone{.op_id = read.OperationId(), .success = true});
    EXPECT_EQ(cache.HostHitTokens(prompt), 16);
}

TEST(DiskTierTest, APageNoTierHoldsStillLeavesTheHostPagesUnderItDead) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(3, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    cache.EvictFromHost(std::span{keys}.first(1));
    cache.coordinator.CollectDeadEntries();
    EXPECT_EQ(cache.coordinator.Host().NumHostCachedBlocks(), 0);
}

TEST(DiskTierTest, AStateCheckpointIsCleanedWithItsHistoryAndPrefetchedWithIt) {
    DiskTierCache cache(FullAndState(), /*disk_parents=*/16);
    const Prompt prompt(3, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    ASSERT_EQ(keys.size(), 4u) << "three pages and the checkpoint at the prompt end";
    ASSERT_EQ(cache.HostHitTokens(prompt), 12);

    // Cleaning the checkpoint takes the pages under it along.
    const CacheKey checkpoint = keys.back();
    ASSERT_EQ(checkpoint.group_id, 1u);
    const DiskOperation write = cache.Clean(std::span{&checkpoint, 1});
    EXPECT_EQ(write.group_ids.size(), 4u);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    cache.EvictFromHost(keys);
    ASSERT_EQ(cache.HostHitTokens(prompt), 0);

    StorageLease read = cache.disks.StartRead(prompt.Prefix());
    ASSERT_TRUE(read);
    const std::vector<DiskOperation> commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(commands[0].group_ids.size(), 4u);
    cache.disks.Complete(cache::DiskDone{.op_id = read.OperationId(), .success = true});
    EXPECT_EQ(cache.HostHitTokens(prompt), 12) << "the checkpoint resumes the prompt end";
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(cache.coordinator).empty());
}

// A Host placement drops clean entries -- the Disk holds them too -- before
// dirty ones, whatever their age (docs/rfcs/0006-disk-tier.md, Invariant 3).
TEST(DiskTierTest, AWriteBackPlacementDropsANewerCleanPageBeforeAnOlderDirtyOne) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16, Limits(), /*host_parents=*/4);
    const std::vector<CacheKey> dirty = cache.OnHostOnly(Prompt(2, 1));
    const std::vector<CacheKey> clean = cache.OnHostOnly(Prompt(2, 2));
    cache.CleanAll(clean);
    ASSERT_EQ(cache.coordinator.Host().NumFreeLcmBlocks(), 0);

    const HostCache::HostAllocationBatch batch = cache.PlaceSegment(1);
    EXPECT_EQ(batch.stats.allocated, 1u);
    EXPECT_FALSE(cache.OnHost(clean[1])) << "the clean leaf leaves, though it is the newest";
    EXPECT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(clean[1]));
    EXPECT_TRUE(cache.OnHost(clean[0]));
    EXPECT_TRUE(std::ranges::all_of(dirty, [&](const CacheKey& key) { return cache.OnHost(key); }))
        << "no dirty entry is lost while a clean one can make room";
    EXPECT_EQ(cache.coordinator.Evictor().HostEvictions().clean.blocks, 1u);
    EXPECT_EQ(cache.coordinator.Evictor().HostEvictions().lru.blocks, 0u);
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(cache.coordinator).empty());
}

TEST(DiskTierTest, AWriteBackPlacementDropsACleanCheckpointWithItsHistoryBeforeADirtyOne) {
    DiskTierCache cache(FullAndState(), /*disk_parents=*/16, Limits(), /*host_parents=*/8);
    const std::vector<CacheKey> dirty = cache.OnHostOnly(Prompt(3, 1));
    const std::vector<CacheKey> clean = cache.OnHostOnly(Prompt(3, 2));
    ASSERT_EQ(clean.size(), 4u) << "three pages and the checkpoint at the prompt end";
    cache.CleanAll(clean);
    ASSERT_EQ(cache.coordinator.Host().NumFreeLcmBlocks(), 0);

    EXPECT_EQ(cache.PlaceSegment(1).stats.allocated, 1u);
    EXPECT_TRUE(std::ranges::none_of(clean, [&](const CacheKey& key) { return cache.OnHost(key); }));
    EXPECT_TRUE(std::ranges::all_of(dirty, [&](const CacheKey& key) { return cache.OnHost(key); }));
    EXPECT_EQ(cache.coordinator.Evictor().HostEvictions().clean.blocks, 4u);
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(cache.coordinator).empty());
}

// A checkpoint whose State the Disk holds still ranks as dirty when the
// victim it becomes would take a Host page the Disk lacks.
TEST(DiskTierTest, ACleanCheckpointAboveADirtyPageRanksAsDirty) {
    DiskTierCache cache(FullAndState(), /*disk_parents=*/16, Limits(), /*host_parents=*/8);
    const std::vector<CacheKey> older = cache.OnHostOnly(Prompt(3, 1));
    const std::vector<CacheKey> newer = cache.OnHostOnly(Prompt(3, 2));
    cache.CleanAll(newer);
    // The newer prompt's root page is dirty again; its checkpoint is not.
    ASSERT_TRUE(cache.coordinator.Disk().DropUnreadableEntry(newer[0]));
    ASSERT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(newer.back()));

    EXPECT_EQ(cache.PlaceSegment(1).stats.allocated, 1u);
    EXPECT_TRUE(std::ranges::all_of(newer, [&](const CacheKey& key) { return cache.OnHost(key); }))
        << "taking the newer checkpoint would take its dirty root page along";
    EXPECT_TRUE(std::ranges::none_of(older, [&](const CacheKey& key) { return cache.OnHost(key); }))
        << "the older checkpoint leaves first, by recency";
    EXPECT_EQ(cache.coordinator.Evictor().HostEvictions().clean.blocks, 0u);
    EXPECT_EQ(cache.coordinator.Evictor().HostEvictions().lru.blocks, 4u);
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(cache.coordinator).empty());
}

// A holder lists, for a peer, each page from the fastest tier holding it;
// its Disk pages are pushed (docs/rfcs/0006-disk-tier.md, Peer sharing).
TEST(DiskTierTest, APeerListingNamesEachPageFromTheFastestTierHoldingIt) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(4, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    const DiskOperation write = cache.Clean(keys);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    cache.EvictFromHost(std::span{keys}.last(2));

    const cache::PeerLookup lookup{.lease = {.reader = "reader", .id = 1},
                                   .prefix_hashes = prompt.hashes,
                                   .tokens = prompt.tokens,
                                   .max_tokens = static_cast<std::int32_t>(prompt.tokens.size())};
    const PeerAnswer answer = PeerExports::AnswerLookup(cache.coordinator, lookup, /*include_disk=*/true);
    EXPECT_EQ(answer.num_tokens, 16) << "the Disk extends the Host's two pages";
    ASSERT_EQ(answer.pages.size(), 4u);
    for (std::int32_t slot = 0; slot < 4; ++slot) {
        EXPECT_EQ(answer.pages[static_cast<std::size_t>(slot)].slot, slot);
        EXPECT_EQ(answer.pages[static_cast<std::size_t>(slot)].tier, slot < 2 ? CacheTier::kHost : CacheTier::kDisk);
    }
    EXPECT_EQ(answer.pages[2].block_id, write.disk_blocks[2]) << "a Disk page names its Disk block";
    EXPECT_TRUE(PeerExports::Matches(cache.coordinator, lookup, answer.num_tokens, answer.pages));

    // The hard pin holds the Disk blocks too, until the release.
    PeerExports exports{PeerConfig{.name = "holder",
                                   .incarnation = 1,
                                   .max_leases = 4,
                                   .soft_pin_us = 100,
                                   .hard_pin_us = 1000,
                                   .lookup_timeout_us = 100,
                                   .min_fetch_tokens = 1,
                                   .landing_tier = PeerLandingTier::kDevice,
                                   .max_host_landing_blocks = 0,
                                   .host_ready_timeout_us = 0}};
    exports.Lookup(cache.coordinator, lookup, /*now_us=*/0);
    exports.Pin(cache.coordinator,
                cache::PeerPin{.lease = lookup.lease, .num_tokens = answer.num_tokens, .pages = answer.pages},
                /*now_us=*/1, nullptr);
    EXPECT_TRUE(exports.Held(lookup.lease));
    EXPECT_EQ(cache.coordinator.Disk().NumPinnedDiskCachedBlocks(), 2);
    exports.Release(cache.coordinator, cache::PeerRelease{.lease = lookup.lease, .demote = false}, /*now_us=*/2);
    EXPECT_EQ(cache.coordinator.Disk().NumPinnedDiskCachedBlocks(), 0);
}

TEST(DiskTierTest, APeerListingWithoutADiskGainIsTheDeviceAndHostOne) {
    DiskTierCache cache(FullAndState(), /*disk_parents=*/16);
    const Prompt prompt(3, 1);
    cache.OnHostOnly(prompt);
    const cache::PeerLookup lookup{.lease = {.reader = "reader", .id = 1},
                                   .prefix_hashes = prompt.hashes,
                                   .tokens = prompt.tokens,
                                   .max_tokens = static_cast<std::int32_t>(prompt.tokens.size())};
    const PeerAnswer answer = PeerExports::AnswerLookup(cache.coordinator, lookup, /*include_disk=*/true);
    EXPECT_EQ(answer.num_tokens, 12);
    EXPECT_TRUE(std::ranges::none_of(answer.pages, [](const PeerPage& page) { return page.tier == CacheTier::kDisk; }));
}

TEST(DiskTierTest, AClearDropsTheDiskTierWithTheOthers) {
    DiskTierCache cache(FullAndState(), /*disk_parents=*/16);
    const Prompt prompt(2, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    const DiskOperation write = cache.Clean(keys);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    ASSERT_GT(cache.coordinator.Disk().NumDiskCachedBlocks(), 0);
    ASSERT_TRUE(cache.coordinator.CacheIsClearable(/*include_host=*/true));
    ASSERT_TRUE(cache.coordinator.ClearCache());
    EXPECT_EQ(cache.coordinator.Disk().NumDiskCachedBlocks(), 0);
    EXPECT_EQ(cache.coordinator.Store().Boundaries().Size(), 0u);
}

TEST(DiskTierTest, AHeldDiskEntryRefusesAClear) {
    DiskTierCache cache(FullOnly(), /*disk_parents=*/16);
    const Prompt prompt(2, 1);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    const DiskOperation write = cache.Clean(keys);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    cache.EvictFromHost(keys);
    StorageLease read = cache.disks.StartRead(prompt.Prefix());
    ASSERT_TRUE(read);
    EXPECT_FALSE(cache.coordinator.CacheIsClearable(/*include_host=*/true));
    EXPECT_FALSE(cache.coordinator.ClearCache());
}

// The scheduler's side: a waiting prompt prefetches its Disk-only prefix
// before admission, and write-backs queue the Host entries they bring for
// cleaning.
class DiskTierSchedulerTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig config = SchedulerTestSuite::MakeConfig();
        config.prefix_granularity = 4;
        // Eight usable Device blocks: pressure's 23-token prompt evicts the
        // second of seed's two cached pages, which the write-back brings to
        // Host; the first stays on the Device.
        config.device_allocator.total_pages = 9;
        config.host_allocator.total_pages = 65;
        config.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
        config.disk_allocator.total_pages = 33;
        config.disk_config = DiskConfig{.max_operations = 4,
                                        .max_blocks_per_operation = 64,
                                        .max_host_blocks = 32,
                                        .max_write_host_blocks = 16,
                                        .max_queued_cleanings = 64,
                                        .min_prefetch_tokens = 1,
                                        .wait_timeout_us = 1000,
                                        .ready_timeout_us = 1000};
        config.cache_groups.front().block_granularity = 4;
        config.cache_groups.front().total_pages = 9;
        SetTestSnapshotPool(config);
        return config;
    }

    // The clock and permissions the control plane agrees on each round.
    void Tick(std::int64_t now_us) {
        ExecutionEvent event;
        event.With(cache::StorageTick{
            .now_us = now_us, .allow_forward = true, .allow_storage = true, .pressure = StoragePressure{0, 0, 0}});
        scheduler_->Advance(std::move(event));
    }

    // Whether the plan runs no forward row.
    static bool Idle(const ExecutionPlan& plan) {
        const ForwardBatch* batch = FindForwardBatch(plan);
        return batch == nullptr || batch->request_ids.empty();
    }

    static std::vector<DiskOperation> DiskOps(const ExecutionPlan& plan) {
        std::vector<DiskOperation> ops;
        for (const CacheOperation& op : ExtractCacheOpsOfKind<DiskOperation>(plan)) {
            ops.push_back(std::get<DiskOperation>(op));
        }
        return ops;
    }

    void AckDisk(const DiskOperation& op, bool success) {
        ExecutionEvent event;
        event.With(cache::DiskDone{.op_id = op.op_id, .success = success});
        scheduler_->Advance(std::move(event));
    }

    // seed's second page ends on Host: pressure evicts it from the Device.
    void SeedOnHost() {
        Submit(RequestSpec{.request_id = "seed", .admission_id = 11, .tokens = MakeTokens(8), .max_new_tokens = 1});
        ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
        SendForwardDone("seed", {900});
        SendFinish("seed");
        Submit(RequestSpec{
            .request_id = "pressure", .admission_id = 22, .tokens = MakeTokens(23, 101), .max_new_tokens = 1});
        const ExecutionPlan pressure = PlanOnce();
        ASSERT_FALSE(ExtractCacheOpsOfKind<WriteBackBatch>(pressure).empty());
        SendForwardDone("pressure", {900});
        SendFinish("pressure");
        AckWriteBacks(pressure);
    }

    // The Host drops seed's page, as a placement under pressure would drop a
    // clean entry.
    void DropSeedFromHost() {
        CacheCoordinator& coordinator = SchedulerTestAccess::Coordinator(*scheduler_);
        const std::vector<std::int32_t> tokens = MakeTokens(8);
        std::vector<std::span<const std::int32_t>> pages{std::span{tokens}.first(4), std::span{tokens}.last(4)};
        std::int32_t dropped = 0;
        for (const CacheKey& key : coordinator.Store().KeysForGroup(ComputePrefixHashes(pages, ""), 0)) {
            const auto location = coordinator.Store().LocationOf(*coordinator.Store().HostPool(), 0, key);
            if (location) {
                ASSERT_TRUE(coordinator.Disk().ContainsDiskCachedBlock(key)) << "only a clean entry is dropped";
                ASSERT_TRUE(
                    coordinator.Evictor().EvictHostCachedBlock(0, *location, &CacheEvictor::HostEvictionStats::lru));
                ++dropped;
            }
        }
        ASSERT_EQ(dropped, 1);
    }
};

TEST_F(DiskTierSchedulerTest, AWriteBackIsCleanedToDiskInTheNextPlan) {
    Tick(1);
    SeedOnHost();
    EXPECT_EQ(scheduler_->GetDiskStats().queued_cleanings, 1u);
    const std::vector<DiskOperation> writes = DiskOps(PlanOnce());
    ASSERT_EQ(writes.size(), 1u);
    EXPECT_EQ(writes[0].kind, DiskOpKind::kWrite);
    EXPECT_EQ(writes[0].group_ids.size(), 1u);
    EXPECT_TRUE(scheduler_->StoragePending()) << "a Disk write in flight counts as pending storage";
    AckDisk(writes[0], /*success=*/true);
    EXPECT_EQ(scheduler_->GetDiskStats().disk_blocks, 1u);
    EXPECT_FALSE(scheduler_->StoragePending());
}

TEST_F(DiskTierSchedulerTest, APageOnlyTheDiskHoldsIsPrefetchedThenAdmittedAsAHostHit) {
    Tick(1);
    SeedOnHost();
    const std::vector<DiskOperation> writes = DiskOps(PlanOnce());
    ASSERT_EQ(writes.size(), 1u);
    AckDisk(writes[0], /*success=*/true);
    DropSeedFromHost();

    Submit(RequestSpec{.request_id = "hit", .tokens = MakeTokens(9), .max_new_tokens = 1});
    const ExecutionPlan prefetch = PlanOnce();
    EXPECT_TRUE(Idle(prefetch)) << "the prompt waits for its prefetch, holding no pages";
    const std::vector<DiskOperation> reads = DiskOps(prefetch);
    ASSERT_EQ(reads.size(), 1u);
    EXPECT_EQ(reads[0].kind, DiskOpKind::kRead);
    EXPECT_EQ(reads[0].group_ids.size(), 1u) << "the Device holds the first page";
    EXPECT_EQ(scheduler_->GetDiskStats().waiting_requests, 1u);
    EXPECT_TRUE(Idle(PlanOnce())) << "still reading";

    AckDisk(reads[0], /*success=*/true);
    const ExecutionPlan admitted = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(admitted);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"hit"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{8})
        << "the Device page and the prefetched page resume the prompt";
    EXPECT_FALSE(ExtractCacheOpsOfKind<LoadBackBatch>(admitted).empty());
    EXPECT_EQ(scheduler_->GetDiskStats().ready_requests, 0u) << "the admission took the Ready hold";
}

TEST_F(DiskTierSchedulerTest, APrefetchPastItsDeadlineAdmitsFromTheLocalHit) {
    Tick(1);
    SeedOnHost();
    const std::vector<DiskOperation> writes = DiskOps(PlanOnce());
    ASSERT_EQ(writes.size(), 1u);
    AckDisk(writes[0], /*success=*/true);
    DropSeedFromHost();

    Submit(RequestSpec{.request_id = "hit", .tokens = MakeTokens(9), .max_new_tokens = 1});
    const std::vector<DiskOperation> reads = DiskOps(PlanOnce());
    ASSERT_EQ(reads.size(), 1u);
    Tick(5000);
    const ExecutionPlan admitted = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(admitted);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{4}) << "the Device page alone";
    EXPECT_EQ(scheduler_->GetDiskStats().wait_timeouts, 1u);
    EXPECT_TRUE(scheduler_->StoragePending()) << "the read keeps its blocks until it retires";
    AckDisk(reads[0], /*success=*/true);
    EXPECT_FALSE(scheduler_->StoragePending());
}

TEST_F(DiskTierSchedulerTest, DiskPrefetchAcquiresNoSlotUntilItsHostLoadHasAnAdmission) {
    auto sink = std::make_shared<RecordingCacheEventSink>();
    scheduler_->SetCacheEventSink(sink);
    Tick(1);
    SeedOnHost();
    const auto writes = DiskOps(PlanOnce());
    ASSERT_EQ(writes.size(), 1u);
    AckDisk(writes.front(), /*success=*/true);
    DropSeedFromHost();
    const auto prior_ownership = sink->slots.size();
    ASSERT_TRUE(sink->prefix_loads.empty());

    Submit(RequestSpec{.request_id = "hit", .admission_id = 33, .tokens = MakeTokens(9), .max_new_tokens = 1});
    const auto prefetch = PlanOnce();
    ASSERT_TRUE(Idle(prefetch));
    const auto reads = DiskOps(prefetch);
    ASSERT_EQ(reads.size(), 1u);
    EXPECT_EQ(reads.front().kind, DiskOpKind::kRead);
    EXPECT_EQ(scheduler_->AllocatedRequestSlots(), 0);
    EXPECT_EQ(sink->slots.size(), prior_ownership);
    EXPECT_TRUE(sink->prefix_loads.empty()) << "Disk-to-Host is not a Device prefix load";

    AckDisk(reads.front(), /*success=*/true);
    EXPECT_EQ(sink->slots.size(), prior_ownership) << "Disk retirement alone assigns no request slot";
    const auto admitted = PlanOnce();
    const auto loads = ExtractCacheOpsOfKind<LoadBackBatch>(admitted);
    ASSERT_EQ(loads.size(), 1u);
    const auto& load = std::get<LoadBackBatch>(loads.front());
    ASSERT_EQ(load.op_ids.size(), 1u);
    ASSERT_EQ(sink->prefix_loads.size(), 1u);
    const auto scheduled = sink->prefix_loads.front();
    ASSERT_NE(FindForwardBatch(admitted), nullptr);
    EXPECT_EQ(FindForwardBatch(admitted)->extend_prefix_lens, std::vector<std::int32_t>{8});
    EXPECT_EQ(load.src_pages.front(), reads.front().host_blocks);
    EXPECT_EQ(scheduled.op_id, load.op_ids.front());
    EXPECT_EQ(scheduled.telemetry, load.telemetry.front());
    EXPECT_EQ(scheduled.telemetry.admission_id, 33u);
    EXPECT_EQ(scheduled.telemetry.request_pool_index, sink->slots.back().slot);
    EXPECT_EQ(scheduled.telemetry.slot_generation, sink->slots.back().generation);

    // A late Host-to-Device completion still belongs to this admission,
    // even after the external request id is reused by another owner.
    SendAbortEvent("hit");
    DrainForwardResults("hit");
    PlanOnce();
    Submit(RequestSpec{.request_id = "hit", .admission_id = 44, .tokens = MakeTokens(2, 500), .max_new_tokens = 1});
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    EXPECT_EQ(sink->slots.back().admission_id, 44u);
    EXPECT_EQ(load.telemetry.front().admission_id, 33u);
    SendLoadBackDone(scheduled.op_id);
    EXPECT_EQ(scheduler_->AllocatedRequestSlots(), 1);
    EXPECT_TRUE(sink->restores.empty());
}

// Routers credit Disk hits too: a cleaned boundary is out on the Disk stream.
class DiskTierKvEventsTest : public DiskTierSchedulerTest {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig config = DiskTierSchedulerTest::MakeConfig();
        config.enable_kv_cache_events = true;
        return config;
    }
};

TEST_F(DiskTierKvEventsTest, ACleanedBoundaryIsOutOnTheDiskStream) {
    Tick(1);
    SeedOnHost();
    scheduler_->DrainKvEvents();
    const std::vector<DiskOperation> writes = DiskOps(PlanOnce());
    ASSERT_EQ(writes.size(), 1u);
    AckDisk(writes[0], /*success=*/true);
    const std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    EXPECT_TRUE(std::ranges::any_of(events, [](const KvCacheEvent& event) {
        const auto* stored = std::get_if<KvBlockStoredEvent>(&event);
        return stored != nullptr && stored->medium == KvEventMedium::kDisk;
    })) << "the cleaned page is out on the Disk stream";
}

TEST_F(DiskTierSchedulerTest, ACacheClearDrainsTheDiskOperationsInFlight) {
    Tick(1);
    SeedOnHost();
    const std::vector<DiskOperation> writes = DiskOps(PlanOnce());
    ASSERT_EQ(writes.size(), 1u);
    EXPECT_FALSE(scheduler_->CanClearCache()) << "the write is in flight";
    EXPECT_TRUE(scheduler_->CanPrepareStorageClear()) << "the drain discounts the write's Host sources";
    EXPECT_FALSE(scheduler_->CanClearCacheRetainingStorage()) << "a write finishing later would publish old contents";
    scheduler_->CancelStorageForClear();
    EXPECT_TRUE(scheduler_->StorageCurrentPending()) << "cancelling leaves the issued write to retire";
    AckDisk(writes[0], /*success=*/true);
    EXPECT_FALSE(scheduler_->StorageCurrentPending());
    ASSERT_TRUE(scheduler_->ClearCache());
    EXPECT_EQ(scheduler_->GetDiskStats().disk_blocks, 0u);
}

TEST(DiskTierConfigTest, ADiskTierNeedsHostWriteBackAndItsLimits) {
    SchedulerConfig config{};
    config.prefix_granularity = 4;
    config.device_allocator.total_pages = 9;
    config.host_allocator.total_pages = 9;
    config.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
    config.max_scheduled_tokens = 64;
    config.max_batch_size = 4;
    config.max_resident_requests = 4;
    config.cache_groups.push_back(
        CacheGroupConfig{.group_id = "full", .block_granularity = 4, .total_pages = 9, .attn_kind = AttnKind::kFull});
    config.host_group_packing = {1};
    SetTestSnapshotPool(config);
    config.disk_allocator.total_pages = 9;
    EXPECT_THROW(config.Validate(), std::invalid_argument) << "no disk_config";
    config.disk_config = DiskConfig{.max_operations = 1,
                                    .max_blocks_per_operation = 1,
                                    .max_host_blocks = 1,
                                    .max_write_host_blocks = 1,
                                    .max_queued_cleanings = 1,
                                    .min_prefetch_tokens = 1,
                                    .wait_timeout_us = 1,
                                    .ready_timeout_us = 1};
    EXPECT_THROW(config.Validate(), std::invalid_argument) << "write-through Host";
    config.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
    EXPECT_NO_THROW(config.Validate());
    // Credits count blocks: up to the eight usable parents times the packing.
    config.host_group_packing = {3};
    config.disk_config->max_host_blocks = 24;
    EXPECT_NO_THROW(config.Validate());
    config.disk_config->max_host_blocks = 25;
    EXPECT_THROW(config.Validate(), std::invalid_argument) << "more credits than Host blocks";
    config.disk_config->max_host_blocks = 1;
    config.disk_allocator.total_pages = 1;
    EXPECT_THROW(config.Validate(), std::invalid_argument) << "disk_config without a Disk tier";
}

TEST(DiskTierTest, CleaningAdmissionAndReadSlotReservation) {
    // 1. Cleaning reserves foreground read slots (`max_write_ops = max(1, 4/4) = 1`).
    DiskConfig limits = Limits();
    limits.max_operations = 4;
    limits.max_blocks_per_operation = 1;
    DiskTierCache c1(FullOnly(), 16, limits);
    c1.disks.QueueCleaning(c1.OnHostOnly(Prompt(2, 1)));
    c1.disks.QueueCleaning(c1.OnHostOnly(Prompt(2, 2)));
    c1.disks.StartCleaning();
    ASSERT_EQ(c1.disks.TakeCommands().size(), 1u);
    c1.disks.StartCleaning();
    EXPECT_TRUE(c1.disks.TakeCommands().empty());
    EXPECT_TRUE(c1.disks.HasReadCredits());

    // 2. min_clean_tokens and require_reuse_for_cleaning drop unready branches.
    DiskConfig gated = Limits();
    gated.min_clean_tokens = 12;
    DiskTierCache c2(FullOnly(), 16, gated);
    c2.disks.QueueCleaning(c2.OnHostOnly(Prompt(2, 1)));  // 8 < 12
    c2.disks.StartCleaning();
    EXPECT_EQ(c2.disks.DroppedCleanings(), 2u);
    const auto long_keys = c2.OnHostOnly(Prompt(4, 2));  // 16 >= 12
    EXPECT_EQ(c2.Clean(std::span{long_keys}.last(1)).group_ids.size(), 4u);

    DiskConfig reuse = Limits();
    reuse.require_reuse_for_cleaning = true;
    DiskTierCache c3(FullOnly(), 16, reuse);
    const auto cold = c3.OnHostOnly(Prompt(2, 1));
    c3.disks.QueueCleaning(cold);
    c3.disks.StartCleaning();
    // Not reused yet: nothing is written, and nothing is given up either.
    EXPECT_TRUE(c3.disks.TakeCommands().empty());
    EXPECT_EQ(c3.disks.DroppedCleanings(), 0u);
    EXPECT_EQ(c3.disks.NumAwaitingReuse(), 2u);
    // A later admission acquires the Host entries: the next plan cleans them.
    {
        auto& index = c3.coordinator.Store().Groups()[0].Index();
        const auto probe = FullAttnMatcher{}.Probe(index, c3.host, cold, 0, 2, /*extra_hits=*/nullptr);
        const PrefixMatch reused =
            index.AcquireMatched(c3.host, cold, 0, probe, CacheCoordinatorTestAccess::NextAccessEpoch(c3.coordinator),
                                 /*session=*/nullptr);
        EXPECT_EQ(reused.blocks.size(), 2u);
    }
    c3.disks.StartCleaning();
    const auto writes = c3.disks.TakeCommands();
    ASSERT_EQ(writes.size(), 1u);
    EXPECT_EQ(writes[0].group_ids.size(), 2u);
    EXPECT_EQ(c3.disks.NumAwaitingReuse(), 0u);
    EXPECT_EQ(c3.disks.DroppedCleanings(), 0u);

    // An entry the Host drops while it waits is forgotten.
    DiskTierCache c4(FullOnly(), 16, reuse);
    const auto gone = c4.OnHostOnly(Prompt(2, 3));
    c4.disks.QueueCleaning(gone);
    c4.disks.StartCleaning();
    c4.EvictFromHost(std::span{gone}.last(1));
    c4.EvictFromHost(std::span{gone}.first(1));
    c4.disks.StartCleaning();
    EXPECT_EQ(c4.disks.NumAwaitingReuse(), 0u);
    EXPECT_EQ(c4.disks.DroppedCleanings(), 2u);
}

TEST(DiskTierTest, PartialHostCreditExhaustionRetryableAndInt32BlockBound) {
    DiskConfig limits = Limits();
    limits.max_operations = 4;
    limits.max_blocks_per_operation = 8;
    limits.max_host_blocks = 5;
    limits.max_write_host_blocks = 5;
    DiskTierCache cache(FullAndState(), 16, limits);
    const Prompt p1(3, 1), p2(3, 2);
    const auto k1 = cache.OnHostOnly(p1), k2 = cache.OnHostOnly(p2);
    cache.disks.Complete(cache::DiskDone{.op_id = cache.Clean(k1).op_id, .success = true});
    cache.disks.Complete(cache::DiskDone{.op_id = cache.Clean(k2).op_id, .success = true});
    cache.EvictFromHost(k1);
    cache.EvictFromHost(k2);
    const auto r1 = cache.disks.TryStartRead(p1.Prefix(), /*local_tokens=*/0);
    ASSERT_TRUE(r1.lease);
    const auto r2 = cache.disks.TryStartRead(p2.Prefix(), /*local_tokens=*/0);
    EXPECT_FALSE(r2.lease);
    EXPECT_TRUE(r2.retryable);

    SchedulerConfig cfg{};
    cfg.prefix_granularity = 4;
    cfg.device_allocator.total_pages = 9;
    cfg.host_allocator.total_pages = 9;
    cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
    cfg.max_scheduled_tokens = 64;
    cfg.max_batch_size = 4;
    cfg.max_resident_requests = 4;
    cfg.cache_groups.push_back(
        CacheGroupConfig{.group_id = "full", .block_granularity = 4, .total_pages = 9, .attn_kind = AttnKind::kFull});
    cfg.host_group_packing = {4};
    SetTestSnapshotPool(cfg);
    cfg.disk_allocator.total_pages = (std::numeric_limits<std::int32_t>::max() / 4) + 2;
    cfg.disk_config = Limits();
    EXPECT_THROW(cfg.Validate(), std::invalid_argument);
}

TEST_F(DiskTierKvEventsTest, FailedReadDropsDiskEntryAndUnblocksWithoutPermanentLockout) {
    Tick(1);
    SeedOnHost();
    scheduler_->DrainKvEvents();
    const auto writes = DiskOps(PlanOnce());
    ASSERT_EQ(writes.size(), 1u);
    AckDisk(writes[0], /*success=*/true);
    ASSERT_FALSE(scheduler_->DrainKvEvents().empty());
    DropSeedFromHost();

    Submit(RequestSpec{.request_id = "hit", .tokens = MakeTokens(9), .max_new_tokens = 1});
    const ExecutionPlan prefetch = PlanOnce();
    ASSERT_TRUE(Idle(prefetch));
    const auto reads = DiskOps(prefetch);
    ASSERT_EQ(reads.size(), 1u);

    // Failed Disk read drops the unreadable Disk entry and lets the request admit locally.
    AckDisk(reads[0], /*success=*/false);
    EXPECT_EQ(scheduler_->GetDiskStats().disk_blocks, 0u);
    EXPECT_EQ(scheduler_->GetDiskStats().failed_reads, 1u);
    const ExecutionPlan admitted = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(admitted);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{4});
}

TEST_F(DiskTierKvEventsTest, SaturatedDiskCreditsDeferRetryUntilInFlightOperationFinishes) {
    Tick(1);
    SeedOnHost();
    const auto writes = DiskOps(PlanOnce());
    ASSERT_EQ(writes.size(), 1u);
    AckDisk(writes[0], /*success=*/true);
    DropSeedFromHost();

    // First request consumes the sole Disk read operation slot (`max_operations = 1`?
    // Fill remaining slots or saturate read credits by submitting multiple hits).
    const std::vector<std::int32_t> tokens = MakeTokens(9);
    for (int i = 0; i < 4; ++i) {
        Submit(RequestSpec{.request_id = "r" + std::to_string(i), .tokens = tokens, .max_new_tokens = 1});
    }
    // Plan with forward disallowed so waiting requests stay queued while read credits saturate.
    const ExecutionPlan first = scheduler_->NextExecutionPlan({.allow_prefill = false});
    const auto first_reads = DiskOps(first);
    ASSERT_FALSE(first_reads.empty());
    // While in-flight operations occupy all credits, another round emits no duplicate reads
    // and leaves waiting requests ready to retry once credits free up.
    const ExecutionPlan second = scheduler_->NextExecutionPlan({.allow_prefill = false});
    EXPECT_TRUE(DiskOps(second).empty());
    for (const auto& op : first_reads) {
        AckDisk(op, /*success=*/true);
    }
    const ExecutionPlan admitted = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(admitted);
    ASSERT_NE(batch, nullptr);
    EXPECT_FALSE(batch->request_ids.empty());
}

TEST(DiskTierTest, HybridStateDiskOnlyResidencyAndHotDescendantVisibility) {
    DiskTierCache cache(FullAndState(), 16);
    std::vector<ContentHash> observed;
    cache.coordinator.SetCacheMutationSink([&](const CacheKey& key, CacheMutation m) {
        if (m == CacheMutation::kLowerTierResidency) observed.push_back(key.content_hash);
    });

    const Prompt turn1(2, 41), turn2(4, 41);
    cache.disks.Complete(cache::DiskDone{.op_id = cache.Clean(cache.OnHostOnly(turn1)).op_id, .success = true});
    cache.disks.Complete(cache::DiskDone{.op_id = cache.Clean(cache.OnHostOnly(turn2)).op_id, .success = true});

    // Evicting P1 State while descendant P3 State stays on Host keeps RAM visibility.
    const ContentHash p1{turn1.hashes[1]}, p3{turn2.hashes[3]};
    cache.EvictFromHost(std::vector<CacheKey>{CacheKey{.group_id = 1, .content_hash = p1}});
    for (const auto& h : turn2.hashes) {
        EXPECT_TRUE(
            cache.coordinator.GetBoundaryEventStatus(CacheKey{.content_hash = ContentHash{h}}, 0).host_publishable);
    }

    // Evicting P3 State too makes all pages Disk-only and notifies every ancestor with kLowerTierResidency.
    observed.clear();
    cache.EvictFromHost(std::vector<CacheKey>{CacheKey{.group_id = 1, .content_hash = p3}});
    for (const auto& h : turn2.hashes) {
        const ContentHash ch{h};
        EXPECT_NE(std::ranges::find(observed, ch), observed.end());
        const auto st = cache.coordinator.GetBoundaryEventStatus(CacheKey{.content_hash = ch}, 0);
        EXPECT_FALSE(st.host_publishable);
        EXPECT_TRUE(st.disk_publishable);
    }

    // Prefetching P3 State back to Host restores host_publishable and notifies P1.
    observed.clear();
    StorageLease read = cache.disks.StartRead(turn2.Prefix());
    ASSERT_TRUE(read);
    ASSERT_EQ(cache.disks.TakeCommands().size(), 1u);
    cache.disks.Complete(cache::DiskDone{.op_id = read.OperationId(), .success = true});
    EXPECT_NE(std::ranges::find(observed, p1), observed.end());
    for (const auto& h : turn2.hashes) {
        EXPECT_TRUE(
            cache.coordinator.GetBoundaryEventStatus(CacheKey{.content_hash = ContentHash{h}}, 0).host_publishable);
    }
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(cache.coordinator).empty());
}

TEST(DiskTierTest, HybridPeerListAndPinKeepsDeviceStateWhenKvIsOnDisk) {
    DiskTierCache cache(FullAndState(), 16);
    const Prompt prompt(2, 77);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    cache.disks.Complete(cache::DiskDone{.op_id = cache.Clean(keys).op_id, .success = true});
    // Evict every group from Host so the KV prefix lives exclusively on Disk.
    cache.EvictFromHost(keys);
    // Re-publish only the State checkpoint on Device at the same boundary:
    // ListWithDisk has split_tokens == 0 (KV pages live on Disk) while the
    // State checkpoint resides on Device.
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(cache.coordinator, tables, static_cast<std::int32_t>(prompt.tokens.size())));
    CacheCompletedBlocksForTest(cache.coordinator, tables, prompt.hashes,
                                CacheCoordinatorTestAccess::NextAccessEpoch(cache.coordinator),
                                /*first_new_prefix_page=*/0, static_cast<std::int32_t>(prompt.tokens.size()),
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/true);
    cache.coordinator.Free(tables);
    // Drop Device KV blocks (group 0) while retaining the Device State block (group 1).
    for (const auto& h : prompt.hashes) {
        const CacheKey key{.group_id = 0, .content_hash = ContentHash{h}};
        const auto location = cache.coordinator.Store().LocationOf(cache.device, 0, key);
        ASSERT_TRUE(location);
        ASSERT_TRUE(cache.coordinator.Evictor().EvictCachedBlockAlone(
            0, *location, &CacheEvictor::DeviceEvictionStats::admission, /*write_back_segment=*/nullptr));
    }

    const cache::PeerLookup lookup{.lease = {.reader = "tp8-reader", .id = 1},
                                   .prefix_hashes = prompt.hashes,
                                   .tokens = prompt.tokens,
                                   .max_tokens = static_cast<std::int32_t>(prompt.tokens.size())};
    const PeerAnswer answer = PeerExports::AnswerLookup(cache.coordinator, lookup, /*include_disk=*/true);
    ASSERT_EQ(answer.num_tokens, static_cast<std::int32_t>(prompt.tokens.size()));
    ASSERT_EQ(answer.pages.size(), 3u);
    auto find_page = [&](std::uint32_t group, std::int32_t slot) {
        const auto it = std::ranges::find_if(answer.pages,
                                             [&](const PeerPage& p) { return p.group_id == group && p.slot == slot; });
        EXPECT_NE(it, answer.pages.end());
        return it != answer.pages.end() ? *it : PeerPage{};
    };
    EXPECT_EQ(find_page(0u, 0).tier, CacheTier::kDisk);
    EXPECT_EQ(find_page(0u, 1).tier, CacheTier::kDisk);
    EXPECT_EQ(find_page(1u, kPeerStateSlot).tier, CacheTier::kDevice);

    PeerConfig peer_cfg{.name = "holder",
                        .incarnation = 1,
                        .max_leases = 4,
                        .soft_pin_us = 1000,
                        .hard_pin_us = 10000,
                        .lookup_timeout_us = 1000,
                        .min_fetch_tokens = kGrain,
                        .landing_tier = PeerLandingTier::kDevice,
                        .max_host_landing_blocks = 0,
                        .host_ready_timeout_us = 0,
                        .versioned_shards = true};
    PeerExports exports{peer_cfg};
    exports.Lookup(cache.coordinator, lookup, /*now_us=*/0);
    exports.Pin(cache.coordinator,
                cache::PeerPin{.lease = answer.lease, .num_tokens = answer.num_tokens, .pages = answer.pages},
                /*now_us=*/0, nullptr);
    EXPECT_TRUE(exports.Held(answer.lease));
    EXPECT_EQ(exports.NumPinnedBlocks(), 3u);
    EXPECT_FALSE(cache.coordinator.ClearCache()) << "Disk + Device pins hold blocks";

    exports.Release(cache.coordinator, cache::PeerRelease{.lease = answer.lease}, /*now_us=*/0);
    EXPECT_EQ(exports.NumPinnedBlocks(), 0u);
    EXPECT_EQ(exports.NumLeases(), 0u);
    EXPECT_TRUE(cache.coordinator.ClearCache());
}

TEST(DiskTierTest, HybridPeerListWithoutDiskKeepsDeviceStateWhenKvIsOnHost) {
    // Regression test for P2-1: when Device KV is evicted (split_tokens == 0)
    // while Host KV and a Device State checkpoint are both present, both
    // `AnswerLookupWithoutDisk` (`ListPrefix`) and `ExportPrefix` must report
    // the Device State checkpoint (`kDevice`) alongside Host KV (`kHost`).
    DiskTierCache cache(FullAndState(), 16);
    const Prompt prompt(2, 83);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(cache.coordinator, tables, static_cast<std::int32_t>(prompt.tokens.size())));
    CacheCompletedBlocksForTest(cache.coordinator, tables, prompt.hashes,
                                CacheCoordinatorTestAccess::NextAccessEpoch(cache.coordinator),
                                /*first_new_prefix_page=*/0, static_cast<std::int32_t>(prompt.tokens.size()),
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/true);
    cache.coordinator.Free(tables);
    // Copy KV pages to Host and evict KV from Device while retaining Device State.
    for (const auto& h : prompt.hashes) {
        const CacheKey key{.group_id = 0, .content_hash = ContentHash{h}};
        CacheBlockRef host_blk = cache.coordinator.Host().AcquireHostBlock(0, 0);
        ASSERT_TRUE(host_blk);
        cache.coordinator.Host().CacheHostBlock(host_blk, key, std::nullopt);
        const auto dev_loc = cache.coordinator.Store().LocationOf(cache.device, 0, key);
        ASSERT_TRUE(dev_loc);
        ASSERT_TRUE(cache.coordinator.Evictor().EvictCachedBlockAlone(
            0, *dev_loc, &CacheEvictor::DeviceEvictionStats::admission, /*write_back_segment=*/nullptr));
    }

    const cache::PeerLookup lookup{.lease = {.reader = "no-disk-reader", .id = 1},
                                   .prefix_hashes = prompt.hashes,
                                   .tokens = prompt.tokens,
                                   .max_tokens = static_cast<std::int32_t>(prompt.tokens.size())};
    const PeerAnswer answer = PeerExports::AnswerLookup(cache.coordinator, lookup, /*include_disk=*/false);
    ASSERT_EQ(answer.num_tokens, static_cast<std::int32_t>(prompt.tokens.size()));
    ASSERT_EQ(answer.pages.size(), 3u);
    auto find_page = [&](std::uint32_t group, std::int32_t slot) {
        const auto it = std::ranges::find_if(answer.pages,
                                             [&](const PeerPage& p) { return p.group_id == group && p.slot == slot; });
        EXPECT_NE(it, answer.pages.end());
        return it != answer.pages.end() ? *it : PeerPage{};
    };
    EXPECT_EQ(find_page(0u, 0).tier, CacheTier::kHost);
    EXPECT_EQ(find_page(0u, 1).tier, CacheTier::kHost);
    EXPECT_EQ(find_page(1u, kPeerStateSlot).tier, CacheTier::kDevice);

    const auto exported = cache.coordinator.ExportPrefix(cache.coordinator.Prober().ProbePromptPrefix(prompt.Prefix()));
    EXPECT_EQ(exported.num_tokens, answer.num_tokens);
    ASSERT_EQ(exported.blocks.size(), 3u);
    EXPECT_EQ(exported.blocks[2].tier, CacheTier::kDevice);
}

TEST(DiskTierTest, ShardedDiskPrefetchPreservesDiskBlockOwnerBucket) {
    // Reproduces KVP8 (shard_count = 8) local SSD prefetch where Disk blocks
    // reside in arbitrary owner buckets `(j + 3) % 8 != logical_block % 8`.
    std::vector<CacheGroupSpec> specs{
        {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 8, .block_granularity = kGrain, .shard_count = 8},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0, .shard_count = 1},
    };
    DiskTierCache cache(specs, /*disk_parents=*/16);
    const Prompt prompt(/*pages=*/5, /*seed=*/91);

    // Publish on Device first to register boundary tree nodes, then place each
    // Host KV block in bucket `(j + 3) % 8` so none matches `logical_block % 8`.
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(cache.coordinator, tables, static_cast<std::int32_t>(prompt.tokens.size())));
    CacheCompletedBlocksForTest(cache.coordinator, tables, prompt.hashes,
                                CacheCoordinatorTestAccess::NextAccessEpoch(cache.coordinator),
                                /*first_new_prefix_page=*/0, static_cast<std::int32_t>(prompt.tokens.size()),
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/true);
    cache.coordinator.Free(tables);
    std::vector<CacheKey> keys = cache.DeviceKeys(prompt);
    ASSERT_EQ(keys.size(), 6u);  // 5 KV pages + 1 State checkpoint
    std::int32_t kv_idx = 0;
    for (const CacheKey& key : keys) {
        const std::int32_t bucket = key.group_id == 0 ? ((kv_idx++ + 3) % 8) : 0;
        CacheBlockRef block = cache.coordinator.Host().AcquireHostBlock(key.group_id, bucket);
        ASSERT_TRUE(block);
        cache.coordinator.Host().CacheHostBlock(block, key, std::nullopt);
    }
    ASSERT_TRUE(cache.coordinator.ClearDeviceCache());

    // Clean Host -> Disk (Disk preserves each Host block's owner bucket).
    const DiskOperation write = cache.Clean(keys);
    ASSERT_EQ(write.group_ids.size(), 6u);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    cache.EvictFromHost(keys);

    // Prefetch Disk -> Host: every row must pair equal-owner Host and Disk blocks.
    StorageLease read = cache.disks.StartRead(prompt.Prefix());
    ASSERT_TRUE(read);
    const auto ops = cache.disks.TakeCommands();
    ASSERT_EQ(ops.size(), 1u);
    const DiskOperation& op = ops[0];
    ASSERT_EQ(op.group_ids.size(), 6u);
    for (std::size_t row = 0; row < op.group_ids.size(); ++row) {
        if (op.group_ids[row] == 0u) {
            EXPECT_EQ((op.host_blocks[row] - 1) % 8, (op.disk_blocks[row] - 1) % 8)
                << "row " << row << " paired Host and Disk blocks from different KVP8 owners";
        }
    }
    cache.disks.Complete(cache::DiskDone{.op_id = op.op_id, .success = true});
    for (const CacheKey& key : keys) {
        EXPECT_TRUE(cache.OnHost(key));
    }
}

TEST(DiskTierTest, ShardedDiskPrefetchMixedTiersPreservesOwners) {
    // Pages 0..1 stay on Host, pages 2..4 live only on Disk with shifted KVP8
    // owner buckets `(j + 5) % 8`; Disk prefetch must skip pages already on Host
    // and allocate Host destinations in each Disk block's own bucket.
    std::vector<CacheGroupSpec> specs{
        {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 8, .block_granularity = kGrain, .shard_count = 8},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0, .shard_count = 1},
    };
    DiskTierCache cache(specs, /*disk_parents=*/16);
    const Prompt prompt(/*pages=*/5, /*seed=*/97);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(cache.coordinator, tables, static_cast<std::int32_t>(prompt.tokens.size())));
    CacheCompletedBlocksForTest(cache.coordinator, tables, prompt.hashes,
                                CacheCoordinatorTestAccess::NextAccessEpoch(cache.coordinator),
                                /*first_new_prefix_page=*/0, static_cast<std::int32_t>(prompt.tokens.size()),
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/true);
    cache.coordinator.Free(tables);
    std::vector<CacheKey> keys = cache.DeviceKeys(prompt);
    ASSERT_EQ(keys.size(), 6u);
    std::int32_t kv_idx = 0;
    for (const CacheKey& key : keys) {
        const std::int32_t bucket = key.group_id == 0 ? ((kv_idx++ + 5) % 8) : 0;
        CacheBlockRef block = cache.coordinator.Host().AcquireHostBlock(key.group_id, bucket);
        ASSERT_TRUE(block);
        cache.coordinator.Host().CacheHostBlock(block, key, std::nullopt);
    }
    ASSERT_TRUE(cache.coordinator.ClearDeviceCache());
    const DiskOperation write = cache.Clean(keys);
    cache.disks.Complete(cache::DiskDone{.op_id = write.op_id, .success = true});
    // Evict only pages 2..4 + State from Host; keep pages 0..1 on Host.
    std::vector<CacheKey> tail_keys(keys.begin() + 2, keys.end());
    cache.EvictFromHost(tail_keys);

    StorageLease read = cache.disks.StartRead(prompt.Prefix());
    ASSERT_TRUE(read);
    const auto ops = cache.disks.TakeCommands();
    ASSERT_EQ(ops.size(), 1u);
    const DiskOperation& op = ops[0];
    ASSERT_EQ(op.group_ids.size(), 4u);  // only the 3 Disk KV tail pages + State
    for (std::size_t row = 0; row < op.group_ids.size(); ++row) {
        if (op.group_ids[row] == 0u) {
            EXPECT_EQ((op.host_blocks[row] - 1) % 8, (op.disk_blocks[row] - 1) % 8);
        }
    }
    cache.disks.Complete(cache::DiskDone{.op_id = op.op_id, .success = true});
    for (const CacheKey& key : keys) {
        EXPECT_TRUE(cache.OnHost(key));
    }
}

TEST(DiskTierTest, DiskNodeEntriesIncludesDiskOnlyStateCheckpoint) {
    // Regression test for Fix #3: CacheStore::NodeEntries(disk_pool, node) must
    // check `BoundaryTree::kDisk` (not `kDevice`) so Disk-only State checkpoints
    // are enumerated even when neither Device nor Host holds a checkpoint.
    DiskTierCache cache(FullAndState(), 16);
    const Prompt prompt(2, 101);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    cache.disks.Complete(cache::DiskDone{.op_id = cache.Clean(keys).op_id, .success = true});
    cache.EvictFromHost(keys);

    const BoundaryTree::Node* tip = cache.coordinator.Store().Boundaries().Find(ContentHash{prompt.hashes.back()});
    ASSERT_NE(tip, nullptr);
    ASSERT_FALSE(tip->HasCheckpoint(BoundaryTree::kDevice));
    ASSERT_FALSE(tip->HasCheckpoint(BoundaryTree::kHost));
    ASSERT_TRUE(tip->HasCheckpoint(BoundaryTree::kDisk));

    const auto entries = cache.coordinator.Store().NodeEntries(*cache.coordinator.Store().DiskPool(), *tip);
    const bool found_state =
        std::ranges::any_of(entries, [](const CacheStore::NodeEntry& e) { return e.group_id == 1; });
    const bool found_full =
        std::ranges::any_of(entries, [](const CacheStore::NodeEntry& e) { return e.group_id == 0; });
    EXPECT_TRUE(found_state);
    EXPECT_TRUE(found_full);
}

TEST(DiskTierTest, SwaDiskOnlyBoundaryReportsDiskPublishable) {
    // Regression test for Fix #4a: when a non-replayable SWA group (e.g. DFLASH
    // draft) has been evicted from Host/Device to Disk while Full History stays
    // in Host RAM, GetBoundaryEventStatus must downgrade the boundary from
    // Host/Device to Disk publishable.
    std::vector<CacheGroupSpec> specs{
        {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kGrain},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .sink_tokens = 2,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = kGrain},
    };
    DiskTierCache cache(specs, 16);
    const Prompt prompt(4, 103);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    cache.disks.Complete(cache::DiskDone{.op_id = cache.Clean(keys).op_id, .success = true});

    // Evict only SWA (group 1) from Host while keeping Full History (group 0) on Host.
    std::vector<CacheKey> swa_keys;
    for (const CacheKey& key : keys) {
        if (key.group_id == 1) swa_keys.push_back(key);
    }
    ASSERT_FALSE(swa_keys.empty());
    cache.EvictFromHost(swa_keys);

    const auto status =
        cache.coordinator.GetBoundaryEventStatus(CacheKey{.content_hash = ContentHash{prompt.hashes.back()}}, 0);
    EXPECT_FALSE(status.device_publishable);
    EXPECT_FALSE(status.host_publishable);
    EXPECT_TRUE(status.disk_publishable);
}

TEST(DiskTierTest, SwaDiskOnlyBoundaryWithStateRespectsHotterRamSubtree) {
    // Covers the `NumStateGroups() > 0` branch of `swa_disk_only` in
    // `CacheCoordinator::GetBoundaryEventStatus`: when SWA is Disk-only while
    // Full History and State remain in Host RAM, the boundary downgrades to
    // Disk publishable if no deeper child holds a RAM checkpoint, and stays
    // Host publishable when a deeper descendant holds a hotter RAM checkpoint.
    std::vector<CacheGroupSpec> specs = FullAndState();
    specs.push_back({.attn_kind = AttnKind::kSlidingWindow,
                     .sliding_window = 8,
                     .sink_tokens = 2,
                     .cache_blocks_per_lcm_block = 1,
                     .block_granularity = kGrain});
    DiskTierCache cache(specs, 24);
    const Prompt prompt(3, 105);
    const std::vector<CacheKey> keys = cache.OnHostOnly(prompt);
    cache.disks.Complete(cache::DiskDone{.op_id = cache.Clean(keys).op_id, .success = true});

    // Evict SWA (group 2) at the middle boundary (page 1) and tip (page 2) from Host.
    std::vector<CacheKey> swa_keys;
    for (const CacheKey& key : keys) {
        if (key.group_id == 2 &&
            (key.content_hash == ContentHash{prompt.hashes[1]} || key.content_hash == ContentHash{prompt.hashes[2]})) {
            swa_keys.push_back(key);
        }
    }
    ASSERT_EQ(swa_keys.size(), 2u);
    cache.EvictFromHost(swa_keys);

    // At the tip (page 2), no deeper child holds a RAM checkpoint -> Disk publishable.
    const auto tip_status =
        cache.coordinator.GetBoundaryEventStatus(CacheKey{.content_hash = ContentHash{prompt.hashes[2]}}, 0);
    EXPECT_FALSE(tip_status.device_publishable);
    EXPECT_FALSE(tip_status.host_publishable);
    EXPECT_TRUE(tip_status.disk_publishable);

    // At the middle boundary (page 1), page 2 still holds a State checkpoint in Host RAM,
    // so `HasRamCheckpointInProperSubtree()` is true and page 1 stays Host publishable.
    const auto mid_status =
        cache.coordinator.GetBoundaryEventStatus(CacheKey{.content_hash = ContentHash{prompt.hashes[1]}}, 0);
    EXPECT_FALSE(mid_status.device_publishable);
    EXPECT_TRUE(mid_status.host_publishable);
    EXPECT_TRUE(mid_status.disk_publishable);
}

DiskConfig GdsDiskLimits() {
    auto limits = Limits();
    limits.max_host_blocks = 64;
    limits.max_write_host_blocks = 64;
    return limits;
}

void ConfigureDiskGds(DiskTierCache& cache) {
    cache.coordinator.ConfigureEviction(
        PrefixCacheEvictionConfig::Gds(PrefillCost{1, 0}, std::vector<double>(cache.coordinator.NumGroups(), 0.01)));
}

void CopyToDisk(DiskTierCache& cache, std::span<const CacheKey> keys) {
    for (const auto& key : keys) {
        const std::array group{key.group_id};
        const std::array<std::int32_t, 1> bucket{0};
        auto batch = cache.coordinator.Disk().AcquireDiskBlocks(group, bucket);
        ASSERT_EQ(batch.allocated, 1u);
        const auto source = cache.coordinator.Store().Groups()[key.group_id].Index().Inspect(cache.host, key);
        ASSERT_TRUE(source);
        cache.coordinator.Disk().CacheDiskBlock(batch.blocks[0], key, source->metadata);
    }
}

std::vector<CacheBlockRef> HoldFreeHost(DiskTierCache& cache) {
    std::vector<CacheBlockRef> held;
    const std::array<std::int32_t, 1> bucket{0};
    for (;;) {
        auto blocks = cache.host.AcquireBlocksInBuckets(0, bucket);
        if (blocks.empty()) return held;
        held.push_back(std::move(blocks.front()));
    }
}

TEST(DiskGdsTest, CleaningGatesTheTipAndPreservesColdAncestorDependencies) {
    for (bool disk_tip : {false, true}) {
        SCOPED_TRACE(disk_tip);
        auto limits = GdsDiskLimits();
        limits.min_clean_tokens = 12;
        limits.require_reuse_for_cleaning = true;
        DiskTierCache cache(FullAndState(), 32, limits);
        ConfigureDiskGds(cache);
        const auto short_keys = cache.OnHostOnly(Prompt(2, 1));
        const auto long_keys = cache.OnHostOnly(Prompt(4, 2));
        cache.disks.StartCleaning();
        EXPECT_TRUE(cache.disks.TakeCommands().empty());
        // Reuse both tips: the short branch stays gated by length. Cold
        // ancestors still belong to the eligible long branch's Disk closure.
        for (const auto& key : {short_keys[1], long_keys[3]}) {
            const std::array tip{key};
            auto& index = cache.coordinator.Store().Groups()[0].Index();
            const auto probe = FullAttnMatcher{}.Probe(index, cache.host, tip, 0, 1, /*extra_hits=*/nullptr);
            const auto reused = index.AcquireMatched(cache.host, tip, 0, probe,
                                                     CacheCoordinatorTestAccess::NextAccessEpoch(cache.coordinator),
                                                     /*session=*/nullptr);
            ASSERT_EQ(reused.blocks.size(), 1u);
        }
        const std::array tip{long_keys[3]};
        if (disk_tip) {
            CopyToDisk(cache, tip);
            cache.EvictFromHost(tip);
        }
        cache.disks.StartCleaning();
        const auto writes = cache.disks.TakeCommands();
        ASSERT_EQ(writes.size(), 1u);
        EXPECT_EQ(writes[0].group_ids.size(), long_keys.size() - (disk_tip ? 1u : 0u));
        cache.disks.Complete(cache::DiskDone{writes[0].op_id, true});
        for (const auto& key : long_keys) EXPECT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(key));
        for (const auto& key : short_keys) EXPECT_FALSE(cache.coordinator.Disk().ContainsDiskCachedBlock(key));
    }
}

TEST(DiskGdsTest, CleaningLeavesReadCreditsAndPartialHostCreditsRemainRetryable) {
    auto limits = GdsDiskLimits();
    limits.max_host_blocks = 5;
    limits.max_write_host_blocks = 5;
    limits.max_blocks_per_operation = 8;
    DiskTierCache cache(FullAndState(), 32, limits, /*host_parents=*/12);
    ConfigureDiskGds(cache);
    const Prompt first(3, 1), second(3, 2);
    const auto first_keys = cache.OnHostOnly(first);
    const auto second_keys = cache.OnHostOnly(second);
    CopyToDisk(cache, first_keys);
    CopyToDisk(cache, second_keys);
    cache.EvictFromHost(first_keys);
    cache.EvictFromHost(second_keys);
    const auto first_read = cache.disks.TryStartRead(first.Prefix(), /*local_tokens=*/0);
    ASSERT_TRUE(first_read.lease);
    const auto second_read = cache.disks.TryStartRead(second.Prefix(), /*local_tokens=*/0);
    EXPECT_FALSE(second_read.lease);
    EXPECT_TRUE(second_read.retryable);
    cache.disks.TakeCommands();
    cache.disks.Complete(cache::DiskDone{first_read.lease.OperationId(), true});

    auto write_limits = GdsDiskLimits();
    write_limits.max_blocks_per_operation = 1;
    DiskTierCache writing(FullAndState(), 32, write_limits);
    ConfigureDiskGds(writing);
    writing.OnHostOnly(Prompt(2, 3));
    writing.OnHostOnly(Prompt(2, 4));
    writing.disks.StartCleaning();
    ASSERT_EQ(writing.disks.TakeCommands().size(), 1u);
    writing.disks.StartCleaning();
    EXPECT_TRUE(writing.disks.TakeCommands().empty());
    EXPECT_TRUE(writing.disks.HasReadCredits());
}

TEST(DiskGdsTest, DiskAncestorProtectsHostStateAndReadArrivalDoesNotCountAsReuse) {
    DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
    ConfigureDiskGds(cache);
    const Prompt a(4, 1), b(1, 2);
    const auto a_keys = cache.OnHostOnly(a);
    const auto b_keys = cache.OnHostOnly(b);
    {
        const auto used = cache.coordinator.ExportListing(
            cache.coordinator.Prober().ListPrefix(cache.coordinator.Prober().ProbePromptPrefix(b.Prefix())));
        ASSERT_GT(used.num_tokens, 0);
    }
    CopyToDisk(cache, std::span{a_keys}.first(1));
    cache.EvictFromHost(std::span{a_keys}.first(1));
    const ContentHash hash{a.hashes.back()};
    auto& store = cache.coordinator.Store();
    const auto route = store.RecoveryDependencies().At(hash);
    EXPECT_FALSE(route.ready);
    EXPECT_TRUE(route.disk_recoverable);
    EXPECT_NEAR(route.disk_read_cost, 0.01, 1e-12);
    EXPECT_EQ(store.GdsStats().broken_units, 0u);
    auto held = HoldFreeHost(cache);
    auto destination = cache.coordinator.Host().AcquireHostBlock(1, 0);
    ASSERT_TRUE(destination);
    EXPECT_TRUE(cache.coordinator.Host().ContainsHostCachedBlock(a_keys.back()));
    EXPECT_FALSE(cache.coordinator.Host().ContainsHostCachedBlock(b_keys.back()));
    EXPECT_GT(store.GdsStats().host_watermark, 0);
    auto before = store.Checkpoints().RecoveryPoints();
    auto read = cache.disks.StartRead(a.Prefix());
    ASSERT_TRUE(read);
    auto commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(commands[0].group_ids.size(), 1u);
    cache.disks.Complete(cache::DiskDone{read.OperationId(), true});
    EXPECT_TRUE(store.RecoveryDependencies().At(hash).ready);
    EXPECT_EQ(cache.HostHitTokens(a), 16);
    const auto after = store.Checkpoints().RecoveryPoints();
    ASSERT_EQ(before.size(), after.size());
    EXPECT_DOUBLE_EQ(before.front().stamp, after.front().stamp);
    EXPECT_DOUBLE_EQ(before.front().host_stamp, after.front().host_stamp);
}

TEST(DiskGdsTest, RejectedHostSegmentRestoresCoverageBeforeLaterSingleRows) {
    DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
    ConfigureDiskGds(cache);
    const Prompt prompt(1, 1);
    const auto keys = cache.OnHostOnly(prompt);
    ASSERT_EQ(keys.size(), 2u);
    CopyToDisk(cache, std::span{keys}.first(1));
    auto held = HoldFreeHost(cache);
    const std::array<std::uint32_t, 5> groups{1, 1, 1, 1, 1};
    const std::array<std::int32_t, 5> buckets{0, 0, 0, 0, 0};
    const std::array<std::size_t, 4> ends{2, 3, 4, 5};
    const auto& store = cache.coordinator.Store();
    // Only the clean KV page can leave. The two-row segment tries that
    // removal but cannot fit; one following row can use it, and no more can.
    auto selected = PlanTieredHostAllocation(store, groups, buckets, ends, false, false);
    EXPECT_EQ(selected.accepted, (std::vector<bool>{false, false, true, false, false}));
    ASSERT_EQ(selected.plan->host_removals.size(), 1u);
    EXPECT_EQ(selected.plan->host_removals.front().key, keys.front());
    EXPECT_EQ(selected.plan->stats.host_clean_copies_released, 1u);
    EXPECT_EQ(selected.plan->stats.host_copy_rejections, 4u);
    EXPECT_TRUE(selected.plan->losses.empty());
    EXPECT_TRUE(selected.plan->fresh_rejections.empty());
    EXPECT_DOUBLE_EQ(selected.plan->stats.host_watermark, 0);
    for (const auto& key : keys) EXPECT_TRUE(cache.OnHost(key));

    std::pmr::monotonic_buffer_resource arena;
    auto victims = TieredVictims(selected.plan, &arena);
    CacheCoordinatorTestAccess::CommitVictims(cache.coordinator, victims);
    EXPECT_FALSE(cache.OnHost(keys.front()));
    EXPECT_TRUE(cache.OnHost(keys.back()));
    EXPECT_EQ(cache.host.NumEmptyLcmBlocks(), 1);
    EXPECT_TRUE(store.RecoveryDependencies().At(ContentHash{prompt.hashes.back()}).disk_recoverable);
    EXPECT_EQ(store.GdsStats().logical_evictions, 0u);
    EXPECT_DOUBLE_EQ(store.GdsStats().host_watermark, 0);
}

TEST(DiskGdsTest, OptionalRowsReclaimDeviceDuplicatesThenDiskCopiesWithoutLogicalLoss) {
    DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
    ConfigureDiskGds(cache);
    const Prompt old(1, 1), active(1, 2);
    const auto old_keys = cache.OnHostOnly(old);
    CopyToDisk(cache, std::span{old_keys}.first(1));
    auto tables = cache.OnDevice(active);
    const CacheKey duplicate{.group_id = 1, .content_hash = ContentHash{active.hashes.back()}};
    {
        auto block = cache.host.AcquireBlock(1);
        ASSERT_TRUE(block);
        cache.coordinator.Host().CacheHostBlock(block, duplicate, std::nullopt);
    }
    auto held = HoldFreeHost(cache);
    const std::array<std::uint32_t, 3> groups{1, 1, 1};
    const std::array<std::int32_t, 3> buckets{};
    auto placed = cache.coordinator.Host().AcquireHostCopies(groups, buckets);
    ASSERT_EQ(placed.blocks.size(), groups.size());
    EXPECT_TRUE(placed.blocks[0]);
    EXPECT_TRUE(placed.blocks[1]);
    EXPECT_FALSE(placed.blocks[2]);
    EXPECT_FALSE(cache.OnHost(duplicate));
    EXPECT_FALSE(cache.OnHost(old_keys.front()));
    EXPECT_TRUE(cache.OnHost(old_keys.back()));
    const auto& store = cache.coordinator.Store();
    const auto& route = store.RecoveryDependencies().At(ContentHash{old.hashes.back()});
    EXPECT_TRUE(route.disk_recoverable);
    EXPECT_GT(route.disk_read_cost, 0);
    EXPECT_TRUE(store.RecoveryDependencies().At(duplicate.content_hash).device_ready);
    EXPECT_EQ(store.GdsStats().host_clean_copies_released, 1u);
    EXPECT_EQ(store.GdsStats().host_copy_rejections, 1u);
    EXPECT_EQ(store.GdsStats().logical_evictions, 0u);
    EXPECT_EQ(store.GdsStats().logical_rejections, 0u);
    EXPECT_DOUBLE_EQ(store.GdsStats().host_watermark, 0);
    cache.coordinator.Free(tables);
}

TEST(DiskGdsTest, CleanStatePrecedesDirtyLossUnderBothPolicies) {
    for (bool gds : {false, true}) {
        SCOPED_TRACE(gds);
        DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
        if (gds) ConfigureDiskGds(cache);
        const auto a = cache.OnHostOnly(Prompt(4, 1));
        const auto b = cache.OnHostOnly(Prompt(1, 2));
        CopyToDisk(cache, std::span{a}.last(1));
        auto held = HoldFreeHost(cache);
        auto block = cache.coordinator.Host().AcquireHostBlock(1, 0);
        ASSERT_TRUE(block);
        EXPECT_FALSE(cache.coordinator.Host().ContainsHostCachedBlock(a.back()));
        EXPECT_TRUE(cache.coordinator.Host().ContainsHostCachedBlock(b.back()));
        EXPECT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(a.back()));
        const auto stats = cache.coordinator.Store().GdsStats();
        EXPECT_DOUBLE_EQ(stats.host_watermark, 0);
        EXPECT_EQ(stats.logical_evictions, 0u);
        EXPECT_EQ(stats.logical_rejections, 0u);
        if (gds) EXPECT_EQ(stats.host_clean_copies_released, 1u);
    }
}

TEST(DiskGdsTest, BlockPlacementReclaimsDeadEntriesBeforeCleanCopies) {
    for (bool gds : {false, true}) {
        for (bool whole : {false, true}) {
            for (bool gap : {false, true}) {
                SCOPED_TRACE(::testing::Message() << gds << ":" << whole << ":" << gap);
                DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
                if (gds) ConfigureDiskGds(cache);
                const auto clean = cache.OnHostOnly(Prompt(1, 1));
                CopyToDisk(cache, clean);
                const auto dead = cache.OnHostOnly(Prompt(2, 2));
                cache.EvictFromHost(gap ? std::span{dead}.first(1) : std::span{dead}.last(1));
                const auto dead_before =
                    std::ranges::count_if(dead, [&](const CacheKey& key) { return cache.OnHost(key); });
                auto held = HoldFreeHost(cache);
                const std::array<std::uint32_t, 1> groups{0};
                const std::array<std::int32_t, 1> buckets{0};
                auto placed = whole ? cache.coordinator.Host().AcquireHostLanding(groups, buckets)
                                    : cache.coordinator.Host().AcquireHostBlocks(groups, buckets);
                ASSERT_EQ(placed.stats.allocated, 1u);
                for (const auto& key : clean) EXPECT_TRUE(cache.OnHost(key));
                EXPECT_LT(std::ranges::count_if(dead, [&](const CacheKey& key) { return cache.OnHost(key); }),
                          dead_before);
                EXPECT_EQ(cache.coordinator.Store().GdsStats().host_watermark, 0);
            }
        }
    }
}

TEST(DiskGdsTest, BlockPlacementReclaimsDeviceCopiesBeforeCleanCopies) {
    for (bool gds : {false, true}) {
        for (bool whole : {false, true}) {
            SCOPED_TRACE(::testing::Message() << gds << ":" << whole);
            DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
            if (gds) ConfigureDiskGds(cache);
            const auto clean = cache.OnHostOnly(Prompt(1, 1));
            CopyToDisk(cache, clean);
            const Prompt duplicate_prompt(1, 2);
            const auto duplicate = cache.OnHostOnly(duplicate_prompt);
            auto device_tables = cache.OnDevice(duplicate_prompt);
            auto held = HoldFreeHost(cache);
            const std::array<std::uint32_t, 2> groups{0, 1};
            const std::array<std::int32_t, 2> buckets{0, 0};
            auto placed = whole ? cache.coordinator.Host().AcquireHostLanding(groups, buckets)
                                : cache.coordinator.Host().AcquireHostBlocks(groups, buckets);
            ASSERT_EQ(placed.stats.allocated, 2u);
            for (const auto& key : clean) EXPECT_TRUE(cache.OnHost(key));
            for (const auto& key : duplicate) EXPECT_FALSE(cache.OnHost(key));
            EXPECT_EQ(cache.coordinator.Prober().ProbePromptPrefix(duplicate_prompt.Prefix()).device.num_common_tokens,
                      4);
            EXPECT_EQ(cache.coordinator.Store().GdsStats().host_watermark, 0);
            cache.coordinator.Free(device_tables);
        }
    }
}

TEST(DiskGdsTest, BlockPlacementKeepsHostDescendantsOfADiskAncestorLive) {
    for (bool gds : {false, true}) {
        SCOPED_TRACE(gds);
        DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
        if (gds) ConfigureDiskGds(cache);
        const auto clean = cache.OnHostOnly(Prompt(1, 1));
        CopyToDisk(cache, clean);
        const auto disk_backed = cache.OnHostOnly(Prompt(2, 2));
        CopyToDisk(cache, std::span{disk_backed}.first(1));
        cache.EvictFromHost(std::span{disk_backed}.first(1));
        auto held = HoldFreeHost(cache);
        auto block = cache.coordinator.Host().AcquireHostBlock(1, 0);
        ASSERT_TRUE(block);
        for (const auto& key : std::span{disk_backed}.subspan(1)) EXPECT_TRUE(cache.OnHost(key));
        EXPECT_LT(std::ranges::count_if(clean, [&](const CacheKey& key) { return cache.OnHost(key); }), clean.size());
        EXPECT_EQ(cache.coordinator.Store().GdsStats().host_watermark, 0);
    }
}

TEST(DiskGdsTest, AFailedWholeLandingLeavesHostContentAndPolicyUnchanged) {
    for (bool gds : {false, true}) {
        SCOPED_TRACE(gds);
        DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
        if (gds) ConfigureDiskGds(cache);
        const auto keys = cache.OnHostOnly(Prompt(1, 1));
        auto held = HoldFreeHost(cache);
        const auto before = cache.coordinator.Store().GdsStats();
        const std::array<std::uint32_t, 3> groups{1, 1, 1};
        const std::array<std::int32_t, 3> buckets{0, 0, 0};
        auto landing = cache.coordinator.Host().AcquireHostLanding(groups, buckets);
        EXPECT_EQ(landing.stats.allocated, 0u);
        for (const auto& key : keys) EXPECT_TRUE(cache.coordinator.Host().ContainsHostCachedBlock(key));
        const auto after = cache.coordinator.Store().GdsStats();
        EXPECT_EQ(after.committed_plans, before.committed_plans);
        EXPECT_EQ(after.logical_evictions, before.logical_evictions);
        EXPECT_DOUBLE_EQ(after.host_watermark, before.host_watermark);
    }
}

TEST(DiskGdsTest, DiskRemovalInvalidatesRoutesEvenWhileAnIoOwnerKeepsTheBytes) {
    DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
    ConfigureDiskGds(cache);
    const Prompt prompt(3, 1);
    const auto keys = cache.OnHostOnly(prompt);
    CopyToDisk(cache, std::span{keys}.first(1));
    cache.EvictFromHost(std::span{keys}.first(1));
    auto source = cache.coordinator.Disk().AcquireDiskCachedBlock(keys[0]);
    const auto free_before = cache.disk.NumEmptyLcmBlocks();
    ASSERT_TRUE(cache.coordinator.Disk().DropUnreadableEntry(keys[0]));
    EXPECT_FALSE(cache.coordinator.Disk().ContainsDiskCachedBlock(keys[0]));
    EXPECT_EQ(cache.disk.NumEmptyLcmBlocks(), free_before);
    EXPECT_FALSE(cache.coordinator.Store().RecoveryDependencies().At(ContentHash{prompt.hashes.back()}).Available());
    EXPECT_EQ(cache.coordinator.Store().GdsStats().disk_route_losses, 1u);
    EXPECT_DOUBLE_EQ(cache.coordinator.Store().GdsStats().host_watermark, 0);
    source.reset();
    EXPECT_EQ(cache.disk.NumEmptyLcmBlocks(), free_before + 1);
}

TEST(DiskGdsTest, CleaningUsesCurrentHostScoresInsteadOfAckOrder) {
    auto limits = GdsDiskLimits();
    limits.max_blocks_per_operation = 1;
    DiskTierCache cache(FullAndState(), 32, limits);
    ConfigureDiskGds(cache);
    const auto older = cache.OnHostOnly(Prompt(4, 1));
    const auto cheaper = cache.OnHostOnly(Prompt(1, 2));
    cache.disks.QueueCleaning(older);
    cache.disks.QueueCleaning(cheaper);
    cache.disks.StartCleaning();
    const auto commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    ASSERT_EQ(commands[0].host_blocks.size(), 1u);
    auto source =
        cache.coordinator.Store().Groups()[cheaper.front().group_id].Index().Find(cache.host, cheaper.front());
    EXPECT_EQ(commands[0].host_blocks[0], cache.coordinator.Allocator(0).ResolveCacheBlockId(*source));
    EXPECT_EQ(commands[0].group_ids[0], 0u) << "history before its State";
    EXPECT_DOUBLE_EQ(cache.coordinator.Store().GdsStats().host_watermark, 0);
    cache.disks.Complete(cache::DiskDone{commands[0].op_id, true});
}

TEST(DiskGdsTest, CleaningUsesTheDirtyIndexWithoutAccumulatingAckNotifications) {
    auto limits = GdsDiskLimits();
    limits.max_queued_cleanings = 1;
    DiskTierCache cache(FullAndState(), 32, limits);
    ConfigureDiskGds(cache);
    const auto keys = cache.OnHostOnly(Prompt(1, 1));
    for (int repeat = 0; repeat < 3; ++repeat) cache.disks.QueueCleaning(keys);
    EXPECT_EQ(cache.disks.NumQueuedCleanings(), 0u);
    EXPECT_EQ(cache.disks.DroppedCleanings(), 0u);
    cache.disks.StartCleaning();
    const auto commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(commands.front().group_ids.size(), keys.size());
    cache.disks.Complete(cache::DiskDone{commands.front().op_id, true});
    for (const auto& key : keys) EXPECT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(key));
    EXPECT_EQ(cache.disks.DroppedCleanings(), 0u);
}

TEST(DiskGdsTest, DiskLossMakesHostDirtyWithoutAnotherWriteBackAck) {
    DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
    ConfigureDiskGds(cache);
    const auto keys = cache.OnHostOnly(Prompt(1, 1));
    CopyToDisk(cache, keys);
    for (const auto& key : keys) ASSERT_TRUE(cache.coordinator.Disk().DropUnreadableEntry(key));
    EXPECT_EQ(cache.disks.NumQueuedCleanings(), 0u);
    cache.disks.StartCleaning();
    auto commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(commands[0].group_ids.size(), keys.size());
    cache.disks.Complete(cache::DiskDone{commands[0].op_id, true});
    for (const auto& key : keys) EXPECT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(key));
}

TEST(DiskGdsTest, CreditRejectionDoesNotEvictHostForAnUnstartedRead) {
    auto limits = GdsDiskLimits();
    limits.max_operations = 1;
    DiskTierCache cache(FullAndState(), 32, limits);
    ConfigureDiskGds(cache);
    const Prompt disk_prompt(1, 1);
    const auto disk_keys = cache.OnHostOnly(disk_prompt);
    CopyToDisk(cache, disk_keys);
    cache.EvictFromHost(disk_keys);
    const auto host_keys = cache.OnHostOnly(Prompt(1, 2));
    cache.disks.StartCleaning();
    auto commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    auto held = HoldFreeHost(cache);
    const auto before = cache.coordinator.Store().GdsStats();
    EXPECT_FALSE(cache.disks.StartRead(disk_prompt.Prefix()));
    for (const auto& key : host_keys) EXPECT_TRUE(cache.coordinator.Host().ContainsHostCachedBlock(key));
    EXPECT_EQ(cache.coordinator.Store().GdsStats().committed_plans, before.committed_plans);
    EXPECT_DOUBLE_EQ(cache.coordinator.Store().GdsStats().host_watermark, before.host_watermark);
    cache.disks.Complete(cache::DiskDone{commands[0].op_id, true});
}

TEST(DiskGdsTest, HandoffStopsQueuedCleaningAndSuppressesAnIssuedWritesPublication) {
    for (bool issued : {false, true}) {
        SCOPED_TRACE(issued);
        DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
        ConfigureDiskGds(cache);
        const auto keys = cache.OnHostOnly(Prompt(2, 1));
        cache.disks.StartCleaning();
        std::vector<DiskOperation> commands;
        if (issued) commands = cache.disks.TakeCommands();
        ASSERT_EQ(cache.disks.NumOperations(), 1u);
        const auto epoch = CacheCoordinatorTestAccess::NextAccessEpoch(cache.coordinator);
        EXPECT_GT(cache.coordinator.DemoteExportedChain(keys, epoch), 0);
        if (issued) {
            ASSERT_EQ(commands.size(), 1u);
            EXPECT_GT(cache.coordinator.Host().NumPinnedHostCachedBlocks(), 0);
            cache.disks.Complete(cache::DiskDone{commands[0].op_id, true});
        } else {
            EXPECT_TRUE(cache.disks.TakeCommands().empty());
        }
        for (const auto& key : keys) EXPECT_FALSE(cache.coordinator.Disk().ContainsDiskCachedBlock(key));
        EXPECT_EQ(cache.coordinator.Host().NumPinnedHostCachedBlocks(), 0);
        EXPECT_EQ(cache.disks.NumOperations(), 0u);
    }
}

TEST(DiskGdsTest, NewUseCancelsHandoffBeforeCleaningAckAtTheSameWatermark) {
    DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
    ConfigureDiskGds(cache);
    const Prompt prompt(2, 1);
    const auto keys = cache.OnHostOnly(prompt);
    cache.disks.StartCleaning();
    auto commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    const auto epoch = CacheCoordinatorTestAccess::NextAccessEpoch(cache.coordinator);
    ASSERT_GT(cache.coordinator.DemoteExportedChain(keys, epoch), 0);
    auto exported = cache.coordinator.ExportListing(
        cache.coordinator.Prober().ListPrefix(cache.coordinator.Prober().ProbePromptPrefix(prompt.Prefix())));
    EXPECT_GT(exported.num_tokens, 0);
    cache.disks.Complete(cache::DiskDone{commands[0].op_id, true});
    for (const auto& key : keys) EXPECT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(key));
    EXPECT_DOUBLE_EQ(cache.coordinator.Store().GdsStats().host_watermark, 0);
}

TEST(DiskGdsTest, CleaningWaitsForPublishedPrerequisitesAndKeepsThemDuringPlacement) {
    auto limits = GdsDiskLimits();
    limits.max_blocks_per_operation = 1;
    DiskTierCache cache(FullAndState(), 1, limits);
    ConfigureDiskGds(cache);
    const auto keys = cache.OnHostOnly(Prompt(1, 1));
    cache.disks.StartCleaning();
    cache.disks.StartCleaning();
    auto commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u) << "an in-flight ancestor is not a Disk prerequisite";
    EXPECT_EQ(commands[0].group_ids, std::vector<std::uint32_t>{0});
    cache.disks.Complete(cache::DiskDone{commands[0].op_id, false});
    cache.disks.StartCleaning();
    commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(commands[0].group_ids, std::vector<std::uint32_t>{0});
    cache.disks.Complete(cache::DiskDone{commands[0].op_id, true});
    cache.disks.StartCleaning();
    EXPECT_TRUE(cache.disks.TakeCommands().empty()) << "one Disk parent cannot hold the whole closure";
    EXPECT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(keys.front()));
    EXPECT_FALSE(cache.coordinator.Disk().ContainsDiskCachedBlock(keys.back()));
    EXPECT_EQ(cache.coordinator.Host().NumPinnedHostCachedBlocks(), 0);
}

TEST(DiskGdsTest, CleaningVisitorStopsBeforeLaterCandidateExpansion) {
    DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
    // A merged HLD interval must not price KV blocks as the much larger State block.
    cache.coordinator.ConfigureEviction(PrefixCacheEvictionConfig::Gds(PrefillCost{1, 0}, {0.01, 10}));
    cache.OnHostOnly(Prompt(4, 1));
    cache.OnHostOnly(Prompt(4, 2));
    std::vector<ContentHash> candidates;
    VisitGdsHostCleaningCandidates(cache.coordinator.Store(), [&](HostCleaningCandidate candidate) {
        candidates.push_back(candidate.hash);
        return true;
    });
    ASSERT_EQ(candidates.size(), 2u);
    std::vector<ContentHash> visited;
    VisitGdsHostCleaningCandidates(cache.coordinator.Store(), [&](HostCleaningCandidate candidate) {
        visited.push_back(candidate.hash);
        return false;
    });
    EXPECT_EQ(visited, std::vector<ContentHash>{candidates.front()});
}

TEST(DiskGdsTest, AnOlderMemoryAncestorCanBeCheaperThanTheNearestDiskAncestor) {
    DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
    cache.coordinator.ConfigureEviction(PrefixCacheEvictionConfig::Gds(PrefillCost{1, 0}, {0.01, 10}));
    cache.OnHostOnly(Prompt(1, 1));
    const auto middle = cache.OnHostOnly(Prompt(2, 1));
    const Prompt deep(4, 1);
    cache.OnHostOnly(deep);
    CopyToDisk(cache, std::span{middle}.last(1));
    cache.EvictFromHost(std::span{middle}.last(1));
    std::vector<HostCleaningCandidate> candidates;
    VisitGdsHostCleaningCandidates(cache.coordinator.Store(), [&](HostCleaningCandidate candidate) {
        candidates.push_back(std::move(candidate));
        return true;
    });
    const auto found =
        std::ranges::find_if(candidates, [&](const auto& c) { return c.hash == ContentHash{deep.hashes.back()}; });
    ASSERT_NE(found, candidates.end());
    // Resume at 4 costs 12; resume at 8 costs 10 + 8. The deep point
    // exclusively retains two KV slots and its State slot.
    EXPECT_DOUBLE_EQ(found->score, 12.0 / 3);
}

TEST(DiskGdsTest, ALateFailedReadDoesNotInvalidateANewCopyOfTheSameKey) {
    DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
    ConfigureDiskGds(cache);
    const auto keys = cache.OnHostOnly(Prompt(1, 1));
    CopyToDisk(cache, keys);
    auto old_source = cache.coordinator.Disk().AcquireDiskCachedBlock(keys.front());
    ASSERT_TRUE(cache.coordinator.Disk().DropUnreadableEntry(keys.front(), old_source));
    CopyToDisk(cache, std::span{keys}.first(1));
    EXPECT_FALSE(cache.coordinator.Disk().DropUnreadableEntry(keys.front(), old_source));
    EXPECT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(keys.front()));
    EXPECT_NE(cache.coordinator.Disk().AcquireDiskCachedBlock(keys.front()), old_source);
}

TEST(DiskGdsTest, DiskOnlyTailRestoresFullHistoryStateAndTheRequiredWindow) {
    auto specs = FullAndState();
    specs.push_back({.attn_kind = AttnKind::kSlidingWindow,
                     .sliding_window = 4,
                     .cache_blocks_per_lcm_block = 2,
                     .block_granularity = 2});
    DiskTierCache cache(specs, 32, GdsDiskLimits());
    ConfigureDiskGds(cache);
    Prompt prompt(3, 1);
    prompt.tokens.resize(11);
    prompt.hashes.resize(2);
    auto prefix = prompt.Prefix();
    prefix.tail = TailPublication{
        .tail_tokens = 3, .hash = HashPrefixPage(std::span{prompt.tokens}.last(3), ContentHash{prompt.hashes.back()})};
    std::vector<BlockTable> tables(specs.size());
    ASSERT_TRUE(AdmitForTest(cache.coordinator, tables, 11));
    cache.coordinator.CacheCompletedBlocks(
        tables,
        {.completed_pages =
             CompletedPages{
                 .tokens = prompt.tokens, .prefix_hashes = prompt.hashes, .tail = prefix.tail, .publish_state = true},
         .num_computed_tokens = 11},
        CacheCoordinatorTestAccess::NextAccessEpoch(cache.coordinator));
    cache.coordinator.Free(tables);
    const auto required = cache.coordinator.Storage().SharedPrefixKeys(prefix, 64);
    ASSERT_EQ(required.max_tokens, 11);
    for (const auto& key : required.keys) {
        auto block = cache.coordinator.Host().AcquireHostBlock(key.group_id, 0);
        ASSERT_TRUE(block);
        cache.coordinator.Host().CacheHostBlock(block, key, std::nullopt);
    }
    cache.coordinator.Host().CacheHostTail(ContentHash{prompt.hashes.back()}, *prefix.tail);
    ASSERT_TRUE(cache.coordinator.ClearDeviceCache());
    CopyToDisk(cache, required.keys);
    cache.EvictFromHost(required.keys);
    const auto route = cache.coordinator.Store().RecoveryDependencies().At(prefix.tail->hash);
    EXPECT_TRUE(route.disk_recoverable);
    EXPECT_FALSE(route.ready);
    EXPECT_NEAR(route.disk_read_cost, required.keys.size() * 0.01, 1e-12);
    auto lease = cache.disks.StartRead(prefix);
    ASSERT_TRUE(lease);
    const auto commands = cache.disks.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(commands[0].group_ids.size(), required.keys.size());
    cache.disks.Complete(cache::DiskDone{lease.OperationId(), true});
    EXPECT_EQ(cache.coordinator.Prober().ProbePromptPrefix(prefix).host.num_common_tokens, 11);
    EXPECT_TRUE(cache.coordinator.Store().RecoveryDependencies().At(prefix.tail->hash).ready);
}

TEST(DiskGdsTest, CleanPlacementPricesSharedReadsBeforeUniqueReads) {
    DiskTierCache cache(FullAndState(), 32, GdsDiskLimits());
    ConfigureDiskGds(cache);
    const auto shallow = cache.OnHostOnly(Prompt(1, 1));
    const auto deep = cache.OnHostOnly(Prompt(3, 1));
    // The oldest KV belongs to both points. The deep State has one
    // consumer, with the same block cost and physical relief.
    CopyToDisk(cache, std::span{shallow}.first(1));
    CopyToDisk(cache, std::span{deep}.last(1));
    auto held = HoldFreeHost(cache);
    auto destination = cache.coordinator.Host().AcquireHostBlock(1, 0);
    ASSERT_TRUE(destination);
    EXPECT_TRUE(cache.coordinator.Host().ContainsHostCachedBlock(shallow.front()));
    EXPECT_FALSE(cache.coordinator.Host().ContainsHostCachedBlock(deep.back()));
    EXPECT_TRUE(cache.coordinator.Host().ContainsHostCachedBlock(shallow.back()));
    EXPECT_DOUBLE_EQ(cache.coordinator.Store().GdsStats().host_watermark, 0);
    EXPECT_EQ(cache.coordinator.Store().GdsStats().logical_evictions, 0u);
}

TEST(DiskGdsTest, PartialOnDiskBoundaryCleansRemainingDirtySwaAndStateKeys) {
    // Verify that under GDS, when Full History at a boundary is already on Disk
    // while State and SWA(sink>0) keys at the same boundary are still dirty on
    // Host, GDS cleaning still discovers and flushes the remaining dirty keys.
    std::vector<CacheGroupSpec> specs = FullAndState();
    specs.push_back({.attn_kind = AttnKind::kSlidingWindow,
                     .sliding_window = 8,
                     .sink_tokens = 2,
                     .cache_blocks_per_lcm_block = 1,
                     .block_granularity = kGrain});
    DiskTierCache cache(specs, 32, GdsDiskLimits());
    ConfigureDiskGds(cache);
    const Prompt prompt(3, 111);
    const auto keys = cache.OnHostOnly(prompt);

    std::vector<CacheKey> full_only;
    std::vector<CacheKey> dirty_expected;
    for (const auto& key : keys) {
        if (key.group_id == 0) {
            full_only.push_back(key);
        } else {
            dirty_expected.push_back(key);
        }
    }
    ASSERT_FALSE(full_only.empty());
    ASSERT_FALSE(dirty_expected.empty());
    CopyToDisk(cache, full_only);

    // Every boundary node already has `OnDisk() == true` via group 0, yet GDS
    // cleaning must still flush the dirty State (group 1) and SWA (group 2) keys.
    cache.disks.StartCleaning();
    const auto commands = cache.disks.TakeCommands();
    ASSERT_FALSE(commands.empty());
    for (const auto& cmd : commands) {
        cache.disks.Complete(cache::DiskDone{.op_id = cmd.op_id, .success = true});
    }
    for (const auto& key : dirty_expected) {
        EXPECT_TRUE(cache.coordinator.Disk().ContainsDiskCachedBlock(key))
            << "GDS cleaning missed dirty group " << key.group_id;
    }
}

}  // namespace
}  // namespace tokenspeed::test
