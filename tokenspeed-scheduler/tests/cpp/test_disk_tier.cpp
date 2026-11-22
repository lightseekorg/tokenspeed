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
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
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
    DiskTierCache(std::vector<CacheGroupSpec> specs, std::int32_t disk_parents, DiskConfig limits = Limits())
        : device{64, Packing(specs)},
          host{64, Packing(specs)},
          disk{disk_parents, Packing(specs)},
          coordinator{MakeCoordinator(specs, kGrain, device, /*enable_l3_storage=*/false, &host, &disk,
                                      /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                      /*write_back_evictions=*/true, /*checkpoint_retention_requests=*/0)},
          disks{coordinator, limits} {}

    // Publishes the prompt on the Device -- a checkpoint at its end with State
    // groups -- copies every entry to a fresh Host block, as a write-back would,
    // and empties the Device: the prompt then lives in dirty Host entries.
    std::vector<CacheKey> OnHostOnly(const Prompt& prompt) {
        std::vector<BlockTable> tables(coordinator.NumGroups());
        EXPECT_TRUE(AdmitForTest(coordinator, tables, static_cast<std::int32_t>(prompt.tokens.size())));
        CacheCompletedBlocksForTest(coordinator, tables, prompt.hashes,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, static_cast<std::int32_t>(prompt.tokens.size()),
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/true);
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
    const PeerAnswer answer = PeerExports::AnswerLookup(cache.coordinator, lookup);
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
                /*now_us=*/1);
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
    const PeerAnswer answer = PeerExports::AnswerLookup(cache.coordinator, lookup);
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

}  // namespace
}  // namespace tokenspeed::test
