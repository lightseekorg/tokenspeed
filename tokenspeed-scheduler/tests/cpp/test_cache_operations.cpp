#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache/tier/transfer.h"
#include "cache/tier/transfer_manager.h"
#include "cache/tier/storage_manager.h"
#include "cache_test_access.h"
#include "integration_test_helper.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"

namespace tokenspeed::test {

namespace {
CacheCoordinator::PromptPrefix StoragePrompt(std::span<const std::string> hashes, std::int32_t grain) {
    return {.prefix_hashes = hashes, .tokens = {}, .max_tokens = static_cast<std::int32_t>(hashes.size()) * grain};
}
}  // namespace

static_assert(std::is_aggregate_v<WriteBackOperation>);
static_assert(std::is_aggregate_v<LoadBackOperation>);

TEST(CacheOperationTest, WriteBackFlattensOpsInOrderWithPerOpGuard) {
    WriteBackOperation op;
    op.op_id = 7;
    op.transfers = {CacheTransfer{0, 1, 11}, CacheTransfer{0, 2, 22}};
    WriteBackOperation pinned;
    pinned.op_id = 8;
    pinned.transfers = {CacheTransfer{0, 3, 33}};
    pinned.source_pinned = true;

    WriteBackBatch batch({op, pinned});

    ASSERT_EQ(batch.op_ids, std::vector<std::uint32_t>({7, 8}));
    EXPECT_EQ(batch.group_ids[0], std::vector<std::uint32_t>({0, 0}));
    EXPECT_EQ(batch.src_pages[0], std::vector<std::int32_t>({1, 2}));
    EXPECT_EQ(batch.dst_pages[0], std::vector<std::int32_t>({11, 22}));
    EXPECT_EQ(batch.src_pages[1], std::vector<std::int32_t>({3}));
    EXPECT_EQ(batch.dst_pages[1], std::vector<std::int32_t>({33}));
    EXPECT_EQ(batch.source_pinned, std::vector<bool>({false, true}))
        << "the guard travels per op; an unset op reads as stream-ordered";
}

TEST(CacheOperationTest, RepeatedTransferWithinOnePlanIsASchedulerBug) {
    // A store skips keys already in flight and a load targets freshly acquired
    // pages, so the same (group, source, destination) cannot legitimately
    // appear twice in one plan -- neither within an op nor across ops. The
    // wire type refuses it instead of silently dropping the repeat, which
    // would hand the runtime an op it can never acknowledge.
    WriteBackOperation within;
    within.op_id = 7;
    within.transfers = {CacheTransfer{0, 1, 11}, CacheTransfer{0, 1, 11}};
    EXPECT_THROW(WriteBackBatch{{within}}, std::runtime_error);

    WriteBackOperation first;
    first.op_id = 7;
    first.transfers = {CacheTransfer{0, 2, 22}};
    WriteBackOperation second;
    second.op_id = 8;
    second.transfers = {CacheTransfer{0, 2, 22}};
    EXPECT_THROW(WriteBackBatch({first, second}), std::runtime_error);

    LoadBackOperation load;
    load.op_id = 9;
    load.transfers = {CacheTransfer{0, 10, 20}, CacheTransfer{0, 10, 20}};
    EXPECT_THROW(LoadBackBatch{{load}}, std::runtime_error);
}

TEST(CacheOperationTest, OpWithoutTransfersIsASchedulerBug) {
    WriteBackOperation store;
    store.op_id = 7;
    EXPECT_THROW(WriteBackBatch{{store}}, std::runtime_error);

    LoadBackOperation load;
    load.op_id = 9;
    EXPECT_THROW(LoadBackBatch{{load}}, std::runtime_error);
}

TEST(CacheOperationTest, SnapshotStoreMayCarryNoPageRowsButNeverADuplicate) {
    // The tail leg always rides the plan for its slot-state blob, so an op
    // with no page row is legal; a repeated (group, source, destination) is a
    // scheduler bug, within and across ops.
    SnapshotStoreOperation blob_only{.op_id = 3, .request_id = "r", .request_pool_index = 2, .snapshot_slot = 1};
    SnapshotStoreOperation pages{.op_id = 4,
                                 .request_id = "s",
                                 .request_pool_index = 3,
                                 .snapshot_slot = 2,
                                 .transfers = {CacheTransfer{0, 5, 7}, CacheTransfer{1, 5, 7}}};
    SnapshotStoreBatch batch({blob_only, pages});
    ASSERT_EQ(batch.op_ids, std::vector<std::uint32_t>({3, 4}));
    EXPECT_EQ(batch.request_ids, std::vector<std::string>({"r", "s"}));
    EXPECT_EQ(batch.request_pool_indices, std::vector<std::int32_t>({2, 3}));
    EXPECT_EQ(batch.snapshot_slots, std::vector<std::int32_t>({1, 2}));
    EXPECT_TRUE(batch.src_pages[0].empty());
    EXPECT_EQ(batch.group_ids[1], std::vector<std::uint32_t>({0, 1}));
    EXPECT_EQ(batch.src_pages[1], std::vector<std::int32_t>({5, 5}));
    EXPECT_EQ(batch.dst_pages[1], std::vector<std::int32_t>({7, 7}));

    SnapshotStoreOperation repeated = pages;
    repeated.op_id = 5;
    EXPECT_THROW(SnapshotStoreBatch({pages, repeated}), std::runtime_error);
    SnapshotStoreOperation unslotted = blob_only;
    unslotted.snapshot_slot = -1;
    EXPECT_THROW(SnapshotStoreBatch({unslotted}), std::runtime_error);
}

TEST(CacheOperationTest, SnapshotRestoreRowsNameTheirSourceTierAndCarryL2Keys) {
    SnapshotRestoreOperation op{
        .op_id = 9,
        .request_id = "r",
        .request_pool_index = 4,
        .snapshot_slot = 1,
        .transfers = {CacheTransfer{.group_id = 0, .source_page = 10, .destination_page = 20, .content_hash = "h0"},
                      CacheTransfer{.group_id = 0, .source_page = 11, .destination_page = 21}},
        .source_tier = {HostTier::kL2, HostTier::kSnapshotPool},
    };
    SnapshotRestoreBatch batch({op});
    ASSERT_EQ(batch.op_ids, std::vector<std::uint32_t>({9}));
    EXPECT_EQ(batch.request_pool_indices, std::vector<std::int32_t>({4}));
    EXPECT_EQ(batch.src_pages[0], std::vector<std::int32_t>({10, 11}));
    EXPECT_EQ(batch.dst_pages[0], std::vector<std::int32_t>({20, 21}));
    EXPECT_EQ(batch.content_hashes[0], std::vector<std::string>({"h0", ""}));
    EXPECT_EQ(batch.source_tiers[0], std::vector<std::uint8_t>({0, 1}));

    SnapshotRestoreOperation untiered = op;
    untiered.source_tier.pop_back();
    EXPECT_THROW(SnapshotRestoreBatch({untiered}), std::runtime_error) << "every row names its source tier";
    SnapshotRestoreOperation repeated = op;
    repeated.op_id = 10;
    EXPECT_THROW(SnapshotRestoreBatch({op, repeated}), std::runtime_error);
}

TEST(CacheOperationTest, EveryTierTransferPairsBlocksOfEqualResidue) {
    // Under page-cyclic sharding the rank that owns one end of a copy must own
    // the other: a Host block in another bucket than its Device source is
    // refused when the op is built.
    BlockPool device_pool{2, {2}};
    BlockPool host_pool{2, {2}};
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 2,
        .block_granularity = 2,
        .shard_count = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    TierTransferManager transfers{coordinator};
    std::vector<CacheBlockRef> device = device_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{0});
    std::vector<CacheBlockRef> same_bucket = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{0});
    std::vector<CacheBlockRef> other_bucket = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{1});
    ASSERT_EQ(device.size() + same_bucket.size() + other_bucket.size(), 3u);
    const CacheKey key{.group_id = 0, .content_hash = "h0"};
    coordinator.CacheHostBlock(same_bucket.front(), key);
    coordinator.CacheHostBlock(other_bucket.front(), CacheKey{.group_id = 0, .content_hash = "h1"});

    std::vector<BlockTransfer> ok;
    ok.push_back(
        BlockTransfer{.group_id = 0, .source = same_bucket.front(), .destination = device.front(), .key = key});
    EXPECT_NO_THROW(transfers.StartPrefixLoad(std::move(ok)));

    std::vector<BlockTransfer> crossed;
    crossed.push_back(BlockTransfer{.group_id = 0,
                                    .source = other_bucket.front(),
                                    .destination = device.front(),
                                    .key = CacheKey{.group_id = 0, .content_hash = "h1"}});
    EXPECT_THROW(transfers.StartPrefixLoad(std::move(crossed)), std::runtime_error);
}

TEST(CacheOperationTest, SamePagesInDifferentGroupsAreDistinctTransfers) {
    WriteBackOperation op;
    op.op_id = 10;
    op.transfers = {
        CacheTransfer{.group_id = 0, .source_page = 1, .destination_page = 11},
        CacheTransfer{.group_id = 1, .source_page = 1, .destination_page = 11},
    };

    WriteBackBatch batch({op});

    EXPECT_EQ(batch.group_ids[0], std::vector<std::uint32_t>({0, 1}));
    EXPECT_EQ(batch.src_pages[0], std::vector<std::int32_t>({1, 1}));
    EXPECT_EQ(batch.dst_pages[0], std::vector<std::int32_t>({11, 11}));
}

TEST(CacheOperationTest, LoadBackPreservesTransferOrder) {
    LoadBackOperation op;
    op.op_id = 9;
    op.transfers = {
        CacheTransfer{0, 10, 20},
        CacheTransfer{0, 30, 40},
    };

    LoadBackBatch batch({op});

    ASSERT_EQ(batch.op_ids, std::vector<std::uint32_t>({9}));
    EXPECT_EQ(batch.group_ids[0], std::vector<std::uint32_t>({0, 0}));
    EXPECT_EQ(batch.src_pages[0], std::vector<std::int32_t>({10, 30}));
    EXPECT_EQ(batch.dst_pages[0], std::vector<std::int32_t>({20, 40}));
}

TEST(CacheOperationTest, HostCacheAndContinuousStreamingAreSeparatePolicies) {
    SchedulerConfig config;
    config.host_allocator.total_pages = 2;

    config.role = Role::kFused;
    EXPECT_TRUE(config.HasHostCache());
    EXPECT_TRUE(config.StreamsDeviceCacheToHost());

    config.role = Role::kD;
    EXPECT_TRUE(config.HasHostCache());
    EXPECT_FALSE(config.StreamsDeviceCacheToHost());

    config.disable_l2_cache = true;
    EXPECT_FALSE(config.HasHostCache());
    EXPECT_FALSE(config.StreamsDeviceCacheToHost());
}

TEST(CacheOperationTest, DecodeCanStartWithoutHostL2) {
    const auto make_config = [] {
        SchedulerConfig config;
        config.prefix_granularity = 2;
        config.device_allocator.total_pages = 4;
        config.host_allocator.total_pages = 4;
        config.max_scheduled_tokens = 2;
        config.max_batch_size = 1;
        config.role = Role::kD;
        config.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full",
            .block_granularity = 2,
            .total_pages = 4,
            .attn_kind = AttnKind::kFull,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        SetTestSnapshotPool(config);
        return config;
    };

    SchedulerConfig disabled = make_config();
    disabled.disable_l2_cache = true;
    SetTestSnapshotPool(disabled);
    EXPECT_NO_THROW(Scheduler{std::move(disabled)});

    SchedulerConfig empty = make_config();
    empty.host_allocator.total_pages = 1;
    SetTestSnapshotPool(empty);
    EXPECT_NO_THROW(Scheduler{std::move(empty)});
}

TEST(CacheOperationTest, DeviceRequestLimitDoesNotDependOnHostCapacity) {
    const auto make_config = [](std::int32_t host_pages) {
        SchedulerConfig config;
        config.prefix_granularity = 2;
        config.device_allocator.total_pages = 9;
        config.host_allocator.total_pages = host_pages;
        config.max_scheduled_tokens = 8;
        config.max_batch_size = 2;
        config.role = Role::kD;
        config.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full",
            .block_granularity = 2,
            .total_pages = 9,
            .attn_kind = AttnKind::kFull,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        SetTestSnapshotPool(config);
        return config;
    };

    Scheduler small_host{make_config(/*host_pages=*/2)};
    Scheduler large_host{make_config(/*host_pages=*/64)};

    EXPECT_EQ(small_host.MaxSingleRequestTokens(), large_host.MaxSingleRequestTokens());
}

TEST(CacheOperationTest, StreamOrderedStorePinsNoDeviceSource) {
    BlockPool device_pool{2, {1}};
    BlockPool host_pool{1, {1}};
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false);
    TierTransferManager transfers{coordinator};

    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{2}}};
    auto admission = coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt);
    ASSERT_TRUE(admission);
    const std::array<std::string, 1> hashes{"h0"};
    coordinator.CacheFullBlocks(tables, hashes, admission->access_epoch, /*first_slot=*/0, CacheBoundaryKind::kChunk);

    coordinator.QueueCachedBlocksForStore(hashes);
    auto write_back = transfers.StartPendingStores(StoreSourceGuard::kStreamOrdered);
    ASSERT_TRUE(write_back);
    EXPECT_FALSE(write_back->source_pinned);
    EXPECT_FALSE(transfers.HasPinnedStoresInFlight());
    coordinator.Free(tables);
    // The ticket pins no Device source: the runtime orders the D2H copy on
    // the forward thread's stream ahead of any reuse, so the cache stays clearable.
    EXPECT_TRUE(coordinator.ClearDeviceCache());

    transfers.CompleteWriteBack(write_back->op_id);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(CacheKey{.group_id = 0, .content_hash = "h0"}));
}

TEST(CacheOperationTest, PinnedStoreHoldsDeviceSourceUntilAck) {
    BlockPool device_pool{1, {1}};
    BlockPool host_pool{1, {1}};
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false);
    TierTransferManager transfers{coordinator};

    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{2}}};
    auto admission = coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt);
    ASSERT_TRUE(admission);
    const std::array<std::string, 1> hashes{"h0"};
    coordinator.CacheFullBlocks(tables, hashes, admission->access_epoch, /*first_slot=*/0, CacheBoundaryKind::kChunk);

    coordinator.QueueCachedBlocksForStore(hashes);
    auto write_back = transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck);
    ASSERT_TRUE(write_back);
    EXPECT_TRUE(write_back->source_pinned);
    EXPECT_TRUE(transfers.HasPinnedStoresInFlight());
    coordinator.Free(tables);
    // The owner is gone, but the ticket still holds the Device source: it is
    // neither evictable nor clearable until the runtime acknowledges the copy.
    EXPECT_FALSE(coordinator.ClearDeviceCache());
    std::vector<BlockTable> newcomer(1);
    std::vector<GroupDemand> newcomer_demands{{.table = &newcomer[0], .extent = DenseGrowth{2}}};
    EXPECT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), newcomer_demands, RequestProgress{}, std::nullopt))
        << "the only Device block is pinned by the in-flight store";

    transfers.CompleteWriteBack(write_back->op_id);
    EXPECT_FALSE(transfers.HasPinnedStoresInFlight());
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(CacheKey{.group_id = 0, .content_hash = "h0"}));
    EXPECT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), newcomer_demands, RequestProgress{}, std::nullopt))
        << "the ACK released the pin; the block is evictable again";
}

TEST(CacheOperationTest, HostDestinationCannotBeReusedBeforeWriteBackAck) {
    BlockPool device_pool{2, {1}};
    BlockPool host_pool{1, {1}};
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false);
    TierTransferManager transfers{coordinator};
    const auto cache_device = [&](const CacheKey& key) {
        CacheBlockRef block = device_pool.AcquireBlock(key.group_id);
        ASSERT_TRUE(block);
        coordinator.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id))
            .Register(device_pool, block, key, /*access_epoch=*/1, /*logical_block_index=*/-1,
                      CacheBoundaryKind::kChunk,
                      /*newly_cached=*/nullptr);
    };

    const CacheKey first_key{.group_id = 0, .content_hash = "first"};
    cache_device(first_key);
    const std::array first_hashes{first_key.content_hash};
    coordinator.QueueCachedBlocksForStore(first_hashes);
    auto first = transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck);
    ASSERT_TRUE(first);
    ASSERT_EQ(first->transfers.size(), 1u);
    const std::int32_t destination = first->transfers.front().destination_page;

    const CacheKey second_key{.group_id = 0, .content_hash = "second"};
    cache_device(second_key);
    const std::array second_hashes{second_key.content_hash};
    coordinator.QueueCachedBlocksForStore(second_hashes);
    EXPECT_FALSE(transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck));

    transfers.CompleteWriteBack(first->op_id);
    coordinator.QueueCachedBlocksForStore(second_hashes);
    auto second = transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck);
    ASSERT_TRUE(second);
    ASSERT_EQ(second->transfers.size(), 1u);
    EXPECT_EQ(second->transfers.front().destination_page, destination);
    transfers.CompleteWriteBack(second->op_id);
}

TEST(CacheOperationTest, RetractionStoreSkipsWhenHostHasNoPlacement) {
    BlockPool device_pool{2, {1}};
    BlockPool host_pool{1, {1}};
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false);
    TierTransferManager transfers{coordinator};

    CacheBlockRef host_pin = host_pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(host_pin);
    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{2}}};
    auto admission = coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt);
    ASSERT_TRUE(admission);
    const std::array<std::string, 1> hashes{"h0"};
    coordinator.CacheFullBlocks(tables, hashes, admission->access_epoch, /*first_slot=*/0, CacheBoundaryKind::kChunk);

    coordinator.QueueCachedBlocksForStore(hashes);
    EXPECT_FALSE(transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck));
    coordinator.Free(tables);
    EXPECT_TRUE(coordinator.ClearDeviceCache());
}

TEST(CacheOperationTest, PendingStoresAllocateInCandidateOrderAcrossGroups) {
    BlockPool device_pool{3, {2, 1}};
    BlockPool host_pool{1, {2, 1}};
    const std::array specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 2,
            .block_granularity = 2,
        },
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 1,
            .block_granularity = 2,
        },
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false);
    TierTransferManager transfers{coordinator};

    const auto cache_block = [&](BlockPool& pool, const CacheKey& key) {
        GroupAllocator& allocator = coordinator.Allocator(static_cast<std::int32_t>(key.group_id));
        CacheBlockRef block = pool.AcquireBlock(key.group_id);
        EXPECT_TRUE(block);
        const std::int32_t page = allocator.ResolveCacheBlockId(block->Location());
        coordinator.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id))
            .Register(pool, block, key, /*access_epoch=*/1, /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                      /*newly_cached=*/nullptr);
        block.reset();
        return page;
    };

    cache_block(host_pool, CacheKey{.group_id = 0, .content_hash = "old-0"});
    cache_block(host_pool, CacheKey{.group_id = 0, .content_hash = "old-1"});

    const CacheKey group_one{.group_id = 1, .content_hash = "new-1"};
    const std::int32_t group_one_source = cache_block(device_pool, group_one);
    const std::array group_one_hashes{group_one.content_hash};
    coordinator.QueueCachedBlocksForStore(group_one_hashes);

    const CacheKey group_zero_first{.group_id = 0, .content_hash = "new-0"};
    const CacheKey group_zero_second{.group_id = 0, .content_hash = "new-2"};
    cache_block(device_pool, group_zero_first);
    cache_block(device_pool, group_zero_second);
    const std::array group_zero_hashes{group_zero_first.content_hash, group_zero_second.content_hash};
    coordinator.QueueCachedBlocksForStore(group_zero_hashes);

    auto write_back = transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck);

    ASSERT_TRUE(write_back);
    // The first candidate reclaims the old parent across groups. Its new
    // destination is pinned until the ACK, so the later stores cannot take it.
    ASSERT_EQ(write_back->transfers.size(), 1u);
    EXPECT_EQ(write_back->transfers[0].group_id, 1u);
    EXPECT_EQ(write_back->transfers[0].source_page, group_one_source);

    transfers.CompleteWriteBack(write_back->op_id);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(group_one));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(group_zero_first));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(group_zero_second));
}

TEST(CacheOperationTest, RetractionReleaseEstimateExcludesBlocksOwnedByAnotherRequest) {
    BlockPool device_pool{2, {1}};
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool,
                                                   /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);

    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{4}}};
    auto admission = coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt);
    ASSERT_TRUE(admission);
    const std::array<std::string, 2> hashes{"h0", "h1"};
    coordinator.CacheFullBlocks(tables, hashes, admission->access_epoch, /*first_slot=*/0, CacheBoundaryKind::kChunk);

    CacheBlockRef other_request_ref = tables[0].Blocks()[1];
    EXPECT_EQ(coordinator.NumNewlyReleasableLcmBlocks(tables), 1);
    other_request_ref.reset();
    EXPECT_EQ(coordinator.NumNewlyReleasableLcmBlocks(tables), 2);
}

TEST(CacheOperationTest, DecodeRejectsRequestWhoseMaximumExtentCannotFitDevice) {
    SchedulerConfig config;
    config.prefix_granularity = 2;
    config.device_allocator.total_pages = 4;
    config.host_allocator.total_pages = 10;
    config.max_scheduled_tokens = 8;
    config.max_batch_size = 2;
    config.role = Role::kD;
    config.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full",
        .block_granularity = 2,
        .total_pages = 4,
        .attn_kind = AttnKind::kFull,
        .transfer_policy = CacheTransferPolicy::FullSuffix,
    });
    SetTestSnapshotPool(config);
    Scheduler scheduler{std::move(config)};
    ASSERT_EQ(scheduler.MaxSingleRequestTokens(), 6);
    RequestSpec spec{
        .request_id = "too-large-for-device",
        .tokens = {1, 2, 3, 4},
        .max_new_tokens = 4,
    };

    EXPECT_THROW(scheduler.SubmitRequests({spec}), std::invalid_argument);
}

TEST(CacheOperationTest, PrefillAcceptsPromptThatFitsWithoutReservingDecodeTokens) {
    SchedulerConfig config;
    config.prefix_granularity = 2;
    config.device_allocator.total_pages = 5;
    config.host_allocator.total_pages = 10;
    config.max_scheduled_tokens = 8;
    config.max_batch_size = 2;
    config.role = Role::kP;
    config.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full",
        .block_granularity = 2,
        .total_pages = 5,
        .attn_kind = AttnKind::kFull,
        .transfer_policy = CacheTransferPolicy::FullSuffix,
    });
    SetTestSnapshotPool(config);
    Scheduler scheduler{std::move(config)};
    // Four usable blocks: three for the prompt and the decode width the P
    // reserves for its drafter, one for a tail changing hands.
    ASSERT_EQ(scheduler.MaxSingleRequestTokens(), 6);
    RequestSpec spec{
        .request_id = "prefill-only-capacity",
        .tokens = {1, 2, 3, 4, 5, 6},
        .max_new_tokens = 100,
    };

    EXPECT_NO_THROW(scheduler.SubmitRequests({spec}));
}

// A State group's slot 1 is published only at a natural checkpoint
// (publish_state) and then streams to Host like any other entry; an ordinary
// chunk end leaves it request-private.
TEST(CacheOperationTest, StateCheckpointStreamsOnlyWhenPublished) {
    for (const auto guard : {StoreSourceGuard::kPinnedUntilAck, StoreSourceGuard::kStreamOrdered}) {
        SCOPED_TRACE(static_cast<int>(guard));
        BlockPool pool(2, {1});
        BlockPool host_pool(2, {1});
        const std::array specs{CacheGroupSpec{.attn_kind = AttnKind::kState, .block_granularity = 0}};
        auto coordinator =
            MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr, true);
        TierTransferManager transfers(coordinator);
        const std::vector<std::string> hashes{"state2"};
        const CacheKey key{.group_id = 0, .content_hash = hashes[0]};
        std::vector<BlockTable> tables{BlockTable::FromBlocks({pool.AcquireBlock(0), pool.AcquireBlock(0)}, 0)};
        CacheCompletedBlocksForTest(coordinator, tables, hashes, 1, 0, 2, CacheBoundaryKind::kChunk, true,
                                    /*publish_state=*/false);
        EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, key));
        EXPECT_FALSE(transfers.StartPendingStores(guard));
        CacheCompletedBlocksForTest(coordinator, tables, hashes, 2, 0, 2, CacheBoundaryKind::kEndpoint, true,
                                    /*publish_state=*/true);
        EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, key));
        EXPECT_EQ(coordinator.GroupPrefixIndex(0).Find(pool, key), tables[0].Blocks()[1]) << "slot 1 is the checkpoint";
        const auto store = transfers.StartPendingStores(guard);
        ASSERT_TRUE(store);
        ASSERT_EQ(store->transfers.size(), 1u);
        EXPECT_EQ(store->source_pinned, guard == StoreSourceGuard::kPinnedUntilAck);
        coordinator.Free(tables);
        EXPECT_EQ(coordinator.ClearDeviceCache(), guard == StoreSourceGuard::kStreamOrdered);
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(key));
        transfers.CompleteWriteBack(store->op_id);
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key));
        EXPECT_FALSE(transfers.HasAnyInFlight());
        EXPECT_TRUE(coordinator.ClearDeviceCache());
    }
}

TEST(CacheOperationTest, HostRestoredStateCheckpointRemainsCachedAfterLoadAckAndWorkingRelease) {
    BlockPool pool(1, {1});
    BlockPool host_pool(1, {1});
    const std::array specs{CacheGroupSpec{.attn_kind = AttnKind::kState, .block_granularity = 0}};
    auto coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr, false);
    TierTransferManager transfers(coordinator);
    const std::vector<std::string> hashes{"restored-state"};
    const CacheKey key{.group_id = 0, .content_hash = hashes[0]};
    CacheBlockRef source = coordinator.AcquireHostBlock(0, /*bucket=*/0);
    ASSERT_TRUE(source);
    coordinator.CacheHostBlock(source, key);
    CacheBlockRef destination = pool.AcquireBlock(0);
    const auto location = destination->Location();
    coordinator.CacheDeviceBlock(destination, key, /*logical_block_index=*/-1, CacheBoundaryKind::kEndpoint);
    std::vector<BlockTable> tables{BlockTable::FromBlocks({std::move(destination)}, 0)};
    std::vector<BlockTransfer> pairs;
    pairs.push_back(BlockTransfer{.source = std::move(source), .destination = tables[0].Blocks()[0]});
    const auto load = transfers.StartPrefixLoad(std::move(pairs));
    coordinator.Free(tables);
    EXPECT_FALSE(coordinator.ClearDeviceCache()) << "load completion still owns the destination";
    transfers.CompleteLoadBack(load.op_id);
    EXPECT_FALSE(transfers.HasAnyInFlight());
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, key));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key));
    const auto metadata = coordinator.GroupPrefixIndex(0).MetadataFor(pool, location);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->boundary_kind, CacheBoundaryKind::kEndpoint);
    EXPECT_EQ(coordinator.ProbePrefix(hashes).device.num_common_tokens, 2);
    EXPECT_TRUE(coordinator.ClearDeviceCache());
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
}

TEST(CacheOperationTest, L3StorageRequiresHostCache) {
    SchedulerConfig config;
    config.prefix_granularity = 2;
    config.device_allocator.total_pages = 4;
    config.host_allocator.total_pages = 1;
    config.max_scheduled_tokens = 2;
    config.max_batch_size = 1;
    config.enable_l3_storage = true;
    config.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full",
        .block_granularity = 2,
        .total_pages = 4,
        .attn_kind = AttnKind::kFull,
    });
    SetTestSnapshotPool(config);
    EXPECT_THROW(Scheduler{std::move(config)}, std::invalid_argument);
}

TEST(CacheOperationTest, L3StorageAcceptsHostCache) {
    SchedulerConfig config;
    config.prefix_granularity = 2;
    config.device_allocator.total_pages = 4;
    config.host_allocator.total_pages = 8;
    config.max_scheduled_tokens = 2;
    config.max_batch_size = 1;
    config.enable_l3_storage = true;
    config.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full",
        .block_granularity = 2,
        .total_pages = 4,
        .attn_kind = AttnKind::kFull,
    });
    SetTestSnapshotPool(config);
    config.storage_config = StorageConfig{
        8, 32, 4, 32, 16, 2, 100, 100, StoragePressure{8, 1048576, 1000}, StoragePressure{4, 524288, 500}};
    EXPECT_NO_THROW(Scheduler{std::move(config)});
}

TEST(CacheOperationTest, HostHitsLoadWithoutKeys) {
    BlockPool device_pool{4, {1}};
    BlockPool host_pool{4, {1}};
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool,
                                                   /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    ASSERT_FALSE(coordinator.EnablesL3Storage());

    CacheBlockRef host_block = host_pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(host_block);
    const CacheKey key{.group_id = 0, .content_hash = "h0"};
    coordinator.CacheHostBlock(host_block, key);
    host_block.reset();

    auto probe = coordinator.ProbePrefix(std::array<std::string, 1>{"h0"});
    EXPECT_EQ(probe.host.num_common_tokens, 2);

    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{2}}};
    auto admission =
        coordinator.Admit(std::move(probe), demands, RequestProgress{}, /*request_access_epoch=*/std::nullopt);
    ASSERT_TRUE(admission);
    ASSERT_EQ(admission->load_pairs.size(), 1u);
    EXPECT_TRUE(admission->load_pairs[0].key.content_hash.empty());

    TierTransferManager transfers(coordinator);
    LoadBackOperation op = transfers.StartPrefixLoad(std::move(admission->load_pairs));
    transfers.CompleteLoadBack(op.op_id);
}

TEST(CacheOperationTest, ExpandPrefixKeysCoversGroupsAndOffsets) {
    BlockPool device_pool{4, {1}};
    BlockPool host_pool{4, {1}};
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true);
    const std::vector<CacheKey> keys = coordinator.ExpandPrefixKeys(std::array<std::string, 1>{"h0"});
    ASSERT_EQ(keys.size(), 2u);
    EXPECT_EQ(keys[0].content_hash, "h0");
    EXPECT_EQ(keys[0].page_offset, 0);
    EXPECT_EQ(keys[1].page_offset, 1);
}

TEST(CacheOperationTest, WriteBackBatchCarriesStorageKeys) {
    WriteBackOperation op;
    op.op_id = 3;
    op.transfers = {CacheTransfer{
        .group_id = 0,
        .source_page = 1,
        .destination_page = 11,
        .content_hash = "h0",
        .page_offset = 0,
    }};

    WriteBackBatch batch({op});

    ASSERT_EQ(batch.content_hashes[0], std::vector<std::string>({"h0"}));
    ASSERT_EQ(batch.page_offsets[0], std::vector<std::int32_t>({0}));
}

TEST(CacheOperationTest, AdmissionLoadPairsKeepHostPinnedAfterTableFree) {
    BlockPool device_pool{8, {1}};
    BlockPool host_pool{2, {1}};
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*enable_l3_storage=*/true, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true);
    const CacheKey key_h0{.group_id = 0, .content_hash = "h0"};
    const CacheKey key_h1{.group_id = 0, .content_hash = "h1"};
    for (const CacheKey& key : {key_h0, key_h1}) {
        CacheBlockRef host_block = coordinator.AcquireHostBlock(/*group_id=*/0, /*bucket=*/0);
        ASSERT_TRUE(host_block);
        coordinator.CacheHostBlock(host_block, key);
    }

    auto probe = coordinator.ProbePrefix(std::array<std::string, 2>{"h0", "h1"});
    ASSERT_EQ(probe.host.num_common_tokens, 4);
    std::vector<BlockTable> tables(1);
    std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{2}}};
    auto admission =
        coordinator.Admit(std::move(probe), demands, RequestProgress{}, /*request_access_epoch=*/std::nullopt);
    ASSERT_TRUE(admission);
    ASSERT_FALSE(admission->load_pairs.empty());
    EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 2);

    coordinator.Free(tables);
    EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 2)
        << "Host sources in load_pairs stay pinned after Free(tables); retry must reset admission";
    admission.reset();
    EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 0);
}

class AsyncStorageTransactionTest : public ::testing::Test {
protected:
    BlockPool device{16, {1}};
    BlockPool host{4, {1}};
    std::array<CacheGroupSpec, 1> specs{
        {CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2}}};
    CacheCoordinator coordinator{MakeCoordinator(specs, 2, device, true, &host, nullptr, true)};
    StorageTransferManager storage{coordinator,
                                   StorageConfig{4, 16, 4, 3, 32, 2, 100, 100, {8, 1048576, 1000}, {4, 524288, 500}}};
    std::vector<std::string> hashes{"h0", "h1", "h2"};
    CacheKey key(const std::string& hash) { return CacheKey{.group_id = 0, .content_hash = hash}; }
};

class AsyncStorageTailTest : public ::testing::Test {
protected:
    BlockPool device{64, {2, 2, 1}};
    BlockPool host{64, {2, 2, 1}};
    std::array<CacheGroupSpec, 3> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2, .shard_count = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 2,
                       .block_granularity = 2,
                       .shard_count = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
    CacheCoordinator coordinator{MakeCoordinator(specs, 4, device, true, &host, nullptr, true)};
    StorageTransferManager storage{coordinator,
                                   StorageConfig{4, 64, 32, 16, 64, 1, 100, 100, {8, 1048576, 1000}, {4, 524288, 500}}};
    std::vector<std::int32_t> tokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<std::string> hashes{
        HashPrefixPage(std::span{tokens}.first(4), ""),
        HashPrefixPage(std::span{tokens}.subspan(4, 4), HashPrefixPage(std::span{tokens}.first(4), ""))};

    CacheCoordinator::PromptPrefix prompt(std::int32_t bound, bool include_tail) const {
        CacheCoordinator::PromptPrefix result{
            .prefix_hashes = std::span{hashes}.first(bound / 4), .tokens = tokens, .max_tokens = bound};
        if (include_tail && bound % 4 != 0) {
            const ContentHash prior = result.prefix_hashes.empty() ? ContentHash{} : result.prefix_hashes.back();
            result.tail =
                TailPublication{.tail_tokens = bound % 4,
                                .hash = HashPrefixPage(std::span{tokens}.subspan(bound / 4 * 4, bound % 4), prior)};
        }
        return result;
    }
};

TEST_F(AsyncStorageTailTest, RetiredGetPublishesTailForOrdinaryMatching) {
    const auto prefix = prompt(11, true);
    const auto query = coordinator.MissingStorageKeys(prefix, 64);
    ASSERT_EQ(query.max_tokens, 11);
    ASSERT_EQ(coordinator.NumTailEntries(), 0u);
    const auto id = storage.StartRead(prefix, query.keys);
    ASSERT_NE(id, 0u);
    const auto commands = storage.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    const auto& op = std::get<StorageGetOperation>(commands[0]);
    // Six full-history blocks, the two SWA lookback blocks and one state.
    ASSERT_EQ(op.keys.size(), 9u);
    EXPECT_EQ(coordinator.ProbePromptPrefix(prompt(11, false)).host.num_common_tokens, 0);
    storage.Complete(cache::StorageDone{id, std::vector<std::uint8_t>(9, 1), true, std::vector<std::uint8_t>(9, 0)});
    const auto ordinary = coordinator.ProbePromptPrefix(prompt(11, false));
    ASSERT_EQ(ordinary.host.num_common_tokens, 11);
    ASSERT_TRUE(ordinary.host.tail);
    EXPECT_EQ(ordinary.host.tail->hash, prefix.tail->hash);
    EXPECT_EQ(coordinator.NumTailEntries(), 1u);
    for (std::size_t i = 0; i < op.keys.size(); ++i) {
        const auto& key = op.keys[i];
        if (key.group_id == 2 || key.content_hash != prefix.tail->hash) continue;
        const auto block = coordinator.FindHostCachedBlock(key);
        ASSERT_TRUE(block);
        EXPECT_EQ(coordinator.Allocator(key.group_id).BucketOf(block->Location()), key.page_offset % 2);
    }
    storage.Detach(id);
    EXPECT_EQ(storage.NumHeldBlocks(), 0u);
}

TEST_F(AsyncStorageTailTest, FirstPageTailNeedsNoFullPageHash) {
    const auto prefix = prompt(3, true);
    const auto query = coordinator.MissingStorageKeys(prefix, 64);
    ASSERT_EQ(query.max_tokens, 3);
    ASSERT_TRUE(prefix.prefix_hashes.empty());
    const auto id = storage.StartRead(prefix, query.keys);
    ASSERT_NE(id, 0u);
    const auto op = std::get<StorageGetOperation>(storage.TakeCommands()[0]);
    storage.Complete(cache::StorageDone{id, std::vector<std::uint8_t>(op.keys.size(), 1), true,
                                        std::vector<std::uint8_t>(op.keys.size(), 0)});
    EXPECT_EQ(coordinator.ProbePromptPrefix(prompt(3, false)).host.num_common_tokens, 3);
}

TEST_F(AsyncStorageTailTest, QueryBudgetKeepsTailGroupsTogether) {
    const auto prefix = prompt(11, true);
    // The tail needs nine objects, while the preceding aligned boundary
    // needs seven. Eight credits must not include a partial tail group set.
    const auto query = coordinator.MissingStorageKeys(prefix, 8);
    EXPECT_EQ(query.max_tokens, 8);
    EXPECT_EQ(query.keys.size(), 7u);
    EXPECT_TRUE(
        std::ranges::none_of(query.keys, [&](const auto& key) { return key.content_hash == prefix.tail->hash; }));
}

TEST_F(AsyncStorageTailTest, QueryBudgetPrioritizesTheCompleteTail) {
    const auto prefix = prompt(11, true);
    const auto query = coordinator.MissingStorageKeys(prefix, 12);
    EXPECT_EQ(query.max_tokens, 11);
    EXPECT_LE(query.keys.size(), 12u);
    const auto reservation = coordinator.ReserveStoragePrefix(prefix, query.keys, 9);
    ASSERT_TRUE(reservation.tail);
    EXPECT_EQ(reservation.tail->hash, prefix.tail->hash);
    EXPECT_EQ(reservation.blocks.size(), 9u);
}

TEST_F(AsyncStorageTailTest, HostBudgetRetriesTheImmediatelyPrecedingAlignedBoundary) {
    const auto prefix = prompt(11, true);
    const auto query = coordinator.MissingStorageKeys(prefix, 64);
    const auto reservation = coordinator.ReserveStoragePrefix(prefix, query.keys, 8);
    // T=11 needs nine references; T=8 needs seven. Subtracting a full
    // prefix granularity from 11 would incorrectly skip this boundary.
    EXPECT_EQ(reservation.blocks.size(), 7u);
    EXPECT_FALSE(reservation.tail);
    EXPECT_EQ(std::ranges::count_if(reservation.keys, [](const auto& key) { return key.group_id == 2; }), 1);
    EXPECT_TRUE(
        std::ranges::none_of(reservation.keys, [&](const auto& key) { return key.content_hash == prefix.tail->hash; }));
}

TEST_F(AsyncStorageTailTest, PartialGetCannotResumeWithoutItsExactState) {
    const auto prefix = prompt(11, true);
    const auto query = coordinator.MissingStorageKeys(prefix, 64);
    const auto id = storage.StartRead(prefix, query.keys);
    ASSERT_NE(id, 0u);
    const auto op = std::get<StorageGetOperation>(storage.TakeCommands()[0]);
    std::vector<std::uint8_t> success, failures;
    for (const auto& key : op.keys) {
        success.push_back(key.group_id != 2);
        failures.push_back(key.group_id == 2);
    }
    storage.Complete(cache::StorageDone{id, success, true, failures});
    EXPECT_EQ(coordinator.ProbePromptPrefix(prompt(11, false)).host.num_common_tokens, 0);
    EXPECT_EQ(coordinator.NumTailEntries(), 1u);
}

TEST_F(AsyncStorageTailTest, DetachedGetNeverPublishesTail) {
    const auto prefix = prompt(11, true);
    const auto query = coordinator.MissingStorageKeys(prefix, 64);
    const auto id = storage.StartRead(prefix, query.keys);
    ASSERT_NE(id, 0u);
    const auto op = std::get<StorageGetOperation>(storage.TakeCommands()[0]);
    storage.Detach(id);
    storage.Complete(cache::StorageDone{id, std::vector<std::uint8_t>(op.keys.size(), 1), true,
                                        std::vector<std::uint8_t>(op.keys.size(), 0)});
    EXPECT_EQ(coordinator.NumTailEntries(), 0u);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 0);
    EXPECT_EQ(storage.NumHeldBlocks(), 0u);
}

TEST_F(AsyncStorageTailTest, KnownTailBelowTheRequestBoundCanBeRestored) {
    const auto known = prompt(7, true);
    auto state = coordinator.AcquireHostBlock(2, 0);
    ASSERT_TRUE(state);
    coordinator.CacheHostBlock(state, {.group_id = 2, .content_hash = known.tail->hash});
    coordinator.CacheHostTail(hashes[0], *known.tail);
    state.reset();
    const auto longer = prompt(11, true);
    const auto query = coordinator.MissingStorageKeys(longer, 64);
    std::vector<CacheKey> hits;
    for (const auto& key : query.keys) {
        if ((key.group_id == 0 && key.content_hash == hashes[0]) || key.content_hash == known.tail->hash) {
            hits.push_back(key);
        }
    }
    const auto id = storage.StartRead(longer, hits);
    ASSERT_NE(id, 0u);
    const auto op = std::get<StorageGetOperation>(storage.TakeCommands()[0]);
    ASSERT_EQ(op.keys.size(), 6u);
    EXPECT_EQ(storage.NumHeldBlocks(), 6u) << "only new GET destinations hold I/O credits";
    storage.Complete(cache::StorageDone{id, std::vector<std::uint8_t>(6, 1), true, std::vector<std::uint8_t>(6, 0)});
    EXPECT_EQ(coordinator.ProbePromptPrefix(prompt(11, false)).host.num_common_tokens, 7);
}

TEST_F(AsyncStorageTailTest, DifferentTokensUnderTheSamePriorDoNotQueryTheKnownTail) {
    const auto known = prompt(7, true);
    auto state = coordinator.AcquireHostBlock(2, 0);
    coordinator.CacheHostBlock(state, {.group_id = 2, .content_hash = known.tail->hash});
    coordinator.CacheHostTail(hashes[0], *known.tail);
    tokens[6] = 100;
    const auto query = coordinator.MissingStorageKeys(prompt(7, true), 64);
    EXPECT_TRUE(
        std::ranges::none_of(query.keys, [&](const auto& key) { return key.content_hash == known.tail->hash; }));
    EXPECT_EQ(coordinator.ProbePromptPrefix(prompt(7, false)).host.num_common_tokens, 0);
}

TEST_F(AsyncStorageTailTest, UnknownRetirementDoesNotPublishTailMetadata) {
    const auto prefix = prompt(11, true);
    const auto query = coordinator.MissingStorageKeys(prefix, 64);
    const auto id = storage.StartRead(prefix, query.keys);
    ASSERT_NE(id, 0u);
    const auto op = std::get<StorageGetOperation>(storage.TakeCommands()[0]);
    storage.Complete(cache::StorageDone{id, std::vector<std::uint8_t>(op.keys.size(), 1), false,
                                        std::vector<std::uint8_t>(op.keys.size(), 0)});
    EXPECT_EQ(coordinator.NumTailEntries(), 0u);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 0);
    EXPECT_EQ(storage.NumQuarantined(), 1u);
    EXPECT_EQ(storage.NumHeldBlocks(), op.keys.size());
}

TEST_F(AsyncStorageTransactionTest, ReadReservesHostWithoutDeviceAllocation) {
    const auto id = storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    ASSERT_NE(id, 0u);
    EXPECT_EQ(storage.NumHeldBlocks(), 3u);
    EXPECT_EQ(device.NumEmptyLcmBlocks(), 16);
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 0);
    storage.TakeCommands();
    storage.Complete(cache::StorageDone{id, {1, 1, 1}, true, {0, 0, 0}});
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 6);
    storage.Detach(id);
    EXPECT_EQ(storage.NumHeldBlocks(), 0u);
}

TEST_F(AsyncStorageTransactionTest, PartialGetPublishesOnlyCompleteObjects) {
    const auto id = storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    storage.TakeCommands();
    storage.Complete(cache::StorageDone{id, {1, 0, 1}, true, {0, 1, 0}});
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 2);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(key("h1")));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key("h2")));
    storage.Detach(id);
    EXPECT_EQ(storage.NumHeldBlocks(), 0u);
}

TEST_F(AsyncStorageTransactionTest, UnknownRetirementKeepsTargetsQuarantined) {
    const auto id = storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    storage.TakeCommands();
    storage.Complete(cache::StorageDone{id, {0, 0, 0}, false, {0, 0, 0}});
    storage.Detach(id);
    EXPECT_EQ(storage.NumHeldBlocks(), 3u);
    EXPECT_EQ(storage.NumQuarantined(), 1u);
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 0);
    EXPECT_THROW(storage.ClearMetadata(), std::logic_error);
}

TEST_F(AsyncStorageTransactionTest, CancellationBeforeDispatchReleasesReservation) {
    const auto id = storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    storage.Detach(id);
    EXPECT_TRUE(storage.TakeCommands().empty());
    EXPECT_FALSE(storage.HasPending());
    EXPECT_EQ(host.NumEmptyLcmBlocks(), 4);
}

TEST_F(AsyncStorageTransactionTest, HostPublicationDoesNotProveRemoteExistence) {
    auto ref = host.AcquireBlock(0);
    coordinator.CacheHostBlock(ref, key("h0"));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key("h0")));
    EXPECT_EQ(storage.NumOperations(), 0u);
    EXPECT_TRUE(storage.TakeCommands().empty());
}

TEST_F(AsyncStorageTransactionTest, PutPinsOnlyAtDispatchAndFailureKeepsL2) {
    auto ref = host.AcquireBlock(0);
    coordinator.CacheHostBlock(ref, key("h0"));
    ref.reset();
    storage.QueueBackups(std::array{key("h0")});
    EXPECT_EQ(storage.NumHeldBlocks(), 0u);
    storage.StartBackups();
    auto commands = storage.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<StoragePutOperation>(commands[0]));
    EXPECT_EQ(storage.NumHeldBlocks(), 1u);
    storage.Complete(cache::StorageDone{std::get<StoragePutOperation>(commands[0]).op_id, {0}, true, {0}});
    EXPECT_EQ(storage.NumHeldBlocks(), 0u);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key("h0")));
}

TEST_F(AsyncStorageTransactionTest, SlowBackupsLeaveHostCreditsForAnUnrelatedRead) {
    StorageTransferManager transfers{
        coordinator, StorageConfig{4, 16, 4, 16, 32, 2, 100, 100, {8, 1048576, 1000}, {4, 524288, 500}}};
    std::vector<CacheKey> backup_keys;
    for (int i = 0; i < 4; ++i) {
        auto ref = host.AcquireBlock(0);
        auto cache_key = key("backup" + std::to_string(i));
        coordinator.CacheHostBlock(ref, cache_key);
        backup_keys.push_back(cache_key);
    }
    transfers.QueueBackups(backup_keys);
    transfers.StartBackups();
    const auto puts = transfers.TakeCommands();
    ASSERT_EQ(puts.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<StoragePutOperation>(puts[0]));
    // The PUT remains in flight while a different prefix needs Host targets.
    const auto read = transfers.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    ASSERT_NE(read, 0u);
    const auto gets = transfers.TakeCommands();
    ASSERT_EQ(gets.size(), 1u);
    EXPECT_EQ(std::get<StorageGetOperation>(gets[0]).keys.size(), 2u);
    EXPECT_EQ(transfers.NumHeldBlocks(), 4u);
    transfers.Complete(cache::StorageDone{read, {1, 1}, true, {0, 0}});
    transfers.Detach(read);
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 4);
    EXPECT_EQ(transfers.NumHeldBlocks(), std::get<StoragePutOperation>(puts[0]).keys.size());
    transfers.Complete(
        cache::StorageDone{std::get<StoragePutOperation>(puts[0]).op_id,
                           std::vector<std::uint8_t>(std::get<StoragePutOperation>(puts[0]).keys.size(), 1), true,
                           std::vector<std::uint8_t>(std::get<StoragePutOperation>(puts[0]).keys.size(), 0)});
    EXPECT_EQ(transfers.NumHeldBlocks(), 0u);
    transfers.StartBackups();
    EXPECT_TRUE(transfers.TakeCommands().empty());
    EXPECT_EQ(transfers.NumUnavailableDroppedBackups(), 2u)
        << "the unrelated read evicted the nonowning queued sources";
    EXPECT_EQ(transfers.NumBackupCandidates(), 0u);
}

TEST_F(AsyncStorageTransactionTest, TemporaryWriteCreditExhaustionRetainsResidentCandidates) {
    for (const auto& hash : hashes) {
        auto ref = host.AcquireBlock(0);
        coordinator.CacheHostBlock(ref, key(hash));
    }
    storage.QueueBackups(coordinator.ExpandPrefixKeys(hashes));
    storage.StartBackups();
    auto commands = storage.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    ASSERT_EQ(std::get<StoragePutOperation>(commands[0]).keys.size(), 2u);
    const auto first = std::get<StoragePutOperation>(commands[0]).op_id;
    for (int round = 0; round < 3; ++round) {
        storage.StartBackups();
        EXPECT_TRUE(storage.TakeCommands().empty());
        EXPECT_EQ(storage.NumBackupCandidates(), 1u);
        EXPECT_EQ(storage.NumHeldBlocks(), 2u);
    }
    storage.Complete(cache::StorageDone{first, {1, 1}, true, {0, 0}});
    storage.StartBackups();
    commands = storage.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(std::get<StoragePutOperation>(commands[0]).keys, std::vector<CacheKey>{key("h2")});
    EXPECT_EQ(storage.NumBackupCandidates(), 0u);
    storage.Complete(cache::StorageDone{std::get<StoragePutOperation>(commands[0]).op_id, {1}, true, {0}});
    EXPECT_EQ(storage.NumHeldBlocks(), 0u);
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 6);
}

TEST_F(AsyncStorageTransactionTest, ReadCreditsDelayBackupsWithoutDroppingCandidates) {
    const auto read = storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    storage.TakeCommands();
    storage.Complete(cache::StorageDone{read, {1, 1, 1}, true, {0, 0, 0}});
    storage.QueueBackups(coordinator.ExpandPrefixKeys(hashes));
    storage.StartBackups();
    auto commands = storage.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    ASSERT_EQ(std::get<StoragePutOperation>(commands[0]).keys.size(), 1u);
    storage.StartBackups();
    EXPECT_EQ(storage.NumBackupCandidates(), 2u);
    EXPECT_EQ(storage.NumHeldBlocks(), 4u);
    storage.Complete(cache::StorageDone{std::get<StoragePutOperation>(commands[0]).op_id, {1}, true, {0}});
    storage.Detach(read);
    storage.StartBackups();
    commands = storage.TakeCommands();
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(std::get<StoragePutOperation>(commands[0]).keys.size(), 2u);
    EXPECT_EQ(storage.NumBackupCandidates(), 0u);
}

TEST_F(AsyncStorageTransactionTest, BackupQueueIsBoundedAndCleanupCountsOnlyQueuedCandidates) {
    storage.QueueBackups(std::array{key("h0"), key("h1"), key("h2"), key("h3")});
    EXPECT_EQ(storage.NumBackupCandidates(), 3u);
    EXPECT_EQ(storage.NumOverflowDroppedBackups(), 1u);
    EXPECT_EQ(storage.NumHeldBlocks(), 0u);
    storage.DropBackups();
    storage.DropBackups();
    EXPECT_EQ(storage.NumCleanupDroppedBackups(), 3u);
    EXPECT_EQ(storage.NumBackupCandidates(), 0u);
    storage.QueueBackups(std::array{key("h0")});
    storage.ClearMetadata();
    EXPECT_EQ(storage.NumCleanupDroppedBackups(), 4u);
    EXPECT_EQ(storage.NumOverflowDroppedBackups(), 1u);
    EXPECT_EQ(storage.NumPressureDroppedBackups(), 0u);
}

TEST_F(AsyncStorageTransactionTest, SlowBackupsLeaveOperationCreditsForQueries) {
    StorageTransferManager transfers{
        coordinator, StorageConfig{4, 16, 16, 16, 32, 2, 100, 100, {8, 1048576, 1000}, {4, 524288, 500}}};
    for (int i = 0; i < 4; ++i) {
        auto ref = host.AcquireBlock(0);
        const auto cache_key = key("backup" + std::to_string(i));
        coordinator.CacheHostBlock(ref, cache_key);
        ref.reset();
        transfers.QueueBackups(std::array{cache_key});
        transfers.StartBackups();
        transfers.TakeCommands();
    }
    const auto query = transfers.StartQuery(coordinator.ExpandPrefixKeys(hashes));
    EXPECT_NE(query, 0u);
    EXPECT_LE(transfers.NumHeldBlocks(), 2u);
}

TEST_F(AsyncStorageTransactionTest, AOneCreditBudgetStillAllowsReads) {
    StorageTransferManager transfers{
        coordinator, StorageConfig{1, 16, 1, 16, 32, 2, 100, 100, {8, 1048576, 1000}, {4, 524288, 500}}};
    auto ref = host.AcquireBlock(0);
    coordinator.CacheHostBlock(ref, key("backup"));
    ref.reset();
    transfers.QueueBackups(std::array{key("backup")});
    transfers.StartBackups();
    EXPECT_TRUE(transfers.TakeCommands().empty());
    EXPECT_EQ(transfers.NumHeldBlocks(), 0u);
    EXPECT_EQ(transfers.NumBackupCandidates(), 0u);
    EXPECT_EQ(transfers.NumDisabledDroppedBackups(), 1u);
    const auto query = transfers.StartQuery(coordinator.ExpandPrefixKeys(hashes));
    ASSERT_NE(query, 0u);
    transfers.Complete(cache::StorageDone{query, {1, 1, 1}, true, {0, 0, 0}});
    transfers.Detach(query);
    EXPECT_NE(transfers.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes)), 0u);
}

TEST_F(AsyncStorageTransactionTest, EitherZeroWriteBudgetRejectsCandidatesImmediately) {
    for (const auto [operations, blocks] : std::array{std::pair{1, 4}, std::pair{4, 1}}) {
        StorageTransferManager transfers{
            coordinator,
            StorageConfig{operations, 16, blocks, 3, 32, 2, 100, 100, {8, 1048576, 1000}, {4, 524288, 500}}};
        transfers.QueueBackups(coordinator.ExpandPrefixKeys(hashes));
        transfers.StartBackups();
        EXPECT_EQ(transfers.NumBackupCandidates(), 0u);
        EXPECT_EQ(transfers.NumDisabledDroppedBackups(), 3u);
        EXPECT_EQ(transfers.NumHeldBlocks(), 0u);
        EXPECT_TRUE(transfers.TakeCommands().empty());
    }
}

TEST_F(AsyncStorageTransactionTest, PressureGateDropsBackupsAndRequiresAllLowWatermarks) {
    auto ref = host.AcquireBlock(0);
    coordinator.CacheHostBlock(ref, key("backup"));
    ref.reset();
    storage.QueueBackups(std::array{key("backup")});
    storage.UpdatePressure({8, 0, 0});
    EXPECT_TRUE(storage.PressureGateClosed());
    EXPECT_EQ(storage.NumBackupCandidates(), 0u);
    storage.QueueBackups(std::array{key("backup")});
    storage.StartBackups();
    EXPECT_EQ(storage.NumPressureDroppedBackups(), 2u);
    EXPECT_EQ(storage.StartQuery(coordinator.ExpandPrefixKeys(hashes)), 0u);
    EXPECT_EQ(storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes)), 0u);
    EXPECT_EQ(storage.NumHeldBlocks(), 0u);
    EXPECT_TRUE(storage.TakeCommands().empty());
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key("backup")));
    for (const auto pressure :
         std::array{StoragePressure{5, 0, 0}, StoragePressure{0, 524289, 0}, StoragePressure{0, 0, 501}}) {
        storage.UpdatePressure(pressure);
        EXPECT_TRUE(storage.PressureGateClosed());
    }
    storage.UpdatePressure({4, 524288, 500});
    EXPECT_FALSE(storage.PressureGateClosed());
    storage.StartBackups();
    EXPECT_TRUE(storage.TakeCommands().empty()) << "dropped backups must not burst on recovery";
    EXPECT_NE(storage.StartQuery(coordinator.ExpandPrefixKeys(hashes)), 0u);
}

TEST_F(AsyncStorageTransactionTest, QueuedBytesAndStalledNativeWorkCanClosePressureGate) {
    for (const auto pressure : std::array{StoragePressure{0, 1048576, 0}, StoragePressure{0, 0, 1000}}) {
        storage.UpdatePressure(pressure);
        EXPECT_TRUE(storage.PressureGateClosed());
        storage.UpdatePressure({0, 0, 0});
        EXPECT_FALSE(storage.PressureGateClosed());
    }
}

TEST_F(AsyncStorageTransactionTest, PressureDoesNotRetireActiveReadOrPreventItsPublication) {
    const auto id = storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    storage.TakeCommands();
    storage.UpdatePressure({0, 0, 1000});
    EXPECT_EQ(storage.NumHeldBlocks(), 3u);
    EXPECT_TRUE(storage.TakeCommands().empty());
    storage.Complete(cache::StorageDone{id, {1, 1, 1}, true, {0, 0, 0}});
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 6);
    EXPECT_EQ(storage.NumHeldBlocks(), 3u) << "Ready data still holds its references";
    storage.Detach(id);
    EXPECT_EQ(storage.NumHeldBlocks(), 0u);
}

TEST_F(AsyncStorageTransactionTest, PressureRecoveryCannotClearQuarantine) {
    const auto id = storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    storage.TakeCommands();
    storage.UpdatePressure({0, 0, 1000});
    storage.Complete(cache::StorageDone{id, {0, 0, 0}, false, {0, 0, 0}});
    storage.UpdatePressure({0, 0, 0});
    EXPECT_FALSE(storage.PressureGateClosed());
    EXPECT_FALSE(storage.CanStart());
    EXPECT_EQ(storage.NumHeldBlocks(), 3u);
    EXPECT_EQ(storage.NumQuarantined(), 1u);
}

TEST_F(AsyncStorageTransactionTest, HostCreditExhaustionRejectsQueryBeforeBackendWork) {
    hashes.push_back("h3");
    ASSERT_NE(storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes)), 0u);
    ASSERT_EQ(storage.NumHeldBlocks(), 4u);
    EXPECT_EQ(storage.StartQuery(std::vector<CacheKey>{key("new")}), 0u);
}

TEST_F(AsyncStorageTransactionTest, NegativeMetadataRemainsBoundedAfterRetirement) {
    for (int i = 0; i < 20; ++i) {
        std::vector<std::string> unique{std::to_string(i)};
        const auto id = storage.StartRead(StoragePrompt(unique, 2), coordinator.ExpandPrefixKeys(unique));
        storage.TakeCommands();
        storage.Complete(cache::StorageDone{id, {0}, true, {1}});
        storage.Detach(id);
    }
    EXPECT_EQ(storage.NumNegativeKeys(), 3u);
    EXPECT_FALSE(storage.HasPending());
    EXPECT_EQ(host.NumEmptyLcmBlocks(), 4);
}

TEST(CacheOperationTest, ShardedL3PartialReadUsesOrdinaryOwnerPreservingAdmission) {
    for (const std::int32_t shards : {2, 4}) {
        SCOPED_TRACE(shards);
        BlockPool device{16, {4, 1}};
        BlockPool host{16, {4, 1}};
        const std::array specs{
            CacheGroupSpec{.attn_kind = AttnKind::kFull,
                           .cache_blocks_per_lcm_block = 4,
                           .block_granularity = 2,
                           .shard_count = shards},
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        };
        auto coordinator = MakeCoordinator(specs, 4, device, true, &host, nullptr, true);
        StorageTransferManager storage{
            coordinator, StorageConfig{4, 32, 16, 16, 32, 1, 100, 100, {8, 1048576, 1000}, {4, 524288, 500}}};
        const std::array<std::string, 3> hashes{"h0", "h1", "h2"};
        const auto id = storage.StartRead(StoragePrompt(hashes, 4), coordinator.ExpandPrefixKeys(hashes));
        ASSERT_NE(id, 0u);
        const auto commands = storage.TakeCommands();
        ASSERT_EQ(commands.size(), 1u);
        const auto& get = std::get<StorageGetOperation>(commands[0]);
        ASSERT_EQ(get.keys.size(), 9u);
        EXPECT_EQ(device.NumEmptyLcmBlocks(), 16) << "L3 reads hold no Device capacity";
        std::vector<std::uint8_t> successes;
        std::vector<std::uint8_t> failures;
        for (std::size_t i = 0; i < get.keys.size(); ++i) {
            const auto& key = get.keys[i];
            const auto page = std::find(hashes.begin(), hashes.end(), key.content_hash) - hashes.begin();
            const auto bucket = key.group_id == 0 ? (page * 2 + key.page_offset) % shards : 0;
            EXPECT_EQ((get.host_blocks[i] - 1) % specs[key.group_id].shard_count, bucket);
            successes.push_back(key.content_hash != "h2");
            failures.push_back(key.content_hash == "h2");
        }
        storage.Complete(cache::StorageDone{id, successes, true, failures});
        EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 8);
        EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 6);
        std::vector<BlockTable> tables(2);
        std::vector<GroupDemand> demands{
            {.table = &tables[0], .extent = DenseGrowth{12}},
            {.table = &tables[1], .extent = DenseGrowth{12}},
        };
        auto admission = coordinator.Admit(coordinator.ProbePrefix(hashes), demands, RequestProgress{},
                                           /*request_access_epoch=*/std::nullopt);
        ASSERT_TRUE(admission);
        EXPECT_EQ(admission->host_prefix_tokens, 8);
        ASSERT_EQ(admission->load_pairs.size(), 6u);
        for (const auto& pair : admission->load_pairs) {
            const auto& allocator = coordinator.Allocator(pair.group_id);
            EXPECT_EQ(allocator.BucketOf(pair.source->Location()), allocator.BucketOf(pair.destination->Location()));
        }
        storage.Detach(id);
        TierTransferManager transfers(coordinator);
        auto load = transfers.StartPrefixLoad(std::move(admission->load_pairs));
        transfers.CompleteLoadBack(load.op_id);
        EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 0);
        coordinator.Free(tables);
    }
}

TEST(CacheOperationTest, ShardedL3ReservationRespectsBucketPressureAndExistingHostPrefix) {
    BlockPool device{4, {4}};
    BlockPool host{1, {4}};
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = 2, .shard_count = 4}};
    auto coordinator = MakeCoordinator(specs, 2, device, true, &host, nullptr, true);
    const std::array<std::string, 4> hashes{"h0", "h1", "h2", "h3"};
    auto pinned = host.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{2});
    auto warm = host.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{0});
    coordinator.CacheHostBlock(warm.front(), CacheKey{.group_id = 0, .content_hash = "h0"});
    warm.clear();
    const auto prompt = StoragePrompt(hashes, 2);
    const auto query = coordinator.MissingStorageKeys(prompt, 16);
    ASSERT_EQ(query.keys.size(), 3u);
    // Bucket 2 cannot accept h2, so only the shorter complete prefix fits.
    auto short_read = coordinator.ReserveStoragePrefix(prompt, query.keys, 4);
    ASSERT_EQ(short_read.keys.size(), 1u);
    EXPECT_EQ(short_read.keys[0].content_hash, "h1");
    EXPECT_EQ(coordinator.Allocator(0).BucketOf(short_read.blocks[0]->Location()), 1);
    short_read.blocks.clear();
    pinned.clear();
    const auto full_read = coordinator.ReserveStoragePrefix(prompt, query.keys, 4);
    ASSERT_EQ(full_read.keys.size(), 3u);
    for (std::size_t i = 0; i < full_read.keys.size(); ++i) {
        EXPECT_EQ(full_read.keys[i].content_hash, hashes[i + 1]);
        EXPECT_EQ(coordinator.Allocator(0).BucketOf(full_read.blocks[i]->Location()), static_cast<std::int32_t>(i + 1));
    }
    EXPECT_EQ(device.NumEmptyLcmBlocks(), 4);
    EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 0);
}

TEST(CacheOperationTest, StorageReservationPinsAllWarmHitsBeforeEviction) {
    BlockPool device{4, {1}};
    BlockPool host{2, {1}};
    std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2}};
    auto coordinator = MakeCoordinator(specs, 2, device, true, &host, nullptr, true);
    const CacheKey later{.group_id = 0, .content_hash = "h1"};
    const CacheKey unrelated{.group_id = 0, .content_hash = "other"};
    auto first = host.AcquireBlock(0);
    auto second = host.AcquireBlock(0);
    coordinator.CacheHostBlock(first, later);
    coordinator.CacheHostBlock(second, unrelated);
    first.reset();
    second.reset();
    const std::array hashes{std::string{"h0"}, std::string{"h1"}};
    auto reservation = coordinator.ReserveStoragePrefix(StoragePrompt(hashes, 2),
                                                        std::array{CacheKey{.group_id = 0, .content_hash = "h0"}}, 2);
    ASSERT_EQ(reservation.blocks.size(), 1u);
    EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 0);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(later));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(unrelated));
    auto reused = coordinator.AcquireHostBlock(0, 0);
    ASSERT_TRUE(reused);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(later));
}

TEST_F(AsyncStorageTransactionTest, AnIndependentPutStillProtectsAWarmGetDependency) {
    auto warm = coordinator.AcquireHostBlock(0, 0);
    coordinator.CacheHostBlock(warm, key("h0"));
    const auto location = warm->Location();
    warm.reset();
    storage.QueueBackups(std::array{key("h0")});
    storage.StartBackups();
    const auto writes = storage.TakeCommands();
    ASSERT_EQ(writes.size(), 1u);
    const auto put = std::get<StoragePutOperation>(writes[0]);
    const auto read = storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    ASSERT_NE(read, 0u);
    const auto reads = storage.TakeCommands();
    ASSERT_EQ(reads.size(), 1u);
    const auto get = std::get<StorageGetOperation>(reads[0]);
    EXPECT_EQ(storage.NumHeldBlocks(), put.keys.size() + get.keys.size());
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Evict(host, location));
    storage.Complete(cache::StorageDone{put.op_id, {1}, true, {0}});
    ASSERT_TRUE(coordinator.GroupPrefixIndex(0).Evict(host, location));
    EXPECT_EQ(storage.NumHeldBlocks(), get.keys.size());
    storage.Complete(cache::StorageDone{read, {1, 1}, true, {0, 0}});
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 0);
}

TEST_F(AsyncStorageTransactionTest, RejectedReadDoesNotPoisonObjectsAndCanRetry) {
    auto id = storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    storage.TakeCommands();
    storage.Complete(cache::StorageDone{id, {0, 0, 0}, true, {0, 0, 0}});
    EXPECT_EQ(storage.NumNegativeKeys(), 0u);
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 0);
    storage.Detach(id);
    EXPECT_EQ(storage.NumHeldBlocks(), 0u);
    const auto query = storage.StartQuery(coordinator.ExpandPrefixKeys(hashes));
    storage.TakeCommands();
    storage.Complete(cache::StorageDone{query, {1, 1, 1}, true, {0, 0, 0}});
    const auto hits = storage.QueryHits(query);
    EXPECT_EQ(hits.size(), 3u);
    storage.Detach(query);
    id = storage.StartRead(StoragePrompt(hashes, 2), hits);
    ASSERT_NE(id, 0u);
    storage.TakeCommands();
    storage.Complete(cache::StorageDone{id, {1, 1, 1}, true, {0, 0, 0}});
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 6);
    storage.Detach(id);
}

TEST_F(AsyncStorageTransactionTest, OnlyConfirmedObjectFailuresEnterNegativeCache) {
    const auto id = storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    storage.TakeCommands();
    storage.Complete(cache::StorageDone{id, {0, 0, 1}, true, {0, 1, 0}});
    EXPECT_EQ(storage.NumNegativeKeys(), 1u);
    EXPECT_FALSE(storage.IsNegative(key("h0")));
    EXPECT_TRUE(storage.IsNegative(key("h1")));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key("h2")));
    storage.Detach(id);
}

TEST_F(AsyncStorageTransactionTest, MalformedObjectFactsCannotCompleteOrReleaseRead) {
    const auto id = storage.StartRead(StoragePrompt(hashes, 2), coordinator.ExpandPrefixKeys(hashes));
    storage.TakeCommands();
    EXPECT_THROW(storage.Complete(cache::StorageDone{id, {1, 1, 1}, true, {0}}), std::invalid_argument);
    EXPECT_THROW(storage.Complete(cache::StorageDone{id, {1, 1, 1}, true, {0, 1, 0}}), std::invalid_argument);
    EXPECT_EQ(storage.NumHeldBlocks(), 3u);
    EXPECT_FALSE(storage.Completed(id));
    EXPECT_EQ(coordinator.ProbePrefix(hashes).host.num_common_tokens, 0);
    storage.Complete(cache::StorageDone{id, {1, 1, 1}, true, {0, 0, 0}});
    storage.Detach(id);
}

}  // namespace tokenspeed::test
