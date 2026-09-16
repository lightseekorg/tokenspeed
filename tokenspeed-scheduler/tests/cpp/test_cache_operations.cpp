#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache/tier/transfer.h"
#include "cache/tier/transfer_manager.h"
#include "cache_test_access.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"

namespace tokenspeed::test {

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
            .retention = CacheGroupConfig::Retention::FullHistory,
            .family = CacheGroupFamily::History,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        return config;
    };

    SchedulerConfig disabled = make_config();
    disabled.disable_l2_cache = true;
    EXPECT_NO_THROW(Scheduler{std::move(disabled)});

    SchedulerConfig empty = make_config();
    empty.host_allocator.total_pages = 1;
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
            .retention = CacheGroupConfig::Retention::FullHistory,
            .family = CacheGroupFamily::History,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
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
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, &host_pool,
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
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, &host_pool,
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
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, &host_pool,
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
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, &host_pool,
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

TEST(CacheOperationTest, PendingStoresUseBatchHostAllocation) {
    BlockPool device_pool{3, {2, 1}};
    BlockPool host_pool{1, {2, 1}};
    const std::array specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 2,
            .block_granularity = 2,
        },
        CacheGroupSpec{
            .kind = AttnKind::kFull,
            .cache_blocks_per_lcm_block = 1,
            .block_granularity = 2,
        },
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, &host_pool,
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
    cache_block(device_pool, group_one);
    const std::array group_one_hashes{group_one.content_hash};
    coordinator.QueueCachedBlocksForStore(group_one_hashes);

    const CacheKey group_zero_first{.group_id = 0, .content_hash = "new-0"};
    const CacheKey group_zero_second{.group_id = 0, .content_hash = "new-2"};
    const std::int32_t first_source = cache_block(device_pool, group_zero_first);
    const std::int32_t second_source = cache_block(device_pool, group_zero_second);
    const std::array group_zero_hashes{group_zero_first.content_hash, group_zero_second.content_hash};
    coordinator.QueueCachedBlocksForStore(group_zero_hashes);

    auto write_back = transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck);

    ASSERT_TRUE(write_back);
    ASSERT_EQ(write_back->transfers.size(), 2u);
    EXPECT_EQ(write_back->transfers[0].group_id, 0u);
    EXPECT_EQ(write_back->transfers[0].source_page, first_source);
    EXPECT_EQ(write_back->transfers[1].group_id, 0u);
    EXPECT_EQ(write_back->transfers[1].source_page, second_source);

    transfers.CompleteWriteBack(write_back->op_id);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(group_one));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(group_zero_first));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(group_zero_second));
}

TEST(CacheOperationTest, RetractionReleaseEstimateExcludesBlocksOwnedByAnotherRequest) {
    BlockPool device_pool{2, {1}};
    const std::array specs{CacheGroupSpec{
        .kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device_pool, /*host_pool=*/nullptr,
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
        .retention = CacheGroupConfig::Retention::FullHistory,
        .family = CacheGroupFamily::History,
        .transfer_policy = CacheTransferPolicy::FullSuffix,
    });
    Scheduler scheduler{std::move(config)};
    ASSERT_EQ(scheduler.MaxSingleRequestTokens(), 6);
    RequestSpec spec{
        .request_id = "too-large-for-device",
        .tokens = {1, 2, 3, 4},
        .max_new_tokens = 4,
    };

    EXPECT_THROW(scheduler.SubmitRequests({spec}), std::invalid_argument);
}

TEST(CacheOperationTest, RetractionReleaseEstimateCountsWorkingReferencesAndSharedOwners) {
    BlockPool pool(3, {1});
    const std::array specs{CacheGroupSpec{.kind = AttnKind::kMambaState, .block_granularity = 2}};
    auto coordinator = MakeCoordinator(specs, 2, pool, nullptr, false);
    std::vector<BlockTable> tables(1);
    const std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{4}}};
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));
    const auto latest = CacheStateBoundaryForTest(coordinator, tables, std::vector<std::string>{"latest"}, 2, 1,
                                                  CacheBoundaryKind::kChunk);
    ASSERT_TRUE(latest);
    coordinator.ReclaimExpired(tables, 2);
    ASSERT_EQ(tables[0].ReclaimedPrefixBlocks(), 0);
    ASSERT_TRUE(tables[0].Blocks()[0]);
    EXPECT_EQ(tables[0].Blocks()[0].use_count(), 2u);
    EXPECT_EQ(coordinator.NumNewlyReleasableLcmBlocks(tables), 2)
        << "the input and output working blocks are both in this request's table";

    std::vector<BlockTable> other(1);
    other[0] = BlockTable::FromBlocks({coordinator.AcquireDeviceCachedBlock(latest->blocks[0].key)}, 0);
    coordinator.ReclaimExpired(other, 2);
    EXPECT_EQ(coordinator.NumNewlyReleasableLcmBlocks(tables), 1);
    coordinator.Free(other);
    EXPECT_EQ(coordinator.NumNewlyReleasableLcmBlocks(tables), 2);
    coordinator.Free(tables);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), pool.NumLcmBlocks());
}

TEST(CacheOperationTest, RetractionReleaseEstimateDoesNotReleaseASharedPackedParent) {
    BlockPool pool(2, {2});
    const std::array specs{
        CacheGroupSpec{.kind = AttnKind::kMambaState, .cache_blocks_per_lcm_block = 2, .block_granularity = 2}};
    auto coordinator = MakeCoordinator(specs, 2, pool, nullptr, false);
    std::vector<BlockTable> tables(1);
    const std::vector<GroupDemand> demands{{.table = &tables[0], .extent = DenseGrowth{6}}};
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));
    const auto latest = CacheStateBoundaryForTest(coordinator, tables, std::vector<std::string>{"latest"}, 2, 1,
                                                  CacheBoundaryKind::kChunk);
    ASSERT_TRUE(latest);
    std::vector<BlockTable> other(1);
    other[0] = BlockTable::FromBlocks({tables[0].EvictToNull(1)}, 0);
    ASSERT_EQ(tables[0].Blocks()[0]->Location().lcm_block_id, other[0].Blocks()[0]->Location().lcm_block_id);
    coordinator.ReclaimExpired(tables, 2);
    EXPECT_EQ(coordinator.NumNewlyReleasableLcmBlocks(tables), 1)
        << "only the separate working parent is released while the packed sibling has another owner";
    coordinator.Free(other);
    EXPECT_EQ(coordinator.NumNewlyReleasableLcmBlocks(tables), 2);
    coordinator.Free(tables);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), pool.NumLcmBlocks());
}

TEST(CacheOperationTest, PrefillAcceptsPromptThatFitsWithoutReservingDecodeTokens) {
    SchedulerConfig config;
    config.prefix_granularity = 2;
    config.device_allocator.total_pages = 4;
    config.host_allocator.total_pages = 10;
    config.max_scheduled_tokens = 8;
    config.max_batch_size = 2;
    config.role = Role::kP;
    config.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full",
        .block_granularity = 2,
        .total_pages = 4,
        .retention = CacheGroupConfig::Retention::FullHistory,
        .family = CacheGroupFamily::History,
        .transfer_policy = CacheTransferPolicy::FullSuffix,
    });
    Scheduler scheduler{std::move(config)};
    ASSERT_EQ(scheduler.MaxSingleRequestTokens(), 6);
    RequestSpec spec{
        .request_id = "prefill-only-capacity",
        .tokens = {1, 2, 3, 4, 5, 6},
        .max_new_tokens = 100,
    };

    EXPECT_NO_THROW(scheduler.SubmitRequests({spec}));
}

TEST(CacheOperationTest, StateEndpointStoreKeepsGuardContractAndExistingHostCopy) {
    for (const auto guard : {StoreSourceGuard::kPinnedUntilAck, StoreSourceGuard::kStreamOrdered}) {
        SCOPED_TRACE(static_cast<int>(guard));
        BlockPool pool(2, {1});
        BlockPool host_pool(2, {1});
        const std::array specs{CacheGroupSpec{.kind = AttnKind::kMambaState, .block_granularity = 2}};
        auto coordinator = MakeCoordinator(specs, 2, pool, &host_pool, true);
        TierTransferManager transfers(coordinator);
        const std::vector<std::string> hashes{"state2"};
        const CacheKey key{.group_id = 0, .content_hash = hashes[0]};
        std::vector<BlockTable> tables{BlockTable::FromBlocks({pool.AcquireBlock(0)}, 0)};
        CacheCompletedBlocksForTest(coordinator, tables, hashes, 1, 0, 2, CacheBoundaryKind::kChunk, true,
                                    std::array{2});
        EXPECT_FALSE(transfers.StartPendingStores(guard)) << "Prefill Chunk must not auto-stream";
        const auto snapshot = CacheCoordinatorTestAccess::CaptureStateSnapshot(coordinator, hashes, 2);
        ASSERT_TRUE(snapshot);
        coordinator.QueueStateSnapshotForStore(*snapshot);
        EXPECT_FALSE(transfers.StartPendingStores(guard)) << "explicit queue still requires a retained endpoint";
        const auto retained = coordinator.RetainLatestStateSnapshot(hashes, CacheBoundaryKind::kEndpoint);
        ASSERT_TRUE(retained);
        EXPECT_EQ(retained->boundary_tokens, 2);
        coordinator.QueueStateSnapshotForStore(*retained);
        const auto store = transfers.StartPendingStores(guard);
        ASSERT_TRUE(store);
        ASSERT_EQ(store->transfers.size(), 1u);
        EXPECT_EQ(store->transfers[0].group_id, 0u);
        EXPECT_EQ(store->source_pinned, guard == StoreSourceGuard::kPinnedUntilAck);
        coordinator.QueueStateSnapshotForStore(*snapshot);
        EXPECT_FALSE(transfers.StartPendingStores(guard)) << "do not duplicate an in-flight key";
        coordinator.Free(tables);
        EXPECT_EQ(coordinator.ClearDeviceCache(), guard == StoreSourceGuard::kStreamOrdered);
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(key)) << "Host publication waits for the ACK";
        transfers.CompleteWriteBack(store->op_id);
        EXPECT_FALSE(transfers.HasAnyInFlight());
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key));
        EXPECT_TRUE(coordinator.ClearDeviceCache());

        // A new ordinary Device Chunk with the same key does not erase or
        // replace an already valid Host checkpoint.
        tables[0] = BlockTable::FromBlocks({pool.AcquireBlock(0)}, 0);
        CacheCompletedBlocksForTest(coordinator, tables, hashes, 2, 0, 2, CacheBoundaryKind::kChunk, true,
                                    std::array{2});
        const auto replacement = CacheCoordinatorTestAccess::CaptureStateSnapshot(coordinator, hashes, 2);
        ASSERT_TRUE(replacement);
        EXPECT_FALSE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(coordinator, *snapshot));
        EXPECT_FALSE(coordinator.CanStoreDeviceCachedBlock(key));
        coordinator.QueueStateSnapshotForStore(*replacement);
        EXPECT_FALSE(transfers.StartPendingStores(guard));
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key));
        EXPECT_EQ(coordinator.NumHostCachedBlocks(), 1);
        coordinator.Free(tables);
    }
}

TEST(CacheOperationTest, PendingEndpointStoreRejectsReinsertedOrdinaryStateChunk) {
    BlockPool pool(1, {1});
    BlockPool host_pool(1, {1});
    const std::array specs{CacheGroupSpec{.kind = AttnKind::kMambaState, .block_granularity = 2}};
    auto coordinator = MakeCoordinator(specs, 2, pool, &host_pool, true);
    TierTransferManager transfers(coordinator);
    const std::vector<std::string> hashes{"same-key"};
    const CacheKey key{.group_id = 0, .content_hash = hashes[0]};
    std::vector<BlockTable> tables{BlockTable::FromBlocks({pool.AcquireBlock(0)}, 0)};
    const auto snapshot = CacheStateBoundaryForTest(coordinator, tables, hashes, 2, 1, CacheBoundaryKind::kEndpoint);
    ASSERT_TRUE(snapshot);
    const auto old_location = tables[0].Blocks()[0]->Location();
    coordinator.QueueStateSnapshotForStore(*snapshot);
    coordinator.Free(tables);
    // ClearDeviceCache also clears pending_stores_, which would bypass the
    // send-time metadata check. Evict only the old entry and retain its queue.
    ASSERT_EQ(coordinator.GroupPrefixIndex(0).Evict(pool, old_location), key);
    tables[0] = BlockTable::FromBlocks({pool.AcquireBlock(0)}, 0);
    const auto replacement = CacheStateBoundaryForTest(coordinator, tables, hashes, 2, 2, CacheBoundaryKind::kChunk);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(tables[0].Blocks()[0]->Location(), old_location);
    EXPECT_NE(replacement->blocks[0].generation, snapshot->blocks[0].generation);
    const auto metadata = coordinator.GroupPrefixIndex(0).MetadataFor(pool, key);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->boundary_kind, CacheBoundaryKind::kChunk);
    EXPECT_EQ(metadata->generation, replacement->blocks[0].generation);
    const auto pending = CacheCoordinatorTestAccess::PendingStores(coordinator);
    ASSERT_EQ(pending.size(), 1u) << "the old Endpoint candidate must still reach StartPendingStores";
    EXPECT_EQ(pending[0].key, key);
    EXPECT_FALSE(transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck))
        << "recheck current metadata, not the stale queue's former Endpoint kind";
    EXPECT_TRUE(CacheCoordinatorTestAccess::PendingStores(coordinator).empty());
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(key));
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 0);
    coordinator.Free(tables);
}

TEST(CacheOperationTest, LoadAckReclaimsUnusedStateButPreservesWorkingInputAndExistingHost) {
    for (const bool keep_working : {false, true}) {
        SCOPED_TRACE(keep_working);
        BlockPool pool(1, {1});
        BlockPool host_pool(1, {1});
        const std::array specs{CacheGroupSpec{.kind = AttnKind::kMambaState, .block_granularity = 2}};
        auto coordinator = MakeCoordinator(specs, 2, pool, &host_pool, true);
        TierTransferManager transfers(coordinator);
        const std::vector<std::string> hashes{"loaded-state"};
        const CacheKey key{.group_id = 0, .content_hash = hashes[0]};
        std::vector<BlockTable> tables{BlockTable::FromBlocks({pool.AcquireBlock(0)}, 0)};
        const auto location = tables[0].Blocks()[0]->Location();
        const auto snapshot = CacheStateBoundaryForTest(coordinator, tables, hashes, 2, 1, CacheBoundaryKind::kChunk);
        ASSERT_TRUE(snapshot);
        CacheBlockRef host_source = coordinator.AcquireHostBlock(0);
        ASSERT_TRUE(host_source);
        coordinator.CacheHostBlock(host_source, key);
        std::vector<BlockTransfer> pairs;
        pairs.push_back(
            BlockTransfer{.source = std::move(host_source), .destination = coordinator.AcquireDeviceCachedBlock(key)});
        const auto load = transfers.StartPrefixLoad(std::move(pairs));
        ASSERT_EQ(load.transfers.size(), 1u);
        std::vector<BlockTable> working_tables(1);
        if (keep_working) {
            working_tables[0] = BlockTable::FromBlocks({coordinator.AcquireDeviceCachedBlock(key)}, 0);

            coordinator.ReclaimExpired(working_tables, 2);
        }

        coordinator.Free(tables);
        EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, key));
        EXPECT_TRUE(transfers.HasAnyInFlight());
        transfers.CompleteLoadBack(load.op_id);
        EXPECT_FALSE(transfers.HasAnyInFlight());
        EXPECT_EQ(coordinator.GroupPrefixIndex(0).Contains(pool, key), keep_working);
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key));
        EXPECT_EQ(coordinator.NumHostCachedBlocks(), 1);
        EXPECT_TRUE(coordinator.TakePendingStores().empty()) << "ordinary Chunk starts no new L2 store";
        coordinator.Free(working_tables);
        EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, key));

        // A duplicate old ACK must not act on a new registration that reused
        // both its key and its physical destination.
        tables[0] = BlockTable::FromBlocks({pool.AcquireBlock(0)}, 0);
        ASSERT_EQ(tables[0].Blocks()[0]->Location(), location);
        const auto replacement =
            CacheStateBoundaryForTest(coordinator, tables, hashes, 2, 2, CacheBoundaryKind::kChunk);
        ASSERT_TRUE(replacement);
        ASSERT_NE(replacement->blocks[0].generation, snapshot->blocks[0].generation);
        transfers.CompleteLoadBack(load.op_id);
        EXPECT_TRUE(CacheCoordinatorTestAccess::StateSnapshotIsCurrent(coordinator, *replacement));
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key));
        coordinator.Free(tables);
    }
}

}  // namespace tokenspeed::test
