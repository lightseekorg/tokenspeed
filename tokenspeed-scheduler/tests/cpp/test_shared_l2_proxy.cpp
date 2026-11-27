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
#include <spdlog/sinks/stdout_sinks.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache/tier/shared_l2_proxy.h"
#include "cache/tier/transfer_manager.h"
#include "cache_test_access.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kPage = 4;
constexpr std::uint32_t kFull = 0;
constexpr std::uint32_t kState = 1;

std::vector<std::int32_t> Sequence(std::int32_t count, std::int32_t first = 1) {
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(count));
    std::iota(tokens.begin(), tokens.end(), first);
    return tokens;
}

// A prompt as a request publishes it at its end: Full KV pages, the partial
// page as a tail, and the State checkpoint after its last token.
struct Prompt {
    std::vector<std::int32_t> tokens;
    std::vector<std::string> hashes;
    std::optional<TailPublication> tail;

    explicit Prompt(std::vector<std::int32_t> input) : tokens{std::move(input)} {
        ContentHash prior;
        for (std::size_t i = 0; i + kPage <= tokens.size(); i += kPage) {
            prior = HashPrefixPage(std::span{tokens}.subspan(i, kPage), prior);
            hashes.push_back(prior.ToString());
        }
        if (tokens.size() % kPage != 0) {
            tail = TailPublication{.tail_tokens = static_cast<std::int32_t>(tokens.size() % kPage),
                                   .hash = HashPrefixPage(std::span{tokens}.subspan(hashes.size() * kPage), prior)};
        }
    }
    ContentHash End() const { return tail ? tail->hash : ContentHash{hashes.back()}; }
    // The Full group's keys, then the State checkpoint's.
    std::vector<CacheKey> Keys() const {
        std::vector<CacheKey> keys;
        for (const std::string& hash : hashes) {
            keys.push_back(CacheKey{.group_id = kFull, .content_hash = ContentHash{hash}});
        }
        if (tail) keys.push_back(CacheKey{.group_id = kFull, .content_hash = tail->hash});
        keys.push_back(CacheKey{.group_id = kState, .content_hash = End()});
        return keys;
    }
};

// A rank of a shared L2 pool: a Device tier that streams its publications to
// the store queue, and a Host tier that is the pool's proxy.
struct Rank {
    BlockPool device;
    BlockPool host;
    CacheCoordinator cache;
    TierTransferManager transfers{cache};
    SharedL2Proxy proxy{cache};

    Rank(std::int32_t device_blocks, std::int32_t host_blocks)
        : device{device_blocks, {1, 1}},
          host{host_blocks, {1, 1}},
          cache{MakeCoordinator(
              std::vector<CacheGroupSpec>{
                  CacheGroupSpec{
                      .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kPage},
                  CacheGroupSpec{
                      .attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}},
              kPage, device, /*enable_l3_storage=*/false, &host, /*disk_pool=*/nullptr, /*snapshot_pool=*/nullptr,
              /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false,
              /*checkpoint_retention_requests=*/64)} {}

    // A new request computes the prompt and publishes it at its end; returns
    // the request's access epoch.
    std::uint64_t Publish(const Prompt& prompt) {
        std::vector<BlockTable> tables(2);
        auto admission = AdmitForTest(cache, tables, static_cast<std::int32_t>(prompt.tokens.size()));
        EXPECT_TRUE(admission.has_value());
        cache.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.tokens = prompt.tokens,
                                                              .prefix_hashes = prompt.hashes,
                                                              .tail = prompt.tail,
                                                              .publish_state = true,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint,
                                                              .stream_completed_to_host = true},
                            .num_computed_tokens = static_cast<std::int32_t>(prompt.tokens.size())},
            admission->access_epoch);
        cache.Free(tables);
        return admission->access_epoch;
    }

    // The pool's answer to a lookup: the prompt's keys at pool pages 100, 101, ...
    SharedL2Answer Answer(const Prompt& prompt, std::uint64_t ticket) const {
        SharedL2Answer result{.ticket = ticket,
                              .num_tokens = static_cast<std::int32_t>(prompt.tokens.size()),
                              .kv_tokens = static_cast<std::int32_t>(prompt.tokens.size()),
                              .tail = prompt.tail,
                              .keys = prompt.Keys()};
        result.buckets.assign(result.keys.size(), 0);
        for (std::size_t i = 0; i < result.keys.size(); ++i)
            result.pool_pages.push_back(100 + static_cast<std::int32_t>(i));
        return result;
    }
    ContentHash TailPrior(const Prompt& prompt) const {
        return prompt.hashes.empty() ? ContentHash{} : ContentHash{prompt.hashes.back()};
    }
};

template <typename Operation>
std::vector<Operation> CommandsOf(const std::vector<SharedL2Operation>& commands) {
    std::vector<Operation> out;
    for (const SharedL2Operation& command : commands) {
        if (const auto* operation = std::get_if<Operation>(&command)) out.push_back(*operation);
    }
    return out;
}

TEST(SharedL2Proxy, AnAnswerBecomesProxyEntriesHeldUntilTheLookupsHoldGoes) {
    Rank rank(16, 16);
    const Prompt prompt{Sequence(10)};
    std::vector<CacheBlockRef> hold;

    const std::vector<SharedL2ProxyBlock> created =
        rank.proxy.MirrorAnswer(rank.Answer(prompt, /*ticket=*/7), rank.TailPrior(prompt), hold);
    // Each key under a Host block of its group, which stands for its pool page.
    ASSERT_EQ(created.size(), 4u);
    for (std::size_t i = 0; i < created.size(); ++i) {
        EXPECT_EQ(created[i].group_id, prompt.Keys()[i].group_id);
        EXPECT_EQ(created[i].pool_page, 100 + static_cast<std::int32_t>(i));
        EXPECT_EQ(created[i].bucket, 0);
        EXPECT_GE(created[i].host_block, 1);
    }
    EXPECT_EQ(rank.cache.Host().NumPinnedHostCachedBlocks(), 4);
    // A prompt that extends it resumes at the proxied checkpoint.
    const Prompt longer{Sequence(14)};
    const PrefixProbe probe = rank.cache.Prober().ProbePromptPrefix(
        {.prefix_hashes = longer.hashes, .tokens = longer.tokens, .max_tokens = 13});
    EXPECT_EQ(probe.host.num_common_tokens, 10);

    // Held: nothing is dropped, nothing released.
    EXPECT_TRUE(rank.proxy.NextOperations({}).empty());
    // The hold goes (an admission consumed the state): the entries leave at
    // the next pass, then the pool's ticket.
    hold.clear();
    EXPECT_EQ(rank.cache.Host().NumHostCachedBlocks(), 4);
    const std::vector<SharedL2Operation> commands = rank.proxy.NextOperations({});
    EXPECT_EQ(rank.cache.Host().NumHostCachedBlocks(), 0);
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(CommandsOf<SharedL2ReleaseOperation>(commands).front().ticket, 7u);
    EXPECT_EQ(rank.cache.Evictor().HostEvictions().released.blocks, 4u);
    // No ticket is left to release.
    EXPECT_TRUE(rank.proxy.NextOperations({}).empty());
}

TEST(SharedL2Proxy, LookupsThatShareProxyEntriesKeepTheFirstTicket) {
    Rank rank(16, 16);
    const Prompt shorter{Sequence(8)}, longer{Sequence(12)};
    std::vector<CacheBlockRef> first, second, third;
    EXPECT_EQ(rank.proxy.MirrorAnswer(rank.Answer(shorter, 1), ContentHash{}, first).size(), 3u);
    // The same pages again: nothing new, and the second ticket goes back.
    EXPECT_TRUE(rank.proxy.MirrorAnswer(rank.Answer(shorter, 2), ContentHash{}, second).empty());
    EXPECT_EQ(CommandsOf<SharedL2ReleaseOperation>(rank.proxy.NextOperations({})).front().ticket, 2u);
    // A longer hit: only its own page and checkpoint are new, the last two of
    // its four keys. Each is under a Host block that stands for the pool page
    // of its own key.
    const SharedL2Answer extended = rank.Answer(longer, 3);
    const std::vector<SharedL2ProxyBlock> created = rank.proxy.MirrorAnswer(extended, ContentHash{}, third);
    ASSERT_EQ(created.size(), 2u);
    for (std::size_t j = 0; j < created.size(); ++j) {
        const CacheKey& key = extended.keys[2 + j];
        const CacheBlockRef block = rank.cache.Store().AcquireHostCachedBlock(key);
        ASSERT_TRUE(static_cast<bool>(block));
        EXPECT_EQ(created[j].group_id, key.group_id);
        EXPECT_EQ(created[j].host_block,
                  rank.cache.Allocator(static_cast<std::int32_t>(key.group_id)).ResolveCacheBlockId(*block));
        EXPECT_EQ(created[j].pool_page, extended.pool_pages[2 + j]);
    }

    // The first ticket stays while any lookup holds its entries.
    first.clear();
    EXPECT_TRUE(rank.proxy.NextOperations({}).empty());
    second.clear();
    // The shorter prompt's checkpoint is held by nothing now and leaves; its
    // pages are the longer hit's too.
    EXPECT_TRUE(rank.proxy.NextOperations({}).empty());
    EXPECT_EQ(rank.cache.Host().NumHostCachedBlocks(), 4);
    third.clear();
    const std::vector<SharedL2ReleaseOperation> released =
        CommandsOf<SharedL2ReleaseOperation>(rank.proxy.NextOperations({}));
    ASSERT_EQ(released.size(), 2u);
    EXPECT_EQ(rank.cache.Host().NumHostCachedBlocks(), 0);
}

TEST(SharedL2Proxy, AnEntryALoadBackReadsStaysUntilTheLoadLands) {
    Rank rank(16, 16);
    const Prompt prompt{Sequence(10)};
    std::vector<CacheBlockRef> hold;
    rank.proxy.MirrorAnswer(rank.Answer(prompt, 5), rank.TailPrior(prompt), hold);
    // The load-back's ticket holds its sources as block references.
    std::vector<CacheBlockRef> load;
    for (const CacheKey& key : prompt.Keys()) load.push_back(rank.cache.Store().AcquireHostCachedBlock(key));
    hold.clear();
    for (std::int32_t plan = 0; plan < 3; ++plan) {
        EXPECT_TRUE(rank.proxy.NextOperations({}).empty());
        EXPECT_EQ(rank.cache.Host().NumHostCachedBlocks(), 4);
    }
    // A load of the tail and the checkpoint alone: the pages leave first,
    // the ticket with the last entry.
    load.erase(load.begin(), load.begin() + 2);
    EXPECT_TRUE(rank.proxy.NextOperations({}).empty());
    load.clear();
    EXPECT_EQ(CommandsOf<SharedL2ReleaseOperation>(rank.proxy.NextOperations({})).front().ticket, 5u);
    EXPECT_EQ(rank.cache.Host().NumHostCachedBlocks(), 0);
}

TEST(SharedL2Proxy, AnAnswerTheProxyTierCannotHoldIsFatal) {
    Rank rank(16, /*host_blocks=*/3);
    const Prompt prompt{Sequence(10)};
    std::vector<CacheBlockRef> hold;
    EXPECT_DEATH(
        {
            // The default logger writes to stdout; the death test reads stderr.
            spdlog::set_default_logger(spdlog::stderr_logger_mt("fatal-check-test"));
            (void)rank.proxy.MirrorAnswer(rank.Answer(prompt, 5), rank.TailPrior(prompt), hold);
        },
        "the shared L2 proxy Host tier is full");
}

TEST(SharedL2Proxy, AMalformedAnswerIsRefusedBeforeAnythingOfItIsKept) {
    Rank rank(16, 16);
    const Prompt prompt{Sequence(10)};
    std::vector<CacheBlockRef> hold;
    SharedL2Answer unpinned = rank.Answer(prompt, /*ticket=*/0);
    EXPECT_THROW(rank.proxy.MirrorAnswer(unpinned, ContentHash{}, hold), std::invalid_argument);
    SharedL2Answer ragged = rank.Answer(prompt, 5);
    ragged.pool_pages.pop_back();
    EXPECT_THROW(rank.proxy.MirrorAnswer(ragged, ContentHash{}, hold), std::invalid_argument);
    SharedL2Answer unknown_group = rank.Answer(prompt, 5);
    unknown_group.keys.back().group_id = 2;
    EXPECT_THROW(rank.proxy.MirrorAnswer(unknown_group, ContentHash{}, hold), std::invalid_argument);
    SharedL2Answer unknown_bucket = rank.Answer(prompt, 5);
    unknown_bucket.buckets.front() = 1;
    EXPECT_THROW(rank.proxy.MirrorAnswer(unknown_bucket, ContentHash{}, hold), std::invalid_argument);
    SharedL2Answer unhashed_key = rank.Answer(prompt, 5);
    unhashed_key.keys.front().content_hash = ContentHash{};
    EXPECT_THROW(rank.proxy.MirrorAnswer(unhashed_key, rank.TailPrior(prompt), hold), std::invalid_argument);
    SharedL2Answer unhashed_tail = rank.Answer(prompt, 5);
    unhashed_tail.tail->hash = ContentHash{};
    EXPECT_THROW(rank.proxy.MirrorAnswer(unhashed_tail, rank.TailPrior(prompt), hold), std::invalid_argument);
    for (const std::int32_t tail_tokens : {0, kPage}) {
        SharedL2Answer misplaced_tail = rank.Answer(prompt, 5);
        misplaced_tail.tail->tail_tokens = tail_tokens;
        EXPECT_THROW(rank.proxy.MirrorAnswer(misplaced_tail, rank.TailPrior(prompt), hold), std::invalid_argument);
    }
    EXPECT_TRUE(hold.empty());
    EXPECT_TRUE(rank.proxy.NextOperations({}).empty());
    EXPECT_EQ(rank.cache.Host().NumHostCachedBlocks(), 0);
}

TEST(SharedL2Proxy, DroppingAHostEntryLeavesAHeldOneAndCountsAMissingOneDone) {
    Rank rank(16, 16);
    const CacheKey page{.group_id = kFull, .content_hash = Prompt{Sequence(4)}.End()};
    EXPECT_TRUE(rank.cache.Host().TryEvictHostEntry(page));
    CacheBlockRef block = rank.cache.Host().AcquireHostBlock(kFull, 0);
    rank.cache.Host().CacheHostBlock(block, page, std::nullopt);
    EXPECT_FALSE(rank.cache.Host().TryEvictHostEntry(page));
    EXPECT_TRUE(rank.cache.Host().ContainsHostCachedBlock(page));
    block = {};
    EXPECT_TRUE(rank.cache.Host().TryEvictHostEntry(page));
    EXPECT_FALSE(rank.cache.Host().ContainsHostCachedBlock(page));
    EXPECT_EQ(rank.cache.Evictor().HostEvictions().released.blocks, 1u);
    EXPECT_EQ(rank.host.NumEmptyLcmBlocks(), 16);
}

TEST(SharedL2Proxy, DeviceHitsAreCollectedOnceEnabled) {
    Rank rank(16, 16);
    const Prompt prompt{Sequence(10)};
    rank.Publish(prompt);
    const Prompt longer{Sequence(14)};
    const PromptPrefix prefix{.prefix_hashes = longer.hashes, .tokens = longer.tokens, .max_tokens = 13};
    CacheCoordinatorTestAccess::MatchPromptPrefix(rank.cache, prefix);
    EXPECT_TRUE(rank.cache.TakeDeviceHits().empty());
    rank.cache.SetDeviceHitCollection(true);
    CacheCoordinatorTestAccess::MatchPromptPrefix(rank.cache, prefix);
    std::vector<CacheKey> hits = rank.cache.TakeDeviceHits();
    std::vector<CacheKey> expected = prompt.Keys();
    const auto order = [](const CacheKey& key) { return std::pair{key.group_id, key.content_hash.ToString()}; };
    std::ranges::sort(hits, {}, order);
    std::ranges::sort(expected, {}, order);
    EXPECT_EQ(hits, expected);
    EXPECT_TRUE(rank.cache.TakeDeviceHits().empty());
}

TEST(SharedL2Proxy, PoolWritesAreBoundedAndOnlyThoseInFlightPinTheirSources) {
    Rank rank(16, 16);
    const Prompt first{Sequence(10)}, second{Sequence(10, 100)};
    const std::uint64_t first_request = rank.Publish(first);
    rank.Publish(second);
    const SharedL2Config limits{.max_writes_in_flight = 2, .max_keys_per_write = 3};

    std::vector<SharedL2WriteOperation> writes = rank.transfers.StartSharedL2Writes(limits);
    ASSERT_EQ(writes.size(), 2u);
    EXPECT_EQ(writes[0].keys, (std::vector<CacheKey>{first.Keys()[0], first.Keys()[1], first.Keys()[2]}));
    EXPECT_EQ(writes[1].keys, (std::vector<CacheKey>{first.Keys()[3], second.Keys()[0], second.Keys()[1]}));
    for (const SharedL2WriteOperation& write : writes) {
        EXPECT_EQ(write.buckets, (std::vector<std::int32_t>{0, 0, 0}));
        EXPECT_EQ(write.device_blocks.size(), 3u);
    }
    // The checkpoint rides with its State block: the second write.
    EXPECT_TRUE(writes[0].checkpoints.empty());
    ASSERT_EQ(writes[1].checkpoints.size(), 1u);
    EXPECT_EQ(
        writes[1].checkpoints.front(),
        (CheckpointTree::Checkpoint{
            .tokens = first.tokens, .prefix_hashes = first.hashes, .tail = first.tail, .access_epoch = first_request}));
    EXPECT_NE(writes[0].op_id, writes[1].op_id);
    EXPECT_TRUE(rank.transfers.HasAnyInFlight());
    EXPECT_TRUE(rank.transfers.HasPinnedStoresInFlight());

    // Saturated: the rest stays queued and pins nothing, so the Device may
    // evict it -- all but the six blocks the two writes read.
    EXPECT_TRUE(rank.transfers.StartSharedL2Writes(limits).empty());
    EXPECT_TRUE(rank.cache.Host().HasPendingStores());
    EXPECT_FALSE(rank.cache.ClearDeviceCache());
    // A request that needs every block but those six.
    std::vector<BlockTable> pressure(2);
    ASSERT_TRUE(AdmitForTest(rank.cache, pressure, 8 * kPage));
    rank.cache.Free(pressure);
    for (const SharedL2WriteOperation& write : writes) {
        for (const CacheKey& key : write.keys) EXPECT_TRUE(static_cast<bool>(rank.cache.AcquireDeviceCachedBlock(key)));
    }
    EXPECT_FALSE(static_cast<bool>(rank.cache.AcquireDeviceCachedBlock(second.Keys()[3])));

    // A committed write lets its sources go; only a write in flight has a
    // completion.
    rank.transfers.CompleteSharedL2Write(cache::SharedL2WriteDone{.op_id = writes[0].op_id});
    EXPECT_THROW(rank.transfers.CompleteSharedL2Write(cache::SharedL2WriteDone{.op_id = writes[0].op_id}),
                 std::invalid_argument);
    rank.transfers.CompleteSharedL2Write(cache::SharedL2WriteDone{.op_id = writes[1].op_id});
    EXPECT_FALSE(rank.transfers.HasAnyInFlight());
    // The queue drains: what the Device evicted meanwhile is counted, not written.
    const std::vector<SharedL2WriteOperation> late = rank.transfers.StartSharedL2Writes(limits);
    const TierTransferManager::WriteBackStats& stats = rank.transfers.WriteBackStatsFor(HostStoreCause::kPublication);
    EXPECT_EQ(stats.evicted_before_drain, 2u);
    EXPECT_EQ(std::accumulate(stats.blocks.begin(), stats.blocks.end(), std::uint64_t{0}), 6u);
    EXPECT_TRUE(late.empty());
    EXPECT_FALSE(rank.cache.Host().HasPendingStores());
    EXPECT_TRUE(rank.cache.ClearDeviceCache());
}

TEST(SharedL2Proxy, PoolWritesSkipWhatTheProxyHoldsAndWhatAWriteCarries) {
    Rank rank(16, 16);
    const Prompt prompt{Sequence(10)};
    // The prompt's first page came from the pool.
    std::vector<CacheBlockRef> hold;
    SharedL2Answer answer{.ticket = 1,
                          .num_tokens = kPage,
                          .kv_tokens = kPage,
                          .keys = {prompt.Keys().front()},
                          .buckets = {0},
                          .pool_pages = {50}};
    rank.proxy.MirrorAnswer(answer, ContentHash{}, hold);
    rank.Publish(prompt);
    const SharedL2Config limits{.max_writes_in_flight = 4, .max_keys_per_write = 8};
    std::vector<SharedL2WriteOperation> writes = rank.transfers.StartSharedL2Writes(limits);
    ASSERT_EQ(writes.size(), 1u);
    EXPECT_EQ(writes[0].keys, (std::vector<CacheKey>{prompt.Keys()[1], prompt.Keys()[2], prompt.Keys()[3]}));
    const TierTransferManager::WriteBackStats& stats = rank.transfers.WriteBackStatsFor(HostStoreCause::kPublication);
    EXPECT_EQ(stats.host_cached, 1u);
    // Queued again while the write is out (a finish queues every entry).
    rank.cache.Host().QueueCachedBlocksForStore(prompt.hashes);
    EXPECT_TRUE(rank.transfers.StartSharedL2Writes(limits).empty());
    EXPECT_EQ(stats.in_flight, 1u);
    EXPECT_EQ(stats.host_cached, 2u);
    rank.transfers.CompleteSharedL2Write(cache::SharedL2WriteDone{.op_id = writes[0].op_id});
}

}  // namespace
}  // namespace tokenspeed::test
