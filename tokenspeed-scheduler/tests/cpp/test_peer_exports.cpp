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
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache/tier/peer_exports.h"
#include "cache_test_access.h"
#include "unit_test_helper.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kP = 4;
constexpr std::uint32_t kFull = 0;
constexpr std::uint32_t kState = 1;

PeerConfig Config() {
    return PeerConfig{.self = "holder",
                      .max_leases = 2,
                      .soft_pin_us = 100,
                      .hard_pin_us = 1000,
                      .lookup_timeout_us = 100,
                      .min_fetch_tokens = kP};
}

std::vector<std::string> PageHashes(const std::vector<std::int32_t>& tokens) {
    std::vector<std::span<const std::int32_t>> pages;
    for (std::size_t page = 0; page + kP <= tokens.size(); page += kP) {
        pages.emplace_back(tokens.data() + page, static_cast<std::size_t>(kP));
    }
    return ComputePrefixHashes(pages, "");
}

cache::PeerLookup Lookup(std::uint64_t id, const std::vector<std::int32_t>& tokens, std::int32_t max_tokens) {
    const std::vector<std::int32_t> prefix(tokens.begin(), tokens.begin() + max_tokens);
    std::vector<std::string> hashes = PageHashes(prefix);
    return cache::PeerLookup{
        .lease = {.reader = "reader", .id = id}, .prefix_hashes = hashes, .tokens = prefix, .max_tokens = max_tokens};
}

// A holder with one KV group and one State group, the Host as large as the
// Device, under the write-back policy.
class PeerExportsTest : public ::testing::Test {
protected:
    static constexpr std::array kSpecs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = kP},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };

    // Admits a request over `tokens`, a whole number of pages, and publishes
    // them on the Device with the State group's checkpoint at the end.
    void Publish(const std::vector<std::int32_t>& tokens) {
        const auto num_tokens = static_cast<std::int32_t>(tokens.size());
        std::vector<BlockTable> tables(2);
        const std::array demands{
            GroupDemand{.table = &tables[kFull], .extent = DenseGrowth{num_tokens}},
            GroupDemand{.table = &tables[kState], .extent = StateStep{StateStep::Kind::kRotate}},
        };
        const std::optional<AdmissionResult> admitted =
            coordinator_.Admit(coordinator_.Prober().ProbePrefix({}), demands, RequestProgress{}, /*copy_in_blocks=*/{},
                               /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                               /*session_id=*/{}, /*cache_group_id=*/{});
        ASSERT_TRUE(admitted);
        const std::vector<std::string> hashes = PageHashes(tokens);
        coordinator_.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.tokens = tokens,
                                                              .prefix_hashes = hashes,
                                                              .first_new_prefix_page = 0,
                                                              .publish_state = true,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
                            .num_computed_tokens = num_tokens},
            admitted->access_epoch);
        coordinator_.Free(tables);
    }

    bool Demoted(std::uint32_t group, const std::string& hash) const {
        const auto entry = coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(group))
                               .Inspect(device_, CacheKey{.group_id = group, .content_hash = ContentHash{hash}});
        return entry && entry->metadata.demoted;
    }

    // Looks the first `tokens` up, pins and releases them, asking to demote.
    PeerAnswer ReadAndRelease(std::uint64_t id, const std::vector<std::int32_t>& tokens, bool demote) {
        const PeerAnswer looked_up =
            exports_.Lookup(coordinator_, Lookup(id, tokens, static_cast<std::int32_t>(tokens.size())), 0);
        exports_.Pin(coordinator_, cache::PeerPin{.lease = looked_up.lease}, 0);
        return exports_.Release(coordinator_, cache::PeerRelease{.lease = looked_up.lease, .demote = demote});
    }

    std::int32_t DeviceBlockId(std::uint32_t group, const std::string& hash) const {
        const CacheBlockRef block = coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(group))
                                        .Find(device_, CacheKey{.group_id = group, .content_hash = ContentHash{hash}});
        return block ? coordinator_.Allocator(static_cast<std::int32_t>(group)).ResolveCacheBlockId(*block) : -1;
    }

    BlockPool device_{32, {4, 1}};
    BlockPool host_{32, {4, 1}};
    CacheCoordinator coordinator_ =
        MakeCoordinator(kSpecs, kP, device_, /*enable_l3_storage=*/false, &host_, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true,
                        /*checkpoint_retention_requests=*/0);
    // Declared after the coordinator, so its pins go first.
    PeerExports exports_{Config()};
};

TEST_F(PeerExportsTest, ALookupAnswersTheBoundaryAndAPinExportsItsBlocksAtTheirSlots) {
    const std::vector<std::int32_t> tokens = MakeTokens(3 * kP);
    Publish({tokens.begin(), tokens.begin() + 2 * kP});
    const std::vector<std::string> hashes = PageHashes(tokens);

    const PeerAnswer looked_up = exports_.Lookup(coordinator_, Lookup(1, tokens, 3 * kP), /*now_us=*/0);
    EXPECT_EQ(looked_up.kind, PeerAnswerKind::kLookup);
    EXPECT_EQ(looked_up.num_tokens, 2 * kP);
    EXPECT_TRUE(looked_up.pages.empty());
    EXPECT_EQ(exports_.NumPinnedBlocks(), 0u) << "a soft pin holds no block";

    const PeerAnswer pinned = exports_.Pin(coordinator_, cache::PeerPin{.lease = looked_up.lease}, /*now_us=*/10);
    EXPECT_EQ(pinned.num_tokens, 2 * kP);
    EXPECT_EQ(pinned.pages,
              (std::vector<PeerPage>{
                  {.group_id = kFull, .slot = 0, .tier = CacheTier::kDevice, .block_id = DeviceBlockId(kFull, hashes[0])},
                  {.group_id = kFull, .slot = 1, .tier = CacheTier::kDevice, .block_id = DeviceBlockId(kFull, hashes[1])},
                  {.group_id = kState,
                   .slot = kPeerStateSlot,
                   .tier = CacheTier::kDevice,
                   .block_id = DeviceBlockId(kState, hashes[1])},
              }));
    EXPECT_EQ(exports_.NumPinnedBlocks(), 3u);
    EXPECT_FALSE(coordinator_.ClearCache()) << "the pin holds the blocks";

    const PeerAnswer released = exports_.Release(coordinator_, cache::PeerRelease{.lease = looked_up.lease});
    EXPECT_EQ(released.kind, PeerAnswerKind::kRelease);
    EXPECT_TRUE(released.held);
    EXPECT_EQ(exports_.NumLeases(), 0u);
    EXPECT_TRUE(coordinator_.ClearCache());
}

TEST_F(PeerExportsTest, APinNeverReachesPastTheLookedUpBoundary) {
    const std::vector<std::int32_t> tokens = MakeTokens(4 * kP);
    Publish({tokens.begin(), tokens.begin() + kP});
    const PeerAnswer looked_up = exports_.Lookup(coordinator_, Lookup(1, tokens, 4 * kP), /*now_us=*/0);
    ASSERT_EQ(looked_up.num_tokens, kP);
    // A longer prefix published after the lookup is not what the reader admitted.
    Publish({tokens.begin(), tokens.begin() + 3 * kP});

    const PeerAnswer pinned = exports_.Pin(coordinator_, cache::PeerPin{.lease = looked_up.lease}, /*now_us=*/0);
    EXPECT_EQ(pinned.num_tokens, kP);
    EXPECT_EQ(pinned.pages.size(), 2u) << "one KV page and its checkpoint";
}

TEST_F(PeerExportsTest, AnExpiredPinLetsItsBlocksGoAndItsReleaseReportsIt) {
    const std::vector<std::int32_t> tokens = MakeTokens(2 * kP);
    Publish(tokens);
    const PeerAnswer looked_up = exports_.Lookup(coordinator_, Lookup(1, tokens, 2 * kP), /*now_us=*/0);
    ASSERT_EQ(exports_.Pin(coordinator_, cache::PeerPin{.lease = looked_up.lease}, /*now_us=*/0).num_tokens, 2 * kP);

    exports_.Expire(/*now_us=*/999);
    EXPECT_EQ(exports_.NumPinnedBlocks(), 3u) << "before its deadline";
    exports_.Expire(/*now_us=*/1000);
    EXPECT_EQ(exports_.NumPinnedBlocks(), 0u);
    EXPECT_TRUE(coordinator_.ClearCache());
    EXPECT_EQ(exports_.NumLeases(), 1u) << "the tombstone";
    EXPECT_FALSE(exports_.Release(coordinator_, cache::PeerRelease{.lease = looked_up.lease}).held);
    EXPECT_EQ(exports_.NumLeases(), 0u);
}

TEST_F(PeerExportsTest, AnUnreleasedTombstoneGoesAfterAnotherPinLifetime) {
    const std::vector<std::int32_t> tokens = MakeTokens(2 * kP);
    Publish(tokens);
    const PeerAnswer looked_up = exports_.Lookup(coordinator_, Lookup(1, tokens, 2 * kP), /*now_us=*/0);
    ASSERT_GT(exports_.Pin(coordinator_, cache::PeerPin{.lease = looked_up.lease}, /*now_us=*/0).num_tokens, 0);
    exports_.Expire(/*now_us=*/1000);
    exports_.Expire(/*now_us=*/2000);
    EXPECT_EQ(exports_.NumLeases(), 0u);
}

TEST_F(PeerExportsTest, ASoftPinExpiresAndAPinAfterItMisses) {
    const std::vector<std::int32_t> tokens = MakeTokens(2 * kP);
    Publish(tokens);
    const PeerAnswer looked_up = exports_.Lookup(coordinator_, Lookup(1, tokens, 2 * kP), /*now_us=*/0);
    ASSERT_EQ(looked_up.num_tokens, 2 * kP);
    exports_.Expire(/*now_us=*/100);
    EXPECT_EQ(exports_.NumLeases(), 0u);

    const PeerAnswer pinned = exports_.Pin(coordinator_, cache::PeerPin{.lease = looked_up.lease}, /*now_us=*/100);
    EXPECT_EQ(pinned.num_tokens, 0);
    EXPECT_TRUE(pinned.pages.empty());
    EXPECT_FALSE(exports_.Release(coordinator_, cache::PeerRelease{.lease = looked_up.lease}).held);
}

TEST_F(PeerExportsTest, APinOfAPrefixEvictedSinceTheLookupMisses) {
    const std::vector<std::int32_t> tokens = MakeTokens(2 * kP);
    Publish(tokens);
    const PeerAnswer looked_up = exports_.Lookup(coordinator_, Lookup(1, tokens, 2 * kP), /*now_us=*/0);
    ASSERT_EQ(looked_up.num_tokens, 2 * kP);
    // A soft pin holds nothing: the cache may still let the prefix go.
    ASSERT_TRUE(coordinator_.ClearCache());

    EXPECT_EQ(exports_.Pin(coordinator_, cache::PeerPin{.lease = looked_up.lease}, /*now_us=*/0).num_tokens, 0);
    EXPECT_EQ(exports_.NumLeases(), 0u);
}

TEST_F(PeerExportsTest, ALookupPastTheLeaseLimitOrNotCoveringItsBoundMisses) {
    const std::vector<std::int32_t> tokens = MakeTokens(2 * kP);
    Publish(tokens);
    EXPECT_GT(exports_.Lookup(coordinator_, Lookup(1, tokens, 2 * kP), 0).num_tokens, 0);
    EXPECT_GT(exports_.Lookup(coordinator_, Lookup(2, tokens, 2 * kP), 0).num_tokens, 0);
    EXPECT_EQ(exports_.Lookup(coordinator_, Lookup(3, tokens, 2 * kP), 0).num_tokens, 0) << "two leases at most";
    EXPECT_EQ(exports_.NumLeases(), 2u);
    // A lookup repeated under a lease starts that lease over.
    EXPECT_GT(exports_.Lookup(coordinator_, Lookup(2, tokens, 2 * kP), 0).num_tokens, 0);
    EXPECT_EQ(exports_.NumLeases(), 2u);

    cache::PeerLookup short_tokens = Lookup(4, tokens, 2 * kP);
    short_tokens.tokens.pop_back();
    cache::PeerLookup missing_hash = Lookup(4, tokens, 2 * kP);
    missing_hash.prefix_hashes.pop_back();
    // Releasing a soft pin frees its lease and reports no hard pin.
    EXPECT_FALSE(exports_.Release(coordinator_, cache::PeerRelease{.lease = {.reader = "reader", .id = 1}}).held);
    EXPECT_EQ(exports_.Lookup(coordinator_, short_tokens, 0).num_tokens, 0);
    EXPECT_EQ(exports_.Lookup(coordinator_, missing_hash, 0).num_tokens, 0);
}

TEST_F(PeerExportsTest, AReleaseAfterACompleteReadDemotesTheLeafChain) {
    const std::vector<std::int32_t> tokens = MakeTokens(2 * kP);
    Publish(tokens);
    const std::vector<std::string> hashes = PageHashes(tokens);
    EXPECT_TRUE(ReadAndRelease(1, tokens, /*demote=*/true).held);
    EXPECT_TRUE(Demoted(kFull, hashes[0]));
    EXPECT_TRUE(Demoted(kFull, hashes[1]));
    EXPECT_TRUE(Demoted(kState, hashes[1]));
}

TEST_F(PeerExportsTest, TheDemotedChainStopsBelowABranch) {
    const std::vector<std::int32_t> tokens = MakeTokens(2 * kP);
    std::vector<std::int32_t> sibling = tokens;
    sibling[kP] += 1000;  // the same first page, another second one
    Publish(tokens);
    Publish(sibling);
    const std::vector<std::string> hashes = PageHashes(tokens);
    ReadAndRelease(1, tokens, /*demote=*/true);
    EXPECT_TRUE(Demoted(kFull, hashes[1]));
    EXPECT_FALSE(Demoted(kFull, hashes[0])) << "the sibling still continues it";
}

TEST_F(PeerExportsTest, NothingIsDemotedThatWasAccessedSinceThePin) {
    const std::vector<std::int32_t> tokens = MakeTokens(2 * kP);
    Publish(tokens);
    const std::vector<std::string> hashes = PageHashes(tokens);
    const PeerAnswer looked_up = exports_.Lookup(coordinator_, Lookup(1, tokens, 2 * kP), 0);
    exports_.Pin(coordinator_, cache::PeerPin{.lease = looked_up.lease}, 0);
    // Another reader's lookup refreshes the same entries.
    exports_.Lookup(coordinator_, Lookup(2, tokens, 2 * kP), 0);
    EXPECT_TRUE(exports_.Release(coordinator_, cache::PeerRelease{.lease = looked_up.lease, .demote = true}).held);
    EXPECT_FALSE(Demoted(kFull, hashes[1]));
    EXPECT_FALSE(Demoted(kState, hashes[1]));
}

TEST_F(PeerExportsTest, NoDemotionUnlessAskedOrAfterAPinThatExpired) {
    const std::vector<std::int32_t> tokens = MakeTokens(2 * kP);
    Publish(tokens);
    const std::vector<std::string> hashes = PageHashes(tokens);
    ReadAndRelease(1, tokens, /*demote=*/false);
    EXPECT_FALSE(Demoted(kFull, hashes[1]));

    const PeerAnswer looked_up = exports_.Lookup(coordinator_, Lookup(2, tokens, 2 * kP), 0);
    exports_.Pin(coordinator_, cache::PeerPin{.lease = looked_up.lease}, 0);
    exports_.Expire(/*now_us=*/1000);
    EXPECT_FALSE(exports_.Release(coordinator_, cache::PeerRelease{.lease = looked_up.lease, .demote = true}).held);
    EXPECT_FALSE(Demoted(kFull, hashes[1]));
}

// A holder with one KV group: the Device holds the prompt's first page, the
// Host its second, which a pin exports after the Device's hits.
TEST(PeerExportsHostTest, AHostSuffixIsExportedAfterTheDevicesHits) {
    static constexpr std::array kSpecs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = kP},
    };
    BlockPool device{8, {4}};
    BlockPool host{8, {4}};
    CacheCoordinator coordinator =
        MakeCoordinator(kSpecs, kP, device, /*enable_l3_storage=*/false, &host, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true,
                        /*checkpoint_retention_requests=*/0);
    PeerExports exports{Config()};
    const std::vector<std::int32_t> tokens = MakeTokens(2 * kP);
    const std::vector<std::string> hashes = PageHashes(tokens);

    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, GroupDemand{.extent = DenseGrowth{kP}}));
    const std::vector<std::int32_t> first_page(tokens.begin(), tokens.begin() + kP);
    coordinator.CacheCompletedBlocks(
        tables,
        RequestProgress{.completed_pages = CompletedPages{.tokens = first_page,
                                                          .prefix_hashes = std::span{hashes}.first(1),
                                                          .first_new_prefix_page = 0},
                        .num_computed_tokens = kP},
        /*access_epoch=*/1);
    coordinator.Free(tables);
    const std::array groups{kFull};
    const std::array buckets{0};
    CacheBlockRef host_block = std::move(coordinator.Host().AcquireHostBlocks(groups, buckets).blocks.front());
    ASSERT_TRUE(host_block);
    const std::int32_t host_block_id = coordinator.Allocator(0).ResolveCacheBlockId(*host_block);
    coordinator.Host().CacheHostBlock(host_block, CacheKey{.group_id = kFull, .content_hash = ContentHash{hashes[1]}},
                                      std::nullopt);
    host_block.reset();

    const PeerAnswer looked_up = exports.Lookup(coordinator, Lookup(1, tokens, 2 * kP), /*now_us=*/0);
    ASSERT_EQ(looked_up.num_tokens, 2 * kP);
    const PeerAnswer pinned = exports.Pin(coordinator, cache::PeerPin{.lease = looked_up.lease}, /*now_us=*/0);
    ASSERT_EQ(pinned.pages.size(), 2u);
    EXPECT_EQ(pinned.pages[0].slot, 0);
    EXPECT_EQ(pinned.pages[0].tier, CacheTier::kDevice);
    EXPECT_EQ(pinned.pages[1],
              (PeerPage{.group_id = kFull, .slot = 1, .tier = CacheTier::kHost, .block_id = host_block_id}));
    exports.Release(coordinator, cache::PeerRelease{.lease = looked_up.lease});
}

TEST(PeerConfigTest, EveryLimitMustBePositiveAndTheNameSet) {
    EXPECT_NO_THROW(Config().Validate());
    PeerConfig unnamed = Config();
    unnamed.self.clear();
    EXPECT_THROW(unnamed.Validate(), std::invalid_argument);
    PeerConfig no_leases = Config();
    no_leases.max_leases = 0;
    EXPECT_THROW(no_leases.Validate(), std::invalid_argument);
    PeerConfig no_pin = Config();
    no_pin.hard_pin_us = 0;
    EXPECT_THROW(no_pin.Validate(), std::invalid_argument);
}

}  // namespace
}  // namespace tokenspeed::test
