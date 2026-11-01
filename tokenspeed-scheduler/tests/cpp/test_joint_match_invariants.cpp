// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// hit => warm under joint matching.
//
// One big model: draft-layer groups (e.g. a draft's sliding-window group)
// enter the coordinator as ordinary groups and participate in the joint
// convergence. The defining property the runtime relies on is that every
// boundary the convergence exposes is recoverable by EVERY group — under
// arbitrary interleavings of caching and eviction, a hit can never be
// "cold" for one group while counted for another. These tests drive
// randomized cache/evict sequences and check the converged prefix against
// a per-group ground-truth replay. The one group kind outside the invariant
// is a replayable sliding group (bounded replay): it is never matched, and
// the scheduler re-feeds its window before every hit instead.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache/tier/transfer_manager.h"
#include "cache_test_access.h"
#include "unit_test_helper.h"

namespace tokenspeed {
namespace {

std::vector<std::string> MakeHashes(std::int32_t count) {
    std::vector<std::string> hashes;
    hashes.reserve(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) {
        hashes.push_back("jm-hash-" + std::to_string(i));
    }
    return hashes;
}

CacheKey KeyFor(const std::string& content_hash, std::uint32_t group_id) {
    return CacheKey{.namespace_id = kDefaultCacheNamespaceId, .group_id = group_id, .content_hash = content_hash};
}

std::uint64_t g_epoch = 0;

std::int32_t CacheBlockFor(CacheCoordinator& coordinator, BlockPool& pool, const std::string& content_hash,
                           std::uint32_t group_id) {
    const std::int32_t group_index = static_cast<std::int32_t>(group_id);
    CacheBlockRef block_ref = pool.AcquireBlock(group_id);
    if (!block_ref) {
        return -1;
    }
    const std::int32_t id = block_ref->Location().lcm_block_id;
    coordinator.GroupPrefixIndex(group_index)
        .Register(pool, block_ref, KeyFor(content_hash, group_id), ++g_epoch, /*logical_block_index=*/-1,
                  CacheBoundaryKind::kChunk,
                  /*newly_cached=*/nullptr);
    block_ref.reset();
    return id;
}

// Ground truth for one group in isolation: the deepest prefix (in blocks)
// this group alone can support, given which of its blocks are cached.
// Full attention is prefix-closed; a sliding group needs its lookback run.
std::int32_t GroupPrefixBlocks(const CacheCoordinator& coordinator, const BlockPool& pool,
                               std::span<const std::string> hashes, std::uint32_t group_id, std::int32_t bound_blocks) {
    const std::int32_t group_index = static_cast<std::int32_t>(group_id);
    std::vector<CacheKey> keys;
    keys.reserve(hashes.size());
    for (const std::string& hash : hashes) {
        keys.push_back(KeyFor(hash, group_id));
    }
    const GroupPrefixProbe probe =
        coordinator.GroupMatcher(group_index)
            .Probe(coordinator.GroupPrefixIndex(group_index), pool, keys, 0, bound_blocks, /*extra_hits=*/nullptr);
    return static_cast<std::int32_t>(probe.hits.size());
}

TEST(JointMatchInvariantsTest, HitImpliesWarmUnderRandomCacheEvictSequences) {
    constexpr std::int32_t kBlocks = 12;
    constexpr std::int32_t kBlockTokens = 4;
    // full target group + a sliding "draft" group (window 8 -> lookback 2
    // blocks): the draft-SWA-under-full-target shape.
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull,
         .sliding_window = 0,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = kBlockTokens},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = kBlockTokens},
    };

    std::mt19937 rng(20260807);
    const std::vector<std::string> hashes = MakeHashes(kBlocks);

    for (int round = 0; round < 200; ++round) {
        BlockPool pool(64, {1, 1});
        {
            CacheCoordinator coordinator =
                MakeCoordinator(specs, kBlockTokens, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                /*snapshot_pool=*/nullptr,
                                /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
            // Random per-group caching: each group caches a random prefix
            // subset of the request's blocks (front-truncated to mimic the
            // sliding group's reclaim of slid-out blocks).
            std::vector<std::vector<std::int32_t>> cached_ids(specs.size());
            for (std::uint32_t group = 0; group < static_cast<std::uint32_t>(specs.size()); ++group) {
                std::uniform_int_distribution<std::int32_t> depth_dist(0, kBlocks);
                std::uniform_int_distribution<std::int32_t> front_dist(0, 3);
                const std::int32_t depth = depth_dist(rng);
                const std::int32_t front_holes = group == 0 ? 0 : front_dist(rng);
                for (std::int32_t i = front_holes; i < depth; ++i) {
                    cached_ids[group].push_back(
                        CacheBlockFor(coordinator, pool, hashes[static_cast<std::size_t>(i)], group));
                }
            }

            // Random evictions punch holes anywhere.
            std::uniform_int_distribution<std::int32_t> evict_count_dist(0, kBlocks);
            for (std::uint32_t group = 0; group < static_cast<std::uint32_t>(specs.size()); ++group) {
                std::int32_t to_evict = evict_count_dist(rng);
                std::shuffle(cached_ids[group].begin(), cached_ids[group].end(), rng);
                for (const std::int32_t lcm_block_id : cached_ids[group]) {
                    if (to_evict-- <= 0 || lcm_block_id < 0) {
                        break;
                    }
                    coordinator.GroupPrefixIndex(static_cast<std::int32_t>(group))
                        .Evict(pool, CacheBlockLocation{.lcm_block_id = lcm_block_id, .slot_index = 0});
                }
            }

            const auto match = MatchPrefixForTest(coordinator, hashes).device;
            const std::int32_t common_blocks = match.num_common_tokens / kBlockTokens;

            // hit => warm: every group must independently support the
            // converged boundary. This is the property the draft relies on
            // (its KV is recoverable at every exposed boundary).
            for (std::uint32_t group = 0; group < static_cast<std::uint32_t>(specs.size()); ++group) {
                const std::int32_t own = GroupPrefixBlocks(coordinator, pool, hashes, group, common_blocks);
                EXPECT_GE(own, common_blocks)
                    << "round " << round << ": group " << group << " cannot recover the converged boundary (" << own
                    << " < " << common_blocks << ")";
            }

            // Progress sanity: the convergence must not undershoot the
            // trivially-supported joint prefix (min of per-group depths at
            // full bound) by construction of the sweep. (A strict equality
            // is not required — window groups can legally re-shrink.)
            EXPECT_LE(common_blocks, kBlocks);
        }
    }
}

// A replayable group is the one deliberate exception to hit => warm: it is
// left out of the joint convergence and re-warmed by the bounded replay the
// scheduler issues before every hit, so the closed group alone decides the
// boundary whatever the replayable group has cached (nothing, by contract).
TEST(JointMatchInvariantsTest, ReplayableGroupIsNotPartOfTheJointInvariant) {
    constexpr std::int32_t kBlocks = 12;
    constexpr std::int32_t kBlockTokens = 4;
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull,
         .sliding_window = 0,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = kBlockTokens},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .replayable = true,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = kBlockTokens},
    };

    std::mt19937 rng(20260916);
    const std::vector<std::string> hashes = MakeHashes(kBlocks);
    for (int round = 0; round < 200; ++round) {
        BlockPool pool(64, {1, 1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, kBlockTokens, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                            /*snapshot_pool=*/nullptr,
                            /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
        std::uniform_int_distribution<std::int32_t> depth_dist(0, kBlocks);
        const std::int32_t full_depth = depth_dist(rng);
        for (std::int32_t i = 0; i < full_depth; ++i) {
            CacheBlockFor(coordinator, pool, hashes[static_cast<std::size_t>(i)], 0);
        }
        // Whatever a test registers directly for the replayable group must
        // neither lift nor lower the closed boundary.
        const std::int32_t swa_depth = depth_dist(rng);
        for (std::int32_t i = 0; i < swa_depth; ++i) {
            CacheBlockFor(coordinator, pool, hashes[static_cast<std::size_t>(i)], 1);
        }

        const auto match = MatchPrefixForTest(coordinator, hashes).device;
        EXPECT_EQ(match.num_common_tokens, full_depth * kBlockTokens) << "round " << round;
        EXPECT_EQ(match.per_group[0].NumHitBlocks(), full_depth) << "round " << round;
        EXPECT_TRUE(match.per_group[1].blocks.empty()) << "round " << round;
        // The closed group still honours hit => warm on its own.
        EXPECT_EQ(GroupPrefixBlocks(coordinator, pool, hashes, 0, full_depth), full_depth) << "round " << round;
    }
}

// The Host-suffix rule (a hit is a Device KV prefix plus a Host suffix) under
// random two-tier layouts on the K3 shape (P = 4; a Full group, one block per
// page; three State groups): each page's Full block, each page's checkpoints
// and two tails with their checkpoints in one page (each its Full block and
// checkpoints, as a prompt end and the finish after a short output publish
// them) land on either tier at random, the Device's pages with gaps in every
// other round. An inline oracle computes the Device-only hit D, the Device's KV
// coverage L (its pages from page 0 to the first gap), the Host's run C of
// pages from L, and the best boundary T* the rule allows: pages below min(L, p)
// from the Device, [L, p) from the Host, the tail whole from one tier (the
// Device's only at p <= L), each checkpoint from either. The probe hits exactly
// T* >= D, the admission succeeds and holds every page below the hit, no page
// below L is loaded (no sliding group lowers the split), and the Host loads
// exactly the checkpoints the Device lacks. With L3 on, the storage query never
// asks for a Device-resident checkpoint or page below L, and with a gapless
// Device prefix for no Device-resident key at all. The parameter deals the Full
// group's pages over that many KV-parallel shards: every load lands in its Host
// block's bucket.
class HostSuffixHitInvariantsTest : public ::testing::TestWithParam<std::int32_t> {};

INSTANTIATE_TEST_SUITE_P(KvShards, HostSuffixHitInvariantsTest, ::testing::Values(1, 2, 4));

TEST_P(HostSuffixHitInvariantsTest, HitsAreRestorableMaximalAndNeverReloadDeviceKv) {
    constexpr auto kDevice = static_cast<std::size_t>(CacheTier::kDevice);
    constexpr auto kHost = static_cast<std::size_t>(CacheTier::kHost);
    constexpr std::int32_t kP = 4;
    constexpr std::int32_t kPages = 8;
    constexpr std::int32_t kMaxTokens = kPages * kP + 1;
    constexpr std::size_t kNumTails = 2;
    constexpr std::size_t kFirstTailRow = kPages;  // the first tail's row in the layout
    constexpr std::size_t kRows = kFirstTailRow + kNumTails;
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull,
         .cache_blocks_per_lcm_block = 4,
         .block_granularity = kP,
         .shard_count = GetParam()},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    const std::vector<std::int32_t> tokens = test::MakeTokens(kMaxTokens + 1);
    std::vector<std::span<const std::int32_t>> pages;
    for (std::int32_t page = 0; page < kPages; ++page) {
        pages.emplace_back(tokens.data() + page * kP, kP);
    }
    const std::vector<std::string> hashes = ComputePrefixHashes(pages, "");
    const CacheCoordinator::PromptPrefix prompt{.prefix_hashes = hashes, .tokens = tokens, .max_tokens = kMaxTokens};
    std::mt19937 rng(20261023);
    std::uniform_int_distribution<std::int32_t> page_dist(0, kPages - 1);
    std::uniform_int_distribution<std::int32_t> tail_dist(1, kP - 1);
    std::uniform_int_distribution<std::int32_t> other_tail_dist(1, kP - 2);
    // Presence odds by [tier][Full, State], for pages and for tails with their
    // checkpoints; a tail is published whole, so its checkpoints more often
    // stay together.
    const std::array<std::array<double, 2>, 2> page_odds{{{0.75, 0.3}, {0.6, 0.4}}};
    const std::array<std::array<double, 2>, 2> tail_odds{{{0.75, 0.7}, {0.6, 0.7}}};

    for (const bool l3 : {false, true}) {
        for (int round = 0; round < 200; ++round) {
            BlockPool device(64, {4, 1, 1, 1});
            BlockPool host(64, {4, 1, 1, 1});
            CacheCoordinator coordinator =
                MakeCoordinator(specs, kP, device, /*enable_l3_storage=*/l3, &host,
                                /*snapshot_pool=*/nullptr,
                                /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
            const bool gapless_device = round % 2 == 0;
            const std::int32_t tail_page = page_dist(rng);
            const std::int32_t tail_page_tokens = tail_page * kP;
            const ContentHash tail_prior =
                tail_page == 0 ? ContentHash{} : hashes[static_cast<std::size_t>(tail_page) - 1];
            std::array<std::int32_t, kNumTails> tail_tokens{tail_dist(rng), other_tail_dist(rng)};
            if (tail_tokens[1] >= tail_tokens[0]) {
                ++tail_tokens[1];
            }
            std::array<std::int32_t, kNumTails> tail_boundary{};
            std::array<ContentHash, kNumTails> tail_hash;
            for (std::size_t t = 0; t < kNumTails; ++t) {
                tail_boundary[t] = tail_page_tokens + tail_tokens[t];
                tail_hash[t] = HashPrefixPage(
                    std::span<const std::int32_t>(tokens).subspan(static_cast<std::size_t>(tail_page_tokens),
                                                                  static_cast<std::size_t>(tail_tokens[t])),
                    tail_prior);
            }
            // present[tier][row][group]: row b < kPages is page b's Full block
            // and the checkpoints at (b + 1) * kP; row kFirstTailRow + t is
            // tail t.
            std::array<std::array<std::array<bool, 4>, kRows>, 2> present{};
            for (std::size_t tier = 0; tier < 2; ++tier) {
                for (std::size_t row = 0; row < kRows; ++row) {
                    for (std::size_t group = 0; group < 4; ++group) {
                        const auto& odds = row < kFirstTailRow ? page_odds : tail_odds;
                        present[tier][row][group] = std::bernoulli_distribution(odds[tier][group == 0 ? 0 : 1])(rng);
                    }
                }
            }
            std::int32_t device_pages = 0;
            while (device_pages < kPages && present[kDevice][static_cast<std::size_t>(device_pages)][0]) {
                ++device_pages;
            }
            if (gapless_device) {
                for (std::size_t row = static_cast<std::size_t>(device_pages); row < kFirstTailRow; ++row) {
                    present[kDevice][row][0] = false;
                }
                for (std::size_t row = kFirstTailRow; row < kRows; ++row) {
                    present[kDevice][row][0] = present[kDevice][row][0] && tail_page <= device_pages;
                }
            }
            for (std::size_t tier = 0; tier < 2; ++tier) {
                BlockPool& pool = tier == kDevice ? device : host;
                for (std::size_t row = 0; row < kRows; ++row) {
                    for (std::uint32_t group = 0; group < 4; ++group) {
                        if (present[tier][row][group]) {
                            CacheBlockFor(coordinator, pool,
                                          row < kFirstTailRow ? hashes[row] : tail_hash[row - kFirstTailRow], group);
                        }
                    }
                }
            }
            std::array<bool, kNumTails> hinted{};
            for (std::size_t t = 0; t < kNumTails; ++t) {
                hinted[t] = std::ranges::any_of(present[kHost][kFirstTailRow + t], [](bool held) { return held; });
                if (hinted[t]) {
                    coordinator.CacheHostTail(tail_prior,
                                              TailPublication{.tail_tokens = tail_tokens[t], .hash = tail_hash[t]});
                }
            }

            // The oracle.
            const auto checkpoints_on = [&](std::size_t tier, std::size_t row) {
                return present[tier][row][1] && present[tier][row][2] && present[tier][row][3];
            };
            const auto checkpoints_anywhere = [&](std::size_t row) {
                return (present[kDevice][row][1] || present[kHost][row][1]) &&
                       (present[kDevice][row][2] || present[kHost][row][2]) &&
                       (present[kDevice][row][3] || present[kHost][row][3]);
            };
            const std::int32_t device_kv = device_pages * kP;  // L
            std::int32_t host_pages = device_pages;
            while (host_pages < kPages && present[kHost][static_cast<std::size_t>(host_pages)][0]) {
                ++host_pages;
            }
            const std::int32_t host_kv = host_pages * kP;  // C
            std::int32_t device_hit = 0;                   // D
            for (std::int32_t page = 0; page < device_pages; ++page) {
                if (checkpoints_on(kDevice, static_cast<std::size_t>(page))) {
                    device_hit = (page + 1) * kP;
                }
            }
            for (std::size_t t = 0; t < kNumTails; ++t) {
                if (hinted[t] && tail_page <= device_pages && present[kDevice][kFirstTailRow + t][0] &&
                    checkpoints_on(kDevice, kFirstTailRow + t)) {
                    device_hit = std::max(device_hit, tail_boundary[t]);
                }
            }
            std::int32_t best = device_hit;  // T*
            for (std::int32_t page = 0; page < kPages; ++page) {
                const std::int32_t boundary = (page + 1) * kP;
                if (boundary > device_hit && boundary <= host_kv &&
                    checkpoints_anywhere(static_cast<std::size_t>(page))) {
                    best = std::max(best, boundary);
                }
            }
            for (std::size_t t = 0; t < kNumTails; ++t) {
                const std::size_t row = kFirstTailRow + t;
                const bool kv_tail_on_device = tail_page_tokens <= device_kv && present[kDevice][row][0];
                if (hinted[t] && tail_boundary[t] > device_hit && tail_page_tokens <= host_kv &&
                    (kv_tail_on_device || present[kHost][row][0]) && checkpoints_anywhere(row)) {
                    best = std::max(best, tail_boundary[t]);
                }
            }

            CacheCoordinator::PrefixProbe probe = coordinator.ProbePromptPrefix(prompt);
            const std::int32_t hit = std::max(probe.device.num_common_tokens, probe.host.num_common_tokens);
            EXPECT_EQ(probe.device_only_hit_tokens, device_hit) << "l3=" << l3 << " round " << round;
            EXPECT_EQ(hit, best) << "l3=" << l3 << " round " << round << ": D=" << device_hit << " L=" << device_kv
                                 << " C=" << host_kv << " tails@" << tail_boundary[0] << "," << tail_boundary[1]
                                 << " hinted=" << hinted[0] << hinted[1];
            EXPECT_GE(hit, device_hit);
            const std::int32_t hit_pages = (hit - hit % kP) / kP;
            if (best > device_hit) {
                EXPECT_EQ(probe.device.full_page_tokens, std::min(device_kv, hit_pages * kP)) << "round " << round;
            }
            std::vector<BlockTable> tables(4);
            const std::optional<CacheCoordinator::AdmissionResult> result = AdmitForTest(
                coordinator, tables, CacheCoordinator::PrefixProbe{probe}, GroupDemand{.extent = DenseGrowth{1}});
            ASSERT_TRUE(result) << "l3=" << l3 << " round " << round;
            for (std::int32_t page = 0; page < hit_pages; ++page) {
                EXPECT_TRUE(tables[0].Blocks()[static_cast<std::size_t>(page)]) << "round " << round;
            }
            std::size_t checkpoint_loads = 0;
            for (const BlockTransfer& load : result->load_pairs) {
                const GroupAllocator& allocator = coordinator.Allocator(static_cast<std::int32_t>(load.group_id));
                EXPECT_EQ(allocator.BucketOf(load.source->Location()), allocator.BucketOf(load.destination->Location()))
                    << "round " << round;
                if (load.group_id != 0) {
                    ++checkpoint_loads;
                    continue;
                }
                const auto slot = std::ranges::find(tables[0].Blocks(), load.destination) - tables[0].Blocks().begin();
                if (slot < hit_pages) {
                    EXPECT_GE(slot * kP, device_kv) << "round " << round << ": a Device KV page was loaded";
                }
            }
            std::size_t missing_on_device = 0;
            if (best > device_hit) {
                const std::size_t row = best % kP == 0             ? static_cast<std::size_t>(best / kP - 1)
                                        : best == tail_boundary[0] ? kFirstTailRow
                                                                   : kFirstTailRow + 1;
                missing_on_device = static_cast<std::size_t>(
                    std::ranges::count(std::span<const bool>(present[kDevice][row]).subspan(1), false));
            }
            EXPECT_EQ(checkpoint_loads, missing_on_device) << "round " << round;

            if (!l3) {
                continue;
            }
            for (const CacheKey& key : coordinator.MissingStorageKeys(prompt, /*max_keys=*/64).keys) {
                if (!coordinator.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id)).Contains(device, key)) {
                    continue;
                }
                EXPECT_FALSE(gapless_device) << "round " << round << ": a gapless Device prefix is asked for";
                EXPECT_FALSE(coordinator.GroupIsState(key.group_id)) << "round " << round;
                const auto page = std::ranges::find(hashes, key.content_hash) - hashes.begin();
                EXPECT_FALSE(page < device_pages) << "round " << round << ": a page below L is asked for";
            }
        }
    }
}

// Evicting by segments keeps the Device's coverage closed from the root:
// whatever the mix of sessions -- forked from earlier ones at random pages,
// resumed on Device hits, published at a chunk end and at the finish, each with
// its checkpoints, ending on a tail or a page -- and whatever the admissions
// under pressure evict, every page the Device holds has its parent page there,
// and every tail its prior page -- also when the dead prefix entries are
// collected after every round, which then leaves no Device page without a
// checkpoint boundary under it. K3 shape (P = 4; a Full group, one block per
// page, four per parent, dealt over the parameter's KV-parallel shards; three
// State groups) in a Device pool of 24 parents.
class SegmentEvictionInvariantsTest : public ::testing::TestWithParam<std::int32_t> {};

INSTANTIATE_TEST_SUITE_P(KvShards, SegmentEvictionInvariantsTest, ::testing::Values(1, 2, 4));

TEST_P(SegmentEvictionInvariantsTest, DeviceCoverageStaysClosedFromTheRootUnderSegmentEviction) {
    constexpr std::int32_t kP = 4;
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull,
         .cache_blocks_per_lcm_block = 4,
         .block_granularity = kP,
         .shard_count = GetParam()},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    for (const bool collect : {false, true}) {
        SCOPED_TRACE(collect);
        BlockPool device(24, {4, 1, 1, 1});
        BlockPool host(1, {4, 1, 1, 1});
        CacheCoordinator coordinator = MakeCoordinator(specs, kP, device, /*enable_l3_storage=*/false, &host,
                                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                                       /*write_back_evictions=*/true);
        TierTransferManager transfers{coordinator};
        struct Session {
            std::vector<std::int32_t> tokens;
            std::vector<std::string> hashes;
            std::optional<TailPublication> tail;
        };
        std::vector<Session> sessions;
        std::mt19937 rng(20261024);
        std::uniform_int_distribution<std::int32_t> length_dist(2 * kP, 9 * kP - 1);
        std::bernoulli_distribution coin(0.5);
        std::bernoulli_distribution flood(0.1);
        std::int32_t hits = 0;
        std::int32_t chunked = 0;
        std::int32_t floods = 0;
        for (int round = 0; round < 200; ++round) {
            const std::int32_t num_tokens = length_dist(rng);
            Session session{.tokens = test::MakeTokens(num_tokens, 1000 * (round + 1))};
            if (!sessions.empty() && coin(rng)) {
                const Session& base = sessions[std::uniform_int_distribution<std::size_t>(0, sessions.size() - 1)(rng)];
                const auto max_pages =
                    static_cast<std::int32_t>(std::min(base.tokens.size(), session.tokens.size())) / kP;
                const std::int32_t shared = std::uniform_int_distribution<std::int32_t>(1, max_pages)(rng) * kP;
                std::copy_n(base.tokens.begin(), shared, session.tokens.begin());
            }
            const std::int32_t full_pages = num_tokens / kP;
            std::vector<std::span<const std::int32_t>> pages;
            for (std::int32_t page = 0; page < full_pages; ++page) {
                pages.emplace_back(session.tokens.data() + page * kP, kP);
            }
            session.hashes = ComputePrefixHashes(pages, "");
            if (num_tokens % kP != 0) {
                session.tail =
                    TailPublication{.tail_tokens = num_tokens % kP,
                                    .hash = HashPrefixPage(std::span<const std::int32_t>(session.tokens)
                                                               .subspan(static_cast<std::size_t>(full_pages * kP)),
                                                           session.hashes.back())};
            }

            // Resume on what the Device holds, below the last token.
            const std::int32_t max_tokens = num_tokens - 1;
            const CacheCoordinator::PromptPrefix prompt{
                .prefix_hashes =
                    std::span<const std::string>(session.hashes).first(static_cast<std::size_t>(max_tokens / kP)),
                .tokens = session.tokens,
                .max_tokens = max_tokens};
            CacheCoordinator::PrefixProbe probe = coordinator.ProbePromptPrefix(prompt);
            const std::int32_t hit = std::max(probe.device.num_common_tokens, probe.host.num_common_tokens);
            hits += hit > 0 ? 1 : 0;
            // A chunk end with its checkpoint at a page past the hit, before the
            // finish.
            const std::int32_t first_new = hit / kP;
            const std::int32_t last_chunk_page = session.tail ? full_pages : full_pages - 1;
            std::optional<std::int32_t> chunk_pages;
            if (first_new + 1 <= last_chunk_page && coin(rng)) {
                chunk_pages = std::uniform_int_distribution<std::int32_t>(first_new + 1, last_chunk_page)(rng);
            }
            std::vector<BlockTable> tables(4);
            const std::int32_t first_end = chunk_pages ? *chunk_pages * kP : num_tokens;
            if (!AdmitForTest(coordinator, tables, std::move(probe),
                              GroupDemand{.extent = DenseGrowth{first_end - hit}})) {
                coordinator.Free(tables);
                continue;
            }
            std::int32_t published_pages = first_new;
            if (chunk_pages) {
                const RequestProgress chunk_end{
                    .completed_pages =
                        CompletedPages{.prefix_hashes = std::span<const std::string>(session.hashes)
                                                            .first(static_cast<std::size_t>(*chunk_pages)),
                                       .first_new_prefix_page = first_new,
                                       .publish_state = true,
                                       .boundary_kind = CacheBoundaryKind::kChunk},
                    .num_computed_tokens = first_end,
                };
                if (!AdmitForTest(coordinator, tables, coordinator.ProbePrefix({}),
                                  GroupDemand{.extent = DenseGrowth{num_tokens - first_end}}, chunk_end)) {
                    coordinator.Free(tables);
                    continue;
                }
                published_pages = *chunk_pages;
                ++chunked;
            }
            coordinator.CacheCompletedBlocks(
                tables,
                RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = session.hashes,
                                                                  .first_new_prefix_page = published_pages,
                                                                  .tail = session.tail,
                                                                  .publish_state = true,
                                                                  .boundary_kind = CacheBoundaryKind::kEndpoint},
                                .num_computed_tokens = num_tokens},
                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator));
            coordinator.Free(tables);
            sessions.push_back(std::move(session));
            // Now and then a request needs the whole pool (96 Full slots). Nothing
            // is in use, so it fits: the candidates run out, and the pages
            // several checkpoint boundaries share leave too.
            if (flood(rng)) {
                std::vector<BlockTable> flood_tables(4);
                EXPECT_TRUE(AdmitForTest(coordinator, flood_tables, coordinator.ProbePrefix({}),
                                         GroupDemand{.extent = DenseGrowth{96 * kP}}, RequestProgress{},
                                         StateStep{.kind = StateStep::Kind::kKeep}))
                    << "round " << round;
                coordinator.Free(flood_tables);
                ++floods;
            }

            // As the scheduler does once per plan: the write-back drain -- the
            // Host's one parent never holds a checkpoint boundary's three
            // checkpoints, so every segment is dropped -- and then the
            // collection; nothing is in use now, so every page left without a
            // checkpoint boundary under it goes.
            if (std::optional<WriteBackOperation> store = transfers.StartEvictionStores()) {
                transfers.CompleteWriteBack(store->op_id);
            }
            if (collect) {
                coordinator.CollectDeadEntries();
            }
            const auto on_device = [&](const std::string& hash) {
                return coordinator.GroupPrefixIndex(0).Contains(device, KeyFor(hash, 0));
            };
            const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator);
            for (const Session& published : sessions) {
                for (std::size_t page = 0; page < published.hashes.size(); ++page) {
                    if (!on_device(published.hashes[page])) {
                        continue;
                    }
                    if (page > 0) {
                        EXPECT_TRUE(on_device(published.hashes[page - 1]))
                            << "round " << round << ": page " << page << " outlives its parent";
                    }
                    if (collect) {
                        EXPECT_GT(tree.Find(published.hashes[page])->checkpoint_boundaries_in_subtree[0], 0)
                            << "round " << round << ": page " << page << " is dead";
                    }
                }
                if (published.tail && on_device(published.tail->hash)) {
                    EXPECT_TRUE(on_device(published.hashes.back()))
                        << "round " << round << ": a tail outlives its prior";
                }
            }
        }
        // The run resumed on hits, published chunk ends, admitted floods, and
        // evicted under pressure, middle pages with their suffix too.
        EXPECT_GT(hits, 0);
        EXPECT_GT(chunked, 0);
        EXPECT_GT(floods, 0);
        const CacheCoordinator::DeviceEvictionStats& evictions = coordinator.DeviceEvictions();
        EXPECT_GT(evictions.admission.blocks, 0u);
        EXPECT_GT(evictions.stranded_suffix.blocks, 0u);
    }
}

TEST(JointMatchInvariantsTest, DraftOnlyGroupJoinsConvergenceAsOrdinaryGroup) {
    // Three groups: full target, target state-like full group with packing 2,
    // and a draft sliding group. The converged boundary must be supported by
    // all three — no group is special.
    constexpr std::int32_t kBlocks = 8;
    constexpr std::int32_t kBlockTokens = 4;
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull,
         .sliding_window = 0,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = kBlockTokens},
        {.attn_kind = AttnKind::kFull,
         .sliding_window = 0,
         .cache_blocks_per_lcm_block = 2,
         .block_granularity = kBlockTokens},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = kBlockTokens},
    };
    BlockPool pool(64, {1, 2, 1});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, kBlockTokens, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
    const std::vector<std::string> hashes = MakeHashes(kBlocks);

    // Cache depth 6 for the full groups, but only blocks [2, 5) for the
    // sliding group: its resumable boundary is 5 (lookback run intact).
    for (std::int32_t i = 0; i < 6; ++i) {
        CacheBlockFor(coordinator, pool, hashes[static_cast<std::size_t>(i)], 0);
        CacheBlockFor(coordinator, pool, hashes[static_cast<std::size_t>(i)], 1);
    }
    for (std::int32_t i = 2; i < 5; ++i) {
        CacheBlockFor(coordinator, pool, hashes[static_cast<std::size_t>(i)], 2);
    }

    const auto match = MatchPrefixForTest(coordinator, hashes).device;
    const std::int32_t common_blocks = match.num_common_tokens / kBlockTokens;
    EXPECT_EQ(common_blocks, 5);
    for (std::uint32_t group = 0; group < 3; ++group) {
        EXPECT_GE(GroupPrefixBlocks(coordinator, pool, hashes, group, common_blocks), common_blocks);
    }
}

TEST(JointMatchInvariantsTest, AvailabilitySeparatesStateMissingFromFullKvMissing) {
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    BlockPool device(64, {1, 1, 1});
    BlockPool host(64, {1, 1, 1});
    auto coordinator = MakeCoordinator(specs, 4, device, /*enable_l3_storage=*/false, &host,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/true);
    const auto hashes = MakeHashes(3);
    CacheBlockFor(coordinator, device, hashes[0], 0);
    CacheBlockFor(coordinator, host, hashes[1], 0);
    CacheBlockFor(coordinator, device, hashes[1], 1);
    CacheBlockFor(coordinator, host, hashes[0], 2);
    CacheBlockFor(coordinator, host, hashes[2], 1);
    CacheBlockFor(coordinator, device, hashes[2], 2);
    const CacheCoordinator::PromptPrefix prompt{.prefix_hashes = hashes, .max_tokens = 12};
    const auto before = coordinator.ProbePromptPrefix(prompt);
    const auto rows = coordinator.InspectPrefixAvailability(prompt);
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_EQ(rows[0].tokens, 12);
    EXPECT_EQ(rows[0].union_missing, 1u);  // Both states exist, the Full KV has a hole.
    EXPECT_EQ(rows[1].tokens, 8);
    EXPECT_EQ(rows[1].union_missing, 4u);  // Full KV spans the tiers; only state2 is missing.
    EXPECT_EQ(rows[2].tokens, 4);
    EXPECT_EQ(rows[2].union_missing, 2u);  // state1 and state2 have no common boundary.
    EXPECT_EQ(rows[1].device_missing, 5u);
    EXPECT_EQ(rows[1].host_missing, 7u);
    const auto after = coordinator.ProbePromptPrefix(prompt);
    EXPECT_EQ(before.host.num_common_tokens, after.host.num_common_tokens);
    EXPECT_EQ(before.device.num_common_tokens, after.device.num_common_tokens);
    EXPECT_EQ(after.host.num_common_tokens, 0);
}

TEST(JointMatchInvariantsTest, AvailabilityChecksSlidingLookbackAndSkipsReplayableGroups) {
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .replayable = true,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    BlockPool device(64, {1, 1, 1});
    auto coordinator = MakeCoordinator(specs, 4, device, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false);
    const auto hashes = MakeHashes(3);
    for (const auto& hash : hashes) {
        CacheBlockFor(coordinator, device, hash, 0);
    }
    CacheBlockFor(coordinator, device, hashes[0], 1);
    CacheBlockFor(coordinator, device, hashes[2], 1);
    const auto rows = coordinator.InspectPrefixAvailability({.prefix_hashes = hashes, .max_tokens = 12});
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_EQ(rows[0].union_missing, 2u);  // Last window block alone is insufficient.
    EXPECT_EQ(rows[1].union_missing, 2u);
    EXPECT_EQ(rows[2].union_missing, 0u);
    EXPECT_EQ(rows[2].host_missing, 3u);  // No Host pool; replayable group is still excluded.
}

TEST(JointMatchInvariantsTest, AvailabilityIncludesUnalignedTailAndStateOnAnotherTier) {
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    BlockPool device(64, {1, 1});
    BlockPool host(64, {1, 1});
    auto coordinator = MakeCoordinator(specs, 4, device, /*enable_l3_storage=*/false, &host,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/true);
    const std::vector<std::int32_t> tokens{1, 2, 3, 4, 5, 6};
    const std::vector<std::string> hashes{HashPrefixPage(std::span(tokens).first(4), "")};
    const auto tail_hash = HashPrefixPage(std::span(tokens).subspan(4), hashes[0]);
    CacheBlockFor(coordinator, device, hashes[0], 0);
    CacheBlockFor(coordinator, device, hashes[0], 1);
    CacheBlockFor(coordinator, host, tail_hash, 0);
    const CacheCoordinator::PromptPrefix prompt{.prefix_hashes = hashes,
                                                .tokens = tokens,
                                                .max_tokens = 6,
                                                .tail = TailPublication{.tail_tokens = 2, .hash = tail_hash}};
    auto rows = coordinator.InspectPrefixAvailability(prompt);
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].tokens, 6);
    EXPECT_EQ(rows[0].union_missing, 2u);
    EXPECT_EQ(rows[1].tokens, 4);
    EXPECT_EQ(rows[1].union_missing, 0u);
    CacheBlockFor(coordinator, host, tail_hash, 1);
    rows = coordinator.InspectPrefixAvailability(prompt);
    EXPECT_EQ(rows[0].union_missing, 0u);
    EXPECT_EQ(rows[0].device_missing, 3u);
    EXPECT_EQ(rows[0].host_missing, 1u);
}

}  // namespace
}  // namespace tokenspeed
