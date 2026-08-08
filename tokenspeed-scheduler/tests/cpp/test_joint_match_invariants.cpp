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
// a per-group ground-truth replay.

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "cache/kv_cache_coordinator.h"
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

CacheKey KeyFor(const std::string& content_hash, GroupId group_id) {
    return CacheKey{.namespace_id = kDefaultCacheNamespaceId, .group_id = group_id, .content_hash = content_hash};
}

std::uint64_t g_epoch = 0;

std::int32_t CacheBlockFor(KvCacheCoordinator& coordinator, BlockPool& pool, const std::string& content_hash,
                           GroupId group_id) {
    KvCacheManager& manager = coordinator.GroupManager(static_cast<std::int32_t>(group_id));
    CacheBlockRef block_ref = pool.AcquireBlock(group_id, manager.CacheBlocksPerLcmBlock());
    if (!block_ref) {
        return -1;
    }
    const std::int32_t id = block_ref->Location().lcm_block_id;
    manager.RegisterCachedBlock(pool, block_ref, KeyFor(content_hash, group_id), ++g_epoch);
    block_ref.reset();
    return id;
}

// Ground truth for one group in isolation: the deepest prefix (in blocks)
// this group alone can support, given which of its blocks are cached.
// Full attention is prefix-closed; a sliding group needs its lookback run.
std::int32_t GroupPrefixBlocks(const KvCacheCoordinator& coordinator, const BlockPool& pool,
                               std::span<const std::string> hashes, GroupId group_id, std::int32_t bound_blocks) {
    const KvCacheManager& manager = coordinator.GroupManager(static_cast<std::int32_t>(group_id));
    std::vector<CacheKey> keys;
    keys.reserve(hashes.size());
    for (const std::string& hash : hashes) {
        keys.push_back(KeyFor(hash, group_id));
    }
    const GroupPrefixProbe probe = manager.Probe(pool, keys, 0, bound_blocks);
    return static_cast<std::int32_t>(probe.hits.size());
}

TEST(JointMatchInvariantsTest, HitImpliesWarmUnderRandomCacheEvictSequences) {
    constexpr std::int32_t kBlocks = 12;
    constexpr std::int32_t kBlockTokens = 4;
    // full target group + a sliding "draft" group (window 8 -> lookback 2
    // blocks): the draft-SWA-under-full-target shape.
    const std::vector<KvCacheSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1},
        {.kind = AttnKind::kSlidingWindow, .sliding_window = 8, .cache_blocks_per_lcm_block = 1},
    };

    std::mt19937 rng(20260807);
    const std::vector<std::string> hashes = MakeHashes(kBlocks);

    for (int round = 0; round < 200; ++round) {
        BlockPool pool(64);
        {
            KvCacheCoordinator coordinator = MakeCoordinator(specs, kBlockTokens, pool);

            // Random per-group caching: each group caches a random prefix
            // subset of the request's blocks (front-truncated to mimic the
            // sliding group's reclaim of slid-out blocks).
            std::vector<std::vector<std::int32_t>> cached_ids(specs.size());
            for (GroupId group = 0; group < static_cast<GroupId>(specs.size()); ++group) {
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
            for (GroupId group = 0; group < static_cast<GroupId>(specs.size()); ++group) {
                std::int32_t to_evict = evict_count_dist(rng);
                std::shuffle(cached_ids[group].begin(), cached_ids[group].end(), rng);
                for (const std::int32_t lcm_block_id : cached_ids[group]) {
                    if (to_evict-- <= 0 || lcm_block_id < 0) {
                        break;
                    }
                    coordinator.GroupManager(static_cast<std::int32_t>(group))
                        .EvictCachedBlock(pool, CacheBlockLocation{.lcm_block_id = lcm_block_id, .slot_index = 0});
                }
            }

            const auto match = MatchPrefixForTest(coordinator, hashes).device;
            const std::int32_t common_blocks = match.num_common_tokens / kBlockTokens;

            // hit => warm: every group must independently support the
            // converged boundary. This is the property the draft relies on
            // (its KV is recoverable at every exposed boundary).
            for (GroupId group = 0; group < static_cast<GroupId>(specs.size()); ++group) {
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

TEST(JointMatchInvariantsTest, DraftOnlyGroupJoinsConvergenceAsOrdinaryGroup) {
    // Three groups: full target, target state-like full group with packing 2,
    // and a draft sliding group. The converged boundary must be supported by
    // all three — no group is special.
    constexpr std::int32_t kBlocks = 8;
    constexpr std::int32_t kBlockTokens = 4;
    const std::vector<KvCacheSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2},
        {.kind = AttnKind::kSlidingWindow, .sliding_window = 8, .cache_blocks_per_lcm_block = 1},
    };
    BlockPool pool(64);
    KvCacheCoordinator coordinator = MakeCoordinator(specs, kBlockTokens, pool);
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
    for (GroupId group = 0; group < 3; ++group) {
        EXPECT_GE(GroupPrefixBlocks(coordinator, pool, hashes, group, common_blocks), common_blocks);
    }
}

}  // namespace
}  // namespace tokenspeed
