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
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>

#include "cache/coordinator/admission_planner.h"
#include "cache/coordinator/gds_coverage.h"

#include "cache/coordinator/gds_planner.h"
#include "cache/coordinator/gds_tiered.h"
#include "cache/coordinator/recovery_route.h"
#include "fsm/forward_states.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache_test_access.h"
#include "cache/tier/transfer_manager.h"
#include "scheduler/operations/prefill_chunk.h"

namespace tokenspeed::test {
namespace {
constexpr std::int32_t kPage = 4;
struct Point {
    std::vector<std::int32_t> tokens;
    std::vector<std::string> hashes;
    std::optional<TailPublication> tail;
    explicit Point(std::int32_t size) : tokens(static_cast<std::size_t>(size)) {
        std::iota(tokens.begin(), tokens.end(), 1);
        Rehash();
    }
    void Rehash() {
        hashes.clear();
        tail.reset();
        const auto size = tokens.size();
        ContentHash prior;
        for (std::size_t i = 0; i + kPage <= tokens.size(); i += kPage) {
            prior = HashPrefixPage(std::span{tokens}.subspan(i, kPage), prior);
            hashes.push_back(prior.ToString());
        }
        if (size % kPage)
            tail = TailPublication{.tail_tokens = static_cast<std::int32_t>(size % kPage),
                                   .hash = HashPrefixPage(std::span{tokens}.subspan(hashes.size() * kPage), prior)};
    }
    ContentHash Hash() const { return tail ? tail->hash : ContentHash{hashes.back()}; }
};
std::vector<CacheGroupSpec> Specs() {
    return {{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kPage},
            {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
}
void PublishPoint(CacheCoordinator& cache, const Point& point) {
    std::vector<BlockTable> tables(cache.NumGroups());
    auto admitted = AdmitForTest(cache, tables, static_cast<std::int32_t>(point.tokens.size()));
    ASSERT_TRUE(admitted);
    cache.CacheCompletedBlocks(
        tables,
        RequestProgress{.completed_pages = CompletedPages{.tokens = point.tokens,
                                                          .prefix_hashes = point.hashes,
                                                          .tail = point.tail,
                                                          .publish_state = true,
                                                          .boundary_kind = CacheBoundaryKind::kEndpoint,
                                                          .stream_completed_to_host = true},
                        .num_computed_tokens = static_cast<std::int32_t>(point.tokens.size())},
        admitted->access_epoch);
    cache.Free(tables);
}

// The independent per-key query is the matcher-facing route contract. Check
// exact dependencies too: matching just the ready bit can hide stale ranges.
void CheckIndexedRoute(CacheCoordinator& cache, const Point& point) {
    const auto& store = CacheCoordinatorTestAccess::Store(cache);
    std::vector<ContentHash> hashes;
    for (const auto& hash : point.hashes) hashes.emplace_back(hash);
    const auto contains = [&](const CacheKey& key, CacheTier tier) {
        const auto* pool = tier == CacheTier::kDevice ? &store.DevicePool() : store.HostPool();
        return pool && store.Groups()[key.group_id].Index().Contains(*pool, key);
    };
    auto expected = ResolveRecoveryRoute(store, hashes, point.Hash(), static_cast<std::int32_t>(point.tokens.size()),
                                         store.HostPool() != nullptr, true, contains);
    const auto& actual = store.RecoveryDependencies().At(point.Hash());
    EXPECT_EQ(actual.ready, expected.has_value());
    EXPECT_EQ(actual.device_ready, expected && expected->device_only);
    if (!expected)
        expected = ResolveRecoveryRoute(store, hashes, point.Hash(), static_cast<std::int32_t>(point.tokens.size()),
                                        store.HostPool() != nullptr, true, [&](const CacheKey& key, CacheTier tier) {
                                            return contains(key, tier) ||
                                                   (tier == CacheTier::kHost && store.PendingHostKeys().contains(key));
                                        });
    EXPECT_EQ(actual.pending, !actual.ready && expected.has_value());
    std::unordered_map<CacheKey, std::optional<CacheKey>, CacheKeyHash> previous;
    for (const auto& leaf : store.RecoveryDependencies().Leaves()) previous.emplace(leaf.key, leaf.previous);
    std::unordered_set<CacheKey, CacheKeyHash> wanted, found;
    if (expected)
        for (const auto& dependency : expected->dependencies)
            if (dependency.tier == CacheTier::kDevice) wanted.insert(dependency.key);
    for (const auto& path : actual.device_paths) {
        auto key = path.last;
        for (;;) {
            found.insert(key);
            if (key == path.first) break;
            ASSERT_TRUE(previous.at(key));
            key = *previous.at(key);
        }
    }
    EXPECT_EQ(found, wanted);
}

class GdsEviction : public testing::Test {
protected:
    BlockPool device{64, {1, 1}};
    BlockPool host{64, {1, 1}};
    CacheCoordinator cache;
    std::pmr::monotonic_buffer_resource arena;
    std::vector<CacheBlockRef> pins;

    GdsEviction() : GdsEviction(false, false) {}
    GdsEviction(bool write_back, bool stream)
        : cache{MakeCoordinator(Specs(), kPage, device, false, (write_back || stream) ? &host : nullptr, nullptr,
                                nullptr, stream, write_back, 0)} {
        cache.ConfigureEviction(PrefixCacheEvictionConfig::Gds({0.25, 0}));
    }
    void Publish(const Point& point) {
        std::vector<BlockTable> tables(2);
        auto admitted = AdmitForTest(cache, tables, static_cast<std::int32_t>(point.tokens.size()));
        ASSERT_TRUE(admitted);
        cache.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.tokens = point.tokens,
                                                              .prefix_hashes = point.hashes,
                                                              .tail = point.tail,
                                                              .publish_state = true,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint,
                                                              .stream_completed_to_host = true},
                            .num_computed_tokens = static_cast<std::int32_t>(point.tokens.size())},
            admitted->access_epoch);
        cache.Free(tables);
    }
    void CheckProductionAttribution() {
        const Point parent{16}, produced{32};
        Publish(parent);
        Use(parent);
        auto stats = std::make_shared<CheckpointProductionStats>();
        auto receipt = std::make_shared<CheckpointLifecycle>(stats, 1);
        {
            std::vector<BlockTable> tables(2);
            auto admission = AdmitForTest(cache, tables, 32);
            ASSERT_TRUE(admission);
            std::vector<CacheBlockRef> extra{CacheBlockRef{}, device.AcquireBlock(1)};
            receipt->Materialize();
            cache.CacheCompletedBlocks(tables,
                                       {.completed_pages = CompletedPages{.tokens = produced.tokens,
                                                                          .prefix_hashes = produced.hashes,
                                                                          .state_checkpoint_pages = 8,
                                                                          .state_checkpoint_blocks = extra,
                                                                          .checkpoint_lifecycle = receipt},
                                        .num_computed_tokens = 32},
                                       admission->access_epoch);
            cache.Free(tables);
        }
        ASSERT_TRUE(Holds(produced));
        EXPECT_EQ(stats->published, 1u);
        Fill();
        std::vector<CacheBlockRef> host_pins;
        while (host.NumEmptyLcmBlocks()) host_pins.push_back(host.AcquireBlock(0));
        EXPECT_FALSE(Plan(1000));
        EXPECT_EQ(stats->fresh_rejected, 0u);
        {
            auto preview = Plan(1);
            ASSERT_TRUE(preview);
        }
        EXPECT_EQ(stats->fresh_rejected, 0u);
        auto plan = Plan(1);
        ASSERT_TRUE(plan);
        CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
        EXPECT_FALSE(Holds(produced));
        EXPECT_TRUE(Holds(parent));
        EXPECT_EQ(stats->fresh_rejected, 1u);
        EXPECT_EQ(stats->rejected_extra_blocks, 1u);
        EXPECT_EQ(stats->rejected_unmeasured, 1u);
        receipt->RejectFresh();
        EXPECT_EQ(stats->fresh_rejected, 1u);
        EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
    }
    void Use(const Point& point) {
        auto acquired = CacheCoordinatorTestAccess::MatchPromptPrefix(
            cache, PromptPrefix{.prefix_hashes = point.hashes,
                                .tokens = point.tokens,
                                .max_tokens = static_cast<std::int32_t>(point.tokens.size())});
    }
    void Fill() {
        while (device.NumEmptyLcmBlocks() > 0) pins.push_back(device.AcquireBlock(0));
    }
    std::optional<PlannedEviction> Plan(std::int64_t count) {
        auto& store = CacheCoordinatorTestAccess::Store(cache);
        CacheEvictionPlan capacity{device, store.Groups()};
        AdmissionCandidateConstraints constraints{&arena};
        std::array<BlockPlacementDemand, 2> demand{};
        for (auto& group : demand) group.blocks_by_bucket = {0};
        demand[1].any_bucket_blocks = count;
        return PlanGdsEviction(store, constraints, demand, capacity, &arena);
    }
    bool Holds(const Point& point) {
        return cache.GroupPrefixIndex(1).Contains(device, CacheKey{.group_id = 1, .content_hash = point.Hash()});
    }
    std::size_t Blocks(const PlannedEviction& plan) {
        std::size_t count = 0;
        for (const auto& set : plan.victims) count += set.members.size();
        return count;
    }
};

TEST(GdsAdmission, NoEvictionStillPublishesAndAllocates) {
    for (const auto config : {PrefixCacheEvictionConfig::Lru(), PrefixCacheEvictionConfig::Gds({0.25, 0})}) {
        SCOPED_TRACE(config.policy == PrefixCacheEvictionConfig::Policy::kGds ? "gds" : "lru");
        BlockPool device{16, {1, 1}};
        auto cache = MakeCoordinator(Specs(), kPage, device, false, nullptr, nullptr, nullptr, false, false, 0);
        cache.ConfigureEviction(config);
        std::vector<BlockTable> tables(2);
        const Point point{4};
        auto first = AdmitForTest(cache, tables, 4);
        ASSERT_TRUE(first);
        const CacheBlockLocation published = tables[0].Blocks().front()->Location();
        const RequestProgress progress{.completed_pages = CompletedPages{.tokens = point.tokens,
                                                                         .prefix_hashes = point.hashes,
                                                                         .publish_state = true,
                                                                         .boundary_kind = CacheBoundaryKind::kEndpoint},
                                       .num_computed_tokens = 4};
        auto next = AdmitForTest(cache, tables, GroupDemand{.extent = DenseGrowth{4}}, progress);
        ASSERT_TRUE(next);
        EXPECT_EQ(tables[0].NumBlocks(), 2);
        EXPECT_EQ(tables[0].Blocks().front()->Location(), published);
        ASSERT_EQ(next->new_page_ids.size(), 2u);
        EXPECT_EQ(next->new_page_ids[0].size(), 1u);
        EXPECT_EQ(cache.Prober().ProbePrefix(point.hashes).device.num_common_tokens, 4);
        EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
        EXPECT_EQ(cache.GdsStats().committed_plans, 0u);
        cache.Free(tables);
        EXPECT_EQ(cache.Prober().ProbePrefix(point.hashes).device.num_common_tokens, 4);
    }
}

TEST_F(GdsEviction, AdmissionCommitsGdsMetadataWithoutPhysicalVictims) {
    std::vector<BlockTable> tables(2);
    auto first = AdmitForTest(cache, tables, 4);
    ASSERT_TRUE(first);
    const std::array demands{
        GroupDemand{.table = &tables[0], .extent = DenseGrowth{4}},
        GroupDemand{.table = &tables[1], .extent = StateStep{StateStep::Kind::kKeep}},
    };
    // Exercise commit independently of selection: logical metadata remains
    // work even when the plan removes no physical Device blocks.
    AdmissionPlan plan{.prefix = cache.Prober().ProbePrefix({})};
    plan.eviction.gds = std::make_unique<GdsEvictionCommit>();
    plan.eviction.gds->stats.watermark = 0.8;
    plan.eviction.gds->stats.committed_plans = 1;
    auto result = CacheCoordinatorTestAccess::CommitAdmission(
        cache, std::move(plan), demands, RequestProgress{.num_computed_tokens = 4}, first->access_epoch);
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    ASSERT_EQ(result.new_page_ids.size(), 2u);
    EXPECT_EQ(result.new_page_ids[0].size(), 1u);
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0.8);
    EXPECT_EQ(cache.GdsStats().committed_plans, 1u);
    cache.Free(tables);
}

TEST_F(GdsEviction, SharedPrefixChangesTheSizeWithoutRefreshingTheStamp) {
    Point r4{16}, r8{32};
    Publish(r4);
    Publish(r8);
    Use(r4);
    Use(r8);
    Fill();
    auto first = Plan(1);
    ASSERT_TRUE(first);
    EXPECT_EQ(Blocks(*first), 5u);
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
    CacheCoordinatorTestAccess::CommitVictims(cache, *first);
    EXPECT_TRUE(Holds(r4));
    EXPECT_FALSE(Holds(r8));
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0.8);
    Fill();
    auto second = Plan(1);
    ASSERT_TRUE(second);
    EXPECT_EQ(Blocks(*second), 5u);
    EXPECT_DOUBLE_EQ(second->gds->stats.watermark, 0.8);
}

TEST_F(GdsEviction, EvictsMiddleStateWithoutDiscardingSharedKV) {
    struct DebugScores {
        spdlog::level::level_enum previous{spdlog::get_level()};
        DebugScores() { spdlog::set_level(spdlog::level::debug); }
        ~DebugScores() { spdlog::set_level(previous); }
    } debug_scores;
    Point r4{16}, r6{24}, r8{32};
    Publish(r4);
    Publish(r6);
    Publish(r8);
    Use(r6);
    CacheCoordinatorTestAccess::Store(cache).Checkpoints().CommitWatermark(2);
    Use(r4);
    Use(r8);
    Fill();
    auto plan = Plan(1);
    ASSERT_TRUE(plan);
    ASSERT_EQ(Blocks(*plan), 1u);
    const auto choice =
        std::ranges::find(plan->gds->trace, GdsScoreObservation::Cause::kEviction, &GdsScoreObservation::cause);
    ASSERT_NE(choice, plan->gds->trace.end());
    EXPECT_EQ(choice->hash, r6.Hash());
    EXPECT_EQ(choice->tokens, 24);
    EXPECT_EQ(choice->fallback_tokens, 16);
    EXPECT_DOUBLE_EQ(choice->relief, 1);
    EXPECT_DOUBLE_EQ(choice->score, 2);
    // Observing the prospective choice neither advances time nor evicts it.
    EXPECT_TRUE(Holds(r6));
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 2);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_TRUE(Holds(r4));
    EXPECT_FALSE(Holds(r6));
    EXPECT_TRUE(Holds(r8));
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 2);
    Fill();
    auto after = Plan(1);
    ASSERT_TRUE(after);
    EXPECT_EQ(Blocks(*after), 5u);
    EXPECT_DOUBLE_EQ(after->gds->stats.watermark, 2.8);
    // Reproducing the middle state interns the existing KV identities.
    pins.clear();
    Publish(r6);
    Fill();
    auto republished = Plan(1);
    ASSERT_TRUE(republished);
    EXPECT_EQ(Blocks(*republished), 3u);
    EXPECT_DOUBLE_EQ(republished->gds->stats.watermark, 2 + 2.0 / 3);
}

TEST_F(GdsEviction, MovedPlansCommitEachLogicalDecisionOnceAcrossSeveralBlocks) {
    const Point parent{16}, child{32};
    Publish(parent);
    Publish(child);
    Use(parent);
    Use(child);
    Fill();
    {
        auto discarded = Plan(1);
        ASSERT_TRUE(discarded);
        auto moved = std::move(*discarded);
        discarded.reset();
    }
    EXPECT_TRUE(Holds(parent));
    EXPECT_TRUE(Holds(child));
    EXPECT_EQ(device.NumEmptyLcmBlocks(), 0);
    EXPECT_EQ(cache.GdsStats().committed_plans, 0u);
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);

    auto selected = Plan(1);
    ASSERT_TRUE(selected);
    auto moved = std::move(*selected);
    selected.reset();
    CacheCoordinatorTestAccess::CommitVictims(cache, moved);
    EXPECT_TRUE(Holds(parent));
    EXPECT_FALSE(Holds(child));
    EXPECT_EQ(device.NumEmptyLcmBlocks(), 5);
    EXPECT_EQ(cache.GdsStats().committed_plans, 1u);
    EXPECT_EQ(cache.GdsStats().evicted_units, 1u);
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0.8);

    Fill();
    auto next = Plan(1);
    ASSERT_TRUE(next);
    CacheCoordinatorTestAccess::CommitVictims(cache, *next);
    EXPECT_FALSE(Holds(parent));
    EXPECT_EQ(device.NumEmptyLcmBlocks(), 5);
    EXPECT_EQ(cache.GdsStats().committed_plans, 2u);
    EXPECT_EQ(cache.GdsStats().evicted_units, 2u);
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0.8);
}

TEST_F(GdsEviction, UnpublishedMiddleCheckpointChangesScoringWithoutMutatingTheIndex) {
    Point r4{16}, r6{24}, r8{32};
    Publish(r4);
    Publish(r8);
    auto& store = CacheCoordinatorTestAccess::Store(cache);
    store.Checkpoints().CommitWatermark(2);
    Use(r4);
    Use(r8);
    std::vector<BlockTable> tables(2);
    const auto producer = AdmitForTest(cache, tables, 24);
    ASSERT_TRUE(producer);
    const CheckpointPublication publication{.tokens = r6.tokens, .prefix_hashes = r6.hashes, .publish_state = true};
    std::array<GroupDemand, 2> demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{4}},
                                       GroupDemand{.table = &tables[1], .extent = StateStep{StateStep::Kind::kRotate}}};
    const auto before = store.RecoveryDependencies().GetStats();
    AdmissionCandidateConstraints constraints{&arena};
    constraints.publication = std::make_unique<PublicationPreview>(store, demands, publication);
    for (const auto& [tier, location] : constraints.publication->Protected())
        constraints.protected_locations.insert(location);
    Fill();
    CacheEvictionPlan capacity{device, store.Groups()};
    std::array<BlockPlacementDemand, 2> demand{};
    for (auto& group : demand) group.blocks_by_bucket = {0};
    demand[1].any_bucket_blocks = 1;
    auto plan = PlanGdsEviction(store, constraints, demand, capacity, &arena);
    ASSERT_TRUE(plan);
    EXPECT_EQ(Blocks(*plan), 3u);
    EXPECT_DOUBLE_EQ(plan->gds->stats.watermark, 2 + 2.0 / 3);
    EXPECT_FALSE(Holds(r6));
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 2);
    EXPECT_EQ(store.RecoveryDependencies().GetStats().points, before.points);
    EXPECT_EQ(store.RecoveryDependencies().GetStats().route_updates, before.route_updates);
    for (const auto& set : plan->victims)
        for (const auto& member : set.members) EXPECT_FALSE(constraints.protected_locations.contains(member.location));
    demand[1].any_bucket_blocks = 64;
    CacheEvictionPlan impossible{device, store.Groups()};
    EXPECT_FALSE(PlanGdsEviction(store, constraints, demand, impossible, &arena));
    EXPECT_EQ(cache.GdsStats().evicted_units, 0u);
    EXPECT_FALSE(Holds(r6));
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    cache.CacheCompletedBlocks(tables, {.completed_pages = publication, .num_computed_tokens = 24},
                               producer->access_epoch);
    EXPECT_TRUE(Holds(r6));
    EXPECT_TRUE(store.RecoveryDependencies().At(r6.Hash()).ready);
    cache.Free(tables);
}

TEST_F(GdsEviction, PublicationPreviewHandlesExactTailsWithoutClaimingIncompleteState) {
    Point longer{23}, middle{21};
    Publish(longer);
    auto& store = CacheCoordinatorTestAccess::Store(cache);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(cache, tables, 21));
    const CheckpointPublication publication{
        .tokens = middle.tokens, .prefix_hashes = middle.hashes, .tail = middle.tail, .publish_state = true};
    std::array<GroupDemand, 2> demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{0}},
                                       GroupDemand{.table = &tables[1], .extent = StateStep{StateStep::Kind::kKeep}}};
    const PublicationPreview preview{store, demands, publication};
    const auto points = preview.Points(store);
    const auto existing = std::ranges::find(points, longer.Hash(), &CheckpointTree::RecoveryPoint::hash);
    ASSERT_NE(existing, points.end());
    EXPECT_EQ(existing->parent, middle.Hash());
    EXPECT_FALSE(Holds(middle));
    cache.Free(tables);
    const PublicationPreview missing{store, demands, publication};
    EXPECT_FALSE(missing.Holds(middle.Hash()));
    EXPECT_EQ(missing.Points(store).size(), 1u);
}

TEST(CheckpointMaterialization, OutputBecomesMaterializedOnlyAtItsComputedFrontier) {
    BlockPool pool{4, {1}};
    auto stats = std::make_shared<CheckpointProductionStats>();
    fsm::CacheProgress progress;
    progress.state_checkpoint_tokens = 16;
    progress.state_checkpoint_blocks.push_back(pool.AcquireBlock(0));
    progress.checkpoint_lifecycle = std::make_shared<CheckpointLifecycle>(stats, 1);
    EXPECT_FALSE(progress.TakeMaterializedCheckpoint(12));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 3);
    auto materialized = progress.TakeMaterializedCheckpoint(20);
    ASSERT_TRUE(materialized);
    EXPECT_EQ(materialized->tokens, 16);
    EXPECT_EQ(materialized->computed_tokens, 20);
    EXPECT_EQ(stats->allocated, 1u);
    EXPECT_EQ(stats->materialized, 1u);
    EXPECT_EQ(stats->published, 0u);
    EXPECT_FALSE(progress.TakeMaterializedCheckpoint(24));
    materialized.reset();
    EXPECT_EQ(stats->cancelled, 1u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 4);
}

TEST(CheckpointMaterialization, RetriedPublicationDoesNotDuplicateLifecycleEvents) {
    auto stats = std::make_shared<CheckpointProductionStats>();
    auto output = std::make_shared<CheckpointLifecycle>(stats, 2);
    output->Materialize();
    output->Materialize();
    output->Publish();
    output->Publish();
    output.reset();
    EXPECT_EQ(stats->allocated, 1u);
    EXPECT_EQ(stats->extra_blocks, 2u);
    EXPECT_EQ(stats->materialized, 1u);
    EXPECT_EQ(stats->published, 1u);
    EXPECT_EQ(stats->cancelled, 0u);
}

TEST_F(GdsEviction, FreshLossChargesPublishedProductionOnlyAtCommit) {
    CheckProductionAttribution();
}

TEST_F(GdsEviction, RejectsFreshRetentionWithoutAdvancingWatermark) {
    Point r4{16}, r8{32};
    Publish(r4);
    Use(r4);
    Publish(r8);
    Fill();
    auto plan = Plan(1);
    ASSERT_TRUE(plan);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_FALSE(Holds(r8));
    EXPECT_TRUE(Holds(r4));
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
}

TEST_F(GdsEviction, FailedPressureAndReadOnlyProbesDoNotChangeTheCache) {
    Point r4{16}, r8{32};
    Publish(r4);
    Publish(r8);
    Use(r4);
    Use(r8);
    Fill();
    for (int retry = 0; retry < 3; ++retry) {
        EXPECT_FALSE(Plan(12));
        EXPECT_EQ(cache.Prober().ProbePrefix(r8.hashes).device.num_common_tokens, 32);
        EXPECT_TRUE(Holds(r4));
        EXPECT_TRUE(Holds(r8));
        EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
    }
    auto plan = Plan(1);
    ASSERT_TRUE(plan);
    EXPECT_DOUBLE_EQ(plan->gds->stats.watermark, 0.8);
}

TEST_F(GdsEviction, PinnedStateCannotBeUsedToFundCapacity) {
    Point r4{16}, r8{32};
    Publish(r4);
    Publish(r8);
    Use(r4);
    Use(r8);
    auto held = CacheCoordinatorTestAccess::MatchPrefix(cache, r8.hashes);
    Fill();
    auto plan = Plan(1);
    ASSERT_TRUE(plan);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_FALSE(Holds(r4));
    EXPECT_TRUE(Holds(r8));
}

TEST_F(GdsEviction, SamePageTailsUseTokenLineage) {
    Point shorter{17}, longer{19};
    Publish(longer);
    Publish(shorter);
    Use(longer);
    Use(shorter);
    Fill();
    auto plan = Plan(1);
    ASSERT_TRUE(plan);
    // Both own a State and a tail block. The deeper point gains only two tokens.
    EXPECT_EQ(Blocks(*plan), 2u);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_TRUE(Holds(shorter));
    EXPECT_FALSE(Holds(longer));
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0.25);
}

TEST_F(GdsEviction, RepeatedPressurePlansReuseResidentDependencies) {
    for (int pages = 1; pages <= 16; ++pages) Publish(Point{pages * kPage});
    Fill();
    const auto& index = CacheCoordinatorTestAccess::Store(cache).RecoveryDependencies();
    const auto before = index.GetStats();
    for (int repeat = 0; repeat < 8; ++repeat) {
        auto plan = Plan(3);
        ASSERT_TRUE(plan);
        EXPECT_GT(Blocks(*plan), 0);
    }
    EXPECT_EQ(index.GetStats().route_updates, before.route_updates);
    EXPECT_EQ(index.GetStats().range_visits, before.range_visits);
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
    CheckIndexedRoute(cache, Point{64});
}

TEST_F(GdsEviction, ResidencyChangesOnlyRefreshAffectedRecoveryPoints) {
    Point r4{16}, r8{32}, branch{32};
    branch.tokens[0] = -1;
    branch.Rehash();
    Publish(r4);
    Publish(r8);
    Publish(branch);
    auto& store = CacheCoordinatorTestAccess::Store(cache);
    const auto before = store.RecoveryDependencies().GetStats().route_updates;
    const CacheKey key{.group_id = 0, .content_hash = ContentHash{r8.hashes[5]}};
    const auto location = store.LocationOf(device, 0, key);
    ASSERT_TRUE(location);
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictAloneInto(cache, 0, *location,
                                                           &CacheEvictor::DeviceEvictionStats::admission, nullptr));
    EXPECT_EQ(store.RecoveryDependencies().GetStats().route_updates, before + 1);
    EXPECT_FALSE(store.RecoveryDependencies().At(r8.Hash()).ready);
    EXPECT_TRUE(store.RecoveryDependencies().At(r4.Hash()).ready);
    EXPECT_TRUE(store.RecoveryDependencies().At(branch.Hash()).ready);
    CheckIndexedRoute(cache, r4);
    CheckIndexedRoute(cache, r8);
    CheckIndexedRoute(cache, branch);
    auto replacement = device.AcquireBlock(0);
    cache.GroupPrefixIndex(0).Register(device, replacement, key, 1, 5, CacheBoundaryKind::kChunk, nullptr, nullptr);
    EXPECT_EQ(store.RecoveryDependencies().GetStats().route_updates, before + 2);
    CheckIndexedRoute(cache, r8);
    EXPECT_TRUE(store.RecoveryDependencies().At(r8.Hash()).device_ready);
}

TEST_F(GdsEviction, ClearDropsRangesAndRepublicationHasNoStaleDependencies) {
    const Point point{27};
    Publish(point);
    CheckIndexedRoute(cache, point);
    ASSERT_TRUE(cache.ClearCache());
    const auto& index = CacheCoordinatorTestAccess::Store(cache).RecoveryDependencies();
    EXPECT_EQ(index.GetStats().points, 0u);
    EXPECT_EQ(index.GetStats().ranges, 0u);
    EXPECT_EQ(index.GetStats().leaves, 0u);
    Publish(point);
    CheckIndexedRoute(cache, point);
    EXPECT_TRUE(index.At(point.Hash()).device_ready);
}

TEST_F(GdsEviction, MultipleBranchVictimsUpdateSharedRangeCapacity) {
    Point shared{16}, left{32}, right{32};
    right.tokens[16] = -1;
    right.Rehash();
    Publish(shared);
    Publish(left);
    Publish(right);
    Use(shared);
    Use(left);
    Use(right);
    CheckIndexedRoute(cache, shared);
    CheckIndexedRoute(cache, left);
    CheckIndexedRoute(cache, right);
    Fill();
    auto plan = Plan(11);
    ASSERT_TRUE(plan);
    EXPECT_EQ(Blocks(*plan), 15u);
    EXPECT_DOUBLE_EQ(plan->gds->stats.watermark, 0.8);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_FALSE(Holds(shared));
    EXPECT_FALSE(Holds(left));
    EXPECT_FALSE(Holds(right));
    EXPECT_EQ(CacheCoordinatorTestAccess::Store(cache).RecoveryDependencies().GetStats().ranges, 0u);
}

TEST(PersistentRecovery, DenseCheckpointsKeepSharedRangeMemoryBounded) {
    constexpr int points = 256;
    BlockPool device{points * 4, {1, 1}};
    auto cache = MakeCoordinator(Specs(), kPage, device, false, nullptr, nullptr, nullptr, false, false, 0);
    cache.ConfigureEviction(PrefixCacheEvictionConfig::Gds({0.25, 0}));
    for (int pages = 1; pages <= points; ++pages) PublishPoint(cache, Point{pages * kPage});
    const auto& index = CacheCoordinatorTestAccess::Store(cache).RecoveryDependencies();
    const auto stats = index.GetStats();
    EXPECT_EQ(stats.points, points);
    // A quadratic per-checkpoint history representation exceeds this budget.
    EXPECT_LT(stats.ranges, 8u * points);
    EXPECT_EQ(stats.leaves, 2u * points);
    CheckIndexedRoute(cache, Point{points * kPage});
    ASSERT_TRUE(cache.ClearCache());
    EXPECT_EQ(index.GetStats().ranges, 0u);
}

TEST(RecoveryCoverage, RangeSubtractionsMatchIndividualLiveConsumers) {
    constexpr std::size_t n = 73;
    for (unsigned seed = 0; seed < 32; ++seed) {
        std::mt19937 random(seed);
        std::vector<RecoveryInterval> ranges;
        std::vector<std::unordered_set<std::size_t>> consumers(n);
        std::vector<bool> eligible(n, true);
        for (std::size_t i = 0; i < n; ++i) eligible[i] = i % 5 != 0;
        for (std::size_t u = 1; u <= 64; ++u) {
            const auto first = random() % n;
            const auto end = first + 1 + random() % (n - first);
            ranges.push_back({first, end});
            for (auto i = first; i < end; ++i) consumers[i].insert(u);
        }
        std::vector<RecoveryCoverage::Leaf> initial(n);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t owners = 0;
            for (auto u : consumers[i]) owners ^= u;
            initial[i] = {eligible[i], static_cast<std::int64_t>(consumers[i].size()), owners};
        }
        RecoveryCoverage coverage{initial};
        std::vector<std::size_t> order(64);
        std::iota(order.begin(), order.end(), 1);
        std::shuffle(order.begin(), order.end(), random);
        for (auto u : order) {
            const auto range = ranges[u - 1];
            std::vector<std::size_t> exclusive, expected;
            coverage.ForEachExclusive(range, u, [&](auto i) { exclusive.push_back(i); });
            for (auto i = range.first; i < range.past_end; ++i)
                if (eligible[i] && consumers[i].size() == 1) expected.push_back(i);
            EXPECT_EQ(exclusive, expected);
            std::vector<std::size_t> changed, wanted;
            coverage.Remove(range, u, [&](std::size_t i, std::int64_t count, std::size_t owner) {
                changed.push_back(i);
                EXPECT_EQ(count, static_cast<std::int64_t>(consumers[i].size()) - 1);
                if (count == 1) {
                    auto remaining = consumers[i];
                    remaining.erase(u);
                    EXPECT_EQ(owner, *remaining.begin());
                } else
                    EXPECT_EQ(owner, 0u);
            });
            for (auto i = range.first; i < range.past_end; ++i) {
                consumers[i].erase(u);
                if (eligible[i] && consumers[i].size() <= 1) wanted.push_back(i);
                if (consumers[i].empty() && eligible[i]) {
                    coverage.Disable(i);
                    eligible[i] = false;
                }
            }
            EXPECT_EQ(changed, wanted);
        }
    }
}

class GdsWriteBack : public GdsEviction {
protected:
    GdsWriteBack() : GdsEviction(true, false) {}
    TierTransferManager transfers{cache};
    void CompareHostLoss(int incoming_tokens, int resident_tokens, bool reject_incoming, bool fresh) {
        Point incoming{incoming_tokens}, resident{resident_tokens};
        incoming.tokens[0] = 100;
        incoming.Rehash();
        Publish(incoming);
        Publish(resident);
        if (!fresh) Use(incoming);
        Use(resident);
        auto& store = CacheCoordinatorTestAccess::Store(cache);
        auto production = std::make_shared<CheckpointProductionStats>();
        auto receipt = std::make_shared<CheckpointLifecycle>(production, 1);
        receipt->Materialize();
        receipt->Publish();
        store.Checkpoints().AttachProduction(incoming.Hash(), receipt);
        const CacheKey state{.group_id = 1, .content_hash = resident.Hash()};
        {
            auto block = host.AcquireBlock(1);
            cache.Host().CacheHostBlock(block, state, std::nullopt);
        }
        ASSERT_TRUE(CacheCoordinatorTestAccess::EvictAloneInto(cache, 1, *store.LocationOf(device, 1, state),
                                                               &CacheEvictor::DeviceEvictionStats::admission, nullptr));
        Fill();
        std::vector<CacheBlockRef> host_pins;
        while (host.NumEmptyLcmBlocks()) host_pins.push_back(host.AcquireBlock(0));
        AdmissionCandidateConstraints constraints{&arena};
        // These history copies remain on Device; pressure asks to release
        // the incoming point, not to migrate the resident point's suffix.
        for (const auto& hash : resident.hashes) {
            const CacheKey key{.group_id = 0, .content_hash = ContentHash{hash}};
            constraints.protected_locations.insert(*store.LocationOf(device, 0, key));
        }
        std::array<BlockPlacementDemand, 2> demand{};
        for (auto& group : demand) group.blocks_by_bucket = {0};
        demand[1].any_bucket_blocks = 1;
        CacheEvictionPlan capacity{device, store.Groups()};
        auto plan = PlanGdsEviction(store, constraints, demand, capacity, &arena);
        ASSERT_TRUE(plan);
        EXPECT_EQ(cache.GdsStats().logical_evictions, 0u);
        EXPECT_EQ(production->fresh_rejected, 0u);
        CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
        EXPECT_EQ(production->fresh_rejected, fresh && reject_incoming ? 1u : 0u);
        EXPECT_EQ(production->rejected_extra_blocks, production->fresh_rejected);
        EXPECT_EQ(production->rejected_unmeasured, production->fresh_rejected);
        auto op = transfers.StartEvictionStores();
        if (reject_incoming)
            EXPECT_FALSE(op);
        else {
            ASSERT_TRUE(op);
            transfers.CompleteWriteBack(op->op_id);
        }
        EXPECT_EQ(cache.Prober().ProbePrefix(incoming.hashes).host.num_common_tokens,
                  reject_incoming ? 0 : incoming_tokens);
        EXPECT_EQ(cache.Prober().ProbePrefix(resident.hashes).host.num_common_tokens,
                  reject_incoming ? resident_tokens : 0);
        EXPECT_EQ(cache.GdsStats().logical_evictions, fresh ? 0u : 1u);
        EXPECT_EQ(cache.GdsStats().logical_rejections, fresh ? 1u : 0u);
        if (fresh) EXPECT_DOUBLE_EQ(cache.GdsStats().host_watermark, 0);
        EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
    }
};

TEST_F(GdsWriteBack, FreshLossChargesPublishedProductionOnlyAtCommit) {
    CompareHostLoss(16, 32, true, true);
}

TEST_F(GdsWriteBack, WriteBackBecomesRecoverableOnlyAfterAckAndKeepsDeviceSourceUnpinned) {
    Point r4{16}, r8{32};
    Publish(r4);
    Publish(r8);
    Use(r4);
    Use(r8);
    Fill();
    auto plan = Plan(1);
    ASSERT_TRUE(plan);
    auto moved = std::move(*plan);
    plan.reset();
    CacheCoordinatorTestAccess::CommitVictims(cache, moved);
    EXPECT_EQ(cache.GdsStats().committed_plans, 1u);
    EXPECT_EQ(device.NumEmptyLcmBlocks(), 1);
    auto op = transfers.StartEvictionStores();
    ASSERT_TRUE(op);
    EXPECT_FALSE(op->source_pinned);
    EXPECT_EQ(op->transfers.size(), 1u);
    EXPECT_EQ(cache.Prober().ProbePrefix(r8.hashes).host.num_common_tokens, 16);
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
    transfers.CompleteWriteBack(op->op_id);
    EXPECT_EQ(cache.Prober().ProbePrefix(r8.hashes).host.num_common_tokens, 32);
    EXPECT_TRUE(Holds(r4));
    // R8's Host State still needs Device KV. Removing R4 must preserve it.
    Fill();
    auto middle = Plan(1);
    ASSERT_TRUE(middle);
    ASSERT_EQ(Blocks(*middle), 1u);
    CacheCoordinatorTestAccess::CommitVictims(cache, *middle);
    EXPECT_TRUE(Holds(r4));
    auto suffix = transfers.StartEvictionStores();
    ASSERT_TRUE(suffix);
    EXPECT_EQ(cache.Prober().ProbePrefix(r8.hashes).host.num_common_tokens, 16);
    transfers.CompleteWriteBack(suffix->op_id);
    EXPECT_EQ(cache.Prober().ProbePrefix(r8.hashes).host.num_common_tokens, 32);
    EXPECT_EQ(cache.GdsStats().logical_evictions, 0u);
    EXPECT_EQ(cache.GdsStats().committed_plans, 2u);
}

TEST_F(GdsWriteBack, AdmissionPreviewPreservesTheNewPointDuringRequiredMigration) {
    Point r4{16}, r6{24}, r8{32};
    Publish(r4);
    Publish(r8);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(cache, tables, 24));
    Fill();
    const RequestProgress progress{
        .completed_pages =
            CheckpointPublication{.tokens = r6.tokens, .prefix_hashes = r6.hashes, .publish_state = true},
        .num_computed_tokens = 24};
    auto admitted = AdmitForTest(cache, tables, GroupDemand{.extent = DenseGrowth{4}}, progress);
    ASSERT_TRUE(admitted);
    EXPECT_TRUE(Holds(r6));
    auto& index = CacheCoordinatorTestAccess::Store(cache).RecoveryDependencies();
    EXPECT_TRUE(index.At(r6.Hash()).ready);
    auto stores = transfers.StartEvictionStores();
    ASSERT_TRUE(stores);
    transfers.CompleteWriteBack(stores->op_id);
    EXPECT_TRUE(index.At(r6.Hash()).ready);
    const auto stats = cache.GdsStats();
    EXPECT_EQ(stats.ready_units + stats.pending_units + stats.broken_units, stats.dependency_points);
    EXPECT_GT(stats.host_copies_reserved, 0u);
    cache.Free(tables);
}

TEST_F(GdsWriteBack, PendingWriteBackProtectsSharedHistoryAcrossAdmissions) {
    Point r4{16}, r8{32};
    Publish(r4);
    Publish(r8);
    Use(r4);
    Use(r8);
    Fill();
    auto first = Plan(1);
    ASSERT_TRUE(first);
    CacheCoordinatorTestAccess::CommitVictims(cache, *first);
    auto op = transfers.StartEvictionStores();
    ASSERT_TRUE(op);
    EXPECT_FALSE(op->source_pinned);
    Fill();
    auto second = Plan(1);
    ASSERT_TRUE(second);
    ASSERT_EQ(Blocks(*second), 1u);
    CacheCoordinatorTestAccess::CommitVictims(cache, *second);
    EXPECT_TRUE(Holds(r4));
    auto suffix = transfers.StartEvictionStores();
    ASSERT_TRUE(suffix);
    EXPECT_EQ(cache.Prober().ProbePrefix(r8.hashes).host.num_common_tokens, 16);
    transfers.CompleteWriteBack(op->op_id);
    EXPECT_EQ(cache.Prober().ProbePrefix(r8.hashes).host.num_common_tokens, 16);
    transfers.CompleteWriteBack(suffix->op_id);
    EXPECT_EQ(cache.Prober().ProbePrefix(r8.hashes).host.num_common_tokens, 32);
}

TEST_F(GdsWriteBack, ImportedHostStateProtectsSharedDeviceHistory) {
    Point r4{16}, r8{32};
    Publish(r8);
    Use(r8);
    {
        auto metadata = cache.TrackRecoveryMetadata(
            PromptPrefix{.prefix_hashes = r4.hashes, .tokens = r4.tokens, .max_tokens = 16}, 16);
        auto state = host.AcquireBlock(1);
        cache.Host().CacheHostBlock(state, CacheKey{.group_id = 1, .content_hash = r4.Hash()}, std::nullopt);
    }
    ASSERT_EQ(cache.Prober().ProbePrefix(r4.hashes).host.num_common_tokens, 16);
    Fill();
    auto plan = Plan(1);
    ASSERT_TRUE(plan);
    EXPECT_EQ(Blocks(*plan), 1u);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_EQ(cache.Prober().ProbePrefix(r4.hashes).host.num_common_tokens, 16);
}

TEST_F(GdsEviction, CancelledImportDropsItsMetadataWithoutChangingResidency) {
    const Point point{19};
    auto& tree = CacheCoordinatorTestAccess::Store(cache).Checkpoints();
    {
        auto metadata = cache.TrackRecoveryMetadata(
            PromptPrefix{.prefix_hashes = point.hashes, .tokens = point.tokens, .max_tokens = 19}, 19);
        EXPECT_EQ(tree.Size(), 1u);
        EXPECT_EQ(cache.Prober().ProbePrefix(point.hashes).device.num_common_tokens, 0);
        EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
    }
    EXPECT_EQ(tree.Size(), 0u);
    EXPECT_EQ(tree.PrefixSegments(), 0u);
}

TEST_F(GdsWriteBack, HostSuffixChangesInvalidateAndRepairIndexedRoutes) {
    const Point point{32};
    Publish(point);
    auto& store = CacheCoordinatorTestAccess::Store(cache);
    const CacheKey seventh{.group_id = 0, .content_hash = ContentHash{point.hashes[6]}};
    const CacheKey eighth{.group_id = 0, .content_hash = ContentHash{point.hashes[7]}};
    {
        auto block = host.AcquireBlock(0);
        cache.Host().CacheHostBlock(block, seventh, std::nullopt);
    }
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictAloneInto(cache, 0, *store.LocationOf(device, 0, seventh),
                                                           &CacheEvictor::DeviceEvictionStats::admission, nullptr));
    EXPECT_FALSE(store.RecoveryDependencies().At(point.Hash()).ready);
    CheckIndexedRoute(cache, point);
    {
        auto block = host.AcquireBlock(0);
        cache.Host().CacheHostBlock(block, eighth, std::nullopt);
    }
    EXPECT_TRUE(store.RecoveryDependencies().At(point.Hash()).ready);
    EXPECT_FALSE(store.RecoveryDependencies().At(point.Hash()).device_ready);
    CheckIndexedRoute(cache, point);
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(cache, seventh));
    CheckIndexedRoute(cache, point);
    EXPECT_FALSE(store.RecoveryDependencies().At(point.Hash()).ready);
}

TEST(PersistentRecovery, SlidingSinkAndPartialTailsMatchKeyBasedRoutes) {
    for (int window : {1, 3, 9}) {
        for (int grain : {1, 2, 4}) {
            auto specs = Specs();
            specs.push_back(CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                                           .sliding_window = window,
                                           .sink_tokens = window >= kPage ? 2 : 0,
                                           .cache_blocks_per_lcm_block = 1,
                                           .block_granularity = grain});
            BlockPool device{128, {1, 1, 1}}, host{128, {1, 1, 1}};
            auto cache = MakeCoordinator(specs, kPage, device, false, &host, nullptr, nullptr, false, true, 0);
            cache.ConfigureEviction(PrefixCacheEvictionConfig::Gds({0.25, 0}));
            for (int tokens : {15, 16, 19, 24, 27}) {
                const Point point{tokens};
                PublishPoint(cache, point);
                CheckIndexedRoute(cache, point);
            }
            ASSERT_TRUE(cache.ClearCache());
            EXPECT_EQ(CacheCoordinatorTestAccess::Store(cache).RecoveryDependencies().GetStats().ranges, 0u);
        }
    }
}

TEST_F(GdsWriteBack, HostWithoutCapacityRequiresJointLogicalEviction) {
    Point r4{16}, r8{32};
    Publish(r4);
    Publish(r8);
    Use(r4);
    Use(r8);
    Fill();
    std::vector<CacheBlockRef> host_pins;
    while (host.NumEmptyLcmBlocks() > 0) host_pins.push_back(host.AcquireBlock(0));
    auto plan = Plan(1);
    ASSERT_TRUE(plan);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_FALSE(transfers.StartEvictionStores());
    EXPECT_EQ(device.NumEmptyLcmBlocks(), 5);
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
    EXPECT_GT(cache.GdsStats().host_watermark, 0);
    EXPECT_EQ(cache.GdsStats().logical_evictions, 1u);
    EXPECT_TRUE(Holds(r4));
    EXPECT_EQ(cache.Prober().ProbePrefix(r8.hashes).device.num_common_tokens, 16);
}

TEST_F(GdsWriteBack, ExplicitPeerDemotionIsDroppedWithoutWriteBack) {
    Point r4{16}, r8{32};
    Publish(r4);
    Publish(r8);
    Use(r4);
    Use(r8);
    cache.SetCheckpointDemotion(true);
    const std::array keys{CacheKey{.group_id = 1, .content_hash = r4.Hash()}};
    const auto demoted = cache.DemoteEntriesWithEvents(keys, CacheCoordinator::DemotionCause::kPeerHandoff);
    ASSERT_EQ(demoted.size(), 1u);
    EXPECT_EQ(demoted[0].key, keys[0]);
    EXPECT_EQ(demoted[0].tier, CacheTier::kDevice);
    EXPECT_EQ(demoted[0].cause, CacheCoordinator::DemotionCause::kPeerHandoff);
    Fill();
    auto plan = Plan(1);
    ASSERT_TRUE(plan);
    ASSERT_EQ(Blocks(*plan), 1u);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_FALSE(Holds(r4));
    EXPECT_TRUE(Holds(r8));
    EXPECT_FALSE(transfers.StartEvictionStores());
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
}

TEST_F(GdsEviction, SupersededRetentionHintDoesNotOverrideDeviceGds) {
    Point r4{16}, r8{32};
    Publish(r4);
    Publish(r8);
    Use(r4);
    Use(r8);
    cache.SetCheckpointDemotion(true);
    const std::array keys{CacheKey{.group_id = 1, .content_hash = r4.Hash()}};
    EXPECT_EQ(cache.DemoteEntries(keys, CacheCoordinator::DemotionCause::kSupersededCheckpoint), 0);
    Fill();
    auto plan = Plan(1);
    ASSERT_TRUE(plan);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_TRUE(Holds(r4));
    EXPECT_FALSE(Holds(r8));
}

TEST_F(GdsWriteBack, AdmissionProtectsHostHitsBeforeTheyBecomePins) {
    Point hit{16}, other{32};
    other.tokens[0] = 100;
    other.Rehash();
    Publish(hit);
    Publish(other);
    Use(hit);
    Use(other);
    auto& store = CacheCoordinatorTestAccess::Store(cache);
    const CacheKey state{.group_id = 1, .content_hash = hit.Hash()};
    {
        auto block = host.AcquireBlock(1);
        cache.Host().CacheHostBlock(block, state, std::nullopt);
    }
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictAloneInto(cache, 1, *store.LocationOf(device, 1, state),
                                                           &CacheEvictor::DeviceEvictionStats::admission, nullptr));
    Fill();
    std::vector<CacheBlockRef> host_pins;
    while (host.NumEmptyLcmBlocks()) host_pins.push_back(host.AcquireBlock(0));
    auto prefix = cache.Prober().ProbePrefix(hit.hashes);
    std::array<GroupDemand, 2> groups{};
    auto constraints = BuildAdmissionCandidateConstraints(store, groups, std::nullopt, prefix, {}, &arena, nullptr);
    EXPECT_TRUE(constraints.host_protected_locations.contains(*store.LocationOf(host, 1, state)));
    std::array<BlockPlacementDemand, 2> demand{};
    for (auto& group : demand) group.blocks_by_bucket = {0};
    demand[1].any_bucket_blocks = 1;
    CacheEvictionPlan capacity{device, store.Groups()};
    auto plan = PlanGdsEviction(store, constraints, demand, capacity, &arena);
    ASSERT_TRUE(plan);
    auto acquired = CacheCoordinatorTestAccess::MatchPrefix(cache, hit.hashes);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_EQ(acquired.host.num_common_tokens, 16);
    EXPECT_TRUE(cache.GroupPrefixIndex(1).Contains(host, state));
}

TEST_F(GdsWriteBack, HostFullCanCancelCheaperIncomingMigration) {
    CompareHostLoss(16, 32, true, false);
}

TEST_F(GdsWriteBack, HostFullCanEvictCheaperResidentAndReserveMigration) {
    CompareHostLoss(32, 16, false, false);
}

TEST_F(GdsWriteBack, HostPressureRejectsCheaperFreshMigrationWithoutAging) {
    CompareHostLoss(16, 32, true, true);
}

TEST_F(GdsWriteBack, UnreservedWriteBackIsNotARecoveryPromise) {
    Point point{16};
    Publish(point);
    auto& store = CacheCoordinatorTestAccess::Store(cache);
    const CacheKey state{.group_id = 1, .content_hash = point.Hash()};
    CacheEvictor::EvictedSegment optional;
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictAloneInto(cache, 1, *store.LocationOf(device, 1, state),
                                                           &CacheEvictor::DeviceEvictionStats::admission, &optional));
    ASSERT_EQ(optional.members.size(), 1u);
    EXPECT_TRUE(store.PendingHostKeys().contains(state));
    EXPECT_FALSE(store.HasReservedHostKey(state));
    EXPECT_FALSE(store.RecoveryDependencies().At(point.Hash()).pending);
    EXPECT_EQ(cache.Prober().ProbePrefix(point.hashes).host.num_common_tokens, 0);
}

TEST_F(GdsWriteBack, OverlayChangesBothTiersWithoutChangingTheLiveIndex) {
    Point point{19};
    Publish(point);
    auto& store = CacheCoordinatorTestAccess::Store(cache);
    const CacheKey state{.group_id = 1, .content_hash = point.Hash()};
    RecoveryIndex::View view{store.RecoveryDependencies()};
    const std::array changes{RecoveryIndex::View::Change{state, CacheTier::kHost, true},
                             RecoveryIndex::View::Change{state, CacheTier::kDevice, false}};
    view.Apply(changes);
    EXPECT_TRUE(view.Resolve(point.Hash(), false).ready);
    EXPECT_FALSE(view.Resolve(point.Hash(), false).device_ready);
    const RecoveryIndex::View::Change lost{state, CacheTier::kHost, false};
    view.Apply(std::span{&lost, 1});
    EXPECT_FALSE(view.Resolve(point.Hash(), false).ready);
    EXPECT_TRUE(store.RecoveryDependencies().At(point.Hash()).device_ready);
    EXPECT_TRUE(Holds(point));
}

TEST_F(GdsWriteBack, HostOnlyPressureUsesTheSameLogicalRetentionDecision) {
    Point point{16};
    Publish(point);
    Use(point);
    Fill();
    auto migration = Plan(5);
    ASSERT_TRUE(migration);
    CacheCoordinatorTestAccess::CommitVictims(cache, *migration);
    auto op = transfers.StartEvictionStores();
    ASSERT_TRUE(op);
    transfers.CompleteWriteBack(op->op_id);
    ASSERT_EQ(cache.Prober().ProbePrefix(point.hashes).host.num_common_tokens, 16);
    std::vector<CacheBlockRef> host_pins;
    while (host.NumEmptyLcmBlocks()) host_pins.push_back(host.AcquireBlock(0));
    const std::array<std::uint32_t, 1> groups{1};
    const std::array<std::int32_t, 1> buckets{0};
    const auto before = cache.GdsStats();
    auto allocated = cache.Host().AcquireHostBlocks(groups, buckets);
    ASSERT_EQ(allocated.blocks.size(), 1u);
    EXPECT_TRUE(allocated.blocks[0]);
    EXPECT_EQ(cache.Prober().ProbePrefix(point.hashes).host.num_common_tokens, 0);
    EXPECT_EQ(cache.GdsStats().logical_evictions, 1u);
    EXPECT_GT(cache.GdsStats().host_watermark, 0);
    EXPECT_EQ(cache.GdsStats().committed_plans, before.committed_plans + 1);
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), before.watermark);
}

TEST_F(GdsWriteBack, FailedJointPlanLeavesResidencyAndWatermarksUntouched) {
    Point point{19};
    Publish(point);
    Use(point);
    Fill();
    std::vector<CacheBlockRef> host_pins;
    while (host.NumEmptyLcmBlocks()) host_pins.push_back(host.AcquireBlock(0));
    const auto before = cache.GdsStats();
    auto impossible = Plan(64);
    EXPECT_FALSE(impossible);
    EXPECT_TRUE(Holds(point));
    EXPECT_EQ(cache.GdsStats().logical_evictions, before.logical_evictions);
    EXPECT_EQ(cache.GdsStats().watermark, before.watermark);
    EXPECT_EQ(cache.GdsStats().host_watermark, before.host_watermark);
    EXPECT_FALSE(transfers.StartEvictionStores());
}

TEST(GdsCapacityCompleteness, PinnedStateDoesNotReserveUnownedHistory) {
    for (bool write_back : {false, true}) {
        for (bool gds : {false, true}) {
            SCOPED_TRACE(testing::Message() << "write_back=" << write_back << " gds=" << gds);
            BlockPool device{16, {1, 1}}, host{16, {1, 1}};
            auto cache =
                MakeCoordinator(Specs(), kPage, device, false, &host, nullptr, nullptr, !write_back, write_back, 0);
            if (gds) cache.ConfigureEviction(PrefixCacheEvictionConfig::Gds({0.25, 0}));
            const Point point{16};
            PublishPoint(cache, point);
            auto state = cache.GroupPrefixIndex(1).Find(device, CacheKey{.group_id = 1, .content_hash = point.Hash()});
            ASSERT_TRUE(state);
            std::vector<CacheBlockRef> device_pins, host_pins;
            while (device.NumEmptyLcmBlocks()) device_pins.push_back(device.AcquireBlock(0));
            while (host.NumEmptyLcmBlocks()) host_pins.push_back(host.AcquireBlock(0));
            std::vector<BlockTable> tables(2);
            const auto admission = AdmitForTest(cache, tables, kPage);
            // Legacy write-back keeps the whole segment while its State is
            // held. Its write-through path can already reclaim this history.
            if (!gds && write_back) {
                EXPECT_FALSE(admission);
                continue;
            }
            ASSERT_TRUE(admission);
            EXPECT_TRUE(
                cache.GroupPrefixIndex(1).Contains(device, CacheKey{.group_id = 1, .content_hash = point.Hash()}));
            if (gds) {
                EXPECT_GT(cache.GdsStats().fallback_blocks, 0u);
                EXPECT_EQ(cache.GdsStats().fallback_units, 1u);
                EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
                EXPECT_DOUBLE_EQ(cache.GdsStats().host_watermark, 0);
            }
            TierTransferManager transfers{cache};
            EXPECT_FALSE(transfers.StartEvictionStores());
            cache.Free(tables);
        }
    }
}

TEST_F(GdsWriteBack, PhysicalFallbackAccountsForSharedLossesAndPreservesProtectedKeys) {
    const Point r4{16}, r8{32};
    Publish(r4);
    Publish(r8);
    auto s4 = cache.GroupPrefixIndex(1).Find(device, CacheKey{.group_id = 1, .content_hash = r4.Hash()});
    auto s8 = cache.GroupPrefixIndex(1).Find(device, CacheKey{.group_id = 1, .content_hash = r8.Hash()});
    ASSERT_TRUE(s4);
    ASSERT_TRUE(s8);
    Fill();
    std::vector<CacheBlockRef> host_pins;
    while (host.NumEmptyLcmBlocks()) host_pins.push_back(host.AcquireBlock(0));
    auto& store = CacheCoordinatorTestAccess::Store(cache);
    const CacheKey protected_key{.group_id = 0, .content_hash = ContentHash{r4.hashes.front()}};
    AdmissionCandidateConstraints constraints{&arena};
    constraints.protected_locations.insert(*store.LocationOf(device, 0, protected_key));
    std::array<BlockPlacementDemand, 2> demand{};
    for (auto& group : demand) group.blocks_by_bucket = {0};
    demand[1].any_bucket_blocks = 8;
    CacheEvictionPlan impossible{device, store.Groups()};
    EXPECT_FALSE(PlanGdsEviction(store, constraints, demand, impossible, &arena));
    EXPECT_EQ(cache.GdsStats().fallback_units, 0u);
    EXPECT_EQ(cache.Prober().ProbePrefix(r8.hashes).device.num_common_tokens, 32);

    demand[1].any_bucket_blocks = 7;
    CacheEvictionPlan capacity{device, store.Groups()};
    auto plan = PlanGdsEviction(store, constraints, demand, capacity, &arena);
    ASSERT_TRUE(plan);
    EXPECT_EQ(cache.GdsStats().fallback_units, 0u);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_EQ(device.NumEmptyLcmBlocks(), 7);
    EXPECT_TRUE(cache.GroupPrefixIndex(0).Contains(device, protected_key));
    EXPECT_TRUE(Holds(r4));
    EXPECT_TRUE(Holds(r8));
    EXPECT_EQ(cache.GdsStats().fallback_units, 2u);
    EXPECT_EQ(cache.GdsStats().fallback_blocks, 7u);
    EXPECT_FALSE(store.RecoveryDependencies().At(r4.Hash()).Available());
    EXPECT_FALSE(store.RecoveryDependencies().At(r8.Hash()).Available());
    EXPECT_DOUBLE_EQ(cache.GdsWatermark(), 0);
    EXPECT_DOUBLE_EQ(cache.GdsStats().host_watermark, 0);
    EXPECT_FALSE(transfers.StartEvictionStores());
}

TEST(GdsCapacityCompleteness, EveryFeasibleLegacyAdmissionAlsoFitsGds) {
    for (bool write_back : {false, true}) {
        for (int hold_mask = 0; hold_mask < 8; ++hold_mask) {
            for (int tokens : {4, 16, 40}) {
                bool lru_fits = false;
                for (bool gds : {false, true}) {
                    SCOPED_TRACE(testing::Message() << "write_back=" << write_back << " held=" << hold_mask
                                                    << " tokens=" << tokens << " gds=" << gds);
                    BlockPool device{64, {1, 1}}, host{16, {1, 1}};
                    auto cache = MakeCoordinator(Specs(), kPage, device, false, &host, nullptr, nullptr, !write_back,
                                                 write_back, 0);
                    if (gds) cache.ConfigureEviction(PrefixCacheEvictionConfig::Gds({0.25, 0}));
                    std::array<Point, 3> points{Point{11}, Point{23}, Point{32}};
                    points[2].tokens.back() = 100;
                    points[2].Rehash();
                    for (const auto& point : points) PublishPoint(cache, point);
                    std::vector<CacheBlockRef> held, device_pins, host_pins;
                    for (std::size_t i = 0; i < points.size(); ++i)
                        if (hold_mask & (1 << i))
                            held.push_back(cache.GroupPrefixIndex(1).Find(
                                device, CacheKey{.group_id = 1, .content_hash = points[i].Hash()}));
                    while (device.NumEmptyLcmBlocks()) device_pins.push_back(device.AcquireBlock(0));
                    while (host.NumEmptyLcmBlocks()) host_pins.push_back(host.AcquireBlock(0));
                    std::vector<BlockTable> tables(2);
                    const auto admission = AdmitForTest(cache, tables, tokens);
                    if (!gds)
                        lru_fits = admission.has_value();
                    else if (lru_fits)
                        EXPECT_TRUE(admission);
                    cache.Free(tables);
                }
            }
        }
    }
}

TEST_F(GdsWriteBack, CompactionMayRemoveADuplicateButNeverTheLastRoute) {
    Point point{16};
    Publish(point);
    auto& store = CacheCoordinatorTestAccess::Store(cache);
    const CacheKey state{.group_id = 1, .content_hash = point.Hash()};
    AdmissionCandidateConstraints constraints{&arena};
    EXPECT_FALSE(PlanTieredCompaction(store, constraints, state, &arena));
    {
        auto block = host.AcquireBlock(1);
        cache.Host().CacheHostBlock(block, state, std::nullopt);
    }
    auto safe = PlanTieredCompaction(store, constraints, state, &arena);
    ASSERT_TRUE(safe);
    CacheCoordinatorTestAccess::CommitVictims(cache, *safe);
    EXPECT_FALSE(Holds(point));
    EXPECT_EQ(cache.Prober().ProbePrefix(point.hashes).host.num_common_tokens, 16);
    EXPECT_EQ(cache.GdsStats().logical_evictions, 0u);
    EXPECT_FALSE(transfers.StartEvictionStores());
}

TEST_F(GdsWriteBack, SupersededHintDoesNotOverrideHostGds) {
    Point point{16};
    Publish(point);
    const CacheKey key{.group_id = 1, .content_hash = point.Hash()};
    {
        auto block = host.AcquireBlock(1);
        cache.Host().CacheHostBlock(block, key, std::nullopt);
    }
    cache.SetCheckpointDemotion(true);
    EXPECT_EQ(cache.DemoteEntries(std::span{&key, 1}, CacheCoordinator::DemotionCause::kSupersededCheckpoint), 0);
    EXPECT_FALSE(cache.GroupPrefixIndex(1).Inspect(host, key)->metadata.demoted);
    EXPECT_FALSE(cache.GroupPrefixIndex(1).Inspect(device, key)->metadata.demoted);
}

class GdsWriteThrough : public GdsEviction {
protected:
    GdsWriteThrough() : GdsEviction(false, true) {}
    TierTransferManager transfers{cache};
};

TEST_F(GdsWriteThrough, OptionalCopyCannotEvictAnotherPointsOnlyHostRoute) {
    Point old{16}, incoming{20};
    incoming.tokens[0] = 100;
    incoming.Rehash();
    Publish(old);
    auto copy = transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck, HostStoreCause::kPublication);
    ASSERT_TRUE(copy);
    ASSERT_TRUE(copy->source_pinned);
    transfers.CompleteWriteBack(copy->op_id);
    auto& store = CacheCoordinatorTestAccess::Store(cache);
    const CacheKey key{.group_id = 1, .content_hash = old.Hash()};
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictAloneInto(cache, 1, *store.LocationOf(device, 1, key),
                                                           &CacheEvictor::DeviceEvictionStats::admission, nullptr));
    std::vector<CacheBlockRef> host_pins;
    while (host.NumEmptyLcmBlocks()) host_pins.push_back(host.AcquireBlock(0));
    Publish(incoming);
    auto optional = transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck, HostStoreCause::kPublication);
    if (optional) transfers.CompleteWriteBack(optional->op_id);
    EXPECT_TRUE(Holds(incoming));
    EXPECT_EQ(cache.Prober().ProbePrefix(old.hashes).host.num_common_tokens, 16);
    EXPECT_EQ(cache.GdsStats().logical_evictions, 0u);
    EXPECT_GT(cache.GdsStats().host_copy_rejections, 0u);
}

TEST_F(GdsWriteThrough, SourceRemainsPinnedUntilAckAndCopiesAreNotResubmitted) {
    Point point{19};
    Publish(point);
    auto copy = transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck, HostStoreCause::kPublication);
    ASSERT_TRUE(copy);
    EXPECT_TRUE(copy->source_pinned);
    EXPECT_FALSE(transfers.StartPendingStores(StoreSourceGuard::kPinnedUntilAck, HostStoreCause::kPublication));
    Fill();
    EXPECT_FALSE(Plan(1));
    transfers.CompleteWriteBack(copy->op_id);
    auto plan = Plan(1);
    ASSERT_TRUE(plan);
    CacheCoordinatorTestAccess::CommitVictims(cache, *plan);
    EXPECT_EQ(cache.GdsStats().logical_evictions, 0u);
    EXPECT_FALSE(transfers.StartEvictionStores());
    EXPECT_TRUE(CacheCoordinatorTestAccess::Store(cache).RecoveryDependencies().At(point.Hash()).ready);
}

TEST(CheckpointMaterialization, TailResumeAndExtraOutputFailureKeepTheSameTarget) {
    const std::array groups{
        CacheGroupConfig{.group_id = "state", .attn_kind = AttnKind::kState},
        CacheGroupConfig{
            .group_id = "window", .attn_kind = AttnKind::kSlidingWindow, .sliding_window_tokens = 5, .sink_tokens = 2}};
    const auto interior = PlanCheckpointMaterialization(16, 8, 12, 0, 16, 4, {true}, groups);
    EXPECT_TRUE(interior.NeedsExtraOutput());
    EXPECT_EQ(interior.checkpoint_offset, 8);
    EXPECT_EQ(interior.groups[0].extra_blocks, 1);
    EXPECT_EQ(interior.groups[1].history_begin, 12);
    EXPECT_EQ(interior.groups[1].history_end, 16);
    EXPECT_EQ(interior.groups[1].sink_end, 2);
    const auto fallback = interior.WithoutExtraOutput();
    EXPECT_EQ(fallback.target_tokens, 16);
    EXPECT_EQ(fallback.chunk_tokens, 8);
    EXPECT_FALSE(fallback.NeedsExtraOutput());
    EXPECT_EQ(fallback.groups[0].extra_blocks, 0);
    EXPECT_EQ(fallback.groups[1].history_begin, 12);
    const auto tail_resume = PlanCheckpointMaterialization(16, 9, 12, 0, 16, 4, {true}, groups);
    EXPECT_EQ(tail_resume.chunk_tokens, 7);
    EXPECT_FALSE(tail_resume.NeedsExtraOutput());
    const auto budget_cut = PlanCheckpointMaterialization(24, 8, 15, 0, 16, 4, {true}, groups);
    EXPECT_EQ(budget_cut.chunk_tokens, 12);
    EXPECT_FALSE(budget_cut.NeedsExtraOutput());
}

TEST(GdsCost, ValidatesExplicitCoefficientsAndWidensTokenArithmetic) {
    EXPECT_THROW(PrefixCacheEvictionConfig::Parse("gds", std::nullopt), std::invalid_argument);
    EXPECT_THROW(PrefixCacheEvictionConfig::Gds({0, 1}), std::invalid_argument);
    EXPECT_THROW(PrefixCacheEvictionConfig::Gds({1, -1}), std::invalid_argument);
    EXPECT_THROW(PrefixCacheEvictionConfig::Gds({std::numeric_limits<double>::infinity(), 0}), std::invalid_argument);
    EXPECT_THROW(PrefixCacheEvictionConfig::Gds({1, std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
    const PrefillCost cost{1, 2};
    EXPECT_DOUBLE_EQ(cost.Between(1000000, 1000001), 1 + 2.0 * 2000001 / 200000);
}
TEST_F(GdsWriteBack, RouteOverlayRepairsSharedHistoryAndRollsBackByValue) {
    Point ancestor{16}, left{32}, right{32};
    right.tokens[20] = 100;
    right.Rehash();
    Publish(ancestor);
    Publish(left);
    Publish(right);
    const auto& index = CacheCoordinatorTestAccess::Store(cache).RecoveryDependencies();
    RecoveryIndex::View view{index};
    const auto change = [&](const CacheKey& key, CacheTier tier, bool present) {
        const RecoveryIndex::View::Change event{key, tier, present};
        view.Apply(std::span{&event, 1});
    };
    const CacheKey shared{.group_id = 0, .content_hash = ContentHash{left.hashes[0]}};
    change(shared, CacheTier::kDevice, false);
    for (const Point* point : {&ancestor, &left, &right}) {
        EXPECT_FALSE(view.At(point->Hash()).Available());
        EXPECT_TRUE(index.At(point->Hash()).device_ready);
    }
    change(shared, CacheTier::kDevice, true);
    for (const Point* point : {&ancestor, &left, &right}) EXPECT_TRUE(view.At(point->Hash()).device_ready);
    const CacheKey state{.group_id = 1, .content_hash = left.Hash()};
    change(state, CacheTier::kHost, true);
    change(state, CacheTier::kDevice, false);
    EXPECT_TRUE(view.At(left.Hash()).ready);
    EXPECT_FALSE(view.At(left.Hash()).device_ready);
    EXPECT_TRUE(view.At(right.Hash()).device_ready);
    auto saved = view;
    change(state, CacheTier::kHost, false);
    EXPECT_FALSE(view.At(left.Hash()).Available());
    EXPECT_TRUE(saved.At(left.Hash()).ready);
    view = saved;
    EXPECT_TRUE(view.At(left.Hash()).ready);
    EXPECT_FALSE(view.At(left.Hash()).device_ready);
    change(state, CacheTier::kDevice, true);
    EXPECT_TRUE(view.At(left.Hash()).device_ready);
    EXPECT_FALSE(saved.At(left.Hash()).device_ready);
    EXPECT_TRUE(index.At(left.Hash()).device_ready);
}

TEST_F(GdsWriteBack, BatchedOverlayChangesKeepExactTailRoutesAndCopyAlternativesSeparate) {
    const Point point{19};
    Publish(point);
    const auto& store = CacheCoordinatorTestAccess::Store(cache);
    RecoveryIndex::View view{store.RecoveryDependencies()};
    const CacheKey state{.group_id = 1, .content_hash = point.Hash()};
    const CacheKey tail{.group_id = 0, .content_hash = point.Hash()};
    const std::array remove{RecoveryIndex::View::Change{state, CacheTier::kDevice, false},
                            RecoveryIndex::View::Change{tail, CacheTier::kDevice, false}};
    view.Apply(remove);
    EXPECT_FALSE(view.At(point.Hash()).Available());
    EXPECT_TRUE(view.Resolve(point.Hash(), true).Available());
    EXPECT_FALSE(view.At(point.Hash()).Available());
    const std::array migrate{RecoveryIndex::View::Change{state, CacheTier::kHost, true},
                             RecoveryIndex::View::Change{tail, CacheTier::kHost, true}};
    view.Apply(migrate);
    const auto& route = view.At(point.Hash());
    ASSERT_TRUE(route.ready);
    EXPECT_FALSE(route.device_ready);
    EXPECT_EQ(route.host_paths.size(), 2u);
    // An unchanged batch preserves readiness; repair and removal in one
    // transaction must resolve its final contents rather than an intermediate state.
    view.Apply(migrate);
    const std::array replace{RecoveryIndex::View::Change{state, CacheTier::kHost, false},
                             RecoveryIndex::View::Change{state, CacheTier::kHost, true}};
    view.Apply(replace);
    EXPECT_TRUE(view.At(point.Hash()).ready);
    EXPECT_TRUE(store.RecoveryDependencies().At(point.Hash()).device_ready);
}

}  // namespace
}  // namespace tokenspeed::test
