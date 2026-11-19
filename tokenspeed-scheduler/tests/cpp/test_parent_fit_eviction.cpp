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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cache/coordinator/admission_planner.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache_test_access.h"
#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kPage = 8;

// The walk alone: what the parent-fit shortcut must reproduce.
class WalkPlanner final : public AdmissionPlanner {
public:
    using AdmissionPlanner::AdmissionPlanner;

protected:
    std::optional<std::vector<VictimCandidate>> candidatesThroughParentFit(
        std::span<const BlockPlacementDemand>, const AdmissionCandidateConstraints&) const override {
        return std::nullopt;
    }
};

// The planner admissions use, counting the plans the shortcut selected.
class CountingPlanner final : public AdmissionPlanner {
public:
    using AdmissionPlanner::AdmissionPlanner;
    static inline std::size_t shortcuts = 0;

protected:
    std::optional<std::vector<VictimCandidate>> candidatesThroughParentFit(
        std::span<const BlockPlacementDemand> block_demands,
        const AdmissionCandidateConstraints& constraints) const override {
        std::optional<std::vector<VictimCandidate>> through_fit =
            AdmissionPlanner::candidatesThroughParentFit(block_demands, constraints);
        shortcuts += through_fit.has_value() ? 1 : 0;
        return through_fit;
    }
};

// DeepSeek V4.1's layout in miniature: two cached full-history groups and two
// replayable windows that never cache, so a prefill needs empty parents.
SchedulerConfig FourGroupConfig(std::int32_t parents, std::int32_t shards) {
    SchedulerConfig config{};
    config.prefix_granularity = kPage;
    config.device_allocator.total_pages = parents + 1;
    config.host_allocator.total_pages = 1;
    config.snapshot_allocator.total_pages = 1;
    config.max_retracted_requests = 0;
    config.max_scheduled_tokens = 64;
    config.max_batch_size = 8;
    config.max_resident_requests = 8;
    config.decode_input_tokens = 1;
    config.overlap_schedule_depth = 0;
    config.disable_l2_cache = true;
    config.enable_l3_storage = false;
    config.enable_kv_cache_events = false;
    config.role = Role::kFused;
    config.disable_prefix_cache = false;
    config.prefix_replay_tokens = kPage;
    config.checkpoint_retention_requests = 0;
    const auto group = [&](const char* id, std::int32_t granularity, std::int32_t packing, std::int32_t group_shards,
                           std::optional<std::int32_t> window) {
        config.cache_groups.push_back(CacheGroupConfig{
            .group_id = id,
            .block_granularity = granularity,
            .total_pages = parents * packing + 1,
            .cache_blocks_per_lcm_block = packing,
            .attn_kind = window ? AttnKind::kSlidingWindow : AttnKind::kFull,
            .sliding_window_tokens = window,
            .replayable = window.has_value(),
            .shard_count = group_shards,
        });
    };
    group("swa", 4, 1, 1, 8);
    group("global_r2", 8, 4, shards, std::nullopt);
    group("global_r1", 4, 6, shards, std::nullopt);
    group("compressor_tail", 2, 5, 1, 2);
    return config;
}

using Member = std::tuple<std::uint32_t, std::int32_t, std::int32_t, bool, bool>;

std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>> CopyInsOf(const AdmissionPlan& plan) {
    std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>> copy_ins;
    for (const std::vector<CopyInReservation>& group : plan.copy_ins) {
        std::vector<std::pair<std::int32_t, std::int32_t>>& reservations = copy_ins.emplace_back();
        for (const CopyInReservation& reservation : group) {
            reservations.emplace_back(reservation.slot, reservation.bucket);
        }
    }
    return copy_ins;
}

std::vector<std::vector<Member>> VictimsOf(const AdmissionPlan& plan) {
    std::vector<std::vector<Member>> victims;
    for (const VictimSet& set : plan.victims) {
        std::vector<Member>& members = victims.emplace_back();
        for (const VictimSet::Member& member : set.members) {
            members.emplace_back(member.group_id, member.location.lcm_block_id, member.location.slot_index,
                                 member.in_stranded_suffix, member.is_request_reclaim);
        }
    }
    return victims;
}

struct Request {
    std::vector<std::int32_t> tokens;
    std::vector<std::string> hashes;
    std::vector<BlockTable> tables;
    std::int32_t hit_pages{0};
    std::int32_t computed{0};
    std::uint64_t access_epoch{0};
};

class ParentFitEvictionTest : public ::testing::TestWithParam<std::int32_t> {
protected:
    ParentFitEvictionTest() : scheduler_{FourGroupConfig(/*parents=*/40, GetParam())} {}

    CacheCoordinator& Coordinator() { return SchedulerTestAccess::Coordinator(scheduler_); }
    const BlockPool& Pool() { return SchedulerTestAccess::Pool(scheduler_); }

    // Plans one admission with the shortcut and with the walk alone, expects
    // the same plan, and commits it.
    bool Admit(Request& request, std::int32_t tokens) {
        const bool first_chunk = request.computed == 0;
        const std::span<const std::string> probed =
            first_chunk ? std::span<const std::string>{request.hashes}.first(
                              static_cast<std::size_t>(request.tokens.size() - 1) / kPage)
                        : std::span<const std::string>{};
        std::vector<GroupDemand> demands;
        for (BlockTable& table : request.tables) {
            demands.push_back(GroupDemand{.table = &table, .extent = DenseGrowth{tokens}, .reserve_tokens = 0});
        }
        const std::optional<std::int32_t> computed =
            first_chunk ? std::nullopt : std::optional<std::int32_t>{request.computed};
        // Admit plans the demands its replay window reshapes.
        const std::vector<GroupDemand> replayed =
            Coordinator().Store().ReplayedDemands(Coordinator().Prober().ProbePrefix(probed), demands);
        const auto plan = [&]<typename Planner>(std::optional<std::int32_t> copy_in_point) {
            return planAdmissionWith<Planner>(
                Coordinator().Store(), Coordinator().History().CheckpointAgingFor(CacheTier::kDevice),
                Coordinator().Prober().ProbePrefix(probed), replayed.empty() ? demands : replayed, computed,
                copy_in_point, /*shapes=*/{}, /*claimed=*/{}, /*evicts_segments=*/false);
        };
        const auto expect_same_plan = [&](std::optional<std::int32_t> copy_in_point) {
            const std::optional<AdmissionPlan> shortcut = plan.template operator()<CountingPlanner>(copy_in_point);
            const std::optional<AdmissionPlan> walk = plan.template operator()<WalkPlanner>(copy_in_point);
            EXPECT_EQ(shortcut.has_value(), walk.has_value());
            if (shortcut && walk) {
                EXPECT_EQ(VictimsOf(*shortcut), VictimsOf(*walk));
                EXPECT_EQ(CopyInsOf(*shortcut), CopyInsOf(*walk));
            }
            return walk.has_value();
        };
        const bool fits = expect_same_plan(std::nullopt);
        // A prompt ending inside a page also reserves copy-in blocks, in
        // pairs whose bucket the fit chooses.
        const auto prompt_tokens = static_cast<std::int32_t>(request.tokens.size());
        if (prompt_tokens % kPage != 0) {
            expect_same_plan(prompt_tokens);
        }

        const std::optional<AdmissionResult> result = Coordinator().Admit(
            Coordinator().Prober().ProbePrefix(probed), demands,
            first_chunk ? RequestProgress{} : RequestProgress{.num_computed_tokens = request.computed}, {},
            std::nullopt, first_chunk ? std::nullopt : std::optional<std::uint64_t>{request.access_epoch},
            /*session_id=*/{}, /*cache_group_id=*/{});
        EXPECT_EQ(result.has_value(), fits);
        if (!result) {
            return false;
        }
        if (first_chunk) {
            request.hit_pages = result->device_prefix_tokens / kPage;
            request.computed = result->device_prefix_tokens;
            request.access_epoch = result->access_epoch;
        }
        request.computed += tokens;
        return true;
    }

    // Publishes the request's pages past its hit, then releases its blocks.
    void Finish(Request& request) {
        const std::int32_t pages = request.computed / kPage;
        if (request.hit_pages < pages) {
            CacheCompletedBlocksForTest(Coordinator(), request.tables,
                                        std::span<const std::string>{request.hashes}.first(pages), request.access_epoch,
                                        request.hit_pages, request.computed, CacheBoundaryKind::kEndpoint,
                                        /*stream_completed_to_host=*/false,
                                        /*publish_state=*/false);
        }
        request.tables.clear();
    }

    Scheduler scheduler_;
};

TEST_P(ParentFitEvictionTest, SelectsWhatTheWalkSelects) {
    std::mt19937 random(20261118 + GetParam());
    std::vector<std::vector<std::int32_t>> histories;
    std::vector<Request> active;
    std::int32_t next_token = 1;
    CountingPlanner::shortcuts = 0;
    for (int step = 0; step < 20000; ++step) {
        const auto pick = [&](std::size_t size) { return static_cast<std::size_t>(random() % size); };
        const std::uint32_t action = random() % 10;
        if (action < 4 && active.size() < 6) {
            // A new prompt, half of them continuing an earlier one.
            Request request;
            if (!histories.empty() && random() % 2 == 0) {
                const std::vector<std::int32_t>& base = histories[pick(histories.size())];
                request.tokens.assign(
                    base.begin(), base.begin() + static_cast<std::ptrdiff_t>((1 + pick(base.size() / kPage)) * kPage));
            }
            request.tokens.push_back(-next_token++);
            const auto length = request.tokens.size() + kPage * (1 + pick(10)) + pick(kPage);
            while (request.tokens.size() < length) {
                request.tokens.push_back(next_token++);
            }
            const TokenContainer container(request.tokens);
            request.hashes = ComputePrefixHashes(container.FullPrefixPages(kPage, false), "");
            request.tables.resize(static_cast<std::size_t>(Coordinator().NumGroups()));
            const std::int32_t chunk = kPage * (1 + static_cast<std::int32_t>(pick(4)));
            const std::int32_t hit = Coordinator()
                                         .Prober()
                                         .ProbePrefix(std::span<const std::string>{request.hashes}.first(
                                             (request.tokens.size() - 1) / kPage))
                                         .device.num_common_tokens;
            if (Admit(request, std::min<std::int32_t>(chunk, static_cast<std::int32_t>(request.tokens.size()) - hit))) {
                active.push_back(std::move(request));
            }
        } else if (action < 8 && !active.empty()) {
            // The next chunk of a request in flight, or its finish.
            const std::size_t index = pick(active.size());
            Request& request = active[index];
            const std::int32_t remaining = static_cast<std::int32_t>(request.tokens.size()) - request.computed;
            if (remaining == 0 || !Admit(request, std::min<std::int32_t>(remaining, kPage * (1 + pick(4))))) {
                if (random() % 4 != 0) {
                    Finish(request);
                    histories.push_back(request.tokens);
                }
                active.erase(active.begin() + static_cast<std::ptrdiff_t>(index));
            }
        } else {
            // Scattered evictions mix the epochs inside parents.
            std::vector<std::pair<std::uint32_t, CacheBlockLocation>> cached;
            for (std::uint32_t g = 0; g < static_cast<std::uint32_t>(Coordinator().NumGroups()); ++g) {
                Coordinator().GroupPrefixIndex(g).ForEachEntry(Pool(), [&](const CacheKey&, const auto& snapshot) {
                    if (!snapshot.pinned) cached.emplace_back(g, snapshot.location);
                });
            }
            for (int i = 0; i < 3 && !cached.empty(); ++i) {
                const auto [g, location] = cached[pick(cached.size())];
                CacheCoordinatorTestAccess::EvictCachedBlock(Coordinator(), g, location);
            }
        }
        if (HasFailure()) {
            FAIL() << "step " << step;
        }
    }
    for (Request& request : active) {
        request.tables.clear();
    }
    // Guards against a vacuous comparison: the shortcut must have run.
    EXPECT_GT(CountingPlanner::shortcuts, 1000u);
}

INSTANTIATE_TEST_SUITE_P(Shards, ParentFitEvictionTest, ::testing::Values(1, 2));

}  // namespace
}  // namespace tokenspeed::test
