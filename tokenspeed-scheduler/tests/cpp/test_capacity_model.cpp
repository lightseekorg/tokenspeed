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

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "integration_test_helper.h"
#include "scheduler/capacity_model.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"

namespace tokenspeed::test {

namespace {

CacheGroupConfig Full(const std::string& id, std::int32_t block_granularity, std::int32_t packing) {
    return CacheGroupConfig{
        .group_id = id,
        .block_granularity = block_granularity,
        .cache_blocks_per_lcm_block = packing,
        .attn_kind = AttnKind::kFull,
    };
}

CacheGroupConfig Sliding(const std::string& id, std::int32_t block_granularity, std::int32_t window,
                         std::int32_t packing) {
    return CacheGroupConfig{
        .group_id = id,
        .block_granularity = block_granularity,
        .cache_blocks_per_lcm_block = packing,
        .attn_kind = AttnKind::kSlidingWindow,
        .sliding_window_tokens = window,

    };
}

// A State group declares no token span: two slots per request.
CacheGroupConfig State(const std::string& id, std::int32_t packing) {
    return CacheGroupConfig{
        .group_id = id,
        .block_granularity = 0,
        .cache_blocks_per_lcm_block = packing,
        .attn_kind = AttnKind::kState,
    };
}

// A sizing config: page counts stay zero, exactly as the Python recipes hand
// it to the model before any pool exists.
SchedulerConfig SizingConfig(Role role, std::int32_t prefix_granularity, std::int32_t chunk_tokens,
                             std::int32_t decode_width, std::int32_t overlap_depth, bool disable_prefix_cache,
                             std::vector<CacheGroupConfig> groups) {
    SchedulerConfig cfg{};
    cfg.role = role;
    cfg.prefix_granularity = prefix_granularity;
    cfg.max_scheduled_tokens = chunk_tokens;
    cfg.max_batch_size = 1;
    cfg.max_resident_requests = cfg.max_batch_size;
    cfg.decode_input_tokens = decode_width;
    cfg.overlap_schedule_depth = overlap_depth;
    cfg.disable_prefix_cache = disable_prefix_cache;
    cfg.disable_l2_cache = true;
    for (CacheGroupConfig& group : groups) {
        if (role != Role::kFused) {
            group.transfer_policy = group.attn_kind == AttnKind::kState ? CacheTransferPolicy::LatestSnapshot
                                                                        : CacheTransferPolicy::FullSuffix;
        }
        cfg.cache_groups.push_back(std::move(group));
    }
    return cfg;
}

// The same config once the pool is sized: every group can address every
// usable parent, as the runtime contract publishes it.
SchedulerConfig SizedConfig(SchedulerConfig cfg, std::int32_t usable_lcm_blocks) {
    cfg.device_allocator.total_pages = usable_lcm_blocks + 1;
    for (CacheGroupConfig& group : cfg.cache_groups) {
        group.total_pages = 1 + usable_lcm_blocks * group.cache_blocks_per_lcm_block;
    }
    SetTestSnapshotPool(cfg);
    return cfg;
}

TEST(CapacityModelSink, CountsTheRetainedHeadOncePerRequest) {
    for (const auto role : {Role::kP, Role::kD, Role::kFused}) {
        auto config = SizingConfig(role, 128, 4096, 8, 1, false, {Sliding("swa", 128, 1024, 24)});
        config.max_batch_size = 8;
        config.max_resident_requests = config.max_batch_size;
        const auto without = CapacityModel(config);
        config.cache_groups[0].sink_tokens = 16;
        const auto with = CapacityModel(config);
        EXPECT_EQ(with.SingleRequestGroupPages(524288)[0], without.SingleRequestGroupPages(524288)[0] + 1);
        EXPECT_EQ(with.ConcurrentGroupPages(8LL * 524288, 524288)[0],
                  without.ConcurrentGroupPages(8LL * 524288, 524288)[0] + 8);
        EXPECT_EQ(with.SingleRequestGroupPages(0)[0], without.SingleRequestGroupPages(0)[0]);
    }
}

std::vector<SchedulerConfig> Sweep() {
    std::vector<SchedulerConfig> configs;
    const std::vector<std::vector<CacheGroupConfig>> group_sets = {
        {Full("full", 4, 1)},
        {Full("full", 4, 2), State("state", 1)},
        {Full("full", 4, 3), Sliding("swa", 4, 5, 1), Sliding("tail", 2, 9, 2)},
        {Sliding("swa", 4, 3, 1), State("state", 1)},
    };
    for (const Role role : {Role::kFused, Role::kP, Role::kD}) {
        for (const std::int32_t chunk_tokens : {4, 8, 9, 16}) {
            for (const std::int32_t decode_width : {1, 3}) {
                for (const std::int32_t overlap_depth : {0, 1}) {
                    for (const bool disable_prefix_cache : {false, true}) {
                        for (const std::vector<CacheGroupConfig>& groups : group_sets) {
                            configs.push_back(SizingConfig(role, 4, chunk_tokens, decode_width, overlap_depth,
                                                           disable_prefix_cache, groups));
                        }
                    }
                }
            }
        }
    }
    return configs;
}

}  // namespace

TEST(CapacityModelTest, L3SizingDoesNotRequireHostPagesBeforeAllocation) {
    SchedulerConfig sizing = SizingConfig(Role::kFused, 4, 8, 1, 0, false, {Full("full", 4, 1)});
    sizing.disable_l2_cache = false;
    sizing.enable_l3_storage = true;
    const CapacityModel model{sizing};
    SchedulerConfig sized = SizedConfig(sizing, 8);
    EXPECT_THROW(sized.Validate(), std::invalid_argument);
    sized.host_allocator.total_pages = 9;
    sized.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
    sized.storage_config = StorageConfig{8, 64, 4, 16, 64, 4, 100, 100, {8, 1048576, 1000}, {4, 524288, 500}};
    EXPECT_EQ(Scheduler{WithHostPackingFromDevice(sized)}.MaxSingleRequestTokens(), model.MaxSingleRequestTokens(8));
}

TEST(CapacityModelTest, MatchesTheSchedulerStartupBound) {
    for (const SchedulerConfig& sizing : Sweep()) {
        for (std::int32_t usable_lcm_blocks = 2; usable_lcm_blocks <= 12; ++usable_lcm_blocks) {
            const SchedulerConfig sized = SizedConfig(sizing, usable_lcm_blocks);
            EXPECT_EQ(Scheduler{WithHostPackingFromDevice(sized)}.MaxSingleRequestTokens(),
                      CapacityModel{sizing}.MaxSingleRequestTokens(usable_lcm_blocks))
                << "role=" << static_cast<int>(sizing.role) << " chunk=" << sizing.max_scheduled_tokens
                << " decode=" << sizing.decode_input_tokens << " overlap=" << sizing.overlap_schedule_depth
                << " groups=" << sizing.cache_groups.size() << " usable=" << usable_lcm_blocks;
        }
    }
}

TEST(CapacityModelTest, ConcurrentDemandCoversEveryAdmissibleSingleRequest) {
    // A pool sized for one live request of L tokens must hold the working
    // set the startup bound charges that request, in every group.
    for (const SchedulerConfig& sizing : Sweep()) {
        const CapacityModel model{sizing};
        for (std::int32_t token_limit = 0; token_limit <= 64; ++token_limit) {
            const std::vector<std::int64_t> single = model.SingleRequestGroupPages(token_limit);
            const std::vector<std::int64_t> concurrent = model.ConcurrentGroupPages(token_limit, token_limit);
            ASSERT_EQ(single.size(), concurrent.size());
            for (std::size_t g = 0; g < single.size(); ++g) {
                EXPECT_GE(concurrent[g], single[g])
                    << "group=" << sizing.cache_groups[g].group_id << " role=" << static_cast<int>(sizing.role)
                    << " chunk=" << sizing.max_scheduled_tokens << " decode=" << sizing.decode_input_tokens
                    << " overlap=" << sizing.overlap_schedule_depth
                    << " prefix_cache_off=" << sizing.disable_prefix_cache << " token_limit=" << token_limit;
            }
        }
    }
}

TEST(CapacityModelTest, ConcurrentDemandByKind) {
    SchedulerConfig cfg = SizingConfig(Role::kFused, 64, 8192, /*decode_width=*/4, /*overlap_depth=*/1, false,
                                       {Full("full", 64, 12), State("state", 1), Sliding("swa", 64, 128, 3)});
    cfg.max_batch_size = 16;
    cfg.max_resident_requests = cfg.max_batch_size;
    const CapacityModel model{cfg};
    const std::vector<std::int64_t> pages =
        model.ConcurrentGroupPages(/*max_total_tokens=*/65536, /*max_context_len=*/4096);
    ASSERT_EQ(pages.size(), 3u);
    // Dense history, plus per request one unaligned tail page that the four
    // protected tokens may spill past, one block a tail handover holds for
    // an admission and the copy-in block it may hold beside it: 1024 + 16 *
    // (ceil(67 / 64) + 2 * ceil(63 / 64)).
    EXPECT_EQ(pages[0], 1024 + 16 * (2 + 1 + 1));
    // A State group is two slots per request -- the checkpoint the next
    // forward reads and the block it writes -- whatever its history.
    EXPECT_EQ(model.SingleRequestGroupPages(4096)[1], 2);
    EXPECT_EQ(model.SingleRequestGroupPages(0)[1], 0);
    EXPECT_EQ(pages[1], 16 * 2);
    // Each request retains ceil((127 + 4 + 4 + 63) / 64) window pages plus
    // the one lookback block a tail handover holds and the copy-in block
    // beside it; one in-flight chunk adds the three blocks a 128-token window
    // can span behind an unaligned boundary and 8192 / 64 rows.
    EXPECT_EQ(pages[2], 16 * (4 + 1 + 1) + 3 + 128);

    // The prefill role protects no in-flight decode step and keeps writing
    // past no published tail block: the full group's protected spill and its
    // copy-in block disappear; the State group's two slots hold on every
    // role.
    SchedulerConfig prefill = cfg;
    prefill.role = Role::kP;
    for (CacheGroupConfig& group : prefill.cache_groups) {
        group.transfer_policy =
            group.attn_kind == AttnKind::kState ? CacheTransferPolicy::LatestSnapshot : CacheTransferPolicy::FullSuffix;
    }
    const std::vector<std::int64_t> prefill_pages = CapacityModel{prefill}.ConcurrentGroupPages(65536, 4096);
    EXPECT_EQ(prefill_pages[0], 1024 + 16 * (1 + 1));
    EXPECT_EQ(prefill_pages[1], 16 * 2);
    SchedulerConfig decode = prefill;
    decode.role = Role::kD;
    EXPECT_EQ(CapacityModel{decode}.ConcurrentGroupPages(65536, 4096)[1], 16 * 2);
}

// A request holds copy-in blocks beside the copy sources a tail handover
// pins only on the role whose first decode publishes its prompt end and keeps
// writing past it: the fused role. A junction publishes no KV tail, so a
// prefill role holds none even with a State group. One 64-token-page full
// group and a 128-token limit: two pages of history.
TEST(CapacityModelTest, CopyInBlocksAreChargedWhereARoleReservesThem) {
    const auto full_pages = [](Role role, bool with_state, bool disable_prefix_cache) {
        std::vector<CacheGroupConfig> groups{Full("full", 64, 1)};
        if (with_state) {
            groups.push_back(State("state", 1));
        }
        const SchedulerConfig cfg = SizingConfig(role, 64, 8192, /*decode_width=*/1, /*overlap_depth=*/0,
                                                 disable_prefix_cache, std::move(groups));
        return CapacityModel{cfg}.SingleRequestGroupPages(128)[0];
    };
    EXPECT_EQ(full_pages(Role::kFused, /*with_state=*/false, /*disable_prefix_cache=*/false), 2 + 1 + 1)
        << "a tail handover and the copy-in block of the prompt end";
    EXPECT_EQ(full_pages(Role::kFused, /*with_state=*/true, /*disable_prefix_cache=*/false), 2 + 1 + 1)
        << "a junction adds none";
    EXPECT_EQ(full_pages(Role::kP, /*with_state=*/true, /*disable_prefix_cache=*/false), 2 + 1)
        << "a tail handover only";
    EXPECT_EQ(full_pages(Role::kD, /*with_state=*/true, /*disable_prefix_cache=*/false), 2)
        << "the D role lands its partial block fresh and keeps writing past no publication";
    EXPECT_EQ(full_pages(Role::kFused, /*with_state=*/false, /*disable_prefix_cache=*/true), 2);
}

// The startup bound charges the blocks a request holds beyond its history
// while a tail changes hands: here the fused owner's copy-in block at an
// unaligned prompt end, which the chunk completing the prompt reserves and
// the first decode copies the published partial block into.
class TailHandoverBoundSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg =
            SizingConfig(Role::kFused, /*prefix_granularity=*/4, /*chunk_tokens=*/16, /*decode_width=*/1,
                         /*overlap_depth=*/0, /*disable_prefix_cache=*/false, {Full("full", 4, 1)});
        // The pool the model sizes for one 12-token request: ceil(12 / 4)
        // history blocks, one for a tail handover and one for the copy-in
        // block beside it.
        const CapacityModel model{cfg};
        const std::int64_t usable = model.LcmBlocksNeededFor(model.SingleRequestGroupPages(12));
        return SizedConfig(cfg, static_cast<std::int32_t>(usable));
    }
};

TEST_F(TailHandoverBoundSuite, AnUnalignedPromptAtTheBoundReachesItsFirstDecode) {
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 5);
    EXPECT_EQ(scheduler_->MaxSingleRequestTokens(), 12);
    EXPECT_EQ(Scheduler{SizedConfig(config_, /*usable_lcm_blocks=*/4)}.MaxSingleRequestTokens(), 8)
        << "without the copy-in block an eleven-token prompt is refused";

    // Eleven prompt tokens and one generated (a twelve-token limit): the
    // prompt's chunk takes its three blocks and the copy-in block of the
    // prompt end at 11; the first decode publishes the partial block [8, 11)
    // and moves off it into that block, taking none.
    Submit(RequestSpec{.request_id = "r", .tokens = MakeTokens(11), .max_new_tokens = 1});
    const ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* chunk = FindForwardBatch(prefill);
    ASSERT_NE(chunk, nullptr);
    EXPECT_EQ(chunk->input_lengths, std::vector<std::int32_t>{11});
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 1);
    SendForwardDone("r", {101});
    const ExecutionPlan first_decode = PlanOnce();
    const ForwardBatch* decode = FindForwardBatch(first_decode);
    ASSERT_NE(decode, nullptr) << "the copy-in block was reserved with the prompt";
    EXPECT_EQ(decode->request_ids, std::vector<std::string>{"r"});
    EXPECT_EQ(first_decode.pages_to_copy.at("full").size(), 1u);
    EXPECT_TRUE(first_decode.pages_to_zero.at("full").empty()) << "the copy fills the block whole";
    SendForwardDone("r", {102});
    SendFinish("r");
    PlanOnce();
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

TEST(CapacityModelTest, FoldsGroupPagesIntoLcmBlocksByPacking) {
    const SchedulerConfig cfg =
        SizingConfig(Role::kFused, 64, 8192, 1, 0, false, {Full("full", 64, 12), State("state", 1)});
    const CapacityModel model{cfg};
    EXPECT_EQ(model.NumGroups(), 2);
    const std::vector<std::int64_t> pages{25, 3};
    EXPECT_EQ(model.LcmBlocksNeededFor(pages), 3 + 3);
}

TEST(CapacityModelTest, ReadsNoPageCounts) {
    SchedulerConfig cfg = SizingConfig(Role::kFused, 4, 8, 1, 0, false, {Full("full", 4, 1)});
    // Unsized: the model accepts it, the Scheduler's full validation does not.
    EXPECT_NO_THROW(CapacityModel{cfg});
    EXPECT_THROW(cfg.Validate(), std::invalid_argument);
    // The model still rejects the sizing inputs it does read.
    cfg.overlap_schedule_depth = 2;
    EXPECT_THROW(CapacityModel{cfg}, std::invalid_argument);
}

}  // namespace tokenspeed::test
