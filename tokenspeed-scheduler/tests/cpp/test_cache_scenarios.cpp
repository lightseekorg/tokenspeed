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

// End-to-end scenario tests for the two-level KV-cache FSM path. Fused
// scheduling uses release-and-requeue (see RetractSuite); Decode PD uses Host
// writeback and recovery.

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include "scheduler/operations/cache.h"
#include "cache/core/cache_types.h"
#include "cache_test_access.h"
#include "integration_test_helper.h"

namespace tokenspeed::test {

namespace {

static_assert(
    std::is_same_v<decltype(std::declval<fsm::SchedulePrefillFirstChunkEvent&>()(std::declval<fsm::Submitted&&>())),
                   std::variant<fsm::PrefillDone, fsm::PrefillAwaitingResult, fsm::Prefilling, fsm::RemotePrefilling>>);
static_assert(std::is_same_v<decltype(std::declval<fsm::SchedulePrefillEvent&>()(std::declval<fsm::Prefilling&&>())),
                             std::variant<fsm::PrefillDone, fsm::PrefillAwaitingResult, fsm::Prefilling>>);

std::pair<bool, std::string> ClearL1CacheWithCapturedLog(Scheduler* scheduler) {
    std::ostringstream output;
    auto previous_logger = spdlog::default_logger();
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(output);
    auto logger = std::make_shared<spdlog::logger>("clear-l1-cache-test", std::move(sink));
    logger->set_pattern("%v");
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(std::move(logger));
    const bool cleared = scheduler->ClearL1Cache();
    spdlog::set_default_logger(std::move(previous_logger));
    return {cleared, output.str()};
}

CacheGroupConfig MakeGroup(const std::string& id, std::int32_t block_granularity, std::int32_t total_pages,
                           AttnKind kind, std::int32_t sliding_window_tokens = 0) {
    CacheGroupConfig g;
    g.group_id = id;
    g.block_granularity = kind == AttnKind::kState ? 0 : block_granularity;
    g.total_pages = total_pages;
    g.attn_kind = kind;
    if (sliding_window_tokens > 0) {
        g.sliding_window_tokens = sliding_window_tokens;
    }
    return g;
}

// Drives a request's FSM through a capacity retraction the way the scheduler
// does: image the data slots into `snapshot_pool` (no Host L2 here), take a
// blob slot, and suspend. The store pairs are dropped as the transfer manager
// would after resolving them into the wire op.
void RetractForTest(Request& request, CacheCoordinator& coordinator, SnapshotSlotAllocator& slots, std::int64_t epoch) {
    // These FSM-level tests extend tokens without admitting pages for them,
    // so image what the tables actually hold.
    std::int32_t num_tokens = request.NumComputedTokens();
    for (std::int32_t g = 0; g < coordinator.NumGroups(); ++g) {
        const BlockTable& table = request.BlockTablesRef()[static_cast<std::size_t>(g)];
        num_tokens =
            std::min(num_tokens, table.NumBlocks() * coordinator.GroupBlockGranularity(g) - table.AvailableTokens());
    }
    std::optional<CacheCoordinator::ImageTaken> taken =
        coordinator.TakeImage(request.BlockTablesRef(), num_tokens,
                              std::vector<std::vector<ImageSlot>>(static_cast<std::size_t>(coordinator.NumGroups())));
    ASSERT_TRUE(taken) << "the test's snapshot pool must hold the image";
    taken->store_pairs.clear();
    request.Apply(fsm::SnapshotRetractEvent{&coordinator, epoch, request.HasGeneratedOutput(), std::move(taken->image),
                                            std::make_shared<SnapshotSlotIndex>(slots.Allocate()),
                                            /*pending_store_ops=*/{}});
}

// Collect every real (>0) physical page id across all rows of a group.
std::vector<std::int32_t> RealPages(const std::vector<std::vector<std::int32_t>>& group) {
    std::vector<std::int32_t> out;
    for (const auto& row : group) {
        for (std::int32_t id : row) {
            if (id > 0) out.push_back(id);
        }
    }
    return out;
}

void ExpectDecodeWorkingState(const Scheduler& scheduler, const ForwardBatch& batch, std::int32_t computed_tokens,
                              std::int32_t prefix_granularity, std::int32_t usable_blocks) {
    ASSERT_EQ(batch.request_ids, std::vector<std::string>{"parent"});
    const auto& state = batch.block_tables.at("state").at(0);
    const std::int32_t input_slot = (computed_tokens - 1) / prefix_granularity;
    ASSERT_GT(state.size(), static_cast<std::size_t>(input_slot));
    for (std::int32_t slot = 0; slot < input_slot; ++slot) {
        EXPECT_EQ(state[slot], 0) << "expired state at the exact accepted frontier: " << slot;
    }
    EXPECT_GT(state[input_slot], 0) << "the next forward still needs its input state";
    for (const std::int32_t page : batch.block_tables.at("full").at(0)) {
        EXPECT_GT(page, 0) << "full-history pages never expire";
    }
    // These prompts are shorter than P, so there is no retained prefill
    // state. Once its table reference expires, a working state frees outright.
    EXPECT_EQ(usable_blocks - scheduler.EmptyLcmBlocks(), scheduler.ActiveLcmBlocks());
}

}  // namespace

// ---------------------------------------------------------------------------
// Chunked prefill: first-chunk admission followed by one admission per chunk.
// ---------------------------------------------------------------------------
class ChunkedPrefillSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 4;  // 4 tokens = 2 pages per chunk
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/4),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(ChunkedPrefillSuite, MultiChunkPrefillGrowsFullTableThenDecodes) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    // 8 tokens (4 pages) with max_scheduled_tokens=4 -> 2 prefill chunks.
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));

    ExecutionPlan chunk1 = PlanOnce();
    const ForwardBatch* op1 = FindForwardBatch(chunk1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->block_tables.count("full"), 1u);
    const std::size_t full_after_c1 = op1->block_tables.at("full").at(0).size();
    EXPECT_GT(full_after_c1, 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    SendForwardDone("r1");

    ExecutionPlan chunk2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(chunk2);
    ASSERT_NE(op2, nullptr);
    const auto& full_c2 = op2->block_tables.at("full").at(0);
    EXPECT_GT(full_c2.size(), full_after_c1) << "second chunk should extend the full-history block table";
    for (std::int32_t id : full_c2) {
        EXPECT_GT(id, 0) << "full-history row must have no null hole";
    }

    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    SendForwardDone("r1", {100});

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start)
        << "all pages returned to the pool after a chunked-prefill request finishes";
}

TEST_F(ChunkedPrefillSuite, FirstChunkPrepaysPromptHeadroomOnlyInFullHistoryGroup) {
    // 16 tokens in 4-token chunks with a declared budget of 6: the first
    // chunk prepays the 12 unscheduled prompt tokens plus 6 tokens of decode
    // headroom. The full-history group holds all of it (4 + 18 tokens -> 11
    // pages); the sliding-window group recycles slid-out pages and, with no
    // other request holding pages, holds only the chunk itself. An
    // intermediate chunk banks no tail and no decode slot, so it reserves
    // nothing there.
    RequestSpec request = MakeRequestSpec("r1", /*num_pages=*/8);
    request.max_new_tokens = 6;
    Submit(request);

    ExecutionPlan chunk1 = PlanOnce();
    const ForwardBatch* op1 = FindForwardBatch(chunk1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->input_lengths, (std::vector<std::int32_t>{4}));
    EXPECT_EQ(op1->block_tables.at("full").at(0).size(), 11u);
    EXPECT_EQ(op1->block_tables.at("swa").at(0).size(), 2u);
}

// ---------------------------------------------------------------------------
// Three cache groups: full + two sliding windows. Group 0 stays full-history
// to honor the batch consumer's block_tables_[0] contract.
// ---------------------------------------------------------------------------
class ThreeGroupSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 96;
        cfg.host_allocator.total_pages = 96;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("swa_small", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/4),
            MakeGroup("swa_big", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/8),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(ThreeGroupSuite, ThreeGroupsEachEmitARowAndReclaim) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(prefill);
    ASSERT_NE(op, nullptr);

    ASSERT_EQ(op->block_tables.count("full"), 1u);
    ASSERT_EQ(op->block_tables.count("swa_small"), 1u);
    ASSERT_EQ(op->block_tables.count("swa_big"), 1u);
    EXPECT_EQ(op->block_tables.at("full").size(), 1u);
    EXPECT_EQ(op->block_tables.at("swa_small").size(), 1u);
    EXPECT_EQ(op->block_tables.at("swa_big").size(), 1u);

    auto full_pages = RealPages(op->block_tables.at("full"));
    auto small_pages = RealPages(op->block_tables.at("swa_small"));
    auto big_pages = RealPages(op->block_tables.at("swa_big"));
    std::set<std::int32_t> all(full_pages.begin(), full_pages.end());
    all.insert(small_pages.begin(), small_pages.end());
    all.insert(big_pages.begin(), big_pages.end());
    EXPECT_EQ(all.size(), full_pages.size() + small_pages.size() + big_pages.size())
        << "groups must not share physical pages";

    SendForwardDone("r1", {42});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Sub-page (w=3 < P=4) and page-straddling (w=5 = P+1) windows (M14): pins
// per-group slide independence and the <=2-real-page steady state.
// ---------------------------------------------------------------------------
class SubPageWindowSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 4;
        cfg.device_allocator.total_pages = 96;
        cfg.host_allocator.total_pages = 96;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("swa_w3", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/3),
            MakeGroup("swa_w5", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/5),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(SubPageWindowSuite, SubPageWindowsPlateauAtTwoRealPages) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    ExecutionPlan prefill = PlanOnce();
    ASSERT_NE(FindForwardBatch(prefill), nullptr);
    SendForwardDone("r1", {1000});

    for (std::int32_t step = 0; step < 24; ++step) {
        ExecutionPlan decode = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(decode);
        ASSERT_NE(op, nullptr) << "decode step " << step;
        // fullySlidOutBlocks frees only FULLY slid-out pages: 1 <= real pages <= 2.
        const std::size_t w3_real = RealPages(op->block_tables.at("swa_w3")).size();
        const std::size_t w5_real = RealPages(op->block_tables.at("swa_w5")).size();
        EXPECT_GE(w3_real, 1u) << "w=3 lost its live tail page at step " << step;
        EXPECT_LE(w3_real, 2u) << "w=3 working set exceeded 2 pages at step " << step;
        EXPECT_GE(w5_real, 1u) << "w=5 lost its live tail page at step " << step;
        EXPECT_LE(w5_real, 2u) << "w=5 working set exceeded 2 pages at step " << step;
        SendForwardDone("r1", {1001 + step});
    }

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

TEST_F(SubPageWindowSuite, StraddlingWindowHoldsPreviousPage) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    PlanOnce();
    SendForwardDone("r1", {1000});

    bool diverged = false;
    for (std::int32_t step = 0; step < 8; ++step) {
        ExecutionPlan decode = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(decode);
        ASSERT_NE(op, nullptr);
        const std::size_t w3_real = RealPages(op->block_tables.at("swa_w3")).size();
        const std::size_t w5_real = RealPages(op->block_tables.at("swa_w5")).size();
        EXPECT_LE(w3_real, w5_real) << "a smaller window can never hold more pages, step " << step;
        if (w3_real < w5_real) {
            diverged = true;  // the straddling window (w=5) holds one more real page
        }
        SendForwardDone("r1", {1001 + step});
    }
    EXPECT_TRUE(diverged) << "w=3 and w=5 never diverged: per-group slides are not independent";

    SendFinish("r1");
    PlanOnce();
}

// ---------------------------------------------------------------------------
// Two full-history groups (no sliding window at all).
// ---------------------------------------------------------------------------
class AllFullTwoGroupSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full_a", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("full_b", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(AllFullTwoGroupSuite, BothFullGroupsKeepHistoryNoHoles) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();  // prefill
    SendForwardDone("r1", {42});

    std::optional<ExecutionPlan> last;
    int tok = 43;
    for (int i = 0; i < 4; ++i) {
        last = PlanOnce();
        ASSERT_NE(FindForwardBatch(*last), nullptr);
        SendForwardDone("r1", {tok++});
    }
    const ForwardBatch* op = FindForwardBatch(*last);
    ASSERT_NE(op, nullptr);
    for (const char* key : {"full_a", "full_b"}) {
        const auto& row = op->block_tables.at(key).at(0);
        for (std::int32_t id : row) {
            EXPECT_GT(id, 0) << key << " (full-history) must not develop a null hole";
        }
    }

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Shared-pool accounting: out-of-order finishes each return exactly their pages.
// ---------------------------------------------------------------------------
class PoolAccountingSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/4),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(PoolAccountingSuite, ThreeRequestsOutOfOrderFinishReclaimExactly) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/4, /*start=*/101));
    Submit(MakeRequestSpec("r3", /*num_pages=*/3, /*start=*/201));
    PlanOnce();  // prefill all three (max_scheduled_tokens=64 covers them)
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    const std::int32_t free_after_prefill = scheduler_->AvailableLcmBlocks();
    EXPECT_LT(free_after_prefill, free_at_start) << "prefill must consume pages from the shared pool";

    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    SendForwardDone("r3", {242});

    // Each plan decodes the live requests; those decodes land before the next
    // finish, so every finish is quiescent and releases its pages at once.
    SendFinish("r2");
    PlanOnce();
    SendForwardDone("r1", {43});
    SendForwardDone("r3", {243});
    SendFinish("r1");
    PlanOnce();
    EXPECT_LT(scheduler_->AvailableLcmBlocks(), free_at_start) << "pool not fully reclaimed while r3 is still live";
    SendForwardDone("r3", {244});
    SendFinish("r3");
    PlanOnce();

    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start)
        << "every page returns to the pool once all requests finish";
}

// Chunked prefill slides the SWA window DURING prefill, then decode keeps
// sliding. Window convention used below: with N = tokens computed BEFORE a
// round's forward, the pending query at N attends keys [N-W+1, N], so the
// first kept page is (N-W+1)/block_granularity and everything below it is freed.
TEST_F(ChunkedPrefillSuite, ChunkedPrefillThenSwaSlidesToNullHole) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    // 12 tokens (6 pages), max_scheduled_tokens=4 -> 3 prefill chunks.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    PlanOnce();  // chunk 1
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    SendForwardDone("r1");
    // Chunk 2: N=4 -> first kept token 4-4+1=1 -> first kept page 0: no hole.
    ExecutionPlan chunk2 = PlanOnce();
    const ForwardBatch* c2op = FindForwardBatch(chunk2);
    ASSERT_NE(c2op, nullptr);
    {
        const auto& swa_c2 = c2op->block_tables.at("swa").at(0);
        ASSERT_EQ(swa_c2.size(), 4u);
        EXPECT_EQ(std::count(swa_c2.begin(), swa_c2.end(), 0), 0)
            << "N=4, W=4: no page fully below token 1, so chunk 2 punches nothing";
    }
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    SendForwardDone("r1");
    const std::int32_t free_after_c2 = scheduler_->AvailableLcmBlocks();

    // Chunk 3: N=8 -> first kept token 5 -> page 5/2=2: slots 0,1 punched MID-PREFILL.
    ExecutionPlan chunk3 = PlanOnce();  // chunk 3 (last)
    const ForwardBatch* c3op = FindForwardBatch(chunk3);
    ASSERT_NE(c3op, nullptr);
    {
        const auto& swa_c3 = c3op->block_tables.at("swa").at(0);
        ASSERT_EQ(swa_c3.size(), 7u);
        for (int s = 0; s <= 1; ++s) EXPECT_EQ(swa_c3[s], 0) << "slot " << s << " punched during prefill";
        for (int s = 2; s <= 6; ++s) EXPECT_GT(swa_c3[s], 0) << "slot " << s;
        for (std::int32_t id : c3op->block_tables.at("full").at(0)) {
            EXPECT_GT(id, 0) << "full group keeps every chunk-built page";
        }
    }
    // Chunk-3 balance: slide frees 2 SWA pages, the chunk takes 2/group and
    // the physically-backed decode reservation takes 1/group.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_after_c2 + 2 - 4 - 2)
        << "the mid-prefill slide must return the out-of-window pages to the pool";

    SendForwardDone("r1", {99});  // container size 13 (12 prompt + 1 sampled)

    // swa_rows[i] = the swa row round i's op carried (after slide + acquire).
    std::vector<std::vector<std::int32_t>> swa_rows;
    int tok = 100;
    for (int i = 0; i < 4; ++i) {
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(plan);
        ASSERT_NE(op, nullptr);
        for (std::int32_t id : op->block_tables.at("full").at(0)) {
            EXPECT_GT(id, 0) << "full group must keep chunk-built history without holes (round " << i << ")";
        }
        swa_rows.push_back(op->block_tables.at("swa").at(0));
        SendForwardDone("r1", {tok++});
    }

    auto null_count = [](const std::vector<std::int32_t>& row) { return std::count(row.begin(), row.end(), 0); };

    // Round 0 (finalize): N=12 -> first kept page 4; + reserve page -> 7 slots, 4 holes.
    ASSERT_EQ(swa_rows[0].size(), 7u);
    EXPECT_EQ(null_count(swa_rows[0]), 4) << "finalize slides at the full prefill length";
    for (int s = 0; s <= 3; ++s) EXPECT_EQ(swa_rows[0][s], 0) << "slot " << s;
    for (int s = 4; s <= 6; ++s) EXPECT_GT(swa_rows[0][s], 0) << "slot " << s;

    // Round 1: N=13 -> first kept page 5; tail room absorbs the acquire.
    ASSERT_EQ(swa_rows[1].size(), 7u);
    EXPECT_EQ(null_count(swa_rows[1]), 5);
    for (int s = 0; s <= 4; ++s) EXPECT_EQ(swa_rows[1][s], 0) << "slot " << s;
    for (int s = 5; s <= 6; ++s) EXPECT_GT(swa_rows[1][s], 0) << "slot " << s;

    // Round 2: N=14 -> first kept token 11 -> page 5 (unchanged); acquire adds
    // page 7. Sliding at the container size 15 instead would free slot 5 early.
    ASSERT_EQ(swa_rows[2].size(), 8u);
    EXPECT_EQ(null_count(swa_rows[2]), 5);
    EXPECT_GT(swa_rows[2][5], 0) << "slot 5 must survive round 2: key 11 of the pending query lives there";
    for (int s = 6; s <= 7; ++s) EXPECT_GT(swa_rows[2][s], 0) << "slot " << s;

    // Round 3: N=15 -> first kept token 12 -> first kept page 6.
    ASSERT_EQ(swa_rows[3].size(), 8u);
    EXPECT_EQ(null_count(swa_rows[3]), 6);
    EXPECT_EQ(swa_rows[3][5], 0) << "slot 5 slides out once the query window has moved past key 11";
    for (int s = 6; s <= 7; ++s) EXPECT_GT(swa_rows[3][s], 0) << "slot " << s;

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

TEST_F(ThreeGroupSuite, TwoRequestsBatchedAcrossThreeGroupsNoCollision) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(prefill);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 2u);

    for (const char* key : {"full", "swa_small", "swa_big"}) {
        ASSERT_EQ(op->block_tables.count(key), 1u) << key;
        EXPECT_EQ(op->block_tables.at(key).size(), 2u) << key;
    }

    std::vector<std::int32_t> every;
    for (const char* key : {"full", "swa_small", "swa_big"}) {
        auto pages = RealPages(op->block_tables.at(key));
        every.insert(every.end(), pages.begin(), pages.end());
    }
    std::vector<std::int32_t> sorted = every;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end())
        << "no physical page may be shared across requests or groups";

    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    SendFinish("r1");
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Mixed batch: with enable_mixed_prefill_decode a decode and a prefill share
// one SoA op; stable_partition puts prefill rows ahead of decode rows.
// ---------------------------------------------------------------------------
class MixedBatchSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        cfg.enable_mixed_prefill_decode = true;  // decode + prefill in one plan

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/4),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(MixedBatchSuite, PrefillAndDecodeShareOnePlan) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();                   // r1 prefill
    SendForwardDone("r1", {42});  // r1 -> decode

    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan mixed = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(mixed);
    ASSERT_NE(op, nullptr);

    ASSERT_EQ(op->request_ids.size(), 2u);
    EXPECT_EQ(op->NumExtends(), 1u) << "exactly one prefill row (r2)";
    EXPECT_EQ(op->decode_input_ids.size(), 1u) << "exactly one decode row (r1)";

    EXPECT_EQ(op->request_ids.at(0), "r2") << "prefill partitioned first";
    EXPECT_EQ(op->request_ids.at(1), "r1") << "decode after prefill";

    for (const char* key : {"full", "swa"}) {
        ASSERT_EQ(op->block_tables.count(key), 1u) << key;
        ASSERT_EQ(op->block_tables.at(key).size(), 2u) << key;
        auto pages = RealPages(op->block_tables.at(key));
        std::vector<std::int32_t> sorted = pages;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end())
            << key << ": two requests must not share a physical page";
    }

    SendForwardDone("r1", {43});
    SendForwardDone("r2", {142});
    SendFinish("r1");
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

// Swa eviction state is tracked independently per request, not batch-wide.
TEST_F(MixedBatchSuite, PerRequestSwaHoleAtDifferentDecodeDepths) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    PlanOnce();  // both prefill together (mixed batch)
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});

    // r1 goes well past the window (W=4 = 2 pages); r2 advances once, staying inside it.
    std::optional<ExecutionPlan> last;
    int t1 = 43, t2 = 143;
    for (int step = 0; step < 5; ++step) {
        last = PlanOnce();
        ASSERT_NE(FindForwardBatch(*last), nullptr);
        SendForwardDone("r1", {t1++});
        if (step == 0) {
            SendForwardDone("r2", {t2++});  // r2 advances only once
        }
    }
    const ForwardBatch* op = FindForwardBatch(*last);
    ASSERT_NE(op, nullptr);

    // Row order within the op is not guaranteed.
    const auto& ids = op->request_ids;
    auto row_of = [&](const std::string& id) -> std::size_t {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == id) return i;
        }
        ADD_FAILURE() << "request " << id << " not in op";
        return 0;
    };

    // r2 may or may not remain in the batch; assert only on rows present.
    const auto& swa = op->block_tables.at("swa");
    const auto& full = op->block_tables.at("full");
    if (std::find(ids.begin(), ids.end(), "r1") != ids.end()) {
        std::size_t r1 = row_of("r1");
        EXPECT_NE(std::find(swa.at(r1).begin(), swa.at(r1).end(), 0), swa.at(r1).end())
            << "r1 drove past the window -> swa row must have a null hole";
        for (std::int32_t id : full.at(r1)) {
            EXPECT_GT(id, 0) << "r1 full-history row must stay hole-free";
        }
    }

    SendFinish("r1");
    if (scheduler_->DecodingSize() > 0) SendFinish("r2");
    PlanOnce();
}

// ---------------------------------------------------------------------------
// prefix_granularity = 1: the batch path is not hard-wired to prefix_granularity=2.
// ---------------------------------------------------------------------------
class PrefixGranularityOneSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 1;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/2),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(PrefixGranularityOneSuite, TokenGranularPagesSlideAndReclaim) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* pop = FindForwardBatch(prefill);
    ASSERT_NE(pop, nullptr);
    EXPECT_EQ(pop->block_tables.at("full").at(0).size(), 4u) << "three prompt pages plus one preallocated decode page";

    SendForwardDone("r1", {42});

    std::optional<ExecutionPlan> last;
    int tok = 43;
    for (int i = 0; i < 4; ++i) {
        last = PlanOnce();
        ASSERT_NE(FindForwardBatch(*last), nullptr);
        SendForwardDone("r1", {tok++});
    }
    const ForwardBatch* op = FindForwardBatch(*last);
    ASSERT_NE(op, nullptr);
    for (std::int32_t id : op->block_tables.at("full").at(0)) {
        EXPECT_GT(id, 0) << "full group hole-free at prefix_granularity=1";
    }
    const auto& swa = op->block_tables.at("swa").at(0);
    EXPECT_NE(std::find(swa.begin(), swa.end(), 0), swa.end())
        << "swa group must develop a null hole at prefix_granularity=1 too";

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

namespace {

void SendAbort(Scheduler& scheduler, const std::string& id) {
    ExecutionEvent event;
    event.With(forward::Abort{.request_id = id});
    scheduler.Advance(std::move(event));
}

}  // namespace

// ---------------------------------------------------------------------------
// Pool-exhaustion admission. The first-chunk gate charges prompt + decode
// reserve = groups * ceil((tokens + 1) / prefix_granularity) blocks.
// ---------------------------------------------------------------------------
class TinyPoolSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        // 11 physical pages -> 10 usable (page 0 is the null placeholder):
        // one 4-page prompt over 2 groups (8 prefill + 2 reserve) = the pool.
        cfg.device_allocator.total_pages = 11;
        cfg.host_allocator.total_pages = 11;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/4),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(TinyPoolSuite, ExhaustedPoolDefersSecondRequestUntilFirstFinishes) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    ASSERT_EQ(free_at_start, 10);

    // r1 exact admission acquires 8 prefill + 2 reserve blocks: free 0.
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    ExecutionPlan plan1 = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan1), nullptr);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0);

    // r2 needs 4 blocks while r1 owns the whole pool: deferred.
    Submit(MakeRequestSpec("r2", /*num_pages=*/1, /*start=*/101));
    SendForwardDone("r1", {99});
    ExecutionPlan blocked = PlanOnce();
    const ForwardBatch* blocked_op = FindForwardBatch(blocked);
    ASSERT_NE(blocked_op, nullptr);
    ASSERT_EQ(blocked_op->request_ids.size(), 1u) << "only r1's reserved decode step fits this round";
    EXPECT_EQ(blocked_op->request_ids.at(0), "r1");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "deferred r2 stays intact in the waiting set";
    // Finalize consumes the reservation and exposes two slid-out SWA parents.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 2);

    SendForwardDone("r1", {100});
    SendFinish("r1");
    ExecutionPlan plan2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(plan2);
    ASSERT_NE(op2, nullptr) << "deferred request must be schedulable after pages free up";
    ASSERT_EQ(op2->request_ids.size(), 1u);
    EXPECT_EQ(op2->request_ids.at(0), "r2");
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    SendForwardDone("r2", {142});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start)
        << "pool back to baseline after the deferred request completes";
}

// ---------------------------------------------------------------------------
// Prefill-slide admission: a long chunked prompt fits ONLY because the gate
// credits the slide the chunk itself performs (BlocksFreedByAdvance).
// ---------------------------------------------------------------------------
class PrefillSlideAdmissionSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 13;
        cfg.host_allocator.total_pages = 16;  // 15 usable + the null placeholder (page 0): 12 pages + a tail per group
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 4;  // 4-token prefill chunks
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/4),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(PrefillSlideAdmissionSuite, LongPromptAdmittedOnlyBecausePrefillSlides) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    ASSERT_EQ(free_at_start, 12);

    // page=2, W=4, 4-token chunks: c1 charges 4 blocks (2/group), 12 -> 8;
    // c2 (slide credit 0) charges 4, acquires 4 -> free 4.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    ExecutionPlan c1 = PlanOnce();
    ASSERT_NE(FindForwardBatch(c1), nullptr);
    ASSERT_EQ(FindForwardBatch(c1)->request_ids.size(), 1u);
    SendForwardDone("r1");
    ExecutionPlan c2 = PlanOnce();
    ASSERT_NE(FindForwardBatch(c2), nullptr);
    ASSERT_EQ(FindForwardBatch(c2)->request_ids.size(), 1u);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 4);
    SendForwardDone("r1");

    // c3 gate: chunk + reserve = 3 blocks/group = 6 vs raw free 4; the pending
    // slide at N=8 frees the 2 swa pages below token 5 -> 4 + 2 = 6, admitted.
    ExecutionPlan c3 = PlanOnce();
    const ForwardBatch* c3op = FindForwardBatch(c3);
    ASSERT_NE(c3op, nullptr);
    ASSERT_EQ(c3op->request_ids.size(), 1u) << "final chunk must be admitted via the prefill slide credit";
    // Op balance: punch 2, acquire 2/group plus 1 reserved/group -> exact fit.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0);

    // Decode transition: gate needs 2, finalize-slide credit at N=12 gives 2.
    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    ASSERT_EQ(FindForwardBatch(decode)->request_ids.size(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 2);

    SendForwardDone("r1", {100});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

TEST_F(PrefillSlideAdmissionSuite, InFlightStorePinsItsSourcesUntilAck) {
    // Sink ON over the LongPromptAdmittedOnlyBecausePrefillSlides math: device
    // 13 -> 12 usable, c1+c2 charge 8, c3 needs 6 = free 4 + slide credit 2.
    // An ordinary store ticket pins its device sources until the ACK, so the
    // slid SWA pages op1 is copying are not evictable yet: c3 waits one round
    // for the ACK -- and waits, rather than retracting anyone (least of all
    // itself) for capacity that the ACK is about to return.
    config_.disable_l2_cache = false;
    config_.host_allocator.total_pages = 15;  // 12 streamed pages plus the finish-time tail in both groups
    config_.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    ASSERT_EQ(free_at_start, 12);

    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    ExecutionPlan c1 = PlanOnce();
    ASSERT_NE(FindForwardBatch(c1), nullptr);
    ASSERT_EQ(FindForwardBatch(c1)->request_ids.size(), 1u);
    SendForwardDone("r1");

    ExecutionPlan c2 = PlanOnce();  // registers pages 0,1 both groups: streaming op1
    ASSERT_NE(FindForwardBatch(c2), nullptr);
    ASSERT_EQ(FindForwardBatch(c2)->request_ids.size(), 1u);
    auto wb1 = ExtractCacheOpsOfKind<WriteBackBatch>(c2);
    ASSERT_EQ(wb1.size(), 1u);
    const auto op1 = std::get<WriteBackBatch>(wb1.front());
    ASSERT_EQ(op1.op_ids.size(), 1u);
    EXPECT_EQ(op1.src_pages.at(0).size(), 4u) << "the first completed Full+SWA pages stream together";
    EXPECT_EQ(op1.source_pinned, std::vector<bool>{true}) << "an ordinary publication pins its sources";
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 4);
    SendForwardDone("r1");

    ExecutionPlan stalled = PlanOnce();  // slide credit 2 is pinned by op1: c3 cannot admit yet
    const ForwardBatch* stalled_forward = FindForwardBatch(stalled);
    ASSERT_TRUE(stalled_forward == nullptr || stalled_forward->request_ids.empty())
        << "an in-flight pinned store holds the slid pages until the ACK";
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(stalled).empty())
        << "a retraction would have emitted a snapshot store; the pinned store defers retraction instead";
    EXPECT_EQ(scheduler_->WaitingSize(), 0u) << "the prefill keeps its place; nothing was retracted";
    EXPECT_EQ(scheduler_->PrefillSize(), 1u);

    SendWriteBackDone(op1.op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 4);

    ExecutionPlan c3 = PlanOnce();  // the ACK released the pins: slide credit 2 -> admitted; emits op2
    ASSERT_NE(FindForwardBatch(c3), nullptr);
    ASSERT_EQ(FindForwardBatch(c3)->request_ids.size(), 1u) << "the ACK returns the slid pages; c3 admits";
    auto wb2 = ExtractCacheOpsOfKind<WriteBackBatch>(c3);
    ASSERT_EQ(wb2.size(), 1u);
    const auto op2 = std::get<WriteBackBatch>(wb2.front());
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0);

    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();  // last prefill pages stream on PrefillDone
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    ASSERT_EQ(FindForwardBatch(decode)->request_ids.size(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    auto wb3 = ExtractCacheOpsOfKind<WriteBackBatch>(decode);
    ASSERT_EQ(wb3.size(), 1u);
    const auto op3 = std::get<WriteBackBatch>(wb3.front());

    SendForwardDone("r1", {100});
    SendFinish("r1");
    ExecutionPlan finish = PlanOnce();  // the partial page at token 13 streams as a tail (full + swa)
    auto wb4 = ExtractCacheOpsOfKind<WriteBackBatch>(finish);
    ASSERT_EQ(wb4.size(), 1u);
    const auto op4 = std::get<WriteBackBatch>(wb4.front());
    EXPECT_EQ(op4.src_pages.at(0).size(), 2u);
    EXPECT_LT(scheduler_->AvailableLcmBlocks(), free_at_start)
        << "op2/op3/op4 still pin their sources: the pool does not balance before their ACKs";
    SendWriteBackDone(op2.op_ids.at(0));
    SendWriteBackDone(op3.op_ids.at(0));
    SendWriteBackDone(op4.op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 14);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "the ACKs return every pinned source";
}

// Pool 17 -> 16 usable: swa at full prompt length would need 10+10+2 = 22
// (infeasible); the plateau ceil((chunk+W-1)/P) = ceil(7/2) = 4 keeps the peak
// at full 10 + swa 4 + reserve 2 = 16 (exact fit) -- the batch-swa-alloc contract.
class PrefillPlateauSuite : public PrefillSlideAdmissionSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = PrefillSlideAdmissionSuite::MakeConfig();
        cfg.device_allocator.total_pages = 17;
        cfg.host_allocator.total_pages = 17;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(PrefillPlateauSuite, SwaWorkingSetPlateausWhileFullGrowsToPromptLength) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    ASSERT_EQ(free_at_start, 16);

    Submit(MakeRequestSpec("r1", /*num_pages=*/10));  // 20 tokens, 5 chunks of 4
    std::size_t swa_peak = 0;
    std::size_t full_last = 0;
    for (std::int32_t chunk = 0; chunk < 5; ++chunk) {
        if (chunk > 0) {
            SendForwardDone("r1");  // the previous chunk landed: KV only
        }
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(plan);
        ASSERT_NE(op, nullptr) << "chunk " << chunk;
        ASSERT_EQ(op->request_ids.size(), 1u) << "chunk " << chunk << " must be admitted";
        const std::size_t swa_real = RealPages(op->block_tables.at("swa")).size();
        const std::size_t full_real = RealPages(op->block_tables.at("full")).size();
        EXPECT_LE(swa_real, 5u) << "swa exceeded its four-page window plus decode reserve at chunk " << chunk;
        EXPECT_GE(full_real, full_last) << "full group must grow monotonically, chunk " << chunk;
        swa_peak = std::max(swa_peak, swa_real);
        full_last = full_real;
    }
    EXPECT_EQ(swa_peak, 5u) << "the four-page window plus decode reserve must be reached";
    EXPECT_EQ(full_last, 11u) << "ten prompt pages plus one preallocated decode page";

    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);

    SendForwardDone("r1", {100});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Capacity blocking: once no work or result is in flight, the scheduler
// immediately retracts the largest running request.
// ---------------------------------------------------------------------------
class CapacityBlockSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        // 13 physical pages -> 12 usable: two 2-page prompts charge
        // 2*ceil(5/2) = 6 blocks each at admission = exactly the pool.
        cfg.device_allocator.total_pages = 13;
        cfg.host_allocator.total_pages = 14;  // 13 usable + the null placeholder (page 0)
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full_a", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("full_b", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

// Same pool arithmetic as CapacityBlockSuite, with the prefix cache live so
// that page publication is observable through a later request's hit length.
class CapacityBlockPrefixCacheSuite : public CapacityBlockSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = CapacityBlockSuite::MakeConfig();
        cfg.disable_prefix_cache = false;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(CapacityBlockPrefixCacheSuite, FailedAdmissionLeavesCacheProgressUncommitted) {
    // A decode round advances the request's prefix-hash chain for the page
    // its last step completed and asks admission to publish that page. When
    // admission fails for capacity, NOTHING of that may stick: the retry must
    // see the same page as still-unpublished, or it silently never enters the
    // prefix cache. Resources and progress land when admission succeeds --
    // never before.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));  // tokens 1..4
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    PlanOnce();
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    PlanOnce();  // r1 = [1 2 3 4 42]: publishes pages 0,1
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});
    PlanOnce();  // r1 = [1 2 3 4 42 43]
    SendForwardDone("r1", {44});
    // r1 = [1 2 3 4 42 43 44]: page 2 ([42 43]) is complete and due for
    // publication -- but the pool is full and r2's result is still out, so
    // r1's decode admission fails and the round stays quiet.
    ExecutionPlan quiet = PlanOnce();
    ASSERT_TRUE(FindForwardBatch(quiet)->request_ids.empty());

    // r2 leaves; r1's retried decode admission now succeeds and publishes
    // page 2 as part of the same admission.
    SendForwardDone("r2", {144});
    SendFinish("r2");
    ExecutionPlan resumed = PlanOnce();
    ASSERT_EQ(FindForwardBatch(resumed)->request_ids, std::vector<std::string>{"r1"});
    SendForwardDone("r1", {45});
    SendFinish("r1");
    PlanOnce();

    // A prompt sharing r1's first six tokens must hit all three pages.
    Submit(RequestSpec{.request_id = "r3", .tokens = {1, 2, 3, 4, 42, 43, 7, 8}});
    ExecutionPlan hit = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(hit);
    ASSERT_EQ(op->request_ids, std::vector<std::string>{"r3"});
    EXPECT_EQ(op->extend_prefix_lens.at(0), 6)
        << "page 2 was never published: the failed admission's progress leaked into the request";
}

TEST_F(CapacityBlockSuite, AllBusyResidentsWaitUntilTheirResultsLand) {
    Submit(MakeRequestSpec("r1", 2));
    Submit(MakeRequestSpec("r2", 2, 101));
    PlanOnce();
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    PlanOnce();
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});
    PlanOnce();
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    Submit(MakeRequestSpec("waiting", 1, 201));
    const ExecutionPlan waiting = PlanOnce();
    EXPECT_TRUE(waiting.aborts.empty());
    EXPECT_EQ(FindSnapshotStore(waiting), nullptr);
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 2u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0);

    SendForwardDone("r1", {44});
    SendForwardDone("r2", {144});
    const ExecutionPlan resumed = PlanOnce();
    const SnapshotStoreBatch* store = FindSnapshotStore(resumed);
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->request_ids, std::vector<std::string>{"r1"});
    EXPECT_TRUE(resumed.aborts.empty());
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
}

TEST_F(CapacityBlockSuite, RetractsLargestRunningRequestImmediately) {
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 12);

    // Round 1: both exact admissions include physical decode reservations.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* op1 = FindForwardBatch(prefill);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 2u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});

    // Round 2: both decode transitions consume their 2-block reservations.
    ExecutionPlan round2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(round2);
    ASSERT_NE(op2, nullptr);
    ASSERT_EQ(op2->request_ids.size(), 2u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});

    // Round 3: both next steps fit their tail pages (0 fresh blocks).
    ExecutionPlan round3 = PlanOnce();
    const ForwardBatch* op3 = FindForwardBatch(round3);
    ASSERT_NE(op3, nullptr);
    ASSERT_EQ(op3->request_ids.size(), 2u);

    // A blocked round with r2's decode result STILL IN FLIGHT must stay quiet.
    SendForwardDone("r1", {44});
    ExecutionPlan quiet = PlanOnce();
    const ForwardBatch* quiet_op = FindForwardBatch(quiet);
    ASSERT_NE(quiet_op, nullptr);
    EXPECT_TRUE(quiet_op->request_ids.empty());

    // Nothing is in flight now, so this blocked round retracts immediately.
    // r1 and r2 tie at 2 generated tokens; deterministic candidate order
    // picks r1, and r1 being the blocker itself, its pages serve the other
    // blocked decode, r2, in the same round.
    SendForwardDone("r2", {144});
    ExecutionPlan retract_round = PlanOnce();
    const ForwardBatch* retract_op = FindForwardBatch(retract_round);
    ASSERT_NE(retract_op, nullptr);
    EXPECT_EQ(retract_op->request_ids, std::vector<std::string>{"r2"});
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 6 - 2) << "r1's 3 pages x 2 groups return; r2 takes a page pair";
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "r1 is suspended with its image";
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    // No Host cache: the whole image (6 computed tokens = 3 pages x 2 groups)
    // goes to the snapshot pool; the store rides beside the empty batch.
    const SnapshotStoreBatch* store = FindSnapshotStore(retract_round);
    ASSERT_NE(store, nullptr);
    ASSERT_EQ(store->request_ids, std::vector<std::string>{"r1"});
    EXPECT_EQ(store->src_pages.at(0).size(), 6u);
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(retract_round).empty());
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 12 - 6);

    // The remaining request keeps decoding on the released pages. r1's image
    // has not been acknowledged, so no restore is attempted yet.
    SendForwardDone("r2", {145});
    ExecutionPlan resumed = PlanOnce();
    const ForwardBatch* resumed_op = FindForwardBatch(resumed);
    ASSERT_NE(resumed_op, nullptr);
    ASSERT_EQ(resumed_op->request_ids, std::vector<std::string>{"r2"});
    EXPECT_EQ(FindRestore(resumed), nullptr);
    AckImageStores(retract_round);
    SendForwardDone("r2", {146});
    SendFinish("r2");

    // With r2 reaped, r1 is restored: a cache op beside an empty batch, no
    // prefill of any kind, and the prompt window untouched.
    ExecutionPlan readmit = PlanOnce();
    ASSERT_TRUE(FindForwardBatch(readmit)->request_ids.empty());
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr);
    ASSERT_EQ(restore->request_ids, std::vector<std::string>{"r1"});
    EXPECT_EQ(restore->src_pages.at(0).size(), 6u);
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "Restoring until the ACK";
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    AckRestores(readmit);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u) << "a decoding victim resumes Decoding";
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 12) << "the image is released once copied back";
    EXPECT_EQ(scheduler_->RequestTokenSize("r1"), 7);

    ExecutionPlan decode = PlanOnce();
    const ForwardBatch* decode_op = FindForwardBatch(decode);
    ASSERT_NE(decode_op, nullptr);
    ASSERT_EQ(decode_op->request_ids, std::vector<std::string>{"r1"});
    EXPECT_EQ(decode_op->NumExtends(), 0u) << "no recovery prefill: the next step is an ordinary decode";
    EXPECT_EQ(decode_op->prefill_lengths.at(0), 4) << "nothing was rebased";
    SendForwardDone("r1", {45});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 12);
}

class ConsumedHeadroomRetractionSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 4096;
        // Four usable blocks: each request initially holds one prompt block
        // and one 4096-token headroom block, exactly filling the pool.
        cfg.device_allocator.total_pages = 5;
        cfg.host_allocator.total_pages = 0;
        cfg.max_scheduled_tokens = 8192;
        cfg.max_batch_size = 2;
        cfg.decode_input_tokens = 1024;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(ConsumedHeadroomRetractionSuite, ConsumedPartialHeadroomDoesNotDisableRetraction) {
    RequestSpec first = MakeRequestSpec("a", /*num_pages=*/1, /*start=*/1);
    first.max_new_tokens = 6000;
    RequestSpec second = MakeRequestSpec("b", /*num_pages=*/1, /*start=*/5000);
    second.max_new_tokens = 6000;
    Submit({first, second});

    const ExecutionPlan prefill_plan = PlanOnce();
    const ForwardBatch* prefill = FindForwardBatch(prefill_plan);
    ASSERT_NE(prefill, nullptr);
    ASSERT_EQ(prefill->request_ids, (std::vector<std::string>{"a", "b"}));
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    SendForwardDone("a", {42});
    SendForwardDone("b", {43});

    const std::vector<std::int32_t> decode_result(1024, 7);
    // Each decode also holds the next 1024-token verify window, so three rounds fill the headroom block.
    for (std::int32_t round = 0; round < 3; ++round) {
        const ExecutionPlan decode_plan = PlanOnce();
        const ForwardBatch* decode = FindForwardBatch(decode_plan);
        ASSERT_NE(decode, nullptr) << "decode round " << round;
        ASSERT_EQ(decode->request_ids, (std::vector<std::string>{"a", "b"})) << "decode round " << round;
        SendForwardDone("a", decode_result);
        SendForwardDone("b", decode_result);
    }

    // Both requests have consumed their partial 4096-token admission
    // headroom and now need another block. Their shorter remaining budgets
    // must not retroactively turn that spent headroom into a full-generation
    // reserve, or chooseVictim finds nobody and this state never changes.
    const ExecutionPlan blocked_plan = PlanOnce();
    const ForwardBatch* blocked = FindForwardBatch(blocked_plan);
    ASSERT_NE(blocked, nullptr);
    EXPECT_EQ(blocked->request_ids, std::vector<std::string>{"b"}) << "a's pages serve b's blocked decode";
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 1) << "a's two blocks freed, b took one";

    AckImageStores(blocked_plan);
    SendForwardDone("b", decode_result);
    SendFinish("b");
    // "a" is restored -- its computed tokens come back by copy, not by a
    // prefill -- and resumes decoding.
    const ExecutionPlan readmit_plan = PlanOnce();
    ASSERT_TRUE(FindForwardBatch(readmit_plan)->request_ids.empty());
    const SnapshotRestoreBatch* restore = FindRestore(readmit_plan);
    ASSERT_NE(restore, nullptr);
    ASSERT_EQ(restore->request_ids, std::vector<std::string>{"a"});
    EXPECT_EQ(restore->src_pages.at(0).size(), 2u) << "two 4096-token pages hold the computed tokens";
    AckRestores(readmit_plan);
    const ExecutionPlan decode_plan = PlanOnce();
    const ForwardBatch* decode = FindForwardBatch(decode_plan);
    ASSERT_NE(decode, nullptr);
    ASSERT_EQ(decode->request_ids, std::vector<std::string>{"a"});
    EXPECT_EQ(decode->NumExtends(), 0u);
    SendForwardDone("a", {44});
    SendFinish("a");
    PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 4);
}

class FusedRetractionL2TestSuite : public CapacityBlockSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = CapacityBlockSuite::MakeConfig();
        cfg.disable_l2_cache = false;
        cfg.disable_prefix_cache = false;
        // Room for a finished request's partial page in both groups beside
        // the pinned image and the streamed prompt pages.
        cfg.host_allocator.total_pages = 16;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void CompleteStores(const ExecutionPlan& plan) { AckImageStores(plan); }
};

TEST_F(FusedRetractionL2TestSuite, RetractionStoresTheLatestCompletedBoundary) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    CompleteStores(prefill);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});

    ExecutionPlan first_decode = PlanOnce();
    CompleteStores(first_decode);
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});

    ExecutionPlan second_decode = PlanOnce();
    CompleteStores(second_decode);
    SendForwardDone("r1", {44});
    PlanOnce();  // r2 still has a forward result in flight, so retraction waits.
    SendForwardDone("r2", {144});

    const ExecutionPlan retraction = PlanOnce();
    // r1 = [1 2 3 4 42 43 44]: pages [1 2] [3 4] [42 43] are published and
    // ride Host L2. The prompt pages were streamed to Host at prefill already
    // (so were r2's), so the image pins those entries and the stream-ordered
    // store copies only [42 43]; 6 computed tokens end on a page boundary, so
    // no tail page rides the snapshot pool.
    const auto write_backs = ExtractCacheOpsOfKind<WriteBackBatch>(retraction);
    ASSERT_EQ(write_backs.size(), 1u) << "the image's L2 leg is a stream-ordered store";
    const auto& write_back = std::get<WriteBackBatch>(write_backs.front());
    ASSERT_EQ(write_back.op_ids.size(), 1u);
    EXPECT_EQ(write_back.source_pinned, std::vector<bool>{false})
        << "the victim's pages are granted away this round; the runtime fences the copy ahead of reuse";
    EXPECT_EQ(write_back.src_pages.at(0).size(), 2u) << "the one page not yet on Host, in both groups";
    const SnapshotStoreBatch* store = FindSnapshotStore(retraction);
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->request_ids, std::vector<std::string>{"r1"});
    EXPECT_TRUE(store->src_pages.at(0).empty()) << "no tail page; the op still carries the slot-state blob";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 4) << "the image pins the Host entries that already exist";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 8) << "r1's and r2's prompt pages";
    CompleteStores(retraction);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 10) << "the ACK publishes the new entries";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6) << "the suspended request pins every Host entry of its image";
    EXPECT_FALSE(scheduler_->CanClearCache()) << "pinned entries refuse a flush";
}

// A victim whose frontier is not page-aligned publishes its partial page as a
// tail: the image's L2 leg carries it, and the restore copies it back into a
// fresh private block -- never claiming the published block the victim will
// keep writing -- on an unkeyed row, while the full pages are claimed.
class ForcedRetractionL2TailSuite : public FusedRetractionL2TestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FusedRetractionL2TestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 33;  // no capacity pressure: the knob alone retracts
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        SetTestSnapshotPool(cfg);
        cfg.debug_force_retraction_interval = 4;
        return cfg;
    }
};

TEST_F(ForcedRetractionL2TailSuite, AVictimsPartialPageIsPublishedAndRestoredAsAPrivateCopy) {
    Submit(RequestSpec{.request_id = "r1", .tokens = {1, 2, 3}});
    CompleteStores(PlanOnce());  // plan 1: prefill
    SendForwardDone("r1", {42});
    // Plan 2: the first decode is the prompt end. Page [1 2] streams; the
    // partial page [3] is published as a tail and r1 moves to a fresh copy
    // of it in both groups, so the published block is never written again.
    const ExecutionPlan first_decode = PlanOnce();
    EXPECT_EQ(scheduler_->TailEntries(), 1u);
    ASSERT_EQ(first_decode.pages_to_copy.size(), 2u);
    for (const auto& [group_id, pairs] : first_decode.pages_to_copy) {
        ASSERT_EQ(pairs.size(), 1u) << group_id;
        EXPECT_NE(pairs.front().first, pairs.front().second) << group_id;
        const auto& zeroed = first_decode.pages_to_zero.at(group_id);
        EXPECT_EQ(std::ranges::find(zeroed, pairs.front().second), zeroed.end())
            << group_id << ": a copy destination is filled whole, never zeroed";
    }
    CompleteStores(first_decode);
    SendForwardDone("r1", {43});
    CompleteStores(PlanOnce());  // plan 3: decode
    SendForwardDone("r1", {44});
    // Plan 4 retracts r1 = [1 2 3 42 43 44]: 5 computed tokens, pages [1 2]
    // [3 42] and the partial page [43].
    const ExecutionPlan retraction = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    const auto write_backs = ExtractCacheOpsOfKind<WriteBackBatch>(retraction);
    ASSERT_EQ(write_backs.size(), 1u);
    const auto& l2_leg = std::get<WriteBackBatch>(write_backs.front());
    EXPECT_EQ(l2_leg.src_pages.at(0).size(), 4u) << "page [3 42] and the tail [43], in both groups";
    const SnapshotStoreBatch* store = FindSnapshotStore(retraction);
    ASSERT_NE(store, nullptr);
    EXPECT_TRUE(store->src_pages.at(0).empty()) << "every data slot is published; the pool carries the blob only";
    CompleteStores(retraction);
    EXPECT_EQ(scheduler_->TailEntries(), 2u) << "the prompt-end tail [3] and the retraction's tail [43]";

    const ExecutionPlan readmit = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr);
    ASSERT_EQ(restore->request_ids, std::vector<std::string>{"r1"});
    const auto& hashes = restore->content_hashes.at(0);
    const auto& tiers = restore->source_tiers.at(0);
    ASSERT_EQ(hashes.size(), 2u) << "the full pages are still Device-cached and claimed; only the tail is copied";
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        EXPECT_EQ(tiers[i], static_cast<std::uint8_t>(HostTier::kL2));
        EXPECT_TRUE(hashes[i].empty()) << "the partial page's row is a private copy, not republished";
    }
    AckRestores(readmit);
    EXPECT_EQ(scheduler_->TailEntries(), 2u) << "the published tails stay cached and enumerable";
    const ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(FindForwardBatch(decode)->request_ids, std::vector<std::string>{"r1"});
    EXPECT_EQ(scheduler_->RequestTokenSize("r1"), 6);
}

TEST_F(FusedRetractionL2TestSuite, AReadmissionThatDoesNotFitWaitsWithoutRetracting) {
    // Drive r1 into Retracted with an L2 snapshot while r2 keeps decoding.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    CompleteStores(prefill);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    CompleteStores(PlanOnce());
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});
    CompleteStores(PlanOnce());
    SendForwardDone("r1", {44});
    SendForwardDone("r2", {144});
    const ExecutionPlan retraction = PlanOnce();  // retracts r1; the grant serves r2's blocked decode
    CompleteStores(retraction);
    ASSERT_EQ(scheduler_->WaitingSize(), 1u);
    const auto decoding_before = scheduler_->DecodingSize();

    // r1's readmission cannot fit while r2 holds the pool. It must WAIT --
    // retracting r2 to readmit r1 would be pure thrash -- and r2's decodes
    // keep running unstalled.
    for (std::int32_t token = 145; token < 149; ++token) {
        const ExecutionPlan round = PlanOnce();
        EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "the readmission waits; nothing new is retracted";
        EXPECT_EQ(scheduler_->DecodingSize(), decoding_before)
            << "resident decodes are not sacrificed for a readmission";
        const ForwardBatch* op = FindForwardBatch(round);
        ASSERT_NE(op, nullptr);
        if (!op->request_ids.empty()) {
            SendForwardDone("r2", {token});
        }
        CompleteStores(round);
    }
}

// A decoding victim resumes decoding with the same tokens: nothing is rebased
// and nothing is recomputed, whatever RequestSpec::max_cached_prefix_tokens
// bounded its first admission to.
TEST_F(FusedRetractionL2TestSuite, ADecodingVictimResumesDecodingWithoutRecomputing) {
    // r1 returns prompt logprobs from position 0: nothing may be matched at
    // its first admission. r2 shares the pool and keeps decoding.
    RequestSpec capped = MakeRequestSpec("r1", /*num_pages=*/2);
    capped.max_cached_prefix_tokens = 0;
    Submit(capped);
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* first = FindForwardBatch(prefill);
    ASSERT_NE(first, nullptr);
    ASSERT_EQ(first->request_ids.size(), 2u);
    EXPECT_EQ(first->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(first->input_lengths.at(0), 4);
    CompleteStores(prefill);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    CompleteStores(PlanOnce());
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});
    CompleteStores(PlanOnce());
    SendForwardDone("r1", {44});
    SendForwardDone("r2", {144});
    // r1 = [1 2 3 4 42 43 44]: the blocked round retracts it; its published
    // pages [1 2] [3 4] [42 43] ride Host L2.
    const ExecutionPlan retraction = PlanOnce();
    CompleteStores(retraction);
    ASSERT_EQ(scheduler_->WaitingSize(), 1u);
    ASSERT_EQ(scheduler_->DecodingSize(), 1u);
    SendForwardDone("r2", {145});  // the decode the grant served

    // r2 leaves; r1 is restored. The bound of 0 would have recomputed all 7
    // tokens on a fresh admission; the restore copies the image back instead
    // and the next round is an ordinary decode step.
    const ExecutionPlan resumed = PlanOnce();
    const ForwardBatch* r2_only = FindForwardBatch(resumed);
    ASSERT_NE(r2_only, nullptr);
    ASSERT_EQ(r2_only->request_ids, std::vector<std::string>{"r2"});
    CompleteStores(resumed);
    SendForwardDone("r2", {146});
    SendFinish("r2");
    const ExecutionPlan readmit = PlanOnce();
    ASSERT_TRUE(FindForwardBatch(readmit)->request_ids.empty());
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr);
    ASSERT_EQ(restore->request_ids, std::vector<std::string>{"r1"});
    // r2's decode grant evicted some of r1's published Device blocks; those
    // come back from L2 (keyed rows), the ones still Device-cached are
    // claimed without a copy. Nothing rides the snapshot pool: 6 computed
    // tokens end on a page boundary.
    const auto& tiers = restore->source_tiers.at(0);
    EXPECT_EQ(std::ranges::count(tiers, static_cast<std::uint8_t>(HostTier::kSnapshotPool)), 0);
    EXPECT_EQ(std::ranges::count(tiers, static_cast<std::uint8_t>(HostTier::kL2)), 2)
        << "r2 took two of the six freed blocks; the other four stayed Device-cached and are claimed";
    for (const std::string& hash : restore->content_hashes.at(0)) {
        EXPECT_FALSE(hash.empty()) << "L2 rows carry their key so the ACK republishes them";
    }
    AckRestores(readmit);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0) << "the restore released the pins";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 10) << "the entries stay cached for ordinary reuse";
    CompleteStores(readmit);

    const ExecutionPlan decode = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(decode);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids, std::vector<std::string>{"r1"});
    EXPECT_EQ(op->NumExtends(), 0u);
    EXPECT_EQ(op->prefill_lengths.at(0), 4) << "the prompt window is untouched";
    EXPECT_EQ(scheduler_->RequestTokenSize("r1"), 7);

    // The restored pages are published again: a prompt sharing r1's first
    // six tokens hits all three.
    SendForwardDone("r1", {45});
    SendFinish("r1");
    CompleteStores(PlanOnce());  // r1's finish stores its tail; the ACK unpins the Device sources
    Submit(RequestSpec{.request_id = "r3", .tokens = {1, 2, 3, 4, 42, 43, 7, 8}});
    const ExecutionPlan hit = PlanOnce();
    const ForwardBatch* hit_op = FindForwardBatch(hit);
    ASSERT_NE(hit_op, nullptr);
    ASSERT_EQ(hit_op->request_ids, std::vector<std::string>{"r3"});
    EXPECT_EQ(hit_op->extend_prefix_lens.at(0), 6);
}

// The KV-event feed on: every Device publication mutates a boundary that must
// already carry its token descriptor, and a drain drops the descriptor of a
// boundary with no cached child.
class FusedRetractionKvEventsSuite : public FusedRetractionL2TestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FusedRetractionL2TestSuite::MakeConfig();
        cfg.enable_kv_cache_events = true;
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // The boundaries the feed currently reports as stored, keyed by their
    // tokens; a drain's events are applied in order (removals first).
    using Published = std::map<std::vector<std::int32_t>, std::uint64_t>;
    static void Apply(Published& published, const std::vector<KvCacheEvent>& events) {
        std::map<std::uint64_t, std::vector<std::int32_t>> by_hash;
        for (const auto& [tokens, hash] : published) {
            by_hash.emplace(hash, tokens);
        }
        for (const KvCacheEvent& event : events) {
            if (const auto* removed = std::get_if<KvBlockRemovedEvent>(&event)) {
                for (const std::uint64_t hash : removed->block_hashes) {
                    ASSERT_TRUE(by_hash.contains(hash)) << "a removal names a boundary never stored";
                    published.erase(by_hash.at(hash));
                }
            } else {
                const auto& stored = std::get<KvBlockStoredEvent>(event);
                ASSERT_EQ(stored.block_hashes.size(), 1u);
                ASSERT_TRUE(published.emplace(stored.token_ids, stored.block_hashes.front()).second)
                    << "a boundary is stored once until it is removed";
            }
        }
    }
    static std::size_t NumStored(const std::vector<KvCacheEvent>& events) {
        return static_cast<std::size_t>(std::ranges::count_if(
            events, [](const KvCacheEvent& event) { return std::holds_alternative<KvBlockStoredEvent>(event); }));
    }
    static std::size_t NumRemoved(const std::vector<KvCacheEvent>& events) { return events.size() - NumStored(events); }
};

TEST_F(FusedRetractionKvEventsSuite, RetractionAndRestorePublishWithTheirDescriptorsRegistered) {
    // Both publications a retraction makes happen outside an admission: the
    // retraction itself publishes the victim's decode-completed page (which
    // no admission has registered), and the restore's ACK republishes the
    // L2-tier destinations of boundaries whose descriptors a drain dropped
    // when the victim's pages left the Device. Either would be fatal without
    // the registration that precedes it.
    Published published;
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    CompleteStores(PlanOnce());
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    CompleteStores(PlanOnce());  // the first decodes publish both prompts' pages
    Apply(published, scheduler_->DrainKvEvents());
    EXPECT_EQ(published.size(), 4u) << "two prompt pages each";
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});
    CompleteStores(PlanOnce());
    SendForwardDone("r1", {44});
    SendForwardDone("r2", {144});

    // r1 = [1 2 3 4 42 43 44]: the blocked round retracts it, publishing the
    // decode page [42 43] on the way out, and grants two of its freed blocks
    // to r2's decode in the same round, which evicts two of r1's six
    // cache-only blocks: whichever boundary they belong to is incomplete by
    // the time of the drain, so it is either reported removed or (published
    // and evicted within the round) never reported at all -- and its
    // descriptor is dropped either way.
    const ExecutionPlan retraction = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    CompleteStores(retraction);
    Apply(published, scheduler_->DrainKvEvents());
    const std::vector<std::vector<std::int32_t>> r1_pages{{1, 2}, {3, 4}, {42, 43}};
    const auto missing = [&] {
        std::vector<std::vector<std::int32_t>> result;
        for (const auto& page : r1_pages) {
            if (!published.contains(page)) {
                result.push_back(page);
            }
        }
        return result;
    };
    const std::vector<std::vector<std::int32_t>> broken = missing();
    ASSERT_FALSE(broken.empty()) << "r2's grant evicted part of r1's published prefix";
    EXPECT_TRUE(published.contains({142, 143})) << "r2's own decode page was published by the same admission";
    SendForwardDone("r2", {145});  // the decode the grant served

    CompleteStores(PlanOnce());
    SendForwardDone("r2", {146});
    SendFinish("r2");
    Apply(published, scheduler_->DrainKvEvents());  // r2's finish publishes its own decode page
    EXPECT_EQ(missing(), broken);

    // The restore copies the evicted pages back from L2 and republishes them
    // at the ACK: every boundary of r1's is whole again, through freshly
    // registered descriptors, and nothing else moves.
    const ExecutionPlan readmit = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr);
    ASSERT_EQ(restore->request_ids, std::vector<std::string>{"r1"});
    EXPECT_EQ(restore->src_pages.at(0).size(), 2u * broken.size()) << "the evicted pages, in both groups";
    // The restore's own admission may evict r2's leftover cache-only entries
    // for its pages; it publishes nothing before the copy lands.
    const std::vector<KvCacheEvent> before_landing = scheduler_->DrainKvEvents();
    EXPECT_EQ(NumStored(before_landing), 0u);
    Apply(published, before_landing);
    EXPECT_EQ(missing(), broken);
    AckRestores(readmit);
    const std::vector<KvCacheEvent> after_restore = scheduler_->DrainKvEvents();
    EXPECT_EQ(NumStored(after_restore), broken.size());
    EXPECT_EQ(NumRemoved(after_restore), 0u);
    Apply(published, after_restore);
    EXPECT_TRUE(missing().empty()) << "every page of the restored request is a stored boundary again";

    const ExecutionPlan decode = PlanOnce();
    ASSERT_EQ(FindForwardBatch(decode)->request_ids, std::vector<std::string>{"r1"});
    SendForwardDone("r1", {45});
    SendFinish("r1");
    PlanOnce();
}

TEST_F(FusedRetractionL2TestSuite, TheImageWaitsForAnEarlierStoreThatCarriesOneOfItsKeys) {
    // r1's prompt pages are streamed to Host at its first decode (a pinned
    // store). That store is NOT acknowledged before r1 is retracted: the
    // image pins the in-flight ticket's Host block instead of copying the
    // page again, and waits for that older op as well as its own two legs.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(prefill).empty());
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    ExecutionPlan first_decode = PlanOnce();
    const auto pinned_stores = ExtractCacheOpsOfKind<WriteBackBatch>(first_decode);
    ASSERT_EQ(pinned_stores.size(), 1u) << "the first decode streams both prompts' pages";
    const auto& pinned = std::get<WriteBackBatch>(pinned_stores.front());
    ASSERT_EQ(pinned.op_ids.size(), 1u);
    EXPECT_EQ(pinned.source_pinned, std::vector<bool>{true});
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});
    // A pinned store in flight defers retraction: the blocked round stays
    // quiet until it is acknowledged... so acknowledge nothing yet and let
    // the next step fit its tail page first.
    ExecutionPlan tail = PlanOnce();
    ASSERT_EQ(FindForwardBatch(tail)->request_ids.size(), 2u);
    SendForwardDone("r1", {44});
    SendForwardDone("r2", {144});
    ExecutionPlan deferred = PlanOnce();
    EXPECT_EQ(scheduler_->RetractedSize(), 0u) << "an in-flight pinned store defers retraction";
    EXPECT_TRUE(FindForwardBatch(deferred)->request_ids.empty());

    // Acknowledging the pinned store lets the retraction happen; its own L2
    // leg copies only the page the pinned store did not carry.
    SendWriteBackDone(pinned.op_ids.front());
    ExecutionPlan retraction = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    const auto image_stores = ExtractCacheOpsOfKind<WriteBackBatch>(retraction);
    ASSERT_EQ(image_stores.size(), 1u);
    const auto& image_store = std::get<WriteBackBatch>(image_stores.front());
    EXPECT_EQ(image_store.src_pages.at(0).size(), 2u) << "[42 43] in both groups; the prompt pages are Host-warm";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 4) << "the image pins the Host-warm prompt pages";
    const TierTransferManager::WriteBackStats image_stats = scheduler_->GetHostCacheStats().retraction_stores;
    EXPECT_EQ(image_stats.host_cached, 4u) << "the two prompt pages in both groups";
    EXPECT_EQ(image_stats.in_flight, 0u) << "their store was acknowledged first";
    EXPECT_EQ(image_stats.ops, 1u);
    EXPECT_EQ(image_stats.blocks, (std::vector<std::uint64_t>{1, 1}));
    const SnapshotStoreBatch* tail_store = FindSnapshotStore(retraction);
    ASSERT_NE(tail_store, nullptr);
    SendForwardDone("r2", {145});
    SendFinish("r2");

    // Neither leg acknowledged: no restore. One leg: still none. Both: the
    // restore is issued.
    EXPECT_EQ(FindRestore(PlanOnce()), nullptr) << "no store ACK yet";
    SendWriteBackDone(image_store.op_ids.front());
    EXPECT_EQ(FindRestore(PlanOnce()), nullptr) << "the tail leg is still in flight";
    SendSnapshotDone(tail_store->op_ids.front());
    ExecutionPlan readmit = PlanOnce();
    ASSERT_NE(FindRestore(readmit), nullptr);
    EXPECT_EQ(FindRestore(readmit)->request_ids, std::vector<std::string>{"r1"});
}

TEST_F(FusedRetractionL2TestSuite, FinishWhileRetractedReleasesTheImage) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    CompleteStores(prefill);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    CompleteStores(PlanOnce());
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});
    CompleteStores(PlanOnce());
    SendForwardDone("r1", {44});
    SendForwardDone("r2", {144});
    const ExecutionPlan retraction = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    CompleteStores(retraction);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6);

    // The client gave up on the suspended request: its Host pins drop (the
    // entries stay cached) and its blob slot returns; the Device pool never
    // held anything of it.
    SendFinish("r1");
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 10);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 12);
    EXPECT_EQ(FindRestore(PlanOnce()), nullptr);
}

// An L3 prefetch of a key the retraction's L2 leg is also storing: whichever
// publication lands second is redirected to the first one's Host block, and
// the image must end up pinning the canonical entry rather than an unindexed
// block of its own. Mixed mode so the prefetching prompt is considered beside
// the victim's decode; the knob retracts the victim at the fourth plan.
class ImageFollowsCanonicalHostEntrySuite : public FusedRetractionL2TestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FusedRetractionL2TestSuite::MakeConfig();
        cfg.enable_l3_storage = true;
        cfg.storage_config = StorageConfig{16,
                                           32,
                                           cfg.host_allocator.total_pages - 1,
                                           64,
                                           32,
                                           2,
                                           100,
                                           100,
                                           StoragePressure{16, 1048576, 1000},
                                           StoragePressure{8, 524288, 500}};
        cfg.enable_mixed_prefill_decode = true;
        cfg.debug_force_retraction_interval = 6;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(ImageFollowsCanonicalHostEntrySuite, ARedirectedStoreAckRePointsTheImageToTheCanonicalEntry) {
    // r1 = [1 2 3 4] decodes 42, 43, 44: its prompt pages stream to Host at
    // the first decode; the decode page [42 43] completes at plan 3 and is
    // published only when r1 is retracted at plan 4.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();  // plan 1
    SendForwardDone("r1", {42});
    const ExecutionPlan p2 = PlanOnce();
    AckWriteBacks(p2);  // K1, K2 are Host-cached (and remembered as L3 objects)
    SendForwardDone("r1", {43});

    // r3 shares r1's first six tokens. Its third page [42 43] is not on the
    // Device yet (r1 publishes it only at its retraction) but is registered as
    // an L3 object, so at plan 3 r3 goes to prefetch it into a fresh Host
    // block beside r1's decode.
    const RequestSpec r3{.request_id = "r3", .tokens = {1, 2, 3, 4, 42, 43, 7, 8}};
    const std::vector<std::string> hashes = scheduler_->PrefixHashesForTokens(r3.tokens);
    ASSERT_EQ(hashes.size(), 3u);
    Submit(r3);
    ExecutionEvent tick;
    tick.With(cache::StorageTick{1, false, true, StoragePressure{0, 0, 0}});
    scheduler_->Advance(std::move(tick));
    const auto query_plan = PlanOnce();
    std::vector<StorageOperation> query_operations;
    for (const auto& operation : query_plan.Operations()) {
        if (const auto* cache = std::get_if<CacheOperation>(&operation)) {
            if (const auto* storage = std::get_if<StorageOperation>(cache)) query_operations.push_back(*storage);
        }
    }
    ASSERT_EQ(query_operations.size(), 1u);
    const auto& query = std::get<StorageQueryOperation>(query_operations.front());
    std::vector<std::uint8_t> present;
    for (const auto& key : query.keys) {
        // This race stores complete pages only, not r3's unrelated tail.
        present.push_back(std::ranges::find(hashes, key.content_hash) != hashes.end());
    }
    ExecutionEvent query_done;
    query_done.With(cache::StorageDone{query.op_id, present, true, std::vector<std::uint8_t>(query.keys.size(), 0)});
    scheduler_->Advance(std::move(query_done));
    const auto get_plan = PlanOnce();
    std::vector<StorageOperation> get_operations;
    for (const auto& operation : get_plan.Operations()) {
        if (const auto* cache = std::get_if<CacheOperation>(&operation)) {
            if (const auto* storage = std::get_if<StorageOperation>(cache)) get_operations.push_back(*storage);
        }
    }
    ASSERT_EQ(get_operations.size(), 1u);
    const auto& get = std::get<StorageGetOperation>(get_operations.front());
    ASSERT_EQ(get.keys.size(), 2u) << "[42 43] in both groups";
    ExecutionEvent resume;
    resume.With(cache::StorageTick{2, true, false, StoragePressure{0, 0, 0}});
    scheduler_->Advance(std::move(resume));
    const ExecutionPlan p3 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p3)->request_ids, std::vector<std::string>{"r1"});
    SendForwardDone("r1", {44});

    // Plan 4: the knob retracts r1. Its L2 leg finds [42 43] neither
    // Host-cached nor on an in-flight store (the prefetch is no store) and
    // copies it into Host blocks of its own.
    const ExecutionPlan p4 = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    const auto write_backs = ExtractCacheOpsOfKind<WriteBackBatch>(p4);
    ASSERT_EQ(write_backs.size(), 1u);
    const auto& batch = std::get<WriteBackBatch>(write_backs.front());
    ASSERT_EQ(batch.op_ids.size(), 1u);
    ASSERT_FALSE(batch.source_pinned.at(0));
    ASSERT_EQ(batch.src_pages.at(0).size(), 2u) << "[42 43] in both groups";
    const std::uint32_t image_store = batch.op_ids.at(0);
    const std::int32_t host_free_during_race = scheduler_->HostPoolFreeBlocks();

    // The prefetch lands first: its blocks become canonical for [42 43].
    ExecutionEvent get_done;
    get_done.With(cache::StorageDone{get.op_id, std::vector<std::uint8_t>(get.keys.size(), 1), true,
                                     std::vector<std::uint8_t>(get.keys.size(), 0)});
    scheduler_->Advance(std::move(get_done));
    const std::int32_t host_cached_after_prefetch = scheduler_->HostPoolCachedBlocks();
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6) << "r3 pins its two, the image pins K1, K2";
    // The image's store lands second and is redirected: the image follows the
    // canonical entries, its own two blocks return to the pool, and every
    // Host entry of the image is pinned by it (and [42 43] by r3 as well).
    SendWriteBackDone(image_store);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), host_cached_after_prefetch) << "no second entry for [42 43]";
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), host_free_during_race + 2) << "the redirected store's blocks return";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6);

    // r3 admits on the Host hit (K1, K2 and the prefetched [42 43]), then
    // leaves; a churn prompt evicts every cache-only Device block before the
    // image lands, so the restore must copy [42 43] back from Host.
    const ExecutionPlan admit = PlanOnce();
    ASSERT_EQ(FindForwardBatch(admit)->request_ids, std::vector<std::string>{"r3"});
    EXPECT_EQ(FindForwardBatch(admit)->extend_prefix_lens.at(0), 6);
    for (const CacheOperation& op : ExtractCacheOpsOfKind<LoadBackBatch>(admit)) {
        for (const std::uint32_t id : std::get<LoadBackBatch>(op).op_ids) {
            SendLoadBackDone(id);
        }
    }
    AckWriteBacks(admit);
    SendForwardDone("r3", {9});
    AckWriteBacks(PlanOnce());
    SendForwardDone("r3", {10});
    SendFinish("r3");
    AckWriteBacks(PlanOnce());
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6) << "the image's six pins remain";
    // This unrelated prompt tests eviction, without starting another L3 read.
    ExecutionEvent no_storage;
    no_storage.With(cache::StorageTick{3, true, false, StoragePressure{0, 0, 0}});
    scheduler_->Advance(std::move(no_storage));
    Submit(MakeRequestSpec("churn", /*num_pages=*/4, /*start=*/501));
    const ExecutionPlan churn = PlanOnce();
    ASSERT_EQ(FindForwardBatch(churn)->request_ids, std::vector<std::string>{"churn"});
    AckWriteBacks(churn);
    SendForwardDone("churn", {509});
    SendFinish("churn");
    AckWriteBacks(PlanOnce());
    for (const CacheOperation& op : ExtractCacheOpsOfKind<SnapshotStoreBatch>(p4)) {
        for (const std::uint32_t id : std::get<SnapshotStoreBatch>(op).op_ids) {
            SendSnapshotDone(id);
        }
    }

    const ExecutionPlan readmit = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr) << "a pinned entry the restore cannot find would have asserted here";
    ASSERT_EQ(restore->request_ids, std::vector<std::string>{"r1"});
    EXPECT_GE(restore->src_pages.at(0).size(), 2u) << "[42 43] comes back from the canonical Host entries";
    for (const std::uint8_t tier : restore->source_tiers.at(0)) {
        EXPECT_EQ(tier, static_cast<std::uint8_t>(HostTier::kL2));
    }
    AckRestores(readmit);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
    EXPECT_EQ(scheduler_->RequestTokenSize("r1"), 7);
    AckWriteBacks(readmit);
}

// The D role's admission probes Device alone: its cache arrives by the
// ordinary remote landing, so even with L3 enabled and storage allowed it
// must never start a query, a GET or a backup. Dropping the kD check in
// prepareStorage would emit a StorageQueryOperation at this plan.
class L3DecodeRoleSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.role = Role::kD;
        cfg.enable_l3_storage = true;
        cfg.storage_config = StorageConfig{16,
                                           32,
                                           cfg.host_allocator.total_pages - 1,
                                           64,
                                           32,
                                           2,
                                           100,
                                           100,
                                           StoragePressure{16, 1048576, 1000},
                                           StoragePressure{8, 524288, 500}};
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.transfer_policy = CacheTransferPolicy::FullSuffix;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{request_id});
        scheduler_->Advance(std::move(event));
    }
};

TEST_F(L3DecodeRoleSuite, TheDecodeRoleNeverPreparesStorage) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    // Storage healthy and idle: a role that prepares storage would query now.
    ExecutionEvent tick;
    tick.With(cache::StorageTick{1, true, true, StoragePressure{0, 0, 0}});
    scheduler_->Advance(std::move(tick));
    SendBootstrapped("r1");
    const ExecutionPlan plan = PlanOnce();
    EXPECT_TRUE(ExtractCacheOpsOfKind<StorageOperation>(plan).empty());
    const ForwardBatch* admission = FindRemoteAdmission(plan);
    ASSERT_NE(admission, nullptr);
    EXPECT_EQ(admission->request_ids, std::vector<std::string>{"r1"});
    EXPECT_EQ(admission->extend_prefix_lens.at(0), 0) << "the whole prompt is the peer's work";
}

TEST_F(L3DecodeRoleSuite, AClosedPressureGateIsNotADecodeBypass) {
    // The gate closes at the high watermark, but a request that never
    // prepares storage is not "skipped by pressure".
    ExecutionEvent tick;
    tick.With(cache::StorageTick{1, true, true, StoragePressure{16, 0, 0}});
    scheduler_->Advance(std::move(tick));
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    SendBootstrapped("r1");
    const ExecutionPlan plan = PlanOnce();
    EXPECT_TRUE(ExtractCacheOpsOfKind<StorageOperation>(plan).empty());
    EXPECT_TRUE(scheduler_->GetStorageStats().pressure_gate_closed);
    EXPECT_EQ(scheduler_->GetStorageStats().pressure_bypasses, 0);
}

// The L3 prefetch window (operations/storage.cpp): the lookahead opens at
// most max_prefetch_tokens past the local hit, and a gap below
// min_prefetch_tokens starts nothing. Both edges are observable in the
// emitted query's keys.
class L3PrefetchWindowSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 128;
        cfg.host_allocator.total_pages = 64;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 128;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        cfg.enable_l3_storage = true;
        cfg.storage_config = StorageConfig{16,
                                           32,
                                           cfg.host_allocator.total_pages - 1,
                                           64,
                                           32,
                                           2,
                                           100,
                                           100,
                                           StoragePressure{16, 1048576, 1000},
                                           StoragePressure{8, 524288, 500}};
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(L3PrefetchWindowSuite, TheLookaheadStopsAtMaxPrefetchTokensPastTheLocalHit) {
    // "warm" leaves [1 2 3 4] in the Device cache but its last page stays the
    // replay tail while it decodes, so "long" probes a 2-token local hit.
    // The window opens at 2 + 32 = 34 tokens, so the query covers pages
    // 2..17 only: 16 keys, not the 32 an uncapped remainder would fill
    // (max_keys_per_operation).
    Submit(MakeRequestSpec("warm", /*num_pages=*/2));
    PlanOnce();
    SendForwardDone("warm", {42});

    Submit(MakeRequestSpec("long", /*num_pages=*/40));
    ExecutionEvent tick;
    tick.With(cache::StorageTick{1, false, true, StoragePressure{0, 0, 0}});
    scheduler_->Advance(std::move(tick));
    const ExecutionPlan plan = PlanOnce();
    const auto queries = ExtractCacheOpsOfKind<StorageOperation>(plan);
    ASSERT_EQ(queries.size(), 1u);
    const auto& query = std::get<StorageQueryOperation>(std::get<StorageOperation>(queries.front()));
    EXPECT_EQ(query.keys.size(), 16u);
}

TEST_F(L3PrefetchWindowSuite, AGapBelowMinPrefetchTokensStartsNoQuery) {
    // A finished "warm" leaves its whole 78-token prefix matchable; "long"
    // adds a single cacheable token past it. The gap (1) is below
    // min_prefetch_tokens (2), so "long" takes the ordinary admission with
    // its local hit and no query starts.
    Submit(MakeRequestSpec("warm", /*num_pages=*/39));
    PlanOnce();
    SendForwardDone("warm", {42});
    SendFinish("warm");
    PlanOnce();  // the finish settles; the pages stay cached

    Submit(MakeRequestSpec("long", /*num_pages=*/40));
    ExecutionEvent tick;
    tick.With(cache::StorageTick{1, true, true, StoragePressure{0, 0, 0}});
    scheduler_->Advance(std::move(tick));
    const ExecutionPlan plan = PlanOnce();
    EXPECT_TRUE(ExtractCacheOpsOfKind<StorageOperation>(plan).empty());
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, std::vector<std::string>{"long"});
}

// Two victims whose images share pages carried by one earlier, still
// unacknowledged store: neither copies them again, and both wait for it.
class SharedImagePagesSuite : public FusedRetractionL2TestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FusedRetractionL2TestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 33;
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        SetTestSnapshotPool(cfg);
        cfg.debug_force_retraction_interval = 3;  // plans 3 and 6 retract the oldest quiescent decode
        return cfg;
    }
};

TEST_F(SharedImagePagesSuite, VictimsPinAnInFlightStoresBlocksAndWaitForItsAck) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();  // plan 1: r1 prefills
    SendForwardDone("r1", {42});
    const ExecutionPlan first_decode = PlanOnce();  // plan 2: r1's first decode streams its prompt pages (pinned)
    const auto pinned_stores = ExtractCacheOpsOfKind<WriteBackBatch>(first_decode);
    ASSERT_EQ(pinned_stores.size(), 1u);
    const auto& pinned = std::get<WriteBackBatch>(pinned_stores.front());
    ASSERT_EQ(pinned.src_pages.at(0).size(), 4u) << "two prompt pages in two groups";
    SendForwardDone("r1", {43});

    // Plan 3 retracts r1 (quiescent). Its prompt pages are travelling on the
    // pinned store, so the image pins that store's Host blocks; the partial
    // page with token 42 is published as a tail by the retraction and is the
    // only block its own L2 leg copies, so the snapshot pool carries the blob
    // alone. r2, sharing r1's prompt, hits those pages on Device.
    Submit(MakeRequestSpec("r2", /*num_pages=*/2));
    const ExecutionPlan p3 = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    const auto p3_stores = ExtractCacheOpsOfKind<WriteBackBatch>(p3);
    ASSERT_EQ(p3_stores.size(), 1u);
    const auto& r1_l2_leg = std::get<WriteBackBatch>(p3_stores.front());
    EXPECT_EQ(r1_l2_leg.src_pages.at(0).size(), 2u) << "the tail page with token 42, in both groups";
    EXPECT_EQ(r1_l2_leg.source_pinned, std::vector<bool>{false}) << "an image leg is stream-ordered";
    const TierTransferManager::WriteBackStats r1_image = scheduler_->GetHostCacheStats().retraction_stores;
    EXPECT_EQ(r1_image.in_flight, 4u) << "the two prompt pages in both groups ride the pinned store";
    EXPECT_EQ(r1_image.host_cached, 0u);
    EXPECT_EQ(r1_image.blocks, (std::vector<std::uint64_t>{1, 1}));
    const SnapshotStoreBatch* r1_tail = FindSnapshotStore(p3);
    ASSERT_NE(r1_tail, nullptr);
    EXPECT_TRUE(r1_tail->src_pages.at(0).empty()) << "every data slot is published: the pool carries the blob only";
    const ForwardBatch* r2_prefill = FindForwardBatch(p3);
    ASSERT_EQ(r2_prefill->request_ids, std::vector<std::string>{"r2"});
    EXPECT_EQ(r2_prefill->extend_prefix_lens.at(0), 2) << "r2 hits r1's first page (the second is the replay tail)";
    SendForwardDone("r2", {42});
    const ExecutionPlan p4 = PlanOnce();  // r2's first decode
    ASSERT_EQ(FindForwardBatch(p4)->request_ids, std::vector<std::string>{"r2"});
    SendForwardDone("r2", {43});
    const ExecutionPlan p5 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p5)->request_ids, std::vector<std::string>{"r2"});
    SendForwardDone("r2", {44});

    // Plan 6 retracts r2: the shared page is still on the pinned store, so
    // r2 pins the same ticket block and waits for the same op.
    const ExecutionPlan p6 = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 2u);
    const TierTransferManager::WriteBackStats both_images = scheduler_->GetHostCacheStats().retraction_stores;
    EXPECT_EQ(both_images.in_flight, 8u) << "r2's two prompt pages ride the same pinned store";
    EXPECT_EQ(both_images.ops, 2u);
    EXPECT_EQ(both_images.blocks, (std::vector<std::uint64_t>{2, 2}));
    const SnapshotStoreBatch* r2_tail = FindSnapshotStore(p6);
    ASSERT_NE(r2_tail, nullptr);
    AckImageStores(p3);
    AckImageStores(p4);
    AckImageStores(p5);
    AckImageStores(p6);
    EXPECT_EQ(FindRestore(PlanOnce()), nullptr) << "both images wait for the pinned store they share";
    EXPECT_EQ(scheduler_->RetractedSize(), 2u);

    SendWriteBackDone(pinned.op_ids.front());
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), scheduler_->HostPoolCachedBlocks())
        << "every published Host entry is pinned by an image";
    const ExecutionPlan restore_r1 = PlanOnce();
    ASSERT_NE(FindRestore(restore_r1), nullptr);
    EXPECT_EQ(FindRestore(restore_r1)->request_ids, std::vector<std::string>{"r1"}) << "the older victim first";
    AckRestores(restore_r1);
    const ExecutionPlan restore_r2 = PlanOnce();  // plan 9: the knob also strikes the restored r1 again
    ASSERT_NE(FindRestore(restore_r2), nullptr);
    EXPECT_EQ(FindRestore(restore_r2)->request_ids, std::vector<std::string>{"r2"});
    AckRestores(restore_r2);
    AckImageStores(restore_r2);
    EXPECT_EQ(scheduler_->RequestTokenSize("r2"), 7) << "resumed with the same tokens";
    SendAbortEvent("r1");
    SendAbortEvent("r2");
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 32);
}

// The debug knob retracts a decoding request on schedule, exemption or not,
// so a test oracle can force a retract/restore cycle on any request.
class ForcedRetractionSuite : public CapacityBlockSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = CapacityBlockSuite::MakeConfig();
        cfg.device_allocator.total_pages = 33;  // ample: nothing is ever blocked for capacity
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        SetTestSnapshotPool(cfg);
        cfg.debug_force_retraction_interval = 3;
        return cfg;
    }
};

TEST_F(ForcedRetractionSuite, EveryThirdPlanRetractsTheOldestQuiescentDecodingRequest) {
    RequestSpec exempt = MakeRequestSpec("r1", /*num_pages=*/2);
    exempt.max_new_tokens = 4;  // ReserveCoversGeneration would exempt it from the victim policy
    Submit(exempt);
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan p1 = PlanOnce();  // plan 1: both prefill
    ASSERT_EQ(FindForwardBatch(p1)->request_ids.size(), 2u);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    ExecutionPlan p2 = PlanOnce();  // plan 2: both decode
    ASSERT_EQ(FindForwardBatch(p2)->request_ids.size(), 2u);
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});

    // Plan 3 arms the oldest Decoding request, r1 -- quiescent, so it is
    // retracted at once, exemption notwithstanding -- while r2 decodes on.
    ExecutionPlan p3 = PlanOnce();
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    EXPECT_EQ(FindForwardBatch(p3)->request_ids, std::vector<std::string>{"r2"});
    ASSERT_NE(FindSnapshotStore(p3), nullptr);
    EXPECT_EQ(FindSnapshotStore(p3)->request_ids, std::vector<std::string>{"r1"});
    AckImageStores(p3);
    const std::int32_t r1_tokens = scheduler_->RequestTokenSize("r1");

    // Plan 4 restores it (nothing blocks the restore); plan 5 decodes both.
    ExecutionPlan p4 = PlanOnce();
    ASSERT_NE(FindRestore(p4), nullptr);
    EXPECT_EQ(FindRestore(p4)->request_ids, std::vector<std::string>{"r1"});
    AckRestores(p4);
    SendForwardDone("r2", {144});
    ExecutionPlan p5 = PlanOnce();
    EXPECT_EQ(FindForwardBatch(p5)->request_ids, (std::vector<std::string>{"r1", "r2"}));
    EXPECT_EQ(scheduler_->RequestTokenSize("r1"), r1_tokens);
    // Both are in flight at plan 6: the armed victim (r1 again, the oldest)
    // sits the round out and is retracted at plan 7 once its result landed.
    SendForwardDone("r2", {145});
    ExecutionPlan p6 = PlanOnce();
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(FindForwardBatch(p6)->request_ids, std::vector<std::string>{"r2"}) << "the armed victim sits out";
    SendForwardDone("r1", {44});
    SendForwardDone("r2", {146});
    ExecutionPlan p7 = PlanOnce();
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    ASSERT_NE(FindSnapshotStore(p7), nullptr);
    EXPECT_EQ(FindSnapshotStore(p7)->request_ids, std::vector<std::string>{"r1"});
}

// ---------------------------------------------------------------------------
// Cache retract: a blocked round picks the decoding request that frees the
// most and has streamed least, releases every page and suspends it with its
// image; the restore continues it where it stopped. Accepted request lengths
// must obey Scheduler::MaxSingleRequestTokens().
// ---------------------------------------------------------------------------
class RetractSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        // 15 physical pages -> 14 usable: "a" (3-page prompt) charges
        // 2*ceil(7/2) = 8 and "b" (2-page prompt) 2*ceil(5/2) = 6 = the pool.
        cfg.device_allocator.total_pages = 15;
        cfg.host_allocator.total_pages = 16;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full_a", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("full_b", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Drives "a" (6-token prompt) and "b" (4-token prompt) into the exact-fit
    // capacity block that retracts "a".
    // Post: "a" Retracted with 9 tokens (8 computed), "b" Decoding with 7
    // tokens, free = 8; a's image store rides retract_round_.
    void DriveToRetractOfA() {
        ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 14);
        Submit(MakeRequestSpec("a", /*num_pages=*/3));
        Submit(MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101));

        ExecutionPlan prefill = PlanOnce();
        const ForwardBatch* prefill_op = FindForwardBatch(prefill);
        ASSERT_NE(prefill_op, nullptr);
        ASSERT_EQ(prefill_op->request_ids.size(), 2u);
        ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
        SendForwardDone("a", {42});
        SendForwardDone("b", {142});

        // Both decode transitions consume their reservations: free 0.
        ExecutionPlan decode = PlanOnce();
        const ForwardBatch* decode_op = FindForwardBatch(decode);
        ASSERT_NE(decode_op, nullptr);
        ASSERT_EQ(decode_op->request_ids.size(), 2u);
        ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
        SendForwardDone("a", {43});   // 8 tokens = a's capacity
        SendForwardDone("b", {143});  // 6 tokens = b's capacity

        // Both next steps still fit their tail pages (0 fresh blocks).
        ExecutionPlan tail_round = PlanOnce();
        const ForwardBatch* tail_op = FindForwardBatch(tail_round);
        ASSERT_NE(tail_op, nullptr);
        ASSERT_EQ(tail_op->request_ids.size(), 2u);
        SendForwardDone("a", {44});   // 9 tokens: past capacity
        SendForwardDone("b", {144});  // 7 tokens: past capacity

        // The first fully blocked round retracts "a" (4 pages x 2 groups free
        // 8 blocks against b's 6) and, a being the blocker itself, grants a
        // page pair to the other blocked decode, b, in the same round.
        retract_round_ = PlanOnce();
        const ForwardBatch* retract_op = FindForwardBatch(retract_round_);
        ASSERT_NE(retract_op, nullptr);
        ASSERT_EQ(retract_op->request_ids, std::vector<std::string>{"b"});
        ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 8 - 2) << "a's 4 pages x 2 groups return; b takes a pair";
        ASSERT_EQ(scheduler_->WaitingSize(), 1u) << "a is suspended with its image";
        ASSERT_EQ(scheduler_->DecodingSize(), 1u);
        ASSERT_NE(FindSnapshotStore(retract_round_), nullptr);
        SendForwardDone("b", {145});  // 8 tokens; b is quiescent again
    }

    ExecutionPlan retract_round_;
};

TEST_F(RetractSuite, DecodingVictimResumesDecodingWithTheSameTokens) {
    DriveToRetractOfA();
    EXPECT_EQ(scheduler_->RequestTokenSize("a"), 9);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 14 - 8) << "8 computed tokens = 4 pages x 2 groups imaged";

    // The survivor proceeds on the freed pages while a waits; its image is
    // acknowledged meanwhile.
    ExecutionPlan resumed = PlanOnce();
    const ForwardBatch* resumed_op = FindForwardBatch(resumed);
    ASSERT_NE(resumed_op, nullptr);
    ASSERT_EQ(resumed_op->request_ids, std::vector<std::string>{"b"});
    AckImageStores(retract_round_);
    SendForwardDone("b", {146});
    SendFinish("b");

    // b reaped -> a is restored: no forward, one restore op for its 8 pages
    // plus the decode slot it needs next.
    ExecutionPlan readmit = PlanOnce();
    ASSERT_TRUE(FindForwardBatch(readmit)->request_ids.empty());
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr);
    ASSERT_EQ(restore->request_ids, std::vector<std::string>{"a"});
    EXPECT_EQ(restore->src_pages.at(0).size(), 8u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 14 - 10) << "4 imaged pages + 1 decode slot, per group";
    AckRestores(readmit);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 14);

    ExecutionPlan decode = PlanOnce();
    const ForwardBatch* decode_op = FindForwardBatch(decode);
    ASSERT_NE(decode_op, nullptr);
    ASSERT_EQ(decode_op->request_ids, std::vector<std::string>{"a"});
    EXPECT_EQ(decode_op->NumExtends(), 0u) << "a decoding victim resumes with a decode step";
    EXPECT_EQ(decode_op->prefill_lengths.at(0), 6) << "the prompt window is not rebased";
    EXPECT_EQ(decode_op->input_lengths.at(0), 1);
    EXPECT_EQ(decode_op->decode_input_ids.at(0), 44)
        << "the first decode after a restore carries its input: the new slot holds no in-flight capture";
    SendForwardDone("a", {45});
    ExecutionPlan second = PlanOnce();
    const ForwardBatch* second_op = FindForwardBatch(second);
    ASSERT_NE(second_op, nullptr);
    ASSERT_EQ(second_op->request_ids, std::vector<std::string>{"a"});
    EXPECT_EQ(second_op->decode_input_ids.at(0), -1) << "from then on the fused decode reads its own capture";
    SendForwardDone("a", {46});
    SendFinish("a");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 14) << "pool balances after the full retract cycle";
}

TEST_F(RetractSuite, ARestoreWaitsForBothStoreAcksAndNeverRunsAPrefill) {
    DriveToRetractOfA();
    SendFinish("b");
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 14);

    // The pool is free, but a's image has not landed: no restore yet, and no
    // prefill of a's tokens either.
    for (int round = 0; round < 3; ++round) {
        ExecutionPlan waiting = PlanOnce();
        EXPECT_TRUE(FindForwardBatch(waiting)->request_ids.empty());
        EXPECT_EQ(FindRestore(waiting), nullptr) << "round " << round;
        EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    }
    AckImageStores(retract_round_);
    ExecutionPlan readmit = PlanOnce();
    ASSERT_NE(FindRestore(readmit), nullptr);
    EXPECT_TRUE(FindForwardBatch(readmit)->request_ids.empty());
}

TEST_F(RetractSuite, AFlushIsRefusedWhileARequestIsSuspendedWithASnapshotOnlyImage) {
    // Without a Host cache the whole image rides the snapshot pool, which no
    // prefix index sees: once the stores are acknowledged and the survivor
    // is gone, nothing is in flight and no cached block is pinned -- only the
    // suspended request itself stands between a flush and KV that would be
    // restored under new weights.
    DriveToRetractOfA();
    AckImageStores(retract_round_);
    SendFinish("b");
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    ASSERT_EQ(scheduler_->HostPoolPinnedBlocks(), 0) << "no Host cache: the image is visible through no pin";
    EXPECT_FALSE(scheduler_->CanClearCache());
    EXPECT_FALSE(scheduler_->ClearCache());
    EXPECT_FALSE(scheduler_->ClearL1Cache());

    // Restoring counts as suspended too: the image is still the request's KV.
    ExecutionPlan readmit = PlanOnce();
    ASSERT_NE(FindRestore(readmit), nullptr);
    EXPECT_FALSE(scheduler_->CanClearCache());
    AckRestores(readmit);
    ASSERT_EQ(scheduler_->RetractedSize(), 0u);

    // With the request resumed and then finished, the flush goes through.
    SendFinish("a");
    PlanOnce();
    EXPECT_TRUE(scheduler_->CanClearCache());
    EXPECT_TRUE(scheduler_->ClearCache());
}

TEST_F(RetractSuite, AbortWhileRetractedReleasesTheImage) {
    DriveToRetractOfA();
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 14 - 8);
    SendAbortEvent("a");
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 14 - 8) << "the in-flight store still pins its destinations";
    AckImageStores(retract_round_);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 14) << "the ACK of an aborted victim's store frees the pool";
    EXPECT_EQ(FindRestore(PlanOnce()), nullptr) << "an aborted victim is never restored";
}

TEST_F(RetractSuite, AbortWhileRestoringFreesPagesOnceTheCopyIsAcknowledged) {
    DriveToRetractOfA();
    AckImageStores(retract_round_);
    SendFinish("b");
    ExecutionPlan readmit = PlanOnce();
    ASSERT_NE(FindRestore(readmit), nullptr);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 14 - 10);

    SendAbortEvent("a");
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    // The copy may still be writing the pages: the op pins both ends until
    // its ACK, so the pages are not re-granted meanwhile.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 14 - 8) << "the decode slot freed; the 8 copy destinations are pinned";
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 14 - 8);
    AckRestores(readmit);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 14);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 14);
    PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
}

// The slot-state blob slot and the request-pool row an image op uses belong
// to the op until its ACK, whatever the request does meanwhile: one blob
// slot and two request rows, so a held slot or row is observable.
class SlotPinSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        cfg.max_batch_size = 2;
        cfg.max_retracted_requests = 1;
        return cfg;
    }
};

TEST_F(SlotPinSuite, AnAbortWhileRestoringKeepsTheRowAndSlotUntilTheAck) {
    DriveToRetractOfA();
    AckImageStores(retract_round_);
    SendFinish("b");
    const ExecutionPlan readmit = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr);
    const std::int32_t restored_row = restore->request_pool_indices.at(0);
    ASSERT_EQ(restore->snapshot_slots.at(0), 1) << "the one blob slot";

    // The client gives up while the import is still writing row and slot:
    // both stay with the op. A newcomer takes the other row; a second one
    // finds none until the ACK.
    const std::int32_t device_free_restoring = scheduler_->AvailableLcmBlocks();
    SendAbortEvent("a");
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 14 - 8) << "the image's pool blocks are the copy's sources";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), device_free_restoring + 2)
        << "only the decode slots return now; the 8 copy destinations stay pinned";
    Submit(MakeRequestSpec("c", /*num_pages=*/1, /*start=*/201));
    const ExecutionPlan admit_c = PlanOnce();
    const ForwardBatch* c_op = FindForwardBatch(admit_c);
    ASSERT_NE(c_op, nullptr);
    ASSERT_EQ(c_op->request_ids, std::vector<std::string>{"c"});
    EXPECT_NE(c_op->request_pool_indices.at(0), restored_row) << "the row being imported into is not re-granted";
    SendForwardDone("c", {207});
    Submit(MakeRequestSpec("d", /*num_pages=*/1, /*start=*/301));
    const ExecutionPlan no_row = PlanOnce();
    EXPECT_EQ(FindForwardBatch(no_row)->request_ids, std::vector<std::string>{"c"}) << "d waits for a row";
    SendForwardDone("c", {208});

    AckRestores(readmit);
    const ExecutionPlan admit_d = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(admit_d);
    ASSERT_NE(batch, nullptr);
    const auto d_row = std::ranges::find(batch->request_ids, "d");
    ASSERT_NE(d_row, batch->request_ids.end()) << "the ACK returned the row";
    EXPECT_EQ(
        batch->request_pool_indices.at(static_cast<std::size_t>(std::distance(batch->request_ids.begin(), d_row))),
        restored_row);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 14) << "the image is gone with the ACK";
}

class SlotPinKnobSuite : public SlotPinSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SlotPinSuite::MakeConfig();
        cfg.debug_force_retraction_interval = 5;  // plan 5 strikes right after DriveToRetractOfA's plan 4
        return cfg;
    }
};

TEST_F(SlotPinKnobSuite, AnAbortWhileRetractedKeepsTheBlobSlotUntilTheStoreAck) {
    DriveToRetractOfA();  // a is Retracted on the one blob slot; its tail store is in flight
    SendAbortEvent("a");
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);

    // Plan 5: the knob arms b, but the slot is still the in-flight store's,
    // so the forced retraction is refused and b decodes on.
    const ExecutionPlan p5 = PlanOnce();
    EXPECT_EQ(scheduler_->RetractedSize(), 0u) << "no blob slot: the export into it has not been acknowledged";
    EXPECT_EQ(FindSnapshotStore(p5), nullptr);
    EXPECT_TRUE(p5.aborts.empty()) << "a forced retraction refused before its publication aborts nobody";
    ASSERT_EQ(FindForwardBatch(p5)->request_ids, std::vector<std::string>{"b"});
    SendForwardDone("b", {146});

    // The ACK returns the slot; the knob's next strike retracts b onto it.
    AckImageStores(retract_round_);
    for (int plan = 6; plan < 10; ++plan) {
        const ExecutionPlan decode = PlanOnce();
        ASSERT_EQ(FindForwardBatch(decode)->request_ids, std::vector<std::string>{"b"});
        SendForwardDone("b", {140 + plan});
    }
    const ExecutionPlan p10 = PlanOnce();
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    const SnapshotStoreBatch* store = FindSnapshotStore(p10);
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->request_ids, std::vector<std::string>{"b"});
    EXPECT_EQ(store->snapshot_slots.at(0), 1) << "the slot the aborted image held";
}

// The host side is finite: when no candidate's image fits the snapshot pool
// (or no blob slot is free), the capacity retraction aborts the newest
// retractable resident instead of imaging anyone -- its pages free in the
// same round and the blocked grant proceeds. RetractSuite's pool arithmetic:
// "a" (3-page prompt, decoding at 9 tokens) images 4 pages x 2 groups = 8
// blocks, "b" (2-page prompt, at 7 tokens) 3 x 2 = 6.
class ImageDoesNotFitSuite : public RetractSuite {
protected:
    virtual std::int32_t SnapshotPoolBlocks() const = 0;

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        cfg.snapshot_allocator.total_pages = SnapshotPoolBlocks() + 1;
        // The null page alone takes no blob slots (Validate ties the two).
        cfg.max_retracted_requests = SnapshotPoolBlocks() == 0 ? 0 : cfg.max_batch_size;
        return cfg;
    }

    // Both residents decode to the exact-fit capacity block (RetractSuite's
    // DriveToRetractOfA up to the blocked round), without asserting who gives
    // way. Post: a at 9 tokens and b at 7, both quiescent, the pool full.
    void DriveToTheBlockedRound() {
        Submit(MakeRequestSpec("a", /*num_pages=*/3));
        Submit(MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101));
        PlanOnce();
        SendForwardDone("a", {42});
        SendForwardDone("b", {142});
        PlanOnce();
        SendForwardDone("a", {43});
        SendForwardDone("b", {143});
        PlanOnce();
        SendForwardDone("a", {44});
        SendForwardDone("b", {144});
        ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    }
};

// Six pool blocks: b's image fits, a's does not.
class SmallerCandidateFitsSuite : public ImageDoesNotFitSuite {
protected:
    std::int32_t SnapshotPoolBlocks() const override { return 6; }
};

// One pool block: no image fits.
class NoImageFitsSuite : public ImageDoesNotFitSuite {
protected:
    std::int32_t SnapshotPoolBlocks() const override { return 1; }
};

TEST_F(SmallerCandidateFitsSuite, ACandidateWhoseImageFitsIsRetractedInsteadOfAbortingTheNewest) {
    DriveToTheBlockedRound();
    // The victim policy would take a (8 releasable blocks against b's 6),
    // but a's image does not fit the six-block pool and b's does: b is
    // retracted, nobody is aborted, and a's blocked decode gets b's pages.
    const ExecutionPlan round = PlanOnce();
    EXPECT_TRUE(round.aborts.empty());
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    EXPECT_EQ(scheduler_->RequestTokenSize("b"), 7) << "b is the one suspended";
    const SnapshotStoreBatch* store = FindSnapshotStore(round);
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->request_ids, std::vector<std::string>{"b"});
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 0) << "b's six blocks fill the pool exactly";
    const ForwardBatch* granted = FindForwardBatch(round);
    ASSERT_NE(granted, nullptr);
    EXPECT_EQ(granted->request_ids, std::vector<std::string>{"a"}) << "the blocked decode runs on b's pages";
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
}

TEST_F(NoImageFitsSuite, WhenNoImageFitsTheNewestResidentIsAbortedAndTheGrantProceeds) {
    DriveToTheBlockedRound();
    // Neither image fits one pool block: the newest resident, b, is aborted
    // in this very round -- recorded on the plan for the runtime, its pages
    // granted to a's blocked decode -- and nothing is imaged.
    const ExecutionPlan round = PlanOnce();
    ASSERT_EQ(round.aborts.size(), 1u);
    EXPECT_EQ(round.aborts.front().request_id, "b");
    EXPECT_EQ(round.aborts.front().reason, AbortReason::kImageDoesNotFit);
    EXPECT_NE(round.aborts.front().detail.find("--retraction-snapshot-host-gb"), std::string::npos)
        << round.aborts.front().detail;
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(FindSnapshotStore(round), nullptr);
    EXPECT_TRUE(ExtractCacheOps(round).empty());
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 1);
    const ForwardBatch* granted = FindForwardBatch(round);
    ASSERT_NE(granted, nullptr);
    EXPECT_EQ(granted->request_ids, std::vector<std::string>{"a"}) << "the grant proceeds as after a retraction";
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    // A late client abort for the same id is harmless, and the next plan
    // reaps the finished request.
    SendAbortEvent("b");
    SendForwardDone("a", {45});
    PlanOnce();
    EXPECT_EQ(scheduler_->RequestTokenSize("b"), -1) << "b is gone once the plan reaped it";
    SendForwardDone("a", {46});
    SendFinish("a");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 14) << "the pool balances after the abort";
}

// With Host L2 the published pages leave the pool to the tail alone. A
// decode victim that completed a prefix page since its last admission has
// not hashed it yet; the retraction publishes it first and sends it to L2, so
// the fit probe must not count it against the pool.
class L2TailOnlyPoolSuite : public ImageDoesNotFitSuite {
protected:
    std::int32_t SnapshotPoolBlocks() const override { return 1; }

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = ImageDoesNotFitSuite::MakeConfig();
        cfg.disable_l2_cache = false;
        return cfg;
    }
};

TEST_F(L2TailOnlyPoolSuite, AnUnhashedCompletedPageRidesL2AndTheVictimIsRetractedNotAborted) {
    // The prefill publication streams the prompt pages to Host (pinned
    // stores); acknowledge them as they appear so the blocked round may
    // retract.
    Submit(MakeRequestSpec("a", /*num_pages=*/3));
    Submit(MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101));
    AckWriteBacks(PlanOnce());
    SendForwardDone("a", {42});
    SendForwardDone("b", {142});
    AckWriteBacks(PlanOnce());
    SendForwardDone("a", {43});
    SendForwardDone("b", {143});
    AckWriteBacks(PlanOnce());
    SendForwardDone("a", {44});
    SendForwardDone("b", {144});
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);

    // a = [1..6 42 43 44]: 8 computed tokens, four whole pages; page 3
    // ([43 44]) completed since a's last admission and is unhashed. All four
    // ride L2 once published, so the one-block pool holds a's (empty) tail:
    // a is imaged, not aborted.
    const ExecutionPlan round = PlanOnce();
    EXPECT_TRUE(round.aborts.empty()) << "nothing is aborted while an image fits";
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    EXPECT_EQ(scheduler_->RequestTokenSize("a"), 9) << "a, the ranked victim, is the one suspended";
    const SnapshotStoreBatch* store = FindSnapshotStore(round);
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->request_ids, std::vector<std::string>{"a"});
    EXPECT_TRUE(store->src_pages.at(0).empty()) << "no tail: the pool holds only the blob";
    const auto write_backs = ExtractCacheOpsOfKind<WriteBackBatch>(round);
    ASSERT_FALSE(write_backs.empty()) << "the unhashed page goes to Host with the L2 leg";
    const ForwardBatch* granted = FindForwardBatch(round);
    ASSERT_NE(granted, nullptr);
    EXPECT_EQ(granted->request_ids, std::vector<std::string>{"b"}) << "the blocked decode runs on a's pages";
}

// The probe plans the L2 leg as the retraction places it, so a published
// page the Host cannot take counts against the pool, and a candidate whose
// pool cannot hold those pages fails the probe -- before anything is acquired
// or evicted for it -- under write-through as under write-back. One usable
// Host block: neither a's eight published blocks nor b's six fit beside the
// one-block pool.
class L2LegFallbackSuite : public L2TailOnlyPoolSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = L2TailOnlyPoolSuite::MakeConfig();
        cfg.host_allocator.total_pages = 2;  // one usable Host block: the L2 leg cannot hold a prefix
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        return cfg;
    }
};

TEST_F(L2LegFallbackSuite, WhenNoL2LegFitsTheAbortNamesTheHostSize) {
    Submit(MakeRequestSpec("a", /*num_pages=*/3));
    Submit(MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101));
    AckWriteBacks(PlanOnce());
    SendForwardDone("a", {42});
    SendForwardDone("b", {142});
    AckWriteBacks(PlanOnce());
    SendForwardDone("a", {43});
    SendForwardDone("b", {143});
    AckWriteBacks(PlanOnce());
    SendForwardDone("a", {44});
    SendForwardDone("b", {144});
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    const std::int32_t host_cached = scheduler_->HostPoolCachedBlocks();
    ASSERT_EQ(host_cached, 1) << "the prompts' write-through filled the one Host block";

    // Neither image fits, so the newest resident, b, is aborted -- not a,
    // the victim the policy ranks first -- and the abort names the Host's
    // size; the Host is as it was.
    const ExecutionPlan round = PlanOnce();
    ASSERT_EQ(round.aborts.size(), 1u);
    EXPECT_EQ(round.aborts.front().request_id, "b");
    EXPECT_EQ(round.aborts.front().reason, AbortReason::kImageDoesNotFit);
    const std::string& detail = round.aborts.front().detail;
    EXPECT_NE(detail.find("--kvstore-size"), std::string::npos) << detail;
    EXPECT_NE(detail.find("num_host_pages=2"), std::string::npos) << detail;
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(FindSnapshotStore(round), nullptr);
    const ForwardBatch* granted = FindForwardBatch(round);
    ASSERT_NE(granted, nullptr);
    EXPECT_EQ(granted->request_ids, std::vector<std::string>{"a"});
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), host_cached);
    const HostCacheStats stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.host_evictions.lru.blocks, 0u) << "nothing was evicted for either probe";
    EXPECT_EQ(stats.retraction_stores.ops, 0u);
    EXPECT_EQ(stats.retraction_stores.snapshot_fallback, 0u);
    EXPECT_EQ(stats.retraction_stores.image_refused, 0u);
}

// Seven Host pages: the write-through fills six usable Host blocks with a's
// prompt pages, which a's own image would pin, so a's last completed page
// finds no Host block and its two blocks do not fit the one-block pool. b's
// six published blocks take the Host blocks a's prompt pages leave.
class L2LegOfTheSecondCandidateFitsSuite : public L2LegFallbackSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = L2LegFallbackSuite::MakeConfig();
        cfg.host_allocator.total_pages = 7;
        return cfg;
    }
};

TEST_F(L2LegOfTheSecondCandidateFitsSuite, TheVictimRuleSkipsACandidateWhoseL2LegTheHostCannotHold) {
    Submit(MakeRequestSpec("a", /*num_pages=*/3));
    Submit(MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101));
    AckWriteBacks(PlanOnce());
    SendForwardDone("a", {42});
    SendForwardDone("b", {142});
    AckWriteBacks(PlanOnce());
    SendForwardDone("a", {43});
    SendForwardDone("b", {143});
    AckWriteBacks(PlanOnce());
    SendForwardDone("a", {44});
    SendForwardDone("b", {144});
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    ASSERT_EQ(scheduler_->HostPoolFreeBlocks(), 0);

    // a ranks first but its L2 leg does not fit: b is retracted instead,
    // nobody is aborted, and a's blocked decode runs on b's pages.
    const ExecutionPlan round = PlanOnce();
    EXPECT_TRUE(round.aborts.empty());
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    const SnapshotStoreBatch* store = FindSnapshotStore(round);
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->request_ids, std::vector<std::string>{"b"});
    const ForwardBatch* granted = FindForwardBatch(round);
    ASSERT_NE(granted, nullptr);
    EXPECT_EQ(granted->request_ids, std::vector<std::string>{"a"});
    const HostCacheStats stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.retraction_stores.blocks, (std::vector<std::uint64_t>{3, 3})) << "b's three pages in both groups";
    EXPECT_EQ(stats.retraction_stores.snapshot_fallback, 0u);
    EXPECT_EQ(stats.retraction_stores.image_refused, 0u);
    // Write-through places the leg block by block: a's prompt pages, still
    // on the Device, leave the Host as lru, where segments would count them
    // as duplicates.
    EXPECT_EQ(stats.host_evictions.lru.blocks, 6u);
    EXPECT_EQ(stats.host_evictions.duplicate.blocks, 0u);
}

// The knob's victim takes the same probe, which plans the L2 leg: a victim
// whose published pages the Host cannot take beside a too small pool is
// refused before the retraction publishes anything, so it runs on.
class ForcedL2LegFallbackSuite : public L2LegFallbackSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = L2LegFallbackSuite::MakeConfig();
        cfg.debug_force_retraction_interval = 3;
        return cfg;
    }
};

TEST_F(ForcedL2LegFallbackSuite, AForcedVictimWhoseL2LegTheHostCannotTakeRunsOnWithNothingPublished) {
    Submit(MakeRequestSpec("a", /*num_pages=*/2));
    AckWriteBacks(PlanOnce());  // plan 1: prefill
    SendForwardDone("a", {42});
    AckWriteBacks(PlanOnce());  // plan 2: the first decode
    SendForwardDone("a", {43});

    // Plan 3 arms a = [1 2 3 4 42 43], quiescent: 5 computed tokens, the
    // partial page [42]. Of its six published blocks at most one gets a Host
    // block and the rest do not fit a pool of one block: the probe refuses
    // a, nothing is published, and a decodes on.
    const ExecutionPlan forced = PlanOnce();
    EXPECT_TRUE(forced.aborts.empty());
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(FindSnapshotStore(forced), nullptr);
    EXPECT_EQ(scheduler_->TailEntries(), 0u) << "the partial page is not published";
    ASSERT_EQ(FindForwardBatch(forced)->request_ids, std::vector<std::string>{"a"}) << "a keeps its pages and runs";
    EXPECT_EQ(scheduler_->GetHostCacheStats().host_evictions.lru.blocks, 0u) << "nothing was evicted for the probe";
    SendForwardDone("a", {44});

    // Plan 4 publishes the page [42 43] a completed on its own block.
    const ExecutionPlan next = PlanOnce();
    ASSERT_EQ(FindForwardBatch(next)->request_ids, std::vector<std::string>{"a"});
    AckWriteBacks(next);
    SendForwardDone("a", {45});
    SendFinish("a");
    AckWriteBacks(PlanOnce());
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 14);
}

// The null page alone: the engine never images, and says so.
class NoPoolSuite : public ImageDoesNotFitSuite {
protected:
    std::int32_t SnapshotPoolBlocks() const override { return 0; }
};

TEST_F(NoPoolSuite, WithoutAPoolACapacityBlockAbortsAndNamesThePoolKnobNotTheSlots) {
    ASSERT_FALSE(Config().HasSnapshotPool());
    ASSERT_EQ(Config().max_retracted_requests, 0);
    DriveToTheBlockedRound();
    const ExecutionPlan round = PlanOnce();
    ASSERT_EQ(round.aborts.size(), 1u);
    EXPECT_EQ(round.aborts.front().request_id, "b");
    EXPECT_EQ(round.aborts.front().reason, AbortReason::kImageDoesNotFit);
    const std::string& detail = round.aborts.front().detail;
    EXPECT_NE(detail.find("--retraction-snapshot-ratio"), std::string::npos) << detail;
    EXPECT_NE(detail.find("--retraction-snapshot-host-gb"), std::string::npos) << detail;
    EXPECT_EQ(detail.find("--retraction-snapshot-max-requests"), std::string::npos)
        << "slots are not the shortfall when no pool exists: " << detail;
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    const ForwardBatch* granted = FindForwardBatch(round);
    ASSERT_NE(granted, nullptr);
    EXPECT_EQ(granted->request_ids, std::vector<std::string>{"a"});
}

TEST_F(NoImageFitsSuite, TheDebugKnobLeavesAVictimItCannotImageRunningWithNothingPublished) {
    // Forced retraction is not capacity pressure: a victim it cannot image
    // keeps running. The knob probes the image before the retraction
    // publishes anything: that publication would register a's partial page
    // as a tail on the block a writes next, and the page's own publication
    // would later find that block under the tail's key.
    config_.debug_force_retraction_interval = 3;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    Submit(MakeRequestSpec("a", /*num_pages=*/2));
    PlanOnce();  // plan 1: prefill
    SendForwardDone("a", {42});
    PlanOnce();  // plan 2: the first decode
    SendForwardDone("a", {43});

    // Plan 3 arms a = [1 2 3 4 42 43], quiescent: 5 computed tokens, the
    // partial page [42]. Its 3 pages x 2 groups do not fit one pool block.
    const ExecutionPlan forced = PlanOnce();
    EXPECT_TRUE(forced.aborts.empty());
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(FindSnapshotStore(forced), nullptr);
    EXPECT_EQ(scheduler_->TailEntries(), 0u) << "the partial page is not published";
    ASSERT_EQ(FindForwardBatch(forced)->request_ids, std::vector<std::string>{"a"}) << "a keeps its pages and runs";
    SendForwardDone("a", {44});

    // Plan 4 publishes the page [42 43] a completed on its own block.
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"a"});
    SendForwardDone("a", {45});
    SendFinish("a");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 14);
}

// Head-of-line among readmissions: a 60K-token image that does not fit must
// not seal the queue while a 2K image behind it would. 1024-token pages, 64
// usable Device blocks, one full-attention group; the debug knob retracts
// the oldest quiescent decode every third plan.
class ReadmissionScanSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 1024;
        cfg.device_allocator.total_pages = 65;
        cfg.host_allocator.total_pages = 0;
        cfg.max_scheduled_tokens = 65536;  // every prompt is one chunk
        cfg.max_batch_size = 8;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull)};
        SetTestSnapshotPool(cfg);
        cfg.debug_force_retraction_interval = 3;
        return cfg;
    }
};

TEST_F(ReadmissionScanSuite, ASmallerLandedImageRestoresBehindALargeOneThatDoesNotFit) {
    // big: 60 pages (61440 tokens) -> 61 blocks with its decode slot; small:
    // 2 pages -> 3 blocks. Together exactly the pool.
    Submit(MakeRequestSpec("big", /*num_pages=*/60));
    Submit(MakeRequestSpec("small", /*num_pages=*/2, /*start=*/70001));
    const ExecutionPlan p1 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p1)->request_ids, (std::vector<std::string>{"big", "small"}));
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    SendForwardDone("big", {7});
    SendForwardDone("small", {8});
    const ExecutionPlan p2 = PlanOnce();  // first decodes
    ASSERT_EQ(FindForwardBatch(p2)->request_ids.size(), 2u);
    SendForwardDone("big", {7});
    SendForwardDone("small", {8});

    // Plan 3: the knob retracts big (the oldest decode); small decodes on.
    const ExecutionPlan p3 = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    ASSERT_NE(FindSnapshotStore(p3), nullptr);
    ASSERT_EQ(FindSnapshotStore(p3)->request_ids, std::vector<std::string>{"big"});
    ASSERT_EQ(FindForwardBatch(p3)->request_ids, std::vector<std::string>{"small"});
    SendForwardDone("small", {8});
    for (const ExecutionPlan& plan : {PlanOnce(), PlanOnce()}) {  // plans 4 and 5
        ASSERT_EQ(FindForwardBatch(plan)->request_ids, std::vector<std::string>{"small"});
        SendForwardDone("small", {8});
    }

    // Plan 6: the knob retracts small too; filler (58 pages -> 59 blocks) is
    // admitted on the freed pool while neither image has landed. Five blocks
    // stay free: room for small's 3-block image, not for big's 61.
    Submit(MakeRequestSpec("filler", /*num_pages=*/58, /*start=*/80001));
    const ExecutionPlan p6 = PlanOnce();
    ASSERT_EQ(scheduler_->RetractedSize(), 2u);
    ASSERT_NE(FindSnapshotStore(p6), nullptr);
    ASSERT_EQ(FindSnapshotStore(p6)->request_ids, std::vector<std::string>{"small"});
    ASSERT_EQ(FindForwardBatch(p6)->request_ids, std::vector<std::string>{"filler"});
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 5);
    AckImageStores(p3);
    AckImageStores(p6);
    SendForwardDone("filler", {9});

    // Plan 7: big ranks first (oldest epoch) and does not fit; the scan goes
    // on to small, which does. big's wait still seals new prompts: the
    // 1-page newcomer, which would fit the two remaining blocks, is not
    // admitted beside filler's first decode.
    Submit(MakeRequestSpec("newcomer", /*num_pages=*/1, /*start=*/90001));
    const ExecutionPlan p7 = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(p7);
    ASSERT_NE(restore, nullptr);
    EXPECT_EQ(restore->request_ids, std::vector<std::string>{"small"}) << "the image that fits restores";
    EXPECT_EQ(restore->src_pages.at(0).size(), 3u);
    EXPECT_EQ(FindForwardBatch(p7)->request_ids, std::vector<std::string>{"filler"})
        << "the newcomer is sealed out while big waits";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 2);
    EXPECT_EQ(scheduler_->WaitingSize(), 3u) << "big (Retracted), small (Restoring), newcomer (Submitted)";
    AckRestores(p7);
    EXPECT_EQ(scheduler_->RequestTokenSize("small"), 2048 + 5);
    EXPECT_EQ(scheduler_->DecodingSize(), 2u) << "small resumed beside filler";
}

// A mid-prefill victim resumes its next chunk: with max_scheduled_tokens = 4
// the 6-token prompt of "a" takes two chunks, and "a" is retracted between
// them.
class RetractMidPrefillSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        cfg.max_scheduled_tokens = 4;
        // 11 physical pages -> 10 usable: "holder" (2-page prompt) charges 6
        // and the first 4-token chunk of "half" (no declared budget, so no
        // headroom) 4 = the pool.
        cfg.device_allocator.total_pages = 11;
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(RetractMidPrefillSuite, MidPrefillVictimResumesItsNextChunk) {
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 10);
    // "holder" takes 6 blocks and decodes; "half" gets its first chunk (4
    // blocks) -- exactly the pool.
    Submit(MakeRequestSpec("holder", /*num_pages=*/2, /*start=*/201));
    ExecutionPlan h1 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(h1)->request_ids, std::vector<std::string>{"holder"});
    SendForwardDone("holder", {242});
    Submit(MakeRequestSpec("half", /*num_pages=*/3, /*start=*/301));
    ExecutionPlan c1 = PlanOnce();
    const ForwardBatch* chunk = FindForwardBatch(c1);
    ASSERT_NE(chunk, nullptr);
    const auto half_row = std::ranges::find(chunk->request_ids, "half");
    ASSERT_NE(half_row, chunk->request_ids.end());
    const auto index = static_cast<std::size_t>(std::distance(chunk->request_ids.begin(), half_row));
    ASSERT_EQ(chunk->input_lengths.at(index), 4) << "the first chunk";
    ASSERT_EQ(chunk->request_ids.size(), 1u) << "outside mixed mode a prefill round carries no decode";
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    SendForwardDone("half");

    // half's second chunk needs a page it cannot get, but nobody else does:
    // retracting it would serve no one, so it waits while holder spends its
    // decode reserve (two steps).
    for (const std::int32_t token : {243, 244}) {
        ExecutionPlan decode = PlanOnce();
        ASSERT_EQ(FindForwardBatch(decode)->request_ids, std::vector<std::string>{"holder"});
        EXPECT_EQ(scheduler_->RetractedSize(), 0u) << "a stalled prefill with nobody to serve is not retracted";
        SendForwardDone("holder", {token});
    }

    // holder's next decode needs a page too: the blocked round retracts the
    // incomplete prefill (tier 1) and grants the page to holder.
    ExecutionPlan retract_round = PlanOnce();
    ASSERT_EQ(FindForwardBatch(retract_round)->request_ids, std::vector<std::string>{"holder"});
    ASSERT_EQ(scheduler_->WaitingSize(), 1u) << "half is suspended";
    const SnapshotStoreBatch* store = FindSnapshotStore(retract_round);
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->src_pages.at(0).size(), 4u) << "only the 4 computed tokens (2 pages x 2 groups) are imaged";
    AckImageStores(retract_round);
    SendForwardDone("holder", {245});
    SendFinish("holder");

    // Restored, "half" runs its SECOND chunk: positions [4, 6), no recompute
    // of the first.
    ExecutionPlan readmit = PlanOnce();
    ASSERT_NE(FindRestore(readmit), nullptr);
    ASSERT_TRUE(FindForwardBatch(readmit)->request_ids.empty());
    AckRestores(readmit);
    ExecutionPlan next_chunk = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(next_chunk);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids, std::vector<std::string>{"half"});
    EXPECT_EQ(op->extend_prefix_lens.at(0), 4) << "the computed chunk is not redone";
    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->prefill_lengths.at(0), 6);
    SendForwardDone("half", {345});
    SendFinish("half");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 10);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 10);
}

class PrefillHeadOfLineSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        // 19 physical pages -> 18 usable. "holder" owns 6 parents and the
        // first chunk of "active" owns 8, leaving just enough for "queued".
        cfg.device_allocator.total_pages = 19;
        cfg.host_allocator.total_pages = 20;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 8;
        for (auto& group : cfg.cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(PrefillHeadOfLineSuite, RetractingAnIncompletePrefillPublishesOnlyComputedPagesAndResumesAfterThem) {
    // The victim is a prefill that has been through some chunks but not all.
    // Only those chunks may reach the prefix cache: publishing the whole
    // prompt would hand later requests pages that were never computed. And
    // the restore continues exactly after them.
    Submit(MakeRequestSpec("done", /*num_pages=*/2));
    ExecutionPlan p1 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p1)->request_ids, std::vector<std::string>{"done"});
    SendForwardDone("done", {42});

    Submit(MakeRequestSpec("half", /*num_pages=*/8, /*start=*/101));
    ExecutionPlan p2 = PlanOnce();
    const ForwardBatch* chunk = FindForwardBatch(p2);
    ASSERT_EQ(chunk->request_ids, std::vector<std::string>{"half"});
    const std::int32_t computed = chunk->extend_prefix_lens.at(0) + chunk->input_lengths.at(0);
    ASSERT_LT(computed, chunk->prefill_lengths.at(0)) << "the victim is mid-prefill";
    SendForwardDone("half");  // the chunk landed, so the victim is retractable

    // "waiting" cannot be admitted beside the stalled prefill (head of line),
    // so the incomplete prefill gives way for it; outside mixed mode the
    // grant itself waits for the next prefill round.
    Submit(MakeRequestSpec("waiting", /*num_pages=*/1, /*start=*/201));
    ExecutionPlan retract_round = PlanOnce();
    ASSERT_EQ(scheduler_->WaitingSize(), 2u) << "the incomplete prefill gave way; waiting is not admitted yet";
    ASSERT_EQ(FindForwardBatch(retract_round)->request_ids, std::vector<std::string>{"done"});
    const SnapshotStoreBatch* store = FindSnapshotStore(retract_round);
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->src_pages.at(0).size(), static_cast<std::size_t>(computed / PrefixGranularity() * 2))
        << "only the computed pages are imaged, two groups each";
    AckImageStores(retract_round);
    SendForwardDone("done", {43});

    // The restore comes first and takes the computed pages back; "waiting"
    // is admitted beside it. half's next chunk then does not fit while both
    // residents hold the pool -- and since nobody else waits for capacity,
    // nothing is retracted for it: it waits.
    ExecutionPlan readmit = PlanOnce();
    ASSERT_NE(FindRestore(readmit), nullptr);
    ASSERT_EQ(FindRestore(readmit)->request_ids, std::vector<std::string>{"half"});
    ASSERT_EQ(FindForwardBatch(readmit)->request_ids, std::vector<std::string>{"waiting"});
    AckRestores(readmit);
    SendForwardDone("waiting", {207});
    ExecutionPlan blocked = PlanOnce();
    EXPECT_EQ(scheduler_->RetractedSize(), 0u) << "a stalled prefill with nobody to serve is not re-retracted";
    for (const std::string& id : FindForwardBatch(blocked)->request_ids) {
        EXPECT_NE(id, "half");
        SendForwardDone(id, {7});
    }
    SendFinish("done");
    SendFinish("waiting");

    // With the pool free, half runs its NEXT chunk: positions from the
    // computed ones on, never redoing them.
    ExecutionPlan resumed = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(resumed);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids, std::vector<std::string>{"half"});
    EXPECT_EQ(op->extend_prefix_lens.at(0), computed) << "the next chunk starts after the computed ones";
    EXPECT_EQ(op->input_lengths.at(0), 8);
}

TEST(ExtendResultEvent, AwaitingResultAbsorbsEmptyIntermediateResults) {
    // Under the PP chunk pipeline, older intermediate chunk results (empty
    // by contract) can land AFTER the final chunk was scheduled, i.e. while
    // the request already sits in PrefillAwaitingResult. Only the final
    // chunk's result carries a token; an empty arrival must keep waiting,
    // or the handoff batch goes out before the bootstrap token is real.
    BlockPool pool(/*num_lcm_blocks=*/8, {1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0, fsm::CacheProgress{},
                                                      /*load_pairs=*/{},
                                                      /*awaits_result=*/true});
    ASSERT_TRUE(request.Is<fsm::PrefillAwaitingResult>());

    request.Apply(fsm::ExtendResultEvent{{}});  // an older intermediate chunk's empty result
    EXPECT_TRUE(request.Is<fsm::PrefillAwaitingResult>()) << "an empty result must not end the wait";

    request.Apply(fsm::ExtendResultEvent{{42}});  // the final chunk's token
    EXPECT_TRUE(request.Is<fsm::PrefillDone>());
    EXPECT_EQ(request.LastToken(), 42);
}

TEST(ExtendResultEvent, AwaitingResultCarriesForwardsInFlightIntoPrefillDone) {
    // The mirror image of the test above: the final chunk's token lands
    // FIRST, while an older intermediate chunk's forward is still out. The
    // transition to PrefillDone must keep that forward on the books -- it is
    // still writing KV into these pages -- so the late empty result can land
    // (ResultLanded fatally rejects a result nobody is waiting for) and the
    // retraction policy keeps treating the pages as busy.
    BlockPool pool(/*num_lcm_blocks=*/8, {1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/2,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0, fsm::CacheProgress{},
                                                      /*load_pairs=*/{},
                                                      /*awaits_result=*/true});
    request.TrackScheduledForward();
    ASSERT_TRUE(request.Is<fsm::Prefilling>());
    request.Apply(fsm::SchedulePrefillEvent{/*tokens_this_round=*/2, /*reserve_num_tokens_in_next_schedule_event=*/1,
                                            /*awaits_result=*/true});
    request.TrackScheduledForward();
    ASSERT_TRUE(request.Is<fsm::PrefillAwaitingResult>());
    ASSERT_EQ(request.ResultsInFlight(), 2);

    request.NoteResultLanded();
    request.Apply(fsm::ExtendResultEvent{{42}});  // the final chunk's token, ahead of chunk 1's result
    EXPECT_TRUE(request.Is<fsm::PrefillDone>());
    EXPECT_EQ(request.ResultsInFlight(), 1) << "chunk 1's forward is still out against these pages";

    request.NoteResultLanded();
    request.Apply(fsm::ExtendResultEvent{{}});  // chunk 1's empty result, late
    EXPECT_TRUE(request.Is<fsm::PrefillDone>());
    EXPECT_EQ(request.ResultsInFlight(), 0);
}

TEST(SnapshotRetractEvent, StampsResumePriorityAndKeepsTheResumePoint) {
    // The readmission order is derived off the Retracted states: a victim
    // with generated output a client is reading resumes ahead of one that
    // had produced nothing, whatever their retraction epochs say
    // (nextReadmission reads resumes_generation first, then the epoch). The
    // state also records exactly where the victim stopped.
    BlockPool pool(/*num_lcm_blocks=*/8, {1});
    BlockPool snapshot_pool(/*num_lcm_blocks=*/8, {1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    ReqPoolAllocator req_pool{4};
    SnapshotSlotAllocator slots{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/2,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0, fsm::CacheProgress{},
                                                      /*load_pairs=*/{}, /*awaits_result=*/false});
    ASSERT_TRUE(request.Is<fsm::Prefilling>());
    request.TrackScheduledForward();
    request.NoteResultLanded();
    request.Apply(fsm::ExtendResultEvent{{}});  // the first chunk landed

    RetractForTest(request, coordinator, slots, /*epoch=*/1);
    const auto* retracted = request.GetIf<fsm::Retracted>();
    ASSERT_NE(retracted, nullptr);
    EXPECT_FALSE(retracted->ResumesGeneration()) << "no output yet: it resumes behind decode-origin victims";
    EXPECT_TRUE(retracted->ImageLanded()) << "no store op is pending in this test";
    const auto* shape = std::get_if<fsm::ResumePrefilling>(&retracted->shape);
    ASSERT_NE(shape, nullptr) << "a mid-prefill victim resumes its next chunk";
    EXPECT_EQ(shape->window.begin, 0);
    EXPECT_EQ(shape->window.size, 2);
    EXPECT_EQ(request.PrefillSize(), 4) << "nothing is rebased";
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8) << "the Device pages are released";
    EXPECT_EQ(snapshot_pool.NumEmptyLcmBlocks(), 7) << "one computed page is imaged";
    EXPECT_EQ(slots.AvailableSlots(), 3) << "the blob slot is taken";
}

TEST(RetractionHeadroom, EscalatesPerRetractionAndStopsAtTheGenerationBudget) {
    // Every admission secures one safe-step window of decode headroom up
    // front; each retraction raises the bar -- the previous admission was
    // still too optimistic -- until it reaches the budget the request could
    // ever use.
    RequestSpec spec;
    spec.request_id = "r";
    spec.tokens = {1, 2, 3, 4};
    spec.max_new_tokens = 6000;
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};

    constexpr std::int32_t kSafeSteps = 4096;
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 4096) << "a fresh request secures one window";

    request.NoteRetracted();
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 6000)
        << "capped by the generation budget, so the escalation terminates";

    request.NoteRetracted();
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 6000) << "and stays there";
}

TEST(RetractionHeadroom, ReservesOnlyTheRemainingGenerationBudget) {
    // A restore re-reserves on top of the tokens already generated. A
    // readmission that still reserved the full declared budget would demand
    // prompt + generated + max_new -- more than the request can ever write,
    // and near the single-request limit more than the pool holds, leaving it
    // Retracted forever.
    BlockPool pool(/*num_lcm_blocks=*/16, {1});
    BlockPool snapshot_pool(/*num_lcm_blocks=*/16, {1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    ReqPoolAllocator req_pool{4};
    SnapshotSlotAllocator slots{4};

    RequestSpec spec{.request_id = "r", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    spec.max_new_tokens = 6000;
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0, fsm::CacheProgress{},
                                                      /*load_pairs=*/{}, /*awaits_result=*/false});
    request.Apply(fsm::ExtendResultEvent{{42}});
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1});
    request.Apply(fsm::ExtendResultEvent{std::vector<std::int32_t>(999, 7)});
    ASSERT_EQ(request.GeneratedTokens(), 1000);

    constexpr std::int32_t kSafeSteps = 4096;
    EXPECT_EQ(request.RemainingNewTokens(), 5000);
    EXPECT_EQ(request.RemainingNewTokensAtAdmission(), 6000) << "the budget the admission saw, before any decode";
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 4096) << "one window, not yet the remaining budget";
    EXPECT_FALSE(request.ReserveCoversGeneration(kSafeSteps)) << "one window did not cover the 6000 open then";
    request.NoteRetracted();
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 5000)
        << "capped by the REMAINING budget: the generated 1000 are already held";

    // Retract and restore: the prefill window is untouched, and the restore
    // is the admission that records the budget open at that point.
    RetractForTest(request, coordinator, slots, /*epoch=*/1);
    EXPECT_EQ(request.PrefillSize(), 4) << "a snapshot retraction never rebases";
    EXPECT_EQ(request.RemainingNewTokens(), 5000);
    EXPECT_EQ(request.RemainingNewTokensAtAdmission(), 6000) << "the suspended request still carries its admission";
    std::vector<BlockTable> restored(coordinator.NumGroups());
    std::vector<GroupDemand> demands{GroupDemand{.table = &restored[0], .extent = DenseGrowth{0}, .reserve_tokens = 1}};
    ASSERT_TRUE(coordinator.Restore(request.GetIf<fsm::Retracted>()->image, demands, /*request_access_epoch=*/1));
    request.Apply(fsm::ScheduleRestoreEvent{&coordinator, std::move(restored), /*restore_op=*/7});
    ASSERT_TRUE(request.Is<fsm::Restoring>());
    EXPECT_EQ(request.RemainingNewTokensAtAdmission(), 5000) << "what the readmission saw as open";
    EXPECT_TRUE(request.ReserveCoversGeneration(kSafeSteps)) << "two windows cover it: the readmission is exempt";
    request.Apply(fsm::RestoreDoneEvent{req_pool.Allocate()});
    EXPECT_TRUE(request.Is<fsm::Decoding>());
    EXPECT_EQ(request.TokenSize(), 1004) << "the same tokens, resumed where they stopped";
}

TEST(RetractionHeadroom, SpendingTheWindowNeverMakesItCoverTheRemainder) {
    // A 6000-token budget admitted behind one 4096 window is not covered, and
    // must stay uncovered while decode spends that window: the remainder
    // shrinks in step with the headroom, so holding the window against the
    // CURRENT remainder would call the request covered the moment the
    // remainder dipped under 4096 -- exactly when the spent window forces it
    // to ask for a new page. With every resident request misjudged that way
    // retraction has no victim and the pool never frees.
    BlockPool pool(/*num_lcm_blocks=*/16, {1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    spec.max_new_tokens = 6000;
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0, fsm::CacheProgress{},
                                                      /*load_pairs=*/{}, /*awaits_result=*/false});
    request.Apply(fsm::ExtendResultEvent{{42}});
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1});
    request.Apply(fsm::ExtendResultEvent{std::vector<std::int32_t>(4096, 7)});
    ASSERT_EQ(request.GeneratedTokens(), 4097);

    constexpr std::int32_t kSafeSteps = 4096;
    EXPECT_EQ(request.RemainingNewTokens(), 1903) << "the remainder has dipped under one window";
    EXPECT_EQ(request.RemainingNewTokensAtAdmission(), 6000) << "but the admission's budget has not moved";
    EXPECT_FALSE(request.ReserveCoversGeneration(kSafeSteps)) << "the window it holds is spent, not covering";
}

TEST(RetractionHeadroom, AnUndeclaredGenerationBudgetDemandsNone) {
    // max_new_tokens == 0 is the opt-out: with no declared budget there is
    // nothing to prepay, and no cap either -- an escalation without a cap
    // could demand more than the pool holds and make the request
    // unreadmittable, which is worse than admitting it optimistically. So it
    // stays optimistic and relies on the victim policy for progress.
    RequestSpec spec;
    spec.request_id = "r";
    spec.tokens = {1, 2, 3, 4};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};

    constexpr std::int32_t kSafeSteps = 4096;
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 0);
    EXPECT_FALSE(request.ReserveCoversGeneration(kSafeSteps)) << "an undeclared budget never qualifies for exemption";
    request.NoteRetracted();
    EXPECT_EQ(request.AdmissionHeadroom(kSafeSteps), 0) << "even after a retraction";
}

TEST_F(RetractSuite, AFreshAdmissionPrepaysItsGenerationBudget) {
    // Without declared budgets "a" (6 tokens) and "b" (4 tokens) pack into
    // one round: 2*ceil(7/2) + 2*ceil(5/2) = 14 = the pool. A 5-token budget
    // on "a" is prepaid at admission -- 2*ceil((6+5)/2) = 12 -- so "b"
    // (needing 6) no longer fits beside it and waits.
    RequestSpec a = MakeRequestSpec("a", /*num_pages=*/3);
    a.max_new_tokens = 5;
    Submit(a);
    Submit(MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101));

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->request_ids, std::vector<std::string>{"a"});
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 2);
}

class AdmissionHeadroomPrefillRoleSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        cfg.role = Role::kP;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.transfer_policy = CacheTransferPolicy::FullSuffix;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{request_id});
        scheduler_->Advance(std::move(event));
    }
};

TEST_F(AdmissionHeadroomPrefillRoleSuite, ThePrefillRoleDoesNotPrepayDecodeHeadroom) {
    // The P role never retracts, so a declared generation budget -- even one
    // far beyond this pool -- charges nothing at admission. What the
    // completing chunk does hold is the same decode slot every role reserves
    // (the drafter writes its first candidate block there): the prompt plus
    // one token, 2*ceil(7/2) = 8 blocks, exactly the fused charge for "a".
    RequestSpec heavy = MakeRequestSpec("heavy", /*num_pages=*/3);
    heavy.max_new_tokens = 6000;
    Submit(heavy);
    SendBootstrapped("heavy");

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->request_ids, std::vector<std::string>{"heavy"});
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 6);
}

TEST_F(PrefillHeadOfLineSuite, AnIncompletePrefillGivesWayBeforeACompletedOne) {
    // Two retraction candidates: "done" finished its prefill and owns a
    // sampled token a client is waiting on; "half" has produced nothing. The
    // one with no committed output gives way, even though it is larger.
    Submit(MakeRequestSpec("done", /*num_pages=*/2));
    ExecutionPlan p1 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p1)->request_ids, std::vector<std::string>{"done"});
    SendForwardDone("done", {42});

    Submit(MakeRequestSpec("half", /*num_pages=*/8, /*start=*/101));
    ExecutionPlan p2 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p2)->request_ids, std::vector<std::string>{"half"});
    SendForwardDone("half");  // its chunk landed; the pages are no longer in use

    // "half" cannot get its next chunk, so capacity must be freed.
    Submit(MakeRequestSpec("waiting", /*num_pages=*/1, /*start=*/201));
    PlanOnce();

    EXPECT_EQ(scheduler_->DecodingSize(), 1u) << "the completed request kept its pages and took its first decode step";
    EXPECT_EQ(scheduler_->WaitingSize(), 2u) << "the incomplete prefill was the victim";
}

TEST_F(PrefillHeadOfLineSuite, AnIncompletePrefillIsNotRetractedWhileItsChunkIsInFlight) {
    // Regression pin: an incomplete prefill is the preferred victim, but a
    // chunk forward writes KV into its pages. Retract it before that write
    // lands and the result hits pages another request now owns -- which the
    // attention backend sees as a batch whose metadata does not match.
    Submit(MakeRequestSpec("done", /*num_pages=*/2));
    ExecutionPlan p1 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p1)->request_ids, std::vector<std::string>{"done"});
    SendForwardDone("done", {42});

    Submit(MakeRequestSpec("half", /*num_pages=*/8, /*start=*/101));
    ExecutionPlan p2 = PlanOnce();
    ASSERT_EQ(FindForwardBatch(p2)->request_ids, std::vector<std::string>{"half"});
    // Deliberately do NOT report the chunk: it is still on the GPU.

    Submit(MakeRequestSpec("waiting", /*num_pages=*/1, /*start=*/201));
    const std::int32_t waiting_before = static_cast<std::int32_t>(scheduler_->WaitingSize());
    PlanOnce();
    EXPECT_EQ(static_cast<std::int32_t>(scheduler_->WaitingSize()), waiting_before)
        << "nothing may be retracted while a forward is out against its pages";

    // Once the chunk lands, the same round retracts as before.
    SendForwardDone("half");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 1u) << "the completed request kept its pages";
    EXPECT_EQ(scheduler_->WaitingSize(), 2u) << "the incomplete prefill was the victim";
}

TEST_F(PrefillHeadOfLineSuite, BlockedLaterChunkDoesNotStartSubmittedRequest) {
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 18);

    Submit(MakeRequestSpec("holder", /*num_pages=*/2));
    ExecutionPlan holder_prefill = PlanOnce();
    ASSERT_EQ(FindForwardBatch(holder_prefill)->request_ids, std::vector<std::string>{"holder"});
    SendForwardDone("holder", {42});

    Submit(MakeRequestSpec("active", /*num_pages=*/8, /*start=*/101));
    ExecutionPlan first_chunk = PlanOnce();
    ASSERT_EQ(FindForwardBatch(first_chunk)->request_ids, std::vector<std::string>{"active"});
    ASSERT_EQ(FindForwardBatch(first_chunk)->input_lengths.at(0), 8);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 4);
    SendForwardDone("active");  // the chunk landed; its pages are retractable

    Submit(MakeRequestSpec("queued", /*num_pages=*/1, /*start=*/201));
    ExecutionPlan blocked = PlanOnce();
    // The stalled resident prefill seals new-prompt admission ("queued"
    // must not strand it further), but the completed "holder" still takes
    // its decode step: decodes are never hostage to a stalled prefill.
    ASSERT_EQ(FindForwardBatch(blocked)->request_ids, std::vector<std::string>{"holder"})
        << "a blocked active prefill must prevent lower-priority admission";
    // The stalled prefill is the victim, not the completed "holder": it has
    // produced no output a client is reading, so it gives way and retries
    // once the capacity is there.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 12) << "the incomplete prefill released its pages";
    EXPECT_EQ(scheduler_->WaitingSize(), 2u) << "the retracted prefill and the untouched submitted request wait";
    EXPECT_EQ(scheduler_->DecodingSize(), 1u) << "the completed request keeps its pages";
}

// Two prompts mid-prefill hold every Device page while a retracted request
// waits to restore: 13 physical pages -> 12 usable, one full-attention group
// of 2-token pages, 8-token chunks, and the knob retracting the oldest decode
// every sixth plan, under each Host write policy.
class PrefillsHoldingTheDeviceSuite : public SchedulerTestSuite,
                                      public ::testing::WithParamInterface<SchedulerConfig::HostWritePolicy> {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 13;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        cfg.host_write_policy = GetParam();
        cfg.max_scheduled_tokens = 8;
        cfg.debug_force_retraction_interval = 6;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

INSTANTIATE_TEST_SUITE_P(HostWritePolicies, PrefillsHoldingTheDeviceSuite,
                         ::testing::Values(SchedulerConfig::HostWritePolicy::kWriteThrough,
                                           SchedulerConfig::HostWritePolicy::kWriteBack));

// h, the head of line, blocks on its own next chunk and, as the largest
// incomplete prefill, is the victim. Its admission is the only one that
// failed: the head-of-line rule kept x from trying, and r's waiting restore
// seals new prompts. No decode, no chunk and no copy is left to run, so a
// completion never comes: h gives way to x.
TEST_P(PrefillsHoldingTheDeviceSuite, ABlockedPrefillGivesWayToThePrefillItHoldsBack) {
    // r decodes on 4 pages: its 3-page prompt and the decode page. Every
    // Host store is acknowledged in the plan after its own: a pinned one in
    // flight would hold off any retraction.
    Submit(MakeRequestSpec("r", /*num_pages=*/3));
    AckWriteBacks(PlanOnce());
    SendForwardDone("r", {7});
    AckWriteBacks(PlanOnce());
    SendForwardDone("r", {7});

    // h, a 10-page prompt, runs two chunks beside r; x, 9 pages, arrives.
    Submit(MakeRequestSpec("h", /*num_pages=*/10, /*start=*/1001));
    const ExecutionPlan h_first = PlanOnce();
    ASSERT_EQ(FindForwardBatch(h_first)->request_ids, std::vector<std::string>{"h"});
    SendForwardDone("h");
    AckWriteBacks(h_first);
    Submit(MakeRequestSpec("x", /*num_pages=*/9, /*start=*/2001));
    const ExecutionPlan h_second = PlanOnce();
    ASSERT_EQ(FindForwardBatch(h_second)->request_ids, std::vector<std::string>{"h"});
    SendForwardDone("h");
    AckWriteBacks(h_second);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);

    // h's last chunk does not fit beside r: h gives way to x, whose grant
    // waits for the next prefill round (r's decode runs alone outside mixed
    // mode).
    const ExecutionPlan h_retracted = PlanOnce();
    ASSERT_NE(FindSnapshotStore(h_retracted), nullptr);
    ASSERT_EQ(FindSnapshotStore(h_retracted)->request_ids, std::vector<std::string>{"h"});
    ASSERT_EQ(FindForwardBatch(h_retracted)->request_ids, std::vector<std::string>{"r"});
    SendForwardDone("r", {7});
    AckImageStores(h_retracted);

    // The knob retracts r; h restores onto its pages still cached; x runs
    // its first chunk.
    const ExecutionPlan h_restored = PlanOnce();
    ASSERT_NE(FindSnapshotStore(h_restored), nullptr);
    ASSERT_EQ(FindSnapshotStore(h_restored)->request_ids, std::vector<std::string>{"r"});
    ASSERT_NE(FindRestore(h_restored), nullptr);
    ASSERT_EQ(FindRestore(h_restored)->request_ids, std::vector<std::string>{"h"});
    ASSERT_EQ(FindForwardBatch(h_restored)->request_ids, std::vector<std::string>{"x"});
    SendForwardDone("x");
    AckImageStores(h_restored);
    AckRestores(h_restored);
    ASSERT_EQ(scheduler_->PrefillSize(), 2u) << "h and x mid-prefill";
    ASSERT_EQ(scheduler_->RetractedSize(), 1u) << "r waits to restore";
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);

    const ExecutionPlan gave_way = PlanOnce();
    ASSERT_NE(FindSnapshotStore(gave_way), nullptr);
    EXPECT_EQ(FindSnapshotStore(gave_way)->request_ids, std::vector<std::string>{"h"});
    ASSERT_NE(FindForwardBatch(gave_way), nullptr);
    ASSERT_EQ(FindForwardBatch(gave_way)->request_ids, std::vector<std::string>{"x"});
    SendForwardDone("x");
    AckImageStores(gave_way);

    // From here every request finishes at the token its prompt ends with (r
    // at its next decode), each copy acknowledged in the plan after its own.
    for (std::int32_t round = 0;
         round < 32 && (scheduler_->RequestTokenSize("h") != -1 || scheduler_->RequestTokenSize("x") != -1 ||
                        scheduler_->RequestTokenSize("r") != -1);
         ++round) {
        const ExecutionPlan plan = PlanOnce();
        if (const ForwardBatch* batch = FindForwardBatch(plan)) {
            for (std::size_t row = 0; row < batch->request_ids.size(); ++row) {
                const std::string& id = batch->request_ids[row];
                if (row < batch->NumExtends() &&
                    batch->extend_prefix_lens[row] + batch->input_lengths[row] < batch->prefill_lengths[row]) {
                    SendForwardDone(id);
                    continue;
                }
                SendForwardDone(id, {7});
                SendFinish(id);
            }
        }
        AckImageStores(plan);
        AckRestores(plan);
    }
    EXPECT_EQ(scheduler_->RequestTokenSize("h"), -1);
    EXPECT_EQ(scheduler_->RequestTokenSize("x"), -1);
    EXPECT_EQ(scheduler_->RequestTokenSize("r"), -1);
}

// Four decodes with declared budgets fill an 8-token budget at 2 tokens a
// step beside d1, which declares none and so is the only request the victim
// policy may take, and p, a 16-token prompt mid-prefill: 41 physical pages
// -> 40 usable, one full-attention group of 2-token pages, mixed batches,
// under each Host write policy.
class DecodesFillingTheBatchSuite : public SchedulerTestSuite,
                                    public ::testing::WithParamInterface<SchedulerConfig::HostWritePolicy> {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 41;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        cfg.host_write_policy = GetParam();
        cfg.max_scheduled_tokens = 8;
        cfg.decode_input_tokens = 2;
        cfg.enable_mixed_prefill_decode = true;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

INSTANTIATE_TEST_SUITE_P(HostWritePolicies, DecodesFillingTheBatchSuite,
                         ::testing::Values(SchedulerConfig::HostWritePolicy::kWriteThrough,
                                           SchedulerConfig::HostWritePolicy::kWriteBack));

// d1 blocks on its next page while e1-e4 fill the batch, so the prefill
// phases never run: the full batch, not the head-of-line rule, leaves p out,
// and p could not run this round anyway. Nobody else's admission failed and
// no prompt waits, so d1 waits for a completion instead of giving way to p.
TEST_P(DecodesFillingTheBatchSuite, ABlockedDecodeDoesNotGiveWayToAPrefillTheFullBatchLeftOut) {
    const std::map<std::string, std::int32_t> limits = {{"d1", 8},  {"e1", 12}, {"e2", 12},
                                                        {"e3", 12}, {"e4", 12}, {"p", 2}};
    Submit(RequestSpec{.request_id = "d1", .tokens = MakeTokens(2)});
    for (std::int32_t i = 1; i <= 4; ++i) {
        const std::string id = "e" + std::to_string(i);
        Submit(
            RequestSpec{.request_id = id, .tokens = MakeTokens(1, 1001 + 1000 * i), .max_new_tokens = limits.at(id)});
    }
    Submit(RequestSpec{.request_id = "p", .tokens = MakeTokens(16, 1001), .max_new_tokens = limits.at("p")});

    // Lands every row of the plan; a request finishes at its limit (d1 at
    // one it never declared).
    std::map<std::string, std::int32_t> generated;
    const auto land = [&](const ExecutionPlan& plan) {
        if (const ForwardBatch* batch = FindForwardBatch(plan)) {
            for (std::size_t row = 0; row < batch->request_ids.size(); ++row) {
                const std::string& id = batch->request_ids[row];
                if (row < batch->NumExtends() &&
                    batch->extend_prefix_lens[row] + batch->input_lengths[row] < batch->prefill_lengths[row]) {
                    SendForwardDone(id);
                    continue;
                }
                SendForwardDone(id, {7});
                if (++generated[id] == limits.at(id)) {
                    SendFinish(id);
                }
            }
        }
        AckImageStores(plan);
        AckRestores(plan);
    };

    // Every prompt prefills, p by its first chunk; then d1 and e1-e3 decode
    // three times, e4 and p left out of the full batch.
    for (std::int32_t round = 0; round < 4; ++round) {
        const ExecutionPlan plan = PlanOnce();
        ASSERT_EQ(FindSnapshotStore(plan), nullptr);
        land(plan);
    }

    const ExecutionPlan blocked = PlanOnce();
    ASSERT_EQ(FindForwardBatch(blocked)->request_ids, (std::vector<std::string>{"e1", "e2", "e3", "e4"}))
        << "d1 blocks on its next page";
    ASSERT_EQ(scheduler_->PrefillSize(), 1u) << "p mid-prefill";
    EXPECT_EQ(FindSnapshotStore(blocked), nullptr) << "d1 gave way to p, which the full batch keeps out of the round";
    EXPECT_TRUE(blocked.aborts.empty());
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    land(blocked);

    // Then every request runs to its limit.
    for (std::int32_t round = 0;
         round < 32 &&
         std::ranges::any_of(limits,
                             [this](const auto& limit) { return scheduler_->RequestTokenSize(limit.first) != -1; });
         ++round) {
        land(PlanOnce());
    }
    for (const auto& [id, limit] : limits) {
        EXPECT_EQ(scheduler_->RequestTokenSize(id), -1) << id;
    }
}

// Exact-fit re-admission after a retract: the whole freed budget (pages AND any
// stale decode reserve) must be spendable by the next request.
class RetractExactFitSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        // 9 physical pages -> 8 usable: one 3-page prompt charges exactly the pool.
        cfg.device_allocator.total_pages = 9;
        cfg.host_allocator.total_pages = 10;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(RetractExactFitSuite, ReportsSingleRequestTokenCapacity) {
    // Two full-history groups share eight usable parents. Each group needs
    // ceil(tokens / 2) parents, so one request can address eight tokens.
    EXPECT_EQ(scheduler_->MaxSingleRequestTokens(), 8);
}

TEST_F(RetractExactFitSuite, ReportsCapacityUsingEachGroupsBlockGranularity) {
    SchedulerConfig config = MakeConfig();
    config.cache_groups[1].block_granularity = 1;
    SetTestSnapshotPool(config);
    Scheduler scheduler{WithHostPackingFromDevice(std::move(config))};

    // Eight parents fit ceil(tokens / 2) pages for the first group and one
    // page per token for the second group. Five tokens use 3 + 5 parents.
    EXPECT_EQ(scheduler.MaxSingleRequestTokens(), 5);
}

TEST_F(RetractExactFitSuite, IncludesOverlapDecodeReserveInTokenCapacity) {
    SchedulerConfig config = MakeConfig();
    config.overlap_schedule_depth = 1;
    SetTestSnapshotPool(config);
    Scheduler scheduler{WithHostPackingFromDevice(std::move(config))};
    // The extra decode token shares the fourth page with token seven. Counting
    // it as a separately rounded page would incorrectly report only six.
    EXPECT_EQ(scheduler.MaxSingleRequestTokens(), 7);
}

TEST(PdSlidingCapacityTest, CountsPrefixIslandPhasePageAndGroupPacking) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = 4;
    cfg.device_allocator.total_pages = 3;  // null + two usable LCM parents
    cfg.host_allocator.total_pages = 0;
    cfg.max_scheduled_tokens = 16;
    cfg.max_batch_size = 1;
    cfg.decode_input_tokens = 1;
    cfg.role = Role::kD;
    cfg.disable_l2_cache = true;

    CacheGroupConfig sliding;
    sliding.group_id = "sliding";
    sliding.block_granularity = 2;  // group q=2 while scheduler P=4
    sliding.total_pages = 5;        // null + two parents packing two children each
    sliding.cache_blocks_per_lcm_block = 2;
    sliding.attn_kind = AttnKind::kSlidingWindow;
    sliding.sliding_window_tokens = 4;
    sliding.transfer_policy = CacheTransferPolicy::FullSuffix;
    cfg.cache_groups = {sliding};

    SetTestSnapshotPool(cfg);
    Scheduler scheduler{WithHostPackingFromDevice(std::move(cfg))};

    // W=4, q=2 and a one-token decode reserve can require two cached
    // lookback pages plus a three-page phase-shifted remote tail, capped by
    // a dense table; one more child page is pinned while the hit's full tail
    // block is copied. Six is the exact maximum with four child pages.
    EXPECT_EQ(scheduler.MaxSingleRequestTokens(), 6);
}

TEST_F(RetractExactFitSuite, ReserveRefundBalances) {
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 8);
    Submit(MakeRequestSpec("a", /*num_pages=*/3));  // charge 2*ceil(7/2) = 8: exact fit
    ExecutionPlan prefill = PlanOnce();
    ASSERT_EQ(FindForwardBatch(prefill)->request_ids.size(), 1u);
    SendForwardDone("a", {42});
    PlanOnce();  // decode transition consumes the reserve: free 0
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    SendForwardDone("a", {43});  // 8 tokens = capacity
    PlanOnce();                  // tail-page decode (0 fresh blocks)
    SendForwardDone("a", {44});  // 9 tokens: past capacity

    // "d" needs EXACTLY the released capacity: a leaked reservation from a
    // would make this admission fail. The blocked round retracts a for d and
    // grants d its pages in the same round.
    Submit(MakeRequestSpec("d", /*num_pages=*/3, /*start=*/201));
    ExecutionPlan retract_round = PlanOnce();
    ASSERT_EQ(scheduler_->WaitingSize(), 1u) << "a is suspended";
    const ForwardBatch* op = FindForwardBatch(retract_round);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids, std::vector<std::string>{"d"})
        << "exact-fit admission proves the full budget was refunded";
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);

    // Once a's image lands, its restore does not fit beside d and WAITS: d
    // keeps decoding and is never retracted for a readmission.
    AckImageStores(retract_round);
    SendForwardDone("d", {99});
    ExecutionPlan d_decodes = PlanOnce();
    ASSERT_EQ(FindForwardBatch(d_decodes)->request_ids, std::vector<std::string>{"d"});
    EXPECT_EQ(FindRestore(d_decodes), nullptr);
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    SendForwardDone("d", {100});
    SendFinish("d");
    SendAbort(*scheduler_, "a");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 8);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 8);
}

// Two capacity-block cycles on one pool: each cycle retracts a different
// request, and a suspended request's restore waits for capacity rather than
// retracting a resident for it.
class RetractTrioSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        // 25 physical pages -> 24 usable: r1 charges 10, r2 8, r3 6 = the pool.
        cfg.device_allocator.total_pages = 25;
        cfg.host_allocator.total_pages = 26;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(RetractTrioSuite, TwoCapacityBlocksRetractDifferentRequestsAndTheRestoreWaits) {
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 24);
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    Submit(MakeRequestSpec("r3", /*num_pages=*/2, /*start=*/201));

    ExecutionPlan prefill = PlanOnce();
    ASSERT_EQ(FindForwardBatch(prefill)->request_ids.size(), 3u);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    SendForwardDone("r3", {242});

    ExecutionPlan decode = PlanOnce();  // all three consume their reserves
    ASSERT_EQ(FindForwardBatch(decode)->request_ids.size(), 3u);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    SendForwardDone("r1", {43});   // 10 = capacity
    SendForwardDone("r2", {143});  // 8 = capacity
    SendForwardDone("r3", {243});  // 6 = capacity
    PlanOnce();                    // tail-page decodes (0 fresh blocks)
    SendForwardDone("r1", {44});   // 11: past capacity
    SendForwardDone("r2", {144});  // 9: past capacity
    SendForwardDone("r3", {244});  // 7: past capacity

    // Cycle 1 retracts r1 -- all three tie at two generated tokens and r1
    // frees the most (5 pages x 2 groups) -- and, r1 being the blocker
    // itself, grants the freed pages to the next blocked decode, r2, in the
    // same round.
    ExecutionPlan first_retract = PlanOnce();
    EXPECT_EQ(FindForwardBatch(first_retract)->request_ids, std::vector<std::string>{"r2"});
    ASSERT_EQ(scheduler_->WaitingSize(), 1u);
    ASSERT_EQ(scheduler_->RequestTokenSize("r1"), 11);
    AckImageStores(first_retract);
    SendForwardDone("r2", {145});

    // r2 and r3 ride the freed pages until capacity blocks again. r1's
    // restore (10 imaged blocks + 2 for its decode slot) never fits
    // meanwhile and waits without retracting anyone, until a second victim
    // is taken for the residents' own growth.
    std::int32_t next_r2 = 146;
    std::int32_t next_r3 = 245;
    bool second_retraction = false;
    for (int round = 0; round < 32 && !second_retraction; ++round) {
        ExecutionPlan plan = PlanOnce();
        EXPECT_EQ(FindRestore(plan), nullptr) << "r1's restore must wait for capacity, round " << round;
        AckImageStores(plan);
        second_retraction = scheduler_->WaitingSize() == 2u;
        for (const std::string& id : FindForwardBatch(plan)->request_ids) {
            SendForwardDone(id, {id == "r2" ? next_r2++ : next_r3++});
        }
    }
    ASSERT_TRUE(second_retraction) << "a second capacity block retracts a second request";
    // The second victim is a DIFFERENT request -- the cycles must not keep
    // picking the same one -- and the first is still suspended.
    EXPECT_EQ(scheduler_->RetractedSize(), 2u);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);

    // The second victim is r2: it holds more pages than r3 and frees the
    // most. Its pages serve r1's restore -- the older victim first -- beside
    // r3's decode in the next round; r2's own restore waits for r3 to finish.
    const std::int32_t r1_tokens_before = scheduler_->RequestTokenSize("r1");
    const std::int32_t r2_tokens_before = scheduler_->RequestTokenSize("r2");
    ExecutionPlan restore_r1 = PlanOnce();
    ASSERT_NE(FindRestore(restore_r1), nullptr);
    EXPECT_EQ(FindRestore(restore_r1)->request_ids, std::vector<std::string>{"r1"}) << "the older victim first";
    EXPECT_EQ(FindForwardBatch(restore_r1)->request_ids, std::vector<std::string>{"r3"}) << "r3 is the survivor";
    AckRestores(restore_r1);
    SendForwardDone("r3", {next_r3++});
    ExecutionPlan waiting = PlanOnce();
    EXPECT_EQ(FindRestore(waiting), nullptr) << "r2's restore does not fit beside r1 and r3";
    for (const std::string& id : FindForwardBatch(waiting)->request_ids) {
        SendForwardDone(id, {7});
    }
    SendFinish("r3");
    EXPECT_EQ(scheduler_->RequestTokenSize("r1"), r1_tokens_before + 1) << "r1 resumed decoding where it stopped";
    // r2's 16-block restore (7 imaged pages + a decode slot, two groups) does
    // not fit beside r1 either: it keeps waiting until r1 is gone.
    ExecutionPlan still_waiting = PlanOnce();
    EXPECT_EQ(FindRestore(still_waiting), nullptr);
    EXPECT_EQ(FindForwardBatch(still_waiting)->request_ids, std::vector<std::string>{"r1"});
    SendForwardDone("r1", {45});
    SendAbort(*scheduler_, "r1");
    ExecutionPlan restore_r2 = PlanOnce();
    ASSERT_NE(FindRestore(restore_r2), nullptr);
    EXPECT_EQ(FindRestore(restore_r2)->request_ids, std::vector<std::string>{"r2"});
    AckRestores(restore_r2);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->RequestTokenSize("r2"), r2_tokens_before) << "resumed with the same tokens";
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 24);

    SendAbort(*scheduler_, "r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 24);
}

// A retracted request whose config carries a mamba-style state group
// (family=State, FullHistory retention) must release state pages too.
class RetractStateGroupSuite : public RetractSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = RetractSuite::MakeConfig();
        cfg.device_allocator.total_pages = 9;  // 8 usable
        cfg.host_allocator.total_pages = 10;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("state", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kState),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(RetractStateGroupSuite, StateGroupRequestRetractsCleanly) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    Submit(MakeRequestSpec("a", /*num_pages=*/2));
    ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* prefill_op = FindForwardBatch(prefill);
    ASSERT_NE(prefill_op, nullptr);
    ASSERT_EQ(prefill_op->request_ids.size(), 1u) << "the prompt must admit into the state-group config";
    ASSERT_EQ(prefill_op->block_tables.count("state"), 1u);
    SendForwardDone("a", {1000});

    // A prompt that does not fit beside a waits for it. a keeps decoding
    // (a request with a forward out is never retracted) until its own next
    // page is refused as well; that round retracts it for b -- every one of
    // its pages, full-history AND state, returns to the pool and its image
    // goes to the snapshot pool -- and b is admitted on them.
    Submit(MakeRequestSpec("b", /*num_pages=*/4, /*start=*/101));
    std::int32_t tok = 1001;
    ExecutionPlan retract_round;
    bool retracted = false;
    for (int round = 0; round < 64 && !retracted; ++round) {
        retract_round = PlanOnce();
        retracted = scheduler_->RetractedSize() == 1u;
        if (!retracted) {
            ASSERT_EQ(FindForwardBatch(retract_round)->request_ids, std::vector<std::string>{"a"}) << "round " << round;
            SendForwardDone("a", {tok++});
        }
    }
    ASSERT_TRUE(retracted) << "the grower must exhaust capacity and give way";
    ASSERT_EQ(FindForwardBatch(retract_round)->request_ids, std::vector<std::string>{"b"}) << "b admits on a's pages";
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_LT(scheduler_->SnapshotPoolFreeBlocks(), free_at_start) << "history pages and the live state are imaged";
    const SnapshotStoreBatch* store = FindSnapshotStore(retract_round);
    ASSERT_NE(store, nullptr);
    EXPECT_GE(std::ranges::count(store->group_ids.at(0), 1u), 1) << "the state group's live checkpoint is imaged";

    SendForwardDone("b", {2000});
    SendAbort(*scheduler_, "a");
    SendAbort(*scheduler_, "b");
    AckImageStores(retract_round);
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), free_at_start) << "the aborted victim's image is released";
}

TEST(CacheProgressTest, JunctionSurvivesPrefillRounds) {
    BlockPool pool(/*num_lcm_blocks=*/8, {1, 1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState,
                       .sliding_window = 0,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 0},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/6, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> admission =
        AdmitForTest(coordinator, tables, /*num_tokens=*/4);
    ASSERT_TRUE(admission);

    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/0, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0,
                                                      fsm::CacheProgress{
                                                          .access_epoch = admission->access_epoch,
                                                          .junction_tokens = 8,
                                                      },
                                                      /*load_pairs=*/{}, /*awaits_result=*/false});
    ASSERT_TRUE(request.Is<fsm::Prefilling>());
    EXPECT_EQ(request.CacheProgress().junction_tokens, 8);

    request.Apply(fsm::SchedulePrefillEvent{
        /*tokens_this_round=*/4,
        /*reserve_num_tokens_in_next_schedule_event=*/1,
        /*awaits_result=*/false,
    });
    ASSERT_TRUE(request.Is<fsm::Prefilling>());
    EXPECT_EQ(request.CacheProgress().junction_tokens, 8);
}

// A hybrid prompt whose KV is cached further than any state checkpoint is cut
// at the junction: its first chunk ends where the KV coverage ends, the next
// admission publishes the natural checkpoint there, and later prompts sharing
// that prefix resume at the junction.
class JunctionCutSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 8;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = false;
        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("state", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kState),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(JunctionCutSuite, CutsTheFirstChunkAtTheJunctionAndPublishesTheCheckpointThere) {
    RequestSpec seed = MakeRequestSpec("seed", /*num_pages=*/6);
    Submit(seed);
    PlanOnce();  // chunk end at token 8: KV pages published, no state
    PlanOnce();  // prompt end at token 12
    SendForwardDone("seed", {42});
    PlanOnce();  // first decode: the prompt-end checkpoint (state@12) is published
    SendForwardDone("seed", {43});
    SendFinish("seed");
    PlanOnce();
    ASSERT_EQ(scheduler_->WaitingSize(), 0u);
    ASSERT_EQ(scheduler_->DecodingSize(), 0u);

    // `first` shares the seed's first six tokens: the KV hit reaches 6, no
    // state checkpoint lies at or below it, so it resumes at 0 with the
    // junction at 6 -- and, incomplete, holds the head of line.
    RequestSpec first = seed;
    first.request_id = "first";
    for (std::size_t i = 6; i < first.tokens.size(); ++i) {
        first.tokens[i] += 1000;
    }
    RequestSpec second = MakeRequestSpec("second", /*num_pages=*/6, /*start=*/2001);
    Submit({first, second});
    ASSERT_EQ(scheduler_->WaitingSize(), 2u);

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"first"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{0});
    EXPECT_EQ(batch->input_lengths, std::vector<std::int32_t>{6}) << "the chunk ends at the junction";
    const std::vector<std::int32_t> state_row = batch->block_tables.at("state").at(0);
    ASSERT_EQ(state_row.size(), 2u);
    EXPECT_EQ(std::ranges::count(plan.pages_to_zero.at("state"), state_row[0]), 1)
        << "no checkpoint: the chunk starts from a zeroed block";
    EXPECT_NE(state_row[1], 0);
    SendForwardDone("first");
    const PrefixHitStats at_first = scheduler_->GetHostCacheStats().prefix_hits;
    EXPECT_EQ(at_first.unresumable_kv_tokens, 6u) << "first's six cached KV tokens are recomputed";
    EXPECT_EQ(at_first.device_hit_tokens, 0u);

    // The next chunk's admission publishes the junction's natural checkpoint
    // (state@6 under the sixth token's page hash) and rotates the slots. The
    // chunk completes the prompt, so `second` joins the round behind it.
    plan = PlanOnce();
    batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, (std::vector<std::string>{"first", "second"}));
    EXPECT_EQ(batch->extend_prefix_lens.at(0), 6);
    const std::vector<std::int32_t> rotated = batch->block_tables.at("state").at(0);
    EXPECT_EQ(rotated[0], state_row[1]) << "the previous chunk's output is this chunk's input";
    EXPECT_NE(rotated[1], 0);
    EXPECT_NE(rotated[1], rotated[0]);
    SendForwardDone("first", {7});
    SendForwardDone("second");

    // `third` shares the same six tokens and resumes at the junction.
    RequestSpec third = seed;
    third.request_id = "third";
    for (std::size_t i = 6; i < third.tokens.size(); ++i) {
        third.tokens[i] += 3000;
    }
    Submit(third);
    for (int round = 0; round < 4; ++round) {
        plan = PlanOnce();
        batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        const auto it = std::ranges::find(batch->request_ids, "third");
        if (it == batch->request_ids.end()) {
            continue;
        }
        const auto row = static_cast<std::size_t>(it - batch->request_ids.begin());
        EXPECT_EQ(batch->extend_prefix_lens.at(row), 6) << "the junction checkpoint is a hit";
        EXPECT_EQ(batch->block_tables.at("state").at(row)[0], rotated[0]) << "resumes from the published checkpoint";
        const PrefixHitStats at_third = scheduler_->GetHostCacheStats().prefix_hits;
        EXPECT_EQ(at_third.device_hit_tokens, 6u);
        EXPECT_EQ(at_third.unresumable_kv_tokens, 6u) << "third's KV ends at its hit";
        EXPECT_EQ(at_third.host_hit_tokens, 0u) << "no Host pool";
        return;
    }
    FAIL() << "third was never scheduled";
}

class JunctionCutHostSuite : public JunctionCutSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = JunctionCutSuite::MakeConfig();
        cfg.disable_l2_cache = false;
        return cfg;
    }
};

// The same prompts with the seed on Host alone: the Host's KV reaches 6 but
// it holds no state checkpoint there, so `first` resumes at the Device's
// empty match, its first chunk is cut at the Host's KV reach, and `third`
// resumes at the checkpoint published there.
TEST_F(JunctionCutHostSuite, HostOnlyKvCutsTheFirstChunkAtTheJunction) {
    RequestSpec seed = MakeRequestSpec("seed", /*num_pages=*/6);
    Submit(seed);
    AckWriteBacks(PlanOnce());  // chunk end at token 8
    SendForwardDone("seed");
    AckWriteBacks(PlanOnce());  // prompt end at token 12
    SendForwardDone("seed", {42});
    SendFinish("seed");
    AckWriteBacks(PlanOnce());
    ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 7) << "six KV pages and state@12";
    ASSERT_TRUE(scheduler_->ClearL1Cache());
    // The clear evicts the six pages and state@12; Host holds all seven.
    const CacheCoordinator::DeviceEvictionCounts flushed = scheduler_->GetHostCacheStats().device_evictions.flush;
    EXPECT_EQ(flushed.blocks, 7u);
    EXPECT_EQ(flushed.host_resident, 7u);

    RequestSpec first = seed;
    first.request_id = "first";
    for (std::size_t i = 6; i < first.tokens.size(); ++i) {
        first.tokens[i] += 1000;
    }
    Submit(first);
    ExecutionPlan plan = PlanOnce();
    AckWriteBacks(plan);
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"first"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{0});
    EXPECT_EQ(batch->input_lengths, std::vector<std::int32_t>{6}) << "the chunk ends at the junction";
    SendForwardDone("first");
    const PrefixHitStats at_first = scheduler_->GetHostCacheStats().prefix_hits;
    EXPECT_EQ(at_first.unresumable_kv_tokens, 6u) << "first's six Host KV tokens are recomputed";
    EXPECT_EQ(at_first.host_hit_tokens, 0u);

    // The next chunk's admission publishes the junction's natural checkpoint
    // (state@6) over the KV the Host already holds.
    plan = PlanOnce();
    AckWriteBacks(plan);
    batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"first"});
    EXPECT_EQ(batch->extend_prefix_lens.at(0), 6);
    const std::int32_t checkpoint = batch->block_tables.at("state").at(0)[0];
    SendForwardDone("first", {7});

    // `third` shares the same six tokens and resumes at the junction.
    RequestSpec third = seed;
    third.request_id = "third";
    for (std::size_t i = 6; i < third.tokens.size(); ++i) {
        third.tokens[i] += 3000;
    }
    Submit(third);
    for (int round = 0; round < 4; ++round) {
        plan = PlanOnce();
        AckWriteBacks(plan);
        batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        const auto it = std::ranges::find(batch->request_ids, "third");
        if (it == batch->request_ids.end()) {
            continue;
        }
        const auto row = static_cast<std::size_t>(it - batch->request_ids.begin());
        EXPECT_EQ(batch->extend_prefix_lens.at(row), 6) << "the junction checkpoint is a hit";
        EXPECT_EQ(batch->block_tables.at("state").at(row)[0], checkpoint) << "resumes from the published checkpoint";
        const PrefixHitStats at_third = scheduler_->GetHostCacheStats().prefix_hits;
        EXPECT_EQ(at_third.device_hit_tokens, 6u);
        EXPECT_EQ(at_third.host_hit_tokens, 0u) << "the Device holds the junction's KV and checkpoint";
        return;
    }
    FAIL() << "third was never scheduled";
}

TEST(CacheProgressTest, RemotePrefillPreservesDecodeReserve) {
    BlockPool pool(/*num_lcm_blocks=*/8, {1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kD};
    request.Apply(fsm::BootstrappedEvent{});
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> admission =
        AdmitForTest(coordinator, tables, GroupDemand{.extent = DenseGrowth{4}, .reserve_tokens = 3});
    ASSERT_TRUE(admission);

    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/3, &req_pool,
                                                      fsm::PrefillSource::kRemote, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0,
                                                      fsm::CacheProgress{.access_epoch = admission->access_epoch},
                                                      /*load_pairs=*/{}, /*awaits_result=*/false});
    ASSERT_TRUE(request.Is<fsm::RemotePrefilling>());

    request.Apply(fsm::RemotePrefillDoneEvent{/*token=*/42});

    ASSERT_TRUE(request.Is<fsm::PrefillDone>());
    EXPECT_EQ(request.ReserveNumTokensInNextScheduleEvent(), 3);
}

TEST(RetractionStateFsmTest, ADecodingVictimSuspendsAndResumesDecodingWithTheSameTokens) {
    BlockPool device_pool(/*num_lcm_blocks=*/12, {1});
    BlockPool snapshot_pool(/*num_lcm_blocks=*/12, {1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    ReqPoolAllocator req_pool{4};
    SnapshotSlotAllocator slots{4};
    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kD};
    request.Apply(fsm::BootstrappedEvent{});
    std::vector<BlockTable> tables(coordinator.NumGroups());
    auto admission = AdmitForTest(coordinator, tables, GroupDemand{.extent = DenseGrowth{4}, .reserve_tokens = 1});
    ASSERT_TRUE(admission);
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        /*tokens_this_round=*/4,
        /*reserve_num_tokens_in_next_schedule_event=*/1,
        &req_pool,
        fsm::PrefillSource::kRemote,
        &coordinator,
        std::move(tables),
        /*hit_tokens=*/0,
        fsm::CacheProgress{.access_epoch = admission->access_epoch},
        /*load_pairs=*/{},
        /*awaits_result=*/false,
    });
    request.Apply(fsm::RemotePrefillDoneEvent{/*token=*/42});
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1});
    coordinator.ConsumeReservedTokens(request.BlockTablesRef(), 1);
    ASSERT_TRUE(request.Is<fsm::Decoding>());
    ASSERT_EQ(request.TokenSize(), 5);

    RetractForTest(request, coordinator, slots, /*epoch=*/1);
    ASSERT_TRUE(request.Is<fsm::Retracted>());
    EXPECT_EQ(request.PrefillSize(), 4) << "no rebase: the prompt window is intact";
    EXPECT_EQ(request.TokenSize(), 5);
    EXPECT_FALSE(request.HoldsPages());
    EXPECT_EQ(device_pool.NumEmptyLcmBlocks(), device_pool.NumLcmBlocks());
    EXPECT_EQ(req_pool.AvailableSlots(), 4) << "the request pool slot is released";
    const auto* retracted = request.GetIf<fsm::Retracted>();
    ASSERT_TRUE(std::holds_alternative<fsm::ResumeDecoding>(retracted->shape));
    EXPECT_TRUE(retracted->ResumesGeneration());
    EXPECT_EQ(retracted->cache_progress.access_epoch, admission->access_epoch) << "the progress survives";

    // The restore rebuilds the tables and takes a request pool slot again;
    // the request is not schedulable until the copy's ACK.
    std::vector<BlockTable> restored(coordinator.NumGroups());
    std::vector<GroupDemand> demands{GroupDemand{.table = &restored[0], .extent = DenseGrowth{0}, .reserve_tokens = 1}};
    const auto restore = coordinator.Restore(retracted->image, demands, admission->access_epoch);
    ASSERT_TRUE(restore);
    ASSERT_EQ(restore->snapshot_pairs.size(), 2u) << "the 4 computed prompt tokens; the decode input has no KV yet";
    request.Apply(fsm::ScheduleRestoreEvent{&coordinator, std::move(restored), /*restore_op=*/3});
    ASSERT_TRUE(request.Is<fsm::Restoring>());
    EXPECT_TRUE(request.HoldsPages());
    EXPECT_EQ(request.ResultsInFlight(), 0);
    EXPECT_EQ(request.GetIf<fsm::Restoring>()->restore_op, 3u);
    EXPECT_EQ(snapshot_pool.NumEmptyLcmBlocks(), 12 - 2) << "the image lives until the ACK";

    request.Apply(fsm::RestoreDoneEvent{req_pool.Allocate()});
    ASSERT_TRUE(request.Is<fsm::Decoding>());
    EXPECT_EQ(request.TokenSize(), 5);
    EXPECT_EQ(request.NumComputedTokens(), 4);
    EXPECT_EQ(request.ReserveNumTokensInNextScheduleEvent(), 1);
    EXPECT_EQ(request.BlockTablesRef()[0].NumBlocks(), 3) << "the imaged pages plus the re-reserved decode slot";
    EXPECT_EQ(slots.AvailableSlots(), 4) << "the blob slot returns with the image";
    EXPECT_TRUE(request.ResumedByRestore()) << "the first decode must carry its token: the new slot has no capture";
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1});
    EXPECT_FALSE(request.ResumedByRestore()) << "consumed by the first decode";
    // The restore op's pairs (held by the transfer manager in production) are
    // the last owners of the image blocks.
    EXPECT_EQ(snapshot_pool.NumEmptyLcmBlocks(), 12 - 2);
}

// Drive the FSM directly to pin the PrefillDone retract overload.
TEST(SnapshotRetractEvent, APrefillDoneVictimResumesAsPrefillDone) {
    BlockPool pool(/*num_lcm_blocks=*/8, {1, 1});
    BlockPool snapshot_pool(/*num_lcm_blocks=*/8, {1, 1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    ReqPoolAllocator req_pool{4};
    SnapshotSlotAllocator slots{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> admission =
        AdmitForTest(coordinator, tables, /*num_tokens=*/4);
    ASSERT_TRUE(admission);

    // Whole 4-token prompt in one chunk -> PrefillDone: holds pages, no decode yet.
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                      /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                      fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                      /*hit_tokens=*/0,
                                                      fsm::CacheProgress{.access_epoch = admission->access_epoch},
                                                      /*load_pairs=*/{}, /*awaits_result=*/false});
    ASSERT_TRUE(request.Is<fsm::PrefillDone>());
    EXPECT_EQ(request.CacheProgress().access_epoch, admission->access_epoch);
    ASSERT_LT(pool.NumEmptyLcmBlocks(), 8);

    // The last chunk's ExtendResult lands while still PrefillDone.
    request.Apply(fsm::ExtendResultEvent{{42}});

    RetractForTest(request, coordinator, slots, /*epoch=*/1);
    EXPECT_TRUE(request.Is<fsm::Retracted>());
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8) << "the retract must release every page";
    EXPECT_EQ(request.TokenSize(), 5);
    EXPECT_EQ(request.PrefillSize(), 4) << "the sampled token is not folded into the prompt";
    const auto* retracted = request.GetIf<fsm::Retracted>();
    const auto* shape = std::get_if<fsm::ResumePrefillDone>(&retracted->shape);
    ASSERT_NE(shape, nullptr);
    EXPECT_EQ(shape->window.begin + shape->window.size, 4);
    EXPECT_EQ(retracted->image.tables.size(), 2u) << "every group is imaged";
    EXPECT_EQ(retracted->image.tables[0].slots.size(), 2u);
    EXPECT_EQ(retracted->image.tables[1].slots.size(), 2u);

    std::vector<BlockTable> restored(coordinator.NumGroups());
    std::vector<GroupDemand> demands{
        GroupDemand{.table = &restored[0], .extent = DenseGrowth{0}, .reserve_tokens = 1},
        GroupDemand{.table = &restored[1], .extent = DenseGrowth{0}, .reserve_tokens = 1},
    };
    ASSERT_TRUE(coordinator.Restore(retracted->image, demands, admission->access_epoch));
    request.Apply(fsm::ScheduleRestoreEvent{&coordinator, std::move(restored), /*restore_op=*/1});
    request.Apply(fsm::RestoreDoneEvent{req_pool.Allocate()});
    ASSERT_TRUE(request.Is<fsm::PrefillDone>());
    EXPECT_EQ(request.NumComputedTokens(), 4);
    EXPECT_EQ(request.LastToken(), 42) << "the bootstrap token for its first decode is still there";
    EXPECT_TRUE(request.ResumedByRestore()) << "and that decode carries it explicitly on every role";
}

// ---------------------------------------------------------------------------
// Abort-mid-flight pool balance: abort mid-chunked-prefill or mid-decode must
// return every page to the pool.
// ---------------------------------------------------------------------------
TEST_F(ChunkedPrefillSuite, AbortMidPrefillRestoresPoolBaseline) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    // 12 tokens (6 pages), max_scheduled_tokens=4 -> abort lands mid-prefill.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    PlanOnce();  // chunk 1
    PlanOnce();  // chunk 2 -> still Prefilling
    EXPECT_LT(scheduler_->AvailableLcmBlocks(), free_at_start);

    SendAbort(*scheduler_, "r1");
    PlanOnce();  // reap the aborted request
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start)
        << "abort mid-prefill must return every page (both groups) to the pool";
}

TEST_F(ChunkedPrefillSuite, AbortDuringDecodeRestoresPoolBaseline) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();  // single-chunk prefill (4 tokens)
    SendForwardDone("r1", {42});
    PlanOnce();  // decode step
    SendForwardDone("r1", {43});
    EXPECT_LT(scheduler_->AvailableLcmBlocks(), free_at_start);

    SendAbort(*scheduler_, "r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start)
        << "abort during decode must return every page to the pool";
}

// Admission owns the prepared refs before the event. If the independent request
// slot allocation fails, event destruction releases those refs through RAII.
TEST(EventFailurePath, ReqPoolExhaustionAtFirstChunkLeavesPoolBalanced) {
    BlockPool pool(/*num_lcm_blocks=*/31, {1, 1});  // Pages are not the constraint.
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    ReqPoolAllocator req_pool{1};
    ReqPoolIndex held = req_pool.Allocate();  // exhaust the single slot
    ASSERT_EQ(req_pool.AvailableSlots(), 0);

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*granularity=*/2)};
    Request request{spec, /*prefix_granularity=*/2, Role::kFused};
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 27);

    EXPECT_THROW(
        request.Apply(fsm::SchedulePrefillFirstChunkEvent{/*tokens_this_round=*/4,
                                                          /*reserve_num_tokens_in_next_schedule_event=*/1, &req_pool,
                                                          fsm::PrefillSource::kLocal, &coordinator, std::move(tables),
                                                          /*hit_tokens=*/0,
                                                          /*cache_progress=*/{},
                                                          /*load_pairs=*/{}, /*awaits_result=*/false}),
        std::runtime_error);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 31) << "a failed req-pool Allocate must not leak block-pool pages";

    EXPECT_NO_THROW(request.Apply(fsm::AbortEvent{&coordinator}));
    EXPECT_TRUE(request.Is<fsm::Finished>());
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 31);
}

// ---------------------------------------------------------------------------
// SWA off-by-one regression: decode admission slides at the number of tokens
// computed before the pending query.
// ---------------------------------------------------------------------------
TEST(SwaWindowBoundary, DecodeStepKeepsOldestInWindowPageAtPageBoundary) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .extent = DenseGrowth{1},
                             },
                             RequestProgress{
                                 .num_computed_tokens = 4,
                             }));

    const auto swa_slot_null = [&](std::int32_t i) { return !tables[1].Blocks()[i]; };
    ASSERT_EQ(tables[1].NumBlocks(), 3);
    EXPECT_FALSE(swa_slot_null(0));

    // N=5; keys [2,5] -> page 0 out: slot 0 punched, slot 1 kept.
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .extent = DenseGrowth{1},
                             },
                             RequestProgress{
                                 .num_computed_tokens = 5,
                             }));
    EXPECT_TRUE(swa_slot_null(0));
    EXPECT_FALSE(swa_slot_null(1));

    // N=6; keys [3,6]: key 3 still lives in page 1, so slot 1 survives.
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .extent = DenseGrowth{1},
                             },
                             RequestProgress{
                                 .num_computed_tokens = 6,
                             }));
    EXPECT_FALSE(swa_slot_null(1)) << "key 3 of the pending query lives in page 1; freeing it is the off-by-one";
    EXPECT_TRUE(swa_slot_null(0));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before - 2);

    // N=7; keys [4,7] -> page 1 fully out, punched exactly now.
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .extent = DenseGrowth{1},
                             },
                             RequestProgress{
                                 .num_computed_tokens = 7,
                             }));
    EXPECT_TRUE(swa_slot_null(1));
    EXPECT_FALSE(swa_slot_null(2));

    for (const CacheBlockRef& block : tables[0].Blocks()) {
        EXPECT_TRUE(block);
    }
    FreeRequest(coordinator, tables);
}

// ---------------------------------------------------------------------------
// Physically-backed decode reservations cannot be stolen before consumption.
// ---------------------------------------------------------------------------
class PhysicalReserveSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 11;
        cfg.host_allocator.total_pages = 11;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full_a", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("full_b", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(PhysicalReserveSuite, LaterRequestCannotStealReservedDecodeHeadroom) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    ASSERT_EQ(free_at_start, 10);

    // a: exact admission acquires 6 prefill blocks plus 2 decode-reserve
    // blocks, leaving 2. b needs 4 and must defer.
    Submit(MakeRequestSpec("a", /*num_pages=*/3));
    Submit(MakeRequestSpec("b", /*num_pages=*/1, /*start=*/101));
    ExecutionPlan round1 = PlanOnce();
    const ForwardBatch* op1 = FindForwardBatch(round1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 1u) << "b must not be admitted into a's reserved decode blocks";
    EXPECT_EQ(op1->request_ids.at(0), "a");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 2);

    // a's decode transition consumes its already-owned reservation.
    SendForwardDone("a", {99});
    ExecutionPlan round2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(round2);
    ASSERT_NE(op2, nullptr);
    ASSERT_EQ(op2->request_ids.size(), 1u) << "a's decode must proceed into its reserved pages";
    EXPECT_EQ(op2->request_ids.at(0), "a");
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 2);
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);

    SendForwardDone("a", {100});
    SendFinish("a");
    ExecutionPlan round3 = PlanOnce();
    const ForwardBatch* op3 = FindForwardBatch(round3);
    ASSERT_NE(op3, nullptr);
    ASSERT_EQ(op3->request_ids.size(), 1u);
    EXPECT_EQ(op3->request_ids.at(0), "b");

    SendForwardDone("b", {142});
    SendFinish("b");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

TEST_F(PhysicalReserveSuite, AbortWithOutstandingReservationLeavesNoPhantom) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    ASSERT_EQ(free_at_start, 10);

    // a owns a 2-block physical decode reservation (see above).
    Submit(MakeRequestSpec("a", /*num_pages=*/3));
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 2);

    // Abort before the reserve is consumed: RAII must release it too.
    SendAbort(*scheduler_, "a");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);

    // b needs the whole pool: gate 2*ceil(9/2) = 10 <= 10 only without a phantom.
    Submit(MakeRequestSpec("b", /*num_pages=*/4, /*start=*/101));
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u) << "a leaked reservation would defer b forever";
    EXPECT_EQ(op->request_ids.at(0), "b");
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0);

    SendForwardDone("b", {142});
    SendFinish("b");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Cross-request prefix hits, end to end: admission match -> FSM claim -> input
// window starts past the hit (disable_prefix_cache=false, W=32).
// ---------------------------------------------------------------------------
class PrefixHitSuite : public SchedulerTestSuite {
protected:
    virtual std::int32_t SlidingWindowTokens() const { return 32; }
    virtual bool DisablePrefixCache() const { return false; }
    virtual std::int32_t TotalPages() const { return 64; }

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = TotalPages();
        cfg.host_allocator.total_pages = TotalPages();
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = DisablePrefixCache();

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      SlidingWindowTokens()),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    RequestSpec MakeSpecWithTokens(const std::string& id, std::vector<std::int32_t> tokens) {
        return RequestSpec{.request_id = id, .tokens = std::move(tokens)};
    }

    // Prefill -> one decode round -> finish; returns the PREFILL op's per-group
    // rows. The decode round is load-bearing: the finalize registers the page
    // hashes, and finish frees the blocks WITH hashes intact (still matchable).
    std::map<std::string, std::vector<std::int32_t>> RunLifecycle(const RequestSpec& spec) {
        Submit(spec);
        ExecutionPlan prefill = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(prefill);
        EXPECT_NE(op, nullptr);
        std::map<std::string, std::vector<std::int32_t>> rows;
        if (op != nullptr) {
            for (const auto& [gid, table] : op->block_tables) {
                rows[gid] = table.at(0);
            }
        }
        SendForwardDone(spec.request_id, {9001});
        PlanOnce();  // PrefillDone -> Decoding: finalize registers the hashes
        SendForwardDone(spec.request_id, {9002});
        SendFinish(spec.request_id);
        PlanOnce();  // reap
        return rows;
    }

    static void ExpectRowPrefixEq(const std::vector<std::int32_t>& row,
                                  const std::vector<std::int32_t>& expected_prefix, const char* what) {
        ASSERT_GE(row.size(), expected_prefix.size()) << what;
        for (std::size_t i = 0; i < expected_prefix.size(); ++i) {
            EXPECT_EQ(row[i], expected_prefix[i]) << what << " slot " << i;
        }
    }
};

TEST_F(PrefixHitSuite, TwoRequestsSharePrefixReusePages) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "r1 must fully reclaim before r2 runs";
    ASSERT_EQ(r1_rows.at("full").size(), 5u);
    ASSERT_EQ(r1_rows.at("swa").size(), 5u);

    // r2: 12 tokens, first 8 == r1's. Hit: cap = (12-1)/2 = 5 pages; r1
    // registered 4, r2's page-4 hash chains off different tail tokens -> full
    // hits 4; swa (W=32, needed 16 > 4) keeps 4 -> fixpoint 4 blocks = 8 tokens.
    std::vector<std::int32_t> r2_tokens =
        MakeAlignedTokens(/*num_pages=*/4, PrefixGranularity());  // tokens 1..8 == r1's
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/4, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 4);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->prefill_lengths.at(0), 12);
    EXPECT_EQ(op->input_ids, tail);
    // Complete group rows contain 4 claimed + ceil(4/2) fresh + 1
    // preallocated decode page = 7.
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 7u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 7u);

    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    const std::vector<std::int32_t> swa_prefix(r1_rows.at("swa").begin(), r1_rows.at("swa").begin() + 4);
    ExpectRowPrefixEq(op->block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op->block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: 4 hit blocks/group remain active, 2 fresh blocks/group are
    // acquired, and 1 decode-reserve block/group is physically held.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 14);

    // Finalize registers pages 4..5 and consumes the existing reservation.
    SendForwardDone("r2", {199});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 14);

    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "pool back to baseline after r2 finishes";
}

TEST_F(PrefixHitSuite, FinishPublishesPagesFromLastForward) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    Submit(first);
    const ExecutionPlan first_plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(first_plan), nullptr);

    // Finish before another scheduling round can publish the prefill pages.
    SendForwardDone("r1", {9001});
    SendFinish("r1");
    PlanOnce();

    Submit(RequestSpec{.request_id = "r2", .tokens = first.tokens});
    const ExecutionPlan second_plan = PlanOnce();
    const ForwardBatch* second = FindForwardBatch(second_plan);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(second->request_ids.size(), 1u);
    EXPECT_EQ(second->input_lengths.at(0), 2);
    EXPECT_EQ(second->extend_prefix_lens.at(0), 6);
}

TEST_F(PrefixHitSuite, ClearL1CacheRemovesAnIdlePrefix) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    RunLifecycle(first);

    const auto [cleared, log] = ClearL1CacheWithCapturedLog(scheduler_.get());
    ASSERT_TRUE(cleared);
    EXPECT_NE(log.find("flush L1 cache completed"), std::string::npos);
    // r1 published 9 computed tokens: four pages and a one-token tail in each
    // of the two groups. No Host pool holds anything.
    const CacheCoordinator::DeviceEvictionStats evictions = scheduler_->GetHostCacheStats().device_evictions;
    EXPECT_EQ(evictions.flush.blocks, 10u);
    EXPECT_EQ(evictions.flush.host_resident, 0u);
    EXPECT_EQ(evictions.admission.blocks + evictions.restore.blocks + evictions.tail_cap.blocks, 0u);
    Submit(RequestSpec{.request_id = "r2", .tokens = first.tokens});
    const ExecutionPlan second_plan = PlanOnce();
    const ForwardBatch* second = FindForwardBatch(second_plan);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(second->request_ids.size(), 1u);
    EXPECT_EQ(second->input_lengths.at(0), 8);
    EXPECT_EQ(second->extend_prefix_lens.at(0), 0);
}

TEST_F(PrefixHitSuite, ClearL1CacheRejectsAnActiveRequestAndPreservesItsPrefix) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    Submit(first);
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    SendForwardDone("r1", {9001});
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);

    // The active request's pages are pinned, so the coordinator refuses the
    // clear and its prefix survives for the next request to match.
    const auto [cleared, log] = ClearL1CacheWithCapturedLog(scheduler_.get());
    EXPECT_FALSE(cleared);
    EXPECT_NE(log.find("cached blocks are still pinned"), std::string::npos);
    SendForwardDone("r1", {9002});
    SendFinish("r1");
    PlanOnce();

    Submit(RequestSpec{.request_id = "r2", .tokens = first.tokens});
    const ExecutionPlan second_plan = PlanOnce();
    const ForwardBatch* second = FindForwardBatch(second_plan);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(second->request_ids.size(), 1u);
    EXPECT_EQ(second->input_lengths.at(0), 2);
    EXPECT_EQ(second->extend_prefix_lens.at(0), 6);
}

// The hit is capped at (PrefillSize-1)/prefix_granularity pages so the last token is
// always recomputed to produce logits.
TEST_F(PrefixHitSuite, FullHitCapsAtLastToken) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    const RequestSpec r1 = MakeRequestSpec("r1", /*num_pages=*/4);  // 8 tokens
    RunLifecycle(r1);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);

    // r2 = the same 8 tokens: cap = (8-1)/2 = 3 pages -> hit 3 = 6 tokens.
    Submit(MakeSpecWithTokens("r2", r1.tokens));
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 6);
    // input = tokens [6, 8) of the 1..8 sequence.
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/7));
    // 3 claimed + 1 fresh + 1 preallocated decode page per group.
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 5u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 5u);
    // Pool: 3 hit + 1 fresh + 1 reserved block per group.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 10);

    // Reserve: 1 fresh page per group (tail full).
    SendForwardDone("r2", {199});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 10);
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

class PrefixReplaySuite : public PrefixHitSuite {
protected:
    virtual std::int32_t PrefixReplayTokens() const { return 4; }

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = PrefixHitSuite::MakeConfig();
        cfg.prefix_replay_tokens = PrefixReplayTokens();
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(PrefixReplaySuite, FullHitReplaysPrivateTailPages) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);  // 8 tokens
    const auto first_rows = RunLifecycle(first);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);

    Submit(MakeSpecWithTokens("r2", first.tokens));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    // Four replay tokens cap the hit at two 2-token pages. The remaining
    // prompt tail is ordinary prefill input, not a claimed shared page.
    EXPECT_EQ(op->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(op->input_lengths.at(0), 4);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/4, /*start=*/5));
    for (const char* group_id : {"full", "swa"}) {
        const auto& row = op->block_tables.at(group_id).at(0);
        ASSERT_EQ(row.size(), 5u);  // 2 hit + 2 private replay + 1 decode reserve
        EXPECT_EQ(row[0], first_rows.at(group_id)[0]);
        EXPECT_EQ(row[1], first_rows.at(group_id)[1]);
        EXPECT_NE(row[2], first_rows.at(group_id)[2])
            << group_id << " replay page must not alias the cached shared page";
        EXPECT_GT(row[2], 0);
        EXPECT_GT(row[3], 0);
        EXPECT_GT(row[4], 0);
    }
}

TEST_F(PrefixReplaySuite, ExistingUncachedSuffixSatisfiesReplayRequirement) {
    RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));  // tokens 1..8

    // Only four tokens match. The eight-token uncached suffix already exceeds
    // prefix_replay_tokens=4, so the ordinary partial hit stays unchanged.
    std::vector<std::int32_t> tokens = MakeTokens(/*count=*/4);
    const std::vector<std::int32_t> suffix = MakeTokens(/*count=*/8, /*start=*/801);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    Submit(MakeSpecWithTokens("r2", tokens));

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->input_ids, suffix);
}

TEST_F(PrefixReplaySuite, ReplayedPagesArePublishedForTheNextRequest) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    RunLifecycle(first);

    // r2 recomputes and republishes the tail behind the capped four-token hit.
    RunLifecycle(MakeSpecWithTokens("r2", first.tokens));

    // Extending the prompt lets r3 probe beyond its own four-token replay tail.
    // It must hit all eight tokens from r2, including r2's replayed pages.
    std::vector<std::int32_t> extended = first.tokens;
    const std::vector<std::int32_t> suffix = MakeTokens(/*count=*/4, /*start=*/801);
    extended.insert(extended.end(), suffix.begin(), suffix.end());
    Submit(MakeSpecWithTokens("r3", extended));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->input_lengths.at(0), 4);
    EXPECT_EQ(op->input_ids, suffix);
}

class PrefixReplayLargerThanPromptSuite : public PrefixReplaySuite {
protected:
    std::int32_t PrefixReplayTokens() const override { return 32; }
};

TEST_F(PrefixReplayLargerThanPromptSuite, FallsBackToFullPromptPrefill) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    RunLifecycle(first);

    Submit(MakeSpecWithTokens("r2", first.tokens));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->input_ids, first.tokens);
}

class PrefixReplayDisabledSuite : public PrefixReplaySuite {
protected:
    bool DisablePrefixCache() const override { return true; }
};

TEST_F(PrefixReplayDisabledSuite, DisabledPrefixCacheStillPrefillsTheFullPrompt) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);
    RunLifecycle(first);

    Submit(MakeSpecWithTokens("r2", first.tokens));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->input_ids, first.tokens);
}

class PrefixReplayHeterogeneousSuite : public PrefixHitSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 8;
        cfg.device_allocator.total_pages = 128;
        cfg.host_allocator.total_pages = 0;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = false;
        cfg.prefix_replay_tokens = 8;

        CacheGroupConfig history =
            MakeGroup("history", /*block_granularity=*/8, cfg.device_allocator.total_pages, AttnKind::kFull);
        CacheGroupConfig state =
            MakeGroup("state", /*block_granularity=*/2, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/32);
        state.cache_blocks_per_lcm_block = 4;
        cfg.cache_groups = {history, state};
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(PrefixReplayHeterogeneousSuite, ReplayTailIsPrivateAcrossPackedGroups) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/4);  // 32 tokens
    const auto first_rows = RunLifecycle(first);

    Submit(MakeSpecWithTokens("r2", first.tokens));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 24);
    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/8, /*start=*/25));

    const auto& history = op->block_tables.at("history").at(0);
    ASSERT_EQ(history.size(), 5u);  // 3 hit + 1 replay + 1 decode reserve
    EXPECT_EQ(history[0], first_rows.at("history")[0]);
    EXPECT_EQ(history[1], first_rows.at("history")[1]);
    EXPECT_EQ(history[2], first_rows.at("history")[2]);
    EXPECT_NE(history[3], first_rows.at("history")[3]);

    const auto& state = op->block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 17u);  // 12 hit + 4 replay + 1 decode reserve
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(state[i], first_rows.at("state")[i]);
    }
    for (std::size_t i = 12; i < 16; ++i) {
        EXPECT_NE(state[i], first_rows.at("state")[i]);
        EXPECT_GT(state[i], 0);
    }
    EXPECT_GT(state[16], 0);
}

// A request returning prompt logprobs from position `s` caps its probe at `s`
// (RequestSpec::max_cached_prefix_tokens) so positions >= s are recomputed.
TEST_F(PrefixHitSuite, MaxCachedPrefixTokensCapsTheProbe) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/6);  // 12 tokens
    const auto first_rows = RunLifecycle(first);

    // Logprobs from position 5: the probe may claim at most 5 tokens, which
    // rounds down to two 2-token pages; positions 4.. are recomputed.
    RequestSpec capped = MakeSpecWithTokens("r2", first.tokens);
    capped.max_cached_prefix_tokens = 5;
    Submit(capped);
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/8, /*start=*/5));
    const auto& row = op->block_tables.at("full").at(0);
    ASSERT_GE(row.size(), 3u);
    EXPECT_EQ(row[0], first_rows.at("full")[0]);
    EXPECT_EQ(row[1], first_rows.at("full")[1]);
    EXPECT_NE(row[2], first_rows.at("full")[2]) << "a recomputed page must not alias the cached one";
    SendForwardDone("r2", {9001});
    PlanOnce();
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    PlanOnce();

    // A cap of zero means every position is recomputed: no hit at all.
    RequestSpec uncached = MakeSpecWithTokens("r3", first.tokens);
    uncached.max_cached_prefix_tokens = 0;
    Submit(uncached);
    plan = PlanOnce();
    op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(op->input_lengths.at(0), 12);
    EXPECT_EQ(op->input_ids, first.tokens);
}

TEST_F(PrefixHitSuite, MaxCachedPrefixTokensDefaultLeavesTheProbeUnbounded) {
    const RequestSpec first = MakeRequestSpec("r1", /*num_pages=*/6);  // 12 tokens
    RunLifecycle(first);

    // The default (INT32_MAX) keeps the ordinary replay-tail rule: 12 - 1
    // cacheable tokens -> 5 pages hit, one recomputed tail page.
    Submit(MakeSpecWithTokens("r2", first.tokens));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 10);
    EXPECT_EQ(op->input_lengths.at(0), 2);

    RequestSpec negative = MakeSpecWithTokens("r3", first.tokens);
    negative.max_cached_prefix_tokens = -1;
    EXPECT_THROW(Submit(negative), std::invalid_argument);
}

// The probe bound across a retraction. 8-token chunks, prefix cache live, no
// host tier: a victim's own pages are gone when it readmits, and only another
// request's equal pages can answer its probe -- which is exactly what the
// bound must keep it from claiming past what it had computed.
class ProbeBoundReadmissionSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        // 31 physical pages -> 30 usable. "twin" decoding over 18 tokens holds
        // 2*9 = 18 blocks and "capped"'s first 8-token chunk 2*4 = 8, leaving
        // 4: too few for the second chunk's 2*4 pages plus its decode slot.
        cfg.device_allocator.total_pages = 31;
        cfg.host_allocator.total_pages = 32;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 8;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = false;
        cfg.cache_groups = {
            MakeGroup("full_a", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("full_b", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // "twin" prefills a 16-token prompt in two chunks and takes one decode
    // step, which publishes all eight prompt pages; it holds them while it
    // decodes.
    void RunTwinToDecoding() {
        const RequestSpec twin = MakeRequestSpec("twin", /*num_pages=*/8);
        prompt_ = twin.tokens;
        Submit(twin);
        ExecutionPlan chunk1 = PlanOnce();
        ASSERT_NE(FindForwardBatch(chunk1), nullptr);
        ASSERT_EQ(FindForwardBatch(chunk1)->input_lengths.at(0), 8);
        SendForwardDone("twin");
        ExecutionPlan chunk2 = PlanOnce();
        ASSERT_NE(FindForwardBatch(chunk2), nullptr);
        ASSERT_EQ(FindForwardBatch(chunk2)->extend_prefix_lens.at(0), 8);
        SendForwardDone("twin", {42});
        PlanOnce();  // first decode step: publishes the second chunk's pages
        SendForwardDone("twin", {43});
    }

    // "capped" is the same prompt returning logprobs from position 0: its
    // first admission may match nothing. Schedules its first chunk.
    void SubmitCappedFirstChunk() {
        RequestSpec capped = RequestSpec{.request_id = "capped", .tokens = prompt_};
        capped.max_cached_prefix_tokens = 0;
        Submit(capped);
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(plan);
        ASSERT_NE(op, nullptr);
        const auto row = std::ranges::find(op->request_ids, "capped");
        ASSERT_NE(row, op->request_ids.end());
        const auto i = static_cast<std::size_t>(std::distance(op->request_ids.begin(), row));
        ASSERT_EQ(op->extend_prefix_lens.at(i), 0) << "the bound of 0 must not match the twin's pages";
        ASSERT_EQ(op->input_lengths.at(i), 8);
    }

    // Drives rounds until "capped" is scheduled again and returns that row's
    // (extend_prefix_len, input_length).
    std::pair<std::int32_t, std::int32_t> ReadmitCapped() {
        for (int round = 0; round < 6; ++round) {
            const ExecutionPlan plan = PlanOnce();
            const ForwardBatch* op = FindForwardBatch(plan);
            if (op == nullptr) {
                continue;
            }
            const auto row = std::ranges::find(op->request_ids, "capped");
            if (row == op->request_ids.end()) {
                SendForwardDone("twin", {44 + round});
                continue;
            }
            const auto i = static_cast<std::size_t>(std::distance(op->request_ids.begin(), row));
            return {op->extend_prefix_lens.at(i), op->input_lengths.at(i)};
        }
        ADD_FAILURE() << "capped was never readmitted";
        return {-1, -1};
    }

    std::vector<std::int32_t> prompt_;
};

// A victim retracted mid-prefill never probes: its restore copies the chunks
// it computed back and the next chunk follows them, whatever the twin's equal
// pages would have offered.
TEST_F(ProbeBoundReadmissionSuite, APrefillVictimResumesAfterItsComputedChunksWithoutProbing) {
    RunTwinToDecoding();
    SubmitCappedFirstChunk();
    SendForwardDone("capped");  // the chunk landed: positions [0, 8) have logits

    // The second chunk does not fit. Nobody else needs pages while the twin
    // decodes inside its reserve, so nothing is retracted; once the twin
    // needs a page too, the blocked round retracts the incomplete prefill for
    // it and images its 8 computed tokens.
    ExecutionPlan retract_round;
    std::int32_t next_twin = 44;
    for (int round = 0; round < 8 && scheduler_->RetractedSize() == 0u; ++round) {
        retract_round = PlanOnce();
        for (const std::string& id : FindForwardBatch(retract_round)->request_ids) {
            ASSERT_EQ(id, "twin");
            SendForwardDone(id, {next_twin++});
        }
    }
    ASSERT_EQ(scheduler_->RetractedSize(), 1u) << "the incomplete prefill gave way";
    ASSERT_NE(FindSnapshotStore(retract_round), nullptr);
    AckImageStores(retract_round);
    SendFinish("twin");

    const ExecutionPlan readmit = PlanOnce();
    ASSERT_NE(FindRestore(readmit), nullptr);
    ASSERT_TRUE(FindForwardBatch(readmit)->request_ids.empty()) << "no prefill before the restore lands";
    AckRestores(readmit);
    const auto [prefix_len, input_len] = ReadmitCapped();
    EXPECT_EQ(prefix_len, 8) << "the next chunk starts after the computed ones";
    EXPECT_EQ(input_len, 8);
}

TEST(PrefixReplayConfigTest, RejectsNegativeReplayTokens) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = 2;
    cfg.device_allocator.total_pages = 8;
    cfg.max_scheduled_tokens = 8;
    cfg.max_batch_size = 1;
    cfg.prefix_replay_tokens = -1;
    cfg.cache_groups = {
        MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
    };
    SetTestSnapshotPool(cfg);
    EXPECT_THROW((void)Scheduler(std::move(cfg)), std::invalid_argument);
}

class PrefixHitDisabledSuite : public PrefixHitSuite {
protected:
    bool DisablePrefixCache() const override { return true; }
};

TEST_F(PrefixHitDisabledSuite, DisablePrefixCacheSkipsMatch) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);

    std::vector<std::int32_t> r2_tokens = MakeAlignedTokens(/*num_pages=*/4, PrefixGranularity());
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/4, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 12) << "no hit -> the whole prompt is the input";
    EXPECT_EQ(op->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(op->input_ids, r2_tokens);
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 7u) << "six prompt pages plus one preallocated decode page";
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 7u);
    // Pool: 6 live pages plus one physically reserved decode page per group.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 14);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

TEST_F(PrefixHitSuite, PartialHit) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));  // tokens 1..8
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);

    // r2: 12 tokens, only the first 4 match r1 (pages 0..1); the hash chain
    // propagates the divergence to every later page. Hit = 2 pages = 4 tokens.
    std::vector<std::int32_t> r2_tokens = MakeTokens(/*count=*/4);  // 1..4 == r1's first 4
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/8, /*start=*/801);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(op->input_ids, tail);
    // 2 claimed + 4 fresh + 1 preallocated decode page per group.
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 7u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 7u);

    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 2);
    const std::vector<std::int32_t> swa_prefix(r1_rows.at("swa").begin(), r1_rows.at("swa").begin() + 2);
    ExpectRowPrefixEq(op->block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op->block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: 2 hit + 4 fresh + 1 reserved block per group.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 14);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

// Small window: the SWA group's bounded right-to-left scan stops once its
// contiguous run is satisfied, claiming r1's punched slots as null holes.
class PrefixHitSmallWindowSuite : public PrefixHitSuite {
protected:
    std::int32_t SlidingWindowTokens() const override { return 4; }
};

TEST_F(PrefixHitSmallWindowSuite, SwaGroupHitRespectsWindow) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    // r1's finalize REGISTERS all 4 swa hashes BEFORE ReclaimExpired(8) punches
    // slots 0,1 -- punched blocks reach the free list with hashes, matchable.
    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
    ASSERT_EQ(r1_rows.at("swa").size(), 5u);

    // r2: 10 tokens, first 8 == r1's. Fixpoint (W=4, page=2, pages_needed
    // = ceil(3/2) = 2): cap = (10-1)/2 = 4, full matches 4; swa scan stops at
    // run 2 -> keep 4 with 2 holes -> common stays 4 = 8 hit tokens.
    std::vector<std::int32_t> r2_tokens = MakeAlignedTokens(/*num_pages=*/4, PrefixGranularity());  // 1..8 == r1's
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/2, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->input_ids, tail);
    // 4 claimed slots (real or hole) + 1 fresh + 1 preallocated decode page.
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 6u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 6u);

    const auto& full_row = op->block_tables.at("full").at(0);
    ASSERT_EQ(full_row.size(), 6u);
    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    ExpectRowPrefixEq(full_row, full_prefix, "full row");
    EXPECT_GT(full_row[4], 0);
    EXPECT_GT(full_row[5], 0);

    const auto& swa_row = op->block_tables.at("swa").at(0);
    ASSERT_EQ(swa_row.size(), 6u);
    EXPECT_EQ(swa_row[0], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[1], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[2], r1_rows.at("swa")[2]);
    EXPECT_EQ(swa_row[3], r1_rows.at("swa")[3]);
    EXPECT_GT(swa_row[4], 0);
    EXPECT_GT(swa_row[5], 0);
    // Window invariant (mirrors ExpectSwaWindowIntact): the last
    // pages_needed = 2 slots of the claimed prefix must be real.
    for (std::size_t i = 2; i < 4; ++i) {
        EXPECT_GT(swa_row[i], 0) << "null hole inside the last window of the claimed prefix at slot " << i;
    }

    // Pool: full claims 4 + swa claims 2 (holes claim nothing) + 1 fresh
    // page/group + one physically reserved decode page/group = 10.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 10);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

// Near capacity, exact admission must protect prefix-hit parents while finding
// placements for the fresh suffix and decode reservation.
class PrefixHitTightPoolSuite : public PrefixHitSuite {
protected:
    // Twelve usable parents: the startup bound admits an eight-token prompt
    // (five blocks plus a tail handover per group).
    std::int32_t TotalPages() const override { return 13; }
};

TEST_F(PrefixHitTightPoolSuite, ProtectedHitAndFreshDemandMustFitTogether) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    ASSERT_EQ(free_at_start, 12);

    // r1 leaves 4 cached, evictable parents plus 8 empty parents. The capacity
    // metric reports all 12 as available, but only the latter are unbound.
    RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/2));
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);

    // r3 holds 2 prefill pages plus 1 decode-reserve page per group: 12 -> 6.
    Submit(MakeRequestSpec("r3", /*num_pages=*/2, /*start=*/501));
    ExecutionPlan r3_prefill = PlanOnce();
    ASSERT_NE(FindForwardBatch(r3_prefill), nullptr);
    ASSERT_EQ(FindForwardBatch(r3_prefill)->request_ids.size(), 1u);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 6);

    // r3's finalize consumes its already physical decode reservation.
    SendForwardDone("r3", {599});
    PlanOnce();
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 6);

    // r2: 8 tokens, first 4 == r1's. The 4 cached hit parents are protected.
    // Its suffix and reserve need 6 empty parents, but r3 pins 6 and leaves only
    // 2 empty, so the whole admission defers without acquiring the hits.
    std::vector<std::int32_t> r2_tokens =
        MakeAlignedTokens(/*num_pages=*/2, PrefixGranularity());  // tokens 1..4 == r1's
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/4, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));
    ExecutionPlan blocked = PlanOnce();
    const ForwardBatch* blocked_op = FindForwardBatch(blocked);
    ASSERT_NE(blocked_op, nullptr);
    ASSERT_EQ(blocked_op->request_ids.size(), 1u) << "r2 must be deferred, not admitted into a short pool";
    EXPECT_EQ(blocked_op->request_ids.at(0), "r3");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "deferred r2 stays intact in the waiting set";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 6) << "a deferred first chunk must not touch the pool";

    // Both of r3's decodes land and it finishes -> 12 available parents, 6 of
    // them r3's cached blocks. r2 protects 4 hit parents and acquires 4 fresh
    // plus 2 physically reserved parents: the 2 empty ones and 4 evicted from
    // r3, leaving 2 of r3's.
    SendForwardDone("r3", {600});
    SendForwardDone("r3", {601});
    SendFinish("r3");
    ExecutionPlan plan2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(plan2);
    ASSERT_NE(op2, nullptr) << "deferred request must be schedulable after r3 releases its pages";
    ASSERT_EQ(op2->request_ids.size(), 1u);
    EXPECT_EQ(op2->request_ids.at(0), "r2");
    EXPECT_EQ(op2->input_lengths.at(0), 4) << "only the 4-token remainder is computed";
    EXPECT_EQ(op2->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 2);

    // r2's finalize consumes the existing reservation, so capacity stays put.
    SendForwardDone("r2", {699});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 2);

    SendForwardDone("r2", {700});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "pool back to baseline after both complete";
}

// ---------------------------------------------------------------------------
// M13 decode-block caching: pages filled DURING decode register via the hash
// chain (admission: register -> slide -> acquire), so a later turn hits PAST
// the previous prompt boundary. Fill timing: a round at container Size s has
// N = s - 1 computed and registers pages up to N/prefix_granularity -- a tail page
// registers one round late, or at the finish, which publishes the frontier
// once every forward has landed.
// ---------------------------------------------------------------------------
class DecodeCachingSuite : public PrefixHitSuite {
protected:
    // Deliver one sampled token and run the next schedule round, returning the
    // per-group rows the round's op carried. Single-request rounds only.
    std::map<std::string, std::vector<std::int32_t>> AdvanceOneRound(const std::string& id, std::int32_t token) {
        SendForwardDone(id, {token});
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(plan);
        EXPECT_NE(op, nullptr);
        std::map<std::string, std::vector<std::int32_t>> rows;
        if (op != nullptr) {
            for (const auto& [gid, table] : op->block_tables) {
                rows[gid] = table.at(0);
            }
        }
        return rows;
    }

    // Turn 1: prompt {1,2,3,4}, generated 101..106 (page=2). Finalize registers
    // prompt pages 0,1; +103 (N=6) registers page 2; +105 (N=8) registers page
    // 3 (tail one round late: 105 exists only to push N past 8). That round's
    // decode lands (106) so the finish is quiescent and publishes the tail
    // [105], which no later turn shares. Returns the +105 round's rows: 5
    // slots, the first 4 = the conversation's pages 0..3.
    std::map<std::string, std::vector<std::int32_t>> RunTurnOne() {
        Submit(MakeRequestSpec("r1", /*num_pages=*/2));
        ExecutionPlan prefill = PlanOnce();
        EXPECT_NE(FindForwardBatch(prefill), nullptr);
        AdvanceOneRound("r1", 101);
        AdvanceOneRound("r1", 102);
        AdvanceOneRound("r1", 103);
        AdvanceOneRound("r1", 104);
        auto rows = AdvanceOneRound("r1", 105);
        SendForwardDone("r1", {106});
        SendFinish("r1");
        PlanOnce();  // reap
        return rows;
    }

    // Turn-2 prompt: r1's 4 prompt tokens + first 4 generated + 2 new = 10;
    // pages 0..3 match r1's registration by content.
    std::vector<std::int32_t> MakeTurnTwoPrompt() {
        std::vector<std::int32_t> tokens =
            MakeAlignedTokens(/*num_pages=*/2, PrefixGranularity());                        // {1,2,3,4} == r1's prompt
        const std::vector<std::int32_t> response = MakeTokens(/*count=*/4, /*start=*/101);  // r1's generated 101..104
        tokens.insert(tokens.end(), response.begin(), response.end());
        const std::vector<std::int32_t> fresh = MakeTokens(/*count=*/2, /*start=*/901);
        tokens.insert(tokens.end(), fresh.begin(), fresh.end());
        return tokens;
    }

    // Turn-3 prompt: turn 2's full 13-token stream + 3 new tokens = 16.
    std::vector<std::int32_t> MakeTurnThreePrompt() {
        std::vector<std::int32_t> tokens = MakeTurnTwoPrompt();
        const std::vector<std::int32_t> r2_response = MakeTokens(/*count=*/3, /*start=*/201);
        tokens.insert(tokens.end(), r2_response.begin(), r2_response.end());
        const std::vector<std::int32_t> fresh = MakeTokens(/*count=*/3, /*start=*/951);
        tokens.insert(tokens.end(), fresh.begin(), fresh.end());
        return tokens;
    }
};

TEST_F(DecodeCachingSuite, DecodeFilledPageBecomesHittable) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    const auto r1_rows = RunTurnOne();
    ASSERT_EQ(r1_rows.at("full").size(), 5u);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "r1 must fully reclaim before r2 runs";

    // Hit: cap = (10-1)/2 = 4 -> pages 0..3, all registered by r1 (RunTurnOne);
    // swa (W=32, needed 16 > 4) keeps 4 -> fixpoint 4 blocks = 8 hit tokens.
    Submit(MakeSpecWithTokens("r2", MakeTurnTwoPrompt()));
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->prefill_lengths.at(0), 10);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/901));
    // 4 claimed + 1 fresh + 1 preallocated decode page.
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 6u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 6u);

    // Slots 2,3 are the pages r1's decode filled, beyond its prompt boundary.
    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    const std::vector<std::int32_t> swa_prefix(r1_rows.at("swa").begin(), r1_rows.at("swa").begin() + 4);
    ExpectRowPrefixEq(op->block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op->block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: claim 4/group (8) + 1 fresh/group (2) + 1 reserved/group (2).
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 12);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "pool back to baseline after r2 finishes";
}

TEST_F(DecodeCachingSuite, MultiTurnConversationReusesResponsePages) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    RunTurnOne();  // registers conversation pages 0..3
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);

    // Turn 2: hit 4 pages, then decode 201..203: +201 finalize registers page
    // 4 = {901,902}; the finish after 203 lands publishes the frontier at N=12
    // and registers page 5 -- a page boundary, so no tail of 203 is published
    // for turn 3 to hit.
    Submit(MakeSpecWithTokens("r2", MakeTurnTwoPrompt()));
    ExecutionPlan turn2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(turn2);
    ASSERT_NE(op2, nullptr);
    EXPECT_EQ(op2->extend_prefix_lens.at(0), 8) << "turn 2 hits r1's prompt + response pages";
    AdvanceOneRound("r2", 201);
    const auto r2_rows = AdvanceOneRound("r2", 202);
    ASSERT_EQ(r2_rows.at("full").size(), 6u);  // ceil(12/2)
    SendForwardDone("r2", {203});
    SendFinish("r2");
    PlanOnce();  // reap
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);

    // Turn 3 hit: cap = (16-1)/2 = 7; pages 0..5 registered (0..3 by r1, 4..5
    // by r2), page 6 never full in any request -> fixpoint 6 blocks = 12 hit
    // tokens, into r2's response (page 5).
    Submit(MakeSpecWithTokens("r3", MakeTurnThreePrompt()));

    ExecutionPlan turn3 = PlanOnce();
    const ForwardBatch* op3 = FindForwardBatch(turn3);
    ASSERT_NE(op3, nullptr);
    ASSERT_EQ(op3->request_ids.size(), 1u);
    EXPECT_EQ(op3->extend_prefix_lens.at(0), 12) << "hit grows across turns: 8 -> 12 tokens";
    EXPECT_EQ(op3->input_lengths.at(0), 4);
    EXPECT_EQ(op3->prefill_lengths.at(0), 16);
    EXPECT_EQ(op3->input_ids, (std::vector<std::int32_t>{203, 951, 952, 953}));
    // 6 claimed + 2 fresh + 1 preallocated decode page.
    EXPECT_EQ(op3->block_tables.at("full").at(0).size(), 9u);
    EXPECT_EQ(op3->block_tables.at("swa").at(0).size(), 9u);

    // Slots 0..3 are r1's blocks (re-freed cached by r2), 4..5 r2's own pages.
    const std::vector<std::int32_t> full_prefix(r2_rows.at("full").begin(), r2_rows.at("full").begin() + 6);
    const std::vector<std::int32_t> swa_prefix(r2_rows.at("swa").begin(), r2_rows.at("swa").begin() + 6);
    ExpectRowPrefixEq(op3->block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op3->block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: 6 claimed/group (12) + 2 fresh/group (4) + 1 reserved/group (2).
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 18);

    SendForwardDone("r3", {299});
    PlanOnce();
    SendForwardDone("r3", {300});
    SendFinish("r3");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "pool back to baseline after all three turns";
}

// A decode page registers before the admission slide, and a later
// ReclaimExpired punches it: the punch frees the block WITH its hash intact.
class DecodeCachingSmallWindowSuite : public DecodeCachingSuite {
protected:
    std::int32_t SlidingWindowTokens() const override { return 4; }
};

TEST_F(DecodeCachingSmallWindowSuite, SwaPunchedDecodePageStillHittable) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    // RunTurnOne's fill timing, inlined because the punch round +106 must land
    // BEFORE finish. W=4 slides on top (punched pages = (N-3)/2): +102 punches
    // slot 0, +103 registers page 2, +104 punches slot 1, +105 registers page 3.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    ExecutionPlan r1_prefill = PlanOnce();
    ASSERT_NE(FindForwardBatch(r1_prefill), nullptr);
    AdvanceOneRound("r1", 101);
    AdvanceOneRound("r1", 102);
    AdvanceOneRound("r1", 103);
    AdvanceOneRound("r1", 104);
    const auto r1_rows = AdvanceOneRound("r1", 105);
    ASSERT_EQ(r1_rows.at("swa").size(), 5u);
    EXPECT_EQ(r1_rows.at("swa")[0], 0);
    EXPECT_EQ(r1_rows.at("swa")[1], 0);
    ASSERT_GT(r1_rows.at("swa")[2], 0) << "page 2 is registered AND still live after the +105 round";
    ASSERT_GT(r1_rows.at("swa")[3], 0);

    // +106 -> N=9 -> first kept page 3: slot 2 (REGISTERED at +103) is punched;
    // its block reaches the free list with the hash intact.
    const auto punched = AdvanceOneRound("r1", 106);
    EXPECT_EQ(punched.at("swa")[2], 0) << "the registered decode page must be punched by now";
    SendForwardDone("r1", {107});  // that round's decode lands: the finish is quiescent
    SendFinish("r1");
    PlanOnce();  // reap
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);

    // r2: same 8-token prefix + 2 new. Fixpoint (W=4, needed 2): cap =
    // (10-1)/2 = 4, all four hashes cached (0,1,2 punched WITH hash); full
    // matches 4, swa bounded scan keeps 4 (2 holes) -> common 4 = 8 hit tokens.
    std::vector<std::int32_t> r2_tokens = MakeTurnTwoPrompt();
    Submit(MakeSpecWithTokens("r2", r2_tokens));
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/901));
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 6u);
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 6u);

    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    ExpectRowPrefixEq(op->block_tables.at("full").at(0), full_prefix, "full row");

    // Slot 2's expected id was captured at the +105 round, before the punch.
    const auto& swa_row = op->block_tables.at("swa").at(0);
    ASSERT_EQ(swa_row.size(), 6u);
    EXPECT_EQ(swa_row[0], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[1], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[2], r1_rows.at("swa")[2]) << "punched decode page claimed back by hash";
    EXPECT_EQ(swa_row[3], r1_rows.at("swa")[3]);
    EXPECT_GT(swa_row[4], 0);
    EXPECT_GT(swa_row[5], 0);

    // Pool: full claims 4 + swa claims 2 + 1 fresh/group + 1 reserved/group = 10.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 10);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

// Registration writes hashes only -- never refcounts.
TEST_F(DecodeCachingSuite, PoolBalanceAcrossDecodeCaching) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    RunTurnOne();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "turn 1: decode registration must not hold refs";

    Submit(MakeSpecWithTokens("r2", MakeTurnTwoPrompt()));
    ExecutionPlan turn2 = PlanOnce();
    ASSERT_NE(FindForwardBatch(turn2), nullptr);
    EXPECT_LT(scheduler_->AvailableLcmBlocks(), free_at_start) << "turn 2 holds claimed + fresh pages while live";
    AdvanceOneRound("r2", 201);
    AdvanceOneRound("r2", 202);
    SendForwardDone("r2", {203});
    SendFinish("r2");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "turn 2: claimed and fresh pages all return";

    Submit(MakeSpecWithTokens("r3", MakeTurnThreePrompt()));
    ExecutionPlan turn3 = PlanOnce();
    const ForwardBatch* op3 = FindForwardBatch(turn3);
    ASSERT_NE(op3, nullptr);
    EXPECT_EQ(op3->extend_prefix_lens.at(0), 12);
    SendForwardDone("r3", {299});
    PlanOnce();
    SendForwardDone("r3", {300});
    SendFinish("r3");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "baseline restored after the whole conversation";
}

// ---------------------------------------------------------------------------
// M15 streaming L2 sink: pages registered by a planning round batch into ONE
// D2H write-back; WriteBackDone commits the host index and unpins the source
// blocks (an ordinary store pins its Device sources until the ACK; only a
// retraction's snapshot store leaves them to the runtime's stream order).
// Byte movement itself is Phase D.
// ---------------------------------------------------------------------------
class StreamingSinkSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 7;  // 6 usable + the null placeholder (page 0, device convention)
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = false;
        cfg.disable_prefix_cache = true;

        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("swa", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/4),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Prefill -> finalize; the finalize round registers the prompt's page
    // hashes and streams every completed Full+SWA page.
    ExecutionPlan RunToFinalize(const RequestSpec& spec) {
        Submit(spec);
        PlanOnce();  // prefill
        SendForwardDone(spec.request_id, {9001});
        return PlanOnce();  // PrefillDone -> Decoding: registration + drain
    }

    ExecutionPlan FinishAndReap(const std::string& id) {
        SendForwardDone(id, {9002});
        SendFinish(id);
        return PlanOnce();  // leftover decode pages + latest snapshots, if any
    }

    static std::optional<WriteBackBatch> FindWriteBack(const ExecutionPlan& plan) {
        auto ops = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
        if (ops.empty()) {
            return std::nullopt;
        }
        EXPECT_EQ(ops.size(), 1u) << "the plan must carry at most one merged write-back list";
        return std::get<WriteBackBatch>(ops.front());
    }
};

TEST_F(StreamingSinkSuite, RegisteredPagesEmitWriteBackAndIndexOnDone) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    ExecutionPlan finalize = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value()) << "finalize must stream the 4 Full + 2 SWA completed pages";
    ASSERT_EQ(stream_wb->op_ids.size(), 1u);
    EXPECT_EQ(stream_wb->src_pages.at(0).size(), 6u);
    EXPECT_EQ(stream_wb->dst_pages.at(0).size(), 6u);
    EXPECT_EQ(stream_wb->source_pinned, std::vector<bool>{true}) << "an ordinary publication pins its sources";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0) << "nothing indexed until WriteBackDone";
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 0);

    ExecutionPlan finish = FinishAndReap("r1");
    EXPECT_FALSE(FindWriteBack(finish).has_value()) << "already-streamed prefill pages must not rewrite at finish";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 6)
        << "the six sources stay pinned until the ACK: the finished request's other pages return, these do not";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0);

    SendWriteBackDone(stream_wb->op_ids.at(0));
    PlanOnce();
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "the ACK returns the pinned sources";
}

TEST_F(StreamingSinkSuite, DuplicateRegistrationsAreDroppedAtDrain) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    ExecutionPlan finalize1 = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb1 = FindWriteBack(finalize1);
    ASSERT_TRUE(stream_wb1.has_value());
    EXPECT_FALSE(FindWriteBack(FinishAndReap("r1")).has_value());
    SendWriteBackDone(stream_wb1->op_ids.at(0));
    PlanOnce();
    ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);

    ExecutionPlan finalize2 = RunToFinalize(MakeRequestSpec("r2", /*num_pages=*/4));  // identical tokens
    EXPECT_FALSE(FindWriteBack(finalize2).has_value()) << "already-indexed keys must not re-emit a write-back";
    // r1's finish found the Host pool fully in flight, so its tail never
    // reached Host; r2's identical tail is the first store of that key and
    // replaces two committed, unpinned entries in the 6-page pool.
    ExecutionPlan finish2 = FinishAndReap("r2");
    auto finish_wb2 = FindWriteBack(finish2);
    ASSERT_TRUE(finish_wb2.has_value());
    EXPECT_EQ(finish_wb2->src_pages.at(0).size(), 2u) << "the tail with token 9001, in both groups";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 2)
        << "duplicate candidates are unpinned at drain; only the tail's sources are pinned";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 4);
    SendWriteBackDone(finish_wb2->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

TEST_F(StreamingSinkSuite, HostPoolExhaustionSkipsSilently) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    ExecutionPlan finalize1 = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb1 = FindWriteBack(finalize1);
    ASSERT_TRUE(stream_wb1.has_value());
    EXPECT_FALSE(FindWriteBack(FinishAndReap("r1")).has_value());
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 6) << "r1's six sources stay pinned in flight";
    ASSERT_EQ(scheduler_->HostPoolFreeBlocks(), 0) << "r1 holds all 6 host pages in flight";

    ExecutionPlan finalize2 = RunToFinalize(MakeRequestSpec("r2", /*num_pages=*/4, /*start=*/501));
    EXPECT_FALSE(FindWriteBack(finalize2).has_value())
        << "a fully-consumed host pool drops every candidate: no op at all";
    ExecutionPlan finish2 = FinishAndReap("r2");
    EXPECT_FALSE(FindWriteBack(finish2).has_value())
        << "finish-created candidates must also skip a fully-consumed host pool";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 6)
        << "dropped candidates pin nothing; only r1's in-flight sources are held";

    SendWriteBackDone(stream_wb1->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "everything balances after r1's commit";
}

TEST_F(StreamingSinkSuite, CommittedColdEntriesAreReplacedWhenHostPoolIsFull) {
    ExecutionPlan finalize1 = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb1 = FindWriteBack(finalize1);
    ASSERT_TRUE(stream_wb1.has_value());
    EXPECT_FALSE(FindWriteBack(FinishAndReap("r1")).has_value());
    SendWriteBackDone(stream_wb1->op_ids.at(0));
    ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    ASSERT_EQ(scheduler_->HostPoolFreeBlocks(), 0);

    ExecutionPlan finalize2 = RunToFinalize(MakeRequestSpec("r2", /*num_pages=*/4, /*start=*/501));
    auto stream_wb2 = FindWriteBack(finalize2);
    ASSERT_TRUE(stream_wb2.has_value()) << "committed, unpinned Host entries are replaceable";
    EXPECT_EQ(stream_wb2->src_pages.at(0).size(), 6u);
    EXPECT_FALSE(FindWriteBack(FinishAndReap("r2")).has_value());
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0) << "old keys are removed before replacement D2H";
    SendWriteBackDone(stream_wb2->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
}

TEST_F(StreamingSinkSuite, SameRoundDuplicateKeysDedupeAtDrain) {
    // Host pool with headroom (12 usable) so duplicates are dropped by the drain's batch
    // dedupe, NOT by pool exhaustion: two IDENTICAL prompts registering in one round drain
    // 12 candidates into 6 pairs.
    config_.host_allocator.total_pages = 13;
    config_.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    Submit(MakeRequestSpec("r2", /*num_pages=*/4));
    PlanOnce();  // both prefill (batch 2 <= max_batch_size 8, 16 tokens <= budget 64)
    SendForwardDone("r1", {9001});
    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();  // both register the same completed pages, one merged drain
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value());
    ASSERT_EQ(stream_wb->op_ids.size(), 1u);
    EXPECT_EQ(stream_wb->src_pages.at(0).size(), 6u) << "each Full+SWA key must be emitted at most once";
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 6) << "duplicates must not consume extra host pages";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 0);

    // Both prompts end with the same generated tokens, so their tails share
    // one hash: r1's finish stores it, r2's finds it in flight and drops it.
    // r2 finishes while its decode is still out, so it settles only when that
    // decode lands -- after r1's store was emitted.
    SendFinish("r2");
    auto tail_wb = FindWriteBack(FinishAndReap("r1"));
    ASSERT_TRUE(tail_wb.has_value());
    EXPECT_EQ(tail_wb->src_pages.at(0).size(), 2u);
    SendForwardDone("r2", {9002});
    EXPECT_FALSE(FindWriteBack(PlanOnce()).has_value()) << "same-round duplicates must be dropped";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 8)
        << "one pinned source per emitted key; the dropped duplicates pin nothing";

    SendWriteBackDone(stream_wb->op_ids.at(0));
    SendWriteBackDone(tail_wb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 8);
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 4) << "the eight cached host pages remain occupied";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "the ACKs return the pinned sources";
}

TEST_F(StreamingSinkSuite, MidDrainPoolFillEmitsPartialOp) {
    // 4 usable host pages against 6 candidates: the drain emits the 4 that fit and drops the
    // rest -- a partial op IS the contract when the pool fills mid-batch.
    config_.host_allocator.total_pages = 5;
    config_.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();

    ExecutionPlan finalize = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value());
    EXPECT_EQ(stream_wb->src_pages.at(0).size(), 4u) << "the drain emits the 4 Full pages that fit first";
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 0);

    ExecutionPlan finish = FinishAndReap("r1");
    EXPECT_FALSE(FindWriteBack(finish).has_value())
        << "in-flight Full pages still occupy the host pool, so leftover SWA must skip";
    SendWriteBackDone(stream_wb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 4);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "dropped candidates unpinned at drain";
}

TEST_F(StreamingSinkSuite, DuplicateWriteBackDoneIsIgnored) {
    ExecutionPlan finalize = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value());
    EXPECT_FALSE(FindWriteBack(FinishAndReap("r1")).has_value());
    const std::int32_t free_after_reap = scheduler_->AvailableLcmBlocks();

    SendWriteBackDone(stream_wb->op_ids.at(0));
    ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    const std::int32_t free_after_ack = scheduler_->AvailableLcmBlocks();
    EXPECT_EQ(free_after_ack, free_after_reap + 6) << "the ack publishes host entries and returns the six pins";

    // A replayed ack must be a no-op (the ledger already retired the op).
    SendWriteBackDone(stream_wb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_after_ack);
}

// ---------------------------------------------------------------------------
// M15 host-hit load-back: an admission whose device match ends inside the host
// index extends it from the host tier; the plan carries one H2D load-back and
// LoadBackDone releases the host load pins and the destination-page pins.
// ---------------------------------------------------------------------------
class HostHitSuite : public StreamingSinkSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = StreamingSinkSuite::MakeConfig();
        cfg.disable_prefix_cache = false;
        // 15 device pages -> 14 free (page 0 is null): a 5-page churn request's
        // peak (10 prefill + 2 reserve) takes 12 of them, and the startup bound
        // keeps one block per group for a tail handover; two churns in a row
        // recycle every one of r1's cached pages.
        cfg.device_allocator.total_pages = 15;
        cfg.host_allocator.total_pages =
            49;  // ample (+null page 0): r1's, both churns' and a filler's entries fit un-evicted
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    static std::optional<LoadBackBatch> FindLoadBack(const ExecutionPlan& plan) {
        auto ops = ExtractCacheOpsOfKind<LoadBackBatch>(plan);
        if (ops.empty()) {
            return std::nullopt;
        }
        EXPECT_EQ(ops.size(), 1u) << "the plan must carry at most one merged load-back list";
        return std::get<LoadBackBatch>(ops.front());
    }

    // Full sink lifecycle: completed Full+SWA pages stream at finalize; finish
    // emits leftover decode pages, the partial page at the computed frontier
    // (both groups) and latest snapshots.
    std::vector<WriteBackBatch> RunSinkLifecycle(const RequestSpec& spec) {
        ExecutionPlan finalize = RunToFinalize(spec);
        ExecutionPlan finish = FinishAndReap(spec.request_id);
        std::vector<WriteBackBatch> write_backs;
        if (auto wb = FindWriteBack(finalize)) {
            write_backs.push_back(*wb);
        }
        if (auto wb = FindWriteBack(finish)) {
            write_backs.push_back(*wb);
        }
        return write_backs;
    }

    // r1 (tokens 1..8, two generated) indexes 8 host entries (4 Full + 2 SWA
    // at finalize, the one-token tail in both groups at finish); two churn
    // requests then flood the free list so r1's pages survive ONLY on the
    // host tier (one churn evicts six of r1's eight; the second, finding the
    // pool full, evicts the oldest entries first: r1's last two).
    void SeedHostThenEvictDevice() {
        auto wb1 = RunSinkLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
        ASSERT_EQ(wb1.size(), 2u);
        for (const auto& wb : wb1) {
            SendWriteBackDone(wb.op_ids.at(0));
        }
        ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 8);
        // Host cache entries retain their host blocks until host eviction.
        ASSERT_EQ(scheduler_->HostPoolFreeBlocks(), 40);

        for (const auto& [id, start] : {std::pair{"churn", 501}, std::pair{"churn2", 801}}) {
            auto wb3 = RunSinkLifecycle(MakeRequestSpec(id, /*num_pages=*/5, start));
            ASSERT_EQ(wb3.size(), 2u);
            for (const auto& wb : wb3) {
                SendWriteBackDone(wb.op_ids.at(0));
            }
        }
        ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 26) << "per churn 5 Full + 2 SWA + its tail in both groups";
        ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 14) << "every seeding request fully retired";
    }
};

// A tail that survives only on Host is loaded straight into the hitter's
// private pages: one unkeyed load row per tail block, no Device entry for
// them, and the hitter resumes at the token the tail ends at.
TEST_F(HostHitSuite, AHostTailLoadsIntoPrivatePagesWithoutRepublication) {
    SeedHostThenEvictDevice();
    // r1's stream was [1..8, 9001, 9002]; its finish published the tail [9001]
    // at token 9. r2 repeats the nine tokens and diverges after them.
    Submit(RequestSpec{.request_id = "r2", .tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9001, 77, 78}});
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids, std::vector<std::string>{"r2"});
    EXPECT_EQ(op->extend_prefix_lens.at(0), 9) << "the Host tail is the hit boundary";
    EXPECT_EQ(op->input_lengths.at(0), 2);
    auto lb = FindLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    EXPECT_EQ(lb->src_pages.at(0).size(), 7u)
        << "4 full pages and the full tail block; the SWA lookback behind token 9 is page block [6,8) and the tail";
    EXPECT_TRUE(plan.pages_to_copy.at("full").empty()) << "a Host tail is loaded, not copied";
    const auto& full_row = op->block_tables.at("full").at(0);
    EXPECT_EQ(lb->dst_pages.at(0).at(4), full_row.at(4)) << "the tail block lands in the hitter's own slot 4";
    EXPECT_EQ(scheduler_->TailEntries(), 3u) << "r1's and the churns' tails stay enumerable while Host holds them";

    SendLoadBackDone(lb->op_ids.at(0));
    SendForwardDone("r2", {9003});
    const ExecutionPlan first_decode = PlanOnce();
    // r2's own prompt end publishes its tail at 11 and copies its partial
    // pages; the loaded tail block of token 9 was never republished, so the
    // page [9001 77] r2 completed is published under r2's own chain.
    auto stream_wb = FindWriteBack(first_decode);
    ASSERT_TRUE(stream_wb.has_value());
    EXPECT_EQ(plan.pages_to_copy.at("full").size(), 0u);
    EXPECT_EQ(first_decode.pages_to_copy.at("full").size(), 1u);
    SendWriteBackDone(stream_wb->op_ids.at(0));
    SendForwardDone("r2", {9004});
    SendFinish("r2");
    AckWriteBacks(PlanOnce());
    PlanOnce();
}

TEST_F(HostHitSuite, HostHitLoadsBackAfterDeviceEviction) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    ASSERT_EQ(free_at_start, 14);
    SeedHostThenEvictDevice();

    // r2 extends r1 by one page, so r1's 8-token endpoint is a matchable boundary:
    // full extension 4 blocks + SWA resume tail 2 blocks = 6 transfer pairs.
    Submit(MakeRequestSpec("r2", /*num_pages=*/5));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(lb->op_ids.size(), 1u);
    ASSERT_EQ(lb->src_pages.at(0).size(), 6u);
    ASSERT_EQ(lb->dst_pages.at(0).size(), 6u);

    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    // The input window skips the 8 host-hit tokens exactly as a device hit would.
    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->prefill_lengths.at(0), 10);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/9));
    EXPECT_EQ(op->block_tables.at("full").at(0).size(), 6u) << "4 extension + 2 fresh pages, all new to the table";
    EXPECT_EQ(op->block_tables.at("swa").at(0).size(), 6u);

    // Wire pairs are group-major: full ext slots 0..3, then SWA slots 2..3.
    const auto& full_row = op->block_tables.at("full").at(0);
    const auto& swa_row = op->block_tables.at("swa").at(0);
    ASSERT_EQ(full_row.size(), 6u);
    ASSERT_EQ(swa_row.size(), 6u);
    const auto& dst = lb->dst_pages.at(0);
    EXPECT_EQ(dst.at(0), full_row.at(0));
    EXPECT_EQ(dst.at(1), full_row.at(1));
    EXPECT_EQ(dst.at(2), full_row.at(2));
    EXPECT_EQ(dst.at(3), full_row.at(3));
    EXPECT_EQ(swa_row.at(0), 0) << "swa slot 0 is the pre-window hole";
    EXPECT_EQ(swa_row.at(1), 0) << "swa slot 1 is the pre-window hole";
    EXPECT_EQ(dst.at(4), swa_row.at(2));
    EXPECT_EQ(dst.at(5), swa_row.at(3));

    // The 6 matched host entries stay load-pinned until LoadBackDone retires the op.
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6);
    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);

    // r2 holds 6 loaded blocks and 4 fresh blocks.
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 10);

    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value()) << "r2's newly completed prefill pages must stream";
    SendWriteBackDone(stream_wb->op_ids.at(0));
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    AckWriteBacks(PlanOnce());
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "pool balances after the host-hit request";
}

TEST_F(HostHitSuite, EmptyHostIndexEmitsNoLoadBack) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    ExecutionPlan plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan), nullptr);
    EXPECT_FALSE(FindLoadBack(plan).has_value()) << "an empty host index must emit no load-back";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
}

TEST_F(HostHitSuite, AbandonedAdmissionUnpins) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    SeedHostThenEvictDevice();

    // Filler: 5 pages -> 10 prefill + 2 reserve = all but two parents while it decodes.
    Submit(MakeRequestSpec("filler", /*num_pages=*/5, /*start=*/701));
    PlanOnce();
    SendForwardDone("filler", {9001});
    ExecutionPlan filler_finalize = PlanOnce();  // finalization slides three expired SWA pages: free = 5
    auto filler_wb = FindWriteBack(filler_finalize);
    ASSERT_TRUE(filler_wb.has_value());
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 5);

    // r2's host match takes 6 pins, but the gate needs 4 + 6 ext > 5 free: the
    // abandoning return must give the pins back.
    Submit(MakeRequestSpec("r2", /*num_pages=*/5));
    ExecutionPlan blocked = PlanOnce();
    EXPECT_FALSE(FindLoadBack(blocked).has_value());
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0) << "an abandoned admission must unpin its host match";
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->GetHostCacheStats().prefix_hits.host_hit_tokens, 0u) << "a failed attempt counts no hit";

    // Free the filler (its write-back pins included): both decodes planned so
    // far land and it finishes -> r2 admits with the load-back.
    SendWriteBackDone(filler_wb->op_ids.at(0));
    SendForwardDone("filler", {9002});
    SendForwardDone("filler", {9003});
    SendFinish("filler");
    ExecutionPlan after_finish = PlanOnce();
    AckWriteBacks(after_finish);
    auto lb = FindLoadBack(after_finish);
    if (!lb.has_value()) {
        after_finish = PlanOnce();
        lb = FindLoadBack(after_finish);
    }
    ASSERT_TRUE(lb.has_value());
    EXPECT_EQ(lb->src_pages.at(0).size(), 6u);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    const PrefixHitStats admitted_hits = scheduler_->GetHostCacheStats().prefix_hits;
    EXPECT_EQ(admitted_hits.host_hit_tokens, 8u) << "r1's 8 tokens, counted once at the admission";
    EXPECT_EQ(admitted_hits.device_hit_tokens, 0u);

    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value()) << "r2's newly completed prefill pages must stream";
    SendWriteBackDone(stream_wb->op_ids.at(0));
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    AckWriteBacks(PlanOnce());
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "pool balances after the deferred host hit";
}

// A Host win cuts the Device back to its full pages: the Device's tail gives
// way, so the Host loads more than it adds over the Device's own boundary.
TEST_F(SchedulerTestSuite, AHostWinOverADeviceTailLoadsMoreThanItGains) {
    // r0 = [1..8] leaves its four pages on Host alone.
    Submit(MakeRequestSpec("r0", /*num_pages=*/4));
    AckWriteBacks(PlanOnce());
    SendForwardDone("r0", {42});
    SendFinish("r0");
    AckWriteBacks(PlanOnce());
    ASSERT_TRUE(scheduler_->ClearL1Cache());
    // r1 = [1..5] loads pages 0 and 1 from Host and republishes them on the
    // Device; its finish publishes the tail [5] at token 5 there.
    Submit(RequestSpec{.request_id = "r1", .tokens = MakeTokens(5)});
    const ExecutionPlan r1_admitted = PlanOnce();
    for (const CacheOperation& op : ExtractCacheOpsOfKind<LoadBackBatch>(r1_admitted)) {
        SendLoadBackDone(std::get<LoadBackBatch>(op).op_ids.at(0));
    }
    SendForwardDone("r1", {43});
    SendFinish("r1");
    AckWriteBacks(PlanOnce());
    const HostCacheStats before = scheduler_->GetHostCacheStats();

    // r2 = [1..8, 77, 78]: the Device reaches 5 (pages 0 and 1, then the
    // tail), the Host 8 from the Device's four full-page tokens.
    std::vector<std::int32_t> tokens = MakeTokens(8);
    tokens.insert(tokens.end(), {77, 78});
    Submit(RequestSpec{.request_id = "r2", .tokens = tokens});
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"r2"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{8});
    const HostCacheStats after = scheduler_->GetHostCacheStats();
    EXPECT_EQ(after.prefix_loads.ops - before.prefix_loads.ops, 1u);
    EXPECT_EQ(after.prefix_hits.device_hit_tokens - before.prefix_hits.device_hit_tokens, 4u)
        << "the Device's full pages";
    EXPECT_EQ(after.prefix_hits.host_hit_tokens - before.prefix_hits.host_hit_tokens, 4u)
        << "the Host loads the boundary less the Device's full pages";
    EXPECT_EQ(after.prefix_hits.host_gain_tokens - before.prefix_hits.host_gain_tokens, 3u)
        << "and adds the boundary less the Device's own boundary at 5";
    EXPECT_EQ(after.prefix_loads.ops, 2u) << "r1's load and r2's";
    EXPECT_EQ(after.prefix_loads.blocks, (std::vector<std::uint64_t>{4})) << "r1's two pages and r2's two";
}

// A Submitted request that waits for a slot counts its prompt and its hit
// once, when it is admitted.
TEST_F(SchedulerTestSuite, AFirstAdmissionWaitingForASlotCountsItsHitOnce) {
    config_.max_batch_size = 1;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    Submit(MakeRequestSpec("r0", /*num_pages=*/2));
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"r0"});
    SendForwardDone("r0", {42});
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"r0"});
    EXPECT_EQ(scheduler_->GetHostCacheStats().prefix_hits.prompt_tokens, 4u) << "r0 holds the one slot";

    SendForwardDone("r0", {43});
    SendFinish("r0");
    const ExecutionPlan admitted = PlanOnce();
    ASSERT_NE(FindForwardBatch(admitted), nullptr);
    ASSERT_EQ(FindForwardBatch(admitted)->request_ids, std::vector<std::string>{"r1"});
    const PrefixHitStats hits = scheduler_->GetHostCacheStats().prefix_hits;
    EXPECT_EQ(hits.device_hit_tokens, 2u) << "r0's first page; the probe stops before r1's last token";
    EXPECT_EQ(hits.prompt_tokens, 4u + 4u);
}

TEST_F(HostHitSuite, AbortDuringLoadKeepsPagesPinned) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    SeedHostThenEvictDevice();

    Submit(MakeRequestSpec("r2", /*num_pages=*/5));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(lb->dst_pages.at(0).size(), 6u);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 10);

    // Abort while the H2D copy is in flight: the reap returns only the 4 fresh pages;
    // the 6 load destinations must stay off the free list until LoadBackDone.
    SendAbort(*scheduler_, "r2");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 6)
        << "in-flight load destinations must not be reusable";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6) << "the host sources stay pinned too";

    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "LoadBackDone releases the destinations";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
}

// An abort-during-load leaves pages owned only by the load ledger. Capacity
// handling must wait for LoadBackDone rather than retract or abort a request
// blocked on those pages.
TEST_F(HostHitSuite, CapacityBlockWaitsForInFlightLoads) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    SeedHostThenEvictDevice();

    // Same shape as AbortDuringLoadKeepsPagesPinned: 6 destinations stay ticket-held.
    Submit(MakeRequestSpec("r2", /*num_pages=*/5));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    SendAbort(*scheduler_, "r2");
    PlanOnce();  // reap
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 6);

    // r3 (fresh tokens, no host hit) charges 8 prefill + 2 reserve = 10 > 6 free: deferred.
    Submit(MakeRequestSpec("r3", /*num_pages=*/4, /*start=*/901));
    ExecutionPlan blocked1 = PlanOnce();
    ASSERT_NE(FindForwardBatch(blocked1), nullptr);
    EXPECT_TRUE(FindForwardBatch(blocked1)->request_ids.empty());
    // A second blocked round is also quiet while the load ledger owns pages.
    ExecutionPlan blocked2 = PlanOnce();
    ASSERT_NE(FindForwardBatch(blocked2), nullptr);
    EXPECT_TRUE(FindForwardBatch(blocked2)->request_ids.empty());
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "deferred r3 stays intact in the waiting set";

    // LoadBackDone frees the 6 destinations: r3's 10-block gate now clears.
    SendLoadBackDone(lb->op_ids.at(0));
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
    ExecutionPlan admitted = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(admitted);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    EXPECT_EQ(op->request_ids.at(0), "r3");
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
}

// A LoadBackDone whose op_id was already retired must hit the silent-ignore arm:
// no crash, no double UnpinLoad, no double-free of the destination pages.
TEST_F(HostHitSuite, DuplicateLoadBackDoneIsIgnored) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    SeedHostThenEvictDevice();

    Submit(MakeRequestSpec("r2", /*num_pages=*/5));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 10);

    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
    const std::int32_t free_after_first = scheduler_->AvailableLcmBlocks();
    EXPECT_EQ(free_after_first, free_at_start - 10) << "destinations still table-held: no free-list change";

    SendLoadBackDone(lb->op_ids.at(0));  // duplicate
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_after_first) << "a duplicate Done must not double-free";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);

    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();
    auto stream_wb = FindWriteBack(finalize);
    ASSERT_TRUE(stream_wb.has_value()) << "r2's newly completed prefill pages must stream";
    SendWriteBackDone(stream_wb->op_ids.at(0));
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    AckWriteBacks(PlanOnce());
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "pool balances despite the duplicate event";
}

// ---------------------------------------------------------------------------
// M15 host hit + chunked prefill: later chunks must count the host extension as
// computed tokens, and an SWA slide that punches a still-loading destination
// page must leave it ticket-protected until LoadBackDone.
// ---------------------------------------------------------------------------
// A Host tail whose blocks a sliding group only partly needs (W=2 over
// two-token blocks needs the one block holding the row before T; a
// three-token tail spans two): the block below the window is a hole the
// request never writes, so it gets no page and the plan counts none.
class HostTailBelowWindowSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 4;
        cfg.device_allocator.total_pages = 13;  // twelve usable
        cfg.host_allocator.total_pages = 33;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = false;
        cfg.disable_prefix_cache = false;
        cfg.cache_groups = {
            MakeGroup("full", 4, cfg.device_allocator.total_pages, AttnKind::kFull),
            MakeGroup("swa", 2, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/2),
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(HostTailBelowWindowSuite, ATailBlockBelowTheWindowStaysAHoleAndCostsNothing) {
    // r1 = [1..7] + one decode: finish publishes page 0 and the tail at 7 --
    // the full group's block [4, 7), the sliding group's window block [6, 7)
    // -- and streams them to Host.
    Submit(RequestSpec{.request_id = "r1", .tokens = MakeTokens(7), .max_new_tokens = 2});
    AckWriteBacks(PlanOnce());
    SendForwardDone("r1", {101});
    AckWriteBacks(PlanOnce());
    SendForwardDone("r1", {102});
    SendFinish("r1");
    AckWriteBacks(PlanOnce());
    ASSERT_TRUE(scheduler_->ClearL1Cache()) << "the Host tier alone holds r1";
    ASSERT_EQ(scheduler_->TailEntries(), 1u);

    // A filler holds seven of the twelve blocks: 3 full (seven tokens, the
    // decode slot and the two-token admission headroom) + 4 sliding.
    Submit(RequestSpec{.request_id = "filler", .tokens = MakeTokens(7, 501), .max_new_tokens = 2});
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 5);

    // r2 = [1..7, 77]: the Host tail at 7 is the hit. The full group loads
    // page 0 and the tail block (2) and grows by one (3); the sliding group
    // loads the window block [6, 7) (1) and grows by one (2): five blocks,
    // exactly what is free, with the tail block [4, 6) below the window left
    // a hole.
    Submit(RequestSpec{.request_id = "r2",
                       .tokens =
                           [] {
                               std::vector<std::int32_t> tokens = MakeTokens(7);
                               tokens.push_back(77);
                               return tokens;
                           }(),
                       .max_new_tokens = 1});
    const ExecutionPlan admitted = PlanOnce();
    const ForwardBatch* r2 = FindForwardBatch(admitted);
    ASSERT_NE(r2, nullptr);
    ASSERT_EQ(r2->request_ids, std::vector<std::string>{"r2"});
    EXPECT_EQ(r2->extend_prefix_lens, std::vector<std::int32_t>{7});
    const auto loads = ExtractCacheOpsOfKind<LoadBackBatch>(admitted);
    ASSERT_EQ(loads.size(), 1u);
    EXPECT_EQ(std::get<LoadBackBatch>(loads.front()).src_pages.at(0).size(), 3u)
        << "page 0 and the tail block of the full group; the window block [6, 7) of the sliding group";
    const auto& swa = r2->block_tables.at("swa").at(0);
    ASSERT_GE(swa.size(), 4u);
    EXPECT_EQ(swa.at(0), 0);
    EXPECT_EQ(swa.at(1), 0);
    EXPECT_EQ(swa.at(2), 0) << "the tail block [4, 6) lies below the window: a hole, not a fresh page";
    EXPECT_GT(swa.at(3), 0) << "the loaded window block";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0);
}

class ChunkedHostHitSuite : public HostHitSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = HostHitSuite::MakeConfig();
        cfg.max_scheduled_tokens = 4;  // 4-token prefill chunks
        // 21 -> 20 free: r2's first chunk holds 10 (6 ext + 4 fresh) and its second
        // chunk charges 6 with zero slide credit (ticket-held punches don't count).
        cfg.device_allocator.total_pages = 21;
        // 36 usable: both seeding requests' pages and tails stay Host-resident
        // (32 entries) with room for r2's own streams.
        cfg.host_allocator.total_pages = 37;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        for (auto& g : cfg.cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Chunked twin of RunSinkLifecycle: ACK prefill Full+SWA streams round by round,
    // then ACK any leftover finish write-back.
    void RunChunkedSinkLifecycle(const RequestSpec& spec, std::int32_t prefill_rounds) {
        Submit(spec);
        for (std::int32_t i = 0; i < prefill_rounds; ++i) {
            ExecutionPlan plan = PlanOnce();
            ASSERT_NE(FindForwardBatch(plan), nullptr) << "chunk " << i;
            ASSERT_EQ(FindForwardBatch(plan)->request_ids.size(), 1u) << "chunk " << i << " must be admitted";
            AckWriteBacks(plan);
            if (i + 1 < prefill_rounds) {
                SendForwardDone(spec.request_id);  // an intermediate chunk lands: KV only
            }
        }
        SendForwardDone(spec.request_id, {9001});
        AckWriteBacks(PlanOnce());  // finalize: registration + drain
        AckWriteBacks(FinishAndReap(spec.request_id));
    }

    // r1 (4 pages) indexes 8 host entries over 2 chunks plus its tail in both
    // groups at finish; the churn request must pop 22 free-list entries (full
    // 10 + swa 10 + reserve 2) = the 10 fresh blocks still unused plus ALL 12
    // of r1's cached blocks, so r1 survives host-only.
    void SeedHostThenEvictDeviceChunked() {
        RunChunkedSinkLifecycle(MakeRequestSpec("r1", /*num_pages=*/4), /*prefill_rounds=*/2);
        ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 10);
        RunChunkedSinkLifecycle(MakeRequestSpec("churn", /*num_pages=*/10, /*start=*/501), /*prefill_rounds=*/5);
        ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 32);
        ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 20) << "both seeding requests fully retired";
    }
};

TEST_F(ChunkedHostHitSuite, ChunkedPrefillAfterHostHit) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    ASSERT_EQ(free_at_start, 20);
    SeedHostThenEvictDeviceChunked();

    // r2: 16 tokens sharing r1's first 8. Host extension = 4 blocks (full run 0..3;
    // swa tail ceil((W-1)/P)=2 at the boundary) -> real pages full 4 + swa 2 = 6.
    Submit(MakeRequestSpec("r2", /*num_pages=*/8));
    ExecutionPlan c1 = PlanOnce();
    auto lb = FindLoadBack(c1);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(lb->src_pages.at(0).size(), 6u);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6);

    const ForwardBatch* op1 = FindForwardBatch(c1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 1u);
    // First chunk: the 8 host-hit tokens are computed; chunk covers tokens [8,12).
    EXPECT_EQ(op1->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op1->input_lengths.at(0), 4);
    EXPECT_EQ(op1->prefill_lengths.at(0), 16);
    // 6 ext + 2 fresh/group: 10 blocks held.
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 10);
    SendForwardDone("r2");

    // Chunk 2 completes prefill; its slide at num_computed=12 punches swa ext slots
    // 2,3 = LOADED destinations mid-copy. The ticket must keep them off the free list.
    ExecutionPlan c2 = PlanOnce();
    const ForwardBatch* op2 = FindForwardBatch(c2);
    ASSERT_NE(op2, nullptr);
    ASSERT_EQ(op2->request_ids.size(), 1u);
    EXPECT_EQ(op2->extend_prefix_lens.at(0), 12) << "chunk 2 must see ext(8) + chunk1(4) as computed";
    EXPECT_EQ(op2->input_lengths.at(0), 4);
    AckWriteBacks(c2);  // pages 4,5 registered this round; ack so only the ticket pins remain
    // Four input tokens plus one decode-reserve token require 3 new pages per
    // group; the 2 punched load destinations stay ticket-held.
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 16);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 6) << "the copy is still in flight";

    // LoadBackDone releases exactly the 2 punched destinations (the other 4 stay table-held).
    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start - 14);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);

    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();
    ASSERT_NE(FindForwardBatch(finalize), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    AckWriteBacks(finalize);
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    AckWriteBacks(PlanOnce());  // any remaining decode write-back + reap
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start) << "pool balances after the chunked host hit";
}

class BoundedReplaySuite : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kReplayWindow = 16;

    virtual std::int32_t MaxScheduledTokens() const { return 64; }
    virtual std::int32_t MaxBatchSize() const { return 8; }
    virtual bool MixedPrefillDecode() const { return false; }
    virtual std::int32_t PrefixReplayTokens() const { return 0; }

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 8;
        cfg.device_allocator.total_pages = 256;
        cfg.host_allocator.total_pages = 256;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = MaxScheduledTokens();
        cfg.max_batch_size = MaxBatchSize();
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.enable_mixed_prefill_decode = MixedPrefillDecode();
        cfg.prefix_replay_tokens = PrefixReplayTokens();

        CacheGroupConfig swa =
            MakeGroup("swa", /*block_granularity=*/4, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/kReplayWindow);
        swa.replayable = true;
        CacheGroupConfig tail =
            MakeGroup("tail", /*block_granularity=*/2, cfg.device_allocator.total_pages, AttnKind::kSlidingWindow,
                      /*sliding_window_tokens=*/2);
        tail.replayable = true;
        cfg.cache_groups = {
            MakeGroup("full", cfg.prefix_granularity, cfg.device_allocator.total_pages, AttnKind::kFull),
            swa,
            tail,
        };
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    RequestSpec MakeSpecWithTokens(const std::string& id, std::vector<std::int32_t> tokens) {
        return RequestSpec{.request_id = id, .tokens = std::move(tokens)};
    }

    // Prefill (one chunk) -> one decode round -> finish; returns the prefill
    // op's per-group rows. The decode round publishes the page hashes.
    std::map<std::string, std::vector<std::int32_t>> RunLifecycle(const RequestSpec& spec) {
        Submit(spec);
        const ExecutionPlan prefill = PlanOnce();
        const ForwardBatch* op = FindForwardBatch(prefill);
        EXPECT_NE(op, nullptr);
        std::map<std::string, std::vector<std::int32_t>> rows;
        if (op != nullptr) {
            EXPECT_EQ(op->extend_replay_lens.at(0), 0) << "a cold prompt re-feeds nothing";
            for (const auto& [gid, table] : op->block_tables) {
                rows[gid] = table.at(0);
            }
        }
        SendForwardDone(spec.request_id, {9001});
        PlanOnce();
        SendForwardDone(spec.request_id, {9002});
        SendFinish(spec.request_id);
        PlanOnce();
        return rows;
    }

    static std::vector<std::int32_t> Slice(const std::vector<std::int32_t>& tokens, std::int32_t begin,
                                           std::int32_t end) {
        return {tokens.begin() + begin, tokens.begin() + end};
    }

    static void ExpectHolesThenPages(const std::vector<std::int32_t>& row, std::int32_t first_page,
                                     std::int32_t min_pages, const char* what) {
        ASSERT_GE(static_cast<std::int32_t>(row.size()), min_pages) << what;
        for (std::int32_t slot = 0; slot < first_page; ++slot) {
            EXPECT_EQ(row[static_cast<std::size_t>(slot)], 0) << what << " slot " << slot << " must be a hole";
        }
        for (std::int32_t slot = first_page; slot < min_pages; ++slot) {
            EXPECT_GT(row[static_cast<std::size_t>(slot)], 0) << what << " slot " << slot << " must be a page";
        }
    }
};

TEST_F(BoundedReplaySuite, FirstChunkAfterHitReplaysWindow) {
    const std::int32_t free_at_start = scheduler_->AvailableLcmBlocks();
    const RequestSpec r1 = MakeRequestSpec("r1", /*num_pages=*/4);  // 32 tokens
    const auto r1_rows = RunLifecycle(r1);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);

    // r2 = r1's 32 tokens + 8 new: the closed group hits P=32 on its own; the
    // replayable groups claim nothing and the window [16, 32) is re-fed.
    std::vector<std::int32_t> tokens = r1.tokens;
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/8, /*start=*/901);
    tokens.insert(tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", tokens));

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 16);
    EXPECT_EQ(op->extend_replay_lens.at(0), kReplayWindow);
    EXPECT_EQ(op->input_lengths.at(0), kReplayWindow + 8);
    EXPECT_EQ(op->prefill_lengths.at(0), 40);
    EXPECT_EQ(op->input_ids, Slice(tokens, 16, 40));

    // Closed group: r1's four pages are shared, then fresh pages follow.
    const auto& full_row = op->block_tables.at("full").at(0);
    ASSERT_GE(full_row.size(), 5u);
    for (std::size_t slot = 0; slot < 4; ++slot) {
        EXPECT_EQ(full_row[slot], r1_rows.at("full").at(slot)) << "full slot " << slot;
    }
    EXPECT_GT(full_row[4], 0);
    // Replayable groups: holes below s=16, private pages from there through
    // the 40 computed tokens plus the decode slot.
    ExpectHolesThenPages(op->block_tables.at("swa").at(0), /*first_page=*/16 / 4, /*min_pages=*/41 / 4 + 1, "swa");
    ExpectHolesThenPages(op->block_tables.at("tail").at(0), /*first_page=*/16 / 2, /*min_pages=*/41 / 2 + 1, "tail");

    // Progress is the chunk, not the replay: r2 decodes from position 40 and
    // frees everything on finish.
    SendForwardDone("r2", {199});
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), free_at_start);
}

TEST_F(BoundedReplaySuite, HitShorterThanWindowReplaysWholePrefix) {
    const RequestSpec r1 = MakeRequestSpec("r1", /*num_pages=*/1);  // 8 tokens
    RunLifecycle(r1);

    std::vector<std::int32_t> tokens = r1.tokens;
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/8, /*start=*/901);
    tokens.insert(tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", tokens));

    const ExecutionPlan op_plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(op_plan);
    ASSERT_NE(op, nullptr);
    // P=8 < W=16: the whole hit prefix is re-fed and the forward starts at 0.
    EXPECT_EQ(op->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(op->extend_replay_lens.at(0), 8);
    EXPECT_EQ(op->input_lengths.at(0), 16);
    EXPECT_EQ(op->input_ids, tokens);
    EXPECT_GT(op->block_tables.at("swa").at(0).at(0), 0) << "no hole: the private suffix starts at 0";
}

TEST_F(BoundedReplaySuite, NoHitNoReplay) {
    Submit(MakeSpecWithTokens("r1", MakeTokens(/*count=*/12)));
    const ExecutionPlan op_plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(op_plan);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(op->extend_replay_lens.at(0), 0);
    EXPECT_EQ(op->input_lengths.at(0), 12);
}

TEST_F(BoundedReplaySuite, FinalChunkKeepsAtLeastTheWindow) {
    // 70 tokens with a 64-token budget would leave a 6-token final chunk; the
    // first chunk is shortened so the final chunk holds exactly the window.
    const std::vector<std::int32_t> tokens = MakeTokens(/*count=*/70);
    Submit(MakeSpecWithTokens("r1", tokens));

    const ExecutionPlan chunk1_plan = PlanOnce();
    const ForwardBatch* chunk1 = FindForwardBatch(chunk1_plan);
    ASSERT_NE(chunk1, nullptr);
    EXPECT_EQ(chunk1->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(chunk1->extend_replay_lens.at(0), 0);
    EXPECT_EQ(chunk1->input_lengths.at(0), 70 - kReplayWindow);

    const ExecutionPlan chunk2_plan = PlanOnce();
    const ForwardBatch* chunk2 = FindForwardBatch(chunk2_plan);
    ASSERT_NE(chunk2, nullptr);
    EXPECT_EQ(chunk2->extend_prefix_lens.at(0), 70 - kReplayWindow);
    EXPECT_EQ(chunk2->extend_replay_lens.at(0), 0);
    EXPECT_EQ(chunk2->input_lengths.at(0), kReplayWindow);
    EXPECT_EQ(chunk2->input_ids, Slice(tokens, 54, 70));

    SendForwardDone("r1", {});
    SendForwardDone("r1", {9001});
    const ExecutionPlan decode_plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode_plan), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u) << "decode resumes at position 70";
}

TEST_F(BoundedReplaySuite, FinalChunkAlreadyAtLeastTheWindowIsUnchanged) {
    Submit(MakeSpecWithTokens("r1", MakeTokens(/*count=*/84)));
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    const ExecutionPlan chunk2_plan = PlanOnce();
    const ForwardBatch* chunk2 = FindForwardBatch(chunk2_plan);
    ASSERT_NE(chunk2, nullptr);
    EXPECT_EQ(chunk2->extend_prefix_lens.at(0), 64);
    EXPECT_EQ(chunk2->extend_replay_lens.at(0), 0);
    EXPECT_EQ(chunk2->input_lengths.at(0), 20);
}

TEST_F(BoundedReplaySuite, ReplayRowsDebitTokenBudget) {
    const RequestSpec r1 = MakeRequestSpec("r1", /*num_pages=*/4);
    RunLifecycle(r1);

    std::vector<std::int32_t> tokens = r1.tokens;
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/8, /*start=*/901);
    tokens.insert(tokens.end(), tail.begin(), tail.end());
    // r2 costs 16 replay + 8 new rows; r3's first chunk gets the remaining 40.
    Submit({MakeSpecWithTokens("r2", tokens), MakeSpecWithTokens("r3", MakeTokens(/*count=*/60, /*start=*/5001))});

    const ExecutionPlan op_plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(op_plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids, (std::vector<std::string>{"r2", "r3"}));
    EXPECT_EQ(op->extend_replay_lens, (std::vector<std::int32_t>{kReplayWindow, 0}));
    EXPECT_EQ(op->input_lengths, (std::vector<std::int32_t>{kReplayWindow + 8, 64 - kReplayWindow - 8}));
}

// r1's 60 tokens leave r2 a budget of 4: its 10-token prompt would end in a
// 6-token chunk, shorter than the window, and no earlier cut leaves the
// window whole. r2 waits a round, and its prompt counts once, when it is
// admitted the next round.
TEST_F(BoundedReplaySuite, AFirstChunkWithNoRoomForTheFinalWindowWaitsARound) {
    Submit({MakeSpecWithTokens("r1", MakeTokens(/*count=*/60)),
            MakeSpecWithTokens("r2", MakeTokens(/*count=*/10, /*start=*/5001))});
    const ExecutionPlan first = PlanOnce();
    ASSERT_NE(FindForwardBatch(first), nullptr);
    EXPECT_EQ(FindForwardBatch(first)->request_ids, std::vector<std::string>{"r1"});
    EXPECT_EQ(scheduler_->GetHostCacheStats().prefix_hits.prompt_tokens, 60u) << "r1's alone";

    SendForwardDone("r1", {9001});
    const ExecutionPlan second = PlanOnce();
    ASSERT_NE(FindForwardBatch(second), nullptr);
    EXPECT_EQ(FindForwardBatch(second)->request_ids, std::vector<std::string>{"r2"});
    EXPECT_EQ(FindForwardBatch(second)->input_lengths, std::vector<std::int32_t>{10});
    EXPECT_EQ(scheduler_->GetHostCacheStats().prefix_hits.prompt_tokens, 60u + 10u);
}

// The DSpark cap (prefix_replay_tokens) shortens the probe first; bounded
// replay then re-feeds its window before whatever hit remains.
class BoundedReplayWithDSparkCapSuite : public BoundedReplaySuite {
protected:
    std::int32_t PrefixReplayTokens() const override { return 8; }
};

TEST_F(BoundedReplayWithDSparkCapSuite, DSparkCapComposesWithBoundedReplay) {
    const RequestSpec r1 = MakeRequestSpec("r1", /*num_pages=*/4);  // 32 tokens
    RunLifecycle(r1);

    std::vector<std::int32_t> tokens = r1.tokens;
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/4, /*start=*/901);
    tokens.insert(tokens.end(), tail.begin(), tail.end());  // 36 tokens
    Submit(MakeSpecWithTokens("r2", tokens));

    const ExecutionPlan op_plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(op_plan);
    ASSERT_NE(op, nullptr);
    // Cap: (36 - 8) / 8 = 3 pages -> P = 24; window 16 -> s = 8.
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->extend_replay_lens.at(0), kReplayWindow);
    EXPECT_EQ(op->input_lengths.at(0), 36 - 8);
    EXPECT_EQ(op->input_ids, Slice(tokens, 8, 36));
}

// Mixed mode: the decode batch leaves two replay windows for a pending local
// prefill, so a hit chunk always fits in one forward.
class BoundedReplayMixedSuite : public BoundedReplaySuite {
protected:
    std::int32_t MaxScheduledTokens() const override { return 2 * kReplayWindow + 6; }
    std::int32_t MaxBatchSize() const override { return 16; }
    bool MixedPrefillDecode() const override { return true; }
};

TEST_F(BoundedReplayMixedSuite, DecodeBatchLeavesRoomForTheReplayWindow) {
    // Seed the cache with a 24-token prompt (one chunk under the 38 budget).
    const RequestSpec r1 = MakeRequestSpec("r1", /*num_pages=*/3);
    RunLifecycle(r1);

    // Eight decoding requests, each a 4-token prompt.
    std::vector<std::string> decoders;
    for (int i = 0; i < 8; ++i) {
        const std::string id = "d" + std::to_string(i);
        Submit(MakeSpecWithTokens(id, MakeTokens(/*count=*/4, /*start=*/2000 + 10 * i)));
        ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
        SendForwardDone(id, {7000 + i});
        decoders.push_back(id);
    }
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    for (const std::string& id : decoders) {
        SendForwardDone(id, {8000});
    }
    EXPECT_EQ(scheduler_->DecodingSize(), 8u);

    // r2 hits P=24 (> W) and adds 15 new tokens: fewer than a window, so the
    // whole W + 15 = 31 rows must go in one chunk. With a 38-token budget the
    // decode batch must stop once 2W = 32 tokens remain, i.e. after 6 rows.
    std::vector<std::int32_t> tokens = r1.tokens;
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/15, /*start=*/901);
    tokens.insert(tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", tokens));
    const ExecutionPlan op_plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(op_plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->NumExtends(), 1);
    EXPECT_EQ(op->request_ids.at(0), "r2");
    EXPECT_EQ(op->extend_prefix_lens.at(0), 24 - kReplayWindow);
    EXPECT_EQ(op->extend_replay_lens.at(0), kReplayWindow);
    EXPECT_EQ(op->input_lengths.at(0), kReplayWindow + 15);
    EXPECT_EQ(op->request_ids.size(), 1u + 6u);
}

// PD: a replayable group travels like any sliding window -- the prefill role
// replays locally on its own hits and transfers the retained tail; the decode
// role lands that tail and never re-feeds anything.
class BoundedReplayPrefillRoleSuite : public BoundedReplaySuite {
protected:
    virtual Role RoleUnderTest() const { return Role::kP; }

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = BoundedReplaySuite::MakeConfig();
        cfg.role = RoleUnderTest();
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.transfer_policy = CacheTransferPolicy::FullSuffix;
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{request_id});
        scheduler_->Advance(std::move(event));
    }
};

TEST_F(BoundedReplayPrefillRoleSuite, LocalHitReplaysAndTheTailIsWhatTransfers) {
    // r1: a 32-token prompt prefilled in one chunk; its ExtendResult lands
    // the bootstrap token and the closed group's four pages are published.
    const RequestSpec r1 = MakeRequestSpec("r1", /*num_pages=*/4);
    Submit(r1);
    SendBootstrapped("r1");
    const ExecutionPlan first_plan = PlanOnce();
    const ForwardBatch* first = FindForwardBatch(first_plan);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->extend_replay_lens, std::vector<std::int32_t>{0});
    SendForwardDone("r1", {42});
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());

    // r2 = r1's 32 tokens + 8 new: the closed group hits P=32 and the whole
    // swa retention window [16, 32) is re-fed, so the tail a decode peer lands
    // (`full_suffix`: from token 40 - 16 + 1 = 25, page 6) is materialized.
    std::vector<std::int32_t> tokens = r1.tokens;
    const std::vector<std::int32_t> tail = MakeTokens(/*count=*/8, /*start=*/901);
    tokens.insert(tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", tokens));
    SendBootstrapped("r2");
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids, std::vector<std::string>{"r2"});
    EXPECT_EQ(op->extend_prefix_lens.at(0), 16);
    EXPECT_EQ(op->extend_replay_lens.at(0), kReplayWindow);
    EXPECT_EQ(op->input_lengths.at(0), kReplayWindow + 8);
    EXPECT_EQ(op->input_ids, Slice(tokens, 16, 40));
    ExpectHolesThenPages(op->block_tables.at("swa").at(0), /*first_page=*/16 / 4, /*min_pages=*/40 / 4, "swa");
    ExpectHolesThenPages(op->block_tables.at("tail").at(0), /*first_page=*/16 / 2, /*min_pages=*/40 / 2, "tail");
}

class BoundedReplayDecodeRoleSuite : public BoundedReplayPrefillRoleSuite {
protected:
    Role RoleUnderTest() const override { return Role::kD; }
};

TEST_F(BoundedReplayDecodeRoleSuite, RemoteAdmissionLandsTheRetainedTailAndReplaysNothing) {
    const RequestSpec spec = MakeRequestSpec("r1", /*num_pages=*/5);  // 40 tokens
    Submit(spec);
    SendBootstrapped("r1");
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* admission = FindRemoteAdmission(plan);
    ASSERT_NE(admission, nullptr);
    EXPECT_EQ(admission->extend_prefix_lens, std::vector<std::int32_t>{0});
    EXPECT_EQ(admission->extend_replay_lens, std::vector<std::int32_t>{0});
    EXPECT_EQ(admission->input_lengths, std::vector<std::int32_t>{40});
    // The closed group lands the whole prompt; the replayable groups land
    // exactly the tail their retention keeps (swa window 16 -> from token 25,
    // page 6; tail window 2 -> from token 39, page 19), as any sliding group.
    const auto& full_row = admission->block_tables.at("full").at(0);
    ASSERT_GE(full_row.size(), 5u);
    for (std::size_t slot = 0; slot < 5; ++slot) {
        EXPECT_GT(full_row[slot], 0) << "full slot " << slot;
    }
    ExpectHolesThenPages(admission->block_tables.at("swa").at(0), /*first_page=*/25 / 4, /*min_pages=*/40 / 4, "swa");
    ExpectHolesThenPages(admission->block_tables.at("tail").at(0), /*first_page=*/39 / 2, /*min_pages=*/40 / 2, "tail");
    EXPECT_EQ(FindForwardBatch(plan)->request_ids.size(), 0u) << "the peer prefills; nothing runs locally";
}

}  // namespace tokenspeed::test
