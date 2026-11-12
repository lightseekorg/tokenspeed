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

// Lone-request compaction (docs/design/scheduler.md §1.4): a request no other
// one holds pages beside, within max_single_request_tokens, whose packed-group
// blocks sit one per parent beside cached pages after the hole-first placement
// of its first admission. The layout throughout: 28 parents, a history group
// of two 8-token blocks per parent and a 64-token sliding-window group of one
// block per parent, 64-token chunks. Eight one-page prompts admitted in one
// plan each take a history parent (the page plus the block reserved for the
// first decode) and a sliding parent; they finish on their first token, so
// every history parent is left [cached page | hole] and every sliding page
// stays cached. A 126-token prompt then arrives alone: its first chunk
// prepays all 16 history blocks and the copy-in block of its prompt end,
// which fill the eight holes first and span 13 parents instead of 9; its
// second chunk needs eight more sliding parents and one for the sliding
// group's copy-in block, and evicting the cached pages beside its own blocks
// frees none.

#include <gtest/gtest.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/core/block_pool.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache_test_access.h"
#include "integration_test_helper.h"
#include "scheduler/operations/group_demands.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kP = 8;
constexpr std::int32_t kParents = 28;

class LoneRequestCompactionSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = kP;
        cfg.device_allocator.total_pages = kParents + 1;
        cfg.host_allocator.total_pages = 0;
        cfg.disable_l2_cache = true;
        cfg.enable_l3_storage = false;
        cfg.snapshot_allocator.total_pages = 1;
        cfg.max_retracted_requests = 0;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 64;
        cfg.max_resident_requests = cfg.max_batch_size;
        cfg.decode_input_tokens = 1;
        cfg.overlap_schedule_depth = 0;
        cfg.disable_prefix_cache = false;
        cfg.role = Role::kFused;
        cfg.cache_groups.push_back(CacheGroupConfig{.group_id = "full_attention",
                                                    .block_granularity = kP,
                                                    .total_pages = (kParents + 1) * 2,
                                                    .cache_blocks_per_lcm_block = 2,
                                                    .attn_kind = AttnKind::kFull,
                                                    .transfer_policy = CacheTransferPolicy::FullSuffix});
        cfg.cache_groups.push_back(CacheGroupConfig{.group_id = "sliding",
                                                    .block_granularity = kP,
                                                    .total_pages = kParents + 1,
                                                    .attn_kind = AttnKind::kSlidingWindow,
                                                    .sliding_window_tokens = 64,
                                                    .transfer_policy = CacheTransferPolicy::FullSuffix});
        return cfg;
    }

    // Lands every row of the plan's batch: an intermediate chunk produces no
    // token; a last chunk or a decode produces one, on which the request
    // finishes (every request here generates one token) -- on the P role the
    // peer finishes it, once it has read the pages (pd::SucceededEvent).
    void Land(const ExecutionPlan& plan) {
        const ForwardBatch* batch = FindForwardBatch(plan);
        if (batch == nullptr) {
            return;
        }
        ExecutionEvent results;
        std::vector<std::string> finishes;
        for (std::size_t i = 0; i < batch->request_ids.size(); ++i) {
            const bool mid_chunk = i < batch->NumExtends() &&
                                   batch->extend_prefix_lens[i] + batch->input_lengths[i] < batch->prefill_lengths[i];
            results.With(forward::ExtendResult{
                .request_id = batch->request_ids[i],
                .tokens = mid_chunk ? std::vector<std::int32_t>{} : std::vector<std::int32_t>{++next_token_}});
            if (!mid_chunk && config_.role != Role::kP) {
                finishes.push_back(batch->request_ids[i]);
            }
        }
        for (const std::string& id : finishes) {
            results.With(forward::Finish{.request_id = id});
        }
        scheduler_->Advance(results);
    }

    // The eight one-page prompts, admitted in one plan and finished on their
    // first token: eight [cached page | hole] history parents and eight
    // cached sliding pages, 12 parents empty.
    void FragmentThePool() {
        std::vector<RequestSpec> prompts;
        for (std::int32_t k = 0; k < 8; ++k) {
            prompts.push_back(RequestSpec{
                .request_id = "a" + std::to_string(k), .tokens = MakeTokens(kP, 100 + 50 * k), .max_new_tokens = 1});
        }
        Submit(prompts);
        for (int plan = 0; plan < 4 && scheduler_->RequestTokenSize("a7") != -1; ++plan) {
            Land(PlanOnce());
        }
        ASSERT_EQ(scheduler_->RequestTokenSize("a7"), -1);
        ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 12);
        ASSERT_EQ(scheduler_->AvailableLcmBlocks(), kParents);
    }

    // Another layout: `parents` parents of four history blocks of
    // `history_block_tokens` beside the sliding group, without a Host tier
    // or under write-back. a0 and a1 (a page each) and b (three pages),
    // admitted in one plan, leave history parents with holes; X, a prompt of
    // `x_tokens` ending inside a page, then takes some of them. Once X
    // finishes, its prompt end is a cached tail entry.
    void LeaveATailBesideCache(std::int32_t parents, std::int32_t history_block_tokens, std::int32_t x_tokens,
                               bool write_back) {
        config_ = MakeConfig();
        config_.device_allocator.total_pages = parents + 1;
        config_.cache_groups[0].block_granularity = history_block_tokens;
        config_.cache_groups[0].total_pages = (parents + 1) * 4;
        config_.cache_groups[0].cache_blocks_per_lcm_block = 4;
        config_.cache_groups[1].total_pages = parents + 1;
        if (write_back) {
            config_.disable_l2_cache = false;
            config_.host_allocator.total_pages = 40;
            config_.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
        }
        config_ = WithHostPackingFromDevice(config_);
        scheduler_ = std::make_unique<Scheduler>(config_);
        Submit(std::vector<RequestSpec>{
            RequestSpec{.request_id = "a0", .tokens = MakeTokens(kP, 100), .max_new_tokens = 1},
            RequestSpec{.request_id = "a1", .tokens = MakeTokens(kP, 150), .max_new_tokens = 1},
            RequestSpec{.request_id = "b", .tokens = MakeTokens(3 * kP, 200), .max_new_tokens = 1}});
        for (int plan = 0; plan < 4 && scheduler_->RequestTokenSize("b") != -1; ++plan) {
            const ExecutionPlan next = PlanOnce();
            AckWriteBacks(next);
            Land(next);
        }
        ASSERT_EQ(scheduler_->RequestTokenSize("b"), -1);
        Submit(RequestSpec{.request_id = "X", .tokens = MakeTokens(x_tokens, 20000), .max_new_tokens = 1});
        for (int plan = 0; plan < 4 && scheduler_->RequestTokenSize("X") != -1; ++plan) {
            const ExecutionPlan next = PlanOnce();
            AckWriteBacks(next);
            Land(next);
        }
        ASSERT_EQ(scheduler_->RequestTokenSize("X"), -1);
        ASSERT_EQ(scheduler_->TailEntries(), 1u);
        ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 1);
    }

    // Whatever a test did, the boundary tree ends in step with the indexes.
    void TearDown() override {
        const std::vector<std::string> mismatches =
            CacheCoordinatorTestAccess::BoundaryTreeMismatches(SchedulerTestAccess::Coordinator(*scheduler_));
        EXPECT_TRUE(mismatches.empty()) << mismatches.size() << " mismatches, first: " << mismatches.front();
    }

    std::int32_t next_token_{7000};
};

// The compaction itself, on the layout above: the second chunk's admission
// fails, the request's one history block alone in a fresh parent (slot 14,
// page 5 in parent 2) and the copy-in block held for its prompt end (page 8,
// beside cached page 7 in parent 3) are re-placed into the slots of the
// cached pages beside the request's own blocks in parents 6 and 9 (pages 13
// and 19, cleared) -- slot 14's parent first, as it holds no bound block --
// and the two parents they leave fund the retry in the
// same plan: the sliding group takes them (pages 3 and 4, parent 3 once the
// retry evicts page 7) beside the seven cached sliding pages it evicts, for
// its eight blocks and its own copy-in block. An unwritten block moves
// without a copy: its new page is zeroed. The copy-in block's is not: the
// copy-in fills it whole.
TEST_F(LoneRequestCompactionSuite, RePlacesItsReserveBlockAndRetriesInTheSamePlan) {
    FragmentThePool();
    ASSERT_EQ(scheduler_->MaxSingleRequestTokens(), 128);
    const std::vector<std::int32_t> prompt = MakeTokens(126, 3000);
    Submit(RequestSpec{.request_id = "R", .tokens = prompt, .max_new_tokens = 1});

    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(first);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"R"});
    EXPECT_EQ(batch->input_lengths, std::vector<std::int32_t>{64});
    // The prompt end's block (slot 15) and its copy-in block take the first
    // two holes (pages 2 and 8); the rest fill the six other holes first,
    // then four fresh parents whole and one half.
    const std::vector<std::int32_t> before{14, 20, 26, 32, 38, 44, 49, 50, 51, 52, 53, 54, 55, 56, 5, 2};
    EXPECT_EQ(batch->block_tables.at("full_attention").AbsoluteRows(), before);
    EXPECT_EQ(batch->block_tables.at("sliding").AbsoluteRows(),
              (std::vector<std::int32_t>{6, 9, 12, 15, 18, 21, 24, 2}));
    EXPECT_EQ(scheduler_->EmptyLcmBlocks(), 0);
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 0u);
    Land(first);

    const ExecutionPlan second = PlanOnce();
    batch = FindForwardBatch(second);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"R"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{64});
    std::vector<std::int32_t> after = before;
    after[14] = 13;
    EXPECT_EQ(batch->block_tables.at("full_attention").AbsoluteRows(), after);
    EXPECT_EQ(batch->block_tables.at("sliding").AbsoluteRows(),
              (std::vector<std::int32_t>{6, 9, 12, 15, 18, 21, 24, 2, 4, 8, 11, 14, 17, 20, 23, 3}));
    // Page 19, the copy-in block's new place, is not zeroed.
    EXPECT_EQ(second.pages_to_zero.at("full_attention"), (std::vector<std::int32_t>{13}));
    EXPECT_EQ(second.pages_to_zero.at("sliding"), (std::vector<std::int32_t>{4, 8, 11, 14, 17, 20, 23, 3}));
    for (const auto& [group, copies] : second.pages_to_copy) {
        EXPECT_TRUE(copies.empty()) << group;
    }
    HostCacheStats stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.lone_compactions, 1u);
    EXPECT_EQ(stats.lone_compaction_moves, 2u);
    EXPECT_EQ(stats.device_evictions.compaction.blocks, 2u);
    Land(second);
    PlanOnce();
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), -1);
    stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.lone_compactions, 1u);
    EXPECT_EQ(stats.lone_compaction_moves, 2u);

    // The same prompt again hits the 15 pages it published, where they are.
    Submit(RequestSpec{.request_id = "R2", .tokens = prompt, .max_new_tokens = 1});
    const ExecutionPlan hit = PlanOnce();
    batch = FindForwardBatch(hit);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"R2"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{120});
    const std::vector<std::int32_t>& row = batch->block_tables.at("full_attention").AbsoluteRows();
    ASSERT_EQ(row.size(), 16u);
    EXPECT_TRUE(std::equal(row.begin(), row.begin() + 15, after.begin()));
}

// The same prompt on an empty pool takes the same two plans without a
// compaction: it runs only when an admission fails.
TEST_F(LoneRequestCompactionSuite, CompactsNothingWhenTheAdmissionFits) {
    Submit(RequestSpec{.request_id = "R", .tokens = MakeTokens(126, 3000), .max_new_tokens = 1});
    for (int plan = 0; plan < 2; ++plan) {
        const ExecutionPlan next = PlanOnce();
        ASSERT_NE(FindForwardBatch(next), nullptr) << plan;
        for (const auto& [group, copies] : next.pages_to_copy) {
            EXPECT_TRUE(copies.empty()) << group;
        }
        Land(next);
    }
    PlanOnce();
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), -1);
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 0u);
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.compaction.blocks, 0u);
}

// Under KV parallelism (two shards, one block of each per parent) the twelve
// one-page prompts leave their holes all in bucket 1, where the prompt end's
// block (slot 31, page 20) and the copy-in block held for it land first and
// the balanced allocator fills seven more before it opens fresh parents, so
// the lone request's written blocks lean to bucket 1. Its unwritten reserve
// is what re-places. The two bucket-bound blocks each take a bucket-1 slot
// one of the request's free unwritten blocks gives up (pages 82 and 6), and
// those two move to the bucket-0 slots of cached pages beside the request's
// own blocks (pages 31 and 37, cleared): only an unwritten block without a
// copy-in block beside it changes bucket, and nothing is copied.
TEST_F(LoneRequestCompactionSuite, RePlacesUnwrittenBlocksAcrossBucketsUnderKvParallelism) {
    config_.device_allocator.total_pages = 45;
    config_.max_scheduled_tokens = 128;
    config_.cache_groups[0].total_pages = 45 * 2;
    config_.cache_groups[0].shard_count = 2;
    config_.cache_groups[1].total_pages = 45;
    scheduler_ = std::make_unique<Scheduler>(config_);
    std::vector<RequestSpec> prompts;
    for (std::int32_t k = 0; k < 12; ++k) {
        prompts.push_back(RequestSpec{
            .request_id = "a" + std::to_string(k), .tokens = MakeTokens(kP, 100 + 50 * k), .max_new_tokens = 1});
    }
    Submit(prompts);
    for (int plan = 0; plan < 4 && scheduler_->RequestTokenSize("a11") != -1; ++plan) {
        Land(PlanOnce());
    }
    ASSERT_EQ(scheduler_->RequestTokenSize("a11"), -1);
    ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 20);
    ASSERT_EQ(scheduler_->MaxSingleRequestTokens(), 256);
    Submit(RequestSpec{.request_id = "R", .tokens = MakeTokens(254, 3000), .max_new_tokens = 1});

    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(first);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"R"});
    std::vector<std::int32_t> before{32, 38, 44, 50, 56, 62, 68};
    for (std::int32_t page = 73; page <= 88; ++page) {
        before.push_back(page);
    }
    before.insert(before.end(), {5, 6, 11, 12, 17, 18, 23, 24, 20});
    EXPECT_EQ(batch->block_tables.at("full_attention").AbsoluteRows(), before);
    Land(first);

    const ExecutionPlan second = PlanOnce();
    batch = FindForwardBatch(second);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{128});
    std::vector<std::int32_t> after = before;
    after[16] = 31;
    after[24] = 37;
    after[31] = 6;
    EXPECT_EQ(batch->block_tables.at("full_attention").AbsoluteRows(), after);
    // A page's bucket is (page - 1) % shards: the prompt end's block stays in
    // bucket 1 (20 -> 6); the free blocks it and the copy-in block displace
    // leave it (82 -> 31, 6 -> 37).
    for (const std::int32_t page : {20, 6, 82}) {
        EXPECT_EQ((page - 1) % 2, 1) << page;
    }
    for (const std::int32_t page : {31, 37}) {
        EXPECT_EQ((page - 1) % 2, 0) << page;
    }
    // Page 82, the copy-in block's new place, is not zeroed.
    EXPECT_EQ(second.pages_to_zero.at("full_attention"), (std::vector<std::int32_t>{31, 37, 6}));
    for (const auto& [group, copies] : second.pages_to_copy) {
        EXPECT_TRUE(copies.empty()) << group;
    }
    const HostCacheStats stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.lone_compactions, 1u);
    EXPECT_EQ(stats.lone_compaction_moves, 4u);
    EXPECT_EQ(stats.device_evictions.compaction.blocks, 2u);
    Land(second);
    PlanOnce();
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), -1);
}

// Under the write-back policy the cached page a compaction clears from a
// destination slot is written back like any eviction: recorded as a segment
// of this plan and drained into its write-back batch, beside the pages the
// retry's own eviction takes.
TEST_F(LoneRequestCompactionSuite, WritesBackTheClearedPageInTheSamePlan) {
    config_.host_allocator.total_pages = 33;
    config_.disable_l2_cache = false;
    config_.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
    config_ = WithHostPackingFromDevice(config_);
    scheduler_ = std::make_unique<Scheduler>(config_);
    FragmentThePool();
    Submit(RequestSpec{.request_id = "R", .tokens = MakeTokens(126, 3000), .max_new_tokens = 1});
    const ExecutionPlan first = PlanOnce();
    AckWriteBacks(first);
    Land(first);

    const ExecutionPlan second = PlanOnce();
    ASSERT_NE(FindForwardBatch(second), nullptr);
    EXPECT_EQ(second.pages_to_zero.at("full_attention"), (std::vector<std::int32_t>{13}));
    // Under write-back a cleared page takes the node's other Device entry
    // along: its sliding twin.
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.compaction.blocks, 4u);
    // History pages 13 and 19 (the cached pages cleared from the destination
    // slots) are rows of the plan's write-back, with their sliding twins and
    // the pages the retry's own eviction takes.
    std::vector<std::pair<std::uint32_t, std::int32_t>> rows;
    for (const CacheOperation& op : ExtractCacheOpsOfKind<WriteBackBatch>(second)) {
        const WriteBackBatch& batch = std::get<WriteBackBatch>(op);
        for (std::size_t i = 0; i < batch.op_ids.size(); ++i) {
            EXPECT_FALSE(batch.source_pinned[i]);
            for (std::size_t j = 0; j < batch.src_pages[i].size(); ++j) {
                rows.emplace_back(batch.group_ids[i][j], batch.src_pages[i][j]);
            }
        }
    }
    EXPECT_NE(std::ranges::find(rows, std::pair<std::uint32_t, std::int32_t>{0u, 13}), rows.end());
    EXPECT_NE(std::ranges::find(rows, std::pair<std::uint32_t, std::int32_t>{0u, 19}), rows.end());
    EXPECT_EQ(rows.size(), 13u);
    AckWriteBacks(second);
    Land(second);
    const ExecutionPlan last = PlanOnce();
    AckWriteBacks(last);
    Land(last);
    AckWriteBacks(PlanOnce());
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), -1);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0);
}

// When the cached history pages beside the request's blocks are held (here
// by a test reference, as a store ticket would), the second chunk's
// compaction finds no slot to move into while the scattered chunk-1 blocks
// are the pages this very admission publishes, so the completed pages are
// published first, by a zero-growth admission, and the plan runs nothing.
// The next plan's compaction moves nothing either: every parent the request
// could leave keeps a held page, so no move would free one. From then on the
// request makes no progress for Scheduler::kNoProgressAbortPlans plans and
// is aborted with the reason, its pages staying published.
TEST_F(LoneRequestCompactionSuite, PublishesCompletedPagesFirstAndAbortsWhenNothingCanEverMove) {
    FragmentThePool();
    CacheCoordinator& coordinator = SchedulerTestAccess::Coordinator(*scheduler_);
    const BlockPool& pool = SchedulerTestAccess::Pool(*scheduler_);
    std::vector<CacheBlockRef> held;
    for (std::int32_t k = 0; k < 8; ++k) {
        // A prompt's candidate pages exclude its last token: one more token
        // names the finished prompt's one full page. The history pages are
        // held; the sliding pages stay evictable, as the first chunk needs
        // one of their parents.
        for (const CacheKey& key :
             scheduler_->ExpandPrefixKeys(scheduler_->PrefixHashesForTokens(MakeTokens(kP + 1, 100 + 50 * k)))) {
            if (key.group_id != 0) {
                continue;
            }
            CacheBlockRef block = coordinator.GroupPrefixIndex(0).Find(pool, key);
            ASSERT_TRUE(block);
            held.push_back(std::move(block));
        }
    }
    ASSERT_EQ(held.size(), 8u);
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 20);

    const std::vector<std::int32_t> prompt = MakeTokens(126, 3000);
    Submit(RequestSpec{.request_id = "R", .tokens = prompt, .max_new_tokens = 1});
    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(first);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->block_tables.at("full_attention").AbsoluteRows(),
              (std::vector<std::int32_t>{14, 20, 26, 32, 38, 44, 49, 50, 51, 52, 53, 54, 55, 56, 5, 2}));
    EXPECT_EQ(scheduler_->EmptyLcmBlocks(), 0);
    Land(first);
    const std::vector<CacheKey> keys = scheduler_->ExpandPrefixKeys(scheduler_->PrefixHashesForTokens(prompt));
    ASSERT_EQ(keys.size(), 30u);  // 15 full pages of the prompt, in each group
    EXPECT_FALSE(CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, keys[0]));

    const ExecutionPlan second = PlanOnce();
    ASSERT_NE(FindForwardBatch(second), nullptr);
    EXPECT_TRUE(FindForwardBatch(second)->request_ids.empty());
    EXPECT_TRUE(second.aborts.empty());
    for (const auto& [group, pages] : second.pages_to_zero) {
        EXPECT_TRUE(pages.empty()) << group;
    }
    for (const auto& [group, copies] : second.pages_to_copy) {
        EXPECT_TRUE(copies.empty()) << group;
    }
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 0u);
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.compaction.blocks, 0u);
    // The first chunk's eight pages are published; the rest are not.
    for (std::size_t i = 0; i < 15; ++i) {
        EXPECT_EQ(CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, keys[i]).has_value(), i < 8) << i;
    }
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), 126);

    // The hole beside the request's slot 14 (page 6) is free, but every
    // parent the request could leave (slots 2 to 5, pages 26 to 44) keeps a
    // held page beside its block: nothing moves and the retry still fails.
    const ExecutionPlan third = PlanOnce();
    EXPECT_TRUE(FindForwardBatch(third)->request_ids.empty());
    EXPECT_TRUE(third.aborts.empty());
    EXPECT_TRUE(third.pages_to_copy.at("full_attention").empty());
    EXPECT_TRUE(third.pages_to_zero.at("full_attention").empty());
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 0u);
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compaction_moves, 0u);
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.compaction.blocks, 0u);
    // The publication's plan was the first built empty; every plan up to the
    // one before the kNoProgressAbortPlans-th moves nothing and aborts
    // nothing.
    for (std::int64_t stalled = 2; stalled + 1 < Scheduler::kNoProgressAbortPlans; ++stalled) {
        const ExecutionPlan next = PlanOnce();
        ASSERT_TRUE(FindForwardBatch(next)->request_ids.empty()) << stalled;
        ASSERT_TRUE(next.aborts.empty()) << stalled;
        ASSERT_TRUE(next.pages_to_copy.at("full_attention").empty()) << stalled;
    }
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 0u);
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), 126);
    const ExecutionPlan last = PlanOnce();
    EXPECT_TRUE(FindForwardBatch(last)->request_ids.empty());
    ASSERT_EQ(last.aborts.size(), 1u);
    EXPECT_EQ(last.aborts.front().request_id, "R");
    EXPECT_EQ(last.aborts.front().reason, AbortReason::kNoProgress);
    EXPECT_NE(last.aborts.front().detail.find("no progress for " + std::to_string(Scheduler::kNoProgressAbortPlans) +
                                              " consecutive plans"),
              std::string::npos);
    EXPECT_NE(last.aborts.front().detail.find("its admission needs Device pages"), std::string::npos);
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 0u);
    // Finished by the abort, it leaves at the next plan: its unwritten
    // blocks are freed (three fresh parents and the half parent empty; the
    // fourth fresh one keeps two published pages), its published pages (and
    // their sliding twins) stay cached beside the eight held ones.
    EXPECT_TRUE(PlanOnce().aborts.empty());
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), -1);
    EXPECT_EQ(scheduler_->EmptyLcmBlocks(), 4);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 20);
    held.clear();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), kParents);
}

// The no-progress abort never takes the compaction's own plan. The layout
// above, stalled up to the plan before the kNoProgressAbortPlans-th; then
// a2's history page, beside the published block of slot 0 (page 14) in a
// kept parent, is released. The kNoProgressAbortPlans-th plan re-places the
// unwritten slot 14 (page 5) into that page's slot (page 13, cleared): its
// parent then empties, but the retry needs two parents and still fails, so
// the plan runs nothing -- it is the compaction's, not the abort's; the
// next plan, with nothing left to move, is.
TEST_F(LoneRequestCompactionSuite, NeverAbortsInTheCompactionsOwnPlan) {
    FragmentThePool();
    CacheCoordinator& coordinator = SchedulerTestAccess::Coordinator(*scheduler_);
    const BlockPool& pool = SchedulerTestAccess::Pool(*scheduler_);
    std::vector<CacheBlockRef> held;
    for (std::int32_t k = 0; k < 8; ++k) {
        for (const CacheKey& key :
             scheduler_->ExpandPrefixKeys(scheduler_->PrefixHashesForTokens(MakeTokens(kP + 1, 100 + 50 * k)))) {
            if (key.group_id == 0) {
                held.push_back(coordinator.GroupPrefixIndex(0).Find(pool, key));
            }
        }
    }
    ASSERT_EQ(held.size(), 8u);
    Submit(RequestSpec{.request_id = "R", .tokens = MakeTokens(126, 3000), .max_new_tokens = 1});
    Land(PlanOnce());
    EXPECT_TRUE(FindForwardBatch(PlanOnce())->request_ids.empty());  // publishes the first chunk's pages
    for (std::int64_t stalled = 1; stalled + 1 < Scheduler::kNoProgressAbortPlans; ++stalled) {
        const ExecutionPlan next = PlanOnce();
        ASSERT_TRUE(next.aborts.empty()) << stalled;
        ASSERT_TRUE(next.pages_to_copy.at("full_attention").empty()) << stalled;
    }
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 0u);
    held[2] = CacheBlockRef{};
    const ExecutionPlan own = PlanOnce();
    EXPECT_TRUE(FindForwardBatch(own)->request_ids.empty());
    EXPECT_TRUE(own.aborts.empty());
    EXPECT_TRUE(own.pages_to_copy.at("full_attention").empty());
    EXPECT_EQ(own.pages_to_zero.at("full_attention"), std::vector<std::int32_t>{13});
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 1u);
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compaction_moves, 1u);
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.compaction.blocks, 1u);
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), 126);
    const ExecutionPlan last = PlanOnce();
    ASSERT_EQ(last.aborts.size(), 1u);
    EXPECT_EQ(last.aborts.front().request_id, "R");
    EXPECT_EQ(last.aborts.front().reason, AbortReason::kNoProgress);
    EXPECT_NE(last.aborts.front().detail.find(
                  "no progress for " + std::to_string(Scheduler::kNoProgressAbortPlans + 1) + " consecutive plans"),
              std::string::npos);
    EXPECT_TRUE(PlanOnce().aborts.empty());
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), -1);
    held.clear();
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), kParents);
}

// Under KV parallelism a lone first admission whose probe hits a Device tail
// can fail every plan with nothing to move: three parents of two blocks, one
// of each of two buckets, and no other group. X (17 tokens) leaves its two
// pages cached in bucket 1 and its one-token tail in bucket 0. R, X plus 14
// tokens, hits both pages and the tail: the tail's private copy takes a
// second bucket-0 slot, and the prompt end's block and its copy-in block,
// which share a bucket, find two free slots in neither; the hit pages and
// the tail keep their buckets, so compaction moves nothing. On the fused role
// R, still Submitted, is aborted at the kNoProgressAbortPlans-th plan, which
// ends the stall streak: S, the same prompt submitted right after, stalls
// the same way and is aborted at its own kNoProgressAbortPlans-th plan, not
// at its first.
TEST_F(LoneRequestCompactionSuite, AbortsALoneFirstAdmissionItsDeviceTailBindsToOneBucket) {
    config_ = MakeConfig();
    config_.device_allocator.total_pages = 4;
    config_.max_scheduled_tokens = 16;
    config_.cache_groups.resize(1);
    config_.cache_groups[0].total_pages = 4 * 2;
    config_.cache_groups[0].shard_count = 2;
    scheduler_ = std::make_unique<Scheduler>(config_);
    ASSERT_EQ(scheduler_->MaxSingleRequestTokens(), 32);
    Submit(RequestSpec{.request_id = "X", .tokens = MakeTokens(17, 20000), .max_new_tokens = 1});
    for (int plan = 0; plan < 4 && scheduler_->RequestTokenSize("X") != -1; ++plan) {
        Land(PlanOnce());
    }
    ASSERT_EQ(scheduler_->RequestTokenSize("X"), -1);
    ASSERT_EQ(scheduler_->TailEntries(), 1u);
    std::vector<std::int32_t> prompt = MakeTokens(17, 20000);
    const std::vector<std::int32_t> more = MakeTokens(14, 50000);
    prompt.insert(prompt.end(), more.begin(), more.end());
    for (const std::string id : {"R", "S"}) {
        SCOPED_TRACE(id);
        Submit(RequestSpec{.request_id = id, .tokens = prompt, .max_new_tokens = 1});
        for (std::int64_t stalled = 1; stalled < Scheduler::kNoProgressAbortPlans; ++stalled) {
            const ExecutionPlan next = PlanOnce();
            ASSERT_TRUE(FindForwardBatch(next)->request_ids.empty()) << stalled;
            ASSERT_TRUE(next.aborts.empty()) << stalled;
            ASSERT_EQ(scheduler_->WaitingSize(), 1u) << stalled;
        }
        const ExecutionPlan last = PlanOnce();
        ASSERT_EQ(last.aborts.size(), 1u);
        EXPECT_EQ(last.aborts.front().request_id, id);
        EXPECT_EQ(last.aborts.front().reason, AbortReason::kNoProgress);
        EXPECT_NE(last.aborts.front().detail.find(
                      "no progress for " + std::to_string(Scheduler::kNoProgressAbortPlans) + " consecutive plans"),
                  std::string::npos);
        EXPECT_NE(last.aborts.front().detail.find("its admission needs Device pages"), std::string::npos);
        EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 0u);
    }
    EXPECT_TRUE(PlanOnce().aborts.empty());
    EXPECT_EQ(scheduler_->RequestTokenSize("S"), -1);
    EXPECT_EQ(scheduler_->TailEntries(), 1u);
}

// The P and D roles keep only the stall warning: a stuck Submitted request
// there has its PD transfer registered, which a scheduler abort does not
// conclude. The layout above on the P role, which reserves no copy-in
// blocks: X (33 tokens) leaves four pages and its tail in five of the six
// slots, the free one in the other bucket than the tail; R, X plus a token,
// needs the tail's private copy in the tail's bucket, and waits past
// kNoProgressAbortPlans plans with the warning logged and no abort.
TEST_F(LoneRequestCompactionSuite, PRoleNeverAbortsALoneStuckFirstAdmission) {
    config_ = MakeConfig();
    config_.role = Role::kP;
    config_.device_allocator.total_pages = 4;
    config_.max_scheduled_tokens = 16;
    config_.cache_groups.resize(1);
    config_.cache_groups[0].total_pages = 4 * 2;
    config_.cache_groups[0].shard_count = 2;
    scheduler_ = std::make_unique<Scheduler>(config_);
    ASSERT_EQ(scheduler_->MaxSingleRequestTokens(), 40);
    const auto submit = [&](const std::string& id, const std::vector<std::int32_t>& tokens) {
        Submit(RequestSpec{.request_id = id, .tokens = tokens, .max_new_tokens = 1});
        ExecutionEvent bootstrapped;
        bootstrapped.With(pd::BootstrappedEvent{id});
        scheduler_->Advance(bootstrapped);
    };
    std::vector<std::int32_t> prompt = MakeTokens(33, 20000);
    submit("X", prompt);
    for (int plan = 0; plan < 8 && scheduler_->RequestTokenSize("X") != -1; ++plan) {
        const ExecutionPlan next = PlanOnce();
        Land(next);
        if (next.remote_decode) {
            ExecutionEvent read;
            read.With(pd::SucceededEvent{"X"});
            scheduler_->Advance(read);
        }
    }
    ASSERT_EQ(scheduler_->RequestTokenSize("X"), -1);
    ASSERT_EQ(scheduler_->TailEntries(), 1u);
    prompt.push_back(1);
    submit("R", prompt);

    std::ostringstream log;
    const std::shared_ptr<spdlog::logger> previous = spdlog::default_logger();
    spdlog::set_default_logger(
        std::make_shared<spdlog::logger>("stall", std::make_shared<spdlog::sinks::ostream_sink_mt>(log)));
    std::size_t aborts = 0;
    std::size_t forwards = 0;
    for (std::int64_t plan = 0; plan < Scheduler::kNoProgressAbortPlans + 8; ++plan) {
        const ExecutionPlan next = PlanOnce();
        aborts += next.aborts.size();
        forwards += FindForwardBatch(next)->request_ids.size();
    }
    spdlog::set_default_logger(previous);
    EXPECT_EQ(aborts, 0u);
    EXPECT_EQ(forwards, 0u);
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 0u);
    EXPECT_NE(log.str().find("no work scheduled for " + std::to_string(Scheduler::kNoProgressAbortPlans) +
                             " consecutive rounds"),
              std::string::npos);
    EXPECT_EQ(log.str().find("no-progress abort"), std::string::npos);
}

// Once the held pages are released but one (a0's history page, so the seven
// cached sliding pages fund seven of the nine sliding parents the second
// chunk needs: eight blocks and the sliding copy-in block), a published
// chunk-1 block moves too: the unwritten slot 14 (page 5), alone in its
// parent, re-places first, into the cleared slot of a1's page beside the
// request's copy-in block (page 7), and the published block in the last of
// the request's single-block parents (slot 5, page 44 in parent 21) is
// copied into the cleared slot of a2's page beside slot 0's block (page 13),
// its index entry following under the same key; parents 2 and 21 fund the
// retry. The same prompt afterwards hits all 15 published pages, slot 5 of
// them at page 13.
TEST_F(LoneRequestCompactionSuite, MovesPublishedPagesByCopyAndKeepsTheirKeys) {
    FragmentThePool();
    CacheCoordinator& coordinator = SchedulerTestAccess::Coordinator(*scheduler_);
    const BlockPool& pool = SchedulerTestAccess::Pool(*scheduler_);
    std::vector<CacheBlockRef> held;
    for (std::int32_t k = 0; k < 8; ++k) {
        for (const CacheKey& key :
             scheduler_->ExpandPrefixKeys(scheduler_->PrefixHashesForTokens(MakeTokens(kP + 1, 100 + 50 * k)))) {
            // a0's sliding page stays evictable: the first chunk needs its
            // parent.
            if (key.group_id == 0 || k > 0) {
                held.push_back(coordinator.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id)).Find(pool, key));
            }
        }
    }
    ASSERT_EQ(held.size(), 15u);
    const std::vector<std::int32_t> prompt = MakeTokens(126, 3000);
    Submit(RequestSpec{.request_id = "R", .tokens = prompt, .max_new_tokens = 1});
    Land(PlanOnce());
    const std::vector<CacheKey> keys = scheduler_->ExpandPrefixKeys(scheduler_->PrefixHashesForTokens(prompt));
    const ExecutionPlan publishes = PlanOnce();  // publishes the first chunk's pages
    EXPECT_TRUE(FindForwardBatch(publishes)->request_ids.empty());
    EXPECT_TRUE(publishes.aborts.empty());
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), 126);
    const std::optional<CacheBlockLocation> published_at =
        CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, keys[5]);
    ASSERT_TRUE(published_at);
    EXPECT_EQ(pool.ResolveCacheBlockId(0, *published_at), 44);
    const std::optional<PrefixCacheIndex::CachedBlockSnapshot> before =
        coordinator.GroupPrefixIndex(0).Inspect(pool, keys[5]);
    ASSERT_TRUE(before);
    const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator);
    const BoundaryTree::Node* node = tree.Find(keys[5].content_hash);
    ASSERT_NE(node, nullptr);
    const std::int32_t node_entries = node->Entries(BoundaryTree::kDevice);
    held.erase(held.begin() + 1, held.end());  // a0's history page stays held
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 7);

    const ExecutionPlan second = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(second);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"R"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{64});
    EXPECT_EQ(batch->block_tables.at("full_attention").AbsoluteRows(),
              (std::vector<std::int32_t>{14, 20, 26, 32, 38, 13, 49, 50, 51, 52, 53, 54, 55, 56, 7, 2}));
    EXPECT_EQ(second.pages_to_copy.at("full_attention"),
              (std::vector<std::pair<std::int32_t, std::int32_t>>{{44, 13}}));
    EXPECT_TRUE(second.pages_to_copy.at("sliding").empty());
    EXPECT_EQ(second.pages_to_zero.at("full_attention"), std::vector<std::int32_t>{7});
    // Parents 2 and 21, vacated, go to the sliding group (pages 3 and 22)
    // with the parents of the cached sliding pages the retry evicts.
    EXPECT_EQ(second.pages_to_zero.at("sliding"), (std::vector<std::int32_t>{8, 11, 14, 17, 20, 23, 22, 3}));
    const HostCacheStats stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.lone_compactions, 1u);
    EXPECT_EQ(stats.lone_compaction_moves, 2u);
    EXPECT_EQ(stats.device_evictions.compaction.blocks, 2u);
    // The entry moved with its key: same epochs, logical page, boundary kind
    // and session at the new location, nothing left at the old one, and its
    // boundary-tree node never changed hands.
    const std::optional<CacheBlockLocation> moved_to =
        CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, keys[5]);
    ASSERT_TRUE(moved_to);
    EXPECT_EQ(pool.ResolveCacheBlockId(0, *moved_to), 13);
    EXPECT_FALSE(pool.IsOccupied(*published_at) &&
                 coordinator.GroupPrefixIndex(0).KeyOf(pool, *published_at).has_value());
    const std::optional<PrefixCacheIndex::CachedBlockSnapshot> after =
        coordinator.GroupPrefixIndex(0).Inspect(pool, keys[5]);
    ASSERT_TRUE(after);
    EXPECT_EQ(after->location, *moved_to);
    EXPECT_EQ(after->metadata.last_access_epoch, before->metadata.last_access_epoch);
    EXPECT_EQ(after->metadata.max_access_epoch, before->metadata.max_access_epoch);
    EXPECT_EQ(after->metadata.logical_block_index, before->metadata.logical_block_index);
    EXPECT_EQ(after->metadata.boundary_kind, before->metadata.boundary_kind);
    EXPECT_EQ(after->metadata.was_acquired, before->metadata.was_acquired);
    EXPECT_EQ(after->metadata.session, before->metadata.session);
    EXPECT_TRUE(after->pinned);  // the request's own table holds it
    EXPECT_EQ(tree.Find(keys[5].content_hash), node);
    EXPECT_EQ(node->boundary_tokens, 6 * kP);
    EXPECT_EQ(node->Entries(BoundaryTree::kDevice), node_entries);
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(coordinator).empty());
    Land(second);
    PlanOnce();
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), -1);

    Submit(RequestSpec{.request_id = "R2", .tokens = prompt, .max_new_tokens = 1});
    const ExecutionPlan hit = PlanOnce();
    batch = FindForwardBatch(hit);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"R2"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{120});
    const std::vector<std::int32_t>& row = batch->block_tables.at("full_attention").AbsoluteRows();
    ASSERT_EQ(row.size(), 16u);
    EXPECT_EQ(std::vector<std::int32_t>(row.begin(), row.begin() + 15),
              (std::vector<std::int32_t>{14, 20, 26, 32, 38, 13, 49, 50, 51, 52, 53, 54, 55, 56, 7}));
}

// A parent is vacated only when the retry can take what stays in it. The
// layout above with a2's and a3's history pages held as well, so the kept
// parents offer two slots: a1's page beside the request's copy-in block
// (page 7) and a4's page beside its slot 2 (page 25). The unwritten slot 14,
// alone in its parent, takes the first. With a7's page held too, parent 21,
// the next to vacate -- slot 5 (page 44) beside that page -- stays bound
// whatever leaves it and is skipped: parent 18 (slot 4, page 38, beside a6's
// evictable page) takes the other slot and, with parent 2, funds the retry
// in the same plan, and nothing of parent 21 is copied, zeroed or evicted.
// With a7's page evictable, parents 2 and 21 go.
TEST_F(LoneRequestCompactionSuite, NeverVacatesAParentAHeldPageKeepsBound) {
    for (const bool hold_a7 : {true, false}) {
        SCOPED_TRACE(hold_a7 ? "a7's page held" : "a7's page evictable");
        scheduler_ = std::make_unique<Scheduler>(config_);
        FragmentThePool();
        CacheCoordinator& coordinator = SchedulerTestAccess::Coordinator(*scheduler_);
        const BlockPool& pool = SchedulerTestAccess::Pool(*scheduler_);
        std::vector<CacheBlockRef> history;
        std::vector<CacheBlockRef> sliding;
        for (std::int32_t k = 0; k < 8; ++k) {
            for (const CacheKey& key :
                 scheduler_->ExpandPrefixKeys(scheduler_->PrefixHashesForTokens(MakeTokens(kP + 1, 100 + 50 * k)))) {
                if (key.group_id == 0) {
                    history.push_back(coordinator.GroupPrefixIndex(0).Find(pool, key));
                } else if (k > 0) {
                    // a0's sliding page stays evictable: the first chunk
                    // needs its parent.
                    sliding.push_back(coordinator.GroupPrefixIndex(1).Find(pool, key));
                }
            }
        }
        ASSERT_EQ(history.size(), 8u);
        ASSERT_EQ(sliding.size(), 7u);
        const std::vector<std::int32_t> prompt = MakeTokens(126, 3000);
        Submit(RequestSpec{.request_id = "R", .tokens = prompt, .max_new_tokens = 1});
        Land(PlanOnce());
        ASSERT_TRUE(FindForwardBatch(PlanOnce())->request_ids.empty());  // publishes the first chunk's pages
        const std::vector<CacheKey> keys = scheduler_->ExpandPrefixKeys(scheduler_->PrefixHashesForTokens(prompt));
        const CacheBlockLocation slot5_at = *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, keys[5]);
        ASSERT_EQ(pool.ResolveCacheBlockId(0, slot5_at), 44);
        ASSERT_EQ(history[7]->Location().lcm_block_id, slot5_at.lcm_block_id);
        sliding.clear();
        for (const std::int32_t k : {1, 4, 5, 6}) {
            history[static_cast<std::size_t>(k)] = CacheBlockRef{};
        }
        if (!hold_a7) {
            history[7] = CacheBlockRef{};
        }

        const ExecutionPlan second = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(second);
        ASSERT_NE(batch, nullptr);
        EXPECT_EQ(batch->request_ids, std::vector<std::string>{"R"});
        EXPECT_EQ(second.pages_to_copy.at("full_attention"),
                  (std::vector<std::pair<std::int32_t, std::int32_t>>{{hold_a7 ? 38 : 44, 25}}));
        EXPECT_EQ(second.pages_to_zero.at("full_attention"), std::vector<std::int32_t>{7});
        const HostCacheStats stats = scheduler_->GetHostCacheStats();
        EXPECT_EQ(stats.lone_compactions, 1u);
        EXPECT_EQ(stats.lone_compaction_moves, 2u);
        EXPECT_EQ(stats.device_evictions.compaction.blocks, 2u);  // a1's and a4's pages, from pages 7 and 25
        if (hold_a7) {
            EXPECT_EQ(CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, keys[5]), slot5_at);
            EXPECT_EQ(pool.OccupiedCount(slot5_at.lcm_block_id), 2);
        }
        EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(coordinator).empty());
        Land(second);
        PlanOnce();
        EXPECT_EQ(scheduler_->RequestTokenSize("R"), -1);
        history.clear();
        EXPECT_EQ(scheduler_->AvailableLcmBlocks(), kParents);
    }
}

// A request whose compaction moved its blocks is neither retracted nor
// aborted in the compaction's own plan: its image's stores would read the
// moved blocks, or an abort's grant hand them out, before the plan's copies
// land. The held-pages layout with a1's sliding page held too, so the second
// chunk needs three history parents beside the six cached sliding pages
// left. After the publication plan a2's, a3's and a7's history pages are
// released, and S, a one-page prompt, waits, with a snapshot pool or
// without: the compaction re-places slot 14 (page 5) into page 13 and
// copies slot 5 (page 44) into page 19, which frees two parents, the retry
// still fails, and the plan runs nothing. The next plan, with nothing left
// to move, retracts R (a snapshot pool) or aborts it, and admits S.
TEST_F(LoneRequestCompactionSuite, NeverRetractsTheRequestItsCompactionCopiesInto) {
    for (const bool snapshot_pool : {true, false}) {
        SCOPED_TRACE(snapshot_pool ? "snapshot pool" : "no snapshot pool");
        config_ = MakeConfig();
        if (snapshot_pool) {
            config_.snapshot_allocator.total_pages = config_.device_allocator.total_pages;
            config_.max_retracted_requests = 8;
        }
        config_ = WithHostPackingFromDevice(config_);
        scheduler_ = std::make_unique<Scheduler>(config_);
        FragmentThePool();
        CacheCoordinator& coordinator = SchedulerTestAccess::Coordinator(*scheduler_);
        const BlockPool& pool = SchedulerTestAccess::Pool(*scheduler_);
        std::vector<CacheBlockRef> held;
        CacheBlockRef held_sliding;
        for (std::int32_t k = 0; k < 8; ++k) {
            for (const CacheKey& key :
                 scheduler_->ExpandPrefixKeys(scheduler_->PrefixHashesForTokens(MakeTokens(kP + 1, 100 + 50 * k)))) {
                if (key.group_id == 0) {
                    held.push_back(coordinator.GroupPrefixIndex(0).Find(pool, key));
                } else if (k == 1) {
                    held_sliding = coordinator.GroupPrefixIndex(1).Find(pool, key);
                }
            }
        }
        ASSERT_EQ(held.size(), 8u);
        ASSERT_TRUE(held_sliding);
        Submit(RequestSpec{.request_id = "R", .tokens = MakeTokens(126, 3000), .max_new_tokens = 1});
        Land(PlanOnce());
        ASSERT_TRUE(FindForwardBatch(PlanOnce())->request_ids.empty());  // publishes the first chunk's pages
        for (const std::size_t k : {2, 3, 7}) {
            held[k] = CacheBlockRef{};
        }
        Submit(RequestSpec{.request_id = "S", .tokens = MakeTokens(kP, 9000), .max_new_tokens = 1});

        const ExecutionPlan own = PlanOnce();
        EXPECT_TRUE(FindForwardBatch(own)->request_ids.empty());
        EXPECT_EQ(own.pages_to_copy.at("full_attention"),
                  (std::vector<std::pair<std::int32_t, std::int32_t>>{{44, 19}}));
        EXPECT_EQ(own.pages_to_zero.at("full_attention"), std::vector<std::int32_t>{13});
        EXPECT_TRUE(own.aborts.empty());
        EXPECT_EQ(FindSnapshotStore(own), nullptr);
        EXPECT_EQ(scheduler_->RetractedSize(), 0u);
        EXPECT_EQ(scheduler_->RequestTokenSize("R"), 126);
        EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 1u);

        const ExecutionPlan next = PlanOnce();
        EXPECT_EQ(FindForwardBatch(next)->request_ids, std::vector<std::string>{"S"});
        EXPECT_TRUE(next.pages_to_copy.at("full_attention").empty());
        if (snapshot_pool) {
            EXPECT_NE(FindSnapshotStore(next), nullptr);
            EXPECT_EQ(scheduler_->RetractedSize(), 1u);
        } else {
            ASSERT_EQ(next.aborts.size(), 1u);
            EXPECT_EQ(next.aborts.front().request_id, "R");
        }
    }
}

// The same for a compaction that only re-places blocks the request has not
// written, which its plan zeroes: an abort's grant could hand one out as
// another request's copy destination, which the zeroing would then clear
// after the copy. The first layout with a5's, a6's and a7's sliding pages
// held (as store tickets would hold them): the second chunk's compaction
// re-places its reserve (pages 13, 25, 31 and 37 zeroed) and its copy-in
// block, copies nothing, and the retry still fails; with S waiting, the
// plan runs nothing. The next plan, with nothing left to move, retracts R
// (a snapshot pool) or aborts it, and admits S.
TEST_F(LoneRequestCompactionSuite, NeverRetractsTheRequestWhoseBlocksItsCompactionZeroes) {
    for (const bool snapshot_pool : {true, false}) {
        SCOPED_TRACE(snapshot_pool ? "snapshot pool" : "no snapshot pool");
        config_ = MakeConfig();
        if (snapshot_pool) {
            config_.snapshot_allocator.total_pages = config_.device_allocator.total_pages;
            config_.max_retracted_requests = 8;
        }
        config_ = WithHostPackingFromDevice(config_);
        scheduler_ = std::make_unique<Scheduler>(config_);
        FragmentThePool();
        CacheCoordinator& coordinator = SchedulerTestAccess::Coordinator(*scheduler_);
        const BlockPool& pool = SchedulerTestAccess::Pool(*scheduler_);
        std::vector<CacheBlockRef> held;
        for (std::int32_t k = 5; k < 8; ++k) {
            for (const CacheKey& key :
                 scheduler_->ExpandPrefixKeys(scheduler_->PrefixHashesForTokens(MakeTokens(kP + 1, 100 + 50 * k)))) {
                if (key.group_id == 1) {
                    held.push_back(coordinator.GroupPrefixIndex(1).Find(pool, key));
                }
            }
        }
        ASSERT_EQ(held.size(), 3u);
        Submit(RequestSpec{.request_id = "R", .tokens = MakeTokens(126, 3000), .max_new_tokens = 1});
        Land(PlanOnce());
        Submit(RequestSpec{.request_id = "S", .tokens = MakeTokens(kP, 9000), .max_new_tokens = 1});

        const ExecutionPlan own = PlanOnce();
        EXPECT_TRUE(FindForwardBatch(own)->request_ids.empty());
        EXPECT_EQ(own.pages_to_zero.at("full_attention"), (std::vector<std::int32_t>{13, 25, 31, 37}));
        EXPECT_TRUE(own.pages_to_zero.at("sliding").empty());
        for (const auto& [group, copies] : own.pages_to_copy) {
            EXPECT_TRUE(copies.empty()) << group;
        }
        EXPECT_TRUE(own.aborts.empty());
        EXPECT_EQ(FindSnapshotStore(own), nullptr);
        EXPECT_EQ(scheduler_->RequestTokenSize("R"), 126);
        EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 1u);
        EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compaction_moves, 5u);

        const ExecutionPlan next = PlanOnce();
        EXPECT_EQ(FindForwardBatch(next)->request_ids, std::vector<std::string>{"S"});
        if (snapshot_pool) {
            EXPECT_NE(FindSnapshotStore(next), nullptr);
            EXPECT_EQ(scheduler_->RetractedSize(), 1u);
        } else {
            ASSERT_EQ(next.aborts.size(), 1u);
            EXPECT_EQ(next.aborts.front().request_id, "R");
        }
    }
}

// The P role never prepays (it never decodes locally), so a request's
// history blocks beyond its first chunk's holes fill fresh parents whole and
// nothing of its own scatters; and its chunk admissions never compact
// anyway, since the transfer pin holds its pages until the peer has read
// them. The same layout completes without a compaction.
TEST_F(LoneRequestCompactionSuite, PRoleChunkAdmissionsNeverCompact) {
    config_.role = Role::kP;
    scheduler_ = std::make_unique<Scheduler>(config_);
    std::vector<RequestSpec> prompts;
    ExecutionEvent bootstrapped;
    for (std::int32_t k = 0; k < 8; ++k) {
        prompts.push_back(RequestSpec{
            .request_id = "a" + std::to_string(k), .tokens = MakeTokens(kP, 100 + 50 * k), .max_new_tokens = 1});
        bootstrapped.With(pd::BootstrappedEvent{"a" + std::to_string(k)});
    }
    Submit(prompts);
    scheduler_->Advance(bootstrapped);
    Land(PlanOnce());
    const ExecutionPlan handoff = PlanOnce();
    ASSERT_TRUE(handoff.remote_decode);
    ASSERT_EQ(handoff.remote_decode->request_ids.size(), 8u);
    ExecutionEvent read;
    for (const std::string& id : handoff.remote_decode->request_ids) {
        read.With(pd::SucceededEvent{id});
    }
    scheduler_->Advance(read);
    PlanOnce();
    ASSERT_EQ(scheduler_->RequestTokenSize("a7"), -1);
    ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 12);

    Submit(RequestSpec{.request_id = "R", .tokens = MakeTokens(134, 3000), .max_new_tokens = 1});
    ExecutionEvent bootstrap;
    bootstrap.With(pd::BootstrappedEvent{"R"});
    scheduler_->Advance(bootstrap);
    for (const std::int32_t computed : {0, 64, 128}) {
        const ExecutionPlan next = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(next);
        ASSERT_NE(batch, nullptr) << computed;
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{"R"}) << computed;
        EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{computed});
        for (const auto& [group, copies] : next.pages_to_copy) {
            EXPECT_TRUE(copies.empty()) << group;
        }
        Land(next);
        EXPECT_TRUE(scheduler_->PdTransferPinned("R")) << computed;
    }
    // Its first chunk took the eight holes; the second took the slots of four
    // cached pages it evicted and two fresh parents, the last one more slot.
    const ExecutionPlan done = PlanOnce();
    ASSERT_TRUE(done.remote_decode);
    EXPECT_EQ(done.remote_decode->request_ids, std::vector<std::string>{"R"});
    EXPECT_EQ(done.remote_decode->block_tables.at("full_attention").AbsoluteRows(),
              (std::vector<std::int32_t>{2, 8, 14, 20, 26, 32, 38, 44, 1, 7, 13, 19, 29, 30, 35, 36, 25}));
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 0u);
    EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.compaction.blocks, 0u);
    ExecutionEvent read_back;
    read_back.With(pd::SucceededEvent{"R"});
    scheduler_->Advance(read_back);
    PlanOnce();
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), -1);
}

// At a first admission the probe's Device tail block may be relocated as part
// of vacating its parent, like a hit page, but is never cleared as a
// destination: the retry still reads it. Another layout, with four history
// blocks per parent and a replayable 32-token sliding group in 20 parents,
// under both write policies: eight 24-token prompts, then a prompt Q that
// resumes b7's three pages and adds 70 tokens, leave Q's 6-token tail at page
// 50, beside a cached page (49) and b7's first two pages (51, 52), and Q's last
// two pages (57, 58) alone beside two holes. One cached page of every other
// parent the probe does not reach is held (as a store ticket would), so none of
// them can be freed whole, and so is page 51 (a hit another holder pins), which
// keeps the tail's parent. The same prompt plus 48 tokens, probed alone, hits
// the 11 pages and the tail but cannot be admitted: its sliding blocks need one
// parent more than the ten empty ones. Its compaction vacates the parent of
// pages 57 and 58 by copy: 57 into the slot of cached page 49 (cleared) and 58,
// which would take the tail's slot next were it any cached page, into a hole of
// another kept parent (page 66); the tail stays where it is, and the retry
// admits and copies it from there.
TEST_F(LoneRequestCompactionSuite, FirstAdmissionCompactionNeverClearsTheProbesTailBlock) {
    for (const bool write_back : {false, true}) {
        SCOPED_TRACE(write_back ? "write-back" : "write-through");
        config_ = MakeConfig();
        config_.device_allocator.total_pages = 20;
        config_.host_allocator.total_pages = write_back ? 20 : 0;
        config_.disable_l2_cache = !write_back;
        if (write_back) {
            config_.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
        }
        config_.cache_groups[0].total_pages = 20 * 4;
        config_.cache_groups[0].cache_blocks_per_lcm_block = 4;
        config_.cache_groups[1].total_pages = 20;
        config_.cache_groups[1].sliding_window_tokens = 32;
        config_.cache_groups[1].replayable = true;
        config_ = WithHostPackingFromDevice(config_);
        scheduler_ = std::make_unique<Scheduler>(config_);
        std::vector<RequestSpec> prompts;
        for (std::int32_t k = 0; k < 8; ++k) {
            prompts.push_back(RequestSpec{
                .request_id = "b" + std::to_string(k), .tokens = MakeTokens(24, 100 + 50 * k), .max_new_tokens = 1});
        }
        Submit(prompts);
        for (int plan = 0; plan < 16 && scheduler_->RequestTokenSize("b7") != -1; ++plan) {
            const ExecutionPlan next = PlanOnce();
            AckWriteBacks(next);
            Land(next);
        }
        ASSERT_EQ(scheduler_->RequestTokenSize("b7"), -1);
        std::vector<std::int32_t> q_prompt = MakeTokens(24, 450);  // b7's prompt
        const std::vector<std::int32_t> q_new = MakeTokens(70, 3000);
        q_prompt.insert(q_prompt.end(), q_new.begin(), q_new.end());
        Submit(RequestSpec{.request_id = "Q", .tokens = q_prompt, .max_new_tokens = 1});
        for (int plan = 0; plan < 16 && scheduler_->RequestTokenSize("Q") != -1; ++plan) {
            const ExecutionPlan next = PlanOnce();
            AckWriteBacks(next);
            Land(next);
        }
        ASSERT_EQ(scheduler_->RequestTokenSize("Q"), -1);
        ASSERT_EQ(scheduler_->TailEntries(), 1u);
        CacheCoordinator& coordinator = SchedulerTestAccess::Coordinator(*scheduler_);
        const BlockPool& pool = SchedulerTestAccess::Pool(*scheduler_);
        const PrefixCacheIndex& index = coordinator.GroupPrefixIndex(0);

        std::vector<std::int32_t> r_prompt = q_prompt;
        const std::vector<std::int32_t> extra = MakeTokens(48, 5000);
        r_prompt.insert(r_prompt.end(), extra.begin(), extra.end());
        const std::vector<std::string> hashes = scheduler_->PrefixHashesForTokens(r_prompt);
        const auto probe_now = [&] {
            return coordinator.ProbePromptPrefix(CacheCoordinator::PromptPrefix{
                .prefix_hashes = hashes,
                .tokens = r_prompt,
                .max_tokens = static_cast<std::int32_t>(r_prompt.size()) - 1,
            });
        };
        const CacheCoordinator::PrefixProbe probe = probe_now();
        ASSERT_EQ(probe.device.num_common_tokens, 94);
        ASSERT_EQ(probe.device.full_page_tokens, 88);
        ASSERT_TRUE(probe.device.tail);
        ASSERT_EQ(probe.device.tail->tail_tokens, 6);
        const std::vector<CacheBlockLocation> tail_at = index.MatchedLocations(
            pool, probe.device.tail->group_keys[0], /*begin_blocks=*/0, probe.device.tail->per_group[0]);
        ASSERT_EQ(tail_at.size(), 1u);
        EXPECT_EQ(pool.ResolveCacheBlockId(0, tail_at.front()), 50);
        const CacheKey tail_key = *index.KeyOf(pool, tail_at.front());
        const std::vector<CacheBlockLocation> hits =
            index.MatchedLocations(pool, (*probe.group_keys)[0], /*begin_blocks=*/0, probe.device.per_group[0]);
        std::vector<std::int32_t> hit_pages;
        for (const CacheBlockLocation hit : hits) {
            hit_pages.push_back(pool.ResolveCacheBlockId(0, hit));
        }
        ASSERT_EQ(hit_pages, (std::vector<std::int32_t>{51, 52, 65, 67, 68, 53, 54, 55, 56, 57, 58}));
        std::vector<CacheBlockRef> held{index.Find(pool, *index.KeyOf(pool, hits.front()))};
        std::vector<std::int32_t> held_pages;
        for (std::int32_t parent = 1; parent <= pool.NumLcmBlocks(); ++parent) {
            const CacheBlockLocation first{.lcm_block_id = parent, .slot_index = 0};
            if (pool.BoundGroup(parent) == std::optional<std::uint32_t>{0u} && parent != tail_at.front().lcm_block_id &&
                std::ranges::none_of(hits, [&](CacheBlockLocation hit) { return hit.lcm_block_id == parent; })) {
                held.push_back(index.Find(pool, *index.KeyOf(pool, first)));
                held_pages.push_back(pool.ResolveCacheBlockId(0, first));
            }
        }
        ASSERT_EQ(held_pages, (std::vector<std::int32_t>{1, 21, 33, 41, 61}));
        ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 10);
        // The page under the tail, one of the two that move.
        const CacheKey moved_key = *index.KeyOf(pool, hits.back());
        const PrefixCacheIndex::CachedBlockMetadata moved_before = *index.MetadataFor(pool, hits.back());
        const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator);
        const BoundaryTree::Node* moved_node = tree.Find(moved_key.content_hash);
        ASSERT_NE(moved_node, nullptr);

        std::vector<BlockTable> tables(static_cast<std::size_t>(coordinator.NumGroups()));
        const std::vector<GroupDemand> demands =
            MakeGroupDemands(tables, config_.cache_groups, GroupDemand{.extent = DenseGrowth{48}},
                             StateStep{.kind = StateStep::Kind::kRotate});
        ASSERT_FALSE(coordinator.Admit(probe_now(), demands, RequestProgress{}, /*copy_in_blocks=*/{},
                                       /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                                       /*evict_blocks_past_segments=*/true, /*session_id=*/{}));
        const HostCacheStats before = scheduler_->GetHostCacheStats();
        const std::optional<CacheCoordinator::AdmissionResult> compacted =
            coordinator.CompactLoneRequest(probe_now(), demands, RequestProgress{},
                                           /*copy_in_blocks=*/{}, /*copy_in_point=*/std::nullopt);
        ASSERT_TRUE(compacted);
        EXPECT_EQ(compacted->pages_to_copy[0],
                  (std::vector<std::pair<std::int32_t, std::int32_t>>{{57, 49}, {58, 66}}));
        EXPECT_TRUE(compacted->pages_to_copy[1].empty());
        for (const std::vector<std::int32_t>& zeroed : compacted->new_page_ids) {
            EXPECT_TRUE(zeroed.empty());
        }
        // The coordinator counts the compaction and its moves, whoever calls it.
        const HostCacheStats after = scheduler_->GetHostCacheStats();
        EXPECT_EQ(after.lone_compactions - before.lone_compactions, 1u);
        EXPECT_EQ(after.lone_compaction_moves - before.lone_compaction_moves, 2u);
        EXPECT_EQ(after.device_evictions.compaction.blocks - before.device_evictions.compaction.blocks, 1u);
        EXPECT_EQ(scheduler_->EmptyLcmBlocks(), 11);
        EXPECT_EQ(scheduler_->TailEntries(), 1u);
        EXPECT_EQ(index.KeyOf(pool, tail_at.front()), tail_key);
        // The moved page's entry went with its copy, its node untouched.
        const std::optional<PrefixCacheIndex::CachedBlockSnapshot> moved_after = index.Inspect(pool, moved_key);
        ASSERT_TRUE(moved_after);
        EXPECT_EQ(pool.ResolveCacheBlockId(0, moved_after->location), 66);
        EXPECT_EQ(moved_after->metadata.max_access_epoch, moved_before.max_access_epoch);
        EXPECT_EQ(tree.Find(moved_key.content_hash), moved_node);
        EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(coordinator).empty());

        const std::optional<CacheCoordinator::AdmissionResult> admitted =
            coordinator.Admit(probe_now(), demands, RequestProgress{}, /*copy_in_blocks=*/{},
                              /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                              /*evict_blocks_past_segments=*/true, /*session_id=*/{});
        ASSERT_TRUE(admitted);
        EXPECT_EQ(admitted->device_prefix_tokens, 94);
        EXPECT_TRUE(
            std::ranges::any_of(admitted->pages_to_copy[0],
                                [](const std::pair<std::int32_t, std::int32_t>& copy) { return copy.first == 50; }));
        EXPECT_EQ(tables[0].NumBlocks(), 18);  // 11 hit pages, the tail's copy, 48 new tokens
    }
}

// At a first admission the probe's tail block moves like a hit page when its
// parent is one to vacate, and the retry's private copy of it reads where it
// was. On ten parents X, a 43-token prompt, puts its prompt end in the hole
// beside b's pages and its five pages elsewhere. R, X's prompt and one token
// more, alone, hits X's five pages and its 3-token tail (page 24); its
// sliding blocks need two parents beyond the one empty, so the tail's
// parent must be freed, and the tail block alone binds it. The compaction
// moves the tail into a kept parent with its entry: into the slot of a0's
// page 1 (cleared) without a Host tier; under write-back, where X's pages
// took other parents, into page 9, a hole. The retry admits R in the same
// plan and copies the tail into its own block from page 24, not from the
// moved block, whose copy lands with it.
TEST_F(LoneRequestCompactionSuite, FirstAdmissionCompactionMovesTheProbesTailBlock) {
    for (const bool write_back : {false, true}) {
        SCOPED_TRACE(write_back ? "write-back" : "no Host tier");
        LeaveATailBesideCache(/*parents=*/10, /*history_block_tokens=*/kP, /*x_tokens=*/43, write_back);
        CacheCoordinator& coordinator = SchedulerTestAccess::Coordinator(*scheduler_);
        const BlockPool& pool = SchedulerTestAccess::Pool(*scheduler_);
        const PrefixCacheIndex& index = coordinator.GroupPrefixIndex(0);
        const std::vector<std::int32_t> prompt = MakeTokens(44, 20000);
        const CacheCoordinator::PrefixProbe probe = coordinator.ProbePromptPrefix(CacheCoordinator::PromptPrefix{
            .prefix_hashes = scheduler_->PrefixHashesForTokens(prompt),
            .tokens = prompt,
            .max_tokens = static_cast<std::int32_t>(prompt.size()) - 1,
        });
        ASSERT_EQ(probe.device.num_common_tokens, 43);
        ASSERT_TRUE(probe.device.tail);
        const std::vector<CacheBlockLocation> tail_at = index.MatchedLocations(
            pool, probe.device.tail->group_keys[0], /*begin_blocks=*/0, probe.device.tail->per_group[0]);
        ASSERT_EQ(tail_at.size(), 1u);
        ASSERT_EQ(pool.ResolveCacheBlockId(0, tail_at.front()), 24);
        const CacheKey tail_key = *index.KeyOf(pool, tail_at.front());

        Submit(RequestSpec{.request_id = "R", .tokens = prompt, .max_new_tokens = 1});
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{"R"});
        EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{43});
        const std::int32_t moved_to = write_back ? 9 : 1;
        EXPECT_EQ(plan.pages_to_copy.at("full_attention"),
                  (std::vector<std::pair<std::int32_t, std::int32_t>>{{24, moved_to}, {24, write_back ? 19 : 2}}));
        // The sliding group's copy of its own tail block, which stays.
        EXPECT_EQ(
            plan.pages_to_copy.at("sliding"),
            (std::vector<std::pair<std::int32_t, std::int32_t>>{write_back ? std::pair{10, 2} : std::pair{5, 10}}));
        const HostCacheStats stats = scheduler_->GetHostCacheStats();
        EXPECT_EQ(stats.lone_compactions, 1u);
        EXPECT_EQ(stats.lone_compaction_moves, 1u);
        EXPECT_EQ(stats.device_evictions.compaction.blocks, write_back ? 0u : 1u);
        const std::optional<CacheBlockLocation> tail_now =
            CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, tail_key);
        ASSERT_TRUE(tail_now);
        EXPECT_EQ(pool.ResolveCacheBlockId(0, *tail_now), moved_to);
        EXPECT_EQ(scheduler_->TailEntries(), 1u);
        AckWriteBacks(plan);
        Land(plan);
        AckWriteBacks(PlanOnce());
        EXPECT_EQ(scheduler_->RequestTokenSize("R"), -1);
    }
}

// The plan a compaction ran in admits no later prompt: a block it moved
// that nothing holds -- the tail block once the retry copied it, a hit page
// of a retry that failed -- gets its bytes from the plan's copies, which the
// runtime runs after the plan's stores, so a later admission must not evict
// it. The layout above without a Host tier, with a1's page 3 held (as a
// store ticket would hold it), so R's retry still fails after the tail
// moved to page 1; S, a one-page prompt, waits behind R. The compaction's
// plan copies the tail and runs nothing; S is admitted in the next plan.
TEST_F(LoneRequestCompactionSuite, CompactionPlanAdmitsNoLaterPrompt) {
    LeaveATailBesideCache(/*parents=*/10, /*history_block_tokens=*/kP, /*x_tokens=*/43, /*write_back=*/false);
    CacheCoordinator& coordinator = SchedulerTestAccess::Coordinator(*scheduler_);
    const BlockPool& pool = SchedulerTestAccess::Pool(*scheduler_);
    const PrefixCacheIndex& index = coordinator.GroupPrefixIndex(0);
    const CacheBlockLocation page3{.lcm_block_id = 1, .slot_index = 2};
    ASSERT_EQ(pool.ResolveCacheBlockId(0, page3), 3);
    CacheBlockRef held = index.Find(pool, *index.KeyOf(pool, page3));
    ASSERT_TRUE(held);
    Submit(RequestSpec{.request_id = "R", .tokens = MakeTokens(44, 20000), .max_new_tokens = 1});
    Submit(RequestSpec{.request_id = "S", .tokens = MakeTokens(kP, 60000), .max_new_tokens = 1});

    const ExecutionPlan own = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(own);
    EXPECT_TRUE(batch == nullptr || batch->request_ids.empty());
    EXPECT_EQ(own.pages_to_copy.at("full_attention"), (std::vector<std::pair<std::int32_t, std::int32_t>>{{24, 1}}));
    for (const auto& [group, pages] : own.pages_to_zero) {
        EXPECT_TRUE(pages.empty()) << group;
    }
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 1u);

    const ExecutionPlan next = PlanOnce();
    batch = FindForwardBatch(next);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, std::vector<std::string>{"S"});
    Land(next);
    PlanOnce();
    EXPECT_EQ(scheduler_->RequestTokenSize("S"), -1);
}

// At most one of the probe's tail blocks moves per compaction: the retry
// copies each from where it was, and two whose copies land on each other's
// old places would need their bytes swapped. On thirteen parents with
// history blocks of half a page, X (22 tokens) ends in a 6-token tail of two
// blocks, which take the holes a0 and b left (pages 3 and 14), each beside
// cache. With a cached page held in every other parent the probe does not
// reach (as a store ticket would hold it), R, X's prompt and three tokens
// more, alone, needs both tail parents freed. Its first compaction moves
// the second tail block (14 -> 11) and the retry still fails; the next
// plan's moves the first (3 -> 12) and the retry admits R, copying each tail
// block from where it was in that plan: the first from page 3, the second
// from page 11.
TEST_F(LoneRequestCompactionSuite, MovesOneProbeTailBlockPerCompaction) {
    LeaveATailBesideCache(/*parents=*/13, /*history_block_tokens=*/kP / 2, /*x_tokens=*/22, /*write_back=*/false);
    CacheCoordinator& coordinator = SchedulerTestAccess::Coordinator(*scheduler_);
    const BlockPool& pool = SchedulerTestAccess::Pool(*scheduler_);
    const std::vector<std::int32_t> prompt = MakeTokens(25, 20000);
    const CacheCoordinator::PrefixProbe probe = coordinator.ProbePromptPrefix(CacheCoordinator::PromptPrefix{
        .prefix_hashes = scheduler_->PrefixHashesForTokens(prompt),
        .tokens = prompt,
        .max_tokens = static_cast<std::int32_t>(prompt.size()) - 1,
    });
    ASSERT_EQ(probe.device.num_common_tokens, 22);
    ASSERT_TRUE(probe.device.tail);
    std::vector<CacheBlockRef> held;
    std::vector<CacheKey> tail_keys;
    for (std::uint32_t group = 0; group < 2; ++group) {
        const PrefixCacheIndex& index = coordinator.GroupPrefixIndex(static_cast<std::int32_t>(group));
        std::vector<CacheBlockLocation> reached =
            index.MatchedLocations(pool, (*probe.group_keys)[group], /*begin_blocks=*/0, probe.device.per_group[group]);
        const std::vector<CacheBlockLocation> tail_at = index.MatchedLocations(
            pool, probe.device.tail->group_keys[group], /*begin_blocks=*/0, probe.device.tail->per_group[group]);
        reached.insert(reached.end(), tail_at.begin(), tail_at.end());
        if (group == 0) {
            ASSERT_EQ(tail_at.size(), 2u);
            for (const CacheBlockLocation location : tail_at) {
                tail_keys.push_back(*index.KeyOf(pool, location));
            }
            EXPECT_EQ(pool.ResolveCacheBlockId(0, tail_at[0]), 3);
            EXPECT_EQ(pool.ResolveCacheBlockId(0, tail_at[1]), 14);
        }
        for (std::int32_t parent = 1; parent <= pool.NumLcmBlocks(); ++parent) {
            if (pool.BoundGroup(parent) != std::optional<std::uint32_t>{group} ||
                std::ranges::any_of(reached, [&](CacheBlockLocation at) { return at.lcm_block_id == parent; })) {
                continue;
            }
            for (std::int32_t slot = 0; slot < pool.Packing(group); ++slot) {
                if (const std::optional<CacheKey> key =
                        index.KeyOf(pool, CacheBlockLocation{.lcm_block_id = parent, .slot_index = slot})) {
                    held.push_back(index.Find(pool, *key));
                    break;
                }
            }
        }
    }
    const auto tail_pages = [&] {
        std::vector<std::int32_t> pages;
        for (const CacheKey& key : tail_keys) {
            pages.push_back(
                pool.ResolveCacheBlockId(0, *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, key)));
        }
        return pages;
    };
    Submit(RequestSpec{.request_id = "R", .tokens = prompt, .max_new_tokens = 1});

    const ExecutionPlan first = PlanOnce();
    EXPECT_TRUE(FindForwardBatch(first)->request_ids.empty());
    EXPECT_EQ(first.pages_to_copy.at("full_attention"), (std::vector<std::pair<std::int32_t, std::int32_t>>{{14, 11}}));
    EXPECT_EQ(tail_pages(), (std::vector<std::int32_t>{3, 11}));
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 1u);

    const ExecutionPlan second = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(second);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"R"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{22});
    EXPECT_EQ(second.pages_to_copy.at("full_attention"),
              (std::vector<std::pair<std::int32_t, std::int32_t>>{{3, 12}, {3, 27}, {11, 28}}));
    EXPECT_EQ(tail_pages(), (std::vector<std::int32_t>{12, 11}));
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 2u);
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compaction_moves, 2u);
    Land(second);
    PlanOnce();
    EXPECT_EQ(scheduler_->RequestTokenSize("R"), -1);
}

// On the D role the probe leaves a tail's partial KV block a hole for the
// peer to fill whole, and claims each State group's checkpoint at the prompt
// end (state_at_prompt_end), which sits at that tail's hash: a compaction may
// clear the hole's slot as a destination, and that eviction takes the tail's
// other entries along -- the claimed checkpoint among them, unless held.
// Every Device block the probe hit is held through the compaction, so the
// retry still finds it. Eight parents of a history group of two 4-token
// blocks per parent beside a State group; eight one-page prompts, then X (41
// tokens: five pages and a one-token tail), each landed by the peer and
// finished on its first decode. R, the same prompt, hits X's pages and
// claims the checkpoint; its admission fails, the compaction moves a hit
// page into the slot of the tail's history block, evicting that block alone,
// and the retry admits R.
TEST_F(LoneRequestCompactionSuite, DRoleCompactionHoldsThePromptEndCheckpointItClaims) {
    config_ = MakeConfig();
    config_.role = Role::kD;
    config_.device_allocator.total_pages = 9;
    config_.cache_groups[0].block_granularity = 4;
    config_.cache_groups[0].total_pages = 9 * 2 * 2;
    config_.cache_groups[1] = CacheGroupConfig{.group_id = "state",
                                               .block_granularity = 0,
                                               .total_pages = 9,
                                               .attn_kind = AttnKind::kState,
                                               .transfer_policy = CacheTransferPolicy::LatestSnapshot};
    scheduler_ = std::make_unique<Scheduler>(config_);
    ASSERT_EQ(scheduler_->MaxSingleRequestTokens(), 44);
    // The peer's prefill lands with the first token, and the request
    // finishes on its first decode.
    const auto run = [&](const std::vector<RequestSpec>& prompts) {
        Submit(prompts);
        ExecutionEvent bootstrapped;
        for (const RequestSpec& spec : prompts) {
            bootstrapped.With(pd::BootstrappedEvent{spec.request_id});
        }
        scheduler_->Advance(bootstrapped);
        for (int plan = 0; plan < 32 && scheduler_->RequestTokenSize(prompts.back().request_id) != -1; ++plan) {
            const ExecutionPlan next = PlanOnce();
            ExecutionEvent landed;
            if (next.remote_prefill) {
                for (const std::string& id : next.remote_prefill->request_ids) {
                    landed.With(pd::RemotePrefillDoneEvent{id, ++next_token_});
                    landed.With(forward::Finish{.request_id = id});
                }
            }
            for (const std::string& id : FindForwardBatch(next)->request_ids) {
                landed.With(forward::ExtendResult{.request_id = id, .tokens = {++next_token_}});
                landed.With(forward::Finish{.request_id = id});
            }
            scheduler_->Advance(landed);
        }
        ASSERT_EQ(scheduler_->RequestTokenSize(prompts.back().request_id), -1);
    };
    std::vector<RequestSpec> pages;
    for (std::int32_t k = 0; k < 8; ++k) {
        pages.push_back(RequestSpec{
            .request_id = "a" + std::to_string(k), .tokens = MakeTokens(kP, 100 + 50 * k), .max_new_tokens = 1});
    }
    run(pages);
    const std::vector<std::int32_t> prompt = MakeTokens(41, 3000);
    run({RequestSpec{.request_id = "X", .tokens = prompt, .max_new_tokens = 1}});
    EXPECT_EQ(scheduler_->GetHostCacheStats().lone_compactions, 0u);
    run({RequestSpec{.request_id = "R", .tokens = prompt, .max_new_tokens = 1}});
    const HostCacheStats stats = scheduler_->GetHostCacheStats();
    EXPECT_EQ(stats.lone_compactions, 1u);
    EXPECT_EQ(stats.lone_compaction_moves, 1u);
    EXPECT_EQ(stats.device_evictions.compaction.blocks, 1u);
}

// The plan's copies compose as they are recorded
// (Scheduler::recordAdmittedPages): the runtime reads every source as the
// plan found it, so a copy from a block an earlier copy of the plan fills
// reads that copy's source, through any chain; one composed back onto its
// own source is dropped; a copy that overwrites an earlier copy's source
// stays as it is (read before it is overwritten); a destination is written
// once, and never zeroed in the plan, whichever of the two is recorded
// first: the runtime copies before it zeroes, so the zero would wipe the
// copy. A copy's source may be zeroed (read before it is), and a page of
// another group is another page.
TEST_F(LoneRequestCompactionSuite, ComposesACopyFromABlockTheSamePlanFills) {
    using Copies = std::vector<std::pair<std::int32_t, std::int32_t>>;
    const auto record = [&](ExecutionPlan& plan, Copies copies, std::vector<std::int32_t> zeroes = {},
                            std::vector<std::int32_t> sliding_zeroes = {}) {
        SchedulerTestAccess::RecordAdmittedPages(
            *scheduler_, plan,
            CacheCoordinator::AdmissionResult{
                .new_page_ids = std::vector<std::vector<std::int32_t>>{std::move(zeroes), std::move(sliding_zeroes)},
                .pages_to_copy = std::vector<Copies>{std::move(copies), {}},
            });
        return plan.pages_to_copy.at("full_attention");
    };
    {
        ExecutionPlan plan;
        record(plan, {{24, 1}});
        EXPECT_EQ(record(plan, {{1, 2}, {7, 8}}), (Copies{{24, 1}, {24, 2}, {7, 8}}));
        EXPECT_TRUE(plan.pages_to_copy.at("sliding").empty());
    }
    {
        ExecutionPlan plan;
        record(plan, {{24, 1}});
        EXPECT_EQ(record(plan, {{1, 24}}), (Copies{{24, 1}}));
    }
    {
        ExecutionPlan plan;
        record(plan, {{24, 1}});
        record(plan, {{1, 2}});
        EXPECT_EQ(record(plan, {{2, 3}}), (Copies{{24, 1}, {24, 2}, {24, 3}}));
    }
    {
        ExecutionPlan plan;
        record(plan, {{24, 1}});
        EXPECT_EQ(record(plan, {{5, 24}}), (Copies{{24, 1}, {5, 24}}));
    }
    {
        ExecutionPlan plan;
        record(plan, {{24, 1}});
        EXPECT_THROW(record(plan, {{9, 1}}), std::runtime_error);
    }
    {
        ExecutionPlan plan;
        record(plan, {{24, 1}});
        EXPECT_THROW(record(plan, {}, {1}), std::runtime_error);
    }
    {
        ExecutionPlan plan;
        record(plan, {}, {1});
        EXPECT_THROW(record(plan, {{24, 1}}), std::runtime_error);
    }
    {
        ExecutionPlan plan;
        EXPECT_THROW(record(plan, {{24, 1}}, {1}), std::runtime_error);
    }
    {
        ExecutionPlan plan;
        EXPECT_EQ(record(plan, {{24, 1}}, {24}, {1}), (Copies{{24, 1}}));
        EXPECT_EQ(plan.pages_to_zero.at("full_attention"), std::vector<std::int32_t>{24});
        EXPECT_EQ(plan.pages_to_zero.at("sliding"), std::vector<std::int32_t>{1});
    }
}

// A retracted request alone whose restore would claim a cached page that
// binds a parent the rest of its image needs plans again claiming nothing.
// Six parents, Host L2 write-through, the knob every four plans. f's page 1
// stays cached in parent 1, and V's first block takes the hole beside it
// (page 2), its reserve parent 4, its sliding blocks parents 5 and 6. The
// knob retracts V at plan 4, where O is admitted into parents 2 to 6. O
// finishes on its first token, and the Host stores of the pages it published
// pin them until their ACK. At plan 5 V is alone: claiming page 2 keeps
// parent 1 bound, leaving one parent for its two sliding blocks, so the
// restore claims nothing and copies every slot back -- history into the
// holes beside O's pages (pages 4 and 6), sliding into parents 6 and 1 --
// without waiting for O's stores.
TEST_F(LoneRequestCompactionSuite, ALoneRestoreDropsClaimsThatLeaveNoPlan) {
    config_.device_allocator.total_pages = 7;
    config_.cache_groups[0].total_pages = 7 * 2;
    config_.cache_groups[1].total_pages = 7;
    config_.disable_l2_cache = false;
    config_.host_allocator.total_pages = 28;
    config_.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
    config_.max_batch_size = 8;
    config_.max_resident_requests = config_.max_batch_size;
    config_.debug_force_retraction_interval = 4;
    SetTestSnapshotPool(config_);
    config_ = WithHostPackingFromDevice(config_);
    scheduler_ = std::make_unique<Scheduler>(config_);
    Submit(RequestSpec{.request_id = "f", .tokens = MakeTokens(kP, 100), .max_new_tokens = 1});
    const ExecutionPlan f_prefill = PlanOnce();
    AckWriteBacks(f_prefill);
    Land(f_prefill);
    Submit(RequestSpec{.request_id = "V", .tokens = MakeTokens(kP, 3000), .max_new_tokens = 0});
    for (int plan = 2; plan <= 3; ++plan) {
        const ExecutionPlan next = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(next);
        ASSERT_NE(batch, nullptr) << plan;
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{"V"}) << plan;
        if (plan == 2) {
            EXPECT_EQ(batch->block_tables.at("full_attention").AbsoluteRows(), (std::vector<std::int32_t>{2, 7}));
            EXPECT_EQ(batch->block_tables.at("sliding").AbsoluteRows(), (std::vector<std::int32_t>{5, 6}));
        }
        AckWriteBacks(next);
        SendForwardDone("V", {++next_token_});
    }
    Submit(RequestSpec{.request_id = "O", .tokens = MakeTokens(kP + 1, 9000), .max_new_tokens = 1});
    const ExecutionPlan retraction = PlanOnce();
    ASSERT_NE(FindSnapshotStore(retraction), nullptr);
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    ASSERT_NE(FindForwardBatch(retraction), nullptr);
    ASSERT_EQ(FindForwardBatch(retraction)->request_ids, std::vector<std::string>{"O"});
    AckImageStores(retraction);
    Land(retraction);

    const ExecutionPlan alone = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(alone);
    ASSERT_NE(restore, nullptr) << "a lone restore does not wait for O's stores";
    EXPECT_EQ(restore->group_ids.front(), (std::vector<std::uint32_t>{0, 0, 1, 1}));
    EXPECT_EQ(restore->dst_pages.front(), (std::vector<std::int32_t>{4, 6, 6, 1}));
    EXPECT_EQ(restore->source_tiers.front(), (std::vector<std::uint8_t>{0, 0, 0, 0})) << "all from Host L2";
}

// A copy-in block the request holds re-places within its bucket and is never
// zeroed: the copy-in fills it whole, and the plan that spends it may be the
// compaction's own, whose copies the runtime runs before its zeroes. The call
// sequence of admitWithKvEventTracking at a first decode, on the coordinator:
// four history blocks per parent, one State block per parent, nine parents.
// Seven tables of four history blocks publish three pages (the first two) or
// one (the rest) and are freed, so parents 1 and 2 keep three cached pages
// and parents 3 to 7 one. R admits 126 tokens with the copy-in point at its
// prompt end: the prompt end's block and its copy-in block take the holes of
// the two three-page parents, the 15 dense blocks fill parents 3 to 7, the
// State slots parents 8 and 9; chunk 2's admission publishes pages 0 to 7.
// With State slot 0 held elsewhere (a store ticket, another request's
// retained checkpoint), the first decode's rotation needs a parent nothing
// can free: the compaction vacates the copy-in block's parent -- the one
// single-block parent with nothing fixed -- into a cleared slot of parent 1
// (page 17) without a zero, and the retry's copy-in fills that page.
TEST(LoneRequestCopyInCompactionTest, RePlacesAHeldCopyInBlockWithoutZeroingIt) {
    BlockPool pool(/*num_lcm_blocks=*/9, {4, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = kP},
            CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}},
        kP, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    const auto page_hashes = [](const std::vector<std::int32_t>& tokens, std::int32_t pages) {
        std::vector<std::span<const std::int32_t>> spans;
        for (std::int32_t page = 0; page < pages; ++page) {
            spans.emplace_back(tokens.data() + page * kP, static_cast<std::size_t>(kP));
        }
        return ComputePrefixHashes(spans, "");
    };
    std::vector<std::vector<BlockTable>> freed;
    for (std::int32_t k = 0; k < 7; ++k) {
        freed.emplace_back(2);
        ASSERT_TRUE(AdmitForTest(coordinator, freed.back(), coordinator.ProbePrefix({}),
                                 GroupDemand{.extent = DenseGrowth{32}}, RequestProgress{},
                                 StateStep{StateStep::Kind::kKeep}, /*copy_in_blocks=*/{},
                                 /*copy_in_point=*/std::nullopt))
            << k;
        const std::vector<std::string> hashes = page_hashes(MakeTokens(32, 1000 + 100 * k), 4);
        coordinator.CacheFullBlocks(freed.back(), std::span<const std::string>(hashes).first(k < 2 ? 3 : 1),
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator), /*first_slot=*/0,
                                    CacheBoundaryKind::kChunk);
    }
    for (std::vector<BlockTable>& tables : freed) {
        coordinator.Free(tables);
    }
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 2);

    const std::vector<std::int32_t> prompt = MakeTokens(126, 3000);
    std::vector<BlockTable> r(2);
    std::optional<CacheCoordinator::AdmissionResult> admitted = AdmitForTest(
        coordinator, r, coordinator.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{126}}, RequestProgress{},
        StateStep{StateStep::Kind::kRotate}, /*copy_in_blocks=*/{}, CacheCoordinator::CopyInPoint{.tokens = 126});
    ASSERT_TRUE(admitted);
    std::vector<std::vector<CacheBlockRef>> copy_in = std::move(admitted->copy_in_blocks);
    ASSERT_EQ(copy_in.size(), 2u);
    ASSERT_EQ(copy_in[0].size(), 1u);
    ASSERT_TRUE(copy_in[0][0]);
    const GroupAllocator& allocator = coordinator.Allocator(0);
    EXPECT_EQ(allocator.ResolveCacheBlockId(*r[0].Blocks().back()), 4);
    EXPECT_EQ(allocator.ResolveCacheBlockId(*copy_in[0][0]), 8);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);
    const std::vector<std::string> hashes = page_hashes(prompt, 15);
    coordinator.CacheFullBlocks(r, std::span<const std::string>(hashes).first(8), admitted->access_epoch,
                                /*first_slot=*/0, CacheBoundaryKind::kChunk);
    const CacheBlockRef held_state_input = r[1].Blocks()[0];

    const TailPublication tail{
        .tail_tokens = 6, .hash = HashPrefixPage(std::span<const std::int32_t>(prompt.data() + 120, 6), hashes.back())};
    const RequestProgress progress{
        .completed_pages = CompletedPages{.tokens = std::span<const std::int32_t>(prompt.data(), 126),
                                          .prefix_hashes = hashes,
                                          .first_new_prefix_page = 8,
                                          .tail = tail,
                                          .owner_tail = OwnerTail::kCopiedIn,
                                          .publish_state = false,
                                          .boundary_kind = CacheBoundaryKind::kEndpoint},
        .num_computed_tokens = 126,
    };
    std::vector<GroupDemand> demands(r.size(), GroupDemand{.extent = DenseGrowth{1}});
    for (std::size_t i = 0; i < r.size(); ++i) {
        demands[i].table = &r[i];
        if (coordinator.GroupIsState(i)) {
            demands[i].extent = StateStep{.kind = StateStep::Kind::kRotate};
        }
    }
    ASSERT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, copy_in,
                                   /*copy_in_point=*/std::nullopt, admitted->access_epoch,
                                   /*evict_blocks_past_segments=*/true, /*session_id=*/{}));
    const std::optional<CacheCoordinator::AdmissionResult> compacted = coordinator.CompactLoneRequest(
        coordinator.ProbePrefix({}), demands, progress, copy_in, /*copy_in_point=*/std::nullopt);
    ASSERT_TRUE(compacted);
    EXPECT_EQ(allocator.ResolveCacheBlockId(*copy_in[0][0]), 17);
    for (std::size_t i = 0; i < compacted->new_page_ids.size(); ++i) {
        EXPECT_TRUE(compacted->new_page_ids[i].empty()) << i;
        EXPECT_TRUE(compacted->pages_to_copy[i].empty()) << i;
    }
    EXPECT_EQ(coordinator.LoneCompactions(), 1u);
    EXPECT_EQ(coordinator.LoneCompactionMoves(), 1u);
    EXPECT_EQ(coordinator.DeviceEvictions().compaction.blocks, 1u);
    const std::optional<CacheCoordinator::AdmissionResult> retried =
        coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, copy_in, /*copy_in_point=*/std::nullopt,
                          admitted->access_epoch, /*evict_blocks_past_segments=*/true, /*session_id=*/{});
    ASSERT_TRUE(retried);
    EXPECT_EQ(retried->pages_to_copy[0], (std::vector<std::pair<std::int32_t, std::int32_t>>{{4, 17}}));
    // The plan carries both results: no page one of them copies into is
    // zeroed by either.
    for (const auto& [source, destination] : retried->pages_to_copy[0]) {
        EXPECT_EQ(std::ranges::find(compacted->new_page_ids[0], destination), compacted->new_page_ids[0].end());
        EXPECT_EQ(std::ranges::find(retried->new_page_ids[0], destination), retried->new_page_ids[0].end());
    }
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(coordinator).empty());
}

// A parent whose other block is the request's own expired window block is
// vacated like one beside evictable cache: the retry reclaims that block
// before it evicts, so the parent empties once the request's live block
// leaves it. On the coordinator: a sliding-window group of two 8-token
// blocks per parent (a 16-token window), a State group, four parents. G's
// two pages stay cached in parent 1. R's first 16 tokens take parent 4
// (pages 7 and 8) and two State parents; its next 7 tokens place slot 2 at
// page 1, evicting G's first page. At 23 computed tokens slot 0 has slid
// out of the window, and with its State input held elsewhere (a store
// ticket, another request's retained checkpoint) the decode's rotation
// needs a parent nothing frees: the compaction keeps parent 1 and moves
// slot 1 (page 8) into the slot of G's cached page 2 (cleared), and the
// retry reclaims slot 0 and rotates into parent 4.
TEST(LoneRequestWindowCompactionTest, VacatesAParentBesideAnExpiredBlockTheRetryReclaims) {
    BlockPool pool(/*num_lcm_blocks=*/4, {2, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                           .sliding_window = 16,
                           .cache_blocks_per_lcm_block = 2,
                           .block_granularity = kP},
            CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}},
        kP, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    std::vector<BlockTable> g(2);
    ASSERT_TRUE(AdmitForTest(coordinator, g, GroupDemand{.extent = DenseGrowth{16}}));
    const std::vector<std::int32_t> g_tokens = MakeTokens(16, 1000);
    const std::vector<std::string> g_hashes = ComputePrefixHashes(
        std::vector<std::span<const std::int32_t>>{std::span(g_tokens).first(kP), std::span(g_tokens).subspan(kP)}, "");
    CacheFullBlocksForTest(coordinator, g, g_hashes);
    coordinator.Free(g);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 3);

    std::vector<BlockTable> r(2);
    ASSERT_TRUE(AdmitForTest(coordinator, r, GroupDemand{.extent = DenseGrowth{16}}));
    ASSERT_TRUE(AdmitForTest(coordinator, r, GroupDemand{.extent = DenseGrowth{7}},
                             RequestProgress{.num_computed_tokens = 16}));
    const GroupAllocator& allocator = coordinator.Allocator(0);
    std::vector<std::int32_t> pages;
    for (const CacheBlockRef& block : r[0].Blocks()) {
        pages.push_back(allocator.ResolveCacheBlockId(*block));
    }
    ASSERT_EQ(pages, (std::vector<std::int32_t>{7, 8, 1}));
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);
    const CacheBlockRef held_state_input = r[1].Blocks()[0];

    const std::vector<GroupDemand> demands{
        GroupDemand{.table = &r[0], .extent = DenseGrowth{1}},
        GroupDemand{.table = &r[1], .extent = StateStep{.kind = StateStep::Kind::kRotate}}};
    const RequestProgress progress{.num_computed_tokens = 23};
    ASSERT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{},
                                   /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                                   /*evict_blocks_past_segments=*/true, /*session_id=*/{}));
    const std::optional<CacheCoordinator::AdmissionResult> compacted = coordinator.CompactLoneRequest(
        coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{}, /*copy_in_point=*/std::nullopt);
    ASSERT_TRUE(compacted);
    EXPECT_EQ(compacted->pages_to_copy[0], (std::vector<std::pair<std::int32_t, std::int32_t>>{{8, 2}}));
    EXPECT_TRUE(compacted->pages_to_copy[1].empty());
    for (const std::vector<std::int32_t>& zeroed : compacted->new_page_ids) {
        EXPECT_TRUE(zeroed.empty());
    }
    EXPECT_EQ(coordinator.LoneCompactions(), 1u);
    EXPECT_EQ(coordinator.LoneCompactionMoves(), 1u);
    EXPECT_EQ(coordinator.DeviceEvictions().compaction.blocks, 1u);
    EXPECT_EQ(pool.OccupiedCount(4), 1);  // slot 0's expired block alone

    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{},
                                  /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                                  /*evict_blocks_past_segments=*/true, /*session_id=*/{}));
    EXPECT_FALSE(r[0].Blocks()[0]);
    EXPECT_EQ(pool.BoundGroup(4), std::optional<std::uint32_t>{1u});
    EXPECT_EQ(coordinator.Allocator(1).ResolveCacheBlockId(*r[1].Blocks()[1]), 4);
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(coordinator).empty());
}

// A free block one vacate round displaced can be displaced again by the
// next, and the page it leaves then takes that round's copy: only the page
// it ends on is zeroed (the runtime copies before it zeroes). Under KV
// parallelism with four buckets, one slot of each per parent: a group of
// four blocks per parent beside a group of one, 13 parents. R's 64 tokens
// take parents 1 and 2 (pages 1-8) and parents 3-10. Its written blocks of
// slots 4 and 5 are then placed alone in parents 13 (page 50, bucket 1) and
// 12 (page 45, bucket 0), slot 6 at page 8 and its unwritten slot 7 at page
// 6 (bucket 1), and parent 11 keeps a block another holder holds. At 56
// computed tokens a 16-token growth needs two parents of the second group,
// and none is free. The compaction keeps parents 1 and 2: slot 4 takes page
// 6, slot 7 giving it up for page 5; slot 5 then takes page 5, slot 7
// moving on to page 7, and the retry admits.
TEST(LoneRequestBucketCompactionTest, ZeroesOnlyThePageADisplacedBlockEndsOn) {
    BlockPool pool(/*num_lcm_blocks=*/13, {4, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            CacheGroupSpec{.attn_kind = AttnKind::kFull,
                           .cache_blocks_per_lcm_block = 4,
                           .block_granularity = kP,
                           .shard_count = 4},
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    std::vector<BlockTable> r(2);
    ASSERT_TRUE(AdmitForTest(coordinator, r, GroupDemand{.extent = DenseGrowth{64}}));
    GroupAllocator& allocator = coordinator.Allocator(0);
    ASSERT_EQ(allocator.BlockTablePageIds(r[0]), (std::vector<std::int32_t>{1, 2, 3, 4, 5, 6, 7, 8}));
    // Parents 11 to 13, held whole, are bound to the first group while R's
    // blocks are placed into them.
    std::vector<CacheBlockRef> held = pool.AcquireUpToBlocks(0, 12);
    ASSERT_EQ(held.size(), 12u);
    const auto place = [&](std::int32_t slot, CacheBlockLocation destination) {
        std::erase_if(held, [&](const CacheBlockRef& block) { return block->Location() == destination; });
        allocator.SwapInFreshBlock(pool, r[0], slot, r[0].AvailableTokens(), destination);
    };
    place(5, CacheBlockLocation{.lcm_block_id = 12, .slot_index = 0});
    place(4, CacheBlockLocation{.lcm_block_id = 13, .slot_index = 1});
    place(7, CacheBlockLocation{.lcm_block_id = 2, .slot_index = 1});
    place(6, CacheBlockLocation{.lcm_block_id = 2, .slot_index = 3});
    std::erase_if(held, [](const CacheBlockRef& block) {
        return block->Location() != CacheBlockLocation{.lcm_block_id = 11, .slot_index = 0};
    });
    ASSERT_EQ(allocator.BlockTablePageIds(r[0]), (std::vector<std::int32_t>{1, 2, 3, 4, 50, 45, 8, 6}));
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);

    const std::vector<GroupDemand> demands{GroupDemand{.table = &r[0], .extent = DenseGrowth{16}},
                                           GroupDemand{.table = &r[1], .extent = DenseGrowth{16}}};
    const RequestProgress progress{.num_computed_tokens = 56};
    ASSERT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{},
                                   /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                                   /*evict_blocks_past_segments=*/true, /*session_id=*/{}));
    const std::optional<CacheCoordinator::AdmissionResult> compacted = coordinator.CompactLoneRequest(
        coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{}, /*copy_in_point=*/std::nullopt);
    ASSERT_TRUE(compacted);
    EXPECT_EQ(compacted->pages_to_copy[0], (std::vector<std::pair<std::int32_t, std::int32_t>>{{50, 6}, {45, 5}}));
    EXPECT_EQ(compacted->new_page_ids[0], std::vector<std::int32_t>{7});
    EXPECT_EQ(allocator.BlockTablePageIds(r[0]), (std::vector<std::int32_t>{1, 2, 3, 4, 6, 5, 8, 7}));
    EXPECT_EQ(coordinator.LoneCompactionMoves(), 4u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);
    EXPECT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{},
                                  /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                                  /*evict_blocks_past_segments=*/true, /*session_id=*/{}));
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(coordinator).empty());
}

// Of parents holding equally many of the request's blocks, one holding only
// free blocks is vacated first: they re-place in any bucket, with no copy
// and no displacement. Two buckets, one slot of each per parent, five
// parents: a group of two blocks per parent beside a group of one, laid out
// by admissions and releases alone. Six one-block tables take parents 1 to
// 3; the first and the fifth leave bucket-0 holes, which R's first 16 tokens
// take (slots 0 and 1, pages 1 and 5) beside parents 4 and 5 for its second
// group; the third leaves page 3, which R's next 8 tokens take for slot 2;
// the others leave. At 16 computed tokens an 8-token growth needs a parent
// of the second group, and none is free. The compaction keeps parents 1 and
// 3, which hold a written block each, and vacates parent 2: unwritten slot 2
// moves to page 2 and is zeroed, nothing is copied, and the retry admits.
// Kept by id instead (parents 1 and 2), slot 1 found no bucket-0 slot and
// took page 3, which slot 2 gave up -- emptying parent 2 unless slot 2 moved
// beside it.
TEST(LoneRequestBucketCompactionTest, VacatesAParentOfFreeBlocksFirstAmongEquallyDenseOnes) {
    BlockPool pool(/*num_lcm_blocks=*/5, {2, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            CacheGroupSpec{.attn_kind = AttnKind::kFull,
                           .cache_blocks_per_lcm_block = 2,
                           .block_granularity = kP,
                           .shard_count = 2},
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    const auto admit = [&](std::vector<BlockTable>& tables, std::int32_t first, std::int32_t second,
                           const RequestProgress& progress) {
        const std::vector<GroupDemand> demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{first}},
                                               GroupDemand{.table = &tables[1], .extent = DenseGrowth{second}}};
        return coordinator
            .Admit(coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{},
                   /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                   /*evict_blocks_past_segments=*/true, /*session_id=*/{})
            .has_value();
    };
    std::vector<std::vector<BlockTable>> others(6, std::vector<BlockTable>(2));
    for (std::vector<BlockTable>& tables : others) {
        ASSERT_TRUE(admit(tables, kP, 0, RequestProgress{}));
    }
    const GroupAllocator& allocator = coordinator.Allocator(0);
    for (std::size_t k = 0; k < others.size(); ++k) {
        ASSERT_EQ(allocator.BlockTablePageIds(others[k][0]),
                  std::vector<std::int32_t>{static_cast<std::int32_t>(k) + 1})
            << k;
    }
    others[0] = std::vector<BlockTable>(2);
    others[4] = std::vector<BlockTable>(2);
    std::vector<BlockTable> r(2);
    ASSERT_TRUE(admit(r, 2 * kP, 2 * kP, RequestProgress{}));
    others[2] = std::vector<BlockTable>(2);
    ASSERT_TRUE(admit(r, kP, 0, RequestProgress{.num_computed_tokens = 2 * kP}));
    others.clear();
    ASSERT_EQ(allocator.BlockTablePageIds(r[0]), (std::vector<std::int32_t>{1, 5, 3}));
    ASSERT_EQ(coordinator.Allocator(1).BlockTablePageIds(r[1]), (std::vector<std::int32_t>{4, 5}));
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);

    const std::vector<GroupDemand> demands{GroupDemand{.table = &r[0], .extent = DenseGrowth{kP}},
                                           GroupDemand{.table = &r[1], .extent = DenseGrowth{kP}}};
    const RequestProgress progress{.num_computed_tokens = 2 * kP};
    ASSERT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{},
                                   /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                                   /*evict_blocks_past_segments=*/true, /*session_id=*/{}));
    const std::optional<CacheCoordinator::AdmissionResult> compacted = coordinator.CompactLoneRequest(
        coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{}, /*copy_in_point=*/std::nullopt);
    ASSERT_TRUE(compacted);
    EXPECT_TRUE(compacted->pages_to_copy[0].empty());
    EXPECT_EQ(compacted->new_page_ids[0], std::vector<std::int32_t>{2});
    EXPECT_EQ(allocator.BlockTablePageIds(r[0]), (std::vector<std::int32_t>{1, 5, 2}));
    EXPECT_EQ(coordinator.LoneCompactionMoves(), 1u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
    EXPECT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{},
                                  /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                                  /*evict_blocks_past_segments=*/true, /*session_id=*/{}));
    EXPECT_EQ(pool.BoundGroup(2), std::optional<std::uint32_t>{1u});
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(coordinator).empty());
}

// A free block that gives its slot up re-places in its own parent first: it
// leaves before the block taking the slot arrives. Two buckets, two slots of
// each per parent: a group of four blocks per parent beside a group of one,
// nine parents. R's 48 tokens take parent 1 and half of parent 2 (pages 1 to
// 6) and parents 3 to 8; its slot 3 is then placed alone in parent 9 (page
// 33), slot 4 at page 7 and its unwritten slot 5 at page 5. At 40 computed
// tokens an 8-token growth of the second group needs a parent, and none is
// free. The compaction keeps parents 1 and 2 and vacates parent 9: slot 3
// (written, bucket 0) finds no free bucket-0 slot in them, so slot 5 gives
// page 5 up and moves beside it to page 6 -- not to page 4, the free slot of
// parent 1, which the kept order reaches first -- and the retry admits.
TEST(LoneRequestBucketCompactionTest, ReplacesADisplacedBlockInItsOwnParentFirst) {
    BlockPool pool(/*num_lcm_blocks=*/9, {4, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            CacheGroupSpec{.attn_kind = AttnKind::kFull,
                           .cache_blocks_per_lcm_block = 4,
                           .block_granularity = kP,
                           .shard_count = 2},
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    std::vector<BlockTable> r(2);
    ASSERT_TRUE(AdmitForTest(coordinator, r, GroupDemand{.extent = DenseGrowth{6 * kP}}));
    GroupAllocator& allocator = coordinator.Allocator(0);
    ASSERT_EQ(allocator.BlockTablePageIds(r[0]), (std::vector<std::int32_t>{1, 2, 3, 4, 5, 6}));
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);
    // Parent 9 and the rest of parent 2, held, are bound to the first group
    // while R's blocks are placed into them.
    std::vector<CacheBlockRef> held = pool.AcquireUpToBlocks(0, 6);
    ASSERT_EQ(held.size(), 6u);
    const auto place = [&](std::int32_t slot, CacheBlockLocation destination) {
        std::erase_if(held, [&](const CacheBlockRef& block) { return block->Location() == destination; });
        allocator.SwapInFreshBlock(pool, r[0], slot, r[0].AvailableTokens(), destination);
    };
    place(3, CacheBlockLocation{.lcm_block_id = 9, .slot_index = 0});
    place(4, CacheBlockLocation{.lcm_block_id = 2, .slot_index = 2});
    place(5, CacheBlockLocation{.lcm_block_id = 2, .slot_index = 0});
    held.clear();
    ASSERT_EQ(allocator.BlockTablePageIds(r[0]), (std::vector<std::int32_t>{1, 2, 3, 33, 7, 5}));
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);

    const std::vector<GroupDemand> demands{GroupDemand{.table = &r[0], .extent = DenseGrowth{0}},
                                           GroupDemand{.table = &r[1], .extent = DenseGrowth{kP}}};
    const RequestProgress progress{.num_computed_tokens = 5 * kP};
    ASSERT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{},
                                   /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                                   /*evict_blocks_past_segments=*/true, /*session_id=*/{}));
    const std::optional<CacheCoordinator::AdmissionResult> compacted = coordinator.CompactLoneRequest(
        coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{}, /*copy_in_point=*/std::nullopt);
    ASSERT_TRUE(compacted);
    EXPECT_EQ(compacted->pages_to_copy[0], (std::vector<std::pair<std::int32_t, std::int32_t>>{{33, 5}}));
    EXPECT_EQ(compacted->new_page_ids[0], std::vector<std::int32_t>{6});
    EXPECT_EQ(allocator.BlockTablePageIds(r[0]), (std::vector<std::int32_t>{1, 2, 3, 5, 7, 6}));
    EXPECT_EQ(r[0].Blocks()[5]->Location().lcm_block_id, 2);
    EXPECT_EQ(coordinator.LoneCompactionMoves(), 2u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
    EXPECT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, /*copy_in_blocks=*/{},
                                  /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                                  /*evict_blocks_past_segments=*/true, /*session_id=*/{}));
    EXPECT_EQ(pool.BoundGroup(9), std::optional<std::uint32_t>{1u});
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(coordinator).empty());
}

}  // namespace
}  // namespace tokenspeed::test
