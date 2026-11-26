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

#include "integration_test_helper.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace tokenspeed::test {

class AdmissionPromptHashSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 9;
        cfg.cache_groups[0].total_pages = 9;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
        cfg.enable_kv_cache_events = true;
        return cfg;
    }
};

TEST_F(AdmissionPromptHashSuite, AQueuedAdmissionSeesCachePublishedSinceItsPreviousPlan) {
    Submit({MakeRequestSpec("owner", 3), MakeRequestSpec("queued", 5)});
    const auto first = PlanOnce();
    ASSERT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());
    for (const auto& operation : first.Operations()) {
        if (const auto* batch = std::get_if<ForwardBatch>(&operation)) {
            EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"owner"}));
        }
    }

    SendForwardDone("owner", {42});
    SendFinish("owner");
    const auto stored = scheduler_->DrainKvEvents();
    ASSERT_EQ(stored.size(), 3u);
    for (const auto& event : stored) {
        EXPECT_TRUE(std::holds_alternative<KvBlockStoredEvent>(event));
    }
    const auto retry = PlanOnce();
    bool admitted = false;
    for (const auto& operation : retry.Operations()) {
        if (const auto* batch = std::get_if<ForwardBatch>(&operation)) {
            ASSERT_EQ(batch->request_ids, (std::vector<std::string>{"queued"}));
            EXPECT_EQ(batch->extend_prefix_lens, (std::vector<std::int32_t>{6}));
            admitted = true;
        }
    }
    EXPECT_TRUE(admitted);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());
}

class LoadBackViaCacheTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 7;
        cfg.host_allocator.total_pages = 32;
        cfg.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    // Both seeds finish right after their prefill (a zero decode width plans
    // no decode step): the quiescent finish publishes the prompt pages, whose
    // stream to Host rides the next plan. Of the six usable Device pages,
    // r_fill's five (four pages and the copy-in block of its prompt end at 7)
    // evict one of r_seed's.
    void SetupHostCache() {
        Submit(MakeRequestSpec("r_seed", /*num_pages=*/2, /*start=*/1));
        PlanOnce();
        SendForwardDone("r_seed", {42});
        SendFinish("r_seed");
        const ExecutionPlan seed_stream = PlanOnce();
        ASSERT_FALSE(ExtractCacheOpsOfKind<WriteBackBatch>(seed_stream).empty());
        AckWriteBacks(seed_stream);
        PlanOnce();

        Submit(RequestSpec{.request_id = "r_fill", .tokens = MakeTokens(7, 100)});
        PlanOnce();
        SendForwardDone("r_fill", {200});
        SendFinish("r_fill");
        AckWriteBacks(PlanOnce());
        PlanOnce();
    }
};

TEST_F(LoadBackViaCacheTestSuite, LoadBack_TriggeredAfterPrefetchPopulatesHostCache) {
    SetupHostCache();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2, /*start=*/1));
    auto plan = PlanOnce();
    auto lb = ExtractCacheOpsOfKind<LoadBackBatch>(plan);

    bool r1_in_forward = false;
    for (const auto& op : plan.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            for (const auto& rid : fwd->request_ids) {
                if (rid == "r1") r1_in_forward = true;
            }
        }
    }
    EXPECT_TRUE(r1_in_forward || !lb.empty())
        << "host cache hit should trigger LoadBack inline or r1 should be in forward";
}

TEST_F(SchedulerTestSuite, LoadBack_NotTriggeredWithoutHostCacheHit) {
    Submit(MakeRequestSpec("r1", 4));
    auto plan = PlanOnce();
    auto lb = ExtractCacheOpsOfKind<LoadBackBatch>(plan);
    EXPECT_TRUE(lb.empty());
}

TEST_F(SchedulerTestSuite, NoCacheOps_WhenNoRequests) {
    auto plan = PlanOnce();
    auto cache_ops = ExtractCacheOps(plan);
    EXPECT_TRUE(cache_ops.empty());
}

TEST_F(SchedulerTestSuite, NoCacheOps_PlainRequestNoCacheHit) {
    Submit(MakeRequestSpec("r1", 2));
    auto plan = PlanOnce();
    auto cache_ops = ExtractCacheOps(plan);
    EXPECT_TRUE(cache_ops.empty());
}

// A tail hit's copy source may be evicted by a later admission of the same
// round, once the hitter's table holds the fresh copy and the source is
// cache-only again. The plan then carries the pair in pages_to_copy and the
// source in pages_to_zero: the runtime's copies-before-zeroing order is what
// keeps the copy reading the old bytes.
class CopySourceRegrantSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.disable_l2_cache = true;
        cfg.host_allocator.total_pages = 0;
        // 7 usable: r1 leaves 2 cached blocks; r2 claims one and needs 2
        // (its copy and growth); r3 needs 4 and must evict r2's copy source.
        cfg.device_allocator.total_pages = 8;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(CopySourceRegrantSuite, ACopySourceEvictedInTheSameRoundIsBothCopiedAndZeroed) {
    Submit(RequestSpec{.request_id = "r1", .tokens = {1, 2, 3}});
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();  // prompt end: tail [3] published, r1 moves to a copy
    SendForwardDone("r1", {43});
    SendFinish("r1");  // frontier 4: page [3 42] published, nothing new is a tail
    PlanOnce();
    ASSERT_EQ(scheduler_->TailEntries(), 1u);

    Submit(RequestSpec{.request_id = "r2", .tokens = {1, 2, 3, 7}});
    Submit(RequestSpec{.request_id = "r3", .tokens = {101, 102, 103, 104, 105, 106}});
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, (std::vector<std::string>{"r2", "r3"}));
    EXPECT_EQ(batch->extend_prefix_lens.at(0), 3) << "r2 resumes at the tail";
    EXPECT_EQ(batch->extend_prefix_lens.at(1), 0);
    const auto& copies = plan.pages_to_copy.at("full_attention");
    ASSERT_EQ(copies.size(), 1u);
    const auto [source, destination] = copies.front();
    EXPECT_EQ(batch->BlockTables().at("full_attention").at(0).at(1), destination);
    const auto& zeroed = plan.pages_to_zero.at("full_attention");
    EXPECT_NE(std::ranges::find(zeroed, source), zeroed.end()) << "r3 took the source once the copy left it cache-only";
    EXPECT_EQ(std::ranges::find(zeroed, destination), zeroed.end()) << "the copy fills the destination whole";
    EXPECT_EQ(scheduler_->TailEntries(), 0u) << "the tail left the index with its block";
}

class DisablePrefixCacheTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.disable_prefix_cache = true;
        return cfg;
    }
};

TEST_F(DisablePrefixCacheTestSuite, SamePromptDoesNotReuseDevicePrefix) {
    Submit(MakeRequestSpec("r_seed", 2));
    PlanOnce();
    SendForwardDone("r_seed", {100});
    PlanOnce();
    SendFinish("r_seed");
    PlanOnce();

    Submit(MakeRequestSpec("r1", 2));
    auto plan = PlanOnce();
    const auto& op = plan.Operations()[0];
    auto* fwd = std::get_if<ForwardBatch>(&op);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->request_ids.size(), 1u);
    EXPECT_EQ(fwd->request_ids[0], "r1");
    EXPECT_EQ(fwd->extend_prefix_lens[0], 0);
    EXPECT_EQ(fwd->input_lengths[0], 4);
    EXPECT_TRUE(ExtractCacheOpsOfKind<LoadBackBatch>(plan).empty());
}

class StableCandidateOrderingSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        // Force the candidates loop to break after exactly one push so the
        // tiebreaker decides which request wins.
        cfg.max_batch_size = 1;
        cfg.max_resident_requests = cfg.max_batch_size;
        return cfg;
    }
};

TEST_F(StableCandidateOrderingSuite, ForwardOperationsFollowSubmissionOrder) {
    // TP-determinism + FIFO: requests_ is a vector in submission order, and
    // the mirrored schedulers receive identical submission sequences -- so
    // when the loop budget admits only a subset, every rank picks the same
    // request, and it is the OLDEST one, whatever its id sorts like.
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    auto plan = PlanOnce();
    std::vector<std::string> ids;
    for (const auto& op : plan.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            ids = fwd->request_ids;
        }
    }
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], "r_ccc") << "first submitted, first scheduled";
}

TEST_F(StableCandidateOrderingSuite, ForwardBatchIsReproducibleAcrossMirroredSchedulers) {
    // Two scheduler instances fed the SAME submission sequence must build
    // the same batch -- the mirrored-ranks invariant. (Different submission
    // orders legitimately differ: FIFO means arrival order is meaningful.)
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    auto plan_a = PlanOnce();
    std::vector<std::string> ids_a;
    for (const auto& op : plan_a.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            ids_a = fwd->request_ids;
        }
    }

    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    auto plan_b = PlanOnce();
    std::vector<std::string> ids_b;
    for (const auto& op : plan_b.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            ids_b = fwd->request_ids;
        }
    }

    ASSERT_FALSE(ids_a.empty());
    EXPECT_EQ(ids_a, ids_b);
}

// A request hashes its full pages for KV events once each, every page chained
// on the one before: asking for more pages extends the chain, asking for
// fewer reads a prefix of it.
TEST(RequestKvEventBlockHashes, ChainEachFullPageOnTheOneBefore) {
    Request request(RequestSpec{.request_id = "r", .tokens = {1, 2, 3, 4, 5, 6, 7}}, /*prefix_granularity=*/2,
                    /*awaits_bootstrap=*/false);
    const std::uint64_t first = HashKvBlock(std::vector<std::int32_t>{1, 2}, std::nullopt);
    const std::uint64_t second = HashKvBlock(std::vector<std::int32_t>{3, 4}, first);
    const std::uint64_t third = HashKvBlock(std::vector<std::int32_t>{5, 6}, second);
    const auto hashes = [&request](std::size_t num_pages) {
        const std::span<const std::uint64_t> span = request.KvEventBlockHashes(num_pages);
        return std::vector<std::uint64_t>(span.begin(), span.end());
    };
    EXPECT_EQ(hashes(2), (std::vector<std::uint64_t>{first, second}));
    EXPECT_EQ(hashes(3), (std::vector<std::uint64_t>{first, second, third}));
    EXPECT_EQ(hashes(1), std::vector<std::uint64_t>{first});
    EXPECT_THROW(request.KvEventBlockHashes(4), std::runtime_error) << "token 7 fills no fourth page";
}

class SchedulerKvCacheEventTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 5;  // four usable: two one-page prompts with their decode slots
        cfg.disable_l2_cache = true;
        cfg.enable_kv_cache_events = true;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(SchedulerKvCacheEventTestSuite, PublishesStoredBlockAndDrainsItOnce) {
    const RequestSpec spec = MakeRequestSpec("r1", 1);
    Submit(spec);
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    SendFinish("r1");

    std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[0]));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(events[0]).token_ids, spec.tokens);
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());
}

TEST_F(SchedulerKvCacheEventTestSuite, PartialPrefillRetractionUsesItsActualTailParent) {
    config_.device_allocator.total_pages = 16;
    config_.cache_groups.front().total_pages = 16;
    config_.max_scheduled_tokens = 3;
    config_.debug_force_retraction_interval = -1;
    SetTestSnapshotPool(config_);
    scheduler_ = std::make_unique<Scheduler>(config_);
    Submit(RequestSpec{.request_id = "partial", .tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9}});
    const ExecutionPlan first = PlanOnce();
    ASSERT_NE(FindForwardBatch(first), nullptr);
    EXPECT_EQ(FindForwardBatch(first)->input_lengths, std::vector<std::int32_t>{3});
    SendForwardDone("partial");
    const ExecutionPlan retraction = PlanOnce();
    ASSERT_NE(FindSnapshotStore(retraction), nullptr);
    const std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 2u);
    ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[0]));
    ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[1]));
    const auto& page = std::get<KvBlockStoredEvent>(events[0]);
    const auto& tail = std::get<KvBlockStoredEvent>(events[1]);
    EXPECT_EQ(page.token_ids, (std::vector<std::int32_t>{1, 2}));
    EXPECT_EQ(tail.token_ids, std::vector<std::int32_t>{3});
    EXPECT_EQ(tail.parent_block_hash, page.block_hashes.front());
}

TEST_F(SchedulerKvCacheEventTestSuite, PublishesRemovalWhenAdmissionEvictsBlock) {
    Submit(MakeRequestSpec("seed", 1));
    PlanOnce();
    SendForwardDone("seed", {42});
    SendFinish("seed");  // quiescent: publishes its page and releases its blocks at once

    std::vector<KvCacheEvent> stored = scheduler_->DrainKvEvents();
    ASSERT_EQ(stored.size(), 1u);
    const std::uint64_t stored_hash = std::get<KvBlockStoredEvent>(stored[0]).block_hashes.front();

    // The filler takes two of the three free blocks; the replacement's
    // admission then evicts the seed's page for its second.
    Submit(MakeRequestSpec("filler", 1, 100));
    PlanOnce();
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());
    Submit(MakeRequestSpec("replacement", 1, 200));
    PlanOnce();

    std::vector<KvCacheEvent> removed = scheduler_->DrainKvEvents();
    ASSERT_EQ(removed.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<KvBlockRemovedEvent>(removed[0]));
    EXPECT_EQ(std::get<KvBlockRemovedEvent>(removed[0]).block_hashes, std::vector<std::uint64_t>{stored_hash});
}

// A finish whose frontier is not page-aligned publishes the partial page as a
// tail: one more BlockStored whose token_ids are the rows below T (shorter
// than block_size), chained to the last full page, and removed with it.
TEST_F(SchedulerKvCacheEventTestSuite, PublishesATailDescriptorAtAnUnalignedFinish) {
    config_.device_allocator.total_pages = 5;
    config_.cache_groups.front().total_pages = config_.device_allocator.total_pages;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(config_));
    const RequestSpec spec{.request_id = "r1", .tokens = {1, 2, 3}};
    Submit(spec);
    PlanOnce();
    SendForwardDone("r1", {42});
    SendFinish("r1");  // computed tokens [1 2 3]: page [1 2] and the tail [3]

    std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 2u);
    ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[0]));
    ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[1]));
    const auto& page = std::get<KvBlockStoredEvent>(events[0]);
    const auto& tail = std::get<KvBlockStoredEvent>(events[1]);
    EXPECT_EQ(page.token_ids, (std::vector<std::int32_t>{1, 2}));
    EXPECT_FALSE(page.parent_block_hash.has_value());
    EXPECT_EQ(tail.token_ids, (std::vector<std::int32_t>{3}));
    EXPECT_EQ(tail.parent_block_hash, page.block_hashes.front()) << "the tail chains to the last full page";
    EXPECT_EQ(tail.block_size, PrefixGranularity());
    EXPECT_NE(tail.block_hashes.front(), page.block_hashes.front());
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());

    // Two prompts sharing nothing fill the pool: the first takes the two
    // free blocks, the second's admission evicts both boundaries, the tail
    // (deeper) first.
    Submit(MakeRequestSpec("filler", 1, 100));
    PlanOnce();
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());
    Submit(MakeRequestSpec("replacement", 1, 200));
    PlanOnce();
    std::vector<KvCacheEvent> removed = scheduler_->DrainKvEvents();
    ASSERT_EQ(removed.size(), 2u);
    ASSERT_TRUE(std::holds_alternative<KvBlockRemovedEvent>(removed[0]));
    ASSERT_TRUE(std::holds_alternative<KvBlockRemovedEvent>(removed[1]));
    EXPECT_EQ(std::get<KvBlockRemovedEvent>(removed[0]).block_hashes, tail.block_hashes);
    EXPECT_EQ(std::get<KvBlockRemovedEvent>(removed[1]).block_hashes, page.block_hashes);
}

class HybridKvCacheEventTestSuite : public SchedulerKvCacheEventTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerKvCacheEventTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 16;
        CacheGroupConfig state = cfg.cache_groups.front();
        state.group_id = "linear_attention_0";
        state.attn_kind = AttnKind::kState;
        state.block_granularity = 0;
        cfg.cache_groups.push_back(std::move(state));
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

// A State group has no token span, so a tail boundary's residency must not ask it for a block granularity.
TEST_F(HybridKvCacheEventTestSuite, DrainsATailBoundaryAlongsideAStateGroup) {
    Submit(RequestSpec{.request_id = "r1", .tokens = {1, 2, 3}});
    PlanOnce();
    SendForwardDone("r1", {42});
    SendFinish("r1");

    // Only the prompt end has state, but the router needs the history parent
    // before the tail can refer to it. This does not grant a state-cache hit.
    std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 2u);
    const auto& page = std::get<KvBlockStoredEvent>(events[0]);
    const auto& tail = std::get<KvBlockStoredEvent>(events[1]);
    EXPECT_EQ(page.token_ids, (std::vector<std::int32_t>{1, 2}));
    EXPECT_FALSE(page.parent_block_hash);
    EXPECT_EQ(tail.token_ids, (std::vector<std::int32_t>{3}));
    EXPECT_EQ(tail.parent_block_hash, page.block_hashes.front());
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());

    Submit(RequestSpec{.request_id = "repeat", .tokens = {1, 2, 3}});
    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(first);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->extend_prefix_lens.at(0), 0);
    // The first repeat creates a checkpoint at the history junction. A later
    // repeat may reuse it, whereas the complete prompt-end state cannot supply
    // the logits for the prompt's last token.
    SendForwardDone("repeat", {});
    PlanOnce();
    SendForwardDone("repeat", {43});
    SendFinish("repeat");
    PlanOnce();
    Submit(RequestSpec{.request_id = "hit", .tokens = {1, 2, 3}});
    const ExecutionPlan repeated = PlanOnce();
    batch = FindForwardBatch(repeated);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->extend_prefix_lens.at(0), 2);
}

class UnsupportedKvCacheEventLayoutSuite : public HybridKvCacheEventTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = HybridKvCacheEventTestSuite::MakeConfig();
        cfg.enable_kv_cache_events = false;
        return cfg;
    }
};

TEST_F(UnsupportedKvCacheEventLayoutSuite, RejectsEventsWithoutHistoryButKeepsServingLayouts) {
    for (const bool state : {false, true}) {
        for (const bool window : {false, true}) {
            if (!state && !window) continue;
            for (const bool replayable : {false, true}) {
                if (replayable && !window) continue;
                auto cfg = config_;
                auto swa = cfg.cache_groups.front();
                auto checkpoint = cfg.cache_groups.back();
                cfg.cache_groups.clear();
                if (state) cfg.cache_groups.push_back(checkpoint);
                if (window) {
                    swa.attn_kind = AttnKind::kSlidingWindow;
                    swa.sliding_window_tokens = 2;
                    swa.replayable = replayable;
                    cfg.cache_groups.push_back(swa);
                }
                SCOPED_TRACE(::testing::Message()
                             << "state=" << state << " window=" << window << " replayable=" << replayable);
                SetTestSnapshotPool(cfg);
                cfg.enable_kv_cache_events = true;
                try {
                    Scheduler rejected(WithHostPackingFromDevice(cfg));
                    FAIL() << "unsupported event layout was accepted";
                } catch (const std::invalid_argument& error) {
                    EXPECT_NE(std::string(error.what()).find("KV cache events require"), std::string::npos);
                }
                cfg.enable_kv_cache_events = false;
                scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(cfg));
                Submit(RequestSpec{.request_id = "seed", .tokens = {1, 2, 3, 4, 5, 6}});
                PlanOnce();
                SendForwardDone("seed", {42});
                SendFinish("seed");
                EXPECT_TRUE(scheduler_->DrainKvEvents().empty());
                EXPECT_TRUE(scheduler_->ClearCache());
            }
        }
    }
}

TEST(KvCacheEventStatusTest, HistoryCountsMatchPerKeyResidencyAcrossTiersAndShards) {
    for (const std::int32_t tail_tokens : {0, 1, 2, 3, 4, 5, 6, 7}) {
        SCOPED_TRACE(tail_tokens);
        BlockPool device(12, {4, 2, 1, 1});
        BlockPool host(12, {4, 2, 1, 1});
        const std::array specs{
            CacheGroupSpec{.attn_kind = AttnKind::kFull,
                           .cache_blocks_per_lcm_block = 4,
                           .block_granularity = 2,
                           .shard_count = 2},
            CacheGroupSpec{.attn_kind = AttnKind::kFull,
                           .cache_blocks_per_lcm_block = 2,
                           .block_granularity = 4,
                           .shard_count = 2},
            CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
            CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                           .sliding_window = 2,
                           .cache_blocks_per_lcm_block = 1,
                           .block_granularity = 8}};
        auto coordinator =
            MakeCoordinator(specs, 8, device, false, &host, /*disk_pool=*/nullptr, nullptr, false, false, 0);
        const CacheKey boundary{.content_hash = ContentHash{"prefix"}};
        std::vector<CacheKey> history;
        for (std::uint32_t group = 0; group < 2; ++group) {
            const auto granularity = specs[group].block_granularity;
            const auto tokens = tail_tokens == 0 ? 8 : tail_tokens;
            for (std::int32_t offset = 0; offset < (tokens + granularity - 1) / granularity; ++offset) {
                history.push_back(
                    CacheKey{.group_id = group, .content_hash = boundary.content_hash, .page_offset = offset});
            }
        }
        const auto check = [&]() {
            bool complete = true;
            for (const auto& key : history) {
                complete &= coordinator.GroupPrefixIndex(key.group_id).Contains(device, key);
            }
            EXPECT_EQ(coordinator.GetBoundaryEventStatus(boundary, tail_tokens).device_publishable, complete);
        };
        const auto store = [&](BlockPool& pool, const CacheKey& key) {
            auto block = pool.AcquireBlock(key.group_id);
            coordinator.GroupPrefixIndex(key.group_id)
                .Register(pool, block, key, 1, -1, CacheBoundaryKind::kChunk, nullptr, nullptr);
        };
        const auto remove_from = [&](BlockPool& pool, const CacheKey& key) {
            auto& index = coordinator.GroupPrefixIndex(key.group_id);
            const auto entry = index.Inspect(pool, key);
            ASSERT_TRUE(entry.has_value());
            EXPECT_TRUE(index.Evict(pool, entry->location).has_value());
        };
        const auto remove = [&](const CacheKey& key) { remove_from(device, key); };
        check();
        EXPECT_FALSE(coordinator.GetBoundaryEventStatus(boundary, tail_tokens).host_publishable);
        for (const auto& key : history) store(host, key);
        check();  // Host history must not satisfy Device publication.
        const auto on_host = coordinator.GetBoundaryEventStatus(boundary, tail_tokens);
        EXPECT_TRUE(on_host.host_publishable) << "complete Host history publishes on the Host medium";
        EXPECT_TRUE(on_host.retains_descriptor) << "a Host entry keeps the descriptor";
        const CacheKey state{.group_id = 2, .content_hash = boundary.content_hash};
        const CacheKey swa{.group_id = 3, .content_hash = boundary.content_hash};
        store(device, state);
        store(device, swa);
        check();  // Auxiliary entries cannot fill missing history counts.
        for (const auto& key : history) {
            store(device, key);
            check();
        }
        EXPECT_TRUE(coordinator.GetBoundaryEventStatus(boundary, tail_tokens).device_publishable);
        remove(state);
        remove(swa);
        check();  // Complete history remains publishable without auxiliary state.
        EXPECT_EQ(coordinator.Store().DeviceBoundaryResidency(boundary, tail_tokens),
                  CacheStore::BoundaryResidency::kPartial);
        store(device, state);
        store(device, swa);
        // Removing any sub-block of either sharded Full group withdraws the
        // event; restoring it makes the same boundary publishable again.
        for (const auto& key : history) {
            remove(key);
            check();
            EXPECT_FALSE(coordinator.GetBoundaryEventStatus(boundary, tail_tokens).device_publishable);
            store(device, key);
            check();
        }
        for (const auto& key : history) {
            remove(key);
            check();
        }
        EXPECT_TRUE(coordinator.GetBoundaryEventStatus(boundary, tail_tokens).retains_descriptor);
        remove(state);
        remove(swa);
        check();
        EXPECT_TRUE(coordinator.GetBoundaryEventStatus(boundary, tail_tokens).retains_descriptor)
            << "the Host history still holds the boundary";
        for (const auto& key : history) remove_from(host, key);
        const auto gone = coordinator.GetBoundaryEventStatus(boundary, tail_tokens);
        EXPECT_FALSE(gone.host_publishable);
        EXPECT_FALSE(gone.retains_descriptor);
    }
}

class FullSwaStateKvCacheEventTestSuite : public HybridKvCacheEventTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = HybridKvCacheEventTestSuite::MakeConfig();
        auto swa = cfg.cache_groups.front();
        swa.group_id = "sliding_attention";
        swa.attn_kind = AttnKind::kSlidingWindow;
        swa.sliding_window_tokens = 2;
        swa.replayable = false;
        cfg.cache_groups.push_back(std::move(swa));
        return cfg;
    }
};

TEST_F(FullSwaStateKvCacheEventTestSuite, WindowDoesNotLeaveHolesInHistoryEvents) {
    Submit(RequestSpec{.request_id = "seed", .tokens = {1, 2, 3, 4, 5, 6}});
    PlanOnce();
    SendForwardDone("seed", {42});
    SendFinish("seed");
    const auto events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 3u);
    std::optional<std::uint64_t> parent;
    for (const auto& event : events) {
        const auto& block = std::get<KvBlockStoredEvent>(event);
        EXPECT_EQ(block.parent_block_hash, parent);
        parent = block.block_hashes.front();
    }
    ASSERT_TRUE(scheduler_->ClearCache());
    EXPECT_EQ(scheduler_->DrainKvEvents().size(), 3u);
}

class MultiGroupKvCacheEventTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 9;
        cfg.disable_l2_cache = true;
        cfg.enable_kv_cache_events = true;
        CacheGroupConfig second = cfg.cache_groups.front();
        second.group_id = "full_attention_1";
        cfg.cache_groups.push_back(std::move(second));
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(MultiGroupKvCacheEventTestSuite, PublishesOneEventAfterAllGroupsCacheBoundary) {
    Submit(MakeRequestSpec("r1", 1));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    SendFinish("r1");

    std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[0]));
}

class SubpageKvCacheEventTestSuite : public SchedulerKvCacheEventTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerKvCacheEventTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 5;
        auto& group = cfg.cache_groups.front();
        group.block_granularity = 1;
        group.total_pages = 2 * cfg.device_allocator.total_pages;
        group.cache_blocks_per_lcm_block = 2;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(SubpageKvCacheEventTestSuite, PublishesOneEventAfterAllChildBlocksCacheBoundary) {
    const RequestSpec spec = MakeRequestSpec("r1", 1);
    Submit(spec);
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    SendFinish("r1");

    std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[0]));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(events[0]).token_ids, spec.tokens);
}

TEST_F(SubpageKvCacheEventTestSuite, PublishesRemovalWhenAnyChildBlockIsEvicted) {
    Submit(MakeRequestSpec("seed", 2));
    PlanOnce();
    SendForwardDone("seed", {42});
    SendFinish("seed");  // quiescent: publishes its pages and releases its blocks at once

    std::vector<KvCacheEvent> stored = scheduler_->DrainKvEvents();
    ASSERT_EQ(stored.size(), 2u);
    const std::uint64_t first_stored_hash = std::get<KvBlockStoredEvent>(stored[0]).block_hashes.front();
    const std::uint64_t second_stored_hash = std::get<KvBlockStoredEvent>(stored[1]).block_hashes.front();

    // Five tokens, the decode slot and the copy-in block of its prompt end
    // fill the four parents: both of seed's pages leave.
    Submit(RequestSpec{.request_id = "replacement", .tokens = MakeTokens(5, 100)});
    PlanOnce();

    std::vector<KvCacheEvent> removed = scheduler_->DrainKvEvents();
    ASSERT_EQ(removed.size(), 2u);
    std::unordered_set<std::uint64_t> remaining_hashes{first_stored_hash, second_stored_hash};
    for (const KvCacheEvent& event : removed) {
        ASSERT_TRUE(std::holds_alternative<KvBlockRemovedEvent>(event));
        const auto& removed_hashes = std::get<KvBlockRemovedEvent>(event).block_hashes;
        ASSERT_EQ(removed_hashes.size(), 1u);
        EXPECT_EQ(remaining_hashes.erase(removed_hashes[0]), 1u);
    }
    EXPECT_TRUE(remaining_hashes.empty());
}

class PrefillRoleKvCacheEventTestSuite : public SchedulerKvCacheEventTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerKvCacheEventTestSuite::MakeConfig();
        cfg.role = Role::kP;
        cfg.max_scheduled_tokens = 9;
        cfg.device_allocator.total_pages = 8;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        cfg.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{request_id});
        scheduler_->Advance(std::move(event));
    }

    void SendPdSucceeded(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::SucceededEvent{request_id});
        scheduler_->Advance(std::move(event));
    }
};

// A P-role admission publishes the pages the request completed so far, and the
// same Admit may evict to fit the next chunk. When the victim is another
// request's cached copy of a page this request recomputed, the boundary is
// removed and stored again within one Admit: the net state is unchanged, and
// the boundary's descriptor must survive for its later removal.
TEST_F(PrefillRoleKvCacheEventTestSuite, AdmissionThatEvictsAndRepublishesABoundaryKeepsItPublished) {
    RequestSpec seed = MakeRequestSpec("seed", 3);
    seed.tokens.resize(5);
    // dup shares seed's first two pages, then diverges.
    RequestSpec dup = MakeRequestSpec("dup", 5);
    std::copy_n(seed.tokens.begin(), 4, dup.tokens.begin());
    for (std::size_t i = 4; i < dup.tokens.size(); ++i) {
        dup.tokens[i] = 500 + static_cast<std::int32_t>(i);
    }
    Submit({seed, dup});
    SendBootstrapped("seed");
    SendBootstrapped("dup");

    // Neither prompt is cached yet: both compute the shared pages.
    const ExecutionPlan first_round = PlanOnce();
    const ForwardBatch* first = FindForwardBatch(first_round);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->request_ids, (std::vector<std::string>{"seed", "dup"}));
    EXPECT_EQ(first->input_lengths, (std::vector<std::int32_t>{5, 4}));

    // seed's remote decode publishes the shared pages; dup's next chunk does
    // not fit while seed still holds its pages for the PD transfer.
    SendForwardDone("seed", {42});
    SendForwardDone("dup");  // its first chunk landed: KV only
    const ExecutionPlan second_round = PlanOnce();
    ASSERT_TRUE(second_round.remote_decode.has_value());
    const ForwardBatch* second = FindForwardBatch(second_round);
    ASSERT_NE(second, nullptr);
    EXPECT_TRUE(second->request_ids.empty());
    std::vector<std::uint64_t> shared_hashes;
    for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
        ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(event));
        shared_hashes.push_back(std::get<KvBlockStoredEvent>(event).block_hashes.front());
    }
    ASSERT_EQ(shared_hashes.size(), 2u);

    // The PD ACK leaves seed's copies cached and evictable. dup's last chunk
    // (positions 4..9) plus its one-token decode reserve spans pages 2..5,
    // more than are free, so its admission evicts seed's copy of a shared
    // page, then publishes its own.
    SendPdSucceeded("seed");
    const std::int32_t pages_needed = 4;
    ASSERT_LT(scheduler_->CacheGroupAvailablePages("full_attention"), pages_needed);
    const ExecutionPlan third_round = PlanOnce();
    const ForwardBatch* third = FindForwardBatch(third_round);
    ASSERT_NE(third, nullptr);
    EXPECT_EQ(third->request_ids, std::vector<std::string>{"dup"});
    EXPECT_EQ(third->extend_prefix_lens, std::vector<std::int32_t>{4});
    EXPECT_EQ(third->input_lengths, std::vector<std::int32_t>{6});
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());

    // Both shared pages are still published: flushing the cache removes them.
    SendForwardDone("dup", {42});
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    SendPdSucceeded("dup");
    scheduler_->DrainKvEvents();  // dup's own later pages
    ASSERT_TRUE(scheduler_->ClearL1Cache());
    std::unordered_set<std::uint64_t> removed;
    for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
        ASSERT_TRUE(std::holds_alternative<KvBlockRemovedEvent>(event));
        removed.insert(std::get<KvBlockRemovedEvent>(event).block_hashes.front());
    }
    for (const std::uint64_t hash : shared_hashes) {
        EXPECT_TRUE(removed.contains(hash));
    }
}

// Two cache groups with skewed block granularities, the DeepSeek V4.1 shape:
// comparing position ranks across groups makes eviction strip the fine
// group's copies of a whole boundary range before the coarse group gives up
// a page, leaving those boundaries partially resident.
class SkewedGroupPrefillKvCacheEventTestSuite : public PrefillRoleKvCacheEventTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = PrefillRoleKvCacheEventTestSuite::MakeConfig();
        cfg.prefix_granularity = 4;
        cfg.max_scheduled_tokens = 13;
        cfg.device_allocator.total_pages = 9;
        // The fine group comes first so it wins eviction-rank ties against
        // the coarse group's copy of the same boundary.
        CacheGroupConfig fine = cfg.cache_groups.front();
        fine.group_id = "fine_attention";
        fine.block_granularity = 2;
        fine.cache_blocks_per_lcm_block = 2;
        fine.total_pages = 2 * cfg.device_allocator.total_pages;
        fine.transfer_policy = CacheTransferPolicy::FullSuffix;
        CacheGroupConfig& coarse = cfg.cache_groups.front();
        coarse.block_granularity = cfg.prefix_granularity;
        coarse.total_pages = cfg.device_allocator.total_pages;
        cfg.cache_groups.insert(cfg.cache_groups.begin(), std::move(fine));
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Runs a fresh prompt to its remote decode and optionally releases its
    // PD pin, without draining any KV events.
    void RunPrefill(const RequestSpec& spec, bool ack) {
        Submit(spec);
        SendBootstrapped(spec.request_id);
        const ExecutionPlan compute_round = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(compute_round);
        ASSERT_NE(batch, nullptr);
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{spec.request_id});
        SendForwardDone(spec.request_id, {42});
        ASSERT_TRUE(PlanOnce().remote_decode.has_value());
        if (ack) {
            SendPdSucceeded(spec.request_id);
        }
    }
};

// Evictions mark boundaries suffix-first, so a partially resident run whose
// last copies are evicted and then recomputed within one drain window is
// marked child-first. The drain must still emit the recomputed run's Stored
// events parent-first: consumers resolve each event's parent_block_hash
// against what they have already received, so a child arriving ahead of its
// parent is dropped.
TEST_F(SkewedGroupPrefillKvCacheEventTestSuite, RepublishedRunEmitsStoredParentFirst) {
    // seed publishes a three-boundary chain; the evictions below strip it
    // from its suffix.
    RequestSpec seed = MakeRequestSpec("seed", 3);
    RunPrefill(seed, /*ack=*/true);
    std::vector<std::uint64_t> chain_hashes;
    for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
        ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(event));
        chain_hashes.push_back(std::get<KvBlockStoredEvent>(event).block_hashes.front());
    }
    ASSERT_EQ(chain_hashes.size(), 3u);

    // filler1's admission evicts across the groups' skewed ranks, taking the
    // chain's fine-group pages and some coarse residues from the back. The
    // first two boundaries come out partially resident: reported removed,
    // descriptors kept.
    RunPrefill(MakeRequestSpec("filler1", 2, 100), /*ack=*/true);
    std::unordered_set<std::uint64_t> removed;
    for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
        if (std::holds_alternative<KvBlockRemovedEvent>(event)) {
            removed.insert(std::get<KvBlockRemovedEvent>(event).block_hashes.front());
        }
    }
    EXPECT_TRUE(removed.contains(chain_hashes[0]));
    EXPECT_TRUE(removed.contains(chain_hashes[1]));

    // No drain from here on: the window covers everything below, as one
    // production round covers one admission that evicts and one that
    // publishes. filler2's admission evicts the surviving copies
    // suffix-first, marking the deeper boundary before its parent.
    RunPrefill(MakeRequestSpec("filler2", 2, 200), /*ack=*/true);

    // dup recomputes the chain; its remote decode publishes the new copies.
    RequestSpec dup = MakeRequestSpec("dup", 3);
    dup.tokens = seed.tokens;
    RunPrefill(dup, /*ack=*/false);

    // The drain must order the chain's Stored events parent-first even
    // though the evictions marked it child-first.
    std::vector<std::uint64_t> stored_order;
    for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
        if (!std::holds_alternative<KvBlockStoredEvent>(event)) {
            continue;
        }
        const std::uint64_t hash = std::get<KvBlockStoredEvent>(event).block_hashes.front();
        if (std::find(chain_hashes.begin(), chain_hashes.end(), hash) != chain_hashes.end()) {
            stored_order.push_back(hash);
        }
    }
    EXPECT_EQ(stored_order, chain_hashes);
}

// A full and a sliding group whose window (six tokens over four-token pages)
// reaches one page back behind a page boundary, without a Host pool: a
// retracted request's image rides the snapshot pool whole, and its restore
// copies every slot back as a private copy. The knob retracts at plan 5.
class SlidingWindowRestoreKvCacheEventTestSuite : public SchedulerKvCacheEventTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerKvCacheEventTestSuite::MakeConfig();
        cfg.prefix_granularity = 4;
        cfg.max_scheduled_tokens = 8;
        // Twelve usable: r's six cache-only blocks beside o's six fill them.
        // The startup bound charges r's 14-token limit a tail handover and
        // the copy-in block beside it in each group.
        cfg.device_allocator.total_pages = 13;
        cfg.debug_force_retraction_interval = 5;
        CacheGroupConfig& full = cfg.cache_groups.front();
        full.block_granularity = cfg.prefix_granularity;
        full.total_pages = cfg.device_allocator.total_pages;
        CacheGroupConfig sliding = full;
        sliding.group_id = "sliding";
        sliding.attn_kind = AttnKind::kSlidingWindow;
        sliding.sliding_window_tokens = cfg.prefix_granularity + 2;
        cfg.cache_groups.push_back(std::move(sliding));
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Applies a drain to the boundaries the feed reports stored: a removal
    // names one stored before, and none is stored twice.
    void Drain() {
        for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
            if (const auto* removed = std::get_if<KvBlockRemovedEvent>(&event)) {
                for (const std::uint64_t hash : removed->block_hashes) {
                    EXPECT_EQ(stored_.erase(hash), 1u) << "a removal names a boundary never stored";
                }
                continue;
            }
            const auto& stored = std::get<KvBlockStoredEvent>(event);
            EXPECT_TRUE(stored_.emplace(stored.block_hashes.front(), stored.token_ids).second)
                << "a boundary is stored twice";
        }
    }
    ExecutionPlan PlanAndDrain() {
        ExecutionPlan plan = PlanOnce();
        Drain();
        return plan;
    }
    bool Stored(const std::vector<std::int32_t>& tokens) const {
        return std::ranges::any_of(stored_, [&tokens](const auto& entry) { return entry.second == tokens; });
    }

    std::map<std::uint64_t, std::vector<std::int32_t>> stored_;
};

// r = [1..8] is retracted at frontier 11 (plan 5), leaving its published
// blocks cached: both prompt pages and the tail [101 102 103], in both
// groups. o = [200..207], admitted while r's image is in flight, takes the
// six free blocks, so the restore (plan 7) funds its private copies by
// evicting every one of those entries, and the drain drops their descriptors.
// r's page completion at 12 (plan 9) publishes the window behind it, rows 7 to
// 11: its private copy of page 1 goes into the index under page 1's hash,
// whose descriptor that publication registers again.
TEST_F(SlidingWindowRestoreKvCacheEventTestSuite,
       APageCompletionAfterAPoolRestoreRegistersThePageItsWindowReachesBackInto) {
    Submit(RequestSpec{.request_id = "r", .tokens = MakeTokens(8), .max_new_tokens = 6});
    PlanAndDrain();  // plan 1: r's prefill
    SendForwardDone("r", {101});
    PlanAndDrain();  // plan 2: the first decode publishes both prompt pages
    EXPECT_TRUE(Stored({1, 2, 3, 4}));
    EXPECT_TRUE(Stored({5, 6, 7, 8}));
    SendForwardDone("r", {102});
    PlanAndDrain();
    SendForwardDone("r", {103});
    PlanAndDrain();
    SendForwardDone("r", {104});
    const ExecutionPlan retraction = PlanAndDrain();  // plan 5
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    EXPECT_TRUE(Stored({101, 102, 103}));

    Submit(RequestSpec{.request_id = "o", .tokens = MakeTokens(8, 200), .max_new_tokens = 2});
    ASSERT_EQ(FindForwardBatch(PlanAndDrain())->request_ids, std::vector<std::string>{"o"});  // plan 6
    ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 0);
    ASSERT_TRUE(Stored({5, 6, 7, 8})) << "o's admission evicted nothing";
    AckImageStores(retraction);
    SendForwardDone("o", {300});
    const ExecutionPlan readmit = PlanAndDrain();  // plan 7: the restore, beside o's first decode
    const SnapshotRestoreBatch* restore = FindRestore(readmit);
    ASSERT_NE(restore, nullptr);
    ASSERT_EQ(restore->request_ids, std::vector<std::string>{"r"});
    for (const std::string& hash : restore->content_hashes.at(0)) {
        EXPECT_TRUE(hash.empty()) << "every row is a private copy";
    }
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0) << "no cache-only entry of r's is left";
    EXPECT_FALSE(Stored({1, 2, 3, 4}));
    EXPECT_FALSE(Stored({5, 6, 7, 8}));
    EXPECT_FALSE(Stored({101, 102, 103}));
    AckRestores(readmit);
    Drain();
    SendForwardDone("o", {301});
    SendFinish("o");

    ASSERT_EQ(FindForwardBatch(PlanAndDrain())->request_ids, std::vector<std::string>{"r"});  // plan 8
    SendForwardDone("r", {105});
    PlanAndDrain();  // plan 9
    // Registration still succeeds after the private pool restore, but the
    // earlier history pages are not published. An orphan cannot be routed.
    EXPECT_FALSE(Stored({101, 102, 103, 104}));
    SendForwardDone("r", {106});
    SendFinish("r");
    PlanAndDrain();
    EXPECT_EQ(scheduler_->RequestTokenSize("r"), -1);
}

// One prefill row of a plan: the request, the prompt positions it computes
// and its prompt's length.
struct PrefillRow {
    std::string id;
    std::int32_t start{0};
    std::int32_t end{0};
    std::int32_t prompt{0};
    bool operator==(const PrefillRow&) const = default;
};
using PrefillRows = std::vector<PrefillRow>;

inline std::ostream& operator<<(std::ostream& out, const PrefillRow& row) {
    return out << row.id << "[" << row.start << ", " << row.end << ")/" << row.prompt;
}

// SchedulerConfig::prefill_chunk_yield_tokens on the P role: a chunk that
// cannot finish its prompt this round -- a resident's, or a new prompt's
// first -- ends on the last page at least the tokens of the bootstrapped
// prompts behind it that fit whole before its budget's end, and those prompts
// complete in its forward, their rows ahead of its own: in every round the
// completing prefills form a prefix of the batch. A chunk that keeps the head
// of line for any reason fills the budget it still leaves with whole prompts
// the same way, with the knob at 0 too. 4-token pages, a 32-token budget, 16
// tokens yielded at most, one full-attention group; with `hybrid_` three
// State groups beside it.
class PrefillChunkYieldSuite : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kPage = 4;
    static constexpr std::int32_t kBudget = 32;

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.role = Role::kP;
        cfg.prefix_granularity = kPage;
        cfg.device_allocator.total_pages = pages_;
        cfg.disable_l2_cache = true;
        cfg.max_scheduled_tokens = kBudget;
        cfg.max_batch_size = batch_;
        cfg.max_resident_requests = cfg.max_batch_size;
        cfg.max_kv_page_num = kv_pages_;
        cfg.prefill_chunk_yield_tokens = yield_;
        cfg.prefill_chunk_yield_max_rounds = max_rounds_;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .block_granularity = kPage,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kFull,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        for (std::int32_t i = 0; hybrid_ && i < 3; ++i) {
            cfg.cache_groups.push_back(CacheGroupConfig{
                .group_id = "linear_attention_" + std::to_string(i),
                .block_granularity = 0,
                .total_pages = cfg.device_allocator.total_pages,
                .attn_kind = AttnKind::kState,
                .transfer_policy = CacheTransferPolicy::LatestSnapshot,
            });
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Rebuilds the scheduler with the knob at `yield` tokens.
    void Restart(std::int32_t yield) {
        yield_ = yield;
        SetUp();
    }

    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{request_id});
        scheduler_->Advance(std::move(event));
    }

    void SendPdSucceeded(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::SucceededEvent{request_id});
        scheduler_->Advance(std::move(event));
    }

    void SubmitPrompt(const std::string& id, std::vector<std::int32_t> tokens, bool bootstrap = true) {
        Submit(RequestSpec{.request_id = id, .tokens = std::move(tokens)});
        if (bootstrap) {
            SendBootstrapped(id);
        }
    }

    // Plans one round and returns its prefill rows in batch order, the
    // completing ones first. Every prompt the plan hands off is then
    // acknowledged (unless `ack` is false) and every row lands, a completed
    // prompt with a sampled token.
    PrefillRows Round(bool ack = true, ExecutionPlan* plan_out = nullptr) {
        ExecutionPlan plan = PlanOnce();
        PrefillRows rows;
        if (const ForwardBatch* batch = FindForwardBatch(plan)) {
            for (std::size_t i = 0; i < batch->NumExtends(); ++i) {
                rows.push_back(PrefillRow{
                    .id = batch->request_ids[i],
                    .start = batch->extend_prefix_lens[i] + batch->extend_replay_lens[i],
                    .end = batch->extend_prefix_lens[i] + batch->input_lengths[i],
                    .prompt = batch->prefill_lengths[i],
                });
            }
            EXPECT_TRUE(CompletingPrefillsFormAPrefix(*batch)) << ::testing::PrintToString(rows);
            // Each row's block table is its own request's -- the row of the
            // chunk that gave way too, moved behind the prompts it carries: a
            // page for every position up to the chunk's end and, when the
            // chunk completes its prompt, for its decode slot.
            const auto tables = batch->BlockTables();
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const std::vector<std::int32_t>& pages = tables.at("full_attention")[i];
                const std::int32_t extent =
                    rows[i].end + (rows[i].end == rows[i].prompt ? config_.decode_input_tokens : 0);
                EXPECT_EQ(std::ranges::count_if(pages, [](std::int32_t page) { return page > 0; }),
                          (extent + kPage - 1) / kPage)
                    << rows[i];
            }
        }
        handed_off_ = plan.remote_decode ? plan.remote_decode->request_ids : std::vector<std::string>{};
        for (const std::string& id : handed_off_) {
            if (ack) {
                SendPdSucceeded(id);
            }
        }
        for (const PrefillRow& row : rows) {
            SendForwardDone(row.id,
                            row.end == row.prompt ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
        }
        if (plan_out != nullptr) {
            *plan_out = std::move(plan);
        }
        return rows;
    }

    // Runs `id` to its hand-off and acknowledges it, publishing its prompt.
    void Publish(const std::string& id, std::vector<std::int32_t> tokens) {
        SubmitPrompt(id, std::move(tokens));
        for (int round = 0; round < 8; ++round) {
            Round();
            if (std::ranges::find(handed_off_, id) != handed_off_.end()) {
                return;
            }
        }
        ADD_FAILURE() << id << " was not handed off";
    }

    // A prompt sharing `shared`'s tokens, then `tail` of its own.
    static std::vector<std::int32_t> Extending(std::vector<std::int32_t> shared,
                                               const std::vector<std::int32_t>& tail) {
        shared.insert(shared.end(), tail.begin(), tail.end());
        return shared;
    }

    std::vector<std::string> handed_off_;
    std::int32_t yield_{16};
    std::int32_t max_rounds_{std::numeric_limits<std::int32_t>::max()};
    std::int32_t batch_{8};
    std::int32_t pages_{257};
    std::int32_t kv_pages_{0};
    bool hybrid_{false};
};

// Off, a long prompt's chunk takes the whole budget and holds the head of
// line: the short prompts behind it wait.
TEST_F(PrefillChunkYieldSuite, OffTheLongChunkRunsAlone) {
    Restart(0);
    SubmitPrompt("long", MakeTokens(100, 1000));
    SubmitPrompt("s1", MakeTokens(6, 2000));
    SubmitPrompt("s2", MakeTokens(6, 3000));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 100}}));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 32, 64, 100}}));
}

// A resident chunk that cannot finish ends on the page at 52 (32 + 32 - 11
// rounded down), leaving 12 tokens, enough for both waiting prompts' 11: they
// complete in its forward, ahead of it, wait for their results and leave on
// the next plan's remote decode, while the long prompt keeps the head of line.
TEST_F(PrefillChunkYieldSuite, AResidentChunkGivesWayToWholePromptsBehindIt) {
    SubmitPrompt("long", MakeTokens(100, 1000));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 100}}));
    SubmitPrompt("s1", MakeTokens(5, 2000));
    SubmitPrompt("s2", MakeTokens(6, 3000));
    EXPECT_EQ(Round(), (PrefillRows{{"s1", 0, 5, 5}, {"s2", 0, 6, 6}, {"long", 32, 52, 100}}));
    EXPECT_TRUE(handed_off_.empty());
    EXPECT_EQ(Round(), (PrefillRows{{"long", 52, 84, 100}}));
    EXPECT_EQ(handed_off_, (std::vector<std::string>{"s1", "s2"}));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 84, 100, 100}}));
}

// A new prompt's first chunk gives way to the prompts behind it the same way:
// 32 - 12 = 20, a page boundary.
TEST_F(PrefillChunkYieldSuite, ANewPromptsFirstChunkGivesWayToWholePromptsBehindIt) {
    SubmitPrompt("long", MakeTokens(100, 1000));
    SubmitPrompt("s1", MakeTokens(6, 2000));
    SubmitPrompt("s2", MakeTokens(6, 3000));
    EXPECT_EQ(Round(), (PrefillRows{{"s1", 0, 6, 6}, {"s2", 0, 6, 6}, {"long", 0, 20, 100}}));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 20, 52, 100}}));
}

// The chunk that gives way is admitted before the prompts it carries but runs
// after them: a prompt admitted ahead of a new prompt's first chunk keeps its
// row, the prompt carried behind that chunk takes the next one, and the chunk
// -- 28 - 4 = 24 tokens, a page boundary -- runs last.
TEST_F(PrefillChunkYieldSuite, TheChunkThatGaveWayRunsAfterThePromptsItCarries) {
    SubmitPrompt("p0", MakeTokens(4, 500));
    SubmitPrompt("long", MakeTokens(100, 1000));
    SubmitPrompt("s0", MakeTokens(4, 2000));
    EXPECT_EQ(Round(), (PrefillRows{{"p0", 0, 4, 4}, {"s0", 0, 4, 4}, {"long", 0, 24, 100}}));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 24, 56, 100}}));
}

// What a prompt takes is counted past its hit: 46 tokens, 40 of them cached
// by an earlier prompt, take 6 (32 + 32 - 6 = 58, down to 56). With State
// groups the hit resumes at the State checkpoint the earlier prompt's finish
// published at its end.
TEST_F(PrefillChunkYieldSuite, APromptWhoseUncachedRestFitsIsCarried) {
    for (const bool hybrid : {false, true}) {
        SCOPED_TRACE(hybrid ? "State groups" : "full attention");
        hybrid_ = hybrid;
        Restart(16);
        const std::vector<std::int32_t> earlier = MakeTokens(40, 5000);
        Publish("earlier", earlier);
        SubmitPrompt("long", MakeTokens(100, 1000));
        EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 100}}));
        SubmitPrompt("follow_up", Extending(earlier, MakeTokens(6, 7000)));
        EXPECT_EQ(Round(), (PrefillRows{{"follow_up", 40, 46, 46}, {"long", 32, 56, 100}}));
    }
}

// Without a State group, the long chunk's admission publishing pages a carried
// prompt shares only lengthens its hit: probed at 32 cached tokens, the
// prompt takes 10 and the chunk gives up as much (64 + 32 - 10 = 86, down to
// 84), but it resumes at 40, all its pages but its last.
TEST_F(PrefillChunkYieldSuite, APromptSharingTheChunksUnpublishedPagesResumesFurther) {
    const std::vector<std::int32_t> long_tokens = MakeTokens(100, 1000);
    SubmitPrompt("long", long_tokens);
    EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 100}}));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 32, 64, 100}}));
    SubmitPrompt("sharing", Extending({long_tokens.begin(), long_tokens.begin() + 40}, {7001, 7002}));
    EXPECT_EQ(Round(), (PrefillRows{{"sharing", 40, 42, 42}, {"long", 64, 84, 100}}));
}

// With State groups, the same pages would be cached past any State
// checkpoint once that admission publishes them -- a junction its first chunk
// ends at -- so a prompt sharing them is not carried on its stale probe: the
// long chunk keeps the whole budget. Once they are published, the junction
// keeps it out until the long prompt's final chunk.
TEST_F(PrefillChunkYieldSuite, APromptSharingTheChunksUnpublishedPagesIsNotCarriedBesideStateGroups) {
    hybrid_ = true;
    Restart(16);
    const std::vector<std::int32_t> long_tokens = MakeTokens(100, 1000);
    SubmitPrompt("long", long_tokens);
    EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 100}}));
    SubmitPrompt("sharing", Extending({long_tokens.begin(), long_tokens.begin() + 8}, {7001, 7002}));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 32, 64, 100}})) << "nothing of the long prompt was published yet";
    EXPECT_EQ(Round(), (PrefillRows{{"long", 64, 96, 100}}));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 96, 100, 100}, {"sharing", 0, 8, 10}}))
        << "its first chunk ends at the junction";
}

// Without prefix caching no probe can hit what the long chunk's admission
// publishes, so the same prompt is carried at once, its whole prompt computed.
TEST_F(PrefillChunkYieldSuite, APromptSharingTheChunksPagesIsCarriedBesideStateGroupsWithoutPrefixCaching) {
    hybrid_ = true;
    Restart(16);
    config_.disable_prefix_cache = true;
    scheduler_ = std::make_unique<Scheduler>(config_);
    const std::vector<std::int32_t> long_tokens = MakeTokens(100, 1000);
    SubmitPrompt("long", long_tokens);
    EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 100}}));
    SubmitPrompt("sharing", Extending({long_tokens.begin(), long_tokens.begin() + 8}, {7001, 7002}));
    EXPECT_EQ(Round(), (PrefillRows{{"sharing", 0, 10, 10}, {"long", 32, 52, 100}}));
}

// With only prompts longer than the budget waiting, the plans are those
// without the knob.
TEST_F(PrefillChunkYieldSuite, WithOnlyLongPromptsWaitingThePlansAreThoseWithoutTheKnob) {
    const auto run = [this](std::int32_t yield, std::vector<PrefillRows>& rows) {
        Restart(yield);
        SubmitPrompt("long", MakeTokens(100, 1000));
        SubmitPrompt("other", MakeTokens(60, 2000));
        std::vector<ExecutionPlan> plans;
        for (int round = 0; round < 7; ++round) {
            ExecutionPlan plan;
            rows.push_back(Round(/*ack=*/true, &plan));
            plans.push_back(std::move(plan));
        }
        return plans;
    };
    std::vector<PrefillRows> on;
    std::vector<PrefillRows> off;
    const std::vector<ExecutionPlan> plans_on = run(16, on);
    const std::vector<ExecutionPlan> plans_off = run(0, off);
    EXPECT_EQ(on, off);
    EXPECT_TRUE(plans_on == plans_off);
    EXPECT_EQ(on[0], (PrefillRows{{"long", 0, 32, 100}}));
    EXPECT_EQ(on[3], (PrefillRows{{"long", 96, 100, 100}, {"other", 0, 28, 60}}));
}

// A final chunk never gives way: it completes, and the prompts behind it are
// admitted as without the knob, a long one's first chunk taking what is left.
TEST_F(PrefillChunkYieldSuite, AFinalChunkDoesNotGiveWay) {
    const auto run = [this](std::int32_t yield) {
        Restart(yield);
        SubmitPrompt("long", MakeTokens(40, 1000));
        std::vector<PrefillRows> rows{Round()};
        SubmitPrompt("s1", MakeTokens(6, 2000));
        SubmitPrompt("s2", MakeTokens(6, 3000));
        SubmitPrompt("other", MakeTokens(60, 4000));
        for (int round = 0; round < 3; ++round) {
            rows.push_back(Round());
        }
        return rows;
    };
    const std::vector<PrefillRows> on = run(16);
    EXPECT_EQ(on[1], (PrefillRows{{"long", 32, 40, 40}, {"s1", 0, 6, 6}, {"s2", 0, 6, 6}, {"other", 0, 12, 60}}));
    EXPECT_EQ(on, run(0));
}

// A prompt still waiting for its bootstrap is not carried; once bootstrapped
// it is (64 + 32 - 6 = 90, down to 88).
TEST_F(PrefillChunkYieldSuite, APromptWaitingForItsBootstrapIsNotCarried) {
    SubmitPrompt("long", MakeTokens(100, 1000));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 100}}));
    SubmitPrompt("s1", MakeTokens(6, 2000), /*bootstrap=*/false);
    EXPECT_EQ(Round(), (PrefillRows{{"long", 32, 64, 100}}));
    SendBootstrapped("s1");
    EXPECT_EQ(Round(), (PrefillRows{{"s1", 0, 6, 6}, {"long", 64, 88, 100}}));
}

// The carried prompts take at most the knob's 16 tokens, in FIFO order: a
// third 6-token prompt waits for the next round. The long prompt keeps at
// least 32 - 16 - (4 - 1) = 13 tokens of every round: 20, 20 and 16 here.
TEST_F(PrefillChunkYieldSuite, TheCarriedPromptsTakeAtMostTheKnob) {
    SubmitPrompt("long", MakeTokens(120, 1000));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 120}}));
    for (int i = 0; i < 4; ++i) {
        SubmitPrompt("s" + std::to_string(i), MakeTokens(6, 2000 + 100 * i));
    }
    EXPECT_EQ(Round(), (PrefillRows{{"s0", 0, 6, 6}, {"s1", 0, 6, 6}, {"long", 32, 52, 120}}));
    EXPECT_EQ(Round(), (PrefillRows{{"s2", 0, 6, 6}, {"s3", 0, 6, 6}, {"long", 52, 72, 120}}));
    SubmitPrompt("s4", MakeTokens(15, 3000));
    EXPECT_EQ(Round(), (PrefillRows{{"s4", 0, 15, 15}, {"long", 72, 88, 120}})) << "72 + 32 - 15 = 89, down to 88";
    EXPECT_EQ(Round(), (PrefillRows{{"long", 88, 120, 120}}));
}

// A prompt larger than the knob never changes what a chunk gives way to: a
// 17-token one is passed over, not a stop, so the chunk still gives way to
// the two 8-token prompts behind it. And with 12 tokens left after p0, the
// 100-token prompt's first chunk still gives way to the 16-token prompt, not
// to the 4-token one behind it: giving up 16 would leave the chunk less than
// a page, so it takes all 12 and neither prompt runs.
TEST_F(PrefillChunkYieldSuite, APromptLargerThanTheKnobLeavesWhatAChunkGivesWayTo) {
    SubmitPrompt("long", MakeTokens(120, 1000));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 120}}));
    SubmitPrompt("s17", MakeTokens(17, 2000));
    SubmitPrompt("s8a", MakeTokens(8, 3000));
    SubmitPrompt("s8b", MakeTokens(8, 4000));
    EXPECT_EQ(Round(), (PrefillRows{{"s8a", 0, 8, 8}, {"s8b", 0, 8, 8}, {"long", 32, 48, 120}}));

    Restart(16);
    SubmitPrompt("p0", MakeTokens(20, 500));
    SubmitPrompt("long", MakeTokens(100, 1000));
    SubmitPrompt("s16", MakeTokens(16, 2000));
    SubmitPrompt("s4", MakeTokens(4, 3000));
    EXPECT_EQ(Round(), (PrefillRows{{"p0", 0, 20, 20}, {"long", 0, 12, 100}}));
}

// The chunk that gave way ends on a page, which may leave a few tokens past
// what it gave way for: after p0's 5, the 100-token prompt gives way to s1
// and s2's 16 tokens and ends at 8 (27 - 16 = 11, down to 8), and s3, which
// would have passed the knob, fills the 3 tokens left.
TEST_F(PrefillChunkYieldSuite, AChunkThatGaveWayFillsWhatItsPageEndLeaves) {
    SubmitPrompt("p0", MakeTokens(5, 500));
    SubmitPrompt("long", MakeTokens(100, 1000));
    SubmitPrompt("s1", MakeTokens(8, 2000));
    SubmitPrompt("s2", MakeTokens(8, 3000));
    SubmitPrompt("s3", MakeTokens(3, 4000));
    EXPECT_EQ(Round(),
              (PrefillRows{{"p0", 0, 5, 5}, {"s1", 0, 8, 8}, {"s2", 0, 8, 8}, {"s3", 0, 3, 3}, {"long", 0, 8, 100}}));
}

// A prompt whose first chunk would end at a junction -- its KV is cached
// past the last State checkpoint it can resume at -- cannot run whole, so it
// is not carried, and alone it makes nothing give way.
TEST_F(PrefillChunkYieldSuite, APromptCutAtAJunctionIsNotCarried) {
    hybrid_ = true;
    Restart(16);
    const std::vector<std::int32_t> shared = MakeTokens(8, 5000);
    Publish("earlier", Extending(shared, {5100, 5101, 5102}));
    SubmitPrompt("long", MakeTokens(100, 1000));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 100}}));
    SubmitPrompt("junction", Extending(shared, {6100, 6101}));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 32, 64, 100}}));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 64, 96, 100}}));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 96, 100, 100}, {"junction", 0, 8, 10}}))
        << "its first chunk ends at the junction";
}

// A new prompt's first chunk cut at its junction leaves even more of the
// round to the prompts it carries.
TEST_F(PrefillChunkYieldSuite, AFirstChunkCutAtItsJunctionStillCarriesWholePrompts) {
    hybrid_ = true;
    Restart(16);
    const std::vector<std::int32_t> shared = MakeTokens(8, 5000);
    Publish("earlier", Extending(shared, {5100, 5101, 5102}));
    SubmitPrompt("holder", Extending(shared, MakeTokens(92, 8000)));
    SubmitPrompt("s1", MakeTokens(6, 2000));
    SubmitPrompt("s2", MakeTokens(6, 3000));
    EXPECT_EQ(Round(), (PrefillRows{{"s1", 0, 6, 6}, {"s2", 0, 6, 6}, {"holder", 0, 8, 100}}));
    EXPECT_EQ(Round(), (PrefillRows{{"holder", 8, 40, 100}}));
}

// A new prompt that fits the budget whole gives way to nothing, but cut at
// its junction it keeps the head of line with 24 tokens left: the whole
// prompts behind it fill them, their rows ahead of its own, and it finishes
// next round. The fill does not depend on the knob: at 0 the plans are the
// same.
TEST_F(PrefillChunkYieldSuite, AFirstChunkCutAtItsJunctionFillsTheBudgetItLeaves) {
    const auto run = [this](std::int32_t yield) {
        hybrid_ = true;
        Restart(yield);
        const std::vector<std::int32_t> shared = MakeTokens(8, 5000);
        Publish("earlier", Extending(shared, {5100, 5101, 5102}));
        SubmitPrompt("holder", Extending(shared, MakeTokens(12, 8000)));
        SubmitPrompt("s1", MakeTokens(6, 2000));
        SubmitPrompt("s2", MakeTokens(6, 3000));
        return std::vector<PrefillRows>{Round(), Round()};
    };
    const std::vector<PrefillRows> filled{{{"s1", 0, 6, 6}, {"s2", 0, 6, 6}, {"holder", 0, 8, 20}},
                                          {{"holder", 8, 20, 20}}};
    EXPECT_EQ(run(16), filled);
    EXPECT_EQ(run(0), filled);
}

// The knob bounds what a chunk gives way to, not what fills the budget it
// leaves: the 100-token prompt gives way to s1's 6 tokens, passing over s2's
// 18, ends at its junction at 8, and s2 then fills the 18 tokens left.
TEST_F(PrefillChunkYieldSuite, TheFillTakesMoreThanTheKnob) {
    hybrid_ = true;
    Restart(16);
    const std::vector<std::int32_t> shared = MakeTokens(8, 5000);
    Publish("earlier", Extending(shared, {5100, 5101, 5102}));
    SubmitPrompt("holder", Extending(shared, MakeTokens(92, 8000)));
    SubmitPrompt("s1", MakeTokens(6, 2000));
    SubmitPrompt("s2", MakeTokens(18, 3000));
    EXPECT_EQ(Round(), (PrefillRows{{"s1", 0, 6, 6}, {"s2", 0, 18, 18}, {"holder", 0, 8, 100}}));
}

// With enable_short_prefill_first the new prompts are ranked by the work left
// past their hit, and the fill follows the ranking: the 16-token holder
// resumes at the State one earlier prompt left at 8 and is cut at 12, where
// another's KV ends, so it ranks first with 8 tokens left and keeps the head
// of line with 28 of the budget's 32 left. The other prompts fill them least
// work first, f2's 9 and then "ahead"'s 10, though ahead was submitted before
// the holder, and complete ahead of its row; f1's 12 no longer fit the 9 left
// and wait for the next round, with the knob at 0 too. A holder that gives
// way fills from what the ranking still holds once those it carried are out:
// resuming at 24 with 12 tokens left, the 36-token holder gives way to g1's
// 13 behind it and is cut at its junction at 28, and the 15 tokens left go to
// ahead's 14, not to g2's 15 behind it.
TEST_F(PrefillChunkYieldSuite, TheFillTakesTheLeastWorkFirstWhenShortPrefillsGoFirst) {
    hybrid_ = true;
    for (const std::int32_t yield : {16, 0}) {
        SCOPED_TRACE(::testing::Message() << "knob " << yield);
        Restart(yield);
        config_.enable_short_prefill_first = true;
        scheduler_ = std::make_unique<Scheduler>(config_);
        const std::vector<std::int32_t> shared = Extending(MakeTokens(8, 5000), {5100, 5101, 5102, 5103});
        Publish("checkpoint", MakeTokens(8, 5000));
        Publish("earlier", Extending(shared, {5200, 5201, 5202}));
        SubmitPrompt("ahead", MakeTokens(10, 4000));
        SubmitPrompt("holder", Extending(shared, MakeTokens(4, 8000)));
        SubmitPrompt("f1", MakeTokens(12, 2000));
        SubmitPrompt("f2", MakeTokens(9, 3000));
        EXPECT_EQ(Round(), (PrefillRows{{"f2", 0, 9, 9}, {"ahead", 0, 10, 10}, {"holder", 8, 12, 16}}));
        EXPECT_EQ(Round(), (PrefillRows{{"holder", 12, 16, 16}, {"f1", 0, 12, 12}}));
        EXPECT_EQ(handed_off_, (std::vector<std::string>{"ahead", "f2"}));
    }

    Restart(16);
    config_.enable_short_prefill_first = true;
    scheduler_ = std::make_unique<Scheduler>(config_);
    const std::vector<std::int32_t> longer = Extending(MakeTokens(24, 5000), {5100, 5101, 5102, 5103});
    Publish("checkpoint", MakeTokens(24, 5000));
    Publish("earlier", Extending(longer, {5200, 5201, 5202}));
    SubmitPrompt("ahead", MakeTokens(14, 4000));
    SubmitPrompt("holder", Extending(longer, MakeTokens(8, 8000)));
    SubmitPrompt("g1", MakeTokens(13, 2000));
    SubmitPrompt("g2", MakeTokens(15, 3000));
    EXPECT_EQ(Round(), (PrefillRows{{"g1", 0, 13, 13}, {"ahead", 0, 14, 14}, {"holder", 24, 28, 36}}));
}

// The fill takes a protected prompt first, protection reevaluated after each
// admission: seven 4-token prompts filling a 7-row batch pass "ahead" over
// seven times, and their transfers then hold the request slots for a round.
// The holder ranks first as above, and its own admission is the eighth
// bypass: protected, ahead fills 10 of the 28 tokens left before the three
// 9-token prompts that rank ahead of it by work, two of which fill the rest.
// After six passes in a 6-row batch, f1's admission is the eighth: ahead
// follows it, then f2. A protected prompt the budget left cannot take is
// passed over, not a stop: at 30 tokens, ahead leaves the 28 to the three
// 9-token prompts and starts next round.
TEST_F(PrefillChunkYieldSuite, TheFillTakesAProtectedPromptFirst) {
    const auto run = [this](int passes, std::int32_t ahead_tokens) {
        hybrid_ = true;
        batch_ = passes;
        Restart(16);
        config_.enable_short_prefill_first = true;
        scheduler_ = std::make_unique<Scheduler>(config_);
        const std::vector<std::int32_t> shared = Extending(MakeTokens(8, 5000), {5100, 5101, 5102, 5103});
        Publish("checkpoint", MakeTokens(8, 5000));
        Publish("earlier", Extending(shared, {5200, 5201, 5202}));
        SubmitPrompt("ahead", MakeTokens(ahead_tokens, 4000));
        for (int i = 0; i < passes; ++i) {
            SubmitPrompt("b" + std::to_string(i), MakeTokens(4, 6000 + 100 * i));
        }
        EXPECT_EQ(Round().size(), static_cast<std::size_t>(passes));
        EXPECT_EQ(Round(), PrefillRows{});
        SubmitPrompt("holder", Extending(shared, MakeTokens(4, 8000)));
        for (int i = 1; i <= 3; ++i) {
            SubmitPrompt("f" + std::to_string(i), MakeTokens(9, 2000 + 100 * i));
        }
        return std::vector<PrefillRows>{Round(), Round()};
    };
    EXPECT_EQ(run(7, 10),
              (std::vector<PrefillRows>{{{"ahead", 0, 10, 10}, {"f1", 0, 9, 9}, {"f2", 0, 9, 9}, {"holder", 8, 12, 16}},
                                        {{"holder", 12, 16, 16}, {"f3", 0, 9, 9}}}));
    EXPECT_EQ(run(6, 10),
              (std::vector<PrefillRows>{{{"f1", 0, 9, 9}, {"ahead", 0, 10, 10}, {"f2", 0, 9, 9}, {"holder", 8, 12, 16}},
                                        {{"holder", 12, 16, 16}, {"f3", 0, 9, 9}}}));
    EXPECT_EQ(run(7, 30),
              (std::vector<PrefillRows>{{{"f1", 0, 9, 9}, {"f2", 0, 9, 9}, {"f3", 0, 9, 9}, {"holder", 8, 12, 16}},
                                        {{"holder", 12, 16, 16}, {"ahead", 0, 28, 30}}}));
}

// A resident chunk's fill follows the ranking too, ranked after the chunk's
// admission. Cut at its junction at 40, the 100-token prompt's second chunk
// leaves 24 tokens and, with a 4-token knob or none, gives way to none of the
// prompts waiting: d, c and b fill 19 of them, least work first, and a's 12
// wait. And a chunk's admission publishes the State its last chunk left at
// its junction: ranked after it, a sibling sharing that State takes 2 tokens,
// which fill the 2 a chunk at 8 leaves after giving way to f's 14.
TEST_F(PrefillChunkYieldSuite, AResidentChunkFillsInRankOrderAfterItsAdmission) {
    hybrid_ = true;
    for (const std::int32_t yield : {4, 0}) {
        SCOPED_TRACE(::testing::Message() << "knob " << yield);
        Restart(yield);
        config_.enable_short_prefill_first = true;
        scheduler_ = std::make_unique<Scheduler>(config_);
        const std::vector<std::int32_t> shared = MakeTokens(40, 5000);
        Publish("earlier", Extending(shared, {5100, 5101, 5102}));
        SubmitPrompt("holder", Extending(shared, MakeTokens(60, 8000)));
        EXPECT_EQ(Round(), (PrefillRows{{"holder", 0, 32, 100}}));
        SubmitPrompt("a", MakeTokens(12, 2000));
        SubmitPrompt("b", MakeTokens(8, 3000));
        SubmitPrompt("c", MakeTokens(6, 4000));
        SubmitPrompt("d", MakeTokens(5, 6000));
        EXPECT_EQ(Round(), (PrefillRows{{"d", 0, 5, 5}, {"c", 0, 6, 6}, {"b", 0, 8, 8}, {"holder", 32, 40, 100}}));
    }

    Restart(16);
    config_.enable_short_prefill_first = true;
    scheduler_ = std::make_unique<Scheduler>(config_);
    const std::vector<std::int32_t> junction = MakeTokens(8, 5000);
    Publish("earlier", Extending(junction, {5100, 5101, 5102}));
    SubmitPrompt("holder", Extending(junction, MakeTokens(92, 8000)));
    EXPECT_EQ(Round(), (PrefillRows{{"holder", 0, 8, 100}}));
    SubmitPrompt("sibling", Extending(junction, {6100, 6101}));
    SubmitPrompt("f", MakeTokens(14, 2000));
    EXPECT_EQ(Round(), (PrefillRows{{"f", 0, 14, 14}, {"sibling", 8, 10, 10}, {"holder", 8, 24, 100}}));
}

// In rank order too, the fill ends at its first capacity failure. In a
// 28-page pool no page is free once the 100-token prompt's second chunk and
// s1 and s2, which it gives way to, hold theirs. "a" ranks first, 1 token
// past the 40 "earlier" left cached, but its own hit keeps those pages from
// being evicted for the page it needs, so the fill ends there: "b", which
// could evict one, waits, though it fits the 2 tokens left and was submitted
// before a. "x" fits neither the knob beside s1 and s2 nor what is left.
TEST_F(PrefillChunkYieldSuite, TheRankedFillEndsAtItsFirstCapacityFailure) {
    pages_ = 28;
    Restart(16);
    config_.enable_short_prefill_first = true;
    scheduler_ = std::make_unique<Scheduler>(config_);
    const std::vector<std::int32_t> earlier = MakeTokens(40, 5000);
    Publish("earlier", earlier);
    SubmitPrompt("holder", MakeTokens(100, 1000));
    EXPECT_EQ(Round(), (PrefillRows{{"holder", 0, 32, 100}}));
    SubmitPrompt("s1", MakeTokens(8, 2000));
    SubmitPrompt("s2", MakeTokens(6, 3000));
    SubmitPrompt("x", MakeTokens(3, 4000));
    SubmitPrompt("b", MakeTokens(2, 6000));
    SubmitPrompt("a", Extending(earlier, {7001}));
    EXPECT_EQ(Round(), (PrefillRows{{"s1", 0, 8, 8}, {"s2", 0, 6, 6}, {"holder", 32, 48, 100}}));
}

// In submission order and with the knob at 0 too, the fill ends at its first
// capacity failure. Cut at its junction at 8, the 20-token prompt leaves 24
// tokens and b0 fills 4 of them; "a" fits the 20 left, but a 28-page pool has
// no room for it, so the fill ends there and b waits, though without a it
// fills 4 more.
TEST_F(PrefillChunkYieldSuite, TheFillEndsAtItsFirstCapacityFailure) {
    const auto run = [this](bool with_a) {
        hybrid_ = true;
        pages_ = 28;
        Restart(0);
        const std::vector<std::int32_t> shared = MakeTokens(8, 5000);
        Publish("earlier", Extending(shared, {5100, 5101, 5102}));
        SubmitPrompt("holder", Extending(shared, MakeTokens(12, 8000)));
        SubmitPrompt("b0", MakeTokens(4, 2000));
        if (with_a) {
            SubmitPrompt("a", MakeTokens(20, 3000));
        }
        SubmitPrompt("b", MakeTokens(4, 4000));
        return Round();
    };
    EXPECT_EQ(run(true), (PrefillRows{{"b0", 0, 4, 4}, {"holder", 0, 8, 20}}));
    EXPECT_EQ(run(false), (PrefillRows{{"b0", 0, 4, 4}, {"b", 0, 4, 4}, {"holder", 0, 8, 20}}));
}

// A resident chunk cut at its junction fills the budget it leaves too: the
// 100-token prompt's KV is cached to 40 but no State checkpoint on its path,
// so its second chunk gives way to s1 and s2 (s3 would pass the knob), still
// ends at 40, and s3 fills 10 of the 12 tokens left. With nothing to give way
// to, the same chunk leaves 24 tokens to a prompt larger than the knob.
TEST_F(PrefillChunkYieldSuite, AResidentChunkCutAtItsJunctionFillsTheBudgetItLeaves) {
    hybrid_ = true;
    const std::vector<std::int32_t> shared = MakeTokens(40, 5000);
    const auto start = [&] {
        Restart(16);
        Publish("earlier", Extending(shared, {5100, 5101, 5102}));
        SubmitPrompt("holder", Extending(shared, MakeTokens(60, 8000)));
        EXPECT_EQ(Round(), (PrefillRows{{"holder", 0, 32, 100}}));
    };
    start();
    SubmitPrompt("s1", MakeTokens(6, 2000));
    SubmitPrompt("s2", MakeTokens(6, 3000));
    SubmitPrompt("s3", MakeTokens(10, 4000));
    EXPECT_EQ(Round(), (PrefillRows{{"s1", 0, 6, 6}, {"s2", 0, 6, 6}, {"s3", 0, 10, 10}, {"holder", 32, 40, 100}}));
    start();
    SubmitPrompt("s4", MakeTokens(20, 6000));
    EXPECT_EQ(Round(), (PrefillRows{{"s4", 0, 20, 20}, {"holder", 32, 40, 100}}));
}

// A resident chunk whose rest fits the budget gives way to nothing, so the
// round's list is probed only after its admission: cut at its junction at 40,
// the chunk leaves 24 tokens, which a 20-token prompt fills, with the knob at
// 0 too.
TEST_F(PrefillChunkYieldSuite, AResidentChunkWhoseRestFitsTheBudgetFillsAfterItsAdmission) {
    const auto run = [this](std::int32_t yield) {
        hybrid_ = true;
        Restart(yield);
        const std::vector<std::int32_t> shared = MakeTokens(40, 5000);
        Publish("earlier", Extending(shared, {5100, 5101, 5102}));
        SubmitPrompt("holder", Extending(shared, MakeTokens(20, 8000)));
        std::vector<PrefillRows> rows{Round()};
        SubmitPrompt("s1", MakeTokens(20, 2000));
        rows.push_back(Round());
        rows.push_back(Round());
        return rows;
    };
    const std::vector<PrefillRows> filled{
        {{"holder", 0, 32, 60}}, {{"s1", 0, 20, 20}, {"holder", 32, 40, 60}}, {{"holder", 40, 60, 60}}};
    EXPECT_EQ(run(16), filled);
    EXPECT_EQ(run(0), filled);
}

// What fills the budget holds its pages until its transfer is acknowledged,
// like any completed prompt, so in a tight pool the chunk's next admission
// may wait for that: in a 34-page pool the 24-token prompt filling what the
// junction leaves keeps the chunk's third chunk out until its transfer is
// acknowledged.
TEST_F(PrefillChunkYieldSuite, AChunkWaitsForTheTransferOfWhatFilledItsBudgetInATightPool) {
    hybrid_ = true;
    pages_ = 34;
    Restart(16);
    const std::vector<std::int32_t> shared = MakeTokens(8, 5000);
    Publish("earlier", Extending(shared, {5100, 5101, 5102}));
    SubmitPrompt("holder", Extending(shared, MakeTokens(92, 8000)));
    SubmitPrompt("big", MakeTokens(24, 2000));
    EXPECT_EQ(Round(/*ack=*/false), (PrefillRows{{"big", 0, 24, 24}, {"holder", 0, 8, 100}}));
    EXPECT_EQ(Round(/*ack=*/false), (PrefillRows{{"holder", 8, 40, 100}}));
    EXPECT_EQ(handed_off_, std::vector<std::string>{"big"});
    EXPECT_EQ(Round(/*ack=*/false), PrefillRows{});
    SendPdSucceeded("big");
    EXPECT_EQ(Round(), (PrefillRows{{"holder", 40, 72, 100}}));
}

// A prompt sharing the chunk's junction is not carried: the junction's State
// is published only when the next chunk is admitted, so until then the
// prompt's own first chunk would end there. It resumes there next round,
// once the chunk's admission has published it.
TEST_F(PrefillChunkYieldSuite, APromptSharingTheChunksJunctionResumesThereNextRound) {
    hybrid_ = true;
    Restart(16);
    const std::vector<std::int32_t> shared = MakeTokens(8, 5000);
    Publish("earlier", Extending(shared, {5100, 5101, 5102}));
    SubmitPrompt("holder", Extending(shared, MakeTokens(12, 8000)));
    SubmitPrompt("sibling", Extending(shared, {6100, 6101}));
    SubmitPrompt("s1", MakeTokens(6, 2000));
    EXPECT_EQ(Round(), (PrefillRows{{"s1", 0, 6, 6}, {"holder", 0, 8, 20}}));
    EXPECT_EQ(Round(), (PrefillRows{{"holder", 8, 20, 20}, {"sibling", 8, 10, 10}}));
}

// Every carried prompt takes a batch row beside the long prompt's: with
// three rows two are carried.
TEST_F(PrefillChunkYieldSuite, TheBatchBoundsTheCarriedPrompts) {
    batch_ = 3;
    Restart(16);
    SubmitPrompt("long", MakeTokens(100, 1000));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 100}}));
    for (int i = 0; i < 3; ++i) {
        SubmitPrompt("s" + std::to_string(i), MakeTokens(4, 2000 + 100 * i));
    }
    EXPECT_EQ(Round(), (PrefillRows{{"s0", 0, 4, 4}, {"s1", 0, 4, 4}, {"long", 32, 56, 100}}));
}

// ... and a request slot: with two completed prompts holding theirs until
// their transfers are acknowledged, two of the five slots are free beside
// the long prompt's, though the batch has room for four.
TEST_F(PrefillChunkYieldSuite, TheFreeRequestSlotsBoundTheCarriedPrompts) {
    batch_ = 5;
    Restart(16);
    SubmitPrompt("p0", MakeTokens(4, 500));
    SubmitPrompt("p1", MakeTokens(4, 600));
    SubmitPrompt("long", MakeTokens(100, 1000));
    EXPECT_EQ(Round(), (PrefillRows{{"p0", 0, 4, 4}, {"p1", 0, 4, 4}, {"long", 0, 24, 100}}));
    for (int i = 0; i < 3; ++i) {
        SubmitPrompt("s" + std::to_string(i), MakeTokens(4, 2000 + 100 * i));
    }
    EXPECT_EQ(Round(/*ack=*/false), (PrefillRows{{"s0", 0, 4, 4}, {"s1", 0, 4, 4}, {"long", 24, 48, 100}}));
    EXPECT_EQ(handed_off_, (std::vector<std::string>{"p0", "p1"}));
}

// The fill behind a chunk cut at its junction is bounded the same way, the
// chunk counted once: with three batch rows two prompts fill its 24 tokens;
// with two of five request slots pinned by unacknowledged transfers, two
// more beside its own; with a 6-page KV budget, two 2-page prompts beside
// its 2 pages.
TEST_F(PrefillChunkYieldSuite, TheBatchSlotsAndKvPagesBoundTheFill) {
    hybrid_ = true;
    const std::vector<std::int32_t> shared = MakeTokens(8, 5000);
    const auto submit = [&] {
        SubmitPrompt("holder", Extending(shared, MakeTokens(12, 8000)));
        for (int i = 0; i < 3; ++i) {
            SubmitPrompt("s" + std::to_string(i), MakeTokens(6, 2000 + 100 * i));
        }
    };
    const PrefillRows filled{{"s0", 0, 6, 6}, {"s1", 0, 6, 6}, {"holder", 0, 8, 20}};
    batch_ = 3;
    Restart(16);
    Publish("earlier", Extending(shared, {5100, 5101, 5102}));
    submit();
    EXPECT_EQ(Round(), filled) << "three batch rows";

    batch_ = 5;
    Restart(16);
    Publish("earlier", Extending(shared, {5100, 5101, 5102}));
    SubmitPrompt("p0", MakeTokens(4, 500));
    SubmitPrompt("p1", MakeTokens(4, 600));
    EXPECT_EQ(Round(), (PrefillRows{{"p0", 0, 4, 4}, {"p1", 0, 4, 4}}));
    submit();
    EXPECT_EQ(Round(/*ack=*/false), filled) << "two free request slots";
    EXPECT_EQ(handed_off_, (std::vector<std::string>{"p0", "p1"}));

    batch_ = 8;
    kv_pages_ = 6;
    Restart(16);
    Publish("earlier", Extending(shared, {5100, 5101, 5102}));
    submit();
    EXPECT_EQ(Round(), filled) << "a 6-page KV budget";
}

// The fill takes from the round's one list of whole prompts, so it probes at
// most eight waiting prompts: behind seven too long to fill the 24 tokens the
// junction leaves, the eighth fills 4 of them and the ninth, never probed,
// waits, with the knob at 0 too.
TEST_F(PrefillChunkYieldSuite, TheFillProbesAtMostEightPrompts) {
    hybrid_ = true;
    for (const std::int32_t yield : {16, 0}) {
        SCOPED_TRACE(::testing::Message() << "knob " << yield);
        Restart(yield);
        const std::vector<std::int32_t> shared = MakeTokens(8, 5000);
        Publish("earlier", Extending(shared, {5100, 5101, 5102}));
        SubmitPrompt("holder", Extending(shared, MakeTokens(12, 8000)));
        for (int i = 1; i <= 7; ++i) {
            SubmitPrompt("long" + std::to_string(i), MakeTokens(40, 10000 + 100 * i));
        }
        SubmitPrompt("s8", MakeTokens(4, 2000));
        SubmitPrompt("s9", MakeTokens(4, 3000));
        EXPECT_EQ(Round(), (PrefillRows{{"s8", 0, 4, 4}, {"holder", 0, 8, 20}}));
    }
}

// The round's list is probed before the long chunk's admission, which needs
// three of the cached pages in an 18-page pool and evicts the oldest, the
// 12 tokens follow_up resumes at: it is not carried after all, since its
// whole prompt now takes 20 tokens, and s2, which the knob left out, takes
// its place. Nothing fails.
TEST_F(PrefillChunkYieldSuite, APromptWhosePrefixTheChunkEvictedIsNotCarried) {
    pages_ = 19;
    Restart(16);
    const std::vector<std::int32_t> earlier = MakeTokens(12, 5000);
    Publish("earlier", earlier);
    Publish("padding", MakeTokens(24, 6000));
    SubmitPrompt("long", MakeTokens(68, 1000));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 0, 32, 68}}));
    SubmitPrompt("follow_up", Extending(earlier, MakeTokens(8, 7000)));
    SubmitPrompt("s1", MakeTokens(8, 2000));
    SubmitPrompt("s2", MakeTokens(8, 3000));
    EXPECT_EQ(Round(), (PrefillRows{{"s1", 0, 8, 8}, {"s2", 0, 8, 8}, {"long", 32, 48, 68}}));
}

// Under a steady stream of short prompts the long prompt still advances by at
// least 32 - 16 - 3 = 13 tokens a round (16 here), so its 200 tokens take 12
// rounds -- within 1 + ceil((200 - 32) / 13) = 14, against 7 without the knob
// -- and every prompt carried beside it completes in that round, ahead of it.
TEST_F(PrefillChunkYieldSuite, ALongPromptStillFinishesUnderAStreamOfShortPrompts) {
    SubmitPrompt("long", MakeTokens(200, 1000));
    int rounds = 0;
    for (int next = 0; rounds < 20;) {
        for (int i = 0; i < 3; ++i, ++next) {
            SubmitPrompt("s" + std::to_string(next), MakeTokens(5, 10000 + 10 * next));
        }
        const PrefillRows rows = Round();
        ++rounds;
        ASSERT_FALSE(rows.empty());
        // The final chunk completes and runs first; every earlier one gives
        // way and runs last.
        if (rows[0].id == "long" && rows[0].end == 200) {
            break;
        }
        const PrefillRow& chunk = rows.back();
        ASSERT_EQ(chunk.id, "long") << "round " << rounds;
        EXPECT_GE(chunk.end - chunk.start, 13) << "round " << rounds;
        EXPECT_EQ(chunk.end % kPage, 0) << "round " << rounds;
        ASSERT_EQ(rows.size(), 4u) << "round " << rounds;
        for (std::size_t i = 0; i + 1 < rows.size(); ++i) {
            EXPECT_EQ(rows[i].start, 0);
            EXPECT_EQ(rows[i].end, rows[i].prompt) << rows[i].id << " is carried whole";
        }
    }
    EXPECT_EQ(rounds, 12);
}

// A round whose chunk ends where it would without the yield, and beside which
// no carried prompt runs, does not count toward
// prefill_chunk_yield_max_rounds. The resident chunk at 32 ends at its
// junction (40) however much budget it keeps, and follow_up, carried, is not
// admitted beside it in this 35-page pool. With a bound of one the long prompt
// therefore still gives way to s3 in the next round.
TEST_F(PrefillChunkYieldSuite, ARoundThatGaveNothingDoesNotCountTowardTheBound) {
    hybrid_ = true;
    pages_ = 35;
    max_rounds_ = 1;
    Restart(16);
    const std::vector<std::int32_t> shared = MakeTokens(40, 5000);
    const std::vector<std::int32_t> other = MakeTokens(12, 6000);
    Publish("other", other);
    Publish("earlier", Extending(shared, {5100, 5101, 5102}));
    SubmitPrompt("holder", Extending(shared, MakeTokens(60, 8000)));
    EXPECT_EQ(Round(), (PrefillRows{{"holder", 0, 32, 100}}));
    SubmitPrompt("follow_up", Extending(other, MakeTokens(4, 7000)));
    EXPECT_EQ(Round(), (PrefillRows{{"holder", 32, 40, 100}}));
    SubmitPrompt("s3", MakeTokens(6, 4000));
    EXPECT_EQ(Round(), (PrefillRows{{"s3", 0, 6, 6}, {"holder", 40, 64, 100}}));
}

// Two ranks' schedulers given the same events build the same plans.
TEST_F(PrefillChunkYieldSuite, MirroredSchedulersPlanAlike) {
    std::vector<std::unique_ptr<Scheduler>> ranks;
    ranks.push_back(std::make_unique<Scheduler>(config_));
    // So the second rank's allocations are not the first's at a fixed offset.
    const std::vector<std::uint8_t> heap_perturbation(4096);
    ranks.push_back(std::make_unique<Scheduler>(config_));
    const auto advance = [&ranks](const ExecutionEvent& event) {
        for (const std::unique_ptr<Scheduler>& rank : ranks) {
            rank->Advance(event);
        }
    };
    const auto submit = [&](const std::string& id, const std::vector<std::int32_t>& tokens) {
        for (const std::unique_ptr<Scheduler>& rank : ranks) {
            rank->SubmitRequests({RequestSpec{.request_id = id, .tokens = tokens}});
        }
        ExecutionEvent bootstrapped;
        bootstrapped.With(pd::BootstrappedEvent{id});
        advance(bootstrapped);
    };
    submit("long", MakeTokens(200, 1000));
    int carrying_rounds = 0;
    for (int round = 0, next = 0; round < 12; ++round) {
        for (int i = 0; i < 2; ++i, ++next) {
            submit("s" + std::to_string(next), MakeTokens(1 + next % 7, 10000 + 10 * next));
        }
        const Scheduler::PlanOptions options{.allow_prefill = true, .apply_prefill_watermark = false};
        const ExecutionPlan plan = ranks[0]->NextExecutionPlan(options);
        ASSERT_TRUE(ranks[1]->NextExecutionPlan(options) == plan) << "round " << round;
        if (plan.remote_decode) {
            ExecutionEvent acks;
            for (const std::string& id : plan.remote_decode->request_ids) {
                acks.With(pd::SucceededEvent{id});
            }
            advance(acks);
        }
        const ForwardBatch* batch = FindForwardBatch(plan);
        if (batch == nullptr || batch->NumExtends() == 0) {
            continue;
        }
        // The long prompt's chunk runs after the prompts it carries.
        EXPECT_TRUE(CompletingPrefillsFormAPrefix(*batch)) << "round " << round;
        const std::size_t last = batch->NumExtends() - 1;
        const bool last_incomplete =
            batch->extend_prefix_lens[last] + batch->input_lengths[last] < batch->prefill_lengths[last];
        carrying_rounds += last_incomplete && batch->NumExtends() > 1;
        ExecutionEvent results;
        for (std::size_t i = 0; i < batch->NumExtends(); ++i) {
            const bool complete = batch->extend_prefix_lens[i] + batch->input_lengths[i] == batch->prefill_lengths[i];
            results.With(forward::ExtendResult{
                .request_id = batch->request_ids[i],
                .tokens = complete ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{},
            });
        }
        advance(results);
    }
    EXPECT_GT(carrying_rounds, 0);
}

TEST_F(SchedulerTestSuite, SubmitRequestsRejectsEmptyTokens) {
    EXPECT_THROW(Submit(RequestSpec{.request_id = "empty"}), std::invalid_argument);
}

TEST_F(SchedulerTestSuite, SubmitRequestsValidatesWholeBatchBeforeInsertion) {
    const RequestSpec valid = MakeRequestSpec("valid", 1);
    RequestSpec invalid = MakeRequestSpec("invalid", 1);
    invalid.max_new_tokens = -1;

    EXPECT_THROW(Submit(std::vector<RequestSpec>{valid, invalid}), std::invalid_argument);
    EXPECT_NO_THROW(Submit(valid));
}

class HybridJunctionCutTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 128;
        cfg.disable_l2_cache = true;
        for (std::int32_t i = 0; i < 3; ++i) {
            CacheGroupConfig state = cfg.cache_groups.front();
            state.group_id = "linear_attention_" + std::to_string(i);
            state.attn_kind = AttnKind::kState;
            state.block_granularity = 0;
            cfg.cache_groups.push_back(std::move(state));
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    RequestSpec MakeHybridRequest(const std::string& id, std::int32_t suffix_start) {
        RequestSpec spec = MakeRequestSpec(id, /*num_pages=*/4);
        spec.tokens.push_back(suffix_start);
        spec.tokens.push_back(suffix_start + 1);
        spec.tokens.push_back(suffix_start + 2);
        return spec;
    }
};

// The seed publishes KV and state at its prompt end (11) only. The second
// request shares the first 8 tokens: KV reaches 8, no state does, so its first
// chunk is cut at the junction and publishes a state checkpoint there. The
// third request then resumes from 8 in both families.
TEST_F(HybridJunctionCutTestSuite, ThirdRequestResumesFromTheJunctionCheckpoint) {
    Submit(MakeHybridRequest("seed", 100));
    const ExecutionPlan seed_plan = PlanOnce();
    const ForwardBatch* seed = FindForwardBatch(seed_plan);
    ASSERT_NE(seed, nullptr);
    EXPECT_EQ(seed->input_lengths, std::vector<std::int32_t>{11});
    SendForwardDone("seed", {900});
    PlanOnce();
    SendFinish("seed");
    PlanOnce();

    Submit(MakeHybridRequest("junction", 200));
    const ExecutionPlan junction_plan = PlanOnce();
    const ForwardBatch* junction = FindForwardBatch(junction_plan);
    ASSERT_NE(junction, nullptr);
    ASSERT_EQ(junction->request_ids, std::vector<std::string>{"junction"});
    EXPECT_EQ(junction->input_lengths, std::vector<std::int32_t>{8});
    const ExecutionPlan remainder_plan = PlanOnce();
    const ForwardBatch* remainder = FindForwardBatch(remainder_plan);
    ASSERT_NE(remainder, nullptr);
    EXPECT_EQ(remainder->input_lengths, std::vector<std::int32_t>{3});
    SendForwardDone("junction", {901});
    PlanOnce();
    SendFinish("junction");
    PlanOnce();

    Submit(MakeHybridRequest("reuse", 300));
    const ExecutionPlan reuse_plan = PlanOnce();
    const ForwardBatch* reuse = FindForwardBatch(reuse_plan);
    ASSERT_NE(reuse, nullptr);
    ASSERT_EQ(reuse->request_ids, std::vector<std::string>{"reuse"});
    EXPECT_EQ(reuse->extend_prefix_lens, std::vector<std::int32_t>{8});
    for (std::int32_t i = 0; i < 3; ++i) {
        const auto row = reuse->BlockTables().at("linear_attention_" + std::to_string(i)).at(0);
        EXPECT_GT(row.at(0), 0) << "the junction checkpoint is the chunk's input";
    }
}

// The fused role fills nothing: the whole prompt behind a chunk cut at its
// junction waits for the round the chunk finishes in, though it fits the 56
// tokens that chunk leaves (on the P role it runs beside it,
// PrefillChunkYieldSuite).
TEST_F(HybridJunctionCutTestSuite, TheFusedRoleLeavesTheBudgetBehindAJunctionCutIdle) {
    Submit(MakeHybridRequest("seed", 100));
    PlanOnce();
    SendForwardDone("seed", {900});
    PlanOnce();
    SendFinish("seed");
    PlanOnce();

    Submit(MakeHybridRequest("junction", 200));
    Submit(MakeRequestSpec("short", /*num_pages=*/2, /*start=*/500));
    const ExecutionPlan junction_plan = PlanOnce();
    const ForwardBatch* junction = FindForwardBatch(junction_plan);
    ASSERT_NE(junction, nullptr);
    EXPECT_EQ(junction->request_ids, std::vector<std::string>{"junction"});
    EXPECT_EQ(junction->input_lengths, std::vector<std::int32_t>{8});
    const ExecutionPlan remainder_plan = PlanOnce();
    const ForwardBatch* remainder = FindForwardBatch(remainder_plan);
    ASSERT_NE(remainder, nullptr);
    EXPECT_EQ(remainder->request_ids, (std::vector<std::string>{"junction", "short"}));
    EXPECT_EQ(remainder->input_lengths, (std::vector<std::int32_t>{3, 4}));
}

// prefill_decode_interval: after a round that schedules a prefill chunk, the
// next N rounds that can schedule a decode run decodes alone, but no prompt
// is held back more than N rounds in total.
class PrefillDecodeIntervalSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.prefill_decode_interval = 2;
        // Two pages per chunk: a 6-page prompt takes three prefill rounds.
        cfg.max_scheduled_tokens = 4;
        return cfg;
    }

    // Plans one round, lands its forward and returns what the round ran:
    // "P:<ids>" when it scheduled a prefill chunk, "D:<ids>" for decodes only,
    // "-" for no forward. `chunking` names prompts whose chunk this round
    // leaves incomplete (an intermediate chunk produces no token). Retraction
    // images and restores the plan starts land with it.
    std::string Round(const std::set<std::string>& chunking = {}) {
        const ExecutionPlan plan = PlanOnce();
        AckImageStores(plan);
        AckRestores(plan);
        const ForwardBatch* batch = FindForwardBatch(plan);
        if (batch == nullptr || batch->request_ids.empty()) {
            return "-";
        }
        std::string ran = batch->NumExtends() > 0 ? "P:" : "D:";
        for (const std::string& id : batch->request_ids) {
            ran += id;
            SendForwardDone(id, chunking.contains(id) ? std::vector<std::int32_t>{} : std::vector<std::int32_t>{7});
        }
        return ran;
    }
};

TEST_F(PrefillDecodeIntervalSuite, ANewPromptWaitsForTheIntervalsDecodeRounds) {
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    EXPECT_EQ(Round(), "D:a");
    Submit(MakeRequestSpec("b", 2, /*start=*/100));
    EXPECT_EQ(Round(), "D:a");  // the second decode round owed since a's prefill
    EXPECT_EQ(Round(), "P:b");
    EXPECT_EQ(Round(), "D:ab");
}

TEST_F(PrefillDecodeIntervalSuite, ShortPrefillOrderingResumesAfterTheDecodeInterval) {
    auto cfg = MakeConfig();
    cfg.enable_short_prefill_first = true;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(cfg));
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("long", 6, /*start=*/100));
    Submit(MakeRequestSpec("short", 1, /*start=*/200));
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"long"}), "P:shortlong");
    // Both prompts used their interval allowance. The remaining long chunks
    // run without another decode delay, even after short's first admission.
    EXPECT_EQ(Round({"long"}), "P:long");
    EXPECT_EQ(Round({"long"}), "P:long");
    EXPECT_EQ(Round(), "P:long");
}

TEST_F(PrefillDecodeIntervalSuite, APromptIsHeldBackAtMostTheIntervalAcrossItsChunks) {
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 6, /*start=*/100));
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:b");
    // b's first chunk started a new interval, but b has already been held
    // back two rounds: the interval is suspended until b is done.
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "P:b");
    EXPECT_EQ(Round(), "D:ab");
}

TEST_F(PrefillDecodeIntervalSuite, SuspendedRoundsDoNotCountForAWaitingPrompt) {
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 6, /*start=*/100));
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:b");
    Submit(MakeRequestSpec("c", 1, /*start=*/200));
    // b's chunks fill each round's budget, so c waits behind them; the
    // interval is suspended, so these rounds do not count against c.
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "P:b");
    EXPECT_EQ(Round(), "D:ab");
    EXPECT_EQ(Round(), "D:ab");
    EXPECT_EQ(Round(), "P:c");
}

// max_consecutive_prefill_rounds: an overdue prompt suspends the interval
// for at most K back-to-back prefill rounds, and each chunk restarts its
// prompt's held-back count.
class CappedPrefillDecodeIntervalSuite : public PrefillDecodeIntervalSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillDecodeIntervalSuite::MakeConfig();
        cfg.max_consecutive_prefill_rounds = cap_;
        cfg.prefill_decode_min_interval = min_interval_;
        cfg.enable_mixed_prefill_decode = mixed_;
        // Pages for every prompt at once: no capacity retraction interleaves
        // its rounds with the interval's.
        cfg.device_allocator.total_pages = 64;
        cfg.cache_groups.front().total_pages = 64;
        SetTestSnapshotPool(cfg);
        if (mixed_) {
            // Room for a's decode beside a two-page chunk.
            cfg.max_scheduled_tokens = 5;
        }
        return cfg;
    }

    // Rebuilds the scheduler with the cap, the minimum interval and mixed
    // mode as given.
    void Restart(std::int32_t cap, std::int32_t min_interval = 0, bool mixed = false) {
        cap_ = cap;
        min_interval_ = min_interval;
        mixed_ = mixed;
        scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(MakeConfig()));
    }

    // Runs a decoding request beside three overdue three-chunk prompts until
    // they finish, and returns the longest run of back-to-back prefill
    // rounds. Every run holds at least one chunk and is followed by the
    // interval, so all nine chunks finish within 9 * (1 + interval) rounds,
    // and a prompt's first chunk runs within (1 + interval) rounds per chunk
    // queued ahead of it, plus the interval it arrived in.
    int LongestPrefillRunOfAQueue() {
        constexpr int kInterval = 2;  // the suite's prefill_decode_interval
        constexpr int kChunks = 3;
        const std::vector<std::string> prompts{"b", "c", "d"};
        Submit(MakeRequestSpec("a", 2));
        EXPECT_EQ(Round(), "P:a");
        Submit({MakeRequestSpec("b", 6, /*start=*/100), MakeRequestSpec("c", 6, /*start=*/200),
                MakeRequestSpec("d", 6, /*start=*/300)});
        std::map<std::string, int> chunks_run;
        std::map<std::string, int> first_round;
        int consecutive_prefill = 0;
        int longest = 0;
        int round = 0;
        for (; round < 60; ++round) {
            std::set<std::string> chunking;
            for (const std::string& id : prompts) {
                if (chunks_run[id] < kChunks - 1) {
                    chunking.insert(id);
                }
            }
            const std::string ran = Round(chunking);
            trace_ += ran + " ";
            EXPECT_NE(ran, "-");
            if (ran.starts_with("P:")) {
                longest = std::max(longest, ++consecutive_prefill);
                for (const std::string& id : prompts) {
                    if (ran.find(id, 2) != std::string::npos) {
                        first_round.try_emplace(id, round);
                        ++chunks_run[id];
                    }
                }
            } else {
                consecutive_prefill = 0;
            }
            if (std::ranges::all_of(prompts, [&](const std::string& id) { return chunks_run[id] == kChunks; })) {
                break;
            }
        }
        EXPECT_LT(round, kChunks * static_cast<int>(prompts.size()) * (1 + kInterval)) << trace_;
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const std::string& id = prompts[i];
            EXPECT_TRUE(first_round.contains(id)) << id << " was never admitted";
            EXPECT_LE(first_round[id], static_cast<int>(i) * kChunks * (1 + kInterval) + kInterval) << id;
        }
        return longest;
    }

    std::string trace_;
    std::int32_t cap_{2};
    std::int32_t min_interval_{0};
    bool mixed_{false};
};

// The cap (2) is never reached here: what bounds b is that each chunk
// restarts its held-back count.
TEST_F(CappedPrefillDecodeIntervalSuite, EachChunkRestartsItsPromptsHeldBackCount) {
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 6, /*start=*/100));
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:b");
    // Without the cap, b (held back two rounds in total) would suspend the
    // interval until it is done. Here its first chunk restarted the count,
    // so its next chunk waits the interval like a new prompt.
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "P:b");
    EXPECT_EQ(Round(), "D:ab");
}

// a decodes in every round, so every run of prefill rounds stalls it.
// Uncapped, the prompts held back since the queue formed would run all nine
// chunks back to back; capped, overdue prompts still suspend the interval up
// to the cap.
TEST_F(CappedPrefillDecodeIntervalSuite, AQueueOfLongPromptsRunsAtMostTheCapBackToBack) {
    for (const std::int32_t cap : {1, 2}) {
        SCOPED_TRACE(cap);
        Restart(cap);
        EXPECT_EQ(LongestPrefillRunOfAQueue(), cap);
    }
}

// With prefill_decode_min_interval every prefill round owes a decode round,
// so no run reaches a cap of two.
TEST_F(CappedPrefillDecodeIntervalSuite, TheMinimumIntervalStillOwesItsRoundsUnderTheCap) {
    Restart(/*cap=*/2, /*min_interval=*/1);
    EXPECT_EQ(LongestPrefillRunOfAQueue(), 1);
}

// Nothing decodes, so a run of prefill stalls nothing: the cap does not
// apply and the prompt's chunks run back to back.
TEST_F(CappedPrefillDecodeIntervalSuite, APrefillRunWithNothingToDecodeIsNotLimited) {
    Restart(/*cap=*/1);
    Submit(MakeRequestSpec("b", 6, /*start=*/100));
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "P:b");
    EXPECT_EQ(Round(), "D:b");
}

// In mixed mode a round that decodes beside its prefill ends the run, so the
// cap is never reached; each chunk still restarts its prompt's held-back
// count, so each waits the interval.
TEST_F(CappedPrefillDecodeIntervalSuite, InMixedModeEachChunkWaitsTheInterval) {
    Restart(/*cap=*/1, /*min_interval=*/0, /*mixed=*/true);
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 6, /*start=*/100));
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:ba");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:ba");
}

TEST(SchedulerConstructionTest, RefusesAPrefillRoundCapWithoutAnInterval) {
    const auto config = [](std::int32_t interval, std::int32_t cap) {
        SchedulerConfig cfg{};
        cfg.role = Role::kFused;
        cfg.prefix_granularity = 2;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.device_allocator.total_pages = 32;
        cfg.disable_l2_cache = true;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .block_granularity = cfg.prefix_granularity,
            .total_pages = 32,
        });
        cfg.max_resident_requests = cfg.max_batch_size;
        SetTestSnapshotPool(cfg);
        cfg.prefill_decode_interval = interval;
        cfg.max_consecutive_prefill_rounds = cap;
        return WithHostPackingFromDevice(cfg);
    };
    EXPECT_NO_THROW(config(4, 2).Validate());
    EXPECT_NO_THROW(config(0, 0).Validate());
    for (const auto& [interval, cap] : {std::pair{0, 2}, std::pair{4, -1}}) {
        try {
            config(interval, cap).Validate();
            ADD_FAILURE() << "max_consecutive_prefill_rounds " << cap << " with interval " << interval
                          << " was accepted";
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string{error.what()}.find("max_consecutive_prefill_rounds"), std::string::npos)
                << error.what();
        }
    }
}

// prefill_interval_min_decoding T (requires the cap K): the interval (and the
// minimum interval) is owed only while at least T requests are ready to
// decode before the round. With fewer, K prefill rounds in a row that hold a
// decode back are followed by one decode round; otherwise the round runs as
// if none were owed. 0 always owes the interval.
class PrefillIntervalMinDecodingSuite : public PrefillDecodeIntervalSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillDecodeIntervalSuite::MakeConfig();
        cfg.prefill_interval_min_decoding = min_decoding_;
        cfg.max_consecutive_prefill_rounds = cap_;
        cfg.max_batch_size = max_batch_size_;
        cfg.max_resident_requests = cfg.max_batch_size;
        // Pages for every prompt at once: no capacity retraction interleaves
        // its rounds with the interval's.
        cfg.device_allocator.total_pages = 64;
        cfg.cache_groups.front().total_pages = 64;
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Rebuilds the scheduler with the threshold, the cap and the batch slots
    // as given.
    void Restart(std::int32_t min_decoding, std::int32_t cap, std::int32_t max_batch_size) {
        min_decoding_ = min_decoding;
        cap_ = cap;
        max_batch_size_ = max_batch_size;
        scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(MakeConfig()));
    }

    std::int32_t min_decoding_{0};
    std::int32_t cap_{0};
    std::int32_t max_batch_size_{8};
};

// T = 0 owes the interval as before; so does any T the decodes meet (here 1,
// with a alone decoding).
TEST_F(PrefillIntervalMinDecodingSuite, AMetThresholdOwesTheIntervalAsBefore) {
    for (const std::int32_t min_decoding : {0, 1}) {
        SCOPED_TRACE(min_decoding);
        Restart(min_decoding, /*cap=*/4, /*max_batch_size=*/8);
        Submit(MakeRequestSpec("a", 2));
        EXPECT_EQ(Round(), "P:a");
        EXPECT_EQ(Round(), "D:a");
        Submit(MakeRequestSpec("b", 2, /*start=*/100));
        EXPECT_EQ(Round(), "D:a");
        EXPECT_EQ(Round(), "P:b");
        EXPECT_EQ(Round(), "D:ab");
    }
}

// One request decodes, fewer than T = 2: b is prefilled at once although a's
// prefill owes two decode rounds (the cap, 4, is not reached). b's prefill
// starts a new interval, and with a and b decoding c waits it out.
TEST_F(PrefillIntervalMinDecodingSuite, TooFewDecodesOweNoInterval) {
    Restart(/*min_decoding=*/2, /*cap=*/4, /*max_batch_size=*/8);
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 2, /*start=*/100));
    EXPECT_EQ(Round(), "P:b");
    Submit(MakeRequestSpec("c", 2, /*start=*/200));
    EXPECT_EQ(Round(), "D:ab");
    EXPECT_EQ(Round(), "D:ab");
    EXPECT_EQ(Round(), "P:c");
    EXPECT_EQ(Round(), "D:abc");
}

// Two requests decode, T = 2: the interval is owed as without the threshold.
TEST_F(PrefillIntervalMinDecodingSuite, EnoughDecodesOweTheInterval) {
    Restart(/*min_decoding=*/2, /*cap=*/4, /*max_batch_size=*/8);
    // One page each: both fit one round's budget.
    Submit({MakeRequestSpec("a", 1), MakeRequestSpec("b", 1, /*start=*/100)});
    EXPECT_EQ(Round(), "P:ab");
    Submit(MakeRequestSpec("c", 2, /*start=*/200));
    EXPECT_EQ(Round(), "D:ab");
    EXPECT_EQ(Round(), "D:ab");
    EXPECT_EQ(Round(), "P:c");
}

// The threshold changes only whether the interval is owed, not the request
// slots: a decoding and b mid-prefill fill both, so c waits for one even
// though too few decode for the interval.
TEST_F(PrefillIntervalMinDecodingSuite, AdmissionStillCountsPrefillingAndDecodingRequests) {
    Restart(/*min_decoding=*/3, /*cap=*/4, /*max_batch_size=*/2);
    Submit(MakeRequestSpec("a", 1));
    EXPECT_EQ(Round(), "P:a");
    // b's first chunk fills the budget; its 2-token remainder leaves room
    // for c's 2 tokens, but no slot.
    Submit({MakeRequestSpec("b", 3, /*start=*/100), MakeRequestSpec("c", 1, /*start=*/200)});
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "P:b");
    EXPECT_EQ(Round(), "D:ab");
    SendFinish("a");
    EXPECT_EQ(Round(), "P:c");
}

TEST(SchedulerConstructionTest, RefusesAPrefillIntervalMinDecodingItCannotApply) {
    const auto config = [](Role role, std::int32_t interval, std::int32_t min_interval, std::int32_t cap,
                           std::int32_t min_decoding) {
        SchedulerConfig cfg{};
        cfg.role = role;
        cfg.prefix_granularity = 2;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.max_resident_requests = cfg.max_batch_size;
        cfg.device_allocator.total_pages = 32;
        cfg.disable_l2_cache = true;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .block_granularity = cfg.prefix_granularity,
            .total_pages = 32,
        });
        if (role != Role::kFused) {
            cfg.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        }
        SetTestSnapshotPool(cfg);
        cfg.prefill_decode_interval = interval;
        cfg.prefill_decode_min_interval = min_interval;
        cfg.max_consecutive_prefill_rounds = cap;
        cfg.prefill_interval_min_decoding = min_decoding;
        return WithHostPackingFromDevice(cfg);
    };
    EXPECT_NO_THROW(config(Role::kFused, 4, 0, 1, 8).Validate());
    EXPECT_NO_THROW(config(Role::kFused, 4, 2, 1, 8).Validate());
    EXPECT_NO_THROW(config(Role::kP, 0, 0, 0, 0).Validate());
    EXPECT_NO_THROW(config(Role::kD, 0, 0, 0, 0).Validate());
    struct Refused {
        Role role;
        std::int32_t interval;
        std::int32_t min_interval;
        std::int32_t cap;
        std::int32_t min_decoding;
    };
    // P and D roles; no cap (with the interval, the minimum interval alone,
    // or nothing); a negative threshold.
    for (const Refused& refused :
         {Refused{Role::kP, 0, 0, 0, 8}, Refused{Role::kD, 0, 0, 0, 8}, Refused{Role::kFused, 4, 0, 0, 8},
          Refused{Role::kFused, 0, 2, 0, 8}, Refused{Role::kFused, 0, 0, 0, 8}, Refused{Role::kFused, 4, 0, 1, -1}}) {
        try {
            config(refused.role, refused.interval, refused.min_interval, refused.cap, refused.min_decoding).Validate();
            ADD_FAILURE() << "prefill_interval_min_decoding " << refused.min_decoding << " was accepted";
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string{error.what()}.find("prefill_interval_min_decoding"), std::string::npos)
                << error.what();
        }
    }
}

// Below T the interval stays waived, but K back-to-back prefill rounds that
// hold a decode back are followed by one decode round, which ends the run.
TEST_F(PrefillIntervalMinDecodingSuite, BelowTheThresholdTheCapStillEndsTheRun) {
    for (const std::int32_t cap : {1, 2, 4}) {
        SCOPED_TRACE(::testing::Message() << "cap " << cap);
        // T 16 is never met: at most 2 * cap + 1 requests decode.
        Restart(/*min_decoding=*/16, cap, /*max_batch_size=*/16);
        Submit(MakeRequestSpec("a", 2));
        EXPECT_EQ(Round(), "P:a");
        const std::string prompts = std::string{"bcdefghi"}.substr(0, 2 * cap);
        std::vector<RequestSpec> specs;
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            specs.push_back(MakeRequestSpec(std::string(1, prompts[i]), 2, /*start=*/100 * (i + 1)));
        }
        Submit(specs);
        std::string decoding = "a";
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const std::string id(1, prompts[i]);
            EXPECT_EQ(Round(), "P:" + id);
            decoding += id;
            // The run ends at the cap. The suite's 4-token budget decodes the
            // 4 oldest.
            if ((i + 1) % cap == 0) {
                EXPECT_EQ(Round(), "D:" + decoding.substr(0, 4));
            }
        }
        EXPECT_EQ(Round(), "D:" + decoding.substr(0, 4));
    }
}

// With K = 1, b's prefill stalls a and reaches the cap: below T = 3 the next
// round decodes once, and c's prefill
// starts a new run. With a, b and c decoding T is met, and the interval c's
// prefill holds is owed in full, as without the threshold.
TEST_F(PrefillIntervalMinDecodingSuite, TheForcedDecodeEndsTheRunAndTheThresholdOwesTheHeldInterval) {
    Restart(/*min_decoding=*/3, /*cap=*/1, /*max_batch_size=*/8);
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit({MakeRequestSpec("b", 2, /*start=*/100), MakeRequestSpec("c", 2, /*start=*/200)});
    EXPECT_EQ(Round(), "P:b");
    EXPECT_EQ(Round(), "D:ab");
    EXPECT_EQ(Round(), "P:c");
    Submit(MakeRequestSpec("d", 2, /*start=*/300));
    EXPECT_EQ(Round(), "D:abc");
    EXPECT_EQ(Round(), "D:abc");
    EXPECT_EQ(Round(), "P:d");
}

// The run reached the cap, but its decodes finished before the next round:
// with nothing to decode the round runs prefill work, it does not idle.
TEST_F(PrefillIntervalMinDecodingSuite, TheCapForcesNoRoundWithNothingToDecode) {
    Restart(/*min_decoding=*/8, /*cap=*/1, /*max_batch_size=*/8);
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 2, /*start=*/100));
    EXPECT_EQ(Round(), "P:b");
    SendFinish("a");
    SendFinish("b");
    Submit(MakeRequestSpec("c", 2, /*start=*/200));
    EXPECT_EQ(Round(), "P:c");
}

// At or above T the threshold changes nothing: two requests decode, T = 2,
// K = 1, and the held interval is owed as without the threshold.
TEST_F(PrefillIntervalMinDecodingSuite, AMetThresholdOwesTheHeldInterval) {
    Restart(/*min_decoding=*/2, /*cap=*/1, /*max_batch_size=*/8);
    Submit({MakeRequestSpec("a", 1), MakeRequestSpec("b", 1, /*start=*/100)});
    EXPECT_EQ(Round(), "P:ab");
    Submit({MakeRequestSpec("c", 2, /*start=*/200), MakeRequestSpec("d", 2, /*start=*/300)});
    EXPECT_EQ(Round(), "D:ab");
    EXPECT_EQ(Round(), "D:ab");
    EXPECT_EQ(Round(), "P:c");
    EXPECT_EQ(Round(), "D:abc");
    EXPECT_EQ(Round(), "D:abc");
    EXPECT_EQ(Round(), "P:d");
}

// The invariants of docs/design/scheduler.md §3.3 that the threshold touches:
// the minimum interval, the held-back bound below T, retraction, mixed mode
// and the roles. Adds the minimum interval, mixed mode and forced retraction
// to the threshold's suite.
class PrefillIntervalMinDecodingInvariantSuite : public PrefillIntervalMinDecodingSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillIntervalMinDecodingSuite::MakeConfig();
        cfg.prefill_decode_min_interval = min_interval_;
        cfg.enable_mixed_prefill_decode = mixed_;
        if (mixed_) {
            // Room for the decodes beside a two-page chunk.
            cfg.max_scheduled_tokens = 6;
        }
        cfg.debug_force_retraction_interval = force_retraction_;
        if (force_retraction_ != 0) {
            cfg.debug_force_retraction_max_per_request = 1;
        }
        return cfg;
    }

    // Rebuilds the scheduler with the threshold, the cap, the minimum
    // interval, mixed mode and the forced retraction period as given.
    void Restart(std::int32_t min_decoding, std::int32_t cap, std::int32_t min_interval, bool mixed,
                 std::int32_t force_retraction) {
        min_interval_ = min_interval;
        mixed_ = mixed;
        force_retraction_ = force_retraction;
        PrefillIntervalMinDecodingSuite::Restart(min_decoding, cap, /*max_batch_size=*/8);
    }

    std::int32_t min_interval_{0};
    bool mixed_{false};
    std::int32_t force_retraction_{0};
};

// The minimum interval M is waived below T like the interval: with a alone
// decoding (below T 3), b's three chunks run back to back, which M alone
// never allows while a decode can run. With a, b and c decoding T is met and
// M is owed again: e waits M's three decode rounds, more than the interval's
// two.
TEST_F(PrefillIntervalMinDecodingInvariantSuite, BelowTheThresholdTheMinimumIntervalIsWaived) {
    Restart(/*min_decoding=*/3, /*cap=*/8, /*min_interval=*/3, /*mixed=*/false, /*force_retraction=*/0);
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 6, /*start=*/100));
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "P:b");
    Submit(MakeRequestSpec("c", 2, /*start=*/200));
    EXPECT_EQ(Round(), "P:c");
    Submit(MakeRequestSpec("e", 2, /*start=*/300));
    EXPECT_EQ(Round(), "D:abc");
    EXPECT_EQ(Round(), "D:abc");
    EXPECT_EQ(Round(), "D:abc");
    EXPECT_EQ(Round(), "P:e");
    EXPECT_EQ(Round(), "D:abce");
}

// Below T the decode round K forces is not suspended by an overdue prompt:
// c, queued behind b's three chunks with K 1, counts each forced round as
// held back, three in all, past the interval's N 2. It is still held only
// one round per K prefill rounds, never two decode rounds in a row, so its
// wait is bounded by the chunks ahead of it, about one round per chunk.
TEST_F(PrefillIntervalMinDecodingInvariantSuite, BelowTheThresholdAWaitingPromptIsHeldOneRoundPerCap) {
    Restart(/*min_decoding=*/16, /*cap=*/1, /*min_interval=*/0, /*mixed=*/false, /*force_retraction=*/0);
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit({MakeRequestSpec("b", 6, /*start=*/100), MakeRequestSpec("c", 2, /*start=*/200)});
    std::vector<std::string> ran;
    ran.push_back(Round({"b"}));
    ran.push_back(Round());
    ran.push_back(Round({"b"}));
    ran.push_back(Round());
    ran.push_back(Round());
    ran.push_back(Round());
    ran.push_back(Round());
    ran.push_back(Round());
    EXPECT_EQ(ran, (std::vector<std::string>{"P:b", "D:a", "P:b", "D:a", "P:b", "D:ab", "P:c", "D:abc"}));
    const auto before_c = std::ranges::find(ran, "P:c");
    const auto held = std::count_if(ran.begin(), before_c, [](const std::string& round) { return round[0] == 'D'; });
    EXPECT_GT(held, 2) << "c's held-back count stayed within N";
    for (auto round = ran.begin(); round + 1 < before_c; ++round) {
        EXPECT_FALSE((*round)[0] == 'D' && (*(round + 1))[0] == 'D') << "c was held two rounds in a row";
    }
}

// A retracted decode is not ready to decode, so it leaves the count: a and b
// decode (T 2 met) and c waits the interval (EnoughDecodesOweTheInterval
// runs D:ab twice). The forced retraction of a in the third plan drops the
// count to 1, so that round waives the rest of the interval and runs c. a is
// restored beside the next decode; with b and c decoding the count is met
// again, and d waits the whole interval c's prefill started.
TEST_F(PrefillIntervalMinDecodingInvariantSuite, ARetractedDecodeLeavesTheThresholdCount) {
    Restart(/*min_decoding=*/2, /*cap=*/4, /*min_interval=*/0, /*mixed=*/false, /*force_retraction=*/3);
    Submit({MakeRequestSpec("a", 1), MakeRequestSpec("b", 1, /*start=*/100)});
    EXPECT_EQ(Round(), "P:ab");
    Submit(MakeRequestSpec("c", 2, /*start=*/200));
    EXPECT_EQ(Round(), "D:ab");
    EXPECT_EQ(Round(), "P:c");  // plan 3 retracts a, the oldest decode
    Submit(MakeRequestSpec("d", 2, /*start=*/300));
    EXPECT_EQ(Round(), "D:bc");  // a's restore rides beside the decode
    EXPECT_EQ(Round(), "D:abc");
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(Round(), "P:d");
}

// In mixed mode below T nothing is owed, so b's chunks ride beside a's decode
// at once instead of waiting the interval (CappedPrefillDecodeIntervalSuite.
// InMixedModeEachChunkWaitsTheInterval). Every such round decodes, which ends
// the run, so K never forces a round: the threshold behaves as if K were off.
TEST_F(PrefillIntervalMinDecodingInvariantSuite, InMixedModeTooFewDecodesOweNoIntervalAndTheCapIsNeverReached) {
    Restart(/*min_decoding=*/3, /*cap=*/1, /*min_interval=*/0, /*mixed=*/true, /*force_retraction=*/0);
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 6, /*start=*/100));
    EXPECT_EQ(Round({"b"}), "P:ba");
    EXPECT_EQ(Round({"b"}), "P:ba");
    EXPECT_EQ(Round(), "P:ba");
    EXPECT_EQ(Round(), "D:ab");
}
class PrefillDecodeIntervalTwoSlotsSuite : public PrefillDecodeIntervalSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillDecodeIntervalSuite::MakeConfig();
        cfg.max_batch_size = 2;
        cfg.max_resident_requests = cfg.max_batch_size;
        return cfg;
    }
};

TEST_F(PrefillDecodeIntervalTwoSlotsSuite, APromptWaitingForASlotCountsTheRoundsItIsHeldBack) {
    Submit(MakeRequestSpec("a", 1));
    EXPECT_EQ(Round(), "P:a");
    Submit({MakeRequestSpec("b", 6, /*start=*/100), MakeRequestSpec("c", 1, /*start=*/200)});
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    // b takes the second slot; c, held back two rounds, waits for a slot and
    // suspends the interval meanwhile.
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "P:b");
    EXPECT_EQ(Round(), "D:ab");
    SendFinish("a");
    // b's last chunk started an interval that still owes a round, but c is
    // overdue: it runs as soon as it has a slot.
    EXPECT_EQ(Round(), "P:c");
}

TEST_F(PrefillDecodeIntervalSuite, APromptRunsAtOnceWhenNoDecodeIsRunning) {
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    SendFinish("a");
    Submit(MakeRequestSpec("b", 2, /*start=*/100));
    // Two decode rounds are owed, but nothing can decode: the engine would
    // idle, so the prefill runs.
    EXPECT_EQ(Round(), "P:b");
}

TEST_F(PrefillDecodeIntervalTwoSlotsSuite, SuspendedDecodeRoundsDoNotCountForAWaitingPrompt) {
    Submit(MakeRequestSpec("a", 1));
    EXPECT_EQ(Round(), "P:a");
    Submit({MakeRequestSpec("b", 6, /*start=*/100), MakeRequestSpec("c", 1, /*start=*/200)});
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "P:b");
    Submit(MakeRequestSpec("d", 1, /*start=*/300));
    // c, overdue and without a slot, suspends the interval: these decode
    // rounds do not count against d.
    EXPECT_EQ(Round(), "D:ab");
    EXPECT_EQ(Round(), "D:ab");
    SendFinish("a");
    EXPECT_EQ(Round(), "P:c");
    EXPECT_EQ(Round(), "D:bc");
    SendFinish("b");
    // d has been held back one round, so the interval still holds it.
    EXPECT_EQ(Round(), "D:c");
    EXPECT_EQ(Round(), "P:d");
}

class PrefillDecodeIntervalRetractionSuite : public PrefillDecodeIntervalSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillDecodeIntervalSuite::MakeConfig();
        cfg.debug_force_retraction_interval = -5;  // the oldest Prefilling request, every fifth plan
        return cfg;
    }
};

TEST_F(PrefillDecodeIntervalRetractionSuite, ARestoredPromptKeepsTheRoundsItWasHeldBack) {
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 6, /*start=*/100));
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "D:a");  // plan 5 retracts b
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    EXPECT_EQ(Round(), "D:a");  // b's restore rides beside the decode
    ASSERT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(Round({"b"}), "P:b");
    // b's chunk started a new interval, but b keeps the two rounds it was
    // held back before its retraction.
    EXPECT_EQ(Round(), "P:b");
    EXPECT_EQ(Round(), "D:ab");
}

class PrefillDecodeIntervalCapacitySuite : public PrefillDecodeIntervalSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillDecodeIntervalSuite::MakeConfig();
        cfg.prefill_decode_interval = 9;
        cfg.max_scheduled_tokens = 8;
        cfg.device_allocator.total_pages = 14;
        cfg.cache_groups.front().total_pages = 14;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(PrefillDecodeIntervalCapacitySuite, APromptRetractedWhileHeldBackCountsThatRound) {
    Submit({MakeRequestSpec("a", 2), MakeRequestSpec("d", 1, /*start=*/300), MakeRequestSpec("b", 6, /*start=*/100)});
    EXPECT_EQ(Round({"b"}), "P:adb");
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(Round(), "D:ad");
    }
    // The ninth round b is held back, a's next page retracts it.
    EXPECT_EQ(Round(), "D:ad");
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    SendFinish("a");
    EXPECT_EQ(Round(), "D:d");  // b's restore rides beside the decode
    EXPECT_EQ(Round({"b"}), "P:b");
    // b's chunk started a new interval, but the round b was retracted in
    // counts: b has been held back nine rounds and runs on.
    EXPECT_EQ(Round(), "P:b");
}

class MixedPrefillDecodeIntervalSuite : public PrefillDecodeIntervalSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillDecodeIntervalSuite::MakeConfig();
        cfg.enable_mixed_prefill_decode = true;
        cfg.max_scheduled_tokens = 64;
        return cfg;
    }
};

TEST_F(MixedPrefillDecodeIntervalSuite, ADeferredPromptJoinsNoDecodeRound) {
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 2, /*start=*/100));
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "P:ba");  // the mixed round: prefill rows first
}

// Ten pages: few enough that a running decode runs out of pages while a
// prompt waits.
class MixedPrefillDecodeIntervalCapacitySuite : public PrefillDecodeIntervalSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillDecodeIntervalSuite::MakeConfig();
        cfg.enable_mixed_prefill_decode = true;
        cfg.prefill_decode_interval = 4;
        cfg.max_scheduled_tokens = 8;
        cfg.device_allocator.total_pages = 10;
        cfg.cache_groups.front().total_pages = 10;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(MixedPrefillDecodeIntervalCapacitySuite, AGrantedFirstAdmissionCountsTowardOldestProtection) {
    for (const bool abort_victim : {false, true}) {
        SCOPED_TRACE(abort_victim ? "abort grant" : "retraction grant");
        auto cfg = MakeConfig();
        cfg.prefill_decode_interval = 0;
        cfg.enable_short_prefill_first = true;
        if (abort_victim) {
            cfg.snapshot_allocator.total_pages = 1;
            cfg.max_retracted_requests = 0;
        }
        scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(cfg));
        Submit(
            {MakeRequestSpec("c", 1, /*start=*/300), MakeRequestSpec("e", 1, /*start=*/500), MakeRequestSpec("a", 2)});
        EXPECT_EQ(Round(), "P:cea");
        EXPECT_EQ(Round(), "D:cea");
        EXPECT_EQ(Round(), "D:cea");

        Submit(MakeRequestSpec("old", 6, /*start=*/100));
        Submit(MakeRequestSpec("short", 4, /*start=*/200));
        // Neither prompt fits until a gives up its pages. The shorter prompt receives
        // the freed capacity after its ordinary admission failed this round.
        const auto grant = PlanOnce();
        const auto* batch = FindForwardBatch(grant);
        ASSERT_NE(batch, nullptr);
        EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"short", "c", "e"}));
        ASSERT_EQ(grant.aborts.size(), abort_victim ? 1u : 0u);
        if (abort_victim) {
            EXPECT_EQ(grant.aborts.front().request_id, "a");
            EXPECT_EQ(grant.aborts.front().reason, AbortReason::kImageDoesNotFit);
        }
        ASSERT_EQ(scheduler_->RetractedSize(), abort_victim ? 0u : 1u);
        AckImageStores(grant);
        SendForwardDone("short");
        SendForwardDone("c", {7});
        SendForwardDone("e", {7});
        SendFinish("c");
        SendFinish("e");
        if (!abort_victim) {
            SendAbortEvent("a");
        }
        SendAbortEvent("short");

        // The grant was the first bypass. Seven ordinary admissions fill the
        // remaining allowance; each consumes the round's entire token budget.
        for (int i = 0; i < 7; ++i) {
            const std::string id = "s" + std::to_string(i);
            Submit(MakeRequestSpec(id, 4, /*start=*/1000 + i * 100));
            EXPECT_EQ(Round(), "P:" + id);
            SendFinish(id);
        }
        Submit(MakeRequestSpec("later", 4, /*start=*/2000));
        EXPECT_EQ(Round({"old"}), "P:old");
    }
}

TEST_F(MixedPrefillDecodeIntervalCapacitySuite, AGrantAdmittedPromptIsNotHeldBackThatRound) {
    Submit({MakeRequestSpec("c", 1, /*start=*/300), MakeRequestSpec("e", 1, /*start=*/500), MakeRequestSpec("a", 2)});
    EXPECT_EQ(Round(), "P:cea");
    Submit(MakeRequestSpec("b", 4, /*start=*/100));
    EXPECT_EQ(Round(), "D:cea");
    EXPECT_EQ(Round(), "D:cea");
    // The interval still owes this round, but a's next page does not fit:
    // the scheduler retracts a and grants its pages to b, whose first chunk
    // runs beside the decodes and starts a new interval.
    EXPECT_EQ(Round({"b"}), "P:bce");
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    SendFinish("e");  // frees the pages b's next chunk needs
    // b has been held back two rounds, not three: the new interval holds its
    // next chunk for two more.
    EXPECT_EQ(Round(), "D:c");
    EXPECT_EQ(Round(), "D:c");
    EXPECT_EQ(Round(), "P:bc");
}

class PrefillDecodeMinIntervalSuite : public PrefillDecodeIntervalSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillDecodeIntervalSuite::MakeConfig();
        cfg.prefill_decode_min_interval = 1;
        return cfg;
    }
};

TEST_F(PrefillDecodeMinIntervalSuite, AnOverduePromptsChunksAlternateWithDecodeRounds) {
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 6, /*start=*/100));
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:b");
    // b has been held back two rounds and suspends the interval, but every
    // chunk still owes the minimum's decode round.
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "P:b");
    EXPECT_EQ(Round(), "D:ab");
}

TEST_F(PrefillDecodeMinIntervalSuite, TheMinimumAloneOwesItsRoundsAfterEveryChunk) {
    auto cfg = MakeConfig();
    cfg.prefill_decode_interval = 0;
    scheduler_ = std::make_unique<Scheduler>(WithHostPackingFromDevice(cfg));
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 6, /*start=*/100));
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:b");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "P:b");
    EXPECT_EQ(Round(), "D:ab");
}

TEST_F(PrefillDecodeMinIntervalSuite, APromptRunsAtOnceWhenNoDecodeIsRunning) {
    Submit(MakeRequestSpec("a", 6));
    EXPECT_EQ(Round({"a"}), "P:a");
    // The minimum owes a decode round, but a is the only request and has
    // nothing to decode yet: its next chunk runs.
    EXPECT_EQ(Round({"a"}), "P:a");
    EXPECT_EQ(Round(), "P:a");
    EXPECT_EQ(Round(), "D:a");
}

class MixedPrefillDecodeMinIntervalSuite : public PrefillDecodeMinIntervalSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillDecodeMinIntervalSuite::MakeConfig();
        cfg.enable_mixed_prefill_decode = true;
        // One decode token and a two-page chunk.
        cfg.max_scheduled_tokens = 5;
        return cfg;
    }
};

TEST_F(MixedPrefillDecodeMinIntervalSuite, AnOverduePromptsMixedRoundsAlternateWithDecodeRounds) {
    Submit(MakeRequestSpec("a", 2));
    EXPECT_EQ(Round(), "P:a");
    Submit(MakeRequestSpec("b", 6, /*start=*/100));
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:ba");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round({"b"}), "P:ba");
    EXPECT_EQ(Round(), "D:a");
    EXPECT_EQ(Round(), "P:ba");
    EXPECT_EQ(Round(), "D:ab");
}

TEST(SchedulerConstructionTest, RefusesAPrefillDecodeMinIntervalOutsideTheFusedRole) {
    const auto config = [](Role role, std::int32_t interval) {
        SchedulerConfig cfg{};
        cfg.role = role;
        cfg.prefix_granularity = 2;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.max_resident_requests = cfg.max_batch_size;
        cfg.device_allocator.total_pages = 32;
        cfg.disable_l2_cache = true;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .block_granularity = cfg.prefix_granularity,
            .total_pages = 32,
        });
        if (role != Role::kFused) {
            cfg.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        }
        SetTestSnapshotPool(cfg);
        cfg.prefill_decode_min_interval = interval;
        return WithHostPackingFromDevice(cfg);
    };
    EXPECT_NO_THROW(config(Role::kFused, 2).Validate());
    for (const auto& [role, interval] : {std::pair{Role::kP, 2}, std::pair{Role::kD, 2}, std::pair{Role::kFused, -1}}) {
        try {
            config(role, interval).Validate();
            ADD_FAILURE() << "prefill_decode_min_interval " << interval << " was accepted";
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string{error.what()}.find("prefill_decode_min_interval"), std::string::npos) << error.what();
        }
    }
}

TEST(SchedulerConstructionTest, RefusesAPrefillDecodeIntervalOutsideTheFusedRole) {
    const auto config = [](Role role, std::int32_t interval) {
        SchedulerConfig cfg{};
        cfg.role = role;
        cfg.prefix_granularity = 2;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.max_resident_requests = cfg.max_batch_size;
        cfg.device_allocator.total_pages = 32;
        cfg.disable_l2_cache = true;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .block_granularity = cfg.prefix_granularity,
            .total_pages = 32,
        });
        if (role != Role::kFused) {
            cfg.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        }
        SetTestSnapshotPool(cfg);
        cfg.prefill_decode_interval = interval;
        return WithHostPackingFromDevice(cfg);
    };
    EXPECT_NO_THROW(config(Role::kFused, 4).Validate());
    for (const auto& [role, interval] : {std::pair{Role::kP, 4}, std::pair{Role::kD, 4}, std::pair{Role::kFused, -1}}) {
        try {
            config(role, interval).Validate();
            ADD_FAILURE() << "prefill_decode_interval " << interval << " was accepted";
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string{error.what()}.find("prefill_decode_interval"), std::string::npos) << error.what();
        }
    }
}

// A prompt in three chunks on 4-token pages: its first admission prepays the
// whole prompt and its generation, the block of its partial last page
// [8, 10) included, and reserves the copy-in block for that block with it.
// The later chunks allocate nothing, and the first decode publishes the
// prompt end and copies the published block into the copy-in block.
class MultiChunkCopyInSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.prefix_granularity = 4;
        cfg.cache_groups.front().block_granularity = 4;
        cfg.disable_l2_cache = true;
        cfg.max_scheduled_tokens = 4;
        return cfg;
    }

    // Parents held neither by a request's table nor as evictable cache.
    std::int32_t HeldOutsideTables() const {
        return config_.device_allocator.NumUsableBlocks() - scheduler_->AvailableLcmBlocks() -
               scheduler_->ActiveLcmBlocks();
    }
};

TEST_F(MultiChunkCopyInSuite, TheFirstAdmissionReservesThePromptEndsCopyIn) {
    Submit(RequestSpec{.request_id = "r", .tokens = MakeTokens(10, 1), .max_new_tokens = 2});
    const ExecutionPlan first = PlanOnce();
    ASSERT_EQ(FindForwardBatch(first)->input_lengths, std::vector<std::int32_t>{4});
    EXPECT_EQ(first.pages_to_zero.at("full_attention").size(), 3u) << "the prompt and its generation, prepaid";
    EXPECT_EQ(HeldOutsideTables(), 1) << "the copy-in block of the page [8, 12)";
    SendForwardDone("r");
    for (const std::int32_t tokens : {4, 2}) {
        const ExecutionPlan chunk = PlanOnce();
        ASSERT_EQ(FindForwardBatch(chunk)->input_lengths, std::vector<std::int32_t>{tokens});
        EXPECT_TRUE(chunk.pages_to_zero.at("full_attention").empty()) << "a later chunk needs no block";
        EXPECT_EQ(HeldOutsideTables(), 1);
        SendForwardDone("r", tokens == 2 ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
    }
    const ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(decode), nullptr);
    EXPECT_EQ(decode.pages_to_copy.at("full_attention").size(), 1u) << "the prompt end's block moves to its copy";
    EXPECT_TRUE(decode.pages_to_zero.at("full_attention").empty());
    EXPECT_EQ(HeldOutsideTables(), 0);
    EXPECT_EQ(scheduler_->TailEntries(), 1u);
}

TEST(SchedulerConstructionTest, ValidatesConfigBeforeBuildingPools) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = 2;
    cfg.max_scheduled_tokens = 64;
    cfg.max_batch_size = 8;
    cfg.max_resident_requests = cfg.max_batch_size;
    cfg.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full_attention",
        .block_granularity = cfg.prefix_granularity,
        .total_pages = 32,
    });
    // device_allocator.total_pages stays 0, so the block pool would be built
    // with a negative usable count and assert before the config diagnostic.
    try {
        SetTestSnapshotPool(cfg);
        Scheduler scheduler{WithHostPackingFromDevice(cfg)};
        FAIL() << "a device cache without usable capacity was accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string{error.what()}.find("device cache"), std::string::npos) << error.what();
    }
}

// prefill_chunk_yield_tokens is off by default on every role, and on is a P
// role option of at most half the chunk budget.
TEST(SchedulerConstructionTest, PrefillChunkYieldIsAPrefillRoleOptionOfAtMostHalfTheBudget) {
    const auto config = [](Role role, std::int32_t budget, std::int32_t yield) {
        SchedulerConfig cfg{};
        cfg.role = role;
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 32;
        cfg.disable_l2_cache = true;
        cfg.max_scheduled_tokens = budget;
        cfg.max_batch_size = 8;
        cfg.max_resident_requests = cfg.max_batch_size;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .block_granularity = cfg.prefix_granularity,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kFull,
            .transfer_policy =
                role == Role::kFused ? CacheTransferPolicy::Unspecified : CacheTransferPolicy::FullSuffix,
        });
        cfg.prefill_chunk_yield_tokens = yield;
        SetTestSnapshotPool(cfg);
        return WithHostPackingFromDevice(cfg);
    };
    const auto refusal = [&](Role role, std::int32_t budget, std::int32_t yield) -> std::string {
        try {
            config(role, budget, yield).Validate();
        } catch (const std::invalid_argument& error) {
            return error.what();
        }
        return {};
    };
    for (const Role role : {Role::kP, Role::kD, Role::kFused}) {
        EXPECT_EQ(SchedulerConfig{}.prefill_chunk_yield_tokens, 0);
        EXPECT_EQ(refusal(role, 64, 0), "") << "off by default";
        EXPECT_NE(refusal(role, 64, -1).find("must be >= 0"), std::string::npos);
    }
    EXPECT_NE(refusal(Role::kFused, 64, 8).find("P role only"), std::string::npos);
    EXPECT_NE(refusal(Role::kD, 64, 8).find("P role only"), std::string::npos);
    EXPECT_EQ(refusal(Role::kP, 64, 32), "") << "half the budget";
    EXPECT_NE(refusal(Role::kP, 64, 33).find("at most half"), std::string::npos);
    EXPECT_EQ(refusal(Role::kP, 65, 32), "");
    EXPECT_NE(refusal(Role::kP, 65, 33).find("at most half"), std::string::npos);
    // The Scheduler refuses it before building anything.
    EXPECT_THROW(Scheduler{config(Role::kFused, 64, 8)}, std::invalid_argument);
    EXPECT_NO_THROW(Scheduler{config(Role::kP, 64, 32)});
}

// prefill_chunk_yield_max_rounds is unbounded by default and must be positive
// while the yield is on; without the yield it is inert.
TEST(SchedulerConstructionTest, PrefillChunkYieldMaxRoundsIsPositiveWhileTheYieldIsOn) {
    const auto refusal = [](std::int32_t yield, std::int32_t max_rounds) {
        SchedulerConfig cfg{};
        cfg.role = Role::kP;
        cfg.prefix_granularity = 2;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.device_allocator.total_pages = 32;
        cfg.disable_l2_cache = true;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .block_granularity = cfg.prefix_granularity,
            .total_pages = 32,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        cfg.max_resident_requests = cfg.max_batch_size;
        cfg.prefill_chunk_yield_tokens = yield;
        cfg.prefill_chunk_yield_max_rounds = max_rounds;
        SetTestSnapshotPool(cfg);
        try {
            WithHostPackingFromDevice(cfg).Validate();
        } catch (const std::invalid_argument& error) {
            return std::string{error.what()};
        }
        return std::string{};
    };
    EXPECT_EQ(SchedulerConfig{}.prefill_chunk_yield_max_rounds, std::numeric_limits<std::int32_t>::max());
    EXPECT_EQ(refusal(8, std::numeric_limits<std::int32_t>::max()), "");
    EXPECT_EQ(refusal(8, 1), "");
    for (const std::int32_t max_rounds : {0, -1}) {
        EXPECT_NE(refusal(8, max_rounds).find("prefill_chunk_yield_max_rounds must be positive"), std::string::npos)
            << max_rounds;
        EXPECT_EQ(refusal(0, max_rounds), "") << "inert without the yield";
    }
}

// A round counts toward prefill_chunk_yield_max_rounds only when the chunk
// really gives way. Here x leaves 18 of the 32-token budget, so cutting the
// long prompt's first chunk for s (16 tokens) would leave it less than a page:
// the chunk takes all 18 and s, no longer fitting, waits. That round does not
// count, so with a bound of one the long prompt still gives way to s in the
// next round; afterwards it gives way to no one.
TEST_F(PrefillChunkYieldSuite, ARoundCountsTowardTheBoundOnlyWhenTheChunkGivesWay) {
    max_rounds_ = 1;
    Restart(16);
    SubmitPrompt("x", MakeTokens(14, 3000));
    SubmitPrompt("long", MakeTokens(100, 1000));
    SubmitPrompt("s", MakeTokens(16, 2000));
    EXPECT_EQ(Round(), (PrefillRows{{"x", 0, 14, 14}, {"long", 0, 18, 100}}));
    // Given way: the chunk ends on the last page at least 16 tokens before
    // its budget's end (18 + 32 - 16 = 34, so 32).
    EXPECT_EQ(Round(), (PrefillRows{{"s", 0, 16, 16}, {"long", 18, 32, 100}}));
    SubmitPrompt("t", MakeTokens(8, 4000));
    EXPECT_EQ(Round(), (PrefillRows{{"long", 32, 64, 100}}));
}

// The per-holder bound: a long prompt gives way in at most
// prefill_chunk_yield_max_rounds rounds, then its chunks take the whole budget
// although short prompts keep arriving. Every round here starts with the whole
// budget (its chunk is admitted first), so it finishes its U tokens within the
// bound plus ceil(U / budget) of its rounds.
TEST_F(PrefillChunkYieldSuite, ALongPromptGivesWayInAtMostTheBoundsRoundsThenFinishes) {
    max_rounds_ = 2;
    Restart(16);
    SubmitPrompt("long", MakeTokens(100, 1000));
    std::vector<PrefillRow> long_rows;
    for (int round = 0; round < 12; ++round) {
        // Two short prompts arrive before every round: there is always
        // someone to give way to.
        SubmitPrompt("a" + std::to_string(round), MakeTokens(5, 2000 + 100 * round));
        SubmitPrompt("b" + std::to_string(round), MakeTokens(6, 5000 + 100 * round));
        for (const PrefillRow& row : Round()) {
            if (row.id == "long") {
                long_rows.push_back(row);
            }
        }
        if (!long_rows.empty() && long_rows.back().end == 100) {
            break;
        }
    }
    ASSERT_FALSE(long_rows.empty());
    EXPECT_EQ(long_rows.back().end, 100) << "the long prompt finished";
    const auto shared = std::ranges::count_if(
        long_rows, [](const PrefillRow& row) { return row.end - row.start < kBudget && row.end < row.prompt; });
    EXPECT_EQ(shared, 2) << ::testing::PrintToString(long_rows);
    EXPECT_LE(static_cast<std::int64_t>(long_rows.size()), 2 + (100 + kBudget - 1) / kBudget)
        << ::testing::PrintToString(long_rows);
}

// Whole workloads planned round by round until every request is done, with
// each round written as one line, compared with recorded sequences.
class PlanSequenceSuite : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kPage = 4;
    static constexpr std::int32_t kChunk = 32;

    struct Prompt {
        std::int32_t tokens;
        std::int32_t output;
    };
    // `concurrency` requests are live at a time; the next prompt is
    // submitted when one finishes.
    struct Workload {
        std::vector<Prompt> prompts;
        std::int32_t concurrency;
    };
    struct Settings {
        Role role;
        std::int32_t interval;  // prefill_decode_interval
        std::int32_t yield;     // prefill_chunk_yield_tokens
        // The two bounds; empty leaves the config's default.
        std::optional<std::int32_t> cap;
        std::optional<std::int32_t> yield_max_rounds;
        // prefill_interval_min_decoding; empty leaves the config's default.
        std::optional<std::int32_t> min_decoding;
    };

    // Long prompts arriving together, four chunks each, no shared prefix.
    static Workload ManyLongPrompts() { return Workload{std::vector<Prompt>(6, Prompt{4 * kChunk, 4}), 6}; }

    // Short prompts queued behind long ones, a few live at a time.
    static Workload MixedQueue() {
        Workload workload{.prompts = {}, .concurrency = 3};
        for (const std::int32_t tokens : {136, 6, 10, 72, 5, 100, 12, 7, 9, 64}) {
            workload.prompts.push_back(Prompt{tokens, 6});
        }
        return workload;
    }

    // Runs `workload` on a new scheduler and returns one line per round: P
    // (prefill), D (decode), M (both) or - (no forward), then each row as
    // id@first_position+tokens (prefill) or id+tokens (decode), then the
    // prompts handed to the decode side as >id. Every forward lands before
    // the next plan: a chunk that ends its prompt and every decode step
    // produce one token. A fused request finishes after its output tokens, a
    // P-role one when its hand-off is acknowledged.
    std::vector<std::string> Run(const Workload& workload, const Settings& settings) {
        const bool fused = settings.role == Role::kFused;
        std::int32_t longest = 0;
        for (const Prompt& prompt : workload.prompts) {
            longest = std::max(longest, prompt.tokens + prompt.output);
        }
        SchedulerConfig cfg{};
        cfg.role = settings.role;
        cfg.prefix_granularity = kPage;
        cfg.max_scheduled_tokens = kChunk;
        cfg.max_batch_size = workload.concurrency;
        cfg.max_resident_requests = workload.concurrency;
        cfg.disable_l2_cache = true;
        // Pages for every live request at once: nothing is retracted.
        cfg.device_allocator.total_pages = workload.concurrency * (longest / kPage + 2) + 1;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .block_granularity = kPage,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kFull,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        cfg.prefill_decode_interval = settings.interval;
        cfg.prefill_chunk_yield_tokens = settings.yield;
        if (settings.cap) {
            cfg.max_consecutive_prefill_rounds = *settings.cap;
        }
        if (settings.yield_max_rounds) {
            cfg.prefill_chunk_yield_max_rounds = *settings.yield_max_rounds;
        }
        if (settings.min_decoding) {
            cfg.prefill_interval_min_decoding = *settings.min_decoding;
        }
        SetTestSnapshotPool(cfg);
        config_ = WithHostPackingFromDevice(cfg);
        scheduler_ = std::make_unique<Scheduler>(config_);

        const auto send = [this](auto event_body) {
            ExecutionEvent event;
            event.With(std::move(event_body));
            scheduler_->Advance(std::move(event));
        };
        std::vector<std::string> rounds;
        std::map<std::string, std::int32_t> output_left;  // per live request
        std::size_t next = 0;
        std::size_t done = 0;
        while (done < workload.prompts.size() && rounds.size() < 500) {
            for (; next < workload.prompts.size() && std::ssize(output_left) < workload.concurrency; ++next) {
                const std::string id = "r" + std::to_string(next);
                const auto first_token = static_cast<std::int32_t>(100000 * (next + 1));
                Submit(RequestSpec{.request_id = id, .tokens = MakeTokens(workload.prompts[next].tokens, first_token)});
                if (!fused) {
                    send(pd::BootstrappedEvent{id});
                }
                output_left[id] = workload.prompts[next].output;
            }
            const ExecutionPlan plan = PlanOnce();
            std::string round = "-";
            if (plan.remote_decode) {
                for (const std::string& id : plan.remote_decode->request_ids) {
                    send(pd::SucceededEvent{id});
                    output_left.erase(id);
                    ++done;
                    round += " >" + id;
                }
            }
            const ForwardBatch* batch = FindForwardBatch(plan);
            const std::size_t rows = batch == nullptr ? 0 : batch->request_ids.size();
            if (rows > 0) {
                round[0] = batch->NumExtends() == rows ? 'P' : batch->NumExtends() == 0 ? 'D' : 'M';
            }
            for (std::size_t i = 0; i < rows; ++i) {
                const std::string& id = batch->request_ids[i];
                const bool prefill = i < batch->NumExtends();
                const std::int32_t end = prefill ? batch->extend_prefix_lens[i] + batch->input_lengths[i] : 0;
                round += " " + id + (prefill ? "@" + std::to_string(batch->extend_prefix_lens[i]) : "") + "+" +
                         std::to_string(batch->input_lengths[i]);
                const bool produces = !prefill || end == batch->prefill_lengths[i];
                SendForwardDone(id, produces ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
                if (fused && produces && --output_left.at(id) == 0) {
                    SendFinish(id);
                    output_left.erase(id);
                    ++done;
                }
            }
            rounds.push_back(std::move(round));
        }
        EXPECT_EQ(done, workload.prompts.size()) << "the workload did not finish: " << ::testing::PrintToString(rounds);
        return rounds;
    }

    static Settings Fused(std::int32_t interval) {
        return Settings{.role = Role::kFused,
                        .interval = interval,
                        .yield = 0,
                        .cap = {},
                        .yield_max_rounds = {},
                        .min_decoding = {}};
    }
    static Settings PrefillRoleWithYield() {
        return Settings{.role = Role::kP,
                        .interval = 0,
                        .yield = kChunk / 2,
                        .cap = {},
                        .yield_max_rounds = {},
                        .min_decoding = {}};
    }

    // The sequences below were recorded with the scheduler as it was before
    // max_consecutive_prefill_rounds and prefill_chunk_yield_max_rounds
    // existed, and are unchanged with both at their defaults. A deliberate
    // change to default scheduling must update them.
    static std::vector<std::string> ManyLongPromptsUnderTheInterval() {
        return {
            "P r0@0+32",
            "P r0@32+32",
            "P r0@64+32",
            "P r0@96+32",
            "D r0+1",
            "D r0+1",
            "P r1@0+32",
            "P r1@32+32",
            "P r1@64+32",
            "P r1@96+32",
            "P r2@0+32",
            "P r2@32+32",
            "P r2@64+32",
            "P r2@96+32",
            "P r3@0+32",
            "P r3@32+32",
            "P r3@64+32",
            "P r3@96+32",
            "P r4@0+32",
            "P r4@32+32",
            "P r4@64+32",
            "P r4@96+32",
            "P r5@0+32",
            "P r5@32+32",
            "P r5@64+32",
            "P r5@96+32",
            "D r0+1 r1+1 r2+1 r3+1 r4+1 r5+1",
            "D r1+1 r2+1 r3+1 r4+1 r5+1",
            "D r1+1 r2+1 r3+1 r4+1 r5+1",
        };
    }
    static std::vector<std::string> ManyLongPromptsWithoutAnInterval() {
        return {
            "P r0@0+32",
            "P r0@32+32",
            "P r0@64+32",
            "P r0@96+32",
            "P r1@0+32",
            "P r1@32+32",
            "P r1@64+32",
            "P r1@96+32",
            "P r2@0+32",
            "P r2@32+32",
            "P r2@64+32",
            "P r2@96+32",
            "P r3@0+32",
            "P r3@32+32",
            "P r3@64+32",
            "P r3@96+32",
            "P r4@0+32",
            "P r4@32+32",
            "P r4@64+32",
            "P r4@96+32",
            "P r5@0+32",
            "P r5@32+32",
            "P r5@64+32",
            "P r5@96+32",
            "D r0+1 r1+1 r2+1 r3+1 r4+1 r5+1",
            "D r0+1 r1+1 r2+1 r3+1 r4+1 r5+1",
            "D r0+1 r1+1 r2+1 r3+1 r4+1 r5+1",
        };
    }
    // Recorded with the scheduler as it was before
    // prefill_interval_min_decoding existed, interval 2 and a cap of 1.
    static std::vector<std::string> ManyLongPromptsUnderTheIntervalCappedAtOne() {
        return {
            "P r0@0+32", "P r0@32+32", "P r0@64+32", "P r0@96+32", "D r0+1", "D r0+1", "P r1@0+32",
            "D r0+1",    "P r1@32+32", "P r1@64+32", "P r1@96+32", "D r1+1", "D r1+1", "P r2@0+32",
            "D r1+1",    "P r2@32+32", "P r2@64+32", "P r2@96+32", "D r2+1", "D r2+1", "P r3@0+32",
            "D r2+1",    "P r3@32+32", "P r3@64+32", "P r3@96+32", "D r3+1", "D r3+1", "P r4@0+32",
            "D r3+1",    "P r4@32+32", "P r4@64+32", "P r4@96+32", "D r4+1", "D r4+1", "P r5@0+32",
            "D r4+1",    "P r5@32+32", "P r5@64+32", "P r5@96+32", "D r5+1", "D r5+1", "D r5+1",
        };
    }
    static std::vector<std::string> MixedQueueUnderTheIntervalCappedAtOne() {
        return {
            "P r0@0+32",
            "P r0@32+32",
            "P r0@64+32",
            "P r0@96+32",
            "P r0@128+8 r1@0+6 r2@0+10",
            "D r0+1 r1+1 r2+1",
            "D r0+1 r1+1 r2+1",
            "D r0+1 r1+1 r2+1",
            "D r0+1 r1+1 r2+1",
            "D r0+1 r1+1 r2+1",
            "P r3@0+32",
            "P r3@32+32",
            "P r3@64+8 r4@0+5 r5@0+19",
            "D r3+1 r4+1",
            "D r3+1 r4+1",
            "P r5@19+32",
            "D r3+1 r4+1",
            "D r3+1 r4+1",
            "P r5@51+32",
            "D r3+1 r4+1",
            "P r5@83+17 r6@0+12 r7@0+3",
            "D r5+1 r6+1",
            "D r5+1 r6+1",
            "P r7@3+4",
            "D r5+1 r6+1 r7+1",
            "D r5+1 r6+1 r7+1",
            "D r5+1 r6+1 r7+1",
            "P r8@0+9 r9@0+23",
            "D r7+1 r8+1",
            "D r7+1 r8+1",
            "P r9@23+32",
            "D r8+1",
            "D r8+1",
            "P r9@55+9",
            "D r8+1 r9+1",
            "D r9+1",
            "D r9+1",
            "D r9+1",
            "D r9+1",
        };
    }
    static std::vector<std::string> MixedQueueUnderTheInterval() {
        return {
            "P r0@0+32",
            "P r0@32+32",
            "P r0@64+32",
            "P r0@96+32",
            "P r0@128+8 r1@0+6 r2@0+10",
            "D r0+1 r1+1 r2+1",
            "D r0+1 r1+1 r2+1",
            "D r0+1 r1+1 r2+1",
            "D r0+1 r1+1 r2+1",
            "D r0+1 r1+1 r2+1",
            "P r3@0+32",
            "P r3@32+32",
            "P r3@64+8 r4@0+5 r5@0+19",
            "D r3+1 r4+1",
            "D r3+1 r4+1",
            "P r5@19+32",
            "P r5@51+32",
            "P r5@83+17",
            "D r3+1 r4+1 r5+1",
            "D r3+1 r4+1 r5+1",
            "D r3+1 r4+1 r5+1",
            "P r6@0+12 r7@0+7",
            "D r5+1 r6+1 r7+1",
            "D r5+1 r6+1 r7+1",
            "P r8@0+9",
            "D r6+1 r7+1 r8+1",
            "D r6+1 r7+1 r8+1",
            "D r6+1 r7+1 r8+1",
            "P r9@0+32",
            "D r8+1",
            "D r8+1",
            "P r9@32+32",
            "D r9+1",
            "D r9+1",
            "D r9+1",
            "D r9+1",
            "D r9+1",
        };
    }
    static std::vector<std::string> MixedQueueWithTheChunkYield() {
        return {
            "P r1@0+6 r2@0+10 r0@0+16",
            "P >r1 >r2 r0@16+32",
            "P r4@0+5 r0@48+24",
            "P >r4 r0@72+32",
            "P r0@104+32",
            "P >r0 r3@0+32",
            "P r6@0+12 r3@32+20",
            "P >r6 r3@52+20 r5@0+12",
            "P >r3 r7@0+7 r5@12+24",
            "P >r7 r8@0+9 r5@36+20",
            "P >r8 r5@56+32",
            "P r5@88+12 r9@0+20",
            "P >r5 r9@20+32",
            "P r9@52+12",
            "- >r9",
        };
    }
};

// Fused role: the defaults plan the workload as recorded, without a
// prefill_decode_interval and with one of 2. Under the interval a cap of 1 or
// 2 plans it differently, so the recorded rounds do cover the suspension the
// cap bounds.
TEST_F(PlanSequenceSuite, ManyLongPromptsPlanAsRecordedUnderTheDecodeInterval) {
    const std::vector<std::string> recorded = ManyLongPromptsUnderTheInterval();
    EXPECT_EQ(Run(ManyLongPrompts(), Fused(/*interval=*/2)), recorded);
    EXPECT_EQ(Run(ManyLongPrompts(), Fused(/*interval=*/0)), ManyLongPromptsWithoutAnInterval());
    for (const std::int32_t cap : {1, 2}) {
        Settings capped = Fused(/*interval=*/2);
        capped.cap = cap;
        EXPECT_NE(Run(ManyLongPrompts(), capped), recorded) << "a cap of " << cap << " changed nothing";
    }
}

TEST_F(PlanSequenceSuite, AMixedQueuePlansAsRecordedUnderTheDecodeInterval) {
    const std::vector<std::string> recorded = MixedQueueUnderTheInterval();
    EXPECT_EQ(Run(MixedQueue(), Fused(/*interval=*/2)), recorded);
    for (const std::int32_t cap : {1, 2}) {
        Settings capped = Fused(/*interval=*/2);
        capped.cap = cap;
        EXPECT_NE(Run(MixedQueue(), capped), recorded) << "a cap of " << cap << " changed nothing";
    }
}

// P role, chunk yield on: the default plans the queue as recorded. A long
// prompt gives way more than once in it, so a bound of one round plans it
// differently.
TEST_F(PlanSequenceSuite, AMixedQueuePlansAsRecordedWithTheChunkYield) {
    const std::vector<std::string> recorded = MixedQueueWithTheChunkYield();
    EXPECT_EQ(Run(MixedQueue(), PrefillRoleWithYield()), recorded);
    Settings one_round = PrefillRoleWithYield();
    one_round.yield_max_rounds = 1;
    EXPECT_NE(Run(MixedQueue(), one_round), recorded);
}

// The defaults written out (no cap, an unbounded yield) are the fields left
// out.
TEST_F(PlanSequenceSuite, TheExplicitDefaultsPlanAsTheFieldsLeftOut) {
    Settings no_cap = Fused(/*interval=*/2);
    no_cap.cap = 0;
    EXPECT_EQ(Run(MixedQueue(), no_cap), MixedQueueUnderTheInterval());
    Settings unbounded = PrefillRoleWithYield();
    unbounded.yield_max_rounds = std::numeric_limits<std::int32_t>::max();
    EXPECT_EQ(Run(MixedQueue(), unbounded), MixedQueueWithTheChunkYield());
}

// prefill_interval_min_decoding T 0 plans both workloads as recorded, with
// the cap or without. So does T 1 with the cap (T requires it): a round with
// nothing ready to decode owes no decode round under either rule, so T 1 is
// met whenever it matters. Under the cap of 1 the recorded rounds pass
// through fewer ready decodes than the workload's concurrency, so a threshold
// at the concurrency plans them differently.
TEST_F(PlanSequenceSuite, TheThresholdUnsetOrAlwaysMetPlansAsRecorded) {
    Settings unset = Fused(/*interval=*/2);
    unset.min_decoding = 0;
    EXPECT_EQ(Run(ManyLongPrompts(), unset), ManyLongPromptsUnderTheInterval());
    EXPECT_EQ(Run(MixedQueue(), unset), MixedQueueUnderTheInterval());
    struct Recorded {
        Workload workload;
        std::vector<std::string> capped;
    };
    for (const Recorded& recorded : {Recorded{ManyLongPrompts(), ManyLongPromptsUnderTheIntervalCappedAtOne()},
                                     Recorded{MixedQueue(), MixedQueueUnderTheIntervalCappedAtOne()}}) {
        SCOPED_TRACE(recorded.workload.prompts.size());
        Settings capped = Fused(/*interval=*/2);
        capped.cap = 1;
        EXPECT_EQ(Run(recorded.workload, capped), recorded.capped);
        for (const std::int32_t min_decoding : {0, 1}) {
            capped.min_decoding = min_decoding;
            EXPECT_EQ(Run(recorded.workload, capped), recorded.capped) << "threshold " << min_decoding;
        }
        capped.min_decoding = recorded.workload.concurrency;
        EXPECT_NE(Run(recorded.workload, capped), recorded.capped) << "a threshold at the concurrency changed nothing";
    }
}

}  // namespace tokenspeed::test
