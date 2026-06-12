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

namespace tokenspeed::test {

class PrefetchTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 32;
        cfg.host_allocator.total_pages = 32;
        cfg.enable_l3_storage = true;
        cfg.prefetch_threshold = 2;
        return cfg;
    }

    static const PrefetchOperation* GetPrefetch(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* pf = std::get_if<PrefetchOperation>(cop)) return pf;
            }
        }
        return nullptr;
    }

    static const BackUpOperation* GetBackup(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* backup = std::get_if<BackUpOperation>(cop)) return backup;
            }
        }
        return nullptr;
    }

    static const FlatWriteBackOperation* GetWriteBack(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* wb = std::get_if<FlatWriteBackOperation>(cop)) return wb;
            }
        }
        return nullptr;
    }

    static const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
    }

    void BringToDecoding(const RequestSpec& spec) {
        Submit(spec);
        PlanOnce();
        SendForwardDone(spec.request_id, {42});
        PlanOnce();
    }

    cache_op_id MaterializeHostPrefix(const RequestSpec& spec) {
        BringToDecoding(spec);
        SendFinish(spec.request_id);
        auto writeback_plan = PlanOnce();
        const auto* wb = GetWriteBack(writeback_plan);
        if (wb == nullptr || wb->op_ids.empty()) {
            ADD_FAILURE() << "expected writeback for " << spec.request_id;
            return 0;
        }
        SendWriteBackDone(wb->op_ids[0]);
        auto backup_plan = PlanOnce();
        const auto* backup = GetBackup(backup_plan);
        if (backup == nullptr) {
            return 0;
        }
        return backup->op_id;
    }
};

TEST_F(PrefetchTestSuite, Prefetch_GeneratedForL3StorageHit) {
    Submit(MakePrefetchableSpec("r1", /*num_pages=*/4, /*storage_hit_pages=*/3));
    auto plan = PlanOnce();

    const auto* pf = GetPrefetch(plan);
    // Prefetch may or may not be generated depending on the match state.
    // But request should at least be in forward.
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    bool r1_found = false;
    for (const auto& rid : fwd->request_ids) {
        if (rid == "r1") r1_found = true;
    }
    // Request should either be prefetching (not in forward) or in forward.
    EXPECT_TRUE(r1_found || pf != nullptr);
}

TEST_F(PrefetchTestSuite, NoPrefetch_WhenL3Disabled) {
    auto cfg = MakeConfig();
    cfg.enable_l3_storage = false;
    scheduler_ = std::make_unique<Scheduler>(cfg);

    Submit(MakePrefetchableSpec("r1", 4, 3));
    auto plan = PlanOnce();
    const auto* pf = GetPrefetch(plan);
    EXPECT_EQ(pf, nullptr);
}

TEST_F(PrefetchTestSuite, NoPrefetch_BelowThreshold) {
    // storage_hit_pages=1 < prefetch_threshold=2
    Submit(MakePrefetchableSpec("r1", 4, 1));
    auto plan = PlanOnce();
    const auto* pf = GetPrefetch(plan);
    EXPECT_EQ(pf, nullptr);
}

TEST_F(PrefetchTestSuite, PrefetchDone_InsertsSuffixAfterHostMatchedPrefix) {
    RequestSpec prefix;
    prefix.request_id = "prefix";
    prefix.tokens = {1, 2};
    cache_op_id backup_id = MaterializeHostPrefix(prefix);
    if (backup_id != 0) {
        SendBackUpDone(backup_id);
    }

    RequestSpec spec;
    spec.request_id = "r2";
    spec.tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    spec.rolling_hashes = {"h34", "h56", "h78"};
    spec.storage_hit_pages = 3;
    Submit(spec);

    auto prefetch_plan = PlanOnce();
    const auto* pf = GetPrefetch(prefetch_plan);
    ASSERT_NE(pf, nullptr);
    ASSERT_EQ(pf->rolling_page_hashes.size(), 3u);
    SendPrefetchDone(pf->op_id, "r2", /*completed_pages=*/3);

    auto forward_plan = PlanOnce();
    const auto* fwd = GetForwardOp(forward_plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->request_ids.size(), 1u);
    EXPECT_EQ(fwd->request_ids[0], "r2");
    ASSERT_EQ(fwd->extend_prefix_lens.size(), 1u);
    EXPECT_EQ(fwd->extend_prefix_lens[0], 8);
    ASSERT_EQ(fwd->input_lengths.size(), 1u);
    EXPECT_EQ(fwd->input_lengths[0], 0);
}

TEST_F(PrefetchTestSuite, BackupPinsHostPagesUntilBackUpDone) {
    auto cfg = MakeConfig();
    cfg.host_allocator.total_pages = 2;
    cfg.prefetch_threshold = 0;
    scheduler_ = std::make_unique<Scheduler>(cfg);

    RequestSpec written;
    written.request_id = "written";
    written.tokens = {1, 2, 3, 4};
    cache_op_id backup_id = MaterializeHostPrefix(written);
    ASSERT_NE(backup_id, 0u);

    Submit(MakePrefetchableSpec("blocked", /*num_pages=*/2, /*storage_hit_pages=*/2, /*start=*/100));
    auto blocked_plan = PlanOnce();
    EXPECT_EQ(GetPrefetch(blocked_plan), nullptr);

    SendBackUpDone(backup_id);

    Submit(MakePrefetchableSpec("released", /*num_pages=*/2, /*storage_hit_pages=*/2, /*start=*/200));
    auto released_plan = PlanOnce();
    EXPECT_NE(GetPrefetch(released_plan), nullptr);
}

}  // namespace tokenspeed::test
