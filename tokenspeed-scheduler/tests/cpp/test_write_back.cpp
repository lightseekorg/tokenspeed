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

#include <chrono>
#include <set>
#include <thread>
#include "integration_test_helper.h"

namespace tokenspeed::test {

// ============================================================
//  WriteBack: Finish → Draining → WritingBack → Finished
// ============================================================

class WriteBackTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 32;
        cfg.host_allocator.total_pages = 32;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    void SendReserveNumTokens(const std::string& id, std::int32_t n) {
        ExecutionEvent event;
        event.With(ForwardEvent{forward::UpdateReserveNumTokens{
            .request_id = id,
            .reserve_num_tokens_in_next_schedule_event = n,
        }});
        scheduler_->Advance(std::move(event));
    }

    // Submitted → PrefillDone → Decoding.
    // decoding_peers: requests already in Decoding that need reserve set before the next PlanOnce.
    void BringToDecoding(const std::string& id, std::int32_t num_pages, token_t start = 1,
                         const std::vector<std::string>& decoding_peers = {}) {
        Submit(MakeRequestSpec(id, num_pages, start));
        PlanOnce();
        SendForwardDone(id, {42});
        // Peers already in Decoding need their reserve set before the next plan,
        // because the scheduler asserts the value is initialised before scheduling decode.
        for (const auto& peer : decoding_peers) {
            SendReserveNumTokens(peer, 0);
        }
        PlanOnce();
    }

    static const FlatWriteBackOperation* GetWriteBack(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* wb = std::get_if<FlatWriteBackOperation>(cop)) {
                    return wb;
                }
            }
        }
        return nullptr;
    }

    static bool RequestInFwd(const ExecutionPlan& plan, const std::string& id) {
        for (const auto& op : plan.Operations()) {
            if (auto* fwd = std::get_if<FlatForwardOperation>(&op)) {
                for (const auto& rid : fwd->request_ids) {
                    if (rid == id) return true;
                }
            }
        }
        return false;
    }
};

class WriteBackLRUTestSuite : public WriteBackTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = WriteBackTestSuite::MakeConfig();
        cfg.host_allocator.total_pages = 5;  // 4 usable host pages.
        return cfg;
    }

    cache_op_id FinishAndPlanWriteBack(const std::string& id) {
        SendFinish(id);
        auto plan = PlanOnce();
        const auto* wb = GetWriteBack(plan);
        EXPECT_NE(wb, nullptr);
        if (wb == nullptr || wb->op_ids.empty()) {
            return 0;
        }
        return wb->op_ids[0];
    }

    static const FlatForwardOperation* GetForward(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* fwd = std::get_if<FlatForwardOperation>(&op)) {
                return fwd;
            }
        }
        return nullptr;
    }

    static std::int32_t FindRequestIndex(const FlatForwardOperation& fwd, const std::string& id) {
        for (std::size_t i = 0; i < fwd.request_ids.size(); ++i) {
            if (fwd.request_ids[i] == id) {
                return static_cast<std::int32_t>(i);
            }
        }
        return -1;
    }
};

// Finish → Draining; next PlanOnce generates a WriteBack op.
TEST_F(WriteBackTestSuite, WriteBack_GeneratedAfterFinish) {
    BringToDecoding("r1", /*num_pages=*/2);
    SendFinish("r1");

    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    ASSERT_FALSE(wb->op_ids.empty());
    bool any_pages = false;
    for (const auto& pages : wb->src_pages) {
        if (!pages.empty()) {
            any_pages = true;
            break;
        }
    }
    EXPECT_TRUE(any_pages);
}

// WritingBack request must not appear in the forward batch.
TEST_F(WriteBackTestSuite, WriteBack_RequestNotInForwardWhileWritingBack) {
    BringToDecoding("r1", /*num_pages=*/2);
    SendFinish("r1");

    auto plan = PlanOnce();
    EXPECT_FALSE(RequestInFwd(plan, "r1"));
}

// WriteBackDone → Finished; request cleaned up on next plan.
TEST_F(WriteBackTestSuite, WriteBack_FinishedAfterWriteBackDone) {
    BringToDecoding("r1", /*num_pages=*/2);
    SendFinish("r1");

    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    ASSERT_FALSE(wb->op_ids.empty());

    SendWriteBackDone(wb->op_ids[0]);

    PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
}

// Two requests finish in the same round → two separate WriteBack ops.
TEST_F(WriteBackTestSuite, WriteBack_MultipleRequestsGetSeparateOps) {
    BringToDecoding("r1", /*num_pages=*/2);
    BringToDecoding("r2", /*num_pages=*/2, /*start=*/50, /*decoding_peers=*/{"r1"});

    SendFinish("r1");
    SendFinish("r2");

    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    EXPECT_EQ(wb->op_ids.size(), 2u);
    EXPECT_EQ(wb->src_pages.size(), 2u);
}

// r1=[1,2,3,4] and r2=[1,2,5,6] share page [1,2] in the radix tree.
// After splitChild, r2's DevicePath must cover the shared prefix node too.
TEST_F(WriteBackTestSuite, WriteBack_SharedDevicePath_BothGetSeparateOps) {
    // radix tree after both prefills:
    //   root → node([1,2]) → node([3,4])  (r1)
    //                      → node([5,6])  (r2)
    BringToDecoding("r1", /*num_pages=*/2, /*start=*/1);
    {
        RequestSpec spec;
        spec.request_id = "r2";
        spec.tokens = {1, 2, 5, 6};  // first page [1,2] shared with r1
        Submit(spec);
        PlanOnce();
        SendForwardDone("r2", {99});
        // r1 is already Decoding; set its reserve before next PlanOnce to satisfy assert.
        SendReserveNumTokens("r1", 0);
        PlanOnce();
    }

    SendFinish("r1");
    SendFinish("r2");

    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    ASSERT_EQ(wb->op_ids.size(), 2u);
    ASSERT_EQ(wb->src_pages.size(), 2u);

    for (std::size_t i = 0; i < wb->src_pages.size(); ++i) {
        EXPECT_FALSE(wb->src_pages[i].empty());
    }
    EXPECT_NE(wb->op_ids[0], wb->op_ids[1]);

    // radix tree after split:
    //   root → node([1,2]) → node([3,4])  (r1 suffix)
    //                       → node([5,6])  (r2 suffix)
    // r1 op: covers node([3,4])              → 1 pair
    // r2 op: covers node([5,6]) + node([1,2]) → 2 pairs
    // Total: 3 (device_page, host_page) pairs across both ops.
    std::size_t total_pairs = 0;
    for (const auto& pages : wb->src_pages) {
        total_pairs += pages.size();
    }
    EXPECT_EQ(total_pairs, 3u);

    // All page IDs should be valid (positive).
    for (std::size_t i = 0; i < wb->src_pages.size(); ++i) {
        for (std::size_t j = 0; j < wb->src_pages[i].size(); ++j) {
            EXPECT_GT(wb->src_pages[i][j], 0);
            EXPECT_GT(wb->dst_pages[i][j], 0);
        }
    }

    // The two ops should cover different device pages (no overlap).
    std::set<int32_t> src0(wb->src_pages[0].begin(), wb->src_pages[0].end());
    std::set<int32_t> src1(wb->src_pages[1].begin(), wb->src_pages[1].end());
    for (int32_t p : src0) {
        EXPECT_EQ(src1.count(p), 0u) << "device page " << p << " appears in both ops";
    }

    SendWriteBackDone(wb->op_ids[0]);
    SendWriteBackDone(wb->op_ids[1]);
    PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
}

TEST_F(WriteBackLRUTestSuite, WriteBackDone_RefreshesHostLRUUnderPressure) {
    BringToDecoding("recent", /*num_pages=*/2, /*start=*/1);
    const cache_op_id recent_op_id = FinishAndPlanWriteBack("recent");
    ASSERT_NE(recent_op_id, 0u);

    BringToDecoding("older", /*num_pages=*/2, /*start=*/100);
    const cache_op_id older_op_id = FinishAndPlanWriteBack("older");
    ASSERT_NE(older_op_id, 0u);
    SendWriteBackDone(older_op_id);
    PlanOnce();

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    SendWriteBackDone(recent_op_id);
    PlanOnce();

    BringToDecoding("pressure", /*num_pages=*/2, /*start=*/200);
    const cache_op_id pressure_op_id = FinishAndPlanWriteBack("pressure");
    ASSERT_NE(pressure_op_id, 0u);

    Submit(MakeRequestSpec("probe_recent", /*num_pages=*/2, /*start=*/1));
    auto probe_plan = PlanOnce();
    const auto* fwd = GetForward(probe_plan);
    ASSERT_NE(fwd, nullptr);
    const std::int32_t probe_index = FindRequestIndex(*fwd, "probe_recent");
    ASSERT_GE(probe_index, 0);
    ASSERT_LT(static_cast<std::size_t>(probe_index), fwd->input_lengths.size());

    EXPECT_EQ(fwd->input_lengths[probe_index], PageSize())
        << "the writeback completed most recently, so host pressure should evict the older host page first";
}

}  // namespace tokenspeed::test
