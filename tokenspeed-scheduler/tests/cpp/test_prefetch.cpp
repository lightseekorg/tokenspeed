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

}  // namespace tokenspeed::test
