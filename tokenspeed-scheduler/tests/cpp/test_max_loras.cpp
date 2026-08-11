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

#include <string>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

std::vector<std::string> BatchRequestIds(const ExecutionPlan& plan) {
    for (const auto& op : plan.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            return fwd->request_ids;
        }
    }
    return {};
}

class MaxLorasSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.max_loras = 2;
        return cfg;
    }
};

TEST_F(MaxLorasSuite, BatchNeverExceedsDistinctAdapterCap) {
    Submit(MakeRequestSpec("r1", 2, 100, 1));
    Submit(MakeRequestSpec("r2", 2, 200, 2));
    Submit(MakeRequestSpec("r3", 2, 300, 3));
    Submit(MakeRequestSpec("r4", 2, 400, 4));

    const std::vector<std::string> ids = BatchRequestIds(PlanOnce());
    ASSERT_FALSE(ids.empty());
    // Candidates are visited in request-id order, so the two admitted adapters
    // are a and b; c and d are deferred to a later step.
    EXPECT_EQ(ids, (std::vector<std::string>{"r1", "r2"}));
}

TEST_F(MaxLorasSuite, RepeatedAdapterCostsNoAdditionalSlot) {
    // Four requests but only two distinct adapters: nothing should be deferred.
    Submit(MakeRequestSpec("r1", 2, 100, 1));
    Submit(MakeRequestSpec("r2", 2, 200, 1));
    Submit(MakeRequestSpec("r3", 2, 300, 2));
    Submit(MakeRequestSpec("r4", 2, 400, 2));

    const std::vector<std::string> ids = BatchRequestIds(PlanOnce());
    EXPECT_EQ(ids, (std::vector<std::string>{"r1", "r2", "r3", "r4"}));
}

TEST_F(MaxLorasSuite, BaseModelRequestsAreNotCharged) {
    // Base-model requests hold no adapter slot, so they coexist with a full
    // adapter budget rather than being deferred behind it.
    Submit(MakeRequestSpec("r1", 2, 100, 1));
    Submit(MakeRequestSpec("r2", 2, 200, 2));
    Submit(MakeRequestSpec("r3", 2, 300));
    Submit(MakeRequestSpec("r4", 2, 400));

    const std::vector<std::string> ids = BatchRequestIds(PlanOnce());
    EXPECT_EQ(ids, (std::vector<std::string>{"r1", "r2", "r3", "r4"}));
}

class LorasDisabledSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.max_loras = 0;
        return cfg;
    }
};

TEST_F(LorasDisabledSuite, ZeroMaxLorasEnforcesNoCap) {
    // max_loras == 0 means LoRA scheduling is off, not "admit zero adapters":
    // the cap simply is not applied, matching the pre-LoRA scheduler exactly.
    Submit(MakeRequestSpec("r1", 2, 100, 1));
    Submit(MakeRequestSpec("r2", 2, 200, 2));
    Submit(MakeRequestSpec("r3", 2, 300, 3));

    const std::vector<std::string> ids = BatchRequestIds(PlanOnce());
    EXPECT_EQ(ids, (std::vector<std::string>{"r1", "r2", "r3"}));
}

}  // namespace
}  // namespace tokenspeed::test
