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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

// Regression tests for the hybrid-path "OwnedPages::TakeFirst: count out of range" crash. The
// LocalKVAllocator is sized at schedule time, but InsertHybridCache/FinishEvent/scheduleRetract
// recompute pages-to-insert later, so MTP/overlap growth or a prefix shrink can exceed it. Synthetic.

#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {

class HybridTakeFirstOverflowTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_mamba = true;
        cfg.mamba_pool_total_chunks = 16;
        cfg.decode_input_tokens = 0;
        cfg.block_size = 2;
        cfg.device_allocator.total_pages = 64;  // ample, so the only possible failure is count > size
        cfg.host_allocator.total_pages = 64;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    void BringToDecoding(const std::string& id, std::int32_t num_pages = 1, token_t start = 1) {
        Submit(MakeRequestSpec(id, num_pages, start));
        PlanOnce();
        SendForwardDone(id, {42});
        PlanOnce();
    }
};

// Growing the tokens past the allocator via an unscheduled extend, then finishing, must clamp not throw.
TEST_F(HybridTakeFirstOverflowTest, FinishAfterUnscheduledExtend_ClampsInsteadOfOverflow) {
    BringToDecoding("r1", 1, 1);

    std::vector<std::int32_t> big;
    big.reserve(60);
    for (int i = 0; i < 60; ++i) big.push_back(1000 + i);
    SendForwardDone("r1", big);

    ASSERT_NO_THROW({
        SendFinish("r1");
        PlanOnce();
    });
}

class HybridTakeFirstOverflowMitigationTest : public HybridTakeFirstOverflowTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = HybridTakeFirstOverflowTest::MakeConfig();
        cfg.disable_prefix_cache = true;
        return cfg;
    }
};

// The guard clamps even with prefix caching off, so --no-enable-prefix-caching is not a mitigation.
TEST_F(HybridTakeFirstOverflowMitigationTest, DisablePrefixCache_AlsoClampsCleanly) {
    BringToDecoding("r1", 1, 1);

    std::vector<std::int32_t> big;
    big.reserve(60);
    for (int i = 0; i < 60; ++i) big.push_back(1000 + i);
    SendForwardDone("r1", big);

    ASSERT_NO_THROW({
        SendFinish("r1");
        PlanOnce();
    });
}

// After the clamped finish the engine is not wedged: r1 drains and a fresh request still schedules.
TEST_F(HybridTakeFirstOverflowTest, FinishAfterUnscheduledExtend_ClampsAndStaysUsable) {
    BringToDecoding("r1", 1, 1);

    std::vector<std::int32_t> big;
    big.reserve(60);
    for (int i = 0; i < 60; ++i) big.push_back(1000 + i);
    SendForwardDone("r1", big);

    ASSERT_NO_THROW({
        SendFinish("r1");
        PlanOnce();
    }) << "FinishEvent must clamp page-aligned, not throw and not relocate the "
          "throw into KVPrefixCache::Insert TakeLast";

    EXPECT_EQ(scheduler_->DecodingSize(), 0u) << "r1 should have left Decoding after finish";

    Submit(MakeRequestSpec("r2", 1, 5000));
    auto plan = PlanOnce();
    bool r2_scheduled = false;
    for (const auto& op : plan.Operations()) {
        if (auto* fwd = std::get_if<FlatForwardOperation>(&op)) {
            for (const auto& rid : fwd->request_ids) {
                if (rid == "r2") r2_scheduled = true;
            }
        }
    }
    EXPECT_TRUE(r2_scheduled) << "engine wedged: a fresh request after the clamped finish was not scheduled";
}

class HybridRetractNegativeAllocTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_mamba = true;
        cfg.mamba_pool_total_chunks = 16;
        cfg.decode_input_tokens = 0;
        cfg.block_size = 2;
        cfg.device_allocator.total_pages = 6;  // small, so the device exhausts and a retract is forced
        cfg.host_allocator.total_pages = 6;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    // Commit the device node to an exact page boundary so the prefix holds N pages while
    // GetFullPagedTokens(true) drops the last page, leaving N-1 token pages.
    void BringToDecodingExactPages(const std::string& id, std::int32_t num_pages, token_t start) {
        Submit(MakeRequestSpec(id, num_pages, start));
        PlanOnce();
        PlanOnce();
    }
};

// A device prefix deeper than the tokens drives alloc_count negative; scheduleRetract must skip, not TakeFirst(<0).
TEST_F(HybridRetractNegativeAllocTest, RetractWithDeepPrefix_NegativeAllocCount_SkipsCleanly) {
    BringToDecodingExactPages("r1", 2, 1);
    BringToDecodingExactPages("r2", 2, 100);

    SendReserveNumTokens("r1", 4);
    SendReserveNumTokens("r2", 4);

    ASSERT_NO_THROW({ PlanOnce(); }) << "scheduleRetract must skip negative alloc_count, not call TakeFirst(<0)";

    EXPECT_GE(scheduler_->RetractedSize(), 1u) << "no retract occurred — the negative-alloc path was not reached";
}

}  // namespace tokenspeed::test
