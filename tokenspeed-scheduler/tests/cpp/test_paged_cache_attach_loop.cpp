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

// Coverage: end-to-end borrowed-prefix re-import on a fully-cached prefill.

#include <gtest/gtest.h>

#include <cstdint>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

// page=2, LCM=4 raw tokens (2 KV pages per segment); 12-token prompt spans 3 segments.
class PagedCacheAttachLoopTest : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kLcmRawTokens = 4;

    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 16;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;

        PagedCacheGroupConfig fh{};
        fh.group_id = "fh";
        fh.rows_per_page = 4;
        fh.entry_stride_tokens = 1;
        fh.total_pages = 32;
        fh.retention = PagedCacheGroupConfig::Retention::FullHistory;
        cfg.paged_cache_groups.push_back(fh);

        PagedCacheGroupConfig swa{};
        swa.group_id = "swa";
        swa.rows_per_page = 2;
        swa.entry_stride_tokens = 1;
        swa.total_pages = 32;
        swa.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
        swa.sliding_window_tokens = 8;
        cfg.paged_cache_groups.push_back(swa);

        // Enable prefix-cache adjunct (LCM and sliding window derived from groups).
        PrefixCacheAdjunctSpec spec{};
        spec.required_groups = {"fh", "swa"};
        cfg.prefix_cache_adjunct = spec;

        return cfg;
    }

    static const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
    }
};

}  // namespace

// R1 primes 12 tokens; R2's same prefix must skip commit, re-import borrowed
// pages, and populate FlatForwardOperation.paged_cache_block_tables.
TEST_F(PagedCacheAttachLoopTest, FullyCachedPrefillBorrowedPrefixReimported) {
    // R1 primes the cache with 12 tokens.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6, /*start=*/1));
    PlanOnce();
    SendForwardDone("r1", {99});
    PlanOnce();
    SendFinish("r1");
    PlanOnce();

    // R2 uses the same prefix and should import all 3 LCM segments.
    // borrowed.
    Submit(MakeRequestSpec("r2", /*num_pages=*/6, /*start=*/1));
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);

    // (a) prefix-hit covers at least one LCM segment.
    EXPECT_GE(fwd->extend_prefix_lens[0], kLcmRawTokens);

    // (b) per-group tables already contain borrowed pages.
    auto fh_ids = scheduler_->GetRequestPagedCachePageIds("r2", "fh");
    EXPECT_GE(fh_ids.size(), 1u) << "borrowed fh prefix must be imported";

    // (c) paged_cache_block_tables populated for the executor.
    EXPECT_FALSE(fwd->paged_cache_block_tables.empty());
    auto fh_it = fwd->paged_cache_block_tables.find("fh");
    ASSERT_NE(fh_it, fwd->paged_cache_block_tables.end());
    EXPECT_FALSE(fh_it->second.empty());
    EXPECT_FALSE(fh_it->second[0].empty())
        << "fh block table row must not be empty for cached prefill";
}

}  // namespace tokenspeed::test
