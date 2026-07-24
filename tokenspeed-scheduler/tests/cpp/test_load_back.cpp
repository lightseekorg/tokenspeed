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

inline const FlatForwardOperation* GetForwardOp(const std::vector<Operation>& ops) {
    for (const auto& op : ops) {
        if (auto* f = std::get_if<FlatForwardOperation>(&op)) {
            return f;
        }
    }
    return nullptr;
}

inline std::int32_t FindRequestIndex(const FlatForwardOperation* fwd, const std::string& rid) {
    if (fwd == nullptr) return -1;
    for (std::size_t i = 0; i < fwd->request_ids.size(); ++i) {
        if (fwd->request_ids[i] == rid) return static_cast<std::int32_t>(i);
    }
    return -1;
}

// SetupHostCache creates a host-only cache entry for tokens [1,2,3,4] (2 pages).
//
// Config: block_size=2, decode_input_tokens=0, device_total=5 (4 usable), host_total=32.
//
// Steps:
//   r1 (2 pages, tokens [1,2,3,4]):
//     prefill → ForwardDone([42]) → Decoding → Finish → Draining → WriteBack → Finished → erase.
//     After erase: node([1,2],[3,4]) in radix tree with device+host pages, both evictable.
//
//   r_fill (3 pages, tokens [100..105]):
//     Needs 3 device pages. Available=2, evict r1's 2 device pages → available=4. OK.
//     r_fill prefill uses 3 pages. After r_fill lifecycle, radix tree has:
//       node([1,2],[3,4]): host-only (device evicted by r_fill).
//       node([100..105]): device+host evictable.
//
// After SetupHostCache: Match([1,2,3,4]) → device.matched=0, host.matched=2.
// A new request with tokens [1,2,3,4] triggers LoadBack.
class LoadBackTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 5;
        cfg.host_allocator.total_pages = 32;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    void SetupHostCache() {
        Submit(MakeRequestSpec("r1", /*num_pages=*/2, /*start=*/1));
        PlanOnce();
        SendForwardDone("r1", {42});
        PlanOnce();
        SendFinish("r1");
        auto plan1 = PlanOnce();
        const FlatWriteBackOperation* wb = nullptr;
        for (const auto& op : plan1.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* w = std::get_if<FlatWriteBackOperation>(cop)) {
                    wb = w;
                    break;
                }
            }
        }
        ASSERT_NE(wb, nullptr) << "SetupHostCache: expected WriteBack op for r1";
        ASSERT_FALSE(wb->op_ids.empty());
        SendWriteBackDone(wb->op_ids[0]);
        PlanOnce();

        Submit(MakeRequestSpec("r_fill", /*num_pages=*/3, /*start=*/100));
        PlanOnce();
        SendForwardDone("r_fill", {200});
        PlanOnce();
        SendFinish("r_fill");
        auto plan2 = PlanOnce();
        for (const auto& op : plan2.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* w = std::get_if<FlatWriteBackOperation>(cop)) {
                    if (!w->op_ids.empty()) SendWriteBackDone(w->op_ids[0]);
                    break;
                }
            }
        }
        PlanOnce();
    }
};

TEST_F(LoadBackTestSuite, LoadBack_GeneratedOnHostCacheHit) {
    SetupHostCache();

    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/1));
    auto plan = PlanOnce();

    const auto* lb = GetLoadBack(plan);
    ASSERT_NE(lb, nullptr) << "expected a LoadBack op when host cache hit";
    bool has_pages = false;
    for (const auto& pages : lb->src_pages) {
        if (!pages.empty()) {
            has_pages = true;
            break;
        }
    }
    EXPECT_TRUE(has_pages);
}

TEST_F(LoadBackTestSuite, LoadBack_PagesAreNonEmpty) {
    SetupHostCache();
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/1));
    auto plan = PlanOnce();

    const auto* lb = GetLoadBack(plan);
    ASSERT_NE(lb, nullptr);
    for (std::size_t i = 0; i < lb->src_pages.size(); ++i) {
        for (std::size_t j = 0; j < lb->src_pages[i].size(); ++j) {
            EXPECT_GE(lb->src_pages[i][j], 0);
            EXPECT_GE(lb->dst_pages[i][j], 0);
        }
    }
}

TEST_F(LoadBackTestSuite, LoadBack_PrefillInputLengthReduced) {
    SetupHostCache();
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/1));
    auto plan = PlanOnce();

    const auto* fwd = GetForwardOp(plan.Operations());
    ASSERT_NE(fwd, nullptr);
    std::int32_t idx = FindRequestIndex(fwd, "r2");
    ASSERT_GE(idx, 0);
    // With block_size=2 and 4 prefill tokens, GetFullPagedTokens(except_last=true)
    // yields 3 tokens → 1 matchable page [[1,2]].  The host cache has 2 pages
    // (inserted at FinishEvent time with 5 tokens → 2 full pages), but only 1 matches.
    // unscheduled = PrefillSize(4) - max(0,1)*2 = 2, so input_length = 2.
    EXPECT_EQ(fwd->input_lengths[idx], 2) << "host hit covers 1 of 2 pages; 2 tokens remain";
}

TEST_F(LoadBackTestSuite, LoadBack_DuplicatePagesDeduped) {
    SetupHostCache();
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/1));
    Submit(MakeRequestSpec("r3", /*num_pages=*/2, /*start=*/1));
    auto plan = PlanOnce();

    const auto* lb = GetLoadBack(plan);
    ASSERT_NE(lb, nullptr);
    std::size_t total_pairs = 0;
    for (const auto& pages : lb->src_pages) total_pairs += pages.size();
    // GetFullPagedTokens(true) on 4-token requests yields 1 matchable page,
    // so only 1 host page is loaded back per request. Shared pages are deduped.
    EXPECT_EQ(total_pairs, 1u) << "shared host page should appear once in the flat LoadBack";
}

}  // namespace tokenspeed::test
