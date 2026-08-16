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

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "scheduler/scheduler.h"
#include "scheduler/execution_plan.h"
#include "scheduler/execution_event.h"
#include "cache/tier/transfer.h"
#include "scheduler/operations/inc.h"
#include "scheduler/types.h"

#include "unit_test_helper.h"

namespace tokenspeed::test {

class SchedulerTestSuite : public ::testing::Test {
protected:
    // Subclasses can override this to customize the config.
    virtual SchedulerConfig MakeConfig() {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 32;
        cfg.host_allocator.total_pages = 32;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .rows_per_page = cfg.prefix_granularity,
            .entry_stride_tokens = 1,
            .total_pages = cfg.device_allocator.total_pages,
            .retention = CacheGroupConfig::Retention::FullHistory,
            .family = CacheGroupFamily::History,
        });
        return cfg;
    }

    void SetUp() override {
        config_ = MakeConfig();
        scheduler_ = std::make_unique<Scheduler>(config_);
    }

    const SchedulerConfig& Config() const { return config_; }
    std::int32_t PrefixGranularity() const { return config_.prefix_granularity; }
    RequestSpec MakeRequestSpec(const std::string& id, std::int32_t num_pages, std::int32_t start = 1) {
        auto tokens = MakeAlignedTokens(num_pages, PrefixGranularity(), start);
        return RequestSpec{
            .request_id = id,
            .tokens = tokens,
        };
    }

    void Submit(const RequestSpec& spec) { scheduler_->SubmitRequests({spec}); }

    void Submit(const std::vector<RequestSpec>& specs) { scheduler_->SubmitRequests(specs); }

    ExecutionPlan PlanOnce() { return scheduler_->NextExecutionPlan(); }

    static std::vector<CacheOperation> ExtractCacheOps(const ExecutionPlan& plan) {
        std::vector<CacheOperation> result;
        for (const auto& op : plan.Operations()) {
            if (auto* cache_op = std::get_if<CacheOperation>(&op)) {
                result.push_back(*cache_op);
            }
        }
        return result;
    }

    template <typename Kind>
    static std::vector<CacheOperation> FilterByKind(const std::vector<CacheOperation>& ops) {
        std::vector<CacheOperation> result;
        for (const auto& op : ops) {
            if (std::holds_alternative<Kind>(op)) {
                result.push_back(op);
            }
        }
        return result;
    }

    template <typename Kind>
    static std::vector<CacheOperation> ExtractCacheOpsOfKind(const ExecutionPlan& plan) {
        return FilterByKind<Kind>(ExtractCacheOps(plan));
    }

    void SendWriteBackDone(std::uint32_t op_id) {
        ExecutionEvent event;
        event.With(cache::WriteBackDone{
            .op_id = op_id,
        });
        scheduler_->Advance(std::move(event));
    }

    void SendLoadBackDone(std::uint32_t op_id) {
        ExecutionEvent event;
        event.With(cache::LoadBackDone{
            .op_id = op_id,
        });
        scheduler_->Advance(std::move(event));
    }

    // Send ExtendResult (new decode tokens) to the scheduler.
    void SendForwardDone(const std::string& request_id, const std::vector<std::int32_t>& tokens) {
        ExecutionEvent event;
        event.With(forward::ExtendResult{
            .request_id = request_id,
            .tokens = tokens,
        });
        scheduler_->Advance(std::move(event));
    }

    // Send Finish (generation complete) to the scheduler.
    // This triggers FinishEvent: Decoding → Draining (or Finished if no writeback needed).
    void SendFinish(const std::string& request_id) {
        ExecutionEvent event;
        event.With(forward::Finish{
            .request_id = request_id,
        });
        scheduler_->Advance(std::move(event));
    }

    void SendAbortEvent(const std::string& request_id) {
        ExecutionEvent event;
        event.With(forward::Abort{.request_id = request_id});
        scheduler_->Advance(std::move(event));
    }

    SchedulerConfig config_{};
    std::unique_ptr<Scheduler> scheduler_;
};

// Returns the first ForwardBatch in `plan`, or nullptr if none.
inline const ForwardBatch* FindForwardBatch(const ExecutionPlan& plan) {
    for (const auto& op : plan.Operations()) {
        if (const auto* batch = std::get_if<ForwardBatch>(&op)) return batch;
    }
    return nullptr;
}

// Inserts every real (>0) page id found in `rows` into `seen`, expecting each
// id to be new — i.e. not repeated within `rows` and disjoint from any id
// already in `seen` (callers can chain groups through one set to assert
// cross-group disjointness, or use a fresh set per group).
//
// Arguments:
//   rows: block-table rows to scan (one vector<int32> per request row).
//   seen: accumulator of page ids; grows with the ids found here.
//   context: label prepended to failure messages (e.g. the group id).
// Returns the number of real (>0) entries visited, so callers can check
// `seen.size()` growth against it.
inline std::size_t CollectDisjointRealPages(const std::vector<std::vector<std::int32_t>>& rows,
                                            std::set<std::int32_t>& seen, const std::string& context) {
    std::size_t real_entries = 0;
    for (const auto& row : rows) {
        for (std::int32_t id : row) {
            if (id > 0) {
                ++real_entries;
                EXPECT_TRUE(seen.insert(id).second) << context << ": page " << id << " handed out more than once";
            }
        }
    }
    return real_entries;
}

// Kimi-K3-shaped KV-cache fixture: one full-attention (history) group
// plus three linear-attention (state) groups drawing from one global pool.
// Shared by test_kv_cache_lifecycle.cpp and test_cache_scenarios.cpp.
class KimiFourGroupSuite : public SchedulerTestSuite {
protected:
    static const std::array<std::string, 4>& GroupIds() {
        static const std::array<std::string, 4> ids{"full_attention", "linear_attention_0", "linear_attention_1",
                                                    "linear_attention_2"};
        return ids;
    }

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 33;
        cfg.host_allocator.total_pages = 0;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = false;

        for (std::size_t i = 0; i < GroupIds().size(); ++i) {
            CacheGroupConfig group;
            group.group_id = GroupIds()[i];
            group.rows_per_page = cfg.prefix_granularity;
            group.entry_stride_tokens = 1;
            group.total_pages = cfg.device_allocator.total_pages;
            group.retention = CacheGroupConfig::Retention::FullHistory;
            group.family = i == 0 ? CacheGroupFamily::History : CacheGroupFamily::State;
            cfg.cache_groups.push_back(std::move(group));
        }
        return cfg;
    }

    void AbortRequest(const std::string& id) { SendAbortEvent(id); }
};

}  // namespace tokenspeed::test
