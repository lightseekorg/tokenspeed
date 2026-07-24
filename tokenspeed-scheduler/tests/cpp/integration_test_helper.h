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

#include <algorithm>

#include "scheduler/scheduler.h"
#include "scheduler/execution_plan.h"
#include "scheduler/execution_event.h"
#include "scheduler/operations/cache.h"
#include "scheduler/operations/inc.h"
#include "scheduler/types.h"

#include "unit_test_helper.h"

namespace tokenspeed::test {

class SchedulerTestSuite : public ::testing::Test {
protected:
    // Subclasses can override this to customize the config.
    virtual SchedulerConfig MakeConfig() {
        SchedulerConfig cfg{};
        cfg.block_size = 2;
        cfg.device_allocator.total_pages = 32;
        cfg.host_allocator.total_pages = 32;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = true;
        cfg.prefetch_threshold = 2;
        return cfg;
    }

    void SetUp() override {
        config_ = MakeConfig();
        scheduler_ = std::make_unique<Scheduler>(config_);
    }

    const SchedulerConfig& Config() const { return config_; }
    std::int32_t PageSize() const { return config_.block_size; }
    RequestSpec MakeRequestSpec(const std::string& id, std::int32_t num_pages, token_t start = 1) {
        auto tokens = MakeAlignedTokens(num_pages, PageSize(), start);
        return RequestSpec{
            .request_id = id,
            .tokens = tokens,
        };
    }

    RequestSpec MakePrefetchableSpec(const std::string& id, std::int32_t num_pages, std::int32_t storage_hit_pages,
                                     token_t start = 1) {
        auto tokens = MakeAlignedTokens(num_pages, PageSize(), start);
        return RequestSpec{
            .request_id = id,
            .tokens = tokens,
            .rolling_hashes = MakePageHashes(storage_hit_pages, "rh"),
            .storage_hit_pages = storage_hit_pages,
        };
    }

    void Submit(const RequestSpec& spec) { scheduler_->SubmitRequests({spec}); }

    void Submit(const std::vector<RequestSpec>& specs) { scheduler_->SubmitRequests(specs); }

    ExecutionPlan PlanOnce() { return scheduler_->NextExecutionPlan(); }

    static const FlatForwardOperation* GetForward(const std::vector<Operation>& operations) {
        for (const auto& op : operations) {
            if (auto* forward = std::get_if<FlatForwardOperation>(&op)) return forward;
        }
        return nullptr;
    }

    static const FlatForwardOperation* GetForward(const ExecutionPlan& plan) { return GetForward(plan.Operations()); }

    template <typename PlanOrOperations>
    static const FlatForwardOperation* GetForwardOp(const PlanOrOperations& value) {
        return GetForward(value);
    }

    template <typename Kind>
    static const Kind* GetCacheOperation(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cache_op = std::get_if<CacheOperation>(&op)) {
                if (auto* result = std::get_if<Kind>(cache_op)) return result;
            }
        }
        return nullptr;
    }

    static const FlatWriteBackOperation* GetWriteBack(const ExecutionPlan& plan) {
        return GetCacheOperation<FlatWriteBackOperation>(plan);
    }

    static const FlatLoadBackOperation* GetLoadBack(const ExecutionPlan& plan) {
        return GetCacheOperation<FlatLoadBackOperation>(plan);
    }

    static bool RequestInFwd(const ExecutionPlan& plan, const std::string& id) {
        for (const auto& op : plan.Operations()) {
            if (auto* forward = std::get_if<FlatForwardOperation>(&op);
                forward != nullptr &&
                std::find(forward->request_ids.begin(), forward->request_ids.end(), id) != forward->request_ids.end()) {
                return true;
            }
        }
        return false;
    }

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

    void SendWriteBackDone(cache_op_id op_id, bool success = true) {
        ExecutionEvent event;
        event.With(CacheEvent{cache::WriteBackDone{
            .op_id = op_id,
            .success = success,
        }});
        scheduler_->Advance(std::move(event));
    }

    void SendLoadBackDone(cache_op_id op_id, bool success = true) {
        ExecutionEvent event;
        event.With(CacheEvent{cache::LoadBackDone{
            .op_id = op_id,
            .success = success,
        }});
        scheduler_->Advance(std::move(event));
    }

    // Send ExtendResult (new decode tokens) to the scheduler.
    void SendForwardDone(const std::string& request_id, const std::vector<std::int32_t>& tokens) {
        ExecutionEvent event;
        event.With(ForwardEvent{forward::ExtendResult{
            .request_id = request_id,
            .tokens = tokens,
        }});
        scheduler_->Advance(std::move(event));
    }

    // Send Finish (generation complete) to the scheduler.
    // This triggers FinishEvent: Decoding → Draining (or Finished if no writeback needed).
    void SendFinish(const std::string& request_id) {
        ExecutionEvent event;
        event.With(ForwardEvent{forward::Finish{
            .request_id = request_id,
        }});
        scheduler_->Advance(std::move(event));
    }

    void SendAbortEvent(const std::string& request_id) {
        ExecutionEvent event;
        event.With(ForwardEvent{forward::Abort{.request_id = request_id}});
        scheduler_->Advance(std::move(event));
    }

    void SendReserveNumTokens(const std::string& request_id, std::int32_t value) {
        ExecutionEvent event;
        event.With(ForwardEvent{forward::UpdateReserveNumTokens{
            .request_id = request_id,
            .reserve_num_tokens_in_next_schedule_event = value,
        }});
        scheduler_->Advance(std::move(event));
    }

    SchedulerConfig config_{};
    std::unique_ptr<Scheduler> scheduler_;
};
}  // namespace tokenspeed::test
