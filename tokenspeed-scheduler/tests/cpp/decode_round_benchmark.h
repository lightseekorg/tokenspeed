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

#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "full_cache_benchmark.h"

namespace tokenspeed::benchmark {

struct DecodeRoundConfig {
    std::int32_t page_size;
    std::int32_t context_tokens;
    std::int32_t num_requests;
    // Tokens each request accepts per round: the decode (verify) width.
    std::int32_t decode_tokens;
    // Rounds measured or used as warmups once the tables have started to grow.
    std::int32_t num_rounds;

    // Admission reserves part of a request's generation budget up front, so
    // its table grows only once decode has consumed that reserve. The
    // workload decodes up to this many tokens per request to get there.
    static constexpr std::int32_t kMaxAdmissionReserveTokens = 8192;

    void Validate() const {
        Require(page_size > 0 && context_tokens > 0 && num_requests > 0 && decode_tokens > 0 && num_rounds > 0,
                "sizes must be positive");
        Require(context_tokens % page_size == 0, "context length must be page aligned");
        Require(std::int64_t{num_requests} * (std::int64_t{context_tokens} + page_size) +
                        std::int64_t{num_requests} * MaxNewTokens() <
                    std::numeric_limits<std::int32_t>::max(),
                "synthetic token IDs must fit int32");
    }

    // Request i's prompt is a few tokens longer than the context, so the
    // requests cross page boundaries at different tokens, as in a running
    // batch.
    std::int32_t PromptTokens(std::int32_t request) const { return context_tokens + request % page_size; }
    std::int64_t MaxNewTokens() const {
        return std::int64_t{kMaxAdmissionReserveTokens} + std::int64_t{num_rounds + 2} * decode_tokens;
    }
};

inline SchedulerConfig MakeDecodeSchedulerConfig(const DecodeRoundConfig& workload) {
    workload.Validate();
    // Every request's prompt, its whole generation budget and a reserve page.
    const std::int64_t pages_per_request =
        (workload.context_tokens + workload.page_size + workload.MaxNewTokens() + workload.decode_tokens) /
            workload.page_size +
        2;
    const std::int64_t total_pages = pages_per_request * workload.num_requests + 1;
    Require(total_pages < std::numeric_limits<std::int32_t>::max(), "page count must fit int32");
    SchedulerConfig config{};
    config.prefix_granularity = workload.page_size;
    config.device_allocator.total_pages = static_cast<std::int32_t>(total_pages);
    config.host_allocator.total_pages = 1;
    config.snapshot_allocator.total_pages = 1;
    config.max_retracted_requests = 0;
    config.max_scheduled_tokens = workload.num_requests * (workload.context_tokens + workload.page_size);
    config.max_batch_size = workload.num_requests;
    config.decode_input_tokens = workload.decode_tokens;
    config.overlap_schedule_depth = 0;
    config.disable_l2_cache = true;
    config.enable_l3_storage = false;
    config.enable_kv_cache_events = false;
    config.enable_short_prefill_first = false;
    config.role = Role::kFused;
    config.disable_prefix_cache = false;
    config.prefix_replay_tokens = 1;
    config.checkpoint_retention_requests = 0;
    config.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full_attention",
        .block_granularity = workload.page_size,
        .total_pages = config.device_allocator.total_pages,
        .attn_kind = AttnKind::kFull,
    });
    return config;
}

struct DecodeRoundSample {
    Duration plan;
    Duration advance;
};

// A batch of long-context requests decoding together: the steady state of a
// decode server, where each round plans one decode step for every request,
// every request accepts decode_tokens tokens, and the block tables keep
// growing. Construction prefills the prompts and decodes, unmeasured, until
// the requests have consumed what their admission reserved.
class DecodeRoundWorkload {
public:
    explicit DecodeRoundWorkload(DecodeRoundConfig config)
        : config_(config),
          scheduler_(MakeDecodeSchedulerConfig(config)),
          request_ids_{},
          generated_tokens_{0},
          rounds_run_{0},
          next_token_{0} {
        std::vector<RequestSpec> requests;
        for (std::int32_t i = 0; i < config_.num_requests; ++i) {
            // Disjoint token ranges: no request hits another's prefix.
            std::vector<std::int32_t> tokens(config_.PromptTokens(i));
            std::iota(tokens.begin(), tokens.end(), i * (config_.context_tokens + config_.page_size) + 1);
            request_ids_.push_back("decode-" + std::to_string(i));
            requests.push_back(RequestSpec{.request_id = request_ids_.back(),
                                           .tokens = std::move(tokens),
                                           .max_new_tokens = static_cast<std::int32_t>(config_.MaxNewTokens())});
        }
        next_token_ = config_.num_requests * (config_.context_tokens + config_.page_size) + 1;
        scheduler_.SubmitRequests(requests);
        const ExecutionPlan prefill =
            scheduler_.NextExecutionPlan({.allow_prefill = true, .apply_prefill_watermark = false});
        const ForwardBatch* batch = test::FindForwardBatch(prefill);
        Require(batch != nullptr && batch->NumExtends() == request_ids_.size(),
                "every prompt must complete its prefill in the first plan");
        scheduler_.Advance(Results(1));
        generated_tokens_ = 1;
        // Decode until the tables grow, then for one more page so every
        // request has started growing its table.
        const std::int32_t admitted_blocks = scheduler_.ActiveLcmBlocks();
        while (scheduler_.ActiveLcmBlocks() == admitted_blocks) {
            Require(generated_tokens_ < DecodeRoundConfig::kMaxAdmissionReserveTokens,
                    "block tables must grow once the admission reserve is consumed");
            decode();
        }
        for (std::int32_t tokens = 0; tokens < config_.page_size; tokens += config_.decode_tokens) decode();
    }

    // One decode round: the plan, then the results of its forward.
    DecodeRoundSample RunRound() {
        Require(rounds_run_ < config_.num_rounds, "the requests have no decode rounds left");
        DecodeRoundSample sample = decode();
        ++rounds_run_;
        return sample;
    }

    // Tokens each request holds now, prompt and generated.
    std::int32_t RequestTokens(std::int32_t request) const { return config_.PromptTokens(request) + generated_tokens_; }
    std::int32_t ActiveLcmBlocks() const { return scheduler_.ActiveLcmBlocks(); }

private:
    DecodeRoundSample decode() {
        DecodeRoundSample sample;
        ExecutionPlan plan;
        sample.plan = Measure(
            [&] { plan = scheduler_.NextExecutionPlan({.allow_prefill = true, .apply_prefill_watermark = false}); });
        const ForwardBatch* batch = test::FindForwardBatch(plan);
        Require(batch != nullptr && batch->NumExtends() == 0 && batch->request_ids.size() == request_ids_.size(),
                "every request must decode in every round");
        ExecutionEvent results = Results(config_.decode_tokens);
        sample.advance = Measure([&] { scheduler_.Advance(std::move(results)); });
        generated_tokens_ += config_.decode_tokens;
        return sample;
    }

    ExecutionEvent Results(std::int32_t accepted) {
        ExecutionEvent event;
        for (const auto& id : request_ids_) {
            std::vector<std::int32_t> tokens(accepted);
            std::iota(tokens.begin(), tokens.end(), next_token_);
            next_token_ += accepted;
            event.With(forward::ExtendResult{.request_id = id, .tokens = std::move(tokens)});
        }
        return event;
    }

    DecodeRoundConfig config_;
    Scheduler scheduler_;
    std::vector<std::string> request_ids_;
    std::int32_t generated_tokens_;
    std::int32_t rounds_run_;
    std::int32_t next_token_;
};

}  // namespace tokenspeed::benchmark
