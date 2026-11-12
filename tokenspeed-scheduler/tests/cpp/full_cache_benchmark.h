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
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

#include "cache/prefix/prefix_hasher.h"
#include "cache_test_access.h"
#include "integration_test_helper.h"

namespace tokenspeed::benchmark {

inline void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct FullCacheConfig {
    std::int32_t pool_pages;
    std::int32_t page_size;
    std::int32_t request_tokens;
    std::int32_t batch_size;
    std::int32_t hit_percent;

    void Validate() const {
        Require(pool_pages > 0 && page_size > 0 && request_tokens > 0 && batch_size > 0, "sizes must be positive");
        Require(request_tokens % page_size == 0, "request length must be page aligned");
        Require(hit_percent >= 0 && hit_percent < 100, "hit percent must be in [0, 100)");
        const auto prompt_pages = request_tokens / page_size;
        Require(std::int64_t{pool_pages} >= std::int64_t{batch_size} * (2 * std::int64_t{prompt_pages} + 1),
                "pool must hold the query prefixes and a disjoint batch of eviction victims");
        Require((std::int64_t{pool_pages} + std::int64_t{batch_size} * prompt_pages) * page_size <
                    std::numeric_limits<std::int32_t>::max(),
                "synthetic token IDs must fit int32");
    }

    std::int32_t HitTokens() const {
        return static_cast<std::int32_t>(std::int64_t{request_tokens / page_size} * hit_percent / 100) * page_size;
    }
    std::uint64_t EvictedPages() const {
        // Each completing prompt also reserves its first decode token.
        return std::uint64_t{static_cast<std::uint32_t>(batch_size)} * ((request_tokens - HitTokens()) / page_size + 1);
    }
};

inline SchedulerConfig MakeSchedulerConfig(const FullCacheConfig& workload) {
    workload.Validate();
    SchedulerConfig config{};
    config.prefix_granularity = workload.page_size;
    config.device_allocator.total_pages = workload.pool_pages + 1;
    config.host_allocator.total_pages = 1;
    config.snapshot_allocator.total_pages = 1;
    config.max_retracted_requests = 0;
    config.max_scheduled_tokens = workload.batch_size * workload.request_tokens;
    config.max_batch_size = workload.batch_size;
    config.decode_input_tokens = 1;
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

struct Duration {
    double wall_ms{0};
    double cpu_ms{0};
    Duration& operator+=(const Duration& other) {
        wall_ms += other.wall_ms;
        cpu_ms += other.cpu_ms;
        return *this;
    }
};

inline std::int64_t ThreadCpuNs() {
    timespec now{};
    Require(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now) == 0, "thread CPU clock unavailable");
    return std::int64_t{now.tv_sec} * 1000000000 + now.tv_nsec;
}

template <typename F>
Duration Measure(F&& operation) {
    const auto cpu_start = ThreadCpuNs();
    const auto start = std::chrono::steady_clock::now();
    operation();
    const auto end = std::chrono::steady_clock::now();
    const auto cpu_end = ThreadCpuNs();
    return {std::chrono::duration<double, std::milli>(end - start).count(), (cpu_end - cpu_start) / 1e6};
}

struct CacheSample {
    Duration hash;
    Duration prepare_keys;
    Duration probe;
    Duration admit;
    std::uint64_t hit_tokens{0};
    std::uint64_t evicted_pages{0};
};

struct SchedulerSample {
    Duration submit;
    Duration plan;
    std::uint64_t hit_tokens{0};
    std::uint64_t evicted_pages{0};
    std::size_t admitted{0};
};

struct LifecycleSample {
    Duration submit;
    Duration first_plan;
    Duration advance;
    Duration publish_plan;
    Duration finish;
    Duration cleanup_plan;
    Duration events;
    std::size_t published_prompt_pages{0};
    std::size_t stored_event_blocks{0};
};

enum class CacheLayout { kSequential, kShuffled };

// Only metadata is allocated. Setup publishes real SHA-256 chains through
// CacheCompletedBlocks, so both the prefix index and boundary tree are full.
class FullCacheWorkload {
public:
    explicit FullCacheWorkload(FullCacheConfig config, bool enable_events, CacheLayout layout)
        : config_(config),
          enable_events_(enable_events),
          scheduler_([&] {
              auto scheduler_config = MakeSchedulerConfig(config);
              scheduler_config.enable_kv_cache_events = enable_events;
              return scheduler_config;
          }()),
          histories_{},
          requests_{} {
        if (layout == CacheLayout::kShuffled) {
            // Exercise noncontiguous physical placement without changing
            // registration order, tokens, epochs, or the amount of free space.
            auto& pool = const_cast<BlockPool&>(Pool());
            auto blocks = pool.AcquireBlocks(0, config_.pool_pages);
            std::mt19937 random(8731);
            std::shuffle(blocks.begin(), blocks.end(), random);
            for (auto& block : blocks) block.reset();
        }
        const std::int32_t pages_per_request = config_.request_tokens / config_.page_size;
        for (std::int32_t remaining = config_.pool_pages; remaining > 0;) {
            const auto pages = std::min(remaining, pages_per_request);
            const auto tokens = HistoryTokens(static_cast<std::int32_t>(histories_.size()), pages * config_.page_size);
            const TokenContainer container(tokens);
            histories_.push_back(ComputePrefixHashes(container.FullPrefixPages(config_.page_size, false), ""));
            RestoreHistory(histories_.back());
            remaining -= pages;
        }
        // Hit distinct recently published histories; older disjoint histories
        // supply victims. No request or pin remains from the fill.
        const auto full_histories = config_.pool_pages / pages_per_request;
        for (std::int32_t i = 0; i < config_.batch_size; ++i) {
            auto tokens = HistoryTokens(full_histories - config_.batch_size + i, config_.request_tokens);
            std::iota(tokens.begin() + config_.HitTokens(), tokens.end(),
                      config_.pool_pages * config_.page_size + i * config_.request_tokens + 1);
            requests_.push_back(RequestSpec{
                .request_id = "benchmark-" + std::to_string(i), .tokens = std::move(tokens), .max_new_tokens = 1});
        }
        CheckFull();
    }

    CacheCoordinator& Coordinator() { return SchedulerTestAccess::Coordinator(scheduler_); }
    const BlockPool& Pool() const { return SchedulerTestAccess::Pool(scheduler_); }
    const auto& Histories() const { return histories_; }

    void CheckFull() {
        Require(scheduler_.EmptyLcmBlocks() == 0, "benchmark must start with zero free pages");
        Require(Coordinator().GroupPrefixIndex(0).NumEntries(Pool()) == config_.pool_pages,
                "every physical page must be cached");
        Require(scheduler_.ActiveLcmBlocks() == 0 && scheduler_.WaitingSize() == 0,
                "setup must leave no active or waiting request");
        Require(CacheCoordinatorTestAccess::BoundaryTreeOf(Coordinator()).Size() ==
                    static_cast<std::size_t>(config_.pool_pages),
                "every prefix page must have a boundary-tree node");
    }

    CacheSample RunCacheSample() {
        CheckFull();
        CacheSample sample;
        const auto evictions_before = Coordinator().DeviceEvictions().admission.blocks;
        std::vector<BlockTable> tables(config_.batch_size);
        for (std::int32_t i = 0; i < config_.batch_size; ++i) {
            const TokenContainer tokens(requests_[i].tokens);
            std::vector<std::string> hashes;
            // Request::PromptPrefixHashes hashes all full prompt pages, even
            // though the probe excludes the final token's page.
            sample.hash +=
                Measure([&] { hashes = ComputePrefixHashes(tokens.FullPrefixPages(config_.page_size, false), ""); });
            const auto candidates = std::span<const std::string>{hashes}.first(hashes.size() - 1);
            PreparedPrefixKeys keys;
            sample.prepare_keys += Measure([&] { keys = Coordinator().PreparePrefixKeys(candidates); });
            CacheCoordinator::PrefixProbe probe;
            sample.probe += Measure([&] {
                probe = Coordinator().ProbePromptPrefix({.prefix_hashes = candidates,
                                                         .tokens = requests_[i].tokens,
                                                         .max_tokens = config_.request_tokens - 1,
                                                         .prepared_keys = keys});
            });
            Require(probe.device.num_common_tokens == config_.HitTokens(), "unexpected prefix hit");
            const std::array demands{GroupDemand{.table = &tables[i],
                                                 .extent = DenseGrowth{config_.request_tokens - config_.HitTokens()},
                                                 .reserve_tokens = 1}};
            std::optional<CacheCoordinator::AdmissionResult> result;
            sample.admit += Measure([&] {
                result = Coordinator().Admit(std::move(probe), demands, RequestProgress{}, {}, std::nullopt,
                                             std::nullopt, /*evict_blocks_past_segments=*/false, /*session_id=*/{});
            });
            Require(result.has_value(), "cache admission failed");
            Require(result->new_page_ids[0].size() == config_.EvictedPages() / config_.batch_size,
                    "unexpected fresh-page count");
            Require(tables[i].NumBlocks() == config_.request_tokens / config_.page_size + 1,
                    "admission must cover the prompt and decode reserve");
            sample.hit_tokens += result->device_prefix_tokens;
        }
        sample.evicted_pages = Coordinator().DeviceEvictions().admission.blocks - evictions_before;
        Require(sample.evicted_pages == config_.EvictedPages(), "unexpected eviction count");
        // All 16 tables stay alive through the last admission.
        tables.clear();
        RestoreFullCache();
        return sample;
    }

    SchedulerSample RunSchedulerSample() {
        CheckFull();
        SchedulerSample sample;
        const auto stats_before = scheduler_.GetHostCacheStats();
        sample.submit = Measure([&] { scheduler_.SubmitRequests(requests_); });
        {
            ExecutionPlan plan;
            sample.plan = Measure([&] {
                plan = scheduler_.NextExecutionPlan({.allow_prefill = true, .apply_prefill_watermark = false});
            });
            const ForwardBatch* batch = test::FindForwardBatch(plan);
            Require(batch != nullptr && batch->NumExtends() == requests_.size(), "all requests must be admitted");
            sample.admitted = batch->NumExtends();
            for (std::size_t i = 0; i < sample.admitted; ++i) {
                Require(batch->extend_prefix_lens[i] == config_.HitTokens(), "scheduler prefix hit differs");
                Require(batch->input_lengths[i] + batch->extend_prefix_lens[i] == config_.request_tokens,
                        "all prompts must complete in this plan");
            }
        }
        const auto stats_after = scheduler_.GetHostCacheStats();
        sample.hit_tokens = stats_after.prefix_hits.device_hit_tokens - stats_before.prefix_hits.device_hit_tokens;
        sample.evicted_pages =
            stats_after.device_evictions.admission.blocks - stats_before.device_evictions.admission.blocks;
        Require(sample.evicted_pages == config_.EvictedPages(), "scheduler eviction count differs");
        // Cancel before acknowledging the synthetic forward: release resources
        // without publishing new query KV into the baseline cache.
        for (const auto& request : requests_) {
            scheduler_.Advance(ExecutionEvent{}.With(forward::Abort{.request_id = request.request_id}));
            scheduler_.Advance(
                ExecutionEvent{}.With(forward::ExtendResult{.request_id = request.request_id, .tokens = {}}));
        }
        scheduler_.NextExecutionPlan({.allow_prefill = true, .apply_prefill_watermark = false});
        RestoreFullCache();
        return sample;
    }

    LifecycleSample RunLifecycleSample() {
        CheckFull();
        scheduler_.DrainKvEvents();
        LifecycleSample sample;
        auto requests = requests_;
        for (auto& request : requests) request.max_new_tokens = 2;
        sample.submit = Measure([&] { scheduler_.SubmitRequests(requests); });
        ExecutionPlan first;
        sample.first_plan = Measure(
            [&] { first = scheduler_.NextExecutionPlan({.allow_prefill = true, .apply_prefill_watermark = false}); });
        const auto* batch = test::FindForwardBatch(first);
        Require(batch && batch->NumExtends() == requests.size(), "lifecycle must admit every request");
        // Packing is one: these physical IDs also identify the locations to
        // remove after completion, including any published decode tail.
        const auto rows = batch->BlockTables().at("full_attention");
        const auto entries_before = Coordinator().GroupPrefixIndex(0).NumEntries(Pool());
        ExecutionEvent result;
        for (const auto& request : requests)
            result.With(forward::ExtendResult{.request_id = request.request_id, .tokens = {17}});
        sample.advance = Measure([&] { scheduler_.Advance(std::move(result)); });
        ExecutionPlan next;
        sample.publish_plan = Measure(
            [&] { next = scheduler_.NextExecutionPlan({.allow_prefill = true, .apply_prefill_watermark = false}); });
        const auto* decode = test::FindForwardBatch(next);
        Require(decode && decode->NumExtends() == 0 && decode->request_ids.size() == requests.size(),
                "lifecycle must schedule one decode per request");
        sample.published_prompt_pages = Coordinator().GroupPrefixIndex(0).NumEntries(Pool()) - entries_before;
        Require(sample.published_prompt_pages == config_.EvictedPages() - requests.size(),
                "publication must add only the missing prompt pages");
        ExecutionEvent finish;
        for (const auto& request : requests) {
            finish.With(forward::ExtendResult{.request_id = request.request_id, .tokens = {19}});
            finish.With(forward::Finish{.request_id = request.request_id});
        }
        sample.finish = Measure([&] { scheduler_.Advance(std::move(finish)); });
        sample.cleanup_plan =
            Measure([&] { scheduler_.NextExecutionPlan({.allow_prefill = true, .apply_prefill_watermark = false}); });
        sample.events = Measure([&] {
            for (const auto& event : scheduler_.DrainKvEvents())
                if (const auto* stored = std::get_if<KvBlockStoredEvent>(&event))
                    sample.stored_event_blocks += stored->block_hashes.size();
        });
        Require(scheduler_.ActiveLcmBlocks() == 0 && scheduler_.WaitingSize() == 0,
                "lifecycle must release every request");
        // Remove only the new query suffixes, from leaves towards their
        // historical prefix. Abort alone would leave these cached.
        for (const auto& row : rows) {
            for (std::size_t i = row.size(); i-- > static_cast<std::size_t>(config_.HitTokens() / config_.page_size);) {
                CacheCoordinatorTestAccess::EvictCachedBlock(Coordinator(), 0, {row[i], 0});
            }
        }
        scheduler_.DrainKvEvents();
        RestoreFullCache();
        return sample;
    }

private:
    std::vector<std::int32_t> HistoryTokens(std::int32_t history, std::int32_t count) const {
        std::vector<std::int32_t> tokens(count);
        std::iota(tokens.begin(), tokens.end(), history * config_.request_tokens + 1);
        return tokens;
    }

    void RestoreHistory(const std::vector<std::string>& hashes) {
        auto probe = Coordinator().ProbePrefix(hashes);
        const auto hit = probe.device.num_common_tokens;
        const auto total = static_cast<std::int32_t>(hashes.size()) * config_.page_size;
        Require(hit < total, "restoration must add missing pages");
        std::array<BlockTable, 1> tables;
        const std::array demands{
            GroupDemand{.table = &tables[0], .extent = DenseGrowth{total - hit}, .reserve_tokens = 0}};
        const auto result = Coordinator().Admit(std::move(probe), demands, RequestProgress{}, {}, std::nullopt,
                                                std::nullopt, /*evict_blocks_past_segments=*/false, /*session_id=*/{});
        Require(result.has_value(), "cache fill failed");
        if (enable_events_) {
            const auto history = static_cast<std::int32_t>(&hashes - histories_.data());
            Request request(RequestSpec{.request_id = "fill", .tokens = HistoryTokens(history, total)},
                            config_.page_size, Role::kFused);
            SchedulerTestAccess::RegisterKvEventPages(scheduler_, request, hashes, hit / config_.page_size);
        }
        CacheCompletedBlocksForTest(Coordinator(), tables, hashes, result->access_epoch, hit / config_.page_size, total,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);
    }

    void RestoreFullCache() {
        const auto evictions = Coordinator().DeviceEvictions().admission.blocks;
        for (const auto& hashes : histories_) {
            // Equal-epoch full-history eviction takes the suffix first. The
            // endpoint lookup avoids scanning millions of untouched entries.
            if (!Coordinator().GroupPrefixIndex(0).Contains(
                    Pool(), CacheKey{.group_id = 0, .content_hash = ContentHash{hashes.back()}})) {
                RestoreHistory(hashes);
            }
        }
        Require(Coordinator().DeviceEvictions().admission.blocks == evictions, "restoration must not evict pages");
        CheckFull();
    }

    FullCacheConfig config_;
    bool enable_events_;
    Scheduler scheduler_;
    std::vector<std::vector<std::string>> histories_;
    std::vector<RequestSpec> requests_;
};

}  // namespace tokenspeed::benchmark
