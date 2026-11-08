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

// What the KV-event publisher costs the scheduler's step. Not a test: it
// prints JSON and asserts only that the workload it timed was the intended
// one.
//
// Reproduce with the same optimized build and an otherwise idle CPU:
//   cmake -S tokenspeed-scheduler -B /tmp/tokenspeed-scheduler-bench \
//     -DCMAKE_BUILD_TYPE=Release -DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON \
//     -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=OFF
//   cmake --build /tmp/tokenspeed-scheduler-bench \
//     --target tokenspeed_scheduler_kv_events_benchmark -j4
//   /tmp/tokenspeed-scheduler-bench/tokenspeed_scheduler_kv_events_benchmark \
//     <history|hybrid|hybrid_window> <on|off> [repeats]
//
// One layout and one publisher setting per process. The workload is 254
// prefix-sharing chat requests (three system prompts, 48 sessions of three to
// six turns, each turn re-sending the conversation plus 24 new tokens and
// generating 12, a quarter of cold one-shot requests mixed in), run once to
// warm the cache and the allocator, then `repeats` times (default 10) with
// fresh request ids, under a closed loop of eight. Every round plans, advances
// the forwards the plan asked for, and, with the publisher on, drains the KV
// events as the control plane does after each round. Timings are per round
// (plan + advances + drain), per plan and per drain; allocations are the
// global operator new calls the round made; RSS is read from /proc after the
// warm-up and at the end. JSON on stdout; timings are informational, with no
// threshold in CI.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <new>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "scheduler/scheduler.h"

// How a round drains with the publisher on: the frame the control plane
// hands to its publisher thread, counted in bytes.
#define KV_EVENTS_BENCH_DRAIN(scheduler) ((scheduler).DrainKvEventFrame(0.0, 0).size())

namespace {

std::atomic<std::uint64_t> g_allocations{0};

}  // namespace

// Every operator new of the process counts: the scheduler's containers, the
// event descriptors and the publisher's frames alike. The benchmark's own
// bookkeeping allocates outside the timed and counted regions.
void* operator new(std::size_t size) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
    return operator new(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return operator new(size, std::nothrow);
}
void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete[](void* p) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}

namespace tokenspeed {
namespace {

using Clock = std::chrono::steady_clock;

void Require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "kv_events benchmark: " << message << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

constexpr std::int32_t kPage = 16;
constexpr std::int32_t kDevicePages = 1025;
constexpr std::size_t kConcurrency = 8;

SchedulerConfig MakeConfig(std::string_view layout, bool publisher_on) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = kPage;
    cfg.device_allocator.total_pages = kDevicePages;
    cfg.host_allocator.total_pages = 0;
    cfg.snapshot_allocator.total_pages = 1;  // the null page alone: no retraction images
    cfg.max_scheduled_tokens = 512;
    cfg.max_batch_size = 16;
    cfg.enable_l3_storage = false;
    cfg.disable_l2_cache = true;
    cfg.disable_prefix_cache = false;
    cfg.enable_kv_cache_events = publisher_on;
    cfg.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full_attention",
        .block_granularity = kPage,
        .total_pages = kDevicePages,
        .attn_kind = AttnKind::kFull,
    });
    if (layout == "hybrid" || layout == "hybrid_window") {
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "linear_attention_0",
            .block_granularity = 0,
            .total_pages = kDevicePages,
            .attn_kind = AttnKind::kState,
        });
    }
    if (layout == "hybrid_window") {
        CacheGroupConfig window{
            .group_id = "sliding_attention",
            .block_granularity = kPage,
            .total_pages = kDevicePages,
            .attn_kind = AttnKind::kSlidingWindow,
        };
        window.sliding_window_tokens = 4 * kPage;
        cfg.cache_groups.push_back(std::move(window));
    }
    return cfg;
}

struct Turn {
    std::string id;
    std::vector<std::int32_t> tokens;
    std::int32_t max_new;
};

// The same stream as the lane's Python driver: deterministic under the seed.
std::vector<Turn> Workload(unsigned seed) {
    std::mt19937 rng(seed);
    auto randint = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    auto coin = [&](double p) { return std::uniform_real_distribution<double>(0.0, 1.0)(rng) < p; };
    std::vector<std::vector<std::int32_t>> prefixes(3);
    for (int p = 0; p < 3; ++p) {
        for (int i = 0; i < 96; ++i) {
            prefixes[p].push_back(1000 * (p + 1) + i);
        }
    }
    struct Session {
        int s, t;
        std::vector<std::int32_t> tokens;
    };
    std::vector<Session> sessions;
    for (int s = 0; s < 48; ++s) {
        std::vector<std::int32_t> history = prefixes[s % 3];
        const int turns = randint(3, 6);
        for (int t = 0; t < turns; ++t) {
            for (int k = 0; k < 24; ++k) {
                history.push_back(50'000 + s * 1000 + t * 100 + k);
            }
            sessions.push_back(Session{s, t, history});
            for (int k = 0; k < 12; ++k) {
                history.push_back(60'000 + s * 1000 + t * 100 + k);
            }
        }
    }
    std::map<int, std::vector<std::pair<int, std::vector<std::int32_t>>>> by_turn;
    for (auto& session : sessions) {
        by_turn[session.t].emplace_back(session.s, session.tokens);
    }
    std::vector<Turn> requests;
    int serial = 0;
    for (auto& [t, turn] : by_turn) {
        std::shuffle(turn.begin(), turn.end(), rng);
        for (auto& [s, tokens] : turn) {
            requests.push_back(Turn{"s" + std::to_string(s) + "t" + std::to_string(t), tokens, 12});
            ++serial;
            if (coin(0.25)) {
                std::vector<std::int32_t> cold;
                const int n = randint(40, 200);
                for (int k = 0; k < n; ++k) {
                    cold.push_back(70'000 + serial * 10 + k);
                }
                requests.push_back(Turn{"cold" + std::to_string(serial), cold, 12});
            }
        }
    }
    return requests;
}

long ResidentKb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            return std::strtol(line.c_str() + 6, nullptr, 10);
        }
    }
    return -1;
}

struct Live {
    std::int32_t prompt{0};
    std::int32_t computed{0};
    std::int32_t generated{0};
    std::int32_t max_new{0};
};

struct Samples {
    std::vector<std::int64_t> round_ns, plan_ns, drain_ns;
    std::vector<std::uint64_t> allocations;
    std::uint64_t events{0};
    std::uint64_t rounds{0};
};

// One pass of the stream through the scheduler. Measures when `samples` is
// given. Request ids carry the pass so a replay is new work for the queue and
// a cache hit for the prefix index.
void RunStream(Scheduler& scheduler, const std::vector<Turn>& requests, bool publisher_on, int pass, Samples* samples) {
    std::vector<Turn> pending = requests;
    for (Turn& turn : pending) {
        turn.id = "p" + std::to_string(pass) + turn.id;
    }
    std::size_t next = 0;
    std::unordered_map<std::string, Live> live;
    std::vector<std::string> finished;
    std::vector<ExecutionEvent> advances;
    std::uint64_t rounds = 0;
    while (next < pending.size() || !live.empty()) {
        while (next < pending.size() && live.size() < kConcurrency) {
            const Turn& turn = pending[next++];
            scheduler.SubmitRequests({RequestSpec{
                .request_id = turn.id,
                .tokens = turn.tokens,
                .max_new_tokens = turn.max_new,
            }});
            live[turn.id] = Live{static_cast<std::int32_t>(turn.tokens.size()), 0, 0, turn.max_new};
        }
        // The round: plan, the forwards' results, the finishes, the drain.
        const std::uint64_t allocations_before = g_allocations.load(std::memory_order_relaxed);
        const auto round_start = Clock::now();
        const ExecutionPlan plan = scheduler.NextExecutionPlan();
        const auto plan_end = Clock::now();
        finished.clear();
        for (const auto& operation : plan.Operations()) {
            const auto* batch = std::get_if<ForwardBatch>(&operation);
            if (batch == nullptr || batch->request_ids.empty()) {
                continue;
            }
            for (std::size_t i = 0; i < batch->request_ids.size(); ++i) {
                const std::string& id = batch->request_ids[i];
                Live& st = live.at(id);
                const std::int32_t prefix = i < batch->extend_prefix_lens.size() ? batch->extend_prefix_lens[i] : 0;
                if (st.computed == 0 && prefix > 0) {
                    st.computed = prefix;
                }
                st.computed += batch->input_lengths[i];
                ExecutionEvent event;
                if (st.computed >= st.prompt + st.generated) {
                    ++st.generated;
                    event.With(forward::ExtendResult{.request_id = id, .tokens = {90'000 + st.generated}});
                    if (st.generated >= st.max_new) {
                        finished.push_back(id);
                    }
                } else {
                    event.With(forward::ExtendResult{.request_id = id, .tokens = {}});
                }
                scheduler.Advance(std::move(event));
            }
        }
        for (const std::string& id : finished) {
            ExecutionEvent event;
            event.With(forward::Finish{.request_id = id});
            scheduler.Advance(std::move(event));
            live.erase(id);
        }
        const auto drain_start = Clock::now();
        std::size_t drained = 0;
        if (publisher_on) {
            drained = KV_EVENTS_BENCH_DRAIN(scheduler);
        }
        const auto round_end = Clock::now();
        const std::uint64_t allocations_after = g_allocations.load(std::memory_order_relaxed);
        ++rounds;
        if (samples != nullptr) {
            samples->round_ns.push_back((round_end - round_start).count());
            samples->plan_ns.push_back((plan_end - round_start).count());
            samples->drain_ns.push_back((round_end - drain_start).count());
            samples->allocations.push_back(allocations_after - allocations_before);
            samples->events += drained;
        }
        Require(rounds < 200'000, "the closed loop did not converge");
    }
    if (samples != nullptr) {
        samples->rounds += rounds;
    }
}

template <typename T>
double Mean(const std::vector<T>& xs) {
    return xs.empty() ? 0.0 : std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

template <typename T>
T Percentile(std::vector<T> xs, double p) {
    if (xs.empty()) {
        return T{};
    }
    std::sort(xs.begin(), xs.end());
    return xs[static_cast<std::size_t>(p * static_cast<double>(xs.size() - 1))];
}

int Main(int argc, char** argv) {
    Require(argc >= 3, "usage: <history|hybrid|hybrid_window> <on|off> [repeats]");
    const std::string layout = argv[1];
    const std::string publisher = argv[2];
    Require(layout == "history" || layout == "hybrid" || layout == "hybrid_window", "unknown layout");
    Require(publisher == "on" || publisher == "off", "publisher is on or off");
    const int repeats = argc >= 4 ? std::atoi(argv[3]) : 10;
    const bool on = publisher == "on";
    const std::vector<Turn> requests = Workload(7);
    Require(requests.size() > 200, "the workload has its requests");

    SchedulerConfig cfg = MakeConfig(layout, on);
    cfg.Validate();
    Scheduler scheduler(cfg);
    RunStream(scheduler, requests, on, 0, nullptr);  // warm-up: cold cache, allocator growth
    const long rss_after_warmup_kb = ResidentKb();
    Samples samples;
    for (int pass = 1; pass <= repeats; ++pass) {
        RunStream(scheduler, requests, on, pass, &samples);
    }
    const long rss_end_kb = ResidentKb();
    Require(samples.rounds > 0, "no rounds measured");

    const std::uint64_t allocation_total =
        std::accumulate(samples.allocations.begin(), samples.allocations.end(), 0ULL);
    std::size_t rounds_with_allocations = 0;
    for (const auto a : samples.allocations) {
        rounds_with_allocations += a > 0 ? 1 : 0;
    }
    std::printf(
        "{\"layout\":\"%s\",\"publisher\":\"%s\",\"repeats\":%d,\"requests_per_pass\":%zu,\"rounds\":%llu,"
        "\"drained_bytes\":%llu,"
        "\"round_ns\":{\"mean\":%.1f,\"p50\":%lld,\"p99\":%lld},"
        "\"plan_ns\":{\"mean\":%.1f,\"p50\":%lld,\"p99\":%lld},"
        "\"drain_ns\":{\"mean\":%.1f,\"p50\":%lld,\"p99\":%lld},"
        "\"allocations_per_round\":{\"mean\":%.3f,\"max\":%llu,\"rounds_with_any\":%zu},"
        "\"rss_kb\":{\"after_warmup\":%ld,\"end\":%ld}}\n",
        layout.c_str(), publisher.c_str(), repeats, requests.size(), static_cast<unsigned long long>(samples.rounds),
        static_cast<unsigned long long>(samples.events), Mean(samples.round_ns),
        static_cast<long long>(Percentile(samples.round_ns, 0.5)),
        static_cast<long long>(Percentile(samples.round_ns, 0.99)), Mean(samples.plan_ns),
        static_cast<long long>(Percentile(samples.plan_ns, 0.5)),
        static_cast<long long>(Percentile(samples.plan_ns, 0.99)), Mean(samples.drain_ns),
        static_cast<long long>(Percentile(samples.drain_ns, 0.5)),
        static_cast<long long>(Percentile(samples.drain_ns, 0.99)),
        static_cast<double>(allocation_total) / static_cast<double>(samples.rounds),
        static_cast<unsigned long long>(*std::max_element(samples.allocations.begin(), samples.allocations.end())),
        rounds_with_allocations, rss_after_warmup_kb, rss_end_kb);
    return 0;
}

}  // namespace
}  // namespace tokenspeed

int main(int argc, char** argv) {
    return tokenspeed::Main(argc, argv);
}
