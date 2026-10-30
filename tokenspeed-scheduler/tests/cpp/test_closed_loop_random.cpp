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

// Randomized closed loops over the prefill (P) and fused roles' scheduler with
// a Host tier, driven as the event loop does: a plan is built up to the
// overlap depth before the results of the plans ahead of it land, a live
// request is now and then aborted midway, and every cache op is acknowledged
// zero to three plans late. On the P role the transfer peer bootstraps
// waiting prompts and acknowledges or fails each hand-off; the fused role
// decodes its own prompts, finishing (now and then aborting) each at its
// token limit or by chance, and an op of a plan that only restores is
// acknowledged two to five plans late. Prompts share system prefixes, repeat
// live ones and continue finished ones (on the fused role with what they
// generated), on pools small enough to evict. Each seed draws its
// configuration: the Host write policy, the history block, in-forward
// checkpoints, the Host and Device sizes and how many KV-parallel shards the
// history group is dealt over (1, 2, 4 or 8; eight blocks per parent at 8);
// on the P role the overlap depth, on the fused role (overlap depth 1) mixed
// batches and retraction: none, for capacity into a snapshot pool, or forced
// every third plan on the oldest decode (3) or prefill (-3), into a pool too
// small for some of the images the knob takes.
//
// The scheduler must never raise; under write-back no copy pins its Device
// source; every write-back row keeps its two ends in one shard's bucket;
// after every plan no retraction image was refused once planned; and under
// write-back, after every plan, the blocks evicted for the cache (by
// admission, stranded suffix or restore) equal the sum of the drain's
// outcomes. Once everything is acknowledged, no Host entry is pinned and
// nothing is in flight.
//
// The random runs acknowledge op by op, out of order and now and then twice.
// The mirrored runs drive two schedulers of one seed, as two ranks of a
// replica, with a heap block sized by the seed allocated between their
// constructions: each rank draws when an op lands on it, the op is
// acknowledged once both hold it, and the round's ACKs arrive in one batch in
// (op id, kind) order, as the runtime delivers the ACKs it intersected over
// the replica. The two must build equal plans, round after round.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kP = 8;
constexpr std::int32_t kRounds = 300;

struct ClosedLoop {
    std::mt19937 rng;
    Role role;
    bool mirrored;
    bool write_back{false};
    std::int32_t overlap_depth{0};
    std::int32_t kv_shards{1};
    // The fused role's draws: mixed batches, and retraction -- none, for
    // capacity, or forced every third plan on the oldest decode (knob 3) or
    // prefill (knob -3).
    bool mixed{false};
    bool capacity_retraction{false};
    std::int32_t force_retraction_interval{0};
    // Mirrored: allocated between the two schedulers' constructions, so the
    // second one's allocations are not the first one's at a fixed offset.
    std::vector<std::uint8_t> heap_perturbation;
    // One per rank: every event goes to each, and each builds every plan.
    std::vector<std::unique_ptr<Scheduler>> schedulers;
    std::int32_t round{0};

    std::vector<std::vector<std::int32_t>> systems;
    struct Live {
        std::vector<std::int32_t> prompt;
        bool bootstrapped{false};
        bool transferring{false};
        bool finished{false};
        // The fused role's generation: its token limit and what landed.
        std::int32_t max_new{0};
        std::vector<std::int32_t> out;
    };
    std::map<std::string, Live> live;
    std::vector<std::vector<std::int32_t>> done;
    std::deque<ExecutionPlan> pending;
    // An unacknowledged cache op: the round its ACK is due, and its kind.
    enum class Kind : std::uint8_t { kWriteBack, kLoad, kSnapshot, kRestore };
    struct Ack {
        std::int32_t due;
        std::uint32_t op_id;
        Kind kind;
    };
    std::vector<Ack> acks;
    std::int32_t next_id{0};
    std::int32_t next_token{5000};

    std::uint64_t aborts{0};
    std::uint64_t handoffs{0};
    // Plans that only decode and write back what they evict, and the
    // eviction write-back ops so far.
    std::uint64_t decode_only_write_backs{0};
    std::uint64_t eviction_ops{0};

    int Rand(int lo, int hi_exclusive) { return std::uniform_int_distribution<int>(lo, hi_exclusive - 1)(rng); }
    double Uniform() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); }

    ClosedLoop(unsigned seed, Role loop_role, bool mirrored_ranks)
        : rng(seed), role(loop_role), mirrored(mirrored_ranks) {
        write_back = Rand(0, 2) == 0;
        if (role == Role::kFused) {
            overlap_depth = 1;
            mixed = Rand(0, 2) == 0;
            const int retraction = Rand(0, 4);
            capacity_retraction = retraction == 1;
            force_retraction_interval = std::vector<std::int32_t>{0, 0, 3, -3}[retraction];
        } else {
            overlap_depth = Rand(0, 2);
        }
        const std::int32_t history_block = Rand(0, 2) == 0 ? kP : kP / 2;
        const bool in_forward = Rand(0, 2) == 0;
        const std::int32_t host_pages = std::vector<std::int32_t>{9, 17, 33}[Rand(0, 3)];
        kv_shards = std::vector<std::int32_t>{1, 2, 4, 8}[Rand(0, 4)];
        const std::int32_t device_pages = std::vector<std::int32_t>{24, 40, 64}[Rand(0, 3)];
        // A parent holds at least one history block per shard.
        const std::int32_t packing = std::max(4, kv_shards);
        SchedulerConfig cfg{};
        cfg.role = role;
        cfg.prefix_granularity = kP;
        cfg.device_allocator.total_pages = device_pages;
        cfg.host_allocator.total_pages = host_pages;
        cfg.disable_l2_cache = false;
        cfg.host_write_policy =
            write_back ? SchedulerConfig::HostWritePolicy::kWriteBack : SchedulerConfig::HostWritePolicy::kWriteThrough;
        cfg.max_scheduled_tokens = 48;
        cfg.max_batch_size = 6;
        cfg.decode_input_tokens = 1;
        cfg.overlap_schedule_depth = overlap_depth;
        cfg.enable_mixed_prefill_decode = mixed;
        cfg.state_checkpoint_in_forward = in_forward;
        cfg.cache_groups.push_back(CacheGroupConfig{.group_id = "full_attention",
                                                    .block_granularity = history_block,
                                                    .total_pages = (device_pages - 1) * packing + 1,
                                                    .cache_blocks_per_lcm_block = packing,
                                                    .attn_kind = AttnKind::kFull,
                                                    .transfer_policy = CacheTransferPolicy::FullSuffix,
                                                    .shard_count = kv_shards});
        for (const std::string group_id : {"linear_attention_0", "linear_attention_1"}) {
            cfg.cache_groups.push_back(CacheGroupConfig{.group_id = group_id,
                                                        .block_granularity = 0,
                                                        .total_pages = device_pages,
                                                        .attn_kind = AttnKind::kState,
                                                        .transfer_policy = CacheTransferPolicy::LatestSnapshot});
        }
        // Keep the existing random stream while covering ordinary and
        // replayable windows under both policies and mirrored rank plans.
        if (seed % 3 == 0) {
            cfg.cache_groups.push_back(CacheGroupConfig{.group_id = "sliding",
                                                        .block_granularity = history_block,
                                                        .total_pages = (device_pages - 1) * 2 + 1,
                                                        .cache_blocks_per_lcm_block = 2,
                                                        .attn_kind = AttnKind::kSlidingWindow,
                                                        .sliding_window_tokens = 12,
                                                        .replayable = seed % 5 == 0,
                                                        .transfer_policy = CacheTransferPolicy::FullSuffix});
        }
        SetTestSnapshotPool(cfg);
        if (role == Role::kFused && !capacity_retraction && force_retraction_interval == 0) {
            cfg.snapshot_allocator.total_pages = 1;
            cfg.max_retracted_requests = 0;
        } else if (role == Role::kFused && force_retraction_interval != 0) {
            // Too small for some of the images the knob takes: those are
            // refused by their plan.
            cfg.snapshot_allocator.total_pages = 17;
            cfg.max_retracted_requests = 2;
            cfg.debug_force_retraction_interval = force_retraction_interval;
        }
        schedulers.push_back(std::make_unique<Scheduler>(WithHostPackingFromDevice(cfg)));
        if (mirrored) {
            heap_perturbation.resize(256 * (seed + 1));
            schedulers.push_back(std::make_unique<Scheduler>(WithHostPackingFromDevice(cfg)));
        }
        for (int i = 0; i < 3; ++i) {
            std::vector<std::int32_t> system;
            const int length = Rand(3, 40);
            for (int j = 0; j < length; ++j) {
                system.push_back(Rand(2, 500));
            }
            systems.push_back(std::move(system));
        }
    }

    void Advance(const ExecutionEvent& event) {
        for (const std::unique_ptr<Scheduler>& scheduler : schedulers) {
            scheduler->Advance(event);
        }
    }

    void Submit(int count) {
        std::vector<RequestSpec> specs;
        for (int c = 0; c < count; ++c) {
            std::vector<std::int32_t> user;
            const int length = Rand(1, 30);
            for (int j = 0; j < length; ++j) {
                user.push_back(Rand(2, 500));
            }
            const double draw = Uniform();
            std::vector<std::int32_t> tokens;
            if (draw < 0.15 && !live.empty()) {
                auto it = live.begin();
                std::advance(it, Rand(0, static_cast<int>(live.size())));
                tokens = it->second.prompt;
            } else if (draw < 0.45 && !done.empty()) {
                tokens = done[static_cast<std::size_t>(Rand(0, static_cast<int>(done.size())))];
                if (Uniform() < 0.5) {
                    tokens.insert(tokens.end(), user.begin(), user.end());
                }
            } else {
                tokens = systems[static_cast<std::size_t>(Rand(0, 3))];
                tokens.insert(tokens.end(), user.begin(), user.end());
            }
            const std::int32_t max_new = Rand(1, 10);
            // Room for the output beside the prompt.
            const auto max_prompt_tokens =
                static_cast<std::size_t>(std::max(1, schedulers[0]->MaxSingleRequestTokens() - max_new - 1));
            tokens.resize(std::min(tokens.size(), max_prompt_tokens));
            const std::string id = "r" + std::to_string(next_id++);
            specs.push_back(RequestSpec{.request_id = id, .tokens = tokens, .max_new_tokens = max_new});
            live[id] = Live{.prompt = tokens, .max_new = max_new};
        }
        for (const std::unique_ptr<Scheduler>& scheduler : schedulers) {
            scheduler->SubmitRequests(specs);
        }
    }

    // One result per planned row: an intermediate chunk an empty one, the
    // final chunk or a decode the sampled token. On the fused role a request
    // finishes on the landed token that reaches its limit, or by chance, while
    // the next decode may already be out; one finish in ten is an abort.
    void Land(const ExecutionPlan& plan) {
        ExecutionEvent results;
        const ForwardBatch* batch = FindForwardBatch(plan);
        if (batch == nullptr || batch->request_ids.empty()) {
            return;
        }
        std::vector<std::string> finishes;
        for (std::size_t i = 0; i < batch->request_ids.size(); ++i) {
            const std::string& id = batch->request_ids[i];
            const bool mid_chunk = i < batch->NumExtends() &&
                                   batch->extend_prefix_lens[i] + batch->input_lengths[i] < batch->prefill_lengths[i];
            if (mid_chunk) {
                results.With(forward::ExtendResult{.request_id = id, .tokens = {}});
                continue;
            }
            const std::int32_t token = ++next_token;
            results.With(forward::ExtendResult{.request_id = id, .tokens = {token}});
            const auto state = live.find(id);
            if (role != Role::kFused || state == live.end() || state->second.finished) {
                continue;  // a finished request's delayed row
            }
            state->second.out.push_back(token);
            if (static_cast<std::int32_t>(state->second.out.size()) >= state->second.max_new || Uniform() < 0.03) {
                state->second.finished = true;
                finishes.push_back(id);
            } else if (i >= batch->NumExtends()) {
                results.With(
                    forward::UpdateReserveNumTokens{.request_id = id, .reserve_num_tokens_in_next_schedule_event = 1});
            }
        }
        for (const std::string& id : finishes) {
            if (Uniform() < 0.1) {
                results.With(forward::Abort{.request_id = id});
            } else {
                results.With(forward::Finish{.request_id = id});
            }
        }
        Advance(results);
    }

    // The transfer peer: bootstraps waiting prompts, and acknowledges (or
    // fails) the hand-offs it was given.
    void Peer(const ExecutionPlan* plan) {
        ExecutionEvent events;
        bool any = false;
        if (plan != nullptr && plan->remote_decode.has_value()) {
            for (const std::string& id : plan->remote_decode->request_ids) {
                live[id].transferring = true;
                ++handoffs;
            }
        }
        for (auto& [id, state] : live) {
            if (!state.bootstrapped && Uniform() < 0.7) {
                events.With(pd::BootstrappedEvent{id});
                state.bootstrapped = true;
                any = true;
            }
            if (state.transferring && Uniform() < 0.5) {
                if (Uniform() < 0.1) {
                    events.With(pd::FailedEvent{id});
                } else {
                    events.With(pd::SucceededEvent{id});
                }
                state.transferring = false;
                state.finished = true;
                any = true;
            }
        }
        if (any) {
            Advance(events);
        }
    }

    // Each rank builds the round's plan; the plan's cache ops are checked and
    // their ACKs scheduled.
    ExecutionPlan NextPlan() {
        ExecutionPlan plan = schedulers[0]->NextExecutionPlan();
        for (std::size_t rank = 1; rank < schedulers.size(); ++rank) {
            if (!(schedulers[rank]->NextExecutionPlan() == plan)) {
                throw std::logic_error("the mirrored schedulers planned apart at round " + std::to_string(round));
            }
        }
        std::vector<std::pair<std::uint32_t, Kind>> ops;
        bool restores = false;
        for (const Operation& operation : plan.Operations()) {
            const auto* cache_operation = std::get_if<CacheOperation>(&operation);
            if (cache_operation == nullptr) {
                continue;
            }
            if (const auto* store = std::get_if<WriteBackBatch>(cache_operation)) {
                for (std::size_t op = 0; op < store->op_ids.size(); ++op) {
                    EXPECT_FALSE(write_back && store->source_pinned[op])
                        << "round " << round << ": a write-back pinned a Device source";
                    for (std::size_t row = 0; row < store->src_pages[op].size(); ++row) {
                        // A page's bucket is (page id - 1) % shard_count; the
                        // history group is group 0, the State groups unsharded.
                        const std::int32_t shards = store->group_ids[op][row] == 0 ? kv_shards : 1;
                        EXPECT_EQ((store->src_pages[op][row] - 1) % shards, (store->dst_pages[op][row] - 1) % shards)
                            << "round " << round << ": a copy crosses buckets";
                    }
                    ops.emplace_back(store->op_ids[op], Kind::kWriteBack);
                }
            } else if (const auto* load = std::get_if<LoadBackBatch>(cache_operation)) {
                for (const std::uint32_t op_id : load->op_ids) {
                    ops.emplace_back(op_id, Kind::kLoad);
                }
            } else if (const auto* snapshot = std::get_if<SnapshotStoreBatch>(cache_operation)) {
                for (const std::uint32_t op_id : snapshot->op_ids) {
                    ops.emplace_back(op_id, Kind::kSnapshot);
                }
            } else if (const auto* restore = std::get_if<SnapshotRestoreBatch>(cache_operation)) {
                for (const std::uint32_t op_id : restore->op_ids) {
                    ops.emplace_back(op_id, Kind::kRestore);
                }
                restores = true;
            }
        }
        const ForwardBatch* batch = FindForwardBatch(plan);
        const bool runs_forward = batch != nullptr && !batch->request_ids.empty();
        // An op of a plan that only restores lands two to five plans late:
        // no forward fences it.
        const int earliest = restores && !runs_forward ? 2 : 0;
        const int latest = restores && !runs_forward ? 6 : 4;
        for (const auto& [op_id, kind] : ops) {
            // Mirrored: each rank draws when the op lands on it, the ACK
            // waits for the later.
            const int landing =
                mirrored ? std::max(Rand(earliest, latest), Rand(earliest, latest)) : Rand(earliest, latest);
            acks.push_back(Ack{.due = round + landing, .op_id = op_id, .kind = kind});
        }
        const HostCacheStats stats = schedulers[0]->GetHostCacheStats();
        if (stats.retraction_stores.image_refused != 0) {
            throw std::logic_error("round " + std::to_string(round) + ": a planned retraction image was refused");
        }
        if (write_back) {
            // The blocks evicted for the cache (by admission, stranded suffix
            // or restore) add up to the drain's outcomes: a Host entry
            // already, carried by a store in flight, without Host room,
            // copied, or dropped as dead; recopied and cancelled blocks are
            // among the copied ones. The counts are cumulative, so the first
            // round they differ ends the run.
            const TierTransferManager::WriteBackStats& eviction = stats.eviction_stores;
            std::uint64_t outcomes =
                eviction.host_cached + eviction.in_flight + eviction.no_placement + stats.dead_entries_not_written_back;
            for (const std::uint64_t blocks : eviction.blocks) {
                outcomes += blocks;
            }
            const std::uint64_t evicted = stats.device_evictions.admission.blocks +
                                          stats.device_evictions.stranded_suffix.blocks +
                                          stats.device_evictions.restore.blocks;
            if (outcomes != evicted) {
                throw std::logic_error("round " + std::to_string(round) + ": the drain's outcomes add up to " +
                                       std::to_string(outcomes) + " blocks, the evictions for the cache to " +
                                       std::to_string(evicted));
            }
            if (runs_forward && batch->NumExtends() == 0 && eviction.ops > eviction_ops) {
                ++decode_only_write_backs;
            }
            eviction_ops = eviction.ops;
        }
        return plan;
    }

    // Random runs: the due ACKs one by one, in a random order, one in twenty
    // twice. Mirrored runs: in one batch, in (op id, kind) order, once.
    void Acknowledge(bool everything) {
        std::vector<Ack> due;
        std::erase_if(acks, [&](const Ack& ack) {
            if (!everything && ack.due > round) {
                return false;
            }
            due.push_back(ack);
            return true;
        });
        if (mirrored) {
            std::ranges::sort(due, {}, [](const Ack& ack) { return std::tuple{ack.op_id, ack.kind}; });
        } else {
            std::ranges::shuffle(due, rng);
        }
        ExecutionEvent batch;
        for (const Ack& ack : due) {
            const int deliveries = !mirrored && Uniform() < 0.05 ? 2 : 1;
            for (int delivery = 0; delivery < deliveries; ++delivery) {
                ExecutionEvent single;
                ExecutionEvent& event = mirrored ? batch : single;
                switch (ack.kind) {
                    case Kind::kWriteBack:
                        event.With(cache::WriteBackDone{.op_id = ack.op_id});
                        break;
                    case Kind::kLoad:
                        event.With(cache::LoadBackDone{.op_id = ack.op_id});
                        break;
                    case Kind::kSnapshot:
                        event.With(cache::SnapshotDone{.op_id = ack.op_id});
                        break;
                    case Kind::kRestore:
                        event.With(cache::RestoreDone{.op_id = ack.op_id});
                        break;
                }
                if (!mirrored) {
                    Advance(single);
                }
            }
        }
        if (mirrored && !due.empty()) {
            Advance(batch);
        }
    }

    // Requests that left the scheduler; a finished prompt may come back as
    // the start of a later one.
    void Reap() {
        for (auto it = live.begin(); it != live.end();) {
            if (schedulers[0]->RequestTokenSize(it->first) == -1 && (it->second.finished || !it->second.transferring)) {
                if (it->second.finished) {
                    // With what the fused role generated: every landed token
                    // but the last was consumed.
                    std::vector<std::int32_t> sequence = it->second.prompt;
                    if (!it->second.out.empty()) {
                        sequence.insert(sequence.end(), it->second.out.begin(), it->second.out.end() - 1);
                    }
                    if (done.size() < 20) {
                        done.push_back(std::move(sequence));
                    } else {
                        done[static_cast<std::size_t>(Rand(0, 20))] = std::move(sequence);
                    }
                }
                it = live.erase(it);
            } else {
                ++it;
            }
        }
    }

    void Run() {
        Submit(Rand(2, 6));
        for (round = 0; round < kRounds; ++round) {
            if (Uniform() < 0.3) {
                Submit(Rand(1, 4));
            }
            if (role == Role::kP) {
                Peer(nullptr);
            }
            pending.push_back(NextPlan());
            if (role == Role::kP) {
                Peer(&pending.back());
            }
            // Now and then a live request is aborted: before its admission,
            // while it prefills (its admission's write-back maybe not
            // acknowledged yet), while its KV goes out, while it decodes or
            // while it waits to be restored; a fused run that has aborted none
            // by mid-run aborts at the first round from then on with a live
            // request.
            std::vector<std::string> abortable;
            for (const auto& [id, state] : live) {
                if (!state.finished) {
                    abortable.push_back(id);
                }
            }
            if (!abortable.empty() &&
                (Uniform() < 0.05 || (role == Role::kFused && aborts == 0 && round >= kRounds / 2))) {
                const std::string id = abortable[static_cast<std::size_t>(Rand(0, static_cast<int>(abortable.size())))];
                ExecutionEvent abort;
                abort.With(forward::Abort{.request_id = id});
                Advance(abort);
                live.erase(id);
                ++aborts;
            }
            while (static_cast<std::int32_t>(pending.size()) > overlap_depth) {
                Land(pending.front());
                pending.pop_front();
            }
            if (Uniform() < 0.1) {
                for (; !pending.empty(); pending.pop_front()) {
                    Land(pending.front());
                }
            }
            Reap();
            Acknowledge(/*everything=*/false);
        }
        // No new prompt: land what is out and acknowledge everything, round
        // after round, until every request left and a plan has nothing to run.
        for (; round < 10 * kRounds; ++round) {
            for (; !pending.empty(); pending.pop_front()) {
                Land(pending.front());
            }
            Acknowledge(/*everything=*/true);
            if (role == Role::kP) {
                Peer(nullptr);
            }
            const ExecutionPlan plan = NextPlan();
            if (role == Role::kP) {
                Peer(&plan);
            }
            Reap();
            const ForwardBatch* batch = FindForwardBatch(plan);
            const bool has_cache_operation = std::ranges::any_of(plan.Operations(), [](const Operation& operation) {
                return std::holds_alternative<CacheOperation>(operation);
            });
            if (live.empty() && (batch == nullptr || batch->request_ids.empty()) && !has_cache_operation &&
                !plan.remote_decode.has_value() && acks.empty()) {
                return;
            }
            pending.push_back(plan);
        }
        throw std::logic_error("the loop did not drain");
    }
};

}  // namespace

// 64 seeds, each of its own configuration, acknowledged op by op.
TEST(ClosedLoopRandomTest, PrefillRoleNeverRaisesAndDrains) {
    // Per write policy and shard count: the eviction write-backs and the
    // prefix loads the runs made.
    std::map<std::pair<bool, std::int32_t>, std::pair<std::uint64_t, std::uint64_t>> exercised;
    std::uint64_t aborts = 0;
    std::uint64_t handoffs = 0;
    for (unsigned seed = 0; seed < 64; ++seed) {
        ClosedLoop loop(seed, Role::kP, /*mirrored_ranks=*/false);
        SCOPED_TRACE(::testing::Message() << "seed " << seed << (loop.write_back ? " write-back" : " write-through")
                                          << " overlap " << loop.overlap_depth << " shards " << loop.kv_shards);
        try {
            loop.Run();
        } catch (const std::exception& error) {
            ADD_FAILURE() << "raised: " << error.what();
            continue;
        }
        const Scheduler& scheduler = *loop.schedulers[0];
        EXPECT_EQ(scheduler.HostPoolPinnedBlocks(), 0);
        EXPECT_TRUE(scheduler.CanClearCache());
        const HostCacheStats stats = scheduler.GetHostCacheStats();
        auto& [write_backs, loads] = exercised[{loop.write_back, loop.kv_shards}];
        write_backs += stats.eviction_stores.ops;
        loads += stats.prefix_loads.ops;
        aborts += loop.aborts;
        handoffs += loop.handoffs;
    }
    // Write-back evicted, wrote back and loaded back at every shard count.
    for (const std::int32_t shards : {1, 2, 4, 8}) {
        const auto& [write_backs, loads] = exercised[{true, shards}];
        EXPECT_GT(write_backs, 0u) << "shards " << shards;
        EXPECT_GT(loads, 0u) << "shards " << shards;
    }
    EXPECT_GT(aborts, 0u);
    EXPECT_GT(handoffs, 0u);
}

// 16 seeds, each driving two schedulers as the ranks of one replica. Seeds 16
// to 31 draw write-back at every shard count.
TEST(ClosedLoopRandomTest, MirroredPrefillSchedulersPlanAlike) {
    // Per write policy and shard count: the eviction write-backs and the
    // prefix loads the runs made.
    std::map<std::pair<bool, std::int32_t>, std::pair<std::uint64_t, std::uint64_t>> exercised;
    for (unsigned seed = 16; seed < 32; ++seed) {
        ClosedLoop loop(seed, Role::kP, /*mirrored_ranks=*/true);
        SCOPED_TRACE(::testing::Message() << "seed " << seed << (loop.write_back ? " write-back" : " write-through")
                                          << " overlap " << loop.overlap_depth << " shards " << loop.kv_shards);
        try {
            loop.Run();
        } catch (const std::exception& error) {
            ADD_FAILURE() << "raised: " << error.what();
            continue;
        }
        for (const std::unique_ptr<Scheduler>& scheduler : loop.schedulers) {
            EXPECT_EQ(scheduler->HostPoolPinnedBlocks(), 0);
            EXPECT_TRUE(scheduler->CanClearCache());
        }
        const HostCacheStats stats = loop.schedulers[0]->GetHostCacheStats();
        auto& [write_backs, loads] = exercised[{loop.write_back, loop.kv_shards}];
        write_backs += stats.eviction_stores.ops;
        loads += stats.prefix_loads.ops;
    }
    // Write-back evicted, wrote back and loaded back at every shard count.
    for (const std::int32_t shards : {1, 2, 4, 8}) {
        const auto& [write_backs, loads] = exercised[{true, shards}];
        EXPECT_GT(write_backs, 0u) << "shards " << shards;
        EXPECT_GT(loads, 0u) << "shards " << shards;
    }
}

// 64 seeds of the fused role, each of its own configuration, acknowledged op
// by op.
TEST(ClosedLoopRandomTest, FusedRoleNeverRaisesAndDrains) {
    // Per write policy and shard count: the eviction write-backs and the
    // prefix loads the runs made.
    std::map<std::pair<bool, std::int32_t>, std::pair<std::uint64_t, std::uint64_t>> exercised;
    std::uint64_t images = 0;
    std::uint64_t restores = 0;
    std::uint64_t decode_only_write_backs = 0;
    std::uint64_t aborts = 0;
    for (unsigned seed = 0; seed < 64; ++seed) {
        ClosedLoop loop(seed, Role::kFused, /*mirrored_ranks=*/false);
        SCOPED_TRACE(::testing::Message()
                     << "seed " << seed << (loop.write_back ? " write-back" : " write-through") << " mixed "
                     << loop.mixed << " capacity retraction " << loop.capacity_retraction << " knob "
                     << loop.force_retraction_interval << " shards " << loop.kv_shards);
        try {
            loop.Run();
        } catch (const std::exception& error) {
            ADD_FAILURE() << "raised: " << error.what();
            continue;
        }
        const Scheduler& scheduler = *loop.schedulers[0];
        EXPECT_EQ(scheduler.HostPoolPinnedBlocks(), 0);
        EXPECT_TRUE(scheduler.CanClearCache());
        const HostCacheStats stats = scheduler.GetHostCacheStats();
        auto& [write_backs, loads] = exercised[{loop.write_back, loop.kv_shards}];
        write_backs += stats.eviction_stores.ops;
        loads += stats.prefix_loads.ops;
        images += stats.retraction_stores.ops;
        restores += stats.restore_loads.ops;
        decode_only_write_backs += loop.decode_only_write_backs;
        aborts += loop.aborts;
    }
    // Write-back evicted, wrote back and loaded back at every shard count.
    for (const std::int32_t shards : {1, 2, 4, 8}) {
        const auto& [write_backs, loads] = exercised[{true, shards}];
        EXPECT_GT(write_backs, 0u) << "shards " << shards;
        EXPECT_GT(loads, 0u) << "shards " << shards;
    }
    EXPECT_GT(images, 0u);
    EXPECT_GT(restores, 0u);
    EXPECT_GT(decode_only_write_backs, 0u);
    EXPECT_GT(aborts, 0u);
}

// 16 seeds of the fused role, each driving two schedulers as the ranks of one
// replica.
TEST(ClosedLoopRandomTest, MirroredFusedSchedulersPlanAlike) {
    std::uint64_t write_backs = 0;
    std::uint64_t restores = 0;
    for (unsigned seed = 64; seed < 80; ++seed) {
        ClosedLoop loop(seed, Role::kFused, /*mirrored_ranks=*/true);
        SCOPED_TRACE(::testing::Message()
                     << "seed " << seed << (loop.write_back ? " write-back" : " write-through") << " mixed "
                     << loop.mixed << " capacity retraction " << loop.capacity_retraction << " knob "
                     << loop.force_retraction_interval << " shards " << loop.kv_shards);
        try {
            loop.Run();
        } catch (const std::exception& error) {
            ADD_FAILURE() << "raised: " << error.what();
            continue;
        }
        for (const std::unique_ptr<Scheduler>& scheduler : loop.schedulers) {
            EXPECT_EQ(scheduler->HostPoolPinnedBlocks(), 0);
            EXPECT_TRUE(scheduler->CanClearCache());
        }
        const HostCacheStats stats = loop.schedulers[0]->GetHostCacheStats();
        write_backs += stats.eviction_stores.ops;
        restores += stats.restore_loads.ops;
    }
    EXPECT_GT(write_backs, 0u);
    EXPECT_GT(restores, 0u);
}

}  // namespace tokenspeed::test
