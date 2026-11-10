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
// small for some of the images the knob takes. One P-role test turns the
// in-forward checkpoints on for every seed and drops short prompts' ends
// (drop_short_prompt_end_checkpoints).
//
// The scheduler must never raise; under write-back no copy pins its Device
// source; every write-back row keeps its two ends in one shard's bucket;
// after every plan no retraction image was refused once planned; and under
// write-back, after every plan, the blocks evicted for the cache (by
// admission, stranded suffix or restore) equal the sum of the drain's
// outcomes, the dead and demoted entries it dropped included. On the P role
// a plan leaves at most one prefill incomplete, and it runs last, after any
// whole prompts its chunk gave way to (prefill_chunk_yield_tokens): the
// completing prefills form a prefix of the batch. With KV cache events on,
// every plan's drain removes only blocks the feed stored and stores none
// twice, alike on mirrored ranks. Once everything is acknowledged, no Host
// entry is pinned and nothing is in flight.
//
// The demotion runs turn superseded checkpoint demotion on, with the
// in-forward checkpoints it requires (so the dead prompt ends and finishes
// they skip as well), and draw conversations: most prompts are the next turn
// of a finished one, which renders its last tokens anew, so a turn resumes at
// the in-forward checkpoint the turn before wrote and chains of them form.
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
#include <optional>
#include <random>
#include <set>
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

// What a run changes in its seed's draws. The defaults change nothing: every
// draw is still made, so a seed keeps its random stream.
struct LoopKnobs {
    // SchedulerConfig::demote_superseded_state_checkpoints.
    bool demote{false};
    // Replace the drawn write policy (true: write-back) and in-forward
    // checkpoints when set.
    std::optional<bool> write_back;
    std::optional<bool> in_forward;
    // A prompt draw below repeat_below repeats a live prompt; then one below
    // continue_below continues a finished prompt, with new user tokens always
    // under extend_always, otherwise half the time (or as the next turn of a
    // conversation, below). Any other draw is a system prompt with new user
    // tokens.
    double repeat_below{0.15};
    double continue_below{0.45};
    bool extend_always{false};
    // A continued prompt is the next turn of a conversation: the finished
    // prompt (without what it generated) with its last one to four tokens
    // rendered anew, then new user tokens; no later prompt continues the same
    // one.
    bool conversations{false};
    // Multiplies the drawn Device size.
    std::int32_t device_pages_factor{1};
    // A chunk budget in place of 48 tokens (0: none).
    std::int32_t chunk_budget{0};
    // SchedulerConfig::prefill_chunk_yield_tokens (0: off; the P role only).
    std::int32_t yield_tokens{0};
    // Leave out the State groups and the sliding window: no junction cut and
    // no final-window rule then shapes a chunk.
    bool full_attention_only{false};
    // SchedulerConfig::enable_kv_cache_events, the feed drained and checked
    // after every plan.
    bool kv_events{false};
    // Restricted State groups of this checkpoint grain (0: unrestricted),
    // with in-forward checkpoints and no sliding window.
    std::int32_t state_grain{0};
    // P role only: SchedulerConfig::drop_short_prompt_end_checkpoints, with
    // the in-forward checkpoints on whatever in_forward says.
    bool drop_short_prompt_ends{false};
};

struct ClosedLoop {
    std::mt19937 rng;
    Role role;
    bool mirrored;
    LoopKnobs knobs;
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
    // Ended (aborted or failed) while their transfer may still touch their
    // pages: the peer reports each transfer stopped some rounds later.
    std::set<std::string> stopping;
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
    // The P role's plans with an incomplete prefill that whole prompts ran
    // beside (prefill_chunk_yield_tokens), how many of those chunks ended on
    // a page (with restricted State groups, a segment), and the prompts they
    // carried.
    std::uint64_t yielding_chunks{0};
    std::uint64_t yielding_chunks_on_page{0};
    std::uint64_t carried_prompts{0};
    // The P role's requests a plan left incomplete: each later chunk of
    // theirs is a resident's.
    std::set<std::string> prefilling;
    // With KV cache events: per rank, the blocks the feed reports stored,
    // and how many stored events the first rank's feed carried.
    std::vector<std::set<std::uint64_t>> kv_stored_blocks;
    std::uint64_t kv_stored_events{0};

    int Rand(int lo, int hi_exclusive) { return std::uniform_int_distribution<int>(lo, hi_exclusive - 1)(rng); }
    double Uniform() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); }
    // A request's place in submission order: ids are "r" and a counter.
    static int Serial(const std::string& id) { return std::stoi(id.substr(1)); }

    // Drains every rank's KV event feed after a plan: a removal names a block
    // the feed stored, no block is stored twice without a removal between,
    // and mirrored ranks report the same events.
    void DrainKvEvents() {
        if (!knobs.kv_events) {
            return;
        }
        kv_stored_blocks.resize(schedulers.size());
        std::vector<std::pair<bool, std::uint64_t>> first_feed;
        for (std::size_t rank = 0; rank < schedulers.size(); ++rank) {
            std::vector<std::pair<bool, std::uint64_t>> feed;
            for (const KvCacheEvent& event : schedulers[rank]->DrainKvEvents()) {
                if (const auto* removed = std::get_if<KvBlockRemovedEvent>(&event)) {
                    for (const std::uint64_t hash : removed->block_hashes) {
                        if (kv_stored_blocks[rank].erase(hash) == 0) {
                            throw std::logic_error("round " + std::to_string(round) +
                                                   ": a KV event removes a block never stored");
                        }
                        feed.emplace_back(false, hash);
                    }
                    continue;
                }
                const auto& stored = std::get<KvBlockStoredEvent>(event);
                for (const std::uint64_t hash : stored.block_hashes) {
                    if (!kv_stored_blocks[rank].insert(hash).second) {
                        throw std::logic_error("round " + std::to_string(round) + ": a KV event stores a block twice");
                    }
                    feed.emplace_back(true, hash);
                }
                kv_stored_events += rank == 0 ? 1 : 0;
            }
            if (rank == 0) {
                first_feed = std::move(feed);
            } else if (feed != first_feed) {
                throw std::logic_error("round " + std::to_string(round) + ": mirrored ranks' KV event feeds differ");
            }
        }
    }

    ClosedLoop(unsigned seed, Role loop_role, bool mirrored_ranks, LoopKnobs loop_knobs)
        : rng(seed), role(loop_role), mirrored(mirrored_ranks), knobs(loop_knobs) {
        // value_or's argument is evaluated either way: the draw is made.
        write_back = knobs.write_back.value_or(Rand(0, 2) == 0);
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
        // Restricted State groups require in-forward checkpoints.
        const bool in_forward = knobs.in_forward.value_or(Rand(0, 2) == 0) || knobs.state_grain > 1;
        const std::int32_t host_pages = std::vector<std::int32_t>{9, 17, 33}[Rand(0, 3)];
        kv_shards = std::vector<std::int32_t>{1, 2, 4, 8}[Rand(0, 4)];
        const std::int32_t device_pages = std::vector<std::int32_t>{24, 40, 64}[Rand(0, 3)] * knobs.device_pages_factor;
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
        cfg.max_scheduled_tokens = knobs.chunk_budget > 0 ? knobs.chunk_budget : 48;
        cfg.max_batch_size = 6;
        cfg.enable_kv_cache_events = knobs.kv_events;
        cfg.decode_input_tokens = 1;
        cfg.overlap_schedule_depth = overlap_depth;
        cfg.enable_mixed_prefill_decode = mixed;
        cfg.state_checkpoint_in_forward = in_forward || knobs.drop_short_prompt_ends;
        cfg.drop_short_prompt_end_checkpoints = knobs.drop_short_prompt_ends;
        cfg.demote_superseded_state_checkpoints = knobs.demote;
        cfg.prefill_chunk_yield_tokens = knobs.yield_tokens;
        cfg.cache_groups.push_back(CacheGroupConfig{.group_id = "full_attention",
                                                    .block_granularity = history_block,
                                                    .total_pages = (device_pages - 1) * packing + 1,
                                                    .cache_blocks_per_lcm_block = packing,
                                                    .attn_kind = AttnKind::kFull,
                                                    .transfer_policy = CacheTransferPolicy::FullSuffix,
                                                    .shard_count = kv_shards});
        for (const std::string group_id : {"linear_attention_0", "linear_attention_1"}) {
            if (knobs.full_attention_only) {
                break;
            }
            cfg.cache_groups.push_back(CacheGroupConfig{.group_id = group_id,
                                                        .block_granularity = 0,
                                                        .total_pages = device_pages,
                                                        .attn_kind = AttnKind::kState,
                                                        .transfer_policy = CacheTransferPolicy::LatestSnapshot,
                                                        .checkpoint_granularity = std::max(1, knobs.state_grain)});
        }
        // Keep the existing random stream while covering ordinary and
        // replayable windows under both policies and mirrored rank plans.
        if (seed % 3 == 0 && !knobs.full_attention_only && knobs.state_grain <= 1) {
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
            if (draw < knobs.repeat_below && !live.empty()) {
                auto it = live.begin();
                std::advance(it, Rand(0, static_cast<int>(live.size())));
                tokens = it->second.prompt;
            } else if (draw < knobs.continue_below && !done.empty()) {
                const auto continued = static_cast<std::size_t>(Rand(0, static_cast<int>(done.size())));
                tokens = done[continued];
                if (knobs.conversations) {
                    done.erase(done.begin() + static_cast<std::ptrdiff_t>(continued));
                    tokens.resize(tokens.size() - std::min(tokens.size(), static_cast<std::size_t>(Rand(1, 5))));
                    tokens.insert(tokens.end(), user.begin(), user.end());
                } else if (knobs.extend_always || Uniform() < 0.5) {
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
                    stopping.insert(id);
                } else {
                    events.With(pd::SucceededEvent{id});
                }
                state.transferring = false;
                state.finished = true;
                any = true;
            }
        }
        for (auto it = stopping.begin(); it != stopping.end();) {
            if (Uniform() < 0.5) {
                events.With(pd::TransferStoppedEvent{*it});
                it = stopping.erase(it);
                any = true;
            } else {
                ++it;
            }
        }
        if (any) {
            Advance(events);
        }
    }

    // Each rank builds the round's plan; the plan's cache ops are checked and
    // their ACKs scheduled.
    ExecutionPlan NextPlan() {
        ExecutionPlan plan = schedulers[0]->NextExecutionPlan(
            Scheduler::PlanOptions{.allow_prefill = true, .apply_prefill_watermark = false});
        for (std::size_t rank = 1; rank < schedulers.size(); ++rank) {
            if (!(schedulers[rank]->NextExecutionPlan(
                      Scheduler::PlanOptions{.allow_prefill = true, .apply_prefill_watermark = false}) == plan)) {
                throw std::logic_error("the mirrored schedulers planned apart at round " + std::to_string(round));
            }
        }
        DrainKvEvents();
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
        if (role == Role::kP && runs_forward) {
            // At most one prefill is left incomplete, the head of line, and
            // it runs last: the completing prefills form a prefix of the
            // batch, whole prompts its chunk gave way to included.
            std::optional<std::size_t> incomplete;
            for (std::size_t i = 0; i < batch->NumExtends(); ++i) {
                if (batch->extend_prefix_lens[i] + batch->input_lengths[i] < batch->prefill_lengths[i]) {
                    if (incomplete) {
                        throw std::logic_error("round " + std::to_string(round) +
                                               ": two prefills were left incomplete");
                    }
                    incomplete = i;
                }
            }
            if (!CompletingPrefillsFormAPrefix(*batch)) {
                throw std::logic_error("round " + std::to_string(round) +
                                       ": a completing prefill runs after the incomplete one");
            }
            if (incomplete) {
                // The prompts its chunk carried: first chunks admitted after
                // it -- beside a resident's chunk, which otherwise admits
                // nothing behind it, every one; beside a new prompt's, those
                // submitted after it.
                const std::string& holder = batch->request_ids[*incomplete];
                const bool resident = prefilling.contains(holder);
                std::uint64_t carried = 0;
                for (std::size_t i = 0; i < *incomplete; ++i) {
                    const std::string& id = batch->request_ids[i];
                    carried += !prefilling.contains(id) && (resident || Serial(id) > Serial(holder)) ? 1 : 0;
                }
                if (carried > 0) {
                    if (knobs.yield_tokens == 0) {
                        throw std::logic_error("round " + std::to_string(round) +
                                               ": a prompt was admitted behind an incomplete prefill");
                    }
                    ++yielding_chunks;
                    carried_prompts += carried;
                    // A page, or a restricted State group's whole segment.
                    const std::int32_t end = batch->extend_prefix_lens[*incomplete] + batch->input_lengths[*incomplete];
                    if (end % std::max(kP, knobs.state_grain) == 0) {
                        ++yielding_chunks_on_page;
                    }
                }
            }
            for (std::size_t i = 0; i < batch->NumExtends(); ++i) {
                if (incomplete && i == *incomplete) {
                    prefilling.insert(batch->request_ids[i]);
                } else {
                    prefilling.erase(batch->request_ids[i]);
                }
            }
        }
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
            // or restore, including reusable released inputs) add up to the drain's outcomes: a Host entry
            // already, carried by a store in flight, without Host room,
            // copied, or dropped as dead or as demoted; recopied and
            // cancelled blocks are among the copied ones. The counts are
            // cumulative, so the first round they differ ends the run.
            const TierTransferManager::WriteBackStats& eviction = stats.eviction_stores;
            std::uint64_t outcomes = eviction.host_cached + eviction.in_flight + eviction.no_placement +
                                     stats.dead_entries_not_written_back + stats.demoted_entries_not_written_back;
            for (const std::uint64_t blocks : eviction.blocks) {
                outcomes += blocks;
            }
            const std::uint64_t evicted =
                stats.device_evictions.admission.blocks + stats.device_evictions.stranded_suffix.blocks +
                stats.device_evictions.restore.blocks + stats.device_evictions.request_reclaim_for_write_back;
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
                    // but the last was consumed. A conversation's next turn
                    // renders the reply anew, so it keeps the prompt alone.
                    std::vector<std::int32_t> sequence = it->second.prompt;
                    if (!it->second.out.empty() && !knobs.conversations) {
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
                if (role == Role::kP) {
                    // The simulated peer has stopped reading; delayed local
                    // forwards still land below before the pages are released.
                    abort.With(pd::FailedEvent{id});
                }
                Advance(abort);
                live.erase(id);
                if (role != Role::kFused) {
                    stopping.insert(id);
                }
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
            if (live.empty() && stopping.empty() && (batch == nullptr || batch->request_ids.empty()) &&
                !has_cache_operation && !plan.remote_decode.has_value() && acks.empty()) {
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
        ClosedLoop loop(seed, Role::kP, /*mirrored_ranks=*/false, LoopKnobs{});
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
        ClosedLoop loop(seed, Role::kP, /*mirrored_ranks=*/true, LoopKnobs{});
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

// 32 seeds of the P role with the in-forward checkpoints on and short
// prompts' ends dropped, each otherwise of its own configuration; every
// fourth drives two schedulers as the ranks of one replica.
TEST(ClosedLoopRandomTest, PrefillRoleDroppingShortPromptEndsNeverRaisesAndDrains) {
    // The eviction write-backs the write-back runs made.
    std::uint64_t write_backs = 0;
    for (unsigned seed = 0; seed < 32; ++seed) {
        const bool mirrored = seed % 4 == 3;
        ClosedLoop loop(seed, Role::kP, mirrored, LoopKnobs{.drop_short_prompt_ends = true});
        SCOPED_TRACE(::testing::Message()
                     << "seed " << seed << (loop.write_back ? " write-back" : " write-through") << " overlap "
                     << loop.overlap_depth << " shards " << loop.kv_shards << (mirrored ? " mirrored" : ""));
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
        write_backs += loop.schedulers[0]->GetHostCacheStats().eviction_stores.ops;
    }
    EXPECT_GT(write_backs, 0u);
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
        ClosedLoop loop(seed, Role::kFused, /*mirrored_ranks=*/false, LoopKnobs{});
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
        ClosedLoop loop(seed, Role::kFused, /*mirrored_ranks=*/true, LoopKnobs{});
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

// Superseded checkpoint demotion on conversations: 16 seeds per role and
// write policy, every fourth with mirrored ranks, with in-forward checkpoints
// and a Device four times the drawn size, so a conversation's checkpoints
// often outlive its next two turns; three prompts in four are a finished
// prompt's next turn. Under write-back the demoted entries an eviction drops
// count among the drain's outcomes.
TEST(ClosedLoopRandomTest, DemotionChainsNeverRaiseAndDrain) {
    // Per role and write policy: the entries demoted, and those an eviction
    // dropped instead of writing them back.
    std::map<std::pair<Role, bool>, std::pair<std::uint64_t, std::uint64_t>> demoted;
    for (const Role role : {Role::kP, Role::kFused}) {
        for (const bool write_back : {false, true}) {
            for (unsigned seed = 0; seed < 16; ++seed) {
                ClosedLoop loop(seed, role, /*mirrored_ranks=*/seed % 4 == 3,
                                LoopKnobs{.demote = true,
                                          .write_back = write_back,
                                          .in_forward = true,
                                          .repeat_below = 0.05,
                                          .continue_below = 0.8,
                                          .conversations = true,
                                          .device_pages_factor = 4});
                SCOPED_TRACE(::testing::Message()
                             << "seed " << seed << (role == Role::kP ? " P" : " fused")
                             << (write_back ? " write-back" : " write-through") << " shards " << loop.kv_shards);
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
                const Scheduler& scheduler = *loop.schedulers[0];
                auto& [entries, dropped] = demoted[{role, write_back}];
                entries += scheduler.DemotedCheckpointEntries();
                dropped += scheduler.GetHostCacheStats().demoted_entries_not_written_back;
            }
        }
    }
    for (const Role role : {Role::kP, Role::kFused}) {
        for (const bool write_back : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << (role == Role::kP ? "P" : "fused") << (write_back ? " write-back" : " write-through"));
            const auto& [entries, dropped] = demoted[{role, write_back}];
            EXPECT_GT(entries, 0u) << "the runs formed demotion chains";
            if (write_back) {
                EXPECT_GT(dropped, 0u) << "and evicted demoted entries";
            } else {
                EXPECT_EQ(dropped, 0u);
            }
        }
    }
}

// prefill_chunk_yield_tokens on the P role, at half the chunk budget: 16
// seeds per variant -- drawn configurations; in-forward checkpoints; with a
// 24-token budget and prompts that mostly continue finished ones, State
// groups as drawn (also with the KV event feed checked after every plan) and
// full attention alone; drawn configurations with the KV event feed; and
// restricted State groups of a two-page grain -- every fourth with mirrored
// ranks. Besides draining with nothing pinned, no plan leaves two prefills
// incomplete or runs a completing prefill after the incomplete one
// (NextPlan), so every prompt carried beside a chunk that gave way completes
// in its round, ahead of that chunk. Such a chunk ends on a page unless a
// junction or the final-window rule cuts it; with full attention alone
// neither does, and restricted State groups end it on a whole segment.
TEST(ClosedLoopRandomTest, PrefillChunkYieldNeverRaisesAndDrains) {
    const LoopKnobs short_budget{.continue_below = 0.7, .extend_always = true, .chunk_budget = 24, .yield_tokens = 12};
    LoopKnobs short_budget_kv_events = short_budget;
    short_budget_kv_events.kv_events = true;
    LoopKnobs full_attention = short_budget;
    full_attention.full_attention_only = true;
    const struct {
        const char* name;
        LoopKnobs knobs;
    } variants[] = {
        {"drawn", LoopKnobs{.yield_tokens = 24}},
        {"in-forward", LoopKnobs{.in_forward = true, .yield_tokens = 24}},
        {"24-token budget", short_budget},
        {"24-token budget, KV events", short_budget_kv_events},
        {"full attention", full_attention},
        {"drawn, KV events", LoopKnobs{.yield_tokens = 24, .kv_events = true}},
        {"restricted State groups", LoopKnobs{.yield_tokens = 24, .state_grain = 2 * kP}},
    };
    for (const auto& variant : variants) {
        std::uint64_t yielding_chunks = 0;
        std::uint64_t on_page = 0;
        std::uint64_t carried = 0;
        std::uint64_t kv_stored_events = 0;
        for (unsigned seed = 0; seed < 16; ++seed) {
            ClosedLoop loop(seed, Role::kP, /*mirrored_ranks=*/seed % 4 == 3, variant.knobs);
            SCOPED_TRACE(::testing::Message()
                         << variant.name << " seed " << seed << (loop.write_back ? " write-back" : " write-through")
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
            yielding_chunks += loop.yielding_chunks;
            on_page += loop.yielding_chunks_on_page;
            carried += loop.carried_prompts;
            kv_stored_events += loop.kv_stored_events;
        }
        SCOPED_TRACE(variant.name);
        EXPECT_GT(yielding_chunks, 0u) << "chunks gave way";
        EXPECT_GE(carried, yielding_chunks);
        if (variant.knobs.full_attention_only || variant.knobs.state_grain > 1) {
            EXPECT_EQ(on_page, yielding_chunks) << "every chunk that gave way ended on a page or segment";
        } else {
            EXPECT_GT(on_page, 0u);
        }
        if (variant.knobs.kv_events) {
            EXPECT_GT(kv_stored_events, 0u) << "the feed reported stored blocks";
        }
    }
}

}  // namespace tokenspeed::test
