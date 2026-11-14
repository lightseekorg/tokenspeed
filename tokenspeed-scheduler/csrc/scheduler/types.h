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
#include <cstdint>
#include <optional>

#include "cache/tier/storage.h"
#include <vector>

#include "cache/core/cache_config.h"
#include "utils.h"

namespace tokenspeed {

struct SchedulerConfig {
    std::int32_t prefix_granularity{};
    struct AllocatorConfig {
        // Page 0 is the null placeholder, so usable = total - 1. Validate
        // rejects a negative total_pages, which could wrap this.
        std::int32_t total_pages{};
        std::int32_t NumUsableBlocks() const { return total_pages - 1; }
    };
    AllocatorConfig host_allocator;
    AllocatorConfig device_allocator;
    // The request-private pool a retracted request's image tail lives in
    // (the unaligned tail pages, groups that never publish, and -- with no
    // Host cache -- the whole image). Never prefix-indexed, never evicted.
    // Required: total_pages == 1 (the null page alone) means the engine never
    // retracts; anything above it enables retraction on the fused and D roles.
    AllocatorConfig snapshot_allocator;
    // Slot-state blob slots in the runtime's arena, hence the most requests
    // that can be retracted at once. Required with a snapshot pool above the
    // null page, and 0 without one.
    std::int32_t max_retracted_requests{};
    // Test knob: every |N| plans the fused/D grammar retracts the oldest
    // quiescent Decoding request (N > 0) or the one Prefilling request
    // between its chunks (N < 0), bypassing ReserveCoversGeneration but not
    // the image-fit refusal. 0 is off; non-zero requires a snapshot pool;
    // INT32_MIN, with no magnitude in int32, is rejected.
    // Lives here so every mirrored rank decides identically.
    std::int32_t debug_force_retraction_interval{0};

    std::vector<CacheGroupConfig> cache_groups{};
    // Virtual children per compact Host parent, shared by L2 and snapshots.
    // Required when either Host pool exists; no capacity is reserved by group.
    std::vector<std::int32_t> host_group_packing{};

    bool HasHostCache() const { return !disable_l2_cache && host_allocator.total_pages > 1; }
    bool HasSnapshotPool() const { return snapshot_allocator.total_pages > 1; }

    // What feeds Host L2 from ordinary Device prefix entries; required with a
    // Host tier (Validate says where kWriteBack may run).
    //   kWriteThrough  published entries stream to Host at chunk ends, the
    //                  prompt end and finish.
    //   kWriteBack     nothing streams at publication: what an admission
    //                  or a restore evicts is written back to Host in the
    //                  same plan, by whole segments.
    enum class HostWritePolicy : std::uint8_t { kUnspecified, kWriteThrough, kWriteBack };
    HostWritePolicy host_write_policy{HostWritePolicy::kUnspecified};

    // The D role streams no ordinary Device cache entry to Host; its Host
    // cache holds only the published pages of retracted requests (the image's
    // L2 leg) until they are restored.
    bool StreamsDeviceCacheToHost() const {
        return HasHostCache() && role != Role::kD && host_write_policy == HostWritePolicy::kWriteThrough;
    }

    // Whether a request's completed pages are published into the Device index
    // as it computes and when it finishes, for later admissions to match.
    // Without prefix caching no admission matches either tier or prefetches
    // from L3, so a published page would only hold its block until an
    // allocation evicted it, and a Host copy of it would never be read. A
    // retraction publishes its victim regardless; with a Host cache the
    // image's L2 leg is built from that publication.
    bool PublishesCompletedPages() const { return !disable_prefix_cache; }

    std::int32_t max_scheduled_tokens{};
    // Sum of ceil((cached prefix + this chunk) / prefix_granularity) across
    // all local prefill requests in one batch. Zero leaves this unbounded.
    std::int32_t max_kv_page_num{0};
    std::int32_t max_batch_size{};
    std::int32_t decode_input_tokens{1};
    // Number of scheduler iterations that may be dispatched before the
    // accepted decode length is committed. The current event loop supports
    // only the non-overlapped (0) and one-step-overlapped (1) contracts.
    std::int32_t overlap_schedule_depth{0};
    bool disable_l2_cache{false};
    bool enable_l3_storage{false};
    std::optional<StorageConfig> storage_config;
    // Device residency events require at least one continuous Full/history group.
    bool enable_kv_cache_events{false};
    bool enable_mixed_prefill_decode{false};
    // Fused role only: decode rounds owed after a local prefill round
    // (docs/design/scheduler.md §3.3). 0 is off.
    std::int32_t prefill_decode_interval{0};
    // P and fused roles: rank new local prefills by work after Device/Host
    // prefix reuse, keeping stable ties and resident chunks first. Only the
    // oldest waiting prompt counts successful later first admissions; after
    // eight, it gains selection priority. Off preserves submission order.
    // See docs/design/scheduler.md section 1.1.
    bool enable_short_prefill_first{false};

    // The P and D roles ARE the cache-transfer PD protocol: there is no
    // disaggregated deployment without it, so everything PD-specific
    // (transfer pins, destination layouts, transfer_policy validation) keys
    // on the role directly.
    Role role{Role::kFused};

    bool disable_prefix_cache{false};
    // Minimum prompt tail that must be recomputed after a prefix-cache hit.
    // Zero preserves the default logits contract, which already recomputes at
    // least the final prompt token. The effective hit is page-aligned down.
    std::int32_t prefix_replay_tokens{0};
    // Write a reusable State checkpoint inside the prefill forward that spans
    // the page boundary before a long prompt's replay tail
    // (PrefillStateCheckpointTokens), without cutting a chunk there: that
    // forward holds one more block per State group, and the request's next
    // publication point publishes it under the boundary's page hash. The
    // runtime writes the state at ForwardBatch::extend_state_checkpoint_lens
    // into the third entry of every State group's row. The State checkpoints,
    // and their KV tails, that a next turn re-rendering the end of the
    // previous prompt and its reply cannot resume at are then skipped: a
    // prompt end that checkpoint covers (P and fused roles) and a fused finish
    // past the prompt end are published like chunk ends
    // (docs/design/scheduler.md §1.2). A model without State groups is
    // unaffected. Requires prefix caching, except on the D role, which never
    // computes a prompt and publishes as ever.
    bool state_checkpoint_in_forward{false};
    // P role, with state_checkpoint_in_forward: the finish of a prompt too
    // short for the in-forward checkpoint (PrefillStateCheckpointTokens is 0:
    // by default a prompt within one chunk budget) whose end lies inside a
    // page publishes like a chunk end -- its full pages, no KV tail and no
    // State checkpoint -- and marks no input superseded. A next turn that
    // re-renders the end of the prompt cannot resume at that tail; a longer
    // prompt that repeats the whole prompt resumes at an earlier checkpoint
    // instead. A page-aligned prompt end, a prompt long enough for the
    // in-forward checkpoint and a model without State groups are unaffected
    // (docs/design/scheduler.md §1.2). The other roles refuse it. The fused
    // role keeps its short prompt ends because this is a P-role policy,
    // chosen from PD measurements; the same case exists at the fused first
    // decode, and applying the drop there is a possible follow-up.
    bool drop_short_prompt_end_checkpoints{false};
    // A finish that publishes an in-forward State checkpoint past the one it
    // resumed from, itself an in-forward checkpoint whose request resumed from
    // an earlier one, demotes that earlier one -- two turns back on the
    // conversation -- unless another request shares it: its State entries and
    // the sliding-window blocks behind it are dated back, so both tiers evict
    // them early, and no new store or eviction write-back copies them to the
    // Host (docs/design/scheduler.md §1.2).
    // Requires state_checkpoint_in_forward, except on the D role, where it has
    // no effect; so does a model without State groups.
    bool demote_superseded_state_checkpoints{false};
    // Under cache pressure, a State checkpoint more than
    // checkpoint_retention_leaf_distance from the newest checkpoint on its
    // branch ranks as if last used this many new requests ago, or at its own
    // use when older: so it leaves before the entries used since, while entries
    // unused for longer leave first. 0 ranks every entry by its own use. Unset
    // means 16 * max_batch_size; at most kMaxCheckpointRetentionRequests.
    std::optional<std::int64_t> checkpoint_retention_requests;
    // Checkpoints within this many later requests of the newest on their
    // branch rank as it does: 0 only the newest request's, 1 the previous
    // turn's too. Below CheckpointTree::kMaxDistance.
    std::int32_t checkpoint_retention_leaf_distance{0};
    std::int64_t CheckpointRetentionRequests() const {
        return checkpoint_retention_requests.value_or(
            std::min<std::int64_t>(std::int64_t{16} * max_batch_size, kMaxCheckpointRetentionRequests));
    }
    // P role: the room, in tokens, a local prefill chunk that cannot finish
    // its prompt this round makes for bootstrapped prompts waiting behind it
    // that fit whole, so they run complete in the same forward: the chunk
    // ends on an earlier page and gives up fewer than this plus one page
    // (docs/design/scheduler.md §1.1). A chunk yields only while such prompts
    // wait, and only what they take. 0 (the default) is off; at most half of
    // max_scheduled_tokens, so a long prompt keeps about half of every round
    // or more.
    std::int32_t prefill_chunk_yield_tokens{0};

    // The single validation entry point for a scheduler configuration: every
    // scheduler scalar, every group's own invariants, and the cross-checks
    // between them. Throws std::invalid_argument on the first violation.
    //
    // The Scheduler runs this before constructing any member, because the pools
    // and the cache coordinator assert on the same fields and their assertions
    // would otherwise preempt these diagnostics.
    void Validate() const;
    // The subset of Validate() the CapacityModel needs: every field it reads
    // to size a pool, none of the page counts that describe a sized one.
    void ValidateCapacityInputs() const;
};

}  // namespace tokenspeed
