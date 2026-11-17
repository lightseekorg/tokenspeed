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

// The vocabulary of a request's admission, shared by the cache coordinator,
// its components and the scheduler: what a
// probe matches against and what it found, the point whose copy-in blocks an
// admission reserves, and what an admission or a restore hands back.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/coordinator/session_registry.h"
#include "cache/core/cache_block_ref.h"
#include "cache/core/cache_types.h"
#include "cache/core/content_hash.h"

namespace tokenspeed {

// What a probe may match against: the chain hashes of the prompt's full
// pages below max_tokens, and the tokens themselves so tail hashes can be
// computed on demand (tokens.size() >= max_tokens). A probe without
// tokens matches full pages only.
struct PromptPrefix {
    std::span<const std::string> prefix_hashes;
    std::span<const std::int32_t> tokens;
    std::int32_t max_tokens{0};
    // An additional known candidate at max_tokens, used by storage before
    // any local publication. It is a hint, never proof of cached data.
    std::optional<TailPublication> tail;
    // Optional keys prepared by PrefixProber::PreparePrefixKeys from a hash chain that
    // includes prefix_hashes as a prefix. Consumers stay within
    // max_tokens and the prompt hash bounds.
    PreparedPrefixKeys prepared_keys;
    // Bounded request-local remote hints; no global tail-table mutation.
    std::span<const std::int32_t> shared_prefix_tokens;
};

struct PrefixProbe {
    // A tail entry the tier matched: the boundary's offset inside its
    // page, its hash, and per group the keys of its blocks below T with
    // the hits over them (a hole marks a block the group does not need or
    // does not hold; the D role leaves the partial block a hole so the
    // peer's transfer fills a fresh page instead of a copy).
    struct TailHit {
        std::int32_t tail_tokens{0};
        ContentHash hash;
        std::vector<std::vector<CacheKey>> group_keys;
        std::vector<GroupPrefixProbe> per_group;
    };
    struct Tier {
        // The boundary T this tier resumes at, in tokens.
        std::int32_t num_common_tokens{0};
        // The full pages below it: num_common_tokens less the tail.
        std::int32_t full_page_tokens{0};
        // The KV groups' own boundary before the State groups shortened
        // it: the junction a chunk may be cut at so the next prompt
        // resumes there. On the Host tier, the KV reach of the Device's
        // prefix plus a Host suffix when the Host holds part of that KV,
        // else 0; the junction takes the larger of the two tiers'.
        std::int32_t kv_tokens{0};
        // Device prefix-closed coverage before windows or State shorten
        // the hit. Reused only within this probe and its storage queries.
        std::int32_t closed_kv_tokens{0};
        // Hits over group_keys (the full pages): a token-spanned group's
        // relative to the tier's floor (the Device's full pages for the
        // Host tier), a State group's from page 0 on both tiers, since a
        // Host checkpoint may sit on the Device's last full page.
        std::vector<GroupPrefixProbe> per_group;
        // Present when T is not page-aligned and the tier holds some
        // group's blocks of the tail.
        std::optional<TailHit> tail;
    };

    PreparedPrefixKeys group_keys;
    // The D role only: every State group's checkpoint at exactly the
    // prompt end (keys and hits for the State groups alone), claimed as
    // slot 0 so the peer's transfer can omit the state. Absent when any
    // State group lacks it.
    std::optional<TailHit> state_at_prompt_end;
    // The Host-suffix rule: a hit at T takes the Device's KV pages up to
    // its full_page_tokens and the Host's from there on (the Host's floor),
    // the KV tail whole from one tier, and each State group's checkpoint
    // from whichever tier holds it, the Device first. When the Device alone
    // resumes furthest the Host tier mirrors its boundary and claims
    // nothing.
    Tier device;
    // The Host suffix of the hit. Admission loads these; nothing an
    // admission touches can miss.
    Tier host;
    // Accounting only, never read by a decision: the Device-only
    // boundary before a Host suffix replaced it. Zero without a Host pool
    // and on the D role.
    std::int32_t device_only_hit_tokens{0};
};
// The publication point an admission reserves copy-in blocks for: the
// fused role's prompt end, which the request's first decode publishes and
// keeps writing past (OwnerTail::kCopiedIn).
struct CopyInPoint {
    std::int32_t tokens{0};
};
struct AdmissionResult {
    std::int32_t device_prefix_tokens{0};
    std::int32_t host_prefix_tokens{0};
    // The KV groups' boundary when it exceeds the admitted one (a State
    // group had no checkpoint there): the junction a later chunk is cut
    // at so its natural checkpoint lands where the KV is already cached.
    std::int32_t junction_tokens{0};
    // The D role: every State group's prompt-end checkpoint was claimed,
    // so the peer omits the state transfer.
    bool state_prefix_cached{false};
    std::uint64_t access_epoch{0};
    // Host L2 -> Device copies: an admission's Host prefix hits (unkeyed;
    // their Device destinations are published at admission), or a
    // restore's published slots (keyed, re-published at the ACK).
    std::vector<BlockTransfer> load_pairs;
    // Restore only: snapshot pool -> Device copies of the private slots.
    std::vector<BlockTransfer> snapshot_pairs;
    // Fresh device child pages appended by ordinary Acquire, aligned by
    // group_id, for the plan to zero. Cache hits, copy destinations
    // (Host hits, a restore's rebuilt slots, the pairs below) and the
    // blocks a remote landing fills whole (GroupDemand::landed_tokens,
    // StateStep::kLand's slot 1) are excluded: the copy fills them whole.
    std::vector<std::vector<std::int32_t>> new_page_ids;
    // Device -> Device whole-block copies, aligned by group_id, as
    // (source, destination) virtual block ids: a hitter's private copy of
    // the partial block containing its boundary, or an owner's copy-in at
    // a publication point it keeps writing past (into the block it
    // reserved for the slot). Both ends share a bucket; the destination
    // sits in the request's table.
    std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>> pages_to_copy;
    // Aligned by group_id: the fresh block each State group's
    // StateStep::checkpoint acquired outside the table (also listed in
    // new_page_ids), null elsewhere. The caller holds it until the
    // request's next publication point publishes it.
    std::vector<CacheBlockRef> state_checkpoint_blocks;
    // With a copy-in point: aligned by group_id, one entry per slot of
    // CopyInSlots(i, point), in slot order -- the copy-in block this
    // admission reserved for a slot whose block it allocates, fills from
    // the prefix or rebuilds, in that block's bucket, and null for a slot
    // an earlier admission covered or a later one will. Held outside the
    // table -- and not in new_page_ids: the copy-in fills a block whole --
    // until the admission that publishes the point spends it (Admit's
    // copy_in_blocks). Empty without a copy-in point.
    std::vector<std::vector<CacheBlockRef>> copy_in_blocks;
    // The tail whose checkpoints the Device supplied as the State groups'
    // slot 0: a first admission resuming inside a page, where an earlier
    // turn ended. The end state the request publishes supersedes it
    // (MarkSupersededStateInput).
    std::optional<ContentHash> resumed_tail_hash;
    // The first admission of a request with a session id: that session's
    // registration under access_epoch, for the caller to hold as long as
    // the request lives. Empty otherwise.
    SessionAccess session_access;
};

}  // namespace tokenspeed
