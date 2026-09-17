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
#include <span>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/core/cache_config.h"
#include "cache/core/cache_types.h"

namespace tokenspeed {

// What a scheduled step asks the coordinator for, one GroupDemand per cache
// group: the shape of the pages it materializes (a dense extent, or a sparse
// suffix for groups whose absolute slots below stay holes) and the reserve it
// holds beyond the tokens it computes. The forward planner states the round
// once (tokens, PrefillReserve) and these helpers derive each group's share
// by its retention and family; replayable groups are shaped by
// CacheCoordinator::Admit itself.

// One demand per table, all copies of `prototype` bound to their table.
std::vector<GroupDemand> MakeGroupDemands(std::vector<BlockTable>& tables, GroupDemand prototype);

// What a prefill admission holds beyond the chunk it computes, stated once
// per round in tokens. Every cache group derives its own reserve from it in
// ReservePrefillDemands -- the only writer of GroupDemand::reserve_tokens.
struct PrefillReserve {
    // Width of the decode step that follows the completed prompt; 0 on the P
    // role, which never decodes locally.
    std::int32_t decode_input_tokens{};
    bool completes_prefill{false};
    // Rest of the prompt plus escalating decode room, prepaid by a decoding
    // role at first-chunk admission (Request::AdmissionHeadroom); 0 on later
    // chunks and on the P role.
    std::int32_t prompt_headroom_tokens{};
    // Whether this admission finishes shaping the snapshot-state groups: the
    // chunk that completes the prompt, or a remote landing.
    bool reserve_snapshot_state_growth{false};

    // The decode slot when the chunk completes the prompt, else 0.
    std::int32_t DecodeTokens() const { return completes_prefill ? decode_input_tokens : 0; }
};

// Reserve in tokens beyond a snapshot-state endpoint: at least one group block
// or the full decode width, whichever is larger. It is additional to the
// materialized suffix. Admission and its startup bound share this rule;
// callers decide whether the role and round need any reserve at all.
std::int64_t SnapshotStateReserveTokens(std::int64_t block_granularity, std::int64_t decode_tokens);

// Sets every group's reserve_tokens from the round's PrefillReserve, by
// retention: full-history groups hold every token the round is accountable
// for, including the prepaid prompt headroom; sliding-window groups recycle
// slid-out pages and hold only the decode slot; snapshot-state groups bank
// one growth block (SnapshotStateReserveTokens) on the admission that
// finishes shaping them and 0 otherwise.
void ReservePrefillDemands(std::span<GroupDemand> demands, std::span<const CacheGroupConfig> cache_groups,
                           const PrefillReserve& reserve);

// Absolute token boundary of the first output state in (before_tokens,
// after_tokens]: the latest prefix boundary crossed, or after_tokens if none.
// This is not a block-table slot; callers convert it with (boundary - 1) / the
// group's block_granularity. An internal checkpoint and the endpoint are both
// materialized in the same model forward; an aligned endpoint needs one output.
std::int32_t StateCheckpointMaterializationStart(std::int32_t before_tokens, std::int32_t after_tokens,
                                                 std::int32_t prefix_granularity);

// A local prefill chunk over (before_tokens, after_tokens] materializes each
// snapshot-state group as a sparse suffix: the last completed checkpoint and
// the final continuation state, both written by this one model forward, with
// earlier slots left as holes to preserve absolute block-table positions.
void MakeSnapshotStatePrefillSparse(std::span<GroupDemand> demands, std::span<const CacheGroupConfig> cache_groups,
                                    const CacheCoordinator& coordinator, std::int32_t before_tokens,
                                    std::int32_t after_tokens);

}  // namespace tokenspeed
