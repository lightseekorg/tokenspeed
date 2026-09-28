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
// group: the shape the group's table takes (a dense extent or a sparse suffix
// for token-spanned groups, a StateStep for a State group's two slots) and
// the reserve it holds beyond the tokens it computes. The forward planner
// states the round once (tokens, PrefillReserve, the state step) and these
// helpers derive each group's share by its kind (CacheGroupConfig::Kind);
// replayable groups are shaped by CacheCoordinator::Admit itself.

// One demand per table: a copy of `prototype` bound to each History table,
// `state_step` for each State table (whose reserve is always 0).
std::vector<GroupDemand> MakeGroupDemands(std::vector<BlockTable>& tables,
                                          std::span<const CacheGroupConfig> cache_groups, GroupDemand prototype,
                                          StateStep state_step);

// What a prefill admission holds beyond the chunk it computes, stated once
// per round in tokens. Every cache group derives its own reserve from it in
// ReservePrefillDemands -- the only writer of GroupDemand::reserve_tokens.
struct PrefillReserve {
    // Width of the decode step that follows the completed prompt. The P role
    // never decodes locally, but the forward that completes a prompt drafts
    // the first candidate window into this slot before the remote decode.
    std::int32_t decode_input_tokens{};
    bool completes_prefill{false};
    // Rest of the prompt plus escalating decode room, prepaid by a decoding
    // role at first-chunk admission (Request::AdmissionHeadroom); 0 on later
    // chunks and on the P role.
    std::int32_t prompt_headroom_tokens{};
    // Sliding-window growth recycling cannot fund, prepaid by a first chunk
    // beside another page holder (docs/design/scheduler.md §1): the prefix hit
    // and the prompt left for later local chunks, both 0 on later chunks and
    // with no other holder. Whether the hit counts is decided per group, from
    // the prompt's own tokens after it.
    std::int32_t swa_hit_tokens{0};
    std::int32_t swa_later_chunk_tokens{0};
    std::int32_t swa_prompt_after_hit_tokens{0};

    // The decode slot when the chunk completes the prompt, else 0.
    std::int32_t DecodeTokens() const { return completes_prefill ? decode_input_tokens : 0; }
};

// Sets every History group's reserve_tokens from the round's PrefillReserve,
// by kind: full-history groups hold every token the round is accountable
// for, including the prepaid prompt headroom; sliding-window groups recycle
// slid-out pages and hold the decode slot, raised on a first chunk to the
// growth recycling cannot fund (the swa_* terms, capped by the window's
// lookback and the prompt headroom). A State group's two slots hold no
// token reserve.
void ReservePrefillDemands(std::span<GroupDemand> demands, std::span<const CacheGroupConfig> cache_groups,
                           const PrefillReserve& reserve);

}  // namespace tokenspeed
