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

#include "scheduler/operations/prefill_chunk.h"

#include <algorithm>

#include "utils.h"

namespace tokenspeed {

std::int32_t PrefillStateCheckpointTokens(const CacheCoordinator& coordinator, std::int32_t prompt_tokens,
                                          std::int32_t chunk_budget, std::int32_t prefix_replay_tokens) {
    _assert(prompt_tokens >= 0 && chunk_budget > 0 && prefix_replay_tokens >= 0,
            "prefill checkpoint inputs must be non-negative with a positive budget");
    const std::int32_t tail = std::max({1, coordinator.ReplayWindowTokens(), prefix_replay_tokens});
    if (prompt_tokens <= chunk_budget || prompt_tokens <= tail) {
        return 0;
    }
    for (std::int32_t group = 0; group < coordinator.NumGroups(); ++group) {
        if (coordinator.GroupIsState(group)) {
            const std::int32_t page = coordinator.PrefixGranularity();
            return (prompt_tokens - tail) / page * page;
        }
    }
    return 0;
}

std::int32_t ChunkKeepingFinalWindow(std::int32_t prefill_tokens, std::int32_t unscheduled,
                                     std::int32_t replay_window) {
    _assert(0 <= prefill_tokens && prefill_tokens <= unscheduled, "a chunk cannot exceed the unscheduled prompt");
    _assert(replay_window >= 0, "replay_window must be non-negative");
    const std::int32_t remainder = unscheduled - prefill_tokens;
    if (replay_window == 0 || remainder == 0 || remainder >= replay_window) {
        return prefill_tokens;
    }
    return std::max(0, unscheduled - replay_window);
}

std::int32_t PrefillChunkTokens(const CacheCoordinator& coordinator, std::int32_t first_pos, bool resumes_hit,
                                std::int32_t unscheduled, std::int32_t token_budget, std::int32_t junction_tokens) {
    _assert(first_pos >= 0 && unscheduled >= 0 && token_budget >= 0, "prefill positions must be non-negative");
    // The re-fed window of a hit rides in the same forward, so it comes out of
    // the budget before any new token; with nothing left the request waits.
    const std::int32_t budget = token_budget - (resumes_hit ? coordinator.ReplayTokens(first_pos) : 0);
    if (budget <= 0) {
        return 0;
    }
    const std::int32_t plain = std::min(budget, unscheduled);
    std::int32_t prefill_tokens = plain;
    if (junction_tokens > first_pos) {
        prefill_tokens = std::min(prefill_tokens, junction_tokens - first_pos);
    }
    const std::int32_t window = coordinator.ReplayWindowTokens();
    const std::int32_t kept = ChunkKeepingFinalWindow(prefill_tokens, unscheduled, window);
    if (kept > 0 || prefill_tokens == plain) {
        return kept;
    }
    // The junction lies inside the prompt's final window, where no chunk end
    // satisfies both rules and the request would wait forever. The junction
    // is a reuse opportunity, not an invariant: pass it and size the chunk
    // from the plain budget under the window rule alone.
    return ChunkKeepingFinalWindow(plain, unscheduled, window);
}

std::int32_t MinPrefillChunkTokens(const CacheCoordinator& coordinator) {
    // A hit chunk re-feeds up to one replay window and then still needs room
    // to carry the prompt's final window whole, or at least a prefix page of
    // new tokens when the window is smaller than a page.
    const std::int32_t window = coordinator.ReplayWindowTokens();
    return window > 0 ? window + std::max(window, coordinator.PrefixGranularity()) : 0;
}

}  // namespace tokenspeed
