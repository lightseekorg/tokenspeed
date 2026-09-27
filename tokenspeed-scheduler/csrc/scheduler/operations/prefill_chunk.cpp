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

std::int32_t AlignPrefillChunk(std::int32_t first_pos, std::int32_t unscheduled, std::int32_t token_budget,
                               std::int32_t prefix_granularity, std::int32_t promotion_boundary_tokens) {
    _assert(first_pos >= 0 && unscheduled >= 0 && token_budget >= 0, "prefill positions must be non-negative");
    _assert(prefix_granularity > 0, "prefix_granularity must be > 0");
    std::int32_t chunk_size = std::min(unscheduled, token_budget);
    if (promotion_boundary_tokens > first_pos) {
        chunk_size = std::min(chunk_size, promotion_boundary_tokens - first_pos);
    }
    if (chunk_size == unscheduled) {
        return chunk_size;
    }

    const std::int32_t prefix_page_offset = first_pos % prefix_granularity;
    if (prefix_page_offset != 0) {
        const std::int32_t tokens_to_boundary = prefix_granularity - prefix_page_offset;
        return token_budget >= tokens_to_boundary ? tokens_to_boundary : 0;
    }
    return chunk_size - chunk_size % prefix_granularity;
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
                                std::int32_t unscheduled, std::int32_t token_budget,
                                std::int32_t promotion_boundary_tokens) {
    // The re-fed window of a hit rides in the same forward, so it comes out of
    // the budget before any new token; with nothing left the request waits.
    const std::int32_t budget = token_budget - (resumes_hit ? coordinator.ReplayTokens(first_pos) : 0);
    if (budget <= 0) {
        return 0;
    }
    const std::int32_t plain = std::min(budget, unscheduled);
    std::int32_t prefill_tokens = plain;
    if (coordinator.HasMambaStateGroup() || promotion_boundary_tokens > 0) {
        prefill_tokens = AlignPrefillChunk(first_pos, unscheduled, budget, coordinator.PrefixGranularity(),
                                           promotion_boundary_tokens);
        if (prefill_tokens == 0) {
            return 0;
        }
    }
    const std::int32_t window = coordinator.ReplayWindowTokens();
    const std::int32_t kept = ChunkKeepingFinalWindow(prefill_tokens, unscheduled, window);
    if (kept > 0 || prefill_tokens == plain) {
        return kept;
    }
    // Alignment stopped the chunk at a promotion boundary that lies inside the
    // prompt's final window, where no chunk end satisfies both rules and the
    // request would wait forever. The boundary is a reuse opportunity, not an
    // invariant: pass it (the closed group recomputes the promoted pages) and
    // size the chunk from the plain budget under the window rule alone.
    return ChunkKeepingFinalWindow(plain, unscheduled, window);
}

std::int32_t MinPrefillChunkTokens(const CacheCoordinator& coordinator) {
    // A hit chunk re-feeds up to one replay window and then still needs room
    // to advance: every new token when fewer than a window remain, or one
    // prefix page when a promotion boundary aligns it.
    const std::int32_t window = coordinator.ReplayWindowTokens();
    const std::int32_t replay_reserve = window > 0 ? window + std::max(window, coordinator.PrefixGranularity()) : 0;
    return std::max(coordinator.HasMambaStateGroup() ? coordinator.PrefixGranularity() : 0, replay_reserve);
}

}  // namespace tokenspeed
