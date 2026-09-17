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

#include "cache/coordinator/cache_coordinator.h"

namespace tokenspeed {

// How a local prefill chunk is cut. A chunk is min(budget, unscheduled)
// shaped by what the cache groups need: whole-checkpoint chunks for
// snapshot-state groups and boundary-aligned chunks under a pending promotion
// (AlignPrefillChunk); for replayable groups the re-fed window is debited
// from a hit chunk's budget and a prompt's final chunk is never shorter than
// the window (ChunkKeepingFinalWindow). Every function here is pure token
// arithmetic on the coordinator's facts; the forward planner calls
// PrefillChunkTokens and never branches on a group kind itself.

// Token count of the next chunk starting at first_pos, bounded by unscheduled
// tokens, token_budget and a pending promotion boundary (0 means none).
// Incomplete chunks end on a prefix boundary; a final extent stays whole even
// when it ends off-boundary. Returns 0 when no legal chunk fits the budget.
std::int32_t AlignPrefillChunk(std::int32_t first_pos, std::int32_t unscheduled, std::int32_t token_budget,
                               std::int32_t prefix_granularity, std::int32_t promotion_boundary_tokens);

// Chunk size that never leaves the prompt's final chunk shorter than
// replay_window: bounded replay narrows the model's decoder to the prompt's
// last window, which must therefore sit in one forward. A chunk that would
// leave 0 < remainder < replay_window is shortened so exactly replay_window
// tokens remain; when the whole remainder must go in one chunk and does not
// fit, 0 is returned and the request waits for a fresher budget. With
// replay_window 0 the chunk is returned unchanged.
std::int32_t ChunkKeepingFinalWindow(std::int32_t prefill_tokens, std::int32_t unscheduled, std::int32_t replay_window);

// Token count of a local prefill chunk starting at first_pos with the given
// unscheduled prompt, remaining round budget and pending promotion boundary
// (0 means none). `resumes_hit` marks a hit's first chunk, whose re-fed
// replay window (CacheCoordinator::ReplayTokens) comes out of the budget
// before any new token. A promotion boundary inside the prompt's final replay
// window yields to the window rule: the chunk passes the boundary rather than
// waiting for a chunk end that satisfies both. Returns 0 when no chunk fits
// this round's budget.
std::int32_t PrefillChunkTokens(const CacheCoordinator& coordinator, std::int32_t first_pos, bool resumes_hit,
                                std::int32_t unscheduled, std::int32_t token_budget,
                                std::int32_t promotion_boundary_tokens);

// Budget a fused mixed-mode decode batch leaves untouched so a pending local
// prefill can still advance: one state-checkpoint page for a snapshot-state
// group; for replayable groups the replay window plus the larger of another
// window (a final chunk) and one prefix page (a promotion-aligned chunk);
// 0 otherwise. SchedulerConfig::Validate holds max_scheduled_tokens to it.
std::int32_t MinPrefillChunkTokens(const CacheCoordinator& coordinator);

}  // namespace tokenspeed
