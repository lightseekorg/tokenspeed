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
// shaped by what the cache groups need: a junction cut when a hit's KV
// coverage reached further than its state checkpoint; for replayable groups
// the re-fed window is debited from a hit chunk's budget and a prompt's final
// chunk is never shorter than the window (ChunkKeepingFinalWindow). Every
// function here is pure token arithmetic on the coordinator's facts; the
// forward planner calls PrefillChunkTokens and never branches on a group kind
// itself.

// The page boundary before a long prompt's replay tail where a reusable
// State checkpoint is worth writing: a later prompt that resends this one or
// diverges after the boundary resumes there instead of at an earlier
// checkpoint. The tail is the final token, the prefix replay tokens and the
// largest replayable window, whichever is longest. Zero for a prompt within
// one chunk budget (a later prompt re-prefills it cheaply) and for a model
// without State groups. The forward that spans the boundary writes the
// state there (SchedulerConfig::state_checkpoint_in_forward); no chunk is cut.
// Restricted State groups instead save the last aligned segment entrance,
// including prompts that fit one chunk budget.
std::int32_t PrefillStateCheckpointTokens(const CacheCoordinator& coordinator, std::int32_t prompt_tokens,
                                          std::int32_t chunk_budget, std::int32_t prefix_replay_tokens);

// Chunk size that never leaves the prompt's final chunk shorter than
// replay_window: bounded replay narrows the model's decoder to the prompt's
// last window, which must therefore sit in one forward. A chunk that would
// leave 0 < remainder < replay_window is shortened so exactly replay_window
// tokens remain; when the whole remainder must go in one chunk and does not
// fit, 0 is returned and the request waits for a fresher budget. With
// replay_window 0 the chunk is returned unchanged.
std::int32_t ChunkKeepingFinalWindow(std::int32_t prefill_tokens, std::int32_t unscheduled, std::int32_t replay_window);

// Token count of a local prefill chunk starting at first_pos with the given
// unscheduled prompt, remaining round budget and pending junction (0 means
// none): a chunk is cut to end at the junction so the natural checkpoint it
// leaves coincides with KV already cached, and the next prompt resumes there.
// `resumes_hit` marks a hit's first chunk, whose re-fed replay window
// (CacheCoordinator::ReplayTokens) comes out of the budget before any new
// token. A junction inside the prompt's final replay window yields to the
// window rule: the chunk passes the junction rather than waiting for a chunk
// end that satisfies both. Returns 0 when no chunk fits this round's budget.
// For restricted State groups, nonterminal chunks contain whole checkpoint
// segments; a terminal remainder is carried whole whenever it fits.
std::int32_t PrefillChunkTokens(const CacheCoordinator& coordinator, std::int32_t first_pos, bool resumes_hit,
                                std::int32_t unscheduled, std::int32_t token_budget, std::int32_t junction_tokens);

// Budget a fused mixed-mode decode batch leaves untouched so a pending local
// prefill can still advance: for replayable groups the replay window plus the
// larger of another window (a final chunk carried whole) and one prefix page;
// 0 otherwise. SchedulerConfig::Validate holds max_scheduled_tokens to it.
std::int32_t MinPrefillChunkTokens(const CacheCoordinator& coordinator);

}  // namespace tokenspeed
