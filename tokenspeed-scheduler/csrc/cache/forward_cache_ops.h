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
#include <map>
#include <span>
#include <string>
#include <vector>

#include "cache/cache_types.h"
#include "cache/kv_cache_coordinator.h"

namespace tokenspeed {

struct SchedulerConfig;  // defined in scheduler/types.h; only used by-ref below

// Prefill first chunk: claim the admission-layer prefix match into each table,
// then allocate pages for the NEW tokens only. The match itself happens once at
// scheduler admission (single num_common_blocks source drives the token math,
// the gate charge and window.begin; see the M9 spec), so the ops layer is pure
// claim + Acquire(remainder). Claimed pages are FULL pages -- ClaimHitBlocks
// leaves no tail credit -- so the incremental Acquire's page math is exact for
// num_new_tokens. Returns false if the shared pool cannot supply the new pages;
// the Acquire is check-then-act (allocates nothing on failure), but the prefix
// blocks claimed by ClaimCommonPrefix REMAIN in the tables -- on failure the
// caller must FreeRequest the tables to release those refs (unchanged
// contract). tables must be sized to coordinator.NumGroups(). No window slide
// happens here: nothing is computed before the first chunk (num_computed = 0
// frees nothing).
//
// TODO(flat-swa-alloc): Acquire allocates ceil(chunk/page) SWA pages even when
// chunk >> window -- the pages below the window are only released by the NEXT
// op's slide, so the SWA group's peak-transient allocation is the chunk size,
// not the ~ceil((window-1)/page)+1 live pages the window actually needs.
// Not allocating slid-out pages in the first place changes Acquire semantics;
// until then the first-chunk admission gate must charge this transient peak
// (full chunk), and does (schedulePrefillFirstChunk).
bool PrefillFirstChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                       const CoordinatorMatch& hit, std::int32_t num_new_tokens);

// Subsequent prefill chunk: register the pages the PRIOR chunks completed,
// slide the SWA window forward to num_computed_tokens, then allocate this
// chunk's pages. num_full_blocks = number of fully-filled logical pages before
// this chunk (content_hashes covers them). num_computed_tokens counts tokens
// whose KV write is enqueued in an earlier scheduled batch (possibly still in
// flight under overlap scheduling; execution-stream ordering makes the slide
// safe) -- the tokens of chunks 0..k-1 when scheduling chunk k (same
// convention as DecodeStep): the
// chunk's earliest query at position C needs keys back to C - window + 1, so
// pages fully below that can go mid-prefill instead of being retained until
// the second decode step. Ordering inside the op is load-bearing:
//   - CacheFullBlocks BEFORE AdvanceWindow: registration skips null holes, so
//     the reverse order would lose the punched pages' hashes forever. A page
//     registered and immediately slid out becomes a cached-with-hash free
//     block -- the pool's normal reusable-cached state.
//   - AdvanceWindow BEFORE Acquire (slide-before-acquire, see DecodeStep): the
//     slide's freed pages fund this chunk's own Acquire; schedulePrefill's
//     admission gate credits the slide via BlocksFreedByAdvance in lockstep.
// Registration and slide run unconditionally (they only touch already-computed
// pages); false means the Acquire found the pool short and allocated nothing.
bool PrefillChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                  std::span<const std::string> content_hashes, std::int32_t num_tokens,
                  std::int32_t num_full_blocks, std::int32_t num_computed_tokens);

// One decode step: slide the SWA window forward FIRST (frees pages fully out
// of window in sliding-window groups), then allocate the new token's page
// need. Slide-before-acquire lets a decoding request fund its own step from
// the pages its slide releases; the reverse order can gate every decoding
// request off under pool pressure even though their slides would free enough
// (collective starvation). num_computed_tokens counts tokens whose KV is
// already computed BEFORE this round's forward (the pending query sits at
// that position). No hash-registration ordering is affected: CacheFullBlocks
// runs in the prefill/finalize ops, never here.
bool DecodeStep(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                std::int32_t num_tokens, std::int32_t num_computed_tokens);

// Prefill -> decode transition: register the remaining full prefill pages'
// content hashes so later requests can prefix-hit them, slide the SWA window
// forward to num_computed_tokens (the full prefill length: every prefill
// token's KV is computed before the pending first decode step, whose query at
// position P needs keys back to P - window + 1), then allocate the pages
// reserved for the first decode step. Sliding here instead of leaving it to
// the SECOND decode step's DecodeStep releases the last chunk's out-of-window
// pages one round earlier. Same load-bearing op ordering as PrefillChunk:
// CacheFullBlocks, then AdvanceWindow, then Acquire; scheduleDecode's gate for
// a PrefillDone request credits this slide. Returns false if the shared pool
// cannot supply the reservation (registration and slide have already happened;
// nothing is allocated on failure).
bool FinalizePrefillAndReserveDecode(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                                     std::span<const std::string> content_hashes, std::int32_t reserve_tokens,
                                     std::int32_t num_computed_tokens);

// Translate the Python-provided per-group cache config into KvCacheSpecs (one
// per paged_cache_group, group_id = index). All groups share config.page_size.
std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config);

// Finish / abort: return every page in every table to the pool. No-op on empty
// tables (nothing allocated yet, or already released by a failure path).
void FreeRequest(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables);

// flat per-group block table extraction. One row per group: the BlockId()
// sequence of table.Blocks(), with null-block holes written as 0, in absolute
// logical-page order (no compaction). key = group_id string, taken from
// config_.paged_cache_groups[i].group_id (e.g. "full"/"swa"), so the keys match
// the downstream / Python assertions rather than a bare index.
std::map<std::string, std::vector<std::int32_t>> BuildFlatBlockTables(
    const std::vector<BlockTable>& tables, std::span<const std::string> group_ids);

}  // namespace tokenspeed
