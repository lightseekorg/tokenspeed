# Scheduler

TokenSpeed uses a native scheduler in `tokenspeed-scheduler/` to decide which
requests run in each engine iteration, which KV pages are allocated, and which
cache-transfer operations must run alongside model forward work. The Python
runtime owns the server loop and model execution. The scheduler owns request
state, page accounting, prefix-cache matching, and the execution plan for the
next step.

## Runtime Loop

The serving loop is event driven:

1. Python receives and validates new API requests.
2. Ready requests are converted to `RequestSpec` objects and submitted to the
   scheduler.
3. Completed cache operations from previous iterations are committed.
4. `next_execution_plan()` asks the native scheduler for the next forward and
   cache operations.
5. Python submits cache copies, dispatches the forward operation to the model
   executor, then converts model outputs into scheduler events.
6. `advance()` applies those events and moves each request to its next state.

This keeps scheduling decisions CPU-side and deterministic while model execution
and cache copies run through the runtime-specific executors.

## Core Objects

| Object | Role |
| --- | --- |
| `SchedulerConfig` | Static limits and memory shape: page size, device pages, host pages, token budget, batch budget, role, and cache flags. |
| `RequestSpec` | Submitted request metadata: request ID, input tokens, optional rolling hashes, and optional storage-hit count. |
| `Request` | Native state-machine wrapper for one request. It owns the token container and its current FSM state. |
| `ExecutionPlan` | The scheduler output for one iteration. It may contain one flat forward operation and zero or more cache operations. |
| `ExecutionEvent` | Runtime feedback from Python, cache executors, or prefill/decode disaggregation. |

## Request States

Requests move through a finite-state machine rather than separate ad hoc
queues.

| State | Meaning |
| --- | --- |
| `Submitted` | Request has tokens but no request-pool slot or local KV allocation yet. |
| `Prefetching` | A host or storage-cache prefetch has been issued and the request waits for completion. |
| `PrefetchDone` | Prefetch completed; the request can be scheduled like a new submitted request with host cache populated. |
| `Prefilling` | Some prompt tokens have been scheduled, but the full prompt is not complete. |
| `PrefillDone` | Prompt prefill is complete; the next eligible plan can move the request into decode. |
| `Decoding` | Request is in steady-state generation. Decode steps usually issue one token, or more when speculative decoding is configured. |
| `Draining` | Request has finished and needs a device-to-host cache write-back before it can be released. |
| `WritingBack` | The write-back operation is in flight. |
| `Retracting` | A live request is being moved out of active decode to free device pages. |
| `Retracted` | Request is parked in host/prefix cache and can be loaded back later. |
| `Finished` | Request is terminal and is removed from the scheduler on the next planning pass. |

## Planning Priority

`next_execution_plan()` starts by handling pending write-backs, removing
finished requests, and collecting candidates that are not blocked by cache I/O.
It then sorts candidates by priority:

1. Continue existing `Prefilling` requests.
2. Admit `Submitted` and `PrefetchDone` requests.
3. Schedule `PrefillDone` and `Decoding` requests.
4. Recover `Retracted` requests.

The main policy is prefill-first. If any prefill work is scheduled in an
iteration, decode work is not mixed into that same forward batch. This favors
prompt progress and gives chunked prefill a simple mental model: finish the
current prefill chunks before spending the iteration on decode.

Decode is batched when no prefill work is pending. Multiple decoding requests
can be emitted in one `FlatForwardOperation` until the batch-size or token
budget is reached.

## Budgets And Limits

The scheduler enforces both request-count and token-count limits.

| Runtime knob | Native scheduler field | Effect |
| --- | --- | --- |
| `--block-size` | `page_size` | Number of tokens represented by one KV page. |
| `--max-total-tokens` or automatic memory sizing | `device_allocator.total_pages` | Device KV page pool size. |
| Host memory executor sizing | `host_allocator.total_pages` | Host KV page pool size for write-back and load-back. |
| `--chunked-prefill-size` | `max_scheduled_tokens` | Per-iteration token issue budget. |
| `--max-num-seqs` divided by attention DP size | `max_batch_size` | Per-DP-rank request-pool and forward-batch limit. |
| Speculative decoding config | `decode_input_tokens` | Number of decode tokens reserved or issued per decode step. |
| `--disaggregation-mode` | `role` | Selects fused, prefill-only, or decode-only scheduling behavior. |

For data-parallel attention, `max_batch_size` is intentionally per rank. The
Python event loop divides global `--max-num-seqs` by the attention DP size
before building `SchedulerConfig`, because the native request-pool allocator is
rank-local.

## Chunked Prefill

A prompt larger than `max_scheduled_tokens` is split across multiple plans. The
first chunk moves `Submitted -> Prefilling` or `Submitted -> PrefillDone` if the
whole prompt fits. Later chunks move `Prefilling -> Prefilling` until the last
chunk moves the request to `PrefillDone`.

Each emitted prefill operation contains:

- `input_lengths`: tokens scheduled for this chunk per request.
- `input_ids`: concatenated chunk tokens.
- `shifted_input_ids`: next-token targets for prefill logprob and alignment
  paths, padded with `-1` at the prompt boundary.
- `extend_prefix_lens`: how many prompt tokens were already scheduled before
  this chunk.
- `occupied_pages`, `begins`, and `sizes`: all KV pages visible to the request
  plus the subrange newly allocated for this operation.

Example with `max_scheduled_tokens=4` and an 8-token prompt:

| Plan | State before | Operation | State after |
| --- | --- | --- | --- |
| 1 | `Submitted` | Prefill tokens 0-3, `extend_prefix_len=0` | `Prefilling` |
| 2 | `Prefilling` | Prefill tokens 4-7, `extend_prefix_len=4` | `PrefillDone` |
| 3 | `PrefillDone` | Decode bootstrap step | `Decoding` |

## Decode And Reservation

Decode steps allocate enough KV capacity for the scheduler's current decode
reservation. The runtime updates this reservation with
`UpdateReserveNumTokens` events. Normal decoding usually reserves one token.
Speculative decoding can reserve more, so the scheduler may allocate additional
pages before dispatching a decode operation.

For decode operations:

- `num_extends() == 0`.
- `decode_input_ids` carries the bootstrap token when a decode-only worker
  needs one.
- `hist_token_lens` is set when a retracted request is loaded back and resumed.
- `sizes` can be zero when the existing tail page has enough free capacity.

## KV And Prefix Cache

The scheduler works in pages, not raw token slots. Device and host pages are
allocated by page allocators. Per-request local KV allocators hold the tail pages
owned only by that request. Full pages can be inserted into the radix-tree prefix
cache and shared by later requests.

On first prefill scheduling, the scheduler matches the submitted prompt against
the prefix cache:

- Device hits are reused immediately.
- Host hits can trigger load-back into device pages when the host cache tier is
  enabled.
- Missed full pages are allocated locally and later inserted into the prefix
  cache when enough prompt tokens have been scheduled.
- The tail page remains local because it may still be partially filled.

Before allocating pages, the scheduler calls capacity checks that may evict
unlocked prefix-cache nodes. If no safe eviction can provide enough pages, the
candidate is skipped for that planning pass.

## Cache Operations

The native operation model supports cache operations in addition to forward
work.

| Operation | Direction | Why it is emitted |
| --- | --- | --- |
| `LoadBackOp` | Host to device | A request matched host-side prefix pages that must be restored before forward execution. |
| `WriteBackOp` | Device to host | A finished or retracted request has device pages that should become reusable host-cache pages. |
| `PrefetchOp` | Storage to host | Storage-cache metadata indicates useful cached pages should be fetched to host memory. |

The runtime reports completion through cache events. `WriteBackDoneEvent`
releases the pinned device/host tree references and moves `WritingBack` to
`Finished`, or `Retracting` to `Retracted`. `PrefetchDoneEvent` inserts completed
host pages into the prefix cache and moves `Prefetching` to `PrefetchDone`.

## Retraction

If all eligible forward candidates fail because device memory is exhausted, the
scheduler chooses the longest active `Decoding` or `PrefillDone` request as a
victim. It inserts that request's full pages into the prefix cache, allocates
host pages for pages not already present in host cache, and emits a
`WriteBackOp`.

After the write-back completes, the request is `Retracted`. A later plan can
recover it with `ScheduleDecodeFromRetractedEvent`, which reloads missing host
pages, allocates a request-pool slot again, and resumes decode.

Retraction is a pressure-relief path. It avoids failing the server immediately
when a temporary memory spike can be resolved by parking one long request.

## Disaggregated Roles

`role` changes which phases a scheduler instance is allowed to run:

| Role | Behavior |
| --- | --- |
| `Fused` | Prefill and decode run in one scheduler instance. |
| `P` | Prefill-side scheduler. It avoids decode work after prefill. |
| `D` | Decode-side scheduler. It handles decode recovery and bootstrap behavior for remotely prefetched requests. |

The Python runtime wires this from `--disaggregation-mode`. In non-disaggregated
serving, the default fused role is used.

## Invariants

- A request receives a request-pool slot only when it first enters forward
  scheduling.
- Finished requests are erased only during the next planning pass.
- Decode is not scheduled in an iteration that already emitted prefill work.
- Prefix-cache pages are protected by node references while an operation needs
  them and become evictable when those references are released.
- `max_scheduled_tokens` is a per-iteration issue budget, not the total KV pool
  size.
- `max_batch_size` is per scheduler instance; with attention DP it is per DP
  rank.
- Tail pages stay request-local until they become full pages that can be safely
  inserted into the prefix cache.

## Source Map

| Area | Path |
| --- | --- |
| Python event loop | `python/tokenspeed/runtime/engine/event_loop.py` |
| Python scheduler helpers | `python/tokenspeed/runtime/engine/scheduler_utils.py` |
| Native scheduler entry point | `tokenspeed-scheduler/csrc/scheduler/scheduler.cpp` |
| Planning logic | `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp` |
| Cache-operation planning | `tokenspeed-scheduler/csrc/scheduler/operations/cache.cpp` |
| Request FSM | `tokenspeed-scheduler/csrc/fsm/` |
| Python bindings | `tokenspeed-scheduler/bindings/python_module.cpp` |
