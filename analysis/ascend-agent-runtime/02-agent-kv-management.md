# 02. Agent KV Management

## Why Agent KV Is Different

Agentic workloads are not merely long-context workloads. They are multi-turn, short-decode, high-churn workloads:

```text
shared system prompt / tool schema / history prefix
  -> short generation
  -> tool call or intermediate reasoning
  -> tool result appended
  -> another short generation
  -> possible abort, timeout, grammar failure, branch pruning
```

This creates KV-specific pressure:

- repeated prefix should not be re-prefilled;
- short decode amplifies tail page waste and CPU/GPU scheduling gaps;
- long sessions occupy device KV while short requests need admission;
- cancellation must release slots quickly;
- async writeback/loadback must not race with new writes;
- DP ranks can have skewed KV footprints.

## Core Finding

TokenSpeed's KV value is not "it has a prefix cache." The value is:

```text
KV page lifetime is encoded in C++ request FSM states.
```

Request states carry resources. A transition moves ownership, not just a status enum.

Important states:

- `Submitted`
- `Prefilling`
- `PrefillDone`
- `Decoding`
- `Draining`
- `WritingBack`
- `Retracting`
- `Retracted`
- `Finished`

Relevant code:

- `tokenspeed-scheduler/csrc/fsm/forward_states.h`
- `tokenspeed-scheduler/csrc/fsm/forward_events.cpp`

## Ownership Building Blocks

### `OwnedPages`

`OwnedPages` is a move-only RAII wrapper around page ids:

- cannot be copied;
- deallocates pages on destruction if still owned;
- can split with `TakeFirst` / `TakeLast`;
- can merge with `Append`;
- can detach without freeing.

This protects complex paths from early-free, double-free, and dangling-page mistakes.

Relevant code:

- `tokenspeed-scheduler/csrc/resource/allocator/owned_pages.h`
- `tokenspeed-scheduler/csrc/resource/allocator/owned_pages.cpp`

### Request-local pages vs prefix-tree pages

TokenSpeed separates:

- pages already owned by radix-tree prefix nodes;
- request-local pages held by `LocalKVAllocator`, especially tail pages.

`PageContainer` merges both when building the page list for forward.

Relevant code:

- `tokenspeed-scheduler/csrc/resource/page_container.cpp`
- `tokenspeed-scheduler/csrc/resource/allocator/kv_allocator.cpp`
- `tokenspeed-scheduler/csrc/resource/kv_prefix_cache/kv_prefix_cache.cpp`

This matters for short decode. A partially filled tail page should remain request-local. Full pages can move into the prefix tree and become reusable/evictable.

## Prefix Reuse and LoadBack

During first prefill scheduling, TokenSpeed matches full paged tokens against both device and host sides:

```text
device_matched = match_result.device.DepthInPage()
host_matched   = match_result.host.DepthInPage()
loadback_diff  = host nodes without device pages
```

If host depth is deeper than device depth, scheduler can allocate device resources for host-only nodes and generate a `LoadBackOperation`.

Relevant code:

- `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp:schedulePrefillFirstChunk`
- `tokenspeed-scheduler/csrc/fsm/forward_events.cpp:SchedulePrefillFirstChunkEvent`

For DeepSeek V4, this host path is currently blocked by the `enable_kvstore` limitation. The architectural mechanism exists, but should not be counted as current V4 gain.

## Decode Reserve

Decode is not just "append one token." Scheduler checks tail-page capacity before scheduling:

```text
tail_available = request->TailPageAvailableTokens()
extra_tokens = reserve_next - tail_available
pages_needed = ceil(extra_tokens / page_size)
```

After model execution, Python `OutputProcessor` sends `UpdateReserveNumTokens` based on accepted output length. That feeds the next scheduler decision.

Relevant code:

- `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp:scheduleDecode`
- `python/tokenspeed/runtime/engine/generation_output_processor.py`
- `python/tokenspeed/runtime/engine/scheduler_utils.py`

This is useful for short decode and speculative decode: avoid static over-reservation while still failing admission before forward rather than inside kernels.

## Finish Path

A finished request is not simply freed. `FinishEvent`:

1. inserts request-local full pages into the device prefix tree;
2. optionally allocates host resources if host depth is behind device depth;
3. enters `Draining`;
4. later emits `WriteBackOperation`;
5. transitions `WritingBack -> Finished` after copy completion.

Relevant code:

- `tokenspeed-scheduler/csrc/fsm/forward_events.cpp:FinishEvent`
- `tokenspeed-scheduler/csrc/scheduler/scheduler.cpp:newWriteBackOperation`
- `tokenspeed-scheduler/csrc/scheduler/outside_event_handler.cpp`

For multi-turn agents, this is important because "current response" becomes "next-turn prefix." If KV survives as prefix-tree-owned pages, next turn can avoid repeated prefill.

## Retract / Recovery

When device KV pressure prevents all active decode requests from being scheduled, TokenSpeed chooses the longest active request for retraction:

```text
ops.empty()
  -> choose max TokenSize among Decoding / PrefillDone
  -> newRetractOperation(victim)
```

Retract does not discard the request. It:

- inserts completed request-local pages into device prefix cache;
- allocates host pages for device-only nodes;
- captures concrete `(device_page, host_page)` pairs;
- emits writeback;
- transitions to `Retracted`;
- later recovers via host/device match and loadback;
- restores `valid_cache_lengths` using `hist_token_lens`.

Relevant code:

- `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp:scheduleRetract`
- `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp:scheduleDecodeFromRetracted`
- `tokenspeed-scheduler/csrc/fsm/forward_events.cpp:ScheduleRetractEvent`
- `python/tokenspeed/runtime/execution/model_executor.py:reset_valid_cache_length`

This is the strongest agent-KV mechanism found so far. It is p95/p99-oriented: it matters under KV pressure and long-session/short-request mixing.

## Tool Call Boundary

No explicit tool-call-aware KV offload path was found.

What exists:

- tokenizer/chat-template code can insert `tools` into the conversation;
- Llama-style `<|eom_id|>` can be treated as an additional stop token;
- a tool-call generation can end like any other generation and trigger normal finish;
- normal finish may write back if kvstore is enabled and supported;
- retract may write back if KV pressure occurs.

What was not found:

```text
ToolCallEvent -> write this request's KV to DDR/host
```

Therefore the correct claim is:

> TokenSpeed has generic finish/retract/writeback/recovery primitives that could support an agent-turn-boundary offload policy, but the analyzed code does not implement a tool-call-specific trigger.

Relevant code:

- `python/tokenspeed/runtime/utils/hf_transformers_utils.py`
- `tokenspeed-scheduler/csrc/fsm/forward_events.cpp`
- `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp`

## Copyability

| Mechanism | vLLM/vLLM-Ascend Copy Difficulty | Why |
|---|---:|---|
| prefix hash/cache | medium | Common serving capability |
| page allocator | medium | Engineering work, not architecture by itself |
| request-local tail page ownership | medium-high | Requires scheduler/KV lifecycle changes |
| dynamic decode reserve | medium-high | Needs output feedback into scheduling |
| finish-to-prefix ownership transfer | high | Must define safe page ownership after request ends |
| retract/recovery | high | Needs request FSM, host/device transfer, valid-length recovery |
| cache + forward in one plan | high | Runtime protocol change |

