# 01. Runtime Architecture

## Correct Mental Model

TokenSpeed should not be modeled as a simple `AsyncLLM -> Scheduler -> ModelExecutor -> Model` chain. The more useful view is a five-plane runtime:

| Plane | Main Components | State Held | Why It Matters |
|---|---|---|---|
| Main Python process | `Engine`, `AsyncLLM`, frontend output processor | user coroutine state, stream collectors, cancellation state | Converts user/API lifecycle into scheduler requests and aborts |
| Optional DP controller | `DataParallelController` | per-DP load budget | Can dispatch by queue size or cache page usage |
| Scheduler worker process | `EventLoop`, `RequestHandler`, `ModelExecutor`, `OutputProcessor`, `MemoryExecutor` | worker runtime state, per-rank orchestration | Bridges Python control flow, C++ scheduler, cache ops, and GPU execution |
| Embedded C++ scheduler | `Request` FSM, `KVPrefixCache`, `PageAllocator`, `ReqPoolAllocator`, `ExecutionPlan` | logical KV page ownership and request lifecycle | Core of Scheduler/KV moat candidate |
| GPU execution plane | `token_to_kv_pool`, `req_to_page`, `InputBuffers`, `ForwardContext`, `ModelRunner` | physical KV tensors, block tables, forward inputs | Materializes scheduler decisions into kernels |

The scheduler does **not** directly own GPU KV tensors. It owns logical page lifetime. `ModelExecutor` converts scheduler `ForwardOp` pages into `req_to_page` and input buffers. Attention kernels read/write the physical `token_to_kv_pool`.

## Request Execution Loop

A generate request flows through the system roughly as:

```text
AsyncLLM
  -> tokenizer / RequestHandler
  -> C++ Scheduler::SubmitRequests
  -> EventLoop iteration
  -> Scheduler::NextExecutionPlan
  -> optional MemoryExecutor CacheOps
  -> ModelExecutor.update_block_table
  -> ModelExecutor.reset_valid_cache_length
  -> ModelExecutor.execute_forward_op
  -> OutputProcessor.post_process_forward_op
  -> scheduler.advance(ExtendResult / Finish / UpdateReserve)
```

Important code paths:

- `python/tokenspeed/runtime/engine/event_loop.py`
- `python/tokenspeed/runtime/engine/request_handler.py`
- `python/tokenspeed/runtime/engine/generation_output_processor.py`
- `python/tokenspeed/runtime/execution/model_executor.py`
- `tokenspeed-scheduler/csrc/scheduler/scheduler.cpp`
- `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp`

## EventLoop Is Not Just Glue

`EventLoop` is a runtime protocol layer. It:

- receives and broadcasts requests across attention-TP ranks;
- registers requests in `OutputProcessor`;
- queries L3 hints when `MemoryExecutor` exists;
- commits cache operation results back to the C++ scheduler;
- obtains `ExecutionPlan`;
- submits cache ops;
- dispatches forward;
- commits output and advances the scheduler;
- synchronizes DP metadata and triggers idle forward when needed.

The overlapping event loop is especially relevant to agentic workloads:

```text
dispatch current forward first
  -> GPU starts current step
  -> CPU commits previous step's results
  -> scheduler.advance happens while GPU is busy
```

This is aimed at short decode steps where CPU postprocess can otherwise create visible GPU gaps.

Relevant code:

- `event_loop.py:event_loop_overlap`
- `event_loop.py:_commit_cache_results`
- `event_loop.py:_dp_sync_and_check`
- `model_executor.py:execute_idle_forward`

## DP Controller and Cache-Aware Dispatch

For attention-DP mode, `DataParallelController` can use:

- `ROUND_ROBIN`
- `SHORTEST_QUEUE`
- `MINIMUM_CACHE_USAGE`

`MINIMUM_CACHE_USAGE` uses worker-reported `num_pages`, derived from C++ scheduler available KV pages. This matters because agent sessions can have very different KV footprints. Request count is not a reliable proxy for memory pressure.

Relevant code:

- `python/tokenspeed/runtime/engine/data_parallel_controller.py`
- `event_loop.py:_get_load`

## DeepSeek V4 KVStore Limitation

The generic architecture has `MemoryExecutor`, host KV pool, storage backend, loadback, writeback, and prefetch. However, the current DeepSeek V4 pool path disables this:

```text
if token_to_kv_pool is DeepseekV4TokenToKVPool and enable_kvstore:
    raise NotImplementedError
```

This is a crucial reporting boundary. For DeepSeek V4 / Kimi-like PoC, do not count host/DDR hierarchical KVStore as already working. Focus first on device-side ownership, prefix reuse, retraction semantics, DP cache-aware dispatch, and parallel communication.

Relevant code:

- `python/tokenspeed/runtime/engine/event_loop.py`
- `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py`

