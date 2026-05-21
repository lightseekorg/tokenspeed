# 05. Open Questions and Code Map

## Code Areas Already Read

### Runtime architecture

- `python/tokenspeed/runtime/engine/event_loop.py`
- `python/tokenspeed/runtime/engine/request_handler.py`
- `python/tokenspeed/runtime/engine/generation_output_processor.py`
- `python/tokenspeed/runtime/engine/output_processor.py`
- `python/tokenspeed/runtime/engine/data_parallel_controller.py`

Findings:

- `EventLoop` is the main Python runtime protocol.
- Cache results and forward results both advance the C++ scheduler.
- DP ranks all-gather CPU metadata before GPU forward.
- Idle forward exists so zero-token ranks still participate in collectives.
- Abort and grammar admission affect resource cleanup and wasted decode steps.

### C++ Scheduler / KV

- `tokenspeed-scheduler/csrc/scheduler/scheduler.cpp`
- `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp`
- `tokenspeed-scheduler/csrc/scheduler/operations/cache.cpp`
- `tokenspeed-scheduler/csrc/scheduler/outside_event_handler.cpp`
- `tokenspeed-scheduler/csrc/fsm/forward_states.h`
- `tokenspeed-scheduler/csrc/fsm/forward_events.cpp`
- `tokenspeed-scheduler/csrc/resource/allocator/owned_pages.*`
- `tokenspeed-scheduler/csrc/resource/allocator/kv_allocator.*`
- `tokenspeed-scheduler/csrc/resource/page_container.*`
- `tokenspeed-scheduler/csrc/resource/kv_prefix_cache/kv_prefix_cache.cpp`
- `tokenspeed-scheduler/csrc/resource/radix_tree/tree_resource.h`

Findings:

- FSM state owns resources.
- Prefix tree receives ownership from request-local pages.
- Finish and retract both can produce writeback.
- Retracted requests can recover through loadback and `hist_token_lens`.
- `ExecutionPlan` combines forward and cache operations.

### GPU materialization

- `python/tokenspeed/runtime/execution/model_executor.py`
- `python/tokenspeed/runtime/execution/input_buffer.py`
- `python/tokenspeed/runtime/execution/forward_batch_info.py`

Findings:

- Scheduler pages are materialized into `req_to_page`.
- Retraction recovery modifies `valid_cache_lengths`.
- Fully prefix-cached prefill can produce zero model tokens.
- DP metadata enters `ForwardContext`.

### Parallel infrastructure

- `python/tokenspeed/runtime/distributed/mapping.py`
- `python/tokenspeed/runtime/distributed/comm_manager.py`
- `python/tokenspeed/runtime/distributed/comm_ops.py`
- `python/tokenspeed/runtime/models/base/placement.py`
- `python/tokenspeed/runtime/models/base/compiler.py`
- `python/tokenspeed/runtime/models/base/comm_ops.py`
- `python/tokenspeed/runtime/models/deepseek_v4.py`

Findings:

- Split parallel domains exist conceptually.
- Base placement compiler exists.
- DeepSeek V4 path appears to use explicit `CommManager` more than generic compiler.

### MLA / latent-KV

- `tokenspeed-mla/`
- `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`
- `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`
- `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py`

Findings:

- MLA/latent-KV implementation matters for runtime layout and metadata.
- Single-kernel moat is weaker because vLLM has already adopted TokenSpeed-related operator support.

## Important Negative Findings

### No tool-call-specific KV offload found

No analyzed path maps semantic tool calls to host/DDR offload. Current behavior is normal generation finish/stop behavior plus generic finish/retract KV lifecycle.

### DeepSeek V4 KVStore is disabled

`EventLoop` raises `NotImplementedError` when DeepSeek V4 pool is used with `enable_kvstore`. Host/DDR KVStore should be treated as generic architecture, not current V4 evidence.

### local-SPMD is not proven as V4 main path

The base compiler is real, but current DeepSeek V4 evidence points to `CommManager`. The final report must separate:

- generic capability;
- model-specific actual path;
- PoC assumptions.

## Open Questions

### P0: DeepSeek V4 parallel execution ledger

Need to answer:

- What are the exact attention/dense/MoE placements per layer?
- Where are all-gather, reduce-scatter, all-reduce, and all-to-all invoked?
- Which communication ops are token-aware?
- How does `CommManager` select groups?
- How do shared experts and routed experts interact with TP/EP/DP?
- Are there hard constraints such as MoE TP and EP not both greater than one?

### P0: MoE backend and dispatch

Need to read:

- `python/tokenspeed/runtime/layers/moe/`
- `python/tokenspeed/runtime/layers/moe/dispatcher/`
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/`
- `python/tokenspeed/runtime/moe/expert_location.py`
- `python/tokenspeed/runtime/moe/eplb_algorithms/`

Need to produce:

```text
token routing -> expert location -> dispatch -> expert compute -> combine -> communication bytes
```

### P1: MemoryExecutor / HostExecutor real overlap

Need to read:

- `python/tokenspeed/runtime/cache/executor/memory_executor.py`
- `python/tokenspeed/runtime/cache/executor/host_executor.py`
- `python/tokenspeed/runtime/cache/executor/storage_executor.py`
- `python/tokenspeed/runtime/cache/kv_cache_host.py`

Need to clarify:

- actual stream/event ordering;
- writeback/loadback granularity;
- layerwise loadback consumer semantics;
- whether/how this maps to Ascend DDR/HBM/HCCL/ACL memory paths.

### P1: Ascend portability

Need to classify each mechanism:

| Mechanism | Ascend Portability Question |
|---|---|
| C++ scheduler FSM | Mostly portable, but bindings/build integration needed |
| `MemoryExecutor` | Needs Ascend DDR/HBM copy implementation |
| EventLoop overlap | Portable concept, stream primitives differ |
| DP idle forward | HCCL collective compatibility required |
| MoE backend | Major porting risk |
| MLA kernels | CUDA binary not portable; layout ideas may port |
| placement compiler | Portable concept; comm ops need HCCL backend |

### P2: Deck conversion

The deck should be rebuilt from these docs, not from the older shallow slide script.

Recommended chapter order:

1. Research question and target workload.
2. Corrected TokenSpeed runtime architecture.
3. Agent Runtime and KV ownership deep dive.
4. Parallel strategy deep dive.
5. local-SPMD / placement compiler as implementation mechanism.
6. MLA/latent-KV as lower-weight implementation note.
7. Performance model and PoC win lines.

## Maintainer Notes

When updating this analysis:

- Keep claims tied to source paths.
- Separate "implemented in current V4 path" from "generic architecture exists."
- Mark negative findings explicitly.
- Do not convert source evidence into generic feature lists.
- Add counters whenever claiming performance impact.

