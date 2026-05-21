# TokenSpeed Ascend Agent Runtime Analysis

Last updated: 2026-05-21

This directory collects the current Chinese research notes for evaluating whether TokenSpeed is worth adapting to Ascend 910C + 950DT for DeepSeek V4 / Kimi-like MoE models and agentic workloads.

The material is source-code based. It is not intended to be official TokenSpeed user documentation. It is a structured analysis package for architecture review, PoC planning, and a later technical deck.

## Core Question

TokenSpeed exposes many familiar serving capabilities: prefix cache, chunked prefill, TP/DP/EP-style parallelism, MoE backends, MLA kernels, and PD disaggregation paths. The research question is not whether these features exist.

The real question is:

> Does TokenSpeed encode request lifecycle, KV ownership, parallel-domain transitions, and model-specific execution semantics deeply enough that vLLM / vLLM-Ascend cannot easily copy the result with config flags or small backend patches?

## Current Conclusions

1. **Strongest moat candidate: Scheduler / KV ownership / EventLoop coupling**
   TokenSpeed's C++ scheduler models KV page lifetime as request FSM state. Request states carry `DeviceNodeRef`, `HostNodeRef`, `LocalKVAllocator`, `ReqPoolIndex`, and transfer page pairs. This is stronger than a loose scheduler plus KV cache manager interface.

2. **Agent runtime value is mostly p95/p99-oriented**
   The likely gain is not universal average throughput. It should appear in repeated-prefix, short-decode, high-cancellation, KV-pressure, and DP-cache-skew traces.

3. **No tool-call-specific DDR offload path found**
   Tool use is mostly represented as normal chat template / stop-token behavior. A tool call may end a generation and trigger the normal `FinishEvent` path, but there is no explicit `ToolCallEvent -> offload this request KV to DDR` policy in the analyzed code.

4. **DeepSeek V4 hierarchical KVStore is currently not usable**
   `EventLoop` rejects `enable_kvstore` when `token_to_kv_pool` is `DeepseekV4TokenToKVPool`. Therefore host/DDR writeback/loadback should not be counted as an already-realized DeepSeek V4 benefit.

5. **Parallel strategy is a real candidate, but current evidence is mixed**
   TokenSpeed has split parallelism controls and a `Mapping` object that separates attention/dense/MoE domains. It also has a base placement compiler. However, the current DeepSeek V4 path appears to rely mainly on model-specific `CommManager`, not the generic `CompiledDecoderLayer` path. This needs further code reading before making a strong moat claim.

6. **MLA / latent-KV should be lower weight**
   vLLM already supports TokenSpeed-related MLA kernels, so a single kernel is unlikely to remain a strong moat. The remaining value is end-to-end metadata/layout/runtime integration, not the kernel alone.

## Reading Order

1. [Runtime Architecture](01-runtime-architecture.md)
2. [Agent KV Management](02-agent-kv-management.md)
3. [Parallel Strategy and Placement](03-parallel-strategy-and-placement.md)
4. [Performance Model and PoC Plan](04-performance-model-and-poc.md)
5. [Open Questions and Code Map](05-open-questions-and-code-map.md)

## How This Should Feed a PPT

The deck should not say "TokenSpeed has prefix cache, C++ scheduler, EP, and MLA." That would be shallow and easy to dismiss.

The deck should instead use this narrative:

1. **Request-context runtime view**: where state lives across Python, C++, GPU, DP/TP ranks.
2. **Agent KV lifecycle**: prefix match, page ownership, decode reserve, finish, retract, writeback/loadback.
3. **Parallel execution plan**: attention/dense/MoE can be different parallel domains; explain how hidden/residual placement moves between them.
4. **Copyability analysis**: config-copy vs model-patch vs runtime-protocol vs architecture-level copy.
5. **Performance accounting**: which counters must move before claiming 10-25% or 20-30% gains.

## Repo Snapshot

This analysis was prepared against the local TokenSpeed source tree at commit:

```text
ce376da
```

Key source roots:

- `python/tokenspeed/runtime/engine/`
- `python/tokenspeed/runtime/execution/`
- `python/tokenspeed/runtime/models/`
- `python/tokenspeed/runtime/models/base/`
- `python/tokenspeed/runtime/distributed/`
- `python/tokenspeed/runtime/layers/`
- `tokenspeed-scheduler/csrc/`
- `tokenspeed-mla/`

