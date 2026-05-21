# 04. Performance Model and PoC Plan

## Principle

The expected benefits cannot be added linearly:

```text
Scheduler/KV + parallel strategy + local-SPMD + MLA != total gain by summation
```

Each mechanism only helps when it moves the active bottleneck.

## Workload Assumptions

Target workload:

- DeepSeek V4 / Kimi-like MoE;
- long context with repeated prefix;
- multi-turn agent sessions;
- tool-call / grammar / abort churn;
- short decode steps;
- p95/p99 latency sensitivity;
- attention latent-KV / compressed KV;
- MoE EP/TP and all-to-all pressure;
- deployment target: Ascend 910C for feasibility, 950DT for upper-bound validation.

## Mechanism-to-Counter Matrix

| Mechanism | Expected Counter Movement | User Metric Affected | When It Helps |
|---|---|---|---|
| prefix reuse / finish insert | cached token ratio up, prefill tokens down | TTFT, TPM/GPU | repeated prefix is high |
| tail page ownership | tail page waste down, admission failure down | p95 ITL | short decode and high concurrency |
| dynamic decode reserve | over-reserved pages down | p95 ITL, throughput stability | spec decode / variable accept lengths |
| retract/recovery | queue stall down, recompute down | p95/p99 ITL, queue delay | device KV pressure exists |
| event loop overlap | GPU gap down, CPU commit overlap up | p95 ITL | short decode CPU gap visible |
| cache-aware DP dispatch | pages/rank variance down | queue p95, retraction count | long sessions skew DP ranks |
| parallel-domain strategy | collective bytes/time down | TPM/GPU, TPS/user | attention/dense/MoE optimal splits differ |
| placement compiler | implementation iteration cost down, extra comm bugs down | indirect | many parallel variants must be tested |
| MLA/latent-KV backend | KV bandwidth/storage down | decode throughput | latent-KV dominates memory traffic |

## Expected Benefit Ranges

These are hypotheses to validate, not claims:

| Area | Expected Range | Confidence | Notes |
|---|---:|---:|---|
| Scheduler/KV/EventLoop | 10-25% on p95/p99 in repeated-prefix + KV-pressure traces | medium | Strong code evidence; benefit workload-dependent |
| Agent KV tail/reserve cleanup | 2-8% stability / p95 improvement | medium | More about avoiding tail degradation |
| Cache-aware DP dispatch | 5-15% tail improvement under DP skew | low-medium | Needs DP multi-session trace |
| Parallel strategy | 10-30% depending on baseline equivalence | medium-low | Needs DeepSeek V4 token movement ledger |
| local-SPMD/compiler | 5-15% direct, larger engineering-speed value | low-medium | Only if target path uses compiler effectively |
| MLA/latent-KV | 10-30% where kernel/runtime path dominates | low for moat | vLLM already has TokenSpeed operator support |

## What Not To Claim Yet

1. Do not claim tool-call-specific DDR offload.
2. Do not claim DeepSeek V4 hierarchical KVStore benefit until `DeepseekV4TokenToKVPool` supports it.
3. Do not claim V4 is fully generic placement-compiler based.
4. Do not claim average throughput gain without bottleneck-specific counters.
5. Do not claim MLA kernel is the main moat now that vLLM supports the operator path.

## PoC Success Criteria

Against a well-tuned vLLM-Ascend baseline:

- `TPM/device` improves by 20-30% on target traces;
- `TPS/user` does not regress;
- p95 ITL does not regress, ideally improves;
- p95 TTFT improves on repeated-prefix traces;
- output quality and stop/tool behavior do not regress;
- memory pressure is handled with fewer queue stalls or recompute events;
- MoE communication counters improve or at least explain the same performance with simpler tuning.

## Required Counters

### Scheduler / KV

- device prefix hit pages/tokens;
- host prefix hit pages/tokens;
- cached token ratio;
- prefill tokens saved;
- active KV pages;
- available KV pages;
- cached pages;
- tail page waste;
- request pool occupancy;
- admission failure count;
- retraction count;
- retract victim token length;
- retracted duration;
- recovery loadback bytes;
- recovery latency;
- writeback/loadback op latency;
- TP common cache event lag.

### EventLoop / Runtime

- scheduler iteration time;
- CPU output commit time;
- GPU forward gap between iterations;
- output D2H copy wait;
- overlap ratio;
- wasted decode steps after abort;
- grammar admission queue time;
- idle-forward count under DP.

### Parallel / MoE

- collective time by op type;
- all-reduce / reduce-scatter / all-gather bytes;
- MoE all-to-all bytes and latency;
- routed token count per expert;
- tokens/rank variance;
- expert imbalance;
- DeepEP / dispatcher time;
- placement transition count per layer.

### User Metrics

- TTFT p50/p95/p99;
- ITL p50/p95/p99;
- TPS/user;
- TPM/device;
- queue delay p95/p99;
- abort cleanup latency;
- tool-turn latency if tool-call traces are used.

## Suggested Traces

1. **Repeated-prefix multi-turn trace**
   Same long system/tool/history prefix, small append per turn.

2. **Short decode heavy trace**
   Many sessions generating 1-8 tokens per step.

3. **KV-pressure mixed trace**
   A small number of long sessions plus many short sessions.

4. **DP skew trace**
   Long sessions intentionally concentrated on some DP ranks, then compare queue/cache-aware dispatch.

5. **MoE communication trace**
   Same request shape, varying attention/dense/MoE parallel shape.

6. **Tool-call trace**
   Use tool-call stop tokens, but measure as normal finish/next-turn behavior unless a tool-call-aware offload policy is added.

## Decision Rule

TokenSpeed is worth adapting if the measured gain comes from:

```text
Scheduler/KV ownership
  + agent runtime event-loop overlap
  + better parallel-domain execution plan
```

It is less compelling if the only measurable gain comes from:

```text
a single MLA kernel path already accepted by vLLM
```

