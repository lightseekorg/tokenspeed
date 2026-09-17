# Gated residual kernels

The Blackwell CuTe mix fuses Down projection, scale/SiLU, Up projection, sigmoid,
and the four-branch mean. It supports positive token counts for contiguous,
16-byte-aligned BF16/FP16 HC4/H2560/R320 inputs, with 320 or 324 projection rows.
Other shapes/layouts/dtypes retain the existing dispatch fallback. The separate
combine/normalization fusion is unchanged.

Default dispatch has no T<=16 restriction: batch size selects a CTA tactic
inside the same CuTe implementation. The runtime mixer already supplies the
required `weights_independent=True` contract, so no override is needed for
larger batches. Blackwell capability, dtype, shape, alignment and resource
safety checks remain in force.

## One parameterized CTA pipeline

Down computes `[P, T] = [P, 10240] @ [T, 10240]^T`; Up computes
`[4F, T] = [4F, 320] @ [T, 320]^T`. Projection columns are independent Down
outputs, but become Up's reduction axis: splitting them requires a handoff.

The kernel parameters are:

| Parameter | Meaning |
| --- | --- |
| `projection_tile` | Down M tile: 64 or 128 columns |
| `token_tile` | Native MMA N tile: 8, 16, 32 or 64 independent tokens |
| `projection_tiles` | Projection tiles computed sequentially by one cluster |
| `split_k` | 1, 2, 4, 8 or 16 K partitions in a cluster |
| `batch_tiles` | Consecutive token microtiles per logical batch job |
| `workers` | Physical CTA groups, each processing one or more jobs |
| `stages` | Down asynchronous pipeline depth |
| `final_tile` | 16 or 32 final hidden columns per Up output tile |

With `C = ceil(P / (projection_tile * projection_tiles))`, the launch is
`grid=(C, split_k, workers)`, `cluster=(1, split_k, 1)`, with 192 threads/CTA.
Each CTA loops over Up tiles `local_cta, local_cta + C*split_k, ...`, so reducing
the CTA group size cannot leave hidden columns unwritten.

A job contains `token_tile * batch_tiles` tokens. Workers loop over jobs with
stride `workers`; the native MMA tile and a CTA's total batch ownership are
different parameters. Increasing `batch_tiles` alone does **not** cache weights
across microtiles. A wider native MMA does reuse weights across more tokens.

For `C > 1`, the full cooperative grid must reside simultaneously. Warmup
queries the **actual loaded function** with 192 threads, its launch dynamic
shared memory, cluster dimensions, and current stream. Worker count is bounded
by `max_active_clusters // C`; the selected compiled grid is checked again.
The conservative legacy platform-availability filter remains in place.

For `C == 1`, each cluster owns the complete projection and waits only for its
own ranks. Cooperative launch is unnecessary: more logical jobs than resident
clusters may execute in ordinary hardware-scheduled waves.

## Current default tactics

These defaults were measured on a 152-SM L20B (SM103); they are not a claim of
optimality on every Blackwell device. Worker counts always use the real
configuration's occupancy query, not a hard-coded SM multiple.

| Tokens | M_proj | B_mma | split-K | Down stages | F |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–8 | 64 | 8 | 16 | 5 | 32 |
| 9–16 | 64 | 16 | 16 | 5 | 32 |
| 17–32 | 64 | 16 | 8 | 5 | 32 |
| 33–96 | 64 | 16 | 4 | 5 | 32 |
| 97–192 | 64 | 32 | 4 | 5 | 32 |
| >192 | 128 | 32 | 4 | 4 | 32 |

Default `projection_tiles=batch_tiles=1`; the same kernel also implements and
tests larger jobs and complete-projection cluster loops. On the measured GPU,
T=1024/P324 uses 3 projection clusters × 4 ranks × 12 workers = 144 physical
CTAs. Those workers process 32 logical 32-token tiles in three rounds.

The T≤16 specialization preserves the original warp execution order, tile
sizes, 227-KiB launch allocation, and arithmetic. Wider tiles reuse Down/Up
weight storage only after `down_done`, and reuse the reduction mailbox for
gate values only after reduction readers finish. Tile and pipeline depth are
selected together. Unsplit and smaller split counts were measured before
selecting the large-batch split.

## Ordering and ownership

Each microtile performs Down, fixed-rank ordered DSM reduction, scale/SiLU,
activation publication, Up tile loops, and gate/output stores. There are no
floating-point atomics. Up stage reuse waits for its previous MMA reader;
TMEM reuse waits for all epilogue readers. Projection-loop barriers prevent DSM
mailbox reuse while another rank still accesses it.

Scratch storage is keyed by device, capture/warmup stream, projection rows,
worker count, projection-cluster count and slot height. Its 16-bit activation
allocation can serve ordered BF16/FP16 calls. The epoch array contains two
halves: `READY[worker,cluster]` and `CONSUMED[worker,cluster]`. Generation
selection runs on the device. Before a worker reuses an activation slot, every
CTA has finished its epilogue and every projection cluster has published and
acquired the consumed generation. A ready flag alone is insufficient.

Every graph variant must be warmed on its capture stream. Cold capture fails
rather than compiling, querying occupancy or allocating persistent scratch.
Graphs sharing a workspace must be replayed in order; use separate capture
streams for concurrently replayed graphs. Eager and graph execution use the
same plan and kernel, and CUDA graphs retain their capture-time PDL setting.
The independent-weight promise is unchanged. PDL consumers still wait for the
producer's completion before consuming its data; an early launch notification
is not a data-readiness signal. See NVIDIA's
[PDL ordering contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

## Validation and reproduction

The NVIDIA batched tests cover tile/worker boundaries through T=1024, both
dtypes, optional inject, PDL on/off, changed-input graph replay, deterministic
outputs, large epochs, interleaved streams, projection/batch loops, and normal
multi-wave cluster scheduling. Existing HC and cold-capture tests remain
applicable. FP64 reference tolerances remain BF16 0.04 and FP16 0.008.

Run in the repository's Python environment:

```bash
pytest tokenspeed-kernel/test/ops/test_hyperconnection.py \
  tokenspeed-kernel/test/nvidia/ops/test_hyperconnection_capture.py \
  tokenspeed-kernel/test/nvidia/ops/test_hyperconnection_batched.py
```

For performance, save the pre-change `thirdparty/cute_dsl/hc_fused.py` and use
the paired benchmark (repeat for FP16):

```bash
python tokenspeed-kernel/benchmarks/bench_hyperconnection_mix.py \
  --baseline-file /path/to/original_hc_fused.py \
  --rows 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 32 128 1024 \
  --dtype bf16 --projection-rows 324 --pdl on \
  --captures 5 --calls 128 --repeats 31
```

This times the entire GPU operator in warmed CUDA graphs, excluding host
allocation and JIT work. It alternates capture and A/B timing order, rebuilds
both workspaces with each fresh capture stream, retains per-capture medians,
and checks bitwise small-batch outputs. T>16 compares with
the unchanged portable implementation. One graph capture can show placement/
timing variation; do not interpret a single capture as a regression proof.

## Measured validation

Measurements used L20B/SM103, PyTorch 2.11.0+cu130, CUTLASS DSL 4.7.1 and
tokenspeed-triton 3.8.10.post20260709. The installed CUDA toolkit is 13.0;
the DSL's generated PTX identifies its bundled compiler as NVVM CUDA 13.3,
PTX 9.3, target `sm_103a`.

For every T=1 through 16, both BF16 and FP16 with P324 and PDL enabled, five
fresh graph captures each used 128 calls and 31 alternating paired measurements.
The final sweep also alternated capture order and recreated both workspaces.
All small-batch outputs were bitwise equal to the original CuTe kernel.
Changes below are geometric means of candidate/baseline latency ratios;
positive means slower.

| Small-batch dtype | Across all 16 shapes | Largest per-shape increase |
| --- | ---: | ---: |
| BF16 | -0.29% | +1.03% |
| FP16 | -0.36% | +0.63% |

These results show no systematic small-batch regression on this GPU. Individual
captures varied by about +/-4%, sometimes reversing on recapture; an exact
zero-regression guarantee is not implied.

An earlier independent five-capture sweep with fixed capture order also found
no systematic regression: -0.09%/-0.30% overall and +1.51%/+1.44% maximum
per-shape increases for BF16/FP16, respectively.

An additional T=16/FP16/P320/PDL-off check used 15 captures and measured +0.26%,
with bitwise-equal outputs. An initial asymmetric benchmark reused the old
workspace but recreated the new one, always capturing the baseline first;
that measured +2.4%. Recreating both workspaces and alternating capture order
exposed the same approximately 6.6/7.1-us timing modes in both implementations.
The reproduction benchmark includes this control; allocation/capture placement
is a plausible source of the variation, not an established kernel bottleneck.

The staged evaluator promoted the batched candidate through L2; its final
release correctness checks passed. Representative warmed-graph whole-operator
latencies (microseconds, BF16/P324, scale 1, PDL enabled) were:

| T | Previous portable fallback | Fused CuTe | Speedup |
| --- | ---: | ---: | ---: |
| 17 | 29.00 | 8.43 | 3.44x |
| 32 | 30.61 | 8.38 | 3.65x |
| 64 | 37.03 | 13.19 | 2.81x |
| 128 | 43.02 | 14.03 | 3.07x |
| 256 | 47.11 | 21.34 | 2.21x |
| 1024 | 77.73 | 67.19 | 1.16x |

At T=1024 the loaded function used 222208 bytes of dynamic shared memory,
168 registers/thread, zero local bytes, one CTA/SM and up to 36 resident
four-CTA clusters. With the same M128/N32/four-stage layout, split-K 1 and 2
measured approximately 87.07 and 89.89 microseconds, respectively: fewer splits
alone did not improve latency. Increasing native MMA N and changing shared
storage lifetimes were necessary to make the larger tile practical.

Nsight Systems cross-checks used 1280 kernel instances per capture. T=1
old/new min/average/max were 5.632/5.772/8.224 and 5.664/5.866/6.592 us;
T=16 were 5.984/6.200/6.912 and 5.984/6.248/6.720 us. These instrumented
durations are separate from the uninstrumented paired timing above.

The 75 existing HC/capture tests and 80 added batched/configuration cases passed.
Default-dispatch tests cover T=17 through 1024, both dtypes and both PDL settings
without an override, checking that CuTe is selected and matches the FP64 oracle.
The configuration checks ensure that a wider native MMA tile cannot inherit
the legacy small-tile shared-memory allocation override. T=256 changed-input
graph replay also passed device memcheck with zero errors and
racecheck with zero hazards. CUDA API error reporting was disabled for the
memcheck run because optional CUDA-binding symbol probes reported
`cuGetProcAddress_v2` errors; device memory diagnostics were not suppressed.

Validation loaded the real kernel and public residual dispatch in an isolated
package namespace: unrelated optional CUDA-12 extension binaries cannot load
in this CUDA-13 environment. No gated-residual math was mocked. This is kernel
and dispatch validation, not a full runtime/model end-to-end claim.
