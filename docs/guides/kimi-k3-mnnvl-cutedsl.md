# Opt-in K3 MNNVL prefill fusion

This page covers the runtime-integrated first stage. The same PR also contains
the experimental [fused shared-RS + up-projection + AG](./kimi-k3-fused-rs-up-projection-ag.md)
back half, with separate qualification evidence and no model runtime dispatch.

This path combines routed-expert finalize, the **first** AllReduce, and RMSNorm
for TP8 K3 prefill. It is disabled by default. It does not replace the sharded
up-projection, shared/residual arithmetic, second AllReduce, or final clone.

![Baseline and opt-in prefill paths, including native H3584 geometry](/images/k3-mnnvl-prefill-path.png)

## Dispatch and ownership

Enable with `TOKENSPEED_K3_MNNVL_CUTEDSL=1`. Qualification requires TP8/EP1,
attention DP=CP=1, a sharded up-projection, deferred expert finalize support,
BF16 routed width 3584, top-k 16, compatible MNNVL fabric, and FlashInfer
0.6.18 with PDL on data-center Blackwell (SM 10.0–10.3).

| Protocol | Qualified captured token counts | Implementation |
| --- | --- | --- |
| BT | 256, 384, 512, 768, 1024 | FlashInfer balanced-tree finalize/reduce/RMS stages |
| HT | 1280, 2048, 4096, 6144, 8192 | Native-H3584 persistent CuTe DSL specialization |
| Existing route | Decode, eager prefill, other counts or unsupported capabilities | Unchanged fallback |

These are exact buckets, not permission to interpolate between them. Runtime
also checks the allocated workspace capacity. The default prefill graph
capacity is 2048; qualifying an HT kernel at M4096/8192 does not automatically
capture or exercise those buckets in a server.

A separate, exception-safe `prefill_graph_phase` selects this path during
breakable-prefill warmup/capture. It intentionally does not enable full decode
graph stream forks across eager attention breaks. The existing unified decode
metadata contract is unchanged.

The runtime calls only `tokenspeed-kernel`. Its registered communication
operation owns validation and workspace lifetime; an optional, lazily imported
FlashInfer adapter owns backend construction. The native specialization and
upstream provenance are documented in the kernel's `mnnvl_k3_ht/README.md`.

Allocation, compilation, rank agreement, and fabric rendezvous happen before
capture. Outputs alias persistent workspace and must be consumed before the
next invocation; calls sharing a workspace are ordered on one stream.
Concurrent or independently replayed graphs need independent workspaces.
Configuration mismatches are rejected before launching the collective.

## Why native H3584 matters

A BF16x8 vector covers 16 bytes. A 3584-column row has 448 vectors, or 56 per
TP-owned reduction shard. Two reduction warps provide 64 lanes: only lanes
0–55 issue loads/stores. Lanes 56–63 are predicated off, including optional
residual loads. This avoids padding to H4096, copying padded tensors, or
rescaling RMSNorm.

The qualified HT tuning uses 448 consumer threads, one vector per thread,
seven producer stages, two reduction warps, two RMS token groups, three RMS
pipeline stages, and PDL. The synchronization protocol is inherited; this is
not a new synchronization algorithm or an RSAG/up-projection fusion.

## Performance and its limits

![Historical complete-tail and expert-plus-tail p50 latency; not serving TTFT](/images/k3-mnnvl-prefill-performance.png)

Historical qualification on 8 GB300 GPUs, TP8/EP1, PyTorch 2.13,
FlashInfer 0.6.18, cuDNN 92000, CUTLASS DSL 4.7.1. Each measurement is an
unprofiled CUDA Graph, reduced by MAX across ranks **before** calculating
percentiles. Baseline and candidate alternate over 15 paired rounds.
These are prior qualification results, not a new measurement of a repackaged
commit. Ratios below use the original unrounded measurements.

| Boundary | M | Baseline p50 / p90 (µs) | Candidate p50 / p90 (µs) | p50 / p90 speedup |
| --- | ---: | ---: | ---: | ---: |
| Complete tail | 4096 | 412.35 / 412.68 | 346.21 / 346.47 | 1.1911× / 1.1911× |
| Complete tail | 8192 | 785.64 / 786.28 | 654.11 / 654.66 | 1.2011× / 1.2010× |
| NVFP4 SiTU expert MoE + tail | 4096 | 845.17 / 845.53 | 779.70 / 782.79 | 1.0840× / 1.0801× |
| NVFP4 SiTU expert MoE + tail | 8192 | 1682.16 / 1683.02 | 1659.32 / 1659.63 | 1.0138× / 1.0141× |

Complete tail starts from deferred expert outputs and includes finalize,
both collective stages, RMSNorm, sharded up-projection, shared/prefix
injection, and output clone. Expert GEMMs and shared-expert GEMMs are outside.
Four independent layer slots are measured per graph.

The SiTU benchmark also includes the registered NVFP4 expert GEMMs and uses
two independent slots, synthetic packed weights and balanced top-k routing
over 896 experts. Shared-expert GEMMs are still outside. Its finalized and
deferred expert paths have their own applicable tactics; it measures the
integration, not an isolated communication-kernel substitution.

**Serving TTFT is mixed, not a stable large-M qualification.** The available
order-balanced serving sweep observed live M512 BT and M2048 HT buckets.
M4096 smoke used eager fallback. It therefore cannot establish M4096/8192 HT
serving benefits.

TTFT here is the client-observed arithmetic mean per concurrency point, not
tail latency or a TTFT percentile.

| Concurrency | Mean TTFT speedup, A/B order | Mean TTFT speedup, B/A order | Geometric mean of orders |
| --- | ---: | ---: | ---: |
| 1 | 0.9933× | 0.9749× | 0.9841× |
| 2 | 1.1650× | 0.9855× | 1.0715× |
| 4 | 1.0091× | 1.0538× | 1.0312× |
| 8 | 0.9703× | 0.9881× | 0.9791× |
| 16 | 1.0252× | 1.0142× | 1.0197× |

The across-concurrency geometric mean is 1.0166× TTFT, 1.0083× TPS/user,
and 1.0079× TPS/GPU. Both arms use the same qualified tactic table, TP8/EP1,
native EAGLE3 (three steps, four draft tokens, top-k one), FP8 KV, and the
same workload. Each arm completes all 786 requests, with 500 output tokens.
One run per order is not a repeated statistical qualification. Do not use the
aggregate to hide the regressions at concurrency 1 and 8 or label tail
speedups as TTFT speedups.

The figure's public summary data is in
[/data/k3-mnnvl-prefill-performance.json](/data/k3-mnnvl-prefill-performance.json).
The corresponding 15-round samples are provided for
[complete tail](/data/ht-complete-tail-samples.json) and
[expert MoE + tail](/data/ht-situ-tail-samples.json). These exports retain
measurement fields and omit environment-specific paths and identifiers.

## Correctness contract

The independent oracle preserves the BF16 local-finalize boundary, then
performs a fixed-order FP32 rank reduction, BF16 conversion, FP32 RMSNorm,
and BF16 output conversion. BT is checked within two BF16 ULPs; HT within
three. This does not claim bitwise identity between different collective
trees. Full-tail absolute, relative and L2 gates are additionally enforced.

Tests cover random inputs, identity gamma, invalid `-1` routes, ranks without
local routed rows, negative/zero inputs, changed-input replay, mixed BT/HT
layers, independent graph instances, output aliases and unwritten tails.
Prefill tests cover exception cleanup and an eager attention break followed
by the HT tail. The distributed protocol suite reaches M2048; large-M
end-to-end checks are also performed by the benchmark scripts.

## Reproduction

Use an installed TokenSpeed development environment with the versions above.
Run from the repository root. No checkpoint is needed for these synthetic
benchmarks. Set `MASTER_ADDR` to the rendezvous host, `MASTER_PORT` to an
available port, and `NODE_RANK` to 0 or 1. Run the following commands on both
four-GPU nodes in the same MNNVL domain:

```bash
export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python"
export REQUIRE_MNNVL_CUTEDSL=1

torchrun --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m pytest -q \
  tokenspeed-kernel/test/nvidia/ops/communication/test_mnnvl_cutedsl_finalize_distributed.py \
  tokenspeed-kernel/test/nvidia/thirdparty/test_k3_mnnvl_breakable_prefill_graph.py

BENCH_PROTOCOL=ht BENCH_TOKENS=1280,2048,4096,6144,8192 \
BENCH_GRAPH_LAYERS=4 BENCH_WARMUP=10 BENCH_ITERS=15 BENCH_ROUNDS=15 \
BENCH_MIN_QUALIFIED_SPEEDUP=1.03 BENCH_GATE_P90=1 \
BENCH_OUTPUT=ht-complete-tail.json \
torchrun --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  tokenspeed-kernel/test/nvidia/thirdparty/bench_k3_mnnvl_cutedsl_deferred_full_tail.py

BENCH_PROTOCOL=ht BENCH_TOKENS=1280,2048,4096,6144,8192 \
BENCH_GRAPH_LAYERS=2 BENCH_WARMUP=5 BENCH_ITERS=10 BENCH_ROUNDS=15 \
BENCH_MIN_SPEEDUP=1.0 BENCH_GATE_P90=1 BENCH_OUTPUT=ht-situ-tail.json \
torchrun --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  tokenspeed-kernel/test/nvidia/thirdparty/bench_k3_mnnvl_cutedsl_situ_moe_tail.py
```

For BT use `BENCH_PROTOCOL=bt`, tokens `256,384,512,768,1024`, and
20 timed replays for complete tail. Keep the other boundaries and tolerances
unchanged. JSON outputs contain per-round rank-MAX samples, tuning and
correctness results. Preserve them with the tested source revision and
environment manifest when qualifying a new commit.

Local integration tests in the full development environment:

For a device-independent packaging check, run
`python -m pytest -q test/runtime/test_k3_mnnvl_public_contract.py`.
It tests selected pure policy definitions, provenance and sample/summary
consistency; it does not import the GPU runtime or replace the tests below.

```bash
python -m pytest -q \
  test/runtime/test_k3_moe_tail_tier.py \
  test/runtime/test_prefill_graph_phase_isolation.py \
  tokenspeed-kernel/test/nvidia/ops/communication/test_mnnvl_cutedsl_finalize_boundary.py \
  tokenspeed-kernel/test/nvidia/test_moe_tactic_sweep.py
pre-commit run --all-files
```

Before enabling this by default, repeat large-M serving A/B qualification with
observed M4096/8192 HT dispatch and report per-concurrency TTFT p50/p90 and
paired confidence intervals. Kernel eligibility alone is not that evidence.
