# Kimi-K3 NVFP4 latent input fusion

## Scope and status

Automatic preparation of the NVFP4 routed-expert input after K3's
column-sharded latent down projection. The default policy is `auto`: eligible
configurations use fusion without an environment override. Unsupported
configurations and widths retain the BF16 mailbox/gather followed by FlashInfer
input quantization. `TOKENSPEED_K3_DOWN_NVFP4_FUSION=off` explicitly restores
that original path for every width.

This branch changes the default. Fresh full-model acceptance and node-level
profiling remain required before merging; the historical results below do not
certify the current default policy. This document describes the implementation
contract, not a measured performance claim.

The validated serving target is one eight-GPU Blackwell TP8 attention / TP8 MoE
replica with the NVFP4 SiTU FlashInfer TRT-LLM backend. Runtime eligibility
requires an eight-rank MoE TP group, EP=1, and the existing multicast producer.
EP-routed configurations (including DEP8/TEP8), other MoE TP widths, other
quantization recipes and other expert backends retain their existing input
path. Additional data-parallel replicas or hybrid attention mappings are not
validated by this experiment. This work does not change routed up-projection,
shared experts, routing, sampling or EAGLE3 settings.

## Ready-group module validation

The integrated ready-group policy passed 108 GPU unit cases, including PDL
off/on, partial warps, scale bytes, live-row reset and 1000 graph replays.
An eight-rank GB300 module check covered 57 widths, including every integer
M=1–8 and M=96–128, the 95/96/97 boundary, and mailbox capacity 1279/1280.
Repeated four-input/two-mailbox reuse and individually delayed peers passed.
The initialized op handled every tested width without further grouped-kernel
compilation. M=0 and widths above 1280 remain outside auto's eligibility.

Against the previous auto policy, every M=96–128 improved in both measured
graph organizations. With 64 projections per graph, full projection-through-
quantization time fell 2.8–7.3%. Timing used six alternating rounds, taking
the slowest rank per round and then the median. These are module results,
not new full-model TPS/User or TPS/GPU measurements.

## Previous-policy validation summary

The serving results below validate the previous auto policy: cooperative
separate at M=1–8, cooperative fused at M=9–128, and scalar separate at
M=129–1280. They do not establish end-to-end gains for the ready-group policy
described below. Fresh full-model acceptance of that default policy is pending.

The implementation was compared with unmodified main
`eaf66b5be72ca1e9aed8b0a4c23165f4be072a2d` on 8xGB300, using the main K3
agentic TP8 EAGLE3 configuration, FP8 KV cache, and an 8192-token prefill
budget/chunk/graph maximum. Both arms used the same runtime, dataset and
deployment overrides. The runtime included FlashInfer 0.6.18 and
PyTorch 2.13.0+cu130. Numerical gates covered PDL off/on, dispatch boundaries,
delayed peers and 1000 graph replays. Loaded-block checks included the
8192-row fallback. All 4716 timed turns across three paired clean boots
completed successfully with 500 output tokens each.

Changes below are geometric means of paired candidate/main ratios:

| CC | Raw TPS/User | Raw TPS/GPU | Acceptance-normalized TPS/User |
| --- | ---: | ---: | ---: |
| 1 | -0.94% | -0.74% | -0.20% |
| 2 | +0.67% | -0.89% | +1.01% |
| 4 | +2.66% | +1.40% | +1.79% |
| 8 | +0.16% | -0.23% | +1.16% |
| 16 | -0.91% | -1.09% | -0.17% |

Normalization applies `TPS/User * 3.5 / Tok/Iter` to each run before paired
aggregation. It is a first-order sensitivity analysis, not a measurement with
acceptance held fixed: iteration cost need not remain constant. Across CCs,
raw TPS/User changes by +0.32%, raw TPS/GPU by -0.31%, and normalized TPS/User
by +0.72% (descriptive 95% paired interval: -0.64% to +2.09%). These results
do not establish an overall serving throughput improvement. The small
normalized CC1/CC16 regressions were accepted without per-shape exceptions
for that previous opt-in policy. First/second-turn node-level profiling and
fresh full-model acceptance of the current default policy remain merge gates.

## Dataflow

The input GEMM continues to round to BF16. Eight ranks own disjoint 448-column
shards of the 3584-column routed latent. The receiving expert's processed
`w13_input_scale_quant` is the encoding multiplier; it is not a weight scale or
the checkpoint's inverse/dequantization multiplier.

The forced `mailbox`/`all` modes retain the original experimental kernels:

| Width | Producer (unchanged) | Experimental preparation |
| --- | --- | --- |
| 1–8 | SIMT GEMM into BF16 multicast mailbox | Poll each complete 16-value block, quantize, write local FP4/scales, rearm mailbox |
| 9–1280 | cuBLAS GEMM into BF16 multicast mailbox | Same fused mailbox consumer |
| 1281–prepared capacity | Local BF16 shard GEMM | Quantize the local shard, multicast FP4 plus linear scales, release/acquire completion |

The first two paths remove BF16 materialization and the following standalone
quantization launch. The large-width path communicates quantized bytes instead
of gathering BF16 and quantizing independently on every rank. It does not
change the GEMM's accumulator or output rounding. No small-width up-projection
mailbox kernel is reused for the large-width exchange.

## Quantization and kernel boundary

`Nvfp4Activation` is a typed, borrowed bundle of packed uint8 `[M,H/2]`, linear
E4M3 `[M,H/16]` scales, logical BF16 shape, and the receiving multiplier. Block
scales cover 16 values and are not swizzled. The format is explicitly registered
as an alternate input to the existing NVFP4 SiTU expert kernel; unrelated
backends cannot select it accidentally. No direct vendor-library imports are
added to runtime code.

The numerical primitive implements FlashInfer's default fast-math recipe:

1. Find the absolute maximum of the 16 BF16 values.
2. Round `(amax * approximate_reciprocal(6)) * global_scale` to E4M3.
3. Normalize values with the same reciprocal order as FlashInfer.
4. Round/saturate each value to E2M1 and pack low nibble first.

`FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH` and `FLASHINFER_NVFP4_4OVER6` select
different recipes and therefore decline this optimization. Non-finite or
non-positive encoding multipliers are errors. Large-width multicast requires
equal multipliers on every rank; unequal multipliers retain mailbox-only
fusion so each receiver still quantizes with its own scale.

## Communication and lifetime invariants

- BF16 producers and sentinel values are unchanged. The consumer checks both
  128-bit fragments before quantizing a block and rearms only consumed rows.
- Quantized values can contain every byte pattern. They are never used as
  readiness sentinels. Large-width exchange uses separate per-CTA, per-peer
  release/acquire flags and alias fences for multicast/unicast mappings.
- A rank publishes only its own output columns. No reduction is needed.
- Output and scale buffers rotate with the existing two mailbox slots. Their
  symmetric-memory handles remain alive. A returned activation is borrowed
  for immediate, same-stream expert consumption, not retained across layers.
- Storage and compile-time choices are prepared after expert weight processing
  and before CUDA Graph capture. Forward performs no collective initialization,
  host readback or value-based dispatch. Unsupported widths use the old path.
- The mailbox kernel can signal an early dependent launch before its reset
  pass finishes. This is scheduling permission, not data readiness: a PDL
  consumer must execute `griddepcontrol.wait` before reading these outputs,
  which waits for predecessor completion and visibility. Large-width multicast
  does not signal an early expert launch. Peer data and scales must be complete
  before kernel return. See the [PTX synchronization contract](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol).

## Startup controls

No environment override is needed to enable the optimization. If overriding
`TOKENSPEED_K3_DOWN_NVFP4_FUSION`, set it consistently on all serving ranks
before model construction:

- `off`: explicitly restore the original main input path.
- `mailbox`: fuse preparation only for M ≤ 1280.
- `all`: also enable quantized multicast above 1280, within prepared capacity.
- `auto` (default): use ready-group fused mailbox consumption at M=1–8 and M=96–1280;
  retain the original cooperative fused consumer at M=9–95. Above 1280 retain
  main, including its FlashInfer quantizer. Auto allocates only mailbox-sized
  local outputs and does not prepare quantized-multicast storage or kernels.
  The retained cooperative kernel uses 608 CTAs, 128 threads and 2/4/8 values
  per lane for M=9–32/33–64/65–95. The ready-group kernel uses 8 values/lane,
  128 threads and max(256, ceil(M * hidden / 1024)) CTAs. Its grid is a launch
  argument: one compiled kernel covers every M without capture-time JIT.
  This is a fixed shape policy, not runtime autotuning or a per-M benchmark
  lookup. Forced modes remain available for diagnostics.

The ready-group consumer probes one 128-bit fragment per pending lane per
iteration. A warp-wide ballot identifies complete eight-lane groups, each
covering 64 BF16 values and one packed four-scale word. Ready groups quantize,
write and reset immediately; other groups keep polling. All lanes participate
in the ballots, including padding and completed groups. Completed groups never
reread their reset fragments. The producer, quantization math and two-mailbox
rotation are unchanged, and there is no BF16 materialization or extra kernel.
M must still fit the prepared mailbox, and hidden must be divisible by 64.

The implementation must log its resolved mode and capacity. Benchmark manifests
must record the base commit, patch hash, libraries, GPU topology, checkpoint
revisions and dataset SHA256. Missing multicast support is not a valid fused
performance result.

## Acceptance gates

1. Byte-identical payload and scales against pinned FlashInfer for zero, signed
   zero, tiny values, rounding boundaries, saturation, random inputs and all
   dispatch boundaries. Test PDL off/on and mailbox reset over 1000 graph replays.
2. Compare ordinary and prequantized expert execution through public registry
   dispatch, including empty input, finalized output, and deferred finalize.
   Ignore unspecified scratch padding, not valid expert rows.
3. On TP8, compare the exact main mailbox or `all_gather_inner` plus quantization
   against fusion, including M=8/9 and 1280/1281, prefill-sized rows, repeated
   buffers, and delayed peers. Timing uses the slowest rank and alternating
   paired measurements. Never replace the baseline collective with NCCL for
   convenience.
4. After local correctness and performance gates, proceed directly to the
   agreed full-model gold-standard benchmark, without an additional serving
   smoke test. Use an untouched main installation for the baseline.
5. Freeze the dispatch policy, then run three independent paired clean boots of
   main and the final candidate with the main K3 agentic TP8 EAGLE3 benchmark.
   Leave `TOKENSPEED_K3_DOWN_NVFP4_FUSION` unset on every rank in both arms so
   the candidate exercises the default startup path; verify its preparation
   logs report `mode=auto`.
   Preserve dataset, warmup, concurrency/number pairs, all request parameters,
   FP8 KV cache and CUDA Graph settings. Report all runs; do not choose the best.
6. Report TPS/User = 1000 / TPOT(ms), TPS/GPU = total throughput / 8, cache hit,
   iteration-weighted Tok/Iter and paired confidence intervals. Attribute a
   serving improvement only when it exceeds measurement noise without a
   correctness or acceptance-length regression.
7. Profile separate first/second-turn captures at CC1, CC8 and CC16 with CUDA
   Graph node tracing. Verify the intended producer/consumer replacement and
   inspect prefill and decode separately. Profiling runs are not timing samples.

Only unit tests for the new scalar, cooperative and ready-group kernels are
included under `tokenspeed-kernel/test/nvidia/ops/moe/`:
`test_latent_down_nvfp4_gpu.py`, `test_latent_down_nvfp4_cooperative.py` and
`test_latent_down_nvfp4_grouped.py`.
Historical distributed and end-to-end validation above is not an additional
checked-in test suite.

The pinned EvalScope client advances its dataset offset within a sweep. The
five CC points therefore use zero-based conversation slices `[0:4]`, `[4:12]`,
`[12:20]`, `[20:36]`, and `[36:68]`, respectively; they do not all start at zero.
Warmup is a separate invocation using `[68:70]`. Keep this behavior identical
for both arms and validate `benchmark_args.json` plus each conversation's turn
count. The profile-only first wave at each CC must come from the corresponding
slice; truncating it to two turns must not change the performance dataset.
Before a CC8 or CC16 profile, replay all earlier sweep points outside the
capture window, after the normal warmup. This preserves the sweep's cache
conditioning order; a fresh server with only the two warmup conversations does
not reproduce that state. Validate the conditioning requests as well as the
captured requests. Profiling remains separate from the timed pairs.
The two-turn truncation introduces a drain tail when faster conversations end.
Report fully occupied pure-decode steps separately from partial batches, and
retain mixed-turn/mixed-prefill steps as separate categories. Do not interpret
the entire short capture as a steady-state full-concurrency measurement.

## Running the kernel unit tests

The scalar, cooperative and ready-group kernels are checked against
FlashInfer's packed payload and linear scales. Tests cover numerical edge
cases, partial rows, PDL off/on, CUDA Graph replay and mailbox reset. The cooperative cases exercise
2/4/8 values per lane and multiple launch geometries. Fusion is in the mailbox
consumer, not the GEMM.

Run with the same Blackwell runtime and FlashInfer version used for serving:

```bash
python -m pytest tokenspeed-kernel/test/nvidia/ops/moe/test_latent_down_nvfp4_gpu.py -q
python -m pytest tokenspeed-kernel/test/nvidia/ops/moe/test_latent_down_nvfp4_cooperative.py -q
python -m pytest tokenspeed-kernel/test/nvidia/ops/moe/test_latent_down_nvfp4_grouped.py -q
```
