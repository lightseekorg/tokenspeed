# Kimi-K3 NVFP4 latent input fusion

## Scope and status

Automatic preparation of the NVFP4 routed-expert input after K3's
column-sharded latent down projection. The default policy is `auto`: eligible
configurations use fusion without an environment override. Unsupported
configurations and widths retain the BF16 mailbox/gather followed by FlashInfer
input quantization. `TOKENSPEED_K3_DOWN_NVFP4_FUSION=off` explicitly restores
that original path for every width.

This branch changes the default. The rebased numerical/module gates, three
paired full-model sweeps and CC1/8/16 node-level profiling below are complete.
The module improves, but the serving measurements do not establish an overall
end-to-end throughput improvement.

The validated serving target is one eight-GPU Blackwell TP8 attention / TP8 MoE
replica with the NVFP4 SiTU FlashInfer TRT-LLM backend. Runtime eligibility
requires an eight-rank MoE TP group, EP=1, and the existing multicast producer.
EP-routed configurations (including DEP8/TEP8), other MoE TP widths, other
quantization recipes and other expert backends retain their existing input
path. Additional data-parallel replicas or hybrid attention mappings are not
validated by this experiment. This work does not change routed up-projection,
shared experts, routing, sampling or EAGLE3 settings.

## Rebased default-policy validation

The frozen comparison uses unmodified main
`78c6518720e41b7a4a82697e593bda7a5ab05dc3` and candidate
`519e904266874e0887df93d1b5b7d1253f6024f1`, on 8xGB300 with FlashInfer 0.6.18
and PyTorch 2.13.0+cu130. The rebase preserves main's tuple-based prequantized
input and attention-DP/all-to-all support; the typed activation is an additional
registered input, not a replacement for that interface.

All 761 numerical/integration cases passed without skips: 94 scalar, 204
cooperative, 111 ready-group/default-policy, 42 expert compatibility and 310
runtime cases. Distributed PDL off/on validation covered 18 widths and 1000
graph replays, checking each of eight individually delayed peers. Additional
changing-input stress covered 16 widths, four inputs and two mailbox slots,
with 64 projections per graph and 64 replays for every delayed peer.

The table measures the complete projection-through-quantization module, not
the quantizer alone. Each timing sample takes the slowest TP rank; reported
times are medians of six alternating rounds, with two projections per graph
and 1000 graph replays per round. PDL is enabled. Positive values mean reduced
time against the exact main path, including its existing collective and
FlashInfer quantizer.

| M | Main (us) | Default auto (us) | Time reduction |
| --- | ---: | ---: | ---: |
| 1 | 6.804 | 6.018 | +11.56% |
| 2 | 7.483 | 6.566 | +12.26% |
| 4 | 8.197 | 7.107 | +13.29% |
| 8 | 9.722 | 8.761 | +9.88% |
| 9 | 10.248 | 8.853 | +13.62% |
| 32 | 10.509 | 9.444 | +10.14% |
| 33 | 11.826 | 10.623 | +10.17% |
| 64 | 10.764 | 10.050 | +6.63% |
| 65 | 12.844 | 12.298 | +4.25% |
| 95 | 12.343 | 11.718 | +5.06% |
| 96 | 12.319 | 11.125 | +9.69% |
| 97 | 11.995 | 10.948 | +8.73% |
| 128 | 12.059 | 11.275 | +6.50% |
| 129 | 13.349 | 12.254 | +8.20% |
| 1279 | 32.775 | 26.112 | +20.33% |
| 1280 | 32.794 | 26.112 | +20.38% |
| 1281 (unchanged fallback) | 41.638 | 41.627 | +0.03% |
| 8192 (unchanged fallback) | 146.622 | 146.786 | -0.11% |

The fallback differences are measurement noise; neither width executes the
new kernel.

## Rebased full-model results

Three paired clean boots compared the frozen main and candidate above on the
same eight GPUs within each pair, in AB/BA/AB order. Both arms used the same
runtime, checkpoints and main K3 agentic TP8 EAGLE3 configuration, FP8 KV cache,
and an 8192-token prefill budget/chunk/graph maximum. The optimization variable
was unset on both arms; every candidate rank prepared all 92 layers in `auto`.
CUDA Graph and PDL remained enabled. There was no additional serving smoke test.

All 4716 timed turns completed successfully with exactly 500 output tokens.
The effective configurations, user histories, warmup, conversation slices,
library versions and paired GPU identities passed comparison. The dataset
SHA256 was `2405552a4f320bbaaf084a0f25f57dc1c9af44e69612d0c9d098a137db3463f2`,
using EvalScope `acd09b44384d53174768bb1063f675420f76fae9`.

Changes are geometric means of candidate/main ratios from all three pairs.
Intervals are descriptive paired 95% t intervals on log ratios (df=2), not
multiple-comparison-adjusted. No run was selected or dropped.

| CC | Raw TPS/User | 95% interval | Raw TPS/GPU | 95% interval | Tok/Iter change | Normalized TPS/User |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| 1 | +0.78% | [-0.94%, +2.53%] | +0.42% | [-0.90%, +1.74%] | +0.71% | +0.07% |
| 2 | -1.62% | [-2.53%, -0.71%] | -1.38% | [-3.90%, +1.21%] | -1.84% | +0.22% |
| 4 | -0.40% | [-2.45%, +1.69%] | -0.92% | [-3.07%, +1.27%] | -0.55% | +0.14% |
| 8 | -0.60% | [-3.43%, +2.32%] | -1.04% | [-5.10%, +3.19%] | -0.04% | -0.55% |
| 16 | +0.87% | [-2.08%, +3.91%] | -0.30% | [-3.41%, +2.92%] | +0.28% | +0.59% |
| Across-CC geomean | -0.20% | [-1.50%, +1.12%] | -0.65% | [-2.61%, +1.36%] | -0.29% | +0.09% |

`TPS/User = 1000 / TPOT(ms)` and `TPS/GPU = total throughput / 8`.
Tok/Iter uses the main collector's iteration-weighted statistic within each
run. Normalization applies `TPS/User * 3.5 / Tok/Iter` before paired aggregation;
its across-CC interval is [-0.23%, +0.42%]. It is a first-order sensitivity
estimate, not measured performance with acceptance held fixed: iteration cost
need not remain constant, and normalization does not erase the raw results.

The CC2 raw TPS/User decrease accompanies a 1.84% lower Tok/Iter (descriptive
interval [-2.54%, -1.13%]); these measurements alone do not establish its cause.
CC16 mean TTFT increases 4.42% (interval [+0.54%, +8.44%]). Neither observation
is omitted from acceptance. Mean cache-hit changes are within 0.013 percentage
points at every CC. Local module gains must not be reported as serving gains.

## Node-level validation

Independent main/candidate captures covered the first two turns at CC1, CC8
and CC16, using Nsight Systems 2026.4.1 with CUDA Graph node tracing
(`cuda-sw,nvtx`, `node:host-only`). The normal warmup and all earlier sweep
points ran outside capture to preserve cache conditioning. Serving parameters
matched the timed comparison, apart from NVTX and deployment-specific fields.
CUDA Graph and PDL stayed enabled; no eager path was substituted.

All 100 captured benchmark turns succeeded with 500 output tokens. The two
nodes in each arm have identical ordered forward-step sequences across all
eight ranks. All 12 reports passed analysis, with 99.03–99.65% of GPU kernel
time attributed to forward steps. Every actual-request pure-decode step
contains CUDA Graph node events.
Attribution follows CUDA API/NVTX ownership; zero-correlation graph nodes are
recovered only by unique containment in API-proven GPU-step bounds in the same
process/device/context. Turn labels use rank-zero monotonic time and the
validated TP-step ordinal, not a union of unsynchronized host clocks.

| CC | Main captured TP steps | Auto captured TP steps | Main full-CC decode steps | Auto full-CC decode steps |
| --- | ---: | ---: | ---: | ---: |
| 1 | 466 | 477 | 298 | 309 |
| 8 | 617 | 608 | 277 | 277 |
| 16 | 641 | 636 | 282 | 263 |

Captured counts include connection-check/setup work; full-CC decode counts
exclude it and exclude draining batches. There are 2497 actual-request TP
steps across the six arms. Eight ranks are not counted as eight independent
measurements. First-turn, second-turn and overlapping-turn windows are retained
separately, as are prefill and pure decode.

Every actual-request target forward step has exactly 92 input encodings. Main
decode uses 92 mailbox materializers plus 92 standalone quantizers. Default
auto replaces them with 92 ready-group consumers at CC1 (M4) and 92 cooperative
consumers at CC8/16 (M32/M64). Draining to one or two requests correctly selects
ready-group; no forced large-width quantized-multicast kernel was observed.
These traces verify the intended replacement and graph coverage, not a serving
speedup. Kernel-duration sums can overlap and are not critical-path latency;
none of the instrumented timings enters the paired performance results above.

## Earlier ready-group module validation

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

The current serving results above supersede the previous opt-in policy's
comparison against `eaf66b5be72ca1e9aed8b0a4c23165f4be072a2d`. Those historical
boots are not pooled with this rebased default-policy campaign.

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
Distributed and end-to-end validation above is not an additional
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
