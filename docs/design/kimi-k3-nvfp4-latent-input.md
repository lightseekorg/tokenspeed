# Kimi-K3 NVFP4 latent input fusion

## Scope and status

Experimental, opt-in preparation of the NVFP4 routed-expert input after K3's
column-sharded latent down projection. The default remains the unchanged BF16
mailbox/gather followed by FlashInfer input quantization. Correctness and
end-to-end performance must pass the gates below before enabling any width by
default. This document describes the implementation contract, not a measured
performance claim.

The validated serving target is one eight-GPU Blackwell TP8 attention / TP8 MoE
replica with the NVFP4 SiTU FlashInfer TRT-LLM backend. Runtime eligibility
requires an eight-rank MoE TP group, EP=1, and the existing multicast producer.
EP-routed configurations (including DEP8/TEP8), other MoE TP widths, other
quantization recipes and other expert backends retain their existing input
path. Additional data-parallel replicas or hybrid attention mappings are not
validated by this experiment. This work does not change routed up-projection,
shared experts, routing, sampling or EAGLE3 settings.

## Validation summary

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
normalized CC1/CC16 regressions are accepted without per-shape exceptions;
the optimization remains opt-in. First/second-turn node-level profiling for
this final candidate remains a release gate before considering default-on.

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

## Experiment controls

Set `TOKENSPEED_K3_DOWN_NVFP4_FUSION` consistently on all serving ranks before
model construction:

- `off` (default): original main input path.
- `mailbox`: fuse preparation only for M ≤ 1280.
- `all`: also enable quantized multicast above 1280, within prepared capacity.
- `auto`: use original BF16 gather plus the cooperative quantizer at M=1–8;
  cooperative fused mailbox consumption at M=9–128; original BF16 gather plus
  the original scalar quantizer at M=129–1280. Above 1280 retain main, including
  its FlashInfer quantizer. Auto allocates only mailbox-sized local outputs and
  does not prepare quantized-multicast storage or kernels. The separate small
  kernel uses 8 values/lane, 256 CTAs and 128 threads. The cooperative fused
  kernel uses 608 CTAs, 128 threads and 2/4/8 values per lane for M=9–32/33–64/
  65–128. The scalar separate kernel uses 608 CTAs and 256 threads. All choices
  are frozen before capture; no per-M benchmark lookup or prefill/decode fork.
  This deliberately simple policy accepts the measured small regressions at
  intermediate M rather than adding exceptions. Correctness remains mandatory;
  acceptance is based on complete serving results, not a >1% win at every M.
  These fixed launch bands were checked in numerical gates and three paired
  serving runs; per-width tuned results alone do not validate them. Forced
  modes remain available for diagnostics.

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
4. After numerical gates, validate a complete K3 block and a serving smoke. Use
   an untouched main installation for baseline, not merely an assumed-disabled
   historical patch stack.
5. Freeze the dispatch policy, then run three independent paired clean boots of
   main and the final candidate with the main K3 agentic TP8 EAGLE3 benchmark.
   Preserve dataset, warmup, concurrency/number pairs, all request parameters,
   FP8 KV cache and CUDA Graph settings. Report all runs; do not choose the best.
6. Report TPS/User = 1000 / TPOT(ms), TPS/GPU = total throughput / 8, cache hit,
   iteration-weighted Tok/Iter and paired confidence intervals. Attribute a
   serving improvement only when it exceeds measurement noise without a
   correctness or acceptance-length regression.
7. Profile separate first/second-turn captures at CC1, CC8 and CC16 with CUDA
   Graph node tracing. Verify the intended producer/consumer replacement and
   inspect prefill and decode separately. Profiling runs are not timing samples.

GPU tests live under `tokenspeed-kernel/test/nvidia/ops/moe/`:
`test_latent_down_nvfp4_gpu.py`, the prequantized regression in
`test_nvfp4_situ_flashinfer_trtllm.py`, and the eight-rank
`bench_latent_down_nvfp4.py` driver.

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

## Reproducing the numerical gate

The cooperative quantizer used by the auto policy lives in
`nvfp4_input_cooperative.py`. The quantization group stays 16 values, while
2/4/8 lanes share its scale computation (8/4/2 values per lane). Producer
geometry, BF16 rounding, mailbox sentinels, readiness and reset remain
unchanged. The same arithmetic also accepts a plain local tensor, allowing
quantizer-only, separate gather/quantize and fused timings to be compared.
The separate benchmark driver also compares it with the original scalar
quantizer without changing the serving dispatch policy.
Acceptance requires exact payload/scales, partial-warp masks,
live-row-only reset, rotating mailboxes, delayed peers and graph replay with
both PDL policies; performance is decided against main with its original PDL
setting, not against an artificially disabled baseline.

The experiment names distinguish quantizer organization from fusion:

| Variant | Mailbox consumption | Quantization |
| --- | --- | --- |
| `main` | Original BF16 gather | Separate FlashInfer kernel |
| `old-fused` | Fused with quantization | Original single-lane 16-value scale group |
| `fused-v*-c*` | Fused with quantization | Cooperative scale group |
| `separate-v*-c*` | Original BF16 gather | Separate cooperative kernel |

The two cooperative variants use the same quantization primitive, with different
source readiness/reset handling. Fusion happens in the mailbox consumer, not
the GEMM. Their fastest launch configurations need not be identical. Pipeline
time reductions use `main` as the denominator, not `old-fused`; quantizer-only
reductions instead use `q-fi`. Above 1280, separate variants use the original
`all_gather_inner` path as described below, with no mailbox-fusion variant.

Run the single-GPU tests with the same installed runtime and FlashInfer version
used for serving:

```bash
python -m pytest tokenspeed-kernel/test/nvidia/ops/moe/test_latent_down_nvfp4_gpu.py -q
python -m pytest tokenspeed-kernel/test/nvidia/ops/moe/test_nvfp4_situ_flashinfer_trtllm.py -q
python -m pytest test/runtime/test_kimi_k3_moe_fork_warmup.py -q
```

On each of two four-GPU nodes in one verified multicast-capable allocation,
launch the distributed driver with the same rendezvous address. Supply the
actual master address, rank and unique output path through task-specific
environment variables:

```bash
torchrun --nnodes=2 --nproc-per-node=4 \
  --node-rank="$K3_NODE_RANK" --master-addr="$K3_MASTER_ADDR" --master-port=8573 \
  tokenspeed-kernel/test/nvidia/ops/moe/bench_latent_down_nvfp4.py \
  --mode all --pdl 1 --replays 1000 --output "$K3_GATE_OUTPUT"
```

Repeat with `--mode all --pdl 0` and `--mode auto --pdl 1`, each with its own
output file. A graph replay spans both rotating workspaces. The reported
microseconds are per projection, measured on the slowest rank, with six
alternating paired timing rounds. Inputs and weights are reused: this is a
hot-buffer microbenchmark, not a cold-weight or full-model performance claim.

For the cooperative quantizer, run its separate gate:

```bash
python -m pytest tokenspeed-kernel/test/nvidia/ops/moe/test_latent_down_nvfp4_cooperative.py -q
torchrun --nnodes=2 --nproc-per-node=4 \
  --node-rank="$K3_NODE_RANK" --master-addr="$K3_MASTER_ADDR" --master-port=8573 \
  tokenspeed-kernel/test/nvidia/ops/moe/bench_latent_down_nvfp4_cooperative.py \
  --pdl 1 --rounds 6 --tokens 1,4,8,9,16,32 --ctas 4,16,64 --replays 1000 \
  --output "$K3_COOPERATIVE_OUTPUT"
```

Repeat with PDL disabled as a diagnostic, not a replacement baseline. The full
pipeline graph still contains two alternating mailbox slots. Quantizer-only
controls contain 16 operations per graph to amortize host launch gaps; their
reported duration is divided by 16 and must not be presented as pipeline time.
Both quantizer controls retain 16 distinct payload and scale buffers, matching
FlashInfer's captured allocation pattern. Reusing only two custom outputs would
give it a different cache working set and bias large-M comparisons. Full
pipelines retain two distinct output buffers for their two mailbox slots.
Select configurations on one sweep, then examine those same configurations on
an independent sweep. Do not select a new minimum from the confirmation sweep.

The cooperative driver accepts widths up to 8192 and an explicit CTA list.
Expand the grid search for larger M instead of extrapolating from a 64-CTA
ceiling. Above 1280, the baseline and separate candidate both use main's local
shard GEMM plus `all_gather_inner`; there is no cooperative mailbox-fusion
result at these widths. The original mailbox capacity and runtime dispatch
remain unchanged. Record pure-quantizer and full-pipeline gains separately;
neither sparse positive points nor a different best geometry at every width
establish a deployable contiguous interval without boundary validation.

Eager reference construction must complete on every peer before a correctness
check reuses the same mailbox slot. A local clone or device synchronization is
not a cross-rank lifetime fence. The cooperative driver therefore adds a barrier
between reference generation and its first checked variant, outside all timed
regions. Timed graphs keep the existing two-slot rotation without new barriers.
