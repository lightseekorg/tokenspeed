# Simulated forward benchmarks

Simulation shortens the iteration cycle for inference performance work. It
avoids checkpoint weight reads and long-context prefill while retaining the
ordinary model arithmetic, kernel selection, tensor shapes and distributed
communication. The objective is realistically timed forwards, not meaningful
generated text. Always compare against real weights and computed KV before
using a synthetic workload to guide an optimization.

## Design

`--simulate-tokens` selects dummy weight initialization and enables simulation at the
final writeback of token-dependent routing and supported decode sparse-selection
operations.
The original projections, scoring, top-k selection and normalization still run.
The selected IDs remain inputs to the final mapping so the compiler cannot
remove that work. The mapping spreads accesses across valid expert or history
indices; it does not reproduce every trained router's load imbalance or every
prompt's attention locality. Unsupported kernel implementations fail explicitly
rather than silently changing the selected backend.

Synthetic state uses the ordinary scheduler and LCM cache allocator. The direct
benchmark advances chunked allocation on the CPU, holds completed requests
until the whole batch is ready, and obtains an ordinary decode plan. It then
clears cache payloads, seeds sparse index keys through the normal cache codecs,
and initializes request lengths, token history and recurrent/window state.
No request prefill runs on the GPU. Cache groups retain ownership of page
geometry, allocation and lifetime; no benchmark-specific addresses are invented.

Weights have valid bounded floating values and packed quantization encodings.
Initialization precedes derived projections and layout conversion, so derived
buffers agree with their source tensors. FP8 initialization uses bounded FP16
scratch rather than a whole-weight temporary. Main KV, recurrent and window
payloads start at zero; sparse index keys are nonzero to avoid all-tied scores.
These are deliberate benchmark inputs whose timing fidelity must be measured.

The flag alone does not bypass prefill through the public serving API. Synthetic
prefix preparation is an exclusive operation of the direct benchmark; synthetic
prefixes must never be published to a server handling real requests.

## Measurement contract

`test/forward_benchmark/bench_decode.py` loads the execution stack once per
invocation and tests its batch/context matrix. Real mode computes ordinary
chunked prefill. Simulated mode allocates and seeds state without that prefill.
Both modes then call the normal forward method with real collectives.

Each replay restores writable KV tail pages, all live sliding/recurrent pages,
and per-request state. Immutable prefix pages are not copied. Snapshot restore,
hardware-cache eviction and the pre-forward CPU barrier are outside the forward
timer. The full benchmark wall time includes them, warmups and snapshot creation.
Per-rank JSON records GPU and wall times, observed JIT compilations, output
lengths, cache capacity, preparation time and startup phases. The rank receipt
also records whether `--enable-nan-detection` was enabled; use it to reject
nonfinite output during each replay. Compare aligned
iterations using the slowest rank, then compute the median and tail latency.

Use identical token lists, precision, sharding and other execution flags for the
real and simulated invocations. Only the output directory and `--simulate-tokens`
should differ. Token JSON contains one token list per request, with at least
`max(contexts) + 1` tokens per list. Input IDs and configured capacities are
validated before weight allocation. Input preparation is outside the invocation
timer; importing the runtime and launching ranks are included.

```bash
python test/forward_benchmark/bench_decode.py \
  --output /absolute/new-results \
  --tokens-json /absolute/prompts.json \
  --batches 4,1 --contexts 250000,8192 \
  --iterations 100 --warmup 5 --evict-mib 256 -- \
  --model /absolute/model --world-size 8 --attn-tp-size 8 \
  --max-num-seqs 4 --max-cudagraph-capture-size 4 \
  --chunked-prefill-size 8192 --gpu-memory-utilization 0.90 \
  --disable-overlap-schedule --disable-prefix-caching --disable-kvstore \
  --retraction-snapshot-ratio 0 --enable-nan-detection --simulate-tokens
```

Choose model-specific tokenizer, quantization and parallelism options below.
Put the desired startup case first in the batch/context lists. Use the machine's
GPU reservation mechanism; the driver does not acquire locks itself.

## Supported configurations and validation

Implemented gfx950 paths cover target-only TP8 Kimi K3.1, TP4 DeepSeek V4.1
and TP8 Muse Spark. Batch sizes 1/4 and context lengths 8192/250000 form a
reference comparison matrix. Require finite outputs and no measured JIT
compilations before interpreting forward timings.
EP, speculative decoding and other workloads require their own full-model
comparisons; retaining their normal communication does not establish fidelity.
The driver currently requires serial, aggregate execution with attention DP=1,
PP=1, KVP=1 and device-resident cache, without prefix reuse or retraction snapshots.

- Kimi K3.1 at text TP8 needs `--mm-encoder-tp-mode data`: its 12 vision heads
  cannot divide across eight ranks. Supply `--trust-remote-code` and a tokenizer
  directory with the checkpoint's custom tokenizer files.
  The CuTe decode MoE chain has no final-ID simulation and is rejected when
  it would otherwise be selected.
  DSA sparse-selection simulation currently covers decode; DSA prefill,
  including startup warmups, retains its ordinary selection. This flag does
  not establish a simulated prefill-benchmark contract.
- Muse Spark uses the ordinary dense/sliding KV arena; it has no sparse index
  keys to seed. Its gfx950 fused all-gather, RMSNorm, MXFP4 quantization and
  routing kernel retains the fusion and permutes only the final expert IDs.
  Prepared NVFP4 routing is unsupported and fails during initialization.
  The validated TP8 setup uses the Quark MXFP4 checkpoint and memory
  utilization 0.85.
- DeepSeek V4.1 uses world size and attention TP size 4 in the validated setup,
  with memory utilization 0.85. Supply `--trust-remote-code` for Engram's
  tokenizer. Bounded n-gram inputs are separate from full request-token-history
  seeds; the planner follows `ModelConfig.requires_request_token_history`.

Sparse final-store simulation retains unique causal indices and uses stratified
sampling rather than uniform subset sampling. Full candidate-block selection
includes the newest visible block; reuse layers consume the shared selection.
The index cache must have every visible page initialized in its normal encoding.
The supported selector paths and invariants have GPU tests for graph replay,
cache write isolation, bounds and changing runtime seeds without recompilation.

## Measured timing comparisons

These gfx950 measurements compare real and simulated target-only text decode
with the same precision and execution settings: Kimi K3.1 TP8, DeepSeek V4.1
Flash TP4 and Muse Spark Quark MXFP4 TP8. Each invocation runs B4 then B1 at
250000 then 8192 tokens/request, with 100 measured forwards and five warmups
per case. Finite-output checks passed and no timed forward compiled a kernel.

### Initialization and benchmark wall time

All values below are seconds, written as **real / simulated**. Initialization
and individual startup phases are host-timed maxima across ranks. Phase maxima
can come from different ranks and include nested work; do not add these rows.
The first KV preparation is B4 at 250000 tokens/request: real execution prefills
1000000 tokens, while simulation allocates and seeds the equivalent cache state.

| Phase | Kimi K3.1 TP8 | DeepSeek V4.1 Flash TP4 | Muse Spark MXFP4 TP8 |
| --- | ---: | ---: | ---: |
| Distributed initialization | 12.29 / 12.29 | 3.77 / 4.19 | 12.50 / 12.09 |
| Weight allocation | 0.57 / 0.61 | 0.43 / 0.44 | 0.65 / 0.66 |
| Checkpoint read/copy or dummy fill | 145.71 / 0.04 | 50.05 / 0.07 | 27.10 / 0.02 |
| Weight layout postprocessing | 7.46 / 8.50 | 0.01 / 2.26 | 2.14 / 2.65 |
| Communication preparation | 4.47 / 4.38 | 0.00 / 0.00 | 1.73 / 1.53 |
| KV arena construction | 1.01 / 0.20 | 0.10 / 0.16 | 0.19 / 0.20 |
| Kernel autotuning | 0.92 / 0.81 | 1.36 / 1.38 | 0.72 / 1.93 |
| Graph probe/rebind | 24.74 / 3.89 | 17.08 / 16.91 | 12.09 / 16.01 |
| Final graph capture | 2.85 / 3.07 | 54.73 / 53.12 | 50.42 / 53.97 |
| Initialization total | 199.71 / 33.25 | 131.25 / 82.22 | 110.02 / 91.85 |
| First KV preparation | 71.58 / 0.47 | 18.33 / 0.16 | 33.62 / 0.21 |
| Whole four-case invocation | 305.37 / 50.48 | 169.61 / 97.25 | 175.56 / 115.34 |

Initialization includes the other setup phases, such as configuration,
multimodal modules and executor construction. Whole-invocation time also
includes runtime imports, rank launch, all four KV preparations, snapshots,
warmups, timed replays, cache eviction and teardown. It excludes token-list
preparation and GPU queue waiting. Existing compilation-cache directories were
reused, but startup still recorded JIT events: 187/rank for Kimi, 173/rank for
DeepSeek and 204/rank for Muse in both modes. These are observed invocation
times, not controlled cold-cache or warm-cache startup comparisons.

### Steady-state forward latency

Each value is the median of 100 GPU-time maxima over aligned rank iterations.
Normal kernels and collectives are included. Snapshot restoration, CPU barriers
and 256 MiB hardware-cache eviction before every replay are outside this timer.

| Model | Batch | Context/request | Real (ms) | Simulated (ms) | Difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kimi K3.1 | 1 | 8192 | 14.975 | 15.047 | +0.5% |
| Kimi K3.1 | 1 | 250000 | 15.430 | 15.714 | +1.8% |
| Kimi K3.1 | 4 | 8192 | 16.703 | 16.825 | +0.7% |
| Kimi K3.1 | 4 | 250000 | 17.714 | 17.935 | +1.2% |
| DeepSeek V4.1 Flash | 1 | 8192 | 9.438 | 9.430 | -0.1% |
| DeepSeek V4.1 Flash | 1 | 250000 | 9.943 | 9.768 | -1.8% |
| DeepSeek V4.1 Flash | 4 | 8192 | 9.882 | 9.922 | +0.4% |
| DeepSeek V4.1 Flash | 4 | 250000 | 10.587 | 10.290 | -2.8% |
| Muse Spark | 1 | 8192 | 5.564 | 5.514 | -0.9% |
| Muse Spark | 1 | 250000 | 6.040 | 6.000 | -0.7% |
| Muse Spark | 4 | 8192 | 7.062 | 7.050 | -0.2% |
| Muse Spark | 4 | 250000 | 8.467 | 8.487 | +0.2% |

Timing agreement applies to these text TP cases. It does not establish fidelity
for EP, speculation, multimodal requests or arbitrary trained-router imbalance.
