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
Put the desired startup case first in the batch/context lists.

## Supported configurations and validation

Simulation supports Kimi K3.1, DeepSeek V4.1 and Muse Spark through the
implemented gfx950 paths. Compare real and simulated execution for each workload,
requiring finite outputs and no measured JIT compilations before interpreting
forward timings.
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
  Prepared NVFP4 routing and decode monokernels are unsupported and fail during
  initialization because they do not simulate the final expert IDs.
  Use a Quark MXFP4 checkpoint on gfx950.
- DeepSeek V4.1 needs `--trust-remote-code` for Engram's tokenizer. Bounded
  n-gram inputs are separate from full request-token-history seeds; the planner
  follows `ModelConfig.requires_request_token_history`.

Sparse final-store simulation retains unique causal indices and uses stratified
sampling rather than uniform subset sampling. Full candidate-block selection
includes the newest visible block; reuse layers consume the shared selection.
The index cache must have every visible page initialized in its normal encoding.
The supported selector paths and invariants have GPU tests for graph replay,
cache write isolation, bounds and changing runtime seeds without recompilation.
