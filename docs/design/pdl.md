# Programmatic dependent launch ordering

GDN, Qwen4-Exp gated residual and QSA use the same device ordering in eager
execution and CUDA graphs. The host selects PDL before launch; compiled CuTe
variants include that setting in their cache key. Changing the global switch
does not change an existing graph.

## Launch readiness and data readiness

`griddepcontrol.launch_dependents` permits the next kernel to start. It does
not publish the producer's results. `griddepcontrol.wait` waits for predecessor
completion and global-memory visibility. These semantics follow NVIDIA's
[PDL execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

Our common sequence is:

1. Prepare descriptors, registers, shared memory and other independent state.
   Load weights early only when the caller explicitly guarantees their readiness.
2. Wait before reading or overwriting data the predecessor may still use.
3. Trigger successor setup after the wait, as early as residency allows.
4. Compute and store results; the successor's wait protects their consumption.

Waiting before triggering also orders ancestor inputs that a successor may
preload. For example, HC Up preloads weights that Down never writes, but a
producer preceding Down may write them. Down must wait for that producer
before triggering Up. Every CTA must trigger or finish, including padding
and empty splits. Warp-specialized kernels keep the wait in every reader warp
or propagate readiness through their existing barriers. No path relies on
concurrent execution for forward progress.

## GDN

The QKV split, convolution, gating, recurrent update, chunked prefill stages
and gated norm release successors before their main computation. Convolution
and recurrent-state reads/writes remain behind the incoming wait, including
replay payload and per-token state destinations. FlashInfer's private adapter
waits and triggers before inlining the original body, preserving upstream
launch geometry and private PDL compilation caches.

Gated RMSNorm's required `weights_independent` argument permits a static norm
weight to load while GDN is running. Runtime model callers own immutable
weights during forward. Generic callers must pass False for producer-written
weights. If normalization materializes a contiguous weight copy, its actual
load stays after the wait regardless of the original promise. Activations and
gates always stay behind the wait.

## QSA

Metadata preparation, compression, scoring, merge, selected-slot mapping,
FlashInfer metadata packing and CuTe sparse attention participate in PDL.
Compression and both score implementations wait before reading inputs,
including query, compressed cache, page tables and row metadata. Both merge
levels wait before consuming partial keys and release their successors early.
Empty scoring splits still wait and trigger before skipping their stores.
The streaming score kernel triggers just before storing partial keys. Its
merge performs little independent work; releasing those CTAs before the
dot/sort loop competes with resident scoring CTAs and regresses larger
batches. Logits scoring and both merge levels release successors after the
incoming wait. The downstream slot mapper and attention still overlap setup.

The recent writer can preload token keys and row metadata because compression
has waited for their producer before triggering it, and never changes those
inputs. Its raw-ring and position-header writes remain after its wait for
compression to finish reading the old ring. The writer triggers scoring after
that wait, including when no row writes. Its API requires this
compression-to-recent-write ordering when PDL is enabled; unrelated producers
do not establish the preload contract.

CuTe attention prefetches the Q descriptor and initializes its local pipelines
and TMEM before waiting. All warps then wait before reading Q, K/V, selected
slots or the softmax tail, and trigger successor setup before attention work.
The launch flag and compilation-cache key use the same PDL setting. BF16 and
FP8 caches, eager execution and graph replay share this implementation.

The PDL wait supplements the cache subsystem's layerwise load fences; it does
not replace ownership, transfer completion or page-lifetime rules.

## Validation

- `test/runtime/test_gdn_pdl.py` exercises decode, verify and prefill with
  delayed producers, state/cache parity, poisoned projection buffers, weight
  readiness, strided weights and capture-time PDL toggles.
- `tokenspeed-kernel/test/ops/test_qsa_pdl.py` publishes query, cache and metadata
  only after a delayed producer has triggered successors. It checks empty and
  partial rows, both scoring paths, two-level merge, raw-ring reuse, slot
  mapping and BF16/FP8 attention under PDL on/off graph replay.
- `tokenspeed-kernel/test/ops/test_hyperconnection.py` covers all HC backends,
  ancestor weight updates, conservative and independent weights, scratch
  epochs, concurrent streams and changed inputs across graph replays.

Measure graph chains on the same idle GPU:

```bash
PYTHONPATH=python:tokenspeed-kernel/python python \
  tokenspeed-kernel/test/ops/bench_pdl_attention.py \
  --operation gdn --solution flashinfer --batches 1,4,8,16 --steps 1 --pdl on

PYTHONPATH=python:tokenspeed-kernel/python python \
  tokenspeed-kernel/test/ops/bench_pdl_attention.py \
  --operation qsa --solution stream --batches 1,4,8,16 --steps 1 --pdl on

PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/test/ops/bench_hyperconnection.py \
  --backend default --operation chain --rows 1,4,8,16 --mode graph \
  --weight-banks 32 --pdl on
```

Use GDN `--steps 4` for verify or `--solution triton` for its portable backend;
QSA also supports `--solution logits`. `--pdl off` measures serialized launches.
The attention benchmark measures recurrent-update-plus-norm and
selection-plus-attention, with resident inputs and batched graph calls. The
HC benchmark rotates 32 weight banks. These are device-chain measurements,
not end-to-end serving throughput.

### B300 sample measurements

Microseconds before → after this change, BF16, batch sizes 1/4/8/16, using the
commands above. Each value is the median of five measurements of repeated
CUDA graph execution on the same idle GPU. GDN uses FP32 state; QSA uses
4096 compressed blocks, four 128-wide index heads, six 256-wide attention
heads and 2051 selected slots. HC rotates 32 weight banks.

| Device chain | Batch 1 | Batch 4 | Batch 8 | Batch 16 |
| --- | ---: | ---: | ---: | ---: |
| GDN decode + norm | 4.606 → 4.354 | 4.895 → 4.742 | 5.759 → 5.507 | 7.930 → 7.254 |
| GDN verify (T=4) + norm | 4.991 → 4.869 | 7.167 → 7.250 | 9.729 → 9.492 | 14.274 → 13.993 |
| HC norm/mix/combine | 10.116 → 9.491 | 10.163 → 9.507 | 10.789 → 10.149 | 11.171 → 10.531 |
| QSA stream selection + attention | 45.008 → 44.090 | 46.132 → 44.589 | 49.969 → 48.216 | 68.053 → 66.790 |
| QSA logits selection + attention | 40.994 → 39.745 | 42.015 → 40.715 | 45.632 → 44.477 | 63.871 → 63.420 |

GDN verify changes are small and not uniformly positive. These samples do
not establish serving throughput or performance on other GPU architectures.
The delayed-producer correctness tests are necessary even where latency is
unchanged: the old QSA compression and score entrypoints could consume stale
inputs under PDL.

### End-to-end validation

Qwen3.8-Flash-Next NVFP4 was exercised on four B300 GPUs with TP4, BF16
activation/KV, FlashInfer TRTLLM MoE, MTP three draft steps and four verification
tokens, MTP index sharing, PDL and overlap enabled.

The agentic workload used a frozen SWE-Smith dataset at concurrency one with
CUDA Graph capture size one. Each of two rounds flushed the request cache,
warmed up two conversations, then measured four conversations: 100 measured
turns in total, each producing 500 tokens. All turns completed. The combined
mean TPOT was 1.581 ms, decode throughput was 632.6 tokens/s, mean TTFT was
1.282 s and end-to-end output throughput was 241.4 tokens/s. Decode throughput
is the reciprocal of the raw mean TPOT; end-to-end throughput divides total
output tokens by total measurement time.

Repeated GSM8K evaluation used the full 1,319-question test split, four-shot
prompts, temperature zero, seed 42, thinking disabled and a 1,024-token output
limit. Each round flushed the request cache and used fresh predictions with
request retries disabled. CUDA Graph capture sizes were expanded to
1/2/4/8/16 to cover concurrent graph execution.

| Client concurrency | Correct / total | Accuracy | Output-limit truncations |
| --- | ---: | ---: | ---: |
| 4 | 1278 / 1319 | 96.89% | 12 |
| 16 | 1274 / 1319 | 96.59% | 20 |
| 32 | 1273 / 1319 | 96.51% | 20 |

All 3,957 requests completed, producing 1,286,381 output tokens, with no API
errors, empty outputs or CUDA errors. The service admitted at most sixteen
active requests; client concurrency 32 also exercised queueing. Truncated
answers remain included in the accuracy scores. These serving measurements
validate the current implementation under load; they do not isolate the
performance or accuracy effect of PDL from other implementation choices.
