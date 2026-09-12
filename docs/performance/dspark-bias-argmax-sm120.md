# SM120 DSpark bias-argmax candidate

This experimental candidate fuses the per-step bias postprocessing used by the
DSpark proposal consumer at TokenSpeed revision
`7978c48b5c56445c600880f0ff275a611e584cef`. It does not change the Markov weights,
proposal width, token semantics, verification, or scheduler. The completed C9 validation is summarized below. The proposal region improves,
but complete Engine eager latency regresses, so this candidate remains a draft.

## Consumer and numerical contract

`DSpark._sample_block` already computes the block base-logit projection in one
GEMM. Each subsequent position embeds the previous token, projects its Markov
bias, and passes a positive-strided `next_tokens[:, position]` output view into
`DFlash._greedy_argmax_vocab_parallel`. The candidate preserves these projections and
their dependencies. It replaces the native bias cast, addition,
maximum/index reduction, vocabulary offset, and output copy with two Triton
launches: vocabulary-tile partials and a final strided-output reduction.

For logits dtype `d`, the exact comparison scores are
`round_d(logits + round_d(bias))`. Both rounding boundaries are explicit. NaNs
win at the first index, equal maxima choose the smallest index, and signed
zeros/infinities follow the original Torch result. The kernel uses int64
address arithmetic and supports int32 or int64 outputs with checked global-ID
bounds. It does not use the generic sampling argmax API, whose NaN policy is
different.

For example, fp16 logits `[1, 1]` and fp32 bias `[0.0003, 0.0004]` both round to
score 1 and select index 0; omitting the result rounding selects index 1.
Separately, logits `[-1, -1]` and bias `[1.0002, 1.0003]` select index 0 after
bias conversion, while omitting that conversion leaves distinct small results
after the addition rounding and selects index 1.

## Scope and state lifetime

The consumer enables this candidate only for SM120, TP1, a nonempty original
vocabulary shard with no added-vocabulary section, no distributed argmax state,
a Markov bias, and a supplied output. The current CUDA device must match the
inputs. Vocabulary sizes 1 through 262144 are the experimental scope, not an
upstream vocabulary limit. FP16, BF16, and FP32 logits and bias may use different
dtypes and positive strides. The output may have any positive stride.

Scratch is allocated in `wire_target`, after the real head has been bound and
target wiring checks have succeeded, using the full input-buffer batch
capacity and real vocabulary width. `_init_native_buffers` only initializes the
empty workspace state because the target head is not yet bound there. All
capture buckets reuse this storage sequentially. Different streams need separate
workspaces or explicit ordering. The eager and CUDA Graph consumer use the same
implementation.

Unsupported metadata returns `False` before selection or writes. Checks cover
workspace dtype/shape/contiguity, device, offsets, nonpositive strides, nested
or sparse tensors, unresolved negative/conjugate views, and conservative byte
span overlap between every write and all other read/write regions. Read/read
aliasing is legal. Inputs must be ordinary strided Torch tensors, including
parameters; custom Tensor subclasses that redefine storage/dispatch semantics
are outside this experimental scope.

The original collective probe stays before every rank-local branch. The bias
closure executes once; a rejected candidate reuses that computed bias in the
original fallback. `out=None` retains the original fresh-result behavior.
Unsupported hardware, TP, added vocabulary, and distributed state use the
existing path.

## Launcher reuse

The first call uses the normal Triton JIT. Later matching calls may reuse the
official `CompiledKernel[grid]` runners, passing every current tensor and
constant again. The cache retains code and grid, not input pointers or streams.
Keys cover device, dtypes, full strides/constants, grids, kernel source, and
all five tensor pointer residues modulo 16. Each alignment class first passes
through normal JIT specialization, including unaligned output columns.

Runtime launch hooks remain active through the official compiled interface.
Debugging, stage inspection, instrumentation, or JIT pre-run hooks bypass the
runner cache. Interpreter mode and an active `jit_cache_hook` reject the
candidate before either launch: a hook may suppress compilation on a cold
miss, so finalizing old partials would be incorrect. The original Torch consumer
still provides the fallback in those modes.

## Verification and timing harness

Kernel tests cover all nine input dtype pairs, vocabulary tails and size
bounds, NaNs/ties/infinities, both rounding witnesses, strided reads/writes,
offset boundaries, malformed/alias metadata, pointer/alignment cache reuse,
streams, changing graph inputs, shared capture-bucket scratch, and hooks.
Consumer tests execute unchanged source methods with explicit dependency
fixtures and fixed original source snapshots. The CPU configuration injects
an unavailable-kernel dependency without importing GPU packages or fabricating
capabilities; it only verifies host branches and workspace wiring.

`tokenspeed-kernel/benchmarks/bench_dspark_bias_argmax_sm120.py` requires an
explicit directory containing original `dflash.py`, `dspark.py`, `nvtx.py`, and
`model_dspark.py` from the fixed baseline. Each complete file has a checked SHA.
The support loader selects unchanged source methods through AST and preserves
the original disabled NVTX wrapper. Heavy model construction and unrelated
imports are replaced by declared boundary fixtures; this is not a model run.

The benchmark has two independent timing regions:

- `postprocess_precomputed_bias`: the complete actual DFlash method with the
  same precomputed base logits and bias in both arms; neither projection GEMM
  is in the timed region.
- `proposal_walk`: the complete actual DSpark `_sample_block`, including anchor
  copy, block base GEMM and layout, previous-token clamp/embedding, every Markov
  GEMM and DFlash call, and final output clamp. `N` includes the anchor.

Each shape/dtype/region runs eager and CUDA Graph separately, with independent
but equal arm buffers/weights/graphs, warmup outside timing, profiler path
checks outside timing, and random complete APPA/PAAP blocks. The primary
latency reduction is `100 * (1 - exp(mean(block_log_ratio)))`, where each block
ratio is `mean(log(P1), log(P2)) - mean(log(A1), log(A2))`. Keep runs and regions
separate; these descriptive samples do not establish model throughput.

Raw evidence files are created exclusively and never overwrite prior or
partial runs. Failure appends `completion.success=false` with error details.
Success requires full record counts, exact output equivalence, unchanged
immutable inputs/parameters and source hashes, and the expected two-stage
kernel path. Evidence includes GPU UUID, logical CUDA device/CVD, software
versions, source paths/hashes, tensor hashes/shapes/strides/dtypes/pointers,
and all block positions. Proposal base logits and bias are temporary results
inside each actual method call; the precomputed group additionally hashes those
persistent input buffers.

The preregistered primary synthetic-weight geometry follows the released
`deepseek-ai/dspark_qwen3_4b_block7` configuration (revision
`3457dff1417cb84927f6098a5fcb7cee85c934b7`) paired with `Qwen/Qwen3-4B`
(revision `1cfa9a7208912126459214e8b04321603b3df60c`): vocabulary 151936,
hidden width 2560, Markov rank 256, BF16, and seven draft positions plus one
anchor (`N=8`). Batch sizes are 1, 8, and 16. Primary timing uses eight
APPA/PAAP blocks, 200 iterations per arm, and five capture warmup calls. All
primary shapes, both regions, and both modes must be retained, including flat
or slower results. Small/tail vocabulary shapes primarily test correctness.
This geometry uses generated parameters and does not claim to load those
checkpoint weights or to establish compatibility of a complete model run.

## C9 validation and limitations

The frozen GPU run passed 120 regression tests with zero skips, 324 numerical boundary rows, and all 384 timing arms across 12 paths. These are source-method/region measurements with generated weights, not complete-model speedups.

| Batch | Postprocess eager reduction | Postprocess Graph reduction | Proposal walk eager reduction | Proposal walk Graph reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 59.863% | 60.455% | 1.984% | 2.090% |
| 8 | 59.526% | 64.919% | 3.708% | 3.658% |
| 16 | 59.996% | 69.781% | 4.001% | 3.698% |

The final complete Qwen3-4B target plus DSpark eager validation used four fresh processes in candidate/baseline/baseline/candidate order, batches 1/4/8, fixed 128-token outputs, two warmups, and seven measurements per process. Both arms include the same prerequisite runtime fixes. All ordered token IDs, text, finish reasons, and speculative-work counts matched exactly. Source, native-library, process-role and eager-path audits passed.

Complete Engine eager latency increased **3.168% / 3.825% / 3.807%** for batches 1/4/8. These are descriptive ratios of geometric means of process medians; no confidence interval or general-model speedup is claimed. The two candidate processes varied. Earlier timings affected by shared-host overlap were rejected and are not included. No complete Engine Graph improvement is attributed to C9.

C9 releases the GIL only during its two CUDA driver calls, retaining the vendor launcher's thread-interleaving contract and all live metadata/instrumentation checks. An absent or ABI-incompatible optional extension retains the original fallback. CUDA wheel packaging and broad hardware/runtime support require further validation. The limited route gains do not justify default promotion.
