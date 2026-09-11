# Hyperconnection mix

Gated-residual mix has one public execution path for eager calls, CUDA graphs,
prefill, decode and speculative verification. Backend selection depends on
shape, device capacity and the caller's weight-readiness contract.

`gated_residual_mix` requires `weights_independent` explicitly. Model callers
pass True only when both projection weights are already ready and remain
immutable within forward. Weight loading and online reload happen outside
forward; normalization produces only the activation. Generic callers pass
False when a preceding PDL kernel may write either weight.

For HC=4, hidden size 2560 and rank 320, automatic selection uses:

| Backend | Requirements beyond the common BF16/FP16 shape contract |
| --- | --- |
| Single fused CuTe | Blackwell, T=1–16, independent weights, contiguous 16-byte-aligned operands, and capacity for six resident 16-CTA clusters |
| Two-GEMM CuTe | Blackwell, T=1–8, PDL enabled, contiguous 32-byte-aligned inputs, and prepared up weights |
| Persistent Triton | Supported small shapes and non-deterministic dispatch |
| General GEMM/epilogue | Remaining supported shapes |

The same policy handles verification batches above 16 rows through the general
fallback. Explicit backend overrides support comparisons: fused CuTe accepts
PDL off and conservative weight loading, and two-GEMM CuTe accepts T up to 32
and PDL off. HC weights are replicated across TP ranks; tensor parallelism
does not divide the 10240-wide input or these weights.

## Single fused CuTe kernel

The CTA has 192 threads with the following roles:

| Warp | Work |
| --- | --- |
| 0 | Down/Up activation TMA, then epilogue |
| 1–3 | Epilogue |
| 4 | Barrier initialization and unified Down/Up weight TMA |
| 5 | TMEM allocation, incoming PDL wait, both MMA chains, projection publication and TMEM retirement |

Warp 5 passes activation readiness through a local barrier. Independent
weights can load before activation readiness and before Down completes.
With the conservative explicit override, warp 4 also waits for the incoming
PDL producer. All 128 epilogue threads participate in shared barriers and
TMEM read acknowledgements.

Both projections use native `tcgen05.mma.cta_group::1.kind::f16` with BF16/FP16
operands and FP32 accumulation. Down uses M64 and split-K=16: each CTA owns
640 of the 10240 input channels, and each cluster owns 64 projection columns.
Projection rows 320/324 require five/six clusters, or 80/96 CTAs. The first
80 CTAs each compute an M128 Up tile, covering four branches for 32 hidden
positions. Every launched CTA participates in Down; the optional sixth
cluster produces inject logits and does not compute Up.

The kernel requests 227 KiB opt-in SMEM per block. A current-stream cluster
capacity check requires six resident clusters, including under green contexts.
A cooperative launch enforces the residency needed for the cross-cluster
handoff, also with concurrent streams. Smaller device partitions fall back.

### Operand pipeline

For all supported token counts, Down has five K128 weight/activation buffers
and Up has three K128 buffers. MMA N is 8 for T<=8 and 16 otherwise. The
weight warp issues the initial Down loads and all Up loads before any wait
for a recycled Down slot.

Both producers arrive on a stage's `full` mbarrier and account for their
transaction bytes. MMA waits only for the current stage's weight and activation
transfers. It can start Down's first eight K16 instructions without waiting
for the remaining four buffers. Up likewise has no whole-weight barrier.
Its final K128 block contains 64 useful channels, so Down and Up issue 40
and 20 native K16 MMAs respectively.

When GEMM K fits the allocated buffers without overwriting a stage, compilation
omits the unused `empty` initialization, producer waits and per-stage completion
commits. Down checks `k_tiles <= down_stages`; the supported Up K320 fits three
K128 buffers. Eligibility depends on K and buffer capacity, not token count.

With more Down K tiles than buffers, the same loops retain the full `empty`
protocol: MMA commits stage completion, and producers wait only for their
destination slot. Initial empty phases are one; subsequent wraps toggle the
phase. Both cases retain final `down_done`/`up_done` commits, reader
acknowledgements and the same buffer/barrier address layout.

### Distributed reduction

Each cluster rank `r` reduces the four projection columns `[4r,4r+4)`, or
`[T,4]` valid elements. Epilogue threads store their TMEM partial in a local
FP32 tile with a 68-element row stride, then scatter aligned 16-byte vectors
using `st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32`.
Adjacent lanes cover consecutive token rows at the same destination.

Each receiver has a mailbox `[16 source ranks,N,4]` and waits for the expected
transaction bytes. The mailbox occupies 2/4 KiB for N8/N16; the padded source
tile occupies 2.125/4.25 KiB. Transfers include the padded N rows. Each reducer
sums ranks 0 through 15 in fixed FP32 order, applies scale and SiLU after the
complete sum, converts once to BF16/FP16, and stores activation. Inject columns
store scaled raw projections to the invocation's inject output. Fixed-order
reduction supports deterministic dispatch.

Scratch belongs to `(device, CUDA stream, projection rows)`: opaque int16
`[16,320]` storage viewed as the invocation's dtype, plus an int64 epoch per
cluster. It occupies 10 KiB plus 40/48 bytes. Ordered mixed-dtype calls reuse
this workspace and overwrite every consumed activation. Up reads completed
activation through TMA; channels 320–383 are zero-filled from the tensor bound.
Padded token rows do not contribute to valid outputs.

### Publication and retirement

Each activation writer executes an async-global proxy fence after storing.
The 128-thread epilogue acknowledgement orders these writes before warp 5
publishes a release-cluster arrival to rank 0. All sixteen CTAs arrive;
rank 0 waits with cluster acquire semantics and writes its own epoch with
release-GPU ordering. Clusters do not contend on a global atomic counter.

Each compute warp checks all cluster epochs with separate lanes and GPU
acquire loads. A warp memory barrier combines that visibility before the
elected thread signals the activation producer. Up activation TMA uses the
same per-stage full/empty protocol as weight TMA. This chain retains the
CTA, cluster, GPU and async-proxy visibility required by the
[PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar).

Each CTA reads its cluster's previous epoch after its incoming PDL wait and
targets `epoch+1`. An owner cannot publish until all ranks have read that
epoch and completed their projections. Eager and graph execution share this
device protocol without a reset, scratch clearing pass or host generation.
All clusters have completed their DSM receives before any CTA passes the
global handoff, preserving remote SMEM lifetime. Every epilogue reader
acknowledges completion before TMEM reuse or deallocation.

Outgoing PDL is triggered after the Down epilogue acknowledgement, before
cluster publication and Up computation. Successors must still wait before
consuming outputs; the trigger does not make the outputs ready.

## Two-GEMM CuTe fallback

This backend has separate weight DMA, activation DMA, MMA and epilogue warps.
Both GEMMs use native Blackwell tensor-core instructions, and Down reduces
split-K partials within the cluster in FP32. Only the HC Down/inject and Up/gate
epilogues are implemented; the wrapper selects explicit tactics.

Down waits for preceding input and weight writes in both DMA warps. It triggers
Up after issuing its weight loads, while computation may still be running.
Up weights can load independently because Down never writes them; its
activation DMA warp waits for Down. Down's incoming wait also orders ancestor
writes to the prepared up weight before Up's early loads.

Down writes SiLU activation for columns 0–319, optional inject logits to a
separate output, and zero padding through rank 384. Up fuses sigmoid, branch
weighting and the four-branch sum. Every invocation owns its activation and
outputs, so concurrent streams and consecutive graph calls retain their
results. This backend needs no global grid barrier or generation counter.

`prepare_gated_residual_weight_cache` pads Up to rank 384 and reorders rows as
`hidden_position * 4 + branch`. Preparation runs in the runtime weight loader;
online reload refreshes the same allocation outside graph capture. Forward
only looks up the prepared tensor. Compilation is cached by device, dtype,
tactic, epilogue, PDL and scale.

DLPack exports detached views: parameter views can retain `requires_grad` in
inference contexts, while DLPack rejects such exports. Detaching shares storage
and preserves the source parameter's flags and graph-visible reload address.
CuTe remains optional, with implementation under `thirdparty/` and registration
through `ops/residual/`; runtime imports only `tokenspeed-kernel`. Mix and
combine use the shared `residual` registry family, under `hyperconnection_mix`
and `hyperconnection_combine` respectively.

## Persistent Triton fallback

Its workspace is keyed by device, stream and projection rows and contains two
FP32 projection buffers plus int64 cumulative-arrival and generation counters.
The 324-column projection uses a 352-element row stride to keep 32-column
atomic tiles in aligned 128-byte segments. The buffers occupy 44 KiB, plus
16 bytes of counters. Only valid token rows issue FP32 atomics; padding is
cleared so different token counts can reuse the workspace.

Both buffers and counters start at zero. For generation `g`, each CTA:

1. Waits for its PDL producer before any workspace access.
2. Accumulates into buffer `g & 1` and clears disjoint slices of the other
   buffer for the next launch.
3. Publishes writes with a GPU `acq_rel` arrival atomic and polls with acquire
   loads until arrivals reach `(g+1) * num_ctas`.
4. Consumes the projection; CTA zero advances the generation after the barrier.

Arrival counts stay monotonic, so late pollers can observe completion.
The following launch still waits for its stream/PDL dependency before touching
scratch. Grid size is fixed to the device SM count and must remain resident;
changing it requires a new workspace or a revised counter contract. Generation
selection happens on device for both eager and graph execution, including odd
numbers of calls. FP32 atomic accumulation excludes deterministic dispatch.

With PDL on SM90+, each CTA issues bounded 48 KiB bulk L2 prefetch hints for
Down/Up weights before waiting. These are cache hints; all actual operand
loads and workspace accesses remain after `gdc_wait`. Unaligned weight views
skip the hint. Mix triggers successors after projection work, before the grid
barrier and Up. Combine triggers after its incoming wait so the following
normalization can prepare. Consumers still wait before reading outputs.
See NVIDIA's [PDL execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
and [bulk prefetch instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch).

## Validation and benchmarking

`tokenspeed-kernel/test/ops/test_hyperconnection.py` covers FP64 reference
parity, BF16/FP16, token tails, inject and scale, PDL on/off, dispatch alignment
and capacity, determinism, weight preparation/reload, and parameter views.
State tests cover changing shapes/dtypes, concurrent eager/graph streams,
retained outputs, odd/even graph replay counts, PDL input/weight producers,
and epochs crossing `2**32`. Sparse K128 tests update the first/last channel of
every stage between graph replays, with both five buffers and a two-buffer
ring that must recycle and wrap phases. Opposing split partials check that
SiLU follows the complete reduction. Runtime tests check the weight contract
and kernel-package boundary.

Run the graph benchmark from the repository root:

```bash
PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/test/ops/bench_hyperconnection.py \
  --backend cute_fused --operation mix --rows 1,4,8,16 --mode graph --pdl on

PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/test/ops/bench_hyperconnection.py \
  --backend default --operation chain --rows 1,4,8,16 --mode graph \
  --weight-banks 32 --pdl on
```

Use `--backend cute_dsl` or `--backend persistent` to measure the fallbacks,
`--pdl off` to compare dependency scheduling, and `--dtype fp16` for FP16.
The benchmark skips unsupported shapes, prepares weights before timing and
warms the capture stream. Batched graph calls avoid Python replay gaps.
The chain feeds each combined output into the next normalization.

Compare implementations on the same GPU with the same timing and cache
conditions. One weight pair measures resident reuse; 32 pairs occupy about
403 MiB and exceed B300 L2. Weight rotation and explicit cache eviction are
different workloads. Graph/node profiling overhead and model dependencies
also affect latency; kernel or chain timing alone is not service throughput.

## Gated residual normalization

Qwen4-Exp carries residual streams shaped `[tokens, branches * hidden_size]`.
Each branch is independently Gemma RMS-normalized over `hidden_size`, with
effective per-feature scale `1 + weight`. Mixing uses the normalized streams;
sublayer injection updates the unnormalized residual streams.

### Adjacent combine and norm

When one sublayer's combine directly precedes the next mixer, the next mixer
accepts the previous sublayer output and injection logits together with the
original residual. `tokenspeed-kernel.gated_residual_combine_norm` computes:

```text
updated[t, g, :] = residual[t, g, :]
                  + output[t, :] * 2 * sigmoid(inject_logits[t, g])
normalized[t, g, :] = GemmaRMSNorm(updated[t, g, :], next_norm_weight[g, :])
```

Both tensors are outputs. The next sublayer's combine needs `updated`, so
normalization must not overwrite or replace that residual. The normalizer's
weight belongs to the consuming mixer, while the injection logits belong to
the preceding mixer. Shared branch weights remain supported.

The grouped RMSNorm Triton kernel owns both the ordinary norm and this optional
combine prologue. Its grid is `(tokens, branches)`; each CTA handles a full
branch with `BLOCK = next_power_of_2(hidden_size)`. The wrapper ensures contiguous
storage, so the kernel derives row offsets directly from the tensor widths.
Combine, normalization and both stores occur within that CTA, without a second
launch or a reload of the
updated residual. Masked lanes do not participate in the mean square.

The updated residual is rounded to its storage dtype **before** computing the
RMS statistics. This preserves the BF16/FP16 rounding boundary of separate
combine and norm kernels; retaining the FP32 sum through normalization would
change model numerics. FP32 arithmetic and the `1 + weight` scale otherwise
match standalone grouped Gemma RMSNorm.

### Execution boundaries

The shared decoder code fuses attention combine into MLP normalization after
attention output communication and residual-row alignment. GDN, full attention
and the MTP draft use the same path, including idle and CUDA graph execution.
No normalized tensors are cached across forwards.

`preload_residual=True` explicitly promises that both the residual and norm
weight are already visible before the current PDL producer begins. The fused
kernel loads them before `gdc_wait`; block output and injection logits remain
after the wait. Attention-to-MLP can make this promise because attention has
already consumed the residual, communication preserves it or changes its row
view, and a newly gathered residual is consumed by `norm_for` before injection.
The MTP row selection likewise materializes its residual before its norm and
gate selections. Normal grouped RMSNorm waits before reading its activation.
If the kernel wrapper must copy a noncontiguous residual or weight, it disables
preloading for that invocation so the copy's output is not read prematurely.

The MLP returns a forward-local `GatedResidualUpdate` holding the original
residual, block output and injection logits. When the next attention has no PLE
or row gather, its mixer consumes this update through the same fused
combine-and-norm kernel. The final output mixer consumes the last MLP update
the same way and returns the fused kernel's updated, unnormalized residual as
the HC hidden state required by MTP. No update escapes the model forward or is
stored on a module; eager, prefill, decode, idle and CUDA graphs share this path.

Operations between injection and normalization still consume a materialized
residual: the model resolves the update before a multimodal deepstack addition,
and attention preparation resolves it before PLE or a row all-gather. Final
row gathering likewise resolves it before communication. `CommManager` owns
the gather predicates used by both communication and these fusion boundaries.
Normalization is never moved ahead of an operation that changes its input.
In the TP4 all-reduce text path, MLP tails without an intervening PLE use the
fused kernel, including the one-layer MTP draft's final mixer. Residual
preloading is valid because the MLP has already consumed those streams before
producing its output.

### Validation

`tokenspeed-kernel/test/ops/test_hyperconnection.py` compares fused and separate
kernels for BF16, FP16 and FP32, shared/per-branch weights, strided inputs,
non-power-of-two widths, empty inputs, and CUDA graph replay with PDL on/off.
An intentionally delayed producer and poisoned output buffers check that
activation loads remain behind the PDL wait.
`test/runtime/test_hyperconnection_kernel_boundary.py` checks consumer norm
weights, subsequent residual injection, attention-to-MLP row alignment, MLP
tails across PLE/deepstack/gather boundaries, final HC hidden states, and graph
replay with changing inputs. Communication predicate tests keep the fusion
boundaries aligned with the actual all-gather decisions.

Compare separate combine and norm with the fused kernel, with and without
preloading:

```bash
PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/test/ops/bench_hyperconnection.py \
  --operation combine_norm --dtype bf16 --rows 1,4,16,128,2048 --mode graph
```

Graph timings capture repeated operations in one graph to amortize host launch
overhead and report the median of five device-event measurements. These are
combine-plus-norm timings, not end-to-end model throughput.
