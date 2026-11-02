# GDN kernels

## CuTe DSL WY verification

`gdn_decode_mtp` selects the output-only TF32 MMA kernel for ReplaySSM through
the normal registry. It accepts BF16/FP16 Q/K/V with K=V=128, grouped value
heads, BF16/FP16/FP32 K-last initial states, and 1–16 tokens. Chains and trees
share the same WY computation; the facade's existing tree workspace contract
restricts its output-only trees to 8–16 nodes.

The registration constrains platform, dtype, head dimensions, token count and
output-only execution; its specialized priority wins over general recurrence
on supported NVIDIA geometry, for both BF16 and FP32 state. Tree support is a
trait, so chain-only kernels are excluded without forcing a solution. Explicit
`solution` and `override` requests still apply. Cache-pool capability queries
use the same execution traits and state dtype when checking for packed
projection views. Output-only chains use this same registry entry, including
the Triton chunked-chain candidate described below.

Automatic BF16 state-writing verify and ordinary decode require the
`padding_safe` trait, so the selected kernel preserves the reserved null page.
The operator owns this policy; runtime code supplies no solution choice.
Explicit solutions retain their documented padding contract. Read-only verify
uses normal selection, including fallback kernels for
unsupported WY geometry. It may read the reserved null page for padding, but
never writes it; padded outputs are ignored.

The leaf reads the committed state once per value tile, forms the small WY
output transform and computes every token's output without writing the state
pool or materializing node states. An optional payload receives exact raw
`[k | v | a | b]` rows for the existing replay commit. State-writing options
are rejected. Q/K/V and gates may be strided; state-pool rows may have padding.

Products use native `mma.sync.m16n8k8` with FP32 accumulation. FP32 operands
round to TF32; expanded BF16/FP16 state values already fit TF32 exactly and
preserve their bits. Normalization uses native FP32 reciprocal square root,
and gates and WY coefficients stay FP32. For four tokens, backward query propagation
produces the same initial-state coefficient
and value weights without forming Gram matrices or running a triangular solve.
Tests compare against an independent FP64 token recurrence at atol=1e-4,
rtol=2^-7, including strong path gates and unit-scale states. This leaf makes
no bitwise-numerics claim.

Token bounds are bucketed to 4/8/16; MMA reductions use at least eight elements.
Short windows transpose the products to put values along the 16-row MMA axis.
Their Q/K/V workspaces store eight token rows. Eight-token windows form the
two 8-by-8 Gram matrices together as `[K; Q] @ K.T` in one warp, while two
other warps compute segment decays. Their coefficient workspaces contain only
the eight useful rows; larger windows retain 16-row workspaces.
Value tiles use 64 or 128 columns according to window length, state dtype and
the coarse number of request/head pairs. Eight-token windows with at most 64
request/head pairs interleave two FP32 accumulators within each 128-element
TF32 reduction. Other shapes retain one accumulator. This bounded choice is
part of the compilation key; batch and token counts themselves are runtime.
Actual batch size, token count, strides and scale remain runtime arguments.
Warm each selected bucket/dtype/model geometry before CUDA Graph capture;
replays consume updated inputs, parameters, state rows and ancestor masks.
The kernel supports PDL and keeps PDL compilation variants separate.

Aligned Q/K/V tiles load through TMA into raw shared-memory buffers before
FP32 normalization and value staging. This requires unit inner strides,
16-byte-aligned base addresses and outer byte strides divisible by 16.
Other layouts retain four-element vector reads when each physical row has
unit inner stride and eight-byte alignment, with scalar reads otherwise.
Normalization and raw payload stores reuse the loaded values. Four-token
64-value tiles issue gate and parameter reads before Q/K normalization.

Checkpoint tiles retain their source dtype and padded K-last shared layout.
Aligned pools with a 16-byte-aligned physical row stride use TMA. The transfer
box includes the row padding beyond K=128, which TMA zero-fills without
reading past a global row. Odd pool strides and unaligned bases retain the
per-row 16-byte asynchronous or scalar copies. Negative indices zero-fill
the tile in either transport. Checkpoint completion is independent of Q/K/V
completion, so coefficient preparation overlaps the state transfer.

BF16 four-token windows with 256–1024 request/head pairs stream the checkpoint
through one shared buffer in two 64-key stages. This reduces shared-memory
use without shrinking the 128-value output tile. The second transfer overlaps
the value product, and the second state product continues the first stage's
accumulator. A CTA barrier protects buffer reuse, and each transfer has its
own completion phase. TMA row padding rereads eight keys in the first stage
and zero-fills beyond the final stage; other tiers stage all 128 keys at once.

Warmup compiles all four combinations of Q/K/V and checkpoint transport.
Graph capture can therefore refresh to a different supported alignment or
stride without compiling inside capture. Descriptor extents and strides stay
runtime values; the transport choices are bounded compilation variants.
Host descriptor construction and warmup are outside the GPU kernel timing.

TF32 products share one accumulation loop for both one- and two-accumulator
variants, preserving their reduction order. Four-token windows allocate only
the backward-propagation workspace; larger windows initialize the Gram tiles
directly through MMA.

Performance validation on SM103 compares the output-only leaf with the
chunked Triton chain leaf for Qwen3.8-Flash-Next TP1 (H/HV=16/48) and
Qwen3.8-27B TP4 (H/HV=4/12), with BF16 inputs/state, B=1–512 and T=3/4/8.
The 64 conditions cover warm/cold state pools and PDL on/off. CUDA Graph
measurements exclude host descriptor construction, convolution and replay
commit; they measure the leaf rather than end-to-end model latency.

## FlashInfer PDL adapters

The FlashInfer adapter retains the upstream launch geometry and runtime ABI,
while wrapping device bodies with PDL synchronization. Adapted functions and
in-memory compilation caches live in private namespaces.

Shared CuTe adaptation lives in `thirdparty/flashinfer/gdn_pdl.py`; vendor
adapters do not import the op registration package. This keeps direct adapter
imports independent of the registry's initialization order.

The MTP registration declares K=V=128, the geometry supported by both its BF16
and FP32 state paths. Other value widths use portable verification. Direct
BF16-state calls outside that geometry are rejected before the FP32-only
adapter. Runtime tests cover non-square states, padding and accepted-prefix
commit; WY tests cover 4/8/16-token wide value tiles with BF16 and FP16 state.

Decode and prefill explicitly select FlashInfer's CuTe backend. FlashInfer
0.7.0's automatic backend selection can use Cake GDN for supported shapes,
which bypasses the PDL-wrapped device bodies. Regression tests cover both
power-of-two and other head groupings and reject calls into that alternate path.

FlashInfer 0.7.0 also persists CuTe-DSL kernels to disk. PDL artifacts use a
separate `tokenspeed_pdl_` module namespace so an ordinary kernel cannot satisfy
a PDL cache lookup, or vice versa. Cache invalidation includes all upstream
source files plus the local PDL and GDN adapter sources. Installed upstream kernel functions and shared compilation caches remain
untouched. Reapplying the API decorator registers the private entry point with
FlashInfer's normal trace metadata.

GPU regression tests cover persistent cache hits after clearing the in-memory
cache, invalidation after source changes, and both launch orders. Runtime GDN
tests additionally check CUDA Graph dependency edges and exact replay results
for decode, MTP, BF16 state, and prefill.

## Rubin FlashInfer

The pinned FlashInfer Python/cubin release is `0.7.2.post1`, which includes
both Rubin entry points. Registrations live in `flashinfer.py`; the explicit
`flashinfer_rubin` decode solution remains available.

The Rubin adapters check the release version, API parameters, and SHA256 of
the upstream sources before registration. Source checks cover the decode CTA
mapping used by the padding guard and the prefill dispatch/decorator contract.
Missing or modified sources leave the portable kernels available; a direct
adapter call reports the compatibility error. Updating the accepted sources
requires reviewing these contracts and repeating GPU validation.

SM107 prefill uses FlashInfer's Rubin chunked implementation when it is
installed. The existing K-last state, device sequence bounds and in-forward
checkpoint contracts are unchanged. The PDL adapter wraps the upstream
variant table and reapplies the public API decorator, retaining logging, tensor
dumps, trace generation and the input validation inside the API body.

`flashinfer_rubin_gdn_decode_step` (`flashinfer_rubin`) calls the installed
PR6604 T=1 kernel for BF16 input/state and K=V=128. It reads aligned projection
and page-strided state views directly. Unaligned input views are copied to
meet the upstream DLPack boundary's 32-byte base alignment. The upstream kernel
redirects negative indices to row 0; TokenSpeed must preserve that null state, so a uniform CTA
guard skips negative read or write rows before entering the upstream body.
PDL waits precede those index reads and every CTA signals, including padding.
Masked and PDL variants have independent memory and disk compilation caches.
No upstream kernel source or separate state pool is copied into TokenSpeed.

Automatic decode and state-writing verify selection includes the bound state
dtype. The generic FlashInfer BF16 path is excluded there because its padding
writes the null page; the masked Rubin decode and portable verify retain the
ordinary cache arena. Read-only verifies may still select the generic BF16
path, which only reads the null page for padding (see the padding policy
above). Explicit legacy `solution="flashinfer"` calls retain their documented
padding semantics; other named solutions still validate the supplied state
dtype for state-writing calls.
Missing Rubin APIs leave the existing portable implementations
available. The unmerged Rubin MTP implementation and uGPU-localized allocations
are not part of this adapter.

To compare the paths in the same environment, use the normal registry overrides:
`TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_GDN_CHUNK_PREFILL=triton_gdn_chunk_prefill`
and `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_GDN_DECODE_STEP=triton_gdn_decode_step`
select the portable baseline. Choose `flashinfer_gdn_chunk_prefill` and
`flashinfer_rubin_gdn_decode_step` respectively to force the Rubin pair.

The decode and MTP benchmark generators require an explicit `state_dtype`
(`float32` or `bfloat16`) alongside the BF16 input `dtype`. The selected state
dtype is used for dispatch, pool allocation, and the recorded parameters; use
the same value for both sides of a comparison. Rubin T=1 decode requires
`state_dtype="bfloat16"`; requesting FP32 with `solution="flashinfer_rubin"`
is rejected as not applicable during preparation. Prefill benchmarks retain
their FP32 state contract.

Prefill implementations have different intermediate rounding. Both normalize
Q/K through the same FP32 reduction and round them to their input dtype before
the scan. Triton applies the output scale after the intra-chunk product;
FlashInfer scales the QK scores before storing their low-precision intermediate.
Even a single token with zero initial state can therefore differ bitwise.
Compare full outputs and recurrent states against an independent recurrence;
matching one implementation is not a substitute for that check. Kernel-level
tolerances do not establish equivalent generated sequences or model quality.

## Chain verification

A ReplaySSM chain verify reads the committed state and writes none. The runtime
always calls `gdn_decode_mtp`, passing `tree_ancestors=None` for a chain and the
actual ancestor masks for a tree. `select_kernel` ranks CuTe WY, Triton's
chunked-chain candidate and the general MTP kernels under that one contract.
Explicit `solution` and `override` requests use the same selection.

The `chunked_chain` trait describes the Triton candidate's supported input:
an output-only 2–16-token Blackwell chain, BF16 state, equal power-of-two head
dimensions from 64, contiguous heads and request-major rows. Packed projection
views satisfy that layout; arbitrary inner strides do not. A replay payload
destination excludes this candidate. CuTe WY has higher priority for its
supported 128-by-128 geometry; unsupported inputs retain normal fallback.
State-writing and tree verifies never select the chain candidate.

The general Triton fallback may use its tree computation with a cached
one-path mask for 13–16-token chains with at least 24 states. This policy and
the constant masks belong to the kernel package. The direct `gdn_chain_verify`
API remains available for callers explicitly requesting its chain contract.

The chain form computes a request's whole window from the committed state in
one program per (value tile, request, value head), without materializing a
state: the chain case of the tree-chunked kernel. With
`D[t, s] = exp(g_{s+1} + ... + g_t)`, the updates solve
`(I + b D_strict k k^T) U = b (v - exp(G) k h0)` and row t's output is
`exp(G_t) q_t h0 + sum_{s<=t} D[t, s] (q_t . k_s) U_s`. Its rows come in a
power-of-two bucket of the chain's length (at least 4); the products with the
state and the key/query Gram matrices run on tensor cores in the state dtype,
with the normalized key's rounding residual folded back in; the window algebra
stays fp32 (dots from 8 rows, outer products below) and decays sum each
segment's gates directly. Outputs round differently from the step kernels; the
pool is never written. The row count and the gates' layout (alignment, row
stride) are not specialized, and the launch (value tile, key splits, warps,
pipeline stages) follows three measured tiers of request-heads per row bucket
(`_CHAIN_VERIFY_LAUNCH_TIERS`), so the kernel compiles once per bucket and
tier. During the normal startup forward under
`ops.tuning.kernel_warmup(enabled=True)`, the selected implementation compiles
its reachable variants. CuTe prepares its launch buckets and transports;
Triton uses JIT warmup for its launch tiers with the actual operand types,
alignments and strides. The general fallback prepares both step and tree forms
when a chain can cross their batch-size threshold, including when startup warms
only the largest batch. These additional variants are compiled without
launching on caller buffers. Runtime workspace allocation performs no
kernel-specific warmup, and serving calls reuse the prepared kernels.
Tests cover automatic selection, explicit overrides, exact accepted-prefix
commit under graph replay and no compilation at serving batch sizes.

## Variable prefill lengths

The Triton L2-normalization fallback uses a runtime token count. The launch
grid and masked block pointers cover the exact rows, while head width and tile
size stay compile-time parameters. Consecutive agentic turns therefore reuse
one compiled kernel across different token counts instead of compiling a new
variant for each length. Tests compare irregular lengths around the usual
prefill graph buckets against an independent PyTorch result.

Compilation guards warm these kernels, then sweep token counts (and PLE
request counts) without allowing new Triton specializations. PLE covers
transitions between single-request, uniform multi-request, and ragged indexing
within one compiled kernel. Those selections use runtime values so startup
warmup also covers batch layouts first seen during serving.

## Accepted-prefix replay

`triton_gdn_replay_commit` commits accepted prefixes for all layers in one
portable Gluon launch. Each CTA owns a layer, request, value head, and V tile.
Keys and gates are prepared in fixed chunks, with K/V staged in shared memory.

State updates stay in FP32 until the final write, and rejected tokens are
masked during loading. Aligned state accesses are vectorized, with fallbacks
for unaligned rows and arbitrary row strides. Batch size, window length, and
payload layer stride remain runtime scalars, so serving shape changes reuse
the compiled kernel.
