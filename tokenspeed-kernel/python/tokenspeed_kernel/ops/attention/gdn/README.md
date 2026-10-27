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
use the same execution traits when checking for packed projection views.

State-writing verify and ordinary decode retain the BF16 padding-safe Triton
policy. Read-only verify uses normal selection, including fallback kernels for
unsupported WY geometry. It may read the reserved null page for padding, but
never writes it; padded outputs are ignored.

The leaf reads the committed state once per value tile, forms the small WY
output transform and computes every token's output without writing the state
pool or materializing node states. An optional payload receives exact raw
`[k | v | a | b]` rows for the existing replay commit. State-writing options
are rejected. Q/K/V and gates may be strided; state-pool rows may have padding.

Products use native `mma.sync.m16n8k8` TF32 input rounding and FP32
accumulation; normalization uses native FP32 reciprocal square root, and gates
and WY coefficients stay FP32. For four tokens, backward query propagation
produces the same initial-state coefficient
and value weights without forming Gram matrices or running a triangular solve.
Tests compare against an independent FP64 token recurrence at atol=1e-4,
rtol=2^-7, including strong path gates and unit-scale states. This leaf makes
no bitwise-numerics claim.

Token bounds are bucketed to 4/8/16; MMA reductions use at least eight elements.
Short windows transpose the products to put values along the 16-row MMA axis.
Value tiles use 64 or 128 columns according to window length, state dtype and
the coarse number of request/head pairs. Eight-token windows with at most 64
request/head pairs interleave two FP32 accumulators within each 128-element
TF32 reduction. Other shapes retain one accumulator. This bounded choice is
part of the compilation key; batch and token counts themselves are runtime.
Actual batch size, token count, strides and scale remain runtime arguments.
Warm each selected bucket/dtype/model geometry before CUDA Graph capture;
replays consume updated inputs, parameters, state rows and ancestor masks.
The kernel supports PDL and keeps PDL compilation variants separate.

Q/K/V use four-element vector reads when the physical row has a unit inner
stride and eight-byte alignment. Both conditions are checked on the device;
arbitrary strides and unaligned rows use scalar reads. Normalization and raw
payload stores reuse the loaded values. Four-token 64-value tiles also
issue raw gate and parameter reads before Q/K normalization,
then convert and consume them in the normal FP32 gate computation.

TF32 products share one accumulation loop for both one- and two-accumulator
variants, preserving their reduction order. Four-token windows allocate only
the backward-propagation workspace; larger windows initialize the Gram tiles
directly through MMA.

Checkpoint tiles retain their source dtype in shared memory. Aligned rows use
16-byte asynchronous copies issued before Q/K preparation and completed before
the first checkpoint product. Row alignment is evaluated on the device, so
odd physical row strides and changed indices in a graph use the scalar copy
where needed. Negative indices zero-fill the tile. This changes state transport
without reducing the precision of any matrix product.

Release validation on SM103 covers 64 chain/tree, BF16/FP32,
warm/cold-cache and PDL configurations, with B=1–64, T=4/8/16 and H/HV=8/24,
16/32 or 8/32. Blackwell 13–16-token chains already use chunked verification,
so comparisons include that path in addition to the serial Triton leaf.
The facade and model's ReplaySSM verify use the same registry selection.

## FlashInfer PDL adapters

The FlashInfer adapter retains the upstream launch geometry and runtime ABI,
while wrapping device bodies with PDL synchronization. Adapted functions and
in-memory compilation caches live in private namespaces.

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
source files plus the local PDL and GDN adapter sources. Installed FlashInfer
modules and the process-wide cache configuration remain untouched.

GPU regression tests cover persistent cache hits after clearing the in-memory
cache, invalidation after source changes, and both launch orders. Runtime GDN
tests additionally check CUDA Graph dependency edges and exact replay results
for decode, MTP, BF16 state, and prefill.

## Chain verification

A ReplaySSM chain verify reads the committed state and writes none, so it may
run in any of three forms (`gdn_chain_verify_form`), all on Blackwell only:

- the chain form, `gdn_chain_verify` (`_triton/chain_verify.py`), for 2 to 16
  tokens of bf16 state (the dtype it was measured for) with equal power-of-two
  key and value head dims;
- else the tree form, `gdn_decode_mtp` with a one-path ancestor mask, for 13–16
  tokens and at least 24 states;
- else step by step, as on other shapes and platforms.

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
tier; the GDN backend runs it once per tier its batch sizes reach when it
preallocates its verify workspace, before serving.
`test/runtime/test_gdn_verify_policy.py` checks the selection and its
boundaries without launching device kernels.

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
