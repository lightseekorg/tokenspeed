# GDN kernels

## FlashInfer PDL adapters

The FlashInfer adapter retains the upstream launch geometry and runtime ABI,
while wrapping device bodies with PDL synchronization. Adapted functions and
in-memory compilation caches live in private namespaces.

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
