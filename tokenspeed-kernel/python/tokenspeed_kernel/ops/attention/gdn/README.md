# GDN kernels

## FlashInfer PDL adapters

The FlashInfer adapter retains the upstream launch geometry and runtime ABI,
while wrapping device bodies with PDL synchronization. Adapted functions and
in-memory compilation caches live in private namespaces.

Shared CuTe adaptation lives in `thirdparty/flashinfer/gdn_pdl.py`; vendor
adapters do not import the op registration package. This keeps direct adapter
imports independent of the registry's initialization order.

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

Automatic decode/verify selection includes the bound state dtype. The generic
FlashInfer BF16 path is excluded because its padding writes the null page;
the masked Rubin decode and portable verify retain the ordinary cache arena.
Explicit legacy `solution="flashinfer"` calls retain their documented padding
semantics; other named solutions still validate the supplied state dtype.
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
