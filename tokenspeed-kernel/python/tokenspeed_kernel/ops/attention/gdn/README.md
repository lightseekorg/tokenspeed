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

## Variable prefill lengths

The Triton L2-normalization fallback uses a runtime token count. The launch
grid and masked block pointers cover the exact rows, while head width and tile
size stay compile-time parameters. Consecutive agentic turns therefore reuse
one compiled kernel across different token counts instead of compiling a new
variant for each length. Tests compare irregular lengths around the usual
prefill graph buckets against an independent PyTorch result.

Compilation guards warm these kernels, then sweep token counts (and PLE
request counts) without allowing new Triton specializations. PLE covers
transitions between single-request, uniform multi-request and ragged indexing
within one compiled kernel. Those selections use runtime values so startup
warmup also covers batch layouts first seen during serving.

## Accepted-prefix replay

`triton_gdn_replay_commit` commits all layers in one launch. Each CTA owns one
layer, request, value head and V tile, with the full K dimension. V tiles hold
up to 64 columns; four warps distribute contiguous K elements across lanes and
independent V columns across warps. Adjacent CTAs visit adjacent V tiles, then
heads and requests; layers occupy the other grid dimension. Grid indexing
uses model constants rather than division by the runtime batch size. Uniform
parameters and destination indices are loaded before the state read so their
latency can overlap it.

Aligned BF16/FP16 state is loaded and stored through an unsigned 16-bit view,
letting the compiler vectorize the accesses without explicit integer packing.
FP32 state retains pairs packed in 64-bit words. On NVIDIA, 128-dimensional
state uses 128-bit accesses when the actual row address is 16-byte aligned;
other configurations use 64-bit accesses with an 8-byte alignment check.
Alignment hints apply only to a fresh unsigned pointer in the aligned branch;
the original typed pointer handles unaligned rows or K dimensions that cannot
be packed. Pool row strides remain arbitrary. A 128-register limit and compiler
register scheduling keep the wider 16-bit layout from spilling. These choices
depend only on the device, model geometry and state dtype.

The portable Gluon kernel normalizes K and computes gates for a fixed chunk of
accepted tokens, staging K/V in shared memory for the sequential state update.
The state stays in FP32 registers throughout replay and is rounded only at the
final write. Key normalization retains its own warp layout independently of
the state access width. The chunk holds eight tokens for every state dtype;
longer prefixes reuse the same shared allocation. Rejected tokens
are masked during loading. Batch size, window length and payload layer stride stay
runtime scalars, so changing serving shapes does not create specializations.
