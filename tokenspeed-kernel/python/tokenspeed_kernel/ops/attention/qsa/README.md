# QSA sparse attention

`qsa_sparse_attention` consumes per-query physical cache slots. Pass
`max_seqlen_q=None` for prefill, including one-token prefill; uniform decode
passes its query width explicitly.

Prefill may also supply `QSAPrefillMetadata`, an equivalent logical selection:
completed block IDs, logical positions, request IDs, the paged KV table, and CPU
cumulative query lengths. The block IDs must have a dense valid prefix of
`min(block_topk, (position + 1) // block_size)` distinct completed blocks.
The partial causal tail is implicit. These inputs describe the same candidates
as the physical slots, which remain available to fallback kernels.

On SM100/SM103, BF16 Q/K/V with 64/128/256-dimensional heads prefer FlashInfer
PrimTS QSA from `flashinfer-python==0.7.1rc5`, including the CuTe DSL 4.8
task-scheduling compatibility fix. It supports up to 512 selected
blocks, block sizes 4/8/16/32/64/128, and paged caches. Query groups use the
upstream group-size policy with split-KV disabled and end at request boundaries.
The adapter passes an HND view of the existing NHD cache without copying KV.
Unused cache padding must be finite because TMA reads complete fragments.

Plans are bounded and share a high-water workspace per device and CUDA stream.
Each run refreshes the selected union and membership masks; candidates are not
cached across forwards. Warm up each geometry, starting with the largest
workspace, before capturing the kernel directly; captured plans stay pinned.
The runtime invokes QSA prefill at its existing eager graph break.

The PrimTS QSA route requires matching Q/K/V dtypes. BF16 queries with FP8 KV
continue through FA2 to preserve query precision. Tensor-valued K/V scales and
unsupported shapes also use FA2. Select `solution="flashinfer_prims"` to require
PrimTS through normal capability filtering, or `solution="flashinfer"` for FA2.
An explicit kernel override retains its usual meaning; unsupported PrimTS
arguments are rejected. CuTe DSL uniform decode selection is unchanged.
