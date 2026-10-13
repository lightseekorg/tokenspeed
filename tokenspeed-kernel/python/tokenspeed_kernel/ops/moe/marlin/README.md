# Marlin MoE kernels

The Marlin grouped GEMM multiplies BF16 activations with weight-only MXFP4
(E2M1) or MXFP8 (E4M3) experts, both with one E8M0 scale per 32 input
channels. It dequantizes inside the mainloop on Ampere-style `mma.sync`, so it
runs on Hopper, which has no tensor-core path for either format.

## Routes outside the local experts

`moe_align_block_size` sorts routes by expert and pads every expert to the
GEMM's token block. Ids outside the aligned expert range, `-1` included, share
one extra lane whose blocks carry expert id `-1`. Only the GEMM's EP
specialization (`is_ep=True`) skips those blocks; the other one would read
expert `-1`'s weights out of bounds. The MXFP8 apply and the MXFP4 local apply
therefore always launch the EP specialization, also at `ep_size == 1`, where
the lane holds only `-1` slots. Skipping costs one scan of the block expert ids
per CTA.

## MXFP8 W8A16 apply

`marlin_mxfp8_precomputed_moe_apply` serves checkpoints whose block-FP8 loader
reports `(1, 32)` scale blocks (MXFP8) on SM90 and later. It registers with
portable priority and declares that block shape, so `(128, 128)` block-FP8
checkpoints keep their FlashInfer and DeepGEMM kernels.
`marlin_mxfp8_moe_weights` repacks the loader's E4M3 weights and raw E8M0
scales into the Marlin layout once and records the expert geometry in the
layer's backend state.

Routing carries global expert ids. Each EP rank owns the contiguous range
`[ep_rank * local_experts, (ep_rank + 1) * local_experts)`; ids outside it,
`-1` included, contribute zero. The alignment, the SwiGLU kernel and the top-k
sum all test that range, so the skipped routes' uninitialized GEMM rows are
never read. On the Marlin path an id repeated within a token contributes once
per slot.

Both GEMMs reduce split-K partial tiles in FP32 in a fixed lock order, GEMM2
applies the route weights, and the top-k sum adds the routes in slot order.
Routes land in GEMM blocks through the alignment, whose default scatter uses
global atomics: a route's block, and with it the split-K partition it is summed
under, can change from run to run. `deterministic_finalize` selects a stable
single-block sort that keeps each expert's routes in route order, which makes
the output bitwise reproducible. Batches of fewer than 1024 routes over at most
63 local experts take a single-block alignment kernel whose layout is fixed
either way. The token block size and the dense path below follow the batch
size, so reproducible results are still not batch-invariant.

Marlin leaves Hopper's tensor cores idle once a batch is compute bound. When
the routes per expert, counted over all ranks as
`tokens * top_k / (local_experts * ep_size)`, reach 192, the apply instead
dequantizes the local experts exactly into BF16 and runs FlashInfer's CUTLASS
grouped GEMM. 192 is the H20 crossover at HY V4 expert shapes; other SKUs use
the same threshold. This dense path needs the intermediate size per partition
to be a multiple of 128, the dequant kernel's K width, and otherwise stays on
Marlin. Its BF16 scratch holds one layer's local experts per device and
geometry. It is allocated with the weights, so KV-cache sizing sees it and no
CUDA-graph pool owns it, and every layer reuses it because MoE layers run one
at a time on the compute stream. CUTLASS skips the ids outside the rank's
range, `-1` included; `deterministic_finalize` selects its unfused top-k
reduction. Its permutation assumes distinct expert ids per token. The path
calls CUTLASS the way the BF16 CUTLASS apply does, with PDL off and that
apply's persistent workspace.


### Input and package compatibility

MXFP8 loading rejects E8M0 codes `0x00` (the nonzero scale `2**-127`,
which the Marlin BF16 decoder cannot preserve) and `0xFF` (NaN) before
repacking any parameter. Supported scale codes are `[1, 254]` on the
Marlin, GEMV and dequantized CUTLASS branches. Apply normalizes activation
and routing strides before selecting a branch; no value checks or host
synchronization are added to the captured forward.

The existing public `moe_plan` entry retains omission of
`requires_deterministic_finalize` as the legacy `False` behavior. This is
an explicit compatibility exception at the independently released package
boundary, not an implicit model policy: every new runtime caller and
`MoELayer` must state its requirement. The runtime negotiates older kernel
planners only for `False`, and rejects a `True` requirement rather than
dropping it. Release the paired kernel before raising the runtime's
minimum package version; do not declare an unpublished version available.

The MXFP8 dequantization helper wraps every output row when swapping the
gate/up halves, including matrices with an odd count of 64-row tiles.
Destination guard tests cover these non-production helper shapes.

`dequantize_marlin_mxfp8` requires contiguous int32 packed weights,
float8_e8m0fnu scales and a BF16 destination on one CUDA device. It validates
storage dtypes, dimensions, shapes, tile alignment, strides and devices before
launching. These checks read metadata only; they do not copy tensors or
synchronize a captured forward. Raw scale bytes belong to the loader, not
this post-repacking interface.
