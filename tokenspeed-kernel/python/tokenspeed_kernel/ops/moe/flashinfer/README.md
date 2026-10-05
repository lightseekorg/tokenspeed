# FlashInfer MoE adapters

## NVFP4 routing-map padding

The TRT-LLM NVFP4 entry points use a private native launcher that initializes
`permuted_idx_to_token_idx` to `-1` before routing. Routing then writes the
real token assignments. Expert-tile padding and the guard entry stay invalid,
so the first GEMM does not gather activations using stale workspace values.

The initialization uses the native tensor's allocated length and current CUDA
stream. It applies to both routing from logits and precomputed top-k, including
different token counts, expert partitions and GEMM tile sizes. It runs in eager
execution and is recorded inside CUDA graphs, so every replay refreshes padding
even when graphs reuse a memory pool. No host synchronization or extra routing
buffer is needed.

`thirdparty/flashinfer/trtllm_moe.py` builds a source-keyed private JIT module.
The adapter ships as a Python package in both source distributions and wheels;
it does not require a source checkout on `PYTHONPATH`.
It adds a checked fill-kernel launch after the named map allocation; the kernel
waits on and releases programmatic dependents, so the routing chain keeps PDL
where a memset graph node would cost about 4 us per MoE layer. It retains
FlashInfer's routing and GEMM implementations and Python API signatures. The
small-batch tactic policy below narrows the tuner's candidates for one model.
The installed package and stock JIT modules are unchanged. The first warmup
requires FlashInfer's usual JIT toolchain and compiles the private module;
subsequent processes reuse its cache. Warmup must finish before graph capture.
An unrecognized native allocation layout raises an error rather than silently
running without initialization. Review this adapter when updating FlashInfer.
It can be removed once the minimum supported FlashInfer version guarantees
the same padding initialization before every routing invocation.

Regression coverage includes live and padded mapping entries, local expert
partitions, PDL on/off, changing routing within one captured shape, graph replay
after workspace corruption, and output equality with the upstream operator.
Expert-partition tests retain the loader's global activation input scales while
sharding expert weights, and select SiTU through FlashInfer's activation enum.

## Qwen3.8 low-batch tactic

For Qwen3.8's 2,560-hidden, 640-intermediate, 512-expert NVFP4 MoE with
TP4/EP4 and top-k 10, the private runner restricts inputs of at most 32 tokens
to FlashInfer's valid tile-32 tactics. If the native launcher does not offer
tile 32 for a profile, all native tactics remain available. Other models and
larger token counts use the full tuner. Only the matching shape gets a separate
tuning-cache key, so a previously cached tile-8 choice cannot bypass this
policy without forcing unrelated shapes to retune. The adapter raises an error
if FlashInfer removes either tuning hook, changes the cache-key builder
signature, or moves runner construction outside the cloned entrypoints.

The policy tag is derived from the target profile in the generated cache key.
During autotuning this is the profile selected by `p.get_opt_shapes()`; during
serving it is the bucket matched to the request. FlashInfer passes caller
tensors when checking a profile, but synthesized tensors when storing its
winner, so token-dependent tags must not be computed from those tensors. A
scoped hook on FlashInfer's shared cache-key builder adjusts only the private
runner's keys. Other runners, cache persistence and measurement remain
unchanged. Different token profiles continue to store independent winners.

This is a temporary workaround for FlashInfer 0.7's MoE tactic selection.
Remove it when upstream tuning handles the full decode graph.

The policy targets CUDA-graph latency across routing, shared experts and MoE
GEMMs. On the measured BS1 MTP3 graph, FlashInfer 0.7's isolated-kernel tuner
selected tile 8, leaving an idle interval before routing. This policy keeps
tile 32 available for that shape while retaining upstream routing and GEMM
implementations.

## NVFP4 squared ReLU

Nemotron-H experts use a non-gated `relu(x)**2` activation. GEMM1 holds only
the up projection, so `w13` is `[E, I, H]` rather than `[E, 2I, H]`. The weight
preprocessor skips the gate/up half swap and permutes with
`is_gated_act_gemm=False`. FlashInfer tiles non-gated GEMM1 rows by 128, so
the kernels declare an `ispp_alignment` of 128 instead of 64.

The kernel applies `output1_scale_gate_scalar` to the GEMM1 accumulator before
squaring and `output1_scale_scalar` after. Squaring is not linear, so the
GEMM1 dequant `a13 * w13_scale_2` must go in the first scale and the second
scale carries only the GEMM2-input requant `1 / a2`. This is the SiTU recipe.
The SwiGLU recipe folds the up-half dequant into the second scale, which would
be wrong here. The kernel test checks non-unit input scales against a
dequantized reference, and checks in-kernel sigmoid-plus-bias routing with 512
experts and top-22.

## Unquantized CUTLASS MoE workspace

`flashinfer_cutlass_unquant_moe_apply` hands `cutlass_fused_moe` one
persistent scratch buffer per device (`cutlass_unquant_moe_workspace`),
sized through `cutlass_fused_moe_workspace_size`, zero-filled when it is
allocated or grown, and reused across layers and calls. Left to allocate its
own scratch per call, the SM90 chain read bytes it never wrote: for some
(EP rank, routing) combinations the rank's routed output was NaN for finite
inputs, reproducibly for that call, while the identical call with any
caller-provided buffer matched the fp32 reference. Consecutive MoE layers are
stream-ordered through their activations, so one buffer per device is never
in use by two calls at once; a model that overlapped two MoE calls on
separate streams would need one buffer per stream. PDL stays off in this
chain for the race described in `cutlass_unquant.py`.
