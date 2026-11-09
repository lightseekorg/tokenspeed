# MoE ops

Per-op notes for `ops/moe/`.

## Deferred route outputs

`bf16_moe_routes` returns individual, unweighted expert outputs. Deferred MXFP4
execution also returns a map from each original route to its output row, with
`-1` for remote experts. Consumers normalize each route before weighting and
summation. Combining routes before normalization changes the operation.

`gather_moe_routes` restores token/route order from a deferred output map and
zeros absent routes. A kernel declaring `deferred_output_is_token_order` writes
each route it owns to row `token * topk + slot`; with EP > 1 the other rows are
stale, so `gather_moe_routes` still zeros them.

The solution modules register these operations. Shared Triton helpers live in
`_triton/`, and the CuTe DSL routed reduction lives in `_cute_dsl/`. They use
the same registry and public dispatchers as the other MoE operations.

## Expert placement dispatch

With redundant experts (`--ep-num-redundant-experts R`), a logical expert has
several physical replicas across the EP ranks, and routing must emit physical
ids. Under replicated-input EP, every rank runs the same routing over the same
all-gathered tokens, so the replica choice has to be a pure function of the
token row and the route rank for exactly one rank to own each (token, expert)
pair: `replicas[logical, (row + rank) % count]`.

`ExpertDispatch` holds one layer's tables (`replicas [num_logical, X]` int32,
`-1` padded to a fixed width; `num_replicas [num_logical]` int32). The runtime
keeps them as views of the placement's device tables so an in-place
rebalance reaches captured graphs without reallocating.

`dispatch_topk_ids(topk_ids, dispatch)` is the production implementation of
the replica choice: two gathers and a modulo over `[tokens, top_k]`, in
tensor ops, graph-capturable. Every id must be a real expert in
`[0, num_logical)`; the runtime masks zero experts and padded slots around
the call and counts the mapped ids into its load counters afterwards.

Follow-up: a fused Triton kernel for the same contract (one launch instead of
four tensor ops per MoE layer per forward). Its arguments must follow the
compile-time parameter rule: token count, top-k, and table width are runtime
arguments or bucketed, never `constexpr`.

## FlashInfer adapters (`flashinfer.py`)

### Opt-in SM90 block-FP8 grouped MoE

`prepare_sm90_fp8_moe(mode=...)` prepares a callable before warmup and graph
capture. Choose `default` for upstream FlashInfer, `expected_m` for the per-call
hint backport, or `optimized` to also enable expert-tile prefix sums and direct
tile-to-expert lookup. Both private modes accept independent pipeline stage
caps for FC1 and FC2.
Existing `moe_plan` selection stays unchanged.

```python
from tokenspeed_kernel.ops.moe import prepare_sm90_fp8_moe

expert_apply = prepare_sm90_fp8_moe(
    mode="optimized", max_num_stages_fc1=4, max_num_stages_fc2=-1
)
# Each omitted stage bound defaults to -1 and keeps upstream selection.
# Pass the usual FlashInfer CUTLASS MoE arguments plus:
# expert_apply(..., expected_m=expected_tokens_per_expert)
```

Private modes apply `sm90_moe.patch` to a cached source copy of FlashInfer
0.7.3 and compile isolated modules, operators, and tuning caches. They require
matching upstream sources, JIT enabled, and both CUDA toolkit and runtime
12.8 or newer. Initialization reports unsupported sources or build settings.
Private cold JIT compilations bind their include path to that versioned source
tree. FlashInfer shares one mutable include-directory store across native
modules, so the private modules use their own: a module loaded later must not
redirect private kernel compilation, and a private module must not redirect the
installed module's cold compilation to patched headers.
The installed FlashInfer package is unchanged. To switch settings, prepare a
new callable, warm it up, and recapture graphs.

`expected_m` is a positive uint32 hint for both expert GEMMs in one call;
omitting it preserves the native estimate. For complete local expert groups,
use `max(1, ceil(local_rows * top_k / local_experts))`; replicated-input EP
must account for remote routes using the global expert count. Routing,
workspace bounds, and the runner's stored estimate are unchanged.

`max_num_stages_fc1` and `max_num_stages_fc2` independently bound the two
expert GEMMs. Each defaults to `-1`, preserving upstream stage selection for
that GEMM. A positive value only excludes candidates above its limit, in
both regular and SwapAB layouts. Tile selection, stage candidates, and
shared-memory checks follow upstream logic. Bounds must be at least four,
the backend's minimum candidate depth; zero is invalid. Different bound
pairs use isolated module, operator, tuning, and kernel caches.

Prefix/lookup scheduling supports SM90 DeepSeek block-FP8 SwapAB grouped GEMMs.
The prefix scan handles expert counts independent of block and warp size and
refreshes metadata on every replay. A bounded lookup covers the first tiles,
then falls back to prefix binary search. Configurations whose scheduler
metadata would exceed the shared-memory budget retain the upstream scheduler.
Regression tests compare eager and graph outputs with upstream at
`atol=rtol=0`. Performance depends on shape and routing; the mean hint alone
can regress on concentrated routes, so benchmark the intended settings.

Both private modes also change how the SwapAB grouped GEMM moves data, without
changing its accumulation order. When a weight-scale row is a multiple of 16
bytes, the TMA thread prefetches each tile's weight scales with the tile's first
pipeline stage, so the math warps no longer load scales and synchronize at every
tile boundary. Weight loads carry an L2 evict-first hint: weights are read once
per call, and evicting them first keeps the GEMM from evicting, and writing
back, data that earlier kernels left in L2, which would otherwise fill the
pipeline gaps the prefetch removes. With prefix/lookup scheduling, each block
reads the expert offsets once into shared memory while its barriers are set up,
and fills the lookup table per expert instead of searching per tile.

### Fused SM90 gated FC1

The private `optimized` mode fuses eligible block-FP8 SwiGLU calls with
hidden size 512, intermediate size 1024, at most 256 input rows, and an
`expected_m` value that is positive and less than 32. Bias, LoRA, AWQ,
min-latency mode, and a separate FC2 FP8 quantization scale retain their
existing paths. The supported paired tile has 128 output channels per half and
at most 32 token rows; other shapes retain the existing execution path.

A paired tile computes matching gate/up blocks and applies SwiGLU and 1x128
FP8 quantization before writing its output. Both existing BF16 rounding points
are preserved: after FC1 accumulation and after SwiGLU. FC2 consumes the dense
FP8 buffer and padded scales directly. Its BF16 output uses separate workspace,
and the input descriptor is bounded by the current invocation's capacity.

With `enable_pdl=True`, only the fused FC1 and its FC2 launch as programmatic
dependents; other grouped GEMMs stay stream-ordered, as upstream. The matching
eight-stage FC2 can load its first weight tile and weight scales into shared
memory before waiting for FC1. FC1 triggers dependents only after its own
dependency wait and writes neither routing metadata nor weights; activation
and activation-scale reads remain after the dependency wait.
`enable_pdl=False` keeps all grouped launches ordered and disables this
first-tile prefetch.

Prepare and warm the callable before graph capture. Fusion and prefetch variants
have distinct content-keyed kernel caches, and route metadata is refreshed on
replay. Performance depends on routing; benchmark the intended workload.

### NVFP4 routing-map padding

The pinned FlashInfer version (0.7.3) initializes unused rows inside active
expert tiles of `permuted_idx_to_token_idx` to `-1` in its routing producers.
This prevents unnecessary activation loads in the first GEMM without a separate
fill-kernel launch. Routing refreshes live mappings and tile padding on every
invocation, including CUDA graph replay and both logits and precomputed top-k
entry points. Allocation slack beyond the active tiles and the launcher's extra
guard entry are not part of this initialization contract.

`thirdparty/flashinfer/trtllm_moe.py` uses the upstream native module while
retaining the private runner for the small-batch tactic policy described
later. Warmup must finish before graph capture.

Prepared routing gates PDL with the caller's `enable_pdl` and FlashInfer's public
`utils.device_support_pdl` check, matching the upstream fused entry points.

Regression coverage includes live and padded mapping entries, local expert
partitions, PDL on/off, changing routing within one captured shape, graph replay
after workspace corruption, untouched allocation slack/guard, and output
equality with the upstream operator.
Expert-partition tests retain the loader's global activation input scales while
sharding expert weights, and select SiTU through FlashInfer's activation enum.

### BF16 intermediate sizes that are multiples of 64

FlashInfer's BF16 TRT-LLM launcher (`Bf16MoeLauncher::check_moe`) rejects an
intermediate size per partition that is not a multiple of 128. Its BF16 cubins
need only 64 for gated activations: GEMM1's rows, twice the intermediate size,
are tiled by 128, and GEMM2 reads its K, the intermediate size, in 128-byte
blocks of 64 elements. Non-gated activations still need 128.

`thirdparty/flashinfer/trtllm_bf16_moe.py` builds a private copy of the
installed launcher in which only that check requires 64 for gated and 128 for
non-gated activations. It uses a source-keyed JIT
module, private operator names, and cloned entry points, and refuses a launcher
whose check it does not find exactly once. The BF16 TRT-LLM SiLU/SwiGLU
kernels declare `ispp_alignment` 64 when the adapter applies to the installed
FlashInfer and FlashInfer's JIT can compile the private module, and 128
otherwise, with a warning that names the reason. The ReLU2 kernels in that file
keep 128. The private module is not in FlashInfer's AOT jit-cache, and ninja
rebuilds a module left in the JIT workspace whenever the build changes, so this
needs `FLASHINFER_DISABLE_JIT` unset and FlashInfer's nvcc (`FLASHINFER_NVCC`,
else `bin/nvcc` under its CUDA home). The adapter decides this once per
process at import, without compiling, so kernel selection and layer padding
agree. Only sizes that are not multiples of 128 run on the private launcher;
the first such layer JIT-compiles the whole private TRT-LLM MoE module during
warmup. Other sizes keep FlashInfer's stock module. These layers keep
FlashInfer's BF16 routing behavior. Once the minimum supported FlashInfer
accepts these sizes, remove the adapter.

### FP32 correction bias for in-kernel DeepSeekV3 routing

FlashInfer's grouped DeepSeekV3 router (`routingMainKernel` in
`trtllm_fused_moe_routing_deepseek.cu`, used when `n_group > 1`) casts the
correction bias to the BF16 output type before adding it to the FP32 sigmoid
score, and its tanh-form sigmoid returns 0 for strongly negative logits.
Reference routers such as DeepSeek-V3's add the bias in FP32, and so do
TokenSpeed's top-k routers. A model opts in per layer with
`routing_config["fp32_correction_bias"] = True`; nothing enables it by default.

`moe_plan(fp32_correction_bias=True)` keeps a kernel that routes from logits
only if its `_tokenspeed_fp32_correction_bias` hook returns True, and otherwise
plans precomputed top-k. Only the BF16 SiLU/SwiGLU TRT-LLM kernel that routes
from logits has the hook; the BF16 ReLU2 kernels do not. That hook builds,
when the plan is made, a copy of FlashInfer's TRT-LLM MoE module whose routing
source carries the edit of flashinfer-ai/flashinfer#5557 (FP32 bias,
`1 / (1 + exp(-x))` sigmoid), together with the 64-aligned launcher above when
that applies, under private operator names. Each edit must be found exactly
once in its stock form or in #5557's form; any other source, or a failed
build, logs a warning and the layer plans precomputed top-k. A FlashInfer
whose routing already has both edits is used as is, so this module is no
longer built once the minimum supported FlashInfer includes #5557; remove the
adapter then.

### Qwen3.8 low-batch tactic

For Qwen3.8's 2,560-hidden, 640-intermediate, 512-expert NVFP4 MoE with
TP4/EP4 and top-k 10, the private runner restricts inputs of at most 32 tokens
to FlashInfer's valid tile-32 tactics. If the native launcher does not offer
tile 32 for a profile, all native tactics remain available. Other models and
larger token counts use the full tuner. Only the matching shape gets a separate
tuning-cache key, so a previously cached tile-8 choice cannot bypass this
policy without forcing unrelated shapes to retune. If FlashInfer removes
either tuning hook, changes the cache-key builder signature, or moves runner
construction outside the cloned entry points, the adapter raises an error.
When FlashInfer splits its module factory and runner-namespace builder, both
are rebound so the operator closures keep the same private tactic policy.

The cache-key builder derives the policy tag from the target profile.
During autotuning, this is the profile that `p.get_opt_shapes()` selects; during
serving it is the bucket matched to the request. FlashInfer passes caller
tensors when checking a profile, but synthesized tensors when storing its
winner, so the cache-key builder must not compute token-dependent tags from
those tensors. A scoped hook on FlashInfer's shared cache-key builder adjusts
only the private runner's keys. Other runners, cache persistence, and
measurement remain unchanged. Different token profiles continue to store
independent winners.

This is a temporary workaround for FlashInfer 0.7's MoE tactic selection.
When upstream tuning handles the full decode graph, remove it.

The policy targets CUDA-graph latency across routing, shared experts, and MoE
GEMMs. On the measured BS1 MTP3 graph, FlashInfer 0.7's isolated-kernel tuner
selected tile 8, leaving an idle interval before routing. This policy keeps
tile 32 available for that shape while retaining upstream routing and GEMM
implementations.

### NVFP4 squared ReLU

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

### Packed FP8 expert intermediate

The matrix-based routed-expert path converts the clamped gate/up result into
E4M3 inputs for the down projection, with one FP32 scale per 128 activations.
It preserves BF16 intermediate rounding and clipping before conversion.
Unsplit projections
fuse conversion into the activation producer; split projections combine their
partial results before the nonlinear activation and conversion. Only FP8 values
and scales are materialized. The first BF16 input conversion still uses the
shared quantizer rather than an upstream model operation.

### Unquantized CUTLASS MoE workspace

`flashinfer_cutlass_unquant_moe_apply` hands `cutlass_fused_moe` one
persistent scratch buffer per device (`cutlass_unquant_moe_workspace`),
sized through `cutlass_fused_moe_workspace_size`, zero-filled when it is
allocated or grown, and reused across layers and calls. Left to allocate its
own scratch per call, the SM90 chain read bytes it never wrote: for some
(EP rank, routing) combinations the rank's routed output was NaN for finite
inputs, reproducibly for that call, while the identical call with any
caller-provided buffer matched the FP32 reference. Stream ordering serializes
consecutive MoE layers through their activations, so one buffer per device
is never in use by two calls at once; a model that overlapped two MoE calls on
separate streams would need one buffer per stream. PDL stays off in this
chain for the race described in the BF16 CUTLASS section of `flashinfer.py`.

FP16/BF16 unquantized CUTLASS plans default to fused finalize. Fusing
the top-k expert reduction into the GEMM2 epilogue uses floating-point
atomics and does not preserve the reduction order across calls. The plan's
`cutlass_fused_finalize` value is passed explicitly to both the workspace
size query and the MoE call. Set it to `False` for a separate reduction.
Choose the value before warmup and CUDA graph capture; changing it after
capture does not update recorded kernels. This controls
one reduction, rather than the complete deployment's determinism contract.

### FP8 per-channel experts

The FP8 per-channel TRT-LLM section of `flashinfer.py` runs `fp8_channel`
experts (compressed-tensors FP8 dynamic W8A8: one FP32 scale per weight row,
activations quantized per token) on FlashInfer's
`trtllm_fp8_per_channel_scale_moe`. The kernel reads the per-row scales in the
weights' own resident row order, so the weight preprocessor moves every scale
with its row: the `[gate | up]` to `[up | gate]` swap, the gated-activation
interleave, and the epilogue-tile shuffle. Its per-expert output scalars stay 1.
Each rank pads its own intermediate partition to a multiple of 64 with zero rows
and columns whose scales are 1, so a rank's channels do not move and the padding
adds exactly zero, also to the per-route FC2-input quantization. Partitions that
are a multiple of 64 need no padding: on GB200 the kernel output for a multiple
of 64 is bit-identical to the same experts padded to the next multiple of 128.
The FC2 input is quantized per routed row by TRT-LLM's `perTokenQuantization`
(scale `max(amax / 448, 1 / (448 * 512))`, codes `x * min(448 / amax, 448 *
512)` in FP32, rounded to BF16 and then to E4M3, `amax` over the BF16 row). The
cubins need `usePerTokenSfA` and `usePerTokenSfB`; they are not among the BF16
and NVFP4 MoE cubins, so an offline cache must hold them.

The kernel takes precomputed routes only as BF16 weights packed with the
expert ids. With FP32 route weights the routed kernel stops before its
finalize and `moe_finalize_fuse_shared` combines the per-route outputs with the
FP32 weights, accumulating in FP32 and rounding once to BF16.

## FP8 per-channel experts

Per-channel FP8 experts store weights in FP8 E4M3 with FP32 scales per output
channel and quantize activations per token. FP32 routing weights remain FP32
during combination. Native FP8 matrix arithmetic uses FP32 accumulators but
need not match sequential FP32 arithmetic.

On SM90 the Triton kernel stays at 32 rows per tile. A 64-row FP8 tile lowers
to wgmma, whose accumulator is coarser than FP32, and misses the FP64 reference
by 5e-3 to 7e-3. The smaller tiles stay within 5e-4 but give up wgmma's
throughput: from 2048 tokens, and on wide experts at every size above 512, they
run slower than 64-row tiles would.

## Shared MoE precision policies

The unquantized Triton MoE normally keeps GEMM1 accumulators in FP32 through
SiLU and combines route outputs with FP32 weights without intermediate rounding.
Existing callers retain that behavior.

An execution plan with `precision_policy="bf16_route_accum_fp32"` and an explicit
positive finite `output_scale` requests these distinct boundaries:

1. Round each GEMM1 gate/up projection to BF16; compute SiLU and multiplication
   in FP32, then store the intermediate in BF16.
2. Accumulate GEMM2 in FP32 and store each expert output in BF16.
3. Multiply each expert output by its unscaled FP32 route weight, then round
   each weighted route to BF16.
4. Sum those routes in ascending expert-ID order in FP32, multiply by
   `output_scale` in FP32, and only then store the final BF16 output.

The result is rank-local: any tensor-parallel reduction is owned by the caller
and occurs after this final BF16 boundary. Scaling an already-finalized BF16
sum, or scaling routing weights before the per-route rounding, is not equivalent.
Routing must independently implement the caller's selection and normalization
contract. The expert kernel does not renormalize or rescale route weights.

The runtime's existing `MoELayer` accepts this policy through `routing_config`
keys `precision_policy` and `output_scale`. It requires unquantized BF16 SiLU,
precomputed FP32 top-k weights, unit `routed_scaling_factor`, no expert bias, and
no expert-parallel/all-to-all mode. Automatic backend selection chooses the
shared Triton BF16 implementation. An explicitly requested unsupported backend
fails closed; CUTLASS is not declared equivalent to these rounding boundaries.
No new scheduling or model execution path is introduced.

This correctness-first policy sorts the small selected-expert list before the
existing dispatch. It is not a performance claim; check its numerics and graph
replay before using its results as an optimization baseline.

## Vendor policy `bf16_vendor_fp32_route`

`precision_policy="bf16_vendor_fp32_route"` runs FlashInfer's TRT-LLM BF16
experts (`--moe-backend flashinfer_trtllm`) with the DeepSeekV3 correction bias
added in FP32. It is a different arithmetic contract, not an equivalent of
`bf16_route_accum_fp32`:

| Step | `bf16_route_accum_fp32` | `bf16_vendor_fp32_route` |
| --- | --- | --- |
| Route weights | Unscaled FP32 | Routed scale applied, rounded to BF16 for the kernel |
| GEMM1 gate/up | Rounded to BF16 | Not rounded |
| Weighted route | Each route rounded to BF16 | Not rounded |
| Combine | Ascending expert ID, FP32, times `output_scale` | Vendor slot order, FP32 |
| Output scale | `output_scale` | Must be 1 |

`MoELayer` accepts it for BF16 unquantized SiLU experts with DeepSeekV3
routing, without expert bias, all-to-all, `--moe-combine-order slot`, or an
output scale. The policy asks `moe_plan` for `fp32_correction_bias`: the plan
routes inside the kernel (`flashinfer_trtllm_unquant_moe_apply`) when the
kernel's hook builds FlashInfer's routing with the FP32 bias
(`thirdparty/flashinfer/trtllm_bf16_moe.py`), and otherwise plans precomputed
top-k (`flashinfer_trtllm_unquant_routed_moe_apply`), where the caller's
router adds the bias in FP32. Under expert parallelism, the layer plans
precomputed top-k. In the kernel, `routing_config["routed_scaling_factor"]`
is applied to the BF16 route weights; on precomputed routes the caller folds
it into the FP32 route weights (Beam: in its route epilogue). The intermediate size per partition must be a
multiple of the kernels' `TRTLLM_UNQUANT_ISPP_ALIGNMENT` (64 where FlashInfer's
launcher is relaxed, else 128); the policy refuses a size the layer would pad.
The kernel finalizes its output itself; deferred finalize is refused.

The two routings are not byte-identical: the kernel computes
`fl32(fl32(s * scale) / fl32(sum + 1e-20))` with its fast-math division, the
precomputed path `fl32(fl32(s / fl32(sum + 1e-20)) * scale)`
(Beam's route epilogue with the routed scale, the bytes of TopK's fold). An IEEE
CPU emulation of the two orders on Beam's routing shape differs in about 5 of
10^6 BF16 weights. A
model whose ranks must agree on one routing checks the plan's
`support_routing` across its MoE group.

## Beam's grouped route

`beam_route_weights` (`python/tokenspeed/runtime/models/beam.py`) routes each
token with a grouped sigmoid: the FP32 sigmoid of the router logits plus the
correction bias, the sum of the top two scores in each of 8 groups, the top 4
groups, the top 8 experts of the kept groups, their unbiased scores renormalized
(sum plus `1e-20`) and multiplied by the routed scale.

- `grouped_route.torch_grouped_route` is those statements in PyTorch (any device).
- `_triton/router_select.triton_router_route_lanepar` runs them in one launch,
  one row per program on one warp, byte for byte: the FP32 sigmoid with Torch's
  CUDA recipe (libdevice `exp`, inline `add.rn.f32` / `div.rn.f32`, no FTZ), the
  grouped selection with Torch's radix keys, fill order, and trimmed bitonic
  network (`_triton/grouped_select.py`; the tie permutation is the pinned
  Torch's), and the epilogue with the sum in Torch's CUDA K8 order
  `((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))` (`_triton/route_epilogue.py`). Its host
  guard `router_route_rejection` admits Beam's routing shape only (256 experts,
  8 groups, top 2 per group, 4 groups, top 8, contiguous FP32 CUDA tensors, at
  least one row, a scale that rounds to a finite zero or normal FP32). It
  launches as a programmatic dependent where `pdl_enabled()`.

Beam calls the Triton kernel on NVIDIA SM90 to SM103 where the guard admits the
call and the Torch statements otherwise. The GPU tests
(`test/nvidia/ops/moe/test_router_select_cuda.py`) pin Torch's FP32 sigmoid and
K8 sum order first (a mismatch fails, never skips), then compare the two byte for
byte on random rows, tie, edge, and NaN corpora and in CUDA-graph replay.

## Routing weight storage and deferred finalize

`moe_plan` records the selected expert kernel's `topk_weights_dtype`. DeepSeek V4
routing stores its weights in that dtype after FP32 scoring, normalization, and
scaling.

On NVIDIA, the DeepSeek V4 MoE defers finalization for BF16 activations
whenever the selected kernel supports it (TRT-LLM MXFP4 SwiGLU, Triton MXFP4,
and the other deferred-capable kernels). The kernel returns expert rows and row
indices, and `moe_finalize_fuse_shared` combines them with the precomputed
route weights and the shared expert output in FP32, trimming to the original
hidden width. Some kernels (Triton MXFP4) return no weights; `MoELayer` then
returns the precomputed route weights it passed in. FP16 inputs, ROCm builds
without the fused finalizer, and kernels without deferred finalize keep the
kernel's own finalized output.

## Kimi-K3 router

On SM100/SM103, batches from 2,560 tokens use a vendored CUDA top-16 router
over 896 experts. It consumes contiguous FP32 logits and bias and writes
normalized FP32 weights (scale 1) and INT32 IDs into caller-owned buffers.
The routing-only kernel has no expert-plan workspace or 8,192-token limit.
Other contracts retain their existing routers.

## Kimi-K3 front projections

`try_latent_moe_input_projections` runs a registered fused implementation or
returns `None` before any projection. This lets the runtime retain the
separate projections and their stream scheduling for unsupported inputs.
`latent_moe_input_projections` keeps its portable composition when no fused
implementation is available.

`cuda_latent_input` is the vendored FlashInfer/Cake fused front
(`thirdparty/cuda/kimi3_latent_front.py` and `csrc/kimi3_latent_front/`): one
persistent launch produces the FP32 router logits, the BF16 latent input and
the BF16 SiTU-activated shared-expert input. It is registered on SM103 with 152
SMs for 2048-16384 BF16 rows, Kimi-K3's replicated router/latent weights, full
shared-expert weights, and SiTU clamps 4 and 25. Attention-DP uses it only
without shared-expert TP; EP routing/quantization and the shared down projection
keep their existing owners. Smaller shapes and shared-expert TP retain their
existing paths.

The programs are JIT-built through FlashInfer's toolchain like the Kimi-K3
router and do not depend on the FlashInfer package shipping its experimental
kernel sources (release wheels omit them). Prepare/warm up before graph capture;
no persistent activation tensors are cached per layer by this adapter.

## NVFP4 MegaMoE for Qwen3.5

The `mega_moe` NVFP4 adapter supports gated SiLU (`silu(gate) * up`) and
Kimi K3 SiTU. Gated SiLU selects the existing kernel epilogue with both SiTU
parameters set to `None`. BF16 inputs are quantized and staged with routing
directly into the symmetric workspace in one kernel, including masking unused
capacity rows. BF16 staging rejects routing tensors whose row counts differ
from the live activation count before launching the staging kernel.
Prequantized packed E2M1 inputs with linear E4M3 block scales
remain supported for existing callers. `get_runner` requires an explicit
`prequantized=True` for packed NVFP4 inputs (including direct workspace writes),
or `prequantized=False` for fused BF16 quantization and staging.
Input format is part of the runner's
tuning key, and calibrated input scales stay runtime tensors across graph replay;
weights are converted once to interleaved gate/up rows and blocked scales.
Gate/up global weight scales must match. The adapter currently targets
SM100/SM103 and requires MoE TP=1.

Qwen3.5 uses precomputed Qwen top-k and a separate, fully replicated shared
expert with its original sigmoid gate. Select
`--moe-backend mega_moe --all2all-backend none` with NVFP4 routed weights,
MoE TP=1 and EP>1. Attention rows must already be scattered before the MoE,
for example DP4/TP2/EP8 or DP8/TP1/EP8. Layouts with attention TP equal to
MoE TP*EP (including TP8/EP8) are rejected before weights are constructed.
The common communication manager handles input and final normalization and
row gathering; the model needs no additional all-reduce bypass or row slicing.

`Qwen3_5MoeSparseMoeBlock` checks MegaMoE options during construction: EPLB, redundant experts, nontrivial placement
and external all-to-all backends are unsupported. The NVFP4 weight adapter
checks MoE TP=1. Weight dtype, activation and
hardware compatibility are checked by the common MoE kernel planner.

All ranks, including empty token owners, must enter the expert call with the
same capacity. Existing runner warmup and autotuning reserve the workspace
before graph capture; returned outputs alias that workspace. Shared experts
run independently on the existing auxiliary stream during CUDA graph capture
and replay, overlapping routed experts; the streams join before applying the
shared sigmoid gate and gathering rows. Warmup retains serialized execution
so kernel tuning and compilation finish before capture.

Run the kernel regression on Blackwell, for example:

```bash
PYTHONPATH="tokenspeed-kernel/test:$PYTHONPATH" torchrun --standalone --nproc-per-node=2 \
  tokenspeed-kernel/test/nvidia/ops/moe/test_trtllm_mega_moe.py \
  --activation silu --input-format bf16 --intermediate-size 192 --capacity 8 --tokens 5
```

Repeat with `--input-format nvfp4` to cover prequantized inputs, and with
`--autotune` for tactic coverage. The test compares a dequantized
reference, uneven/idle ranks, graph replay and alternating capacities.

`test_trtllm_mega_moe_input_stage.py` separately checks byte-exact packed data
and scales against FlashInfer quantization, INT32/INT64 route inputs and outputs,
zero-token owners, changing live counts and alternating captured graphs on the
same workspace, with a compile guard across changing live counts.
The stager converts route IDs and scores to their destination buffer dtypes
and keys compiled specializations by all input and output dtypes; current
MegaMoE workspaces store INT32 IDs.

Qwen3.5 follows K3's empty-owner path: gate, top-k, quantization and shared
expert work are skipped when there are no local tokens, but the rank still
participates in the collective MegaMoE call. Nonempty ranks use normal tuning;
the model does not disable GEMM tuning for uneven batches.

MegaMoE reserves scratch for all candidate tactics before CUDA graph capture,
whether tactic autotuning is enabled or disabled. With tuning disabled,
`forward()` selects the original capacity-based default; enabling
`MEGAMOE_TACTIC_AUTOTUNE=1` searches the candidates. The candidate set excludes
FC2 write-back tactics. Token capacity and persistent addresses do not change,
and launches with an unreserved tactic or capacity are rejected.

The temporary full-capacity workspace experiment reserves the configured maximum
per-rank token capacity before capture, including the prefill ceiling (8192 in
the capacity test) and the MTP verification batch (64 requests times 4 tokens).
This follows the serving-capacity allocation policy of vLLM's MegaMoE buffer;
the CuTe kernel retains its own layouts and alignment. Each forward executes
one collective with no 128-token chunk loop. All layers and captured capacities
continue to share stable workspace addresses. Kimi K3 writes packed activations
and routes directly into views of this full-capacity symmetric allocation.

## DeepSeek V4 MegaMoE shared expert (`deep_gemm.py`)

DeepGEMM's MXFP4 MegaMoE can run one MXFP8 shared expert of the routed
experts' intermediate size inside the same launch. The MoE plan exposes this
as `fuse_shared_expert`; `MoELayer.fuse_shared_expert` hands over the shared
expert's E4M3 weights and UE8M0 scale bytes once the routed weights are
processed. They are re-laid out in place like a routed expert (gate/up rows
interleaved by eight, scales packed four per int32 and UTCCP-transposed), and
the routed output then includes the shared expert. The input staging kernel
also writes the activation scales the shared expert reads: MN-major, with rows
grouped by the BLOCK_M MegaMoE picks for the token count. The shared expert's
SwiGLU uses the routed experts' clamp.

## Kimi-K3 TP8 decode chain (`_cute_dsl/k3_decode.py`)

On Blackwell with one TP8 group shared by attention and MoE, a decode forward
(speculative verify included) of 1-8, 16, 32, or 64 token rows that a captured
decode graph runs takes every MoE layer through one chain of vendored CuTe DSL
kernels (`thirdparty/cute_dsl/k3_decode`).
The kernels hand rows to each other through Lamport mailboxes in multicast
symmetric memory instead of NCCL collectives:

1. One GEMM computes this rank's 112 router logits and 448 latent-down rows
   and multicasts both to every rank.
2. One kernel takes the top-16 experts (sigmoid plus correction bias,
   renormalized, scaled) and gathers the full latent row.
3. Persistent MXFP4 experts compute this rank's intermediate slice of the
   chosen experts and combine them into a weighted latent partial.
4. One tail kernel all-reduces the partials, applies the latent RMSNorm, runs
   this rank's up-projection rows, adds the shared expert (run on a side
   stream), and multicasts the output row.
5. A ring kernel adds that row to the residual.

Activations enter the experts as MXFP8, as in the TRT-LLM MoE. The routed SiTU
uses the approximate `tanh`; the shared expert's is exact. A mailbox reader
re-arms what it read, so the chain needs no separate reset and replays in a
CUDA graph. Every rank must run the same batches, since each mailbox is shared
by the group. Other row counts and every non-decode forward keep the TRT-LLM
MoE.

At 5-8 rows the routing kernel lets the experts launch as it starts, and the
shared expert runs on a highest-priority side stream that the tail never waits
for: the tail polls the shared rows from their mailbox. Those tails keep their
release after the output multicast: a release at entry lets kernels in that can
hold the SMs the side stream still needs. The forward rejoins that side stream once
at its end (`KimiK3DecodeMoE.join`), since a CUDA graph capture fails while a
forked stream is not joined. These only reorder work; the outputs are the same
bit for bit.

Up to 8 rows a decode step is bound by HBM bandwidth, which the attention core
and the all-reduces leave idle, so the chain prefetches weights into L2
(`_cute_dsl/k3_decode_prefetch.py` has the row counts and caps). A side stream
prefetches the MoE front's packed weight after the o-projection exit; the
o-projection shard and the MoE layer's front, shared-expert and up-projection
weights after the KDA input projection or the MLA core; and, once the routed
experts launch, the next layer's attention weights up to a cap. At 4-7 rows the
routed experts instead prefetch the up-projection shard and the next attention's
weights themselves, once their last weight load is issued. The forward rejoins
the prefetch stream at its end as well. The prefetch tables hold weight
addresses, so the weights must not move after binding. Prefetching changes no
result.

Three places decide where the chain runs. The Kimi-K3 MoE execution plan
allows it for one tensor-parallel group shared by attention and MoE on
Blackwell TRT-LLM. `supports()` in `_cute_dsl/k3_decode_support.py` (no
`cutlass` import, so any platform can ask) declares what the kernels serve: TP8
MXFP4 experts behind an ungrouped sigmoid top-16 router over 896 experts, a
linear SiTU cap, a BF16 router, PDL enabled, and no batch-invariant reduction.
The model then binds it where both hold for every MoE layer, the node decodes
(not a prefill-only disaggregated role), the group has multicast, and the
decode graphs run a supported row count. Under speculative decoding the
shared expert uses the splits tuned for verify steps. Each MoE layer keeps a contiguous copy
of its router rows and latent-down shard (about 7.7 MiB per layer at TP8)
beside the mailboxes and scratch.

When every decoder layer's attention also qualifies, the model runs the whole
decode step through the chain (`KimiK3DecodeAttention`). The model checks that
the attention has no attention-side parallel communication, no FP8 o-projection
input, and an output gate on MLA; `attention_supports()` in
`_cute_dsl/k3_decode_support.py` checks the projections the kernels serve
(BF16, this model's TP8 KDA and MLA input projections, and a 1536-wide
o-projection shard). Each layer then does the following:

1. Add the previous feed-forward output to the residual prefix and mix it
   with the block snapshots (AttnRes), reading a MoE layer's output row
   straight from the tail's ring.
2. Run the packed input projection as a split-K GEMM with per-batch tactics.
3. Run the attention core as usual, through the attention's `o_proj_input`
   (Kimi-K3.1's DSA layers select their indexer's top-k there, as in their
   `forward`).
4. Multiply by this rank's o-projection shard and multicast the partial;
   a consumer kernel sums the partials, updates the prefix, and applies the
   post-attention AttnRes. Above 32 rows each rank sums its column share and
   multicasts it back (two-shot) instead of reading every rank's partial.

A closing AttnRes reads the last MoE row and applies the final norm. A layer's
output exists as the residual prefix once the next entry (or the closing
AttnRes) has added its feed-forward row, so drafter captures are taken there.
