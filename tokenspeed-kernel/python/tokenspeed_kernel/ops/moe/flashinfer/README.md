# FlashInfer MoE adapters

## NVFP4 routing-map padding

The pinned FlashInfer version (0.7.1rc2) initializes unused rows inside active
expert tiles of `permuted_idx_to_token_idx` to `-1` in its routing producers.
This prevents unnecessary activation loads in the first GEMM without a separate
fill-kernel launch. Routing refreshes live mappings and tile padding on every
invocation, including CUDA graph replay and both logits and precomputed top-k
entry points. Allocation slack beyond the active tiles and the launcher's extra
guard entry are not part of this initialization contract.

`thirdparty/flashinfer/trtllm_moe.py` uses the upstream native module while
retaining the private runner for the small-batch tactic policy described
later. Warmup must finish before graph capture.

Regression coverage includes live and padded mapping entries, local expert
partitions, PDL on/off, changing routing within one captured shape, graph replay
after workspace corruption, untouched allocation slack/guard, and output
equality with the upstream operator.
Expert-partition tests retain the loader's global activation input scales while
sharding expert weights, and select SiTU through FlashInfer's activation enum.

## BF16 intermediate sizes that are multiples of 64

FlashInfer's BF16 TRT-LLM launcher (`Bf16MoeLauncher::check_moe`) rejects an
intermediate size per partition that is not a multiple of 128. Its BF16 cubins
need only 64 for gated activations: GEMM1's rows, twice the intermediate size,
are tiled by 128, and GEMM2 reads its K, the intermediate size, in 128-byte
blocks of 64 elements. Non-gated activations still need 128.

`thirdparty/flashinfer/trtllm_bf16_moe.py` builds a private copy of the
installed launcher in which only that check requires 64 for gated and 128 for
non-gated activations. It uses a source-keyed JIT
module, private operator names and cloned entry points, and refuses a launcher
whose check it does not find exactly once. `trtllm_unquant.py`'s SiLU/SwiGLU
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

## Qwen3.8 low-batch tactic

For Qwen3.8's 2,560-hidden, 640-intermediate, 512-expert NVFP4 MoE with
TP4/EP4 and top-k 10, the private runner restricts inputs of at most 32 tokens
to FlashInfer's valid tile-32 tactics. If the native launcher does not offer
tile 32 for a profile, all native tactics remain available. Other models and
larger token counts use the full tuner. Only the matching shape gets a separate
tuning-cache key, so a previously cached tile-8 choice cannot bypass this
policy without forcing unrelated shapes to retune. If FlashInfer removes
either tuning hook, changes the cache-key builder signature, or moves runner
construction outside the cloned entrypoints, the adapter raises an error.

The cache-key builder derives the policy tag from the target profile.
During autotuning this is the profile that `p.get_opt_shapes()` selects; during
serving it is the bucket matched to the request. FlashInfer passes caller
tensors when checking a profile, but synthesized tensors when storing its
winner, so the cache-key builder must not compute token-dependent tags from
those tensors. A scoped hook on FlashInfer's shared cache-key builder adjusts
only the private runner's keys. Other runners, cache persistence and
measurement remain unchanged. Different token profiles continue to store
independent winners.

This is a temporary workaround for FlashInfer 0.7's MoE tactic selection.
When upstream tuning handles the full decode graph, remove it.

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
caller-provided buffer matched the fp32 reference. Stream ordering serializes
consecutive MoE layers through their activations, so one buffer per device
is never in use by two calls at once; a model that overlapped two MoE calls on
separate streams would need one buffer per stream. PDL stays off in this
chain for the race described in `cutlass_unquant.py`.
