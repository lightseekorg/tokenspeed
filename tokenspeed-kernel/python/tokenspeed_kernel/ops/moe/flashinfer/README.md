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
It adds a checked `cudaMemsetAsync` after the named map allocation and retains
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

## BF16 intermediate sizes that are multiples of 64

FlashInfer's BF16 TRT-LLM launcher (`Bf16MoeLauncher::check_moe`) rejects an
intermediate size per partition that is not a multiple of 128. Its BF16 cubins
need only 64 for gated activations: GEMM1's rows, twice the intermediate size,
are tiled by 128, and GEMM2 reads its K, the intermediate size, in 128-byte
blocks of 64 elements. Non-gated activations still need 128.

`thirdparty/flashinfer/trtllm_bf16_moe.py` builds a private copy of the
installed launcher in which only that check requires 64 for gated and 128 for
non-gated activations. Like the NVFP4 adapter, it uses a source-keyed JIT
module, private operator names and cloned entry points, and refuses a launcher
whose check it does not find exactly once. `trtllm_unquant.py` declares
`ispp_alignment` 64 when the adapter applies to the installed FlashInfer and
128 otherwise. Only sizes that are not multiples of 128 run on the private
launcher; the first such layer JIT-compiles the whole private TRT-LLM MoE
module during warmup. Other sizes keep FlashInfer's stock module. The NVFP4
routing-map initialization is not added: the routing workspace does not depend
on the intermediate size, so these layers keep FlashInfer's BF16 routing
behavior. Remove the adapter once the minimum supported FlashInfer accepts
these sizes.

## FP32 correction bias for in-kernel DeepSeekV3 routing

FlashInfer's grouped DeepSeekV3 router (`routingMainKernel` in
`trtllm_fused_moe_routing_deepseek.cu`, used when `n_group > 1`) casts the
correction bias to the BF16 output type before adding it to the FP32 sigmoid
score, and its tanh-form sigmoid returns 0 for strongly negative logits.
Reference routers such as DeepSeek-V3's add the bias in FP32, and so do
TokenSpeed's top-k routers. A model opts in per layer with
`routing_config["fp32_correction_bias"] = True`; nothing enables it by default.

`moe_plan(fp32_correction_bias=True)` keeps a kernel that routes from logits
only if its `_tokenspeed_fp32_correction_bias` hook returns True, and otherwise
plans precomputed top-k. Only the BF16 TRT-LLM kernel has the hook. It builds,
when the plan is made, a copy of FlashInfer's TRT-LLM MoE module whose routing
source carries the edit of flashinfer-ai/flashinfer#5557 (FP32 bias,
`1 / (1 + exp(-x))` sigmoid), together with the 64-aligned launcher above when
that applies, under private operator names. Each edit must be found exactly
once in its stock form or in #5557's form; any other source, or a failed
build, logs a warning and the layer plans precomputed top-k. A FlashInfer
whose routing already has both edits is used as is, so this module is no
longer built once the minimum supported FlashInfer includes #5557; remove the
adapter then.

## Qwen3.8 low-batch tactic

For Qwen3.8's 2,560-hidden, 640-intermediate, 512-expert NVFP4 MoE with
TP4/EP4 and top-k 10, the private runner restricts inputs of at most 32 tokens
to FlashInfer's valid tile-32 tactics. If the native launcher does not offer
tile 32 for a profile, all native tactics remain available. Other models and
larger token counts use the full tuner. Only the matching shape gets a separate
tuning-cache key, so a previously cached tile-8 choice cannot bypass this
policy without forcing unrelated shapes to retune. The adapter raises an error
if FlashInfer removes either tuning hook or moves runner construction outside
the cloned entrypoints.

This is a temporary workaround for FlashInfer 0.7's MoE tactic selection.
Remove it when upstream tuning handles the full decode graph.

The policy targets CUDA-graph latency across routing, shared experts and MoE
GEMMs. On the measured BS1 MTP3 graph, FlashInfer 0.7's isolated-kernel tuner
selected tile 8, leaving an idle interval before routing. This policy keeps
tile 32 available for that shape while retaining upstream routing and GEMM
implementations.
