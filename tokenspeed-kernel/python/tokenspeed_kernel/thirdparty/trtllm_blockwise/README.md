# TRT-LLM blockwise FP8 GEMM

`_kernel.py` is vendored from NVIDIA TensorRT-LLM commit
`ffddc5abd188f308adb5a2906112c2b1e7192d7d`:
`tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockwise_gemm/blockwise_gemm.py`.
Its original Apache-2.0 and BSD-3-Clause notices are retained; local source
changes are limited to repository formatting. The adapter uses
the upstream dynamic-shape wrapper; it does not require importing TensorRT-LLM.

The contract is E4M3 A[M,K], E4M3 B[N,K], FP32 activation scales [M,K/128],
and FP32 weight scales [N/128,K/128]. Output is BF16, accumulation is FP32.
N and K must be divisible by 128. Scale layout copies do not change values.
No E8M0 conversion or weight requantization is performed.

The explicit `trtllm_cutedsl` backend prepares three dynamic-shape tile variants
(M=64/128/256) per GPU and selects a tile by token count.
The adapter passes the current PyTorch stream on every launch, so the same
call path works in eager execution, auxiliary streams, and CUDA graphs.
Input pointers and stream handles are never cached.

Its registered op retains reference priority and is selected explicitly with
`--dense-gemm-backend trtllm_cutedsl`.

On SM100/SM103, automatic BF16 block-FP8 selection also considers the existing
CUTLASS/Triton implementation alongside CuTe DSL tile/cluster tactics. It uses
TokenSpeed's shared startup autotuning context and persistent FlashInfer cache.
The tuned operation accepts contiguous matrices and canonical FP32 scales.
Token counts above 32 use 64-row buckets, matching the smallest tile increment;
coarse power-of-two buckets can select a poor tactic near their lower edge.
Matrix widths, fixed dtypes and output shapes remain in the tuning key.
Tactics compile before tuning/capture, and serving uses cached
choices. An untuned shape retains the existing implementation. Explicit
`trtllm_cutedsl` selection keeps the original fixed-tile comparison path.

Startup enumeration can be widened for an operator with `autotune_max_rows`.
The 64-row serving bucket map stays fixed, so wider-range entries remain
usable after the scope exits and after cache reload. MLA KV expansion declares
its cached-history chunk capacity; ordinary projections retain the engine's
normal token range. Serving never tunes a missing shape.
