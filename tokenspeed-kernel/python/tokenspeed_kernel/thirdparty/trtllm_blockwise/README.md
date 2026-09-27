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

CuTe-DSL and TVM-FFI are optional until this backend is explicitly prepared.
Preparation compiles three dynamic-shape tile variants (M=64/128/256) per GPU.
The adapter selects a tile by token count, not a runtime autotuning loop.
The adapter passes the current PyTorch stream on every launch, so the same
call path works in eager execution, auxiliary streams, and CUDA graphs.
Input pointers and stream handles are never cached.

The registered op has reference priority to preserve automatic selection.
Use the prepared plan via `--dense-gemm-backend trtllm_cutedsl` to opt in.
