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

CuTe-DSL and TVM-FFI load when this backend is first prepared.
Preparation compiles three dynamic-shape tile variants (M=64/128/256) per GPU.
The adapter selects a tile by token count, not a runtime autotuning loop.
The adapter passes the current PyTorch stream on every launch, so the same
call path works in eager execution, auxiliary streams, and CUDA graphs.
Input pointers and stream handles are never cached.
CuTe tensor and pointer descriptors are built during compilation. Each launch
passes the current tensor addresses and the Torch output view directly through
TVM-FFI, avoiding repeated DLPack conversion and layout marking on the host.
The launch checks the compiled inputs' and output's 16-byte alignment requirement.
It reads the current stream's raw handle for the input device on each call,
avoiding a temporary Torch Stream object. A single strided view supplies the
output descriptor while preserving its storage offset and compiled layout.

Automatic selection prefers this backend for compatible GEMMs with M > 16.
The low-priority registration also permits explicit selection at smaller M
with `--dense-gemm-backend trtllm_cutedsl`. See the
[GEMM selection policy](../../ops/gemm/README.md#blackwell-block-fp8-automatic-selection)
for eligibility and fallback behavior.
