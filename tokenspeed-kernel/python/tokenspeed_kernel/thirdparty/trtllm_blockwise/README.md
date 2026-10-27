# TRT-LLM blockwise FP8 GEMM

`_kernel.py` is vendored from NVIDIA TensorRT-LLM commit
`ffddc5abd188f308adb5a2906112c2b1e7192d7d`:
`tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockwise_gemm/blockwise_gemm.py`.
Its original Apache-2.0 and BSD-3-Clause notices are retained; local source
changes are limited to repository formatting. The adapter uses
the upstream dynamic-shape wrapper; it does not require importing TensorRT-LLM.

`__init__.py` exposes the public entry points and loads the autotuner lazily.
`runner.py` owns tactic search and cache lookup; `_launch.py` owns compilation,
scale-layout copies and launch. The runner imports the launch module directly,
so neither implementation module imports the package entry points.

The contract is E4M3 A[M,K], E4M3 B[N,K], FP32 activation scales [M,K/128],
and FP32 weight scales [N/128,K/128]. Output is BF16, accumulation is FP32.
N and K must be divisible by 128. Scale layout copies do not change values.
No E8M0 conversion or weight requantization is performed.

CuTe-DSL and TVM-FFI load when this backend is first prepared.
During startup autotuning, the adapter searches the configurations enumerated by
TRT-LLM's `CuteDSLFp8BlackwellRunner` at the source commit above: MMA tiles
64x128, 128x128 and 256x128, 1-CTA/2-CTA instructions, and clusters whose
M/N dimensions are 1, 2 or 4. The upstream `can_implement` check filters invalid
combinations. The kernel source itself is unchanged.

The runner uses TokenSpeed's shared FlashInfer autotuner. It measures the
adapter, including scale-layout copies, with CUDA graphs and a cold L2 cache.
M uses powers-of-two buckets rounded down; N/K and the scale contract remain
in the cache key. Activation scales and output rows follow each M bucket.
The normal startup cache save/load path persists the selected configurations.
FlashInfer's preparation phase compiles the valid dynamic-shape variants only
when a tuning-cache miss needs profiling. Cache hits skip that phase and
prepare the fixed fallback tiles and encountered winners during eager warmup.

Outside the tuning context, calls use the cached choice. Without a matching
entry, the previous rule applies: M <= 64 uses tile 64x128, M <= 128 uses
128x128, and larger M uses 256x128 with 2-CTA instructions. `--disable-autotune`
skips new searches; an existing loaded cache can still be used. The first eager
launch prepares all three fallback configurations. Loaded non-default winners
must be warmed eagerly for their capture buckets; TokenSpeed's graph capture
already runs each batch eagerly before capturing it. `prepare(device)` also
prepares the three fallback configurations explicitly.
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
