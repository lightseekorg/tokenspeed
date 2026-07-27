# tokenspeed-kernel-intel

Intel XPU (GPU) specific high-performance kernels for TokenSpeed.

Unlike the portable Triton path, these kernels dispatch to
[vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) — SYCL/DPC++
kernels that leverage oneDNN for deep-learning primitives on Intel GPUs.

## How it works

This package plugs into the standard TokenSpeed kernel registry. Importing
`tokenspeed_kernel_intel` (done automatically by `tokenspeed_kernel` when running
on an Intel XPU platform) registers XPU-specialized implementations of the public
operator families (attention, layernorm, activation, GEMM, MoE, ...) via
`@register_kernel(..., capability=CapabilityRequirement(vendors={"intel"}))`.

Because these kernels register at a higher priority band than the portable Triton
kernels, the selector prefers them automatically when running on Intel XPU, and
falls back to Triton where an XPU kernel is not (yet) provided.

## Requirements

- Intel GPU (XPU) with a `torch` XPU build (e.g. `torch==2.11.0+xpu`).
- `vllm-xpu-kernels==0.1.7` (targets PyTorch 2.11).
- Intel oneAPI runtime (for the SYCL/oneDNN kernels shipped by vllm-xpu-kernels).

## Status

Scaffold with wired templates. Every wrapper must be verified against the exact
`vllm_xpu_kernels` v0.1.7 API and validated against the TokenSpeed numerics
reference before it is trusted.

Covered (Qwen3 dense bf16 path):

| Op | Module | Integration | State |
|----|--------|-------------|-------|
| MHA prefill | `ops/attention.py` | registry (`attention.mha_prefill`) | wired template |
| MHA paged decode | `ops/attention.py` | registry | NOT registered (falls back to Triton; layout TODO) |
| Rotary embedding | `ops/embedding.py` | registry (`embedding.rope`) | wired template |
| RMSNorm / fused-add | `ops/layernorm.py` | direct call (runtime prefers it on XPU) | wired template |
| SiLU-and-mul | `ops/activation.py` | direct call (runtime prefers it on XPU) | wired template |

Not yet covered: GEMM (bf16 already uses torch/oneDNN; quantized fp8/mxfp4 TODO),
MoE, sampling. Add following the same pattern.
