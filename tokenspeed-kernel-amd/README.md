# AMD-specific High Performance Kernels

This directory contains high-performance kernel implementations for AMD GPUs.
Kernels are organized by hardware generation, operator, and stable data format.

```text
python/tokenspeed_kernel_amd/ops/
  gfx950/
    attention/
    gemm/
      fp16/
    moe/
      fp16/
      mxfp4/
      _common.py
    sampling/
  gfx1250/
    attention/
    moe/
      mxfp4/
      _common.py
```

The `fp16` directory is the umbrella for 16-bit floating-point formats,
including FP16 and BF16. MoE directories describe the weight format. The
`mxfp4` package may use BF16, FP8, or dynamically quantized MXFP4 activations
internally. Implementations are not shared between hardware generations.

Tests are organized by operator and weight format. Generation-specific kernels
are selected inside otherwise hardware-neutral test cases.
