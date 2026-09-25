# Gluon Petit MegaMoE

This directory contains the vendored runtime closure of the Gluon MegaMoE
implementation derived from
[causalflow-ai/petit-kernel](https://github.com/causalflow-ai/petit-kernel),
whose package metadata identifies version 0.0.5. The upstream BSD 3-Clause
license is reproduced in `LICENSE.txt`.

The vendored files preserve Petit's source without import rewrites or kernel
changes. TokenSpeed's package boundary adds the vendor root to Python's module
search path so the upstream `lib` and `petit_kernel` imports resolve unchanged.
The package uses stock Triton 3.8.0 for the Gluon compiler contract used by the
upstream kernels.

The retained runtime supports the registered GFX950 MegaMoE configurations,
including GPT-OSS 120B (EP8, 128 experts, top-4, 2880x3072, biased OpenAI
SwiGLU) and DeepSeek V4 (EP8, 384 experts, top-6, 7168x3072, bias-free SiLU).
The HIP VMM binding is compiled on first use through
`torch.utils.cpp_extension`, so a ROCm development environment is required.

The TokenSpeed adapter lives in `tokenspeed_kernel.ops.moe.gluon.petit` and
registers this runtime as the explicit `petit_gluon` MoE solution.

## Integration status

The source is intentionally kept at the upstream Gluon kernel contract. The
currently pinned `tokenspeed-triton` compiler rejects two constructs used by
that contract: numeric LDS pointer address space `3` and the 8196-word shared
allocation used by MegaMoE. This port therefore uses stock Triton 3.8.0 until
the separately reviewed compiler compatibility fix is available in
`tokenspeed-triton`; the Petit kernel source is not adapted around either
compiler limitation.

## Runtime contract

Select the backend with both `--moe-backend petit_gluon` and
`--all2all-backend petit_gluon`. It requires one 8-GPU GFX950 node, EP8/TP1,
BF16 model activations, serialized MXFP4 expert weights, trivial expert
placement, and no more than 1024 tokens per rank. The registered profiles are
GPT-OSS 120B and DeepSeek V4. DeepSeek V4 uses Petit's unchanged unclamped
SiLU operation even when the checkpoint declares an activation clamp; the
runtime warns when it ignores such a clamp.
