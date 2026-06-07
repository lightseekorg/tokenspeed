# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Masked grouped MoE computation using FlashInfer CuteDSL for NVFP4."""

import torch
from tokenspeed_kernel.registry import error_fn

try:
    from tokenspeed_kernel.ops.moe.flashinfer import (
        grouped_gemm_nt_masked,
        scaled_fp4_grouped_quantize,
        silu_and_mul_scaled_nvfp4_experts_quantize,
    )

    _cutedsl_masked_available = all(
        fn is not error_fn
        for fn in (
            grouped_gemm_nt_masked,
            scaled_fp4_grouped_quantize,
            silu_and_mul_scaled_nvfp4_experts_quantize,
        )
    )
except ImportError:
    _cutedsl_masked_available = False


def _get_cute_dtype(t: torch.Tensor) -> str:
    if t.dtype == torch.bfloat16:
        return "bfloat16"
    elif t.dtype == torch.float16:
        return "float16"
    elif t.dtype == torch.float32:
        return "float32"
    else:
        raise ValueError(f"Unsupported cute dtype {t.dtype}")


def cutedsl_moe_masked(
    hidden_states: tuple[torch.Tensor, torch.Tensor | None],
    input_global_scale: torch.Tensor | None,
    w1: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2: torch.Tensor,
    a2_global_scale: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alpha: torch.Tensor,
    masked_m: torch.Tensor,
):
    """Masked grouped MoE: DeepEP low-latency dispatch output -> CuteDSL NVFP4 GEMM.

    Args:
        hidden_states: Either
            * (bf16_tensor[num_experts, m, k], None)  -- needs internal FP4 quant
            * (uint8_tensor, float8_scale)             -- pre-quantized FP4
        input_global_scale: Per-expert global scale for FP4 quantization (None if pre-quantized).
        w1: FP4 packed weights [num_experts, 2*n, k//2], uint8.
        w1_blockscale: Block scale factors, float8_e4m3fn.
        w1_alpha: Per-expert alpha [num_experts], float32.
        w2: FP4 packed weights [num_experts, k, n//2], uint8.
        a2_global_scale: Per-expert global scale for down-proj input [num_experts], float32.
        w2_blockscale: Block scale factors, float8_e4m3fn.
        w2_alpha: Per-expert alpha [num_experts], float32.
        masked_m: Per-expert token count [num_experts].
    """
    if not _cutedsl_masked_available:
        raise ImportError(
            "flashinfer CuteDSL masked grouped GEMM is unavailable. "
            "Install a flashinfer build with cute_dsl support."
        )

    if input_global_scale is not None and not input_global_scale.is_contiguous():
        input_global_scale = input_global_scale.contiguous()
    if not a2_global_scale.is_contiguous():
        a2_global_scale = a2_global_scale.contiguous()

    n = w2.shape[-1] * 2  # intermediate dimension

    if hidden_states[1] is not None:
        a_q = hidden_states[0].view(torch.uint8)
        a_q_sf = hidden_states[1].view(torch.float8_e4m3fn)
        m, k_by_2, num_experts = a_q.shape
        k = k_by_2 * 2
    else:
        num_experts, m, k = hidden_states[0].shape
        a_q, a_q_sf = scaled_fp4_grouped_quantize(
            hidden_states[0],
            masked_m,
            input_global_scale,
        )

    sf_vec_size = 16
    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    c_dtype = "bfloat16"

    # Gemm1: up-proj + gate-proj
    gateup_output = torch.empty(
        (num_experts, m, n * 2), dtype=torch.bfloat16, device=a_q.device
    )
    gateup_output = gateup_output.permute(1, 2, 0)
    grouped_gemm_nt_masked(
        (a_q, a_q_sf),
        (w1.permute(1, 2, 0), w1_blockscale),
        gateup_output,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w1_alpha.view(1, 1, num_experts),
        alpha_dtype=_get_cute_dtype(w1_alpha),
    )

    # SiLU-and-mul + FP4 quantization for down-proj input
    diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
        gateup_output.permute(2, 0, 1),
        masked_m,
        a2_global_scale,
    )

    # Gemm2: down-proj
    out = torch.empty((num_experts, m, k), dtype=torch.bfloat16, device=a_q.device)
    out = out.permute(1, 2, 0)
    grouped_gemm_nt_masked(
        (diq, diq_sf),
        (w2.permute(1, 2, 0), w2_blockscale),
        out,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w2_alpha.view(1, 1, num_experts),
        alpha_dtype=_get_cute_dtype(w2_alpha),
    )

    return out.permute(2, 0, 1)  # [num_experts, m, k]
