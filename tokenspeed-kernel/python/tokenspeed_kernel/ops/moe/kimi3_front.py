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

"""Kimi-K3 fused-front preparation kernels."""

import torch
from tokenspeed_kernel.thirdparty.cuda import moe_route_pack_quant_mxfp8


def kimi3_route_pack_quant_mxfp8(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_input: torch.Tensor,
    *,
    routed_scaling_factor: float,
    renormalize: bool,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route and quantize strided views from Kimi-K3's FP32 fused front.

    Args:
        router_logits: Inner-contiguous FP32 logits shaped ``[M, 896]``.
        correction_bias: Contiguous FP32 selection bias shaped ``[896]``.
        routed_input: Inner-contiguous FP32/BF16 activation shaped
            ``[M, 3584]``.
        routed_scaling_factor: Scale applied after optional normalization.
        renormalize: Normalize the selected sigmoid weights when true.
        enable_pdl: Allow programmatic dependent launch from the front GEMM.

    Returns:
        BF16 route weights, INT32 route ids, packed TRT-LLM routes, MXFP8
        activations, and flattened UE8M0 scales.
    """
    return moe_route_pack_quant_mxfp8(
        router_logits,
        correction_bias,
        routed_input,
        routed_scaling_factor=routed_scaling_factor,
        renormalize=renormalize,
        enable_pdl=enable_pdl,
    )


__all__ = ["kimi3_route_pack_quant_mxfp8"]
