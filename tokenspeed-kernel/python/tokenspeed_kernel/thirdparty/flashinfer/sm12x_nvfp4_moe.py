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


from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.tuning import local_autotune


@dataclass(frozen=True)
class Sm12xNvfp4MoEWeights:
    """Packed CUTLASS weights; activation scales remain per local expert."""

    w13: torch.Tensor
    sf13: torch.Tensor
    w2: torch.Tensor
    sf2: torch.Tensor
    fc1_quant: torch.Tensor
    fc1_alpha: torch.Tensor
    fc2_quant: torch.Tensor
    fc2_alpha: torch.Tensor
    swiglu_alpha: torch.Tensor
    swiglu_beta: torch.Tensor
    swiglu_limit: torch.Tensor


def _swizzle(scales: torch.Tensor) -> torch.Tensor:
    e, n, k = scales.shape
    np, kp = (n + 127) // 128 * 128, (k + 3) // 4 * 4
    padded = torch.zeros((e, np, kp), dtype=scales.dtype, device=scales.device)
    padded[:, :n, :k] = scales
    return (
        padded.reshape(e, np // 128, 4, 32, kp // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(e, np, kp)
    )


def prepare_sm12x_nvfp4(w: torch.nn.Module) -> Sm12xNvfp4MoEWeights:
    """Convert loader [gate; up] weights once; never dequantize weights."""
    e, h, ip = w.w2_weight.shape
    if w.w13_input_layout != "concatenated" or w.activation != "swiglu":
        raise ValueError("NVFP4 requires canonical gate/up SwiGLU weights")
    if h % 128 or ip * 2 % 128:
        raise ValueError("SM120 NVFP4 requires H and I divisible by 128")
    if w.w13_input_scale.shape != (e,) or w.w2_input_scale.shape != (e,):
        raise ValueError("NVFP4 FC1/FC2 calibrated input scales must be per expert")
    if not torch.equal(w.w13_weight_scale_2[:, 0], w.w13_weight_scale_2[:, 1]):
        raise ValueError("Fused NVFP4 w1/w3 global scales must agree")
    if w.swiglu_arg is None or w.swiglu_arg.limit is None:
        raise ValueError("DeepSeek NVFP4 requires an explicit SwiGLU clamp")
    if w.swiglu_arg.alpha not in (None, 1.0) or w.swiglu_beta not in (None, 0.0):
        raise ValueError("DeepSeek NVFP4 requires alpha=1 and beta=0")
    for t in (
        w.w13_input_scale,
        w.w2_input_scale,
        w.w13_weight_scale_2,
        w.w2_weight_scale_2,
    ):
        if not torch.isfinite(t).all() or not (t > 0).all():
            raise ValueError("NVFP4 global scales must be finite and positive")
    if w.w13_weight.dtype != torch.uint8 or w.w2_weight.dtype != torch.uint8:
        raise TypeError("NVFP4 weights must remain packed E2M1 bytes")
    if (
        w.w13_weight_scale.dtype != torch.float8_e4m3fn
        or w.w2_weight_scale.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("NVFP4 block scales must be G16 E4M3")
    if w.w13_weight_scale.shape != (
        *w.w13_weight.shape[:2],
        h // 16,
    ) or w.w2_weight_scale.shape != (e, h, ip // 8):
        raise ValueError("Invalid G16 NVFP4 weight scale layout")
    gate, up = w.w13_weight.chunk(2, 1)
    w.w13_weight.copy_(torch.cat((up, gate), 1))
    gate_sf, up_sf = w.w13_weight_scale.chunk(2, 1)
    sf13 = _swizzle(torch.cat((up_sf, gate_sf), 1))
    sf2 = _swizzle(w.w2_weight_scale)
    device = w.w2_weight.device
    return Sm12xNvfp4MoEWeights(
        w.w13_weight,
        sf13,
        w.w2_weight,
        sf2,
        w.w13_input_scale.reciprocal().contiguous(),
        (w.w13_input_scale * w.w13_weight_scale_2[:, 0]).contiguous(),
        w.w2_input_scale.reciprocal().contiguous(),
        (w.w2_input_scale * w.w2_weight_scale_2).contiguous(),
        torch.ones(e, dtype=torch.float32, device=device),
        torch.zeros(e, dtype=torch.float32, device=device),
        torch.full((e,), w.swiglu_arg.limit, dtype=torch.float32, device=device),
    )


def apply_sm12x_nvfp4(
    x: torch.Tensor,
    state: Sm12xNvfp4MoEWeights,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    """Compute local routed contributions with two NVFP4 x NVFP4 GEMMs.

    x is BF16 token-major; CUTLASS expands routes and quantizes each expert's
    activation using its own calibrated scale. Negative IDs are unrouted.
    Routing weights are applied once by this operation.
    """
    from flashinfer import ActivationType, cutlass_fused_moe

    if x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("SM120 NVFP4 expects contiguous BF16 rows before quantization")
    if local_ids.shape != weights.shape or local_ids.shape[0] != x.shape[0]:
        raise ValueError("NVFP4 routes must match input rows")
    if not x.shape[0]:
        return torch.empty_like(x)
    output = torch.empty_like(x)
    # DeepEP already dispatched routes; local profile counts may differ by rank.
    with local_autotune():
        return cutlass_fused_moe(
            input=x,
            output=output,
            token_selected_experts=local_ids.to(torch.int32).contiguous(),
            token_final_scales=weights.float().contiguous(),
            fc1_expert_weights=state.w13.view(torch.int64),
            fc2_expert_weights=state.w2.view(torch.int64),
            output_dtype=torch.bfloat16,
            input_sf=None,
            quant_scales=[
                state.fc1_quant,
                state.sf13.view(torch.int32),
                state.fc1_alpha,
                state.fc2_quant,
                state.sf2.view(torch.int32),
                state.fc2_alpha,
            ],
            ep_size=1,
            ep_rank=0,
            tp_size=1,
            tp_rank=0,
            activation_type=ActivationType.SwigluBias,
            swiglu_alpha=state.swiglu_alpha,
            swiglu_beta=state.swiglu_beta,
            swiglu_limit=state.swiglu_limit,
            enable_pdl=enable_pdl,
        )[0]
