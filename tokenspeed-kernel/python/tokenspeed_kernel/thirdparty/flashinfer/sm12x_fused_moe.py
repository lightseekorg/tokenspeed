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


@dataclass
class Sm12xFusedMoEWeights:
    w13: torch.Tensor
    sf13: torch.Tensor
    w2: torch.Tensor
    sf2: torch.Tensor
    limit: float | None


def _pack_scales(scales: torch.Tensor) -> torch.Tensor:
    experts, rows, groups = scales.shape
    return (
        scales.reshape(experts, rows, groups // 4, 4)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(experts, groups // 4, rows * 4)
    )


def prepare_sm12x_fused_moe(w: torch.nn.Module) -> Sm12xFusedMoEWeights:
    """Prepare native up/gate weights once without keeping a second weight copy."""
    if w.w13_input_layout != "concatenated":
        raise ValueError("SM12x fused MoE requires canonical gate/up weights")
    if w.activation != "swiglu":
        raise ValueError("SM12x fused MoE requires SwiGLU")
    if (
        w.swiglu_arg is not None and w.swiglu_arg.alpha not in (None, 1.0)
    ) or w.swiglu_beta not in (None, 0.0):
        raise ValueError("SM12x fused MoE requires standard SwiGLU")
    if w.w13_weight.shape[-1] * 2 % 512 or w.w2_weight.shape[-1] * 2 % 128:
        raise ValueError("SM12x fused MoE requires H aligned to 512 and I to 128")
    gate, up = w.w13_weight.chunk(2, 1)
    swapped = torch.cat((up, gate), 1)
    w.w13_weight.copy_(swapped)
    del swapped
    gate_sf, up_sf = w.w13_weight_scale.chunk(2, 1)
    return Sm12xFusedMoEWeights(
        w.w13_weight,
        _pack_scales(torch.cat((up_sf, gate_sf), 1)),
        w.w2_weight,
        _pack_scales(w.w2_weight_scale),
        None if w.swiglu_arg is None else w.swiglu_arg.limit,
    )


def apply_sm12x_fused_moe(
    x: torch.Tensor,
    state: Sm12xFusedMoEWeights,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    """Run local expert contributions; negative IDs are unrouted."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1 import (
        cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4,
        out_sf_shape,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize import (
        cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_q0_route_triton import (
        make_mxfp8_q0_route_workspace,
        mxfp8_q0_route_triton,
    )

    experts, hidden, packed_intermediate = state.w2.shape
    intermediate = packed_intermediate * 2
    ws = make_mxfp8_q0_route_workspace(x, local_ids, experts)
    # Unrouted rows otherwise retain allocator contents. Initialize the fixed
    # capacity on every invocation, including replays with changed routing.
    for tensor in vars(ws).values():
        tensor.zero_()
    offsets, token_map, route_weights, aq, asc = mxfp8_q0_route_triton(
        x,
        local_ids,
        weights,
        experts,
        workspace=ws,
        enable_pdl=enable_pdl,
    )
    q1 = torch.zeros(
        (local_ids.numel(), intermediate), dtype=torch.float8_e4m3fn, device=x.device
    )
    sf1 = torch.zeros(
        out_sf_shape(local_ids.numel(), intermediate, experts),
        dtype=torch.int32,
        device=x.device,
    )
    # A singleton input may be contiguous with row stride zero. CuTe
    # requires a canonical output stride when marking compact dimensions.
    output = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
        aq,
        asc,
        state.w13,
        state.sf13,
        offsets,
        tune=False,
        enable_pdl=enable_pdl,
        swiglu_limit=state.limit,
        out_q=q1,
        out_sf=sf1,
    )
    cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4(
        q1,
        sf1,
        state.w2,
        state.sf2,
        offsets,
        token_map,
        route_weights,
        x.shape[0],
        tune=False,
        enable_pdl=enable_pdl,
        out=output,
    )
    return output
