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

"""Packed FP8 MoE schedule selection and stage launches for gfx950."""

from collections.abc import Callable

import torch
from tokenspeed_kernel_amd._triton import triton


def _packed_schedule(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    compute_units: int,
    num_experts_local: int,
    num_experts_total: int,
) -> tuple[str, int]:
    """Select tiny warp, small split, or large expert computation."""
    if 0 < num_tokens <= 4 and num_experts_local == num_experts_total:
        return "warp", 1
    if num_tokens > 32:
        return "large", 1
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_small import (
        packed_fp8_split_count,
    )

    return "small", packed_fp8_split_count(
        num_tokens,
        hidden_size,
        intermediate_size,
        topk,
        compute_units,
        num_experts_local,
        num_experts_total,
    )


def launch_gluon_fp8_packed_moe_gfx950(
    hidden_states: torch.Tensor,
    w13_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    swiglu_limit: float,
    expert_start: int,
    *,
    num_experts_total: int,
    quantize_activation: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Launch packed FP8 experts with fused intermediate conversion.

    Tiny fully local batches retain BF16 activations and use FP32 vector math.
    Other batches retain BF16 activation rounding, then emit FP8 values and per-128
    scales in the producing kernel, with shared input quantization beforehand.

    Args:
        hidden_states: BF16 input rows ``[tokens, hidden]``.
        w13_packed, w2_packed: N16/K32-packed E4M3 expert parameters.
        w13_scale, w2_scale: FP32 inverse scales for 128-by-128 weight blocks.
        topk_ids, topk_weights: Global expert IDs and route weights.
        swiglu_limit: Positive gate/up clamp before the activation.
        expert_start: First expert owned by this rank.
        num_experts_total: Declared global expert count, used with the packed
            bank's local count to estimate local routing work.
        quantize_activation: Shared token-group FP8 quantizer supplied by the
            caller, keeping this package independent of tokenspeed-kernel.

    Returns:
        BF16 contribution from this rank's local experts, matching the input shape.
    """
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
        moe_align_block_size_device,
    )

    num_tokens, hidden_size = hidden_states.shape
    num_experts, twice_intermediate, _ = w13_packed.shape
    intermediate_size = twice_intermediate // 2
    topk = topk_ids.shape[1]
    compute_units = torch.cuda.get_device_properties(
        hidden_states.device
    ).multi_processor_count
    schedule, split_count = _packed_schedule(
        num_tokens,
        hidden_size,
        intermediate_size,
        topk,
        compute_units,
        num_experts,
        num_experts_total,
    )
    if schedule == "warp":
        from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.warp_decode import (
            gluon_fp8_block_warp_decode_moe,
        )

        return gluon_fp8_block_warp_decode_moe(
            hidden_states,
            w13_packed,
            w2_packed,
            w13_scale,
            w2_scale,
            topk_ids,
            topk_weights,
            swiglu_limit,
            expert_start,
            False,
            packed_weights=True,
        )
    block_m = 64 if schedule == "large" else 16
    use_nt = num_tokens * topk // num_experts < 64 if schedule == "large" else False
    out = torch.empty_like(hidden_states)
    if schedule == "large" or num_tokens > 16:
        ids, experts, weights, num_valid = moe_align_block_size_device(
            topk_ids,
            topk_weights,
            num_experts,
            block_m,
            expert_start=expert_start,
        )
    else:
        from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_fused import (
            moe_align_block_size_fused,
        )

        ids, experts, weights, num_valid = moe_align_block_size_fused(
            topk_ids,
            topk_weights,
            num_experts,
            block_m,
            expert_start=expert_start,
        )
    out.zero_()
    x_fp8, x_scales = quantize_activation(hidden_states)
    em = ids.numel()
    use_reciprocal = (
        compute_units == 256
        and num_tokens * topk * intermediate_size // 128 >= 32 * compute_units
    )
    inter_fp8 = torch.empty(
        (num_tokens * topk, intermediate_size),
        dtype=torch.float8_e4m3fn,
        device=hidden_states.device,
    )
    inter_scales = torch.empty(
        (num_tokens * topk, intermediate_size // 128),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    if split_count > 1:
        from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_small import (
            gluon_fp8_packed_small_activation_gfx950,
            gluon_fp8_packed_small_stage1_gfx950,
        )

        gate_up = torch.zeros(
            (num_tokens * topk, 2 * intermediate_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        gluon_fp8_packed_small_stage1_gfx950[
            (2 * intermediate_size // 128, triton.cdiv(em, 16), split_count)
        ](
            x_fp8.view(torch.uint8),
            x_scales,
            w13_packed.view(torch.uint8),
            w13_scale,
            gate_up,
            inter_scales,
            ids,
            experts,
            num_valid,
            num_tokens,
            em,
            swiglu_limit,
            K=hidden_size,
            I=intermediate_size,
            TOPK=topk,
            SPLIT_COUNT=split_count,
            USE_RECIPROCAL=use_reciprocal,
            num_warps=4,
        )
        gluon_fp8_packed_small_activation_gfx950[
            (triton.cdiv(num_tokens * topk * intermediate_size, 256),)
        ](
            gate_up,
            inter_fp8,
            inter_scales,
            num_tokens * topk,
            swiglu_limit,
            I=intermediate_size,
            BLOCK=256,
            USE_RECIPROCAL=use_reciprocal,
        )
    else:
        if schedule == "large":
            from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_large import (
                gluon_fp8_packed_large_stage1_gfx950,
            )

            stage1 = gluon_fp8_packed_large_stage1_gfx950
            stage1_options = {"USE_NT": use_nt, "num_stages": 1}
        else:
            from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_small import (
                gluon_fp8_packed_small_stage1_gfx950,
            )

            stage1 = gluon_fp8_packed_small_stage1_gfx950
            stage1_options = {"SPLIT_COUNT": 1}
        stage1[(triton.cdiv(em, block_m) * (intermediate_size // 128),)](
            x_fp8.view(torch.uint8),
            x_scales,
            w13_packed.view(torch.uint8),
            w13_scale,
            inter_fp8,
            inter_scales,
            ids,
            experts,
            num_valid,
            num_tokens,
            em,
            swiglu_limit,
            K=hidden_size,
            I=intermediate_size,
            TOPK=topk,
            USE_RECIPROCAL=use_reciprocal,
            num_warps=4,
            **stage1_options,
        )

    if schedule == "large":
        from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_large import (
            gluon_fp8_packed_large_stage2_gfx950,
        )

        stage2 = gluon_fp8_packed_large_stage2_gfx950
        stage2_options = {"USE_NT": use_nt, "num_stages": 1}
    else:
        from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_small import (
            gluon_fp8_packed_small_stage2_gfx950,
        )

        stage2 = gluon_fp8_packed_small_stage2_gfx950
        stage2_options = {}
    stage2[(triton.cdiv(em, block_m) * (hidden_size // 128),)](
        inter_fp8.view(torch.uint8),
        inter_scales,
        w2_packed.view(torch.uint8),
        w2_scale,
        out,
        ids,
        experts,
        weights,
        num_valid,
        num_tokens,
        em,
        K=intermediate_size,
        N=hidden_size,
        TOPK=topk,
        num_warps=4,
        **stage2_options,
    )
    return out
