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

"""Hopper W4A8 MoE: packed INT4 weights × dynamic per-token FP8 activations.

Serves ``compressed-tensors`` mixed-precision checkpoints whose routed experts
are symmetric INT4 group_size=128 (``pack-quantized`` uint4b8/int32) with
dynamic FP8 activations — e.g. GLM-5.3 W4A8.

The checkpoint keeps eight unsigned-offset INT4 values per ``int32`` word.
``process_weights`` repacks them into CUTLASS signed nibbles (two INT4 per
``int8``) without expanding VRAM. Apply prefers a CUTLASS grouped GEMM when
``sgl_kernel.cutlass_w4a8_moe_mm`` is importable; otherwise it dequantizes one
expert at a time into a scratch buffer (INT4 stays in memory).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

logger = logging.getLogger(__name__)

platform = current_platform()


def unpack_int32_uint4b8_to_cutlass_int8(weight_packed: torch.Tensor) -> torch.Tensor:
    """Convert compressed-tensors pack_to_int32 uint4b8 to CUTLASS int8 nibbles.

    ``pack_to_int32`` stores 8 unsigned-offset int4 values per int32.
    CUTLASS expects pairs of signed int4 packed into int8 (low nibble = even
    index, high nibble = odd index, two's complement).

    Args:
        weight_packed: ``[E, N, K // 8]`` int32.

    Returns:
        ``[E, N, K // 2]`` int8 in CUTLASS layout.
    """
    num_bits = 4
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    offset = 1 << (num_bits - 1)
    pair_factor = pack_factor // 2

    out = torch.empty(
        (*weight_packed.shape[:-1], weight_packed.shape[-1], pair_factor),
        dtype=torch.int8,
        device=weight_packed.device,
    )
    for pair_idx in range(pair_factor):
        low_shift = num_bits * (2 * pair_idx)
        high_shift = low_shift + num_bits
        low_nibbles = ((weight_packed >> low_shift) & mask) - offset
        high_nibbles = ((weight_packed >> high_shift) & mask) - offset
        out[..., pair_idx] = ((high_nibbles << 4) | (low_nibbles & 0x0F)).to(torch.int8)
    return out.flatten(-2).contiguous()


def interleave_w4a8_scales(scales: torch.Tensor) -> torch.Tensor:
    """Interleave group scales in groups of 4 for the CUTLASS W4A8 epilogue."""
    s_shape = scales.shape
    alignment = 4 if s_shape[2] % 4 == 0 else 1
    scales_interleaved = scales.reshape(
        s_shape[0], s_shape[1], (s_shape[2] // alignment), alignment
    )
    scales_interleaved = scales_interleaved.permute(0, 2, 1, 3)
    scales_interleaved = scales_interleaved.reshape(
        s_shape[0], s_shape[2] // alignment, s_shape[1] * alignment
    )
    return scales_interleaved.contiguous()


def dequant_cutlass_int4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize one expert's CUTLASS-packed INT4 weights to ``dtype``.

    Args:
        packed: ``[N, K // 2]`` int8.
        scale: ``[N, K // group_size]`` (typically bf16).
        dtype: Output dtype.
    """
    n, k_half = packed.shape
    k = k_half * 2
    packed_i32 = packed.to(torch.int32)
    low = packed_i32 & 0x0F
    high = (packed_i32 >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    weight = torch.empty(n, k, dtype=dtype, device=packed.device)
    weight[:, 0::2] = low.to(dtype)
    weight[:, 1::2] = high.to(dtype)
    num_groups = scale.shape[-1]
    group_size = k // num_groups
    weight = weight.view(n, num_groups, group_size) * scale.to(dtype).unsqueeze(-1)
    return weight.reshape(n, k)


def _try_sgl_cutlass_w4a8_moe(
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor | None:
    """Run SGLang's Hopper CUTLASS grouped GEMM when ``sgl_kernel`` is present."""
    try:
        from sgl_kernel import cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data
    except ImportError:
        return None

    device = x.device
    w1_q = w.w13_weight_packed
    w2_q = w.w2_weight_packed
    w1_scale = w.w13_weight_scale_interleaved
    w2_scale = w.w2_weight_scale_interleaved
    num_local_experts = w1_q.size(0)
    m = x.size(0)
    k = w1_q.size(2) * 2
    n = w2_q.size(2) * 2
    topk = topk_ids.size(1)

    a_strides1 = torch.full((num_local_experts, 3), k, device=device, dtype=torch.int64)
    c_strides1 = torch.full(
        (num_local_experts, 3), 2 * n, device=device, dtype=torch.int64
    )
    a_strides2 = torch.full((num_local_experts, 3), n, device=device, dtype=torch.int64)
    c_strides2 = torch.full((num_local_experts, 3), k, device=device, dtype=torch.int64)
    expert_offsets = torch.empty(
        num_local_experts + 1, dtype=torch.int32, device=device
    )
    problem_sizes1 = torch.empty(
        (num_local_experts, 3), dtype=torch.int32, device=device
    )
    problem_sizes2 = torch.empty(
        (num_local_experts, 3), dtype=torch.int32, device=device
    )
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    get_cutlass_w4a8_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_local_experts,
        n,
        k,
    )

    a_fp8 = x.to(torch.float8_e4m3fn)
    a1_scale = torch.ones(1, dtype=torch.float32, device=device)
    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
    cutlass_w4a8_moe_mm(
        c1,
        a_fp8,
        w1_q,
        a1_scale,
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        a_strides1,
        a_strides1,
        c_strides1,
        c_strides1,
        128,
        topk,
    )
    gate, up = c1.chunk(2, dim=-1)
    intermediate = F.silu(gate) * up
    intermediate_fp8 = intermediate.to(torch.float8_e4m3fn)
    a2_scale = torch.ones(1, dtype=torch.float32, device=device)
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)
    cutlass_w4a8_moe_mm(
        c2,
        intermediate_fp8,
        w2_q,
        a2_scale,
        w2_scale,
        expert_offsets[:-1],
        problem_sizes2,
        a_strides2,
        a_strides2,
        c_strides2,
        c_strides2,
        128,
        topk,
    )
    routed = c2.view(m, topk, k) * topk_weights.to(c2.dtype).unsqueeze(-1)
    return routed.sum(dim=1).to(x.dtype)


def _torch_w4a8_moe_apply(
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-expert INT4 dequant + SiLU-and-mul GEMM. Keeps packed weights in VRAM."""
    output = torch.zeros_like(x)
    unique_experts = torch.unique(topk_ids)
    for expert_id in unique_experts.tolist():
        expert_id = int(expert_id)
        if expert_id < 0 or expert_id >= w.w13_weight_packed.shape[0]:
            continue
        token_mask = topk_ids == expert_id
        if not bool(token_mask.any()):
            continue
        token_index, slot = torch.where(token_mask)
        tokens = x.index_select(0, token_index)
        w13 = dequant_cutlass_int4(
            w.w13_weight_packed[expert_id],
            w.w13_weight_scale[expert_id],
            x.dtype,
        )
        w2 = dequant_cutlass_int4(
            w.w2_weight_packed[expert_id],
            w.w2_weight_scale[expert_id],
            x.dtype,
        )
        gate_up = F.linear(tokens, w13)
        gate, up = gate_up.chunk(2, dim=-1)
        down = F.linear(F.silu(gate) * up, w2)
        scaled = down * topk_weights[token_index, slot].to(down.dtype).unsqueeze(-1)
        output.index_add_(0, token_index, scaled.to(output.dtype))
    return output


def flashinfer_cutlass_w4a8_moe_weights(plan: dict, w: torch.nn.Module):
    """Repack uint4b8/int32 expert weights into CUTLASS signed int8 nibbles."""
    del plan
    if getattr(w, "is_w4a8_converted", False):
        return None
    if w.w13_weight_packed.dtype == torch.int32:
        w.w13_weight_packed = torch.nn.Parameter(
            unpack_int32_uint4b8_to_cutlass_int8(w.w13_weight_packed.data),
            requires_grad=False,
        )
        w.w2_weight_packed = torch.nn.Parameter(
            unpack_int32_uint4b8_to_cutlass_int8(w.w2_weight_packed.data),
            requires_grad=False,
        )
    w.w13_weight_scale = torch.nn.Parameter(
        w.w13_weight_scale.data.to(torch.bfloat16), requires_grad=False
    )
    w.w2_weight_scale = torch.nn.Parameter(
        w.w2_weight_scale.data.to(torch.bfloat16), requires_grad=False
    )
    w.w13_weight_scale_interleaved = interleave_w4a8_scales(w.w13_weight_scale.data)
    w.w2_weight_scale_interleaved = interleave_w4a8_scales(w.w2_weight_scale.data)
    for shape_name in ("w13_weight_shape", "w2_weight_shape"):
        if hasattr(w, shape_name):
            delattr(w, shape_name)
    w.is_w4a8_converted = True
    return None


if platform.is_nvidia:

    @register_kernel(
        "moe",
        "apply",
        name="cutlass_w4a8_moe_apply",
        solution="flashinfer_cutlass",
        weight_preprocessor=flashinfer_cutlass_w4a8_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(9, 0),
            max_arch_version=ArchVersion(9, 0),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"w4a8"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({128}),
            "internal_activation_dtype": frozenset({"fp8"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def cutlass_w4a8_moe_apply(
        plan: dict,
        x: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        num_tokens_global: int | None = None,
        max_num_tokens_per_gpu: int | None = None,
        do_finalize: bool = True,
        enable_pdl: bool = False,
    ):
        del plan, num_tokens_global, max_num_tokens_per_gpu, do_finalize, enable_pdl
        if x.shape[0] == 0:
            return x
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        cutlass_out = _try_sgl_cutlass_w4a8_moe(x, w, topk_weights, topk_ids)
        if cutlass_out is not None:
            return cutlass_out
        return _torch_w4a8_moe_apply(x, w, topk_weights, topk_ids)
