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

"""FlatQuant INT4 weight preparation and composed Prefill experts."""

import math
from dataclasses import dataclass, replace

import torch


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack eight signed [-8,7] values into each INT32, low nibble first."""
    if (
        values.ndim == 0
        or values.shape[-1] % 8
        or values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError("INT4 packing requires integer rows divisible by eight")
    if torch.any((values < -8) | (values > 7)):
        raise ValueError("INT4 values must be in [-8,7]")
    chunks = values.to(torch.int64).reshape(*values.shape[:-1], -1, 8) & 15
    shifts = torch.arange(8, device=values.device, dtype=torch.int64) * 4
    return (chunks << shifts).sum(dim=-1).to(torch.int32)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Decode signed INT4 without interpreting INT32 as a numeric weight."""
    if packed.ndim == 0 or packed.dtype != torch.int32:
        raise ValueError(
            "Packed INT4 must be an INT32 tensor with at least one dimension"
        )
    # Scalar shifts avoid the broadcast-output bug in the 910B torch_npu
    # RightShift binding. This helper runs only during checkpoint preparation.
    nibbles = torch.stack([(packed >> shift) & 15 for shift in range(0, 32, 4)], dim=-1)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    return signed.reshape(*packed.shape[:-1], packed.shape[-1] * 8).to(torch.int8)


@dataclass(frozen=True)
class FlatQuantStage:
    # left is pre-transposed, matching the native FlatQuant matrix contract.
    left: torch.Tensor
    right: torch.Tensor
    clip: float


@dataclass(frozen=True)
class W4A4MoePlan:
    w13: torch.Tensor
    w2: torch.Tensor
    # FP32 Decode scales are omitted from a Prefill-only grouped view.
    scale13: torch.Tensor | None
    scale2: torch.Tensor | None
    first: FlatQuantStage
    second: FlatQuantStage
    transposed: bool
    # Decode keeps its checkpoint packing; official GMM needs NN/N-packed INT4.
    prefill_weights: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass(frozen=True)
class W4A4GroupedPlan:
    weights: W4A4MoePlan
    scale13: torch.Tensor
    scale2: torch.Tensor


def prepare_w4a4_grouped_plan(
    weights: W4A4MoePlan, *, prefill_only: bool
) -> W4A4GroupedPlan:
    """Select the Prefill NZ weights and encode the official GMM scales."""

    def encode(scale):
        bits = scale.cpu().contiguous().view(torch.int32).to(torch.int64)
        return (bits & 0xFFFFFFFF).view(torch.uint64).unsqueeze(1).to(scale.device)

    if weights.prefill_weights is not None:
        weights = replace(
            weights,
            w13=weights.prefill_weights[0],
            w2=weights.prefill_weights[1],
            transposed=False,
            prefill_weights=None,
        )
    scale13, scale2 = encode(weights.scale13), encode(weights.scale2)
    if prefill_only:
        weights = replace(weights, scale13=None, scale2=None)
    return W4A4GroupedPlan(weights, scale13, scale2)


def w4a4_grouped_experts(
    x,
    ids,
    weights,
    *,
    expert_start: int,
    num_experts: int,
    plan: W4A4GroupedPlan,
):
    """Return local BF16 expert partials using official routing and GMMs.

    ``ids`` contains group-local expert IDs, with the same [tokens, topk]
    shape as ``weights``. Only the interval starting at ``expert_start`` and
    owned by ``plan.weights`` contributes. ``num_experts`` includes identity
    experts, which stay outside that interval. No EP communication happens
    here. FlatQuant13 runs once per input token before routing; NZ weights
    retain their prepared storage. Official GMM's BF16 output rounds through
    FP16, unlike the fused operator's INT32/vector dequantization.
    """
    import flash_ops  # noqa: F401
    import torch_npu

    state = plan.weights
    local_experts = state.w13.shape[0]
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or ids.ndim != 2
        or ids.shape != weights.shape
        or ids.shape[0] != x.shape[0]
        or ids.shape[1] == 0
        or ids.dtype != torch.int32
        or weights.dtype not in (torch.bfloat16, torch.float32)
        or ids.device != x.device
        or weights.device != x.device
        or expert_start < 0
        or expert_start + local_experts > num_experts
    ):
        raise ValueError("Invalid W4A4 expert routing inputs or local interval")
    if x.shape[0] == 0:
        return torch.empty_like(x)

    def quantize(value, stage, *, counts):
        packed, scales = torch.ops.custom.flat_quant(
            value.reshape(value.shape[0], stage.left.shape[0], stage.right.shape[0]),
            stage.left,
            stage.right,
            clip_ratio=stage.clip,
            dst_dtype=16,
            group_list=counts,
            group_list_type=1,
        )
        return packed.flatten(1), scales.float()

    def matmul(value, scales, weight, weight_scale):
        if state.transposed:
            weight = weight.transpose(-1, -2)
        return torch_npu.npu_grouped_matmul(
            [value],
            [weight],
            scale=[weight_scale],
            per_token_scale=[scales],
            group_list=counts,
            split_item=3,
            output_dtype=torch.bfloat16,
            group_type=0,
            group_list_type=1,
        )[0]

    packed, scales = quantize(x, state.first, counts=None)
    # quant_mode=-1 copies packed INT4 as opaque INT8 bytes and applies the
    # same permutation to FP32 token scales; it does not quantize again.
    expanded, rows, counts, scales = torch_npu.npu_moe_init_routing_v2(
        packed.view(torch.int8),
        ids,
        scale=scales,
        active_num=ids.numel(),
        expert_num=num_experts,
        quant_mode=-1,
        active_expert_range=[expert_start, expert_start + local_experts],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        drop_pad_mode=0,
    )
    packed = expanded.view(torch.int32)
    hidden = matmul(packed, scales, state.w13, plan.scale13)
    # The capacity tail can be unwritten when other shards or identity
    # experts own routes. Group-aware consumers read only the valid prefix.
    hidden = torch_npu.mlp_split_swiglu(
        hidden,
        counts.to(torch.int32),
        local_exp_start=0,
        local_exp_end=local_experts,
    )
    packed, scales = quantize(hidden, state.second, counts=counts)
    hidden = matmul(packed, scales, state.w2, plan.scale2)
    return torch_npu.npu_moe_finalize_routing(
        hidden.unsqueeze(1), None, None, None, weights, rows, ids, 3
    )


def prepare_flatquant_stage(left, right, clip) -> FlatQuantStage:
    """Validate and prepare constants once, before graph capture."""
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[0] != left.shape[1]
        or right.shape[0] != right.shape[1]
        or left.dtype != torch.bfloat16
        or right.dtype != torch.bfloat16
        or left.device != right.device
        or clip.device != left.device
        or clip.numel() != 1
        or not clip.is_floating_point()
    ):
        raise ValueError("Invalid FlatQuant matrices/clip for the activation width")
    ratio = float(clip.item())
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("FlatQuant clip must be finite and in (0,1]")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("FlatQuant matrices must be finite")
    return FlatQuantStage(left.t().contiguous(), right.contiguous(), ratio)


def prepare_w4a4_moe_weights(
    w13, w2, scale13, scale2, first, second, *, transposed: bool, role: str
) -> W4A4MoePlan:
    """Convert checkpoint INT32 tensors to NZ without changing their INT4 layout.

    Input checkpoint shapes are [E,N,K/8] (transposed) or [E,N/8,K].
    Fused Decode uses [E,K,N/8] or [E,N,K/8] according to ``transposed``.
    Prefill needs the eight packed output channels contiguous in each word.
    The unified role ("null") retains both layouts for NT checkpoints. Scales
    remain FP32 [E,N] until the grouped plan encodes them for official GMM.
    """
    if role not in ("prefill", "decode", "null"):
        raise ValueError(f"Unsupported W4A4 service role {role!r}")
    if type(transposed) is not bool:
        raise ValueError("W4A4 transpose selection must be a boolean")
    if (
        w13.ndim != 3
        or w2.ndim != 3
        or w13.dtype != torch.int32
        or w2.dtype != torch.int32
    ):
        raise ValueError("W4A4 weights must be rank-three packed INT32")
    e = w13.shape[0]
    h = w13.shape[2] * 8 if transposed else w13.shape[2]
    i = w13.shape[1] // 2 if transposed else w13.shape[1] * 4
    expected13 = (e, 2 * i, h // 8) if transposed else (e, 2 * i // 8, h)
    expected2 = (e, h, i // 8) if transposed else (e, h // 8, i)
    if (
        min(e, h, i) <= 0
        or h % 8
        or i % 8
        or w13.shape != expected13
        or w2.shape != expected2
    ):
        raise ValueError("Inconsistent W4A4 MM13/MM2 packed shapes")
    if scale13.shape != (e, 2 * i) or scale2.shape != (e, h):
        raise ValueError("W4A4 scales must match unpacked output channels")
    tensors = (
        w13,
        w2,
        scale13,
        scale2,
        first.left,
        first.right,
        second.left,
        second.right,
    )
    if any(t.device != w13.device for t in tensors):
        raise ValueError("All W4A4 tensors must be on the same device")
    for scale in (scale13, scale2):
        if (
            scale.dtype != torch.float32
            or not torch.isfinite(scale).all()
            or torch.any(scale <= 0)
        ):
            raise ValueError("W4A4 scales must be finite positive FP32")
    if (
        first.left.shape[0] * first.right.shape[0] != h
        or second.left.shape[0] * second.right.shape[0] != i
    ):
        raise ValueError("FlatQuant transforms do not match the GEMM input widths")
    import torch_npu

    def values_to_nz(values):
        experts, rows, cols = values.shape
        if rows % 16 or cols % 64:
            raise ValueError(
                "910B INT4 NZ requires storage rows/columns aligned to 16/64"
            )
        # Expert blocks remain contiguous on NZ's outer column-tile axis.
        merged = (
            values.permute(1, 0, 2)
            .reshape(rows, experts * cols)
            .to(torch.int32)
            .contiguous()
        )
        from tokenspeed_kernel_npu.ops.gemm import prepare_weight_nz

        nz = prepare_weight_nz(merged)
        config = torch.npu.config
        previous = getattr(config, "allow_internal_format", None)
        try:
            config.allow_internal_format = True
            packed = torch_npu.npu_convert_weight_to_int4pack(nz)
        finally:
            config.allow_internal_format = previous
        return packed.view(experts, rows, cols // 8)

    def to_nz(weight):
        if torch_npu.get_npu_format(weight) == 29:
            raise ValueError(
                "W4A4 checkpoint weights are already NZ; do not prepare twice"
            )
        if not transposed:
            weight = weight.transpose(1, 2)
        values = unpack_int4(weight)
        if transposed and role == "prefill":
            return values_to_nz(values.transpose(1, 2)), None
        base = values_to_nz(values)
        # Transposing packed words leaves the nibbles on K. Repack logical
        # INT4 values so official Prefill GMM gets [E,K,N/8]; preserve the
        # original NT packing and flag used by the fused Decode operation.
        prefill = (
            values_to_nz(values.transpose(1, 2))
            if transposed and role == "null"
            else None
        )
        return base, prefill

    base13, prefill13 = to_nz(w13)
    base2, prefill2 = to_nz(w2)
    return W4A4MoePlan(
        base13,
        base2,
        scale13.contiguous(),
        scale2.contiguous(),
        first,
        second,
        transposed and role != "prefill",
        (prefill13, prefill2) if prefill13 is not None else None,
    )
