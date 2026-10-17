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

"""FlatQuant INT4 preparation and explicitly selected expert implementations."""

import math
from dataclasses import dataclass

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
    scale13: torch.Tensor
    scale2: torch.Tensor
    first: FlatQuantStage
    second: FlatQuantStage
    transposed: bool


@dataclass(frozen=True)
class W4A4GroupedPlan:
    weights: W4A4MoePlan
    scale13: torch.Tensor
    scale2: torch.Tensor


def prepare_w4a4_grouped_plan(weights: W4A4MoePlan) -> W4A4GroupedPlan:
    """Encode official GMM scales once; share the fused path's NZ weights."""

    def encode(scale):
        bits = scale.cpu().contiguous().view(torch.int32).to(torch.int64)
        return (bits & 0xFFFFFFFF).view(torch.uint64).unsqueeze(1).to(scale.device)

    return W4A4GroupedPlan(weights, encode(weights.scale13), encode(weights.scale2))


def w4a4_grouped_experts(x, counts, *, rows, plan: W4A4GroupedPlan):
    """Execute sorted, unpadded rows with Cube FlatQuant and two separate GMMs.

    ``counts`` is INT64 [local_experts], containing row counts, not offsets.
    ``rows`` gathers input tokens into contiguous expert order. FlatQuant13
    runs once per input token before that gather. No EP communication
    happens here. NZ-transposed weights are passed as transpose views, not
    repacked in forward. GMM's BF16 output has the official FP16 intermediate
    rounding, unlike the fused operator's INT32/vector dequantization.
    """
    import flash_ops  # noqa: F401
    import torch_npu

    if rows.numel() == 0:
        return x.new_empty((0, x.shape[1]))
    state = plan.weights

    def quantize(value, stage):
        packed, scales = torch.ops.custom.flat_quant(
            value.reshape(value.shape[0], stage.left.shape[0], stage.right.shape[0]),
            stage.left,
            stage.right,
            clip_ratio=stage.clip,
            dst_dtype=16,
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

    packed, scales = quantize(x, state.first)
    packed, scales = packed[rows], scales[rows]
    hidden = matmul(packed, scales, state.w13, plan.scale13)
    hidden = torch_npu.npu_swiglu(hidden, dim=-1)
    packed, scales = quantize(hidden, state.second)
    return matmul(packed, scales, state.w2, plan.scale2)


def prepare_flatquant_stage(left, right, clip, *, width: int) -> FlatQuantStage:
    """Validate and prepare constants once, before graph capture."""
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[0] != left.shape[1]
        or right.shape[0] != right.shape[1]
        or left.shape[0] * right.shape[0] != width
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
    w13, w2, scale13, scale2, first, second, *, transposed: bool
) -> W4A4MoePlan:
    """Convert checkpoint INT32 tensors to NZ without changing their INT4 layout.

    Input checkpoint shapes are [E,N,K/8] (transposed) or [E,N/8,K].
    Returned non-transposed weights are [E,K,N/8]; transposed weights remain
    [E,N,K/8]. The consumer must honor ``transposed`` for BOTH GEMMs.
    Scales remain FP32 [E,N] for fused DSQ, not official-GMM packed UINT64.
    """
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

    def to_nz(weight):
        if torch_npu.get_npu_format(weight) == 29:
            raise ValueError(
                "W4A4 checkpoint weights are already NZ; do not prepare twice"
            )
        if not transposed:
            weight = weight.transpose(1, 2)
        # 910B uses the dedicated INT4 converter: customize_dtype=3 is not
        # supported there. Merge experts along NZ's outer (column-tile) axis
        # so each expert occupies one contiguous physical NZ block.
        values = unpack_int4(weight)
        experts, rows, cols = values.shape
        if rows % 16 or cols % 64:
            raise ValueError(
                "910B INT4 NZ requires storage rows/columns aligned to 16/64"
            )
        merged = values.permute(1, 0, 2).reshape(rows, experts * cols).to(torch.int32)
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

    return W4A4MoePlan(
        to_nz(w13),
        to_nz(w2),
        scale13.contiguous(),
        scale2.contiguous(),
        first,
        second,
        transposed,
    )


def require_fused_flatquant(op) -> None:
    """Refuse the old smooth-only ABI rather than silently miscompute W4A4."""
    required = {
        "quant_mode": "str",
        "flatquant13_left": "Optional[Tensor]",
        "flatquant13_right": "Optional[Tensor]",
        "flatquant2_left": "Optional[Tensor]",
        "flatquant2_right": "Optional[Tensor]",
        "clip13": "float",
        "clip2": "float",
        "weight_transposed": "bool",
    }
    arguments = {arg.name: str(arg.type) for arg in op._schema.arguments}
    if any(arguments.get(name) != dtype for name, dtype in required.items()):
        raise RuntimeError(
            "fused_mm13_mm2 lacks the FlatQuant W4A4/NZ-transpose ABI; "
            "rebuild the operator with transform/clip/layout inputs. "
            "The smooth-only INT4 ABI is not compatible; split fallback is disabled."
        )


def ascend_int4_process_moe_weights(*, plan: dict, w: torch.nn.Module) -> None:
    """Prepare constants for the explicitly selected fused FlatQuant solution."""
    import flash_ops  # noqa: F401

    op = torch.ops.custom.fused_mm13_mm2.default
    require_fused_flatquant(op)
    if w._int4_plan is not None:
        return
    if len(w._int4_loaded) != 6 * w.num_local_experts:
        raise ValueError("Missing W4A4 expert weights/scales")
    for module in (w.w1_trans, w.w2_trans, w.act_quantizer_w1, w.act_quantizer_w2):
        module.validate_loaded()
    first = prepare_flatquant_stage(
        w.w1_trans.matrix_left,
        w.w1_trans.matrix_right,
        w.act_quantizer_w1.clip_factor_a,
        width=w.hidden_size,
    )
    second = prepare_flatquant_stage(
        w.w2_trans.matrix_left,
        w.w2_trans.matrix_right,
        w.act_quantizer_w2.clip_factor_a,
        width=w.intermediate_size,
    )
    w._int4_plan = prepare_w4a4_moe_weights(
        w.w13_weight,
        w.w2_weight,
        w.w13_weight_scale,
        w.w2_weight_scale,
        first,
        second,
        transposed=w.w4a4_gmm_is_transposed,
    )
    w.w13_weight.data = w._int4_plan.w13
    w.w2_weight.data = w._int4_plan.w2
    plan["routed_chain_op"] = op
    plan.setdefault("expert_start", w.ep_rank * w.num_local_experts)
    plan["routed_chain_boundary"] = torch.zeros(
        25, dtype=torch.int32, device=w.w13_weight.device
    )


def ascend_int4_precomputed_moe_apply(
    *,
    plan,
    x,
    w,
    router_logits,
    topk_weights=None,
    topk_ids=None,
    num_tokens_global=None,
    max_num_tokens_per_gpu=None,
    do_finalize=True,
    enable_pdl=False,
):
    """Invoke only the fused operator; dispatch and EP combine remain external."""
    if (
        not do_finalize
        or enable_pdl
        or plan.get("activation") not in {"silu", "swiglu"}
    ):
        raise ValueError("FlatQuant fused MoE requires finalized SwiGLU without PDL")
    if topk_ids is None or topk_weights is None or topk_ids.shape != topk_weights.shape:
        raise ValueError("FlatQuant fused MoE requires matching precomputed TopK")
    if w._int4_plan is None:
        raise RuntimeError("FlatQuant fused MoE weights were not prepared")
    state = w._int4_plan
    boundary = plan["routed_chain_boundary"]
    return plan["routed_chain_op"](
        x.contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        state.w13,
        state.scale13,
        state.w2,
        state.scale2,
        topk_weights.contiguous(),
        boundary,
        boundary,
        boundary,
        expert_start=plan["expert_start"],
        quant_bits=4,
        quant_mode="flatquant",
        flatquant13_left=state.first.left,
        flatquant13_right=state.first.right,
        flatquant2_left=state.second.left,
        flatquant2_right=state.second.right,
        clip13=state.first.clip,
        clip2=state.second.clip,
        weight_transposed=state.transposed,
    )
