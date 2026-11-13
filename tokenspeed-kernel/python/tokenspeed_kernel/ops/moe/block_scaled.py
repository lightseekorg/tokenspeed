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

"""Per-route block-scaled experts with explicit activation rounding.

The NVFP4 contract uses per-token outer denominator 1536, per-16 E4M3
scales selected by four/six reconstruction MAE, and checkpoint-owned weight
outer scales. MXFP8 uses per-32 UE8M0 scales. Both return BF16 route rows
before any EP reduction; gamma enters the capped activation before its BF16
store and second quantization. The caller reduces route rows before folding
top-k slots. The Torch solution is an explicit numerical reference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.communication.mailbox import MailboxTarget
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.gemm.cute_dsl import (
    cute_dsl_ll_bf16_router,
    ll_bf16_router_supported,
)
from tokenspeed_kernel.ops.gemm.triton_skinny import skinny_mm, skinny_mm_supported
from tokenspeed_kernel.ops.moe import ExpertDispatch, dispatch_topk_ids
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _fp4_round(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    magnitude = x.abs()
    code = (
        (magnitude > 0.25).to(torch.int32)
        + (magnitude >= 0.75).to(torch.int32)
        + (magnitude > 1.25).to(torch.int32)
        + (magnitude >= 1.75).to(torch.int32)
        + (magnitude > 2.5).to(torch.int32)
        + (magnitude >= 3.5).to(torch.int32)
        + (magnitude > 5.0).to(torch.int32)
    )
    table = x.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    sign = torch.signbit(x)
    rounded = torch.where(sign, -table[code.long()], table[code.long()])
    return (code | (sign.to(torch.int32) << 3)).to(torch.uint8), rounded


def reference_block_quantize(
    x: torch.Tensor, quant_format: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize contiguous rows without changing the declared input policy.

    Args:
        x: BF16 contiguous matrix with width divisible by 16 or 32.
        quant_format: Explicit ``nvfp4`` or ``mxfp8`` activation contract.

    Returns:
        Stored values, block scales, and FP32 row maxima for NVFP4 (None for
        MXFP8). FP4 nibbles store the earlier column in the low four bits.
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Block quantization requires a contiguous BF16 matrix")
    if quant_format not in {"nvfp4", "mxfp8"}:
        raise ValueError(f"Unsupported block format: {quant_format}")
    group = 16 if quant_format == "nvfp4" else 32
    if x.shape[1] % group:
        raise ValueError(f"Input width must be divisible by {group}")
    rows, width = x.shape
    grouped = x.float().view(rows, width // group, group)
    maxima = grouped.abs().amax(dim=-1)
    if quant_format == "mxfp8":
        raw = maxima * (1.0 / 448.0)
        exponent = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
        exponent = torch.where(raw <= 2.0**-127, 0, exponent.clamp(max=254))
        exponent = torch.where(raw.isnan(), 255, exponent)
        scale = torch.exp2(exponent.float() - 127.0)
        scale = torch.where(exponent == 255, float("nan"), scale)
        inverse = torch.where(exponent == 0, 2.0**127, scale.reciprocal())
        inverse = torch.where(exponent == 255, 0.0, inverse)
        values = (grouped * inverse[..., None]).clamp(-448.0, 448.0)
        return (
            values.reshape_as(x).to(torch.float8_e4m3fn),
            exponent.to(torch.uint8),
            None,
        )
    amax = x.float().abs().amax(dim=1)
    encoder = (torch.full_like(amax, 1536.0) / amax).clamp(
        max=torch.finfo(torch.float32).max
    )
    decoder = encoder.reciprocal()
    outer = amax * (1.0 / 1536.0)
    codes, scales, errors = [], [], []
    for target in (6.0, 4.0):
        scale = (maxima * (encoder[:, None] * (1.0 / target))).clamp(max=448.0)
        scale = scale.to(torch.float8_e4m3fn).float()
        inverse = (
            (scale * decoder[:, None])
            .reciprocal()
            .clamp(max=torch.finfo(torch.float32).max)
        )
        code, rounded = _fp4_round(grouped * inverse[..., None])
        reconstructed = (rounded * scale[..., None]) * outer[:, None, None]
        error = (grouped - reconstructed).abs()
        for _ in range(4):
            error = error.reshape(*error.shape[:-1], error.shape[-1] // 2, 2)
            error = error[..., 0] + error[..., 1]
        codes.append(code)
        scales.append(scale)
        errors.append(error.squeeze(-1))
    choose4 = errors[1] < errors[0]
    code = torch.where(choose4[..., None], codes[1], codes[0]).reshape(rows, width)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    scale = torch.where(choose4, scales[1], scales[0]).to(torch.float8_e4m3fn)
    return packed, scale, amax


def _decode_values(
    values: torch.Tensor, scales: torch.Tensor, quant_format: str
) -> torch.Tensor:
    if quant_format == "nvfp4":
        codes = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2)
        table = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=values.device
        )
        data = table[(codes & 7).long()]
        data = torch.where((codes & 8) != 0, -data, data)
        return data * scales.float().repeat_interleave(16, dim=-1)
    exponent = scales.float()
    scale = torch.exp2(exponent - 127.0)
    scale = torch.where(scales == 255, float("nan"), scale)
    return values.float() * scale.repeat_interleave(32, dim=-1)


def reference_capped_swiglu(
    gate_up: torch.Tensor, gamma: torch.Tensor, gate_cap: float, up_cap: float
) -> torch.Tensor:
    """Compute interleaved capped SwiGLU in FP32, then store BF16.

    The oracle uses exact sigmoid and tanh; NVIDIA kernels use the reference
    kernels' PTX approximations of both, within a BF16 rounding of it.

    Args:
        gate_up: BF16 ``[routes, 2*intermediate]`` with interleaved gate/up.
        gamma: FP32 ``[routes]`` routing weights, applied after activation.
        gate_cap: Positive tanh gate cap, or zero to disable the cap.
        up_cap: Positive tanh up cap, or zero to disable the cap.

    Returns:
        BF16 activated rows, including gamma before their rounding boundary.
    """
    gate, up = gate_up[:, 0::2].float(), gate_up[:, 1::2].float()
    sigmoid = torch.sigmoid(gate)
    if gate_cap > 0:
        gate = gate_cap * torch.tanh(gate / gate_cap)
    if up_cap > 0:
        up = up_cap * torch.tanh(up / up_cap)
    return (((gate * sigmoid) * up) * gamma[:, None]).to(torch.bfloat16)


def _validate_routes(
    x: torch.Tensor, w2: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor
) -> None:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError(
            "Block-scaled experts require contiguous BF16 [tokens, hidden]"
        )
    if x.shape[1] != w2.shape[1] or x.device != w2.device:
        raise ValueError("Tokens must match the experts' hidden width and device")
    if (
        ids.ndim != 2
        or weights.shape != ids.shape
        or ids.shape[0] != x.shape[0]
        or ids.shape[1] == 0
    ):
        raise ValueError("Routing tensors must have shape [tokens, top_k]")
    if ids.dtype not in {torch.int32, torch.int64} or weights.dtype != torch.float32:
        raise ValueError("Routing requires integer IDs and FP32 weights")
    for tensor in (weights, ids):
        if tensor.device != x.device or not tensor.is_contiguous():
            raise ValueError("Expert operands must be contiguous and on one device")


@register_kernel(
    "moe",
    "block_scaled_routes",
    name="torch_block_scaled_moe_routes",
    solution="torch",
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    traits={
        "w13_weight_dtype": frozenset({"nvfp4", "mxfp8"}),
        "w2_weight_dtype": frozenset({"nvfp4", "mxfp8"}),
    },
    priority=Priority.REFERENCE,
)
def torch_block_scaled_moe_routes(
    x: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    a13: torch.Tensor | None,
    d13: torch.Tensor | None,
    w2: torch.Tensor,
    s2: torch.Tensor,
    a2: torch.Tensor | None,
    d2: torch.Tensor | None,
    weights: torch.Tensor,
    ids: torch.Tensor,
    *,
    formats: tuple[str, str],
    experts: int,
    expert_start: int,
    gate_cap: float,
    up_cap: float,
    zero_remote_routes: bool,
    fold_slots: bool,
    fold_into: MailboxTarget | torch.Tensor | None,
) -> torch.Tensor | None:
    del experts  # only a batched GEMM sizes its tiles from the global count
    del zero_remote_routes  # every remote row reads zero here
    first, first_scale, first_amax = reference_block_quantize(x, formats[0])
    decoded_x = _decode_values(first, first_scale, formats[0])
    routes, hidden = ids.numel(), x.shape[1]
    output = torch.zeros((routes, hidden), dtype=torch.bfloat16, device=x.device)
    for expert in range(w13.shape[0]):
        locations = (ids.flatten() == expert + expert_start).nonzero().flatten()
        if locations.numel() == 0:
            continue
        token_ids = locations // ids.shape[1]
        weight = _decode_values(w13[expert], s13[expert], formats[0])
        gate_up = decoded_x[token_ids] @ weight.T
        if formats[0] == "nvfp4":
            scale = (first_amax[token_ids] * a13[expert]) * (
                1536.0 * d13[0]
            ).reciprocal()
            gate_up = gate_up * scale[:, None]
        activated = reference_capped_swiglu(
            gate_up.to(torch.bfloat16), weights.flatten()[locations], gate_cap, up_cap
        )
        values, scales, amax = reference_block_quantize(
            activated.contiguous(), formats[1]
        )
        weight = _decode_values(w2[expert], s2[expert], formats[1])
        down = _decode_values(values, scales, formats[1]) @ weight.T
        if formats[1] == "nvfp4":
            scale = (amax * a2[expert]) * (1536.0 * d2[0]).reciprocal()
            down = down * scale[:, None]
        output[locations] = down.to(torch.bfloat16)
    rows = output.view(x.shape[0], ids.shape[1], hidden)
    if fold_slots:
        # Remote rows are zero: the slot-order FP32 sum adds nothing for them.
        folded = torch.zeros((x.shape[0], hidden), dtype=torch.float32, device=x.device)
        for slot in range(ids.shape[1]):
            folded += rows[:, slot].float()
        folded = folded.to(torch.bfloat16)
        if fold_into is None:
            return folded.unsqueeze(1)
        if isinstance(fold_into, torch.Tensor):
            return fold_into.copy_(folded).unsqueeze(1)
        # Producers publish +0.0 for -0.0, the armed value.
        folded[folded == 0] = 0.0
        mailbox, tokens = fold_into.mailbox, x.shape[0]
        for peer in range(mailbox.tp):
            mailbox.peer_slots(peer, fold_into.parity)[mailbox.rank, :tokens].copy_(
                folded
            )
        return None
    return rows


@dataclass(frozen=True)
class _BlockScaledRoutesPlan:
    kernel: Callable
    weight_preprocessor: Callable | None
    formats: tuple[str, str]


def block_scaled_moe_routes_plan(
    *,
    formats: tuple[str, str],
    hidden: int,
    intermediate: int,
    device: torch.device,
    solution: str | None,
) -> object:
    """Select a route implementation and its persistent weight layout.

    Args:
        formats: Explicit activation/weight precision for each projection;
            selection matches w13_weight_dtype and w2_weight_dtype separately.
        hidden: Token width (the first projection's K, the second's N).
        intermediate: Per-expert intermediate width (the SwiGLU output).
        device: Device of the tokens and experts.
        solution: Registered implementation, or None for automatic selection.

    Returns:
        An opaque plan for ``block_scaled_moe_routes_process_tensor`` and
        ``block_scaled_moe_routes``.
    """
    if len(formats) != 2 or hidden <= 0 or intermediate <= 0:
        raise ValueError("Invalid block-scaled expert contract")
    if device.type == "npu":
        raise NotImplementedError("Block-scaled MoE routes are not supported on NPU")
    kernel = select_kernel(
        "moe",
        "block_scaled_routes",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        traits={
            "w13_weight_dtype": formats[0],
            "w2_weight_dtype": formats[1],
            "hidden": hidden,
            "ispp": intermediate,
        },
        solution="torch" if solution is None and device.type != "cuda" else solution,
    )
    spec = KernelRegistry.get().get_by_name(kernel.name)
    return _BlockScaledRoutesPlan(kernel, spec.weight_preprocessor, tuple(formats))


def _require_routes_plan(plan: object) -> _BlockScaledRoutesPlan:
    if not isinstance(plan, _BlockScaledRoutesPlan):
        raise TypeError("plan must be returned by block_scaled_moe_routes_plan")
    return plan


_EXPERT_TENSOR_ROLES = frozenset({"w13_weight", "w13_scale", "w2_weight", "w2_scale"})


def block_scaled_moe_routes_process_tensor(
    plan: object, tensor: torch.Tensor, *, role: str
) -> None:
    """Rewrite one just-loaded expert tensor into the plan's layout in place.

    Loaders call it on every load of an expert weight or block-scale tensor,
    so a reload or a weight update lands in the layout the kernel reads.

    Args:
        plan: From ``block_scaled_moe_routes_plan``.
        tensor: The rank's ``[local_experts, ...]`` weights or block scales,
            in the checkpoint's original layout.
        role: ``w13_weight``, ``w13_scale``, ``w2_weight`` or ``w2_scale``.
    """
    typed_plan = _require_routes_plan(plan)
    if role not in _EXPERT_TENSOR_ROLES:
        raise ValueError(f"Unknown expert tensor role {role!r}")
    if typed_plan.weight_preprocessor is not None:
        typed_plan.weight_preprocessor(tensor, role=role, formats=typed_plan.formats)


def block_scaled_moe_routes(
    plan: object,
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_amax: torch.Tensor | None,
    w13_denominator: torch.Tensor | None,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_amax: torch.Tensor | None,
    w2_denominator: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    experts: int,
    expert_start: int,
    gate_cap: float,
    up_cap: float,
    zero_remote_routes: bool,
    fold_slots: bool,
    fold_into: MailboxTarget | torch.Tensor | None,
) -> torch.Tensor | None:
    """Execute local experts, leaving the BF16 route reduction to the caller.

    Under PDL a decode-sized TRT-LLM-gen call reads ``x`` before its
    predecessor kernel finishes: ``x`` must be complete once the kernel
    before that one has waited, as the router's input is.

    Args:
        plan: From ``block_scaled_moe_routes_plan``; every expert weight and
            scale went through ``block_scaled_moe_routes_process_tensor``.
        x: Contiguous BF16 token matrix ``[T,H]``.
        w13, w2: Expert weights in the plan's layout.
        w13_scale, w2_scale: Block scales in the plan's layout.
        w13_amax, w2_amax: NVFP4 per-expert maxima, otherwise None.
        w13_denominator, w2_denominator: NVFP4 stored scalar denominators.
        topk_weights, topk_ids: FP32 weights and global expert IDs ``[T,K]``.
        experts: Global expert count, over which routes spread.
        expert_start: First global expert owned by this contiguous EP shard.
        gate_cap, up_cap: Tanh caps; zero disables a cap.
        zero_remote_routes: Write zero rows for routes of other ranks' experts,
            as an all-reduce of the rows needs; a fold that reads only local
            routes passes False and leaves those rows unspecified.
        fold_slots: Return each token's local routes summed in FP32 slot
            order and rounded once, for an all-reduce of ``[T,H]``; routes of
            other ranks' experts add nothing. Requires
            ``zero_remote_routes=False``.
        fold_into: With ``fold_slots``, a mailbox target publishes the sums
            into slot ``rank`` of every rank's mailbox parity
            (``ops.communication.mailbox``, at most ``MAILBOX_ROWS`` tokens)
            instead of returning them, so the mailbox's consumer completes
            the all-reduce; a contiguous BF16 ``[T,H]`` tensor (such as a
            symmetric buffer a reduce-scatter reads) receives the sums; None
            returns them in a new tensor.

    Returns:
        BF16 ``[T,K,H]`` route rows, ``[T,1,H]`` with ``fold_slots`` (a view of
        a tensor ``fold_into``), or None with a mailbox ``fold_into``.
        Gamma is already applied inside each expert, before second-input
        quantization.
    """
    typed_plan = _require_routes_plan(plan)
    if (
        expert_start < 0
        or expert_start + w13.shape[0] > experts
        or gate_cap < 0
        or up_cap < 0
        or (fold_slots and zero_remote_routes)
        or (fold_into is not None and not fold_slots)
    ):
        raise ValueError("Invalid block-scaled expert contract")
    _validate_routes(x, w2, topk_weights, topk_ids)
    return typed_plan.kernel(
        x,
        w13,
        w13_scale,
        w13_amax,
        w13_denominator,
        w2,
        w2_scale,
        w2_amax,
        w2_denominator,
        topk_weights,
        topk_ids,
        formats=typed_plan.formats,
        experts=experts,
        expert_start=expert_start,
        gate_cap=gate_cap,
        up_cap=up_cap,
        zero_remote_routes=zero_remote_routes,
        fold_slots=fold_slots,
        fold_into=fold_into,
    )


@register_kernel(
    "moe",
    "block_scaled_topk",
    name="torch_block_scaled_topk",
    solution="torch",
    signatures=frozenset({format_signature(logits=dense_tensor_format(torch.float32))}),
    priority=Priority.REFERENCE,
)
def torch_block_scaled_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    top_k: int,
    route_scale: float,
    dispatch: ExpertDispatch | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(logits) + bias
    ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :top_k]
    selected = logits.gather(1, ids)
    log_probs = torch.nn.functional.logsigmoid(selected)
    weights = torch.exp(log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True))
    ids = ids.to(torch.int32)
    if dispatch is not None:
        ids = dispatch_topk_ids(ids, dispatch)
    return weights * route_scale, ids


def block_scaled_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    top_k: int,
    route_scale: float,
    dispatch: ExpertDispatch | None,
    solution: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select sigmoid+bias routes, then normalize original logits in log space.

    Args:
        logits: FP32 ``[tokens, experts]`` router projection.
        bias: FP32 per-expert selection bias, excluded from weighting.
        top_k: Number of routes per token; ties favor the lowest expert ID.
        route_scale: Multiplier after selected-logit normalization.
        dispatch: The layer's expert placement tables; the selected logical
            experts are mapped onto physical replicas as ``dispatch_topk_ids``
            does. None returns the logical IDs.
        solution: Registered implementation, or None for automatic selection.

    Returns:
        FP32 route weights and INT32 expert IDs (physical under a placement),
        each ``[tokens, top_k]``.
    """
    if logits.ndim != 2 or logits.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError("Router requires FP32 logits and bias")
    if bias.shape != (logits.shape[1],) or not 1 <= top_k <= logits.shape[1]:
        raise ValueError("Router dimensions or top-k are invalid")
    if dispatch is not None and dispatch.replicas.shape[0] != logits.shape[1]:
        raise ValueError("Expert placement tables do not match the router width")
    kernel = select_kernel(
        "moe",
        "block_scaled_topk",
        format_signature(logits=dense_tensor_format(logits.dtype)),
        solution="torch" if solution is None and not logits.is_cuda else solution,
    )
    return kernel(logits, bias, top_k=top_k, route_scale=route_scale, dispatch=dispatch)


# Up to this many tokens the CuTe-DSL dot-product router GEMM beats the skinny
# GEMM at a 384 x 4608 BF16 router on B300 (8 tokens: 3.3 vs 4.0 us; 16: 3.5
# vs 4.1 us; 32: 4.6 vs 4.1 us).
_LL_ROUTER_MAX_TOKENS = 16


def block_scaled_router_mm(
    x: torch.Tensor, weight: torch.Tensor, *, solution: str | None
) -> torch.Tensor:
    """Project BF16 input/weight into FP32 router logits without a BF16 store.

    Args:
        x: BF16 token input matrix.
        weight: BF16 expert-by-hidden router weight.
        solution: Torch reference, or None for GEMM dispatch.

    Returns:
        FP32 router logits, before selection bias.
    """
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("Router projection requires BF16 input and weights")
    if solution == "torch" or (solution is None and not x.is_cuda):
        return x.float() @ weight.float().T
    if solution is not None:
        raise ValueError(f"Unsupported router projection solution: {solution}")
    if (
        x.ndim != 2
        or weight.ndim != 2
        or weight.shape[1] != x.shape[1]
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.device != weight.device
        or not x.is_cuda
    ):
        raise ValueError("Router projection requires contiguous matching GPU matrices")
    if x.shape[0] <= _LL_ROUTER_MAX_TOKENS and ll_bf16_router_supported(
        x, weight, x.shape[0]
    ):
        return cute_dsl_ll_bf16_router(x, weight)
    if skinny_mm_supported(x, weight):
        return skinny_mm(x, weight, None, out_dtype=torch.float32)
    return mm(x, weight, out_dtype=torch.float32)


# These GPU kernels require Triton features unavailable in Ascend Triton.
if not current_platform().is_npu:
    import tokenspeed_kernel.ops.moe._triton.block_scaled  # noqa: E402,F401

if current_platform().is_nvidia:
    import tokenspeed_kernel.ops.moe._flashinfer.block_scaled  # noqa: E402,F401
