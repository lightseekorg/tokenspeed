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

"""Marlin W8A16 MXFP8 experts with raw E4M3 weights and E8M0 group-32 scales.

The weight-only grouped GEMMs dequantize into BF16 operands and reduce split-K
partials in FP32. The BF16 route-weight product and fixed-slot final reduction
are shared with other implementations of the same model arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.moe._triton.bf16_epilogue import (
    clamped_swiglu_bf16,
    finalize_bf16_routes,
)
from tokenspeed_kernel.ops.moe.marlin import _swiglu_limit
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.marlin import gptq_marlin_repack
from tokenspeed_kernel.thirdparty.cuda.marlin_moe import (
    FLOAT8_E4M3FN_ID,
    marlin_make_workspace,
    marlin_permute_scales,
    moe_align_block_size,
    moe_wna16_marlin_gemm,
    mxfp4_marlin_process_scales,
)

MXFP8_BLOCK = 32
_ISPP_ALIGNMENT = 64
_HIDDEN_ALIGNMENT = 128


@dataclass(frozen=True)
class _MarlinMxfp8State:
    expert_start: int
    hidden_size: int
    ispp: int


def _block_size_m(routes_per_expert: float) -> int:
    """Pick the token block granularity Marlin schedules over.

    Grow the block until each expert holds under ~0.9 blocks of routed tokens,
    keeping decode batches on tight blocks and prefill batches on wide ones.
    """
    block = 8
    for candidate in (8, 16, 32, 48, 64):
        block = candidate
        if routes_per_expert / candidate < 0.9:
            break
    return block


def _validate_mxfp8_scales(name: str, scale: torch.Tensor) -> None:
    """Reject scale encodings the native BF16 decoder cannot represent.

    E8M0 0x00 is 2**-127, not zero. Marlin's BF16 scale decoder flushes it;
    0xFF denotes NaN. Check once while loading, before repacking any parameter,
    rather than silently changing the scale or synchronizing captured applies.
    """
    if bool(((scale == 0) | (scale == 255)).any()):
        raise ValueError(
            f"Marlin MXFP8 {name} contains unsupported E8M0 scale encoding "
            "0x00 (2**-127) or 0xFF (NaN); scales must be in [1, 254]."
        )


def marlin_mxfp8_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Repack loader-format MXFP8 experts into the Marlin layout, once.

    Input (from the block-FP8 weight loader with ``(1, 32)`` blocks):
    ``w13_weight`` ``[E, 2*ispp, hidden]`` / ``w2_weight`` ``[E, hidden, ispp]``
    float8_e4m3fn and ``w13_weight_scale_inv`` / ``w2_weight_scale_inv`` raw
    E8M0 bytes as uint8 ``[E, N, K//32]``. Output: Marlin-repacked int32
    weights and permuted float8_e8m0fnu scales written back under the same
    names, and the expert geometry in ``w._moe_backend_state``, which must be
    None until then.
    """
    if isinstance(w._moe_backend_state, _MarlinMxfp8State):
        return

    activation = plan["activation"] or w.activation
    if activation not in {"silu", "swiglu"}:
        raise ValueError(f"Marlin MXFP8 MoE does not support activation {activation!r}")
    limit = _swiglu_limit(w)
    if limit is not None and (not math.isfinite(limit) or limit <= 0):
        raise ValueError("SwiGLU clamp limit must be finite and positive")

    w13 = w.w13_weight.data
    w2 = w.w2_weight.data
    w13_scale = w.w13_weight_scale_inv.data
    w2_scale = w.w2_weight_scale_inv.data
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"Marlin MXFP8 expects float8_e4m3fn experts, got {w13.dtype}/{w2.dtype}"
        )
    if w13_scale.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
        raise TypeError(
            "Marlin MXFP8 expects raw E8M0 scales as uint8, got "
            f"{w13_scale.dtype}/{w2_scale.dtype}"
        )
    tensors = (w13, w2, w13_scale, w2_scale)
    if any(t.ndim != 3 for t in tensors):
        raise ValueError("MXFP8 weights and scales must be three-dimensional")
    if any(t.device != w13.device for t in tensors):
        raise ValueError("MXFP8 weights and scales must share a device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("MXFP8 weights and scales must be contiguous")
    num_experts, two_ispp, hidden = w13.shape
    ispp = w2.shape[2]
    if min(num_experts, hidden, ispp) <= 0:
        raise ValueError("MXFP8 expert dimensions must be positive")
    if w.ep_size <= 0 or not 0 <= w.ep_rank < w.ep_size:
        raise ValueError("invalid expert-parallel rank or size")
    if w2.shape[0] != num_experts or two_ispp != 2 * ispp or w2.shape[1] != hidden:
        raise ValueError(
            f"w13 {tuple(w13.shape)} and w2 {tuple(w2.shape)} disagree on the "
            "expert geometry"
        )
    if hidden % _HIDDEN_ALIGNMENT != 0 or ispp % _ISPP_ALIGNMENT != 0:
        raise ValueError(
            f"Marlin MXFP8 needs hidden%{_HIDDEN_ALIGNMENT}==0 and "
            f"ispp%{_ISPP_ALIGNMENT}==0, got hidden={hidden}, ispp={ispp}"
        )
    expected_scales = (
        (num_experts, two_ispp, hidden // MXFP8_BLOCK),
        (num_experts, hidden, ispp // MXFP8_BLOCK),
    )
    if (tuple(w13_scale.shape), tuple(w2_scale.shape)) != expected_scales:
        raise ValueError(
            f"MXFP8 scales {tuple(w13_scale.shape)}/{tuple(w2_scale.shape)} do "
            f"not match the 1x{MXFP8_BLOCK} blocks of the weights {expected_scales}"
        )

    _validate_mxfp8_scales("w13_weight_scale_inv", w13_scale)
    _validate_mxfp8_scales("w2_weight_scale_inv", w2_scale)

    if w13.device.type != "cuda":
        raise ValueError("Marlin MXFP8 preparation requires CUDA weights")
    for name, weight in (("w13", w13), ("w2", w2)):
        # Validate one expert at a time so loading does not expand all local
        # experts to FP32 merely to check E4M3's two NaN byte encodings.
        for expert in weight:
            if bool(((expert.view(torch.uint8) & 0x7F) == 0x7F).any()):
                raise ValueError(f"Marlin MXFP8 {name} contains non-finite weights")
    device = w13.device
    perm = torch.empty(0, dtype=torch.int, device=device)

    def _repack(weight: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
        # gptq_marlin_repack wants int32 [size_k/4, size_n]: four E4M3 bytes
        # per int32 along K, i.e. the transposed int32 view of [size_n, size_k].
        out = torch.empty(
            (num_experts, size_k // 16, size_n * 4), dtype=torch.int32, device=device
        )
        for e in range(num_experts):
            qw = weight[e].view(torch.int32).T.contiguous()
            out[e] = gptq_marlin_repack(
                qw, perm, size_k=size_k, size_n=size_n, num_bits=8
            )
        return out

    def _permute(scale: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
        out = torch.empty(
            (num_experts, size_k // MXFP8_BLOCK, size_n),
            dtype=torch.uint8,
            device=device,
        )
        for e in range(num_experts):
            permuted = marlin_permute_scales(
                scale[e].T.contiguous(),
                size_k=size_k,
                size_n=size_n,
                group_size=MXFP8_BLOCK,
            )
            out[e] = mxfp4_marlin_process_scales(permuted).view(torch.uint8)
        return out.view(torch.float8_e8m0fnu)

    w13_marlin = _repack(w13, size_n=two_ispp, size_k=hidden)
    w13_scale_marlin = _permute(w13_scale, size_n=two_ispp, size_k=hidden)
    w2_marlin = _repack(w2, size_n=hidden, size_k=ispp)
    w2_scale_marlin = _permute(w2_scale, size_n=hidden, size_k=ispp)
    # Publish only after every validation and preparation step succeeds.
    w.w13_weight = torch.nn.Parameter(w13_marlin, requires_grad=False)
    w.w13_weight_scale_inv = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)
    w.w2_weight = torch.nn.Parameter(w2_marlin, requires_grad=False)
    w.w2_weight_scale_inv = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)

    w._moe_backend_state = _MarlinMxfp8State(
        expert_start=w.ep_rank * num_experts,
        hidden_size=hidden,
        ispp=ispp,
    )


@register_kernel(
    "moe",
    "apply",
    name="marlin_mxfp8_precomputed_moe_apply",
    solution="marlin",
    weight_preprocessor=marlin_mxfp8_moe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "fp8_scale_block_shape": frozenset({(1, MXFP8_BLOCK)}),
        "activation": frozenset({"silu", "swiglu"}),
        # silu_and_mul has no alpha/beta (see _swiglu_limit): standard form only.
        "swiglu_form": frozenset({"standard"}),
        "expert_id_repeats": frozenset({False, True}),
        "activation_clamped": frozenset({False, True}),
        "combine_order": frozenset({"rank"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "deterministic_finalize": frozenset({False, True}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({_ISPP_ALIGNMENT}),
        "hidden_alignment": frozenset({_HIDDEN_ALIGNMENT}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def marlin_mxfp8_precomputed_moe_apply(
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
) -> torch.Tensor:
    """Apply a Marlin W8A16 MXFP8 MoE with precomputed routing.

    Args:
        plan: MoE plan; ``activation`` (``silu``/``swiglu``) computes
            ``silu(gate) * up`` clamped to ``w.swiglu_arg.limit`` when the
            checkpoint sets one (``gate`` from above, ``up`` on both sides).
            ``deterministic_finalize`` selects stable route alignment.
            Both GEMMs use FP32 split-K reduction; finalize folds slots in order.
        x: bf16 hidden states ``[tokens, hidden]``.
        w: MoE layer processed by :func:`marlin_mxfp8_moe_weights`: the
            Marlin-repacked ``w13_weight``/``w2_weight`` (int32), permuted
            ``w13_weight_scale_inv``/``w2_weight_scale_inv`` (float8_e8m0fnu)
            and ``_moe_backend_state``, plus the layer's ``ep_size``/``ep_rank``
            and ``tp_size``/``tp_rank``.
        router_logits: Unused; routing is precomputed.
        topk_weights: Route weights ``[tokens, top_k]``.
        topk_ids: Global expert ids ``[tokens, top_k]``; a token's valid ids
            may repeat. Ids outside this rank's expert range, ``-1``
            included, contribute zero.
        num_tokens_global: Unused; distributed EP dispatch is not owned here.
        max_num_tokens_per_gpu: Unused capacity hint.
        do_finalize: Must be true (no deferred finalize).
        enable_pdl: Unused launch hint; this native kernel does not use PDL.

    Returns:
        This rank's expert contributions ``[tokens, hidden]`` in the dtype of
        ``x``, summed over the top-k routes in a fixed order.
    """
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("Marlin MXFP8 MoE cannot defer finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Marlin MXFP8 MoE requires precomputed top-k")
    if plan["combine_order"] != "rank":
        raise ValueError("Marlin MXFP8 requires rank-order communication")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Marlin MXFP8 MoE requires bf16 activations, got {x.dtype}")

    activation = plan["activation"] or w.activation
    if activation not in {"silu", "swiglu"}:
        raise ValueError(f"Marlin MXFP8 MoE does not support activation {activation!r}")
    state = w._moe_backend_state
    if not isinstance(state, _MarlinMxfp8State):
        raise TypeError("invalid Marlin MXFP8 MoE state")
    hidden = state.hidden_size
    ispp = state.ispp
    num_local_experts = int(w.w13_weight.shape[0])

    if (
        x.ndim != 2
        or x.shape[1] != hidden
        or topk_ids.ndim != 2
        or topk_ids.shape[0] != x.shape[0]
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError("MXFP8 activations and routing shapes disagree")
    if any(t.device != x.device for t in (topk_ids, topk_weights, w.w13_weight)):
        raise ValueError("MXFP8 activations, routing and weights must share a device")
    # Native GEMMs read dense rows. A dtype conversion alone
    # preserves strided FP32 route-weight views (including a flattened ::2).
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("Marlin MXFP8 expert IDs must be int32 or int64")
    x = x.contiguous()
    if topk_ids.dtype == torch.int64:
        # Preserve invalid IDs before narrowing to the native int32 ABI.
        topk_ids = torch.where(
            (topk_ids >= 0) & (topk_ids < num_local_experts * w.ep_size),
            topk_ids,
            -1,
        )
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    num_tokens, top_k = topk_ids.shape
    if top_k <= 0:
        raise ValueError("Marlin MXFP8 requires a positive top-k")
    if num_tokens == 0:
        return torch.empty_like(x)
    routes_per_expert = num_tokens * top_k / (num_local_experts * w.ep_size)
    expert_start = state.expert_start
    block_m = _block_size_m(routes_per_expert)
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_m,
        num_local_experts,
        expert_start=expert_start,
        deterministic=plan["deterministic_finalize"],
    )
    workspace = marlin_make_workspace(x.device)

    # GEMMs only touch local routes (1/ep_size of the rows), so the outputs are
    # left uninitialized and the activation and the finalize skip the rest
    # instead of streaming all T * top_k rows. The other routes, -1 included,
    # fill blocks of expert id -1 with or without EP; only the EP
    # specialization skips them instead of reading expert -1's weights.
    num_routes = num_tokens * top_k
    gemm1_n = 2 * ispp
    intermediate1 = moe_wna16_marlin_gemm(
        x,
        torch.empty(num_routes, gemm1_n, dtype=x.dtype, device=x.device),
        w.w13_weight,
        w.w13_weight_scale_inv,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        b_q_type_id=FLOAT8_E4M3FN_ID,
        moe_block_size=block_m,
        top_k=top_k,
        mul_topk_weights=False,
        is_ep=True,
        size_m=num_tokens,
        size_n=gemm1_n,
        size_k=hidden,
        use_fp32_reduce=True,
    )
    intermediate2 = clamped_swiglu_bf16(
        intermediate1,
        topk_ids,
        expert_start=expert_start,
        num_local_experts=num_local_experts,
        limit=_swiglu_limit(w),
    )

    # GEMM2 folds the route weights in, so finalize is a plain sum over top_k.
    intermediate3 = moe_wna16_marlin_gemm(
        intermediate2,
        torch.empty(num_routes, hidden, dtype=x.dtype, device=x.device),
        w.w2_weight,
        w.w2_weight_scale_inv,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        b_q_type_id=FLOAT8_E4M3FN_ID,
        moe_block_size=block_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=True,
        size_m=num_routes,
        size_n=hidden,
        size_k=ispp,
        use_fp32_reduce=True,
    )
    return finalize_bf16_routes(
        intermediate3,
        topk_weights,
        topk_ids,
        expert_start=expert_start,
        num_local_experts=num_local_experts,
        weighted_routes=True,
    )
