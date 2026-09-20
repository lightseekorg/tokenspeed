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

"""PyTorch reference kernels for MoE routing and expert application.

The routing references register as the ``"torch"`` solution in the PORTABLE
band: they are the numeric ground truth for the fused routing kernels and
the last-resort path when no fused kernel covers an input (CPU tensors,
unusual dtypes, or shapes outside a fused kernel's traits). The
expert-application reference sits in the REFERENCE band: it loops over
experts in FP32 and is reachable only by naming a solution (``"torch"``, or
the ``"reference"`` meta solution that resolves to it).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_ROUTER_LOGITS_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)


@register_kernel(
    "moe",
    "pack_topk_router_logits",
    name="torch_pack_topk_router_logits",
    solution="torch",
    signatures=format_signatures("topk_weights", "dense", {torch.float32}),
    traits={},
    priority=Priority.PORTABLE,
    tags={"determinism", "portability"},
)
def torch_pack_topk_router_logits(
    *,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Pack selected routes as dense FP32 log-probabilities.

    Args:
        topk_weights: FP32 route weights shaped ``[num_tokens, top_k]``.
        topk_ids: Expert indices shaped ``[num_tokens, top_k]``.
        num_experts: Width of the dense output.

    Returns:
        FP32 ``[num_tokens, num_experts]`` holding the log weight of every
        selected expert and a large negative value elsewhere.
    """
    tokens = topk_weights.shape[0]
    output = torch.full(
        (tokens, num_experts),
        -1.0e20,
        dtype=torch.float32,
        device=topk_weights.device,
    )
    output.scatter_(
        1,
        topk_ids.long(),
        topk_weights.clamp_min(torch.finfo(torch.float32).tiny).log(),
    )
    return output


@register_kernel(
    "moe",
    "softmax_topk",
    name="torch_softmax_topk",
    solution="torch",
    signatures=format_signatures("router_logits", "dense", _ROUTER_LOGITS_DTYPES),
    traits={},
    priority=Priority.PORTABLE,
    tags={"determinism", "portability"},
)
def torch_softmax_topk(
    *,
    router_logits: torch.Tensor,
    topk: int,
    topk_indices_dtype: torch.dtype,
    renormalize: bool,
    routed_scaling_factor: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordinary softmax top-k routing.

    Args:
        router_logits: Router logits shaped ``[tokens, experts]``.
        topk: Number of experts selected per token.
        topk_indices_dtype: Integer dtype of the returned expert ids.
        renormalize: Softmax over the selected logits when true; otherwise
            gather from the softmax over all experts.
        routed_scaling_factor: Scale applied to every selected weight.
        enable_pdl: Accepted for signature parity with fused kernels; unused.

    Returns:
        ``(topk_weights, topk_ids)`` shaped ``[tokens, topk]``; weights are
        FP32 and ids use ``topk_indices_dtype``.
    """
    del enable_pdl
    logits = router_logits.float()
    topk_logits, topk_ids = torch.topk(logits, topk, dim=-1, sorted=True)
    if renormalize:
        topk_weights = torch.softmax(topk_logits, dim=-1)
    else:
        topk_weights = torch.softmax(logits, dim=-1).gather(-1, topk_ids)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.to(topk_indices_dtype)


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="torch_sigmoid_bias_topk",
    solution="torch",
    signatures=format_signatures("router_logits", "dense", _ROUTER_LOGITS_DTYPES),
    traits={},
    priority=Priority.PORTABLE,
    tags={"determinism", "portability"},
)
def torch_sigmoid_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Biased sigmoid top-k routing matching Kimi's routing path.

    Args:
        router_logits: Router logits shaped ``[tokens, experts]``.
        correction_bias: Expert-selection bias shaped ``[experts]``; it
            affects selection only, not the returned weights.
        topk: Number of experts selected per token.
        routed_scaling_factor: Scale applied after optional normalization.
        normalize_topk_weights: Normalize the selected sigmoid scores to sum
            to one when true.
        weights_dtype: Output dtype of the route weights.

    Returns:
        ``(topk_weights, topk_ids)`` shaped ``[tokens, topk]``; ids are
        INT32.
    """
    scores = router_logits.sigmoid()
    topk_ids = torch.topk(
        scores + correction_bias.unsqueeze(0), topk, dim=-1, sorted=False
    ).indices
    topk_weights = scores.gather(1, topk_ids)
    if normalize_topk_weights:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(weights_dtype), topk_ids.to(torch.int32)


def _activate(
    gate_up: torch.Tensor,
    *,
    activation: str,
    dtype: torch.dtype,
    situ_beta: float,
    situ_linear_beta: float | None,
) -> torch.Tensor:
    """Apply the gated activation the way the fused kernels round it.

    ``gate_up`` is the FP32 ``[rows, 2 * intermediate]`` projection with gate
    columns first. The fused kernels round the projection to the activation
    dtype before SiTU and round the activated product once.
    """
    gate, up = gate_up.chunk(2, dim=-1)
    if activation == "situ":
        gate = gate.to(dtype).float()
        up = up.to(dtype).float()
        gate = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
        if situ_linear_beta is not None:
            up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
        return (gate * up).to(dtype)
    return (F.silu(gate) * up).to(dtype)


@register_kernel(
    "moe",
    "apply",
    name="torch_precomputed_moe_apply",
    solution="torch",
    signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"unquant"}),
        "activation": frozenset({"silu", "situ", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    num_tokens_global: int | None,
    max_num_tokens_per_gpu: int | None,
    do_finalize: bool,
    enable_pdl: bool,
) -> torch.Tensor:
    """Unquantized MoE with precomputed top-k routing, one expert at a time.

    Each expert's GEMMs run in FP32 over the tokens routed to it; the
    intermediate and the expert output round to ``x.dtype`` like the fused
    kernels, and the route-weighted sum accumulates in FP32.

    Args:
        plan: MoE plan; its ``activation`` selects SiLU/SwiGLU or SiTU, with
            ``w.activation`` as the fallback.
        x: ``[tokens, hidden]`` FP16/BF16 hidden states.
        w: Module with ``w13_weight`` ``[E, 2I, H]`` (gate rows first) and
            ``w2_weight`` ``[E, H, I]``; SiTU reads ``activation_situ_beta``
            and ``activation_situ_linear_beta`` from it.
        router_logits: Unused; routing is precomputed.
        topk_weights: ``[tokens, top_k]`` route weights.
        topk_ids: ``[tokens, top_k]`` expert ids; out-of-range ids contribute
            zero.
        num_tokens_global: Unused.
        max_num_tokens_per_gpu: Unused.
        do_finalize: Must be true.
        enable_pdl: Unused.

    Returns:
        ``[tokens, hidden]`` finalized hidden states in ``x.dtype``.
    """
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("the MoE reference always finalizes")
    if topk_weights is None or topk_ids is None:
        raise ValueError("the MoE reference requires precomputed topk weights and ids")
    activation = plan.get("activation") or getattr(w, "activation", "silu")
    if activation not in {"silu", "situ", "swiglu"}:
        raise ValueError(
            f"the MoE reference does not support activation {activation!r}"
        )
    w13 = w.w13_weight
    w2 = w.w2_weight
    num_experts = w13.shape[0]
    situ_beta = float(getattr(w, "activation_situ_beta", 1.0) or 1.0)
    situ_linear_beta = getattr(w, "activation_situ_linear_beta", None)

    output = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert_id in range(num_experts):
        token_ids, slots = torch.where(topk_ids == expert_id)
        if token_ids.numel() == 0:
            continue
        gate_up = F.linear(x[token_ids].float(), w13[expert_id].float())
        intermediate = _activate(
            gate_up,
            activation=activation,
            dtype=x.dtype,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
        expert_output = F.linear(intermediate.float(), w2[expert_id].float())
        output.index_add_(
            0,
            token_ids,
            expert_output.to(x.dtype).float()
            * topk_weights[token_ids, slots].float()[:, None],
        )
    return output.to(x.dtype)


__all__ = [
    "torch_pack_topk_router_logits",
    "torch_precomputed_moe_apply",
    "torch_sigmoid_bias_topk",
    "torch_softmax_topk",
]
