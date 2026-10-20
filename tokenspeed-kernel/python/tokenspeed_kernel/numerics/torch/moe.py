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

"""PyTorch references for ``tokenspeed_kernel.ops.moe``.

Routing references for the registered top-k operators live with the
operators in ``ops/moe/native.py``; this module holds the expert compute.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from tokenspeed_kernel.numerics.torch.activation import torch_situ_and_mul
from tokenspeed_kernel.numerics.torch.quantization import dequantize_mxfp4
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

Transform = Callable[[torch.Tensor], torch.Tensor]


class DequantizedExperts(Sequence[torch.Tensor]):
    """Per-expert weights dequantized on access, so a reference over many
    experts never holds all of them in FP32 at once."""

    def __init__(
        self,
        dequantize: Callable[..., torch.Tensor],
        *quantized: torch.Tensor,
    ) -> None:
        """``dequantize(*(t[expert] for t in quantized))`` is expert ``expert``."""
        self.dequantize = dequantize
        self.quantized = quantized

    def __len__(self) -> int:
        return self.quantized[0].shape[0]

    def __getitem__(self, expert: int) -> torch.Tensor:
        return self.dequantize(*(t[expert] for t in self.quantized))


@register_kernel(
    "moe",
    "experts",
    name="torch_moe_experts",
    solution="torch",
    signatures=format_signatures(
        "hidden_states", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_moe_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor | Sequence[torch.Tensor],
    w2: torch.Tensor | Sequence[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: Transform,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    quantize_input: Transform | None,
    quantize_intermediate: Transform | None,
    round_expert_output: Transform | None,
    first_expert: int,
) -> torch.Tensor:
    """Routed experts over dequantized weights, summed per token.

    For each local expert ``e``, the tokens routed to global expert
    ``first_expert + e`` run ``fc2(activation(fc1(x)))``, and each result is
    scaled by its route weight and added to its token's row. Routes to other
    experts contribute nothing.

    Args:
        hidden_states: ``[tokens, hidden]`` inputs.
        w13: ``[local_experts, n1, hidden]`` first projections (``n1`` is
            ``2 * intermediate`` for gated activations), or any sequence of
            per-expert matrices, such as :class:`DequantizedExperts`; only
            routed experts are read.
        w2: ``[local_experts, hidden_out, intermediate]`` second projections,
            or a sequence of per-expert matrices.
        topk_ids: ``[tokens, top_k]`` global expert ids.
        topk_weights: ``[tokens, top_k]`` route weights.
        activation: Maps the FP32 ``fc1`` output to the intermediate, e.g.
            ``lambda g: torch_silu_and_mul(g, None)``.
        w13_bias: Optional ``[local_experts, n1]`` first projection bias.
        w2_bias: Optional ``[local_experts, hidden_out]`` second projection bias.
        quantize_input: Optional fake quantization (quantize, then dequantize)
            of ``hidden_states`` as given, modeling the kernel's activation
            format.
        quantize_intermediate: Optional rounding or fake quantization of the
            activation output before ``fc2``, e.g. ``torch.Tensor.bfloat16``.
        round_expert_output: Optional rounding of each expert's ``fc2``
            output before weighting, e.g. ``torch.Tensor.bfloat16``.
        first_expert: Global id of local expert 0.

    Returns:
        ``[tokens, hidden_out]`` sums in FP64 if ``hidden_states`` or ``w13``
        is FP64, else in FP32.
    """
    compute = (
        torch.float64
        if torch.float64 in (hidden_states.dtype, w13[0].dtype)
        else torch.float32
    )

    def gemm_input(values: torch.Tensor, quantize: Transform | None) -> torch.Tensor:
        # The quantizer sees the values in their own dtype.
        return (values if quantize is None else quantize(values)).to(compute)

    def bias(table: torch.Tensor | None, expert: int) -> torch.Tensor | None:
        return None if table is None else table[expert].to(compute)

    x = gemm_input(hidden_states, quantize_input)
    out = torch.zeros(
        hidden_states.shape[0], w2[0].shape[0], dtype=compute, device=x.device
    )
    for expert in range(len(w13)):
        tokens, slots = torch.where(topk_ids == first_expert + expert)
        if tokens.numel() == 0:
            continue
        fc1 = F.linear(x[tokens], w13[expert].to(compute), bias(w13_bias, expert))
        fc2 = F.linear(
            gemm_input(activation(fc1), quantize_intermediate),
            w2[expert].to(compute),
            bias(w2_bias, expert),
        )
        if round_expert_output is not None:
            fc2 = round_expert_output(fc2).to(compute)
        out.index_add_(0, tokens, fc2 * topk_weights[tokens, slots, None].to(compute))
    return out


def mxfp4_situ_moe_experts(
    hidden_states: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation_dtype: torch.dtype,
    situ_beta: float,
    situ_linear_beta: float | None,
) -> torch.Tensor:
    """:func:`torch_moe_experts` of MXFP4 experts with a SiTU activation.

    Each GEMM input is cast to ``activation_dtype``; the ``fc1`` output, the
    activation, each expert's output and the sum round to ``hidden_states``'s
    dtype, the explicit activation-quantization boundaries of Kimi K3.
    """
    dtype = hidden_states.dtype

    def cast(values: torch.Tensor) -> torch.Tensor:
        return values.to(activation_dtype)

    def round_output(values: torch.Tensor) -> torch.Tensor:
        return values.to(dtype)

    return torch_moe_experts(
        hidden_states,
        DequantizedExperts(dequantize_mxfp4, w13_packed, w13_scales),
        DequantizedExperts(dequantize_mxfp4, w2_packed, w2_scales),
        topk_ids,
        topk_weights,
        activation=lambda gate_up: torch_situ_and_mul(
            gate_up.to(dtype), situ_beta, situ_linear_beta
        ),
        w13_bias=None,
        w2_bias=None,
        quantize_input=cast,
        quantize_intermediate=cast,
        round_expert_output=round_output,
        first_expert=0,
    ).to(dtype)


@register_kernel(
    "moe",
    "softmax_topk",
    name="torch_stable_softmax_topk",
    solution="torch",
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_stable_softmax_topk(
    *,
    router_logits: torch.Tensor,
    topk: int,
    topk_indices_dtype: torch.dtype,
    renormalize: bool,
    routed_scaling_factor: float,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax top-k routing with ties going to the lowest expert id.

    Weights are the softmax over the selected logits when ``renormalize``,
    else over all logits, times ``routed_scaling_factor``. ``enable_pdl`` is a
    launch hint and has no effect here.
    """
    del enable_pdl
    logits = router_logits.float()
    values, ids = torch.sort(logits, dim=-1, descending=True, stable=True)
    values, ids = values[:, :topk], ids[:, :topk]
    if renormalize:
        weights = torch.softmax(values, dim=-1)
    else:
        weights = torch.softmax(logits, dim=-1).gather(-1, ids)
    return weights * routed_scaling_factor, ids.to(topk_indices_dtype)


@register_kernel(
    "moe",
    "inkling_topk",
    name="torch_inkling_topk",
    solution="torch",
    signatures=format_signatures("logits", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_inkling_topk(
    logits: torch.Tensor,
    gate_bias: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    top_k: int,
    n_routed: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inkling gate: biased sigmoid top-k with shared-expert-sink weights.

    Selects ``topk(sigmoid(routed) + gate_bias)``; the weights over the
    selected routed logits and the shared tail logits are
    ``exp(logsigmoid(z) - logsumexp(logsigmoid(z)))`` times ``route_scale *
    global_scale``.

    Returns:
        ``(weights [tokens, top_k + n_shared] FP32, topk_ids [tokens, top_k])``.
    """
    routed = logits[:, :n_routed].float()
    _, ids = torch.topk(routed.sigmoid() + gate_bias, top_k, dim=-1)
    active = torch.cat([routed.gather(-1, ids), logits[:, n_routed:].float()], dim=-1)
    log_probs = F.logsigmoid(active)
    weights = torch.exp(log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True))
    return weights * route_scale * global_scale, ids


@register_kernel(
    "moe",
    "latent_input",
    name="torch_latent_input",
    solution="torch",
    signatures=frozenset(
        {
            format_signature(
                hidden_states=dense_tensor_format(dtype),
                router_weight=dense_tensor_format(dtype),
                routed_weight=dense_tensor_format(dtype),
                shared_gate_up_weight=dense_tensor_format(dtype),
            )
            for dtype in (torch.bfloat16, torch.float16)
        }
    ),
    priority=Priority.REFERENCE,
)
def torch_latent_input(
    *,
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    gate_clamp: float,
    up_clamp: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three projections of a latent-MoE input.

    Returns:
        ``(router_logits, routed_input, shared_input)``: FP32 router logits,
        the routed latent in the input dtype, and the SiTU-gated shared
        expert input (gate and up rounded to the input dtype first).
    """
    router_logits = F.linear(hidden_states.float(), router_weight.float())
    routed_input = F.linear(hidden_states, routed_weight)
    shared_input = torch_situ_and_mul(
        F.linear(hidden_states, shared_gate_up_weight), gate_clamp, up_clamp
    )
    return router_logits, routed_input, shared_input


@register_kernel(
    "moe",
    "finalize",
    name="torch_moe_finalize",
    solution="torch",
    signatures=format_signatures("gemm2_out", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_moe_finalize(
    gemm2_out: torch.Tensor,
    expanded_idx: torch.Tensor,
    weights: torch.Tensor,
    shared: torch.Tensor | None,
    top_k: int,
) -> torch.Tensor:
    """Combine each token's routed rows by their weights, plus shared output.

    ``expanded_idx`` maps each (token, slot) to its ``gemm2_out`` row; ``-1``
    drops the slot. ``shared`` is either one ``[tokens, hidden]`` output added
    as is, or ``[shared_experts, tokens, hidden]`` outputs weighted by the
    trailing ``weights`` columns.

    Returns:
        ``[tokens, hidden]`` BF16.
    """
    num_tokens = weights.shape[0]
    idx = expanded_idx.view(num_tokens, top_k).long()
    valid = idx >= 0
    rows = gemm2_out.float()[idx.clamp(min=0)] * valid.unsqueeze(-1)
    out = (rows * weights.float()[:, :top_k, None]).sum(dim=1)
    if shared is not None:
        if shared.dim() == 2:
            out = out + shared.float()
        else:
            out = out + torch.einsum(
                "sth,ts->th", shared.float(), weights.float()[:, top_k:]
            )
    return out.to(torch.bfloat16)


@register_kernel(
    "moe",
    "grouped_sigmoid_topk",
    name="torch_grouped_sigmoid_topk",
    solution="torch",
    signatures=format_signatures(
        "logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_grouped_sigmoid_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    n_group: int,
    topk_group: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V3's grouped router with FP32 sigmoid scores and FP32 bias.

    Groups rank by the sum of their two best biased scores; the top
    ``topk_group`` groups' experts compete for the ``top_k`` best biased
    scores. Weights are the unbiased scores, renormalized and scaled.

    Returns:
        ``(ids [tokens, top_k] int64, best first; weights [tokens, top_k] FP32)``.
    """
    scores = torch.sigmoid(logits.float())
    biased = scores + bias.float()
    tokens, experts = logits.shape
    groups = biased.view(tokens, n_group, -1).topk(2, dim=-1).values.sum(dim=-1)
    kept = torch.zeros_like(groups, dtype=torch.bool)
    kept.scatter_(1, groups.topk(topk_group, dim=-1).indices, True)
    kept = kept.repeat_interleave(experts // n_group, dim=1)
    ids = biased.masked_fill(~kept, float("-inf")).topk(top_k, dim=-1).indices
    weights = scores.gather(1, ids)
    return ids, weights / weights.sum(dim=-1, keepdim=True) * scale
