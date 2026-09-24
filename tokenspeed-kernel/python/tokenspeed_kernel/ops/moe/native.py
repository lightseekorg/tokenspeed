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

"""Capability helpers for native latent-space MoE execution."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@register_kernel(
    "moe",
    "topk",
    name="torch_sqrt_softplus_topk",
    solution="torch",
    signatures=frozenset(
        format_signature(router_logits=dense_tensor_format(dtype))
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ),
    traits={
        "routing_kind": frozenset({"plain", "bias", "hash"}),
        "score_function": frozenset({"sqrt_softplus"}),
    },
    priority=Priority.REFERENCE,
)
def torch_sqrt_softplus_topk(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    need_scores: bool,
    routed_scaling_factor: float,
    weights_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.sqrt(F.softplus(router_logits.float()))
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash routing requires input_ids")
        table = hash_indices_table.to(device=scores.device, dtype=torch.int64)
        ids = input_ids.reshape(-1).to(device=scores.device, dtype=torch.int64)
        topk_ids = table[ids]
    else:
        selection_scores = scores
        if correction_bias is not None:
            bias = correction_bias.to(device=scores.device, dtype=scores.dtype)
            if bias.ndim == 1:
                bias = bias.unsqueeze(0)
            selection_scores = selection_scores + bias
        topk_ids = torch.topk(
            selection_scores,
            k=top_k,
            dim=-1,
            sorted=True,
        ).indices
    topk_weights = scores.gather(1, topk_ids.long())
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(topk_weights.dtype).tiny)
    output_scores = scores if need_scores else router_logits
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(weights_dtype), topk_ids.to(torch.int32), output_scores


def native_latent_moe_available() -> bool:
    """Return whether the active backend provides native latent-space MoE."""

    return current_platform().is_amd


__all__ = ["native_latent_moe_available"]
