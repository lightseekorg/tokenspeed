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

"""PyTorch MoE implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@register_kernel(
    "moe",
    "deepseek_v4_select_experts",
    name="torch_deepseek_v4_select_experts",
    solution="torch",
    signatures=frozenset(
        format_signature(router_logits=dense_tensor_format(dtype))
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ),
    priority=Priority.PORTABLE,
    tags={"portability", "reference", "routing"},
)
def torch_deepseek_v4_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    need_scores: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run portable sqrt-softplus expert selection with PyTorch."""
    scores = torch.sqrt(F.softplus(router_logits.float()))
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
        table = hash_indices_table.to(device=scores.device, dtype=torch.int64)
        ids = input_ids.reshape(-1).to(device=scores.device, dtype=torch.int64)
        topk_ids = table[ids]
    else:
        scores_for_choice = scores
        if correction_bias is not None:
            scores_for_choice = scores_for_choice + correction_bias.to(
                device=scores.device,
                dtype=scores.dtype,
            ).unsqueeze(0)
        topk_ids = torch.topk(
            scores_for_choice,
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
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32), scores


__all__ = ["torch_deepseek_v4_select_experts"]
