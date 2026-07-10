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

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe import gluon_bf16_moe

    @register_kernel(
        "moe",
        "apply",
        name="gluon_bf16_precomputed_moe_apply",
        solution="gluon",
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            # warp-decode stage 2 tiles the I_r reduction at BLOCK_K=256 and the
            # warp path is auto-on for small M, so I_r (intermediate-size-per-
            # partition) must be a multiple of 256; other ispp falls back.
            "ispp_alignment": frozenset({256}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        # gluon is narrowly gated to gfx950.
        priority=Priority.SPECIALIZED,
    )
    def gluon_bf16_precomputed_moe_apply(
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
        """Unquantized bf16 two-stage fused MoE (gfx950, precomputed top-k).

        Args:
            plan: MoE execution plan from ``moe_plan`` (unused here).
            x: Hidden states ``[tokens, hidden]`` bf16.
            w: Weight module exposing ``w13_weight`` ``[E, 2*I, D]`` (gate rows
                ``[0:I]``, up rows ``[I:2I]``) and ``w2_weight`` ``[E, D, I]``,
                both bf16, plus ``top_k``.
            router_logits: ``[tokens, num_experts]``; only used to derive top-k
                when ``topk_ids`` / ``topk_weights`` are not supplied.
            topk_weights: Precomputed expert weights ``[tokens, top_k]``.
            topk_ids: Precomputed expert ids ``[tokens, top_k]``.
            num_tokens_global, max_num_tokens_per_gpu, do_finalize, enable_pdl:
                Unused (no EP / deferred finalize support).

        Returns:
            MoE output ``[tokens, hidden]`` bf16.
        """
        del num_tokens_global, max_num_tokens_per_gpu, do_finalize, enable_pdl
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return gluon_bf16_moe(
            x,
            w.w13_weight,
            w.w2_weight,
            topk_ids,
            topk_weights.to(torch.float32),
        )
