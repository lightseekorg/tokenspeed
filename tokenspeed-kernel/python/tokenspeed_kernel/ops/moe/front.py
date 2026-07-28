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

"""K3 merged MoE front: gate router + latent down-projection in one input read.

Design source: the SGLang K3 fused-front PR (Apache-2.0). The router gate GEMV
and the routed-expert down-projection both consume the same ``hidden_states``,
so their weights are stacked and swept together, then a top-k epilogue selects
experts from the gate slice while the latent slice becomes the dense routed
input.

**Decode (bs1) is intentionally NOT routed here.** At one token the front's
critical path is ``router -> down`` on the main stream, with the top-k hidden
on the side stream under the down-projection. Merging serializes the top-k
after the full merged sweep (it needs the gate logits the sweep produces),
which the down-projection can no longer hide -- measured a clear loss on B300
(merged sweep 8.2us + top-k 6.2us = 14.4us vs the split fork's 12.4us). The
split fork with the streaming rowcta router (``kimi3_router_projection``
``solution="rowcta"``) is the decode winner. ``merged_front_strategy`` encodes
this: "off" at bs1, available above it for the prefill regime where a single
merged sweep amortizes across rows (matching the SGLang table, which caps the
merged front at ~1024 tokens).
"""

from __future__ import annotations

import torch

NUM_EXPERTS = 896
TOPK = 16

# Above one token, the merged single-read sweep can pay off (fewer launches,
# one input read amortized across the routed rows). At exactly one token it
# loses -- the top-k it feeds can no longer overlap the down-projection.
MERGED_FRONT_MIN_TOKENS = 2
MERGED_FRONT_MAX_TOKENS = 1024


def merged_front_strategy(num_tokens: int) -> str:
    """Return the front strategy for ``num_tokens`` (auto, no tunable knob).

    ``"split"`` -- run the router and the latent down-projection as separate
    GEMVs so the top-k overlaps the down-projection (the bs1 decode winner).
    ``"merged"`` -- one stacked ``[gate | latent]`` sweep (a prefill-regime
    candidate; see the module docstring for why it loses at one token).
    """
    if MERGED_FRONT_MIN_TOKENS <= num_tokens <= MERGED_FRONT_MAX_TOKENS:
        return "merged"
    return "split"


def kimi3_merged_front(
    hidden_states: torch.Tensor,
    merged_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    latent: int,
    *,
    topk: int = TOPK,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
    weights_dtype: torch.dtype = torch.float32,
    logical_to_physical_map: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merged front GEMV + fused sigmoid top-k for one K3 decode token.

    Reads ``hidden_states`` once, sweeping ``merged_weight`` (the router rows
    stacked on the latent down-projection rows) to produce the fp32 router
    logits and the bf16 routed input, then selects experts from the logits.

    Args:
        hidden_states: ``[1, 7168]`` contiguous bf16 activation row.
        merged_weight: ``[896 + latent, 7168]`` contiguous bf16 weight
            (``cat([gate_weight, routed_down_weight])``).
        correction_bias: ``[896]`` fp32 selection bias.
        latent: routed-expert latent width (``merged_weight`` rows minus 896).
        topk: experts selected (16 for K3).
        routed_scaling_factor: scale applied to selected route weights.
        normalize_topk_weights: normalize selected sigmoid scores when true.
        weights_dtype: output dtype for the route weights.
        logical_to_physical_map: optional ``[896]`` static EP dispatch map.
        enable_pdl: chain the GEMV -> top-k launch with programmatic dependent
            launch (NVIDIA only; ignored elsewhere).

    Returns:
        ``(topk_weights [1, topk], topk_ids [1, topk] int32,
        routed_input [1, latent] bf16)``.
    """
    from tokenspeed_kernel.ops.gemm.triton_gemv import rowcta_merged_front
    from tokenspeed_kernel.ops.moe.triton.kimi3_sigmoid_topk import (
        kimi3_sigmoid_bias_topk,
    )

    if hidden_states.shape[0] != 1:
        raise ValueError("kimi3_merged_front is a single-token (decode) front")
    if merged_weight.shape[0] != NUM_EXPERTS + latent:
        raise ValueError("merged_weight must stack the 896 gate rows on latent rows")

    gate_logits, routed_input = rowcta_merged_front(
        hidden_states,
        merged_weight,
        gate_rows=NUM_EXPERTS,
        enable_pdl=enable_pdl,
    )
    topk_weights, topk_ids = kimi3_sigmoid_bias_topk(
        gate_logits,
        correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        logical_to_physical_map=logical_to_physical_map,
        weights_dtype=weights_dtype,
        enable_pdl=enable_pdl,
    )
    return topk_weights, topk_ids, routed_input


__all__ = [
    "MERGED_FRONT_MAX_TOKENS",
    "MERGED_FRONT_MIN_TOKENS",
    "NUM_EXPERTS",
    "TOPK",
    "kimi3_merged_front",
    "merged_front_strategy",
]
