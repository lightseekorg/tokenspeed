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

"""Kimi-K3's fused decode MoE layer tail.

One launch runs a decoder layer from its attention output to the layer's
output: the attention o-projection and its all-reduce across the
tensor-parallel ranks, the MLP-side AttnRes mix and post-attention norm, the
router, the latent projections, the shared and routed experts, and their
reduction. It can also write the next layer's attention input (its AttnRes
mix and input norm), which that layer then reads instead of mixing.

The ranks exchange values through each other's memory inside the launch, so
every rank of the group runs the same layers' tails in the same order.

The runtime asks which expert bank layout the tail reads while it builds the
layers (:func:`kimi_k3_moe_tail_expert_layout`), prepares the per-rank buffers
once before graph capture (:func:`prepare_kimi_k3_moe_tail`), packs each
layer's weights (:func:`pack_kimi_k3_moe_tail_layer`), and then runs a layer
per step (:func:`kimi_k3_moe_tail`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "KimiK3MoeTail",
    "KimiK3MoeTailLayer",
    "KimiK3MoeTailShape",
    "KimiK3NextMix",
    "kimi_k3_moe_tail",
    "kimi_k3_moe_tail_applies_attn_norm",
    "kimi_k3_moe_tail_expert_layout",
    "pack_kimi_k3_moe_tail_layer",
    "prepare_kimi_k3_moe_tail",
]


@dataclass(frozen=True)
class KimiK3MoeTailShape:
    """The model and parallel sizes a tail serves.

    Attributes:
        rows: The token rows of every step the tail runs (one request's
            decode or speculative verify).
        hidden_size: The model's hidden size.
        tp_size: The tensor-parallel group size; the attention, the experts
            and the shared experts are all split over it.
        num_heads: The attention heads over all ranks.
        head_dim: The attention value head dimension.
        num_experts: The routed experts.
        top_k: The routed experts per token.
        latent_size: The routed experts' latent hidden size.
        intermediate_size: One routed expert's intermediate size.
        shared_intermediate_size: The shared experts' intermediate size.
        weight_dtype: The routed experts' stored weight format.
        num_expert_groups: The router's expert groups.
        topk_groups: The groups each token's routes come from.
        scoring_func: The router's score function.
        routed_scaling_factor: The routed experts' output scale.
        renormalize: Whether the top-k weights are renormalized to sum to 1.
        situ_beta: The experts' SiTU activation beta.
        situ_linear_beta: The experts' SiTU linear-branch beta.
    """

    rows: int
    hidden_size: int
    tp_size: int
    num_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    latent_size: int
    intermediate_size: int
    shared_intermediate_size: int
    weight_dtype: str
    num_expert_groups: int
    topk_groups: int
    scoring_func: str
    routed_scaling_factor: float
    renormalize: bool
    situ_beta: float
    situ_linear_beta: float | None


@dataclass(frozen=True)
class KimiK3MoeTail:
    """The selected tail kernels and this rank's buffers that every layer's
    launch shares.

    Attributes:
        shape: The sizes the tail serves.
        pack: The selected kernel that packs a layer.
        decode: The selected kernel that runs a layer.
        workspace: This rank's buffers (opaque to the caller).
    """

    shape: KimiK3MoeTailShape
    pack: Any
    decode: Any
    workspace: Any


@dataclass(frozen=True)
class KimiK3MoeTailLayer:
    """One layer packed for its tail launch.

    Attributes:
        tail: The tail the layer runs on.
        weights: The layer's weights as the kernel reads them (opaque).
        options: The layer's fixed launch options (opaque).
    """

    tail: KimiK3MoeTail
    weights: Any
    options: Any


@dataclass(frozen=True)
class KimiK3NextMix:
    """The next layer's attention input, which a tail can write as well: the
    next layer's attention-side AttnRes mix of its block partial and this
    layer's output, then its input norm (``attnres_combine``).

    Attributes:
        score_weight: The next layer's attention-side AttnRes score norm
            weight times its score projection, ``[hidden]`` BF16.
        input_norm: The next layer's input norm weight, ``[hidden]`` BF16.
        partial: The next layer's attention-side FP32 block partial
            ``(max [rows], sum [rows], accumulator [rows, hidden])``.
        out: Where the next layer's input goes, ``[rows, hidden]`` BF16.
    """

    score_weight: torch.Tensor
    input_norm: torch.Tensor
    partial: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    out: torch.Tensor


def _select(mode: str, shape: KimiK3MoeTailShape):
    signature = format_signature(hidden_states=dense_tensor_format(torch.bfloat16))
    traits = {
        "rows": shape.rows,
        "hidden_size": shape.hidden_size,
        "tp_size": shape.tp_size,
        "num_heads": shape.num_heads,
        "head_dim": shape.head_dim,
        "num_experts": shape.num_experts,
        "top_k": shape.top_k,
        "latent_size": shape.latent_size,
        "intermediate_size": shape.intermediate_size,
        "shared_intermediate_size": shape.shared_intermediate_size,
        "weight_dtype": shape.weight_dtype,
        "num_expert_groups": shape.num_expert_groups,
        "topk_groups": shape.topk_groups,
        "scoring_func": shape.scoring_func,
        "routed_scaling_factor": shape.routed_scaling_factor,
        "renormalize": shape.renormalize,
        "situ_beta": shape.situ_beta,
        "situ_linear_beta": shape.situ_linear_beta,
        # A tail launch keeps every workgroup resident at once, one a compute
        # unit.
        "compute_units": current_platform().sm_count,
    }
    return select_kernel("kimi_k3", mode, signature, traits=traits)


def kimi_k3_moe_tail_expert_layout(shape: KimiK3MoeTailShape) -> str | None:
    """The routed-expert bank layout the tail reads at ``shape``, for the MoE
    layers to store their banks in; None when no tail kernel serves ``shape``
    on this platform.

    Args:
        shape: The sizes the tail would serve.
    """
    try:
        decode = _select("moe_tail", shape)
    except NoKernelFoundError:
        return None
    (layout,) = KernelRegistry.get().get_by_name(decode.name).traits["expert_layout"]
    return layout


def prepare_kimi_k3_moe_tail(
    shape: KimiK3MoeTailShape,
    *,
    rank: int,
    cpu_group: Any,
    device: torch.device,
) -> KimiK3MoeTail:
    """Build this rank's tail buffers for a ``shape`` that
    :func:`kimi_k3_moe_tail_expert_layout` found a tail for.

    Collective over ``cpu_group``: every rank of the tensor-parallel group
    calls it at the same point, before graph capture, and gets the other
    ranks' buffer addresses.

    Args:
        shape: The sizes the tail serves.
        rank: This rank in the tensor-parallel group.
        cpu_group: The tensor-parallel group's CPU (gloo) process group.
        device: This rank's device.
    """
    prepare = _select("moe_tail_prepare", shape)
    pack = _select("moe_tail_pack", shape)
    decode = _select("moe_tail", shape)
    workspace = prepare(shape=shape, rank=rank, cpu_group=cpu_group, device=device)
    return KimiK3MoeTail(shape=shape, pack=pack, decode=decode, workspace=workspace)


def pack_kimi_k3_moe_tail_layer(
    tail: KimiK3MoeTail,
    *,
    layer_index: int,
    starts_step: bool,
    num_blocks: int,
    restarts_prefix: bool,
    eps: dict[str, float],
    tensors: dict[str, torch.Tensor],
    writes_next_mix: bool,
) -> KimiK3MoeTailLayer:
    """Check a layer's weights and fix its launch options, once before graph
    capture.

    Args:
        tail: The prepared tail.
        layer_index: The layer's index in the model; it keeps the layer's
            handoffs apart from the other layers' within a step.
        starts_step: Whether this is the first layer of a step to launch a
            tail, which advances the step counter that tags every handoff.
        num_blocks: The AttnRes snapshot blocks the MLP-side mix reads.
        restarts_prefix: Whether the layer writes a new snapshot block, so the
            tail starts its residual prefix from zero instead of the incoming
            stream.
        eps: The norms' epsilons: ``"mlp_res"`` (the MLP-side AttnRes score
            norm), ``"post_attention"`` (the post-attention norm),
            ``"latent"`` (the routed experts' latent norm) and, with
            ``writes_next_mix``, ``"next_mix"`` (the next layer's
            attention-side AttnRes score norm, which also normalizes its
            input).
        tensors: The layer's weights by role: ``o_proj``, ``mlp_res_norm``,
            ``mlp_res_proj``, ``post_attention_norm``, ``router``,
            ``router_bias``, ``latent_down``, ``latent_norm``, ``latent_up``,
            ``shared_gate_up``, ``shared_down``, and the routed experts'
            ``w13``, ``w13_scale``, ``w2`` and ``w2_scale`` as the MoE layer
            stores them in the tail's bank layout.
        writes_next_mix: Whether every step's tail also writes the next
            layer's attention input (:class:`KimiK3NextMix`).

    Raises:
        ValueError: When a weight's dtype, shape or layout is not the one the
            kernel reads.
    """
    weights, options = tail.pack(
        workspace=tail.workspace,
        shape=tail.shape,
        layer_index=layer_index,
        starts_step=starts_step,
        num_blocks=num_blocks,
        restarts_prefix=restarts_prefix,
        eps=eps,
        tensors=tensors,
        writes_next_mix=writes_next_mix,
    )
    return KimiK3MoeTailLayer(tail=tail, weights=weights, options=options)


def kimi_k3_moe_tail(
    layer: KimiK3MoeTailLayer,
    attn_output: torch.Tensor,
    stream: torch.Tensor | None,
    block_residual: torch.Tensor,
    out: torch.Tensor,
    *,
    next_mix: KimiK3NextMix | None,
    attn_norm: Any = None,
) -> torch.Tensor:
    """Run one layer's tail.

    Every rank of the group runs the same layers' tails in the same order. A
    layer's first call is collective as well: every rank compiles the launch,
    then the ranks meet before any of them launches.

    Args:
        layer: The packed layer.
        attn_output: This rank's gated attention output, the o-projection's
            input, ``[rows, num_heads / tp_size * head_dim]`` BF16.
        stream: The residual stream entering the layer, ``[rows, hidden]``
            BF16, read only; None on a layer that restarts the prefix.
        block_residual: The AttnRes snapshots, ``[blocks, rows, hidden]``
            with the hidden dimension contiguous.
        out: The layer's output stream, ``[rows, hidden]`` BF16.
        next_mix: The next layer's attention input to write, given exactly
            when the layer was packed with ``writes_next_mix``.
        attn_norm: Where :func:`kimi_k3_moe_tail_applies_attn_norm` holds, a
            KDA layer's gated output norm (``gate``, ``weight``, ``eps``), with
            ``attn_output`` the raw recurrence output it normalizes.

    Returns:
        ``out``.
    """
    layer.tail.decode(
        layer.tail.workspace,
        layer.weights,
        layer.options,
        attn_output=attn_output,
        stream=stream,
        block_residual=block_residual,
        out=out,
        next_mix=next_mix,
        **({} if attn_norm is None else {"attn_norm": attn_norm}),
    )
    return out


def kimi_k3_moe_tail_applies_attn_norm(layer: KimiK3MoeTailLayer) -> bool:
    """Whether :func:`kimi_k3_moe_tail` can apply a KDA layer's gated output
    norm to ``layer``'s attention output."""
    build = getattr(layer.options, "build", None)
    if build is None or not hasattr(build, "gated_norm"):
        return False
    from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail.contract import (
        gated_norm_rows,
    )

    return gated_norm_rows(build.tokens)


# Registers the tail kernels with the registry.
import tokenspeed_kernel.ops.model.kimi_k3.gluon  # noqa: E402,F401
