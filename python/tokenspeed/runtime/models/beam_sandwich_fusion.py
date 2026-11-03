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

"""Beam's all-reduce boundaries through the sandwich norm helper.

Each of Beam's 2 x L all-reduce boundaries is the all-reduce of the sublayer
partial, the sandwich norm (``attn_output_layernorm``, ``mlp_output_layernorm``)
and the norm that consumes the depth-scale pair (``post_attention_layernorm``,
the next layer's ``input_layernorm`` or the final norm). Under
``--enable-allreduce-fusion`` (set, or auto-enabled by
``ServerArgs.resolve_communication``) the consuming norm runs the boundary as
one call of
:func:`tokenspeed.runtime.layers.layernorm.sandwich_rmsnorm_with_allreduce`: the
comm manager hands the partial on unreduced and the sandwich norm passes it
through. The helper fuses the call into one MNNVL launch where the group's
workspace serves it; elsewhere it runs the all-reduce and the two Triton norms
the option-off forward runs, with the same arguments, so such a call equals the
option-off boundary bit for bit. A fused call differs from it only in
summation order: the two sums of squares, and the all-reduce sum where the
option-off all-reduce does not take the MNNVL path (IPC, NCCL).

The choice is made once, at model construction, from launch options every rank
shares, so every rank and every module of a boundary agree. The option is
auto-enabled for Beam at any node count; on a launch across nodes the
boundaries take the helper only where the attention TP group has an armed
workspace, which across nodes is the MNNVL one (CUDA IPC does not cross nodes).
Without the option, or where a static condition does not hold (the reason is
logged), nothing is replaced. The shared decoder forward is not changed; the
classes are swapped in place (same parameters, same state_dict keys).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.layernorm import sandwich_rmsnorm_with_allreduce
from tokenspeed.runtime.models.beam import (
    BeamCommManager,
    BeamResidualFoldRMSNorm,
    BeamRMSNorm,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)


def beam_sandwich_fusion_refusal(model: Any) -> str:
    """Why the boundaries keep the option-off wiring; empty when they take the
    helper.

    ``--enable-allreduce-fusion``; attention TP of two or more ranks without DP
    attention or context parallelism and with expert parallelism 1; one
    all-reduce group for attention, dense and MoE; an all-reduce at every
    boundary; the rank-order MoE combine; on a launch across nodes, a workspace
    armed for the attention TP group.
    """
    if not global_server_args_dict.get("enable_allreduce_fusion", False):
        return "--enable-allreduce-fusion is off"
    if not current_platform().is_nvidia:
        return "sandwich boundary fusion requires NVIDIA kernels"
    mapping = model.mapping
    attn = mapping.attn
    if attn.tp_size <= 1:
        return "attention TP is 1: there is no all-reduce to fuse"
    if attn.has_dp:
        return "DP attention is not supported"
    if attn.has_kvp or attn.has_qcp:
        return "context parallelism is not supported"
    if mapping.moe.ep_size != 1:
        return "expert parallelism is not supported"
    if not attn.tp_group == mapping.dense.tp_group == mapping.moe.tp_ep_group:
        return "attention, dense and MoE do not share one all-reduce group"
    if global_server_args_dict.get("moe_combine_order", "rank") != "rank":
        return "the MoE combine is not the rank-order all-reduce"
    for layer in model.layers:
        comm = layer.comm_manager
        if type(comm) is not BeamCommManager:
            return f"layer {layer.layer_id} has comm manager {type(comm).__name__}"
        if not (comm.use_all_reduce(True) and comm.use_all_reduce(False)):
            return f"layer {layer.layer_id} does not all-reduce every boundary"
    if mapping.nnodes > 1 and not attn_tp_workspace_armed(mapping):
        return (
            "the launch spans nodes and the attention TP group has no armed "
            "all-reduce workspace (across nodes, no MNNVL workspace)"
        )
    return ""


def attn_tp_workspace_armed(mapping: Any) -> bool:
    """Whether the attention TP group's all-reduce fusion workspace is armed.

    ``DistributedInitializer`` arms it, collectively, before the model is built,
    so every rank reads the same answer here. A group across nodes can only
    hold the MNNVL workspace. Without an armed workspace each fused call would
    try to arm one again, a collective, at every boundary; the boundaries keep
    the option-off wiring instead.
    """
    if not current_platform().is_nvidia:
        return False
    from tokenspeed_kernel.ops.communication.trtllm import armed_workspace_hidden_dim

    group = pg_manager.get_device_process_group(mapping.attn.tp_group)
    return armed_workspace_hidden_dim(group) > 0


def _boundaries(model: Any) -> list[tuple[Any, Any, str]]:
    """``(producer, consumer, kind)`` of every boundary, in forward order: layer
    i's attention boundary, then its MLP boundary, which layer i + 1's input
    norm (or the final norm) consumes."""
    layers = list(model.layers)
    rows = []
    for index, layer in enumerate(layers):
        rows.append(
            (layer.attn_output_layernorm, layer.post_attention_layernorm, "attention")
        )
        consumer = (
            layers[index + 1].input_layernorm if index + 1 < len(layers) else model.norm
        )
        rows.append((layer.mlp_output_layernorm, consumer, "mlp"))
    return rows


class BeamSandwichCommManager(BeamCommManager):
    """Hands the sublayer partial on unreduced: the consuming norm reduces it."""

    def post_attn_comm(self, hidden_states, residual, ctx):
        return hidden_states, residual

    def post_dense_comm(self, hidden_states, residual, ctx):
        return hidden_states, residual

    def post_moe_comm(self, hidden_states, residual, ctx):
        return hidden_states, residual


def _deferred(module: Any, what: str) -> ValueError:
    return ValueError(
        f"{type(module).__name__} got {what}: its boundary runs in the consuming norm"
    )


class BeamDeferredSandwichNorm(BeamRMSNorm):
    """``attn_output_layernorm`` / ``mlp_output_layernorm``: passes the partial
    through; the consuming norm applies this norm's weight."""

    def forward(self, x, residual=None, out=None, *, x_scale=None, residual_scale=None):
        if residual is not None or out is not None:
            raise _deferred(self, "a residual or an out tensor")
        if x_scale is not None or residual_scale is not None:
            raise _deferred(self, "a depth-scale pair")
        return x


def _boundary(norm, x, residual, out, x_scale, residual_scale):
    if residual is None or out is not None:
        raise _deferred(norm, "no residual or an out tensor")
    mapping = norm._sandwich_mapping
    # As BeamRMSNorm: a norm whose consumer runs an FP8 GEMM on these rows
    # (``_token_fp8_consumer``) has the boundary write them per-token FP8 too
    # and offers them, so the consumer skips its own quantization.
    consumer = norm._token_fp8_consumer
    token_fp8 = None
    if consumer is not None and consumer.takes_token_fp8(x):
        token_fp8 = (
            torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device),
        )
    fp8_kwargs = {} if token_fp8 is None else {"token_fp8_out": token_fp8}
    result = sandwich_rmsnorm_with_allreduce(
        x,
        residual,
        norm._sandwich_producer,
        norm,
        rank=mapping.attn.tp_rank,
        group=mapping.attn.tp_group,
        x_scale=x_scale,
        residual_scale=residual_scale,
        **fp8_kwargs,
    )
    if token_fp8 is not None:
        consumer.offer_token_fp8(result[0], *token_fp8)
    return result


class BeamSandwichNorm(BeamRMSNorm):
    """``post_attention_layernorm``: the attention boundary, with the pair the
    layer passes."""

    def forward(self, x, residual=None, out=None, *, x_scale=None, residual_scale=None):
        if x_scale is None or residual_scale is None:
            raise _deferred(self, "no depth-scale pair")
        return _boundary(self, x, residual, out, x_scale, residual_scale)


class BeamSandwichFoldNorm(BeamResidualFoldRMSNorm):
    """An input norm of layer 1..L-1 or the final norm: the MLP boundary of the
    layer before, with this norm's ``residual_fold`` pair."""

    def forward(self, x, residual=None, out=None, *, x_scale=None, residual_scale=None):
        if x_scale is not None or residual_scale is not None:
            raise _deferred(self, "a second depth-scale pair")
        return _boundary(self, x, residual, out, *self.residual_fold)


def _swap(module: Any, cls: type, expected: type) -> None:
    if type(module) is not expected:
        raise ValueError(
            f"the sandwich boundary expects a {expected.__name__}, got "
            f"{type(module).__name__}"
        )
    module.__class__ = cls


def install_beam_sandwich_fusion(model: Any) -> None:
    """Route every boundary of ``model`` through the sandwich norm helper when
    :func:`beam_sandwich_fusion_refusal` admits it; log one line either way.

    Called once at the end of ``BeamModel.__init__`` on every rank.
    """
    refusal = beam_sandwich_fusion_refusal(model)
    if refusal:
        logger.info(f"Beam sandwich all-reduce boundaries: unfused ({refusal})")
        return
    for layer in model.layers:
        _swap(layer.comm_manager, BeamSandwichCommManager, BeamCommManager)
    boundaries = _boundaries(model)
    for producer, consumer, kind in boundaries:
        _swap(producer, BeamDeferredSandwichNorm, BeamRMSNorm)
        if kind == "attention":
            _swap(consumer, BeamSandwichNorm, BeamRMSNorm)
        else:
            _swap(consumer, BeamSandwichFoldNorm, BeamResidualFoldRMSNorm)
        # Plain attributes, not submodules: no new parameter or state_dict key.
        object.__setattr__(consumer, "_sandwich_producer", producer)
        object.__setattr__(consumer, "_sandwich_mapping", model.mapping)
    logger.info(
        "Beam sandwich all-reduce boundaries: through "
        "sandwich_rmsnorm_with_allreduce "
        f"(x{len(boundaries)} per forward; fused where the MNNVL workspace serves "
        f"the call, at most {global_server_args_dict['comm_fusion_max_num_tokens']} "
        "tokens)"
    )
