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

"""Qwen4-Exp decode megakernel seam.

With ``--decode-megakernel``, every eligible decoder layer runs its whole
decode step as one persistent launch from ``tokenspeed_kernel.ops.megakernel``.
The model touches this module in three places:

* after loading, :func:`enable_decode_megakernels` packs the layers into
  ``model.megakernels`` (:class:`Qwen4ExpMegakernels`);
* each forward asks :meth:`Qwen4ExpMegakernels.plan` once for a
  :class:`MegakernelStep`, or None for the regular path;
* the layer loop runs ``step.layer(i)`` in place of layer ``i`` where it is
  not None.

The decode step of a layer launch contains:

* both hyper-connection mixers,
* the Gated DeltaNet, or the QSA attention with its indexer,
* the output projection,
* the MoE.

The launch boundary is the layer's hyper-connection streams
``[rows, hc_count * hidden]`` with no pending residual. Every Qwen4-Exp layer
and the final mixer already accept this form, so fused and regular layers mix
freely.

The launches tag their mailbox writes with a device step counter that
:meth:`MegakernelSteps.advance` bumps once per forward, inside the CUDA
graph, and with the layer index.

A target's MTP verify (``k + 1`` tokens per request, request-major rows) runs
the same launches with ``chain = k + 1``:

1. The kernels read each request's committed state.
2. The kernels write each token's state where TokenSpeed's regular verify
   does. These places are the Gated DeltaNet and PLE verify scratch rows, the
   ReplaySSM payload and the QSA verify staging.
3. TokenSpeed's post-verify commit then writes the accepted state to the
   pages.

The MTP draft's first step (its catch-up over one request's verified rows)
runs as a chain too. As TokenSpeed's draft narrowing, every row writes its
K / V, but only the live row (the last accepted token's) runs the rest of the
layer, so the launch outputs one row.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

import torch
from tokenspeed_kernel.ops.gemm import MXFP4_LINEAR_MFMA_LAYOUT
from tokenspeed_kernel.ops.megakernel import (
    FusedDecodeLayer,
    prepare_qwen4_exp_delta_layer,
    prepare_qwen4_exp_qsa_layer,
    qwen4_exp_delta_layer_decode,
    qwen4_exp_qsa_layer_decode,
)

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    qwen4_exp_backend,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.dense.mxfp4 import Mxfp4LinearMethod
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)

# Decode steps of up to this many token rows take the megakernels. The rows
# are the requests, or a speculative verify's requests times their tokens.
# Larger batches keep the regular path.
MAX_ROWS = 4


class MegakernelSteps:
    """Model-wide megakernel state: the mailbox scratch, the step counter, and
    the operands that all layers of one forward share."""

    def __init__(self, device: torch.device, scratch_pairs: int) -> None:
        self.scratch = torch.zeros(scratch_pairs, dtype=torch.int64, device=device)
        # The launches' tag and a count of a launch's finished CTAs (0
        # between launches): a forward's last launch moves the tag on as it
        # ends (qwen4_exp_qsa_layer_decode(advance_step=True)).
        self.step = torch.zeros(2, dtype=torch.int32, device=device)
        # Constants the launches read, made here, outside any CUDA graph
        # capture: a capture records a kernel without running it, so a
        # constant made there holds garbage for every other graph. A launch
        # serves at most MAX_ROWS rows.
        self._no_page = torch.full((MAX_ROWS,), -1, dtype=torch.int32, device=device)
        # A verify's PLE scratch rows, r * (chain + 1) + 1 + t for token t
        # of request r, by (requests, chain).
        self._scratch_rows = {
            (bs, chain): (
                torch.arange(bs, dtype=torch.int32, device=device)[:, None]
                * (chain + 1)
                + 1
                + torch.arange(chain, dtype=torch.int32, device=device)[None, :]
            ).reshape(-1)
            for bs in range(1, MAX_ROWS + 1)
            for chain in range(2, MAX_ROWS // bs + 1)
        }

    def verify_scratch_rows(self, bs: int, chain: int) -> torch.Tensor:
        """A verify's PLE scratch row of each token: ``r * (chain + 1) + 1 +
        t`` for token ``t`` of request ``r`` (int32, request major)."""
        return self._scratch_rows[(bs, chain)]

    def no_page(self, requests: int) -> torch.Tensor:
        """``requests`` entries of -1 (int32): requests that write no state
        page."""
        return self._no_page[:requests]

    def advance(self) -> None:
        """Give the next forward's launches a new tag (graph safe), after a
        forward whose last launch did not."""
        self.step[:1].add_(1)


def _bf16(proj: torch.nn.Module) -> torch.Tensor | None:
    """The weight of an unquantized BF16 projection (the kernel reads it in
    place), else None."""
    if not isinstance(proj.quant_method, UnquantizedLinearMethod):
        return None
    weight = proj.weight.data
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        return None
    return weight


def _bf16_all(projections) -> list[torch.Tensor] | None:
    """The weight of each projection when all are BF16 (:func:`_bf16`)."""
    weights = [_bf16(p) for p in projections]
    return None if any(w is None for w in weights) else weights


def _mxfp4(proj: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, str] | None:
    """``(weight, scale, layout)`` of an MXFP4 projection (MXFP4 weights and
    activations), else None."""
    if not isinstance(proj.quant_method, Mxfp4LinearMethod):
        return None
    prepared = proj.mxfp4_linear_weight
    if prepared is None:
        return None
    return prepared.weight, prepared.weight_scale, prepared.layout


# Why a layer has no megakernel. The startup report groups layers by these.
NOT_SHARED_MOE = "not a sparse MoE with a shared expert"
ATTENTION_TP = "attention tensor parallelism"
EXPERT_PARALLEL = "expert parallelism"
NOT_BF16 = "an attention, GDN or PLE projection is not BF16"
NOT_MXFP4 = "the shared expert is not MXFP4 in one layout"
NOT_BF16_SHARED = "the shared expert is not BF16"
NOT_SUPPORTED_EXPERTS = (
    "the routed experts run neither the W4A4 MXFP4 nor the BF16 kernel"
)
# The routed experts' kernels whose weights the megakernel reads in place,
# and the expert format (the kernel's ``layer.EXPERTS``) of each: the W4A4
# kernel's N16 banks, or the unquantized BF16 weights (the Quark
# checkpoint's MTP draft layer).
EXPERT_KERNELS = {
    "gluon_mxfp4_a4w4_moe_apply_gfx950": "mxfp4",
    "gluon_bf16_precomputed_moe_apply": "bf16",
}
NO_KERNEL = "no megakernel for this platform"


def _common_reasons(layer: torch.nn.Module) -> list[str]:
    """Why a layer is not a sparse MoE with a shared expert, its routed
    experts on a kernel of :data:`EXPERT_KERNELS`, without attention or
    expert parallelism."""
    reasons = []
    if not (layer.is_moe and layer.mlp.shared_expert is not None):
        reasons.append(NOT_SHARED_MOE)
    elif layer.mlp.experts.plan["apply_kernel_name"] not in EXPERT_KERNELS:
        reasons.append(NOT_SUPPORTED_EXPERTS)
    if layer.mapping.attn.tp_size != 1:
        reasons.append(ATTENTION_TP)
    if layer.mapping.moe.world_size != 1:
        reasons.append(EXPERT_PARALLEL)
    return reasons


def _mixer(hc) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        hc.hc_norm.weight.data,
        hc.mix_inject_proj.weight.data,
        hc.input_mix_weight_up.weight.data,
    )


def _moe_tensors(mlp) -> tuple[dict, bool] | str:
    """``(tensors, dense_tiled)`` of a layer's MoE block (the prepare ops'
    MoE tensors, and whether its MXFP4 dense weights are in the MFMA
    layout), or the reason why the megakernel cannot read it. The routed
    experts' kernel is one of :data:`EXPERT_KERNELS`."""
    experts = mlp.experts
    fmt = EXPERT_KERNELS[experts.plan["apply_kernel_name"]]
    tensors = {
        "router_w": mlp.gate.weight.data,
        "shared_gate_w": mlp.shared_expert_gate.weight.data,
        "experts": fmt,
    }
    if fmt == "bf16":
        dense = _bf16_all((mlp.shared_expert.gate_up_proj, mlp.shared_expert.down_proj))
        if dense is None:
            return NOT_BF16_SHARED
        tensors["w13"] = experts.w13_weight.data
        tensors["w2"] = experts.w2_weight.data
        tensors["shared_gu"], tensors["shared_dn"] = dense
        return tensors, False
    shared = _shared_expert(mlp)
    if shared is None:
        return NOT_MXFP4
    (gate_up, down), tiled = shared
    tensors["banks"] = (
        experts.w13_weight.data,
        experts.w13_weight_scale.data,
        experts.w2_weight.data,
        experts.w2_weight_scale.data,
    )
    tensors["shared_gu"], tensors["shared_dn"] = gate_up, down
    return tensors, tiled


def _shared_expert(mlp) -> tuple[list, bool] | None:
    """``(weight, scale)`` of the shared expert's gate / up and down
    projections and whether they are in the MFMA layout. None unless both
    are MXFP4 projections in one layout."""
    dense = [
        _mxfp4(p) for p in (mlp.shared_expert.gate_up_proj, mlp.shared_expert.down_proj)
    ]
    if any(d is None for d in dense) or len({d[2] for d in dense}) != 1:
        return None
    return [(d[0], d[1]) for d in dense], dense[0][2] == MXFP4_LINEAR_MFMA_LAYOUT


def prepare_delta_layer(layer: torch.nn.Module) -> FusedDecodeLayer | list[str]:
    """Pack a ``Qwen4ExpLinearDecoderLayer`` for the megakernel, or say why
    not (a list of reasons).

    An eligible layer has:

    * the conditions of :func:`_common_reasons`,
    * one fused ``in_proj_qkvzba``,
    * BF16 GDN (and PLE) projections, as the Quark W4A4 checkpoint holds
      them, and a shared expert in the routed experts' format.
    """
    attn, mlp, ple = layer.linear_attn, layer.mlp, layer.ple
    reasons = _common_reasons(layer)
    if attn._split_in_proj:
        reasons.append("split in_proj_qkvz / in_proj_ba")
    if reasons:
        return reasons
    projections = [attn.in_proj_qkvzba, attn.out_proj]
    if ple is not None:
        projections.append(ple.kv_proj)
    dense = _bf16_all(projections)
    if dense is None:
        return [NOT_BF16]
    moe = _moe_tensors(mlp)
    if isinstance(moe, str):
        return [moe]
    (qkvzba, out, *ple_kv), (moe_tensors, tiled) = dense, moe
    tensors = {
        "attn_mixer": _mixer(layer.attn_hyper_connection),
        "qkvzba": qkvzba,
        "conv_w": attn.conv_weights,
        "a_log": attn.A_log.data,
        "dt_bias": attn.dt_bias.data,
        "norm_w": attn.norm.weight.data,
        "out": out,
        "mlp_mixer": _mixer(layer.mlp_hyper_connection),
        **moe_tensors,
    }
    if ple is not None:
        embedding = ple.ple_embedding
        lookup = embedding.lookup
        if (
            lookup.offload_embedding
            or lookup.embed_store_dtype is not None
            or lookup.ngram_embedding.tp_size != 1
        ):
            return ["a PLE table that is offloaded, cast or sharded"]
        tensors["ple"] = {
            "kv": ple_kv[0],
            "norm_key": ple.norm_key.gemma_weight,
            "norm_query": ple.norm_query.gemma_weight,
            "norm_conv": ple.norm_conv.gemma_weight,
            "conv_w": ple.conv1d.weight.data,
            "table": lookup.ngram_embedding.weight.data,
            "multipliers": embedding.layer_multipliers,
            "sizes": embedding.ngram_heads_vocab_sizes,
            "offsets": embedding.ngram_heads_offsets,
            "eos": embedding.eos_token_id,
        }
    fused = prepare_qwen4_exp_delta_layer(
        max_rows=MAX_ROWS, tensors=tensors, dense_tiled=tiled
    )
    return [NO_KERNEL] if fused is None else fused


def prepare_qsa_layer(
    layer: torch.nn.Module, final_mixer: torch.nn.Module | None
) -> FusedDecodeLayer | list[str]:
    """Pack a ``Qwen4ExpAttentionDecoderLayer`` for the megakernel, or say why
    not (a list of reasons).

    An eligible layer has:

    * the conditions of :func:`_common_reasons`,
    * a QSA indexer,
    * the sigmoid output gate,
    * an FP32 rotary cache of 64 dims,
    * BF16 attention and indexer projections, as the Quark W4A4 checkpoint
      holds them, and a shared expert in the routed experts' format (MXFP4
      for the target's layers, BF16 for the MTP draft layer).

    Args:
        layer: The decoder layer.
        final_mixer: The model's final hyper-connection mixer when ``layer``
            is the last layer: the launch then also runs it. Else ``None``.
    """
    mlp, indexer = layer.mlp, layer.indexer
    reasons = _common_reasons(layer)
    if indexer is None:
        reasons.append("no QSA indexer")
    if not layer.attn_output_gate:
        reasons.append("no attention output gate")
    cos_sin = layer.rotary_emb.cos_sin_cache
    if cos_sin.dtype != torch.float32 or cos_sin.shape[-1] != 64:
        reasons.append("a rotary cache that is not FP32 with 64 dims")
    if reasons:
        return reasons
    dense = _bf16_all((layer.qkv_proj, indexer.index_qk_proj, layer.o_proj))
    if dense is None:
        return [NOT_BF16]
    moe = _moe_tensors(mlp)
    if isinstance(moe, str):
        return [moe]
    (qkv, index_qk, out), (moe_tensors, tiled) = dense, moe
    tensors = {
        "attn_mixer": _mixer(layer.attn_hyper_connection),
        "qkv": qkv,
        "index_qk": index_qk,
        "q_norm_w": layer.q_norm.weight.data,
        "k_norm_w": layer.k_norm.weight.data,
        "iq_norm_w": indexer.q_layernorm.weight.data,
        "ik_norm_w": indexer.k_layernorm.weight.data,
        "out": out,
        "mlp_mixer": _mixer(layer.mlp_hyper_connection),
        **moe_tensors,
    }
    if final_mixer is not None:
        tensors["final_mixer"] = _mixer(final_mixer)
    fused = prepare_qwen4_exp_qsa_layer(
        max_rows=MAX_ROWS, tensors=tensors, dense_tiled=tiled
    )
    return [NO_KERNEL] if fused is None else fused


def forward_chain(rows: int, ctx: ForwardContext) -> int | None:
    """The tokens per request (``Plan.chain``) of a forward of ``rows`` token
    rows that the megakernels serve, else None (fixed per graph).

    The forward must be a decode step of at most ``MAX_ROWS`` token rows
    (padded graph rows included). It must have one token per request (a
    decode step or an MTP draft step), or ``k + 1``: a target verify (with
    ReplaySSM, whose payload the kernels fill) or the draft's first step of
    one request, which catches up on the verified rows (its launch narrows to
    the request's live row).
    """
    if not (
        ctx.forward_mode.is_decode()
        and 1 <= ctx.bs
        and rows <= MAX_ROWS
        and rows % ctx.bs == 0
    ):
        return None
    chain = rows // ctx.bs
    if chain == 1:
        return chain
    if ctx.draft_narrowing is not None:
        return chain if ctx.bs == 1 else None
    return chain if ctx.attn_backend.linear_attn_backend.replay_ssm else None


def _out_rows(rows: int, chain: int, ctx: ForwardContext) -> int:
    """The rows a launch outputs: one per request in the draft's catch-up
    step (TokenSpeed's draft narrowing), else every row."""
    return ctx.bs if chain > 1 and ctx.draft_narrowing is not None else rows


def unserved_report(unserved: dict[int, list[str]]) -> str:
    """``unserved`` (layer index -> reasons) grouped by reason, for a log or an
    error: ``"reason: layers 0, 2; other reason: layer 5"``."""
    layers_by_reason: dict[str, list[int]] = {}
    for index, reasons in sorted(unserved.items()):
        for reason in reasons:
            layers_by_reason.setdefault(reason, []).append(index)
    return "; ".join(
        f"{reason}: layer{'s' if len(layers) > 1 else ''} "
        f"{', '.join(str(i) for i in layers)}"
        for reason, layers in layers_by_reason.items()
    )


class Qwen4ExpMegakernels:
    """The decode megakernels of one Qwen4-Exp backbone (the target's or the
    MTP draft's), packed after loading.

    Attributes:
        layers: The backbone's decoder layers.
        delta: The indices of the Gated DeltaNet layers. The others are QSA
            layers.
        fused: The packed launch of each served layer, by layer index.
        unserved: The reasons of each layer without a launch, by layer index.
            Those layers keep the regular path.
        final_mixer_layer: The index of the last layer when its launch also
            runs the model's final mixer, else None.
        steps: The mailbox scratch and the step counter of all launches.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.layers = model.layers
        last = len(self.layers) - 1
        # A last QSA layer's launch also runs the final mixer, unless the
        # streams need an all-gather first.
        fold_final = (
            last >= 0
            and not isinstance(self.layers[last], model.LINEAR_LAYER_CLS)
            and not self.layers[last].comm_manager.needs_final_all_gather()
        )
        self.fused: dict[int, FusedDecodeLayer] = {}
        self.unserved: dict[int, list[str]] = {}
        self.delta = {
            index
            for index, layer in enumerate(self.layers)
            if isinstance(layer, model.LINEAR_LAYER_CLS)
        }
        for index, layer in enumerate(self.layers):
            if index in self.delta:
                prepared = prepare_delta_layer(layer)
            else:
                final_mixer = model.hyper_connection_mixer
                prepared = prepare_qsa_layer(
                    layer, final_mixer if fold_final and index == last else None
                )
            if isinstance(prepared, FusedDecodeLayer):
                self.fused[index] = prepared
            else:
                self.unserved[index] = prepared
        self.final_mixer_layer = last if fold_final and last in self.fused else None
        self.steps: MegakernelSteps | None = None
        if self.fused:
            device = next(model.parameters()).device
            scratch_pairs = max(f.scratch_pairs for f in self.fused.values())
            self.steps = MegakernelSteps(device, scratch_pairs)

    def plan(self, rows: int, ctx: ForwardContext) -> MegakernelStep | None:
        """This forward's step of ``rows`` token rows, or None when the
        megakernels do not serve it (the regular path)."""
        chain = forward_chain(rows, ctx)
        return None if chain is None else MegakernelStep(self, chain)


class MegakernelStep:
    """One forward that the megakernels serve.

    Attributes:
        lm_head_input: The final mixer's output, set by the last layer's
            launch when it also runs the final mixer. The model then returns
            it and skips its own final mixer.
    """

    def __init__(self, megakernels: Qwen4ExpMegakernels, chain: int) -> None:
        self._megakernels = megakernels
        self._chain = chain
        self.lm_head_input: torch.Tensor | None = None

    def layer(self, index: int) -> Callable | None:
        """The launch that replaces decoder layer ``index``, with the layer's
        forward arguments, or None when the layer keeps the regular path."""
        if index not in self._megakernels.fused:
            return None
        return functools.partial(self._run, index)

    def finish(self) -> None:
        """End the forward: give the next forward's launches a new tag
        (graph safe), unless the last launch did."""
        if self.lm_head_input is None:
            self._megakernels.steps.advance()

    def _run(
        self,
        index: int,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Run layer ``index``'s launch. Returns ``(streams, None)``: the
        layer's output streams with no pending residual."""
        megakernels = self._megakernels
        layer, fused, steps = (
            megakernels.layers[index],
            megakernels.fused[index],
            megakernels.steps,
        )
        if index in megakernels.delta:
            out = _delta_decode(
                layer,
                fused,
                steps,
                self._chain,
                hidden_states,
                residual,
                input_ids,
                ctx,
            )
            return out, None
        if index == megakernels.final_mixer_layer:
            self.lm_head_input = hidden_states.new_empty(
                _out_rows(hidden_states.shape[0], self._chain, ctx), layer.hidden_size
            )
        out = _qsa_decode(
            layer,
            fused,
            steps,
            self._chain,
            positions,
            hidden_states,
            residual,
            ctx,
            self.lm_head_input,
        )
        return out, None


def _streams(
    layer: torch.nn.Module, hidden_states: torch.Tensor, residual
) -> torch.Tensor:
    """The layer's input hyper-connection streams, or the one row (the
    embedding) that stands for all of them. The kernel reads this row as
    streams that are all that row, without a copy."""
    if residual is not None:
        hidden_states = layer.attn_hyper_connection.combine(hidden_states, residual)
    return hidden_states.contiguous()


def _paged(field: torch.Tensor, rows: int) -> torch.Tensor:
    """``[pages, rows, 1, dim]`` view of a QSA key field, which the cache
    arena may present with the page axis folded into rows."""
    if field.dim() == 3:
        return field.view(-1, rows, *field.shape[1:])
    return field


def _delta_decode(
    layer: torch.nn.Module,
    fused: FusedDecodeLayer,
    steps: MegakernelSteps,
    chain: int,
    hidden_states: torch.Tensor,
    residual,
    input_ids: torch.Tensor,
    ctx: ForwardContext,
) -> torch.Tensor:
    """Run one Gated DeltaNet layer launch. Returns the output streams.

    For the layer with a PLE block, the kernel also does the n-gram lookup
    from the token and the block's context page. The model skips the lookup
    prefetch for such a step.
    """
    rows = hidden_states.shape[0]
    hidden_states = _streams(layer, hidden_states, residual)
    gdn = ctx.attn_backend.linear_attn_backend
    state_in, state_out, conv_state, ssm_state = gdn.layer_state(layer.layer_id)
    ple = None
    if layer.ple is not None:
        verify_out = None if chain == 1 else steps.verify_scratch_rows(ctx.bs, chain)
        ple = {
            "token": input_ids[:rows],
            **layer.ple.fused_operands(ctx, verify_out),
        }
    if chain == 1:
        conv_out, conv_out_rows, replay = conv_state, state_out, None
    else:
        # A target verify: each token's state goes to the verify scratch and
        # the replay payload. The post-verify commit writes the pages.
        attn = layer.linear_attn
        conv_out, conv_out_rows, replay = gdn.fused_verify_operands(
            layer.layer_id, attn.A_log.data, attn.dt_bias.data
        )
        # The launch reads each request's input page; it writes no page.
        state_in, state_out = state_in[: ctx.bs], steps.no_page(ctx.bs)
    out = hidden_states.new_empty(rows, layer.hc_count * layer.hidden_size)
    qwen4_exp_delta_layer_decode(
        fused,
        hidden_states,
        out,
        chain=chain,
        conv_state=conv_state,
        ssm_state=ssm_state,
        state_in=state_in,
        state_out=state_out,
        conv_out=conv_out,
        conv_out_rows=conv_out_rows,
        replay=replay,
        scratch=steps.scratch,
        step=steps.step,
        layer_index=layer.layer_id,
        ple=ple,
    )
    return out


def _qsa_decode(
    layer: torch.nn.Module,
    fused: FusedDecodeLayer,
    steps: MegakernelSteps,
    chain: int,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual,
    ctx: ForwardContext,
    mixed_out: torch.Tensor | None,
) -> torch.Tensor:
    """Run one QSA layer launch. Returns the output streams.

    Reads the same per-step tables that the regular path reads. All of them
    refresh in place for CUDA graph replay. The tables are the indexer
    backend's sequence lengths and QSA page tables, and the full-attention
    page table. With ``mixed_out``, the launch also runs the model's final
    mixer into it.
    """
    rows = hidden_states.shape[0]
    hidden_states = _streams(layer, hidden_states, residual)
    backend = qwen4_exp_backend(ctx.attn_backend).indexer_backend
    metadata = backend.metadata_for(ctx.forward_mode)
    full = backend.full_attn_backend.group_view(FULL_ATTENTION, ctx.bs)
    pool = ctx.token_to_kv_pool
    k_cache, v_cache = pool.get_kv_buffer(layer.layer_id)
    raw_key, compressed, rope_position = layer.indexer.cache_fields(pool)
    if positions.dim() == 1:
        # Plain RoPE positions: all three MRoPE axes read the same row (the
        # cast copies only positions that are not int64).
        positions = positions.to(torch.int64).unsqueeze(0).expand(3, -1)
    tables = (
        full.page_table[: ctx.bs],
        metadata.qsa_block_table[: ctx.bs],
        metadata.recent_block_table[: ctx.bs],
    )
    draft = None
    accepted = None
    if chain == 1:
        seq_len, staging = metadata.seq_lens[: ctx.bs], None
        if backend.is_draft:
            # An MTP draft step: the index keys are speculative. They go to
            # the draft scratch, as on the regular path, not the recent ring.
            draft = layer.indexer.draft_scratch(
                ctx.bs, torch.bfloat16, hidden_states.device
            )
    else:
        # A verify: the launch takes each request's length and tables; token
        # t of request r sits at seq_lens[r] - chain + t (TokenSpeed's verify
        # layout). MRoPE positions can differ from this layout. Each row of a
        # target verify stages its index key for the post-verify commit
        # instead of writing the recent ring.
        seq_len = metadata.seq_lens[: ctx.bs]
        if backend.is_draft:
            # The draft's first step: the verified rows as a chain. Every row
            # writes its K / V. Rows past the accepted length write no index
            # key (TokenSpeed's draft write mask). Only the live row runs the
            # rest of the layer (the draft narrowing). The step publishes the
            # accepted lengths as the draft's decode lengths, so the rows'
            # lengths are a copy from before, and starts a new draft scratch
            # round.
            staging = None
            seq_len = seq_len.clone()
            ctx.draft_narrowing.publish_accepted_prefix()
            accepted = metadata.seq_lens[: ctx.bs]
            layer.indexer.reset_draft_tags(ctx.bs, torch.bfloat16, hidden_states.device)
        else:
            token_k, position_values, logical, recent = backend.verify_staging_buffers(
                layer.layer_id, ctx.bs
            )
            staging = (
                token_k.reshape(rows, -1),
                position_values.reshape(rows, -1),
                logical.reshape(rows),
                recent.reshape(rows),
            )
    out = hidden_states.new_empty(
        _out_rows(rows, chain, ctx), layer.hc_count * layer.hidden_size
    )
    qwen4_exp_qsa_layer_decode(
        fused,
        hidden_states,
        out,
        chain=chain,
        cos_sin=layer.rotary_emb.cos_sin_cache,
        positions=positions,
        seq_len=seq_len,
        full_table=tables[0],
        page_size=full.kernel_page_size,
        k_cache=k_cache,
        v_cache=v_cache,
        qsa_table=tables[1],
        recent_table=tables[2],
        compressed=_paged(compressed, 64),
        raw_key=_paged(raw_key, 4),
        rope_position=rope_position,
        staging=staging,
        draft=draft,
        accepted=accepted,
        scratch=steps.scratch,
        step=steps.step,
        layer_index=layer.layer_id,
        mixed_out=mixed_out,
        # The final mixer's launch is the forward's last.
        advance_step=mixed_out is not None,
    )
    return out


def coverage(
    name: str, layers: int, unserved: dict[int, list[str]], *, required: bool
) -> str | None:
    """The startup report of a backbone's megakernel coverage.

    Args:
        name: The backbone's class name.
        layers: Its number of decoder layers.
        unserved: The reasons of each layer without a megakernel.
        required: Whether the backbone must have a megakernel layer.

    Returns:
        A warning that names the layers on the regular path and why, None when
        every layer has a megakernel.

    Raises:
        ValueError: ``required`` and no layer has a megakernel.
    """
    if not unserved:
        return None
    report = unserved_report(unserved)
    if len(unserved) == layers and required:
        raise ValueError(
            f"--decode-megakernel: no layer of {name} has a megakernel "
            f"({report}). Serve without --decode-megakernel."
        )
    served = layers - len(unserved)
    return (
        f"Decode megakernel: {served} of {layers} {name} layers; the rest keep "
        f"the regular path ({report})"
    )


def enable_decode_megakernels(model: torch.nn.Module, *, required: bool) -> None:
    """Pack ``model``'s decode megakernels into ``model.megakernels`` when
    ``--decode-megakernel`` is set, and report the layers that keep the
    regular path, with the reasons.

    Args:
        model: A Qwen4-Exp backbone (target or MTP draft).
        required: Whether the backbone must have a megakernel layer. The
            target's must: the switch then fails startup instead of serving
            without megakernels. The draft's need not: without one, the draft
            steps keep the regular path.

    Raises:
        ValueError: ``required`` and no layer has a megakernel.
    """
    if not global_server_args_dict["decode_megakernel"]:
        return
    megakernels = Qwen4ExpMegakernels(model)
    name = type(model).__name__
    layers = len(megakernels.layers)
    warning = coverage(name, layers, megakernels.unserved, required=required)
    if warning is not None:
        logger.warning(warning)
    if megakernels.fused:
        model.megakernels = megakernels
        logger.info(
            f"Decode megakernel: {len(megakernels.fused)} of {layers} {name} layers"
        )


__all__ = [
    "MAX_ROWS",
    "MegakernelStep",
    "MegakernelSteps",
    "Qwen4ExpMegakernels",
    "coverage",
    "enable_decode_megakernels",
    "forward_chain",
    "prepare_delta_layer",
    "prepare_qsa_layer",
    "unserved_report",
]
