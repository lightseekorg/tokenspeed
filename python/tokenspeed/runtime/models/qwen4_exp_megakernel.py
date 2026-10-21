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

With ``--qwen4-exp-decode-megakernel``, every eligible decoder layer runs
its whole decode step as one persistent launch from
``tokenspeed_kernel.ops.megakernel``. The decode step contains:

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
"""

from __future__ import annotations

import logging

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

logger = logging.getLogger(__name__)

# Decode steps of up to this many token rows take the megakernels. The rows
# are the requests, or a speculative verify's requests times their tokens.
# Larger batches keep the regular path.
MAX_ROWS = 4


class MegakernelSteps:
    """Model-wide megakernel state: the mailbox scratch and the step counter."""

    def __init__(self, device: torch.device, scratch_pairs: int) -> None:
        self.scratch = torch.zeros(scratch_pairs, dtype=torch.int64, device=device)
        self.step = torch.zeros(1, dtype=torch.int32, device=device)

    def advance(self) -> None:
        """Give the next forward's launches a new tag (graph safe)."""
        self.step.add_(1)


def _dense(proj: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, str] | None:
    """``(weight, scale, layout)`` of a W4A8 MX projection, else None."""
    if not isinstance(proj.quant_method, Mxfp4LinearMethod):
        return None
    prepared = proj.mxfp4_linear_weight
    if prepared is None or prepared.activation_format != "mxfp8":
        return None
    return prepared.weight, prepared.weight_scale, prepared.layout


def _common_eligible(layer: torch.nn.Module) -> bool:
    """A sparse MoE with a shared expert, no attention or expert
    parallelism."""
    mlp = layer.mlp
    return (
        layer.is_moe
        and mlp.shared_expert is not None
        and layer.mapping.attn.tp_size == 1
        and layer.mapping.moe.world_size == 1
    )


def _mixer(hc) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        hc.hc_norm.weight.data,
        hc.mix_inject_proj.weight.data,
        hc.input_mix_weight_up.weight.data,
    )


def _moe_tensors(mlp, shared_gate_up, shared_down) -> dict:
    experts = mlp.experts
    return {
        "router_w": mlp.gate.weight.data,
        "shared_gate_w": mlp.shared_expert_gate.weight.data,
        "banks": (
            experts.w13_weight.data,
            experts.w13_weight_scale.data,
            experts.w2_weight.data,
            experts.w2_weight_scale.data,
        ),
        "shared_gu": shared_gate_up,
        "shared_dn": shared_down,
    }


def _dense_all(projections) -> tuple[list, bool] | None:
    """``(weight, scale)`` of each W4A8 MX projection and whether they are in
    the MFMA layout. None unless all are W4A8 MX projections in one layout."""
    dense = [_dense(p) for p in projections]
    if any(d is None for d in dense) or len({d[2] for d in dense}) != 1:
        return None
    return [(d[0], d[1]) for d in dense], dense[0][2] == MXFP4_LINEAR_MFMA_LAYOUT


def prepare_delta_layer(layer: torch.nn.Module) -> FusedDecodeLayer | None:
    """Pack a ``Qwen4ExpLinearDecoderLayer`` for the megakernel, if eligible.

    An eligible layer has:

    * the conditions of :func:`_common_eligible`,
    * split ``in_proj_qkvz`` / ``in_proj_ba``,
    * every dense projection served as MXFP4 weights with MXFP8 activations.
    """
    attn, mlp, ple = layer.linear_attn, layer.mlp, layer.ple
    if not _common_eligible(layer) or not attn._split_in_proj:
        return None
    projections = [
        attn.in_proj_qkvz,
        attn.out_proj,
        mlp.shared_expert.gate_up_proj,
        mlp.shared_expert.down_proj,
    ]
    if ple is not None:
        projections.append(ple.kv_proj)
    dense = _dense_all(projections)
    if dense is None:
        return None
    (qkvz, out, gate_up, down, *ple_kv), tiled = dense
    tensors = {
        "attn_mixer": _mixer(layer.attn_hyper_connection),
        "qkvz": qkvz,
        "conv_w": attn.conv_weights,
        "ba_w": attn.in_proj_ba.weight.data,
        "a_log": attn.A_log.data,
        "dt_bias": attn.dt_bias.data,
        "norm_w": attn.norm.weight.data,
        "out": out,
        "mlp_mixer": _mixer(layer.mlp_hyper_connection),
        **_moe_tensors(mlp, gate_up, down),
    }
    if ple is not None:
        embedding = ple.ple_embedding
        lookup = embedding.lookup
        if (
            lookup.offload_embedding
            or lookup.embed_store_dtype is not None
            or lookup.ngram_embedding.tp_size != 1
        ):
            return None
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
    return prepare_qwen4_exp_delta_layer(
        max_rows=MAX_ROWS, tensors=tensors, dense_tiled=tiled
    )


def prepare_qsa_layer(
    layer: torch.nn.Module, final_mixer: torch.nn.Module | None
) -> FusedDecodeLayer | None:
    """Pack a ``Qwen4ExpAttentionDecoderLayer`` for the megakernel, if eligible.

    An eligible layer has:

    * the conditions of :func:`_common_eligible`,
    * a QSA indexer,
    * the sigmoid output gate,
    * an FP32 rotary cache of 64 dims,
    * every dense projection (the attention's and the indexer's) served as
      MXFP4 weights with MXFP8 activations.

    Args:
        layer: The decoder layer.
        final_mixer: The model's final hyper-connection mixer when ``layer``
            is the last layer: the launch then also runs it (see
            :func:`forward_final_qsa_layer`). Else ``None``.
    """
    mlp, indexer = layer.mlp, layer.indexer
    if not _common_eligible(layer) or indexer is None or not layer.attn_output_gate:
        return None
    cos_sin = layer.rotary_emb.cos_sin_cache
    if cos_sin.dtype != torch.float32 or cos_sin.shape[-1] != 64:
        return None
    dense = _dense_all(
        (
            layer.qkv_proj,
            indexer.index_qk_proj,
            layer.o_proj,
            mlp.shared_expert.gate_up_proj,
            mlp.shared_expert.down_proj,
        )
    )
    if dense is None:
        return None
    (qkv, index_qk, out, gate_up, down), tiled = dense
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
        **_moe_tensors(mlp, gate_up, down),
    }
    if final_mixer is not None:
        tensors["final_mixer"] = _mixer(final_mixer)
    return prepare_qwen4_exp_qsa_layer(
        max_rows=MAX_ROWS, tensors=tensors, dense_tiled=tiled
    )


def runs_fused(hidden_states: torch.Tensor, ctx: ForwardContext) -> bool:
    """Whether this forward can take a megakernel layer (fixed per graph).

    The forward must be a decode step of at most ``MAX_ROWS`` token rows
    (padded graph rows included). It must have one token per request, or a
    target verify's ``k + 1`` (with ReplaySSM, whose payload the kernels
    fill).
    """
    rows = hidden_states.shape[0]
    if not (
        ctx.forward_mode.is_decode()
        and 1 <= ctx.bs
        and rows <= MAX_ROWS
        and rows % ctx.bs == 0
    ):
        return False
    return rows == ctx.bs or ctx.attn_backend.linear_attn_backend.replay_ssm


def _chain(hidden_states: torch.Tensor, ctx: ForwardContext) -> int:
    """Tokens per request of this forward's rows (``Plan.chain``)."""
    return hidden_states.shape[0] // ctx.bs


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


def forward_delta_layer(
    layer: torch.nn.Module,
    fused: FusedDecodeLayer,
    steps: MegakernelSteps,
    hidden_states: torch.Tensor,
    residual,
    input_ids: torch.Tensor,
    ctx: ForwardContext,
) -> tuple[torch.Tensor, None]:
    """Run a packed Gated DeltaNet layer. Returns ``(streams, None)``.

    For the layer with a PLE block, the kernel also does the n-gram lookup
    from the token and the block's context page. The model skips the lookup
    prefetch for such a step.
    """
    chain = _chain(hidden_states, ctx)
    rows = hidden_states.shape[0]
    hidden_states = _streams(layer, hidden_states, residual)
    gdn = ctx.attn_backend.linear_attn_backend
    state_in, state_out, conv_state, ssm_state = gdn.layer_state(layer.layer_id)
    ple = None
    if layer.ple is not None:
        ple = {"token": input_ids[:rows], **layer.ple.fused_operands(ctx)}
    if chain == 1:
        conv_out, conv_out_rows, replay = conv_state, state_out, None
    else:
        # A target verify: each token's state goes to the verify scratch and
        # the replay payload. The post-verify commit writes the pages.
        attn = layer.linear_attn
        conv_out, conv_out_rows, replay = gdn.fused_verify_operands(
            layer.layer_id, attn.A_log.data, attn.dt_bias.data
        )
        state_in = state_in[: ctx.bs].repeat_interleave(chain)
        state_out = torch.full_like(state_in, -1)
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
    return out, None


def forward_qsa_layer(
    layer: torch.nn.Module,
    fused: FusedDecodeLayer,
    steps: MegakernelSteps,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual,
    ctx: ForwardContext,
) -> tuple[torch.Tensor, None]:
    """Run a packed QSA attention layer. Returns ``(streams, None)``.

    Reads the same per-step tables that the regular path reads. All of them
    refresh in place for CUDA graph replay. The tables are the indexer
    backend's sequence lengths and QSA page tables, and the full-attention
    page table.
    """
    out = _qsa_decode(
        layer, fused, steps, positions, hidden_states, residual, ctx, None
    )
    return out, None


def forward_final_qsa_layer(
    layer: torch.nn.Module,
    fused: FusedDecodeLayer,
    steps: MegakernelSteps,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual,
    ctx: ForwardContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the last layer packed with the model's final mixer (see
    :func:`prepare_qsa_layer`).

    Returns:
        ``(mixed, streams)``: the final mixer's output (the LM head's input)
        and the layer's output streams.
    """
    rows = hidden_states.shape[0]
    mixed = torch.empty(
        rows, layer.hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )
    out = _qsa_decode(
        layer, fused, steps, positions, hidden_states, residual, ctx, mixed
    )
    return mixed, out


def _qsa_decode(
    layer: torch.nn.Module,
    fused: FusedDecodeLayer,
    steps: MegakernelSteps,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual,
    ctx: ForwardContext,
    mixed_out: torch.Tensor | None,
) -> torch.Tensor:
    """Run one QSA layer launch. Returns the output streams."""
    chain = _chain(hidden_states, ctx)
    rows = hidden_states.shape[0]
    hidden_states = _streams(layer, hidden_states, residual)
    backend = qwen4_exp_backend(ctx.attn_backend).indexer_backend
    metadata = backend.metadata_for(ctx.forward_mode)
    full = backend.full_attn_backend.group_view(FULL_ATTENTION, ctx.bs)
    pool = ctx.token_to_kv_pool
    k_cache, v_cache = pool.get_kv_buffer(layer.layer_id)
    raw_key, compressed, rope_position = layer.indexer.cache_fields(pool)
    if positions.dim() == 1:
        # Plain RoPE positions: all three MRoPE axes read the same row.
        positions = positions.unsqueeze(0).expand(3, -1)
    tables = (
        full.page_table[: ctx.bs],
        metadata.qsa_block_table[: ctx.bs],
        metadata.recent_block_table[: ctx.bs],
    )
    if chain == 1:
        seq_len, staging = metadata.seq_lens[: ctx.bs], None
    else:
        # A target verify: each token row has its own length and its
        # request's tables. Each row stages its index key for the post-verify
        # commit instead of writing the recent ring.
        # Token t of request r sits at seq_lens[r] - chain + t (TokenSpeed's
        # verify layout). MRoPE positions can differ from this layout.
        token = torch.arange(
            1 - chain, 1, dtype=torch.int32, device=hidden_states.device
        )
        seq_len = (metadata.seq_lens[: ctx.bs].int()[:, None] + token[None, :]).reshape(
            -1
        )
        tables = tuple(t.repeat_interleave(chain, dim=0) for t in tables)
        token_k, position_values, logical, recent = backend.verify_staging_buffers(
            layer.layer_id, ctx.bs
        )
        staging = (
            token_k.reshape(rows, -1),
            position_values.reshape(rows, -1),
            logical.reshape(rows),
            recent.reshape(rows),
        )
    out = hidden_states.new_empty(rows, layer.hc_count * layer.hidden_size)
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
        scratch=steps.scratch,
        step=steps.step,
        layer_index=layer.layer_id,
        mixed_out=mixed_out,
    )
    return out


__all__ = [
    "MAX_ROWS",
    "MegakernelSteps",
    "forward_delta_layer",
    "forward_qsa_layer",
    "prepare_delta_layer",
    "prepare_qsa_layer",
    "runs_fused",
]
