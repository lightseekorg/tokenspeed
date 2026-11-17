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

"""Qwen4-Exp decode monokernel layers.

With ``--decode-monokernel``, the model builds its decoder layers from the
monokernel classes (``Qwen4ExpModel.decoder_layer_cls``): subclasses of the
regular layers with :class:`DeltaMonokernelLayer` or
:class:`QSAMonokernelLayer`. Such a layer runs a decode step of at most
``MAX_ROWS`` rows as one persistent launch from
``tokenspeed_kernel.ops.model.qwen4_exp.monokernel``, and every other
forward (prefill, larger steps) with its regular code. After loading, :func:`enable_decode_monokernels`
packs every layer and gives all of them one :class:`Qwen4ExpMonokernelState`.
Startup fails when a layer cannot be packed: the switch never serves a layer
without its kernel. ``tokenspeed.runtime.models.monokernel`` holds the part
that every model shares.

The decode step of a layer launch contains:

* both hyper-connection mixers,
* the Gated DeltaNet, or the QSA attention with its indexer,
* the output projection,
* the MoE.

The launch boundary is the layer's hyper-connection streams
``[rows, hc_count * hidden]`` with no pending residual. Every Qwen4-Exp layer
and the final mixer already accept this form, so a layer's launch and its
regular code (prefill, larger steps) read and write the same caches and
follow each other freely.

The launches tag their mailbox writes with the layer index and a device step
counter, which changes once per forward that launches, inside the CUDA graph:
the last layer's launch moves it when it also runs the final mixer, else
``MonokernelState.finish_step`` does.

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

import logging

import torch
from tokenspeed_kernel.ops.gemm import MXFP4_LINEAR_MFMA_LAYOUT
from tokenspeed_kernel.ops.model.qwen4_exp.monokernel import (
    FusedDecodeLayer,
    prepare_qwen4_exp_delta_layer,
    prepare_qwen4_exp_qsa_layer,
    qwen4_exp_delta_layer_decode,
    qwen4_exp_qsa_layer_decode,
)

from tokenspeed.runtime.execution.context import DraftNarrowing, ForwardContext
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    qwen4_exp_backend,
)
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_QSA_COMPRESSED_ROWS_PER_PAGE,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.dense.mxfp4 import Mxfp4LinearMethod
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.models.monokernel import (
    NO_KERNEL,
    MonokernelLayer,
    MonokernelState,
    NoMonokernel,
    unserved_error,
)

logger = logging.getLogger(__name__)


# Decode steps of up to this many token rows take the monokernels. The rows
# are the requests, or a speculative verify's requests times their tokens.
# Larger batches keep the regular path.
MAX_ROWS = 4


class Qwen4ExpMonokernelState(MonokernelState):
    """:class:`MonokernelState` with the constants that the Qwen4-Exp
    launches read and the draft's lengths of the current forward."""

    def __init__(self, device: torch.device, scratch_pairs: int) -> None:
        super().__init__(torch.zeros(scratch_pairs, dtype=torch.int64, device=device))
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
        # The verified rows' lengths of the current step, when it is a
        # draft's first step. start_step resets it.
        self._draft_lengths: torch.Tensor | None = None

    def start_step(self) -> None:
        super().start_step()
        self._draft_lengths = None

    def draft_lengths(
        self, seq_lens: torch.Tensor, narrowing: DraftNarrowing
    ) -> torch.Tensor:
        """The verified rows' lengths in a draft's first step.

        The step publishes the accepted lengths into ``seq_lens``, the buffer
        that the rows' lengths come from. So the step's first call copies
        ``seq_lens`` and then publishes through ``narrowing``, and every later
        layer reuses that copy.
        """
        if self._draft_lengths is None:
            self._draft_lengths = seq_lens.clone()
            narrowing.publish_accepted_prefix()
        return self._draft_lengths

    def verify_scratch_rows(self, bs: int, chain: int) -> torch.Tensor:
        """A verify's PLE scratch row of each token: ``r * (chain + 1) + 1 +
        t`` for token ``t`` of request ``r`` (int32, request major)."""
        return self._scratch_rows[(bs, chain)]

    def no_page(self, requests: int) -> torch.Tensor:
        """``requests`` entries of -1 (int32): requests that write no state
        page."""
        return self._no_page[:requests]


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


# Why a layer has no monokernel. The startup error groups layers by these.
NOT_SHARED_MOE = "not a sparse MoE with a shared expert"
ATTENTION_TP = "attention tensor parallelism"
EXPERT_PARALLEL = "expert parallelism"
NOT_BF16 = "an attention, GDN or PLE projection is not BF16"
NOT_MXFP4 = "the shared expert is not MXFP4 in one layout"
NOT_BF16_SHARED = "the shared expert is not BF16"
NOT_SUPPORTED_EXPERTS = (
    "the routed experts run neither the W4A4 MXFP4 nor the BF16 kernel"
)
NOT_SPLIT_IN_PROJ = "split in_proj_qkvz / in_proj_ba"
NOT_PLE_TABLE = "a PLE table that is offloaded, cast or sharded"
NO_INDEXER = "no QSA indexer"
NO_OUTPUT_GATE = "no attention output gate"
NOT_ROTARY = "a rotary cache that is not FP32 with 64 dims"
# The routed experts' kernels whose weights the monokernel reads in place,
# and the expert format (the kernel's ``layer.EXPERTS``) of each: the W4A4
# kernel's N16 banks, or the unquantized BF16 weights (the Quark
# checkpoint's MTP draft layer).
EXPERT_KERNELS = {
    "gluon_mxfp4_a4w4_moe_apply_gfx950": "mxfp4",
    "gluon_bf16_precomputed_moe_apply": "bf16",
}


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


def _moe_tensors(mlp) -> tuple[dict, bool]:
    """``(tensors, dense_tiled)`` of a layer's MoE block: the prepare ops' MoE
    tensors, and whether its MXFP4 dense weights are in the MFMA layout. The
    routed experts' kernel is one of :data:`EXPERT_KERNELS`.

    Raises:
        NoMonokernel: The monokernel cannot read the shared expert.
    """
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
            raise NoMonokernel(NOT_BF16_SHARED)
        tensors["w13"] = experts.w13_weight.data
        tensors["w2"] = experts.w2_weight.data
        tensors["shared_gu"], tensors["shared_dn"] = dense
        return tensors, False
    (gate_up, down), tiled = _shared_expert(mlp)
    tensors["banks"] = (
        experts.w13_weight.data,
        experts.w13_weight_scale.data,
        experts.w2_weight.data,
        experts.w2_weight_scale.data,
    )
    tensors["shared_gu"], tensors["shared_dn"] = gate_up, down
    return tensors, tiled


def _shared_expert(mlp) -> tuple[list, bool]:
    """``(weight, scale)`` of the shared expert's gate / up and down
    projections and whether they are in the MFMA layout.

    Raises:
        NoMonokernel: They are not both MXFP4 projections in one layout.
    """
    dense = [
        _mxfp4(p) for p in (mlp.shared_expert.gate_up_proj, mlp.shared_expert.down_proj)
    ]
    if any(d is None for d in dense) or len({d[2] for d in dense}) != 1:
        raise NoMonokernel(NOT_MXFP4)
    return [(d[0], d[1]) for d in dense], dense[0][2] == MXFP4_LINEAR_MFMA_LAYOUT


def forward_chain(rows: int, ctx: ForwardContext) -> int | None:
    """The tokens per request (``Plan.chain``) of a forward of ``rows`` token
    rows that the monokernels serve, else None (fixed per graph).

    The forward must be a decode step of at most ``MAX_ROWS`` token rows
    (padded graph rows included). It must have one token per request (a
    decode step or an MTP draft step), or ``k + 1``: a target verify (with
    Gated DeltaNet layers and ReplaySSM, whose payload the kernels fill; a
    view without them keeps the regular path) or the draft's first step of
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
    backend = ctx.attn_backend
    if not backend.has_linear_attn:
        return None
    return chain if backend.linear_attn_backend.replay_ssm else None


def _out_rows(rows: int, chain: int, ctx: ForwardContext) -> int:
    """The rows a launch outputs: one per request in the draft's catch-up
    step (TokenSpeed's draft narrowing), else every row."""
    return ctx.bs if chain > 1 and ctx.draft_narrowing is not None else rows


class _Qwen4ExpMonokernelLayer(MonokernelLayer):
    """A Qwen4-Exp decoder layer that runs a decode step of at most
    ``MAX_ROWS`` rows (:func:`forward_chain`) as one monokernel launch.

    The launch takes and returns the regular layer's boundary: the
    hyper-connection streams, with no pending residual.
    :func:`enable_decode_monokernels` packs the layer after loading.
    """

    # Whether this layer type's launch can also run the model's final mixer,
    # when the layer is the last one.
    FOLDS_FINAL_MIXER: bool

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Set after loading by enable_decode_monokernels: the packed layer,
        # and the next layer's late weights for the launch to warm (None for
        # the last layer).
        self.fused: FusedDecodeLayer | None = None
        self.warm: torch.Tensor | None = None

    def prepare_monokernel(self, final_mixer: torch.nn.Module | None) -> None:
        """Pack the layer for its monokernel into ``fused``.

        Args:
            final_mixer: The model's final hyper-connection mixer when the
                layer is the last one and ``FOLDS_FINAL_MIXER`` holds: the
                launch then also runs it. Else ``None``.

        Raises:
            NoMonokernel: The layer cannot be packed.
        """
        raise NotImplementedError

    def launch_chain(self, rows: int, ctx: ForwardContext) -> int | None:
        return forward_chain(rows, ctx)

    def prefetches_ple(self, rows: int, ctx: ForwardContext) -> bool:
        """Whether the model starts this layer's PLE lookup before the layer
        runs. A launch does its own lookup."""
        return self.ple is not None and self.launch_chain(rows, ctx) is None


class DeltaMonokernelLayer(_Qwen4ExpMonokernelLayer):
    """A Gated DeltaNet decoder layer with a monokernel (its PLE block
    included)."""

    FOLDS_FINAL_MIXER = False

    def prepare_monokernel(self, final_mixer: torch.nn.Module | None) -> None:
        """Pack a ``Qwen4ExpLinearDecoderLayer``. An eligible layer has:

        * the conditions of :func:`_common_reasons`,
        * one fused ``in_proj_qkvzba``,
        * BF16 GDN (and PLE) projections, as the Quark W4A4 checkpoint holds
          them, and a shared expert in the routed experts' format.
        """
        del final_mixer  # Always None: FOLDS_FINAL_MIXER is False.
        attn, mlp, ple = self.linear_attn, self.mlp, self.ple
        reasons = _common_reasons(self)
        if attn._split_in_proj:
            reasons.append(NOT_SPLIT_IN_PROJ)
        if reasons:
            raise NoMonokernel(*reasons)
        projections = [attn.in_proj_qkvzba, attn.out_proj]
        if ple is not None:
            projections.append(ple.kv_proj)
        dense = _bf16_all(projections)
        if dense is None:
            raise NoMonokernel(NOT_BF16)
        moe_tensors, tiled = _moe_tensors(mlp)
        qkvzba, out, *ple_kv = dense
        tensors = {
            "attn_mixer": _mixer(self.attn_hyper_connection),
            "qkvzba": qkvzba,
            "conv_w": attn.conv_weights,
            "a_log": attn.A_log.data,
            "dt_bias": attn.dt_bias.data,
            "norm_w": attn.norm.weight.data,
            "out": out,
            "mlp_mixer": _mixer(self.mlp_hyper_connection),
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
                raise NoMonokernel(NOT_PLE_TABLE)
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
        self.fused = prepare_qwen4_exp_delta_layer(
            max_rows=MAX_ROWS, tensors=tensors, dense_tiled=tiled
        )
        if self.fused is None:
            raise NoMonokernel(NO_KERNEL)

    def _decode(
        self,
        chain: int,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Run the Gated DeltaNet layer's launch. Returns the output streams.

        For the layer with a PLE block, the kernel also does the n-gram lookup
        from the token and the block's context page. The model skips the
        lookup prefetch for such a step. The launch may read ``warm`` into
        the memory-side cache.
        """
        del positions
        state = self.monokernel_state
        rows = hidden_states.shape[0]
        hidden_states = _streams(hidden_states, residual)
        gdn = ctx.attn_backend.linear_attn_backend
        state_in, state_out, conv_state, ssm_state = gdn.layer_state(self.layer_id)
        ple = None
        if self.ple is not None:
            verify_out = (
                None if chain == 1 else state.verify_scratch_rows(ctx.bs, chain)
            )
            ple = {
                "token": input_ids[:rows],
                **self.ple.fused_operands(ctx, verify_out),
            }
        if chain == 1:
            conv_out, conv_out_rows, replay = conv_state, state_out, None
        else:
            # A target verify: each token's state goes to the verify scratch
            # and the replay payload. The post-verify commit writes the pages.
            attn = self.linear_attn
            conv_out, conv_out_rows, replay = gdn.fused_verify_operands(
                self.layer_id, attn.A_log.data, attn.dt_bias.data
            )
            # The launch reads each request's input page; it writes no page.
            state_in, state_out = state_in[: ctx.bs], state.no_page(ctx.bs)
        out = hidden_states.new_empty(rows, self.hc_count * self.hidden_size)
        qwen4_exp_delta_layer_decode(
            self.fused,
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
            scratch=state.scratch,
            step=state.step,
            layer_index=self.layer_id,
            ple=ple,
            warm=self.warm,
        )
        return out, None


class QSAMonokernelLayer(_Qwen4ExpMonokernelLayer):
    """A QSA attention decoder layer with a monokernel. The last layer's
    launch also runs the model's final mixer."""

    FOLDS_FINAL_MIXER = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Whether the launch also runs the model's final mixer. Set by
        # prepare_monokernel.
        self.folds_final_mixer = False

    def prepare_monokernel(self, final_mixer: torch.nn.Module | None) -> None:
        """Pack a ``Qwen4ExpAttentionDecoderLayer``. An eligible layer has:

        * the conditions of :func:`_common_reasons`,
        * a QSA indexer,
        * the sigmoid output gate,
        * an FP32 rotary cache of 64 dims,
        * BF16 attention and indexer projections, as the Quark W4A4 checkpoint
          holds them, and a shared expert in the routed experts' format
          (MXFP4 for the target's layers, BF16 for the MTP draft layer).
        """
        mlp, indexer = self.mlp, self.indexer
        reasons = _common_reasons(self)
        if indexer is None:
            reasons.append(NO_INDEXER)
        if not self.attn_output_gate:
            reasons.append(NO_OUTPUT_GATE)
        cos_sin = self.rotary_emb.cos_sin_cache
        if cos_sin.dtype != torch.float32 or cos_sin.shape[-1] != 64:
            reasons.append(NOT_ROTARY)
        if reasons:
            raise NoMonokernel(*reasons)
        dense = _bf16_all((self.qkv_proj, indexer.index_qk_proj, self.o_proj))
        if dense is None:
            raise NoMonokernel(NOT_BF16)
        moe_tensors, tiled = _moe_tensors(mlp)
        qkv, index_qk, out = dense
        tensors = {
            "attn_mixer": _mixer(self.attn_hyper_connection),
            "qkv": qkv,
            "index_qk": index_qk,
            "q_norm_w": self.q_norm.weight.data,
            "k_norm_w": self.k_norm.weight.data,
            "iq_norm_w": indexer.q_layernorm.weight.data,
            "ik_norm_w": indexer.k_layernorm.weight.data,
            "out": out,
            "mlp_mixer": _mixer(self.mlp_hyper_connection),
            **moe_tensors,
        }
        if final_mixer is not None:
            tensors["final_mixer"] = _mixer(final_mixer)
        self.fused = prepare_qwen4_exp_qsa_layer(
            max_rows=MAX_ROWS, tensors=tensors, dense_tiled=tiled
        )
        if self.fused is None:
            raise NoMonokernel(NO_KERNEL)
        self.folds_final_mixer = final_mixer is not None

    def _decode(
        self,
        chain: int,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Run the QSA layer's launch. Returns the output streams.

        Reads the same per-step tables that the regular path reads. All of
        them refresh in place for CUDA graph replay. The tables are the
        indexer backend's sequence lengths and QSA page tables, and the
        full-attention page table. When the launch also runs the final mixer,
        its output becomes the state's ``lm_head_input``.
        """
        del input_ids
        state = self.monokernel_state
        rows = hidden_states.shape[0]
        hidden_states = _streams(hidden_states, residual)
        backend = qwen4_exp_backend(ctx.attn_backend).indexer_backend
        metadata = backend.metadata_for(ctx.forward_mode)
        full = backend.full_attn_backend.group_view(FULL_ATTENTION, ctx.bs)
        pool = ctx.token_to_kv_pool
        k_cache, v_cache = pool.get_kv_buffer(self.layer_id)
        raw_key, compressed, rope_position = self.indexer.cache_fields(pool)
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
                draft = self.indexer.draft_scratch(
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
                # lengths are a copy from before (one copy per forward), and
                # starts a new draft scratch round.
                staging = None
                seq_len = state.draft_lengths(seq_len, ctx.draft_narrowing)
                accepted = metadata.seq_lens[: ctx.bs]
                self.indexer.reset_draft_tags(
                    ctx.bs, torch.bfloat16, hidden_states.device
                )
            else:
                token_k, position_values, logical, recent = (
                    backend.verify_staging_buffers(self.layer_id, ctx.bs)
                )
                staging = (
                    token_k.reshape(rows, -1),
                    position_values.reshape(rows, -1),
                    logical.reshape(rows),
                    recent.reshape(rows),
                )
        out_rows = _out_rows(rows, chain, ctx)
        mixed_out = None
        if self.folds_final_mixer:
            mixed_out = hidden_states.new_empty(out_rows, self.hidden_size)
            state.lm_head_input = mixed_out
        out = hidden_states.new_empty(out_rows, self.hc_count * self.hidden_size)
        qwen4_exp_qsa_layer_decode(
            self.fused,
            hidden_states,
            out,
            chain=chain,
            cos_sin=self.rotary_emb.cos_sin_cache,
            positions=positions,
            seq_len=seq_len,
            full_table=tables[0],
            page_size=full.kernel_page_size,
            k_cache=k_cache,
            v_cache=v_cache,
            qsa_table=tables[1],
            recent_table=tables[2],
            compressed=_paged(compressed, QWEN4_EXP_QSA_COMPRESSED_ROWS_PER_PAGE),
            raw_key=_paged(raw_key, self.indexer.compress_ratio),
            rope_position=rope_position,
            staging=staging,
            draft=draft,
            accepted=accepted,
            scratch=state.scratch,
            step=state.step,
            layer_index=self.layer_id,
            mixed_out=mixed_out,
            # The final mixer's launch is the forward's last.
            advance_step=self.folds_final_mixer,
        )
        return out, None


def _streams(hidden_states: torch.Tensor, residual) -> torch.Tensor:
    """The layer's input hyper-connection streams, or the one row (the
    embedding) that stands for all of them. The kernel reads this row as
    streams that are all that row, without a copy.

    Every layer of a forward launches or none does, so a launch gets the
    previous launch's complete streams, never a pending residual.
    """
    if residual is not None:
        raise RuntimeError(
            "a monokernel launch got a pending residual: every layer of a "
            "forward must launch, or none"
        )
    return hidden_states.contiguous()


def _paged(field: torch.Tensor, rows: int) -> torch.Tensor:
    """``[pages, rows, 1, dim]`` view of a QSA key field, which the cache
    arena may present with the page axis folded into rows."""
    if field.dim() == 3:
        return field.view(-1, rows, *field.shape[1:])
    return field


def enable_decode_monokernels(model: torch.nn.Module) -> None:
    """Pack the monokernel layers of ``model`` after loading.

    ``model`` builds its layers from the monokernel classes with
    ``--decode-monokernel``. Then every layer is packed, and all of them and
    ``model.monokernel_state`` get one :class:`Qwen4ExpMonokernelState`. Without
    monokernel layers this does nothing.

    The last layer's launch also runs the final mixer when its type can
    (``FOLDS_FINAL_MIXER``). With attention tensor parallelism, the final
    mixer would need an all-gather of the streams first, but packing rejects
    attention tensor parallelism.

    Args:
        model: A Qwen4-Exp backbone (target or MTP draft).

    Raises:
        ValueError: A layer cannot be packed. The error names every such
            layer with its reasons.
    """
    layers = model.layers
    if not any(isinstance(layer, MonokernelLayer) for layer in layers):
        return
    last = len(layers) - 1
    unserved: dict[int, list[str]] = {}
    for index, layer in enumerate(layers):
        folds = index == last and layer.FOLDS_FINAL_MIXER
        try:
            layer.prepare_monokernel(model.hyper_connection_mixer if folds else None)
        except NoMonokernel as error:
            unserved[index] = error.reasons
    if unserved:
        raise ValueError(unserved_error(type(model).__name__, unserved))
    device = next(model.parameters()).device
    state = Qwen4ExpMonokernelState(
        device, max(layer.fused.scratch_pairs for layer in layers)
    )
    for index, layer in enumerate(layers):
        if index < last:
            layer.warm = layers[index + 1].fused.late_weights
        layer.monokernel_state = state
    model.monokernel_state = state
    logger.info(f"Decode monokernel: all {len(layers)} {type(model).__name__} layers")


__all__ = [
    "MAX_ROWS",
    "DeltaMonokernelLayer",
    "QSAMonokernelLayer",
    "Qwen4ExpMonokernelState",
    "enable_decode_monokernels",
]
